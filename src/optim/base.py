import copy
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import wandb
import yaml
from tqdm import tqdm

from logger.logger import DynamicsLogger
from optim.weight_averaging import (
    ExponentialWeightAverager,
    WeightAverager,
    eval_ewa,
    eval_wa,
)

from models.compress import (
    grad_effective_ranks,
    model_effective_ranks,
    optimizer_state_effective_ranks,
)

from .utils import (
    eval, get_batch, get_parameter_norms, load_checkpoint,
    load_worker_state, log_prodigy_lr, save_checkpoint,
    save_worker_state, visualize_routing,
)


def _append_local_metric(exp_dir: Path, payload: dict[str, object]) -> None:
    with (exp_dir / "metrics.jsonl").open("a") as metric_file:
        metric_file.write(json.dumps(payload, sort_keys=True) + "\n")


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
    downstream_evaluator=None,
    lm_evaluator=None,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        load_worker_state(ckpt_dir)
    elif cfg.init_from:
        # Fine-tuning / optimizer-change start: load only the model weights and
        # begin a fresh run (new optimizer & scheduler, curr_iter=0). Optimizer
        # and worker/dataloader state are intentionally NOT restored.
        print(f"\nInitializing Model Weights From {cfg.init_from}")
        ckpt = torch.load(
            Path(cfg.init_from) / "main.pt", map_location=cfg.device
        )
        model.load_state_dict(ckpt["model"])
        curr_iter = 0
    else:
        curr_iter = 0

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )
    if cfg.exponential_weight_average:
        ewa = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ewa_interval,
            decay=cfg.ewa_decay,
            warmup=cfg.warmup_steps if cfg.ewa_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": [], "downstream": [], "aux_lm": []}
    grad_norms = []
    _pending_grad_er: dict[str, float] = {}
    model.train()

    pbar = tqdm(total=cfg.iterations, initial=curr_iter, desc="Training", dynamic_ncols=True)
    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_and_log(
                tokens,
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                exp_dir,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

            if cfg.exponential_weight_average:
                eval_ewa(
                    curr_iter,
                    not_compiled_model,
                    ewa,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if downstream_evaluator is not None and downstream_evaluator.should_run(curr_iter):
            downstream_logs = downstream_evaluator.evaluate(
                curr_iter, model, type_ctx, distributed_backend
            )
            if downstream_logs is not None:
                stats["downstream"].append(downstream_logs)

        if lm_evaluator is not None and lm_evaluator.should_run(curr_iter):
            lm_logs = lm_evaluator.evaluate(curr_iter, model, type_ctx, distributed_backend)
            if lm_logs is not None:
                stats["aux_lm"].append(lm_logs)

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y, moe=cfg.moe)

            loss = outputs["loss"] / cfg.acc_steps
            loss.backward()
            substep += 1

        if cfg.grad_clip != 0.0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), cfg.grad_clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
            grad_norms.append(grad_norm)

        if getattr(cfg, "wd_schedule", "none") != "none" and cfg.wd_final is not None:
            t = curr_iter / cfg.iterations
            if cfg.wd_schedule == "linear":
                current_wd = cfg.weight_decay + (cfg.wd_final - cfg.weight_decay) * t
            else:  # cos
                current_wd = cfg.wd_final + 0.5 * (cfg.weight_decay - cfg.wd_final) * (
                    1 + math.cos(math.pi * t)
                )
            for g in opt.param_groups:
                g["weight_decay"] = current_wd

        if (
            getattr(cfg, "spectral_l1_reg_schedule", "none") != "none"
            and getattr(cfg, "spectral_l1_reg_coef_final", None) is not None
        ):
            t = curr_iter / cfg.iterations
            coef_init = cfg.spectral_l1_reg_coef
            coef_final = cfg.spectral_l1_reg_coef_final
            if cfg.spectral_l1_reg_schedule == "linear":
                current_coef = coef_init + (coef_final - coef_init) * t
            else:  # cos
                current_coef = coef_final + 0.5 * (coef_init - coef_final) * (
                    1 + math.cos(math.pi * t)
                )
            for g in opt.param_groups:
                g["spectral_l1_reg_coef"] = current_coef

        if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
            opt.train()
        (
            opt.step()
            if cfg.opt != "sophiag"
            else opt.step(bs=cfg.sophia_bs * cfg.sequence_length)
        )
        if cfg.scheduler != "none":
            scheduler.step()

        if (
            cfg.opt in ("adamw-spectral-l1-reg", "lion-spectral-l1-reg", "galore")
            and getattr(cfg, "spectral_l1_reg_switch_step", None) is not None
            and getattr(cfg, "spectral_l1_reg_coef_final", None) is not None
            and curr_iter + 1 == cfg.spectral_l1_reg_switch_step
        ):
            for g in opt.param_groups:
                g["spectral_l1_reg_coef"] = cfg.spectral_l1_reg_coef_final

        _pending_grad_er = {}
        if (
            cfg.wandb
            and cfg.effective_rank_interval > 0
            and (curr_iter + 1) % cfg.effective_rank_interval == 0
            and distributed_backend.is_master_process()
        ):
            _pending_grad_er = grad_effective_ranks(
                distributed_backend.get_raw_model(model)
            )

        if cfg.opt == "sophiag":
            opt.zero_grad(set_to_none=True)
            if curr_iter % cfg.precondition_frequency == cfg.precondition_frequency - 1:
                sample_again = model(x, targets=y, get_logits=True)
                samp_dist = torch.distributions.Categorical(
                    logits=sample_again["logits"]
                )
                y_sample = samp_dist.sample()
                loss_sampled = torch.nn.functional.cross_entropy(
                    sample_again["logits"].view(-1, sample_again["logits"].size(-1)),
                    y_sample.view(-1),
                    ignore_index=-1,
                )
                (loss_sampled / cfg.acc_steps).backward()
                opt.update_hessian()
                opt.zero_grad(set_to_none=True)
                model.zero_grad()
        elif cfg.opt == "mars":
            opt.zero_grad(set_to_none=True)
            opt.update_last_grad()
        else:
            opt.zero_grad(set_to_none=True)

        if cfg.weight_average:
            weight_averager.step(
                not_compiled_model, distributed_backend.is_master_process()
            )
        if cfg.exponential_weight_average:
            ewa.step(not_compiled_model, distributed_backend.is_master_process())

        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1
        pbar.update(1)

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps
            pbar.set_postfix(loss=f"{train_loss:.3f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
            train_aux_losses = {
                f"train/{k}": v for k, v in outputs["aux_losses"].items()
            }

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            if cfg.opt == "prodigy":
                prodigy_efective_lrs = log_prodigy_lr(opt)

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e}"
            )
            _append_local_metric(
                exp_dir,
                {
                    "kind": "train",
                    "iter": curr_iter,
                    "loss": train_loss,
                    "lr": current_lrs[0],
                },
            )
            if cfg.opt == "prodigy":
                print(f"effective_lr={prodigy_efective_lrs[0]:.2e}")

            if cfg.wandb:
                wandb_logs = {
                    "tokens": tokens,
                    "iter": curr_iter,
                    "train/loss": train_loss,
                    "train/perplexity": 2.71828**train_loss,
                    "lr": current_lrs[0],
                    "weight_decay": opt.param_groups[0].get("weight_decay", 0.0),
                    **({"spectral_l1_reg_coef": opt.param_groups[0]["spectral_l1_reg_coef"]}
                       if "spectral_l1_reg_coef" in opt.param_groups[0] else {}),
                    **({"spectral_l1_reg_coupled": int(opt.param_groups[0]["coupled"])}
                       if "coupled" in opt.param_groups[0] else {}),
                    "iter_dt": dt,
                    "max_grad_norm": max(grad_norms).item() if grad_norms else 0,
                    "mean_grad_norm": (
                        torch.tensor(grad_norms).mean().item() if grad_norms else 0
                    ),
                    **train_aux_losses,
                }

                if cfg.opt == "prodigy":
                    wandb_logs["effective_lr"] = prodigy_efective_lrs[0]

                if cfg.log_parameter_norms:
                    raw_model = distributed_backend.get_raw_model(model)
                    model_norm = get_parameter_norms(raw_model, order=cfg.norm_order)
                    wandb_logs["model_norm"] = model_norm

                wandb.log(wandb_logs)

            grad_norms = []

        if (
            cfg.effective_rank_interval > 0
            and curr_iter % cfg.effective_rank_interval == 0
            and distributed_backend.is_master_process()
        ):
            raw_model = distributed_backend.get_raw_model(model)
            er_logs = model_effective_ranks(raw_model)
            opt_er_logs = optimizer_state_effective_ranks(raw_model, opt)
            rank_logs = {
                "kind": "rank",
                "iter": curr_iter,
                "effective_rank/mean_weighted": er_logs.get("effective_rank/mean_weighted", 0.0),
                **_pending_grad_er,
                **opt_er_logs,
            }
            _append_local_metric(exp_dir, rank_logs)
            print(f"RANK_METRICS {json.dumps(rank_logs, sort_keys=True)}")
            if cfg.wandb:
                wandb.log(rank_logs)

    pbar.close()
    return stats


def eval_and_log(
    tokens,
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    exp_dir,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()
    if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity, val_aux_losses, router_logits = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        moe=cfg.moe,
        get_router_logits=cfg.moe and cfg.plot_router_logits,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )
    _append_local_metric(
        exp_dir,
        {
            "kind": "val",
            "iter": curr_iter,
            "loss": val_loss,
            "perplexity": val_perplexity,
            "accuracy": val_acc,
        },
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "tokens": tokens,
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
                **val_aux_losses,
            }
        else:
            logs = {
                "tokens": tokens,
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
                **val_aux_losses,
            }
        if cfg.moe and cfg.plot_router_logits:
            routing_logs = visualize_routing(router_logits, cfg)
            logs = {**logs, **routing_logs}

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
