"""模型训练、逐轮测试和单批次诊断。"""

import importlib.util
import math
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .metrics import ClassificationMeter
from .reproducibility import resolve_device
from .visualization import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_training_history,
    save_json,
)


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    num_classes,
    gradient_clip=1.0,
    mixed_precision=False,
    scaler=None,
    progress=True,
    description="Train",
):
    """训练一个 epoch 并返回训练指标。"""

    model.train()
    meter = ClassificationMeter(num_classes)
    amp_enabled = bool(mixed_precision and device.type == "cuda")

    progress_bar = tqdm(
        data_loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
        disable=not progress,
    )
    for inputs, targets in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _clip_gradients(model, gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            _clip_gradients(model, gradient_clip)
            optimizer.step()

        meter.update(logits, targets, loss)

    return meter.compute()


def evaluate(
    model,
    data_loader,
    criterion,
    device=None,
    num_classes=12,
    mixed_precision=False,
    progress=True,
    description="Test",
):
    """在指定数据集上计算损失、分类指标和混淆矩阵。"""

    device = resolve_device(device)
    model.eval()
    meter = ClassificationMeter(num_classes)
    amp_enabled = bool(mixed_precision and device.type == "cuda")

    with torch.inference_mode():
        progress_bar = tqdm(
            data_loader,
            desc=description,
            leave=False,
            dynamic_ncols=True,
            disable=not progress,
        )
        for inputs, targets in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast_context(device, amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            meter.update(logits, targets, loss)

    return meter.compute()


def fit(
    model,
    train_loader,
    test_loader,
    output_dir,
    epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    gradient_clip=1.0,
    device=None,
    mixed_precision=False,
    normalization_state=None,
    config=None,
    resume_from=None,
    class_names=None,
    verbose=True,
    warmup_epochs=5,
    warmup_start_factor=0.2,
    eval_interval=1,
    checkpoint_interval=1,
    fast_mode=True,
    prefer_cupy=True,
    gpu_resident_data=True,
):
    """训练模型并按测试准确率保存最佳检查点。

    测试集参与最佳模型选择，因此 ``best.pt`` 是 test-selected checkpoint，
    其测试指标不能解释为完全无偏的最终泛化估计。默认参数保持逐轮测试和
    保存；Notebook 可提高间隔以减少验证与磁盘同步开销。
    """

    _validate_fit_options(
        epochs,
        learning_rate,
        weight_decay,
        gradient_clip,
        warmup_epochs,
        warmup_start_factor,
        eval_interval,
        checkpoint_interval,
        fast_mode,
        prefer_cupy,
        gpu_resident_data,
    )
    device = resolve_device(device)
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    _configure_cuda_fast_paths(device, fast_mode)
    model.to(device)
    amp_enabled = bool(mixed_precision and device.type == "cuda")
    train_runtime_loader, test_runtime_loader, resident_data = _prepare_data_loaders(
        train_loader,
        test_loader,
        device=device,
        amp_enabled=amp_enabled,
        enabled=bool(fast_mode and gpu_resident_data),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer, optimizer_backend = _create_optimizer(
        model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        use_fused=fast_mode,
    )
    scheduler, effective_warmup_epochs = _create_learning_rate_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
    )
    scaler = _create_grad_scaler(amp_enabled)
    num_classes = _infer_num_classes(model)

    training_config = {
        "epochs": epochs,
        "batch_size": getattr(train_loader, "batch_size", None),
        "optimizer": optimizer_backend,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": "LinearWarmupCosineAnnealingLR",
        "warmup_epochs": effective_warmup_epochs,
        "warmup_start_factor": float(warmup_start_factor),
        "cosine_epochs": epochs - effective_warmup_epochs,
        "gradient_clip": gradient_clip,
        "mixed_precision": bool(mixed_precision),
        "device": str(device),
        "eval_interval": eval_interval,
        "checkpoint_interval": checkpoint_interval,
        "fast_mode": bool(fast_mode),
        "gpu_resident_data": resident_data,
        "selection_split": "test",
        "test_selected_checkpoint": True,
    }
    training_config.update(dict(config or {}))

    start_epoch = 1
    history = []
    best_test_accuracy = float("-inf")
    best_test_loss = float("inf")
    if resume_from is not None:
        resumed = load_checkpoint(
            resume_from,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(resumed.get("epoch", 0)) + 1
        history = list(resumed.get("history") or [])
        best_test_accuracy = _optional_float(
            resumed.get("best_test_accuracy"),
            default=float("-inf"),
        )
        best_test_loss = _optional_float(
            resumed.get("best_test_loss"),
            default=float("inf"),
        )

    execution_model, execution_backend = _prepare_execution_model(
        model,
        device=device,
        enabled=fast_mode,
        prefer_cupy=prefer_cupy,
    )
    _warm_up_execution_model(
        execution_model,
        train_runtime_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        amp_enabled=amp_enabled,
    )
    selected_batch_size = getattr(train_runtime_loader, "batch_size", None)
    execution_backend = getattr(
        execution_model,
        "backend_name",
        execution_backend,
    )
    training_config["execution_backend"] = execution_backend

    if verbose:
        print(
            f"device={device}, amp={amp_enabled}, backend={execution_backend}, "
            f"gpu_resident_data={resident_data}, batch_size={selected_batch_size}, "
            f"epochs={start_epoch}..{epochs}"
        )

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate_now = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model=execution_model,
            data_loader=train_runtime_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            gradient_clip=gradient_clip,
            mixed_precision=amp_enabled,
            scaler=scaler,
            progress=verbose,
            description=f"Train {epoch:03d}/{epochs:03d}",
        )
        should_evaluate = epoch % eval_interval == 0 or epoch == epochs
        test_metrics = None
        if should_evaluate:
            test_metrics = evaluate(
                model=execution_model,
                data_loader=test_runtime_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                mixed_precision=amp_enabled,
                progress=verbose,
                description=f"Test  {epoch:03d}/{epochs:03d}",
            )
        elapsed = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"] if test_metrics is not None else None,
            "test_accuracy": (
                test_metrics["accuracy"] if test_metrics is not None else None
            ),
            "learning_rate": learning_rate_now,
            "epoch_time": elapsed,
        }
        history.append(record)

        is_best = bool(
            test_metrics is not None
            and _is_better_test_result(
                test_metrics["accuracy"],
                test_metrics["loss"],
                best_test_accuracy,
                best_test_loss,
            )
        )
        if is_best:
            best_test_accuracy = test_metrics["accuracy"]
            best_test_loss = test_metrics["loss"]

        scheduler.step()
        checkpoint_options = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "history": history,
            "normalization_state": normalization_state,
            "config": training_config,
            "best_test_accuracy": best_test_accuracy,
            "best_test_loss": best_test_loss,
            "test_metrics": test_metrics,
        }
        should_checkpoint = epoch % checkpoint_interval == 0 or epoch == epochs
        if should_checkpoint:
            save_checkpoint(checkpoint_dir / "last.pt", **checkpoint_options)
            save_json(history, output_dir / "history.json")
        if is_best:
            save_checkpoint(checkpoint_dir / "best.pt", **checkpoint_options)

        last_epoch = epoch
        if verbose:
            marker = " *" if is_best else ""
            message = (
                f"epoch {epoch:03d}/{epochs:03d} | "
                f"train loss {train_metrics['loss']:.4f}, "
                f"acc {train_metrics['accuracy'] * 100:.2f}%"
            )
            if test_metrics is not None:
                message += (
                    f" | test loss {test_metrics['loss']:.4f}, "
                    f"acc {test_metrics['accuracy'] * 100:.2f}%"
                )
            message += f" | lr {learning_rate_now:.3e} | {elapsed:.2f}s{marker}"
            print(message)

    if last_epoch < 1:
        raise RuntimeError("没有执行任何训练 epoch，请检查 epochs 和恢复检查点")

    save_checkpoint(
        checkpoint_dir / "final.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=last_epoch,
        history=history,
        normalization_state=normalization_state,
        config=training_config,
        best_test_accuracy=best_test_accuracy,
        best_test_loss=best_test_loss,
    )

    best_path = checkpoint_dir / "best.pt"
    best_checkpoint = load_checkpoint(best_path, model, map_location=device)
    best_metrics = evaluate(
        model=execution_model,
        data_loader=test_runtime_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        mixed_precision=amp_enabled,
        progress=verbose,
        description="Best test",
    )
    best_metrics.update(
        {
            "best_epoch": int(best_checkpoint["epoch"]),
            "selection_split": "test",
            "test_selected_checkpoint": True,
        }
    )

    save_json(best_metrics, output_dir / "metrics.json")
    plot_training_history(history, figure_dir / "training_curves.png")
    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        figure_dir / "confusion_matrix.png",
        class_names=class_names,
    )
    plot_per_class_accuracy(
        best_metrics["per_class_accuracy"],
        figure_dir / "per_class_accuracy.png",
        class_names=class_names,
    )
    return history


def overfit_one_batch(
    model,
    batch,
    max_steps=500,
    target_accuracy=0.99,
    learning_rate=1e-3,
    weight_decay=0.0,
    gradient_clip=1.0,
    device=None,
    verbose=True,
):
    """反复训练一个批次，诊断数据到梯度的完整链路。"""

    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps 必须是正整数")
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy 必须在 (0, 1] 内")

    device = resolve_device(device)
    model.to(device)
    model.train()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history = []

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        _clip_gradients(model, gradient_clip)
        optimizer.step()

        accuracy = float((logits.argmax(dim=1) == targets).float().mean())
        history.append(
            {"step": step, "loss": float(loss.detach()), "accuracy": accuracy}
        )
        if verbose and (step == 1 or step % 25 == 0 or accuracy >= target_accuracy):
            print(
                f"step {step:03d}/{max_steps:03d} | "
                f"loss {float(loss.detach()):.4f} | acc {accuracy * 100:.2f}%"
            )
        if accuracy >= target_accuracy:
            break
    return history


class _ModelExecutor:
    """为 CuPy 提供一次失败后回退到 PyTorch eager 的轻量代理。"""

    def __init__(self, model, cupy_active=False):
        self.model = model
        self.cupy_active = cupy_active

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    @property
    def backend_name(self):
        if self.cupy_active:
            return "spikingjelly-cupy"
        return "torch-eager"

    def __call__(self, *args, **kwargs):
        if self.cupy_active:
            try:
                return self.model(*args, **kwargs)
            except Exception as error:
                if _is_cuda_out_of_memory(error):
                    raise
                self.fallback(error)

        return self.model(*args, **kwargs)

    def fallback(self, error):
        if self.cupy_active:
            # 兼容问题已经在启用前处理；其他运行时失败则无警告回退。
            self.cupy_active = False
            _set_snn_backend(self.model, "torch")
            _reset_model_state(self.model)
            return True
        return False


def _warm_up_execution_model(
    execution_model,
    data_loader,
    criterion,
    optimizer,
    device,
    amp_enabled,
):
    if not isinstance(execution_model, _ModelExecutor):
        return

    if hasattr(data_loader, "features") and hasattr(data_loader, "labels"):
        warmup_size = min(512, data_loader.labels.shape[0])
        inputs = data_loader.features[:warmup_size]
        targets = data_loader.labels[:warmup_size]
    else:
        try:
            inputs, targets = next(iter(data_loader))
        except StopIteration:
            raise RuntimeError("训练 DataLoader 不能为空")

    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    # CuPy 失败时只重试一次 PyTorch eager，不进入其他执行后端。
    for _ in range(2):
        try:
            execution_model.train()
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_enabled):
                logits = execution_model(inputs)
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            _reset_model_state(execution_model.model)
            return
        except Exception as error:
            optimizer.zero_grad(set_to_none=True)
            _reset_model_state(execution_model.model)
            if not execution_model.fallback(error):
                raise

    raise RuntimeError("CuPy 与 PyTorch eager 均无法完成前向和反向预热")


def _prepare_data_loaders(
    train_loader,
    test_loader,
    device,
    amp_enabled,
    enabled,
):
    if not enabled or device.type != "cuda":
        return train_loader, test_loader, False

    from ..data import create_device_data_loader

    storage_dtype = torch.float16 if amp_enabled else torch.float32
    fast_train_loader = None
    fast_test_loader = None
    try:
        fast_train_loader = create_device_data_loader(
            train_loader,
            device=device,
            dtype=storage_dtype,
        )
        fast_test_loader = create_device_data_loader(
            test_loader,
            device=device,
            dtype=storage_dtype,
        )
    except Exception as error:
        # 自定义 Dataset/Transform 不支持批变换时继续使用原 DataLoader。
        warnings.warn(
            f"数据无法常驻 GPU，自动回退到原 DataLoader：{error}",
            RuntimeWarning,
        )
        del fast_train_loader, fast_test_loader
        torch.cuda.empty_cache()
        return train_loader, test_loader, False

    return fast_train_loader, fast_test_loader, True


def _is_cuda_out_of_memory(error):
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    typed_oom = cuda_oom_type is not None and isinstance(error, cuda_oom_type)
    return typed_oom or "out of memory" in str(error).lower()


def _prepare_execution_model(
    model,
    device,
    enabled,
    prefer_cupy,
):
    if not enabled or device.type != "cuda":
        return model, "torch-eager"

    cupy_available = bool(prefer_cupy and importlib.util.find_spec("cupy") is not None)
    if cupy_available:
        _ensure_spikingjelly_numpy_compatibility()
    if cupy_available and _set_snn_backend(model, "cupy") > 0:
        executor = _ModelExecutor(model, cupy_active=True)
        return executor, "spikingjelly-cupy"

    return model, "torch-eager"


def _ensure_spikingjelly_numpy_compatibility():
    import numpy as np

    # 旧版 SpikingJelly 的 CuPy 内核生成器仍读取 np.int；新 NumPy 已移除此别名。
    if "int" not in np.__dict__:
        np.__dict__["int"] = int


def _set_snn_backend(model, backend):
    switched = 0
    for module in model.modules():
        if not hasattr(module, "backend"):
            continue
        try:
            supported_backends = module.supported_backends
        except (AttributeError, ValueError):
            continue
        if backend in supported_backends:
            module.backend = backend
            switched += 1

    if switched > 0 and hasattr(model, "backend"):
        model.backend = backend
    return switched


def _reset_model_state(model):
    reset = getattr(model, "reset", None)
    if callable(reset):
        reset()


def _configure_cuda_fast_paths(device, enabled):
    if not enabled or device.type != "cuda":
        return

    # 固定形状的全连接层可使用 Tensor Core；TF32 是 FP32 回退路径的加速保障。
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True


def _create_optimizer(
    parameters,
    learning_rate,
    weight_decay,
    device,
    use_fused,
):
    options = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    if use_fused and device.type == "cuda":
        try:
            return torch.optim.AdamW(parameters, fused=True, **options), "AdamW(fused)"
        except (TypeError, RuntimeError) as error:
            warnings.warn(
                f"fused AdamW 不可用，回退到标准 AdamW：{error}",
                RuntimeWarning,
            )
    return torch.optim.AdamW(parameters, **options), "AdamW"


def _validate_fit_options(
    epochs,
    learning_rate,
    weight_decay,
    gradient_clip,
    warmup_epochs,
    warmup_start_factor,
    eval_interval,
    checkpoint_interval,
    fast_mode,
    prefer_cupy,
    gpu_resident_data,
):
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs 必须是正整数")
    if learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if weight_decay < 0:
        raise ValueError("weight_decay 不能小于 0")
    if gradient_clip is not None and gradient_clip <= 0:
        raise ValueError("gradient_clip 必须大于 0 或为 None")
    if (
        not isinstance(warmup_epochs, int)
        or isinstance(warmup_epochs, bool)
        or warmup_epochs < 0
    ):
        raise ValueError("warmup_epochs 必须是非负整数")
    if (
        not isinstance(warmup_start_factor, (int, float))
        or isinstance(warmup_start_factor, bool)
        or not 0.0 < warmup_start_factor <= 1.0
    ):
        raise ValueError("warmup_start_factor 必须在 (0, 1] 范围内")
    interval_options = {
        "eval_interval": eval_interval,
        "checkpoint_interval": checkpoint_interval,
    }
    for name, value in interval_options.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    boolean_options = {
        "fast_mode": fast_mode,
        "prefer_cupy": prefer_cupy,
        "gpu_resident_data": gpu_resident_data,
    }
    for name, value in boolean_options.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} 必须是 bool")


def _create_learning_rate_scheduler(
    optimizer,
    epochs,
    warmup_epochs=5,
    warmup_start_factor=0.2,
):
    """创建线性 warmup 后接余弦退火的统一学习率调度器。"""

    # 短跑训练至少保留一个余弦阶段 epoch，正式 100 epoch 配置保持 5+95。
    effective_warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))
    cosine_epochs = max(epochs - effective_warmup_epochs, 1)

    def learning_rate_multiplier(step):
        if effective_warmup_epochs > 0 and step < effective_warmup_epochs:
            progress = step / effective_warmup_epochs
            return warmup_start_factor + (1.0 - warmup_start_factor) * progress

        cosine_step = step - effective_warmup_epochs
        cosine_progress = min(max(cosine_step / cosine_epochs, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_multiplier,
    )
    return scheduler, effective_warmup_epochs


def _infer_num_classes(model):
    num_classes = getattr(model, "num_classes", None)
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise AttributeError("model 必须提供正整数 num_classes 属性")
    return num_classes


def _clip_gradients(model, gradient_clip):
    if gradient_clip is not None:
        parameters = list(model.parameters())
        use_foreach = bool(parameters and parameters[0].device.type == "cuda")
        try:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                gradient_clip,
                foreach=use_foreach,
            )
        except TypeError:
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)


def _autocast_context(device, enabled):
    if enabled:
        return torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=True,
        )
    return nullcontext()


def _create_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _is_better_test_result(accuracy, loss, best_accuracy, best_loss):
    if accuracy > best_accuracy:
        return True
    return accuracy == best_accuracy and loss < best_loss


def _optional_float(value, default):
    return default if value is None else float(value)
