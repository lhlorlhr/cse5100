import json
from pathlib import Path


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return str(value)


_BACKEND_SUMMARY_WRITER = None

try:
    from tensorboardX import SummaryWriter as _BACKEND_SUMMARY_WRITER
except ModuleNotFoundError:
    try:
        from torch.utils.tensorboard import SummaryWriter as _BACKEND_SUMMARY_WRITER
    except Exception:
        _BACKEND_SUMMARY_WRITER = None


class SummaryWriter:
    def __init__(self, log_dir, *args, **kwargs):
        self.log_dir = log_dir
        self._scalars = {}
        self._backend = None

        if _BACKEND_SUMMARY_WRITER is not None:
            self._backend = _BACKEND_SUMMARY_WRITER(log_dir, *args, **kwargs)

    def _record_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        step_key = "None" if global_step is None else str(global_step)
        step_scalars = self._scalars.setdefault(step_key, {})
        values = step_scalars.setdefault(main_tag, {})
        values.update(
            {str(tag): _to_jsonable(value) for tag, value in tag_scalar_dict.items()}
        )

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        self._record_scalars(main_tag, tag_scalar_dict, global_step)
        if self._backend is not None:
            self._backend.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        self._record_scalars(tag, {tag: scalar_value}, global_step)
        if self._backend is not None:
            self._backend.add_scalar(tag, scalar_value, global_step, *args, **kwargs)

    def export_scalars_to_json(self, path):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._backend is not None and hasattr(self._backend, "export_scalars_to_json"):
            self._backend.export_scalars_to_json(str(output_path))
            return

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self._scalars, handle, indent=2, sort_keys=True)

    def close(self):
        if self._backend is not None:
            self._backend.close()

    def __getattr__(self, name):
        if self._backend is None:
            raise AttributeError(f"{self.__class__.__name__!s} object has no attribute {name!r}")
        return getattr(self._backend, name)
