import logging
from pathlib import Path
import warnings


class _RuntimeLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR or (
            record.name == "memmamba.epoch" and record.levelno == logging.INFO
        )


def configure_runtime_logging(log_dir: Path | None = None) -> logging.Logger:
    # Suppress Python warnings and ordinary dependency/model log messages.
    # Exceptions and ERROR/CRITICAL records remain visible.
    warnings.filterwarnings("ignore")
    handlers = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "train.log", encoding="utf-8"))

    for handler in handlers:
        handler.addFilter(_RuntimeLogFilter())
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    epoch_logger = logging.getLogger("memmamba.epoch")
    epoch_logger.setLevel(logging.INFO)
    epoch_logger.propagate = True
    return epoch_logger
