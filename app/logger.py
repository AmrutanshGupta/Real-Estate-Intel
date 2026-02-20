import logging
import sys

from app.config import Config


def _build(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.DEBUG)
        h.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        log.addHandler(h)
    return log


logger = _build(Config.PROJECT_NAME)
