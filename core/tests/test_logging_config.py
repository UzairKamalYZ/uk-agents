import logging
from logging.handlers import RotatingFileHandler
from unittest.mock import patch

import logging_config


def test_log_dir_is_anchored_to_project_root():
    assert logging_config.LOG_DIR.is_absolute()
    assert logging_config.LOG_DIR.name == "logs"


def test_setup_logging_rotates_file_and_quiets_token_leaking_loggers(tmp_path):
    with patch.object(logging_config, "LOG_DIR", tmp_path), \
         patch("logging_config.logging.basicConfig") as basic:
        logging_config.setup_logging()
    handlers = basic.call_args.kwargs["handlers"]
    files = [h for h in handlers if isinstance(h, RotatingFileHandler)]
    assert len(files) == 1 and files[0].baseFilename == str(tmp_path / "agent.log")
    assert any(type(h) is logging.StreamHandler for h in handlers)
    assert logging.getLogger("httpx").level == logging.WARNING
    for h in handlers:
        h.close()


def test_setup_logging_without_console_uses_own_file(tmp_path):
    with patch.object(logging_config, "LOG_DIR", tmp_path), \
         patch("logging_config.logging.basicConfig") as basic, \
         patch.dict("os.environ", {"LOG_LEVEL": "debug"}):
        logging_config.setup_logging(console=False, log_file="own-process.log")
    kwargs = basic.call_args.kwargs
    assert [h.baseFilename for h in kwargs["handlers"]] == [str(tmp_path / "own-process.log")]
    assert kwargs["level"] == "DEBUG"
    for h in kwargs["handlers"]:
        h.close()
