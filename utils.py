import logging
import os


def get_logger(name, path):
    """Get a logger that writes to both file and console.
    Avoids duplicate handlers on repeated calls.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    pathname = os.path.join(path, "log.txt")
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_list_to_file(path, thelist):
    """Save a list to file, one item per line."""
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)
