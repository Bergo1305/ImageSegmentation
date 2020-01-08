import os
import logging
import numpy as np


def create_logger(name, level=logging.DEBUG):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(
        logging.Formatter(
            '[%(name)s:%(filename)s:%(lineno)d] - [%(process)d] - %(asctime)s - %(levelname)s - %(message)s'
        )
    )

    logger.addHandler(stdout_handler)

    return logger


LOGGER = create_logger("IMAGE-SEGMENTATION")

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

KERNEL_VERTICAL = np.array(
                    [
                       [-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]
                    ],
                    np.float32
)

KERNEL_HORIZONTAL = np.array(
                    [
                        [1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]
                    ],
                    np.float32
)

LAPLACIAN = np.array(
                    [
                        [1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]
                    ],
                    np.float32
)
