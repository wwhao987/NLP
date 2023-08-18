import logging

import utils

if __name__ == '__main__':
    utils.set_logger(save=True, log_path=r".\train.log")
    logging.info("Test!!!!")
