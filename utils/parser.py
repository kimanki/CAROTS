# adapted from: https://github.com/facebookresearch/SlowFast

"""Argument parser functions."""

import argparse
import sys
import os

from config import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(
        description="Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    # Setup cfg.
    cfg = get_cfg_defaults()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    
    # Overwrite cfg.DATA.N_VAR and related variables according to dataset
    valid_datasets = {
        "SMD": 38,
        "MSL": 55,
        "PSM": 25,
        "SWaT": 51,
        "WADI": 123,
        "Lorenz96": 128,
        "VAR": 128,
        "itransformer": 862,
    }
    if cfg.DATA.NAME == 'SWaT':
        cfg.DATA.N_VAR = valid_datasets["SWaT"]
        cfg.DATA.DOWNSAMPLE_RATE = 5
    elif cfg.DATA.NAME == 'WADI':
        cfg.DATA.N_VAR = valid_datasets["WADI"]
        cfg.DATA.DOWNSAMPLE_RATE = 5
        cfg.SCORER.TYPE = "cos"
    elif "SMD" in cfg.DATA.NAME:
        cfg.DATA.N_VAR = valid_datasets["SMD"]
    elif "MSL" in cfg.DATA.NAME:
        cfg.DATA.N_VAR = valid_datasets["MSL"]
    elif "Lorenz96" in cfg.DATA.NAME:
        cfg.DATA.N_VAR = valid_datasets["Lorenz96"]
        cfg.DATA.WIN_SIZE = 2
        cfg.CUTS_PLUS.INPUT_STEP = 1
    elif "VAR" in cfg.DATA.NAME:
        cfg.DATA.N_VAR = valid_datasets["VAR"]
        cfg.DATA.WIN_SIZE = 4
        cfg.CUTS_PLUS.INPUT_STEP = 3
        cfg.SCORER.TYPE = "cos"
        cfg.TRANSFORM.AddBias.bias_candidates = (1.5, 1.0, 0.5, -0.5, -1.0, -1.5)
    else:
        cfg.DATA.N_VAR = valid_datasets[cfg.DATA.NAME]

    # Update model-specific configurations
    cfg.TIMESNET.enc_in = cfg.DATA.N_VAR
    cfg.TIMESNET.c_out = cfg.DATA.N_VAR
    cfg.ITRANSFORMER.enc_in = cfg.DATA.N_VAR
    cfg.ITRANSFORMER.c_out = cfg.DATA.N_VAR
    cfg.CUTS_PLUS.N_NODES = cfg.DATA.N_VAR
    cfg.CUTS_PLUS.N_GROUPS = cfg.DATA.N_VAR
    
    cfg.TRAIN.CHECKPOINT_DIR = os.path.join(cfg.TRAIN.CHECKPOINT_DIR, cfg.DATA.NAME)
    cfg.RESULT_DIR = os.path.join(cfg.RESULT_DIR, cfg.DATA.NAME)

    return cfg
