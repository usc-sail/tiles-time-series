import argparse


def parse_args():
    """
    Parse the args

    Returns:
    args - args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to where config files are saved")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    
    args = parser.parse_args()

    # Print out if args is empty
    if args.tiles_path is None: print('Warning: \'tiles_path\' (Path to the root folder containing TILES data) is not specified, use default value for now')
    if args.config is None: print('Warning: \'config\' (Path to where config files are saved) is not specified, use default value for now')
    if args.experiment is None: print('Warning: \'experiment\' (Experiment name) is not specified, use default value for now')

    return args