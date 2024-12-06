import sys
import argparse

def preparse_args():
    if '-h' in sys.argv or '--help' in sys.argv:
        return None, None
    parser = argparse.ArgumentParser(description='preparse some arguments before loading core packages.')
    add_preparse_argmuents(parser)
    return parser.parse_known_args()

def add_preparse_argmuents(parser):
    parser.add_argument("--zhilight-version",
                        type=str,
                        default=None,
                        help="If provided, the server will use the special version zhilight.")
    parser.add_argument(
        "--environ",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI environs to apply to the app. "
        "We accept multiple --environ arguments. "
        'The value should be "Key1=Value1:Key2=Value2:...:Keyn=Valuen". '
        'Or using --environ="Key1=Value1" --environ="Key2=Value2" mode.')
    parser.add_argument(
        "--pip",
        type=str,
        action="append",
        default=[],
        help="Additional packages to be forced installed to the container. "
        'use --pip="torch==2.0.1" --pip="transformers" mode.')