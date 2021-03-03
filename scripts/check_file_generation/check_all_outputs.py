import argparse
import logging
import os
import datetime as dt
import sys

sys.path.append("/PAGER/WP8/data_management/")

from data_management.check.wp2.check_swift import SwiftCheck
# from data_management.check.wp3.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY-%mm-%dd")
    parser.add_argument('-logdir', action="store", default=None, type=str,
                        help="Log directory if logging is to be enabled.")

    args = parser.parse_args()

    if args.logdir is not None:
        if args.date is not None:
            log_file = "wp3_plot_all_kp_{}.log".format(args.date)
        else:
            log_date = dt.datetime.utcnow().replace(hour=0, minute=0, second=0,
                                                    microsecond=0).strptime(args.date, "%Y-%m-%d")
            log_file = "wp3_plot_all_kp_{}.log".format(log_date)
        logging.basicConfig(filename=os.path.join(args.logdir, log_file), level=logging.INFO)

    # WP2
    # SWIFT

    checker = SwiftCheck()
    checker.run_check()
