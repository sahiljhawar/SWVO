import argparse
import logging
import os
import datetime as dt
import sys
import inspect

LOCAL_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(LOCAL_PATH, "../../"))

from data_management.check.wp2.check_swift import SwiftCheck
from data_management.check.wp3.check_kp import KpDataCheck
from data_management.check.wp3.check_plasma import PlasmaDataCheck
from data_management.check.wp6.check_rbm_forecast import RBMForecastCheck
from data_management.check.wp6.check_ring_current import RingCurrentCheck


def wp2_check_swift(date, notify=True):
    logging.info("Checking SWIFT output... ")
    checker = SwiftCheck()
    checker.run_check(date, notify)


def wp3_check_kp(date, product, model=None, notify=True):
    logging.info("Checking Kp output for product {}".format(product.upper()))
    checker = KpDataCheck()
    checker.run_check(product, model, date, notify)


def wp3_check_plasma(date, product, notify=True):
    logging.info("Checking Plasma output for product {}".format(product.upper()))
    checker = PlasmaDataCheck()
    checker.run_check(product, date, notify)


def wp6_check_rbm_forecast(date, notify=True):
    logging.info("Checking RBM Forecast output...")
    checker = RBMForecastCheck()
    checker.run_check(date, notify)


def wp6_check_ring_current(date, notify=True):
    logging.info("Checking Ring Current output...")
    checker = RingCurrentCheck()
    checker.run_check(date, notify)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY%mm%ddT%HH to specify the hours or"
                             "in the format %YYYY%mm%dd with default hour zero")
    parser.add_argument('-logdir', action="store", default=None, type=str,
                        help="Log directory if logging is to be enabled.")
    parser.add_argument('-notify', action="store", default=False, type=bool,
                        help="Log directory if logging is to be enabled.")

    args = parser.parse_args()

    if args.date is not None:
        if "T" not in args.date:
            check_date = args.date + "T000000"
        else:
            check_date = args.date + "0000"
    else:
        check_date = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0).strftime("%Y%m%dT%H%M%S")

    if args.logdir is not None:
        log_file = "wp8_output_checker_{}.log".format(check_date)
        logging.basicConfig(filename=os.path.join(args.logdir, log_file), level=logging.INFO)

    check_date = dt.datetime.strptime(check_date, "%Y%m%dT%H%M%S")
    logging.info("Start checking all outputs for date {} \n".format(check_date))
    # WP2
    logging.info("Starting Check for WP2 modules \n")
    wp2_check_swift(check_date, notify=args.notify)
    logging.info("")
    # WP3
    logging.info("Starting Check for WP3 modules \n")
    wp3_check_kp(check_date, "swpc", notify=args.notify)
    wp3_check_kp(check_date, "niemegk", notify=args.notify)
    wp3_check_kp(check_date - dt.timedelta(hours=1), "swift", notify=args.notify)
    wp3_check_kp(check_date - dt.timedelta(hours=1), "l1", model="KP-FULL-SW-PAGER", notify=args.notify)
    wp3_check_kp(check_date - dt.timedelta(hours=1), "l1", model="HP60-FULL-SW-SWAMI-PAGER", notify=args.notify)
    logging.info("\n")
    # PLASMA
    wp3_check_plasma(check_date, "ca", notify=args.notify)
    wp3_check_plasma(check_date - dt.timedelta(hours=1), "gfz_plasma", notify=args.notify)
    logging.info("\n")
    # WP6
    logging.info("Starting Check for WP6 modules \n")

    wp6_check_rbm_forecast(check_date - dt.timedelta(hours=1), notify=args.notify)
    wp6_check_ring_current(check_date, notify=args.notify)
