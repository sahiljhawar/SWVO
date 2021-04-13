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


def wp2_check_swift(date):
    logging.info("Checking SWIFT output... ")
    checker = SwiftCheck()
    checker.run_check(date)


def wp3_check_kp(date, product, model=None):
    logging.info("Checking Kp output for product {}".format(product))
    checker = KpDataCheck()
    checker.run_check(product, model, date)


def wp3_check_plasma(date, product):
    logging.info("Checking Plasma output for product {}".format(product))
    checker = PlasmaDataCheck()
    checker.run_check(product, date)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY%mm%ddT%HH")
    parser.add_argument('-logdir', action="store", default=None, type=str,
                        help="Log directory if logging is to be enabled.")

    args = parser.parse_args()

    if args.date is not None:
        check_date = args.date + "0000"
    else:
        check_date = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0).strftime("%Y%m%dT%H%M%S")

    if args.logdir is not None:
        log_file = "wp8_output_checker_{}.log".format(check_date)
        logging.basicConfig(filename=os.path.join(args.logdir, log_file), level=logging.INFO)

    check_date = dt.datetime.strptime(check_date, "%Y%m%dT%H%M%S")
    logging.info("Start checking outputs for date {}".format(check_date))
    # WP2
    # SWIFT
    wp2_check_swift(check_date)

    # WP3
    # KP
    wp3_check_kp(check_date, "swpc")
    wp3_check_kp(check_date, "niemegk")
    wp3_check_kp(check_date, "swift")
    wp3_check_kp(check_date, "l1", model="KP-FULL-SW-PAGER")
    wp3_check_kp(check_date, "l1", model="HP60-FULL-SW-SWAMI-PAGER")

    # PLASMA
    wp3_check_plasma(check_date, "ca")
    wp3_check_plasma(check_date, "gfz_plasma")