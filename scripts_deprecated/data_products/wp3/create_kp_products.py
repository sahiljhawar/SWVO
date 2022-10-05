import os
import argparse
import datetime as dt
import logging

from data_management.formats.wp3.kp_formats import RawFormat
from data_management.io.wp3.read_kp import KPReader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY%mm%dd%HH")
    parser.add_argument('-output', action="store", default="/PAGER/WP3/data/products/", type=str,
                        help="Path to a folder where to store the produced figures")
    parser.add_argument('-input', action="store", default="/PAGER/WP3/data/outputs/", type=str,
                        help="Path to a folder where the data to plot is stored...(be more precise)")
    parser.add_argument('-logdir', action="store", default=None, type=str,
                        help="Log directory if logging is to be enabled.")

    args = parser.parse_args()

    date_now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if args.date is None:
        product_date = date_now
    else:
        try:
            product_date = dt.datetime.strptime(args.date, "%Y-%m-%d")
        except TypeError:
            msg = "Provided date {} not in correct format %Y-%m-%d. Aborting...".format(args.date)
            logging.error(msg)
            raise RuntimeError(msg)

    if args.logdir is not None:
        log_file = "wp3_kp_products_{}.log".format(product_date.strftime("%Y%m%dT%H%M%S"))
        logging.basicConfig(filename=os.path.join(args.logdir, log_file), level=logging.INFO,
                            datefmt="%Y-%m-%d %H:%M:%S",
                            format="%(asctime)s;%(levelname)s;%(message)s")

    RESULTS_PATH = args.output

    reader = KPReader(args.input)
    for model in ["KP-FULL-SW-PAGER", "HP60-FULL-SW-SWAMI-PAGER"]:
        data, _ = reader.read(source="l1", requested_date=product_date, model_name=model)

        wdc_path = os.path.join(args.output, "FORECAST_{}_{}_{}.wdc".format(model, "dscovr_rt",
                                                                            product_date.strftime("%Y%m%dT%H%M%S")))
        omni_path = os.path.join(args.output, "FORECAST_{}_{}_{}.dat".format(model, "dscovr_rt",
                                                                             product_date.strftime("%Y%m%dT%H%M%S")))
        RawFormat.wdc(data, wdc_path)
        RawFormat.omniweb(data, omni_path)
        wdc_path = os.path.join(args.output, "FORECAST_{}_{}_LAST.wdc".format(model, "dscovr_rt"))
        omni_path = os.path.join(args.output, "FORECAST_{}_{}_LAST.dat".format(model, "dscovr_rt"))
        RawFormat.wdc(data, wdc_path)
        RawFormat.omniweb(data, omni_path)
