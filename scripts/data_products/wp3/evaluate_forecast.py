import os
import argparse
import datetime as dt
import logging
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec


from data_management.io.wp3.read_kp import KPReader
from data_management.plotting.wp3.kp.plot_kp import PlotKpOutput


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default="2021110400", type=str,
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
            product_date = dt.datetime.strptime(args.date, "%Y%m%d%H")
        except TypeError:
            msg = "Provided date {} not in correct format %Y%m%d%H. Aborting...".format(args.date)
            logging.error(msg)
            raise RuntimeError(msg)

    if args.logdir is not None:
        log_file = "wp3_evaluate_kp_l1_{}.log".format(product_date.strftime("%Y%m%dT%H%M%S"))
        logging.basicConfig(filename=os.path.join(args.logdir, log_file), level=logging.INFO,
                            datefmt="%Y-%m-%d %H:%M:%S",
                            format="%(asctime)s;%(levelname)s;%(message)s")

    RESULTS_PATH = args.output

    reader = KPReader(args.input)
    data_l1, _ = reader.read(source="l1", requested_date=product_date, model_name="KP-FULL-SW-PAGER")
    data_niemegk, _ = reader.read(source="niemegk", requested_date=product_date)

    plotter = PlotKpOutput()

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])

    gs.update(left=0.1, right=0.9, wspace=0.05, hspace=0.1)



    plotter.plot_output(data_l1, ax1)
    plotter.plot_output(data_niemegk, ax2)

    plt.savefig("prova.png")


