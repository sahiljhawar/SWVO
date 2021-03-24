import os


class RawFormat(object):
    def __init__(self):
        self.data = None

    @staticmethod
    def csv(data, filename, index, header=True):
        if os.path.exists(filename):
            os.remove(filename)
        f = open(filename, 'a')
        if header:
            f.write('# Most recent {} index forecast'.format(index))
            f.write('\n')
            f.write('# Produced by GFZ Section 2.7')
            f.write('\n\n')
        data["forecast"] = data["forecast"].apply(RawFormat._reformat_index)
        data.to_csv(f, header=True, index=True)
        if header:
            f.write('\n')
            f.write('# For more information contact Dr. Ruggero Vasile (ruggero@gfz-potsdam.de)')

    @staticmethod
    def omniweb(data, filename):
        if os.path.exists(filename):
            os.remove(filename)
        data["year"] = data.index.map(lambda x: x.year)
        data["doy"] = data.index.map(lambda x: x.timetuple().tm_yday)
        data["hour"] = data.index.map(lambda x: x.hour)
        forecast = data["forecast"].apply(RawFormat._reformat_index)
        data.drop(["forecast"], 1, inplace=True)
        data["forecast"] = forecast
        to_drop = [k for k in data if k not in ["year","doy","hour","forecast"]]
        data.drop(to_drop, 1, inplace=True)
        data.to_csv(filename, header=False, index=False, sep=" ")

    # TODO To finish
    @staticmethod
    def wdc(data, filename, forecast_date=None):
        if os.path.exists(filename):
            os.remove(filename)
        if forecast_date is None:
            forecast_date = data.index[0]
        forecast = data["forecast"].apply(RawFormat._reformat_index_plusminus)
        values = "".join(list(forecast.values))
        f = open(filename, 'a')
        f.write(forecast_date.strftime("%Y%m%d%H"))
        f.write(' ')
        f.write(values)
        f.close()

    @staticmethod
    def _reformat_index(x):
        if abs(x - int(x)) < 0.00001:
            return int(x)
        elif x - int(x) < 0.4:
            return float(int(x) + 0.3)
        else:
            return float(int(x) + 0.7)

    @staticmethod
    def _reformat_index_plusminus(x):
        if abs(x - int(x)) < 0.00001:
            return str(int(x))
        elif x - int(x) < 0.4:
            return str(int(x)) + "+"
        else:
            return str(int(x) + 1) + "-"


if __name__ == "__main__":
    input_file = "/home/ruggero/repositories/data_management/data/outputs/L1_FORECAST/FORECAST_KP-FULL-SW-PAGER_dscovr_rt_2021-03-20_01:00:00.csv"
    import pandas as pd
    import datetime as dt
    data = pd.read_csv(input_file, index_col=0)
    data.index = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in data.index]
    data["forecast"] = data["kp"]
    RawFormat.wdc(data,"prova.wdc")
    RawFormat.omniweb(data,"prova.dat")
