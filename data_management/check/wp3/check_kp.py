from data_management.check.base_file_checker import BaseFileCheck
from data_management.notifications.email_notifier import send_failure_email

import datetime as dt
import glob
import logging
from email.headerregistry import Address


class KpDataCheck(BaseFileCheck):
    def __init__(self):
        super().__init__()
        self.file_folder = "/PAGER/WP3/data/outputs/"
        self.subject_email = "PAGER WP3, KP Module, DATA FAILURE..."
        self.email_recipients = self._get_email_recipients()

    @staticmethod
    def _get_email_recipients():
        addresses = tuple()
        addresses += (Address("Ruggero Vasile", "ruggero", "gfz-potsdam.de"),)
        return addresses

    def _check_swpc_file_exists(self, check_date=None):
        if check_date is None:
            check_date = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            check_date = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
        check_date_str = check_date.strftime("%Y%m%d")
        try:
            file = glob.glob(self.file_folder + "/SWPC/SWPC_KP_FORECAST_{}.csv".format(check_date_str))[0]
            success = True
            logging.info("SWPC output Kp for date {} found!!".format(check_date))
        except IndexError:
            file = None
            success = False
            logging.warning("SWPC KP forecast for date {} not found ...".format(check_date_str))
        return success, file

    def _check_niemegk_file_exists(self, check_date=None):
        if check_date is None:
            check_date = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            check_date = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
        check_date_str = check_date.strftime("%Y%m%d")
        try:
            file = glob.glob(self.file_folder + "/NIEMEGK/NIEMEGK_KP_NOWCAST_{}.csv".format(check_date_str))[0]
            success = True
            logging.info("NIEMEGK output Kp for date {} found!!".format(check_date))
        except IndexError:
            file = None
            success = False
            logging.warning("NIEMEGK KP nowcast for date {} not found ...".format(check_date_str))
        return success, file

    def _check_swift_file_exists(self, check_date=None):
        if check_date is None:
            check_date = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        else:
            check_date = check_date.replace(minute=0, second=0, microsecond=0)
        check_date_str = check_date.strftime("%Y-%m-%d_%H:%M:%S")
        try:
            file = glob.glob(self.file_folder + "/SWIFT/FORECAST_PAGER_SWIFT_swift_{}.csv".format(check_date_str))[0]
            success = True
            logging.info("SWIFT output Kp for date {} found!!".format(check_date))
        except IndexError:
            file = None
            success = False
            logging.warning("SWIFT Kp forecast for date {} not found ...".format(check_date_str))
        return success, file

    def _check_l1_file_exists(self, model, spc, check_date=None):
        if check_date is None:
            check_date = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        else:
            check_date = check_date.replace(minute=0, second=0, microsecond=0)
        check_date_str = check_date.strftime("%Y-%m-%d_%H:%M:%S")
        try:
            file = glob.glob(self.file_folder + "/L1_FORECAST/FORECAST_{}_{}_{}.csv".format(model,
                                                                                            spc, check_date_str))[0]
            success = True
            logging.info("L1 Kp forecast for date {},"
                         " spacecraft source {} and model {} found!!".format(check_date_str, spc, model))
        except IndexError:
            file = None
            success = False
            logging.warning("L1 Kp forecast for date {},"
                            " spacecraft source {} and model {} not found ...".format(check_date_str, spc, model))
        return success, file

    def run_check(self, product, model=None, date=None):
        """
        It runs a check about existence of given outputs of WP3 Kp modules. If the files are not found
        for a given date provided, it sends an email to a default list of users and notifies them
        of the problem. If date is not specified then the check is performed on the date of the day
        the script has been called.

        We will add also a check on the file format and content later on...

        :param product: The Kp product to check, choose among "l1", "swpc", "niemegk", "swift"
        :type product: str
        :param model: Used at the moment for product L1. It is the name of the model to check for outputs.
        :type model: str
        :param date: Date on which to run the check.
        :type date: datetime.datetime
        """
        if product == "swpc":
            success, file, = self._check_swpc_file_exists(date)
        elif product == "niemegk":
            success, file, = self._check_niemegk_file_exists(date)
        elif product == "swift":
            success, file, = self._check_swift_file_exists(date)
        elif product == "l1":
            if model is None:
                msg = "Kp product checker for l1 output needs a model name to proceed"
                logging.error(msg)
                raise RuntimeError(msg)
            success, file, = self._check_l1_file_exists(model, "dscovr_rt", date)
        else:
            msg = "Kp product checker product {} not among listed outputs..."
            logging.error(msg)
            raise RuntimeError(msg)

        if not success:
            content = "Output files for Kp WP3 module {} not generated yet today...".format(product)
            logging.info(content)
            logging.info("Sending email notification...")
            send_failure_email(subject=self.subject_email, content=content, addresses_to=self.email_recipients,
                               address_from=self.email_sender)
