from data_management.check.base_file_checker import BaseFileCheck
from data_management.notifications.email_notifier import send_failure_email

import datetime as dt
import glob
import logging
from email.headerregistry import Address


class PlasmaDataCheck(BaseFileCheck):
    def __init__(self):
        super().__init__()
        self.file_folder = "/PAGER/WP3/data/outputs/"
        self.subject_email = "PAGER WP3, Plasmasphere Module, DATA FAILURE..."
        self.email_recipients = self._get_email_recipients()

    @staticmethod
    def _get_email_recipients():
        addresses = tuple()
        addresses += (Address("Ruggero Vasile", "ruggero", "gfz-potsdam.de"),)
        addresses += (Address("Stefano Bianco", "bianco", "gfz-potsdam.de"),)
        return addresses

    def _check_ca_file_exists(self, check_date=None):
        if check_date is None:
            check_date = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            check_date = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
        check_date_str = check_date.strftime("%Y-%m-%d")
        try:
            file = glob.glob(self.file_folder + "/CA/plasmapause_{}.csv".format(check_date_str))[0]
            success = True
            logging.warning("Carpenter Anderson Plasmapause for date {} found!!".format(check_date_str))
        except IndexError:
            file = None
            success = False
            logging.warning("Carpenter Anderson Plasmapause for date {} not found ...".format(check_date_str))
        return success, file

    def _check_plasma_file_exists(self, check_date=None):
        if check_date is None:
            check_date = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            check_date = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
        check_date_str = check_date.strftime("%Y-%m-%d-%H-%M")
        try:
            file = glob.glob(self.file_folder + "/GFZ_PLASMA/plasmasphere_density_{}.csv".format(check_date_str))[0]
            success = True
            logging.warning("Plasma density for date {} found!!".format(check_date_str))
        except IndexError:
            file = None
            success = False
            logging.warning("Plasma density for date {} not found ...".format(check_date_str))
        return success, file

    def run_check(self, product, date=None):
        """
        It runs a check about existence of given outputs of WP3 Plasma modules. If the files are not found
        for a given date provided, it sends an email to a default list of users and notifies them
        of the problem. If date is not specified then the check is performed on the date of the day
        the script has been called.

        We will add also a check on the file format and content later on...
        :param product: The plasma density product to check, choose among "ca", "gfz_plasma"
        :type product: str
        :param date: Date on which to run the check.
        :type date: datetime.datetime
        """
        if product == "ca":
            success, file, = self._check_ca_file_exists(date)
        elif product == "gfz_plasma":
            success, file, = self._check_plasma_file_exists(date)
        else:
            msg = "Kp product checker product {} not among listed outputs..."
            logging.error(msg)
            raise RuntimeError(msg)

        if not success:
            content = "Output files for Plasma WP3 module {} not generated yet today...".format(product)
            logging.info(content)
            logging.info("Sending email notification...")
            send_failure_email(subject=self.subject_email, content=content, addresses_to=self.email_recipients,
                               address_from=self.email_sender)
