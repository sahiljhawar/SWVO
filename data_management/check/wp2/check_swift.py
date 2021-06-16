from data_management.check.base_file_checker import BaseFileCheck
from data_management.notifications.email_notifier import send_failure_email
from data_management.io.wp2.read_swift import SwiftReader

import datetime as dt
import glob
import logging
from email.headerregistry import Address


class SwiftCheck(BaseFileCheck):
    def __init__(self):
        super().__init__()
        self.file_folder = "/PAGER/WP2/data/outputs/SWIFT/"
        self.subject_email = "PAGER WP2, SWIFT Module, DATA FAILURE..."
        self.email_recipients = self._get_email_recipients()

    @staticmethod
    def _get_email_recipients():
        addresses = tuple()
        addresses += (Address("Ruggero Vasile", "ruggero", "gfz-potsdam.de"),)
        return addresses

    @staticmethod
    def _extract_date_from_folder(folder):
        folder = folder.split("/")[-1]
        folder = folder.split("t")[0]
        date = dt.datetime.strptime(folder, "%Y%m%d")
        return date

    def check_files_exists(self, check_date=None):
        if check_date is None:
            time_to_check = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            time_to_check = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
        folder_list = glob.glob(self.file_folder + "*")

        correct_folder = None
        for folder in folder_list:
            date = SwiftCheck._extract_date_from_folder(folder)
            if date == time_to_check:
                correct_folder = folder
                break

        success = False
        if correct_folder is not None:
            try:
                gsm_file = glob.glob(correct_folder + "/gsm*")[0]
                success = True
            except IndexError:
                logging.warning("GSM output file for swift not found ....")
                gsm_file = None
            try:
                hgc_file = glob.glob(correct_folder + "/hgc*")[0]

            except IndexError:
                logging.warning("HGC output file for swift not found ....")
                hgc_file = None
                success = False

            return success, gsm_file, hgc_file
        else:
            logging.warning("Folder for WP2 SWIFT current output not found...")
            return success, None, None

    def check_file_format(self, gsm_file, hgc_file):
        # TODO To implement this
        SwiftReader()
        success = True
        return success

    def run_check(self, date=None):
        """
        It runs a check about existence of given outputs of swift module. If the files are not found
        for a given date provided, it sends an email to a default list of users and notifies them
        of the problem. If date is not specified then the check is performed on the date of the day
        the script has been called.

        We will add also a check on the file format and content later on...

        :param date: Date on which to run the check.
        :type date: datetime.datetime
        """
        success, gsm_file, hgc_file = self.check_files_exists(date)
        if not success:
            logging.info("SWIFT GSM and HGC files for date {} not found...sending notification email".format(date))
            content = "Output files not generated yet today..."
            send_failure_email(subject=self.subject_email, content=content, addresses_to=self.email_recipients,
                               address_from=self.email_sender)
        if success:
            logging.info("SWIFT GSM and HGC files for date {} found!!".format(date))
