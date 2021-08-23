from . import webapp
from flask import render_template
import pandas as pd
import glob
import datetime as dt


def get_latest_kp():
    PATH = "/PAGER/WP3/data/figures/Niemegk*.png"
    files = glob.glob(PATH)
    file = sorted(files)[-1]
    return file


@webapp.route('/')
@webapp.route('/index')
def index():
    niemegk_last = get_latest_kp()
    return render_template('index.html', niemegk=niemegk_last)
