from . import webapp
from flask import render_template
import pandas as pd


@webapp.route('/')
@webapp.route('/index')
def index():
    return render_template('index.html')
