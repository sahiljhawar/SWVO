from flask import Flask

webapp = Flask(__name__, static_url_path='/PAGER/WP3/data/figures/')

from . import routes