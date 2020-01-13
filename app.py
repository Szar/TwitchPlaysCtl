from flask_api import FlaskAPI, status, exceptions
from flask_api.decorators import set_renderers
import os, sys

print(os.path.dirname(os.path.realpath(__file__)))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

app = FlaskAPI(__name__)

if (sys.version_info > (3, 0)): app.config['DEFAULT_RENDERERS'] = ['flask_api.renderers.JSONRenderer']
else: app.config['DEFAULT_RENDERERS'] = [ 'flask.ext.api.renderers.JSONRenderer', ]

@app.route("/", methods=['GET', 'POST'])
def main():
    return ["DO_SOMETHING_BY_DEFAULT"]

@app.route("/url_1", methods=['GET', 'POST'])
def url_1():
    return ["url_1"]

@app.route("/url_2", methods=['GET', 'POST'])
def url_2():
    return ["url_2"]


@app.route("/url_3/<int:pid>/", methods=['GET', 'POST'])
#@set_renderers([renderers.JSONRenderer()])
def url_3(pid):
    return ["url_3 "+str(pid)]



if __name__ == "__main__":
    app.run(host= '0.0.0.0', port=3100, debug=True)