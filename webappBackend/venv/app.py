from flask import Flask


app = Flask(__name__)


@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/airplane")
def airplane():
    return 501


@app.route("/automobile")
def automobile():
    return 501


@app.route("/bird")
def bird():
    return 501


@app.route("/cat")
def cat():
    return 501


@app.route("/deer")
def deer():
    return 501

@app.route("/dog")
def dog():
    return 501


@app.route("/frog")
def frog():
    return 501


@app.route("/horse")
def horse():
    return 501


@app.route("/ship")
def ship():
    return 501


@app.route("/truck")
def truck():
    return 501


