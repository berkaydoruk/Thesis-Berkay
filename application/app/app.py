from flask import Flask
from routes.home import home_bp
from routes.sales_prediction import salesPredict_bp
app = Flask(__name__)

app.secret_key = 'key'

app.register_blueprint(home_bp)
app.register_blueprint(salesPredict_bp)

if __name__ == '__main__':
    app.run(debug=True)