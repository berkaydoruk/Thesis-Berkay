from flask import Flask
from routes.home import home_bp
from routes.sales_prediction import salesPredict_bp
from routes.customer_seg import customerSeg_bp
from routes.returns import returns_bp

app = Flask(__name__)

app.secret_key = 'key'

app.register_blueprint(home_bp)
app.register_blueprint(salesPredict_bp)
app.register_blueprint(customerSeg_bp)
app.register_blueprint(returns_bp)

if __name__ == '__main__':
    app.run(debug=True)