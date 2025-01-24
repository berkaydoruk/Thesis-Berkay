from flask import Blueprint, render_template

home_bp = Blueprint('home', __name__) 

@home_bp.route('/')
def home():
    return render_template('home.html')

@home_bp.route('/navigation')
def navigation():
    return render_template('navigation.html')