from asyncio.windows_events import NULL
from email.mime import image
import os
import secrets
from turtle import title
from unittest import result
from PIL import Image
from flask import Flask, render_template, url_for, flash, redirect
from form import PictureForm

app = Flask(__name__)

app.config['SECRET_KEY'] = '7d0fb494ff22bc777a2384efae5670f5b6c41fafb4738df9cd4f50741d15dd78'

print(app.root_path.removesuffix(r"server") + r"model\dataset")

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static\\model\\test_data', picture_fn)
    # os.path.join(app.root_path, 'static/images', picture_fn)

    # output_size = (125, 125)
    i = Image.open(form_picture)
    # i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = PictureForm()
    image_file = url_for('static', filename='images/default.png')
    
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            os.chdir('static\\model')
            os.system('python face_recognizer.py ' + picture_file)
            os.chdir('..\\..')
            image_file = url_for('static', filename='model/result_image/' + picture_file)
    return render_template('home.html', image_file=image_file, title='Home', form=form)

@app.route('/about')
def about():
    return render_template('about.html', title='About')


if __name__ == '__main__':
    app.run(host='localhost', port='8888', debug=True)