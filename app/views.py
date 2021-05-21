from flask import request, render_template, redirect, url_for, flash, jsonify, session, Response
from werkzeug.utils import secure_filename
import json
import cv2
import numpy as np
import os
from app import app

import app.CONFIG as CONFIG
from app.utils.register import Capture_Images, register_capture_images, process_existing_images
from app.utils.train import train, show_dataset
from app.utils.inference import Infernce, inference_webcam
from app.utils.delete import delete_person


@app.route('/')
def index():
    return render_template('index.html', status=session)

@app.route('/register_capture_images', methods=['GET', 'POST'])
def register_capture_images_():
    if request.method == 'POST':
        name = request.form['known_face_name']
        cam = Capture_Images(name, training_images=CONFIG.TRAINING_IMAGES, gpu=CONFIG.GPU)
        return Response(register_capture_images(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_existing_images', methods=['GET', 'POST'])
def register_existing_images_():
    if request.method == 'POST':
        images = []
        # image = request.files['known_face_img']
        files = request.files.getlist("known_face_img")
        if len(files) < CONFIG.TRAINING_IMAGES:
            session['status'] = 'failed'
            flash(f"# of images required: {CONFIG.TRAINING_IMAGES} but selected only {len(files)}")
            return redirect(url_for('admin'))

        for file in files:
            # change the name of the image before saving into the specified directory
            file.save(os.path.join(CONFIG.TEMP_FILES_PATH, secure_filename(file.filename)))

        session['status'] = 'warning'
        name = request.form['known_face_name']
        flash(f"adding person {name}. Please be patient...")
        status = process_existing_images(name, training_images=CONFIG.TRAINING_IMAGES, gpu=CONFIG.GPU)
        if status["status"]:
            session['status'] = 'success'
            flash(status["message"])
        else:
            session['status'] = 'failed'
            flash(status["message"])
    return redirect(url_for('admin'))

@app.route('/train_s', methods=['GET', 'POST'])
def train_classifier():
    if request.method == 'POST':
        session['status'] = 'warning'
        flash(f"training classifier. This may take a few minutes, please be patient...")
        status = train()
        if status["status"]:
            session['status'] = 'success'
            flash(status["message"])
        else:
            session['status'] = 'failed'
            flash(status["message"])
    return redirect(url_for('admin'))

@app.route('/inference_webcam', methods=["GET", "POST"])
def _inference_webcam():
    if request.method == 'POST':
        cam = Infernce(threshold=CONFIG.CONFIDENCE_THRESHOLD, resize_scale=CONFIG.RESIZE_SCALE, gpu=CONFIG.GPU)
        return Response(inference_webcam(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/delete_person', methods=["GET", "POST"])
def delete():
    if request.method == 'POST':
        name = request.form['del_face_name']
        status = delete_person(name)
        if status["status"]:
            session['status'] = 'success'
            flash(status["message"])
        else:
            session['status'] = 'failed'
            flash(status["message"])

    return redirect(url_for('admin'))
    
@app.route('/registered_people', methods=['POST', 'GET'])
def registered_people():
    data = show_dataset()
    return jsonify(data)


@app.route('/admin', methods=['POST', 'GET'])
def admin():
    return render_template('admin.html', status=session)

@app.route('/user', methods=['POST', 'GET'])
def user():
    return render_template('user.html', status=session)