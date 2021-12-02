from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import re
import PIL
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    """
    Display the home page and display the processed image result
    """

    if request.method == 'GET':
        print("GET")
        return render_template('index.html.twig')
    else:
        print("POST")
        # Upload image
        full_image_path, short_image_path = upload_file()

        # Preprocess image
        image = preprocess_image(full_image_path)
        img_name = full_image_path.split('/')[-1]
        print(img_name)

        # Blur image
        print(image.shape)
        blurred_path = blur_image(image, img_name, request)

        # Display result
        return render_template('index.html.twig',
                               image_path = short_image_path,
                               blurred_path = blurred_path)

def preprocess_image(image_path):
    """
    Preprocess image before feeding the model for blurring
    :param image_path: path to the image file (uploaded file)
    :return: preprocessed image ready to be fed to the model
    """
    # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

    # Cast array
    img = np.array(img)

    return img

def blur_image(img, img_name, request):
    """
    Blur image

    :param img: image to blur
    :param img_name: name of the image file
    :param request: the request
    :return: path to the processed image result
    """

    face_detector = MTCNN()
    faces = face_detector.detect_faces(img)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    for face in faces:
        x,y,w,h = face['box']

        if(request.form.get("blurFaces") is not None):
            print("blurFaces process")
            roi = img[y:y+h, x:x+w]
            # applying a blur over this new rectangle area
            roi = cv2.blur(roi, (50,50))
            # impose this blurred image on original image to get final image
            img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

        if(request.form.get("showBoxes") is not None):
            print("showBoxes process")
            # Create a Rectangle around the face
            img = cv2.rectangle(img,
                                (x,y),
                                (x+w,y+h),
                                (0,128,0),
                                10)

        if(request.form.get("showKeypoints") is not None):
            print("showKeypoints process")
            # draw the dots for key points
            for key, value in face['keypoints'].items():
                radius = 20
                img = cv2.circle(img, value, radius, (0,0,139), -1) # thickness -1 to fill the circle



    # save result
    blurred_path = 'static/img/uploads/blurred/'+img_name
    plt.imsave(blurred_path, img[...,::-1]) # img[...,::-1] to keepp matplotlib from changing image colors
    # plt.savefig(blurred_path)

    return re.sub("static/", "", blurred_path) # we don't want the "static" part in path since full path is generated in template

def upload_file():
    """
    Handle file upload after form submission
    """

    # Get the file from post request
    file = request.files['image']

    # Resize uploaded image
    fixed_height = 512
    image = Image.open(file)
    # print(image.size)
    height_percent = (fixed_height / float(image.size[1]))
    width_size = int((float(image.size[0]) * float(height_percent)))
    image = image.resize((width_size, fixed_height), PIL.Image.NEAREST)

    # Save the file to static/images/uploads
    full_file_path = os.path.join('static/img/uploads/', secure_filename(file.filename))
    image.save(full_file_path)
    
    short_path = 'img/uploads/' + secure_filename(file.filename)
    return full_file_path, short_path


@app.route('/about', methods=['GET'])
def about():
    """
    Display about page
    :return:
    """
    return render_template("about.html.twig")

if __name__ == '__main__':
    app.run(None, None, True)
