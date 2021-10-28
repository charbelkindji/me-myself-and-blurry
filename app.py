from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from PIL import Image, ImageDraw
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
        blurred_path = blur_image(image, img_name)
        print("-***** PATHS")
        print(short_image_path)
        print(blurred_path)

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
    # Load in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

    # Cast array
    img = np.array(img)
    # img = Image.open(image_path)

    # Return processsed image
    return img

def blur_image(img, img_name, blur=True, showBoxes=True, showKeyPoints=True):
    """
    Blur image

    :param img: image to blur
    :param img_name: name of the image file
    :param blur: to apply blur or not
    :param showBoxes: to show boxes around faces or not
    :param showKeyPoints: to show the face keypoints or not
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

        if(showBoxes):
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

        if(showKeyPoints):
            # draw the dots
            for key, value in face['keypoints'].items():
                # create and draw dot
                dot = patches.Circle(value, radius=30, color='blue')
                ax.add_patch(dot)


        if(blur):

            # Convert Image to array
            img = np.array(img)

            roi = img[y:y+h, x:x+w]
            # applying a blur over this new rectangle area
            roi = cv2.blur(roi, (50,50))
            # impose this blurred image on original image to get final image
            img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

    # save result
    blurred_path = 'static/img/uploads/blurred/'+img_name
    plt.imsave(blurred_path, img)
    # plt.savefig(blurred_path)

    return re.sub("static/", "", blurred_path) # we don't want the "static" part in path since full path is generated in template

def upload_file():
    """
    Handle file upload after form submission
    """

    # Get the file from post request
    file = request.files['image']

    # Save the file to static/images/uploads
    basepath = os.path.dirname(__file__)
    full_file_path = os.path.join(
        basepath, 'static/img/uploads/', secure_filename(file.filename))

    file.save(full_file_path)
    short_path = 'img/uploads/' + secure_filename(file.filename)
    return full_file_path, short_path

if __name__ == '__main__':
    app.run(None, None, True)
