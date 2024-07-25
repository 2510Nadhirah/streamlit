import openvino as ov
import cv2
import numpy as np
import matplotlib.pyplot as plt

core = ov.Core()

model_face = core.read_model(model='Model/face-detection-adas-0001.xml')
compiled_model_face = core.compile_model(model = model_face, device_name= "CPU")

input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)

model_ag = core.read_model(model='Model/age-gender-recognition-retail-0013.xml')
compiled_model_ag = core.compile_model(model = model_ag, device_name= "CPU")

input_layer_ag = compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output(0)

compiled_model_ag = core.compile_model(model = model_ag, device_name= "CPU")

input_layer_ag = compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output

def preprocess(image, input_layer):
    N,input_channels,input_height, input_width = input_layer.shape
    
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0 )
    
    return input_image

def find_faceboxes(image, results, confidence_threshold):
    results = results.squeeze()

    scores = results[:,2]
    boxes = results[:, -4:]

    face_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]

    image_h, image_w, image_channels = image.shape
    face_boxes = face_boxes*np.array([image_w, image_h, image_w, image_h])
    face_boxes = face_boxes.astype(np.int64)

    return face_boxes, scores

def draw_age_gender(face_boxes, image):

    show_image = image.copy()

    for i in range(len(face_boxes)):

        xmin, ymin, xmax, ymax = face_boxes[i]
        face = image[ymin:ymax, xmin:xmax]
        
       
        # --- age and gender ---
        input_image_ag = preprocess(face, input_layer_ag)
        results_ag = compiled_model_ag([input_image_ag])
        age, gender = results_ag[1],results_ag[0]
        age = np.squeeze(age)
        age = int(age*100)

        gender = np.squeeze(gender)

        if gender[0] >= 0.65:
            gender = "female"
            box_color = (200, 200, 0)
        elif gender[1] >= 0.55:
            gender = "male"
            box_color = (0, 200, 200)
        else:
            gender = "unknown"
            box_color = (200, 200, 200)
        
        if age < 30:
            box_color = (255, 0, 0)

         # --- age and gender ---

        fontScale = image.shape[1]/750
        
        text = gender + ' ' + str(age)
        cv2.putText(show_image, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,200,0), 8)
        cv2.rectangle(img=show_image, pt1=(xmin,ymin), pt2=(xmax,ymax), color=box_color, thickness=2)

    return show_image

def predict_image(image, conf_threshold):
    input_image = preprocess(image, input_layer_face)
    results = compiled_model_face([input_image])[output_layer_face]
    face_boxes, scores = find_faceboxes(image, results, conf_threshold)
    visualize_image = draw_age_gender(face_boxes, image)

    return visualize_image