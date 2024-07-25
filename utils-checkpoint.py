import winsound
import openvino as ov
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Windows-specific sound module

def draw_age_gender(face_boxes, image):
    show_image = image.copy()

    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        face = image[ymin:ymax, xmin:xmax]

        # --- age and gender ---
        input_image_ag = preprocess(face, input_layer_ag)
        results_ag = compiled_model_ag([input_image_ag])
        age, gender = results_ag[1], results_ag[0]
        age = np.squeeze(age)
        age = int(age * 100)

        gender = np.squeeze(gender)

        if gender[0] >= 0.65:
            gender_text = "female"
            box_color = (200, 200, 0)
        elif gender[1] >= 0.55:
            gender_text = "male"
            box_color = (0, 200, 200)
        else:
            gender_text = "unknown"
            box_color = (200, 200, 200)
        # --- age and gender ---

        if age >= 60:
            box_color = (0, 0, 255)  # Red color for 60+ age
            winsound.Beep(1000, 500)  # Beep sound for 500 ms

        fontScale = image.shape[1] / 750

        text = f"{gender_text} {age}"
        cv2.putText(show_image, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 200, 0), 8)
        cv2.rectangle(img=show_image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=box_color, thickness=2)

    return show_image

def predict_image(image, conf_threshold):
    input_image = preprocess(image, input_layer_face)
    results = compiled_model_face([input_image])[output_layer_face]
    face_boxes, scores = find_faceboxes(image, results, conf_threshold)
    visualize_image = draw_age_gender(face_boxes, image)

    return visualize_image