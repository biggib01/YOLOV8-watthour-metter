from ultralytics import YOLO
import cv2
import json
from qreader import QReader


# def perform_yolo_ocr_on_area(image_path, confidence_threshold=0.6):
#     # Load Yolov8 model
#
#     # Load the image
#     img = cv2.imread(image_path)
#
#     # Perform text detection using the YOLO model
#     results = model(img)
#
#     # Initialize variables to store detections and warnings
#     detections = []
#     warnings = []
#
#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()
#         confidences = result.boxes.conf.cpu().numpy()
#         class_ids = result.boxes.cls.cpu().numpy().astype(int)
#         labels = [model.names[class_id] for class_id in class_ids]
#
#         for i, (box, label, confidence) in enumerate(zip(boxes, labels, confidences)):
#             if confidence < confidence_threshold:
#                 warnings.append({i+1})
#
#             detection = {
#                 'index': i,
#                 'class': label,
#                 'confidence': float(confidence)
#             }
#             detections.append(detection)
#
#     # Combine results into a final output
#     if warnings:
#         return {
#             'detections': detections,
#             'warnings': warnings
#         }
#     elif not detections:
#         return "No detections found."
#     else:
#         return {
#             'detections': detections,
#             'warnings': []  # No warnings if all confidences are acceptable
#         }


def predict_numbers(image_path, model_ver=0):
    # Predict using the model
    if model_ver == 0:
        model = YOLO('best.pt')
    elif model_ver == 1:
        model = YOLO('best_qr.pt')
    else:
        pass

    results = model(image_path)

    # Parse results
    predictions = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())  # Class of detected object (e.g., number)
            conf = float(box.conf.item())  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates

            # Append prediction details
            predictions.append({
                "class": cls,
                "confidence": conf,
                "coordinates": [x1, y1, x2, y2]  # Change tuple to list for JSON compatibility
            })

    # Sort predictions by the x1 coordinate (the least x value)
    predictions.sort(key=lambda x: x["coordinates"][0])

    # read QR code
    # Create a QReader instance
    qreader = QReader()

    # Get the image that contains the QR code
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Use the detect_and_decode function to get the decoded QR data
    decoded_text = qreader.detect_and_decode(image=image)

    #print(decoded_text[0])

    # Initialize warning list for low confidence detections
    warning_list = []

    # Check confidence and add to warning list if below 0.6
    for i, prediction in enumerate(predictions):
        if prediction["confidence"] < 0.6:
            warning_list.append(i + 1)

    # Create the final output dictionary
    output = {
        "predictions": predictions,
        "warnings": warning_list,
        "room_number": decoded_text[0]
    }

    # Return the JSON-formatted string
    return json.dumps(output, indent=4)


# def predict_numbers(image_path):
#     # Predict using the model
#     results = model(image_path)
#
#     # Parse results
#     predictions = []
#     for result in results:
#         for box in result.boxes:
#             cls = int(box.cls.item())  # Class of detected object (e.g., number)
#             conf = float(box.conf.item())  # Confidence score
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
#
#             # Append prediction details
#             predictions.append({
#                 "class": cls,
#                 "confidence": conf,
#                 "coordinates": (x1, y1, x2, y2)
#             })
#
#     # Sort predictions by the x1 coordinate (the least x value)
#     predictions.sort(key=lambda x: x["coordinates"][0])
#
#     # Initialize warning list for low confidence detections
#     warning_list = []
#
#     # Check confidence and add to warning list if below 0.6
#     for i, prediction in enumerate(predictions):
#         if prediction["confidence"] < 0.6:
#             warning_list.append(i+1)
#
#     return predictions, warning_list
