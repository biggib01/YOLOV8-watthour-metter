import sys,os
from wasteDetection.pipeline.training_pipeline import TrainPipeline
from wasteDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from wasteDetection.constant.application import APP_HOST, APP_PORT
from function import predict_numbers
import requests


app = Flask(__name__)
CORS(app)

# Define the path where the image will be saved
SAVE_PATH = os.path.join(os.getcwd(), "data")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!" 


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    # Get the URL from the request (assume it's passed as JSON)
    data = request.json
    blob_url = data.get('url')

    if not blob_url:
        return jsonify({"error": "URL is required"}), 400

    try:
        # Download the image from the URL
        response = requests.get(blob_url)
        response.raise_for_status()  # Raise an error for bad responses

        # Save the image with the name "inputImage"
        image_path = os.path.join(SAVE_PATH, "inputImage.jpg")

        with open(image_path, 'wb') as f:
            f.write(response.content)

        # Call the OCR prediction function
        ocr_result = predict_numbers(image_path)

        # Clean up by removing unnecessary directories or files
        os.system("rm -rf yolov5/runs")

        # Return the OCR result in JSON format
        return jsonify(ocr_result), 200

    except requests.exceptions.RequestException as req_err:
        print(req_err)
        return jsonify({"error": "Failed to download the image", "details": str(req_err)}), 500
    except ValueError as val_err:
        print(val_err)
        return jsonify({"error": "Value error", "details": str(val_err)}), 400
    except KeyError as key_err:
        print(key_err)
        return jsonify({"error": "Key error", "details": str(key_err)}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500




# @app.route("/live", methods=['GET'])
# @cross_origin()
# def predictLive():
#     try:
#         os.system("cd yolov5/ && python detect.py --weights my_model.pt --img 416 --conf 0.5 --source 0")
#         os.system("rm -rf yolov5/runs")
#         return "Camera starting!!"
#
#     except ValueError as val:
#         print(val)
#         return Response("Value not found inside  json data")
#



if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)

