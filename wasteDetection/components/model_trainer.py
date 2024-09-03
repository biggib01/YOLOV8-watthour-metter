import os,sys
import yaml
from wasteDetection.utils.main_utils import read_yaml_file
from wasteDetection.logger import logging
from wasteDetection.exception import AppException
from wasteDetection.entity.config_entity import ModelTrainerConfig
from wasteDetection.entity.artifacts_entity import ModelTrainerArtifact
from ultralytics import YOLO



class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config


    

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            os.system("unzip data.zip")
            os.system("rm data.zip")

            # with open("data.yaml", 'r') as stream:
            #     num_classes = str(yaml.safe_load(stream)['nc'])
            #
            # model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            # print(model_config_file_name)
            #
            #
            # # train on custom config
            # config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")
            #
            # config['nc'] = int(num_classes)
            #
            #
            # with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
            #     yaml.dump(config, f)

            print("1")
            # try to change to train yolov8

            # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
            # # Train the model with 2 GPUs
            # results = model.train(data='data.yaml', epochs=25, imgsz=640,
            #                       device='mps', name='yolov8_results')

            # yolo task=detect mode=train model=yolov8s.pt data=/content/datasets/data.yaml epochs=1 imgsz=640 plots=True
            # yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=640 plots=True
            # os.system(f"cd yolov5/ && yolo task=detect mode=train model=yolov8s.pt data=../data.yaml epochs={self.model_trainer_config.no_epochs} imgsz=640 device=mps plots=True name=yolov5s_results")
            os.system(f"python test.py")
            print("2")
            os.system("cp runs/detect/yolov8_results/weights/best.pt yolov5/")
            print("3")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            print("4")
            os.system(f"cp runs/detect/yolov8_results/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")
            #
            os.system("rm -rf runs")
            os.system("rm -rf train")
            os.system("rm -rf valid")
            os.system("rm -rf test")
            os.system("rm -rf data.yaml")
            os.system("rm -rf README.dataset.txt")
            os.system("rm -rf README.roboflow.txt")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise AppException(e, sys)

