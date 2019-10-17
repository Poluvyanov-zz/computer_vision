from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(
	exec_path, "models/resnet50_coco_best_v2.0.1.h5")
)
detector.loadModel()

list = detector.detectObjectsFromImage(
	input_image=os.path.join(exec_path, "input/panda.jpeg"),
	output_image_path=os.path.join(exec_path, "output/new_objects.jpg"),
	minimum_percentage_probability=90,
	display_percentage_probability=True,
	display_object_name=True
)

for eachItem in list:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])