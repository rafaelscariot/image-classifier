from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model('../resources/model.model')

cap = cv2.VideoCapture(0)

while True:
	has_frame, frame = cap.read()

	if not has_frame:
		break

	#(h, w) = frame.shape[:2]

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224))
	frame = img_to_array(frame)
	frame = preprocess_input(frame)
	frame = np.expand_dims(frame, axis=0)

	(dog, cat) = model.predict(frame)[0]

	confidence = max(cat, dog) * 100

	if confidence < 90:
		print('low confidence')
	else:
		label = 'Class2' if cat > dog else 'Class1'
		label = '{}: {:.2f}%'.format(label, confidence)

		print(label)