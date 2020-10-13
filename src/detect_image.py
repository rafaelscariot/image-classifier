from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model('../resources/model.model')

image = cv2.imread('../images_test/test.jpg')
#(h, w) = image.shape[:2]

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = img_to_array(image)
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)

(class1, class2) = model.predict(image)[0]

confidence = max(cat, dog) * 100

if confidence < 90:
	print('low confidence')
else:
	label = 'Class1' if cat > dog else 'Class2'
	label = '{}: {:.2f}%'.format(label, confidence)

	print(label)