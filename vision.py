import pandas as pd
import firebase_admin

from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud import vision

cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred, {
	'projectId': 'foodi-266917',
	})

db = firestore.client()

client = vision.ImageAnnotatorClient()
image_path = '/home/pi/Monitor/lastsnap.jpg'

with open(image_path, 'rb') as image_file:
	content = image_file.read()
	
image = vision.types.Image(content=content)
response = client.object_localization(image=image)
localized_object_annotations = response.localized_object_annotations

df = pd.DataFrame(columns=['name','score'])
for obj in localized_object_annotations:
	df = df.append(
		dict(
			name = obj.name, score = obj.score
		),
		ignore_index=True)
	doc_ref = db.collection(u'detections').document(u'{}'.format(str(obj.name)))
	doc_ref.set({
		u'score': u'{}'.format(str(obj.score))
	})

print(df)


