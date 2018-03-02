# coding=UTF-8

import numpy as np
import cv2
import glob
import os
from os.path import basename, dirname, abspath

frontal_face_cascade = cv2.CascadeClassifier('classifier_xml/haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier('classifier_xml/haarcascade_profileface.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化  
    ## cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)  
    return cv_img 

# mark face and output face images
def mark_face(img, face_cascade=frontal_face_cascade, output_filename='face', output_path='output', filename_prefix=''):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	# exit function. If no face in images
	if len(faces) == 0:
		return

	# create output folde
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	face_img = img.copy();
	face_index = 0;
	output_filename = filename_prefix + '_' + output_filename
	full_path = output_path + '/' + output_filename

	# start mark face
	for (x,y,w,h) in faces:
		str_index = str(face_index)
		# mark face and set label
		cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.putText(face_img,'face_'+str_index,(x,y+h+25), font, 1,(255,255,255),2,cv2.LINE_AA)
		roi_color = img[y:y+h, x:x+w]
		roi_gray = gray[y:y+h, x:x+w]
		# save face image
		cv2.imencode('.jpg', roi_color)[1].tofile(full_path+'_face_'+str_index+'.jpg')
		face_index += 1
	
	# output face marked image
	cv2.imencode('.jpg', face_img)[1].tofile(full_path+'.jpg') 
	return

# resize image and use different classifier
def detect_face(img_path, output_path):
	print(img_path)
	img = cv_imread(img_path)

	img_half = cv2.resize(img,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
	mark_face(img_half, face_cascade=frontal_face_cascade, output_filename='img_0.5', output_path=output_path, filename_prefix='frontal')
	mark_face(img_half, face_cascade=profile_face_cascade, output_filename='img_0.5', output_path=output_path, filename_prefix='profile')
	img_quarter = cv2.resize(img,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
	mark_face(img_quarter, face_cascade=frontal_face_cascade, output_filename='img_0.25', output_path=output_path, filename_prefix='frontal')
	mark_face(img_quarter, face_cascade=profile_face_cascade, output_filename='img_0.25', output_path=output_path, filename_prefix='profile')
	return;

if __name__ == '__main__':
	# list all file
	images = glob.glob("data/**/*.JPG");
	for img_file in images:
		img_filename = basename(img_file)
		img_path = abspath(img_file)
		face_img_folder = img_path+'_face'

		# start detect face
		detect_face(img_path, face_img_folder)
