import numpy as np
import cv2
import glob
import os
from os.path import basename, dirname, abspath

frontal_face_cascade = cv2.CascadeClassifier('haar_data/haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier('haar_data/haarcascade_profileface.xml')

def cv_imread(filePath):  
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化  
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)  
    return cv_img 

def mark_face(img, output_filename='face', output_path='output'):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = frontal_face_cascade.detectMultiScale(gray, 1.3, 5)
	face_img = img;
	for (x,y,w,h) in faces:
		face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
	
	# output face image
	cv2.imencode('.jpg', face_img)[1].tofile(output_path+'/'+output_filename+'.jpg') 
	return


# list all file
# print(glob.glob("pics/**/*.jpg"))
images = glob.glob("pics/**/*.jpg");


for img_file in images:
	img_filename = basename(img_file)
	img_path = abspath(img_file)
	face_img_folder = img_path+'_face'
	
	# create folde for each image
	if not os.path.exists(face_img_folder):
		os.makedirs(face_img_folder)

	# start detect face
	img = cv_imread(img_path)

	img_half = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	mark_face(img_half, 'img_0.5', face_img_folder)
	img_half = cv2.resize(img,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
	mark_face(img_half, 'img_0.25', face_img_folder)
	img_half = cv2.resize(img,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)
	mark_face(img_half, 'img_0.1', face_img_folder)