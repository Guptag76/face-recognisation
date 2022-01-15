import numpy as np
import cv2
import sklearn
import pickle
import  os 
from django.conf import settings
STATIC_DIR = settings.STATIC_DIR

face_detector_model=cv2.dnn.readNetFromCaffe(os.path.join(STATIC_DIR,"models/deploy.prototxt.txt"),
                                             os.path.join(STATIC_DIR,"models/res10_300x300_ssd_iter_140000.caffemodel"))

feature_extraction_model=cv2.dnn.readNetFromTorch(os.path.join(STATIC_DIR,"models/openface.nn4.small2.v1.t7"))

face_recognise_model=pickle.load(open(os.path.join(STATIC_DIR,"models/my_train_model.pkl"),mode='rb'))

emotion_recognise_model=pickle.load(open(os.path.join(STATIC_DIR,"models/my_train_model_for_emotion.pkl"),mode='rb'))



def pipeline_model(path):
    img=cv2.imread(path)
    image=img.copy()
    h,w=image.shape[:2]
    img_blob=cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123),swapRB=False,crop=False)
    face_detector_model.setInput(img_blob)
    detections=face_detector_model.forward()

    machine_learning_results=dict(confidence=[],
                                  face_name=[],
                                  face_score=[],
                                  emotion=[],
                                  emotion_score=[],
                                  count=[])
    count=1
    
    
    if len(detections)>0:
        for i in range(0,detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>0.5:
                box=detections[0,0,i,3:7]*np.array([w,h,w,h])
                x1,y1,x2,y2=box.astype(int)
                cv2.rectangle(image,(x1,y1),(x2,y2),(100,200,0),1)
                image_roi=image[y1:y2,x1:x2].copy()
                face_blob=cv2.dnn.blobFromImage(image_roi,1/255,(96,96),(0,0,0),swapRB=True,crop=True)
                feature_extraction_model.setInput(face_blob)
                vectors=feature_extraction_model.forward()
                
                rec_face=face_recognise_model.predict(vectors)[0]
                prob_face=face_recognise_model.predict_proba(vectors).max()
                #print(rec_face,prob_face)
                text="{} : {}%".format(rec_face,int(prob_face*100))
                cv2.putText(image,text,(x1,y1),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,250),1)
                
                emotion=emotion_recognise_model.predict(vectors)[0]
                prob_emotion=emotion_recognise_model.predict_proba(vectors).max()
                #print(emotion,prob_emotion)
                cv2.putText(image,emotion,(x1,y2),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,250),1)
                
                
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/upload_img.jpg'),image)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/roi_{}.jpg').format(count),image_roi)
                
                
                machine_learning_results["confidence"].append(confidence)
                machine_learning_results["face_name"].append(rec_face)
                machine_learning_results["face_score"].append(prob_face)
                machine_learning_results["emotion"].append(emotion)
                machine_learning_results["emotion_score"].append(prob_emotion)
                machine_learning_results["count"].append(count)
                count+=1
    return machine_learning_results
                


