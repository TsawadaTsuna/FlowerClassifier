import numpy as np
import cv2 as cv
import glob
import pickle
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV



if os.path.exists("model.pkl"):
    svc = pickle.load(open('model.pkl','rb'))
else:
    datos = []
    #images = glob.glob('./marchitas/*.*')
    images = glob.glob('./DatasetPlanttas/marchitas/*.*')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        de = list(hog(gray))
        de.append("marchita")
        datos.append(de)

    #images = glob.glob('./normales/*.*')
    images = glob.glob('./DatasetPlanttas/normales/*.*')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        de = list(hog(gray))
        de.append("normal")
        datos.append(de)

    images = glob.glob('./DatasetPlanttas/Margaritas/Marchitas/*.*')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        de = list(hog(gray))
        de.append("marchita")
        datos.append(de)

    #images = glob.glob('./normales/*.*')
    images = glob.glob('./DatasetPlanttas/Margaritas/Normales/*.*')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        de = list(hog(gray))
        de.append("normal")
        datos.append(de)

    images = glob.glob('./DatasetPlanttas/Rosas/Marchitas/*.*')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        de = list(hog(gray))

        de.append("marchita")
        datos.append(de)

    #images = glob.glob('./normales/*.*')
    images = glob.glob('./DatasetPlanttas/Rosas/Normales/*.*')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        de = list(hog(gray))
        de.append("normal")
        datos.append(de)

    train, test=train_test_split(datos, test_size=0.20)
    trainx=[]
    trainy=[]
    testx=[]
    testy=[]
    for elem in train:
        trainy.append(elem.pop(-1))
        trainx.append(elem[0:30])
    for elem in test:
        testy.append(elem.pop(-1))
        testx.append(elem[0:30])  
    #print(testy)
    #svc = MLPClassifier(solver='lbfgs', alpha=1e-05,activation='logistic', hidden_layer_sizes=(10, 10), random_state=1, max_iter=1000)
    #svc = SVC(kernel='linear')
    svc = DecisionTreeClassifier()
    svc.fit(trainx,trainy)
    pickle.dump(svc, open('model.pkl', 'wb'))
    print(classification_report(testy,svc.predict(testx),labels=['marchita','normal']))
    #print(trainx[0])
    #print(de[0:30])]))


cap = cv.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    #print(des[0][0]+1)
    de = list(hog(gray))
    prediccion = svc.predict([de[0:30]])
    
    print(prediccion)
    cv.putText(frame, prediccion[0], (10,400), cv.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 4)
    cv.imshow('frame',frame)
# When everything done, release the capture

cap.release()

# finally, close the window
cv.destroyAllWindows()
cv.waitKey(1)

#print(trainx[0])
#print(de[0:30])