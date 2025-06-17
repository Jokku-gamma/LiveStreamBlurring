import eventlet
eventlet.monkey_patch()

import cv2
import numpy as np
from flask import Flask,render_template,request
from flask_socketio import SocketIO,emit
import base64
from PIL import Image
import io

app=Flask(__name__)
socketio=SocketIO(app,cors_allowed_origins='*',async_mode='eventlet',max_http_buffer_size=100 * 1024 * 1024)
facenet=cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)

plate_cascade=cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
blur_settings={
    'face_blur':True,
    'plate_blur':True,
    'blur_intensity':100
}


def apply_blur(frame):
    plate_det_fr=frame.copy()
    h,w=frame.shape[:2]

    if blur_settings['face_blur']:
        blob=cv2.dnn.blobFromImage(
            cv2.resize(frame,(500,500)),
            1.0,
            (500,500),
            (104.0,177.0,123.0)
        )
        facenet.setInput(blob)
        dets=facenet.forward()

        for i in range(dets.shape[2]):
            conf=dets[0,0,i,2]
            if conf>0.5:
                box=dets[0,0,i,3:7] * np.array([w,h,w,h])
                (startX,startY,endX,endY)=box.astype("int")

                startX,startY=max(0,startX),max(0,startY)
                endX,endY=min(w,endX),min(h,endY)
                if endY>startY and endX>startX:
                    face_roi=frame[startY:endY,startX:endX]
                    k=blur_settings['blur_intensity'] //2 * 2 +1
                    frame[startY:endY,startX:endX]=cv2.GaussianBlur(
                        face_roi,(k,k),0
                    )
    if blur_settings['plate_blur']:
        gray=cv2.cvtColor(plate_det_fr,cv2.COLOR_BGR2GRAY)
        plates=plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
        )
        for (x,y,w,h) in plates:
            plate_roi=frame[y:y+h,x:x+w]
            k=blur_settings['blur_intensity'] // 2 * 2 +1
            frame[y:y+h,x:x+w]=cv2.GaussianBlur(plate_roi,(k,k),0)
    return frame

@socketio.on('video_frame')
def handle_video(data):
    img_data=base64.b64decode(data.split(',')[1])
    nparr=np.frombuffer(img_data,np.uint8)
    frame=cv2.imdecode(nparr,cv2.IMREAD_COLOR)

    proc_frame=apply_blur(frame)
    ret,buff=cv2.imencode('.jpg',proc_frame)
    proc_base64=base64.b64encode(buff).decode('utf-8')

    emit('processed_frame','data:image/jpeg:base64' +proc_base64)


@app.route('/update_settings',methods=['POST'])
def update_settings():
    global blur_settings
    data=request.json
    blur_settings['face_blur']=data.get('face_blur',True)
    blur_settings['plate_blur']=data.get('plate_blur',True)
    blur_settings['blur_intensity']=int(data.get('blur_intensity',100))

    socketio.emit('settings_updated',blur_settings)
    return '',204

@app.route('/')
def index():
    return render_template('index.html',settings=blur_settings)

if __name__=="__main__":
    socketio.run(app,host='0.0.0.0',port=5000,debug=True)


