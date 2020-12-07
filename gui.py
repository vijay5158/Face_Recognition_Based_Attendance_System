import tkinter as tk
from tkinter import Message, Text, messagebox
import cv2, os

from tkinter import *
import numpy as np
from PIL import Image
from PIL import ImageTk
import pandas as pd
import datetime
import time

import tkinter.font as font


window = tk.Tk()
window.title("Face Recognition Based Attendance System")
window.configure(background='#fcf9ae')



image_path = ImageTk.PhotoImage(Image.open("ngfcet.jpg"))

background_label = Label( window,image = image_path, height=768,width=1366)
background_label.pack()


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

label1 = tk.Label(window, text="Enter Name",width=20  ,bg="#fcf9ae"    ,height=2 ,font=('times', 15, ' bold ')) 
label1.place(x=400, y=50)

student_name = tk.Entry(window,width=20  ,bg="#fcf9ae",font=('times', 15, ' bold ')  )
student_name.place(x=700, y=50)

label2 = tk.Label(window, text="Enter Roll No",width=20  ,height=2 ,bg="#fcf9ae" ,font=('times', 15, ' bold ') ) 
label2.place(x=400, y=150)

rollno = tk.Entry(window,width=20  ,bg="#fcf9ae",font=('times', 15, ' bold '))
rollno.place(x=700, y=150)

label3 = tk.Label(window, text="Enter Your Email",width=20  ,height=2 ,bg="#fcf9ae" ,font=('times', 15, ' bold ') ) 
label3.place(x=400, y=250)

email = tk.Entry(window,width=20  ,bg="#fcf9ae",font=('times', 15, ' bold '))
email.place(x=700, y=250)

label4 = tk.Label(window, text="Status : ",width=20  ,bg="#fcf9ae"  ,height=2 ,font=('times', 15, ' bold underline ')) 
label4.place(x=400, y=410)


message1 = tk.Label(window, text="" ,bg="#fcf9ae"   ,width=30  ,height=3, font=('times', 15, ' bold ')) 
message1.place(x=700, y=400)

label5 = tk.Label(window, text="Attendance : ",width=20  ,bg="#fcf9ae"  ,height=2 ,font=('times', 15, ' bold  underline')) 
label5.place(x=400, y=640)

message2 = tk.Label(window, text=""    ,bg="#fcf9ae",width=35  ,height=4  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=600)

def register_students():
    name = student_name.get()
    roll_no = rollno.get()
    student_email = email.get()

    if is_number(roll_no) and name.isalpha():
        cap = cv2.VideoCapture(0)
        haarcascade_path = 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haarcascade_path)
        num_img = 0

        while True:
            frames, img = cap.read()            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
            for (x1, y1, x2, y2) in faces:                
                cv2.rectangle(img, (x1,y1),(x1+x2,y1+y2), (255,255,255),2)                
                num_img += 1
                path = 'images'
                cv2.imwrite(os.path.join(path,name+ "." +roll_no+ '.' +str(num_img) +".jpg") , gray[y1:y1+y2, x1:x1+x2])
                cv2.imshow('frame',img)
        
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
            elif num_img>60:
                break
        
        cap.release()
        cv2.destroyAllWindows()

        msg = "Your images are saved \nRoll No : " +roll_no + " \nName : "+ name
        line = [roll_no, name, student_email]
        
        column_names = ['Roll No', 'Name', 'Email']
        students = pd.DataFrame(columns=column_names)
        
        students.loc[len(students)] = line

        student_detail_path = 'students_details.csv'
        
        if os.path.exists(student_detail_path):
            students.to_csv('students_details.csv',mode = 'a', header=False, index=False)
    
        else:
            students.to_csv('students_details.csv',index=False)
        
        df_students = pd.read_csv('students_details.csv')    
        df_students = df_students.drop_duplicates(subset=['Roll No'])    
        
        df_students.to_csv('students_details.csv',index = False)        
        message1.configure(text=msg)
    else:
        msg = "Enter Alphabetical Name"
        message1.configure(text=msg)




def clear_name():
    student_name.delete(0, 'end')    
    msg = ""
    message1.configure(text= msg) 

def clear_rollno():
    rollno.delete(0, 'end')    
    msg = ""
    message1.configure(text= msg)

def clear_email():
    email.delete(0, 'end')    
    msg = ""
    message1.configure(text= msg)  

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass 
    return False

def train_facial_data():
    face_recognizer= cv2.face.LBPHFaceRecognizer_create()
    path = 'images'
    image_paths=[os.path.join(path,i) for i in os.listdir(path)]
    
    faces=[]
    Rollno=[]
    
    for image_path in image_paths:
        pil_image=Image.open(image_path).convert('L')
        image_np=np.array(pil_image,'uint8')
        
        rollno=int((os.path.split(image_path)[-1].split(".")[1]))
        faces.append(image_np)
        Rollno.append(rollno)        
    
    
    
    face_recognizer.train(faces, np.array(Rollno))
    path = 'imagesLabels'
    
    face_recognizer.save(os.path.join(path,"label_model.yml"))
    msg = "Your facial data have been Trained."
    message1.configure(text= msg)


def attendance_marking():
    message2.configure(text='')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    path = 'imagesLabels'
    model_path = os.path.join(path,"label_model.yml")
    face_recognizer.read(model_path)

    haarcascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haarcascade_path);    
    
    df=pd.read_csv("students_details.csv")
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    
    col_names =  ['Roll No','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        frames, img =cap.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray, 1.2,5)    
    
        for(x1,y1,x2,y2) in faces:
            cv2.rectangle(img,(x1,y1),(x1+x2,y1+y2),(225,0,0),2)
            Roll_no, conf = face_recognizer.predict(gray[y1:y1+y2,x1:x1+x2])                                   
    
            if(conf < 40):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                name=df.loc[df['Roll No'] == Roll_no]['Name'].values
                detail=str(Roll_no)+"-"+name
                attendance.loc[len(attendance)] = [Roll_no,name,date,timeStamp]

            else:
                detail = str('Unknown Face')                     
                
            cv2.putText(img,str(detail),(x1,y1+y2), font, 1,(0,0,0),2)        
    
        attendance = attendance.drop_duplicates(subset=['Roll No'],keep='first')    
        ts = time.time()      
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')    
    
        path="attendance"
        attendance_path = os.path.join(path,"attendance_" + date +".csv")
    
        if os.path.exists(attendance_path):
            attendance.to_csv(attendance_path,mode = 'a', header=False, index=False)
    
        else:
            attendance.to_csv(attendance_path,index = False)   
    
        df_attendance = pd.read_csv(attendance_path)  
    
        df_attendance = df_attendance.drop_duplicates(subset=['Roll No'],keep='first')    
        df_attendance.to_csv(attendance_path,index = False)
    
        cv2.imshow('img',img) 
    
        if (cv2.waitKey(1)==ord('q')):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if detail == 'Unknown Face':
        msg = 'You are not registered.'
    
    else:
        msg='Attendance marked for \n '+str(attendance.loc[0][:2])
    message2.configure(text= msg)

clear_button2 = tk.Button(window, text="Clear Name", command=clear_name  ,bg="#fcf9ae"  ,width=20  ,height=2,
                            font=('times', 15, ' bold '))
clear_button2.place(x=950, y=45)

clear_button1 = tk.Button(window, text="Clear Roll No", command=clear_rollno,bg="#fcf9ae"  ,
                            width=20  ,height=2 ,font=('times', 15, ' bold '))
clear_button1.place(x=950, y=145)

clear_button2 = tk.Button(window, text="Clear Email", command=clear_email  ,bg="#fcf9ae"  ,width=20  ,
                            height=2,font=('times', 15, ' bold '))
clear_button2.place(x=950, y=245)    
register_button = tk.Button(window, text="Register Yourself \nOr \nUpdate Your Images", command=register_students  ,
                            bg="#fcf9ae"  ,width=20  ,height=3 ,font=('times', 15, ' bold '))
register_button.place(x=110, y=500)
train_button = tk.Button(window, text="Train Your Facial Data", command=train_facial_data  ,bg="#fcf9ae"  ,width=20  ,
                            height=3 ,font=('times', 15, ' bold '))
train_button.place(x=410, y=500)
attendance_button = tk.Button(window, text="Mark Attendance", command=attendance_marking  ,bg="#fcf9ae"  ,width=20  ,
                            height=3 ,font=('times', 15, ' bold '))
attendance_button.place(x=710, y=500)
quit_button = tk.Button(window, text="Quit", command=window.destroy  ,bg="#fcf9ae"  ,width=20  ,
                            height=3,font=('times', 15, ' bold '))
quit_button.place(x=1010, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=2,font=('times', 15, 'italic bold underline'))
copyWrite.insert("insert", "Developed by VIJAY")
copyWrite.pack(side="left")
copyWrite.place(x=1145, y=675)
window.mainloop()

