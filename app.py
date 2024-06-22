from cgitb import html
import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect
from sklearn.neighbors import KNeighborsClassifier
from datetime import date, datetime
import joblib
import face_recognition
import cv2
from collections import defaultdict
# from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
# app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'DATABASE_URL'
db = SQLAlchemy(app)


class Student(db.Model):
    __tablename__ = 'students'

    student_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    roll_number = db.Column(db.Integer, nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey(
        'classes.class_id'), nullable=False)

    # Define a relationship to the classes table
    class_info = db.relationship('Class', backref='students')

    def __init__(self, name, roll_number, class_id):
        self.name = name
        self.roll_number = roll_number
        self.class_id = class_id


class Class(db.Model):
    __tablename__ = 'classes'

    class_id = db.Column(db.Integer, primary_key=True)
    class_name = db.Column(db.String(255), nullable=False)

    def __init__(self, class_id, class_name):
        self.class_id = class_id
        self.class_name = class_name


class Attendance(db.Model):
    __tablename__ = 'attendance'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(10), nullable=False)
    date = db.Column(db.Date, nullable=False)
    class_id = db.Column(db.Integer, nullable=False)

    def __init__(self, name, roll, status, date, class_id):
        self.name = name
        self.roll = roll
        self.status = status
        self.date = date
        self.class_id = class_id


class Cumulative(db.Model):
    __tablename__ = 'cumulative_attendance'

    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    roll_number = db.Column(db.Integer, nullable=False)
    total_classes = db.Column(db.Integer, nullable=False)
    classes_attended = db.Column(db.Integer, nullable=False)
    class_id = db.Column(db.Integer, nullable=False)

    def __init__(self, student_name, roll_number, class_id, total_classes=0, classes_attended=0):
        self.student_name = student_name
        self.roll_number = roll_number
        self.class_id = class_id
        self.total_classes = total_classes
        self.classes_attended = classes_attended


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
# datetoday = '09_11_23'  # Replace with your desired date format

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        if img.shape != (0, 0, 0):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = face_detector.detectMultiScale(gray, 1.3, 5)
            return face_points
        else:
            return []
    except:
        return []


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


def train_model():
    encodings = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img_path = f'static/faces/{user}/{imgname}'
            img = face_recognition.load_image_file(img_path)
            face_encoding = face_recognition.face_encodings(img)
            if len(face_encoding) > 0:
                encodings.append(face_encoding[0])
                labels.append(user)

    model = {
        'encodings': encodings,
        'labels': labels
    }

    # Save the model using joblib
    joblib.dump(model, 'static/face_recognition_model.pkl')


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    encodings = model['encodings']
    labels = model['labels']

    face_to_identify = face_recognition.face_encodings(facearray)

    distances = face_recognition.face_distance(encodings, face_to_identify)
    min_distance_index = np.argmin(distances)
    identified_person = labels[min_distance_index]

    return identified_person


@app.route('/')
def home():
    return render_template('dummy.html')


@app.route('/home')
def homee():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/manual_attendance', methods=['GET', 'POST'])
def manual_attendance_selection():
    if request.method == 'POST':

        selected_class_id = request.form.get('class_id')
        # Query the database to fetch students by the selected class_id
        students = Student.query.filter_by(class_id=selected_class_id).all()
        return render_template('mark_manual_attendance.html', students=students, selected_class_id=selected_class_id)

    # Fetch all unique class_ids from the database
    unique_class_ids = db.session.query(Student.class_id).distinct().all()

    # Convert class_ids from a list of tuples to a list of strings
    class_ids = [class_id[0] for class_id in unique_class_ids]

    return render_template('manual_attendance_selection.html', class_ids=class_ids)


@app.route('/mark_manual_attendance', methods=['POST'])
def mark_manual_attendance():
    if request.method == 'POST':
        selected_class_id = request.form.get('class_id')
        students = Student.query.filter_by(class_id=selected_class_id).all()

        # Create or update attendance records in the attendance table
        date_today = datetime.today().date()
        for student in students:
            roll_number = student.roll_number
            status = 'Present' if str(roll_number) in request.form.getlist(
                'attendance[]') else 'Absent'
            attendance_record = Attendance(
                name=student.name,
                roll=roll_number,
                status=status,
                date=date_today,
                class_id=selected_class_id
            )
            db.session.add(attendance_record)

        db.session.commit()

        # Update or create cumulative attendance records in the cumulative table
        for student in students:
            roll_number = student.roll_number

            # Calculate total classes and classes attended for the student
            attendance_records = Attendance.query.filter_by(
                roll=roll_number, class_id=selected_class_id).all()
            total_classes = len(attendance_records)
            classes_attended = sum(
                1 for record in attendance_records if record.status == 'Present')

        # Find or create the cumulative record for the student
            cumulative_record = Cumulative.query.filter_by(
                roll_number=roll_number, class_id=selected_class_id).first()
        
            if cumulative_record:
                # Update existing cumulative record
                cumulative_record.total_classes = total_classes
                cumulative_record.classes_attended = classes_attended
            else:
                # Create a new cumulative record
                cumulative_record = Cumulative(
                    student_name=student.name,
                    roll_number=roll_number,
                    class_id=selected_class_id,
                    total_classes=total_classes,
                    classes_attended=classes_attended
                )
                db.session.add(cumulative_record)
        db.session.commit()
# Commit changes to the database

        return redirect('/manual_attendance')


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return render_template('home.html', message='No image uploaded.')

    image = request.files['image']
    if image.filename == '':
        return render_template('home.html', message='No selected image.')

    image_path = os.path.join('static', 'uploaded_image.jpg')
    image.save(image_path)

    try:
        rgb_image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(
            rgb_image, face_locations)
    except Exception as e:
        return render_template('home.html', message=f"Error: {e}")

    model = joblib.load('static/face_recognition_model.pkl')

    recognized_faces = []

    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(
            model['encodings'], face_encoding)
        min_distance_index = np.argmin(face_distances)

        if face_distances[min_distance_index] < 0.5:
            recognized_faces.append(model['labels'][min_distance_index])
            add_attendance(model['labels'][min_distance_index])
        else:
            recognized_faces.append("Unrecognized")

    for (top, right, bottom, left), face in zip(face_locations, recognized_faces):
        cv2.rectangle(rgb_image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(rgb_image, face, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Face Recognition', rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("successful")

    return redirect('/home')

@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    if request.method == 'POST':
        selected_year = request.form.get('year')

        # Query the database to fetch cumulative attendance records for the selected year
        cumulative_records = Cumulative.query.filter_by(class_id=selected_year).all()

        return render_template('analytics.html', cumulative_records=cumulative_records)

    # Fetch all unique class_ids (years) from the database
    unique_class_ids = db.session.query(Cumulative.class_id).distinct().all()

    # Convert class_ids from a list of tuples to a list of strings
    years = [str(class_id[0]) for class_id in unique_class_ids]

    return render_template('select_year.html', years=years)



@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == 100:
            break
        cv2.imshow('Adding new student', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


if __name__ == '__main__':
    app.run(debug=True)
