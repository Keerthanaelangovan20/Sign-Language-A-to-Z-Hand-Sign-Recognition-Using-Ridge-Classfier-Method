import sqlite3  
from flask import Flask, render_template, Response, request,jsonify
import cv2
import numpy as np
import mediapipe as mp 
import pickle
import pandas as pd
import time
import nltk
import time
import speech_recognition as sr
import pyttsx3
import os
from werkzeug.utils import secure_filename
database="finals.db"
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
from gtts import gTTS
import os
import pyttsx3
from playsound import playsound
nltk.download('punkt_tab')



def createtable():
    conn=sqlite3.connect(database)
    cursor=conn.cursor()
    cursor.execute("create table if not exists register (id integer primary key autoincrement, name text, mail text, password text)")   
    conn.commit()
    conn.close()
createtable()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
porter = nltk.stem.PorterStemmer()
wnl = nltk.stem.WordNetLemmatizer()

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
holistic = mp_holistic.Holistic()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

with open('hand_sign.pkl', 'rb') as f:
    model = pickle.load(f)

detected_letters = []
current_letter_duration = 0
min_continuous_duration = 30
@app.route('/')
@app.route('/register',methods=["GET","POST"])
def register():
    if request.method=="POST":
         name=request.form['name']         
         mail=request.form['mail']         
         password=request.form['password']
         confirm_pass= request.form['confirm_pass']
         if password != confirm_pass:
            return " sorry your password does not match"
         con=sqlite3.connect(database)
         cur=con.cursor()
         cur.execute("SELECT mail FROM register WHERE mail=?", (mail,))
         registered = cur.fetchall()
         if registered:
            return " your mailid already registered"
         else:   
             cur.execute("insert into register(name, mail, password)values(?,?,?)",(name, mail, password))
             con.commit()
             return render_template('login.html')
    return render_template('register.html')


@app.route('/login',methods = ["GET","POST"])
def login():
    if request.method=="POST":
        mail=request.form['mail']
        password=request.form['password']
        con=sqlite3.connect(database)
        cur=con.cursor()
        cur.execute("select * from register where mail=? and password=?",(mail,password))
        data=cur.fetchone()
        if data is None:
                return "failed"        
        else:  
            con=sqlite3.connect(database)
            cur=con.cursor()
            cur.execute("select *from register where mail=?",(mail,))
            results = cur.fetchone()            
            con.commit()
            title = 'Crop Recommendation'
            return render_template('index.html', title=title)
    return render_template('login.html')



@app.route('/index')
def index():
    return render_template('index.html')



english=[]
detected_letters = []
current_keyword = ""
whole_word_text = ""



@app.route('/generate_frames', methods=['GET', 'POST'])
def generate_frames():
    global detected_letters, last_detected_letter, whole_word_text, body_language_class
    global body_language_class, tamil, english, whole_word_text_data

    global current_keyword

    predicted_values = []
  
    last_detected_letter = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return render_template("index.html")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            

            if not ret:
                print("Error: Could not read frame from the camera.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                       )

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                       )

            body_language_class = "None"
            body_language_prob = None
        

            if results.left_hand_landmarks:
                    pose = results.left_hand_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    row = pose_row

                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    print(body_language_class)
                
                     
           
          
            key = cv2.waitKey(5)
            if key == 105:
                        
                        if body_language_class != "None":
                                detected_letters.append(body_language_class)
                                last_detected_letter = body_language_class.split(' ')[0]
                                whole_word_text = "".join(detected_letters)                                
                                                             
                               
                        
            elif key == 99:  
                            detected_letters += " "
                            

            elif key == ord('p'):
                            print("audio")

                            if detected_letters:
                                speak = gTTS(text=whole_word_text)
                                speak.save("show.mp3")
                                playsound('show.mp3')
                                os.remove('show.mp3')
            elif  cv2.waitKey(10) & 0xFF == ord('q'):
                break

            else:
                        current_keyword += body_language_class.split(' ')[0]

         
            last_letter_text = f'Letter: {last_detected_letter}'

            cv2.putText(image, whole_word_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, last_letter_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            cv2.imshow("Detect Sign",image)
           

    cap.release()
    cv2.destroyAllWindows()
    return render_template("home.html")

##@app.route('/play_english', methods=['GET', 'POST'])
##def play_english():
##    #for i in english:
##        speak = gTTS(text=whole_word_text)
##        speak.save("show.mp3")
##        playsound('show.mp3')
##        os.remove('show.mp3')
##        return render_template("home.html")


LANG_CODES = {
    "english": "en",
    "tamil": "ta",
    "hindi": "hi",
    "malayalam": "ml",     # Telugu language added
    "kannada": "kn"
}

def generate_speech(text, lang_code):
    print(f"Original Text: {text}")
    print(f"Target Language Code: {lang_code}")

    translator = Translator()
    
    # Translate text to the target language
    translated_text = translator.translate(text, dest=lang_code).text
    print(f"Translated Text: {translated_text}")

    try:
        # Convert text to speech
        tts = gTTS(text=translated_text, lang=lang_code)
        filename = "show.mp3"
        tts.save(filename)

        # Play the speech
        playsound(filename)

        # Remove file after playing
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print(f"File not found: {filename}")

    except Exception as e:
        print(f"Error generating speech: {e}")

    return filename  # Returning filename for better handling

@app.route('/play/<lang>', methods=['GET', 'POST'])
def play_voice(lang):
    global whole_word_text

    if lang in LANG_CODES:
        print(f"Requested Language: {lang}")
        generate_speech(whole_word_text, LANG_CODES[lang])
        return "Speech generated successfully!", 200
 
##    else:
##        return "Language not supported!", 400

@app.route('/delete', methods=['GET', 'POST'])
def delete():
    global whole_word_text, last_detected_letter
    
    detected_letters.clear()
    whole_word_text = ""
    last_detected_letter = ""
    return render_template("home.html")

@app.route('/get_data')
def get_data():
    try:
        return jsonify({'data': whole_word_text})
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/sign', methods=['GET','POST'])
def sign():
    return render_template('sign.html')
processed=[]
tokens_sign_lan=[]
@app.route('/text', methods=['GET','POST'])
def text(text):
        processed.clear()
        tokens_sign_lan.clear()
        stop = nltk.corpus.stopwords.words('english')
        stop_words=['@','#',"http",":","is","the","are","am","a","it","was","were","an",",",".","?","!",";","/"]
        for i in stop_words:
            stop.append(i)
        tokenized_text = nltk.tokenize.word_tokenize(text)
        lemmed = [wnl.lemmatize(word) for word in tokenized_text]
        
        for i in lemmed :
            if i == "i" or i == "I":
                processed.append("me")
            elif i not in stop:
                i=i.lower()
                processed.append((i))
                print(text)
        assets_list=['0.mp4', '1.mp4', '2.mp4', '3.mp4', '4.mp4', '5.mp4','6.mp4', '7.mp4', '8.mp4', '9.mp4', 'a.mp4', 'after.mp4',
             'again.mp4', 'against.mp4', 'age.mp4', 'all.mp4', 'alone.mp4','also.mp4', 'and.mp4', 'ask.mp4', 'at.mp4', 'b.mp4', 'be.mp4',
             'beautiful.mp4', 'before.mp4', 'best.mp4', 'better.mp4', 'busy.mp4', 'but.mp4', 'bye.mp4', 'c.mp4', 'can.mp4', 'cannot.mp4',
             'change.mp4', 'college.mp4', 'come.mp4', 'computer.mp4', 'd.mp4', 'day.mp4', 'distance.mp4', 'do not.mp4', 'do.mp4', 'does not.mp4',
             'e.mp4', 'eat.mp4', 'engineer.mp4', 'f.mp4', 'fight.mp4', 'finish.mp4', 'from.mp4', 'g.mp4', 'glitter.mp4', 'go.mp4', 'god.mp4',
             'gold.mp4', 'good.mp4', 'great.mp4', 'h.mp4', 'hand.mp4', 'hands.mp4', 'happy.mp4', 'hello.mp4', 'help.mp4', 'her.mp4', 'here.mp4',
             'his.mp4', 'home.mp4', 'homepage.mp4', 'how.mp4', 'i.mp4', 'invent.mp4', 'it.mp4', 'j.mp4', 'k.mp4', 'keep.mp4', 'l.mp4', 'language.mp4', 'laugh.mp4',
             'learn.mp4', 'm.mp4', 'me.mp4', 'mic3.png', 'more.mp4', 'my.mp4', 'n.mp4', 'name.mp4', 'next.mp4', 'not.mp4', 'now.mp4', 'o.mp4', 'of.mp4', 'on.mp4',
             'our.mp4', 'out.mp4', 'p.mp4', 'pretty.mp4', 'q.mp4', 'r.mp4', 'right.mp4', 's.mp4', 'sad.mp4', 'safe.mp4', 'see.mp4', 'self.mp4', 'sign.mp4', 'sing.mp4', 
             'so.mp4', 'sound.mp4', 'stay.mp4', 'study.mp4', 't.mp4', 'talk.mp4', 'television.mp4', 'thank you.mp4', 'thank.mp4', 'that.mp4', 'they.mp4', 'this.mp4', 'those.mp4', 
             'time.mp4', 'to.mp4', 'type.mp4', 'u.mp4', 'us.mp4', 'v.mp4', 'w.mp4', 'walk.mp4', 'wash.mp4', 'way.mp4', 'we.mp4', 'welcome.mp4', 'what.mp4', 'when.mp4', 'where.mp4', 
             'which.mp4', 'who.mp4', 'whole.mp4', 'whose.mp4', 'why.mp4', 'will.mp4', 'with.mp4', 'without.mp4', 'words.mp4', 'work.mp4', 'world.mp4', 'wrong.mp4', 'x.mp4', 'y.mp4',
             'you.mp4', 'your.mp4', 'yourself.mp4', 'z.mp4']
       
        for word in processed:
            string = str(word+".mp4")
            if string in assets_list:
                tokens_sign_lan.append(str("assets/"+string))
            else:
                for j in word:
                    tokens_sign_lan.append(str("assets/"+j+".mp4"))
        for i in tokens_sign_lan:
            label = i.replace("assets/","").replace(".mp4","")
            cap = cv2.VideoCapture(i)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
            fps= int(cap.get(cv2.CAP_PROP_FPS))
            if cap.isOpened() == False:
                print("Error File Not Found")
            while cap.isOpened():
                ret,frame= cap.read()
                if ret == True:
                    time.sleep(1/fps)
                    #print('work')
                    cv2.putText(frame, label,(10,60),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 1, cv2.LINE_AA)
                    out.write(frame)
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    data = jpeg.tobytes()
                    yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n')    
                    #print('not work')
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        #return ' work'
    #return render_template('sign.html')
        
r = sr.Recognizer()
mic = sr.Microphone()



@app.route('/video_feed_endpoint', methods=['POST'])
def video_feed_endpoint():
    submitted_text = request.form.get('text', '')
    print(type(submitted_text))
    return Response(text(submitted_text), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/home', methods=['GET','POST'])
def home():
    return render_template('home.html')



def speak1():
    #global text
    with mic as audio_file:
        print("Speak Now...")
        r.adjust_for_ambient_noise(audio_file)
        audio = r.listen(audio_file)
        print("Converting Speech to Text...")
        text= r.recognize_google(audio)
        text=text.lower()
        print("Input:",text)
        return text

@app.route('/speak', methods=['GET','POST'])
def speak():
    print("start")
    speech=speak1()
    print(type(speech))
    return Response(text(speech), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False,port=690)
