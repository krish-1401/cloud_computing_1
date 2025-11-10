import cv2
from deepface import DeepFace
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3
import time

class EmotionGreeter:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.tts_engine = pyttsx3.init()
        self.vid = cv2.VideoCapture(0)
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        self.emotion_label = tk.Label(window, text="Detecting emotion...",
                                      font=("Arial", 16))
        self.emotion_label.pack()
        self.greeting_label = tk.Label(window, text="", font=("Arial", 16), fg="blue")
        self.greeting_label.pack()

        self.current_emotion = "Waiting..."
        self.delay = 15  
        self.last_detection_time = 0
        self.detection_interval = 3 

        self.update_video()
        self.btn_quit = tk.Button(window, text="Quit", width=10, command=self.quit_app)
        self.btn_quit.pack()

    def update_video(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            current_time = time.time()
            if current_time - self.last_detection_time > self.detection_interval:
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    self.current_emotion = result[0]['dominant_emotion']
                    self.display_greeting(self.current_emotion)
                except Exception:
                    self.current_emotion = "Detection Error"

                self.last_detection_time = current_time
            self.emotion_label.config(text=f"Emotion: {self.current_emotion}")
        self.window.after(self.delay, self.update_video)

    def display_greeting(self, emotion):
        greetings = {
            "happy": "You look happy today!",
            "sad": "Don't worry, things will get better.",
            "angry": "Take a deep breath. Calm down.",
            "surprise": "Wow! You seem surprised.",
            "neutral": "You look calm and neutral.",
            "fear": "You look a bit scared, everything’s okay!",
            "disgust": "Hmm, something smells bad?"
        }

        greeting = greetings.get(emotion.lower(), "Hello there!")
        self.greeting_label.config(text=greeting)
        self.speak_greeting(greeting)

    def speak_greeting(self, greeting):
        self.tts_engine.say(greeting)
        self.tts_engine.runAndWait()

    def quit_app(self):
        self.vid.release()
        self.window.destroy()

root = tk.Tk()
app = EmotionGreeter(root, "Emotion-Based Greeting")
root.mainloop()