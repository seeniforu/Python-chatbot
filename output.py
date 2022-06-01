import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from textblob import TextBlob
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
import pyttsx3
engine=pyttsx3.init('sapi5')
voice=engine.getProperty("voices")
engine.setProperty('voice',voice[1].id)
engine.setProperty('rate',150)
import speech_recognition as sr
import pyaudio


def speak(msg):
    engine.say(msg)
    engine.runAndWait()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words



def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if (ints):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "Sorry Data Not Available"

def autocorrect(msg2):
    text=TextBlob(msg2)
    msg2=str(text.correct())
    return msg2

import tkinter
from tkinter import *

def send(e):
    j=e
    msg1 = EntryBox.get("1.0",'end-1c').strip()
    msg = autocorrect(msg1)
    EntryBox.delete("0.0",END)
    
    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "\nYou: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12 ))
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        ChatBox.insert(END, "Bot: " + res + '\n\n')
        ChatBox.config(state=NORMAL)
        ChatBox.yview(END)
        #speak(res)

def buttonclick():
    r = sr.Recognizer()
    r.pause_threshold = 0.7
    r.energy_threshold = 400
    with sr.Microphone() as source:
        try:
            audio = r.listen(source, timeout=5)
            msg = str(r.recognize_google(audio))
            msge = autocorrect(msg)
            EntryBox.insert("1.0",msge)
        except sr.UnknownValueError:
            ChatBox.insert(END, "Cant recognize")
            


root = Tk()
root.title("JIO SUPPORT ASSISTANT")                  
root.geometry("450x450")
root.resizable(width=TRUE, height=TRUE)

ChatBox = Text(root, bd=0, bg="white", height="80", width="200", font="Arial",wrap=WORD,highlightbackground="black")         
ChatBox.config(state=NORMAL)

photo = PhotoImage(file = r"E:\Project\Jio\download.png") 
photoimage = photo.subsample(9,9) 

mic = Button(root, font=("Sans Serif",12,'bold italic'), text="MIC", command=buttonclick, bd=0, bg='white',activebackground="#3c9d9b", overrelief='groove', relief='sunken', image= photoimage)


SendButton = Button(root, font=("Sans Serif",12,'bold italic'), text="SEND", width="12", height=5,          
                    bd=0,bg='green',activebackground="#3c9d9b",fg='#000000',
                    command= lambda: send(2), overrelief='groove', relief='sunken')



EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font="Arial" )
EntryBox.bind("<Return>",send)

ChatBox.place(x=6,y=6, height=390, width=450)                   
EntryBox.place(x=6, y=401, height=45, width=265)
SendButton.place(x=270, y=401, height=45)
mic.place(x=398, y=401, height=45, width=50)

root.mainloop()
