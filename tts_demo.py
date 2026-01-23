import pyttsx3

engine = pyttsx3.init(driverName='espeak')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 50)  
engine.setProperty('volume', 1)  
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


engine.say("Hello sir, how may I help you, sir.")
engine.runAndWait()