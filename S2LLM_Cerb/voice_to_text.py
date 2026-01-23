import speech_recognition as sr

def listen_and_transcribe():
    # Create recognizer
    recognizer = sr.Recognizer()

    # Open the default microphone as source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... (3 sec)")
        recognizer.adjust_for_ambient_noise(source, duration=3)

        print("Say something!")
        audio = recognizer.listen(source)

    try:
        # Use Google Web Speech API
        text = recognizer.recognize_google(audio)
        print("🗣️ User:", text)
        #print(type(text))
        return text

    except sr.UnknownValueError:
        print("Sorry – could not understand audio.")
        return ""

    except sr.RequestError as e:
        print(f"Could not request results – {e}")
        return ""

if __name__ == "__main__":
    while True:
        command_text = listen_and_transcribe()
        if command_text.lower() in ["quit", "exit", "stop"]:
            print("Stopping voice interface.")
            break
