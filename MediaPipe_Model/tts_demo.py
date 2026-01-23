import pyttsx3


def text_to_speech(text, rate=150, volume=0.7, pitch=75, voice_index=1):
    """
    Convert text to speech with customizable parameters.
    
    Args:
        text (str): The text to be spoken
        rate (int): Speaking rate (default: 150)
        volume (float): Volume level between 0 and 1 (default: 0.7)
        pitch (int): Pitch value between 0-100 (default: 75)
        voice_index (int): Voice index (0 for male, 1 for female, default: 1)
    """
    engine = pyttsx3.init()  # object creation
    
    # Set RATE
    engine.setProperty("rate", rate)
    
    # Set VOLUME
    engine.setProperty("volume", volume)
    
    # Set VOICE
    voices = engine.getProperty("voices")
    if voice_index < len(voices):
        engine.setProperty("voice", voices[voice_index].id)
    
    # Set PITCH
    engine.setProperty("pitch", pitch)
    
    # Speak the text
    engine.say(text)
    engine.runAndWait()
    engine.stop()


# Example usage
if __name__ == "__main__":
    text_to_speech("Confirmation Doctor, I will be terminating my task now!")
