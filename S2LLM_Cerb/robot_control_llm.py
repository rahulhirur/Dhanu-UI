from voice_to_text import listen_and_transcribe
from llm import query_cerebras

def main():
    print("🤖 Robot Voice Interface Started")
    print("Say 'stop' or 'exit' to quit\n")

    while True:
        user_text = listen_and_transcribe()
        #print("🗣️ User:", user_text)
        #print(type(user_text))

        if not user_text:
            continue

        if user_text.lower() in ["stop", "exit", "quit"]:
            print("Shutting down.")
            break

        llm_response = query_cerebras(user_text)
        print("🤖 LLM:", llm_response)
        print("-" * 50)

if __name__ == "__main__":
    main()
