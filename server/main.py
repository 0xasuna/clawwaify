from pocket_tts import TTSModel
import scipy.io.wavfile
import requests
import json
from systemprompt import systemprompt
from stt import SpeechToText
import subprocess
import sys


def play_audio(filepath):
    """Play audio file using system player."""
    try:
        # Try different players
        for player in ["aplay", "paplay", "ffplay -nodisp -autoexit"]:
            try:
                subprocess.run(
                    player.split() + [filepath], capture_output=True, check=True
                )
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    except Exception as e:
        print(f"Could not play audio: {e}")


def main():
    # Initialize TTS
    print("Loading TTS model...")
    tts_model = TTSModel.load_model()
    voice_state = tts_model.get_state_for_audio_prompt("fantine")

    # Initialize STT (use tiny.en for fastest response)
    # For GPU: device="cuda", compute_type="float16"
    print("Loading STT model...")
    stt = SpeechToText(model_size="base.en", device="cpu", compute_type="int8")

    # LLM API config
    url = "http://192.168.29.228:8000/ai"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}

    # Conversation history
    # messages = [{"role": "system", "content": systemprompt}]
    messages = []

    print("\n" + "=" * 50)
    print("Voice Chat Ready!")
    print("Press Enter to speak, 'q' to quit, 't' to type")
    print("=" * 50)

    while True:
        cmd = input("\n[Enter=speak, t=type, q=quit] > ").strip().lower()

        if cmd == "q":
            print("Goodbye!")
            break
        elif cmd == "t":
            prompt = input("Type your message: ")
        else:
            # Voice input
            prompt = stt.listen(duration=10)

        print(f"You said: {prompt}")

        if not prompt.strip():
            print("(No speech detected, try again)")
            continue

        # Add user message to history
        messages.append({"role": "user", "content": prompt})

        # Get LLM response
        payload = {"prompt": prompt}

        try:
            response = requests.post(url, data=json.dumps(payload), headers=headers)
            content = response.json()["output"]
        except Exception as e:
            print(f"Error connecting to LLM: {e}")
            messages.pop()  # Remove failed message
            continue

        print(f"\nAI: {content}")

        # Add assistant response to history
        messages.append({"role": "assistant", "content": content})

        # Generate and play TTS
        audio = tts_model.generate_audio(voice_state, content)
        scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())
        play_audio("output.wav")


if __name__ == "__main__":
    main()

