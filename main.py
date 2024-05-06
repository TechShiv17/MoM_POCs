from ollama import Client
from faster_whisper import WhisperModel

model_size = "large-v2"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("1to1mtg.mp3", beam_size=1)

transcriptText = ""
for segment in segments:
    transcriptText += segment.text

# Replace with your desired prompt
prompt = f"Summarize the following text:{transcriptText}"

# Create a client instance
client = Client()

# Send the prompt to Mistral model
response = client.generate(model="mistral:instruct", prompt=prompt)

# Print the Mistral generated response
print(response['response'])