import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "Hello, this is a test of the English text-to-speech model."
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    waveform = outputs.waveform

# Move waveform back to CPU for saving or further processing
waveform = waveform.cpu()

# Save the waveform to a WAV file
scipy.io.wavfile.write("output.wav", rate=model.config.sampling_rate, data=waveform.squeeze().numpy())
