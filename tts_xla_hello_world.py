import soundfile
from espnet2.bin.tts_inference import Text2Speech
try:
    import torch_xla.core.xla_model as xm
except:
    xm = None

device = "xla" if xm else "cuda"
# device = "cuda"  # force cuda use for comparison if desired

print(f"TTS generation using: {device.upper()}")

# Prepare the model
text2speech = Text2Speech.from_pretrained("kan-bayashi/ljspeech_vits", device=device)

# Run the model
speech = text2speech("hello world")["wav"]

# Save the output audio
soundfile.write(f"{device}.out.wav", speech.detach().cpu().numpy(), text2speech.fs, "PCM_16")