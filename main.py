from transformers import pipeline
import scipy.io.wavfile as wavfile

synth = pipeline(
    "text-to-audio",
    model="facebook/musicgen-small",
    device=0
)

prompt = "relaxing ambient soundscape"
print("⏳ Generating...")
out = synth(prompt, forward_params={"do_sample": True})

wavfile.write("musicgen_output.wav", out["sampling_rate"], out["audio"])
print("✅ Saved: musicgen_output.wav")
