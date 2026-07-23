# scripts/train_wakeword.py
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add src to python path so we can import factories and config
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import load_config
from factories import build_tts
from openwakeword.utils import AudioFeatures

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 1. Initialize config and TTS
cfg = load_config("config.yaml")
tts = build_tts(cfg)
print(f"[Training] Using TTS voice model from: {cfg.tts_voice_path}")

def resample_22050_to_16000(audio: np.ndarray) -> np.ndarray:
    original_length = len(audio)
    target_length = int(original_length * 16000 / 22050)
    indices = np.linspace(0, original_length - 1, target_length)
    resampled = np.interp(indices, np.arange(original_length), audio)
    return resampled.astype(np.int16)

def generate_variations(text: str, count: int) -> list[np.ndarray]:
    base_pcm = tts._synthesize_to_pcm(text)
    if base_pcm is None:
        raise ValueError(f"Failed to synthesize text: {text}")
    
    base_16k = resample_22050_to_16000(base_pcm)
    
    clips = []
    for _ in range(count):
        pcm = base_16k.copy()
        
        # Vary speed (between 0.8 and 1.25)
        speed = random.uniform(0.8, 1.25)
        if speed != 1.0:
            original_len = len(pcm)
            target_len = int(original_len / speed)
            indices = np.linspace(0, original_len - 1, target_len)
            pcm = np.interp(indices, np.arange(original_len), pcm).astype(np.int16)
            
        # Vary volume
        volume = random.uniform(0.5, 1.5)
        pcm = np.clip(pcm * volume, -32768, 32767).astype(np.int16)
        
        # Add slight background noise
        noise_level = random.uniform(0.001, 0.01)
        noise = np.random.normal(0, noise_level * 32767, len(pcm)).astype(np.int16)
        pcm = np.clip(pcm + noise, -32768, 32767).astype(np.int16)
            
        clips.append(pcm)
    return clips

feat_extractor = AudioFeatures()

def extract_windows(audio_clips: list[np.ndarray]) -> np.ndarray:
    windows = []
    for clip in audio_clips:
        # Pad with silence to ensure we get enough frames
        padded = np.concatenate([np.zeros(1600, dtype=np.int16), clip, np.zeros(1600, dtype=np.int16)])
        feat_extractor.reset()
        
        # Feed in chunks of 1280 samples
        chunk_size = 1280
        for i in range(0, len(padded) - chunk_size, chunk_size):
            feat_extractor(padded[i:i+chunk_size])
            
        feats = np.array(feat_extractor.feature_buffer)
        if len(feats) >= 16:
            for j in range(len(feats) - 16 + 1):
                windows.append(feats[j:j+16])
    return np.array(windows)

# Generate positive clips
print("[Training] Generating positive audio samples for 'Baymax'...", flush=True)
pos_clips = []
pos_clips.extend(generate_variations("hey baymax", 60))
pos_clips.extend(generate_variations("baymax", 40))

# Generate negative clips
print("[Training] Generating negative audio samples...", flush=True)
neg_clips = []
neg_phrases = ["hey jarvis", "hey alexa", "hey siri", "hello reachy", "good morning", "weather today", "NCERT books", "National Public School", "Computer Vision", "Offline AI assistant"]
for phrase in neg_phrases:
    neg_clips.extend(generate_variations(phrase, 12))
    
for _ in range(35):
    noise = np.random.normal(0, random.uniform(0.005, 0.05) * 32767, 16000 * 2).astype(np.int16)
    neg_clips.append(noise)

X_pos = extract_windows(pos_clips)
X_neg = extract_windows(neg_clips)
print(f"[Training] Extracted {X_pos.shape[0]} positive samples and {X_neg.shape[0]} negative samples.", flush=True)

X = np.concatenate([X_pos, X_neg], axis=0).astype(np.float32)
y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])], axis=0).astype(np.float32)

# Shuffle
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

model = Classifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("[Training] Starting training loop...", flush=True)
model.train()
for epoch in range(60):
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/60 | Loss: {epoch_loss/len(dataloader):.4f}", flush=True)

print("[Training] Exporting model to ONNX format...", flush=True)
model.eval()
dummy_input = torch.randn(1, 16, 96)
output_path = "assets/wakeword_models/hey_baymax_v0.1.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"[Training] Successfully saved custom wake word model to '{output_path}'!", flush=True)
