import matplotlib.pyplot as plt
import numpy as np
from playsound3 import playsound #audio player
import time
import random
import math
import json

import os

from scipy.io import wavfile


# Expanded notes vocabulary list including all rhythmic combinations present in the dataset
notes = [
    "[S]", "[E]", "-",
    "A_l_4", "A_l_8", "A_l_16",
    "B_l_4", "B_l_8", "B_l_16",
    "C_4",   "C_8",   "C_16",
    "D_4",   "D_8",   "D_16",
    "E_4",   "E_8",   "E_16",
    "F_4",   "F_8",   "F_16",
    "G_4",   "G_8",   "G_16",
    "A_h_4", "A_h_8", "A_h_16",
    "B_h_4", "B_h_8", "B_h_16",
    "C_h_4", "C_h_8", "C_h_16",
    "D_h_4", "D_h_8", "D_h_16"
]

note_freqs = {
    "[S]": 0.0,
    "[E]": 0.0,
    "-":   0.0,
    "A_l_4": 220.00, "A_l_8": 220.00, "A_l_16": 220.00,
    "B_l_4": 246.94, "B_l_8": 246.94, "B_l_16": 246.94,
    "C_4":   261.63, "C_8":   261.63, "C_16":   261.63,
    "D_4":   293.66, "D_8":   293.66, "D_16":   293.66,
    "E_4":   329.63, "E_8":   329.63, "E_16":   329.63,
    "F_4":   349.23, "F_8":   349.23, "F_16":   349.23,
    "G_4":   392.00, "G_8":   392.00, "G_16":   392.00,
    "A_h_4": 440.00, "A_h_8": 440.00, "A_h_16": 440.00,
    "B_h_4": 493.88, "B_h_8": 493.88, "B_h_16": 493.88,
    "C_h_4": 523.25, "C_h_8": 523.25, "C_h_16": 523.25,
    "D_h_4": 587.33, "D_h_8": 587.33, "D_h_16": 587.33
}


#possible other options (for future) sharps = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
#flats = ["A", "Bb", "B", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab"]
#jazzy_blues = ["A", "C", "D", "D#", "E", "G"]

chords = ["C_M", "A_m", "F_M", "D_m", "E_m", "-"] 
chord_freqs = {
    "A_m": [220.00, 261.63, 329.63], # A3, C4, E4
    "C_M": [261.63, 329.63, 392.00], # C4, E4, G4
    "F_M": [174.61, 220.00, 261.63], # F3, A3, C4
    "D_m": [146.83, 174.61, 220.00], # D3, F3, A3
    "E_m": [164.81, 196.00, 246.94], # E3, G3, B3
    "-":   [0.0, 0.0, 0.0]             # Rest
}

stoi = {s:i for i, s in enumerate(notes)}
itos = {i[1]:i[0] for i in stoi.items()}
ctoi = {c:i for i, c in enumerate(chords)}
itoc = {i:c for c, i in ctoi.items()}


def render_song(melody_arr, chord_arr, name, whole_note):
    base_path = os.path.dirname(__file__)
    song_waves = []
    
    for m, c in zip(np.asarray(melody_arr).ravel(), np.asarray(chord_arr).ravel()):

        melody = itos[m] if isinstance(m, (int, np.integer)) else m
        chord = itoc[c] if isinstance(c, (int, np.integer)) else c
        
        note_value = int(melody.split("_")[-1]) if melody[-1].isdigit() else 8
        print(melody.split("_"))
        dur = whole_note / note_value 
        
        l = np.linspace(0, dur, int(44100 * dur), False) # range of timestamps for each note
        
        m_freq = note_freqs.get(melody, 0.0) #frequency lookup
        m_wave = np.sin(2 * np.pi * m_freq * l) * 32767 if m_freq > 0 else np.zeros_like(l) # zero for S, E, and rests
        
        c_notes = chord_freqs.get(chord, [0.0, 0.0, 0.0])
        active_tones = [f for f in c_notes if f > 0]
        c_wave = (sum(np.sin(2 * np.pi * f * l) for f in active_tones) / len(active_tones)) * 32767 if active_tones else np.zeros_like(l)
            
        song_waves.append(np.column_stack((m_wave, c_wave)).astype(np.int16))
        
    wavfile.write(f"{base_path}/{name}.wav", 44100, np.concatenate(song_waves, axis=0))
    print("song saved")
    playsound(f"{base_path}/{name}.wav")


dataset_path = os.path.join(os.path.dirname(__file__), "prl_strms_improved_dataset.json")

with open(dataset_path) as jsonfile:
    file = json.load(jsonfile)
data = []
for item in file:
    data.append(item)


render_song(data[110]['melody'], data[110]['chord'], "aa", 3.0)

#init model
class Model():
    def __init__(self, kaiming=True):
        super().__init__()
        self.melody_vocab_size = len(notes)
        self.chord_vocab_size = len(chords)
        self.d_model = 128 #hidden dimension/embedding dimension
        kaiming_scale = np.sqrt(2.0/self.d_model)
        self.melody_embeddings = np.random.normal(loc=0.0, scale=kaiming_scale/np.sqrt(2), size=(self.melody_vocab_size, self.d_model))
        self.chord_embeddings = np.random.normal(loc=0.0, scale=kaiming_scale/np.sqrt(2), size=(self.chord_vocab_size, self.d_model))
        self.stream_melody_id = np.random.normal(loc=0.0, scale=kaiming_scale/np.sqrt(2), size=(1, self.d_model))
        self.stream_chord_id = np.random.normal(loc=0.0, scale=kaiming_scale/np.sqrt(2), size=(1, self.d_model))
        #single head self-attention
        self.Q_proj = np.random.normal(loc=0.0, scale=kaiming_scale, size=(self.d_model, self.d_model))
        self.K_proj = np.random.normal(loc=0.0, scale=kaiming_scale, size=(self.d_model, self.d_model))
        self.V_proj = np.random.normal(loc=0.0, scale=kaiming_scale, size=(self.d_model, self.d_model))
        self.out_melody = np.random.normal(loc=0.0, scale=kaiming_scale, size=(self.d_model, self.melody_vocab_size))
        self.out_chord = np.random.normal(loc=0.0, scale=kaiming_scale, size=(self.d_model, self.chord_vocab_size))
    def __call__(self, melody_input, chord_input):
        #forward pass
        B, T = melody_input.shape
        self.melody_input = melody_input
        self.chord_input = chord_input
        sequence_len = T
        self.emb_mel = self.melody_embeddings[melody_input]
        self.emb_chord = self.chord_embeddings[chord_input]
        self.melody_x = self.emb_mel + self.stream_melody_id
        self.chord_x = self.emb_chord + self.stream_chord_id
        self.x_inter = np.stack((self.melody_x, self.chord_x), axis=2).reshape(B, sequence_len*2, self.d_model)
       #self-attention 
        self.Q = self.x_inter @ self.Q_proj.T
        self.K = self.x_inter @ self.K_proj.T
        self.V = self.x_inter @ self.V_proj.T
        wei = (self.Q @ self.K.transpose(0, 2, 1)) / math.sqrt(self.d_model) 
        self.attention_mask = np.where(np.tril(np.ones((2 * sequence_len, 2 * sequence_len))), 0.0, -1e9)
        self.wei = wei +self.attention_mask
        self.exp = np.exp(self.wei- np.max(self.wei, axis=-1, keepdims=True))
        self.exp = self.exp / (np.sum(self.exp, axis=-1, keepdims=True) + 1e-9)
        self.attn_outs = self.exp @ self.V
        self.mel_outs = self.attn_outs[:, 0::2, :]
        self.chord_outs = self.attn_outs[:, 1::2, :]
        print(self.mel_outs.shape, B, T)
        self.melody_logits = self.mel_outs @ self.out_melody
        self.chord_logits = self.chord_outs @ self.out_chord
    def backward(self, dmelody_logits, dchord_logits):
        dmel_outs = dmelody_logits @ self.out_melody.T
        dchord_outs = dchord_logits @ self.out_chord.T
        self.dout_melody = np.einsum('btd,btv->dv', self.mel_outs, dmelody_logits)
        self.dout_chord = np.einsum('btd,btv->dv', self.chord_outs, dchord_logits)
        dattn_outs = np.zeros_like(self.attn_outs)
        dattn_outs[:, 0::2, :] = dmel_outs
        dattn_outs[:, 1::2, :] = dchord_outs
        self.dV_proj = np.einsum('btd,btv->dv', self.x_inter, self.exp.transpose(0, 2, 1) @ dattn_outs)
        dV = self.exp.transpose(0, 2, 1) @ dattn_outs
        dexp = dattn_outs @ self.V.transpose(0, 2, 1)
        dwei = self.exp * (dexp - np.sum(dexp * self.exp, axis=-1, keepdims=True))
        dQ = (dwei / math.sqrt(self.d_model)) @ self.K
        dK = (dwei.transpose(0, 2, 1) / math.sqrt(self.d_model)) @ self.Q
        
        self.dQ_proj = np.einsum('btd,btv->dv', self.x_inter, dQ)
        self.dK_proj = np.einsum('btd,btv->dv', self.x_inter, dK)
        
        dx_inter = (dQ @ self.Q_proj.T) + (dK @ self.K_proj.T) + (dV @ self.V_proj.T)
        dx_inter = (dQ @ self.Q_proj.T) + (dK @ self.K_proj.T) + (dV @ self.V_proj.T)
        
        B, Total_Tokens, D = dx_inter.shape
        dx_reshaped = dx_inter.reshape(B, -1, 2, D)
        dmelody_x = dx_reshaped[:, :, 0, :]
        dchord_x = dx_reshaped[:, :, 1, :]
        self.dstream_melody_id = np.sum(dmelody_x, axis=(0, 1), keepdims=True)
        self.dstream_chord_id = np.sum(dchord_x, axis=(0, 1), keepdims=True)
        self.dmelody_embeddings = np.zeros_like(self.melody_embeddings)
        self.dchord_embeddings = np.zeros_like(self.chord_embeddings)
        for b in range(B):
            for t in range(T):
                self.dmelody_embeddings[self.melody_input[b, t]] += dmelody_x[b, t]
                self.dchord_embeddings[self.chord_input[b, t]] += dchord_x[b, t]

    def parameters(self):
        return [
            self.melody_embeddings,
            self.chord_embeddings,
            self.stream_melody_id,
            self.stream_chord_id,
            self.Q_proj,
            self.K_proj,
            self.V_proj,
            self.out_melody,
            self.out_chord
        ]
    def update_weights(self, lr=0.01):
        self.Q_proj -= lr * self.dQ_proj
        self.K_proj -= lr * self.dK_proj
        self.V_proj -= lr * self.dV_proj
        self.out_melody -= lr * self.dout_melody
        self.out_chord -= lr * self.dout_chord
        self.melody_embeddings -= lr * self.dmelody_embeddings
        self.chord_embeddings -= lr * self.dchord_embeddings
        self.stream_melody_id -= lr * self.dstream_melody_id.reshape(self.stream_melody_id.shape)
        self.stream_chord_id -= lr * self.dstream_chord_id.reshape(self.stream_chord_id.shape)
m = Model()
def assemble_train_batch(melody_raw, chord_raw, block_size=4):
    mel = np.pad([stoi[n] for n in melody_raw], (block_size, 0), constant_values=stoi["-"])
    ch  = np.pad([ctoi[c] for c in chord_raw],  (block_size, 0), constant_values=ctoi["-"])
    N = len(melody_raw)
    X_mel = np.array([mel[i : i + block_size] for i in range(N)])
    X_chord = np.array([ch[i : i + block_size] for i in range(N)])
    return (X_mel, X_chord), (np.roll(X_mel, -1, axis=1), np.roll(X_chord, -1, axis=1))

train_iters = 1000

learning_rate = 0.05
    

for t in range(train_iters):
    #construct training data
    r = random.randint(0, len(data)-1)
    inputs, targets = assemble_train_batch(data[r]['melody'], data[r]['chord'], block_size=12)
    X_mel, X_ch = inputs
    Y_mel, Y_ch = targets
    #forward pass
    m(X_mel, X_ch)
    predicted_melody_indices = np.argmax(m.melody_logits[m.sequence_len-1], axis=-1)
    predicted_chord_indices = np.argmax(m.chord_logits[m.sequence_len-1], axis=-1)
    mel_probs = np.exp(m.melody_logits - np.max(m.melody_logits, axis=-1, keepdims=True))
    mel_probs /= np.sum(mel_probs, axis=-1, keepdims=True)
    ch_probs = np.exp(m.chord_logits - np.max(m.chord_logits, axis=-1, keepdims=True))
    ch_probs /= np.sum(ch_probs, axis=-1, keepdims=True)
    B, T = m.melody_logits.shape[:2]
    loss_m = -np.log(mel_probs[np.arange(B)[:, None], np.arange(T)[None, :], Y_mel] + 1e-9).mean()
    loss_c = -np.log(ch_probs[np.arange(B)[:, None], np.arange(T)[None, :], Y_ch] + 1e-9).mean()
    total_loss = (loss_m + loss_c)    
    #partial backward
    dmel_logits = mel_probs.copy()
    dmel_logits[np.arange(B)[:, None], np.arange(T)[None, :], Y_mel] -= 1.0
    dmel_logits /= (B * T)    
    dch_logits = ch_probs.copy()
    dch_logits[np.arange(B)[:, None], np.arange(T)[None, :], Y_ch] -= 1.0
    dch_logits /= (B * T)    
    m.backward(dmel_logits, dch_logits)
    m.update_weights(lr=learning_rate)
    if t % 10 == 0:
        if t >= 1000:
            t -= 0.01
        print(f"Iter {t} Loss: {total_loss:.4f}")
m(np.array([stoi["-"] for _ in range(12)]).reshape(1, -1),  np.array([ctoi["-"] for _ in range(12)]).reshape(1, -1))

print(m.chord_logits.shape, m.melody_logits.shape)
print(m.melody_logits)
pred_mel = [notes[i] for i in np.argmax(m.melody_logits[-1], axis=-1)]
pred_ch  = [chords[i] for i in np.argmax(m.chord_logits[-1], axis=-1)]
render_song(pred_mel, pred_ch, "output_song", 3.0)
#TODO: write inference engine with KV caching, understand einsum and sinusoidal position encoding
