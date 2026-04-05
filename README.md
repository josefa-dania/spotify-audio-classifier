# 🎵 Scan2Sound - DSP Audio Classifier

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **Digital Signal Processing (DSP)** project that analyzes Spotify tracks using real audio features like tempo, spectral centroid, RMS energy, and zero-crossing rate to classify songs into categories.

## ✨ Features

- 📷 **QR Code Scanning** - Scan Spotify track QR codes
- 🔗 **Manual Link Input** - Paste any Spotify track URL
- 🎵 **YouTube Audio Download** - Fetches actual audio for DSP analysis
- 🔬 **Real DSP Analysis** - Extracts tempo, energy, spectral centroid, ZCR, MFCCs
- 🏷️ **Intelligent Classification** - Party, Dance, Rock, Slow, Chill, Pop
- 💾 **Spotify Integration** - Saves analyzed tracks to your library
- 🎨 **Modern Dashboard UI** - Professional side-by-side layout

## 🎯 DSP Features Extracted

| Feature | DSP Method | Description |
|---------|------------|-------------|
| Tempo (BPM) | Onset detection + Autocorrelation | Beat/speed detection |
| RMS Energy | Amplitude envelope | Loudness/energy level |
| Spectral Centroid | FFT frequency weighting | Brightness of sound |
| Zero Crossing Rate | Sign changes | Percussive content |
| Spectral Bandwidth | Frequency spread | Sound texture |
| MFCCs | Mel-frequency cepstrum | Timbre features |

## 📊 Classification Logic
