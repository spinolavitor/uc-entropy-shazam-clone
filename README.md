# 📊 Information Theory Lab: Entropy & Audio Recognition

This repository contains a Python implementation of analytical tools for signal processing, data compression, and a sliding-window based audio recognition system. The project was developed as a practical application of fundamental Information Theory concepts, specifically focusing on information, redundancy, entropy, and mutual information

Developed as part of the Informatics Engineering curriculum at the University of Coimbra

## 🎯 Project Overview

The main goal of this project is to calculate and analyze the theoretical compression limits of various information sources (images and audio) and to build a music identification simulator (similar to Shazam) utilizing probability theory and mutual information. All mathematical processing and graphical visualizations are built on top of the Python data stack

## ✨ Features

- **Statistical Analysis & Entropy:**
  * Generates and visualizes occurrence histograms for any information source given a defined alphabet;
  * Calculates the entropy of the source to determine the maximum possible non-destructive compression rate;
  * Supports analysis of image files (`lena.bmp`, `ct1.bmp`, `binaria.bmp`) and audio files (`saxriff.wav`).

- **Huffman Coding:**
  * Integrates with the `huffmancodec` module to generate optimal prefix codes;
  * Calculates the average number of bits per symbol and the weighted variance of the resulting code lengths.

- **Symbol Grouping Analysis:**
  * Expands the entropy analysis by treating sequences of two contiguous symbols as a single block, evaluating its impact on redundancy and compression efficiency.

- **Audio Search Engine (Mutual Information):**
  * Implements a pattern recognition algorithm that searches for an audio snippet (*query*) within a larger signal (*target*);
  * Uses a "sliding window" technique (with step control) to calculate the mutual information vector over time;
  * Capable of identifying original signals even when corrupted by noise, successfully tested against `target01 - repeat.wav` and `target02 - repeat Noise.wav`;
  * **Music Identification Simulator:** Processes a database of songs (`Song*.wav`), extracts the maximum mutual information for each file against a query, and returns a ranked list of matches in descending order.

## 🛠️ Technologies Used

- **Language:** Python 3.x
- **Numerical & Signal Processing:** NumPy, SciPy
- **Data Visualization:** Matplotlib (configured for headless environments like GitHub Codespaces)
- **Auxiliary Modules:** `huffmancodec.py`

## 📂 Repository Structure

```text
├── data/                       # Source files (lena.bmp, saxriff.wav, targets, songs, etc.)
├── graficos_shazam/            # Auto-generated folder for Shazam simulation plots
├── histogramas_pares/          # Auto-generated folder for grouped symbol histograms
├── histogramas_simples/        # Auto-generated folder for single symbol histograms
├── .gitignore                  # Rules to ignore cache and generated plots
├── huffmancodec.py             # Auxiliary Huffman coding module
├── main.py                     # Consolidated main script containing all tasks
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies lists
```

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/uc-entropy-shazam-clone.git
   cd uc-entropy-shazam-clone
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the complete analysis and music identification simulator:
   ```bash
   python main.py
   ```

*Note: For all audio processing tasks, the scripts strictly process the first channel (first column) of the provided audio files. Visualizations will be automatically saved as `.png` files inside their respective output folders*
