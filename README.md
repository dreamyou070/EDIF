# EDIF: Feedback-Driven Structure-Preserving Image Editing

> Official repository for the paper:  
> **EDIF: Feedback-Driven Structure-Preserving Image Editing**  
> Chanhum Park et al., 2025  
> 📄 [Paper (PDF)](link-to-pdf)  
> 📥 Coming soon: [Project Page](#), [Demo](#), [Colab](#)

---

## 🌟 Overview

**EDIF** (Editing via Dynamic Interactive Feedback) is a feedback-driven framework for text-based image editing that achieves a strong balance between **structural preservation** and **semantic fidelity**.  
It introduces two novel components:
- **Edif-S**: A structure-aware module guided by SSIM-based feedback
- **Edif-E**: A semantic feedback module using VLM-based alignment signals

Unlike prior methods, EDIF dynamically adjusts conditioning strength at each transformer block based on intermediate editing outputs.

---

## 🎯 Key Contributions

- 🔄 **Feedback Modulation**: Dynamically adjusts editing behavior using SSIM and VLM scores.
- 🧠 **Blockwise Control**: Selective latent modulation across transformer layers.
- 🪞 **Latent Switching**: Blends source and edited features based on structure-preserving signals.
- 📊 Extensive evaluation on scene-centric editing benchmarks (Places365, real-world CCTV, etc.)

---

## 🖼️ Qualitative Examples

> Add a few sample images (before & after editing, with instructions)

---

## 📦 Installation

```bash
git clone https://github.com/your-username/edif.git
cd edif
conda create -n edif python=3.10
conda activate edif
pip install -r requirements.txt
