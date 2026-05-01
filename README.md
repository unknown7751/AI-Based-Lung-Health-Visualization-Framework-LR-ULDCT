# ULDCT Enhancement and Segmentation Pipeline

## Project Overview
This project presents a state-of-the-art 5-Stage 2D Medical CT Enhancement and Analysis Pipeline. It is engineered to take Ultra-Low-Dose CT (ULDCT) scans, clean the quantum noise, hallucinate lost high-frequency details, segment the anatomical structures, and deterministically output an objective clinical severity score.

## Pipeline Architecture

### Stage 1: Standard 2D Data Preprocessing
Before any AI is involved, the raw scanner data must be mathematically standardized into a format the neural networks can digest.
* **DICOM Extraction & HU Windowing**: The pipeline reads the raw `.dcm` files. It applies a specific Hounsfield Unit (HU) "Lung Window" (typically -1250 to +250). This flattens bones to maximum white and background air to absolute black, forcing the AI to focus entirely on the soft lung tissues.
* **Normalization**: The HU values are squashed into a strict mathematical range of 0.0 to 1.0 (or -1.0 to 1.0) to stabilize the neural network gradients during training.
* **Padding & Resizing**: Because patients and scanners vary, every slice is padded with black pixels to make it perfectly square, then mathematically resized (using bilinear interpolation) to a fixed dimension like 512x512.
* **The Handoff**: The PyTorch DataLoader passes a standard $1 \times 512 \times 512$ grayscale tensor to the first deep learning model.

### Stage 2: 2D Residual Noise Restoration (DnCNN)
Ultra-Low-Dose CT (ULDCT) scans are incredibly safe but suffer from severe quantum mottle (visual graininess).
* **The Input**: The single noisy $1 \times 512 \times 512$ slice.
* **The Operation**: The slice passes through a Deep Denoising Convolutional Neural Network (DnCNN). Crucially, this network uses Residual Learning. Instead of trying to paint a clean lung, the network's mathematical objective is to strictly identify and separate the noise layer from the tissue.
* **The Architecture**: It uses continuous Conv2D and BatchNorm layers with no max pooling. This ensures the output remains exactly $512 \times 512$.
* **The Output**: The network mathematically subtracts the predicted noise from the original image, yielding a clean, denoised 2D slice.

### Stage 3: 2D Detail Enhancement (SRGAN)
Removing heavy noise naturally leaves the image looking slightly blurred. You must restore the microscopic, high-frequency textures (like tiny blood vessels and alveoli boundaries) so the final diagnosis is accurate.
* **The Input**: The clean, but slightly soft, 2D slice from Stage 2.
* **The Generator**: A Generative Adversarial Network acts as a "detail hallucinator." It uses residual blocks and a technique called PixelShuffle (subpixel convolutions) to aggressively sharpen the image based on purely 2D spatial patterns.
* **The Discriminator**: A secondary VGG-style network acts as quality control. It looks at the generated 2D slice and a real Standard-Dose 2D slice, forcing the Generator to produce hyper-realistic textures.
* **The Output**: A diagnostic-quality, high-resolution $1 \times 512 \times 512$ tensor.

### Stage 4: 2D Anatomical Segmentation (U-Net)
This is the semantic engine. Its sole purpose is to understand the medical anatomy and output exact digital boundaries for the lung and the infection.
* **The Input**: The super-resolved 2D slice from Stage 3.
* **The Encoder (Contracting Path)**: The network uses standard Conv2D and MaxPool2D layers to shrink the image, extracting deep, abstract mathematical features (e.g., learning the visual difference between healthy tissue and a Ground Glass Opacity).
* **The Decoder (Expanding Path)**: The network uses ConvTranspose2D to rebuild the image back to $512 \times 512$. It uses Skip Connections to copy the sharp, high-resolution spatial edges from the Encoder directly across the network, ensuring the final boundaries are razor-sharp.
* **The Output**: A $1 \times 1$ convolution and a Sigmoid activation collapse the deep features into a $2 \times 512 \times 512$ probability tensor (Channel 1 = GGO probability, Channel 2 = Lobe probability).

### Stage 5: Deterministic Clinical Scoring
This final stage bridges the gap between deep learning and clinical medicine using rigid, algorithmic post-processing.
* **Binarization**: The soft probabilities from the U-Net are converted into hard masks. Any pixel $> 0.5$ becomes a 1 (positive), else 0 (negative).
* **Logical Isolation**: The algorithm performs element-wise tensor multiplication (Logical AND) between the Lobe mask and the GGO mask. This strictly isolates which GGOs belong to which of the 5 lung lobes.
* **Percentage Calculation**: It sums the physical pixel area of the infection and divides it by the total pixel area of the specific lobe.
* **The Final Diagnosis**: Using a clinical heuristic, the percentage is mapped to a score of 0 to 5 for each lobe. These are summed together, outputting a final, objective 25-Point CT Severity Score for the patient.

## Results and Visualizations (SRGAN Stage)

The following visualizations demonstrate the performance of the Stage 3 SRGAN detail enhancement mechanism that restores the diagnostic fidelity of the denoised images.

### Quantitative Evaluation (Epoch 85)

| Dataset | Batches | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | MSE ↓ | MAE ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal Cases** | 665 | 43.632 | 0.9918 | 0.0058 | 0.000064 | 0.0035 |
| **COVID-19 Cases** | 1,516 | 43.201 | 0.9916 | 0.0066 | 0.000073 | 0.0036 |
| **Overall** | 2,180 | 43.331 | 0.9916 | 0.0064 | 0.000070 | 0.0036 |

### SRGAN Generated Samples
![SR Samples](SRGAN_Report/Overall/eval_outputs/sr_samples_epoch85.png)

### Bicubic vs SRGAN Comparison
![Bicubic vs SRGAN](SRGAN_Report/Overall/eval_outputs/bicubic_vs_srgan_epoch85.png)

### Difference Maps
![Difference Maps](SRGAN_Report/Overall/eval_outputs/diff_maps_epoch85.png)

### Metric Distribution (PSNR, SSIM, LPIPS)
![Metric Distributions](SRGAN_Report/Overall/eval_outputs/metric_distribution_epoch85.png)

## Conclusion
This 5-stage pipeline effectively transforms highly degraded, noisy ULDCT raw data into high-contrast, segmented clinical insights. By leveraging consecutive specialized deep learning networks—DnCNN for isolating quantum mottle, SRGAN for aggressive texture and edge recovery, and U-Net for razor-sharp multi-class semantic segmentation—the system eliminates the diagnostic uncertainty typical of low-dose protocols. Finally, the deterministic scoring module bridges the gap between raw probability tensors and actionable medical metrics, enabling an objective, automated 25-point CT severity assessment. This robust framework proves that AI-driven post-processing can make safe, ultra-low-radiation scanning protocols diagnostically viable for complex lung health evaluation.

---

## Dataset Structure

This repository utilizes a COVID-19 Low-Dose Computed Tomography (LDCT) dataset organized into two main subsets.

```
COVID-LDCT/
├── Dataset-S1/
│   ├── Clinical-S1.csv
│   ├── LDCT-SL-Labels-S1.csv
│   ├── Radiologist-S1.csv
│   ├── COVID-S1/
│   │   ├── C001/ - C104/
│   │   │   └── IM0001.dcm - IM00XX.dcm
│   └── Normal-S1/
│       └── N001/ - N056/
│           └── IM0001.dcm - IM00XX.dcm
└── Dataset-S2/
    ├── Clinical-S2.csv
    └── COVID-S2/
        ├── PCRP-Lung-Negative/
        └── PCRP-Lung-Positive/
```

### Recommendations for Loading the Dataset
1. **Patient Privacy**: Ensure compliance with data privacy regulations when handling patient data.
2. **DICOM Reading**: Use libraries like `pydicom` (Python) or `dcmtk` to read DICOM files.
3. **Cross-referencing**: Use patient IDs to link imaging data with corresponding CSV metadata.
