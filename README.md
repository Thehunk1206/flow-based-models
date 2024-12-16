# Conditional Flow Matching: Optimal Transport Implementation

## 🚀 Overview
This project implements Conditional Flow Matching (CFM) with a focus on Optimal Transport Conditional Flow Matching (OTCFM), providing a flexible framework for generative modeling.

## 📄 Research Papers
- [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)
- [Rectified Flows: A Probabilistic Flow Model](https://arxiv.org/pdf/2209.03003)

## 🔍 Helpful Resources
### Technical Blogs
1. [Flow Matching Explanation by Tomczak](https://jmtomczak.github.io/blog/18/18_fm.html)
2. [Cambridge MLG Flow Matching Blog](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)
3. [Rectified Flows Introduction](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)

## ✨ Features
- Optimal Transport Conditional Flow Matcher (OTCFM)
- Independent Conditional Flow Matcher (ICFM)
- EMA (Exponential Moving Average) model tracking
- TensorBoard logging
- Flexible sampling methods (Euler and Heun solvers)

## 🔬 Experimental Setup
### Automated Experiment Runner
The project includes a comprehensive bash script for running and tracking multiple experiments:

#### Experiment Configurations
1. **OTCFM Default**
   - Model: OTCFM
   - Learning Rate: 1e-4
   - Batch Size: 16
   - Total Training Steps: 1,000,000

2. **OTCFM Large Batch**
   - Model: OTCFM
   - Learning Rate: 1e-4
   - Batch Size: 32
   - Total Training Steps: 1,000,000

3. **ICFM Default**
   - Model: ICFM
   - Learning Rate: 1e-4
   - Batch Size: 16
   - Total Training Steps: 1,000,000

### Experiment Tracking
- Automated logging of completed experiments
- Error handling and resumable experiments
- Customizable experiment parameters

## 🛠 Model Architecture
- UNet-based generative model
- Configurable channel dimensions
- Multi-head attention mechanisms
- Adaptive learning rate with warmup

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/Thehunk1206/flow-based-models.git
cd flow-based-models
pip install -r requirements.txt
```

### Running Experiments
```bash
# Run all predefined experiments
./experiment_runner.sh

# Or run a specific configuration
python train_cfm.py --model otcfm --batch_size 16
```

## 🔧 Configurations
- Supports both OTCFM and ICFM variants
- Customizable hyperparameters
- Flexible image preprocessing
- TensorBoard integration for tracking

## 📊 Hyperparameter Exploration
- Learning Rate: Adaptive with warmup
- Gradient Clipping
- EMA Decay
- Batch Size Variations
- Total Training Steps

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License
[Add your license here]

## 🎉 Acknowledgments
- Original Flow Matching Paper Authors
- PyTorch Community
- Open-source Generative Modeling Researchers