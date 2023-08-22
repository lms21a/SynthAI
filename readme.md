# SynthAI Playground
Inspired by Andrej Karpathy's NanoGPT and MinGPT, Synth Playground is an NLP playground for everyone of all levels to bring their ideas to life. 
In 3 files, one can prepare any dataset from hugging face, pretrain a prebuilt model or their own creation, and inference with the same model.
Utilizing Lightning, we can further provide control, utilizing their 40+ flags for training. And due to the modularity of the code, anyone can pick and choose what they need and want.

# Features
* Simpliclity
* Diverse Training Tools
* Synthetically Created Datasets Available!
* Pre-Built Models
* Components Library (Anyone is more than welcome to contribute a component)

# Some Model Results
Some Prebuilt Models include my 50M parameter Model, LlamaKinda, which is kinda based off the Llama architecture, was trained for 93.5 hours on OpenWebText 2. Here are some training charts of experiments, as well as a quote from the model about the meaning of life.
Keep in mind that this was a "document completion" since the model is not yet fine-tuned.

```
Prefix Prompt: The meaning of life is 
LlamaKinda: The meaning of life is ambiguous — and lending in to existence — the key to each of us.
```
Furthermore, here we have the validation loss for LlamaKinda, which was trained for ~ 4 Days. It acheived a Validation Perplexity of 3.381, which is comparable to NanoGPT's 124M model's Validation Perplexity at 2.905.
![Val Loss](assets/val_loss.png)

This also allowed for experimentation with GQA versus MHA. In the following image, we can see the gray run which utilizes GQA with a group_dim of 2, while the yellow run utilizes the tradtional MHA. We can notice the plateu earlier in time. 
![GQA vs. MHA](assets/gqa_vs_mha.png)
You can learn more about [GQA vs MHA](https://arxiv.org/abs/2305.13245)

# Important Information
- All prebuilt models are trained and optimized for a single NVIDIA RTX 4090 GPU. Lightning should handle DDP or FSDP training, as well as multi-device training
- Currently Created for Linux / Mac OS users, however WSL2 works just fine. See instructions for WSL install below
---
### Installing Windows Subsystem for Linux (WSL) 2

Follow these step-by-step instructions to install WSL 2 on your Windows machine.

### Prerequisites

- Windows 10 Version 1903 or higher, with Build 18362 or higher.
- A 64-bit processor with Second Level Address Translation (SLAT).
- Virtualization technology must be enabled in BIOS.

### Step 1: Enable WSL

Open PowerShell as Administrator and run the following command:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

### Step 2: Enable Virtual Machine Platform

Run the following command in PowerShell:

```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

### Step 3: Download the Linux Kernel Update Package

Download the latest WSL 2 Linux Kernel update package from the [official Microsoft link](https://aka.ms/wsl2kernel).

### Step 4: Set WSL 2 as the Default Version

Run the following command in PowerShell:

```powershell
wsl --set-default-version 2
```

### Step 5: Install a Linux Distribution

Choose a Linux distribution from the Microsoft Store (such as Ubuntu) and click the "Install" button.

### Step 6: Set Up Your Linux Distribution

Launch the Linux distribution you installed and follow the on-screen instructions to set up your username and password.

### Congratulations!

You have successfully installed WSL 2 on your Windows machine. You can now run Linux commands and applications alongside your Windows apps.

For more details and troubleshooting, visit the [official Microsoft WSL documentation](https://docs.microsoft.com/en-us/windows/wsl/).

# Stay Tuned
In development is Assemble! a larger-scale, more robust codebased used to train even LARGER models!