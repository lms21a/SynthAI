# SynthAI Playground
Inspired by Andrej Karpathy's NanoGPT and MinGPT, Synth Playground is an NLP playground for everyone of all levels to bring their ideas to life. 
In 3 files, one can prepare any dataset from hugging face or utilize a synthetic dataset, pretrain a prebuilt model or their own creation, and inference with the same model.
Utilizing Lightning, we can further provide control, utilizing their 40+ flags for training. And due to the modularity of the code, anyone can pick and choose what they need and want.

# Features
* Simpliclity
* Diverse Training Tools
* Synthetically Created Datasets Available!
* Pre-Built Models
* Components Library (Anyone is more than welcome to contribute a component)

# Some Model Results



# Important Information
- All prebuilt models are trained and optimized for a NVIDIA RTX 4090 GPU. Lightning should handle DDP or FSDP training, as well as multi-device training
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
