## Overview

This is a real-time facial authentication system that is designed to verify a person’s identity using just their webcam, with a strong focus on preventing spoofing and fake login attempts. The system ensures that only real, live users can access protected areas by combining face recognition with smart security checks.

## Features

- **Secure User Enrollment**: Lets users enroll their facial data to the database.

- **3-Stage anti-spoofing verification**:
  - **Fourier Analysis**: Detects screen replay attacks by analyzing frequency patterns.
  - **3D Shape Analysis**: Uses MediaPipe's landmark depth data (Z-coordinate) to defend against flat photo attacks.
  - **Liveness Detection**: Confirms the subject is alive by monitoring for natural blinks and subtle head movements.

## Setup

### Clone the repository

```bash
git clone https://aprnbj1/authenticationv.git

cd authenticationv
```

### Setup python enviroment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
```

### Run the application

```bash
python3 main.py
```

Open your web browser and navigate to:

`http://localhost`

