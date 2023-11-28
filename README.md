# LongTxt2SpeechTTS

## Introduction

LongTxt2SpeechTTS is a powerful application that converts long text into speech and voice cloning using a Text-to-Speech (TTS) model. This application is built with Python and uses the Streamlit library for the web interface, making it user-friendly and easy to use.

Here are some of the key features of LongTxt2SpeechTTS:

- **Multilingual Support**: The application supports a wide range of languages, including English, French, Italian, Spanish, German, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, and Hindi, allowing users to generate speech in their preferred language.
- **Customizable Voice Templates**: Users can select from a variety of voice templates or upload their own voice template. This feature allows users to clone any voice given the right .wav file.
- **Adjustable Speed**: Users can adjust the speed of the generated speech according to their preference.
- **Output Format Selection**: Users have the option to generate the speech in either WAV or MP3 format.
- **Bitrate Selection**: When generating speech in MP3 format, users can select the desired bitrate.

## Installation

The installation procedure is available for Ubuntu and WSL2 under Windows. The application requires Python 3.10 or higher.

Follow these steps to install LongText2SpeechTTS:

1. Clone the GitHub repository:

```bash
git clone https://github.com/ay3ur/longtxt2speechtts.git
```

2. Navigate to the cloned repository:

```bash
cd longtxt2speechtts
```

3. Download the model.pth file and insert it into the tts_models/multilingual/multi-dataset/xtts_v2 directory :

```bash
https://mega.nz/file/IStwBDLA#lhzj2X7QknFfRMf5avl4QSk3s3pdVSoVQ1F0QSD577s
```

4. Make the installation script executable:

```bash
chmod +x install.sh
```

5. Run the installation script with administrative privileges:

```bash
sudo ./install.sh
```

6. **Launch the Application**: Launch the application with:

```bash
./run.sh
```

This will start the LongText2SpeechTTS application on your system. You can access the user interface of the application via a web browser at `localhost:8501`. Enjoy converting long text into speech with ease!
