from moviepy.editor import VideoFileClip
import os
import shutil
import json
import numpy as np

def video2wav(filename):
    wav_file_path = filename.split(".")[0] + '.wav'
    video_clip = VideoFileClip(filename)
    audio_clip = video_clip.audio
    # Export the audio as a WAV file
    audio_clip.write_audiofile(wav_file_path)
    video_clip.close()
    audio_clip.close()
    return wav_file_path


def save_audio_file(person_name, audio_file_path):
    if not os.path.exists('audios'):
        os.mkdir('audios')
    person_folder = os.path.join('audios', person_name)
    if not os.path.exists(person_folder):
        os.mkdir(person_folder)
    existing_files = os.listdir(person_folder)
    new_file_name = str(len(existing_files) + 1) + "." +audio_file_path.split(".")[-1]
    new_audio_file_path = os.path.join(person_folder, f'{new_file_name}')
    shutil.copy(audio_file_path, new_audio_file_path)
    return new_audio_file_path


def save_embeddings(speakers=None):
    data = []  # This is the new data you want to append
    
    # Load the existing data from the JSON file if it exists
    
    try:
        with open("speaker_embeddings.json", "r") as json_file:
            existing_data = json.load(json_file)
        data.extend(existing_data)
    except FileNotFoundError:
        pass
    
    for speaker_info in speakers:
        embedding = speaker_info["embeddings"]
        label_name = speaker_info["label_name"]
        audio_file = speaker_info["audio_file"]
    
        # Check if label_name already exists in the data
        label_exists = False
        for idx, entry in enumerate(data):
            if entry["label_name"] == label_name:
                # If label_name exists, remove the old entry
                del data[idx]
                label_exists = True
                break
        
        # Store the information in a dictionary
        entry = {
            "label_name": label_name,
            "audio_embeddings": embedding.tolist(),
            "audio_file": audio_file,
        }
        data.append(entry) if not label_exists else data.append(entry)  # Append the new entry
        
    # Save the updated data to the JSON file
    with open("speaker_embeddings.json", "w") as json_file:
        json.dump(data, json_file)

    print("Data appended to speaker_embeddings.json")    




def load_embeddings():
    # Initialize lists to store the loaded data
    label_names = []
    embeddings = []
    try:
        # Load the data from the JSON file
        with open("speaker_embeddings.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("=================================== ERROR ===================================")
        print("                 JSONFile Contaning the Audio Embedding Not FOUND")
        print("-"*50)
        return label_names, embeddings

    
    # Extract the label names and embeddings for each speaker
    for entry in data:
        label_name = entry["label_name"]
        embedding = np.array(entry["audio_embeddings"]).reshape(1, 192)
    
        label_names.append(label_name)
        embeddings.append(embedding)

    return label_names, embeddings

