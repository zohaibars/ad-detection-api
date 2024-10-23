import datetime
import json
import pandas as pd
import csv
from pydub.utils import mediainfo
import os
import torch
import torchaudio
from scipy.spatial.distance import cdist
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from IPython.display import Audio as player
from utils import *
from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

confidence = 0.9
validation_data_folder = "validation"
csv_file_path = '/home/ali/AliAhmed/AdsDetection/DetectionByVoice/Ads_Info.csv'
output_dir = '/home/ali/AliAhmed/AdsDetection/DetectionByVoice/speaker_embeddings.json'
model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
audio = Audio(sample_rate=16000, mono="downmix")

model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb")
audio = Audio(sample_rate=16000, mono="downmix")


def embeddings_distance(embedding1, embedding2):
    return cdist(embedding1, embedding2, metric="cosine")[0,0]


def get_part_embedding(audio_file, start_time, end_time, model=model, audio=audio):
    speaker_voice = Segment(float(start_time), float(end_time))
    waveform, sample_rate = audio.crop(audio_file, speaker_voice)
    embedding = model(waveform[None])
    return embedding

def audio_embedding(audio_file, model=model, audio=audio):
    speaker_voice = Segment(0., float(audio.get_duration(audio_file)))
    waveform, sample_rate = audio.crop(audio_file, speaker_voice)
    embedding = model(waveform[None])
    return embedding

def remove_files_from_directory(directory):
    # Check if directory is not empty
    if os.listdir(directory):
        # Get list of files and directories in directory
        entries = os.listdir(directory)
        # Loop through each entry
        for entry in entries:
            entry_path = os.path.join(directory, entry)
            # Check if entry is a file and remove it
            if os.path.isfile(entry_path):
                os.remove(entry_path)
            # Check if entry is a directory and remove it recursively
            elif os.path.isdir(entry_path):
                remove_files_from_directory(entry_path)
                os.rmdir(entry_path)
    else:
        print(f"The directory '{directory}' is already empty.")

def recognition_audio_segments(audio_path, segment_duration=5, overlap=1, model=model):
    try:
        audio_duration = audio.get_duration(audio_path)
        segment_start = 0
        segment_end = segment_duration
        
        embeddings = []
        segment_times = []

        # Split audio into segments
        while segment_start < audio_duration:
            # Adjust segment_end if it exceeds audio_duration
            if segment_end > audio_duration:
                segment_end = audio_duration

            # Obtain embedding for current segment
            embeddings.append(get_part_embedding(audio_path, segment_start, segment_end))
            segment_times.append((segment_start, segment_end))

            # Move to the next segment
            segment_start += segment_duration - overlap
            segment_end = segment_start + segment_duration

        # Load known embeddings
        label_names, label_embeddings = load_embeddings()

        # Initialize variables for multiple speakers
        multiple_speakers = []
        current_speaker = None
        current_segment_start = None

        for i, embedding in enumerate(embeddings):
            for label_name, label_embedding in zip(label_names, label_embeddings):
                distance = embeddings_distance(embedding, label_embedding)
                if distance < 1 - confidence:
                    if label_name != current_speaker:
                        if current_speaker is not None:
                            multiple_speakers.append((current_speaker, current_segment_start, segment_times[i-1][1]))
                        current_speaker = label_name
                        current_segment_start = segment_times[i][0]
                    break
            else:
                if current_speaker is not None:
                    multiple_speakers.append((current_speaker, current_segment_start, segment_times[i][1]))
                    current_speaker = None

        # Check if the last segment contains speaker
        if current_speaker is not None:
            multiple_speakers.append((current_speaker, current_segment_start, audio_duration))

        return multiple_speakers

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return [("ERROR", 0, audio_duration)]

def remove_expired_rows():
    current_time = datetime.datetime.now()
    current_time = current_time.replace(microsecond=0)
    updated_rows = []
    deleted_label = None
    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            label, ads_ending_date_str,ads_ending_time_str = row
            if label == "Label":
                continue
            else:
                try:
                    ads_ending_datetime_str = f"{ads_ending_date_str} {ads_ending_time_str}"
                    ads_ending_datetime = datetime.datetime.strptime(ads_ending_datetime_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    return JSONResponse(content={"error": "Datetime string is not in the correct format."}, status_code=400)
                if current_time >= ads_ending_datetime:
                    with open(output_dir, "r") as json_file:
                        data = json.load(json_file)
                        filtered_data = [entry for entry in data if entry['label_name'] != label]

                        with open(output_dir, "w") as json_file:
                            json.dump(filtered_data,json_file,indent=4)
                     
                    deleted_label = label
                else:
                    updated_rows.append(row)

    # Write back the updated rows to the CSV file
    with open(csv_file_path, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(updated_rows)
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) == 0:
        return JSONResponse(content={"message": "No Data to perform Prediction"}, status_code=200)  
    return deleted_label


def main():
    root_folder = validation_data_folder
    audio_dict = {}
    for person_folder in os.listdir(root_folder):
        person_folder_path = os.path.join(root_folder, person_folder)
        if os.path.isdir(person_folder_path):
            audio_paths = []
            for audio_file in os.listdir(person_folder_path):
                audio_file_path = os.path.join(person_folder_path, audio_file)
                if os.path.isfile(audio_file_path):
                    audio_paths.append(audio_file_path)
            if audio_paths:
                audio_dict[person_folder] = audio_paths
   
    # Read existing speaker durations from AdsTime.csv
    ads_time_filename = "AdsTime.csv"
    if os.path.exists(ads_time_filename):
        ads_time_df = pd.read_csv(ads_time_filename, index_col=0)
    else:
        print("AdsTime.csv not found.")
        return
    
    # Initialize a list to store speaker durations
    speaker_durations = []
    
    for person, audio_paths in audio_dict.items():
        for audio_path in audio_paths:
            segments = recognition_audio_segments(audio_path)
            for segment in segments:
                speaker, start_time, end_time = segment
                detection_duration = end_time - start_time
                
                # Calculate percentage of total duration
                if speaker in ads_time_df.index:
                    ads_duration = ads_time_df.loc[speaker, "Total Duration (seconds)"]
                    percentage = (detection_duration / ads_duration) * 100 if ads_duration != 0 else 0
                else:
                    percentage = 0
                
                speaker_durations.append({
                    "Speaker": speaker,
                    "Start Time": start_time,
                    "End Time": end_time,
                    "Detection Duration": detection_duration,
                    "AdsTime Total Duration": ads_duration,
                    "Percentage of Total Duration": percentage
                })

    # Convert the list of speaker durations to a DataFrame
    speaker_durations_df = pd.DataFrame(speaker_durations)
    
    # Save the speaker durations to speaker_durations.csv
    csv_filename = "speaker_durations.csv"
    speaker_durations_df.to_csv(csv_filename, index=False)
    print(f"Speaker durations updated in {csv_filename}")


@app.post("/upload/")
async def process_audio(file: UploadFile = File(...)):
    try:
        file_to_remove = "speaker_durations.csv"
        # Check if the file exists before attempting to remove it
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)
            print(f"{file_to_remove} has been successfully removed.")
        else:
            print(f"{file_to_remove} does not exist.")
        # Create subfolder based on video name
        video_name = file.filename
        video_folder = os.path.join(validation_data_folder, video_name)
        os.makedirs(video_folder, exist_ok=True)

        # Save the uploaded video
        video_path = os.path.join(video_folder, video_name)
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())
        deleted_label = remove_expired_rows()
        main()
        remove_files_from_directory(validation_data_folder)
        csv_filename = "speaker_durations.csv"
        return FileResponse(csv_filename, media_type='text/csv', filename=csv_filename)

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("predictapi:app", host="192.168.18.164", port=8002, reload = True)