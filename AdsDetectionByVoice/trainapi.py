import datetime
import os
import csv
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from pydub.utils import mediainfo
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from scipy.spatial.distance import cdist
import shutil
from utils import *

app = FastAPI()
csv_file_path = 'Ads_Info.csv'
output_dir = 'speaker_embeddings.json'

# Define constants
output_csv_file = "AdsTime.csv"
training_audios_folder = "traning"

# Create directories if they don't exist
if not os.path.exists(output_csv_file):
    with open(output_csv_file, mode='w'):
        pass

if not os.path.exists(training_audios_folder):
    os.makedirs(training_audios_folder)
model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
audio = Audio(sample_rate=16000, mono="downmix")


def save_audio_file(ad_name, audio_file):
    ad_folder = os.path.join(training_audios_folder, ad_name)
    os.makedirs(ad_folder, exist_ok=True)
    destination_path = os.path.join(ad_folder, audio_file.filename)
    with open(destination_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    return destination_path


def embeddings_distance(embedding1, embedding2):
    return cdist(embedding1, embedding2, metric="cosine")[0, 0]


def get_part_embedding(audio_file, start_time, end_time, model=model, audio=audio):
    speaker_voice = Segment(float(start_time), float(end_time))
    waveform, sample_rate = audio.crop(audio_file, speaker_voice)
    embedding = model(waveform[None])
    return embedding


def audio_embedding(audio_file, model=model, audio=audio):
    speaker_voice = Segment(0.0, float(audio.get_duration(audio_file)))
    waveform, sample_rate = audio.crop(audio_file, speaker_voice)
    embedding = model(waveform[None])
    return embedding


def training(label, audio_path):
    audio = Audio(sample_rate=16000, mono="downmix")
    speaker_voice = Segment(0.0, float(audio.get_duration(audio_path)))
    waveform, sample_rate = audio.crop(audio_path, speaker_voice)  # Changed audio_file to audio_path
    embedding = model(waveform[None])
    speaker = [{"label_name": label, "embeddings": embedding, "audio_file": audio_path}]
    save_embeddings(speaker)
    
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
        
    return deleted_label

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

def save_to_csv(label, ads_ending_date,ads_ending_time):
    """Save information to a CSV file"""
    header = ['Label', 'Ads_Ending_Date','Ads_Ending_Time']
    if not os.path.isfile(csv_file_path) or os.path.getsize(csv_file_path) == 0:
        # Write header only if the file is empty or doesn't exist
        with open(csv_file_path, mode='w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
    with open(csv_file_path, mode='a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([label, ads_ending_date,ads_ending_time])

def main():
    root_folder = training_audios_folder
    audio_dict = {}
    for person_folder in os.listdir(root_folder):
        person_folder_path = os.path.join(root_folder, person_folder)
        if os.path.isdir(person_folder_path):
            audio_paths = []
            total_duration = 0
            for audio_file in os.listdir(person_folder_path):
                audio_file_path = os.path.join(person_folder_path, audio_file)
                if os.path.isfile(audio_file_path):
                    audio_paths.append(audio_file_path)
                    duration_str = mediainfo(audio_file_path)["duration"]
                    duration = float(duration_str)  # duration in milliseconds, convert to seconds
                    total_duration += duration
            if audio_paths:
                audio_dict[person_folder] = {"paths": audio_paths, "duration": total_duration}

    speaker_durations = {}
    for person, data in audio_dict.items():
        print(f"{person}:")
        total_duration = data["duration"]
        for audio_path in data["paths"]:
            try:
                print(audio_path)
                result = training(person, audio_path)
            except Exception as e:
                print(e)
        speaker_durations[person] = total_duration
        print("===" * 20)
    
    print("Embeddings are extracted and Saved to Json File")

    if not os.path.exists(output_csv_file):
        with open(output_csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Speaker Name", "Total Duration (seconds)"])

    with open(output_csv_file, mode="r+") as file:
        reader = csv.reader(file)
        lines = list(reader)
        headers = lines[0] if lines else []
        speaker_column_index = (
            headers.index("Speaker Name") if "Speaker Name" in headers else None
        )
        duration_column_index = (
            headers.index("Total Duration (seconds)")
            if "Total Duration (seconds)" in headers
            else None
        )

        if speaker_column_index is not None and duration_column_index is not None:
            for line in lines[1:]:
                if line[speaker_column_index] in speaker_durations:
                    line[duration_column_index] = str(speaker_durations[line[speaker_column_index]])

            file.seek(0)
            writer = csv.writer(file)
            writer.writerows(lines)

            existing_speakers = set(line[speaker_column_index] for line in lines[1:])
            new_speakers = speaker_durations.keys() - existing_speakers
            if new_speakers:
                writer.writerows(
                    [[speaker, speaker_durations[speaker]] for speaker in new_speakers]
                )

    print(f"Speaker durations appended/updated in {output_csv_file}")

@app.post("/process_audio/")
async def process_audio(ad_name: str, video_file: UploadFile = File(...), ads_ending_date: datetime.date = Query(None, description="Date when the ads end"),
                      ads_ending_time: str = Query(None, description="Time when the ads end (format: HH:MM)")):
    
    remove_files_from_directory(training_audios_folder)
    ad_folder_path = os.path.join(training_audios_folder, ad_name)
    os.makedirs(ad_folder_path, exist_ok=True)
    video_path = os.path.join(ad_folder_path, video_file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)
    
    save_to_csv(ad_name, ads_ending_date,ads_ending_time)
    # Process the video
    main()

    
    # os.remove(output_csv_file)
    deleted_label = remove_expired_rows()
    # Return success message
    return {"message": f"Video '{video_file.filename}' saved and processed for ad '{ad_name}'."}



# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("trainapi:app", host="192.168.18.164", port=8000, reload= True)
