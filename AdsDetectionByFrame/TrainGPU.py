from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
import os
import shutil
import cv2
import json
import pandas as pd
from datetime import date
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import datetime
import csv

app = FastAPI()

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ResNet50 model and move it to the GPU
model = models.resnet50(pretrained=True).to(device)
model.eval()

def get_frame_embeddings(frame):
    """Extract embeddings from a frame using ResNet50 model"""
    frame_tensor = transforms.ToTensor()(frame).to(device)
    frame_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame_tensor)
    resized_frame = torch.nn.functional.interpolate(frame_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    with torch.no_grad():
        embeddings = model(resized_frame)
        embeddings = embeddings.squeeze().cpu().tolist()  # Move the result back to CPU
    return embeddings

def extract_frames(video_path, frames_dir, target_fps=1):
    """Extract frames from a video"""
    try:
        vidcap = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        output_fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(output_fps / target_fps))
        
        while vidcap.isOpened():
            success, image = vidcap.read()
            if not success:
                break
            frame_count += 1
            if frame_count % frame_interval == 0 or frame_count == 1:
                cv2.imwrite(os.path.join(frames_dir, f"frame{frame_count // frame_interval:03d}.jpg"), image) 
    finally:
        vidcap.release()

# Function to remove all files from a directory if it's not empty
def remove_files_from_directory(directory):
    # Check if directory is not empty
    if os.listdir(directory):
        # Get list of files in directory
        files = os.listdir(directory)
        # Loop through each file and remove it
        for file in files:
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        print(f"The directory '{directory}' is already empty.")

def process_video(video_file, videos_dir, output_dir, label):
    """Process a video file"""

    if video_file.endswith('.mp4'):
        video_path = os.path.join(videos_dir, video_file)
        frames_dir = os.path.join(output_dir, 'Frames', os.path.splitext(video_file)[0])
        output_json = os.path.join(output_dir, f'frame_embeddings_{os.path.splitext(label)[0]}.json')

        os.makedirs(frames_dir, exist_ok=True)
        extract_frames(video_path, frames_dir)

        frame_embeddings = {}

        vidcap = cv2.VideoCapture(video_path)

        total_duration_seconds = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) / vidcap.get(cv2.CAP_PROP_FPS)

        frame_files = sorted(os.listdir(frames_dir))
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            embeddings = get_frame_embeddings(frame)
            frame_embeddings[frame_file] = embeddings

        metadata = {
            "label": label,  
            "total_duration_seconds": total_duration_seconds  
        }
        data_to_save = {
            "metadata": metadata,
            "frame_embeddings": frame_embeddings
        }

        with open(output_json, 'w') as json_file:
            json.dump(data_to_save, json_file)

        shutil.rmtree(frames_dir)
        print(f"Frame embeddings and metadata for {os.path.splitext(video_file)[0]} saved to JSON file successfully!")

csv_file_path = '/home/ali/AliAhmed/AdsDetection/DetectionByFrame/Ads_Info.csv'
output_dir = '/home/ali/AliAhmed/AdsDetection/DetectionByFrame/JSON_Data'

def save_to_csv(label, ads_ending_date, ads_ending_time):
    """Save information to a CSV file"""
    header = ['Label', 'Ads_Ending_Date', 'Ads_Ending_Time']
    if not os.path.isfile(csv_file_path) or os.path.getsize(csv_file_path) == 0:
        # Write header only if the file is empty or doesn't exist
        with open(csv_file_path, mode='w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
    with open(csv_file_path, mode='a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([label, ads_ending_date, ads_ending_time])

def remove_expired_rows():
    current_time = datetime.datetime.now()
    updated_rows = []
    deleted_label = None
    print("current time", current_time)
    print("\n\n\n")
    print(current_time)

    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            label, ads_ending_date_str, ads_ending_time_str = row
            if label == "Label":
                continue
            else:
                try:
                    ads_ending_datetime_str = f"{ads_ending_date_str} {ads_ending_time_str}"
                    ads_ending_datetime = datetime.datetime.strptime(ads_ending_datetime_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    return JSONResponse(content={"error": "Datetime string is not in the correct format."}, status_code=400)
                current_time = current_time.replace(microsecond=0)
                if current_time >= ads_ending_datetime:                             
                    json_file_path = os.path.join(output_dir, f'frame_embeddings_{label}.json')
                    if os.path.exists(json_file_path):
                        os.remove(json_file_path)
                        
                    deleted_label = label
                else:
                    updated_rows.append(row)

    # Write back the updated rows to the CSV file
    with open(csv_file_path, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(updated_rows)
        
    return deleted_label

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...), label: str = None, ads_ending_date: date = Query(None, description="Date in format Year-Month-Day"),
                      ads_ending_time: str = Query(None, description="Time when the ads end (format: HH:MM)")):
    """Handle video upload"""
    try:
        videos_dir = '/home/ali/AliAhmed/AdsDetection/DetectionByFrame/Videos_Data'

        save_to_csv(label, ads_ending_date, ads_ending_time)

        deleted_label = remove_expired_rows()

        if deleted_label is not None:
            print(f"Deleted JSON file and row for label: {deleted_label}")

        if label is None:
            raise HTTPException(status_code=400, detail="Label toh Likh ustad.")

        # Save the uploaded video
        video_path = os.path.join(videos_dir, file.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the video
        process_video(file.filename, videos_dir, output_dir, label)
        # remove_files_from_directory(videos_dir)
        return JSONResponse(content={"message": "Video uploaded and processed successfully"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("TrainGPU:app", host="192.168.18.164", port=8015, reload=True)
