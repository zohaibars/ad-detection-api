from collections import defaultdict
import json
import os
import shutil
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Query
import csv
import re
import datetime
import ffmpeg
app = FastAPI()

# Initialize ResNet50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

def get_frame_embeddings(frame):
    """Extract embeddings from a frame using ResNet50 model"""
    height, width, _ = frame.shape
    frame = frame[:int(height*0.85), :]  # Exclude bottom 15% (ticker area)
    frame_tensor = transforms.ToTensor()(frame).to(device)
    frame_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame_tensor)
    resized_frame = torch.nn.functional.interpolate(frame_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    with torch.no_grad():
        embeddings = model(resized_frame)
        embeddings = embeddings.squeeze().tolist()
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

csv_file_path = 'Ads_Info.csv'
output_dir = 'JSON_Data'

def remove_expired_rows():
    current_time = datetime.datetime.now()
    updated_rows = []
    deleted_label = None

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

def calculate_embedding_similarity(embeddings1, embeddings2):
    """Calculate similarity between two embeddings"""
    embeddings1 = torch.tensor(embeddings1).to(device)
    embeddings2 = torch.tensor(embeddings2).to(device)
    
    with torch.no_grad():
        similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2.unsqueeze(0)).item()
    return similarity

@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    """Handle video upload and processing"""
    try:
        deleted_label = remove_expired_rows()

        if deleted_label is not None:
            print(f"Deleted JSON file and row for label: {deleted_label}")
        
        # Save the uploaded video
        videos_dir = 'Validation'
        os.makedirs(videos_dir, exist_ok=True)
        video_path = os.path.join(videos_dir, file.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert .ts to .mp4 if necessary
        if video_path.endswith('.ts'):
            mp4_path = video_path.replace('.ts', '.mp4')
            ffmpeg.input(video_path).output(mp4_path).run()
            video_path = mp4_path
        
        # Create directory to store frames
        frames_dir = 'Test2'
        os.makedirs(frames_dir, exist_ok=True)

        # Extract frames from the video
        extract_frames(video_path, frames_dir)              
        
        # Load previous embeddings
        prev_embeddings_directory = 'JSON_Data'
        video_embeddings = {}

        frame_files = sorted(os.listdir(frames_dir))
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            embeddings = get_frame_embeddings(frame)
            video_embeddings[frame_file] = embeddings
        
        with open('video_embeddings.json', 'w') as json_file:
            json.dump(video_embeddings, json_file)


        csv_file_path = 'video_analysis.csv'
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Label', 'Start Time', 'End Time', 'Detected', 'Total Duration', 'Percentage Shown'])

            occupied_frames = set()

            for json_file in os.listdir(prev_embeddings_directory):
                if json_file.endswith(".json"):
                    json_file_path = os.path.join(prev_embeddings_directory, json_file)
                    with open(json_file_path, 'r') as f:
                        prev_embeddings_data = json.load(f)
                    
                    label_name = prev_embeddings_data["metadata"]["label"]
                    total_duration_seconds = prev_embeddings_data["metadata"]["total_duration_seconds"]
                    
                    output_fps = len(prev_embeddings_data["frame_embeddings"]) / prev_embeddings_data["metadata"]["total_duration_seconds"]

                    matched_time_intervals = []
                    first_matched_frame = None
                    last_matched_frame = None
                    video_embeddings_tensor = {frame_file: torch.tensor(embeddings).to(device) for frame_file, embeddings in video_embeddings.items()}

                    for frame_file, frame_embedding in video_embeddings_tensor.items():
                        current_frame_second = int(re.search(r'(\d+)\.jpg', frame_file).group(1))
                        
                        # Skip frame if it is already occupied
                        if current_frame_second in occupied_frames:
                            continue
                        
                        for prev_frame_file, prev_embedding in prev_embeddings_data["frame_embeddings"].items():
                            prev_embedding_tensor = torch.tensor(prev_embedding).to(device)
                            similarity = calculate_embedding_similarity(frame_embedding, prev_embedding_tensor)
                            if similarity > 0.75:
                                print(f"similarity of Frame : {current_frame_second} label : {label_name} Similarity {similarity}")
                                if first_matched_frame is None:
                                    first_matched_frame = current_frame_second
                                last_matched_frame = current_frame_second
                                break  # Ensuring a single match per frame

                        if first_matched_frame is not None:
                            matched_time_intervals.append((first_matched_frame, last_matched_frame + 1))
                            first_matched_frame = None
                    
                    if matched_time_intervals:
                        start_frame = matched_time_intervals[0][0]
                        end_frame = matched_time_intervals[-1][1]
                        start_time_seconds = int(start_frame / output_fps)
                        end_time_seconds = int(end_frame / output_fps)
                        start_time = f"{start_time_seconds // 60:02d}:{start_time_seconds % 60:02d}"
                        end_time = f"{end_time_seconds // 60:02d}:{end_time_seconds % 60:02d}"
                        matching_seconds = end_time_seconds - start_time_seconds

                        # Mark frames as occupied
                        for frame in range(start_frame, end_frame):
                            occupied_frames.add(frame)
                    else:
                        start_frame = 0
                        end_frame = 0
                        start_time = "00:00"
                        end_time = "00:00"
                        matching_seconds = 0
                    
                    percentage_ad_shown = (matching_seconds / total_duration_seconds) * 100

                    csv_writer.writerow([label_name, start_time, end_time, f'{matching_seconds} sec', f'{total_duration_seconds} sec', f'{percentage_ad_shown:.2f}%'])

        # Delete frames directory
        shutil.rmtree(frames_dir)

        # Delete the uploaded video
        os.remove(video_path)

        # Return CSV file as response
        return FileResponse(csv_file_path, media_type='text/csv', filename='video_analysis.csv')

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("PredictGPU:app", host="192.168.18.164", port=8016, reload=True)
