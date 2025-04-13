Project Goal: Create a Dockerized Streamlit application for generating cricket match clips from YouTube videos and a corresponding match data CSV file. The application should be simplified from the provided examples (frontend.py, clipCreatorLocal.py), focusing only on core local functionality and adding a clip preview/adjustment feature before final generation.
Core Requirements:
Application Structure:
Create a main Streamlit application file (e.g., app.py).
Create a separate Python module (e.g., video_utils.py) to encapsulate the video downloading, data processing, and clip generation logic.
Use clear function separation for distinct tasks (downloading, CSV parsing, timestamp adjustment, event filtering, clip cutting, etc.).
Frontend (app.py - Streamlit):
Input:
Text input for YouTube video URL.
File uploader for the match data CSV file.
Text input for specifying a local output directory within the Docker container (this path will likely be mapped to a host volume).
Number inputs (Hours, Minutes, Seconds) for the user to specify the exact time in the video that corresponds to the first event listed in the CSV.
Workflow & Display:
Button to initiate video download based on the URL. Display download progress/status.
Once downloaded, display basic video metadata (Title, Duration, potentially Publish Date).
Button to load and process the uploaded CSV after the video is downloaded and the first event timestamp is set.
Display the processed match data (Pandas DataFrame) with calculated video timestamps for each event (see Requirement 3).
UI elements (e.g., multi-select, select boxes) to define event categories for clip creation (e.g., "Player X - 4s", "All Players - Wickets", "Player Y - 6s").
Button to "Prepare Clips" based on the defined categories. This should filter the data and prepare a list of potential clips with calculated start/end times.
NEW Feature: Clip Preview & Adjustment:
After "Prepare Clips" is clicked, display a new section listing the prepared clips (e.g., "Player X - 4 at 0:15:32").
For each potential clip:
Show the calculated start and end time (e.g., Start: 0:15:27, End: 0:15:47).
Provide a small embedded video player (st.video) showing the video segment defined by the current start/end time for that specific clip.
Provide number inputs or sliders to allow the user to adjust the start and end time (e.g., modify the offset relative to the event timestamp). For example, inputs for "Start Offset (s)" and "End Offset (s)".
An "Update Preview" button next to each clip's adjustment inputs to reload the st.video player with the newly adjusted time range.
Output:
A final "Generate All Clips" button. This button triggers the actual video cutting process using the final, potentially user-adjusted start/end times for all prepared clips.
Display status during clip generation (e.g., using st.spinner).
Once complete, display a success message and list the file paths of the created clips within the specified output directory.
Backend (video_utils.py):
Video Handling:
Function to download a YouTube video using pytubefix to a specified local path. Return the video file path and YT object/metadata.
Function to get video duration using moviepy.
Data Processing:
Function to load a CSV into a Pandas DataFrame, keeping necessary columns (Timestamp, Batter, Bowler, Runs, Wicket, etc.). Parse timestamp columns correctly.
Function to calculate the time offset between the CSV's first event timestamp and the user-provided video timestamp for that event.
Function to apply this offset to all event timestamps in the DataFrame, creating a new 'Video Timestamp' column (in seconds).
Event Handling:
Function to filter the DataFrame based on user-defined event categories (player, type, runs).
Function to determine the default start and end time (in seconds) for a clip based on an event's 'Video Timestamp' and a standard clip length (e.g., timestamp - 5s to timestamp + 15s).
Clipping:
Function that takes a list of clip definitions (each including video path, output path, final start time, final end time, event details for filename).
Use moviepy (VideoFileClip.subclip) to extract the segment for each definition.
Save the subclip to the specified output directory with a descriptive filename (e.g., MatchDate_Player_Event_Timestamp.mp4). Handle potential errors during clipping. Ensure filenames are sanitized.
Dockerization:
Create a Dockerfile:
Use a suitable Python base image (e.g., python:3.10-slim).
Install ffmpeg (required by moviepy).
Copy requirements.txt and install dependencies.
Copy the application code (app.py, video_utils.py).
Set the working directory.
Expose the Streamlit port (8501).
Set the CMD to run the Streamlit app (e.g., streamlit run app.py --server.address=0.0.0.0).
Create a requirements.txt file listing all necessary Python libraries (streamlit, pandas, pytubefix, moviepy).
(Optional but Recommended): Provide instructions or a simple docker-compose.yml example showing how to run the container and mount a local host directory (e.g., ./output) to the container's output path (e.g., /app/output) for persistent clip storage.
General Considerations:
Implement error handling (e.g., invalid URL, CSV format issues, video download failures, clipping errors).
Provide user feedback throughout the process (e.g., spinners, status messages).
Ensure the code is clean, well-commented (where necessary), and follows good practices.