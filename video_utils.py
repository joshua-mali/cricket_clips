# Placeholder for video utility functions 

import logging
import os
import re
from datetime import timedelta
from pathlib import Path

import pandas as pd
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pytube.exceptions import PytubeError
from pytubefix import YouTube
from pytubefix.cli import on_progress  # Import the progress callback

# from pytubefix.cli import on_progress # Requires rich, avoid for simplicity unless requested

# --- Constants ---
TEMP_VIDEO_DIR = Path("/app/temp_video")
TEMP_PREVIEW_DIR = Path("/app/temp_previews")
DEFAULT_CLIP_BEFORE_EVENT_S = 5
DEFAULT_CLIP_AFTER_EVENT_S = 15

# Ensure temporary directories exist
TEMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
TEMP_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def sanitize_filename(filename):
    """Removes or replaces characters invalid for filenames."""
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores (optional, but common)
    sanitized = sanitized.replace(' ', '_')
    # Limit length if necessary (optional)
    # max_len = 200
    # sanitized = sanitized[:max_len]
    return sanitized

def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

# --- Core Functions ---

def download_video(url: str):
    """
    Downloads a YouTube video to a temporary directory.

    Args:
        url: The YouTube video URL.

    Returns:
        A tuple (video_file_path, metadata_dict) or (None, None) on error.
        metadata_dict contains 'title', 'duration', 'publish_date'.
    """
    logging.info(f"Attempting to download video from: {url}")
    try:
        # Clear previous temporary video files if any
        for item in TEMP_VIDEO_DIR.iterdir():
            if item.is_file():
                logging.info(f"Removing old temp video file: {item}")
                item.unlink()

        yt = YouTube(
            url,
            use_oauth=False,
            allow_oauth_cache=True,
            on_progress_callback=on_progress # Enable the progress callback
        )

        # Get the highest resolution MP4 stream (video only or combined)
        stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()

        # Deprecated logic trying progressive first
        # stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        # if not stream:
        #     stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first() # Try non-progressive

        if not stream:
            logging.error("No suitable MP4 stream found.")
            return None, None

        logging.info(f"Selected highest resolution MP4 stream: {stream}")
        output_filename = sanitize_filename(f"{yt.title}.mp4")
        video_file_path = TEMP_VIDEO_DIR / output_filename

        logging.info(f"Downloading to: {video_file_path}")
        stream.download(output_path=TEMP_VIDEO_DIR, filename=output_filename)
        logging.info("Download complete.")

        metadata = {
            "title": yt.title,
            "duration": yt.length, # Duration in seconds
            "publish_date": yt.publish_date.strftime("%Y-%m-%d") if yt.publish_date else "N/A",
        }
        return str(video_file_path), metadata

    except PytubeError as e:
        logging.error(f"PytubeError downloading video: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error downloading video: {e}", exc_info=True)
        return None, None


# List of essential columns identified by the user + Wicket
# Ensure 'Timestamp' matches the exact column name in your CSV
ESSENTIAL_COLUMNS = [
    'Match', 'Date', 'Batter', 'Bowler', 'Runs', 'Timestamp',
    'Winning Team', 'Batting Team', 'Wicket' # Keeping Wicket for filtering, adjust if needed
]

def load_and_process_csv(uploaded_file, first_event_video_seconds: float):
    """
    Loads essential columns from the match data CSV, processes timestamps,
    and calculates video timestamps.

    Args:
        uploaded_file: The file-like object from st.file_uploader.
        first_event_video_seconds: The time (in seconds) in the video corresponding
                                   to the *first* timestamp in the CSV.

    Returns:
        Pandas DataFrame with added 'Video Timestamp (s)' column, or None on error.
    """
    logging.info("Loading and processing CSV (Essential Columns).")
    try:
        # Use 'usecols' to load only necessary columns
        # Add error handling in case a required column isn't in the list or file
        def check_cols(cols_to_check):
            if not all(col in cols_to_check for col in ESSENTIAL_COLUMNS):
                missing = [col for col in ESSENTIAL_COLUMNS if col not in cols_to_check]
                raise ValueError(f"CSV is missing essential columns: {missing}. Required: {ESSENTIAL_COLUMNS}")
            return [col for col in cols_to_check if col in ESSENTIAL_COLUMNS]

        # Read initially with usecols=check_cols (requires pandas >= 1.4.0)
        # df = pd.read_csv(uploaded_file, usecols=check_cols)

        # Safer alternative for broader pandas compatibility: Read headers, check, then read with usecols list
        temp_df_headers = pd.read_csv(uploaded_file, nrows=0) # Read only headers
        actual_cols_in_file = temp_df_headers.columns.tolist()
        logging.info(f"CSV contains columns: {actual_cols_in_file}")

        missing = [col for col in ESSENTIAL_COLUMNS if col not in actual_cols_in_file]
        if missing:
             raise ValueError(f"CSV is missing essential columns: {missing}. Required: {ESSENTIAL_COLUMNS}")

        # Reload with the validated essential columns
        uploaded_file.seek(0) # Reset file pointer
        df = pd.read_csv(uploaded_file, usecols=ESSENTIAL_COLUMNS)

        logging.info(f"CSV loaded successfully with essential columns: {df.columns.tolist()}")

        # --- Data Cleaning and Timestamp Parsing ---
        timestamp_col = 'Timestamp' # Still assuming this name

        # Parse the timestamp column (rest of the parsing logic remains similar)
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce', format='mixed', dayfirst=True)
        except Exception as parse_error:
             logging.warning(f"Initial datetime parse failed: {parse_error}. Trying without format.")
             try:
                 df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
             except Exception as fallback_parse_error:
                  logging.error(f"Could not parse timestamp column '{timestamp_col}': {fallback_parse_error}", exc_info=True)
                  raise ValueError(f"Failed to parse dates in column '{timestamp_col}'. Check format.")

        # Drop rows where timestamp parsing failed
        original_rows = len(df)
        df.dropna(subset=[timestamp_col], inplace=True)
        if len(df) < original_rows:
            logging.warning(f"Dropped {original_rows - len(df)} rows due to unparseable timestamps.")
        if df.empty:
             raise ValueError("No valid timestamp data found in CSV after filtering columns/rows.")

        # Sort by timestamp
        df.sort_values(by=timestamp_col, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # --- Calculate Video Timestamp Offset ---
        first_csv_timestamp = df[timestamp_col].iloc[0]
        time_diff_seconds = (df[timestamp_col] - first_csv_timestamp).dt.total_seconds()
        df['Video Timestamp (s)'] = first_event_video_seconds + time_diff_seconds

        logging.info("CSV processing complete with essential columns.")
        return df

    except ValueError as ve: # Catch specific ValueError for missing columns
        logging.error(f"CSV Loading/Processing Error: {ve}")
        # Re-raise or return None, depending on how you want app.py to handle it
        # Returning None allows app.py to display the specific error
        return None
    except Exception as e:
        logging.error(f"Error processing CSV: {e}", exc_info=True)
        return None


def prepare_clip_list(df: pd.DataFrame, event_categories: list):
    """
    Filters the DataFrame based on selected event categories and prepares clip definitions.

    Args:
        df: The processed DataFrame with 'Video Timestamp (s)'.
        event_categories: A list of strings defining the desired events
                          (e.g., "All 4s", "Player A - Wickets").

    Returns:
        A list of dictionaries, each representing a clip to be potentially generated.
        Each dict includes: 'event_desc', 'start', 'end', 'adjusted_start', 'adjusted_end'.
        Returns an empty list if no matching events are found.
    """
    logging.info(f"Preparing clip list for categories: {event_categories}")
    prepared_clips = []

    if df is None or df.empty:
        logging.warning("Input DataFrame is empty or None. Cannot prepare clips.")
        return []

    required_cols = ['Video Timestamp (s)', 'Batter', 'Runs', 'Wicket'] # Minimal required columns
    if not all(col in df.columns for col in required_cols):
        logging.error(f"DataFrame missing one or more required columns: {required_cols}")
        raise ValueError("DataFrame must contain 'Video Timestamp (s)', 'Batter', 'Runs', 'Wicket' columns.")


    for category in event_categories:
        logging.debug(f"Processing category: {category}")
        filtered_df = pd.DataFrame() # Initialize empty DataFrame for this category

        # --- Logic to filter based on category string ---
        # This needs to parse the category string (e.g., "Player A - 4s")
        parts = category.split(' - ')
        player_filter = parts[0].strip()
        event_filter = parts[1].strip() if len(parts) > 1 else "All" # Default if no specific event type

        temp_df = df.copy() # Work on a copy for filtering

        # Apply player filter
        if player_filter != "All Players":
            # Check both Batter and Bowler based on context (wicket)
            # This logic might need refinement based on exact desired behavior for player involvement
            is_batter = temp_df['Batter'] == player_filter
            is_bowler_on_wicket = (temp_df['Bowler'] == player_filter) & (temp_df['Wicket'] == True) # Assuming Wicket column is boolean or similar
            temp_df = temp_df[is_batter | is_bowler_on_wicket]


        # Apply event filter (4s, 6s, Wickets)
        if "Wickets" in event_filter:
            # Assuming Wicket column indicates a wicket event (e.g., True, or non-null text)
            # Adjust condition based on how 'Wicket' column is represented
            filtered_df = temp_df[pd.notna(temp_df['Wicket']) & (temp_df['Wicket'] != False) & (temp_df['Wicket'] != 0)] # More robust check for wicket
        elif "4s" in event_filter:
            filtered_df = temp_df[temp_df['Runs'] == 4]
        elif "6s" in event_filter:
            filtered_df = temp_df[temp_df['Runs'] == 6]
        elif event_filter == "All": # Handle categories like "All Players" which might imply all events for them
             filtered_df = temp_df # No specific event filter needed if it's just a player filter

        # --- Generate clip definitions from filtered rows ---
        for index, row in filtered_df.iterrows():
            event_time_s = row['Video Timestamp (s)']
            start_time_s = max(0, event_time_s - DEFAULT_CLIP_BEFORE_EVENT_S)
            end_time_s = event_time_s + DEFAULT_CLIP_AFTER_EVENT_S

            # Create a descriptive label for the event
            event_desc = f"{category} at {format_time(event_time_s)}"
            if player_filter == "All Players" and 'Batter' in row and pd.notna(row['Batter']):
                 event_desc = f"{row['Batter']} - {event_filter} at {format_time(event_time_s)}" # Add player name if known


            clip_def = {
                "event_desc": event_desc,
                "csv_row_index": index, # Keep track of original row if needed
                "start": start_time_s,
                "end": end_time_s,
                "adjusted_start": start_time_s, # Initially same as default
                "adjusted_end": end_time_s,     # Initially same as default
                "preview_path": None, # Placeholder for temporary preview file
            }
            prepared_clips.append(clip_def)
            logging.debug(f"Added clip definition: {event_desc} ({start_time_s:.2f}s - {end_time_s:.2f}s)")

    logging.info(f"Prepared {len(prepared_clips)} clip definitions.")
    return prepared_clips


def generate_clip_preview(video_path: str, preview_start_s: float, preview_end_s: float, clip_index: int):
    """
    Generates a short, low-quality temporary preview clip using ffmpeg_extract_subclip.

    Args:
        video_path: Path to the full source video.
        preview_start_s: Start time in seconds for the preview.
        preview_end_s: End time in seconds for the preview.
        clip_index: An index to make the temporary filename unique.

    Returns:
        Path to the generated temporary preview clip, or None on error.
    """
    if not video_path or not Path(video_path).exists():
        logging.error(f"Source video not found at {video_path}")
        return None
    if preview_start_s >= preview_end_s or preview_start_s < 0:
         logging.warning(f"Invalid preview time range: {preview_start_s} - {preview_end_s}")
         return None

    logging.info(f"Generating preview for clip {clip_index}: {preview_start_s:.2f}s - {preview_end_s:.2f}s")

    # Create a unique temporary filename
    preview_filename = f"preview_{clip_index}_{int(preview_start_s)}_{int(preview_end_s)}.mp4"
    preview_output_path = TEMP_PREVIEW_DIR / preview_filename

    # Clean up any previous preview for this index
    for old_preview in TEMP_PREVIEW_DIR.glob(f"preview_{clip_index}_*.mp4"):
         logging.info(f"Removing old preview file: {old_preview}")
         try:
             old_preview.unlink()
         except OSError as e:
             logging.warning(f"Could not remove old preview {old_preview}: {e}")

    try:
        # Use ffmpeg_extract_subclip for potentially faster, lower-overhead extraction
        # It directly calls ffmpeg command
        ffmpeg_extract_subclip(
            video_path,
            preview_start_s,
            preview_end_s,
            str(preview_output_path) # Pass as 4th positional arg
        )

        # Alternative using moviepy objects (might be slower, more resource intensive)
        # with VideoFileClip(video_path) as video:
        #     preview_clip = video.subclip(preview_start_s, preview_end_s)
        #     # Lower quality settings for speed? May not be easily controllable here.
        #     preview_clip.write_videofile(str(preview_output_path), codec="libx264", audio=False, logger=None) # Disable audio for speed

        if preview_output_path.exists():
            logging.info(f"Preview clip generated: {preview_output_path}")
            return str(preview_output_path)
        else:
            logging.error(f"Preview clip generation failed (file not created): {preview_output_path}")
            return None

    except Exception as e:
        logging.error(f"Error generating preview clip: {e}", exc_info=True)
        # Clean up potentially corrupted file
        if preview_output_path.exists():
            try:
                preview_output_path.unlink()
            except OSError:
                pass
        return None


def generate_clips(clip_definitions: list, video_path: str, output_dir: str):
    """
    Generates final video clips based on the provided definitions.

    Args:
        clip_definitions: List of dictionaries from prepare_clip_list (using adjusted times).
        video_path: Path to the full source video.
        output_dir: The directory path inside the container to save final clips.

    Returns:
        A list of file paths for the successfully generated clips.
    """
    if not video_path or not Path(video_path).exists():
        logging.error(f"Source video not found at {video_path}")
        return []
    if not clip_definitions:
        logging.warning("No clip definitions provided.")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    generated_files = []
    logging.info(f"Starting clip generation. Total clips to generate: {len(clip_definitions)}")

    try:
        # Load the main video file once
        with VideoFileClip(video_path) as video:
            video_duration = video.duration
            logging.info(f"Source video duration: {video_duration:.2f}s")

            for i, clip_def in enumerate(clip_definitions):
                start_s = clip_def['adjusted_start']
                end_s = clip_def['adjusted_end']
                event_desc = clip_def.get('event_desc', f'clip_{i+1}') # Fallback description

                # --- Validation ---
                if start_s >= end_s:
                    logging.warning(f"Skipping clip {i+1} ('{event_desc}') due to invalid time range: start {start_s:.2f} >= end {end_s:.2f}")
                    continue
                if start_s < 0:
                     logging.warning(f"Adjusting start time for clip {i+1} ('{event_desc}') from {start_s:.2f} to 0")
                     start_s = 0
                if end_s > video_duration:
                     logging.warning(f"Adjusting end time for clip {i+1} ('{event_desc}') from {end_s:.2f} to video duration {video_duration:.2f}")
                     end_s = video_duration
                if start_s >= video_duration:
                     logging.warning(f"Skipping clip {i+1} ('{event_desc}') as start time {start_s:.2f} is beyond video duration {video_duration:.2f}")
                     continue


                # --- Filename Generation ---
                base_filename = sanitize_filename(event_desc)
                # Add timestamp to ensure uniqueness if descriptions are similar
                filename = f"{base_filename}_{int(start_s)}s_{int(end_s)}s.mp4"
                output_filepath = output_path / filename

                logging.info(f"Generating clip {i+1}/{len(clip_definitions)}: '{filename}' ({start_s:.2f}s - {end_s:.2f}s)")

                try:
                    # Extract subclip
                    sub_clip = video.subclip(start_s, end_s)

                    # Write the subclip to the file
                    # Use sensible defaults. Add options for quality/codec if needed later.
                    # logger=None prevents moviepy from printing to stdout/stderr directly
                    sub_clip.write_videofile(str(output_filepath), codec="libx264", audio_codec="aac", logger=None)

                    generated_files.append(str(output_filepath))
                    logging.info(f"Successfully generated: {output_filepath}")

                except Exception as clip_error:
                    logging.error(f"Failed to generate clip {i+1} ('{filename}'): {clip_error}", exc_info=True)
                    # Optionally try to clean up partial file
                    if output_filepath.exists():
                         try:
                             output_filepath.unlink()
                         except OSError:
                              pass
                finally:
                    # Close the subclip to release resources? (MoviePy handles this reasonably well with context manager)
                    # sub_clip.close() # Generally not needed if using 'with VideoFileClip'
                    pass # Ensure loop continues

    except Exception as main_error:
         logging.error(f"Error during batch clip generation: {main_error}", exc_info=True)
    finally:
        # Clean up temporary preview files (optional, could be done elsewhere)
        logging.info("Cleaning up temporary preview files.")
        for item in TEMP_PREVIEW_DIR.iterdir():
            if item.is_file() and item.name.startswith("preview_"):
                 try:
                      item.unlink()
                 except OSError as e:
                      logging.warning(f"Could not remove temp preview {item}: {e}")


    logging.info(f"Clip generation complete. Successfully generated {len(generated_files)} clips.")
    return generated_files 