# Placeholder for Streamlit application code 

import io  # Import io
import os  # Add os import
import shutil  # Import shutil
import zipfile  # Import zipfile
from pathlib import Path  # Import Path

import pandas as pd
import streamlit as st
from moviepy import VideoFileClip  # Corrected import

import video_utils  # Import the backend module

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Cricket Clip Generator")

# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if 'current_step' not in st.session_state:
    st.session_state.current_step = "1. Video Input"
if 'video_url' not in st.session_state:
    st.session_state.video_url = ""
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = "/app/output" # Default output dir inside container
if 'video_file_path' not in st.session_state:
    st.session_state.video_file_path = None
if 'video_metadata' not in st.session_state:
    st.session_state.video_metadata = None
if 'csv_file' not in st.session_state:
    st.session_state.csv_file = None
if 'match_df' not in st.session_state:
    st.session_state.match_df = None
if 'first_event_timestamp' not in st.session_state:
    st.session_state.first_event_timestamp = {"hours": 0, "minutes": 0, "seconds": 0}
if 'event_categories' not in st.session_state:
    st.session_state.event_categories = [] # Store selected categories
if 'prepared_clips' not in st.session_state:
    st.session_state.prepared_clips = [] # List of dicts {event_desc: "...", start: sec, end: sec, adjusted_start: sec, adjusted_end: sec}
if 'generated_clips' not in st.session_state:
    st.session_state.generated_clips = [] # List of final clip file paths
if 'preview_paths' not in st.session_state:
    st.session_state.preview_paths = {} # Store preview paths {clip_index: path}
if 'adjustments' not in st.session_state:
    st.session_state.adjustments = {} # {clip_index: {'start_delta': 0, 'end_delta': 0}}


# --- Sidebar Navigation ---
st.sidebar.title("Workflow Steps")
steps = [
    "1. Video Input",
    "2. Timestamp & CSV",
    "3. Define Events",
    "4. Preview & Adjust Clips",
    "5. Generate Clips",
]
st.session_state.current_step = st.sidebar.radio(
    "Go to step:", steps, index=steps.index(st.session_state.current_step)
)

# --- Main Page Content ---
st.title("Cricket Clip Generator")

# =========================
# Step 1: Video Input
# =========================
if st.session_state.current_step == "1. Video Input":
    st.header("Step 1: Video Input")

    input_method = st.radio("Choose Input Method:", ("YouTube URL", "Upload Video File"), horizontal=True)

    # --- YouTube Download --- #
    if input_method == "YouTube URL":
        st.subheader("YouTube Video")
        st.session_state.video_url = st.text_input("Enter YouTube video URL:", st.session_state.video_url)

        st.subheader("Output Directory (within Docker)")
        st.session_state.output_dir = st.text_input("Specify Internal Output Directory:", st.session_state.output_dir)
        st.caption(f"Clips will be temporarily generated inside the container at: {st.session_state.output_dir}")
        st.caption("You will be able to download the generated clips directly from the app in the final step.")

        if st.button("Download Video"):
            if st.session_state.video_url and st.session_state.output_dir:
                with st.spinner("Downloading video..."):
                    try:
                        video_path, metadata = video_utils.download_video(st.session_state.video_url)
                        if video_path and metadata:
                            st.session_state.video_file_path = video_path
                            st.session_state.video_metadata = metadata
                            st.success(f"Video '{st.session_state.video_metadata['title']}' downloaded.")
                            st.session_state.current_step = "2. Timestamp & CSV" # Move to next step
                            # Clear potentially stale data from previous runs
                            st.session_state.match_df = None
                            st.session_state.prepared_clips = []
                            st.session_state.generated_clips = []
                            st.session_state.preview_paths = {}
                            st.rerun()
                        else:
                            st.error("Failed to download video. Check URL and logs.")
                    except Exception as e:
                        st.error(f"An error occurred during download: {e}")
                        video_utils.logging.error(f"Download exception: {e}", exc_info=True)
            else:
                st.warning("Please provide both a YouTube URL and an Output Directory.")

    # --- File Upload --- #
    elif input_method == "Upload Video File":
        st.subheader("Upload Video File")
        uploaded_video_file = st.file_uploader(
            "Choose a video file (MP4, MOV, AVI...)",
            type=['mp4', 'mov', 'avi', 'mkv'] # Add common video types
        )

        st.subheader("Output Directory (within Docker)")
        st.session_state.output_dir = st.text_input("Specify Internal Output Directory:", st.session_state.output_dir)
        st.caption(f"Clips will be temporarily generated inside the container at: {st.session_state.output_dir}")
        st.caption("You will be able to download the generated clips directly from the app in the final step.")

        if uploaded_video_file is not None:
            if st.button("Load Uploaded Video"):
                 if not st.session_state.output_dir:
                     st.warning("Please specify an Output Directory.")
                 else:
                    with st.spinner("Processing uploaded video..."):
                        try:
                            # Define temp path
                            temp_dir = Path(video_utils.TEMP_VIDEO_DIR)
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            # Clean up old temp files
                            for item in temp_dir.iterdir():
                                if item.is_file(): item.unlink()
                            # Create a safe filename
                            safe_filename = video_utils.sanitize_filename(uploaded_video_file.name)
                            temp_video_path = temp_dir / safe_filename

                            # Save the uploaded file to the temporary path
                            with open(temp_video_path, "wb") as f:
                                shutil.copyfileobj(uploaded_video_file, f)
                            video_utils.logging.info(f"Uploaded video saved to: {temp_video_path}")

                            # Get metadata (duration)
                            duration = 0
                            try:
                                with VideoFileClip(str(temp_video_path)) as clip:
                                    duration = clip.duration
                            except Exception as meta_err:
                                 video_utils.logging.warning(f"Could not read duration using moviepy: {meta_err}")
                                 st.warning("Could not automatically determine video duration.")

                            st.session_state.video_file_path = str(temp_video_path)
                            st.session_state.video_metadata = {
                                "title": uploaded_video_file.name, # Use original filename for title
                                "duration": duration,
                                "publish_date": "N/A" # No publish date for uploads
                            }

                            st.success(f"Video '{st.session_state.video_metadata['title']}' loaded.")
                            st.session_state.current_step = "2. Timestamp & CSV" # Move to next step
                            # Clear potentially stale data from previous runs
                            st.session_state.match_df = None
                            st.session_state.prepared_clips = []
                            st.session_state.generated_clips = []
                            st.session_state.preview_paths = {}
                            st.rerun()

                        except Exception as e:
                            st.error(f"An error occurred loading the uploaded video: {e}")
                            video_utils.logging.error(f"Upload processing exception: {e}", exc_info=True)

    # --- Display Video Info (Common to both methods) --- #
    if st.session_state.video_file_path and st.session_state.video_metadata:
        st.subheader("Video Information")
        st.write(f"**Title:** {st.session_state.video_metadata.get('title', 'N/A')}")
        duration_sec = st.session_state.video_metadata.get('duration', 0)
        st.write(f"**Duration:** {int(duration_sec // 3600):02d}:{int((duration_sec % 3600) // 60):02d}:{int(duration_sec % 60):02d}")
        st.write(f"**Publish Date:** {st.session_state.video_metadata.get('publish_date', 'N/A')}")
        # Optional: Display video preview if needed, but might be slow for large files initially
        # st.video(st.session_state.video_file_path)

    st.sidebar.markdown("---")
    if st.sidebar.button("Next Step: Timestamp & CSV"):
        if st.session_state.video_file_path:
            st.session_state.current_step = "2. Timestamp & CSV"
            st.rerun()
        else:
            st.sidebar.warning("Download video first.")


# =========================
# Step 2: Timestamp & CSV
# =========================
elif st.session_state.current_step == "2. Timestamp & CSV":
    st.header("Step 2: Timestamp & CSV Input")

    if not st.session_state.video_file_path:
        st.warning("Please download a video in Step 1 first.")
        if st.button("Go back to Step 1"):
            st.session_state.current_step = "1. Video Input"
            st.rerun()
    else:
        st.info(f"Video Loaded: {st.session_state.video_metadata.get('title', 'Unknown')}")

        st.subheader("First Event Timestamp in Video")
        st.write("Enter the exact time in the video when the *first* event listed in your CSV occurs.")
        t_cols = st.columns(3)
        with t_cols[0]:
            st.session_state.first_event_timestamp['hours'] = st.number_input("Hours", min_value=0, max_value=23, step=1, value=st.session_state.first_event_timestamp['hours'])
        with t_cols[1]:
            st.session_state.first_event_timestamp['minutes'] = st.number_input("Minutes", min_value=0, max_value=59, step=1, value=st.session_state.first_event_timestamp['minutes'])
        with t_cols[2]:
            st.session_state.first_event_timestamp['seconds'] = st.number_input("Seconds", min_value=0, max_value=59, step=1, value=st.session_state.first_event_timestamp['seconds'])

        st.write(f"Selected Time: {st.session_state.first_event_timestamp['hours']:02d}:{st.session_state.first_event_timestamp['minutes']:02d}:{st.session_state.first_event_timestamp['seconds']:02d}")

        st.subheader("Match Data CSV")
        # Use a key to persist the uploaded file object across reruns within this step
        uploaded_csv = st.file_uploader("Upload the match data CSV file:", type=["csv"], key="csv_uploader")
        if uploaded_csv is not None:
             st.session_state.csv_file = uploaded_csv # Store if successful upload

        if st.button("Load and Process CSV"):
            if st.session_state.csv_file is not None:
                first_event_video_seconds = (st.session_state.first_event_timestamp['hours'] * 3600 +
                                            st.session_state.first_event_timestamp['minutes'] * 60 +
                                            st.session_state.first_event_timestamp['seconds'])
                with st.spinner("Processing CSV..."):
                    try:
                        # Make sure the file pointer is at the beginning if reusing the object
                        st.session_state.csv_file.seek(0)
                        df = video_utils.load_and_process_csv(st.session_state.csv_file, first_event_video_seconds)
                        if df is not None:
                            st.session_state.match_df = df
                            st.success("CSV processed successfully.")
                            st.session_state.current_step = "3. Define Events" # Move to next step
                            # Clear downstream stale data
                            st.session_state.prepared_clips = []
                            st.session_state.generated_clips = []
                            st.session_state.preview_paths = {}
                            st.rerun()
                        else:
                            st.error("Failed to process CSV. Check file format, required columns (Timestamp, Batter, Runs, Wicket), and logs.")
                    except ValueError as ve:
                        st.error(f"CSV Processing Error: {ve}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during CSV processing: {e}")
                        video_utils.logging.error(f"CSV processing exception: {e}", exc_info=True)
            else:
                st.warning("Please upload a CSV file first.")

        if st.session_state.match_df is not None:
            st.subheader("Processed Match Data Preview")
            st.dataframe(st.session_state.match_df.head())

    st.sidebar.markdown("---")
    if st.sidebar.button("Next Step: Define Events"):
        if st.session_state.match_df is not None:
            st.session_state.current_step = "3. Define Events"
            st.rerun()
        else:
            st.sidebar.warning("Load CSV first.")


# =========================
# Step 3: Define Events
# =========================
elif st.session_state.current_step == "3. Define Events":
    st.header("Step 3: Define Event Categories for Clips")

    if st.session_state.match_df is None:
        st.warning("Please load video and CSV data in previous steps.")
        if st.button("Go back to Step 2"):
            st.session_state.current_step = "2. Timestamp & CSV"
            st.rerun()
    else:
        try:
            # --- Team Selection ---
            st.subheader("Select Team for Clips")
            if 'Batting Team' not in st.session_state.match_df.columns:
                st.error("CSV is missing the required 'Batting Team' column for team selection.")
            else:
                teams = sorted(list(st.session_state.match_df['Batting Team'].dropna().unique()))
                if not teams:
                    st.warning("Could not find any team names in the 'Batting Team' column.")
                    st.session_state.selected_team = None
                elif 'selected_team' not in st.session_state or st.session_state.selected_team not in teams:
                    # Initialize or reset if previous selection is invalid
                     st.session_state.selected_team = teams[0] # Default to first team

                # Use radio buttons for team selection
                st.session_state.selected_team = st.radio(
                    "Select the team you want to generate clips *for*:",
                    options=teams,
                    index=teams.index(st.session_state.selected_team) if st.session_state.selected_team in teams else 0,
                    horizontal=True,
                    key="team_selector" # Add key
                )
                st.info(f"Generating clips for: **{st.session_state.selected_team}** (e.g., 4s/6s hit by this team, wickets taken *against* this team).")


            # --- Event Category Definition ---
            if st.session_state.selected_team: # Only show event selection if a team is selected
                st.subheader("Define Event Categories")
                # Dynamically generate event types based on actual data and selected team
                team_df = st.session_state.match_df[st.session_state.match_df['Batting Team'] == st.session_state.selected_team]
                all_batters_on_team = sorted(list(team_df['Batter'].dropna().unique()))

                # Base events remain the same conceptually
                base_event_types = ["All 4s", "All 6s", "All Wickets"]

                # Player events are now filtered by the selected team's players
                player_event_types = []
                for player in all_batters_on_team:
                    player_event_types.append(f"{player} - 4s")
                    player_event_types.append(f"{player} - 6s")
                    # Wickets taken by player (handled in backend logic based on Bowler)
                    # Can still offer the category selection here
                    player_event_types.append(f"{player} - Wickets")

                available_categories = base_event_types + player_event_types

                # Determine the default selection
                # If event_categories already has items (e.g., from previous run/selection), use them.
                # Otherwise, default to the base_event_types.
                current_selection = st.session_state.get('event_categories', [])
                # Ensure defaults are actually present in the available options
                valid_defaults = [d for d in base_event_types if d in available_categories]
                default_selection = current_selection if current_selection else valid_defaults

                st.session_state.event_categories = st.multiselect(
                    "Select event categories to find:",
                    options=available_categories,
                    # default=st.session_state.event_categories, # Old default
                    default=default_selection, # New default logic
                    key="event_multiselect" # Add key
                )

                if st.button("Prepare Clips"):
                    if st.session_state.event_categories and st.session_state.selected_team:
                        with st.spinner("Preparing clip definitions..."):
                            try:
                                prepared_list = video_utils.prepare_clip_list(
                                    st.session_state.match_df,
                                    st.session_state.event_categories,
                                    st.session_state.selected_team # Pass selected team
                                )
                                st.session_state.prepared_clips = prepared_list
                                # Clear old previews
                                st.session_state.preview_paths = {}
                                if prepared_list:
                                    st.success(f"{len(prepared_list)} clip definitions prepared.")
                                    st.session_state.current_step = "4. Preview & Adjust Clips"
                                    st.rerun()
                                else:
                                    st.warning("No events found matching the selected categories.")
                            except ValueError as ve:
                                st.error(f"Error preparing clips: {ve}")
                            except Exception as e:
                                st.error(f"An unexpected error occurred while preparing clips: {e}")
                                video_utils.logging.error(f"Prepare clips exception: {e}", exc_info=True)
                    elif not st.session_state.selected_team:
                         st.warning("Please select a team first.")
                    else: # No categories selected
                        st.warning("Please select at least one event category.")
            else:
                st.warning("Cannot define event categories until a team is selected (or team names are found in CSV).")

        except KeyError as ke:
             st.error(f"Missing required column in CSV: {ke}. Ensure 'Batter', 'Runs', 'Wicket', 'Batting Team' exist.")
        except Exception as e:
             st.error(f"An error occurred setting up event definitions: {e}")
             video_utils.logging.error(f"Event setup exception: {e}", exc_info=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("Next Step: Preview Clips"):
        if st.session_state.prepared_clips:
            st.session_state.current_step = "4. Preview & Adjust Clips"
            st.rerun()
        else:
            st.sidebar.warning("Prepare clips first.")


# =========================
# Step 4: Preview & Adjust
# =========================
elif st.session_state.current_step == "4. Preview & Adjust Clips":
    st.header("Step 4: Preview and Adjust Clips")

    if not st.session_state.prepared_clips:
        st.warning("No clips prepared yet. Please go back to Step 3.")
        if st.button("Go back to Step 3"):
            st.session_state.current_step = "3. Define Events"
            st.rerun()
    else:
        st.info("Review the prepared clips. Use the buttons to adjust start/end times by 5 seconds, then update the preview.")

        for i, clip_info in enumerate(st.session_state.prepared_clips):
            # Ensure current clip has an entry in adjustments state
            if i not in st.session_state.adjustments:
                st.session_state.adjustments[i] = {'start_delta': 0.0, 'end_delta': 0.0}

            st.subheader(f"Clip {i+1}: {clip_info['event_desc']}")
            main_cols = st.columns([2, 3]) # Info | Preview + Adjustments

            with main_cols[0]: # Left column for info
                # Display times based on current session state including pending adjustments
                current_start_adj = clip_info['adjusted_start']
                current_end_adj = clip_info['adjusted_end']
                st.write(f"Original Range: {clip_info['start']:.1f}s - {clip_info['end']:.1f}s")
                st.write(f"**Current Adjusted Range:** {current_start_adj:.1f}s - {current_end_adj:.1f}s")

            with main_cols[1]: # Right column for preview and adjustments
                preview_placeholder = st.empty() # Placeholder for the video player

                # --- Adjustment Buttons ---
                adj_cols = st.columns(5) # -5s Start, +5s Start, -5s End, +5s End, Update Button
                with adj_cols[0]:
                    if st.button("Start -5s", key=f"start_minus_{i}"):
                        st.session_state.adjustments[i]['start_delta'] -= 5.0
                        # No rerun, just update state value for next "Update" click
                        # st.success("-5s added to start adjustment.") # Feedback can be noisy
                        st.rerun() # Rerun needed to show updated pending caption

                with adj_cols[1]:
                    if st.button("Start +5s", key=f"start_plus_{i}"):
                        st.session_state.adjustments[i]['start_delta'] += 5.0
                        # st.success("+5s added to start adjustment.")
                        st.rerun()

                with adj_cols[2]:
                    if st.button("End -5s", key=f"end_minus_{i}"):
                        st.session_state.adjustments[i]['end_delta'] -= 5.0
                        # st.success("-5s added to end adjustment.")
                        st.rerun()

                with adj_cols[3]:
                     if st.button("End +5s", key=f"end_plus_{i}"):
                        st.session_state.adjustments[i]['end_delta'] += 5.0
                        # st.success("+5s added to end adjustment.")
                        st.rerun()

                # Display pending adjustments
                pending_start_delta = st.session_state.adjustments[i]['start_delta']
                pending_end_delta = st.session_state.adjustments[i]['end_delta']
                st.caption(f"Pending Adjustment: Start {pending_start_delta:+.1f}s, End {pending_end_delta:+.1f}s")

                # --- Update & Preview Button --- # Col 5
                with adj_cols[4]:
                    if st.button("Update & Preview", key=f"update_{i}"):
                        # Apply the *cumulative pending* adjustments from state
                        start_delta = st.session_state.adjustments[i]['start_delta']
                        end_delta = st.session_state.adjustments[i]['end_delta']

                        # Calculate new times based on *original* start/end + cumulative deltas
                        new_start = clip_info['start'] + start_delta
                        new_end = clip_info['end'] + end_delta

                        # Basic validation
                        if new_start < new_end and new_start >= 0:
                            # Update the *actual* adjusted times in prepared_clips
                            st.session_state.prepared_clips[i]['adjusted_start'] = new_start
                            st.session_state.prepared_clips[i]['adjusted_end'] = new_end
                            st.success(f"Clip {i+1} time range updated.")

                            # Reset pending adjustments for this clip after applying
                            st.session_state.adjustments[i] = {'start_delta': 0.0, 'end_delta': 0.0}

                            # Generate the preview for the *new* applied times
                            with st.spinner(f"Generating preview for clip {i+1}..."):
                                try:
                                    preview_path = video_utils.generate_clip_preview(
                                        st.session_state.video_file_path,
                                        new_start,
                                        new_end,
                                        i
                                    )
                                    st.session_state.preview_paths[i] = preview_path # Store path even if None
                                    if not preview_path:
                                         st.warning("Could not generate preview clip.")
                                except Exception as e:
                                     st.session_state.preview_paths[i] = None
                                     st.error(f"Error generating preview: {e}")
                                     video_utils.logging.error(f"Preview generation exception: {e}", exc_info=True)
                            st.rerun() # Rerun to update displayed times and video player
                        else:
                            st.warning(f"Invalid time range after adjustment ({new_start:.1f}s - {new_end:.1f}s). Start must be < End and >= 0. Adjustments not applied.")

                # --- Display Preview --- # Outside adjustment columns
                current_preview_path = st.session_state.preview_paths.get(i)
                if current_preview_path:
                    if os.path.exists(current_preview_path):
                        try:
                            # Read the file content and pass bytes to st.video
                            with open(current_preview_path, "rb") as f:
                                video_bytes = f.read()
                            preview_placeholder.video(video_bytes, format="video/mp4")
                        except Exception as e:
                             preview_placeholder.error(f"Error reading/displaying preview video: {e}")
                             video_utils.logging.error(f"Preview display error: {e}", exc_info=True)
                    else:
                        preview_placeholder.warning(f"Preview file not found at {current_preview_path}. Regenerate preview.")
                else:
                    preview_placeholder.caption(f"Preview for {current_start_adj:.1f}s - {current_end_adj:.1f}s (click 'Update & Preview')")

            st.divider()

    st.sidebar.markdown("---")
    if st.sidebar.button("Next Step: Generate Clips"):
        if st.session_state.prepared_clips:
            st.session_state.current_step = "5. Generate Clips"
            st.rerun()
        else:
            st.sidebar.warning("No clips to generate.")

# =========================
# Step 5: Generate Clips
# =========================
elif st.session_state.current_step == "5. Generate Clips":
    st.header("Step 5: Generate Final Clips")

    if not st.session_state.prepared_clips:
        st.warning("No clips prepared for generation. Go back to previous steps.")
        if st.button("Go back to Step 4"):
            st.session_state.current_step = "4. Preview & Adjust Clips"
            st.rerun()
    else:
        st.info(f"Ready to generate {len(st.session_state.prepared_clips)} clips based on the adjusted times.")
        st.write(f"Clips will be saved temporarily to: `{st.session_state.output_dir}` before download.") # Adjusted text slightly

        if st.button("✨ Generate All Clips ✨", type="primary"):
            with st.spinner("Generating clips... This may take a while."):
                try:
                    # Pass the list of clip definitions (with adjusted times)
                    result_paths = video_utils.generate_clips(
                        st.session_state.prepared_clips,
                        st.session_state.video_file_path,
                        st.session_state.output_dir
                    )
                    st.session_state.generated_clips = result_paths
                    if result_paths:
                        st.success(f"{len(result_paths)} clips generated successfully!")
                    else:
                        st.warning("Clip generation finished, but no clips were successfully created. Check logs.")
                    # Clear previews after final generation
                    st.session_state.preview_paths = {}
                    # No rerun needed here, download buttons will appear below
                except Exception as e:
                    st.error(f"An error occurred during clip generation: {e}")
                    video_utils.logging.error(f"Clip generation exception: {e}", exc_info=True)

        if st.session_state.generated_clips:
            st.subheader("Download Generated Clips")

            # --- Option 1: Individual Downloads (Keep existing logic) ---
            st.write("Download individual clips:")
            for clip_path in st.session_state.generated_clips:
                clip_filename = os.path.basename(clip_path)
                try:
                    # Read the actual generated clip file for download
                    with open(clip_path, "rb") as fp:
                        st.download_button(
                            label=f"Download {clip_filename}",
                            data=fp,
                            file_name=clip_filename,
                            mime="video/mp4",
                            key=f"download_{clip_filename}" # Add key for robustness
                        )
                except FileNotFoundError:
                     st.error(f"Error preparing download: File not found at {clip_path}. It might have been deleted or generation failed partially.")
                except Exception as e:
                     st.error(f"Error preparing download for {clip_filename}: {e}")
                     video_utils.logging.error(f"Download button preparation error for {clip_filename}: {e}", exc_info=True)

            st.divider()

            # --- Option 2: Download All as ZIP ---
            st.write("Download all clips as a single ZIP file:")
            # Create ZIP in memory
            zip_buffer = io.BytesIO()
            try:
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for file_path in st.session_state.generated_clips:
                         if os.path.exists(file_path):
                             file_name = os.path.basename(file_path)
                             zip_file.write(file_path, arcname=file_name) # arcname avoids storing full path
                         else:
                             # Use logging from video_utils if available, or print
                             (video_utils.logging if video_utils else print)(f"Warning: File not found when creating ZIP: {file_path}")

                zip_buffer.seek(0) # Rewind buffer

                # Suggest a filename for the ZIP
                zip_filename = "cricket_clips.zip"
                if st.session_state.video_metadata and 'title' in st.session_state.video_metadata:
                    sanitized_title = video_utils.sanitize_filename(st.session_state.video_metadata['title'])
                    zip_filename = f"{sanitized_title}_clips.zip"

                st.download_button(
                    label="Download All Clips (.zip)",
                    data=zip_buffer,
                    file_name=zip_filename,
                    mime="application/zip",
                    key="download_all_zip"
                )
            except Exception as e:
                 st.error(f"Could not create ZIP file: {e}")
                 video_utils.logging.error(f"ZIP creation error: {e}", exc_info=True)


# --- Fallback for unknown state ---
else:
    st.error("Invalid step selected.")
    st.session_state.current_step = "1. Video Input"
    st.button("Reset to Step 1")
    st.rerun() 