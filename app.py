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
        st.warning("Please download or upload a video in Step 1 first.") # Adjusted message
        if st.button("Go back to Step 1"):
            st.session_state.current_step = "1. Video Input"
            st.rerun()
    else:
        # --- Display Video for Timestamp Finding --- #
        st.subheader("Video Preview")
        st.info("Use the player to find the time of the first event in your CSV.")
        try:
            st.video(st.session_state.video_file_path)
        except Exception as e:
            st.error(f"Could not display video preview: {e}")
            video_utils.logging.error(f"Video preview error: {e}", exc_info=True)
        # ------------------------------------------- #

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

                # Determine the default selection ONLY if it hasn't been set before in this session
                if 'event_categories_initialized' not in st.session_state:
                    default_selection = [cat for cat in base_event_types if cat in available_categories]
                    st.session_state.event_categories = default_selection # Initialize it
                    st.session_state.event_categories_initialized = True # Mark as initialized
                # Otherwise, use the current value (even if empty)
                current_selection = st.session_state.event_categories

                # The multiselect value is now managed by session state directly
                st.session_state.event_categories = st.multiselect(
                    "Select event categories to find:",
                    options=available_categories,
                    default=current_selection, # Use the potentially modified current value
                    key="event_multiselect"
                )

                if st.button("Prepare Clips"):
                    # Ensure event_categories is up-to-date before preparing
                    # categories_to_prepare = st.session_state.event_categories
                    if st.session_state.event_categories and st.session_state.selected_team:
                        with st.spinner("Preparing clip definitions..."):
                            try:
                                prepared_list = video_utils.prepare_clip_list(
                                    st.session_state.match_df,
                                    st.session_state.event_categories, # Use current state
                                    st.session_state.selected_team
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
        # --- Clip Selection Dropdown ---
        clip_options = {i: f"Clip {i+1}: {clip['event_desc']}" for i, clip in enumerate(st.session_state.prepared_clips)}

        # Initialize selected clip index if not present or invalid
        if 'selected_clip_index' not in st.session_state or st.session_state.selected_clip_index not in clip_options:
            st.session_state.selected_clip_index = 0 # Default to the first clip

        st.session_state.selected_clip_index = st.selectbox(
            "Select Clip to Preview/Adjust:",
            options=list(clip_options.keys()),
            format_func=lambda x: clip_options[x], # Show descriptive names
            index=list(clip_options.keys()).index(st.session_state.selected_clip_index), # Set current index
            key="clip_selector"
        )

        st.divider()

        # --- Display and Adjust Selected Clip ---
        idx = st.session_state.selected_clip_index
        clip_info = st.session_state.prepared_clips[idx]

        # Initialize adjustment state for the selected clip if needed
        if idx not in st.session_state.adjustments:
             st.session_state.adjustments[idx] = {'start_delta': 0.0} # Only track start delta now

        st.subheader(f"Adjusting: {clip_options[idx]}")

        main_cols = st.columns([1, 2]) # Info | Preview + Adjustments

        with main_cols[0]: # Left column for info
            current_start_adj = clip_info['adjusted_start']
            current_end_adj = current_start_adj + 20 # End is always start + 20s
            st.write(f"Original Start: {clip_info['start']:.1f}s")
            st.write(f"**Current Adjusted Start:** {current_start_adj:.1f}s")
            st.write(f"(Clip runs from {current_start_adj:.1f}s to {current_end_adj:.1f}s)")


        with main_cols[1]: # Right column for preview and adjustments
            preview_placeholder = st.empty() # Placeholder for the video player

            # --- Adjustment Buttons --- # Below the preview area now
            adj_cols = st.columns(3) # -5s Start, +5s Start, Update Button
            with adj_cols[0]:
                if st.button("Start -5s", key=f"start_minus_{idx}"):
                    st.session_state.adjustments[idx]['start_delta'] -= 5.0
                    st.rerun() # Rerun to show updated pending caption

            with adj_cols[1]:
                if st.button("Start +5s", key=f"start_plus_{idx}"):
                    st.session_state.adjustments[idx]['start_delta'] += 5.0
                    st.rerun()

            # Display pending adjustments
            pending_start_delta = st.session_state.adjustments[idx]['start_delta']
            st.caption(f"Pending Start Adjustment: {pending_start_delta:+.1f}s")

            # --- Update & Preview Button --- # Col 3
            with adj_cols[2]:
                if st.button("Update & Preview", key=f"update_{idx}"):
                    start_delta = st.session_state.adjustments[idx]['start_delta']
                    # Calculate new start based on *original* start + cumulative delta
                    new_start = clip_info['start'] + start_delta
                    new_end = new_start + 20 # End is always 20s after start

                    # Basic validation (Start >= 0)
                    if new_start >= 0:
                        # Update the *actual* adjusted times in prepared_clips
                        st.session_state.prepared_clips[idx]['adjusted_start'] = new_start
                        # End time is implicitly start + 20s
                        st.success(f"Clip {idx+1} start time updated.")

                        # Reset pending adjustments for this clip
                        st.session_state.adjustments[idx] = {'start_delta': 0.0}

                        # Generate the preview for the *new* applied times
                        with st.spinner(f"Generating preview for clip {idx+1}..."):
                           # ... (keep preview generation try/except block)
                           pass
                        st.rerun()
                    else:
                        st.warning(f"Invalid time range after adjustment ({new_start:.1f}s). Start must be >= 0. Adjustments not applied.")

            # --- Display Preview --- # After buttons
            current_preview_path = st.session_state.preview_paths.get(idx)
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
    st.header("Step 5: Generate and Download Clips")

    if not st.session_state.prepared_clips:
        st.warning("No clips prepared for generation. Go back to previous steps.")
        if st.button("Go back to Step 4"):
            st.session_state.current_step = "4. Preview & Adjust Clips"
            st.rerun()
    else:
        st.info(f"Ready to generate {len(st.session_state.prepared_clips)} clips based on the adjusted times.")
        st.write(f"Clips will be generated and then offered as a single ZIP download.")

        # Combine Generate and Download
        if st.button("✨ Generate & Download All Clips (.zip) ✨", type="primary"):
            generated_files = [] # Keep track of generated files locally
            zip_buffer = io.BytesIO()
            try:
                with st.spinner("Generating clips... This may take a while."):
                    generated_files = video_utils.generate_clips(
                        st.session_state.prepared_clips,
                        st.session_state.video_file_path,
                        st.session_state.output_dir # Temp dir for generation
                    )
                    st.session_state.generated_clips = generated_files # Update state if needed elsewhere

                if not generated_files:
                     st.warning("Clip generation finished, but no clips were successfully created. Check logs.")
                else:
                    st.success(f"{len(generated_files)} clips generated successfully!")
                    # --- Create ZIP --- #
                    with st.spinner("Creating ZIP file..."):
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file: # Use 'w' for new zip
                            for file_path in generated_files:
                                if os.path.exists(file_path):
                                    file_name = os.path.basename(file_path)
                                    zip_file.write(file_path, arcname=file_name)
                                else:
                                    (video_utils.logging if video_utils else print)(f"Warning: File not found when creating ZIP: {file_path}")
                        zip_buffer.seek(0)

                        # --- ZIP Filename --- #
                        zip_filename = "cricket_clips.zip" # Default
                        try: # Safely try to get match details
                             if st.session_state.match_df is not None and not st.session_state.match_df.empty:
                                 match_date_str = "" # Default date string
                                 if 'Date' in st.session_state.match_df.columns:
                                     try: # Try parsing date
                                         match_date_str = pd.to_datetime(st.session_state.match_df['Date'].iloc[0]).strftime('%Y-%m-%d')
                                     except Exception as date_e:
                                         video_utils.logging.warning(f"Could not parse Date column for ZIP filename: {date_e}")
                                         match_date_str = str(st.session_state.match_df['Date'].iloc[0]) # Use raw value if parse fails

                                 match_name = st.session_state.match_df['Match'].iloc[0] if 'Match' in st.session_state.match_df.columns else "Match"
                                 team_name = st.session_state.selected_team or "SelectedTeam"
                                 safe_match = video_utils.sanitize_filename(str(match_name))
                                 safe_team = video_utils.sanitize_filename(str(team_name))
                                 safe_date = video_utils.sanitize_filename(match_date_str)
                                 zip_filename = f"{safe_date}_{safe_match}_{safe_team}_clips.zip".replace("__", "_")
                        except Exception as fn_e:
                             video_utils.logging.warning(f"Could not generate detailed ZIP filename: {fn_e}")

                        # --- Offer ZIP Download --- #
                        st.download_button(
                            label="Download ZIP Now", # Button appears after generation
                            data=zip_buffer,
                            file_name=zip_filename,
                            mime="application/zip",
                            key="download_final_zip"
                        )
                        st.info("Click the button above to download your ZIP file.")

            except Exception as e:
                 st.error(f"An error occurred during clip generation or ZIP creation: {e}")
                 video_utils.logging.error(f"Generate/ZIP exception: {e}", exc_info=True)

# --- Fallback for unknown state ---
else:
    st.error("Invalid step selected.")
    st.session_state.current_step = "1. Video Input"
    st.button("Reset to Step 1")
    st.rerun() 