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
if 'previews_generated' not in st.session_state:
    st.session_state.previews_generated = False # Flag to track if previews are ready
if 'last_prepared_clips_hash' not in st.session_state:
    st.session_state.last_prepared_clips_hash = None # For detecting changes
if 'event_categories_initialized' not in st.session_state:
    st.session_state.event_categories_initialized = False # Flag for multiselect default


# --- Sidebar Navigation ---
st.sidebar.title("Workflow Steps")
steps = [
    "1. Video Input",
    "2. Timestamp & CSV",
    "3. Define Events",
    "4. Preview & Adjust Clips",
    # "5. Generate Clips", # Removed
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
                            st.session_state.previews_generated = False # Reset preview flag
                            st.session_state.last_prepared_clips_hash = None # Reset hash
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
                            st.session_state.previews_generated = False # Reset preview flag
                            st.session_state.last_prepared_clips_hash = None # Reset hash
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
            st.sidebar.warning("Download or upload video first.") # Updated warning


# =========================
# Step 2: Timestamp & CSV
# =========================
elif st.session_state.current_step == "2. Timestamp & CSV":
    st.header("Step 2: Timestamp & CSV Input")

    if not st.session_state.video_file_path:
        st.warning("Please download or upload a video in Step 1 first.")
        if st.button("Go back to Step 1"):
            st.session_state.current_step = "1. Video Input"
            st.rerun()
    else:
        # --- Display Video for Timestamp Finding --- #
        st.subheader("Video Preview")
        # Display YouTube video directly if URL is available (more reliable for timestamp finding)
        if st.session_state.video_url:
            st.info("Use the player to find the time of the first event in your CSV.")
            try:
                st.video(st.session_state.video_url)
            except Exception as e:
                st.error(f"Could not display YouTube video preview: {e}")
                video_utils.logging.error(f"YouTube preview error: {e}", exc_info=True)
        # If file was uploaded, try displaying the local file path
        elif st.session_state.video_file_path:
             st.info("Use the player to find the time of the first event in your CSV.")
             try:
                 st.video(st.session_state.video_file_path)
             except Exception as e:
                 st.error(f"Could not display uploaded video preview: {e}")
                 video_utils.logging.error(f"Uploaded video preview error: {e}", exc_info=True)
        # ------------------------------------------- #

        st.info(f"Video Source Ready: {st.session_state.video_metadata.get('title', 'Unknown')}")

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
                            st.session_state.previews_generated = False # Reset preview flag
                            st.session_state.last_prepared_clips_hash = None # Reset hash
                            st.rerun()
                        else:
                            # Error message already shown by load_and_process_csv if it returns None
                            st.error("Failed to process CSV. Check logs for details.")
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
                st.session_state.selected_team = None # Ensure selected_team is None
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

                # Initialize defaults using the flag approach
                if not st.session_state.event_categories_initialized:
                    # Set initial default only once per session/step entry
                    initial_defaults = [cat for cat in base_event_types if cat in available_categories]
                    st.session_state.event_categories = initial_defaults
                    st.session_state.event_categories_initialized = True
                    video_utils.logging.info(f"Initialized event_categories state with defaults: {initial_defaults}")

                # Ensure current state value is list (might be None if reset elsewhere)
                current_selection = st.session_state.get('event_categories', [])

                # Use the multiselect widget
                selected_categories = st.multiselect(
                    "Select event categories to find:",
                    options=available_categories,
                    default=current_selection,
                    key="event_multiselect"
                )

                # Update the session state *after* getting the result
                # This comparison prevents unnecessary state updates if the selection didn't change
                if selected_categories != st.session_state.event_categories:
                     st.session_state.event_categories = selected_categories
                     # No rerun needed here, state is updated for the *next* run automatically

                if st.button("Prepare Clips"):
                    # Reset initialized flag so defaults apply if user comes back to this step
                    st.session_state.event_categories_initialized = False
                    if st.session_state.event_categories and st.session_state.selected_team:
                        with st.spinner("Preparing clip definitions..."):
                            try:
                                prepared_list = video_utils.prepare_clip_list(
                                    st.session_state.match_df,
                                    st.session_state.event_categories, # Use current state
                                    st.session_state.selected_team
                                )
                                st.session_state.prepared_clips = prepared_list
                                # Clear old previews and reset flags
                                st.session_state.preview_paths = {}
                                st.session_state.previews_generated = False
                                st.session_state.last_prepared_clips_hash = None # Reset hash tracking
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
    # Remove the old sidebar button for Step 4 -> Step 5
    # if st.sidebar.button("Next Step: Preview Clips"):
    #     if st.session_state.prepared_clips:
    #         st.session_state.current_step = "4. Preview & Adjust Clips"
    #         st.rerun()
    #     else:
    #         st.sidebar.warning("Prepare clips first.")


# =========================
# Step 4: Preview, Adjust & Generate
# =========================
elif st.session_state.current_step == "4. Preview & Adjust Clips":
    st.header("Step 4: Preview, Adjust & Generate Clips") # Renamed header

    if not st.session_state.prepared_clips:
        st.warning("No clips prepared yet. Please go back to Step 3.")
        if st.button("Go back to Step 3"):
            st.session_state.current_step = "3. Define Events"
            # Reset flags when explicitly going back
            st.session_state.previews_generated = False
            st.session_state.preview_paths = {}
            st.session_state.last_prepared_clips_hash = None
            st.rerun()
    else:
        # Button to generate all previews initially
        if not st.session_state.previews_generated:
            st.info(f"{len(st.session_state.prepared_clips)} clip definitions ready. Generate previews to view and adjust.")
            if st.button("📊 Generate All Previews", key="gen_all_previews"):
                with st.spinner(f"Generating {len(st.session_state.prepared_clips)} previews... This might take a while."):
                    success_count = 0
                    preview_paths_temp = {} # Store paths temporarily
                    for i, clip_info in enumerate(st.session_state.prepared_clips):
                         # Generate preview using the current adjusted times
                         start_s = clip_info['adjusted_start']
                         end_s = start_s + 20 # Fixed 20s duration
                         try:
                             preview_path = video_utils.generate_clip_preview(
                                 st.session_state.video_file_path,
                                 start_s,
                                 end_s,
                                 i
                             )
                             preview_paths_temp[i] = preview_path # Store path
                             if preview_path:
                                 success_count += 1
                         except Exception as e:
                              preview_paths_temp[i] = None
                              st.error(f"Error generating preview for clip {i+1}: {e}")
                              video_utils.logging.error(f"Preview generation exception for clip {i+1}: {e}", exc_info=True)
                    st.session_state.preview_paths = preview_paths_temp # Update state
                    st.session_state.previews_generated = True
                    st.success(f"Generated {success_count} / {len(st.session_state.prepared_clips)} previews.")
                    st.rerun() # Rerun to show the previews

        else:
            # Display all clips with previews and adjustment buttons
            st.info("Review previews. Use buttons to adjust start time by 10s. Changes trigger individual preview updates.")

            for i, clip_info in enumerate(st.session_state.prepared_clips):
                st.subheader(f"Clip {i+1}: {clip_info['event_desc']}")

                # --- Main Layout: Preview | Info & Adjust --- #
                layout_cols = st.columns([3, 2]) # Preview | Info/Adjust

                with layout_cols[0]: # Preview Area
                     preview_placeholder = st.empty()
                     current_preview_path = st.session_state.preview_paths.get(i)
                     if current_preview_path:
                         video_utils.logging.info(f"Attempting to display preview from path: {current_preview_path}")
                         if os.path.exists(current_preview_path):
                             try:
                                 video_utils.logging.info(f"Preview file exists. Reading bytes...")
                                 with open(current_preview_path, "rb") as f:
                                     video_bytes = f.read()
                                 video_utils.logging.info(f"Read {len(video_bytes)} bytes for preview.")
                                 if len(video_bytes) > 1024: # Basic check
                                     preview_placeholder.video(video_bytes, format="video/mp4")
                                     video_utils.logging.info("Called preview_placeholder.video()")
                                 else:
                                     preview_placeholder.warning("Preview file invalid (too small or empty).")
                                     video_utils.logging.warning(f"Preview file empty or too small: {current_preview_path}")
                             except Exception as e:
                                  preview_placeholder.error(f"Error reading/displaying preview video: {e}")
                                  video_utils.logging.error(f"Preview display error: {e}", exc_info=True)
                         else:
                              preview_placeholder.warning(f"Preview file missing at {current_preview_path}.")
                              video_utils.logging.warning(f"Preview file path exists in state, but file not found at: {current_preview_path}")
                     else:
                         preview_placeholder.caption("Preview not generated or failed.")

                with layout_cols[1]: # Info & Adjust Area
                    # Info
                    current_start_adj = clip_info['adjusted_start']
                    current_end_adj = current_start_adj + 20
                    st.write(f"Original Start: {clip_info['start']:.1f}s")
                    st.write(f"**Current Start:** {current_start_adj:.1f}s")
                    st.write(f"(Clip runs {current_start_adj:.1f}s - {current_end_adj:.1f}s)")

                    # Adjustment Buttons (Change to 10s)
                    st.write("Adjust Start Time:")
                    adj_cols = st.columns(2) # -10s Start, +10s Start
                    with adj_cols[0]:
                        if st.button("Start -10s", key=f"start_minus_{i}"):
                            new_start = clip_info['adjusted_start'] - 10.0
                            if new_start >= 0:
                                st.session_state.prepared_clips[i]['adjusted_start'] = new_start
                                # Regenerate just this preview
                                with st.spinner(f"Updating preview for Clip {i+1}..."): # Add spinner
                                    new_end = new_start + 20
                                    try:
                                        new_preview_path = video_utils.generate_clip_preview(
                                            st.session_state.video_file_path, new_start, new_end, i
                                        )
                                        st.session_state.preview_paths[i] = new_preview_path # Update path
                                        if not new_preview_path: st.warning("Preview update failed.")
                                    except Exception as e:
                                        st.session_state.preview_paths[i] = None
                                        st.error(f"Error updating preview: {e}")
                                st.rerun()
                            else:
                                st.warning("Cannot adjust start time below 0s.")

                    with adj_cols[1]:
                        if st.button("Start +10s", key=f"start_plus_{i}"):
                            new_start = clip_info['adjusted_start'] + 10.0
                            # Add check against video duration if available?
                            video_duration = st.session_state.video_metadata.get('duration')
                            if video_duration and new_start >= video_duration:
                                 st.warning(f"Cannot adjust start time beyond video duration ({video_duration:.1f}s).")
                            else:
                                st.session_state.prepared_clips[i]['adjusted_start'] = new_start
                                # Regenerate just this preview
                                with st.spinner(f"Updating preview for Clip {i+1}..."): # Add spinner
                                     new_end = new_start + 20
                                     try:
                                        new_preview_path = video_utils.generate_clip_preview(
                                            st.session_state.video_file_path, new_start, new_end, i
                                        )
                                        st.session_state.preview_paths[i] = new_preview_path # Update path
                                        if not new_preview_path: st.warning("Preview update failed.")
                                     except Exception as e:
                                        st.session_state.preview_paths[i] = None
                                        st.error(f"Error updating preview: {e}")
                                st.rerun()
                st.divider() # Divider between clips

            # --- Final Generate & Download Button --- #
            st.divider()
            st.header("Generate Final Output")
            if st.button("✨ Generate Final Clips & Download ZIP ✨", type="primary", key="final_generate"):
                generated_files = [] # Keep track of generated files locally
                zip_buffer = io.BytesIO()
                try:
                    with st.spinner("Generating final clips... This may take a while."):
                        generated_files = video_utils.generate_clips(
                            st.session_state.prepared_clips, # Use latest adjusted clips
                            st.session_state.video_file_path,
                            st.session_state.output_dir # Temp dir for generation
                        )
                        # Removed state update st.session_state.generated_clips = generated_files as it's not needed anymore

                    if not generated_files:
                         st.warning("Final clip generation finished, but no clips were successfully created. Check logs.")
                    else:
                        st.success(f"{len(generated_files)} final clips generated successfully!")
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
                     st.error(f"An error occurred during final clip generation or ZIP creation: {e}")
                     video_utils.logging.error(f"Final Generate/ZIP exception: {e}", exc_info=True)

# Remove Step 5 Block
# =========================
# Step 5: Generate Clips
# =========================
# elif st.session_state.current_step == "5. Generate Clips":
#     # ... (Delete entire block) ...
#     pass


# --- Fallback for unknown state ---
else:
    st.error("Invalid step selected.")
    st.session_state.current_step = "1. Video Input"
    st.session_state.previews_generated = False # Reset preview flag on error
    st.session_state.last_prepared_clips_hash = None # Reset hash tracking
    st.session_state.event_categories_initialized = False
    st.button("Reset to Step 1")
    st.rerun() 