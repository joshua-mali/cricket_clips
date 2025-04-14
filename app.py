# Placeholder for Streamlit application code 

import io  # Import io
import os  # Add os import
import zipfile  # Import zipfile
from pathlib import Path  # Import Path

import pandas as pd
import streamlit as st

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
        st.info("Define the types of events you want to create clips for.")
        try:
            # Dynamically generate event types based on actual data
            all_batters = sorted(list(st.session_state.match_df['Batter'].dropna().unique()))
            # Consider bowlers too if needed for wicket filtering, e.g., all_bowlers = sorted(list(st.session_state.match_df['Bowler'].dropna().unique()))
            # Combine players involved
            all_players = sorted(list(set(all_batters))) # Add bowlers here if needed

            base_event_types = ["All 4s", "All 6s", "All Wickets"]
            player_event_types = []
            for player in all_players:
                player_event_types.append(f"{player} - 4s")
                player_event_types.append(f"{player} - 6s")
                player_event_types.append(f"{player} - Wickets")

            available_categories = base_event_types + player_event_types

            st.session_state.event_categories = st.multiselect(
                "Select event categories:",
                options=available_categories,
                default=st.session_state.event_categories
            )

            if st.button("Prepare Clips"):
                if st.session_state.event_categories:
                    with st.spinner("Preparing clip definitions..."):
                        try:
                            prepared_list = video_utils.prepare_clip_list(st.session_state.match_df, st.session_state.event_categories)
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
                else:
                    st.warning("Please select at least one event category.")
        except KeyError as ke:
             st.error(f"Missing required column in CSV for generating event types: {ke}. Ensure 'Batter', 'Runs', 'Wicket' exist.")
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
        st.info("Review the prepared clips. Adjust start/end times as needed and update preview.")

        for i, clip_info in enumerate(st.session_state.prepared_clips):
            st.subheader(f"Clip {i+1}: {clip_info['event_desc']}")
            # Use columns for layout
            main_cols = st.columns([2, 3]) # Info | Preview + Adjustments

            with main_cols[0]: # Left column for info
                st.write(f"Original Range: {clip_info['start']:.2f}s - {clip_info['end']:.2f}s")
                st.write(f"**Adjusted Range: {clip_info['adjusted_start']:.2f}s - {clip_info['adjusted_end']:.2f}s**")

            with main_cols[1]: # Right column for preview and adjustments
                preview_placeholder = st.empty() # Placeholder for the video player
                adj_cols = st.columns([1, 1, 1]) # Start Offset, End Offset, Button

                with adj_cols[0]:
                     # Calculate current offset from original start for the input value
                     current_start_offset = clip_info['adjusted_start'] - clip_info['start']
                     start_offset = st.number_input(f"Start Offset (s)", key=f"start_adj_{i}", value=current_start_offset, step=0.5, format="%.1f")

                with adj_cols[1]:
                     # Calculate current offset from original end
                     current_end_offset = clip_info['adjusted_end'] - clip_info['end']
                     end_offset = st.number_input(f"End Offset (s)", key=f"end_adj_{i}", value=current_end_offset, step=0.5, format="%.1f")

                with adj_cols[2]:
                    if st.button("Update & Preview", key=f"update_{i}"):
                        # Calculate new adjusted times based on *original* start/end + offsets
                        new_start = clip_info['start'] + start_offset
                        new_end = clip_info['end'] + end_offset

                        # Basic validation
                        if new_start < new_end and new_start >= 0:
                            # Update the adjusted times in session state
                            st.session_state.prepared_clips[i]['adjusted_start'] = new_start
                            st.session_state.prepared_clips[i]['adjusted_end'] = new_end
                            st.success(f"Clip {i+1} time range updated in memory.")

                            # Generate the preview for the *new* adjusted times
                            with st.spinner(f"Generating preview for clip {i+1}..."):
                                try:
                                    preview_path = video_utils.generate_clip_preview(
                                        st.session_state.video_file_path,
                                        new_start,
                                        new_end,
                                        i # Pass index for unique temp filename
                                    )
                                    if preview_path:
                                        st.session_state.preview_paths[i] = preview_path
                                        st.info("Preview updated.")
                                    else:
                                        st.session_state.preview_paths[i] = None
                                        st.warning("Could not generate preview.")
                                except Exception as e:
                                     st.session_state.preview_paths[i] = None
                                     st.error(f"Error generating preview: {e}")
                                     video_utils.logging.error(f"Preview generation exception: {e}", exc_info=True)
                            st.rerun() # Rerun to update video player and displayed times
                        else:
                            st.warning("Invalid time range after adjustment. Start must be < End and >= 0.")

                # Display the preview video using the path from session state
                current_preview_path = st.session_state.preview_paths.get(i)
                if current_preview_path:
                    try:
                        # Use st.video for the temporary preview file
                        preview_placeholder.video(current_preview_path)
                    except Exception as e:
                         preview_placeholder.error(f"Error displaying preview video: {e}. Was it deleted?")
                else:
                    # Display adjusted time range even if no preview exists yet
                    preview_placeholder.caption(f"Preview for {clip_info['adjusted_start']:.2f}s - {clip_info['adjusted_end']:.2f}s (click 'Update & Preview')")

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