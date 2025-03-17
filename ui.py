# ui/app.py

import streamlit as st
import numpy as np
import cv2
import torch
from data.data_loader import load_clip_from_hdf5  # Your custom loader
from models.inference import run_inference  # Your function to get model predictions
from data.preprocessor import apply_transformation  # Your transformation function

def display_clip(frames, caption):
    """Display a clip frame-by-frame (or as a video) in Streamlit."""
    st.subheader(caption)
    # For demonstration, we'll display the first frame.
    if frames is not None and len(frames) > 0:
        # Convert the first frame from tensor/numpy to image format.
        frame = frames[0]
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        # Assuming the frame is in CHW format (channels, height, width)
        frame = np.transpose(frame, (1, 2, 0))
        frame = (frame * 255).astype(np.uint8)
        st.image(frame, channels="RGB")
    else:
        st.write("No frames to display")

def main():
    st.title("Video Clip Visualization & Model Classification")

    # Sidebar: Select clip index (for demonstration, we assume indices 0-9)
    clip_index = st.sidebar.number_input("Clip Index", min_value=0, max_value=9, value=0, step=1)
    
    # Load the clip (implement load_clip_from_hdf5 accordingly)
    clip_data = load_clip_from_hdf5(clip_index)
    original_frames = clip_data['frames']
    
    # Apply transformation if needed
    transformed_frames = apply_transformation(original_frames)

    # Run inference on both versions (if you want to compare model outputs)
    predictions_original = run_inference(original_frames)
    predictions_transformed = run_inference(transformed_frames)

    st.header("Original Clip")
    display_clip(original_frames, "Original")
    
    st.header("Transformed Clip")
    display_clip(transformed_frames, "Transformed")
    
    st.header("Model Predictions")
    st.subheader("Original Clip Predictions")
    st.write(predictions_original)
    
    st.subheader("Transformed Clip Predictions")
    st.write(predictions_transformed)

if __name__ == "__main__":
    main()
