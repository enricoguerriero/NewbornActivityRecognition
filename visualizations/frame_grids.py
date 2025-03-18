import os
import matplotlib.pyplot as plt
import cv2

def generate_frame_grids(frames_dir, output_grid_dir, splits=["train", "validation", "test"], grid_cols=4):
    os.makedirs(output_grid_dir, exist_ok=True)
    for split in splits:
        split_dir = os.path.join(frames_dir, split)
        if not os.path.exists(split_dir):
            print(f"Frames directory for split '{split}' not found: {split_dir}")
            continue
        video_ids = os.listdir(split_dir)
        for video in video_ids:
            video_frame_dir = os.path.join(split_dir, video)
            image_files = [f for f in os.listdir(video_frame_dir) if f.endswith(".jpg") or f.endswith(".png")]
            if not image_files:
                print(f"No image files found in {video_frame_dir}")
                continue
            image_files.sort()
            n_images = len(image_files)
            grid_rows = (n_images + grid_cols - 1) // grid_cols
            plt.figure(figsize=(grid_cols * 3, grid_rows * 3))
            for idx, img_file in enumerate(image_files):
                img_path = os.path.join(video_frame_dir, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(grid_rows, grid_cols, idx + 1)
                plt.imshow(img)
                plt.title(img_file, fontsize=8)
                plt.axis("off")
            plt.suptitle(f"Frame Grid for Video {video} in {split} split", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            output_file = os.path.join(output_grid_dir, f"frame_grid_{video}_{split}.png")
            plt.savefig(output_file)
            plt.close()
            print(f"Saved frame grid for video {video} in {split} at {output_file}")

if __name__ == "__main__":
    frames_dir = os.path.join("visualizations", "extracted_frames")
    output_grid_dir = os.path.join("visualizations", "frame_grids")
    generate_frame_grids(frames_dir, output_grid_dir)
