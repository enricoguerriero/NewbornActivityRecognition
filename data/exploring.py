import os

def explore_annotations(annotations_folder, actions):
    """
    Explore the annotations folder and return the list of files with action durations.
    """
    annotation_paths = [os.path.join(annotations_folder, file) for file in os.listdir(annotations_folder)]
    
    summaries = {}
    
    for annotation_path in annotation_paths:
        summary = {}
        with open(annotation_path, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                action = " ".join(parts[:-3])
                try:
                    start = int(parts[-3])
                    end = int(parts[-2])
                    duration = int(parts[-1])
                except ValueError:
                    continue
                
                if action in actions:
                    summary[action] = summary.get(action, 0) + duration
            
        summaries[annotation_path] = summary
                
    return summaries

def pretty_print_summaries(summaries):
    """
    Pretty print the summaries from explore_annotations.
    """
    for annotation_path, summary in summaries.items():
        print(f"Annotation file: {annotation_path}")
        for action, duration in summary.items():
            print(f"  Action: {action}, Duration: {duration} seconds")
        print("\n")

def explore_annotations_with_video_duration(annotations_folder, actions):
    """
    Explore the annotations folder and return a summary that includes:
      - the overall video duration (computed as the maximum 'end' timestamp),
      - the total duration for each specified action, and 
      - the percentage of the video duration for each action.
    """
    annotation_paths = [os.path.join(annotations_folder, file) for file in os.listdir(annotations_folder)]
    
    extended_summaries = {}
    
    for annotation_path in annotation_paths:
        event_summary = {}
        video_duration = 0  # Will be computed as the max 'end' time
        
        with open(annotation_path, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                action = " ".join(parts[:-3])
                try:
                    start = int(parts[-3])
                    end = int(parts[-2])
                    duration = int(parts[-1])
                except ValueError:
                    continue
                
                # Determine the overall video duration using the max end timestamp
                if end > video_duration:
                    video_duration = end
                
                if action in actions:
                    event_summary[action] = event_summary.get(action, 0) + duration
        
        # Compute the percentage duration for each event relative to the video duration
        percentages = {}
        for action, dur in event_summary.items():
            percentages[action] = (dur / video_duration) * 100 if video_duration > 0 else 0
        
        extended_summaries[annotation_path] = {
            "video_duration": video_duration,
            "event_durations": event_summary,
            "event_percentages": percentages
        }
    
    return extended_summaries

def pretty_print_extended_summaries(extended_summaries):
    """
    Pretty print the extended summaries including video duration and event percentages.
    """
    for annotation_path, summary in extended_summaries.items():
        print(f"Annotation file: {annotation_path}")
        print(f"Video duration: {summary['video_duration']} seconds")
        print("Events:")
        for action, duration in summary['event_durations'].items():
            percentage = summary['event_percentages'][action]
            print(f"  Action: {action}, Duration: {duration} seconds, Percentage: {percentage:.2f}%")
        print("\n")

if __name__ == "__main__":
    annotations_folder = "data/annotations"
    actions = ["Baby visible", "CPAP", "PPV", "Stimulation back/nates",
               "Stimulation extremities", "Stimulation trunk", "Suction"]
    
    # Using the original function
    for folder in ["train", "validation", "test"]:
        print(f"Exploring annotations for {folder} set (basic summary)...")
        summaries = explore_annotations(os.path.join(annotations_folder, folder), actions)
        pretty_print_summaries(summaries)
    
    # Using the new function that includes video duration and percentages
    for folder in ["train", "validation", "test"]:
        print(f"Exploring annotations for {folder} set (extended summary)...")
        extended_summaries = explore_annotations_with_video_duration(os.path.join(annotations_folder, folder), actions)
        pretty_print_extended_summaries(extended_summaries)
