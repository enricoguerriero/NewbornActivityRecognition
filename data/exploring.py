import os

def explore_annotations(annotations_folder, actions):
    """
    Explore the annotations folder and return the list of files.
    """
    
    annotation_folders = os.listdir(annotations_folder)
    annotation_folders = [os.path.join(annotations_folder, folder) for folder in annotation_folders]
    annotation_paths = [os.listdir(folder) for folder in annotation_folders]
    annotation_paths = [os.path.join(folder, path) for folder in annotation_folders for path in annotation_paths]
    
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


if __name__ == "__main__":
    annotations_folder = "/home/luca/Projects/NewbornActivityRecognition/data/annotations"
    actions = ["Baby visible", "CPAP", "PPV", "Stimulation back/nates",
                                 "Stimulation extremities", "Stimulation trunk", "Suction"]
    
    summaries = explore_annotations(annotations_folder, actions)
    print(summaries)