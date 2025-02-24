import os

def collect_event_categories(annotation_folder):
    """
    Scan all annotation files in a folder to collect a sorted list of event names.
    Each annotation file is expected to have lines with at least 4 tokens, where:
      - tokens[:-3] form the event name,
      - tokens[-3] is the event start time (in ms),
      - tokens[-2] is the event end time (in ms).
    """
    events_set = set()
    for fname in os.listdir(annotation_folder):
        ann_path = os.path.join(annotation_folder, fname)
        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 4:
                    continue
                event_name = " ".join(tokens[:-3]).strip()
                if event_name:
                    events_set.add(event_name)
    return sorted(list(events_set))