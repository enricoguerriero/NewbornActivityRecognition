BASE_CONFIG = {
    
    # data creation
    "clip_length": 3,
    "frames_per_second": 8,
    "overlap": 2,
    "target_size": (256, 256),
    "event_categories": ["Baby visible", "Ventilation", "Stimulation", "Suction"],
   
    # training parameters
    "batch_size": 8,
    "num_workers": 4,
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "criterion": "bce",
    "threshold": 0.5,
    "device": "cuda",
    "momentum": 0.9,
    
    # wandb
    "wandb_project": "newborn-activity-recognition",
    
    # folder structure
    "video_folder": "data/videos",
    "annotation_folder": "data/annotations",
    
    # prompts
    "system_message": """
        You are assisting in a medical simulation analysis. A camera is positioned above a table. The simulation involves a mannequin representing a newborn baby, which may or may not be present on the table.

        Your tasks are as follows:

        1. Determine Presence
        - Check if the mannequin (baby) is visible and present on the table.
        - If not present or not visible, no treatment is being performed.
        - If present, continue to the next steps.

        2. Identify the Mannequin's Face
        - Locate the face of the mannequin. This is the key area for identifying some treatments.

        3. Detect Medical Treatments
        If the mannequin is present, identify whether the following treatments are being performed. These treatments can occur individually or at the same time:

        - Ventilation:
            - A healthworker is holding a ventilation mask over the mannequin's face.
            - This means ventilation is being administered.

        - Suction:
            - A tube is inserted into the mannequin's mouth or nose.
            - This means suction is being performed.

        - Stimulation:
            - A healthworker is applying stimulation to the mannequin's back, buttocks (nates), or trunk.
            - Stimulation is indicated by:
            - Hands placed on one of these areas
            - Up-and-down repetitive hand movement

        Repeat: If the mannequin is not visible, no treatment is being performed.

        Respond clearly based on what is visible in the image. Use concise and structured output when possible.
        """,
    "questions": ["Is the baby visible?", "Is the baby receiving ventilation?", "Is the baby receiving stimulation?", "Is the baby receiving suction?"],
    "question": "You need to anwer with the following format: [i, j, k, l] where all of them can be 0 or 1. i = 1 if baby is visible, j = 1 if baby is receiving ventilation, k = 1 if baby is receiving stimulation, l = 1 if baby is receiving suction. If the baby is not visible, all of them should be 0. If the baby is visible, but not receiving any treatment, all of them should be 0 except for the i.",
    
    }

