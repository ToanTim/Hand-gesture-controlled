from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import json
import os
import cv2
from pathlib import Path
from ..recognizers.gesture_recognizer import GestureRecognizer
from ..detectors.hand_detector import HandDetector
import matplotlib.pyplot as plt


def load_hagrid_samples(data_dir, labels, samples_per_label=50):
    """
    Load samples from HAGRID dataset for specific labels.
    
    Args:
        data_dir: Path to hagrid-sample-30k-384p directory
        labels: List of gesture labels to load (e.g., ['fist', 'palm', 'like', 'one', 'peace', 'ok'])
        samples_per_label: Number of samples to load per label (default 50)
    
    Returns:
        List of test data dicts with landmarks, fingers, distance, and labels
    """
    test_data = []
    ann_dir = os.path.join(data_dir, 'ann_train_val')
    img_dir = os.path.join(data_dir, 'hagrid_30k')
    
    hand_detector = HandDetector()
    
    for label in labels:
        ann_file = os.path.join(ann_dir, f'{label}.json')
        
        if not os.path.exists(ann_file):
            print(f"Warning: Annotation file not found for label '{label}'")
            continue
        
        print(f"Loading samples for '{label}'...")
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        count = 0
        for img_id, data in annotations.items():
            if count >= samples_per_label:
                break
            
            # Find the image file
            label_dir = os.path.join(img_dir, f'train_val_{label}')
            img_path = os.path.join(label_dir, f'{img_id}.jpg')
            
            if not os.path.exists(img_path):
                continue
            
            try:
                # Read image and detect hand landmarks
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                frame = hand_detector.find_hands(frame, draw=False)
                landmark_list = hand_detector.get_position(frame)
                
                if not landmark_list:
                    continue
                
                # Get finger states
                fingers = hand_detector.fingers_up(landmark_list)
                
                # Get distance between thumb and index
                distance, _ = hand_detector.get_distance(4, 8, landmark_list)
                
                # Add to test data
                test_data.append({
                    'landmarks': landmark_list,
                    'fingers': fingers,
                    'dist': distance,
                    'label': label,
                    'img_id': img_id
                })
                
                count += 1
            
            except Exception as e:
                print(f"Error processing {img_id}: {e}")
                continue
        
        print(f"Loaded {count} samples for '{label}'")
    
    hand_detector.close()
    return test_data


def evaluate_recognizer(recognizer, test_data):
    """
    Evaluates the GestureRecognizer against a labeled test set.
    
    Args:
        recognizer: Your GestureRecognizer instance
        test_data: A list of dicts: [
            {'landmarks': [...], 'fingers': [...], 'dist': 50, 'label': 'fist'},
            ...
        ]
    """
    y_true = []
    y_pred = []
    
    print(f"--- Starting Evaluation on {len(test_data)} samples ---")
    
    for sample in test_data:
        # 1. Get prediction from your current logic
        prediction = recognizer.recognize_gesture(
            fingers=sample['fingers'],
            landmark_list=sample['landmarks'],
            distance_thumb_index=sample.get('dist')
        )
        
        y_true.append(sample['label'])
        y_pred.append(prediction)

    # 2. Generate detailed metrics
    report = classification_report(y_true, y_pred, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(recognizer.gesture_names.keys()))
    
    print("\n[Classification Report]")
    print(report)
    
    # 3. Create a clean Confusion Matrix DataFrame for easier reading
    df_cm = pd.DataFrame(
        matrix, 
        index=[f"True:{g}" for g in recognizer.gesture_names.keys()],
        columns=[f"Pred:{g}" for g in recognizer.gesture_names.keys()]
    )
    
    # Filter out empty rows/cols to see only gestures that were tested
    active_gestures = df_cm.loc[(df_cm.sum(axis=1) > 0), (df_cm.sum(axis=0) > 0)]
    print("\n[Confusion Matrix (Active Gestures Only)]")
    print(active_gestures)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent.parent / 'docs/evaluation_result/recognizer_evaluator'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save report and confusion matrix to text file
    report_file = output_dir / 'evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GESTURE RECOGNIZER EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Samples Evaluated: {len(test_data)}\n\n")
        f.write("[Classification Report]\n")
        f.write("-" * 60 + "\n")
        f.write(report)
        f.write("\n[Confusion Matrix (Active Gestures Only)]\n")
        f.write("-" * 60 + "\n")
        f.write(active_gestures.to_string())
        f.write("\n")
    
    print(f"\nEvaluation report saved to: {report_file}")
    
    if not active_gestures.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(active_gestures, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion Matrix (Active Gestures)')
        fig.colorbar(cax)
        ax.set_xticks(range(len(active_gestures.columns)))
        ax.set_xticklabels(active_gestures.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(active_gestures.index)))
        ax.set_yticklabels(active_gestures.index)
        plt.tight_layout()
        matrix_img_path = output_dir / 'confusion_matrix.png'
        plt.savefig(matrix_img_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix figure saved to: {matrix_img_path}")
    else:
        print("\nNo active gestures to plot.")
    
    return report, active_gestures


if __name__ == '__main__':
    # Load data from HAGRID dataset
    data_dir = '/home/toantim/ToanFolder/Hand-gesture-controlled/data/hagrid-sample-30k-384p'
    labels = ['fist', 'palm', 'like', 'one', 'peace', 'ok']
    
    print("Loading HAGRID dataset samples...")
    test_set = load_hagrid_samples(data_dir, labels, samples_per_label=50)
    
    print(f"\nTotal samples loaded: {len(test_set)}")
    
    # Evaluate recognizer
    gr = GestureRecognizer()
    evaluate_recognizer(gr, test_set)