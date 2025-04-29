import streamlit as st
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import torch.nn as nn
# Removed matplotlib as it's not strictly needed for the Streamlit app display
# import matplotlib.pyplot as plt

# Define the CRNN model class
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(0.3),
            nn.Conv2d(512, 512, kernel_size=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Define the AnswerSheetExtractor class
class AnswerSheetExtractor:
    def __init__(self, primary_yolo_weights_path, fallback_yolo_weights_path, register_crnn_model_path, subject_crnn_model_path):
        # Ensure directories exist
        os.makedirs("cropped_register_numbers", exist_ok=True)
        os.makedirs("cropped_subject_codes", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load both YOLO models
        self.primary_yolo_model = YOLO(primary_yolo_weights_path)
        self.fallback_yolo_model = YOLO(fallback_yolo_weights_path) # Load the second model

        # Load Register Number CRNN model
        self.register_crnn_model = CRNN(num_classes=11)  # 10 digits + blank
        self.register_crnn_model.to(self.device)
        checkpoint = torch.load(register_crnn_model_path, map_location=self.device)
        # Handle potential 'module.' prefix if model was trained with DataParallel
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v # remove 'module.' prefix
            else:
                new_state_dict[k] = v
        self.register_crnn_model.load_state_dict(new_state_dict)
        self.register_crnn_model.eval()

        # Load Subject Code CRNN model
        self.subject_crnn_model = CRNN(num_classes=37)  # blank + 0-9 + A-Z
        self.subject_crnn_model.to(self.device)
        # Handle potential 'module.' prefix similarly for subject model
        subject_checkpoint = torch.load(subject_crnn_model_path, map_location=self.device)
        subject_state_dict = subject_checkpoint['model_state_dict'] if isinstance(subject_checkpoint, dict) and 'model_state_dict' in subject_checkpoint else subject_checkpoint # Handle if only state_dict was saved
        new_subject_state_dict = {}
        for k, v in subject_state_dict.items():
             if k.startswith('module.'):
                 new_subject_state_dict[k[7:]] = v # remove 'module.' prefix
             else:
                 new_subject_state_dict[k] = v
        self.subject_crnn_model.load_state_dict(new_subject_state_dict)
        self.subject_crnn_model.eval()

        # Define image transforms
        self.register_transform = transforms.Compose([
            transforms.Resize((32, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.subject_transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Define character map for subject code
        self.char_map = {i: str(i-1) for i in range(1, 11)} # 1-10 -> 0-9
        self.char_map.update({i: chr(i - 11 + ord('A')) for i in range(11, 37)}) # 11-36 -> A-Z
        self.char_map[0] = '' # Map blank to empty string


    def detect_regions(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # --- Step 1: Run Primary YOLO Model ---
        st.info("Running primary YOLO model...")
        results_primary = self.primary_yolo_model(image)
        detections_primary = results_primary[0].boxes
        classes_primary = results_primary[0].names

        register_regions = []
        subject_regions = []

        # Process primary detections
        for i, box in enumerate(detections_primary):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = classes_primary[class_id]
            cropped_region = image[y1:y2, x1:x2]

            # Keep all register number detections from primary
            if label == "RegisterNumber" and confidence > 0.5:
                # Use a distinct name for primary detections
                save_path = f"cropped_register_numbers/register_number_primary_{i}.jpg"
                cv2.imwrite(save_path, cropped_region)
                register_regions.append((save_path, confidence))
            # Temporarily store subject detections from primary
            elif label == "SubjectCode" and confidence > 0.5:
                 # Use a distinct name for primary detections
                 save_path = f"cropped_subject_codes/subject_code_primary_{i}.jpg"
                 cv2.imwrite(save_path, cropped_region)
                 subject_regions.append((save_path, confidence))


        # --- Step 2: Check if Primary found SubjectCode and run fallback if necessary ---
        final_subject_regions = subject_regions # Start with primary results for subject

        if not final_subject_regions: # If primary model found NO subject codes
            st.warning("Primary model did not detect Subject Code. Running fallback YOLO model...")
            results_fallback = self.fallback_yolo_model(image)
            detections_fallback = results_fallback[0].boxes
            classes_fallback = results_fallback[0].names # Should be same classes as primary

            subject_regions_fallback = []
            # Process fallback detections, but only look for SubjectCode
            for i, box in enumerate(detections_fallback):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = classes_fallback[class_id]
                cropped_region = image[y1:y2, x1:x2]

                if label == "SubjectCode" and confidence > 0.5:
                    # Use a distinct name for fallback detections
                    save_path = f"cropped_subject_codes/subject_code_fallback_{i}.jpg"
                    cv2.imwrite(save_path, cropped_region)
                    subject_regions_fallback.append((save_path, confidence))

            # Replace subject regions with fallback results if fallback found any
            final_subject_regions = subject_regions_fallback
            if not final_subject_regions:
                 st.warning("Fallback model also did not detect Subject Code.")
            else:
                 st.success(f"Fallback model detected {len(final_subject_regions)} Subject Code region(s).")


        # Return the register regions from the primary model
        # and the subject regions (either from primary or fallback)
        return register_regions, final_subject_regions

    # Keep extract_register_number and extract_subject_code methods as they are
    # (They will process the cropped images saved by detect_regions)
    def extract_register_number(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            image_tensor = self.register_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.register_crnn_model(image_tensor).squeeze(1)
                # Applying softmax for clarity, although argmax on logits is equivalent
                output_probs = output.softmax(1)
                output = output_probs.argmax(1)
                seq = output.cpu().numpy()
                prev = -1 # CTC decoding requires tracking previous character
                result = []
                for s in seq:
                    # s != 0 checks for blank token (index 0)
                    # s != prev checks for consecutive identical non-blank tokens
                    if s != 0 and s != prev:
                        result.append(s - 1) # Map 1-10 to 0-9
                    prev = s
            return ''.join(map(str, result))
        except Exception as e:
            st.error(f"Error extracting register number from {image_path}: {e}")
            return "EXTRACTION ERROR"

    def extract_subject_code(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            image_tensor = self.subject_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.subject_crnn_model(image_tensor).squeeze(1)
                # Applying softmax for clarity
                output_probs = output.softmax(1)
                output = output_probs.argmax(1)
                seq = output.cpu().numpy()
                prev = 0 # Blank token index is 0
                result = []
                for s in seq:
                    # s != 0 checks for blank token (index 0)
                    # s != prev checks for consecutive identical non-blank tokens
                    if s != 0 and s != prev:
                         # Map index to character using self.char_map
                        result.append(self.char_map.get(s, ''))
                    prev = s
            return ''.join(result)
        except Exception as e:
            st.error(f"Error extracting subject code from {image_path}: {e}")
            return "EXTRACTION ERROR"

    # Keep process_answer_sheet method as it is
    # (It will now receive the combined results from detect_regions)
    def process_answer_sheet(self, image_path):
        # detect_regions now handles the fallback logic internally
        register_regions, subject_regions = self.detect_regions(image_path)
        results = []
        register_cropped_path = None
        subject_cropped_path = None # Initialize to None

        if register_regions:
            # Assume the best register region is from the primary model's findings
            best_region = max(register_regions, key=lambda x: x[1])
            register_cropped_path = best_region[0]
            st.info(f"Extracting Register Number from: {register_cropped_path}")
            register_number = self.extract_register_number(register_cropped_path)
            results.append(("Register Number", register_number))
        else:
             st.warning("No Register Number region detected.")

        if subject_regions:
            # Assume the best subject region is from the list returned by detect_regions
            # (which is either primary or fallback results)
            best_subject = max(subject_regions, key=lambda x: x[1])
            subject_cropped_path = best_subject[0]
            st.info(f"Extracting Subject Code from: {subject_cropped_path}")
            subject_code = self.extract_subject_code(subject_cropped_path)
            results.append(("Subject Code", subject_code))
        else:
            st.warning("No Subject Code region detected.")


        return results, register_cropped_path, subject_cropped_path

# Streamlit app
def main():
    st.title("Answer Sheet Extractor")

    # Load models
    with st.spinner("Loading models..."):
        try:
            # Instantiate AnswerSheetExtractor with both YOLO model paths
            # Ensure these files ('improved_weights.pt' and 'weights.pt')
            # are present in your GitHub repository root.
            extractor = AnswerSheetExtractor(
                primary_yolo_weights_path="improved_weights.pt",
                fallback_yolo_weights_path="weights.pt",
                register_crnn_model_path="best_crnn_model(git).pth",
                subject_crnn_model_path="best_subject_model_final.pth"
            )
            st.success("Models loaded successfully")
        except Exception as e:
            st.error(f"Failed to load models. Please check your model paths and ensure they are in your repository. Error: {e}")
            # Optionally, st.exception(e) for more detailed traceback in logs
            return

    # Upload image
    uploaded_file = st.file_uploader("Upload Answer Sheet Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Create a temporary directory if needed, or save directly
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, uploaded_file.name) # Use original filename

        # Save uploaded image
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display uploaded image
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

        # Process image
        if st.button("Extract Information"):
            # Clear previous results (optional, but good UI practice)
            # st.empty() # Not ideal, might clear the button

            with st.spinner("Processing image..."):
                try:
                    results, register_cropped, subject_cropped = extractor.process_answer_sheet(image_path)
                    st.success("Extraction complete")

                    # Display results
                    if results:
                        st.subheader("Extracted Information:")
                        for label, value in results:
                            st.write(f"**{label}:** {value}")
                    else:
                        st.warning("No information could be extracted.")


                    # Display cropped images
                    st.subheader("Detected Regions:")
                    if register_cropped:
                         # Load image using PIL to display in Streamlit
                         register_img = Image.open(register_cropped)
                         st.image(register_img, caption="Cropped Register Number", width=250)
                    else:
                         st.info("No Register Number region found to display.")

                    if subject_cropped:
                         # Load image using PIL to display in Streamlit
                         subject_img = Image.open(subject_cropped)
                         st.image(subject_img, caption="Cropped Subject Code", width=250)
                    else:
                         st.info("No Subject Code region found to display.")


                except Exception as e:
                    st.error(f"Failed to process image: {e}")
                    st.exception(e) # Display full traceback in Streamlit logs

if __name__ == "__main__":
    main()
