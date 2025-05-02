import streamlit as st
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from pdf2image import convert_from_path, pdfinfo_from_path # Import pdf2image
import tempfile # Import tempfile for creating temporary files

# Define the CRNN model class (remains the same)
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

# Define the AnswerSheetExtractor class (remains mostly the same, detection logic updated)
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
        state_dict = checkpoint.get('model_state_dict', checkpoint) # Handle if state_dict was saved directly
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
        subject_state_dict = subject_checkpoint.get('model_state_dict', subject_checkpoint) # Handle if state_dict was saved directly
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
        self.char_map[0] = '' # Map blank (index 0) to empty string


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
        subject_regions_primary = [] # Keep primary subject detections separate initially

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
                 subject_regions_primary.append((save_path, confidence))


        # --- Step 2: Check if Primary found SubjectCode and run fallback if necessary ---
        final_subject_regions = subject_regions_primary # Start with primary results for subject

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

            # Replace final subject regions with fallback results
            final_subject_regions = subject_regions_fallback
            if not final_subject_regions:
                 st.warning("Fallback model also did not detect Subject Code.")
            else:
                 st.success(f"Fallback model detected {len(final_subject_regions)} Subject Code region(s).")
        else:
             st.success(f"Primary model detected {len(final_subject_regions)} Subject Code region(s).")


        # Return the register regions from the primary model
        # and the subject regions (either from primary or fallback)
        return register_regions, final_subject_regions

    # Keep extract_register_number method as is
    def extract_register_number(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            image_tensor = self.register_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.register_crnn_model(image_tensor).squeeze(1)
                output = output.softmax(1).argmax(1)
                seq = output.cpu().numpy()
                prev = -1 # CTC decoding requires tracking previous character
                result = []
                for s in seq:
                    if s != 0 and s != prev: # s != 0 is blank token (index 0 for digits usually)
                        result.append(s - 1) # Map 1-10 to 0-9
                    prev = s
            return ''.join(map(str, result))
        except Exception as e:
            st.error(f"Error extracting register number from {image_path}: {e}")
            return "EXTRACTION ERROR"

    # Keep extract_subject_code method as is
    def extract_subject_code(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            image_tensor = self.subject_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.subject_crnn_model(image_tensor).squeeze(1)
                output = output.softmax(1).argmax(1)
                seq = output.cpu().numpy()
                prev = 0 # Blank token index is 0 for subject code mapping
                result = []
                for s in seq:
                    if s != 0 and s != prev: # s != 0 checks for blank token (index 0)
                         # Map index to character using self.char_map
                         result.append(self.char_map.get(s, ''))
                    prev = s
            return ''.join(result)
        except Exception as e:
            st.error(f"Error extracting subject code from {image_path}: {e}")
            return "EXTRACTION ERROR"

    # process_answer_sheet method remains the same, it takes an image path
    # The PDF to image conversion happens before calling this method in main()
    def process_answer_sheet(self, image_path):
        # detect_regions now handles the fallback logic internally for subject code
        register_regions, subject_regions = self.detect_regions(image_path)
        results = []
        register_cropped_path = None
        subject_cropped_path = None # Initialize to None

        # Select the best Register Number region (highest confidence)
        if register_regions:
            best_region = max(register_regions, key=lambda x: x[1])
            register_cropped_path = best_region[0]
            st.info(f"Extracting Register Number from: {register_cropped_path}")
            register_number = self.extract_register_number(register_cropped_path)
            results.append(("Register Number", register_number))
        else:
             st.warning("No Register Number region detected.")


        # Select the Subject Code region based on the new rule
        if subject_regions: # Only proceed if at least one subject region was found
            if len(subject_regions) >= 2:
                # Select the SECOND detected region (index 1)
                # Ensure index 1 exists before accessing
                best_subject = subject_regions[1] if len(subject_regions) > 1 else subject_regions[0]
                subject_cropped_path = best_subject[0]
                st.info(f"Multiple Subject Code regions detected ({len(subject_regions)}). Selecting the second one: {subject_cropped_path}")
            else: # len(subject_regions) == 1
                # Select the only detected region (index 0)
                best_subject = subject_regions[0]
                subject_cropped_path = best_subject[0]
                st.info(f"One Subject Code region detected. Selecting it: {subject_cropped_path}")

            # Now extract the subject code from the selected region
            subject_code = self.extract_subject_code(subject_cropped_path)
            results.append(("Subject Code", subject_code))
        else:
            st.warning("No Subject Code region detected.")


        return results, register_cropped_path, subject_cropped_path

# Streamlit app
def main():
    st.title("Answer Sheet Extractor")

    # Load models
    # Use Streamlit's caching to avoid reloading models on every interaction
    @st.cache_resource
    def load_extractor():
        with st.spinner("Loading models..."):
            try:
                # Instantiate AnswerSheetExtractor with both YOLO model paths
                # Ensure these files ('improved_weights.pt' and 'weights.pt')
                # are present in your GitHub repository root.
                extractor = AnswerSheetExtractor(
                    primary_yolo_weights_path="improved_weights.pt",
                    fallback_yolo_weights_path="weights.pt", # Your previous weights
                    register_crnn_model_path="best_crnn_model(git).pth",
                    subject_crnn_model_path="best_subject_code_model-fulldataset.pth"
                )
                st.success("Models loaded successfully")
                return extractor
            except Exception as e:
                st.error(f"Failed to load models. Please check your model paths and ensure they are in your repository. Error: {e}")
                st.exception(e) # Display full traceback in logs
                return None

    extractor = load_extractor()

    if extractor is None:
        st.stop() # Stop the app if models failed to load

    # Input source selection
    input_source = st.radio("Select Input Source", ("Upload File", "Webcam (Experimental)"))

    image_to_process_path = None # This will store the path to the image file

    # Create a temporary directory for uploads and processed images
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    if input_source == "Upload File":
        uploaded_file = st.file_uploader("Upload Answer Sheet PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

        if uploaded_file is not None:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)

            # Save the uploaded file
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'pdf':
                st.info(f"Processing uploaded PDF: {uploaded_file.name}")
                try:
                    # Convert the first page of the PDF to an image
                    images_from_pdf = convert_from_path(temp_file_path, dpi=300, first_page=1, last_page=1)
                    if images_from_pdf:
                        first_page_image = images_from_pdf[0]
                        # Save the converted image to a temporary file
                        image_filename = os.path.splitext(uploaded_file.name)[0] + "_page_1.jpg"
                        image_to_process_path = os.path.join(temp_dir, image_filename)
                        first_page_image.save(image_to_process_path, "JPEG")
                        st.success("Successfully converted first page of PDF to image.")
                    else:
                        st.error("Could not convert first page of PDF to image.")
                        # Clean up the temporary PDF file
                        os.remove(temp_file_path)


                except Exception as e:
                    st.error(f"Error converting PDF to image: {e}")
                    st.exception(e)
                    # Clean up the temporary PDF file
                    os.remove(temp_file_path)


                finally:
                    # Clean up the temporary PDF file after processing attempt
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)


            elif file_extension in ['png', 'jpg', 'jpeg']:
                st.info(f"Processing uploaded image: {uploaded_file.name}")
                # If it's already an image, the temporary file is the one to process
                image_to_process_path = temp_file_path

            else:
                st.error(f"Unsupported file type: {file_extension}")
                # Clean up the temporary file
                os.remove(temp_file_path)


    elif input_source == "Webcam":
        st.info("Using webcam. Please ensure your answer sheet is clearly visible.")
        camera_image = st.camera_input("Take a picture of the answer sheet")

        if camera_image is not None:
            # Save the captured image to a temporary file
            # Use tempfile to create a unique temporary file
            with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".jpg", delete=False) as tmp_file:
                tmp_file.write(camera_image.getvalue())
                image_to_process_path = tmp_file.name
            st.success("Image captured from webcam.")


    # Process the image if one is available from either source
    if image_to_process_path and os.path.exists(image_to_process_path):
         # Display the image that will be processed
         st.image(image_to_process_path, caption="Image to Process", use_column_width=True)

         # Process image
         if st.button("Extract Information"):
             with st.spinner("Processing image..."):
                 try:
                     results, register_cropped, subject_cropped = extractor.process_answer_sheet(image_to_process_path)
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
                     if register_cropped and os.path.exists(register_cropped):
                          register_img = Image.open(register_cropped)
                          st.image(register_img, caption="Cropped Register Number", width=250)
                     else:
                          st.info("No Register Number region found to display.")

                     if subject_cropped and os.path.exists(subject_cropped):
                          subject_img = Image.open(subject_cropped)
                          st.image(subject_img, caption="Cropped Subject Code", width=250)
                     else:
                          st.info("No Subject Code region found to display.")

                 except Exception as e:
                     st.error(f"Failed to process image: {e}")
                     st.exception(e) # Display full traceback in Streamlit logs
                 finally:
                     # Clean up temporary cropped images after processing
                     for folder in ["cropped_register_numbers", "cropped_subject_codes", "results"]:
                         if os.path.exists(folder):
                             # Iterate over a copy of the list to avoid issues when deleting
                             for file in list(os.listdir(folder)):
                                 file_path = os.path.join(folder, file)
                                 try:
                                     os.remove(file_path)
                                 except OSError as e:
                                     st.warning(f"Could not remove temporary file {file_path}: {e}")

                     # Clean up the temporary image file that was processed
                     if os.path.exists(image_to_process_path):
                         try:
                             os.remove(image_to_process_path)
                         except OSError as e:
                             st.warning(f"Could not remove temporary processed image {image_to_process_path}: {e}")


    # Add a placeholder to keep the "Extract Information" button visible
    # even if no file is uploaded or image captured yet
    # (This is a common Streamlit pattern for conditional buttons)
    if input_source == "Upload File" and uploaded_file is None:
        st.empty() # Or a message like "Upload a file to proceed"
    elif input_source == "Webcam" and camera_image is None:
         st.empty() # Or a message like "Take a picture to proceed"


if __name__ == "__main__":
    main()
