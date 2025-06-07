import os
import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset

class NiiToDcmModel:
    def __init__(self, input_dir, output_dir, num_parts=60):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_parts = num_parts

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_nii(self, nii_file):
        """Load the .nii file using nibabel and return the image data."""
        nii_img = nib.load(nii_file)
        return nii_img.get_fdata()

    def split_image(self, img_data):
        """Split the image into the specified number of parts along the z-axis."""
        x, y, z = img_data.shape
        part_size = z // self.num_parts  # Split along the z-axis
        
        # Generate the slices
        slices = []
        for i in range(self.num_parts):
            start_z = i * part_size
            end_z = (i + 1) * part_size if i < self.num_parts - 1 else z
            img_part = img_data[:, :, start_z:end_z]
            slices.append(img_part)
        
        return slices

    def convert_to_dcm(self, img_part, part_idx, slice_idx, output_subfolder):
        """Convert a 2D slice from the image part to a DICOM file."""
        # Choose a single slice (2D) from the 3D image part
        dicom_data = img_part[:, :, slice_idx]

        # Initialize File Meta Information
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # Use the desired transfer syntax
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage  # Example SOP Class
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()  # Generate a unique UID
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

        # Create a new DICOM dataset
        ds = Dataset()
        ds.file_meta = file_meta  # Assign the file_meta to the dataset
        ds.PixelData = dicom_data.tobytes()
        ds.Rows, ds.Columns = dicom_data.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PatientName = "Test^Patient"
        ds.PatientID = "12345"
        ds.Modality = 'CT'  # Or another appropriate modality
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID  # Match with file_meta

        # Create DICOM filename
        dcm_filename = f"{output_subfolder}/part_{part_idx+1}_slice_{slice_idx+1}.dcm"

        # Save the DICOM file
        pydicom.dcmwrite(dcm_filename, ds, write_like_original=False)
        print(f"Saved: {dcm_filename}")



    def process_nii_to_dcm(self, nii_file_path, output_root_folder):
        """Process a single .nii file and save its slices as .dcm files."""
        import nibabel as nib

        # Load the .nii file
        nii_data = nib.load(nii_file_path)
        img_data = nii_data.get_fdata()

        # Get the filename without extension
        nii_filename = os.path.splitext(os.path.basename(nii_file_path))[0]

        # Create a unique folder for this .nii file
        output_folder = os.path.join(output_root_folder, nii_filename)
        os.makedirs(output_folder, exist_ok=True)

        # Split the image into parts (e.g., into 60 parts or slices)
        num_parts = 60  # Define the number of parts you want to split into
        part_size = img_data.shape[2] // num_parts

        for part_idx in range(num_parts):
            start_slice = part_idx * part_size
            end_slice = start_slice + part_size if part_idx < num_parts - 1 else img_data.shape[2]
            img_part = img_data[:, :, start_slice:end_slice]

            for slice_idx in range(img_part.shape[2]):
                self.convert_to_dcm(img_part, part_idx, slice_idx, output_folder)

        print(f"Processed: {nii_file_path} -> {output_folder}")

    def process_dataset(self):
        """Process all .nii files in the input directory."""
        for subfolder in os.listdir(self.input_dir):
            subfolder_path = os.path.join(self.input_dir, subfolder)
            if os.path.isdir(subfolder_path):
                # Create an output subfolder for each input subfolder
                output_subfolder = os.path.join(self.output_dir, subfolder)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                
                # Process each .nii file in the subfolder
                for nii_file in os.listdir(subfolder_path):
                    if nii_file.endswith(".nii"):
                        nii_file_path = os.path.join(subfolder_path, nii_file)
                        print(f"Processing: {nii_file_path}")
                        self.process_nii_to_dcm(nii_file_path, output_subfolder)

# Example usage
input_dir = "C:/Users/piyus/Downloads/Liver-Segmentation-Using-Monai-and-PyTorch-main/dataset"  # Replace with the path to your dataset folder
output_dir = "output_dcm"  # Replace with the desired output folder
model = NiiToDcmModel(input_dir, output_dir)
model.process_dataset()
