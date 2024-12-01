import os
from PyPDF2 import PdfReader
from PIL import Image
import fitz  # PyMuPDF
import io
import re
"""
    Renders each page of all PDF files in a folder as a high-resolution JPG image.

    Args:
        input_folder (str): Path to the folder containing PDF files.
        output_folder (str): Path to the folder where JPG images will be saved.
        zoom (float): Zoom factor for rendering pages (higher values produce higher resolution).
"""
def render_pdf_pages_as_jpg(input_folder, output_folder, zoom=2.0):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        try:
            # Open the PDF file
            document = fitz.open(pdf_path)
            file_base_name = os.path.splitext(pdf_file)[0]

            for page_number in range(len(document)):
                page = document[page_number]

                # Set zoom factor for high-resolution rendering
                matrix = fitz.Matrix(zoom, zoom)

                # Render the page to a pixmap
                pix = page.get_pixmap(matrix=matrix)

                # Convert the pixmap to an image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Save the image as JPG
                jpg_file_name = f"{file_base_name}_page{page_number + 1}.jpg"
                jpg_path = os.path.join(output_folder, jpg_file_name)
                img.save(jpg_path, "JPEG")
                print(f"Saved high-resolution image: {jpg_path}")

        except Exception as e:
            print(f"Error processing file {pdf_file}: {e}")

""""
    Renames image files in the specified folder from:
    'One Piece - Digital Colored Comics v<Volume Number> (Just Kidding Productions)_page<PageNumber>.jpg'
    to:
    'v<VolumeNumber>_page<PageNumber>.jpg'

    Args:
        folder_path (str): Path to the folder containing the images.
    """
def rename_images_in_folder(folder_path):

    # Regular expression to match the current file name format
    pattern = r"One Piece - Digital Colored Comics (v\d+) \([^)]+\)_page(\d+)\.jpg"
    for filename in os.listdir(folder_path):
        # Check if the file matches the expected pattern
        match = re.match(pattern, filename)
        if match:
            volume_number = match.group(1)
            page_number = match.group(2)

            # Construct the new file name
            new_filename = f"{volume_number}_page{page_number}.jpg"

            # Rename the file
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")
        else:
            print(f"Skipped: {filename} (does not match the expected pattern)")

# Example usage
input_folder = "PDFSource"  # Replace with the path to your folder containing PDFs
output_folder = "TablesImages"
#render_pdf_pages_as_jpg(input_folder, output_folder)
rename_images_in_folder(output_folder)