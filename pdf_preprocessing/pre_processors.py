import os
import tempfile
import io
import fitz
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
class PDF_Extractor:
    """
    Extract high-resolution images from PDF documents.
    
    Iterates through a directory of PDFs and renders pages as images in memory.
    Primarily targets the first page for ID documents (typically single-page).
    """
    
    def __init__(self, pdf_directory):
        """
        Initialize the PDF extractor.
        
        Args:
            pdf_directory (str): Path to directory containing PDF files
        """
        self.pdf_directory = pdf_directory
        self.dpi_multiplier = 3.0  # Increase for higher resolution (default fitz is ~72 DPI)
    
    def extract_images_from_directory(self, extract_all_pages=False):
        """
        Iterate through all PDFs in the directory and extract images.
        
        Args:
            extract_all_pages (bool): If True, extract all pages; if False, extract first page only
        
        Yields:
            tuple: (pdf_filename, page_number, pix_data) 
                   where pix_data is the Pixmap object containing image data in memory
        
        Note:
            Data storage (local/cloud) for inputs and outputs to be determined.
            Currently storing images in memory (Pixmap format).
        """
        pdf_files = self._get_pdf_files()
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            
            try:
                pix_list = self._render_pdf_pages(
                    pdf_path, 
                    all_pages=extract_all_pages
                )
                
                for page_num, pix in pix_list:
                    yield (pdf_file, page_num, pix)
                    
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue
    
    def _get_pdf_files(self):
        """
        Get list of PDF files in the directory.
        
        Returns:
            list: Sorted list of PDF filenames
        """
        if not os.path.isdir(self.pdf_directory):
            raise ValueError(f"Directory not found: {self.pdf_directory}")
        
        pdf_files = [f for f in os.listdir(self.pdf_directory) 
                     if f.lower().endswith('.pdf')]
        
        return sorted(pdf_files)
    
    def _render_pdf_pages(self, pdf_path, all_pages=False):
        """
        Render PDF pages to high-resolution images in memory.
        
        Args:
            pdf_path (str): Path to the PDF file
            all_pages (bool): If True, render all pages; if False, render first page only
        
        Returns:
            list: List of tuples (page_number, pix) containing in-memory Pixmap objects
        """
        images = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            # Determine page range
            if all_pages:
                page_range = range(len(pdf_document))
            else:
                page_range = range(min(1, len(pdf_document)))  # First page only
            
            for page_num in page_range:
                page = pdf_document[page_num]
                
                # Render page with high resolution
                # mat scales the page rendering (e.g., 3.0 = 3x resolution)
                mat = fitz.Matrix(self.dpi_multiplier, self.dpi_multiplier)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                images.append((page_num, pix))
            
            pdf_document.close()
            
        except RuntimeError:
            raise RuntimeError(f"Invalid or corrupted PDF: {pdf_path}")
        
        return images
    
    def extract_first_page_image(self, pdf_path):
        """
        Extract the first page of a single PDF as a high-resolution image.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            Pixmap: Image data in memory (PyMuPDF Pixmap object)
                    
        Note:
            TODO: Determine final storage mechanism for extracted images
            (local disk, cloud storage, database, etc.)
        """
        try:
            pdf_document = fitz.open(pdf_path)
            first_page = pdf_document[0]
            
            # Render with high resolution
            mat = fitz.Matrix(self.dpi_multiplier, self.dpi_multiplier)
            pix = first_page.get_pixmap(matrix=mat, alpha=False)
            
            pdf_document.close()
            
            return pix
            
        except RuntimeError:
            raise RuntimeError(f"Invalid or corrupted PDF: {pdf_path}")
        except IndexError:
            raise IndexError(f"PDF has no pages: {pdf_path}")

class ELA_Generator:
    """    
    Generate ELA (Error Level Analysis) images from extracted high-resolution
    images of PDF pages. Supports both PIL Images and file paths as input.
    """
    
    def __init__(self, default_quality=90, temp_dir=None):
        """
        Initialize the ELA generator.
        
        Args:
            default_quality (int): Default JPEG quality for ELA processing (1-100)
            temp_dir (str): Directory for temporary files. If None, uses system temp directory
        """
        self.default_quality = default_quality
        self.temp_dir = temp_dir or tempfile.gettempdir()

    def generate_ela(self, image_input, output_path, quality=None, target_size=256):
        """
        Generate ELA (Error Level Analysis) image from input image.
        
        Args:
            image_input: Either a PIL Image object or path to image file
            output_path (str): Path where the ELA image will be saved
            quality (int): JPEG quality for ELA (1-100). If None, uses default_quality
            target_size (int): Target size for resizing image to (target_size x target_size)
        
        Returns:
            Image: The generated and resized ELA image (PIL Image object)
        """
        if quality is None:
            quality = self.default_quality
        
        # Convert input to PIL Image if it's a file path
        if isinstance(image_input, str):
            original = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            original = image_input.convert('RGB')
        else:
            raise TypeError("image_input must be a PIL Image or file path")
        
        # Resize original to target size for consistency
        original = original.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        try:
            # 1. Save original to in-memory buffer at target quality
            buffer = io.BytesIO()
            original.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)  # Reset buffer pointer to beginning
            
            # 2. Open the re-compressed image from buffer
            compressed = Image.open(buffer).convert('RGB')
            
            # 3. Calculate absolute difference between original and compressed
            ela_image = ImageChops.difference(original, compressed)
            
            # 4. Normalize brightness based on maximum difference
            ela_array = np.array(ela_image)
            max_diff = np.max(ela_array)
            
            if max_diff > 0:
                scale = 255.0 / max_diff
                ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
            
            # 5. Resize ELA image to target size for CNN input
            ela_image = ela_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # 6. Save the ELA map
            ela_image.save(output_path)
            
            return ela_image
            
        finally:
            # Buffer cleanup is automatic when it goes out of scope
            pass
    
    def pixmap_to_pil(self, pixmap, target_size=256):
        """
        Convert PyMuPDF Pixmap (from PDF_Extractor) to PIL Image and resize.
        
        Args:
            pixmap: PyMuPDF Pixmap object
            target_size (int): Target size for resizing image to (target_size x target_size)
        
        Returns:
            Image: Resized PIL Image object ready for CNN input
        """
        image_data = pixmap.tobytes("ppm")
        image = Image.open(io.BytesIO(image_data))
        
        # Resize to target size for CNN pipeline
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return image