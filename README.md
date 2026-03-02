# Document Forgery Detection System

A machine learning system for detecting forged identity documents using dual-branch CNN architecture. This project leverages synthetic document generation, advanced image forensics, and deep learning to classify authentic and manipulated identity cards.

## 🎯 System Features

### Synthetic Document Generation
- **Authentic ID Card Creation**: Generates realistic identity documents with synthesized data (name, date of birth, citizen ID, dates) and real-world face images
- **Multi-Class Forgery Generation**: Creates 4 types of forged documents with pixel-level manipulation masks:
  - **Class 1**: Photo replacement with quality degradation (60% compression)
  - **Class 2**: Text alteration with precise erasure and repositioning (name field)
  - **Class 4**: Region overlay attacks (date modifications)
  - **Class 3**: Reserved for future forgery techniques
- **Mask Generation**: Automatic creation of binary masks indicating tampered regions for supervised learning

### PDF Preprocessing Pipeline
- **High-Resolution PDF Extraction**: Renders PDFs at 3x resolution (3.0 DPI multiplier) using PyMuPDF
- **Batch Processing**: Iterates through document directories with built-in skip-resume capability
- **Memory Efficient**: Garbage collection after each document and dual-path output architecture

### Dual-Branch CNN Input Generation
- **Branch A**: Clean extracted images resized to 256×256 pixels (JPEG format)
- **Branch B**: Error Level Analysis (ELA) maps for compression artifact detection
- **Format**: Compatible with modern CNN architectures for forensic document analysis

## 🔧 Core Classes

### `SyntheticDataGenerator`
Orchestrates synthetic document creation from template images.

**Key Methods:**
- `generate_faker_dictionaries()`: Creates synthetic citizen records with realistic data
- `create_authentic_id_cards(data)`: Renders baseline authentic documents with overlaid text and photos
- `create_forged_id_cards(class_counts)`: Generates class-specific forgeries with corresponding masks

**Features:**
- Intelligent face cycling to avoid repetitive use of the 600+ available photographs
- Class imbalance detection with warnings
- Deterministic card-to-face distribution for reproducible datasets

### `PDF_Extractor`
Extracts high-resolution images from PDF documents.

**Key Methods:**
- `extract_images_from_directory(extract_all_pages=False)`: Yields images from all PDFs in a directory
- `extract_first_page_image(pdf_path)`: Extracts single-page ID document images
- `_render_pdf_pages(pdf_path, all_pages=False)`: Renders pages with configurable DPI scaling

**Configuration:**
- 3.0× DPI multiplier for high-quality forensic analysis
- In-memory Pixmap storage for efficiency

### `ELA_Generator`
Generates Error Level Analysis images for forensic document authentication.

**Key Methods:**
- `generate_ela(image_input, output_path, quality, target_size)`: Creates ELA maps from images/paths
- `pixmap_to_pil(pixmap, target_size)`: Converts PyMuPDF Pixmap objects to PIL Images

**ELA Process:**
1. Resizes image to 256×256 pixels
2. Compresses image to JPEG at specified quality
3. Calculates pixel-level difference (original vs. recompressed)
4. Normalizes brightness to enhance compression artifacts
5. Outputs forensic visualization map

## 📊 Data Pipeline

```
PDFs / Synthetic Templates
        ↓
   PDF_Extractor / SyntheticDataGenerator
        ↓
    Clean Images (256×256)
        ↓
   ┌─────────────────────────┐
   ↓                         ↓
Branch A: Images       Branch B: ELA Maps
(CNN Input)           (Forensic Analysis)
        ↓                     ↓
     CNN Model (Dual-Branch Architecture)
        ↓
  Forgery Classification
```

## 🚀 Getting Started

### Prerequisites
```
Python 3.8+
Dependencies listed in requirements.txt
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Configure your Python environment
pyenv shell ml-venv
```

### Project Structure
```
├── main.py                          # PDF extraction pipeline orchestration
├── requirements.txt                 # Python dependencies
├── data/
│   ├── data_generator.py           # Synthetic document generation
│   ├── test_data_generator.py      # Testing utilities
│   ├── photos/cfd/                 # Face image database (~600 images)
│   ├── templates/                  # ID card template image
│   ├── fonts/                      # Typography assets
│   └── synthetic/
│       ├── generated/
│       │   ├── authentic/          # Genuine synthetic ID cards
│       │   ├── forged/             # Forged document variants
│       │   └── masks/              # Tampering region masks (binary)
│       └── metadata/               # Document metadata storage
└── pdf_preprocessing/
    ├── pre_processors.py           # PDF_Extractor and ELA_Generator
    └── __init__.py
```

## 📖 Usage Examples

### Generate Synthetic Training Dataset
```python
from data.data_generator import SyntheticDataGenerator

# Create 1000 authentic and forged documents
generator = SyntheticDataGenerator(num_records=1000)
data = generator.generate_faker_dictionaries()
generator.create_authentic_id_cards(data)

# Generate 400 forged documents across 2 classes
generator.create_forged_id_cards({
    1: 200,  # Photo replacement forgeries
    2: 200   # Text alteration forgeries
})
```

### Extract PDF Documents for Analysis
```python
from pdf_preprocessing.pre_processors import PDF_Extractor, ELA_Generator

# Initialize extractors
extractor = PDF_Extractor(pdf_directory="path/to/pdfs")
ela_gen = ELA_Generator(default_quality=90)

# Process all PDFs
for pdf_filename, page_num, pixmap in extractor.extract_images_from_directory():
    # Convert to PIL Image
    image = ela_gen.pixmap_to_pil(pixmap, target_size=256)
    
    # Generate ELA map
    ela_gen.generate_ela(image, "output_ela.jpg", target_size=256)
```

## 📈 System Architecture

**Dual-Branch CNN Approach:**
- **Branch A**: Raw image analysis (learns authentic document patterns)
- **Branch B**: ELA map analysis (detects compression artifacts from tampering)
- **Fusion**: Concatenated feature maps for final classification

**Supported Forgery Classes:**
- Authentic documents
- Photo replacement (compression-based attacks)
- Text-field alteration (erasure and rewriting)
- Region overlay attacks (date field manipulation)

## 📝 License

Academic project for document forensics research.

---

**Last Updated:** March 2026 | **Python:** 3.8+ | **Key Technologies:** PyMuPDF, Pillow, NumPy, OpenCV, Faker