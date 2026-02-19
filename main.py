import os
import gc
from pathlib import Path
from pdf_preprocessing.pre_processors import PDF_Extractor, ELA_Generator

def setup_directories(base_dir="data"):
    """Creates the necessary data directory structure if it doesn't exist."""
    paths = {
        "raw": Path(base_dir) / "raw_pdfs",
        "images": Path(base_dir) / "processed_images",
        "elas": Path(base_dir) / "processed_elas"
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        
    return paths

def main():
    # 1. Setup Data Paths
    paths = setup_directories()
    
    # 2. Initialize the Pipeline Classes
    extractor = PDF_Extractor(pdf_directory=str(paths["raw"]))
    ela_gen = ELA_Generator(default_quality=90)
    
    print(f"Starting extraction pipeline from: {paths['raw']}")
    
    # 3. Check if there are PDFs to process
    try:
        pdf_files = extractor._get_pdf_files()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if not pdf_files:
        print("No PDFs found in raw_pdfs directory!")
        return
    
    print(f"Found {len(pdf_files)} PDF(s) to process.\n")
    
    # 4. Initialize statistics tracking
    processed = 0
    failed = 0
    skipped = 0
    
    # 5. The Orchestration Loop
    # We grab the first page of each PDF since IDs are typically single-page
    for pdf_filename, page_num, pixmap in extractor.extract_images_from_directory(extract_all_pages=False):
        base_name = os.path.splitext(pdf_filename)[0]
        
        # Prepare output paths
        image_output_path = paths["images"] / f"{base_name}_page{page_num}.jpg"
        ela_output_path = paths["elas"] / f"{base_name}_page{page_num}_ela.jpg"
        
        # Resume Capability: Skip if both outputs already exist
        if image_output_path.exists() and ela_output_path.exists():
            print(f"Skipping: {pdf_filename} (Page {page_num}) - Already processed")
            skipped += 1
            gc.collect()  # Free memory from pixmap
            continue
        
        print(f"Processing: {pdf_filename} (Page {page_num})")
        
        try:
            # A. Convert PyMuPDF Pixmap to PIL Image & Resize
            clean_image = ela_gen.pixmap_to_pil(pixmap, target_size=256)
            
            # B. Save the Clean Image (CNN Branch A)
            clean_image.save(image_output_path, "JPEG")
            
            # C. Generate and Save the ELA Map (CNN Branch B)
            ela_gen.generate_ela(
                image_input=clean_image, 
                output_path=str(ela_output_path),
                target_size=256
            )
            
            processed += 1
            
        except Exception as e:
            print(f"Failed to process {pdf_filename}: {e}")
            failed += 1
            
        finally:
            # Memory cleanup after each PDF
            del pixmap
            gc.collect()
            
    # Print final statistics
    print(f"\n{'='*50}")
    print(f"Phase 1 Pipeline Complete!")
    print(f"{'='*50}")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Total:     {processed + skipped + failed}")
    print(f"\nYour data is ready for the neural network.")

if __name__ == "__main__":
    main()