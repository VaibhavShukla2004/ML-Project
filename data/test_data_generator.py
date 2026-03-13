from data_generator import SyntheticDataGenerator
import os

def generate_ml_dataset():
    """Generate a full synthetic dataset for the ML pipeline."""
    num_records = 1000
    class_counts = {
        1: 500,
        2: 250,
        4: 250,
    }
    total_forged = sum(class_counts.values())
    
    # Step 1: Initialize generator for full dataset generation
    print("=" * 60)
    print(f"STEP 1: Creating Synthetic Data Generator ({num_records} records)")
    print("=" * 60)
    generator = SyntheticDataGenerator(num_records=num_records)
    
    # Step 2: Generate synthetic data
    print("\nSTEP 2: Generating Faker Data")
    print("-" * 60)
    data = generator.generate_faker_dictionaries()
    
    # Inspect the generated data
    for citizen_id, details in list(data.items())[:2]:  # Show first 2
        print(f"\nCitizen ID: {citizen_id}")
        print(f"  Name: {details['name']}")
        print(f"  DOB: {details['date_of_birth']}")
        print(f"  Country: {details['country']}")
        print(f"  Issue: {details['date_of_issue']}")
        print(f"  Expiry: {details['date_of_expiry']}")
    
    # Step 3: Create authentic ID cards
    print("\n" + "=" * 60)
    print(f"STEP 3: Creating Authentic ID Cards ({num_records} cards)")
    print("=" * 60)
    generator.create_authentic_id_cards(data)
    
    # Verify authentic cards were created
    authentic_count = len([f for f in os.listdir(generator.authentic_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg'))])
    print(f"✓ Created {authentic_count} authentic ID cards")
    
    # Step 4: Create forged ID cards for the training pipeline
    print("\n" + "=" * 60)
    print("STEP 4: Creating Forged ID Cards (ML Split)")
    print("=" * 60)

    print(f"Forging distribution: {class_counts}")
    print(f"Total forged samples: {total_forged}")
    generator.create_forged_id_cards(class_counts)
    
    # Verify forged cards and masks
    print("\n" + "=" * 60)
    print("STEP 5: Verification Summary")
    print("=" * 60)
    
    forged_count = len([f for f in os.listdir(generator.forged_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg'))])
    masks_count = len([f for f in os.listdir(generator.masks_dir) 
                      if f.lower().endswith('.png')])
    
    print(f"✓ Authentic cards: {authentic_count}")
    print(f"✓ Forged cards: {forged_count}")
    print(f"✓ Mask files: {masks_count}")
    print(f"\nAuthentic dir: {generator.authentic_dir}")
    print(f"Forged dir: {generator.forged_dir}")
    print(f"Masks dir: {generator.masks_dir}")

if __name__ == "__main__":
    generate_ml_dataset()