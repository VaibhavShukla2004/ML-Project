from faker import Faker
import random
from PIL import Image, ImageDraw, ImageFont
import os
from io import BytesIO

# Define template coordinates for text and photo placement
TEMPLATE_COORDINATES ={
    'name': (700, 320),
    'photo': (75, 250),
    'photo_size': (245, 375),
    'dob': (700, 460),
    'country': (696, 740),
    'citizen_id': (700, 600),
    'date_of_issue': (700, 880),
    'date_of_expiry': (700, 1020)
}

# Synthetic Data Generator Class
class SyntheticDataGenerator:

    #Define directories and load fonts in the constructor, also initialize Faker instance for data generation
    def __init__(self, num_records):
        self.num_records = num_records
        self.faker = Faker()
        self.authentic_dir = os.path.join(os.getcwd(), 'synthetic', 'generated', 'authentic')
        self.forged_dir = os.path.join(os.getcwd(), 'synthetic', 'generated', 'forged')
        self.font_dir = os.path.join(os.getcwd(), 'fonts')
        self.photos_dir = os.path.join(os.getcwd(), 'photos')
        self.template_path = os.path.join(os.getcwd(), 'templates', 'base_template_v1.png')
        os.makedirs(self.authentic_dir, exist_ok=True)
        os.makedirs(self.forged_dir, exist_ok=True)
        self.font_main = ImageFont.truetype(os.path.join(self.font_dir, "LiberationSans-Regular.ttf"), 42)
        self.font_mono = ImageFont.truetype(os.path.join(self.font_dir, "LiberationMono-Regular.ttf"), 46)

    # Generate synthetic data using Faker for the required fields in template
    def generate_faker_dictionaries(self):
        print("Generating synthetic data using Faker...")
        data = {}
        for _ in range(self.num_records):
            name = self.faker.name().upper()
            dob = self.faker.date_of_birth(minimum_age=18, maximum_age=65)
            country = self.faker.country()[:10].upper()
            date_of_issue = self.faker.date_between(start_date='-10y', end_date='today')
            date_of_expiry = date_of_issue.replace(year=date_of_issue.year + 10)
            yy = date_of_issue.strftime('%y')
            rr = f"{random.randint(1, 99):02d}"
            ssss = f"{random.randint(1, 9999):04d}"
            citizen_id = f"NCC-{yy}{rr}-{ssss}"
            data[citizen_id] = {
                'date_of_birth': dob.strftime('%d %b %Y').upper(),
                'country': country,
                'date_of_issue': date_of_issue.strftime('%d %b %Y').upper(),
                'date_of_expiry': date_of_expiry.strftime('%d %b %Y').upper(),
                'name': name
            }
        print(f"Generated {len(data)} synthetic records.")
        return data
    
    # Create authentic ID cards by overlaying text and photos onto the template using the generated data using PIL
    def create_authentic_id_cards(self, data):
        print("Generating authentic ID cards...")
        available_faces = [f for f in os.listdir(self.photos_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not available_faces:
            print("No photos found in the photos directory. Please add some face images to generate ID cards.")
            return
        
        for citizen_id, details in data.items():
            base_image = Image.open(self.template_path).convert("RGB")
            draw = ImageDraw.Draw(base_image)
            text_color = "#1a1a1a"

            #Inject text using the coordinates mapped from the template
            draw.text(TEMPLATE_COORDINATES['name'], details['name'], font=self.font_main, fill=text_color)
            draw.text(TEMPLATE_COORDINATES['dob'], details['date_of_birth'], font=self.font_main, fill=text_color)
            draw.text(TEMPLATE_COORDINATES['country'], details['country'], font=self.font_main, fill=text_color)
            draw.text(TEMPLATE_COORDINATES['date_of_issue'], details['date_of_issue'], font=self.font_main, fill=text_color)
            draw.text(TEMPLATE_COORDINATES['date_of_expiry'], details['date_of_expiry'], font=self.font_main, fill=text_color)
            
            #Mono font for the citizen ID to give it a more official look
            draw.text(TEMPLATE_COORDINATES['citizen_id'], citizen_id, font=self.font_mono, fill=text_color)

            try:
                random_face = random.choice(available_faces)
                face_image = Image.open(os.path.join(self.photos_dir, random_face)).convert("RGB")
                face_image = face_image.resize(TEMPLATE_COORDINATES['photo_size'], Image.Resampling.LANCZOS)
                base_image.paste(face_image, TEMPLATE_COORDINATES['photo'])
            except Exception as e:
                print(f"Error processing photo for {citizen_id}: {e}")
                continue

            #Save as JPEG for ELA and loss analysis for the CNN model
            output_path = os.path.join(self.authentic_dir, f"{citizen_id}.jpg")
            base_image.save(output_path, "JPEG", quality=90)
        print(f"Generated {len(data)} authentic ID cards in {self.authentic_dir}")

    # Create forged ID cards by applying random transformations to the authentic ID cards
    def create_forged_id_cards(self, class_counts):
        print("Generating forged ID cards...")

        # Get list of authentic ID cards and available face photos
        authentic_cards = [f for f in os.listdir(self.authentic_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        available_faces = [f for f in os.listdir(self.photos_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train_dir = os.path.join(self.forged_dir, 'train', 'images')
        train_masks_dir = os.path.join(self.forged_dir, 'train', 'masks')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(train_masks_dir, exist_ok=True)
        
        if not authentic_cards:
            print("No authentic ID cards found. Please generate authentic cards first.")
            return
        
        if not available_faces:
            print("No photos found in the photos directory.")
            return

        # Class 1: Photo Replacement - Replace the original photo with a random face from the photos directory, after 60% compression to simulate quality loss
        class_1_count = class_counts.get(1, 0)
        print(f"Generating {class_1_count} Class 1 forged ID cards (Photo Replacement)...")
        
        for i in range(class_1_count):
            # Select a random authentic ID card
            authentic_card_filename = random.choice(authentic_cards)
            authentic_card_path = os.path.join(self.authentic_dir, authentic_card_filename)
            
            # Open the authentic ID card
            id_card = Image.open(authentic_card_path).convert("RGB")
            
            # Select a random face image
            random_face = random.choice(available_faces)
            face_image = Image.open(os.path.join(self.photos_dir, random_face)).convert("RGB")
            
            # Resize the face to match the photo dimensions
            face_image = face_image.resize(TEMPLATE_COORDINATES['photo_size'], Image.Resampling.LANCZOS)
            
            # Compress the face to 60% quality in memory
            compressed_face_buffer = BytesIO()
            face_image.save(compressed_face_buffer, "JPEG", quality=60)
            compressed_face_buffer.seek(0)
            compressed_face = Image.open(compressed_face_buffer).convert("RGB")
            
            # Paste the compressed face over the photo coordinates
            id_card.paste(compressed_face, TEMPLATE_COORDINATES['photo'])
            
            # Save the forged ID card
            forged_filename = f"class1_{i+1:04d}.jpg"
            output_path = os.path.join(train_dir, forged_filename)
            id_card.save(output_path, "JPEG", quality=90)
            
            # Create a binary mask showing the forged region (white = forged, black = authentic)
            mask = Image.new("L", id_card.size, 0)  # Black background
            draw_mask = ImageDraw.Draw(mask)
            photo_x, photo_y = TEMPLATE_COORDINATES['photo']
            photo_w, photo_h = TEMPLATE_COORDINATES['photo_size']
            draw_mask.rectangle([photo_x, photo_y, photo_x + photo_w, photo_y + photo_h], fill=255)
            
            mask_path = os.path.join(train_masks_dir, forged_filename.replace('.jpg', '_mask.png'))
            mask.save(mask_path, "PNG")
        
        print(f"Generated {class_1_count} Class 1 forged ID cards in {train_dir}")

        # Class 2: Text Alteration (Name) - Erase the original name and replace with a new one with slight misalignment
        class_2_count = class_counts.get(2, 0)
        print(f"Generating {class_2_count} Class 2 forged ID cards (Name Alteration)...")
        
        for i in range(class_2_count):
            # Select a random authentic ID card
            authentic_card_filename = random.choice(authentic_cards)
            authentic_card_path = os.path.join(self.authentic_dir, authentic_card_filename)
            
            # Open the authentic ID card
            id_card = Image.open(authentic_card_path).convert("RGB")
            
            # Get the name coordinates
            name_x, name_y = TEMPLATE_COORDINATES['name']
            
            # Draw text to calculate bounding box of a typical name
            test_draw = ImageDraw.Draw(id_card)
            test_bbox = test_draw.textbbox((name_x, name_y), "PLACEHOLDER_NAME", font=self.font_main)
            name_width = test_bbox[2] - test_bbox[0]
            name_height = test_bbox[3] - test_bbox[1]
            
            # Define padding around the name for the erase patch
            padding = 10
            erase_box = (name_x - padding, name_y - padding, name_x + name_width + padding, name_y + name_height + padding)
            
            # Open the pristine blank template to steal a matching background patch
            clean_template = Image.open(self.template_path).convert("RGB")
            
            # Extract the exact matching background from the clean template
            erase_box_int = (int(erase_box[0]), int(erase_box[1]), int(erase_box[2]), int(erase_box[3]))
            texture_patch = clean_template.crop(erase_box_int)
            
            # Paste the perfect texture patch over the original name to erase it
            id_card.paste(texture_patch, erase_box_int)
            
            # Generate a new name
            new_name = self.faker.name().upper()
            
            # Draw the new name with a slight Y shift (simulate human misalignment)
            draw = ImageDraw.Draw(id_card)
            new_name_y = name_y + 2  # Shift Y by 2 pixels
            text_color = "#1a1a1a"
            draw.text((name_x, new_name_y), new_name, font=self.font_main, fill=text_color)
            
            # Save the forged ID card
            forged_filename = f"class2_{i+1:04d}.jpg"
            output_path = os.path.join(train_dir, forged_filename)
            id_card.save(output_path, "JPEG", quality=90)
            
            # Create a binary mask showing the forged region (white = forged, black = authentic)
            mask = Image.new("L", id_card.size, 0)  # Black background
            draw_mask = ImageDraw.Draw(mask)
            draw_mask.rectangle(erase_box_int, fill=255)
            
            mask_path = os.path.join(train_masks_dir, forged_filename.replace('.jpg', '_mask.png'))
            mask.save(mask_path, "PNG")
        
        print(f"Generated {class_2_count} Class 2 forged ID cards in {train_dir}")

        # Class 4: Region Overlay (Date of Expiry) - Cover the entire Date of Expiry area with white rectangle and write a fake date
        class_4_count = class_counts.get(4, 0)
        print(f"Generating {class_4_count} Class 4 forged ID cards (Region Overlay)...")
        
        for i in range(class_4_count):
            # Select a random authentic ID card
            authentic_card_filename = random.choice(authentic_cards)
            authentic_card_path = os.path.join(self.authentic_dir, authentic_card_filename)
            
            # Open the authentic ID card
            id_card = Image.open(authentic_card_path).convert("RGB")
            
            # Get the date of expiry coordinates
            expiry_x, expiry_y = TEMPLATE_COORDINATES['date_of_expiry']
            
            # Calculate bounding box for the date of expiry field
            test_draw = ImageDraw.Draw(id_card)
            test_bbox = test_draw.textbbox((expiry_x, expiry_y), "99 DEC 2099", font=self.font_main)
            expiry_width = test_bbox[2] - test_bbox[0]
            expiry_height = test_bbox[3] - test_bbox[1]
            
            # Define padded bounding box to cover the entire area (including guilloche)
            padding = 15
            overlay_box = (int(expiry_x - padding), int(expiry_y - padding), 
                          int(expiry_x + expiry_width + padding), int(expiry_y + expiry_height + padding))
            
            # Draw a solid white rectangle over the date of expiry area
            draw = ImageDraw.Draw(id_card)
            draw.rectangle(overlay_box, fill=(255, 255, 255))
            
            # Generate a fake date of expiry
            fake_issue_date = self.faker.date_between(start_date='-10y', end_date='today')
            fake_expiry_date = fake_issue_date.replace(year=fake_issue_date.year + 10)
            fake_expiry_str = fake_expiry_date.strftime('%d %b %Y').upper()
            
            # Draw the new date on top of the white rectangle
            text_color = "#1a1a1a"
            draw.text((expiry_x, expiry_y), fake_expiry_str, font=self.font_main, fill=text_color)
            
            # Save the forged ID card
            forged_filename = f"class4_{i+1:04d}.jpg"
            output_path = os.path.join(train_dir, forged_filename)
            id_card.save(output_path, "JPEG", quality=90)
            
            # Create a binary mask showing the forged region (white = forged, black = authentic)
            mask = Image.new("L", id_card.size, 0)  # Black background
            draw_mask = ImageDraw.Draw(mask)
            draw_mask.rectangle(overlay_box, fill=255)
            
            mask_path = os.path.join(train_masks_dir, forged_filename.replace('.jpg', '_mask.png'))
            mask.save(mask_path, "PNG")
        
        print(f"Generated {class_4_count} Class 4 forged ID cards in {train_dir}")

