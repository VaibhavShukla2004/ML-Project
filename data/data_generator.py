import calendar
from faker import Faker
import random
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import os
from io import BytesIO

# Define template coordinates for text and photo placement
TEMPLATE_COORDINATES ={
    'name': (690, 310),
    'photo': (75, 250),
    'photo_size': (245, 375),
    'dob': (690, 440),
    'country': (690, 690),
    'citizen_id': (690, 570),
    'date_of_issue': (690, 810),
    'date_of_expiry': (690, 960)
}

# Shared artifact settings used by text-forgery classes
FORGERY_ARTIFACT_CONFIG = {
    'patch_jpeg_quality': 78,
    'forged_jpeg_quality': 90,
    'class1_face_jpeg_quality': 60,
    'class1_geom_offset_range': (-3, 3),
    'class1_geom_size_delta_range': (-4, 4),
    'class1_brightness_range': (0.88, 1.12),
    'class1_color_range': (0.85, 1.18),
    'class1_tint_strength_range': (0.0, 0.08),
    'patch_brightness_range': (0.97, 1.03),
    'name_y_jitter_range': (1, 2),
    'expiry_y_jitter_range': (0, 2),
    'name_patch_padding': 10,
    'expiry_patch_padding': 15,
    'forged_text_colors': ('#2a2a2a', '#0f0f0f'),
    'forged_text_blur_radius_range': (0.5, 1.0),
    'ghost_text_opacity_range': (0.03, 0.05),
    'ghost_text_colors': ('#161616', '#222222'),
    'textbox_jpeg_quality_range': (70, 82),
}


def add_years_safe(date_value, years):
    """Add calendar years without failing on leap-day or short-month edge cases."""
    target_year = date_value.year + years
    try:
        return date_value.replace(year=target_year)
    except ValueError:
        last_day = calendar.monthrange(target_year, date_value.month)[1]
        return date_value.replace(year=target_year, day=min(date_value.day, last_day))

# Synthetic Data Generator Class
class SyntheticDataGenerator:

    #Define directories and load fonts in the constructor, also initialize Faker instance for data generation
    def __init__(self, num_records, text_forgery_artifact_config=None):
        self.num_records = num_records
        self.artifact_cfg = FORGERY_ARTIFACT_CONFIG.copy()
        if text_forgery_artifact_config:
            self.artifact_cfg.update(text_forgery_artifact_config)
        self.faker = Faker()
        self.data = {}  # Store lookup for all citizen data (name, dob, country, dates, etc.)
        self.authentic_dir = os.path.join(os.getcwd(), 'synthetic', 'generated', 'authentic')
        self.forged_dir = os.path.join(os.getcwd(), 'synthetic', 'generated', 'forged')
        self.masks_dir = os.path.join(os.getcwd(), 'synthetic', 'generated', 'masks')
        self.font_dir = os.path.join(os.getcwd(), 'fonts')
        self.photos_dir = os.path.join(os.getcwd(), 'photos', 'cfd')
        self.template_path = os.path.join(os.getcwd(), 'templates', 'base_template_v1.png')
        os.makedirs(self.authentic_dir, exist_ok=True)
        os.makedirs(self.forged_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        self.font_main = ImageFont.truetype(os.path.join(self.font_dir, "LiberationSans-Regular.ttf"), 42)
        self.font_mono = ImageFont.truetype(os.path.join(self.font_dir, "LiberationMono-Regular.ttf"), 46)
        self.forged_text_font = self._load_forged_text_font()

    def _load_forged_text_font(self):
        """Prefer a subtly different sans-serif font for forged text, fallback safely."""
        candidate_fonts = [
            "Arial.ttf",
            "Helvetica.ttf",
            "DejaVuSans.ttf",
            "LiberationSans-Bold.ttf",
            "LiberationSans-Regular.ttf",
        ]
        for font_name in candidate_fonts:
            font_path = os.path.join(self.font_dir, font_name)
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, 42)
        return self.font_main

    def _draw_forged_text(self, base_image, position, text):
        """Render forged text on a separate layer and blur only that layer."""
        text_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        forged_color = random.choice(self.artifact_cfg['forged_text_colors'])
        draw.text(position, text, font=self.forged_text_font, fill=forged_color)
        min_blur, max_blur = self.artifact_cfg['forged_text_blur_radius_range']
        blur_radius = random.uniform(min_blur, max_blur)
        text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        merged = Image.alpha_composite(base_image.convert("RGBA"), text_layer)
        return merged.convert("RGB")

    def _draw_ghost_text(self, base_image, position, text, font=None):
        """Render faint residue of original text before forged overwrite."""
        if font is None:
            font = self.font_main
        min_opacity, max_opacity = self.artifact_cfg['ghost_text_opacity_range']
        ghost_alpha = int(255 * random.uniform(min_opacity, max_opacity))
        ghost_color = random.choice(self.artifact_cfg['ghost_text_colors'])

        ghost_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ghost_layer)
        draw.text(position, text, font=font, fill=ghost_color + f"{ghost_alpha:02x}")

        merged = Image.alpha_composite(base_image.convert("RGBA"), ghost_layer)
        return merged.convert("RGB")

    def _apply_local_textbox_compression(self, base_image, box):
        """Apply local JPEG recompression around forged text to create ELA-visible edge ringing."""
        q_min, q_max = self.artifact_cfg['textbox_jpeg_quality_range']
        local_quality = random.randint(q_min, q_max)

        x1, y1, x2, y2 = box
        local_patch = base_image.crop((x1, y1, x2, y2))
        patch_buffer = BytesIO()
        local_patch.save(patch_buffer, "JPEG", quality=local_quality)
        patch_buffer.seek(0)
        local_patch = Image.open(patch_buffer).convert("RGB")
        base_image.paste(local_patch, (x1, y1, x2, y2))
        return base_image

    def _render_authentic_base_card(self, citizen_id, details, face_image_filename):
        """Create one authentic-style card in memory directly from clean template."""
        base_image = Image.open(self.template_path).convert("RGB")
        draw = ImageDraw.Draw(base_image)
        text_color = "#1a1a1a"

        draw.text(TEMPLATE_COORDINATES['name'], details['name'], font=self.font_main, fill=text_color)
        draw.text(TEMPLATE_COORDINATES['dob'], details['date_of_birth'], font=self.font_main, fill=text_color)
        draw.text(TEMPLATE_COORDINATES['country'], details['country'], font=self.font_main, fill=text_color)
        draw.text(TEMPLATE_COORDINATES['date_of_issue'], details['date_of_issue'], font=self.font_main, fill=text_color)
        draw.text(TEMPLATE_COORDINATES['date_of_expiry'], details['date_of_expiry'], font=self.font_main, fill=text_color)
        draw.text(TEMPLATE_COORDINATES['citizen_id'], citizen_id, font=self.font_mono, fill=text_color)

        face_image = Image.open(os.path.join(self.photos_dir, face_image_filename)).convert("RGB")
        face_image = face_image.resize(TEMPLATE_COORDINATES['photo_size'], Image.Resampling.LANCZOS)
        base_image.paste(face_image, TEMPLATE_COORDINATES['photo'])
        return base_image

    def _apply_class1_lighting_mismatch(self, face_image):
        """Apply subtle local color/brightness mismatch to forged face only."""
        min_brightness, max_brightness = self.artifact_cfg['class1_brightness_range']
        min_color, max_color = self.artifact_cfg['class1_color_range']
        min_tint, max_tint = self.artifact_cfg['class1_tint_strength_range']

        face_image = ImageEnhance.Brightness(face_image).enhance(
            random.uniform(min_brightness, max_brightness)
        )
        face_image = ImageEnhance.Color(face_image).enhance(
            random.uniform(min_color, max_color)
        )

        # Add a weak random warm/cool tint to create contextual lighting mismatch.
        tint_alpha = random.uniform(min_tint, max_tint)
        tint_color = random.choice([(235, 225, 210), (205, 220, 240)])
        tint_layer = Image.new("RGB", face_image.size, tint_color)
        return Image.blend(face_image, tint_layer, tint_alpha)

    def _get_class1_paste_box(self, card_size):
        """Return a slightly mismatched photo box to expose seam artifacts."""
        photo_x, photo_y = TEMPLATE_COORDINATES['photo']
        photo_w, photo_h = TEMPLATE_COORDINATES['photo_size']

        min_offset, max_offset = self.artifact_cfg['class1_geom_offset_range']
        min_delta, max_delta = self.artifact_cfg['class1_geom_size_delta_range']

        dx = random.randint(min_offset, max_offset)
        dy = random.randint(min_offset, max_offset)
        dw = random.randint(min_delta, max_delta)
        dh = random.randint(min_delta, max_delta)

        paste_x = max(0, photo_x + dx)
        paste_y = max(0, photo_y + dy)
        paste_w = max(20, photo_w + dw)
        paste_h = max(20, photo_h + dh)

        card_w, card_h = card_size
        paste_w = min(paste_w, card_w - paste_x)
        paste_h = min(paste_h, card_h - paste_y)

        return (paste_x, paste_y, paste_x + paste_w, paste_y + paste_h)

    # Generate synthetic data using Faker for the required fields in template
    def generate_faker_dictionaries(self):
        print("Generating synthetic data using Faker...")
        data = {}
        for _ in range(self.num_records):
            name = self.faker.name().upper()
            dob = self.faker.date_of_birth(minimum_age=18, maximum_age=65)
            country = self.faker.country()[:10].upper()
            date_of_issue = self.faker.date_between(start_date='-10y', end_date='today')
            date_of_expiry = add_years_safe(date_of_issue, 10)
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
        # Store the data as instance variable for use in forgery functions
        self.data = data
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
            try:
                random_face = random.choice(available_faces)
                base_image = self._render_authentic_base_card(citizen_id, details, random_face)
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
        
        # Ensure output directories exist
        os.makedirs(self.forged_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        
        if not authentic_cards:
            print("No authentic ID cards found. Please generate authentic cards first.")
            return
        
        if not available_faces:
            print("No photos found in the photos directory.")
            return
        
        if not self.data:
            raise ValueError(
                "self.data is empty. Run generate_faker_dictionaries() and create_authentic_id_cards() "
                "in this session before creating forgeries."
            )

        # Validate total forged count doesn't exceed available authentic cards
        total_forged = sum(class_counts.values())
        if total_forged > len(authentic_cards):
            raise ValueError(f"Total forged count ({total_forged}) exceeds available authentic cards ({len(authentic_cards)}). "
                           f"Maximum possible: {len(authentic_cards)}")
        
        # Check for class imbalance (warn if max/min ratio > 3)
        class_values = [class_counts.get(cls, 0) for cls in [1, 2, 3, 4, 5]]
        non_zero_classes = [v for v in class_values if v > 0]
        if non_zero_classes:
            max_class = max(non_zero_classes)
            min_class = min(non_zero_classes)
            balance_ratio = max_class / min_class if min_class > 0 else float('inf')
            if balance_ratio > 3:
                print(f"WARNING: Class imbalance detected! Max/Min ratio: {balance_ratio:.2f}x")
                print(f"         Consider more balanced distribution: {class_counts}")
        
        # Initialize face cycling to avoid excessive repetition
        # Each face will appear at most twice before all 600 faces are exhausted
        shuffled_faces = available_faces.copy()
        random.shuffle(shuffled_faces)
        face_index = 0
        
        def get_next_face():
            nonlocal face_index, shuffled_faces
            if face_index >= len(shuffled_faces):
                shuffled_faces = available_faces.copy()
                random.shuffle(shuffled_faces)
                face_index = 0
            face = shuffled_faces[face_index]
            face_index += 1
            return face
        
        # Distribute authentic cards cyclically to prevent same card from being used multiple times
        random.shuffle(authentic_cards)
        card_index = 0
        
        def get_next_authentic_card():
            nonlocal card_index
            if card_index >= len(authentic_cards):
                card_index = 0  # Wrap around if necessary (only happens if total_forged == len(authentic_cards))
            card = authentic_cards[card_index]
            card_index += 1
            return card
        
        # Class 1: Photo Replacement - Replace face with subtle geometry + lighting mismatch artifacts
        class_1_count = class_counts.get(1, 0)
        if class_1_count > 0:
            print(f"Generating {class_1_count} Class 1 forged ID cards (Photo Replacement)...")
            
            for i in range(class_1_count):
                # Deterministically cycle through authentic cards
                authentic_card_filename = get_next_authentic_card()
                source_id = os.path.splitext(authentic_card_filename)[0]
                source_details = self.data.get(source_id)
                if source_details is None:
                    print(f"Skipping {source_id}: no source metadata found in self.data")
                    continue
                
                # Build forged sample directly from clean template in memory
                source_face = get_next_face()
                id_card = self._render_authentic_base_card(source_id, source_details, source_face)
                
                # Cycle through faces to avoid heavy repetition
                face_image_filename = get_next_face()
                face_image = Image.open(os.path.join(self.photos_dir, face_image_filename)).convert("RGB")
                
                # Resize to target slot and apply local lighting/color mismatch before compression
                face_image = face_image.resize(TEMPLATE_COORDINATES['photo_size'], Image.Resampling.LANCZOS)
                face_image = self._apply_class1_lighting_mismatch(face_image)
                
                # Compress the face to 60% quality in memory
                compressed_face_buffer = BytesIO()
                face_image.save(compressed_face_buffer, "JPEG", quality=self.artifact_cfg['class1_face_jpeg_quality'])
                compressed_face_buffer.seek(0)
                compressed_face = Image.open(compressed_face_buffer).convert("RGB")
                
                # Paste with slight box mismatch to expose seam/boundary anomalies.
                paste_box = self._get_class1_paste_box(id_card.size)
                paste_w = paste_box[2] - paste_box[0]
                paste_h = paste_box[3] - paste_box[1]
                compressed_face = compressed_face.resize((paste_w, paste_h), Image.Resampling.LANCZOS)
                id_card.paste(compressed_face, (paste_box[0], paste_box[1]))
                
                # Save the forged ID card
                forged_filename = f"class1_{i+1:04d}__src__{source_id}.jpg"
                output_path = os.path.join(self.forged_dir, forged_filename)
                id_card.save(output_path, "JPEG", quality=90)
                
                # Create a binary mask showing the forged region (white = forged, black = authentic)
                mask = Image.new("L", id_card.size, 0)  # Black background
                draw_mask = ImageDraw.Draw(mask)
                draw_mask.rectangle([paste_box[0], paste_box[1], paste_box[2], paste_box[3]], fill=255)
                
                mask_path = os.path.join(self.masks_dir, forged_filename.replace('.jpg', '_mask.png'))
                mask.save(mask_path, "PNG")
            
            print(f"Generated {class_1_count} Class 1 forged ID cards in {self.forged_dir}")

        # Class 2: Text Alteration (Name) - Erase the original name and replace with a new one with slight misalignment
        class_2_count = class_counts.get(2, 0)
        if class_2_count > 0:
            print(f"Generating {class_2_count} Class 2 forged ID cards (Name Alteration)...")
            clean_template = Image.open(self.template_path).convert("RGB")
            
            for i in range(class_2_count):
                # Deterministically cycle through authentic cards
                authentic_card_filename = get_next_authentic_card()
                # Extract citizen_id from filename (remove extension)
                citizen_id = os.path.splitext(authentic_card_filename)[0]
                source_details = self.data.get(citizen_id)
                if source_details is None:
                    print(f"Skipping {citizen_id}: no source metadata found in self.data")
                    continue
                
                source_face = get_next_face()
                id_card = self._render_authentic_base_card(citizen_id, source_details, source_face)
                
                # Get the name coordinates
                name_x, name_y = TEMPLATE_COORDINATES['name']
                
                # Get the original name from the lookup to calculate exact erase box size
                original_name = source_details['name']
                
                # Draw text to calculate bounding box of the ACTUAL original name
                test_draw = ImageDraw.Draw(id_card)
                test_bbox = test_draw.textbbox((name_x, name_y), original_name, font=self.font_main)
                name_width = test_bbox[2] - test_bbox[0]
                name_height = test_bbox[3] - test_bbox[1]
                
                # Define padding around the name for the erase patch
                padding = self.artifact_cfg['name_patch_padding']
                erase_box = (name_x - padding, name_y - padding, name_x + name_width + padding, name_y + name_height + padding)
                
                # Extract the exact matching background from the clean template
                erase_box_int = (int(erase_box[0]), int(erase_box[1]), int(erase_box[2]), int(erase_box[3]))
                texture_patch = clean_template.crop(erase_box_int)

                # Add subtle local tampering artifacts for learnable ELA signals
                patch_buffer = BytesIO()
                texture_patch.save(patch_buffer, "JPEG", quality=self.artifact_cfg['patch_jpeg_quality'])
                patch_buffer.seek(0)
                texture_patch = Image.open(patch_buffer).convert("RGB")
                enhancer = ImageEnhance.Brightness(texture_patch)
                min_brightness, max_brightness = self.artifact_cfg['patch_brightness_range']
                texture_patch = enhancer.enhance(random.uniform(min_brightness, max_brightness))
                
                # Paste the perfect texture patch over the original name to erase it
                id_card.paste(texture_patch, erase_box_int)
                
                # Generate a new name
                new_name = self.faker.name().upper()

                # Keep faint residue of original text under the forged text.
                id_card = self._draw_ghost_text(id_card, (name_x, name_y), original_name, font=self.font_main)
                
                # Draw forged name with slight Y shift, altered font/color, and local blur
                min_name_jitter, max_name_jitter = self.artifact_cfg['name_y_jitter_range']
                new_name_y = name_y + random.randint(min_name_jitter, max_name_jitter)
                id_card = self._draw_forged_text(id_card, (name_x, new_name_y), new_name)
                id_card = self._apply_local_textbox_compression(id_card, erase_box_int)
                
                # Save the forged ID card
                forged_filename = f"class2_{i+1:04d}__src__{citizen_id}.jpg"
                output_path = os.path.join(self.forged_dir, forged_filename)
                id_card.save(output_path, "JPEG", quality=self.artifact_cfg['forged_jpeg_quality'])
                
                # Create a binary mask showing the forged region (white = forged, black = authentic)
                mask = Image.new("L", id_card.size, 0)  # Black background
                draw_mask = ImageDraw.Draw(mask)
                draw_mask.rectangle(erase_box_int, fill=255)
                
                mask_path = os.path.join(self.masks_dir, forged_filename.replace('.jpg', '_mask.png'))
                mask.save(mask_path, "PNG")
            
            print(f"Generated {class_2_count} Class 2 forged ID cards in {self.forged_dir}")

        # Class 3: [Placeholder for future forgery class]
        class_3_count = class_counts.get(3, 0)
        if class_3_count > 0:
            print(f"Generating {class_3_count} Class 3 forged ID cards...")
            # TODO: Implement Class 3 forgery logic
            for i in range(class_3_count):
                authentic_card_filename = get_next_authentic_card()
                # TODO: Add Class 3 implementation
            print(f"Generated {class_3_count} Class 3 forged ID cards in {self.forged_dir}")

        # Class 4: Region Overlay (Date of Expiry) - Cover the entire Date of Expiry area with white rectangle and write a fake date
        class_4_count = class_counts.get(4, 0)
        if class_4_count > 0:
            print(f"Generating {class_4_count} Class 4 forged ID cards (Region Overlay)...")
            clean_template = Image.open(self.template_path).convert("RGB")
            
            for i in range(class_4_count):
                # Deterministically cycle through authentic cards
                authentic_card_filename = get_next_authentic_card()
                source_id = os.path.splitext(authentic_card_filename)[0]
                source_details = self.data.get(source_id)
                if source_details is None:
                    print(f"Skipping {source_id}: no source metadata found in self.data")
                    continue
                
                source_face = get_next_face()
                id_card = self._render_authentic_base_card(source_id, source_details, source_face)
                
                # Get the date of expiry coordinates
                expiry_x, expiry_y = TEMPLATE_COORDINATES['date_of_expiry']
                
                # Calculate bounding box for the date of expiry field
                test_draw = ImageDraw.Draw(id_card)
                test_bbox = test_draw.textbbox((expiry_x, expiry_y), "99 DEC 2099", font=self.font_main)
                expiry_width = test_bbox[2] - test_bbox[0]
                expiry_height = test_bbox[3] - test_bbox[1]
                
                # Define padded bounding box to cover the entire area (including guilloche)
                padding = self.artifact_cfg['expiry_patch_padding']
                overlay_box = (int(expiry_x - padding), int(expiry_y - padding), 
                              int(expiry_x + expiry_width + padding), int(expiry_y + expiry_height + padding))
                
                # Replace with a matching template patch, then add subtle local artifacts
                texture_patch = clean_template.crop(overlay_box)
                patch_buffer = BytesIO()
                texture_patch.save(patch_buffer, "JPEG", quality=self.artifact_cfg['patch_jpeg_quality'])
                patch_buffer.seek(0)
                texture_patch = Image.open(patch_buffer).convert("RGB")
                enhancer = ImageEnhance.Brightness(texture_patch)
                min_brightness, max_brightness = self.artifact_cfg['patch_brightness_range']
                texture_patch = enhancer.enhance(random.uniform(min_brightness, max_brightness))
                id_card.paste(texture_patch, overlay_box)
                
                # Generate a fake date of expiry
                fake_issue_date = self.faker.date_between(start_date='-10y', end_date='today')
                fake_expiry_date = add_years_safe(fake_issue_date, 10)
                fake_expiry_str = fake_expiry_date.strftime('%d %b %Y').upper()
                original_expiry_str = source_details['date_of_expiry']

                # Add faint residue of original expiry text below forged value.
                id_card = self._draw_ghost_text(id_card, (expiry_x, expiry_y), original_expiry_str, font=self.font_main)
                
                # Draw forged expiry text with subtle font/color/blur mismatch
                min_expiry_jitter, max_expiry_jitter = self.artifact_cfg['expiry_y_jitter_range']
                fake_expiry_y = expiry_y + random.randint(min_expiry_jitter, max_expiry_jitter)
                id_card = self._draw_forged_text(id_card, (expiry_x, fake_expiry_y), fake_expiry_str)
                id_card = self._apply_local_textbox_compression(id_card, overlay_box)
                
                # Save the forged ID card
                forged_filename = f"class4_{i+1:04d}__src__{source_id}.jpg"
                output_path = os.path.join(self.forged_dir, forged_filename)
                id_card.save(output_path, "JPEG", quality=self.artifact_cfg['forged_jpeg_quality'])
                
                # Create a binary mask showing the forged region (white = forged, black = authentic)
                mask = Image.new("L", id_card.size, 0)  # Black background
                draw_mask = ImageDraw.Draw(mask)
                draw_mask.rectangle(overlay_box, fill=255)
                
                mask_path = os.path.join(self.masks_dir, forged_filename.replace('.jpg', '_mask.png'))
                mask.save(mask_path, "PNG")
            
            print(f"Generated {class_4_count} Class 4 forged ID cards in {self.forged_dir}")

        # Class 5: [Placeholder for future forgery class]
        class_5_count = class_counts.get(5, 0)
        if class_5_count > 0:
            print(f"Generating {class_5_count} Class 5 forged ID cards...")
            # TODO: Implement Class 5 forgery logic
            for i in range(class_5_count):
                authentic_card_filename = get_next_authentic_card()
                # TODO: Add Class 5 implementation
            print(f"Generated {class_5_count} Class 5 forged ID cards in {self.forged_dir}")

