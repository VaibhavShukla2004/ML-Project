import os
from torch.utils.data import Dataset
from PIL import Image

LABEL_MAP = {
    'authentic': 0,
    'class1': 1,
    'class2': 2,
    'class4': 3,
}

class DocumentDataset(Dataset):
    def __init__(self, authentic_dir, forged_dir, transform=None):
        self.transform = transform
        self.samples = []  # List of (filepath, label) tuples

        if not os.path.isdir(authentic_dir):
            raise ValueError(f"Authentic directory not found: {authentic_dir}")
        if not os.path.isdir(forged_dir):
            raise ValueError(f"Forged directory not found: {forged_dir}")

        # Load authentic samples
        for fname in sorted(os.listdir(authentic_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((
                    os.path.join(authentic_dir, fname),
                    LABEL_MAP['authentic']
                ))

        # Load forged samples — infer class from filename prefix
        for fname in sorted(os.listdir(forged_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                prefix = fname.split('_')[0]  # 'class1', 'class2', 'class4'
                if prefix in LABEL_MAP:
                    self.samples.append((
                        os.path.join(forged_dir, fname),
                        LABEL_MAP[prefix]
                    ))
                else:
                    print(f"Warning: unrecognised prefix in {fname}, skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label