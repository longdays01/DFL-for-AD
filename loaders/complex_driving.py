from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def augment(image, steering_angle, range_x=100, range_y=10):
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    return image, steering_angle

transformations = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.CenterCrop((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class ComplexDrivingData(Dataset):
    def __init__(self, file_path, device, target_size=(320, 240), crop_size=(200, 200), color_mode='grayscale'):
        self.data_file = Path(file_path)
        self.device = device
        self.target_size = tuple(target_size)
        self.crop_size = tuple(crop_size)

        self.filenames = []
        self.imgs = None
        self.ground_truths = []

        if self.data_file.is_file():
            self._load_data()

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode, '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            self.image_shape = self.crop_size + (3,)
        else:
            self.image_shape = self.crop_size + (1,)

    def _load_data(self):
        if self.imgs is None:
            print("Loading data from {}...".format(self.data_file))
            with np.load(self.data_file) as data:
                self.filenames = data['file_names']
                self.ground_truths = data['ground_truths']
            print("Filenames: ", self.filenames.shape)
            print("Ground truths: ", self.ground_truths.shape)
            print("Done!")

    def __len__(self):
        return self.ground_truths.shape[0]

    def __getitem__(self, index):
        img_filename = os.path.join("data", self.filenames[index])
        img = load_img(img_filename, self.color_mode == "grayscale", self.target_size, self.crop_size)
        steering_angle = self.ground_truths[index]
        img = Image.fromarray(img)
        img = transformations(img)
        return img, steering_angle

def load_img(path, grayscale=False, target_size=None, crop_size=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image from path: {path}")

    if target_size:
        img = cv2.resize(img, target_size)

    if crop_size:
        img = central_image_crop(img, crop_size[0], crop_size[1])

    return np.asarray(img, dtype=np.float32)

def central_image_crop(img, crop_width=150, crop_height=150):
    half_the_width = int(img.shape[1] / 2)
    img = img[img.shape[0] - crop_height: img.shape[0], half_the_width - int(crop_width / 2): half_the_width + int(crop_width / 2)]
    return img

def get_iterator_complex_driving(file_path, device, batch_size=1, num_workers=4):
    dataset = ComplexDrivingData(file_path, device=device)
    iterator = DataLoader(dataset, shuffle='test' not in file_path and 'validation' not in file_path, batch_size=batch_size, num_workers=num_workers)
    return iterator
