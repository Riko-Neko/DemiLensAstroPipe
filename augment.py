import csv
import glob
import hashlib
import math
import os
import sys
from functools import partial

import numpy as np
import torch
from PIL import Image
from astropy.io import fits
from scipy.ndimage import affine_transform
from scipy.ndimage import convolve
from scipy.special import j1
from torch.utils.data import DataLoader
from torchvision.transforms import v2


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_size=None, data_dir=None, csv_file=None, pos_dir=None, neg_dir=None, num=None,
                 augment_mode=None, color_jitter=False, add_noise=False, adaptation_mode='padding',
                 channel_expansion_mode=None, mix_channels=False, csv_samples_catalog_reader=None,
                 predicting_mode=False, pos_label=1.0, **norm_kwargs):
        self.img_size = img_size
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.num = num
        self.band = None
        self.format = None
        self.augment_mode = augment_mode
        self.color_jitter = color_jitter
        self.add_noise = add_noise
        self.adaptation_mode = adaptation_mode
        self.channel_expansion_mode = channel_expansion_mode
        self.mix_channels = mix_channels
        self.norm = norm_kwargs.get('norm', False)
        self.update_mean_std = norm_kwargs.get('update_mean_std', False)
        self.device = norm_kwargs.get('device', 'cpu')
        self.mean = norm_kwargs.get('mean', [0.485, 0.456, 0.406])
        self.std = norm_kwargs.get('std', [0.229, 0.224, 0.225])
        self.samples_catalog_reader = csv_samples_catalog_reader
        self.predicting_mode = predicting_mode
        self.pos_label = float(pos_label)

        # Load file paths for positive and negative samples using csv file or existing directory, or file for predicting
        if self.pos_dir is None and self.neg_dir is None and self.predicting_mode is not True and self.csv_file is None:
            raise ValueError(
                "Paths for samples are undefined. If you want to perform inference, please check if parameter 'predicting_mode' is set to True. Or if you want to use csv index file, please provide it in 'csv_file' parameter.")
        if predicting_mode is True:
            if self.csv_file is not None or self.pos_dir is not None or self.neg_dir is not None:
                print(
                    "[\033[93mWarning\033[0m] predicting_mode is set to True. pos_dir, neg_dir, and csv_file will be ignored.")
            self.augment_mode, self.mix_channels, self.update_mean_std = None, False, False
            self.inference = self.get_path(self.data_dir)
        elif self.csv_file is not None:
            if self.pos_dir is not None or self.neg_dir is not None:
                print("[\033[93mWarning\033[0m] csv_file is provided. pos_dir and neg_dir will be ignored.")
            self.csv_dir_proofreading = DatasetToolkit.csv_dir_proofreading
            self.csv_data, samples_catalog = self.load_csv(self.csv_file, self.samples_catalog_reader)
            self.csv_data_dir, matching_ratio = self.get_path(self.data_dir, samples_catalog=samples_catalog)
            if matching_ratio < 1:
                if matching_ratio == 0:
                    raise FileNotFoundError(
                        f"No files found in {self.data_dir} for samples in {self.csv_file}.")
                if input(
                        f"[\033[93mWarning\033[0m] current data directory:\n{self.csv_data_dir}\nwhich have \033[93minsufficient\033[0m samples.\nStill proceed?(y/n)").lower() == 'n':
                    sys.exit("\033[91mAborted.\033[0m")
            self.csv_data_names, self.csv_data_labels = self.csv_data['name'], self.csv_data['label']
            self.csv_data_samples = self.csv_data[self.samples_catalog_reader] if samples_catalog is not [] else []
        elif self.pos_dir is None:
            print("[\033[93mWarning\033[0m] pos_dir is None. Without positive samples might lead to poor performance.")
        elif self.neg_dir is None:
            print("[\033[93mWarning\033[0m] neg_dir is None. Without negative samples might lead to poor performance.")
        else:
            self.positives = self.get_path(self.pos_dir)
            self.negatives = self.get_path(self.neg_dir)

        # Define channel expansion
        if channel_expansion_mode is not None:
            if channel_expansion_mode == 'low_pass_filter' or 'low_pass' or 'lpf':
                self.channel_expansion = lambda img: AugmentationLib.fft_low_pass_filter(img, cutoff_frequency=0.5)
            elif channel_expansion_mode == 'high_pass_filter' or 'high_pass' or 'hpf':
                pass
            elif channel_expansion_mode == 'gradients' or 'gradient' or 'grad':
                self.channel_expansion = lambda img: AugmentationLib.get_image_gradients(img)
            else:
                raise ValueError(
                    'Invalid channel expansion mode. (available modes: "low_pass_filter", "high_pass_filter", "gradients")')

        # Define image transformation and augmentation
        transform_list = [v2.ToImage()]
        if adaptation_mode == 'resizing':
            transform_list.extend([v2.RandomResize(min_size=self.img_size, max_size=int(self.img_size * 1.05),
                                                   interpolation=Image.BICUBIC) if self.img_size is not None else None,
                                   v2.RandomCrop(
                                       (self.img_size, self.img_size)) if self.img_size is not None else None])
        elif adaptation_mode == 'padding':
            transform_list.extend(
                [v2.CenterCrop(int(self.img_size * 1.1)) if self.augment_mode else v2.CenterCrop(int(self.img_size)), ])
        elif adaptation_mode == 'original':
            print(
                "[\033[93mWarning\033[0m] Using original image. Make sure the image size is matched, or the model is able to handle the image size. Otherwise, error would stop the program.")
        else:
            raise ValueError('Invalid adaptation mode. (available modes: "resizing", "padding", "original")')

        augmentation_list_full = [v2.RandomCrop((self.img_size, self.img_size), pad_if_needed=True,
                                                padding_mode='reflect') if self.img_size is not None else None,
                                  v2.RandomHorizontalFlip(),
                                  v2.RandomVerticalFlip(),
                                  v2.RandomChoice([v2.Identity(),
                                                   v2.Lambda(partial(torch.rot90, k=2, dims=(-2, -1))), ]),
                                  v2.RandomChoice([v2.Identity(),
                                                   v2.Lambda(partial(torch.rot90, k=1, dims=(-2, -1))),
                                                   v2.Lambda(partial(torch.rot90, k=2, dims=(-2, -1))),
                                                   v2.Lambda(partial(torch.rot90, k=3, dims=(-2, -1))), ]),
                                  # v2.RandomRotation(180), # Introduce pixel change!
                                  # v2.RandomRotation(90),
                                  # v2.RandomGrayscale(0.05),
                                  ]

        augmentation_list_astro = [AugmentationLib.RandomEllipticalDistortion(max_scale=0.05),
                                   AugmentationLib.PSFConvolution(kernel_size=5),
                                   AugmentationLib.AsinhNormalization(scale=0.1),
                                   v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1),
                                                   shear=(-2.5, 2.5)),
                                   v2.RandomHorizontalFlip(),
                                   v2.RandomVerticalFlip(),
                                   ]

        color_jitter_list = [v2.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.)
                             ]

        # Define image normalization
        normalization_list = []
        if self.norm:
            if self.update_mean_std:
                self.mean, self.std = DatasetToolkit.mean_std_update(self.device, self.get_raw_dataset())
            normalization_list = [v2.Normalize(mean=self.mean, std=self.std)
                                  ]

        if self.augment_mode is not None:
            if self.augment_mode == 'full':
                transform_list.extend(augmentation_list_full)
            elif self.augment_mode == 'astro' or 'astro-augmentation' or 'astro-data':
                transform_list.extend(augmentation_list_astro)
            else:
                raise ValueError(f"Invalid augmentation mode: {self.augment_mode}")
        if self.color_jitter:
            transform_list.extend(color_jitter_list)
        if self.add_noise:
            transform_list.append(lambda img: self.add_gaussian_noise(img))  # For linux only!
        transform_list.append(v2.ToDtype(torch.float32, scale=True))
        if self.norm:
            transform_list.extend(normalization_list)
        self.transform = v2.Compose(transform_list)

    def __len__(self):
        if self.predicting_mode is not True:
            if self.csv_file is not None:
                return len(self.csv_data_names)
            else:
                return len(self.positives) + len(self.negatives)
        else:
            return len(self.inference)

    def __getitem__(self, index):
        if self.predicting_mode is not True:
            if self.csv_file is not None:
                data_dir = self.csv_data_dir[self.csv_data_samples[index]] if self.csv_data_dir != [] else self.data_dir
                path = os.path.join(data_dir, self.csv_data_names[index])
                if not os.path.isfile(path):
                    return self.__getitem__((index + 1) % len(self))
                label = self.csv_data_labels[index]
            elif index < len(self.positives):
                path = self.positives[index]
                label = self.pos_label
            else:
                path = self.negatives[index - len(self.positives)]
                label = 0
        else:
            path = self.inference[index]
            label = None

        if os.path.splitext(path)[1].lower() == '.fits':
            try:
                img, self.band = self.load_fits(path)
            except Exception as e:
                return self.__getitem__((index + 1) % len(self))
            self.format = 'fits'

        elif os.path.splitext(path)[1].lower() == '.png':
            try:
                img, self.band = self.load_png(path)
            except Exception as e:
                return self.__getitem__((index + 1) % len(self))
            self.format = 'png'

        else:
            raise ValueError(f"Unsupported file format: {os.path.splitext(path)[1]}")

        if not isinstance(self.band, int) or self.band < 1:
            raise ValueError("[\033[91mError\033[0m] Failed to read band info, please check the data.")

        img = self.transform(img)

        if self.channel_expansion_mode is not None:
            expanded_img = self.channel_expansion(img)
            img = torch.cat((img, expanded_img), dim=0)

        if self.mix_channels:
            pass
        if self.predicting_mode is not True:
            return img, label
        else:
            return img, path

    def get_path(self, data_dir, samples_catalog=None):
        path, flag = [], []
        if samples_catalog is not None:
            if len(data_dir) == 1 or isinstance(data_dir, str):
                self.data_dir = data_dir[0] if isinstance(data_dir, list) else data_dir
                return [[item, data_dir] for item in samples_catalog]
            if len(data_dir) != len(samples_catalog):
                print("\033[91mWarning: invalid csv samples reader.\033[0m")
            return self.csv_dir_proofreading(self.csv_data, self.data_dir, samples_catalog, self.samples_catalog_reader)
        else:
            for dir in data_dir or []:
                if not os.path.exists(dir):
                    raise FileNotFoundError(f"The directory {dir} does not exist.")
                path.extend(glob.glob(dir + '/*'))
            return path[:self.num]

    def load_csv(self, csv_file, samples_catalog_reader=None):
        """Load the CSV file and return its contents formatted as a dictionary based on the CSV headers."""
        data = {}
        samples = set()
        with open(csv_file, newline='') as file:
            reader = csv.DictReader(file)
            for header in reader.fieldnames:
                data[header] = []
            for i, row in enumerate(reader):
                for header in reader.fieldnames:
                    data[header].append(row[header])
                if samples_catalog_reader in row:
                    samples.add(row['sample'])
                if self.num is not None and i + 1 >= self.num:
                    break
        if 'label' in data:
            data['label'] = [int(label) for label in data['label']]
        return data, list(samples)

    def load_fits(self, path):
        hdul = fits.open(path)
        bands = []
        reference_shape = None

        for i in range(len(hdul)):
            data = hdul[i].data
            if data is None:
                continue

            # 处理 BinTableHDU
            if isinstance(data, np.ndarray) and data.dtype.names is not None:
                # 结构化数组，取第一个非字符串字段（例如 1681D）
                for name in data.dtype.names:
                    col = data[name][0]
                    if isinstance(col, (np.ndarray, list)):
                        size = int(len(col))
                        data = np.array(col, dtype=np.float32).reshape(size, size)
                        break

            if reference_shape is None:
                reference_shape = data.shape
            if data.shape != reference_shape:
                continue  # skip different shape

            band_data = self.pixel_filter(data)
            if band_data.dtype.byteorder != '=':
                band_data = band_data.byteswap().view(band_data.dtype.newbyteorder('='))

            bands.append(torch.from_numpy(band_data).float())

        if not bands:
            return None

        if len(bands) == 1:
            bands[0] = torch.sqrt(bands[0])

        combined_tensor = torch.cat(bands)
        max_value = 1e-13 if (max_value := torch.max(combined_tensor)).item() == 0 else max_value
        for idx in range(len(bands)):
            bands[idx] /= max_value
            # bands[idx] *= 255.0

        image = torch.stack(bands, dim=0)
        return image, len(bands)

    def load_png(self, path):
        image = Image.open(path)
        band = len(image.getbands()) if image.mode in ('RGB', 'RGBA', 'L', 'CMYK') else -1
        return image, band

    @staticmethod
    def pixel_filter(image):
        image[image < 0] = 0.0
        return image

    def add_gaussian_noise(self, img, mean=0., std=0.1, std_mod='unif'):
        img = img.to(torch.float32)
        if std_mod == 'unif':
            std = 10 ** (torch.FloatTensor(1).uniform_(math.log10(std), math.log10(std * 10)).item())
        elif std_mod == 'log':
            pass
        elif std_mod == 'exp':
            pass
        elif std_mod == 'constant' or std_mod is None:
            pass
        else:
            raise ValueError(f"Unsupported std_mod: {std_mod}")

        std = std / 255.0 / 10
        noisy = torch.randn_like(img) * std + mean

        if self.format == 'fits':
            noisy = self.pixel_filter(noisy)

        return noisy

    def get_raw_dataset(self, hide_process=True, test=False, pred=False):
        from config import Interface
        stdout_controller = Interface.StdoutController()
        num = self.num if not test or pred else None

        if hide_process:
            stdout_controller.stdout_block()
        try:
            dataset = ImageDataset(self.img_size, self.data_dir, self.csv_file, self.pos_dir, self.neg_dir, num, None,
                                   False, False, self.adaptation_mode, self.channel_expansion_mode, self.mix_channels,
                                   self.samples_catalog_reader, pred, 1.0, norm=self.norm, update_mean_std=False,
                                   mean=self.mean, std=self.std)
        finally:
            if hide_process:
                stdout_controller.stdout_re()

        return dataset


class AugmentationLib:
    @staticmethod
    def fft_low_pass_filter(img, cutoff_frequency):
        """
        Apply a low-pass filter to each channel of the image using Fast Fourier Transform (FFT).

        This function processes the image by performing the following steps:
        1. It computes the 2D FFT for each channel of the input image.
        2. It constructs a circular low-pass filter based on the specified cutoff frequency.
        3. It applies the low-pass filter by zeroing out frequencies that exceed the cutoff.
        4. It performs an inverse FFT to transform the filtered frequency domain representation back into the spatial domain.

        :param img: Input image tensor with shape (channels, height, width).
        :param cutoff_frequency: The cutoff frequency for the low-pass filter. Frequencies above this value will be attenuated.
        :return: The filtered image tensor, maintaining the same shape as the input image.
        """
        channels, height, width = img.shape
        filtered_img = torch.zeros_like(img)

        for c in range(channels):
            # Compute the FFT for each channel
            f_image = torch.fft.fft2(img[c])

            # Create a low-pass filter
            f_shape = f_image.shape
            center_y, center_x = f_shape[0] // 2, f_shape[1] // 2
            Y, X = torch.meshgrid(torch.arange(f_shape[0]), torch.arange(f_shape[1]),
                                  indexing='ij')  # Create grid coordinates
            radius = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)  # Compute distances from the center

            # Apply the low-pass filter
            low_pass_mask = radius <= cutoff_frequency
            f_image[~low_pass_mask] = 0  # Zero out high-frequency components

            # Perform inverse FFT to obtain the filtered image in the spatial domain
            filtered_img[c] = torch.fft.ifft2(f_image).real

        return filtered_img

    @staticmethod
    def get_image_gradients(img):
        """
        Calculate the image gradients for each channel using the Sobel operator.

        This function processes the input image by applying the Sobel filter to compute gradients
        in both the x and y directions. The output is the magnitude of the gradient, which highlights
        the edges in the image.

        :param img: Input image tensor with shape (channels, height, width).
        :return: Gradient image tensor, maintaining the same shape as the input image.
        """
        # Define Sobel operator for x direction
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 3, 3)

        # Define Sobel operator for y direction
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 3, 3)

        channels, height, width = img.shape
        gradient_img = torch.zeros_like(img)

        for c in range(channels):
            # Compute the gradient in the x direction
            grad_x = torch.nn.functional.conv2d(img[c].unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
            # Compute the gradient in the y direction
            grad_y = torch.nn.functional.conv2d(img[c].unsqueeze(0).unsqueeze(0), sobel_y, padding=1)

            # Calculate the magnitude of the gradient
            gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            gradient_img[c] = gradient_magnitude.squeeze(0).squeeze(0)  # Return the gradients for this channel

        return gradient_img

    class RandomEllipticalDistortion:
        """
        simulate gravitational lensing with elliptical distortions.

        This class simulates gravitational lensing with elliptical distortions by applying a random affine transformation
        to the input image.
        """

        def __init__(self, max_scale=0.2):
            self.max_scale = max_scale

        def __call__(self, img):
            scale_x = 1 + np.random.uniform(-self.max_scale, self.max_scale)
            scale_y = 1 + np.random.uniform(-self.max_scale, self.max_scale)
            angle = np.random.uniform(0, 360)

            theta = np.deg2rad(angle)
            matrix = np.array([
                [scale_x * np.cos(theta), -scale_y * np.sin(theta)],
                [scale_x * np.sin(theta), scale_y * np.cos(theta)]
            ])

            if isinstance(img, torch.Tensor):
                img = img.numpy().transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)

            distorted = np.stack([
                affine_transform(
                    img[..., c],
                    matrix,
                    order=1,
                    mode='reflect'
                ) for c in range(img.shape[-1])
            ], axis=-1)

            return torch.from_numpy(distorted.transpose(2, 0, 1))  # (H,W,C) -> (C,H,W)

    class PSFConvolution:
        """
        Simulate astronomical Point Spread Function Convolution.

        This class simulates astronomical Point Spread Function Convolution by applying a convolution to the input image.
        """

        def __init__(self, kernel_size=5):
            self.kernel_size = kernel_size

        def _gaussian_kernel(self, sigma):
            """
            Generate 2D Gaussian Kernel
            """
            x = np.linspace(-3 * sigma, 3 * sigma, self.kernel_size)
            xx, yy = np.meshgrid(x, x)
            kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
            return kernel / kernel.sum()

        def _airy_kernel(self):
            """
            Generate 2D Airy Disk Kernel
            """
            x = np.linspace(-3, 3, self.kernel_size)
            xx, yy = np.meshgrid(x, x)
            r = np.sqrt(xx ** 2 + yy ** 2)

            with np.errstate(divide='ignore', invalid='ignore'):
                kernel = (2 * j1(r) / r) ** 2
            kernel[np.isnan(kernel)] = 1.0
            kernel /= kernel.sum()
            return kernel

        def __call__(self, img):
            choice = np.random.choice(['gaussian', 'airy', 'tophat'])

            if choice == 'gaussian':
                sigma = np.random.uniform(0.5, 2.0)
                kernel = self._gaussian_kernel(sigma)
            elif choice == 'airy':
                kernel = self._airy_kernel()
            else:  # tophat
                kernel = np.ones((self.kernel_size, self.kernel_size))
                kernel /= kernel.sum()

            if isinstance(img, torch.Tensor):
                img = img.numpy().transpose(1, 2, 0)  # (C,H,W)->(H,W,C)

            convolved = np.stack([
                convolve(
                    img[..., c],
                    kernel,
                    mode='reflect'
                ) for c in range(img.shape[-1])
            ], axis=-1)

            return torch.from_numpy(convolved.transpose(2, 0, 1))  # (H,W,C)->(C,H,W)

    class AsinhNormalization:
        """
        Apply asinh normalization to the input image.

        This class applies asinh normalization to the input image by mapping the pixel values to the range [0, 1]
        """

        def __init__(self, scale=0.1):
            self.scale = scale

        def __call__(self, img):
            img = img.float()
            normalized = torch.arcsinh(img / self.scale) / torch.arcsinh(torch.tensor(1.0 / self.scale))
            return torch.clamp(normalized, 0, 1)


class DatasetToolkit:
    @staticmethod
    def mean_std_update(device, dataset=None, data_dir=None, pos_dir=None, neg_dir=None, num=None):
        r"""Function to calculate the mean and standard deviation of images in a dataset.

            It updates the computed mean and standard deviation for RGB channels based on the dataset provided
            or the paths specified if the dataset is not provided.

            Args:
                device (torch.device): The device to use for calculations.
                dataset (ImageDataset, optional): The ImageDataset object to compute mean and std from.
                data_dir (str, optional): The base directory containing data.
                pos_dir (list, optional): List of directories containing positive samples.
                neg_dir (list, optional): List of directories containing negative samples.
                num (int, optional): The number of samples to compute the mean and std for. If None, all samples will be used.

            Returns:
                mean (list): A list containing the mean values for each channel.
                std (list): A list containing the standard deviation values for each channel.

            Raises:
                ValueError: If dataset is not provided and required paths are missing.
        """

        print('[\033[92mInfo\033[0m] Preparing to update mean and std', end=' ==> ')
        if dataset is None and (data_dir is not None and pos_dir is not None and neg_dir is not None):
            print('Loading dataset from the path')
            dataset = ImageDataset(data_dir=data_dir, pos_dir=pos_dir, neg_dir=neg_dir, num=num,
                                   augment_mode=None, mix_channels=False, norm=False)
        elif dataset is not None:
            print('Using current dataset')
        else:
            raise ValueError(
                "Either dataset must be provided or all of data_dir, pos_dir, and neg_dir must be specified.")

        print('[\033[92mInfo\033[0m] Updating mean and std... It might take several minutes.\n')
        if num is None:
            num = len(dataset)

        channel_sum = torch.zeros(3, device=device)
        channel_squared_sum = torch.zeros(3, device=device)
        total_pixels = 0

        for i, (image, _) in enumerate(dataset):
            if i >= num:
                break

            if image.dtype == torch.uint8:
                image = image.float() / 255.0

            image = image.double().to(device)  # [C, H, W]
            c, h, w = image.shape
            total_pixels += h * w

            channel_sum += image.sum(dim=[1, 2])
            channel_squared_sum += (image ** 2).sum(dim=[1, 2])

        mean = (channel_sum / total_pixels).tolist()
        std = ((channel_squared_sum / total_pixels - torch.tensor(mean, device=device) ** 2) ** 0.5).tolist()

        print(f"[\033[92mInfo\033[0m] Mean: {mean}, Std: {std}")

        return mean, std

    @staticmethod
    def check_loader(dataloader, max_batches=10, start_batch=0, verbose=False, calculate_proportion=False):
        r"""Function to iterate through a DataLoader and print information about the batches.
        It supports both of training and testing datasets.

        Args:
            dataloader (DataLoader): The DataLoader object to iterate through.
            max_batches (int): The maximum number of batches to print information for.
            start_batch (int): The batch index to start printing from.
            verbose (bool): If True, print detailed information for each image and label in the batch.
            calculate_proportion (bool): If True, print the proportion of label 1 and the ratio of label 1 to label 0.
        """
        batch_counter = 0
        count_label_1 = 0
        total_samples = 0
        info_to_print = []
        warned_uint8 = False  # 只抛出一次警告

        print('Dataloader Attributes:', dataloader,
              'Len:', len(dataloader) * dataloader.batch_size,
              'Batches:', len(dataloader))

        for batches, (images, labels) in enumerate(dataloader):
            if batches < start_batch:
                continue

            if images.dtype == torch.uint8 and not warned_uint8:
                print(
                    "\033[93m[Warning]\033[0m Images are in uint8 format. Consider normalizing or converting to float.")
                warned_uint8 = True

            batch_info = f"Batch {batches + 1 - start_batch}\n"
            batch_info += f"Images shape: {images.size()}, dtype: {images.dtype}\n"
            batch_info += f"Labels shape: {labels.size()}, dtype: {labels.dtype}\n"

            for label in labels:
                if label.item() > 0.5:
                    count_label_1 += 1
                total_samples += 1

            if verbose:
                batch_info += "Image shapes and Labels:\n"
                for image, label in zip(images, labels):
                    batch_info += f"Image shape: {image.size()}, Label: {label.item()}\n"

            info_to_print.append(batch_info)
            batch_counter += 1
            if batch_counter >= max_batches:
                break

        for info in info_to_print:
            print(info, end='')

        if calculate_proportion and total_samples > 0:
            proportion_label_1 = count_label_1 / total_samples
            ratio_label_1_to_0 = count_label_1 / (
                    total_samples - count_label_1) if total_samples - count_label_1 > 0 else 0
            print(f"\033[94mProportion of label 1: {proportion_label_1:.2f}\033[0m")
            print(f"\033[94mRatio(label_1:label_0): {ratio_label_1_to_0:.2f}:1\033[0m\n")
        elif calculate_proportion and total_samples == 0:
            print("\033[94mNo samples to calculate proportion.\033[0m\n")

    @staticmethod
    def csv_dir_proofreading(csv_data, sample_dir, sample_catalog, sample_catalog_reader, freq_threshold=10,
                             auto_increase_freq=True, _recursion_count=0):
        """
            Validates the existence of sample files listed in `csv_data` against the specified directories in `sample_dir`.

            This function checks whether the files corresponding to each sample listed in `csv_data` exist in their respective
            directories. It also calculates and reports the matching ratio of found files to the expected files per sample type.

            Parameters:
            - csv_data (dict): A dictionary representing the CSV data. It must contain the following keys:
                - 'name': A list of file names (e.g., ['1.fits', '2.fits', ...]) corresponding to samples.
                - sample_catalog_reader: A list of sample types (e.g., ['sim_lens', 'real_galaxies', ...]) indicating the category of each sample.
                - Any additional keys, including 'label', which may adjust the data structure.

            - sample_dir (list): A list containing the paths to the directories where sample files are expected to be found.
              The length of this list must match the length of `sample_catalog`.

            - sample_catalog (list): A list of sample categories corresponding to each directory in `sample_dir`.
              The length of this list must match the length of `sample_dir`.

            - sample_catalog_reader (str): The column name in `csv_data` that contains the sample categories.

            - freq_threshold (int, optional): The initial frequency threshold for the minimum number of file matches expected for each sample type. Defaults to 10.

            - auto_increase_freq (bool, optional): Indicates whether the function should automatically increase `freq_threshold`
              to allow for a more lenient search if the initial search fails. Defaults to True.

            - recursion_count (int, optional): The current recursion count. This parameter s used internally to control the recursion depth. Never set this parameter manually.

            Returns:
            - dict: A dictionary mapping sample types to their respective directory paths.
            - float: The ratio of matching files to expected files, a value between 0 and 1. If no matches are found or if
              the files cannot be verified, the function returns 0.0.

            Notes:
            - It is strongly required that the lengths of `sample_dir` and `sample_catalog` are equal.
            - `csv_data` must include the 'name' and the column specified by `sample_catalog_reader`.
            - The structure of `csv_data` should follow the specified format:
              {'name': ['1.fits', '2.fits', ...],
               sample_catalog_reader: ['sim_lens', 'real_galaxies', ...],
               ...,
               'label': [1, 0, ...]}.

            Example usage:
            >>> csv_data = {
                    'name': ['1.fits', '2.fits', ...],
                    'sim_lens': ['sample1', 'sample2', ...],
                    'real_galaxies': ['sample3', 'sample4', ...],
                    'label': [1, 0, ...]
                }
            >>> sample_dir = ['/path/to/sim_lens', '/path/to/real_galaxies']
            >>> sample_catalog = ['sim_lens', 'real_galaxies']
            >>> sample_catalog_reader = 'sim_lens'
            >>> DatasetToolkit.csv_dir_proofreading(csv_data, sample_dir, sample_catalog, sample_catalog_reader)
            """
        from itertools import permutations

        csv_data_names = csv_data['name']
        csv_data_samples = csv_data[sample_catalog_reader]
        csv_data_dir = dict(zip(sample_catalog, sample_dir))
        names_dict = {}
        freq_actual = freq_threshold

        perm = permutations(sample_catalog)
        all_catalogs = [list(p) for p in perm]

        max_recursion = int(math.log2(len(csv_data_names) / (freq_threshold / (2 ** _recursion_count))))
        print(f"\rRecursion depth: {_recursion_count}/{max_recursion}", end=" ", flush=True)
        if _recursion_count >= max_recursion:
            print(
                f"\033[91m\nError: Maximum recursion depth ({max_recursion}) exceeded. Data cannot be verified.\033[0m")
            return csv_data_dir, 0.0

        for catalogs in all_catalogs:
            mismatch = False
            count_all = 0
            csv_data_dir = dict(zip(catalogs, sample_dir))
            mismatch_info = set()

            for sample in catalogs:
                try:
                    freq_actual = min(freq_threshold, len(csv_data_names))
                    names_dict[sample] = []
                    for index, name in enumerate(csv_data_names):
                        if csv_data_samples[index] == sample:
                            names_dict[sample].append(name)
                            if len(names_dict[sample]) >= freq_actual:
                                break
                    if len(names_dict[sample]) < freq_actual:
                        print(
                            f"[\033[93mWarning: Insufficient matches for sample '{sample}'. Only {len(names_dict[sample])} found.\033[0m")
                except KeyError:
                    print(f"\033[91mThe sample '{sample}' does not exist in the dictionary.\033[0m")
                except ValueError as ve:
                    print(f"\033[91mValue error occurred with sample '{sample}': {ve}\033[0m")
                except Exception as e:
                    print(f"\033[91mAn unexpected error occurred while processing sample '{sample}': {e}\033[0m")

            for key in names_dict:
                count = 0

                for name in names_dict[key]:
                    if os.path.isfile(os.path.join(csv_data_dir[key], name)):
                        count += 1
                if count == 0:
                    mismatch = True
                    break
                if count < freq_actual:
                    mismatch_info.add(key)

                count_all += count

            if mismatch:
                continue
            else:
                print()
                if len(names_dict) == 0:
                    print(
                        "\033[91mError: No samples were found. Please check the CSV data or sample directory paths.\033[0m")
                    return csv_data_dir, 0.0

                matching_ratio = count_all / (freq_actual * len(names_dict))
                if matching_ratio == 1:
                    print(f"\033[92mData is healthy.(100% matched)\033[0m")
                else:
                    for i in mismatch_info:
                        print(f"\033[93mWarning: data in category '{i}' is mismatched.\033[0m")
                    print(f"\033[93mWarning: data is not healthy.({matching_ratio:.2f}% matched)\033[0m")
                return csv_data_dir, matching_ratio
        if auto_increase_freq:
            freq_threshold *= 2
            return DatasetToolkit.csv_dir_proofreading(csv_data, sample_dir, sample_catalog, sample_catalog_reader,
                                                       freq_threshold,
                                                       auto_increase_freq, _recursion_count + 1)
        print(
            f"\033[91mError: data cannot be verified, it might due to data loss or incorrect data_dir. Or try to set auto_increase_freq to True?\033[0m")
        return csv_data_dir, 0.0

    @staticmethod
    def calculate_pytorch_dataset_hash(dataset, batch_size=64):
        """
        Calculates the hash value of a PyTorch dataset.

        :param dataset: A PyTorch Dataset object.
        :param batch_size: The batch size for processing the dataset (default: 64).
        :return: A hash string representing the dataset's current state.
        """
        hasher = hashlib.sha256()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in dataloader:
            # Assume batch is a tuple (data, labels)
            data, labels = batch

            # Convert data and labels to bytes
            data_bytes = data.numpy().tobytes()
            label_bytes = labels.numpy().tobytes()

            # Update the hash value with the data and labels
            hasher.update(data_bytes)
            hasher.update(label_bytes)

        return hasher.hexdigest()

    @staticmethod
    def has_dataset_changed(dataset, previous_hash, batch_size=64):
        """
        Checks whether the dataset has changed by comparing its current hash to a previous hash.

        :param dataset: A PyTorch Dataset object.
        :param previous_hash: The hash string calculated previously for the dataset.
        :param batch_size: The batch size for processing the dataset (default: 64).
        :return: A boolean indicating whether the dataset has changed (True if changed, False otherwise).
        """
        current_hash = DatasetToolkit.calculate_pytorch_dataset_hash(dataset, batch_size)
        return current_hash != previous_hash
