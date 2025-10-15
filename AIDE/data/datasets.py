# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import io
import torch
from .dct import DCT_base_Rec_Module
import random
import copy
import json
import glob

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import kornia.augmentation as K

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

transform_before = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0])
    ]
)
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

CONFIG_DEFAULT = {
    "train_dataset": ["UADFV_FairFD_Indian", "UADFV_HAV-DF_train"],
    "test_dataset": ["UADFV_HAV-DF_test"],
    "frame_num": {'train': 8, 'test': 8},
    "dataset_json_folder": './data/dataset_json_train',
    "label_dict": {"UADFV_Fake": 1, "UADFV_Real": 0},

}


def collect_img_and_label_for_one_dataset(config: dict, dataset_name: str):
    """Collects image and label lists.

    Args:
        dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

    Returns:
        list: A list of image paths.
        list: A list of labels.

    Raises:
        ValueError: If image paths or labels are not found.
        NotImplementedError: If the dataset is not implemented yet.
    """
    # Initialize the label and frame path lists
    label_list = []
    frame_path_list = []

    # Record video name for video-level metrics
    video_name_list = []
    mode = config["mode"]
    frame_num = config['frame_num'][mode]
    try:
        with open(os.path.join(config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
            dataset_info = json.load(f)
    except Exception as e:
        print(e)
        raise ValueError(f'dataset {dataset_name} not exist!')

    # If JSON file exists, do the following data collection
    # FIXME: ugly, need to be modified here.
    cp = None
    if dataset_name == 'FaceForensics++_c40':
        dataset_name = 'FaceForensics++'
        cp = 'c40'
    elif dataset_name == 'FF-DF_c40':
        dataset_name = 'FF-DF'
        cp = 'c40'
    elif dataset_name == 'FF-F2F_c40':
        dataset_name = 'FF-F2F'
        cp = 'c40'
    elif dataset_name == 'FF-FS_c40':
        dataset_name = 'FF-FS'
        cp = 'c40'
    elif dataset_name == 'FF-NT_c40':
        dataset_name = 'FF-NT'
        cp = 'c40'
    elif dataset_name == 'UADFV_HAV-DF_train' or dataset_name == 'UADFV_HAV-DF_test':
        dataset_name = 'UADFV_HAV-DF'
    # Get the information for the current dataset
    for label in dataset_info[dataset_name]:
        sub_dataset_info = dataset_info[dataset_name][label][mode]

        # Iterate over the videos in the dataset
        for video_name, video_info in sub_dataset_info.items():
            # Unique video name
            unique_video_name = video_info['label'] + '_' + video_name

            # Get the label and frame paths for the current video
            if video_info['label'] not in config['label_dict']:
                raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
            label = config['label_dict'][video_info['label']]
            frame_paths = video_info['frames']
            if len(frame_paths)==0:
                print(f"{unique_video_name} is None. Let's skip it.")
                continue
            

            # Consider the case when the actual number of frames (e.g., 270) is larger than the specified (i.e., self.frame_num=32)
            # In this case, we select self.frame_num frames from the original 270 frames
            total_frames = len(frame_paths)
            if frame_num < total_frames:
                total_frames = frame_num

                if True:
                    # Select self.frame_num frames evenly distributed throughout the video
                    step = total_frames // frame_num
                    frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:frame_num]

            # Otherwise, extend the label and frame paths to the lists according to the number of frames
            if True:
                # Extend the label and frame paths to the lists according to the number of frames
                label_list.extend([label] * total_frames)
                frame_path_list.extend(frame_paths)
                # video name save
                video_name_list.extend([unique_video_name] * len(frame_paths))

    # Shuffle the label and frame path lists in the same order
    shuffled = list(zip(label_list, frame_path_list, video_name_list))
    random.shuffle(shuffled)
    label_list, frame_path_list, video_name_list = zip(*shuffled)

    return frame_path_list, label_list, video_name_list


class TrainDataset(Dataset):
    def __init__(self, is_train, args):

        self.config = copy.deepcopy(CONFIG_DEFAULT)
        mode = "train" if is_train else "test"
        ### assert mode == "train", f"mode is: {mode}, expecting train"
        #self.mode = mode
        self.config["mode"] = mode
        #self.frame_num = self.config['frame_num'][mode]
        if mode == "test":
            #self.config["dataset_json_folder"] = './data/dataset_json_test' # this is fine
            self.config['frame_num']['test'] = 1000000

        self.data_list = []

        # Dataset dictionary
        self.image_list = []
        self.label_list = []

                # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = self.config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = collect_img_and_label_for_one_dataset(self.config, one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
        elif mode == 'test':
            one_data = self.config['test_dataset'][0]
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list, name_list = collect_img_and_label_for_one_dataset(self.config, one_data)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list


        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list,
        }

        ds_size = len(self.image_list)
        for idx in range(ds_size):
            image_path, label = self.data_dict['image'][idx], self.data_dict['label'][idx]
            self.data_list.append({"image_path": image_path, "label" : label})

        """
        if 'GenImage' in root and root.split('/')[-1] != 'train':
            file_path = root

            if '0_real' not in os.listdir(file_path):
                for folder_name in os.listdir(file_path):
                
                    assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                    for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                 
                    for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
            
            else:
                for image_path in os.listdir(os.path.join(file_path, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        else:

            for filename in os.listdir(root):

                file_path = os.path.join(root, filename)

                if '0_real' not in os.listdir(file_path):
                    for folder_name in os.listdir(file_path):
                    
                        assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                        for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                            self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                    
                        for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                            self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
                
                else:
                    for image_path in os.listdir(os.path.join(file_path, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                    for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        """        
        self.dct = DCT_base_Rec_Module()

    


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))


        image = transform_before(image)

        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        except:
            print(f'image error: {image_path}, c, h, w: {image.shape}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        


        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

    

class TestDataset(Dataset):
    def __init__(self, is_train, args, eval_dataset_name):
        self.config = copy.deepcopy(CONFIG_DEFAULT)
        mode = "train" if is_train else "test"
        assert mode == "test", f"mode is: {mode}, expecting test"

        self.config["mode"] = mode
        self.config["dataset_json_folder"] = './data/dataset_json_test_genai_v2' # only for test, not for validation
        self.config['test_dataset'] = eval_dataset_name #["HIDF_dataset_Indian_images", "HIDF_dataset_Indian_videos"]
        print(f"dataset json used: {self.config["dataset_json_folder"]}")
        if True:
            self.config['frame_num']['test'] = 1000000
        

        self.data_list = []

         # Dataset dictionary
        self.image_list = []
        self.label_list = []

                # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = self.config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = collect_img_and_label_for_one_dataset(self.config, one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
        elif mode == 'test':
            one_data = self.config['test_dataset']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list, name_list = collect_img_and_label_for_one_dataset(self.config, one_data)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list


        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list,
        }

        ds_size = len(self.image_list)
        for idx in range(ds_size):
            image_path, label = self.data_dict['image'][idx], self.data_dict['label'][idx]
            self.data_list.append({"image_path": image_path, "label" : label})

        """
        if '0_real' not in os.listdir(file_path):
            for folder_name in os.listdir(file_path):
    
                assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
                
                for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                
                for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
        
        else:
            for image_path in os.listdir(os.path.join(file_path, '0_real')):
                self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
            for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})

        """
        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        image = Image.open(image_path).convert('RGB')
        if True and (int(targets) == 1): # fake image, for gemini watermark on bottom-right
            # crop_h, crop_w = image.shape[0], int(image.shape[1] * 0.9)
            # image = image[0: crop_h, 0: crop_w, :].copy()
            img_w, img_h = image.size
            crop_h, crop_w = img_h, int(img_w * 0.9)
            image = image.crop((0, 0, crop_w, crop_h)) #(left, upper, right, lower)
            #print(f"{image_path}, {img_w}:{img_h}, {image.size}")


        image = transform_before_test(image)

        # x_max, x_min, x_max_min, x_minmin = self.dct(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)


        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        
        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

