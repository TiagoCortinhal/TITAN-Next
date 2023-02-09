import os

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from common.laserscan import LaserScan, SemLaserScan

try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import os.path


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def my_collate(batch):
    return batch


class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, batch):
        self.data_source = data_source
        self.batch = batch

    def __iter__(self):
        self.size = len(self.data_source) // self.batch * self.batch
        self.sliding = rolling_window(np.array(list(range(self.size))), 4)
        self.rolling = rolling_window(np.array(list(range(self.size))), 4)

        self.idx = self.rolling[
            np.random.choice(len(self.rolling), size=len(self.data_source) // self.batch, replace=False)].reshape(-1)

        return iter(self.idx)

    def __len__(self):
        return len(self.data_source)


EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_IMAGE = ['.png']
EXTENSIONS_FLOW = ['.npy']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)


def is_flow(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_FLOW)


class SemanticKitti(Dataset):

    def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 labels,  # label dict: (e.g 10: "car")
                 color_map,  # colors dict bgr (e.g 10: [255, 0, 0])
                 learning_map,  # classes to learn (0 to N-1 for xentropy)
                 learning_map_inv,  # inverse of previous (recover labels)
                 sensor,  # sensor to parse scans from
                 max_points=150000,  # max number of points present in dataset
                 gt=True,
                 transform=False):  # send ground truth?
        # save deats
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"],
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                            dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt
        self.transform = transform

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.nclasses = len(self.learning_map_inv)

        # sanity checks

        # make sure directory exists
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        # make sure labels is a dict
        assert (isinstance(self.labels, dict))

        # make sure color_map is a dict
        assert (isinstance(self.color_map, dict))

        # make sure learning_map is a dict
        assert (isinstance(self.learning_map, dict))

        # make sure sequences is a list
        assert (isinstance(self.sequences, list))

        # placeholder for filenames
        self.scan_files = []
        self.label_files = []
        self.img2_files = []
        self.rgbseg_files = []
        self.lidarseg_files = []
        self.flow_files = []

        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
            # to string
            seq = '{0:02d}'.format(int(seq))

            print("parsing seq {}".format(seq))

            # get paths for each
            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")
            img2_path = os.path.join(self.root, seq, "image_2")
            lidarseg_path = os.path.join(self.root, seq, "lidar_segmented")
            rgblabels_path = os.path.join(self.root, seq, "rgblabel_2")
            flow_path = os.path.join(self.root, seq, "depth")

            # get files
            flow_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(flow_path)) for f in fn if is_flow(f)]
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_path)) for f in fn if is_label(f)]
            img2_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(img2_path)) for f in fn if is_image(f)]
            rgbseg_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(rgblabels_path)) for f in fn if is_label(f)]
            lidarseg_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(lidarseg_path)) for f in fn if is_image(f)]

            # check all scans have labels
            if self.gt:
                #assert (len(scan_files) == len(label_files))
                assert (len(scan_files) == len(img2_files))
                assert (len(scan_files) == len(rgbseg_files))

            # extend list
            self.flow_files.extend(flow_path)
            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)
            self.img2_files.extend(img2_files)
            self.rgbseg_files.extend(rgbseg_files)
            self.lidarseg_files.extend(lidarseg_files)

        # sort for correspondance
        self.scan_files.sort()
        self.label_files.sort()
        self.img2_files.sort()
        self.rgbseg_files.sort()
        self.lidarseg_files.sort()
        self.flow_files.sort()
        print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                        self.sequences))

    def get_item(self, index):
        # get item in tensor shape
        scan_file_1 = self.scan_files[index]
        # scan_file_2 = self.scan_files[index+1]

        img2_file_1 = self.img2_files[index]
        # img2_file_2 = self.img2_files[index+1]

        rgbseg_file_1 = self.rgbseg_files[index]
        # rgbseg_file_2 = self.rgbseg_files[index+1]
        flow_file = np.load(self.flow_files[index])

        if self.gt:
            label_file_1 = self.label_files[index]
            # label_file_2 = self.label_files[index+1]
        if self.gt:
            scan_1 = SemLaserScan(self.color_map,
                                  project=True,
                                  H=self.sensor_img_H,
                                  W=self.sensor_img_W,
                                  fov_up=self.sensor_fov_up,
                                  fov_down=self.sensor_fov_down)
        else:
            scan_1 = LaserScan(project=True,
                               H=self.sensor_img_H,
                               W=self.sensor_img_W,
                               fov_up=self.sensor_fov_up,
                               fov_down=self.sensor_fov_down)

        # open and obtain scan

        scan_1.open_scan(scan_file_1)
        # scan_2.open_scan(scan_file_2)
        if self.gt:
            scan_1.open_label(label_file_1)
            # scan_2.open_label(label_file_2)
            # map unused classes to used classes (also for projection)
            scan_1.sem_label = self.map(scan_1.sem_label, self.learning_map)
            # scan_2.sem_label = self.map(scan_2.sem_label, self.learning_map)

            scan_1.proj_sem_label = self.map(scan_1.proj_sem_label, self.learning_map)
            # scan_2.proj_sem_label = self.map(scan_2.proj_sem_label, self.learning_map)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points_1 = scan_1.points.shape[0]
        unproj_xyz_1 = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz_1[:unproj_n_points_1] = torch.from_numpy(scan_1.points)
        unproj_range_1 = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range_1[:unproj_n_points_1] = torch.from_numpy(scan_1.unproj_range)
        unproj_remissions_1 = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions_1[:unproj_n_points_1] = torch.from_numpy(scan_1.remissions)
        if self.gt:
            unproj_labels_1 = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels_1[:unproj_n_points_1] = torch.from_numpy(scan_1.sem_label)
        else:
            unproj_labels_1 = []

        # get points and labels
        proj_range_1 = torch.from_numpy(scan_1.proj_range).clone()
        proj_xyz_1 = torch.from_numpy(scan_1.proj_xyz).clone()
        proj_remission_1 = torch.from_numpy(scan_1.proj_remission).clone()
        proj_mask_1 = torch.from_numpy(scan_1.proj_mask)
        if self.gt:
            proj_labels_1 = torch.from_numpy(scan_1.proj_sem_label).clone()
            proj_labels_1 = (proj_labels_1 * proj_mask_1)  # [..., 768:1280]
        else:
            proj_labels_1 = []
        proj_x_1 = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x_1[:unproj_n_points_1] = torch.from_numpy(scan_1.proj_x)
        proj_y_1 = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y_1[:unproj_n_points_1] = torch.from_numpy(scan_1.proj_y)
        proj_1 = torch.cat([proj_range_1.unsqueeze(0).clone(),
                            proj_xyz_1.clone().permute(2, 0, 1),
                            proj_remission_1.unsqueeze(0).clone()])
        proj_1 = (proj_1 - self.sensor_img_means[:, None, None]
                  ) / self.sensor_img_stds[:, None, None]
        proj_1 = (proj_1 * proj_mask_1.float())  # [..., 768:1280]

        # get name and sequence
        path_norm_1 = os.path.normpath(scan_file_1)
        path_split_1 = path_norm_1.split(os.sep)
        path_seq_1 = path_split_1[-3]
        path_name_1 = path_split_1[-1].replace(".bin", ".label")

        ###TODO images
        # return rgb, pc, p_x, p_y, proj_mask
        img2_file_1 = transforms.ToTensor()(Image.open(img2_file_1).convert('RGB'))
        img2_file_1 = torch.nn.Upsample(size=(376, 1241), mode='bilinear', align_corners=True)(
            img2_file_1.unsqueeze(0)).squeeze(0)
        # lidarseg_file_1 = transforms.ToTensor()(Image.open(lidarseg_file_1).convert('RGB'))
        rgblabels_file_1 = np.fromfile(rgbseg_file_1, dtype=int)
        rgblabels_file_1 = torch.from_numpy(rgblabels_file_1)
        rgblabels_file_1 = rgblabels_file_1.view((376, 1241))

        ###TODO images
        # rgb,rgbseg,lidarseg,rgb_labels,proj_labels,proj_mask,_ = batch
        img2_file = img2_file_1  # torch.stack([img2_file_1,img2_file_2])
        rgblabels_file = rgblabels_file_1  # torch.stack([rgblabels_file_1,rgblabels_file_2])
        # lidarseg_file = torch.stack([lidarseg_file_1,lidarseg_file_2])
        proj_labels = proj_labels_1  # torch.stack([proj_labels_1,proj_labels_2])
        proj_mask = proj_mask_1  # torch.stack([proj_mask_1,proj_mask_2])
        proj = proj_1  # torch.stack([proj_1,proj_2])
        flow = torch.from_numpy(flow_file)

        rgblabels_file_1 = rgblabels_file_1.unsqueeze(0).unsqueeze(0)
        # rgblabels_file_2 = rgblabels_file_2.unsqueeze(0).unsqueeze(0)

        one_hot = torch.zeros(rgblabels_file_1.size(0), 15, rgblabels_file_1.size(2), rgblabels_file_1.size(3))
        rgb_labels_one_hot_1 = one_hot.scatter_(1, rgblabels_file_1.data, 1).squeeze(0)

        # one_hot = torch.zeros(rgblabels_file_2.size(0), 15, rgblabels_file_2.size(2), rgblabels_file_2.size(3))
        # rgb_labels_one_hot_2 = one_hot.scatter_(1, rgblabels_file_2.data, 1).squeeze(0)
        rgb_labels_one_hot = rgb_labels_one_hot_1  # torch.stack([rgb_labels_one_hot_1, rgb_labels_one_hot_2])

        proj_labels_1 = proj_labels_1.unsqueeze(0).unsqueeze(0).long()
        one_hot = torch.zeros(proj_labels_1.size(0), 20, proj_labels_1.size(2), proj_labels_1.size(3))
        proj_labels_one_hot_1 = one_hot.scatter_(1, proj_labels_1.data, 1).squeeze(0)

        # proj_labels_2 = proj_labels_2.unsqueeze(0).unsqueeze(0).long()
        # one_hot = torch.zeros(proj_labels_2.size(0), 20, proj_labels_2.size(2), proj_labels_2.size(3))
        # proj_labels_one_hot_2 = one_hot.scatter_(1, proj_labels_2.data, 1).squeeze(0)
        proj_labels_one_hot = proj_labels_one_hot_1  # torch.stack([proj_labels_one_hot_1, proj_labels_one_hot_2])



        return img2_file, rgblabels_file, proj_labels, proj_mask, proj, rgb_labels_one_hot, proj_labels_one_hot, flow,path_seq_1,path_name_1\


    def __getitem__(self, index):
        if isinstance(index, list):
            img2_file_l = []
            rgblabels_file_l = []
            proj_labels_l = []
            proj_mask_l = []
            proj_l = []
            rgb_labels_one_hot_l = []
            proj_labels_one_hot_l = []
            flow_l = []
            for idx in index:
                img2_file, rgblabels_file, proj_labels, proj_mask, proj, rgb_labels_one_hot, proj_labels_one_hot, flow,path_seq_1,path_name_1 = self.get_item(
                    idx)
                img2_file_l.append(img2_file)
                rgb_labels_one_hot_l.append(rgb_labels_one_hot)
                proj_labels_l.append(proj_labels)
                proj_mask_l.append(proj_mask)
                proj_l.append(proj)
                proj_labels_one_hot_l.append(proj_labels_one_hot)
                flow_l.append(flow)
                rgblabels_file_l.append(rgblabels_file)

            img2_file = torch.stack(img2_file_l)
            rgblabels_file = torch.stack(rgblabels_file_l)
            proj_labels = torch.stack(proj_labels_l)
            proj_mask = torch.stack(proj_mask_l)
            proj = torch.stack(proj_l)
            rgb_labels_one_hot = torch.stack(rgb_labels_one_hot_l)
            proj_labels_one_hot = torch.stack(proj_labels_one_hot_l)
            flow = torch.stack(flow_l)
            return img2_file, \
                   rgblabels_file, \
                   proj_labels, \
                   proj_mask, \
                   proj, \
                   rgb_labels_one_hot, \
                   proj_labels_one_hot, \
                   flow,path_seq_1,path_name_1
        else:
            return self.get_item(index)

    def __len__(self):
        return len(self.scan_files)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]


class Parser():
    # standard conv, BN, relu
    def __init__(self,
                 root,  # directory for data
                 train_sequences,  # sequences to train
                 valid_sequences,  # sequences to validate.
                 test_sequences,  # sequences to test (if none, don't get)
                 labels,  # labels in data
                 color_map,  # color for each label
                 learning_map,  # mapping for training labels
                 learning_map_inv,  # recover labels from xentropy
                 sensor,  # sensor to use
                 max_points,  # max points in each scan in entire dataset
                 batch_size,  # batch size for train and val
                 workers,  # threads to load data
                 gt=True,  # get gt?
                 shuffle_train=False,  # shuffle training set?
                 transform=False):
        super(Parser, self).__init__()

        # if I am training, get the dataset
        self.root = root
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.max_points = max_points
        self.batch_size = batch_size
        self.workers = workers
        self.gt = gt
        self.shuffle_train = shuffle_train
        self.transform = transform

        # number of classes that matters is the one for xentropy
        self.nclasses = len(self.learning_map_inv)

        # Data loading code
        self.train_dataset = SemanticKitti(root=self.root,
                                           sequences=self.train_sequences,
                                           labels=self.labels,
                                           color_map=self.color_map,
                                           learning_map=self.learning_map,
                                           learning_map_inv=self.learning_map_inv,
                                           sensor=self.sensor,
                                           max_points=max_points,
                                           transform=transform,
                                           gt=self.gt)

        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=self.workers,
                                                       drop_last=True,
                                                       # sampler=BatchSampler(
                                                       #     SequentialSampler(self.train_dataset,4),
                                                       #     batch_size=4, drop_last=True),
                                                       # collate_fn=my_collate
                                                       )

        assert len(self.trainloader) > 0
        self.trainiter = iter(self.trainloader)

        self.valid_dataset = SemanticKitti(root=self.root,
                                           sequences=self.valid_sequences,
                                           labels=self.labels,
                                           color_map=self.color_map,
                                           learning_map=self.learning_map,
                                           learning_map_inv=self.learning_map_inv,
                                           sensor=self.sensor,
                                           max_points=max_points,
                                           gt=self.gt)

        self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=self.workers,
                                                       drop_last=True)
        assert len(self.validloader) > 0
        self.validiter = iter(self.validloader)

        if self.test_sequences:
            self.test_dataset = SemanticKitti(root=self.root,
                                              sequences=self.test_sequences,
                                              labels=self.labels,
                                              color_map=self.color_map,
                                              learning_map=self.learning_map,
                                              learning_map_inv=self.learning_map_inv,
                                              sensor=self.sensor,
                                              max_points=max_points,
                                              gt=False)

            self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          num_workers=self.workers,
                                                          drop_last=True)
            assert len(self.testloader) > 0
            self.testiter = iter(self.testloader)

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()
        return scans

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)

    def get_n_classes(self):
        return self.nclasses

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        # put label in original values
        return SemanticKitti.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        # put label in xentropy values
        return SemanticKitti.map(label, self.learning_map)

    def to_color(self, label):
        # put label in original values
        label = SemanticKitti.map(label, self.learning_map_inv)
        # put label in color
        return SemanticKitti.map(label, self.color_map)
