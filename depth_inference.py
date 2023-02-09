import os
import numpy as np
import cv2
import torch

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

EXTENSIONS_IMAGE = ['.png']


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)


if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


root = os.path.join('/data/tiacor/Downloads/dataset', "sequences")
for seq in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
            '18', '19', '20', '21']:
    # to string
    img2_files = []
    seq = '{0:02d}'.format(int(seq))

    print("parsing seq {}".format(seq))

    # get paths for each
    img2_path = os.path.join(root, seq, "image_2")

    # get files
    img2_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(img2_path)) for f in fn if is_image(f)]

    # extend list
    img2_files.extend(img2_files)
    img2_files.sort()
    for img_file in img2_files:
        img = cv2.imread(img_file)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)
        directory = '/data/tiacor/Downloads/dataset/sequences/{}/depth'.format(seq)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(376,1241),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        np.save(directory+'/'+img_file.split('/')[-1].split('.')[0],output)
        print("\rTreating Frame {}/{} of Sequence {}".format(img_file.split('/')[-1].split('.')[0],len(img2_files),seq), end='')

    # sort for correspondance



if __name__ == "__main__":
    pass