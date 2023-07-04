from PIL import Image
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """

        # parameters
        self.img = []
        self.label = []
        self.track = []

        # for convert label to int
        label_dict = {'background':0, 'kart':1, 'pickup':2, 'nitro':3, 'bomb':4, 'projectile':5}

        # argument for transform image to tensor
        image_to_tensor = transforms.ToTensor()

        # read in labels.csv 
        with open(dataset_path + '/labels.csv', 'r') as csv_file:

            # get data into reader and skip the first row
            reader = csv.reader(csv_file)
            next(reader)

            # deal with each row in labels.csv
            for row in reader:

                # read in the image and change it into torch.tensor(3,64,64)
                I = Image.open(dataset_path + '/' + row[0])
                self.img.append(image_to_tensor(I))

                # change the label to correspond integer
                self.label.append(label_dict[row[1]])

                # append tracks
                self.track.append(row[2])
        
        #raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        return len(self.label)

        #raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.img[idx], self.label[idx]

        #raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
