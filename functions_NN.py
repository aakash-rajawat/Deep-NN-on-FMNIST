from classes_NN import *
from dataset import *


def load_data():
    """
    Samples 32 datapoints at random for the batch size.
    :return: Training DataLoader - train_dataloader from the dataset.
    """
    train_data = DatasetTransformer(training_images, training_labels)
    train_loaded_data = DataLoader(train_data, batch_size=32, shuffle=True)
    return train_loaded_data


if __name__ == '__main__':
    loaded_train = load_data()
    for batch_1 in loaded_train:
        batch = batch_1
        print(batch[0].shape)
        break
