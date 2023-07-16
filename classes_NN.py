from torch.utils.data import Dataset

device = "cpu"


class DatasetTransformer(Dataset):
    """
    Shall take two arrays/tensors for parameters training_images (x) and
    training_labels (y).
    """
    def __init__(self, x, y):
        x = x.float() / 255
        x = x.view(-1, 28 * 28)
        self.x, self.y = x, y

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.x)
