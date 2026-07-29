from torch.utils.data import DataLoader

from src.datasets.mammogram_dataset import MammogramDataset

train_dataset = MammogramDataset(
    fold=0,
    mode="train",
)

valid_dataset = MammogramDataset(
    fold=0,
    mode="valid",
)

print("=" * 60)

print("Train Images :", len(train_dataset))

print("Validation Images :", len(valid_dataset))

image, label = train_dataset[0]

print()

print("Tensor Shape :", image.shape)

print("Label :", label)

loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
)

images, labels = next(iter(loader))

print()

print("Batch Shape :", images.shape)

print("Batch Labels :", labels)

print("=" * 60)