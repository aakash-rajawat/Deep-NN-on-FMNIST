from torch.optim import SGD
from functions_NN import *


def run_model():
    model = nn.Sequential(
        nn.Linear(28 * 28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = SGD(model.parameters(), lr=0.01)
    return model, loss_fn, optimiser


def train_each_batch(x, y, model, optimizer, loss_fn):
    model.train()
    prediction = model(x)
    loss_batch = loss_fn(prediction, y)
    loss_batch.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss_batch.item()


def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_vals, argmaxs = prediction.max(-1)
    is_correct = argmaxs == y
    return is_correct.cpu().numpy().tolist()


training_data_loader = load_data()
model, loss_fn, optimiser = run_model()

losses, accuracies = [], []

for epoch in range(5):
    print(epoch)
    losses_epoch, accuracies_epoch = [], []
    for index, batch in enumerate(iter(training_data_loader)):
        x, y = batch
        loss_batch = train_each_batch(x, y, model, optimiser, loss_fn)
        losses_epoch.append(loss_batch)

    epoch_loss = np.array(losses_epoch).mean()

    for index, batch in enumerate(iter(training_data_loader)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        accuracies_epoch.extend(is_correct)

    epoch_accuracy = np.mean(accuracies_epoch)

    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

epochs = np.arange(5) + 1
plt.figure(figsize=(20, 5))

plt.subplot(121)
plt.title('Loss value over epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()
plt.show()

plt.subplot(122)
plt.title("Accuracy over epochs")
plt.plot(epochs, accuracies, label="Training Accuracy")
#plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
plt.legend()
plt.show()
