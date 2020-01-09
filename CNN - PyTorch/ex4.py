from gcommand_loader import GCommandLoader
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

file_name = "test_y"

# Hyper parameters:
epochs = 15
num_classes = 30
batch_size = 50
learning_rate = 0.001


class Database:
    def __init__(self, train_dir, valid_dir, test_dir):
        self.train_set, _ = self.get_data(train_dir, shuffle=True)
        self.valid_set, _ = self.get_data(valid_dir, shuffle=False)
        self.test_set, self.test_loader = self.get_data(test_dir, shuffle=False)

    def get_data(self, dir_name, shuffle):
        # dataset = GCommandLoader('./ML4_dataset/data/' + dir_name)
        dataset = GCommandLoader('./' + dir_name)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=20, pin_memory=True, sampler=None), dataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = self.layer_setup(1, 10)
        self.layer2 = self.layer_setup(10, 20)
        self.layer3 = self.layer_setup(20, 20)
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(4800, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x)

    def layer_setup(self, size1, size2):
        conv = nn.Conv2d(size1, size2, kernel_size=5, stride=1, padding=2)
        relu = nn.ReLU()
        max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        return nn.Sequential(conv, relu, max_pool)


def test(model, test_set):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for audios, tags in test_set:
            # audios = audios.to("cuda")
            # tags = tags.to("cuda")
            outputs = model(audios)
            _, predicted = torch.max(outputs.data, 1)
            total += tags.size(0)
            correct += (predicted == tags).sum().item()
        print('results: {} %'.format((correct / total) * 100))


def train(training_set, criterion, model):
    model.train()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    total_step = len(training_set)
    running_loss_list = []
    for e in range(epochs):
        for i, (audios, tags) in enumerate(training_set):
            # audios = audios.to("cuda")
            # tags = tags.to("cuda")
            outputs = model(audios)
            loss = criterion(outputs, tags)
            running_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total = tags.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == tags).sum().item()

            if (i + 1) % 10 == 0:
                print('epoch [{}/{}], step [{}/{}], accuracy: {:.2f}%' .format(e + 1, epochs, i + 1, total_step, (correct / total) * 100))


def start_training(train_set, model):
    loss_function = nn.CrossEntropyLoss()
    train(train_set, loss_function, model)


def start_testing(validation_set, model):
    print("End of training.")
    print("testing...")
    test(model, validation_set)


def make_predictions(test_set, test_loader, model):
    model.eval()
    predictions = []
    dic = {}

    with torch.no_grad():
        for audios, tags in test_set:
            # audios = audios.to("cuda")
            outputs = model(audios)
            _, predicted = torch.max(outputs.data, 1)
            for p in predicted:
                predictions.append(p.item())

    for i, data in enumerate(test_loader.spects):
        name = data[0].split('/')[5]
        dic.update({name: predictions[i]})
    print(len(dic))
    return dic


def save_predictions(predictions):
    file = open(file_name, 'w')
    for name, predict in predictions.items():
        file.write(name + ', ' + np.str(predict) + '\n')
    file.close()


def main():
    database = Database('train', 'valid', 'mainTest')
    # model = Net().to("cuda")
    model = Net()
    start_training(database.train_set, model)
    start_testing(database.valid_set, model)
    predictions = make_predictions(database.test_set, database.test_loader, model)
    save_predictions(predictions)
    print("done")


if __name__ == '__main__':
    main()

