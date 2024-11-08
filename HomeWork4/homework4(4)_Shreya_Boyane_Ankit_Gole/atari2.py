import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

class CNNPolicyNet(nn.Module):
    def __init__(self, numActions):
        super(CNNPolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(4608, 128)

        self.policy_head = nn.Linear(128, numActions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0),-1)
        x = F.relu(self.fc(x))
        x = F.dropout(x)

        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value
    def predict(self, x, device):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x = x.permute(0, 3, 1, 2)
        l, v = self.forward(x)
        return F.softmax(l, dim=-1), v

def train_model(model, dataloader, optimizer, criterion, device):
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    model.train()
    for batch_obs, batch_actions in dataloader:
        batch_obs = batch_obs.to(device)
        batch_actions = batch_actions.to(device)

        logits, _ = model(batch_obs)

        loss = criterion(logits, batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(logits,dim=1)
        correct_predictions += (predicted == batch_actions).sum().item()
        total_samples += batch_actions.size(0)
        total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_samples

    return model, avg_loss, accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actions = torch.load("pong_actions.pt", weights_only=True)
    observations = torch.load("pong_observations.pt", weights_only=True)
    observations = observations.permute(0, 3, 1, 2)
    #observations = observations.float() / 255.0
    dataloader = DataLoader(TensorDataset(observations, actions), batch_size=32, shuffle=True)

    num_actions = 6
    model = CNNPolicyNet(num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    for epoch in range(num_epochs):
        model, avg_loss, accuracy = train_model(model, dataloader, optimizer, criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("-" * 50)

    torch.save(model.state_dict(), "model_atari.pth")

