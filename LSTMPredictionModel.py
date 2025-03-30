import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data = pd.read_csv('AMZN.csv')
print(data)


### Preparing Data

## The only variable to take it into account is the closing price of the stock.
## Hence, we only keep the date and closing price column
data = data[["Date", "Close"]]

data['Date'] = pd.to_datetime(data['Date']) ## convert date column from string to actual date datatype
plt.plot(data["Date"], data["Close"])
plt.show()


def shift_dataframe(df, n_steps):
    # Create a copy of the dataframe to avoid modifying the input
    df = df.copy()

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i) # shifts values of the whole column backward in time by i positions

    df.dropna(inplace=True)
    # remove rows with missing values which were created by the shifting operation
    # therefore first entry will be the seventh day

    return df


lookback = 7 # Sequential data later will be in packs of seven days
shifted_df = shift_dataframe(data, lookback)
print(shifted_df)

# convert to numpy and scale closing price from -1 to 1
shifted_df_np = shifted_df.to_numpy()
scaler = MinMaxScaler((-1, 1))
shifted_df_np = scaler.fit_transform(shifted_df_np)

# differentiating training features and predicted feature
X = shifted_df_np[:, 1:]
y = shifted_df_np[:, 0] # first column is date
X = np.flip(X, axis=1).copy()
# Original order is from Close(t-1) to Close(t-7), whereas we want the oldest data/day to come first

# 95% training data, 5% testing data. The reason for this is because there is so much data that 5% is enough for testing purposes
train_test_split_index = int(len(X) * 0.95)
X_train = X[:train_test_split_index]
X_test = X[train_test_split_index:]
y_train = y[:train_test_split_index]
y_test = y[train_test_split_index:]

## Shaping the training and testing datasets to fit LSTM model
# Future input tensors must be in the form (batch_size, sequence_length, num_features).
# LSTM expects target output to be (batch_size, output_size). output_size is 1 as in Closing Price
X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# Since dataset is huge, I will be splitting the data into mini-batches

# Wraps our existing data into a Dataset object.
class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

training_dataset = Dataset(X_train, y_train)
testing_dataset = Dataset(X_test, y_test)

batch_size = 16 # Later on, 16 samples will be processed at a time
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
# Randomizes order to prevent overfitting/overspecialization of model to the training data
# Weight updates will no longer be biased towards recent trends in batch sequence
test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
# train_loader and test_loader are basically iterators that will help us load the batches during training and testing

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # For those who have a GPU

# LSTM Architecture
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True) # Emphasize LSTM will be taking in batches

        self.fc = nn.Linear(hidden_size, 1) # map hidden layer to final output of only one feature (Closing price to be predicted)

    def forward(self, x):
        batch_size = x.size(0)
        # h0 and c0 are the hidden state (short-term memory) and cell state (long-term memory) respectively
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        # Initialized to 0 at the start of each sequence as we don't want memory of previous sequences to influence present sequence
        out, _ = self.lstm(x, (h0, c0)) # Pass training features into LSTM
        out = self.fc(out[:, -1, :])  # Selects the last timestep's hidden state (short-term memory) which is the prediction
        return out

model = LSTM(1, 4, 1) # 4 hidden layers and non-stacked LSTM. Found this to be the best settings.
model.to(device) # Load model to GPU if available

# Training The LSTM
learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss() # Mean-Squared Loss - standard loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_one_epoch():
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device) # loads batch into GPU if available

        output = model(x_batch) # generate output prediction
        loss = loss_function(output, y_batch) # compares output prediction to actual value
        running_loss += loss.item()

        optimizer.zero_grad() # resets gradients of previous cycle
        loss.backward() # computes gradient of loss with respect to all parameters
        optimizer.step() # moves weights towards opposite of gradient direction (gradient descent)

        if batch_index % 100 == 99:  # every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1, avg_loss_across_batches))
            print() #newline
            running_loss = 0.0

# Validation - Mock testing our model in its training period by testing it against unseen-before-data (In this case we will just take data from our testing dataset)
def validate_one_epoch():
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print()

for epoch in range(num_epochs):
    model.train(True)
    train_one_epoch()
    model.train(False)
    validate_one_epoch()
    print("------------------")


### Graphing Visualizations
## Graphing our Training Predictions
with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy() #numpy doesn't work on gpu

# Converting predictions back into actual scale
# We have to make our predictions into the original dimensions first before conducting inverse transformation
train_predictions = predicted.flatten()

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)

train_predictions = dummies[:, 0].copy()
print(train_predictions)

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = dummies[:, 0].copy()
print(new_y_train)

plt.plot(new_y_train, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

## Graphing our Testing Predictions
# generate predictions
test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()
# Inverse Scaling as usual
dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dummies[:, 0].copy()
print(train_predictions)

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dummies[:, 0].copy()
print(new_y_test)

plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()