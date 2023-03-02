import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import random

# Load data from CSV
data = pd.read_csv('MatchTimelinesFirst15.csv')

# Split data into input (X) and output (Y) variables
X = data.drop(['blue_win', 'matchId'], axis=1).values
Y = data['blue_win'].values

# Apply PCA to reduce the number of features to 30
pca = PCA(n_components=17)
X = pca.fit_transform(X)

# Scale the input variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=69)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.666, random_state=69)

# Shape of the used data
print(data.shape)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# Define the neural network model
model = MLPClassifier(hidden_layer_sizes=(32, 16, 8), activation='relu', solver='adam', max_iter=500)

# Train the model
model.fit(X_train, Y_train)

# Evaluate the model 
accuracy = model.score(X_test, Y_test)
print('Accuracy:', accuracy)
accuracy = model.score(X_val, Y_val)
print('Validation accuracy:', accuracy)

# Choose a random game ID from the dataset
random_game_id = random.choice(data['matchId'])

# Retrieve the corresponding row from the dataset
new_data = data.loc[data['matchId'] == random_game_id].drop(['blue_win', 'matchId'], axis=1).values

# Apply PCA and scaling to the new data
new_data = pca.transform(new_data)
new_data = scaler.transform(new_data)

# Predict the winner for the new data
prediction = model.predict(new_data)

# Print the prediction and the game ID
if prediction == 1:
    print("Team Blue wins")
else:
    print("Team Red wins")
print("Game ID:", random_game_id)