from preprocessing import preprocess_data

# Get preprocessed data
X_train, X_test, y_train, y_test = preprocess_data()

# Find the size of each client
client_size = len(X_train) // 3

# Split training data into 3 clients
client1_X = X_train[:client_size]
client1_y = y_train[:client_size]

client2_X = X_train[client_size:2 * client_size]
client2_y = y_train[client_size:2 * client_size]

client3_X = X_train[2 * client_size:]
client3_y = y_train[2 * client_size:]

# Display client sizes
print("Client 1:", client1_X.shape)
print("Client 2:", client2_X.shape)
print("Client 3:", client3_X.shape)