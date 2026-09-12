from model import create_model
from client import DiabetesClient
from client_data import client1_X, client1_y, client2_X, client2_y, client3_X, client3_y

# Create 3 clients
client1 = DiabetesClient(client1_X, client1_y)
client2 = DiabetesClient(client2_X, client2_y)
client3 = DiabetesClient(client3_X, client3_y)

# Create global model
global_model = create_model()

# Get initial weights
global_weights = global_model.get_weights()

# Federated training
for round in range(1, 4):

    print("\nRound", round)

    # Train each client
    weights1, n1 = client1.train(global_weights)
    weights2, n2 = client2.train(global_weights)
    weights3, n3 = client3.train(global_weights)

    # Weighted FedAvg
    new_weights = []

    for i in range(len(weights1)):
        average = (n1 * weights1[i] + n2 * weights2[i] + n3 * weights3[i]) / (n1 + n2 + n3)
        new_weights.append(average)

    # Update global weights
    global_weights = new_weights

    print("Federated training completed for round", round)

# Put final weights into global model
global_model.set_weights(global_weights)

# Save final model
global_model.save("../results/federated_model.keras")

print("\nFederated training finished!")
print("Final model saved.")