from model import create_model
from client_data import client1_X, client1_y, client2_X, client2_y, client3_X, client3_y

global_model = create_model()
global_weights = global_model.get_weights()


for round in range(1, 4):
    print("\nRound", round)

    model1 = create_model()
    model1.set_weights(global_weights)

    class_weight = {0: 1, 1: 2}

    model1.fit(client1_X, client1_y, epochs=5, batch_size=32, class_weight=class_weight)

    weights1 = model1.get_weights()
    n1 = len(client1_X)


    model2 = create_model()
    model2.set_weights(global_weights)

    model2.fit(client2_X, client2_y, epochs=5, batch_size=32, class_weight=class_weight)

    weights2 = model2.get_weights()
    n2 = len(client2_X)


    model3 = create_model()
    model3.set_weights(global_weights)

    model3.fit(client3_X, client3_y, epochs=5, batch_size=32, class_weight=class_weight)

    weights3 = model3.get_weights()
    n3 = len(client3_X)


    new_weights = []

    for i in range(len(weights1)):

        average = (n1 * weights1[i] + n2 * weights2[i] + n3 * weights3[i]) / (n1 + n2 + n3)
        new_weights.append(average)

    global_weights = new_weights
    print("Federated training completed for round", round)


global_model.set_weights(global_weights)

global_model.save("../results/federated_model.keras")

print("\nFederated training finished!")
print("Final model saved.")