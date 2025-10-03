import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

def hill_climber(x, y):
    # Ws = [-1, -1, -1, -1, -1, -1]
    Ws = np.random.choice([-1, 1], size=6)
    best_erW = MSE(Ws, x, y) 
    history = [best_erW]

    while True:
        improved = False
        # change Ws to different adjaent solution
        for i in range(len(Ws)):
            neighbor = Ws.copy()
            neighbor[i] *= -1

            # check current error and see if lower than current best
            current_erW = MSE(neighbor, x, y)
            if current_erW < best_erW:
                Ws = neighbor
                history.append(current_erW)
                best_erW = current_erW
                improved = True

        if not improved: # stuck in local min
            break

    return Ws, best_erW, history
        


# function to calculate the mean standard error. 
# mean((predicted - actual outcome)^2)
def MSE(w, x, y):
    w = np.array(w)  
    prediction = x @ w
    return np.mean((prediction - y) ** 2)



    
# read in credit card data and change all values to integers
def load_data(filename="CreditCard.csv"):
    df = pd.read_csv(filename)

    # encode string values to integers
    df["Gender"] = df["Gender"].map({"M": 1, "F": 0})
    df["CarOwner"] = df["CarOwner"].map({"Y": 1, "N": 0})
    df["PropertyOwner"] = df["PropertyOwner"].map({"Y": 1, "N": 0})

    # Features (X) and labels (y)
    X = df[["Gender", "CarOwner", "PropertyOwner", "#Children", "WorkPhone", "Email_ID"]].values
    y = df["CreditApprove"].values

    return X, y





if __name__ == "__main__":
    
    # get data from CreditCard.csv
    x,y = load_data()

    Ws, erW, history = hill_climber(x, y)

    plt.plot(range(len(history)), history, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Hill Climbing Error Over Time')

    print("Final weights:", Ws)
    print("Final error:", erW)
    plt.show()
