from models import *

# File where we will call the functions from the other files. Serves for testing and organization purposes.

def main():
    #team_values_model_gs()
    #team_values_model_rf() # Accuracy: 0.64

    #player_values_model_gs() # Accuracy: 0.5454545
    model_gradientboost()

    #model_svc()

    #global_model_rf() # Accuracy: 0.5454545

if __name__ == "__main__":
    main()