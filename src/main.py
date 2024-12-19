from models import *

# File where we will call the functions from the other files. Serves for testing and organization purposes.

def main():

    model_randomforest()
    
    model_xgboost()
    model_gradientboost()
    #model_gradientboost_nopca()
    #model_badgb()
    model_svc()
    model_adaboost()
    model_knn()
    model_decisiontree()
    model_mlp()


if __name__ == "__main__":
    main()