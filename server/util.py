import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bed,bath):
     try:
         loc_index = __data_columns.index(location.lower())
     except:
         loc_index = -1

     x = np.zeros(len(__data_columns))
     x[0]= sqft
     x[1]= bath
     x[2] = bed
     if loc_index >= 0:
         x[loc_index] = 1

    return __model.predict([x])

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("Loading saving artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json",'r') as f:
       __data_columns = json.load(f)['data_columns']
       __locations = __data_columns[4:]

    with open("./artifacts/banglore_home_prices_model.pickle",'rb') as g:
        __model = pickle.load(g)
    print("Loading saved artifacts ... done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())