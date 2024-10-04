import pandas as pd
import numpy as np 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

if __name__ == '__main__':
    n_patients = 100 
    
    n_tests = 10 
    test_name = [f"f{x}" for x in range(n_tests)]
    
    n_days = 5 # no of days a patient was admitted.
    
    mega_tests = np.random.randn(n_patients * n_days, n_tests)
    patient_name = [f"P{x}" for x in range(n_patients) for y in range(n_days)]
    df = pd.DataFrame(list(zip(patient_name, *zip(*mega_tests))), columns= ["name"] + test_name)
    
    # Adding Prediction.
    wts = np.random.randn(n_tests, 1)
    std = 0.1
    y = mega_tests @ wts + std * np.random.randn(n_patients * n_days, 1)
    df['disease_prob'] = sigmoid(y)
    df.to_csv('Disease.csv', index=False)