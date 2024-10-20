import pandas as pd
import numpy as np 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

if __name__ == '__main__':
########### Training Set ###########
    n_patients = 1000 
    
    n_tests = 10 
    test_name = [f"f{x}" for x in range(n_tests)]
    
    mega_tests = np.random.randn(n_patients, n_tests)
    patient_name = [f"P{x}" for x in range(n_patients)]
    df = pd.DataFrame(list(zip(patient_name, *zip(*mega_tests))), columns= ["name"] + test_name)
    
    # Adding Prediction.
    wts = np.random.randn(n_tests, 1)
    std = 0.1
    y = mega_tests @ wts + std * np.random.randn(n_patients, 1)
    df['has_disease'] = sigmoid(y)
############ Test Set #############
    n_patients = 15 
    
    n_tests = 10 
    test_name = [f"f{x}" for x in range(n_tests)]
    
    mega_tests = np.random.randn(n_patients, n_tests)
    patient_name = [f"PT{x}" for x in range(n_patients)]
    dft = pd.DataFrame(list(zip(patient_name, *zip(*mega_tests))), columns= ["name"] + test_name)
########### Save ###########
    df = pd.concat([df, dft])
    df.loc[df['has_disease'] >= 0.5, 'has_disease'] = 1.0
    df.loc[df['has_disease'] < 0.5, 'has_disease'] = 0.0
    df.to_csv('Disease_e432.csv', index=False)