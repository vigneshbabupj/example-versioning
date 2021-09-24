
import os
import pandas as pd

def test_data_file():
    file = 'data.zip'
    assert not os.path.exists(file), "data.zip is getting uploaded"

def test_model_file():
    file = 'model.h5'
    assert not os.path.exists(file), "model.h5 is getting uploaded"
    
def test_model_acc():
    metrics = pd.read_csv('Metrics.csv')
    acc = metrics.iloc[-1]['accuracy']
    assert acc > 0.7, "Overall acc is < 70%"

def test_cat_acc():
    metrics = pd.read_csv('Metrics.csv')
    acc = metrics.iloc[-1]['cat_acc']
    assert acc > 0.7, "Cat class acc is < 70%"
        
def test_dog_acc():
    metrics = pd.read_csv('Metrics.csv')
    acc = metrics.iloc[-1]['dog_acc']
    assert acc > 0.7, "Dog class acc is < 70%"