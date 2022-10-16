import pandas as pd
import numpy as np
from utils.all_utils import prepare_data, save_model, save_plot
from utils.model import Perceptron

def main(data,eta,epochs,filename,PlotFilename):
    
    df = pd.DataFrame(data)
    print(df)

    X,y = prepare_data(df)

    model = Perceptron(eta = eta, epochs = epochs)
    model.fit(X,y)

    _ = model.total_loss() # _ is a dummy variable

    save_model(model, filename=filename)
    save_plot(df, PlotFilename, model)

if __name__ == '__main__': # entry point

    OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1]
    }

    ETA = 0.3 # 0 to 1
    EPOCHS = 10

    main(data=OR, eta=ETA, epochs=EPOCHS, filename="or.model", PlotFilename="or.png")