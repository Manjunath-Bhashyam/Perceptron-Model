import pandas as pd
import numpy as np
from utils.all_utils import prepare_data, save_model, save_plot
from utils.model import Perceptron
import logging
import os


logging_str = "[%(asctime)s - %(levelname)s - %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str)

def main(data,eta,epochs,filename,PlotFilename):
    
    df = pd.DataFrame(data)
    logging.info(f"This is actual DataFrame{df}")

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

    try:
        logging.info(">>>>> Starting Training here <<<<<")
        main(data=OR, eta=ETA, epochs=EPOCHS, filename="or.model", PlotFilename="or.png")
        logging.info(">>>>>Training done successfully<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e