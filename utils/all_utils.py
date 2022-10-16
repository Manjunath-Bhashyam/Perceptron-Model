import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from matplotlib.colors import ListedColormap

plt.style.use("fivethirtyeight")

def prepare_data(df):
  X = df.drop("y", axis=1)
  y = df["y"]
  return X, y

# Saving Perceptron Model

def save_model(model, filename):
  model_dir = "models"
  os.makedirs(model_dir,exist_ok=True) # Only create if model_dir diesn't exist
  filePath = os.path.join(model_dir, filename) # creates models\filename
  
  joblib.dump(model,filePath)

# Saving plot

def save_plot(df, file_name, model):
  def _create_base_plot(df):
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter") #Pandas direct plot method c means color level outcome s-dia of circle
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(10,8)

  def _plot_decision_regions(X, y, classifier, resolution=0.02):
    colors = ("red","blue","lightgreen","gray","cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values # need X as array not as DataFrame
    x1 = X[:,0]
    x2 = X[:,1]
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution), np.arange(x2_min,x2_max,resolution))

    print(xx1)
    print(xx1.ravel())

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()

  X,y = prepare_data(df)
  
  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) # only create if plot_dir doesn't exists
  plotpath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotpath)