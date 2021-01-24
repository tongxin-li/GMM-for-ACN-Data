import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from sklearn import datasets


## Generate an anime
def visualization(X, model):

    # Plotting
    # proba_lists = model.predict_proba(X)
    predict_label = model.predict(X)

    # We are going to do 20 plots, for 20 different angles
    for angle in range(70, 210, 2):

        # Make the plot
        fig = plt.figure(1, figsize=(7, 7))
        ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                   c=predict_label, edgecolor="k", s=20, cmap='gist_rainbow')  # viridis , hot, gist_ncar
        ax.set_xlabel("Arrival Time")
        ax.set_ylabel("Sojourn")
        ax.set_zlabel("Energy")

        # plt.title("EV Clustering via GMMs", fontsize=14)
        plt.show()

        # Set the angle of the camera
        ax.view_init(30, angle)

        # Save it
        filename = 'results/Volcano_step' + str(angle) + '.png'
        plt.savefig(filename, dpi=96)
        plt.gca()