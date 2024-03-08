# file to decode the logistic files and calculate the accuracy
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Path: test-clustering/interpret_log.py
# Compare this snippet from test-clustering/cluster_algs.py:

def get_accuracy(df):
    # get the max of first two coluns and argmax
    max_values = np.argmax(df[:, :2], axis=1)
    print(max_values)
    print(df[:, :2])
    # compute accurcay of 1 and 0   etween maxvals and the last column  
    accuracy = np.mean(max_values == df[:, 2])
    print(accuracy)
    fpr, tpr, thresholds = roc_curve(y_true=df[:,2], y_score=df[:,1])
    print(thresholds)
    roc_auc = auc(fpr, tpr)

# Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    


def main():

    filename = "C:\\Users\\Leo\\OneDrive\\Dokumente\\Files 2024\\clustering_hp\\test-clustering\\cluster_results_2_0.1.npy"
    output = np.load(filename)
    print(output)
    get_accuracy(output)


main()


