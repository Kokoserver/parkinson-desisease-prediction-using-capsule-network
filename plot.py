'''
Graph plotting functions.
'''

import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure(figsize=(20, 5))


def plot_loss_acc(path, num_epoch, train_accuracies, train_losses, test_accuracies, test_losses):
    '''
    Plot line graphs for the accuracies and loss at every epochs for both training and testing.
    '''


    epochs = [x for x in range(num_epoch+1)]

    train_accuracy_df = pd.DataFrame({"Epochs": epochs, "Accuracy": train_accuracies, "Mode": [
                                     'train']*(num_epoch+1)}).melt(ignore_index=False)

    test_accuracy_df = pd.DataFrame({"Epochs": epochs, "Accuracy": test_accuracies, "Mode": [
                                    'test']*(num_epoch+1)}).melt(ignore_index=False)


    train_data = pd.concat([train_accuracy_df, test_accuracy_df])
    train_data.to_csv("output1.csv")



    train_loss_df = pd.DataFrame(
        {"Epochs": epochs, "Loss": train_losses, "Mode": ['train']*(num_epoch+1)})
    test_loss_df = pd.DataFrame(
        {"Epochs": epochs, "Loss": test_losses, "Mode": ['test']*(num_epoch+1)})

    test_data = pd.concat([train_loss_df, test_loss_df])
    test_data.to_csv("output.csv")


    return None


def plot_reconstruction(path, num_epoch, original_images, reconstructed_images, predicted_classes, true_classes):
    '''
    Plots 10 reconstructed and original images from testing set.
    '''
    global fig
    plt.clf()
    columns = 10
    rows = 2

    for i in range(1, columns*rows+1):
        img = None
        title = None
        if i > 10:
            title = "Pred Label : "+str(predicted_classes[i-11].item())
            img = reconstructed_images[i-11].permute(1, 2, 0).cpu().numpy()
        else:
            title = "Original Label : "+str(true_classes[i-1].item())
            img = original_images[i-1].permute(1, 2, 0).cpu().numpy()

        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.xticks(())
        plt.yticks(())

    plt.savefig(path+f"Original_vs_Reconstructed_Epoch_{num_epoch}.png")

    return None


def plot(train_acc, train_loss, test_acc, test_loss):
    train_data = pd.DataFrame(list(zip(train_acc, train_loss)), columns=["train_acc", "train_loss"])
    test_data = pd.DataFrame(list(zip(test_acc, test_loss)), columns=["test_acc", "test_loss"])
    loss_data =  pd.DataFrame(list(zip(train_loss, test_loss)), columns=["train_loss", "test_loss"])
    acc_data =  pd.DataFrame(list(zip(train_acc, test_acc)), columns=["train_acc", "test_acc"])

    if train_acc and test_acc and train_loss and test_loss:
        pass
