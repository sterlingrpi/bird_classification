import matplotlib.pyplot as pyplot
import numpy as np

def inspect_birds(birds, class_numbers, class_names):
    pyplot.figure(figsize=(10 ,10))
    for i in range(16):
        pyplot.subplot(4 ,4 , i +1)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.grid(False)
        pyplot.imshow(birds[i], cmap=pyplot.cm.binary)
        pyplot.xlabel(class_names[class_numbers[i]])
    pyplot.show()

def inspect_bird_results(model, birds, class_numbers, class_names):
    #score and plot
    test_loss, test_acc = model.evaluate(birds, class_numbers)
    print("test loss = ", test_loss )
    print("accuracy = ", test_acc)

    predictions = model.predict(birds)

    num_rows = 3
    num_cols = 3
    pyplot.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(9):
        pyplot.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, class_numbers, birds, class_names)
        pyplot.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, class_numbers)
    pyplot.show()

def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])

    pyplot.imshow(img, cmap=pyplot.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    pyplot.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                            100 * np.max(predictions_array),
                                            class_names[true_label]),
                  color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])
    thisplot = pyplot.bar(range(2), predictions_array, color="#777777")
    pyplot.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')