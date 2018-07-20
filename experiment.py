"""Author: Brandon Trabucco.
An experiment where humans match captions with attributes.
Args:
    batches: Number of shuffled batches to annotate.
    examples: Number of examples per batch.
"""


import numpy as np
import random
import json
import argparse
from matplotlib.pyplot import title, figure, imshow, axis, text, close
from matplotlib.image import imread
from collections import namedtuple


random.seed(1234567)
np.random.seed(1234567)
Struct = namedtuple("Struct", 
    ["data", "label"])


if __name__ == "__main__":
    """Program entry point, load the files.
    """

    parser = argparse.ArgumentParser("Matching experiment.")
    parser.add_argument("-b", "--batches", 
        type=int, default=1, help="Number of shuffled batches to annotate.")
    parser.add_argument("-e", "--examples", 
        type=int, default=3, help="Number of examples per batch.")
    args = parser.parse_args()

    with open("captions.json", "r") as f:
        data = json.load(f)

    answers = []

    # Loop and create batches.
    for b in range(args.batches):

        # Random indices into the dataset.
        indices = np.random.choice(
            len(data), args.examples, replace=False)

        # Separate the image and captions with ids.
        images, captions = [], []
        for i in indices:
            c = data[i]
            images.append(Struct(
                data=c["image_name"], label=i))
            captions.append(Struct(
                data=c["captions"][0][0], label=i))

        # Randomly shuffle each list.
        random.shuffle(images)
        random.shuffle(captions)

        # Construct a graphical quiz
        fig = figure()
        title("Image Batch {0} of {1}".format(b, args.batches))
        axis('off')
        for i, img in enumerate(images):

            # Display images.
            fig.add_subplot(2, args.examples, 1 + i)
            image = imread(img.data)
            imshow(image)
            axis('off')

            # Display text.
            fig.add_subplot(2, args.examples, 
                1 + i + args.examples)
            text(0.5, 1.0, "Image {0:3d}".format(img.label), 
                horizontalalignment="center")
            axis('off')

        # Build the captions selection.
        question = "Captions were:"
        for i, cap in enumerate(captions):
            question += "\n  ({0:3d}) {1}".format(i, cap.data)

        # Ask for user input.
        fig.show()
        print(question)

        # Collect the answers.
        for img in images:
            while True:
                
                # Read inputs separated by spaces
                iis = input("Image {0:3d} matches captions... ".format(
                    img.label)).strip().split(" ")
                
                # User has entered the empty string
                if len(iis) == 1 and iis[0] == "":
                    answers.append([img, []])
                    break
                
                try:
                    # Try to decode integers
                    jjs = list(set([int(i) for i in iis]))
                    if all([j >= 0 and j < args.examples for j in jjs]):
                        answers.append([img, [captions[j] for j in jjs]])
                        break
                    print("Inputs {0} not in range(0, {1})".format(jjs, args.examples))
                except:
                    print("Inputs {0} not numbers".format(iis))

        # Close the plot window.
        close()

    # Keep track of these.
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    all_labels = set(range(args.examples))

    # Calculate how many answers are correct.
    for img, caps in answers:
        has_correct_label = False
        present_labels = set()

        for cap in caps:
            present_labels.add(cap.label)

            if cap.label == img.label:
                has_correct_label = True
                true_positives += 1

            if cap.label != img.label:
                false_positives += 1

        if not has_correct_label:
            false_negatives += 1

        for z in all_labels.difference(present_labels):
            if z != img.label:
                true_negatives += 1

    # Export the collected statistics.
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    print("Precision is {0:.3f} amnd recall is {1:.3f}".format(precision, recall))

    # Write a data file.
    with open("statistics.json", "w") as f:
        json.dump({
            "precision": precision, "recall": recall,
            "true_positives": true_positives, "true_negatives": true_negatives,
            "false_positives": false_positives, "false_negatives": false_negatives}, f)
