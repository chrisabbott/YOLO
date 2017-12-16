import itertools
import os
import numpy as np
import skimage.io as io


def main():
    cwd = os.getcwd()

    # Get the labels and map each one to an integer ID number (e.g, "n03444034": 42)
    labels_file = os.path.join(cwd, 'datasets/tiny-imagenet-200/wnids.txt')
    with open(labels_file) as f:
        all_labels = f.readlines()
    counter = itertools.count()
    label_map = {label.strip(): next(counter) for label in all_labels}

    train_dir = os.path.join(cwd, 'datasets/tiny-imagenet-200/train/')
    test_dir = os.path.join(cwd, 'datasets/tiny-imagenet-200/val/')

    train_images = []
    train_labels = []
    for category in os.listdir(train_dir):
        label = label_map[category]
        category_dir = os.path.join(train_dir, category)
        images_dir = os.path.join(category_dir, 'images/')
        for filename in os.listdir(images_dir):
            image_path = os.path.join(images_dir, filename)
            image = io.imread(image_path)
            train_images.append(image)
            train_labels.append(label)

    test_labels_file = os.path.join(test_dir, 'val_annotations.txt')
    test_image_label_map = {}
    with open(test_labels_file) as f:
        for line in f:
            filename, label = line.split('\t')[0:2]
            test_image_label_map[filename] = label_map[label]

    test_images = []
    test_labels = []
    test_images_dir = os.path.join(test_dir, 'images/')
    for filename in os.listdir(test_images_dir):
        label = test_image_label_map[filename]
        image_path = os.path.join(test_images_dir, filename)
        image = io.imread(image_path)
        test_images.append(image)
        test_labels.append(label)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)


if __name__ == '__main__':
    main()
