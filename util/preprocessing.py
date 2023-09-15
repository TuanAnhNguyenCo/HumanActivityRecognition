import glob
from .img2bone import HandDetector
import os
import numpy as np
import cv2
import torch


def img2bone(url,  train_size=0.6, test_size=0.2, val_size=0.2):
    classes = {}
    folders = glob.glob(os.path.join(url, "*"))
    train = []
    train_label = []
    test = []
    test_label = []
    val = []
    val_label = []
    classes = read_class_from_file()

    handDetector = HandDetector()
    for folder in folders:
        class_name = folder[folder.index("Class ")+6:]
        print("Class name", class_name)
        videos = glob.glob(folder+"/*")
        train_vid = videos[:int(len(videos) * train_size)]
        test_vid = videos[int(len(videos) * train_size)
                              :int(len(videos) * (train_size + test_size))]
        val_vid = videos[int(len(videos) * (train_size+test_size)):]
        for video in train_vid:
            frames, bones = readVideoAndCovertToBone(video)
            if bones is not None:
                train.append(bones)
                train_label.append(classes[class_name])

        for video in test_vid:
            frames, bones = readVideoAndCovertToBone(video)
            if bones is not None:
                test.append(bones)
                test_label.append(classes[class_name])

        for video in val_vid:
            frames, bones = readVideoAndCovertToBone(video)
            if bones is not None:
                val.append(bones)
                val_label.append(classes[class_name])

        print("Done", class_name)

    torch.save((torch.tensor(train), torch.tensor(
        train_label)), '../data/108_new/train.pkl')
    torch.save((torch.tensor(test), torch.tensor(
        test_label)), '../data/108_new/test.pkl')
    torch.save((torch.tensor(val), torch.tensor(val_label)),
               '../data/108_new/val.pkl')


def img2BoneWithHaGRID(ROOT, file_save_path, n=5):
    folder_name = os.listdir(ROOT)
    labels = {}
    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []
    handDetector = HandDetector()
    # read and save to X,y
    for i in range(len(folder_name)):
        labels[folder_name[i]] = i
        file_list = glob.glob(os.path.join(ROOT, folder_name[i])+"/*")
        subset_len = len(file_list)//n
        # shuffle
        np.random.shuffle(file_list)

        X_train.extend(file_list[:subset_len*(n-2)])
        X_val.extend(file_list[subset_len*(n-2):subset_len*(n-1)])
        X_test.extend(file_list[subset_len*(n-1):])

        y_train.extend(np.full(len(file_list[:subset_len*(n-2)]), i))
        y_val.extend(
            np.full(len(file_list[subset_len*(n-2):subset_len*(n-1)]), i))
        y_test.extend(np.full(len(file_list[subset_len*(n-1):]), i))

    for idx, url in enumerate(X_train):
        train_joint_nodes = handDetector.findHands(url, y_train[idx], 0)
        if train_joint_nodes != None:
            write_to_file(file_save_path, "train.txt", train_joint_nodes)

    for idx, url in enumerate(X_val):
        val_joint_nodes = handDetector.findHands(url, y_val[idx], 0)
        if val_joint_nodes != None:
            write_to_file(file_save_path, "val.txt", val_joint_nodes)

    for idx, url in enumerate(X_test):
        test_joint_nodes = handDetector.findHands(url, y_test[idx], 0)
        if test_joint_nodes != None:
            write_to_file(file_save_path, "test.txt", test_joint_nodes)

    write_classes_to_file(file_save_path, "classes.txt", labels)


def readVideoAndCovertToBone(url):
    video_path = url
    frames = []
    cap = cv2.VideoCapture(video_path)
    handDetector = HandDetector()
    frame_count = 0
    bones = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    if not cap.isOpened():
        print("Can capture video", url)
        return None, None
    n_frame = 64

    while True:
        ret, frame = cap.read()
        # Kiểm tra xem đã đọc hết video hay chưa
        if not ret:
            break

        bone = handDetector.findHands(frame)
        if bone is None:
            continue

        bones.append(bone)
        frames.append(frame)
        frame_count += 1
        if frame_count == n_frame:
            return frames, bones

    if frame_count < n_frame:
        return None, None


def write_to_file(file_url, file_name, line):
    with open(file_url+"/"+file_name, 'a+') as f:
        line = ','.join(str(item) for item in line)
        f.write(f"{line}\n")


def write_classes_to_file():
    class_name = []
    data = {}
    with open("../data/108_new/alphabet.txt", 'r') as f:
        class_name = f.readlines()
    for idx, name in enumerate(class_name):
        name = name.replace("\n", "")
        data[name] = idx
    with open("data/108_new/class.txt", 'w') as f:
        for key, value in data.items():
            f.write('%s:%s\n' % (key, value))


def read_class_from_file():
    data = {}
    with open("../data/108_new/class.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace("\n", "").split(":")
        data[line[0]] = int(line[1])

    return data


img2bone("../data/108_new/new_video")
