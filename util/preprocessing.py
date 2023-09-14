import glob
from .img2bone import HandDetector
import os
import numpy as np
import cv2


def img2bone(train_url, test_url, file_url, valid_size=0.3):
    classes = {}
    train_folders = glob.glob(train_url+"/*")
    test_folders = glob.glob(test_url+"/*")
    cnt = 0
    train_joint_nodes = []
    val_joint_nodes = []
    test_joint_nodes = []
    handDetector = HandDetector()
    for folder_name in train_folders:
        class_name = folder_name[folder_name.index("hand_")+5:]
        if classes.get(class_name) == None:
            cnt += 1
            classes[class_name] = cnt - 1
        image_urls = glob.glob(folder_name+"/*")
        train_idx = int(len(image_urls)*(1-valid_size))
        training_image_urls = image_urls[:train_idx]
        validation_image_urls = image_urls[train_idx:]
        for url in training_image_urls:
            train_joint_nodes = handDetector.findHands(
                url, classes[class_name], 0)
            if train_joint_nodes != None:
                write_to_file(file_url, "train.txt", train_joint_nodes)

        for url in validation_image_urls:
            val_joint_nodes = handDetector.findHands(
                url, classes[class_name], 1)
            if val_joint_nodes != None:
                write_to_file(file_url, "validation.txt", val_joint_nodes)

        print("Finish train folder = ", class_name)

    for folder_name in test_folders:
        class_name = folder_name[folder_name.index("hand_")+5:]
        if classes.get(class_name) == None:
            cnt += 1
            classes[class_name] = cnt - 1
        image_urls = glob.glob(folder_name+"/*")

        for url in image_urls:
            test_joint_nodes = handDetector.findHands(
                url, classes[class_name], 2)
            if test_joint_nodes != None:
                write_to_file(file_url, "test.txt", test_joint_nodes)
        print("Finish test folder = ", class_name)

    write_classes_to_file(file_url, "classes.txt", classes)


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
        return None, None
    n_frame = 64

    while True:
        ret, frame = cap.read()
        # Kiểm tra xem đã đọc hết video hay chưa
        if not ret:
            break

        bone = handDetector.findHands(frame, 1, 1)
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


def write_classes_to_file(file_url, file_name, data):
    with open(file_url+"/"+file_name, 'w+') as f:
        for key, value in data.items():
            f.write('%s:%s\n' % (key, value))
