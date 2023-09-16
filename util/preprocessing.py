import glob
from img2bone import HandDetector
import os
import numpy as np
import cv2
import pandas as pd
import torch
from datetime import timedelta
import random


def img2bone(url,  train_size=0.6, test_size=0.2, val_size=0.2):
    classes = {}
    folders = glob.glob(os.path.join(url, "*"))
    train = []
    train_label = []
    train_emg = []
    test = []
    test_label = []
    test_emg = []
    val = []
    val_label = []
    val_emg = []
    classes = read_class_from_file()

    handDetector = HandDetector()
    for folder in folders:
        class_name = folder[folder.index("Class ")+6:]
        print("Class name", class_name)
        videos = glob.glob(folder+"/*")
        random.shuffle(videos)
        train_vid = videos[:int(len(videos) * train_size)]
        test_vid = videos[int(len(videos) * train_size)
                              :int(len(videos) * (train_size + test_size))]
        val_vid = videos[int(len(videos) * (train_size+test_size)):]

        for video in train_vid:
            frames, bones = readVideoAndCovertToBone(video)
            if bones is not None:
                user_id = video[video.rfind("_")+1:video.rfind(".")]
                emg = merge_EMG_CSV_file(
                    "../data/108_new/EMG", class_name, user_id)
                if emg is None:
                    print("class - ", class_name, " user ", user_id)
                    continue
                train.append(bones)
                train_label.append(classes[class_name])
                train_emg.append(emg)

        for video in test_vid:
            frames, bones = readVideoAndCovertToBone(video)
            if bones is not None:
                user_id = video[video.rfind("_")+1:video.rfind(".")]
                emg = merge_EMG_CSV_file(
                    "../data/108_new/EMG", class_name, user_id)
                if emg is None:
                    print("class - ", class_name, " user ", user_id)
                    continue
                test_emg.append(emg)
                test.append(bones)
                test_label.append(classes[class_name])

        for video in val_vid:
            frames, bones = readVideoAndCovertToBone(video)
            if bones is not None:
                user_id = video[video.rfind("_")+1:video.rfind(".")]
                emg = merge_EMG_CSV_file(
                    "../data/108_new/EMG", class_name, user_id)
                if emg is None:
                    print("class - ", class_name, " user ", user_id)
                    continue
                val_emg.append(emg)
                val.append(bones)
                val_label.append(classes[class_name])

        print("Done", class_name)
    train_emg = np.array(train_emg)
    val_emg = np.array(val_emg)
    test_emg = np.array(test_emg)

    torch.save((torch.tensor(train), torch.tensor(train_label),
               torch.tensor(train_emg)), '../data/108_new/train.pkl')
    torch.save((torch.tensor(test), torch.tensor(
        test_label), torch.tensor(test_emg)), '../data/108_new/test.pkl')
    torch.save((torch.tensor(val), torch.tensor(val_label), torch.tensor(val_emg)),
               '../data/108_new/val.pkl')


def readVideoAndCovertToBone(url):
    video_path = url
    frames = []
    cap = cv2.VideoCapture(video_path)
    handDetector = HandDetector()
    frame_count = 0
    bones = []
    if not cap.isOpened():
        print("Can capture video", url)
        return None, None
    n_frame = 75
    cnt = 0

    while True:

        ret, frame = cap.read()
        # Kiểm tra xem đã đọc hết video hay chưa
        if not ret:
            break

        bone = handDetector.findHands(frame)
        if bone is None:
            bone = bones[-1]

        bones.append(bone)
        frames.append(frame)
        frame_count += 1

        if frame_count == n_frame:
            break

    return frames, bones


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


def extract_frame(text, key):
    idx = text.find(key)
    return int(text[idx+len(key)])


def merge_EMG_CSV_file(url, class_name, user_id):  # test and val are 2s
    classes = read_class_from_file()
    data = []
    cnt = 0
    f = 44100

    file_list = glob.glob(url + f"/{class_name}_*{user_id}.csv")
    if file_list != []:
        col = pd.DataFrame()
        sorted_files = sorted(
            file_list, key=lambda x: extract_frame(x, class_name+"_"))
        for file in sorted_files:
            emg_data = pd.read_csv(file)
            col = pd.concat([col, emg_data], axis=1, ignore_index=True)

        col = col.iloc[:f*3, :]
        return col.to_numpy()

    return None


img2bone("../data/108_new/new_video")
