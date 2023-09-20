import glob
from img2bone import HandDetector
import os
import numpy as np
import cv2
import pandas as pd
import torch
from datetime import timedelta
import random


def img2bone(url,  trainset=[1, 2, 3, 4, 5], testset=[7], valset=[6]):
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
    train_cnt = 0
    test_cnt = 0
    val_cnt = 0

    handDetector = HandDetector()
    for type_url in folders:
        type_name = type_url[type_url.rfind('/')+1:]
        days = glob.glob(os.path.join(type_url, "*"))
        for day in days:
            d = day[day.rfind('/')+1:]
            persons = glob.glob(os.path.join(day, "*"))

            for person in persons:
                p_id = person[person.rfind('/')+1:]
                print(type_name, d, p_id)
                # read class folders
                folders = glob.glob(os.path.join(person, "*"))
                # divide user into sets
                if int(p_id) in trainset:
                    for folder in folders:
                        class_name = folder[folder.index("Class ")+6:]
                        videos = glob.glob(folder+"/*")
                        emg = merge_EMG_CSV_file(
                            f"../data/new_data/EMG/{type_name}/CSV/{d}/{p_id}/Class {class_name}", class_name)
                        if emg is None:
                            print("class - ", class_name,
                                  " user ", p_id)
                            continue
                        for video in videos:
                            frames, bones = readVideoAndCovertToBone(video)
                            if bones is not None:
                                # train.append(bones)
                                # train_label.append(classes[class_name])
                                # train_emg.append(emg)
                                torch.save((torch.tensor(bones), torch.tensor(
                                    classes[class_name]), torch.tensor(emg)
                                ), f'../data/new_data/train/{type_name}_{d}_{p_id}_{class_name}_{train_cnt}.pkl')
                                train_cnt += 1
                        print("Train Done person ", p_id,
                              " class ", class_name)
                if int(p_id) in testset:
                    for folder in folders:
                        class_name = folder[folder.index("Class ")+6:]
                        videos = glob.glob(folder+"/*")
                        emg = merge_EMG_CSV_file(
                            f"../data/new_data/EMG/{type_name}/CSV/{d}/{p_id}/Class {class_name}", class_name)
                        if emg is None:
                            print("class - ", class_name,
                                  " user ", p_id)
                            continue
                        for video in videos:
                            frames, bones = readVideoAndCovertToBone(video)
                            if bones is not None:
                                # test.append(bones)
                                # test_label.append(classes[class_name])
                                # test_emg.append(emg)
                                torch.save((torch.tensor(bones), torch.tensor(
                                    classes[class_name]), torch.tensor(emg)
                                ), f'../data/new_data/test/{type_name}_{d}_{p_id}_{class_name}_{test_cnt}.pkl')
                                test_cnt += 1
                        print("Test Done person ", p_id, " class ", class_name)

                if int(p_id) in valset:
                    for folder in folders:
                        class_name = folder[folder.index("Class ")+6:]
                        videos = glob.glob(folder+"/*")
                        emg = merge_EMG_CSV_file(
                            f"../data/new_data/EMG/{type_name}/CSV/{d}/{p_id}/Class {class_name}", class_name)
                        if emg is None:
                            print("class - ", class_name,
                                  " user ", p_id)
                            continue
                        for video in videos:
                            frames, bones = readVideoAndCovertToBone(video)
                            if bones is not None:
                                # val.append(bones)
                                # val_label.append(classes[class_name])
                                # val_emg.append(emg)
                                torch.save((torch.tensor(bones), torch.tensor(
                                    classes[class_name]), torch.tensor(emg)
                                ), f'../data/new_data/val/{type_name}_{d}_{p_id}_{class_name}_{val_cnt}.pkl')
                                val_cnt += 1
                        print("Val Done person ", p_id, " class ", class_name)


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
            try:
                bone = bones[-1]
            except:
                return None, None

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
    with open("../data/new_data/alphabet.txt", 'r') as f:
        class_name = f.readlines()
    for idx, name in enumerate(class_name):
        name = name.replace("\n", "")
        data[name] = idx
    with open("../data/new_data/class.txt", 'w') as f:
        for key, value in data.items():
            f.write('%s:%s\n' % (key, value))


def read_class_from_file():
    data = {}
    with open("../data/new_data/class.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace("\n", "").split(":")
        data[line[0]] = int(line[1])

    return data


def extract_frame(text, key):
    idx = text.find(key)
    return int(text[idx+len(key)])


def merge_EMG_CSV_file(url, class_name):  # test and val are 2s
    data = []
    cnt = 0
    f = 44100
    file_list = glob.glob(url + "/*")
    if file_list != []:
        try:
            col = pd.DataFrame()
            sorted_files = sorted(
                file_list, key=lambda x: extract_frame(x, class_name+"_"))

            for file in sorted_files:
                emg_data = pd.read_csv(file)
                col = pd.concat([col, emg_data], axis=1, ignore_index=True)

            col = col.iloc[:f*3, :]

            return col.to_numpy()
        except pd.errors.EmptyDataError:
            print(url, " is EmptyColumn")
            return None

    return None


img2bone("../data/new_data/VIDEO")
