import glob
from img2bone import HandDetector
import os
import numpy as np
import cv2
import pandas as pd
import torch
from datetime import timedelta
import random
import gc
import math

# # 1-5 train 6 val 7 test
# def img2bone(url, t=['PhongTrang'], time=['20230701'], trainset=[1], testset=[], valset=[]):
#     classes = {}
#     folders = glob.glob(os.path.join(url, "*"))
#     train = []
#     train_label = []
#     train_emg = []
#     test = []
#     test_label = []
#     test_emg = []
#     val = []
#     val_label = []
#     val_emg = []
#     classes = read_class_from_file()
#     train_cnt = 0
#     test_cnt = 0
#     val_cnt = 0

#     handDetector = HandDetector()
#     for type_url in folders:
#         type_name = type_url[type_url.rfind('/')+1:]
#         days = glob.glob(os.path.join(type_url, "*"))
#         if type_name not in t:
#             continue
#         for day in days:
#             d = day[day.rfind('/')+1:]
#             persons = glob.glob(os.path.join(day, "*"))
#             if d not in time:
#                 continue
#             for person in persons:
#                 p_id = person[person.rfind('/')+1:]
#                 print(type_name, d, p_id)
#                 # read class folders
#                 folders = glob.glob(os.path.join(person, "*"))
#                 # divide user into sets
#                 if int(p_id) in trainset:
#                     for folder in folders:
#                         class_name = folder[folder.index("Class ")+6:]
#                         videos = glob.glob(folder+"/*")
#                         emg = merge_EMG_CSV_file(
#                             f"../data/new_data/EMG/{type_name}/CSV/{d}/{p_id}/Class {class_name}", class_name)
#                         if emg is None:
#                             print("class - ", class_name,
#                                   " user ", p_id)
#                             continue
#                         save_trainset(
#                             videos, classes[class_name], emg, type_name, d, p_id, class_name)
#                         print("Train Done person ", p_id,
#                               " class ", class_name)
#                 if int(p_id) in testset:
#                     for folder in folders:
#                         class_name = folder[folder.index("Class ")+6:]
#                         videos = glob.glob(folder+"/*")
#                         emg = merge_EMG_CSV_file(
#                             f"../data/new_data/EMG/{type_name}/CSV/{d}/{p_id}/Class {class_name}", class_name)
#                         if emg is None:
#                             print("class - ", class_name,
#                                   " user ", p_id)
#                             continue
#                         save_testset(
#                             videos, classes[class_name], emg, type_name, d, p_id, class_name)
#                         print("Test Done person ", p_id, " class ", class_name)

#                 if int(p_id) in valset:
#                     for folder in folders:
#                         class_name = folder[folder.index("Class ")+6:]
#                         videos = glob.glob(folder+"/*")
#                         emg = merge_EMG_CSV_file(
#                             f"../data/new_data/EMG/{type_name}/CSV/{d}/{p_id}/Class {class_name}", class_name)
#                         if emg is None:
#                             print("class - ", class_name,
#                                   " user ", p_id)
#                             continue
#                         save_valset(
#                             videos, classes[class_name], emg, type_name, d, p_id, class_name)
#                         print("Val Done person ", p_id, " class ", class_name)


# 1-5 train 6 val 7 test
def img2bone(url, t=['PhongXanh'], time=['20230702'], trainset=[1,2,3,4,5,6,7], testset=[], valset=[]):
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
        if type_name not in t:
            continue
        for day in days:
            d = day[day.rfind('/')+1:]
            persons = glob.glob(os.path.join(day, "*"))
            if d not in time:
                continue
            for person in persons:
                p_id = person[person.rfind('/')+1:]
                print(type_name, d, p_id)
                # read class folders
                folders = glob.glob(os.path.join(person, "*"))
                # divide user into sets
                print(p_id)
                if int(p_id) in trainset:
                    for folder in folders:
                        class_name = folder[folder.index("Class ")+6:]
                        if class_name == "X":
                            continue
                        videos = glob.glob(folder+"/*")
                        emg = merge_EMG_CSV_file(
                            f"../data/new_data/EMG/{type_name}/CSV/{d}/{p_id}/Class {class_name}", class_name)
                        if emg is None:
                            print("class - ", class_name,
                                  " user ", p_id)
                            continue
                        else:

                            # for idx, video in enumerate(videos):
                            #     readVideoAndCovertToBone(
                            #         video, type_name, d, p_id, class_name, idx)

                           

                            f = 44100
                            window = int(f*0.2)
                            length = int(emg.shape[0]//window)
                            for i in range(length):
                                e = emg[i*window:(i+1)*window, :]

                               
                                
                                torch.save((e,classes[class_name]), f"../data/new_data/emg_data/{type_name}_{d}_{p_id}_{class_name}_{i}.pkl")
                            print("Done class = ", class_name)

                # if int(p_id) in testset:
                #     for folder in folders:
                #         class_name = folder[folder.index("Class ")+6:]
                #         videos = glob.glob(folder+"/*")
                #         emg = merge_EMG_CSV_file(
                #             f"../data/new_data/EMG/{type_name}/CSV/{d}/{p_id}/Class {class_name}", class_name)
                #         if emg is None:
                #             print("class - ", class_name,
                #                   " user ", p_id)
                #             continue
                #         else:

                            # f = 44100
                            # window = int(f*0.2)
                            # length = int(emg.shape[0]//window)
                            # for i in range(length):
                            #     e = emg[i*window:(i+1)*window, :]

                            #     e1 = torch.tensor((e - np.min(e, axis=0)) /
                            #                       (np.max(e, axis=0) - np.min(e, axis=0)))

                            #     torch.save((torch.tensor(
                            #         e), e1, classes[class_name]), f"../data/new_data/test2/{type_name}_{d}_{p_id}_{class_name}_{i}.pkl")

                # if int(p_id) in valset:
                #     for folder in folders:
                #         class_name = folder[folder.index("Class ")+6:]
                #         videos = glob.glob(folder+"/*")
                #         emg = merge_EMG_CSV_file(
                #             f"../data/new_data/EMG/{type_name}/CSV/{d}/{p_id}/Class {class_name}", class_name)
                #         if emg is None:
                #             print("class - ", class_name,
                #                   " user ", p_id)
                #             continue
                #         else:

                #             f = 44100
                #             window = int(f*0.2)
                #             length = int(emg.shape[0]//window)
                #             for i in range(length):
                #                 e = emg[i*window:(i+1)*window, :]

                #                 e1 = torch.tensor((e - np.min(e, axis=0)) /
                #                                   (np.max(e, axis=0) - np.min(e, axis=0)))
                #                 torch.save((torch.tensor(
                #                     e), e1, classes[class_name]), f"../data/new_data/val2/{type_name}_{d}_{p_id}_{class_name}_{i}.pkl")


def readVideoAndCovertToBone(url, type_name, d, p_id, class_name, idx):
    video_path = url
    frames = []
    cap = cv2.VideoCapture(video_path)
    handDetector = HandDetector()
    frame_count = 0
    bones = []
    if not cap.isOpened():
        print("Can capture video", url)
        return None
    f = 25
    bones = []
    cnt = 0
    good_bone = []
    while True:

        ret, frame = cap.read()
        # Kiểm tra xem đã đọc hết video hay chưa
        if not ret:
            break

        bone = handDetector.findHands(frame)
        if bone is None:
            bone = good_bone
        else:
            good_bone = bone

        bones.append(bone)
        if len(bones) % (f*0.2) == 0:
            torch.save(
                (bones), f"../data/new_data/vid_data/{type_name}_{d}_{p_id}_{class_name}_{cnt}_{idx}.pkl")
            cnt += 1
            bones = []

    cap.release()
    cv2.destroyAllWindows()


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


def save_trainset(videos, n_class, emg, type_name, d, p_id, class_name):
    train_cnt = 0
    for video in videos:
        frames, bones = readVideoAndCovertToBone(video)
        if bones is not None:
            # train.append(bones)
            # train_label.append(classes[class_name])
            # train_emg.append(emg)
            torch.save((torch.tensor(bones), torch.tensor(
                n_class), torch.tensor(emg)
            ), f'../data/new_data/train1/{type_name}_{d}_{p_id}_{class_name}_{train_cnt}.pkl')
            train_cnt += 1


def save_testset(videos, n_class, emg, type_name, d, p_id, class_name):
    test_cnt = 0
    for video in videos:
        frames, bones = readVideoAndCovertToBone(video)
        if bones is not None:
            # test.append(bones)
            # test_label.append(classes[class_name])
            # test_emg.append(emg)
            torch.save((torch.tensor(bones), torch.tensor(
                n_class), torch.tensor(emg)
            ), f'../data/new_data/test1/{type_name}_{d}_{p_id}_{class_name}_{test_cnt}.pkl')
            test_cnt += 1


def save_valset(videos, n_class, emg, type_name, d, p_id, class_name):
    val_cnt = 0
    for video in videos:
        frames, bones = readVideoAndCovertToBone(video)
        if bones is not None:
            # val.append(bones)
            # val_label.append(classes[class_name])
            # val_emg.append(emg)
            torch.save((torch.tensor(bones), torch.tensor(
                n_class), torch.tensor(emg)
            ), f'../data/new_data/val1/{type_name}_{d}_{p_id}_{class_name}_{val_cnt}.pkl')
            val_cnt += 1


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

            return col.to_numpy()
        except pd.errors.EmptyDataError:
            print(url, " is EmptyColumn")
            return None

    return None


img2bone("../data/new_data/VIDEO")
