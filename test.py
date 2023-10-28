import torch
from collections import defaultdict

train_set = torch.load("data/new_data/new_train_files.pkl")
test_set = torch.load("data/new_data/new_test_files.pkl")
val_set = torch.load("data/new_data/new_val_files.pkl")
vid_train = []
emg_train = []

vid_test = []
emg_test = []

vid_val = []
emg_val = []

train_lb = defaultdict(lambda:0)
test_lb = defaultdict(lambda:0)
val_lb = defaultdict(lambda:0)


for i,j in train_set:
    vid_train.append(i)
    emg_train.append(j)
    # _,label = torch.load(j)
    # train_lb[label] +=1

for i,j in test_set:
    vid_test.append(i)
    emg_test.append(j)
    # _,label = torch.load(j)
    # test_lb[label] +=1

for i,j in val_set:
    vid_val.append(i)
    emg_val.append(j)
    # _,label = torch.load(j)
    # val_lb[label] +=1
    
vid = [*vid_train,*vid_test,*vid_val]
emg = [*emg_train,*emg_val,*emg_test]

torch.save(list(set(emg_train)),"data/new_data/emg_train.pkl")
torch.save(list(set(emg_val)),"data/new_data/emg_val.pkl")
torch.save(list(set(emg_test)),"data/new_data/emg_test.pkl")

print(f'vid_train {len(vid_train)} - {len(set(vid_train))} emg train {len(set(emg_train))}')
print(f'vid_test {len(vid_test)} - {len(set(vid_test))} emg test {len(set(emg_test))}')
print(f'vid_val {len(vid_val)} - {len(set(vid_val))} emg val {len(set(emg_val))}')

print(f"vid {len(vid)} - {len(set(vid))} emg {len(set(emg))}")

# print("train lb")
# for i in range(41):
#     print(f"Label: {i} Num {train_lb[i]}")
    
# print("test lb")
# for i in range(41):
#     print(f"Label: {i} Num {test_lb[i]}")
    
# print("val lb")
# for i in range(41):
#     print(f"Label: {i} Num {val_lb[i]}")
