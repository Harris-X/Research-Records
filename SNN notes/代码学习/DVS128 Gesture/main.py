# https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based/neuromorphic_datasets.html

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

root_dir = r'C:\Users\Harris\Documents\GitWarehouse\Dataset\DVS128Gesture'
train_set = DVS128Gesture(root_dir, train=True, data_type='event')

event, label = train_set[0]
for k in event.keys():
    print(k, event[k])
print('label', label)