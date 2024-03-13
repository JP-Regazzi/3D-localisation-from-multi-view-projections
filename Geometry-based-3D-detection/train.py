import numpy as np
import os
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from layer.loss_func import orientation_loss
from util.process_data import load_and_process_annotation_data,train_data_gen
from net.bbox_3D_net import bbox_3D_net
from keras.metrics import MeanIoU, Recall, Precision, AUC, F1Score
from sklearn.metrics import f1_score
import keras.backend as K

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Construct the network
model = bbox_3D_net((224,224,3),bin_num=6,vgg_weights='imagenet')

minimizer = Adam(learning_rate=1e-5)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq=1)
tensorboard = TensorBoard(log_dir='../logs/', histogram_freq=0, write_graph=True, write_images=False)

model.compile(optimizer=minimizer,#minimizer,
            loss={'dimension': 'mean_squared_error', 'orientation': orientation_loss, 'confidence': 'categorical_crossentropy'},
            metrics=[AUC(name='average_precision'), Recall(), Precision()],
            loss_weights={'dimension': 2., 'orientation': 1., 'confidence': 4.})


image_dir = 'C:/Users/pcc/Desktop/s7_project/3D_detection-master/custom/images/'
label_dir = 'C:/Users/pcc/Desktop/s7_project/3D_detection-master/custom/labels'

classes = [line.strip() for line in open(r'dataset/voc_labels.txt').readlines()]
cls_to_ind = {str(i):i for i,cls in enumerate(classes)}

dims_avg = np.loadtxt(r'dataset/voc_dims.txt',delimiter=',')

objs = load_and_process_annotation_data(label_dir,dims_avg,cls_to_ind)

objs_num = len(objs)
train_num = int(0.9*objs_num)
batch_size = 2          # Need to change the batch size, not to forget !
np.random.shuffle(objs)

train_gen = train_data_gen(objs[:train_num], image_dir, batch_size, bin_num=6)
valid_gen = train_data_gen(objs[train_num:], image_dir, batch_size, bin_num=6)

train_epoch_num = int(np.ceil(train_num/batch_size))
valid_epoch_num = int(np.ceil((objs_num - train_num)/batch_size))

model.fit_generator(generator = train_gen,
                    steps_per_epoch = train_epoch_num,
                    epochs = 3,
                    verbose = 1,
                    validation_data = valid_gen,
                    validation_steps = valid_epoch_num,
                    callbacks = [early_stop, checkpoint, tensorboard]
                    )

model.save_weights(r'model_saved/weights.h5')


