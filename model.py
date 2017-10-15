import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
#from skimage.transform import resize
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

DEBUG = False
BATCH_SZ = 256
EPOCH_NUM = 4
INPUT_SZ = [320, 160] # 320x160
CROP_Y_SZ = [71, 25]
RESIZED_SZ = [200, 66] # 200x66 - using the same size as in Nvidia's CNN
ADD_FLIP = 1
ADD_TRANS = 50

# getting filename using dir postfix
def get_img_fname(row, fname_postfix, idx=0):
    src_path = row[idx]
    return 'data/IMG_' + fname_postfix + '/' + src_path.split('/')[-1]

# opening and processing driving_log.csv file's contetns
def process_csv(fname_postfix, angle_correction, skip_header=False):

    lines = []

    with open('data/driving_log_' + fname_postfix + '.csv') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for line in reader:
            i += 1
            lines.append(line)

        print ("Parsing file", csvfile.name, "with", i, "lines...")

    paths_angles = []

    for line in lines:
        if skip_header:
            skip_header = False
            continue

        # reading all three images and adjusting angles for left/right
        steering_data = float(line[3])
        paths_angles.extend([
            ['data/IMG_' + fname_postfix + '/' + line[0].split('/')[-1], steering_data],                     # center
            ['data/IMG_' + fname_postfix + '/' + line[1].split('/')[-1], steering_data + angle_correction],  # left
            ['data/IMG_' + fname_postfix + '/' + line[2].split('/')[-1], steering_data - angle_correction]]) # right

        if DEBUG: break

    # shuffling pairs together
    return paths_angles

# affine transfromation using cv2
def trans_image(img, angle, trans_range=50):
    # randomly shifting by axis x
    x_new = int(trans_range * np.random.uniform() - trans_range / 2)
    ang_new =  round(angle + x_new * .4 / trans_range, 5)
    # transformation matix: [[M11*x + M12*y + M13], [M21*x + M22*y + M23]]
    T = np.float32([[1, 0, x_new], [0, 1, 0]])
    image_tr = cv2.warpAffine(img, T, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
    return image_tr, ang_new #, tr_x, tr_y

# data generator
def generator(paths_angles, batch_size=32, add_flip=False, add_trans=False):
    num_img_paths = len(paths_angles)
    if add_flip: batch_size = int(batch_size / 2)
    if add_trans: batch_size = int(batch_size / 2)
    while 1: # Loop forever so the generator never terminates
        paths_angles = shuffle(paths_angles)
        for offset in range(0, num_img_paths, batch_size):
            #print ('offset:', offset, "end:", min(num_img_paths, offset+batch_size), "bsz:", batch_size)
            batch_paths_angles = paths_angles[offset : min(num_img_paths, offset+batch_size)]
            images = []
            angles = []
            for path_angle in batch_paths_angles:
                img = cv2.imread(path_angle[0])

                assert img.shape[0] == INPUT_SZ[1] and img.shape[1] == INPUT_SZ[0]
                # cropping
                img = img[CROP_Y_SZ[0]:-CROP_Y_SZ[1], :, :]
                img = cv2.resize(img, (RESIZED_SZ[0], RESIZED_SZ[1]), interpolation = cv2.INTER_AREA)

                # conerting to YUV colorspace - empirically better results
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                images.append(img)
                angles.append(path_angle[1])

                # flipping image for having more training data
                if add_flip:
                    images.append(cv2.flip(img, 1))
                    angles.append(-1.0 * path_angle[1])

            # moving image along the x axis in order to have more training data
            if add_trans:
                imax = len(images)
                for i in range(0, imax):
                    img_trans, angle_trans = trans_image(images[i], angles[i], ADD_TRANS)
                    #print ('transform: i=', i, 'a=', angles[i], 'atr=', angle_trans)
                    images.append(img_trans)
                    angles.append(angle_trans)
                    i += 1

            X = np.array(images)
            y = np.array(angles)

            #print ('X:', X.shape, 'y:', y.shape)
            yield (X, y)

# processing input csvs
paths_angles = []
paths_angles.extend(process_csv('stock', angle_correction=0.249, skip_header=True))
paths_angles.extend(process_csv('fwd', angle_correction=0.252, skip_header=False))
paths_angles.extend(process_csv('bwd', angle_correction=0.252, skip_header=False))
paths_angles.extend(process_csv('drift2', angle_correction=0.252, skip_header=False))
paths_angles.extend(process_csv('curve', angle_correction=0.252, skip_header=False))

# separating validation data from training data (taking random 20%)
paths_angles_train, paths_angles_valid = train_test_split(paths_angles, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(paths_angles_train, batch_size=BATCH_SZ, add_flip=(ADD_FLIP > 0), add_trans=(ADD_TRANS > 0))
validation_generator = generator(paths_angles_valid, batch_size=BATCH_SZ, add_flip=False, add_trans=False)

# calculating samples/epoch based on settings
SAMPLES_PER_EPOCH = len(paths_angles_train)
if ADD_FLIP > 0: SAMPLES_PER_EPOCH *= 2
if ADD_TRANS > 0: SAMPLES_PER_EPOCH *= 2

if DEBUG: exit(0)

# test model 1
def build_model_1():
    model = Sequential()
    model.add(Cropping2D(cropping=((CROP_Y_SZ[0],CROP_Y_SZ[1]), (0,0)), input_shape=(INPUT_SZ[1], INPUT_SZ[0], 3)))
    print('crp:', model.input_shape, model.output_shape)
    model.add(Lambda(lambda x: x / 127.5 - 1.0))
    model.add(Convolution2D(6, 5, 5, activation='elu', subsample=(2,2)))
    print('c_1:', model.input_shape, model.output_shape)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print('p_1:', model.input_shape, model.output_shape)
    model.add(Convolution2D(6, 5, 5, activation='elu', subsample=(2,2)))
    print('c_2:', model.input_shape, model.output_shape)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print('p_2:', model.input_shape, model.output_shape)
    model.add(Flatten())
    print('flt:', model.input_shape, model.output_shape)
    model.add(Dense(120, activation='elu'))
    print('d_1:', model.input_shape, model.output_shape)
    model.add(Dense(84, activation='elu'))
    print('d_2:', model.input_shape, model.output_shape)
    model.add(Dense(1, activation='elu'))
    print('d_3:', model.input_shape, model.output_shape)
    return model

# test model 2
def build_model_2():
    model = Sequential()
    #model.add(BatchNormalization(epsilon=0.001, mode=2, axis=1))
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(RESIZED_SZ[1], RESIZED_SZ[0], 3)))
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    print('c_1:', model.input_shape, model.output_shape)
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    print('c_2:', model.input_shape, model.output_shape)
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    print('c_3:', model.input_shape, model.output_shape)
    model.add(Convolution2D(64, 3, 3, activation='elu', subsample=(1, 1)))
    print('c_4:', model.input_shape, model.output_shape)
    model.add(Convolution2D(64, 3, 3, activation='elu', subsample=(1, 1)))
    print('c_5:', model.input_shape, model.output_shape)
    model.add(Flatten())
    print('flt:', model.input_shape, model.output_shape)
    model.add(Dropout(0.3))
    model.add(Dense(1152, activation='elu'))
    print('d_1:', model.input_shape, model.output_shape)
    model.add(Dropout(0.6))
    model.add(Dense(100, activation='elu'))
    print('d_2:', model.input_shape, model.output_shape)
    #model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    print('d_3:', model.input_shape, model.output_shape)
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    print('d_4:', model.input_shape, model.output_shape)
    model.add(Dense(1, activation='elu'))
    print('d_5:', model.input_shape, model.output_shape)
    return model

if DEBUG: exit(0)

# using model 2
model = build_model_2()

# Adam optimizer w/ learn rate of 0.02%
model.compile(loss='mse', optimizer=Adam(lr=0.0002))

# feeding data
model.fit_generator(train_generator, samples_per_epoch=SAMPLES_PER_EPOCH, validation_data=validation_generator, nb_val_samples=len(paths_angles_valid), nb_epoch=EPOCH_NUM, verbose=1)

# saving model w/ weights
model.save('model.h5')
print(model.summary())

# saving weights separately
model.save_weights('model_weights.h5')

# saving model in keras json format
json_string = model.to_json()
with open('model.json', 'w') as f:
    f.write(json_string)

#print("Testing")
#X_test, y_test = load_images('fwd')
#metrics = model.evaluate(X_test, y_test, verbose=1)
#for metric_i in range(len(model.metrics_names)):
#  metric_name = model.metrics_names[metric_i]
#  #metric_value = metrics[metric_i]
#  #print('{}: {}'.format(metric_name, metric_value))
#  print(metric_name, metrics)


