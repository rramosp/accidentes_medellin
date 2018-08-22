import os
import re
import pandas as pd
from configparser import ConfigParser

import numpy as np
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, Dense, Flatten, Dropout, Activation
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.applications import InceptionV3, VGG16, VGG19, Xception, ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Leer split de imagenes
# 2. Del split de imagenes, crear el mismo split en CSV
#   - Limpiar las columnas
#   - Normalizar los datos
#   - Combinar los datasets
# 3. Entrenar
#    - solo imagenes
#    - solo csv
#    - combinar imagenes y csv


def clean_up_columns(data):
    data.columns = map(str.lower, data.columns)
    data.columns = map(str.strip, data.columns)
    if 'geometry' in data.columns:
        data.drop('geometry', axis=1, inplace=True)
    if 'processed' in data.columns:
        data.drop('processed', axis=1, inplace=True)
    if 'sample' in data.columns:
        data.drop('sample', axis=1, inplace=True)
    data = data.dropna()
    data.set_index('id', inplace=True)
    return data

def join_dataframes(*multiple_df):
    output = []
    for df in multiple_df:
        if len(output):
            output = df.join(output)
        else:
            output = df
    return output.dropna()       

def normalize_data(df):
    return (df - df.mean()) / (df.max() - df.min())

def create_data_splits(accidents_dir, fetex_dir, metrics_dir, grid_id):
    # Load Accidents
    accidents = pd.read_csv(f'{accidents_dir}/m{grid_id}_c.csv')
    accidents = clean_up_columns(accidents) 

    # Load Fetex
    fetex = pd.read_csv(f'{fetex_dir}/r_fetex_m{grid_id}.txt')
    fetex = clean_up_columns(fetex)

    # Load metrics data
    metrics = pd.read_csv(f'{metrics_dir}/{grid_id}.csv', delimiter=';')
    metrics = clean_up_columns(metrics)
    metrics.circuity_avg = pd.to_numeric(metrics.circuity_avg)
    
    # Get train split from file
    train_split = pd.read_csv(f'../support_data/{grid_id}/train.csv', index_col='id')
    test_split = pd.read_csv(f'../support_data/{grid_id}/test.csv', index_col='id')
    
    # Create all joined dataframes
    acc_train = join_dataframes(train_split, accidents)
    acc_test  = join_dataframes(test_split, accidents)
    fx_train  = join_dataframes(train_split, fetex)
    fx_test   = join_dataframes(test_split, fetex)
    metrics_train = join_dataframes(train_split, metrics)
    metrics_test  = join_dataframes(test_split, metrics)
    acc_fx_train  = join_dataframes(train_split, accidents, fetex)
    acc_fx_test   = join_dataframes(test_split, accidents, fetex)
    fx_metrics_train  = join_dataframes(train_split, fetex, metrics)
    fx_metrics_test   = join_dataframes(test_split, fetex, metrics)
    acc_metrics_train = join_dataframes(train_split, accidents, metrics)
    acc_metrics_test  = join_dataframes(test_split, accidents, metrics)
    acc_fx_metrics_train = join_dataframes(train_split, accidents, fetex, metrics)
    acc_fx_metrics_test  = join_dataframes(test_split, accidents, fetex, metrics)

    # Create joins
    single_join = {'accidents':(acc_train, acc_test), 'fetex':(fx_train, fx_test), 'metrics':(metrics_train, metrics_test)}
    double_join = {'accidents_fetex':(acc_fx_train, acc_fx_test), 'fetex_metrics':(fx_metrics_train, fx_metrics_test), 'accidents_metrics':(acc_metrics_train, acc_metrics_test)}
    triple_join = {'acc_fx_metrics':(acc_fx_metrics_train, acc_fx_metrics_test)}
    
    return [single_join, double_join, triple_join]


def get_model_only_images(model_name, main_input, img_width, img_height):
    if model_name == 'inceptionV3':
        base_model = InceptionV3(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False, input_tensor=main_input)
        last_layer_number = 249
    elif model_name == 'vgg16':
        base_model = VGG16(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False, input_tensor=main_input)
        last_layer_number = len(base_model.layers)
    elif model_name == 'vgg19':
        base_model = VGG16(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False, input_tensor=main_input)
        last_layer_number = len(base_model.layers)
    elif model_name == 'xception':
        base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False, input_tensor=main_input)
        last_layer_number = len(base_model.layers)
    elif model_name == 'resnet50':
        base_model = ResNet50(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False, input_tensor=main_input)
    
    return base_model, last_layer_number

def get_callback_list(top_weights_path):
    callback_list = [
        TensorBoard(log_dir=f'logs/{top_weights_path}', write_grads=True, write_images=True), 
        EarlyStopping(monitor='val_acc', patience=10, verbose=1),
        ModelCheckpoint(f'models/{top_weights_path}', verbose=1, save_best_only=True, monitor='val_acc'),
        ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.1)
    ]
    return callback_list

def train_only_images(model_name, num_classes, lr_rate, img_width, img_height, imgs_dir, grid_id):
    np.random.seed(1)

    train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    test_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(
        f'{imgs_dir}/{grid_id}/train/',
        target_size=(img_width, img_height),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(
        f'{imgs_dir}/{grid_id}/test/',
        target_size=(img_width, img_height),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

    main_input = Input(shape=(img_width, img_height, 3))
    base_model, last_layer_number = get_model_only_images(model_name, main_input, img_width, img_height)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=main_input, outputs=predictions)
    top_weights_path = f'only_images_{model_name}_lr{lr_rate}.h5'

    for layer in base_model.layers:
        layer.trainable = False

    adam = Adam(lr=lr_rate, epsilon=1e-7)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callback_list = get_callback_list(top_weights_path)

    model.fit_generator(
        train_generator,
        steps_per_epoch= train_generator.n // 32,
        epochs=100,
        validation_data=test_generator,
        validation_steps= test_generator.n // 32,
        callbacks=callback_list)

    model.save(f'models/{top_weights_path}')
    

def train_simple_net(train, test, lr_rate, dataset_name, num_classes):
    np.random.seed(1)
        
    x_train, y_train = train.drop('label', axis=1), train.label
    x_test, y_test = test.drop('label', axis=1), test.label
    x_train = normalize_data(x_train).fillna(0)
    x_test = normalize_data(x_test).fillna(0)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    _, features = x_train.shape

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    aux_input = Input(shape=(features,) )
    aux = Dense(1024, activation='relu')(aux_input)
    aux = Dense(1024, activation='relu')(aux)
    aux = Dense(1024, activation='relu')(aux)
    aux = Dense(1024, activation='relu')(aux)
    aux = Dense(num_classes, activation='sigmoid')(aux)
    predictions = Activation('softmax')(aux)

    model = Model(inputs=aux_input, outputs=predictions)
    adam = Adam(lr=lr_rate, epsilon=1e-7)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    top_weights_path = f'simple_{dataset_name}_lr{lr_rate}'
    callback_list = get_callback_list(top_weights_path)

    model.fit(x_train, 
        y_train, 
        validation_data=(x_test, y_test), 
        epochs=100, 
        shuffle=True,
        batch_size=32,
        callbacks=callback_list, verbose=1)

    model.save(f'models/{top_weights_path}')
    #model.load_weights(top_weights_path)


def train_combined_model(model_name, num_classes, lr_rate, img_width, img_height, imgs_dir, grid_id, train, test, dataset_name):
    np.random.seed(1)

    # Create image and csv generators
    x_train, y_train = train.drop('label', axis=1), train.label
    x_test, y_test = test.drop('label', axis=1), test.label
    x_train = normalize_data(x_train).fillna(0)
    x_test = normalize_data(x_test).fillna(0)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    _, features = x_train.shape

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    test_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(
        f'{imgs_dir}/{grid_id}/train/',
        target_size=(img_width, img_height),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        f'{imgs_dir}/{grid_id}/test/',
        target_size=(img_width, img_height),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

    def my_generator(image_gen, data):
        while True:
            i = image_gen.batch_index
            batch = image_gen.batch_size
            row = data[i*batch:(i+1)*batch]
            images, labels = image_gen.next()
            
            yield [images, row], labels
               
    csv_train_generator = my_generator(train_generator, x_train)
    csv_test_generator = my_generator(test_generator, x_test)

    # Load Image Network
    main_input = Input(shape=(img_width, img_height, 3))
    base_model, last_layer_number = get_model_only_images(model_name, img_width, img_height)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    for layer in base_model.layers:
        layer.trainable = False

    # Load CSV Network
    aux_input = Input(shape=(features,))
    aux = Dense(1024, activation='relu')(aux_input)
    aux = Dense(1024, activation='relu')(aux_input)
    aux = Dense(1024, activation='relu')(aux_input)
    aux = Dense(1024, activation='relu')(aux_input)

    # Merge models
    merge = concatenate([x, aux])
    merge = Dense(1024, activation='relu')(merge)
    merge = Dense(1024, activation='relu')(merge)
    merge = Dense(num_classes, activation='sigmoid')(merge)
    predictions = Activation('softmax')(merge)

    model = Model(inputs=[main_input, aux_input], outputs=predictions)
    top_weights_path = f'combined_{model_name}_{dataset_name}_lr{lr_rate}'
    callback_list = get_callback_list(top_weights_path)

    adam = Adam(lr=lr_rate, epsilon=1e-7)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(
        csv_train_generator,
        steps_per_epoch=train_generator.n//32,
        epochs=100,
        validation_data=csv_test_generator,
        validation_steps=test_generator.n//32,
        callbacks=callback_list,
        verbose=1)

    model.save(f'models/{top_weights_path}')
    

if __name__ == '__main__':

    cfg = ConfigParser()
    cfg.read('../config.ini')
    fetex_dir = cfg.get('dirs','fetex')
    accidents_dir = cfg.get('dirs', 'accidentes')
    metrics_dir = cfg.get('dirs', 'road_metrics')
    imgs_dir = cfg.get('dirs', 'img_tiles')
    grid_id = cfg.get('grid', 'default')
    grid_id = re.sub("\D", "", grid_id)
    print(f'using {grid_id}')

    for model in ['inceptionV3', 'vgg16', 'vgg19', 'xception', 'resnet50']:
        train_only_images(model, 4, lr_rate, 420, 420, imgs_dir, grid_id)
                
    for data_list in create_data_splits(accidents_dir, fetex_dir, metrics_dir, grid_id):
        for key, (train, test) in data_list.items():
            for lr_rate in [0.01, 0.0001]:
                train_simple_net(train, test, lr_rate, key, 4)
                
                for model in ['inceptionV3', 'vgg16', 'vgg19', 'xception', 'resnet50']:
                    train_combined_model(model, 4, lr_rate, 420, 420, imgs_dir, grid_id, train, test, key)
