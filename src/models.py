import tensorflow as tf
from keras.layers import Conv3D, BatchNormalization, Activation, Add, Dense, Input, ZeroPadding3D, MaxPooling3D, AveragePooling3D, Flatten
from keras.models import Model
from typing import List, Tuple

def identity_block(X: tf.Tensor, level: int, block: int, filters: List[int]) -> tf.Tensor:

    conv_name  = f'conv{level}_{block}' + '_{layer}_{type}'
    f1, f2, f3 = filters
    X_shortcut = X

    X = Conv3D(filters=f1, kernel_size=(1, 1, 1), strides=(1, 1, 1), name=conv_name.format(layer=1, type='conv', padding='valid'))(X)
    X = BatchNormalization(axis=-1, name=conv_name.format(layer=1, type='bn'))(X)
    X = Activation('elu', name=conv_name.format(layer=1, type='elu'))(X)

    X = Conv3D(filters=f2, kernel_size=(3, 3, 3), strides=(1, 1, 1), name=conv_name.format(layer=2, type='conv', padding='same'))(X)
    X = BatchNormalization(axis=-1, name=conv_name.format(layer=2, type='bn'))(X)
    X = Activation('elu', name=conv_name.format(layer=2, type='elu'))(X)

    X = Conv3D(filters=f3, kernel_size=(1, 1, 1), strides=(1, 1, 1), name=conv_name.format(layer=3, type='conv', padding='valid'))(X)
    X = BatchNormalization(axis=-1, name=conv_name.format(layer=3, type='bn'))(X)
    
    X = Add()([X, X_shortcut])
    X = Activation('elu', name=conv_name.format(layer=3, type='elu'))(X)

    return X

def convolutional_block(X: tf.Tensor, level: int, block: int, filters: List[int], strides: Tuple[int, int, int]) -> tf.Tensor:

    conv_name  = f'conv{level}_{block}' + '_{layer}_{type}'
    f1, f2, f3 = filters
    X_shortcut = X

    X = Conv3D(filters=f1, kernel_size=(1, 1, 1), strides=strides, name=conv_name.format(layer=1, type='conv', padding='valid'))(X)
    X = BatchNormalization(axis=-1, name=conv_name.format(layer=1, type='bn'))(X)
    X = Activation('elu', name=conv_name.format(layer=1, type='elu'))(X)

    X = Conv3D(filters=f2, kernel_size=(3, 3, 3), strides=(1, 1, 1), name=conv_name.format(layer=2, type='conv', padding='same'))(X)
    X = BatchNormalization(axis=-1, name=conv_name.format(layer=2, type='bn'))(X)
    X = Activation('elu', name=conv_name.format(layer=2, type='elu'))(X) 

    X = Conv3D(filters=f3, kernel_size=(1, 1, 1), strides=(1, 1, 1), name=conv_name.format(layer=3, type='conv', padding='valid'))(X)
    X = BatchNormalization(axis=-1, name=conv_name.format(layer=3, type='bn'))(X)
    

    X_shortcut = Conv3D(filters=f3, kernel_size=(1, 1, 1), strides=strides, name=conv_name.format(layer='short', type='conv', padding='valid'))(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1, name=conv_name.format(layer='short', type='bn'))(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('elu', name=conv_name.format(layer=3, type='elu'))(X)

    return X

def resnet50_score(input_size: Tuple[int, int, int, int], model_name: str) -> Model:

    X_input = Input(input_size)

    X = ZeroPadding3D((3, 3, 3))(X_input)

    X = Conv3D(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2), name='conv1_1_1_conv', padding='valid')(X)
    X = BatchNormalization(name='conv1_1_1_bn')(X)
    X = Activation('elu')(X)

    X = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(X)

    X = convolutional_block(X, level=2, block=1, filters=[64, 64, 256], strides=(1, 1, 1))
    
    X = identity_block(X, level=2, block=2, filters=[64, 64, 256])
    X = identity_block(X, level=2, block=3, filters=[64, 64, 256])

    X = convolutional_block(X, level=3, block=1, filters=[128, 128, 512], strides=(2, 2, 2))
    
    X = identity_block(X, level=3, block=2, filters=[128, 128, 512])
    X = identity_block(X, level=3, block=3, filters=[128, 128, 512])
    X = identity_block(X, level=3, block=4, filters=[128, 128, 512])

    X = convolutional_block(X, level=4, block=1, filters=[256, 256, 1024], strides=(2, 2, 2))

    X = identity_block(X, level=4, block=2, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=3, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=4, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=5, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=6, filters=[256, 256, 1024])

    X = convolutional_block(X, level=5, block=1, filters=[512, 512, 2048], strides=(2, 2, 2))

    X = identity_block(X, level=5, block=2, filters=[512, 512, 2048])
    X = identity_block(X, level=5, block=3, filters=[512, 512, 2048])

    X = AveragePooling3D(pool_size=(2, 2, 2), name='avg_pool')(X)

    X = Flatten()(X)
    X = Dense(1, name='score_layer')(X)

    model = Model(inputs=X_input, outputs=X, name=model_name)

    return model

def resnet50_actor(input_size: Tuple[int, int, int, int], classes: int, model_name: str) -> Model:

    X_input = Input(input_size)

    X = ZeroPadding3D((3, 3, 3))(X_input)

    X = Conv3D(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2), name='conv1_1_1_conv', padding='valid')(X)
    X = BatchNormalization(name='conv1_1_1_bn')(X)
    X = Activation('elu')(X)

    X = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(X)

    X = convolutional_block(X, level=2, block=1, filters=[64, 64, 256], strides=(1, 1, 1))
    
    X = identity_block(X, level=2, block=2, filters=[64, 64, 256])
    X = identity_block(X, level=2, block=3, filters=[64, 64, 256])

    X = convolutional_block(X, level=3, block=1, filters=[128, 128, 512], strides=(2, 2, 2))
    
    X = identity_block(X, level=3, block=2, filters=[128, 128, 512])
    X = identity_block(X, level=3, block=3, filters=[128, 128, 512])
    X = identity_block(X, level=3, block=4, filters=[128, 128, 512])

    X = convolutional_block(X, level=4, block=1, filters=[256, 256, 1024], strides=(2, 2, 2))

    X = identity_block(X, level=4, block=2, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=3, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=4, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=5, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=6, filters=[256, 256, 1024])

    X = convolutional_block(X, level=5, block=1, filters=[512, 512, 2048], strides=(2, 2, 2))

    X = identity_block(X, level=5, block=2, filters=[512, 512, 2048])
    X = identity_block(X, level=5, block=3, filters=[512, 512, 2048])

    X = AveragePooling3D(pool_size=(2, 2, 2), name='avg_pool')(X)

    X = Flatten()(X)
    X = Dense(classes, activation='relu', name='fn_' + str(model_name))(X)

    model = Model(inputs=X_input, outputs=X, name=model_name)

    return model

def resnet50_actor_critic(input_size: Tuple[int, int, int, int], classes: int, model_name: str='actor_critic') -> Model:

    X_input = Input(input_size)

    X = ZeroPadding3D((3, 3, 3))(X_input)

    X = Conv3D(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2), name='conv1_1_1_conv', padding='valid')(X)
    X = BatchNormalization(name='conv1_1_1_bn')(X)
    X = Activation('elu')(X)

    X = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(X)

    X = convolutional_block(X, level=2, block=1, filters=[64, 64, 256], strides=(1, 1, 1))
    
    X = identity_block(X, level=2, block=2, filters=[64, 64, 256])
    X = identity_block(X, level=2, block=3, filters=[64, 64, 256])

    X = convolutional_block(X, level=3, block=1, filters=[128, 128, 512], strides=(2, 2, 2))
    
    X = identity_block(X, level=3, block=2, filters=[128, 128, 512])
    X = identity_block(X, level=3, block=3, filters=[128, 128, 512])
    X = identity_block(X, level=3, block=4, filters=[128, 128, 512])

    X = convolutional_block(X, level=4, block=1, filters=[256, 256, 1024], strides=(2, 2, 2))

    X = identity_block(X, level=4, block=2, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=3, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=4, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=5, filters=[256, 256, 1024])
    X = identity_block(X, level=4, block=6, filters=[256, 256, 1024])

    X = convolutional_block(X, level=5, block=1, filters=[512, 512, 2048], strides=(2, 2, 2))

    X = identity_block(X, level=5, block=2, filters=[512, 512, 2048])
    X = identity_block(X, level=5, block=3, filters=[512, 512, 2048])

    X = AveragePooling3D(pool_size=(2, 2, 2), name='avg_pool')(X)

    X = Flatten()(X)

    actor  = Dense(classes, activation='softmax', name='actor')(X)
    critic = Dense(1, name='critic')(X)

    model = Model(inputs=X_input, outputs=[actor, critic], name=model_name)

    return model

