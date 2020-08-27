# This is the code which achieved better diagnosablity compared with JACC_CV model (https://imaging.onlinejacc.org/content/13/2_Part_1/374.abstract), to detect cardiac wall motion abnormality.

input_img = Input(shape = shape)

def conv_block(x):
    x = Conv3D(filters = 32, kernel_size = (3, 3, 3), strides = (1, 1, 1),
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.05))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.02)(x)
    return x


def Incep_block(x):
    x = Conv3D(filters = 32, kernel_size = (3, 3, 3), strides = (1, 1, 1),
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.05))(x)
    x = LeakyReLU(alpha = 0.02)(x)
    x = Dropout(0.5)(x)
    x = Conv3D(filters = 32, kernel_size = (3, 3, 3), strides = (1, 1, 1),
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.05))(x)

    a = Conv3D(filters = 4, kernel_size = (3, 1, 1), strides = (1, 1, 1),
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.05))(x)
    b = Conv3D(filters = 4, kernel_size = (1, 5, 1), strides = (1, 1, 1),
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.05))(x)
    c = Conv3D(filters = 4, kernel_size = (1, 1, 5), strides = (1, 1, 1),
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.05))(x)

    x = Concatenate(axis = -1)([x, a, b, c])
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.02)(x)
    return x


def Res_block(x):
    y = x
    x = Conv3D(filters = 32, kernel_size = (3, 3, 3), strides = (1, 1, 1),
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.05))(x)
    x = Conv3D(filters = 32, kernel_size = (3, 3, 3), strides = (1, 1, 1),
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.05))(x)
    x = Concatenate(axis = -1)([x, y])
    x = LeakyReLU(alpha = 0.02)(x)
    x = BatchNormalization()(x)
    return x


x = conv_block(input_img)
x = MaxPooling3D((2, 4, 4), padding = 'valid')(x)
x = Dropout(0.5)(x)

x = Incep_block(x)
x = AveragePooling3D((1, 2, 2), padding = 'valid')(x)
x = Dropout(0.5)(x)

x = Res_block(x)
x = Dropout(0.5)(x)
x = Res_block(x)
x = Dropout(0.5)(x)
x = Res_block(x)
x = GlobalAveragePooling3D()(x)#SMaxPooling3D((2, 4, 4), padding = 'valid')(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.02)(x)
x = Dropout(0.5)(x)
x = Dense(num_class)(x)
answer = Activation("softmax")(x)

model = Model(input_img, answer)
model.summary()
