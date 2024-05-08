from keras.preprocessing.image import ImageDataGenerator
from model import conv_block, encoder_block, decoder_block
from loss_function import DiceLoss
from tensorflow.keras.models import load_model


folder = './processed_data/'
X_train    = np.load(folder+'Train_img.npy')
X_test    = np.load(folder+'Test_img.npy')
y_train    = np.load(folder+'Train_mask.npy')
y_test    = np.load(folder+'Test_mask.npy')

print('Dataset loaded')


#Data Augmentation
seed=24
img_data_gen_args = dict(
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.2,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True

                     )

mask_data_gen_args = dict(
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.2,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True) #Binarize the output again.

image_data_generator = ImageDataGenerator(**img_data_gen_args)
#image_data_generator.fit(X_train, augment=True, seed=seed)

batch_size= 32

image_generator = image_data_generator.flow(X_train, seed=seed, batch_size=batch_size)
valid_img_generator = image_data_generator.flow(X_test, seed=seed, batch_size=batch_size) #Default batch size 32, if not specified here

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
#mask_data_generator.fit(y_train, augment=True, seed=seed)
mask_generator = mask_data_generator.flow(y_train, seed=seed, batch_size=batch_size)
valid_mask_generator = mask_data_generator.flow(y_test, seed=seed, batch_size=batch_size)  #Default batch size 32, if not specified here

def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)



my_generator = my_image_mask_generator(image_generator, mask_generator)

validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)


#Visualizing Augmented Images
x = image_generator.next()
y = mask_generator.next()
for i in range(0,1):
    image = x[i]
    mask = y[i]
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.show()



#build model
def build_model(input_shape):
    input_layer = Input(input_shape)

    s1, p1 = encoder_block(input_layer, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    output_layer = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(input_layer, output_layer, name="U-Net")
    return model

model = build_model(input_shape=(128, 128, 1))
model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=["accuracy"])
tf.keras.utils.plot_model(model, show_shapes=True)
model.summary()


steps_per_epoch = 3*(len(X_train))//batch_size

history = model.fit(my_generator,
                    epochs = 100,
                    validation_data = validation_datagen,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=steps_per_epoch)


model.save('model.h5')

#Further trainng model for better results with Dice Coeffiecient as loss function
# When loading the model, specify the custom_objects dictionary with 'DiceLoss'
model = load_model('model.h5', custom_objects={'DiceLoss': DiceLoss})


learning_rate=0.0001
steps_per_epoch = 3*(len(X_train))//batch_size


model.compile(loss=[DiceLoss], optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"])

history = model.fit(my_generator, epochs = 50, validation_data = validation_datagen, steps_per_epoch=steps_per_epoch,validation_steps=steps_per_epoch)
model.save('model.h5')