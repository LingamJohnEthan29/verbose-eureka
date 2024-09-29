###IMPORTING LIBRARIES
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import tensorflow_addons as tfa
path = 'C:/Users/ethan/PycharmProjects/GANS NETWORK/selfie2anime/'

faces = glob.glob(path + '/trainA/*.jpg')
animes = glob.glob(path + "/trainB/*.jpg")
faces_test = glob.glob(path + '/testA/*.jpg')
animes_test = glob.glob(path + "/testB/*.jpg")
#CHECKING LENGTH OF DATSET
#print(len(faces))
#print(len(animes))
#print(len(faces_test))
#print(len(animes_test))
import cv2
#for file in faces[:10]:
    #img = cv2.imread(file)
    #print (img.shape)

####DISPLAY FEW SAMPLES
#print ("Human Faces")
#for k in range(2):
 #   plt.figure(figsize=(13, 13))
  #  for j in range(9):
   #     file = np.random.choice(faces)
     #   img = cv2.imread(file)
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #    img = cv2.resize(img, (128,128))
    #    plt.subplot(990 + 1 + j)
    #    plt.imshow(img)
    #    plt.axis('off')
        #plt.title(trainY[i])
    #plt.show()

#print ("-"*80)
#print ("Anime Faces")
#for k in range(2):
#    plt.figure(figsize=(13, 13))
#    for j in range(9):
#        file = np.random.choice(animes)
#        img = cv2.imread(file)
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        img = cv2.resize(img, (128,128))
 #       plt.subplot(990 + 1 + j)
  #      plt.imshow(img)
   #     plt.axis('off')
    #    #plt.title(trainY[i])
    #plt.show()
###DEFINING GENERATOR MODEL
#####DEFINING ENCODER ADN DECODER LEVELS

def encoder_layer(input_layer, filters, bn=True):
    x = tf.keras.layers.Conv2D(filters, kernel_size=(4,4), strides=(2,2), padding='same')(input_layer)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    if bn:
        #x = tensorflow.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tfa.layers.BatchNormalization()(x)
    return x

def decoder_layer(input_layer, skip_input, filters):
    #x = tensorflow.keras.layers.UpSampling2D(size=2)(input_layer)
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(4,4), strides=(2,2), padding='same')(input_layer)
    x = tf.keras.layers.Activation('relu')(x)
    #x = tensorflow.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tfa.layers.BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, skip_input])
    return x

def make_generator():
    source_image = tf.keras.layers.Input(shape=(128, 128, 3))
    target_style = tf.keras.layers.Input(shape=(16, 16, 512))

    e1 = encoder_layer(source_image, 64, bn=False)
    e2 = encoder_layer(e1, 128)
    e3 = encoder_layer(e2, 256)
    # e4 = encoder_layer(e3, 256)
    e5 = encoder_layer(e3, 512)
    e6 = encoder_layer(e5, 512)
    e7 = encoder_layer(e6, 512)

    bottle_neck = tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same')(e7)
    b = tf.keras.layers.Activation('relu')(bottle_neck)

    d1 = decoder_layer(b, e7, 512)
    d2 = decoder_layer(d1, e6, 512)
    d3 = decoder_layer(d2, e5, 512)
    # d4 = decoder_layer(d3, e4, 256)
    d5 = decoder_layer(d3, e3, 256)
    d5 = tf.keras.layers.Concatenate()([d5, target_style])
    d6 = decoder_layer(d5, e2, 128)
    d7 = decoder_layer(d6, e1, 64)

    decoded = tf.keras.layers.Conv2DTranspose(3, kernel_size=(4,4), strides=(2,2), padding='same')(d7)
    translated_image = tf.keras.layers.Activation('tanh')(decoded)
    return source_image, target_style, translated_image

source_image, target_style, translated_image = make_generator()
generator_network =tf.keras.models.Model(inputs=[source_image, target_style], outputs=translated_image)
#print (generator_network.summary())

###Define Discriminator Network
def my_conv_layer(input_layer, filters, bn=True):
    x = tf.keras.layers.Conv2D(filters, kernel_size=(4,4), strides=(2,2), padding='same')(input_layer)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    if bn:
        #x = tensorflow.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tfa.layers.InstanceNormalization()(x)
    return x
def make_discriminator():
    target_image_input = tf.keras.layers.Input(shape=(128, 128, 3))

    x = my_conv_layer(target_image_input, 64, bn=False)
    x = my_conv_layer(x, 128)
    x = my_conv_layer(x, 256)
    # x = my_conv_layer(x, 512)
    x = my_conv_layer(x, 512)

    patch_features = tf.keras.layers.Conv2D(1, kernel_size=(4,4), strides=(1,1), padding='same')(x)
    return target_image_input, patch_features


target_image_input, patch_features = make_discriminator()
discriminator_network = tf.keras.models.Model(inputs=target_image_input, outputs=patch_features)
#print (discriminator_network.summary())
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5)
discriminator_network.compile(loss='mse', optimizer=adam_optimizer,metrics=['accuracy'])

###Load and Extract VGG features
image_input = tf.keras.layers.Input(shape=(128, 128, 3))

pre_trained_vgg = tf.keras.applications.vgg19.VGG19(weights='imagenet', input_shape=(128, 128, 3), include_top=False)
pre_trained_vgg_model = tf.keras.models.Model(inputs=pre_trained_vgg.input, outputs=pre_trained_vgg.get_layer('block4_conv4').output)

pre_trained_image_features = pre_trained_vgg_model(image_input)

custom_vgg = tf.keras.models.Model(inputs=image_input, outputs=pre_trained_image_features)
#print (custom_vgg.summary())

###Define Customized-Face2Anime-GAN
source_image = tf.keras.layers.Input(shape=(128, 128, 3))
target_features = tf.keras.layers.Input(shape=(16, 16, 512))

# Domain Transfer
custom_vgg.trainable=False
fake_anime = generator_network([source_image, target_features])

discriminator_network.trainable=False

# Tell Real vs Fake
real_vs_fake = discriminator_network(fake_anime)

face2anime_gan = tf.keras.models.Model(inputs =[source_image, target_features], outputs = [real_vs_fake, fake_anime, fake_anime])
#print(face2anime_gan.summary())

###Custom Content Loss (vgg features Loss)
def custom_content_loss(y_true, y_pred):
    custom_vgg.trainable=False
    y_true_features = custom_vgg(y_true)
    y_pred_features = custom_vgg(y_pred)
    content_loss = tf.keras.losses.mean_absolute_error(y_true_features, y_pred_features)
    return content_loss

def custom_content_loss2(y_true, y_pred):
    custom_vgg.trainable=False
    y_true_features = y_true
    y_pred_features = custom_vgg(y_pred)
    content_loss = tf.keras.losses.mean_absolute_error(y_true_features, y_pred_features)
    return content_loss

###COMPILING MODELS
face2anime_gan.compile(loss=['mse',custom_content_loss, custom_content_loss],\
                       optimizer=adam_optimizer, loss_weights=[1,1,0.1])
###Define Data Generators
def faces_to_animes(faces, styles, generator_network):
    styles = custom_vgg(styles)
    generated_samples = generator_network.predict_on_batch([faces, styles])
    return generated_samples

def get_training_samples(batch_size):
    random_files = np.random.choice(faces, size=batch_size)
    images = []
    for file in random_files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        images.append((img-127.5)/127.5)
    face_images = np.array(images)

    random_files = np.random.choice(animes, size=batch_size)
    images = []
    for file in random_files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        images.append((img-127.5)/127.5)
    anime_images = np.array(images)
    return face_images, anime_images

def show_generator_results(generator_network):
    images = []
    styles = []
    for j in range(7):
        file = np.random.choice(faces_test)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        images.append(img)
        file = np.random.choice(animes_test)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        styles.append(img)

    print('Human Face Images')
    plt.figure(figsize=(13, 13))
    for j, img in enumerate(images):
        plt.subplot(770 + 1 + j)
        plt.imshow(img)
        plt.axis('off')
        #plt.title(trainY[i])
    plt.show()

    # print ('Style Images')
    # plt.figure(figsize=(13, 13))
    # for j, img in enumerate(styles):
    #     plt.subplot(770 + 1 + j)
    #     plt.imshow(img)
    #     plt.axis('off')
    #     #plt.title(trainY[i])
    # plt.show()

    print('Customized Anime Version')
    plt.figure(figsize=(13, 13))
    for j, img in enumerate(images):
        img = (img-127.5)/127.5
        style = (styles[j]-127.5)/127.5
        output = faces_to_animes(np.array([img]), np.array([style]), generator_network)[0]
        output = (output+1.0)/2.0
        plt.subplot(770 + 1 + j)
        plt.imshow(output)
        plt.axis('off')
        #plt.title(trainY[i])
    plt.show()

###Training Face2Anime-GAN
epochs = 5 #500
batch_size = 1
steps = 100 #3400

for i in range(0,epochs):
    for j in range(steps):
        if j%200 == 0:
            show_generator_results(generator_network)
            generator_network.save(path+"weights/model_"+str(i)+"_"+str(j))
        human_faces, anime_faces = get_training_samples(batch_size)

        fake_patch = np.zeroes((batch_size,8,8,1))
        real_patch = np.ones((batch_size,8,8,1))

        custom_vgg.trainable=False
        styles = custom_vgg(anime_faces)
        fake_anime_faces = generator_network([human_faces, styles])

        #Updating Discriminator weights
        discriminator_network.trainablee=True
        loss_d_real = discriminator_network.train_on_batch(anime_faces, real_patch)
        loss_d_fake = discriminator_network.train_on_batch(fake_anime_faces, fake_patch)

        loss_d = np.add(loss_d_real, loss_d_fake)/2.0

        # Make the Discriminator belive that these are real samples and calculate loss to train the generator
        discriminator_network.trainable=False
        custom_vgg.trainable=False
        y_true_features1 = custom_vgg(human_faces)
        y_true_features2 = custom_vgg(anime_faces)

        avg_features = np.add(y_true_features1, y_true_features2)/2.0

        # Updating Generator weights
        loss_g = face2anime_gan.train_on_batch([human_faces, styles],[real_patch, human_faces, anime_faces])

        if j%100 == 0:
            print ("Epoch:%.0f, Step:%.0f, D-Loss:%.3f, D-Acc:%.3f, G-Loss:%.3f"%(i,j,loss_d[0],loss_d[1]*100,loss_d[1]*100))
            loss_g[0]

