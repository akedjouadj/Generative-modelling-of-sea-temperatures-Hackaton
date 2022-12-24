#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# G_\theta(Z) = np.max(0, \theta.Z)
############################################################################


#### ----------- Library---------------####
from tensorflow import keras 
from keras.models import Model, load_model 
from keras.layers import Input, Dense, Embedding, Reshape, Concatenate, Flatten, Dropout 
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU, LeakyReLU 
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.optimizers import Adam  
import pandas as pd
from tqdm import tqdm
import numpy as np 


########------ DATA LOADING -------########
'''
data_train = pd.read_csv("data/df_train.csv")
data_test = pd.read_csv("data/df_test.csv")
data = pd.concat([data_train,data_test], ignore_index=True)
noise = np.load("data/noise.npy")
positions = np.load("data/position.npy").reshape((2,6))

data_train = np.array(data_train.iloc[:,1:])
data_test = np.array(data_test.iloc[:,1:])

n_stations = 6
data_loader_train = np.zeros((n_stations, 1374, 7))
for i in range(n_stations):
    for j in range(1374):
        data_loader_train[i, j, :] = data_train[(j*7):((j+1)*7),i] 
'''

######------ CONDITIONNAL GAN MODEL-------#######
def generator(latent_dim, in_shape=(7,), n_stations=12): 
        
    # Label Inputs
    in_label = Input(shape=(1), name='Generator-Label-Input-Layer') # Input Layer
    lbls = Embedding(n_stations, 50, name='Generator-Label-Embedding-Layer')(in_label) # Embed label to vector
    
    # Scale up to time-serie dimensions
    n_nodes = in_shape[0] 
    lbls = Dense(n_nodes, name='Generator-Label-Dense-Layer')(lbls)
    lbls = Reshape((in_shape[0],), name='Generator-Label-Reshape-Layer')(lbls) # New shape

    # Generator Inputs (latent vector)
    in_latent = Input(shape=(latent_dim,), name='Generator-Latent-Input-Layer')
    
    # data generator foundation 
    n_nodes = 7 * 3 # number of nodes in the initial layer
    g = Dense(n_nodes, name='Generator-Foundation-Layer')(in_latent)
    g = LeakyReLU(alpha=0.2, name='Generator-Foundation-Layer-Activation-1')(g)
    g = Reshape((in_shape[0]*3,), name='Generator-Foundation-Layer-Reshape-1')(g)
    
    # Combine both inputs so it has two channels
    concat = Concatenate(name='Generator-Combine-Layer')([g, lbls])

    # Hidden Layer 1
    n_nodes = 7 * 2
    g = Dense(n_nodes, name = 'Generator-Hidden-Layer-1')(concat)
    g = LeakyReLU(alpha = 0.2, name='Generator-Hidden-Layer-Activation-1')(g)
    
    # Output Layer
    n_nodes = 7 * 1
    output_layer = Dense(n_nodes, name = 'Generator-Output-Layer')(g)
    output_layer = LeakyReLU(alpha = 0.2, name='Generator-Output-Layer-Activation')(output_layer)
    
    
    # Define model
    model = Model([in_latent, in_label], output_layer, name='Generator')
    return model

def discriminator(in_shape=(7,), n_stations = 12):
    
    # Label Inputs
    in_label = Input(shape=(1,), name='Discriminator-Label-Input-Layer') # Input Layer
    lbls = Embedding(n_stations, 50, name='Discriminator-Label-Embedding-Layer')(in_label) # Embed label to vector
    
    # Scale up to time-serie dimensions
    n_nodes = in_shape[0]
    lbls = Dense(n_nodes, name='Discriminator-Label-Dense-Layer')(lbls)
    lbls = Reshape((in_shape[0],), name='Discriminator-Label-Reshape-Layer')(lbls) # New shape

    # time-serie Inputs
    in_ts = Input(shape=in_shape, name='Discriminator-time-serie-Input-Layer')
    
    # Combine both inputs so it has two channels
    concat = Concatenate(name='Discriminator-Combine-Layer')([in_ts, lbls])

    # Hidden Layer 1
    n_nodes = 7*5
    h = Dense(n_nodes, name = 'Discriminator-Hidden-Layer-1')(concat)
    h = LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1')(h)
    
    # Hidden Layer 2
    n_nodes = 7*2
    h = Dense(n_nodes, name = 'Discriminator-Hidden-Layer-2')(h)
    h = LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2')(h)
    
    # Dropout and Output Layers
    h = Dropout(0.2, name='Discriminator-Hidden-Layer-Dropout')(h) # Randomly drop some connections for better generalization
   
    output_layer = Dense(1, activation='sigmoid', name='Discriminator-Output-Layer')(h) # Output Layer
    
    # Define model
    model = Model([in_ts, in_label], output_layer, name='Discriminator')
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model

def def_gan(generator, discriminator):
    
    # We don't want to train the weights of discriminator at this stage. Hence, make it not trainable
    discriminator.trainable = False
    
    # Get Generator inputs / outputs
    gen_latent, gen_label = generator.input # Latent and label inputs from the generator
    gen_output = generator.output # Generator output ts
    
    # Connect ts and label from the generator to use as input into the discriminator
    gan_output = discriminator([gen_output, gen_label])
    
    # Define GAN model
    model = Model([gen_latent, gen_label], gan_output, name="cDCGAN")
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model


######-------- FONCTIONS TO USE ---------########
def latent_vector(n, latent_dim):
    
    # Generate points in the latent space
    latent_input = np.random.randn(latent_dim * n).reshape(n, latent_dim)
    
    return latent_input

def fake_samples(generator, latent_dim, sta_labels_fake, batch_size=19):
    
    # Generate points in latent space
    latent_output = latent_vector(batch_size*6, latent_dim)

    # Predict outputs (i.e., generate fake samples)
    X = generator.predict([latent_output, sta_labels_fake]) 
    # pour chaque vecteur gaussien de la ligne i de latent output, le generateur génère en débruitant, un vecteur de taille 7 correspondant à la station de la ligne i de stations[stations_indx]     
    return X

def pred(gen_model, n_stations, labels):

    # Generate latent points
    latent_points = latent_vector(470*n_stations, 50)
   
    # Generate time series
    gen_time_series  = gen_model.predict([latent_points, labels], verbose = 0)
    
    X_pred = np.zeros((3290, n_stations))
    l = np.arange(0, 3291, 7)

    k = 0
    for i in range(n_stations): 
        for j in range(470):
            X_pred[l[j]:l[j+1], i] = gen_time_series[k,:]
            k = k+1
    
    return X_pred[0:3288, :]


#####------TRAINING FONCTION------######
'''
## version finale
def train(g_model, d_model, gan_model, stations= stations, 
             data = data_loader_train, latent_dim= 50, n_epochs=15, batch_size=19):
    # Number of batches to use per each epoch
    batch_per_epoch = int(data.shape[1] / batch_size) # = 229
    print(' batch_per_epoch: ',  batch_per_epoch)
    # Our batch to train the discriminator will consist of half real images and half fake (generated) images
    # half_batch_size = int(batch_size / 2)
    
    # We will manually enumare epochs 
    for i in tqdm(range(n_epochs)):
        
        # Enumerate batches over the training set
        for j in range(batch_per_epoch):
    
        # Discriminator training
            # Prep real samples
            x_real = data_loader_train[:, (j*batch_size):((j+1)*batch_size), :] # pour chaque station considéré, il a batch_size vecteurs de taille 7
            x_real = x_real.reshape((batch_size*6, 7)) # on concatène pour toutes les stations
            y_real = np.ones((batch_size*6, 1)) 
            sta_labels_real = np.array([stations[i] for i in range(6) for j in range(batch_size)])
            # bout de code à tester sur une matrice lambda truc lambda
            # Train discriminator with real samples
            # le dicriminateur prend un vecteur réel de taille 7 avec le label auquel il correspond
            # on a 9618 valeur de train pour chaque sta, ie 9618/7 = 1374 données à fournir au discriminateur. 
            # on peut les lui fournir un à un ou par batch_size à définir, idéalement 6
            discriminator_loss1, _ = d_model.train_on_batch([x_real, sta_labels_real], y_real) 
            
            # Prep fake (generated) samples
            sta_labels_fake = sta_labels_real
            x_fake = fake_samples(g_model, latent_dim, sta_labels_fake, batch_size)
            y_fake = np.zeros((batch_size*6, 1))
            # Train discriminator with fake samples
            discriminator_loss2, _ = d_model.train_on_batch([x_fake, sta_labels_fake], y_fake)


        # Generator training
            # Get values from the latent space to be used as inputs for the generator
            latent_input = latent_vector(batch_size*6, latent_dim) # variable aléatoire gaussienne de la taille du batch, à extraire du noise
            # While we are generating fake samples, 
            # we want GAN generator model to create examples that resemble the real ones,
            # hence we want to pass labels corresponding to real samples, i.e. y=1, not 0.
            y_gan = np.ones((batch_size*6, 1))

            # Train the generator via a composite GAN model
            generator_loss = gan_model.train_on_batch([latent_input, sta_labels_fake], y_gan)
        
        # Summarize training progress and loss
        print('Epoch: %d, D_Loss_Real=%.3f, D_Loss_Fake=%.3f Gen_Loss=%.3f' % 
                (i+1, discriminator_loss1, discriminator_loss2, generator_loss))
                #show_fakes(g_model, latent_dim)
'''
       
# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise, position):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """

    latent_dim=50 # Our latent space has dz = 50 dimensions.
    
    # Instantiate generator
    gen_model = generator(latent_dim)
    
    # Instantiate discriminator
    dis_model = discriminator()
    
    # Instantiate GAN
    gan_model = def_gan(gen_model, dis_model)
    
    # if you want to retrain your model
    '''
    train(gen_model, dis_model, gan_model, stations = [1,3,5,7,9,11])
    gen_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    gen_model.save('parameters/cgan_generator_lambda.h5')
    '''

    gen_model = load_model('parameters/cgan_generator_final.h5')
    
    stations_indx_test = [0,2,4,6,8,10] # positions numbers

    labels = np.array([i for i in stations_indx_test for j in range(470)])
    
    x = pred(gen_model, 6, labels)
    
    return x  


