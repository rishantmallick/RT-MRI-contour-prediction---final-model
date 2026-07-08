# RT-MRI-contour-prediction---final-model
Preprocessing Techniques
Interpolation used : It is called Arc Length Interpolation. Here we first compute the distance between two consecutive points . Then we set the first point (x1,y1) of the given contour as the reference point and take its distance from it to be zero. Then we compute the cumulative sum of the distances between the two contours. Then divide the cumulative distances by maximum cumulative distance. np.unique return unique values and their index and compute pts and arc and then use linear interpolation for each fx and fy, stack it and return.
Code : 
def resample(pts,n=100):
  try:
    pts = np.array(pts,dtype=np.float32)   
    if pts.ndim != 2:
      return None
    if pts.shape[0] == 2:
      pts = pts.T
    if pts.shape[1] != 2:
      return None
    if len(pts) < 2:
      return None
    diffs = np.diff(pts,axis=0)
    arc = np.concatenate([[0],np.cumsum(np.sqrt((diffs**2).sum(1)))])
    if arc[-1]==0:
      return None
    arc /=arc[-1]
    _,idx = np.unique(arc,return_index=True)
    arc = arc[idx]
    pts = pts[idx]
    if len(arc) < 2:
      return None
    t = np.linspace(0, 1, n)
    fx = interp1d(arc,pts[:,0],kind='linear',fill_value='extrapolate')
    fy = interp1d(arc,pts[:,1],kind='linear',fill_value='extrapolate')
    pts = np.stack([fx(t),fy(t)], axis=1)
    return pts.astype(np.float32)

Normalization used :-> I concatenated future pts and past pts and computed their mean and std. Normalize future pts, past pts, ground truth all of them using this mean and std.
Feature Engineering :-> I augmented the dataset by slightly rotating it and translating and having different rotation and translation for past frame and future as normally different frames have different augmentation.

Post Processing Techniques
Deenormalization Method :-> denormalized the predicted contour using mean and std calculated by concatenating future pts and past pts.
Residual Addition :-> before denormalization, we added pred_residual to linear ((past_pts + future_pts)/2)   without any residual scaling while training and while testing both.
Coordinate Reconstruction :-> After Model Prediction, I added the  predicted residual to linear without any residual scaling and then denormalized it.
Any operations performed after model prediction:-> Residual Addition and Denormalization
Executions
Required Packet Installation:->
!pip install torch
!pip install scipy numpy pandas fastdtw
Steps to run the inference/test script:->
Put it in google colab and then run it . we can download it in colab and just in case if the weights don’t work the access is already given in google drive 
Description of the input and output files:->
Both are Mat files
Model Architecture and parameters:->
Contour3
For Contour3 , I will use an architecture which has cnn encoder which will help in extracting the local features and help us to understand the dynamics of contour3 and then using adaptive pooling for reducing the dimension and for extracting maximum features and then mlp decoder help us to calculate final contour coordinates.
2 layer cnn + Adaptive Pooling + 2layer MLP
Hyperparameters:
Residual scaling=1
    'Contour3': {
        'lr': 2.5e-5,
        'weight_decay': 1e-5,
        'smooth_weight': 0.01

    }
EPOCHS = 100
BATCH_SIZE = 32
        ‘optimizer’  : torch.optim.Adam(
      model.parameters(),
      lr = hp['lr'],
      weight_decay = hp['weight_decay']
  )
Used early stopping with a patience of 40
Loss function = huber loss and smooth loss
Contour2
Used a 4 layer mlp encoder, computed mu and sigma and then reparametrize  it to get the latent(features)  and then passed the latent through 3 layer mlp decoder  with latent dimension = 64

Hyperparameters:
Residual scaling=1
HYPERPARAMS = {

    'Contour2': {
        'lr': 5e-4,
        'weight_decay': 1e-6,
        'smooth_weight': 0.02
    }
}

Loss function = kl loss and reconstruction loss(here I used huber loss with delta = 0.1)  and smooth loss. Kl loss and reconstruction loss was used to compute vae loss and then vae loss was considered as regression loss and was combined with smooth loss to calculate the total loss
Used early stopping with a patience of 40
EPOCHS = 100
BATCH_SIZE = 32
        ‘optimizer’  : torch.optim.Adam(
      model.parameters(),
      lr = hp['lr'],
      weight_decay = hp['weight_decay']
  )
Contour1
Model:
Used a 3 layer mlp encoder, computed mu and sigma and then reparametrize  it to get the latent(features)  and then passed the latent through 2 layer mlp decoder with latent dimension = 64
Hyperparameters:
Loss function = kl loss and reconstruction loss(here I used huber loss with delta = 0.1)  and smooth loss. Kl loss and reconstruction loss was used to compute vae loss and then vae loss was considered as regression loss and was combined with smooth loss to calculate the total loss Used early stopping with a patience of 40
EPOCHS = 100
HYPERPARAMS = {
    'Contour1': {
        'lr': 2e-4,
        'weight_decay': 1e-5,
        'smooth_weight': 0.01
    }

}

BATCH_SIZE = 32
        ‘optimizer’  : torch.optim.Adam(
      model.parameters(),
      lr = hp['lr'],
      weight_decay = hp['weight_decay']
  )
Residual scaling=1
Model Description for both Contour1 and contour2
Variational Autoencoder (VAE) for Contour Prediction
The proposed Variational Autoencoder (VAE) predicts the residual between the linearly interpolated contour and the ground-truth middle contour. The architecture consists of three main components: an encoder, a latent representation generated using the reparameterization trick, and a decoder.
Encoder
The encoder receives the concatenated past and future contours as input. Each contour contains 100 points with two coordinates, resulting in a 200-dimensional representation. Concatenating the two contours produces a 400-dimensional input vector.
The input is processed through three fully connected layers that progressively reduce the feature dimension:
self.encoder = nn.Sequential(
    nn.Linear(400, 256),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(128, 64),
    nn.ReLU()
)
These layers extract increasingly compact representations of the articulatory movement between the past and future contours. The final 64-dimensional feature vector is not used directly as the latent representation. Instead, two separate fully connected layers estimate the parameters of a Gaussian distribution:
self.fc_mu = nn.Linear(64, latent_dim)
self.fc_logvar = nn.Linear(64, latent_dim)
The first layer predicts the latent mean ((\mu)), while the second predicts the logarithm of the latent variance ((\log\sigma^2)). Together, these define the latent probability distribution learned by the encoder.
Reparameterization
Since sampling from a Gaussian distribution is not differentiable, the reparameterization trick is employed.
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std
The logarithmic variance is first converted into the standard deviation. A random noise vector is then sampled from a standard normal distribution having the same dimensions as the standard deviation. The latent representation is finally obtained by shifting and scaling the random noise using the predicted mean and standard deviation. Because the randomness is isolated in the sampled noise vector, gradients can propagate through both the mean and variance prediction layers during backpropagation.
Decoder
The decoder reconstructs the residual contour from the sampled latent representation. It consists of a single fully connected layer:
self.decoder = nn.Sequential(
    nn.Linear(latent_dim, 200)
)
The decoder predicts only the residual between the linearly interpolated contour and the actual middle contour. The final contour prediction is obtained by
predicted_contour = linear_interpolation + predicted_residual
Predicting only the residual significantly reduces the learning complexity because the linearly interpolated contour already provides a good approximation of the target.
Loss Function
The model is trained using a weighted combination of reconstruction loss and KL divergence.
The reconstruction loss is computed as
reconstruction_loss = 10 * huber_loss(prediction, target)
where the Huber loss measures the difference between the predicted contour and the ground-truth contour. A larger weight is assigned to this term to prioritise reconstruction accuracy.
The KL divergence regularises the latent space and is computed as
kl_loss = -0.5 * torch.mean(
    1 + logvar - mu.pow(2) - logvar.exp()
)
The total loss is therefore
loss = reconstruction_loss + kl_weight * kl_loss
To avoid excessive regularisation during the initial stages of training, the KL weight is gradually increased over the first twenty epochs (KL annealing) until it reaches its final value of 0.005.
Inference
During testing, stochastic sampling is disabled and the decoder receives only the latent mean:
z = mu
prediction = decoder(z)
Using the latent mean eliminates randomness during inference and produces deterministic and reproducible contour predictions. The predicted residual is finally added to the linearly interpolated contour to obtain the estimated intermediate articulatory contour.

