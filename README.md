# MNIST_latent_space_exploring
we project the MNIST dataset to 2D latent space using convolutional autoencoder
## convolutional autoencoder Model architecture
encoder architecture as follow 
![fig0](https://github.com/zaky-fetoh/MNIST_latent_space_exploring/blob/main/Resulting_figs/enco.png) 

decoder architecture as follow 
![fig1](https://github.com/zaky-fetoh/MNIST_latent_space_exploring/blob/main/Resulting_figs/decoder.png) 
 
## Training 
we trained the ae model using adam optimizer and the SSIM as loss function 


![fig2](https://wikimedia.org/api/rest_v1/media/math/render/svg/63349f3ee17e396915f6c25221ae488c3bb54b66)

## Results 
traversing latent space video
[link](https://drive.google.com/file/d/1RzooyS3OhfFtioWa0Os7E2yhLXMz9w5A/view)

ploting of the projected Training data 
![fig3](https://github.com/zaky-fetoh/MNIST_latent_space_exploring/blob/main/Resulting_figs/mnist_training_dtlatent_space.png)

ploting of the projected testset
![fig4](https://github.com/zaky-fetoh/MNIST_latent_space_exploring/blob/main/Resulting_figs/mnist_testing_set_latent_space.png)

entropy of the latent space
![fig5](https://github.com/zaky-fetoh/MNIST_latent_space_exploring/blob/main/Resulting_figs/mnist_trainingset_latent_space_entropy.png)

## END
