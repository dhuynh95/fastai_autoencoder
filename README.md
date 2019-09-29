# fastai_autoencoder

Fastai autoencoder is a library around Fastai to make the training of autoencoders easier.

To create an autoencoder learner you need to split your model in three steps, an encoder, a bottleneck, and a decoder, then use AutoEncoderLearner.

To train VAE, a Fastai hook called VAEHook can be used.
