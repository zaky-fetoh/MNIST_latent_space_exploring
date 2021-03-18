import keras.layers as layers
import keras.models as models
import keras.optimizers as opt
import tensorflow as tf



def conv_block(X, units, kernel=(3, 3), pad='same', pooling=True, drop=False,
               Activation=layers.LeakyReLU()):
    X = layers.Conv2D(units, kernel, padding=pad)(X)
    X = layers.BatchNormalization()(X)
    if pooling:
        X = layers.MaxPool2D()(X)
    X = Activation(X)
    if drop:
        X = layers.GaussianDropout(.3)(X)
    return X


def get_encoder(inp_layer=layers.Input((28, 28, 1)), latent_dims=2,
                conv_block_def=conv_block,
                layers_size=[64, 32, 16, 8, 4],
                kernel=[3] * 5, pading=['same'] * 5,
                is_pooling=[False, True, True, True, True],
                droping=[False, True, False, False, False]):
    X = inp_layer
    for i in range(len(layers_size)):
        X = conv_block_def(X, units=layers_size[i],
                           kernel=kernel[i], pad=pading[i],
                           pooling=is_pooling[i], drop=droping[i])
    X = layers.Flatten()(X)
    X = layers.Dense(latent_dims)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    return models.Model(inp_layer, X)


def deconv_block(X, units, kernel=(3, 3), pad='same', upsampling=True, drop=False,
                 Activation=layers.LeakyReLU()):
    if upsampling:
        X = layers.UpSampling2D()(X)
    X = layers.Conv2D(units, kernel, padding=pad)(X)
    X = layers.BatchNormalization()(X)
    X = Activation(X)
    if drop:
        X = layers.GaussianDropout(.3)(X)
    return X


def get_decoder(inp_layer=layers.Input((2,)),
                deconv_block_def=deconv_block,
                layers_size=[32, 16, 8, 4, 1],
                kernel=[3] * 5, pad=['valid', 'valid', 'same', 'same', 'same'],
                upsampling=[True, True, False, True, False],
                droping=[False, False, True, False, False]):
    act = layers.Activation('tanh')
    X = inp_layer
    X = layers.Dense(5 * 5 * 64)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Reshape((5, 5, 64))(X)
    for i in range(len(layers_size)):
        X = deconv_block_def(X, units=layers_size[i],
                             kernel=kernel[i], pad=pad[i],
                             upsampling=upsampling[i], drop=droping[i],
                             Activation=act if i + 1 == len(layers_size) else layers.LeakyReLU());
    return models.Model(inp_layer, X)


def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def compile_auto_enco(enco=get_encoder, deco=get_decoder):
    ae =models.Sequential([enco(), deco()])
    ae.compile(optimizer=opt.adam(),loss=ssim_loss, metrics=[ssim_loss,'acc'])
    return ae

def get_trained_model(enco=get_encoder, deco=get_decoder,
                      waight_path = 'model_waights/waight0020.h5'):
    ae = models.Sequential([enco(), deco()])
    ae.load_weights(waight_path)
    return ae



if __name__ == '__main__':
    m = compile_auto_enco()
