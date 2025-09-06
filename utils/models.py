# Sourced from https://github.com/fwillett/speechBCI/blob/main/NeuralDecoder/neuralDecoder/models.py

import tensorflow as tf

class GRU(tf.keras.Model):
    def __init__(self,
                    units,
                    weightReg,
                    nClasses,
                    dropout=0.0,
                    nLayers=2,
                    patch_size=0,
                    patch_stride=0):
        super(GRU,self).__init__()

        weightReg = tf.keras.regularizers.L2(weightReg)
        actReg = None
        recurrent_init = tf.keras.initializers.Orthogonal()
        kernel_init = tf.keras.initializers.glorot_uniform()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.initStates = tf.Variable(initial_value=kernel_init(shape=(1, units)))

        self.rnnLayers = []
        for _ in range(nLayers):
            rnn = tf.keras.layers.GRU(units,
                                      return_sequences=True,
                                      return_state=True,
                                      kernel_regularizer=weightReg,
                                      activity_regularizer=actReg,
                                      recurrent_initializer=recurrent_init,
                                      kernel_initializer=kernel_init,
                                      dropout=dropout)
            self.rnnLayers.append(rnn)

        self.dense = tf.keras.layers.Dense(nClasses)
    

    def call(self, x, states=None, training=False, returnState=False):
        batchSize = tf.shape(x)[0]

        if self.patch_size>0:
            x = tf.image.extract_patches(x[:, None, :, :],
                                         sizes=[1, 1, self.patch_size, 1],
                                         strides=[1, 1, self.patch_stride, 1],
                                         rates=[1, 1, 1, 1],
                                         padding='VALID')
            x = tf.squeeze(x, axis=1)

        if states is None:
            states = []
            states.append(tf.tile(self.initStates, [batchSize, 1]))
            states.extend([None] * (len(self.rnnLayers) - 1))

        new_states = []
        for i, rnn in enumerate(self.rnnLayers):
            x, s = rnn(x, training=training, initial_state=states[i])
            new_states.append(s)

        x = self.dense(x, training=training)

        if returnState:
            return x, new_states
        else:
            return x