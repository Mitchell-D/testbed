"""
Fully-connected neural network (with no state passed between timesteps)
supporting explicit state accumulation between timesteps
"""
import tensorflow as tf

class AccFNNCell(tf.keras.layers.Layer):
    """
    Layer abstracting a stack of LSTM cells, which explicitly discretely
    integrates the output state and cycles it back into subsequent steps
    as an input
    """
    def __init__(self, ann_layer_units:list, pred_units:int, dropout=0.,
            ann_kwargs={}, l2_penalty=0., name="", **kwargs):
        """
        :@param ann_layer_units: List of node counts corresponding to each
            fully-connected cell layer within this cell.
        :@param pred_units: Number of predicted units (for final non-RNN layer)
        :@param ann_kwargs: keyword arguments passed to internal layer inits
        :@param name:
        """
        super().__init__(**kwargs)
        self._ann_units = ann_layer_units
        self._ann_kwargs = ann_kwargs
        self.units = pred_units
        self._name = name
        self._dropout = dropout
        self._l2 = l2_penalty

        ## If requested, initialize dropout layers for between RNN elements
        self._dropout_layers = None
        if self._dropout > 0:
            self._dropout_layers = [
                    tf.keras.layers.Dropout(self._dropout)
                    for i in range(len(self._ann_units))
                    ]

        ## Initialize dense layers for each of the requested layers
        self._ann_layers = [
            tf.keras.layers.Dense(units=units, **self._ann_kwargs)
            for units in self._ann_units
            ]

        self._ann_out_layer = tf.keras.layers.Dense(
                self.units, activation="linear")
        self.state_size = self.units
        self.output_size = self.units

    def call(self, inputs, states, training=False):
        """
        :@param inputs: Tensor input for this cell
        :@param states: tuple of tensor states corresponding to the
            (prev_accumulated, (output, context)) states of the previous cell
        :@param training: determines dropout behavior
        """
        states = states[0]
        prev_layer = tf.concat((inputs, states), axis=-1, name="acc_concat")
        for i,F in enumerate(self._ann_layers):
            ## project the previous state and new inputs to the hidden domain
            prev_layer = F(prev_layer, training=training)
            ## Add magnitude regularization
            if self._l2 > 0.:
                self.add_loss(self._l2 * tf.math.reduce_sum(prev_layer**2))
            ## Apply dropout
            if self._dropout > 0.:
                prev_layer = self._dropout_layers[i](
                        prev_layer, training=training)

        new_res = self._ann_out_layer(prev_layer, training=training)
        ## accumulate from the previous state
        new_acc = states + new_res
        return (new_res, new_acc)

    def get_config(self):
        """
        Returns dict of keyword  parameters for __init__ which enable the
        layer to be integrated into a functionally defined model.
        """
        config = {
                "ann_layer_units": self._ann_units,
                "pred_units": self.units,
                "dropout": self._dropout,
                "ann_kwargs": self._ann_kwargs,
                "name":self._name,
                }
        base_config = super().get_config()
        return {**base_config, **config}

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Method returning a 2-tuple (init_acc_state,init_ann_states) where
        init_acc_state is a zero vector representing the initial accumulated
        prediction state, and init_ann_states is a list of zeroed tensors
        for each of the RNN layers.
        """
        return tf.zeros(self.units)

if __name__=="__main__":
    pass
