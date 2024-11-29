""" Basic RNN supporting explicit state accumulation between cells """
import tensorflow as tf

class AccRNNCell(tf.keras.layers.Layer):
    """
    Layer abstracting a stack of LSTM cells, which explicitly discretely
    integrates the output state and cycles it back into subsequent steps
    as an input
    """
    def __init__(self, ann_layer_units:list, pred_units:int,
            l2_penalty=0., hidden_units=None, dropout=0., ann_kwargs={},
            name="", **kwargs):
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

        ## Default to hidden states that are the same size as their outputs,
        ## but give the user the option to provide an alternative.
        if hidden_units is None:
            self._hidden_units = self._ann_units
        else:
            assert len(hidden_units)==len(self._ann_units), \
                "Hidden units list must have the same number of" + \
                " elements as the number of ANN layers"
            self._hidden_units = hidden_units

        ## 3 dense layers per requested layer
        self._ann_layers = [(
            tf.keras.layers.Dense(units=hidden, **self._ann_kwargs),
            tf.keras.layers.Dense(units=hidden, **self._ann_kwargs),
            tf.keras.layers.Dense(units=units, **self._ann_kwargs)
            ) for hidden,units in zip(self._hidden_units,self._ann_units)]

        self._ann_out_layer = tf.keras.layers.Dense(
                self.units, activation="linear")
        self.state_size = (self.units, tuple(self._ann_units))
        self.output_size = self.units

    def call(self, inputs, states, training=False):
        """
        :@param inputs: Tensor input for this cell
        :@param states: tuple of tensor states corresponding to the
            (prev_accumulated, (output, context)) states of the previous cell
        :@param training: determines dropout behavior
        """
        ## list of 2-tuple (output,context) states for the next cell
        new_hidden_states = []
        ## previous accumulated predicted vector and previous ann cell states
        prev_acc,prev_ann_states = states
        all_in = tf.concat((inputs, prev_acc), axis=-1, name="acc_concat")
        prev_layer = all_in
        for i,((A,B,C),S) in enumerate(zip(self._ann_layers,prev_ann_states)):
            ## project the previous state and new inputs to the hidden domain
            latent_state = A(S[0], training=training)
            latent_input = B(prev_layer, training=training)
            new_state = latent_state + latent_input
            ## calculate the output from the hidden domain
            new_hidden_states.append(new_state)
            prev_layer = C(new_state, training=training)
            if self._l2 > 0.:
                self.add_loss(self._l2 * tf.math.reduce_sum(new_state**2))
                self.add_loss(self._l2 * tf.math.reduce_sum(prev_layer**2))
            if self._dropout > 0:
                prev_layer = self._dropout_layers[i](
                        prev_layer, training=training)

        new_res = self._ann_out_layer(prev_layer, training=training)
        ## accumulate from the previous state
        new_acc = prev_acc + new_res
        return (new_res, (new_acc, new_hidden_states))

    def get_config(self):
        """
        Returns dict of keyword  parameters for __init__ which enable the
        layer to be integrated into a functionally defined model.
        """
        config = {
                "ann_layer_units": self._ann_units,
                "pred_units": self.units,
                "ann_kwargs": self._ann_kwargs,
                "hidden_units": self._hidden_units,
                "l2_penalty":self._l2,
                "dropout": self._dropout,
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
        init_ann_states = [[
            tf.zeros((batch_size, d), dtype=self.compute_dtype)
            ] for d in self._ann_units]
        init_acc_state = tf.zeros((batch_size, self.units))
        return init_acc_state,init_ann_states

if __name__=="__main__":
    pass
