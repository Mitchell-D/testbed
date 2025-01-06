""" Modified keras LSTM supporting explicit state accumulation between cells """
import tensorflow as tf

class AccLSTMCell(tf.keras.layers.Layer):
    """
    Layer abstracting a stack of LSTM cells, which explicitly discretely
    integrates the output state and cycles it back into subsequent steps
    as an input
    """
    def __init__(self, ann_out_units, lstm_layer_units:list, ann_in_units,
            lstm_kwargs={}, ann_in_kwargs={}, ann_out_kwargs={}, name="",
            **kwargs):
        """
        :@param ann_out_units: Node count of the last layer, which should be
            the prediction target since the output of this layer is accumulated
            into the input of subsequent nodes.
        :@param lstm_layer_units: List of node counts corresponding to each
            LSTM cell layer within this cell.
        :@param ann_in_units: Node count of the input embedding layer. Note
            that the previous accumulated state will be concatenated to the
            explicit inputs to this layer prior to evaluating it.
        :@param *_kwargs: keyword arguments passed to internal layer inits
        :@param name:
        """
        super().__init__(**kwargs)
        self._ann_out_units = ann_out_units
        self._lstm_units = lstm_layer_units
        self._ann_in_units = ann_in_units
        self._lstm_kwargs = lstm_kwargs
        self._ann_in_kwargs = ann_in_kwargs
        self._ann_out_kwargs = ann_out_kwargs
        self._lstm_layers = []
        self._name = name

        ## input layer includes normal inputs and concatenated accumulation
        self._ann_in_layer = tf.keras.layers.Dense(
                self._ann_in_units,
                **self._ann_in_kwargs
                )
        self._lstm_layers = [
                tf.keras.layers.LSTMCell(units=units, **self._lstm_kwargs)
                for units in self._lstm_units
                ]
        self._ann_out_layer = tf.keras.layers.Dense(
                self._ann_out_units,
                **self._ann_out_kwargs
                )
        self.units = ann_out_units
        self.state_size = (self.units, tuple((u,u) for u in self._lstm_units))
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
        ## previous accumulated predicted vector and previous lstm cell states
        prev_acc,prev_lstm_states = states
        all_in = tf.concat((inputs, prev_acc), axis=-1, name="acc_concat")
        ## Embed the inputs with a dense layer
        tmp_layer = self._ann_in_layer(all_in, training=training)
        for L,S in zip(self._lstm_layers,prev_lstm_states):
            tmp_layer,tmp_hidden = L(tmp_layer, S, training=training)
            new_hidden_states.append(tmp_hidden)
        new_res = self._ann_out_layer(tmp_layer, training=training)
        ## accumulate from the previous state
        new_acc = prev_acc + new_res
        return (new_res, (new_acc, new_hidden_states))

    def get_config(self):
        """
        Returns dict of keyword  parameters for __init__ which enable the
        layer to be integrated into a functionally defined model.
        """
        config = {
                "ann_out_units": self._ann_out_units,
                "lstm_layer_units": self._lstm_units,
                "ann_in_units": self._ann_in_units,
                "lstm_kwargs": self._lstm_kwargs,
                "ann_in_kwargs": self._ann_in_kwargs,
                "ann_out_kwargs": self._ann_out_kwargs,
                "name":self._name,
                }
        base_config = super().get_config()
        return {**base_config, **config}

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Method returning a 2-tuple (init_acc_state,init_lstm_states) where
        init_acc_state is a zero vector representing the initial accumulated
        prediction state, and init_lstm_states is a list of 2-tuple
        (output, context) zeroed tensors.
        """
        init_lstm_states = [[
            tf.zeros((batch_size, d), dtype=self.compute_dtype),
            tf.zeros((batch_size, d), dtype=self.compute_dtype)
            ] for d in self._lstm_units]
        init_acc_state = tf.zeros((batch_size, self._ann_out_units))
        return init_acc_state,init_lstm_states

if __name__=="__main__":
    pass
