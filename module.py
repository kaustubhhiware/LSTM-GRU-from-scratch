import tensorflow as tf 

def get_states(model, processed_input, initial_hidden):
    all_hidden_states = tf.scan(model, processed_input, initializer=initial_hidden, name='states')
    all_hidden_states = all_hidden_states[:, 0, :, :]
    return all_hidden_states
    
def get_output(Wo, bo, hidden_state):
    output = tf.nn.relu(tf.matmul(hidden_state, Wo) + bo)
    return output


class LSTM_cell(object):

    def __init__(self, input_nodes, hidden_unit, output_nodes):

        self.input_nodes = input_nodes
        self.hidden_unit = hidden_unit
        self.output_nodes = output_nodes

        self.Wi = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.Ui = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))
        self.bi = tf.Variable(tf.zeros([self.hidden_unit]))

        self.Wf = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.Uf = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))
        self.bf = tf.Variable(tf.zeros([self.hidden_unit]))

        self.Wog = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.Uog = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))
        self.bog = tf.Variable(tf.zeros([self.hidden_unit]))

        self.Wc = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.Uc = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))
        self.bc = tf.Variable(tf.zeros([self.hidden_unit]))

        # Weights for output layers
        self.Wo = tf.Variable(tf.truncated_normal([self.hidden_unit, self.output_nodes], mean=0, stddev=.01))
        self.bo = tf.Variable(tf.truncated_normal([self.output_nodes], mean=0, stddev=.01))

        # Placeholder for input vector with shape[batch, seq, embeddings]
        self._inputs = tf.placeholder(tf.float32, shape=[None, None, self.input_nodes], name='inputs')

        # Processing inputs to work with scan function
        # Process tensor of size [5,3,2] to [3,5,2]
        batch_input_ = tf.transpose(self._inputs, perm=[2, 0, 1])
        self.processed_input = tf.transpose(batch_input_)

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(self.initial_hidden, tf.zeros([input_nodes, hidden_unit]))

        self.initial_hidden = tf.stack([self.initial_hidden, self.initial_hidden])

    def Lstm(self, previous_hidden_memory_tuple, x):
        # Take previous hidden stats and memory tuple with i/p &
        # o/p current hidden state

        previous_hidden_state, c_prev = tf.unstack(previous_hidden_memory_tuple)

        i = tf.sigmoid( tf.matmul(x, self.Wi) +
                        tf.matmul(previous_hidden_state, self.Ui) + self.bi)

        f = tf.sigmoid( tf.matmul(x, self.Wf) +
                        tf.matmul(previous_hidden_state, self.Uf) + self.bf)

        o = tf.sigmoid( tf.matmul(x, self.Wog) +
                        tf.matmul(previous_hidden_state, self.Uog) + self.bog)

        c_ = tf.nn.tanh(tf.matmul(x, self.Wc) +
                        tf.matmul(previous_hidden_state, self.Uc) + self.bc)

        # Final Memory cell
        c = f * c_prev + i * c_
        current_hidden_state = o * tf.nn.tanh(c)

        return tf.stack([current_hidden_state, c])

    def get_states(self):
        all_hidden_states = tf.scan(self.Lstm, self.processed_input, initializer=self.initial_hidden, name='states')
        all_hidden_states = all_hidden_states[:, 0, :, :]
        return all_hidden_states

    def get_output(self, hidden_state):
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo) + self.bo)
        return output

    def get_outputs(self):
        all_hidden_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)
        return all_outputs


class GRU_cell(object):

    def __init__(self, input_nodes, hidden_unit, output_nodes):

        self.input_nodes = input_nodes
        self.hidden_unit = hidden_unit
        self.output_nodes = output_nodes

        self.Wx = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))

        self.Wr = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.br = tf.Variable(tf.truncated_normal([self.hidden_unit], mean=1))
        
        self.Wz = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.bz = tf.Variable(tf.truncated_normal([self.hidden_unit], mean=1))

        self.Wh = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))

        self.Wo = tf.Variable(tf.truncated_normal([self.hidden_unit, self.output_nodes], mean=1, stddev=.01))
        self.bo = tf.Variable(tf.truncated_normal([self.output_nodes], mean=1, stddev=.01))

        self._inputs = tf.placeholder(tf.float32,shape=[None, None, self.input_nodes], name='inputs')

        batch_input_ = tf.transpose(self._inputs, perm=[2, 0, 1])
        self.processed_input = tf.transpose(batch_input_)

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(self.initial_hidden, tf.zeros([input_nodes, hidden_unit]))

    def Gru(self, previous_hidden_state, x):

        z = tf.sigmoid(tf.matmul(x, self.Wz) + self.bz)
        r = tf.sigmoid(tf.matmul(x, self.Wr) + self.br)

        h_ = tf.tanh(tf.matmul(x, self.Wx) +
                     tf.matmul(previous_hidden_state, self.Wh) * r)

        current_hidden_state = tf.multiply( (1 - z), h_) + tf.multiply(previous_hidden_state, z)

        return current_hidden_state

    def get_states(self):
        all_hidden_states = tf.scan(self.Gru, self.processed_input, initializer=self.initial_hidden, name='states')
        return all_hidden_states

    def get_output(self, hidden_state):
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo) + self.bo)
        return output

    def get_outputs(self):
        all_hidden_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)
        return all_outputs
