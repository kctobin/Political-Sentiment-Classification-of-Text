import tensorflow as tf
import numpy as np

def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper

class CNN(object):
    def __init__(self, graph=None, *args, **kwargs):
        
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, h, depth_size, filters):
        # Model structure; these need to be fixed for a given model.
        self.h_ = h
        self.filters_ = filters
        self.depth_size_ = depth_size

        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            self.learning_rate_ = tf.placeholder(tf.float32, [], name="learning_rate")
            
            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def BuildCoreGraph(self):
        # Input ids, with dynamic shape depending on input.
        # Should be shape [batch_size, max_time, embed_dim] and matrices of embedded representations of sentences.
        self.input_x_ = tf.placeholder(tf.int32, [None, None, None], name="x")

        # Output logits, which can be used by loss functions or for prediction.
        # Overwrite this with an actual Tensor of shape
        # [batch_size, max_time, V].
        self.logits_ = None

        # Should be the same shape as inputs_w_
        # [batch size, num_of_classes]
        self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

        # Replace this with an actual loss function
        self.loss_ = None

        # Get dynamic shape info from inputs
        with tf.name_scope("batch_size"):
            self.batch_size_ = tf.shape(self.input_x_)[0]
        with tf.name_scope("max_time"):
            self.max_time_ = tf.shape(self.input_x_)[1]
        with tf.name_scope("num_classes"):
            self.num_classes_ = tf.shape(self.target_y_)[1]


        # Construct embedding layer
        # self.W_in_ = tf.get_variable("W_embed",[self.V,self.H],initializer=tf.random_uniform_initializer(minval=-1,maxval=1))
        # self.embed_ = tf.nn.embedding_lookup(self.W_in_, self.input_w_)
        
        
        w2 = tf.get_variable('w2', shape=[self.depth_size_, self.h_])
        b2 = tf.get_variable('b2', shape=[self.h_])
        w3 = tf.get_variable('w3', shape=[self.h_, self.num_classes_])
        b3 = tf.get_variable('b3', shape=[self.num_classes_])
        
        l1 = tf.reshape(self.input_x_, [-1, self.max_time_, self.embed_, 1])
        l2 = tf.nn.dropout(l1, pk1)
        pooled_outputs = []
        for i, filter_size in enumerate(self.filters_):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                w1 = tf.get_variable('w1', shape=[filter_size, self.embed_, 1, self.depth_size_])
                l3 = tf.nn.conv2d(l2, w1, [1,1,1,1], 'SAME')
                l4 = tf.nn.relu(l3)
                l5 = tf.nn.max_pool(l4, [1, (self.max_time_-filter_size+1), 1, 1], [1, 1, 1, 1], 'VALID')
                pooled_outputs.append(l5)
        
        
        l5_pool = tf.concat(pooled_outputs, 3)
        l6 = tf.reshape(l5_pool, [-1, (self.depth_size_ * len(self.filters_))])
        l7 = tf.nn.dropout(l6, pk2)
        l8 = tf.matmul(l7, w2) + b2
        l9 = tf.nn.relu(l8)

        l10 = tf.nn.dropout(l9, pk3)
        self.logits_ = tf.matmul(l10, w3) + b3
        self.preds_ = tf.nn.softmax(self.logits_)


        # Loss computation (true loss, for prediction)
        self.loss_ = tf.losses.softmax_cross_entropy(self.target_y_, self.logits_)

        #### END(YOUR CODE) ####

    @with_self_graph
    def BuildTrainGraph(self, c):
        """Construct the training ops.

        You should define:
        - train_loss_ : sampled softmax loss, for training
        - train_step_ : a training op that can be called once per batch

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).
        """
        # Replace this with an actual training op
        self.train_step_ = None

        # Replace this with an actual loss function
        self.train_loss_ = tf.losses.softmax_cross_entropy(self.target_y_, self.logits_, weights=c)

        # Define optimizer and training op
        self.optimizer_ = tf.train.AdamOptimizer(self.learning_rate_)
        #self.grads_, self.variables_ = zip(*self.optimizer_.compute_gradients(self.train_loss_))
        #self.grads_, _ = tf.clip_by_global_norm(self.grads_, self.max_grad_norm_)
        #self.train_step_ = self.optimizer_.apply_gradients(zip(self.grads_, self.variables_))
        self.train_step_ = self.optimizer_.minimize(self.train_loss_)
        

### code to run
def run_epoch(cnn, session, batch_iterator, train=False, verbose=False, learning_rate=0.1):
    
    total_cost = 0.0  # total cost, summed over all words
    total_batches = 0

    if train:
        train_op = cnn.train_step_
        use_dropout = True
        loss = cnn.train_loss_
    else:
        train_op = tf.no_op()
        use_dropout = False  # no dropout at test time
        loss = cnn.loss_  # true loss, if train_loss is an approximation

    for i, (w, y) in enumerate(batch_iterator):
        cost = 0.0

        feed_dict = {lm.input_w_: w, lm.initial_h_: h, lm.target_y_: y, 
                     lm.learning_rate_: learning_rate, lm.use_dropout_: use_dropout}
        
        _, h, cost = session.run([train_op, lm.final_h_, loss], feed_dict = feed_dict)
        
        total_cost += cost
        total_batches = i + 1

    return total_cost / total_batches

def score_dataset(lm, session, ids, name="Data"):
    # For scoring, we can use larger batches to speed things up.
    bi = utils.rnnlm_batch_generator(ids, batch_size=100, max_time=100)
    cost = run_epoch(lm, session, bi, 
                     learning_rate=1.0, train=False, 
                     verbose=False, tick_s=3600)
    print("{:s}: avg. loss: {:.03f}  (perplexity: {:.02f})".format(name, cost, np.exp(cost)))
    return cost

# Training parameters
max_time = 20
batch_size = 50
learning_rate = 0.01
num_epochs = 5

# Model parameters
model_params = dict(V=vocab.size, 
                    H=100, 
                    softmax_ns=200,
                    num_layers=1)

lm = rnnlm.RNNLM(**model_params)
lm.BuildCoreGraph()
lm.BuildTrainGraph()

# Explicitly add global initializer and variable saver to LM graph
with lm.graph.as_default():
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver() 

with tf.Session(graph=lm.graph) as session:
    # Seed RNG for repeatability
    # tf.set_random_seed(42)

    session.run(initializer)

    for epoch in range(1,num_epochs+1):
        t0_epoch = time.time()
        bi = utils.rnnlm_batch_generator(train_ids, batch_size, max_time)

        # Run a training epoch.
        run_epoch(lm, session, bi, train=True, verbose=False, tick_s=print_interval, learning_rate=learning_rate)
        

    score_dataset(lm, session, train_ids, name="Train set")
    score_dataset(lm, session, test_ids, name="Test set")
