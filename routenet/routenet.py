# Copyright (c) 2018-2019, Krzysztof Rusek
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# author: Krzysztof Rusek, AGH


import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse

hparams = tf.contrib.training.HParams(
    node_count=14,
    link_state_dim=4, 
    path_state_dim=2,
    T=3,
    readout_units=8,
    learning_rate=0.001,
    batch_size=32,
    dropout_rate=0.5,
    l2=0.1,
    l2_2=0.01,
    learn_embedding=True, # If false, only the readout is trained
    readout_layers=2, # number of hidden layers in readout model
)

class RouteNet(tf.keras.Model):
    def __init__(self,hparams, output_units=1, final_activation=None):
        super(RouteNet, self).__init__()

        self.hparams = hparams
        self.output_units = output_units
        self.final_activation = final_activation

          
    def build(self, input_shape=None):
        del input_shape

        self.edge_update = tf.keras.layers.GRUCell(self.hparams.link_state_dim, name="edge_update")
        self.path_update = tf.keras.layers.GRUCell(self.hparams.path_state_dim, name="path_update")

        
        self.readout = tf.keras.models.Sequential(name='readout')

        for i in range(self.hparams.readout_layers):
            self.readout.add(tf.keras.layers.Dense(self.hparams.readout_units, 
                    activation=tf.nn.selu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hparams.l2)))
            self.readout.add(tf.keras.layers.Dropout(rate=self.hparams.dropout_rate))

        self.final = keras.layers.Dense(self.output_units, 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hparams.l2_2),
                activation = self.final_activation )
        
        self.edge_update.build(tf.TensorShape([None,self.hparams.path_state_dim]))
        self.path_update.build(tf.TensorShape([None,self.hparams.link_state_dim]))
        self.readout.build(input_shape = [None,self.hparams.path_state_dim])
        self.final.build(input_shape = [None,self.hparams.path_state_dim + self.hparams.readout_units ])


        self.built = True
    
    
    def call(self, inputs, training=False):
        '''

        outputs:
            Natural parameter
        '''
        f_ = inputs
        shape = tf.stack([f_['n_links'],self.hparams.link_state_dim-1], axis=0)
        #link_state = tf.zeros(shape)
        link_state = tf.concat([
            tf.expand_dims(f_['capacities'],axis=1),
            tf.zeros(shape)
        ], axis=1)

        shape = tf.stack([f_['n_paths'],self.hparams.path_state_dim-1], axis=0)
        path_state = tf.concat([
            tf.expand_dims(f_['traffic'][0:f_["n_paths"]],axis=1),
            tf.zeros(shape)
        ], axis=1)

        links = f_['links']
        paths = f_['paths']
        seqs=  f_['sequences']
        
        for _ in range(self.hparams.T):
        
            h_ = tf.gather(link_state,links)

            #TODO move this to feature calculation
            ids=tf.stack([paths, seqs], axis=1)            
            max_len = tf.reduce_max(seqs)+1
            shape = tf.stack([f_['n_paths'], max_len, self.hparams.link_state_dim])
            lens = tf.segment_sum(data=tf.ones_like(paths),
                                    segment_ids=paths)

            link_inputs = tf.scatter_nd(ids, h_, shape)
            #TODO move to tf.keras.RNN
            outputs, path_state = tf.nn.dynamic_rnn(self.path_update,
                                                    link_inputs,
                                                    sequence_length=lens,
                                                    initial_state = path_state,
                                                    dtype=tf.float32)
            m = tf.gather_nd(outputs,ids)
            m = tf.unsorted_segment_sum(m, links ,f_['n_links'])

            #Keras cell expects a list
            link_state,_ = self.edge_update(m, [link_state])
            
        if self.hparams.learn_embedding:
            r = self.readout(path_state,training=training)
            o = self.final(tf.concat([r,path_state], axis=1))
            
        else:
            r = self.readout(tf.stop_gradient(path_state),training=training)
            o = self.final(tf.concat([r, tf.stop_gradient(path_state)], axis=1) )
            
        return o



def delay_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labrange
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration

   
    model = RouteNet(params, output_units=2)
    model.build()

    predictions = model(features, training=mode==tf.estimator.ModeKeys.TRAIN)

    loc  = predictions[...,0] 
    c = np.log(np.expm1( np.float32(0.098) ))
    scale = tf.math.softplus(c + predictions[...,1]) + np.float32(1e-9)

    delay_prediction = loc
    jitter_prediction = scale**2


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, 
            predictions={'delay':delay_prediction, 'jitter':jitter_prediction}
            )

    with tf.name_scope('heteroscedastic_loss'):
        x=features
        y=labels

        n=x['packets']-y['drops']
        _2sigma = np.float32(2.0)*scale**2
        nll = n*y['jitter']/_2sigma + n*tf.math.squared_difference(y['delay'], loc)/_2sigma + n*tf.math.log(scale)
        loss = tf.reduce_sum(nll)/np.float32(1e6)

    regularization_loss = sum(model.losses)
    total_loss = loss + regularization_loss
    
    tf.summary.scalar('regularization_loss', regularization_loss)


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,loss=loss,
            eval_metric_ops={
                'label/mean/delay':tf.metrics.mean(labels['delay']),
                'label/mean/jitter':tf.metrics.mean(labels['jitter']),
                'prediction/mean/delay': tf.metrics.mean(delay_prediction),
                'prediction/mean/jitter': tf.metrics.mean(jitter_prediction),
                'mae/delay':tf.metrics.mean_absolute_error(labels['delay'], delay_prediction),
                'mae/jitter':tf.metrics.mean_absolute_error(labels['jitter'], jitter_prediction),
                'rho/delay':tf.contrib.metrics.streaming_pearson_correlation(labels=labels['delay'],predictions=delay_prediction),
                'rho/jitter':tf.contrib.metrics.streaming_pearson_correlation(labels=labels['jitter'],predictions=jitter_prediction)
            }
        )
    
    assert mode == tf.estimator.ModeKeys.TRAIN


    trainables = model.variables
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in trainables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    decayed_lr = tf.train.exponential_decay(params.learning_rate,
                                            tf.train.get_global_step(), 50000,
                                            0.9, staircase=True)

    optimizer=tf.train.AdamOptimizer(decayed_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs,
            global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, 
            loss=total_loss, 
            train_op=train_op,
        )

def drop_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labrange
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration

   
    model = RouteNet(params, output_units=1, final_activation=None)
    model.build()

    logits = model(features, training=mode==tf.estimator.ModeKeys.TRAIN)
    logits = tf.squeeze(logits)
    predictions = tf.math.sigmoid(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, 
            predictions={'drops':predictions, 'logits':logits}
            )

    with tf.name_scope('binomial_loss'):
        x=features
        y=labels

        loss_ratio = y['drops']/x['packets']
        # Binomial negative Log-likelihood
        loss = tf.reduce_sum(x['packets']*tf.nn.sigmoid_cross_entropy_with_logits(
            labels = loss_ratio,
            logits = logits
        ))/np.float32(1e5)

    regularization_loss = sum(model.losses)
    total_loss = loss + regularization_loss
    
    tf.summary.scalar('regularization_loss', regularization_loss)


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,loss=loss,
            eval_metric_ops={
                'label/mean/drops':tf.metrics.mean(loss_ratio),
                'prediction/mean/drops': tf.metrics.mean(predictions),
                'mae/drops':tf.metrics.mean_absolute_error(loss_ratio, predictions),
                'rho/drops':tf.contrib.metrics.streaming_pearson_correlation(labels=loss_ratio,predictions=predictions)
            }
        )
    
    assert mode == tf.estimator.ModeKeys.TRAIN


    trainables = model.trainable_variables
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in trainables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    decayed_lr = tf.train.exponential_decay(params.learning_rate,
                                            tf.train.get_global_step(), 50000,
                                            0.9, staircase=True)
    # TODO use decay !
    optimizer=tf.train.AdamOptimizer(decayed_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs,
            global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, 
            loss=total_loss, 
            train_op=train_op,
        )

def scale_fn(k, val):
    '''Scales given feature
    Args:
        k: key
        val: tensor value
    '''

    if k == 'traffic':
        return (val-0.18)/.15
    if k == 'capacities':
        return val/10.0

    return val



def parse(serialized, target=None, normalize=True):
    '''
    Target is the name of predicted variable-deprecated
    '''
    with tf.device("/cpu:0"):
        with tf.name_scope('parse'):    
            #TODO add feature spec class
            features = tf.io.parse_single_example(
                serialized,
                features={
                    'traffic':tf.VarLenFeature(tf.float32),
                    'delay':tf.VarLenFeature(tf.float32),
                    'logdelay':tf.VarLenFeature(tf.float32),
                    'jitter':tf.VarLenFeature(tf.float32),
                    'drops':tf.VarLenFeature(tf.float32),
                    'packets':tf.VarLenFeature(tf.float32),
                    'capacities':tf.VarLenFeature(tf.float32),
                    'links':tf.VarLenFeature(tf.int64),
                    'paths':tf.VarLenFeature(tf.int64),
                    'sequences':tf.VarLenFeature(tf.int64),
                    'n_links':tf.FixedLenFeature([],tf.int64), 
                    'n_paths':tf.FixedLenFeature([],tf.int64),
                    'n_total':tf.FixedLenFeature([],tf.int64)
                })
            for k in ['traffic','delay','logdelay','jitter','drops','packets','capacities','links','paths','sequences']:
                features[k] = tf.sparse.to_dense( features[k] )
                if normalize:
                    features[k] = scale_fn(k, features[k])


    #return {k:v for k,v in features.items() if k is not target },features[target]
    return features

def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max( extractor(v) ) + 1 for v in alist ]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes)-1):
            cummaxes.append( tf.math.add_n(maxes[0:i+1]))
        
    return cummaxes
        

def transformation_func(it, batch_size=32):
    with tf.name_scope("transformation_func"):
        vs = [it.get_next() for _ in range(batch_size)]
        
        links_cummax = cummax(vs,lambda v:v['links'] )
        paths_cummax = cummax(vs,lambda v:v['paths'] )
        
        tensors = ({
                'traffic':tf.concat([v['traffic'] for v in vs], axis=0),
                'capacities': tf.concat([v['capacities'] for v in vs], axis=0),
                'sequences':tf.concat([v['sequences'] for v in vs], axis=0),
                'packets':tf.concat([v['packets'] for v in vs], axis=0),
                'links':tf.concat([v['links'] + m for v,m in zip(vs, links_cummax) ], axis=0),
                'paths':tf.concat([v['paths'] + m for v,m in zip(vs, paths_cummax) ], axis=0),
                'n_links':tf.math.add_n([v['n_links'] for v in vs]),
                'n_paths':tf.math.add_n([v['n_paths'] for v in vs]),
                'n_total':tf.math.add_n([v['n_total'] for v in vs])
            },   {
                'delay' : tf.concat([v['delay'] for v in vs], axis=0),
                'logdelay' : tf.concat([v['logdelay'] for v in vs], axis=0),
                'drops' : tf.concat([v['drops'] for v in vs], axis=0),
                'jitter' : tf.concat([v['jitter'] for v in vs], axis=0),
                }
            )
    
    return tensors

def tfrecord_input_fn(filenames,hparams,shuffle_buf=1000, target='delay'):
    
    files = tf.data.Dataset.from_tensor_slices(filenames)
    files = files.shuffle(len(filenames))

    ds = files.apply(tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))

    if shuffle_buf:
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buf))
    else :
        # sample 10 % for evaluation because it is time consuming
        ds = ds.filter(lambda x: tf.random_uniform(shape=())< 0.1)

    ds = ds.map(lambda buf:parse(buf,target), 
        num_parallel_calls=2)
    ds=ds.prefetch(10)

    it =ds.make_one_shot_iterator()
    sample = transformation_func(it,hparams.batch_size)
    

    return sample

def serving_input_receiver_fn():
    """
    This is used to define inputs to serve the model.
    returns: ServingInputReceiver
    """
    receiver_tensors = {
        'capacities': tf.placeholder(tf.float32, [None]),
        'traffic': tf.placeholder(tf.float32, [None]),
        'links': tf.placeholder(tf.int32, [None]),
        'paths': tf.placeholder(tf.int32, [None]),
        'sequences': tf.placeholder(tf.int32, [None]),
        'n_links': tf.placeholder(tf.int32, []),
        'n_paths':tf.placeholder(tf.int32, []),
    }

    # Convert give inputs to adjust to the model.
    features = {k: scale_fn(k,v) for k,v in receiver_tensors.items() }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors,
                                                    features=features)


def train(args):
    print(args)
    tf.logging.set_verbosity('INFO')    

    if args.hparams:
        hparams.parse(args.hparams)

    model_fn = delay_model_fn if args.target =='delay' else drop_model_fn

    estimator = tf.estimator.Estimator( 
        model_fn = model_fn, 
        model_dir=args.model_dir,
        params=hparams,
        warm_start_from=args.warm
        )

    best_exporter = tf.estimator.BestExporter(
        serving_input_receiver_fn=serving_input_receiver_fn,
        exports_to_keep=2)

    latest_exporter = tf.estimator.LatestExporter(
        name="latests",
        serving_input_receiver_fn=serving_input_receiver_fn,
        exports_to_keep=5)


    train_spec = tf.estimator.TrainSpec(input_fn=lambda:tfrecord_input_fn(args.train,hparams,shuffle_buf=args.shuffle_buf,target=args.target), 
       max_steps=args.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:tfrecord_input_fn(args.evaluation,hparams,shuffle_buf=None,target=args.target),
        steps=args.eval_steps,
        exporters=[best_exporter,latest_exporter],
        #throttle_secs=1800)
        throttle_secs=600)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main():
    parser = argparse.ArgumentParser(description='RouteNet script')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_train = subparsers.add_parser('train', help='Train options')
    parser_train.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" pairs.')
    parser_train.add_argument('--train', help='Train Tfrecords files',  type=str ,nargs='+')
    parser_train.add_argument('--evaluation', help='Evaluation Tfrecords files',  type=str ,nargs='+')
    parser_train.add_argument('--model_dir', help='Model directory',  type=str )
    parser_train.add_argument('--train_steps', help='Training steps',  type=int, default=100 )
    parser_train.add_argument('--eval_steps', help='Evaluation steps, defaul None= all',  type=int, default=None )
    parser_train.add_argument('--shuffle_buf',help = "Buffer size for samples shuffling", type=int, default=10000)
    parser_train.add_argument('--target',help = "Predicted variable", type=str, default='delay')
    parser_train.add_argument('--warm',help = "Warm start from", type=str, default=None)
    parser_train.set_defaults(func=train)
    args = parser.parse_args()

    return args.func(args)
    


    
if __name__ == '__main__':

    main()
