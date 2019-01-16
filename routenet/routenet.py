# author:
# - 'Krzysztof Rusek [^1]'
# 
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
# 


import numpy as np
import pandas as pd
import networkx as nx
#import matplotlib.pyplot as plt
import itertools as it
import tensorflow as tf
from tensorflow import keras
import collections 
import re
import argparse
import sys

def genPath(R,s,d,connections):
    while s != d:
        yield s
        s = connections[s][R[s,d]]
    yield s

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def ned2lists(fname):
    channels = []
    with open(fname) as f:
        p = re.compile(r'\s+node(\d+).port\[(\d+)\]\s+<-->\s+Channel\s+<-->\s+node(\d+).port\[(\d+)\]')
        for line in f:
            m=p.match(line)
            if m:
                #print(line, m.groups())
                channels.append(list(map(int,m.groups())))
    n=max(map(max, channels))+1
    connections = [{} for i in range(n)]
    for c in channels:
        connections[c[0]][c[1]]=c[2]
        connections[c[2]][c[3]]=c[0]
    connections = [[v for k,v in sorted(con.items())] 
                   for con in connections ]
    return connections,n


def extract_links(n, connections):
    A = np.zeros((n,n))

    for a,c in zip(A,connections):
        a[c]=1

    G=nx.from_numpy_array(A, create_using=nx.DiGraph())
    edges=list(G.edges)
    return edges

def load_and_process(routing_file, data_file,edges,connections,n=15):
    R=np.loadtxt(routing_file, dtype=np.int32)
    data = np.loadtxt(data_file)
    traffic = np.reshape(data[:,0:n*n],(-1,n,n))
    delay = np.reshape(data[:,n*n:2*n*n],(-1,n,n))
    #packet_loss = data[:,-1]
    
    paths=[]
    features=[]
    labels=[]

    for i in range(n):
        for j in range(n):
            if i != j:
                paths.append([edges.index(tup) for tup in pairwise(genPath(R,i,j,connections))])
                features.append(traffic[:,i,j])
                labels.append(delay[:,i,j])
    features = np.stack(features).T
    labels = np.stack(labels).T
    return paths,features,labels

def load(data_file,n, full=False):
    names=[]

    TM_index=[]
    delay_index=[]
    if full:
        jitter_index=[]
        drop_index=[]

    counter=0
    for i in range(n):
        for j in range(n):
            names.append('a{}_{}'.format(i,j))
            if i != j:
                TM_index.append(counter)
            counter += 1
    for i in range(n):
        for j in range(n):
            for k in ['average', 'q10','q20','q50','q80','q90','variance']:
                names.append('delay{}_{}_{}'.format(i,j,k))
                if i != j and k == 'average':
                    delay_index.append(counter)
                if i != j and k == 'variance' and full:
                    jitter_index.append(counter)
                counter += 1
    for i in range(n):
        for j in range(n):
            names.append('drop{}_{}'.format(i,j))
            if full and i != j:
                drop_index.append(counter)
            counter += 1
    names.append('empty')
            
    Global=pd.read_csv(data_file ,header=None, names=names,index_col=False)
    if full:
        return Global, TM_index, delay_index, jitter_index, drop_index
    else:
        return Global, TM_index, delay_index

def load_routing(routing_file):
    R = pd.read_csv(routing_file, header=None, index_col=False)
    R=R.drop([R.shape[0]], axis=1)
    return R.values
   

def make_paths(R,connections):
    n = R.shape[0]
    edges = extract_links(n, connections)
    paths=[]
    for i in range(n):
        for j in range(n):
            if i != j:
                paths.append([edges.index(tup) for tup in pairwise(genPath(R,i,j,connections))])
    return paths

def make_indices(paths):
    link_indices=[]
    path_indices=[]
    sequ_indices=[]
    segment=0
    for p in paths:
        link_indices += p
        path_indices += len(p)*[segment]
        sequ_indices += list(range(len(p)))
        segment +=1
    return link_indices, path_indices, sequ_indices

NetworkSamples = collections.namedtuple('NetworkSamples',['features', 
                                                            'labels', 'links', 
                                                            'paths', 'sequances',
                                                            'n_links',
                                                            'n_paths'])

def make_samples(routing_file, data_file,connections,n=15):
    
    edges = extract_links(n,connections)
    paths,features,labels = load_and_process(routing_file, data_file,edges,connections,n)
    
    link_indices=[]
    path_indices=[]
    sequ_indices=[]
    segment=0
    for p in paths:
        link_indices += p
        path_indices += len(p)*[segment]
        sequ_indices += list(range(len(p)))
        segment +=1
        
        
    
    return NetworkSamples(features, 
                          labels, 
                          link_indices, 
                          path_indices, 
                          sequ_indices,
                          n_links = len(edges),n_paths=features.shape[1])


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def make_tfrecord(file_name,samples):
    writer = tf.python_io.TFRecordWriter(file_name)

    for a,d in zip(samples.features,samples.labels):
        example = tf.train.Example(features=tf.train.Features(feature={
            'traffic':_float_features(a),
            'delay':_float_features(d),
            'links':_int64_features(samples.links),
            'paths':_int64_features(samples.paths),
            'sequances':_int64_features(samples.sequances),
            'n_links':_int64_feature(samples.n_links), 
            'n_paths':_int64_feature(samples.n_paths)
        }
        ))

        writer.write(example.SerializeToString())
    writer.close()

def make_tfrecord2(file_name,ned_file,routing_file,data_file):

    con,n = ned2lists(ned_file)
    Global, TM_index, delay_index, jitter_index, drop_index = load(data_file,n, True)
    R = load_routing(routing_file)
    paths = make_paths(R, con)
    link_indices, path_indices, sequ_indices = make_indices(paths)

    delay = Global.take(delay_index,axis=1).values
    TM = Global.take(TM_index,axis=1).values
    jitter =  Global.take(jitter_index,axis=1).values
    drops =  Global.take(drop_index,axis=1).values

    n_paths = delay.shape[1]
    n_links = max(max(paths))+1

    writer = tf.python_io.TFRecordWriter(file_name)
    n_total = len(path_indices)

    for a,d,j,l in zip(TM,delay,jitter,drops):
        example = tf.train.Example(features=tf.train.Features(feature={
            'traffic':_float_features(a),
            'delay':_float_features(d),
            'jitter':_float_features(j),
            'drops':_float_features(l),
            'links':_int64_features(link_indices),
            'paths':_int64_features(path_indices),
            'sequances':_int64_features(sequ_indices),
            'n_links':_int64_feature(n_links), 
            'n_paths':_int64_feature(n_paths),
            'n_total':_int64_feature(n_total)
        }
        ))

        writer.write(example.SerializeToString())
    writer.close()

def infer_routing_nsf(data_file):
    rf=re.sub(r'dGlobal_S\d+_R','Routing_', data_file).\
    replace('datasets','routing')
    return rf

def infer_routing_nsf2(data_file):
    rf=re.sub(r'dGlobal_\d+_\d+_','Routing_', data_file).\
    replace('datasets2','routing2')
    return rf



def infer_routing_nsf3(data_file):
    rf=re.sub(r'dGlobal_\d+_\d+_','Routing_', data_file).\
    replace('delaysNsfnet','routingNsfnet')
    return rf

def infer_routing_geant(data_file):
    rf=re.sub(r'dGlobal_G_\d+_\d+_','RoutingGeant2_', data_file).\
    replace('delaysGeant2','routingsGeant2')
    return rf


def input_fn(samples,hparams,shuffle=True):
    f = ((samples.features-0.2)/0.15).astype(np.float32)
    l = ((samples.labels-2.0)/1.5).astype(np.float32)
    
    ds = tf.data.Dataset.from_tensor_slices(({'traffic':f},l))
    
    ds1 = tf.data.Dataset.from_tensors({samples._fields[i]:samples[i] for i in range(2, len(samples))})
    ds1 = ds1.repeat()
   
    if shuffle:
        ds = ds.repeat()
        ds = ds.shuffle(1000)
    
    ds=tf.data.Dataset.zip((ds,ds1))
    ds = ds.batch(hparams.batch_size)
    sample = ds.make_one_shot_iterator().get_next()
    sample[0][0].update(sample[1])
    return sample[0]

def parse(serialized, target='delay'):
    '''
    Target is the name of predicted variable
    '''
    with tf.device("/cpu:0"):
        with tf.name_scope('parse'):    
            features = tf.parse_single_example(
                serialized,
                features={
                    'traffic':tf.VarLenFeature(tf.float32),
                    target:tf.VarLenFeature(tf.float32),
                    'links':tf.VarLenFeature(tf.int64),
                    'paths':tf.VarLenFeature(tf.int64),
                    'sequances':tf.VarLenFeature(tf.int64),
                    'n_links':tf.FixedLenFeature([],tf.int64), 
                    'n_paths':tf.FixedLenFeature([],tf.int64),
                    'n_total':tf.FixedLenFeature([],tf.int64)
                })
            for k in ['traffic',target,'links','paths','sequances']:
                features[k] = tf.sparse_tensor_to_dense(features[k])
                if k == 'delay':
                    features[k] = (features[k]-2.8)/2.5
                if k == 'traffic':
                    #features[k] = (features[k]-0.76)/.008
                    features[k] = (features[k]-0.5)/.5
                if k == 'drops':
                    features[k] = (features[k])/12000/(0.5*features['traffic']+0.5) #loss rate
                #if k == 'jitter':
                    #features[k] = (tf.math.log( features[k] )-2.0)/2.0 #logjitter


    return {k:v for k,v in features.items() if k is not target },features[target]

def tfrecord_input_fn(filenames,hparams,shuffle_buf=1000, target='delay'):
    
    files = tf.data.Dataset.from_tensor_slices(filenames)
    files = files.shuffle(len(filenames))

    ds = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))

    if shuffle_buf:
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buf))
    ds = ds.map(lambda buf:parse(buf,target), 
        num_parallel_calls=2)


    shapes=({
    'traffic':[hparams.node_count*(hparams.node_count-1)],
    'links':[-1],
    'paths':[-1],
    'sequances':[-1],
    'n_links':[], 
    'n_paths':[],
    'n_total':[]
    },[hparams.node_count*(hparams.node_count-1)])
    ds = ds.padded_batch(hparams.batch_size,shapes)
    
    ds = ds.prefetch(1)

    sample = ds.make_one_shot_iterator().get_next()
    
    return sample



class ComnetModel(tf.keras.Model):
    def __init__(self,hparams, output_units=1):
        super(ComnetModel, self).__init__()
        self.hparams = hparams

        self.edge_update = tf.nn.rnn_cell.GRUCell(hparams.link_state_dim, dtype=tf.float32)
        self.path_update = tf.nn.rnn_cell.GRUCell(hparams.path_state_dim, dtype=tf.float32)

        # wait for tf 1.11
        #self.edge_update = tf.keras.layers.GRUCell(hparams.link_state_dim)
        #self.path_update = tf.keras.layers.GRUCell(hparams.path_state_dim)

        
        self.readout = tf.keras.models.Sequential()

        self.readout.add(keras.layers.Dense(hparams.readout_units, 
                activation=tf.nn.selu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2)))
        self.readout.add(keras.layers.Dropout(rate=hparams.dropout_rate))
        self.readout.add(keras.layers.Dense(hparams.readout_units, 
                activation=tf.nn.selu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2)))
        self.readout.add(keras.layers.Dropout(rate=hparams.dropout_rate))

        self.readout.add(keras.layers.Dense(output_units, kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_2)))

            
    def build(self, input_shape=None):
        del input_shape
        #TODO unify with keras when tf 1.11 is available on plgrid
        self.edge_update.build(tf.TensorShape([None,self.hparams.path_state_dim]))
        self.path_update.build(tf.TensorShape([None,self.hparams.link_state_dim]))
        self.readout.build(input_shape = [None,self.hparams.path_state_dim])
        self.built = True
    
    def call(self, inputs, training=False):
        f_ = inputs
        shape = tf.stack([f_['n_links'],self.hparams.link_state_dim], axis=0)
        link_state = tf.zeros(shape)
        shape = tf.stack([f_['n_paths'],self.hparams.path_state_dim-1], axis=0)
        path_state = tf.concat([
            tf.expand_dims(f_['traffic'],axis=1),
            tf.zeros(shape)
        ], axis=1)

        links = f_['links'][0:f_["n_total"]]
        paths = f_['paths'][0:f_["n_total"]]
        seqs=  f_['sequances'][0:f_["n_total"]]
        
        for _ in range(self.hparams.T):
        
            h_tild = tf.gather(link_state,links)

            ids=tf.stack([paths, seqs], axis=1)            
            max_len = tf.reduce_max(seqs)+1
            shape = tf.stack([f_['n_paths'], max_len, self.hparams.link_state_dim])
            lens = tf.segment_sum(data=tf.ones_like(paths),
                                    segment_ids=paths)

            link_inputs = tf.scatter_nd(ids, h_tild, shape)
            outputs, path_state = tf.nn.dynamic_rnn(self.path_update,
                                                    link_inputs,
                                                    sequence_length=lens,
                                                    initial_state = path_state,
                                                    dtype=tf.float32)
            m = tf.gather_nd(outputs,ids)
            m = tf.unsorted_segment_sum(m, links ,f_['n_links'])
            _,link_state = self.edge_update(m, link_state)

            # wait for tf 1.11
            #link_state,_ = self.edge_update(m, [link_state])
            
        
        r = self.readout(path_state,training=training)

        # remove to have inference from path state
        # Thsi forces additive model for delay
        #r = tf.gather(r,links)
        #r = tf.segment_sum(r,segment_ids=paths)
        return r
    


def model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labrange
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration

   
    model = ComnetModel(params)
    model.build()

    predictions = tf.map_fn(lambda x: model(x,training=mode==tf.estimator.ModeKeys.TRAIN), 
                    features,dtype=tf.float32)
    #predictions = model(features,training=mode==tf.estimator.ModeKeys.TRAIN)
    predictions = tf.squeeze(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, 
            predictions={'predictions':predictions})

    loss =  tf.losses.mean_squared_error(
        labels=labels,
        predictions = predictions,
        reduction=tf.losses.Reduction.MEAN
    )

    #print(model.variables)
    #print('loss', model.losses,sum(model.losses))
    #print(tf.losses.get_regularization_loss())
    #raise Exception('Stop')
    regularization_loss = sum(model.losses)
    total_loss = loss + regularization_loss
    
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)



    #TODO R**2
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,loss=loss,
            eval_metric_ops={
                'label/mean':tf.metrics.mean(labels),
                'prediction/mean': tf.metrics.mean(predictions),
                'mae':tf.metrics.mean_absolute_error(labels, predictions),
                'rho':tf.contrib.metrics.streaming_pearson_correlation(labels=labels,predictions=predictions)
            }
        )
    
    assert mode == tf.estimator.ModeKeys.TRAIN


    trainables = model.variables
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in trainables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]


    optimizer=tf.train.AdamOptimizer(params.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs,
            global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, 
            loss=loss, 
            train_op=train_op,
        )
    

def flush():
    sys.stdout.flush()
    sys.stderr.flush()

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
    l2_2=0.01
)
def train(args):
    print(args)
    tf.logging.set_verbosity('INFO')    

    if args.hparams:
        hparams.parse(args.hparams)

    estimator = tf.estimator.Estimator( 
        model_fn = model_fn, 
        model_dir=args.model_dir,
        params=hparams
        )
    for _ in range(args.epochs):
        flush()
        estimator.train(lambda:tfrecord_input_fn(args.train,hparams,shuffle_buf=args.shuffle_buf,target=args.target), 
            steps=args.train_steps)
        flush()
        estimator.evaluate(lambda:tfrecord_input_fn(args.eval_,hparams,shuffle_buf=None,target=args.target), 
            steps=args.eval_steps)
        flush()



def data(args):
    print(args)
    #if not args.o:
    #    args.o = args.d.replace('txt', 'tfrecords')
    for data_file in args.d:
        tf_file = data_file.replace('txt', 'tfrecords')
        if not args.r:
            args.r = infer_routing_nsf(data_file)
        tf.logging.info('Starting ', data_file)
        make_tfrecord2(tf_file,args.ned,args.r,data_file)


def main():
    parser = argparse.ArgumentParser(description='Netwrok routing ML tool')

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_data = subparsers.add_parser('data', help='data processing')
    parser_data.add_argument('-d', help='data file',  type=str ,  required=True, nargs='+')
    parser_data.add_argument('-r', help='roting file',  type=str ,  required=False)
    parser_data.add_argument('--ned', help='Topology ned file',  type=str ,  required=True)
    #parser_data.add_argument('-o', help='output file, default to input wit tfrecords extention',  type=str)
    parser_data.set_defaults(func=data)

    parser_train = subparsers.add_parser('train', help='Train options')
    parser_train.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" pairs.')
    parser_train.add_argument('--train', help='Train Tfrecords files',  type=str ,nargs='+')
    parser_train.add_argument('--eval_', help='Evaluation Tfrecords files',  type=str ,nargs='+')
    parser_train.add_argument('--model_dir', help='Model directory',  type=str )
    parser_train.add_argument('--train_steps', help='Training steps',  type=int, default=100 )
    parser_train.add_argument('--eval_steps', help='Evaluation steps, defaul None= all',  type=int, default=None )
    parser_train.add_argument('--epochs', help='Train epochs',  type=int, default=300 )
    parser_train.add_argument('--shuffle_buf',help = "Buffer size for samples shuffling", type=int, default=10000)
    parser_train.add_argument('--target',help = "Predicted variable", type=str, default='delay')
    parser_train.set_defaults(func=train)
    args = parser.parse_args()

    return args.func(args)
    


    
if __name__ == '__main__':

    main()
