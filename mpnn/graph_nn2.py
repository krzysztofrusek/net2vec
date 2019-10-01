import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime
import argparse
import os
import io

parser = argparse.ArgumentParser(description='Train the graph neural network')
parser.add_argument('--pad', help='extra padding for node embeding',  type=int, default=12)
parser.add_argument('--pas', help='number of passes',  type=int, default=4)
parser.add_argument('--batch_size', help='batch_size',  type=int, default=64)
parser.add_argument('--lr', help='learning rate',  type=float, default=0.001)
parser.add_argument('--log_dir', help='log dir',  type=str, default='log')
parser.add_argument('--rn', help='number of readout neurons',  type=int, default=8)
parser.add_argument('--buf', help='buffer',  type=int, default=200)
parser.add_argument('-I', help='number of iteration',  type=int, default=80000)
parser.add_argument('--eval', help='evaluatioin file',  type=str, default='eval.tfrecords')
parser.add_argument('--train', help='train file',  type=str, default='train.tfrecords')
parser.add_argument('--test', help='test file',  type=str, default='test.tfrecords')
parser.add_argument('--ninf', help='Number of hidden neurions in inference layer', type=int, default=256)
parser.add_argument('--Mhid', help='Number of hidden neurons in message layer', type=int, default=8)

def stat_args(name, shift=0,scale=1):
    parser.add_argument('--{}-shift'.format(name), 
        help='Shift  for {} (usualy np.mean)'.format(name) ,  
        type=float, default=shift)

    parser.add_argument('--{}-scale'.format(name), 
        help='Scale  for {} (usualy np.std)'.format(name) ,  
        type=float, default=scale)

stat_args('mu',shift=0.34, scale=0.27)
stat_args('W',shift=55.3, scale=22.0)

if __name__ == '__main__':
    args = parser.parse_args()
else:
    args = parser.parse_args([])

def test():
    return args.I

N_PAD=args.pad
N_PAS=args.pas
N_H=2+N_PAD
REUSE=None
batch_size=args.batch_size

#tf.enable_eager_execution()

def parse(serialized):
    with tf.device("/cpu:0"):
        with tf.name_scope('parse'):
            features = tf.parse_single_example(
                serialized,
                features={
                    'mu': tf.VarLenFeature(tf.float32),
                    "Lambda": tf.VarLenFeature( tf.float32),
                    "W":tf.FixedLenFeature([],tf.float32),
                    "R":tf.VarLenFeature(tf.float32),
                    "first":tf.VarLenFeature(tf.int64),
                    "second":tf.VarLenFeature(tf.int64)})

            ar=[(tf.sparse_tensor_to_dense(features['mu'])-args.mu_shift)/args.mu_scale,
                    (tf.sparse_tensor_to_dense(features['Lambda']))]
            x=tf.stack(ar,axis=1)

            e=tf.sparse_tensor_to_dense(features['R'])
            # cecha jest od 0-1
            #e = (tf.expand_dims(e,axis=1)-0.24)/0.09
            e = tf.expand_dims(e,axis=1)

            first=tf.sparse_tensor_to_dense(features['first'])
            second=tf.sparse_tensor_to_dense(features['second'])
            
            W = (features['W']-args.W_shift)/args.W_scale
            
            return ((x,e,first,second),W)

def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max( extractor(v) ) + 1 for v in alist ]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes)-1):
            cummaxes.append( tf.math.add_n(maxes[0:i+1]))
        
    return cummaxes

def transformation_func(it, batch_size=4):
    with tf.name_scope("transformation_func"):
        vs = [it.get_next() for _ in range(batch_size)]
        
        first_offset = cummax(vs,lambda v:v[0][2] )
        second_offset = cummax(vs,lambda v:v[0][3] )
        
    
    return ((tf.concat([v[0][0] for v in vs], axis=0),
           tf.concat([v[0][1] for v in vs], axis=0),
           tf.concat([v[0][2] + m for v,m in zip(vs, first_offset) ], axis=0),
           tf.concat([v[0][3] + m for v,m in zip(vs, second_offset) ], axis=0),
           tf.concat([ tf.cast( tf.zeros_like(vs[i][0][0][:,0]) + i, tf.int32) for i in range(batch_size) ], axis=0) ),
            tf.expand_dims(tf.stack([v[1] for v in vs], axis=0), axis=[1])
           )

def make_set():
    ds = tf.data.TFRecordDataset([args.eval])
    ds = ds.map(parse)
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(args.buf))
    it = ds.make_one_shot_iterator()
    with tf.device("/cpu:0"):
        return transformation_func(it, args.batch_size)


def make_trainset():
    ds = tf.data.TFRecordDataset([args.train])
    ds = ds.map(parse)
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(args.buf))
    it = ds.make_one_shot_iterator()
    with tf.device("/cpu:0"):
        return transformation_func(it, args.batch_size)

def make_testset():
    ds = tf.data.TFRecordDataset([args.test])
    ds = ds.map(parse)
    it = ds.make_one_shot_iterator()
    with tf.device("/cpu:0"):
        return transformation_func(it, args.batch_size)

def line_1(x1,x2):
    xmin=np.min(x1.tolist()+x2.tolist())
    xmax=np.max(x1.tolist()+x2.tolist())
    lines = plt.plot([1.1*xmin,1.1*xmax],[1.1*xmin,1.1*xmax])
    return lines

def fitquality (y,f):
    '''
    Computes $R^2$
    Args:
        x true label
        f predictions
    '''
    #r = np.corrcoef(np.squeeze(y),np.squeeze(f))
    #return r[0,1]
    #R2 = 1-np.var(f-y)/np.var(y)
    ssres=np.sum((y-f)**2)
    sstot=np.sum( (y-np.mean(y))**2 )
    R2 = 1-ssres/sstot

    return R2

class MessagePassing(tf.keras.Model):
    def __init__(self):
        super(MessagePassing, self).__init__()
        
        self.l = tf.keras.Sequential([
            tf.keras.layers.Dense(args.Mhid,activation=tf.nn.selu),
            tf.keras.layers.Dense(N_H*N_H),
            tf.keras.layers.Reshape((N_H,N_H))
        ])
        
        self.b = tf.keras.Sequential([
            tf.keras.layers.Dense(args.Mhid,activation=tf.nn.selu),
            tf.keras.layers.Dense(N_H)
        ])
        
        self.u = tf.keras.layers.GRUCell(N_H)
        
        self.i = tf.keras.Sequential([
            tf.keras.layers.Dense(args.rn,activation=tf.nn.tanh),
            tf.keras.layers.Dense(args.rn)
        ])

        self.j = tf.keras.Sequential([
            tf.keras.layers.Dense(args.rn,activation=tf.nn.selu),
            tf.keras.layers.Dense(args.rn)
        ])
        
        self.f = tf.keras.Sequential([
            tf.keras.layers.Dense(args.ninf,activation=tf.nn.selu),
            tf.keras.layers.Dense(1)            
        ])

    def build(self, input_shape=None):
        del input_shape
        self.l.build(tf.TensorShape([None, 1]))
        self.b.build(tf.TensorShape([None, 1]))
        self.u.build(tf.TensorShape([None, N_H]))
        self.i.build(tf.TensorShape([None, N_H+2]))
        self.j.build(tf.TensorShape([None, N_H+2]))
        self.j.build(tf.TensorShape([None, args.rn]))

        self.built = True
        
    def call(self, inputs, training=False):
        (x,e,first,second,segment) = inputs
        h=tf.pad(x,[[0,0],[0,N_PAD]])
        
        for i in range(N_PAS):
            m = self._M(tf.gather(h,first),e)
            num_segments=tf.cast(tf.reduce_max(second)+1,tf.int32)
            m = tf.unsorted_segment_sum(m,second,num_segments)
            h,_ = self.u(m,[h])
        node_batch = self._R(h,x,segment)
        return self.f(node_batch)
    
    def _M(self,h,e):
        a = self.l(e)
        m=tf.matmul(a,tf.expand_dims(h,axis=2) )
        m = tf.squeeze(m)
        b = self.b(e)
        return m + b
    
    def _R(self,h,x,segment):
        hx=tf.concat([h,x],axis=1)
        RR = tf.nn.sigmoid(self.i(hx))
        RR = tf.multiply(RR,self.j(hx))
        return tf.segment_sum(RR,segment)



if __name__== "__main__":
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    print(args)


    g=tf.Graph()

    with g.as_default():
        global_step = tf.train.get_or_create_global_step()

        ((x,e,first,second,segment),W)=make_trainset()

        model = MessagePassing()

        predictions = model((x,e,first,second,segment),training=True)
        labels=W
        loss= tf.losses.mean_squared_error(W,predictions)        
        rel = tf.reduce_mean(tf.abs( (labels-predictions)/labels) )

        trainables = model.variables
        grads = tf.gradients(loss, trainables)
        grad_var_pairs = zip(grads, trainables)
        
        summaries = [tf.summary.histogram(var.op.name, var) for var in trainables]
        summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]
        summaries.append(tf.summary.scalar('train_mse', loss)) 
        #summaries.append(tf.summary.scalar('train_relative_absolute_error', rel)) 
        summary_op = tf.summary.merge(summaries)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #o=tf.train.RMSPropOptimizer(learning_rate=0.001)
            #train = o.apply_gradients(grad_var_pairs)
            train=tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

        
      
        test_batch, test_labels = make_testset() 

        test_predictions = model(test_batch,training=False)
        test_relative = tf.abs( (test_labels-test_predictions)/(test_labels + args.W_shift/args.W_scale ) )
        mare = tf.reduce_mean(test_relative)

        test_summaries = [tf.summary.histogram('test_relative_absolute_error', test_relative)]
        test_summaries.append(tf.summary.scalar('test_mse', tf.reduce_mean( (test_labels-test_predictions)**2 ) ) )
        test_summary_op = tf.summary.merge(test_summaries)
        
        saver = tf.train.Saver(trainables + [global_step])

    with tf.Session(graph=g) as ses:
        ses.run(tf.local_variables_initializer())
        ses.run(tf.global_variables_initializer())

        ckpt=tf.train.latest_checkpoint(args.log_dir)
        if ckpt:
            print("Loading checkpint: %s" % (ckpt))
            tf.logging.info("Loading checkpint: %s" % (ckpt))
            saver.restore(ses, ckpt)


        writer=tf.summary.FileWriter(args.log_dir, ses.graph)

        for i in range(args.I):
            _,mse_loss,summary_py, step = ses.run([train,loss,summary_op, global_step])
            writer.add_summary(summary_py, global_step=step)
            
            if step % 100 ==0:
                test_label_py, test_predictions_py, test_summary_py = ses.run([test_labels, test_predictions, test_summary_op])
                #test_ae = np.abs((test_predictions_py-test_label_py)/test_label_py)
                test_error = test_predictions_py-test_label_py
                R2 = fitquality(test_label_py,test_predictions_py)

                print('{} step: {} train_mse: {}, test_mse: {} R**2: {}'.format(
                    str(datetime.datetime.now()),
                    step,
                    mse_loss,
                    np.mean(test_error**2),
                    #np.max(np.abs(test_error)),
                    R2 ), flush=True ) 
                
                writer.add_summary(test_summary_py, global_step=step)
                checkpoint_path = os.path.join(args.log_dir, 'model.ckpt')
                saver.save(ses, checkpoint_path, global_step=step)
                #make scatter plot
                fig = plt.figure()
                plt.plot(test_label_py,test_predictions_py,'.')
                line_1(test_label_py, test_label_py)
                plt.xlabel('test label')
                plt.ylabel('test predictions')
                plt.title(str(step))
                #fig_path = os.path.join(args.log_dir,'scatter-{0:08}.png'.format(step) )
                #plt.savefig(fig_path)
                with io.BytesIO() as buf:
                    w,h = fig.canvas.get_width_height()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plt.close()
                    summary = tf.Summary(value= [
                        tf.Summary.Value( tag="regression",
                            image=tf.Summary.Image(height = h, width =w, 
                                colorspace =3 , encoded_image_string = buf.read()) ),
                        tf.Summary.Value(tag="R2", simple_value=R2)
                        ])
                    writer.add_summary(summary, global_step=step)


        writer.flush()
        writer.close()
