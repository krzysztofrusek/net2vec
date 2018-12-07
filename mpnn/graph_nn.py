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

def M(h,e):
    with tf.variable_scope('message'):
        bs = tf.shape(h)[0]
        l = tf.layers.dense(e,args.Mhid ,activation=tf.nn.selu)
        l = tf.layers.dense(l,N_H*N_H)
        l=tf.reshape(l,(bs,N_H,N_H))
        m=tf.matmul(l,tf.expand_dims(h,dim=2) )
        m=tf.reshape(m,(bs,N_H))
        b = tf.layers.dense(e,args.Mhid ,activation=tf.nn.selu)
        b = tf.layers.dense(b,N_H)
        m = m + b


        return m
def U(h,m,x):
    init = tf.truncated_normal_initializer(stddev=0.01)
    with tf.variable_scope('update'):
        wz=tf.get_variable(name='wz',shape=(N_H,N_H),dtype=tf.float32)
        uz=tf.get_variable(name='uz',shape=(N_H,N_H),dtype=tf.float32)
        wr=tf.get_variable(name='wr',shape=(N_H,N_H),dtype=tf.float32)
        ur=tf.get_variable(name='ur',shape=(N_H,N_H),dtype=tf.float32)
        W=tf.get_variable(name='W',shape=(N_H,N_H),dtype=tf.float32)
        U=tf.get_variable(name='U',shape=(N_H,N_H),dtype=tf.float32)
        
        z = tf.nn.sigmoid(tf.matmul(m,wz) + tf.matmul(h,uz))
        r = tf.nn.sigmoid(tf.matmul(m,wr) + tf.matmul(h,ur))
        h_tylda = tf.nn.tanh(tf.matmul(m,W) + tf.matmul(r*h,U) )
        u = (1.0-z)*h + z*h_tylda
        return u

def R(h,x):
    with tf.variable_scope('readout'):
        hx=tf.concat([h,x],axis=1)
        i = tf.layers.dense(hx,args.rn,activation=tf.nn.tanh)
        i = tf.layers.dense(i,args.rn)
        j = tf.layers.dense(h,args.rn,activation=tf.nn.selu)
        j = tf.layers.dense(j,args.rn)

        RR = tf.nn.sigmoid(i)
        RR = tf.multiply(RR,j)

        return tf.reduce_sum(RR,axis=0)

def graph_features(x,e,first,second):
    global REUSE
    
    h=tf.pad(x,[[0,0],[0,N_PAD]])
    #bs = tf.shape(x)[0]
    #h=tf.random_gamma((bs,N_H),2,2)
    #initializer =tf.truncated_normal_initializer(0.0, 0.2)
    initializer =tf.contrib.layers.xavier_initializer()
    for i in range(N_PAS):
        with tf.variable_scope('features',
        reuse=REUSE, 
        initializer=initializer,
        #regularizer=tf.contrib.layers.l2_regularizer(0.00000000001)
        ) as scope:
            to_stack=[
                #tf.gather(x,first),
                tf.gather(h,first),
                e,
                tf.gather(h,second),
                #tf.gather(x,second),
            ]
            
            m=M(tf.gather(h,first),e)
            #Suma wplywajacych do wezla
            #czemu to dziala ?
            #m = tf.segment_sum(m,first) 
            #TODO wyjasnic
            #TODO num_segments jako cecha
            
            num_segments=tf.cast(tf.reduce_max(second)+1,tf.int32)
            m = tf.unsorted_segment_sum(m,second,num_segments)
            h = U(h,m,x)

            REUSE=True
        

    return R(h,x)

def inference(batch,reuse=None):
    #initializer =tf.truncated_normal_initializer(0.0, 0.002)
    initializer =tf.contrib.layers.xavier_initializer()
    with tf.variable_scope("inference",
    reuse=reuse,
    #regularizer=tf.contrib.layers.l2_regularizer(0.00000000000003),
    initializer=initializer):
        l=batch
        l=tf.layers.dense(l, args.ninf, activation=tf.nn.selu)
        l=tf.layers.dense(l,1)
        return l
    
def make_batch(serialized_batch):
    bs = tf.shape(serialized_batch)[0]

    to=tf.TensorArray(tf.float32,size=bs)
    labelto=tf.TensorArray(tf.float32,size=bs)

    condition = lambda i,a1,a2: i < bs
    def body(i,to,lto):
        with tf.device("/cpu:0"):
            #Wypakowanie przykladu1
            with tf.name_scope('load'):    
                features = tf.parse_single_example(
                serialized_batch[i],
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
            
        g_feature = graph_features(x,e,first,second) 
        W = (features['W']-args.W_shift)/args.W_scale # 0.7-0.9

        return i+1,to.write(i,g_feature ),lto.write(i,W)
    
    with tf.control_dependencies([serialized_batch]):
        _,batch,labelst = tf.while_loop(condition,body,[tf.constant(0),to,labelto])
        batch = batch.stack()
        labels = labelst.stack()
        labels = tf.reshape(labels,[bs,1])
    return batch, labels
def make_trainset():
    filename_queue = tf.train.string_input_producer( [args.train])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    serialized_batch= tf.train.shuffle_batch( [serialized_example], 
                                                  batch_size=batch_size, capacity=args.buf, min_after_dequeue=batch_size, num_threads=2)
    return serialized_batch
def make_testset():
    filename_queue = tf.train.string_input_producer( [args.test])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    serialized_batch= tf.train.batch( [serialized_example], batch_size=200)
    
    return serialized_batch

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

if __name__== "__main__":
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    REUSE=None
    g=tf.Graph()

    with g.as_default():
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope('model'):
            serialized_batch = make_trainset()
            batch, labels = make_batch(serialized_batch)
            n_batch = tf.layers.batch_normalization(batch) 
            predictions = inference(n_batch)

        loss= tf.losses.mean_squared_error(labels,predictions)        
        rel = tf.reduce_mean(tf.abs( (labels-predictions)/labels) )

        trainables = tf.trainable_variables()
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

        
        if False:
            trainables = tf.trainable_variables()
            grads = tf.gradients(loss, trainables)
            grads, gg = tf.clip_by_global_norm(grads, clip_norm=1.0)
            grad_var_pairs = zip(grads, trainables)

            gs = tf.Variable(0, trainable=False, dtype=tf.int32)
            lr = tf.train.exponential_decay(
                0.01, gs, 30,
                0.999, staircase=True)
            o=tf.train.GradientDescentOptimizer(learning_rate=0.01)
            #o=tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.1)
            #o=tf.train.RMSPropOptimizer(learning_rate=0.001)
            train = o.apply_gradients(grad_var_pairs,global_step=gs)
        
        # Evaluation
        with tf.variable_scope('model', reuse=True):
            test_batch, test_labels = make_batch(make_testset()) 
            test_batch = tf.layers.batch_normalization(test_batch,reuse=True)
            test_predictions = inference(test_batch,reuse=True)
        test_relative = tf.abs( (test_labels-test_predictions)/(test_labels + args.W_shift/args.W_scale ) )
        mare = tf.reduce_mean(test_relative)

        test_summaries = [tf.summary.histogram('test_relative_absolute_error', test_relative)]
        #test_summaries.append(tf.summary.scalar('test_mean_are', mare ) )
        #test_summaries.append(tf.summary.scalar('test_max_are', tf.reduce_max(test_relative) ) )
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



        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=ses, coord=coord)
        writer=tf.summary.FileWriter(args.log_dir, ses.graph)

        try:
            while not coord.should_stop():
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



                    if step > args.I:
                        coord.request_stop()
        except tf.errors.OutOfRangeError:
            print('OutOfRange' )

        finally:
            coord.request_stop()

        coord.join(threads)
        writer.flush()
        writer.close()
