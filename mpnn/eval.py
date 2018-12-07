import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime
import argparse
import os

import graph_nn


args = graph_nn.args

def make_set():
    #filename_queue = tf.train.string_input_producer( ['test.tfrecords'])
    #reader = tf.TFRecordReader()
    #_, serialized_example = reader.read(filename_queue)
    #serialized_batch= tf.train.batch( [serialized_example], batch_size=200)
    ds = tf.data.TFRecordDataset([args.eval])
    ds = ds.batch(args.batch_size)
    serialized_batch = ds.make_one_shot_iterator().get_next()
    return serialized_batch


def main():
    REUSE=None
    g=tf.Graph()

    with g.as_default():
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope('model'):
            serialized_batch = make_set()
            batch, labels = graph_nn.make_batch(serialized_batch)
            n_batch = tf.layers.batch_normalization(batch) 
            predictions = graph_nn.inference(n_batch)

        loss= tf.losses.mean_squared_error(labels,predictions)        
        
        saver = tf.train.Saver(tf.trainable_variables() + [global_step])

    with tf.Session(graph=g) as ses:
        ses.run(tf.local_variables_initializer())
        ses.run(tf.global_variables_initializer())

        ckpt=tf.train.latest_checkpoint(args.log_dir)
        if ckpt:
            print("Loading checkpint: %s" % (ckpt))
            tf.logging.info("Loading checkpint: %s" % (ckpt))
            saver.restore(ses, ckpt)
        
        label_py=[]
        predictions_py=[]

        for i in range(16):
            val_label_py, val_predictions_py, step = ses.run( [labels,predictions, global_step] )
            label_py.append(val_label_py)
            predictions_py.append(val_predictions_py)

        label_py = np.concatenate(label_py,axis=0)
        predictions_py = np.concatenate(predictions_py,axis=0)
        print(label_py.shape)
        print('{} step: {} mse: {} R**2: {} Pearson: {}'.format(
            str(datetime.datetime.now()),
            step,
            np.mean((label_py-predictions_py)**2),
            #np.max(np.abs(test_error)),
            graph_nn.fitquality(label_py,predictions_py),
            np.corrcoef(label_py,predictions_py, rowvar=False)[0,1] ), flush=True ) 

        plt.figure()
        plt.plot(label_py,predictions_py,'.')
        graph_nn.line_1(label_py, label_py)
        plt.grid('on')
        plt.xlabel('Label')
        plt.ylabel('Prediction')
        plt.title('Evaluation at step {}'.format(step))
        fig_path = os.path.join(args.log_dir,'eval-{0:08}.png'.format(step) )
        fig_path = 'eval.pdf'.format(step)
        plt.savefig(fig_path)
        plt.close()

        plt.figure()
        plt.hist(label_py-predictions_py,50)
        fig_path = 'rez_hist.pdf'
        plt.savefig(fig_path)
        plt.close()


if __name__ == '__main__':
    main()