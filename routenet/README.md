# Dataset
Datasets available at [KDN website](http://knowledgedefinednetworking.org/)
For training simulation, data must be converted to TFrecords. The script provides a function for this.


```{python}
import glob
import os
upc = importlib.reload(routenet)

for fname in glob.glob('geant2/delaysGeant2/*.txt'):
    tfname = fname.replace('txt','tfrecords')
    upc.make_tfrecord2(tfname,
                      'routingNSFNET/NetworkNsfnet.ned',
                       upc.infer_routing_nsf3(fname),
                       fname
                      )
```
# Train

```{bash}
python3 routenet.py train --model_dir ./log \
  --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=16,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8"   \
  --train  nsfnet/tfrecords/train/*.tfrecords \
  --eval_  nsfnet/tfrecords/evaluate/*.tfrecords \
  --train_steps 1000 \
  --shuffle_buf 30000
```