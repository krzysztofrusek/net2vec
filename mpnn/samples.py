import networkx as nx
import numpy as np
import  scipy as sp
import tensorflow as tf
import argparse
import datetime
import glob
import os
import sys

sndlib_networks = None

class GraphProvider:
    def get(self):
        G = self._get()
        G=nx.convert_node_labels_to_integers(G)
        return G

class BarabasiAlbert(GraphProvider):
    def __init__(self,n):
        self.n = n
        self.nmin=10
        self.m = 2
    def _get(self):
        return nx.barabasi_albert_graph(np.random.randint(self.nmin,self.n),self.m)

class ErdosReni(GraphProvider):
    def __init__(self,n):
        self.n = n
        self.p =  2.0/n
    def _get(self):
        G=nx.fast_gnp_random_graph(self.n,self.p,directed=False)
        largest_cc = max(nx.connected_components(G), key=len)
        Gm=G.subgraph(largest_cc)
        return Gm

class SNDLib(GraphProvider):
    def __init__(self,flist):
        self.sndlib_networks = {os.path.split(f)[1][0:-8]:nx.read_graphml(f) for f in flist}
        # UPC hack
        self.sndlib_networks = {k:v for k,v in self.sndlib_networks.items() if len(v) < 38 and len(v) > 19}
        self.names = list(self.sndlib_networks.keys())
        print(self.names)

    def _get(self):
        name = np.random.choice(self.names)
        Gm = nx.Graph( self.sndlib_networks[name] )
        return Gm


def make_sample(provider, rl=0.3, rh=0.7):
    Gm=provider.get()
    A=nx.convert_matrix.to_numpy_matrix(Gm)

    # Make all intensities addup to 1
    L=np.random.uniform(size=(len(Gm),1))
    L = L /np.sum(L)
    p=1.0/(np.sum(A,axis=1)+1.0)
    R=np.multiply(A,p)


    lam=np.linalg.solve(np.identity(len(Gm))-np.transpose( R ) ,L)
    #random utilisation of each node
    rho=np.random.uniform(low=rl,high=rh, size=lam.shape)
    # Beta make higher util more probable, P(rho=1)=0
    #rho = np.random.beta(20,2,size=lam.shape)
    #rho = 0.9 * np.ones(shape=lam.shape)
    mu = lam/rho
    ll=rho/(1-rho)
    W=np.sum(ll)/np.sum(L)

    #  Max value of W is of order n*0.99/(1 -0.99)

    nx.set_node_attributes(Gm, name='mu', values=dict(zip(Gm.nodes(),np.ndarray.tolist(mu[:,0]))))
    nx.set_node_attributes(Gm, name='Lambda', values=dict(zip(Gm.nodes(),np.ndarray.tolist(L[:,0]))))
    it=np.nditer(R, order='F', flags=['multi_index'])
    at = {it.multi_index:float(x) for x in it if x > 0}
    nx.set_edge_attributes(Gm,name='R', values=at)
    Gm.graph['W']=W

    return mu,L,R,W,Gm

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def make_dataset(count, file, producer):
    #n=10
    #p=0.2
    writer = tf.python_io.TFRecordWriter(file)
    for i in range(count):
        if not i % 500:
            print('{} generated {} samples.'.format(str(datetime.datetime.now()) , i ) )

        mu,L,R,W,Gm=producer()
        #while W > 3.3:
        #    mu,L,R,W,Gm=make_sample(n,p)

        mu = mu[:,0].tolist()
        L = L[:,0].tolist()
        first,last=np.nonzero(R)
        e=R[first,last].tolist()[0]

        example = tf.train.Example(features=tf.train.Features(feature={
            'mu': _float_feature(mu),
            'Lambda': _float_feature(L),
            'W':_float_feature([W]),
            'R':_float_feature(e),
            'first':_int64_feature(first.tolist()),
            'second':_int64_feature(last.tolist()) }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ =='__main__':
    random_org_help='''Seed, if none, downloads from random.org'''

    parser = argparse.ArgumentParser(description='Generates saple networks')
    parser.add_argument('-N', help='number of samples',  required=True, type=int)
    parser.add_argument('-n', help='number of nodes',  default=40, type=int)
    parser.add_argument('-o', help='Output file',  required=True, type=str)
    parser.add_argument('--rmin', help='Min rho',  type=float, default=0.3)
    parser.add_argument('--rmax', help='max rho',  type=float, default=0.7)
    parser.add_argument('-s', help=random_org_help,  required=False, type=int)
    parser.add_argument('-g', help='random graph type: [ba | er | snd]',  type=str, default="ba")
    parser.add_argument('--sndlib', help='Sndlib files',  type=str ,nargs='+')
    args = parser.parse_args()

    if args.s is None:
        import urllib.request
        with urllib.request.urlopen('https://www.random.org/integers/?num=1&min=0&max=1000000&col=1&base=10&format=plain&rnd=new') as response:
            rnd_seed = int(response.read())
            print( str(datetime.datetime.now()), "Random response: {}".format(rnd_seed))
            np.random.seed(rnd_seed)
    else:
        np.random.seed(args.s)
    provider = None
    if args.g == 'er':
        provider = ErdosReni(args.n)
    elif args.g == 'ba':
        provider = BarabasiAlbert(args.n)
    elif args.g == 'snd':
        provider = SNDLib(args.sndlib)
    

    make_dataset(args.N,args.o, lambda: make_sample(provider, args.rmin, args.rmax))
