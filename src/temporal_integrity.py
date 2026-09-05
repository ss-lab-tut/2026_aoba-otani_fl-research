"""Reconstruct observed window order and purge cross-split frame sharing."""
import hashlib

import numpy as np


def reconstruct_order(windows, hop=16):
    x=np.asarray(windows)
    if x.ndim!=4 or not 0<hop<x.shape[1]:
        raise ValueError('require N x time x beam x channel and valid hop')
    def digest(a):
        return hashlib.sha256(np.ascontiguousarray(a[:,:,0]).tobytes()).digest()
    prefixes={}
    for i,w in enumerate(x):
        key=digest(w[:-hop])
        if key in prefixes:
            raise ValueError('ambiguous repeated window prefixes; metadata required')
        prefixes[key]=i
    successors={i:prefixes[digest(w[hop:])] for i,w in enumerate(x) if digest(w[hop:]) in prefixes}
    if len(set(successors.values()))!=len(successors):
        raise ValueError('ambiguous predecessor mapping')
    starts=set(range(len(x)))-set(successors.values())
    if len(starts)!=1:
        raise ValueError('expected one reconstructable chain; recording metadata required')
    order=[starts.pop()]
    while order[-1] in successors:
        successor=successors[order[-1]]
        if successor in order:
            raise ValueError('cyclic overlap graph')
        order.append(successor)
    if len(order)!=len(x):
        raise ValueError('not all windows belong to the chain')
    return np.array(order,dtype=int)


def purged_split(order, fold, excluded=(), n_blocks=20, hop=16, window_frames=64):
    """Interleaved time blocks; never call this subject-independent evaluation."""
    order=np.asarray(order)
    if not 0<=fold<5 or n_blocks<5 or len(order)<n_blocks or len(set(order))!=len(order):
        raise ValueError('invalid fold, blocks or order')
    edges=np.linspace(0,len(order),n_blocks+1,dtype=int)
    splits={'train':[],'calibration':[],'test':[]}
    dropped=[]; excluded=set(excluded)
    gap=int(np.ceil(window_frames/hop))-1
    for b,(start,end) in enumerate(zip(edges[:-1],edges[1:])):
        destination='test' if b%5==fold else 'calibration' if b%5==(fold+1)%5 else 'train'
        for position in range(start,end):
            index=int(order[position])
            # Last windows in a block extend into the next block.
            if index in excluded or (b<n_blocks-1 and position>=end-gap):
                dropped.append(index)
            else:
                splits[destination].append(index)
    verify_no_frame_overlap(order,splits,hop,window_frames)
    return dict(**splits,dropped=dropped,fold=fold,n_blocks=n_blocks)


def verify_no_frame_overlap(order,splits,hop=16,window_frames=64):
    positions={int(index):p for p,index in enumerate(order)}
    owners={}
    for name in ('train','calibration','test'):
        for index in splits[name]:
            start=positions[int(index)]*hop
            for frame in range(start,start+window_frames):
                if frame in owners and owners[frame]!=name:
                    raise ValueError('shared raw frame across splits')
                owners[frame]=name
