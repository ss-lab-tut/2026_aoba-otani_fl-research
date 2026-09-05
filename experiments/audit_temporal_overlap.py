"""Find exact informative frame-block sharing; never infer subject identity.

Components conservatively group windows connected by shared observed blocks.
They are overlap groups, not verified recording/person/site identifiers.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def overlap_components(arrays, block_frames=16, minimum_unique_frames=4):
    if not arrays or block_frames<=0:
        raise ValueError('arrays and positive block size required')
    locations=[(split,i) for split,x in enumerate(arrays) for i in range(len(x))]
    parent=list(range(len(locations)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    owners={}; shared=[]; excluded=0
    for global_i,(split,i) in enumerate(locations):
        window=arrays[split][i]
        if window.ndim!=3 or len(window)%block_frames:
            raise ValueError('expected time x beam x channel with divisible block length')
        # Use the first observed input channel. Its physical meaning is not
        # confirmed; a window-local transform of the second channel cannot
        # conceal sharing of this first channel.
        for start in range(0,len(window),block_frames):
            block=np.ascontiguousarray(window[start:start+block_frames,:,0])
            if len({hashlib.sha256(frame.tobytes()).digest() for frame in block})<minimum_unique_frames:
                excluded+=1;continue
            digest=hashlib.sha256(block.tobytes()).hexdigest()
            if digest in owners:
                other=owners[digest]
                a,b=root(global_i),root(other)
                parent[a]=b
                if locations[other][0]!=split:
                    shared.append(dict(first=list(locations[other]),second=[split,i],block_sha256=digest))
            else: owners[digest]=global_i
    roots=[root(i) for i in range(len(locations))]
    mapping={k:i for i,k in enumerate(sorted(set(roots)))}
    groups=np.array([mapping[k] for k in roots])
    crossing=set(groups[sum(map(len,arrays[:item['first'][0]]))+item['first'][1]] for item in shared)
    return dict(group_ids=groups.tolist(),locations=[list(x) for x in locations],
                shared_cross_split_blocks=shared,cross_split_group_ids=sorted(map(int,crossing)),
                n_components=len(mapping),uninformative_blocks_skipped=excluded,
                block_frames=block_frames,minimum_unique_frames=minimum_unique_frames,
                limitation='Exact block overlap components do not establish person, recording or facility independence.')


def main():
    root=Path(__file__).resolve().parents[1]
    names=['X_train.npy','X_val_real.npy']
    arrays=[np.load(root/n,mmap_mode='r') for n in names]
    r=overlap_components(arrays)
    r['files']=names
    ntrain=len(arrays[0]); ids=np.array(r['group_ids'])
    crossing=set(r['cross_split_group_ids'])
    r['train_windows_in_cross_split_components']=sum(int(g in crossing) for g in ids[:ntrain])
    r['validation_windows_in_cross_split_components']=sum(int(g in crossing) for g in ids[ntrain:])
    r['validation_windows_with_direct_shared_blocks']=len({item['second'][1] for item in r['shared_cross_split_blocks'] if item['second'][0]==1})
    counts=np.bincount(ids)
    r['component_sizes']=counts.tolist()
    r['summary']=dict(components=len(counts),max_component=int(counts.max()),
                      validation_shared=r['validation_windows_in_cross_split_components'],
                      validation_total=len(arrays[1]))
    out=root/'results/temporal_overlap_audit.json'
    out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r['summary'],indent=2))


if __name__=='__main__': main()
