import os,sys
import numpy as np, cv2, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import harness
from clouds.matching.matchers import create_matcher
cfg=harness.load_cfg(None)
m=create_matcher('mast3r',{},'cuda')
for date,t0 in (('2025-05-23','12:00:00'),('2025-05-22','12:00:00')):
    pairs=harness.select_frames(cfg,date,t0=t0,t1='12:10:00',step_s=600)
    g1,g2,_,_=harness.prepare_crop(cfg,pairs[0][0],pairs[0][1])
    k0,k1,c=m(g1,g2,return_conf=True)
    c=np.asarray(c)
    print(f"{date} {pairs[0][2]}: n={len(k0)} conf percentiles "
          f"{np.percentile(c,[1,10,25,50,75,90,99]).round(2).tolist()}")
