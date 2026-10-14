import os, sys
import numpy as np, cv2, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import harness
from clouds.matching.matchers import create_matcher

cfg=harness.load_cfg(None)
pairs=harness.select_frames(cfg,'2025-05-23',t0='12:00:00',t1='12:10:00',step_s=600)
g1,g2,_,_=harness.prepare_crop(cfg,pairs[0][0],pairs[0][1])
print('stamp',pairs[0][2])
dev='cuda'
for stint,label in ((0.0,'same'),(5.0,'shift5')):
    gg = harness.subpixel_shift(g2, stint, 0.0)
    m=create_matcher('roma',{'grid_stride':8,'certainty_threshold':0.5},dev)
    k0,k1,conf=m(g1,gg,return_conf=True)
    k0=np.asarray(k0);k1=np.asarray(k1);conf=np.asarray(conf)
    du=k0[:,0]-k1[:,0]; dv=k0[:,1]-k1[:,1]
    print(f"{label}: n={len(k0)} conf med={np.median(conf):.3f} "
          f"du med={np.median(du):.2f} (expect ~-{stint}) dv med={np.median(dv):.2f} "
          f"du IQR={np.percentile(du,75)-np.percentile(du,25):.2f}")
print('--- certainty quantiles all pixels ---')
m=create_matcher('roma',{'grid_stride':8,'certainty_threshold':0.0},dev)
k0,k1,conf=m(g1,g2,return_conf=True)
conf=np.asarray(conf)
print('n',len(conf),'q', np.percentile(conf,[5,25,50,75,95,99]))
