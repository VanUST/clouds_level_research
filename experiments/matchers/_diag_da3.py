import os,sys,time
import numpy as np, cv2, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import harness
# DA3's api.py imports its Gaussian-splat export helpers at module import time;
# those need moviepy/gsplat encoders irrelevant to matching. A stub export module
# keeps the depth/pose API importable without those optional dependencies.
import types as _types
_m = _types.ModuleType('depth_anything_3.utils.export')
_m.export = lambda *a, **k: None
sys.modules['depth_anything_3.utils.export'] = _m
from depth_anything_3.api import DepthAnything3
cfg=harness.load_cfg(None)
pairs=harness.select_frames(cfg,'2025-05-23',t0='12:00:00',t1='12:10:00',step_s=600)
g1,g2,_,_=harness.prepare_crop(cfg,pairs[0][0],pairs[0][1])
rgb1=cv2.cvtColor(g1,cv2.COLOR_GRAY2RGB); rgb2=cv2.cvtColor(g2,cv2.COLOR_GRAY2RGB)
t=time.time()
model=DepthAnything3.from_pretrained('/home/iustimov/da3_base').to('cuda').eval()
print('loaded %.1fs'%(time.time()-t))
t=time.time()
with torch.no_grad():
    pred=model.inference([rgb1,rgb2], process_res=504)
print('inference %.1fs'%(time.time()-t))
print('is_metric',pred.is_metric,'scale_factor',pred.scale_factor)
print('depth',None if pred.depth is None else pred.depth.shape, 'conf', None if pred.conf is None else pred.conf.shape)
print('sky',None if pred.sky is None else np.asarray(pred.sky).shape)
print('extr',None if pred.extrinsics is None else pred.extrinsics.shape)
print('intr',None if pred.intrinsics is None else pred.intrinsics.shape)
if pred.extrinsics is not None:
    E=np.asarray(pred.extrinsics); print('ext0\n',E[0]); print('ext1\n',E[1]);
    R0,t0=E[0][:3,:3],E[0][:3,3]; R1,t1=E[1][:3,:3],E[1][:3,3]
    Rrel=R1@R0.T; trel=t1-Rrel@t0
    print('relative translation',trel,'norm',np.linalg.norm(trel))
if pred.depth is not None:
    d=np.asarray(pred.depth)
    print('depth per view median', [float(np.median(d[i])) for i in range(len(d))])
    print('depth q', [np.percentile(d[i],[5,50,95]).round(3).tolist() for i in range(len(d))])
