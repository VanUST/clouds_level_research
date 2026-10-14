import os,sys
import numpy as np, cv2, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import harness
from PIL import Image
from romatch import roma_outdoor
cfg=harness.load_cfg(None)
pairs=harness.select_frames(cfg,'2025-05-23',t0='12:00:00',t1='12:10:00',step_s=600)
g1,g2,_,_=harness.prepare_crop(cfg,pairs[0][0],pairs[0][1])
dev=torch.device('cuda')
model=roma_outdoor(device=dev, amp_dtype=torch.float16, use_custom_corr=False)
def flow(ga,gb,tag):
    A=Image.fromarray(cv2.cvtColor(ga,cv2.COLOR_GRAY2RGB)); B=Image.fromarray(cv2.cvtColor(gb,cv2.COLOR_GRAY2RGB))
    with torch.no_grad(): w,c=model.match(A,B)
    w=w[0].cpu().numpy(); c=c[0].cpu().numpy(); hw=w.shape[1]//2
    ab=w[:,:hw]; cert=c[:,:hw]
    dx=ab[...,2]*320.0; dy=ab[...,3]*320.0
    print(f"{tag}: shape {dx.shape} cert med={np.median(cert):.3f}")
    for q in (1,10,25,50,75,90,99):
        print(f"    q{q}: dx={np.percentile(dx,q):8.2f} dy={np.percentile(dy,q):8.2f}")
    m=cert>0.5
    print(f"    frac|cert>0.5|={m.mean():.2f}  within2px of 0: dx {np.mean(np.abs(dx[m])<2):.2f} dy {np.mean(np.abs(dy[m])<2):.2f}")
flow(g1,g1,'identity')
flow(g1,g2,'pair')
