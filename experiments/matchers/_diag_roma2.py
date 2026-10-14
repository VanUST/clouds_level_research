import os,sys
import numpy as np, cv2, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import harness
from clouds.matching.matchers import create_matcher
from PIL import Image
from romatch import roma_outdoor

cfg=harness.load_cfg(None)
pairs=harness.select_frames(cfg,'2025-05-23',t0='12:00:00',t1='12:10:00',step_s=600)
g1,g2,_,_=harness.prepare_crop(cfg,pairs[0][0],pairs[0][1])
dev=torch.device('cuda')
model=roma_outdoor(device=dev, amp_dtype=torch.float16, use_custom_corr=False)

def raw(ga,gb,tag):
    A=Image.fromarray(cv2.cvtColor(ga,cv2.COLOR_GRAY2RGB))
    B=Image.fromarray(cv2.cvtColor(gb,cv2.COLOR_GRAY2RGB))
    with torch.no_grad():
        w,c=model.match(A,B)
    w=w[0].cpu().numpy(); c=c[0].cpu().numpy()
    hw=w.shape[1]//2
    ab=w[:,:hw]; cert=c[:,:hw]
    q=np.percentile(cert,[10,50,90])
    # slice near center
    i,j=432,432
    print(f"{tag}: warp shape {w.shape} cert q10/50/90={q}")
    print(f"   center cell A->B: xA,yA,dx,dy = {ab[i,j]}")
    print(f"   second-half cell : {w[i,hw+j]}")
    return ab,cert

ab0,c0=raw(g1,g1,'identity')
ab1,c1=raw(g1,g2,'pair g1,g2')
# predicted flow stats over high-cert pixels
for tag,ab,c in (('identity',ab0,c0),('pair',ab1,c1)):
    m=c>0.5
    dx=ab[...,2][m]; dy=ab[...,3][m]
    print(f"{tag}: high-cert n={m.sum()} dx med={np.median(dx)*320:.2f}px dy med={np.median(dy)*320:.2f}px")
