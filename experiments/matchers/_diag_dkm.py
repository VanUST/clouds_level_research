import os,sys
import numpy as np, cv2, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import harness
from PIL import Image
from dkm import DKMv3_outdoor
cfg=harness.load_cfg(None)
pairs=harness.select_frames(cfg,'2025-05-23',t0='12:00:00',t1='12:10:00',step_s=600)
g1,g2,_,_=harness.prepare_crop(cfg,pairs[0][0],pairs[0][1])
dev=torch.device('cuda')
model=DKMv3_outdoor(device=dev)
print('h_resized',model.h_resized,'w_resized',model.w_resized,'upsample',getattr(model,'upsample_res',None))
def stats(ga,gb,tag):
    A=Image.fromarray(cv2.cvtColor(ga,cv2.COLOR_GRAY2RGB)); B=Image.fromarray(cv2.cvtColor(gb,cv2.COLOR_GRAY2RGB))
    with torch.no_grad(): w,c=model.match(A,B)
    w=w.detach().cpu().numpy(); c=c.detach().cpu().numpy()
    hh,ww=w.shape[0],w.shape[1]; hw=ww//2
    ab=w[:,:hw]; cert=c[:,:hw]
    # normalized -> pixels using the matcher's own grid dims (ws for x, hs for y)
    sx=640.0/ww*2.0* (hw/ww)  # not used; map normalized directly to the ORIGINAL 640
    print(f"{tag}: warp{w.shape} cert{c.shape} certmed={np.median(cert):.3f} frac>0.5={(cert>0.5).mean():.2f}")
    m=cert>0.5
    for name,ch in (('ch0',0),('ch2',2)):
        v=ab[...,ch][m]
        print(f"   {name}: med={np.median(v):.4f} q10={np.percentile(v,10):.3f} q90={np.percentile(v,90):.3f}")
    d_abs=(ab[...,2]-ab[...,0])[m]
    d_flow=ab[...,2][m]
    print(f"   abshyp(x) med={np.median(d_abs):.4f}  flowhyp(x) med={np.median(d_flow):.4f}")
stats(g1,g1,'identity')
stats(g1,harness.subpixel_shift(g2,5.0,0.0),'Bshift+5')
stats(g1,g2,'pair')
