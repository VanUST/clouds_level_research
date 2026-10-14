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
def stats(ga,gb,tag):
    A=Image.fromarray(cv2.cvtColor(ga,cv2.COLOR_GRAY2RGB)); B=Image.fromarray(cv2.cvtColor(gb,cv2.COLOR_GRAY2RGB))
    with torch.no_grad(): w,c=model.match(A,B)
    w=w[0].cpu().numpy(); c=c[0].cpu().numpy(); hw=w.shape[1]//2
    ab=w[:,:hw]; cert=c[:,:hw]
    c0=ab[...,0]*320; c1=ab[...,1]*320; c2=ab[...,2]*320; c3=ab[...,3]*320
    m=cert>0.5
    # hypothesis A: ch2,3 are absolute B coords -> disparity = ch2-ch0
    disp_x_abs=(c2-c0)[m]; disp_y_abs=(c3-c1)[m]
    # hypothesis B: ch2,3 are flow -> disparity = ch2
    disp_x_flow=c2[m]
    print(f"{tag}: certmed={np.median(cert):.3f} frac>0.5={m.mean():.2f}")
    print(f"   ch0 range [{c0.min():.0f},{c0.max():.0f}] ch2 range [{c2.min():.0f},{c2.max():.0f}]")
    print(f"   |ch2-ch0| med {np.median(np.abs(disp_x_abs)):.2f}   |ch2| med {np.median(np.abs(disp_x_flow)):.2f}")
    print(f"   disp_abs med d=({np.median(disp_x_abs):.2f},{np.median(disp_y_abs):.2f})")
stats(g1,g1,'identity')
stats(g1,harness.subpixel_shift(g2,5.0,0.0),'pair Bshift+5')
stats(g1,g2,'pair')
