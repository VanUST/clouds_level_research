import os,sys,time
import numpy as np, cv2, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import harness
sys.path.insert(0,'/home/iustimov/mast3r')
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import extract_correspondences_nonsym
from dust3r.inference import inference
from dust3r.utils.image import load_images

cfg=harness.load_cfg(None)
pairs=harness.select_frames(cfg,'2025-05-23',t0='12:00:00',t1='12:10:00',step_s=600)
g1,g2,_,_=harness.prepare_crop(cfg,pairs[0][0],pairs[0][1])
tmp='/tmp/mast3r_in'; os.makedirs(tmp,exist_ok=True)
p1=os.path.join(tmp,'a.png'); p2=os.path.join(tmp,'b.png')
cv2.imwrite(p1,g1); cv2.imwrite(p2,g2)
dev='cuda'
t=time.time()
model=AsymmetricMASt3R.from_pretrained('/home/iustimov/mast3r/weights').to(dev).eval()
print('loaded %.1fs'%(time.time()-t))
imgs=load_images([p1,p2], size=512, square_ok=True, verbose=False)
print('img shapes', [im['img'].shape for im in imgs], 'true_shape', [im['true_shape'].tolist() for im in imgs])
t=time.time()
out=inference([(imgs[0],imgs[1])],model,dev,batch_size=1,verbose=False)
print('inference %.2fs'%(time.time()-t))
pred1,pred2=out['pred1'],out['pred2']
print('pred1 keys', list(pred1.keys()))
for k in pred1:
    try: print('   ',k, tuple(pred1[k].shape))
    except Exception: print('   ',k,type(pred1[k]))
d1=pred1['desc'][0].cpu().numpy(); d2=pred2['desc'][0].cpu().numpy()
c1=pred1['desc_conf'][0].cpu().numpy(); c2=pred2['desc_conf'][0].cpu().numpy()
print('desc',d1.shape,c1.shape, 'c range',np.percentile(c1,[5,50,95]))
xy1,xy2,conf=extract_correspondences_nonsym(pred1["desc"][0],pred2["desc"][0],c1,c2,subsample=8,device=dev,ptmap_key="desc",pixel_tol=0)
print('corres', xy1.shape, xy2.shape, conf.shape)
_t=lambda z: z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
xy1=_t(xy1); xy2=_t(xy2); conf=_t(conf)
du=xy1[:,0]-xy2[:,0]; dv=xy1[:,1]-xy2[:,1]
print('du med %.2f IQR %.2f  dv med %.2f   (grid 512, scale to 640 = x1.25)'%(np.median(du),np.percentile(du,75)-np.percentile(du,25),np.median(dv)))
print('du*1.25 med %.2f px'%(np.median(du)*640/512))
