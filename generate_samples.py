from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from utils.degradation import degrade
from utils.image_utils import save_image
from config import ROOT
def generate():
 h=w=256; y,x=np.mgrid[0:h,0:w]; a=np.zeros((h,w,3),np.float32);a[...,0]=x/w*.8+.1;a[...,1]=y/h*.7+.15;a[...,2]=.25+.5*(1-x/w); a[((x-125)**2+(y-110)**2)<45**2]=[.95,.72,.18]
 p=ROOT/'data/sample';save_image(p/'clean.png',a);save_image(p/'degraded.png',degrade(a,seed=42));return p/'clean.png',p/'degraded.png'
if __name__=='__main__':print(*generate())
