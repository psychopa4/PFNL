import os
from model.vespcn import VESPCN
from model.ltdvsr import LTDVSR
from model.mcresnet import MCRESNET
from model.drvsr import DRVSR
from model.frvsr import FRVSR
from model.dufvsr import DUFVSR
from model.pfnl import PFNL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
if __name__=='__main__':
    model=PFNL()
    model.train()
    #model.testvideos('/dev/f/data/video/test2/udm10')
