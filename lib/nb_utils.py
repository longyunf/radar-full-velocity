import numpy as np
import matplotlib.pyplot as plt

class sparse_neighbor_connection:   
    def __init__(self, left, right, top, bottom, skip=1):
        self.xy, self.hn = self.getXYoffset(left, right, top, bottom, skip)
        self.num_nbs = len(self.xy)
                       
    def getXYoffset(self, left, right, top, bottom, skip):
        xy = []
        step = skip + 1
        x_pos = np.concatenate( [ np.arange(0, -left-1, -step)[::-1], np.arange(step, right+1, step) ] )
        y_pos = np.concatenate( [ np.arange(0, -top-1, -step)[::-1], np.arange(step, bottom+1, step) ] )
        
        for x in x_pos:
            for y in y_pos:
                xy.append([x,y])
                
        hn = max([left, right, top, bottom])
        
        return xy, hn
            
    def plot_neighbor(self):
        xy = self.xy
        hn = self.hn
       
        M = np.zeros((2*hn + 1, 2*hn + 1), dtype=np.uint8)   
        for x, y in xy:
            x += hn
            y += hn
            M[y,x]=255
        M[hn, hn] = 128
        
        plt.imshow(M, cmap='gray')
        plt.title('%d neighbors' % len(xy))
        plt.show()
