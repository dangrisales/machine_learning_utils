
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import openpyxl
import numpy as np

from matplotlib import rc

rc('font', **{'family':'serif','serif':['Times']})
rc('text', usetex=True)

#plt.rc('font',**{'family':'serif','serif':['Times']})
#plt.rc('text', usetex=True)

yUPDRS_Spanish=np.loadtxt("updrs_Spanish.txt")
yUPDRS_Spanish = yUPDRS_Spanish/132

yUPDRS_Czech=np.loadtxt("updrs_Czech.txt")
yUPDRS_Czech = yUPDRS_Czech/132

yUPDRS_German=np.loadtxt("updrs_German.txt")
yUPDRS_German = yUPDRS_German/132

yUHDRS=np.loadtxt("uhdrs_Czech.txt")
yUHDRS=np.delete(yUHDRS, np.where(np.isnan(yUHDRS))[0]) #Eliminar 2 NaN
yUHDRS = yUHDRS/92



#%%
plt.figure(figsize=(10,6))

aa=[yUPDRS_Spanish,yUPDRS_Czech,yUPDRS_German,yUHDRS]
labels=["PD Spanish","PD Czech","PD German","HD Czech"]
colors = ['#440154', '#8DA0CB', '#E5C494', '#66C2A5']
plt.hist(aa, 9, density=True, histtype='bar', color=colors, label=labels, alpha=0.7, rwidth=0.9)
plt.grid(True, linestyle="dashed", alpha=0.8)
plt.legend(prop={'size': 20})
plt.tick_params(labelsize=20)
plt.xlabel("UHDRS/UPDRS-III Score", fontsize=24)
plt.xlim([0,0.66])
plt.yticks([])
#plt.savefig('histograma_normalizado.pdf')
plt.show()

#%%

from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
                
                
fig = figure(figsize=(10,6))
ax = axes()
#hold(True)

flierprops = dict(marker='+', markersize=10, linestyle=' ', alpha=0.7,markeredgecolor='#440154')
bp1 = boxplot(yUPDRS_Spanish,vert=0, positions = [6], widths = 1,flierprops=flierprops)
setp(bp1['boxes'][0],alpha=0.7, color='#440154',linewidth=4)
setp(bp1['caps'][0],alpha=0.7, color='#440154',linewidth=4)
setp(bp1['caps'][1],alpha=0.7, color='#440154',linewidth=4)
setp(bp1['whiskers'][0],alpha=0.7, color='#440154',linewidth=4)
setp(bp1['whiskers'][1],alpha=0.7, color='#440154',linewidth=4)
setp(bp1['medians'][0],alpha=0.7, color='#440154',linewidth=4)

flierprops = dict(marker='+', markersize=20, linestyle=' ', alpha=1,markeredgecolor='#8DA0CB')
bp2 = boxplot(yUPDRS_Czech,vert=0, positions =[2], widths = 1,flierprops=flierprops)
setp(bp2['boxes'][0], color='#8DA0CB',linewidth=4)
setp(bp2['caps'][0], color='#8DA0CB',linewidth=4)
setp(bp2['caps'][1], color='#8DA0CB',linewidth=4)
setp(bp2['whiskers'][0], color='#8DA0CB',linewidth=4)
setp(bp2['whiskers'][1], color='#8DA0CB',linewidth=4)
setp(bp2['medians'][0], color='#8DA0CB',linewidth=4)   

flierprops = dict(marker='+', markersize=10, linestyle=' ', alpha=0.7,markeredgecolor='#E5C494')
bp3 = boxplot(yUPDRS_German,vert=0, positions = [4], widths = 1,flierprops=flierprops)
setp(bp3['boxes'][0],alpha=0.7, color='#E5C494',linewidth=4)
setp(bp3['caps'][0],alpha=0.7, color='#E5C494',linewidth=4)
setp(bp3['caps'][1],alpha=0.7, color='#E5C494',linewidth=4)
setp(bp3['whiskers'][0],alpha=0.7, color='#E5C494',linewidth=4)
setp(bp3['whiskers'][1],alpha=0.7, color='#E5C494',linewidth=4)
setp(bp3['medians'][0],alpha=0.7, color='#E5C494',linewidth=4)

# second boxplot pair
flierprops = dict(marker='+', markersize=20, linestyle=' ', alpha=1,markeredgecolor='#66C2A5')
bp4 = boxplot(yUHDRS,vert=0, positions =[8], widths = 1,flierprops=flierprops)
setp(bp4['boxes'][0], color='#66C2A5',linewidth=4)
setp(bp4['caps'][0], color='#66C2A5',linewidth=4)
setp(bp4['caps'][1], color='#66C2A5',linewidth=4)
setp(bp4['whiskers'][0], color='#66C2A5',linewidth=4)
setp(bp4['whiskers'][1], color='#66C2A5',linewidth=4)
setp(bp4['medians'][0], color='#66C2A5',linewidth=4)                


legend([bp1["boxes"][0],bp2["boxes"][0],bp3["boxes"][0],bp4["boxes"][0]], ['PD Spanish', 'PD Czech', 'PD German', 'HD Czech'],fontsize=20)

ylim(1,9)
ax.set_yticks([])
ax.set_xticks([])

xlim(0,0.66)
#xlim((18, 87))   # set the xlim to xmin, xmax

#plt.savefig('./imagenes/boxplot_normalizado.pdf')
plt.show()


#%%

fig, axs = plt.subplots(2, 1, figsize=(10,10))
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

#hold(True)
flierprops = dict(marker='+', markersize=10, linestyle=' ',markeredgecolor='grey')
bp1 = axs[0].boxplot(yUPDRS_Spanish,vert=0, positions = [8], widths = 1,flierprops=flierprops)
setp(bp1['boxes'][0], color='grey',linewidth=4)
setp(bp1['caps'][0], color='grey',linewidth=4)
setp(bp1['caps'][1], color='grey',linewidth=4)
setp(bp1['whiskers'][0], color='grey',linewidth=4)
setp(bp1['whiskers'][1], color='grey',linewidth=4)
setp(bp1['medians'][0], color='grey',linewidth=4)

flierprops = dict(marker='+', markersize=20, linestyle=' ', alpha=1,markeredgecolor='#8DA0CB')
bp2 = axs[0].boxplot(yUPDRS_Czech,vert=0, positions =[6], widths = 1,flierprops=flierprops)
setp(bp2['boxes'][0], color='#8DA0CB',linewidth=4)
setp(bp2['caps'][0], color='#8DA0CB',linewidth=4)
setp(bp2['caps'][1], color='#8DA0CB',linewidth=4)
setp(bp2['whiskers'][0], color='#8DA0CB',linewidth=4)
setp(bp2['whiskers'][1], color='#8DA0CB',linewidth=4)
setp(bp2['medians'][0], color='#8DA0CB',linewidth=4)   

flierprops = dict(marker='+', markersize=10, linestyle=' ',markeredgecolor='darkgreen')
bp3 = axs[0].boxplot(yUPDRS_German,vert=0, positions = [4], widths = 1,flierprops=flierprops)
setp(bp3['boxes'][0], color='darkgreen',linewidth=4)
setp(bp3['caps'][0], color='darkgreen',linewidth=4)
setp(bp3['caps'][1], color='darkgreen',linewidth=4)
setp(bp3['whiskers'][0], color='darkgreen',linewidth=4)
setp(bp3['whiskers'][1], color='darkgreen',linewidth=4)
setp(bp3['medians'][0], color='darkgreen',linewidth=4)

# second boxplot pair
flierprops = dict(marker='+', markersize=20, linestyle=' ', alpha=1,markeredgecolor='#440154')
bp4 = axs[0].boxplot(yUHDRS,vert=0, positions =[2], widths = 1,flierprops=flierprops)
setp(bp4['boxes'][0], color='#440154',linewidth=4)
setp(bp4['caps'][0], color='#440154',linewidth=4)
setp(bp4['caps'][1], color='#440154',linewidth=4)
setp(bp4['whiskers'][0], color='#440154',linewidth=4)
setp(bp4['whiskers'][1], color='#440154',linewidth=4)
setp(bp4['medians'][0], color='#440154',linewidth=4) 

axs[0].set_ylim(1, 9)
axs[0].set_xlim([0,0.72])
#axs[0].legend([bp1["boxes"][0],bp2["boxes"][0],bp3["boxes"][0],bp4["boxes"][0]], ['PD Spanish', 'PD Czech', 'PD German', 'HD Czech'],fontsize=20)
axs[0].set_yticks([])
axs[0].set_xticks([])


aa=[yUPDRS_Spanish,yUPDRS_Czech,yUPDRS_German,yUHDRS]
labels=["PD-Spanish","PD-Czech","PD-German","HD-Czech"]
colors = ['grey', '#8DA0CB', 'darkgreen', '#440154']
axs[1].hist(aa, 9, density=True, histtype='bar', color=colors, label=labels, rwidth=0.9)
axs[1].grid(True, linestyle="dashed", alpha=0.8)

axs[1].set_yticks([])
axs[1].legend(prop={'size': 20})
axs[1].tick_params(labelsize=20)
axs[1].set_xlim([0,0.72])

plt.xlabel("MDS-(UPDRS-III/UHDRS)", fontsize=24)
#plt.savefig('histograma_normalizado3.pdf')

plt.show()





