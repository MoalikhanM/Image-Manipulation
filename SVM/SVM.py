# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 23:09:45 2022

@author: aoral
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn import svm
from sklearn.gaussian_process.kernels import RBF


#---------------------------------------------------------------------------------------
# Part-1.Linear SVM

# Construct 100 samples of data
x,y = make_blobs(n_samples = 100, centers = 3, cluster_std = 1.2, random_state = 20)
plt.scatter(x[:,0], x[:,1], c = y, s = 30)
# Save the result as png file
plt.title('random_state=20')
plt.savefig('data/random_state_20.png')
plt.clf()

# Construct 100 samples of data
x,y = make_blobs(n_samples = 100, centers = 3, cluster_std = 1.2, random_state = 30)
plt.scatter(x[:,0], x[:,1], c = y, s = 30)
# Save the result as png file
plt.title('random_state=30')
plt.savefig('data/random_state_30.png')
plt.clf()

# Construct 100 samples of data
x,y = make_blobs(n_samples = 100, centers = 3, cluster_std = 1.2, random_state = 40)
plt.scatter(x[:,0], x[:,1], c = y, s = 30)
# Save the result as png file
plt.title('random_state=40')
plt.savefig('data/random_state_40.png')
plt.clf()


# Part-1-2.Train SVM

# Construct 100 samples of data
x,y = make_blobs(n_samples = 100, centers = 2, cluster_std = 1.2, random_state = 50)

# fit the model
#clf = svm.SVC(kernel='linear', C = 0.1)
#clf = svm.SVC(kernel='linear', C = 1.0)
clf = svm.SVC(kernel='linear', C = 10)
clf.fit(x,y)

plt.scatter(x[:,0], x[:,1], c = y, s = 30, cmap = plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])

# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
"""
plt.title('C = 0.1')
plt.savefig('data/cost_0.1.png')
plt.title('C = 1.0')
plt.savefig('data/cost_1.0.png')
"""
plt.title('C = 10')
plt.savefig('data/cost_10.png')
plt.clf()

#---------------------------------------------------------------------------------------
# Part-2.Nonliear SVM

# factor = R2/R1, noise=std
x,y = make_circles(factor = 0.1, noise = 0.1) 
colors = np.array(["red","yellow"])
plt.scatter(x[:,0], x[:,1], c = y, s = 30, cmap = plt.cm.Paired)
plt.title('Factor = 0.1')
plt.savefig('data/non_linear_svm.png')
plt.clf()


# Part-2-1.Kernel function

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

# factor = R2/R1, noise=std
x,y=make_circles(factor = 0.2, noise = 0.1) 

z = RBF(1.0).__call__(x)[0]

# Plot
ax.scatter(x[:, 0], x[:, 1], z, c = y, s = 30, cmap = plt.cm.Paired)


plt.title('RBF Kernel')
plt.savefig('data/RBF_Kernel.png')
plt.clf()

#---------------------------------------------------------------------------------------
# 3. Train SVM

def svc_decision_function(model):
    ax = None
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y,X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X, Y, P,colors="r",levels=[-1,0,1],alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# factor = R2/R1, noise=std
x,y=make_circles(factor = 0.2, noise = 0.1) 
clf = SVC(kernel="rbf").fit(x,y)
plt.scatter(x[:,0], x[:,1], c = y, s = 30, cmap = plt.cm.Paired)
svc_decision_function(clf)
plt.savefig('data/Train_SVM.png')
                       
