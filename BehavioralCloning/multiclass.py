
# coding: utf-8

# In[69]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


# In[70]:


n_pts=500
#defining centers where the points will be clustered 
centers = [[-1,1],[-1,-1],[1,-1],[1,1],[0,0]]
#creating features and labels
X,Y= datasets.make_blobs(n_pts,random_state=69,centers = centers,cluster_std=0.3)


# In[71]:


plt.scatter(X[Y==0,0],X[Y==0,1]) #read as X[row of X where Y is 0,column 0],X[row of X where Y is 0,column 1]
plt.scatter(X[Y==1,0],X[Y==1,1])
plt.scatter(X[Y==2,0],X[Y==2,1])
plt.scatter(X[Y==3,0],X[Y==3,1])
plt.scatter(X[Y==4,0],X[Y==4,1])


# In[72]:


#convert labels Y to one hot encoded format
y_cat = to_categorical(Y,5)


# In[73]:


model = Sequential()
model.add(Dense(units=5,input_shape = (2,),activation='softmax'))
model.compile(Adam(lr=0.1),loss='categorical_crossentropy',metrics=['accuracy'])


# In[74]:


model.fit(x=X,y=y_cat,verbose=1,batch_size=50,epochs=100)


# In[75]:


#function that creates a grid of points,onto which we run model.predict_classes inorder to draw contour plot
def plot_multiclass_decision_boundary(X, y_cat, model):
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred_func = model.predict_classes(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)


# In[76]:


plot_multiclass_decision_boundary(X, y_cat, model)
plt.scatter(X[Y==0,0],X[Y==0,1])
plt.scatter(X[Y==1,0],X[Y==1,1])
plt.scatter(X[Y==2,0],X[Y==2,1])
plt.scatter(X[Y==3,0],X[Y==3,1])
plt.scatter(X[Y==4,0],X[Y==4,1])


# In[77]:


plot_multiclass_decision_boundary(X, y_cat, model)
plt.scatter(X[Y==0,0],X[Y==0,1])
plt.scatter(X[Y==1,0],X[Y==1,1])
plt.scatter(X[Y==2,0],X[Y==2,1])
plt.scatter(X[Y==3,0],X[Y==3,1])
plt.scatter(X[Y==4,0],X[Y==4,1])
x,y=-0.9,0.3
point = np.array([[x,y]])
pred=model.predict_classes(point)
plt.plot([x],[y],marker='X',markersize=10,color='red')
print("prediction is: ",pred)

