# plot_decision_boundary
def plot decision boundary

import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y):
  #define the axis boundary of the plot and reat a meshgrid
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() - 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() - 0.1

  xx, yy = np.mishgrid(np.linspace(x_min, x_max, 100),
                       np.linespace(y_min, y_max, 100))
  
  # creat X values where we going to make prediction on these
  x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D array together 

  #make predictions 
  y_pred = model.predict(x_in)

  #check for multi class
  if len(y_pred[0]) > 1 :
    print("doing multiclass classification")
    #we have to reshape our predections to get them ready for plotting
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classification")  
    y_pred = np.round(y_pred).reshape(xx.shape)

  #the plot decision boundary
  plt.contour(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha = 0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40,  cmap=plt.cm.RdYlBu)
  plt.xlim(x.min(), x.max())
  plt.ylim(y.min(), y.max())