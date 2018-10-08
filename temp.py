from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# wash data
with open('8.iris.data','r') as f:
    content = f.readlines()
    content_new =[]

    for con in content:
        tem = con.strip()
        tem = tem.split(',')
        content_new.append(tem)

iris_data = [[0 for i in range(4)] for i in range(150)]
iris_target = [[0 for i in range(1)] for i in range(150)]

for i in range(150):
    iris_data[i][0] = content_new[i][0]
    iris_data[i][1] = content_new[i][1]
    iris_data[i][2] = content_new[i][2]
    iris_data[i][3] = content_new[i][3]
    if content_new[i][4] == 'Iris-setosa':
    	iris_target[i][0] = 0
    if content_new[i][4] == 'Iris-versicolor':
    	iris_target[i][0] = 1
    if content_new[i][4] == 'Iris-virginica':
    	iris_target[i][0] = 2
        
        
iris_data = np.array(iris_data)
# train：test=7:3
#x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.4,random_state=0)
x_train = iris_data[30:120]
x_test = iris_data
y_train = iris_target[30:120]
y_test = iris_target


score = [0 for i in range(15)]
# Draw the depth pic
for i in range(2,12,1):
    clf_ = tree.DecisionTreeClassifier(max_depth=i).fit(x_train, y_train)
    score[i] = clf_.score(x_test, y_test)

plt.plot(score[2:12])
plt.xlabel("Max_depth")
plt.ylabel("Accuracy")
plt.show()

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris_data, iris_target)

# draw the pic feature 1 feature 2

n_classes = 3
plot_colors = "ryb"
plot_step = 0.02



iris_data1 = np.array(iris_data)
X = iris_data1[:, 0:2]
X=X.astype(float)


y = iris_target
y = np.array(y)
y =y.astype(float)
# train
clf = tree.DecisionTreeClassifier().fit(X, y)

# Plot the boundary

x_min, x_max = X[:, 0].min() -1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() -1, X[:, 1].max() + 1

x_min =x_min.astype(float)
x_max =x_max.astype(float)
y_min =y_min.astype(float)
y_max =y_max.astype(float)

xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
plt.xlabel("length")
plt.ylabel("width")

for i, color in zip(range(n_classes), plot_colors):
	idx = np.where(y == i)
	plt.scatter(X[idx, 0],X[idx, 1], c=color, 
				cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()

