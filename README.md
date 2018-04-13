
# 线性回归

## 【关键词】最小二乘法，线性

## 一、普通线性回归

### 1、原理

分类的目标变量是标称型数据，而回归将会对连续型的数据做出预测。

应当怎样从一大堆数据里求出回归方程呢？

假定输人数据存放在矩阵X中，而回归系数存放在向量W中。那么对于给定的数据X1, 预测结果将会通过

Y=X*W

给出。现在的问题是，手里有一些X和对应的Y,怎样才能找到W呢？

一个常用的方法就是找出使误差最小的W。这里的误差是指预测Y值和真实Y值之间的差值，使用该误差的简单累加将使得正差值和负差值相互抵消，所以我
们采用平方误差。

最小二乘法

平方误差可以写做:


![Untitled-1-201841319242](http://p693ase25.bkt.clouddn.com/Untitled-1-201841319242.PNG)




对W求导，当导数为零时，平方误差最小，此时W等于：

![Untitled-1-201841319257](http://p693ase25.bkt.clouddn.com/Untitled-1-201841319257.PNG)



例如有下面一张图片：


![Untitled-1-20184131933](http://p693ase25.bkt.clouddn.com/Untitled-1-20184131933.PNG)



求回归曲线，得到：

![Untitled-1-20184131938](http://p693ase25.bkt.clouddn.com/Untitled-1-20184131938.PNG)



### 2、实例

导包


```python
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
%matplotlib inline

# 普通线性回归
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
```

获取糖尿病数据


```python
diabetes = load_diabetes()
```


```python
data = diabetes.data
feature_names = diabetes.feature_names
target = diabetes.target
samples = DataFrame(data=data,columns=feature_names)
samples
```


抽取训练数据和预测数据


```python
from sklearn.model_selection import train_test_split
```


```python
# 研究bmi指标对血糖的影响趋势
samples = samples['bp']
```


```python
train = samples.values.reshape(-1,1)
```

创建数学模型


```python
display(train.shape,target.shape)
```


    (442, 1)



    (442,)



```python
linear = LinearRegression()

linear.fit(train,target)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
# 获取训练数据
xmin,xmax = train.min(),train.max()
X_test = np.linspace(xmin,xmax,100).reshape(-1,1)

y_ = linear.predict(X_test)
```


```python
plt.plot(X_test,y_,color='green')
plt.scatter(train,target,color='orange')
plt.title('bp')
```




    Text(0.5,1,'bp')




![Untitled-1-2018413184851](http://p693ase25.bkt.clouddn.com/Untitled-1-2018413184851.png)



第一步：训练

第二步：预测

第三步：绘制图形


```python
Series(y_).plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22e7fdf81d0>




![Untitled-1-2018413184859](http://p693ase25.bkt.clouddn.com/Untitled-1-2018413184859.png)




```python
# 获取所有的系数
linear.coef_
```




    array([714.7416437])




```python
# boston数据线性关系
```


```python
from sklearn.metrics import r2_score
```


```python
import sklearn.datasets as datasets

boston = datasets.load_boston()
data = boston.data
target = boston.target
feature_names = boston.feature_names
samples = DataFrame(data=data,columns=feature_names)
```


```python
X_train,X_test,y_train,y_test = train_test_split(samples,target,test_size=0.2,random_state=1)

linear = LinearRegression()
linear.fit(X_train,y_train)
y_ = linear.predict(X_test)

r2_score(y_test,y_)
```




    0.7634809220792437




```python
plt.plot(y_,label='Predict')
plt.plot(y_test,label='True')
plt.legend()
```




    <matplotlib.legend.Legend at 0x22e7fa4b2e8>



![Untitled-1-201841318495](http://p693ase25.bkt.clouddn.com/Untitled-1-201841318495.png)



## 二、岭回归


```python
f(x) = w1*x1 + w2*x2 + w3*x3

1   2  4  7
2   5  3  2
3   6  1  9


# 有解方程
1 = a*2 + b*4 + c*7
2 = a*5 + b*3 + c*2
3 = a*6 + b*1 + c*9

# 无解方程
1 = a*2 + b*4 + c*7
2 = a*5 + b*3 + c*2


# 不满秩矩阵 不能求逆
# 数据样本的个数 < 数据特征的个数 需要使用岭回归
# 多重共线性
2  4  7        1 0 0      3  4  7
5  3  2   + λ*0 1 0  =   5  4  2  -> 可以求逆 -> 线性回归模型就可用了
               0 0 1      0  0  1
    
```

### 1、原理

缩减系数来“理解”数据

如果数据的特征比样本点还多应该怎么办？是否还可以使用线性回归和之前的方法来做预测？

答案是否定的，即不能再使用前面介绍的方法。这是因为输入数据的矩阵X不是满秩矩阵。非满秩矩阵在求逆时会出现问题。

为了解决这个问题，统计学家引入了岭回归（ridge regression)的概念
![Untitled-1-201841319313](http://p693ase25.bkt.clouddn.com/Untitled-1-201841319313.PNG)




缩减方法可以去掉不重要的参数，因此能更好地理解数据。此外，与简单的线性回归相比，缩减法能取得更好的预测效果。

岭回归是加了二阶正则项的最小二乘，主要适用于过拟合严重或各变量之间存在多重共线性的时候，岭回归是有bias的，这里的bias是为了让variance更小。

#### 归纳总结

1.岭回归可以解决特征数量比样本量多的问题

2.岭回归作为一种缩减算法可以判断哪些特征重要或者不重要，有点类似于降维的效果

3.缩减算法可以看作是对一个模型增加偏差的同时减少方差

岭回归用于处理下面两类问题：

1.数据点少于变量个数

2.变量间存在共线性（最小二乘回归得到的系数不稳定，方差很大）

### 2、实例

岭回归一般用在样本值不够的时候


```python
from sklearn.linear_model import Ridge
```


```python
x = [[2,1,1],[1,2,3]]
y = [3,1]
```

使用普通线性回归


```python
linear = LinearRegression()
linear.fit(x,y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



使用岭回归


```python
# alpha就是λ
ridge = Ridge(alpha=0.01)
ridge.fit(x,y)
```




    Ridge(alpha=0.01, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001)




```python
linear.coef_
```




    array([ 0.33333333, -0.33333333, -0.66666667])




```python
# 岭回归可以缩减系数
ridge.coef_
```




    array([ 0.33222591, -0.33222591, -0.66445183])



深入研究岭回归

- 理解岭回归缩减系数）

【备注】coef_函数可以获取机器学习模型中各个特征值的系数

拓展：岭回归

创建一个假象数据样本集


```python
X = 1. / (np.arange(1, 11) + np.arange(0, 10).reshape(-1,1))

y = np.array([1,2,3,4,5,6,7,8,9,0])
```


```python
# 研究λ对岭回归系数的影响
# 
alphas = np.logspace(-10, -2, 200)
alphas
```




    array([1.00000000e-10, 1.09698580e-10, 1.20337784e-10, 1.32008840e-10,
           1.44811823e-10, 1.58856513e-10, 1.74263339e-10, 1.91164408e-10,
           2.09704640e-10, 2.30043012e-10, 2.52353917e-10, 2.76828663e-10,
           3.03677112e-10, 3.33129479e-10, 3.65438307e-10, 4.00880633e-10,
           4.39760361e-10, 4.82410870e-10, 5.29197874e-10, 5.80522552e-10,
           6.36824994e-10, 6.98587975e-10, 7.66341087e-10, 8.40665289e-10,
           9.22197882e-10, 1.01163798e-09, 1.10975250e-09, 1.21738273e-09,
           1.33545156e-09, 1.46497140e-09, 1.60705282e-09, 1.76291412e-09,
           1.93389175e-09, 2.12145178e-09, 2.32720248e-09, 2.55290807e-09,
           2.80050389e-09, 3.07211300e-09, 3.37006433e-09, 3.69691271e-09,
           4.05546074e-09, 4.44878283e-09, 4.88025158e-09, 5.35356668e-09,
           5.87278661e-09, 6.44236351e-09, 7.06718127e-09, 7.75259749e-09,
           8.50448934e-09, 9.32930403e-09, 1.02341140e-08, 1.12266777e-08,
           1.23155060e-08, 1.35099352e-08, 1.48202071e-08, 1.62575567e-08,
           1.78343088e-08, 1.95639834e-08, 2.14614120e-08, 2.35428641e-08,
           2.58261876e-08, 2.83309610e-08, 3.10786619e-08, 3.40928507e-08,
           3.73993730e-08, 4.10265811e-08, 4.50055768e-08, 4.93704785e-08,
           5.41587138e-08, 5.94113398e-08, 6.51733960e-08, 7.14942899e-08,
           7.84282206e-08, 8.60346442e-08, 9.43787828e-08, 1.03532184e-07,
           1.13573336e-07, 1.24588336e-07, 1.36671636e-07, 1.49926843e-07,
           1.64467618e-07, 1.80418641e-07, 1.97916687e-07, 2.17111795e-07,
           2.38168555e-07, 2.61267523e-07, 2.86606762e-07, 3.14403547e-07,
           3.44896226e-07, 3.78346262e-07, 4.15040476e-07, 4.55293507e-07,
           4.99450512e-07, 5.47890118e-07, 6.01027678e-07, 6.59318827e-07,
           7.23263390e-07, 7.93409667e-07, 8.70359136e-07, 9.54771611e-07,
           1.04737090e-06, 1.14895100e-06, 1.26038293e-06, 1.38262217e-06,
           1.51671689e-06, 1.66381689e-06, 1.82518349e-06, 2.00220037e-06,
           2.19638537e-06, 2.40940356e-06, 2.64308149e-06, 2.89942285e-06,
           3.18062569e-06, 3.48910121e-06, 3.82749448e-06, 4.19870708e-06,
           4.60592204e-06, 5.05263107e-06, 5.54266452e-06, 6.08022426e-06,
           6.66991966e-06, 7.31680714e-06, 8.02643352e-06, 8.80488358e-06,
           9.65883224e-06, 1.05956018e-05, 1.16232247e-05, 1.27505124e-05,
           1.39871310e-05, 1.53436841e-05, 1.68318035e-05, 1.84642494e-05,
           2.02550194e-05, 2.22194686e-05, 2.43744415e-05, 2.67384162e-05,
           2.93316628e-05, 3.21764175e-05, 3.52970730e-05, 3.87203878e-05,
           4.24757155e-05, 4.65952567e-05, 5.11143348e-05, 5.60716994e-05,
           6.15098579e-05, 6.74754405e-05, 7.40196000e-05, 8.11984499e-05,
           8.90735464e-05, 9.77124154e-05, 1.07189132e-04, 1.17584955e-04,
           1.28989026e-04, 1.41499130e-04, 1.55222536e-04, 1.70276917e-04,
           1.86791360e-04, 2.04907469e-04, 2.24780583e-04, 2.46581108e-04,
           2.70495973e-04, 2.96730241e-04, 3.25508860e-04, 3.57078596e-04,
           3.91710149e-04, 4.29700470e-04, 4.71375313e-04, 5.17092024e-04,
           5.67242607e-04, 6.22257084e-04, 6.82607183e-04, 7.48810386e-04,
           8.21434358e-04, 9.01101825e-04, 9.88495905e-04, 1.08436597e-03,
           1.18953407e-03, 1.30490198e-03, 1.43145894e-03, 1.57029012e-03,
           1.72258597e-03, 1.88965234e-03, 2.07292178e-03, 2.27396575e-03,
           2.49450814e-03, 2.73644000e-03, 3.00183581e-03, 3.29297126e-03,
           3.61234270e-03, 3.96268864e-03, 4.34701316e-03, 4.76861170e-03,
           5.23109931e-03, 5.73844165e-03, 6.29498899e-03, 6.90551352e-03,
           7.57525026e-03, 8.30994195e-03, 9.11588830e-03, 1.00000000e-02])




```python
ridge = Ridge()

coefs = []
for alpha in alphas:
    ridge.set_params(alpha=alpha)
    # 使用不同的λ系数的岭回归模型，训练相同的一组数据集
    ridge.fit(X,y)
    # 每训练一次，都会得到一组系数
    coefs.append(ridge.coef_)
```


```python
# 绘图展示λ和coef之间的关系
plt.figure(figsize=(10,6))
data = plt.plot(alphas,coefs)
plt.xscale('log')
```


![Untitled-1-2018413184911](http://p693ase25.bkt.clouddn.com/Untitled-1-2018413184911.png)

创建一个alpha集合，用以验证种不同alpha值对预测系数的结果的影响

创建岭回归机器学习算法对象

使用不同的alpha进行数据训练，保存所有训练结果的coef_

绘图查看alpha参数和coefs的关系

## 三、lasso回归

### 1、原理

【拉格朗日乘数法】

对于参数w增加一个限定条件，能到达和岭回归一样的效果：


![Untitled-1-201841319324](http://p693ase25.bkt.clouddn.com/Untitled-1-201841319324.PNG)

在lambda足够小的时候，一些系数会因此被迫缩减到0

### 2、实例


```python
# boston房价
boston = datasets.load_boston()
data = boston.data
target = boston.target
feature_names = boston.feature_names
samples = DataFrame(data=data,columns=feature_names)

X_train,X_test,y_train,y_test = train_test_split(samples,target,test_size=0.2,random_state=1)
```


```python
display(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
```


    (404, 13)



    (404,)



    (102, 13)



    (102,)



```python
# a = X_train.values.reshape(-1,1)
# display(a.shape,y_train.shape)
```


```python
# plt.scatter(a,y_train)
```


```python
from sklearn.linear_model import Lasso
ridge = Ridge(alpha=0.8)
lasso = Lasso(alpha=0.0006)

ridge.fit(X_train,y_train)
lasso.fit(X_train,y_train)

y1_ = ridge.predict(X_test)
y2_ = lasso.predict(X_test)

print('ridge r2_score is %f'% r2_score(y_test,y1_))
print('lasso r2_score is %f'% r2_score(y_test,y2_))
```

    ridge r2_score is 0.765714
    lasso r2_score is 0.763532



```python
ridge.coef_
```




    array([-1.06778121e-01,  5.85331114e-02, -1.53308847e-02,  1.96862960e+00,
           -1.19074999e+01,  3.15119992e+00, -1.78841078e-03, -1.38612510e+00,
            2.87468050e-01, -1.18525460e-02, -9.03202511e-01,  7.82130600e-03,
           -5.55011590e-01])




```python
lasso.coef_
```




    array([-1.11200981e-01,  5.80377927e-02,  1.78595557e-02,  2.12213214e+00,
           -1.94134564e+01,  3.08852219e+00,  4.40377433e-03, -1.49711536e+00,
            3.04419685e-01, -1.11358504e-02, -9.87853509e-01,  7.43569257e-03,
           -5.45952385e-01])




```python
linear = LinearRegression()
linear.fit(X_train,y_train)
y_ = linear.predict(X_test)
r2_score(y_test,y_)
```




    0.7634809220792437




```python


```

## 四、普通线性回归、岭回归与lasso回归比较

导包，导入sklearn.metrics.r2_score用于给模型打分

使用numpy创建数据X，创建系数，对系数进行处理，对部分系数进行归零化操作，然后根据系数进行矩阵操作求得目标值  
增加噪声

训练数据和测试数据

分别使用线性回归，岭回归，Lasso回归进行数据预测

数据视图，此处获取各个算法的训练数据的coef_:系数
