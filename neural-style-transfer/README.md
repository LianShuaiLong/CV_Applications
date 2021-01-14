## 用途
给定一张content image和一个style image快速实现content image向style image的风格转换;可以在pytorch_demo.py/nst_demo.py里面修改输入图片的路径
## 原理
利用vgg提取style image的feature进一步将feature的GRAM矩阵作为style image的style feature

利用LBFGS算法可以在有限迭代次数的前提下高质量优化content image

## 优点
理论上可以实现任意风格的转换，不用每组风格训练一个模型

pytorch_demo.py是利用pytorch的原生api做的风格转换
nst_demo.py是利用pystiche框架做的风格转换,pystiche是基于pytorch的风格转换框架

## demo 图
content image:

![image](https://github.com/LianShuaiLong/CV_Applications/blob/master/neural-style-transfer/demo_data/content.png)

style image:

![image](https://github.com/LianShuaiLong/CV_Applications/blob/master/neural-style-transfer/demo_data/style.png) 

result image:

![image](https://github.com/LianShuaiLong/CV_Applications/blob/master/neural-style-transfer/demo_data/pytorch_res.png)

## Reference
1.[pystiche](https://github.com/pmeier/pystiche)

2.[neural_style_tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

