# BP-RBF-SVM-demo

复旦大学研究生课程神经网络及应用 INFO630016.01 的课程 PJ1。

## 作业要求

一、分别用 BP、RBF、SVM 拟合以下函数并进行分析比较：

$$
y = \frac{1}{x^5}, \quad (0 \le x \le 10)  
$$

$$
y = \frac{1+\cos x}{2}, \quad (0 \le x \le 6\pi)  
$$

$$
z = \frac{1}{\sqrt{x^2+y^2}}, \quad (-20 \le x \le 20, -20 \le y \le 20)  
$$

二、分别用 BP、RBF、SVM 对下列函数分类 $(i=1,2,...,400)$:

$$
C_a = \begin{cases}
	x_1 = \frac{1}{25} (i+8)\cos\left(\frac{2\pi}{25}(i+8)-0.25\pi\right)+\alpha \cdot \text{random} \\
	y_1 = \frac{1}{25} (i+8)\sin\left(\frac{2\pi}{25}(i+8)-0.25\pi\right)-0.25+\alpha \cdot \text{random}
\end{cases}
$$

$$
C_B = \begin{cases}
	x_2 = -\frac{1}{25} (i+8)\sin\left(\frac{2\pi}{25}(i+8)+0.25\pi\right)+\alpha \cdot \text{random} \\
	y_2 = \frac{1}{25} (i+8)\cos\left(\frac{2\pi}{25}(i+8)+0.25\pi\right)-0.25+\alpha \cdot \text{random}
\end{cases}
$$

其中， $\alpha·random$为高斯白噪声, $SNR=20dB$。


