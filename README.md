# BP-RBF-SVM-demo

复旦大学研究生课程经网络及应用2024202501INFO630016.01的课程PJ。

## 作业要求

一、分别用BP、RBF、SVM拟合以下函数并进行分析比较：
$$
\begin{aligned}
y=&\frac{1}{x^5},(0 \le x \le 10)\\

y=&\frac{1+cosx}{2},(0 \le x \le 6\pi)\\

z=&\frac{1}{\sqrt{x^2+y^2}},(-20 \le x \le 20,-20 \le y \le 20)
\end{aligned}
$$
二、分别用BP、RBF、SVM对下列函数分类$(i=1,2,...,400)$
$$
C_a=\left\{
	\begin{aligned}
		x_1 = & \frac{1}{25} (i+8)cos(\frac{2\pi}{25}(i+8)-0.25\pi)+\alpha ·random\\
		y_1 = & \frac{1}{25} (i+8)sin(\frac{2\pi}{25}(i+8)-0.25\pi)-0.25+\alpha ·random
	\end{aligned}
	\right.
$$
$$
C_B=\left\{
	\begin{aligned}
		x_2 = & -\frac{1}{25} (i+8)sin(\frac{2\pi}{25}(i+8)+0.25\pi)+\alpha ·random\\
		y_2 = & \frac{1}{25} (i+8)cos(\frac{2\pi}{25}(i+8)+0.25\pi)-0.25+\alpha ·random
	\end{aligned}
	\right.
$$
其中，$\alpha ·random$为高斯白噪声,$SNR=20dB$。