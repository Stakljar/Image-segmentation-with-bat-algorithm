# Image-segmentation-with-bat-algorithm

This project showcases usage of bat algorithm and its variants with different types and methods of image segmentation.
It is used for grayscale, colored and semantic segmentation.
Otsu's method's within-class variance is used as objective function for grayscale segmentation.
For colored segmentation K-means inertia is used as objective function.
For semantic segmentation bat algorithm is fused with CNN model where bat algorithm is used for optimization of learning rate and batch size.
Objective function for that is cross-entropy loss on validation set.

Bat algorithm was developed by X. S. Yang and it is used here.<br>
Reference:<br>
X. S. Yang, Nature-Inspired Optimization Algorithms: Second Edition, Elsevier, London, 2020.

One of the variants of chaotic bat algorithm that was developed by X.S. Yang and A.H. Gandomi is used.<br>
Reference:<br>
A. H. Gandomi and X. S. Yang, »Chaotic bat algorithm«, _Journal of Computational Science_, No. 2, Vol. 5, pp. 224-232, March 2014. 

Bat algorithm with inertia weight that was developed by Z. Cui _et al._ is used.<br>
Reference:<br>
Z. Cui, F. Li and Q. Kang, »Bat algorithm with inertia weight«, _Chinese Automation Congress (CAC)_, pp. 792-796, Wuhan, 2015.

Bat algoritm with K-means that uses K-Means initialization of bat locations based on idea of Improved bat algorithm which uses K-means developed by M. Sujarita _et al._ is used.<br>
Reference:<br>
M. Sujaritha, M. Kavitha, S. Shunmugapriya, R. S. Vikram, C. Somasundaram and R. Yogeshwaran, »Multispectral Satellite Image Segmentation Using Improved Bat Algorithm«,
_International Conference on Advanced Computing Technologies and Applications (ICACTA)_, pp. 1-6, Coimbatore, 2022.

I. Strumberger _et al._ developed architecture of Firefly algorithm with CNN where Firefly algorithm is used for optimization of hyperparameters of CNN, based on that idea bat algorithm with CNN is made here.<br>
Reference:<br>
I. Strumberger, E. Tuba, N. Bacanin, M. Zivkovic, M. Beko and M. Tuba, »Designing Convolutional Neural Network Architecture by the Firefly Algorithm«,
_International Young Engineers Forum (YEF-ECE)_, pp. 59-65, Costa da Caparica, 2019. 

Reference on used inertia weight functions:<br>
J. C. Bansal, P. K. Singh, M. Saraswat, A. Verma, S. S. Jadon and A. Abraham, »Inertia Weight strategies in Particle Swarm Optimization«,
_Third World Congress on Nature and Biologically Inspired Computing_, pp. 633-640, Salamanca, 2011.

References on used chaotic maps:<br>
Logistic:<br>
»Logistic Equation - Chaos & Fractals«<br>
Available on: [https://www.stsci.edu/~lbradley/seminar/logdiffeqn.html](https://www.stsci.edu/~lbradley/seminar/logdiffeqn.html)<br>
[Access date: 17th of June 2024]

Sine:<br>
C. Li, K. Qian, S. He, H. Li and W. Feng, »Dynamics and Optimization Control of a Robust Chaotic Map«, IEEE Access, Vol. 7, pp. 160072-160081, listopad 2019. 

Tent:<br>
M. A. Khan and V. Jeoti, »Modified chaotic tent map with improved robust region«, IEEE 11th Malaysia International Conference on Communications (MICC), pp. 496-499, Kuala Lumpur, 2013. 


