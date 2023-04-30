Download Link: https://assignmentchef.com/product/solved-fys-stk3155-project-1-regression-analysis-and-resampling-methods
<br>
The main aim of this project is to study in more detail various regression methods, including the Ordinary Least Squares (OLS) method, Ridge regression and finally Lasso regression. The methods are in turn combined with resampling techniques.

We will first study how to fit polynomials to a specific two-dimensional function called <a href="http://www.dtic.mil/dtic/tr/fulltext/u2/a081688.pdf">Franke’s function</a><a href="http://www.dtic.mil/dtic/tr/fulltext/u2/a081688.pdf">.</a> This is a function which has been widely used when testing various interpolation and fitting algorithms. Furthermore, after having established the model and the method, we will employ resamling techniques such as cross-validation in order to perform a proper assessment of our models. We will also study in detail the so-called Bias-Variance trade off.

The Franke function, which is a weighted sum of four exponentials reads as follows

<sup> </sup>(9<em>x − </em>2)<sup>2                                         </sup>(9<em>y − </em>2)<sup>2</sup> (9<em>x </em>+ 1)

<em>f</em>(<em>x,y</em>) =exp                 <em>−                     −                            </em>+exp        <em>−</em>




+exp                                         <em>−−</em>exp <em>−</em>(9<em>x − </em>4)<sup>2 </sup><em>− </em>(9<em>y − </em>7)<sup>2</sup><em>.</em>

4                      4

The function will be defined for <em>x,y ∈ </em>[0<em>,</em>1]. Our first step will be to perform an OLS regression analysis of this function, trying out a polynomial fit with an <em>x </em>and <em>y </em>dependence of the form [<em>x,y,x</em><sup>2</sup><em>,y</em><sup>2</sup><em>,xy,…</em>]. We will also include cross-validation as resampling technique. As in homeworks 1 and 2, we can use a uniform distribution to set up the arrays of values for <em>x </em>and <em>y</em>, or as in the example below just a set of fixed values for <em>x </em>and <em>y </em>with a given step size. We will fit a function (for example a polynomial) of <em>x </em>and <em>y</em>. Thereafter we will repeat much of the same procedure using the Ridge and Lasso regression methods, introducing thus a dependence on the bias (penalty) <em>λ</em>.

<em> </em>c 1999-2019, “Data Analysis and Machine Learning

FYS-STK3155/FYS4155″:”http://www.uio.no/studier/emner/matnat/fys/FYS3155/index-

eng.html”. Released under CC Attribution-NonCommercial 4.0

license

Finally we are going to use (real) digital terrain data and try to reproduce these data using the same methods. We will also try to go beyond the secondorder polynomials metioned above and explore which polynomial fits the data best.

The Python fucntion for the Franke function is included here (it performs also a three-dimensional plot of it)

from mpl_toolkits.mplot3d import Axes3D import matplotlib.pyplot as plt from matplotlib import cm from matplotlib.ticker import LinearLocator, FormatStrFormatter import numpy as np from random import random, seed

fig = plt.figure() ax = fig.gca(projection=’3d’)

# Make data.

x = np.arange(0, 1, 0.05) y = np.arange(0, 1, 0.05) x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):

term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) – 0.25*((9*y-2)**2)) term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 – 0.1*(9*y+1)) term3 = 0.5*np.exp(-(9*x-7)**2/4.0 – 0.25*((9*y-3)**2)) term4 = -0.2*np.exp(-(9*x-4)**2 – (9*y-7)**2) return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)

# Plot the surface.

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,

linewidth=0, antialiased=False)

# Customize the z axis. ax.set_zlim(-0.10, 1.40)

ax.zaxis.set_major_locator(LinearLocator(10)) ax.zaxis.set_major_formatter(FormatStrFormatter(’%.02f’))

# Add a color bar which maps values to colors. fig.colorbar(surf, shrink=0.5, aspect=5) plt.show()

<strong>Part a): Ordinary Least Square on the Franke function with resampling. </strong>We will generate our own dataset for a function FrankeFunction(<em>x,y</em>) with <em>x,y ∈ </em>[0<em>,</em>1]. The function <em>f</em>(<em>x,y</em>) is the Franke function. You should explore also the addition an added stochastic noise to this function using the normal distribution <em>N</em>(<em>0,∞</em>).

Write your own code (using either a matrix inversion or a singular value decomposition from e.g., <strong>numpy </strong>) or use your code from homeworks 1 and 2 and perform a standard least square regression analysis using polynomials in <em>x </em>and <em>y </em>up to fifth order. Find the confidence intervals of the parameters <em>β </em>by computing their variances, evaluate the Mean Squared error (MSE)

<em>n−</em>1

1 X <em>− y</em>˜<em><sub>i</sub></em>)<sub>2</sub><em>, MSE</em>(<em>y,</em>ˆ <em>y</em>ˆ˜) = (<em>y<sub>i </sub>n</em>

<em>i</em>=0

and the <em>R</em><sup>2 </sup>score function. If <em>y</em><sup>˜</sup>ˆ<em><sub>i </sub></em>is the predicted value of the <em>i − th </em>sample and <em>y<sub>i </sub></em>is the corresponding true value, then the score <em>R</em><sup>2 </sup>is defined as

<em>R </em>(<em>y,</em>ˆ <em>y</em>

2 ˜ˆ) = 1 <em>− </em>P<sup>P</sup><em>niin</em>=0=0<em>−−</em>11((<em>yy</em><em>i<sub>i</sub>−−y</em>˜<em>y</em>¯<em>i</em>))22<em>,</em>

where we have defined the mean value of <em>y</em>ˆ as

<em>n−</em>1 1

<em>y</em>¯ =   X <em>y<sub>i</sub>. n</em>

<em>i</em>=0

<strong>Part b) Resampling techniques, adding more complexity. </strong>Perform a resampling of the data where you split the data in training data and test data. Here you can write your own function or use the function for splitting training data provided by <strong>Scikit-Learn</strong>. This function is called <em>train</em>_<em>test</em>_<em>split</em>.

It is normal in essentially all Machine Learning studies to split the data in a training set and a test set (sometimes also an additional validation set). There is no explicit recipe for how much data should be included as training data and say test data. An accepted rule of thumb is to use approximately 2<em>/</em>3 to 4<em>/</em>5 of the data as training data.

Implement the <em>k</em>-fold cross-validation algorithm (write your own code) and evaluate again the MSE and the <em>R</em><sup>2 </sup>functions resulting from the test data. You can compare your own code with that from <strong>Scikit-Learn </strong>if needed.

<strong>Part c): Bias-variance tradeoff. </strong>With a code which does OLS and includes resampling techniques, we will now discuss the bias-variance tradeoff in the context of continuous predictions such as regression. However, many of the intuitions and ideas discussed here also carry over to classification tasks and basically all Machine Learning algorithms.

Consider a dataset <em>L </em>consisting of the data <strong>X</strong><em><sub>L </sub></em>= <em>{</em>(<em>y<sub>j</sub>,x<sub>j</sub></em>)<em>,j </em>= 0<em>…n − </em>1<em>}</em>. Let us assume that the true data is generated from a noisy model

<em>y </em>= <em>f</em>(<em>x</em>) + <em>.</em>

Here <em> </em>is normally distributed with mean zero and standard deviation <em>σ</em><sup>2</sup>.

In our derivation of the ordinary least squares method we defined then an approximation to the function <em>f </em>in terms of the parameters <em>β </em>and the design matrix <em>X </em>which embody our model, that is <em>y</em><strong>˜ </strong>= <em>Xβ</em>.

The parameters <em>β </em>are in turn found by optimizing the means squared error via the so-called cost function

<em>n−</em>1

1 X

<em>C</em>(<em>X,β</em>) = <em>                                                           .</em>

<em>n</em>

<em>i</em>=0

Show that you can rewrite this as

<sup>X</sup>(<em>f<sub>i </sub></em>])<sup>2 </sup>+ <sup>1 X </sup><em>− </em>E[<em>y</em><strong>˜</strong>])<sup>2 </sup>+ <em>σ</em><sup>2</sup><em>. − </em>E[<em>y</em><strong>˜      </strong>(<em>y</em>˜<em><sub>i </sub>n       n</em>

<em>i                                                           i</em>

Explain what the terms mean, which one is the bias and which one is the variance and discuss their interpretations.

Discuss the bias and variance tradeoff as function of your model complexity

(the degree of the polynomial) and the number of data points, and possibly also your training and test data.

Try to make a figure similar to Fig. 2.11 of Hastie, Tibshirani, and Friedman, see the references below. You will most likely not get an equally smooth curve!

<strong>Part d): Ridge Regression on the Franke function with resampling. </strong>Write your own code for the Ridge method, either using matrix inversion or the singular value decomposition as done in the previous exercise or howework 2 (see also chapter 3.4 of Hastie <em>et al.</em>, equations (3.43) and (3.44)). Perform the same analysis as in the previous exercises (for the same polynomials and include resampling techniques) but now for different values of <em>λ</em>. Compare and analyze your results with those obtained in parts a-c). Study the dependence on <em>λ</em>.

Study also the bias-variance tradeoff as function of various values of the parameter <em>λ</em>. Comment your results.

<strong>Part e): Lasso Regression on the Franke function with resampling. </strong>This part is essentially a repeat of the previous two ones, but now with Lasso regression. Write either your own code or, in this case, you can also use the functionalities of <strong>Scikit-Learn </strong>(recommended). Give a critical discussion of the three methods and a judgement of which model fits the data best.

<strong>Part f): Introducing real data. </strong>With our codes functioning and having been tested properly on a simpler function we are now ready to look at real data. We will essentially repeat in part g) what was done in parts a-e). However, we need first to download the data and prepare properly the inputs to our codes. We are going to download digital terrain data from the website <a href="https://earthexplorer.usgs.gov/">https:</a>

<a href="https://earthexplorer.usgs.gov/">//earthexplorer.usgs.gov/</a><a href="https://earthexplorer.usgs.gov/">,</a>

In order to obtain data for a specific region, you need to register as a user

(free) at this website and then decide upon which area you want to fetch the digital terrain data from. In order to be able to read the data properly, you need to specify that the format should be <strong>SRTM Arc-Second Global </strong>and download the data as a <strong>GeoTIF </strong>file. The files are then stored in <em>tif </em>format which can be imported into a Python program using scipy.misc.imread

Here is a simple part of a Python code which reads and plots the data from

such files

import numpy as np from imageio import imread import matplotlib.pyplot as plt from mpl_toolkits.mplot3d import Axes3D from matplotlib import cm

# Load the terrain terrain1 = imread(’SRTM_data_Norway_1.tif’)

# Show the terrain plt.figure() plt.title(’Terrain over Norway 1’) plt.imshow(terrain1, cmap=’gray’) plt.xlabel(’X’) plt.ylabel(’Y’) plt.show()

If you should have problems in downloading the digital terrain data, we provide two examples under the data folder of project 1. One is from a region close to Stavanger in Norway and the other Møsvatn Austfjell, again in Norway. Feel free to produce your own terrain data.

<strong>Part g) OLS, Ridge and Lasso regression with resampling. </strong>Our final part deals with the parameterization of your digital terrain data. We will apply all three methods for linear regression as in parts a-c), the same type (or higher order) of polynomial approximation and the same resampling techniques to evaluate which model fits the data best.

At the end, you should pesent a critical evaluation of your results and discuss the applicability of these regression methods to the type of data presented here.

<h1>Background literature</h1>

<ol>

 <li>For a discussion and derivation of the variances and mean squared errors using linear regression, see the <a href="https://arxiv.org/abs/1509.09169">Lecture notes on ridge regression by Wessel</a></li>

 <li><a href="https://arxiv.org/abs/1509.09169"> van Wieringen</a></li>

 <li>The textbook of <a href="https://www.springer.com/gp/book/9780387848570">Trevor Hastie, Robert Tibshirani, Jerome H. Friedman, </a><a href="https://www.springer.com/gp/book/9780387848570">The Elements of Statistical Learning, Springer</a><a href="https://www.springer.com/gp/book/9780387848570">,</a> chapters 3 and 7 are the most relevant ones for the analysis here.</li>

</ol>

<h1>Introduction to numerical projects</h1>

Here follows a brief recipe and recommendation on how to write a report for each project.

<ul>

 <li>Give a short description of the nature of the problem and the eventual numerical methods you have used.</li>

 <li>Describe the algorithm you have used and/or developed. Here you may find it convenient to use pseudocoding. In many cases you can describe the algorithm in the program itself.</li>

 <li>Include the source code of your program. Comment your program properly.</li>

 <li>If possible, try to find analytic solutions, or known limits in order to test your program when developing the code.</li>

 <li>Include your results either in figure form or in a table. Remember to label your results. All tables and figures should have relevant captions and labels on the axes.</li>

 <li>Try to evaluate the reliabilty and numerical stability/precision of your results. If possible, include a qualitative and/or quantitative discussion of the numerical stability, eventual loss of precision etc.</li>

 <li>Try to give an interpretation of you results in your answers to the problems.</li>

 <li>Critique: if possible include your comments and reflections about the exercise, whether you felt you learnt something, ideas for improvements and other thoughts you’ve made when solving the exercise. We wish to keep this course at the interactive level and your comments can help us improve it.</li>

 <li>Try to establish a practice where you log your work at the computerlab. You may find such a logbook very handy at later stages in your work, especially when you don’t properly remember what a previous test version of your program did. Here you could also record the time spent on solving the exercise, various algorithms you may have tested or other topics which you feel worthy of mentioning.</li>

</ul>

<h1>Format for electronic delivery of report and programs</h1>

The preferred format for the report is a PDF file. You can also use DOC or postscript formats or as an ipython notebook file. As programming language we prefer that you choose between C/C++, Fortran2008 or Python. The following prescription should be followed when preparing the report:

<ul>

 <li>Use Devilry to hand in your projects, log in at <a href="http://devilry.ifi.uio.no/">http://devilry.ifi. </a><a href="http://devilry.ifi.uio.no/">no</a> with your normal UiO username and password and choose either ’fysstk3155’ or ’fysstk4155’. There you can load up the files within the deadline.</li>

 <li>Upload <strong>only </strong>the report file! For the source code file(s) you have developed please provide us with your link to your github domain. The report file should include all of your discussions and a list of the codes you have developed. Do not include library files which are available at the course homepage, unless you have made specific changes to them.</li>

 <li>In your git repository, please include a folder which contains selected results. These can be in the form of output from your code for a selected set of runs and input parameters.</li>

 <li>In this and all later projects, you should include tests (for example unit tests) of your code(s).</li>

 <li>Comments from us on your projects, approval or not, corrections to be made etc can be found under your Devilry domain and are only visible to you and the teachers of the course.</li>

</ul>

Finally, we encourage you to collaborate. Optimal working groups consist of 2-3 students. You can then hand in a common report.

<h1>Software and needed installations</h1>

If you have Python installed (we recommend Python3) and you feel pretty familiar with installing different packages, we recommend that you install the following Python packages via <strong>pip </strong>as

<ol>

 <li>pip install numpy scipy matplotlib ipython scikit-learn tensorflow sympy pandas pillow</li>

</ol>

For Python3, replace <strong>pip </strong>with <strong>pip3</strong>.

See below for a discussion of <strong>tensorflow </strong>and <strong>scikit-learn</strong>.

For OSX users we recommend also, after having installed Xcode, to install <strong>brew</strong>. Brew allows for a seamless installation of additional software via for example

<ol>

 <li>brew install python3</li>

</ol>

For Linux users, with its variety of distributions like for example the widely popular Ubuntu distribution you can use <strong>pip </strong>as well and simply install Python as

<ol>

 <li>sudo apt-get install python3 (or python for python2.7)</li>

</ol>

etc etc.

If you don’t want to install various Python packages with their dependencies separately, we recommend two widely used distrubutions which set up all relevant dependencies for Python, namely

<ol>

 <li><a href="https://docs.anaconda.com/">Anaconda</a> Anaconda is an open source distribution of the Python and R programming languages for large-scale data processing, predictive analytics, and scientific computing, that aims to simplify package management and deployment. Package versions are managed by the package management system <strong>conda</strong></li>

 <li><a href="https://www.enthought.com/product/canopy/">Enthought canopy</a> is a Python distribution for scientific and analytic computing distribution and analysis environment, available for free and under a commercial license.</li>

</ol>

Popular software packages written in Python for ML are

<ul>

 <li><a href="http://scikit-learn.org/stable/">Scikit-learn</a><a href="http://scikit-learn.org/stable/">,</a></li>

 <li><a href="https://www.tensorflow.org/">Tensorflow</a><a href="https://www.tensorflow.org/">, </a> <a href="http://pytorch.org/">PyTorch</a> and</li>

 <li><a href="https://keras.io/">Keras</a><a href="https://keras.io/">.</a></li>

</ul>

These are all freely available at their respective GitHub sites. They encompass communities of developers in the thousands or more. And the number of code developers and contributors keeps increasing.