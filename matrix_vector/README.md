#Description

The file matvec.cpp contains code for Ax = y matrix-vector dot product calculation. A is a n x n matrix. x is a vector of dimension n and y is also a vector of dimension n. The dimension is specified during runtime.

#Compilation
gcc -fopenmp -o matvec matvec.cpp

#Run Command
./matvec 1024

The above command will create a Matrix A of dimension 1024x1024 with random integers filled in as its elements. And it will also generate a vector x with a size of n which is also filled in with random integers.

#Observations
The primary objective is to observe the phenomenon of "False Sharing" across Cores. There are extensive comments in the code to explain what's happening. Maybe some day I can write a blog post.
There are report files mat_vec.<number> which contain the results of the simulation on Texas A&M's ADA clusters. One thing which I still don't understand is why dynamic scheduling with low chunk size perfroms so badly. Strange!!
