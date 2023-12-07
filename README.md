# point-cloud-toolbox


Python library for point cloud processing. The main emphasis is on discrete curvature measures, for now. 

The main approach (used in run.py) for calculating the approximate discrete curvatures is essentially:
1. A k-dimensional tree is constructed from the input point cloud to organize the data spatially.
2. For each point, the k nearest neighbors are determined using the k-d tree, supporting both distance-based and epsilon-ball queries.
3. The singular value decomposition (SVD) is employed on each neighborhood to ascertain the characteristic plane, aligning with the first two eigenvectors.
4. The neighborhood is rotated to align this plane with the xy-axis, repositioning the central point at the origin. The z-axis is approximated as the normal.
5. Consistency in curvature signs is ensured by adjusting the orientation based on the dot product between the z-axis and vectors in the neighborhood.
6. A quadratic surface is fitted to the neighborhood points using least-squares regression on a cost-function learning basis, yielding an explicit function F(x, y) = z, represented by using weights as coefficients.
7. Curvatures are computed using classical differential geometry sources such as Do Carmo, Spivak, and Gauss

The utilities live within the PointCloud class in pointCloudToolbox.py, you can see the implementation of known expressions for curvature within.

The work presented here is part of active research at the University of Nevada, Reno - please contact me if you would like to talk about the tools within.

rhutton@unr.edu

![alt text](https://github.com/masnottuh/point-cloud-toolbox/blob/main/img/bunny1.png)
![alt text](https://github.com/masnottuh/point-cloud-toolbox/blob/main/img/bunny2.png)
![alt text](https://github.com/masnottuh/point-cloud-toolbox/blob/main/img/carton1.png)
![alt text](https://github.com/masnottuh/point-cloud-toolbox/blob/main/img/carton2.png)
![alt text](https://github.com/masnottuh/point-cloud-toolbox/blob/main/img/sridge.png)
![alt text](https://github.com/masnottuh/point-cloud-toolbox/blob/main/img/torus.png)
