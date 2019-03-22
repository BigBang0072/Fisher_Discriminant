import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##############################################################################
########################### UTILITY FUNCTION #################################
##############################################################################
def data_handling(datapath):
    '''
    This function will retreieve the data in a dataframe.
    '''
    df=pd.read_csv(datapath,header=None)
    df.columns=["seq","x1","x2","class"]
    df.drop("seq",axis=1,inplace=True)

    print("Printing the head of the dataframe:")
    print(df.head())

    return df

class Fisher_Discriminant():
    #Attributes to the discriminant
    W=None              #The intra-class variance minimizing direction
    W_mean=None         #Just the inter-class mean distance maximizing dirn
    points_class1=None  #Points belonging to class1
    points_class2=None  #Points belonging to class2
    dataset=None


    def get_fisher_discriminant_function(self,dataframe):
        '''
        DESCRIPTION:
            This function will get the projection direction which will maximize the
            distance between the class mean and minimize the intra-class variance.

            Also, we will generate the decision boundary based on the normal-
            distribution of projection of each class on the line.

            The projection direction which we will get will be the normal to the
            discriminating line. So equivalently we are getting the discriminating
            function, by projecting the points in normal direction to that line.
        USAGE:
            INPUT:
                dataframe:  the dataframe holding the points and their class
            OUTPUT:

        '''
        print("\nCalcualting the projection vector")
        #Retreiving the points and class a numpy array
        self.points_class1=df[df["class"]==0][["x1","x2"]].values
        self.points_class2=df[df["class"]==1][["x1","x2"]].values
        # print(points_class2)

        #Calculating the mean of each class
        m1=np.mean(self.points_class1,axis=0)
        m2=np.mean(self.points_class2,axis=0)
        print(m1,m2)
        #Getting the simple projection dir which maximize the mean diff of class
        self.W_mean = (m1-m2).reshape(-1,1)

        #Calculating the covariance matrix for each class
        covar_mat1=self._calculate_covaraince_matrix(self.points_class1,m1)
        covar_mat2=self._calculate_covaraince_matrix(self.points_class2,m2)
        Sw=covar_mat1+covar_mat2

        #Now calculating the projection vector
        self.W = np.matmul(np.linalg.inv(Sw),self.W_mean)
        #Normalizing the projection direction
        self.W = self.W/(np.linalg.norm(self.W))
        print("Shape of projection direction: ",self.W.shape)
        print("Projection Vector:\n ",self.W)

    def _calculate_covaraince_matrix(self,points,mean):
        '''
        This function will calculate the covariance matrix on the given data
        '''
        #Getting the difference matrix
        diff=points-mean
        covar_mat=np.matmul(diff.T,diff)

        return covar_mat

    def _plot_the_actual_points(self):
        '''
        Assumption: The points are 2-dimensional.
        '''
        #Plotting the actual points
        plt.plot(self.points_class1[:,0],
                    self.points_class1[:,1],"r*",alpha=0.2)
        plt.plot(self.points_class2[:,0],
                    self.points_class2[:,1],"g.",alpha=0.2)

        #Plotting the direction of projection or normal to decision boundary
        plt.plot([0.0,self.W[0,0]],[0.0,self.W[1,0]],"k-",alpha=1)
        #Plotting the direction of projection got from mean diff max
        plt.plot([0.0,self.W_mean[0,0]],[0.0,self.W_mean[1,0]],"k--",alpha=0.6)
        plt.grid()
        # plt.show()

    def _plot_the_projection(self):
        '''
        This function will plot the projections along the direction given
        by W.
        '''
        #Projecting the points of class1 on the W direction
        proj_class1=np.matmul(self.points_class1,self.W)
        proj_class2=np.matmul(self.points_class2,self.W)

        #Now getting the position vector along the W direction (back to 2D)
        proj2D_class1=np.matmul(proj_class1,self.W.T)
        proj2D_class2=np.matmul(proj_class2,self.W.T)

        #Plotting the projecting points back on the 2D plane
        plt.plot(proj2D_class1[:,0],proj2D_class1[:,1],"r*",alpha=1)
        plt.plot(proj2D_class2[:,0],proj2D_class2[:,1],"g.",alpha=1)
        plt.grid()
        plt.show()

if __name__=="__main__":
    datapath="dataset/ML-Assignment1-Datasets/dataset_3.csv"
    df=data_handling(datapath)

    #Getting projection and corresponding boundary
    fd1=Fisher_Discriminant()
    fd1.get_fisher_discriminant_function(df)
    fd1._plot_the_actual_points()
    fd1._plot_the_projection()
