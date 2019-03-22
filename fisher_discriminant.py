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

def gaussian_probability(x,mu,sigma):
    prob=np.exp(-((x-mu)**2/(2.0*sigma**2)))/sigma

    return prob

class Fisher_Discriminant():
    #Attributes to the discriminant
    #Decision boundary specific paramters
    W=None              #The intra-class variance minimizing direction
    W_mean=None         #Just the inter-class mean distance maximizing dirn
    mu1=None            #The mean estimate of the normal distribution of class1
    sigma1=None         #The std dev of normal dist of class1
    mu2=None
    sigma2=None
    decision_point=None #The point where the two normal curve intersect

    #Data specific parameters
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

    def estimate_the_normal_dist_params(self):
        '''
        Assumption: The task at hand is just binary classification.
        After projecting the data, we have to get the decision boundary
        on the one-dimensional projected real line.

        Idea 1: We can do this naively by taking the mean of the mid-points
                of the each of the projected class.
        Idea 2: Alternatively we could fit one Normal distribution for each
                of the class on the projected data.
                And take the intersection point of the normal dixtribution
                as the decision boundary.
        '''
        #Finding the parameters for the normal distribution of class 1
        proj_class1=np.matmul(self.points_class1,self.W)
        self.mu1=np.mean(proj_class1)
        self.sigma1=np.std(proj_class1)
        #Finding the parameters for class2's normal distribution
        proj_class2=np.matmul(self.points_class2,self.W)
        self.mu2=np.mean(proj_class2)
        self.sigma2=np.std(proj_class2)

    def estimate_decision_boundary(self):
        '''
        This function will iteratively estimate the intersection point
        of the normal distribution of each of the class.
        '''
        #Estimating the normal distribution of each class projection
        self.estimate_the_normal_dist_params()

        #Initializing the low and high estimate
        hi_mu=self.mu2
        lo_mu=self.mu1
        hi_sigma=self.sigma2
        lo_sigma=self.sigma1
        if(self.mu1>self.mu2):
            lo_mu=self.mu2
            lo_sigma=self.sigma2
            hi_mu=self.mu1
            hi_sigma=self.sigma1


        #Now doing binary search on the intersection point
        print("Calculating the decision point")
        hi=hi_mu
        lo=lo_mu
        print("Initial estiamte of lo:{} hi:{}".format(lo,hi))
        mid=(hi+lo)/2
        prob1=gaussian_probability(mid,lo_mu,lo_sigma)
        prob2=gaussian_probability(mid,hi_mu,hi_sigma)
        while(np.abs(prob1-prob2)>1e-5):
            if(prob1>prob2):
                lo=mid
            else:
                hi=mid

            mid=(lo+hi)/2
            prob1=gaussian_probability(mid,lo_mu,lo_sigma)
            prob2=gaussian_probability(mid,hi_mu,hi_sigma)

        self.decision_point=mid
        print("Decision point is: ",mid)

    def plot_class_normal_distribution():
        

if __name__=="__main__":
    datapath="dataset/ML-Assignment1-Datasets/dataset_3.csv"
    df=data_handling(datapath)

    #Getting projection and corresponding boundary
    fd1=Fisher_Discriminant()
    fd1.get_fisher_discriminant_function(df)
    fd1._plot_the_actual_points()
    fd1._plot_the_projection()
    fd1.estimate_decision_boundary()
