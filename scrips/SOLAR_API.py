import numpy as np
import GPy
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'core'))
print(join(dirname(dirname(abspath(__file__))), 'core'))
import osgpr_GPy
import time

# sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'utilities'))
# import circstats
from copy import copy, deepcopy
from matplotlib import pyplot as plt

def tic():
    return time.time()


def toc(tstart, taskName=""):
    print('%s took: %s sec.\n' % (taskName, (time.time() - tstart)))

class SOLARGP():


    def __init__(self, num_idc=15, wgen=0.975,  max_num_models=10, dim_in = 0, dim_out = 0):

        """
        num_idc:            number of inducing support points, used to sparse the GP model
        dim_in:             input data dimension
        dim_out:            output data dimension

        wgen:               distance metric threshold, used for determine when to generate new local model
                            * it requires noting that the distance metric between a data point and k-th model center is defined by:
                                    w_k = exp(-0.5*(x-c_k).T*W*(x-c_k))
                            Thus the distance between the data piont and the model center is inverse proportional to w_k, and  
                            when w_k < wgen, new model is generated

        mdrift:             a GP model used to initialize the local models (in general sparseGPR from Gpy is used), that is 
                            local models 'drift to' mdrift
        W:                  1/lengthscale**2, width hyperparameters of drift (global?) GP
        models:             local sparseGPR models (from Gpy)
        localDataParts:         a list of dictionary recording the information of each data partition
                            dictionary elements:
                            'nop': total number of data points in this partition
                            'c_i': center of input data of the corresponding local model
                            'c_o': center of output data of the correspinding local model
                            'hasInit': bool flag, set true when the corresponding local model has been initialized (generally with mdrift) 
                            after the new partition occur' set false when the new model corresponding to the new partition has not been initialized.
        num_models:         number of local models
        max_num_models:     max allowed number of partitions

        new_in_part:        parted (according to local center) new input data (used for training)
        new_out_part:       parted (according to local center) new output data (used for training)
        """

        self.wgen = wgen 
        self.W = 0
        self.mdrift = 0

        self.models = [] 
        self.localDataParts = []
        self.num_models = len(self.localDataParts)
        self.max_num_models = max_num_models

        self.num_idc = num_idc
        self.dim_in = dim_in
        self.dim_out = dim_out 

        self.new_in_part = []
        self.new_out_part = []

    def init(self, in_init, out_init):
        '''
        This function initialize the overall solar-gp with given initial training set

        in_init:        initial input data, size n*dim_in
        out_init:       corresponding initial output data, size n*dim_out
        '''

        # initilize input and output dimension
        self.dim_in = in_init.shape[1]
        self.dim_out = out_init.shape[1]

        # initialize mdrift
        self.init_mdrift(in_init,out_init)

        # partition the initial data into local models
        self.partition(in_init,out_init)

        # train the whole solar-gp to fit the initial data
        self.train_local()



    def init_mdrift(self, in_init, out_init):
        '''
        This function initialize a drfit gp model as the base of local models

        in_init:        initial input data, size n*dim_in
        out_init:       corresponding initial output data, size n*dim_out
        '''
        if in_init.shape[0] < self.num_idc:
            sys.exit('Not enough initial data for SOLAR-GP initialization.')

        # randomly select num_idc data points from in_init as the inducing points for the sparse GP
        Z = in_init[np.random.permutation(in_init.shape[0])[0:self.num_idc],:]

        # initialize mdrift as a sparse GP
        self.mdrift = GPy.models.SparseGPRegression(in_init,out_init,GPy.kern.RBF(self.dim_in,ARD=True),Z=Z)
        self.mdrift.optimize()

        # find the width hyperparameter
        W_diag = np.zeros(self.dim_in)
        for i in range(self.dim_in):
            W_diag[i] = 1/self.mdrift.kern.lengthscale[i]**2
        self.W = np.diag(W_diag)

    def partition(self, in_new, out_new):
        '''
        This function parts the new streaming data into partitions according to the distance metric wgen
        in_new:             new input data
        out_new:            corresponding new output data
        '''
        for i in range(in_new.shape[0]):
            if self.num_models > 0:
                # compute the distance metric w between current input and the centers of local models
                w = np.zeros(self.num_models)
                for k in range(self.num_models):
                    # difference between input and center
                    x_c = (in_new[i] - self.localDataParts[k]['c_i']).reshape((self.dim_in,1)) 

                    w[k] = np.exp(-0.5*x_c.T.dot(self.W).dot(x_c))
                
                w_near = np.max(w)
                near_idx = np.argmax(w)

            else: # when there is no local models 
                w_near = -1

            # when the new data point locates inside an existing local model or max number of local model has been reached
            if w_near > self.wgen or self.num_models == self.max_num_models:
                if np.any(self.new_in_part[near_idx] == None):
                    self.new_in_part[near_idx] = in_new[i].reshape(1,self.dim_in)
                    self.new_out_part[near_idx] = out_new[i].reshape(1,self.dim_out)
                else:
                    self.new_in_part[near_idx] = np.vstack((self.new_in_part[near_idx],in_new[i]))
                    self.new_out_part[near_idx] = np.vstack((self.new_out_part[near_idx],out_new[i]))

                # update the information of the correspoding data part
                nop = self.localDataParts[near_idx]['nop'] # number of data points in this part
                self.localDataParts[near_idx]['nop'] += 1
                self.localDataParts[near_idx]['c_i'] = (in_new[i]+self.localDataParts[near_idx]['c_i']*nop)/(nop+1)
                self.localDataParts[near_idx]['c_o'] = (out_new[i]+self.localDataParts[near_idx]['c_o']*nop)/(nop+1)

            elif w_near < self.wgen and self.num_models < self.max_num_models:
                self.new_in_part.append(in_new[i].reshape(1,self.dim_in))
                self.new_out_part.append(out_new[i].reshape(1,self.dim_out))

                # add the new local model information to localModelParts
                m_new = {}
                m_new['nop'] = 1
                m_new['c_i'] = in_new[i].reshape(1,self.dim_in)
                m_new['c_o'] = out_new[i].reshape(1,self.dim_out)
                m_new['hasInit'] = False
                self.localDataParts.append(m_new)

                self.num_models += 1

    def train_global(self, in_new, out_new, m_old, num_idc, use_old_Z=True, driftZ=False, fixTheta=False):
        '''
        This function implements the online training of the global (mdrift) model
        '''
        self.mdrift = self.doOSGPR(in_new, out_new, m_old, num_idc, use_old_Z, driftZ, fixTheta)

        # find the width hyperparameter
        W_diag = np.zeros(self.dim_in)
        for i in range(self.dim_in):
            W_diag[i] = 1/self.mdrift.kern.lengthscale[i]**2
        self.W = np.diag(W_diag)

    def train_local(self):
        '''
        This function implements the online training of local models
        In detail, each local model is trained (updated) given the new data partition
        '''
        for i, dataPart in enumerate(self.localDataParts):
            if dataPart['hasInit']: # the local model has been initialized
                if np.any(self.new_in_part[i] == None): # no new data for this local model
                    continue
                else:
                    # take in count of the new data
                    self.models[i] = self.doOSGPR(self.new_in_part[i],self.new_out_part[i],self.models[i],
                                                  self.num_idc,use_old_Z=True,fixTheta=False)

                    # empty the trained (updated) parts of data
                    self.new_in_part[i] = None
                    self.new_out_part[i] = None

            else:
                print('New local model added')
                m_new = self.doOSGPR(self.new_in_part[i],self.new_out_part[i],self.mdrift,
                                    self.num_idc,use_old_Z=False,driftZ=False,fixTheta=False)
                
                # update the information of the new local model
                self.models.append(m_new)
                self.localDataParts[i]['hasInit'] = True

                # empty the trained (updated) parts of data
                self.new_in_part[i] = None
                self.new_out_part[i] = None


    def doOSGPR(self, in_new, out_new, m_old, num_idc, use_old_Z=True, driftZ=False, fixTheta=False):
        '''
        This function implement the online sparse GP regression to update (train) local models according to the new data
        '''
        # record the old inducing points Z and its parameters
        cur_Z = copy(m_old.Z.param_array)
        mu, Su = m_old.predict(cur_Z, full_cov = True)
        Su = Su + 5e-4*np.eye(mu.shape[0])

        Kaa = m_old.kern.K(cur_Z)
        kern = GPy.kern.RBF(self.dim_in,ARD=True)
        kern.variance = copy(m_old.kern.variance)
        kern.lengthscale = copy(m_old.kern.lengthscale)

        # merge the old Z and the new input data
        Z_new = self.updateZ(cur_Z, in_new, num_idc, use_old_Z, driftZ)

        m_new = osgpr_GPy.OSGPR_VFE(in_new, out_new, kern, mu, Su, Kaa,
            cur_Z, Z_new, m_old.likelihood)


        "Fix parameters"
        if driftZ:
            m_new.Z.fix()

        if fixTheta:
            m_new.kern.fix()
            m_new.likelihood.variance.fix()

        # optimization stuff
        m_new.optimize()
        m_new.Z.unfix()
        m_new.kern.unfix()
        m_new.likelihood.variance.unfix()

        return m_new

    def updateZ(self, cur_Z, in_new, num_idc, use_old_Z=True,driftZ=False):
        """
        This function is used to update the set of inducing points of each local model, 
        following are basic ideas that this function follows:
            1) Add new experience points up to num_inducing (if we change num_inducing)
            2) randomly delete current points down to num_inducing
            3) always add new experience into support point (and randomly delete current support point(s))
        """
        ############################################
        # not sure about this part, come back for this when debuging
        ############################################

        num_idc_old = cur_Z.shape[0]
        if driftZ:
            if num_idc_old < num_idc:
                num_idc_add = num_idc - num_idc_old
                old_Z = cur_Z[np.random.permutation(num_idc_old), :]
                new_Z = in_new[np.random.permutation(in_new.shape[0])[0:num_idc_add],:]
                Z = np.vstack((old_Z, new_Z))
                #print(len(Z))
            else:
                num_idc_add = len(in_new)
                old_Z = cur_Z[num_idc_add:,:]
                new_Z = in_new[np.random.permutation(in_new.shape[0])[0:num_idc_add], :]
                Z = np.vstack((old_Z, new_Z))

        elif num_idc_old < num_idc:
            num_idc_add = num_idc - num_idc_old
            old_Z = cur_Z[np.random.permutation(num_idc_old), :]
            new_Z = in_new[np.random.permutation(in_new.shape[0])[0:num_idc_add],:]
            Z = np.vstack((old_Z, new_Z))

        elif num_idc_old > num_idc:
            num_idc_rm = num_idc_old - num_idc
            old_Z = cur_Z[np.random.permutation(num_idc_old), :]
            Z = np.delete(old_Z, np.arange(0,num_idc_rm),axis = 0)
        elif use_old_Z:
            Z = np.copy(cur_Z)

        else:
            num_idc_stay = num_idc_old - len(in_new)
            num_idc_add = len(in_new)
            old_Z = cur_Z[np.random.permutation(num_idc_old)[0:num_idc_stay], :]
            new_Z = in_new[np.random.permutation(in_new.shape[0])[0:num_idc_add], :]
            Z = np.vstack((old_Z, new_Z))
        return Z


    def prediction(self,in_test, weighted=True, max_model_weighted=3, Y_prev = []):
        #######################################
        # original version
        #######################################
        ypred = np.empty([np.shape(in_test)[0], self.dim_out])
        for n in range(0, np.shape(in_test)[0], 1):
            w = np.empty([self.num_models, 1])
            dcw = np.empty([self.num_models, 1])

            yploc = np.empty([self.num_models,self.dim_out])
            #xploc = np.empty([self.M,self.xdim])

            var = np.empty([self.num_models,1])
            for k in range(0, self.num_models, 1):


                try:
                    c = self.localDataParts[k]['c_i'] #1x2 # input center
                    d = self.localDataParts[k]['c_o'] # output center

                    xW = np.dot((in_test[n]-c),self.W) # 1x2 X 2x2
                    w[k] = np.exp(-0.5*np.dot(xW,np.transpose((in_test[n]-c))))
                    yploc[k], var[k] = self.models[k].predict(in_test[n].reshape(1,self.dim_in))

                    if Y_prev == []:
                        pass
                    else:
                        dcw[k] = np.dot(d-Y_prev[-1].reshape(1,self.dim_out),np.transpose(d-Y_prev[-1].reshape(1,self.dim_out)))
                except:

                    w[k] = 0
                    dcw[k] = float("inf")
                    pass

            if weighted:
                if max_model_weighted > self.num_models:
                    h = self.num_models
                else:
                    h = max_model_weighted
            else:
                h = 1

            s = 0
            if Y_prev == []:
                wv = w/var
            else:
                wv = w*np.exp(-s*dcw)/var
                # wv = w*np.exp(-var)

            wv =np.nan_to_num(wv)
            
            wv = wv.reshape(self.num_models,)
            varmin = np.min(var) # minimum variance of local predictions
            thresh = 0 # 0 uses all models

            "Select for best models"
            # if np.max(wv) < thresh:
            #     ind = wv ==np.max(wv)
            # else:
            #     ind = wv > thresh


            ind = np.argpartition(wv, -h)[-h:]
            # ypred[n] = np.dot(np.transpose(wv[ind]), yploc[ind]) / np.sum(wv[ind])

        
            "Normal Weighted mean"
            ypred[n] = np.dot(np.transpose(wv[ind]), yploc[ind]) / np.sum(wv[ind])

            "Debug Prints"
            # print("wv:" + str(wv))
            # print("w:" + str(w))
            # print("dw:" + str(dw))
            # print("dc:" + str(dc))
            # print("var:" + str(var))
            # print("dcw:" + str(dcw))
            # print("d:" + str(d))
            # print("Yprev:" + str(Y_prev[-1].reshape(1,self.ndim)))
            # print("yploc:" + str(yploc))
            # print("xploc:" + str(xploc))

            # print("ypred:" + str(ypred))


        return ypred, varmin

    def predict(self, in_test, max_model_weighted=None):
        '''
        This function predict the output values corresponding to the test input
        in_test:                test input dataset, n*self.dim_in, n is the number of test data points
        max_model_weighted:     max number of model weighted in prediction. Default None means all local models are weighted
        '''
        # initialize output predictions
        y_pred = np.zeros((in_test.shape[0], self.dim_out))

        # predict based on all test input
        for i in range(in_test.shape[0]):
            # distance metric, used for local model weight computation
            w = np.zeros((self.num_models,1))

            # initialize predicted local mean and variance of output
            y_mu_loc = np.zeros((self.num_models, self.dim_out))
            var_loc = np.zeros((self.num_models,1))

            # iterate through all local models
            for k in range(self.num_models):
                x_c = (in_test[i] - self.localDataParts[k]['c_i']).reshape(self.dim_in,1)
                w[k] = np.exp(-0.5*x_c.T.dot(self.W).dot(x_c))
                y_mu_loc[k], var_loc[k] = self.models[k].predict(in_test[i].reshape(1,self.dim_in))

            if max_model_weighted == None:
                max_model_weighted = self.num_models
            
            # get the index of top max_model_weighted distance metric value (w)
            weighted_model_idx = np.argpartition(w.reshape(self.num_models,),-max_model_weighted)[-max_model_weighted:]

            # LGPR, by Duy, Jan's version
            y_pred[i] = (y_mu_loc[weighted_model_idx].reshape(self.dim_out,weighted_model_idx.shape[0]).dot(w[weighted_model_idx])/w[weighted_model_idx].sum()).reshape(self.dim_out,)
            
            # # Brian's version
            # y_pred[i] = y_mu_loc[weighted_model_idx].T.dot(w[weighted_model_idx]).dot(np.exp(-var_loc[weighted_model_idx]))
            #             /w[weighted_model_idx].T.dot(np.exp(-var_loc[weighted_model_idx]))
        
        return y_pred


# test functions
def test_sin():
    t_start = tic()
    t_step = copy(t_start)
    # construct a SOLARGP to learn sin function
    SOLARGP_sin = SOLARGP(num_idc=15, wgen=0.975,  max_num_models=3)

    # generate data for initialization
    in_init = np.linspace(0.0,2*np.pi,num=20).reshape(20,1)
    out_init = np.sin(in_init)

    # initialization
    SOLARGP_sin.init(in_init,out_init)

    # pseudo-streaming data points
    in_stream = np.linspace(0.0,10*np.pi, num=150).reshape(150,1)
    out_stream = np.sin(in_stream)

    # record predicted output
    out_predict = np.zeros(out_stream.shape)

    for i in range(150):
        if (i+1)%10 == 0:
            # new training stream data
            in_new = in_stream[i:i+15]
            out_new = out_stream[i:i+15]

            # train global model (mdrift)
            SOLARGP_sin.train_global(in_new,out_new,SOLARGP_sin.mdrift,30,use_old_Z=False)

            # patition
            SOLARGP_sin.partition(in_new,out_new)

            # train local model
            SOLARGP_sin.train_local()

            # timing
            toc(t_step, '%d training'%((i+1)/10))
            t_step = tic()

        # predict
        out_predict[i] = SOLARGP_sin.predict(in_stream[i]+1e-1)

    toc(t_start,'Toy Example')
    # plot
    plt.plot(in_stream+1e-1,out_predict,label='predict')
    plt.plot(in_stream,out_stream,'--',label='true value')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    test_sin()



            
