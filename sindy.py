# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:33:55 2019

@author: agoldschmidt
"""
import numpy as np
import abc

from itertools import combinations

import scipy as sci
from scipy import interpolate, integrate
from sklearn import clone

class Derivative(abc.ABC):
    '''Object for computing numerical derivatives for use in SINDy.
    Derivatives are allowed to return np.nan if the implementation
    fails to compute a good derivative.'''
    
    @abc.abstractmethod
    def compute(self, t, x, i):
        '''Compute derivative (dx/dt)[i]'''
        
    def compute_for(self, t, x, indices):
        '''Compute derivative (dx/dt)[i] for i in indices. Overload this if
        desiring a more efficient computation over a list of indices.'''
        for i in indices:
            yield self.compute(t, x, i)


class SavitzkyGolay(Derivative):
    '''Compute the numerical derivative by first finding the best 
    (least-squares) polynomial of order m < 2k using the points in
    [t-left, t+right]. The derivative is computed from the coefficients
    of the polynomial.
    Arguments:
        params['left']: left edge of the window is t-left
        params['right']: right edge of the window is t+right
        params['order']: order of polynomial (m < points in window)
    '''  
    def __init__(self, params):
        self.left = params['left']
        self.right = params['right']
        self.order = params['order']

    def compute(self, t, x, i):
        '''Requires the (t,x) data to be sorted,
        '''

        i_l = np.argmin(np.abs(t - (t[i] - self.left)))
        i_r = np.argmin(np.abs(t - (t[i] + self.right)))
        
        # window too sparse. TODO: issue warning.
        if self.order > (i_r - i_l): 
            return np.nan
        
        # Construct polynomial in t and do least squares regression
        try:
            polyn_t = np.array([np.power(t[i_l:i_r], n)
                            for n in range(self.order+1)]).T
            w,_,_,_ = np.linalg.lstsq(polyn_t, x[i_l:i_r], rcond=None)
        except np.linalg.LinAlgError:
            # Failed to converge, return bad derivative
            return np.nan

        # Compute derivative from fit
        return np.sum([j*w[j]*np.power(t[i], j-1)
                       for j in range(1, self.order+1)])

    def compute_for(self, t, x, indices):
        # If the window cannot reach any points, throw an exception
        # (likely the user forgets to rescale the window parameter)
        if min(t[1:] -t[:-1]) > max(self.left, self.right):
            raise ValueError("Found bad window ({}, {}) for x-axis data."
                .format(self.left, self.right))
        for d in super().compute_for(t, x, indices):
            yield d

 
class CubicSpline(Derivative):
    '''Compute the numerical derivative of y using a (Cubic) spline (the Cubic 
    spline minimizes the curvature of the fit). Compute the derivative from 
    the form of the known Spline polynomials.
    Arguments:
        params['order']: Default is cubic spline (3)
        params['smoothing']: Amount of smoothing
        params['periodic']: Default is False
    '''
    def __init__(self, params):
        self.order = 3
        if 'order' in params:
            self.order = params['order']
            
        self.periodic = False
        if 'periodic' in params:
            self.periodic  = params['periodic']
        
        self.smoothing = params['smoothing']

        self._loaded = False
        self._t = None
        self._x = None
        self._spl = None
    
    def load(self, t, x):
        self._loaded = True
        self._t = t
        self._x = x
        # returns (knots, coefficients, order)
        self._spl = sci.interpolate.splrep(self._t, self._x, k=self.order, 
            s=self.smoothing, per=self.periodic)

    def unload(self):
        self._loaded = False
        self._t = None
        self._x = None
        self._spl = None
  
    def compute(self, t, x, i):
        self.load(t, x)
        return sci.interpolate.splev(self._t[i], self._spl, der=1)
        
    def compute_for(self, t, x, indices):
        self.load(t, x)
        for i in indices:
            yield sci.interpolate.splev(self._t[i], self._spl, der=1)
            
    def compute_global(self, t, x):
        self.load(t, x)
        return lambda t0: sci.interpolate.splev(t0, self._spl, der=1)
        

# ---------------

def multinomial_powers(n, k):
    '''
    Returns all combinations of powers of the expansion (x_1+x_2+...+x_k)^n.
    The motivation for the algorithm is to use dots and bars: 
    e.g.    For (x1+x2+x3)^3, count n=3 dots and k-1=2 bars. 
            ..|.| = [x1^2, x2^1, x3^0]

    Note: Add 1 to k to include a constant term, (1+x+y+z)^n, to get all
    groups of powers less than or equal to n (just ignore elem[0])
    
    Emphasis is on preserving yield behavior of combinatorial iterator.
        
    Arguments:
        n: the order of the multinomial_powers
        k: the number of variables {x_i}
    '''
    for elem in combinations(np.arange(n+k-1), k - 1):
        elem = np.array([-1] + list(elem) + [n+k-1])
        yield elem[1:] - elem[:-1] - 1

# ---------------

class SINDy:
    '''SINDy identifies an (ideally sparse) vector S that satisfies the
    equation X' = Theta(X)*S.
    Arguments:
        params['library']: a list of functions making up the columns of Theta(x)
        params['derivative']: a Derivative object to compute x_dot
        params['model]: For now, the model is from sklearn and requires a \
                        fit and returns a result with a coeff_ attribute
    '''
    def __init__(self, params):
        self.library = params['library']
        self.derivative = params['derivative']
        self.model = params['model']

        # Results
        self._loaded = False
        self.t = None
        self.x = None
        self.ThX = None
        self.x_dot = None
        self._valid_data = None
        self.soln = None
        self._solver = None

        # Dimensions
        self._n = 0
        self._d = 0
        self._l = 0

    def reset(self):
        self._loaded = False
        self.t = None
        self.x = None
        self.ThX = None
        self.x_dot = None
        self._valid_data = None
        self.soln = None
        self._solver = None

        # Dimensions
        self._n = 0
        self._d = 0
        self._l = 0

    def _load(self, t, x):
        '''
            Loads equation. Does not solve anything.
        '''
        arg_t = np.argsort(t)
        self.t = t[arg_t]
        self.x = x[arg_t]

        # Reshape after sort
        self.x = self.x.reshape(-1, 1) if self.x.ndim == 1 else self.x
        self._n,self._d = self.x.shape

        # -- LHS (n'd)
        # Compute derivative
        self.x_dot = np.empty((self._n,self._d))
        for i in range(self._d):
            self.x_dot[:,i] = np.array(list(self.derivative.compute_for(self.t, self.x[:,i], range(self._n))))

        # Limit derivates to universally valid values
        self._valid_data = np.ones(self._n)
        for i in range(self._d):
            self._valid_data *= np.isfinite(self.x_dot[:,i])
        self._valid_data = self._valid_data.astype(bool)

        # -- RHS
        # Compute Library(x)
        self.ThX = np.array([f(self.x) for f in self.library]).T
        _,self._l = self.ThX.shape

        self._loaded = True


    def sequential_thresholding(self, t, x, threshold):
        ''' 
            Repeatedly runs SINDy's identify procedure, suppressing
            terms that fall below the provided threshold. Returns a
            solution of shape (l'd).

            Note: For the original ST-algorithm, initialize SINDy
            with a least squares model (no regularization).

            Arguments:
                t: times (n)
                x: data (n'd)
                threshold: Zero any library terms with coefficients less\
                           than the threshold. Threshold can be universal\
                           or vary across data dimensions (size 0 or d)
        '''
        if self._loaded:
            self.reset()

        self._load(t,x)
        
        # Need a model for each dimension of the data
        models = [clone(self.model) for i in range(self._d)]
        active_terms = np.ones((self._l,self._d)).astype(bool)
        self.soln = np.zeros((self._l,self._d))
        # Allow thresholding to be separate along each dimension
        threshold = np.array([threshold]*self._d) if np.ndim(threshold) == 0 else threshold

        for col in range(self._d):            
            # Can remove at worst one library term per iteration
            for count in range(len(self.library)):
                # Rezero (replace only active terms)
                self.soln[:, col] = 0

                # All terms were removed
                if not np.any(active_terms[:,col]):
                    break

                # Fit model along active terms for valid data (both must be true)
                index = np.outer(self._valid_data, active_terms[:,col])
                models[col].fit(self.ThX[index].reshape(np.sum(self._valid_data), np.sum(active_terms[:,col])),
                                    self.x_dot[self._valid_data, col])
                self.soln[active_terms[:,col], col] = models[col].coef_

                # Next active terms must exceed the threshold
                new_active = np.abs(self.soln[:,col]) > threshold[col]

                # No change in active terms
                if np.all(new_active == active_terms[:,col]):
                    break

                active_terms[:,col] = new_active

        return self.soln
    
    def identify(self, t, x):
        ''' 
            Solves the SINDy regression problem using the sklearn model provided.
            Returns a solution of shape (l'd).

            Arguments:
                t: times (n)
                x: data (n'd)
        '''
        if self._loaded:
            self.reset()

        self._load(t, x)

        # Fit model
        self.model.fit(self.ThX[self._valid_data], self.x_dot[self._valid_data])
        self.soln = self.model.coef_.reshape(self._l,self._d)
        
        return self.soln

    def integrate(self, x0, t_initial, t_final, t_pts):
        if not np.any(self.soln):
            return [np.array([])]*2 # blank time, prediction

        # Equation is 1'l x l'd = 1'd, flatten to d
        rhs = lambda t, x: ((np.array([f(x.reshape(1,-1)) for f in self.library]).T)@(self.soln)).flatten()
        self._solver = sci.integrate.solve_ivp(rhs, [t_initial,t_final], y0=x0, t_eval=np.linspace(t_initial,t_final,t_pts))

        # Save the solver for debugging
        return self._solver.t, self._solver.y.T