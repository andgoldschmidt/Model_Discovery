# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:33:55 2019

@author: agoldschmidt
"""
import warnings
import numpy as np
import abc

from itertools import combinations

import scipy as sci
from scipy import interpolate, integrate

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
        params['library']: the functions mapping in Theta(x)
        params['derivative']: a Derivative object to compute x_dot
        params['model]: For now, the model is from sklearn and requires a \
                        fit and returns a result with a coeff_ attribute
    '''
    def __init__(self, params):
        self.library = params['library']
        self.derivative = params['derivative']
        self.model = params['model']

        # Results
        self.loaded = False
        self.t = None
        self.x = None
        self.ThX = None
        self.x_dot = None
        self.res = None
        self.t_pred = None
        self.x_pred = None
        self.dim_pred = None

    def reset(self):
        self.loaded = False
        self.t = None
        self.x = None
        self.ThX = None
        self.x_dot = None
        self.res = None
        self.t_pred = None
        self.x_pred = None
        self.dim_pred = None
        
    def identify(self, t, x):
        if self.loaded:
            self.reset()
        
        arg_t = np.argsort(t)
        self.t = t[arg_t]
        self.x = x[arg_t]

        # Reshape after sort
        self.x = x.reshape(-1, 1) if x.ndim == 1 else x
        n,d = self.x.shape

        # Compute Library(x)
        self.ThX = np.array([f(self.x) for f in self.library]).T

        # Compute equation along each dimension of x
        self.x_dot = np.empty((n, d))
        self.res = []
        for i in range(d):
            # Compute derivative
            self.x_dot[:,i] = np.array(list(self.derivative.compute_for(self.t, self.x[:,i], range(n))))

            # Limit derivates to valid values
            allowed =  np.isfinite(self.x_dot[:,i])
    
            # Fit model (ignore fit warnings TODO: make more honest)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.res.append(self.model.fit(self.ThX[allowed], self.x_dot[allowed, i]))
        
        self.loaded = True 
        return self.res[0].coef_ if d == 1 else np.array([r.coef_ for r in self.res])

    def integrate(self, t0=None, x0=None, t_step=np.inf):
        if not self.loaded:
            return np.array([])

        n,d = self.x.shape

        # Create the right hand side from the coefficients and library
        # Note: x will have dimensions that match scipy rk45 specs
        rhs = lambda t, x: (np.array([r.coef_ for r in self.res]))\
                           @(np.array([f(x.reshape(-1, d)) for f in self.library]))

        # Load initial state
        x0 = x0 if x0 else self.x[0, :]    
        x0 = x0 if np.ndim(x0) > 0 else [x0]
        t0 = t0 if t0 else self.t[0]

        # Integrate equation (vectorized because functions are f(t,y) for y.shape = dim,pts)
        solver = sci.integrate.RK45(rhs, t0, x0, self.t.max(), max_step=t_step, vectorized=True)
        self.t_pred = []
        self.x_pred = []
        while solver.status == 'running':
           solver.step()
           self.t_pred.append(solver.t)  
           self.x_pred.append(solver.y)
        self.t_pred = np.array(self.t_pred)
        self.x_pred = np.array(self.x_pred)

        return [self.t_pred, self.x_pred]