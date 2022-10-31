from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def F(x0):
    x=x0[0]
    y=x0[1]
    #return array([x**3 -3*x*y*y-1, 3*x*x*y - y**3]) #Task 4 
    #return array([x**3 -3*x*y*y-2*x -2, 3*x*x*y -y**3 -2*y]) #Task 8-1
    return array([x**8 -28*x**6*y*y +70*x**4*y**4 + 15*x**4 -28*x**2*y**6 -90*x**2*y**2 + y**8 + 15*y**4 -16, 8*x**7*y -56*x**5*y**3 + 56*x**3*y**5 +60*x**3*y -8*x*y**7 -60*x*y**3]) #Task 8-2

#def JF(x0):
#    x=x0[0]
#    y=x0[1]
#    return array([[3*x**2-3*y*y, -6*x*y],[6*x*y, 3*x*x-3*y*y]])# For Task 4

      
class fractal2D:
    def __init__(self, f, fp = None, maxIt = 100, tol = 1.e-10, step = 0.001):
        self.f = f
        self.fp = fp
        self.maxIt = maxIt
        self.tol = tol
        self.step = step
        self.zeroes = []
        self.Iterations=[]
        
    def __repr__(self):
        return f"[{self.f}, {self.tol}, self.zeroes, self.zeroes.index, self.Iterations]"
    
    def simplifiednewton(self, x0):
            Jacobinv = np.linalg.pinv(self.fp(x0))
            x0 = array(x0)
            for n in range (1, self.maxIt):
                if np.all(np.isclose(self.f(x0),array([0,0]), atol = self.tol)): #norm istället?
                    self.Iterations.append(n)
                    return x0.tolist()
                else:
                    x0=x0-(Jacobinv @ self.f(x0))
            else:
                self.Iterations.append(n)

    def newton(self, x0):               
        x0 = array(x0)
        #print(x0) #för att kunna se ingångsvärde på x,y
        for n in range (1, self.maxIt):
            if np.all(np.isclose(self.f(x0),array([0,0]), atol = self.tol)): #norm istället?
                self.Iterations.append(n)
                return x0.tolist()
            else:
                Jacobinv = np.linalg.pinv(self.fp(x0))
                x0=x0-(Jacobinv @ self.f(x0))
                #print(x0, n)  #för att kunna se progressionen på x,y
        else:
            self.Iterations.append(n)
                          
    def newton2(self,x0):   #plocka bort en av if satserna...  
        x0 = array(x0)        
        for n in range (1, self.maxIt):
            if np.all(np.isclose(self.f(x0),array([0,0]), atol = self.tol)): 
                self.Iterations.append(n)
                return x0.tolist()
            else:
                x0 = x0 - np.linalg.solve(self.numericalderivative(x0), self.f(x0))
                #print(x0,n)
            if np.linalg.norm(self.f(x0)) < self.tol:
                self.Iterations.append(n)
                return x0.tolist()
        else:
            self.Iterations.append(n)
            
    def numericalderivative(self,x0):
        #x0 behandlas som en vektor och delas inte upp.
        h = self.step
        f1x_ad_h = self.f(x0 + array([h,0]))[0]
        f1x_sub_h = self.f(x0 + array([-h,0]))[0]
        f1y_ad_h = self.f(x0 + array([0,h]))[0]
        f1y_sub_h = self.f(x0 + array([0,-h]))[0]
   
        f1_derivative_x = (f1x_ad_h-f1x_sub_h)/(2*h)
        f1_derivative_y = (f1y_ad_h-f1y_sub_h)/(2*h)   

        f2x_ad_h = self.f(x0 + array([h,0]))[1]
        f2x_sub_h = self.f(x0 + array([-h,0]))[1]
        f2y_ad_h = self.f(x0 + array([0,h]))[1]
        f2y_sub_h = self.f(x0 + array([0,-h]))[1]
        f2_derivative_x = (f2x_ad_h-f2x_sub_h)/(2*h)
        f2_derivative_y = (f2y_ad_h-f2y_sub_h)/(2*h)
        return ([[f1_derivative_x,f1_derivative_y],[f2_derivative_x,f2_derivative_y]])
    
    def roots(self,x0 ,val):
        if not self.fp:
            z=self.newton2(x0)
        elif val:
            z=self.newton(x0)
        else:
            z=self.simplifiednewton(x0)
            
        if z is None:
            return 0
        elif not self.zeroes:
            self.zeroes.append(z)
            return 1
        else:
            for i in self.zeroes:
                if np.all(np.isclose(i,z, atol = self.tol)): # förenkla???
                    return self.zeroes.index(i)+1
            else:
                self.zeroes.append(z)
                return len(self.zeroes)

    def plot(self,N,a,b,c,d,val):
        self.Iterations.clear()
        #u = fractal2D(F,JF)
        u = fractal2D(F)
        s = linspace(a,b,N)
        t = linspace(c,d,N)
        A = np.zeros((N,N))
        with np.nditer(A, flags=['multi_index'], op_flags=['writeonly']) as it:
            for x in it:
                x[...] = u.roots([s[it.multi_index[1]],t[it.multi_index[0]]],val)

        #print(A) #För att kunna se de olika indexerade "nollställena"
        X,Y = np.meshgrid(s,t)
        #print(X)
        plt.pcolor(X,Y,A, shading="auto")
        i=len(u.zeroes)
        for x in range(i):
            plt.plot(u.zeroes[x][0],u.zeroes[x][1],'ro')
            h=plt.text((u.zeroes[x][0])+0.1, (u.zeroes[x][1]), f"{round(u.zeroes[x][0],2)}, {round(u.zeroes[x][1],2)}",Color = "black", fontsize=7)

        plt.xlabel('x1')
        plt.ylabel('x2')
        #plt.title("Task 4, $F(x_1, x_2) = x_1^3 -3x_1x_2^2-1, 3x_1^2x_2 - x_2^3$")
        #plt.savefig('4-X1 and X2 with solutions-2000*2000-500dpi.png', dpi=500)
        
        #plt.title("Task 8_1, $F(x_1, x_2) = x_1^3 -3x_1x_2^2-2x_1 -2, 3x_1^2x_2 - x_2^3 -2x_2$")
        #plt.savefig('8_1_2-X1 and X2 with solutions-2000*2000-500dpi.png', dpi=500)
        
        plt.title("8_2, \n$F(x_1, x_2) = x_1^8 - 28x_1^6x_2^2 + 70x_1^4x_2^4 + 15x_1^4 - 28x_1^2x_2^6 - 90x_1^2x_2^2 + x_2^8 + 15x_2^4 -16$, \n        $8x_1^7x_2 - 56x_1^5x_2^3 + 56x_1^3x_2^5 + 60x_1^3x_2 -8x_1x_2^7 - 60x_1x_2^3$")
        #plt.savefig('8_2-X1 and X2 with solutions-2000*2000-500dpi.png', dpi=500)
        
        plt.show()

    def iterationsplot(self,N,a,b,c,d,val):# ingen skillnad ännu mot funktionen plot
        self.Iterations.clear()
        #u = fractal2D(F,JF)
        u = fractal2D(F)
        s = linspace(a,b,N)
        t = linspace(c,d,N)
        Z = np.zeros((N,N))
        X,Y = np.meshgrid(s,t)
        
        with np.nditer(Z, flags=['multi_index'], op_flags=['writeonly']) as it:
            for x in it:
                x[...] = u.roots([s[it.multi_index[1]],t[it.multi_index[0]]],val)
        #print(u.Iterations)
        Z = [u.Iterations[i * N:(i + 1) * N] for i in range((len(u.Iterations) + N - 1) // N )]
        #print(Z)
        Z = array(Z)
        #print(Z)        
        cm = plt.cm.get_cmap('hot')
        plt.scatter(X, Y, c=Z, cmap=cm)
        plt.colorbar()
        #x = [-0.5,-0.5,1]
        #y = [-sqrt(3)/2,sqrt(3)/2,0] 
        #plt.plot(x, y, 'ro')
        i=len(u.zeroes)
        for x in range(i):
            plt.plot(u.zeroes[x][0],u.zeroes[x][1],'bo')
            h=plt.text((u.zeroes[x][0])+0.1, (u.zeroes[x][1]), f" {round(u.zeroes[x][0],2)}, {round(u.zeroes[x][1],2)}" ,Color = "w", fontsize=7)

        plt.xlabel('x1')
        plt.ylabel('x2')
        #plt.title("Task 4, $F(x_1, x_2) = x_1^3 -3x_1x_2^2-1, 3x_1^2x_2 - x_2^3$, Iterations")
        #plt.savefig('4-Iterations_2000*2000-500dpi.png', dpi=500)
        
        #plt.title("Task 8_1, $F(x_1, x_2) = x_1^3 -3x_1x_2^2-2x_1 -2, 3x_1^2x_2 - x_2^3 -2x_2$, Iterations")
        #plt.savefig('8_1-Iterations_2000*2000-500dpi.png', dpi=500)
        
        plt.title("8_2 Iterations \n $F(x_1, x_2) = x_1^8 - 28x_1^6x_2^2 + 70x_1^4x_2^4 + 15x_1^4 - 28x_1^2x_2^6 - 90x_1^2x_2^2 + x_2^8 + 15x_2^4 -16$, \n                   $8x_1^7x_2 - 56x_1^5x_2^3 + 56x_1^3x_2^5 + 60x_1^3x_2 -8x_1x_2^7 - 60x_1x_2^3$")
        #plt.savefig('8_2-Iterations_2000*2000-500dpi.png', dpi=500)
        plt.show()
        

"""my_list = [1, 2, 3, 4, 5,
              6, 7, 8, 9]
  
# How many elements each
# list should have
n = 4 
  
# using list comprehension
Z = [Iterations[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n )] 
print (final)

        X,Y = np.meshgrid(s,t)
        
        h= plt.contour(s,t,Z)
"""        

def runboth(N, a, b, c, d, val):
    #fractal2D(F, JF).plot(N,a,b,c,d, val)
    #fractal2D(F,JF).iterationsplot(N,a,b,c,d, val)
    fractal2D(F).plot(N,a,b,c,d, val)
    fractal2D(F).iterationsplot(N,a,b,c,d, val)
    
#runboth(2000,-2.5,2.5,-2.5,2.5, True) # For Task 4
runboth(100, 3,-3,3,-3, True) # For Task 8_1
