import sympy
import numpy as np
import matplotlib.pyplot as plt

def plotlensdistortion(eqX,eqY):
    x,y=sympy.symbols('x y')
    spacing=10
    x_i=np.linspace(-1,1,spacing)
    y_i = np.linspace(-1, 1, spacing)
    fig, ax = plt.subplots()


    for i in x_i :
        for j in y_i:
            xout = eqX.subs([(x,i),(y,j)])
            yout = eqY.subs([(x, i), (y, j)])
            m=xout-i/yout-j
            plt.plot(i,j,'o',color='green')

            ax.quiver(i,j,(float)(xout-i),(float)(yout-j),scale=1)

    plt.show()


k_1, k_2, p_1, p_2, r, x, y = sympy.symbols('k_1 k_2 p_1 p_2 r x y')

eqX = x * (1 + k_1 * r ** 2 + k_2 * r ** 4) + 2 * p_1 * x * y + p_2 * (r ** 2 + 2 * x ** 2)
eqY = y * (1 + k_1 * r ** 2 + k_2 * r ** 4) + 2 * p_2 * x * y + p_1 * (r ** 2 + 2 * y ** 2)
## paratemeters=[k_1, k_2, p_1, p_2]
parameters=[0.1,0,0,0]
eqX=eqX.subs([(r,(x ** 2 + y ** 2)**(1/2)),(k_1,parameters[0]),(k_2,parameters[1]),(p_1,parameters[2]),(p_2,parameters[3])])
eqY=eqY.subs([(r,(x ** 2 + y ** 2)**(1/2)),(k_1,parameters[0]),(k_2,parameters[1]),(p_1,parameters[2]),(p_2,parameters[3])])

plotlensdistortion(eqX,eqY)

result=sympy.solve([eqX,eqY],[x,y])

print(result)

