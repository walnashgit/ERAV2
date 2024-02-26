# Back Propagation

## The contents of this folder explains the various steps involved in the calculation for the simulation of a backpropagation example.

This folder contains a file named **ERA-S6-BackPropagation.xlsx** which contains the detailed calulation, data and simulated backpropagation along with a graph showing the loss value through the backpropagation simulation using different values for learning rate.

The below screenshot is from the simulation excel.
<img width="1488" alt="Screenshot 2024-02-26 at 8 16 40 PM" src="https://github.com/walnashgit/ERAV2/assets/73463300/1667bb23-f215-454b-be76-dde4dd72c057">

## Calculation
Explained here are the various major steps and calculation used in the simulation. You can find the same in the excel sheet as well.

<img width="734" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/224ad11d-b048-4ed3-9797-c78fc409895b">

Using the back propagation example from above image:

w1, w2, w3, w4, w5, w5, w6, w7 and w8 are the weights of this neural network as shown.

h1 and h2 are the neurons at first layer.

a_h1 and a_h2 is the result of activation (sigmoid) on these neurons.

similarly o1 and o2 are neurons in the second layer; a_o1 and a_02 is the result of activation on these neurons.

t1 and t2 is the target.

E1 and E2 are the errors from the output of last two neurons.

E_total is the total loss.

### The equations for all the components of this example can be written as follows

h1 = w1*i1 + w2*i2

h2 = w3*i1 + w4*i2

a_h1 = sig(h1) = 1/(1+ e^-h1)

a_h2 = sig(h2) = 1/(1+ e^-h2)

o1 = a_h1*w5 + a_h2*w6

o2 = a_h1*w7 + a_h2*w8

a_o1 = sig(o1)

a_o2 = sig(o2)

E1 = (t1-a_o1)^2/2

E2 = (t2-a_o2)^2/2

E_total = E1 + E2

### To minimise the loss we need to calculate gradient descent of the loss function. We need to calculate the gradient of the total loss wrt to each weights.

## Gradient descent calculation using partial derivative:

    dE_total/dw5 = d(E1+E2)/dw5

dE_total/dw5 = d(E1)/dw5 _>> Since E2 does not depend on w5_

dE1/dw5 = `dE1/da_o1` * `da_o1/do1` * `do1/dw5`  _>> Using chain rule for chaining derivatives_

**Calculating individual derivatives from above**

`dE1/da_o1` = d(1/2(t1-a_o1)^2)/da_o1 = a_o1 - t1 _>> eg: d(1/2(x-y)^2)/dy = y-x_

`da_o1/do1` = dsig(o1)do1 = sig(o1)(1-sig(o1)) = a_o1(1-a_o1)

and 

`do1/dw5` = d(a_h1*w5 + a_h2*w6)/dw5 = a_h1 _>> since second part is constant, its derivatice is 0_

therefore

    dE_total/dw5  = (a_o1 - t1) * a_o1 * (1-a_o1) * a_h1
    
Similarly we calculate the below derivatives:
    
    dE_total/dw6 =  (a_o1 - t1) * a_o1 * (1-a_o1) * a_h2	
    
    dE_total/dw7 = (a_o2 - t2) * a_o2 * (1-a_o2) * a_h1
    
    dE_total/dw8 = (a_o2 - t2) * a_o2 * (1-a_o2) * a_h2			

### Calulation of dE_total/dw1

    dE_total/dw1 = dE_total/a_h1 * da_h1/dh1 * dh1/dw1

**Again, calulating intividual derivatives:**

`dE_total/da_h1` = d(E1+E2)/da_h1 = `dE1/da_h1` + dE2/da_h1

**focusing on `dE1/da_h1`**

`dE1/da_h1` = d(½(t1-a_o1)^2)/da_h1 = (t1 - a_o1) * d(t1 - a_o1)/da_h1 

dE1/da_h1 = (t1 - a_o1)(dt1/da_h1 - da_o1/da_h1)

    dE1/da_h1 = (t1 - a_o1)(0 - da_o1/d_h1) = (a_o1 - t1) * da_o1/da_h1

**focusing on `da_o1/da_h1`**

`da_o1/da_h1` = d(sig(o1)/da_h1 = d(sig(o1)/d(o1) * do1/da_h1 _>> chaining at last step_

da_o1/da_h1 = a_o1(1-a_o1) * w5

so, 

    dE1/da_h1  = (a_o1 - t1) * a_o1 * (1 - a_o1) * w5

similarly

    dE2/da_h1 = (a_o2 - t2) * a_o2 * (1 - a_o2) * w7

therefore we have finally

    dE_total/da_h1 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7

    da_h1/dh1 = d(sig(h1)/dh1 = a_h1 * (1 - a_h1)

    dh1/dw1 = i1

Hence finally:

    dE_total/dw1 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1

similarly

    dE_total/dw2 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
    
    dE_total/dw3 = ((a_o2 - t2) * a_o2 * (1 - a_o2) * w8 + (a_o1 - t1) * a_o1 * (1 - a_o1) * w6) * a_h2 * (1 - a_h2) * i1
    
    dE_total/dw4 = ((a_o2 - t2) * a_o2 * (1 - a_o2) * w8 + (a_o1 - t1) * a_o1 * (1 - a_o1) * w6) * a_h2 * (1 - a_h2) * i2


## Summary of all final derivatives: 

    dE_total/dw1 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1

    dE_total/dw2 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
    
    dE_total/dw3 = ((a_o2 - t2) * a_o2 * (1 - a_o2) * w8 + (a_o1 - t1) * a_o1 * (1 - a_o1) * w6) * a_h2 * (1 - a_h2) * i1
    
    dE_total/dw4 = ((a_o2 - t2) * a_o2 * (1 - a_o2) * w8 + (a_o1 - t1) * a_o1 * (1 - a_o1) * w6) * a_h2 * (1 - a_h2) * i2

    dE_total/dw6 =  (a_o1 - t1) * a_o1 * (1-a_o1) * a_h2	
    
    dE_total/dw7 = (a_o2 - t2) * a_o2 * (1-a_o2) * a_h1
    
    dE_total/dw8 = (a_o2 - t2) * a_o2 * (1-a_o2) * a_h2			

### Using the starting values of weights as:

`w1 = 0.15, w2 = 0.2, w3 = 0.25, w4 = 0.3, w5 = 0.4, w6 = 0.45, w7 = 0.5, w8 = 0.55`

### Target value as:

`t1 =0.5, t2 = 0.5`

### Input value as
`i1 = 0.05, i2 = 0.1`

Calculate the gradient descent using the formula derived above and these starting value for first iteration. For the subsequent iteration, target and input values remain constant. Weights can be calculated as W<sub>new</sub> = W<sub>old</sub> - LR * dEtotal/dW<sub>old</sub>; where LR is `Learning Rate`.

Thus: W1<sub>new</sub> = W1<sub>old</sub> - LR * dEtotal/dW1<sub>old</sub>

Calculate Loss for different values of `LR = 0.1, 0.2, 0.5, 0.8, 1.0, 2.0`

Iterate the calculation/simulation in the sheet for around 80 times.

The graph in the sheet shows how the loss or error changes with different values of LR.

## Error grapshs wrt to LR

### When LR = 0.1

<img width="545" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/65606432-2090-467f-91ac-657d59ad1b0a">

### When LR = 0.2

<img width="543" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/549d4bcd-cde6-457e-bc3c-b43437989fed">

### When LR = 0.5

<img width="543" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/cf4d88f1-4012-4314-9f86-4a102635e8fa">

### When LR = 0.8

<img width="542" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/c3c9992b-81c7-4d56-b5e1-0626f8a284cb">

### When LR = 1

<img width="542" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/25fe24f9-df00-4cb9-96f9-b10bffee2539">

### When LR = 2

<img width="542" alt="image" src="https://github.com/walnashgit/ERAV2/assets/73463300/f41ec6e0-bd18-463f-8939-ceec5c51a37b">

Base on the graphs wrt to different LR, we can say that this neural network learns faster with higher LR.


