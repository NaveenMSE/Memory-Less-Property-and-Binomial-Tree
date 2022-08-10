#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy


# In[3]:


# Probability of an individual success
p=0.3

# Sample size
n=100000

#Generating random variables

x=np.random.geometric(p,n)
x_l=np.array((list((np.random.geometric(p,n)))))
print(x_l)


# In[4]:


#Extracting the unique no of trials value
b=np.unique(x_l)
print(b)


# In[5]:


#Generating the random variable with domain and co-domain

def geo(x_l,n,b):
    def count(l,t):
        a=0
        for i in range(len(l)):
            if l[i]==t:
                a=a+1
            else:
                a=a+0
        return a
    
    rv= np.empty((len(b),2),float)
    for i in range(len(b)):
        p=float(count(x_l,b[i])/n)
        rv[i][0]=int(b[i])
        rv[i][1]=p
    df = pd.DataFrame(rv,columns = ['No_of_trials(X)', 'probability'])
    return df
    


# In[6]:


rv=geo(x_l,n,b)
rv 


# In[ ]:





# In[7]:


#Visualisation
plt.bar(rv['No_of_trials(X)'],rv['probability'])
plt.title('Geometric Distribution', fontsize=14)
plt.xlabel('No of trials(X)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.show()


# In[8]:


# Probability(x>k)
def probgreater(l,t,n):
    a=0
    for i in range(len(l)):
        if l[i]>t:
            a=a+1
        else:
            a=a+0
    return a/n


# P[X>k+a|x>k]=P[x>a]
# 
# first_p=x>k+a
# 
# cond_p=x>k

# In[9]:


k=12
a=8
first_p=probgreater(x_l,k+a,n)
cond_p=probgreater(x_l,k,n)

LHS=first_p/cond_p
print(LHS)


# In[10]:


RHS=probgreater(x_l,a,n)
print(RHS)


# p(x>k)=q^k

# In[11]:


(0.7)**20/0.7**12


# In[12]:


(0.7)**8


# In[ ]:





# Horizontal shift

# In[13]:


def Ggreater(x_l,b,n,k):
    def probgreater(l,t,n):
        a=0
        for i in range(len(l)):
            if l[i]>t:
                a=a+1
            else:
                a=a+0
        return a/n
    def count(l,t):
        a=0
        for i in range(len(l)):
            if l[i]==t:
                a=a+1
            else:
                a=a+0
        return a
    bg=b[b>k]
    cp=float(probgreater(x_l,k,n))
    rv= np.empty((len(bg),2),float)
    for i in range(len(bg)):
        p=count(x_l,bg[i])/n
        rv[i][0]=int(bg[i])
        rv[i][1]=p/cp
    df = pd.DataFrame(rv,columns = ['No_of_trials(X)', 'probability(w=x/w>k)'])
    return df


# In[14]:


k=0
a=Ggreater(x_l,b,n,k)
k=5
b=Ggreater(x_l,b,n,k)


# In[15]:



#Visualisation
plt.bar(a['No_of_trials(X)'],a['probability(w=x/w>k)'])
plt.title('Geometric Distribution', fontsize=14)
plt.xlabel('No of trials(X)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.show()

plt.bar(b['No_of_trials(X)'],b['probability(w=x/w>k)'])
plt.title('Geometric Distribution', fontsize=14)
plt.xlabel('No of trials(X)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.show()


# Binomial distribution

# In[16]:


p=0.5
n=10
x_l=np.array((list((np.random.binomial(n,p,100000)))))
x_l


# In[17]:


b=np.unique(x_l)


# In[ ]:





# In[18]:


def bino(x_l,n,b):
    def count(l,t):
        a=0
        for i in range(len(l)):
            if l[i]==t:
                a=a+1
            else:
                a=a+0
        return a
    
    rv= np.empty((len(b),2),float)
    for i in range(len(b)):
        p=float(count(x_l,b[i])/n)
        rv[i][0]=int(b[i])
        rv[i][1]=p
    df = pd.DataFrame(rv,columns = ['No_of_success(x)', 'probability'])
    return df


# In[19]:


rv=bino(x_l,100000,b)
rv


# In[20]:


plt.bar(rv['No_of_success(x)'],rv['probability'])
plt.title('Binomial', fontsize=14)
plt.xlabel('No_of_success(x)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.show()


# Probability with the given outcome

# In[ ]:





# In[21]:


def bp(n,p,s,outcome,x_l,b):
    x_l=np.array((list((np.random.binomial(n,p,s)))))
    b=np.unique(x_l)
    
    def bino(x_l,n,b):
        def count(l,t):
            a=0
            for i in range(len(l)):
                if l[i]==t:
                    a=a+1
                else:
                    a=a+0
            return a
        rv= np.empty((len(b),2),float)
        for i in range(len(b)):
            p=float(count(x_l,b[i])/n)
            rv[i][0]=int(b[i])
            rv[i][1]=p
        df = pd.DataFrame(rv,columns = ['No_of_success(x)', 'probability'])
        return df
    
    rv=bino(x_l,100000,b)
    
    def c(outcome):
        success=0
        for i in outcome:
            if i==1:
                success=success+1
        return success
    success=c(outcome)
    a=rv[rv['No_of_success(x)']==success]
    return a['probability']

    

    


    
    
    
    


# In[22]:


p=0.4
n=10
s=100000
x_l=np.array((list((np.random.binomial(n,p,100000)))))
x_l
rv=bino(x_l,s,b)
rv


# In[23]:


plt.bar(rv['No_of_success(x)'],rv['probability'])
plt.title('Binomial', fontsize=14)
plt.xlabel('No_of_success(x)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.show()


# In[24]:


outcome=[1,1,1,1,1,0,0,0,1,1]
bp(n,p,s,outcome,x_l,b)


# No of paths

# In[25]:


def path(n,x):
    a=[[1],[0]]
    h=copy.deepcopy(a)
    t=copy.deepcopy(a)
    for i in range(n-1):
        for j in range(len(h)):
            h[j].append(a[0][0])
            t[j].append(a[1][0])
        tot=h+t
        h=copy.deepcopy(tot)
        t=copy.deepcopy(tot)
    def count(l,t):      
        a=0
        for i in range(len(l)):
            if l[i]==t:
                a=a+1
            else:
                a=a+0
        return a
    p=0
    for i in range(len(t)):
        c=count(t[i],1)
        if c==x:
            p=p+1
    return p
    


# In[27]:


path(3,1)


# In[34]:





# In[35]:


path(3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




