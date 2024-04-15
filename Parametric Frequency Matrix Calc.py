# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:02:34 2023

@author: grace
"""
# Grace O. Capek, Garand Group, University of Wisconsin-Madison, 2023
#%% This program calculates the transmission matrix for a linear quadrupole using
#   matrix solutions to the Mathieu-Hill equations (see Konenkov et al 2002).
#   From the trace of the transmission matrix, we get the Mathieu stability
#   parameter Beta and finally, the first order parametric resonant frequencies for X and Y.
import numpy as np
import matplotlib.pyplot as plt
import functools as ft

#%% Define the functions to calculate f values and the first order (K1) parametric/quadrupolar frequencies

# create a function to calculate the matrix when f > 0
def transM_pos(f,t):
    m11 = np.cos(t*np.sqrt(f))
    m12 = (1/np.sqrt(f))*np.sin(t*np.sqrt(f))
    m21 = -np.sqrt(f)*np.sin(t*np.sqrt(f))
    m22 = np.cos(t*np.sqrt(f))
    M = [[m11,m12],[m21,m22]]
    return M
# create a function to calculate the matrix when f < 0
def transM_neg(f,t):
    m11 = np.cosh(t*np.sqrt(-f))
    m12 = (1/np.sqrt(-f))*np.sinh(t*np.sqrt(-f))
    m21 = np.sqrt(-f)*np.sinh(t*np.sqrt(-f))
    m22 = np.cosh(t*np.sqrt(-f))
    M = [[m11,m12],[m21,m22]]
    return M

# create a function to calculate Beta and return the K=1 parametric frequency for a given
# mass range, base frequency, and duty cycle combination
def K1freq_calc(mass_array,baseF_kHz,t1,t3):
    # Define the constants and calculated variables
    t2 = np.absolute(1-t1-t3)               # calculate t2, the time that the quadrupolar potential is zero
    e_C = 1.602176634e-19                   # charge of one electron in Coulombs
    z = 1                                   # in our case, usually only one charge                
    U = (49+-55)-(46+-53)                   # DC potential in Volts
    #U = 1
    V = 50                                  # zero-to-peak voltage of square wave in Volts
    r0 = 0.005                              # trap radius in meters
    baseF_rad = baseF_kHz*1000*2*np.pi      # convert base frequency to radians 
    Beta_array = []
    
    for m_Da in mass_array:
        # Convert mass in Da to kg
        m_kg = m_Da*1.66054e-27
        # Calculate Mathieu a and q stability parameters
        a = (8*e_C*z*U)/(m_kg*(baseF_rad**2)*(r0**2))
        q = (-4*e_C*z*V)/(m_kg*(baseF_rad**2)*(r0**2))
        # Calculate the input to the matrices, tau and f
        tau1 = t1*np.pi
        tau2 = t2*np.pi
        tau3 = t3*np.pi
        f1 = a + 2*q
        f2 = a
        f3 = a - 2*q
        # f1 = -a - 2*q
        # f2 = -a
        # f3 = -a + 2*q
        # Calculate the three transmission matrices for t1, t2, and t3
        if f1 > 0:
            M1 = transM_pos(f1,tau1)
        else:
            M1 = transM_neg(f1,tau1)
        if f2 > 0:
            M2 = transM_pos(f2,tau2)
        else:
            M2 = transM_neg(f2,tau2)
        if f3 > 0:
            M3 = transM_pos(f3,tau3)
        else:
            M3 = transM_neg(f3,tau3)  
        # Multiply the transmission matrices by each other to get the total
        M_total = ft.reduce(np.dot,[M1,M2,M3])
        M_trace = np.trace(M_total)
        # Calculate Beta from the trace of the total transmission matrix
        Beta = np.arccos(np.complex(M_trace/2))/np.pi
        Beta_real = np.real(Beta)
        #Beta = np.arccos(np.absolute(M_trace)/2)/np.pi
        Beta_array.append(Beta_real)
    
    # Calculate the first order quadrupolar resonant frequency from Beta
    K1_freq_rad = [x*baseF_rad for x in Beta_array]
    K1_freq_kHz = [y/(2*np.pi*1000) for y in K1_freq_rad]
    
    return K1_freq_kHz

#%% Plot the results of the calculation with respect to ion m/z for different base frequencies
#   Note that these here are Y dimension frequencies
mass_array = np.linspace(125,250,500)     # Define the mass range in Da

#Plot the predicted K1 frequencies vs. m/z using the K1freq_calc function
#Inputs are the mass array (in Da), base frequency (kHz), t1, and t3
fig, ax = plt.subplots()#
line1, = ax.plot(mass_array,K1freq_calc(mass_array,500,0.52,0.45),color="#FF800E",label=" ")
line2, = ax.plot(mass_array,K1freq_calc(mass_array,525,0.52,0.45),color="#C85200",label=" ")
line3, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.52,0.45),color="#5F9ED1",label=" ")
line4, = ax.plot(mass_array,K1freq_calc(mass_array,575,0.52,0.45),color="#006BA4",label=" ")
line5, = ax.plot(mass_array,K1freq_calc(mass_array,600,0.52,0.45),color="#595959",label=" ")
legend1 = ax.legend(handles=[line1,line2,line3,line4,line5],bbox_to_anchor=(0.65, 0.6))
ax.add_artist(legend1)
plt.xlabel("m/z", fontsize = 13)
plt.ylabel("K=1 Parametric Frequency (kHz)", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

# Plot the experimental values
m_exp = [133,156,161,166,182,190,232]
w1_exp = [291,225,214,204,178,168,126]  # 48/55 Y base 500 kHz
w2_exp = [260,203,196,187,164,155,116]  # 48/55 Y base 525 kHz
w3_exp = [234,183,175,170,146,135,104]  # 48/55 Y base 550 kHz 
w4_exp = [220,172,163,154,139,124,97]   # 48/55 Y base 575 kHz 
w5_exp = [200,159,150,137,127,110,87]   # 48/55 Y base 600 kHz 

e = [6,8,8,9,10,10,13]        # approximate errors in frequency for each m/z
line6 = ax.errorbar(m_exp,w1_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#FF800E",label="500 kHz")
line7 = ax.errorbar(m_exp,w2_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#C85200",label = "525 kHz")
line8 = ax.errorbar(m_exp,w3_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#5F9ED1",label = "550 kHz")
line9 = ax.errorbar(m_exp,w4_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#006BA4",label = "575 kHz")
line10 = ax.errorbar(m_exp,w5_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#595959",label = "600 kHz")
legend2 = ax.legend(handles=[line6,line7,line8,line9,line10], bbox_to_anchor=(0.74, 0.6))
ax.add_artist(legend2)
plt.savefig("NEWfreq_trends.png",dpi=1200)
plt.show()


#%% Plot the calculated K1 frequencies vs. m/z for different duty cycles

# Plot the predicted first order resonant frequencies for the Y dimension
fig, ax = plt.subplots()
line1, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.50,0.44),color="#FF800E",label=" ")
line2, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.50,0.45),color="#C85200",label=" ")
line3, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.50,0.46),color="#5F9ED1",label=" ")
line4, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.50,0.47),color="#006BA4",label=" ")
line5, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.50,0.48),color="#595959",label=" ")
legend1 = ax.legend(handles=[line1,line2,line3,line4,line5],bbox_to_anchor=(0.69, 0.6))
ax.add_artist(legend1)
plt.xlabel("m/z", fontsize = 13)
plt.ylabel("K=1 Parametric Frequency (kHz)", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

# Plot the experimental first order resonant frequencies for the Y dimension
m_exp = [133,156,161,166,182,190,232]
w1_exp = [240,199,186,175,160,144,117]  # 50/56 Y base 550 kHz
w2_exp = [249,208,196,184,170,155,125]  # 50/55 Y base 550 kHz
w3_exp = [258,215,203,192,177,163,132]  # 50/54 Y base 550 kHz
w4_exp = [268,221,212,202,182,173,138]  # 50/53 Y base 550 kHz
w5_exp = [278,231,221,208,188,180,145]  # 50/52 Y base 550 kHz
e = [6,7,7,7,8,8,9.5]        # approximate errors in frequency for each m/z
line6 = ax.errorbar(m_exp,w1_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#FF800E",label = "50:44")
line7 = ax.errorbar(m_exp,w2_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#C85200",label = "50:45")
line8 = ax.errorbar(m_exp,w3_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#5F9ED1",label = "50:46")
line9 = ax.errorbar(m_exp,w4_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#006BA4",label = "50:47")
line10 = ax.errorbar(m_exp,w5_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#595959",label = "50:48")
legend2 = ax.legend(handles=[line6,line7,line8,line9,line10], bbox_to_anchor=(0.78, 0.6))
ax.add_artist(legend2)
#plt.title("duty cycles trends of Y resonant frequencies")
plt.savefig("NEW_Y_DC_trends.png",dpi=1200)
plt.show()

#%%
# Plot the predicted first order resonances for the X dimension
fig, ax = plt.subplots()
line1, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.44,0.50),color="#FF800E",label=" ")
line2, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.45,0.50),color="#C85200",label=" ")
line3, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.46,0.50),color="#5F9ED1",label=" ")
line4, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.47,0.50),color="#006BA4",label=" ")
line5, = ax.plot(mass_array,K1freq_calc(mass_array,550,0.48,0.50),color="#595959",label=" ")
legend1 = ax.legend(handles=[line1,line2,line3,line4,line5],bbox_to_anchor=(0.83, 1))
ax.add_artist(legend1)
plt.xlabel("m/z", fontsize = 13)
plt.ylabel("K=1 Parametric Frequency (kHz)", fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

# Plot the experimental first order resonant frequencies for the X dimension
w6_exp = [326,272,263,255,230,220,184]  # 50/56 X base 550 kHz
w7_exp = [321,267,258,250,226,217,180]  # 50/55 X base 550 kHz
w8_exp = [316,263,252,244,221,212,175]  # 50/54 X base 550 kHz
w9_exp = [310,256,247,239,217,207,172]  # 50/53 X base 550 kHz
w10_exp = [304,251,242,234,212,203,167]  # 50/52 X base 550 kHz
e = [4,3,3,3,3,3,2]        # approximate errors in frequency for each m/z
line6 = ax.errorbar(m_exp,w6_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#FF800E",label = "50:44")
line7 = ax.errorbar(m_exp,w7_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#C85200",label = "50:45")
line8 = ax.errorbar(m_exp,w8_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#5F9ED1",label = "50:46")
line9 = ax.errorbar(m_exp,w9_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#006BA4",label = "50:47")
line10 = ax.errorbar(m_exp,w10_exp,yerr=e,fmt="o",elinewidth=1,capsize=2,color="#595959",label = "50:48")
legend2 = ax.legend(handles=[line6,line7,line8,line9,line10], bbox_to_anchor=(1, 1))
ax.add_artist(legend2)
#plt.title("duty cycle trends of X resonant frequencies")
plt.savefig("NEW_X_DC_trends.png",dpi=1200)
plt.show()

