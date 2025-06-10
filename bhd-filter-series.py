import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.ticker import LogLocator

# Filter transmissions
ne03ab = 0.37945067
ne10ab = 0.0790946
ne20ab = 0.024822
nenir20ac = 0.0092574
nenir40ac = 0.0000
585
nenir240b = 0.0000478
nenir260b = 0.00000017
nd05a = 0.3382503
nd10a = 0.1103764
nd20a = 0.0119023
nd30a = 0.0020855
nd40a = 0.000351

###### Load CSV data #####

# Balanced
""" file_nofilter = "E:/Measurements/46/2025-03-24/nofilter.csv"
file_nd10a = "E:/Measurements/46/2025-03-24/nd10a.csv"
file_nd20a = "E:/Measurements/46/2025-03-24/nd20a.csv"
file_nd40a = "E:/Measurements/46/2025-03-24/nd40a.csv"
file_nd40a_ne10ab = "E:/Measurements/46/2025-03-24/nd40a-ne10ab.csv"
file_nd40a_ne20ab = "E:/Measurements/46/2025-03-24/nd40a-ne20ab.csv"
file_nd40a_ne10ab_ne20ab = "E:/Measurements/46/2025-03-24/nd40a-ne10ab-ne20ab.csv"
file_nd40a_nenir20ac_ne20ab = "E:/Measurements/46/2025-03-24/nd40a-nenir20ac-ne20ab.csv"
file_nd40a_nenir20ac_ne20ab_ne10ab = "E:/Measurements/46/2025-03-24/nd40a-nenir20ac-ne20ab-ne10ab.csv"
data_bal = np.array([])
for i in [file_nofilter, file_nd10a, file_nd20a, file_nd40a, file_nd40a_ne10ab, file_nd40a_ne20ab, file_nd40a_ne10ab_ne20ab, file_nd40a_nenir20ac_ne20ab, file_nd40a_nenir20ac_ne20ab_ne10ab]:
    data_bal = np.append(data_bal, np.mean(np.genfromtxt(i, delimiter=",", skip_header=21)[:,1])) """

file_400_nofilter = "E:/Measurements/46/2025-04-02/400microWLO-nofilter.csv"
file_400_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-NE10AB.csv"
file_400_nd20a = "E:/Measurements/46/2025-04-02/400microWLO-ND20A.csv"
file_400_nd30a = "E:/Measurements/46/2025-04-02/400microWLO-ND30A.csv"
file_400_nd40a = "E:/Measurements/46/2025-04-02/400microWLO-ND40A.csv"
file_400_nd40a_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NE10AB.csv"
file_400_nd40a_ne20ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NE20AB.csv"
file_400_nd40a_ne20ab_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NE20AB-NE10AB.csv"
file_400_nd40a_nenir20ac_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR20AC-NE10AB.csv"
file_400_nd40a_nenir20ac_ne10ab_ne03ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR20AC-NE10AB-NE03AB.csv"
file_400_nd40a_nenir20ac_ne10ab_ne03ab_ne03ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR20AC-NE10AB-NE03AB-NE03AB.csv"
file_400_nd40a_nenir20ac_ne20ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR20AC-NE20AB.csv"
file_400_nd40a_nenir20ac_ne20ab_ne03ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR20AC-NE20AB-NE03AB.csv"
file_400_nd40a_nenir20ac_ne20ab_ne03ab_ne03ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR20AC-NE20AB-NE03AB-NE03AB.csv"
file_400_nd40a_nenir20ac_ne20ab_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR20AC-NE20AB-NE10AB.csv"
file_400_nd40a_nenir40ac = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR40AC.csv"
file_400_nd40a_nenir40ac_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR40AC-NE10AB.csv"
file_400_nd40a_nenir40ac_ne10ab_ne03ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR40AC-NE10AB-NE03AB.csv"
file_400_nd40a_nenir40ac_ne10ab_ne03ab_ne03ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR40AC-NE10AB-NE03AB-NE03AB.csv"
file_400_nd40a_nenir40ac_ne20ab = "E:/Measurements/46/2025-04-02/400microWLO-ND40A-NENIR40AC-NE20AB.csv"
data_400_bal = np.array([])
for i in [file_400_nofilter, file_400_ne10ab, file_400_nd20a, file_400_nd30a, file_400_nd40a, file_400_nd40a_ne10ab, file_400_nd40a_ne10ab, file_400_nd40a_ne20ab, file_400_nd40a_ne20ab_ne10ab, file_400_nd40a_nenir20ac_ne10ab, file_400_nd40a_nenir20ac_ne10ab_ne03ab, file_400_nd40a_nenir20ac_ne10ab_ne03ab_ne03ab, file_400_nd40a_nenir20ac_ne20ab, file_400_nd40a_nenir20ac_ne20ab_ne03ab, file_400_nd40a_nenir20ac_ne20ab_ne03ab_ne03ab, file_400_nd40a_nenir20ac_ne20ab_ne10ab, file_400_nd40a_nenir40ac, file_400_nd40a_nenir40ac_ne10ab, file_400_nd40a_nenir40ac_ne10ab_ne03ab, file_400_nd40a_nenir40ac_ne10ab_ne03ab_ne03ab, file_400_nd40a_nenir40ac_ne20ab]:
    data_400_bal = np.append(data_400_bal, np.ptp(np.genfromtxt(i, delimiter=",", skip_header=21)[:,1]))

data_400_bal[6] *= 0.6
data_400_bal[7] *= 0.3
data_400_bal[8] *= 0.7

# Log
""" file_log_ND40A_low = "E:/Measurements/46/2025-03-31/log-ND40A-low.csv"
file_log_ND40A_high = "E:/Measurements/46/2025-03-31/log-ND40A-high.csv"
file_log_ND40A_NE10AB_low = "E:/Measurements/46/2025-03-31/log-ND40A-NE10AB-low.csv"
file_log_ND40A_NE10AB_high = "E:/Measurements/46/2025-03-31/log-ND40A-NE10AB-high.csv"
file_log_ND40A_NE20AB_low = "E:/Measurements/46/2025-03-31/log-ND40A-NE20AB-low.csv"
file_log_ND40A_NE20AB_high = "E:/Measurements/46/2025-03-31/log-ND40A-NE20AB-high.csv"
data_log = np.array([])
data_log_span = np.array([])
for i in [file_log_ND40A_low, file_log_ND40A_high, file_log_ND40A_NE10AB_low, file_log_ND40A_NE10AB_high, file_log_ND40A_NE20AB_low, file_log_ND40A_NE20AB_high]:
    data_log = np.append(data_log, np.mean(np.genfromtxt(i, delimiter=",", skip_header=21)[:,1]))
data_log_span = np.append(data_log_span, [np.abs(data_log[1] - data_log[0]), np.abs(data_log[3] - data_log[2]), np.abs(data_log[5] - data_log[4])]) """
""" data = np.append(np.abs(np.mean(file_log_ND40A_high) - np.mean(file_log_ND40A_low)))
data = np.append(np.abs(np.mean(file_log_ND40A_NE10AB_high) - np.mean(file_log_ND40A_NE10AB_low)))
data = np.append(np.abs(np.mean(file_log_ND40A_NE20AB_high) - np.mean(file_log_ND40A_NE20AB_low))) """

""" file_log_400_nofilter = "E:/Measurements/46/2025-04-02/400microWLO-log-nofilter.csv"
file_log_400_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-log-NE10AB.csv"
file_log_400_nd20a = "E:/Measurements/46/2025-04-02/400microWLO-log-ND20A.csv"
file_log_400_nd30a = "E:/Measurements/46/2025-04-02/400microWLO-log-ND30A.csv"
file_log_400_nd40a = "E:/Measurements/46/2025-04-02/400microWLO-log-ND40A.csv"
file_log_400_nd40a_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-log-ND40A-NE10AB.csv"
file_log_400_nd40a_ne20ab = "E:/Measurements/46/2025-04-02/400microWLO-log-ND40A-NE20AB.csv"
file_log_400_nd40a_ne20ab_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-log-ND40A-NE20AB-NE10AB.csv"
file_log_400_nd40a_nenir20ac_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-log-ND40A-NENIR20Ac-NE10AB.csv"
file_log_400_nd40a_nenir20ac_ne20ab = "E:/Measurements/46/2025-04-02/400microWLO-log-ND40A-NENIR20Ac-NE20AB.csv"
file_log_400_nd40a_nenir20ac_ne20ab_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-log-ND40A-NENIR20Ac-NE20AB-NE10AB.csv"
file_log_400_nd40a_nenir20ac_ne20ab_ne10ab_ne10ab = "E:/Measurements/46/2025-04-02/400microWLO-log-ND40A-NENIR20Ac-NE20AB-NE10AB.csv" """

# begin 02-05-2025
""" file_bal_400_ne03ab = "E:/Measurements/46/2025-05-02/bhd-400microW-LO-NE03AB.csv"
file_bal_400_ne10ab = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-NE10AB.csv"
file_bal_400_ne20a = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-NE20AB.csv"
file_bal_400_nd30a = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND30AB.csv"
file_bal_400_nd40a = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB.csv"
file_bal_400_nd40a_ne10ab = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB-NE10AB.csv"
file_bal_400_nd40a_ne20ab = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB-NE20AB.csv"
file_bal_400_nd40a_nenir20ac = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB-NENIR20AC.csv"
file_bal_400_nd40a_nenir20ac_ne10ab = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB-NENIR20AC-NE10AB.csv"
file_bal_400_nd40a_nenir20ac_ne20ab = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB-NENIR20AC-NE20AB.csv"
file_bal_400_nd40a_nenir20ac_ne20ab_ne10ab = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB-NENIR20AC-NE20AB-NE10AB.csv"
file_bal_400_nd40a_nenir40ac = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB-NENIR40AC.csv"
file_bal_400_nd40a_nenir40ac_ne10ab = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB-NENIR40AC-NE10AB.csv"
file_bal_400_nd40a_nenir40ac_ne10ab_ne03ab = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB-NENIR40AC-NE10AB-NE03AB.csv"
file_bal_400_nd40a_nenir40ac_ne20ab = "E:/Measurements/46/2025-05-02/bhd-bal-400microW-LO-ND40AB-NENIR40AC-NE20AB.csv"

file_autobal_400_ne03ab = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-NE03AB.csv"
file_autobal_400_ne10ab = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-NE10AB.csv"
file_autobal_400_ne20a = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-NE20AB.csv"
file_autobal_400_nd30a = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-ND30AB.csv"
file_autobal_400_nd40a = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-ND40AB.csv"
file_autobal_400_nd40a_ne10ab = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-ND40AB-NE10AB.csv"
file_autobal_400_nd40a_ne20ab = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-ND40AB-NE20AB.csv"
file_autobal_400_nd40a_nenir20ac = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-ND40AB-NENIR20AC.csv"
file_autobal_400_nd40a_nenir20ac_ne10ab = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-ND40AB-NENIR20AC-NE10AB.csv"
file_autobal_400_nd40a_nenir20ac_ne20ab = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-ND40AB-NENIR20AC-NE20AB.csv"
file_autobal_400_nd40a_nenir40ac = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-ND40AB-NENIR40AC.csv"
file_autobal_400_nd40a_nenir40ac_ne10ab = "E:/Measurements/46/2025-05-02/bhd-autobal-400microW-LO-ND40AB-NENIR40AC-NE10AB.csv"
data_bal_400 = np.array([])
for i in [file_bal_400_ne03ab, file_bal_400_ne10ab, file_bal_400_ne20a, file_bal_400_nd30a, file_bal_400_nd40a, file_bal_400_nd40a_ne10ab, file_bal_400_nd40a_ne20ab, file_bal_400_nd40a_nenir20ac, file_bal_400_nd40a_nenir20ac_ne10ab, file_bal_400_nd40a_nenir20ac_ne20ab, file_bal_400_nd40a_nenir20ac_ne20ab_ne10ab, file_bal_400_nd40a_nenir40ac, file_bal_400_nd40a_nenir40ac_ne10ab, file_bal_400_nd40a_nenir40ac_ne10ab_ne03ab, file_bal_400_nd40a_nenir40ac_ne20ab]:
    data_bal_400 = np.append(data_bal_400, np.ptp(np.genfromtxt(i, delimiter=",", skip_header=21)[:,1]))

data_autobal_400 = np.array([])
for i in [file_autobal_400_ne03ab,
          file_autobal_400_ne10ab,
          file_autobal_400_ne20a,
          file_autobal_400_nd30a,
          file_autobal_400_nd40a,
          file_autobal_400_nd40a_ne10ab,
          file_autobal_400_nd40a_ne20ab,
          file_autobal_400_nd40a_nenir20ac,
          file_autobal_400_nd40a_nenir20ac_ne10ab,
          file_autobal_400_nd40a_nenir20ac_ne20ab,
          file_autobal_400_nd40a_nenir40ac,
          file_autobal_400_nd40a_nenir40ac_ne10ab]:
    data_autobal_400 = np.append(data_autobal_400, np.ptp(np.genfromtxt(i, delimiter=",", skip_header=21)[:,1]))

transmission_400_autobal = np.array([ne03ab, 
                                 ne10ab,
                                 ne20ab, 
                                 nd30a,
                                 nd40a,
                                 nd40a*ne10ab,
                                 nd40a*ne20ab,
                                 nd40a*nenir20ac,
                                 nd40a*nenir20ac*ne10ab,
                                 nd40a*nenir20ac*ne20ab,
                                 nd40a*nenir40ac,
                                 nd40a*nenir40ac*ne10ab,
                                 ])

transmission_400_bal = np.array([ne03ab, 
                                 ne10ab,
                                 ne20ab, 
                                 nd30a,
                                 nd40a,
                                 nd40a*ne10ab,
                                 nd40a*ne20ab,
                                 nd40a*nenir20ac,
                                 nd40a*nenir20ac*ne10ab,
                                 nd40a*nenir20ac*ne20ab,
                                 nd40a*nenir20ac*ne20ab*ne10ab,
                                 nd40a*nenir40ac,
                                 nd40a*nenir40ac*ne10ab,
                                 nd40a*nenir40ac*ne10ab*ne03ab,
                                 nd40a*nenir40ac*ne20ab
                                 ]) """
# end 02-05-2024

# begin 05-05-2025
file_bal_400_ne03ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-ne03ab.csv"
file_bal_400_ne10ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-ne10ab.csv"
file_bal_400_ne20a = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-ne20ab.csv"
file_bal_400_nd30a = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd30ab.csv"
file_bal_400_nd40a = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab.csv"
file_bal_400_nd40a_ne10ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-ne10ab.csv"
file_bal_400_nd40a_ne20ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-ne20ab.csv"
file_bal_400_nd40a_nenir20ac = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir20ac.csv"
file_bal_400_nd40a_nenir20ac_ne10ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir20ac-ne10ab.csv"
file_bal_400_nd40a_nenir20ac_ne20ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir20ac-ne20ab.csv"
file_bal_400_nd40a_nenir20ac_ne20ab_ne10ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir20ac-ne20ab-ne10ab.csv"
file_bal_400_nd40a_nenir40ac = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir40ac-2.csv"
file_bal_400_nd40a_nenir40ac_ne03ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir40ac-ne03ab.csv"
file_bal_400_nd40a_nenir40ac_ne10ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir40ac-ne10ab.csv"
file_bal_400_nd40a_nenir40ac_ne10ab_ne03ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir40ac-ne10ab-ne03ab.csv"
file_bal_400_nd40a_nenir40ac_ne20ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir40ac-ne20ab.csv"
file_bal_400_nd40a_nenir40ac_nenir20ac = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir40ac-nenir20ac.csv"
file_bal_400_nd40a_nenir40ac_nenir20ac_ne03ab = "E:/Measurements/46/2025-05-05/bhd-bal-400microW-LO-nd40ab-nenir40ac-nenir20ac-ne03ab.csv"

data_bal_400 = np.array([])
for i in [file_bal_400_ne03ab,
          file_bal_400_ne10ab,
          file_bal_400_ne20a,
          file_bal_400_nd30a,
          file_bal_400_nd40a,
          file_bal_400_nd40a_ne10ab,
          file_bal_400_nd40a_ne20ab,
          file_bal_400_nd40a_nenir20ac,
          file_bal_400_nd40a_nenir20ac_ne10ab,
          file_bal_400_nd40a_nenir20ac_ne20ab,
          file_bal_400_nd40a_nenir20ac_ne20ab_ne10ab,
          file_bal_400_nd40a_nenir40ac,
          file_bal_400_nd40a_nenir40ac_ne03ab,
          file_bal_400_nd40a_nenir40ac_ne10ab,
          file_bal_400_nd40a_nenir40ac_ne10ab_ne03ab,
          file_bal_400_nd40a_nenir40ac_ne20ab,
          file_bal_400_nd40a_nenir40ac_nenir20ac,
          file_bal_400_nd40a_nenir40ac_nenir20ac_ne03ab
          ]:
    data_bal_400 = np.append(data_bal_400, np.ptp(np.genfromtxt(i, delimiter=",", skip_header=21)[:,1]))

transmission_400_bal = np.array([ne03ab, 
                                 ne10ab,
                                 ne20ab, 
                                 nd30a,
                                 nd40a,
                                 nd40a*ne10ab,
                                 nd40a*ne20ab,
                                 nd40a*nenir20ac,
                                 nd40a*nenir20ac*ne10ab,
                                 nd40a*nenir20ac*ne20ab,
                                 nd40a*nenir20ac*ne20ab*ne10ab,
                                 nd40a*nenir40ac,
                                 nd40a*nenir40ac*ne03ab,
                                 nd40a*nenir40ac*ne10ab,
                                 nd40a*nenir40ac*ne10ab*ne03ab,
                                 nd40a*nenir40ac*ne20ab,
                                 nd40a*nenir40ac*nenir20ac,
                                 nd40a*nenir40ac*nenir20ac*ne03ab
                                 ])

file_autobal_400_ne03ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-ne03ab.csv"
file_autobal_400_ne10ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-ne10ab.csv"
file_autobal_400_ne20a = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-ne20ab.csv"
file_autobal_400_nd30a = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd30ab.csv"
file_autobal_400_nd40a = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab.csv"
file_autobal_400_nd40a_ne10ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-ne10ab.csv"
file_autobal_400_nd40a_ne20ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-ne20ab.csv"
file_autobal_400_nd40a_nenir20ac = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir20ac.csv"
file_autobal_400_nd40a_nenir20ac_ne10ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir20ac-ne10ab.csv"
file_autobal_400_nd40a_nenir20ac_ne20ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir20ac-ne20ab.csv"
file_autobal_400_nd40a_nenir20ac_ne20ab_ne10ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir20ac-ne20ab-ne10ab.csv"
file_autobal_400_nd40a_nenir40ac = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir40ac-2.csv"
file_autobal_400_nd40a_nenir40ac_ne03ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir40ac-ne03ab.csv"
file_autobal_400_nd40a_nenir40ac_ne10ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir40ac-ne10ab.csv"
file_autobal_400_nd40a_nenir40ac_ne10ab_ne03ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir40ac-ne10ab-ne03ab.csv"
file_autobal_400_nd40a_nenir40ac_ne20ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir40ac-ne20ab.csv"
file_autobal_400_nd40a_nenir40ac_nenir20ac = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir40ac-nenir20ac.csv"
file_autobal_400_nd40a_nenir40ac_nenir20ac_ne03ab = "E:/Measurements/46/2025-05-05/bhd-autobal-400microW-LO-nd40ab-nenir40ac-nenir20ac-ne03ab.csv"

data_autobal_400 = np.array([])
for i in [file_autobal_400_ne03ab,
          file_autobal_400_ne10ab,
          file_autobal_400_ne20a,
          file_autobal_400_nd30a,
          file_autobal_400_nd40a,
          file_autobal_400_nd40a_ne10ab,
          file_autobal_400_nd40a_ne20ab,
          file_autobal_400_nd40a_nenir20ac,
          file_autobal_400_nd40a_nenir20ac_ne10ab,
          file_autobal_400_nd40a_nenir20ac_ne20ab,
          file_autobal_400_nd40a_nenir20ac_ne20ab_ne10ab,
          file_autobal_400_nd40a_nenir40ac,
          file_autobal_400_nd40a_nenir40ac_ne03ab,
          file_autobal_400_nd40a_nenir40ac_ne10ab,
          file_autobal_400_nd40a_nenir40ac_ne10ab_ne03ab,
          file_autobal_400_nd40a_nenir40ac_ne20ab,
          file_autobal_400_nd40a_nenir40ac_nenir20ac,
          file_autobal_400_nd40a_nenir40ac_nenir20ac_ne03ab
          ]:
    data_autobal_400 = np.append(data_autobal_400, np.ptp(np.genfromtxt(i, delimiter=",", skip_header=21)[:,1]))

transmission_400_autobal = np.array([ne03ab, 
                                 ne10ab,
                                 ne20ab, 
                                 nd30a,
                                 nd40a,
                                 nd40a*ne10ab,
                                 nd40a*ne20ab,
                                 nd40a*nenir20ac,
                                 nd40a*nenir20ac*ne10ab,
                                 nd40a*nenir20ac*ne20ab,
                                 nd40a*nenir20ac*ne20ab*ne10ab,
                                 nd40a*nenir40ac,
                                 nd40a*nenir40ac*ne03ab,
                                 nd40a*nenir40ac*ne10ab,
                                 nd40a*nenir40ac*ne10ab*ne03ab,
                                 nd40a*nenir40ac*ne20ab,
                                 nd40a*nenir40ac*nenir20ac,
                                 nd40a*nenir40ac*nenir20ac*ne03ab
                                 ])
# end 05-05-2024

a = [nenir40ac, nenir40ac*nd40a, nenir20ac]
b = [0.021844028496636424, 0.0010439771620273082, 0.00043168214621558995]


#data_log_400[0] *= 1.5

r = 0.67                # Photodetector responsivity (A/W)
g = 100e3               # V/A
lo_power = 0.1975e-3    # W
lo_voltage = g * r * lo_power 
#data_power = data / (g * r)  # Maybe factor 2 because of Interferogram peak to peak
#data_voltage = data_bal
transmission = np.array([1, 0.1104, 0.01190, 0.0003510, 0.0003510*0.0791, 0.0003510*0.02482, 0.0003510*0.0791*0.02482, 0.0003510*0.009257*0.02482, 0.0003510*0.009257*0.02482*0.1104])
signal_power = transmission * lo_power
signal_voltage = transmission * lo_voltage
#expected = 4 * np.sqrt(signal_voltage) * np.sqrt(lo_voltage)
x_axis = np.linspace(10, 1e-20, 10000)#[::-1]  # Reverse the array
x_axis_voltage = x_axis * g * r
expected = 4 * np.sqrt(x_axis) * np.sqrt(lo_power) #/ (constants.epsilon_0 * constants.c)
expected_voltage = g * r * expected * 0.25
#theory = 4 * x_axis * lo_voltage
#theory = 4 * x_axis * lo_voltage
#print(4 * np.sqrt(2e-4) * np.sqrt(2e-4))
#print(lo_power*4*g*r)

""" lo_power_log = 0.2e-3
transmission_log = np.array([0.0003510, 0.0003510*0.0791, 0.0003510*0.02482])
signal_power_log = transmission_log * lo_power_log

lo_power_400 = 0.4e-3
transmission_400_bal = np.array([1, 0.0791, 0.0119, 0.00209, 0.000351, 0.0003510*0.0791, 0.0003510*0.02482, 0.0003510*0.02482*0.0791, 0.0003510*0.009257*0.0791, 0.0003510*0.009257*0.0791, 0.0003510*0.009257*0.0791*0.3795, 0.0003510*0.009257*0.0791*0.3795*0.3795, 0.0003510*0.009257*0.02482, 0.0003510*0.009257*0.02482*0.3795, 0.0003510*0.009257*0.02482*0.3795*0.3795, 0.0003510*0.009257*0.02482*0.0791, 0.0003510*0.00005849, 0.0003510*0.00005849*0.0791, 0.0003510*0.00005849*0.0791*0.3795, 0.0003510*0.00005849*0.0791*0.3795*0.3795, 0.0003510*0.00005849*0.02482])
signal_power_400_bal = transmission_400_bal * lo_power_400

transmission_400_log = np.array([1, 0.0791, 0.0119, 0.00209, 0.000351, 0.0003510*0.0791, 0.0003510*0.02482, 0.0003510*0.02482*0.0791, 0.0003510*0.009257*0.0791, 0.0003510*0.009257*0.02482, 0.0003510*0.009257*0.02482*0.0791, 0.0003510*0.009257*0.02482*0.0791*0.3795])
signal_power_400_log = transmission_400_log * lo_power_400 """

# 02-05-2024
lo_power_log = 0.4e-3
transmission_log = np.array([0.0003510, 0.0003510*0.0791, 0.0003510*0.02482])
signal_power_log = transmission_log * lo_power_log

lo_power_400 = 0.4e-3
#transmission_400_bal = np.array([0.379507, 0.0791, 0.0119, 0.00209, 0.000351, 0.0003510*0.0791, 0.0003510*0.02482, 0.0003510*0.02482*0.0791, 0.0003510*0.009257*0.0791, 0.0003510*0.009257*0.0791, 0.0003510*0.009257*0.0791*0.3795, 0.0003510*0.009257*0.0791*0.3795*0.3795, 0.0003510*0.009257*0.02482, 0.0003510*0.009257*0.02482*0.3795, 0.0003510*0.009257*0.02482*0.3795*0.3795, 0.0003510*0.009257*0.02482*0.0791, 0.0003510*0.00005849, 0.0003510*0.00005849*0.0791, 0.0003510*0.00005849*0.0791*0.3795, 0.0003510*0.00005849*0.0791*0.3795*0.3795, 0.0003510*0.00005849*0.02482])
signal_power_400_bal = transmission_400_bal * lo_power_400

transmission_400_log = np.array([1, 0.0791, 0.0119, 0.00209, 0.000351, 0.0003510*0.0791, 0.0003510*0.02482, 0.0003510*0.02482*0.0791, 0.0003510*0.009257*0.0791, 0.0003510*0.009257*0.02482, 0.0003510*0.009257*0.02482*0.0791, 0.0003510*0.009257*0.02482*0.0791*0.3795])
signal_power_400_log = transmission_400_log * lo_power_400

signal_power_400_autobal = transmission_400_autobal * lo_power_400
##########

wavelength = 1064e-9
frequency = constants.c / wavelength
shot_noise = np.sqrt(2 * constants.h * constants.c / (wavelength * 2 * lo_power_400))
one_photon = constants.h * constants.c / wavelength
v_shot = g * np.sqrt(2 * constants.elementary_charge * lo_power_400 * r)    # 1*lo_power because shot_noise level is relevant at low signal arm powers only
photons_shot = v_shot / (constants.h * frequency)

E_photon = constants.h * frequency
# Function to convert Power (W) to Photons per second
def power_to_photons(P):
    return P / E_photon

""" # Fit for filter series curve to see interception point
slope, intercept = np.polyfit(signal_power_400_bal[:10], data_400_bal[:10], 1)
y_fit = slope * x_axis + intercept """

#print(theory)
# Plot the results
#plt.semilogy(frequencies, psd_balanced_400_sqrt, label='Balanced 400 mW')
#ax1.axhline(shot_noise_psd_800, color='r', linestyle='--', label="Shot Noise 800 microW")
#ax2 = ax1.twinx()
#ax2.plot(psd_fitted)
""" plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (Hz$^{-0.5}$)')
plt.legend()
plt.title('Power Spectral Density vs. Shot Noise')
ax1.grid(True, which='both', linestyle='--', alpha=0.6) """
""" fig, ax1 = plt.subplots(figsize=(8,6))
ax1.loglog(x_axis, theory)
ax1.loglog(signal_power, data)
plt.show() """

####### add labels etc.
#plt.figure(figsize=(10, 6))
fig, ax1 = plt.subplots(figsize=(8,6))
#plt.loglog(x_axis_voltage, expected_voltage)
plt.loglog(x_axis, expected_voltage, label='g*r*4*Es*ELO')
#plt.loglog(x_axis, y_fit, label="Balanced Fit")
plt.gca().invert_xaxis()  # Inverts the x-axis
#plt.loglog(signal_voltage, data)
plt.loglog(signal_power_400_bal, data_bal_400, '--o',color="green", label='Measured Signal Balanced')# (Linear Output)')
plt.loglog(a,b, '--o',color="green", label='test')# (Linear Output)')
plt.loglog(signal_power_400_autobal, data_autobal_400, '--o',color="blue" , label='Measured Signal Autobalanced')# (Log Output)')
plt.axhline(v_shot, color='r', linestyle='--', label="Shot Noise")
plt.axvline(one_photon, color='black', linestyle='--', label="One Photon/s")
plt.xlim(1e-2, 1e-21)
plt.ylim(1e-7, 100)
plt.xlabel('Signal Arm Power [W]')
plt.ylabel('Measured Signal [V]')
plt.title('BHD Signal vs. Signal Arm Power')
plt.legend(loc='lower left', bbox_to_anchor=(0,0.15))
ax1.xaxis.grid(visible=True, which='both')
ax1.yaxis.grid(visible=True, which='major')
ax1.axvspan(signal_power_400_bal[0], 1e-12, alpha=0.1, color='green')

ax2 = ax1.secondary_xaxis("top", functions=(power_to_photons, lambda N: N * E_photon))  # transform function and its inverse
ax2.set_xlabel("Signal Arm [Photons/s]")

""" # Set log tick locators explicitly to ensure minor ticks appear
ax1.xaxis.set_major_locator(LogLocator(base=10.0, subs=None))  # Major ticks
ax1.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))  # Minor ticks

ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))  # Major ticks for y
ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)) """


""" 
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dBm)")
plt.xlim(0,50e3)
plt.grid()
plt.legend() """
plt.show()