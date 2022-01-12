import numpy as np
import xlrd
from pathlib import Path
import os

def Excel2NP(xl_list):
    return np.asarray([float(v.value) for v in xl_list if v.ctype==2])


WBname = "population structure.xlsx"
loc = Path("C:/Users/yoel/Dropbox/SocialStructureGraph/statistical materials/" + WBname)
file_path = os.path.dirname(__file__)

# To open Workbook
try:
    wb = xlrd.open_workbook(os.path.join(file_path,WBname))
except:
    #  use local workbook in this dir.
    wb = xlrd.open_workbook(loc)

s = wb.sheet_by_name('age distributions')
headers = s.row(1)
# For row 0 and column 0
print(headers)
ageid = Excel2NP(s.col(0, 2))
age_dist_pdf = Excel2NP(s.col(1, 2))
single_age_pdf = Excel2NP(s.col(2, 2))
mums_age = Excel2NP(s.col(10, 2))
dads_delta = Excel2NP(s.col(9, 2))
dad_factor_x = (-10, 15)
birth_age = Excel2NP(s.col(5, 2))


family_dist = np.asarray([0.211,0.021,0.000,0.000,0.000,0.000,
                          0.168,0.084,0.105,0.105,0.042,0.032,
                          0.070,0.028,0.028,0.014,0.007,0.000,
                          0.026,0.016,0.011,0.005,0.000,0.000,
                          0.008,0.008,0.008,0.000,0.000,0.000])

""""
age_dist_pdf = np.concatenate([0.0994 * np.ones(5)/5, 0.0948 * np.ones(5)/5, 0.0846 * np.ones(5)/5,
                               0.0771 * np.ones(5)/5, 0.0719 * np.ones(5)/5,  0.0665 * np.ones(5)/5, 0.066 * np.ones(5)/5, 0.0652 * np.ones(5)/5,
                               0.063 * np.ones(5)/5, 0.0566 * np.ones(5)/5, 0.0477 * np.ones(5)/5, 0.0437 * np.ones(5)/5, 0.0413 * np.ones(5)/5,
                               0.0411 * np.ones(5)/5, 0.0318 * np.ones(5)/5, 0.0492 * np.ones(25)/25], axis=0)
age_dist_pdf /= np.sum(age_dist_pdf)


single_age_dist = np.concatenate([np.zeros(20), 0.03*np.ones(5)/5, 0.0635*np.ones(5)/5, 0.06*np.ones(5)/5, 0.06*np.ones(5)/5,
                                  0.055*np.ones(5)/5, 0.045*np.ones(5)/5, 0.06*np.ones(5)/5, 0.08*np.ones(5)/5, 0.1075*np.ones(5)/5,
                                  0.1575*np.ones(5)/5, 0.19*np.ones(5)/5, 0.245*np.ones(5)/5, 0.295*np.ones(20)/20], axis=0)

single_age_dist = np.multiply(single_age_dist, age_dist_pdf)
single_age_dist /= np.sum(single_age_dist)

mums_age = np.concatenate([np.zeros(18), 0.01*np.ones(2)/2, 0.15*np.ones(5)/5, 0.42*np.ones(5)/5, 0.69*np.ones(5)/5, 0.75*np.ones(5)/5,
                           0.71*np.ones(5)/5, 0.66*np.ones(5)/5, 0.48*np.ones(5)/5, 0.25*np.ones(5)/5, 0.03*np.ones(40)/40], axis=0)
mums_age_orig = np.concatenate([np.zeros(18), 0.01*np.ones(2)/2, 0.15*np.ones(5)/5, 0.42*np.ones(5)/5, 0.69*np.ones(5)/5, 0.75*np.ones(5)/5,
                           0.71*np.ones(5)/5, 0.71*np.ones(5)/5, 0.66*np.ones(5)/5, 0.48*np.ones(5)/5, 0.25*np.ones(5)/5, 0.03*np.ones(35)/35])
mums_age = np.multiply(mums_age, age_dist_pdf)
mums_age /= np.sum(mums_age)

partner_age_delta = np.asarray([0.00E+00,4.32E-04,2.65E-03,6.16E-03,1.48E-02,8.27E-02,1.15E-01,1.13E-01,9.38E-02,4.20E-02,2.57E-02,1.18E-02,4.35E-03])
partner_age_delta /= np.sum(partner_age_delta)
partner_age_delta_x = (-10, 14)


kids_age_delta = np.concatenate([np.zeros(18), np.ones(2)/2*0.405506884, np.ones(5)/5*2.651564456, np.ones(5)/5*8.390613267,
                                 np.ones(5)/5*11.63053817, np.ones(5)/5*5.788485607,np.ones(2)/5*1.10387985, np.zeros(58)])

kids_age_delta /= np.sum(kids_age_delta)

kids_age_dist = np.concatenate([0.0994 * np.ones(5)/5, 0.0948 * np.ones(5)/5, 0.0846 * np.ones(5), 0.0771 * np.ones(5)/5])
kids_age_dist /= np.sum(kids_age_dist)

kids_x = (0, 18)
kids_x = age_dist_pdf[range(kids_x[0], kids_x[1])]
adults_x = (19, 100)
adults_x = age_dist_pdf[range(adults_x[0], adults_x[1])]
"""


couple_no_kids = np.concatenate([np.zeros(15), 0.172268041*np.ones(15)/(30-15), 0.11*np.ones(20)/20, 0.717731959*np.ones(50)/50])
couple_no_kids = np.multiply(couple_no_kids, age_dist_pdf)
couple_no_kids /= sum(couple_no_kids)


population_age_dist = dict(family_dist=family_dist, age_dist_pdf=age_dist_pdf, single_age_dist=single_age_pdf,
                           couple_no_kids=couple_no_kids, kids_age_dist=[], mums_age=mums_age,
                           partner_age_delta=dads_delta, partner_age_delta_x=dad_factor_x,
                           kids_age_delta=birth_age, kids_x=[], adults_x=[])
