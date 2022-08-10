import numpy as np
import pickle

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

start = 1970
end = 2020
yearTotal = end - start + 1

# year family range
percent = load_variavle(r'chugao\fish_220114_re\cluster\percent_family_2846.txt')
reserve = load_variavle(r'chugao\fish_220114_re\correlation_median_point\reserve_family.txt')
countByYear = load_variavle('chugao\\fish_220114_re\\countByYear_1degree_1970_2020_2846_family.txt')
record = np.array(countByYear)

family_into = [[[] for ran in range(3)] for year in range(yearTotal - 1)]
family_out_of = [[[] for ran in range(3)] for year in range(yearTotal - 1)]

for year in range(yearTotal - 1):
    total = np.sum([record[year: year + 2, old, 90:] for new, old in reserve])
    print(year,total)
    for family in range(2846):
        if family in [old for new,old in reserve]:
            newfamily = [new for new,old in reserve if old == family][0]
            level1 = [new for new,old in reserve if old == family]
            if len(level1):
                level = level1[0]
                for ran in range(3):
                    percent1 = percent[year + 1, newfamily, ran]
                    percent1_3 = percent[year + 1, newfamily]
                    # percent1_3[ran] -= 0.03

                    percent0 = percent[year, newfamily, ran]
                    percent0_3 = percent[year, newfamily]
                    # percent0_3[ran] -= 0.03
                    if percent1 == np.max(percent1_3) and not percent0 == np.max(percent0_3):
                        family_into[year][ran].append(level)
                    if percent0 == np.max(percent0_3) and not percent1 == np.max(percent1_3):
                        family_out_of[year][ran].append(level)

aveFamilyUp = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 1)])
aveFamilyDown = np.array([[0.0 for ran in range(3)] for year in range(yearTotal - 1)])

for year in range(yearTotal - 1):
    for ran in range(3):
        aveFamilyUp[year, ran] = np.mean(family_into[year][ran])
        aveFamilyDown[year, ran] = np.mean(family_out_of[year][ran])

save_variable(family_into,r'chugao\fish_220114_re\figure2\extra\family_into_familyindex.txt')
save_variable(family_out_of,r'chugao\fish_220114_re\figure2\extra\family_out_of_familyindex.txt')