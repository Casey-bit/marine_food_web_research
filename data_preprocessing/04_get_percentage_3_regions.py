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

latitudeByYear = load_variavle(r'chugao\fish_220114_re\cluster\latitudeByYear_occurrence_1970_2020_2846_family.txt')
familynum = len(latitudeByYear[0])

# split these records into hemispheres
standard = [[[[] for hem in range(2)] for category in range(familynum)] for year in range(yearTotal)]
for y in range(yearTotal):
    for c in range(familynum):
        for h in range(2):
            standard[y][c][0] = [item for item in latitudeByYear[y][c] if item >= 0]
            standard[y][c][1] = [item for item in latitudeByYear[y][c] if item < 0]
record = np.array(standard)

# percentage in 3 regions (0-30N  30N-60N  60N-90N)
percent = np.array([[[0.0 for ran in range(3)] for family in range(familynum)] for year in range(yearTotal)])
for year in range(yearTotal):
    for family in range(familynum):
        tmp = record[year, family, 0]
        if len(tmp):
            percent[year, family, 0] = len([i for i in tmp if i < 30]) / len(tmp)
            percent[year, family, 1] = len([i for i in tmp if 30 <= i < 60]) / len(tmp)
            percent[year, family, 2] = len([i for i in tmp if i >= 60]) / len(tmp)
        else:
            if year == 0:
                percent[year, family, 0] = np.nan
                percent[year, family, 1] = np.nan
                percent[year, family, 2] = np.nan
            else:
                percent[year, family, 0] = percent[year - 1, family, 0]
                percent[year, family, 1] = percent[year - 1, family, 1]
                percent[year, family, 2] = percent[year - 1, family, 2]
    print(year)

save_variable(percent,r'chugao\fish_220114_re\cluster\percent_family_2846.txt')