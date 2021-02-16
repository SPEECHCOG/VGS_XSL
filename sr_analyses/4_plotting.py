import scipy.io
import numpy

from scipy import stats
import matplotlib
from matplotlib import pyplot as plt 
###############################################################################
                        # User-defined variables #
###############################################################################
selection_number = 5
n_bins = 50

model ='CNN1'

file_in =  model + '.mat'
path_in = ''

path_out = ''
file_out = 'places_' + model
                
################################################################################ Functions    

def seleting_tops (top,lowest,randsim,selection_number):
                                                                   
    sQ = []
    sL = []
    sR = []
    
    for i in range(len(top)):
        
        q = top[i][0:selection_number]   
        l = lowest[i][0:selection_number]
        r = randsim[i][0:selection_number]
        
        sQ.extend(q)
        sL.extend(l) 
        sR.extend(r)
    return sQ,sL,sR

def load_data(inputPath,inputModel):
    data = scipy.io.loadmat(inputPath + inputModel, variable_names=['top_similarity1','top_similarity2','low_similarity1','low_similarity2',
                                                             'random_similarity1','random_similarity2'])
    top_similarity1= data['top_similarity1']
    lowest_similarity1= data['low_similarity1']
    random_similarity1 = data['random_similarity1']
    
    top_similarity2= data['top_similarity2']
    lowest_similarity2= data['low_similarity2']
    random_similarity2 = data['random_similarity2']
    
    # top_similarity1,lowest_similarity1,random_similarity1 = correct_elements(top_similarity1,lowest_similarity1,random_similarity1)
    # top_similarity2,lowest_similarity2,random_similarity2 = correct_elements(top_similarity2,lowest_similarity2,random_similarity2)
    
    return top_similarity1,top_similarity2,lowest_similarity1,lowest_similarity2,random_similarity1,random_similarity2

###############################################################################
    
                             # Median and SD #
                             
###############################################################################                        


top_similarity1,top_similarity2,lowest_similarity1,lowest_similarity2,random_similarity1,random_similarity2 = load_data(path_in,file_in)

print('............ with repeating words ....................................')

sim_m2 = round(numpy.median(top_similarity2), 2)
print(sim_m2)
sim_std2 = round(numpy.std(top_similarity2),2)
print(sim_std2)

dist_m2 = round(numpy.median(lowest_similarity2), 2)
print(dist_m2)
dist_std2 = round(numpy.std(lowest_similarity2),2)
print(dist_std2)

r_m2 = round(numpy.median(random_similarity2), 2)
print(r_m2)
r_std2 = round(numpy.std(random_similarity2),2)
print(r_std2)


print('............ without repeating words ....................................')

sim_m1 = round(numpy.median(top_similarity1), 2)
print(sim_m1)
sim_std1 = round(numpy.std(top_similarity1),2)
print(sim_std1)

dist_m1 = round(numpy.median(lowest_similarity1), 2)
print(dist_m1)
dist_std1 = round(numpy.std(lowest_similarity1),2)
print(dist_std1)

r_m1 = round(numpy.median(random_similarity1),2)
print(r_m1)
r_std1 = round(numpy.std(random_similarity1),2)
print(r_std1)

###############################################################################

                                    # PLOTTING #

###############################################################################

plt.figure(figsize=[20,10])  
############################################################################### # Similarity 2   
t_S,t_D,t_R = seleting_tops(top_similarity2,lowest_similarity2,random_similarity2,selection_number)
df = numpy.size(t_S)
tvalue_SD, pvalue_D = stats.ttest_ind(t_S,t_D, equal_var = False, nan_policy='raise')
tvalue_SR, pvalue_R = stats.ttest_ind(t_S,t_R, equal_var = False, nan_policy='raise')
################################################################################ plotting 
ax = plt.subplot(1,2,1 )
plt.hist(t_R,bins=n_bins,alpha=1, color='k',histtype='step')
plt.hist(t_R,bins=n_bins,alpha=0.2, color='k',label= '5 random utterances (R)')
plt.hist(t_D,bins=n_bins,alpha=0.3, color='r',label='5 furthest utterances (D)') # histtype='stepfilled'
plt.hist(t_S,bins=n_bins,alpha=0.4, color='green',label='5 nearest utterances (S)')

formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3,2))
ax.yaxis.set_major_formatter(formatter)


plt.ylabel('\n' + model + '\ncounts\n',fontsize=30)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(prop={"size":15})
text_SD = ' S,D : t(' + str(df) + ') = ' + str("%.2f" %tvalue_SD) +' , p < 0.001'
text_SR = ' S,R : t(' + str(df) + ') = ' + str("%.2f" %tvalue_SR) +' , p < 0.001'
plt.text(0.4, 2600, text_SD, fontsize=20, fontstyle='italic')
plt.text(0.4, 2300, text_SR, fontsize=20, fontstyle='italic')
plt.grid()
plt.tight_layout() 
############################################################################### # Similarity 1   
t_S,t_D,t_R = seleting_tops(top_similarity1,lowest_similarity1,random_similarity1,selection_number)
df = numpy.size(t_S)
tvalue_SD, pvalue_D = stats.ttest_ind(t_S,t_D, equal_var = False, nan_policy='raise')
tvalue_SR, pvalue_R = stats.ttest_ind(t_S,t_R, equal_var = False, nan_policy='raise')
################################################################################ plotting 

ax = plt.subplot(1,2,2 )
plt.hist(t_R,bins=n_bins,alpha=1, color='k',histtype='step')
plt.hist(t_R,bins=n_bins,alpha=0.2, color='k',label= '5 random utterances (R)')
plt.hist(t_D,bins=n_bins,alpha=0.3, color='r',label='5 furthest utterances (D)') # histtype='stepfilled'
plt.hist(t_S,bins=n_bins,alpha=0.4, color='green',label='5 nearest utterances (S)')

formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3,2))
ax.yaxis.set_major_formatter(formatter)


#plt.ylabel("counts",fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(prop={"size":15})
text_SD = ' S,D : t(' + str(df) + ') = ' + str("%.2f" %tvalue_SD) +' , p < 0.001'
text_SR = ' S,R : t(' + str(df) + ') = ' + str("%.2f" %tvalue_SR) +' , p < 0.001'
plt.text(0.4, 2100, text_SD, fontsize=20, fontstyle='italic')
plt.text(0.4, 1800, text_SR, fontsize=20, fontstyle='italic')
plt.grid()
plt.tight_layout() 
###############################################################################

###############################################################################
plt.savefig(path_out  + file_out  + '.png', format='png')   