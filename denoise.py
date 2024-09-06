import numpy as np 
from scipy import signal 
import matplotlib.pyplot as plt 
import rasterio 
import pandas as pd  
from skimage.filters import threshold_otsu
import glob
from sklearn.linear_model import Lasso, Ridge 
import re 
from scipy.ndimage import gaussian_filter1d

# ···这是for VNIR的···

############################################################################################################
def get_the_mean_value(link):
    all_image_path = glob.glob(link)
    save_df = pd.DataFrame(index = range(len(all_image_path)), columns = ['link'] + list(range(1,344)))
    for j in range(0, len(all_image_path)):
        save_df['link'][j] = all_image_path[j]
        img = rasterio.open(all_image_path[j])
        binary_array = img.read(229) > 0.15     # this is 800 nm
        mean_value = []
    
        for i in range(1, img.count + 1):
            channel_data = img.read(i)
            vegetation = channel_data[binary_array]
            mean_vegetation = vegetation.mean()
            mean_value.append(mean_vegetation)
        save_df.loc[j,:][1:344] = mean_value
        plt.figure(figsize = (12,9))
        plt.plot(save_df.loc[j,:][1:344], label = "")
        plt.savefig(all_image_path[j].split('tif')[0] + "png")
    
    all_numbers = []
    for links in save_df.link:
        numbers = re.findall(r'\d+', links)
        all_numbers.append(numbers[2])
    save_df.link = all_numbers 
    save_df.to_csv(link.split('*.tif')[0] + "raw_mean_VNIR.csv")
    return(save_df)


get_the_mean_value(r"C:\Users\ft7b6\Desktop\hyperspectral_code_file\segmented_VNIR\*.tif")

#########################################################################################################
##############################################################################################################

# remove noise from the small plot 0711_data
# VNIR  
# 记得手动把column标题改成spectrum

spectrum_name = pd.read_csv(r"C:\Users\ft7b6\Desktop\hyperspectral_code_file\VNIR_SWIR_Spectrum_Band.csv")
ground_truth_N = pd.read_csv(r"C:/Users/ft7b6/Desktop/hyperspectral_code_file/corn_leaf_sample.csv")
VNIR = pd.read_csv(r'C:/Users/ft7b6/Desktop/hyperspectral_code_file/segmented_VNIR/raw_mean_VNIR.csv')
VNIR = VNIR.sort_values(by = 'link', ascending = True).reset_index().drop(['Unnamed: 0', 'index'], axis = 1)    
VNIR.columns.values[1:344] = spectrum_name.VNIR_Spectrum
ground_truth_N = ground_truth_N.loc[:,["link", "TN"]].dropna()
ground_truth_N["link"] = ground_truth_N["link"].astype(int) 
VNIR = pd.merge(VNIR, ground_truth_N, on = 'link').drop('link', axis = 1)


# 只筛选出spectrum value only  检查下数据
ground_truth = VNIR.iloc[:,343] 
VNIR = VNIR.iloc[:,range(343)]
print(VNIR.loc[:, (VNIR > 1).any() | (VNIR < 0).any()].columns)  # 这个会告诉你哪个spectrum不对劲
delete_column = VNIR.loc[:, (VNIR > 1).any() | (VNIR < 0).any() | (VNIR == 0).any()].columns
VNIR = VNIR.drop(delete_column, axis = 1) 
# 开始储存filter之后的了
VNIR_cleaned_save = pd.DataFrame(np.nan, index=VNIR.index, columns=VNIR.columns)
for i in range(0, VNIR.shape[0]):
    filiter_spectrum = signal.savgol_filter(VNIR.iloc[i], window_length = 21, polyorder = 2, deriv = 2, mode = "interp")
    VNIR_cleaned_save.loc[i] = filiter_spectrum 
    
plt.figure(figsize = (12,9))
plt.plot(VNIR_cleaned_save.loc[0,:][0:343], label = "")  

VNIR_cleaned_save = pd.concat([VNIR_cleaned_save, ground_truth], axis=1)
VNIR_cleaned_save.to_csv(r'C:/Users/ft7b6/Desktop/hyperspectral_code_file/segmented_VNIR/filter_mean_VNIR_2st_order.csv')


#我求下corr
data = VNIR_cleaned_save.corr().TN.drop(["TN"])
data.index = data.index.astype(float)
plt.figure(figsize=(10, 6), dpi=1600)
plt.plot(data, linestyle='-')
plt.title('VNIR 2st order', fontsize=20)  
plt.xlabel('Spectrum', fontsize=16) 
plt.ylabel('Coefficient of Correlation', fontsize=16)
plt.axhline(0, color='lightcoral', linestyle='--', linewidth=1)  
plt.xticks(np.arange(400, 1001, 100))
plt.yticks(np.arange(-0.8, 1.0, 0.2)) 
plt.grid(True)
plt.show()


### 看下所有的spectrum分布
### 这个画spectrum分布要把删除的spectrum都归为NA
### 因为VNIR就删1，2个波段，而且有300多个波段，我就没进行特殊处理like SWIR

VNIR_cleaned_save[VNIR_cleaned_save < 0] = np.nan
VNIR_cleaned_save = VNIR_cleaned_save.drop(["TN"], axis = 1) 

plt.figure(dpi=600) 
for i, row in enumerate(VNIR.values):
    plt.plot(VNIR.columns, row, label=f'Row {i+1}')

   
plt.ylim(0, 0.6)
plt.xlabel('Spectrum')
plt.ylabel('Reflectance')
plt.show()



#######################################################################################
#######################################################################################

# ···这是for SWIR的···

def get_the_mean_value(link):
    all_image_path = glob.glob(link)
    save_df = pd.DataFrame(index = range(len(all_image_path)), columns = ['link'] + list(range(1,269)))
    for j in range(0, len(all_image_path)):
        save_df['link'][j] = all_image_path[j]
        img = rasterio.open(all_image_path[j])
        binary_array = (img.read(19) > 0.15) # this is larger than 0.15 on 1000 and smaller than 
        mean_value = []
     
        for i in range(1, img.count + 1):
            channel_data = img.read(i)
            vegetation = channel_data[binary_array]
            mean_vegetation = vegetation.mean()
            mean_value.append(mean_vegetation)
        save_df.loc[j,:][1:269] = mean_value
        plt.figure(figsize = (12,9))
        plt.plot(save_df.loc[j,:][1:269], label = "")
        plt.savefig(all_image_path[j].split('tif')[0] + "png")
        
    all_numbers = []
    for links in save_df.link:
        numbers = re.findall(r'\d+', links)
        all_numbers.append(numbers[2])
    save_df.link = all_numbers 
    save_df.to_csv(link.split('*.tif')[0] + "raw_mean_SWIR.csv")
    return(save_df)    

get_the_mean_value(r"C:\Users\ft7b6\Desktop\hyperspectral_code_file\segmented_SWIR\*.tif")

# remove noise from the small plot 0711_data
# SWIR 
# 记得手动把column标题改成spectrum
SWIR = pd.read_csv(r'C:/Users/ft7b6/Desktop/hyperspectral_code_file/segmented_SWIR/raw_mean_SWIR.csv')
spectrum_name = pd.read_csv(r"C:\Users\ft7b6\Desktop\hyperspectral_code_file\VNIR_SWIR_Spectrum_Band.csv")
ground_truth_N = pd.read_csv(r"C:/Users/ft7b6/Desktop/hyperspectral_code_file/corn_leaf_sample.csv")
SWIR = pd.read_csv(r'C:/Users/ft7b6/Desktop/hyperspectral_code_file/segmented_SWIR/raw_mean_SWIR.csv')

SWIR = SWIR.sort_values(by = 'link', ascending = True).reset_index().drop(['Unnamed: 0', 'index'], axis = 1)    
SWIR.columns.values[1:269] = spectrum_name.SWIR_Spectrum.dropna()
ground_truth_N = ground_truth_N.loc[:,["link", "TN"]].dropna()
ground_truth_N["link"] = ground_truth_N["link"].astype(int) 
SWIR = pd.merge(SWIR, ground_truth_N, on = 'link').drop('link', axis = 1)

# 只筛选出spectrum value only  检查下数据
ground_truth = SWIR.iloc[:,268]
SWIR = SWIR.iloc[:,range(268)]
print(SWIR.loc[:, (SWIR  > 1).any() | (SWIR  == 0).any()].columns)  # 这个会告诉你哪个spectrum不对劲

delete_column = SWIR.loc[:, (SWIR > 1).any() | (SWIR < 0).any() | (SWIR == 0).any()].columns
print([SWIR.columns.get_loc(col) for col in delete_column]) # SWIR比较复杂，根据这个输出来去手动删除

# 手动拆分 ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
SWIR1 = SWIR.iloc[:,0:75];    SWIR_1_zero = SWIR.iloc[:, 75:92]
SWIR2 = SWIR.iloc[:,92:150];   SWIR_2_zero = SWIR.iloc[:, 150:178]
SWIR3 = SWIR.iloc[:,178:260];   SWIR_3_zero = SWIR.iloc[:, 260:269]

# 开始储存filter之后的了
SWIR_cleaned_save1 = pd.DataFrame(np.nan, index=SWIR1.index, columns=SWIR1.columns)
for i in range(0, SWIR1.shape[0]):
    filiter_spectrum = signal.savgol_filter(SWIR1.iloc[i], window_length = 7, polyorder = 2, mode = "interp")
    SWIR_cleaned_save1.loc[i] = filiter_spectrum 

SWIR_cleaned_save2 = pd.DataFrame(np.nan, index=SWIR2.index, columns=SWIR2.columns)
for i in range(0, SWIR2.shape[0]):
    filiter_spectrum = signal.savgol_filter(SWIR2.iloc[i], window_length = 7, polyorder = 2, mode = "interp")
    SWIR_cleaned_save2.loc[i] = filiter_spectrum 
    
SWIR_cleaned_save3 = pd.DataFrame(np.nan, index=SWIR3.index, columns=SWIR3.columns)
for i in range(0, SWIR3.shape[0]):
    filiter_spectrum = signal.savgol_filter(SWIR3.iloc[i], window_length = 7, polyorder = 2, mode = "interp")
    SWIR_cleaned_save3.loc[i] = filiter_spectrum  
    
###########################################################################################################

# 这个是画图专用的
SWIR_cleaned_show = pd.concat([SWIR_cleaned_save1, SWIR_1_zero, SWIR_cleaned_save2, SWIR_2_zero, SWIR_cleaned_save3, SWIR_3_zero, ground_truth], axis=1)
# 这个是跑机器学习专用
SWIR_cleaned_ML = pd.concat([SWIR_cleaned_save1, SWIR_cleaned_save2, SWIR_cleaned_save3, ground_truth], axis=1)
SWIR_cleaned_ML.to_csv(r'C:/Users/ft7b6/Desktop/hyperspectral_code_file/segmented_SWIR/filter_mean_SWIR_2st_order.csv')

#我求下corr
data = SWIR_cleaned_show.corr().TN.drop(["TN"])
data.index = data.index.astype(float)
plt.figure(figsize=(10, 6), dpi=1600)
plt.plot(data, linestyle='-')
plt.title('SWIR 2st order', fontsize=20)    
plt.xlabel('Spectrum', fontsize=16) 
plt.ylabel('Coefficient of Correlation', fontsize=16) 
plt.axhline(0, color='lightcoral', linestyle='--', linewidth=1) 
plt.xticks(np.arange(900, 2501, 200))  
plt.yticks(np.arange(-0.8, 1.0, 0.2)) 
plt.grid(True)
plt.show()

### 看下所有的spectrum分布
### 这个画spectrum分布要把删除的spectrum都归为0

SWIR_1_zero[:] = np.nan
SWIR_2_zero[:] = np.nan
SWIR_3_zero[:] = np.nan
SWIR_cleaned_spectrum_show = pd.concat([SWIR_cleaned_save1, SWIR_1_zero, SWIR_cleaned_save2, SWIR_2_zero, SWIR_cleaned_save3, SWIR_3_zero], axis=1)
SWIR_cleaned_spectrum_show[SWIR_cleaned_spectrum_show < 0] = np.nan

plt.figure(dpi=600) 
for i, row in enumerate(SWIR_cleaned_spectrum_show.values):
    plt.plot(SWIR_cleaned_spectrum_show.columns, row, label=f'Row {i+1}')
plt.ylim(0, 0.6)
plt.xlabel('Spectrum')
plt.ylabel('Reflectance')
plt.show()
#############################################################################################
#########################################画distribution图

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/ft7b6/Desktop/hyperspectral_code_file/corn_leaf_sample.csv")
filtered_data = data[data['nitrogen'].isin([1, 2])]
sns.set_style("whitegrid")
plt.figure(dpi=1000) 
hist = sns.histplot(data=filtered_data, x='TN', hue='nitrogen', element='bars', stat='density', kde=False, palette='coolwarm')
kde = sns.kdeplot(data=filtered_data, x='TN', hue='nitrogen', common_norm=False, palette='coolwarm', linewidth=2, alpha=0.7)
plt.title('Distribution of LNC', fontsize=16)
plt.xlabel('LNC (%)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='N timing', title_fontsize='13', labels=['Class B', 'Class A'], fontsize='12')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

######################################################画散点图

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/ft7b6/Desktop/hyperspectral_code_file/corn_leaf_sample.csv")
filtered_data = data.loc[:,['TN','nitrogen']].iloc[0:48]
filtered_data.nitrogen = filtered_data.nitrogen.astype(int) 

plt.figure(dpi=600) 
ax = plt.gca()  
for key, grp in filtered_data.groupby(['nitrogen']):
    ax.scatter(grp['nitrogen'], grp['TN'], label=f'Nitrogen {key}', s = 20)
ax.set_yticks([1.5, 2, 2.5, 3, 3.5])
ax.set_xlabel('N Treatment', fontsize=12)  # It should be ax.set_xlabel not plt.set_xlabel
ax.set_ylabel('LNC(%)', fontsize=12)
plt.title('Distribution of LNC', fontsize=16)
plt.show()








高斯去噪
本质和上面代码一样，就算法一行变了
###############################################################################################################################  


# remove noise from the small plot 0711_data
# VNIR  
# 记得手动把column标题改成spectrum

spectrum_name = pd.read_csv(r"C:\Users\ft7b6\Desktop\hyperspectral_code_file\VNIR_SWIR_Spectrum_Band.csv")
ground_truth_N = pd.read_csv(r"C:/Users/ft7b6/Desktop/hyperspectral_code_file/corn_leaf_sample.csv")
VNIR = pd.read_csv(r'C:/Users/ft7b6/Desktop/hyperspectral_code_file/segmented_VNIR/raw_mean_VNIR.csv')
VNIR = VNIR.sort_values(by = 'link', ascending = True).reset_index().drop(['Unnamed: 0', 'index'], axis = 1)    
VNIR.columns.values[1:344] = spectrum_name.VNIR_Spectrum
ground_truth_N = ground_truth_N.loc[:,["link", "TN"]].dropna()
ground_truth_N["link"] = ground_truth_N["link"].astype(int) 
VNIR = pd.merge(VNIR, ground_truth_N, on = 'link').drop('link', axis = 1)


# 只筛选出spectrum value only  检查下数据
ground_truth = VNIR.iloc[:,343] 
VNIR = VNIR.iloc[:,range(343)]
print(VNIR.loc[:, (VNIR > 1).any() | (VNIR < 0).any()].columns)  # 这个会告诉你哪个spectrum不对劲
delete_column = VNIR.loc[:, (VNIR > 1).any() | (VNIR < 0).any() | (VNIR == 0).any()].columns
VNIR = VNIR.drop(delete_column, axis = 1) 
# 开始储存filter之后的了
VNIR_cleaned_save = pd.DataFrame(np.nan, index=VNIR.index, columns=VNIR.columns)
for i in range(0, VNIR.shape[0]):
    filiter_spectrum = gaussian_filter1d(VNIR.iloc[i], sigma=3)
    VNIR_cleaned_save.loc[i] = filiter_spectrum 
    
plt.figure(figsize = (12,9))
plt.plot(VNIR_cleaned_save.loc[0,:][0:343], label = "")  

VNIR_cleaned_save = pd.concat([VNIR_cleaned_save, ground_truth], axis=1)
VNIR_cleaned_save.to_csv(r'C:/Users/ft7b6/Desktop/hyperspectral_code_file/segmented_VNIR/filter_mean_VNIR_Guassian.csv')














# remove noise from the small plot 0711_data
# SWIR 
# 记得手动把column标题改成spectrum
SWIR = pd.read_csv(r'C:/Users/ft7b6/Desktop/hyperspectral_code_file/segmented_SWIR/raw_mean_SWIR.csv')
spectrum_name = pd.read_csv(r"C:\Users\ft7b6\Desktop\hyperspectral_code_file\VNIR_SWIR_Spectrum_Band.csv")
ground_truth_N = pd.read_csv(r"C:/Users/ft7b6/Desktop/hyperspectral_code_file/corn_leaf_sample.csv")
SWIR = pd.read_csv(r'C:/Users/ft7b6/Desktop/hyperspectral_code_file/segmented_SWIR/raw_mean_SWIR.csv')

SWIR = SWIR.sort_values(by = 'link', ascending = True).reset_index().drop(['Unnamed: 0', 'index'], axis = 1)    
SWIR.columns.values[1:269] = spectrum_name.SWIR_Spectrum.dropna()
ground_truth_N = ground_truth_N.loc[:,["link", "TN"]].dropna()
ground_truth_N["link"] = ground_truth_N["link"].astype(int) 
SWIR = pd.merge(SWIR, ground_truth_N, on = 'link').drop('link', axis = 1)

# 只筛选出spectrum value only  检查下数据
ground_truth = SWIR.iloc[:,268]
SWIR = SWIR.iloc[:,range(268)]
print(SWIR.loc[:, (SWIR  > 1).any() | (SWIR  == 0).any()].columns)  # 这个会告诉你哪个spectrum不对劲

delete_column = SWIR.loc[:, (SWIR > 1).any() | (SWIR < 0).any() | (SWIR == 0).any()].columns
print([SWIR.columns.get_loc(col) for col in delete_column]) # SWIR比较复杂，根据这个输出来去手动删除

# 手动拆分 ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
SWIR1 = SWIR.iloc[:,0:75];    SWIR_1_zero = SWIR.iloc[:, 75:92]
SWIR2 = SWIR.iloc[:,92:150];   SWIR_2_zero = SWIR.iloc[:, 150:178]
SWIR3 = SWIR.iloc[:,178:260];   SWIR_3_zero = SWIR.iloc[:, 260:269]

# 开始储存filter之后的了
SWIR_cleaned_save1 = pd.DataFrame(np.nan, index=SWIR1.index, columns=SWIR1.columns)
for i in range(0, SWIR1.shape[0]):
    filiter_spectrum = gaussian_filter1d(SWIR1.iloc[i], sigma=2)
    SWIR_cleaned_save1.loc[i] = filiter_spectrum 

SWIR_cleaned_save2 = pd.DataFrame(np.nan, index=SWIR2.index, columns=SWIR2.columns)
for i in range(0, SWIR2.shape[0]):
    filiter_spectrum = gaussian_filter1d(SWIR2.iloc[i], sigma=2)
    SWIR_cleaned_save2.loc[i] = filiter_spectrum 
    
SWIR_cleaned_save3 = pd.DataFrame(np.nan, index=SWIR3.index, columns=SWIR3.columns)
for i in range(0, SWIR3.shape[0]):
    filiter_spectrum = gaussian_filter1d(SWIR3.iloc[i], sigma=2)
    SWIR_cleaned_save3.loc[i] = filiter_spectrum  
    
########################################################################################################### 

# 这个是画图专用的
SWIR_cleaned_show = pd.concat([SWIR_cleaned_save1, SWIR_1_zero, SWIR_cleaned_save2, SWIR_2_zero, SWIR_cleaned_save3, SWIR_3_zero, ground_truth], axis=1)
# 这个是跑机器学习专用
SWIR_cleaned_ML = pd.concat([SWIR_cleaned_save1, SWIR_cleaned_save2, SWIR_cleaned_save3, ground_truth], axis=1)
SWIR_cleaned_ML.to_csv(r'C:/Users/ft7b6/Desktop/hyperspectral_code_file/segmented_SWIR/filter_mean_SWIR_Guassian.csv')

#我求下corr
data = SWIR_cleaned_show.corr().TN.drop(["TN"])
data.index = data.index.astype(float)
plt.figure(figsize=(10, 6), dpi=1600)
plt.plot(data, linestyle='-')
plt.title('SWIR 2st order', fontsize=20)    
plt.xlabel('Spectrum', fontsize=16) 
plt.ylabel('Coefficient of Correlation', fontsize=16) 
plt.axhline(0, color='lightcoral', linestyle='--', linewidth=1) 
plt.xticks(np.arange(900, 2501, 200))  
plt.yticks(np.arange(-0.8, 1.0, 0.2)) 
plt.grid(True)
plt.show()

### 看下所有的spectrum分布
### 这个画spectrum分布要把删除的spectrum都归为0

SWIR_1_zero[:] = np.nan
SWIR_2_zero[:] = np.nan
SWIR_3_zero[:] = np.nan
SWIR_cleaned_spectrum_show = pd.concat([SWIR_cleaned_save1, SWIR_1_zero, SWIR_cleaned_save2, SWIR_2_zero, SWIR_cleaned_save3, SWIR_3_zero], axis=1)
SWIR_cleaned_spectrum_show[SWIR_cleaned_spectrum_show < 0] = np.nan

plt.figure(dpi=600) 
for i, row in enumerate(SWIR_cleaned_spectrum_show.values):
    plt.plot(SWIR_cleaned_spectrum_show.columns, row, label=f'Row {i+1}')
plt.ylim(0, 0.6)
plt.xlabel('Spectrum')
plt.ylabel('Reflectance')
plt.show()




































































































































