import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"F:\download\File\digits.csv")

print(df.info())
'''
RangeIndex: 1797 entries, 0 to 1796
Data columns (total 65 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   pixel_0_0     1797 non-null   float64
 1   pixel_0_1     1797 non-null   float64
 2   pixel_0_2     1797 non-null   float64
 3   pixel_0_3     1797 non-null   float64
 4   pixel_0_4     1797 non-null   float64
 5   pixel_0_5     1797 non-null   float64
 6   pixel_0_6     1797 non-null   float64
 7   pixel_0_7     1797 non-null   float64
 8   pixel_1_0     1797 non-null   float64
 9   pixel_1_1     1797 non-null   float64
 10  pixel_1_2     1797 non-null   float64
 11  pixel_1_3     1797 non-null   float64
 12  pixel_1_4     1797 non-null   float64
 13  pixel_1_5     1797 non-null   float64
 14  pixel_1_6     1797 non-null   float64
 15  pixel_1_7     1797 non-null   float64
 16  pixel_2_0     1797 non-null   float64
 17  pixel_2_1     1797 non-null   float64
 18  pixel_2_2     1797 non-null   float64
 19  pixel_2_3     1797 non-null   float64
 20  pixel_2_4     1797 non-null   float64
 21  pixel_2_5     1797 non-null   float64
 22  pixel_2_6     1797 non-null   float64
 23  pixel_2_7     1797 non-null   float64
 24  pixel_3_0     1797 non-null   float64
 25  pixel_3_1     1797 non-null   float64
 26  pixel_3_2     1797 non-null   float64
 27  pixel_3_3     1797 non-null   float64
 28  pixel_3_4     1797 non-null   float64
 29  pixel_3_5     1797 non-null   float64
 30  pixel_3_6     1797 non-null   float64
 31  pixel_3_7     1797 non-null   float64
 32  pixel_4_0     1797 non-null   float64
 33  pixel_4_1     1797 non-null   float64
 34  pixel_4_2     1797 non-null   float64
 35  pixel_4_3     1797 non-null   float64
 36  pixel_4_4     1797 non-null   float64
 37  pixel_4_5     1797 non-null   float64
 38  pixel_4_6     1797 non-null   float64
 39  pixel_4_7     1797 non-null   float64
 40  pixel_5_0     1797 non-null   float64
 41  pixel_5_1     1797 non-null   float64
 42  pixel_5_2     1797 non-null   float64
 43  pixel_5_3     1797 non-null   float64
 44  pixel_5_4     1797 non-null   float64
 45  pixel_5_5     1797 non-null   float64
 46  pixel_5_6     1797 non-null   float64
 47  pixel_5_7     1797 non-null   float64
 48  pixel_6_0     1797 non-null   float64
 49  pixel_6_1     1797 non-null   float64
 50  pixel_6_2     1797 non-null   float64
 51  pixel_6_3     1797 non-null   float64
 52  pixel_6_4     1797 non-null   float64
 53  pixel_6_5     1797 non-null   float64
 54  pixel_6_6     1797 non-null   float64
 55  pixel_6_7     1797 non-null   float64
 56  pixel_7_0     1797 non-null   float64
 57  pixel_7_1     1797 non-null   float64
 58  pixel_7_2     1797 non-null   float64
 59  pixel_7_3     1797 non-null   float64
 60  pixel_7_4     1797 non-null   float64
 61  pixel_7_5     1797 non-null   float64
 62  pixel_7_6     1797 non-null   float64
 63  pixel_7_7     1797 non-null   float64
 64  number_label  1797 non-null   int64
dtypes: float64(64), int64(1)
memory usage: 912.7 KB
None
'''

# Task: Create a new DataFrame called pixels that contains only pixel attribute values ​​and delete the number_label column.
pixels = pd.DataFrame(data=df.drop('number_label', axis=1), columns=df.drop('number_label', axis=1).columns)

# Task: Get a single image row representation by taking the first row from the DataFrame of pixels.
print(pixels[:1])
'''
   pixel_0_0  pixel_0_1  pixel_0_2  pixel_0_3  ...  pixel_7_4  pixel_7_5  pixel_7_6  pixel_7_7
0        0.0        0.0        5.0       13.0  ...       10.0        0.0        0.0        0.0 

[1 rows x 64 columns]
'''

# Task: Convert this single row series into a numpy array.
pixel_array = np.array(pixels[0:1])
print(pixel_array)
'''
[[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
  15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
   0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
   0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]]
'''

# Task: Use Matplotlib or Seaborn to display the array as a visual representation of the plotted number.
plt.figure(figsize=(8,5), dpi=100)
plt.imshow(pixels[0:1].to_numpy().reshape(8,8))
plt.show()
plt.imshow(pixels[0:1].to_numpy().reshape(8,8), cmap='gray')
plt.show()
sns.heatmap(pixels[0:1].to_numpy().reshape(8,8), annot=True, cmap='gray')
plt.show()

# Task: Use Scikit-Learn to scale a pixel feature DataFrame.
scale = StandardScaler()
scaled_pixels = scale.fit_transform(pixels)

# Task: Perform PCA on a 2-component scaled pixel dataset.
pca = PCA(n_components=2)
pca_pixels = pca.fit_transform(scaled_pixels)

# Task: How much variance is explained by the 2 principal components?
print(np.sum(pca.explained_variance_ratio_))  # 0.2159497050083281

# Task: Create a scatter plot of numbers in a two-dimensional PCA space.
# Set the label color based on the original number_label column in the original dataset.
sns.scatterplot(x=pca_pixels[:, 0], y=pca_pixels[:, 1],
                 hue=df['number_label'].values, palette='viridis')
plt.legend()
plt.xlabel("1st Principal Component")
plt.ylabel("2st Principal Component")
plt.show()

# Task: Create an interactive 3D plot of the PCA result with 3 principal components
pca_model = PCA(n_components=3)
pca_model_pixels = pca_model.fit_transform(scaled_pixels)

ax = plt.axes(projection='3d')
ax.scatter(xs = pca_model_pixels[:, 0],
           ys = pca_model_pixels[:, 1],
           zs = pca_model_pixels[:, 2],
           c=df['number_label']
           )
plt.show()
