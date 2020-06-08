## Readme 
### Description

In a satellite image, you will find lots of different objects like roads, buildings, vehicles, farms, trees, water ways, etc. Dstl has labeled 10 different classes:

- Buildings - large building, residential, non-residential, fuel storage facility, fortified building
- Misc. Manmade structures 
- Road 
- Track - poor/dirt/cart track, footpath/trail
- Trees - woodland, hedgerows, groups of trees, standalone trees
- Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
- Waterway 
- Standing water
- Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
- Vehicle Small - small vehicle (car, van), motorbike

#### DataSet 

- train_wkt.csv-所有训练标签的WKT格式
    - ImageId-图片的ID
    - ClassType-对象类型（1-10）
    - MultipolygonWKT-标记区域，以WKT格式表示的多面几何 

- three_band.zip  -3波段卫星图像的完整数据集。这三个带位于文件名= {ImageId} .tif的图像中。MD5 = 7cf7bf17ba3fa3198a401ef67f4ef9b4 

- sixteen_band.zip -16波段卫星图像的完整数据集。这16个波段分布在图像中，文件名= {ImageId} _ {A / M / P} .tif。MD5 = e2949f19a0d1102827fce35117c5f08a

- grid_sizes.csv- 所有图像的网格大小
    - ImageId-图片的ID
    - Xmax-图像的最大X坐标
    - Ymin-图像的最小Y坐标
    
- sample_submission.csv-格式正确的样本提交文件
    - ImageId-图片的ID
    - ClassType-对象类型（1-10）
    - MultipolygonWKT-标记区域，以WKT格式表示的多面几何

- train_geojson.zip-所有训练标签的geojson格式（本质上，这些信息与train_wkt.csv相同） 

#### adversarial_validation

Train a classifer to predict if train set and test set in the same dataset.
