from osgeo import gdal

driver = gdal.GetDriverByName('GTiff')
# filename = '/media/piotr/KINGSTON/Zagan/RGB_Zagan.tif'
filename = r"/media/piotr/824F-8A2A/Zagan/RGB_Zagan.tif"




def get_pixel_based_on_lon_lat(tif, points_list):
    band = tif.GetRasterBand(1)

    cols = tif.RasterXSize
    rows = tif.RasterYSize

    transform = tif.GetGeoTransform()

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    data = band.ReadAsArray(0, 0, cols, rows)

    for point in points_list:
        col = int((point[0] - xOrigin) / pixelWidth)
        row = int((yOrigin - point[1]) / pixelHeight)

        print(row, col, data[row][col])


if __name__ == '__main__':
    tif = gdal.Open(filename)
    points_list = [(238598.6793652528, 415716.2769632211)]

    # points_list = [(232329.72629500003, 416042.802135)]  # list of X,Y coordinates

    get_pixel_based_on_lon_lat(tif, points_list)