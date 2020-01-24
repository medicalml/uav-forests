from osgeo import gdal
rgb_tif = gdal.Open('/media/piotr/KINGSTON/Zagan/RGB_Zagan.tif')
infrared_tif = gdal.Open('/media/piotr/KINGSTON/Zagan/NIR_Zagan.tif')


def pixel2coord(tif, col, row):
    c, a, b, f, d, e = tif.GetGeoTransform()

    """Returns global coordinates to pixel center using base-0 raster index"""
    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return(xp, yp)


if __name__ == '__main__':
    print(f"Coordinates RGB: {pixel2coord(rgb_tif, 10, 22)}")

    print(f"Coordinates Infrared: {pixel2coord(infrared_tif, 10, 22)}")