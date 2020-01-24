from osgeo import gdal

filename = r"/media/piotr/824F-8A2A/Lubsko/NIR_lubsko.tif"
input_raster = gdal.Open(filename)
output_raster = r"helper.tif"

gdal.Warp(output_raster,input_raster,dstSRS='EPSG:2180')
