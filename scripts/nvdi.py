import rasterio


with rasterio.open('0_rgb.tif') as src:

    print(src.bounds)

    x = (src.bounds.left + src.bounds.right) / 2.0
    y = (src.bounds.bottom + src.bounds.top) / 2.0

    print(x)
    print(y)

    vals = src.sample([(x, y)])
    for val in vals:
        print (list(val))