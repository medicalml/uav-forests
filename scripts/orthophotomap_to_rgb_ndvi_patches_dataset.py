import argparse
import rasterio as rio
import rasterio.transform, rasterio.warp, rasterio.windows
import geopandas as gpd
import shapely as shp
import numpy as np
import tqdm
import cv2
import skimage as ski
import skimage.io
import os
import multiprocessing
import copy
import shutil
from skimage.external.tifffile import imsave
import warnings
warnings.filterwarnings('error')
import sys
from PIL import Image

sys.path.insert(0, '/home/h/uav-forests')
from src.utils.infrared import nir_to_ndvi
from src.utils.coordinates_converters import coordinates_to_window

def geometry_to_pixel_geometry(geometry, transform):
    x, y = geometry.exterior.coords.xy
    x_pix, y_pix = rio.transform.rowcol(transform, x, y)
    pix_geom = shp.geometry.Polygon(zip(x_pix, y_pix))
    return pix_geom

def load_shapes_df(shapefile_path, transform):
    print("geopandas reads shapefile, ", shapefile_path)
    shapes_df = gpd.read_file(shapefile_path)
    print("file readed")
    shapes_df["pixel_geometry"] = shapes_df["geometry"].apply(lambda g: geometry_to_pixel_geometry(g, transform))
    print("setting geometry")
    shapes_df = shapes_df.set_geometry("pixel_geometry")
    shapes_df['pixel_bbox'] = shapes_df["pixel_geometry"].envelope
    shapes_df = shapes_df.merge(shapes_df["pixel_geometry"].bounds.astype(int), left_index=True, right_index=True)
    return shapes_df

def extract_tile(tiff_handler, shapes_df, row_offset, col_offset, tile_size):
    base_window_polygon = shp.geometry.Polygon([[0,0], [0,tile_size], [tile_size,tile_size], [tile_size,0]])
    window_polygon = shp.affinity.translate(base_window_polygon, row_offset, col_offset)
    
    read_window =rio.windows.Window(col_offset, row_offset, tile_size, tile_size)

   
    
    some_window = None
    
    while some_window is None:
        try:
            some_window = tiff_handler.read(window=read_window)
        except rasterio.errors.RasterioIOError as e:
            pass #print("error")
    
    tile = some_window.transpose(1,2,0).copy()
    shapes = shapes_df[~shapes_df["pixel_geometry"].intersection(window_polygon).is_empty]
    result_shapes = []
    for shape in shapes.itertuples():
        translated_shape = shp.affinity.translate(shape.pixel_geometry, -row_offset, -col_offset)
        cut_shape = translated_shape.intersection(base_window_polygon)
        bbox = cut_shape.envelope
        
        result_shapes.append({"global_geo_shape": shape.geometry,
                              "global_pixel_shape": shape.pixel_geometry,
                              "local_pixel_shape": translated_shape,
                              "cut_local_pixel_shape": cut_shape, 
                              "bbox": bbox})

    return tile, result_shapes

def extract_ir_tile(ir_handler, row_offset, col_offset, rgb_tile_size, transform_rgb):
    x0, y0 = rio.transform.xy(transform_rgb, row_offset, col_offset)
    x1, y1 = rio.transform.xy(transform_rgb, row_offset+rgb_tile_size, col_offset+rgb_tile_size)
    ir_row_min, ir_col_min = rio.transform.rowcol(ir_handler.transform, x0, y0)
    ir_row_max, ir_col_max = rio.transform.rowcol(ir_handler.transform, x1, y1)
    ir_row_min, ir_col_min = [0 if dim<0 else dim for dim in [ir_row_min, ir_col_min]]
    x_ir_tile_size = ir_col_max - ir_col_min
    y_ir_tile_size = ir_row_max-ir_row_min
    base_window_polygon = shp.geometry.Polygon([[0,0], [0,x_ir_tile_size ], [y_ir_tile_size, x_ir_tile_size], [y_ir_tile_size,0]])
    window_polygon = shp.affinity.translate(base_window_polygon, ir_row_min, ir_col_min)
    if x_ir_tile_size <= 20 or y_ir_tile_size <= 20:
        return None
    read_window =rio.windows.Window(ir_col_min, ir_row_min, x_ir_tile_size, y_ir_tile_size )

    some_window = None
    
    while some_window is None:
        try:
            some_window = ir_handler.read(window=read_window)
        except rasterio.errors.RasterioIOError as e:
            pass #print("error")
    
    tile = some_window.transpose(1,2,0).copy()

    return tile
    
def write(colrange, row_range, rgb_tif_handler, nir_tif_handler, shapes_df,  tile_size, max_empty_pixels_threshold, target_dir):
    annotations = []
    for row_index, row in tqdm.tqdm(enumerate(row_range), total=len(row_range)):
        for col_nr, col in enumerate(colrange):
            index = row_index*len(colrange)+col_nr
            
            tile_rgb, shapes_rgb = extract_tile(rgb_tif_handler, shapes_df, row, col, tile_size)
            alpha = tile_rgb[:,:,3].astype(np.float32)/255
            x_min, y_min = rio.transform.xy(rgb_tif_handler.transform, row, col)
            x_max, y_max = rio.transform.xy(rgb_tif_handler.transform, row+tile_size, col+tile_size)
            
            new_transform = copy.deepcopy(rgb_tif_handler.transform)
            #new_transform[0] = x_min
            #new_transform[3] = y_min
            
            nir_win = coordinates_to_window(nir_tif_handler,
                                        x_min, y_min, x_max, y_max)
            
            tile_r = tile_rgb[:,:,0]
            try:
                nir_img = nir_tif_handler.read(1, window=nir_win,
                                            out_shape=tile_r.shape)
            except rasterio.errors.RasterioIOError as e:
                print("rio error")
                continue
            #print("min nir", np.min(nir_img), "max nir_img", np.max(nir_img))
            ndvi = nir_to_ndvi(nir_img, tile_r)
            if not isinstance(ndvi, (np.ndarray, np.array, np.generic)):
                print("nie instancja")
                continue
            if ndvi.size == 0:
                print("zero size")
                continue
            ndvi = np.clip(ndvi, -1, 1)
            
            tile_rgb = tile_rgb.astype(np.float32)
            
            if len(shapes_rgb) > 0 or alpha.mean() > max_empty_pixels_threshold:
                print("zapisuje")
                #imb.save(f"{target_dir}/patch_{index}.tif", format="tiff", append_images=[img, imr, imn], save_all=True, quality=100, metadata={'axes': 'BGRN'})
                with rasterio.open(f"{target_dir}/patch_{index}.tif",'w',driver='GTiff',height=tile_rgb.shape[0],width=tile_rgb.shape[1],count=1,dtype=ndvi.dtype,crs='+proj=latlong',
                 transform=new_transform) as dst:
                    #dst.write(tile_rgb[:,:,0], 4)
                    #dst.write(tile_rgb[:,:,1], 2)
                    #dst.write(tile_rgb[:,:,2], 3)
                    dst.write(ndvi, 1)
                annotations += [{"patch_number": index, **s} for s in shapes_rgb]
            else:
                print("brak shapes albo obszar nieznany")
            del tile_rgb
           
       

    return annotations

def my_progressbar(task_number, done_indicator):
    counter = 0
    pbar = tqdm.tqdm(total=task_number)
    while counter != task_number:
        done_indicator.get()
        pbar.update(1)
        counter += 1

    pbar.close()

def rolling_window(tiff_handler, ir, shapes_df, target_dir,
                   min_row, max_row, min_col, max_col, 
                   tile_size, step, 
                   max_empty_pixels_threshold,
                   progressbar):
    
    annotations = []
    rowrange = range(min_row, max_row + 1 - step, step)
    colrange = range(min_col, max_col + 1 - step, step)
    
    annotations = write(colrange, rowrange, tiff_handler, ir, shapes_df, tile_size, max_empty_pixels_threshold, target_dir)

    annotation = gpd.GeoDataFrame(annotations)
    annotation.to_pickle(f"{target_dir}/annotation.pkl")
    annotation.to_csv(f"{target_dir}/annotation.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="orthophotomap_to_patches_dataset.py",
                                     description=("Extract tiles from orthophotomap.\nExample command: \n"
                                                  + " "*4
                                                  + "python orthophotomap_to_patches_dataset.py \\\n"
                                                  + " "*9
                                                  + "  --geotiff file.tiff --shapefile file.shp \\\n"
                                                  + " "*9
                                                  + "  --tile-size 256 --step 128 \\\n"
                                                  + " "*9
                                                  + "  --empty-pixels-threshold 0.5 \\\n"
                                                  + " "*9
                                                  + "  --target-dir ./target_dataset/ \n"),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--geotiff", required=True, help="path to orthophotomap file")
    parser.add_argument("--nirtiff", required=True, help="path to NIR orthophotomap file")
    parser.add_argument("--shapefile", required=True, help="path to shapefile annotation file")
    parser.add_argument("--tile-size", required=True, type=int, help="size of a tile (in pixels)")
    parser.add_argument("--step", required=True, type=int, help="step size for sliding window")
    parser.add_argument("--min-row", type=int, default=0, help="optional: row offset for sliding window")
    parser.add_argument("--max-row", type=int, default=-1, help="optional: max pixel row for sliding window")
    parser.add_argument("--min-col", type=int, default=0, help="optional: col offset for sliding window")
    parser.add_argument("--max-col", type=int, default=-1, help="optional: max pixel col for sliding window")
    parser.add_argument("--empty-pixels-threshold", type=float, default=0.5,
                        help="threshold of max percentage of empty pixels on a tile to use it")
    parser.add_argument("--target-dir", required=True, help="directory to store dataset")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False,
                        help="whether to be verbose")
    
    args = parser.parse_args()
       
    with rio.open(args.geotiff) as geotiff:
        with rio.open(args.nirtiff) as nirtiff:
            print("creating out image which will contain ndvi")
            #image_out = geotiff
            if not os.path.exists(args.target_dir):
                os.makedirs(args.target_dir)
            else: 
                shutil.rmtree(args.target_dir)
                os.makedirs(args.target_dir)
            print("ustanowiono foldery")
            img_shape = geotiff.shape
            print("ładowanie shapeow")
            shapes_df = load_shapes_df(args.shapefile, geotiff.transform)
            print("SHAPE", img_shape)
            rolling_window(geotiff, nirtiff, shapes_df, args.target_dir,
                        args.min_row, args.max_row if args.max_row >=0 else img_shape[0], 
                        args.min_col, args.max_col if args.max_col >=0 else img_shape[1],
                        args.tile_size, args.step, args.empty_pixels_threshold,
                        args.verbose)
        