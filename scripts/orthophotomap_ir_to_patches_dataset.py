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
def geometry_to_pixel_geometry(geometry, transform):
    x, y = geometry.exterior.coords.xy
    x_pix, y_pix = rio.transform.rowcol(transform, x, y)
    pix_geom = shp.geometry.Polygon(zip(x_pix, y_pix))
    return pix_geom

def load_shapes_df(shapefile_path, transform):
    shapes_df = gpd.read_file(shapefile_path)
    shapes_df["pixel_geometry"] = shapes_df["geometry"].apply(lambda g: geometry_to_pixel_geometry(g, transform))
    shapes_df = shapes_df.set_geometry("pixel_geometry")
    shapes_df['pixel_bbox'] = shapes_df["pixel_geometry"].envelope
    shapes_df = shapes_df.merge(shapes_df["pixel_geometry"].bounds.astype(int), left_index=True, right_index=True)
    return shapes_df

def extract_tile(tiff_handler, shapes_df, row_offset, col_offset, tile_size, lock):
    base_window_polygon = shp.geometry.Polygon([[0,0], [0,tile_size], [tile_size,tile_size], [tile_size,0]])
    window_polygon = shp.affinity.translate(base_window_polygon, row_offset, col_offset)
    
    read_window =rio.windows.Window(col_offset, row_offset, tile_size, tile_size)

    #lock.acquire()
    
    some_window = None
    with lock:
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

    #print(ir_row_min, ir_col_min, ir_row_max, ir_col_max)
    ir_tile_size = min(ir_col_max - ir_col_min, ir_row_max-ir_row_min)
    #print(ir_row_min, ir_col_min, ir_tile_size)
    base_window_polygon = shp.geometry.Polygon([[0,0], [0,ir_tile_size ], [ir_tile_size, ir_tile_size], [ir_tile_size,0]])
    window_polygon = shp.affinity.translate(base_window_polygon, ir_row_min, ir_col_min)
    if ir_col_min <= 0 or ir_row_min <= 0:
        return None
    read_window =rio.windows.Window(ir_col_min, ir_row_min, ir_tile_size, ir_tile_size )

    #lock.acquire()
    
    some_window = None
    
    while some_window is None:
        try:
            some_window = ir_handler.read(window=read_window)
        except rasterio.errors.RasterioIOError as e:
            pass #print("error")
    
    tile = some_window.transpose(1,2,0).copy()

    return tile
    
def write(colrange, tiff_handler, ir, shapes_df, rows_and_indexes, tile_size, max_empty_pixels_threshold, target_dir, return_dict, lock, cpu_index, done_indicator):
    annotations = []
    for row_index, row in rows_and_indexes:
        for col_nr, col in enumerate(colrange):
            index = row_index*len(colrange)+col_nr
            
            tile_rgb, shapes_rgb = extract_tile(tiff_handler, shapes_df, row, col, tile_size, lock)
            tile_ir = extract_ir_tile(ir, row, col, tile_size, tiff_handler.transform)
            if tile_ir is None or tile_ir.size <= 0:
                continue
            #second_lock.acquire()
            #print(tile_ir.shape)
            alpha = tile_rgb[:,:,3].astype(np.float32)/255
            #second_lock.release()
            tile_rgb = cv2.cvtColor(tile_rgb, cv2.COLOR_RGBA2BGRA)
            
            #print("raw", np.unique(tile_ir))
            tile_ir = cv2.resize(tile_ir, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
            tile_ndvi = nir_to_ndvi(tile_ir, tile_rgb[:,:,0])
            if tile_ndvi is None:
                continue
            tile_ndvi = tile_ndvi * 255.0
            tile_ndvi = np.clip(tile_ndvi, 0, 255)
            #tile_ndvi = tile_ndvi.astype(np.uint8)
            #print("scaled", np.unique(tile_ir))
            
            tile_ndvi = np.expand_dims(tile_ndvi, -1)
            
            #print(tile_rgb.shape, tile_ndvi.shape)
            tile_rgb = tile_rgb.astype(np.uint8)
            tile_ndvi = tile_ndvi*255.0
            tile_ndvi = tile_ndvi.astype(np.uint8)
            tile_merged = np.append(tile_rgb, tile_ndvi, axis=-1)
            
            #print(tile_merged.shape)
            #print(tile_rgb.dtype)
            
            imr = Image.fromarray(tile_merged[:,:,0]) 
            img = Image.fromarray(tile_merged[:,:,1])
            imb = Image.fromarray(tile_merged[:,:,2]) 
            imn = Image.fromarray(tile_merged[:,:,4]) 
            
            #print(tile_rgb_ir.shape)
            if len(shapes_rgb) > 0 or alpha.mean() > max_empty_pixels_threshold:
                #cv2.imwrite(f"{target_dir}/patch_{index}.png", tile_rgb)
                #cv2.imwrite(f"{target_dir}/patch_ir_{index}.png", tile_ndvi)
                #im.save(f"{target_dir}/patch_ir_{index}.tiff", "TIFF")
                imr.save(f"{target_dir}/patch_{index}.tif", format="tiff", append_images=[img, imb, imn], save_all=True)
                annotations += [{"patch_number": index, **s} for s in shapes_rgb]

            del tile_ir
            del tile_rgb
           
        done_indicator.put(1)

    return_dict[cpu_index] = annotations

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
                   progressbar, single_process):
    
    annotations = []
    rowrange = range(min_row, max_row + 1 - step, step)
    colrange = range(min_col, max_col + 1 - step, step)
    
    manager = multiprocessing.Manager()
    
    return_dict = manager.dict()
    
    done_indicator = multiprocessing.Queue()
    jobs = []
    rows_and_indexes = {}
    if single_process:
        rows_per_cpu = len(rowrange)
    else:
        rows_per_cpu = len(rowrange)//multiprocessing.cpu_count()
    
    if rows_per_cpu == 0:
        rows_per_cpu = 1
    
    lock = multiprocessing.RLock()

    for row_index, row in enumerate(rowrange):
        cpu_index = row_index//rows_per_cpu
        
        if cpu_index not in rows_and_indexes.keys():
            rows_and_indexes[cpu_index] = []
        
        rows_and_indexes[cpu_index] += [(row_index, row)]

    for cpu_index in rows_and_indexes.keys():
        p = multiprocessing.Process(target=write, args=(colrange, tiff_handler, ir, shapes_df, rows_and_indexes[cpu_index], tile_size, max_empty_pixels_threshold, target_dir, return_dict, lock, cpu_index, done_indicator))
        jobs.append(p)
        p.start()
    
    if progressbar:
        p = multiprocessing.Process(target=my_progressbar, args=(len(rowrange), done_indicator))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    for cpu_index in sorted(return_dict.keys()):
        annotations += return_dict[cpu_index]

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
    
    parser.add_argument("--single-process", dest="single_process", action="store_true", default=False,
                        help="whether use only one cpu")
    
    args = parser.parse_args()
    ir = rio.open('/home/h/_drzewaBZBUAS/Szprotawa_translated_EVI.tif')
    with rio.open(args.geotiff) as geotiff:
        
        if not os.path.exists(args.target_dir):
            os.makedirs(args.target_dir)
        else: 
            shutil.rmtree(args.target_dir)
            os.makedirs(args.target_dir)

        img_shape = geotiff.shape
        shapes_df = load_shapes_df(args.shapefile, geotiff.transform)
        print("SHAPE", img_shape)
        print(ir.shape)
        rolling_window(geotiff, ir, shapes_df, args.target_dir,
                       args.min_row, args.max_row if args.max_row >=0 else img_shape[0], 
                       args.min_col, args.max_col if args.max_col >=0 else img_shape[1],
                       args.tile_size, args.step, args.empty_pixels_threshold,
                       args.verbose, args.single_process)
        