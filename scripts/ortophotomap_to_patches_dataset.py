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

    lock.acquire()
    tile = tiff_handler.read(window=read_window).transpose(1,2,0).copy()
    lock.release()


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

def write(colrange, tiff_handler, shapes_df, row, tile_size, max_empty_pixels_threshold, target_dir, row_index, return_dict, lock):
    annotations = []
    pos = 0
    for col in tqdm.tqdm(colrange):
        #print(row_index, colrange, pos)
        index = row_index*len(colrange)+pos
        
        tile, shapes = extract_tile(tiff_handler, shapes_df, 
                                    row, col, tile_size, lock)
        
        alpha = tile[:,:,3].astype(np.float32)/255
        if len(shapes) > 0 or alpha.mean() > max_empty_pixels_threshold:
            print(index)
            cv2.imwrite(f"{target_dir}/patch_{index}.png", 
                        cv2.cvtColor(tile, cv2.COLOR_RGBA2BGRA))
            annotations += [{"patch_number": index, **s} for s in shapes]
            index += 1
        del tile
        pos+=1
    lock.acquire()
    return_dict[row_index] = annotations
    lock.release()

def rolling_window(tiff_handler, shapes_df, target_dir,
                   min_row, max_row, min_col, max_col, 
                   tile_size, step, 
                   max_empty_pixels_threshold=0.5,
                   progressbar=False):
    annotations = []
    rowrange = range(min_row, max_row + 1 - step, step)
    colrange = range(min_col, max_col + 1 - step, step)
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    lock = manager.Lock()
    jobs = []
    for row_index, row in enumerate(rowrange):
        #new_handler_tiff = copy.copy(tiff_handler)
        p = multiprocessing.Process(target=write, args=(colrange, tiff_handler, shapes_df, row, tile_size, max_empty_pixels_threshold, target_dir, row_index, return_dict, lock))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    for val in return_dict.values():
        print(val)
        annotations += val

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
                        help="threshold of max part of empty pixels on a tile to use it")
    parser.add_argument("--target-dir", default="data_out", help="directory to store dataset")
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                        help="whether to be verbose")
    
    args = parser.parse_args()
    
    with rio.open(args.geotiff) as geotiff:
        
        if not os.path.exists(args.target_dir):
            os.makedirs(args.target_dir)

        img_shape = geotiff.shape
        shapes_df = load_shapes_df(args.shapefile, geotiff.transform)

        rolling_window(geotiff, shapes_df, args.target_dir,
                       args.min_row, args.max_row if args.max_row >=0 else img_shape[0], 
                       args.min_col, args.max_col if args.max_col >=0 else img_shape[1],
                       args.tile_size, args.step, args.empty_pixels_threshold,
                       args.verbose)
