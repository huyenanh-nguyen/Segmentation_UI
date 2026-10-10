import math, tifffile, os, time, settings
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

from scipy import ndimage


class Taylor_watershed:
    """https://github.com/MJ-Taylor-Lab/IRAK4_Autophosphorylaytion_Paper
    for this methode we need to clean up and process the data a bit
    """
    def remove_frame(self,image_path, frame_path, tiff_compression_level):
        try:
            # Save name
            path = os.path.dirname(image_path)
            file = os.path.basename(image_path)
            file = os.path.splitext(file)[0] + '_darkframe_removed.tif'
            save_image_path = os.path.join(path, file)
            # Import images
            original_image = tifffile.imread(image_path)
            frame_image = tifffile.imread(frame_path)

            # Subtract images
            # Get frame count
            frames = original_image.shape[0]
            frames = range(0, frames)
            # Subtract
            frame_removed_image = []
            for t in frames:
                frame_removed = original_image[t]
                frame_removed = cv.subtract(frame_removed, frame_image)
                frame_removed_image.append(frame_removed)
            # Save image
            tifffile.imsave(save_image_path, frame_removed_image, bigtiff=True, compress=tiff_compression_level, dtype=frame_removed_image[0].dtype)
            time.sleep(5)
            return save_image_path
        except:
            print("Cannot complete remove")
            pass



    def median_blur_remove(self,img, median_filter_size):
        try:
            # Remove blur
            med = ndimage.median_filter(img, size=median_filter_size)
            med_rm = img - np.minimum(med, img)

            return med_rm
        except:
            print('median_blur_remove error')
            pass

    def remove_median_blur(self,image_path, old_term, new_term, median_size, tiff_compression_level):
        try:
            # Save name
            path = os.path.dirname(image_path)
            file = os.path.basename(image_path)
            file = os.path.splitext(file)[0]
            file = file.replace(old_term, new_term) + '.tif'
            save_image_path = os.path.join(path, file)

            # Import image
            original_image = tifffile.imread(image_path)

            # Get frame count
            frames = original_image.shape[0]
            frames = range(0, frames)

            # Run in parallel
            ray.shutdown()
            ray.init()
            result_ids = []
            for frame in frames:
                result_id = median_blur_remove.remote(original_image[frame], median_size)
                result_ids.append(result_id)
            med_rm_img = settings.parallel.ids_to_vals(result_ids)

            # Save image
            tifffile.imsave(save_image_path, med_rm_img, bigtiff=True, compress=tiff_compression_level, dtype=med_rm_img[0].dtype)
            time.sleep(10)
            ray.shutdown()
            time.sleep(10)
            return save_image_path
        except:
            print("Cannot complete remove")
            pass


    def tracking_image(self,median_image_path, old_term, new_term, median_size, tiff_compression_level):
        try:
            # Save name
            path = os.path.dirname(median_image_path)
            file = os.path.basename(median_image_path)
            file = os.path.splitext(file)[0]
            file = file.replace(old_term, new_term) + '.tif'
            save_image_path = os.path.join(path, file)

            # Import image
            original_image = tifffile.imread(median_image_path)

            # Get frame count
            frames = original_image.shape[0]
            frames = range(0, frames)

            # Blur image
            median_imgs = []
            for frame in frames:
                median_img = ndimage.median_filter(original_image[frame], size=median_size)
                median_imgs.append(median_img)

            # Average frames
            frames = original_image.shape[0]
            frames = range(1, frames-1)
            moving_avg_imgs = []
            for frame in frames:
                moving_avg_img = (median_imgs[frame - 1], median_imgs[frame], median_imgs[frame + 1])
                moving_avg_img = np.array(np.mean(moving_avg_img, axis=0))
                moving_avg_imgs.append(moving_avg_img)

            # Save image
            tifffile.imsave(save_image_path, moving_avg_imgs, bigtiff=True, compress=tiff_compression_level, dtype=original_image[0].dtype)
            time.sleep(5)
            return save_image_path
        except:
            print("Cannot complete remove")
            pass


    def combine_images(self,input_images, output_image, tiff_compression_level):
        try:
            n_inputs = len(input_images)
            n_inputs = range(0, n_inputs)
            existing_images = []
            for x in n_inputs:
                if os.path.exists(input_images[x]):
                    existing_images.append(input_images[x])

            # Import image
            channels = len(existing_images)
            channels = range(0, channels)

            for c in channels:
                original_image = tifffile.imread(existing_images[c])
                if c == 0:
                    summed_image = original_image
                else:
                    summed_image = cv.add(summed_image, original_image)

            # Save image
            tifffile.imsave(output_image, summed_image, bigtiff=True, compress=tiff_compression_level, dtype=original_image[0].dtype)
            time.sleep(5)
            return output_image
        except:
            print("Cannot complete combine_images")
            pass

    def segment(self,segment_image, cell_diameter, tiff_compression_level, name):
        try:
            # Import images
            img = tifffile.imread(segment_image)

            # Max pixel intensity
            img_max = np.max(img, axis=0)
            # Convert to 8-bit
            img_max = img_max - img_max.min()
            img_max = img_max / img_max.max() * 255
            img_max = img_max.astype(np.uint8)

            # Average pixel intensity
            img = np.mean(img, axis=0)
            img = np.round(img)
            # Convert to 8-bit
            img = img - img.min()
            img = img / img.max() * 255
            img = img.astype(np.uint8)

            # Create mask (background vs foreground)
            # Adjust contrast
            contrast_img = img.astype(np.uint16)
            contrast_img = contrast_img * 10
            contrast_img = np.clip(contrast_img, 0, 255)
            contrast_img = contrast_img - contrast_img.min()
            contrast_img = contrast_img / contrast_img.max() * 255
            contrast_img = contrast_img.astype(np.uint8)
            # Blur image
            blur_img = cv.blur(contrast_img, (cell_diameter * 3, cell_diameter * 3))
            blur_img = blur_img * 0.5
            blur_img = blur_img.astype(np.uint8)
            # Median image
            median_img = cv.medianBlur(contrast_img, ksize=cell_diameter)
            # Subtract blurs
            subtracted_img = cv.subtract(median_img, blur_img)

            # Dilate maxima
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (cell_diameter * 3, cell_diameter * 3))
            dilated_img = cv.dilate(subtracted_img, kernel, iterations=1)

            # Gamma correct image again to make dim objects brighter
            mid = 1
            mean = np.mean(dilated_img)
            gamma = math.log(mid * 255) / math.log(mean)
            gamma_img = np.power(dilated_img, gamma).clip(0, 255).astype(np.uint8)

            # Make binary applying a threshold
            ret, mask_img = cv.threshold(gamma_img, 1, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            mask_img_color = cv.cvtColor(mask_img, cv.COLOR_GRAY2BGR)

            # Create marker
            # Adjust contrast
            contrast_img = img_max.astype(np.uint16)
            contrast_img = contrast_img * 6.67
            contrast_img = np.clip(contrast_img, 0, 255)
            contrast_img = contrast_img - contrast_img.min()
            contrast_img = contrast_img / contrast_img.max() * 255
            contrast_img = contrast_img.astype(np.uint8)

            # Blur image
            blur_img = cv.blur(contrast_img, (cell_diameter * 3, cell_diameter * 3))
            # Median image
            median_img = cv.medianBlur(contrast_img, ksize=cell_diameter)
            # Subtract blurs
            subtracted_img = cv.subtract(median_img, blur_img)
            # Gamma correct image
            gamma_img = np.power(subtracted_img, 1.5).clip(0, 255).astype(np.uint8)
            # Median blur to remove noise
            median_img_2 = cv.medianBlur(gamma_img, ksize=cell_diameter)

            # Marker labelling
            ret, marker_img = cv.connectedComponents(median_img_2)

            # Segment
            final_segmentation = cv.watershed(mask_img_color, marker_img)
            final_segmentation.astype(np.uint8)
            # Eliminate background
            final_segmentation = marker_img * (mask_img / 255)
            final_segmentation = final_segmentation.clip(0, 254)
            final_segmentation = final_segmentation.astype('uint8')
            # Save
            # Create folders
            image_path = os.path.dirname(segment_image)
            segmentation_output_path = os.path.join(image_path, f'{name}.tif')
            tifffile.imsave(segmentation_output_path, final_segmentation, bigtiff=True, compress=tiff_compression_level,
                            dtype=final_segmentation[0].dtype)

            # Plot over image
            # Find edges
            final_segmentation[final_segmentation > 0] = [255]
            # Erode Image
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            eroded_img = cv.erode(final_segmentation, kernel, iterations=1)
            eroded_img[eroded_img < 1] = [0]
            eroded_img = eroded_img.astype(np.uint8)
            # Subtract
            eroded_img = final_segmentation - eroded_img
            eroded_img[eroded_img != 0] = [255]
            # Adjust contrast
            contrast_img = img.astype(np.uint16)
            contrast_img = contrast_img * 3
            contrast_img = contrast_img - np.mean(contrast_img)
            contrast_img = np.clip(contrast_img, 0, 255)
            contrast_img = contrast_img - contrast_img.min()
            contrast_img = contrast_img / contrast_img.max() * 255
            final_segmentation_img = contrast_img.astype(np.uint8)
            # Add boundaries
            final_segmentation_img[eroded_img == 255] = [255]

            plt.imshow(final_segmentation_img, cmap='inferno')
            overlay_file = os.path.join(image_path, 'Segmentation.png')
            plt.savefig(overlay_file)

            return segmentation_output_path
        except:
            print('segment error')
            pass


    def make_substacks(self,substack_segmentation_image, substack_image, puncta_diameter, tiff_compression_level):
        try:
            # Import images
            input_image = tifffile.imread(substack_image)
            if os.path.exists(substack_segmentation_image):
                segmentation_image = tifffile.imread(substack_segmentation_image)
                # Get cell count
                n_cells = segmentation_image.max()
                n_cells = range(1, int(n_cells))

                # Get frame count
                frames = input_image.shape[0]
                frames = range(0, frames)

                # Get cells larger than puncta area
                area_results = []
                for cell_x in n_cells:
                    result = sum(sum(segmentation_image == cell_x))
                    result = result > puncta_diameter ** 2
                    area_results.append(result)
                area_results = np.where(area_results)[0] + 1

                for cell_x in area_results:
                    try:
                        # Get cell masks
                        cell_x_mask = segmentation_image == cell_x

                        # Crop cell
                        i, j = np.where(cell_x_mask)
                        indexes = np.meshgrid(np.arange(min(i), max(i) + 1), np.arange(min(j), max(j) + 1), indexing='ij')

                        # Run in parallel
                        cropped_img = []
                        for frame_x in frames:
                            cell_x_img = input_image[frame_x]
                            cutout_img = cell_x_img * cell_x_mask
                            cropped_frame_img = cutout_img[tuple(indexes)]
                            cropped_img.append(cropped_frame_img)
                        # Save image
                        save_path = os.path.dirname(substack_image)
                        cell_path = np.where(area_results == cell_x)[0][0] + 1
                        cell_path = 'Cell_' + cell_path.__str__()
                        cell_path = os.path.join(save_path, cell_path)
                        # Create cell path if it doesn't exist
                        if not os.path.exists(cell_path):
                            os.mkdir(cell_path)
                        # Save name
                        save_name = os.path.basename(substack_image)
                        save_path = os.path.join(cell_path, save_name)
                        # Save image
                        tifffile.imsave(save_path, cropped_img, bigtiff=False, compress=tiff_compression_level,
                                        dtype=cropped_img[0].dtype)
                        time.sleep(5)
                    except:
                        print('Error making substack: ' + substack_image + 'Cell_' + cell_x)
                        pass
            else:
                # Save image
                save_path = os.path.dirname(substack_image)
                cell_path = os.path.join(save_path, 'Cell_1')
                # Create cell path if it doesn't exist
                if not os.path.exists(cell_path):
                    os.mkdir(cell_path)
                # Save name
                save_name = os.path.basename(substack_image)
                save_path = os.path.join(cell_path, save_name)
                # Save image
                tifffile.imsave(save_path, input_image, bigtiff=False, compress=tiff_compression_level,
                                dtype=input_image[0].dtype)
                time.sleep(5)
        except:
            print('make_substacks error')
            pass


    def make_area_list(self,substack_segmentation_image, puncta_diameter):
        try:
            # Save table
            image_path = os.path.dirname(substack_segmentation_image)
            save_path = os.path.join(image_path, 'cell_area.csv')

            if os.path.exists(substack_segmentation_image):
                segmentation_image = tifffile.imread(substack_segmentation_image)
                # Get cell count
                n_cells = segmentation_image.max()
                n_cells = range(1, int(n_cells))
                # Get number of rows
                n_rows = range(0, segmentation_image.shape[0])

                # Get cells larger than puncta area
                area_results = []
                min_x_results = []
                min_y_results = []
                for cell_x in n_cells:
                    # Get area
                    area = sum(sum(segmentation_image == cell_x))
                    if area > puncta_diameter ** 2:
                        area_results.append(area)
                        # Get x,y coordintes of cell
                        min_x = []
                        min_y = []
                        cell_x_mask = segmentation_image == cell_x
                        for row in n_rows:
                            index = np.argmax(cell_x_mask[row] == True)
                            if not index == 0:
                                min_x.append(index)
                                min_y.append(row)
                        # Get minima
                        min_x = min(min_x) + 1
                        min_x_results.append(min_x)
                        min_y = min(min_y) + 1
                        min_y_results.append(min_y)

                # Get cell folder names
                cell_names = range(1, len(area_results) + 1)
                cell_names = list(cell_names)
                new_cell_names = []
                for cell_name in cell_names:
                    cell_name = 'Cell_' + str(cell_name)
                    new_cell_names.append(cell_name)
                # Get new cell count
                n_cells = range(0, len(new_cell_names))
                with open(save_path, 'w') as f:
                    f.write('cell,area,position_x,position_y\n')
                    for cell in n_cells:
                        f.write(new_cell_names[cell] + ',' + str(area_results[cell]) + ',' + str(
                            min_x_results[cell]) + ',' + str(min_y_results[cell]) + '\n')
            else:
                metadata = os.path.join(image_path, 'metadata.csv')
                metadata = pd.read_csv(metadata)
                # Get height
                height = metadata['parameter'] == 'height'
                height = metadata[height]['value']
                height = int(height)
                # Get width
                width = metadata['parameter'] == 'width'
                width = metadata[width]['value']
                width = int(width)
                area = str(height * width)
                # Save table
                with open(save_path, 'w') as f:
                    f.write('cell,area,position_x,position_y\n')
                    f.write('Cell_1,' + area + ',1,1')
        except:
            print('make_substacks error')
            pass