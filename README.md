# Greenland Sea Ice Floe Segmentation and Tracking Via the Ice Floe Tracker Algorithm
The Ice Floe Tracker (IFT) algorithm was developed by Rosalinda Lopez-Acosta to process moderate resolution optical satellite imagery for the retrieval of sea ice floe shapes and dynamics. IFT takes a feature matching (FM) approach to sea ice tracking, extracting ice floe borders from imagery then using a boundary correlation method to identify floe displacement and rotation in subsequent images. For her thesis work (Lopez-Acosta, 2021), images from the Moderate Resolution Imaging Spectroradiometer from 2003-2020 were processed for the Beaufort Sea (Manucharyan et al., 2022) and for the Greenland Sea south of the Fram Strait. The results from the latter were used in her dissertation and have been previously presented in Watkins et al., 2023.

This dataset collects the Greenland Sea results, converted from proprietary MATLAB format into open GeoTiff and CSV format. MATLAB data and script files are included as well as Python scripts to read and format the MATLAB data.  We additionally use a logistic regression classifier to remove false positives from the dataset. The original segmentation results are saved as "labeled_raw" and the results with flagged flase positives removed is saved as "labeled_clean". Access to additional datasets, in the form of true and false color MODIS imagery and sea ice concentration data, is required, and is included in the data repository but due to size limitations not included in the GitHub repository.


## Reprocessing Algorithm Results from Lopez-Acosta (2021)
MODIS imagery from the East Greenland Sea south of Fram Strait for the years 2003-2020 was processed by Rosalinda Lopez-Acosta for her PhD dissertation "Sea Ice Drift in Arctic Marginal Ice Zones Derived from Optical Satellite imagery". The image processing and analysis was carried out using MATLAB software. This repository contains code to extract data from the MATLAB output and convert it into cross-platform readable GeoTiff and CSV files for broader public release.

## Downloading MODIS imagery via IFT Pipeline
The MODIS dataset is large, even when subsetted to the study area, and is therefore not included in this repository. To download the data, we use the Ice Floe Tracker Pipeline. The file `scripts/00_setup_ft_table.py` generates the set of CSV files in the folder `data/modis_download_spec_files`. To download the MODIS imagery on the Oscar HPC system at Brown, after installing the Ice Floe Tracker Pipeline, modify the Cylc graph in `flow_template_hpc.j2` to read:

```R1 = global_setup  => mkpaths<param_set> => pullfetchimage & pulljuliaimage => fetchdata<param_set> & soit<param_set>```

Copy the specification files to the `config` folder. Each year is run separately. For year 2019, as an example, load the IFT environment and run the python command
```
python workflow/scripts/flow_generator.py \
--csvfile "./config/fram_strait_spec_tables/location_specs_2019.csv" \
--template "flow_template_hpc.j2" \
--template_dir "./config/cylc_hpc" \
--crs "epsg3413" \
--minfloearea 100 \
--maxfloearea 90000
```
Then run the IFT pipeline in cylc via
```
cylc install -n fram_strait_images ./config/cylc_hpc && \
cylc validate fram_strait_images && \
cylc play fram_strait_images && \
cylc tui fram_strait_images
```

## Extracting floe property tables
The MATLAB code produces files with floe properties and floe positions for (a) all candidate floe segments and (b) for all floes that were matched to a subsequent image. The file `01_parse_ft_data.py` extracts the floe properties and positions from the MATLAB output. Along with the values originally in the props.mat file, it adds a `floe_label` so that tracked floes can be assembled into trajectories. The files `time_data.csv` were manually created using a variety of sources including saved diagnostic images and output from the SOIT python function. It maps the index in the FLOE_LIBRARY and props.mat to time stamps and specific satellites. 

Data structure for the all_floes tables:

|Column|Description|Units|
|---|---|---|
|'area'| | |
|'perimeter'| | |
|'major_axis'| | |
|'minor_axis'| | |
|'orientation'| | |
|'x_pixel'| Column coordinate in the original image |
|'y_pixel'| Row coordinate in the original image |
|'convex_area'| Area of the best-fit convex polygon |
|'solidity'| Ratio of area to convex area | Unitless |
|'bbox1'| | |
|'bbox2'| | |
|'bbox3'| | |
|'bbox4'| | |
|'orig_idx'|Index of row in floe property table in Matlab files| NA |
|'satellite'|Name of satellite| NA |
|'floe_id'|Unique label assigned to tracked floes|YYYY_NNNNN|
|'datetime'|Time of satellite overpass of the image centroid|YYYY-mm-dd HH:MM|
|'x_stere'|X-position of the floe centroid in NSIDC N. Polar Stereographic|meters|
|'y_stere'|Y-position of the floe centroid in NSIDC N. Polar Stereographic|meters|
|'longitude'|Longitude of the floe centroid|Decimal Degrees|
|'latitude'|Latitude of the floe centroid|Decimal Degrees|
|'nsidc_sic'|Sea ice concentration of nearest grid cell from NSIDC CDR|Fraction|
|'label'|Integer object label in the segmented image| NA |

### TBD for property tables:
* Add area, perimeter in km
* Fix order of columns, names to match updates
* Set it up so that the pixel brightness table is the one saved to the archive
* Change the column order so ID information comes first
* Convert the x_pixel/y_pixel and bounding boxes for the 2020 data to reflect the position in the final image

## Extracting floe shapes
Floe shapes are stored in a MATLAB structure `FLOE_LIBRARY.mat`. This structure efficiently holds the sparse dataset of labeled floe shapes. However it is not easily visualized or shared, as it is not self-describing. The script `02_extract_shapes.py` reads the data in the FLOE LIBRARY and in the floe property tables, then creates a GeoTiff sharing dimensions and coordinate reference system with the reference image `NE_Greenland.2017100.terra.250m.tif`. The file produced is an unfiltered segmented image where the labels of each floe correspond to the index in the FLOE_LIBRARY. A tracked floe will have different label numbers in each image. 

TBD: see if there's a way to more accurately calculate the resize coefficients for the 2020 data.

## Get floe properties
Floe properties were initially calculated in MATLAB and are saved by the `01_parse_ft_data.py` script. There are differences in the algorithms used by scikit image region properties function and the identically named function in MATLAB. For future compatibility with the IFT Julia version, which uses scikit image, we recalculate region properties and add these to the floe property tables. This step also allows us to get consistent bounding boxes and row/col centroid data for the shapes. Using the shapes extracted in the previous step, and the truecolor and falsecolor images, we get the mean intensity for each color channel within each floe. This data is used for filtering true and false positives from the floe property tables.

## Cleaning dataset using logistic regression function
The IFT segmentation step produces a set of candidate ice floes for matching. For estimates of the floe size distribution, ideally all detected floe shapes can be used (rather than only tracked floes). Tracking floes filters out candidate segments corresponding to bright patches in clouds, ice filements, clumps of ice floes below the image resolution, and other similar objects due to the tendency of these objects to deform strongly between images. Buckley et al. (2023) used floe circularity, a function of the floe perimeter and area, to filter out false positives. However, the floe circularity is, in general, a necessary but not sufficient criterion. Many false positives also have similar circularity properties as real floes. 

### References
Buckley, E., Cañuelas, Timmermans, M.-L., and Wilhelmus, M. M. (2023), "Seasonal Evolution of the Sea Ice Floe Size Distribution from Two Decades of MODIS Data," EGUsphere (preprint), https://doi.org/10.5194/egusphere-2024-89

Lopez-Acosta, R., Schodlok, M. P., and Wilhelmus, M. M. (2019). "Ice Floe Tracker: An algorithm to automatically retrieve Lagrangian trajectories via feature matching from moderate-resolution visual imagery", Remote Sensing of Environment, 234, 111406, pp. 1-15. DOI:10.1016/j.rse.2019.111406

Lopez-Acosta, R. (2021), "Sea Ice Drift in Arctic Marginal Ice Zones Derived from Optical Satellite Imagery" Doctoral dissertation, University of California Riverside. 162 pages.

Manucharyan, G., Lopez-Acosta, R., and Wilhelmus, M. M. (2022), "Spinning ice floes reveal intensification of mesoscale eddies in the western Arctic Ocean", Scientific Reports, 12, 7070, pp. 1-13

Pedregosa et al. (2011), "Scikit-learn: Machine Learning in Python", Journal of Machine Learning Research, 12, pp. 2825-2830, 2011.

Van der Walt, S., Schönberger, Johannes L, Nunez-Iglesias, J., Boulogne, Franccois, Warner, J. D., Yager, N., ..., Yu, T. (2014). scikit-image: image processing in Python. PeerJ, 2, e453.

Watkins, D. M., Bliss, A. C., Hutchings, J. K., Wilhelmus, M. M. (2023), "Evidence of abrupt transitions between sea ice dynamical regimes in the East Greenland marginal ice zone", Geophysical Research Letters, 50, e2023GL103558, pp. 1-10

### Contributors
* Daniel Watkins
* Rosalinda Lopez-Acosta
* Minki Kim
* Monica Martinez Wilhelmus
* Ashfaq Ahmed
* Ellen Buckley
* Simon Hatcher
