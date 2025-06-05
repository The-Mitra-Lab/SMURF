import copy
import os
import pickle
import warnings

import anndata
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm
from numba import jit
from numpy.linalg import norm
from PIL import Image
from scipy.sparse import SparseEfficiencyWarning, csr_matrix, lil_matrix
from sklearn.metrics import normalized_mutual_info_score

from .spatial_object import spatial_object


@jit(nopython=True)
def fill_pixels(pixels, row_starts, row_ends, col_starts, col_ends, indices):

    # Function to speed up
    # Fill each region in the pixels array with the corresponding index
    for i in range(len(indices)):
        pixels[row_starts[i] : row_ends[i], col_starts[i] : col_ends[i]] = indices[i]

    return pixels


def prepare_dataframe_image(
    df_path: "square_002um/spatial/tissue_positions.parquet",
    img_path: "Visium_HD_Mouse_Small_Intestine_tissue_image.btf",
    image_format="HE",
    row_number=3350,
    col_nuber=3350,
):

    """
    Prepares the spatial data object by mapping tissue image data to spot positions.

    This function reads the high-resolution tissue image and the corresponding spot position data
    from a Parquet file. It calculates pixel boundaries for each spot and creates a spatial object
    that maps the image to the tissue spots. The spatial object can be used for downstream analysis,
    such as extracting spot-specific image data or overlaying spatial expression data on the image.

    :param df_path:
        The file path to the Parquet file containing spot position data.
        e.g., 'square_002um/spatial/tissue_positions.parquet'. The DataFrame should contain columns like
        'pxl_row_in_fullres', 'pxl_col_in_fullres', 'array_row', 'array_col', and 'in_tissue'.
    :type df_path: str

    :param img_path:
        The file path to the full-resolution image.
        e.g., 'Visium_HD_Mouse_Small_Intestine_tissue_image.btf'.
    :type img_path: str

    :param image_format:
        The format of the image. Must be either 'HE' or 'DAPI'. Defaults to 'HE'.
    :type image_format: str, optional

    :param row_number:
        The number of rows in the spot array (used for calculating average spot sizes). Defaults to 3350.
    :type row_number: int, optional

    :param col_number:
        The number of columns in the spot array (used for calculating average spot sizes). Defaults to 3350.
    :type col_number: int, optional

    :return:
        A spatial_object containing the image array, DataFrame with spot data, adjusted spot boundaries,
        and other spatial mappings necessary for spatial analysis.
    :rtype: spatial_object

    """

    if image_format not in ["HE", "DAPI"]:
        raise ValueError("image_format must be in ['HE','DAPI']")

    # read image
    Image.MAX_IMAGE_PIXELS = None  # Removes the limit entirely

    try:
        # Open the image with Pillow
        image = Image.open(img_path)
        image_array = np.array(image)

    # print('The shape of original image is': image_array.shape)

    except IOError as e:

        print(f"Error opening or processing image: {e}")

    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(df_path, engine="pyarrow")

    # Calculate the average row and column sizes
    avg_row = (df["pxl_row_in_fullres"].max() - df["pxl_row_in_fullres"].min()) / (
        2 * row_number
    )
    avg_col = (df["pxl_col_in_fullres"].max() - df["pxl_col_in_fullres"].min()) / (
        2 * col_nuber
    )

    # Calculate the left, right, top, and bottom pixel boundaries for each spot
    df["pxl_row_left_in_fullres"] = df["pxl_row_in_fullres"] - avg_row
    df["pxl_row_right_in_fullres"] = df["pxl_row_in_fullres"] + avg_row
    df["pxl_col_up_in_fullres"] = df["pxl_col_in_fullres"] - avg_col
    df["pxl_col_down_in_fullres"] = df["pxl_col_in_fullres"] + avg_col

    # Round the boundaries to integers
    df["pxl_row_left_in_fullres"] = df["pxl_row_left_in_fullres"].round().astype(int)
    df["pxl_row_right_in_fullres"] = df["pxl_row_right_in_fullres"].round().astype(int)
    df["pxl_col_up_in_fullres"] = df["pxl_col_up_in_fullres"].round().astype(int)
    df["pxl_col_down_in_fullres"] = df["pxl_col_down_in_fullres"].round().astype(int)

    # Determine the range of spots that are in tissue
    start_row_spot = df[(df["in_tissue"] == 1)]["array_row"].min()
    end_row_spot = df[(df["in_tissue"] == 1)]["array_row"].max() + 1
    start_col_spot = df[(df["in_tissue"] == 1)]["array_col"].min()
    end_col_spot = df[(df["in_tissue"] == 1)]["array_col"].max() + 1

    # Create a temporary DataFrame with spots in the specified range
    df_temp = df[
        (df["array_row"] >= start_row_spot)
        & (df["array_row"] < end_row_spot)
        & (df["array_col"] >= start_col_spot)
        & (df["array_col"] < end_col_spot)
    ].copy()

    # Calculate the cropped image boundaries
    row_left = max(df_temp["pxl_row_left_in_fullres"].min(), 0)
    row_right = df_temp["pxl_row_right_in_fullres"].max()
    col_up = max(df_temp["pxl_col_up_in_fullres"].min(), 0)
    col_down = df_temp["pxl_col_down_in_fullres"].max()

    # Adjust the pixel boundaries relative to the cropped image
    df_temp.loc[:, "pxl_row_left_in_fullres_temp"] = (
        df_temp.loc[:, "pxl_row_left_in_fullres"] - row_left
    )
    df_temp.loc[:, "pxl_row_right_in_fullres_temp"] = (
        df_temp.loc[:, "pxl_row_right_in_fullres"] - row_left
    )
    df_temp.loc[:, "pxl_col_up_in_fullres_temp"] = (
        df_temp.loc[:, "pxl_col_up_in_fullres"] - col_up
    )
    df_temp.loc[:, "pxl_col_down_in_fullres_temp"] = (
        df_temp.loc[:, "pxl_col_down_in_fullres"] - col_up
    )

    # Extract the start and end positions for rows and columns
    row_starts = df_temp["pxl_row_left_in_fullres_temp"].values
    row_ends = df_temp["pxl_row_right_in_fullres_temp"].values
    col_starts = df_temp["pxl_col_up_in_fullres_temp"].values
    col_ends = df_temp["pxl_col_down_in_fullres_temp"].values
    indices = df_temp.index.to_numpy()

    # Initialize the pixels array with -1
    pixels = -1 * np.ones(image_array[row_left:row_right, col_up:col_down].shape[:2])
    # Fill the pixels array
    pixels = fill_pixels(pixels, row_starts, row_ends, col_starts, col_ends, indices)

    # Return a spatial object with relevant data
    return spatial_object(
        image_array,
        df,
        df_temp,
        start_row_spot,
        end_row_spot,
        start_col_spot,
        end_col_spot,
        row_left,
        row_right,
        col_up,
        col_down,
        pixels,
        image_format,
    )


def nuclei_rna(adata, so, min_percent=0.4):

    """
    Creates an AnnData object for nuclei by aggregating spot-level gene expression data.

    This function processes spatial gene expression data to generate a nuclei-level AnnData object.
    It aggregates gene expression counts from spots corresponding to individual nuclei, based on spatial
    relationships and predefined mappings. It also filters spots based on a minimum percentage criterion
    for inclusion.

    :param adata:
        An AnnData object containing spot-level gene expression data.
    :type adata: anndata.AnnData

    :param so:
        A spatial_object containing image data, spot mappings, and spatial relationships.
    :type so: spatial_object

    :param min_percent:
        The minimum percentage of blank area within in a spot threshold for including neighboring spots in the cell aggregation.
        Spots with a blank proportion below this threshold will be excluded. Defaults to 0.4.
    :type min_percent: float, optional

    :return:
        The updated spatial_object `so` with a new attribute `final_nuclei`, which is an AnnData object
        containing nuclei-level gene expression data.
    :rtype: spatial_object

    """

    # Create a mapping from spot positions to indices in the adata object
    set_toindex_data = {}
    indexes = list(adata.obs.index)
    for i in range(len(indexes)):
        set_toindex_data[
            (int(indexes[i].split("_")[2]), int(indexes[i].split("_")[3].split("-")[0]))
        ] = i

    set_cells = {}
    set_tobes = {}
    set_excludes = {}

    length_main = {}
    cell_matrix = {}

    i_num = 0
    data_temp = csr_matrix(adata.X)
    final_data = lil_matrix((len(so.cell_ids), data_temp.shape[1]))

    # Get and sort the cell IDs
    cell_ids = copy.deepcopy(so.cell_ids)
    cell_ids = np.sort(np.array(cell_ids))

    for cell_id in tqdm.tqdm(cell_ids):

        # Initialize sets for the cell
        set_cell = set()
        set_tobe = set()
        set_exclude = set()

        rows = []
        cols = []

        # Get the array_row and array_col for each spot in the cell
        for spot_id in so.cells_main[cell_id]:
            rows.append(so.df.loc[spot_id, "array_row"])
            cols.append(so.df.loc[spot_id, "array_col"])
            set_cell.add(
                (so.df.loc[spot_id, "array_row"], so.df.loc[spot_id, "array_col"])
            )

        data_cells = np.array([rows, cols])

        row_min = {}
        row_max = {}
        col_min = {}
        col_max = {}

        # Initialize min and max dictionaries for rows and columns
        for row in set(rows):
            row_min[row] = float("inf")
            row_max[row] = 0

        for col in set(cols):
            col_min[col] = float("inf")
            col_max[col] = 0

        # Update min and max values for rows and columns
        for i in range(len(so.cells_main[cell_id])):

            if data_cells[0, i] < col_min[data_cells[1, i]]:
                col_min[data_cells[1, i]] = data_cells[0, i]

            if data_cells[0, i] > col_max[data_cells[1, i]]:
                col_max[data_cells[1, i]] = data_cells[0, i]

            if data_cells[1, i] < row_min[data_cells[0, i]]:
                row_min[data_cells[0, i]] = data_cells[1, i]

            if data_cells[1, i] > row_max[data_cells[0, i]]:
                row_max[data_cells[0, i]] = data_cells[1, i]

        # Filter the set_cell to be within valid spot ranges
        set_cell = {
            tup
            for tup in set_cell
            if tup[0] >= so.start_row_spot and tup[0] < so.end_row_spot
        }
        set_cell = {
            tup
            for tup in set_cell
            if tup[1] >= so.start_col_spot and tup[1] < so.end_col_spot
        }
        set_cells[cell_id] = set_cell
        length_main[cell_id] = len(set_cell)

        # Find neighboring spots to consider
        for col, row in col_min.items():
            if (row - 1, col) in set_toindex_data:
                set_tobe.add((row - 1, col))

        for col, row in col_max.items():
            if (row + 1, col) in set_toindex_data:
                set_tobe.add((row + 1, col))

        for row, col in row_min.items():
            if (row, col - 1) in set_toindex_data:
                set_tobe.add((row, col - 1))

        for row, col in row_max.items():
            if (row, col + 1) in set_toindex_data:
                set_tobe.add((row, col + 1))

        # Filter set_tobe to be within valid spot ranges
        set_tobe = {
            tup
            for tup in set_tobe
            if tup[0] >= so.start_row_spot and tup[0] < so.end_row_spot
        }
        set_tobe = {
            tup
            for tup in set_tobe
            if tup[1] >= so.start_col_spot and tup[1] < so.end_col_spot
        }

        sets_temp = copy.copy(set_tobe)
        for i, j in sets_temp:
            spot = so.df.index[so.set_toindex[(i, j)]]
            if 0 in so.spots[spot]:
                if so.spots[spot][0] / sum(so.spots[spot].values()) < min_percent:
                    set_exclude.add((i, j))
            else:
                set_exclude.add((i, j))

        set_tobes[cell_id] = set_tobe
        set_excludes[cell_id] = set_exclude

        # Build the cell matrix by summing expression values
        cell_matrix[cell_id] = [
            set_toindex_data[key] for key in set_cell if key in set_toindex_data
        ]
        cell_matrix[cell_id] = (
            data_temp[cell_matrix[cell_id], :].sum(axis=0).reshape(1, -1)
        )
        final_data[i_num] = cell_matrix[cell_id]
        i_num = i_num + 1

    # Update the spatial object with new attributes
    so.set_toindex_data = copy.deepcopy(set_toindex_data)
    so.set_cells = copy.deepcopy(set_cells)
    so.set_tobes = copy.deepcopy(set_tobes)
    so.set_excludes = copy.deepcopy(set_excludes)
    so.length_main = copy.deepcopy(length_main)
    so.cell_matrix = copy.deepcopy(cell_matrix)

    cell_ids_str = [str(cell_id) for cell_id in cell_ids]

    # Create a new AnnData object for the final nuclei
    so.final_nuclei = anndata.AnnData(
        X=copy.deepcopy(csr_matrix(final_data)),
        obs=pd.DataFrame([], index=cell_ids_str),
        var=adata.var,
    )

    return so


def singlecellanalysis(
    adata,
    save=False,
    iter=None,
    path=None,
    resolution=2,
    regress_out=True,
    random_state=0,
    show=True,
):

    """
    Performs standard single-cell RNA-seq analysis, including preprocessing, dimensionality reduction,
    clustering, and visualization.

    This function takes an AnnData object containing single-cell gene expression data and performs a series
    of standard analysis steps:

    - Filters genes expressed in a minimum number of cells.
    - Calculates quality control (QC) metrics, including mitochondrial gene content.
    - Normalizes and log-transforms the data.
    - Identifies highly variable genes.
    - Regresses out effects of total counts and mitochondrial gene expression (optional).
    - Scales the data.
    - Performs principal component analysis (PCA).
    - Computes the neighborhood graph and UMAP embedding.
    - Performs Leiden clustering.
    - Optionally visualizes the UMAP embedding colored by cluster assignments.

    :param adata:
        An AnnData object containing single-cell gene expression data.
    :type adata: anndata.AnnData

    :param save:
        Whether to save the UMAP plot. If `True`, saves the plot with a default filename.
        If a string is provided, saves the plot with the given filename. Defaults to `False`.
    :type save: bool or str, optional

    :param iter:
        An iteration or index number used in saving the plot filename.
        Only used if `save` is `True`. Defaults to `None`.
    :type i: int or None, optional

    :param path:
        The directory path where the plot will be saved. Not used in the current implementation.
        Defaults to `None`.
    :type path: str or None, optional

    :param resolution:
        The resolution parameter for Leiden clustering, controlling the granularity of the clusters.
        Defaults to `2`.
    :type resolution: float, optional

    :param regress_out:
        Whether to regress out effects of total counts and mitochondrial percentage during preprocessing.
        Defaults to `True`.
    :type regress_out: bool, optional

    :param random_state:
        The seed for random number generators to ensure reproducibility. Defaults to `0`.
    :type random_state: int, optional

    :param show:
        Whether to print progress messages and show plots. Defaults to `True`.
    :type show: bool, optional

    :return:
        The AnnData object after processing, including clustering results and UMAP embeddings.
    :rtype: anndata.AnnData
    """

    # Ignore warnings related to sparse matrices and IOStream
    warnings.simplefilter("ignore", SparseEfficiencyWarning)
    warnings.filterwarnings("ignore", message="IOStream.flush timed out")

    if show:
        print("Starting mt")
    # Filter genes expressed in at least 3 cells
    sc.pp.filter_genes(adata, min_cells=3)
    # Calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    if show:
        print("Starting normalization")
    # Normalize the data and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable].copy()
    if regress_out:
        # Regress out effects of total counts and mitochondrial percentage
        sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
    # Scale the data
    sc.pp.scale(adata, max_value=10)

    if show:
        print("Starting PCA")
    # Perform PCA
    sc.tl.pca(adata, svd_solver="arpack", random_state=random_state)

    if show:
        print("Starting UMAP")
    # Compute the neighborhood graph and UMAP embedding
    sc.pp.neighbors(
        adata,
        n_neighbors=10,
        n_pcs=40,
        random_state=random_state,
    )
    sc.tl.umap(adata)

    if show:
        print("Starting Clustering")
    # Perform Leiden clustering
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        flavor="igraph",
        n_iterations=2,
    )

    if show:
        # Plot the UMAP with clusters
        if save:
            sc.pl.umap(adata, color=["leiden"], save=str(iter) + "_umap.png")
        elif save == False:
            sc.pl.umap(adata, color=["leiden"])
        else:
            sc.pl.umap(adata, color=["leiden"], save=save)

    return adata


def expanding_cells(
    so, adata_sc, weights, iter_cells, data_temp, min_percent=0.4, cortex=False
):

    """
    Expands cell regions by iteratively adding neighboring spots based on a scoring criterion.

    This function aims to grow the regions of cells in spatial data by evaluating neighboring spots
    and including them if they improve a cell-specific score. The score is calculated using the dot
    product of the cell's expression vector and precomputed weights for the cell type, normalized by
    the vector norm. Optionally, the function can perform additional processing for cortical data
    using Delaunay triangulation to fill in cell regions more completely.

    :param so:
        A spatial object containing spatial mappings, spot data, and other necessary attributes.
    :type so: spatial_object

    :param adata_sc:
        An AnnData object containing single-cell gene expression data with cell IDs and cell types.
    :type adata_sc: anndata.AnnData

    :param weights:
        A dictionary mapping cell types to weight vectors used in the scoring function.
    :type weights: dict

    :param iter_cells:
        A dictionary mapping cell IDs to the number of iterations allowed for expanding each cell.
    :type iter_cells: dict

    :param data_temp:
        A sparse matrix containing gene expression data used for calculations during cell expansion.
    :type data_temp: scipy.sparse.csr_matrix or similar

    :param min_percent:
        The minimum percentage threshold for including neighboring spots in the cell expansion.
        Spots with a proportion below this threshold will be excluded. Defaults to `0.4`.
    :type min_percent: float, optional

    :param cortex:
        If `True`, performs additional processing suitable for cortical data using Delaunay triangulation
        to fill in cell regions. Defaults to `False`.
    :type cortex: bool, optional

    :return:
        A tuple containing:

        - `cells_final` (dict): A dictionary mapping cell IDs to their final set of spots after expansion.
        - `final_data` (scipy.sparse.lil_matrix): A matrix containing the aggregated gene expression data
          for each expanded cell.
        - `cellscore` (numpy.ndarray): An array containing the final score for each cell.
        - `length_final` (numpy.ndarray): An array containing the final number of spots for each cell.
    :rtype: tuple

    """

    cells_final = {}

    # Get cell IDs and types
    cell_ids = copy.deepcopy(list(adata_sc.obs.index.astype(float)))
    total_num = len(cell_ids)
    celltypes = copy.deepcopy(list(adata_sc.obs.leiden.astype(int)))
    cellscore = -1 * np.ones([total_num, 1])
    length_final = -1 * np.ones([total_num, 1])
    final_data = copy.deepcopy(lil_matrix((total_num, data_temp.shape[1])))

    # Define a function to fill in points to make the cell region more compact
    def fill_points(points):

        point_set1 = set(map(tuple, points))
        point_set = set(map(tuple, points))
        new_points = set()

        for x, y in point_set1:
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for xx, yy in neighbors:
                if (
                    ((xx, yy) not in point_set)
                    and ((xx + 1, yy) in point_set)
                    and ((xx - 1, yy) in point_set)
                    and ((xx, yy + 1) in point_set)
                    and ((xx, yy - 1) in point_set)
                ):
                    point_set.update([(xx, yy)])

        return point_set

    for i_num in tqdm.tqdm(range(total_num)):

        cell_id = cell_ids[i_num]
        cell_type = celltypes[i_num]

        # Get the initial sets and cell matrix
        set_cell = copy.deepcopy(so.set_cells[cell_id])
        set_tobe = copy.deepcopy(so.set_tobes[cell_id])
        set_exclude = copy.deepcopy(so.set_excludes[cell_id])
        cell_matrix = copy.deepcopy(so.cell_matrix[cell_id])

        # Calculate the initial score
        score = (np.dot(cell_matrix, weights[cell_type]) / norm(cell_matrix))[0, 0]
        tobe_considered = set_tobe - set_cell - set_exclude
        i = 0

        # Begin expanding the cell region
        while len(tobe_considered) != 0:
            for row, col in tobe_considered:
                cell_matrix_temp = (
                    data_temp[so.set_toindex_data[(row, col)], :] + cell_matrix
                )

                score_temp = (
                    np.dot(cell_matrix_temp, weights[cell_type])
                    / norm(cell_matrix_temp)
                )[0, 0]

                if score_temp > score:

                    # If the score improves, update the cell region
                    set_cell.update([(row, col)])
                    cell_matrix = cell_matrix_temp

                    # Check neighboring spots and update set_tobe or set_exclude
                    if (
                        row - 1 >= so.start_row_spot
                        and (row - 1, col) in so.set_toindex_data
                    ):
                        spot = so.df.index[so.set_toindex[(row - 1, col)]]
                        if 0 in so.spots[spot]:
                            if (
                                so.spots[spot][0] / sum(so.spots[spot].values())
                                >= min_percent
                            ):
                                set_tobe.update([(row - 1, col)])
                        else:
                            set_exclude.add((row - 1, col))
                            set_exclude.add((row - 2, col))
                            set_exclude.add((row - 1, col + 1))
                            set_exclude.add((row - 1, col - 1))

                    if (
                        row + 1 < so.end_row_spot
                        and (row + 1, col) in so.set_toindex_data
                    ):
                        so.spot = so.df.index[so.set_toindex[(row + 1, col)]]
                        if 0 in so.spots[spot]:
                            if (
                                so.spots[spot][0] / sum(so.spots[spot].values())
                                >= min_percent
                            ):
                                set_tobe.update([(row + 1, col)])
                        else:
                            set_exclude.add((row + 1, col))
                            set_exclude.add((row + 2, col))
                            set_exclude.add((row + 1, col + 1))
                            set_exclude.add((row + 1, col - 1))

                    if (
                        col - 1 >= so.start_col_spot
                        and (row, col - 1) in so.set_toindex_data
                    ):
                        spot = so.df.index[so.set_toindex[(row, col - 1)]]
                        if 0 in so.spots[spot]:
                            if (
                                so.spots[spot][0] / sum(so.spots[spot].values())
                                >= min_percent
                            ):
                                set_tobe.update([(row, col - 1)])
                        else:
                            set_exclude.add((row, col - 1))
                            set_exclude.add((row, col - 2))
                            set_exclude.add((row + 1, col - 1))
                            set_exclude.add((row - 1, col - 1))

                    if (
                        col + 1 < so.end_col_spot
                        and (row, col + 1) in so.set_toindex_data
                    ):
                        so.spot = so.df.index[so.set_toindex[(row, col + 1)]]
                        if 0 in so.spots[spot]:
                            if (
                                so.spots[spot][0] / sum(so.spots[spot].values())
                                >= min_percent
                            ):
                                set_tobe.update([(row, col + 1)])
                        else:
                            set_exclude.add((row, col + 1))
                            set_exclude.add((row, col + 2))
                            set_exclude.add((row + 1, col + 1))
                            set_exclude.add((row - 1, col + 1))

                    cell_matrix = cell_matrix_temp
                    score = score_temp

                else:
                    set_exclude.update([(row, col)])

            tobe_considered = set_tobe - set_cell - set_exclude
            i = i + 1

            if i >= iter_cells[cell_id]:
                break

        cells_final[cell_id] = copy.deepcopy(set_cell)
        length_final[i_num] = len(set_cell)
        cellscore[i_num] = score
        final_data[i_num] = copy.deepcopy(cell_matrix)

    # make results cortex
    if cortex:
        for i_num in range(total_num):

            cell_id = cell_ids[i_num]
            cell_type = celltypes[i_num]

            points = np.array(list(cells_final[cell_id]))

            if len(points) > 2:

                from scipy.spatial import ConvexHull, Delaunay

                # Example set of integer points
                points = np.array(list(cells_final[cell_id]))

                try:
                    tri = Delaunay(points)

                    min_x, max_x = min(points[:, 0]), max(points[:, 0])
                    min_y, max_y = min(points[:, 1]), max(points[:, 1])

                    for x in range(int(min_x), int(max_x) + 1):
                        for y in range(int(min_y), int(max_y) + 1):
                            test_point = np.array([x, y])
                            if tri.find_simplex(test_point) >= 0:
                                if not any(np.all(points == test_point, axis=1)):
                                    cells_final[cell_id].update([(x, y)])

                    set_cell_index = [
                        so.set_toindex_data[key]
                        for key in cells_final[cell_id]
                        if key in so.set_toindex_data
                    ]
                    cell_matrix = copy.deepcopy(
                        data_temp[set_cell_index, :].sum(axis=0).reshape(1, -1)
                    )
                    cellscore[i_num] = np.dot(cell_matrix, weights[cell_type]) / norm(
                        cell_matrix
                    )
                    length_final[i_num] = len(cells_final[cell_id])
                    final_data[i_num] = copy.deepcopy(cell_matrix)

                except:
                    continue

    else:
        for i_num in tqdm.tqdm(range(total_num)):

            cell_id = cell_ids[i_num]
            cell_type = celltypes[i_num]

            points = np.array(list(cells_final[cell_id]))

            # Fill in the points to make the cell region more continuous
            cells_final[cell_id] = fill_points(points)

            set_cell_index = [
                so.set_toindex_data[key]
                for key in cells_final[cell_id]
                if key in so.set_toindex_data
            ]
            cell_matrix = copy.deepcopy(
                data_temp[set_cell_index, :].sum(axis=0).reshape(1, -1)
            )
            cellscore[i_num] = np.dot(cell_matrix, weights[cell_type]) / norm(
                cell_matrix
            )
            length_final[i_num] = len(cells_final[cell_id])
            final_data[i_num] = copy.deepcopy(cell_matrix)

        return cells_final, final_data, cellscore, length_final


def return_celltype_plot(adata_sc, so, cluster_name="leiden"):

    """
    Generates a cell type assignment array based on clustering results for visualization.

    This function creates an array where each pixel in the segmentation corresponds to a cell type
    cluster. It maps cell IDs from the spatial segmentation to their assigned cluster types obtained
    from clustering analysis (e.g., Leiden clustering) in the `adata_sc` AnnData object.

    :param adata_sc:
        An AnnData object containing single-cell gene expression data and clustering results.
    :type adata_sc: anndata.AnnData

    :param so:
        A spatial object containing the segmentation data and mappings between spatial coordinates
        and cell IDs.
    :type so: spatial_object

    :param cluster_name:
        The key in `adata_sc.obs` that contains the cluster assignments. Defaults to `'leiden'`.
    :type cluster_name: str, optional

    :return:
        An array where each pixel corresponds to a cell type cluster, suitable for visualization.
    :rtype: numpy.ndarray

    :example:
    >>> cell_type_final = return_celltype_plot(adata_sc, so, cluster_name='leiden')

    """

    # Initialize an array to hold the final cell type assignments
    cell_type_final = np.zeros(so.segmentation_final.shape)
    cell_type_dic = {}
    cell_ids = list(adata_sc.obs.index)
    cell_types = list(adata_sc.obs[cluster_name])

    # Create a dictionary mapping cell IDs to their cluster types (adding 1 to avoid zero indexing)
    for i in range(len(adata_sc)):
        cell_type_dic[float(cell_ids[i])] = int(cell_types[i]) + 1

    # Iterate over each pixel in the segmentation to assign cell types
    for i in tqdm.tqdm(range(cell_type_final.shape[0])):
        for j in range(cell_type_final.shape[1]):
            cell_id = so.segmentation_final[i, j]

            if cell_id != 0:
                try:
                    # Assign the cell type to the corresponding pixel in the final array
                    cell_type_final[i, j] = cell_type_dic[cell_id]
                except:
                    pass

    return cell_type_final


def plot_cellcluster_position(cell_cluster_final, col_num=5):

    """
    Plots the spatial distribution of cell clusters and individual cell types.

    This function visualizes the overall distribution of all cell clusters and creates separate
    plots for each individual cell type. It arranges the plots in a grid layout based on the
    specified number of columns.

    :param cell_cluster_final:
        A 2D NumPy array where each element corresponds to a cell cluster label at a particular spatial position.
        Cluster labels are integers starting from 0 (background or unassigned) up to the maximum number of clusters.
    :type cell_cluster_final: numpy.ndarray

    :param col_num:
        The number of columns to use in the grid layout for subplots. Determines how the plots are arranged.
        Defaults to `5`.
    :type col_num: int, optional

    :return:
        None. The function displays the plots and closes the figure after rendering.
    :rtype: None

    :example:
     >>> # Assuming 'cell_cluster_final' is your cluster label array
     >>> plot_cellcluster_position(cell_cluster_final, col_num=4)

    """

    # Determine the maximum cell type value
    max_type = int(np.max(cell_cluster_final))

    # Calculate total number of plots and the number of rows needed
    total_plots = max_type + 1
    rows = (total_plots + col_num - 1) // col_num

    # Get a colormap with enough colors for all cell types
    cmap = plt.get_cmap("tab20b", max_type + 2)

    # Create subplots with the calculated number of rows and columns
    fig, axs = plt.subplots(
        nrows=rows, ncols=col_num, figsize=(12 * col_num, 12 * rows)
    )

    # Plot the overall distribution of all cell types
    ax = axs[0, 0]
    im = ax.imshow(cell_cluster_final, cmap=cmap, vmin=1, vmax=max_type)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cell Cluster")
    cbar.set_ticks(np.arange(1, max_type + 1))
    ax.set_title("Distribution of All Cell Clusters", size=14)
    ax.axis("off")

    # Plot individual cell types
    for i in range(1, max_type + 1):
        ax = axs[(i) // col_num, (i) % col_num]
        ax.imshow(
            np.squeeze(np.array(cell_cluster_final == i)), cmap=cmap, vmin=0, vmax=1
        )
        ax.axis("off")
        ax.set_title("Cell cluster " + str(i - 1), size=14)

    # Turn off any unused subplots
    for i in range(total_plots, rows * col_num):
        axs[i // col_num, i % col_num].axis("off")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    plt.close()


def itering_arragement(
    adata_sc,
    adata_raw,
    adata,
    so,
    resolution=2,
    regress_out=True,
    save_folder="results_example/",
    show=True,
    keep_previous=False,
):

    """
    Performs iterative cell arrangement for spatial transcriptomics analysis.

    This function iteratively refines cell type assignments and spatial arrangements by expanding cells,
    updating weights, and recalculating clustering assignments. It aims to optimize cell type assignments
    based on mutual information scores between iterations.

    :param adata_sc:
        An AnnData object containing single-cell gene expression data with initial clustering results.
    :type adata_sc: anndata.AnnData

    :param adata_raw:
        An AnnData object containing the raw (unprocessed) single-cell gene expression data.
    :type adata_raw: anndata.AnnData

    :param adata:
        An AnnData object containing spatial gene expression data.
    :type adata: anndata.AnnData

    :param so:
        A spatial object containing spatial mappings, spot data, and other necessary attributes.
    :type so: spatial_object

    :param resolution:
        The resolution parameter for clustering during iterative analysis. Controls the granularity of clusters.
        Defaults to `2`.
    :type resolution: float, optional

    :param regress_out:
        Whether to regress out effects of total counts and mitochondrial percentage during preprocessing.
        Defaults to `True`.
    :type regress_out: bool, optional

    :param save_folder:
        The directory path where intermediate and final results will be saved. Defaults to `'results_example/'`.
    :type save_folder: str, optional

    :param show:
        Whether to print progress messages and show plots during the iterative process. Defaults to `True`.
    :type show: bool, optional

    :param keep_previous:
        Whether to keep intermediate files from previous iterations. If `False`, old files will be deleted to save space.
        Defaults to `False`.
    :type keep_previous: bool, optional

    :return:
        None. The function saves intermediate and final results to the specified folder.
    :rtype: None

    """

    new_length = np.zeros((so.indices.shape[0], 2))
    for i in range(so.indices.shape[0]):
        cell_id = so.cell_ids[i]
        new_length[i, 0] = cell_id
        new_length[i, 1] = so.length_main[cell_id]

    adata_type_record = {}

    adata_type_record[0] = list(adata_sc.obs["leiden"])
    weights_record = {}

    # Initialize weights based on the mean expression of each cluster
    weights = np.zeros((len(np.unique(adata_type_record[0])), adata_raw.shape[1]))
    for i in range(len(np.unique(adata_type_record[0]))):
        weights[i] = adata_raw[adata_sc.obs.leiden == str(i)].X.mean(axis=0)

    weights = weights / norm(weights, axis=1).reshape(-1, 1)
    weights_record[0] = weights

    mi_0 = 0
    adatas = {}
    adata_temp = adata_sc
    mis = [0]

    # Recalculate size of cells
    new_length = np.zeros((so.indices.shape[0], 2))
    for i in range(so.indices.shape[0]):
        cell_id = so.cell_ids[i]
        new_length[i, 0] = cell_id
        new_length[i, 1] = so.length_main[cell_id]

    # Compute the initial iterations required for each cell
    re = so.distances[:, 1:4].mean(axis=1) - 2 * np.sqrt(new_length[:, 1] / np.pi)
    iter_cells = {}
    for i in range(so.indices.shape[0]):
        cell_id = so.cell_ids[i]
        iter_cells[cell_id] = re[i]
        if re[i] < 0:
            iter_cells[cell_id] = 1
        else:
            iter_cells[cell_id] = 2

    max_mi = 0
    max_mi_i = 0

    # Begin iterative arrangement
    for i in range(1, 30):

        # Expand cells and update data
        cells_final, final_data, cellscore, length_final = expanding_cells(
            so,
            adata_temp,
            weights,
            iter_cells,
            data_temp=csr_matrix(adata.X),
            min_percent=0.4,
            cortex=False,
        )
        adata_temp = anndata.AnnData(
            X=csr_matrix(final_data),
            obs=pd.DataFrame([], index=list(adata_sc.obs.index)),
            var=adata.var,
        )
        adata_temp.write(save_folder + "adatas_ini_" + str(i) + ".h5ad")
        adata_temp = singlecellanalysis(
            adata_temp,
            True,
            i,
            save_folder,
            regress_out=regress_out,
            resolution=resolution,
            show=show,
        )
        adata_type_record[i] = list(adata_temp.obs["leiden"])

        # Calculate mutual information score
        mi = normalized_mutual_info_score(
            adata_type_record[i], adata_type_record[i - 1]
        )
        mis.append(mi)
        with open(save_folder + "mis.pkl", "wb") as f:
            pickle.dump(mis, f)

        if show:
            print(mi, i)
        if (i > max_mi_i + 1) and max_mi > mi:
            # If mutual information decreases, restore previous best state
            with open(save_folder + "weights_record.pkl", "wb") as f:
                pickle.dump(weights_record[max_mi_i], f)

            os.rename(
                save_folder + "adatas_ini_" + str(max_mi_i) + ".h5ad",
                save_folder + "adatas_ini.h5ad",
            )
            os.rename(
                save_folder + "adatas_" + str(max_mi_i) + ".h5ad",
                save_folder + "adatas.h5ad",
            )
            os.rename(
                save_folder + "cells_final_" + str(max_mi_i) + ".pkl",
                save_folder + "cells_final.pkl",
            )

            if keep_previous != True:
                # Clean up unnecessary files
                for j in range(i + 1):
                    try:
                        os.remove(save_folder + "adatas_ini_" + str(j) + ".h5ad")
                        os.remove(save_folder + "adatas_" + str(j) + ".h5ad")
                        os.remove(save_folder + "cells_final_" + str(j) + ".pkl")
                    except:
                        pass

            break

        else:
            # Update weights if mutual information increases
            if max_mi < mi:
                max_mi = mi

                adata_temp.write(save_folder + "adatas_" + str(i) + ".h5ad")
                with open(save_folder + "cells_final_" + str(i) + ".pkl", "wb") as f:
                    pickle.dump(cells_final, f)

                if keep_previous != True:
                    # Remove older files
                    for j in range(max_mi_i, i):
                        try:
                            os.remove(
                                save_folder + "adatas_ini_" + str(max_mi_i) + ".h5ad"
                            )
                            os.remove(save_folder + "adatas_" + str(max_mi_i) + ".h5ad")
                            os.remove(
                                save_folder + "cells_final_" + str(max_mi_i) + ".pkl"
                            )
                        except:
                            pass

                max_mi_i = i

            # Recalculate weights
            weights = np.zeros((len(np.unique(adata_type_record[i])), adata.shape[1]))
            for j in range(len(np.unique(adata_type_record[i]))):
                # weights[j] = csr_matrix(final_data)[adata_temp.obs.leiden == str(j)].mean(axis = 0)
                weights[j] = csr_matrix(final_data)[
                    (adata_temp.obs.leiden == str(j)).to_numpy()
                ].mean(axis=0)
            weights = weights / norm(weights, axis=1).reshape(-1, 1)
            weights_record[i] = weights


def make_pixels_cells(
    so,
    adata,
    cells_before_ml,
    spot_cell_dic,
    spots_id_dic,
    spots_id_dic_prop,
    nonzero_indices_dic,
    seed=42,
):

    """
    Assigns cells to pixels in the spatial data based on spot compositions.

    This function creates a new pixel-level cell assignment matrix by combining initial cell assignments
    with additional cell composition information. It normalizes the proportions and updates the spatial object
    with the new cell assignments.

    :param so:
        A spatial object containing spatial mappings, spot data, and other necessary attributes.
    :type so: spatial_object

    :param adata:
        An AnnData object containing spatial gene expression data.
    :type adata: anndata.AnnData

    :param cells_before_ml:
        A dictionary containing initial cell assignments before machine learning adjustments.
    :type cells_before_ml: dict

    :param spot_cell_dic:
        A dictionary mapping spots to cell composition data after machine learning adjustments.
    :type spot_cell_dic: dict

    :param spots_id_dic:
        A dictionary mapping spot IDs to their corresponding indices.
    :type spots_id_dic: dict

    :param spots_id_dic_prop:
        A dictionary containing the proportions of spots after adjustments.
    :type spots_id_dic_prop: dict

    :param nonzero_indices_dic:
        A dictionary of non-zero indices for each spot, indicating cell presence.
    :type nonzero_indices_dic: dict

    :param seed:
        The random seed for reproducibility when assigning cells to pixels. Defaults to `42`.
    :type seed: int, optional

    :return:
        The updated spatial object `so` with the new pixel-level cell assignments stored in `so.pixels_cells`.
    :rtype: spatial_object
    """

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Initialize the spots composition dictionary
    spots_composition = {}
    for i in so.index_toset.keys():
        spots_composition[i] = {}

    # Update the spots composition with cells before machine learning
    for i in cells_before_ml.keys():
        for j in cells_before_ml[i]:
            spots_composition[j[0]][i] = j[1]

    # Update the spots composition with additional cell information
    for i in spot_cell_dic.keys():
        for j in range(len(spot_cell_dic[i])):
            for k in range(len(spot_cell_dic[i][j])):
                spots_composition[spots_id_dic[i][j][0]][
                    nonzero_indices_dic[i][j][k]
                ] = (spot_cell_dic[i][j][k] * spots_id_dic_prop[i][j][0])

    # Normalize the proportions in each spot
    keys = list(spots_composition.keys())
    for i in keys:
        if len(spots_composition[i]) == 0:
            del spots_composition[i]
        elif abs(sum(spots_composition[i].values()) - 1) > 0.0001:
            raise ValueError("Proportion Sum for spot" + str(i) + " is not 1.")
        else:
            for k in spots_composition[i].keys():
                spots_composition[i][k] = spots_composition[i][k] / sum(
                    spots_composition[i].values()
                )

    keys = list(spots_composition.keys())
    for i in keys:
        for k in spots_composition[i].keys():
            spots_composition[i][k] = spots_composition[i][k] / sum(
                spots_composition[i].values()
            )

    # Merge dataframes to align indices
    df1 = copy.deepcopy(adata.obs)
    df1["barcode"] = df1.index

    df2 = copy.deepcopy(so.df)
    df2["index"] = df2.index

    df = pd.merge(df1, df2, on=["barcode"], how="inner")
    df_index = np.array(df["index"])

    # Create a new spots dictionary with updated indices
    spots_dict_new = {}

    for i in spots_composition.keys():
        spots_dict_new[df_index[i]] = spots_composition[i]

    # Initialize a new matrix for cell assignments
    original_matrix = so.pixels
    new_matrix = np.zeros_like(original_matrix, dtype=int)

    # Assign cells to pixels based on the spot compositions
    for i in tqdm.tqdm(range(original_matrix.shape[0])):
        for j in range(original_matrix.shape[1]):
            spot_id = original_matrix[i, j]
            if spot_id != -1 and spot_id in spots_dict_new:
                cell_dict = spots_dict_new[spot_id]
                if cell_dict:
                    cells = list(cell_dict.keys())
                    chosen_cell_id = np.random.choice(cells, p=list(cell_dict.values()))
                    new_matrix[i, j] = chosen_cell_id

    # Update the spatial object with the new cell assignments
    so.pixels_cells = new_matrix

    return so


def plot_image_with_overlay(
    ax,
    image,
    cells,
    uniques,
    colors=None,
    dpi=1500,
    transparency=0.6,
    transparent_background=False,
):

    matrix = cells.copy()
    matrix_mod_1 = np.zeros(cells.shape)
    matrix_mod_2 = np.zeros(cells.shape)
    matrix_mod_3 = np.zeros(cells.shape)

    # Apply modular arithmetic to spread out cell IDs across color channels
    if colors is None:
        nums = [9, 5, 7]
        mask = matrix > 0
        matrix_mod_1[mask] = (240 / nums[0] * (matrix[mask] % nums[0])) + 1
        matrix_mod_2[mask] = (240 / nums[1] * (matrix[mask] % nums[1])) + 1
        matrix_mod_3[mask] = (240 / nums[2] * (matrix[mask] % nums[2])) + 1
    else:
        t = 0
        for i in uniques:
            if i == 0:
                continue
            else:
                ct = np.array(cells == i)
                matrix_mod_1[ct] = colors[t][0]
                matrix_mod_2[ct] = colors[t][1]
                matrix_mod_3[ct] = colors[t][2]
                t += 1

    # Stack the modified matrices to create an RGB image
    rgb_matrix = np.stack((matrix_mod_1, matrix_mod_2, matrix_mod_3), axis=-1)
    rgb_matrix = rgb_matrix.astype(np.uint8)

    # Stack the modified matrices to create an RGB image
    rgb_matrix = np.stack((matrix_mod_1, matrix_mod_2, matrix_mod_3), axis=-1)
    rgb_matrix = rgb_matrix.astype(np.uint8)

    if transparent_background:
        # Create an alpha channel
        alpha_channel = (
            np.ones(cells.shape) * transparency * 255
        )  # Multiply by 255 for uint8
        alpha_channel[
            cells == 0
        ] = 0  # Set alpha to 0 where cells == 0 (fully transparent)
    else:
        # Set RGB values to 255 (white) where cells == 0
        rgb_matrix[cells == 0] = 255
        # Create an alpha channel with uniform transparency
        alpha_channel = (
            np.ones(cells.shape) * transparency * 255
        )  # Multiply by 255 for uint8

    # Stack RGB and alpha channels to create an RGBA image
    rgba_matrix = np.dstack((rgb_matrix, alpha_channel.astype(np.uint8)))

    ax.axis("off")
    ax.imshow(image)
    ax.imshow(rgba_matrix, interpolation="nearest")


def plot_colors_with_indices(ax, colors):

    # Plot each color as a rectangle with its index
    for i, color in enumerate(colors):
        color_normalized = [x / 255 for x in color]
        rect = patches.Rectangle(
            (i, 0), 1, 1, linewidth=1, edgecolor="none", facecolor=color_normalized
        )
        ax.add_patch(rect)
        ax.text(
            i + 0.5,
            0.5,
            str(i + 1),
            color="white" if sum(color_normalized[:3]) < 1.5 else "black",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis("off")


def plot_pixels_cells(
    image,
    cells,
    uniques,
    colors=None,
    dpi=1500,
    transparency=0.6,
    figsize=(20, 20),
    save=None,
    transparent_background=False,
):

    # Prepare matrices for RGB channels
    matrix = cells.copy()
    matrix_mod_1 = np.zeros(cells.shape)
    matrix_mod_2 = np.zeros(cells.shape)
    matrix_mod_3 = np.zeros(cells.shape)

    if colors is None:
        nums = [9, 5, 7]
        mask = matrix > 0
        matrix_mod_1[mask] = (240 / nums[0] * (matrix[mask] % nums[0])) + 1
        matrix_mod_2[mask] = (240 / nums[1] * (matrix[mask] % nums[1])) + 1
        matrix_mod_3[mask] = (240 / nums[2] * (matrix[mask] % nums[2])) + 1
    else:
        t = 0
        if len(uniques) - 1 <= len(colors):
            for i in uniques:
                if i == 0:
                    continue
                else:
                    ct = np.array(cells == i)
                    matrix_mod_1[ct] = colors[t][0]
                    matrix_mod_2[ct] = colors[t][1]
                    matrix_mod_3[ct] = colors[t][2]
                    t += 1
        else:
            raise TypeError("Please input colors with enough units.")

    # Stack the modified matrices to create an RGB image
    rgb_matrix = np.stack((matrix_mod_1, matrix_mod_2, matrix_mod_3), axis=-1)
    rgb_matrix = rgb_matrix.astype(np.uint8)

    if transparent_background:
        # Create an alpha channel
        alpha_channel = (
            np.ones(cells.shape) * transparency * 255
        )  # Multiply by 255 for uint8
        alpha_channel[
            cells == 0
        ] = 0  # Set alpha to 0 where cells == 0 (fully transparent)
    else:
        # Set RGB values to 255 (white) where cells == 0
        rgb_matrix[cells == 0] = 255
        # Create an alpha channel with uniform transparency
        alpha_channel = (
            np.ones(cells.shape) * transparency * 255
        )  # Multiply by 255 for uint8

    # Stack RGB and alpha channels to create an RGBA image
    rgba_matrix = np.dstack((rgb_matrix, alpha_channel.astype(np.uint8)))

    # Plot the original image with the cell overlay
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    # Overlay the RGBA image
    ax.imshow(rgba_matrix, interpolation="nearest")

    plt.axis("off")

    # Save and display the figure
    if save not in [False, None]:
        plt.savefig(save, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()


def plot_whole(
    img_np,
    cell_type_final,
    uniques,
    colors,
    dpi=1000,
    transparency=0.6,
    figsize=(20, 20),
    transparent_background=False,
    save=None,
):

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 20]}
    )

    fig.subplots_adjust(hspace=0.05)

    # Plot colors with indices
    plot_colors_with_indices(ax1, colors)

    # Plot image with overlay
    plot_image_with_overlay(
        ax2,
        img_np,
        cell_type_final,
        uniques,
        colors,
        dpi,
        transparency=transparency,
        transparent_background=transparent_background,
    )

    # Save the figure
    if (save != False) and (save is not None):
        plt.savefig(save, dpi=dpi)
    plt.show()
    plt.close()


def plot_results(
    original_image,
    result_image,
    transparency=0.6,
    transparent_background=False,
    include_label=None,
    colors=None,
    dpi=1500,
    figsize=(20, 20),
    save=None,
):

    """
    Plots the original tissue image with cell type assignments overlayed.

    This function visualizes the results of cell type assignments by overlaying them on the original
    tissue image. It handles cases where there are many cell types and adjusts the plot accordingly.
    Optionally, it includes a legend with cell type labels.

    :param original_image:
        The original tissue image as a NumPy array (normally use `so.image_temp()`).
    :type original_image: numpy.ndarray

    :param result_image:
        An array where each pixel corresponds to a cell type label after processing (normally use `so.pixels_cells`).
    :type result_image: numpy.ndarray

    :param transparency:
        The transparency level of the cell type overlay. Must be between 0 and 1. Defaults to `0.6`.
    :type transparency: float, optional

    :param transparent_background:
        Whether to use a transparent background for the overlay. Defaults to `False`.
    :type transparent_background: bool, optional

    :param include_label:
        Whether to include a legend with cell type labels. If `None`, the function decides based on the number of cell types.
        Defaults to `None`.
    :type include_label: bool or None, optional

    :param colors:
        A list of RGB tuples representing colors for each cell type. If `None`, a default color palette is used for up to 50 cell types.
        Defaults to `None`.
    :type colors: list of tuples or None, optional

    :param dpi:
        The resolution of the plot in dots per inch. Defaults to `1500`.
    :type dpi: int, optional

    :param figsize:
        The size of the figure in inches (width, height). Defaults to `(20, 20)`.
    :type figsize: tuple, optional

    :param save:
        The file path to save the figure. If `None`, the figure will not be saved. Defaults to `None`.
    :type save: str or None, optional

    :return:
        None. The function displays the plot and optionally saves it to a file.
    :rtype: None

    """

    if transparency <= 0 or transparency > 1:
        raise TypeError("Please input transparency in (0,1].")

    uniques = np.unique(result_image)
    number = len(uniques) - 1

    if include_label == True:
        if number > 50:
            print("Too many cell labels. include_label will be False")
            include_label = False
    elif include_label != False:
        if number > 50:
            include_label = False
        else:
            include_label = True

    if colors != None:
        if number > len(colors):
            print("Unique cell labels larger than len(colors). ")
            colors = None

    if (colors == None) and (number <= 50):
        colors = [
            (172, 106, 88),
            (203, 205, 199),
            (135, 36, 197),
            (138, 76, 124),
            (208, 67, 111),
            (195, 190, 120),
            (53, 137, 24),
            (218, 161, 52),
            (212, 229, 95),
            (214, 154, 126),
            (201, 147, 47),
            (96, 35, 35),
            (38, 116, 230),
            (69, 170, 54),
            (223, 96, 36),
            (197, 213, 195),
            (29, 101, 119),
            (186, 109, 121),
            (127, 202, 188),
            (76, 216, 34),
            (108, 190, 202),
            (75, 64, 110),
            (109, 76, 90),
            (141, 92, 60),
            (74, 84, 82),
            (45, 166, 110),
            (93, 149, 201),
            (100, 68, 27),
            (167, 158, 108),
            (157, 151, 133),
            (24, 52, 56),
            (47, 51, 169),
            (123, 113, 91),
            (103, 162, 189),
            (176, 28, 225),
            (129, 49, 52),
            (114, 103, 104),
            (69, 118, 27),
            (25, 104, 216),
            (71, 117, 103),
            (200, 110, 26),
            (143, 48, 142),
            (163, 211, 182),
            (140, 149, 222),
            (223, 68, 30),
            (208, 155, 75),
            (132, 190, 119),
            (223, 67, 163),
            (170, 181, 219),
            (145, 220, 27),
        ][:number]

    if include_label == False:
        plot_pixels_cells(
            original_image,
            result_image,
            uniques,
            colors=colors,
            dpi=dpi,
            figsize=figsize,
            transparency=transparency,
            transparent_background=transparent_background,
            save=save,
        )
    else:
        plot_whole(
            original_image,
            result_image,
            uniques,
            colors=colors,
            dpi=dpi,
            figsize=figsize,
            transparency=transparency,
            transparent_background=transparent_background,
            save=save,
        )
