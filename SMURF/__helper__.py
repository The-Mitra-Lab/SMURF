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
    for i in range(len(indices)):
        pixels[row_starts[i] : row_ends[i], col_starts[i] : col_ends[i]] = indices[i]

    return pixels


def prepare_dataframe_image(
    df_path: "square_002um/spatial/tissue_positions.parquet",
    img_path: "Visium_HD_Mouse_Small_Intestine_tissue_image.btf",
    row_number=3350,
    col_nuber=3350,
):

    # read image
    Image.MAX_IMAGE_PIXELS = None  # Removes the limit entirely

    try:
        # Open the image with Pillow
        image = Image.open(img_path)
        image_array = np.array(image)

    # print('The shape of original image is': image_array.shape)

    except IOError as e:

        print(f"Error opening or processing image: {e}")

    df = pd.read_parquet(df_path, engine="pyarrow")

    avg_row = (df["pxl_row_in_fullres"].max() - df["pxl_row_in_fullres"].min()) / (
        2 * row_number
    )
    avg_col = (df["pxl_col_in_fullres"].max() - df["pxl_col_in_fullres"].min()) / (
        2 * col_nuber
    )
    df["pxl_row_left_in_fullres"] = df["pxl_row_in_fullres"] - avg_row
    df["pxl_row_right_in_fullres"] = df["pxl_row_in_fullres"] + avg_row
    df["pxl_col_up_in_fullres"] = df["pxl_col_in_fullres"] - avg_col
    df["pxl_col_down_in_fullres"] = df["pxl_col_in_fullres"] + avg_col
    df["pxl_row_left_in_fullres"] = df["pxl_row_left_in_fullres"].round().astype(int)
    df["pxl_row_right_in_fullres"] = df["pxl_row_right_in_fullres"].round().astype(int)
    df["pxl_col_up_in_fullres"] = df["pxl_col_up_in_fullres"].round().astype(int)
    df["pxl_col_down_in_fullres"] = df["pxl_col_down_in_fullres"].round().astype(int)

    start_row_spot = df[(df["in_tissue"] == 1)]["array_row"].min()
    end_row_spot = df[(df["in_tissue"] == 1)]["array_row"].max() + 1
    start_col_spot = df[(df["in_tissue"] == 1)]["array_col"].min()
    end_col_spot = df[(df["in_tissue"] == 1)]["array_col"].max() + 1

    df_temp = df[
        (df["array_row"] >= start_row_spot)
        & (df["array_row"] < end_row_spot)
        & (df["array_col"] >= start_col_spot)
        & (df["array_col"] < end_col_spot)
    ].copy()

    row_left = max(df_temp["pxl_row_left_in_fullres"].min(), 0)
    row_right = df_temp["pxl_row_right_in_fullres"].max()
    col_up = max(df_temp["pxl_col_up_in_fullres"].min(), 0)
    col_down = df_temp["pxl_col_down_in_fullres"].max()

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

    row_starts = df_temp["pxl_row_left_in_fullres_temp"].values
    row_ends = df_temp["pxl_row_right_in_fullres_temp"].values
    col_starts = df_temp["pxl_col_up_in_fullres_temp"].values
    col_ends = df_temp["pxl_col_down_in_fullres_temp"].values
    indices = df_temp.index.to_numpy()

    pixels = -1 * np.ones(image_array[row_left:row_right, col_up:col_down].shape[:2])
    pixels = fill_pixels(pixels, row_starts, row_ends, col_starts, col_ends, indices)

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
    )


def nuclei_rna(adata, so, min_percent=0.4):

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

    # for cell_id in [103]:
    cell_ids = copy.deepcopy(so.cell_ids)
    cell_ids = np.sort(np.array(cell_ids))
    for cell_id in tqdm.tqdm(cell_ids):

        set_cell = set()
        set_tobe = set()
        set_exclude = set()

        rows = []
        cols = []

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

        for row in set(rows):
            row_min[row] = 100000000
            row_max[row] = 0

        for col in set(cols):
            col_min[col] = 100000000
            col_max[col] = 0

        for i in range(len(so.cells_main[cell_id])):

            if data_cells[0, i] < col_min[data_cells[1, i]]:
                col_min[data_cells[1, i]] = data_cells[0, i]

            if data_cells[0, i] > col_max[data_cells[1, i]]:
                col_max[data_cells[1, i]] = data_cells[0, i]

            if data_cells[1, i] < row_min[data_cells[0, i]]:
                row_min[data_cells[0, i]] = data_cells[1, i]

            if data_cells[1, i] > row_max[data_cells[0, i]]:
                row_max[data_cells[0, i]] = data_cells[1, i]

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

        cell_matrix[cell_id] = [
            set_toindex_data[key] for key in set_cell if key in set_toindex_data
        ]
        cell_matrix[cell_id] = (
            data_temp[cell_matrix[cell_id], :].sum(axis=0).reshape(1, -1)
        )
        final_data[i_num] = cell_matrix[cell_id]
        i_num = i_num + 1

    so.set_toindex_data = copy.deepcopy(set_toindex_data)
    so.set_cells = copy.deepcopy(set_cells)
    so.set_tobes = copy.deepcopy(set_tobes)
    so.set_excludes = copy.deepcopy(set_excludes)
    so.length_main = copy.deepcopy(length_main)
    so.cell_matrix = copy.deepcopy(cell_matrix)

    cell_ids_str = [str(cell_id) for cell_id in cell_ids]

    so.final_nuclei = anndata.AnnData(
        X=copy.deepcopy(csr_matrix(final_data)),
        obs=pd.DataFrame([], index=cell_ids_str),
        var=adata.var,
    )

    return so


def singlecellanalysis(
    adata,
    save=False,
    i=None,
    path=None,
    resolution=1,
    regress_out=True,
    random_state=0,
    show=True,
):

    warnings.simplefilter("ignore", SparseEfficiencyWarning)
    warnings.filterwarnings("ignore", message="IOStream.flush timed out")

    if show:
        print("Starting mt")
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    if show:
        print("Starting normalization")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable].copy()
    if regress_out:
        sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(adata, max_value=10)

    if show:
        print("Starting PCA")
    sc.tl.pca(adata, svd_solver="arpack", random_state=random_state)

    if show:
        print("Starting UMAP")
    sc.pp.neighbors(
        adata,
        n_neighbors=10,
        n_pcs=40,
        random_state=random_state,
    )
    sc.tl.umap(adata)

    if show:
        print("Starting Clustering")
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        flavor="igraph",
        n_iterations=2,
    )

    if show:
        if save:
            sc.pl.umap(adata, color=["leiden"], save=str(i) + "_umap.png")
        elif save == False:
            sc.pl.umap(adata, color=["leiden"])
        else:
            sc.pl.umap(adata, color=["leiden"], save=save)

    return adata


def expanding_cells(
    so, adata_sc, weights, iter_cells, data_temp, min_percent=0.4, cortex=False
):

    cells_final = {}

    cell_ids = copy.deepcopy(list(adata_sc.obs.index.astype(float)))
    total_num = len(cell_ids)
    celltypes = copy.deepcopy(list(adata_sc.obs.leiden.astype(int)))
    cellscore = -1 * np.ones([total_num, 1])
    length_final = -1 * np.ones([total_num, 1])
    final_data = copy.deepcopy(lil_matrix((total_num, data_temp.shape[1])))

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

        set_cell = copy.deepcopy(so.set_cells[cell_id])
        set_tobe = copy.deepcopy(so.set_tobes[cell_id])
        set_exclude = copy.deepcopy(so.set_excludes[cell_id])
        cell_matrix = copy.deepcopy(so.cell_matrix[cell_id])

        score = (np.dot(cell_matrix, weights[cell_type]) / norm(cell_matrix))[0, 0]
        tobe_considered = set_tobe - set_cell - set_exclude
        i = 0

        while len(tobe_considered) != 0:
            for row, col in tobe_considered:
                cell_matrix_temp = (
                    data_temp[so.set_toindex_data[(row, col)], :] + cell_matrix
                )
                # cell_matrix_norm = 10000*cell_matrix_temp/cell_matrix_temp.sum()

                score_temp = (
                    np.dot(cell_matrix_temp, weights[cell_type])
                    / norm(cell_matrix_temp)
                )[0, 0]

                if (
                    score_temp > score
                ):  # data_temp[set_toindex[(row,col)],:].sum() == 0:
                    # print(score_temp)
                    set_cell.update([(row, col)])
                    cell_matrix = cell_matrix_temp

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

    segmentation_final = so.segmentation_final

    cell_type_final = np.zeros(segmentation_final.shape)
    cell_type_dic = {}
    cell_ids = list(adata_sc.obs.index)
    cell_types = list(adata_sc.obs[cluster_name])

    for i in range(len(adata_sc)):
        cell_type_dic[float(cell_ids[i])] = int(cell_types[i]) + 1

    for i in tqdm.tqdm(range(cell_type_final.shape[0])):
        for j in range(cell_type_final.shape[1]):
            cell_id = segmentation_final[i, j]

            if cell_id != 0:
                try:
                    cell_type_final[i, j] = cell_type_dic[cell_id]
                except:
                    1

    return cell_type_final


def plot_celltype_position(cell_type_final, col_num=5):

    max_type = int(np.max(cell_type_final))

    total_plots = max_type + 1
    rows = (total_plots + col_num - 1) // col_num

    cmap = plt.get_cmap("tab20b", max_type + 2)

    fig, axs = plt.subplots(
        nrows=rows, ncols=col_num, figsize=(12 * col_num, 12 * rows)
    )

    ax = axs[0, 0]
    im = ax.imshow(cell_type_final, cmap=cmap, vmin=1, vmax=max_type)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cell Type")
    cbar.set_ticks(np.arange(1, max_type + 1))
    ax.set_title("Distribution of All Cell Types", size=14)
    ax.axis("off")

    for i in range(1, max_type + 1):
        ax = axs[(i) // col_num, (i) % col_num]
        ax.imshow(np.squeeze(np.array(cell_type_final == i)), cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title("Cell type " + str(i), size=14)

    for i in range(total_plots, rows * col_num):
        axs[i // col_num, i % col_num].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_colors_with_indices(ax, colors=None):

    if colors == None:

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
        ]

    for i, color in enumerate(colors):
        color_normalized = [x / 255 for x in color]
        rect = patches.Rectangle(
            (i, 0), 1, 1, linewidth=1, edgecolor="none", facecolor=color_normalized
        )
        ax.add_patch(rect)
        ax.text(
            i + 0.5,
            0.5,
            str(i),
            color="white" if sum(color_normalized[:3]) < 1.5 else "black",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis("off")


def plot_image_with_overlay(ax, so, cell_type_final, colors):
    img_np = so.image_temp()

    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)

    if img_np.shape[2] == 3:
        img_np = np.dstack(
            [img_np, np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.uint8) * 255]
        )

    overlay = np.full(
        (img_np.shape[0], img_np.shape[1], 4), [0, 0, 0, 0], dtype=np.uint8
    )
    for i in tqdm.tqdm(range(1, int(cell_type_final.max()) + 1)):
        ct = np.array(cell_type_final == i)
        overlay[:, :, 0] = overlay[:, :, 0] + colors[i - 1][0] * ct
        overlay[:, :, 1] = overlay[:, :, 1] + colors[i - 1][1] * ct
        overlay[:, :, 2] = overlay[:, :, 2] + colors[i - 1][2] * ct
        overlay[:, :, 3] = overlay[:, :, 3] + int(255) * ct

    result = np.zeros_like(img_np)
    for c in range(4):
        result[:, :, c] = (
            img_np[:, :, c] * (1 - overlay[:, :, 3] / 255)
            + overlay[:, :, c] * (overlay[:, :, 3] / 255)
        ).astype(np.uint8)

    ax.imshow(result)


def plot_whole(so, cell_type_final, colors=None, save="result_whole_pic.pdf"):

    if colors == None:

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
        ]

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20.5, 20), gridspec_kw={"height_ratios": [1, 20]}
    )

    fig.subplots_adjust(hspace=0.05)

    # Plot colors with indices
    plot_colors_with_indices(ax1, colors[0 : int(cell_type_final.max())])

    # Plot image with overlay
    # You need to define `so` and `cell_type_final` properly here
    plot_image_with_overlay(ax2, so, cell_type_final, colors)

    # Save the figure
    plt.savefig(save, dpi=1000)
    plt.show()
    plt.close()


def itering_arragement(
    adata_sc,
    adata_raw,
    adata,
    so,
    resolution=1,
    regress_out=True,
    save_folder="results_example/",
    show=True,
    keep_previous=False,
):

    new_length = np.zeros((so.indices.shape[0], 2))
    for i in range(so.indices.shape[0]):
        cell_id = so.cell_ids[i]
        new_length[i, 0] = cell_id
        new_length[i, 1] = so.length_main[cell_id]

    adata_type_record = {}

    adata_type_record[0] = list(adata_sc.obs["leiden"])
    weights_record = {}

    weights = np.zeros((len(np.unique(adata_type_record[0])), adata_raw.shape[1]))
    for i in range(len(np.unique(adata_type_record[0]))):
        weights[i] = adata_raw[adata_sc.obs.leiden == str(i)].X.mean(axis=0)

    weights = weights / norm(weights, axis=1).reshape(-1, 1)
    weights_record[0] = weights

    mi_0 = 0
    adatas = {}
    adata_temp = adata_sc
    mis = [0]

    new_length = np.zeros((so.indices.shape[0], 2))
    for i in range(so.indices.shape[0]):
        cell_id = so.cell_ids[i]
        new_length[i, 0] = cell_id
        new_length[i, 1] = so.length_main[cell_id]

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

    for i in range(1, 30):

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
        mi = normalized_mutual_info_score(
            adata_type_record[i], adata_type_record[i - 1]
        )
        adata_temp.write(save_folder + "adatas_" + str(i) + ".h5ad")
        # final_Xs[i] = csr_matrix(final_data)
        mis.append(mi)
        with open(save_folder + "mis.pkl", "wb") as f:
            pickle.dump(mis, f)
        with open(save_folder + "cells_final_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(cells_final, f)

        if show:
            print(mi, i)
        if (i > max_mi_i + 1) and max_mi > mi:

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
                for j in range(max_mi_i + 1, i + 1):
                    try:
                        os.remove(save_folder + "adatas_ini_" + str(j) + ".h5ad")
                        os.remove(save_folder + "adatas_" + str(j) + ".h5ad")
                        os.remove(save_folder + "cells_final_" + str(j) + ".pkl")
                    except:
                        1

            break

        else:

            if max_mi < mi:
                max_mi = mi

                if keep_previous != True:
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
                            1

                max_mi_i = i

            weights = np.zeros((len(np.unique(adata_type_record[i])), adata.shape[1]))
            for j in range(len(np.unique(adata_type_record[i]))):
                weights[j] = csr_matrix(final_data)[
                    adata_temp.obs.leiden == str(j)
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

    np.random.seed(seed)
    spots_composition = {}
    for i in so.index_toset.keys():
        spots_composition[i] = {}

    for i in cells_before_ml.keys():
        for j in cells_before_ml[i]:
            spots_composition[j[0]][i] = j[1]

    for i in spot_cell_dic.keys():
        for j in range(len(spot_cell_dic[i])):
            for k in range(len(spot_cell_dic[i][j])):
                spots_composition[spots_id_dic[i][j][0]][
                    nonzero_indices_dic[i][j][k]
                ] = (spot_cell_dic[i][j][k] * spots_id_dic_prop[i][j][0])

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

    df1 = copy.deepcopy(adata.obs)
    df1["barcode"] = df1.index

    df2 = copy.deepcopy(so.df)
    df2["index"] = df2.index

    df = pd.merge(df1, df2, on=["barcode"], how="inner")
    df_index = np.array(df["index"])

    spots_dict_new = {}

    for i in spots_composition.keys():
        spots_dict_new[df_index[i]] = spots_composition[i]

    original_matrix = so.pixels
    new_matrix = np.zeros_like(original_matrix, dtype=int)

    for i in tqdm.tqdm(range(original_matrix.shape[0])):
        for j in range(original_matrix.shape[1]):
            spot_id = original_matrix[i, j]
            if spot_id != -1 and spot_id in spots_dict_new:
                cell_dict = spots_dict_new[spot_id]
                if cell_dict:
                    cells = list(cell_dict.keys())
                    chosen_cell_id = np.random.choice(cells, p=list(cell_dict.values()))
                    new_matrix[i, j] = chosen_cell_id

    so.pixels_cells = new_matrix

    return so


def plot_pixels_cells(
    image,
    cells,
    colors=None,
    dpi=1500,
    transparency=0.6,
    save="combined_image_bin2cell.pdf",
):

    import matplotlib.colors as mcolors

    matrix = copy.deepcopy(cells)
    matrix_mod_1 = copy.deepcopy(matrix)
    matrix_mod_2 = copy.deepcopy(matrix)
    matrix_mod_3 = copy.deepcopy(matrix)

    nums = [9, 5, 7]
    matrix_mod_1[matrix_mod_1 > 0] = (
        240 / nums[0] * (matrix_mod_1[matrix_mod_1 > 0] % nums[0])
    ) + 1
    matrix_mod_2[matrix_mod_2 > 0] = (
        240 / nums[1] * (matrix_mod_2[matrix_mod_2 > 0] % nums[1])
    ) + 1
    matrix_mod_3[matrix_mod_3 > 0] = (
        240 / nums[2] * (matrix_mod_3[matrix_mod_3 > 0] % nums[2])
    ) + 1

    rgb_matrix = np.stack((matrix_mod_1, matrix_mod_2, matrix_mod_3), axis=-1)

    rgb_matrix = (rgb_matrix).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.imshow(255 - rgb_matrix, interpolation="nearest", alpha=transparency)

    plt.axis("off")

    if save != False:
        plt.savefig(save, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.show()


def find_plot(so, image_format="he"):
    if image_format == "he":
        return so.image[so.row_left : so.row_right, so.col_up : so.col_down, :]
    elif image_format == "dapi":
        return so.image[so.row_left : so.row_right, so.col_up : so.col_down]
