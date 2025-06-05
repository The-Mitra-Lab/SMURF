import copy
import math

import anndata
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm
from numpy.linalg import norm
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, lil_matrix


def objective3(v, A, B, C, D):
    x, y, z = v
    return norm(A * x + B * y + C * z - D) ** 2


def objective2(v, A, B, C):
    x, y = v
    return norm(A * x + B * y - C) ** 2


def find_connected_groups(lst):
    from collections import defaultdict

    graph = defaultdict(list)
    for sublst in lst:
        for i in range(len(sublst)):
            for j in range(i + 1, len(sublst)):
                graph[sublst[i]].append(sublst[j])
                graph[sublst[j]].append(sublst[i])

    visited = set()

    def dfs(node, component):
        stack = [node]
        while stack:
            vertex = stack.pop()
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)

    groups = []
    for node in graph:
        if node not in visited:
            visited.add(node)
            component = [node]
            dfs(node, component)
            groups.append(sorted(component))

    return groups


def find_length_of_largest_list(list_of_lists):

    if not list_of_lists:
        return 0
    return max(len(inner_list) for inner_list in list_of_lists)


def make_preparation(
    cells_final, so, adatas_final, adata, weight_to_celltype, maximum_cells=10000
):

    """
    Prepares data for optimization by organizing cells and spots, calculating weights,
    and grouping cells for computational efficiency.

    This function processes the final cell assignments and spatial data to prepare inputs
    for optimization algorithms. It organizes cells and spots,
    calculates cell-type-specific weights, and groups cells to limit computational load,
    ensuring efficient processing especially when dealing with a large number of cells.

    :param cells_final:
        A dictionary mapping cell IDs to their final set of spots after expansion.
    :type cells_final: dict

    :param so:
        A spatial object containing spatial mappings, spot data, and other necessary attributes.
    :type so: spatial_object

    :param adatas_final:
        An AnnData object containing the final single-cell gene expression data after processing.
    :type adatas_final: anndata.AnnData

    :param adata:
        An AnnData object containing spatial gene expression data.
    :type adata: anndata.AnnData

    :param weight_to_celltype:
        A NumPy array where each row corresponds to a cell type and contains weight vectors
        used in the optimization.
    :type weight_to_celltype: numpy.ndarray

    :param maximum_cells:
        The maximum number of cells to include in a group for optimization. This parameter
        helps limit computational load by grouping cells accordingly. Defaults to `10000`.
    :type maximum_cells: int, optional

    :return:
        A tuple containing:

        - `pct_toml_dic` (dict): Dictionary containing spot IDs and their associated proportions and cell types.
        - `spots_X_dic` (dict): Dictionary of spot expression matrices for each group.
        - `celltypes_dic` (dict): Dictionary of cell-type-specific weight matrices for each group.
        - `cells_X_plus_dic` (dict): Dictionary of cell expression matrices for each group.
        - `nonzero_indices_dic` (dict): Dictionary of non-zero indices indicating cell presence in spots for each group.
        - `nonzero_indices_toml` (dict): Dictionary of updated non-zero indices with new IDs for optimization.
        - `cells_before_ml` (dict): Dictionary of cells and their assigned spots before machine learning adjustments.
        - `cells_before_ml_x` (dict): Dictionary of cell expression data aggregated before machine learning.
        - `groups_combined` (dict): Dictionary of cell groups formed to limit computational load.
        - `spots_id_dic` (dict): Dictionary of spot IDs for each group.
        - `spots_id_dic_prop` (dict): Dictionary of spot proportions for each group.

    """
    epsilon = 1e-8

    weight_to_celltype_norm = weight_to_celltype / weight_to_celltype.sum(
        axis=1
    ).reshape(-1, 1)

    x2 = np.array([0.5, 0.5])  # Initial guess for two variables
    x3 = np.array([0.33, 0.33, 0.34])  # Initial guess for three variables

    cons = {"type": "eq", "fun": lambda v: np.sum(v) - 1}

    bounds2 = [(0, 1), (0, 1)]  # Bounds for two variables
    bounds3 = [(0, 1), (0, 1), (0, 1)]  # Bounds for three variables

    spots_final_cells = {}

    for cell_id in cells_final:
        for i in cells_final[cell_id]:
            if i in so.set_toindex_data:
                index = so.set_toindex_data[i]
                if index not in spots_final_cells:
                    spots_final_cells[index] = [cell_id]
                else:
                    spots_final_cells[index].append(cell_id)

    cell_types = {}
    cell_ids = list(adatas_final.obs.index.astype(float))
    celltypes = list(adatas_final.obs.leiden.astype(int))
    for i in range(len(cell_ids)):
        cell_types[int(cell_ids[i])] = celltypes[i]

    cells_before_ml = {}
    for cell_id in so.cells_main.keys():
        cells_before_ml[cell_id] = []

    to_ml = []
    data_temp = csr_matrix(adata.X)

    # Process each spot based on the number of cell types it contains
    for spot_id in tqdm.tqdm(list(spots_final_cells.keys())):

        if len(spots_final_cells[spot_id]) == 1:
            # Single cell in the spot
            cells_before_ml[spots_final_cells[spot_id][0]].append([spot_id, 1])

        elif len(spots_final_cells[spot_id]) > 1:
            # Multiple cells in the spot
            cell_types_all = set([])
            for cell_id in spots_final_cells[spot_id]:
                cell_types_all = cell_types_all | set([cell_types[cell_id]])
            cell_types_all_list = list(cell_types_all)

            if len(cell_types_all) == 1:
                # All cells are of the same type
                # print(spot_id, spots_final_cells[spot_id], cell_types_all)
                to_ml.append(
                    [
                        spot_id,
                        spots_final_cells[spot_id],
                        1,
                        cell_types_all_list[0],
                        cell_types_all,
                    ]
                )

            elif len(cell_types_all) == 2:
                # Two different cell types
                A = weight_to_celltype_norm[cell_types_all_list[0]]
                B = weight_to_celltype_norm[cell_types_all_list[1]]
                C = data_temp[spot_id] / data_temp[spot_id].sum()
                if data_temp[spot_id].sum() > 0:
                    C = data_temp[spot_id] / data_temp[spot_id].sum()
                else:
                    C = data_temp[spot_id]
                x = minimize(
                    objective2,
                    x2,
                    args=(A, B, C),
                    method="SLSQP",
                    constraints=cons,
                    bounds=bounds2,
                ).x
                if len(cell_types_all) < len(spots_final_cells[spot_id]):
                    for i in [0, 1]:
                        if x[i] > 0.000001:
                            num = 0
                            cell_ids = []
                            for cell_id in spots_final_cells[spot_id]:
                                if cell_types[cell_id] == cell_types_all_list[i]:
                                    num = num + 1
                                    cell_ids.append(cell_id)
                            if num > 1:
                                #  print(spot_id, spots_final_cells[spot_id], cell_types_all, cell_types_all_list[i], x[i], cell_ids)
                                to_ml.append(
                                    [
                                        spot_id,
                                        cell_ids,
                                        x[i],
                                        cell_types_all,
                                        cell_types_all_list[i],
                                    ]
                                )
                            else:
                                cells_before_ml[cell_ids[0]].append(
                                    [
                                        spot_id,
                                        x[i],
                                        cell_types_all,
                                        cell_types_all_list[i],
                                    ]
                                )
                else:
                    #   for i in [0,1]:
                    for cell_id in spots_final_cells[spot_id]:
                        if cell_types[cell_id] == cell_types_all_list[0]:
                            if x[0] > 0.000001:
                                cells_before_ml[cell_id].append(
                                    [
                                        spot_id,
                                        x[0],
                                        cell_types_all,
                                        cell_types_all_list[0],
                                    ]
                                )
                        else:
                            if x[1] > 0.000001:
                                cells_before_ml[cell_id].append(
                                    [
                                        spot_id,
                                        x[1],
                                        cell_types_all,
                                        cell_types_all_list[1],
                                    ]
                                )

            elif len(cell_types_all) == 3:
                # Three different cell types
                A = weight_to_celltype_norm[cell_types_all_list[0]]
                B = weight_to_celltype_norm[cell_types_all_list[1]]
                C = weight_to_celltype_norm[cell_types_all_list[2]]
                if data_temp[spot_id].sum() > 0:
                    D = data_temp[spot_id] / data_temp[spot_id].sum()
                else:
                    D = data_temp[spot_id]
                x = minimize(
                    objective3,
                    x3,
                    args=(A, B, C, D),
                    method="SLSQP",
                    constraints=cons,
                    bounds=bounds3,
                ).x
                if len(cell_types_all) < len(spots_final_cells[spot_id]):
                    for i in [0, 1, 2]:
                        if x[i] > 0.000001:
                            num = 0
                            cell_ids = []
                            for cell_id in spots_final_cells[spot_id]:
                                if cell_types[cell_id] == cell_types_all_list[i]:
                                    num = num + 1
                                    cell_ids.append(cell_id)
                            if num > 1:
                                #  print(spot_id, spots_final_cells[spot_id], cell_types_all, cell_types_all_list[i], x[i], cell_ids)
                                to_ml.append(
                                    [
                                        spot_id,
                                        cell_ids,
                                        x[i],
                                        cell_types_all,
                                        cell_types_all_list[i],
                                    ]
                                )
                            else:
                                cells_before_ml[cell_ids[0]].append(
                                    [
                                        spot_id,
                                        x[i],
                                        cell_types_all,
                                        cell_types_all_list[i],
                                    ]
                                )
                else:
                    for cell_id in spots_final_cells[spot_id]:
                        if cell_types[cell_id] == cell_types_all_list[0]:
                            if x[0] > 0.000001:
                                cells_before_ml[cell_id].append(
                                    [
                                        spot_id,
                                        x[0],
                                        cell_types_all,
                                        cell_types_all_list[0],
                                    ]
                                )
                        elif cell_types[cell_id] == cell_types_all_list[1]:
                            if x[1] > 0.000001:
                                cells_before_ml[cell_id].append(
                                    [
                                        spot_id,
                                        x[1],
                                        cell_types_all,
                                        cell_types_all_list[1],
                                    ]
                                )
                        else:
                            if x[2] > 0.000001:
                                cells_before_ml[cell_id].append(
                                    [
                                        spot_id,
                                        x[2],
                                        cell_types_all,
                                        cell_types_all_list[2],
                                    ]
                                )

    # Prepare data for machine learning
    cells_before_ml_x = {}
    for cell_id in cells_before_ml.keys():
        cells_before_ml_x[cell_id] = np.zeros([data_temp.shape[1]])

    for cell_id in cells_before_ml.keys():
        for i in cells_before_ml[cell_id]:
            if len(i) == 2:
                # Full assignment of spot to cell
                cells_before_ml_x[cell_id] = (
                    cells_before_ml_x[cell_id] + data_temp[i[0]]
                )
            else:
                # Partial assignment based on calculated weights
                # print(cells_before_ml[cell_id])
                A = i[1] * weight_to_celltype_norm[i[3]]
                B = (1 - i[1]) * weight_to_celltype_norm[
                    list(set(i[2]) - set([i[3]]))[0]
                ]
                denominator = A + B
                denominator[denominator == 0] = epsilon
                X = np.nan_to_num(
                    np.rint(
                        (A / denominator).reshape(1, -1)
                        * np.array(data_temp[i[0]].todense())
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                cells_before_ml_x[cell_id] = cells_before_ml_x[cell_id] + X

    spots_toml = []

    for i in range(len(to_ml)):
        spots_toml.append(to_ml[i][0])

    nonzero_indices = {}

    for i in np.unique(celltypes):
        nonzero_indices[i] = []

    for i in range(len(to_ml)):
        ct = to_ml[i][4]
        if type(ct) == set:
            ct = list(ct)[0]

        nonzero_indices[ct].append(to_ml[i][1])

    for i in np.unique(celltypes):
        lst = nonzero_indices[i]
        groups = find_connected_groups(lst)
    #  print(len(groups),find_length_of_largest_list(groups))

    # Group cells to limit computational load
    groups_combined = {}
    total = 0
    start = 0

    for i in np.unique(celltypes):

        lst = nonzero_indices[i]
        groups = find_connected_groups(lst)
        for group in groups:
            if total == 0 and len(group) > 0:
                total = len(group)
                groups_combined[start] = group
            elif total + len(group) > maximum_cells:
                total = len(group)
                start = start + 1
                groups_combined[start] = group
            else:
                if len(group) > 0:
                    total = total + len(group)
                    groups_combined[start] = groups_combined[start] + group

    # Map cells to groups
    cells_to_groups = {}
    for group in groups_combined.keys():
        for cell in groups_combined[group]:
            cells_to_groups[cell] = group

    # Prepare dictionaries for grouped data
    cells_toml_dic = {}
    spots_toml_dic = {}
    nonzero_indices_dic = {}
    pct_toml_dic = {}

    for i in groups_combined.keys():
        cells_toml_dic[i] = []
        spots_toml_dic[i] = []
        nonzero_indices_dic[i] = []
        pct_toml_dic[i] = []

    for i in range(len(to_ml)):

        for j in to_ml[i][1]:
            ct = cells_to_groups[j]
            cells_toml_dic[ct].append(j)

        spots_toml_dic[ct].append(to_ml[i][0])
        nonzero_indices_dic[ct].append(to_ml[i][1])
        pct_toml_dic[ct].append([to_ml[i][0], to_ml[i][2], to_ml[i][3], to_ml[i][4]])

    # Map cells to IDs within groups
    cells_toid_toml_dic = {}
    id_tocells_toml_dic = {}

    for i in groups_combined.keys():
        cells_toid_toml_dic[i] = {}
        id_tocells_toml_dic[i] = {}

    for i in groups_combined.keys():
        num = 0
        for j in range(len(cells_toml_dic[i])):
            if int(cells_toml_dic[i][j]) not in cells_toid_toml_dic[i].keys():
                cells_toid_toml_dic[i][int(cells_toml_dic[i][j])] = num
                id_tocells_toml_dic[i][num] = int(cells_toml_dic[i][j])
                num += 1

    # Update nonzero indices with new IDs
    nonzero_indices_toml = copy.deepcopy(nonzero_indices_dic)

    for i in groups_combined.keys():
        for j in range(len(nonzero_indices_dic[i])):
            for k in range(len(nonzero_indices_dic[i][j])):
                nonzero_indices_toml[i][j][k] = cells_toid_toml_dic[i][
                    nonzero_indices_dic[i][j][k]
                ]

    # Prepare matrices for optimization
    cells_X_plus_dic = {}
    spots_X_dic = {}
    celltypes_dic = {}
    spots_id_dic = {}
    spots_id_dic_prop = {}

    # Prepare spot data
    for i in groups_combined.keys():
        cells_X_plus_dic[i] = np.zeros((len(cells_toid_toml_dic[i]), adata.shape[1]))
        celltypes_dic[i] = np.zeros((len(cells_toid_toml_dic[i]), adata.shape[1]))
        for j in range(len(cells_toid_toml_dic[i])):
            cell_id = id_tocells_toml_dic[i][j]
            cells_X_plus_dic[i][j] = np.array(cells_before_ml_x[cell_id])
            celltypes_dic[i][j] = weight_to_celltype[cell_types[cell_id]]

    for i in groups_combined.keys():
        spots_X_dic[i] = np.zeros((len(nonzero_indices_toml[i]), adata.shape[1]))
        spots_id_dic[i] = np.zeros((len(nonzero_indices_toml[i]), 1))
        spots_id_dic_prop[i] = np.zeros((len(nonzero_indices_toml[i]), 1))

        for j in range(len(nonzero_indices_toml[i])):

            spot_id = pct_toml_dic[i][j][0]
            spots_id_dic[i][j] = spot_id

            if pct_toml_dic[i][j][1] < 0.999999999:

                A = (
                    pct_toml_dic[i][j][1]
                    * weight_to_celltype_norm[pct_toml_dic[i][j][3]]
                )
                B = (1 - pct_toml_dic[i][j][1]) * weight_to_celltype_norm[
                    list(set(pct_toml_dic[i][j][2]) - set([pct_toml_dic[i][j][3]]))[0]
                ]
                denominator = A + B
                denominator[denominator == 0] = epsilon
                X = np.nan_to_num(
                    np.rint(
                        (A / denominator).reshape(1, -1)
                        * np.array(data_temp[spot_id].todense())
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                spots_X_dic[i][j] = X
                spots_id_dic_prop[i][j] = pct_toml_dic[i][j][1]

            else:
                spots_X_dic[i][j] = data_temp[spot_id].toarray()
                spots_id_dic_prop[i][j] = 1

    return (
        pct_toml_dic,
        spots_X_dic,
        celltypes_dic,
        cells_X_plus_dic,
        nonzero_indices_dic,
        nonzero_indices_toml,
        cells_before_ml,
        cells_before_ml_x,
        groups_combined,
        spots_id_dic,
        spots_id_dic_prop,
    )


def calculate_radius(area):
    if area < 0:
        raise ValueError("Area cannot be negative")
    return math.sqrt(area / math.pi)


def calculate_distances(A, C, size):
    AC = math.sqrt((C[0] - A[0]) ** 2 + (C[1] - A[1]) ** 2)
    if size > 0:
        return calculate_radius(size) / (AC + 1e-10)
    else:
        return 1e-10 / (AC + 1e-10)


def calculate_weight_to_celltype(adatas_final, adata, cells_final, so):

    """
    Calculates the weight matrix mapping cell types to gene expression profiles.

    This function computes the average gene expression profiles for each cell type based on the provided single-cell data and spatial data. The resulting weight matrix is normalized and can be used in downstream analyses, such as cell type proportion estimation.

    :param adatas_final:
        An AnnData object containing the final single-cell gene expression data after processing.
    :type adatas_final: anndata.AnnData

    :param adata:
        An AnnData object containing spatial gene expression data.
    :type adata: anndata.AnnData

    :param cells_final:
        A dictionary mapping cell IDs to their final set of spots after expansion.
    :type cells_final: dict

    :param so:
        A spatial object containing spatial mappings, spot data, and other necessary attributes.
    :type so: spatial_object

    :return:
        A NumPy array where each row corresponds to a cell type and contains the normalized average gene expression profile.
    :rtype: numpy.ndarray

    :example:
    >>> weight_to_celltype = calculate_weight_to_celltype(adatas_final, adata, cells_final, so)

    """

    cell_ids = list(adatas_final.obs.index.astype(float))

    data_temp = csr_matrix(adata.X)

    final_X = lil_matrix(np.zeros([len(adatas_final.obs), data_temp.shape[1]]))
    for i in tqdm.tqdm(range(len(adatas_final.obs))):
        set_cell_index = [
            so.set_toindex_data[key]
            for key in cells_final[cell_ids[int(i)]]
            if key in so.set_toindex_data
        ]
        final_X[int(i)] = lil_matrix(
            data_temp[set_cell_index, :].sum(axis=0).reshape(1, -1)
        )

    weight_to_celltype = np.zeros(
        (len(adatas_final.obs.leiden.unique()), adata.shape[1])
    )
    for i in range(len(adatas_final.obs.leiden.unique())):
        weight_to_celltype[i] = final_X[adatas_final.obs.leiden == str(i)].mean(axis=0)

    weight_to_celltype = weight_to_celltype / norm(weight_to_celltype, axis=1).reshape(
        -1, 1
    )

    return weight_to_celltype


def get_finaldata(
    adata,
    adatas_final,
    spot_cell_dic,
    weight_to_celltype,
    cells_before_ml,
    groups_combined,
    pct_toml_dic,
    nonzero_indices_dic,
    spots_X_dic=None,
    nonzero_indices_toml=None,
    cells_before_ml_x=None,
    so=None,
):

    """
    Combines cell and spot data after optimization to generate the final single-cell dataset.

    This function aggregates gene expression data from spots and assigns counts to individual cells based on the results of optimization algorithms. It generates a final AnnData object containing single-cell gene expression data, along with cell metadata such as cluster assignments and spatial coordinates.

    :param adata:
        An AnnData object containing spatial gene expression data.
    :type adata: anndata.AnnData

    :param adatas_final:
        An AnnData object containing the final single-cell gene expression data after processing.
    :type adatas_final: anndata.AnnData

    :param spot_cell_dic:
        A dictionary containing the proportion of each cell in each spot after optimization.
    :type spot_cell_dic: dict

    :param weight_to_celltype:
        A NumPy array where each row corresponds to a cell type and contains weight vectors used in the scoring function.
    :type weight_to_celltype: numpy.ndarray

    :param cells_before_ml:
        A dictionary of cells and their assigned spots before machine learning adjustments.
    :type cells_before_ml: dict

    :param groups_combined:
        A dictionary of cell groups formed to limit computational load.
    :type groups_combined: dict

    :param pct_toml_dic:
        Dictionary containing spot IDs and their associated proportions and cell types.
    :type pct_toml_dic: dict

    :param nonzero_indices_dic:
        Dictionary of non-zero indices indicating cell presence in spots for each group.
    :type nonzero_indices_dic: dict

    :param spots_X_dic:
        (Optional) Dictionary of spot expression matrices for each group. If not provided, it will be computed.
    :type spots_X_dic: dict, optional

    :param nonzero_indices_toml:
        (Optional) Dictionary of updated non-zero indices with new IDs for optimization.
    :type nonzero_indices_toml: dict, optional

    :param cells_before_ml_x:
        (Optional) Dictionary of cell expression data aggregated before machine learning.
    :type cells_before_ml_x: dict, optional

    :param so:
        (Optional) A spatial object containing spatial mappings and data. If provided, spatial coordinates will be added to the final dataset.
    :type so: spatial_object, optional

    :return:
        An AnnData object containing the final single-cell gene expression data, along with cell metadata.
    :rtype: anndata.AnnData

    """

    binnumbers = {}
    epsilon = 1e-8

    data_temp = csr_matrix(adata.X)

    # Initialize cells_before_ml_x if not provided
    if cells_before_ml_x == None:
        cells_before_ml_x = {}
        for cell_id in cells_before_ml.keys():
            cells_before_ml_x[cell_id] = np.zeros([data_temp.shape[1]])
            binnumbers[cell_id] = 0

        for cell_id in cells_before_ml.keys():
            for i in cells_before_ml[cell_id]:
                if i[1] > 0.9999999:
                    cells_before_ml_x[cell_id] = (
                        cells_before_ml_x[cell_id] + data_temp[i[0]]
                    )
                    binnumbers[cell_id] = binnumbers[cell_id] + 1
                else:
                    # print(i)
                    A = i[1] * weight_to_celltype[i[3]]
                    B = (1 - i[1]) * weight_to_celltype[
                        list(set(i[2]) - set([i[3]]))[0]
                    ]
                    denominator = A + B
                    denominator[denominator == 0] = epsilon
                    X = np.nan_to_num(
                        np.rint(
                            (A / denominator).reshape(1, -1)
                            * np.array(data_temp[i[0]].todense())
                        ),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    cells_before_ml_x[cell_id] = cells_before_ml_x[cell_id] + X
                    binnumbers[cell_id] = binnumbers[cell_id] + i[1]

    else:
        for cell_id in cells_before_ml.keys():
            binnumbers[cell_id] = 0

        for cell_id in cells_before_ml.keys():
            for i in cells_before_ml[cell_id]:
                if i[1] > 0.9999999:
                    binnumbers[cell_id] = binnumbers[cell_id] + 1
                else:
                    binnumbers[cell_id] = binnumbers[cell_id] + i[1]

    # Prepare spot data if not provided
    if spots_X_dic == None:

        if nonzero_indices_toml is not None:

            spots_X_dic = {}

            for i in groups_combined.keys():
                spots_X_dic[i] = np.zeros(
                    (len(nonzero_indices_toml[i]), adata.shape[1])
                )

                for j in range(len(nonzero_indices_toml[i])):

                    spot_id = pct_toml_dic[i][j][0]

                    if pct_toml_dic[i][j][1] < 0.999999999:

                        A = (
                            pct_toml_dic[i][j][1]
                            * weight_to_celltype[pct_toml_dic[i][j][3]]
                        )
                        B = (1 - pct_toml_dic[i][j][1]) * weight_to_celltype[
                            list(
                                set(pct_toml_dic[i][j][2])
                                - set([pct_toml_dic[i][j][3]])
                            )[0]
                        ]
                        denominator = A + B
                        denominator[denominator == 0] = epsilon
                        X = np.nan_to_num(
                            np.rint(
                                (A / denominator).reshape(1, -1)
                                * np.array(data_temp[spot_id].todense())
                            ),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        spots_X_dic[i][j] = X

                    else:
                        spots_X_dic[i][j] = data_temp[spot_id].toarray()

        else:

            raise Exception(
                "Please input either spots_X_dic or nonzero_indices_toml. We will use spots_X_dic in default."
            )

    cell_types = {}
    cell_ids = list(adatas_final.obs.index.astype(float))
    celltypes = list(adatas_final.obs.leiden.astype(int))
    for i in range(len(cell_ids)):
        cell_types[int(cell_ids[i])] = celltypes[i]

    np.random.seed(0)

    num_temp = np.zeros([data_temp.shape[1]])

    cellid_to_index = {}

    final_X = lil_matrix(np.zeros([len(adatas_final.obs), data_temp.shape[1]]))
    cell_index = list(adatas_final.obs.index)

    # Combine data from cells_before_ml_x
    for i in range(len(cell_index)):
        cell_id = float(cell_index[i])
        cellid_to_index[cell_id] = i
        final_X[i] = csr_matrix(cells_before_ml_x[cell_id])

    # Assign counts from optimization results to cells
    for i in groups_combined.keys():
        for j in range(len(pct_toml_dic[i])):
            for k in range(len(nonzero_indices_dic[i][j])):
                cell_id = nonzero_indices_dic[i][j][k]

                num_temp = np.zeros([data_temp.shape[1]])
                spot_data = spots_X_dic[i][j]
                indices = np.where(spot_data > 0)[0]
                num_temp[indices] = np.random.binomial(
                    spot_data[indices].astype(int),
                    spot_cell_dic[i][j][k],
                    size=len(indices),
                )
                col_indices = np.nonzero(num_temp)[0]
                values = num_temp[col_indices]
                final_X[cellid_to_index[cell_id], col_indices] = values
                binnumbers[cell_id] = (
                    binnumbers[cell_id] + spot_cell_dic[i][j][k] * pct_toml_dic[i][j][1]
                )

    # Prepare cell information
    if so == None:
        cell_info = np.zeros([len(adatas_final.obs), 3])

        for i in range(len(cell_index)):

            cell_id = float(cell_index[i])

            cell_info[i, 0] = cell_types[cell_id]
            cell_info[i, 2] = binnumbers[cell_id]
            if final_X[i].sum() > 0:
                a = np.dot(
                    final_X[i].tocsr().toarray(),
                    weight_to_celltype[cell_types[cell_id]],
                ) / norm(final_X[i].tocsr().toarray())
                cell_info[i, 1] = a[0]
            else:
                cell_info[i, 1] = 0

        adata_sc_final = anndata.AnnData(
            X=final_X.tocsr(),
            obs=pd.DataFrame(
                cell_info,
                columns=["cell_cluster", "cos_simularity", "cell_size"],
                index=cell_index,
            ),
            var=adata.var,
        )

    else:

        cell_info = np.zeros([len(adatas_final.obs), 5])

        for i in range(len(cell_index)):

            cell_id = float(cell_index[i])

            cell_info[i, 0] = cell_types[cell_id]
            cell_info[i, 2] = binnumbers[cell_id]
            cell_info[i, 3] = so.cell_centers[cell_id][0]
            cell_info[i, 4] = so.cell_centers[cell_id][1]

            if final_X[i].sum() > 0:
                a = np.dot(
                    final_X[i].tocsr().toarray(),
                    weight_to_celltype[cell_types[cell_id]],
                ) / norm(final_X[i].tocsr().toarray())
                cell_info[i, 1] = a[0]
            else:
                cell_info[i, 1] = 0

        adata_sc_final = anndata.AnnData(
            X=final_X.tocsr(),
            obs=pd.DataFrame(
                cell_info,
                columns=["cell_cluster", "cos_simularity", "cell_size", "x", "y"],
                index=cell_index,
            ),
            var=adata.var,
        )

    # adata_sc_final.X.eliminate_zeros()

    return adata_sc_final


def get_finaldata_fast(
    cells_final, so, adatas_final, adata, weight_to_celltype, plot=True
):

    """
    Quickly generates the final single-cell dataset by combining cell and spot data without using GPU resources.

    This function provides a faster alternative to generate the final single-cell gene expression dataset,
    suitable for users who do not have access to GPU resources. It aggregates gene expression data from spots
    and assigns counts to individual cells based on optimized proportions. Optionally, it can plot the spatial
    distribution of cells on the tissue image.

    :param cells_final:
        A dictionary mapping cell IDs to their final set of spots after expansion.
    :type cells_final: dict

    :param so:
        A spatial object containing spatial mappings, spot data, and other necessary attributes.
    :type so: spatial_object

    :param adatas_final:
        An AnnData object containing the final single-cell gene expression data after processing.
    :type adatas_final: anndata.AnnData

    :param adata:
        An AnnData object containing spatial gene expression data.
    :type adata: anndata.AnnData

    :param weight_to_celltype:
        A NumPy array where each row corresponds to a cell type and contains weight vectors used in the scoring function.
    :type weight_to_celltype: numpy.ndarray

    :param plot:
        Whether to plot the cells on the spatial map after processing. Defaults to `True`. This may cost some time.
    :type plot: bool, optional

    :return:
        An AnnData object containing the final single-cell gene expression data, along with cell metadata.
    :rtype: anndata.AnnData

    """

    weight_to_celltype_norm = weight_to_celltype / weight_to_celltype.sum(
        axis=1
    ).reshape(-1, 1)

    spots_composition = {}
    for i in so.index_toset.keys():
        spots_composition[i] = {}

    x2 = np.array([0.5, 0.5])
    x3 = np.array([0.33, 0.33, 0.34])

    cons = {"type": "eq", "fun": lambda v: np.sum(v) - 1}

    bounds2 = [(0, 1), (0, 1)]
    bounds3 = [(0, 1), (0, 1), (0, 1)]

    spots_final_cells = {}
    binnumbers = {}

    for cell_id in cells_final:
        binnumbers[cell_id] = 0
        for i in cells_final[cell_id]:
            if i in so.set_toindex_data:
                index = so.set_toindex_data[i]
                if index not in spots_final_cells:
                    spots_final_cells[index] = [cell_id]
                else:
                    spots_final_cells[index].append(cell_id)

    cell_types = {}
    cell_ids = list(adatas_final.obs.index.astype(float))
    celltypes = list(adatas_final.obs.leiden.astype(int))
    for i in range(len(cell_ids)):
        cell_types[int(cell_ids[i])] = celltypes[i]

    cells_before_ml = {}
    for cell_id in so.cells_main.keys():
        cells_before_ml[cell_id] = []

    to_ml = []
    data_temp = csr_matrix(adata.X)

    for spot_id in tqdm.tqdm(list(spots_final_cells.keys())):
        if len(spots_final_cells[spot_id]) == 1:
            cells_before_ml[spots_final_cells[spot_id][0]].append([spot_id, 1])

        if len(spots_final_cells[spot_id]) > 1:
            cell_types_all = set([])
            for cell_id in spots_final_cells[spot_id]:
                cell_types_all = cell_types_all | set([cell_types[cell_id]])

            cell_types_all_list = list(cell_types_all)
            if len(cell_types_all) == 1:
                # print(spot_id, spots_final_cells[spot_id], cell_types_all)
                to_ml.append(
                    [
                        spot_id,
                        spots_final_cells[spot_id],
                        1,
                        cell_types_all_list[0],
                        cell_types_all,
                    ]
                )
            if len(cell_types_all) == 2:
                A = weight_to_celltype_norm[cell_types_all_list[0]]
                B = weight_to_celltype_norm[cell_types_all_list[1]]
                if data_temp[spot_id].sum() > 0:
                    C = data_temp[spot_id] / data_temp[spot_id].sum()
                else:
                    C = data_temp[spot_id]
                x = minimize(
                    objective2,
                    x2,
                    args=(A, B, C),
                    method="SLSQP",
                    constraints=cons,
                    bounds=bounds2,
                ).x
                if len(cell_types_all) < len(spots_final_cells[spot_id]):
                    for i in [0, 1]:
                        if x[i] > 0.000001:
                            num = 0
                            cell_ids = []
                            for cell_id in spots_final_cells[spot_id]:
                                if cell_types[cell_id] == cell_types_all_list[i]:
                                    num = num + 1
                                    cell_ids.append(cell_id)
                            if num > 1:
                                #  print(spot_id, spots_final_cells[spot_id], cell_types_all, cell_types_all_list[i], x[i], cell_ids)
                                to_ml.append(
                                    [
                                        spot_id,
                                        cell_ids,
                                        x[i],
                                        cell_types_all,
                                        cell_types_all_list[i],
                                    ]
                                )
                            else:
                                cells_before_ml[cell_ids[0]].append(
                                    [
                                        spot_id,
                                        x[i],
                                        cell_types_all,
                                        cell_types_all_list[i],
                                    ]
                                )

                else:
                    for cell_id in spots_final_cells[spot_id]:
                        if cell_types[cell_id] == cell_types_all_list[0]:
                            if x[0] > 0.000001:
                                cells_before_ml[cell_id].append(
                                    [
                                        spot_id,
                                        x[0],
                                        cell_types_all,
                                        cell_types_all_list[0],
                                    ]
                                )
                        else:
                            if x[1] > 0.000001:
                                cells_before_ml[cell_id].append(
                                    [
                                        spot_id,
                                        x[1],
                                        cell_types_all,
                                        cell_types_all_list[1],
                                    ]
                                )

            elif len(cell_types_all) == 3:
                A = weight_to_celltype_norm[cell_types_all_list[0]]
                B = weight_to_celltype_norm[cell_types_all_list[1]]
                C = weight_to_celltype_norm[cell_types_all_list[2]]
                if data_temp[spot_id].sum() > 0:
                    D = data_temp[spot_id] / data_temp[spot_id].sum()
                else:
                    D = data_temp[spot_id]
                x = minimize(
                    objective3,
                    x3,
                    args=(A, B, C, D),
                    method="SLSQP",
                    constraints=cons,
                    bounds=bounds3,
                ).x
                if len(cell_types_all) < len(spots_final_cells[spot_id]):
                    for i in [0, 1, 2]:
                        if x[i] > 0.000001:
                            num = 0
                            cell_ids = []
                            for cell_id in spots_final_cells[spot_id]:
                                if cell_types[cell_id] == cell_types_all_list[i]:
                                    num = num + 1
                                    cell_ids.append(cell_id)
                            if num > 1:
                                #  print(spot_id, spots_final_cells[spot_id], cell_types_all, cell_types_all_list[i], x[i], cell_ids)
                                to_ml.append(
                                    [
                                        spot_id,
                                        cell_ids,
                                        x[i],
                                        cell_types_all,
                                        cell_types_all_list[i],
                                    ]
                                )
                            else:
                                cells_before_ml[cell_ids[0]].append(
                                    [
                                        spot_id,
                                        x[i],
                                        cell_types_all,
                                        cell_types_all_list[i],
                                    ]
                                )
                else:
                    for cell_id in spots_final_cells[spot_id]:
                        if cell_types[cell_id] == cell_types_all_list[0]:
                            if x[0] > 0.000001:
                                cells_before_ml[cell_id].append(
                                    [
                                        spot_id,
                                        x[0],
                                        cell_types_all,
                                        cell_types_all_list[0],
                                    ]
                                )
                        elif cell_types[cell_id] == cell_types_all_list[1]:
                            if x[1] > 0.000001:
                                cells_before_ml[cell_id].append(
                                    [
                                        spot_id,
                                        x[1],
                                        cell_types_all,
                                        cell_types_all_list[1],
                                    ]
                                )
                        else:
                            if x[2] > 0.000001:
                                cells_before_ml[cell_id].append(
                                    [
                                        spot_id,
                                        x[2],
                                        cell_types_all,
                                        cell_types_all_list[2],
                                    ]
                                )

    for i in cells_before_ml.keys():
        for j in cells_before_ml[i]:
            spots_composition[j[0]][i] = j[1]

    rows = list(so.df[so.df.in_tissue == 1].array_row)
    cols = list(so.df[so.df.in_tissue == 1].array_col)

    cells_x = {}
    for cell_id in cells_before_ml.keys():
        cells_x[cell_id] = np.zeros([data_temp.shape[1]])

    epsilon = 1e-8
    for cell_id in cells_before_ml.keys():
        for i in cells_before_ml[cell_id]:

            if i[1] > 0.9999999:
                binnumbers[cell_id] = binnumbers[cell_id] + 1
                cells_x[cell_id] = cells_x[cell_id] + data_temp[i[0]]
            else:
                # print(i)
                A = i[1] * weight_to_celltype_norm[i[3]]
                B = (1 - i[1]) * weight_to_celltype_norm[
                    list(set(i[2]) - set([i[3]]))[0]
                ]
                denominator = A + B
                denominator[denominator == 0] = epsilon
                X = np.nan_to_num(
                    np.rint(
                        (A / denominator).reshape(1, -1)
                        * np.array(data_temp[i[0]].todense())
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                cells_x[cell_id] = cells_x[cell_id] + X
                binnumbers[cell_id] = binnumbers[cell_id] + i[1]

    ratios = []
    for i in range(len(to_ml)):
        new = []
        for j in range(len(to_ml[i][1])):
            new.append(
                calculate_distances(
                    (rows[to_ml[i][0]], cols[to_ml[i][0]]),
                    so.cell_centers[to_ml[i][1][j]],
                    binnumbers[to_ml[i][1][j]],
                )
            )
        new = np.array(new)
        new = new / new.sum()
        ratios.append(new)

    np.random.seed(0)
    for i in range(len(to_ml)):

        if to_ml[i][2] > 0.9999999:
            spot_data = data_temp[to_ml[i][0]]
            spot_data = np.array(spot_data.todense())[0]

        else:
            #   print(spot_data.shape)
            A = to_ml[i][2] * weight_to_celltype_norm[to_ml[i][4]]
            B = (1 - to_ml[i][2]) * weight_to_celltype_norm[
                list(set(to_ml[i][3]) - set([to_ml[i][4]]))[0]
            ]
            denominator = A + B
            denominator[denominator == 0] = epsilon
            spot_data = np.nan_to_num(
                np.rint(
                    (A / denominator).reshape(1, -1)
                    * np.array(data_temp[to_ml[i][0]].todense())
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )[0]
        #  spot_data = np.array(spot_data.todense())[0]
        #  print(spot_data.shape)

        for j in range(len(to_ml[i][1])):

            cell_id = to_ml[i][1][j]
            num_temp = np.zeros([data_temp.shape[1]])
            indices = np.where(spot_data > 0)[0]
            num_temp[indices] = np.random.binomial(
                spot_data[indices].astype(int), ratios[i][j], size=len(indices)
            )
            cells_x[cell_id] = cells_x[cell_id] + csr_matrix(num_temp)
            binnumbers[cell_id] = binnumbers[cell_id] + ratios[i][j] * to_ml[i][2]
            spots_composition[to_ml[i][0]][cell_id] = ratios[i][j] * to_ml[i][2]

    final_X = lil_matrix(np.zeros([len(adatas_final.obs), data_temp.shape[1]]))

    cell_index = list(adatas_final.obs.index)
    cell_info = np.zeros([len(adatas_final.obs), 5])

    cell_types = {}
    cell_ids = list(adatas_final.obs.index.astype(float))
    celltypes = list(adatas_final.obs.leiden.astype(int))
    for i in range(len(cell_ids)):
        cell_types[int(cell_ids[i])] = celltypes[i]

    for i in range(len(cell_index)):
        cell_id = float(cell_index[i])
        final_X[i] = lil_matrix(cells_x[cell_id])
        cell_info[i, 0] = cell_types[cell_id]
        cell_info[i, 2] = binnumbers[cell_id]
        cell_info[i, 3] = so.cell_centers[cell_id][0]
        cell_info[i, 4] = so.cell_centers[cell_id][1]

        if norm(cells_x[cell_id]) != 0:
            cell_info[i, 1] = (
                np.dot(
                    cells_x[cell_id], weight_to_celltype[int(cell_info[i, 0])]
                ).item()
            ) / norm(cells_x[cell_id]).item()
        else:
            cell_info[i, 1] = 0

    adata_sc_final = anndata.AnnData(
        X=final_X.tocsr(),
        obs=pd.DataFrame(
            cell_info,
            columns=["cell_cluster", "cos_simularity", "cell_size", "x", "y"],
            index=cell_index,
        ),
        var=adata.var,
    )

    # Optionally plot the cells on the spatial map
    if plot:
        keys = list(spots_composition.keys())
        for i in keys:
            if len(spots_composition[i]) == 0:
                del spots_composition[i]
            elif abs(sum(spots_composition[i].values()) - 1) > 0.0001:
                raise ValueError("Proportion Sum for spot " + str(i) + " is not 1.")
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
                        chosen_cell_id = np.random.choice(
                            cells, p=list(cell_dict.values())
                        )
                        new_matrix[i, j] = chosen_cell_id

        so.pixels_cells = new_matrix

    return adata_sc_final
