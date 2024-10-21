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

    epsilon = 1e-8

    weight_to_celltype_norm = weight_to_celltype / weight_to_celltype.sum(
        axis=1
    ).reshape(-1, 1)

    x2 = np.array([0.5, 0.5])
    x3 = np.array([0.33, 0.33, 0.34])

    cons = {"type": "eq", "fun": lambda v: np.sum(v) - 1}

    bounds2 = [(0, 1), (0, 1)]
    bounds3 = [(0, 1), (0, 1), (0, 1)]

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

    cells_before_ml_x = {}
    for cell_id in cells_before_ml.keys():
        cells_before_ml_x[cell_id] = np.zeros([data_temp.shape[1]])

    for cell_id in cells_before_ml.keys():
        for i in cells_before_ml[cell_id]:
            if len(i) == 2:
                cells_before_ml_x[cell_id] = (
                    cells_before_ml_x[cell_id] + data_temp[i[0]]
                )
            else:
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

    cells_to_groups = {}
    for group in groups_combined.keys():
        for cell in groups_combined[group]:
            cells_to_groups[cell] = group

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

    nonzero_indices_toml = copy.deepcopy(nonzero_indices_dic)

    for i in groups_combined.keys():
        for j in range(len(nonzero_indices_dic[i])):
            for k in range(len(nonzero_indices_dic[i][j])):
                nonzero_indices_toml[i][j][k] = cells_toid_toml_dic[i][
                    nonzero_indices_dic[i][j][k]
                ]

    cells_X_plus_dic = {}
    spots_X_dic = {}
    celltypes_dic = {}
    spots_id_dic = {}
    spots_id_dic_prop = {}

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


def calculate_distances(A, B, C):
    # Calculate distance between A and B
    AB = math.sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2)
    # Calculate distance between A and C
    AC = math.sqrt((C[0] - A[0]) ** 2 + (C[1] - A[1]) ** 2)
    sums = AB + AC
    return AB / sums, AC / sums


def calculate_weight_to_celltype(adatas_final, adata, cells_final, so):

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
):

    binnumbers = {}
    epsilon = 1e-8

    data_temp = csr_matrix(adata.X)

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

    if spots_X_dic == None:

        if nonzero_indices_toml != None:

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
    # cells_after_ml = copy.deepcopy(cells_before_ml_x)

    cellid_to_index = {}

    final_X = lil_matrix(np.zeros([len(adatas_final.obs), data_temp.shape[1]]))
    cell_index = list(adatas_final.obs.index)

    for i in range(len(cell_index)):
        cell_id = float(cell_index[i])
        cellid_to_index[cell_id] = i
        final_X[i] = csr_matrix(cells_before_ml_x[cell_id])

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

    cell_info = np.zeros([len(adatas_final.obs), 3])

    for i in range(len(cell_index)):

        cell_id = float(cell_index[i])

        cell_info[i, 0] = cell_types[cell_id]
        cell_info[i, 2] = binnumbers[cell_id]
        if final_X[i].sum() > 0:
            a = np.dot(
                final_X[i].tocsr().toarray(), weight_to_celltype[cell_types[cell_id]]
            ) / norm(final_X[i].tocsr().toarray())
            cell_info[i, 1] = a[0]
        else:
            cell_info[i, 1] = 0

    adata_sc_final = anndata.AnnData(
        X=final_X.tocsr(),
        obs=pd.DataFrame(
            cell_info,
            columns=["cell_type", "cos_simularity", "cell_size"],
            index=cell_index,
        ),
        var=adata.var,
    )

    # adata_sc_final.X.eliminate_zeros()

    return adata_sc_final


def get_finaldata_fast(cells_final, so, adatas_final, adata, weight_to_celltype):

    weight_to_celltype_norm = weight_to_celltype / weight_to_celltype.sum(
        axis=1
    ).reshape(-1, 1)

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

    rows = list(so.df[so.df.in_tissue == 1].array_row)
    cols = list(so.df[so.df.in_tissue == 1].array_col)

    ratios = []
    for i in range(len(to_ml)):
        AB, AC = calculate_distances(
            (rows[to_ml[i][0]], cols[to_ml[i][0]]),
            so.cell_centers[to_ml[i][1][0]],
            so.cell_centers[to_ml[i][1][1]],
        )
        ratios.append([AB, AC])

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

    np.random.seed(0)
    for i in range(len(to_ml)):

        for cell_id in to_ml[i][1]:

            num_temp = np.zeros([data_temp.shape[1]])
            spot_data = data_temp[to_ml[i][0]]
            spot_data = np.array(spot_data.todense())[0]
            indices = np.where(spot_data > 0)[0]
            num_temp[indices] = np.random.binomial(
                spot_data[indices].astype(int), ratios[i][0], size=len(indices)
            )
            cells_x[cell_id] = cells_x[cell_id] + csr_matrix(num_temp)
            binnumbers[cell_id] = binnumbers[cell_id] + ratios[i][0]

    final_X = lil_matrix(np.zeros([len(adatas_final.obs), data_temp.shape[1]]))

    cell_index = list(adatas_final.obs.index)
    cell_info = np.zeros([len(adatas_final.obs), 3])

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
            columns=["cell_type", "cos_simularity", "cell_size"],
            index=cell_index,
        ),
        var=adata.var,
    )

    return adata_sc_final
