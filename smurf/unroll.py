import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm
from numba import jit
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors


@jit(nopython=True)
def find_intersections(A, B, C):
    for index, element in enumerate(A):
        if element in B:
            return True, element, C[index]
    return False, None, 999999


def add_average_points(points, k=40, iterations=1):

    new_points = points.copy()
    for _ in range(iterations):
        n = new_points.shape[0]
        tree = cKDTree(new_points)
        # For each point, find itself and its k nearest neighbors (including itself)
        distances, indices = tree.query(new_points, k=k)
        avg_points = []
        for i in range(n):
            neighbor_indices = indices[i]
            neighbor_points = new_points[neighbor_indices]
            avg_point = neighbor_points.mean(axis=0)
            avg_points.append(avg_point)
        avg_points = np.array(avg_points)
        # Add the new average points to the point set
        new_points = np.vstack([new_points, avg_points])
    return new_points


def clean_select(
    selected,
    left=None,
    right=None,
    up=None,
    down=None,
    outlier_cutoff=None,
    outlier_neighbors=6,
    delete_xaxis=[np.array([]), np.zeros((0, 2))],
    return_deleted=False,
    save_area=[None, None, None, None],
    k_neighbors=40,
    avg_iterations=1,
):

    # Start with all True values in the mask (points to delete)
    mask = np.ones(len(selected[1]), dtype=bool)
    # Start with all True values in the save mask (points to save)
    save = np.ones(len(selected[1]), dtype=bool)

    # Apply position constraints to create mask
    if left is not None:
        mask &= selected[1][:, 0] > left
    if right is not None:
        mask &= selected[1][:, 0] < right
    if up is not None:
        mask &= selected[1][:, 1] < up
    if down is not None:
        mask &= selected[1][:, 1] > down

    # Apply save area constraints to create save mask
    if save_area[0] is not None:
        save &= selected[1][:, 0] > save_area[0]
    if save_area[1] is not None:
        save &= selected[1][:, 0] < save_area[1]
    if save_area[2] is not None:
        save &= selected[1][:, 1] < save_area[2]
    if save_area[3] is not None:
        save &= selected[1][:, 1] > save_area[3]

    # If no position constraints are given, reset mask to keep everything
    if np.all(mask):
        mask = np.zeros(len(selected[1]), dtype=bool)  # All False

    if np.all(save):
        save = np.zeros(len(selected[1]), dtype=bool)  # All False

    # Add average points to the dataset for outlier detection
    # Using the provided add_average_points function
    augmented_points = add_average_points(
        selected[1], k=k_neighbors, iterations=avg_iterations
    )

    # Fit the NearestNeighbors model on the augmented dataset
    nn_model = NearestNeighbors(
        n_neighbors=outlier_neighbors + 1, algorithm="auto", metric="euclidean"
    )
    nn_model.fit(augmented_points)

    # Calculate mean distances to the nearest neighbors for original points
    distances, indices = nn_model.kneighbors(selected[1])
    # Exclude the first neighbor (the point itself)
    avg_distances = distances[:, 1:].mean(axis=1)

    # Identify outliers among original points based on the cutoff
    filtered = avg_distances > outlier_cutoff

    # Plotting
    ratio = (selected[1][:, 0].max() - selected[1][:, 0].min()) / (
        selected[1][:, 1].max() - selected[1][:, 1].min()
    )
    fig = plt.figure(figsize=(6 * ratio, 6))
    # Plot all original points in gray
    plt.plot(
        selected[1][:, 0],
        selected[1][:, 1],
        ".",
        c="gray",
        markersize=0.3,
        label="Original Points",
    )
    # Plot outliers in blue
    plt.plot(
        selected[1][filtered, 0],
        selected[1][filtered, 1],
        ".",
        c="b",
        label="Outliers",
        markersize=2,
    )
    # Plot points to delete based on position constraints in red
    plt.plot(
        selected[1][mask, 0],
        selected[1][mask, 1],
        ".",
        c="r",
        label="To Delete",
        markersize=2,
    )
    # Plot saved points in green
    plt.plot(
        selected[1][save, 0],
        selected[1][save, 1],
        ".",
        c="green",
        label="Saved",
        markersize=2,
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Keep only the points that are neither in the mask nor outliers, or are in the save area
    keep_mask = ~(mask | filtered) | save

    if return_deleted:
        # Collect deleted x_values and points
        deleted_x_values = np.hstack((delete_xaxis[0], selected[0][~keep_mask]))
        deleted_points_array = np.vstack((delete_xaxis[1], selected[1][~keep_mask]))
        return [selected[0][keep_mask], selected[1][keep_mask]], [
            deleted_x_values,
            deleted_points_array,
        ]
    else:
        return [selected[0][keep_mask], selected[1][keep_mask]]


def select_cells(adata, cluster, cluster_name="cell_cluster", so=None):
    index = np.array(adata[adata.obs[cluster_name].isin(cluster)].obs.index)

    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        xys = np.array(adata[index, :].obs[["x", "y"]])
    elif so is not None:
        xys = []
        for cell_id in index:
            # Adjust the type conversion based on your actual data types
            xys.append(list(so.cell_centers[float(cell_id)]))
        xys = np.array(xys)
    else:
        raise ValueError(
            "Please input either 'so' or ensure 'x' and 'y' are in 'adata.obs'"
        )

    return index, xys


def sort_index(data, order=1):

    data = order * data
    sorted_indices = np.argsort(data)
    rank = np.empty_like(sorted_indices)
    rank[sorted_indices] = np.arange(len(data))

    return rank


def x_axis(
    selected, adata, so=None, seed=42, n_neighbors=50, num_avg=35, unit=5, resolution=2
):

    X = selected[1]

    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=1, method="standard"
    )
    X_unfolded = lle.fit_transform(X)
    color = X_unfolded.T[0] - X_unfolded.T[0].min()

    order = sort_index(color)
    nums = np.array(order)

    axis = np.zeros([len(order), 3])
    for i in range(len(order)):
        idd = int(nums[i])
        axis[idd, 0] = float(selected[0][i])
        axis[idd, 1:3] = X[i, :]

    new = np.zeros([len(order) - num_avg + 1, 3])
    for i in range(len(order) - num_avg + 1):
        new[i, 0] = i
        new[i, 1:3] = axis[i : i + num_avg + 1, 1:3].mean(axis=0)

    initial_points = new[:, 1:3]

    cumulative_distances = np.cumsum(
        np.insert(np.sqrt(np.sum(np.diff(initial_points, axis=0) ** 2, axis=1)), 0, 0)
    )

    total_distance = cumulative_distances[-1]

    interp_x = interp1d(cumulative_distances, initial_points[:, 0], kind="cubic")
    interp_y = interp1d(cumulative_distances, initial_points[:, 1], kind="cubic")

    num_points = int(resolution * total_distance / unit)

    even_distances = np.linspace(0, total_distance, num_points)
    even_points = np.stack(
        (interp_x(even_distances), interp_y(even_distances)), axis=-1
    )

    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        xyss = np.array(adata.obs[["x", "y"]])
    elif so is not None:
        xyss = []
        #  cell_ids_rest = []
        for cell_id in np.array(adata.obs.index):
            if cell_id not in selected[0]:
                #  cell_ids_rest.append(cell_id)
                xyss.append(list(so.cell_centers[float(cell_id)]))

        xyss = np.array(xyss)
    # cell_ids_rest = np.array(cell_ids_rest)
    else:
        raise ValueError(
            "Please input either 'so' or ensure 'x' and 'y' are in 'adata.obs'"
        )

    ratio = (selected[1][:, 0].max() - selected[1][:, 0].min()) / (
        selected[1][:, 1].max() - selected[1][:, 1].min()
    )

    fig, axs = plt.subplots(1, 4, figsize=(4 * 3.5 * ratio, 3.5))

    # First plot
    c = axs[0].scatter(
        X[:, 0], X[:, 1], c=sort_index(color), cmap=plt.cm.Spectral, s=0.5
    )
    axs[0].set_title("Original Swiss Roll")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    cbar = plt.colorbar(c, ax=axs[0], orientation="vertical")
    cbar.set_label("Color Scale")

    # Second plot

    axs[1].plot(
        initial_points[:, 0],
        initial_points[:, 1],
        "o",
        label="Smooth Points",
        markersize=0.2,
    )
    axs[1].plot(X[:, 0], X[:, 1], "o", c="red", label="Initial Points", markersize=0.1)
    axs[1].set_title("Original Points and Smooth Points")
    axs[1].axis("equal")
    axs[1].legend()

    # Third plot
    axs[2].plot(
        even_points[:, 0], even_points[:, 1], "o", label="Final x-axis", markersize=0.2
    )
    axs[2].set_title("X-axis")
    axs[2].axis("equal")
    axs[2].legend()

    # Fourth plot
    axs[3].plot(xyss[:, 0], xyss[:, 1], ".", c="blue", label="Cells", markersize=0.5)
    axs[3].plot(
        even_points[:, 0],
        even_points[:, 1],
        ".",
        c="red",
        label="x-axis",
        markersize=0.5,
    )
    axs[3].set_title("Unwrapping Mouse Brain")
    axs[3].set_xlabel("X")
    axs[3].set_ylabel("Y")
    axs[3].legend()

    plt.tight_layout()
    plt.show()

    print(
        f"One unit length on the x-axis is {total_distance/num_points * resolution:.2f} μm. The total x-axis length is {total_distance * resolution * 0.0001:.4f} cm."
    )

    return even_points


def plot_selected(selected):

    ratio = (selected[1][:, 0].max() - selected[1][:, 0].min()) / (
        selected[1][:, 1].max() - selected[1][:, 1].min()
    )
    fig = plt.figure(figsize=(6 * ratio, 6))

    # Plot with a single color to avoid duplicate plots
    plt.plot(selected[1][:, 0], selected[1][:, 1], ".", markersize=1)

    # Set title and labels correctly
    plt.title("Your data")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Show the plot
    plt.show()
    plt.close()


def y_axis(
    x_axis,
    cells,
    selected=None,
    delete_xaxis=[[np.zeros((0, 1))], np.zeros((0, 2))],
    delete_residues=False,
    unit=5,
    resolution=2,
):
    # Concatenate x_axis with the second element of cells
    data_total = np.concatenate((x_axis, delete_xaxis[1], cells[1]))
    cells_before = x_axis.shape[0] + delete_xaxis[1].shape[0]

    # Initialize and fit NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(np.concatenate((x_axis, delete_xaxis[1])))
    distances, indices = nn.kneighbors(cells[1])

    # Create a DataFrame to store final results
    final_results = pd.DataFrame(
        999999 * np.ones([cells[1].shape[0], 5]),
        columns=["cell_id", "x", "y", "x_flattened", "y_flattened"],
    )
    #  final_results['data_total_id'] = list(range(cells_before, data_total.shape[0]))
    final_results["cell_id"] = cells[0]
    final_results["x"] = cells[1][:, 0]
    final_results["y"] = cells[1][:, 1]
    final_results["x_flattened"] = indices.flatten()
    final_results["y_flattened"] = distances.flatten()
    final_results["y_flattened"] = resolution * final_results["y_flattened"] / unit

    if selected != None:

        if np.all(~np.isin(selected[0], cells[0])):

            data_total = np.concatenate((x_axis, delete_xaxis[1], selected[1]))
            cells_before = x_axis.shape[0] + delete_xaxis[1].shape[0]

            # Initialize and fit NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
            nn.fit(np.concatenate((x_axis, delete_xaxis[1])))
            distances, indices = nn.kneighbors(selected[1])

            final_results2 = pd.DataFrame(
                999999 * np.zeros([len(selected[0]), 5]),
                columns=["cell_id", "x", "y", "x_flattened", "y_flattened"],
            )
            #     final_results2['data_total_id'] = list(range(len(selected[0])))
            final_results2["cell_id"] = selected[0]
            final_results2["x"] = selected[1][:, 0]
            final_results2["y"] = selected[1][:, 1]
            final_results2["x_flattened"] = indices.flatten()
            final_results2["y_flattened"] = 0

            final_results = pd.concat(
                [final_results2, final_results], ignore_index=True
            )

    #  print(delete_xaxis.shape[0])

    if delete_xaxis[1].shape[0] != 0:
        final_results = final_results[(final_results["x_flattened"] < x_axis.shape[0])]
    elif delete_residues:
        final_results = final_results[
            (final_results["x_flattened"] > 0)
            & (final_results["x_flattened"] < x_axis.shape[0] - 1)
        ]

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

    # --- First Scatter Plot on ax1 ---
    scatter_final_ax1 = ax1.scatter(
        final_results["x"],
        final_results["y"],
        c=final_results["x_flattened"],
        cmap="Spectral",
        s=0.5,
        label="Final Results",
    )

    scatter_x_axis_ax1 = ax1.scatter(
        x_axis[:, 0],
        x_axis[:, 1],
        c=np.arange(x_axis.shape[0], dtype=int),
        cmap="Spectral",
        s=50,  # Adjust size for visibility
        edgecolors="black",  # Circle edge color
        linewidth=0.05,
        label="X Axis Points",
    )

    # Set title and labels for the first subplot
    ax1.set_title("Original Swiss Roll - X Flattened")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Add a colorbar for the first scatter plot
    cbar1 = plt.colorbar(scatter_final_ax1, ax=ax1)
    cbar1.set_label("X Flattened Color Scale")

    # Add legend to the first subplot
    ax1.legend()

    # --- Second Scatter Plot on ax2 ---
    scatter_final_ax2 = ax2.scatter(
        final_results["x"],
        final_results["y"],
        c=final_results["y_flattened"],
        cmap="Spectral",
        s=0.5,
        label="Final Results",
    )

    # Set title and labels for the second subplot
    ax2.set_title("Original Swiss Roll - Y Flattened")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # Add a colorbar for the second scatter plot
    cbar2 = plt.colorbar(scatter_final_ax2, ax=ax2)
    cbar2.set_label("Y Flattened Color Scale")

    # Add legend to the second subplot
    ax2.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the combined plots
    plt.show()
    plt.close()

    return final_results


def x_axis_pre(
    selected,
    adata,
    so=None,
    seed=42,
    n_neighbors=50,
):

    if not isinstance(selected, (list, tuple)) or len(selected) < 2:
        raise ValueError(
            "selected should be a list or tuple containing at least two elements."
        )

    X = selected[1]

    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=1, method="standard", random_state=seed
    )
    X_unfolded = lle.fit_transform(X)
    color = X_unfolded[:, 0] - X_unfolded[:, 0].min()

    ratio = (selected[1][:, 0].max() - selected[1][:, 0].min()) / (
        selected[1][:, 1].max() - selected[1][:, 1].min()
    )
    fig, ax = plt.subplots(1, 1, figsize=(4.8 * ratio, 4))

    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=sort_index(color), cmap=plt.cm.Spectral, s=0.5
    )
    ax.set_title("Original Swiss Roll")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_label("Color Scale")

    plt.tight_layout()
    plt.show()


def select_rest_cells(adata, selected=None, delete_xaxis=None, so=None):
    # Start with all indices as a pandas Index for efficient operations
    index = adata.obs.index

    # Collect indices to exclude
    exclude_indices = []
    if selected is not None:
        exclude_indices.extend(selected[0])
    if delete_xaxis is not None:
        exclude_indices.extend(delete_xaxis[0])

    # Exclude the specified indices using pandas Index methods
    if exclude_indices:
        index = index.difference(exclude_indices)

    # Proceed to extract 'x' and 'y' coordinates
    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        # Use pandas loc for efficient selection
        xys = adata.obs.loc[index, ["x", "y"]].values
    elif so is not None:
        # Convert so.cell_centers to a DataFrame if it's a dictionary
        if isinstance(so.cell_centers, dict):
            so_centers_df = pd.DataFrame.from_dict(
                so.cell_centers, orient="index", columns=["x", "y"]
            )
        else:
            # Assume so.cell_centers is already a DataFrame or Series
            so_centers_df = so.cell_centers

        # Ensure the indices match by converting them to strings
        so_centers_df.index = so_centers_df.index.astype(str)

        # Select the coordinates using pandas loc
        xys = so_centers_df.loc[index].values
    else:
        raise ValueError(
            "Please input either 'so' or ensure 'x' and 'y' are in 'adata.obs'"
        )

    # Return indices as a NumPy array
    return index.values, xys


def find_closest_point(array, reference_point):

    differences = array - reference_point
    squared_distances = np.sum(differences**2, axis=1)
    min_index = np.argmin(squared_distances)

    return min_index, np.sqrt(squared_distances[min_index])


def y_axis_circle(
    x_axis,
    cells,
    delete_xaxis=[[np.zeros((0, 1))], np.zeros((0, 2))],
    selected=None,
    n_neighbors_first=18,
    n_neighbors_rest=7,
    distance_cutoff=23,
    center=None,
    outside=True,
    max_iter=1000,
    unit=5,
    resolution=2,
):

    xys_final = cells[1]
    rest_cells = cells[0]

    data_total = np.concatenate((x_axis, delete_xaxis[1], xys_final))
    cells_before = x_axis.shape[0] + np.array(delete_xaxis[1]).shape[0]

    neighbor = NearestNeighbors(n_neighbors=n_neighbors_first)
    neighbor.fit(data_total)
    distances, indices = neighbor.kneighbors(data_total)

    final_results = pd.DataFrame(
        999999 * np.ones([xys_final.shape[0], 7]),
        columns=[
            "data_total_id",
            "cell_id",
            "x",
            "y",
            "x_flattened",
            "y_flattened",
            "layer",
        ],
    )
    # final_results = pd.DataFrame(999999* np.ones([xys_final.shape[0],6]), columns=['cell_id','x','y','x_flattened','y_flattened','layer'])
    final_results["data_total_id"] = list(range(cells_before, data_total.shape[0]))
    final_results["cell_id"] = rest_cells
    final_results["x"] = xys_final[:, 0]
    final_results["y"] = xys_final[:, 1]
    final_results = final_results.set_index("data_total_id")

    layers = {}
    layers_rest = {}

    layers[0] = set(range(cells_before))
    layers_rest[0] = set(range(cells_before, data_total.shape[0]))

    layer_now = 0
    layers[layer_now + 1] = set([])

    pbar = tqdm.tqdm()
    pbar.update(1)

    pbar.set_postfix({"Layer": layer_now})

    for target_cell in layers_rest[layer_now]:

        inter, reference_cell, distance = find_intersections(
            indices[target_cell], layers[layer_now], distances[target_cell]
        )

        if (
            inter
            and (distance < distance_cutoff)
            and (distance < final_results.at[target_cell, "y_flattened"])
        ):

            if center is not None:

                # print('reference_cell')
                A = data_total[target_cell, :]
                AB = center - A
                AC = data_total[reference_cell] - A
                dot_product = np.dot(AB, AC)

                if (dot_product > 0) == outside:

                    final_results.at[target_cell, "x_flattened"] = reference_cell
                    final_results.at[target_cell, "y_flattened"] = distance
                    final_results.at[target_cell, "layer"] = layer_now + 1
                    layers[layer_now + 1].add(target_cell)

            else:

                final_results.at[target_cell, "x_flattened"] = reference_cell
                final_results.at[target_cell, "y_flattened"] = distance
                final_results.at[target_cell, "layer"] = layer_now + 1
                layers[layer_now + 1].add(target_cell)

    layers_rest[layer_now + 1] = layers_rest[layer_now] - layers[layer_now + 1]

    neighbor = NearestNeighbors(n_neighbors=n_neighbors_rest)
    neighbor.fit(data_total)
    distances, indices = neighbor.kneighbors(data_total)

    layer_now = 0

    for layer_now in range(1, max_iter):

        pbar.update(1)
        pbar.set_postfix({"Layer": layer_now})

        # print(layer_now)
        layers[layer_now + 1] = set([])

        for target_cell in layers_rest[layer_now]:

            inter, reference_cell, distance = find_intersections(
                indices[target_cell], layers[layer_now], distances[target_cell]
            )

            if inter and distance < distance_cutoff:

                distance_total = (
                    distance + final_results.at[reference_cell, "y_flattened"]
                )
                reference_cell = final_results.at[reference_cell, "x_flattened"]

                if distance_total < final_results.at[target_cell, "y_flattened"]:

                    if center is not None:

                        # print('reference_cell')
                        A = data_total[target_cell, :]
                        AB = center - A
                        AC = data_total[int(reference_cell)] - A
                        dot_product = np.dot(AB, AC)

                        if (dot_product > 0) == outside:

                            final_results.at[
                                target_cell, "x_flattened"
                            ] = reference_cell
                            final_results.at[
                                target_cell, "y_flattened"
                            ] = distance_total
                            final_results.at[target_cell, "layer"] = layer_now + 1
                            layers[layer_now + 1].add(target_cell)

                    else:

                        final_results.at[target_cell, "x_flattened"] = reference_cell
                        final_results.at[target_cell, "y_flattened"] = distance_total
                        final_results.at[target_cell, "layer"] = layer_now + 1
                        layers[layer_now + 1].add(target_cell)

        layers_rest[layer_now + 1] = layers_rest[layer_now] - layers[layer_now + 1]

        # print()
        #  print(len(layers_rest[layer_now]))
        if len(layers_rest[layer_now + 1]) == 0 or len(layers[layer_now + 1]) == 0:
            break

    pbar.close()
    ids_final = []

    for i in range(len(selected[0])):
        point = selected[1][i]
        index, dis = find_closest_point(x_axis, point)
        ids_final.append([point[0], point[1], index, dis])

    ids_final = np.array(ids_final)

    final_results2 = pd.DataFrame(
        np.zeros([len(selected[0]), 7]),
        columns=[
            "data_total_id",
            "cell_id",
            "x",
            "y",
            "x_flattened",
            "y_flattened",
            "layer",
        ],
    )
    final_results2["data_total_id"] = list(range(len(selected[0])))
    final_results2["cell_id"] = selected[0]
    final_results2["x"] = ids_final[:, 0]
    final_results2["y"] = ids_final[:, 1]
    final_results2["x_flattened"] = ids_final[:, 2]
    final_results2["y_flattened"] = ids_final[:, 3]
    final_results2["layer"] = 0
    final_results2 = final_results2.set_index("data_total_id")

    final_results = final_results.set_index("cell_id")
    final_results2 = final_results2.set_index("cell_id")

    final_result = pd.concat([final_results2, final_results])

    final_result.loc[final_result["y_flattened"] == 999999, "y_flattened"] = np.nan
    final_result.loc[final_result["y_flattened"].isna(), "x_flattened"] = np.nan
    final_result.loc[final_result["y_flattened"].isna(), "layer"] = np.nan
    final_result["y_flattened"] = resolution * final_result["y_flattened"] / unit
    final_result.index = final_result.index.astype(float)
    final_result = final_result.sort_values(by="cell_id", ascending=True)

    final_result = final_result[final_result["x_flattened"] <= x_axis.shape[0]]

    fig, axs = plt.subplots(1, 3, figsize=(36, 6))

    # First subplot
    scatter1 = axs[0].scatter(
        final_result["x"],
        final_result["y"],
        c=final_result["x_flattened"],
        cmap=plt.cm.Spectral,
        s=0.5,
    )

    scatter_x_axis_ax1 = axs[0].scatter(
        x_axis[:, 0],
        x_axis[:, 1],
        c=np.arange(x_axis.shape[0], dtype=int),
        cmap="Spectral",
        s=30,
        edgecolors="black",
        linewidth=0.05,
        label="X Axis Points",
    )

    axs[0].set_title("Original Swiss Roll (X color)")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    # Add a color bar to the first subplot
    cbar1 = fig.colorbar(scatter1, ax=axs[0])
    cbar1.set_label("Color scale")

    # Second subplot
    scatter2 = axs[1].scatter(
        final_result["x"],
        final_result["y"],
        c=final_result["y_flattened"],
        cmap=plt.cm.Spectral,
        s=0.5,
    )
    axs[1].set_title("Original Swiss Roll (Y color)")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    # Add a color bar to the second subplot
    cbar2 = fig.colorbar(scatter2, ax=axs[1])
    cbar2.set_label("Color scale")

    # Third subplot
    scatter3 = axs[2].scatter(
        final_result["x"],
        final_result["y"],
        c=final_result["layer"],
        cmap=plt.cm.Spectral,
        s=0.5,
    )
    axs[2].set_title("Original Swiss Roll (Layer color)")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")
    # Add a color bar to the third subplot
    cbar3 = fig.colorbar(scatter3, ax=axs[2])
    cbar3.set_label("layer")

    plt.show()
    plt.close()

    return final_result


def plot_final_result(final_result):

    if "layer" in list(final_result.columns):

        fig, axs = plt.subplots(1, 3, figsize=(36, 6))

        # First subplot
        scatter1 = axs[0].scatter(
            final_result["x"],
            final_result["y"],
            c=final_result["x_flattened"],
            cmap=plt.cm.Spectral,
            s=0.5,
        )
        axs[0].set_title("Original Swiss Roll (X color)")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        # Add a color bar to the first subplot
        cbar1 = fig.colorbar(scatter1, ax=axs[0])
        cbar1.set_label("Color scale")

        # Second subplot
        scatter2 = axs[1].scatter(
            final_result["x"],
            final_result["y"],
            c=final_result["y_flattened"],
            cmap=plt.cm.Spectral,
            s=0.5,
        )
        axs[1].set_title("Original Swiss Roll (Y color)")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        # Add a color bar to the second subplot
        cbar2 = fig.colorbar(scatter2, ax=axs[1])
        cbar2.set_label("Color scale")

        # Third subplot
        scatter3 = axs[2].scatter(
            final_result["x"],
            final_result["y"],
            c=final_result["layer"],
            cmap=plt.cm.Spectral,
            s=0.5,
        )
        axs[2].set_title("Original Swiss Roll (Layer color)")
        axs[2].set_xlabel("X")
        axs[2].set_ylabel("Y")
        # Add a color bar to the third subplot
        cbar3 = fig.colorbar(scatter3, ax=axs[2])
        cbar3.set_label("layer")

        plt.show()
        plt.close()

    else:

        # Create a figure with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(24, 6))

        # First subplot
        scatter1 = axs[0].scatter(
            final_result["x"],
            final_result["y"],
            c=final_result["x_flattened"],
            cmap=plt.cm.Spectral,
            s=0.5,
        )
        axs[0].set_title("Original Swiss Roll (X color)")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        # Add a color bar to the first subplot
        cbar1 = fig.colorbar(scatter1, ax=axs[0])
        cbar1.set_label("Color scale")

        # Second subplot
        scatter2 = axs[1].scatter(
            final_result["x"],
            final_result["y"],
            c=final_result["y_flattened"],
            cmap=plt.cm.Spectral,
            s=0.5,
        )
        axs[1].set_title("Original Swiss Roll (Y color)")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        # Add a color bar to the second subplot
        cbar2 = fig.colorbar(scatter2, ax=axs[1])
        cbar2.set_label("Color scale")

        plt.show()
