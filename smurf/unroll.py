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

    """
    Cleans and selects data points based on positional constraints and outlier detection.

    This function applies positional constraints (left, right, up, down) and outlier detection to a set
    of selected data points. It optionally returns the deleted points and plots the results, highlighting
    different categories of points: original points, outliers, points to delete, and saved points.

    :param selected:
        A list or tuple of two arrays:
        - `selected[0]`: An array of x-values or indices corresponding to the data points.
        - `selected[1]`: A NumPy array of shape (n_samples, 2), containing the data points.
    :type selected: list or tuple of arrays

    :param left:
        The left boundary (minimum x-value) for points to delete. Points with x-values greater than `left`
        will be marked for deletion. Defaults to `None`.
    :type left: float, optional

    :param right:
        The right boundary (maximum x-value) for points to delete. Points with x-values less than `right`
        will be marked for deletion. Defaults to `None`.
    :type right: float, optional

    :param up:
        The upper boundary (maximum y-value) for points to delete. Points with y-values less than `up`
        will be marked for deletion. Defaults to `None`.
    :type up: float, optional

    :param down:
        The lower boundary (minimum y-value) for points to delete. Points with y-values greater than `down`
        will be marked for deletion. Defaults to `None`.
    :type down: float, optional

    :param outlier_cutoff:
        The distance cutoff for outlier detection. Points with average distance to their nearest neighbors
        greater than this cutoff will be considered outliers. Defaults to `None`.
    :type outlier_cutoff: float, optional

    :param outlier_neighbors:
        The number of nearest neighbors to consider for outlier detection. Defaults to `6`.
    :type outlier_neighbors: int, optional

    :param delete_xaxis:
        A list containing previously deleted x-values and points, to be combined with new deletions.
        It should be of the form `[deleted_x_values_array, deleted_points_array]`.
        Defaults to `[np.array([]), np.zeros((0, 2))]`.
    :type delete_xaxis: list of arrays, optional

    :param return_deleted:
        Whether to return the deleted points along with the kept points. If `True`, the function returns a tuple containing
        the kept points and the deleted points. Defaults to `False`.
    :type return_deleted: bool, optional

    :param save_area:
        A list specifying an area where points should be saved (not deleted), given as `[left, right, up, down]`.
        Points within this area will not be deleted even if they meet other deletion criteria. Defaults to `[None, None, None, None]`.
    :type save_area: list of floats, optional

    :param k_neighbors:
        The number of neighbors to consider when adding average points for outlier detection. Defaults to `40`.
    :type k_neighbors: int, optional

    :param avg_iterations:
        The number of iterations for averaging points when adding average points for outlier detection. Defaults to `1`.
    :type avg_iterations: int, optional

    :return:
        If `return_deleted` is `False`, returns a list `[selected_x_values_kept, selected_points_array_kept]`, where
        `selected_x_values_kept` is an array of x-values or indices for the kept points, and `selected_points_array_kept`
        is a NumPy array of the kept data points.

        If `return_deleted` is `True`, returns a tuple of lists:
        `([selected_x_values_kept, selected_points_array_kept], [deleted_x_values, deleted_points_array])`, where
        `deleted_x_values` and `deleted_points_array` contain the x-values and data points that were deleted.
    :rtype: list or tuple of lists

    """

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

    """
    Selects cells from an AnnData object based on cluster labels and retrieves their spatial coordinates.

    This function selects cells belonging to specified cluster(s) from the provided AnnData object and extracts
    their spatial coordinates. The spatial coordinates are retrieved either from `adata.obs['x']` and `adata.obs['y']`,
    or from a provided spatial object `so` containing `cell_centers`.

    :param adata:
        An AnnData object containing single-cell gene expression data, with cell metadata in `adata.obs`.
    :type adata: anndata.AnnData

    :param cluster:
        A cluster label or a list of cluster labels to select cells from. Cells belonging to these clusters will be selected.
    :type cluster: int, str, or list

    :param cluster_name:
        The name of the column in `adata.obs` that contains the cluster labels. Defaults to `'cell_cluster'`.
    :type cluster_name: str, optional

    :param so:
        A spatial object containing spatial data, including `cell_centers`. If 'x' and 'y' coordinates are not available
        in `adata.obs`, this parameter is used to retrieve positions. Defaults to `None`.
    :type so: spatial_object, optional

    :return:
        A tuple `(index, xys)`, where:
        - `index`: An array of cell IDs (indices) for the selected cells.
        - `xys`: A NumPy array of shape (n_cells, 2), containing the spatial coordinates of the selected cells.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)

    """

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

    """
    Generates an x-axis for spatial unwrapping of data using Locally Linear Embedding and interpolation.

    This function uses Locally Linear Embedding (LLE) to reduce the selected spatial data points to one dimension.
    It then creates a smoothed axis by averaging over a specified number of points and interpolates along this
    axis to generate evenly spaced points. The resulting x-axis can be used for spatial unwrapping or further analysis.
    The function also plots intermediate results and prints information about the axis length.

    :param selected:
        A list or tuple containing:
        - `selected[0]`: An array of cell IDs or indices.
        - `selected[1]`: A NumPy array of shape (n_samples, 2), containing the positions (coordinates) of the selected cells.
    :type selected: list or tuple of arrays

    :param adata:
        An AnnData object containing single-cell gene expression data.
    :type adata: anndata.AnnData

    :param so:
        A spatial object containing spatial data, including `cell_centers`. Used to obtain positions of other cells not in `selected`.
        Defaults to `None`.
    :type so: spatial_object, optional

    :param seed:
        Random seed for reproducibility. Defaults to `42`.
    :type seed: int, optional

    :param n_neighbors:
        Number of neighbors to use in Locally Linear Embedding (LLE). Defaults to `50`.
    :type n_neighbors: int, optional

    :param num_avg:
        Number of points to average when smoothing the axis. This determines the smoothing window size.
        Defaults to `35`.
    :type num_avg: int, optional

    :param unit:
        Unit length for the x-axis, used to calculate the number of points along the axis.
        Defaults to `5`.
    :type unit: float, optional

    :param resolution:
        Resolution factor for determining the number of points along the x-axis. Higher values result in more points.
        Defaults to `2`.
    :type resolution: float, optional

    :return:
        A NumPy array `even_points` of shape (num_points, 2), containing the coordinates of evenly spaced points along the x-axis.
    :rtype: numpy.ndarray

    """

    X = selected[1]

    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=1, method="standard", random_state=seed
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

    """
    Plots the positions of selected data points.

    This function creates a scatter plot of the positions of the selected data points.
    It adjusts the figure size based on the aspect ratio of the data to ensure an appropriate display.

    :param selected:
        A list or tuple containing:
        - `selected[0]`: An array of cell IDs or indices.
        - `selected[1]`: A NumPy array of shape (n_samples, 2), containing the positions (coordinates) of the selected data points.
    :type selected: list or tuple of arrays

    :return:
        None. The function displays the plot and closes the figure after rendering.
    :rtype: None

    """

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
    center=None,
):

    """
    Computes the y-axis positions for spatial unwrapping based on the x-axis and cell positions.

    This function calculates the y-axis values (flattened distances) for each cell by finding the nearest point on the x-axis. It also handles optional parameters for selected cells, deleted x-axis points, and plotting options.

    :param x_axis:
        A NumPy array of shape (n_points, 2) representing the coordinates of points along the x-axis.
    :type x_axis: numpy.ndarray

    :param cells:
        A tuple or list containing:
        - `cells[0]`: An array of cell IDs or indices.
        - `cells[1]`: A NumPy array of shape (n_cells, 2), containing the positions (coordinates) of the cells.
    :type cells: list or tuple

    :param selected:
        (Optional) A tuple or list similar to `cells`, representing selected cells that are to be included with zero y-axis values. Defaults to `None`.
    :type selected: list or tuple, optional

    :param delete_xaxis:
        (Optional) A list containing previously deleted x-values and points, to be considered in the calculations. Defaults to `[[np.zeros((0, 1))], np.zeros((0, 2))]`.
    :type delete_xaxis: list of arrays, optional

    :param delete_residues:
        (Optional) A boolean indicating whether to delete residues (cells with x_flattened values outside the x-axis range). Defaults to `False`.
    :type delete_residues: bool, optional

    :param unit:
        Unit length for scaling the y-axis values. Defaults to `5`.
    :type unit: float, optional

    :param resolution:
        Resolution factor for determining the scaling of y-axis values. Higher values result in finer scaling. Defaults to `2`.
    :type resolution: float, optional

    :param center:
        (Optional) A NumPy array of shape (2,) representing the coordinates of the center point. Used to determine the direction of layering (inside or outside). Defaults to `None`.
    :type center: numpy.ndarray, optional

    :return:
        A pandas DataFrame `final_results` containing the following columns:

        - 'cell_id': Cell IDs.
        - 'x': Original x-coordinate of the cell.
        - 'y': Original y-coordinate of the cell.
        - 'x_flattened': Index of the nearest point on the x-axis.
        - 'y_flattened': Scaled distance from the cell to the nearest point on the x-axis.

    :rtype: pandas.DataFrame

    """

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

    if center is not None:
        final_results["same_direction_by_dot"] = final_results.apply(
            lambda row: check_cos_angle(row, center, x_axis), axis=1
        )
        final_results = final_results[final_results["same_direction_by_dot"]]
        final_results.drop(columns=["same_direction_by_dot"], inplace=True)

    if delete_xaxis[1].shape[0] != 0:
        final_results = final_results[(final_results["x_flattened"] < x_axis.shape[0])]
    elif delete_residues:
        final_results = final_results[
            (final_results["x_flattened"] > 0)
            & (final_results["x_flattened"] < x_axis.shape[0] - 1)
        ]

    final_results.set_index(final_results.columns[0], inplace=True)

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


def check_cos_angle(row, center, x_axis_results):
    x, y = row["x"], row["y"]
    idx = int(row["x_flattened"])

    v1 = np.array([x - center[0], y - center[1]])
    X_idx, Y_idx = x_axis_results[idx]
    v2 = np.array([x - X_idx, y - Y_idx])

    dot_value = np.dot(v1, v2)
    return dot_value > 0


def x_axis_pre(
    selected,
    seed=42,
    n_neighbors=50,
):

    """
    [Only for testing, use before x_axis()] Preprocesses and visualizes the selected data points using Locally Linear Embedding (LLE).

    This function performs LLE on the selected data points to reduce them to one dimension for visualization purposes. It plots the original data colored according to the LLE embedding.

    :param selected:
        A list or tuple containing:

        - `selected[0]`: An array of cell IDs or indices.
        - `selected[1]`: A NumPy array of shape (n_samples, 2), containing the positions (coordinates) of the selected data points.

    :type selected: list or tuple of arrays

    :param adata:
        An AnnData object containing single-cell gene expression data.
    :type adata: anndata.AnnData

    :param so:
        (Optional) A spatial object containing spatial data, including `cell_centers`. If needed for additional processing. Defaults to `None`.
    :type so: spatial_object, optional

    :param seed:
        Random seed for reproducibility. Defaults to `42`.
    :type seed: int, optional

    :param n_neighbors:
        Number of neighbors to use in Locally Linear Embedding (LLE). Defaults to `50`.
    :type n_neighbors: int, optional

    :return:
        None. The function displays a plot of the data points colored according to their LLE embedding.
    :rtype: None
    """

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

    """
    Selects the remaining cells from an AnnData object after excluding specified cells.

    This function retrieves the indices and positions of cells that are not included in the `selected` cells or `delete_xaxis` cells. It extracts the spatial coordinates of the remaining cells for further analysis.

    :param adata:
        An AnnData object containing single-cell gene expression data.
    :type adata: anndata.AnnData

    :param selected:
        (Optional) A list or tuple containing:

        - `selected[0]`: An array of cell IDs or indices to exclude.
        - `selected[1]`: A NumPy array of positions. Only `selected[0]` is used in this function. Defaults to `None`.

    :type selected: list or tuple of arrays, optional

    :param delete_xaxis:
        (Optional) A list or tuple containing:

        - `delete_xaxis[0]`: An array of cell IDs or indices to exclude.
        - `delete_xaxis[1]`: A NumPy array of positions. Only `delete_xaxis[0]` is used in this function. Defaults to `None`.

    :type delete_xaxis: list or tuple of arrays, optional

    :param so:
        (Optional) A spatial object containing spatial data, including `cell_centers`. Used to obtain positions if 'x' and 'y' are not in `adata.obs`. Defaults to `None`.
    :type so: spatial_object, optional

    :return:
        A tuple `(index, xys)`, where:

        - `index`: An array of cell IDs (indices) for the remaining cells.
        - `xys`: A NumPy array of shape (n_cells, 2), containing the spatial coordinates of the remaining cells.

    :rtype: tuple of (numpy.ndarray, numpy.ndarray)

    """

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
    selected=[np.zeros((0)), np.zeros((0, 2))],
    n_neighbors_first=18,
    n_neighbors_rest=7,
    distance_cutoff=23,
    center=None,
    outside=True,
    max_iter=1000,
    unit=5,
    resolution=2,
):

    """
    Generates the y-axis values for circular or layered structures based on the x-axis and cell positions.

    This function computes y-axis (flattened) values for cells in a circular or layered fashion, suitable for data with concentric structures. It assigns layers to cells based on their distance from the x-axis and plots the results.

    :param x_axis:
        A NumPy array of shape (n_points, 2) representing the coordinates of points along the x-axis.
    :type x_axis: numpy.ndarray

    :param cells:
        A tuple or list containing:

        - `cells[0]`: An array of cell IDs or indices.
        - `cells[1]`: A NumPy array of shape (n_cells, 2), containing the positions (coordinates) of the cells.

    :type cells: list or tuple

    :param delete_xaxis:
        (Optional) A list containing previously deleted x-values and points, to be considered in the calculations. Defaults to `[[np.zeros((0, 1))], np.zeros((0, 2))]`.
    :type delete_xaxis: list of arrays, optional

    :param selected:
        (Optional) A tuple or list similar to `cells`, representing selected cells that are to be included with zero y-axis values. Defaults to `None`.
    :type selected: list or tuple, optional

    :param n_neighbors_first:
        Number of neighbors to consider in the first pass of the algorithm. Defaults to `18`.
    :type n_neighbors_first: int, optional

    :param n_neighbors_rest:
        Number of neighbors to consider in subsequent passes of the algorithm. Defaults to `7`.
    :type n_neighbors_rest: int, optional

    :param distance_cutoff:
        Maximum distance to consider when assigning layers to cells. Defaults to `23`.
    :type distance_cutoff: float, optional

    :param center:
        (Optional) A NumPy array of shape (2,) representing the coordinates of the center point. Used to determine the direction of layering (inside or outside). Defaults to `None`.
    :type center: numpy.ndarray, optional

    :param outside:
        (Optional) A boolean indicating the direction of layering relative to the center. If `True`, layers are assigned moving away from the center. Defaults to `True`.
    :type outside: bool, optional

    :param max_iter:
        Maximum number of iterations (layers) to process. Defaults to `1000`.
    :type max_iter: int, optional

    :param unit:
        Unit length for scaling the y-axis values. Defaults to `5`.
    :type unit: float, optional

    :param resolution:
        Resolution factor for determining the scaling of y-axis values. Higher values result in finer scaling. Defaults to `2`.
    :type resolution: float, optional

    :return:
        A pandas DataFrame `final_result` containing the following columns:

        - 'cell_id': Cell IDs.
        - 'x': Original x-coordinate of the cell.
        - 'y': Original y-coordinate of the cell.
        - 'x_flattened': Index of the nearest point on the x-axis.
        - 'y_flattened': Scaled cumulative distance representing the layer.
        - 'layer': Assigned layer number.

    :rtype: pandas.DataFrame

    """

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

    """
    Plots the final results of the spatial unwrapping process.

    This function creates visualizations of the unwrapped spatial data, using the 'x_flattened' and 'y_flattened' values from the `final_result` DataFrame. It helps in analyzing and interpreting the results of the unwrapping process.

    :param final_result:
        A pandas DataFrame containing the final results of the spatial unwrapping, with columns such as 'cell_id', 'x', 'y', 'x_flattened', 'y_flattened', and possibly 'layer'.
    :type final_result: pandas.DataFrame

    :return:
        None. The function displays plots of the unwrapped data.

    :rtype: None

    """

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
