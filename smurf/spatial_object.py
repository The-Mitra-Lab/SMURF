import copy
from collections import defaultdict

import numpy as np
import pandas as pd
import tqdm


def to_dict(d):

    # Recursively convert defaultdicts to regular dictionaries
    if isinstance(d, defaultdict):
        return {key: to_dict(value) for key, value in d.items()}
    return d


def cells(segmentation_final, pixels):

    # Create a dictionary mapping cell labels to pixel values and their counts
    cells = defaultdict(lambda: defaultdict(int))
    for i in tqdm.tqdm(range(segmentation_final.shape[0])):
        for j in range(segmentation_final.shape[1]):
            label = segmentation_final[i, j]
            if label != 0:
                pixel_value = pixels[i, j]
                cells[label][pixel_value] += 1

    return to_dict(cells)


def spots(segmentation_final, pixels, df):

    # Create a dictionary mapping spot IDs to cell labels and their counts
    spots = defaultdict(lambda: defaultdict(int))

    for i in tqdm.tqdm(range(pixels.shape[0])):
        for j in range(pixels.shape[1]):
            pixel_value = pixels[i, j]
            if pixel_value != -1:
                label = segmentation_final[i, j]
                spots[pixel_value][label] += 1

    # Ensure that all spots in tissue are included, even if they have no labels
    for spot_id in list(df[df.in_tissue == 1].index):
        if spot_id not in spots.keys():
            spots[spot_id] = {0: 1}

    return to_dict(spots)


def create_cells_main(cells, spots, cells_main_pct, max_spot):

    # Identify main spots for each cell based on a percentage threshold
    cells_main = {}
    for cell in list(cells.keys()):
        cells_main[cell] = []
        for spot in cells[cell]:
            if spot > 0:
                if cells[cell][spot] > cells_main_pct * sum(spots[spot].values()):
                    cells_main[cell].append(spot)

        # Remove cells that don't meet criteria
        if cells_main[cell] == [] or len(cells_main[cell]) > max_spot:
            del cells_main[cell]

    return cells_main


def knn(
    spots_result, cell_ids, cells_main, start_row_spot, start_col_spot, index_toset
):

    # Compute the k-nearest neighbors for cell centers
    cell_coords_dict = {}

    for ids in range(len(cell_ids)):
        cell_id = cell_ids[ids]
        cell_coords_dict[cell_id] = []
        for spot_id in cells_main[float(cell_id)]:
            i, j = index_toset[spot_id]
            spots_result[i - start_row_spot, j - start_col_spot] = cell_id
            cell_coords_dict[cell_id].append((i, j))

    # Calculate cell centers as the mean of their coordinates
    cell_centers = {}
    for cell_id, coords in cell_coords_dict.items():
        coords_array = np.array(coords)
        center_x, center_y = np.mean(coords_array, axis=0)
        cell_centers[cell_id] = (center_x, center_y)

    center_coords = np.array(list(cell_centers.values()))

    # Use NearestNeighbors to compute distances and indices
    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(n_neighbors=4, algorithm="auto", metric="euclidean")
    nn_model.fit(center_coords)

    distances, indices = nn_model.kneighbors(center_coords)

    return cell_centers, distances, indices


def add_segmentation_temp(segmentation_results, i_max, j_max, loop, gap):

    # Combine segmentation results from different blocks into a final segmentation map
    segmentation_results1 = copy.deepcopy(segmentation_results)
    num = 0

    # Adjust labels to ensure uniqueness across blocks
    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            segmentation_results1[(i, j)][segmentation_results1[(i, j)] != 0] = (
                segmentation_results1[(i, j)][segmentation_results1[(i, j)] != 0] + num
            )
            num = max(num, segmentation_results1[(i, j)].max())

    segmentation_final = np.zeros((i_max, j_max))

    # Merge the blocks into the final segmentation map
    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            if i == 0 and j == 0:
                segmentation_final[
                    i : min(i + loop, i_max), j : min(j + loop, j_max)
                ] = segmentation_results1[(i, j)]
            elif i == 0 and j != 0:
                segmentation_final[
                    i : min(i + loop, i_max), j : min(j + loop, j_max)
                ] = segmentation_results1[(i, j)][:, gap:]
            elif i != 0 and j == 0:
                segmentation_final[
                    i : min(i + loop, i_max), j : min(j + loop, j_max)
                ] = segmentation_results1[(i, j)][gap:, :]
            else:
                segmentation_final[
                    i : min(i + loop, i_max), j : min(j + loop, j_max)
                ] = segmentation_results1[(i, j)][gap:, gap:]

    t = 0

    # Resolve overlaps between blocks
    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            if j != 0:
                if i == 0:
                    a = copy.deepcopy(segmentation_results1[(i, (j - loop))][:, -gap:])
                    b = copy.deepcopy(segmentation_results1[(i, j)][:, :gap])
                    uni_a = np.unique(a[:, -1:])
                    uni_a = uni_a[uni_a > 0]
                    for ele in uni_a:
                        uni_b = np.unique(b[a == ele])
                        uni_b = uni_b[uni_b > 0]

                        if len(uni_b) > 0:

                            a[a == ele] = 0
                            for eleb in uni_b:
                                a[b == eleb] = eleb

                    segmentation_final[
                        i : min(i + loop, i_max), min(j, j_max) - gap : min(j, j_max)
                    ] = copy.deepcopy(a)

                else:
                    a = copy.deepcopy(
                        segmentation_results1[(i, (j - loop))][gap:, -gap:]
                    )
                    b = copy.deepcopy(segmentation_results1[(i, j)][gap:, :gap])
                    uni_a = np.unique(a[:, -1:])
                    uni_a = uni_a[uni_a > 0]
                    for ele in uni_a:
                        uni_b = np.unique(b[a == ele])
                        uni_b = uni_b[uni_b > 0]
                        if len(uni_b) > 0:
                            t = t + 1
                            a[a == ele] = 0

                            for eleb in uni_b:
                                a[b == eleb] = eleb

                    segmentation_final[
                        i : min(i + loop, i_max), min(j, j_max) - gap : min(j, j_max)
                    ] = copy.deepcopy(a)

            if i != 0:
                if j == 0:
                    a = copy.deepcopy(segmentation_results1[((i - loop), j)][-gap:, :])
                    b = copy.deepcopy(segmentation_results1[(i, j)][:gap, :])
                    uni_a = np.unique(a[-1:, :])
                    uni_a = uni_a[uni_a > 0]
                    for ele in uni_a:
                        uni_b = np.unique(b[a == ele])
                        uni_b = uni_b[uni_b > 0]
                        if len(uni_b) > 0:
                            t = t + 1
                            a[a == ele] = 0
                            for eleb in uni_b:
                                a[b == eleb] = eleb

                    segmentation_final[
                        (min(i, i_max) - gap) : min(i, i_max), j : min(j + loop, j_max)
                    ] = copy.deepcopy(a)

                else:

                    a = copy.deepcopy(
                        segmentation_results1[((i - loop), j)][-gap:, gap:]
                    )
                    b = copy.deepcopy(segmentation_results1[(i, j)][:gap, gap:])
                    uni_a = np.unique(a[-1:, :])
                    uni_a = uni_a[uni_a > 0]
                    for ele in uni_a:
                        uni_b = np.unique(b[a == ele])
                        uni_b = uni_b[uni_b > 0]
                        if len(uni_b) > 0:
                            t = t + 1
                            a[a == ele] = 0
                            for eleb in uni_b:
                                a[b == eleb] = eleb

                    segmentation_final[
                        (min(i, i_max) - gap) : min(i, i_max), j : min(j + loop, j_max)
                    ] = copy.deepcopy(a)

    return segmentation_final


class spatial_object:
    """
    A class to represent and manage spatial data, including images, spot information, and cell segmentation.

    This class encapsulates spatial data and provides methods to manipulate and analyze the data,
    such as cropping images, adding segmentation results, and generating cell and spot information.

    :param image:
        The full-resolution tissue image as a NumPy array.
    :type image: numpy.ndarray

    :param df:
        A pandas DataFrame containing spot position data.
    :type df: pandas.DataFrame

    :param df_temp:
        A temporary pandas DataFrame with spot data within specified ranges.
    :type df_temp: pandas.DataFrame

    :param start_row_spot:
        The starting index of the row for spots in tissue.
    :type start_row_spot: int

    :param end_row_spot:
        The ending index of the row for spots in tissue.
    :type end_row_spot: int

    :param start_col_spot:
        The starting index of the column for spots in tissue.
    :type start_col_spot: int

    :param end_col_spot:
        The ending index of the column for spots in tissue.
    :type end_col_spot: int

    :param row_left:
        The left boundary (row index) for cropping the image.
    :type row_left: int

    :param row_right:
        The right boundary (row index) for cropping the image.
    :type row_right: int

    :param col_up:
        The upper boundary (column index) for cropping the image.
    :type col_up: int

    :param col_down:
        The lower boundary (column index) for cropping the image.
    :type col_down: int

    :param pixels:
        A NumPy array representing pixel assignments, initialized with -1.
    :type pixels: numpy.ndarray

    :param image_format:
        The format of the image, either 'HE' or 'DAPI'.
    :type image_format: str
    """

    def __init__(
        self,
        image,
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
    ):

        # Initialize the spatial object with image data and relevant parameters
        self.image = image
        self.df = df
        self.df_temp = df_temp
        self.image_format = image_format

        self.start_row_spot = start_row_spot
        self.end_row_spot = end_row_spot
        self.start_col_spot = start_col_spot
        self.end_col_spot = end_col_spot

        self.row_left = row_left
        self.row_right = row_right
        self.col_up = col_up
        self.col_down = col_down

        self.pixels = pixels

        self.cells = None
        self.spots = (None,)
        self.cells_main = None
        self.segmentation_final = None
        self.set_toindex = None
        self.index_toset = None
        self.distances = None
        self.indices = None
        self.spots_result = None
        self.cell_centers = None
        self.pixels_cells = None
        self.final_nuclei = None

    def image_temp(self):
        """
        Returns the cropped image based on the specified boundaries.

        :return:
            The cropped image as a NumPy array. If `image_format` is 'HE', the image is returned with all color channels.
            If `image_format` is 'DAPI', the image is returned as a single channel.
        :rtype: numpy.ndarray

        :raises ValueError:
            If `image_format` is not 'HE' or 'DAPI'.
        """

        # Return the cropped image based on the specified boundaries
        if self.image_format == "HE":
            return self.image[
                self.row_left : self.row_right, self.col_up : self.col_down, :
            ]
        elif self.image_format == "DAPI":
            return self.image[
                self.row_left : self.row_right, self.col_up : self.col_down
            ]
        else:
            print("Please input 'HE' or 'DAPI'.")

    def add_segmentation(self, segmentation_results, i_max, j_max, loop, gap):
        """
        Adds segmentation results to the spatial object.

        :param segmentation_results:
            The segmentation results to be added.
        :type segmentation_results: any (depends on the format returned by `add_segmentation_temp`)

        :param i_max:
            The maximum index along the first dimension.
        :type i_max: int

        :param j_max:
            The maximum index along the second dimension.
        :type j_max: int

        :param loop:
            Parameter specifying the loop count or iterations.
        :type loop: int

        :param gap:
            The gap parameter used in segmentation adjustments.
        :type gap: int

        :return:
            None. The segmentation results are stored in `self.segmentation_final`.
        :rtype: None
        """

        # Add segmentation results to the spatial object
        self.segmentation_final = add_segmentation_temp(
            segmentation_results, i_max, j_max, loop, gap
        )

    def generate_cell_spots_information(self, max_spot=50, cells_main_pct=float(1 / 6)):
        """
        Generates cells and spots information and creates a nearest neighbor network.

        This method processes the segmentation data to extract cell and spot information.
        It filters cells based on specified criteria and constructs a k-nearest neighbors (kNN) network.

        :param max_spot:
            The maximum number of spots per cell to consider. Defaults to `50`.
        :type max_spot: int, optional

        :param cells_main_pct:
            The minimum percentage of a cell that must be present in a spot to be considered.
            Defaults to `1/6`.
        :type cells_main_pct: float, optional

        :return:
            None. The method updates the spatial object attributes with cells, spots, and network information.
        :rtype: None
        """

        # Generate cells and spots information, and create a nearest neighbor network
        print("Generating cells information.")
        self.cells = cells(self.segmentation_final, self.pixels)

        print("Generating spots information.")

        self.spots = spots(self.segmentation_final, self.pixels, self.df)

        max_len = 0
        for i, inner_dic in self.spots.items():
            max_len = max(max_len, len(inner_dic))

        # print(f"The largest number of nucleis per spot is {max_len}.")

        print("Filtering cells")

        self.cells_main = create_cells_main(
            self.cells, self.spots, cells_main_pct, max_spot
        )
        print(f"We have {len(self.cells_main)} nuclei in total.")

        print("Creating NN network")

        self.set_toindex = {}
        self.index_toset = {}
        indexes = list(self.df.barcode)
        for i in range(len(indexes)):
            self.set_toindex[
                (
                    int(indexes[i].split("_")[2]),
                    int(indexes[i].split("_")[3].split("-")[0]),
                )
            ] = i
            self.index_toset[i] = (
                int(indexes[i].split("_")[2]),
                int(indexes[i].split("_")[3].split("-")[0]),
            )

        self.spots_result = np.zeros(
            [
                self.end_row_spot - self.start_row_spot,
                self.end_col_spot - self.start_col_spot,
            ]
        )
        self.cell_ids = np.sort(np.array(list(self.cells_main.keys())))

        self.cell_centers, self.distances, self.indices = knn(
            self.spots_result,
            self.cell_ids,
            self.cells_main,
            self.start_row_spot,
            self.start_col_spot,
            self.index_toset,
        )
