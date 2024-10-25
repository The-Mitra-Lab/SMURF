import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from py3nvml.py3nvml import *
    from torch.nn.functional import cosine_similarity, softmax
except ImportError:
    torch = None
    nn = None
    optim = None
    softmax = None
    cosine_similarity = None


def softmax_1(x):
    # Custom softmax function for NumPy arrays
    e_x = np.exp(x - np.max(x, axis=0))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)


def start_optimization(
    spots_X,
    celltype_X,
    cells_X_plus,
    nonzero_indices_dic,
    device,
    num_epochs=1000,
    learning_rate=0.1,
    print_each=100,
    epsilon=1e-3,
    random_seed=42,
    print_memory=False,
):

    # Function to start the optimization process using PyTorch
    if (
        torch is None
        or nn is None
        or optim is None
        or softmax is None
        or cosine_similarity is None
    ):
        raise ImportError(
            "Please install the 'advanced' dependencies to use this function."
        )

    class CustomLayer(nn.Module):
        # Custom neural network layer
        def __init__(
            self, num_spots, num_cells, nonzero_indices, device, A, C_D, C_true
        ):
            super(CustomLayer, self).__init__()
            self.num_spots = num_spots
            self.num_cells = num_cells
            self.nonzero_indices = nonzero_indices
            self.device = device
            self.A = A
            self.C_D = C_D
            self.C_true = C_true
            self.weights = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(len(indices)).to(device))
                    for indices in nonzero_indices
                ]
            )

        def forward(self, x):
            # Forward pass of the model
            batch_size = x.size(0)
            # batch_size = 1
            output = torch.zeros(batch_size, self.num_cells).to(x.device)

            for i, indices in enumerate(self.nonzero_indices):
                # Apply softmax to weights to get normalized weights
                normalized_weights = torch.softmax(self.weights[i], dim=0)
                # Update the output tensor with weighted values
                output[i, indices] = x[:, i].unsqueeze(1)[i, :] * normalized_weights
            return output

        def train(
            self, num_epochs=1000, learning_rate=0.1, print_each=100, epsilon=1e-3
        ):

            # Training function for the custom layer
            optimizer = optim.Adam(self.parameters(), lr=0.1)

            losses = []
            prev_loss = None

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                # Compute the predicted cell type matrix
                C_pred = (
                    torch.matmul(
                        self.A, self.forward(torch.eye(self.num_spots).to(self.device))
                    )
                    + self.C_D
                )
                # Calculate loss using cosine similarity
                loss = 1 - cosine_similarity(C_pred.T, self.C_true).mean()
                loss.backward()
                optimizer.step()
                loss_value = loss.item()
                losses.append(loss_value)

                # Print loss every 'print_each' epochs
                if epoch % print_each == 0:
                    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

                # Early stopping if loss improvement is minimal
                if prev_loss is not None and (prev_loss - loss_value) <= epsilon:
                    print(
                        f"Stopping early at epoch {epoch} due to minimal loss improvement."
                    )
                    break

                prev_loss = loss_value

    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)

    spot_cell_dic = {}

    # Iterate over each cell group
    for cell_num in spots_X.keys():

        print("Group " + str(cell_num) + ":")

        num_spots = spots_X[cell_num].shape[0]
        num_cells = celltype_X[cell_num].shape[0]

        # Convert NumPy arrays to PyTorch tensors and move to the specified device
        A = torch.from_numpy(spots_X[cell_num].T).float().to(device)
        C_true = torch.from_numpy(celltype_X[cell_num]).float().to(device)
        C_D = torch.from_numpy(cells_X_plus[cell_num].T).float().to(device)

        # Initialize and train the model for the current group
        model = CustomLayer(
            num_spots, num_cells, nonzero_indices_dic[cell_num], device, A, C_D, C_true
        ).to(device)
        model.train(
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            print_each=print_each,
            epsilon=epsilon,
        )

        # Extract the learned weights from the model
        arrays = [p.detach().numpy() for p in model.weights.to("cpu")]
        spot_cell_dic[cell_num] = [list(softmax_1(np.array(arr))) for arr in arrays]

        if print_memory:

            # Initialize NVML
            nvmlInit()

            # Get handle for the first GPU
            handle = nvmlDeviceGetHandleByIndex(0)

            # Get memory info
            info = nvmlDeviceGetMemoryInfo(handle)
            print(f"Total memory: {info.total / (1024 ** 2)} MB")
            print(f"Used memory: {info.used / (1024 ** 2)} MB")
            print(f"Free memory: {info.free / (1024 ** 2)} MB")

            # Close NVML
            nvmlShutdown()

        del model, A, C_true, C_D
        torch.cuda.empty_cache()

    return spot_cell_dic
