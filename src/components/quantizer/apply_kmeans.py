import joblib
import torch
import torch.nn as nn


class BatchKMeans(nn.Module):
    def __init__(self, km_path: str = None, device: str = "cuda"):
        """
        km_path: path to your joblib-saved sklearn.KMeans
        device: "cuda" or "cpu"
        """
        super().__init__()
        # load cluster centers: shape (n_clusters, D)
        if km_path is not None:
            km = joblib.load(km_path)
            centers = torch.from_numpy(km.cluster_centers_).float()
        else:
            centers = torch.randn(256, 512).float()  # default random centers
        
        # register as buffers so .to(device) moves them
        self.register_buffer("centers", centers)                 
        self.register_buffer("centers_norm", (centers**2).sum(1))# (n_clusters,)
        
        self.to(device)

    def forward(self, x: torch.Tensor, return_indices: bool = False):
        """
        x: Tensor of shape (B, D, T)
        return_indices: if True  → LongTensor of shape (B, T) with cluster IDs
                        else       → FloatTensor of shape (B, D, T) with centroids
        """
        B, D, T = x.shape
        
        # flatten time & batch dims into one
        flat = x.transpose(1, 2).reshape(-1, D)  # (B*T, D)
        
        # squared‐Euclidean distance to each center:
        dist2 = (
            flat.pow(2).sum(1, keepdim=True)           # (B*T, 1)
            - 2 * flat @ self.centers.t()              # (B*T, n_clusters)
            + self.centers_norm.unsqueeze(0)           # (1,  n_clusters)
        )
        
        # nearest cluster for each vector
        ids = dist2.argmin(dim=1)                     # (B*T,)
        
        if return_indices:
            return ids.view(B, T)                     # (B, T)
        else:
            # gather centroids and restore (B, D, T)
            quant = self.centers[ids]                 # (B*T, D)
            quant = quant.view(B, T, D)               # (B, T, D)
            return quant.transpose(1, 2)              # (B, D, T)


class BatchKMeansResidualQuantizer(nn.Module):
    def __init__(self, centers_path: str = None, device: str = "cuda"):
        """
        centers_path: path to a .pth file containing a tensor of shape
                      (I, n_clusters, feature_dim), where I is max iterations.
                      (e.g. torch.save(all_centers_tensor, "km256_centers.pth"))
        device: "cuda" or "cpu"
        """
        super().__init__()

        if centers_path is None:
            all_centers = torch.randn(8, 256, 512)  # default random centers
        else:
            # Load the stacked centers tensor of shape (I, K, D)
            all_centers = torch.load(centers_path, map_location="cpu")
            # all_centers.shape = (num_iters_total, n_clusters, feature_dim)

        if all_centers.ndim != 3:
            raise ValueError(
                f"Expected centers tensor of shape (I, K, D), got {all_centers.shape}"
            )

        # Number of available residual iterations (I), number of clusters (K), feature dimension (D)
        num_iters_total, n_clusters, feature_dim = all_centers.shape

        # Register as buffers so they move with .to(device)
        # Shape: (I, K, D)
        self.register_buffer("centers_all", all_centers.float())
        # Precompute norm² of each center: sum over feature_dim → shape (I, K)
        centers_norm = (all_centers ** 2).sum(dim=2)  # (I, K)
        self.register_buffer("centers_norm_all", centers_norm.float())

        self.num_iters_total = num_iters_total
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim

        # Move buffers to target device
        self.to(device)

    def forward(self, x: torch.Tensor, n_q: int = None, return_indices: bool = False):
        """
        x: Tensor of shape (B, D, T)
        n_q: number of residual quantization iterations to perform (≤ num_iters_total)
        return_indices: if True, returns all iteration cluster IDs as a LongTensor
                        of shape (B, n_q, T); otherwise returns the quantized output (B, D, T).

        The quantized output is the sum of centroids over n_q residual iterations.
        """
        if n_q is None:
            n_q = self.num_iters_total
        if n_q < 1 or n_q > self.num_iters_total:
            raise ValueError(
                f"n_q must be between 1 and {self.num_iters_total}, got {n_q}"
            )

        B, D, T = x.shape
        if D != self.feature_dim:
            raise ValueError(
                f"Input feature dimension D={D} does not match loaded centers’ "
                f"D={self.feature_dim}"
            )

        # Flatten batch and time dims: (B*T, D)
        flat_res = x.transpose(1, 2).reshape(-1, D)  # (N = B*T, D)

        # Prepare tensor to accumulate the quantized sums
        quant_sum_flat = torch.zeros_like(flat_res)  # (B*T, D)

        # List to collect IDs for each iteration: each is (B*T,)
        ids_list = []

        for i in range(n_q):
            # centers_i: (K, D), centers_norm_i: (K,)
            centers_i = self.centers_all[i]            # (n_clusters, D)
            centers_norm_i = self.centers_norm_all[i]  # (n_clusters,)

            # Compute ‖flat_res‖²: shape (N, 1)
            flat_norm2 = flat_res.pow(2).sum(dim=1, keepdim=True)  # (N, 1)

            # Compute pairwise squared distances: ‖x - c‖² = ‖x‖² + ‖c‖² - 2 x·cᵀ
            # Result shape: (N, n_clusters)
            dist2 = (
                flat_norm2
                - 2.0 * flat_res @ centers_i.t()
                + centers_norm_i.unsqueeze(0)
            )

            # Find nearest cluster ID for each vector
            ids_i = dist2.argmin(dim=1)  # (N,)

            # Gather that iteration’s centroid for each vector: (N, D)
            quant_i_flat = centers_i[ids_i]  # (N, D)

            # Accumulate into quant_sum_flat
            quant_sum_flat += quant_i_flat  # (N, D)

            # Prepare residual for next iteration
            flat_res = flat_res - quant_i_flat  # (N, D)

            # Store these IDs
            ids_list.append(ids_i)

        # Reshape quant_sum_flat back to (B, D, T)
        quant_sum = quant_sum_flat.view(B, T, D).permute(0, 2, 1)  # (B, D, T)

        if return_indices:
            # Stack IDs: (n_q, N)
            stacked = torch.stack(ids_list, dim=0)           # (n_q, B*T)
            # Reshape to (n_q, B, T)
            stacked = stacked.view(n_q, B, T)                # (n_q, B, T)
            # Permute to (B, n_q, T)
            return stacked.permute(1, 0, 2)                   # (B, n_q, T)
        else:
            return quant_sum
