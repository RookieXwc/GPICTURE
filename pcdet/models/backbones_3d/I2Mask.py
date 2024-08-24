import torch
import torch.nn as nn
import itertools
from sklearn.cluster import KMeans

class I2Mask_func(nn.Module):
    def __init__(self, n_clusters=8, n_partition=[3, 3, 2], lambda_threshold=0.6, base_mask_ratio=[0.9, 0.45, 0]):
        super(I2Mask_func, self).__init__()
        self.n_clusters = n_clusters
        self.n_partition = n_partition
        self.lambda_threshold = lambda_threshold
        self.base_mask_ratio = base_mask_ratio

    def forward(self, seal):
        # Verify the correctness of input dimensions.
        if seal.size(1) != 64 or len(seal.size()) != 2:
            raise ValueError("Input tensor must be of shape [n, 64]")

        # Convert the PyTorch Tensor to a NumPy array for K-means clustering.
        seal_np = seal.cpu().detach().numpy()

        # Apply K-means clustering.
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        labels = kmeans.fit_predict(seal_np)

        # Convert the clustering labels back to a PyTorch Tensor.
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        cluster_centers = self.get_cluster_centers(seal, labels_tensor)

        # Invoke the inter_class_discrimination_guided_masking function.
        partitions = self.inter_class_discrimination_guided_masking(cluster_centers)

        D_intra, r_c = self.intra_class_discrimination_guided_masking(seal, labels_tensor, cluster_centers)

        r_k = []
        for i in range(self.n_clusters):
            r_ki = (self.base_mask_ratio[0] * float(i in partitions[0]) + self.base_mask_ratio[1] * float(i in partitions[1]) +
            self.base_mask_ratio[2] * float(i in partitions[2])) * r_c[i]
            r_k.append(r_ki)

        voxel_mask = self.masking(labels_tensor, r_k)

        return voxel_mask

    def masking(self, labels_tensor, r_k):
        mask = torch.zeros_like(labels_tensor, dtype=torch.float32)
        for i in range(self.n_clusters):
            # Retrieve the index of the current class.
            class_indices = (labels_tensor == i).nonzero(as_tuple=True)[0]
            # Calculate the number of masks needed.
            mask_count = int(len(class_indices) * r_k[i])
            # Randomly select the indices to be masked.
            mask_indices = class_indices[torch.randperm(len(class_indices))[:mask_count]]
            # Set the masks.
            mask[mask_indices] = 1.0
        return mask

    def inter_class_discrimination_guided_masking(self, cluster_centers):
        # Invoke the fastest_class_sampling function.
        partitions = self.fastest_class_sampling(cluster_centers)

        return partitions

    def intra_class_discrimination_guided_masking(self, seal, labels, class_centers):
        D_intra = []
        for i in range(self.n_clusters):
            # Extract samples belonging to class i.
            class_samples = seal[labels == i]
            class_center = class_centers[i]

            # Calculate the distance of each sample to its class center.
            distances = torch.norm(class_samples - class_center, dim=1)

            # Calculate the number of distances exceeding the threshold lambda.
            greater_than_lambda = distances > self.lambda_threshold

            # Calculate D_intra.
            if greater_than_lambda.sum() > 0:
                D_intra_i = distances[greater_than_lambda].mean()
            else:
                D_intra_i = torch.tensor(0.0)
            D_intra.append(D_intra_i)

        # Calculate the intra-class consistency coefficient r_ci.
        r_c = 1 - (D_intra / D_intra.max())

        return D_intra, r_c

    def get_cluster_centers(self, seal, labels):
        # Compute the center for each class.
        centers = []
        for i in range(self.n_clusters):
            centers.append(torch.mean(seal[labels == i], dim=0))
        return torch.stack(centers)

    def fastest_class_sampling(self, cluster_centers):
        B = list(range(self.n_clusters))  # Assuming class indices range from 0 to n_clusters-1.
        Bt_partitions = []

        for t in range(len(self.n_partition)):
            Bt = []
            combinations = itertools.combinations(B, self.n_partition[t])
            max_avg_distance = 0
            selected_partition = None

            for combination in combinations:
                avg_distance = self.compute_avg_interclass_distance(cluster_centers, combination)
                if avg_distance > max_avg_distance:
                    max_avg_distance = avg_distance
                    selected_partition = combination

            Bt_partitions.append(selected_partition)
            B = [b for b in B if b not in selected_partition]

        return Bt_partitions

    def compute_avg_interclass_distance(self, cluster_centers, combination):
        distances = []
        for i, j in itertools.combinations(combination, 2):
            distance = 1 - torch.dot(cluster_centers[i], cluster_centers[j]) / (torch.norm(cluster_centers[i]) * torch.norm(cluster_centers[j]))
            distances.append(distance)
        return sum(distances) / len(distances)