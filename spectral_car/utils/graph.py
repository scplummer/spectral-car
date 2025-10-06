"""
Graph Construction Utilities functions for spectral CAR models.
"""
from typing import Tuple, Optional

import torch
from scipy.spatial import Delaunay

def create_grid_graph_laplacian(
    n_nodes: int, 
    grid_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create graph Laplacian for a regular 2D grid.
    
    Constructs a square grid graph with 4-connectivity (rook adjacency),
    then computes the graph Laplacian L = D - W and its eigendecomposition.
    
    Args:
        n_nodes: Total number of nodes (should equal grid_size^2)
        grid_size: Size of square grid (produces grid_size x grid_size graph)
        
    Returns:
        eigenvalues: Eigenvalues of Laplacian (n_nodes,)
        eigenvectors: Eigenvectors of Laplacian (n_nodes, n_nodes)
        
    Example:
        >>> # Create 8x8 grid (64 nodes)
        >>> eigenvalues, eigenvectors = create_grid_graph_laplacian(64, 8)
        >>> eigenvalues.shape
        torch.Size([64])
    """
    if n_nodes != grid_size ** 2:
        raise ValueError(f"n_nodes ({n_nodes}) must equal grid_size^2 ({grid_size**2})")
    
    # Create adjacency matrix
    W = torch.zeros(n_nodes, n_nodes)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            
            # Right neighbor
            if j < grid_size - 1:
                W[idx, idx + 1] = 1
                W[idx + 1, idx] = 1
            
            # Bottom neighbor
            if i < grid_size - 1:
                W[idx, idx + grid_size] = 1
                W[idx + grid_size, idx] = 1
    
    # Degree matrix
    D = torch.diag(W.sum(dim=1))
    
    # Graph Laplacian
    L = D - W
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    return eigenvalues, eigenvectors


def create_adjacency_from_coords(
    coords: torch.Tensor,
    adjacency_type: str = 'knn',
    k: int = 5,
    threshold: Optional[float] = None
) -> torch.Tensor:
    """
    Create adjacency matrix from spatial coordinates.
    
    Args:
        coords: Spatial coordinates (n_nodes, d) where d is dimension (usually 2)
        adjacency_type: Type of adjacency ('knn', 'threshold', or 'delaunay')
        k: Number of nearest neighbors (for 'knn')
        threshold: Distance threshold (for 'threshold')
        
    Returns:
        W: Adjacency matrix (n_nodes, n_nodes)
        
    Example:
        >>> coords = torch.randn(100, 2)  # 100 random 2D points
        >>> W = create_adjacency_from_coords(coords, adjacency_type='knn', k=5)
    """
    n_nodes = coords.shape[0]
    
    # Compute pairwise distances
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (n, n, d)
    distances = torch.sqrt(torch.sum(diff**2, dim=2))  # (n, n)
    
    if adjacency_type == 'knn':
        # K-nearest neighbors
        W = torch.zeros(n_nodes, n_nodes)
        _, indices = torch.topk(distances, k=k+1, largest=False, dim=1)  # +1 to exclude self
        
        for i in range(n_nodes):
            neighbors = indices[i, 1:]  # Exclude self (first entry)
            W[i, neighbors] = 1
        
        # Make symmetric
        W = (W + W.T) / 2
        W = (W > 0).float()
        
    elif adjacency_type == 'threshold':
        # Distance threshold
        if threshold is None:
            # Auto-select threshold as median distance
            threshold = torch.median(distances[distances > 0]).item()
        
        W = (distances < threshold).float()
        W.fill_diagonal_(0)  # Remove self-loops
        
    elif adjacency_type == 'delaunay':
        # Delaunay triangulation (requires scipy)
        try:
            coords_np = coords.cpu().numpy()
            tri = Delaunay(coords_np)
            
            W = torch.zeros(n_nodes, n_nodes)
            for simplex in tri.simplices:
                # Add edges for this triangle
                W[simplex[0], simplex[1]] = 1
                W[simplex[1], simplex[0]] = 1
                W[simplex[1], simplex[2]] = 1
                W[simplex[2], simplex[1]] = 1
                W[simplex[2], simplex[0]] = 1
                W[simplex[0], simplex[2]] = 1
        except ImportError:
            raise ImportError("Delaunay triangulation requires scipy")
    else:
        raise ValueError(f"Unknown adjacency_type: {adjacency_type}")
    
    return W


def create_laplacian_from_adjacency(
    W: torch.Tensor,
    normalized: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create graph Laplacian from adjacency matrix.
    
    Args:
        W: Adjacency matrix (n_nodes, n_nodes)
        normalized: If True, compute normalized Laplacian L = I - D^(-1/2) W D^(-1/2)
                   If False, compute unnormalized Laplacian L = D - W
        
    Returns:
        L: Graph Laplacian (n_nodes, n_nodes)
        eigenvalues: Eigenvalues of L (n_nodes,)
        eigenvectors: Eigenvectors of L (n_nodes, n_nodes)
        
    Example:
        >>> W = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float)
        >>> L, eigenvalues, eigenvectors = create_laplacian_from_adjacency(W)
    """
    n_nodes = W.shape[0]
    
    # Compute degree matrix
    degrees = W.sum(dim=1)
    D = torch.diag(degrees)
    
    if normalized:
        # Normalized Laplacian: L = I - D^(-1/2) W D^(-1/2)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + 1e-8))
        L = torch.eye(n_nodes) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        # Unnormalized Laplacian: L = D - W
        L = D - W
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    return L, eigenvalues, eigenvectors


def add_graph_boundary(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    boundary_type: str = 'reflect'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add boundary conditions to graph Laplacian.
    
    Modifies eigendecomposition to incorporate different boundary conditions.
    Useful for handling edge effects in spatial models.
    
    Args:
        eigenvalues: Original eigenvalues (n_nodes,)
        eigenvectors: Original eigenvectors (n_nodes, n_nodes)
        boundary_type: Type of boundary ('reflect', 'periodic', 'dirichlet')
        
    Returns:
        eigenvalues_modified: Modified eigenvalues
        eigenvectors_modified: Modified eigenvectors
        
    Note:
        This is a simplified implementation. For full boundary treatment,
        consider constructing the graph with boundary nodes explicitly.
    """
    if boundary_type == 'reflect':
        # Reflecting boundary: eigenvalues unchanged, eigenvectors adjusted
        return eigenvalues, eigenvectors
    
    elif boundary_type == 'periodic':
        # Periodic boundary: modify smallest eigenvalues
        eigenvalues_modified = eigenvalues.clone()
        eigenvalues_modified[0] = eigenvalues_modified[1]  # Remove zero eigenvalue
        return eigenvalues_modified, eigenvectors
    
    elif boundary_type == 'dirichlet':
        # Dirichlet boundary: zero at boundaries (default for most graphs)
        return eigenvalues, eigenvectors
    
    else:
        raise ValueError(f"Unknown boundary_type: {boundary_type}")
