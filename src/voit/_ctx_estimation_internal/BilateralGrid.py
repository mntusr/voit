from typing import Any, Sequence, Union

#  import seaborn as sns
from scipy.sparse import diags
from scipy.sparse.linalg import cg

MAX_VAL = 255.0
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv

RGB_TO_YUV = np.array(
    [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]
)
YUV_TO_RGB = np.array([[1.0, 0.0, 1.402], [1.0, -0.34414, -0.71414], [1.0, 1.772, 0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)


def rgb2yuv(im: np.ndarray) -> np.ndarray:
    return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET


def yuv2rgb(im: np.ndarray) -> np.ndarray:
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([2], [1]))


##############################################################################
REQUIRES_CONF_GRAD = True
##############################################################################


def get_valid_idx(valid, candidates):
    """Find which values are present in a list and where they are located"""
    locs = np.searchsorted(valid, candidates)
    # Handle edge case where the candidate is larger than all valid values
    locs = np.clip(locs, 0, len(valid) - 1)
    # Identify which values are actually present
    valid_idx = np.flatnonzero(valid[locs] == candidates)
    locs = locs[valid_idx]
    return valid_idx, locs


class BilateralGrid(object):
    def __init__(
        self,
        im: np.ndarray,
        sigma_spatial: float = 32,
        sigma_luma: float = 8,
        sigma_chroma: float = 8,
    ):
        im_yuv = rgb2yuv(im)
        # Compute 5-dimensional XYLUV bilateral-space coordinates
        Iy, Ix = np.mgrid[: im.shape[0], : im.shape[1]]
        x_coords = (Ix / sigma_spatial).astype(int)
        y_coords = (Iy / sigma_spatial).astype(int)
        luma_coords = (im_yuv[..., 0] / sigma_luma).astype(int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
        coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.npixels, self.dim = coords_flat.shape
        # Hacky "hash vector" for coordinates,
        # Requires all scaled coordinates be < MAX_VAL
        self.hash_vec = MAX_VAL ** np.arange(self.dim)
        # Construct S and B matrix
        self._compute_factorization(coords_flat)

    def _compute_factorization(self, coords_flat: np.ndarray) -> None:
        # Hash each coordinate in grid to a unique value
        hashed_coords = self._hash_coords(coords_flat)
        unique_hashes, unique_idx, idx = np.unique(
            hashed_coords, return_index=True, return_inverse=True
        )
        # Identify unique set of vertices
        unique_coords = coords_flat[unique_idx]
        self.nvertices = len(unique_coords)
        # Construct sparse splat matrix that maps from pixels to vertices
        self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
        # Construct sparse blur matrices.
        # Note that these represent [1 0 1] blurs, excluding the central element
        self.blurs: list[csr_matrix] = []
        for d in range(self.dim):
            blur = 0.0
            for offset in (-1, 1):
                offset_vec = np.zeros((1, self.dim))
                offset_vec[:, d] = offset
                neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
                blur = blur + csr_matrix(
                    (np.ones((len(valid_coord),)), (valid_coord, idx)),
                    shape=(self.nvertices, self.nvertices),
                )
            self.blurs.append(blur)

    def _hash_coords(self, coord: np.ndarray) -> np.ndarray:
        """Hacky function to turn a coordinate into a unique value"""
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

    def splat(self, x: np.ndarray) -> np.ndarray:
        return self.S.dot(x)

    def slice(self, y: np.ndarray) -> np.ndarray:
        return self.S.T.dot(y)

    def blur(self, x: np.ndarray) -> np.ndarray:
        """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
        assert x.shape[0] == self.nvertices
        out = 2 * self.dim * x
        for blur in self.blurs:
            out = out + blur.dot(x)
        return out

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Apply bilateral filter to an input x"""
        return self.slice(self.blur(self.splat(x))) / self.slice(
            self.blur(self.splat(np.ones_like(x)))
        )


def bistochastize(
    grid: BilateralGrid, maxiter: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Compute diagonal matrices to bistochastize a bilateral grid"""
    m = grid.splat(np.ones(grid.npixels))
    n = np.ones(grid.nvertices)
    for i in range(maxiter):
        n = np.sqrt(n * m / grid.blur(n))
    # Correct m to satisfy the assumption of bistochastization regardless
    # of how many iterations have been run.
    m = n * grid.blur(n)
    Dm = diags(m, 0)
    Dn = diags(n, 0)
    return Dn, Dm  # type: ignore


class BilateralSolver(object):
    def __init__(self, grid: BilateralGrid, params: dict[str, Any]):
        self.grid = grid
        self.params = params
        self.Dn, self.Dm = bistochastize(grid)

    def solve(self, x: np.ndarray, w: np.ndarray):
        # Check that w is a vector or a nx1 matrix
        if w.ndim == 2:
            assert w.shape[1] == 1
        elif w.ndim == 1:
            w = w.reshape(w.shape[0], 1)
        A_smooth = self.Dm - self.Dn.dot(self.grid.blur(self.Dn))
        w_splat = self.grid.splat(w)
        A_data = diags(w_splat[:, 0], 0)
        A = self.params["lam"] * A_smooth + A_data
        xw = x * w
        b = self.grid.splat(xw)
        # Use simple Jacobi preconditioner
        A_diag = np.maximum(A.diagonal(), self.params["A_diag_min"])
        M = diags(1 / A_diag, 0)
        # Flat initialization
        y0 = self.grid.splat(xw) / np.maximum(w_splat, 1e-10)
        yhat = np.empty_like(y0)
        for d in range(x.shape[-1]):
            yhat[..., d], info = cg(
                A,
                b[..., d],
                x0=y0[..., d],
                M=M,
                maxiter=self.params["cg_maxiter"],
                atol=self.params["cg_tol"],
            )
        xhat = self.grid.slice(yhat)

        return xhat, yhat

    def solveGrad(
        self, x: np.ndarray, w: np.ndarray, saved_yhat: np.ndarray, saved_target
    ) -> tuple[np.ndarray, Union[np.ndarray, None]]:
        # Check that w is a vector or a nx1 matrix
        if w.ndim == 2:
            assert w.shape[1] == 1
        elif w.ndim == 1:
            w = w.reshape(w.shape[0], 1)
        A_smooth = self.Dm - self.Dn.dot(self.grid.blur(self.Dn))
        w_splat = self.grid.splat(w)
        A_data = diags(w_splat[:, 0], 0)
        A = self.params["lam"] * A_smooth + A_data
        b = self.grid.splat(x)
        # Use simple Jacobi preconditioner
        A_diag = np.maximum(A.diagonal(), self.params["A_diag_min"])
        M = diags(1 / A_diag, 0)
        # Flat initialization
        # here we should make all w to 1
        w_1 = np.ones(w.shape, np.double)
        y0 = self.grid.splat(x * w_1) / self.grid.splat(w_1)
        yhat = np.empty_like(y0)
        for d in range(x.shape[-1]):
            yhat[..., d], info = cg(
                A,
                b[..., d],
                x0=y0[..., d],
                M=M,
                maxiter=self.params["cg_maxiter"],
                atol=self.params["cg_tol"],
            )
        grad_f_b = yhat

        slice_grad_f_b = self.grid.slice(grad_f_b)
        grad_t = slice_grad_f_b * w

        ### calculate grad for confidence
        if REQUIRES_CONF_GRAD == True:
            grad_diag_A = -1.0 * (grad_f_b * saved_yhat)
            grad_conf = self.grid.slice(grad_diag_A) + slice_grad_f_b * saved_target
        else:
            grad_conf = None
        return grad_t, grad_conf


def solve(grid, target, confidence, bs_params, im_shape):
    t = target.reshape(-1, im_shape[2]).astype(np.double)
    c = confidence.reshape(-1, 1).astype(np.double)  # / (pow(2,16)-1)
    xhat, yhat = BilateralSolver(grid, bs_params).solve(t, c)
    xhat = xhat.reshape(im_shape)
    return xhat, yhat


def solveForGrad(
    grid: BilateralGrid,
    grad_f_x: np.ndarray,
    confidence: np.ndarray,
    bs_params: dict[str, Any],
    im_shape: Sequence[int],
    yhat: np.ndarray,
    target: np.ndarray,
):
    grad = grad_f_x.reshape(-1, im_shape[2]).astype(np.double)
    c = confidence.reshape(-1, 1).astype(np.double)
    t = target.reshape(-1, im_shape[2]).astype(np.double)
    grad_t, grad_c = BilateralSolver(grid, bs_params).solveGrad(grad, c, yhat, t)
    grad_t = grad_t.reshape(im_shape)

    if REQUIRES_CONF_GRAD == True:
        assert grad_c is not None
        grad_c = grad_c.reshape(im_shape)
        grad_c = grad_c.sum(2)
    else:
        grad_c = None
    return grad_t, grad_c
