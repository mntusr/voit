We use a hierarchical tensor structure.

Basic hierarchy:

```
Im
|- Im_RGBLike
|- Im_Scalar
   |- DepthmapLike
   |- ZbufLike

Ims
|- Ims_RGBLike
|- Ims_Scalar
   |- DepthmapLikes
   |- ZbufLikes

Transform
|- Transform_3x3
   |- TProjMat
|- Transform_4x4

Cubemap

HalfEnvmapLikes

Vec
|-Col_Vec3
|-Col_Vec3h
|-Col_Vec4h

Val1d
|-Val1d_Mask

Vecs
|-Vecs3
```

In general, the top level elements in this hierarchy give dtype of the arrays and the rough meaning of the dimensions. The descendants then refine this.

# Im

Shape: `(C, H, W)`, where:

* C: The channel.
* H: The Y coordinate in the image.
* W: The X coordinate in the image.

The bottom left corner of the screen is `(X=0, Y=-1)`

Dtype: defined by the descendants

## Im_RGBLike

The image contains RGB-like data. The image has the following channel order: `R, G, B`. A such array might contain linear RGB or sRGB data, but it should not mix them within the array.

Dtype: `np.floating`

## Im_Scalar

The image contains a single channel.

Dtype: `np.floating`

### DepthmapLike

The image contains metric depth-like data. Greater positive values mean that the image is farther from the camera.

### ZbufLike

The image contains zbuffer-like data. The following formula is used to calculate the zbuffer-like data from the depthmap-like data:

* $z=\frac{a·d + b}{d}$, where:
  * $z$: the zbuffer-like data
  * $a, b$: camera-dependent real constants 
  * $d$: the equivalent depth data

## Im_Mask

The image contains mask data.

Dtype: `np.bool_`


# Ims

Shape: `(N, C, H, W)`, where:

* N: The sample.
* C: The channel.
* H: The Y coordinate in the image.
* W: The X coordinate in the image.

The array contains multiple images.

The bottom left corner of the screen is `(X=0, Y=-1)`

Dtype: defined by the descendants

## Ims_RGBLike

The images contain RGB-like data. The images have the following channel order: `R, G, B`. A such array might contain linear RGB or sRGB data, but it should not mix them within the whole array.

Dtype: `np.floating`

## Ims_Scalar

The array contains a single channel per image.

Dtype: `np.floating`

### DepthmapLikes

The images contain metric depth-like data. Greater positive values mean that the pixels are farther from the camera.

### ZbufLikes

The images contain zbuffer-like data. The following formula is used to calculate the zbuffer-like data from the depthmap-like data:

* $z=\frac{a·d + b}{d}$, where:
  * $z$: the zbuffer-like data
  * $a, b$: camera-dependent real constants 
  * $d$: the equivalent depth data

## Ims_Mask

The images contain mask data.

Dtype: `np.bool_`


# Transform

Shape: `(Row, Col)`, where:

* `Row`: The index of the row of the matrix.
* `Col`: The index of the column of the matrix.

These arrays describe matrices that transform *column vectors*.

Dtype: `np.floating`

## Transform_3x3

The number of rows and columns in the matrix are 3.

### TProjMat

The matrix describes a theoretical projection matrix of a camera. Unlike the actual projection matrix, this does not specify the data in the zbuffer.

The matrix projects the points of the Y-up left handed view space to the image space. The coordinates in the image space range from -1 to 1. The bottom left corner of the screen is `(-1, -1)`.

Structure: $\begin{bmatrix} s_x&0&c_x \\ 0&s_y&c_y \\ 0&0&1 \end{bmatrix}$

## Transform_4x4

The number of rows and columns in the matrix are 4.

# Cubemap

Shape: `(F, C, H, W)`, where:

* `F`: The index of the face of the cubemap.
* `C`: The channel of the cubemap.
* `H`: The *local* Y-coordinate on a single face of a cubemap.
* `W`: The *local* X-coordinate on a single face of a cubemap.

These arrays contain a Panda3d cubemap. The channel order is `R, G, B`, the semantics of the other dimensions match to the semantics of the corresponding dimensions in the Panda3d cubemaps. These are described [here](https://docs.panda3d.org/1.10/python/programming/texturing/cube-maps)

Dtype: `np.floating`

# PerPixelHalfEnvmaps

Shape: `(H, W, El, Az, C)`, where:

* `H`: The horizontal dimension of the image.
* `W`: The vertical dimension of the image.
* `El`: The elevation-dimension for each environment map.
* `Az`: The azimuth-dimension for each environment map.
* `C`: The channel.

The environment maps are given using RGB channels with `R, G, B` channel order. The bottom left corner of the screen is `(X=0, Y=-1)`

Dtype: `np.floating`

# Vec

Shape: `(Row, 1)`, where:

* `Row`: The index of the row of the matrix.

The array contains a column vector.

Dtype: `np.floating`

## Col_Vec3

The array contains a column vector that does not use homogen coordinates. The vector has exactly 3 rows.

Structure: $\begin{bmatrix}x \\ y \\ z\end{bmatrix}$


## Col_Vec3h

The array contains a column vector that does uses homogen coordinates. The vector has exactly 3 rows and the last one contains 1.

Structure: $\begin{bmatrix}x \\ y \\ 1\end{bmatrix}$

## Col_Vec4h

The array contains a column vector that does uses homogen coordinates. The vector has exactly 4 rows and the last one contains 1.

Structure: $\begin{bmatrix}x \\ y \\ z \\ 1\end{bmatrix}$

# Val1d

Shape: `(V,)`, where:

* `V`: The index of the value.

These arrays are single-dimensional and contain some values.

Dtype: defined by the descendants

## Val1d_Mask

The arrays contain 1D mask data.

Dtype: `np.bool_`


# Vecs

Shape: `(N, Coord)`, where:

* `N`: The index of the point.
* `Coord`: The coordinate.

The array contains a set of points or vectors.

Dtype: `np.floating`

## Vecs3

The array contains vectors with three non-homogenous coordinates (shape: `(N, 3)`).