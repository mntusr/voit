VOIT (**V**irtual **O**bject **I**nsertion **T**oolkit) is a simple library to insert objects to images.

The library focuses on the ease of use and low-level control instead of the raw quality. 

The pre-existing object insertion tool, on which VOIT is heavily based, is in the repositories of the original authors: [inverse renderer](https://github.com/lzqsd/InverseRenderingOfIndoorScene), [object insertion tool](https://github.com/lzqsd/VirtualObjectInsertion).

Main features:

* Handle shadows.
* Handle reflections on the floor.
* Provide tools for authoring of datasets with inserted objects (editing, serialization, deserialization).

# Installation

VOIT depends on Pytorch, but it does not declare it as a requirement in its metadata. You have to manually install it based on the instructions [here](https://https://pytorch.org/). 

You can then install VOIT from PIP or its GitHub repository.

From PIP:

```
pip install voit
```

From GitHub:

```
pip install git+https://github.com/mntusr/voit
```

# Usage

The simplest example of using VOIT looks like this:

```python
import torch
import numpy as np
import voit
from pathlib import Path

# the image to which the object should be inserted
rgb_image: np.ndarray = ...

# the projection matrix of the camera
t_proj_mat: np.ndarray = ...

# where you want to insert the object
the_pixel_of_the_origin_of_the_inserted_object: voit.Vec2i = ...
the_depth_at_the_pixel_where_the_object_is_inserted: float = ...

# the normal vector of the previous surface
the_normal_of_the_surface_on_which_the_object_is_inserted = voit.Vec3 = ...

inserter = voit.Inserter(
    t_proj_mat=t_proj_mat,
    floor_proxy_size=voit.Vec2(5, 5), # good default for most cases
    im_size=voit.Vec2i(image.shape[2], image.shape[1]),
    shadow_map_size=1024, # good default for most cases
    pt_device=torch.device("cuda")
)

try:
    # the object to insert
    obj = inserter.load_model(Path("mymodel.glb"))

    result = inserter.insert(
        input_im=image,
        input_im_linear=False,
        output_im_linear=False,
        pos_px=the_pixel_of_the_origin_of_the_inserted_object,
        pos_depth=the_depth_at_the_pixel_where_the_object_is_inserted,
        normal_vs=the_normal_of_the_surface_on_which_the_object_is_inserted,
        obj=obj,
    )
finally:
    inserter.destroy()

# the image you made 😁
resulting_image = result.im
```

# How to cite

TBD