"""
The function of this module is to enable the manual object insertion editing. This means that this module provides a serialization model, an object that can do the insertion based on this model, a GUI-based editor to manipulate this model and the serialization functions.
"""

from ._serialized import (
    DatasetBasedInserter,
    InsertionSpec,
    load_insertion_specs,
    run_editor,
    save_insertion_specs,
    DatasetWithObjectsInserted,
)

__all__ = [
    "DatasetBasedInserter",
    "load_insertion_specs",
    "save_insertion_specs",
    "InsertionSpec",
    "run_editor",
    "DatasetWithObjectsInserted",
]
