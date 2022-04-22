# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Dispatch class"""

import functools
from types import FunctionType
from typing import Optional, Callable

from arraylias import numpy_alias, LibraryError

ALIAS = numpy_alias()


def asarray(
    array: any,
    dtype: Optional[any] = None,
    order: Optional[str] = None,
    backend: Optional[str] = None,
) -> any:
    """Convert input array to an array on the specified backend.

    This functions like `numpy.asarray` but optionally supports
    casting to other registered array backends.

    Args:
        array: An array_like input.
        dtype: Optional. The dtype of the returned array. This value
                must be supported by the specified array backend.
        order: Optional. The array order. This value must be supported
                by the specified array backend.
        backend: A registered array backend name. If None the
                    default array backend will be used.

    Returns:
        array: an array object of the form specified by the backend
                kwarg.
    """
    return ALIAS("asarray", like=backend)(array, dtype=dtype, order=order)


def requires_backend(backend: str) -> Callable:
    """Return a function and class decorator for checking a backend is available.

    If the the required backend is not in the list of :meth:`.Array.available_backends`
    any decorated function or method will raise an exception when called, and
    any decorated class will raise an exeption when its ``__init__`` is called.

    Args:
        backend: the backend name required by class or function.

    Returns:
        Callable: A decorator that may be used to specify that a function, class,
                  or class method requires a specific backend to be installed.
    """

    def decorator(obj):
        """Specify that the decorated object requires a specifc Array backend."""

        def check_backend(descriptor):
            if backend not in ALIAS.registered_libs():
                raise LibraryError(
                    f"Array backend '{backend}' required by {descriptor} "
                    "is not installed. Please install the optional "
                    f"library '{backend}'."
                )

        # Decorate a function or method
        if isinstance(obj, FunctionType):

            @functools.wraps(obj)
            def decorated_func(*args, **kwargs):
                check_backend(f"function {obj}")
                return obj(*args, **kwargs)

            return decorated_func

        # Decorate a class
        elif isinstance(obj, type):

            obj_init = obj.__init__

            @functools.wraps(obj_init)
            def decorated_init(self, *args, **kwargs):
                check_backend(f"class {obj}")
                obj_init(self, *args, **kwargs)

            obj.__init__ = decorated_init
            return obj

        else:
            raise Exception(f"Cannot decorate object {obj} that is not a class or function.")

    return decorator
