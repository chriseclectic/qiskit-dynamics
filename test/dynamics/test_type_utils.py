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
# pylint: disable=invalid-name

"""Tests for type_utils.py."""

from collections.abc import Iterable
import numpy as np
from scipy.sparse.csr import csr_matrix

from qiskit.quantum_info.operators.operator import Operator
from qiskit_dynamics.array import Array
from qiskit_dynamics.type_utils import (
    convert_state,
    type_spec_from_instance,
    StateTypeConverter,
    vec_dissipator,
    vec_commutator,
    to_array,
    to_csr,
    to_BCOO,
    to_numeric_matrix_type,
)

from .common import QiskitDynamicsTestCase, TestJaxBase, TestQutipBase

try:
    from jax.experimental import sparse as jsparse
except ImportError:
    pass


class TestStateTypeConverter(QiskitDynamicsTestCase):
    """StateTypeConverter tests."""

    def test_convert_state_order_C(self):
        """Test convert_state with order parameter 'C'."""

        type_spec = {"type": "array", "shape": (4,)}
        y = Array([[1, 2], [3, 4]])
        expected = Array([1, 2, 3, 4])

        self.assertAllClose(convert_state(y, type_spec, order="C"), expected)

        type_spec = {"type": "array"}
        y = [[1, 2], [3, 4]]
        expected = Array([[1, 2], [3, 4]])

        self.assertAllClose(convert_state(y, type_spec, order="C"), expected)

    def test_convert_state_order_F(self):
        """Test convert_state with order 'F'.

        Note the order='F' is the default so no need to specify
        """

        type_spec = {"type": "array", "shape": (4,)}
        y = Array([[1, 2], [3, 4]])
        expected = Array([1, 3, 2, 4])

        self.assertAllClose(convert_state(y, type_spec), expected)

        type_spec = {"type": "array"}
        y = [[1, 2], [3, 4]]
        expected = Array([[1, 2], [3, 4]])

        self.assertAllClose(convert_state(y, type_spec), expected)

    def test_type_spec_from_instance(self):
        """Test type_spec_from_instance"""

        y = Array([1, 2, 3, 4])
        type_spec = type_spec_from_instance(y)

        self.assertEqual(type_spec, {"type": "array", "shape": (4,)})

        y = Array([[1, 2], [3, 4], [5, 6]])
        type_spec = type_spec_from_instance(y)

        self.assertEqual(type_spec, {"type": "array", "shape": (3, 2)})

    def test_converter_inner_outer_order_C(self):
        """Test standard constructor of StateTypeConverter along with
        basic state conversion functions with array order 'C'.
        """

        inner_spec = {"type": "array", "shape": (4,)}
        outer_spec = {"type": "array", "shape": (2, 2)}
        converter = StateTypeConverter(inner_spec, outer_spec, order="C")

        y_in = Array([1, 2, 3, 4])
        y_out = Array([[1, 2], [3, 4]])

        convert_out = converter.inner_to_outer(y_in)
        convert_in = converter.outer_to_inner(y_out)

        self.assertAllClose(convert_out, y_out)
        self.assertAllClose(convert_in, y_in)

    def test_converter_inner_outer_order_F(self):
        """Test standard constructor of StateTypeConverter along with
        basic state conversion functions with array order 'C'.
        """

        inner_spec = {"type": "array", "shape": (4,)}
        outer_spec = {"type": "array", "shape": (2, 2)}
        converter = StateTypeConverter(inner_spec, outer_spec, order="F")

        y_in = Array([1, 3, 2, 4])
        y_out = Array([[1, 2], [3, 4]])

        convert_out = converter.inner_to_outer(y_in)
        convert_in = converter.outer_to_inner(y_out)

        self.assertAllClose(convert_out, y_out)
        self.assertAllClose(convert_in, y_in)

    def test_from_instances(self):
        """Test from_instances constructor"""

        inner_y = Array([1, 2, 3, 4])
        outer_y = Array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_instances(inner_y, outer_y)

        self.assertEqual(converter.inner_type_spec, {"type": "array", "shape": (4,)})
        self.assertEqual(converter.outer_type_spec, {"type": "array", "shape": (2, 2)})

        converter = StateTypeConverter.from_instances(inner_y)

        self.assertEqual(converter.inner_type_spec, {"type": "array", "shape": (4,)})
        self.assertEqual(converter.outer_type_spec, {"type": "array", "shape": (4,)})

    def test_from_outer_instance_inner_type_spec(self):
        """Test from_outer_instance_inner_type_spec constructor"""

        # test case for inner type spec with 1d array
        inner_type_spec = {"type": "array", "ndim": 1}
        outer_y = Array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_outer_instance_inner_type_spec(outer_y, inner_type_spec)

        self.assertEqual(converter.inner_type_spec, {"type": "array", "shape": (4,)})
        self.assertEqual(converter.outer_type_spec, {"type": "array", "shape": (2, 2)})

        # inner type spec is a generic array
        inner_type_spec = {"type": "array"}
        outer_y = Array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_outer_instance_inner_type_spec(outer_y, inner_type_spec)

        self.assertEqual(converter.inner_type_spec, {"type": "array", "shape": (2, 2)})
        self.assertEqual(converter.outer_type_spec, {"type": "array", "shape": (2, 2)})

    def test_transform_rhs_funcs_order_C(self):
        """Test rhs function conversion for order C array reshaping."""

        inner_spec = {"type": "array", "shape": (4,)}
        outer_spec = {"type": "array", "shape": (2, 2)}
        converter = StateTypeConverter(inner_spec, outer_spec, order="C")

        X = Array([[0.0, 1.0], [1.0, 0.0]])

        # do matrix multiplication (a truly '2d' operation)
        def rhs(t, y):
            return t * (y @ y)

        # pylint: disable=unused-argument
        def generator(t):
            return X

        new_rhs = converter.rhs_outer_to_inner(rhs)

        test_t = np.pi
        y_2d = Array([[1, 2], [3, 4]])
        y_1d = y_2d.flatten()

        expected_output = rhs(test_t, y_2d).flatten()
        output = new_rhs(test_t, y_1d)

        self.assertAllClose(output, expected_output)

        new_generator = converter.generator_outer_to_inner(generator)

        # verify generator vectorization
        expected_output = np.kron(X, Array(np.eye(2)))
        output = new_generator(test_t)

        self.assertAllClose(output, expected_output)

    def test_transform_rhs_funcs_order_F(self):
        """Test rhs function conversion"""

        inner_spec = {"type": "array", "shape": (4,)}
        outer_spec = {"type": "array", "shape": (2, 2)}
        converter = StateTypeConverter(inner_spec, outer_spec)

        X = Array([[0.0, 1.0], [1.0, 0.0]])

        # do matrix multiplication (a truly '2d' operation)
        def rhs(t, y):
            return t * (y @ y)

        # pylint: disable=unused-argument
        def generator(t):
            return X

        new_rhs = converter.rhs_outer_to_inner(rhs)

        test_t = np.pi
        y_2d = Array([[1, 2], [3, 4]])
        y_1d = y_2d.flatten(order="F")

        expected_output = rhs(test_t, y_2d).flatten(order="F")
        output = new_rhs(test_t, y_1d)

        self.assertAllClose(output, expected_output)

        new_generator = converter.generator_outer_to_inner(generator)

        # verify generator vectorization
        expected_output = np.kron(Array(np.eye(2)), X)
        output = new_generator(test_t)

        self.assertAllClose(output, expected_output)


class TestStateTypeConverterJax(TestStateTypeConverter, TestJaxBase):
    """JAX version of TestStateTypeConverter."""


class Testvec_commutator_dissipator(QiskitDynamicsTestCase):
    """Tests for vec_commutator and vec_dissipator."""

    def test_sparse_commutator_dissipator(self):
        """Tests that vec_commutator and vec_dissipator gives
        identical results, whether the array passed is a (k,n,n)
        Array or a (k,) Array of (n,n) sparse matrices."""
        np.random.seed(21301239)
        r = lambda *args: np.random.uniform(-1, 1, args)

        spm = csr_matrix(r(8, 8))
        self.assertAllClose(vec_commutator(spm).toarray(), vec_commutator(spm.toarray()))
        multi_matrix = r(3, 8, 8)
        den_commutator = vec_commutator(multi_matrix)
        sps_commutator = vec_commutator([csr_matrix(mat) for mat in multi_matrix])
        self.assertTrue(
            np.all(
                [
                    np.allclose(den_com, sps_com.toarray())
                    for den_com, sps_com in zip(den_commutator, sps_commutator)
                ]
            )
        )

        den_dissipator = vec_dissipator(multi_matrix)
        sps_dissipator = vec_dissipator([csr_matrix(mat) for mat in multi_matrix])
        self.assertTrue(
            np.all(
                [
                    np.allclose(den_dis, sps_dis.toarray())
                    for den_dis, sps_dis in zip(den_dissipator, sps_dissipator)
                ]
            )
        )


class Test_to_array(QiskitDynamicsTestCase):
    """Tests for to_array."""

    def test_None_to_None(self):
        """Test that None input returns None."""
        self.assertTrue(to_array(None) is None)

    def test_to_array_Operator(self):
        """Tests for to_array with a single operator"""
        op = Operator.from_label("X")
        self.assertAllClose(to_array(op), Array([[0, 1], [1, 0]]))

    def test_to_array_nparray(self):
        """Tests for to_array with a single numpy array"""
        ndarray = np.array([[0, 1], [1, 0]])
        self.assertAllClose(to_array(ndarray), ndarray)

    def test_to_array_Array(self):
        """Tests for to_array from a qiskit Array"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        arr_in = Array(list_of_ops)
        self.assertAllClose(to_array(arr_in), arr_in)

    def test_to_array_sparse_matrix(self):
        """Tests for to_array with a single sparse matrix"""
        op = Operator.from_label("X")
        spm = csr_matrix(op)
        ar = Array(op)
        self.assertAllClose(to_array(spm), ar)

    def test_to_array_Operator_list(self):
        """Tests for to_array with a list of operators"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        op_arr = [Operator.from_label(s) for s in "XYZ"]
        normal_array = Array(np.array(list_of_ops))
        self.assertAllClose(to_array(op_arr), normal_array)

    def test_to_array_nparray_list(self):
        """Tests for to_array with a list of numpy arrays"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        ndarray_list = [np.array(op) for op in list_of_ops]
        list_of_arrays = [Array(op) for op in list_of_ops]
        self.assertAllClose(to_array(ndarray_list), list_of_arrays)

    def test_to_array_list_of_arrays(self):
        """Tests for to_array with a list of numpy arrays"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        ndarray_list = [np.array(op) for op in list_of_ops]
        list_of_arrays = [Array(op) for op in list_of_ops]
        big_array = Array(list_of_arrays)
        self.assertAllClose(to_array(ndarray_list), big_array)

    def test_to_array_sparse_matrix_list(self):
        """Tests for to_array with a list of sparse matrices"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        list_of_arrays = [Array(op) for op in list_of_ops]
        sparse_matrices = [csr_matrix(op) for op in list_of_ops]
        self.assertAllClose(to_array(sparse_matrices), list_of_arrays)

    def test_to_array_types(self):
        """Type conversion tests for to_array"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        numpy_ops = np.array(list_of_ops)
        normal_array = Array(np.array(list_of_ops))
        op_arr = [Operator.from_label(s) for s in "XYZ"]
        single_op = op_arr[0]
        list_of_arrays = [Array(op) for op in list_of_ops]
        self.assertIsInstance(to_array(numpy_ops), np.ndarray)
        self.assertIsInstance(to_array(normal_array), Array)
        self.assertIsInstance(to_array(op_arr), np.ndarray)
        self.assertIsInstance(to_array(single_op), np.ndarray)
        self.assertIsInstance(to_array(list_of_arrays), np.ndarray)


class Test_to_array_Jax(Test_to_array, TestJaxBase):
    """Jax version of Test_to_array tests."""

    def test_to_array_types(self):
        """Type conversion tests for to_array with jax backend"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        numpy_ops = np.array(list_of_ops)
        normal_array = Array(np.array(list_of_ops))
        op_arr = [Operator.from_label(s) for s in "XYZ"]
        single_op = op_arr[0]
        list_of_arrays = [Array(op) for op in list_of_ops]
        self.assertIsInstance(to_array(numpy_ops), Array)
        self.assertIsInstance(to_array(normal_array), Array)
        self.assertIsInstance(to_array(op_arr), Array)
        self.assertIsInstance(to_array(single_op), Array)
        self.assertIsInstance(to_array(list_of_arrays), Array)

    def test_to_array_BCOO(self):
        """Convert BCOO type to array."""

        bcoo = jsparse.BCOO.fromdense(np.array([[0.0, 1.0], [1.0, 0.0]]))
        out = to_array(bcoo)
        self.assertTrue(isinstance(out, Array))
        self.assertAllClose(out, np.array([[0.0, 1.0], [1.0, 0.0]]))


class Test_to_csr(QiskitDynamicsTestCase):
    """Tests for to_csr."""

    def test_None_to_None(self):
        """Test that None input returns None."""
        self.assertTrue(to_csr(None) is None)

    def test_to_csr_Operator(self):
        """Tests for to_csr with a single operator"""
        op = Operator.from_label("X")
        self.assertAllCloseSparse(to_csr(op), csr_matrix([[0, 1], [1, 0]]))

    def test_to_csr_nparray(self):
        """Tests for to_csr with a single numpy array"""
        nparray = np.array([[0, 1], [1, 0]])
        self.assertAllCloseSparse(to_csr(nparray), csr_matrix(nparray))

    def test_to_csr_array(self):
        """Tests for to_csr with a single sparse matrix"""
        op = Operator.from_label("X")
        spm = csr_matrix(op)
        ar = Array(op)
        self.assertAllCloseSparse(to_csr(ar), spm)

    def test_to_csr_sparse_matrix(self):
        """Tests for to_csr with a single sparse matrix"""
        op = Operator.from_label("X")
        spm = csr_matrix(op)
        self.assertAllCloseSparse(to_csr(spm), spm)

    def test_to_csr_Operator_list(self):
        """Tests for to_csr with a list of operators"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        op_arr = [Operator.from_label(s) for s in "XYZ"]
        sparse_matrices = [csr_matrix(op) for op in list_of_ops]
        self.assertAllCloseSparse(to_csr(op_arr), sparse_matrices)

    def test_to_csr_nparray_list(self):
        """Tests for to_csr with a list of numpy arrays"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        nparray_list = [np.array(op) for op in list_of_ops]
        sparse_matrices = [csr_matrix(op) for op in list_of_ops]
        self.assertAllCloseSparse(to_csr(nparray_list), sparse_matrices)

    def test_to_csr_sparse_matrix_list(self):
        """Tests for to_csr with a list of sparse matrices"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        sparse_matrices = [csr_matrix(op) for op in list_of_ops]
        self.assertAllCloseSparse(to_csr(sparse_matrices), sparse_matrices)

    def test_to_csr_types(self):
        """Type conversion tests for to_csr"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        numpy_ops = np.array(list_of_ops)
        normal_array = Array(np.array(list_of_ops))
        op_arr = [Operator.from_label(s) for s in "XYZ"]
        single_op = op_arr[0]
        list_of_arrays = [Array(op) for op in list_of_ops]
        single_array = list_of_arrays[0]
        sparse_matrices = [csr_matrix(op) for op in list_of_ops]
        assert isinstance(to_csr(normal_array)[0], csr_matrix)
        assert isinstance(to_csr(numpy_ops)[0], csr_matrix)
        assert isinstance(to_csr(op_arr)[0], csr_matrix)
        assert isinstance(to_csr(single_op), csr_matrix)
        assert isinstance(to_csr(single_array), csr_matrix)
        assert isinstance(to_csr(list_of_arrays[0]), csr_matrix)
        assert isinstance(to_csr(sparse_matrices), Iterable)
        assert isinstance(to_csr(sparse_matrices)[0], csr_matrix)


class Testto_BCOO(QiskitDynamicsTestCase, TestJaxBase):
    """Test the to_BCOO function."""

    def test_None_to_None(self):
        """Test that None input returns None."""
        self.assertTrue(to_BCOO(None) is None)

    def test_to_BCOO_Operator(self):
        """Tests for to_BCOO with a single operator"""
        op = Operator.from_label("X")
        bcoo_op = to_BCOO(op)
        self.assertAllClose(Array(bcoo_op.todense()), Array([[0, 1], [1, 0]]))
        self.assertTrue(type(bcoo_op).__name__ == "BCOO")

    def test_to_BCOO_nparray(self):
        """Tests for to_BCOO with a single numpy array"""
        ndarray = np.array([[0, 1], [1, 0]])
        bcoo = to_BCOO(ndarray)
        self.assertTrue(type(bcoo).__name__ == "BCOO")
        self.assertAllClose(to_array(ndarray), ndarray)

    def test_to_array_Array(self):
        """Tests for to_BCOO from a qiskit Array"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        bcoo = to_BCOO(list_of_ops)
        self.assertTrue(type(bcoo).__name__ == "BCOO")
        self.assertAllClose(bcoo.todense(), Array(list_of_ops))

    def test_to_BCOO_Operator_list(self):
        """Tests for to_BCOO with a list of operators"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        op_arr = [Operator.from_label(s) for s in "XYZ"]
        bcoo = to_BCOO(op_arr)
        self.assertTrue(type(bcoo).__name__ == "BCOO")
        self.assertAllClose(bcoo.todense(), Array(list_of_ops))

    def test_to_BCOO_nparray_list(self):
        """Tests for to_BCOO with a list of numpy arrays"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        ndarray_list = [np.array(op) for op in list_of_ops]
        bcoo = to_BCOO(ndarray_list)
        self.assertTrue(type(bcoo).__name__ == "BCOO")
        self.assertAllClose(to_array(ndarray_list), bcoo.todense())

    def test_to_BCOO_list_of_arrays(self):
        """Tests for to_BCOO with a list of numpy arrays"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        list_of_arrays = [Array(op) for op in list_of_ops]
        bcoo = to_BCOO(list_of_arrays)
        self.assertTrue(type(bcoo).__name__ == "BCOO")
        self.assertAllClose(to_array(list_of_arrays), bcoo.todense())

    def test_to_BCOO_sparse_matrix(self):
        """Tests for to_BCOO with a single sparse matrix"""
        op = Operator.from_label("X")
        spm = csr_matrix(op)
        ar = Array(op)
        bcoo = to_BCOO(spm)
        self.assertTrue(type(bcoo).__name__ == "BCOO")
        self.assertAllClose(to_array(bcoo), ar)

    def test_to_BCOO_sparse_matrix_list(self):
        """Tests for to_BCOO with a list of sparse matrices"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        list_of_arrays = [Array(op) for op in list_of_ops]
        sparse_matrices = [csr_matrix(op) for op in list_of_ops]
        bcoo = to_BCOO(sparse_matrices)
        self.assertTrue(type(bcoo).__name__ == "BCOO")
        self.assertAllClose(to_array(bcoo), list_of_arrays)


class Test_to_numeric_matrix_type(QiskitDynamicsTestCase):
    """Test to_numeric_matrix_type"""

    def test_to_numeric_matrix_type(self):
        """Tests for to_numeric_matrix_type"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        normal_array = Array(np.array(list_of_ops))
        list_of_arrays = [Array(op) for op in list_of_ops]
        op_arr = [Operator.from_label(s) for s in "XYZ"]
        sparse_matrices = [csr_matrix(op) for op in list_of_ops]
        self.assertAllClose(to_numeric_matrix_type(list_of_ops), normal_array)
        self.assertAllClose(to_numeric_matrix_type(list_of_arrays), normal_array)
        self.assertAllClose(to_numeric_matrix_type(op_arr), list_of_arrays)
        for i in range(3):
            self.assertAllCloseSparse(
                to_numeric_matrix_type(sparse_matrices)[i], sparse_matrices[i]
            )


class Test_to_numeric_matrix_type_Jax(QiskitDynamicsTestCase, TestJaxBase):
    """Test to_numeric_matrix_type"""

    def test_to_numeric_matrix_type(self):
        """Tests for to_numeric_matrix_type"""
        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        bcoo = jsparse.BCOO.fromdense(to_array(list_of_ops))
        bcoo2 = to_numeric_matrix_type(bcoo)
        self.assertTrue(type(bcoo2).__name__ == "BCOO")
        self.assertAllClose(bcoo.todense(), bcoo2.todense())


class TestTypeUtilsQutip(QiskitDynamicsTestCase, TestQutipBase):
    """Perform type conversion testing for qutip qobj inputs"""

    def test_qutip_conversion(self):
        """Test qutip type conversion to numeric matrices generally, csr, and array"""
        from qutip import Qobj

        list_of_ops = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
        normal_array = Array(np.array(list_of_ops))
        sparse_matrices = [csr_matrix(op) for op in list_of_ops]
        qutip_qobj = [Qobj(op) for op in list_of_ops]
        self.assertAllCloseSparse(to_numeric_matrix_type(qutip_qobj)[0], sparse_matrices[0])
        self.assertAllCloseSparse(to_csr(qutip_qobj)[0], sparse_matrices[0])
        self.assertAllClose(to_array(qutip_qobj)[0], normal_array[0])
