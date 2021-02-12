import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('../..'))

from flowkit import Session, Sample, Matrix, Dimension, gates, transforms
# noinspection PyProtectedMember
from flowkit._models.transforms._base_transform import Transform
# noinspection PyProtectedMember
from flowkit._models.gates._base_gate import Gate
from .gating_strategy_prog_gate_tests import data1_sample, poly1_gate, poly1_vertices, comp_matrix_01, asinh_xform1

fcs_file_paths = [
    "examples/100715.fcs",
    "examples/109567.fcs",
    "examples/113548.fcs"
]


class SessionTestCase(unittest.TestCase):
    """Tests for Session class"""
    def test_load_samples_from_list_of_paths(self):
        fks = Session(fcs_samples=fcs_file_paths)

        self.assertEqual(len(fks.sample_lut.keys()), 3)
        self.assertIsInstance(fks.get_sample('100715.fcs'), Sample)

        sample_ids = ["100715.fcs", "109567.fcs", "113548.fcs"]
        self.assertListEqual(fks.get_sample_ids(), sample_ids)

    def test_load_samples_from_list_of_samples(self):
        samples = [Sample(file_path) for file_path in fcs_file_paths]
        fks = Session(fcs_samples=samples)

        self.assertEqual(len(fks.sample_lut.keys()), 3)
        self.assertIsInstance(fks.get_sample('100715.fcs'), Sample)

    def test_get_comp_matrix(self):
        fks = Session(fcs_samples=data1_sample)
        fks.add_comp_matrix(comp_matrix_01)
        comp_mat = fks.get_comp_matrix('default', 'B07', 'MySpill')

        self.assertIsInstance(comp_mat, Matrix)

    def test_get_group_comp_matrices(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS_simple.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files"
        sample_grp = 'DEN'

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        comp_matrices = fks.get_group_comp_matrices(sample_grp)

        self.assertEqual(len(comp_matrices), 3)
        for cm in comp_matrices:
            self.assertIsInstance(cm, Matrix)

    def test_get_transform(self):
        fks = Session(fcs_samples=data1_sample)
        fks.add_transform(asinh_xform1)
        comp_mat = fks.get_transform('default', 'B07', 'AsinH_10000_4_1')

        self.assertIsInstance(comp_mat, transforms.AsinhTransform)

    def test_get_group_transforms(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS_simple.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files"
        sample_grp = 'DEN'

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        xforms = fks.get_group_transforms(sample_grp)

        self.assertEqual(len(xforms), 69)
        for cm in xforms:
            self.assertIsInstance(cm, Transform)

    def test_get_sample_gates(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS_simple.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        sample_gates = fks.get_sample_gates(sample_grp, sample_id)

        self.assertEqual(len(sample_gates), 4)
        for cm in sample_gates:
            self.assertIsInstance(cm, Gate)

    @staticmethod
    def test_add_poly1_gate():
        fks = Session(fcs_samples=data1_sample)
        fks.add_gate(poly1_gate)
        fks.analyze_samples()
        result = fks.get_gating_results('default', data1_sample.original_filename)

        res_path = 'examples/gate_ref/truth/Results_Polygon1.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon1'))

    @staticmethod
    def test_add_matrix_poly4_gate():
        fks = Session(fcs_samples=data1_sample)

        fks.add_comp_matrix(comp_matrix_01)

        dim1 = Dimension('PE', compensation_ref='MySpill')
        dim2 = Dimension('PerCP', compensation_ref='MySpill')
        dims = [dim1, dim2]

        poly_gate = gates.PolygonGate('Polygon4', None, dims, poly1_vertices)
        fks.add_gate(poly_gate)

        res_path = 'examples/gate_ref/truth/Results_Polygon4.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        fks.analyze_samples()
        result = fks.get_gating_results('default', data1_sample.original_filename)

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon4'))

    @staticmethod
    def test_add_transform_asinh_range1_gate():
        fks = Session(fcs_samples=data1_sample)
        fks.add_transform(asinh_xform1)

        dim1 = Dimension('FL1-H', 'uncompensated', 'AsinH_10000_4_1', range_min=0.37, range_max=0.63)
        dims = [dim1]

        rect_gate = gates.RectangleGate('ScaleRange1', None, dims)
        fks.add_gate(rect_gate)

        res_path = 'examples/gate_ref/truth/Results_ScaleRange1.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        fks.analyze_samples()
        result = fks.get_gating_results('default', data1_sample.original_filename)

        np.testing.assert_array_equal(truth, result.get_gate_indices('ScaleRange1'))

    def test_calculate_comp_from_beads(self):
        bead_dir = "examples/4_color_beads"
        fks = Session()
        comp = fks.calculate_compensation_from_beads(bead_dir)

        self.assertIsInstance(comp, Matrix)
