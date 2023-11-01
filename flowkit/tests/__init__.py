"""
FlowKit Test Suites
"""
import unittest

from .sample_tests import SampleTestCase
from .sample_export_tests import SampleExportTestCase
from .gatingml_tests import GatingMLTestCase
from .gating_strategy_prog_gate_tests import GatingTestCase
from .export_gml_tests import ExportGMLTestCase
from .gating_strategy_tests import GatingStrategyTestCase
from .gating_strategy_reused_gates_tests import GatingStrategyReusedGatesTestCase
from .gating_strategy_remove_gates_tests import GatingStrategyRemoveGatesTestCase
from .gating_strategy_custom_gates_tests import GatingStrategyCustomGatesTestCase
from .session_tests import SessionTestCase
from .session_export_tests import SessionExportTestCase
from .workspace_tests import WorkspaceTestCase
from .gating_results_tests import GatingResultsTestCase
from .matrix_tests import MatrixTestCase
from .transform_tests import TransformsTestCase
from .gate_tests import GateTestCase
from .string_repr_tests import StringReprTestCase

if __name__ == "__main__":
    unittest.main()
