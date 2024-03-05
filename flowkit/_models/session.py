"""
Session class
"""
import gc
import io
import copy
import numpy as np
import pandas as pd
from .._conf import debug
from .._models.gating_strategy import GatingStrategy
from ..exceptions import GateReferenceError, GateTreeError
from .._utils import xml_utils, wsp_utils, sample_utils, gating_utils
import warnings
import flowkit._models.gates as fk_gates


class Session(object):
    """
    The Session class enables the programmatic creation of a gating strategy or for importing
    GatingML compliant documents. A Session combines multiple Sample instances with a single
    GatingStrategy. The gates in the gating strategy can be customized per sample.

    :param fcs_samples: str or list. If given a string, it can be a directory path or a file path.
            If a directory, any .fcs files in the directory will be loaded. If a list, then it must
            be a list of file paths or a list of Sample instances. Lists of mixed types are not
            supported.
    """

    def __init__(self, gating_strategy=None, fcs_samples=None):
        self.sample_lut = {}
        self._results_lut = {}

        if isinstance(gating_strategy, GatingStrategy):
            gating_strategy = gating_strategy
        elif isinstance(gating_strategy, str) or isinstance(gating_strategy, io.IOBase):
            # assume a path to an XML file representing a GatingML document
            gating_strategy = xml_utils.parse_gating_xml(gating_strategy)
        elif gating_strategy is None:
            gating_strategy = GatingStrategy()
        else:
            raise ValueError(
                "'gating_strategy' must be a GatingStrategy instance, GatingML document path, or None"
            )

        self.gating_strategy = gating_strategy

        self.add_samples(fcs_samples)

    def __repr__(self):
        sample_count = len(self.sample_lut)

        return (
            f'{self.__class__.__name__}('
            f'{sample_count} samples)'
        )

    def add_samples(self, fcs_samples):
        """
        Adds FCS samples to the session.

        :param fcs_samples: str or list. If given a string, it can be a directory path or a file path.
            If a directory, any .fcs files in the directory will be loaded. If a list, then it must
            be a list of file paths or a list of Sample instances. Lists of mixed types are not
            supported.
        :return: None
        """
        new_samples = sample_utils.load_samples(fcs_samples)
        for s in new_samples:
            if s.id in self.sample_lut:
                warnings.warn(
                    "A sample with ID %s already exists...skipping" % s.id)
                continue
            self.sample_lut[s.id] = s

    def get_sample_ids(self):
        """
        Retrieve the list of Sample IDs in the Session.

        :return: list of Sample ID strings
        """
        return list(self.sample_lut.keys())

    def get_gate_ids(self):
        """
        Retrieve the list of gate IDs defined for the Session's gating
        strategy. The gate ID is a 2-item tuple where the first item
        is a string representing the gate name and the second item is
        a tuple of the gate path.

        :return: list of gate ID tuples
        """
        return self.gating_strategy.get_gate_ids()

    def add_gate(self, gate, gate_path, sample_id):
        """
        Add a Gate instance to the gating strategy. The gate ID and gate path
        must be unique in the gating strategy. Custom sample gates may be added
        by specifying an optional sample ID. Note, the gate & gate path must
        already exist prior to adding custom sample gates.

        :param gate: an instance of a Gate subclass
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors
        :param sample_id: text string for specifying given gate as a custom Sample gate
        :return: None
        """
        gate.additional_attributes["sample_id"] = sample_id
        self.gating_strategy.add_gate(copy.deepcopy(
            gate), gate_path=gate_path, sample_id=sample_id)

    def edit_gate(self, gate, sample_id):
        """
        Edit a Gate instance in the gating strategy. The gate ID and gate path
        must exist in the gating strategy. Custom sample gates may be edited
        by specifying an optional sample ID.

        :param gate: an instance of a Gate subclass
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors
        :param sample_id: text string for specifying given gate as a custom Sample gate
        :return: None
        """
        gate.additional_attributes["sample_id"] = sample_id
        self.gating_strategy.edit_gate(copy.deepcopy(
            gate), sample_id=sample_id)

    def remove_gate(self, gate_name, keep_children=False):
        """
        Remove a gate from the gate tree. Any descendant gates will also be removed
        unless keep_children=True. In all cases, if a BooleanGate exists that references
        the gate to remove, a GateTreeError will be thrown indicating the BooleanGate
        must be removed prior to removing the gate.

        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :param keep_children: Whether to keep child gates. If True, the child gates will be
            remapped to the removed gate's parent. Default is False, which will delete all
            descendant gates.
        :return: None
        """
        self.gating_strategy.remove_gate(
            gate_name, keep_children=keep_children)

    def add_transform(self, transform):
        """
        Add a Transform instance to use in the gating strategy.

        :param transform: an instance of a Transform subclass
        :return: None
        """
        self.gating_strategy.add_transform(copy.deepcopy(transform))

    def get_transforms(self):
        """
        Retrieve the list of Transform instances stored in the gating strategy.

        :return: list of Transform instances
        """

        return list(self.gating_strategy.transformations.values())

    def get_transform(self, transform_id):
        """
        Retrieve a Transform stored in the gating strategy by its ID.

        :param transform_id: a text string representing a Transform ID
        :return: an instance of a Transform subclass
        """
        return self.gating_strategy.get_transform(transform_id)

    def add_comp_matrix(self, matrix):
        """
        Add a Matrix instance to use in the gating strategy.

        :param matrix: an instance of the Matrix class
        :return: None
        """
        self.gating_strategy.add_comp_matrix(copy.deepcopy(matrix))

    def get_comp_matrices(self):
        """
        Retrieve the list of compensation Matrix instances stored in the gating strategy.

        :return: list of Matrix instances
        """
        return list(self.gating_strategy.comp_matrices.values())

    def get_comp_matrix(self, matrix_id):
        """
        Retrieve a compensation Matrix instance stored in the gating strategy by its ID.

        :param matrix_id: a text string representing a Matrix ID
        :return: a Matrix instance
        """
        return self.gating_strategy.get_comp_matrix(matrix_id)

    def find_matching_gate_paths(self, gate_name):
        """
        Find all gate paths in the gating strategy matching the given gate name.

        :param gate_name: text string of a gate name
        :return: list of gate paths (list of tuples)
        """
        return self.gating_strategy.find_matching_gate_paths(gate_name)

    def get_child_gate_ids(self, gate_name):
        """
        Retrieve list of child gate IDs given the parent gate name (and path if ambiguous)
        in the gating strategy.

        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate.gate_name is ambiguous
        :return: list of Gate IDs (tuple of gate name plus gate path). Returns an empty
            list if no child gates exist.
        :raises GateReferenceError: if gate ID is not found in gating strategy or if gate
            name is ambiguous
        """
        return self.gating_strategy.get_child_gate_ids(gate_name)

    def get_gate(self, gate_name, sample_id=None) -> fk_gates.BooleanGate | fk_gates.QuadrantGate | fk_gates.Quadrant | fk_gates.PolygonGate | fk_gates.RectangleGate | fk_gates.EllipsoidGate:
        """
        Retrieve a gate instance by its gate ID (and sample ID for custom sample gates).

        :param gate_name: text string of a gate ID
        :param gate_path: tuple of gate IDs for unique set of gate ancestors. Required if gate_name is ambiguous
        :param sample_id: a text string representing a Sample instance. If None, the template gate is returned.
        :return: Subclass of a Gate object
        """
        return self.gating_strategy.get_gate(gate_name, sample_id=sample_id)

    def get_sample_gates(self, sample_id):
        """
        Retrieve all gates for a sample in the gating strategy. This returns custom sample
        gates for the specified sample ID.

        :param sample_id: a text string representing a Sample instance
        :return: list of Gate subclass instances
        """
        gate_tuples = self.gating_strategy.get_gate_ids()

        sample_gates = []

        for gate_name, ancestors in gate_tuples:
            gate = self.gating_strategy.get_gate(
                gate_name, sample_id=sample_id)
            if gate is not None:
                sample_gates.append(gate)

        return sample_gates
    
    def get_graph_for_sample(self, sample_id):
        """
        Retrieve the graph of gates for a sample in the gating strategy. This returns custom sample
        gates for the specified sample ID.

        :param sample_id: a text string representing a Sample instance
        :return: list of Gate subclass instances
        """
        return self.gating_strategy.build_graph_for_sample(sample_id)

    def get_gate_hierarchy(self, output='ascii', **kwargs):
        """
        Retrieve the hierarchy of gates in the gating strategy. Output is available
        in several formats, including text, dictionary, or JSON. If output == 'json', extra
        keyword arguments are passed to json.dumps

        :param output: Determines format of hierarchy returned, either 'ascii',
            'dict', or 'JSON' (default is 'ascii')
        :return: gate hierarchy as a text string or a dictionary
        """
        return self.gating_strategy.get_gate_hierarchy(output, **kwargs)

    def export_gml(self, file_handle, sample_id=None):
        """
        Export a GatingML 2.0 file for the gating strategy. Specify the sample ID to use
        that sample's custom gates in the exported file, otherwise the template gates
        will be exported.

        :param file_handle: file handle for exporting data
        :param sample_id: an optional text string representing a Sample instance
        :return: None
        """
        xml_utils.export_gatingml(
            self.gating_strategy, file_handle, sample_id=sample_id)

    def export_wsp(self, file_handle, group_name):
        """
        Export a FlowJo 10 workspace file (.wsp) for the gating strategy.

        :param file_handle: file handle for exporting data
        :param group_name: a text string representing the sample group to add to the WSP file
        :return: None
        """
        samples = self.sample_lut.values()

        wsp_utils.export_flowjo_wsp(
            self.gating_strategy, group_name, samples, file_handle)

    def get_sample(self, sample_id):
        """
        Retrieve a Sample instance from the Session.

        :param sample_id: a text string representing the sample
        :return: Sample instance
        """
        return self.sample_lut[sample_id]

    def analyze_samples(self, sample_id=None, cache_events=False, use_mp=True, verbose=False):
        """
        Process gating strategy for samples. After running, results can be retrieved
        using the `get_gating_results`, `get_report`, and  `get_gate_membership`,
        methods.

        :param sample_id: optional sample ID, if specified only this sample will be processed
        :param cache_events: Whether to cache pre-processed events (compensated and transformed). This can
            be useful to speed up processing of gates that share the same pre-processing instructions for
            the same channel data, but can consume significantly more memory space. See the related
            clear_cache method for additional information. Default is False.
        :param use_mp: Controls whether multiprocessing is used to gate samples (default is True).
            Multiprocessing can fail for large workloads (lots of samples & gates) due to running out of
            memory. If encountering memory errors, set use_mp to False (processing will take longer,
            but will use significantly less memory).
        :param verbose: if True, print a line for every gate processed (default is False)
        :return: None
        """
        # Don't save just the DataFrame report, save the entire
        # GatingResults objects for each sample, since we'll need the gate
        # indices for each sample.
        samples = self.sample_lut.values()
        if len(samples) == 0:
            warnings.warn("No samples have been loaded in the Session")
            return

        if sample_id is not None:
            samples = [self.get_sample(sample_id)]

        sample_data_to_run = []
        for s in samples:
            sample_data_to_run.append(
                {
                    'gating_strategy': self.gating_strategy,
                    'sample': s
                }
            )

            # clear any existing results
            if sample_id in self._results_lut:
                del self._results_lut[sample_id]
                # gc.collect()

        results = gating_utils.gate_samples(
            sample_data_to_run,
            cache_events,
            verbose,
            use_mp=False if debug else use_mp
        )

        for r in results:
            self._results_lut[r.sample_id] = r

    def get_gating_results(self, sample_id):
        """
        Retrieve analyzed gating results gates for a sample.

        :param sample_id: a text string representing a Sample instance
        :return: GatingResults instance
        """
        try:
            gating_result = self._results_lut[sample_id]
        except KeyError:
            raise KeyError(
                "No results for %s. Have you run `analyze_samples`?" % sample_id
            )
        return copy.deepcopy(gating_result)

    def get_analysis_report(self):
        """
        Retrieve the report for the analyzed samples as a pandas DataFrame.

        :return: pandas DataFrame
        """
        all_reports = []

        for s_id, result in self._results_lut.items():
            all_reports.append(result.report)

        return copy.deepcopy(pd.concat(all_reports))

    def get_gate_membership(self, sample_id, gate_name):
        """
        Retrieve a boolean array indicating gate membership for the events in the
        specified sample. Note, the same gate ID may be found in multiple gate paths,
        i.e. the gate ID can be ambiguous. In this case, specify the full gate path
        to retrieve gate indices.

        :param sample_id: a text string representing a Sample instance
        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: NumPy boolean array (length of sample event count)
        """
        gating_result = self._results_lut[sample_id]
        return gating_result.get_gate_membership(gate_name)

    def get_gate_events(self, sample_id, gate_name=None, matrix=None, transform=None):
        """
        Retrieve a pandas DataFrame containing only the events within the specified gate.
        If an optional compensation matrix and/or a transform is provided, the returned
        event data will be compensated or transformed. If both a compensation matrix and
        a transform is provided the event data will be both compensated and transformed.

        :param sample_id: a text string representing a Sample instance
        :param gate_name: text string of a gate name. If None, all Sample events will be returned (i.e. un-gated)
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :param matrix: an instance of the Matrix class
        :param transform: an instance of a Transform subclass
        :return: pandas DataFrame containing only the events within the specified gate
        """
        # TODO: re-evaluate whether this method should be removed or modified...the
        #   ambiguous transforms per channel make this tricky to implement.
        sample = self.get_sample(sample_id)
        sample = copy.deepcopy(sample)

        # default is 'raw' events
        event_source = 'raw'

        if matrix is not None:
            sample.apply_compensation(matrix)
            event_source = 'comp'
        if transform is not None:
            sample.apply_transform(transform)
            event_source = 'xform'

        events_df = sample.as_dataframe(source=event_source)

        if gate_name is not None:
            gate_idx = self.get_gate_membership(
                sample_id, gate_name)
            events_df = events_df[gate_idx]

        return events_df


    def merge_strategy(self, other_strategy: GatingStrategy):

        # get the gates from the other strategy
        for gate_name, ancestors in other_strategy.get_gate_ids():
            # if gate_name
            print(gate_name, ancestors)
            gate = other_strategy.get_gate(gate_name)
            try:
                self.gating_strategy.get_gate(gate_name)
                self.gating_strategy.edit_gate(copy.deepcopy(gate), gate.additional_attributes.get('sample_id', None))
            except GateReferenceError:
                self.gating_strategy.add_gate(copy.deepcopy(gate), gate_path=ancestors, sample_id=gate.additional_attributes.get('sample_id', None))



