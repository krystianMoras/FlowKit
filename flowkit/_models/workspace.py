"""
Workspace class
"""
import gc
import copy
import numpy as np
import pandas as pd
from .._conf import debug
from .._utils import wsp_utils, sample_utils, gating_utils
import warnings


class Workspace(object):
    """
    A Workspace represents an imported FlowJo workspace (.wsp file).

    :param wsp_file_path: FlowJo WSP file as a file name/path, file object, or file-like object
    :param fcs_samples: str or list. If given a string, it can be a directory path or a file path.
        If a directory, any .fcs files in the directory will be found. If a list, then it must
        be a list of file paths or a list of Sample instances. Lists of mixed types are not
        supported. Note that only FCS files matching the ones referenced in the .wsp file will
        be retained in the Workspace.
    :param ignore_missing_files: Controls behavior for missing FCS files. If True, gate data for
        missing FCS files (i.e. not in fcs_samples arg) will still be loaded. If False, warnings
        are issued for FCS files found in the WSP file that were not loaded in the Workspace and
        gate data for these missing files will not be retained. Default is False.
    """

    def __init__(self, wsp_file_path, fcs_samples=None, ignore_missing_files=False):
        # The sample LUT holds sample IDs (keys) only for loaded samples.
        # The values are the Sample instances
        self._sample_lut = {}

        # The sample data LUT holds sample IDs (keys) for all samples
        # that have WSP gating data whether the sample is loaded or not.
        # The value is a dict containing the following items:
        #   - 'keywords': dict of FCS metadata found in WSP file
        #   - 'compensation' dict of spill info found in WSP file
        #   - 'transforms': dict of channel transforms found in WSP file
        #   - 'custom_gate_ids': set of gate paths for custom gates
        #   - 'gating_strategy': GatingStrategy assembled from WSP file
        self._sample_data_lut = {}

        # The group LUT keys are the available sample group names.
        # The values are dicts with keys 'gates' & 'samples'.
        # 'gates' has the common group Gate instances (not a
        # full GatingStrategy though) and 'samples' is a list
        # of the sample IDs belonging to the group.
        self._group_lut = {}

        # For storing GatingResults for analyzed samples.
        # Keys are sample IDs. We could put these in the
        # sample_lut but sample_lut is meant to be created
        # and complete within this constructor. Plus, this
        # makes it easier to determine which samples have
        # been analyzed.
        self._results_lut = {}

        # load samples we were given, we'll cross-reference against wsp below
        tmp_sample_lut = {
            s.id: s for s in sample_utils.load_samples(fcs_samples)}
        self._sample_lut = {}

        wsp_data = wsp_utils.parse_wsp(wsp_file_path)

        # save group sample membership, we'll filter by loaded samples next
        group_lut = wsp_data['groups']

        # save sample data, including the GatingStrategy & Sample instance
        for sample_id, sample_dict in wsp_data['samples'].items():
            if sample_id in tmp_sample_lut:
                # retain sample and add to sample data
                self._sample_lut[sample_id] = tmp_sample_lut[sample_id]
                self._sample_data_lut[sample_id] = sample_dict
            else:
                # we have gating info for a sample that wasn't loaded
                if ignore_missing_files:
                    # we're instructed to ignore missing files, so we'll still
                    # save the gate info for retrieval purposes
                    self._sample_data_lut[sample_id] = sample_dict
                else:
                    # we won't ignore missing files, issue a warning
                    # and remove any references to the sample
                    msg = "WSP references %s, but sample was not loaded." % sample_id
                    warnings.warn(msg)

                    # search for this missing sample ID in group data & remove
                    for group_name, group_dict in group_lut.items():
                        if sample_id in group_dict['samples']:
                            group_dict['samples'].remove(sample_id)

        self._group_lut = group_lut

    def __repr__(self):
        sample_count = len(self._sample_lut)
        sample_group_count = len(self._group_lut)

        return (
            f'{self.__class__.__name__}('
            f'{sample_count} samples loaded, '
            f'{sample_group_count} sample groups)'
        )

    def summary(self):
        """
        Retrieve a summary of Workspace information, including a list of
        sample groups defined, along with the sample and gate counts
        for those sample groups.

        :return: Pandas DataFrame containing Workspace summary information
        """
        sg_list = []

        for group_name, group_dict in self._group_lut.items():
            group_sample_ids = group_dict['samples']
            sample_count = len(group_sample_ids)

            loaded_sample_count = 0
            for g_sample_id in group_sample_ids:
                if g_sample_id in self._sample_lut:
                    loaded_sample_count += 1

            if sample_count > 0:
                # There's at least one sample, so grab the first one
                # as the prototype for the gating strategy. A WSP
                # group isn't guaranteed to have template gates for
                # the complete gate tree, this is the best we can do.
                example_sample_id = group_dict['samples'][0]
                gs = self.get_gating_strategy(example_sample_id)

                gate_count = len(gs.get_gate_ids())
                gate_depth = gs.get_max_depth()
            else:
                gate_count = 0
                gate_depth = 0

            sg_info = {
                'group_name': group_name,
                'samples': sample_count,
                'loaded_samples': loaded_sample_count,
                'gates': gate_count,
                'max_gate_depth': gate_depth
            }

            sg_list.append(sg_info)

        df = pd.DataFrame(sg_list)
        df.set_index('group_name', inplace=True)

        return df

    def get_sample_ids(self, group_name=None, loaded_only=True):
        """
        Retrieve the list of Sample IDs that in the Workspace, optionally
        filtered by sample group and/or loaded status. Default is all loaded
        samples.

        :param group_name: Filter returned sample IDs by a sample group. If None, all sample IDs are returned
        :param loaded_only: Filter returned sample IDs for only loaded samples. If False, all the samples will
            be returned, including any missing sample IDs referenced in the workspace. Default is True for
            returning only loaded sample IDs.
        :return: list of Sample ID strings
        """
        if group_name is not None:
            # group LUT contains all group sample IDs incl. missing ones
            sample_ids = set(self._group_lut[group_name]['samples'])
        else:
            # No group name specified so give user all sample IDs
            # sample data LUT contains all sample IDs, incl. missing IDs
            # referenced in the wsp (if ignore_missing_files was True)
            sample_ids = set(self._sample_data_lut.keys())

        # check if only loaded samples were requested
        if loaded_only:
            # sample LUT contains all the loaded sample IDs
            loaded_sample_ids = set(self._sample_lut.keys())

            # cross-reference sample_ids with loaded_sample_ids
            sample_ids = sample_ids.intersection(loaded_sample_ids)

        return sorted(list(sample_ids))

    def get_sample(self, sample_id):
        """
        Retrieve a Sample instance from the Workspace.

        :param sample_id: a text string representing the sample
        :return: a Sample instance
        """
        return self._sample_lut[sample_id]

    def get_samples(self, group_name=None):
        """
        Retrieve list of Sample instances, optionally filtered by sample group.

        :param group_name: Filter returned samples by a sample group. If None, all samples are returned
        :return: list of Sample instances
        """
        # don't return samples that haven't been loaded
        sample_ids = self.get_sample_ids(group_name=group_name)

        samples = []
        for s_id in sample_ids:
            samples.append(self._sample_lut[s_id])

        return samples

    def get_sample_groups(self):
        """
        Retrieve the list of sample group names defined in the Workspace.

        :return: list of sample group ID strings
        """
        return list(self._group_lut.keys())

    def get_gate_ids(self, sample_id):
        """
        Retrieve the list of gate IDs defined for the specified sample.
        The gate ID is a 2-item tuple where the first item is a string
        representing the gate name and the second item is a tuple of
        the gate path.

        :param sample_id: a text string representing a Sample instance
        :return: list of gate ID tuples
        """
        gs = self._sample_data_lut[sample_id]['gating_strategy']
        return gs.get_gate_ids()

    def find_matching_gate_paths(self, sample_id, gate_name):
        """
        Find all gate paths in the gating strategy for the given Sample
        matching the given gate name.

        :param sample_id: a text string representing a Sample instance
        :param gate_name: text string of a gate name
        :return: list of gate paths (list of tuples)
        """
        gs = self._sample_data_lut[sample_id]['gating_strategy']
        return gs.find_matching_gate_paths(gate_name)

    def get_child_gate_ids(self, sample_id, gate_name, gate_path=None):
        """
        Retrieve list of child gate IDs for a sample given the parent
        gate name (and path if ambiguous) in the gating strategy.

        :param sample_id: a text string representing a Sample instance
        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate.gate_name is ambiguous
        :return: list of Gate IDs (tuple of gate name plus gate path). Returns an empty
            list if no child gates exist.
        :raises GateReferenceError: if gate ID is not found in gating strategy or if gate
            name is ambiguous
        """
        gs = self._sample_data_lut[sample_id]['gating_strategy']
        child_gate_ids = gs.get_child_gate_ids(gate_name, gate_path)

        return child_gate_ids

    def get_gate_hierarchy(self, sample_id, output='ascii', **kwargs):
        """
        Retrieve the hierarchy of gates in the sample's gating strategy. Output is available
        in several formats, including text, dictionary, or JSON. If output == 'json', extra
        keyword arguments are passed to json.dumps

        :param sample_id: a text string representing a Sample instance
        :param output: Determines format of hierarchy returned, either 'ascii',
            'dict', or 'JSON' (default is 'ascii')
        :return: gate hierarchy as a text string or a dictionary
        """
        gs = self._sample_data_lut[sample_id]['gating_strategy']
        return gs.get_gate_hierarchy(output, **kwargs)

    def get_gating_strategy(self, sample_id):
        """
        Retrieve a copy of the GatingStrategy for a specific sample. sample_id is required as
        each sample may have customized gates

        :param sample_id: a text string representing a Sample instance
        :return: a copy of the GatingStrategy instance
        """
        return copy.deepcopy(self._sample_data_lut[sample_id]['gating_strategy'])

    def get_comp_matrix(self, sample_id):
        """
        Retrieve the compensation matrix for a specific sample.

        :param sample_id: a text string representing a Sample instance
        :return: a copy of a Matrix instance
        """
        sample_dict = self._sample_data_lut[sample_id]

        if sample_dict['compensation'] is not None:
            comp_matrix = copy.deepcopy(sample_dict['compensation']['matrix'])
        else:
            comp_matrix = None

        return comp_matrix

    def get_transform(self, sample_id, transform_id):
        """
        Retrieve a single transform for a sample using the transform ID. Transform
        IDs in the Workspace class correspond to a channel label in the sample.

        :param sample_id: a text string representing a Sample instance
        :param transform_id: a text string representing a Transform instance
        :return:
        """
        sample_dict = self._sample_data_lut[sample_id]

        if sample_dict['transforms'] is not None:
            xform = copy.deepcopy(sample_dict['transforms'][transform_id])
        else:
            xform = None

        return xform

    def get_transforms(self, sample_id):
        """
        Retrieve the list of transformations for a specific sample.

        :param sample_id: a text string representing a Sample instance
        :return: a list of Transform instances
        """
        sample_dict = self._sample_data_lut[sample_id]

        if sample_dict['transforms'] is not None:
            xforms = copy.deepcopy(list(sample_dict['transforms'].values()))
        else:
            xforms = None

        return xforms

    def get_gate(self, sample_id, gate_name, gate_path=None):
        """
        Retrieve a gate instance for a sample by its gate ID.

        :param sample_id: a text string representing a Sample instance.
        :param gate_name: text string of a gate ID
        :param gate_path: tuple of gate IDs for unique set of gate ancestors. Required if gate_name is ambiguous
        :return: Subclass of a Gate object
        """
        gs = self._sample_data_lut[sample_id]['gating_strategy']
        return gs.get_gate(gate_name, gate_path=gate_path)

    def analyze_samples(self, group_name=None, sample_id=None, cache_events=False, use_mp=True, verbose=False):
        """
        Process gates for samples. Samples to analyze can be filtered by group name or sample ID.
        After running, results can be retrieved using the `get_gating_results`, `get_group_report`,
        and  `get_gate_membership`, methods.

        :param group_name: optional group name, if specified only samples in this group will be processed
        :param sample_id: optional sample ID, if specified only this sample will be processed (overrides group filter)
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
        if sample_id is not None:
            sample_ids = [sample_id]
        else:
            # If group name is specified, get_sample_ids will return the
            # group sample IDs. If not then it will return all sample IDs.
            sample_ids = self.get_sample_ids(group_name=group_name)

        if len(sample_ids) == 0:
            warnings.warn("No samples were found to analyze")
            return

        sample_data_to_run = []
        for s_id in sample_ids:
            if s_id not in self._sample_data_lut:
                # sample ID provided isn't present in Workspace
                # or was referenced but has no gate data.
                warnings.warn("Sample %s has no gate data" % s_id)
                continue

            sample = self._sample_lut[s_id]
            gating_strategy = self._sample_data_lut[s_id]['gating_strategy']

            sample_data_to_run.append(
                {
                    'gating_strategy': gating_strategy,
                    'sample': sample
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

        # save the results in results LUT
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
                "No results found for %s. Have you run `analyze_samples`?" % sample_id
            )
        return copy.deepcopy(gating_result)

    def get_analysis_report(self, group_name=None):
        """
        Retrieve the report for the analyzed samples as a pandas DataFrame.

        :param group_name: optional group name, if specified only results
            from samples in this group will be processed, otherwise results
            from all analyzed samples will be returned
        :return: pandas DataFrame
        """
        all_reports = []
        group_s_ids = self.get_sample_ids(group_name)

        for s_id in group_s_ids:
            try:
                result = self._results_lut[s_id]
            except KeyError:
                continue

            all_reports.append(result.report)

        return copy.deepcopy(pd.concat(all_reports))

    def _get_processed_events(self, sample_id):
        """
        Retrieve a pandas DataFrame containing processed events for specified sample.
        Compensation and transforms will be applied according to the WSP file.

        :param sample_id: a text string representing a Sample instance
        :return: pandas DataFrame containing the processed sample events
        """
        sample = self.get_sample(sample_id)
        comp_matrix = self.get_comp_matrix(sample_id)
        xforms = self.get_transforms(sample_id)

        xform_lut = {
            xform.id: xform for xform in xforms if not xform.id.startswith('Comp')}

        # default is 'raw' events
        event_source = 'raw'

        if comp_matrix is not None:
            sample.apply_compensation(comp_matrix)
            event_source = 'comp'
        if xforms is not None:
            sample.apply_transform(xform_lut)
            event_source = 'xform'

        events_df = sample.as_dataframe(source=event_source)

        return events_df

    def get_gate_membership(self, sample_id, gate_name, gate_path=None):
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
        gating_result = self.get_gating_results(sample_id)
        return gating_result.get_gate_membership(gate_name, gate_path=gate_path)

    def get_gate_events(self, sample_id, gate_name=None, gate_path=None):
        """
        Retrieve gated events for a specific gate & sample as a pandas DataFrame.
        Gated events are processed according to the sample's compensation &
        channel transforms.

        :param sample_id: a text string representing a Sample instance
        :param gate_name: text string of a gate ID. If None, all Sample events will be returned (i.e. un-gated)
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: a pandas DataFrames with the gated events, compensated & transformed according
            to the group's compensation matrix and transforms
        """
        df_events = self._get_processed_events(sample_id)

        if gate_name is not None:
            gate_idx = self.get_gate_membership(
                sample_id, gate_name, gate_path)
            df_events = df_events[gate_idx]

        # TODO: maybe make an optional kwarg to control column label format
        df_events.columns = [' '.join(col).strip()
                             for col in df_events.columns]

        df_events.insert(0, 'sample_id', sample_id)

        return df_events
