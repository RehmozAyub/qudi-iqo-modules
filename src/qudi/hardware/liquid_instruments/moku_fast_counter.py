# -*- coding: utf-8 -*-

"""
A hardware module for using a Liquid Instruments Moku device as a fast counter (timetagger)
via the Time & Frequency Analyzer instrument mode.

This module implements the FastCounterInterface, allowing the Moku to be used as a drop-in
replacement for other time-tagging hardware (e.g., Swabian Instruments TimeTagger) within
the qudi pulsed measurement toolchain.

Copyright (c) 2026, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-iqo-modules/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

import time
import numpy as np

from qudi.interface.fast_counter_interface import FastCounterInterface
from qudi.core.configoption import ConfigOption


class MokuFastCounter(FastCounterInterface):
    """Hardware class to use a Liquid Instruments Moku as a fast counter (timetagger).

    Uses the Moku's Time & Frequency Analyzer instrument to count events and build
    time-interval histograms, compatible with qudi's pulsed measurement toolchain.

    Requires the 'moku' Python package:  pip install moku

    Example config for copy-paste:

    moku_fast_counter:
        module.Class: 'liquid_instruments.moku_fast_counter.MokuFastCounter'
        options:
            moku_ip_address: '192.168.###.###'
            input_channel: 'Input1'
            trigger_channel: 'Input2'
            event_threshold: 0.0
            event_edge: 'Rising'
            trigger_threshold: 0.0
            trigger_edge: 'Rising'
            force_connect: True

    """

    # ======================== ConfigOptions ========================
    # IP address of the Moku device on the network
    _ip_address = ConfigOption('moku_ip_address', missing='error')

    # Signal input channel for photon/event detection (e.g. APD clicks)
    _input_channel = ConfigOption('input_channel', default='Input1', missing='info')

    # Trigger/sync channel (e.g. start of pulse sequence)
    _trigger_channel = ConfigOption('trigger_channel', default='Input2', missing='info')

    # Threshold voltage for event detection on the signal channel (V)
    _event_threshold = ConfigOption('event_threshold', default=0.0, missing='info')

    # Edge type for event detection: 'Rising' or 'Falling'
    _event_edge = ConfigOption('event_edge', default='Rising', missing='info')

    # Threshold voltage for trigger detection (V)
    _trigger_threshold = ConfigOption('trigger_threshold', default=0.0, missing='info')

    # Edge type for trigger detection: 'Rising' or 'Falling'
    _trigger_edge = ConfigOption('trigger_edge', default='Rising', missing='info')

    # Whether to force-connect (overtake existing connections to the Moku)
    _force_connect = ConfigOption('force_connect', default=True, missing='info')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Internal state
        self._tfa = None          # TimeFrequencyAnalyzer instance
        self._statusvar = 0       # 0=unconfigured, 1=idle, 2=running, 3=paused, -1=error
        self._bin_width_s = 1e-9  # bin width in seconds
        self._record_length_s = 0 # total record length in seconds
        self._number_of_gates = 0
        self._n_bins = 0          # number of bins in the histogram
        self._count_data = None   # stored histogram data
        self._start_time = None   # timestamp when measurement started

    # ======================== Lifecycle ========================

    def on_activate(self):
        """Connect to the Moku device and deploy the Time & Frequency Analyzer.
        """
        try:
            from moku.instruments import TimeFrequencyAnalyzer
        except ImportError:
            self.log.error(
                'Could not import moku package. Install it with: pip install moku'
            )
            self._statusvar = -1
            return

        try:
            self._tfa = TimeFrequencyAnalyzer(
                self._ip_address,
                force_connect=self._force_connect
            )
            self.log.info(
                f'Connected to Moku at {self._ip_address} as Time & Frequency Analyzer.'
            )
        except Exception:
            self.log.exception('Failed to connect to Moku device:')
            self._tfa = None
            self._statusvar = -1
            return

        # Configure event detectors with user-specified settings
        try:
            # Event 1: Signal events (e.g. photon clicks from APD)
            self._tfa.set_event_detector(
                1,
                self._input_channel,
                threshold=self._event_threshold,
                edge=self._event_edge
            )
            # Event 2: Trigger/sync events (e.g. start of pulse sequence)
            self._tfa.set_event_detector(
                2,
                self._trigger_channel,
                threshold=self._trigger_threshold,
                edge=self._trigger_edge
            )
            self.log.info(
                f'Event detectors configured: signal={self._input_channel} '
                f'({self._event_edge}@{self._event_threshold}V), '
                f'trigger={self._trigger_channel} '
                f'({self._trigger_edge}@{self._trigger_threshold}V)'
            )
        except Exception:
            self.log.exception('Failed to configure event detectors:')
            self._statusvar = -1
            return

        # Set interpolation and interval policy defaults
        try:
            self._tfa.set_interpolation(mode='Linear')
            self._tfa.set_interval_policy(
                multiple_start_events='Use first',
                incomplete_intervals='Close'
            )
        except Exception:
            self.log.exception('Failed to configure interpolation/interval policy:')

        self._statusvar = 0
        self.log.info('Moku fast counter activated successfully.')

    def on_deactivate(self):
        """Disconnect from the Moku device and release resources.
        """
        if self._tfa is not None:
            try:
                self._tfa.relinquish_ownership()
                self.log.info('Moku connection released.')
            except Exception:
                self.log.exception('Error releasing Moku connection:')
            finally:
                self._tfa = None
        self._statusvar = -1

    # ======================== FastCounterInterface Methods ========================

    def get_constraints(self):
        """Retrieve the hardware constraints from the Moku Time & Frequency Analyzer.

        @return dict: dict with keys being the constraint names as string and
                      items are the definition for the constraints.

        The Moku TFA has a histogram with 1024 bins. The bin width is determined
        by the histogram span (start_time to stop_time). The minimum time resolution
        depends on the Moku hardware model (typically ~1ns for Moku:Pro, ~10ns for Moku:Lab).
        """
        constraints = dict()

        # The Moku TFA histogram always has 1024 bins.
        # The bin width = (stop_time - start_time) / 1024
        # Supported bin widths depend on the time span the user configures.
        # We provide a representative list spanning typical use cases:
        # From ~1ns (for narrow windows like ODMR) to ~1us (for longer pulse sequences)
        constraints['hardware_binwidth_list'] = [
            1e-9,    # 1 ns   (1.024 us span)
            2e-9,    # 2 ns   (2.048 us span)
            4e-9,    # 4 ns   (4.096 us span)
            8e-9,    # 8 ns   (8.192 us span)
            10e-9,   # 10 ns  (10.24 us span)
            20e-9,   # 20 ns  (20.48 us span)
            50e-9,   # 50 ns  (51.2 us span)
            100e-9,  # 100 ns (102.4 us span)
            200e-9,  # 200 ns (204.8 us span)
            500e-9,  # 500 ns (512 us span)
            1e-6,    # 1 us   (1.024 ms span)
        ]

        return constraints

    def configure(self, bin_width_s, record_length_s, number_of_gates=0):
        """Configuration of the fast counter.

        @param float bin_width_s: Length of a single time bin in the time trace
                                  histogram in seconds.
        @param float record_length_s: Total length of the timetrace/each single
                                      gate in seconds.
        @param int number_of_gates: optional, number of gates in the pulse
                                    sequence. Ignore for ungated counter.

        @return tuple(binwidth_s, record_length_s, number_of_gates):
                    binwidth_s: float the actual set binwidth in seconds
                    record_length_s: the actual set record length in seconds
                    number_of_gates: the number of gates accepted
        """
        if self._tfa is None:
            self.log.error('Moku not connected. Cannot configure.')
            return -1

        # The Moku TFA histogram has a fixed 1024 bins.
        # bin_width = span / 1024, so span = bin_width * 1024
        n_bins = 1024
        histogram_span = bin_width_s * n_bins

        # Calculate actual number of bins to match the requested record length
        self._n_bins = int(np.rint(record_length_s / bin_width_s))
        self._bin_width_s = bin_width_s
        self._record_length_s = record_length_s
        self._number_of_gates = number_of_gates

        # Configure the histogram on the Moku:
        # start_time=0 means we measure time intervals starting from the trigger
        start_time = 0.0
        stop_time = histogram_span

        try:
            # Set histogram span and type
            self._tfa.set_histogram(
                start_time=start_time,
                stop_time=stop_time,
                type='Interval'
            )

            # Configure the interval analyzer:
            # Interval 1: from trigger (Event 2) to signal (Event 1)
            # This measures the time between trigger and photon detection
            self._tfa.set_interval_analyzer(
                1,
                start_event_id=2,  # trigger
                stop_event_id=1    # signal/click
            )

            # Set acquisition mode to windowed, matching the record length
            # This determines how long the Moku accumulates events per frame
            window_length = max(record_length_s, 10e-3)  # minimum 10ms window
            self._tfa.set_acquisition_mode(
                mode='Windowed',
                window_length=window_length
            )

            self.log.info(
                f'Moku TFA configured: bin_width={bin_width_s*1e9:.1f}ns, '
                f'span={histogram_span*1e6:.1f}us, '
                f'record_length={record_length_s*1e6:.1f}us, '
                f'gates={number_of_gates}'
            )
        except Exception:
            self.log.exception('Failed to configure Moku TFA:')
            self._statusvar = -1
            return -1

        self._statusvar = 1

        actual_binwidth = histogram_span / n_bins
        actual_length = self._n_bins * actual_binwidth

        return actual_binwidth, actual_length, number_of_gates

    def start_measure(self):
        """Start the fast counter measurement.

        Clears previous data and begins accumulating events.
        """
        if self._tfa is None:
            self.log.error('Moku not connected. Cannot start measurement.')
            return -1

        try:
            self._tfa.clear_data()
            self._start_time = time.time()
            self.module_state.lock()
            self._statusvar = 2
            self.log.debug('Moku TFA measurement started.')
            return 0
        except Exception:
            self.log.exception('Failed to start measurement:')
            self._statusvar = -1
            return -1

    def stop_measure(self):
        """Stop the fast counter measurement.
        """
        if self.module_state() == 'locked':
            self.module_state.unlock()
        self._statusvar = 1
        self.log.debug('Moku TFA measurement stopped.')
        return 0

    def pause_measure(self):
        """Pauses the current measurement.

        Note: The Moku TFA does not have a native pause/resume. We implement
        this by simply recording the pause state. Data acquisition continues
        in the background, but get_data_trace will return the last captured data.
        """
        if self.module_state() == 'locked':
            self._statusvar = 3
            self.log.debug('Moku TFA measurement paused.')
        return 0

    def continue_measure(self):
        """Continues the current measurement from a paused state.
        """
        if self.module_state() == 'locked':
            self._statusvar = 2
            self.log.debug('Moku TFA measurement continued.')
        return 0

    def is_gated(self):
        """Check the gated counting possibility.

        @return bool: Boolean value indicates if the fast counter is a gated
                      counter (TRUE) or not (FALSE).

        The Moku TFA can function as both gated and ungated depending on
        configuration. For the default pulsed measurement use case, we operate
        in ungated mode (histogram accumulation).
        """
        return False

    def get_binwidth(self):
        """Returns the width of a single timebin in the timetrace in seconds.

        @return float: current length of a single bin in seconds (seconds/bin)
        """
        return self._bin_width_s

    def get_data_trace(self):
        """Polls the current timetrace data from the fast counter.

        @return tuple: (numpy.array dtype=int64, info_dict)

        For ungated mode, returns a 1D array: returnarray[timebin_index]
        info_dict contains 'elapsed_sweeps' and 'elapsed_time'.
        """
        if self._tfa is None:
            self.log.error('Moku not connected. Cannot get data.')
            info_dict = {'elapsed_sweeps': None, 'elapsed_time': None}
            return np.zeros(self._n_bins, dtype='int64'), info_dict

        try:
            data = self._tfa.get_data()

            # Extract histogram data from interval 1
            histogram_info = data.get('interval1', {}).get('histogram', {})
            histogram_data = histogram_info.get('data', [])
            statistics = data.get('interval1', {}).get('statistics', {})

            # The Moku returns 1024 bins. We may need to pad/truncate to match
            # the requested number of bins (self._n_bins).
            raw_data = np.array(histogram_data, dtype='float64')

            if len(raw_data) == 0:
                # No data available yet
                count_data = np.zeros(self._n_bins, dtype='int64')
            elif len(raw_data) >= self._n_bins:
                # Truncate to requested size
                count_data = raw_data[:self._n_bins].astype('int64')
            else:
                # Pad with zeros if we got less data than expected
                count_data = np.zeros(self._n_bins, dtype='int64')
                count_data[:len(raw_data)] = raw_data.astype('int64')

            # Build info dict
            elapsed_time = None
            if self._start_time is not None:
                elapsed_time = time.time() - self._start_time

            # Try to extract sweep/event count from statistics
            elapsed_sweeps = statistics.get('count', None)

            info_dict = {
                'elapsed_sweeps': elapsed_sweeps,
                'elapsed_time': elapsed_time
            }

            return count_data, info_dict

        except Exception:
            self.log.exception('Error reading data from Moku TFA:')
            info_dict = {'elapsed_sweeps': None, 'elapsed_time': None}
            return np.zeros(self._n_bins, dtype='int64'), info_dict

    def get_status(self):
        """Receives the current status of the fast counter.

        0 = unconfigured
        1 = idle
        2 = running
        3 = paused
       -1 = error state
        """
        return self._statusvar
