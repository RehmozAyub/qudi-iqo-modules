# -*- coding: utf-8 -*-

"""
A dummy hardware module for using a Liquid Instruments Moku device as a fast counter (timetagger).

This module implements the FastCounterInterface and simulates the behavior of the Moku
Time & Frequency Analyzer for testing without the physical hardware.

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


class MokuFastCounterDummy(FastCounterInterface):
    """Dummy hardware class for Moku fast counter (timetagger).

    Example config for copy-paste:

    moku_fast_counter_dummy:
        module.Class: 'moku_fast_counter_dummy.MokuFastCounterDummy'
        options:
            moku_ip_address: '192.168.0.100'  # Ignored by dummy, but kept for compatibility
            input_channel: 'Input1'
            trigger_channel: 'Input2'
            event_threshold: 0.0
            event_edge: 'Rising'
            trigger_threshold: 0.0
            trigger_edge: 'Rising'
            force_connect: True
            gated: False
    """

    # Config options mapping directly to the real Moku hardware module
    _ip_address = ConfigOption('moku_ip_address', missing='error')
    _input_channel = ConfigOption('input_channel', default='Input1', missing='info')
    _trigger_channel = ConfigOption('trigger_channel', default='Input2', missing='info')
    _event_threshold = ConfigOption('event_threshold', default=0.0, missing='info')
    _event_edge = ConfigOption('event_edge', default='Rising', missing='info')
    _trigger_threshold = ConfigOption('trigger_threshold', default=0.0, missing='info')
    _trigger_edge = ConfigOption('trigger_edge', default='Rising', missing='info')
    _force_connect = ConfigOption('force_connect', default=True, missing='info')
    _gated = ConfigOption('gated', default=False, missing='info')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Internal state
        self.statusvar = 0        # 0=unconfigured, 1=idle, 2=running, 3=paused, -1=error
        self._bin_width_s = 1e-9  # bin width in seconds
        self._record_length_s = 0 # total record length in seconds
        self._number_of_gates = 0
        self._n_bins = 0          # number of bins in the histogram
        self._start_time = None   # timestamp when measurement started

    def on_activate(self):
        """Simulation of connecting to the Moku device."""
        self.log.info(f"Connecting to dummy Moku at {self._ip_address}...")
        self.statusvar = 0
        self.log.info("Dummy Moku fast counter activated successfully.")

    def on_deactivate(self):
        """Simulation of disconnecting from the Moku device."""
        self.log.info("Dummy Moku connection released.")
        self.statusvar = -1

    def get_constraints(self):
        """Retrieve hardware constraints."""
        constraints = dict()
        constraints['hardware_binwidth_list'] = [
            7.8e-12, 1e-9, 2e-9, 4e-9, 8e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9, 1e-6,
        ]
        return constraints

    def configure(self, bin_width_s, record_length_s, number_of_gates=0):
        """Configuration of the dummy fast counter."""
        self._n_bins = int(np.rint(record_length_s / bin_width_s))
        if self._n_bins > 1024:
             self.log.warning(f"Requested {self._n_bins} bins, but Moku is fixed at 1024 bins. Adjusting...")
             self._n_bins = min(self._n_bins, 1024)

        actual_binwidth = bin_width_s
        actual_length = self._n_bins * actual_binwidth

        self._bin_width_s = actual_binwidth
        self._record_length_s = actual_length
        self._number_of_gates = number_of_gates

        self.log.info(
            f'Dummy Moku configured: bin_width={self._bin_width_s*1e9:.1f}ns, '
            f'record_length={self._record_length_s*1e6:.1f}us, '
            f'gates={number_of_gates}'
        )

        self.statusvar = 1

        return actual_binwidth, actual_length, number_of_gates

    def start_measure(self):
        """Start the measurement."""
        self._start_time = time.time()
        self.module_state.lock()
        self.statusvar = 2
        self.log.debug('Dummy Moku measurement started.')
        return 0

    def stop_measure(self):
        """Stop the measurement."""
        if self.module_state() == 'locked':
            self.module_state.unlock()
        self.statusvar = 1
        self.log.debug('Dummy Moku measurement stopped.')
        return 0

    def pause_measure(self):
        """Pause the measurement."""
        if self.module_state() == 'locked':
            self.statusvar = 3
            self.log.debug('Dummy Moku measurement paused.')
        return 0

    def continue_measure(self):
        """Continue the measurement."""
        if self.module_state() == 'locked':
            self.statusvar = 2
            self.log.debug('Dummy Moku measurement continued.')
        return 0

    def is_gated(self):
        """Check gated counting."""
        return self._gated

    def get_binwidth(self):
        """Returns the width of a single timebin in seconds."""
        return self._bin_width_s

    def get_frequency(self):
        """Returns frequency."""
        return 1e6 # dummy frequency

    def get_data_trace(self):
        """Poll the dummy timetrace data."""
        time.sleep(0.1) # Simulate hardware polling delay

        if self.statusvar != 2 and self.statusvar != 3:
            if self._gated:
                return np.zeros((self._number_of_gates, self._n_bins), dtype='int64'), {'elapsed_sweeps': None, 'elapsed_time': None}
            return np.zeros(self._n_bins, dtype='int64'), {'elapsed_sweeps': None, 'elapsed_time': None}

        # Generate some dummy data resembling plausible timetrace data (e.g. exponential decay + noise)
        x = np.arange(self._n_bins)
        # Add basic exponential decay peak
        decay = np.exp(-x / (max(self._n_bins, 1) / 10.0)) * 1000
        # Add baseline noise
        noise = np.random.poisson(100, size=self._n_bins)
        
        count_data = (decay + noise).astype('int64')

        if self._gated:
            count_data = np.tile(count_data, (self._number_of_gates, 1))
            count_data += np.random.poisson(5, size=(self._number_of_gates, self._n_bins)).astype('int64')

        elapsed_time = None
        if self._start_time is not None:
            elapsed_time = time.time() - self._start_time

        info_dict = {
            'elapsed_sweeps': int(elapsed_time * 10) if elapsed_time else 1,
            'elapsed_time': elapsed_time
        }

        return count_data, info_dict

    def get_status(self):
        """Returns status."""
        return self.statusvar
