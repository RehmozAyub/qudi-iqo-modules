# -*- coding: utf-8 -*-
"""
This file contains the Qudi logic for the Qiskit bridge.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
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

from qudi.core.module import LogicBase
from qudi.logic.pulsed.pulse_objects import (
    PulseBlockElement,
    PulseBlock,
    PulseBlockEnsemble,
    PulseSequence,
)
from qudi.logic.pulsed.sampling_functions import SamplingFunctions


class QiskitBridgeLogic(LogicBase):
    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        return

    def on_deactivate(self):
        """ Deactivate the module properly.
        """
        return

    def _make_pulse_fn(self, a_names, mw_channel='', frequency=2.87e9, amplitude=0.5, phase=0.0):
        """
        DC zero on all analog channels; Sin on mw_channel if specified.

        @param list a_names: List of analog channel names
        @param str mw_channel: The microwave channel name
        @param float frequency: The frequency of the microwave pulse
        @param float amplitude: The amplitude of the microwave pulse
        @param float phase: The phase of the microwave pulse
        @return dict: Dictionary mapping analog channels to their sampling functions
        """
        return {
            ch: SamplingFunctions.Sin(amplitude=amplitude, frequency=frequency, phase=float(phase))
            if ch == mw_channel
            else SamplingFunctions.DC(voltage=0.0)
            for ch in a_names
        }

    def make_block_single_element(self, block_name, init_length_s,
                                  d_ch_names, ch_active,
                                  a_ch_names, increment_s=0,
                                  mw_channel='', frequency=2.87e9, amplitude=0.5, phase=0.0):
        """
        Create a pulse block with a single element.

        @param str block_name: Name of the pulse block
        @param float init_length_s: Initial length of the pulse element in seconds
        @param str d_ch_names: Comma-separated list of digital channel names
        @param str ch_active: The active digital channel name
        @param str a_ch_names: Comma-separated list of analog channel names
        @param float increment_s: Increment length in seconds
        @param str mw_channel: The microwave channel name
        @param float frequency: The frequency of the microwave pulse
        @param float amplitude: The amplitude of the microwave pulse
        @param float phase: The phase of the microwave pulse
        @return PulseBlock: The generated pulse block
        """
        d_names = [c.strip() for c in d_ch_names.split(',') if c.strip()]
        a_names = [c.strip() for c in a_ch_names.split(',') if c.strip()]
        # Only set digital HIGH if ch_active is actually a digital channel
        d_high = {ch: (ch == ch_active and ch in d_names) for ch in d_names}
        pulse_fn = self._make_pulse_fn(
            a_names,
            mw_channel=mw_channel,
            frequency=frequency,
            amplitude=amplitude,
            phase=phase
        )
        element = PulseBlockElement(
            init_length_s=float(init_length_s),
            increment_s=float(increment_s),
            pulse_function=pulse_fn,
            digital_high=d_high
        )
        return PulseBlock(name=str(block_name), element_list=[element])

    def make_block_two_elements(self, block_name,
                                length1, d_ch_names, ch_active1,
                                length2, ch_active2, a_ch_names,
                                mw_channel='', frequency=2.87e9, amplitude=0.5, phase=0.0):
        """
        Create a pulse block with two elements.

        @param str block_name: Name of the pulse block
        @param float length1: Initial length of the first pulse element in seconds
        @param str d_ch_names: Comma-separated list of digital channel names
        @param str ch_active1: The active digital channel name for the first element
        @param float length2: Initial length of the second pulse element in seconds
        @param str ch_active2: The active digital channel name for the second element
        @param str a_ch_names: Comma-separated list of analog channel names
        @param str mw_channel: The microwave channel name
        @param float frequency: The frequency of the microwave pulse
        @param float amplitude: The amplitude of the microwave pulse
        @param float phase: The phase of the microwave pulse
        @return PulseBlock: The generated pulse block
        """
        d_names = [c.strip() for c in d_ch_names.split(',') if c.strip()]
        a_names = [c.strip() for c in a_ch_names.split(',') if c.strip()]
        pulse_fn = self._make_pulse_fn(
            a_names,
            mw_channel=mw_channel,
            frequency=frequency,
            amplitude=amplitude,
            phase=phase
        )
        el1 = PulseBlockElement(
            init_length_s=float(length1),
            increment_s=0,
            pulse_function=pulse_fn,
            digital_high={ch: (ch == ch_active1 and ch in d_names) for ch in d_names}
        )
        el2 = PulseBlockElement(
            init_length_s=float(length2),
            increment_s=0,
            pulse_function=pulse_fn,
            digital_high={ch: (ch == ch_active2 and ch in d_names) for ch in d_names}
        )
        return PulseBlock(name=str(block_name), element_list=[el1, el2])

    def make_ensemble(self, name, block_name, repetitions=0):
        """
        Create a pulse block ensemble.

        @param str name: Name of the ensemble
        @param str block_name: Name of the pulse block to include in the ensemble
        @param int repetitions: Number of repetitions
        @return PulseBlockEnsemble: The generated pulse block ensemble
        """
        return PulseBlockEnsemble(
            name=str(name),
            block_list=[(str(block_name), int(repetitions))]
        )

    def make_sequence(self, name, ensemble_names, rotating_frame=False):
        """
        Create a pulse sequence.

        @param str name: Name of the sequence
        @param str ensemble_names: Comma-separated list of ensemble names
        @param bool rotating_frame: Flag indicating if rotating frame is used
        @return PulseSequence: The generated pulse sequence
        """
        names = [e.strip() for e in ensemble_names.split(',') if e.strip()]
        return PulseSequence(
            name=str(name),
            ensemble_list=names,
            rotating_frame=bool(rotating_frame)
        )