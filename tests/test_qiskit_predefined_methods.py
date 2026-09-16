# -*- coding: utf-8 -*-

"""
This file contains unit tests for the Qiskit predefined generate methods. They run without a qudi
instance against a stand-in for the sequence generator logic.

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

import logging
import math
import pytest

pytest.importorskip('qiskit')

from qudi.logic.pulsed.predefined_generate_methods.qiskit_predefined_methods import (  # noqa: E402
    QiskitPredefinedGenerator,
)
from qudi.logic.pulsed.qiskit_compiler import AreaMatch, SingleQubitGate  # noqa: E402
from qudi.logic.pulsed.sampling_functions import PulseEnvelope, PulseEnvelopeType, SamplingFunctions  # noqa: E402

# The sampling functions are normally collected by the sequence generator logic on activation.
SamplingFunctions.import_sampling_functions([])

RABI_PERIOD = 100e-9
LASER_LENGTH = 3e-6
LASER_DELAY = 500e-9
WAIT_TIME = 1e-6
MW_FREQUENCY = 2.87e9
MW_AMPLITUDE = 0.25
FULL_SCALE_PP = 0.5
# Mean height of a Gaussian truncated at 2 sigma and offset to start and end at zero
GAUSSIAN_AREA_FACTOR = 0.5352
TOLERANCE = 1e-3


class FakeSequenceGeneratorLogic:
    """
    Stands in for SequenceGeneratorLogic. Only what PredefinedGeneratorBase reads is provided.
    """

    def __init__(self):
        self.log = logging.getLogger('FakeSequenceGeneratorLogic')
        self.generation_parameters = {
            'laser_channel': 'd_ch1',
            'sync_channel': '',
            'gate_channel': '',
            'microwave_channel': 'a_ch1',
            'microwave_frequency': MW_FREQUENCY,
            'microwave_amplitude': MW_AMPLITUDE,
            'rabi_period': RABI_PERIOD,
            'laser_length': LASER_LENGTH,
            'laser_delay': LASER_DELAY,
            'wait_time': WAIT_TIME,
            'analog_trigger_voltage': 0.0,
            'pulse_envelope': PulseEnvelope(PulseEnvelopeType.rectangle),
            'pulse_envelope_order': 1,
        }
        self.pulse_generator_settings = {
            'activation_config': ('config4', frozenset({'a_ch1', 'd_ch1', 'd_ch2'})),
            'sample_rate': 25e9,
            'analog_levels': ({'a_ch1': FULL_SCALE_PP}, {'a_ch1': 0.0}),
        }
        self.pulse_generator_constraints = None

    def save_block(self, block):
        pass

    def save_ensemble(self, ensemble):
        pass

    def save_sequence(self, sequence):
        pass

    def analyze_block_ensemble(self, ensemble):
        return dict()

    def analyze_sequence(self, sequence):
        return dict()


@pytest.fixture
def generator():
    """
    Fixture that returns a generator bound to a fresh fake sequence generator logic.
    """
    return QiskitPredefinedGenerator(FakeSequenceGeneratorLogic())


def set_envelope(generator, envelope_type, **parameters):
    """
    Select the pulse envelope in the generation parameters, as the pulsed GUI settings would.
    """
    envelope = PulseEnvelope(envelope_type, dict(parameters)) if parameters else PulseEnvelope(envelope_type)
    generator.generation_parameters = {'pulse_envelope': envelope}


def mw_elements(block):
    """
    The elements of a block that carry a microwave pulse on the microwave channel.
    """
    return [element for element in block.element_list if type(element.pulse_function['a_ch1']).__name__ != 'Idle']


def test_ramsey_block_holds_two_pulses_and_the_readout(generator):
    """
    Tests the block layout: two pi/2 pulses at 0 and 270 degrees, then laser, delay and wait.
    """
    blocks, ensembles, sequences = generator.generate_qiskit_ramsey_virtual_z(name='ramsey', phase_deg=90.0)
    assert sequences == []
    (block,) = blocks
    (ensemble,) = ensembles
    assert block.name == 'ramsey'
    assert ensemble.name == 'ramsey'

    elements = block.element_list
    assert len(elements) == 5
    for pulse in elements[:2]:
        function = pulse.pulse_function['a_ch1']
        assert pulse.init_length_s == pytest.approx(RABI_PERIOD / 4)
        assert type(function).__name__ == 'Sin'
        assert function.amplitude == pytest.approx(MW_AMPLITUDE)
        assert function.frequency == pytest.approx(MW_FREQUENCY)
    assert [pulse.pulse_function['a_ch1'].phase for pulse in elements[:2]] == pytest.approx([0.0, 270.0])

    laser, delay, wait = elements[2:]
    assert laser.laser_on
    assert laser.digital_high['d_ch1']
    assert laser.init_length_s == pytest.approx(LASER_LENGTH)
    assert not delay.laser_on
    assert delay.init_length_s == pytest.approx(LASER_DELAY)
    assert wait.init_length_s == pytest.approx(WAIT_TIME)

    assert ensemble.rotating_frame is True
    assert ensemble.block_list == [('ramsey', 0)]
    assert ensemble.measurement_information['number_of_lasers'] == 1
    assert list(ensemble.measurement_information['controlled_variable']) == [0.0]


def test_gaussian_envelope_stretches_the_pulses(generator):
    """
    Tests duration area matching: a Gaussian pulse is longer by the reciprocal of its area factor.
    """
    set_envelope(generator, PulseEnvelopeType.gaussian)
    (block,), _, _ = generator.generate_qiskit_ramsey_virtual_z(name='ramsey')
    pulse = block.element_list[0]
    function = pulse.pulse_function['a_ch1']
    assert type(function).__name__ == 'SinEnvelopeGaussian'
    assert pulse.init_length_s == pytest.approx(RABI_PERIOD / 4 / GAUSSIAN_AREA_FACTOR, rel=TOLERANCE)
    assert function.amplitude == pytest.approx(MW_AMPLITUDE)


def test_amplitude_area_matching_raises_when_it_would_clip(generator):
    """
    Tests that raising the peak beyond full scale is refused instead of producing a clipped pulse.
    """
    set_envelope(generator, PulseEnvelopeType.gaussian)
    with pytest.raises(ValueError):
        generator.generate_qiskit_xy4(area_match=AreaMatch.amplitude)


def test_amplitude_area_matching_raises_the_peak_when_there_is_headroom(generator):
    """
    Tests amplitude area matching: same length, peak divided by the area factor.
    """
    set_envelope(generator, PulseEnvelopeType.gaussian)
    generator.generation_parameters = {'microwave_amplitude': 0.1}
    (block,), _, _ = generator.generate_qiskit_xy4(area_match=AreaMatch.amplitude)
    pulse = block.element_list[0]
    assert pulse.init_length_s == pytest.approx(RABI_PERIOD / 2)
    assert pulse.pulse_function['a_ch1'].amplitude == pytest.approx(0.1 / GAUSSIAN_AREA_FACTOR, rel=TOLERANCE)


def test_no_area_matching_keeps_length_and_amplitude(generator):
    """
    Tests that AreaMatch.none, also given by name, leaves length and peak untouched.
    """
    set_envelope(generator, PulseEnvelopeType.gaussian)
    for area_match in (AreaMatch.none, 'none'):
        (block,), _, _ = generator.generate_qiskit_xy4(area_match=area_match)
        pulse = block.element_list[0]
        assert pulse.init_length_s == pytest.approx(RABI_PERIOD / 2)
        assert pulse.pulse_function['a_ch1'].amplitude == pytest.approx(MW_AMPLITUDE)


@pytest.mark.parametrize(
    'envelope_type, parameters, expected',
    [
        (PulseEnvelopeType.rectangle, {}, 1.0),
        (PulseEnvelopeType.parabola, {'order': 1}, 2 / 3),
        (PulseEnvelopeType.sin_n, {'order': 1}, 2 / math.pi),
        (PulseEnvelopeType.gaussian, {}, GAUSSIAN_AREA_FACTOR),
    ],
    ids=lambda value: value.name if isinstance(value, PulseEnvelopeType) else None,
)
def test_envelope_area_factor(generator, envelope_type, parameters, expected):
    """
    Tests the numerical area factor for every supported envelope type.
    """
    envelope = PulseEnvelope(envelope_type, dict(parameters)) if parameters else PulseEnvelope(envelope_type)
    assert generator.envelope_area_factor(envelope) == pytest.approx(expected, abs=TOLERANCE)


def test_single_gate_templates(generator):
    """
    Tests two ready-made gates: x is a pi pulse at 0 degrees, y_minus_90 a pi/2 pulse at 270 degrees.
    """
    (block,), _, _ = generator.generate_qiskit_gate(gate=SingleQubitGate.x)
    (pulse,) = mw_elements(block)
    assert pulse.init_length_s == pytest.approx(RABI_PERIOD / 2)
    assert pulse.pulse_function['a_ch1'].phase == pytest.approx(0.0)

    (block,), _, _ = generator.generate_qiskit_gate(gate=SingleQubitGate.y_minus_90)
    (pulse,) = mw_elements(block)
    assert pulse.init_length_s == pytest.approx(RABI_PERIOD / 4)
    assert pulse.pulse_function['a_ch1'].phase == pytest.approx(270.0)


def test_gate_string_and_python_templates_compile(generator):
    """
    Tests the two free-form templates. The z gate emits no pulse.
    """
    (block,), _, _ = generator.generate_qiskit_gate_string(gates='x; y; z')
    assert [pulse.pulse_function['a_ch1'].phase for pulse in mw_elements(block)] == pytest.approx([0.0, 90.0])
    (block,), _, _ = generator.generate_qiskit_python_circuit()
    assert len(mw_elements(block)) == 2


def test_multi_qubit_python_circuit_is_rejected(generator):
    """
    Tests that a two-qubit circuit raises instead of being emitted on one channel.
    """
    with pytest.raises(NotImplementedError):
        generator.generate_qiskit_python_circuit(source='circuit = QuantumCircuit(2); circuit.cx(0, 1)')
