# -*- coding: utf-8 -*-

"""
This file contains tests for the Qiskit pulsed logic module against a running qudi instance with
the dummy pulse generator.

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

import math
import time
import pytest
from qudi.util.network import netobtain

pytest.importorskip('qiskit')

GUI_MODULE = 'qiskit_pulsed_gui'
LOGIC_MODULE = 'qiskit_pulsed_logic'
RAMSEY_TEMPLATE = 'qiskit_ramsey_virtual_z'
TEMPLATES = (
    'qiskit_gate',
    'qiskit_gate_string',
    'qiskit_null_sequence',
    'qiskit_python_circuit',
    RAMSEY_TEMPLATE,
    'qiskit_xy4',
)
RABI_PERIOD = 100e-9
# Mean height of a Gaussian truncated at 2 sigma and offset to start and end at zero
GAUSSIAN_AREA_FACTOR = 0.5352
TOLERANCE = 1e-3
TIMEOUT = 60


@pytest.fixture(scope='module')
def module(qudi_client):
    """
    Fixture that returns the Qiskit pulsed logic instance, activated through its GUI.

    Parameters
    ----------
    qudi_client : fixture
        qudi instance
    """
    module_manager = qudi_client.module_manager
    module_manager.activate_module(GUI_MODULE)
    logic_instance = module_manager._modules[LOGIC_MODULE].instance
    logic_instance.set_generation_parameters(
        {'microwave_frequency': 2.87e9, 'microwave_amplitude': 0.25, 'rabi_period': RABI_PERIOD}
    )
    wait_until_idle(logic_instance)
    return logic_instance


def wait_until_idle(module, timeout=TIMEOUT):
    """
    Wait until generation, sampling and loading have finished.

    Parameters
    ----------
    module : Object
        Qiskit pulsed logic instance
    timeout : float
        Maximum waiting time in seconds
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = netobtain(module.status_dict)
        if not (status['generation_busy'] or status['sampling_busy'] or status['loading_busy']):
            return status
        time.sleep(0.2)
    raise TimeoutError('The Qiskit pulsed logic did not become idle.')


def mw_elements(summary):
    return [element for element in summary['elements'] if element['mw']]


def test_templates_are_listed(module):
    """
    Tests that every qiskit generate method is offered as a template with its parameters.

    Parameters
    ----------
    module : fixture
        Fixture for instance of the Qiskit pulsed logic module
    """
    templates = netobtain(module.templates)
    assert templates == list(TEMPLATES)
    parameters = netobtain(module.template_parameters(RAMSEY_TEMPLATE))
    assert set(parameters) == {'name', 'phase_deg', 'area_match'}
    assert 'virtual Z' in netobtain(module.template_description(RAMSEY_TEMPLATE))


def test_preview_reports_pulses_and_errors(module):
    """
    Tests the compile preview: pulse list for a valid circuit, a message for an invalid one.

    Parameters
    ----------
    module : fixture
        Fixture for instance of the Qiskit pulsed logic module
    """
    preview = netobtain(module.preview(RAMSEY_TEMPLATE, {'phase_deg': 90.0}))
    assert preview['ok']
    assert preview['gates'] == 3
    assert [pulse['phase'] for pulse in preview['pulses']] == pytest.approx([0.0, 270.0])
    preview = netobtain(module.preview('qiskit_gate_string', {'gates': 'cx'}))
    assert not preview['ok']
    assert 'cx' in preview['error']


def test_generate_ramsey(module):
    """
    Tests that the Ramsey template generates an ensemble with two pi/2 pulses at 0 and 270 degrees
    followed by the laser readout.

    Parameters
    ----------
    module : fixture
        Fixture for instance of the Qiskit pulsed logic module
    """
    module.generate_template(RAMSEY_TEMPLATE, {'name': 'test_ramsey', 'phase_deg': 90.0}, False)
    wait_until_idle(module)
    assert netobtain(module.current_asset) == 'test_ramsey'
    summary = netobtain(module.asset_summary('test_ramsey'))
    pulses = mw_elements(summary)
    assert summary['mw_pulses'] == 2
    assert summary['gates'] == 3
    assert [pulse['phase'] for pulse in pulses] == pytest.approx([0.0, 270.0])
    assert [pulse['duration'] for pulse in pulses] == pytest.approx([RABI_PERIOD / 4] * 2)
    assert summary['elements'][2]['laser']
    assert summary['number_of_lasers'] == 1


def test_sample_and_load(module):
    """
    Tests that the generated ensemble is sampled and loaded into the pulse generator.

    Parameters
    ----------
    module : fixture
        Fixture for instance of the Qiskit pulsed logic module
    """
    module.generate_template(RAMSEY_TEMPLATE, {'name': 'test_ramsey', 'phase_deg': 90.0}, True)
    wait_until_idle(module)
    loaded_asset, asset_type = netobtain(module.loaded_asset)
    assert loaded_asset == 'test_ramsey'
    assert asset_type == 'PulseBlockEnsemble'


def test_toggle_pulser(module):
    """
    Tests starting and stopping the pulse generator once an asset is loaded.

    Parameters
    ----------
    module : fixture
        Fixture for instance of the Qiskit pulsed logic module
    """
    module.toggle_pulser(True)
    time.sleep(2)
    assert netobtain(module.pulser_running)
    module.toggle_pulser(False)
    time.sleep(1)
    assert not netobtain(module.pulser_running)


def test_gaussian_envelope_scales_the_pulse_length(module):
    """
    Tests that selecting the Gaussian envelope stretches the pulses by the reciprocal area factor.

    Parameters
    ----------
    module : fixture
        Fixture for instance of the Qiskit pulsed logic module
    """
    from qudi.logic.pulsed.sampling_functions import PulseEnvelope, PulseEnvelopeType

    module.set_pulse_envelope(PulseEnvelope(PulseEnvelopeType.gaussian))
    time.sleep(1)
    shape = netobtain(module.pulse_shape_summary)
    assert shape['envelope'] == 'gaussian'
    assert math.isclose(shape['area_factor'], GAUSSIAN_AREA_FACTOR, abs_tol=TOLERANCE)

    module.generate_template(RAMSEY_TEMPLATE, {'name': 'test_gauss', 'phase_deg': 0.0}, False)
    wait_until_idle(module)
    pulses = mw_elements(netobtain(module.asset_summary('test_gauss')))
    assert [pulse['function'] for pulse in pulses] == ['SinEnvelopeGaussian'] * 2
    assert pulses[0]['duration'] == pytest.approx(RABI_PERIOD / 4 / GAUSSIAN_AREA_FACTOR, rel=TOLERANCE)
    assert pulses[0]['envelope'][0] == pytest.approx(0.0, abs=1e-6)
    assert max(pulses[0]['envelope']) == pytest.approx(1.0, abs=1e-3)

    module.set_pulse_envelope(PulseEnvelope(PulseEnvelopeType.rectangle))
    time.sleep(1)
    assert netobtain(module.pulse_shape_summary)['envelope'] == 'rectangle'
