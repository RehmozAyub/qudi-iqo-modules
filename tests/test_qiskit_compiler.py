# -*- coding: utf-8 -*-

"""
This file contains unit tests for the Qiskit circuit compiler of the pulsed toolchain.

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
import numpy as np
import pytest

qiskit = pytest.importorskip('qiskit')
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402

from qudi.logic.pulsed.qiskit_compiler import (  # noqa: E402
    DEFAULT_GATE_STRING,
    DEFAULT_PYTHON_CIRCUIT,
    DEFAULT_PYTHON_CIRCUIT_ONE_LINE,
    SingleQubitGate,
    build_template_circuit,
    circuit_from_gate_string,
    circuit_from_python_source,
    compile_circuit,
    parse_angle,
)

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def rotation(angle, phase_deg):
    """
    Matrix of a rotation by `angle` about the equatorial axis at `phase_deg`.

    Parameters
    ----------
    angle : float
        Rotation angle in rad.
    phase_deg : float
        Axis angle in degrees, 0 for X and 90 for Y.

    Returns
    -------
    numpy.ndarray
        The 2x2 unitary.
    """
    phi = math.radians(phase_deg)
    axis = math.cos(phi) * PAULI_X + math.sin(phi) * PAULI_Y
    return math.cos(angle / 2) * np.eye(2) - 1j * math.sin(angle / 2) * axis


def z_rotation(angle):
    """
    Matrix of a rotation by `angle` about Z, in the convention of qiskit's RZGate.
    """
    return np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])


def emitted_unitary(compiled):
    """
    The unitary implemented by the emitted pulses, including the frame rotation left at the end.

    Parameters
    ----------
    compiled : CompiledCircuit
        Output of compile_circuit.

    Returns
    -------
    numpy.ndarray
        The 2x2 unitary.
    """
    unitary = np.eye(2, dtype=complex)
    for pulse in compiled.pulses:
        unitary = rotation(pulse.angle, pulse.phase) @ unitary
    return z_rotation(-math.radians(compiled.frame_phase)) @ unitary


def gate_names(circuit):
    return [instruction.operation.name for instruction in circuit.data]


@pytest.mark.parametrize(
    'expression, expected',
    [
        ('pi/2', math.pi / 2),
        ('90deg', math.pi / 2),
        ('-3*pi/4', -3 * math.pi / 4),
        ('2', 2.0),
        ('PI**2 / 4', math.pi**2 / 4),
        ('  -45 deg', -math.pi / 4),
    ],
)
def test_parse_angle(expression, expected):
    """
    Tests that angle expressions in radians and degrees are evaluated.
    """
    assert parse_angle(expression) == pytest.approx(expected)


@pytest.mark.parametrize('expression', ['import os', '__import__("os")', 'foo', 'pi/0', '', 'pi(2)', '[1]', 'True'])
def test_parse_angle_rejects_anything_but_arithmetic(expression):
    """
    Tests that only plain arithmetic on numbers and pi is accepted, so no code can run through it.
    """
    with pytest.raises(ValueError):
        parse_angle(expression)


def test_default_gate_string_builds_three_gates():
    """
    Tests that the default gate string is a valid single-qubit circuit.
    """
    circuit = circuit_from_gate_string(DEFAULT_GATE_STRING)
    assert circuit.num_qubits == 1
    assert gate_names(circuit) == ['rx', 'rz', 'rx']


def test_gate_string_accepts_names_angles_and_separators():
    """
    Tests the gate string grammar: case-insensitive names, angles in brackets, three separators.
    """
    circuit = circuit_from_gate_string('h, T\nry(45deg); SX')
    assert gate_names(circuit) == ['h', 't', 'ry', 'sx']
    assert circuit.data[2].operation.params[0] == pytest.approx(math.pi / 4)


@pytest.mark.parametrize('text', ['', 'rx', 'rx()', 'x(1)', 'cx(0,1)', 'hello', 'rx(pi/2) ry(pi)'])
def test_gate_string_rejects_invalid_input(text):
    """
    Tests that malformed and unsupported gates are rejected with a ValueError.
    """
    with pytest.raises(ValueError):
        circuit_from_gate_string(text)


@pytest.mark.parametrize('source', [DEFAULT_PYTHON_CIRCUIT, DEFAULT_PYTHON_CIRCUIT_ONE_LINE])
def test_default_python_sources_build_the_ramsey_circuit(source):
    """
    Tests that both prefilled Python samples run and build the same three-gate circuit.
    """
    assert gate_names(circuit_from_python_source(source)) == ['rx', 'rz', 'rx']


def test_python_source_finds_a_single_unnamed_circuit():
    """
    Tests the fallback to the only QuantumCircuit when no variable is called "circuit".
    """
    circuit = circuit_from_python_source('qc = QuantumCircuit(1)\nqc.h(0)\nqc.t(0)')
    assert gate_names(circuit) == ['h', 't']


def test_python_source_may_import():
    """
    Tests that the source can import from qiskit like any script.
    """
    source = 'from qiskit.circuit.library import HGate\ncircuit = QuantumCircuit(1)\ncircuit.append(HGate(), [0])'
    assert gate_names(circuit_from_python_source(source)) == ['h']


def test_python_source_without_circuit_is_rejected():
    """
    Tests that source that does not build a circuit raises a ValueError.
    """
    with pytest.raises(ValueError):
        circuit_from_python_source('x = 1')


def test_ramsey_compiles_to_two_pulses_with_a_virtual_z():
    """
    Tests the virtual Z: three gates become two pulses and the rz shifts the second phase by -90 deg.
    """
    compiled = compile_circuit(build_template_circuit('ramsey_virtual_z', {'phase_deg': 90.0}))
    assert compiled.gate_count == 3
    assert [pulse.gate for pulse in compiled.pulses] == ['rx', 'rx']
    assert [pulse.angle for pulse in compiled.pulses] == pytest.approx([math.pi / 2, math.pi / 2])
    assert [pulse.phase for pulse in compiled.pulses] == pytest.approx([0.0, 270.0])
    assert compiled.frame_phase == pytest.approx(270.0)
    assert 'Rx' in compiled.drawing


def test_negative_angle_becomes_opposite_phase():
    """
    Tests that a negative rotation is a positive rotation about the opposite axis.
    """
    circuit = QuantumCircuit(1)
    circuit.rx(-math.pi / 2, 0)
    (pulse,) = compile_circuit(circuit).pulses
    assert pulse.angle == pytest.approx(math.pi / 2)
    assert pulse.phase == pytest.approx(180.0)


def test_angles_wrap_so_no_pulse_exceeds_pi():
    """
    Tests that 3pi/2 about Y is emitted as pi/2 about -Y.
    """
    circuit = QuantumCircuit(1)
    circuit.ry(3 * math.pi / 2, 0)
    (pulse,) = compile_circuit(circuit).pulses
    assert pulse.angle == pytest.approx(math.pi / 2)
    assert pulse.phase == pytest.approx(270.0)


def test_tiny_rotations_and_structural_instructions_emit_nothing():
    """
    Tests that negligible rotations, barriers and identities produce no pulse.
    """
    circuit = QuantumCircuit(1)
    circuit.rx(1e-12, 0)
    circuit.barrier()
    circuit.id(0)
    assert compile_circuit(circuit).pulses == []


def test_multi_qubit_circuits_are_rejected():
    """
    Tests that a circuit with more than one qubit raises instead of being approximated.
    """
    with pytest.raises(NotImplementedError):
        compile_circuit(QuantumCircuit(2))


def test_measurements_are_rejected():
    """
    Tests that measure instructions raise, since readout is the laser pulse.
    """
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()
    with pytest.raises(NotImplementedError):
        compile_circuit(circuit)


def test_unbound_parameters_are_rejected():
    """
    Tests that a circuit with an unbound parameter raises a ValueError.
    """
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter('theta'), 0)
    with pytest.raises(ValueError):
        compile_circuit(circuit)


TEMPLATE_CASES = [('gate', {'gate': gate}) for gate in SingleQubitGate] + [
    ('ramsey_virtual_z', {'phase_deg': 0.0}),
    ('ramsey_virtual_z', {'phase_deg': 137.0}),
    ('xy4', {}),
    ('null_sequence', {}),
    ('gate_string', {'gates': 'h; t; sdg; ry(1.234); rz(-2.5); rx(4); p(0.7); z'}),
    ('python_circuit', {'source': DEFAULT_PYTHON_CIRCUIT}),
]


@pytest.mark.parametrize('template, parameters', TEMPLATE_CASES, ids=str)
def test_emitted_pulses_reproduce_the_circuit_unitary(template, parameters):
    """
    Tests that the pulses of every template reproduce the circuit unitary up to a global phase.
    """
    circuit = build_template_circuit(template, parameters)
    compiled = compile_circuit(circuit)
    assert Operator(circuit).equiv(Operator(emitted_unitary(compiled)))


def test_template_gate_accepts_members_names_and_values():
    """
    Tests that the gate template takes a SingleQubitGate, its name or its gate string.
    """
    for gate in (SingleQubitGate.x, 'x', 'rx(pi/2)'):
        assert build_template_circuit('gate', {'gate': gate}).num_qubits == 1


def test_unknown_template_is_rejected():
    """
    Tests that an unknown template name raises a ValueError.
    """
    with pytest.raises(ValueError):
        build_template_circuit('nope', {})
