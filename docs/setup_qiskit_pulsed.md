# Qiskit pulsed toolchain

Compile a single-qubit [Qiskit](https://www.ibm.com/quantum/qiskit) circuit into a qudi pulse block
ensemble and play it on the pulse generator, from a dedicated GUI or from the pulsed measurement
GUI.

The toolchain consists of these qudi modules:

logic:
- sequence_generator_logic (as for every pulsed measurement)
- qiskit_pulsed_logic

hardware:
- pulsegenerator, e.g. awg.tektronix_awg70k or dummy.pulser_dummy

gui:
- qiskit_pulsed_gui

The compilation itself is a set of predefined generate methods (`QiskitPredefinedGenerator` in
`qudi.logic.pulsed.predefined_generate_methods.qiskit_predefined_methods`), so the same circuits are
also available in the predefined methods tab of `pulsed_gui`.


# Installation

Qiskit is an optional dependency:

    python -m pip install -e ".[qiskit]"

or simply `python -m pip install qiskit`.


# Example config

    gui:
        qiskit_pulsed_gui:
            module.Class: 'qiskit_pulsed.qiskit_pulsed_gui.QiskitPulsedGui'
            connect:
                qiskit_pulsed_logic: 'qiskit_pulsed_logic'

    logic:
        qiskit_pulsed_logic:
            module.Class: 'pulsed.qiskit_pulsed_logic.QiskitPulsedLogic'
            connect:
                sequencegeneratorlogic: 'sequence_generator_logic'

        sequence_generator_logic:
            module.Class: 'pulsed.sequence_generator_logic.SequenceGeneratorLogic'
            connect:
                pulsegenerator: 'pulser_dummy'

    hardware:
        pulser_dummy:
            module.Class: 'dummy.pulser_dummy.PulserDummy'


# Usage

1. Set the generation parameters that every pulsed method uses: laser channel, microwave channel,
   laser length, laser delay, wait time, and above all the microwave frequency, the microwave
   amplitude and the Rabi period measured at that amplitude. The Pulse Shape dock of the Qiskit
   window edits the microwave ones; the rest are set in the pulsed measurement GUI.
2. Pick a template in the Circuit dock. `qiskit_gate` plays one gate (x, y, h, s, t, sx, their
   inverses and the pi/2 rotations), `qiskit_ramsey_virtual_z` plays pi/2, a virtual Z and pi/2,
   `qiskit_xy4` and `qiskit_null_sequence` are four-pulse test circuits, `qiskit_gate_string`
   takes a gate string such as `rx(pi/2); rz(90deg); h`, and `qiskit_python_circuit` takes Python
   code building a `QuantumCircuit`. The preview below the parameters shows the transpiled circuit
   and the pulses it produces, or the reason it cannot be compiled.
3. Press *Generate, Sample & Load*. The timeline shows one repetition of the asset, the Generated
   Asset dock lists every element, and the status bar reports progress.
4. Press *Run*. The pulse generator replays the asset until *Stop*.


# How a gate becomes a pulse

A single-qubit rotation has an axis in the equatorial plane and an angle. The axis becomes the phase
of the microwave drive (rx: 0 deg, ry: 90 deg) and the angle becomes the area of the pulse envelope:

    pulse length = |angle| / pi * rabi_period / 2 * duration scale

Three consequences follow:

- `rz` emits no pulse. It shifts the phase of every later pulse (a virtual Z rotation), which
  costs neither time nor microwave power.
- A negative angle is not a shorter pulse. It is the same rotation about the opposite axis, so its
  sign adds 180 deg to the drive phase. Angles are wrapped so no pulse is longer than a pi pulse.
- Changing the envelope changes the required pulse length, see below.

Circuits are transpiled into rx, ry and rz with optimisation level 0, so the pulses stay one-to-one
with the gates as written. Multi-qubit gates, measurements and unbound parameters are rejected.

One block holds the pulses followed by the laser readout, the laser delay and the wait time, as in
the `rabi` method. The laser pulse of one repetition is the initialisation of the next. The rotating
frame is kept across the ensemble, so the drive phases refer to one continuously running reference.


# Pulse envelope and area matching

The envelope is the `pulse_envelope` generation parameter, shared with all other predefined methods.
Besides the rectangle, parabola and sin^n envelopes, a `gaussian` envelope is available. Its
spectrum has no sidelobes, so it drives far less off-resonant transitions than a rectangular pulse
of the same length (about 25 dB less for a pi pulse). It is truncated at `n_sigma` and, when
`lifted`, offset to start and end at exactly zero.

Because the rotation angle is set by the pulse area, a shaped pulse with the same peak and length as
a rectangular one under-rotates by the mean height of its envelope (0.535 for the default Gaussian).
The `area_match` parameter of every Qiskit method compensates for this:

- `duration` (default): every pulse is stretched by the reciprocal of the mean envelope height.
- `amplitude`: the peak is raised instead. Refused if it would exceed the full scale of the channel.
- `none`: no compensation, for debugging.

The mean envelope height is sampled from the very sampling function the pulses use, so it is right
for every envelope type.


# Limits

- One qubit, one microwave channel. Two-qubit gates need a second driven spin and a native
  interaction, which this toolchain does not provide.
- The asset is generated, loaded and played. Fluorescence readout is not acquired; the ensemble does
  carry the usual measurement information, so it can be used with the pulsed measurement logic.
