# -*- coding: utf-8 -*-

"""
This file contains unit tests for the Gaussian pulse envelope sampling functions.

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

from qudi.logic.pulsed.sampling_functions import PulseEnvelope, PulseEnvelopeType
from qudi.logic.pulsed.sampling_function_defs.shaped_sine_functions import SinEnvelopeGaussian

N_SIGMA = 2.0
# Mean height of a Gaussian truncated at 2 sigma and offset to start and end at zero
LIFTED_AREA_FACTOR = 0.5352
TOLERANCE = 1e-3
SAMPLES = 200000


def envelope_samples(n_sigma, lifted):
    """
    Sample the envelope alone. Frequency 0 and phase 90 degrees turn the carrier into a constant 1.

    Parameters
    ----------
    n_sigma : float
        Truncation point of the Gaussian in units of sigma.
    lifted : bool
        Whether the envelope is offset to start and end at zero.

    Returns
    -------
    numpy.ndarray
        Envelope values.
    """
    function = SinEnvelopeGaussian(amplitude=1.0, frequency=0.0, phase=90.0, n_sigma=n_sigma, lifted=lifted)
    return function.get_samples(np.linspace(0.0, 1e-6, SAMPLES))


def test_gaussian_envelope_type_has_default_parameters():
    """
    Tests that the gaussian envelope type is known to PulseEnvelope and carries its defaults.
    """
    envelope = PulseEnvelope(PulseEnvelopeType.gaussian)
    assert envelope.parameters == {'n_sigma': N_SIGMA, 'lifted': True}


def test_lifted_envelope_starts_at_zero_and_peaks_at_one_in_the_middle():
    """
    Tests the shape of the lifted envelope: zero at the start, full height in the centre.
    """
    samples = envelope_samples(N_SIGMA, True)
    assert samples[0] == pytest.approx(0.0, abs=1e-9)
    assert samples.max() == pytest.approx(1.0, abs=1e-9)
    assert abs(int(np.argmax(samples)) - SAMPLES // 2) <= 1


def test_unlifted_envelope_steps_at_the_edges():
    """
    Tests that without lifting the envelope starts at the truncated Gaussian value.
    """
    samples = envelope_samples(N_SIGMA, False)
    assert samples[0] == pytest.approx(math.exp(-0.5 * N_SIGMA**2), abs=1e-9)


def test_lifted_area_factor():
    """
    Tests the mean envelope height, which sets how much rotation a shaped pulse delivers.
    """
    assert envelope_samples(N_SIGMA, True).mean() == pytest.approx(LIFTED_AREA_FACTOR, abs=TOLERANCE)


def test_envelope_multiplies_the_bare_sine():
    """
    Tests that the shaped samples are the plain sine multiplied by the envelope.
    """
    time_array = np.linspace(0.0, 1e-6, 10001)
    shaped = SinEnvelopeGaussian(amplitude=0.25, frequency=10e6, phase=0.0).get_samples(time_array)
    bare = 0.25 * np.sin(2 * np.pi * 10e6 * time_array)
    envelope = SinEnvelopeGaussian.envelope(np.arange(time_array.size) / time_array.size, N_SIGMA, True)
    np.testing.assert_allclose(shaped, bare * envelope, atol=1e-12)
