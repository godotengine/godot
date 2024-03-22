/*
 * Copyright 2019 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef RESAMPLER_KAISER_WINDOW_H
#define RESAMPLER_KAISER_WINDOW_H

#include <math.h>

#include "ResamplerDefinitions.h"

namespace RESAMPLER_OUTER_NAMESPACE::resampler {

/**
 * Calculate a Kaiser window centered at 0.
 */
class KaiserWindow {
public:
    KaiserWindow() {
        setStopBandAttenuation(60);
    }

    /**
     * @param attenuation typical values range from 30 to 90 dB
     * @return beta
     */
    double setStopBandAttenuation(double attenuation) {
        double beta = 0.0;
        if (attenuation > 50) {
            beta = 0.1102 * (attenuation - 8.7);
        } else if (attenuation >= 21) {
            double a21 = attenuation - 21;
            beta = 0.5842 * pow(a21, 0.4) + (0.07886 * a21);
        }
        setBeta(beta);
        return beta;
    }

    void setBeta(double beta) {
        mBeta = beta;
        mInverseBesselBeta = 1.0 / bessel(beta);
    }

    /**
     * @param x ranges from -1.0 to +1.0
     */
    double operator()(double x) {
        double x2 = x * x;
        if (x2 >= 1.0) return 0.0;
        double w = mBeta * sqrt(1.0 - x2);
        return bessel(w) * mInverseBesselBeta;
    }

    // Approximation of a
    // modified zero order Bessel function of the first kind.
    // Based on a discussion at:
    // https://dsp.stackexchange.com/questions/37714/kaiser-window-approximation
    static double bessel(double x) {
        double y = cosh(0.970941817426052 * x);
        y += cosh(0.8854560256532099 * x);
        y += cosh(0.7485107481711011 * x);
        y += cosh(0.5680647467311558 * x);
        y += cosh(0.3546048870425356 * x);
        y += cosh(0.120536680255323 * x);
        y *= 2;
        y += cosh(x);
        y /= 13;
        return y;
    }

private:
    double mBeta = 0.0;
    double mInverseBesselBeta = 1.0;
};

} /* namespace RESAMPLER_OUTER_NAMESPACE::resampler */

#endif //RESAMPLER_KAISER_WINDOW_H
