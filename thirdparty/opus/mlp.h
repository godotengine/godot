/* Copyright (c) 2017 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef MLP_H_
#define MLP_H_

#include "opus_types.h"

#define WEIGHTS_SCALE (1.f/128)

#define MAX_NEURONS 32

typedef struct {
  const opus_int8 *bias;
  const opus_int8 *input_weights;
  int nb_inputs;
  int nb_neurons;
  int sigmoid;
} AnalysisDenseLayer;

typedef struct {
  const opus_int8 *bias;
  const opus_int8 *input_weights;
  const opus_int8 *recurrent_weights;
  int nb_inputs;
  int nb_neurons;
} AnalysisGRULayer;

extern const AnalysisDenseLayer layer0;
extern const AnalysisGRULayer layer1;
extern const AnalysisDenseLayer layer2;

void analysis_compute_dense(const AnalysisDenseLayer *layer, float *output, const float *input);

void analysis_compute_gru(const AnalysisGRULayer *gru, float *state, const float *input);

#endif /* MLP_H_ */
