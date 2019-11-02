/* Copyright (c) 2017 Google Inc.
   Written by Andrew Allen */
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
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * @file mapping_matrix.h
 * @brief Opus reference implementation mapping matrix API
 */

#ifndef MAPPING_MATRIX_H
#define MAPPING_MATRIX_H

#include "opus_types.h"
#include "opus_projection.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MappingMatrix
{
    int rows; /* number of channels outputted from matrix. */
    int cols; /* number of channels inputted to matrix. */
    int gain; /* in dB. S7.8-format. */
    /* Matrix cell data goes here using col-wise ordering. */
} MappingMatrix;

opus_int32 mapping_matrix_get_size(int rows, int cols);

opus_int16 *mapping_matrix_get_data(const MappingMatrix *matrix);

void mapping_matrix_init(
    MappingMatrix * const matrix,
    int rows,
    int cols,
    int gain,
    const opus_int16 *data,
    opus_int32 data_size
);

#ifndef DISABLE_FLOAT_API
void mapping_matrix_multiply_channel_in_float(
    const MappingMatrix *matrix,
    const float *input,
    int input_rows,
    opus_val16 *output,
    int output_row,
    int output_rows,
    int frame_size
);

void mapping_matrix_multiply_channel_out_float(
    const MappingMatrix *matrix,
    const opus_val16 *input,
    int input_row,
    int input_rows,
    float *output,
    int output_rows,
    int frame_size
);
#endif /* DISABLE_FLOAT_API */

void mapping_matrix_multiply_channel_in_short(
    const MappingMatrix *matrix,
    const opus_int16 *input,
    int input_rows,
    opus_val16 *output,
    int output_row,
    int output_rows,
    int frame_size
);

void mapping_matrix_multiply_channel_out_short(
    const MappingMatrix *matrix,
    const opus_val16 *input,
    int input_row,
    int input_rows,
    opus_int16 *output,
    int output_rows,
    int frame_size
);

/* Pre-computed mixing and demixing matrices for 1st to 3rd-order ambisonics.
 *   foa: first-order ambisonics
 *   soa: second-order ambisonics
 *   toa: third-order ambisonics
 */
extern const MappingMatrix mapping_matrix_foa_mixing;
extern const opus_int16 mapping_matrix_foa_mixing_data[36];

extern const MappingMatrix mapping_matrix_soa_mixing;
extern const opus_int16 mapping_matrix_soa_mixing_data[121];

extern const MappingMatrix mapping_matrix_toa_mixing;
extern const opus_int16 mapping_matrix_toa_mixing_data[324];

extern const MappingMatrix mapping_matrix_foa_demixing;
extern const opus_int16 mapping_matrix_foa_demixing_data[36];

extern const MappingMatrix mapping_matrix_soa_demixing;
extern const opus_int16 mapping_matrix_soa_demixing_data[121];

extern const MappingMatrix mapping_matrix_toa_demixing;
extern const opus_int16 mapping_matrix_toa_demixing_data[324];

#ifdef __cplusplus
}
#endif

#endif /* MAPPING_MATRIX_H */
