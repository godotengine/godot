/* Copyright (C) 2002 Jean-Marc Valin */
/**
   @file vbr.h
   @brief Variable Bit-Rate (VBR) related routines
*/
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
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


#ifndef VBR_H
#define VBR_H

#include "arch.h"

#define VBR_MEMORY_SIZE 5

extern const float vbr_nb_thresh[9][11];
extern const float vbr_hb_thresh[5][11];
extern const float vbr_uhb_thresh[2][11];

/** VBR state. */
typedef struct VBRState {
   float energy_alpha;
   float average_energy;
   float last_energy;
   float last_log_energy[VBR_MEMORY_SIZE];
   float accum_sum;
   float last_pitch_coef;
   float soft_pitch;
   float last_quality;
   float noise_level;
   float noise_accum;
   float noise_accum_count;
   int   consec_noise;
} VBRState;

void vbr_init(VBRState *vbr);

float vbr_analysis(VBRState *vbr, spx_word16_t *sig, int len, int pitch, float pitch_coef);

void vbr_destroy(VBRState *vbr);

#endif
