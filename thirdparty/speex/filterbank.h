/* Copyright (C) 2006 Jean-Marc Valin */
/**
   @file filterbank.h
   @brief Converting between psd and filterbank
 */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef FILTERBANK_H
#define FILTERBANK_H

#include "arch.h"

typedef struct {
   int *bank_left;
   int *bank_right;
   spx_word16_t *filter_left;
   spx_word16_t *filter_right;
#ifndef FIXED_POINT
   float *scaling;
#endif
   int nb_banks;
   int len;
} FilterBank;


FilterBank *filterbank_new(int banks, spx_word32_t sampling, int len, int type);

void filterbank_destroy(FilterBank *bank);

void filterbank_compute_bank32(FilterBank *bank, spx_word32_t *ps, spx_word32_t *mel);

void filterbank_compute_psd16(FilterBank *bank, spx_word16_t *mel, spx_word16_t *psd);

#ifndef FIXED_POINT
void filterbank_compute_bank(FilterBank *bank, float *psd, float *mel);
void filterbank_compute_psd(FilterBank *bank, float *mel, float *psd);
#endif


#endif
