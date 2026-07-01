/* Copyright (c) 2007 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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

#ifndef LAPLACE_H
#define LAPLACE_H

#include "entenc.h"
#include "entdec.h"

/** Encode a value that is assumed to be the realisation of a
    Laplace-distributed random process
 @param enc Entropy encoder state
 @param value Value to encode
 @param fs Probability of 0, multiplied by 32768
 @param decay Probability of the value +/- 1, multiplied by 16384
*/
void ec_laplace_encode(ec_enc *enc, int *value, unsigned fs, int decay);

/** Decode a value that is assumed to be the realisation of a
    Laplace-distributed random process
 @param dec Entropy decoder state
 @param fs Probability of 0, multiplied by 32768
 @param decay Probability of the value +/- 1, multiplied by 16384
 @return Value decoded
 */
int ec_laplace_decode(ec_dec *dec, unsigned fs, int decay);


int ec_laplace_decode_p0(ec_dec *dec, opus_uint16 p0, opus_uint16 decay);
void ec_laplace_encode_p0(ec_enc *enc, int value, opus_uint16 p0, opus_uint16 decay);

#endif
