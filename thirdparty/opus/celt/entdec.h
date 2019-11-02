/* Copyright (c) 2001-2011 Timothy B. Terriberry
   Copyright (c) 2008-2009 Xiph.Org Foundation */
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

#if !defined(_entdec_H)
# define _entdec_H (1)
# include <limits.h>
# include "entcode.h"

/*Initializes the decoder.
  _buf: The input buffer to use.
  Return: 0 on success, or a negative value on error.*/
void ec_dec_init(ec_dec *_this,unsigned char *_buf,opus_uint32 _storage);

/*Calculates the cumulative frequency for the next symbol.
  This can then be fed into the probability model to determine what that
   symbol is, and the additional frequency information required to advance to
   the next symbol.
  This function cannot be called more than once without a corresponding call to
   ec_dec_update(), or decoding will not proceed correctly.
  _ft: The total frequency of the symbols in the alphabet the next symbol was
        encoded with.
  Return: A cumulative frequency representing the encoded symbol.
          If the cumulative frequency of all the symbols before the one that
           was encoded was fl, and the cumulative frequency of all the symbols
           up to and including the one encoded is fh, then the returned value
           will fall in the range [fl,fh).*/
unsigned ec_decode(ec_dec *_this,unsigned _ft);

/*Equivalent to ec_decode() with _ft==1<<_bits.*/
unsigned ec_decode_bin(ec_dec *_this,unsigned _bits);

/*Advance the decoder past the next symbol using the frequency information the
   symbol was encoded with.
  Exactly one call to ec_decode() must have been made so that all necessary
   intermediate calculations are performed.
  _fl:  The cumulative frequency of all symbols that come before the symbol
         decoded.
  _fh:  The cumulative frequency of all symbols up to and including the symbol
         decoded.
        Together with _fl, this defines the range [_fl,_fh) in which the value
         returned above must fall.
  _ft:  The total frequency of the symbols in the alphabet the symbol decoded
         was encoded in.
        This must be the same as passed to the preceding call to ec_decode().*/
void ec_dec_update(ec_dec *_this,unsigned _fl,unsigned _fh,unsigned _ft);

/* Decode a bit that has a 1/(1<<_logp) probability of being a one */
int ec_dec_bit_logp(ec_dec *_this,unsigned _logp);

/*Decodes a symbol given an "inverse" CDF table.
  No call to ec_dec_update() is necessary after this call.
  _icdf: The "inverse" CDF, such that symbol s falls in the range
          [s>0?ft-_icdf[s-1]:0,ft-_icdf[s]), where ft=1<<_ftb.
         The values must be monotonically non-increasing, and the last value
          must be 0.
  _ftb: The number of bits of precision in the cumulative distribution.
  Return: The decoded symbol s.*/
int ec_dec_icdf(ec_dec *_this,const unsigned char *_icdf,unsigned _ftb);

/*Extracts a raw unsigned integer with a non-power-of-2 range from the stream.
  The bits must have been encoded with ec_enc_uint().
  No call to ec_dec_update() is necessary after this call.
  _ft: The number of integers that can be decoded (one more than the max).
       This must be at least 2, and no more than 2**32-1.
  Return: The decoded bits.*/
opus_uint32 ec_dec_uint(ec_dec *_this,opus_uint32 _ft);

/*Extracts a sequence of raw bits from the stream.
  The bits must have been encoded with ec_enc_bits().
  No call to ec_dec_update() is necessary after this call.
  _ftb: The number of bits to extract.
        This must be between 0 and 25, inclusive.
  Return: The decoded bits.*/
opus_uint32 ec_dec_bits(ec_dec *_this,unsigned _ftb);

#endif
