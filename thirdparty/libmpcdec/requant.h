/*
  Copyright (c) 2005-2009, The Musepack Development Team
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.

  * Neither the name of the The Musepack Development Team nor the
  names of its contributors may be used to endorse or promote
  products derived from this software without specific prior
  written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/// \file requant.h
/// Requantization function definitions.
#ifndef _MPCDEC_REQUANT_H_
#define _MPCDEC_REQUANT_H_
#ifdef WIN32
#pragma once
#endif

#include <mpc/mpc_types.h>

#ifdef __cplusplus
extern "C" {
#endif


/* C O N S T A N T S */
extern const mpc_uint8_t      _mpc_Res_bit [18];     ///< Bits per sample for chosen quantizer
extern const MPC_SAMPLE_FORMAT _mpc_Cc    [1 + 18]; ///< Requantization coefficients
extern const mpc_int16_t       _mpc_Dc    [1 + 18]; ///< Requantization offset

#define Cc (_mpc_Cc + 1)
#define Dc (_mpc_Dc + 1)
#define Res_bit _mpc_Res_bit


#ifdef __cplusplus
}
#endif
#endif
