/*---------------------------------------------------------------------------*\
Original Copyright
	FILE........: AK2LSPD.H
	TYPE........: Turbo C header file
	COMPANY.....: Voicetronix
	AUTHOR......: James Whitehall
	DATE CREATED: 21/11/95

Modified by Jean-Marc Valin

    This file contains functions for converting Linear Prediction
    Coefficients (LPC) to Line Spectral Pair (LSP) and back. Note that the
    LSP coefficients are not in radians format but in the x domain of the
    unit circle.

\*---------------------------------------------------------------------------*/
/**
   @file lsp.h
   @brief Line Spectral Pair (LSP) functions.
*/
/* Speex License:

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

#ifndef __AK2LSPD__
#define __AK2LSPD__

#include "arch.h"

int lpc_to_lsp (spx_coef_t *a, int lpcrdr, spx_lsp_t *freq, int nb, spx_word16_t delta, char *stack);
void lsp_to_lpc(spx_lsp_t *freq, spx_coef_t *ak, int lpcrdr, char *stack);

/*Added by JMV*/
void lsp_enforce_margin(spx_lsp_t *lsp, int len, spx_word16_t margin);

void lsp_interpolate(spx_lsp_t *old_lsp, spx_lsp_t *new_lsp, spx_lsp_t *interp_lsp, int len, int subframe, int nb_subframes);

#endif	/* __AK2LSPD__ */
