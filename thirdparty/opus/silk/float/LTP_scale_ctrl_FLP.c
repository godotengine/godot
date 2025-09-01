/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "main_FLP.h"

void silk_LTP_scale_ctrl_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    opus_int                        condCoding                          /* I    The type of conditional coding to use       */
)
{
    opus_int   round_loss;

    if( condCoding == CODE_INDEPENDENTLY ) {
        /* Only scale if first frame in packet */
        round_loss = psEnc->sCmn.PacketLoss_perc * psEnc->sCmn.nFramesPerPacket;
        if ( psEnc->sCmn.LBRR_flag ) {
            /* LBRR reduces the effective loss. In practice, it does not square the loss because
               losses aren't independent, but that still seems to work best. We also never go below 2%. */
            round_loss = 2 + silk_SMULBB( round_loss, round_loss) / 100;
        }
        psEnc->sCmn.indices.LTP_scaleIndex = silk_SMULBB( psEncCtrl->LTPredCodGain, round_loss ) > silk_log2lin( 2900 - psEnc->sCmn.SNR_dB_Q7 );
        psEnc->sCmn.indices.LTP_scaleIndex += silk_SMULBB( psEncCtrl->LTPredCodGain, round_loss ) > silk_log2lin( 3900 - psEnc->sCmn.SNR_dB_Q7 );
    } else {
        /* Default is minimum scaling */
        psEnc->sCmn.indices.LTP_scaleIndex = 0;
    }

    psEncCtrl->LTP_scale = (silk_float)silk_LTPScales_table_Q14[ psEnc->sCmn.indices.LTP_scaleIndex ] / 16384.0f;
}
