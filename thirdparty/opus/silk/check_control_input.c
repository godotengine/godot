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

#include "main.h"
#include "control.h"
#include "errors.h"

/* Check encoder control struct */
opus_int check_control_input(
    silk_EncControlStruct        *encControl                    /* I    Control structure                           */
)
{
    celt_assert( encControl != NULL );

    if( ( ( encControl->API_sampleRate            !=  8000 ) &&
          ( encControl->API_sampleRate            != 12000 ) &&
          ( encControl->API_sampleRate            != 16000 ) &&
          ( encControl->API_sampleRate            != 24000 ) &&
          ( encControl->API_sampleRate            != 32000 ) &&
          ( encControl->API_sampleRate            != 44100 ) &&
          ( encControl->API_sampleRate            != 48000 ) ) ||
        ( ( encControl->desiredInternalSampleRate !=  8000 ) &&
          ( encControl->desiredInternalSampleRate != 12000 ) &&
          ( encControl->desiredInternalSampleRate != 16000 ) ) ||
        ( ( encControl->maxInternalSampleRate     !=  8000 ) &&
          ( encControl->maxInternalSampleRate     != 12000 ) &&
          ( encControl->maxInternalSampleRate     != 16000 ) ) ||
        ( ( encControl->minInternalSampleRate     !=  8000 ) &&
          ( encControl->minInternalSampleRate     != 12000 ) &&
          ( encControl->minInternalSampleRate     != 16000 ) ) ||
          ( encControl->minInternalSampleRate > encControl->desiredInternalSampleRate ) ||
          ( encControl->maxInternalSampleRate < encControl->desiredInternalSampleRate ) ||
          ( encControl->minInternalSampleRate > encControl->maxInternalSampleRate ) ) {
        celt_assert( 0 );
        return SILK_ENC_FS_NOT_SUPPORTED;
    }
    if( encControl->payloadSize_ms != 10 &&
        encControl->payloadSize_ms != 20 &&
        encControl->payloadSize_ms != 40 &&
        encControl->payloadSize_ms != 60 ) {
        celt_assert( 0 );
        return SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
    }
    if( encControl->packetLossPercentage < 0 || encControl->packetLossPercentage > 100 ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_LOSS_RATE;
    }
    if( encControl->useDTX < 0 || encControl->useDTX > 1 ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_DTX_SETTING;
    }
    if( encControl->useCBR < 0 || encControl->useCBR > 1 ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_CBR_SETTING;
    }
    if( encControl->useInBandFEC < 0 || encControl->useInBandFEC > 1 ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_INBAND_FEC_SETTING;
    }
    if( encControl->nChannelsAPI < 1 || encControl->nChannelsAPI > ENCODER_NUM_CHANNELS ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if( encControl->nChannelsInternal < 1 || encControl->nChannelsInternal > ENCODER_NUM_CHANNELS ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if( encControl->nChannelsInternal > encControl->nChannelsAPI ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if( encControl->complexity < 0 || encControl->complexity > 10 ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_COMPLEXITY_SETTING;
    }

    return SILK_NO_ERROR;
}
