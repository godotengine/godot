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
#include "define.h"
#include "API.h"
#include "control.h"
#include "typedef.h"
#include "stack_alloc.h"
#include "structs.h"
#include "tuning_parameters.h"
#ifdef FIXED_POINT
#include "main_FIX.h"
#else
#include "main_FLP.h"
#endif

#ifdef ENABLE_DRED
#include "dred_encoder.h"
#endif

/***************************************/
/* Read control structure from encoder */
/***************************************/
static opus_int silk_QueryEncoder(                      /* O    Returns error code                              */
    const void                      *encState,          /* I    State                                           */
    silk_EncControlStruct           *encStatus          /* O    Encoder Status                                  */
);

/****************************************/
/* Encoder functions                    */
/****************************************/

opus_int silk_Get_Encoder_Size(                         /* O    Returns error code                              */
    opus_int                        *encSizeBytes       /* O    Number of bytes in SILK encoder state           */
)
{
    opus_int ret = SILK_NO_ERROR;

    *encSizeBytes = sizeof( silk_encoder );

    return ret;
}

/*************************/
/* Init or Reset encoder */
/*************************/
opus_int silk_InitEncoder(                              /* O    Returns error code                              */
    void                            *encState,          /* I/O  State                                           */
    int                              arch,              /* I    Run-time architecture                           */
    silk_EncControlStruct           *encStatus          /* O    Encoder Status                                  */
)
{
    silk_encoder *psEnc;
    opus_int n, ret = SILK_NO_ERROR;

    psEnc = (silk_encoder *)encState;

    /* Reset encoder */
    silk_memset( psEnc, 0, sizeof( silk_encoder ) );
    for( n = 0; n < ENCODER_NUM_CHANNELS; n++ ) {
        if( ret += silk_init_encoder( &psEnc->state_Fxx[ n ], arch ) ) {
            celt_assert( 0 );
        }
    }

    psEnc->nChannelsAPI = 1;
    psEnc->nChannelsInternal = 1;

    /* Read control structure */
    if( ret += silk_QueryEncoder( encState, encStatus ) ) {
        celt_assert( 0 );
    }

    return ret;
}

/***************************************/
/* Read control structure from encoder */
/***************************************/
static opus_int silk_QueryEncoder(                      /* O    Returns error code                              */
    const void                      *encState,          /* I    State                                           */
    silk_EncControlStruct           *encStatus          /* O    Encoder Status                                  */
)
{
    opus_int ret = SILK_NO_ERROR;
    silk_encoder_state_Fxx *state_Fxx;
    silk_encoder *psEnc = (silk_encoder *)encState;

    state_Fxx = psEnc->state_Fxx;

    encStatus->nChannelsAPI              = psEnc->nChannelsAPI;
    encStatus->nChannelsInternal         = psEnc->nChannelsInternal;
    encStatus->API_sampleRate            = state_Fxx[ 0 ].sCmn.API_fs_Hz;
    encStatus->maxInternalSampleRate     = state_Fxx[ 0 ].sCmn.maxInternal_fs_Hz;
    encStatus->minInternalSampleRate     = state_Fxx[ 0 ].sCmn.minInternal_fs_Hz;
    encStatus->desiredInternalSampleRate = state_Fxx[ 0 ].sCmn.desiredInternal_fs_Hz;
    encStatus->payloadSize_ms            = state_Fxx[ 0 ].sCmn.PacketSize_ms;
    encStatus->bitRate                   = state_Fxx[ 0 ].sCmn.TargetRate_bps;
    encStatus->packetLossPercentage      = state_Fxx[ 0 ].sCmn.PacketLoss_perc;
    encStatus->complexity                = state_Fxx[ 0 ].sCmn.Complexity;
    encStatus->useInBandFEC              = state_Fxx[ 0 ].sCmn.useInBandFEC;
    encStatus->useDTX                    = state_Fxx[ 0 ].sCmn.useDTX;
    encStatus->useCBR                    = state_Fxx[ 0 ].sCmn.useCBR;
    encStatus->internalSampleRate        = silk_SMULBB( state_Fxx[ 0 ].sCmn.fs_kHz, 1000 );
    encStatus->allowBandwidthSwitch      = state_Fxx[ 0 ].sCmn.allow_bandwidth_switch;
    encStatus->inWBmodeWithoutVariableLP = state_Fxx[ 0 ].sCmn.fs_kHz == 16 && state_Fxx[ 0 ].sCmn.sLP.mode == 0;

    return ret;
}


/**************************/
/* Encode frame with Silk */
/**************************/
/* Note: if prefillFlag is set, the input must contain 10 ms of audio, irrespective of what                     */
/* encControl->payloadSize_ms is set to                                                                         */
opus_int silk_Encode(                                   /* O    Returns error code                              */
    void                            *encState,          /* I/O  State                                           */
    silk_EncControlStruct           *encControl,        /* I    Control status                                  */
    const opus_int16                *samplesIn,         /* I    Speech sample input vector                      */
    opus_int                        nSamplesIn,         /* I    Number of samples in input vector               */
    ec_enc                          *psRangeEnc,        /* I/O  Compressor data structure                       */
    opus_int32                      *nBytesOut,         /* I/O  Number of bytes in payload (input: Max bytes)   */
    const opus_int                  prefillFlag,        /* I    Flag to indicate prefilling buffers no coding   */
    opus_int                        activity            /* I    Decision of Opus voice activity detector        */
)
{
    opus_int   n, i, nBits, flags, tmp_payloadSize_ms = 0, tmp_complexity = 0, ret = 0;
    opus_int   nSamplesToBuffer, nSamplesToBufferMax, nBlocksOf10ms;
    opus_int   nSamplesFromInput = 0, nSamplesFromInputMax;
    opus_int   speech_act_thr_for_switch_Q8;
    opus_int32 TargetRate_bps, MStargetRates_bps[ 2 ], channelRate_bps, LBRR_symbol, sum;
    silk_encoder *psEnc = ( silk_encoder * )encState;
    VARDECL( opus_int16, buf );
    opus_int transition, curr_block, tot_blocks;
    SAVE_STACK;

    if (encControl->reducedDependency)
    {
       psEnc->state_Fxx[0].sCmn.first_frame_after_reset = 1;
       psEnc->state_Fxx[1].sCmn.first_frame_after_reset = 1;
    }
    psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded = psEnc->state_Fxx[ 1 ].sCmn.nFramesEncoded = 0;

    /* Check values in encoder control structure */
    if( ( ret = check_control_input( encControl ) ) != 0 ) {
        celt_assert( 0 );
        RESTORE_STACK;
        return ret;
    }

    encControl->switchReady = 0;

    if( encControl->nChannelsInternal > psEnc->nChannelsInternal ) {
        /* Mono -> Stereo transition: init state of second channel and stereo state */
        ret += silk_init_encoder( &psEnc->state_Fxx[ 1 ], psEnc->state_Fxx[ 0 ].sCmn.arch );
        silk_memset( psEnc->sStereo.pred_prev_Q13, 0, sizeof( psEnc->sStereo.pred_prev_Q13 ) );
        silk_memset( psEnc->sStereo.sSide, 0, sizeof( psEnc->sStereo.sSide ) );
        psEnc->sStereo.mid_side_amp_Q0[ 0 ] = 0;
        psEnc->sStereo.mid_side_amp_Q0[ 1 ] = 1;
        psEnc->sStereo.mid_side_amp_Q0[ 2 ] = 0;
        psEnc->sStereo.mid_side_amp_Q0[ 3 ] = 1;
        psEnc->sStereo.width_prev_Q14 = 0;
        psEnc->sStereo.smth_width_Q14 = SILK_FIX_CONST( 1, 14 );
        if( psEnc->nChannelsAPI == 2 ) {
            silk_memcpy( &psEnc->state_Fxx[ 1 ].sCmn.resampler_state, &psEnc->state_Fxx[ 0 ].sCmn.resampler_state, sizeof( silk_resampler_state_struct ) );
            silk_memcpy( &psEnc->state_Fxx[ 1 ].sCmn.In_HP_State,     &psEnc->state_Fxx[ 0 ].sCmn.In_HP_State,     sizeof( psEnc->state_Fxx[ 1 ].sCmn.In_HP_State ) );
        }
    }

    transition = (encControl->payloadSize_ms != psEnc->state_Fxx[ 0 ].sCmn.PacketSize_ms) || (psEnc->nChannelsInternal != encControl->nChannelsInternal);

    psEnc->nChannelsAPI = encControl->nChannelsAPI;
    psEnc->nChannelsInternal = encControl->nChannelsInternal;

    nBlocksOf10ms = silk_DIV32( 100 * nSamplesIn, encControl->API_sampleRate );
    tot_blocks = ( nBlocksOf10ms > 1 ) ? nBlocksOf10ms >> 1 : 1;
    curr_block = 0;
    if( prefillFlag ) {
        silk_LP_state save_LP;
        /* Only accept input length of 10 ms */
        if( nBlocksOf10ms != 1 ) {
            celt_assert( 0 );
            RESTORE_STACK;
            return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
        }
        if ( prefillFlag == 2 ) {
            save_LP = psEnc->state_Fxx[ 0 ].sCmn.sLP;
            /* Save the sampling rate so the bandwidth switching code can keep handling transitions. */
            save_LP.saved_fs_kHz = psEnc->state_Fxx[ 0 ].sCmn.fs_kHz;
        }
        /* Reset Encoder */
        for( n = 0; n < encControl->nChannelsInternal; n++ ) {
            ret = silk_init_encoder( &psEnc->state_Fxx[ n ], psEnc->state_Fxx[ n ].sCmn.arch );
            /* Restore the variable LP state. */
            if ( prefillFlag == 2 ) {
                psEnc->state_Fxx[ n ].sCmn.sLP = save_LP;
            }
            celt_assert( !ret );
        }
        tmp_payloadSize_ms = encControl->payloadSize_ms;
        encControl->payloadSize_ms = 10;
        tmp_complexity = encControl->complexity;
        encControl->complexity = 0;
        for( n = 0; n < encControl->nChannelsInternal; n++ ) {
            psEnc->state_Fxx[ n ].sCmn.controlled_since_last_payload = 0;
            psEnc->state_Fxx[ n ].sCmn.prefillFlag = 1;
        }
    } else {
        /* Only accept input lengths that are a multiple of 10 ms */
        if( nBlocksOf10ms * encControl->API_sampleRate != 100 * nSamplesIn || nSamplesIn < 0 ) {
            celt_assert( 0 );
            RESTORE_STACK;
            return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
        }
        /* Make sure no more than one packet can be produced */
        if( 1000 * (opus_int32)nSamplesIn > encControl->payloadSize_ms * encControl->API_sampleRate ) {
            celt_assert( 0 );
            RESTORE_STACK;
            return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
        }
    }

    for( n = 0; n < encControl->nChannelsInternal; n++ ) {
        /* Force the side channel to the same rate as the mid */
        opus_int force_fs_kHz = (n==1) ? psEnc->state_Fxx[0].sCmn.fs_kHz : 0;
        if( ( ret = silk_control_encoder( &psEnc->state_Fxx[ n ], encControl, psEnc->allowBandwidthSwitch, n, force_fs_kHz ) ) != 0 ) {
            silk_assert( 0 );
            RESTORE_STACK;
            return ret;
        }
        if( psEnc->state_Fxx[n].sCmn.first_frame_after_reset || transition ) {
            for( i = 0; i < psEnc->state_Fxx[ 0 ].sCmn.nFramesPerPacket; i++ ) {
                psEnc->state_Fxx[ n ].sCmn.LBRR_flags[ i ] = 0;
            }
        }
        psEnc->state_Fxx[ n ].sCmn.inDTX = psEnc->state_Fxx[ n ].sCmn.useDTX;
    }
    celt_assert( encControl->nChannelsInternal == 1 || psEnc->state_Fxx[ 0 ].sCmn.fs_kHz == psEnc->state_Fxx[ 1 ].sCmn.fs_kHz );

    /* Input buffering/resampling and encoding */
    nSamplesToBufferMax =
        10 * nBlocksOf10ms * psEnc->state_Fxx[ 0 ].sCmn.fs_kHz;
    nSamplesFromInputMax =
        silk_DIV32_16( nSamplesToBufferMax *
                           psEnc->state_Fxx[ 0 ].sCmn.API_fs_Hz,
                       psEnc->state_Fxx[ 0 ].sCmn.fs_kHz * 1000 );
    ALLOC( buf, nSamplesFromInputMax, opus_int16 );
    while( 1 ) {
        int curr_nBitsUsedLBRR = 0;
        nSamplesToBuffer  = psEnc->state_Fxx[ 0 ].sCmn.frame_length - psEnc->state_Fxx[ 0 ].sCmn.inputBufIx;
        nSamplesToBuffer  = silk_min( nSamplesToBuffer, nSamplesToBufferMax );
        nSamplesFromInput = silk_DIV32_16( nSamplesToBuffer * psEnc->state_Fxx[ 0 ].sCmn.API_fs_Hz, psEnc->state_Fxx[ 0 ].sCmn.fs_kHz * 1000 );
        /* Resample and write to buffer */
        if( encControl->nChannelsAPI == 2 && encControl->nChannelsInternal == 2 ) {
            opus_int id = psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded;
            for( n = 0; n < nSamplesFromInput; n++ ) {
                buf[ n ] = samplesIn[ 2 * n ];
            }
            /* Making sure to start both resamplers from the same state when switching from mono to stereo */
            if( psEnc->nPrevChannelsInternal == 1 && id==0 ) {
               silk_memcpy( &psEnc->state_Fxx[ 1 ].sCmn.resampler_state, &psEnc->state_Fxx[ 0 ].sCmn.resampler_state, sizeof(psEnc->state_Fxx[ 1 ].sCmn.resampler_state));
            }

            ret += silk_resampler( &psEnc->state_Fxx[ 0 ].sCmn.resampler_state,
                &psEnc->state_Fxx[ 0 ].sCmn.inputBuf[ psEnc->state_Fxx[ 0 ].sCmn.inputBufIx + 2 ], buf, nSamplesFromInput );
            psEnc->state_Fxx[ 0 ].sCmn.inputBufIx += nSamplesToBuffer;

            nSamplesToBuffer  = psEnc->state_Fxx[ 1 ].sCmn.frame_length - psEnc->state_Fxx[ 1 ].sCmn.inputBufIx;
            nSamplesToBuffer  = silk_min( nSamplesToBuffer, 10 * nBlocksOf10ms * psEnc->state_Fxx[ 1 ].sCmn.fs_kHz );
            for( n = 0; n < nSamplesFromInput; n++ ) {
                buf[ n ] = samplesIn[ 2 * n + 1 ];
            }
            ret += silk_resampler( &psEnc->state_Fxx[ 1 ].sCmn.resampler_state,
                &psEnc->state_Fxx[ 1 ].sCmn.inputBuf[ psEnc->state_Fxx[ 1 ].sCmn.inputBufIx + 2 ], buf, nSamplesFromInput );

            psEnc->state_Fxx[ 1 ].sCmn.inputBufIx += nSamplesToBuffer;
        } else if( encControl->nChannelsAPI == 2 && encControl->nChannelsInternal == 1 ) {
            /* Combine left and right channels before resampling */
            for( n = 0; n < nSamplesFromInput; n++ ) {
                sum = samplesIn[ 2 * n ] + samplesIn[ 2 * n + 1 ];
                buf[ n ] = (opus_int16)silk_RSHIFT_ROUND( sum,  1 );
            }
            ret += silk_resampler( &psEnc->state_Fxx[ 0 ].sCmn.resampler_state,
                &psEnc->state_Fxx[ 0 ].sCmn.inputBuf[ psEnc->state_Fxx[ 0 ].sCmn.inputBufIx + 2 ], buf, nSamplesFromInput );
            /* On the first mono frame, average the results for the two resampler states  */
            if( psEnc->nPrevChannelsInternal == 2 && psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded == 0 ) {
               ret += silk_resampler( &psEnc->state_Fxx[ 1 ].sCmn.resampler_state,
                   &psEnc->state_Fxx[ 1 ].sCmn.inputBuf[ psEnc->state_Fxx[ 1 ].sCmn.inputBufIx + 2 ], buf, nSamplesFromInput );
               for( n = 0; n < psEnc->state_Fxx[ 0 ].sCmn.frame_length; n++ ) {
                  psEnc->state_Fxx[ 0 ].sCmn.inputBuf[ psEnc->state_Fxx[ 0 ].sCmn.inputBufIx+n+2 ] =
                        silk_RSHIFT(psEnc->state_Fxx[ 0 ].sCmn.inputBuf[ psEnc->state_Fxx[ 0 ].sCmn.inputBufIx+n+2 ]
                                  + psEnc->state_Fxx[ 1 ].sCmn.inputBuf[ psEnc->state_Fxx[ 1 ].sCmn.inputBufIx+n+2 ], 1);
               }
            }
            psEnc->state_Fxx[ 0 ].sCmn.inputBufIx += nSamplesToBuffer;
        } else {
            celt_assert( encControl->nChannelsAPI == 1 && encControl->nChannelsInternal == 1 );
            silk_memcpy(buf, samplesIn, nSamplesFromInput*sizeof(opus_int16));
            ret += silk_resampler( &psEnc->state_Fxx[ 0 ].sCmn.resampler_state,
                &psEnc->state_Fxx[ 0 ].sCmn.inputBuf[ psEnc->state_Fxx[ 0 ].sCmn.inputBufIx + 2 ], buf, nSamplesFromInput );
            psEnc->state_Fxx[ 0 ].sCmn.inputBufIx += nSamplesToBuffer;
        }

        samplesIn  += nSamplesFromInput * encControl->nChannelsAPI;
        nSamplesIn -= nSamplesFromInput;

        /* Default */
        psEnc->allowBandwidthSwitch = 0;

        /* Silk encoder */
        if( psEnc->state_Fxx[ 0 ].sCmn.inputBufIx >= psEnc->state_Fxx[ 0 ].sCmn.frame_length ) {
            /* Enough data in input buffer, so encode */
            celt_assert( psEnc->state_Fxx[ 0 ].sCmn.inputBufIx == psEnc->state_Fxx[ 0 ].sCmn.frame_length );
            celt_assert( encControl->nChannelsInternal == 1 || psEnc->state_Fxx[ 1 ].sCmn.inputBufIx == psEnc->state_Fxx[ 1 ].sCmn.frame_length );

            /* Deal with LBRR data */
            if( psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded == 0 && !prefillFlag ) {
                /* Create space at start of payload for VAD and FEC flags */
                opus_uint8 iCDF[ 2 ] = { 0, 0 };
                iCDF[ 0 ] = 256 - silk_RSHIFT( 256, ( psEnc->state_Fxx[ 0 ].sCmn.nFramesPerPacket + 1 ) * encControl->nChannelsInternal );
                ec_enc_icdf( psRangeEnc, 0, iCDF, 8 );
                curr_nBitsUsedLBRR = ec_tell( psRangeEnc );

                /* Encode any LBRR data from previous packet */
                /* Encode LBRR flags */
                for( n = 0; n < encControl->nChannelsInternal; n++ ) {
                    LBRR_symbol = 0;
                    for( i = 0; i < psEnc->state_Fxx[ n ].sCmn.nFramesPerPacket; i++ ) {
                        LBRR_symbol |= silk_LSHIFT( psEnc->state_Fxx[ n ].sCmn.LBRR_flags[ i ], i );
                    }
                    psEnc->state_Fxx[ n ].sCmn.LBRR_flag = LBRR_symbol > 0 ? 1 : 0;
                    if( LBRR_symbol && psEnc->state_Fxx[ n ].sCmn.nFramesPerPacket > 1 ) {
                        ec_enc_icdf( psRangeEnc, LBRR_symbol - 1, silk_LBRR_flags_iCDF_ptr[ psEnc->state_Fxx[ n ].sCmn.nFramesPerPacket - 2 ], 8 );
                    }
                }

                /* Code LBRR indices and excitation signals */
                for( i = 0; i < psEnc->state_Fxx[ 0 ].sCmn.nFramesPerPacket; i++ ) {
                    for( n = 0; n < encControl->nChannelsInternal; n++ ) {
                        if( psEnc->state_Fxx[ n ].sCmn.LBRR_flags[ i ] ) {
                            opus_int condCoding;

                            if( encControl->nChannelsInternal == 2 && n == 0 ) {
                                silk_stereo_encode_pred( psRangeEnc, psEnc->sStereo.predIx[ i ] );
                                /* For LBRR data there's no need to code the mid-only flag if the side-channel LBRR flag is set */
                                if( psEnc->state_Fxx[ 1 ].sCmn.LBRR_flags[ i ] == 0 ) {
                                    silk_stereo_encode_mid_only( psRangeEnc, psEnc->sStereo.mid_only_flags[ i ] );
                                }
                            }
                            /* Use conditional coding if previous frame available */
                            if( i > 0 && psEnc->state_Fxx[ n ].sCmn.LBRR_flags[ i - 1 ] ) {
                                condCoding = CODE_CONDITIONALLY;
                            } else {
                                condCoding = CODE_INDEPENDENTLY;
                            }
                            silk_encode_indices( &psEnc->state_Fxx[ n ].sCmn, psRangeEnc, i, 1, condCoding );
                            silk_encode_pulses( psRangeEnc, psEnc->state_Fxx[ n ].sCmn.indices_LBRR[i].signalType, psEnc->state_Fxx[ n ].sCmn.indices_LBRR[i].quantOffsetType,
                                psEnc->state_Fxx[ n ].sCmn.pulses_LBRR[ i ], psEnc->state_Fxx[ n ].sCmn.frame_length );
                        }
                    }
                }

                /* Reset LBRR flags */
                for( n = 0; n < encControl->nChannelsInternal; n++ ) {
                    silk_memset( psEnc->state_Fxx[ n ].sCmn.LBRR_flags, 0, sizeof( psEnc->state_Fxx[ n ].sCmn.LBRR_flags ) );
                }
                curr_nBitsUsedLBRR = ec_tell( psRangeEnc ) - curr_nBitsUsedLBRR;
            }

            silk_HP_variable_cutoff( psEnc->state_Fxx );

            /* Total target bits for packet */
            nBits = silk_DIV32_16( silk_MUL( encControl->bitRate, encControl->payloadSize_ms ), 1000 );
            /* Subtract bits used for LBRR */
            if( !prefillFlag ) {
                /* psEnc->nBitsUsedLBRR is an exponential moving average of the LBRR usage,
                   except that for the first LBRR frame it does no averaging and for the first
                   frame after after LBRR, it goes back to zero immediately. */
                if ( curr_nBitsUsedLBRR < 10 ) {
                    psEnc->nBitsUsedLBRR = 0;
                } else if ( psEnc->nBitsUsedLBRR < 10) {
                    psEnc->nBitsUsedLBRR = curr_nBitsUsedLBRR;
                } else {
                    psEnc->nBitsUsedLBRR = ( psEnc->nBitsUsedLBRR + curr_nBitsUsedLBRR ) / 2;
                }
                nBits -= psEnc->nBitsUsedLBRR;
            }
            /* Divide by number of uncoded frames left in packet */
            nBits = silk_DIV32_16( nBits, psEnc->state_Fxx[ 0 ].sCmn.nFramesPerPacket );
            /* Convert to bits/second */
            if( encControl->payloadSize_ms == 10 ) {
                TargetRate_bps = silk_SMULBB( nBits, 100 );
            } else {
                TargetRate_bps = silk_SMULBB( nBits, 50 );
            }
            /* Subtract fraction of bits in excess of target in previous frames and packets */
            TargetRate_bps -= silk_DIV32_16( silk_MUL( psEnc->nBitsExceeded, 1000 ), BITRESERVOIR_DECAY_TIME_MS );
            if( !prefillFlag && psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded > 0 ) {
                /* Compare actual vs target bits so far in this packet */
                opus_int32 bitsBalance = ec_tell( psRangeEnc ) - psEnc->nBitsUsedLBRR - nBits * psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded;
                TargetRate_bps -= silk_DIV32_16( silk_MUL( bitsBalance, 1000 ), BITRESERVOIR_DECAY_TIME_MS );
            }
            /* Never exceed input bitrate */
            TargetRate_bps = silk_LIMIT( TargetRate_bps, encControl->bitRate, 5000 );

            /* Convert Left/Right to Mid/Side */
            if( encControl->nChannelsInternal == 2 ) {
                silk_stereo_LR_to_MS( &psEnc->sStereo, &psEnc->state_Fxx[ 0 ].sCmn.inputBuf[ 2 ], &psEnc->state_Fxx[ 1 ].sCmn.inputBuf[ 2 ],
                    psEnc->sStereo.predIx[ psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded ], &psEnc->sStereo.mid_only_flags[ psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded ],
                    MStargetRates_bps, TargetRate_bps, psEnc->state_Fxx[ 0 ].sCmn.speech_activity_Q8, encControl->toMono,
                    psEnc->state_Fxx[ 0 ].sCmn.fs_kHz, psEnc->state_Fxx[ 0 ].sCmn.frame_length );
                if( psEnc->sStereo.mid_only_flags[ psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded ] == 0 ) {
                    /* Reset side channel encoder memory for first frame with side coding */
                    if( psEnc->prev_decode_only_middle == 1 ) {
                        silk_memset( &psEnc->state_Fxx[ 1 ].sShape,               0, sizeof( psEnc->state_Fxx[ 1 ].sShape ) );
                        silk_memset( &psEnc->state_Fxx[ 1 ].sCmn.sNSQ,            0, sizeof( psEnc->state_Fxx[ 1 ].sCmn.sNSQ ) );
                        silk_memset( psEnc->state_Fxx[ 1 ].sCmn.prev_NLSFq_Q15,   0, sizeof( psEnc->state_Fxx[ 1 ].sCmn.prev_NLSFq_Q15 ) );
                        silk_memset( &psEnc->state_Fxx[ 1 ].sCmn.sLP.In_LP_State, 0, sizeof( psEnc->state_Fxx[ 1 ].sCmn.sLP.In_LP_State ) );
                        psEnc->state_Fxx[ 1 ].sCmn.prevLag                 = 100;
                        psEnc->state_Fxx[ 1 ].sCmn.sNSQ.lagPrev            = 100;
                        psEnc->state_Fxx[ 1 ].sShape.LastGainIndex         = 10;
                        psEnc->state_Fxx[ 1 ].sCmn.prevSignalType          = TYPE_NO_VOICE_ACTIVITY;
                        psEnc->state_Fxx[ 1 ].sCmn.sNSQ.prev_gain_Q16      = 65536;
                        psEnc->state_Fxx[ 1 ].sCmn.first_frame_after_reset = 1;
                    }
                    silk_encode_do_VAD_Fxx( &psEnc->state_Fxx[ 1 ], activity );
                } else {
                    psEnc->state_Fxx[ 1 ].sCmn.VAD_flags[ psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded ] = 0;
                }
                if( !prefillFlag ) {
                    silk_stereo_encode_pred( psRangeEnc, psEnc->sStereo.predIx[ psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded ] );
                    if( psEnc->state_Fxx[ 1 ].sCmn.VAD_flags[ psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded ] == 0 ) {
                        silk_stereo_encode_mid_only( psRangeEnc, psEnc->sStereo.mid_only_flags[ psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded ] );
                    }
                }
            } else {
                /* Buffering */
                silk_memcpy( psEnc->state_Fxx[ 0 ].sCmn.inputBuf, psEnc->sStereo.sMid, 2 * sizeof( opus_int16 ) );
                silk_memcpy( psEnc->sStereo.sMid, &psEnc->state_Fxx[ 0 ].sCmn.inputBuf[ psEnc->state_Fxx[ 0 ].sCmn.frame_length ], 2 * sizeof( opus_int16 ) );
            }
            silk_encode_do_VAD_Fxx( &psEnc->state_Fxx[ 0 ], activity );

            /* Encode */
            for( n = 0; n < encControl->nChannelsInternal; n++ ) {
                opus_int maxBits, useCBR;

                /* Handling rate constraints */
                maxBits = encControl->maxBits;
                if( tot_blocks == 2 && curr_block == 0 ) {
                    maxBits = maxBits * 3 / 5;
                } else if( tot_blocks == 3 ) {
                    if( curr_block == 0 ) {
                        maxBits = maxBits * 2 / 5;
                    } else if( curr_block == 1 ) {
                        maxBits = maxBits * 3 / 4;
                    }
                }
                useCBR = encControl->useCBR && curr_block == tot_blocks - 1;

                if( encControl->nChannelsInternal == 1 ) {
                    channelRate_bps = TargetRate_bps;
                } else {
                    channelRate_bps = MStargetRates_bps[ n ];
                    if( n == 0 && MStargetRates_bps[ 1 ] > 0 ) {
                        useCBR = 0;
                        /* Give mid up to 1/2 of the max bits for that frame */
                        maxBits -= encControl->maxBits / ( tot_blocks * 2 );
                    }
                }

                if( channelRate_bps > 0 ) {
                    opus_int condCoding;

                    silk_control_SNR( &psEnc->state_Fxx[ n ].sCmn, channelRate_bps );

                    /* Use independent coding if no previous frame available */
                    if( psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded - n <= 0 ) {
                        condCoding = CODE_INDEPENDENTLY;
                    } else if( n > 0 && psEnc->prev_decode_only_middle ) {
                        /* If we skipped a side frame in this packet, we don't
                           need LTP scaling; the LTP state is well-defined. */
                        condCoding = CODE_INDEPENDENTLY_NO_LTP_SCALING;
                    } else {
                        condCoding = CODE_CONDITIONALLY;
                    }
                    if( ( ret = silk_encode_frame_Fxx( &psEnc->state_Fxx[ n ], nBytesOut, psRangeEnc, condCoding, maxBits, useCBR ) ) != 0 ) {
                        silk_assert( 0 );
                    }
                }
                psEnc->state_Fxx[ n ].sCmn.controlled_since_last_payload = 0;
                psEnc->state_Fxx[ n ].sCmn.inputBufIx = 0;
                psEnc->state_Fxx[ n ].sCmn.nFramesEncoded++;
            }
            psEnc->prev_decode_only_middle = psEnc->sStereo.mid_only_flags[ psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded - 1 ];

            /* Insert VAD and FEC flags at beginning of bitstream */
            if( *nBytesOut > 0 && psEnc->state_Fxx[ 0 ].sCmn.nFramesEncoded == psEnc->state_Fxx[ 0 ].sCmn.nFramesPerPacket) {
                flags = 0;
                for( n = 0; n < encControl->nChannelsInternal; n++ ) {
                    for( i = 0; i < psEnc->state_Fxx[ n ].sCmn.nFramesPerPacket; i++ ) {
                        flags  = silk_LSHIFT( flags, 1 );
                        flags |= psEnc->state_Fxx[ n ].sCmn.VAD_flags[ i ];
                    }
                    flags  = silk_LSHIFT( flags, 1 );
                    flags |= psEnc->state_Fxx[ n ].sCmn.LBRR_flag;
                }
                if( !prefillFlag ) {
                    ec_enc_patch_initial_bits( psRangeEnc, flags, ( psEnc->state_Fxx[ 0 ].sCmn.nFramesPerPacket + 1 ) * encControl->nChannelsInternal );
                }

                /* Return zero bytes if all channels DTXed */
                if( psEnc->state_Fxx[ 0 ].sCmn.inDTX && ( encControl->nChannelsInternal == 1 || psEnc->state_Fxx[ 1 ].sCmn.inDTX ) ) {
                    *nBytesOut = 0;
                }

                psEnc->nBitsExceeded += *nBytesOut * 8;
                psEnc->nBitsExceeded -= silk_DIV32_16( silk_MUL( encControl->bitRate, encControl->payloadSize_ms ), 1000 );
                psEnc->nBitsExceeded  = silk_LIMIT( psEnc->nBitsExceeded, 0, 10000 );

                /* Update flag indicating if bandwidth switching is allowed */
                speech_act_thr_for_switch_Q8 = silk_SMLAWB( SILK_FIX_CONST( SPEECH_ACTIVITY_DTX_THRES, 8 ),
                    SILK_FIX_CONST( ( 1 - SPEECH_ACTIVITY_DTX_THRES ) / MAX_BANDWIDTH_SWITCH_DELAY_MS, 16 + 8 ), psEnc->timeSinceSwitchAllowed_ms );
                if( psEnc->state_Fxx[ 0 ].sCmn.speech_activity_Q8 < speech_act_thr_for_switch_Q8 ) {
                    psEnc->allowBandwidthSwitch = 1;
                    psEnc->timeSinceSwitchAllowed_ms = 0;
                } else {
                    psEnc->allowBandwidthSwitch = 0;
                    psEnc->timeSinceSwitchAllowed_ms += encControl->payloadSize_ms;
                }
            }

            if( nSamplesIn == 0 ) {
                break;
            }
        } else {
            break;
        }
        curr_block++;
    }

    psEnc->nPrevChannelsInternal = encControl->nChannelsInternal;

    encControl->allowBandwidthSwitch = psEnc->allowBandwidthSwitch;
    encControl->inWBmodeWithoutVariableLP = psEnc->state_Fxx[ 0 ].sCmn.fs_kHz == 16 && psEnc->state_Fxx[ 0 ].sCmn.sLP.mode == 0;
    encControl->internalSampleRate = silk_SMULBB( psEnc->state_Fxx[ 0 ].sCmn.fs_kHz, 1000 );
    encControl->stereoWidth_Q14 = encControl->toMono ? 0 : psEnc->sStereo.smth_width_Q14;
    if( prefillFlag ) {
        encControl->payloadSize_ms = tmp_payloadSize_ms;
        encControl->complexity = tmp_complexity;
        for( n = 0; n < encControl->nChannelsInternal; n++ ) {
            psEnc->state_Fxx[ n ].sCmn.controlled_since_last_payload = 0;
            psEnc->state_Fxx[ n ].sCmn.prefillFlag = 0;
        }
    }

    encControl->signalType = psEnc->state_Fxx[0].sCmn.indices.signalType;
    encControl->offset = silk_Quantization_Offsets_Q10
                         [ psEnc->state_Fxx[0].sCmn.indices.signalType >> 1 ]
                         [ psEnc->state_Fxx[0].sCmn.indices.quantOffsetType ];
    RESTORE_STACK;
    return ret;
}

