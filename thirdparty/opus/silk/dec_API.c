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
#include "API.h"
#include "main.h"
#include "stack_alloc.h"
#include "os_support.h"

#ifdef ENABLE_OSCE
#include "osce.h"
#include "osce_structs.h"
#endif

/************************/
/* Decoder Super Struct */
/************************/
typedef struct {
    silk_decoder_state          channel_state[ DECODER_NUM_CHANNELS ];
    stereo_dec_state                sStereo;
    opus_int                         nChannelsAPI;
    opus_int                         nChannelsInternal;
    opus_int                         prev_decode_only_middle;
#ifdef ENABLE_OSCE
    OSCEModel                        osce_model;
#endif
} silk_decoder;

/*********************/
/* Decoder functions */
/*********************/



opus_int silk_LoadOSCEModels(void *decState, const unsigned char *data, int len)
{
#ifdef ENABLE_OSCE
    opus_int ret = SILK_NO_ERROR;

    ret = osce_load_models(&((silk_decoder *)decState)->osce_model, data, len);
    ((silk_decoder *)decState)->osce_model.loaded = (ret == 0);
    return ret;
#else
    (void) decState;
    (void) data;
    (void) len;
    return SILK_NO_ERROR;
#endif
}

opus_int silk_Get_Decoder_Size(                         /* O    Returns error code                              */
    opus_int                        *decSizeBytes       /* O    Number of bytes in SILK decoder state           */
)
{
    opus_int ret = SILK_NO_ERROR;

    *decSizeBytes = sizeof( silk_decoder );

    return ret;
}

/* Reset decoder state */
opus_int silk_ResetDecoder(                              /* O    Returns error code                              */
    void                            *decState           /* I/O  State                                           */
)
{
    opus_int n, ret = SILK_NO_ERROR;
    silk_decoder_state *channel_state = ((silk_decoder *)decState)->channel_state;

    for( n = 0; n < DECODER_NUM_CHANNELS; n++ ) {
        ret  = silk_reset_decoder( &channel_state[ n ] );
    }
    silk_memset(&((silk_decoder *)decState)->sStereo, 0, sizeof(((silk_decoder *)decState)->sStereo));
    /* Not strictly needed, but it's cleaner that way */
    ((silk_decoder *)decState)->prev_decode_only_middle = 0;

    return ret;
}


opus_int silk_InitDecoder(                              /* O    Returns error code                              */
    void                            *decState           /* I/O  State                                           */
)
{
    opus_int n, ret = SILK_NO_ERROR;
    silk_decoder_state *channel_state = ((silk_decoder *)decState)->channel_state;
#ifdef ENABLE_OSCE
    ((silk_decoder *)decState)->osce_model.loaded = 0;
#endif
#ifndef USE_WEIGHTS_FILE
    /* load osce models */
    silk_LoadOSCEModels(decState, NULL, 0);
#endif

    for( n = 0; n < DECODER_NUM_CHANNELS; n++ ) {
        ret  = silk_init_decoder( &channel_state[ n ] );
    }
    silk_memset(&((silk_decoder *)decState)->sStereo, 0, sizeof(((silk_decoder *)decState)->sStereo));
    /* Not strictly needed, but it's cleaner that way */
    ((silk_decoder *)decState)->prev_decode_only_middle = 0;

    return ret;
}

/* Decode a frame */
opus_int silk_Decode(                                   /* O    Returns error code                              */
    void*                           decState,           /* I/O  State                                           */
    silk_DecControlStruct*          decControl,         /* I/O  Control Structure                               */
    opus_int                        lostFlag,           /* I    0: no loss, 1 loss, 2 decode fec                */
    opus_int                        newPacketFlag,      /* I    Indicates first decoder call for this packet    */
    ec_dec                          *psRangeDec,        /* I/O  Compressor data structure                       */
    opus_int16                      *samplesOut,        /* O    Decoded output speech vector                    */
    opus_int32                      *nSamplesOut,       /* O    Number of samples decoded                       */
#ifdef ENABLE_DEEP_PLC
    LPCNetPLCState                  *lpcnet,
#endif
    int                             arch                /* I    Run-time architecture                           */
)
{
    opus_int   i, n, decode_only_middle = 0, ret = SILK_NO_ERROR;
    opus_int32 nSamplesOutDec, LBRR_symbol;
    opus_int16 *samplesOut1_tmp[ 2 ];
    VARDECL( opus_int16, samplesOut1_tmp_storage1 );
    VARDECL( opus_int16, samplesOut1_tmp_storage2 );
    VARDECL( opus_int16, samplesOut2_tmp );
    opus_int32 MS_pred_Q13[ 2 ] = { 0 };
    opus_int16 *resample_out_ptr;
    silk_decoder *psDec = ( silk_decoder * )decState;
    silk_decoder_state *channel_state = psDec->channel_state;
    opus_int has_side;
    opus_int stereo_to_mono;
    int delay_stack_alloc;
    SAVE_STACK;

    celt_assert( decControl->nChannelsInternal == 1 || decControl->nChannelsInternal == 2 );

    /**********************************/
    /* Test if first frame in payload */
    /**********************************/
    if( newPacketFlag ) {
        for( n = 0; n < decControl->nChannelsInternal; n++ ) {
            channel_state[ n ].nFramesDecoded = 0;  /* Used to count frames in packet */
        }
    }

    /* If Mono -> Stereo transition in bitstream: init state of second channel */
    if( decControl->nChannelsInternal > psDec->nChannelsInternal ) {
        ret += silk_init_decoder( &channel_state[ 1 ] );
    }

    stereo_to_mono = decControl->nChannelsInternal == 1 && psDec->nChannelsInternal == 2 &&
                     ( decControl->internalSampleRate == 1000*channel_state[ 0 ].fs_kHz );

    if( channel_state[ 0 ].nFramesDecoded == 0 ) {
        for( n = 0; n < decControl->nChannelsInternal; n++ ) {
            opus_int fs_kHz_dec;
            if( decControl->payloadSize_ms == 0 ) {
                /* Assuming packet loss, use 10 ms */
                channel_state[ n ].nFramesPerPacket = 1;
                channel_state[ n ].nb_subfr = 2;
            } else if( decControl->payloadSize_ms == 10 ) {
                channel_state[ n ].nFramesPerPacket = 1;
                channel_state[ n ].nb_subfr = 2;
            } else if( decControl->payloadSize_ms == 20 ) {
                channel_state[ n ].nFramesPerPacket = 1;
                channel_state[ n ].nb_subfr = 4;
            } else if( decControl->payloadSize_ms == 40 ) {
                channel_state[ n ].nFramesPerPacket = 2;
                channel_state[ n ].nb_subfr = 4;
            } else if( decControl->payloadSize_ms == 60 ) {
                channel_state[ n ].nFramesPerPacket = 3;
                channel_state[ n ].nb_subfr = 4;
            } else {
                celt_assert( 0 );
                RESTORE_STACK;
                return SILK_DEC_INVALID_FRAME_SIZE;
            }
            fs_kHz_dec = ( decControl->internalSampleRate >> 10 ) + 1;
            if( fs_kHz_dec != 8 && fs_kHz_dec != 12 && fs_kHz_dec != 16 ) {
                celt_assert( 0 );
                RESTORE_STACK;
                return SILK_DEC_INVALID_SAMPLING_FREQUENCY;
            }
            ret += silk_decoder_set_fs( &channel_state[ n ], fs_kHz_dec, decControl->API_sampleRate );
        }
    }

    if( decControl->nChannelsAPI == 2 && decControl->nChannelsInternal == 2 && ( psDec->nChannelsAPI == 1 || psDec->nChannelsInternal == 1 ) ) {
        silk_memset( psDec->sStereo.pred_prev_Q13, 0, sizeof( psDec->sStereo.pred_prev_Q13 ) );
        silk_memset( psDec->sStereo.sSide, 0, sizeof( psDec->sStereo.sSide ) );
        silk_memcpy( &channel_state[ 1 ].resampler_state, &channel_state[ 0 ].resampler_state, sizeof( silk_resampler_state_struct ) );
    }
    psDec->nChannelsAPI      = decControl->nChannelsAPI;
    psDec->nChannelsInternal = decControl->nChannelsInternal;

    if( decControl->API_sampleRate > (opus_int32)MAX_API_FS_KHZ * 1000 || decControl->API_sampleRate < 8000 ) {
        ret = SILK_DEC_INVALID_SAMPLING_FREQUENCY;
        RESTORE_STACK;
        return( ret );
    }

    if( lostFlag != FLAG_PACKET_LOST && channel_state[ 0 ].nFramesDecoded == 0 ) {
        /* First decoder call for this payload */
        /* Decode VAD flags and LBRR flag */
        for( n = 0; n < decControl->nChannelsInternal; n++ ) {
            for( i = 0; i < channel_state[ n ].nFramesPerPacket; i++ ) {
                channel_state[ n ].VAD_flags[ i ] = ec_dec_bit_logp(psRangeDec, 1);
            }
            channel_state[ n ].LBRR_flag = ec_dec_bit_logp(psRangeDec, 1);
        }
        /* Decode LBRR flags */
        for( n = 0; n < decControl->nChannelsInternal; n++ ) {
            silk_memset( channel_state[ n ].LBRR_flags, 0, sizeof( channel_state[ n ].LBRR_flags ) );
            if( channel_state[ n ].LBRR_flag ) {
                if( channel_state[ n ].nFramesPerPacket == 1 ) {
                    channel_state[ n ].LBRR_flags[ 0 ] = 1;
                } else {
                    LBRR_symbol = ec_dec_icdf( psRangeDec, silk_LBRR_flags_iCDF_ptr[ channel_state[ n ].nFramesPerPacket - 2 ], 8 ) + 1;
                    for( i = 0; i < channel_state[ n ].nFramesPerPacket; i++ ) {
                        channel_state[ n ].LBRR_flags[ i ] = silk_RSHIFT( LBRR_symbol, i ) & 1;
                    }
                }
            }
        }

        if( lostFlag == FLAG_DECODE_NORMAL ) {
            /* Regular decoding: skip all LBRR data */
            for( i = 0; i < channel_state[ 0 ].nFramesPerPacket; i++ ) {
                for( n = 0; n < decControl->nChannelsInternal; n++ ) {
                    if( channel_state[ n ].LBRR_flags[ i ] ) {
                        opus_int16 pulses[ MAX_FRAME_LENGTH ];
                        opus_int condCoding;

                        if( decControl->nChannelsInternal == 2 && n == 0 ) {
                            silk_stereo_decode_pred( psRangeDec, MS_pred_Q13 );
                            if( channel_state[ 1 ].LBRR_flags[ i ] == 0 ) {
                                silk_stereo_decode_mid_only( psRangeDec, &decode_only_middle );
                            }
                        }
                        /* Use conditional coding if previous frame available */
                        if( i > 0 && channel_state[ n ].LBRR_flags[ i - 1 ] ) {
                            condCoding = CODE_CONDITIONALLY;
                        } else {
                            condCoding = CODE_INDEPENDENTLY;
                        }
                        silk_decode_indices( &channel_state[ n ], psRangeDec, i, 1, condCoding );
                        silk_decode_pulses( psRangeDec, pulses, channel_state[ n ].indices.signalType,
                            channel_state[ n ].indices.quantOffsetType, channel_state[ n ].frame_length );
                    }
                }
            }
        }
    }

    /* Get MS predictor index */
    if( decControl->nChannelsInternal == 2 ) {
        if(   lostFlag == FLAG_DECODE_NORMAL ||
            ( lostFlag == FLAG_DECODE_LBRR && channel_state[ 0 ].LBRR_flags[ channel_state[ 0 ].nFramesDecoded ] == 1 ) )
        {
            silk_stereo_decode_pred( psRangeDec, MS_pred_Q13 );
            /* For LBRR data, decode mid-only flag only if side-channel's LBRR flag is false */
            if( ( lostFlag == FLAG_DECODE_NORMAL && channel_state[ 1 ].VAD_flags[ channel_state[ 0 ].nFramesDecoded ] == 0 ) ||
                ( lostFlag == FLAG_DECODE_LBRR && channel_state[ 1 ].LBRR_flags[ channel_state[ 0 ].nFramesDecoded ] == 0 ) )
            {
                silk_stereo_decode_mid_only( psRangeDec, &decode_only_middle );
            } else {
                decode_only_middle = 0;
            }
        } else {
            for( n = 0; n < 2; n++ ) {
                MS_pred_Q13[ n ] = psDec->sStereo.pred_prev_Q13[ n ];
            }
        }
    }

    /* Reset side channel decoder prediction memory for first frame with side coding */
    if( decControl->nChannelsInternal == 2 && decode_only_middle == 0 && psDec->prev_decode_only_middle == 1 ) {
        silk_memset( psDec->channel_state[ 1 ].outBuf, 0, sizeof(psDec->channel_state[ 1 ].outBuf) );
        silk_memset( psDec->channel_state[ 1 ].sLPC_Q14_buf, 0, sizeof(psDec->channel_state[ 1 ].sLPC_Q14_buf) );
        psDec->channel_state[ 1 ].lagPrev        = 100;
        psDec->channel_state[ 1 ].LastGainIndex  = 10;
        psDec->channel_state[ 1 ].prevSignalType = TYPE_NO_VOICE_ACTIVITY;
        psDec->channel_state[ 1 ].first_frame_after_reset = 1;
    }

    /* Check if the temp buffer fits into the output PCM buffer. If it fits,
       we can delay allocating the temp buffer until after the SILK peak stack
       usage. We need to use a < and not a <= because of the two extra samples. */
    delay_stack_alloc = decControl->internalSampleRate*decControl->nChannelsInternal
          < decControl->API_sampleRate*decControl->nChannelsAPI;
    ALLOC( samplesOut1_tmp_storage1, delay_stack_alloc ? ALLOC_NONE
           : decControl->nChannelsInternal*(channel_state[ 0 ].frame_length + 2 ),
           opus_int16 );
    if ( delay_stack_alloc )
    {
       samplesOut1_tmp[ 0 ] = samplesOut;
       samplesOut1_tmp[ 1 ] = samplesOut + channel_state[ 0 ].frame_length + 2;
    } else {
       samplesOut1_tmp[ 0 ] = samplesOut1_tmp_storage1;
       samplesOut1_tmp[ 1 ] = samplesOut1_tmp_storage1 + channel_state[ 0 ].frame_length + 2;
    }

    if( lostFlag == FLAG_DECODE_NORMAL ) {
        has_side = !decode_only_middle;
    } else {
        has_side = !psDec->prev_decode_only_middle
              || (decControl->nChannelsInternal == 2 && lostFlag == FLAG_DECODE_LBRR && channel_state[1].LBRR_flags[ channel_state[1].nFramesDecoded ] == 1 );
    }
    channel_state[ 0 ].sPLC.enable_deep_plc = decControl->enable_deep_plc;
    /* Call decoder for one frame */
    for( n = 0; n < decControl->nChannelsInternal; n++ ) {
        if( n == 0 || has_side ) {
            opus_int FrameIndex;
            opus_int condCoding;

            FrameIndex = channel_state[ 0 ].nFramesDecoded - n;
            /* Use independent coding if no previous frame available */
            if( FrameIndex <= 0 ) {
                condCoding = CODE_INDEPENDENTLY;
            } else if( lostFlag == FLAG_DECODE_LBRR ) {
                condCoding = channel_state[ n ].LBRR_flags[ FrameIndex - 1 ] ? CODE_CONDITIONALLY : CODE_INDEPENDENTLY;
            } else if( n > 0 && psDec->prev_decode_only_middle ) {
                /* If we skipped a side frame in this packet, we don't
                   need LTP scaling; the LTP state is well-defined. */
                condCoding = CODE_INDEPENDENTLY_NO_LTP_SCALING;
            } else {
                condCoding = CODE_CONDITIONALLY;
            }
#ifdef ENABLE_OSCE
            if ( channel_state[n].osce.method != decControl->osce_method ) {
                osce_reset( &channel_state[n].osce, decControl->osce_method );
            }
#endif
            ret += silk_decode_frame( &channel_state[ n ], psRangeDec, &samplesOut1_tmp[ n ][ 2 ], &nSamplesOutDec, lostFlag, condCoding,
#ifdef ENABLE_DEEP_PLC
                n == 0 ? lpcnet : NULL,
#endif
#ifdef ENABLE_OSCE
                &psDec->osce_model,
#endif
                arch);
        } else {
            silk_memset( &samplesOut1_tmp[ n ][ 2 ], 0, nSamplesOutDec * sizeof( opus_int16 ) );
        }
        channel_state[ n ].nFramesDecoded++;
    }

    if( decControl->nChannelsAPI == 2 && decControl->nChannelsInternal == 2 ) {
        /* Convert Mid/Side to Left/Right */
        silk_stereo_MS_to_LR( &psDec->sStereo, samplesOut1_tmp[ 0 ], samplesOut1_tmp[ 1 ], MS_pred_Q13, channel_state[ 0 ].fs_kHz, nSamplesOutDec );
    } else {
        /* Buffering */
        silk_memcpy( samplesOut1_tmp[ 0 ], psDec->sStereo.sMid, 2 * sizeof( opus_int16 ) );
        silk_memcpy( psDec->sStereo.sMid, &samplesOut1_tmp[ 0 ][ nSamplesOutDec ], 2 * sizeof( opus_int16 ) );
    }

    /* Number of output samples */
    *nSamplesOut = silk_DIV32( nSamplesOutDec * decControl->API_sampleRate, silk_SMULBB( channel_state[ 0 ].fs_kHz, 1000 ) );

    /* Set up pointers to temp buffers */
    ALLOC( samplesOut2_tmp,
           decControl->nChannelsAPI == 2 ? *nSamplesOut : ALLOC_NONE, opus_int16 );
    if( decControl->nChannelsAPI == 2 ) {
        resample_out_ptr = samplesOut2_tmp;
    } else {
        resample_out_ptr = samplesOut;
    }

    ALLOC( samplesOut1_tmp_storage2, delay_stack_alloc
           ? decControl->nChannelsInternal*(channel_state[ 0 ].frame_length + 2 )
           : ALLOC_NONE,
           opus_int16 );
    if ( delay_stack_alloc ) {
       OPUS_COPY(samplesOut1_tmp_storage2, samplesOut, decControl->nChannelsInternal*(channel_state[ 0 ].frame_length + 2));
       samplesOut1_tmp[ 0 ] = samplesOut1_tmp_storage2;
       samplesOut1_tmp[ 1 ] = samplesOut1_tmp_storage2 + channel_state[ 0 ].frame_length + 2;
    }
    for( n = 0; n < silk_min( decControl->nChannelsAPI, decControl->nChannelsInternal ); n++ ) {

        /* Resample decoded signal to API_sampleRate */
        ret += silk_resampler( &channel_state[ n ].resampler_state, resample_out_ptr, &samplesOut1_tmp[ n ][ 1 ], nSamplesOutDec );

        /* Interleave if stereo output and stereo stream */
        if( decControl->nChannelsAPI == 2 ) {
            for( i = 0; i < *nSamplesOut; i++ ) {
                samplesOut[ n + 2 * i ] = resample_out_ptr[ i ];
            }
        }
    }

    /* Create two channel output from mono stream */
    if( decControl->nChannelsAPI == 2 && decControl->nChannelsInternal == 1 ) {
        if ( stereo_to_mono ){
            /* Resample right channel for newly collapsed stereo just in case
               we weren't doing collapsing when switching to mono */
            ret += silk_resampler( &channel_state[ 1 ].resampler_state, resample_out_ptr, &samplesOut1_tmp[ 0 ][ 1 ], nSamplesOutDec );

            for( i = 0; i < *nSamplesOut; i++ ) {
                samplesOut[ 1 + 2 * i ] = resample_out_ptr[ i ];
            }
        } else {
            for( i = 0; i < *nSamplesOut; i++ ) {
                samplesOut[ 1 + 2 * i ] = samplesOut[ 0 + 2 * i ];
            }
        }
    }

    /* Export pitch lag, measured at 48 kHz sampling rate */
    if( channel_state[ 0 ].prevSignalType == TYPE_VOICED ) {
        int mult_tab[ 3 ] = { 6, 4, 3 };
        decControl->prevPitchLag = channel_state[ 0 ].lagPrev * mult_tab[ ( channel_state[ 0 ].fs_kHz - 8 ) >> 2 ];
    } else {
        decControl->prevPitchLag = 0;
    }

    if( lostFlag == FLAG_PACKET_LOST ) {
       /* On packet loss, remove the gain clamping to prevent having the energy "bounce back"
          if we lose packets when the energy is going down */
       for ( i = 0; i < psDec->nChannelsInternal; i++ )
          psDec->channel_state[ i ].LastGainIndex = 10;
    } else {
       psDec->prev_decode_only_middle = decode_only_middle;
    }
    RESTORE_STACK;
    return ret;
}

#if 0
/* Getting table of contents for a packet */
opus_int silk_get_TOC(
    const opus_uint8                *payload,           /* I    Payload data                                */
    const opus_int                  nBytesIn,           /* I    Number of input bytes                       */
    const opus_int                  nFramesPerPayload,  /* I    Number of SILK frames per payload           */
    silk_TOC_struct                 *Silk_TOC           /* O    Type of content                             */
)
{
    opus_int i, flags, ret = SILK_NO_ERROR;

    if( nBytesIn < 1 ) {
        return -1;
    }
    if( nFramesPerPayload < 0 || nFramesPerPayload > 3 ) {
        return -1;
    }

    silk_memset( Silk_TOC, 0, sizeof( *Silk_TOC ) );

    /* For stereo, extract the flags for the mid channel */
    flags = silk_RSHIFT( payload[ 0 ], 7 - nFramesPerPayload ) & ( silk_LSHIFT( 1, nFramesPerPayload + 1 ) - 1 );

    Silk_TOC->inbandFECFlag = flags & 1;
    for( i = nFramesPerPayload - 1; i >= 0 ; i-- ) {
        flags = silk_RSHIFT( flags, 1 );
        Silk_TOC->VADFlags[ i ] = flags & 1;
        Silk_TOC->VADFlag |= flags & 1;
    }

    return ret;
}
#endif
