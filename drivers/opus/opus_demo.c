/* Copyright (c) 2007-2008 CSIRO
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

#ifdef OPUS_HAVE_CONFIG_H
#include "opus/opus_config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "opus/opus.h"
#include "opus/silk/debug.h"
#include "opus/opus_types.h"
#include "opus/opus_private.h"
#include "opus/opus_multistream.h"

#define MAX_PACKET 1500

void print_usage( char* argv[] )
{
    fprintf(stderr, "Usage: %s [-e] <application> <sampling rate (Hz)> <channels (1/2)> "
        "<bits per second>  [options] <input> <output>\n", argv[0]);
    fprintf(stderr, "       %s -d <sampling rate (Hz)> <channels (1/2)> "
        "[options] <input> <output>\n\n", argv[0]);
    fprintf(stderr, "mode: voip | audio | restricted-lowdelay\n" );
    fprintf(stderr, "options:\n" );
    fprintf(stderr, "-e                   : only runs the encoder (output the bit-stream)\n" );
    fprintf(stderr, "-d                   : only runs the decoder (reads the bit-stream as input)\n" );
    fprintf(stderr, "-cbr                 : enable constant bitrate; default: variable bitrate\n" );
    fprintf(stderr, "-cvbr                : enable constrained variable bitrate; default: unconstrained\n" );
    fprintf(stderr, "-variable-duration   : enable frames of variable duration (experts only); default: disabled\n" );
    fprintf(stderr, "-bandwidth <NB|MB|WB|SWB|FB> : audio bandwidth (from narrowband to fullband); default: sampling rate\n" );
    fprintf(stderr, "-framesize <2.5|5|10|20|40|60> : frame size in ms; default: 20 \n" );
    fprintf(stderr, "-max_payload <bytes> : maximum payload size in bytes, default: 1024\n" );
    fprintf(stderr, "-complexity <comp>   : complexity, 0 (lowest) ... 10 (highest); default: 10\n" );
    fprintf(stderr, "-inbandfec           : enable SILK inband FEC\n" );
    fprintf(stderr, "-forcemono           : force mono encoding, even for stereo input\n" );
    fprintf(stderr, "-dtx                 : enable SILK DTX\n" );
    fprintf(stderr, "-loss <perc>         : simulate packet loss, in percent (0-100); default: 0\n" );
}

static void int_to_char(opus_uint32 i, unsigned char ch[4])
{
    ch[0] = i>>24;
    ch[1] = (i>>16)&0xFF;
    ch[2] = (i>>8)&0xFF;
    ch[3] = i&0xFF;
}

static opus_uint32 char_to_int(unsigned char ch[4])
{
    return ((opus_uint32)ch[0]<<24) | ((opus_uint32)ch[1]<<16)
         | ((opus_uint32)ch[2]<< 8) |  (opus_uint32)ch[3];
}

static void check_encoder_option(int decode_only, const char *opt)
{
   if (decode_only)
   {
      fprintf(stderr, "option %s is only for encoding\n", opt);
      exit(EXIT_FAILURE);
   }
}

static const int silk8_test[][4] = {
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND, 960*3, 1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND, 960*2, 1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND, 960,   1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND, 480,   1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND, 960*3, 2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND, 960*2, 2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND, 960,   2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND, 480,   2}
};

static const int silk12_test[][4] = {
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND, 960*3, 1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND, 960*2, 1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND, 960,   1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND, 480,   1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND, 960*3, 2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND, 960*2, 2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND, 960,   2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND, 480,   2}
};

static const int silk16_test[][4] = {
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND, 960*3, 1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND, 960*2, 1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND, 960,   1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND, 480,   1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND, 960*3, 2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND, 960*2, 2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND, 960,   2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND, 480,   2}
};

static const int hybrid24_test[][4] = {
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 960, 1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 480, 1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 960, 2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 480, 2}
};

static const int hybrid48_test[][4] = {
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_FULLBAND, 960, 1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_FULLBAND, 480, 1},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_FULLBAND, 960, 2},
      {MODE_SILK_ONLY, OPUS_BANDWIDTH_FULLBAND, 480, 2}
};

static const int celt_test[][4] = {
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      960, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 960, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_WIDEBAND,      960, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_NARROWBAND,    960, 1},

      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      480, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 480, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_WIDEBAND,      480, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_NARROWBAND,    480, 1},

      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      240, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 240, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_WIDEBAND,      240, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_NARROWBAND,    240, 1},

      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      120, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 120, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_WIDEBAND,      120, 1},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_NARROWBAND,    120, 1},

      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      960, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 960, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_WIDEBAND,      960, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_NARROWBAND,    960, 2},

      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      480, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 480, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_WIDEBAND,      480, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_NARROWBAND,    480, 2},

      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      240, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 240, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_WIDEBAND,      240, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_NARROWBAND,    240, 2},

      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      120, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND, 120, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_WIDEBAND,      120, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_NARROWBAND,    120, 2},

};

static const int celt_hq_test[][4] = {
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      960, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      480, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      240, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      120, 2},
};

#if 0 /* This is a hack that replaces the normal encoder/decoder with the multistream version */
#define OpusEncoder OpusMSEncoder
#define OpusDecoder OpusMSDecoder
#define opus_encode opus_multistream_encode
#define opus_decode opus_multistream_decode
#define opus_encoder_ctl opus_multistream_encoder_ctl
#define opus_decoder_ctl opus_multistream_decoder_ctl
#define opus_encoder_create ms_opus_encoder_create
#define opus_decoder_create ms_opus_decoder_create
#define opus_encoder_destroy opus_multistream_encoder_destroy
#define opus_decoder_destroy opus_multistream_decoder_destroy

static OpusEncoder *ms_opus_encoder_create(opus_int32 Fs, int channels, int application, int *error)
{
   int streams, coupled_streams;
   unsigned char mapping[256];
   return (OpusEncoder *)opus_multistream_surround_encoder_create(Fs, channels, 1, &streams, &coupled_streams, mapping, application, error);
}
static OpusDecoder *ms_opus_decoder_create(opus_int32 Fs, int channels, int *error)
{
   int streams;
   int coupled_streams;
   unsigned char mapping[256]={0,1};
   streams = 1;
   coupled_streams = channels==2;
   return (OpusDecoder *)opus_multistream_decoder_create(Fs, channels, streams, coupled_streams, mapping, error);
}
#endif

int main(int argc, char *argv[])
{
    int err;
    char *inFile, *outFile;
    FILE *fin, *fout;
    OpusEncoder *enc=NULL;
    OpusDecoder *dec=NULL;
    int args;
    int len[2];
    int frame_size, channels;
    opus_int32 bitrate_bps=0;
    unsigned char *data[2];
    unsigned char *fbytes;
    opus_int32 sampling_rate;
    int use_vbr;
    int max_payload_bytes;
    int complexity;
    int use_inbandfec;
    int use_dtx;
    int forcechannels;
    int cvbr = 0;
    int packet_loss_perc;
    opus_int32 count=0, count_act=0;
    int k;
    opus_int32 skip=0;
    int stop=0;
    short *in, *out;
    int application=OPUS_APPLICATION_AUDIO;
    double bits=0.0, bits_max=0.0, bits_act=0.0, bits2=0.0, nrg;
    double tot_samples=0;
    opus_uint64 tot_in, tot_out;
    int bandwidth=-1;
    const char *bandwidth_string;
    int lost = 0, lost_prev = 1;
    int toggle = 0;
    opus_uint32 enc_final_range[2];
    opus_uint32 dec_final_range;
    int encode_only=0, decode_only=0;
    int max_frame_size = 960*6;
    int curr_read=0;
    int sweep_bps = 0;
    int random_framesize=0, newsize=0, delayed_celt=0;
    int sweep_max=0, sweep_min=0;
    int random_fec=0;
    const int (*mode_list)[4]=NULL;
    int nb_modes_in_list=0;
    int curr_mode=0;
    int curr_mode_count=0;
    int mode_switch_time = 48000;
    int nb_encoded=0;
    int remaining=0;
    int variable_duration=OPUS_FRAMESIZE_ARG;
    int delayed_decision=0;

    if (argc < 5 )
    {
       print_usage( argv );
       return EXIT_FAILURE;
    }

    tot_in=tot_out=0;
    fprintf(stderr, "%s\n", opus_get_version_string());

    args = 1;
    if (strcmp(argv[args], "-e")==0)
    {
        encode_only = 1;
        args++;
    } else if (strcmp(argv[args], "-d")==0)
    {
        decode_only = 1;
        args++;
    }
    if (!decode_only && argc < 7 )
    {
       print_usage( argv );
       return EXIT_FAILURE;
    }

    if (!decode_only)
    {
       if (strcmp(argv[args], "voip")==0)
          application = OPUS_APPLICATION_VOIP;
       else if (strcmp(argv[args], "restricted-lowdelay")==0)
          application = OPUS_APPLICATION_RESTRICTED_LOWDELAY;
       else if (strcmp(argv[args], "audio")!=0) {
          fprintf(stderr, "unknown application: %s\n", argv[args]);
          print_usage(argv);
          return EXIT_FAILURE;
       }
       args++;
    }
    sampling_rate = (opus_int32)atol(argv[args]);
    args++;

    if (sampling_rate != 8000 && sampling_rate != 12000
     && sampling_rate != 16000 && sampling_rate != 24000
     && sampling_rate != 48000)
    {
        fprintf(stderr, "Supported sampling rates are 8000, 12000, "
                "16000, 24000 and 48000.\n");
        return EXIT_FAILURE;
    }
    frame_size = sampling_rate/50;

    channels = atoi(argv[args]);
    args++;

    if (channels < 1 || channels > 2)
    {
        fprintf(stderr, "Opus_demo supports only 1 or 2 channels.\n");
        return EXIT_FAILURE;
    }

    if (!decode_only)
    {
       bitrate_bps = (opus_int32)atol(argv[args]);
       args++;
    }

    /* defaults: */
    use_vbr = 1;
    bandwidth = OPUS_AUTO;
    max_payload_bytes = MAX_PACKET;
    complexity = 10;
    use_inbandfec = 0;
    forcechannels = OPUS_AUTO;
    use_dtx = 0;
    packet_loss_perc = 0;
    max_frame_size = 2*48000;
    curr_read=0;

    while( args < argc - 2 ) {
        /* process command line options */
        if( strcmp( argv[ args ], "-cbr" ) == 0 ) {
            check_encoder_option(decode_only, "-cbr");
            use_vbr = 0;
            args++;
        } else if( strcmp( argv[ args ], "-bandwidth" ) == 0 ) {
            check_encoder_option(decode_only, "-bandwidth");
            if (strcmp(argv[ args + 1 ], "NB")==0)
                bandwidth = OPUS_BANDWIDTH_NARROWBAND;
            else if (strcmp(argv[ args + 1 ], "MB")==0)
                bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
            else if (strcmp(argv[ args + 1 ], "WB")==0)
                bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            else if (strcmp(argv[ args + 1 ], "SWB")==0)
                bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
            else if (strcmp(argv[ args + 1 ], "FB")==0)
                bandwidth = OPUS_BANDWIDTH_FULLBAND;
            else {
                fprintf(stderr, "Unknown bandwidth %s. "
                                "Supported are NB, MB, WB, SWB, FB.\n",
                                argv[ args + 1 ]);
                return EXIT_FAILURE;
            }
            args += 2;
        } else if( strcmp( argv[ args ], "-framesize" ) == 0 ) {
            check_encoder_option(decode_only, "-framesize");
            if (strcmp(argv[ args + 1 ], "2.5")==0)
                frame_size = sampling_rate/400;
            else if (strcmp(argv[ args + 1 ], "5")==0)
                frame_size = sampling_rate/200;
            else if (strcmp(argv[ args + 1 ], "10")==0)
                frame_size = sampling_rate/100;
            else if (strcmp(argv[ args + 1 ], "20")==0)
                frame_size = sampling_rate/50;
            else if (strcmp(argv[ args + 1 ], "40")==0)
                frame_size = sampling_rate/25;
            else if (strcmp(argv[ args + 1 ], "60")==0)
                frame_size = 3*sampling_rate/50;
            else {
                fprintf(stderr, "Unsupported frame size: %s ms. "
                                "Supported are 2.5, 5, 10, 20, 40, 60.\n",
                                argv[ args + 1 ]);
                return EXIT_FAILURE;
            }
            args += 2;
        } else if( strcmp( argv[ args ], "-max_payload" ) == 0 ) {
            check_encoder_option(decode_only, "-max_payload");
            max_payload_bytes = atoi( argv[ args + 1 ] );
            args += 2;
        } else if( strcmp( argv[ args ], "-complexity" ) == 0 ) {
            check_encoder_option(decode_only, "-complexity");
            complexity = atoi( argv[ args + 1 ] );
            args += 2;
        } else if( strcmp( argv[ args ], "-inbandfec" ) == 0 ) {
            use_inbandfec = 1;
            args++;
        } else if( strcmp( argv[ args ], "-forcemono" ) == 0 ) {
            check_encoder_option(decode_only, "-forcemono");
            forcechannels = 1;
            args++;
        } else if( strcmp( argv[ args ], "-cvbr" ) == 0 ) {
            check_encoder_option(decode_only, "-cvbr");
            cvbr = 1;
            args++;
        } else if( strcmp( argv[ args ], "-variable-duration" ) == 0 ) {
            check_encoder_option(decode_only, "-variable-duration");
            variable_duration = OPUS_FRAMESIZE_VARIABLE;
            args++;
        } else if( strcmp( argv[ args ], "-delayed-decision" ) == 0 ) {
            check_encoder_option(decode_only, "-delayed-decision");
            delayed_decision = 1;
            args++;
        } else if( strcmp( argv[ args ], "-dtx") == 0 ) {
            check_encoder_option(decode_only, "-dtx");
            use_dtx = 1;
            args++;
        } else if( strcmp( argv[ args ], "-loss" ) == 0 ) {
            packet_loss_perc = atoi( argv[ args + 1 ] );
            args += 2;
        } else if( strcmp( argv[ args ], "-sweep" ) == 0 ) {
            check_encoder_option(decode_only, "-sweep");
            sweep_bps = atoi( argv[ args + 1 ] );
            args += 2;
        } else if( strcmp( argv[ args ], "-random_framesize" ) == 0 ) {
            check_encoder_option(decode_only, "-random_framesize");
            random_framesize = 1;
            args++;
        } else if( strcmp( argv[ args ], "-sweep_max" ) == 0 ) {
            check_encoder_option(decode_only, "-sweep_max");
            sweep_max = atoi( argv[ args + 1 ] );
            args += 2;
        } else if( strcmp( argv[ args ], "-random_fec" ) == 0 ) {
            check_encoder_option(decode_only, "-random_fec");
            random_fec = 1;
            args++;
        } else if( strcmp( argv[ args ], "-silk8k_test" ) == 0 ) {
            check_encoder_option(decode_only, "-silk8k_test");
            mode_list = silk8_test;
            nb_modes_in_list = 8;
            args++;
        } else if( strcmp( argv[ args ], "-silk12k_test" ) == 0 ) {
            check_encoder_option(decode_only, "-silk12k_test");
            mode_list = silk12_test;
            nb_modes_in_list = 8;
            args++;
        } else if( strcmp( argv[ args ], "-silk16k_test" ) == 0 ) {
            check_encoder_option(decode_only, "-silk16k_test");
            mode_list = silk16_test;
            nb_modes_in_list = 8;
            args++;
        } else if( strcmp( argv[ args ], "-hybrid24k_test" ) == 0 ) {
            check_encoder_option(decode_only, "-hybrid24k_test");
            mode_list = hybrid24_test;
            nb_modes_in_list = 4;
            args++;
        } else if( strcmp( argv[ args ], "-hybrid48k_test" ) == 0 ) {
            check_encoder_option(decode_only, "-hybrid48k_test");
            mode_list = hybrid48_test;
            nb_modes_in_list = 4;
            args++;
        } else if( strcmp( argv[ args ], "-celt_test" ) == 0 ) {
            check_encoder_option(decode_only, "-celt_test");
            mode_list = celt_test;
            nb_modes_in_list = 32;
            args++;
        } else if( strcmp( argv[ args ], "-celt_hq_test" ) == 0 ) {
            check_encoder_option(decode_only, "-celt_hq_test");
            mode_list = celt_hq_test;
            nb_modes_in_list = 4;
            args++;
        } else {
            printf( "Error: unrecognized setting: %s\n\n", argv[ args ] );
            print_usage( argv );
            return EXIT_FAILURE;
        }
    }

    if (sweep_max)
       sweep_min = bitrate_bps;

    if (max_payload_bytes < 0 || max_payload_bytes > MAX_PACKET)
    {
        fprintf (stderr, "max_payload_bytes must be between 0 and %d\n",
                          MAX_PACKET);
        return EXIT_FAILURE;
    }

    inFile = argv[argc-2];
    fin = fopen(inFile, "rb");
    if (!fin)
    {
        fprintf (stderr, "Could not open input file %s\n", argv[argc-2]);
        return EXIT_FAILURE;
    }
    if (mode_list)
    {
       int size;
       fseek(fin, 0, SEEK_END);
       size = ftell(fin);
       fprintf(stderr, "File size is %d bytes\n", size);
       fseek(fin, 0, SEEK_SET);
       mode_switch_time = size/sizeof(short)/channels/nb_modes_in_list;
       fprintf(stderr, "Switching mode every %d samples\n", mode_switch_time);
    }

    outFile = argv[argc-1];
    fout = fopen(outFile, "wb+");
    if (!fout)
    {
        fprintf (stderr, "Could not open output file %s\n", argv[argc-1]);
        fclose(fin);
        return EXIT_FAILURE;
    }

    if (!decode_only)
    {
       enc = opus_encoder_create(sampling_rate, channels, application, &err);
       if (err != OPUS_OK)
       {
          fprintf(stderr, "Cannot create encoder: %s\n", opus_strerror(err));
          fclose(fin);
          fclose(fout);
          return EXIT_FAILURE;
       }
       opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate_bps));
       opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(bandwidth));
       opus_encoder_ctl(enc, OPUS_SET_VBR(use_vbr));
       opus_encoder_ctl(enc, OPUS_SET_VBR_CONSTRAINT(cvbr));
       opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(complexity));
       opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC(use_inbandfec));
       opus_encoder_ctl(enc, OPUS_SET_FORCE_CHANNELS(forcechannels));
       opus_encoder_ctl(enc, OPUS_SET_DTX(use_dtx));
       opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC(packet_loss_perc));

       opus_encoder_ctl(enc, OPUS_GET_LOOKAHEAD(&skip));
       opus_encoder_ctl(enc, OPUS_SET_LSB_DEPTH(16));
       opus_encoder_ctl(enc, OPUS_SET_EXPERT_FRAME_DURATION(variable_duration));
    }
    if (!encode_only)
    {
       dec = opus_decoder_create(sampling_rate, channels, &err);
       if (err != OPUS_OK)
       {
          fprintf(stderr, "Cannot create decoder: %s\n", opus_strerror(err));
          fclose(fin);
          fclose(fout);
          return EXIT_FAILURE;
       }
    }


    switch(bandwidth)
    {
    case OPUS_BANDWIDTH_NARROWBAND:
         bandwidth_string = "narrowband";
         break;
    case OPUS_BANDWIDTH_MEDIUMBAND:
         bandwidth_string = "mediumband";
         break;
    case OPUS_BANDWIDTH_WIDEBAND:
         bandwidth_string = "wideband";
         break;
    case OPUS_BANDWIDTH_SUPERWIDEBAND:
         bandwidth_string = "superwideband";
         break;
    case OPUS_BANDWIDTH_FULLBAND:
         bandwidth_string = "fullband";
         break;
    case OPUS_AUTO:
         bandwidth_string = "auto";
         break;
    default:
         bandwidth_string = "unknown";
         break;
    }

    if (decode_only)
       fprintf(stderr, "Decoding with %ld Hz output (%d channels)\n",
                       (long)sampling_rate, channels);
    else
       fprintf(stderr, "Encoding %ld Hz input at %.3f kb/s "
                       "in %s mode with %d-sample frames.\n",
                       (long)sampling_rate, bitrate_bps*0.001,
                       bandwidth_string, frame_size);

    in = (short*)malloc(max_frame_size*channels*sizeof(short));
    out = (short*)malloc(max_frame_size*channels*sizeof(short));
    fbytes = (unsigned char*)malloc(max_frame_size*channels*sizeof(short));
    data[0] = (unsigned char*)calloc(max_payload_bytes,sizeof(char));
    if ( use_inbandfec ) {
        data[1] = (unsigned char*)calloc(max_payload_bytes,sizeof(char));
    }
    if(delayed_decision)
    {
       if (variable_duration!=OPUS_FRAMESIZE_VARIABLE)
       {
          if (frame_size==sampling_rate/400)
             variable_duration = OPUS_FRAMESIZE_2_5_MS;
          else if (frame_size==sampling_rate/200)
             variable_duration = OPUS_FRAMESIZE_5_MS;
          else if (frame_size==sampling_rate/100)
             variable_duration = OPUS_FRAMESIZE_10_MS;
          else if (frame_size==sampling_rate/50)
             variable_duration = OPUS_FRAMESIZE_20_MS;
          else if (frame_size==sampling_rate/25)
             variable_duration = OPUS_FRAMESIZE_40_MS;
          else
             variable_duration = OPUS_FRAMESIZE_60_MS;
          opus_encoder_ctl(enc, OPUS_SET_EXPERT_FRAME_DURATION(variable_duration));
       }
       frame_size = 2*48000;
    }
    while (!stop)
    {
        if (delayed_celt)
        {
            frame_size = newsize;
            delayed_celt = 0;
        } else if (random_framesize && rand()%20==0)
        {
            newsize = rand()%6;
            switch(newsize)
            {
            case 0: newsize=sampling_rate/400; break;
            case 1: newsize=sampling_rate/200; break;
            case 2: newsize=sampling_rate/100; break;
            case 3: newsize=sampling_rate/50; break;
            case 4: newsize=sampling_rate/25; break;
            case 5: newsize=3*sampling_rate/50; break;
            }
            while (newsize < sampling_rate/25 && bitrate_bps-fabs(sweep_bps) <= 3*12*sampling_rate/newsize)
               newsize*=2;
            if (newsize < sampling_rate/100 && frame_size >= sampling_rate/100)
            {
                opus_encoder_ctl(enc, OPUS_SET_FORCE_MODE(MODE_CELT_ONLY));
                delayed_celt=1;
            } else {
                frame_size = newsize;
            }
        }
        if (random_fec && rand()%30==0)
        {
           opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC(rand()%4==0));
        }
        if (decode_only)
        {
            unsigned char ch[4];
            err = fread(ch, 1, 4, fin);
            if (feof(fin))
                break;
            len[toggle] = char_to_int(ch);
            if (len[toggle]>max_payload_bytes || len[toggle]<0)
            {
                fprintf(stderr, "Invalid payload length: %d\n",len[toggle]);
                break;
            }
            err = fread(ch, 1, 4, fin);
            enc_final_range[toggle] = char_to_int(ch);
            err = fread(data[toggle], 1, len[toggle], fin);
            if (err<len[toggle])
            {
                fprintf(stderr, "Ran out of input, "
                                "expecting %d bytes got %d\n",
                                len[toggle],err);
                break;
            }
        } else {
            int i;
            if (mode_list!=NULL)
            {
                opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(mode_list[curr_mode][1]));
                opus_encoder_ctl(enc, OPUS_SET_FORCE_MODE(mode_list[curr_mode][0]));
                opus_encoder_ctl(enc, OPUS_SET_FORCE_CHANNELS(mode_list[curr_mode][3]));
                frame_size = mode_list[curr_mode][2];
            }
            err = fread(fbytes, sizeof(short)*channels, frame_size-remaining, fin);
            curr_read = err;
            tot_in += curr_read;
            for(i=0;i<curr_read*channels;i++)
            {
                opus_int32 s;
                s=fbytes[2*i+1]<<8|fbytes[2*i];
                s=((s&0xFFFF)^0x8000)-0x8000;
                in[i+remaining*channels]=s;
            }
            if (curr_read+remaining < frame_size)
            {
                for (i=(curr_read+remaining)*channels;i<frame_size*channels;i++)
                   in[i] = 0;
                if (encode_only || decode_only)
                   stop = 1;
            }
            len[toggle] = opus_encode(enc, in, frame_size, data[toggle], max_payload_bytes);
            nb_encoded = opus_packet_get_samples_per_frame(data[toggle], sampling_rate)*opus_packet_get_nb_frames(data[toggle], len[toggle]);
            remaining = frame_size-nb_encoded;
            for(i=0;i<remaining*channels;i++)
               in[i] = in[nb_encoded*channels+i];
            if (sweep_bps!=0)
            {
               bitrate_bps += sweep_bps;
               if (sweep_max)
               {
                  if (bitrate_bps > sweep_max)
                     sweep_bps = -sweep_bps;
                  else if (bitrate_bps < sweep_min)
                     sweep_bps = -sweep_bps;
               }
               /* safety */
               if (bitrate_bps<1000)
                  bitrate_bps = 1000;
               opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate_bps));
            }
            opus_encoder_ctl(enc, OPUS_GET_FINAL_RANGE(&enc_final_range[toggle]));
            if (len[toggle] < 0)
            {
                fprintf (stderr, "opus_encode() returned %d\n", len[toggle]);
                fclose(fin);
                fclose(fout);
                return EXIT_FAILURE;
            }
            curr_mode_count += frame_size;
            if (curr_mode_count > mode_switch_time && curr_mode < nb_modes_in_list-1)
            {
               curr_mode++;
               curr_mode_count = 0;
            }
        }

#if 0 /* This is for testing the padding code, do not enable by default */
        if (len[toggle]<1275)
        {
           int new_len = len[toggle]+rand()%(max_payload_bytes-len[toggle]);
           if ((err = opus_packet_pad(data[toggle], len[toggle], new_len)) != OPUS_OK)
           {
              fprintf(stderr, "padding failed: %s\n", opus_strerror(err));
              return EXIT_FAILURE;
           }
           len[toggle] = new_len;
        }
#endif
        if (encode_only)
        {
            unsigned char int_field[4];
            int_to_char(len[toggle], int_field);
            if (fwrite(int_field, 1, 4, fout) != 4) {
               fprintf(stderr, "Error writing.\n");
               return EXIT_FAILURE;
            }
            int_to_char(enc_final_range[toggle], int_field);
            if (fwrite(int_field, 1, 4, fout) != 4) {
               fprintf(stderr, "Error writing.\n");
               return EXIT_FAILURE;
            }
            if (fwrite(data[toggle], 1, len[toggle], fout) != (unsigned)len[toggle]) {
               fprintf(stderr, "Error writing.\n");
               return EXIT_FAILURE;
            }
            tot_samples += nb_encoded;
        } else {
            int output_samples;
            lost = len[toggle]==0 || (packet_loss_perc>0 && rand()%100 < packet_loss_perc);
            if (lost)
               opus_decoder_ctl(dec, OPUS_GET_LAST_PACKET_DURATION(&output_samples));
            else
               output_samples = max_frame_size;
            if( count >= use_inbandfec ) {
                /* delay by one packet when using in-band FEC */
                if( use_inbandfec  ) {
                    if( lost_prev ) {
                        /* attempt to decode with in-band FEC from next packet */
                        opus_decoder_ctl(dec, OPUS_GET_LAST_PACKET_DURATION(&output_samples));
                        output_samples = opus_decode(dec, lost ? NULL : data[toggle], len[toggle], out, output_samples, 1);
                    } else {
                        /* regular decode */
                        output_samples = max_frame_size;
                        output_samples = opus_decode(dec, data[1-toggle], len[1-toggle], out, output_samples, 0);
                    }
                } else {
                    output_samples = opus_decode(dec, lost ? NULL : data[toggle], len[toggle], out, output_samples, 0);
                }
                if (output_samples>0)
                {
                    if (!decode_only && tot_out + output_samples > tot_in)
                    {
                       stop=1;
                       output_samples  = tot_in-tot_out;
                    }
                    if (output_samples>skip) {
                       int i;
                       for(i=0;i<(output_samples-skip)*channels;i++)
                       {
                          short s;
                          s=out[i+(skip*channels)];
                          fbytes[2*i]=s&0xFF;
                          fbytes[2*i+1]=(s>>8)&0xFF;
                       }
                       if (fwrite(fbytes, sizeof(short)*channels, output_samples-skip, fout) != (unsigned)(output_samples-skip)){
                          fprintf(stderr, "Error writing.\n");
                          return EXIT_FAILURE;
                       }
                       tot_out += output_samples-skip;
                    }
                    if (output_samples<skip) skip -= output_samples;
                    else skip = 0;
                } else {
                   fprintf(stderr, "error decoding frame: %s\n",
                                   opus_strerror(output_samples));
                }
                tot_samples += output_samples;
            }
        }

        if (!encode_only)
           opus_decoder_ctl(dec, OPUS_GET_FINAL_RANGE(&dec_final_range));
        /* compare final range encoder rng values of encoder and decoder */
        if( enc_final_range[toggle^use_inbandfec]!=0  && !encode_only
         && !lost && !lost_prev
         && dec_final_range != enc_final_range[toggle^use_inbandfec] ) {
            fprintf (stderr, "Error: Range coder state mismatch "
                             "between encoder and decoder "
                             "in frame %ld: 0x%8lx vs 0x%8lx\n",
                         (long)count,
                         (unsigned long)enc_final_range[toggle^use_inbandfec],
                         (unsigned long)dec_final_range);
            fclose(fin);
            fclose(fout);
            return EXIT_FAILURE;
        }

        lost_prev = lost;

        /* count bits */
        bits += len[toggle]*8;
        bits_max = ( len[toggle]*8 > bits_max ) ? len[toggle]*8 : bits_max;
        if( count >= use_inbandfec ) {
            nrg = 0.0;
            if (!decode_only)
            {
                for ( k = 0; k < frame_size * channels; k++ ) {
                    nrg += in[ k ] * (double)in[ k ];
                }
            }
            if ( ( nrg / ( frame_size * channels ) ) > 1e5 ) {
                bits_act += len[toggle]*8;
                count_act++;
            }
            /* Variance */
            bits2 += len[toggle]*len[toggle]*64;
        }
        count++;
        toggle = (toggle + use_inbandfec) & 1;
    }
    fprintf (stderr, "average bitrate:             %7.3f kb/s\n",
                     1e-3*bits*sampling_rate/tot_samples);
    fprintf (stderr, "maximum bitrate:             %7.3f kb/s\n",
                     1e-3*bits_max*sampling_rate/frame_size);
    if (!decode_only)
       fprintf (stderr, "active bitrate:              %7.3f kb/s\n",
               1e-3*bits_act*sampling_rate/(frame_size*(double)count_act));
    fprintf (stderr, "bitrate standard deviation:  %7.3f kb/s\n",
            1e-3*sqrt(bits2/count - bits*bits/(count*(double)count))*sampling_rate/frame_size);
    /* Close any files to which intermediate results were stored */
    SILK_DEBUG_STORE_CLOSE_FILES
    silk_TimerSave("opus_timing.txt");
    opus_encoder_destroy(enc);
    opus_decoder_destroy(dec);
    free(data[0]);
    if (use_inbandfec)
        free(data[1]);
    fclose(fin);
    fclose(fout);
    free(in);
    free(out);
    free(fbytes);
    return EXIT_SUCCESS;
}
