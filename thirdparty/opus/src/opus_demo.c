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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "opus.h"
#include "debug.h"
#include "opus_types.h"
#include "opus_private.h"
#include "opus_multistream.h"
#ifdef ENABLE_LOSSGEN
#include "lossgen.h"
#endif

#define MAX_PACKET 15000

#ifdef ENABLE_QEXT
#define MAX_SAMPLING_RATE 96000
#else
#define MAX_SAMPLING_RATE 48000
#endif

#ifdef USE_WEIGHTS_FILE
# if __unix__
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <unistd.h>
#  include <sys/stat.h>
/* When available, mmap() is preferable to reading the file, as it leads to
   better resource utilization, especially if multiple processes are using the same
   file (mapping will be shared in cache). */
void *load_blob(const char *filename, int *len) {
  int fd;
  void *data;
  struct stat st;
  if (stat(filename, &st)) {
     *len = 0;
     return NULL;
  }
  *len = st.st_size;
  fd = open(filename, O_RDONLY);
  if (fd<0) {
     *len = 0;
     return NULL;
  }
  data = mmap(NULL, *len, PROT_READ, MAP_SHARED, fd, 0);
  if (data == MAP_FAILED) {
     *len = 0;
     data = NULL;
  }
  close(fd);
  return data;
}
void free_blob(void *blob, int len) {
  if (blob) munmap(blob, len);
}
# else
void *load_blob(const char *filename, int *len) {
  FILE *file;
  void *data;
  file = fopen(filename, "r");
  if (file == NULL)
  {
    perror("could not open blob file");
    *len = 0;
    return NULL;
  }
  fseek(file, 0L, SEEK_END);
  *len = ftell(file);
  fseek(file, 0L, SEEK_SET);
  if (*len <= 0) {
     *len = 0;
     return NULL;
  }
  data = malloc(*len);
  if (!data) {
     *len = 0;
     return NULL;
  }
  *len = fread(data, 1, *len, file);
  return data;
}
void free_blob(void *blob, int len) {
  free(blob);
  (void)len;
}
# endif
#endif


void print_usage( char* argv[] )
{
    fprintf(stderr, "Usage: %s [-e] <application> <sampling rate (Hz)> <channels (1/2)> "
        "<bits per second>  [options] <input> <output>\n", argv[0]);
    fprintf(stderr, "       %s -d <sampling rate (Hz)> <channels (1/2)> "
        "[options] <input> <output>\n\n", argv[0]);
    fprintf(stderr, "application: voip | audio | restricted-lowdelay | restricted-silk | restricted-celt\n" );
    fprintf(stderr, "options:\n" );
    fprintf(stderr, "-e                   : only runs the encoder (output the bit-stream)\n" );
    fprintf(stderr, "-d                   : only runs the decoder (reads the bit-stream as input)\n" );
    fprintf(stderr, "-cbr                 : enable constant bitrate; default: variable bitrate\n" );
    fprintf(stderr, "-cvbr                : enable constrained variable bitrate; default: unconstrained\n" );
    fprintf(stderr, "-delayed-decision    : use look-ahead for speech/music detection (experts only); default: disabled\n" );
    fprintf(stderr, "-bandwidth <NB|MB|WB|SWB|FB> : audio bandwidth (from narrowband to fullband); default: sampling rate\n" );
    fprintf(stderr, "-framesize <2.5|5|10|20|40|60|80|100|120> : frame size in ms; default: 20 \n" );
    fprintf(stderr, "-max_payload <bytes> : maximum payload size in bytes, default: 1024\n" );
    fprintf(stderr, "-complexity <comp>   : encoder complexity, 0 (lowest) ... 10 (highest); default: 10\n" );
    fprintf(stderr, "-dec_complexity <comp> : decoder complexity, 0 (lowest) ... 10 (highest); default: 0\n" );
    fprintf(stderr, "-inbandfec           : enable SILK inband FEC\n" );
    fprintf(stderr, "-forcemono           : force mono encoding, even for stereo input\n" );
    fprintf(stderr, "-dtx                 : enable SILK DTX\n" );
    fprintf(stderr, "-loss <perc>         : optimize for loss percentage and simulate packet loss, in percent (0-100); default: 0\n" );
#ifdef ENABLE_LOSSGEN
    fprintf(stderr, "-sim_loss <perc>     : simulate realistic (bursty) packet loss from percentage, using generative model\n" );
#endif
    fprintf(stderr, "-lossfile <file>     : simulate packet loss, reading loss from file\n" );
    fprintf(stderr, "-dred <frames>       : add Deep REDundancy (in units of 10-ms frames)\n" );
    fprintf(stderr, "-enc_loss            : Apply loss on the encoder side (store empty packets)\n" );
#ifdef ENABLE_OSCE_BWE
    fprintf(stderr, "-enable_osce_bwe     : enable OSCE bandwidth extension for wideband signals (48 kHz sampling rate only), raises dec_complexity to 4\n");
#endif
#ifdef ENABLE_QEXT
    fprintf(stderr, "-qext                : enable QEXT\n" );
#endif
}

#define FORMAT_S16_LE 0
#define FORMAT_S24_LE 1
#define FORMAT_F32_LE 2

static const int format_size[3] = {2, 3, 4};

typedef union {
    opus_int32 i;
    float f;
} float_bits;

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

#define check_encoder_option(decode_only, opt) do {if (decode_only) {fprintf(stderr, "option %s is only for encoding\n", opt); goto failure;}} while(0)
#define check_decoder_option(encode_only, opt) do {if (encode_only) {fprintf(stderr, "option %s is only for decoding\n", opt); goto failure;}} while(0)

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

static const int silk_bw_switch_test[][4] = {
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND,       960, 1},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND,     960, 1},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND,     960, 1},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND,  960, 1},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_FULLBAND,       960, 1},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND,       960, 2},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND,     960, 2},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND,     960, 2},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND,  960, 2},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_FULLBAND,       960, 2},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND,       480, 1},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND,     480, 1},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND,     480, 1},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND,  480, 1},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_FULLBAND,       480, 1},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_WIDEBAND,       480, 2},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND,     480, 2},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND,     480, 2},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_SUPERWIDEBAND,  480, 2},
    {MODE_SILK_ONLY, OPUS_BANDWIDTH_FULLBAND,       480, 2}
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
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      120, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      240, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      480, 2},
      {MODE_CELT_ONLY, OPUS_BANDWIDTH_FULLBAND,      960, 2},
};

#if 0 /* This is a hack that replaces the normal encoder/decoder with the multistream version */
#define OpusEncoder OpusMSEncoder
#define OpusDecoder OpusMSDecoder
#define opus_encode24 opus_multistream_encode24
#define opus_decode24 opus_multistream_decode24
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


#ifdef ENABLE_OSCE_TRAINING_DATA
#define COMPLEXITY_MIN 0
#define COMPLEXITY_MAX 10

#define PACKET_LOSS_PERC_MIN 0
#define PACKET_LOSS_PERC_MAX 50
#define PACKET_LOSS_PERC_STEP 5

#define CBR_BITRATE_LIMIT 80000

#define NUM_BITRATES 102
static int bitrates[NUM_BITRATES] = {
        6000,  6060,  6120,  6180,  6240,  6300,  6360,  6420,  6480,
        6525,  6561,  6598,  6634,  6670,  6707,  6743,  6780,  6816,
        6853,  6889,  6926,  6962,  6999,  7042,  7085,  7128,  7171,
        7215,  7258,  7301,  7344,  7388,  7431,  7474,  7512,  7541,
        7570,  7599,  7628,  7657,  7686,  7715,  7744,  7773,  7802,
        7831,  7860,  7889,  7918,  7947,  7976,  8013,  8096,  8179,
        8262,  8344,  8427,  8511,  8605,  8699,  8792,  8886,  8980,
        9100,  9227,  9354,  9480,  9561,  9634,  9706,  9779,  9851,
        9924,  9996, 10161, 10330, 10499, 10698, 10898, 11124, 11378,
       11575, 11719, 11862, 12014, 12345, 12751, 13195, 13561, 13795,
       14069, 14671, 15403, 15790, 16371, 17399, 17968, 19382, 20468,
       22000, 32000, 64000
};

static int randint(int min, int max, int step)
{
    double r = ((double) rand())/ (RAND_MAX + 1.);
    int d;

    d = ((int) ((max + 1 - min) * r / step) * step) + min;

    return d;
}

static void new_random_setting(OpusEncoder *enc)
{
    int bitrate_bps;
    int complexity;
    int packet_loss_perc;
    int use_vbr;

    bitrate_bps = bitrates[randint(0, NUM_BITRATES - 1, 1)];
    complexity  = randint(COMPLEXITY_MIN, COMPLEXITY_MAX, 1);
    packet_loss_perc = randint(PACKET_LOSS_PERC_MIN, PACKET_LOSS_PERC_MAX, PACKET_LOSS_PERC_STEP);
    use_vbr = bitrate_bps < CBR_BITRATE_LIMIT ? 1 : randint(0, 1, 1);

    if (1)
    {
        printf("changing settings to %d\t%d\t%d\t%d\n", bitrate_bps, complexity, packet_loss_perc, use_vbr);
    }

    opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate_bps));
    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(complexity));
    opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC(packet_loss_perc));
    opus_encoder_ctl(enc, OPUS_SET_VBR(use_vbr));
}

#endif

int main(int argc, char *argv[])
{
    int err;
    char *inFile, *outFile;
    FILE *fin=NULL;
    FILE *fout=NULL;
    OpusEncoder *enc=NULL;
    OpusDecoder *dec=NULL;
    OpusDRED *dred=NULL;
    OpusDREDDecoder *dred_dec=NULL;
    int args;
    int len;
    int frame_size, channels;
    opus_int32 bitrate_bps=0;
    unsigned char *data = NULL;
    unsigned char *fbytes=NULL;
    opus_int32 sampling_rate;
    int use_vbr;
    int max_payload_bytes;
    int complexity;
    int dec_complexity;
    int use_inbandfec;
    int use_dtx;
    int forcechannels;
    int cvbr = 0;
    int packet_loss_perc;
#ifdef ENABLE_LOSSGEN
    float lossgen_perc = -1.f;
    LossGenState lossgen;
#endif
    opus_int32 count=0, count_act=0;
    int k;
    opus_int32 skip=0;
    int format=FORMAT_S16_LE;
    int stop=0;
    opus_int32 *in=NULL;
    opus_int32 *out=NULL;
    int application=OPUS_APPLICATION_AUDIO;
    double bits=0.0, bits_max=0.0, bits_act=0.0, bits2=0.0, nrg;
    double tot_samples=0;
    opus_uint64 tot_in, tot_out;
    int bandwidth=OPUS_AUTO;
    const char *bandwidth_string;
    int lost = 0, lost_prev = 1;
    opus_uint32 enc_final_range;
    opus_uint32 dec_final_range;
    int encode_only=0, decode_only=0;
    int max_frame_size = MAX_SAMPLING_RATE*2;
    size_t num_read;
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
    int ret = EXIT_FAILURE;
    int lost_count=0;
    FILE *packet_loss_file=NULL;
    int dred_duration=0;
    int ignore_extensions=0;
    int encoder_loss=0;
#ifdef ENABLE_QEXT
    int enable_qext=0;
#endif
#ifdef ENABLE_OSCE_TRAINING_DATA
    int silk_random_switching = 0;
    int silk_frame_counter = 0;
#endif
#if defined(ENABLE_OSCE) && defined(ENABLE_OSCE_BWE)
    int enable_osce_bwe = 0;
#endif
#ifdef USE_WEIGHTS_FILE
    int blob_len;
    void *blob_data;
    const char *filename = "weights_blob.bin";
    blob_data = load_blob(filename, &blob_len);
#endif

    if (argc < 5 )
    {
       print_usage( argv );
       goto failure;
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
       goto failure;
    }

    if (!decode_only)
    {
       if (strcmp(argv[args], "voip")==0)
          application = OPUS_APPLICATION_VOIP;
       else if (strcmp(argv[args], "restricted-lowdelay")==0)
          application = OPUS_APPLICATION_RESTRICTED_LOWDELAY;
       else if (strcmp(argv[args], "restricted-silk")==0)
          application = OPUS_APPLICATION_RESTRICTED_SILK;
       else if (strcmp(argv[args], "restricted-celt")==0)
          application = OPUS_APPLICATION_RESTRICTED_CELT;
       else if (strcmp(argv[args], "audio")!=0) {
          fprintf(stderr, "unknown application: %s\n", argv[args]);
          print_usage(argv);
          goto failure;
       }
       args++;
    }
    sampling_rate = (opus_int32)atol(argv[args]);
    args++;

    if (sampling_rate != 8000 && sampling_rate != 12000
     && sampling_rate != 16000 && sampling_rate != 24000
     && sampling_rate != 48000
#ifdef ENABLE_QEXT
     && sampling_rate != 96000
#endif
     )
    {
        fprintf(stderr, "Supported sampling rates are 8000, 12000, 16000, 24000"
#ifdef ENABLE_QEXT
              ", 48000 and 96000.\n");
#else
              " and 48000.\n");
#endif
        goto failure;
    }
    frame_size = sampling_rate/50;

    channels = atoi(argv[args]);
    args++;

    if (channels < 1 || channels > 2)
    {
        fprintf(stderr, "Opus_demo supports only 1 or 2 channels.\n");
        goto failure;
    }

    if (!decode_only)
    {
       bitrate_bps = (opus_int32)atol(argv[args]);
       args++;
    }

    /* defaults: */
    use_vbr = 1;
    max_payload_bytes = MAX_PACKET;
    complexity = 10;
    dec_complexity = 0;
    use_inbandfec = 0;
    forcechannels = OPUS_AUTO;
    use_dtx = 0;
    packet_loss_perc = 0;

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
                goto failure;
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
            else if (strcmp(argv[ args + 1 ], "80")==0)
                frame_size = 4*sampling_rate/50;
            else if (strcmp(argv[ args + 1 ], "100")==0)
                frame_size = 5*sampling_rate/50;
            else if (strcmp(argv[ args + 1 ], "120")==0)
                frame_size = 6*sampling_rate/50;
            else {
                fprintf(stderr, "Unsupported frame size: %s ms. "
                                "Supported are 2.5, 5, 10, 20, 40, 60, 80, 100, 120.\n",
                                argv[ args + 1 ]);
                goto failure;
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
        } else if( strcmp( argv[ args ], "-dec_complexity" ) == 0 ) {
            check_decoder_option(encode_only, "-dec_complexity");
            dec_complexity = atoi( argv[ args + 1 ] );
            args += 2;
        } else if( strcmp( argv[ args ], "-16" ) == 0 ) {
           format = FORMAT_S16_LE;
           args++;
        } else if( strcmp( argv[ args ], "-24" ) == 0 ) {
           format = FORMAT_S24_LE;
           args++;
        } else if( strcmp( argv[ args ], "-f32" ) == 0 ) {
           format = FORMAT_F32_LE;
           args++;
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
#ifdef ENABLE_LOSSGEN
        } else if( strcmp( argv[ args ], "-sim_loss" ) == 0 ) {
            lossgen_perc = atof( argv[ args + 1 ] );
            lossgen_init(&lossgen);
            args += 2;
#endif
        } else if( strcmp( argv[ args ], "-lossfile" ) == 0 ) {
            packet_loss_file = fopen( argv[ args + 1 ], "r" );
            if (packet_loss_file == NULL) {
                fprintf(stderr, "failed to open loss file %s\n", argv[ args + 1 ] );
                exit(1);
            }
            args += 2;
        } else if( strcmp( argv[ args ], "-dred" ) == 0 ) {
            dred_duration = atoi( argv[ args + 1 ] );
            args += 2;
        } else if( strcmp( argv[ args ], "-enc_loss") == 0 ) {
            check_encoder_option(decode_only, "-enc_loss");
            encoder_loss = 1;
            args++;
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
        } else if( strcmp( argv[ args ], "-silk_bw_switch_test" ) == 0 ) {
            check_encoder_option(decode_only, "-silk_bw_switch_test");
            mode_list = silk_bw_switch_test;
            nb_modes_in_list = 20;
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
        } else if( strcmp( argv[ args ], "-ignore_extensions" ) == 0 ) {
            check_decoder_option(encode_only, "-ignore_extensions");
            ignore_extensions = 1;
            args++;
#ifdef ENABLE_QEXT
        } else if( strcmp( argv[ args ], "-qext" ) == 0 ) {
            check_encoder_option(decode_only, "-qext");
            enable_qext = 1;
            args++;
#endif
#ifdef ENABLE_OSCE_TRAINING_DATA
        } else if( strcmp( argv[ args ], "-silk_random_switching" ) == 0 ){
            silk_random_switching = atoi( argv[ args + 1 ] );
            printf("switching encoding parameters every %dth frame\n", silk_random_switching);
            args += 2;
#endif
#if defined(ENABLE_OSCE) && defined(ENABLE_OSCE_BWE)
        } else if( strcmp( argv[ args ], "-enable_osce_bwe" ) == 0 ) {
            enable_osce_bwe = 1;
            args++;
#endif
        } else {
            printf( "Error: unrecognized setting: %s\n\n", argv[ args ] );
            print_usage( argv );
            goto failure;
        }
    }

    if (sweep_max)
       sweep_min = bitrate_bps;

    if (max_payload_bytes < 0 || max_payload_bytes > MAX_PACKET)
    {
        fprintf (stderr, "max_payload_bytes must be between 0 and %d\n",
                          MAX_PACKET);
        goto failure;
    }

    inFile = argv[argc-2];
    fin = fopen(inFile, "rb");
    if (!fin)
    {
        fprintf (stderr, "Could not open input file %s\n", argv[argc-2]);
        goto failure;
    }
    if (mode_list)
    {
       int size;
       int sample_size=2;
       if (format == FORMAT_S24_LE) sample_size=3;
       else if (format == FORMAT_F32_LE) sample_size=4;
       fseek(fin, 0, SEEK_END);
       size = ftell(fin);
       fprintf(stderr, "File size is %d bytes\n", size);
       fseek(fin, 0, SEEK_SET);
       mode_switch_time = size/sample_size/channels/nb_modes_in_list;
       fprintf(stderr, "Switching mode every %d samples\n", mode_switch_time);
    }

    outFile = argv[argc-1];
    fout = fopen(outFile, "wb+");
    if (!fout)
    {
        fprintf (stderr, "Could not open output file %s\n", argv[argc-1]);
        goto failure;
    }

    if (!decode_only)
    {
       enc = opus_encoder_create(sampling_rate, channels, application, &err);
       if (err != OPUS_OK)
       {
          fprintf(stderr, "Cannot create encoder: %s\n", opus_strerror(err));
          goto failure;
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
       opus_encoder_ctl(enc, OPUS_SET_LSB_DEPTH((format == FORMAT_S16_LE) ? 16 : 24));
       opus_encoder_ctl(enc, OPUS_SET_EXPERT_FRAME_DURATION(variable_duration));
       if (dred_duration > 0)
       {
          opus_encoder_ctl(enc, OPUS_SET_DRED_DURATION(dred_duration));
       }
#ifdef ENABLE_OSCE_TRAINING_DATA
       opus_encoder_ctl(enc, OPUS_SET_FORCE_MODE(MODE_SILK_ONLY));
       srand(0);
#endif
#ifdef ENABLE_QEXT
       opus_encoder_ctl(enc, OPUS_SET_QEXT(enable_qext));
#endif
    }
    if (!encode_only)
    {
       dec = opus_decoder_create(sampling_rate, channels, &err);
       if (err != OPUS_OK)
       {
          fprintf(stderr, "Cannot create decoder: %s\n", opus_strerror(err));
          goto failure;
       }
#ifdef ENABLE_OSCE_BWE
       if (enable_osce_bwe) {
            opus_decoder_ctl(dec, OPUS_SET_OSCE_BWE(1));
            if (dec_complexity < 4) {dec_complexity = 4;}
       }
#endif
       opus_decoder_ctl(dec, OPUS_SET_COMPLEXITY(dec_complexity));
       opus_decoder_ctl(dec, OPUS_SET_IGNORE_EXTENSIONS(ignore_extensions));
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
         bandwidth_string = "auto bandwidth";
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
                       "in %s with %d-sample frames.\n",
                       (long)sampling_rate, bitrate_bps*0.001,
                       bandwidth_string, frame_size);

    in = (opus_int32*)malloc(max_frame_size*channels*sizeof(opus_int32));
    out = (opus_int32*)malloc(max_frame_size*channels*sizeof(opus_int32));
    /* We need to allocate for 16-bit PCM data, but we store it as unsigned char. */
    fbytes = (unsigned char*)malloc(max_frame_size*channels*sizeof(opus_int32));
    data = (unsigned char*)calloc(max_payload_bytes,sizeof(unsigned char));
    if(delayed_decision)
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
       else if (frame_size==3*sampling_rate/50)
          variable_duration = OPUS_FRAMESIZE_60_MS;
       else if (frame_size==4*sampling_rate/50)
          variable_duration = OPUS_FRAMESIZE_80_MS;
       else if (frame_size==5*sampling_rate/50)
          variable_duration = OPUS_FRAMESIZE_100_MS;
       else
          variable_duration = OPUS_FRAMESIZE_120_MS;
       opus_encoder_ctl(enc, OPUS_SET_EXPERT_FRAME_DURATION(variable_duration));
       frame_size = 2*sampling_rate;
    }
    dred_dec = opus_dred_decoder_create(&err);
    dred = opus_dred_alloc(&err);
#ifdef USE_WEIGHTS_FILE
    if (enc) opus_encoder_ctl(enc, OPUS_SET_DNN_BLOB(blob_data, blob_len));
    if (dec) opus_decoder_ctl(dec, OPUS_SET_DNN_BLOB(blob_data, blob_len));
    if (dred_dec) opus_dred_decoder_ctl(dred_dec, OPUS_SET_DNN_BLOB(blob_data, blob_len));
#endif
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
            while (newsize < sampling_rate/25 && bitrate_bps-abs(sweep_bps) <= 3*12*sampling_rate/newsize)
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
            num_read = fread(ch, 1, 4, fin);
            if (num_read!=4)
                break;
            len = char_to_int(ch);
            if (len>max_payload_bytes || len<0)
            {
                fprintf(stderr, "Invalid payload length: %d\n",len);
                break;
            }
            num_read = fread(ch, 1, 4, fin);
            if (num_read!=4)
                break;
            enc_final_range = char_to_int(ch);
            num_read = fread(data, 1, len, fin);
            if (num_read!=(size_t)len)
            {
                fprintf(stderr, "Ran out of input, "
                                "expecting %d bytes got %d\n",
                                len,(int)num_read);
                break;
            }
        } else {
            int i;
            if (mode_list!=NULL)
            {
                opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(mode_list[curr_mode][1]));
                opus_encoder_ctl(enc, OPUS_SET_FORCE_MODE(mode_list[curr_mode][0]));
                opus_encoder_ctl(enc, OPUS_SET_FORCE_CHANNELS(mode_list[curr_mode][3]));
                frame_size = mode_list[curr_mode][2]*sampling_rate/48000;
            }
#ifdef ENABLE_OSCE_TRAINING_DATA
            if (silk_random_switching)
            {
                silk_frame_counter += 1;
                if (silk_frame_counter % silk_random_switching == 0) {
                    new_random_setting(enc);
                }
            }
#endif
            num_read = fread(fbytes, format_size[format]*channels, frame_size-remaining, fin);
            curr_read = (int)num_read;
            tot_in += curr_read;
            if (format == FORMAT_S16_LE) {
               for(i=0;i<frame_size*channels;i++)
               {
                  opus_int32 s;
                  s=fbytes[2*i+1]<<8|fbytes[2*i];
                  s=((s&0xFFFF)^0x8000)-0x8000;
                  in[i]=s*256;
               }
            } else if (format == FORMAT_S24_LE) {
               for(i=0;i<frame_size*channels;i++)
               {
                  opus_int32 s;
                  s=fbytes[3*i+2]<<16|fbytes[3*i+1]<<8|fbytes[3*i];
                  s=((s&0xFFFFFF)^0x800000)-0x800000;
                  in[i]=s;
               }
            } else if (format == FORMAT_F32_LE) {
               for(i=0;i<frame_size*channels;i++)
               {
                  float_bits s;
                  s.i=(opus_uint32)fbytes[4*i+3]<<24|fbytes[4*i+2]<<16|fbytes[4*i+1]<<8|fbytes[4*i];
                  in[i]=(int)floor(.5 + s.f*8388608);
               }
            }
            if (curr_read+remaining < frame_size)
            {
                for (i=(curr_read+remaining)*channels;i<frame_size*channels;i++)
                   in[i] = 0;
                if (encode_only || decode_only)
                   stop = 1;
            }
            len = opus_encode24(enc, in, frame_size, data, max_payload_bytes);
            if (len < 0)
            {
                fprintf (stderr, "opus_encode() returned %d\n", len);
                goto failure;
            }
            nb_encoded = opus_packet_get_samples_per_frame(data, sampling_rate)*opus_packet_get_nb_frames(data, len);
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
            opus_encoder_ctl(enc, OPUS_GET_FINAL_RANGE(&enc_final_range));
            curr_mode_count += frame_size;
            if (curr_mode_count > mode_switch_time && curr_mode < nb_modes_in_list-1)
            {
               curr_mode++;
               curr_mode_count = 0;
            }
        }

#if 0 /* This is for testing the padding code, do not enable by default */
        if (len<1275)
        {
           int new_len = len+rand()%(max_payload_bytes-len);
           if ((err = opus_packet_pad(data, len, new_len)) != OPUS_OK)
           {
              fprintf(stderr, "padding failed: %s\n", opus_strerror(err));
              goto failure;
           }
           len = new_len;
        }
#endif
        if (encode_only && !encoder_loss) {
            lost = 0;
        } else if (packet_loss_file != NULL) {
            if ( fscanf(packet_loss_file, "%d", &lost) != 1) {
                lost = 0;
            }
#ifdef ENABLE_LOSSGEN
        } else if (lossgen_perc >= 0) {
            lost = sample_loss(&lossgen, lossgen_perc*.01f);
#endif
        } else {
            lost = (packet_loss_perc>0) && (rand()%100 < packet_loss_perc);
        }
        if (encode_only)
        {
            unsigned char int_field[4];
            if (lost) {
               enc_final_range = 0;
               len = 0;
            }
            int_to_char(len, int_field);
            if (fwrite(int_field, 1, 4, fout) != 4) {
               fprintf(stderr, "Error writing.\n");
               goto failure;
            }
            int_to_char(enc_final_range, int_field);
            if (fwrite(int_field, 1, 4, fout) != 4) {
               fprintf(stderr, "Error writing.\n");
               goto failure;
            }
            if (fwrite(data, 1, len, fout) != (unsigned)len) {
               fprintf(stderr, "Error writing.\n");
               goto failure;
            }
            tot_samples += nb_encoded;
        } else {
            int fr;
            int run_decoder;
            int dred_input=0;
            int dred_end=0;
            if (len == 0) lost = 1;
            if (lost)
            {
               lost_count++;
               run_decoder = 0;
            } else {
               run_decoder= 1;
            }
            if (run_decoder)
                run_decoder += lost_count;
            if (!lost && lost_count > 0) {
                opus_int32 output_samples=0;
                opus_decoder_ctl(dec, OPUS_GET_LAST_PACKET_DURATION(&output_samples));
                dred_input = lost_count*output_samples;
                /* Only decode the amount we need to fill in the gap. */
                ret = opus_dred_parse(dred_dec, dred, data, len, IMIN(sampling_rate, IMAX(0, dred_input)), sampling_rate, &dred_end, 0);
                dred_input = ret > 0 ? ret : 0;
            }
            /* FIXME: Figure out how to trigger the decoder when the last packet of the file is lost. */
            for (fr=0;fr<run_decoder;fr++) {
                opus_int32 output_samples=0;
                if (fr == lost_count-1 && opus_packet_has_lbrr(data, len)) {
                   opus_decoder_ctl(dec, OPUS_GET_LAST_PACKET_DURATION(&output_samples));
                   output_samples = opus_decode24(dec, data, len, out, output_samples, 1);
                } else if (fr < lost_count) {
                   opus_decoder_ctl(dec, OPUS_GET_LAST_PACKET_DURATION(&output_samples));
                   if (dred_input > 0)
                      output_samples = opus_decoder_dred_decode24(dec, dred, (lost_count-fr)*output_samples, out, output_samples);
                   else
                      output_samples = opus_decode24(dec, NULL, 0, out, output_samples, 0);
                } else {
                   output_samples = max_frame_size;
                   output_samples = opus_decode24(dec, data, len, out, output_samples, 0);
                }
                if (output_samples>0)
                {
                    if (!decode_only && tot_out + output_samples > tot_in)
                    {
                       stop=1;
                       output_samples = (opus_int32)(tot_in - tot_out);
                    }
                    if (output_samples>skip) {
                       int i;
                       if (format == FORMAT_S16_LE) {
                          for(i=0;i<(output_samples-skip)*channels;i++)
                          {
                             opus_int32 s;
                             s=out[i+(skip*channels)];
                             if (s > 0x007fff00) s = 0x007fff00;
                             if (s < -0x007fff00) s = -0x007fff00;
                             s=(s+128)>>8;
                             fbytes[2*i]=s&0xFF;
                             fbytes[2*i+1]=(s>>8)&0xFF;
                          }
                       } else if (format == FORMAT_S24_LE) {
                          for(i=0;i<(output_samples-skip)*channels;i++)
                          {
                             opus_int32 s;
                             s=out[i+(skip*channels)];
                             if (s > 0x007fffff) s = 0x007fffff;
                             if (s < -0x007fffff) s = -0x007fffff;
                             fbytes[3*i]=s&0xFF;
                             fbytes[3*i+1]=(s>>8)&0xFF;
                             fbytes[3*i+2]=(s>>16)&0xFF;
                          }
                       } else if (format == FORMAT_F32_LE) {
                          for(i=0;i<(output_samples-skip)*channels;i++)
                          {
                             float_bits s;
                             s.f=out[i+(skip*channels)]*(1.f/8388608.f);
                             fbytes[4*i]=s.i&0xFF;
                             fbytes[4*i+1]=(s.i>>8)&0xFF;
                             fbytes[4*i+2]=(s.i>>16)&0xFF;
                             fbytes[4*i+3]=(s.i>>24)&0xFF;
                          }
                       }
                       if (fwrite(fbytes, format_size[format]*channels, output_samples-skip, fout) != (unsigned)(output_samples-skip)){
                          fprintf(stderr, "Error writing.\n");
                          goto failure;
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
        if( enc_final_range!=0  && !encode_only
         && !lost && !lost_prev
         && dec_final_range != enc_final_range ) {
            fprintf (stderr, "Error: Range coder state mismatch "
                             "between encoder and decoder "
                             "in frame %ld: 0x%8lx vs 0x%8lx\n",
                         (long)count,
                         (unsigned long)enc_final_range,
                         (unsigned long)dec_final_range);
            goto failure;
        }

        lost_prev = lost;
        if (!lost)
           lost_count = 0;
        if( count >= use_inbandfec ) {
            /* count bits */
            bits += len*8;
            bits_max = ( len*8 > bits_max ) ? len*8 : bits_max;
            bits2 += len*(double)len*64;
            if (!decode_only)
            {
                nrg = 0.0;
                for ( k = 0; k < frame_size * channels; k++ ) {
                    nrg += in[ k ] * (double)in[ k ];
                }
                nrg /= frame_size * channels;
                if( nrg > 1e5 ) {
                    bits_act += len*8;
                    count_act++;
                }
            }
        }
        count++;
    }

    if(decode_only && count > 0)
        frame_size = (int)(tot_samples / count);
    count -= use_inbandfec;
    if (tot_samples >= 1 && count > 0 && frame_size)
    {
       /* Print out bitrate statistics */
       double var;
       fprintf (stderr, "average bitrate:             %7.3f kb/s\n",
                        1e-3*bits*sampling_rate/tot_samples);
       fprintf (stderr, "maximum bitrate:             %7.3f kb/s\n",
                        1e-3*bits_max*sampling_rate/frame_size);
       if (!decode_only)
          fprintf (stderr, "active bitrate:              %7.3f kb/s\n",
                           1e-3*bits_act*sampling_rate/(1e-15+frame_size*(double)count_act));
       var = bits2/count - bits*bits/(count*(double)count);
       if (var < 0)
          var = 0;
       fprintf (stderr, "bitrate standard deviation:  %7.3f kb/s\n",
                        1e-3*sqrt(var)*sampling_rate/frame_size);
    } else {
       fprintf(stderr, "bitrate statistics are undefined\n");
    }
    silk_TimerSave("opus_timing.txt");
    ret = EXIT_SUCCESS;
failure:
    opus_encoder_destroy(enc);
    opus_decoder_destroy(dec);
    opus_dred_free(dred);
    opus_dred_decoder_destroy(dred_dec);
    free(data);
    if (fin)
        fclose(fin);
    if (fout)
        fclose(fout);
    free(in);
    free(out);
    free(fbytes);
#ifdef USE_WEIGHTS_FILE
    free_blob(blob_data, blob_len);
#endif
    return ret;
}
