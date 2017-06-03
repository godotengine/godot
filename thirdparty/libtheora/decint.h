/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
 *                                                                  *
 ********************************************************************

  function:
    last mod: $Id: decint.h 16503 2009-08-22 18:14:02Z giles $

 ********************************************************************/

#include <limits.h>
#if !defined(_decint_H)
# define _decint_H (1)
# include "theora/theoradec.h"
# include "internal.h"
# include "bitpack.h"

typedef struct th_setup_info oc_setup_info;
typedef struct th_dec_ctx    oc_dec_ctx;

# include "huffdec.h"
# include "dequant.h"

/*Constants for the packet-in state machine specific to the decoder.*/

/*Next packet to read: Data packet.*/
#define OC_PACKET_DATA (0)



struct th_setup_info{
  /*The Huffman codes.*/
  oc_huff_node      *huff_tables[TH_NHUFFMAN_TABLES];
  /*The quantization parameters.*/
  th_quant_info  qinfo;
};



struct th_dec_ctx{
  /*Shared encoder/decoder state.*/
  oc_theora_state      state;
  /*Whether or not packets are ready to be emitted.
    This takes on negative values while there are remaining header packets to
     be emitted, reaches 0 when the codec is ready for input, and goes to 1
     when a frame has been processed and a data packet is ready.*/
  int                  packet_state;
  /*Buffer in which to assemble packets.*/
  oc_pack_buf          opb;
  /*Huffman decode trees.*/
  oc_huff_node        *huff_tables[TH_NHUFFMAN_TABLES];
  /*The index of the first token in each plane for each coefficient.*/
  ptrdiff_t            ti0[3][64];
  /*The number of outstanding EOB runs at the start of each coefficient in each
     plane.*/
  ptrdiff_t            eob_runs[3][64];
  /*The DCT token lists.*/
  unsigned char       *dct_tokens;
  /*The extra bits associated with DCT tokens.*/
  unsigned char       *extra_bits;
  /*The number of dct tokens unpacked so far.*/
  int                  dct_tokens_count;
  /*The out-of-loop post-processing level.*/
  int                  pp_level;
  /*The DC scale used for out-of-loop deblocking.*/
  int                  pp_dc_scale[64];
  /*The sharpen modifier used for out-of-loop deringing.*/
  int                  pp_sharp_mod[64];
  /*The DC quantization index of each block.*/
  unsigned char       *dc_qis;
  /*The variance of each block.*/
  int                 *variances;
  /*The storage for the post-processed frame buffer.*/
  unsigned char       *pp_frame_data;
  /*Whether or not the post-processsed frame buffer has space for chroma.*/
  int                  pp_frame_state;
  /*The buffer used for the post-processed frame.
    Note that this is _not_ guaranteed to have the same strides and offsets as
     the reference frame buffers.*/
  th_ycbcr_buffer      pp_frame_buf;
  /*The striped decode callback function.*/
  th_stripe_callback   stripe_cb;
# if defined(HAVE_CAIRO)
  /*Output metrics for debugging.*/
  int                  telemetry;
  int                  telemetry_mbmode;
  int                  telemetry_mv;
  int                  telemetry_qi;
  int                  telemetry_bits;
  int                  telemetry_frame_bytes;
  int                  telemetry_coding_bytes;
  int                  telemetry_mode_bytes;
  int                  telemetry_mv_bytes;
  int                  telemetry_qi_bytes;
  int                  telemetry_dc_bytes;
  unsigned char       *telemetry_frame_data;
# endif
};

#endif
