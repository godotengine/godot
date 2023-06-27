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
    last mod: $Id$

 ********************************************************************/

#include <limits.h>
#if !defined(_decint_H)
# define _decint_H (1)
# include "theora/theoradec.h"
# include "state.h"
# include "bitpack.h"
# include "huffdec.h"
# include "dequant.h"

typedef struct th_setup_info         oc_setup_info;
typedef struct oc_dec_opt_vtable     oc_dec_opt_vtable;
typedef struct oc_dec_pipeline_state oc_dec_pipeline_state;
typedef struct th_dec_ctx            oc_dec_ctx;



/*Decoder-specific accelerated functions.*/
# if defined(OC_C64X_ASM)
#  include "c64x/c64xdec.h"
# endif

# if !defined(oc_dec_accel_init)
#  define oc_dec_accel_init oc_dec_accel_init_c
# endif
# if defined(OC_DEC_USE_VTABLE)
#  if !defined(oc_dec_dc_unpredict_mcu_plane)
#   define oc_dec_dc_unpredict_mcu_plane(_dec,_pipe,_pli) \
 ((*(_dec)->opt_vtable.dc_unpredict_mcu_plane)(_dec,_pipe,_pli))
#  endif
# else
#  if !defined(oc_dec_dc_unpredict_mcu_plane)
#   define oc_dec_dc_unpredict_mcu_plane oc_dec_dc_unpredict_mcu_plane_c
#  endif
# endif



/*Constants for the packet-in state machine specific to the decoder.*/

/*Next packet to read: Data packet.*/
#define OC_PACKET_DATA (0)



struct th_setup_info{
  /*The Huffman codes.*/
  ogg_int16_t   *huff_tables[TH_NHUFFMAN_TABLES];
  /*The quantization parameters.*/
  th_quant_info  qinfo;
};



/*Decoder specific functions with accelerated variants.*/
struct oc_dec_opt_vtable{
  void (*dc_unpredict_mcu_plane)(oc_dec_ctx *_dec,
   oc_dec_pipeline_state *_pipe,int _pli);
};



struct oc_dec_pipeline_state{
  /*Decoded DCT coefficients.
    These are placed here instead of on the stack so that they can persist
     between blocks, which makes clearing them back to zero much faster when
     only a few non-zero coefficients were decoded.
    It requires at least 65 elements because the zig-zag index array uses the
     65th element as a dumping ground for out-of-range indices to protect us
     from buffer overflow.
    We make it fully twice as large so that the second half can serve as the
     reconstruction buffer, which saves passing another parameter to all the
     acceleration functios.
    It also solves problems with 16-byte alignment for NEON on ARM.
    gcc (as of 4.2.1) only seems to be able to give stack variables 8-byte
     alignment, and silently produces incorrect results if you ask for 16.
    Finally, keeping it off the stack means there's less likely to be a data
     hazard beween the NEON co-processor and the regular ARM core, which avoids
     unnecessary stalls.*/
  OC_ALIGN16(ogg_int16_t dct_coeffs[128]);
  OC_ALIGN16(signed char bounding_values[256]);
  ptrdiff_t           ti[3][64];
  ptrdiff_t           ebi[3][64];
  ptrdiff_t           eob_runs[3][64];
  const ptrdiff_t    *coded_fragis[3];
  const ptrdiff_t    *uncoded_fragis[3];
  ptrdiff_t           ncoded_fragis[3];
  ptrdiff_t           nuncoded_fragis[3];
  const ogg_uint16_t *dequant[3][3][2];
  int                 fragy0[3];
  int                 fragy_end[3];
  int                 pred_last[3][4];
  int                 mcu_nvfrags;
  int                 loop_filter;
  int                 pp_level;
};


struct th_dec_ctx{
  /*Shared encoder/decoder state.*/
  oc_theora_state        state;
  /*Whether or not packets are ready to be emitted.
    This takes on negative values while there are remaining header packets to
     be emitted, reaches 0 when the codec is ready for input, and goes to 1
     when a frame has been processed and a data packet is ready.*/
  int                    packet_state;
  /*Buffer in which to assemble packets.*/
  oc_pack_buf            opb;
  /*Huffman decode trees.*/
  ogg_int16_t           *huff_tables[TH_NHUFFMAN_TABLES];
  /*The index of the first token in each plane for each coefficient.*/
  ptrdiff_t              ti0[3][64];
  /*The number of outstanding EOB runs at the start of each coefficient in each
     plane.*/
  ptrdiff_t              eob_runs[3][64];
  /*The DCT token lists.*/
  unsigned char         *dct_tokens;
  /*The extra bits associated with DCT tokens.*/
  unsigned char         *extra_bits;
  /*The number of dct tokens unpacked so far.*/
  int                    dct_tokens_count;
  /*The out-of-loop post-processing level.*/
  int                    pp_level;
  /*The DC scale used for out-of-loop deblocking.*/
  int                    pp_dc_scale[64];
  /*The sharpen modifier used for out-of-loop deringing.*/
  int                    pp_sharp_mod[64];
  /*The DC quantization index of each block.*/
  unsigned char         *dc_qis;
  /*The variance of each block.*/
  int                   *variances;
  /*The storage for the post-processed frame buffer.*/
  unsigned char         *pp_frame_data;
  /*Whether or not the post-processsed frame buffer has space for chroma.*/
  int                    pp_frame_state;
  /*The buffer used for the post-processed frame.
    Note that this is _not_ guaranteed to have the same strides and offsets as
     the reference frame buffers.*/
  th_ycbcr_buffer        pp_frame_buf;
  /*The striped decode callback function.*/
  th_stripe_callback     stripe_cb;
  oc_dec_pipeline_state  pipe;
# if defined(OC_DEC_USE_VTABLE)
  /*Table for decoder acceleration functions.*/
  oc_dec_opt_vtable      opt_vtable;
# endif
# if defined(HAVE_CAIRO)
  /*Output metrics for debugging.*/
  int                    telemetry_mbmode;
  int                    telemetry_mv;
  int                    telemetry_qi;
  int                    telemetry_bits;
  int                    telemetry_frame_bytes;
  int                    telemetry_coding_bytes;
  int                    telemetry_mode_bytes;
  int                    telemetry_mv_bytes;
  int                    telemetry_qi_bytes;
  int                    telemetry_dc_bytes;
  unsigned char         *telemetry_frame_data;
# endif
};

/*Default pure-C implementations of decoder-specific accelerated functions.*/
void oc_dec_accel_init_c(oc_dec_ctx *_dec);

void oc_dec_dc_unpredict_mcu_plane_c(oc_dec_ctx *_dec,
 oc_dec_pipeline_state *_pipe,int _pli);

#endif
