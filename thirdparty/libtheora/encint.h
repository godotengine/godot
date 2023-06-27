/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

  function:
  last mod: $Id$

 ********************************************************************/
#if !defined(_encint_H)
# define _encint_H (1)
# include "theora/theoraenc.h"
# include "state.h"
# include "mathops.h"
# include "enquant.h"
# include "huffenc.h"
/*# define OC_COLLECT_METRICS*/



typedef oc_mv                         oc_mv2[2];

typedef struct oc_enc_opt_vtable      oc_enc_opt_vtable;
typedef struct oc_enc_opt_data        oc_enc_opt_data;
typedef struct oc_mb_enc_info         oc_mb_enc_info;
typedef struct oc_mode_scheme_chooser oc_mode_scheme_chooser;
typedef struct oc_fr_state            oc_fr_state;
typedef struct oc_qii_state           oc_qii_state;
typedef struct oc_enc_pipeline_state  oc_enc_pipeline_state;
typedef struct oc_mode_rd             oc_mode_rd;
typedef struct oc_iir_filter          oc_iir_filter;
typedef struct oc_frame_metrics       oc_frame_metrics;
typedef struct oc_rc_state            oc_rc_state;
typedef struct th_enc_ctx             oc_enc_ctx;
typedef struct oc_token_checkpoint    oc_token_checkpoint;



/*Encoder-specific accelerated functions.*/
# if defined(OC_X86_ASM)
#  if defined(_MSC_VER)
#   include "x86_vc/x86enc.h"
#  else
#   include "x86/x86enc.h"
#  endif
# endif
# if defined(OC_ARM_ASM)
#  include "arm/armenc.h"
# endif

# if !defined(oc_enc_accel_init)
#  define oc_enc_accel_init oc_enc_accel_init_c
# endif
# if defined(OC_ENC_USE_VTABLE)
#  if !defined(oc_enc_frag_sub)
#   define oc_enc_frag_sub(_enc,_diff,_src,_ref,_ystride) \
  ((*(_enc)->opt_vtable.frag_sub)(_diff,_src,_ref,_ystride))
#  endif
#  if !defined(oc_enc_frag_sub_128)
#   define oc_enc_frag_sub_128(_enc,_diff,_src,_ystride) \
  ((*(_enc)->opt_vtable.frag_sub_128)(_diff,_src,_ystride))
#  endif
#  if !defined(oc_enc_frag_sad)
#   define oc_enc_frag_sad(_enc,_src,_ref,_ystride) \
  ((*(_enc)->opt_vtable.frag_sad)(_src,_ref,_ystride))
#  endif
#  if !defined(oc_enc_frag_sad_thresh)
#   define oc_enc_frag_sad_thresh(_enc,_src,_ref,_ystride,_thresh) \
  ((*(_enc)->opt_vtable.frag_sad_thresh)(_src,_ref,_ystride,_thresh))
#  endif
#  if !defined(oc_enc_frag_sad2_thresh)
#   define oc_enc_frag_sad2_thresh(_enc,_src,_ref1,_ref2,_ystride,_thresh) \
  ((*(_enc)->opt_vtable.frag_sad2_thresh)(_src,_ref1,_ref2,_ystride,_thresh))
#  endif
#  if !defined(oc_enc_frag_intra_sad)
#   define oc_enc_frag_intra_sad(_enc,_src,_ystride) \
  ((*(_enc)->opt_vtable.frag_intra_sad)(_src,_ystride))
#  endif
#  if !defined(oc_enc_frag_satd)
#   define oc_enc_frag_satd(_enc,_dc,_src,_ref,_ystride) \
  ((*(_enc)->opt_vtable.frag_satd)(_dc,_src,_ref,_ystride))
#  endif
#  if !defined(oc_enc_frag_satd2)
#   define oc_enc_frag_satd2(_enc,_dc,_src,_ref1,_ref2,_ystride) \
  ((*(_enc)->opt_vtable.frag_satd2)(_dc,_src,_ref1,_ref2,_ystride))
#  endif
#  if !defined(oc_enc_frag_intra_satd)
#   define oc_enc_frag_intra_satd(_enc,_dc,_src,_ystride) \
  ((*(_enc)->opt_vtable.frag_intra_satd)(_dc,_src,_ystride))
#  endif
#  if !defined(oc_enc_frag_ssd)
#   define oc_enc_frag_ssd(_enc,_src,_ref,_ystride) \
  ((*(_enc)->opt_vtable.frag_ssd)(_src,_ref,_ystride))
#  endif
#  if !defined(oc_enc_frag_border_ssd)
#   define oc_enc_frag_border_ssd(_enc,_src,_ref,_ystride,_mask) \
  ((*(_enc)->opt_vtable.frag_border_ssd)(_src,_ref,_ystride,_mask))
#  endif
#  if !defined(oc_enc_frag_copy2)
#   define oc_enc_frag_copy2(_enc,_dst,_src1,_src2,_ystride) \
  ((*(_enc)->opt_vtable.frag_copy2)(_dst,_src1,_src2,_ystride))
#  endif
#  if !defined(oc_enc_enquant_table_init)
#   define oc_enc_enquant_table_init(_enc,_enquant,_dequant) \
  ((*(_enc)->opt_vtable.enquant_table_init)(_enquant,_dequant))
#  endif
#  if !defined(oc_enc_enquant_table_fixup)
#   define oc_enc_enquant_table_fixup(_enc,_enquant,_nqis) \
  ((*(_enc)->opt_vtable.enquant_table_fixup)(_enquant,_nqis))
#  endif
#  if !defined(oc_enc_quantize)
#   define oc_enc_quantize(_enc,_qdct,_dct,_dequant,_enquant) \
  ((*(_enc)->opt_vtable.quantize)(_qdct,_dct,_dequant,_enquant))
#  endif
#  if !defined(oc_enc_frag_recon_intra)
#   define oc_enc_frag_recon_intra(_enc,_dst,_ystride,_residue) \
  ((*(_enc)->opt_vtable.frag_recon_intra)(_dst,_ystride,_residue))
#  endif
#  if !defined(oc_enc_frag_recon_inter)
#   define oc_enc_frag_recon_inter(_enc,_dst,_src,_ystride,_residue) \
  ((*(_enc)->opt_vtable.frag_recon_inter)(_dst,_src,_ystride,_residue))
#  endif
#  if !defined(oc_enc_fdct8x8)
#   define oc_enc_fdct8x8(_enc,_y,_x) \
  ((*(_enc)->opt_vtable.fdct8x8)(_y,_x))
#  endif
# else
#  if !defined(oc_enc_frag_sub)
#   define oc_enc_frag_sub(_enc,_diff,_src,_ref,_ystride) \
  oc_enc_frag_sub_c(_diff,_src,_ref,_ystride)
#  endif
#  if !defined(oc_enc_frag_sub_128)
#   define oc_enc_frag_sub_128(_enc,_diff,_src,_ystride) \
  oc_enc_frag_sub_128_c(_diff,_src,_ystride)
#  endif
#  if !defined(oc_enc_frag_sad)
#   define oc_enc_frag_sad(_enc,_src,_ref,_ystride) \
  oc_enc_frag_sad_c(_src,_ref,_ystride)
#  endif
#  if !defined(oc_enc_frag_sad_thresh)
#   define oc_enc_frag_sad_thresh(_enc,_src,_ref,_ystride,_thresh) \
  oc_enc_frag_sad_thresh_c(_src,_ref,_ystride,_thresh)
#  endif
#  if !defined(oc_enc_frag_sad2_thresh)
#   define oc_enc_frag_sad2_thresh(_enc,_src,_ref1,_ref2,_ystride,_thresh) \
  oc_enc_frag_sad2_thresh_c(_src,_ref1,_ref2,_ystride,_thresh)
#  endif
#  if !defined(oc_enc_frag_intra_sad)
#   define oc_enc_frag_intra_sad(_enc,_src,_ystride) \
  oc_enc_frag_intra_sad_c(_src,_ystride)
#  endif
#  if !defined(oc_enc_frag_satd)
#   define oc_enc_frag_satd(_enc,_dc,_src,_ref,_ystride) \
  oc_enc_frag_satd_c(_dc,_src,_ref,_ystride)
#  endif
#  if !defined(oc_enc_frag_satd2)
#   define oc_enc_frag_satd2(_enc,_dc,_src,_ref1,_ref2,_ystride) \
  oc_enc_frag_satd2_c(_dc,_src,_ref1,_ref2,_ystride)
#  endif
#  if !defined(oc_enc_frag_intra_satd)
#   define oc_enc_frag_intra_satd(_enc,_dc,_src,_ystride) \
  oc_enc_frag_intra_satd_c(_dc,_src,_ystride)
#  endif
#  if !defined(oc_enc_frag_ssd)
#   define oc_enc_frag_ssd(_enc,_src,_ref,_ystride) \
  oc_enc_frag_ssd_c(_src,_ref,_ystride)
#  endif
#  if !defined(oc_enc_frag_border_ssd)
#   define oc_enc_frag_border_ssd(_enc,_src,_ref,_ystride,_mask) \
  oc_enc_frag_border_ssd_c(_src,_ref,_ystride,_mask)
#  endif
#  if !defined(oc_enc_frag_copy2)
#   define oc_enc_frag_copy2(_enc,_dst,_src1,_src2,_ystride) \
  oc_enc_frag_copy2_c(_dst,_src1,_src2,_ystride)
#  endif
#  if !defined(oc_enc_enquant_table_init)
#   define oc_enc_enquant_table_init(_enc,_enquant,_dequant) \
  oc_enc_enquant_table_init_c(_enquant,_dequant)
#  endif
#  if !defined(oc_enc_enquant_table_fixup)
#   define oc_enc_enquant_table_fixup(_enc,_enquant,_nqis) \
  oc_enc_enquant_table_fixup_c(_enquant,_nqis)
#  endif
#  if !defined(oc_enc_quantize)
#   define oc_enc_quantize(_enc,_qdct,_dct,_dequant,_enquant) \
  oc_enc_quantize_c(_qdct,_dct,_dequant,_enquant)
#  endif
#  if !defined(oc_enc_frag_recon_intra)
#   define oc_enc_frag_recon_intra(_enc,_dst,_ystride,_residue) \
  oc_frag_recon_intra_c(_dst,_ystride,_residue)
#  endif
#  if !defined(oc_enc_frag_recon_inter)
#   define oc_enc_frag_recon_inter(_enc,_dst,_src,_ystride,_residue) \
  oc_frag_recon_inter_c(_dst,_src,_ystride,_residue)
#  endif
#  if !defined(oc_enc_fdct8x8)
#   define oc_enc_fdct8x8(_enc,_y,_x) oc_enc_fdct8x8_c(_y,_x)
#  endif
# endif



/*Constants for the packet-out state machine specific to the encoder.*/

/*Next packet to emit: Data packet, but none are ready yet.*/
#define OC_PACKET_EMPTY (0)
/*Next packet to emit: Data packet, and one is ready.*/
#define OC_PACKET_READY (1)

/*All features enabled.*/
#define OC_SP_LEVEL_SLOW          (0)
/*Enable early skip.*/
#define OC_SP_LEVEL_EARLY_SKIP    (1)
/*Use analysis shortcuts, single quantizer, and faster tokenization.*/
#define OC_SP_LEVEL_FAST_ANALYSIS (2)
/*Use SAD instead of SATD*/
#define OC_SP_LEVEL_NOSATD        (3)
/*Disable motion compensation.*/
#define OC_SP_LEVEL_NOMC          (4)
/*Maximum valid speed level.*/
#define OC_SP_LEVEL_MAX           (4)


/*The number of extra bits of precision at which to store rate metrics.*/
# define OC_BIT_SCALE  (6)
/*The number of extra bits of precision at which to store RMSE metrics.
  This must be at least half OC_BIT_SCALE (rounded up).*/
# define OC_RMSE_SCALE (5)
/*The number of quantizer bins to partition statistics into.*/
# define OC_LOGQ_BINS  (8)
/*The number of SAD/SATD bins to partition statistics into.*/
# define OC_COMP_BINS   (24)
/*The number of bits of precision to drop from SAD and SATD scores
   to assign them to a bin.*/
# define OC_SAD_SHIFT  (6)
# define OC_SATD_SHIFT (9)

/*Masking is applied by scaling the D used in R-D optimization (via rd_scale)
   or the lambda parameter (via rd_iscale).
  These are only equivalent within a single block; when more than one block is
   being considered, the former is the interpretation used.*/

/*This must be at least 4 for OC_RD_SKIP_SCALE() to work below.*/
# define OC_RD_SCALE_BITS (12-OC_BIT_SCALE)
# define OC_RD_ISCALE_BITS (11)

/*This macro is applied to _ssd values with just 4 bits of headroom
   ((15-OC_RMSE_SCALE)*2+OC_BIT_SCALE+2); since we want to allow rd_scales as
   large as 16, and need additional fractional bits, our only recourse that
   doesn't lose precision on blocks with very small SSDs is to use a wider
   multiply.*/
# if LONG_MAX>2147483647
#  define OC_RD_SCALE(_ssd,_rd_scale) \
 ((unsigned)((unsigned long)(_ssd)*(_rd_scale) \
 +((1<<OC_RD_SCALE_BITS)>>1)>>OC_RD_SCALE_BITS))
# else
#  define OC_RD_SCALE(_ssd,_rd_scale) \
 (((_ssd)>>OC_RD_SCALE_BITS)*(_rd_scale) \
 +(((_ssd)&(1<<OC_RD_SCALE_BITS)-1)*(_rd_scale) \
 +((1<<OC_RD_SCALE_BITS)>>1)>>OC_RD_SCALE_BITS))
# endif
# define OC_RD_SKIP_SCALE(_ssd,_rd_scale) \
 ((_ssd)*(_rd_scale)+((1<<OC_RD_SCALE_BITS-4)>>1)>>OC_RD_SCALE_BITS-4)
# define OC_RD_ISCALE(_lambda,_rd_iscale) \
 ((_lambda)*(_rd_iscale)+((1<<OC_RD_ISCALE_BITS)>>1)>>OC_RD_ISCALE_BITS)


/*The bits used for each of the MB mode codebooks.*/
extern const unsigned char OC_MODE_BITS[2][OC_NMODES];

/*The bits used for each of the MV codebooks.*/
extern const unsigned char OC_MV_BITS[2][64];

/*The minimum value that can be stored in a SB run for each codeword.
  The last entry is the upper bound on the length of a single SB run.*/
extern const ogg_uint16_t  OC_SB_RUN_VAL_MIN[8];
/*The bits used for each SB run codeword.*/
extern const unsigned char OC_SB_RUN_CODE_NBITS[7];

/*The bits used for each block run length (starting with 1).*/
extern const unsigned char OC_BLOCK_RUN_CODE_NBITS[30];



/*Encoder specific functions with accelerated variants.*/
struct oc_enc_opt_vtable{
  void     (*frag_sub)(ogg_int16_t _diff[64],const unsigned char *_src,
   const unsigned char *_ref,int _ystride);
  void     (*frag_sub_128)(ogg_int16_t _diff[64],
   const unsigned char *_src,int _ystride);
  unsigned (*frag_sad)(const unsigned char *_src,
   const unsigned char *_ref,int _ystride);
  unsigned (*frag_sad_thresh)(const unsigned char *_src,
   const unsigned char *_ref,int _ystride,unsigned _thresh);
  unsigned (*frag_sad2_thresh)(const unsigned char *_src,
   const unsigned char *_ref1,const unsigned char *_ref2,int _ystride,
   unsigned _thresh);
  unsigned (*frag_intra_sad)(const unsigned char *_src,int _ystride);
  unsigned (*frag_satd)(int *_dc,const unsigned char *_src,
   const unsigned char *_ref,int _ystride);
  unsigned (*frag_satd2)(int *_dc,const unsigned char *_src,
   const unsigned char *_ref1,const unsigned char *_ref2,int _ystride);
  unsigned (*frag_intra_satd)(int *_dc,const unsigned char *_src,int _ystride);
  unsigned (*frag_ssd)(const unsigned char *_src,
   const unsigned char *_ref,int _ystride);
  unsigned (*frag_border_ssd)(const unsigned char *_src,
   const unsigned char *_ref,int _ystride,ogg_int64_t _mask);
  void     (*frag_copy2)(unsigned char *_dst,
   const unsigned char *_src1,const unsigned char *_src2,int _ystride);
  void     (*enquant_table_init)(void *_enquant,
   const ogg_uint16_t _dequant[64]);
  void     (*enquant_table_fixup)(void *_enquant[3][3][2],int _nqis);
  int      (*quantize)(ogg_int16_t _qdct[64],const ogg_int16_t _dct[64],
   const ogg_uint16_t _dequant[64],const void *_enquant);
  void     (*frag_recon_intra)(unsigned char *_dst,int _ystride,
   const ogg_int16_t _residue[64]);
  void     (*frag_recon_inter)(unsigned char *_dst,
   const unsigned char *_src,int _ystride,const ogg_int16_t _residue[64]);
  void     (*fdct8x8)(ogg_int16_t _y[64],const ogg_int16_t _x[64]);
};


/*Encoder specific data that varies according to which variants of the above
   functions are used.*/
struct oc_enc_opt_data{
  /*The size of a single quantizer table.
    This must be a multiple of enquant_table_alignment.*/
  size_t               enquant_table_size;
  /*The alignment required for the quantizer tables.
    This must be a positive power of two.*/
  int                  enquant_table_alignment;
};


void oc_enc_accel_init(oc_enc_ctx *_enc);



/*Encoder-specific macroblock information.*/
struct oc_mb_enc_info{
  /*Neighboring macro blocks that have MVs available from the current frame.*/
  unsigned      cneighbors[4];
  /*Neighboring macro blocks to use for MVs from the previous frame.*/
  unsigned      pneighbors[4];
  /*The number of current-frame neighbors.*/
  unsigned char ncneighbors;
  /*The number of previous-frame neighbors.*/
  unsigned char npneighbors;
  /*Flags indicating which MB modes have been refined.*/
  unsigned char refined;
  /*Motion vectors for a macro block for the current frame and the
     previous two frames.
    Each is a set of 2 vectors against OC_FRAME_GOLD and OC_FRAME_PREV, which
     can be used to estimate constant velocity and constant acceleration
     predictors.
    Uninitialized MVs are (0,0).*/
  oc_mv2        analysis_mv[3];
  /*Current unrefined analysis MVs.*/
  oc_mv         unref_mv[2];
  /*Unrefined block MVs.*/
  oc_mv         block_mv[4];
  /*Refined block MVs.*/
  oc_mv         ref_mv[4];
  /*Minimum motion estimation error from the analysis stage.*/
  ogg_uint16_t  error[2];
  /*MB error for half-pel refinement for each frame type.*/
  unsigned      satd[2];
  /*Block error for half-pel refinement.*/
  unsigned      block_satd[4];
};



/*State machine to estimate the opportunity cost of coding a MB mode.*/
struct oc_mode_scheme_chooser{
  /*Pointers to the a list containing the index of each mode in the mode
     alphabet used by each scheme.
    The first entry points to the dynamic scheme0_ranks, while the remaining 7
     point to the constant entries stored in OC_MODE_SCHEMES.*/
  const unsigned char *mode_ranks[8];
  /*The ranks for each mode when coded with scheme 0.
    These are optimized so that the more frequent modes have lower ranks.*/
  unsigned char        scheme0_ranks[OC_NMODES];
  /*The list of modes, sorted in descending order of frequency, that
    corresponds to the ranks above.*/
  unsigned char        scheme0_list[OC_NMODES];
  /*The number of times each mode has been chosen so far.*/
  unsigned             mode_counts[OC_NMODES];
  /*The list of mode coding schemes, sorted in ascending order of bit cost.*/
  unsigned char        scheme_list[8];
  /*The number of bits used by each mode coding scheme.*/
  ptrdiff_t            scheme_bits[8];
};


void oc_mode_scheme_chooser_init(oc_mode_scheme_chooser *_chooser);



/*State to track coded block flags and their bit cost.
  We use opportunity cost to measure the bits required to code or skip the next
   block, using the cheaper of the cost to code it fully or partially, so long
   as both are possible.*/
struct oc_fr_state{
  /*The number of bits required for the coded block flags so far this frame.*/
  ptrdiff_t  bits;
  /*The length of the current run for the partial super block flag, not
     including the current super block.*/
  unsigned   sb_partial_count:16;
  /*The length of the current run for the full super block flag, not
     including the current super block.*/
  unsigned   sb_full_count:16;
  /*The length of the coded block flag run when the current super block
     started.*/
  unsigned   b_coded_count_prev:6;
  /*The coded block flag when the current super block started.*/
  signed int b_coded_prev:2;
  /*The length of the current coded block flag run.*/
  unsigned   b_coded_count:6;
  /*The current coded block flag.*/
  signed int b_coded:2;
  /*The number of blocks processed in the current super block.*/
  unsigned   b_count:5;
  /*Whether or not it is cheaper to code the current super block partially,
     even if it could still be coded fully.*/
  unsigned   sb_prefer_partial:1;
  /*Whether the last super block was coded partially.*/
  signed int sb_partial:2;
  /*The number of bits required for the flags for the current super block.*/
  unsigned   sb_bits:6;
  /*Whether the last non-partial super block was coded fully.*/
  signed int sb_full:2;
};



struct oc_qii_state{
  ptrdiff_t  bits;
  unsigned   qi01_count:14;
  signed int qi01:2;
  unsigned   qi12_count:14;
  signed int qi12:2;
};



/*Temporary encoder state for the analysis pipeline.*/
struct oc_enc_pipeline_state{
  /*DCT coefficient storage.
    This is kept off the stack because a) gcc can't align things on the stack
     reliably on ARM, and b) it avoids (unintentional) data hazards between
     ARM and NEON code.*/
  OC_ALIGN16(ogg_int16_t dct_data[64*3]);
  OC_ALIGN16(signed char bounding_values[256]);
  oc_fr_state         fr[3];
  oc_qii_state        qs[3];
  /*Skip SSD storage for the current MCU in each plane.*/
  unsigned           *skip_ssd[3];
  /*Coded/uncoded fragment lists for each plane for the current MCU.*/
  ptrdiff_t          *coded_fragis[3];
  ptrdiff_t          *uncoded_fragis[3];
  ptrdiff_t           ncoded_fragis[3];
  ptrdiff_t           nuncoded_fragis[3];
  /*The starting fragment for the current MCU in each plane.*/
  ptrdiff_t           froffset[3];
  /*The starting row for the current MCU in each plane.*/
  int                 fragy0[3];
  /*The ending row for the current MCU in each plane.*/
  int                 fragy_end[3];
  /*The starting superblock for the current MCU in each plane.*/
  unsigned            sbi0[3];
  /*The ending superblock for the current MCU in each plane.*/
  unsigned            sbi_end[3];
  /*The number of tokens for zzi=1 for each color plane.*/
  int                 ndct_tokens1[3];
  /*The outstanding eob_run count for zzi=1 for each color plane.*/
  int                 eob_run1[3];
  /*Whether or not the loop filter is enabled.*/
  int                 loop_filter;
};



/*Statistics used to estimate R-D cost of a block in a given coding mode.
  See modedec.h for more details.*/
struct oc_mode_rd{
  /*The expected bits used by the DCT tokens, shifted by OC_BIT_SCALE.*/
  ogg_int16_t rate;
  /*The expected square root of the sum of squared errors, shifted by
     OC_RMSE_SCALE.*/
  ogg_int16_t rmse;
};

# if defined(OC_COLLECT_METRICS)
#  include "collect.h"
# endif



/*A 2nd order low-pass Bessel follower.
  We use this for rate control because it has fast reaction time, but is
   critically damped.*/
struct oc_iir_filter{
  ogg_int32_t c[2];
  ogg_int64_t g;
  ogg_int32_t x[2];
  ogg_int32_t y[2];
};



/*The 2-pass metrics associated with a single frame.*/
struct oc_frame_metrics{
  /*The log base 2 of the scale factor for this frame in Q24 format.*/
  ogg_int32_t   log_scale;
  /*The number of application-requested duplicates of this frame.*/
  unsigned      dup_count:31;
  /*The frame type from pass 1.*/
  unsigned      frame_type:1;
  /*The frame activity average from pass 1.*/
  unsigned      activity_avg;
};



/*Rate control state information.*/
struct oc_rc_state{
  /*The target average bits per frame.*/
  ogg_int64_t        bits_per_frame;
  /*The current buffer fullness (bits available to be used).*/
  ogg_int64_t        fullness;
  /*The target buffer fullness.
    This is where we'd like to be by the last keyframe the appears in the next
     buf_delay frames.*/
  ogg_int64_t        target;
  /*The maximum buffer fullness (total size of the buffer).*/
  ogg_int64_t        max;
  /*The log of the number of pixels in a frame in Q57 format.*/
  ogg_int64_t        log_npixels;
  /*The exponent used in the rate model in Q8 format.*/
  unsigned           exp[2];
  /*The number of frames to distribute the buffer usage over.*/
  int                buf_delay;
  /*The total drop count from the previous frame.
    This includes duplicates explicitly requested via the
     TH_ENCCTL_SET_DUP_COUNT API as well as frames we chose to drop ourselves.*/
  ogg_uint32_t       prev_drop_count;
  /*The log of an estimated scale factor used to obtain the real framerate, for
     VFR sources or, e.g., 12 fps content doubled to 24 fps, etc.*/
  ogg_int64_t        log_drop_scale;
  /*The log of estimated scale factor for the rate model in Q57 format.*/
  ogg_int64_t        log_scale[2];
  /*The log of the target quantizer level in Q57 format.*/
  ogg_int64_t        log_qtarget;
  /*Will we drop frames to meet bitrate target?*/
  unsigned char      drop_frames;
  /*Do we respect the maximum buffer fullness?*/
  unsigned char      cap_overflow;
  /*Can the reservoir go negative?*/
  unsigned char      cap_underflow;
  /*Second-order lowpass filters to track scale and VFR.*/
  oc_iir_filter      scalefilter[2];
  int                inter_count;
  int                inter_delay;
  int                inter_delay_target;
  oc_iir_filter      vfrfilter;
  /*Two-pass mode state.
    0 => 1-pass encoding.
    1 => 1st pass of 2-pass encoding.
    2 => 2nd pass of 2-pass encoding.*/
  int                twopass;
  /*Buffer for current frame metrics.*/
  unsigned char      twopass_buffer[48];
  /*The number of bytes in the frame metrics buffer.
    When 2-pass encoding is enabled, this is set to 0 after each frame is
     submitted, and must be non-zero before the next frame will be accepted.*/
  int                twopass_buffer_bytes;
  int                twopass_buffer_fill;
  /*Whether or not to force the next frame to be a keyframe.*/
  unsigned char      twopass_force_kf;
  /*The metrics for the previous frame.*/
  oc_frame_metrics   prev_metrics;
  /*The metrics for the current frame.*/
  oc_frame_metrics   cur_metrics;
  /*The buffered metrics for future frames.*/
  oc_frame_metrics  *frame_metrics;
  int                nframe_metrics;
  int                cframe_metrics;
  /*The index of the current frame in the circular metric buffer.*/
  int                frame_metrics_head;
  /*The frame count of each type (keyframes, delta frames, and dup frames);
     32 bits limits us to 2.268 years at 60 fps.*/
  ogg_uint32_t       frames_total[3];
  /*The number of frames of each type yet to be processed.*/
  ogg_uint32_t       frames_left[3];
  /*The sum of the scale values for each frame type.*/
  ogg_int64_t        scale_sum[2];
  /*The start of the window over which the current scale sums are taken.*/
  int                scale_window0;
  /*The end of the window over which the current scale sums are taken.*/
  int                scale_window_end;
  /*The frame count of each type in the current 2-pass window; this does not
     include dup frames.*/
  int                nframes[3];
  /*The total accumulated estimation bias.*/
  ogg_int64_t        rate_bias;
};


void oc_rc_state_init(oc_rc_state *_rc,oc_enc_ctx *_enc);
void oc_rc_state_clear(oc_rc_state *_rc);

void oc_enc_rc_resize(oc_enc_ctx *_enc);
int oc_enc_select_qi(oc_enc_ctx *_enc,int _qti,int _clamp);
void oc_enc_calc_lambda(oc_enc_ctx *_enc,int _frame_type);
int oc_enc_update_rc_state(oc_enc_ctx *_enc,
 long _bits,int _qti,int _qi,int _trial,int _droppable);
int oc_enc_rc_2pass_out(oc_enc_ctx *_enc,unsigned char **_buf);
int oc_enc_rc_2pass_in(oc_enc_ctx *_enc,unsigned char *_buf,size_t _bytes);



/*The internal encoder state.*/
struct th_enc_ctx{
  /*Shared encoder/decoder state.*/
  oc_theora_state          state;
  /*Buffer in which to assemble packets.*/
  oggpack_buffer           opb;
  /*Encoder-specific macroblock information.*/
  oc_mb_enc_info          *mb_info;
  /*DC coefficients after prediction.*/
  ogg_int16_t             *frag_dc;
  /*The list of coded macro blocks, in coded order.*/
  unsigned                *coded_mbis;
  /*The number of coded macro blocks.*/
  size_t                   ncoded_mbis;
  /*Whether or not packets are ready to be emitted.
    This takes on negative values while there are remaining header packets to
     be emitted, reaches 0 when the codec is ready for input, and becomes
     positive when a frame has been processed and data packets are ready.*/
  int                      packet_state;
  /*The maximum distance between keyframes.*/
  ogg_uint32_t             keyframe_frequency_force;
  /*The number of duplicates to produce for the next frame.*/
  ogg_uint32_t             dup_count;
  /*The number of duplicates remaining to be emitted for the current frame.*/
  ogg_uint32_t             nqueued_dups;
  /*The number of duplicates emitted for the last frame.*/
  ogg_uint32_t             prev_dup_count;
  /*The current speed level.*/
  int                      sp_level;
  /*Whether or not VP3 compatibility mode has been enabled.*/
  unsigned char            vp3_compatible;
  /*Whether or not any INTER frames have been coded.*/
  unsigned char            coded_inter_frame;
  /*Whether or not previous frame was dropped.*/
  unsigned char            prevframe_dropped;
  /*Stores most recently chosen Huffman tables for each frame type, DC and AC
     coefficients, and luma and chroma tokens.
    The actual Huffman table used for a given coefficient depends not only on
     the choice made here, but also its index in the zig-zag ordering.*/
  unsigned char            huff_idxs[2][2][2];
  /*Current count of bits used by each MV coding mode.*/
  size_t                   mv_bits[2];
  /*The mode scheme chooser for estimating mode coding costs.*/
  oc_mode_scheme_chooser   chooser;
  /*Temporary encoder state for the analysis pipeline.*/
  oc_enc_pipeline_state    pipe;
  /*The number of vertical super blocks in an MCU.*/
  int                      mcu_nvsbs;
  /*The SSD error for skipping each fragment in the current MCU.*/
  unsigned                *mcu_skip_ssd;
  /*The masking scale factors for chroma blocks in the current MCU.*/
  ogg_uint16_t            *mcu_rd_scale;
  ogg_uint16_t            *mcu_rd_iscale;
  /*The DCT token lists for each coefficient and each plane.*/
  unsigned char          **dct_tokens[3];
  /*The extra bits associated with each DCT token.*/
  ogg_uint16_t           **extra_bits[3];
  /*The number of DCT tokens for each coefficient for each plane.*/
  ptrdiff_t                ndct_tokens[3][64];
  /*Pending EOB runs for each coefficient for each plane.*/
  ogg_uint16_t             eob_run[3][64];
  /*The offset of the first DCT token for each coefficient for each plane.*/
  unsigned char            dct_token_offs[3][64];
  /*The last DC coefficient for each plane and reference frame.*/
  int                      dc_pred_last[3][4];
#if defined(OC_COLLECT_METRICS)
  /*Fragment SAD statistics for MB mode estimation metrics.*/
  unsigned                *frag_sad;
  /*Fragment SATD statistics for MB mode estimation metrics.*/
  unsigned                *frag_satd;
  /*Fragment SSD statistics for MB mode estimation metrics.*/
  unsigned                *frag_ssd;
#endif
  /*The R-D optimization parameter.*/
  int                      lambda;
  /*The average block "activity" of the previous frame.*/
  unsigned                 activity_avg;
  /*The average MB luma of the previous frame.*/
  unsigned                 luma_avg;
  /*The huffman tables in use.*/
  th_huff_code             huff_codes[TH_NHUFFMAN_TABLES][TH_NDCT_TOKENS];
  /*The quantization parameters in use.*/
  th_quant_info            qinfo;
  /*The original DC coefficients saved off from the dequatization tables.*/
  ogg_uint16_t             dequant_dc[64][3][2];
  /*Condensed dequantization tables.*/
  const ogg_uint16_t      *dequant[3][3][2];
  /*Condensed quantization tables.*/
  void                    *enquant[3][3][2];
  /*The full set of quantization tables.*/
  void                    *enquant_tables[64][3][2];
  /*Storage for the quantization tables.*/
  unsigned char           *enquant_table_data;
  /*An "average" quantizer for each frame type (INTRA or INTER) and qi value.
    This is used to parameterize the rate control decisions.
    They are kept in the log domain to simplify later processing.
    These are DCT domain quantizers, and so are scaled by an additional factor
     of 4 from the pixel domain.*/
  ogg_int64_t              log_qavg[2][64];
  /*The "average" quantizer futher partitioned by color plane.
    This is used to parameterize mode decision.
    These are DCT domain quantizers, and so are scaled by an additional factor
     of 4 from the pixel domain.*/
  ogg_int16_t              log_plq[64][3][2];
  /*The R-D scale factors to apply to chroma blocks for a given frame type
     (INTRA or INTER) and qi value.
    The first is the "D" modifier (rd_scale), while the second is the "lambda"
     modifier (rd_iscale).*/
  ogg_uint16_t             chroma_rd_scale[2][64][2];
  /*The interpolated mode decision R-D lookup tables for the current
     quantizers, color plane, and quantization type.*/
  oc_mode_rd               mode_rd[3][3][2][OC_COMP_BINS];
  /*The buffer state used to drive rate control.*/
  oc_rc_state              rc;
# if defined(OC_ENC_USE_VTABLE)
  /*Table for encoder acceleration functions.*/
  oc_enc_opt_vtable        opt_vtable;
# endif
  /*Table for encoder data used by accelerated functions.*/
  oc_enc_opt_data          opt_data;
};


void oc_enc_analyze_intra(oc_enc_ctx *_enc,int _recode);
int oc_enc_analyze_inter(oc_enc_ctx *_enc,int _allow_keyframe,int _recode);



/*Perform fullpel motion search for a single MB against both reference frames.*/
void oc_mcenc_search(oc_enc_ctx *_enc,int _mbi);
/*Refine a MB MV for one frame.*/
void oc_mcenc_refine1mv(oc_enc_ctx *_enc,int _mbi,int _frame);
/*Refine the block MVs.*/
void oc_mcenc_refine4mv(oc_enc_ctx *_enc,int _mbi);



/*Used to rollback a tokenlog transaction when we retroactively decide to skip
   a fragment.
  A checkpoint is taken right before each token is added.*/
struct oc_token_checkpoint{
  /*The color plane the token was added to.*/
  unsigned char pli;
  /*The zig-zag index the token was added to.*/
  unsigned char zzi;
  /*The outstanding EOB run count before the token was added.*/
  ogg_uint16_t  eob_run;
  /*The token count before the token was added.*/
  ptrdiff_t     ndct_tokens;
};



void oc_enc_tokenize_start(oc_enc_ctx *_enc);
int oc_enc_tokenize_ac(oc_enc_ctx *_enc,int _pli,ptrdiff_t _fragi,
 ogg_int16_t *_qdct_out,const ogg_int16_t *_qdct_in,
 const ogg_uint16_t *_dequant,const ogg_int16_t *_dct,
 int _zzi,oc_token_checkpoint **_stack,int _lambda,int _acmin);
int oc_enc_tokenize_ac_fast(oc_enc_ctx *_enc,int _pli,ptrdiff_t _fragi,
 ogg_int16_t *_qdct_out,const ogg_int16_t *_qdct_in,
 const ogg_uint16_t *_dequant,const ogg_int16_t *_dct,
 int _zzi,oc_token_checkpoint **_stack,int _lambda,int _acmin);
void oc_enc_tokenlog_rollback(oc_enc_ctx *_enc,
 const oc_token_checkpoint *_stack,int _n);
void oc_enc_pred_dc_frag_rows(oc_enc_ctx *_enc,
 int _pli,int _fragy0,int _frag_yend);
void oc_enc_tokenize_dc_frag_list(oc_enc_ctx *_enc,int _pli,
 const ptrdiff_t *_coded_fragis,ptrdiff_t _ncoded_fragis,
 int _prev_ndct_tokens1,int _prev_eob_run1);
void oc_enc_tokenize_finish(oc_enc_ctx *_enc);



/*Utility routine to encode one of the header packets.*/
int oc_state_flushheader(oc_theora_state *_state,int *_packet_state,
 oggpack_buffer *_opb,const th_quant_info *_qinfo,
 const th_huff_code _codes[TH_NHUFFMAN_TABLES][TH_NDCT_TOKENS],
 const char *_vendor,th_comment *_tc,ogg_packet *_op);



/*Default pure-C implementations of encoder-specific accelerated functions.*/
void oc_enc_accel_init_c(oc_enc_ctx *_enc);

void oc_enc_frag_sub_c(ogg_int16_t _diff[64],
 const unsigned char *_src,const unsigned char *_ref,int _ystride);
void oc_enc_frag_sub_128_c(ogg_int16_t _diff[64],
 const unsigned char *_src,int _ystride);
unsigned oc_enc_frag_sad_c(const unsigned char *_src,
 const unsigned char *_ref,int _ystride);
unsigned oc_enc_frag_sad_thresh_c(const unsigned char *_src,
 const unsigned char *_ref,int _ystride,unsigned _thresh);
unsigned oc_enc_frag_sad2_thresh_c(const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride,
 unsigned _thresh);
unsigned oc_enc_frag_intra_sad_c(const unsigned char *_src, int _ystride);
unsigned oc_enc_frag_satd_c(int *_dc,const unsigned char *_src,
 const unsigned char *_ref,int _ystride);
unsigned oc_enc_frag_satd2_c(int *_dc,const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride);
unsigned oc_enc_frag_intra_satd_c(int *_dc,
 const unsigned char *_src,int _ystride);
unsigned oc_enc_frag_ssd_c(const unsigned char *_src,
 const unsigned char *_ref,int _ystride);
unsigned oc_enc_frag_border_ssd_c(const unsigned char *_src,
 const unsigned char *_ref,int _ystride,ogg_int64_t _mask);
void oc_enc_frag_copy2_c(unsigned char *_dst,
 const unsigned char *_src1,const unsigned char *_src2,int _ystride);
void oc_enc_enquant_table_init_c(void *_enquant,
 const ogg_uint16_t _dequant[64]);
void oc_enc_enquant_table_fixup_c(void *_enquant[3][3][2],int _nqis);
int oc_enc_quantize_c(ogg_int16_t _qdct[64],const ogg_int16_t _dct[64],
 const ogg_uint16_t _dequant[64],const void *_enquant);
void oc_enc_fdct8x8_c(ogg_int16_t _y[64],const ogg_int16_t _x[64]);

#endif
