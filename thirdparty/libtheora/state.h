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
    last mod: $Id: internal.h 17337 2010-07-19 16:08:54Z tterribe $

 ********************************************************************/
#if !defined(_state_H)
# define _state_H (1)
# include "internal.h"
# include "huffman.h"
# include "quant.h"



/*A single quadrant of the map from a super block to fragment numbers.*/
typedef ptrdiff_t       oc_sb_map_quad[4];
/*A map from a super block to fragment numbers.*/
typedef oc_sb_map_quad  oc_sb_map[4];
/*A single plane of the map from a macro block to fragment numbers.*/
typedef ptrdiff_t       oc_mb_map_plane[4];
/*A map from a macro block to fragment numbers.*/
typedef oc_mb_map_plane oc_mb_map[3];
/*A motion vector.*/
typedef ogg_int16_t     oc_mv;

typedef struct oc_sb_flags              oc_sb_flags;
typedef struct oc_border_info           oc_border_info;
typedef struct oc_fragment              oc_fragment;
typedef struct oc_fragment_plane        oc_fragment_plane;
typedef struct oc_base_opt_vtable       oc_base_opt_vtable;
typedef struct oc_base_opt_data         oc_base_opt_data;
typedef struct oc_state_dispatch_vtable oc_state_dispatch_vtable;
typedef struct oc_theora_state          oc_theora_state;



/*Shared accelerated functions.*/
# if defined(OC_X86_ASM)
#  if defined(_MSC_VER)
#   include "x86_vc/x86int.h"
#  else
#   include "x86/x86int.h"
#  endif
# endif
# if defined(OC_ARM_ASM)
#  include "arm/armint.h"
# endif
# if defined(OC_C64X_ASM)
#  include "c64x/c64xint.h"
# endif

# if !defined(oc_state_accel_init)
#  define oc_state_accel_init oc_state_accel_init_c
# endif
# if defined(OC_STATE_USE_VTABLE)
#  if !defined(oc_frag_copy)
#   define oc_frag_copy(_state,_dst,_src,_ystride) \
  ((*(_state)->opt_vtable.frag_copy)(_dst,_src,_ystride))
#  endif
#  if !defined(oc_frag_copy_list)
#   define oc_frag_copy_list(_state,_dst_frame,_src_frame,_ystride, \
 _fragis,_nfragis,_frag_buf_offs) \
 ((*(_state)->opt_vtable.frag_copy_list)(_dst_frame,_src_frame,_ystride, \
  _fragis,_nfragis,_frag_buf_offs))
#  endif
#  if !defined(oc_frag_recon_intra)
#   define oc_frag_recon_intra(_state,_dst,_dst_ystride,_residue) \
  ((*(_state)->opt_vtable.frag_recon_intra)(_dst,_dst_ystride,_residue))
#  endif
#  if !defined(oc_frag_recon_inter)
#   define oc_frag_recon_inter(_state,_dst,_src,_ystride,_residue) \
  ((*(_state)->opt_vtable.frag_recon_inter)(_dst,_src,_ystride,_residue))
#  endif
#  if !defined(oc_frag_recon_inter2)
#   define oc_frag_recon_inter2(_state,_dst,_src1,_src2,_ystride,_residue) \
  ((*(_state)->opt_vtable.frag_recon_inter2)(_dst, \
   _src1,_src2,_ystride,_residue))
#  endif
# if !defined(oc_idct8x8)
#   define oc_idct8x8(_state,_y,_x,_last_zzi) \
  ((*(_state)->opt_vtable.idct8x8)(_y,_x,_last_zzi))
#  endif
#  if !defined(oc_state_frag_recon)
#   define oc_state_frag_recon(_state,_fragi, \
 _pli,_dct_coeffs,_last_zzi,_dc_quant) \
  ((*(_state)->opt_vtable.state_frag_recon)(_state,_fragi, \
   _pli,_dct_coeffs,_last_zzi,_dc_quant))
#  endif
#  if !defined(oc_loop_filter_init)
#   define oc_loop_filter_init(_state,_bv,_flimit) \
  ((*(_state)->opt_vtable.loop_filter_init)(_bv,_flimit))
#  endif
#  if !defined(oc_state_loop_filter_frag_rows)
#   define oc_state_loop_filter_frag_rows(_state, \
 _bv,_refi,_pli,_fragy0,_fragy_end) \
  ((*(_state)->opt_vtable.state_loop_filter_frag_rows)(_state, \
   _bv,_refi,_pli,_fragy0,_fragy_end))
#  endif
#  if !defined(oc_restore_fpu)
#   define oc_restore_fpu(_state) \
  ((*(_state)->opt_vtable.restore_fpu)())
#  endif
# else
#  if !defined(oc_frag_copy)
#   define oc_frag_copy(_state,_dst,_src,_ystride) \
  oc_frag_copy_c(_dst,_src,_ystride)
#  endif
#  if !defined(oc_frag_copy_list)
#   define oc_frag_copy_list(_state,_dst_frame,_src_frame,_ystride, \
 _fragis,_nfragis,_frag_buf_offs) \
  oc_frag_copy_list_c(_dst_frame,_src_frame,_ystride, \
  _fragis,_nfragis,_frag_buf_offs)
#  endif
#  if !defined(oc_frag_recon_intra)
#   define oc_frag_recon_intra(_state,_dst,_dst_ystride,_residue) \
  oc_frag_recon_intra_c(_dst,_dst_ystride,_residue)
#  endif
#  if !defined(oc_frag_recon_inter)
#   define oc_frag_recon_inter(_state,_dst,_src,_ystride,_residue) \
  oc_frag_recon_inter_c(_dst,_src,_ystride,_residue)
#  endif
#  if !defined(oc_frag_recon_inter2)
#   define oc_frag_recon_inter2(_state,_dst,_src1,_src2,_ystride,_residue) \
  oc_frag_recon_inter2_c(_dst,_src1,_src2,_ystride,_residue)
#  endif
#  if !defined(oc_idct8x8)
#   define oc_idct8x8(_state,_y,_x,_last_zzi) oc_idct8x8_c(_y,_x,_last_zzi)
#  endif
#  if !defined(oc_state_frag_recon)
#   define oc_state_frag_recon oc_state_frag_recon_c
#  endif
#  if !defined(oc_loop_filter_init)
#   define oc_loop_filter_init(_state,_bv,_flimit) \
  oc_loop_filter_init_c(_bv,_flimit)
#  endif
#  if !defined(oc_state_loop_filter_frag_rows)
#   define oc_state_loop_filter_frag_rows oc_state_loop_filter_frag_rows_c
#  endif
#  if !defined(oc_restore_fpu)
#   define oc_restore_fpu(_state) do{}while(0)
#  endif
# endif



/*A keyframe.*/
# define OC_INTRA_FRAME (0)
/*A predicted frame.*/
# define OC_INTER_FRAME (1)
/*A frame of unknown type (frame type decision has not yet been made).*/
# define OC_UNKWN_FRAME (-1)

/*The amount of padding to add to the reconstructed frame buffers on all
   sides.
  This is used to allow unrestricted motion vectors without special casing.
  This must be a multiple of 2.*/
# define OC_UMV_PADDING (16)

/*Frame classification indices.*/
/*The previous golden frame.*/
# define OC_FRAME_GOLD      (0)
/*The previous frame.*/
# define OC_FRAME_PREV      (1)
/*The current frame.*/
# define OC_FRAME_SELF      (2)
/*Used to mark uncoded fragments (for DC prediction).*/
# define OC_FRAME_NONE      (3)

/*The input or output buffer.*/
# define OC_FRAME_IO        (3)
/*Uncompressed prev golden frame.*/
# define OC_FRAME_GOLD_ORIG (4)
/*Uncompressed previous frame. */
# define OC_FRAME_PREV_ORIG (5)

/*Macroblock modes.*/
/*Macro block is invalid: It is never coded.*/
# define OC_MODE_INVALID        (-1)
/*Encoded difference from the same macro block in the previous frame.*/
# define OC_MODE_INTER_NOMV     (0)
/*Encoded with no motion compensated prediction.*/
# define OC_MODE_INTRA          (1)
/*Encoded difference from the previous frame offset by the given motion
   vector.*/
# define OC_MODE_INTER_MV       (2)
/*Encoded difference from the previous frame offset by the last coded motion
   vector.*/
# define OC_MODE_INTER_MV_LAST  (3)
/*Encoded difference from the previous frame offset by the second to last
   coded motion vector.*/
# define OC_MODE_INTER_MV_LAST2 (4)
/*Encoded difference from the same macro block in the previous golden
   frame.*/
# define OC_MODE_GOLDEN_NOMV    (5)
/*Encoded difference from the previous golden frame offset by the given motion
   vector.*/
# define OC_MODE_GOLDEN_MV      (6)
/*Encoded difference from the previous frame offset by the individual motion
   vectors given for each block.*/
# define OC_MODE_INTER_MV_FOUR  (7)
/*The number of (coded) modes.*/
# define OC_NMODES              (8)

/*Determines the reference frame used for a given MB mode.*/
# define OC_FRAME_FOR_MODE(_x) \
 OC_UNIBBLE_TABLE32(OC_FRAME_PREV,OC_FRAME_SELF,OC_FRAME_PREV,OC_FRAME_PREV, \
  OC_FRAME_PREV,OC_FRAME_GOLD,OC_FRAME_GOLD,OC_FRAME_PREV,(_x))

/*Constants for the packet state machine common between encoder and decoder.*/

/*Next packet to emit/read: Codec info header.*/
# define OC_PACKET_INFO_HDR    (-3)
/*Next packet to emit/read: Comment header.*/
# define OC_PACKET_COMMENT_HDR (-2)
/*Next packet to emit/read: Codec setup header.*/
# define OC_PACKET_SETUP_HDR   (-1)
/*No more packets to emit/read.*/
# define OC_PACKET_DONE        (INT_MAX)



#define OC_MV(_x,_y)         ((oc_mv)((_x)&0xFF|(_y)<<8))
#define OC_MV_X(_mv)         ((signed char)(_mv))
#define OC_MV_Y(_mv)         ((_mv)>>8)
#define OC_MV_ADD(_mv1,_mv2) \
  OC_MV(OC_MV_X(_mv1)+OC_MV_X(_mv2), \
   OC_MV_Y(_mv1)+OC_MV_Y(_mv2))
#define OC_MV_SUB(_mv1,_mv2) \
  OC_MV(OC_MV_X(_mv1)-OC_MV_X(_mv2), \
   OC_MV_Y(_mv1)-OC_MV_Y(_mv2))



/*Super blocks are 32x32 segments of pixels in a single color plane indexed
   in image order.
  Internally, super blocks are broken up into four quadrants, each of which
   contains a 2x2 pattern of blocks, each of which is an 8x8 block of pixels.
  Quadrants, and the blocks within them, are indexed in a special order called
   a "Hilbert curve" within the super block.

  In order to differentiate between the Hilbert-curve indexing strategy and
   the regular image order indexing strategy, blocks indexed in image order
   are called "fragments".
  Fragments are indexed in image order, left to right, then bottom to top,
   from Y' plane to Cb plane to Cr plane.

  The co-located fragments in all image planes corresponding to the location
   of a single quadrant of a luma plane super block form a macro block.
  Thus there is only a single set of macro blocks for all planes, each of which
   contains between 6 and 12 fragments, depending on the pixel format.
  Therefore macro block information is kept in a separate set of arrays from
   super blocks to avoid unused space in the other planes.
  The lists are indexed in super block order.
  That is, the macro block corresponding to the macro block mbi in (luma plane)
   super block sbi is at index (sbi<<2|mbi).
  Thus the number of macro blocks in each dimension is always twice the number
   of super blocks, even when only an odd number fall inside the coded frame.
  These "extra" macro blocks are just an artifact of our internal data layout,
   and not part of the coded stream; they are flagged with a negative MB mode.*/



/*Super block information.*/
struct oc_sb_flags{
  unsigned char coded_fully:1;
  unsigned char coded_partially:1;
  unsigned char quad_valid:4;
};



/*Information about a fragment which intersects the border of the displayable
   region.
  This marks which pixels belong to the displayable region.*/
struct oc_border_info{
  /*A bit mask marking which pixels are in the displayable region.
    Pixel (x,y) corresponds to bit (y<<3|x).*/
  ogg_int64_t mask;
  /*The number of pixels in the displayable region.
    This is always positive, and always less than 64.*/
  int         npixels;
};



/*Fragment information.*/
struct oc_fragment{
  /*A flag indicating whether or not this fragment is coded.*/
  unsigned   coded:1;
  /*A flag indicating that this entire fragment lies outside the displayable
     region of the frame.
    Note the contrast with an invalid macro block, which is outside the coded
     frame, not just the displayable one.
    There are no fragments outside the coded frame by construction.*/
  unsigned   invalid:1;
  /*The index of the quality index used for this fragment's AC coefficients.*/
  unsigned   qii:4;
  /*The index of the reference frame this fragment is predicted from.*/
  unsigned   refi:2;
  /*The mode of the macroblock this fragment belongs to.*/
  unsigned   mb_mode:3;
  /*The index of the associated border information for fragments which lie
     partially outside the displayable region.
    For fragments completely inside or outside this region, this is -1.
    Note that the C standard requires an explicit signed keyword for bitfield
     types, since some compilers may treat them as unsigned without it.*/
  signed int borderi:5;
  /*The prediction-corrected DC component.
    Note that the C standard requires an explicit signed keyword for bitfield
     types, since some compilers may treat them as unsigned without it.*/
  signed int dc:16;
};



/*A description of each fragment plane.*/
struct oc_fragment_plane{
  /*The number of fragments in the horizontal direction.*/
  int       nhfrags;
  /*The number of fragments in the vertical direction.*/
  int       nvfrags;
  /*The offset of the first fragment in the plane.*/
  ptrdiff_t froffset;
  /*The total number of fragments in the plane.*/
  ptrdiff_t nfrags;
  /*The number of super blocks in the horizontal direction.*/
  unsigned  nhsbs;
  /*The number of super blocks in the vertical direction.*/
  unsigned  nvsbs;
  /*The offset of the first super block in the plane.*/
  unsigned  sboffset;
  /*The total number of super blocks in the plane.*/
  unsigned  nsbs;
};


typedef void (*oc_state_loop_filter_frag_rows_func)(
 const oc_theora_state *_state,signed char _bv[256],int _refi,int _pli,
 int _fragy0,int _fragy_end);

/*The shared (encoder and decoder) functions that have accelerated variants.*/
struct oc_base_opt_vtable{
  void (*frag_copy)(unsigned char *_dst,
   const unsigned char *_src,int _ystride);
  void (*frag_copy_list)(unsigned char *_dst_frame,
   const unsigned char *_src_frame,int _ystride,
   const ptrdiff_t *_fragis,ptrdiff_t _nfragis,const ptrdiff_t *_frag_buf_offs);
  void (*frag_recon_intra)(unsigned char *_dst,int _ystride,
   const ogg_int16_t _residue[64]);
  void (*frag_recon_inter)(unsigned char *_dst,
   const unsigned char *_src,int _ystride,const ogg_int16_t _residue[64]);
  void (*frag_recon_inter2)(unsigned char *_dst,const unsigned char *_src1,
   const unsigned char *_src2,int _ystride,const ogg_int16_t _residue[64]);
  void (*idct8x8)(ogg_int16_t _y[64],ogg_int16_t _x[64],int _last_zzi);
  void (*state_frag_recon)(const oc_theora_state *_state,ptrdiff_t _fragi,
   int _pli,ogg_int16_t _dct_coeffs[128],int _last_zzi,ogg_uint16_t _dc_quant);
  void (*loop_filter_init)(signed char _bv[256],int _flimit);
  oc_state_loop_filter_frag_rows_func state_loop_filter_frag_rows;
  void (*restore_fpu)(void);
};

/*The shared (encoder and decoder) tables that vary according to which variants
   of the above functions are used.*/
struct oc_base_opt_data{
  const unsigned char *dct_fzig_zag;
};


/*State information common to both the encoder and decoder.*/
struct oc_theora_state{
  /*The stream information.*/
  th_info             info;
# if defined(OC_STATE_USE_VTABLE)
  /*Table for shared accelerated functions.*/
  oc_base_opt_vtable  opt_vtable;
# endif
  /*Table for shared data used by accelerated functions.*/
  oc_base_opt_data    opt_data;
  /*CPU flags to detect the presence of extended instruction sets.*/
  ogg_uint32_t        cpu_flags;
  /*The fragment plane descriptions.*/
  oc_fragment_plane   fplanes[3];
  /*The list of fragments, indexed in image order.*/
  oc_fragment        *frags;
  /*The the offset into the reference frame buffer to the upper-left pixel of
     each fragment.*/
  ptrdiff_t          *frag_buf_offs;
  /*The motion vector for each fragment.*/
  oc_mv              *frag_mvs;
  /*The total number of fragments in a single frame.*/
  ptrdiff_t           nfrags;
  /*The list of super block maps, indexed in image order.*/
  oc_sb_map          *sb_maps;
  /*The list of super block flags, indexed in image order.*/
  oc_sb_flags        *sb_flags;
  /*The total number of super blocks in a single frame.*/
  unsigned            nsbs;
  /*The fragments from each color plane that belong to each macro block.
    Fragments are stored in image order (left to right then top to bottom).
    When chroma components are decimated, the extra fragments have an index of
     -1.*/
  oc_mb_map          *mb_maps;
  /*The list of macro block modes.
    A negative number indicates the macro block lies entirely outside the
     coded frame.*/
  signed char        *mb_modes;
  /*The number of macro blocks in the X direction.*/
  unsigned            nhmbs;
  /*The number of macro blocks in the Y direction.*/
  unsigned            nvmbs;
  /*The total number of macro blocks.*/
  size_t              nmbs;
  /*The list of coded fragments, in coded order.
    Uncoded fragments are stored in reverse order from the end of the list.*/
  ptrdiff_t          *coded_fragis;
  /*The number of coded fragments in each plane.*/
  ptrdiff_t           ncoded_fragis[3];
  /*The total number of coded fragments.*/
  ptrdiff_t           ntotal_coded_fragis;
  /*The actual buffers used for the reference frames.*/
  th_ycbcr_buffer     ref_frame_bufs[6];
  /*The index of the buffers being used for each OC_FRAME_* reference frame.*/
  int                 ref_frame_idx[6];
  /*The storage for the reference frame buffers.
    This is just ref_frame_bufs[ref_frame_idx[i]][0].data, but is cached here
     for faster look-up.*/
  unsigned char      *ref_frame_data[6];
  /*The handle used to allocate the reference frame buffers.*/
  unsigned char      *ref_frame_handle;
  /*The strides for each plane in the reference frames.*/
  int                 ref_ystride[3];
  /*The number of unique border patterns.*/
  int                 nborders;
  /*The unique border patterns for all border fragments.
    The borderi field of fragments which straddle the border indexes this
     list.*/
  oc_border_info      borders[16];
  /*The frame number of the last keyframe.*/
  ogg_int64_t         keyframe_num;
  /*The frame number of the current frame.*/
  ogg_int64_t         curframe_num;
  /*The granpos of the current frame.*/
  ogg_int64_t         granpos;
  /*The type of the current frame.*/
  signed char         frame_type;
  /*The bias to add to the frame count when computing granule positions.*/
  unsigned char       granpos_bias;
  /*The number of quality indices used in the current frame.*/
  unsigned char       nqis;
  /*The quality indices of the current frame.*/
  unsigned char       qis[3];
  /*The dequantization tables, stored in zig-zag order, and indexed by
     qi, pli, qti, and zzi.*/
  ogg_uint16_t       *dequant_tables[64][3][2];
  OC_ALIGN16(oc_quant_table      dequant_table_data[64][3][2]);
  /*Loop filter strength parameters.*/
  unsigned char       loop_filter_limits[64];
};



/*The function type used to fill in the chroma plane motion vectors for a
   macro block when 4 different motion vectors are specified in the luma
   plane.
  _cbmvs: The chroma block-level motion vectors to fill in.
  _lmbmv: The luma macro-block level motion vector to fill in for use in
           prediction.
  _lbmvs: The luma block-level motion vectors.*/
typedef void (*oc_set_chroma_mvs_func)(oc_mv _cbmvs[4],const oc_mv _lbmvs[4]);



/*A table of functions used to fill in the Cb,Cr plane motion vectors for a
   macro block when 4 different motion vectors are specified in the luma
   plane.*/
extern const oc_set_chroma_mvs_func OC_SET_CHROMA_MVS_TABLE[TH_PF_NFORMATS];



int oc_state_init(oc_theora_state *_state,const th_info *_info,int _nrefs);
void oc_state_clear(oc_theora_state *_state);
void oc_state_accel_init_c(oc_theora_state *_state);
void oc_state_borders_fill_rows(oc_theora_state *_state,int _refi,int _pli,
 int _y0,int _yend);
void oc_state_borders_fill_caps(oc_theora_state *_state,int _refi,int _pli);
void oc_state_borders_fill(oc_theora_state *_state,int _refi);
void oc_state_fill_buffer_ptrs(oc_theora_state *_state,int _buf_idx,
 th_ycbcr_buffer _img);
int oc_state_mbi_for_pos(oc_theora_state *_state,int _mbx,int _mby);
int oc_state_get_mv_offsets(const oc_theora_state *_state,int _offsets[2],
 int _pli,oc_mv _mv);

void oc_loop_filter_init_c(signed char _bv[256],int _flimit);
void oc_state_loop_filter(oc_theora_state *_state,int _frame);
# if defined(OC_DUMP_IMAGES)
int oc_state_dump_frame(const oc_theora_state *_state,int _frame,
 const char *_suf);
# endif

/*Default pure-C implementations of shared accelerated functions.*/
void oc_frag_copy_c(unsigned char *_dst,
 const unsigned char *_src,int _src_ystride);
void oc_frag_copy_list_c(unsigned char *_dst_frame,
 const unsigned char *_src_frame,int _ystride,
 const ptrdiff_t *_fragis,ptrdiff_t _nfragis,const ptrdiff_t *_frag_buf_offs);
void oc_frag_recon_intra_c(unsigned char *_dst,int _dst_ystride,
 const ogg_int16_t _residue[64]);
void oc_frag_recon_inter_c(unsigned char *_dst,
 const unsigned char *_src,int _ystride,const ogg_int16_t _residue[64]);
void oc_frag_recon_inter2_c(unsigned char *_dst,const unsigned char *_src1,
 const unsigned char *_src2,int _ystride,const ogg_int16_t _residue[64]);
void oc_idct8x8_c(ogg_int16_t _y[64],ogg_int16_t _x[64],int _last_zzi);
void oc_state_frag_recon_c(const oc_theora_state *_state,ptrdiff_t _fragi,
 int _pli,ogg_int16_t _dct_coeffs[128],int _last_zzi,ogg_uint16_t _dc_quant);
void oc_state_loop_filter_frag_rows_c(const oc_theora_state *_state,
 signed char _bv[256],int _refi,int _pli,int _fragy0,int _fragy_end);
void oc_restore_fpu_c(void);

/*We need a way to call a few encoder functions without introducing a link-time
   dependency into the decoder, while still allowing the old alpha API which
   does not distinguish between encoder and decoder objects to be used.
  We do this by placing a function table at the start of the encoder object
   which can dispatch into the encoder library.
  We do a similar thing for the decoder in case we ever decide to split off a
   common base library.*/
typedef void (*oc_state_clear_func)(theora_state *_th);
typedef int (*oc_state_control_func)(theora_state *th,int _req,
 void *_buf,size_t _buf_sz);
typedef ogg_int64_t (*oc_state_granule_frame_func)(theora_state *_th,
 ogg_int64_t _granulepos);
typedef double (*oc_state_granule_time_func)(theora_state *_th,
 ogg_int64_t _granulepos);


struct oc_state_dispatch_vtable{
  oc_state_clear_func         clear;
  oc_state_control_func       control;
  oc_state_granule_frame_func granule_frame;
  oc_state_granule_time_func  granule_time;
};

#endif
