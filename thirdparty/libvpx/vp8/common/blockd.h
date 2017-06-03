/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef VP8_COMMON_BLOCKD_H_
#define VP8_COMMON_BLOCKD_H_

void vpx_log(const char *format, ...);

#include "vpx_config.h"
#include "vpx_scale/yv12config.h"
#include "mv.h"
#include "treecoder.h"
#include "vpx_ports/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/*#define DCPRED 1*/
#define DCPREDSIMTHRESH 0
#define DCPREDCNTTHRESH 3

#define MB_FEATURE_TREE_PROBS   3
#define MAX_MB_SEGMENTS         4

#define MAX_REF_LF_DELTAS       4
#define MAX_MODE_LF_DELTAS      4

/* Segment Feature Masks */
#define SEGMENT_DELTADATA   0
#define SEGMENT_ABSDATA     1

typedef struct
{
    int r, c;
} POS;

#define PLANE_TYPE_Y_NO_DC    0
#define PLANE_TYPE_Y2         1
#define PLANE_TYPE_UV         2
#define PLANE_TYPE_Y_WITH_DC  3


typedef char ENTROPY_CONTEXT;
typedef struct
{
    ENTROPY_CONTEXT y1[4];
    ENTROPY_CONTEXT u[2];
    ENTROPY_CONTEXT v[2];
    ENTROPY_CONTEXT y2;
} ENTROPY_CONTEXT_PLANES;

extern const unsigned char vp8_block2left[25];
extern const unsigned char vp8_block2above[25];

#define VP8_COMBINEENTROPYCONTEXTS( Dest, A, B) \
    Dest = (A)+(B);


typedef enum
{
    KEY_FRAME = 0,
    INTER_FRAME = 1
} FRAME_TYPE;

typedef enum
{
    DC_PRED,            /* average of above and left pixels */
    V_PRED,             /* vertical prediction */
    H_PRED,             /* horizontal prediction */
    TM_PRED,            /* Truemotion prediction */
    B_PRED,             /* block based prediction, each block has its own prediction mode */

    NEARESTMV,
    NEARMV,
    ZEROMV,
    NEWMV,
    SPLITMV,

    MB_MODE_COUNT
} MB_PREDICTION_MODE;

/* Macroblock level features */
typedef enum
{
    MB_LVL_ALT_Q = 0,               /* Use alternate Quantizer .... */
    MB_LVL_ALT_LF = 1,              /* Use alternate loop filter value... */
    MB_LVL_MAX = 2                  /* Number of MB level features supported */

} MB_LVL_FEATURES;

/* Segment Feature Masks */
#define SEGMENT_ALTQ    0x01
#define SEGMENT_ALT_LF  0x02

#define VP8_YMODES  (B_PRED + 1)
#define VP8_UV_MODES (TM_PRED + 1)

#define VP8_MVREFS (1 + SPLITMV - NEARESTMV)

typedef enum
{
    B_DC_PRED,          /* average of above and left pixels */
    B_TM_PRED,

    B_VE_PRED,           /* vertical prediction */
    B_HE_PRED,           /* horizontal prediction */

    B_LD_PRED,
    B_RD_PRED,

    B_VR_PRED,
    B_VL_PRED,
    B_HD_PRED,
    B_HU_PRED,

    LEFT4X4,
    ABOVE4X4,
    ZERO4X4,
    NEW4X4,

    B_MODE_COUNT
} B_PREDICTION_MODE;

#define VP8_BINTRAMODES (B_HU_PRED + 1)  /* 10 */
#define VP8_SUBMVREFS (1 + NEW4X4 - LEFT4X4)

/* For keyframes, intra block modes are predicted by the (already decoded)
   modes for the Y blocks to the left and above us; for interframes, there
   is a single probability table. */

union b_mode_info
{
    B_PREDICTION_MODE as_mode;
    int_mv mv;
};

typedef enum
{
    INTRA_FRAME = 0,
    LAST_FRAME = 1,
    GOLDEN_FRAME = 2,
    ALTREF_FRAME = 3,
    MAX_REF_FRAMES = 4
} MV_REFERENCE_FRAME;

typedef struct
{
    uint8_t mode, uv_mode;
    uint8_t ref_frame;
    uint8_t is_4x4;
    int_mv mv;

    uint8_t partitioning;
    uint8_t mb_skip_coeff;                                /* does this mb has coefficients at all, 1=no coefficients, 0=need decode tokens */
    uint8_t need_to_clamp_mvs;
    uint8_t segment_id;                  /* Which set of segmentation parameters should be used for this MB */
} MB_MODE_INFO;

typedef struct modeinfo
{
    MB_MODE_INFO mbmi;
    union b_mode_info bmi[16];
} MODE_INFO;

#if CONFIG_MULTI_RES_ENCODING
/* The mb-level information needed to be stored for higher-resolution encoder */
typedef struct
{
    MB_PREDICTION_MODE mode;
    MV_REFERENCE_FRAME ref_frame;
    int_mv mv;
    int dissim;    /* dissimilarity level of the macroblock */
} LOWER_RES_MB_INFO;

/* The frame-level information needed to be stored for higher-resolution
 *  encoder */
typedef struct
{
    FRAME_TYPE frame_type;
    int is_frame_dropped;
    // The frame rate for the lowest resolution.
    double low_res_framerate;
    /* The frame number of each reference frames */
    unsigned int low_res_ref_frames[MAX_REF_FRAMES];
    // The video frame counter value for the key frame, for lowest resolution.
    unsigned int key_frame_counter_value;
    LOWER_RES_MB_INFO *mb_info;
} LOWER_RES_FRAME_INFO;
#endif

typedef struct blockd
{
    short *qcoeff;
    short *dqcoeff;
    unsigned char  *predictor;
    short *dequant;

    int offset;
    char *eob;

    union b_mode_info bmi;
} BLOCKD;

typedef void (*vp8_subpix_fn_t)(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);

typedef struct macroblockd
{
    DECLARE_ALIGNED(16, unsigned char,  predictor[384]);
    DECLARE_ALIGNED(16, short, qcoeff[400]);
    DECLARE_ALIGNED(16, short, dqcoeff[400]);
    DECLARE_ALIGNED(16, char,  eobs[25]);

    DECLARE_ALIGNED(16, short,  dequant_y1[16]);
    DECLARE_ALIGNED(16, short,  dequant_y1_dc[16]);
    DECLARE_ALIGNED(16, short,  dequant_y2[16]);
    DECLARE_ALIGNED(16, short,  dequant_uv[16]);

    /* 16 Y blocks, 4 U, 4 V, 1 DC 2nd order block, each with 16 entries. */
    BLOCKD block[25];
    int fullpixel_mask;

    YV12_BUFFER_CONFIG pre; /* Filtered copy of previous frame reconstruction */
    YV12_BUFFER_CONFIG dst;

    MODE_INFO *mode_info_context;
    int mode_info_stride;

    FRAME_TYPE frame_type;

    int up_available;
    int left_available;

    unsigned char *recon_above[3];
    unsigned char *recon_left[3];
    int recon_left_stride[2];

    /* Y,U,V,Y2 */
    ENTROPY_CONTEXT_PLANES *above_context;
    ENTROPY_CONTEXT_PLANES *left_context;

    /* 0 indicates segmentation at MB level is not enabled. Otherwise the individual bits indicate which features are active. */
    unsigned char segmentation_enabled;

    /* 0 (do not update) 1 (update) the macroblock segmentation map. */
    unsigned char update_mb_segmentation_map;

    /* 0 (do not update) 1 (update) the macroblock segmentation feature data. */
    unsigned char update_mb_segmentation_data;

    /* 0 (do not update) 1 (update) the macroblock segmentation feature data. */
    unsigned char mb_segement_abs_delta;

    /* Per frame flags that define which MB level features (such as quantizer or loop filter level) */
    /* are enabled and when enabled the proabilities used to decode the per MB flags in MB_MODE_INFO */
    vp8_prob mb_segment_tree_probs[MB_FEATURE_TREE_PROBS];         /* Probability Tree used to code Segment number */

    signed char segment_feature_data[MB_LVL_MAX][MAX_MB_SEGMENTS];            /* Segment parameters */

    /* mode_based Loop filter adjustment */
    unsigned char mode_ref_lf_delta_enabled;
    unsigned char mode_ref_lf_delta_update;

    /* Delta values have the range +/- MAX_LOOP_FILTER */
    signed char last_ref_lf_deltas[MAX_REF_LF_DELTAS];                /* 0 = Intra, Last, GF, ARF */
    signed char ref_lf_deltas[MAX_REF_LF_DELTAS];                     /* 0 = Intra, Last, GF, ARF */
    signed char last_mode_lf_deltas[MAX_MODE_LF_DELTAS];                      /* 0 = BPRED, ZERO_MV, MV, SPLIT */
    signed char mode_lf_deltas[MAX_MODE_LF_DELTAS];                           /* 0 = BPRED, ZERO_MV, MV, SPLIT */

    /* Distance of MB away from frame edges */
    int mb_to_left_edge;
    int mb_to_right_edge;
    int mb_to_top_edge;
    int mb_to_bottom_edge;



    vp8_subpix_fn_t  subpixel_predict;
    vp8_subpix_fn_t  subpixel_predict8x4;
    vp8_subpix_fn_t  subpixel_predict8x8;
    vp8_subpix_fn_t  subpixel_predict16x16;

    void *current_bc;

    int corrupted;

#if ARCH_X86 || ARCH_X86_64
    /* This is an intermediate buffer currently used in sub-pixel motion search
     * to keep a copy of the reference area. This buffer can be used for other
     * purpose.
     */
    DECLARE_ALIGNED(32, unsigned char, y_buf[22*32]);
#endif
} MACROBLOCKD;


extern void vp8_build_block_doffsets(MACROBLOCKD *x);
extern void vp8_setup_block_dptrs(MACROBLOCKD *x);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP8_COMMON_BLOCKD_H_
