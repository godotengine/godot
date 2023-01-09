/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
 * Copyright 2009-2010 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef P_SHADER_TOKENS_H
#define P_SHADER_TOKENS_H

#ifdef __cplusplus
extern "C" {
#endif


struct tgsi_header
{
   unsigned HeaderSize : 8;
   unsigned BodySize   : 24;
};

struct tgsi_processor
{
   unsigned Processor  : 4;  /* PIPE_SHADER_ */
   unsigned Padding    : 28;
};

enum tgsi_token_type {
   TGSI_TOKEN_TYPE_DECLARATION,
   TGSI_TOKEN_TYPE_IMMEDIATE,
   TGSI_TOKEN_TYPE_INSTRUCTION,
   TGSI_TOKEN_TYPE_PROPERTY,
};

struct tgsi_token
{
   unsigned Type       : 4;  /**< TGSI_TOKEN_TYPE_x */
   unsigned NrTokens   : 8;  /**< UINT */
   unsigned Padding    : 20;
};

enum tgsi_file_type {
   TGSI_FILE_NULL,
   TGSI_FILE_CONSTANT,
   TGSI_FILE_INPUT,
   TGSI_FILE_OUTPUT,
   TGSI_FILE_TEMPORARY,
   TGSI_FILE_SAMPLER,
   TGSI_FILE_ADDRESS,
   TGSI_FILE_IMMEDIATE,
   TGSI_FILE_SYSTEM_VALUE,
   TGSI_FILE_IMAGE,
   TGSI_FILE_SAMPLER_VIEW,
   TGSI_FILE_BUFFER,
   TGSI_FILE_MEMORY,
   TGSI_FILE_CONSTBUF,
   TGSI_FILE_HW_ATOMIC,
   TGSI_FILE_COUNT,      /**< how many TGSI_FILE_ types */
};


#define TGSI_WRITEMASK_NONE     0x00
#define TGSI_WRITEMASK_X        0x01
#define TGSI_WRITEMASK_Y        0x02
#define TGSI_WRITEMASK_XY       0x03
#define TGSI_WRITEMASK_Z        0x04
#define TGSI_WRITEMASK_XZ       0x05
#define TGSI_WRITEMASK_YZ       0x06
#define TGSI_WRITEMASK_XYZ      0x07
#define TGSI_WRITEMASK_W        0x08
#define TGSI_WRITEMASK_XW       0x09
#define TGSI_WRITEMASK_YW       0x0A
#define TGSI_WRITEMASK_XYW      0x0B
#define TGSI_WRITEMASK_ZW       0x0C
#define TGSI_WRITEMASK_XZW      0x0D
#define TGSI_WRITEMASK_YZW      0x0E
#define TGSI_WRITEMASK_XYZW     0x0F

enum tgsi_interpolate_mode {
   TGSI_INTERPOLATE_CONSTANT,
   TGSI_INTERPOLATE_LINEAR,
   TGSI_INTERPOLATE_PERSPECTIVE,
   TGSI_INTERPOLATE_COLOR,          /* special color case for smooth/flat */
   TGSI_INTERPOLATE_COUNT,
};

enum tgsi_interpolate_loc {
   TGSI_INTERPOLATE_LOC_CENTER,
   TGSI_INTERPOLATE_LOC_CENTROID,
   TGSI_INTERPOLATE_LOC_SAMPLE,
   TGSI_INTERPOLATE_LOC_COUNT,
};

enum tgsi_memory_type {
   TGSI_MEMORY_TYPE_GLOBAL,         /* OpenCL global              */
   TGSI_MEMORY_TYPE_SHARED,         /* OpenCL local / GLSL shared */
   TGSI_MEMORY_TYPE_PRIVATE,        /* OpenCL private             */
   TGSI_MEMORY_TYPE_INPUT,          /* OpenCL kernel input params */
   TGSI_MEMORY_TYPE_COUNT,
};

struct tgsi_declaration
{
   unsigned Type        : 4;  /**< TGSI_TOKEN_TYPE_DECLARATION */
   unsigned NrTokens    : 8;  /**< UINT */
   unsigned File        : 4;  /**< one of TGSI_FILE_x */
   unsigned UsageMask   : 4;  /**< bitmask of TGSI_WRITEMASK_x flags */
   unsigned Dimension   : 1;  /**< any extra dimension info? */
   unsigned Semantic    : 1;  /**< BOOL, any semantic info? */
   unsigned Interpolate : 1;  /**< any interpolation info? */
   unsigned Invariant   : 1;  /**< invariant optimization? */
   unsigned Local       : 1;  /**< optimize as subroutine local variable? */
   unsigned Array       : 1;  /**< extra array info? */
   unsigned Atomic      : 1;  /**< atomic only? for TGSI_FILE_BUFFER */
   unsigned MemType     : 2;  /**< TGSI_MEMORY_TYPE_x for TGSI_FILE_MEMORY */
   unsigned Padding     : 3;
};

struct tgsi_declaration_range
{
   unsigned First   : 16; /**< UINT */
   unsigned Last    : 16; /**< UINT */
};

struct tgsi_declaration_dimension
{
   unsigned Index2D:16; /**< UINT */
   unsigned Padding:16;
};

struct tgsi_declaration_interp
{
   unsigned Interpolate : 4;   /**< one of TGSI_INTERPOLATE_x */
   unsigned Location    : 2;   /**< one of TGSI_INTERPOLATE_LOC_x */
   unsigned Padding     : 26;
};

enum tgsi_semantic {
   TGSI_SEMANTIC_POSITION,
   TGSI_SEMANTIC_COLOR,
   TGSI_SEMANTIC_BCOLOR,       /**< back-face color */
   TGSI_SEMANTIC_FOG,
   TGSI_SEMANTIC_PSIZE,
   TGSI_SEMANTIC_GENERIC,
   TGSI_SEMANTIC_NORMAL,
   TGSI_SEMANTIC_FACE,
   TGSI_SEMANTIC_EDGEFLAG,
   TGSI_SEMANTIC_PRIMID,
   TGSI_SEMANTIC_INSTANCEID,  /**< doesn't include start_instance */
   TGSI_SEMANTIC_VERTEXID,
   TGSI_SEMANTIC_STENCIL,
   TGSI_SEMANTIC_CLIPDIST,
   TGSI_SEMANTIC_CLIPVERTEX,
   TGSI_SEMANTIC_GRID_SIZE,   /**< grid size in blocks */
   TGSI_SEMANTIC_BLOCK_ID,    /**< id of the current block */
   TGSI_SEMANTIC_BLOCK_SIZE,  /**< block size in threads */
   TGSI_SEMANTIC_THREAD_ID,   /**< block-relative id of the current thread */
   TGSI_SEMANTIC_TEXCOORD,    /**< texture or sprite coordinates */
   TGSI_SEMANTIC_PCOORD,      /**< point sprite coordinate */
   TGSI_SEMANTIC_VIEWPORT_INDEX,  /**< viewport index */
   TGSI_SEMANTIC_LAYER,       /**< layer (rendertarget index) */
   TGSI_SEMANTIC_SAMPLEID,
   TGSI_SEMANTIC_SAMPLEPOS,
   TGSI_SEMANTIC_SAMPLEMASK,
   TGSI_SEMANTIC_INVOCATIONID,
   TGSI_SEMANTIC_VERTEXID_NOBASE,
   TGSI_SEMANTIC_BASEVERTEX,
   TGSI_SEMANTIC_PATCH,       /**< generic per-patch semantic */
   TGSI_SEMANTIC_TESSCOORD,   /**< coordinate being processed by tess */
   TGSI_SEMANTIC_TESSOUTER,   /**< outer tessellation levels */
   TGSI_SEMANTIC_TESSINNER,   /**< inner tessellation levels */
   TGSI_SEMANTIC_VERTICESIN,  /**< number of input vertices */
   TGSI_SEMANTIC_HELPER_INVOCATION,  /**< current invocation is helper */
   TGSI_SEMANTIC_BASEINSTANCE,
   TGSI_SEMANTIC_DRAWID,
   TGSI_SEMANTIC_WORK_DIM,    /**< opencl get_work_dim value */
   TGSI_SEMANTIC_SUBGROUP_SIZE,
   TGSI_SEMANTIC_SUBGROUP_INVOCATION,
   TGSI_SEMANTIC_SUBGROUP_EQ_MASK,
   TGSI_SEMANTIC_SUBGROUP_GE_MASK,
   TGSI_SEMANTIC_SUBGROUP_GT_MASK,
   TGSI_SEMANTIC_SUBGROUP_LE_MASK,
   TGSI_SEMANTIC_SUBGROUP_LT_MASK,
   TGSI_SEMANTIC_CS_USER_DATA_AMD,
   TGSI_SEMANTIC_VIEWPORT_MASK,
   TGSI_SEMANTIC_TESS_DEFAULT_OUTER_LEVEL, /**< from set_tess_state */
   TGSI_SEMANTIC_TESS_DEFAULT_INNER_LEVEL, /**< from set_tess_state */
   TGSI_SEMANTIC_COUNT,       /**< number of semantic values */
};

struct tgsi_declaration_semantic
{
   unsigned Name           : 8;  /**< one of TGSI_SEMANTIC_x */
   unsigned Index          : 16; /**< UINT */
   unsigned StreamX        : 2; /**< vertex stream (for GS output) */
   unsigned StreamY        : 2;
   unsigned StreamZ        : 2;
   unsigned StreamW        : 2;
};

struct tgsi_declaration_image {
   unsigned Resource    : 8; /**< one of TGSI_TEXTURE_ */
   unsigned Raw         : 1;
   unsigned Writable    : 1;
   unsigned Format      : 10; /**< one of PIPE_FORMAT_ */
   unsigned Padding     : 12;
};

enum tgsi_return_type {
   TGSI_RETURN_TYPE_UNORM = 0,
   TGSI_RETURN_TYPE_SNORM,
   TGSI_RETURN_TYPE_SINT,
   TGSI_RETURN_TYPE_UINT,
   TGSI_RETURN_TYPE_FLOAT,
   TGSI_RETURN_TYPE_UNKNOWN,
   TGSI_RETURN_TYPE_COUNT
};

struct tgsi_declaration_sampler_view {
   unsigned Resource    : 8; /**< one of TGSI_TEXTURE_ */
   unsigned ReturnTypeX : 6; /**< one of enum tgsi_return_type */
   unsigned ReturnTypeY : 6; /**< one of enum tgsi_return_type */
   unsigned ReturnTypeZ : 6; /**< one of enum tgsi_return_type */
   unsigned ReturnTypeW : 6; /**< one of enum tgsi_return_type */
};

struct tgsi_declaration_array {
   unsigned ArrayID : 10;
   unsigned Padding : 22;
};

enum tgsi_imm_type {
   TGSI_IMM_FLOAT32,
   TGSI_IMM_UINT32,
   TGSI_IMM_INT32,
   TGSI_IMM_FLOAT64,
   TGSI_IMM_UINT64,
   TGSI_IMM_INT64,
};

struct tgsi_immediate
{
   unsigned Type       : 4;  /**< TGSI_TOKEN_TYPE_IMMEDIATE */
   unsigned NrTokens   : 14; /**< UINT */
   unsigned DataType   : 4;  /**< one of TGSI_IMM_x */
   unsigned Padding    : 10;
};

union tgsi_immediate_data
{
   float Float;
   unsigned Uint;
   int Int;
};

enum tgsi_property_name {
   TGSI_PROPERTY_GS_INPUT_PRIM,
   TGSI_PROPERTY_GS_OUTPUT_PRIM,
   TGSI_PROPERTY_GS_MAX_OUTPUT_VERTICES,
   TGSI_PROPERTY_FS_COORD_ORIGIN,
   TGSI_PROPERTY_FS_COORD_PIXEL_CENTER,
   TGSI_PROPERTY_FS_COLOR0_WRITES_ALL_CBUFS,
   TGSI_PROPERTY_FS_DEPTH_LAYOUT,
   TGSI_PROPERTY_VS_PROHIBIT_UCPS,
   TGSI_PROPERTY_GS_INVOCATIONS,
   TGSI_PROPERTY_VS_WINDOW_SPACE_POSITION,
   TGSI_PROPERTY_TCS_VERTICES_OUT,
   TGSI_PROPERTY_TES_PRIM_MODE,
   TGSI_PROPERTY_TES_SPACING,
   TGSI_PROPERTY_TES_VERTEX_ORDER_CW,
   TGSI_PROPERTY_TES_POINT_MODE,
   TGSI_PROPERTY_NUM_CLIPDIST_ENABLED,
   TGSI_PROPERTY_NUM_CULLDIST_ENABLED,
   TGSI_PROPERTY_FS_EARLY_DEPTH_STENCIL,
   TGSI_PROPERTY_FS_POST_DEPTH_COVERAGE,
   TGSI_PROPERTY_NEXT_SHADER,
   TGSI_PROPERTY_CS_FIXED_BLOCK_WIDTH,
   TGSI_PROPERTY_CS_FIXED_BLOCK_HEIGHT,
   TGSI_PROPERTY_CS_FIXED_BLOCK_DEPTH,
   TGSI_PROPERTY_LEGACY_MATH_RULES,
   TGSI_PROPERTY_VS_BLIT_SGPRS_AMD,
   TGSI_PROPERTY_CS_USER_DATA_COMPONENTS_AMD,
   TGSI_PROPERTY_LAYER_VIEWPORT_RELATIVE,
   TGSI_PROPERTY_FS_BLEND_EQUATION_ADVANCED,
   TGSI_PROPERTY_SEPARABLE_PROGRAM,
   TGSI_PROPERTY_COUNT,
};

struct tgsi_property {
   unsigned Type         : 4;  /**< TGSI_TOKEN_TYPE_PROPERTY */
   unsigned NrTokens     : 8;  /**< UINT */
   unsigned PropertyName : 8;  /**< one of TGSI_PROPERTY */
   unsigned Padding      : 12;
};

enum tgsi_fs_coord_origin {
   TGSI_FS_COORD_ORIGIN_UPPER_LEFT,
   TGSI_FS_COORD_ORIGIN_LOWER_LEFT,
};

enum tgsi_fs_coord_pixcenter {
   TGSI_FS_COORD_PIXEL_CENTER_HALF_INTEGER,
   TGSI_FS_COORD_PIXEL_CENTER_INTEGER,
};

enum tgsi_fs_depth_layout {
   TGSI_FS_DEPTH_LAYOUT_NONE,
   TGSI_FS_DEPTH_LAYOUT_ANY,
   TGSI_FS_DEPTH_LAYOUT_GREATER,
   TGSI_FS_DEPTH_LAYOUT_LESS,
   TGSI_FS_DEPTH_LAYOUT_UNCHANGED,
};

struct tgsi_property_data {
   unsigned Data;
};

/* TGSI opcodes.
 *
 * For more information on semantics of opcodes and
 * which APIs are known to use which opcodes, see
 * gallium/docs/source/tgsi.rst
 */
enum tgsi_opcode {
   TGSI_OPCODE_ARL                = 0,
   TGSI_OPCODE_MOV                = 1,
   TGSI_OPCODE_LIT                = 2,
   TGSI_OPCODE_RCP                = 3,
   TGSI_OPCODE_RSQ                = 4,
   TGSI_OPCODE_EXP                = 5,
   TGSI_OPCODE_LOG                = 6,
   TGSI_OPCODE_MUL                = 7,
   TGSI_OPCODE_ADD                = 8,
   TGSI_OPCODE_DP3                = 9,
   TGSI_OPCODE_DP4                = 10,
   TGSI_OPCODE_DST                = 11,
   TGSI_OPCODE_MIN                = 12,
   TGSI_OPCODE_MAX                = 13,
   TGSI_OPCODE_SLT                = 14,
   TGSI_OPCODE_SGE                = 15,
   TGSI_OPCODE_MAD                = 16,
   TGSI_OPCODE_TEX_LZ             = 17,
   TGSI_OPCODE_LRP                = 18,
   TGSI_OPCODE_FMA                = 19,
   TGSI_OPCODE_SQRT               = 20,
   TGSI_OPCODE_LDEXP              = 21,
   TGSI_OPCODE_F2U64              = 22,
   TGSI_OPCODE_F2I64              = 23,
   TGSI_OPCODE_FRC                = 24,
   TGSI_OPCODE_TXF_LZ             = 25,
   TGSI_OPCODE_FLR                = 26,
   TGSI_OPCODE_ROUND              = 27,
   TGSI_OPCODE_EX2                = 28,
   TGSI_OPCODE_LG2                = 29,
   TGSI_OPCODE_POW                = 30,
   TGSI_OPCODE_DEMOTE             = 31,
   TGSI_OPCODE_U2I64              = 32,
   TGSI_OPCODE_CLOCK              = 33,
   TGSI_OPCODE_I2I64              = 34,
   TGSI_OPCODE_READ_HELPER        = 35,
   TGSI_OPCODE_COS                = 36,
   TGSI_OPCODE_DDX                = 37,
   TGSI_OPCODE_DDY                = 38,
   TGSI_OPCODE_KILL               = 39 /* unconditional */,
   TGSI_OPCODE_PK2H               = 40,
   TGSI_OPCODE_PK2US              = 41,
   TGSI_OPCODE_PK4B               = 42,
   TGSI_OPCODE_PK4UB              = 43,
   TGSI_OPCODE_D2U64              = 44,
   TGSI_OPCODE_SEQ                = 45,
   TGSI_OPCODE_D2I64              = 46,
   TGSI_OPCODE_SGT                = 47,
   TGSI_OPCODE_SIN                = 48,
   TGSI_OPCODE_SLE                = 49,
   TGSI_OPCODE_SNE                = 50,
   TGSI_OPCODE_U642D              = 51,
   TGSI_OPCODE_TEX                = 52,
   TGSI_OPCODE_TXD                = 53,
   TGSI_OPCODE_TXP                = 54,
   TGSI_OPCODE_UP2H               = 55,
   TGSI_OPCODE_UP2US              = 56,
   TGSI_OPCODE_UP4B               = 57,
   TGSI_OPCODE_UP4UB              = 58,
   TGSI_OPCODE_U642F              = 59,
   TGSI_OPCODE_I642F              = 60,
   TGSI_OPCODE_ARR                = 61,
   TGSI_OPCODE_I642D              = 62,
   TGSI_OPCODE_CAL                = 63,
   TGSI_OPCODE_RET                = 64,
   TGSI_OPCODE_SSG                = 65 /* SGN */,
   TGSI_OPCODE_CMP                = 66,
   /* gap */
   TGSI_OPCODE_TXB                = 68,
   TGSI_OPCODE_FBFETCH            = 69,
   TGSI_OPCODE_DIV                = 70,
   TGSI_OPCODE_DP2                = 71,
   TGSI_OPCODE_TXL                = 72,
   TGSI_OPCODE_BRK                = 73,
   TGSI_OPCODE_IF                 = 74,
   TGSI_OPCODE_UIF                = 75,
   TGSI_OPCODE_READ_INVOC         = 76,
   TGSI_OPCODE_ELSE               = 77,
   TGSI_OPCODE_ENDIF              = 78,
   TGSI_OPCODE_DDX_FINE           = 79,
   TGSI_OPCODE_DDY_FINE           = 80,
   /* gap */
   TGSI_OPCODE_CEIL               = 83,
   TGSI_OPCODE_I2F                = 84,
   TGSI_OPCODE_NOT                = 85,
   TGSI_OPCODE_TRUNC              = 86,
   TGSI_OPCODE_SHL                = 87,
   TGSI_OPCODE_BALLOT             = 88,
   TGSI_OPCODE_AND                = 89,
   TGSI_OPCODE_OR                 = 90,
   TGSI_OPCODE_MOD                = 91,
   TGSI_OPCODE_XOR                = 92,
   /* gap */
   TGSI_OPCODE_TXF                = 94,
   TGSI_OPCODE_TXQ                = 95,
   TGSI_OPCODE_CONT               = 96,
   TGSI_OPCODE_EMIT               = 97,
   TGSI_OPCODE_ENDPRIM            = 98,
   TGSI_OPCODE_BGNLOOP            = 99,
   TGSI_OPCODE_BGNSUB             = 100,
   TGSI_OPCODE_ENDLOOP            = 101,
   TGSI_OPCODE_ENDSUB             = 102,
   TGSI_OPCODE_ATOMFADD           = 103,
   TGSI_OPCODE_TXQS               = 104,
   TGSI_OPCODE_RESQ               = 105,
   TGSI_OPCODE_READ_FIRST         = 106,
   TGSI_OPCODE_NOP                = 107,

   TGSI_OPCODE_FSEQ               = 108,
   TGSI_OPCODE_FSGE               = 109,
   TGSI_OPCODE_FSLT               = 110,
   TGSI_OPCODE_FSNE               = 111,

   TGSI_OPCODE_MEMBAR             = 112,
                                /* gap */
   TGSI_OPCODE_KILL_IF            = 116  /* conditional kill */,
   TGSI_OPCODE_END                = 117  /* aka HALT */,
   TGSI_OPCODE_DFMA               = 118,
   TGSI_OPCODE_F2I                = 119,
   TGSI_OPCODE_IDIV               = 120,
   TGSI_OPCODE_IMAX               = 121,
   TGSI_OPCODE_IMIN               = 122,
   TGSI_OPCODE_INEG               = 123,
   TGSI_OPCODE_ISGE               = 124,
   TGSI_OPCODE_ISHR               = 125,
   TGSI_OPCODE_ISLT               = 126,
   TGSI_OPCODE_F2U                = 127,
   TGSI_OPCODE_U2F                = 128,
   TGSI_OPCODE_UADD               = 129,
   TGSI_OPCODE_UDIV               = 130,
   TGSI_OPCODE_UMAD               = 131,
   TGSI_OPCODE_UMAX               = 132,
   TGSI_OPCODE_UMIN               = 133,
   TGSI_OPCODE_UMOD               = 134,
   TGSI_OPCODE_UMUL               = 135,
   TGSI_OPCODE_USEQ               = 136,
   TGSI_OPCODE_USGE               = 137,
   TGSI_OPCODE_USHR               = 138,
   TGSI_OPCODE_USLT               = 139,
   TGSI_OPCODE_USNE               = 140,
   TGSI_OPCODE_SWITCH             = 141,
   TGSI_OPCODE_CASE               = 142,
   TGSI_OPCODE_DEFAULT            = 143,
   TGSI_OPCODE_ENDSWITCH          = 144,

   /* resource related opcodes */
   TGSI_OPCODE_SAMPLE             = 145,
   TGSI_OPCODE_SAMPLE_I           = 146,
   TGSI_OPCODE_SAMPLE_I_MS        = 147,
   TGSI_OPCODE_SAMPLE_B           = 148,
   TGSI_OPCODE_SAMPLE_C           = 149,
   TGSI_OPCODE_SAMPLE_C_LZ        = 150,
   TGSI_OPCODE_SAMPLE_D           = 151,
   TGSI_OPCODE_SAMPLE_L           = 152,
   TGSI_OPCODE_GATHER4            = 153,
   TGSI_OPCODE_SVIEWINFO          = 154,
   TGSI_OPCODE_SAMPLE_POS         = 155,
   TGSI_OPCODE_SAMPLE_INFO        = 156,

   TGSI_OPCODE_UARL               = 157,
   TGSI_OPCODE_UCMP               = 158,
   TGSI_OPCODE_IABS               = 159,
   TGSI_OPCODE_ISSG               = 160,

   TGSI_OPCODE_LOAD               = 161,
   TGSI_OPCODE_STORE              = 162,
   TGSI_OPCODE_IMG2HND            = 163,
   TGSI_OPCODE_SAMP2HND           = 164,
   /* gap */
   TGSI_OPCODE_BARRIER            = 166,

   TGSI_OPCODE_ATOMUADD           = 167,
   TGSI_OPCODE_ATOMXCHG           = 168,
   TGSI_OPCODE_ATOMCAS            = 169,
   TGSI_OPCODE_ATOMAND            = 170,
   TGSI_OPCODE_ATOMOR             = 171,
   TGSI_OPCODE_ATOMXOR            = 172,
   TGSI_OPCODE_ATOMUMIN           = 173,
   TGSI_OPCODE_ATOMUMAX           = 174,
   TGSI_OPCODE_ATOMIMIN           = 175,
   TGSI_OPCODE_ATOMIMAX           = 176,

   /* to be used for shadow cube map compares */
   TGSI_OPCODE_TEX2               = 177,
   TGSI_OPCODE_TXB2               = 178,
   TGSI_OPCODE_TXL2               = 179,

   TGSI_OPCODE_IMUL_HI            = 180,
   TGSI_OPCODE_UMUL_HI            = 181,

   TGSI_OPCODE_TG4                = 182,

   TGSI_OPCODE_LODQ               = 183,

   TGSI_OPCODE_IBFE               = 184,
   TGSI_OPCODE_UBFE               = 185,
   TGSI_OPCODE_BFI                = 186,
   TGSI_OPCODE_BREV               = 187,
   TGSI_OPCODE_POPC               = 188,
   TGSI_OPCODE_LSB                = 189,
   TGSI_OPCODE_IMSB               = 190,
   TGSI_OPCODE_UMSB               = 191,

   TGSI_OPCODE_INTERP_CENTROID    = 192,
   TGSI_OPCODE_INTERP_SAMPLE      = 193,
   TGSI_OPCODE_INTERP_OFFSET      = 194,

   /* sm5 marked opcodes are supported in D3D11 optionally - also DMOV, DMOVC */
   TGSI_OPCODE_F2D                = 195 /* SM5 */,
   TGSI_OPCODE_D2F                = 196,
   TGSI_OPCODE_DABS               = 197,
   TGSI_OPCODE_DNEG               = 198 /* SM5 */,
   TGSI_OPCODE_DADD               = 199 /* SM5 */,
   TGSI_OPCODE_DMUL               = 200 /* SM5 */,
   TGSI_OPCODE_DMAX               = 201 /* SM5 */,
   TGSI_OPCODE_DMIN               = 202 /* SM5 */,
   TGSI_OPCODE_DSLT               = 203 /* SM5 */,
   TGSI_OPCODE_DSGE               = 204 /* SM5 */,
   TGSI_OPCODE_DSEQ               = 205 /* SM5 */,
   TGSI_OPCODE_DSNE               = 206 /* SM5 */,
   TGSI_OPCODE_DRCP               = 207 /* eg, cayman */,
   TGSI_OPCODE_DSQRT              = 208 /* eg, cayman also has DRSQ */,
   TGSI_OPCODE_DMAD               = 209,
   TGSI_OPCODE_DFRAC              = 210 /* eg, cayman */,
   TGSI_OPCODE_DLDEXP             = 211 /* eg, cayman */,
   TGSI_OPCODE_DFRACEXP           = 212 /* eg, cayman */,
   TGSI_OPCODE_D2I                = 213,
   TGSI_OPCODE_I2D                = 214,
   TGSI_OPCODE_D2U                = 215,
   TGSI_OPCODE_U2D                = 216,
   TGSI_OPCODE_DRSQ               = 217 /* eg, cayman also has DRSQ */,
   TGSI_OPCODE_DTRUNC             = 218 /* nvc0 */,
   TGSI_OPCODE_DCEIL              = 219 /* nvc0 */,
   TGSI_OPCODE_DFLR               = 220 /* nvc0 */,
   TGSI_OPCODE_DROUND             = 221 /* nvc0 */,
   TGSI_OPCODE_DSSG               = 222,

   TGSI_OPCODE_VOTE_ANY           = 223,
   TGSI_OPCODE_VOTE_ALL           = 224,
   TGSI_OPCODE_VOTE_EQ            = 225,

   TGSI_OPCODE_U64SEQ             = 226,
   TGSI_OPCODE_U64SNE             = 227,
   TGSI_OPCODE_I64SLT             = 228,
   TGSI_OPCODE_U64SLT             = 229,
   TGSI_OPCODE_I64SGE             = 230,
   TGSI_OPCODE_U64SGE             = 231,

   TGSI_OPCODE_I64MIN             = 232,
   TGSI_OPCODE_U64MIN             = 233,
   TGSI_OPCODE_I64MAX             = 234,
   TGSI_OPCODE_U64MAX             = 235,

   TGSI_OPCODE_I64ABS             = 236,
   TGSI_OPCODE_I64SSG             = 237,
   TGSI_OPCODE_I64NEG             = 238,

   TGSI_OPCODE_U64ADD             = 239,
   TGSI_OPCODE_U64MUL             = 240,
   TGSI_OPCODE_U64SHL             = 241,
   TGSI_OPCODE_I64SHR             = 242,
   TGSI_OPCODE_U64SHR             = 243,

   TGSI_OPCODE_I64DIV             = 244,
   TGSI_OPCODE_U64DIV             = 245,
   TGSI_OPCODE_I64MOD             = 246,
   TGSI_OPCODE_U64MOD             = 247,

   TGSI_OPCODE_DDIV               = 248,

   TGSI_OPCODE_LOD                = 249,

   TGSI_OPCODE_ATOMINC_WRAP       = 250,
   TGSI_OPCODE_ATOMDEC_WRAP       = 251,

   TGSI_OPCODE_LAST               = 252,
};


/**
 * Opcode is the operation code to execute. A given operation defines the
 * semantics how the source registers (if any) are interpreted and what is
 * written to the destination registers (if any) as a result of execution.
 *
 * NumDstRegs and NumSrcRegs is the number of destination and source registers,
 * respectively. For a given operation code, those numbers are fixed and are
 * present here only for convenience.
 *
 * Saturate controls how are final results in destination registers modified.
 */

struct tgsi_instruction
{
   unsigned Type       : 4;  /* TGSI_TOKEN_TYPE_INSTRUCTION */
   unsigned NrTokens   : 8;  /* UINT */
   unsigned Opcode     : 8;  /* TGSI_OPCODE_ */
   unsigned Saturate   : 1;  /* BOOL */
   unsigned NumDstRegs : 2;  /* UINT */
   unsigned NumSrcRegs : 4;  /* UINT */
   unsigned Label      : 1;
   unsigned Texture    : 1;
   unsigned Memory     : 1;
   unsigned Precise    : 1;
   unsigned Padding    : 1;
};

/*
 * If tgsi_instruction::Label is TRUE, tgsi_instruction_label follows.
 *
 * If tgsi_instruction::Texture is TRUE, tgsi_instruction_texture follows.
 *   if texture instruction has a number of offsets,
 *   then tgsi_instruction::Texture::NumOffset of tgsi_texture_offset follow.
 *
 * Then, tgsi_instruction::NumDstRegs of tgsi_dst_register follow.
 *
 * Then, tgsi_instruction::NumSrcRegs of tgsi_src_register follow.
 *
 * tgsi_instruction::NrTokens contains the total number of words that make the
 * instruction, including the instruction word.
 */

enum tgsi_swizzle {
   TGSI_SWIZZLE_X,
   TGSI_SWIZZLE_Y,
   TGSI_SWIZZLE_Z,
   TGSI_SWIZZLE_W,
};

struct tgsi_instruction_label
{
   unsigned Label    : 24;   /* UINT */
   unsigned Padding  : 8;
};

enum tgsi_texture_type {
   TGSI_TEXTURE_BUFFER,
   TGSI_TEXTURE_1D,
   TGSI_TEXTURE_2D,
   TGSI_TEXTURE_3D,
   TGSI_TEXTURE_CUBE,
   TGSI_TEXTURE_RECT,
   TGSI_TEXTURE_SHADOW1D,
   TGSI_TEXTURE_SHADOW2D,
   TGSI_TEXTURE_SHADOWRECT,
   TGSI_TEXTURE_1D_ARRAY,
   TGSI_TEXTURE_2D_ARRAY,
   TGSI_TEXTURE_SHADOW1D_ARRAY,
   TGSI_TEXTURE_SHADOW2D_ARRAY,
   TGSI_TEXTURE_SHADOWCUBE,
   TGSI_TEXTURE_2D_MSAA,
   TGSI_TEXTURE_2D_ARRAY_MSAA,
   TGSI_TEXTURE_CUBE_ARRAY,
   TGSI_TEXTURE_SHADOWCUBE_ARRAY,
   TGSI_TEXTURE_UNKNOWN,
   TGSI_TEXTURE_COUNT,
};

struct tgsi_instruction_texture
{
   unsigned Texture  : 8;    /* TGSI_TEXTURE_ */
   unsigned NumOffsets : 4;
   unsigned ReturnType : 3; /* TGSI_RETURN_TYPE_x */
   unsigned Padding : 17;
};

/* for texture offsets in GLSL and DirectX.
 * Generally these always come from TGSI_FILE_IMMEDIATE,
 * however DX11 appears to have the capability to do
 * non-constant texture offsets.
 */
struct tgsi_texture_offset
{
   int      Index    : 16;
   unsigned File     : 4;  /**< one of TGSI_FILE_x */
   unsigned SwizzleX : 2;  /* TGSI_SWIZZLE_x */
   unsigned SwizzleY : 2;  /* TGSI_SWIZZLE_x */
   unsigned SwizzleZ : 2;  /* TGSI_SWIZZLE_x */
   unsigned Padding  : 6;
};

/**
 * File specifies the register array to access.
 *
 * Index specifies the element number of a register in the register file.
 *
 * If Indirect is TRUE, Index should be offset by the X component of the indirect
 * register that follows. The register can be now fetched into local storage
 * for further processing.
 *
 * If Negate is TRUE, all components of the fetched register are negated.
 *
 * The fetched register components are swizzled according to SwizzleX, SwizzleY,
 * SwizzleZ and SwizzleW.
 *
 */

struct tgsi_src_register
{
   unsigned File        : 4;  /* TGSI_FILE_ */
   unsigned Indirect    : 1;  /* BOOL */
   unsigned Dimension   : 1;  /* BOOL */
   int      Index       : 16; /* SINT */
   unsigned SwizzleX    : 2;  /* TGSI_SWIZZLE_ */
   unsigned SwizzleY    : 2;  /* TGSI_SWIZZLE_ */
   unsigned SwizzleZ    : 2;  /* TGSI_SWIZZLE_ */
   unsigned SwizzleW    : 2;  /* TGSI_SWIZZLE_ */
   unsigned Absolute    : 1;    /* BOOL */
   unsigned Negate      : 1;    /* BOOL */
};

/**
 * If tgsi_src_register::Indirect is TRUE, tgsi_ind_register follows.
 *
 * File, Index and Swizzle are handled the same as in tgsi_src_register.
 *
 * If ArrayID is zero the whole register file might be indirectly addressed,
 * if not only the Declaration with this ArrayID is accessed by this operand.
 *
 */

struct tgsi_ind_register
{
   unsigned File    : 4;  /* TGSI_FILE_ */
   int      Index   : 16; /* SINT */
   unsigned Swizzle : 2;  /* TGSI_SWIZZLE_ */
   unsigned ArrayID : 10; /* UINT */
};

/**
 * If tgsi_src_register::Dimension is TRUE, tgsi_dimension follows.
 */

struct tgsi_dimension
{
   unsigned Indirect    : 1;  /* BOOL */
   unsigned Dimension   : 1;  /* BOOL */
   unsigned Padding     : 14;
   int      Index       : 16; /* SINT */
};

struct tgsi_dst_register
{
   unsigned File        : 4;  /* TGSI_FILE_ */
   unsigned WriteMask   : 4;  /* TGSI_WRITEMASK_ */
   unsigned Indirect    : 1;  /* BOOL */
   unsigned Dimension   : 1;  /* BOOL */
   int      Index       : 16; /* SINT */
   unsigned Padding     : 6;
};

#define TGSI_MEMORY_COHERENT (1 << 0)
#define TGSI_MEMORY_RESTRICT (1 << 1)
#define TGSI_MEMORY_VOLATILE (1 << 2)
/* The "stream" cache policy will minimize memory cache usage if other
 * memory operations need the cache.
 */
#define TGSI_MEMORY_STREAM_CACHE_POLICY (1 << 3)

/**
 * Specifies the type of memory access to do for the LOAD/STORE instruction.
 */
struct tgsi_instruction_memory
{
   unsigned Qualifier : 4;  /* TGSI_MEMORY_ */
   unsigned Texture   : 8;  /* only for images: TGSI_TEXTURE_ */
   unsigned Format    : 10; /* only for images: PIPE_FORMAT_ */
   unsigned Padding   : 10;
};

#define TGSI_MEMBAR_SHADER_BUFFER (1 << 0)
#define TGSI_MEMBAR_ATOMIC_BUFFER (1 << 1)
#define TGSI_MEMBAR_SHADER_IMAGE  (1 << 2)
#define TGSI_MEMBAR_SHARED        (1 << 3)
#define TGSI_MEMBAR_THREAD_GROUP  (1 << 4)

#ifdef __cplusplus
}
#endif

#endif /* P_SHADER_TOKENS_H */
