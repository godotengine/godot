/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 * Copyright (C) 2009  VMware, Inc.  All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * \file mtypes.h
 * Main Mesa data structures.
 *
 * Please try to mark derived values with a leading underscore ('_').
 */

#ifndef MTYPES_H
#define MTYPES_H


#include <stdint.h>             /* uint32_t */
#include <stdbool.h>
#include "c11/threads.h"

#include "util/glheader.h"
#include "main/glthread.h"
#include "main/consts_exts.h"
#include "main/shader_types.h"
#include "main/glconfig.h"
#include "main/menums.h"
#include "main/config.h"
#include "glapi/glapi.h"
#include "math/m_matrix.h"	/* GLmatrix */
#include "compiler/shader_enums.h"
#include "compiler/shader_info.h"
#include "main/formats.h"       /* MESA_FORMAT_COUNT */
#include "compiler/glsl/list.h"
#include "compiler/glsl/ir_uniform.h"
#include "util/u_idalloc.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "util/mesa-sha1.h"
#include "vbo/vbo.h"

#include "pipe/p_state.h"

#include "frontend/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GET_COLORMASK_BIT(mask, buf, chan) (((mask) >> (4 * (buf) + (chan))) & 0x1)
#define GET_COLORMASK(mask, buf) (((mask) >> (4 * (buf))) & 0xf)


/**
 * \name Some forward type declarations
 */
/*@{*/
struct _mesa_HashTable;
struct gl_attrib_node;
struct gl_list_extensions;
struct gl_meta_state;
struct gl_program_cache;
struct gl_texture_object;
struct gl_debug_state;
struct gl_context;
struct st_context;
struct gl_uniform_storage;
struct prog_instruction;
struct gl_program_parameter_list;
struct gl_shader_spirv_data;
struct set;
struct shader_includes;
/*@}*/


/** Extra draw modes beyond GL_POINTS, GL_TRIANGLE_FAN, etc */
#define PRIM_MAX                 GL_PATCHES
#define PRIM_OUTSIDE_BEGIN_END   (PRIM_MAX + 1)
#define PRIM_UNKNOWN             (PRIM_MAX + 2)

/**
 * Bit flags for all renderbuffers
 */
#define BUFFER_BIT_FRONT_LEFT   (1 << BUFFER_FRONT_LEFT)
#define BUFFER_BIT_BACK_LEFT    (1 << BUFFER_BACK_LEFT)
#define BUFFER_BIT_FRONT_RIGHT  (1 << BUFFER_FRONT_RIGHT)
#define BUFFER_BIT_BACK_RIGHT   (1 << BUFFER_BACK_RIGHT)
#define BUFFER_BIT_DEPTH        (1 << BUFFER_DEPTH)
#define BUFFER_BIT_STENCIL      (1 << BUFFER_STENCIL)
#define BUFFER_BIT_ACCUM        (1 << BUFFER_ACCUM)
#define BUFFER_BIT_COLOR0       (1 << BUFFER_COLOR0)
#define BUFFER_BIT_COLOR1       (1 << BUFFER_COLOR1)
#define BUFFER_BIT_COLOR2       (1 << BUFFER_COLOR2)
#define BUFFER_BIT_COLOR3       (1 << BUFFER_COLOR3)
#define BUFFER_BIT_COLOR4       (1 << BUFFER_COLOR4)
#define BUFFER_BIT_COLOR5       (1 << BUFFER_COLOR5)
#define BUFFER_BIT_COLOR6       (1 << BUFFER_COLOR6)
#define BUFFER_BIT_COLOR7       (1 << BUFFER_COLOR7)

/**
 * Mask of all the color buffer bits (but not accum).
 */
#define BUFFER_BITS_COLOR  (BUFFER_BIT_FRONT_LEFT | \
                            BUFFER_BIT_BACK_LEFT | \
                            BUFFER_BIT_FRONT_RIGHT | \
                            BUFFER_BIT_BACK_RIGHT | \
                            BUFFER_BIT_COLOR0 | \
                            BUFFER_BIT_COLOR1 | \
                            BUFFER_BIT_COLOR2 | \
                            BUFFER_BIT_COLOR3 | \
                            BUFFER_BIT_COLOR4 | \
                            BUFFER_BIT_COLOR5 | \
                            BUFFER_BIT_COLOR6 | \
                            BUFFER_BIT_COLOR7)

/* Mask of bits for depth+stencil buffers */
#define BUFFER_BITS_DEPTH_STENCIL (BUFFER_BIT_DEPTH | BUFFER_BIT_STENCIL)


#define FRONT_MATERIAL_BITS   (MAT_BIT_FRONT_EMISSION | \
                               MAT_BIT_FRONT_AMBIENT | \
                               MAT_BIT_FRONT_DIFFUSE | \
                               MAT_BIT_FRONT_SPECULAR | \
                               MAT_BIT_FRONT_SHININESS | \
                               MAT_BIT_FRONT_INDEXES)

#define BACK_MATERIAL_BITS    (MAT_BIT_BACK_EMISSION | \
                               MAT_BIT_BACK_AMBIENT | \
                               MAT_BIT_BACK_DIFFUSE | \
                               MAT_BIT_BACK_SPECULAR | \
                               MAT_BIT_BACK_SHININESS | \
                               MAT_BIT_BACK_INDEXES)

#define ALL_MATERIAL_BITS     (FRONT_MATERIAL_BITS | BACK_MATERIAL_BITS)
/*@}*/


/**
 * Material state.
 */
struct gl_material
{
   GLfloat Attrib[MAT_ATTRIB_MAX][4];
};


/**
 * Light state flags.
 */
/*@{*/
#define LIGHT_SPOT         0x1
#define LIGHT_LOCAL_VIEWER 0x2
#define LIGHT_POSITIONAL   0x4
#define LIGHT_NEED_VERTICES (LIGHT_POSITIONAL|LIGHT_LOCAL_VIEWER)
/*@}*/


/**
 * Light source state.
 */
struct gl_light
{
   GLboolean Enabled;		/**< On/off flag */

   /**
    * \name Derived fields
    */
   /*@{*/
   GLbitfield _Flags;		/**< Mask of LIGHT_x bits defined above */

   GLfloat _Position[4];	/**< position in eye/obj coordinates */
   GLfloat _VP_inf_norm[3];	/**< Norm direction to infinite light */
   GLfloat _h_inf_norm[3];	/**< Norm( _VP_inf_norm + <0,0,1> ) */
   GLfloat _NormSpotDirection[4]; /**< normalized spotlight direction */
   GLfloat _VP_inf_spot_attenuation;

   GLfloat _MatAmbient[2][3];	/**< material ambient * light ambient */
   GLfloat _MatDiffuse[2][3];	/**< material diffuse * light diffuse */
   GLfloat _MatSpecular[2][3];	/**< material spec * light specular */
   /*@}*/
};


/**
 * Light model state.
 */
struct gl_lightmodel
{
   GLfloat Ambient[4];		/**< ambient color */
   GLboolean LocalViewer;	/**< Local (or infinite) view point? */
   GLboolean TwoSide;		/**< Two (or one) sided lighting? */
   GLenum16 ColorControl;	/**< either GL_SINGLE_COLOR
                                     or GL_SEPARATE_SPECULAR_COLOR */
};


/**
 * Accumulation buffer attribute group (GL_ACCUM_BUFFER_BIT)
 */
struct gl_accum_attrib
{
   GLfloat ClearColor[4];	/**< Accumulation buffer clear color */
};


/**
 * Used for storing clear color, texture border color, etc.
 * The float values are typically unclamped.
 */
union gl_color_union
{
   GLfloat f[4];
   GLint i[4];
   GLuint ui[4];
};


/**
 * Color buffer attribute group (GL_COLOR_BUFFER_BIT).
 */
struct gl_colorbuffer_attrib
{
   GLuint ClearIndex;                      /**< Index for glClear */
   union gl_color_union ClearColor;        /**< Color for glClear, unclamped */
   GLuint IndexMask;                       /**< Color index write mask */

   /** 4 colormask bits per draw buffer, max 8 draw buffers. 4*8 = 32 bits */
   GLbitfield ColorMask;

   GLenum16 DrawBuffer[MAX_DRAW_BUFFERS];  /**< Which buffer to draw into */

   /**
    * \name alpha testing
    */
   /*@{*/
   GLboolean AlphaEnabled;		/**< Alpha test enabled flag */
   GLenum16 AlphaFunc;			/**< Alpha test function */
   GLfloat AlphaRefUnclamped;
   GLclampf AlphaRef;			/**< Alpha reference value */
   /*@}*/

   /**
    * \name Blending
    */
   /*@{*/
   GLbitfield BlendEnabled;		/**< Per-buffer blend enable flags */

   /* NOTE: this does _not_ depend on fragment clamping or any other clamping
    * control, only on the fixed-pointness of the render target.
    * The query does however depend on fragment color clamping.
    */
   GLfloat BlendColorUnclamped[4];      /**< Blending color */
   GLfloat BlendColor[4];		/**< Blending color */

   struct
   {
      GLenum16 SrcRGB;             /**< RGB blend source term */
      GLenum16 DstRGB;             /**< RGB blend dest term */
      GLenum16 SrcA;               /**< Alpha blend source term */
      GLenum16 DstA;               /**< Alpha blend dest term */
      GLenum16 EquationRGB;        /**< GL_ADD, GL_SUBTRACT, etc. */
      GLenum16 EquationA;          /**< GL_ADD, GL_SUBTRACT, etc. */
   } Blend[MAX_DRAW_BUFFERS];
   /** Bitfield of color buffers with enabled dual source blending. */
   GLbitfield _BlendUsesDualSrc;
   /** Are the blend func terms currently different for each buffer/target? */
   GLboolean _BlendFuncPerBuffer;
   /** Are the blend equations currently different for each buffer/target? */
   GLboolean _BlendEquationPerBuffer;

   /**
    * Which advanced blending mode is in use (or BLEND_NONE).
    *
    * KHR_blend_equation_advanced only allows advanced blending with a single
    * draw buffer, and NVX_blend_equation_advanced_multi_draw_buffer still
    * requires all draw buffers to match, so we only need a single value.
    */
   enum gl_advanced_blend_mode _AdvancedBlendMode;

   /** Coherency requested via glEnable(GL_BLEND_ADVANCED_COHERENT_KHR)? */
   bool BlendCoherent;
   /*@}*/

   /**
    * \name Logic op
    */
   /*@{*/
   GLboolean IndexLogicOpEnabled;	/**< Color index logic op enabled flag */
   GLboolean ColorLogicOpEnabled;	/**< RGBA logic op enabled flag */
   GLenum16 LogicOp;			/**< Logic operator */
   enum gl_logicop_mode _LogicOp;
   /*@}*/

   GLboolean DitherFlag;           /**< Dither enable flag */

   GLboolean _ClampFragmentColor;  /** < with GL_FIXED_ONLY_ARB resolved */
   GLenum16 ClampFragmentColor; /**< GL_TRUE, GL_FALSE or GL_FIXED_ONLY_ARB */
   GLenum16 ClampReadColor;     /**< GL_TRUE, GL_FALSE or GL_FIXED_ONLY_ARB */

   GLboolean sRGBEnabled;  /**< Framebuffer sRGB blending/updating requested */
};

/**
 * Vertex format to describe a vertex element.
 */
struct gl_vertex_format
{
   union gl_vertex_format_user User;
   enum pipe_format _PipeFormat:16; /**< pipe_format for Gallium */
   GLushort _ElementSize; /**< Size of each element in bytes */
};


/**
 * Current attribute group (GL_CURRENT_BIT).
 */
struct gl_current_attrib
{
   /**
    * \name Current vertex attributes (color, texcoords, etc).
    * \note Values are valid only after FLUSH_VERTICES has been called.
    * \note Index and Edgeflag current values are stored as floats in the
    * SIX and SEVEN attribute slots.
    * \note We need double storage for 64-bit vertex attributes
    */
   GLfloat Attrib[VERT_ATTRIB_MAX][4*2];

   /**
    * \name Current raster position attributes (always up to date after a
    * glRasterPos call).
    */
   GLfloat RasterPos[4];
   GLfloat RasterDistance;
   GLfloat RasterColor[4];
   GLfloat RasterSecondaryColor[4];
   GLfloat RasterTexCoords[MAX_TEXTURE_COORD_UNITS][4];
   GLboolean RasterPosValid;
};


/**
 * Depth buffer attribute group (GL_DEPTH_BUFFER_BIT).
 */
struct gl_depthbuffer_attrib
{
   GLenum16 Func;		/**< Function for depth buffer compare */
   GLclampd Clear;		/**< Value to clear depth buffer to */
   GLboolean Test;		/**< Depth buffering enabled flag */
   GLboolean Mask;		/**< Depth buffer writable? */
   GLboolean BoundsTest;        /**< GL_EXT_depth_bounds_test */
   GLclampd BoundsMin, BoundsMax;/**< GL_EXT_depth_bounds_test */
};


/**
 * Evaluator attribute group (GL_EVAL_BIT).
 */
struct gl_eval_attrib
{
   /**
    * \name Enable bits
    */
   /*@{*/
   GLboolean Map1Color4;
   GLboolean Map1Index;
   GLboolean Map1Normal;
   GLboolean Map1TextureCoord1;
   GLboolean Map1TextureCoord2;
   GLboolean Map1TextureCoord3;
   GLboolean Map1TextureCoord4;
   GLboolean Map1Vertex3;
   GLboolean Map1Vertex4;
   GLboolean Map2Color4;
   GLboolean Map2Index;
   GLboolean Map2Normal;
   GLboolean Map2TextureCoord1;
   GLboolean Map2TextureCoord2;
   GLboolean Map2TextureCoord3;
   GLboolean Map2TextureCoord4;
   GLboolean Map2Vertex3;
   GLboolean Map2Vertex4;
   GLboolean AutoNormal;
   /*@}*/

   /**
    * \name Map Grid endpoints and divisions and calculated du values
    */
   /*@{*/
   GLint MapGrid1un;
   GLfloat MapGrid1u1, MapGrid1u2, MapGrid1du;
   GLint MapGrid2un, MapGrid2vn;
   GLfloat MapGrid2u1, MapGrid2u2, MapGrid2du;
   GLfloat MapGrid2v1, MapGrid2v2, MapGrid2dv;
   /*@}*/
};


/**
 * Compressed fog mode.
 */
enum gl_fog_mode
{
   FOG_NONE,
   FOG_LINEAR,
   FOG_EXP,
   FOG_EXP2,
};


/**
 * Fog attribute group (GL_FOG_BIT).
 */
struct gl_fog_attrib
{
   GLboolean Enabled;		/**< Fog enabled flag */
   GLboolean ColorSumEnabled;
   uint8_t _PackedMode;		/**< Fog mode as 2 bits */
   uint8_t _PackedEnabledMode;	/**< Masked CompressedMode */
   GLfloat ColorUnclamped[4];            /**< Fog color */
   GLfloat Color[4];		/**< Fog color */
   GLfloat Density;		/**< Density >= 0.0 */
   GLfloat Start;		/**< Start distance in eye coords */
   GLfloat End;			/**< End distance in eye coords */
   GLfloat Index;		/**< Fog index */
   GLenum16 Mode;		/**< Fog mode */
   GLenum16 FogCoordinateSource;/**< GL_EXT_fog_coord */
   GLenum16 FogDistanceMode;     /**< GL_NV_fog_distance */
};


/**
 * Hint attribute group (GL_HINT_BIT).
 *
 * Values are always one of GL_FASTEST, GL_NICEST, or GL_DONT_CARE.
 */
struct gl_hint_attrib
{
   GLenum16 PerspectiveCorrection;
   GLenum16 PointSmooth;
   GLenum16 LineSmooth;
   GLenum16 PolygonSmooth;
   GLenum16 Fog;
   GLenum16 TextureCompression;   /**< GL_ARB_texture_compression */
   GLenum16 GenerateMipmap;       /**< GL_SGIS_generate_mipmap */
   GLenum16 FragmentShaderDerivative; /**< GL_ARB_fragment_shader */
   GLuint MaxShaderCompilerThreads; /**< GL_ARB_parallel_shader_compile */
};


struct gl_light_uniforms {
   /* These must be in the same order as the STATE_* enums,
    * which should also match the order of gl_LightSource members.
    */
   GLfloat Ambient[4];           /**< STATE_AMBIENT */
   GLfloat Diffuse[4];           /**< STATE_DIFFUSE */
   GLfloat Specular[4];          /**< STATE_SPECULAR */
   GLfloat EyePosition[4];       /**< STATE_POSITION in eye coordinates */
   GLfloat _HalfVector[4];       /**< STATE_HALF_VECTOR */
   GLfloat SpotDirection[3];     /**< STATE_SPOT_DIRECTION in eye coordinates */
   GLfloat _CosCutoff;           /**< = MAX(0, cos(SpotCutoff)) */
   GLfloat ConstantAttenuation;  /**< STATE_ATTENUATION */
   GLfloat LinearAttenuation;
   GLfloat QuadraticAttenuation;
   GLfloat SpotExponent;
   GLfloat SpotCutoff;           /**< STATE_SPOT_CUTOFF in degrees */
};


/**
 * Lighting attribute group (GL_LIGHT_BIT).
 */
struct gl_light_attrib
{
   /* gl_LightSource uniforms */
   union {
      struct gl_light_uniforms LightSource[MAX_LIGHTS];
      GLfloat LightSourceData[(sizeof(struct gl_light_uniforms) / 4) * MAX_LIGHTS];
   };

   struct gl_light Light[MAX_LIGHTS];	/**< Array of light sources */
   struct gl_lightmodel Model;		/**< Lighting model */

   /**
    * Front and back material values.
    * Note: must call FLUSH_VERTICES() before using.
    */
   struct gl_material Material;

   GLboolean Enabled;			/**< Lighting enabled flag */
   GLboolean ColorMaterialEnabled;

   GLenum16 ShadeModel;			/**< GL_FLAT or GL_SMOOTH */
   GLenum16 ProvokingVertex;              /**< GL_EXT_provoking_vertex */
   GLenum16 ColorMaterialFace;		/**< GL_FRONT, BACK or FRONT_AND_BACK */
   GLenum16 ColorMaterialMode;		/**< GL_AMBIENT, GL_DIFFUSE, etc */
   GLbitfield _ColorMaterialBitmask;	/**< bitmask formed from Face and Mode */


   GLboolean _ClampVertexColor;
   GLenum16 ClampVertexColor;             /**< GL_TRUE, GL_FALSE, GL_FIXED_ONLY */

   /**
    * Derived state for optimizations:
    */
   /*@{*/
   GLbitfield _EnabledLights;	/**< bitmask containing enabled lights */

   GLboolean _NeedEyeCoords;
   GLboolean _NeedVertices;		/**< Use fast shader? */

   GLfloat _BaseColor[2][3];
   /*@}*/
};


/**
 * Line attribute group (GL_LINE_BIT).
 */
struct gl_line_attrib
{
   GLboolean SmoothFlag;	/**< GL_LINE_SMOOTH enabled? */
   GLboolean StippleFlag;	/**< GL_LINE_STIPPLE enabled? */
   GLushort StipplePattern;	/**< Stipple pattern */
   GLint StippleFactor;		/**< Stipple repeat factor */
   GLfloat Width;		/**< Line width */
};


/**
 * Display list attribute group (GL_LIST_BIT).
 */
struct gl_list_attrib
{
   GLuint ListBase;
};


/**
 * Multisample attribute group (GL_MULTISAMPLE_BIT).
 */
struct gl_multisample_attrib
{
   GLboolean Enabled;
   GLboolean SampleAlphaToCoverage;
   GLboolean SampleAlphaToOne;
   GLboolean SampleCoverage;
   GLboolean SampleCoverageInvert;
   GLboolean SampleShading;

   /* ARB_texture_multisample / GL3.2 additions */
   GLboolean SampleMask;

   GLfloat SampleCoverageValue;  /**< In range [0, 1] */
   GLfloat MinSampleShadingValue;  /**< In range [0, 1] */

   /** The GL spec defines this as an array but >32x MSAA is madness */
   GLbitfield SampleMaskValue;

   /* NV_alpha_to_coverage_dither_control */
   GLenum SampleAlphaToCoverageDitherControl;
};


/**
 * A pixelmap (see glPixelMap)
 */
struct gl_pixelmap
{
   GLint Size;
   GLfloat Map[MAX_PIXEL_MAP_TABLE];
};


/**
 * Collection of all pixelmaps
 */
struct gl_pixelmaps
{
   struct gl_pixelmap RtoR;  /**< i.e. GL_PIXEL_MAP_R_TO_R */
   struct gl_pixelmap GtoG;
   struct gl_pixelmap BtoB;
   struct gl_pixelmap AtoA;
   struct gl_pixelmap ItoR;
   struct gl_pixelmap ItoG;
   struct gl_pixelmap ItoB;
   struct gl_pixelmap ItoA;
   struct gl_pixelmap ItoI;
   struct gl_pixelmap StoS;
};


/**
 * Pixel attribute group (GL_PIXEL_MODE_BIT).
 */
struct gl_pixel_attrib
{
   GLenum16 ReadBuffer;		/**< source buffer for glRead/CopyPixels() */

   /*--- Begin Pixel Transfer State ---*/
   /* Fields are in the order in which they're applied... */

   /** Scale & Bias (index shift, offset) */
   /*@{*/
   GLfloat RedBias, RedScale;
   GLfloat GreenBias, GreenScale;
   GLfloat BlueBias, BlueScale;
   GLfloat AlphaBias, AlphaScale;
   GLfloat DepthBias, DepthScale;
   GLint IndexShift, IndexOffset;
   /*@}*/

   /* Pixel Maps */
   /* Note: actual pixel maps are not part of this attrib group */
   GLboolean MapColorFlag;
   GLboolean MapStencilFlag;

   /*--- End Pixel Transfer State ---*/

   /** glPixelZoom */
   GLfloat ZoomX, ZoomY;
};


/**
 * Point attribute group (GL_POINT_BIT).
 */
struct gl_point_attrib
{
   GLfloat Size;		/**< User-specified point size */
   GLfloat Params[3];		/**< GL_EXT_point_parameters */
   GLfloat MinSize, MaxSize;	/**< GL_EXT_point_parameters */
   GLfloat Threshold;		/**< GL_EXT_point_parameters */
   GLboolean SmoothFlag;	/**< True if GL_POINT_SMOOTH is enabled */
   GLboolean _Attenuated;	/**< True if Params != [1, 0, 0] */
   GLboolean PointSprite;	/**< GL_NV/ARB_point_sprite */
   GLbitfield CoordReplace;     /**< GL_ARB_point_sprite*/
   GLenum16 SpriteOrigin;	/**< GL_ARB_point_sprite */
};


/**
 * Polygon attribute group (GL_POLYGON_BIT).
 */
struct gl_polygon_attrib
{
   GLenum16 FrontFace;		/**< Either GL_CW or GL_CCW */
   GLenum FrontMode;		/**< Either GL_POINT, GL_LINE or GL_FILL */
   GLenum BackMode;		/**< Either GL_POINT, GL_LINE or GL_FILL */
   GLboolean CullFlag;		/**< Culling on/off flag */
   GLboolean SmoothFlag;	/**< True if GL_POLYGON_SMOOTH is enabled */
   GLboolean StippleFlag;	/**< True if GL_POLYGON_STIPPLE is enabled */
   GLenum16 CullFaceMode;	/**< Culling mode GL_FRONT or GL_BACK */
   GLfloat OffsetFactor;	/**< Polygon offset factor, from user */
   GLfloat OffsetUnits;		/**< Polygon offset units, from user */
   GLfloat OffsetClamp;		/**< Polygon offset clamp, from user */
   GLboolean OffsetPoint;	/**< Offset in GL_POINT mode */
   GLboolean OffsetLine;	/**< Offset in GL_LINE mode */
   GLboolean OffsetFill;	/**< Offset in GL_FILL mode */
};


/**
 * Scissor attributes (GL_SCISSOR_BIT).
 */
struct gl_scissor_rect
{
   GLint X, Y;			/**< Lower left corner of box */
   GLsizei Width, Height;	/**< Size of box */
};


struct gl_scissor_attrib
{
   GLbitfield EnableFlags;	/**< Scissor test enabled? */
   struct gl_scissor_rect ScissorArray[MAX_VIEWPORTS];
   GLint NumWindowRects;        /**< Count of enabled window rectangles */
   GLenum16 WindowRectMode;     /**< Whether to include or exclude the rects */
   struct gl_scissor_rect WindowRects[MAX_WINDOW_RECTANGLES];
};


/**
 * Stencil attribute group (GL_STENCIL_BUFFER_BIT).
 *
 * Three sets of stencil data are tracked so that OpenGL 2.0,
 * GL_EXT_stencil_two_side, and GL_ATI_separate_stencil can all be supported
 * simultaneously.  In each of the stencil state arrays, element 0 corresponds
 * to GL_FRONT.  Element 1 corresponds to the OpenGL 2.0 /
 * GL_ATI_separate_stencil GL_BACK state.  Element 2 corresponds to the
 * GL_EXT_stencil_two_side GL_BACK state.
 *
 * The derived value \c _BackFace is either 1 or 2 depending on whether or
 * not GL_STENCIL_TEST_TWO_SIDE_EXT is enabled.
 *
 * The derived value \c _TestTwoSide is set when the front-face and back-face
 * stencil state are different.
 */
struct gl_stencil_attrib
{
   GLboolean Enabled;		/**< Enabled flag */
   GLboolean TestTwoSide;	/**< GL_EXT_stencil_two_side */
   GLubyte ActiveFace;		/**< GL_EXT_stencil_two_side (0 or 2) */
   GLubyte _BackFace;           /**< Current back stencil state (1 or 2) */
   GLenum16 Function[3];	/**< Stencil function */
   GLenum16 FailFunc[3];	/**< Fail function */
   GLenum16 ZPassFunc[3];	/**< Depth buffer pass function */
   GLenum16 ZFailFunc[3];	/**< Depth buffer fail function */
   GLint Ref[3];		/**< Reference value */
   GLuint ValueMask[3];		/**< Value mask */
   GLuint WriteMask[3];		/**< Write mask */
   GLuint Clear;		/**< Clear value */
};


/**
 * Bit flags for each type of texture object
 */
/*@{*/
#define TEXTURE_2D_MULTISAMPLE_BIT (1 << TEXTURE_2D_MULTISAMPLE_INDEX)
#define TEXTURE_2D_MULTISAMPLE_ARRAY_BIT (1 << TEXTURE_2D_MULTISAMPLE_ARRAY_INDEX)
#define TEXTURE_CUBE_ARRAY_BIT (1 << TEXTURE_CUBE_ARRAY_INDEX)
#define TEXTURE_BUFFER_BIT   (1 << TEXTURE_BUFFER_INDEX)
#define TEXTURE_2D_ARRAY_BIT (1 << TEXTURE_2D_ARRAY_INDEX)
#define TEXTURE_1D_ARRAY_BIT (1 << TEXTURE_1D_ARRAY_INDEX)
#define TEXTURE_EXTERNAL_BIT (1 << TEXTURE_EXTERNAL_INDEX)
#define TEXTURE_CUBE_BIT     (1 << TEXTURE_CUBE_INDEX)
#define TEXTURE_3D_BIT       (1 << TEXTURE_3D_INDEX)
#define TEXTURE_RECT_BIT     (1 << TEXTURE_RECT_INDEX)
#define TEXTURE_2D_BIT       (1 << TEXTURE_2D_INDEX)
#define TEXTURE_1D_BIT       (1 << TEXTURE_1D_INDEX)
/*@}*/


/**
 * Texture image state.  Drivers will typically create a subclass of this
 * with extra fields for memory buffers, etc.
 */
struct gl_texture_image
{
   GLint InternalFormat;	/**< Internal format as given by the user */
   GLenum16 _BaseFormat;	/**< Either GL_RGB, GL_RGBA, GL_ALPHA,
                                 *   GL_LUMINANCE, GL_LUMINANCE_ALPHA,
                                 *   GL_INTENSITY, GL_DEPTH_COMPONENT or
                                 *   GL_DEPTH_STENCIL_EXT only. Used for
                                 *   choosing TexEnv arithmetic.
                                 */
   mesa_format TexFormat;         /**< The actual texture memory format */

   GLuint Border;		/**< 0 or 1 */
   GLuint Width;		/**< = 2^WidthLog2 + 2*Border */
   GLuint Height;		/**< = 2^HeightLog2 + 2*Border */
   GLuint Depth;		/**< = 2^DepthLog2 + 2*Border */
   GLuint Width2;		/**< = Width - 2*Border */
   GLuint Height2;		/**< = Height - 2*Border */
   GLuint Depth2;		/**< = Depth - 2*Border */
   GLuint WidthLog2;		/**< = log2(Width2) */
   GLuint HeightLog2;		/**< = log2(Height2) */
   GLuint DepthLog2;		/**< = log2(Depth2) */
   GLuint MaxNumLevels;		/**< = maximum possible number of mipmap
                                       levels, computed from the dimensions */

   struct gl_texture_object *TexObject;  /**< Pointer back to parent object */
   GLuint Level;                /**< Which mipmap level am I? */
   /** Cube map face: index into gl_texture_object::Image[] array */
   GLuint Face;

   unsigned FormatSwizzle;
   unsigned FormatSwizzleGLSL130; //for depth formats

   /** GL_ARB_texture_multisample */
   GLuint NumSamples;            /**< Sample count, or 0 for non-multisample */
   GLboolean FixedSampleLocations; /**< Same sample locations for all pixels? */

   /* If stImage->pt != NULL, image data is stored here.
    * Else there is no image data.
    */
   struct pipe_resource *pt;

   /* List of transfers, allocated on demand.
    * transfer[layer] is a mapping for that layer.
    */
   struct st_texture_image_transfer *transfer;
   unsigned num_transfers;

   /* For compressed images unsupported by the driver. Keep track of
    * the original data. This is necessary for mapping/unmapping,
    * as well as image copies.
    */
   struct st_compressed_data* compressed_data;
};


/**
 * Indexes for cube map faces.
 */
typedef enum
{
   FACE_POS_X = 0,
   FACE_NEG_X = 1,
   FACE_POS_Y = 2,
   FACE_NEG_Y = 3,
   FACE_POS_Z = 4,
   FACE_NEG_Z = 5,
   MAX_FACES = 6
} gl_face_index;

/**
 * Sampler state saved and restore by glPush/PopAttrib.
 *
 * Don't put fields here that glPushAttrib shouldn't save.
 * E.g. no GLES fields because GLES doesn't have glPushAttrib.
 */
struct gl_sampler_attrib
{
   GLenum16 WrapS;		/**< S-axis texture image wrap mode */
   GLenum16 WrapT;		/**< T-axis texture image wrap mode */
   GLenum16 WrapR;		/**< R-axis texture image wrap mode */
   GLenum16 MinFilter;		/**< minification filter */
   GLenum16 MagFilter;		/**< magnification filter */
   GLenum16 sRGBDecode;         /**< GL_DECODE_EXT or GL_SKIP_DECODE_EXT */
   GLfloat MinLod;		/**< min lambda, OpenGL 1.2 */
   GLfloat MaxLod;		/**< max lambda, OpenGL 1.2 */
   GLfloat LodBias;		/**< OpenGL 1.4 */
   GLfloat MaxAnisotropy;	/**< GL_EXT_texture_filter_anisotropic */
   GLenum16 CompareMode;	/**< GL_ARB_shadow */
   GLenum16 CompareFunc;	/**< GL_ARB_shadow */
   GLboolean CubeMapSeamless;   /**< GL_AMD_seamless_cubemap_per_texture */
   GLboolean IsBorderColorNonZero; /**< Does the border color have any effect? */
   GLenum16 ReductionMode;      /**< GL_EXT_texture_filter_minmax */

   struct pipe_sampler_state state;  /**< Gallium representation */
};

/**
 * Texture state saved and restored by glPush/PopAttrib.
 *
 * Don't put fields here that glPushAttrib shouldn't save.
 * E.g. no GLES fields because GLES doesn't have glPushAttrib.
 */
struct gl_texture_object_attrib
{
   GLfloat Priority;           /**< in [0,1] */
   GLint BaseLevel;            /**< min mipmap level, OpenGL 1.2 */
   GLint MaxLevel;             /**< max mipmap level (max=1000), OpenGL 1.2 */
   GLenum Swizzle[4];          /**< GL_EXT_texture_swizzle */
   GLushort _Swizzle;          /**< same as Swizzle, but SWIZZLE_* format */
   GLenum16 DepthMode;         /**< GL_ARB_depth_texture */
   GLenum16 ImageFormatCompatibilityType; /**< GL_ARB_shader_image_load_store */
   GLushort MinLayer;          /**< GL_ARB_texture_view */
   GLushort NumLayers;         /**< GL_ARB_texture_view */
   GLboolean GenerateMipmap;   /**< GL_SGIS_generate_mipmap */
   GLbyte ImmutableLevels;     /**< ES 3.0 / ARB_texture_view */
   GLubyte MinLevel;           /**< GL_ARB_texture_view */
   GLubyte NumLevels;          /**< GL_ARB_texture_view */
};


typedef enum
{
   WRAP_S = (1<<0),
   WRAP_T = (1<<1),
   WRAP_R = (1<<2),
} gl_sampler_wrap;

/**
 * Sampler object state.  These objects are new with GL_ARB_sampler_objects
 * and OpenGL 3.3.  Legacy texture objects also contain a sampler object.
 */
struct gl_sampler_object
{
   GLuint Name;
   GLchar *Label;               /**< GL_KHR_debug */
   GLint RefCount;

   struct gl_sampler_attrib Attrib;  /**< State saved by glPushAttrib */

   uint8_t glclamp_mask; /**< mask of GL_CLAMP wraps active */

   /** GL_ARB_bindless_texture */
   bool HandleAllocated;
   struct util_dynarray Handles;
};

/**
 * YUV color space that should be used to sample textures backed by YUV
 * images.
 */
enum gl_texture_yuv_color_space
{
   GL_TEXTURE_YUV_COLOR_SPACE_REC601,
   GL_TEXTURE_YUV_COLOR_SPACE_REC709,
   GL_TEXTURE_YUV_COLOR_SPACE_REC2020,
};

/**
 * Texture object state.  Contains the array of mipmap images, border color,
 * wrap modes, filter modes, and shadow/texcompare state.
 */
struct gl_texture_object
{
   GLint RefCount;             /**< reference count */
   GLuint Name;                /**< the user-visible texture object ID */
   GLenum16 Target;            /**< GL_TEXTURE_1D, GL_TEXTURE_2D, etc. */
   GLchar *Label;              /**< GL_KHR_debug */

   struct gl_sampler_object Sampler;
   struct gl_texture_object_attrib Attrib;  /**< State saved by glPushAttrib */

   gl_texture_index TargetIndex; /**< The gl_texture_unit::CurrentTex index.
                                      Only valid when Target is valid. */
   GLbyte _MaxLevel;           /**< actual max mipmap level (q in the spec) */
   GLfloat _MaxLambda;         /**< = _MaxLevel - BaseLevel (q - p in spec) */
   GLint CropRect[4];          /**< GL_OES_draw_texture */
   GLboolean _BaseComplete;    /**< Is the base texture level valid? */
   GLboolean _MipmapComplete;  /**< Is the whole mipmap valid? */
   GLboolean _IsIntegerFormat; /**< Does the texture store integer values? */
   GLboolean _RenderToTexture; /**< Any rendering to this texture? */
   GLboolean Immutable;        /**< GL_ARB_texture_storage */
   GLboolean _IsFloat;         /**< GL_OES_float_texture */
   GLboolean _IsHalfFloat;     /**< GL_OES_half_float_texture */
   bool HandleAllocated;       /**< GL_ARB_bindless_texture */

   /* This should not be restored by glPopAttrib: */
   bool StencilSampling;       /**< Should we sample stencil instead of depth? */

   /** GL_OES_EGL_image_external */
   GLboolean External;
   GLubyte RequiredTextureImageUnits;

   /** GL_EXT_memory_object */
   GLenum16 TextureTiling;

   /** GL_ARB_texture_buffer_object */
   GLenum16 BufferObjectFormat;
   /** Equivalent Mesa format for BufferObjectFormat. */
   mesa_format _BufferObjectFormat;
   /* TODO: BufferObject->Name should be restored by glPopAttrib(GL_TEXTURE_BIT); */
   struct gl_buffer_object *BufferObject;

   /** GL_ARB_texture_buffer_range */
   GLintptr BufferOffset;
   GLsizeiptr BufferSize; /**< if this is -1, use BufferObject->Size instead */

   /** Actual texture images, indexed by [cube face] and [mipmap level] */
   struct gl_texture_image *Image[MAX_FACES][MAX_TEXTURE_LEVELS];

   /** GL_ARB_bindless_texture */
   struct util_dynarray SamplerHandles;
   struct util_dynarray ImageHandles;

   /** GL_ARB_sparse_texture */
   GLboolean IsSparse;
   GLint VirtualPageSizeIndex;
   GLint NumSparseLevels;

   /* The texture must include at levels [0..lastLevel] once validated:
    */
   GLuint lastLevel;

   unsigned Swizzle;
   unsigned SwizzleGLSL130;

   unsigned int validated_first_level;
   unsigned int validated_last_level;

   /* On validation any active images held in main memory or in other
    * textures will be copied to this texture and the old storage freed.
    */
   struct pipe_resource *pt;

   /* Protect modifications of the sampler_views array */
   simple_mtx_t validate_mutex;

   /* Container of sampler views (one per context) attached to this texture
    * object. Created lazily on first binding in context.
    *
    * Purely read-only accesses to the current context's own sampler view
    * require no locking. Another thread may simultaneously replace the
    * container object in order to grow the array, but the old container will
    * be kept alive.
    *
    * Writing to the container (even for modifying the current context's own
    * sampler view) always requires taking the validate_mutex to protect against
    * concurrent container switches.
    *
    * NULL'ing another context's sampler view is allowed only while
    * implementing an API call that modifies the texture: an application which
    * calls those while simultaneously reading the texture in another context
    * invokes undefined behavior. (TODO: a dubious violation of this rule is
    * st_finalize_texture, which is a lazy operation that corresponds to a
    * texture modification.)
    */
   struct st_sampler_views *sampler_views;

   /* Old sampler views container objects that have not been freed yet because
    * other threads/contexts may still be reading from them.
    */
   struct st_sampler_views *sampler_views_old;

   /* True if this texture comes from the window system. Such a texture
    * cannot be reallocated and the format can only be changed with a sampler
    * view or a surface.
    */
   GLboolean surface_based;

   /* If surface_based is true, this format should be used for all sampler
    * views and surfaces instead of pt->format.
    */
   enum pipe_format surface_format;

   /* If surface_based is true and surface_format is a YUV format, these
    * settings should be used to convert from YUV to RGB.
    */
   enum gl_texture_yuv_color_space yuv_color_space;
   bool yuv_full_range;

   /* When non-negative, samplers should use this level instead of the level
    * range specified by the GL state.
    *
    * This is used for EGL images, which may correspond to a single level out
    * of an imported pipe_resources with multiple mip levels.
    */
   int level_override;

   /* When non-negative, samplers should use this layer instead of the one
    * specified by the GL state.
    *
    * This is used for EGL images and VDPAU interop, where imported
    * pipe_resources may be cube, 3D, or array textures (containing layers
    * with different fields in the case of VDPAU) even though the GL state
    * describes one non-array texture per field.
    */
   int layer_override;

    /**
     * Set when the texture images of this texture object might not all be in
     * the pipe_resource *pt above.
     */
    bool needs_validation;
};


/** Up to four combiner sources are possible with GL_NV_texture_env_combine4 */
#define MAX_COMBINER_TERMS 4


/**
 * Texture combine environment state.
 */
struct gl_tex_env_combine_state
{
   GLenum16 ModeRGB;       /**< GL_REPLACE, GL_DECAL, GL_ADD, etc. */
   GLenum16 ModeA;         /**< GL_REPLACE, GL_DECAL, GL_ADD, etc. */
   /** Source terms: GL_PRIMARY_COLOR, GL_TEXTURE, etc */
   GLenum16 SourceRGB[MAX_COMBINER_TERMS];
   GLenum16 SourceA[MAX_COMBINER_TERMS];
   /** Source operands: GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR, etc */
   GLenum16 OperandRGB[MAX_COMBINER_TERMS];
   GLenum16 OperandA[MAX_COMBINER_TERMS];
   GLubyte ScaleShiftRGB; /**< 0, 1 or 2 */
   GLubyte ScaleShiftA;   /**< 0, 1 or 2 */
   GLubyte _NumArgsRGB;   /**< Number of inputs used for the RGB combiner */
   GLubyte _NumArgsA;     /**< Number of inputs used for the A combiner */
};


/** Compressed TexEnv effective Combine mode */
enum gl_tex_env_mode
{
   TEXENV_MODE_REPLACE,                 /* r = a0 */
   TEXENV_MODE_MODULATE,                /* r = a0 * a1 */
   TEXENV_MODE_ADD,                     /* r = a0 + a1 */
   TEXENV_MODE_ADD_SIGNED,              /* r = a0 + a1 - 0.5 */
   TEXENV_MODE_INTERPOLATE,             /* r = a0 * a2 + a1 * (1 - a2) */
   TEXENV_MODE_SUBTRACT,                /* r = a0 - a1 */
   TEXENV_MODE_DOT3_RGB,                /* r = a0 . a1 */
   TEXENV_MODE_DOT3_RGB_EXT,            /* r = a0 . a1 */
   TEXENV_MODE_DOT3_RGBA,               /* r = a0 . a1 */
   TEXENV_MODE_DOT3_RGBA_EXT,           /* r = a0 . a1 */
   TEXENV_MODE_MODULATE_ADD_ATI,        /* r = a0 * a2 + a1 */
   TEXENV_MODE_MODULATE_SIGNED_ADD_ATI, /* r = a0 * a2 + a1 - 0.5 */
   TEXENV_MODE_MODULATE_SUBTRACT_ATI,   /* r = a0 * a2 - a1 */
   TEXENV_MODE_ADD_PRODUCTS_NV,         /* r = a0 * a1 + a2 * a3 */
   TEXENV_MODE_ADD_PRODUCTS_SIGNED_NV,  /* r = a0 * a1 + a2 * a3 - 0.5 */
};


/** Compressed TexEnv Combine source */
enum gl_tex_env_source
{
   TEXENV_SRC_TEXTURE0,
   TEXENV_SRC_TEXTURE1,
   TEXENV_SRC_TEXTURE2,
   TEXENV_SRC_TEXTURE3,
   TEXENV_SRC_TEXTURE4,
   TEXENV_SRC_TEXTURE5,
   TEXENV_SRC_TEXTURE6,
   TEXENV_SRC_TEXTURE7,
   TEXENV_SRC_TEXTURE,
   TEXENV_SRC_PREVIOUS,
   TEXENV_SRC_PRIMARY_COLOR,
   TEXENV_SRC_CONSTANT,
   TEXENV_SRC_ZERO,
   TEXENV_SRC_ONE,
};


/** Compressed TexEnv Combine operand */
enum gl_tex_env_operand
{
   TEXENV_OPR_COLOR,
   TEXENV_OPR_ONE_MINUS_COLOR,
   TEXENV_OPR_ALPHA,
   TEXENV_OPR_ONE_MINUS_ALPHA,
};


/** Compressed TexEnv Combine argument */
struct gl_tex_env_argument
{
#ifdef __GNUC__
   __extension__ uint8_t Source:4;  /**< TEXENV_SRC_x */
   __extension__ uint8_t Operand:2; /**< TEXENV_OPR_x */
#else
   uint8_t Source;  /**< SRC_x */
   uint8_t Operand; /**< OPR_x */
#endif
};


/***
 * Compressed TexEnv Combine state.
 */
struct gl_tex_env_combine_packed
{
   uint32_t ModeRGB:4;        /**< Effective mode for RGB as 4 bits */
   uint32_t ModeA:4;          /**< Effective mode for RGB as 4 bits */
   uint32_t ScaleShiftRGB:2;  /**< 0, 1 or 2 */
   uint32_t ScaleShiftA:2;    /**< 0, 1 or 2 */
   uint32_t NumArgsRGB:3;     /**< Number of inputs used for the RGB combiner */
   uint32_t NumArgsA:3;       /**< Number of inputs used for the A combiner */
   /** Source arguments in a packed manner */
   struct gl_tex_env_argument ArgsRGB[MAX_COMBINER_TERMS];
   struct gl_tex_env_argument ArgsA[MAX_COMBINER_TERMS];
};


/**
 * TexGenEnabled flags.
 */
/*@{*/
#define S_BIT 1
#define T_BIT 2
#define R_BIT 4
#define Q_BIT 8
#define STR_BITS (S_BIT | T_BIT | R_BIT)
/*@}*/


/**
 * Bit flag versions of the corresponding GL_ constants.
 */
/*@{*/
#define TEXGEN_SPHERE_MAP        0x1
#define TEXGEN_OBJ_LINEAR        0x2
#define TEXGEN_EYE_LINEAR        0x4
#define TEXGEN_REFLECTION_MAP_NV 0x8
#define TEXGEN_NORMAL_MAP_NV     0x10

#define TEXGEN_NEED_NORMALS   (TEXGEN_SPHERE_MAP        | \
                               TEXGEN_REFLECTION_MAP_NV | \
                               TEXGEN_NORMAL_MAP_NV)
#define TEXGEN_NEED_EYE_COORD (TEXGEN_SPHERE_MAP        | \
                               TEXGEN_REFLECTION_MAP_NV | \
                               TEXGEN_NORMAL_MAP_NV     | \
                               TEXGEN_EYE_LINEAR)
/*@}*/



/** Tex-gen enabled for texture unit? */
#define ENABLE_TEXGEN(unit) (1 << (unit))

/** Non-identity texture matrix for texture unit? */
#define ENABLE_TEXMAT(unit) (1 << (unit))


/**
 * Texture coord generation state.
 */
struct gl_texgen
{
   GLenum16 Mode;       /**< GL_EYE_LINEAR, GL_SPHERE_MAP, etc */
   GLbitfield8 _ModeBit; /**< TEXGEN_x bit corresponding to Mode */
};


/**
 * Sampler-related subset of a texture unit, like current texture objects.
 */
struct gl_texture_unit
{
   GLfloat LodBias;		/**< for biasing mipmap levels */
   float LodBiasQuantized;      /**< to reduce pipe_sampler_state variants */

   /** Texture targets that have a non-default texture bound */
   GLbitfield _BoundTextures;

   /** Current sampler object (GL_ARB_sampler_objects) */
   struct gl_sampler_object *Sampler;

   /** Current texture object pointers */
   struct gl_texture_object *CurrentTex[NUM_TEXTURE_TARGETS];

   /** Points to highest priority, complete and enabled texture object */
   struct gl_texture_object *_Current;
};

enum {
   GEN_S,
   GEN_T,
   GEN_R,
   GEN_Q,
   NUM_GEN,
};

/**
 * Fixed-function-related subset of a texture unit, like enable flags,
 * texture environment/function/combiners, and texgen state.
 */
struct gl_fixedfunc_texture_unit
{
   GLbitfield16 Enabled;          /**< bitmask of TEXTURE_*_BIT flags */

   GLenum16 EnvMode;            /**< GL_MODULATE, GL_DECAL, GL_BLEND, etc. */
   GLclampf EnvColor[4];
   GLfloat EnvColorUnclamped[4];

   struct gl_texgen GenS;
   struct gl_texgen GenT;
   struct gl_texgen GenR;
   struct gl_texgen GenQ;

   GLfloat EyePlane[NUM_GEN][4];
   GLfloat ObjectPlane[NUM_GEN][4];

   GLbitfield8 TexGenEnabled;	/**< Bitwise-OR of [STRQ]_BIT values */
   GLbitfield8 _GenFlags;	/**< Bitwise-OR of Gen[STRQ]._ModeBit */

   /**
    * \name GL_EXT_texture_env_combine
    */
   struct gl_tex_env_combine_state Combine;

   /**
    * Derived state based on \c EnvMode and the \c BaseFormat of the
    * currently enabled texture.
    */
   struct gl_tex_env_combine_state _EnvMode;

   /** Current compressed TexEnv & Combine state */
   struct gl_tex_env_combine_packed _CurrentCombinePacked;

   /**
    * Currently enabled combiner state.  This will point to either
    * \c Combine or \c _EnvMode.
    */
   struct gl_tex_env_combine_state *_CurrentCombine;
};


/**
 * Texture attribute group (GL_TEXTURE_BIT).
 */
struct gl_texture_attrib
{
   struct gl_texture_object *ProxyTex[NUM_TEXTURE_TARGETS];

   /** GL_ARB_texture_buffer_object */
   struct gl_buffer_object *BufferObject;

   GLuint CurrentUnit;   /**< GL_ACTIVE_TEXTURE */

   /** Texture coord units/sets used for fragment texturing */
   GLbitfield8 _EnabledCoordUnits;

   /** Texture coord units that have texgen enabled */
   GLbitfield8 _TexGenEnabled;

   /** Texture coord units that have non-identity matrices */
   GLbitfield8 _TexMatEnabled;

   /** Bitwise-OR of all Texture.Unit[i]._GenFlags */
   GLbitfield8 _GenFlags;

   /** Largest index of a texture unit with _Current != NULL. */
   GLshort _MaxEnabledTexImageUnit;

   /** Largest index + 1 of texture units that have had any CurrentTex set. */
   GLubyte NumCurrentTexUsed;

   /** GL_ARB_seamless_cubemap */
   GLboolean CubeMapSeamless;

   struct gl_texture_unit Unit[MAX_COMBINED_TEXTURE_IMAGE_UNITS];
   struct gl_fixedfunc_texture_unit FixedFuncUnit[MAX_TEXTURE_COORD_UNITS];
};


/**
 * Data structure representing a single clip plane (e.g. one of the elements
 * of the ctx->Transform.EyeUserPlane or ctx->Transform._ClipUserPlane array).
 */
typedef GLfloat gl_clip_plane[4];


/**
 * Transformation attribute group (GL_TRANSFORM_BIT).
 */
struct gl_transform_attrib
{
   GLenum16 MatrixMode;				/**< Matrix mode */
   gl_clip_plane EyeUserPlane[MAX_CLIP_PLANES];	/**< User clip planes */
   gl_clip_plane _ClipUserPlane[MAX_CLIP_PLANES]; /**< derived */
   GLbitfield ClipPlanesEnabled;                /**< on/off bitmask */
   GLboolean Normalize;				/**< Normalize all normals? */
   GLboolean RescaleNormals;			/**< GL_EXT_rescale_normal */
   GLboolean RasterPositionUnclipped;           /**< GL_IBM_rasterpos_clip */
   GLboolean DepthClampNear;			/**< GL_AMD_depth_clamp_separate */
   GLboolean DepthClampFar;			/**< GL_AMD_depth_clamp_separate */
   /** GL_ARB_clip_control */
   GLenum16 ClipOrigin;   /**< GL_LOWER_LEFT or GL_UPPER_LEFT */
   GLenum16 ClipDepthMode;/**< GL_NEGATIVE_ONE_TO_ONE or GL_ZERO_TO_ONE */
};


/**
 * Viewport attribute group (GL_VIEWPORT_BIT).
 */
struct gl_viewport_attrib
{
   GLfloat X, Y;		/**< position */
   GLfloat Width, Height;	/**< size */
   GLfloat Near, Far;		/**< Depth buffer range */

   /**< GL_NV_viewport_swizzle */
   GLenum16 SwizzleX, SwizzleY, SwizzleZ, SwizzleW;
};


/**
 * Fields describing a mapped buffer range.
 */
struct gl_buffer_mapping
{
   GLbitfield AccessFlags; /**< Mask of GL_MAP_x_BIT flags */
   GLvoid *Pointer;        /**< User-space address of mapping */
   GLintptr Offset;        /**< Mapped offset */
   GLsizeiptr Length;      /**< Mapped length */
};


/**
 * Usages we've seen for a buffer object.
 */
typedef enum
{
   USAGE_UNIFORM_BUFFER = 0x1,
   USAGE_TEXTURE_BUFFER = 0x2,
   USAGE_ATOMIC_COUNTER_BUFFER = 0x4,
   USAGE_SHADER_STORAGE_BUFFER = 0x8,
   USAGE_TRANSFORM_FEEDBACK_BUFFER = 0x10,
   USAGE_PIXEL_PACK_BUFFER = 0x20,
   USAGE_ARRAY_BUFFER = 0x40,
   USAGE_DISABLE_MINMAX_CACHE = 0x100,
} gl_buffer_usage;


/**
 * GL_ARB_vertex/pixel_buffer_object buffer object
 */
struct gl_buffer_object
{
   GLint RefCount;
   GLuint Name;

   /**
    * The context that holds a global buffer reference for the lifetime of
    * the GL buffer ID to skip refcounting for all its private bind points.
    * Other contexts must still do refcounting as usual. Shared binding points
    * like TBO within gl_texture_object are always refcounted.
    *
    * Implementation details:
    * - Only the context that creates the buffer ("creating context") skips
    *   refcounting.
    * - Only buffers represented by an OpenGL buffer ID skip refcounting.
    *   Other internal buffers don't. (glthread requires refcounting for
    *   internal buffers, etc.)
    * - glDeleteBuffers removes the global buffer reference and increments
    *   RefCount for all private bind points where the deleted buffer is bound
    *   (e.g. unbound VAOs that are not changed by glDeleteBuffers),
    *   effectively enabling refcounting for that context. This is the main
    *   point where the global buffer reference is removed.
    * - glDeleteBuffers called from a different context adds the buffer into
    *   the ZombieBufferObjects list, which is a way to notify the creating
    *   context that it should remove its global buffer reference to allow
    *   freeing the buffer. The creating context walks over that list in a few
    *   GL functions.
    * - xxxDestroyContext walks over all buffers and removes its global
    *   reference from those buffers that it created.
    */
   struct gl_context *Ctx;
   GLint CtxRefCount;   /**< Non-atomic references held by Ctx. */

   gl_buffer_usage UsageHistory; /**< How has this buffer been used so far? */

   struct pipe_resource *buffer;
   struct gl_context *private_refcount_ctx;
   /* This mechanism allows passing buffer references to the driver without
    * using atomics to increase the reference count.
    *
    * This private refcount can be decremented without atomics but only one
    * context (ctx above) can use this counter to be thread-safe.
    *
    * This number is atomically added to buffer->reference.count at
    * initialization. If it's never used, the same number is atomically
    * subtracted from buffer->reference.count before destruction. If this
    * number is decremented, we can pass that reference to the driver without
    * touching reference.count. At buffer destruction we only subtract
    * the number of references we did not return. This can possibly turn
    * a million atomic increments into 1 add and 1 subtract atomic op.
    */
   int private_refcount;

   GLbitfield StorageFlags; /**< GL_MAP_PERSISTENT_BIT, etc. */

   /** Memoization of min/max index computations for static index buffers */
   unsigned MinMaxCacheHitIndices;
   unsigned MinMaxCacheMissIndices;
   struct hash_table *MinMaxCache;
   simple_mtx_t MinMaxCacheMutex;
   bool MinMaxCacheDirty:1;

   bool DeletePending:1;  /**< true if buffer object is removed from the hash */
   bool Immutable:1;    /**< GL_ARB_buffer_storage */
   bool HandleAllocated:1; /**< GL_ARB_bindless_texture */
   GLenum16 Usage;      /**< GL_STREAM_DRAW_ARB, GL_STREAM_READ_ARB, etc. */
   GLchar *Label;       /**< GL_KHR_debug */
   GLsizeiptrARB Size;  /**< Size of buffer storage in bytes */

   /** Counters used for buffer usage warnings */
   GLuint NumSubDataCalls;
   GLuint NumMapBufferWriteCalls;

   struct gl_buffer_mapping Mappings[MAP_COUNT];
   struct pipe_transfer *transfer[MAP_COUNT];
};


/**
 * Client pixel packing/unpacking attributes
 */
struct gl_pixelstore_attrib
{
   GLint Alignment;
   GLint RowLength;
   GLint SkipPixels;
   GLint SkipRows;
   GLint ImageHeight;
   GLint SkipImages;
   GLboolean SwapBytes;
   GLboolean LsbFirst;
   GLboolean Invert;        /**< GL_MESA_pack_invert */
   GLint CompressedBlockWidth;   /**< GL_ARB_compressed_texture_pixel_storage */
   GLint CompressedBlockHeight;
   GLint CompressedBlockDepth;
   GLint CompressedBlockSize;
   struct gl_buffer_object *BufferObj; /**< GL_ARB_pixel_buffer_object */
};


/**
 * Enum for defining the mapping for the position/generic0 attribute.
 *
 * Do not change the order of the values as these are used as
 * array indices.
 */
typedef enum
{
   ATTRIBUTE_MAP_MODE_IDENTITY, /**< 1:1 mapping */
   ATTRIBUTE_MAP_MODE_POSITION, /**< get position and generic0 from position */
   ATTRIBUTE_MAP_MODE_GENERIC0, /**< get position and generic0 from generic0 */
   ATTRIBUTE_MAP_MODE_MAX       /**< for sizing arrays */
} gl_attribute_map_mode;


/**
 * Attributes to describe a vertex array.
 *
 * Contains the size, type, format and normalization flag,
 * along with the index of a vertex buffer binding point.
 *
 * Note that the Stride field corresponds to VERTEX_ATTRIB_ARRAY_STRIDE
 * and is only present for backwards compatibility reasons.
 * Rendering always uses VERTEX_BINDING_STRIDE.
 * The gl*Pointer() functions will set VERTEX_ATTRIB_ARRAY_STRIDE
 * and VERTEX_BINDING_STRIDE to the same value, while
 * glBindVertexBuffer() will only set VERTEX_BINDING_STRIDE.
 */
struct gl_array_attributes
{
   /** Points to client array data. Not used when a VBO is bound */
   const GLubyte *Ptr;
   /** Offset of the first element relative to the binding offset */
   GLuint RelativeOffset;
   /** Vertex format */
   struct gl_vertex_format Format;
   /** Stride as specified with gl*Pointer() */
   GLshort Stride;
   /** Index into gl_vertex_array_object::BufferBinding[] array */
   GLubyte BufferBindingIndex;

   /**
    * Derived effective buffer binding index
    *
    * Index into the gl_vertex_buffer_binding array of the vao.
    * Similar to BufferBindingIndex, but with the mapping of the
    * position/generic0 attributes applied and with identical
    * gl_vertex_buffer_binding entries collapsed to a single
    * entry within the vao.
    *
    * The value is valid past calling _mesa_update_vao_derived_arrays.
    * Note that _mesa_update_vao_derived_arrays is called when binding
    * the VAO to Array._DrawVAO.
    */
   GLubyte _EffBufferBindingIndex;
   /**
    * Derived effective relative offset.
    *
    * Relative offset to the effective buffers offset in
    * gl_vertex_buffer_binding::_EffOffset.
    *
    * The value is valid past calling _mesa_update_vao_derived_arrays.
    * Note that _mesa_update_vao_derived_arrays is called when binding
    * the VAO to Array._DrawVAO.
    */
   GLushort _EffRelativeOffset;
};


/**
 * This describes the buffer object used for a vertex array (or
 * multiple vertex arrays).  If BufferObj points to the default/null
 * buffer object, then the vertex array lives in user memory and not a VBO.
 */
struct gl_vertex_buffer_binding
{
   GLintptr Offset;                    /**< User-specified offset */
   GLsizei Stride;                     /**< User-specified stride */
   GLuint InstanceDivisor;             /**< GL_ARB_instanced_arrays */
   struct gl_buffer_object *BufferObj; /**< GL_ARB_vertex_buffer_object */
   GLbitfield _BoundArrays;            /**< Arrays bound to this binding point */

   /**
    * Derived effective bound arrays.
    *
    * The effective binding handles enabled arrays past the
    * position/generic0 attribute mapping and reduces the refered
    * gl_vertex_buffer_binding entries to a unique subset.
    *
    * The value is valid past calling _mesa_update_vao_derived_arrays.
    * Note that _mesa_update_vao_derived_arrays is called when binding
    * the VAO to Array._DrawVAO.
    */
   GLbitfield _EffBoundArrays;
   /**
    * Derived offset.
    *
    * The absolute offset to that we can collapse some attributes
    * to this unique effective binding.
    * For user space array bindings this contains the smallest pointer value
    * in the bound and interleaved arrays.
    * For VBO bindings this contains an offset that lets the attributes
    * _EffRelativeOffset stay positive and in bounds with
    * Const.MaxVertexAttribRelativeOffset
    *
    * The value is valid past calling _mesa_update_vao_derived_arrays.
    * Note that _mesa_update_vao_derived_arrays is called when binding
    * the VAO to Array._DrawVAO.
    */
   GLintptr _EffOffset;
};


/**
 * A representation of "Vertex Array Objects" (VAOs) from OpenGL 3.1+ /
 * the GL_ARB_vertex_array_object extension.
 */
struct gl_vertex_array_object
{
   /** Name of the VAO as received from glGenVertexArray. */
   GLuint Name;

   GLint RefCount;

   GLchar *Label;       /**< GL_KHR_debug */

   /**
    * Has this array object been bound?
    */
   GLboolean EverBound;

   /**
    * Whether the VAO is changed by the application so often that some of
    * the derived fields are not updated at all to decrease overhead.
    * Also, interleaved arrays are not detected, because it's too expensive
    * to do that before every draw call.
    */
   bool IsDynamic;

   /**
    * Marked to true if the object is shared between contexts and immutable.
    * Then reference counting is done using atomics and thread safe.
    * Is used for dlist VAOs.
    */
   bool SharedAndImmutable;

   /**
    * Number of updates that were done by the application. This is used to
    * decide whether the VAO is static or dynamic.
    */
   unsigned NumUpdates;

   /** Vertex attribute arrays */
   struct gl_array_attributes VertexAttrib[VERT_ATTRIB_MAX];

   /** Vertex buffer bindings */
   struct gl_vertex_buffer_binding BufferBinding[VERT_ATTRIB_MAX];

   /** Mask indicating which vertex arrays have vertex buffer associated. */
   GLbitfield VertexAttribBufferMask;

   /** Mask indicating which vertex arrays have a non-zero instance divisor. */
   GLbitfield NonZeroDivisorMask;

   /** Mask of VERT_BIT_* values indicating which arrays are enabled */
   GLbitfield Enabled;

   /**
    * Mask indicating which VertexAttrib and BufferBinding structures have
    * been changed since the VAO creation. No bit is ever cleared to 0 by
    * state updates. Setting to the default state doesn't update this.
    * (e.g. unbinding) Setting the derived state (_* fields) doesn't update
    * this either.
    */
   GLbitfield NonDefaultStateMask;

   /** Denotes the way the position/generic0 attribute is mapped */
   gl_attribute_map_mode _AttributeMapMode;

   /** "Enabled" with the position/generic0 attribute aliasing resolved */
   GLbitfield _EnabledWithMapMode;

   /** The index buffer (also known as the element array buffer in OpenGL). */
   struct gl_buffer_object *IndexBufferObj;
};


/**
 * Vertex array state
 */
struct gl_array_attrib
{
   /** Currently bound array object. */
   struct gl_vertex_array_object *VAO;

   /** The default vertex array object */
   struct gl_vertex_array_object *DefaultVAO;

   /** The last VAO accessed by a DSA function */
   struct gl_vertex_array_object *LastLookedUpVAO;

   /** These contents are copied to newly created VAOs. */
   struct gl_vertex_array_object DefaultVAOState;

   /** Array objects (GL_ARB_vertex_array_object) */
   struct _mesa_HashTable *Objects;

   GLint ActiveTexture;		/**< Client Active Texture */
   GLuint LockFirst;            /**< GL_EXT_compiled_vertex_array */
   GLuint LockCount;            /**< GL_EXT_compiled_vertex_array */

   /**
    * \name Primitive restart controls
    *
    * Primitive restart is enabled if either \c PrimitiveRestart or
    * \c PrimitiveRestartFixedIndex is set.
    */
   /*@{*/
   GLboolean PrimitiveRestart;
   GLboolean PrimitiveRestartFixedIndex;
   GLboolean _PrimitiveRestart[3]; /**< Enable indexed by index_size_shift. */
   GLuint RestartIndex;
   GLuint _RestartIndex[3]; /**< Restart indices indexed by index_size_shift. */
   /*@}*/

   /* GL_ARB_vertex_buffer_object */
   struct gl_buffer_object *ArrayBufferObj;

   /**
    * Vertex array object that is used with the currently active draw command.
    * The _DrawVAO is either set to the currently bound VAO for array type
    * draws or to internal VAO's set up by the vbo module to execute immediate
    * mode or display list draws.
    */
   struct gl_vertex_array_object *_DrawVAO;

   /**
    * Whether per-vertex edge flags are enabled and should be processed by
    * the vertex shader.
    */
   bool _PerVertexEdgeFlagsEnabled;

   /**
    * Whether all edge flags are false, causing all points and lines generated
    * by polygon mode to be not drawn. (i.e. culled)
    */
   bool _PolygonModeAlwaysCulls;

   /**
    * If gallium vertex buffers are dirty, this flag indicates whether gallium
    * vertex elements are dirty too. If this is false, GL states corresponding
    * to vertex elements have not been changed. Thus, this affects what will
    * happen when ST_NEW_VERTEX_ARRAYS is set.
    *
    * The driver should clear this when it's done.
    */
   bool NewVertexElements;

   /** Legal array datatypes and the API for which they have been computed */
   GLbitfield LegalTypesMask;
   gl_api LegalTypesMaskAPI;
};


/**
 * Feedback buffer state
 */
struct gl_feedback
{
   GLenum16 Type;
   GLbitfield _Mask;    /**< FB_* bits */
   GLfloat *Buffer;
   GLuint BufferSize;
   GLuint Count;
};


/**
 * Selection buffer state
 */
struct gl_selection
{
   GLuint *Buffer;	/**< selection buffer */
   GLuint BufferSize;	/**< size of the selection buffer */
   GLuint BufferCount;	/**< number of values in the selection buffer */
   GLuint Hits;		/**< number of records in the selection buffer */
   GLuint NameStackDepth; /**< name stack depth */
   GLuint NameStack[MAX_NAME_STACK_DEPTH]; /**< name stack */
   GLboolean HitFlag;	/**< hit flag */
   GLfloat HitMinZ;	/**< minimum hit depth */
   GLfloat HitMaxZ;	/**< maximum hit depth */

   /* HW GL_SELECT */
   void *SaveBuffer;        /**< array holds multi stack data */
   GLuint SaveBufferTail;   /**< offset to SaveBuffer's tail */
   GLuint SavedStackNum;    /**< number of saved stacks */

   GLboolean ResultUsed;    /**< whether any draw used result buffer */
   GLuint ResultOffset;     /**< offset into result buffer */
   struct gl_buffer_object *Result; /**< result buffer */
};


/**
 * 1-D Evaluator control points
 */
struct gl_1d_map
{
   GLuint Order;	/**< Number of control points */
   GLfloat u1, u2, du;	/**< u1, u2, 1.0/(u2-u1) */
   GLfloat *Points;	/**< Points to contiguous control points */
};


/**
 * 2-D Evaluator control points
 */
struct gl_2d_map
{
   GLuint Uorder;		/**< Number of control points in U dimension */
   GLuint Vorder;		/**< Number of control points in V dimension */
   GLfloat u1, u2, du;
   GLfloat v1, v2, dv;
   GLfloat *Points;		/**< Points to contiguous control points */
};


/**
 * All evaluator control point state
 */
struct gl_evaluators
{
   /**
    * \name 1-D maps
    */
   /*@{*/
   struct gl_1d_map Map1Vertex3;
   struct gl_1d_map Map1Vertex4;
   struct gl_1d_map Map1Index;
   struct gl_1d_map Map1Color4;
   struct gl_1d_map Map1Normal;
   struct gl_1d_map Map1Texture1;
   struct gl_1d_map Map1Texture2;
   struct gl_1d_map Map1Texture3;
   struct gl_1d_map Map1Texture4;
   /*@}*/

   /**
    * \name 2-D maps
    */
   /*@{*/
   struct gl_2d_map Map2Vertex3;
   struct gl_2d_map Map2Vertex4;
   struct gl_2d_map Map2Index;
   struct gl_2d_map Map2Color4;
   struct gl_2d_map Map2Normal;
   struct gl_2d_map Map2Texture1;
   struct gl_2d_map Map2Texture2;
   struct gl_2d_map Map2Texture3;
   struct gl_2d_map Map2Texture4;
   /*@}*/
};


/**
 * Transform feedback object state
 */
struct gl_transform_feedback_object
{
   GLuint Name;  /**< AKA the object ID */
   GLint RefCount;
   GLchar *Label;     /**< GL_KHR_debug */
   GLboolean Active;  /**< Is transform feedback enabled? */
   GLboolean Paused;  /**< Is transform feedback paused? */
   GLboolean EndedAnytime; /**< Has EndTransformFeedback been called
                                at least once? */
   GLboolean EverBound; /**< Has this object been bound? */

   /**
    * GLES: if Active is true, remaining number of primitives which can be
    * rendered without overflow.  This is necessary to track because GLES
    * requires us to generate INVALID_OPERATION if a call to glDrawArrays or
    * glDrawArraysInstanced would overflow transform feedback buffers.
    * Undefined if Active is false.
    *
    * Not tracked for desktop GL since it's unnecessary.
    */
   unsigned GlesRemainingPrims;

   /**
    * The program active when BeginTransformFeedback() was called.
    * When active and unpaused, this equals ctx->Shader.CurrentProgram[stage],
    * where stage is the pipeline stage that is the source of data for
    * transform feedback.
    */
   struct gl_program *program;

   /** The feedback buffers */
   GLuint BufferNames[MAX_FEEDBACK_BUFFERS];
   struct gl_buffer_object *Buffers[MAX_FEEDBACK_BUFFERS];

   /** Start of feedback data in dest buffer */
   GLintptr Offset[MAX_FEEDBACK_BUFFERS];

   /**
    * Max data to put into dest buffer (in bytes).  Computed based on
    * RequestedSize and the actual size of the buffer.
    */
   GLsizeiptr Size[MAX_FEEDBACK_BUFFERS];

   /**
    * Size that was specified when the buffer was bound.  If the buffer was
    * bound with glBindBufferBase() or glBindBufferOffsetEXT(), this value is
    * zero.
    */
   GLsizeiptr RequestedSize[MAX_FEEDBACK_BUFFERS];

   unsigned num_targets;
   struct pipe_stream_output_target *targets[PIPE_MAX_SO_BUFFERS];

   /* This encapsulates the count that can be used as a source for draw_vbo.
    * It contains stream output targets from the last call of
    * EndTransformFeedback for each stream. */
   struct pipe_stream_output_target *draw_count[MAX_VERTEX_STREAMS];
};


/**
 * Context state for transform feedback.
 */
struct gl_transform_feedback_state
{
   GLenum16 Mode;     /**< GL_POINTS, GL_LINES or GL_TRIANGLES */

   /** The general binding point (GL_TRANSFORM_FEEDBACK_BUFFER) */
   struct gl_buffer_object *CurrentBuffer;

   /** The table of all transform feedback objects */
   struct _mesa_HashTable *Objects;

   /** The current xform-fb object (GL_TRANSFORM_FEEDBACK_BINDING) */
   struct gl_transform_feedback_object *CurrentObject;

   /** The default xform-fb object (Name==0) */
   struct gl_transform_feedback_object *DefaultObject;
};


/**
 * A "performance monitor" as described in AMD_performance_monitor.
 */
struct gl_perf_monitor_object
{
   GLuint Name;

   /** True if the monitor is currently active (Begin called but not End). */
   GLboolean Active;

   /**
    * True if the monitor has ended.
    *
    * This is distinct from !Active because it may never have began.
    */
   GLboolean Ended;

   /**
    * A list of groups with currently active counters.
    *
    * ActiveGroups[g] == n if there are n counters active from group 'g'.
    */
   unsigned *ActiveGroups;

   /**
    * An array of bitsets, subscripted by group ID, then indexed by counter ID.
    *
    * Checking whether counter 'c' in group 'g' is active can be done via:
    *
    *    BITSET_TEST(ActiveCounters[g], c)
    */
   GLuint **ActiveCounters;

   unsigned num_active_counters;

   struct gl_perf_counter_object {
      struct pipe_query *query;
      int id;
      int group_id;
      unsigned batch_index;
   } *active_counters;

   struct pipe_query *batch_query;
   union pipe_query_result *batch_result;
};


union gl_perf_monitor_counter_value
{
   float f;
   uint64_t u64;
   uint32_t u32;
};


struct gl_perf_monitor_counter
{
   /** Human readable name for the counter. */
   const char *Name;

   /**
    * Data type of the counter.  Valid values are FLOAT, UNSIGNED_INT,
    * UNSIGNED_INT64_AMD, and PERCENTAGE_AMD.
    */
   GLenum16 Type;

   /** Minimum counter value. */
   union gl_perf_monitor_counter_value Minimum;

   /** Maximum counter value. */
   union gl_perf_monitor_counter_value Maximum;

   unsigned query_type;
   unsigned flags;
};


struct gl_perf_monitor_group
{
   /** Human readable name for the group. */
   const char *Name;

   /**
    * Maximum number of counters in this group which can be active at the
    * same time.
    */
   GLuint MaxActiveCounters;

   /** Array of counters within this group. */
   const struct gl_perf_monitor_counter *Counters;
   GLuint NumCounters;

   bool has_batch;
};

/**
 * A query object instance as described in INTEL_performance_query.
 *
 * NB: We want to keep this and the corresponding backend structure
 * relatively lean considering that applications may expect to
 * allocate enough objects to be able to query around all draw calls
 * in a frame.
 */
struct gl_perf_query_object
{
   GLuint Id;          /**< hash table ID/name */
   unsigned Used:1;    /**< has been used for 1 or more queries */
   unsigned Active:1;  /**< inside Begin/EndPerfQuery */
   unsigned Ready:1;   /**< result is ready? */
};


/**
 * Context state for AMD_performance_monitor.
 */
struct gl_perf_monitor_state
{
   /** Array of performance monitor groups (indexed by group ID) */
   const struct gl_perf_monitor_group *Groups;
   GLuint NumGroups;

   /** The table of all performance monitors. */
   struct _mesa_HashTable *Monitors;
};


/**
 * Context state for INTEL_performance_query.
 */
struct gl_perf_query_state
{
   struct _mesa_HashTable *Objects; /**< The table of all performance query objects */
};


/**
 * State common to vertex and fragment programs.
 */
struct gl_program_state
{
   GLint ErrorPos;                       /* GL_PROGRAM_ERROR_POSITION_ARB/NV */
   const char *ErrorString;              /* GL_PROGRAM_ERROR_STRING_ARB/NV */
};


/**
 * Context state for vertex programs.
 */
struct gl_vertex_program_state
{
   GLboolean Enabled;            /**< User-set GL_VERTEX_PROGRAM_ARB/NV flag */
   GLboolean PointSizeEnabled;   /**< GL_VERTEX_PROGRAM_POINT_SIZE_ARB/NV */
   GLboolean TwoSideEnabled;     /**< GL_VERTEX_PROGRAM_TWO_SIDE_ARB/NV */
   /** Whether the fixed-func program is being used right now. */
   GLboolean _UsesTnlProgram;

   struct gl_program *Current;  /**< User-bound vertex program */

   /** Currently enabled and valid vertex program (including internal
    * programs, user-defined vertex programs and GLSL vertex shaders).
    * This is the program we must use when rendering.
    */
   struct gl_program *_Current;

   GLfloat Parameters[MAX_PROGRAM_ENV_PARAMS][4]; /**< Env params */

   /** Program to emulate fixed-function T&L (see above) */
   struct gl_program *_TnlProgram;

   /** Cache of fixed-function programs */
   struct gl_program_cache *Cache;

   GLboolean _Overriden;

   bool _VPModeOptimizesConstantAttribs;

   /**
    * If we have a vertex program, a TNL program or no program at all.
    * Note that this value should be kept up to date all the time,
    * nevertheless its correctness is asserted in _mesa_update_state.
    * The reason is to avoid calling _mesa_update_state twice we need
    * this value on draw *before* actually calling _mesa_update_state.
    * Also it should need to get recomputed only on changes to the
    * vertex program which are heavyweight already.
    */
   gl_vertex_processing_mode _VPMode;

   GLbitfield _VaryingInputs;  /**< mask of VERT_BIT_* flags */
   GLbitfield _VPModeInputFilter;
};

/**
 * Context state for tessellation control programs.
 */
struct gl_tess_ctrl_program_state
{
   /** Currently bound and valid shader. */
   struct gl_program *_Current;

   GLint patch_vertices;
   GLfloat patch_default_outer_level[4];
   GLfloat patch_default_inner_level[2];
};

/**
 * Context state for tessellation evaluation programs.
 */
struct gl_tess_eval_program_state
{
   /** Currently bound and valid shader. */
   struct gl_program *_Current;
};

/**
 * Context state for geometry programs.
 */
struct gl_geometry_program_state
{
   /**
    * Currently enabled and valid program (including internal programs
    * and compiled shader programs).
    */
   struct gl_program *_Current;
};

/**
 * Context state for fragment programs.
 */
struct gl_fragment_program_state
{
   GLboolean Enabled;     /**< User-set fragment program enable flag */
   /** Whether the fixed-func program is being used right now. */
   GLboolean _UsesTexEnvProgram;

   struct gl_program *Current;  /**< User-bound fragment program */

   /**
    * Currently enabled and valid fragment program (including internal
    * programs, user-defined fragment programs and GLSL fragment shaders).
    * This is the program we must use when rendering.
    */
   struct gl_program *_Current;

   GLfloat Parameters[MAX_PROGRAM_ENV_PARAMS][4]; /**< Env params */

   /** Program to emulate fixed-function texture env/combine (see above) */
   struct gl_program *_TexEnvProgram;

   /** Cache of fixed-function programs */
   struct gl_program_cache *Cache;
};


/**
 * Context state for compute programs.
 */
struct gl_compute_program_state
{
   /** Currently enabled and valid program (including internal programs
    * and compiled shader programs).
    */
   struct gl_program *_Current;
};


/**
 * ATI_fragment_shader runtime state
 */

struct atifs_instruction;
struct atifs_setupinst;

/**
 * ATI fragment shader
 */
struct ati_fragment_shader
{
   GLuint Id;
   GLint RefCount;
   struct atifs_instruction *Instructions[2];
   struct atifs_setupinst *SetupInst[2];
   GLfloat Constants[8][4];
   GLbitfield LocalConstDef;  /**< Indicates which constants have been set */
   GLubyte numArithInstr[2];
   GLubyte regsAssigned[2];
   GLubyte NumPasses;         /**< 1 or 2 */
   /**
    * Current compile stage: 0 setup pass1, 1 arith pass1,
    * 2 setup pass2, 3 arith pass2.
    */
   GLubyte cur_pass;
   GLubyte last_optype;
   GLboolean interpinp1;
   GLboolean isValid;
   /**
    * Array of 2 bit values for each tex unit to remember whether
    * STR or STQ swizzle was used
    */
   GLuint swizzlerq;
   struct gl_program *Program;
};

/**
 * Context state for GL_ATI_fragment_shader
 */
struct gl_ati_fragment_shader_state
{
   GLboolean Enabled;
   GLboolean Compiling;
   GLfloat GlobalConstants[8][4];
   struct ati_fragment_shader *Current;
};

#define GLSL_DUMP      0x1  /**< Dump shaders to stdout */
#define GLSL_LOG       0x2  /**< Write shaders to files */
#define GLSL_UNIFORMS  0x4  /**< Print glUniform calls */
#define GLSL_NOP_VERT  0x8  /**< Force no-op vertex shaders */
#define GLSL_NOP_FRAG 0x10  /**< Force no-op fragment shaders */
#define GLSL_USE_PROG 0x20  /**< Log glUseProgram calls */
#define GLSL_REPORT_ERRORS 0x40  /**< Print compilation errors */
#define GLSL_DUMP_ON_ERROR 0x80 /**< Dump shaders to stderr on compile error */
#define GLSL_CACHE_INFO 0x100 /**< Print debug information about shader cache */
#define GLSL_CACHE_FALLBACK 0x200 /**< Force shader cache fallback paths */
#define GLSL_SOURCE 0x400 /**< Only dump GLSL */


/**
 * Context state for GLSL vertex/fragment shaders.
 * Extended to support pipeline object
 */
struct gl_pipeline_object
{
   /** Name of the pipeline object as received from glGenProgramPipelines.
    * It would be 0 for shaders without separate shader objects.
    */
   GLuint Name;

   GLint RefCount;

   GLchar *Label;   /**< GL_KHR_debug */

   /**
    * Programs used for rendering
    *
    * There is a separate program set for each shader stage.
    */
   struct gl_program *CurrentProgram[MESA_SHADER_STAGES];

   struct gl_shader_program *ReferencedPrograms[MESA_SHADER_STAGES];

   /**
    * Program used by glUniform calls.
    *
    * Explicitly set by \c glUseProgram and \c glActiveProgramEXT.
    */
   struct gl_shader_program *ActiveProgram;

   GLbitfield Flags;         /**< Mask of GLSL_x flags */
   GLboolean EverBound;      /**< Has the pipeline object been created */
   GLboolean Validated;      /**< Pipeline Validation status */
   GLboolean UserValidated;  /**< Validation status initiated by the user */

   GLchar *InfoLog;
};

/**
 * Context state for GLSL pipeline shaders.
 */
struct gl_pipeline_shader_state
{
   /** Currently bound pipeline object. See _mesa_BindProgramPipeline() */
   struct gl_pipeline_object *Current;

   /** Default Object to ensure that _Shader is never NULL */
   struct gl_pipeline_object *Default;

   /** Pipeline objects */
   struct _mesa_HashTable *Objects;
};

/**
 * Occlusion/timer query object.
 */
struct gl_query_object
{
   GLenum16 Target;    /**< The query target, when active */
   GLuint Id;          /**< hash table ID/name */
   GLchar *Label;      /**< GL_KHR_debug */
   GLuint64EXT Result; /**< the counter */
   GLboolean Active;   /**< inside Begin/EndQuery */
   GLboolean Ready;    /**< result is ready? */
   GLboolean EverBound;/**< has query object ever been bound */
   GLuint Stream;      /**< The stream */

   struct pipe_query *pq;

   /* Begin TIMESTAMP query for GL_TIME_ELAPSED_EXT queries */
   struct pipe_query *pq_begin;

   unsigned type;  /**< PIPE_QUERY_x */
};


/**
 * Context state for query objects.
 */
struct gl_query_state
{
   struct _mesa_HashTable *QueryObjects;
   struct gl_query_object *CurrentOcclusionObject; /* GL_ARB_occlusion_query */
   struct gl_query_object *CurrentTimerObject;     /* GL_EXT_timer_query */

   /** GL_NV_conditional_render */
   struct gl_query_object *CondRenderQuery;

   /** GL_EXT_transform_feedback */
   struct gl_query_object *PrimitivesGenerated[MAX_VERTEX_STREAMS];
   struct gl_query_object *PrimitivesWritten[MAX_VERTEX_STREAMS];

   /** GL_ARB_transform_feedback_overflow_query */
   struct gl_query_object *TransformFeedbackOverflow[MAX_VERTEX_STREAMS];
   struct gl_query_object *TransformFeedbackOverflowAny;

   /** GL_ARB_timer_query */
   struct gl_query_object *TimeElapsed;

   /** GL_ARB_pipeline_statistics_query */
   struct gl_query_object *pipeline_stats[MAX_PIPELINE_STATISTICS];

   GLenum16 CondRenderMode;
};


/** Sync object state */
struct gl_sync_object
{
   GLuint Name;               /**< Fence name */
   GLint RefCount;            /**< Reference count */
   GLchar *Label;             /**< GL_KHR_debug */
   GLboolean DeletePending;   /**< Object was deleted while there were still
                               * live references (e.g., sync not yet finished)
                               */
   GLenum16 SyncCondition;
   GLbitfield Flags;          /**< Flags passed to glFenceSync */
   GLuint StatusFlag:1;       /**< Has the sync object been signaled? */

   struct pipe_fence_handle *fence;
   simple_mtx_t mutex; /**< protects "fence" */
};


/**
 * State which can be shared by multiple contexts:
 */
struct gl_shared_state
{
   simple_mtx_t Mutex;		   /**< for thread safety */
   GLint RefCount;			   /**< Reference count */
   bool DisplayListsAffectGLThread;

   struct _mesa_HashTable *DisplayList;	   /**< Display lists hash table */
   struct _mesa_HashTable *TexObjects;	   /**< Texture objects hash table */

   /** Default texture objects (shared by all texture units) */
   struct gl_texture_object *DefaultTex[NUM_TEXTURE_TARGETS];

   /** Fallback texture used when a bound texture is incomplete */
   struct gl_texture_object *FallbackTex[NUM_TEXTURE_TARGETS];

   /**
    * \name Thread safety and statechange notification for texture
    * objects.
    *
    * \todo Improve the granularity of locking.
    */
   /*@{*/
   simple_mtx_t TexMutex;		/**< texobj thread safety */
   GLuint TextureStateStamp;	        /**< state notification for shared tex */
   /*@}*/

   /**
    * \name Vertex/geometry/fragment programs
    */
   /*@{*/
   struct _mesa_HashTable *Programs; /**< All vertex/fragment programs */
   struct gl_program *DefaultVertexProgram;
   struct gl_program *DefaultFragmentProgram;
   /*@}*/

   /* GL_ATI_fragment_shader */
   struct _mesa_HashTable *ATIShaders;
   struct ati_fragment_shader *DefaultFragmentShader;

   struct _mesa_HashTable *BufferObjects;

   /* Buffer objects released by a different context than the one that
    * created them. Since the creating context holds one global buffer
    * reference for each buffer it created and skips reference counting,
    * deleting a buffer by another context can't touch the buffer reference
    * held by the context that created it. Only the creating context can
    * remove its global buffer reference.
    *
    * This list contains all buffers that were deleted by a different context
    * than the one that created them. This list should be probed by all
    * contexts regularly and remove references of those buffers that they own.
    */
   struct set *ZombieBufferObjects;

   /** Table of both gl_shader and gl_shader_program objects */
   struct _mesa_HashTable *ShaderObjects;

   /* GL_EXT_framebuffer_object */
   struct _mesa_HashTable *RenderBuffers;
   struct _mesa_HashTable *FrameBuffers;

   /* GL_ARB_sync */
   struct set *SyncObjects;

   /** GL_ARB_sampler_objects */
   struct _mesa_HashTable *SamplerObjects;

   /* GL_ARB_bindless_texture */
   struct hash_table_u64 *TextureHandles;
   struct hash_table_u64 *ImageHandles;
   mtx_t HandlesMutex; /**< For texture/image handles safety */

   /* GL_ARB_shading_language_include */
   struct shader_includes *ShaderIncludes;
   /* glCompileShaderInclude expects ShaderIncludes not to change while it is
    * in progress.
    */
   simple_mtx_t ShaderIncludeMutex;

   /**
    * Some context in this share group was affected by a GPU reset
    *
    * On the next call to \c glGetGraphicsResetStatus, contexts that have not
    * been affected by a GPU reset must also return
    * \c GL_INNOCENT_CONTEXT_RESET_ARB.
    *
    * Once this field becomes true, it is never reset to false.
    */
   bool ShareGroupReset;

   /** EXT_external_objects */
   struct _mesa_HashTable *MemoryObjects;

   /** EXT_semaphore */
   struct _mesa_HashTable *SemaphoreObjects;

   /**
    * Some context in this share group was affected by a disjoint
    * operation. This operation can be anything that has effects on
    * values of timer queries in such manner that they become invalid for
    * performance metrics. As example gpu reset, counter overflow or gpu
    * frequency changes.
    */
   bool DisjointOperation;

   /**
    * Whether at least one image has been imported or exported, excluding
    * the default framebuffer. If this is false, glFlush can be executed
    * asynchronously because there is no invisible dependency on external
    * users.
    */
   bool HasExternallySharedImages;

   /* Small display list storage */
   struct {
      union gl_dlist_node *ptr;
      struct util_idalloc free_idx;
      unsigned size;
   } small_dlist_store;
};



/**
 * Renderbuffers represent drawing surfaces such as color, depth and/or
 * stencil.  A framebuffer object has a set of renderbuffers.
 * Drivers will typically derive subclasses of this type.
 */
struct gl_renderbuffer
{
   GLuint ClassID;        /**< Useful for drivers */
   GLuint Name;
   GLchar *Label;         /**< GL_KHR_debug */
   GLint RefCount;
   GLuint Width, Height;
   GLuint Depth;
   GLboolean AttachedAnytime; /**< TRUE if it was attached to a framebuffer */
   GLubyte NumSamples;    /**< zero means not multisampled */
   GLubyte NumStorageSamples; /**< for AMD_framebuffer_multisample_advanced */
   GLenum16 InternalFormat; /**< The user-specified format */
   GLenum16 _BaseFormat;    /**< Either GL_RGB, GL_RGBA, GL_DEPTH_COMPONENT or
                               GL_STENCIL_INDEX. */
   mesa_format Format;      /**< The actual renderbuffer memory format */
   /**
    * Pointer to the texture image if this renderbuffer wraps a texture,
    * otherwise NULL.
    *
    * Note that the reference on the gl_texture_object containing this
    * TexImage is held by the gl_renderbuffer_attachment.
    */
   struct gl_texture_image *TexImage;

   /** Delete this renderbuffer */
   void (*Delete)(struct gl_context *ctx, struct gl_renderbuffer *rb);

   /** Allocate new storage for this renderbuffer */
   GLboolean (*AllocStorage)(struct gl_context *ctx,
                             struct gl_renderbuffer *rb,
                             GLenum internalFormat,
                             GLuint width, GLuint height);

   struct pipe_resource *texture;
   /* This points to either "surface_linear" or "surface_srgb".
    * It doesn't hold the pipe_surface reference. The other two do.
    */
   struct pipe_surface *surface;
   struct pipe_surface *surface_linear;
   struct pipe_surface *surface_srgb;
   GLboolean defined;        /**< defined contents? */

   struct pipe_transfer *transfer; /**< only used when mapping the resource */

   /**
    * Used only when hardware accumulation buffers are not supported.
    */
   boolean software;
   void *data;

   bool use_readpix_cache;

   /* Inputs from Driver.RenderTexture, don't use directly. */
   boolean is_rtt; /**< whether Driver.RenderTexture was called */
   unsigned rtt_face, rtt_slice;
   boolean rtt_layered; /**< whether glFramebufferTexture was called */
   unsigned rtt_nr_samples; /**< from FramebufferTexture2DMultisampleEXT */
};


/**
 * A renderbuffer attachment points to either a texture object (and specifies
 * a mipmap level, cube face or 3D texture slice) or points to a renderbuffer.
 */
struct gl_renderbuffer_attachment
{
   GLenum16 Type; /**< \c GL_NONE or \c GL_TEXTURE or \c GL_RENDERBUFFER_EXT */
   GLboolean Complete;

   /**
    * If \c Type is \c GL_RENDERBUFFER_EXT, this stores a pointer to the
    * application supplied renderbuffer object.
    */
   struct gl_renderbuffer *Renderbuffer;

   /**
    * If \c Type is \c GL_TEXTURE, this stores a pointer to the application
    * supplied texture object.
    */
   struct gl_texture_object *Texture;
   GLuint TextureLevel; /**< Attached mipmap level. */
   GLsizei NumSamples;  /**< from FramebufferTexture2DMultisampleEXT */
   GLuint CubeMapFace;  /**< 0 .. 5, for cube map textures. */
   GLuint Zoffset;      /**< Slice for 3D textures,  or layer for both 1D
                         * and 2D array textures */
   GLboolean Layered;
};


/**
 * A framebuffer is a collection of renderbuffers (color, depth, stencil, etc).
 * In C++ terms, think of this as a base class from which device drivers
 * will make derived classes.
 */
struct gl_framebuffer
{
   simple_mtx_t Mutex;  /**< for thread safety */
   /**
    * If zero, this is a window system framebuffer.  If non-zero, this
    * is a FBO framebuffer; note that for some devices (i.e. those with
    * a natural pixel coordinate system for FBOs that differs from the
    * OpenGL/Mesa coordinate system), this means that the viewport,
    * polygon face orientation, and polygon stipple will have to be inverted.
    */
   GLuint Name;
   GLint RefCount;

   GLchar *Label;       /**< GL_KHR_debug */

   GLboolean DeletePending;

   /**
    * The framebuffer's visual. Immutable if this is a window system buffer.
    * Computed from attachments if user-made FBO.
    */
   struct gl_config Visual;

   /**
    * Size of frame buffer in pixels. If there are no attachments, then both
    * of these are 0.
    */
   GLuint Width, Height;

   /**
    * In the case that the framebuffer has no attachment (i.e.
    * GL_ARB_framebuffer_no_attachments) then the geometry of
    * the framebuffer is specified by the default values.
    */
   struct {
     GLuint Width, Height, Layers, NumSamples;
     GLboolean FixedSampleLocations;
     /* Derived from NumSamples by the driver so that it can choose a valid
      * value for the hardware.
      */
     GLuint _NumSamples;
   } DefaultGeometry;

   /** \name  Drawing bounds (Intersection of buffer size and scissor box)
    * The drawing region is given by [_Xmin, _Xmax) x [_Ymin, _Ymax),
    * (inclusive for _Xmin and _Ymin while exclusive for _Xmax and _Ymax)
    */
   /*@{*/
   GLint _Xmin, _Xmax;
   GLint _Ymin, _Ymax;
   /*@}*/

   /** \name  Derived Z buffer stuff */
   /*@{*/
   GLuint _DepthMax;	/**< Max depth buffer value */
   GLfloat _DepthMaxF;	/**< Float max depth buffer value */
   GLfloat _MRD;	/**< minimum resolvable difference in Z values */
   /*@}*/

   /** One of the GL_FRAMEBUFFER_(IN)COMPLETE_* tokens */
   GLenum16 _Status;

   /** Whether one of Attachment has Type != GL_NONE
    * NOTE: the values for Width and Height are set to 0 in case of having
    * no attachments, a backend driver supporting the extension
    * GL_ARB_framebuffer_no_attachments must check for the flag _HasAttachments
    * and if GL_FALSE, must then use the values in DefaultGeometry to initialize
    * its viewport, scissor and so on (in particular _Xmin, _Xmax, _Ymin and
    * _Ymax do NOT take into account _HasAttachments being false). To get the
    * geometry of the framebuffer, the  helper functions
    *   _mesa_geometric_width(),
    *   _mesa_geometric_height(),
    *   _mesa_geometric_samples() and
    *   _mesa_geometric_layers()
    * are available that check _HasAttachments.
    */
   bool _HasAttachments;

   GLbitfield _IntegerBuffers;  /**< Which color buffers are integer valued */
   GLbitfield _BlendForceAlphaToOne;  /**< Which color buffers need blend factor adjustment */
   GLbitfield _IsRGB;  /**< Which color buffers have an RGB base format? */
   GLbitfield _FP32Buffers; /**< Which color buffers are FP32 */

   /* ARB_color_buffer_float */
   GLboolean _AllColorBuffersFixedPoint; /* no integer, no float */
   GLboolean _HasSNormOrFloatColorBuffer;

   /**
    * The maximum number of layers in the framebuffer, or 0 if the framebuffer
    * is not layered.  For cube maps and cube map arrays, each cube face
    * counts as a layer. As the case for Width, Height a backend driver
    * supporting GL_ARB_framebuffer_no_attachments must use DefaultGeometry
    * in the case that _HasAttachments is false
    */
   GLuint MaxNumLayers;

   /** Array of all renderbuffer attachments, indexed by BUFFER_* tokens. */
   struct gl_renderbuffer_attachment Attachment[BUFFER_COUNT];

   /* In unextended OpenGL these vars are part of the GL_COLOR_BUFFER
    * attribute group and GL_PIXEL attribute group, respectively.
    */
   GLenum16 ColorDrawBuffer[MAX_DRAW_BUFFERS];
   GLenum16 ColorReadBuffer;

   /* GL_ARB_sample_locations */
   GLfloat *SampleLocationTable; /**< If NULL, no table has been specified */
   GLboolean ProgrammableSampleLocations;
   GLboolean SampleLocationPixelGrid;

   /** Computed from ColorDraw/ReadBuffer above */
   GLuint _NumColorDrawBuffers;
   gl_buffer_index _ColorDrawBufferIndexes[MAX_DRAW_BUFFERS];
   gl_buffer_index _ColorReadBufferIndex;
   struct gl_renderbuffer *_ColorDrawBuffers[MAX_DRAW_BUFFERS];
   struct gl_renderbuffer *_ColorReadBuffer;

   /* GL_MESA_framebuffer_flip_y */
   bool FlipY;

   /** Delete this framebuffer */
   void (*Delete)(struct gl_framebuffer *fb);

   struct pipe_frontend_drawable *drawable;
   enum st_attachment_type statts[ST_ATTACHMENT_COUNT];
   unsigned num_statts;
   int32_t stamp;
   int32_t drawable_stamp;
   uint32_t drawable_ID;

   /* list of framebuffer objects */
   struct list_head head;
};

/**
 * A stack of matrices (projection, modelview, color, texture, etc).
 */
struct gl_matrix_stack
{
   GLmatrix *Top;      /**< points into Stack */
   GLmatrix *Stack;    /**< array [MaxDepth] of GLmatrix */
   unsigned StackSize; /**< Number of elements in Stack */
   GLuint Depth;       /**< 0 <= Depth < MaxDepth */
   GLuint MaxDepth;    /**< size of Stack[] array */
   GLuint DirtyFlag;   /**< _NEW_MODELVIEW or _NEW_PROJECTION, for example */
   bool ChangedSincePush;
};


/**
 * \name Bits for image transfer operations
 * \sa __struct gl_contextRec::ImageTransferState.
 */
/*@{*/
#define IMAGE_SCALE_BIAS_BIT                      0x1
#define IMAGE_SHIFT_OFFSET_BIT                    0x2
#define IMAGE_MAP_COLOR_BIT                       0x4
#define IMAGE_CLAMP_BIT                           0x800


/** Pixel Transfer ops */
#define IMAGE_BITS (IMAGE_SCALE_BIAS_BIT | \
                    IMAGE_SHIFT_OFFSET_BIT | \
                    IMAGE_MAP_COLOR_BIT)


/**
 * \name Bits to indicate what state has changed.
 */
/*@{*/
#define _NEW_MODELVIEW         (1u << 0)   /**< gl_context::ModelView */
#define _NEW_PROJECTION        (1u << 1)   /**< gl_context::Projection */
#define _NEW_TEXTURE_MATRIX    (1u << 2)   /**< gl_context::TextureMatrix */
#define _NEW_COLOR             (1u << 3)   /**< gl_context::Color */
#define _NEW_DEPTH             (1u << 4)   /**< gl_context::Depth */
#define _NEW_TNL_SPACES        (1u << 5)  /**< _mesa_update_tnl_spaces */
#define _NEW_FOG               (1u << 6)   /**< gl_context::Fog */
#define _NEW_HINT              (1u << 7)   /**< gl_context::Hint */
#define _NEW_LIGHT_CONSTANTS   (1u << 8)   /**< gl_context::Light */
#define _NEW_LINE              (1u << 9)   /**< gl_context::Line */
#define _NEW_PIXEL             (1u << 10)  /**< gl_context::Pixel */
#define _NEW_POINT             (1u << 11)  /**< gl_context::Point */
#define _NEW_POLYGON           (1u << 12)  /**< gl_context::Polygon */
#define _NEW_POLYGONSTIPPLE    (1u << 13)  /**< gl_context::PolygonStipple */
#define _NEW_SCISSOR           (1u << 14)  /**< gl_context::Scissor */
#define _NEW_STENCIL           (1u << 15)  /**< gl_context::Stencil */
#define _NEW_TEXTURE_OBJECT    (1u << 16)  /**< gl_context::Texture (bindings only) */
#define _NEW_TRANSFORM         (1u << 17)  /**< gl_context::Transform */
#define _NEW_VIEWPORT          (1u << 18)  /**< gl_context::Viewport */
#define _NEW_TEXTURE_STATE     (1u << 19)  /**< gl_context::Texture (states only) */
#define _NEW_LIGHT_STATE       (1u << 20)  /**< gl_context::Light */
#define _NEW_RENDERMODE        (1u << 21)  /**< gl_context::RenderMode, etc */
#define _NEW_BUFFERS           (1u << 22)  /**< gl_context::Visual, DrawBuffer, */
#define _NEW_CURRENT_ATTRIB    (1u << 23)  /**< gl_context::Current */
#define _NEW_MULTISAMPLE       (1u << 24)  /**< gl_context::Multisample */
#define _NEW_TRACK_MATRIX      (1u << 25)  /**< gl_context::VertexProgram */
#define _NEW_PROGRAM           (1u << 26)  /**< New program/shader state */
#define _NEW_PROGRAM_CONSTANTS (1u << 27)
#define _NEW_FF_VERT_PROGRAM   (1u << 28)
#define _NEW_FRAG_CLAMP        (1u << 29)
#define _NEW_MATERIAL          (1u << 30)  /**< gl_context::Light.Material */
#define _NEW_FF_FRAG_PROGRAM   (1u << 31)
#define _NEW_ALL ~0
/*@}*/


/* This has to be included here. */
#include "dd.h"


/** Opaque declaration of display list payload data type */
union gl_dlist_node;


/**
 * Per-display list information.
 */
struct gl_display_list
{
   GLuint Name;
   bool execute_glthread;
   bool small_list;
   GLchar *Label;     /**< GL_KHR_debug */
   /** The dlist commands are in a linked list of nodes */
   union {
      /* Big lists allocate their own storage */
      union gl_dlist_node *Head;
      /* Small lists use ctx->Shared->small_dlist_store */
      struct {
         unsigned start;
         unsigned count;
      };
   };
};


/**
 * State used during display list compilation and execution.
 */
struct gl_dlist_state
{
   struct gl_display_list *CurrentList; /**< List currently being compiled */
   union gl_dlist_node *CurrentBlock; /**< Pointer to current block of nodes */
   GLuint CurrentPos;		/**< Index into current block of nodes */
   GLuint CallDepth;		/**< Current recursion calling depth */
   GLuint LastInstSize;         /**< Size of the last node. */

   GLubyte ActiveAttribSize[VERT_ATTRIB_MAX];
   uint32_t CurrentAttrib[VERT_ATTRIB_MAX][8];

   GLubyte ActiveMaterialSize[MAT_ATTRIB_MAX];
   GLfloat CurrentMaterial[MAT_ATTRIB_MAX][4];

   struct {
      /* State known to have been set by the currently-compiling display
       * list.  Used to eliminate some redundant state changes.
       */
      GLenum16 ShadeModel;
      bool UseLoopback;
   } Current;
};

/**
 * Driver-specific state flags.
 *
 * These are or'd with gl_context::NewDriverState to notify a driver about
 * a state change. The driver sets the flags at context creation and
 * the meaning of the bits set is opaque to core Mesa.
 */
struct gl_driver_flags
{
   /**
    * gl_context::AtomicBufferBindings
    */
   uint64_t NewAtomicBuffer;

   /** gl_context::Color::Alpha* */
   uint64_t NewAlphaTest;

   /** gl_context::Multisample::Enabled */
   uint64_t NewMultisampleEnable;

   /** gl_context::Multisample::(Min)SampleShading */
   uint64_t NewSampleShading;

   /** gl_context::Transform::ClipPlanesEnabled */
   uint64_t NewClipPlaneEnable;

   /** gl_context::Color::ClampFragmentColor */
   uint64_t NewFragClamp;

   /** Shader constants (uniforms, program parameters, state constants) */
   uint64_t NewShaderConstants[MESA_SHADER_STAGES];

   /** For GL_CLAMP emulation */
   uint64_t NewSamplersWithClamp;
};

struct gl_buffer_binding
{
   struct gl_buffer_object *BufferObject;
   /** Start of uniform block data in the buffer */
   GLintptr Offset;
   /** Size of data allowed to be referenced from the buffer (in bytes) */
   GLsizeiptr Size;
   /**
    * glBindBufferBase() indicates that the Size should be ignored and only
    * limited by the current size of the BufferObject.
    */
   GLboolean AutomaticSize;
};

/**
 * ARB_shader_image_load_store image unit.
 */
struct gl_image_unit
{
   /**
    * Texture object bound to this unit.
    */
   struct gl_texture_object *TexObj;

   /**
    * Level of the texture object bound to this unit.
    */
   GLubyte Level;

   /**
    * \c GL_TRUE if the whole level is bound as an array of layers, \c
    * GL_FALSE if only some specific layer of the texture is bound.
    * \sa Layer
    */
   GLboolean Layered;

   /**
    * Layer of the texture object bound to this unit as specified by the
    * application.
    */
   GLushort Layer;

   /**
    * Layer of the texture object bound to this unit, or zero if
    * Layered == false.
    */
   GLushort _Layer;

   /**
    * Access allowed to this texture image.  Either \c GL_READ_ONLY,
    * \c GL_WRITE_ONLY or \c GL_READ_WRITE.
    */
   GLenum16 Access;

   /**
    * GL internal format that determines the interpretation of the
    * image memory when shader image operations are performed through
    * this unit.
    */
   GLenum16 Format;

   /**
    * Mesa format corresponding to \c Format.
    */
   mesa_format _ActualFormat:16;
};

/**
 * Shader subroutines storage
 */
struct gl_subroutine_index_binding
{
   GLuint NumIndex;
   GLuint *IndexPtr;
};

struct gl_texture_handle_object
{
   struct gl_texture_object *texObj;
   struct gl_sampler_object *sampObj;
   GLuint64 handle;
};

struct gl_image_handle_object
{
   struct gl_image_unit imgObj;
   GLuint64 handle;
};

struct gl_memory_object
{
   GLuint Name;            /**< hash table ID/name */
   GLboolean Immutable;    /**< denotes mutability state of parameters */
   GLboolean Dedicated;    /**< import memory from a dedicated allocation */

   struct pipe_memory_object *memory;

   /* TEXTURE_TILING_EXT param from gl_texture_object */
   GLuint TextureTiling;
};

struct gl_semaphore_object
{
   GLuint Name;            /**< hash table ID/name */
   struct pipe_fence_handle *fence;
   enum pipe_fd_type type;
   uint64_t timeline_value;
};

/**
 * One element of the client attrib stack.
 */
struct gl_client_attrib_node
{
   GLbitfield Mask;
   struct gl_array_attrib Array;
   struct gl_vertex_array_object VAO;
   struct gl_pixelstore_attrib Pack;
   struct gl_pixelstore_attrib Unpack;
};

/**
 * The VBO module implemented in src/vbo.
 */
struct vbo_context {
   struct gl_array_attributes current[VBO_ATTRIB_MAX];

   struct gl_vertex_array_object *VAO;

   struct vbo_exec_context exec;
   struct vbo_save_context save;
};

/**
 * glEnable node for the attribute stack. (glPushAttrib/glPopAttrib)
 */
struct gl_enable_attrib_node
{
   GLboolean AlphaTest;
   GLboolean AutoNormal;
   GLboolean Blend;
   GLbitfield ClipPlanes;
   GLboolean ColorMaterial;
   GLboolean CullFace;
   GLboolean DepthClampNear;
   GLboolean DepthClampFar;
   GLboolean DepthTest;
   GLboolean Dither;
   GLboolean Fog;
   GLboolean Light[MAX_LIGHTS];
   GLboolean Lighting;
   GLboolean LineSmooth;
   GLboolean LineStipple;
   GLboolean IndexLogicOp;
   GLboolean ColorLogicOp;

   GLboolean Map1Color4;
   GLboolean Map1Index;
   GLboolean Map1Normal;
   GLboolean Map1TextureCoord1;
   GLboolean Map1TextureCoord2;
   GLboolean Map1TextureCoord3;
   GLboolean Map1TextureCoord4;
   GLboolean Map1Vertex3;
   GLboolean Map1Vertex4;
   GLboolean Map2Color4;
   GLboolean Map2Index;
   GLboolean Map2Normal;
   GLboolean Map2TextureCoord1;
   GLboolean Map2TextureCoord2;
   GLboolean Map2TextureCoord3;
   GLboolean Map2TextureCoord4;
   GLboolean Map2Vertex3;
   GLboolean Map2Vertex4;

   GLboolean Normalize;
   GLboolean PixelTexture;
   GLboolean PointSmooth;
   GLboolean PolygonOffsetPoint;
   GLboolean PolygonOffsetLine;
   GLboolean PolygonOffsetFill;
   GLboolean PolygonSmooth;
   GLboolean PolygonStipple;
   GLboolean RescaleNormals;
   GLbitfield Scissor;
   GLboolean Stencil;
   GLboolean StencilTwoSide;          /* GL_EXT_stencil_two_side */
   GLboolean MultisampleEnabled;      /* GL_ARB_multisample */
   GLboolean SampleAlphaToCoverage;   /* GL_ARB_multisample */
   GLboolean SampleAlphaToOne;        /* GL_ARB_multisample */
   GLboolean SampleCoverage;          /* GL_ARB_multisample */
   GLboolean RasterPositionUnclipped; /* GL_IBM_rasterpos_clip */

   GLbitfield Texture[MAX_TEXTURE_UNITS];
   GLbitfield TexGen[MAX_TEXTURE_UNITS];

   /* GL_ARB_vertex_program */
   GLboolean VertexProgram;
   GLboolean VertexProgramPointSize;
   GLboolean VertexProgramTwoSide;

   /* GL_ARB_fragment_program */
   GLboolean FragmentProgram;

   /* GL_ARB_point_sprite */
   GLboolean PointSprite;
   GLboolean FragmentShaderATI;

   /* GL_ARB_framebuffer_sRGB / GL_EXT_framebuffer_sRGB */
   GLboolean sRGBEnabled;

   /* GL_NV_conservative_raster */
   GLboolean ConservativeRasterization;
};

/**
 * Texture node for the attribute stack. (glPushAttrib/glPopAttrib)
 */
struct gl_texture_attrib_node
{
   GLuint CurrentUnit;   /**< GL_ACTIVE_TEXTURE */
   GLuint NumTexSaved;
   struct gl_fixedfunc_texture_unit FixedFuncUnit[MAX_TEXTURE_COORD_UNITS];
   GLfloat LodBias[MAX_TEXTURE_UNITS];
   float LodBiasQuantized[MAX_TEXTURE_UNITS];

   /** Saved default texture object state. */
   struct gl_texture_object SavedDefaultObj[NUM_TEXTURE_TARGETS];

   /* For saving per texture object state (wrap modes, filters, etc),
    * SavedObj[][].Target is unused, so the value is invalid.
    */
   struct gl_texture_object SavedObj[MAX_COMBINED_TEXTURE_IMAGE_UNITS][NUM_TEXTURE_TARGETS];
};


/**
 * Node for the attribute stack. (glPushAttrib/glPopAttrib)
 */
struct gl_attrib_node
{
   GLbitfield Mask;
   GLbitfield OldPopAttribStateMask;
   struct gl_accum_attrib Accum;
   struct gl_colorbuffer_attrib Color;
   struct gl_current_attrib Current;
   struct gl_depthbuffer_attrib Depth;
   struct gl_enable_attrib_node Enable;
   struct gl_eval_attrib Eval;
   struct gl_fog_attrib Fog;
   struct gl_hint_attrib Hint;
   struct gl_light_attrib Light;
   struct gl_line_attrib Line;
   struct gl_list_attrib List;
   struct gl_pixel_attrib Pixel;
   struct gl_point_attrib Point;
   struct gl_polygon_attrib Polygon;
   GLuint PolygonStipple[32];
   struct gl_scissor_attrib Scissor;
   struct gl_stencil_attrib Stencil;
   struct gl_transform_attrib Transform;
   struct gl_multisample_attrib Multisample;
   struct gl_texture_attrib_node Texture;

   struct viewport_state
   {
      struct gl_viewport_attrib ViewportArray[MAX_VIEWPORTS];
      GLuint SubpixelPrecisionBias[2];
   } Viewport;
};

/**
 * Mesa rendering context.
 *
 * This is the central context data structure for Mesa.  Almost all
 * OpenGL state is contained in this structure.
 * Think of this as a base class from which device drivers will derive
 * sub classes.
 */
struct gl_context
{
   /** State possibly shared with other contexts in the address space */
   struct gl_shared_state *Shared;

   /** Whether Shared->BufferObjects has already been locked for this context. */
   bool BufferObjectsLocked;
   /** Whether Shared->TexMutex has already been locked for this context. */
   bool TexturesLocked;

   /** \name API function pointer tables */
   /*@{*/
   gl_api API;

   /**
    * The current dispatch table for non-displaylist-saving execution, either
    * BeginEnd or OutsideBeginEnd
    */
   struct _glapi_table *Exec;
   /**
    * The normal dispatch table for non-displaylist-saving, non-begin/end
    */
   struct _glapi_table *OutsideBeginEnd;
   /** The dispatch table used between glNewList() and glEndList() */
   struct _glapi_table *Save;
   /**
    * The dispatch table used between glBegin() and glEnd() (outside of a
    * display list).  Only valid functions between those two are set.
    */
   struct _glapi_table *BeginEnd;
   /**
    * Same as BeginEnd except vertex postion set functions. Used when
    * HW GL_SELECT mode instead of BeginEnd.
    */
   struct _glapi_table *HWSelectModeBeginEnd;
   /**
    * Dispatch table for when a graphics reset has happened.
    */
   struct _glapi_table *ContextLost;
   /**
    * Dispatch table used to marshal API calls from the client program to a
    * separate server thread.
    */
   struct _glapi_table *MarshalExec;
   /**
    * Dispatch table currently in use for fielding API calls from the client
    * program.  If API calls are being marshalled to another thread, this ==
    * MarshalExec.  Otherwise it == CurrentServerDispatch.
    */
   struct _glapi_table *CurrentClientDispatch;

   /**
    * Dispatch table currently in use for performing API calls.  == Save or
    * Exec.
    */
   struct _glapi_table *CurrentServerDispatch;

   /*@}*/

   struct glthread_state GLThread;

   struct gl_config Visual;
   struct gl_framebuffer *DrawBuffer;	/**< buffer for writing */
   struct gl_framebuffer *ReadBuffer;	/**< buffer for reading */
   struct gl_framebuffer *WinSysDrawBuffer;  /**< set with MakeCurrent */
   struct gl_framebuffer *WinSysReadBuffer;  /**< set with MakeCurrent */

   /**
    * Device driver function pointer table
    */
   struct dd_function_table Driver;

   /** Core/Driver constants */
   struct gl_constants Const;

   /**
    * Bitmask of valid primitive types supported by this context type,
    * GL version, and extensions, not taking current states into account.
    * Current states can further reduce the final bitmask at draw time.
    */
   GLbitfield SupportedPrimMask;

   /**
    * Bitmask of valid primitive types depending on current states (such as
    * shaders). This is 0 if the current states should result in
    * GL_INVALID_OPERATION in draw calls.
    */
   GLbitfield ValidPrimMask;

   GLenum16 DrawGLError; /**< GL error to return from draw calls */

   /**
    * Same as ValidPrimMask, but should be applied to glDrawElements*.
    */
   GLbitfield ValidPrimMaskIndexed;

   /**
    * Whether DrawPixels/CopyPixels/Bitmap are valid to render.
    */
   bool DrawPixValid;

   /** \name The various 4x4 matrix stacks */
   /*@{*/
   struct gl_matrix_stack ModelviewMatrixStack;
   struct gl_matrix_stack ProjectionMatrixStack;
   struct gl_matrix_stack TextureMatrixStack[MAX_TEXTURE_UNITS];
   struct gl_matrix_stack ProgramMatrixStack[MAX_PROGRAM_MATRICES];
   struct gl_matrix_stack *CurrentStack; /**< Points to one of the above stacks */
   /*@}*/

   /** Combined modelview and projection matrix */
   GLmatrix _ModelProjectMatrix;

   /** \name Display lists */
   struct gl_dlist_state ListState;

   GLboolean ExecuteFlag;	/**< Execute GL commands? */
   GLboolean CompileFlag;	/**< Compile GL commands into display list? */

   /** Extension information */
   struct gl_extensions Extensions;

   /** GL version integer, for example 31 for GL 3.1, or 20 for GLES 2.0. */
   GLuint Version;
   char *VersionString;

   /** \name State attribute stack (for glPush/PopAttrib) */
   /*@{*/
   GLuint AttribStackDepth;
   struct gl_attrib_node *AttribStack[MAX_ATTRIB_STACK_DEPTH];
   /*@}*/

   /** \name Renderer attribute groups
    *
    * We define a struct for each attribute group to make pushing and popping
    * attributes easy.  Also it's a good organization.
    */
   /*@{*/
   struct gl_accum_attrib	Accum;		/**< Accum buffer attributes */
   struct gl_colorbuffer_attrib	Color;		/**< Color buffer attributes */
   struct gl_current_attrib	Current;	/**< Current attributes */
   struct gl_depthbuffer_attrib	Depth;		/**< Depth buffer attributes */
   struct gl_eval_attrib	Eval;		/**< Eval attributes */
   struct gl_fog_attrib		Fog;		/**< Fog attributes */
   struct gl_hint_attrib	Hint;		/**< Hint attributes */
   struct gl_light_attrib	Light;		/**< Light attributes */
   struct gl_line_attrib	Line;		/**< Line attributes */
   struct gl_list_attrib	List;		/**< List attributes */
   struct gl_multisample_attrib Multisample;
   struct gl_pixel_attrib	Pixel;		/**< Pixel attributes */
   struct gl_point_attrib	Point;		/**< Point attributes */
   struct gl_polygon_attrib	Polygon;	/**< Polygon attributes */
   GLuint PolygonStipple[32];			/**< Polygon stipple */
   struct gl_scissor_attrib	Scissor;	/**< Scissor attributes */
   struct gl_stencil_attrib	Stencil;	/**< Stencil buffer attributes */
   struct gl_texture_attrib	Texture;	/**< Texture attributes */
   struct gl_transform_attrib	Transform;	/**< Transformation attributes */
   struct gl_viewport_attrib	ViewportArray[MAX_VIEWPORTS];	/**< Viewport attributes */
   GLuint SubpixelPrecisionBias[2];	/**< Viewport attributes */
   /*@}*/

   /** \name Client attribute stack */
   /*@{*/
   GLuint ClientAttribStackDepth;
   struct gl_client_attrib_node ClientAttribStack[MAX_CLIENT_ATTRIB_STACK_DEPTH];
   /*@}*/

   /** \name Client attribute groups */
   /*@{*/
   struct gl_array_attrib	Array;	/**< Vertex arrays */
   struct gl_pixelstore_attrib	Pack;	/**< Pixel packing */
   struct gl_pixelstore_attrib	Unpack;	/**< Pixel unpacking */
   struct gl_pixelstore_attrib	DefaultPacking;	/**< Default params */
   /*@}*/

   /** \name Other assorted state (not pushed/popped on attribute stack) */
   /*@{*/
   struct gl_pixelmaps          PixelMaps;

   struct gl_evaluators EvalMap;   /**< All evaluators */
   struct gl_feedback   Feedback;  /**< Feedback */
   struct gl_selection  Select;    /**< Selection */

   struct gl_program_state Program;  /**< general program state */
   struct gl_vertex_program_state VertexProgram;
   struct gl_fragment_program_state FragmentProgram;
   struct gl_geometry_program_state GeometryProgram;
   struct gl_compute_program_state ComputeProgram;
   struct gl_tess_ctrl_program_state TessCtrlProgram;
   struct gl_tess_eval_program_state TessEvalProgram;
   struct gl_ati_fragment_shader_state ATIFragmentShader;

   struct gl_pipeline_shader_state Pipeline; /**< GLSL pipeline shader object state */
   struct gl_pipeline_object Shader; /**< GLSL shader object state */

   /**
    * Current active shader pipeline state
    *
    * Almost all internal users want ::_Shader instead of ::Shader.  The
    * exceptions are bits of legacy GLSL API that do not know about separate
    * shader objects.
    *
    * If a program is active via \c glUseProgram, this will point to
    * \c ::Shader.
    *
    * If a program pipeline is active via \c glBindProgramPipeline, this will
    * point to \c ::Pipeline.Current.
    *
    * If neither a program nor a program pipeline is active, this will point to
    * \c ::Pipeline.Default.  This ensures that \c ::_Shader will never be
    * \c NULL.
    */
   struct gl_pipeline_object *_Shader;

   /**
    * NIR containing the functions that implement software fp64 support.
    */
   struct nir_shader *SoftFP64;

   struct gl_query_state Query;  /**< occlusion, timer queries */

   struct gl_transform_feedback_state TransformFeedback;

   struct gl_perf_monitor_state PerfMonitor;
   struct gl_perf_query_state PerfQuery;

   struct gl_buffer_object *DrawIndirectBuffer; /** < GL_ARB_draw_indirect */
   struct gl_buffer_object *ParameterBuffer; /** < GL_ARB_indirect_parameters */
   struct gl_buffer_object *DispatchIndirectBuffer; /** < GL_ARB_compute_shader */

   struct gl_buffer_object *CopyReadBuffer; /**< GL_ARB_copy_buffer */
   struct gl_buffer_object *CopyWriteBuffer; /**< GL_ARB_copy_buffer */

   struct gl_buffer_object *QueryBuffer; /**< GL_ARB_query_buffer_object */

   /**
    * Current GL_ARB_uniform_buffer_object binding referenced by
    * GL_UNIFORM_BUFFER target for glBufferData, glMapBuffer, etc.
    */
   struct gl_buffer_object *UniformBuffer;

   /**
    * Current GL_ARB_shader_storage_buffer_object binding referenced by
    * GL_SHADER_STORAGE_BUFFER target for glBufferData, glMapBuffer, etc.
    */
   struct gl_buffer_object *ShaderStorageBuffer;

   /**
    * Array of uniform buffers for GL_ARB_uniform_buffer_object and GL 3.1.
    * This is set up using glBindBufferRange() or glBindBufferBase().  They are
    * associated with uniform blocks by glUniformBlockBinding()'s state in the
    * shader program.
    */
   struct gl_buffer_binding
      UniformBufferBindings[MAX_COMBINED_UNIFORM_BUFFERS];

   /**
    * Array of shader storage buffers for ARB_shader_storage_buffer_object
    * and GL 4.3. This is set up using glBindBufferRange() or
    * glBindBufferBase().  They are associated with shader storage blocks by
    * glShaderStorageBlockBinding()'s state in the shader program.
    */
   struct gl_buffer_binding
      ShaderStorageBufferBindings[MAX_COMBINED_SHADER_STORAGE_BUFFERS];

   /**
    * Object currently associated with the GL_ATOMIC_COUNTER_BUFFER
    * target.
    */
   struct gl_buffer_object *AtomicBuffer;

   /**
    * Object currently associated w/ the GL_EXTERNAL_VIRTUAL_MEMORY_BUFFER_AMD
    * target.
    */
   struct gl_buffer_object *ExternalVirtualMemoryBuffer;

   /**
    * Array of atomic counter buffer binding points.
    */
   struct gl_buffer_binding
      AtomicBufferBindings[MAX_COMBINED_ATOMIC_BUFFERS];

   /**
    * Array of image units for ARB_shader_image_load_store.
    */
   struct gl_image_unit ImageUnits[MAX_IMAGE_UNITS];

   struct gl_subroutine_index_binding SubroutineIndex[MESA_SHADER_STAGES];
   /*@}*/

   struct gl_meta_state *Meta;  /**< for "meta" operations */

   /* GL_EXT_framebuffer_object */
   struct gl_renderbuffer *CurrentRenderbuffer;

   GLenum16 ErrorValue;      /**< Last error code */

   /**
    * Recognize and silence repeated error debug messages in buggy apps.
    */
   const char *ErrorDebugFmtString;
   GLuint ErrorDebugCount;

   /* GL_ARB_debug_output/GL_KHR_debug */
   simple_mtx_t DebugMutex;
   struct gl_debug_state *Debug;

   GLenum16 RenderMode;      /**< either GL_RENDER, GL_SELECT, GL_FEEDBACK */
   GLbitfield NewState;      /**< bitwise-or of _NEW_* flags */
   GLbitfield PopAttribState; /**< Updated state since glPushAttrib */
   uint64_t NewDriverState;  /**< bitwise-or of flags from DriverFlags */

   struct gl_driver_flags DriverFlags;

   GLboolean ViewportInitialized;  /**< has viewport size been initialized? */
   GLboolean _AllowDrawOutOfOrder;

   /** \name Derived state */
   GLbitfield _ImageTransferState;/**< bitwise-or of IMAGE_*_BIT flags */
   GLfloat _EyeZDir[3];
   GLfloat _ModelViewInvScale; /* may be for model- or eyespace lighting */
   GLfloat _ModelViewInvScaleEyespace; /* always factor defined in spec */
   GLboolean _NeedEyeCoords;

   GLuint TextureStateTimestamp; /**< detect changes to shared state */

   GLboolean PointSizeIsSet; /**< the glPointSize value in the shader is set */

   /** \name For debugging/development only */
   /*@{*/
   GLboolean FirstTimeCurrent;
   /*@}*/

   /**
    * False if this context was created without a config. This is needed
    * because the initial state of glDrawBuffers depends on this
    */
   GLboolean HasConfig;

   GLboolean TextureFormatSupported[MESA_FORMAT_COUNT];

   GLboolean RasterDiscard;  /**< GL_RASTERIZER_DISCARD */
   GLboolean IntelConservativeRasterization; /**< GL_CONSERVATIVE_RASTERIZATION_INTEL */
   GLboolean ConservativeRasterization; /**< GL_CONSERVATIVE_RASTERIZATION_NV */
   GLfloat ConservativeRasterDilate;
   GLenum16 ConservativeRasterMode;

   GLboolean IntelBlackholeRender; /**< GL_INTEL_blackhole_render */

   /** Does glVertexAttrib(0) alias glVertex()? */
   bool _AttribZeroAliasesVertex;

   /**
    * When set, TileRasterOrderIncreasingX/Y control the order that a tiled
    * renderer's tiles should be excecuted, to meet the requirements of
    * GL_MESA_tile_raster_order.
    */
   GLboolean TileRasterOrderFixed;
   GLboolean TileRasterOrderIncreasingX;
   GLboolean TileRasterOrderIncreasingY;

   /**
    * \name Hooks for module contexts.
    *
    * These will eventually live in the driver or elsewhere.
    */
   /*@{*/
   struct vbo_context vbo_context;
   struct st_context *st;
   struct pipe_screen *screen;
   struct pipe_context *pipe;
   struct st_config_options *st_opts;
   struct cso_context *cso_context;
   bool has_invalidate_buffer;
   bool has_string_marker;
   /* On old libGL's for linux we need to invalidate the drawables
    * on glViewpport calls, this is set via a option.
    */
   bool invalidate_on_gl_viewport;

   /*@}*/

   /**
    * \name NV_vdpau_interop
    */
   /*@{*/
   const void *vdpDevice;
   const void *vdpGetProcAddress;
   struct set *vdpSurfaces;
   /*@}*/

   /**
    * Has this context observed a GPU reset in any context in the share group?
    *
    * Once this field becomes true, it is never reset to false.
    */
   GLboolean ShareGroupReset;

   /**
    * \name OES_primitive_bounding_box
    *
    * Stores the arguments to glPrimitiveBoundingBox
    */
   GLfloat PrimitiveBoundingBox[8];

   struct disk_cache *Cache;

   /**
    * \name GL_ARB_bindless_texture
    */
   /*@{*/
   struct hash_table_u64 *ResidentTextureHandles;
   struct hash_table_u64 *ResidentImageHandles;
   /*@}*/

   bool shader_builtin_ref;

   struct pipe_draw_start_count_bias *tmp_draws;
   unsigned num_tmp_draws;
};

#ifndef NDEBUG
extern int MESA_VERBOSE;
extern int MESA_DEBUG_FLAGS;
#else
# define MESA_VERBOSE 0
# define MESA_DEBUG_FLAGS 0
#endif


/** The MESA_VERBOSE var is a bitmask of these flags */
enum _verbose
{
   VERBOSE_VARRAY		= 0x0001,
   VERBOSE_TEXTURE		= 0x0002,
   VERBOSE_MATERIAL		= 0x0004,
   VERBOSE_PIPELINE		= 0x0008,
   VERBOSE_DRIVER		= 0x0010,
   VERBOSE_STATE		= 0x0020,
   VERBOSE_API			= 0x0040,
   VERBOSE_DISPLAY_LIST		= 0x0100,
   VERBOSE_LIGHTING		= 0x0200,
   VERBOSE_PRIMS		= 0x0400,
   VERBOSE_VERTS		= 0x0800,
   VERBOSE_DISASSEM		= 0x1000,
   VERBOSE_SWAPBUFFERS          = 0x4000
};


/** The MESA_DEBUG_FLAGS var is a bitmask of these flags */
enum _debug
{
   DEBUG_SILENT                 = (1 << 0),
   DEBUG_ALWAYS_FLUSH		= (1 << 1),
   DEBUG_INCOMPLETE_TEXTURE     = (1 << 2),
   DEBUG_INCOMPLETE_FBO         = (1 << 3),
   DEBUG_CONTEXT                = (1 << 4)
};

#ifdef __cplusplus
}
#endif

#endif /* MTYPES_H */
