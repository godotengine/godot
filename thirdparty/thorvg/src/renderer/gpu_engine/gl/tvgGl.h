/*
 * Copyright (c) 2025 - 2026 ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

 #ifndef _TVG_GL_H_
 #define _TVG_GL_H_

#ifdef __EMSCRIPTEN__
    #include <GLES3/gl3.h>
    #include <emscripten/html5_webgl.h>
    #define GL_CHECK(stmt) stmt
#else //__EMSCRIPTEN__
    #if defined (THORVG_GL_TARGET_GLES)
        #define TVG_REQUIRE_GL_MAJOR_VER 3
        #define TVG_REQUIRE_GL_MINOR_VER 0
    #else
        #define TVG_REQUIRE_GL_MAJOR_VER 3
        #define TVG_REQUIRE_GL_MINOR_VER 3
    #endif

    #include "tvgCommon.h"

    #ifdef _DEBUG
        #define GL_CHECK(stmt) stmt; assert(glGetError() == GL_NO_ERROR);
    #else
        #define GL_CHECK(stmt) stmt
    #endif

    #ifdef _WIN64
        typedef signed long long int khronos_intptr_t;
        typedef unsigned long long int khronos_uintptr_t;
        typedef signed long long int khronos_ssize_t;
        typedef unsigned long long int khronos_usize_t;
    #else
        typedef signed long int khronos_intptr_t;
        typedef unsigned long int khronos_uintptr_t;
        typedef signed long int khronos_ssize_t;
        typedef unsigned long int khronos_usize_t;
    #endif

    typedef signed char khronos_int8_t;
    typedef unsigned char khronos_uint8_t;
    typedef signed short int khronos_int16_t;
    typedef unsigned short int khronos_uint16_t;
    typedef float khronos_float_t;
    typedef khronos_intptr_t GLintptr;
    typedef khronos_ssize_t GLsizeiptr;

    #ifndef GL_VERSION_1_0
        #define GL_VERSION_1_0 1
        typedef void GLvoid;
        typedef unsigned int GLenum;
        typedef khronos_float_t GLfloat;
        typedef int GLint;
        typedef int GLsizei;
        typedef unsigned int GLbitfield;
        typedef double GLdouble;
        typedef unsigned int GLuint;
        typedef unsigned char GLboolean;
        typedef khronos_uint8_t GLubyte;
        #define GL_DEPTH_BUFFER_BIT               0x00000100
        #define GL_STENCIL_BUFFER_BIT             0x00000400
        #define GL_COLOR_BUFFER_BIT               0x00004000
        #define GL_FALSE                          0
        #define GL_TRUE                           1
        #define GL_POINTS                         0x0000
        #define GL_LINES                          0x0001
        #define GL_LINE_LOOP                      0x0002
        #define GL_LINE_STRIP                     0x0003
        #define GL_TRIANGLES                      0x0004
        #define GL_TRIANGLE_STRIP                 0x0005
        #define GL_TRIANGLE_FAN                   0x0006
        #define GL_QUADS                          0x0007
        #define GL_NEVER                          0x0200
        #define GL_LESS                           0x0201
        #define GL_EQUAL                          0x0202
        #define GL_LEQUAL                         0x0203
        #define GL_GREATER                        0x0204
        #define GL_NOTEQUAL                       0x0205
        #define GL_GEQUAL                         0x0206
        #define GL_ALWAYS                         0x0207
        #define GL_ZERO                           0
        #define GL_ONE                            1
        #define GL_SRC_COLOR                      0x0300
        #define GL_ONE_MINUS_SRC_COLOR            0x0301
        #define GL_SRC_ALPHA                      0x0302
        #define GL_ONE_MINUS_SRC_ALPHA            0x0303
        #define GL_DST_ALPHA                      0x0304
        #define GL_ONE_MINUS_DST_ALPHA            0x0305
        #define GL_DST_COLOR                      0x0306
        #define GL_ONE_MINUS_DST_COLOR            0x0307
        #define GL_SRC_ALPHA_SATURATE             0x0308
        #define GL_NONE                           0
        #define GL_FRONT_LEFT                     0x0400
        #define GL_FRONT_RIGHT                    0x0401
        #define GL_BACK_LEFT                      0x0402
        #define GL_BACK_RIGHT                     0x0403
        #define GL_FRONT                          0x0404
        #define GL_BACK                           0x0405
        #define GL_LEFT                           0x0406
        #define GL_RIGHT                          0x0407
        #define GL_FRONT_AND_BACK                 0x0408
        #define GL_NO_ERROR                       0
        #define GL_INVALID_ENUM                   0x0500
        #define GL_INVALID_VALUE                  0x0501
        #define GL_INVALID_OPERATION              0x0502
        #define GL_OUT_OF_MEMORY                  0x0505
        #define GL_CW                             0x0900
        #define GL_CCW                            0x0901
        #define GL_POINT_SIZE                     0x0B11
        #define GL_POINT_SIZE_RANGE               0x0B12
        #define GL_POINT_SIZE_GRANULARITY         0x0B13
        #define GL_LINE_SMOOTH                    0x0B20
        #define GL_LINE_WIDTH                     0x0B21
        #define GL_LINE_WIDTH_RANGE               0x0B22
        #define GL_LINE_WIDTH_GRANULARITY         0x0B23
        #define GL_POLYGON_MODE                   0x0B40
        #define GL_POLYGON_SMOOTH                 0x0B41
        #define GL_CULL_FACE                      0x0B44
        #define GL_CULL_FACE_MODE                 0x0B45
        #define GL_FRONT_FACE                     0x0B46
        #define GL_DEPTH_RANGE                    0x0B70
        #define GL_DEPTH_TEST                     0x0B71
        #define GL_DEPTH_WRITEMASK                0x0B72
        #define GL_DEPTH_CLEAR_VALUE              0x0B73
        #define GL_DEPTH_FUNC                     0x0B74
        #define GL_STENCIL_TEST                   0x0B90
        #define GL_STENCIL_CLEAR_VALUE            0x0B91
        #define GL_STENCIL_FUNC                   0x0B92
        #define GL_STENCIL_VALUE_MASK             0x0B93
        #define GL_STENCIL_FAIL                   0x0B94
        #define GL_STENCIL_PASS_DEPTH_FAIL        0x0B95
        #define GL_STENCIL_PASS_DEPTH_PASS        0x0B96
        #define GL_STENCIL_REF                    0x0B97
        #define GL_STENCIL_WRITEMASK              0x0B98
        #define GL_VIEWPORT                       0x0BA2
        #define GL_DITHER                         0x0BD0
        #define GL_BLEND_DST                      0x0BE0
        #define GL_BLEND_SRC                      0x0BE1
        #define GL_BLEND                          0x0BE2
        #define GL_LOGIC_OP_MODE                  0x0BF0
        #define GL_DRAW_BUFFER                    0x0C01
        #define GL_READ_BUFFER                    0x0C02
        #define GL_SCISSOR_BOX                    0x0C10
        #define GL_SCISSOR_TEST                   0x0C11
        #define GL_COLOR_CLEAR_VALUE              0x0C22
        #define GL_COLOR_WRITEMASK                0x0C23
        #define GL_DOUBLEBUFFER                   0x0C32
        #define GL_STEREO                         0x0C33
        #define GL_LINE_SMOOTH_HINT               0x0C52
        #define GL_POLYGON_SMOOTH_HINT            0x0C53
        #define GL_UNPACK_SWAP_BYTES              0x0CF0
        #define GL_UNPACK_LSB_FIRST               0x0CF1
        #define GL_UNPACK_ROW_LENGTH              0x0CF2
        #define GL_UNPACK_SKIP_ROWS               0x0CF3
        #define GL_UNPACK_SKIP_PIXELS             0x0CF4
        #define GL_UNPACK_ALIGNMENT               0x0CF5
        #define GL_PACK_SWAP_BYTES                0x0D00
        #define GL_PACK_LSB_FIRST                 0x0D01
        #define GL_PACK_ROW_LENGTH                0x0D02
        #define GL_PACK_SKIP_ROWS                 0x0D03
        #define GL_PACK_SKIP_PIXELS               0x0D04
        #define GL_PACK_ALIGNMENT                 0x0D05
        #define GL_MAX_TEXTURE_SIZE               0x0D33
        #define GL_MAX_VIEWPORT_DIMS              0x0D3A
        #define GL_SUBPIXEL_BITS                  0x0D50
        #define GL_TEXTURE_1D                     0x0DE0
        #define GL_TEXTURE_2D                     0x0DE1
        #define GL_TEXTURE_WIDTH                  0x1000
        #define GL_TEXTURE_HEIGHT                 0x1001
        #define GL_TEXTURE_BORDER_COLOR           0x1004
        #define GL_DONT_CARE                      0x1100
        #define GL_FASTEST                        0x1101
        #define GL_NICEST                         0x1102
        #define GL_BYTE                           0x1400
        #define GL_UNSIGNED_BYTE                  0x1401
        #define GL_SHORT                          0x1402
        #define GL_UNSIGNED_SHORT                 0x1403
        #define GL_INT                            0x1404
        #define GL_UNSIGNED_INT                   0x1405
        #define GL_FLOAT                          0x1406
        #define GL_STACK_OVERFLOW                 0x0503
        #define GL_STACK_UNDERFLOW                0x0504
        #define GL_CLEAR                          0x1500
        #define GL_AND                            0x1501
        #define GL_AND_REVERSE                    0x1502
        #define GL_COPY                           0x1503
        #define GL_AND_INVERTED                   0x1504
        #define GL_NOOP                           0x1505
        #define GL_XOR                            0x1506
        #define GL_OR                             0x1507
        #define GL_NOR                            0x1508
        #define GL_EQUIV                          0x1509
        #define GL_INVERT                         0x150A
        #define GL_OR_REVERSE                     0x150B
        #define GL_COPY_INVERTED                  0x150C
        #define GL_OR_INVERTED                    0x150D
        #define GL_NAND                           0x150E
        #define GL_SET                            0x150F
        #define GL_TEXTURE                        0x1702
        #define GL_COLOR                          0x1800
        #define GL_DEPTH                          0x1801
        #define GL_STENCIL                        0x1802
        #define GL_STENCIL_INDEX                  0x1901
        #define GL_DEPTH_COMPONENT                0x1902
        #define GL_RED                            0x1903
        #define GL_GREEN                          0x1904
        #define GL_BLUE                           0x1905
        #define GL_ALPHA                          0x1906
        #define GL_RGB                            0x1907
        #define GL_RGBA                           0x1908
        #define GL_POINT                          0x1B00
        #define GL_LINE                           0x1B01
        #define GL_FILL                           0x1B02
        #define GL_KEEP                           0x1E00
        #define GL_REPLACE                        0x1E01
        #define GL_INCR                           0x1E02
        #define GL_DECR                           0x1E03
        #define GL_VENDOR                         0x1F00
        #define GL_RENDERER                       0x1F01
        #define GL_VERSION                        0x1F02
        #define GL_EXTENSIONS                     0x1F03
        #define GL_NEAREST                        0x2600
        #define GL_LINEAR                         0x2601
        #define GL_NEAREST_MIPMAP_NEAREST         0x2700
        #define GL_LINEAR_MIPMAP_NEAREST          0x2701
        #define GL_NEAREST_MIPMAP_LINEAR          0x2702
        #define GL_LINEAR_MIPMAP_LINEAR           0x2703
        #define GL_TEXTURE_MAG_FILTER             0x2800
        #define GL_TEXTURE_MIN_FILTER             0x2801
        #define GL_TEXTURE_WRAP_S                 0x2802
        #define GL_TEXTURE_WRAP_T                 0x2803
        #define GL_REPEAT                         0x2901
        typedef void (*PFNGLCULLFACEPROC)(GLenum mode);
        typedef void (*PFNGLFRONTFACEPROC)(GLenum mode);
        typedef void (*PFNGLSCISSORPROC)(GLint x, GLint y, GLsizei width, GLsizei height);
        typedef void (*PFNGLTEXPARAMETERIPROC)(GLenum target, GLenum pname, GLint param);
        typedef void (*PFNGLTEXIMAGE2DPROC)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
        typedef void (*PFNGLDRAWBUFFERPROC)(GLenum buf);
        typedef void (*PFNGLCLEARPROC)(GLbitfield mask);
        typedef void (*PFNGLCLEARCOLORPROC)(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
        typedef void (*PFNGLCLEARSTENCILPROC)(GLint s);
        typedef void (*PFNGLCLEARDEPTHPROC)(GLdouble depth);
        typedef void (*PFNGLCLEARDEPTHFPROC)(GLdouble depth); // GLES
        typedef void (*PFNGLCOLORMASKPROC)(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
        typedef void (*PFNGLDEPTHMASKPROC)(GLboolean flag);
        typedef void (*PFNGLDISABLEPROC)(GLenum cap);
        typedef void (*PFNGLENABLEPROC)(GLenum cap);
        typedef void (*PFNGLBLENDFUNCPROC)(GLenum sfactor, GLenum dfactor);
        typedef void (*PFNGLSTENCILFUNCPROC)(GLenum func, GLint ref, GLuint mask);
        typedef void (*PFNGLSTENCILOPPROC)(GLenum fail, GLenum zfail, GLenum zpass);
        typedef void (*PFNGLDEPTHFUNCPROC)(GLenum func);
        typedef GLenum (*PFNGLGETERRORPROC)(void);
        typedef void (*PFNGLGETINTEGERVPROC)(GLenum pname, GLint *data);
        typedef const GLubyte *(*PFNGLGETSTRINGPROC) (GLenum name);
        typedef void (*PFNGLVIEWPORTPROC) (GLint x, GLint y, GLsizei width, GLsizei height);
        //typedef void (*PFNGLHINTPROC)(GLenum target, GLenum mode);
        //typedef void (*PFNGLLINEWIDTHPROC)(GLfloat width);
        //typedef void (*PFNGLPOINTSIZEPROC)(GLfloat size);
        //typedef void (*PFNGLPOLYGONMODEPROC)(GLenum face, GLenum mode);
        //typedef void (*PFNGLTEXPARAMETERFPROC)(GLenum target, GLenum pname, GLfloat param);
        //typedef void (*PFNGLTEXPARAMETERFVPROC)(GLenum target, GLenum pname, const GLfloat *params);
        //typedef void (*PFNGLTEXPARAMETERIVPROC)(GLenum target, GLenum pname, const GLint *params);
        //typedef void (*PFNGLTEXIMAGE1DPROC)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels);
        //typedef void (*PFNGLSTENCILMASKPROC) (GLuint mask);
        //typedef void (*PFNGLFINISHPROC)(void);
        //typedef void (*PFNGLFLUSHPROC)(void);
        //typedef void (*PFNGLLOGICOPPROC)(GLenum opcode);
        //typedef void (*PFNGLPIXELSTOREFPROC)(GLenum pname, GLfloat param);
        //typedef void (*PFNGLPIXELSTOREIPROC)(GLenum pname, GLint param);
        //typedef void (*PFNGLREADBUFFERPROC)(GLenum src);
        //typedef void (*PFNGLREADPIXELSPROC)(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels);
        //typedef void (*PFNGLGETBOOLEANVPROC)(GLenum pname, GLboolean *data);
        //typedef void (*PFNGLGETDOUBLEVPROC)(GLenum pname, GLdouble *data);
        //typedef void (*PFNGLGETFLOATVPROC)(GLenum pname, GLfloat *data);
        //typedef void (*PFNGLGETTEXIMAGEPROC)(GLenum target, GLint level, GLenum format, GLenum type, void *pixels);
        //typedef void (*PFNGLGETTEXPARAMETERFVPROC)(GLenum target, GLenum pname, GLfloat *params);
        //typedef void (*PFNGLGETTEXPARAMETERIVPROC)(GLenum target, GLenum pname, GLint *params);
        //typedef void (*PFNGLGETTEXLEVELPARAMETERFVPROC)(GLenum target, GLint level, GLenum pname, GLfloat *params);
        //typedef void (*PFNGLGETTEXLEVELPARAMETERIVPROC)(GLenum target, GLint level, GLenum pname, GLint *params);
        //typedef GLboolean (*PFNGLISENABLEDPROC)(GLenum cap);
        //typedef void (*PFNGLDEPTHRANGEPROC)(GLdouble n, GLdouble f);
    #endif /* GL_VERSION_1_0 */

    #ifndef GL_VERSION_1_1
        #define GL_VERSION_1_1 1
        typedef khronos_float_t GLclampf;
        typedef double GLclampd;
        #define GL_COLOR_LOGIC_OP                 0x0BF2
        #define GL_POLYGON_OFFSET_UNITS           0x2A00
        #define GL_POLYGON_OFFSET_POINT           0x2A01
        #define GL_POLYGON_OFFSET_LINE            0x2A02
        #define GL_POLYGON_OFFSET_FILL            0x8037
        #define GL_POLYGON_OFFSET_FACTOR          0x8038
        #define GL_TEXTURE_BINDING_1D             0x8068
        #define GL_TEXTURE_BINDING_2D             0x8069
        #define GL_TEXTURE_INTERNAL_FORMAT        0x1003
        #define GL_TEXTURE_RED_SIZE               0x805C
        #define GL_TEXTURE_GREEN_SIZE             0x805D
        #define GL_TEXTURE_BLUE_SIZE              0x805E
        #define GL_TEXTURE_ALPHA_SIZE             0x805F
        #define GL_DOUBLE                         0x140A
        #define GL_PROXY_TEXTURE_1D               0x8063
        #define GL_PROXY_TEXTURE_2D               0x8064
        #define GL_R3_G3_B2                       0x2A10
        #define GL_RGB4                           0x804F
        #define GL_RGB5                           0x8050
        #define GL_RGB8                           0x8051
        #define GL_RGB10                          0x8052
        #define GL_RGB12                          0x8053
        #define GL_RGB16                          0x8054
        #define GL_RGBA2                          0x8055
        #define GL_RGBA4                          0x8056
        #define GL_RGB5_A1                        0x8057
        #define GL_RGBA8                          0x8058
        #define GL_RGB10_A2                       0x8059
        #define GL_RGBA12                         0x805A
        #define GL_RGBA16                         0x805B
        #define GL_VERTEX_ARRAY                   0x8074
        typedef void (*PFNGLDRAWELEMENTSPROC)(GLenum mode, GLsizei count, GLenum type, const void *indices);
        typedef void (*PFNGLBINDTEXTUREPROC)(GLenum target, GLuint texture);
        typedef void (*PFNGLDELETETEXTURESPROC)(GLsizei n, const GLuint *textures);
        typedef void (*PFNGLGENTEXTURESPROC)(GLsizei n, GLuint *textures);
        //typedef void (*PFNGLDRAWARRAYSPROC)(GLenum mode, GLint first, GLsizei count);
        //typedef void (*PFNGLGETPOINTERVPROC)(GLenum pname, void **params);
        //typedef void (*PFNGLPOLYGONOFFSETPROC)(GLfloat factor, GLfloat units);
        //typedef void (*PFNGLCOPYTEXIMAGE1DPROC)(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border);
        //typedef void (*PFNGLCOPYTEXIMAGE2DPROC)(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border);
        //typedef void (*PFNGLCOPYTEXSUBIMAGE1DPROC)(GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
        //typedef void (*PFNGLCOPYTEXSUBIMAGE2DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
        //typedef void (*PFNGLTEXSUBIMAGE1DPROC)(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels);
        //typedef void (*PFNGLTEXSUBIMAGE2DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
        //typedef GLboolean (*PFNGLISTEXTUREPROC)(GLuint texture);
    #endif /* GL_VERSION_1_1 */

    #ifndef GL_VERSION_1_2
        #define GL_VERSION_1_2 1
        #define GL_UNSIGNED_BYTE_3_3_2            0x8032
        #define GL_UNSIGNED_SHORT_4_4_4_4         0x8033
        #define GL_UNSIGNED_SHORT_5_5_5_1         0x8034
        #define GL_UNSIGNED_INT_8_8_8_8           0x8035
        #define GL_UNSIGNED_INT_10_10_10_2        0x8036
        #define GL_TEXTURE_BINDING_3D             0x806A
        #define GL_PACK_SKIP_IMAGES               0x806B
        #define GL_PACK_IMAGE_HEIGHT              0x806C
        #define GL_UNPACK_SKIP_IMAGES             0x806D
        #define GL_UNPACK_IMAGE_HEIGHT            0x806E
        #define GL_TEXTURE_3D                     0x806F
        #define GL_PROXY_TEXTURE_3D               0x8070
        #define GL_TEXTURE_DEPTH                  0x8071
        #define GL_TEXTURE_WRAP_R                 0x8072
        #define GL_MAX_3D_TEXTURE_SIZE            0x8073
        #define GL_UNSIGNED_BYTE_2_3_3_REV        0x8362
        #define GL_UNSIGNED_SHORT_5_6_5           0x8363
        #define GL_UNSIGNED_SHORT_5_6_5_REV       0x8364
        #define GL_UNSIGNED_SHORT_4_4_4_4_REV     0x8365
        #define GL_UNSIGNED_SHORT_1_5_5_5_REV     0x8366
        #define GL_UNSIGNED_INT_8_8_8_8_REV       0x8367
        #define GL_UNSIGNED_INT_2_10_10_10_REV    0x8368
        #define GL_BGR                            0x80E0
        #define GL_BGRA                           0x80E1
        #define GL_MAX_ELEMENTS_VERTICES          0x80E8
        #define GL_MAX_ELEMENTS_INDICES           0x80E9
        #define GL_CLAMP_TO_EDGE                  0x812F
        #define GL_TEXTURE_MIN_LOD                0x813A
        #define GL_TEXTURE_MAX_LOD                0x813B
        #define GL_TEXTURE_BASE_LEVEL             0x813C
        #define GL_TEXTURE_MAX_LEVEL              0x813D
        #define GL_SMOOTH_POINT_SIZE_RANGE        0x0B12
        #define GL_SMOOTH_POINT_SIZE_GRANULARITY  0x0B13
        #define GL_SMOOTH_LINE_WIDTH_RANGE        0x0B22
        #define GL_SMOOTH_LINE_WIDTH_GRANULARITY  0x0B23
        #define GL_ALIASED_LINE_WIDTH_RANGE       0x846E
        //typedef void (*PFNGLDRAWRANGEELEMENTSPROC)(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices);
        //typedef void (*PFNGLTEXIMAGE3DPROC)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
        //typedef void (*PFNGLTEXSUBIMAGE3DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels);
        //typedef void (*PFNGLCOPYTEXSUBIMAGE3DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
    #endif /* GL_VERSION_1_2 */

    #ifndef GL_VERSION_1_3
        #define GL_VERSION_1_3 1
        #define GL_TEXTURE0                       0x84C0
        #define GL_TEXTURE1                       0x84C1
        #define GL_TEXTURE2                       0x84C2
        #define GL_TEXTURE3                       0x84C3
        #define GL_TEXTURE4                       0x84C4
        #define GL_TEXTURE5                       0x84C5
        #define GL_TEXTURE6                       0x84C6
        #define GL_TEXTURE7                       0x84C7
        #define GL_TEXTURE8                       0x84C8
        #define GL_TEXTURE9                       0x84C9
        #define GL_TEXTURE10                      0x84CA
        #define GL_TEXTURE11                      0x84CB
        #define GL_TEXTURE12                      0x84CC
        #define GL_TEXTURE13                      0x84CD
        #define GL_TEXTURE14                      0x84CE
        #define GL_TEXTURE15                      0x84CF
        #define GL_TEXTURE16                      0x84D0
        #define GL_TEXTURE17                      0x84D1
        #define GL_TEXTURE18                      0x84D2
        #define GL_TEXTURE19                      0x84D3
        #define GL_TEXTURE20                      0x84D4
        #define GL_TEXTURE21                      0x84D5
        #define GL_TEXTURE22                      0x84D6
        #define GL_TEXTURE23                      0x84D7
        #define GL_TEXTURE24                      0x84D8
        #define GL_TEXTURE25                      0x84D9
        #define GL_TEXTURE26                      0x84DA
        #define GL_TEXTURE27                      0x84DB
        #define GL_TEXTURE28                      0x84DC
        #define GL_TEXTURE29                      0x84DD
        #define GL_TEXTURE30                      0x84DE
        #define GL_TEXTURE31                      0x84DF
        #define GL_ACTIVE_TEXTURE                 0x84E0
        #define GL_MULTISAMPLE                    0x809D
        #define GL_SAMPLE_ALPHA_TO_COVERAGE       0x809E
        #define GL_SAMPLE_ALPHA_TO_ONE            0x809F
        #define GL_SAMPLE_COVERAGE                0x80A0
        #define GL_SAMPLE_BUFFERS                 0x80A8
        #define GL_SAMPLES                        0x80A9
        #define GL_SAMPLE_COVERAGE_VALUE          0x80AA
        #define GL_SAMPLE_COVERAGE_INVERT         0x80AB
        #define GL_TEXTURE_CUBE_MAP               0x8513
        #define GL_TEXTURE_BINDING_CUBE_MAP       0x8514
        #define GL_TEXTURE_CUBE_MAP_POSITIVE_X    0x8515
        #define GL_TEXTURE_CUBE_MAP_NEGATIVE_X    0x8516
        #define GL_TEXTURE_CUBE_MAP_POSITIVE_Y    0x8517
        #define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y    0x8518
        #define GL_TEXTURE_CUBE_MAP_POSITIVE_Z    0x8519
        #define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z    0x851A
        #define GL_PROXY_TEXTURE_CUBE_MAP         0x851B
        #define GL_MAX_CUBE_MAP_TEXTURE_SIZE      0x851C
        #define GL_COMPRESSED_RGB                 0x84ED
        #define GL_COMPRESSED_RGBA                0x84EE
        #define GL_TEXTURE_COMPRESSION_HINT       0x84EF
        #define GL_TEXTURE_COMPRESSED_IMAGE_SIZE  0x86A0
        #define GL_TEXTURE_COMPRESSED             0x86A1
        #define GL_NUM_COMPRESSED_TEXTURE_FORMATS 0x86A2
        #define GL_COMPRESSED_TEXTURE_FORMATS     0x86A3
        #define GL_CLAMP_TO_BORDER                0x812D
        typedef void (*PFNGLACTIVETEXTUREPROC)(GLenum texture);
        //typedef void (*PFNGLSAMPLECOVERAGEPROC)(GLfloat value, GLboolean invert);
        //typedef void (*PFNGLCOMPRESSEDTEXIMAGE3DPROC)(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data);
        //typedef void (*PFNGLCOMPRESSEDTEXIMAGE2DPROC)(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data);
        //typedef void (*PFNGLCOMPRESSEDTEXIMAGE1DPROC)(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *data);
        //typedef void (*PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data);
        //typedef void (*PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data);
        //typedef void (*PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC)(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data);
        //typedef void (*PFNGLGETCOMPRESSEDTEXIMAGEPROC)(GLenum target, GLint level, void *img);
    #endif /* GL_VERSION_1_3 */

    #ifndef GL_VERSION_1_4
        #define GL_VERSION_1_4 1
        #define GL_BLEND_DST_RGB                  0x80C8
        #define GL_BLEND_SRC_RGB                  0x80C9
        #define GL_BLEND_DST_ALPHA                0x80CA
        #define GL_BLEND_SRC_ALPHA                0x80CB
        #define GL_POINT_FADE_THRESHOLD_SIZE      0x8128
        #define GL_DEPTH_COMPONENT16              0x81A5
        #define GL_DEPTH_COMPONENT24              0x81A6
        #define GL_DEPTH_COMPONENT32              0x81A7
        #define GL_MIRRORED_REPEAT                0x8370
        #define GL_MAX_TEXTURE_LOD_BIAS           0x84FD
        #define GL_TEXTURE_LOD_BIAS               0x8501
        #define GL_INCR_WRAP                      0x8507
        #define GL_DECR_WRAP                      0x8508
        #define GL_TEXTURE_DEPTH_SIZE             0x884A
        #define GL_TEXTURE_COMPARE_MODE           0x884C
        #define GL_TEXTURE_COMPARE_FUNC           0x884D
        #define GL_BLEND_COLOR                    0x8005
        #define GL_BLEND_EQUATION                 0x8009
        #define GL_CONSTANT_COLOR                 0x8001
        #define GL_ONE_MINUS_CONSTANT_COLOR       0x8002
        #define GL_CONSTANT_ALPHA                 0x8003
        #define GL_ONE_MINUS_CONSTANT_ALPHA       0x8004
        #define GL_FUNC_ADD                       0x8006
        #define GL_FUNC_REVERSE_SUBTRACT          0x800B
        #define GL_FUNC_SUBTRACT                  0x800A
        #define GL_MIN                            0x8007
        #define GL_MAX                            0x8008
        typedef void (*PFNGLBLENDEQUATIONPROC)(GLenum mode);
        //typedef void (*PFNGLBLENDFUNCSEPARATEPROC)(GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
        //typedef void (*PFNGLMULTIDRAWARRAYSPROC)(GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount);
        //typedef void (*PFNGLMULTIDRAWELEMENTSPROC)(GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount);
        //typedef void (*PFNGLPOINTPARAMETERFPROC)(GLenum pname, GLfloat param);
        //typedef void (*PFNGLPOINTPARAMETERFVPROC)(GLenum pname, const GLfloat *params);
        //typedef void (*PFNGLPOINTPARAMETERIPROC)(GLenum pname, GLint param);
        //typedef void (*PFNGLPOINTPARAMETERIVPROC)(GLenum pname, const GLint *params);
        //typedef void (*PFNGLBLENDCOLORPROC)(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
    #endif /* GL_VERSION_1_4 */

    #ifndef GL_VERSION_1_5
        #define GL_VERSION_1_5 1
        typedef khronos_ssize_t GLsizeiptr;
        typedef khronos_intptr_t GLintptr;
        #define GL_BUFFER_SIZE                    0x8764
        #define GL_BUFFER_USAGE                   0x8765
        #define GL_QUERY_COUNTER_BITS             0x8864
        #define GL_CURRENT_QUERY                  0x8865
        #define GL_QUERY_RESULT                   0x8866
        #define GL_QUERY_RESULT_AVAILABLE         0x8867
        #define GL_ARRAY_BUFFER                   0x8892
        #define GL_ELEMENT_ARRAY_BUFFER           0x8893
        #define GL_ARRAY_BUFFER_BINDING           0x8894
        #define GL_ELEMENT_ARRAY_BUFFER_BINDING   0x8895
        #define GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING 0x889F
        #define GL_READ_ONLY                      0x88B8
        #define GL_WRITE_ONLY                     0x88B9
        #define GL_READ_WRITE                     0x88BA
        #define GL_BUFFER_ACCESS                  0x88BB
        #define GL_BUFFER_MAPPED                  0x88BC
        #define GL_BUFFER_MAP_POINTER             0x88BD
        #define GL_STREAM_DRAW                    0x88E0
        #define GL_STREAM_READ                    0x88E1
        #define GL_STREAM_COPY                    0x88E2
        #define GL_STATIC_DRAW                    0x88E4
        #define GL_STATIC_READ                    0x88E5
        #define GL_STATIC_COPY                    0x88E6
        #define GL_DYNAMIC_DRAW                   0x88E8
        #define GL_DYNAMIC_READ                   0x88E9
        #define GL_DYNAMIC_COPY                   0x88EA
        #define GL_SAMPLES_PASSED                 0x8914
        #define GL_SRC1_ALPHA                     0x8589
        typedef void (*PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
        typedef void (*PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint *buffers);
        typedef void (*PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
        typedef void (*PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
        //typedef void (*PFNGLGENQUERIESPROC)(GLsizei n, GLuint *ids);
        //typedef void (*PFNGLDELETEQUERIESPROC)(GLsizei n, const GLuint *ids);
        //typedef GLboolean (*PFNGLISQUERYPROC)(GLuint id);
        //typedef void (*PFNGLBEGINQUERYPROC)(GLenum target, GLuint id);
        //typedef void (*PFNGLENDQUERYPROC)(GLenum target);
        //typedef void (*PFNGLGETQUERYIVPROC)(GLenum target, GLenum pname, GLint *params);
        //typedef void (*PFNGLGETQUERYOBJECTIVPROC)(GLuint id, GLenum pname, GLint *params);
        //typedef void (*PFNGLGETQUERYOBJECTUIVPROC)(GLuint id, GLenum pname, GLuint *params);
        //typedef GLboolean (*PFNGLISBUFFERPROC)(GLuint buffer);
        //typedef void (*PFNGLBUFFERSUBDATAPROC)(GLenum target, GLintptr offset, GLsizeiptr size, const void *data);
        //typedef void (*PFNGLGETBUFFERSUBDATAPROC)(GLenum target, GLintptr offset, GLsizeiptr size, void *data);
        //typedef void *(*PFNGLMAPBUFFERPROC)(GLenum target, GLenum access);
        //typedef GLboolean (*PFNGLUNMAPBUFFERPROC)(GLenum target);
        //typedef void (*PFNGLGETBUFFERPARAMETERIVPROC)(GLenum target, GLenum pname, GLint *params);
        //typedef void (*PFNGLGETBUFFERPOINTERVPROC)(GLenum target, GLenum pname, void **params);
    #endif /* GL_VERSION_1_5 */

    #ifndef GL_VERSION_2_0
        #define GL_VERSION_2_0 1
        typedef char GLchar;
        typedef khronos_int16_t GLshort;
        typedef khronos_int8_t GLbyte;
        typedef khronos_uint16_t GLushort;
        #define GL_BLEND_EQUATION_RGB             0x8009
        #define GL_VERTEX_ATTRIB_ARRAY_ENABLED    0x8622
        #define GL_VERTEX_ATTRIB_ARRAY_SIZE       0x8623
        #define GL_VERTEX_ATTRIB_ARRAY_STRIDE     0x8624
        #define GL_VERTEX_ATTRIB_ARRAY_TYPE       0x8625
        #define GL_CURRENT_VERTEX_ATTRIB          0x8626
        #define GL_VERTEX_PROGRAM_POINT_SIZE      0x8642
        #define GL_VERTEX_ATTRIB_ARRAY_POINTER    0x8645
        #define GL_STENCIL_BACK_FUNC              0x8800
        #define GL_STENCIL_BACK_FAIL              0x8801
        #define GL_STENCIL_BACK_PASS_DEPTH_FAIL   0x8802
        #define GL_STENCIL_BACK_PASS_DEPTH_PASS   0x8803
        #define GL_MAX_DRAW_BUFFERS               0x8824
        #define GL_DRAW_BUFFER0                   0x8825
        #define GL_DRAW_BUFFER1                   0x8826
        #define GL_DRAW_BUFFER2                   0x8827
        #define GL_DRAW_BUFFER3                   0x8828
        #define GL_DRAW_BUFFER4                   0x8829
        #define GL_DRAW_BUFFER5                   0x882A
        #define GL_DRAW_BUFFER6                   0x882B
        #define GL_DRAW_BUFFER7                   0x882C
        #define GL_DRAW_BUFFER8                   0x882D
        #define GL_DRAW_BUFFER9                   0x882E
        #define GL_DRAW_BUFFER10                  0x882F
        #define GL_DRAW_BUFFER11                  0x8830
        #define GL_DRAW_BUFFER12                  0x8831
        #define GL_DRAW_BUFFER13                  0x8832
        #define GL_DRAW_BUFFER14                  0x8833
        #define GL_DRAW_BUFFER15                  0x8834
        #define GL_BLEND_EQUATION_ALPHA           0x883D
        #define GL_MAX_VERTEX_ATTRIBS             0x8869
        #define GL_VERTEX_ATTRIB_ARRAY_NORMALIZED 0x886A
        #define GL_MAX_TEXTURE_IMAGE_UNITS        0x8872
        #define GL_FRAGMENT_SHADER                0x8B30
        #define GL_VERTEX_SHADER                  0x8B31
        #define GL_MAX_FRAGMENT_UNIFORM_COMPONENTS 0x8B49
        #define GL_MAX_VERTEX_UNIFORM_COMPONENTS  0x8B4A
        #define GL_MAX_VARYING_FLOATS             0x8B4B
        #define GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS 0x8B4C
        #define GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS 0x8B4D
        #define GL_SHADER_TYPE                    0x8B4F
        #define GL_FLOAT_VEC2                     0x8B50
        #define GL_FLOAT_VEC3                     0x8B51
        #define GL_FLOAT_VEC4                     0x8B52
        #define GL_INT_VEC2                       0x8B53
        #define GL_INT_VEC3                       0x8B54
        #define GL_INT_VEC4                       0x8B55
        #define GL_BOOL                           0x8B56
        #define GL_BOOL_VEC2                      0x8B57
        #define GL_BOOL_VEC3                      0x8B58
        #define GL_BOOL_VEC4                      0x8B59
        #define GL_FLOAT_MAT2                     0x8B5A
        #define GL_FLOAT_MAT3                     0x8B5B
        #define GL_FLOAT_MAT4                     0x8B5C
        #define GL_SAMPLER_1D                     0x8B5D
        #define GL_SAMPLER_2D                     0x8B5E
        #define GL_SAMPLER_3D                     0x8B5F
        #define GL_SAMPLER_CUBE                   0x8B60
        #define GL_SAMPLER_1D_SHADOW              0x8B61
        #define GL_SAMPLER_2D_SHADOW              0x8B62
        #define GL_DELETE_STATUS                  0x8B80
        #define GL_COMPILE_STATUS                 0x8B81
        #define GL_LINK_STATUS                    0x8B82
        #define GL_VALIDATE_STATUS                0x8B83
        #define GL_INFO_LOG_LENGTH                0x8B84
        #define GL_ATTACHED_SHADERS               0x8B85
        #define GL_ACTIVE_UNIFORMS                0x8B86
        #define GL_ACTIVE_UNIFORM_MAX_LENGTH      0x8B87
        #define GL_SHADER_SOURCE_LENGTH           0x8B88
        #define GL_ACTIVE_ATTRIBUTES              0x8B89
        #define GL_ACTIVE_ATTRIBUTE_MAX_LENGTH    0x8B8A
        #define GL_FRAGMENT_SHADER_DERIVATIVE_HINT 0x8B8B
        #define GL_SHADING_LANGUAGE_VERSION       0x8B8C
        #define GL_CURRENT_PROGRAM                0x8B8D
        #define GL_POINT_SPRITE_COORD_ORIGIN      0x8CA0
        #define GL_LOWER_LEFT                     0x8CA1
        #define GL_UPPER_LEFT                     0x8CA2
        #define GL_STENCIL_BACK_REF               0x8CA3
        #define GL_STENCIL_BACK_VALUE_MASK        0x8CA4
        #define GL_STENCIL_BACK_WRITEMASK         0x8CA5
        typedef void (*PFNGLDRAWBUFFERSPROC)(GLsizei n, const GLenum *bufs);
        typedef void (*PFNGLSTENCILOPSEPARATEPROC)(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
        typedef void (*PFNGLSTENCILFUNCSEPARATEPROC)(GLenum face, GLenum func, GLint ref, GLuint mask);
        typedef void (*PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
        typedef void (*PFNGLCOMPILESHADERPROC)(GLuint shader);
        typedef GLuint (*PFNGLCREATEPROGRAMPROC)(void);
        typedef GLuint (*PFNGLCREATESHADERPROC)(GLenum type);
        typedef void (*PFNGLDELETEPROGRAMPROC)(GLuint program);
        typedef void (*PFNGLDELETESHADERPROC)(GLuint shader);
        typedef void (*PFNGLDISABLEVERTEXATTRIBARRAYPROC)(GLuint index);
        typedef void (*PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
        typedef GLint (*PFNGLGETATTRIBLOCATIONPROC)(GLuint program, const GLchar *name);
        typedef void (*PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint *params);
        typedef void (*PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
        typedef void (*PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint *params);
        typedef void (*PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
        typedef GLint (*PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar *name);
        typedef void (*PFNGLLINKPROGRAMPROC)(GLuint program);
        typedef void (*PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
        typedef void (*PFNGLUSEPROGRAMPROC)(GLuint program);
        typedef void (*PFNGLUNIFORM1FPROC)(GLint location, GLfloat v0);
        typedef void (*PFNGLUNIFORM1FVPROC)(GLint location, GLsizei count, const GLfloat *value);
        typedef void (*PFNGLUNIFORM2FVPROC)(GLint location, GLsizei count, const GLfloat *value);
        typedef void (*PFNGLUNIFORM3FVPROC)(GLint location, GLsizei count, const GLfloat *value);
        typedef void (*PFNGLUNIFORM4FVPROC)(GLint location, GLsizei count, const GLfloat *value);
        typedef void (*PFNGLUNIFORM1IVPROC)(GLint location, GLsizei count, const GLint *value);
        typedef void (*PFNGLUNIFORM2IVPROC)(GLint location, GLsizei count, const GLint *value);
        typedef void (*PFNGLUNIFORM3IVPROC)(GLint location, GLsizei count, const GLint *value);
        typedef void (*PFNGLUNIFORM4IVPROC)(GLint location, GLsizei count, const GLint *value);
        // typedef void (*PFNGLUNIFORMMATRIX4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
        typedef void (*PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
        //typedef void (*PFNGLBLENDEQUATIONSEPARATEPROC)(GLenum modeRGB, GLenum modeAlpha);
        //typedef void (*PFNGLSTENCILMASKSEPARATEPROC)(GLenum face, GLuint mask);
        //typedef void (*PFNGLBINDATTRIBLOCATIONPROC)(GLuint program, GLuint index, const GLchar *name);
        //typedef void (*PFNGLDETACHSHADERPROC)(GLuint program, GLuint shader);
        //typedef void (*PFNGLGETACTIVEATTRIBPROC)(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
        //typedef void (*PFNGLGETACTIVEUNIFORMPROC)(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
        //typedef void (*PFNGLGETATTACHEDSHADERSPROC)(GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders);
        //typedef void (*PFNGLGETSHADERSOURCEPROC)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source);
        //typedef void (*PFNGLGETUNIFORMFVPROC)(GLuint program, GLint location, GLfloat *params);
        //typedef void (*PFNGLGETUNIFORMIVPROC)(GLuint program, GLint location, GLint *params);
        //typedef void (*PFNGLGETVERTEXATTRIBDVPROC)(GLuint index, GLenum pname, GLdouble *params);
        //typedef void (*PFNGLGETVERTEXATTRIBFVPROC)(GLuint index, GLenum pname, GLfloat *params);
        //typedef void (*PFNGLGETVERTEXATTRIBIVPROC)(GLuint index, GLenum pname, GLint *params);
        //typedef void (*PFNGLGETVERTEXATTRIBPOINTERVPROC)(GLuint index, GLenum pname, void **pointer);
        //typedef GLboolean (*PFNGLISPROGRAMPROC)(GLuint program);
        //typedef GLboolean (*PFNGLISSHADERPROC)(GLuint shader);
        //typedef void (*PFNGLUNIFORM2FPROC)(GLint location, GLfloat v0, GLfloat v1);
        //typedef void (*PFNGLUNIFORM3FPROC)(GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
        //typedef void (*PFNGLUNIFORM4FPROC)(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
        //typedef void (*PFNGLUNIFORM1IPROC)(GLint location, GLint v0);
        //typedef void (*PFNGLUNIFORM2IPROC)(GLint location, GLint v0, GLint v1);
        //typedef void (*PFNGLUNIFORM3IPROC)(GLint location, GLint v0, GLint v1, GLint v2);
        //typedef void (*PFNGLUNIFORM4IPROC)(GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
        //typedef void (*PFNGLUNIFORMMATRIX2FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
        typedef void (*PFNGLUNIFORMMATRIX3FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
        //typedef void (*PFNGLVALIDATEPROGRAMPROC)(GLuint program);
        //typedef void (*PFNGLVERTEXATTRIB1DPROC)(GLuint index, GLdouble x);
        //typedef void (*PFNGLVERTEXATTRIB1DVPROC)(GLuint index, const GLdouble *v);
        //typedef void (*PFNGLVERTEXATTRIB1FPROC)(GLuint index, GLfloat x);
        //typedef void (*PFNGLVERTEXATTRIB1FVPROC)(GLuint index, const GLfloat *v);
        //typedef void (*PFNGLVERTEXATTRIB1SPROC)(GLuint index, GLshort x);
        //typedef void (*PFNGLVERTEXATTRIB1SVPROC)(GLuint index, const GLshort *v);
        //typedef void (*PFNGLVERTEXATTRIB2DPROC)(GLuint index, GLdouble x, GLdouble y);
        //typedef void (*PFNGLVERTEXATTRIB2DVPROC)(GLuint index, const GLdouble *v);
        //typedef void (*PFNGLVERTEXATTRIB2FPROC)(GLuint index, GLfloat x, GLfloat y);
        //typedef void (*PFNGLVERTEXATTRIB2FVPROC)(GLuint index, const GLfloat *v);
        //typedef void (*PFNGLVERTEXATTRIB2SPROC)(GLuint index, GLshort x, GLshort y);
        //typedef void (*PFNGLVERTEXATTRIB2SVPROC)(GLuint index, const GLshort *v);
        //typedef void (*PFNGLVERTEXATTRIB3DPROC)(GLuint index, GLdouble x, GLdouble y, GLdouble z);
        //typedef void (*PFNGLVERTEXATTRIB3DVPROC)(GLuint index, const GLdouble *v);
        //typedef void (*PFNGLVERTEXATTRIB3FPROC)(GLuint index, GLfloat x, GLfloat y, GLfloat z);
        //typedef void (*PFNGLVERTEXATTRIB3FVPROC)(GLuint index, const GLfloat *v);
        //typedef void (*PFNGLVERTEXATTRIB3SPROC)(GLuint index, GLshort x, GLshort y, GLshort z);
        //typedef void (*PFNGLVERTEXATTRIB3SVPROC)(GLuint index, const GLshort *v);
        //typedef void (*PFNGLVERTEXATTRIB4NBVPROC)(GLuint index, const GLbyte *v);
        //typedef void (*PFNGLVERTEXATTRIB4NIVPROC)(GLuint index, const GLint *v);
        //typedef void (*PFNGLVERTEXATTRIB4NSVPROC)(GLuint index, const GLshort *v);
        //typedef void (*PFNGLVERTEXATTRIB4NUBPROC)(GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
        //typedef void (*PFNGLVERTEXATTRIB4NUBVPROC)(GLuint index, const GLubyte *v);
        //typedef void (*PFNGLVERTEXATTRIB4NUIVPROC)(GLuint index, const GLuint *v);
        //typedef void (*PFNGLVERTEXATTRIB4NUSVPROC)(GLuint index, const GLushort *v);
        //typedef void (*PFNGLVERTEXATTRIB4BVPROC)(GLuint index, const GLbyte *v);
        //typedef void (*PFNGLVERTEXATTRIB4DPROC)(GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
        //typedef void (*PFNGLVERTEXATTRIB4DVPROC)(GLuint index, const GLdouble *v);
        typedef void (*PFNGLVERTEXATTRIB4FPROC)(GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
        //typedef void (*PFNGLVERTEXATTRIB4FVPROC)(GLuint index, const GLfloat *v);
        //typedef void (*PFNGLVERTEXATTRIB4IVPROC)(GLuint index, const GLint *v);
        //typedef void (*PFNGLVERTEXATTRIB4SPROC)(GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
        //typedef void (*PFNGLVERTEXATTRIB4SVPROC)(GLuint index, const GLshort *v);
        //typedef void (*PFNGLVERTEXATTRIB4UBVPROC)(GLuint index, const GLubyte *v);
        //typedef void (*PFNGLVERTEXATTRIB4UIVPROC)(GLuint index, const GLuint *v);
        //typedef void (*PFNGLVERTEXATTRIB4USVPROC)(GLuint index, const GLushort *v);
    #endif /* GL_VERSION_2_0 */

    #ifndef GL_VERSION_2_1
        #define GL_VERSION_2_1 1
        #define GL_PIXEL_PACK_BUFFER              0x88EB
        #define GL_PIXEL_UNPACK_BUFFER            0x88EC
        #define GL_PIXEL_PACK_BUFFER_BINDING      0x88ED
        #define GL_PIXEL_UNPACK_BUFFER_BINDING    0x88EF
        #define GL_FLOAT_MAT2x3                   0x8B65
        #define GL_FLOAT_MAT2x4                   0x8B66
        #define GL_FLOAT_MAT3x2                   0x8B67
        #define GL_FLOAT_MAT3x4                   0x8B68
        #define GL_FLOAT_MAT4x2                   0x8B69
        #define GL_FLOAT_MAT4x3                   0x8B6A
        #define GL_SRGB                           0x8C40
        #define GL_SRGB8                          0x8C41
        #define GL_SRGB_ALPHA                     0x8C42
        #define GL_SRGB8_ALPHA8                   0x8C43
        #define GL_COMPRESSED_SRGB                0x8C48
        #define GL_COMPRESSED_SRGB_ALPHA          0x8C49
        //typedef void (*PFNGLUNIFORMMATRIX2X3FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
        //typedef void (*PFNGLUNIFORMMATRIX3X2FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
        //typedef void (*PFNGLUNIFORMMATRIX2X4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
        //typedef void (*PFNGLUNIFORMMATRIX4X2FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
        //typedef void (*PFNGLUNIFORMMATRIX3X4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
        //typedef void (*PFNGLUNIFORMMATRIX4X3FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    #endif /* GL_VERSION_2_1 */

    #ifndef GL_VERSION_3_0
        #define GL_VERSION_3_0 1
        typedef khronos_uint16_t GLhalf;
        #define GL_COMPARE_REF_TO_TEXTURE         0x884E
        #define GL_CLIP_DISTANCE0                 0x3000
        #define GL_CLIP_DISTANCE1                 0x3001
        #define GL_CLIP_DISTANCE2                 0x3002
        #define GL_CLIP_DISTANCE3                 0x3003
        #define GL_CLIP_DISTANCE4                 0x3004
        #define GL_CLIP_DISTANCE5                 0x3005
        #define GL_CLIP_DISTANCE6                 0x3006
        #define GL_CLIP_DISTANCE7                 0x3007
        #define GL_MAX_CLIP_DISTANCES             0x0D32
        #define GL_MAJOR_VERSION                  0x821B
        #define GL_MINOR_VERSION                  0x821C
        #define GL_NUM_EXTENSIONS                 0x821D
        #define GL_CONTEXT_FLAGS                  0x821E
        #define GL_COMPRESSED_RED                 0x8225
        #define GL_COMPRESSED_RG                  0x8226
        #define GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT 0x00000001
        #define GL_RGBA32F                        0x8814
        #define GL_RGB32F                         0x8815
        #define GL_RGBA16F                        0x881A
        #define GL_RGB16F                         0x881B
        #define GL_VERTEX_ATTRIB_ARRAY_INTEGER    0x88FD
        #define GL_MAX_ARRAY_TEXTURE_LAYERS       0x88FF
        #define GL_MIN_PROGRAM_TEXEL_OFFSET       0x8904
        #define GL_MAX_PROGRAM_TEXEL_OFFSET       0x8905
        #define GL_CLAMP_READ_COLOR               0x891C
        #define GL_FIXED_ONLY                     0x891D
        #define GL_MAX_VARYING_COMPONENTS         0x8B4B
        #define GL_TEXTURE_1D_ARRAY               0x8C18
        #define GL_PROXY_TEXTURE_1D_ARRAY         0x8C19
        #define GL_TEXTURE_2D_ARRAY               0x8C1A
        #define GL_PROXY_TEXTURE_2D_ARRAY         0x8C1B
        #define GL_TEXTURE_BINDING_1D_ARRAY       0x8C1C
        #define GL_TEXTURE_BINDING_2D_ARRAY       0x8C1D
        #define GL_R11F_G11F_B10F                 0x8C3A
        #define GL_UNSIGNED_INT_10F_11F_11F_REV   0x8C3B
        #define GL_RGB9_E5                        0x8C3D
        #define GL_UNSIGNED_INT_5_9_9_9_REV       0x8C3E
        #define GL_TEXTURE_SHARED_SIZE            0x8C3F
        #define GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH 0x8C76
        #define GL_TRANSFORM_FEEDBACK_BUFFER_MODE 0x8C7F
        #define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS 0x8C80
        #define GL_TRANSFORM_FEEDBACK_VARYINGS    0x8C83
        #define GL_TRANSFORM_FEEDBACK_BUFFER_START 0x8C84
        #define GL_TRANSFORM_FEEDBACK_BUFFER_SIZE 0x8C85
        #define GL_PRIMITIVES_GENERATED           0x8C87
        #define GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN 0x8C88
        #define GL_RASTERIZER_DISCARD             0x8C89
        #define GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS 0x8C8A
        #define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS 0x8C8B
        #define GL_INTERLEAVED_ATTRIBS            0x8C8C
        #define GL_SEPARATE_ATTRIBS               0x8C8D
        #define GL_TRANSFORM_FEEDBACK_BUFFER      0x8C8E
        #define GL_TRANSFORM_FEEDBACK_BUFFER_BINDING 0x8C8F
        #define GL_RGBA32UI                       0x8D70
        #define GL_RGB32UI                        0x8D71
        #define GL_RGBA16UI                       0x8D76
        #define GL_RGB16UI                        0x8D77
        #define GL_RGBA8UI                        0x8D7C
        #define GL_RGB8UI                         0x8D7D
        #define GL_RGBA32I                        0x8D82
        #define GL_RGB32I                         0x8D83
        #define GL_RGBA16I                        0x8D88
        #define GL_RGB16I                         0x8D89
        #define GL_RGBA8I                         0x8D8E
        #define GL_RGB8I                          0x8D8F
        #define GL_RED_INTEGER                    0x8D94
        #define GL_GREEN_INTEGER                  0x8D95
        #define GL_BLUE_INTEGER                   0x8D96
        #define GL_RGB_INTEGER                    0x8D98
        #define GL_RGBA_INTEGER                   0x8D99
        #define GL_BGR_INTEGER                    0x8D9A
        #define GL_BGRA_INTEGER                   0x8D9B
        #define GL_SAMPLER_1D_ARRAY               0x8DC0
        #define GL_SAMPLER_2D_ARRAY               0x8DC1
        #define GL_SAMPLER_1D_ARRAY_SHADOW        0x8DC3
        #define GL_SAMPLER_2D_ARRAY_SHADOW        0x8DC4
        #define GL_SAMPLER_CUBE_SHADOW            0x8DC5
        #define GL_UNSIGNED_INT_VEC2              0x8DC6
        #define GL_UNSIGNED_INT_VEC3              0x8DC7
        #define GL_UNSIGNED_INT_VEC4              0x8DC8
        #define GL_INT_SAMPLER_1D                 0x8DC9
        #define GL_INT_SAMPLER_2D                 0x8DCA
        #define GL_INT_SAMPLER_3D                 0x8DCB
        #define GL_INT_SAMPLER_CUBE               0x8DCC
        #define GL_INT_SAMPLER_1D_ARRAY           0x8DCE
        #define GL_INT_SAMPLER_2D_ARRAY           0x8DCF
        #define GL_UNSIGNED_INT_SAMPLER_1D        0x8DD1
        #define GL_UNSIGNED_INT_SAMPLER_2D        0x8DD2
        #define GL_UNSIGNED_INT_SAMPLER_3D        0x8DD3
        #define GL_UNSIGNED_INT_SAMPLER_CUBE      0x8DD4
        #define GL_UNSIGNED_INT_SAMPLER_1D_ARRAY  0x8DD6
        #define GL_UNSIGNED_INT_SAMPLER_2D_ARRAY  0x8DD7
        #define GL_QUERY_WAIT                     0x8E13
        #define GL_QUERY_NO_WAIT                  0x8E14
        #define GL_QUERY_BY_REGION_WAIT           0x8E15
        #define GL_QUERY_BY_REGION_NO_WAIT        0x8E16
        #define GL_BUFFER_ACCESS_FLAGS            0x911F
        #define GL_BUFFER_MAP_LENGTH              0x9120
        #define GL_BUFFER_MAP_OFFSET              0x9121
        #define GL_DEPTH_COMPONENT32F             0x8CAC
        #define GL_DEPTH32F_STENCIL8              0x8CAD
        #define GL_FLOAT_32_UNSIGNED_INT_24_8_REV 0x8DAD
        #define GL_INVALID_FRAMEBUFFER_OPERATION  0x0506
        #define GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING 0x8210
        #define GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE 0x8211
        #define GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE 0x8212
        #define GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE 0x8213
        #define GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE 0x8214
        #define GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE 0x8215
        #define GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE 0x8216
        #define GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE 0x8217
        #define GL_FRAMEBUFFER_DEFAULT            0x8218
        #define GL_FRAMEBUFFER_UNDEFINED          0x8219
        #define GL_DEPTH_STENCIL_ATTACHMENT       0x821A
        #define GL_MAX_RENDERBUFFER_SIZE          0x84E8
        #define GL_DEPTH_STENCIL                  0x84F9
        #define GL_UNSIGNED_INT_24_8              0x84FA
        #define GL_DEPTH24_STENCIL8               0x88F0
        #define GL_TEXTURE_STENCIL_SIZE           0x88F1
        #define GL_TEXTURE_RED_TYPE               0x8C10
        #define GL_TEXTURE_GREEN_TYPE             0x8C11
        #define GL_TEXTURE_BLUE_TYPE              0x8C12
        #define GL_TEXTURE_ALPHA_TYPE             0x8C13
        #define GL_TEXTURE_DEPTH_TYPE             0x8C16
        #define GL_UNSIGNED_NORMALIZED            0x8C17
        #define GL_FRAMEBUFFER_BINDING            0x8CA6
        #define GL_DRAW_FRAMEBUFFER_BINDING       0x8CA6
        #define GL_RENDERBUFFER_BINDING           0x8CA7
        #define GL_READ_FRAMEBUFFER               0x8CA8
        #define GL_DRAW_FRAMEBUFFER               0x8CA9
        #define GL_READ_FRAMEBUFFER_BINDING       0x8CAA
        #define GL_RENDERBUFFER_SAMPLES           0x8CAB
        #define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE 0x8CD0
        #define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME 0x8CD1
        #define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL 0x8CD2
        #define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE 0x8CD3
        #define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER 0x8CD4
        #define GL_FRAMEBUFFER_COMPLETE           0x8CD5
        #define GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT 0x8CD6
        #define GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT 0x8CD7
        #define GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER 0x8CDB
        #define GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER 0x8CDC
        #define GL_FRAMEBUFFER_UNSUPPORTED        0x8CDD
        #define GL_MAX_COLOR_ATTACHMENTS          0x8CDF
        #define GL_COLOR_ATTACHMENT0              0x8CE0
        #define GL_COLOR_ATTACHMENT1              0x8CE1
        #define GL_COLOR_ATTACHMENT2              0x8CE2
        #define GL_COLOR_ATTACHMENT3              0x8CE3
        #define GL_COLOR_ATTACHMENT4              0x8CE4
        #define GL_COLOR_ATTACHMENT5              0x8CE5
        #define GL_COLOR_ATTACHMENT6              0x8CE6
        #define GL_COLOR_ATTACHMENT7              0x8CE7
        #define GL_COLOR_ATTACHMENT8              0x8CE8
        #define GL_COLOR_ATTACHMENT9              0x8CE9
        #define GL_COLOR_ATTACHMENT10             0x8CEA
        #define GL_COLOR_ATTACHMENT11             0x8CEB
        #define GL_COLOR_ATTACHMENT12             0x8CEC
        #define GL_COLOR_ATTACHMENT13             0x8CED
        #define GL_COLOR_ATTACHMENT14             0x8CEE
        #define GL_COLOR_ATTACHMENT15             0x8CEF
        #define GL_COLOR_ATTACHMENT16             0x8CF0
        #define GL_COLOR_ATTACHMENT17             0x8CF1
        #define GL_COLOR_ATTACHMENT18             0x8CF2
        #define GL_COLOR_ATTACHMENT19             0x8CF3
        #define GL_COLOR_ATTACHMENT20             0x8CF4
        #define GL_COLOR_ATTACHMENT21             0x8CF5
        #define GL_COLOR_ATTACHMENT22             0x8CF6
        #define GL_COLOR_ATTACHMENT23             0x8CF7
        #define GL_COLOR_ATTACHMENT24             0x8CF8
        #define GL_COLOR_ATTACHMENT25             0x8CF9
        #define GL_COLOR_ATTACHMENT26             0x8CFA
        #define GL_COLOR_ATTACHMENT27             0x8CFB
        #define GL_COLOR_ATTACHMENT28             0x8CFC
        #define GL_COLOR_ATTACHMENT29             0x8CFD
        #define GL_COLOR_ATTACHMENT30             0x8CFE
        #define GL_COLOR_ATTACHMENT31             0x8CFF
        #define GL_DEPTH_ATTACHMENT               0x8D00
        #define GL_STENCIL_ATTACHMENT             0x8D20
        #define GL_FRAMEBUFFER                    0x8D40
        #define GL_RENDERBUFFER                   0x8D41
        #define GL_RENDERBUFFER_WIDTH             0x8D42
        #define GL_RENDERBUFFER_HEIGHT            0x8D43
        #define GL_RENDERBUFFER_INTERNAL_FORMAT   0x8D44
        #define GL_STENCIL_INDEX1                 0x8D46
        #define GL_STENCIL_INDEX4                 0x8D47
        #define GL_STENCIL_INDEX8                 0x8D48
        #define GL_STENCIL_INDEX16                0x8D49
        #define GL_RENDERBUFFER_RED_SIZE          0x8D50
        #define GL_RENDERBUFFER_GREEN_SIZE        0x8D51
        #define GL_RENDERBUFFER_BLUE_SIZE         0x8D52
        #define GL_RENDERBUFFER_ALPHA_SIZE        0x8D53
        #define GL_RENDERBUFFER_DEPTH_SIZE        0x8D54
        #define GL_RENDERBUFFER_STENCIL_SIZE      0x8D55
        #define GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE 0x8D56
        #define GL_MAX_SAMPLES                    0x8D57
        #define GL_FRAMEBUFFER_SRGB               0x8DB9
        #define GL_HALF_FLOAT                     0x140B
        #define GL_MAP_READ_BIT                   0x0001
        #define GL_MAP_WRITE_BIT                  0x0002
        #define GL_MAP_INVALIDATE_RANGE_BIT       0x0004
        #define GL_MAP_INVALIDATE_BUFFER_BIT      0x0008
        #define GL_MAP_FLUSH_EXPLICIT_BIT         0x0010
        #define GL_MAP_UNSYNCHRONIZED_BIT         0x0020
        #define GL_COMPRESSED_RED_RGTC1           0x8DBB
        #define GL_COMPRESSED_SIGNED_RED_RGTC1    0x8DBC
        #define GL_COMPRESSED_RG_RGTC2            0x8DBD
        #define GL_COMPRESSED_SIGNED_RG_RGTC2     0x8DBE
        #define GL_RG                             0x8227
        #define GL_RG_INTEGER                     0x8228
        #define GL_R8                             0x8229
        #define GL_R16                            0x822A
        #define GL_RG8                            0x822B
        #define GL_RG16                           0x822C
        #define GL_R16F                           0x822D
        #define GL_R32F                           0x822E
        #define GL_RG16F                          0x822F
        #define GL_RG32F                          0x8230
        #define GL_R8I                            0x8231
        #define GL_R8UI                           0x8232
        #define GL_R16I                           0x8233
        #define GL_R16UI                          0x8234
        #define GL_R32I                           0x8235
        #define GL_R32UI                          0x8236
        #define GL_RG8I                           0x8237
        #define GL_RG8UI                          0x8238
        #define GL_RG16I                          0x8239
        #define GL_RG16UI                         0x823A
        #define GL_RG32I                          0x823B
        #define GL_RG32UI                         0x823C
        #define GL_VERTEX_ARRAY_BINDING           0x85B5
        typedef void (*PFNGLBINDBUFFERRANGEPROC)(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);
        typedef void (*PFNGLBINDRENDERBUFFERPROC)(GLenum target, GLuint renderbuffer);
        typedef void (*PFNGLDELETERENDERBUFFERSPROC)(GLsizei n, const GLuint *renderbuffers);
        typedef void (*PFNGLGENRENDERBUFFERSPROC)(GLsizei n, GLuint *renderbuffers);
        typedef void (*PFNGLINVALIDATEFRAMEBUFFERPROC)(GLenum target, GLsizei numAttachments, const GLenum *attachments); // GLES
        typedef void (*PFNGLBINDFRAMEBUFFERPROC)(GLenum target, GLuint framebuffer);
        typedef void (*PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei n, const GLuint *framebuffers);
        typedef void (*PFNGLGENFRAMEBUFFERSPROC)(GLsizei n, GLuint *framebuffers);
        typedef void (*PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
        typedef void (*PFNGLFRAMEBUFFERRENDERBUFFERPROC)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
        typedef void (*PFNGLBLITFRAMEBUFFERPROC)(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
        typedef void (*PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC)(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
        typedef void (*PFNGLBINDVERTEXARRAYPROC)(GLuint array);
        typedef void (*PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n, const GLuint *arrays);
        typedef void (*PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint *arrays);
        //typedef void (*PFNGLCOLORMASKIPROC)(GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a);
        //typedef void (*PFNGLGETBOOLEANI_VPROC)(GLenum target, GLuint index, GLboolean *data);
        //typedef void (*PFNGLGETINTEGERI_VPROC)(GLenum target, GLuint index, GLint *data);
        //typedef void (*PFNGLENABLEIPROC)(GLenum target, GLuint index);
        //typedef void (*PFNGLDISABLEIPROC)(GLenum target, GLuint index);
        //typedef GLboolean (*PFNGLISENABLEDIPROC)(GLenum target, GLuint index);
        //typedef void (*PFNGLBEGINTRANSFORMFEEDBACKPROC)(GLenum primitiveMode);
        //typedef void (*PFNGLENDTRANSFORMFEEDBACKPROC)(void);
        //typedef void (*PFNGLBINDBUFFERBASEPROC)(GLenum target, GLuint index, GLuint buffer);
        //typedef void (*PFNGLTRANSFORMFEEDBACKVARYINGSPROC)(GLuint program, GLsizei count, const GLchar *const*varyings, GLenum bufferMode);
        //typedef void (*PFNGLGETTRANSFORMFEEDBACKVARYINGPROC)(GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLsizei *size, GLenum *type, GLchar *name);
        //typedef void (*PFNGLCLAMPCOLORPROC)(GLenum target, GLenum clamp);
        //typedef void (*PFNGLBEGINCONDITIONALRENDERPROC)(GLuint id, GLenum mode);
        //typedef void (*PFNGLENDCONDITIONALRENDERPROC)(void);
        //typedef void (*PFNGLVERTEXATTRIBIPOINTERPROC)(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer);
        //typedef void (*PFNGLGETVERTEXATTRIBIIVPROC)(GLuint index, GLenum pname, GLint *params);
        //typedef void (*PFNGLGETVERTEXATTRIBIUIVPROC)(GLuint index, GLenum pname, GLuint *params);
        //typedef void (*PFNGLVERTEXATTRIBI1IPROC)(GLuint index, GLint x);
        //typedef void (*PFNGLVERTEXATTRIBI2IPROC)(GLuint index, GLint x, GLint y);
        //typedef void (*PFNGLVERTEXATTRIBI3IPROC)(GLuint index, GLint x, GLint y, GLint z);
        //typedef void (*PFNGLVERTEXATTRIBI4IPROC)(GLuint index, GLint x, GLint y, GLint z, GLint w);
        //typedef void (*PFNGLVERTEXATTRIBI1UIPROC)(GLuint index, GLuint x);
        //typedef void (*PFNGLVERTEXATTRIBI2UIPROC)(GLuint index, GLuint x, GLuint y);
        //typedef void (*PFNGLVERTEXATTRIBI3UIPROC)(GLuint index, GLuint x, GLuint y, GLuint z);
        //typedef void (*PFNGLVERTEXATTRIBI4UIPROC)(GLuint index, GLuint x, GLuint y, GLuint z, GLuint w);
        //typedef void (*PFNGLVERTEXATTRIBI1IVPROC)(GLuint index, const GLint *v);
        //typedef void (*PFNGLVERTEXATTRIBI2IVPROC)(GLuint index, const GLint *v);
        //typedef void (*PFNGLVERTEXATTRIBI3IVPROC)(GLuint index, const GLint *v);
        //typedef void (*PFNGLVERTEXATTRIBI4IVPROC)(GLuint index, const GLint *v);
        //typedef void (*PFNGLVERTEXATTRIBI1UIVPROC)(GLuint index, const GLuint *v);
        //typedef void (*PFNGLVERTEXATTRIBI2UIVPROC)(GLuint index, const GLuint *v);
        //typedef void (*PFNGLVERTEXATTRIBI3UIVPROC)(GLuint index, const GLuint *v);
        //typedef void (*PFNGLVERTEXATTRIBI4UIVPROC)(GLuint index, const GLuint *v);
        //typedef void (*PFNGLVERTEXATTRIBI4BVPROC)(GLuint index, const GLbyte *v);
        //typedef void (*PFNGLVERTEXATTRIBI4SVPROC)(GLuint index, const GLshort *v);
        //typedef void (*PFNGLVERTEXATTRIBI4UBVPROC)(GLuint index, const GLubyte *v);
        //typedef void (*PFNGLVERTEXATTRIBI4USVPROC)(GLuint index, const GLushort *v);
        //typedef void (*PFNGLGETUNIFORMUIVPROC)(GLuint program, GLint location, GLuint *params);
        //typedef void (*PFNGLBINDFRAGDATALOCATIONPROC)(GLuint program, GLuint color, const GLchar *name);
        //typedef GLint (*PFNGLGETFRAGDATALOCATIONPROC)(GLuint program, const GLchar *name);
        //typedef void (*PFNGLUNIFORM1UIPROC)(GLint location, GLuint v0);
        //typedef void (*PFNGLUNIFORM2UIPROC)(GLint location, GLuint v0, GLuint v1);
        //typedef void (*PFNGLUNIFORM3UIPROC)(GLint location, GLuint v0, GLuint v1, GLuint v2);
        //typedef void (*PFNGLUNIFORM4UIPROC)(GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
        //typedef void (*PFNGLUNIFORM1UIVPROC)(GLint location, GLsizei count, const GLuint *value);
        //typedef void (*PFNGLUNIFORM2UIVPROC)(GLint location, GLsizei count, const GLuint *value);
        //typedef void (*PFNGLUNIFORM3UIVPROC)(GLint location, GLsizei count, const GLuint *value);
        //typedef void (*PFNGLUNIFORM4UIVPROC)(GLint location, GLsizei count, const GLuint *value);
        //typedef void (*PFNGLTEXPARAMETERIIVPROC)(GLenum target, GLenum pname, const GLint *params);
        //typedef void (*PFNGLTEXPARAMETERIUIVPROC)(GLenum target, GLenum pname, const GLuint *params);
        //typedef void (*PFNGLGETTEXPARAMETERIIVPROC)(GLenum target, GLenum pname, GLint *params);
        //typedef void (*PFNGLGETTEXPARAMETERIUIVPROC)(GLenum target, GLenum pname, GLuint *params);
        //typedef void (*PFNGLCLEARBUFFERIVPROC)(GLenum buffer, GLint drawbuffer, const GLint *value);
        //typedef void (*PFNGLCLEARBUFFERUIVPROC)(GLenum buffer, GLint drawbuffer, const GLuint *value);
        //typedef void (*PFNGLCLEARBUFFERFVPROC)(GLenum buffer, GLint drawbuffer, const GLfloat *value);
        //typedef void (*PFNGLCLEARBUFFERFIPROC)(GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil);
        //typedef const GLubyte *(*PFNGLGETSTRINGIPROC)(GLenum name, GLuint index);
        //typedef GLboolean (*PFNGLISRENDERBUFFERPROC)(GLuint renderbuffer);
        //typedef void (*PFNGLRENDERBUFFERSTORAGEPROC)(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
        //typedef void (*PFNGLGETRENDERBUFFERPARAMETERIVPROC)(GLenum target, GLenum pname, GLint *params);
        //typedef GLboolean (*PFNGLISFRAMEBUFFERPROC)(GLuint framebuffer);
        //typedef GLenum (*PFNGLCHECKFRAMEBUFFERSTATUSPROC)(GLenum target);
        //typedef void (*PFNGLFRAMEBUFFERTEXTURE1DPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
        //typedef void (*PFNGLFRAMEBUFFERTEXTURE3DPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset);
        //typedef void (*PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC)(GLenum target, GLenum attachment, GLenum pname, GLint *params);
        //typedef void (*PFNGLGENERATEMIPMAPPROC)(GLenum target);
        //typedef void (*PFNGLFRAMEBUFFERTEXTURELAYERPROC)(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer);
        //typedef void *(*PFNGLMAPBUFFERRANGEPROC)(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
        //typedef void (*PFNGLFLUSHMAPPEDBUFFERRANGEPROC)(GLenum target, GLintptr offset, GLsizeiptr length);
        //typedef GLboolean (*PFNGLISVERTEXARRAYPROC)(GLuint array);
    #endif /* GL_VERSION_3_0 */

    #ifndef GL_VERSION_3_1
        #define GL_VERSION_3_1 1
        #define GL_SAMPLER_2D_RECT                0x8B63
        #define GL_SAMPLER_2D_RECT_SHADOW         0x8B64
        #define GL_SAMPLER_BUFFER                 0x8DC2
        #define GL_INT_SAMPLER_2D_RECT            0x8DCD
        #define GL_INT_SAMPLER_BUFFER             0x8DD0
        #define GL_UNSIGNED_INT_SAMPLER_2D_RECT   0x8DD5
        #define GL_UNSIGNED_INT_SAMPLER_BUFFER    0x8DD8
        #define GL_TEXTURE_BUFFER                 0x8C2A
        #define GL_MAX_TEXTURE_BUFFER_SIZE        0x8C2B
        #define GL_TEXTURE_BINDING_BUFFER         0x8C2C
        #define GL_TEXTURE_BUFFER_DATA_STORE_BINDING 0x8C2D
        #define GL_TEXTURE_RECTANGLE              0x84F5
        #define GL_TEXTURE_BINDING_RECTANGLE      0x84F6
        #define GL_PROXY_TEXTURE_RECTANGLE        0x84F7
        #define GL_MAX_RECTANGLE_TEXTURE_SIZE     0x84F8
        #define GL_R8_SNORM                       0x8F94
        #define GL_RG8_SNORM                      0x8F95
        #define GL_RGB8_SNORM                     0x8F96
        #define GL_RGBA8_SNORM                    0x8F97
        #define GL_R16_SNORM                      0x8F98
        #define GL_RG16_SNORM                     0x8F99
        #define GL_RGB16_SNORM                    0x8F9A
        #define GL_RGBA16_SNORM                   0x8F9B
        #define GL_SIGNED_NORMALIZED              0x8F9C
        #define GL_PRIMITIVE_RESTART              0x8F9D
        #define GL_PRIMITIVE_RESTART_INDEX        0x8F9E
        #define GL_COPY_READ_BUFFER               0x8F36
        #define GL_COPY_WRITE_BUFFER              0x8F37
        #define GL_UNIFORM_BUFFER                 0x8A11
        #define GL_UNIFORM_BUFFER_BINDING         0x8A28
        #define GL_UNIFORM_BUFFER_START           0x8A29
        #define GL_UNIFORM_BUFFER_SIZE            0x8A2A
        #define GL_MAX_VERTEX_UNIFORM_BLOCKS      0x8A2B
        #define GL_MAX_GEOMETRY_UNIFORM_BLOCKS    0x8A2C
        #define GL_MAX_FRAGMENT_UNIFORM_BLOCKS    0x8A2D
        #define GL_MAX_COMBINED_UNIFORM_BLOCKS    0x8A2E
        #define GL_MAX_UNIFORM_BUFFER_BINDINGS    0x8A2F
        #define GL_MAX_UNIFORM_BLOCK_SIZE         0x8A30
        #define GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS 0x8A31
        #define GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS 0x8A32
        #define GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS 0x8A33
        #define GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT 0x8A34
        #define GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH 0x8A35
        #define GL_ACTIVE_UNIFORM_BLOCKS          0x8A36
        #define GL_UNIFORM_TYPE                   0x8A37
        #define GL_UNIFORM_SIZE                   0x8A38
        #define GL_UNIFORM_NAME_LENGTH            0x8A39
        #define GL_UNIFORM_BLOCK_INDEX            0x8A3A
        #define GL_UNIFORM_OFFSET                 0x8A3B
        #define GL_UNIFORM_ARRAY_STRIDE           0x8A3C
        #define GL_UNIFORM_MATRIX_STRIDE          0x8A3D
        #define GL_UNIFORM_IS_ROW_MAJOR           0x8A3E
        #define GL_UNIFORM_BLOCK_BINDING          0x8A3F
        #define GL_UNIFORM_BLOCK_DATA_SIZE        0x8A40
        #define GL_UNIFORM_BLOCK_NAME_LENGTH      0x8A41
        #define GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS  0x8A42
        #define GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES 0x8A43
        #define GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER 0x8A44
        #define GL_UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER 0x8A45
        #define GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER 0x8A46
        #define GL_INVALID_INDEX                  0xFFFFFFFFu
        typedef GLuint (*PFNGLGETUNIFORMBLOCKINDEXPROC)(GLuint program, const GLchar *uniformBlockName);
        typedef void (*PFNGLUNIFORMBLOCKBINDINGPROC)(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
        //typedef void (*PFNGLDRAWARRAYSINSTANCEDPROC)(GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
        //typedef void (*PFNGLDRAWELEMENTSINSTANCEDPROC)(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount);
        //typedef void (*PFNGLTEXBUFFERPROC)(GLenum target, GLenum internalformat, GLuint buffer);
        //typedef void (*PFNGLPRIMITIVERESTARTINDEXPROC)(GLuint index);
        //typedef void (*PFNGLCOPYBUFFERSUBDATAPROC)(GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size);
        //typedef void (*PFNGLGETUNIFORMINDICESPROC)(GLuint program, GLsizei uniformCount, const GLchar *const*uniformNames, GLuint *uniformIndices);
        //typedef void (*PFNGLGETACTIVEUNIFORMSIVPROC)(GLuint program, GLsizei uniformCount, const GLuint *uniformIndices, GLenum pname, GLint *params);
        //typedef void (*PFNGLGETACTIVEUNIFORMNAMEPROC)(GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformName);
        //typedef void (*PFNGLGETACTIVEUNIFORMBLOCKIVPROC)(GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params);
        //typedef void (*PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC)(GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformBlockName);
    #endif /* GL_VERSION_3_1 */

    #if defined(_WIN32) && !defined(__CYGWIN__) && defined(THORVG_GL_TARGET_GL)
        typedef HGLRC (WINAPI *PFNWGLGETCURRENTCONTEXTPROC)(void);
        typedef BOOL (WINAPI *PFNWGLMAKECURRENTPROC)(HDC, HGLRC);
    #endif

    #if defined(THORVG_GL_TARGET_GLES)
        typedef void* EGLDisplay;
        typedef void* EGLSurface;
        typedef void* EGLContext;
        typedef EGLContext (*PFNEGLGETCURRENTCONTEXTPROC)(void);
        typedef unsigned int (*PFNEGLMAKECURRENTPROC)(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx);
    #endif

    //GL_VERSION_1_0
    extern PFNGLCULLFACEPROC               glCullFace;
    extern PFNGLFRONTFACEPROC              glFrontFace;
    extern PFNGLSCISSORPROC                glScissor;
    extern PFNGLTEXPARAMETERIPROC          glTexParameteri;
    extern PFNGLTEXIMAGE2DPROC             glTexImage2D;
    extern PFNGLDRAWBUFFERPROC             glDrawBuffer;
    extern PFNGLDRAWBUFFERSPROC            glDrawBuffers;
    extern PFNGLCLEARPROC                  glClear;
    extern PFNGLCLEARCOLORPROC             glClearColor;
    extern PFNGLCLEARSTENCILPROC           glClearStencil;
    extern PFNGLCLEARDEPTHPROC             glClearDepth;
    extern PFNGLCLEARDEPTHFPROC            glClearDepthf; // GLES
    extern PFNGLCOLORMASKPROC              glColorMask;
    extern PFNGLDEPTHMASKPROC              glDepthMask;
    extern PFNGLDISABLEPROC                glDisable;
    extern PFNGLENABLEPROC                 glEnable;
    extern PFNGLBLENDFUNCPROC              glBlendFunc;
    extern PFNGLSTENCILFUNCPROC            glStencilFunc;
    extern PFNGLSTENCILOPPROC              glStencilOp;
    extern PFNGLDEPTHFUNCPROC              glDepthFunc;
    extern PFNGLGETERRORPROC               glGetError;
    extern PFNGLGETINTEGERVPROC            glGetIntegerv;
    extern PFNGLGETSTRINGPROC              glGetString;
    extern PFNGLVIEWPORTPROC               glViewport;
    //extern PFNGLHINTPROC                   glHint;
    //extern PFNGLLINEWIDTHPROC              glLineWidth;
    //extern PFNGLPOINTSIZEPROC              glPointSize;
    //extern PFNGLPOLYGONMODEPROC            glPolygonMode;
    //extern PFNGLTEXPARAMETERFPROC          glTexParameterf;
    //extern PFNGLTEXPARAMETERFVPROC         glTexParameterfv;
    //extern PFNGLTEXPARAMETERIVPROC         glTexParameteriv;
    //extern PFNGLTEXIMAGE1DPROC             glTexImage1D;
    //extern PFNGLSTENCILMASKPROC            glStencilMask;
    //extern PFNGLFINISHPROC                 glFinish;
    //extern PFNGLFLUSHPROC                  glFlush;
    //extern PFNGLLOGICOPPROC                glLogicOp;
    //extern PFNGLPIXELSTOREFPROC            glPixelStoref;
    //extern PFNGLPIXELSTOREIPROC            glPixelStorei;
    //extern PFNGLREADBUFFERPROC             glReadBuffer;
    //extern PFNGLREADPIXELSPROC             glReadPixels;
    //extern PFNGLGETBOOLEANVPROC            glGetBooleanv;
    //extern PFNGLGETDOUBLEVPROC             glGetDoublev;
    //extern PFNGLGETFLOATVPROC              glGetFloatv;
    //extern PFNGLGETTEXIMAGEPROC            glGetTexImage;
    //extern PFNGLGETTEXPARAMETERFVPROC      glGetTexParameterfv;
    //extern PFNGLGETTEXPARAMETERIVPROC      glGetTexParameteriv;
    //extern PFNGLGETTEXLEVELPARAMETERFVPROC glGetTexLevelParameterfv;
    //extern PFNGLGETTEXLEVELPARAMETERIVPROC glGetTexLevelParameteriv;
    //extern PFNGLISENABLEDPROC              glIsEnabled;
    //extern PFNGLDEPTHRANGEPROC             glDepthRange;

    //GL_VERSION_1_1
    extern PFNGLDRAWELEMENTSPROC      glDrawElements;
    extern PFNGLBINDTEXTUREPROC       glBindTexture;
    extern PFNGLDELETETEXTURESPROC    glDeleteTextures;
    extern PFNGLGENTEXTURESPROC       glGenTextures;
    //extern PFNGLDRAWARRAYSPROC        glDrawArrays;
    //extern PFNGLGETPOINTERVPROC       glGetPointerv;
    //extern PFNGLPOLYGONOFFSETPROC     glPolygonOffset;
    //extern PFNGLCOPYTEXIMAGE1DPROC    glCopyTexImage1D;
    //extern PFNGLCOPYTEXIMAGE2DPROC    glCopyTexImage2D;
    //extern PFNGLCOPYTEXSUBIMAGE1DPROC glCopyTexSubImage1D;
    //extern PFNGLCOPYTEXSUBIMAGE2DPROC glCopyTexSubImage2D;
    //extern PFNGLTEXSUBIMAGE1DPROC     glTexSubImage1D;
    //extern PFNGLTEXSUBIMAGE2DPROC     glTexSubImage2D;
    //extern PFNGLISTEXTUREPROC         glIsTexture;

    //GL_VERSION_1_2
    //extern PFNGLDRAWRANGEELEMENTSPROC glDrawRangeElements;
    //extern PFNGLTEXIMAGE3DPROC        glTexImage3D;
    //extern PFNGLTEXSUBIMAGE3DPROC     glTexSubImage3D;
    //extern PFNGLCOPYTEXSUBIMAGE3DPROC glCopyTexSubImage3D;

    //GL_VERSION_1_3
    extern PFNGLACTIVETEXTUREPROC           glActiveTexture;
    //extern PFNGLSAMPLECOVERAGEPROC          glSampleCoverage;
    //extern PFNGLCOMPRESSEDTEXIMAGE3DPROC    glCompressedTexImage3D;
    //extern PFNGLCOMPRESSEDTEXIMAGE2DPROC    glCompressedTexImage2D;
    //extern PFNGLCOMPRESSEDTEXIMAGE1DPROC    glCompressedTexImage1D;
    //extern PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC glCompressedTexSubImage3D;
    //extern PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC glCompressedTexSubImage2D;
    //extern PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC glCompressedTexSubImage1D;
    //extern PFNGLGETCOMPRESSEDTEXIMAGEPROC   glGetCompressedTexImage;

    //GL_VERSION_1_4
    extern PFNGLBLENDEQUATIONPROC     glBlendEquation;
    //extern PFNGLBLENDFUNCSEPARATEPROC glBlendFuncSeparate;
    //extern PFNGLMULTIDRAWARRAYSPROC   glMultiDrawArrays;
    //extern PFNGLMULTIDRAWELEMENTSPROC glMultiDrawElements;
    //extern PFNGLPOINTPARAMETERFPROC   glPointParameterf;
    //extern PFNGLPOINTPARAMETERFVPROC  glPointParameterfv;
    //extern PFNGLPOINTPARAMETERIPROC   glPointParameteri;
    //extern PFNGLPOINTPARAMETERIVPROC  glPointParameteriv;
    //extern PFNGLBLENDCOLORPROC        glBlendColor;

    //GL_VERSION_1_5
    extern PFNGLBINDBUFFERPROC           glBindBuffer;
    extern PFNGLDELETEBUFFERSPROC        glDeleteBuffers;
    extern PFNGLGENBUFFERSPROC           glGenBuffers;
    extern PFNGLBUFFERDATAPROC           glBufferData;
    //extern PFNGLGENQUERIESPROC           glGenQueries;
    //extern PFNGLDELETEQUERIESPROC        glDeleteQueries;
    //extern PFNGLISQUERYPROC              glIsQuery;
    //extern PFNGLBEGINQUERYPROC           glBeginQuery;
    //extern PFNGLENDQUERYPROC             glEndQuery;
    //extern PFNGLGETQUERYIVPROC           glGetQueryiv;
    //extern PFNGLGETQUERYOBJECTIVPROC     glGetQueryObjectiv;
    //extern PFNGLGETQUERYOBJECTUIVPROC    glGetQueryObjectuiv;
    //extern PFNGLISBUFFERPROC             glIsBuffer;
    //extern PFNGLBUFFERSUBDATAPROC        glBufferSubData;
    //extern PFNGLGETBUFFERSUBDATAPROC     glGetBufferSubData;
    //extern PFNGLMAPBUFFERPROC            glMapBuffer;
    //extern PFNGLUNMAPBUFFERPROC          glUnmapBuffer;
    //extern PFNGLGETBUFFERPARAMETERIVPROC glGetBufferParameteriv;
    //extern PFNGLGETBUFFERPOINTERVPROC    glGetBufferPointerv;

    // GL_VERSION_2_0
    extern PFNGLSTENCILOPSEPARATEPROC        glStencilOpSeparate;
    extern PFNGLSTENCILFUNCSEPARATEPROC      glStencilFuncSeparate;
    extern PFNGLATTACHSHADERPROC             glAttachShader;
    extern PFNGLCOMPILESHADERPROC            glCompileShader;
    extern PFNGLCREATEPROGRAMPROC            glCreateProgram;
    extern PFNGLCREATESHADERPROC             glCreateShader;
    extern PFNGLDELETEPROGRAMPROC            glDeleteProgram;
    extern PFNGLDELETESHADERPROC             glDeleteShader;
    extern PFNGLDISABLEVERTEXATTRIBARRAYPROC glDisableVertexAttribArray;
    extern PFNGLENABLEVERTEXATTRIBARRAYPROC  glEnableVertexAttribArray;
    extern PFNGLGETATTRIBLOCATIONPROC        glGetAttribLocation;
    extern PFNGLGETPROGRAMIVPROC             glGetProgramiv;
    extern PFNGLGETPROGRAMINFOLOGPROC        glGetProgramInfoLog;
    extern PFNGLGETSHADERIVPROC              glGetShaderiv;
    extern PFNGLGETSHADERINFOLOGPROC         glGetShaderInfoLog;
    extern PFNGLGETUNIFORMLOCATIONPROC       glGetUniformLocation;
    extern PFNGLLINKPROGRAMPROC              glLinkProgram;
    extern PFNGLSHADERSOURCEPROC             glShaderSource;
    extern PFNGLUSEPROGRAMPROC               glUseProgram;
    extern PFNGLUNIFORM1FPROC                glUniform1f;
    extern PFNGLUNIFORM1FVPROC               glUniform1fv;
    extern PFNGLUNIFORM2FVPROC               glUniform2fv;
    extern PFNGLUNIFORM3FVPROC               glUniform3fv;
    extern PFNGLUNIFORM4FVPROC               glUniform4fv;
    extern PFNGLUNIFORM1IVPROC               glUniform1iv;
    extern PFNGLUNIFORM2IVPROC               glUniform2iv;
    extern PFNGLUNIFORM3IVPROC               glUniform3iv;
    extern PFNGLUNIFORM4IVPROC               glUniform4iv;
    // extern PFNGLUNIFORMMATRIX4FVPROC         glUniformMatrix4fv;
    extern PFNGLVERTEXATTRIBPOINTERPROC      glVertexAttribPointer;
    //extern PFNGLBLENDEQUATIONSEPARATEPROC    glBlendEquationSeparate;
    //extern PFNGLDRAWBUFFERSPROC              glDrawBuffers;
    //extern PFNGLSTENCILMASKSEPARATEPROC      glStencilMaskSeparate;
    //extern PFNGLBINDATTRIBLOCATIONPROC       glBindAttribLocation;
    //extern PFNGLDETACHSHADERPROC             glDetachShader;
    //extern PFNGLGETACTIVEATTRIBPROC          glGetActiveAttrib;
    //extern PFNGLGETACTIVEUNIFORMPROC         glGetActiveUniform;
    //extern PFNGLGETATTACHEDSHADERSPROC       glGetAttachedShaders;
    //extern PFNGLGETSHADERSOURCEPROC          glGetShaderSource;
    //extern PFNGLGETUNIFORMFVPROC             glGetUniformfv;
    //extern PFNGLGETUNIFORMIVPROC             glGetUniformiv;
    //extern PFNGLGETVERTEXATTRIBDVPROC        glGetVertexAttribdv;
    //extern PFNGLGETVERTEXATTRIBFVPROC        glGetVertexAttribfv;
    //extern PFNGLGETVERTEXATTRIBIVPROC        glGetVertexAttribiv;
    //extern PFNGLGETVERTEXATTRIBPOINTERVPROC  glGetVertexAttribPointerv;
    //extern PFNGLISPROGRAMPROC                glIsProgram;
    //extern PFNGLISSHADERPROC                 glIsShader;
    //extern PFNGLUNIFORM2FPROC                glUniform2f;
    //extern PFNGLUNIFORM3FPROC                glUniform3f;
    //extern PFNGLUNIFORM4FPROC                glUniform4f;
    //extern PFNGLUNIFORM1IPROC                glUniform1i;
    //extern PFNGLUNIFORM2IPROC                glUniform2i;
    //extern PFNGLUNIFORM3IPROC                glUniform3i;
    //extern PFNGLUNIFORM4IPROC                glUniform4i;
    //extern PFNGLUNIFORMMATRIX2FVPROC         glUniformMatrix2fv;
    extern PFNGLUNIFORMMATRIX3FVPROC          glUniformMatrix3fv;
    //extern PFNGLVALIDATEPROGRAMPROC          glValidateProgram;
    //extern PFNGLVERTEXATTRIB1DPROC           glVertexAttrib1d;
    //extern PFNGLVERTEXATTRIB1DVPROC          glVertexAttrib1dv;
    //extern PFNGLVERTEXATTRIB1FPROC           glVertexAttrib1f;
    //extern PFNGLVERTEXATTRIB1FVPROC          glVertexAttrib1fv;
    //extern PFNGLVERTEXATTRIB1SPROC           glVertexAttrib1s;
    //extern PFNGLVERTEXATTRIB1SVPROC          glVertexAttrib1sv;
    //extern PFNGLVERTEXATTRIB2DPROC           glVertexAttrib2d;
    //extern PFNGLVERTEXATTRIB2DVPROC          glVertexAttrib2dv;
    //extern PFNGLVERTEXATTRIB2FPROC           glVertexAttrib2f;
    //extern PFNGLVERTEXATTRIB2FVPROC          glVertexAttrib2fv;
    //extern PFNGLVERTEXATTRIB2SPROC           glVertexAttrib2s;
    //extern PFNGLVERTEXATTRIB2SVPROC          glVertexAttrib2sv;
    //extern PFNGLVERTEXATTRIB3DPROC           glVertexAttrib3d;
    //extern PFNGLVERTEXATTRIB3DVPROC          glVertexAttrib3dv;
    //extern PFNGLVERTEXATTRIB3FPROC           glVertexAttrib3f;
    //extern PFNGLVERTEXATTRIB3FVPROC          glVertexAttrib3fv;
    //extern PFNGLVERTEXATTRIB3SPROC           glVertexAttrib3s;
    //extern PFNGLVERTEXATTRIB3SVPROC          glVertexAttrib3sv;
    //extern PFNGLVERTEXATTRIB4NBVPROC         glVertexAttrib4Nbv;
    //extern PFNGLVERTEXATTRIB4NIVPROC         glVertexAttrib4Niv;
    //extern PFNGLVERTEXATTRIB4NSVPROC         glVertexAttrib4Nsv;
    //extern PFNGLVERTEXATTRIB4NUBPROC         glVertexAttrib4Nub;
    //extern PFNGLVERTEXATTRIB4NUBVPROC        glVertexAttrib4Nubv;
    //extern PFNGLVERTEXATTRIB4NUIVPROC        glVertexAttrib4Nuiv;
    //extern PFNGLVERTEXATTRIB4NUSVPROC        glVertexAttrib4Nusv;
    //extern PFNGLVERTEXATTRIB4BVPROC          glVertexAttrib4bv;
    //extern PFNGLVERTEXATTRIB4DPROC           glVertexAttrib4d;
    //extern PFNGLVERTEXATTRIB4DVPROC          glVertexAttrib4dv;
    extern PFNGLVERTEXATTRIB4FPROC             glVertexAttrib4f;
    //extern PFNGLVERTEXATTRIB4FVPROC          glVertexAttrib4fv;
    //extern PFNGLVERTEXATTRIB4IVPROC          glVertexAttrib4iv;
    //extern PFNGLVERTEXATTRIB4SPROC           glVertexAttrib4s;
    //extern PFNGLVERTEXATTRIB4SVPROC          glVertexAttrib4sv;
    //extern PFNGLVERTEXATTRIB4UBVPROC         glVertexAttrib4ubv;
    //extern PFNGLVERTEXATTRIB4UIVPROC         glVertexAttrib4uiv;
    //extern PFNGLVERTEXATTRIB4USVPROC         glVertexAttrib4usv;

    //GL_VERSION_2_1
    //extern PFNGLUNIFORMMATRIX2X3FVPROC glUniformMatrix2x3fv;
    //extern PFNGLUNIFORMMATRIX3X2FVPROC glUniformMatrix3x2fv;
    //extern PFNGLUNIFORMMATRIX2X4FVPROC glUniformMatrix2x4fv;
    //extern PFNGLUNIFORMMATRIX4X2FVPROC glUniformMatrix4x2fv;
    //extern PFNGLUNIFORMMATRIX3X4FVPROC glUniformMatrix3x4fv;
    //extern PFNGLUNIFORMMATRIX4X3FVPROC glUniformMatrix4x3fv;

    //GL_VERSION_3_0
    extern PFNGLBINDBUFFERRANGEPROC                     glBindBufferRange;
    extern PFNGLBINDRENDERBUFFERPROC                    glBindRenderbuffer;
    extern PFNGLDELETERENDERBUFFERSPROC                 glDeleteRenderbuffers;
    extern PFNGLGENRENDERBUFFERSPROC                    glGenRenderbuffers;
    extern PFNGLINVALIDATEFRAMEBUFFERPROC               glInvalidateFramebuffer;
    extern PFNGLBINDFRAMEBUFFERPROC                     glBindFramebuffer;
    extern PFNGLDELETEFRAMEBUFFERSPROC                  glDeleteFramebuffers;
    extern PFNGLGENFRAMEBUFFERSPROC                     glGenFramebuffers;
    extern PFNGLFRAMEBUFFERTEXTURE2DPROC                glFramebufferTexture2D;
    extern PFNGLFRAMEBUFFERRENDERBUFFERPROC             glFramebufferRenderbuffer;
    extern PFNGLBLITFRAMEBUFFERPROC                     glBlitFramebuffer;
    extern PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC      glRenderbufferStorageMultisample;
    extern PFNGLBINDVERTEXARRAYPROC                     glBindVertexArray;
    extern PFNGLDELETEVERTEXARRAYSPROC                  glDeleteVertexArrays;
    extern PFNGLGENVERTEXARRAYSPROC                     glGenVertexArrays;
    //extern PFNGLCOLORMASKIPROC                          glColorMaski;
    //extern PFNGLGETBOOLEANI_VPROC                       glGetBooleani_v;
    //extern PFNGLGETINTEGERI_VPROC                       glGetIntegeri_v;
    //extern PFNGLENABLEIPROC                             glEnablei;
    //extern PFNGLDISABLEIPROC                            glDisablei;
    //extern PFNGLISENABLEDIPROC                          glIsEnabledi;
    //extern PFNGLBEGINTRANSFORMFEEDBACKPROC              glBeginTransformFeedback;
    //extern PFNGLENDTRANSFORMFEEDBACKPROC                glEndTransformFeedback;
    //extern PFNGLBINDBUFFERBASEPROC                      glBindBufferBase;
    //extern PFNGLTRANSFORMFEEDBACKVARYINGSPROC           glTransformFeedbackVaryings;
    //extern PFNGLGETTRANSFORMFEEDBACKVARYINGPROC         glGetTransformFeedbackVarying;
    //extern PFNGLCLAMPCOLORPROC                          glClampColor;
    //extern PFNGLBEGINCONDITIONALRENDERPROC              glBeginConditionalRender;
    //extern PFNGLENDCONDITIONALRENDERPROC                glEndConditionalRender;
    //extern PFNGLVERTEXATTRIBIPOINTERPROC                glVertexAttribIPointer;
    //extern PFNGLGETVERTEXATTRIBIIVPROC                  glGetVertexAttribIiv;
    //extern PFNGLGETVERTEXATTRIBIUIVPROC                 glGetVertexAttribIuiv;
    //extern PFNGLVERTEXATTRIBI1IPROC                     glVertexAttribI1i;
    //extern PFNGLVERTEXATTRIBI2IPROC                     glVertexAttribI2i;
    //extern PFNGLVERTEXATTRIBI3IPROC                     glVertexAttribI3i;
    //extern PFNGLVERTEXATTRIBI4IPROC                     glVertexAttribI4i;
    //extern PFNGLVERTEXATTRIBI1UIPROC                    glVertexAttribI1ui;
    //extern PFNGLVERTEXATTRIBI2UIPROC                    glVertexAttribI2ui;
    //extern PFNGLVERTEXATTRIBI3UIPROC                    glVertexAttribI3ui;
    //extern PFNGLVERTEXATTRIBI4UIPROC                    glVertexAttribI4ui;
    //extern PFNGLVERTEXATTRIBI1IVPROC                    glVertexAttribI1iv;
    //extern PFNGLVERTEXATTRIBI2IVPROC                    glVertexAttribI2iv;
    //extern PFNGLVERTEXATTRIBI3IVPROC                    glVertexAttribI3iv;
    //extern PFNGLVERTEXATTRIBI4IVPROC                    glVertexAttribI4iv;
    //extern PFNGLVERTEXATTRIBI1UIVPROC                   glVertexAttribI1uiv;
    //extern PFNGLVERTEXATTRIBI2UIVPROC                   glVertexAttribI2uiv;
    //extern PFNGLVERTEXATTRIBI3UIVPROC                   glVertexAttribI3uiv;
    //extern PFNGLVERTEXATTRIBI4UIVPROC                   glVertexAttribI4uiv;
    //extern PFNGLVERTEXATTRIBI4BVPROC                    glVertexAttribI4bv;
    //extern PFNGLVERTEXATTRIBI4SVPROC                    glVertexAttribI4sv;
    //extern PFNGLVERTEXATTRIBI4UBVPROC                   glVertexAttribI4ubv;
    //extern PFNGLVERTEXATTRIBI4USVPROC                   glVertexAttribI4usv;
    //extern PFNGLGETUNIFORMUIVPROC                       glGetUniformuiv;
    //extern PFNGLBINDFRAGDATALOCATIONPROC                glBindFragDataLocation;
    //extern PFNGLGETFRAGDATALOCATIONPROC                 glGetFragDataLocation;
    //extern PFNGLUNIFORM1UIPROC                          glUniform1ui;
    //extern PFNGLUNIFORM2UIPROC                          glUniform2ui;
    //extern PFNGLUNIFORM3UIPROC                          glUniform3ui;
    //extern PFNGLUNIFORM4UIPROC                          glUniform4ui;
    //extern PFNGLUNIFORM1UIVPROC                         glUniform1uiv;
    //extern PFNGLUNIFORM2UIVPROC                         glUniform2uiv;
    //extern PFNGLUNIFORM3UIVPROC                         glUniform3uiv;
    //extern PFNGLUNIFORM4UIVPROC                         glUniform4uiv;
    //extern PFNGLTEXPARAMETERIIVPROC                     glTexParameterIiv;
    //extern PFNGLTEXPARAMETERIUIVPROC                    glTexParameterIuiv;
    //extern PFNGLGETTEXPARAMETERIIVPROC                  glGetTexParameterIiv;
    //extern PFNGLGETTEXPARAMETERIUIVPROC                 glGetTexParameterIuiv;
    //extern PFNGLCLEARBUFFERIVPROC                       glClearBufferiv;
    //extern PFNGLCLEARBUFFERUIVPROC                      glClearBufferuiv;
    //extern PFNGLCLEARBUFFERFVPROC                       glClearBufferfv;
    //extern PFNGLCLEARBUFFERFIPROC                       glClearBufferfi;
    //extern PFNGLGETSTRINGIPROC                          glGetStringi;
    //extern PFNGLISRENDERBUFFERPROC                      glIsRenderbuffer;
    //extern PFNGLRENDERBUFFERSTORAGEPROC                 glRenderbufferStorage;
    //extern PFNGLGETRENDERBUFFERPARAMETERIVPROC          glGetRenderbufferParameteriv;
    //extern PFNGLISFRAMEBUFFERPROC                       glIsFramebuffer;
    //extern PFNGLCHECKFRAMEBUFFERSTATUSPROC              glCheckFramebufferStatus;
    //extern PFNGLFRAMEBUFFERTEXTURE1DPROC                glFramebufferTexture1D;
    //extern PFNGLFRAMEBUFFERTEXTURE3DPROC                glFramebufferTexture3D;
    //extern PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC glGetFramebufferAttachmentParameteriv;
    //extern PFNGLGENERATEMIPMAPPROC                      glGenerateMipmap;
    //extern PFNGLFRAMEBUFFERTEXTURELAYERPROC             glFramebufferTextureLayer;
    //extern PFNGLMAPBUFFERRANGEPROC                      glMapBufferRange;
    //extern PFNGLFLUSHMAPPEDBUFFERRANGEPROC              glFlushMappedBufferRange;
    //extern PFNGLISVERTEXARRAYPROC                       glIsVertexArray;

    //GL_VERSION_3_1
    extern PFNGLGETUNIFORMBLOCKINDEXPROC      glGetUniformBlockIndex;
    extern PFNGLUNIFORMBLOCKBINDINGPROC       glUniformBlockBinding;
    //extern PFNGLDRAWARRAYSINSTANCEDPROC       glDrawArraysInstanced;
    //extern PFNGLDRAWELEMENTSINSTANCEDPROC     glDrawElementsInstanced;
    //extern PFNGLTEXBUFFERPROC                 glTexBuffer;
    //extern PFNGLPRIMITIVERESTARTINDEXPROC     glPrimitiveRestartIndex;
    //extern PFNGLCOPYBUFFERSUBDATAPROC         glCopyBufferSubData;
    //extern PFNGLGETUNIFORMINDICESPROC         glGetUniformIndices;
    //extern PFNGLGETACTIVEUNIFORMSIVPROC       glGetActiveUniformsiv;
    //extern PFNGLGETACTIVEUNIFORMNAMEPROC      glGetActiveUniformName;
    //extern PFNGLGETACTIVEUNIFORMBLOCKIVPROC   glGetActiveUniformBlockiv;
    //extern PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC glGetActiveUniformBlockName;

    #if defined(_WIN32) && !defined(__CYGWIN__) && defined(THORVG_GL_TARGET_GL)
        extern PFNWGLGETCURRENTCONTEXTPROC  tvgWglGetCurrentContext;
        extern PFNWGLMAKECURRENTPROC        tvgWglMakeCurrent;
    #endif

    #if defined(THORVG_GL_TARGET_GLES)
        extern PFNEGLGETCURRENTCONTEXTPROC  tvgEglGetCurrentContext;
        extern PFNEGLMAKECURRENTPROC        tvgEglMakeCurrent;
    #endif

#endif // __EMSCRIPTEN__

bool glInit();
bool glTerm();

#endif // _TVG_GL_H_
