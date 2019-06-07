/*
** Copyright (c) 2013 The Khronos Group Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and/or associated documentation files (the
** "Materials"), to deal in the Materials without restriction, including
** without limitation the rights to use, copy, modify, merge, publish,
** distribute, sublicense, and/or sell copies of the Materials, and to
** permit persons to whom the Materials are furnished to do so, subject to
** the following conditions:
**
** The above copyright notice and this permission notice shall be included
** in all copies or substantial portions of the Materials.
**
** THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
** MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
** IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
** CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
** TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
** MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
*/

#pragma once

#define GL_FLOAT                          0x1406
#define GL_FLOAT_VEC2                     0x8B50
#define GL_FLOAT_VEC3                     0x8B51
#define GL_FLOAT_VEC4                     0x8B52

#define GL_DOUBLE                         0x140A
#define GL_DOUBLE_VEC2                    0x8FFC
#define GL_DOUBLE_VEC3                    0x8FFD
#define GL_DOUBLE_VEC4                    0x8FFE

#define GL_INT                            0x1404
#define GL_INT_VEC2                       0x8B53
#define GL_INT_VEC3                       0x8B54
#define GL_INT_VEC4                       0x8B55

#define GL_UNSIGNED_INT                   0x1405
#define GL_UNSIGNED_INT_VEC2              0x8DC6
#define GL_UNSIGNED_INT_VEC3              0x8DC7
#define GL_UNSIGNED_INT_VEC4              0x8DC8

#define GL_INT64_ARB                      0x140E
#define GL_INT64_VEC2_ARB                 0x8FE9
#define GL_INT64_VEC3_ARB                 0x8FEA
#define GL_INT64_VEC4_ARB                 0x8FEB

#define GL_UNSIGNED_INT64_ARB             0x140F
#define GL_UNSIGNED_INT64_VEC2_ARB        0x8FE5
#define GL_UNSIGNED_INT64_VEC3_ARB        0x8FE6
#define GL_UNSIGNED_INT64_VEC4_ARB        0x8FE7

#define GL_BOOL                           0x8B56
#define GL_BOOL_VEC2                      0x8B57
#define GL_BOOL_VEC3                      0x8B58
#define GL_BOOL_VEC4                      0x8B59

#define GL_FLOAT_MAT2                     0x8B5A
#define GL_FLOAT_MAT3                     0x8B5B
#define GL_FLOAT_MAT4                     0x8B5C
#define GL_FLOAT_MAT2x3                   0x8B65
#define GL_FLOAT_MAT2x4                   0x8B66
#define GL_FLOAT_MAT3x2                   0x8B67
#define GL_FLOAT_MAT3x4                   0x8B68
#define GL_FLOAT_MAT4x2                   0x8B69
#define GL_FLOAT_MAT4x3                   0x8B6A

#define GL_DOUBLE_MAT2                    0x8F46
#define GL_DOUBLE_MAT3                    0x8F47
#define GL_DOUBLE_MAT4                    0x8F48
#define GL_DOUBLE_MAT2x3                  0x8F49
#define GL_DOUBLE_MAT2x4                  0x8F4A
#define GL_DOUBLE_MAT3x2                  0x8F4B
#define GL_DOUBLE_MAT3x4                  0x8F4C
#define GL_DOUBLE_MAT4x2                  0x8F4D
#define GL_DOUBLE_MAT4x3                  0x8F4E

#ifdef AMD_EXTENSIONS
// Those constants are borrowed from extension NV_gpu_shader5
#define GL_FLOAT16_NV                     0x8FF8
#define GL_FLOAT16_VEC2_NV                0x8FF9
#define GL_FLOAT16_VEC3_NV                0x8FFA
#define GL_FLOAT16_VEC4_NV                0x8FFB

#define GL_FLOAT16_MAT2_AMD               0x91C5
#define GL_FLOAT16_MAT3_AMD               0x91C6
#define GL_FLOAT16_MAT4_AMD               0x91C7
#define GL_FLOAT16_MAT2x3_AMD             0x91C8
#define GL_FLOAT16_MAT2x4_AMD             0x91C9
#define GL_FLOAT16_MAT3x2_AMD             0x91CA
#define GL_FLOAT16_MAT3x4_AMD             0x91CB
#define GL_FLOAT16_MAT4x2_AMD             0x91CC
#define GL_FLOAT16_MAT4x3_AMD             0x91CD
#endif

#define GL_SAMPLER_1D                     0x8B5D
#define GL_SAMPLER_2D                     0x8B5E
#define GL_SAMPLER_3D                     0x8B5F
#define GL_SAMPLER_CUBE                   0x8B60
#define GL_SAMPLER_BUFFER                 0x8DC2
#define GL_SAMPLER_1D_ARRAY               0x8DC0
#define GL_SAMPLER_2D_ARRAY               0x8DC1
#define GL_SAMPLER_1D_ARRAY_SHADOW        0x8DC3
#define GL_SAMPLER_2D_ARRAY_SHADOW        0x8DC4
#define GL_SAMPLER_CUBE_SHADOW            0x8DC5
#define GL_SAMPLER_1D_SHADOW              0x8B61
#define GL_SAMPLER_2D_SHADOW              0x8B62
#define GL_SAMPLER_2D_RECT                0x8B63
#define GL_SAMPLER_2D_RECT_SHADOW         0x8B64
#define GL_SAMPLER_2D_MULTISAMPLE         0x9108
#define GL_SAMPLER_2D_MULTISAMPLE_ARRAY   0x910B
#define GL_SAMPLER_CUBE_MAP_ARRAY         0x900C
#define GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW  0x900D
#define GL_SAMPLER_CUBE_MAP_ARRAY_ARB     0x900C
#define GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW_ARB 0x900D

#ifdef AMD_EXTENSIONS
#define GL_FLOAT16_SAMPLER_1D_AMD                       0x91CE
#define GL_FLOAT16_SAMPLER_2D_AMD                       0x91CF
#define GL_FLOAT16_SAMPLER_3D_AMD                       0x91D0
#define GL_FLOAT16_SAMPLER_CUBE_AMD                     0x91D1
#define GL_FLOAT16_SAMPLER_2D_RECT_AMD                  0x91D2
#define GL_FLOAT16_SAMPLER_1D_ARRAY_AMD                 0x91D3
#define GL_FLOAT16_SAMPLER_2D_ARRAY_AMD                 0x91D4
#define GL_FLOAT16_SAMPLER_CUBE_MAP_ARRAY_AMD           0x91D5
#define GL_FLOAT16_SAMPLER_BUFFER_AMD                   0x91D6
#define GL_FLOAT16_SAMPLER_2D_MULTISAMPLE_AMD           0x91D7
#define GL_FLOAT16_SAMPLER_2D_MULTISAMPLE_ARRAY_AMD     0x91D8

#define GL_FLOAT16_SAMPLER_1D_SHADOW_AMD                0x91D9
#define GL_FLOAT16_SAMPLER_2D_SHADOW_AMD                0x91DA
#define GL_FLOAT16_SAMPLER_2D_RECT_SHADOW_AMD           0x91DB
#define GL_FLOAT16_SAMPLER_1D_ARRAY_SHADOW_AMD          0x91DC
#define GL_FLOAT16_SAMPLER_2D_ARRAY_SHADOW_AMD          0x91DD
#define GL_FLOAT16_SAMPLER_CUBE_SHADOW_AMD              0x91DE
#define GL_FLOAT16_SAMPLER_CUBE_MAP_ARRAY_SHADOW_AMD    0x91DF

#define GL_FLOAT16_IMAGE_1D_AMD                         0x91E0
#define GL_FLOAT16_IMAGE_2D_AMD                         0x91E1
#define GL_FLOAT16_IMAGE_3D_AMD                         0x91E2
#define GL_FLOAT16_IMAGE_2D_RECT_AMD                    0x91E3
#define GL_FLOAT16_IMAGE_CUBE_AMD                       0x91E4
#define GL_FLOAT16_IMAGE_1D_ARRAY_AMD                   0x91E5
#define GL_FLOAT16_IMAGE_2D_ARRAY_AMD                   0x91E6
#define GL_FLOAT16_IMAGE_CUBE_MAP_ARRAY_AMD             0x91E7
#define GL_FLOAT16_IMAGE_BUFFER_AMD                     0x91E8
#define GL_FLOAT16_IMAGE_2D_MULTISAMPLE_AMD             0x91E9
#define GL_FLOAT16_IMAGE_2D_MULTISAMPLE_ARRAY_AMD       0x91EA
#endif

#define GL_INT_SAMPLER_1D                 0x8DC9
#define GL_INT_SAMPLER_2D                 0x8DCA
#define GL_INT_SAMPLER_3D                 0x8DCB
#define GL_INT_SAMPLER_CUBE               0x8DCC
#define GL_INT_SAMPLER_1D_ARRAY           0x8DCE
#define GL_INT_SAMPLER_2D_ARRAY           0x8DCF
#define GL_INT_SAMPLER_2D_RECT            0x8DCD
#define GL_INT_SAMPLER_BUFFER             0x8DD0
#define GL_INT_SAMPLER_2D_MULTISAMPLE     0x9109
#define GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY 0x910C
#define GL_INT_SAMPLER_CUBE_MAP_ARRAY     0x900E
#define GL_INT_SAMPLER_CUBE_MAP_ARRAY_ARB 0x900E

#define GL_UNSIGNED_INT_SAMPLER_1D        0x8DD1
#define GL_UNSIGNED_INT_SAMPLER_2D        0x8DD2
#define GL_UNSIGNED_INT_SAMPLER_3D        0x8DD3
#define GL_UNSIGNED_INT_SAMPLER_CUBE      0x8DD4
#define GL_UNSIGNED_INT_SAMPLER_1D_ARRAY  0x8DD6
#define GL_UNSIGNED_INT_SAMPLER_2D_ARRAY  0x8DD7
#define GL_UNSIGNED_INT_SAMPLER_2D_RECT   0x8DD5
#define GL_UNSIGNED_INT_SAMPLER_BUFFER    0x8DD8
#define GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY 0x910D
#define GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY 0x900F
#define GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY_ARB 0x900F
#define GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE 0x910A

#define GL_IMAGE_1D                       0x904C
#define GL_IMAGE_2D                       0x904D
#define GL_IMAGE_3D                       0x904E
#define GL_IMAGE_2D_RECT                  0x904F
#define GL_IMAGE_CUBE                     0x9050
#define GL_IMAGE_BUFFER                   0x9051
#define GL_IMAGE_1D_ARRAY                 0x9052
#define GL_IMAGE_2D_ARRAY                 0x9053
#define GL_IMAGE_CUBE_MAP_ARRAY           0x9054
#define GL_IMAGE_2D_MULTISAMPLE           0x9055
#define GL_IMAGE_2D_MULTISAMPLE_ARRAY     0x9056
#define GL_INT_IMAGE_1D                   0x9057
#define GL_INT_IMAGE_2D                   0x9058
#define GL_INT_IMAGE_3D                   0x9059
#define GL_INT_IMAGE_2D_RECT              0x905A
#define GL_INT_IMAGE_CUBE                 0x905B
#define GL_INT_IMAGE_BUFFER               0x905C
#define GL_INT_IMAGE_1D_ARRAY             0x905D
#define GL_INT_IMAGE_2D_ARRAY             0x905E
#define GL_INT_IMAGE_CUBE_MAP_ARRAY       0x905F
#define GL_INT_IMAGE_2D_MULTISAMPLE       0x9060
#define GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY 0x9061
#define GL_UNSIGNED_INT_IMAGE_1D          0x9062
#define GL_UNSIGNED_INT_IMAGE_2D          0x9063
#define GL_UNSIGNED_INT_IMAGE_3D          0x9064
#define GL_UNSIGNED_INT_IMAGE_2D_RECT     0x9065
#define GL_UNSIGNED_INT_IMAGE_CUBE        0x9066
#define GL_UNSIGNED_INT_IMAGE_BUFFER      0x9067
#define GL_UNSIGNED_INT_IMAGE_1D_ARRAY    0x9068
#define GL_UNSIGNED_INT_IMAGE_2D_ARRAY    0x9069
#define GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY 0x906A
#define GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE 0x906B
#define GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY 0x906C

#define GL_UNSIGNED_INT_ATOMIC_COUNTER    0x92DB
