// Copyright (c) 2020-2024 The Khronos Group Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
// 
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
// 
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
// 
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
// 

#ifndef SPIRV_UNIFIED1_AMD_shader_trinary_minmax_H_
#define SPIRV_UNIFIED1_AMD_shader_trinary_minmax_H_

#ifdef __cplusplus
extern "C" {
#endif

enum {
    AMD_shader_trinary_minmaxRevision = 4,
    AMD_shader_trinary_minmaxRevision_BitWidthPadding = 0x7fffffff
};

enum AMD_shader_trinary_minmaxInstructions {
    AMD_shader_trinary_minmaxFMin3AMD = 1,
    AMD_shader_trinary_minmaxUMin3AMD = 2,
    AMD_shader_trinary_minmaxSMin3AMD = 3,
    AMD_shader_trinary_minmaxFMax3AMD = 4,
    AMD_shader_trinary_minmaxUMax3AMD = 5,
    AMD_shader_trinary_minmaxSMax3AMD = 6,
    AMD_shader_trinary_minmaxFMid3AMD = 7,
    AMD_shader_trinary_minmaxUMid3AMD = 8,
    AMD_shader_trinary_minmaxSMid3AMD = 9,
    AMD_shader_trinary_minmaxInstructionsMax = 0x7fffffff
};


#ifdef __cplusplus
}
#endif

#endif // SPIRV_UNIFIED1_AMD_shader_trinary_minmax_H_
