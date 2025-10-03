// Copyright (c) 2020 The Khronos Group Inc.
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

#ifndef SPIRV_UNIFIED1_NonSemanticClspvReflection_H_
#define SPIRV_UNIFIED1_NonSemanticClspvReflection_H_

#ifdef __cplusplus
extern "C" {
#endif

enum {
    NonSemanticClspvReflectionRevision = 5,
    NonSemanticClspvReflectionRevision_BitWidthPadding = 0x7fffffff
};

enum NonSemanticClspvReflectionInstructions {
    NonSemanticClspvReflectionKernel = 1,
    NonSemanticClspvReflectionArgumentInfo = 2,
    NonSemanticClspvReflectionArgumentStorageBuffer = 3,
    NonSemanticClspvReflectionArgumentUniform = 4,
    NonSemanticClspvReflectionArgumentPodStorageBuffer = 5,
    NonSemanticClspvReflectionArgumentPodUniform = 6,
    NonSemanticClspvReflectionArgumentPodPushConstant = 7,
    NonSemanticClspvReflectionArgumentSampledImage = 8,
    NonSemanticClspvReflectionArgumentStorageImage = 9,
    NonSemanticClspvReflectionArgumentSampler = 10,
    NonSemanticClspvReflectionArgumentWorkgroup = 11,
    NonSemanticClspvReflectionSpecConstantWorkgroupSize = 12,
    NonSemanticClspvReflectionSpecConstantGlobalOffset = 13,
    NonSemanticClspvReflectionSpecConstantWorkDim = 14,
    NonSemanticClspvReflectionPushConstantGlobalOffset = 15,
    NonSemanticClspvReflectionPushConstantEnqueuedLocalSize = 16,
    NonSemanticClspvReflectionPushConstantGlobalSize = 17,
    NonSemanticClspvReflectionPushConstantRegionOffset = 18,
    NonSemanticClspvReflectionPushConstantNumWorkgroups = 19,
    NonSemanticClspvReflectionPushConstantRegionGroupOffset = 20,
    NonSemanticClspvReflectionConstantDataStorageBuffer = 21,
    NonSemanticClspvReflectionConstantDataUniform = 22,
    NonSemanticClspvReflectionLiteralSampler = 23,
    NonSemanticClspvReflectionPropertyRequiredWorkgroupSize = 24,
    NonSemanticClspvReflectionSpecConstantSubgroupMaxSize = 25,
    NonSemanticClspvReflectionArgumentPointerPushConstant = 26,
    NonSemanticClspvReflectionArgumentPointerUniform = 27,
    NonSemanticClspvReflectionProgramScopeVariablesStorageBuffer = 28,
    NonSemanticClspvReflectionProgramScopeVariablePointerRelocation = 29,
    NonSemanticClspvReflectionImageArgumentInfoChannelOrderPushConstant = 30,
    NonSemanticClspvReflectionImageArgumentInfoChannelDataTypePushConstant = 31,
    NonSemanticClspvReflectionImageArgumentInfoChannelOrderUniform = 32,
    NonSemanticClspvReflectionImageArgumentInfoChannelDataTypeUniform = 33,
    NonSemanticClspvReflectionArgumentStorageTexelBuffer = 34,
    NonSemanticClspvReflectionArgumentUniformTexelBuffer = 35,
    NonSemanticClspvReflectionConstantDataPointerPushConstant = 36,
    NonSemanticClspvReflectionProgramScopeVariablePointerPushConstant = 37,
    NonSemanticClspvReflectionPrintfInfo = 38,
    NonSemanticClspvReflectionPrintfBufferStorageBuffer = 39,
    NonSemanticClspvReflectionPrintfBufferPointerPushConstant = 40,
    NonSemanticClspvReflectionInstructionsMax = 0x7fffffff
};


enum NonSemanticClspvReflectionKernelPropertyFlags {
    NonSemanticClspvReflectionNone = 0x0,
    NonSemanticClspvReflectionMayUsePrintf = 0x1,
    NonSemanticClspvReflectionKernelPropertyFlagsMax = 0x7fffffff
};


#ifdef __cplusplus
}
#endif

#endif // SPIRV_UNIFIED1_NonSemanticClspvReflection_H_
