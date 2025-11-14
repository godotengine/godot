// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FFX_FSR1_RESOURCES_H
#define FFX_FSR1_RESOURCES_H

#if defined(FFX_CPU) || defined(FFX_GPU)
#define FFX_FSR1_RESOURCE_IDENTIFIER_NULL                                          0
#define FFX_FSR1_RESOURCE_IDENTIFIER_INPUT_COLOR                                   1
#define FFX_FSR1_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR                       2
#define FFX_FSR1_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT                               3

#define FFX_FSR1_RESOURCE_IDENTIFIER_COUNT                                         4

#define FFX_FSR1_CONSTANTBUFFER_IDENTIFIER_FSR1                                    0

#endif // #if defined(FFX_CPU) || defined(FFX_GPU)

#endif //!defined( FFX_FSR1_RESOURCES_H )
