//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Check whether variables fit within packing limits according to the packing rules from the GLSL ES
// 1.00.17 spec, Appendix A, section 7.

#ifndef COMPILER_TRANSLATOR_VARIABLEPACKER_H_
#define COMPILER_TRANSLATOR_VARIABLEPACKER_H_

#include <vector>

#include <GLSLANG/ShaderLang.h>

namespace sh
{

// Gets how many components in a row a data type takes.
int GetTypePackingComponentsPerRow(sh::GLenum type);

// Gets how many rows a data type takes.
int GetTypePackingRows(sh::GLenum type);

// Returns true if the passed in variables pack in maxVectors.
// T should be ShaderVariable or one of the subclasses of ShaderVariable.
bool CheckVariablesInPackingLimits(unsigned int maxVectors,
                                   const std::vector<ShaderVariable> &variables);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_VARIABLEPACKER_H_
