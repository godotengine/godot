//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ValidateMaxParameters checks if function definitions have more than a set number of parameters.

#ifndef COMPILER_TRANSLATOR_VALIDATEMAXPARAMETERS_H_
#define COMPILER_TRANSLATOR_VALIDATEMAXPARAMETERS_H_

namespace sh
{

class TIntermBlock;

// Return true if valid.
bool ValidateMaxParameters(TIntermBlock *root, unsigned int maxParameters);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_VALIDATEMAXPARAMETERS_H_
