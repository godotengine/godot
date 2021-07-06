//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_VALIDATEGLOBALINITIALIZER_H_
#define COMPILER_TRANSLATOR_VALIDATEGLOBALINITIALIZER_H_

namespace sh
{

class TIntermTyped;

// Returns true if the initializer is valid.
bool ValidateGlobalInitializer(TIntermTyped *initializer,
                               int shaderVersion,
                               bool isWebGL,
                               bool *warning);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_VALIDATEGLOBALINITIALIZER_H_
