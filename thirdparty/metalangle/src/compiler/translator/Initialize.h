//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_INITIALIZE_H_
#define COMPILER_TRANSLATOR_INITIALIZE_H_

#include "compiler/translator/Common.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/SymbolTable.h"

namespace sh
{

void InitExtensionBehavior(const ShBuiltInResources &resources,
                           TExtensionBehavior &extensionBehavior);

// Resets the behavior of the extensions listed in |extensionBehavior| to the
// undefined state. These extensions will only be those initially supported in
// the ShBuiltInResources object for this compiler instance. All other
// extensions will remain unsupported.
void ResetExtensionBehavior(TExtensionBehavior &extensionBehavior);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_INITIALIZE_H_
