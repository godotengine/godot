//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Output the AST intermediate representation of the GLSL code.

#ifndef COMPILER_TRANSLATOR_OUTPUTTREE_H_
#define COMPILER_TRANSLATOR_OUTPUTTREE_H_

namespace sh
{

class TIntermNode;
class TInfoSinkBase;

// Output the AST along with metadata.
void OutputTree(TIntermNode *root, TInfoSinkBase &out);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_OUTPUTTREE_H_
