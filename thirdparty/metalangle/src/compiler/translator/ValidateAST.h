//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_VALIDATEAST_H_
#define COMPILER_TRANSLATOR_VALIDATEAST_H_

#include "compiler/translator/BaseTypes.h"
#include "compiler/translator/Common.h"

namespace sh
{
class TDiagnostics;
class TIntermNode;

// The following options (stored in Compiler) tell the validator what to validate.  Some validations
// are conditional to certain passes.
struct ValidateASTOptions
{
    // TODO: add support for the flags marked with TODO. http://anglebug.com/2733

    // Check that every node always has only one parent,
    bool validateSingleParent = true;
    // Check that all EOpCallFunctionInAST have their corresponding function definitions in the AST,
    // with matching symbol ids. There should also be at least a prototype declaration before the
    // function is called.
    bool validateFunctionCall = true;  // TODO
    // Check that there are no null nodes where they are not allowed, for example as children of
    // TIntermDeclaration or TIntermBlock.
    bool validateNullNodes = true;
    // Check that symbols that reference variables have consistent qualifiers and symbol ids with
    // the variable declaration. For example, references to function out parameters should be
    // EvqOut.
    bool validateQualifiers = true;  // TODO
    // Check that variable declarations that can't have initializers don't have initializers
    // (varyings, uniforms for example).
    bool validateInitializers = true;  // TODO
    // Check that there is only one TFunction with each function name referenced in the nodes (no
    // two TFunctions with the same name, taking internal/non-internal namespaces into account).
    bool validateUniqueFunctions = true;  // TODO
    // Check that references to user-defined structs are matched with the corresponding struct
    // declaration.
    bool validateStructUsage = true;  // TODO
    // Check that expression nodes have the correct type considering their operand(s).
    bool validateExpressionTypes = true;  // TODO
    // If SeparateDeclarations has been run, check for the absence of multi declarations as well.
    bool validateMultiDeclarations = false;  // TODO
};

// Check for errors and output error messages on the context.
// Returns true if there are no errors.
bool ValidateAST(TIntermNode *root, TDiagnostics *diagnostics, const ValidateASTOptions &options);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_VALIDATESWITCH_H_
