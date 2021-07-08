//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_UTIL_H_
#define COMPILER_TRANSLATOR_UTIL_H_

#include <stack>

#include <GLSLANG/ShaderLang.h>
#include "angle_gl.h"

#include "compiler/translator/HashNames.h"
#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/Operator.h"
#include "compiler/translator/Types.h"

// If overflow happens, clamp the value to UINT_MIN or UINT_MAX.
// Return false if overflow happens.
bool atoi_clamp(const char *str, unsigned int *value);

namespace sh
{

// Keeps track of whether an implicit conversion from int/uint to float is possible.
// These conversions are supported in desktop GLSL shaders only.
// Also keeps track of which side of operation should be converted.
enum class ImplicitTypeConversion
{
    Same,
    Left,
    Right,
    Invalid,
};

class TIntermBlock;
class TSymbolTable;
class TIntermTyped;

float NumericLexFloat32OutOfRangeToInfinity(const std::string &str);

// strtof_clamp is like strtof but
//   1. it forces C locale, i.e. forcing '.' as decimal point.
//   2. it sets the value to infinity if overflow happens.
//   3. str should be guaranteed to be in the valid format for a floating point number as defined
//      by the grammar in the ESSL 3.00.6 spec section 4.1.4.
// Return false if overflow happens.
bool strtof_clamp(const std::string &str, float *value);

GLenum GLVariableType(const TType &type);
GLenum GLVariablePrecision(const TType &type);
bool IsVaryingIn(TQualifier qualifier);
bool IsVaryingOut(TQualifier qualifier);
bool IsVarying(TQualifier qualifier);
bool IsGeometryShaderInput(GLenum shaderType, TQualifier qualifier);
InterpolationType GetInterpolationType(TQualifier qualifier);

// Returns array brackets including size with outermost array size first, as specified in GLSL ES
// 3.10 section 4.1.9.
ImmutableString ArrayString(const TType &type);

ImmutableString GetTypeName(const TType &type, ShHashFunction64 hashFunction, NameMap *nameMap);

TType GetShaderVariableBasicType(const sh::ShaderVariable &var);

void DeclareGlobalVariable(TIntermBlock *root, const TVariable *variable);

bool IsBuiltinOutputVariable(TQualifier qualifier);
bool IsBuiltinFragmentInputVariable(TQualifier qualifier);
bool CanBeInvariantESSL1(TQualifier qualifier);
bool CanBeInvariantESSL3OrGreater(TQualifier qualifier);
bool IsShaderOutput(TQualifier qualifier);
bool IsOutputESSL(ShShaderOutput output);
bool IsOutputGLSL(ShShaderOutput output);
bool IsOutputHLSL(ShShaderOutput output);
bool IsOutputVulkan(ShShaderOutput output);
bool IsOutputMetal(ShShaderOutput output);

bool IsInShaderStorageBlock(TIntermTyped *node);

GLenum GetImageInternalFormatType(TLayoutImageInternalFormat iifq);
// ESSL 1.00 shaders nest function body scope within function parameter scope
bool IsSpecWithFunctionBodyNewScope(ShShaderSpec shaderSpec, int shaderVersion);

// Helper functions for implicit conversions
ImplicitTypeConversion GetConversion(TBasicType t1, TBasicType t2);

bool IsValidImplicitConversion(ImplicitTypeConversion conversion, TOperator op);

size_t FindFieldIndex(const TFieldList &fieldList, const char *fieldName);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_UTIL_H_
