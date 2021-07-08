//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/VersionGLSL.h"

#include "angle_gl.h"
#include "compiler/translator/Symbol.h"

namespace sh
{

namespace
{
constexpr const ImmutableString kGlPointCoordString("gl_PointCoord");
}  // anonymous namespace

int ShaderOutputTypeToGLSLVersion(ShShaderOutput output)
{
    switch (output)
    {
        case SH_GLSL_130_OUTPUT:
            return GLSL_VERSION_130;
        case SH_GLSL_140_OUTPUT:
            return GLSL_VERSION_140;
        case SH_GLSL_150_CORE_OUTPUT:
            return GLSL_VERSION_150;
        case SH_GLSL_330_CORE_OUTPUT:
            return GLSL_VERSION_330;
        case SH_GLSL_400_CORE_OUTPUT:
            return GLSL_VERSION_400;
        case SH_GLSL_410_CORE_OUTPUT:
            return GLSL_VERSION_410;
        case SH_GLSL_420_CORE_OUTPUT:
            return GLSL_VERSION_420;
        case SH_GLSL_430_CORE_OUTPUT:
            return GLSL_VERSION_430;
        case SH_GLSL_440_CORE_OUTPUT:
            return GLSL_VERSION_440;
        case SH_GLSL_450_CORE_OUTPUT:
            return GLSL_VERSION_450;
        case SH_GLSL_COMPATIBILITY_OUTPUT:
            return GLSL_VERSION_110;
        default:
            UNREACHABLE();
            return 0;
    }
}

// We need to scan for the following:
// 1. "invariant" keyword: This can occur in both - vertex and fragment shaders
//    but only at the global scope.
// 2. "gl_PointCoord" built-in variable: This can only occur in fragment shader
//    but inside any scope.
// 3. Call to a matrix constructor with another matrix as argument.
//    (These constructors were reserved in GLSL version 1.10.)
// 4. Arrays as "out" function parameters.
//    GLSL spec section 6.1.1: "When calling a function, expressions that do
//    not evaluate to l-values cannot be passed to parameters declared as
//    out or inout."
//    GLSL 1.1 section 5.8: "Other binary or unary expressions,
//    non-dereferenced arrays, function names, swizzles with repeated fields,
//    and constants cannot be l-values."
//    GLSL 1.2 relaxed the restriction on arrays, section 5.8: "Variables that
//    are built-in types, entire structures or arrays... are all l-values."
//
TVersionGLSL::TVersionGLSL(sh::GLenum type, const TPragma &pragma, ShShaderOutput output)
    : TIntermTraverser(true, false, false)
{
    mVersion = ShaderOutputTypeToGLSLVersion(output);
    if (pragma.stdgl.invariantAll)
    {
        ensureVersionIsAtLeast(GLSL_VERSION_120);
    }
    if (type == GL_COMPUTE_SHADER)
    {
        ensureVersionIsAtLeast(GLSL_VERSION_430);
    }
}

void TVersionGLSL::visitSymbol(TIntermSymbol *node)
{
    if (node->variable().symbolType() == SymbolType::BuiltIn &&
        node->getName() == kGlPointCoordString)
    {
        ensureVersionIsAtLeast(GLSL_VERSION_120);
    }
}

bool TVersionGLSL::visitDeclaration(Visit, TIntermDeclaration *node)
{
    const TIntermSequence &sequence = *(node->getSequence());
    if (sequence.front()->getAsTyped()->getType().isInvariant())
    {
        ensureVersionIsAtLeast(GLSL_VERSION_120);
    }
    return true;
}

bool TVersionGLSL::visitInvariantDeclaration(Visit, TIntermInvariantDeclaration *node)
{
    ensureVersionIsAtLeast(GLSL_VERSION_120);
    return true;
}

void TVersionGLSL::visitFunctionPrototype(TIntermFunctionPrototype *node)
{
    size_t paramCount = node->getFunction()->getParamCount();
    for (size_t i = 0; i < paramCount; ++i)
    {
        const TVariable *param = node->getFunction()->getParam(i);
        const TType &type      = param->getType();
        if (type.isArray())
        {
            TQualifier qualifier = type.getQualifier();
            if ((qualifier == EvqOut) || (qualifier == EvqInOut))
            {
                ensureVersionIsAtLeast(GLSL_VERSION_120);
                break;
            }
        }
    }
}

bool TVersionGLSL::visitAggregate(Visit, TIntermAggregate *node)
{
    if (node->getOp() == EOpConstruct && node->getType().isMatrix())
    {
        const TIntermSequence &sequence = *(node->getSequence());
        if (sequence.size() == 1)
        {
            TIntermTyped *typed = sequence.front()->getAsTyped();
            if (typed && typed->isMatrix())
            {
                ensureVersionIsAtLeast(GLSL_VERSION_120);
            }
        }
    }
    return true;
}

void TVersionGLSL::ensureVersionIsAtLeast(int version)
{
    mVersion = std::max(version, mVersion);
}

}  // namespace sh
