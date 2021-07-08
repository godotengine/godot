//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ExtensionGLSL.cpp: Implements the TExtensionGLSL class that tracks GLSL extension requirements
// of shaders.

#include "compiler/translator/ExtensionGLSL.h"

#include "compiler/translator/VersionGLSL.h"

namespace sh
{

TExtensionGLSL::TExtensionGLSL(ShShaderOutput output)
    : TIntermTraverser(true, false, false), mTargetVersion(ShaderOutputTypeToGLSLVersion(output))
{}

const std::set<std::string> &TExtensionGLSL::getEnabledExtensions() const
{
    return mEnabledExtensions;
}

const std::set<std::string> &TExtensionGLSL::getRequiredExtensions() const
{
    return mRequiredExtensions;
}

bool TExtensionGLSL::visitUnary(Visit, TIntermUnary *node)
{
    checkOperator(node);

    return true;
}

bool TExtensionGLSL::visitAggregate(Visit, TIntermAggregate *node)
{
    checkOperator(node);

    return true;
}

void TExtensionGLSL::checkOperator(TIntermOperator *node)
{
    if (mTargetVersion < GLSL_VERSION_130)
    {
        return;
    }

    switch (node->getOp())
    {
        case EOpAbs:
            break;

        case EOpSign:
            break;

        case EOpMix:
            break;

        case EOpFloatBitsToInt:
        case EOpFloatBitsToUint:
        case EOpIntBitsToFloat:
        case EOpUintBitsToFloat:
            if (mTargetVersion < GLSL_VERSION_330)
            {
                // Bit conversion functions cannot be emulated.
                mRequiredExtensions.insert("GL_ARB_shader_bit_encoding");
            }
            break;

        case EOpPackSnorm2x16:
        case EOpPackHalf2x16:
        case EOpUnpackSnorm2x16:
        case EOpUnpackHalf2x16:
            if (mTargetVersion < GLSL_VERSION_420)
            {
                mEnabledExtensions.insert("GL_ARB_shading_language_packing");

                if (mTargetVersion < GLSL_VERSION_330)
                {
                    // floatBitsToUint and uintBitsToFloat are needed to emulate
                    // packHalf2x16 and unpackHalf2x16 respectively and cannot be
                    // emulated themselves.
                    mRequiredExtensions.insert("GL_ARB_shader_bit_encoding");
                }
            }
            break;

        case EOpPackUnorm2x16:
        case EOpUnpackUnorm2x16:
            if (mTargetVersion < GLSL_VERSION_410)
            {
                mEnabledExtensions.insert("GL_ARB_shading_language_packing");
            }
            break;

        default:
            break;
    }
}

}  // namespace sh
