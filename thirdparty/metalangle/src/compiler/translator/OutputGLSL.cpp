//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/OutputGLSL.h"

#include "compiler/translator/Compiler.h"

namespace sh
{

TOutputGLSL::TOutputGLSL(TInfoSinkBase &objSink,
                         ShArrayIndexClampingStrategy clampingStrategy,
                         ShHashFunction64 hashFunction,
                         NameMap &nameMap,
                         TSymbolTable *symbolTable,
                         sh::GLenum shaderType,
                         int shaderVersion,
                         ShShaderOutput output,
                         ShCompileOptions compileOptions)
    : TOutputGLSLBase(objSink,
                      clampingStrategy,
                      hashFunction,
                      nameMap,
                      symbolTable,
                      shaderType,
                      shaderVersion,
                      output,
                      compileOptions)
{}

bool TOutputGLSL::writeVariablePrecision(TPrecision)
{
    return false;
}

void TOutputGLSL::visitSymbol(TIntermSymbol *node)
{
    TInfoSinkBase &out = objSink();

    // All the special cases are built-ins, so if it's not a built-in we can return early.
    if (node->variable().symbolType() != SymbolType::BuiltIn)
    {
        TOutputGLSLBase::visitSymbol(node);
        return;
    }

    // Some built-ins get a special translation.
    const ImmutableString &name = node->getName();
    if (name == "gl_FragDepthEXT")
    {
        out << "gl_FragDepth";
    }
    else if (name == "gl_FragColor" && sh::IsGLSL130OrNewer(getShaderOutput()))
    {
        out << "webgl_FragColor";
    }
    else if (name == "gl_FragData" && sh::IsGLSL130OrNewer(getShaderOutput()))
    {
        out << "webgl_FragData";
    }
    else if (name == "gl_SecondaryFragColorEXT")
    {
        out << "angle_SecondaryFragColor";
    }
    else if (name == "gl_SecondaryFragDataEXT")
    {
        out << "angle_SecondaryFragData";
    }
    else
    {
        TOutputGLSLBase::visitSymbol(node);
    }
}

ImmutableString TOutputGLSL::translateTextureFunction(const ImmutableString &name)
{
    static const char *simpleRename[]       = {"texture2DLodEXT",
                                         "texture2DLod",
                                         "texture2DProjLodEXT",
                                         "texture2DProjLod",
                                         "textureCubeLodEXT",
                                         "textureCubeLod",
                                         "texture2DGradEXT",
                                         "texture2DGradARB",
                                         "texture2DProjGradEXT",
                                         "texture2DProjGradARB",
                                         "textureCubeGradEXT",
                                         "textureCubeGradARB",
                                         nullptr,
                                         nullptr};
    static const char *legacyToCoreRename[] = {
        "texture2D", "texture", "texture2DProj", "textureProj", "texture2DLod", "textureLod",
        "texture2DProjLod", "textureProjLod", "texture2DRect", "texture", "texture2DRectProj",
        "textureProj", "textureCube", "texture", "textureCubeLod", "textureLod",
        // Extensions
        "texture2DLodEXT", "textureLod", "texture2DProjLodEXT", "textureProjLod",
        "textureCubeLodEXT", "textureLod", "texture2DGradEXT", "textureGrad",
        "texture2DProjGradEXT", "textureProjGrad", "textureCubeGradEXT", "textureGrad", "texture3D",
        "texture", "texture3DProj", "textureProj", "texture3DLod", "textureLod", "texture3DProjLod",
        "textureProjLod", nullptr, nullptr};
    const char **mapping =
        (sh::IsGLSL130OrNewer(getShaderOutput())) ? legacyToCoreRename : simpleRename;

    for (int i = 0; mapping[i] != nullptr; i += 2)
    {
        if (name == mapping[i])
        {
            return ImmutableString(mapping[i + 1]);
        }
    }

    return name;
}

}  // namespace sh
