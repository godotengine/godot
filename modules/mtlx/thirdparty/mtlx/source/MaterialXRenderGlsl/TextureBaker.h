//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_TEXTUREBAKER_GLSL
#define MATERIALX_TEXTUREBAKER_GLSL

/// @file
/// Texture baking functionality

#include <iostream>

#include <MaterialXCore/Unit.h>
#include <MaterialXRender/TextureBaker.h>

#include <MaterialXRenderGlsl/Export.h>

#include <MaterialXRenderGlsl/GlslRenderer.h>
#include <MaterialXRenderGlsl/GLTextureHandler.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// A shared pointer to a TextureBaker
using TextureBakerPtr = shared_ptr<class TextureBakerGlsl>;

/// A vector of baked documents with their associated names.
using BakedDocumentVec = std::vector<std::pair<std::string, DocumentPtr>>;

/// @class TextureBakerGlsl
/// An implementation of TextureBaker based on GLSL shader generation.
class MX_RENDERGLSL_API TextureBakerGlsl : public TextureBaker<GlslRenderer, GlslShaderGenerator>
{
  public:
    static TextureBakerPtr create(unsigned int width = 1024, unsigned int height = 1024, Image::BaseType baseType = Image::BaseType::UINT8)
    {
        return TextureBakerPtr(new TextureBakerGlsl(width, height, baseType));
    }

    TextureBakerGlsl(unsigned int width, unsigned int height, Image::BaseType baseType);
};

MATERIALX_NAMESPACE_END

#endif
