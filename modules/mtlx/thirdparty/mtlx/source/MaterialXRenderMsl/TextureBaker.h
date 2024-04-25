//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_TEXTUREBAKERMSL
#define MATERIALX_TEXTUREBAKERMSL

/// @file
/// Texture baking functionality

#include <iostream>

#include <MaterialXCore/Unit.h>

#include <MaterialXRender/TextureBaker.h>

#include <MaterialXRenderMsl/Export.h>

#include <MaterialXRenderMsl/MslRenderer.h>
#include <MaterialXRenderMsl/MetalTextureHandler.h>

#include <MaterialXGenMsl/MslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// A shared pointer to a TextureBakerMsl
using TextureBakerPtr = shared_ptr<class TextureBakerMsl>;

/// A vector of baked documents with their associated names.
using BakedDocumentVec = std::vector<std::pair<std::string, DocumentPtr>>;

/// @class TextureBakerMsl
/// A helper class for baking procedural material content to textures.
/// TODO: Add support for graphs containing geometric nodes such as position
///       and normal.
class MX_RENDERMSL_API TextureBakerMsl : public TextureBaker<MslRenderer, MslShaderGenerator>
{
  public:
    static TextureBakerPtr create(unsigned int width = 1024, unsigned int height = 1024, Image::BaseType baseType = Image::BaseType::UINT8)
    {
        return TextureBakerPtr(new TextureBakerMsl(width, height, baseType));
    }

  protected:
    TextureBakerMsl(unsigned int width, unsigned int height, Image::BaseType baseType);
};

MATERIALX_NAMESPACE_END

#endif
