//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRenderGlsl/TextureBaker.h>

#include <MaterialXRender/OiioImageLoader.h>
#include <MaterialXRender/StbImageLoader.h>
#include <MaterialXRender/Util.h>

#include <MaterialXGenShader/DefaultColorManagementSystem.h>

#include <MaterialXFormat/XmlIo.h>

MATERIALX_NAMESPACE_BEGIN
TextureBakerGlsl::TextureBakerGlsl(unsigned int width, unsigned int height, Image::BaseType baseType) :
    TextureBaker<GlslRenderer, GlslShaderGenerator>(width, height, baseType, true)
{
}
MATERIALX_NAMESPACE_END
