//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/ColorManagementSystem.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/Nodes/SourceCodeNode.h>

MATERIALX_NAMESPACE_BEGIN

//
// ColorSpaceTransform methods
//

ColorSpaceTransform::ColorSpaceTransform(const string& ss, const string& ts, const TypeDesc* t) :
    sourceSpace(ss),
    targetSpace(ts),
    type(t)
{
    if (type != Type::COLOR3 && type != Type::COLOR4)
    {
        throw ExceptionShaderGenError("Color space transform can only be a color3 or color4.");
    }
}

ColorManagementSystem::ColorManagementSystem()
{
}

void ColorManagementSystem::loadLibrary(DocumentPtr document)
{
    _document = document;
}

bool ColorManagementSystem::supportsTransform(const ColorSpaceTransform& transform) const
{
    if (!_document)
    {
        throw ExceptionShaderGenError("No library loaded for color management system");
    }
    return getNodeDef(transform) != nullptr;
}

ShaderNodePtr ColorManagementSystem::createNode(const ShaderGraph* parent, const ColorSpaceTransform& transform, const string& name,
                                                GenContext& context) const
{
    NodeDefPtr nodeDef = getNodeDef(transform);
    if (!nodeDef)
    {
        throw ExceptionShaderGenError("No nodedef found for transform: ('" + transform.sourceSpace + "', '" + transform.targetSpace + "').");
    }

    return ShaderNode::create(parent, name, *nodeDef, context);
}

MATERIALX_NAMESPACE_END
