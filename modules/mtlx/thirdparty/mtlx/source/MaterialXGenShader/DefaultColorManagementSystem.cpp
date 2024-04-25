//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/DefaultColorManagementSystem.h>

#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const string CMS_NAME = "default_cms";

// Remap from legacy color space names to their ACES 1.2 equivalents.
const StringMap COLOR_SPACE_REMAP =
{
    { "gamma18", "g18_rec709" },
    { "gamma22", "g22_rec709" },
    { "gamma24", "rec709_display" },
    { "lin_ap1", "acescg" }
};

} // anonymous namespace

//
// DefaultColorManagementSystem methods
//

DefaultColorManagementSystemPtr DefaultColorManagementSystem::create(const string& target)
{
    return DefaultColorManagementSystemPtr(new DefaultColorManagementSystem(target));
}

DefaultColorManagementSystem::DefaultColorManagementSystem(const string& target) :
    _target(target)
{
}

const string& DefaultColorManagementSystem::getName() const
{
    return CMS_NAME;
}

NodeDefPtr DefaultColorManagementSystem::getNodeDef(const ColorSpaceTransform& transform) const
{
    if (!_document)
    {
        throw ExceptionShaderGenError("No library loaded for color management system");
    }

    string sourceSpace = COLOR_SPACE_REMAP.count(transform.sourceSpace) ? COLOR_SPACE_REMAP.at(transform.sourceSpace) : transform.sourceSpace;
    string targetSpace = COLOR_SPACE_REMAP.count(transform.targetSpace) ? COLOR_SPACE_REMAP.at(transform.targetSpace) : transform.targetSpace;
    string nodeName = sourceSpace + "_to_" + targetSpace;

    for (NodeDefPtr nodeDef : _document->getMatchingNodeDefs(nodeName))
    {
        for (OutputPtr output : nodeDef->getOutputs())
        {
            if (output->getType() == transform.type->getName())
            {
                return nodeDef;
            }
        }
    }
    return nullptr;
}

MATERIALX_NAMESPACE_END
