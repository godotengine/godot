//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMdl/Nodes/SwizzleNodeMdl.h>

#include <MaterialXGenMdl/Nodes/CompoundNodeMdl.h>

#include <MaterialXGenMdl/MdlShaderGenerator.h>

#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr SwizzleNodeMdl::create()
{
    return std::make_shared<SwizzleNodeMdl>();
}

string SwizzleNodeMdl::getVariableName(const ShaderInput* input) const
{
    if (!input->getConnection())
    {
        return input->getVariable();
    }

    // default name in case of connections
    string variableName = input->getConnection()->getVariable();

    // Allow swizzles also on custom types, like UsdUVTexture.
    // Special handling is required because struct field names follow a special naming scheme in this generator.
    const ShaderNode* upstreamNode = input->getConnection()->getNode();
    // Skip upstream nodes that are shader graphs because they don't have an implementation.
    if (upstreamNode && !upstreamNode->isAGraph())
    {
        const CompoundNodeMdl* upstreamNodeMdl = dynamic_cast<const CompoundNodeMdl*>(&upstreamNode->getImplementation());
        if (upstreamNodeMdl && upstreamNodeMdl->isReturnStruct())
        {
            // apply the same channel mask to the names of the struct fields
            size_t pos = variableName.find_last_of('_');
            if (pos != string::npos)
            {
                string channelMask = variableName.substr(pos);
                variableName = upstreamNode->getName() + "_result.mxp" + channelMask;
            }
        }
    }
    return variableName;
}

MATERIALX_NAMESPACE_END
