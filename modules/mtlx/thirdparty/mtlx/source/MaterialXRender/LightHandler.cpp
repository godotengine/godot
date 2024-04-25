//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/LightHandler.h>

#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

const int DEFAULT_ENV_SAMPLE_COUNT = 16;

void LightHandler::addLightSource(NodePtr node)
{
    _lightSources.push_back(node);
}

LightIdMap LightHandler::computeLightIdMap(const vector<NodePtr>& nodes)
{
    std::unordered_map<string, unsigned int> idMap;
    unsigned int id = 1;
    for (const auto& node : nodes)
    {
        auto nodedef = node->getNodeDef();
        if (nodedef)
        {
            const string& name = nodedef->getName();
            if (!idMap.count(name))
            {
                idMap[name] = id++;
            }
        }
    }
    return idMap;
}

void LightHandler::findLights(DocumentPtr doc, vector<NodePtr>& lights)
{
    lights.clear();
    for (NodePtr node : doc->getNodes())
    {
        const TypeDesc* type = TypeDesc::get(node->getType());
        if (*type == *Type::LIGHTSHADER)
        {
            lights.push_back(node);
        }
    }
}

void LightHandler::registerLights(DocumentPtr doc, const vector<NodePtr>& lights, GenContext& context)
{
    // Clear context light user data which is set when bindLightShader()
    // is called. This is necessary in case the light types have already been
    // registered.
    HwShaderGenerator::unbindLightShaders(context);

    if (!lights.empty())
    {
        // Create a list of unique nodedefs and ids for them
        _lightIdMap = computeLightIdMap(lights);
        for (const auto& id : _lightIdMap)
        {
            NodeDefPtr nodeDef = doc->getNodeDef(id.first);
            if (nodeDef)
            {
                HwShaderGenerator::bindLightShader(*nodeDef, id.second, context);
            }
        }
    }

    // Make sure max light count is large enough
    const unsigned int lightCount = (unsigned int) lights.size();
    if (lightCount > context.getOptions().hwMaxActiveLightSources)
    {
        context.getOptions().hwMaxActiveLightSources = lightCount;
    }
}

MATERIALX_NAMESPACE_END
