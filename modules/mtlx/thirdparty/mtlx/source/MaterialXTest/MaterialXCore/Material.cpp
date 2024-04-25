//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Value.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXFormat/Util.h>

#include <unordered_set>

namespace mx = MaterialX;

TEST_CASE("Material", "[material]")
{
    mx::DocumentPtr doc = mx::createDocument();

    // Create a base shader nodedef.
    mx::NodeDefPtr simpleSrf = doc->addNodeDef("ND_simpleSrf", mx::SURFACE_SHADER_TYPE_STRING, "simpleSrf");
    simpleSrf->setInputValue("diffColor", mx::Color3(1.0f));
    simpleSrf->setInputValue("specColor", mx::Color3(0.0f));
    simpleSrf->setInputValue("roughness", 0.25f);
    simpleSrf->setTokenValue("texId", "01");
    REQUIRE(simpleSrf->getInputValue("diffColor")->asA<mx::Color3>() == mx::Color3(1.0f));
    REQUIRE(simpleSrf->getInputValue("specColor")->asA<mx::Color3>() == mx::Color3(0.0f));
    REQUIRE(simpleSrf->getInputValue("roughness")->asA<float>() == 0.25f);
    REQUIRE(simpleSrf->getTokenValue("texId") == "01");

    // Create an inherited shader nodedef.
    mx::NodeDefPtr anisoSrf = doc->addNodeDef("ND_anisoSrf", mx::SURFACE_SHADER_TYPE_STRING, "anisoSrf");
    anisoSrf->setInheritsFrom(simpleSrf);
    anisoSrf->setInputValue("anisotropy", 0.0f);
    REQUIRE(anisoSrf->getInheritsFrom() == simpleSrf);

    // Instantiate shader and material nodes.
    mx::NodePtr shaderNode = doc->addNode(anisoSrf->getNodeString(), "", anisoSrf->getType());
    mx::NodePtr materialNode = doc->addMaterialNode("", shaderNode);
    REQUIRE(materialNode->getUpstreamElement() == shaderNode);
    REQUIRE(shaderNode->getNodeDef() == anisoSrf);

    // Set nodedef and shader node qualifiers.
    shaderNode->setVersionString("2.0");
    REQUIRE(shaderNode->getNodeDef() == nullptr);
    anisoSrf->setVersionString("2");
    shaderNode->setVersionString("2");
    REQUIRE(shaderNode->getNodeDef() == anisoSrf);
    shaderNode->setType(mx::VOLUME_SHADER_TYPE_STRING);
    REQUIRE(shaderNode->getNodeDef() == nullptr);
    shaderNode->setType(mx::SURFACE_SHADER_TYPE_STRING);
    REQUIRE(shaderNode->getNodeDef() == anisoSrf);

    // Bind a shader input to a value.
    mx::InputPtr instanceSpecColor = shaderNode->setInputValue("specColor", mx::Color3(1.0f));
    REQUIRE(instanceSpecColor->getValue()->asA<mx::Color3>() == mx::Color3(1.0f));
    REQUIRE(instanceSpecColor->getDefaultValue()->asA<mx::Color3>() == mx::Color3(0.0f));
    REQUIRE(doc->validate());
}

TEST_CASE("Material Discovery", "[material]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr doc = mx::createDocument();
    mx::readFromXmlFile(doc, "resources/Materials/TestSuite/stdlib/materials/material_node_discovery.mtlx", searchPath);

    // 1. Find all materials referenced by material assignments
    //    which are found in connected nodegraphs
    std::unordered_set<mx::ElementPtr> foundNodes;
    for (auto look : doc->getLooks())
    {
        for (auto materialAssign : look->getMaterialAssigns())
        {
            for (auto assignOutput : materialAssign->getMaterialOutputs())
            {
                mx::NodePtr assignNode = assignOutput->getConnectedNode();
                if (assignNode)
                {
                    foundNodes.insert(assignNode);
                }
            }
        }
    }
    CHECK(foundNodes.size() == 1);

    // 2. Nodegraph Test: Find all graphs with material nodes exposed as outputs.
    //    
    foundNodes.clear();
    for (auto nodeGraph : doc->getNodeGraphs())
    {
        for (auto nodeGraphOutput : nodeGraph->getMaterialOutputs())
        {
            mx::NodePtr nodeGraphNode = nodeGraphOutput->getConnectedNode();
            if (nodeGraphNode)
            {
                foundNodes.insert(nodeGraphNode);
            }
        }
    }
    CHECK(foundNodes.size() == 3);

    // 3. Document test: Find all material nodes within nodegraphs
    //    which are not implementations. This will return less nodes
    //    as implementation graphs exist in the document.
    foundNodes.clear();
    if (doc)
    {
        for (auto documentOutput : doc->getMaterialOutputs())
        {
            mx::NodePtr documentNode = documentOutput->getConnectedNode();
            if (documentNode)
            {
                foundNodes.insert(documentNode);
            }
        }
    }
    CHECK(foundNodes.size() == 2);
}
