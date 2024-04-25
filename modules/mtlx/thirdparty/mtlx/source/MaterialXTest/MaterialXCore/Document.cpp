//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>

namespace mx = MaterialX;

TEST_CASE("Document", "[document]")
{
    // Create a document.
    mx::DocumentPtr doc = mx::createDocument();

    // Create a node graph with a constant color output.
    mx::NodeGraphPtr nodeGraph = doc->addNodeGraph();
    mx::NodePtr constant = nodeGraph->addNode("constant");
    constant->setInputValue("value", mx::Color3(0.5f));
    mx::OutputPtr output = nodeGraph->addOutput();
    output->setConnectedNode(constant);
    REQUIRE(output->isColorType());
    REQUIRE(doc->validate());

    // Create and test a type mismatch in a connection.
    output->setType("float");
    REQUIRE(!doc->validate());
    output->setType("color3");
    REQUIRE(doc->validate());

    // Test hierarchical name paths.
    REQUIRE(constant->getNamePath() == "nodegraph1/node1");
    REQUIRE(constant->getNamePath(nodeGraph) == "node1");

    // Test getting elements by path
    REQUIRE(doc->getDescendant("") == doc);
    REQUIRE(doc->getDescendant("nodegraph1") == nodeGraph);
    REQUIRE(doc->getDescendant("nodegraph1/node1") == constant);
    REQUIRE(doc->getDescendant("missingElement") == mx::ElementPtr());
    REQUIRE(doc->getDescendant("nodegraph1/missingNode") == mx::ElementPtr());
    REQUIRE(nodeGraph->getDescendant("") == nodeGraph);
    REQUIRE(nodeGraph->getDescendant("node1") == constant);
    REQUIRE(nodeGraph->getDescendant("missingNode") == mx::ElementPtr());

    // Create a simple shader interface.
    mx::NodeDefPtr simpleSrf = doc->addNodeDef("", mx::SURFACE_SHADER_TYPE_STRING, "simpleSrf");
    simpleSrf->setInputValue("diffColor", mx::Color3(1.0f));
    simpleSrf->setInputValue("specColor", mx::Color3(0.0f));
    mx::InputPtr roughness = simpleSrf->setInputValue("roughness", 0.25f);
    REQUIRE(!roughness->getIsUniform());
    roughness->setIsUniform(true);
    REQUIRE(roughness->getIsUniform());

    // Instantiate shader and material nodes.
    mx::NodePtr shaderNode = doc->addNodeInstance(simpleSrf);
    mx::NodePtr materialNode = doc->addMaterialNode("", shaderNode);
    REQUIRE(materialNode->getUpstreamElement() == shaderNode);

    // Bind the diffuse color input to the constant color output.
    shaderNode->setConnectedOutput("diffColor", output);
    REQUIRE(shaderNode->getUpstreamElement() == constant);

    // Bind the roughness input to a value.
    mx::InputPtr instanceRoughness = shaderNode->setInputValue("roughness", 0.5f);
    REQUIRE(instanceRoughness->getValue()->asA<float>() == 0.5f);
    REQUIRE(instanceRoughness->getDefaultValue()->asA<float>() == 0.25f);

    // Create a collection 
    mx::CollectionPtr collection = doc->addCollection();
    REQUIRE(doc->getCollections().size() == 1);
    REQUIRE(doc->getCollection(collection->getName()));
    doc->removeCollection(collection->getName());
    REQUIRE(doc->getCollections().size() == 0);

    // Create a property set
    mx::PropertySetPtr propertySet = doc->addPropertySet();
    REQUIRE(doc->getPropertySets().size() == 1);
    REQUIRE(doc->getPropertySet(propertySet->getName()) != nullptr);
    doc->removePropertySet(propertySet->getName());
    REQUIRE(doc->getPropertySets().size() == 0);

    // Validate the document.
    REQUIRE(doc->validate());

    // Create a namespaced custom library.
    mx::DocumentPtr customLibrary = mx::createDocument();
    customLibrary->setNamespace("custom");
    mx::NodeGraphPtr customNodeGraph = customLibrary->addNodeGraph("NG_custom");
    mx::NodeDefPtr customNodeDef = customLibrary->addNodeDef("ND_simpleSrf", mx::SURFACE_SHADER_TYPE_STRING, "simpleSrf");
    mx::ImplementationPtr customImpl = customLibrary->addImplementation("IM_custom");
    customNodeGraph->addNodeInstance(customNodeDef, "custom1");
    customImpl->setNodeDef(customNodeDef);
    REQUIRE(customLibrary->validate());

    // Import the custom library.
    doc->importLibrary(customLibrary);
    mx::NodeGraphPtr importedNodeGraph = doc->getNodeGraph("custom:NG_custom");
    mx::NodeDefPtr importedNodeDef = doc->getNodeDef("custom:ND_simpleSrf");
    mx::ImplementationPtr importedImpl = doc->getImplementation("custom:IM_custom");
    mx::NodePtr importedNode = importedNodeGraph->getNode("custom1");
    REQUIRE(importedNodeDef != nullptr);
    REQUIRE(importedNode->getNodeDef() == importedNodeDef);
    REQUIRE(importedImpl->getNodeDef() == importedNodeDef);

    // Validate the combined document.
    REQUIRE(doc->validate());
}
