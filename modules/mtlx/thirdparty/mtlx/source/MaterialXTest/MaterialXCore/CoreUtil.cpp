//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXCore/Util.h>
#include <MaterialXCore/Document.h>

namespace mx = MaterialX;

TEST_CASE("String utilities", "[coreutil]")
{
    std::string invalidName("test.name");
    REQUIRE(mx::isValidName(invalidName) == false);
    REQUIRE(mx::isValidName(mx::createValidName(invalidName)) == true);

    REQUIRE(mx::createValidName("test.name.1") == "test_name_1");
    REQUIRE(mx::createValidName("test*name>2") == "test_name_2");
    REQUIRE(mx::createValidName("testName...") == "testName___");

    REQUIRE(mx::incrementName("testName") == "testName2");
    REQUIRE(mx::incrementName("testName0") == "testName1");
    REQUIRE(mx::incrementName("testName99") == "testName100");
    REQUIRE(mx::incrementName("1testName1") == "1testName2");
    REQUIRE(mx::incrementName("41") == "42");

    REQUIRE(mx::splitString("robot1, robot2", ", ") == (std::vector<std::string>{"robot1", "robot2"}));
    REQUIRE(mx::splitString("[one...two...three]", "[.]") == (std::vector<std::string>{"one", "two", "three"}));
}

TEST_CASE("Print utilities", "[coreutil]")
{
    // Create a document.
    mx::DocumentPtr doc = mx::createDocument();

    // Create a node graph with the following structure:
    //
    //   [constant1] [constant2]      [image2]
    //           \   /          \    /
    // [image1] [add1]          [add2]
    //        \  /   \______      |   
    //    [multiply]        \__ [add3]         [noise3d]
    //             \____________  |  ____________/
    //                          [mix]
    //                            |
    //                         [output]
    //
    mx::NodeGraphPtr nodeGraph = doc->addNodeGraph();
    mx::NodePtr image1 = nodeGraph->addNode("image");
    mx::NodePtr image2 = nodeGraph->addNode("image");
    mx::NodePtr multiply = nodeGraph->addNode("multiply");
    mx::NodePtr constant1 = nodeGraph->addNode("constant");
    mx::NodePtr constant2 = nodeGraph->addNode("constant");
    mx::NodePtr add1 = nodeGraph->addNode("add");
    mx::NodePtr add2 = nodeGraph->addNode("add");
    mx::NodePtr add3 = nodeGraph->addNode("add");
    mx::NodePtr noise3d = nodeGraph->addNode("noise3d");
    mx::NodePtr mix = nodeGraph->addNode("mix");
    mx::OutputPtr output = nodeGraph->addOutput();
    add1->setConnectedNode("in1", constant1);
    add1->setConnectedNode("in2", constant2);
    add2->setConnectedNode("in1", constant2);
    add2->setConnectedNode("in2", image2);
    add3->setConnectedNode("in1", add1);
    add3->setConnectedNode("in2", add2);
    multiply->setConnectedNode("in1", image1);
    multiply->setConnectedNode("in2", add1);
    mix->setConnectedNode("fg", multiply);
    mix->setConnectedNode("bg", add3);
    mix->setConnectedNode("mask", noise3d);
    output->setConnectedNode(mix);

    // Validate the document.
    REQUIRE(doc->validate());

    const std::string blessed =
        "digraph {\n" \
        "    \"image\" [shape=box];\n" \
        "    \"image2\" [shape=box];\n" \
        "    \"constant\" [shape=box];\n" \
        "    \"constant2\" [shape=box];\n" \
        "    \"noise3d\" [shape=box];\n" \
        "    \"add\" [shape=box];\n" \
        "    \"add2\" [shape=box];\n" \
        "    \"multiply\" [shape=box];\n" \
        "    \"add3\" [shape=box];\n" \
        "    \"mix\" [shape=box];\n" \
        "    \"mix\" -> \"output\" [label=\"\"];\n" \
        "    \"multiply\" -> \"mix\" [label=\"fg\"];\n" \
        "    \"image\" -> \"multiply\" [label=\"in1\"];\n" \
        "    \"add\" -> \"multiply\" [label=\"in2\"];\n" \
        "    \"constant\" -> \"add\" [label=\"in1\"];\n" \
        "    \"constant2\" -> \"add\" [label=\"in2\"];\n" \
        "    \"add3\" -> \"mix\" [label=\"bg\"];\n" \
        "    \"add\" -> \"add3\" [label=\"in1\"];\n" \
        "    \"add2\" -> \"add3\" [label=\"in2\"];\n" \
        "    \"constant2\" -> \"add2\" [label=\"in1\"];\n" \
        "    \"image2\" -> \"add2\" [label=\"in2\"];\n" \
        "    \"noise3d\" -> \"mix\" [label=\"mask\"];\n" \
        "}\n";

    REQUIRE(nodeGraph->asStringDot() == blessed);
}
