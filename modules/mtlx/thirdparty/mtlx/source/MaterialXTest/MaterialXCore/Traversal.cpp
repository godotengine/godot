//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

namespace mx = MaterialX;

TEST_CASE("IntraGraph Traversal", "[traversal]")
{
    // Test null iterators.
    mx::TreeIterator nullTree = mx::NULL_TREE_ITERATOR;
    mx::GraphIterator nullGraph = mx::NULL_GRAPH_ITERATOR;
    REQUIRE(*nullTree == nullptr);
    REQUIRE(*nullGraph == mx::NULL_EDGE);
    ++nullTree;
    ++nullGraph;
    REQUIRE((nullTree == mx::NULL_TREE_ITERATOR));
    REQUIRE((nullGraph == mx::NULL_GRAPH_ITERATOR));

    // Create a document.
    mx::DocumentPtr doc = mx::createDocument();

    // Create a node graph with the following structure:
    //
    // [image1] [constant]     [image2]
    //        \ /                 |   
    //    [multiply]          [contrast]         [noise3d]
    //             \____________  |  ____________/
    //                          [mix]
    //                            |
    //                         [output]
    //
    mx::NodeGraphPtr nodeGraph = doc->addNodeGraph();
    mx::NodePtr image1 = nodeGraph->addNode("image");
    mx::NodePtr image2 = nodeGraph->addNode("image");
    mx::NodePtr constant = nodeGraph->addNode("constant");
    mx::NodePtr multiply = nodeGraph->addNode("multiply");
    mx::NodePtr contrast = nodeGraph->addNode("contrast");
    mx::NodePtr noise3d = nodeGraph->addNode("noise3d");
    mx::NodePtr mix = nodeGraph->addNode("mix");
    mx::OutputPtr output = nodeGraph->addOutput();
    multiply->setConnectedNode("in1", image1);
    multiply->setConnectedNode("in2", constant);
    contrast->setConnectedNode("in", image2);
    mix->setConnectedNode("fg", multiply);
    mix->setConnectedNode("bg", contrast);
    mix->setConnectedNode("mask", noise3d);
    output->setConnectedNode(mix);

    // Validate the document.
    REQUIRE(doc->validate());

    // Traverse the document tree (implicit iterator).
    int nodeCount = 0;
    for (mx::ElementPtr elem : doc->traverseTree())
    {
        REQUIRE(elem->getName() == mx::createValidName(elem->getName()));
        if (elem->isA<mx::Node>())
        {
            nodeCount++;
        }
    }
    REQUIRE(nodeCount == 7);

    // Traverse the document tree (explicit iterator).
    nodeCount = 0;
    size_t maxElementDepth = 0;
    for (mx::TreeIterator it = doc->traverseTree().begin(); it != mx::TreeIterator::end(); ++it)
    {
        mx::ElementPtr elem = it.getElement();
        if (elem->isA<mx::Node>())
        {
            nodeCount++;
        }
        maxElementDepth = std::max(maxElementDepth, it.getElementDepth());
    }
    REQUIRE(nodeCount == 7);
    REQUIRE(maxElementDepth == 3);

    // Traverse the document tree (prune subtree).
    nodeCount = 0;
    for (mx::TreeIterator it = doc->traverseTree().begin(); it != mx::TreeIterator::end(); ++it)
    {
        mx::ElementPtr elem = it.getElement();
        if (elem->isA<mx::Node>())
        {
            nodeCount++;
        }
        if (elem->isA<mx::NodeGraph>())
        {
            it.setPruneSubtree(true);
        }
    }
    REQUIRE(nodeCount == 0);

    // Traverse upstream from the graph output (implicit iterator).
    nodeCount = 0;
    for (mx::Edge edge : output->traverseGraph())
    {
        mx::ElementPtr upstreamElem = edge.getUpstreamElement();
        mx::ElementPtr connectingElem = edge.getConnectingElement();
        mx::ElementPtr downstreamElem = edge.getDownstreamElement();
        if (upstreamElem->isA<mx::Node>())
        {
            nodeCount++;
            if (downstreamElem->isA<mx::Node>())
            {
                REQUIRE(connectingElem->isA<mx::Input>());
            }
        }
    }
    REQUIRE(nodeCount == 7);

    // Traverse upstream from the graph output (explicit iterator).
    nodeCount = 0;
    maxElementDepth = 0;
    size_t maxNodeDepth = 0;
    for (mx::GraphIterator it = output->traverseGraph().begin(); it != mx::GraphIterator::end(); ++it)
    {
        mx::ElementPtr upstreamElem = it.getUpstreamElement();
        mx::ElementPtr connectingElem = it.getConnectingElement();
        mx::ElementPtr downstreamElem = it.getDownstreamElement();
        if (upstreamElem->isA<mx::Node>())
        {
            nodeCount++;
            if (downstreamElem->isA<mx::Node>())
            {
                REQUIRE(connectingElem->isA<mx::Input>());
            }
        }
        maxElementDepth = std::max(maxElementDepth, it.getElementDepth());
        maxNodeDepth = std::max(maxNodeDepth, it.getNodeDepth());
    }
    REQUIRE(nodeCount == 7);
    REQUIRE(maxElementDepth == 3);
    REQUIRE(maxNodeDepth == 3);

    // Traverse upstream from the graph output (prune subgraph).
    nodeCount = 0;
    for (mx::GraphIterator it = output->traverseGraph().begin(); it != mx::GraphIterator::end(); ++it)
    {
        mx::ElementPtr upstreamElem = it.getUpstreamElement();
        if (upstreamElem->isA<mx::Node>())
        {
            nodeCount++;
            if (upstreamElem->getCategory() == "multiply")
            {
                it.setPruneSubgraph(true);
            }
        }
    }
    REQUIRE(nodeCount == 5);

    // Create and detect a cycle.
    multiply->setConnectedNode("in2", mix);
    REQUIRE(output->hasUpstreamCycle());
    REQUIRE(!doc->validate());
    multiply->setConnectedNode("in2", constant);
    REQUIRE(!output->hasUpstreamCycle());
    REQUIRE(doc->validate());

    // Create and detect a loop.
    contrast->setConnectedNode("in", contrast);
    REQUIRE(output->hasUpstreamCycle());
    REQUIRE(!doc->validate());
    contrast->setConnectedNode("in", image2);
    REQUIRE(!output->hasUpstreamCycle());
    REQUIRE(doc->validate());
}

TEST_CASE("InterGraph Traversal", "[traversal]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr doc = mx::createDocument();
    mx::readFromXmlFile(doc, "resources/Materials/TestSuite/stdlib/nodegraph_inputs/nodegraph_nodegraph.mtlx", searchPath);

    for (mx::NodeGraphPtr graph : doc->getNodeGraphs())
    {
        for (mx::InputPtr interfaceInput : graph->getInputs())
        {
            if (!interfaceInput->getNodeName().empty() || !interfaceInput->getNodeGraphString().empty())
            {
                REQUIRE(interfaceInput->getConnectedNode() != nullptr);                    
            }
        }
    }
}
