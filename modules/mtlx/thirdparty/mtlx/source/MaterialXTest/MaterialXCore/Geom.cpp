//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXCore/Document.h>

namespace mx = MaterialX;

TEST_CASE("Geom strings", "[geom]")
{
    // Test conversion between geometry strings and paths.
    mx::StringVec geomStrings =
    {
        "",
        "/",
        "/robot1",
        "/robot1/left_arm"
    };
    for (const std::string& geomString : geomStrings)
    {
        mx::GeomPath geomPath(geomString);
        std::string newGeomString(geomPath);
        REQUIRE(newGeomString == geomString);
    }

    // Test for overlapping paths.
    REQUIRE(mx::geomStringsMatch("/", "/robot1"));
    REQUIRE(mx::geomStringsMatch("/robot1", "/robot1/left_arm"));
    REQUIRE(mx::geomStringsMatch("/robot1, /robot2", "/robot2/left_arm"));
    REQUIRE(!mx::geomStringsMatch("", "/robot1"));
    REQUIRE(!mx::geomStringsMatch("/robot1", "/robot2"));
    REQUIRE(!mx::geomStringsMatch("/robot1, /robot2", "/robot3"));

    // Test that one path contains another.
    REQUIRE(mx::geomStringsMatch("/", "/robot1", true));
    REQUIRE(!mx::geomStringsMatch("/robot1", "/", true));
}

TEST_CASE("Geom elements", "[geom]")
{
    mx::DocumentPtr doc = mx::createDocument();

    // Add geominfos and tokens
    mx::GeomInfoPtr geominfo1 = doc->addGeomInfo("geominfo1", "/robot1, /robot2");
    geominfo1->setTokenValue("asset", std::string("robot"));
    mx::GeomInfoPtr geominfo2 = doc->addGeomInfo("geominfo2", "/robot1");
    geominfo2->setTokenValue("id", std::string("01"));
    mx::GeomInfoPtr geominfo3 = doc->addGeomInfo("geominfo3", "/robot2");
    geominfo3->setTokenValue("id", std::string("02"));
    REQUIRE_THROWS_AS(doc->addGeomInfo("geominfo1"), mx::Exception);

    // Create a node graph with a single image node.
    mx::NodeGraphPtr nodeGraph = doc->addNodeGraph();
    nodeGraph->setFilePrefix("folder/");
    REQUIRE_THROWS_AS(doc->addNodeGraph(nodeGraph->getName()), mx::Exception);
    mx::NodePtr image = nodeGraph->addNode("image");
    image->setInputValue("file", "<asset><id>_diffuse_<UDIM>.tif", mx::FILENAME_TYPE_STRING);

    // Test filename string substitutions.
    mx::InputPtr fileInput = image->getInput("file");
    mx::StringResolverPtr resolver1 = image->createStringResolver("/robot1");
    resolver1->setUdimString("1001");
    mx::StringResolverPtr resolver2 = image->createStringResolver("/robot2");
    resolver2->setUdimString("1002");
    REQUIRE(fileInput->getResolvedValue(resolver1)->asA<std::string>() == "folder/robot01_diffuse_1001.tif");
    REQUIRE(fileInput->getResolvedValue(resolver2)->asA<std::string>() == "folder/robot02_diffuse_1002.tif");

    // Create a geominfo with an attribute.
    mx::GeomInfoPtr geominfo4 = doc->addGeomInfo("geominfo4", "/robot1");
    mx::StringVec udimSet = {"1001", "1002", "1003", "1004"};
    geominfo4->setGeomPropValue(mx::UDIM_SET_PROPERTY, udimSet);
    REQUIRE(doc->getGeomPropValue(mx::UDIM_SET_PROPERTY, "/robot1")->asA<mx::StringVec>() == udimSet);
    REQUIRE(doc->getGeomPropValue(mx::UDIM_SET_PROPERTY, "/robot2") == nullptr);

    // Create a base collection.
    mx::CollectionPtr collection1 = doc->addCollection("collection1");
    collection1->setIncludeGeom("/scene1");
    collection1->setExcludeGeom("/scene1/sphere2");
    REQUIRE(collection1->matchesGeomString("/scene1/sphere1"));
    REQUIRE(!collection1->matchesGeomString("/scene1/sphere2"));

    // Create a derived collection.
    mx::CollectionPtr collection2 = doc->addCollection("collection2");
    collection2->setIncludeCollection(collection1);
    REQUIRE(collection2->matchesGeomString("/scene1/sphere1"));
    REQUIRE(!collection2->matchesGeomString("/scene1/sphere2"));

    // Create and test an include cycle.
    collection1->setIncludeCollection(collection2);
    REQUIRE(!doc->validate());
    collection1->setIncludeCollection(nullptr);
    REQUIRE(doc->validate());

    // Test geometry string substitutions.
    collection1->setGeomPrefix("/root");
    REQUIRE(collection1->matchesGeomString("/root/scene1"));
    REQUIRE(!collection1->matchesGeomString("/root/scene2"));
}

TEST_CASE("GeomPropDef", "[geom]")
{
    mx::DocumentPtr doc = mx::createDocument();

    // Declare a GeomPropDef for world-space normal.
    mx::GeomPropDefPtr worldNormal = doc->addGeomPropDef("Nworld", "normal");
    worldNormal->setSpace("world");

    // Create a NodeDef with an input that defaults to the declared world-space
    // normal property.
    mx::NodeDefPtr nodedef = doc->addNodeDef("ND_foo", "color3", "foo");
    mx::InputPtr input = nodedef->addInput("input1", "vector3");
    input->setDefaultGeomPropString(worldNormal->getName());

    // Validate connections.
    REQUIRE(input->getDefaultGeomProp() == worldNormal);
    REQUIRE(doc->validate());
}
