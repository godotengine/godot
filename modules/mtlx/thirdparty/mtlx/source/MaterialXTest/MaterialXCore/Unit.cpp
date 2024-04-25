//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Unit.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXFormat/Util.h>

namespace mx = MaterialX;

const float EPSILON = 1e-4f;

TEST_CASE("UnitAttribute", "[unit]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr doc = mx::createDocument();
    mx::loadLibraries({ "libraries" }, searchPath, doc);

    std::vector<mx::UnitTypeDefPtr> unitTypeDefs = doc->getUnitTypeDefs();
    REQUIRE(!unitTypeDefs.empty());

    mx::NodeGraphPtr nodeGraph = doc->addNodeGraph();
    nodeGraph->setName("graph1");

    // Basic get/set unit testing
    mx::NodePtr constant = nodeGraph->addNode("constant");
    constant->setName("constant1");
    constant->setInputValue("value", mx::Color3(0.5f));
    mx::InputPtr input = constant->getInput("value");
    input->setUnitType("distance");
    input->setUnit("meter");
    REQUIRE(input->hasUnit());
    REQUIRE(!input->getUnit().empty());

    // Test for valid unit names
    mx::OutputPtr output = nodeGraph->addOutput();
    output->setConnectedNode(constant);
    output->setUnitType("distance");
    output->setUnit("bad unit");
    REQUIRE(!output->validate());
    output->setUnit("foot");
    REQUIRE(output->hasUnit());
    REQUIRE(!output->getUnit().empty());

    REQUIRE(doc->validate());

    // Test for target unit specified on a nodedef
    mx::NodeDefPtr customNodeDef = doc->addNodeDef("ND_dummy", "float", "dummy");
    input = customNodeDef->setInputValue("angle", 23.0f, "float");
    input->setUnit("degree");
    mx::NodePtr custom = doc->addNodeInstance(customNodeDef);
    input = custom->setInputValue("angle", 45.0f, "float");
    input->setUnit("radian");
    REQUIRE(input->getUnit() == "radian");
    REQUIRE(input->getActiveUnit() == "degree");
}

TEST_CASE("UnitEvaluation", "[unit]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr doc = mx::createDocument();
    mx::loadLibraries({ "libraries" }, searchPath, doc);

    //
    // Test distance converter
    //
    mx::UnitTypeDefPtr distanceTypeDef = doc->getUnitTypeDef("distance");
    REQUIRE(distanceTypeDef);

    mx::UnitConverterRegistryPtr registry = mx::UnitConverterRegistry::create();
    mx::UnitConverterRegistryPtr registry2 = mx::UnitConverterRegistry::create();
    REQUIRE(registry == registry2);

    mx::LinearUnitConverterPtr converter = mx::LinearUnitConverter::create(distanceTypeDef);
    REQUIRE(converter);
    registry->addUnitConverter(distanceTypeDef, converter);
    mx::UnitConverterPtr uconverter = registry->getUnitConverter(distanceTypeDef);
    REQUIRE(uconverter);

    // Use converter to convert
    float result = converter->convert(0.1f, "kilometer", "millimeter");
    REQUIRE((result - 100000.0f) < EPSILON);
    result = converter->convert(2.3f, "meter", "meter");
    REQUIRE((result - 2.3f) < EPSILON);
    result = converter->convert(1.0f, "mile", "meter");
    REQUIRE((result - 1609.344f) < EPSILON);
    result = converter->convert(1.0f, "meter", "mile");
    REQUIRE((result - (1.0 / 0.000621f)) < EPSILON);

    // Use explicit converter values
    const std::unordered_map<std::string, float>& unitScale = converter->getUnitScale();
    result = 0.1f * unitScale.find("kilometer")->second / unitScale.find("millimeter")->second;
    REQUIRE((result - 100000.0f) < EPSILON);

    // Test integrer mapping
    unsigned int unitNumber = converter->getUnitAsInteger("mile");
    const std::string& unitName = converter->getUnitFromInteger(unitNumber);
    REQUIRE(unitName == "mile");

    //
    // Add angle converter
    //
    mx::UnitTypeDefPtr angleTypeDef = doc->getUnitTypeDef("angle");
    REQUIRE(angleTypeDef);
    mx::LinearUnitConverterPtr converter2 = mx::LinearUnitConverter::create(angleTypeDef);
    REQUIRE(converter2);
    registry->addUnitConverter(angleTypeDef, converter2);
    mx::UnitConverterPtr uconverter2 = registry->getUnitConverter(angleTypeDef);
    REQUIRE(uconverter2);
    result = converter2->convert(2.5f, "degree", "degree");
    REQUIRE((result - 2.5f) < EPSILON);
    result = converter2->convert(2.0f, "radian", "degree");
    REQUIRE((result - 114.591559026f) < EPSILON);
}

TEST_CASE("UnitDocument", "[unit]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr stdlib = mx::createDocument();
    mx::loadLibraries({ "libraries" }, searchPath, stdlib);
    mx::FilePath examplesPath = searchPath.find("resources/Materials/TestSuite/stdlib/units");
    searchPath.append(examplesPath);

    static const std::string DISTANCE_DEFAULT("meter");

    // Read and validate each example document.
    for (std::string filename : examplesPath.getFilesInDirectory(mx::MTLX_EXTENSION))
    {
        mx::DocumentPtr doc = mx::createDocument();
        mx::readFromXmlFile(doc, filename, searchPath);
        doc->importLibrary(stdlib);

        mx::UnitTypeDefPtr distanceTypeDef = doc->getUnitTypeDef("distance");
        REQUIRE(distanceTypeDef);

        mx::UnitConverterPtr uconverter = mx::LinearUnitConverter::create(distanceTypeDef);
        REQUIRE(uconverter);
        mx::UnitConverterRegistryPtr registry = mx::UnitConverterRegistry::create();
        registry->addUnitConverter(distanceTypeDef, uconverter);
        uconverter = registry->getUnitConverter(distanceTypeDef);
        REQUIRE(uconverter);

        // Traverse the document tree
        for (mx::ElementPtr elem : doc->traverseTree())
        {
            // If we have nodes with inputs
            mx::NodePtr pNode = elem->asA<mx::Node>();
            if (pNode)
            {
                if (pNode->getInputCount())
                {
                    for (mx::InputPtr input : pNode->getInputs())
                    {
                        const std::string type = input->getType();
                        const mx::ValuePtr value = input->getValue();
                        if (input->hasUnit() && value)
                        {
                            if (type == "float")
                            {
                                float originalval = value->asA<float>();
                                float convertedValue = uconverter->convert(originalval, input->getUnit(), DISTANCE_DEFAULT);
                                float reconvert = uconverter->convert(convertedValue, DISTANCE_DEFAULT, input->getUnit());
                                REQUIRE((originalval - reconvert) < EPSILON);
                            }
                            else if (type == "vector2")
                            {
                                mx::Vector2 originalval = value->asA<mx::Vector2>();
                                mx::Vector2 convertedValue = uconverter->convert(originalval, input->getUnit(), DISTANCE_DEFAULT);
                                mx::Vector2 reconvert = uconverter->convert(convertedValue, DISTANCE_DEFAULT, input->getUnit());
                                REQUIRE(originalval == reconvert);
                            }
                            else if (type == "vector3")
                            {
                                mx::Vector3 originalval = value->asA<mx::Vector3>();
                                mx::Vector3 convertedValue = uconverter->convert(originalval, input->getUnit(), DISTANCE_DEFAULT);
                                mx::Vector3 reconvert = uconverter->convert(convertedValue, DISTANCE_DEFAULT, input->getUnit());
                                REQUIRE(originalval == reconvert);
                            }
                            else if (type == "vector4")
                            {
                                mx::Vector4 originalval = value->asA<mx::Vector4>();
                                mx::Vector4 convertedValue = uconverter->convert(originalval, input->getUnit(), DISTANCE_DEFAULT);
                                mx::Vector4 reconvert = uconverter->convert(convertedValue, DISTANCE_DEFAULT, input->getUnit());
                                REQUIRE(originalval == reconvert);
                            }
                        }
                    }
                }
            }
        }
    }
}
