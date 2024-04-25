//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>
#include <MaterialXTest/MaterialXGenOsl/GenOsl.h>

#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <MaterialXGenShader/TypeDesc.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>

#include <MaterialXGenOsl/OslShaderGenerator.h>
#include <MaterialXGenOsl/OslSyntax.h>

namespace mx = MaterialX;

TEST_CASE("GenShader: OSL Syntax", "[genosl]")
{
    mx::SyntaxPtr syntax = mx::OslSyntax::create();

    REQUIRE(syntax->getTypeName(mx::Type::FLOAT) == "float");
    REQUIRE(syntax->getTypeName(mx::Type::COLOR3) == "color");
    REQUIRE(syntax->getTypeName(mx::Type::VECTOR3) == "vector");
    REQUIRE(syntax->getTypeName(mx::Type::FLOATARRAY) == "float");
    REQUIRE(syntax->getTypeName(mx::Type::INTEGERARRAY) == "int");
    REQUIRE(mx::Type::FLOATARRAY->isArray());
    REQUIRE(mx::Type::INTEGERARRAY->isArray());

    REQUIRE(syntax->getTypeName(mx::Type::BSDF) == "BSDF");
    REQUIRE(syntax->getOutputTypeName(mx::Type::BSDF) == "output BSDF");

    // Set fixed precision with one digit
    mx::ScopedFloatFormatting format(mx::Value::FloatFormatFixed, 1);

    std::string value;
    value = syntax->getDefaultValue(mx::Type::FLOAT);
    REQUIRE(value == "0.0");
    value = syntax->getDefaultValue(mx::Type::COLOR3);
    REQUIRE(value == "color(0.0)");
    value = syntax->getDefaultValue(mx::Type::COLOR3, true);
    REQUIRE(value == "color(0.0)");
    value = syntax->getDefaultValue(mx::Type::COLOR4);
    REQUIRE(value == "color4(color(0.0), 0.0)");
    value = syntax->getDefaultValue(mx::Type::COLOR4, true);
    REQUIRE(value == "{color(0.0), 0.0}");
    value = syntax->getDefaultValue(mx::Type::FLOATARRAY, true);
    REQUIRE(value.empty());
    value = syntax->getDefaultValue(mx::Type::INTEGERARRAY, true);
    REQUIRE(value.empty());

    mx::ValuePtr floatValue = mx::Value::createValue<float>(42.0f);
    value = syntax->getValue(mx::Type::FLOAT, *floatValue);
    REQUIRE(value == "42.0");
    value = syntax->getValue(mx::Type::FLOAT, *floatValue, true);
    REQUIRE(value == "42.0");

    mx::ValuePtr color3Value = mx::Value::createValue<mx::Color3>(mx::Color3(1.0f, 2.0f, 3.0f));
    value = syntax->getValue(mx::Type::COLOR3, *color3Value);
    REQUIRE(value == "color(1.0, 2.0, 3.0)");
    value = syntax->getValue(mx::Type::COLOR3, *color3Value, true);
    REQUIRE(value == "color(1.0, 2.0, 3.0)");

    mx::ValuePtr color4Value = mx::Value::createValue<mx::Color4>(mx::Color4(1.0f, 2.0f, 3.0f, 4.0f));
    value = syntax->getValue(mx::Type::COLOR4, *color4Value);
    REQUIRE(value == "color4(color(1.0, 2.0, 3.0), 4.0)");
    value = syntax->getValue(mx::Type::COLOR4, *color4Value, true);
    REQUIRE(value == "{color(1.0, 2.0, 3.0), 4.0}");

    std::vector<float> floatArray = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f };
    mx::ValuePtr floatArrayValue = mx::Value::createValue<std::vector<float>>(floatArray);
    value = syntax->getValue(mx::Type::FLOATARRAY, *floatArrayValue);
    REQUIRE(value == "{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}");

    std::vector<int> intArray = { 1, 2, 3, 4, 5, 6, 7 };
    mx::ValuePtr intArrayValue = mx::Value::createValue<std::vector<int>>(intArray);
    value = syntax->getValue(mx::Type::INTEGERARRAY, *intArrayValue);
    REQUIRE(value == "{1, 2, 3, 4, 5, 6, 7}");
}

TEST_CASE("GenShader: OSL Implementation Check", "[genosl]")
{
    mx::GenContext context(mx::OslShaderGenerator::create());

    mx::StringSet generatorSkipNodeTypes;
    generatorSkipNodeTypes.insert("light");
    mx::StringSet generatorSkipNodeDefs;

    GenShaderUtil::checkImplementations(context, generatorSkipNodeTypes, generatorSkipNodeDefs, 48);
}

TEST_CASE("GenShader: OSL Unique Names", "[genosl]")
{
    mx::GenContext context(mx::OslShaderGenerator::create());
    context.registerSourceCodeSearchPath(mx::getDefaultDataSearchPath());
    GenShaderUtil::testUniqueNames(context, mx::Stage::PIXEL);
}

TEST_CASE("GenShader: OSL Metadata", "[genosl]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr doc = mx::createDocument();
    mx::loadLibraries({ "libraries" }, searchPath, doc);

    //
    // Define custom attributes to be exported as shader metadata
    //

    mx::AttributeDefPtr adNodeName = doc->addAttributeDef("AD_node_name");
    adNodeName->setType("string");
    adNodeName->setAttrName("node_name");
    adNodeName->setExportable(true);

    mx::AttributeDefPtr adNodeCategory = doc->addAttributeDef("AD_node_category");
    adNodeCategory->setType("string");
    adNodeCategory->setAttrName("node_category");
    adNodeCategory->setExportable(true);

    mx::AttributeDefPtr adNodeTypeId = doc->addAttributeDef("AD_node_type_id");
    adNodeTypeId->setType("integer");
    adNodeTypeId->setAttrName("node_type_id");
    adNodeTypeId->setExportable(true);

    mx::AttributeDefPtr adAttributeLongName = doc->addAttributeDef("AD_attribute_long_name");
    adAttributeLongName->setType("string");
    adAttributeLongName->setAttrName("attribute_long_name");
    adAttributeLongName->setExportable(true);

    mx::AttributeDefPtr adAttributeShortName = doc->addAttributeDef("AD_attribute_short_name");
    adAttributeShortName->setType("string");
    adAttributeShortName->setAttrName("attribute_short_name");
    adAttributeShortName->setExportable(true);

    // Define a non-exportable attribute.
    mx::AttributeDefPtr adShouldNotBeExported = doc->addAttributeDef("AD_should_not_be_exported");
    adShouldNotBeExported->setType("float");
    adShouldNotBeExported->setAttrName("should_not_be_exported");
    adShouldNotBeExported->setExportable(false);

    //
    // Assign metadata on a shader nodedef
    //

    mx::NodeDefPtr stdSurfNodeDef = doc->getNodeDef("ND_standard_surface_surfaceshader");
    REQUIRE(stdSurfNodeDef != nullptr);
    stdSurfNodeDef->setAttribute("node_name", "StandardSurface");
    stdSurfNodeDef->setAttribute("node_category", "shader/surface");
    stdSurfNodeDef->setAttribute("node_type_id", "1234");

    mx::InputPtr baseColor = stdSurfNodeDef->getActiveInput("base_color");
    REQUIRE(baseColor != nullptr);
    baseColor->setAttribute("attribute_long_name", "BaseColor");
    baseColor->setAttribute("attribute_short_name", "bc");
    baseColor->setAttribute("should_not_be_exported", "42");

    //
    // Create an instance of this shader and validate that metadata is exported as expected.
    //

    mx::NodePtr stdSurf1 = doc->addNodeInstance(stdSurfNodeDef, "standardSurface1");
    REQUIRE(stdSurf1 != nullptr);

    mx::ShaderGeneratorPtr generator = mx::OslShaderGenerator::create();
    mx::GenContext context(mx::OslShaderGenerator::create());
    context.registerSourceCodeSearchPath(searchPath);

    // Metadata to export must be registered in the context before shader generation starts.
    // Custom generators can override this method to customize which metadata gets registered.
    generator->registerShaderMetadata(doc, context);

    // Generate the shader.
    mx::ShaderPtr shader = generator->generate(stdSurf1->getName(), stdSurf1, context);
    REQUIRE(shader != nullptr);
}

static void generateOslCode()
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();

    mx::FilePathVec testRootPaths;
    testRootPaths.push_back(searchPath.find("resources/Materials/TestSuite"));
    testRootPaths.push_back(searchPath.find("resources/Materials/Examples/StandardSurface"));

    const mx::FilePath logPath("genosl_vanilla_generate_test.txt");

    bool writeShadersToDisk = false;
    OslShaderGeneratorTester tester(mx::OslShaderGenerator::create(), testRootPaths, searchPath, logPath, writeShadersToDisk);
    tester.addSkipLibraryFiles();

    const mx::GenOptions genOptions;
    mx::FilePath optionsFilePath = searchPath.find("resources/Materials/TestSuite/_options.mtlx");
    tester.validate(genOptions, optionsFilePath);
}

TEST_CASE("GenShader: OSL Shader Generation", "[genosl]")
{
    generateOslCode();
}
