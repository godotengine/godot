//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>
#include <MaterialXTest/MaterialXGenShader/GenShaderUtil.h>
#include <MaterialXTest/MaterialXGenMsl/GenMsl.h>

#include <MaterialXCore/Document.h>

#include <MaterialXFormat/File.h>

#include <MaterialXGenShader/TypeDesc.h>

#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <MaterialXGenMsl/MslSyntax.h>
#include <MaterialXGenMsl/MslResourceBindingContext.h>

namespace mx = MaterialX;

TEST_CASE("GenShader: MSL Syntax Check", "[genmsl]")
{
    mx::SyntaxPtr syntax = mx::MslSyntax::create();

    REQUIRE(syntax->getTypeName(mx::Type::FLOAT) == "float");
    REQUIRE(syntax->getTypeName(mx::Type::COLOR3) == "vec3");
    REQUIRE(syntax->getTypeName(mx::Type::VECTOR3) == "vec3");

    REQUIRE(syntax->getTypeName(mx::Type::BSDF) == "BSDF");
    REQUIRE(syntax->getOutputTypeName(mx::Type::BSDF) == "thread BSDF&");

    // Set fixed precision with one digit
    mx::ScopedFloatFormatting format(mx::Value::FloatFormatFixed, 1);

    std::string value;
    value = syntax->getDefaultValue(mx::Type::FLOAT);
    REQUIRE(value == "0.0");
    value = syntax->getDefaultValue(mx::Type::COLOR3);
    REQUIRE(value == "vec3(0.0)");
    value = syntax->getDefaultValue(mx::Type::COLOR3, true);
    REQUIRE(value == "vec3(0.0)");
    value = syntax->getDefaultValue(mx::Type::COLOR4);
    REQUIRE(value == "vec4(0.0)");
    value = syntax->getDefaultValue(mx::Type::COLOR4, true);
    REQUIRE(value == "vec4(0.0)");
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
    REQUIRE(value == "vec3(1.0, 2.0, 3.0)");
    value = syntax->getValue(mx::Type::COLOR3, *color3Value, true);
    REQUIRE(value == "vec3(1.0, 2.0, 3.0)");

    mx::ValuePtr color4Value = mx::Value::createValue<mx::Color4>(mx::Color4(1.0f, 2.0f, 3.0f, 4.0f));
    value = syntax->getValue(mx::Type::COLOR4, *color4Value);
    REQUIRE(value == "vec4(1.0, 2.0, 3.0, 4.0)");
    value = syntax->getValue(mx::Type::COLOR4, *color4Value, true);
    REQUIRE(value == "vec4(1.0, 2.0, 3.0, 4.0)");

    std::vector<float> floatArray = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f };
    mx::ValuePtr floatArrayValue = mx::Value::createValue<std::vector<float>>(floatArray);
    value = syntax->getValue(mx::Type::FLOATARRAY, *floatArrayValue);
    REQUIRE(value == "{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}");

    std::vector<int> intArray = { 1, 2, 3, 4, 5, 6, 7 };
    mx::ValuePtr intArrayValue = mx::Value::createValue<std::vector<int>>(intArray);
    value = syntax->getValue(mx::Type::INTEGERARRAY, *intArrayValue);
    REQUIRE(value == "{1, 2, 3, 4, 5, 6, 7}");
}

TEST_CASE("GenShader: MSL Implementation Check", "[genmsl]")
{
    mx::GenContext context(mx::MslShaderGenerator::create());

    mx::StringSet generatorSkipNodeTypes;
    mx::StringSet generatorSkipNodeDefs;
    GenShaderUtil::checkImplementations(context, generatorSkipNodeTypes, generatorSkipNodeDefs, 47);
}

TEST_CASE("GenShader: MSL Unique Names", "[genmsl]")
{
    mx::GenContext context(mx::MslShaderGenerator::create());
    context.registerSourceCodeSearchPath(mx::getDefaultDataSearchPath());
    GenShaderUtil::testUniqueNames(context, mx::Stage::PIXEL);
}

TEST_CASE("GenShader: MSL Bind Light Shaders", "[genmsl]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr doc = mx::createDocument();
    mx::loadLibraries({ "libraries" }, searchPath, doc);

    mx::NodeDefPtr pointLightShader = doc->getNodeDef("ND_point_light");
    mx::NodeDefPtr spotLightShader = doc->getNodeDef("ND_spot_light");
    REQUIRE(pointLightShader != nullptr);
    REQUIRE(spotLightShader != nullptr);

    mx::GenContext context(mx::MslShaderGenerator::create());
    context.registerSourceCodeSearchPath(searchPath);

    mx::HwShaderGenerator::bindLightShader(*pointLightShader, 42, context);
    REQUIRE_THROWS(mx::HwShaderGenerator::bindLightShader(*spotLightShader, 42, context));
    mx::HwShaderGenerator::unbindLightShader(42, context);
    REQUIRE_NOTHROW(mx::HwShaderGenerator::bindLightShader(*spotLightShader, 42, context));
    REQUIRE_NOTHROW(mx::HwShaderGenerator::bindLightShader(*pointLightShader, 66, context));
    mx::HwShaderGenerator::unbindLightShaders(context);
    REQUIRE_NOTHROW(mx::HwShaderGenerator::bindLightShader(*spotLightShader, 66, context));
}

static void generateMslCode()
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();

    mx::FilePathVec testRootPaths;
    testRootPaths.push_back(searchPath.find("resources/Materials/TestSuite"));
    testRootPaths.push_back(searchPath.find("resources/Materials/Examples/StandardSurface"));

    const mx::FilePath logPath("genmsl_msl23_layout_generate_test.txt");

    bool writeShadersToDisk = false;
    MslShaderGeneratorTester tester(mx::MslShaderGenerator::create(), testRootPaths, searchPath, logPath, writeShadersToDisk);

    // Set binding context to handle resource binding layouts
    tester.addUserData(mx::HW::USER_DATA_BINDING_CONTEXT, mx::MslResourceBindingContext::create());
    
    const mx::GenOptions genOptions;
    mx::FilePath optionsFilePath = searchPath.find("resources/Materials/TestSuite/_options.mtlx");
    tester.validate(genOptions, optionsFilePath);
}

TEST_CASE("GenShader: MSL Shader with Layout Generation", "[genmsl]")
{
    // Generate with standard MSL 2.3
    generateMslCode();
}
