//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>
#include <MaterialXTest/MaterialXGenShader/GenShaderUtil.h>

#include <MaterialXCore/Document.h>

#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/ShaderTranslator.h>
#include <MaterialXGenShader/Util.h>

#ifdef MATERIALX_BUILD_GEN_GLSL
#include <MaterialXGenGlsl/GlslShaderGenerator.h>
#endif
#ifdef MATERIALX_BUILD_GEN_OSL
#include <MaterialXGenOsl/OslShaderGenerator.h>
#endif
#ifdef MATERIALX_BUILD_GEN_MDL
#include <MaterialXGenMdl/MdlShaderGenerator.h>
#endif
#ifdef MATERIALX_BUILD_GEN_MSL
#include <MaterialXGenMsl/MslShaderGenerator.h>
#endif

#include <cstdlib>
#include <iostream>
#include <vector>
#include <set>

namespace mx = MaterialX;

//
// Base tests
//

TEST_CASE("GenShader: Utilities", "[genshader]")
{
    // Test simple text substitution
    std::string test1 = "Look behind you, a $threeheaded $monkey!";
    std::string result1 = "Look behind you, a mighty pirate!";
    mx::StringMap subst1 = { {"$threeheaded","mighty"}, {"$monkey","pirate"} };
    mx::tokenSubstitution(subst1, test1);
    REQUIRE(test1 == result1);

    // Test uniform name substitution
    std::string test2 = "uniform vec3 " + mx::HW::T_ENV_RADIANCE + ";";
    std::string result2 = "uniform vec3 " + mx::HW::ENV_RADIANCE + ";";
    mx::StringMap subst2 = { {mx::HW::T_ENV_RADIANCE, mx::HW::ENV_RADIANCE} };
    mx::tokenSubstitution(subst2, test2);
    REQUIRE(test2 == result2);
}

TEST_CASE("GenShader: Valid Libraries", "[genshader]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr doc = mx::createDocument();
    loadLibraries({ "libraries" }, searchPath, doc);

    std::string validationErrors;
    bool valid = doc->validate(&validationErrors);
    if (!valid)
    {
        std::cout << validationErrors << std::endl;
    }
    REQUIRE(valid);
}

TEST_CASE("GenShader: TypeDesc Check", "[genshader]")
{
    // Make sure the standard types are registered
    const mx::TypeDesc* floatType = mx::TypeDesc::get("float");
    REQUIRE(floatType != nullptr);
    REQUIRE(floatType->getBaseType() == mx::TypeDesc::BASETYPE_FLOAT);
    const mx::TypeDesc* integerType = mx::TypeDesc::get("integer");
    REQUIRE(integerType != nullptr);
    REQUIRE(integerType->getBaseType() == mx::TypeDesc::BASETYPE_INTEGER);
    const mx::TypeDesc* booleanType = mx::TypeDesc::get("boolean");
    REQUIRE(booleanType != nullptr);
    REQUIRE(booleanType->getBaseType() == mx::TypeDesc::BASETYPE_BOOLEAN);
    const mx::TypeDesc* color3Type = mx::TypeDesc::get("color3");
    REQUIRE(color3Type != nullptr);
    REQUIRE(color3Type->getBaseType() == mx::TypeDesc::BASETYPE_FLOAT);
    REQUIRE(color3Type->getSemantic() == mx::TypeDesc::SEMANTIC_COLOR);
    REQUIRE(color3Type->isFloat3());
    const mx::TypeDesc* color4Type = mx::TypeDesc::get("color4");
    REQUIRE(color4Type != nullptr);
    REQUIRE(color4Type->getBaseType() == mx::TypeDesc::BASETYPE_FLOAT);
    REQUIRE(color4Type->getSemantic() == mx::TypeDesc::SEMANTIC_COLOR);
    REQUIRE(color4Type->isFloat4());

    // Make sure we can register a new custom type
    const mx::TypeDesc* fooType = mx::TypeDesc::registerType("foo", mx::TypeDesc::BASETYPE_FLOAT, mx::TypeDesc::SEMANTIC_COLOR, 5);
    REQUIRE(fooType != nullptr);

    // Make sure we can't use a name that is already taken
    REQUIRE_THROWS(mx::TypeDesc::registerType("color3", mx::TypeDesc::BASETYPE_FLOAT));

    // Make sure we can't request an unknown type
    REQUIRE(mx::TypeDesc::get("bar") == nullptr);
}

TEST_CASE("GenShader: Shader Translation", "[translate]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::ShaderTranslatorPtr shaderTranslator = mx::ShaderTranslator::create();

    mx::FilePath testPath = searchPath.find("resources/Materials/Examples/StandardSurface");
    for (mx::FilePath& mtlxFile : testPath.getFilesInDirectory(mx::MTLX_EXTENSION))
    {
        mx::DocumentPtr doc = mx::createDocument();
        loadLibraries({ "libraries/targets", "libraries/stdlib", "libraries/pbrlib", "libraries/bxdf" }, searchPath, doc);

        mx::readFromXmlFile(doc, testPath / mtlxFile, searchPath);
        mtlxFile.removeExtension();

        bool translated = false;
        try
        {
            shaderTranslator->translateAllMaterials(doc, "UsdPreviewSurface");
            translated = true;
        }
        catch (mx::Exception &e)
        {
            std::cout << "Failed translating: " << (testPath / mtlxFile).asString() << ": " << e.what() << std::endl;
        }
        REQUIRE(translated);

        std::string validationErrors;
        bool valid = doc->validate(&validationErrors);
        if (!doc->validate(&validationErrors))
        {
            std::cout << "Shader translation of " << (testPath / mtlxFile).asString() << " failed" << std::endl;
            std::cout << "Validation errors: " << validationErrors << std::endl;
        }
        REQUIRE(valid);
    }
}

TEST_CASE("GenShader: Transparency Regression Check", "[genshader]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr libraries = mx::createDocument();
    mx::loadLibraries({ "libraries" }, searchPath, libraries);

    const mx::FilePath resourcePath = searchPath.find("resources");
    mx::StringVec failedTests;
    mx::FilePathVec testFiles = { 
        "Materials/Examples/StandardSurface/standard_surface_default.mtlx", 
        "Materials/Examples/StandardSurface/standard_surface_glass.mtlx",
        "Materials/TestSuite/libraries/metal/brass_wire_mesh.mtlx"
    };
    std::vector<bool> transparencyTest = { false, true, true };
    for (size_t i=0; i<testFiles.size(); i++)
    {
        const mx::FilePath& testFile = resourcePath / testFiles[i];
        bool testValue = transparencyTest[i];

        mx::DocumentPtr testDoc = mx::createDocument();
        testDoc->importLibrary(libraries);

        try
        {
            mx::readFromXmlFile(testDoc, testFile, searchPath);
            std::vector<mx::TypedElementPtr> renderables = mx::findRenderableElements(testDoc);
            for (auto renderable : renderables)
            {
                mx::NodePtr node = renderable->asA<mx::Node>();
                if (!node)
                {
                    continue;
                }
                if (testValue != mx::isTransparentSurface(node))
                {
                    failedTests.push_back(std::string("File: ") + testFile.asString() + std::string(". Element: ")
                        + renderable->getNamePath() + std::string(" should be:" + std::to_string(testValue)));
                }
            }
        }
        catch (mx::Exception& e)
        {
            INFO(std::string("Test failed: ") + std::string(e.what()));
        }
    }
    for (auto failedTest : failedTests)
    {
        INFO(failedTest);
    }
    CHECK(failedTests.empty());
}

void testDeterministicGeneration(mx::DocumentPtr libraries, mx::GenContext& context)
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePath testFile = searchPath.find("resources/Materials/Examples/StandardSurface/standard_surface_marble_solid.mtlx");
    mx::string testElement = "SR_marble1";

    const size_t numRuns = 10;
    mx::vector<mx::DocumentPtr> testDocs(numRuns);
    mx::StringVec sourceCode(numRuns);

    for (size_t i = 0; i < numRuns; ++i)
    {
        mx::DocumentPtr testDoc = mx::createDocument();
        mx::readFromXmlFile(testDoc, testFile);
        testDoc->importLibrary(libraries);

        // Keep the document alive to make sure
        // new memory is allocated for each run
        testDocs[i] = testDoc;

        mx::ElementPtr element = testDoc->getChild(testElement);
        CHECK(element);

        mx::ShaderPtr shader = context.getShaderGenerator().generate(testElement, element, context);
        sourceCode[i] = shader->getSourceCode();

        if (i > 0)
        {
            // Check if the generated source code is the same
            // for each successive run.
            CHECK(sourceCode[i] == sourceCode[0]);
        }
    }
}

TEST_CASE("GenShader: Deterministic Generation", "[genshader]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr libraries = mx::createDocument();
    mx::loadLibraries({ "libraries" }, searchPath, libraries);

#ifdef MATERIALX_BUILD_GEN_GLSL
    {
        mx::GenContext context(mx::GlslShaderGenerator::create());
        context.registerSourceCodeSearchPath(searchPath);
        testDeterministicGeneration(libraries, context);
    }
#endif
#ifdef MATERIALX_BUILD_GEN_OSL
    {
        mx::GenContext context(mx::OslShaderGenerator::create());
        context.registerSourceCodeSearchPath(searchPath);
        testDeterministicGeneration(libraries, context);
    }
#endif
#ifdef MATERIALX_BUILD_GEN_MDL
    {
        mx::GenContext context(mx::MdlShaderGenerator::create());
        context.registerSourceCodeSearchPath(searchPath);
        testDeterministicGeneration(libraries, context);
    }
#endif
#ifdef MATERIALX_BUILD_GEN_MSL
    {
        mx::GenContext context(mx::MslShaderGenerator::create());
        context.registerSourceCodeSearchPath(searchPath);
        testDeterministicGeneration(libraries, context);
    }
#endif
}

void checkPixelDependencies(mx::DocumentPtr libraries, mx::GenContext& context)
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePath testFile = searchPath.find("resources/Materials/Examples/GltfPbr/gltf_pbr_boombox.mtlx");
    mx::string testElement = "Material_boombox";

    mx::DocumentPtr testDoc = mx::createDocument();
    mx::readFromXmlFile(testDoc, testFile);
    testDoc->importLibrary(libraries);

    mx::ElementPtr element = testDoc->getChild(testElement);
    CHECK(element);

    mx::ShaderPtr shader = context.getShaderGenerator().generate(testElement, element, context);
    std::set<std::string> dependencies = shader->getStage("pixel").getSourceDependencies();
    for (auto dependency : dependencies) {
        mx::FilePath path(dependency);
        REQUIRE(path.exists() == true);
    }
}

TEST_CASE("GenShader: Track Dependencies", "[genshader]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr libraries = mx::createDocument();
    mx::loadLibraries({ "libraries" }, searchPath, libraries);

#ifdef MATERIALX_BUILD_GEN_GLSL
    {
        mx::GenContext context(mx::GlslShaderGenerator::create());
        context.registerSourceCodeSearchPath(searchPath);
        checkPixelDependencies(libraries, context);
    }
#endif
#ifdef MATERIALX_BUILD_GEN_OSL
    {
        mx::GenContext context(mx::OslShaderGenerator::create());
        context.registerSourceCodeSearchPath(searchPath);
        checkPixelDependencies(libraries, context);
    }
#endif
#ifdef MATERIALX_BUILD_GEN_MDL
    {
        mx::GenContext context(mx::MdlShaderGenerator::create());
        context.registerSourceCodeSearchPath(searchPath);
        checkPixelDependencies(libraries, context);
    }
#endif
}

void variableTracker(mx::ShaderNode* node, mx::GenContext& /*context*/)
{
    static mx::StringMap results;
    results["primvar_one"] = "geompropvalue1/geomprop";
    results["primvar_two"] = "geompropvalue2/geomprop";
    results["0"] = "Tworld";
    results["upstream_primvar"] = "constant/value";

    if (node->hasClassification(mx::ShaderNode::Classification::GEOMETRIC))
    {
        const mx::ShaderInput* geomPropInput = node->getInput("geomprop");
        if (geomPropInput && geomPropInput->getValue())
        {
            std::string prop = geomPropInput->getValue()->getValueString();
            REQUIRE(results.count(prop));
            REQUIRE(results[prop] == geomPropInput->getPath());
        }
        else
        {
            const mx::ShaderInput* indexIput = node->getInput("index");
            if (indexIput && indexIput->getValue())
            {
                std::string prop = indexIput->getValue()->getValueString();
                REQUIRE(results.count(prop));
                REQUIRE(results[prop] == indexIput->getPath());
            }
        }
    }
}

TEST_CASE("GenShader: Track Application Variables", "[genshader]")
{
    std::string testDocumentString = 
    "<?xml version=\"1.0\"?> \
      <materialx version=\"1.38\"> \
      <geompropvalue name=\"geompropvalue\" type=\"color3\" >  \
        <input name=\"geomprop\" type=\"string\" uniform=\"true\" nodename=\"constant\" /> \
      </geompropvalue> \
      <geompropvalue name=\"geompropvalue1\" type=\"color3\" > \
        <input name=\"geomprop\" type=\"string\" uniform=\"true\" value=\"primvar_one\" /> \
      </geompropvalue> \
      <geompropvalue name=\"geompropvalue2\" type=\"color3\" > \
        <input name=\"geomprop\" type=\"string\" uniform=\"true\" value=\"primvar_two\" /> \
      </geompropvalue> \
      <multiply name=\"multiply\" type=\"color3\" > \
        <input name=\"in1\" type=\"color3\" nodename=\"geompropvalue\" /> \
        <input name=\"in2\" type=\"color3\" nodename=\"geompropvalue1\" /> \
      </multiply> \
      <add name=\"add\" type=\"color3\"  > \
        <input name=\"in1\" type=\"color3\" nodename=\"multiply\" /> \
        <input name=\"in2\" type=\"color3\" nodename=\"geompropvalue2\" /> \
      </add> \
      <standard_surface name=\"standard_surface\" type=\"surfaceshader\" > \
        <input name=\"base_color\" type=\"color3\" nodename=\"add\" /> \
      </standard_surface> \
      <constant name=\"constant\" type=\"string\" > \
        <input name=\"value\" type=\"string\" uniform=\"true\" value=\"upstream_primvar\" /> \
      </constant> \
      <surfacematerial name=\"surfacematerial\" type=\"material\" > \
        <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface\" /> \
      </surfacematerial> \
    </materialx>";

    const mx::string testElement = "surfacematerial";

    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::DocumentPtr libraries = mx::createDocument();
    mx::loadLibraries({ "libraries" }, searchPath, libraries);

    mx::DocumentPtr testDoc = mx::createDocument();
    mx::readFromXmlString(testDoc, testDocumentString);
    testDoc->importLibrary(libraries);

    mx::ElementPtr element = testDoc->getChild(testElement);
    CHECK(element);

#ifdef MATERIALX_BUILD_GEN_GLSL
    {
        mx::GenContext context(mx::GlslShaderGenerator::create());
        context.registerSourceCodeSearchPath(searchPath);
        context.setApplicationVariableHandler(variableTracker);
        mx::ShaderPtr shader = context.getShaderGenerator().generate(testElement, element, context);
    }
#endif
#ifdef MATERIALX_BUILD_GEN_OSL
    {
        mx::GenContext context(mx::OslShaderGenerator::create());
        context.registerSourceCodeSearchPath(searchPath);
        context.setApplicationVariableHandler(variableTracker);
        mx::ShaderPtr shader = context.getShaderGenerator().generate(testElement, element, context);
    }
#endif
#ifdef MATERIALX_BUILD_GEN_MDL
    {
        mx::GenContext context(mx::MdlShaderGenerator::create());
        context.registerSourceCodeSearchPath(searchPath);
        context.setApplicationVariableHandler(variableTracker);
        mx::ShaderPtr shader = context.getShaderGenerator().generate(testElement, element, context);
    }
#endif
}
