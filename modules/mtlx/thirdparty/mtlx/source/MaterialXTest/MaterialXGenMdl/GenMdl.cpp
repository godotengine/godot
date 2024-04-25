//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>
#include <MaterialXTest/MaterialXGenMdl/GenMdl.h>

#include <MaterialXCore/Document.h>

#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <MaterialXGenMdl/MdlShaderGenerator.h>
#include <MaterialXGenMdl/MdlSyntax.h>

#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Util.h>


namespace mx = MaterialX;

TEST_CASE("GenShader: MDL Syntax", "[genmdl]")
{
    mx::SyntaxPtr syntax = mx::MdlSyntax::create();

    REQUIRE(syntax->getTypeName(mx::Type::FLOAT) == "float");
    REQUIRE(syntax->getTypeName(mx::Type::COLOR3) == "color");
    REQUIRE(syntax->getTypeName(mx::Type::VECTOR3) == "float3");
    REQUIRE(syntax->getTypeName(mx::Type::FLOATARRAY) == "float");
    REQUIRE(syntax->getTypeName(mx::Type::INTEGERARRAY) == "int");
    REQUIRE(mx::Type::FLOATARRAY->isArray());
    REQUIRE(mx::Type::INTEGERARRAY->isArray());

    REQUIRE(syntax->getTypeName(mx::Type::BSDF) == "material");
    REQUIRE(syntax->getOutputTypeName(mx::Type::BSDF) == "material");

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
    REQUIRE(value == "mk_color4(0.0)");
    value = syntax->getDefaultValue(mx::Type::COLOR4, true);
    REQUIRE(value == "mk_color4(0.0)");
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
    REQUIRE(value == "mk_color4(1.0, 2.0, 3.0, 4.0)");
    value = syntax->getValue(mx::Type::COLOR4, *color4Value, true);
    REQUIRE(value == "mk_color4(1.0, 2.0, 3.0, 4.0)");

    std::vector<float> floatArray = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f };
    mx::ValuePtr floatArrayValue = mx::Value::createValue<std::vector<float>>(floatArray);
    value = syntax->getValue(mx::Type::FLOATARRAY, *floatArrayValue);
    REQUIRE(value == "float[](0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)");

    std::vector<int> intArray = { 1, 2, 3, 4, 5, 6, 7 };
    mx::ValuePtr intArrayValue = mx::Value::createValue<std::vector<int>>(intArray);
    value = syntax->getValue(mx::Type::INTEGERARRAY, *intArrayValue);
    REQUIRE(value == "int[](1, 2, 3, 4, 5, 6, 7)");
}


TEST_CASE("GenShader: MDL Implementation Check", "[genmdl]")
{
    mx::GenContext context(mx::MdlShaderGenerator::create());

    mx::StringSet generatorSkipNodeTypes;
    generatorSkipNodeTypes.insert("light");
    mx::StringSet generatorSkipNodeDefs;

    GenShaderUtil::checkImplementations(context, generatorSkipNodeTypes, generatorSkipNodeDefs, 48);
}


class MdlStringResolver : public mx::StringResolver
{
  public:
    /// Create a new string resolver.
    static MdlStringResolverPtr create()
    {
        return MdlStringResolverPtr(new MdlStringResolver());
    }
    ~MdlStringResolver() = default;

    void initialize(
        mx::DocumentPtr document,
        std::ofstream* logFile,
        std::initializer_list<mx::FilePath> additionalSearchpaths)
    {
        mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
        mx::FilePath rootPath = searchPath.isEmpty() ? mx::FilePath() : searchPath[0];
        mx::FilePath coreModulePath = rootPath / std::string(MATERIALX_INSTALL_MDL_MODULE_PATH) / "mdl";
        mx::FilePath coreModulePath2 = coreModulePath / mx::FilePath("materialx");

        // use the source search paths as base
        mx::FileSearchPath paths  = mx::getSourceSearchPath(document);
        paths.append(mx::FilePath(document->getSourceUri()).getParentPath());

        // paths specified by the build system
        paths.append(mx::FilePath(MATERIALX_MDL_IMPL_MODULE_PATH));
        mx::StringVec extraModulePaths = mx::splitString(MATERIALX_MDL_MODULE_PATHS, ",");
        for (const std::string& extraPath : extraModulePaths)
        {
            paths.append(mx::FilePath(extraPath));
        }

        // add additional search paths for the tests
        paths.append(rootPath);
        paths.append(coreModulePath);
        paths.append(coreModulePath2);
        for (const auto& addSp : additionalSearchpaths)
        {
            paths.append(addSp);
        }

        _mdl_searchPaths.clear();
        for (const auto& path : paths)
        {
            // normalize all search paths, as we need this later in `resolve`
            auto normalizedPath = path.getNormalized();
            if (normalizedPath.exists())
                _mdl_searchPaths.append(normalizedPath);
        }

        _logFile = logFile;
    }

    std::string resolve(const std::string& str, const std::string&) const override
    {
        mx::FilePath normalizedPath = mx::FilePath(str).getNormalized();

        // in case the path is absolute we need to find a proper search path to put the file in
        if (normalizedPath.isAbsolute())
        {
            // find the highest priority search path that is a prefix of the resource path
            for (const auto& sp : _mdl_searchPaths)
            {
                if (sp.size() > normalizedPath.size())
                    continue;

                bool isParent = true;
                for (size_t i = 0; i < sp.size(); ++i)
                {
                    if (sp[i] != normalizedPath[i])
                    {
                        isParent = false;
                        break;
                    }
                }

                if (!isParent)
                    continue;

                // found a search path that is a prefix of the resource
                std::string resource_path = normalizedPath.asString().substr(sp.asString().size());
                if (resource_path[0] != '/')
                    resource_path = "/" + resource_path;
                return resource_path;
            }
        }

        *_logFile << "MaterialX resource can not be accessed through an MDL search path. "
            << "Dropping the resource from the Material. Resource Path: "
            << normalizedPath.asString().c_str() << std::endl;

        // drop the resource by returning the empty string.
        // alternatively, the resource could be copied into an MDL search path,
        // maybe even only temporary.
        return "";
    }

    const  mx::FileSearchPath& getMdlSearchPaths() const { return _mdl_searchPaths; }

  private:
    // list of MDL search paths from which we can locate resources.
    mx::FileSearchPath _mdl_searchPaths;

    // log file of the tester
    std::ofstream* _logFile;
};

void MdlShaderGeneratorTester::preprocessDocument(mx::DocumentPtr doc)
{
    if (!_mdlCustomResolver)
        _mdlCustomResolver = MdlStringResolver::create();

    _mdlCustomResolver->initialize(doc, &_logFile, { _searchPath.asString() });
    mx::flattenFilenames(doc, _mdlCustomResolver->getMdlSearchPaths(), _mdlCustomResolver);
}

void MdlShaderGeneratorTester::compileSource(const std::vector<mx::FilePath>& sourceCodePaths)
{
    if (sourceCodePaths.empty() || sourceCodePaths[0].isEmpty())
        return;

    mx::FilePath moduleToTestPath = sourceCodePaths[0].getParentPath();
    mx::FilePath module = sourceCodePaths[0];
    std::string moduleToTest = module[module.size()-1];
    moduleToTest = moduleToTest.substr(0, moduleToTest.size() - sourceCodePaths[0].getExtension().length() - 1);

    std::string renderExec(MATERIALX_MDL_RENDER_EXECUTABLE);
    bool testMDLC = renderExec.empty();
    if (testMDLC)
    {
        std::string mdlcExec(MATERIALX_MDLC_EXECUTABLE);
        if (mdlcExec.empty())
        {
            return;
        }

        std::string mdlcCommand = mdlcExec;

        // use the same paths as the resolver
        for (const auto& sp : _mdlCustomResolver->getMdlSearchPaths())
        {
            mdlcCommand += " -p \"" + sp.asString() + "\"";
        }

        // additionally the generated module needs to found in a search path too
        mdlcCommand += " -p \"" + moduleToTestPath.asString() + "\"";
        mdlcCommand += " -p \"" + moduleToTestPath.getParentPath().asString() + "\"";

        mdlcCommand += " -W \"181=off\" -W \"183=off\"  -W \"225=off\"";
        mdlcCommand += " " + moduleToTest;
        mx::FilePath errorFile = moduleToTestPath / (moduleToTest + ".mdl_compile_errors.txt");
        mdlcCommand += " > " + errorFile.asString() + " 2>&1";

        int returnValue = std::system(mdlcCommand.c_str());
        std::ifstream errorStream(errorFile);
        mx::StringVec result;
        std::string line;
        bool writeErrorCode = false;
        while (std::getline(errorStream, line))
        {
            if (!writeErrorCode)
            {
                _logFile << mdlcCommand << std::endl;
                _logFile << "\tReturn code: " << std::to_string(returnValue) << std::endl;
                writeErrorCode = true;
            }
            _logFile << "\tError: " << line << std::endl;
        }

        CHECK(returnValue == 0);
    }
    else
    {
        std::string renderCommand = renderExec;

        // use the same paths as the resolver
        for (const auto& sp : _mdlCustomResolver->getMdlSearchPaths())
        {
            renderCommand += " --mdl_path \"" + sp.asString() + "\"";
        }

        // additionally the generated module needs to found in a search path too
        renderCommand += " --mdl_path \"" + moduleToTestPath.asString() + "\"";
        renderCommand += " --mdl_path \"" + moduleToTestPath.getParentPath().asString() + "\"";

        mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
        mx::FilePath rootPath = searchPath.isEmpty() ? mx::FilePath() : searchPath[0];
        // set environment
        std::string iblFile = (rootPath / "resources/lights/san_giuseppe_bridge.hdr").asString();
        renderCommand += " --hdr \"" + iblFile + "\" --hdr_rotate 90";
        // set scene
        renderCommand += " --uv_scale 0.5 1.0 --uv_offset 0.0 0.0 --uv_repeat";
        renderCommand += " --uv_flip"; // this will flip the v coordinate of the vertices, which flips all the
                                       // UV operations. In contrast, the fileTextureVerticalFlip option will
                                       // only flip the image access nodes.
        renderCommand += " --camera 0 0 3 0 0 0 --fov 45";

        // set the material
        // compute the MDL module name as fully qualified name wrt to the "rootPath/resources" as MDL search path
        std::string mdlModuleName = "::resources::";
        for (size_t s = rootPath.size() + 1; s < moduleToTestPath.size(); ++s)
        {
            mdlModuleName += moduleToTestPath[s] + "::";
        }
        mdlModuleName += moduleToTest;
        renderCommand += " --mat " + mdlModuleName + "::*";

        // This must be a render args option. Rest are consistent between dxr and cuda example renderers.
        std::string renderArgs(MATERIALX_MDL_RENDER_ARGUMENTS);
        if (renderArgs.empty())
        {
            // Assume MDL example DXR is being used and set reasonable arguments automatically
            renderCommand += " --nogui --res 512 512 --iterations 1024 --max_path_length 3 --noaux --no_firefly_clamp";
            renderCommand += " --background 0.073239 0.073239 0.083535";
        }
        else
        {
            renderCommand += " " + renderArgs;
        }

        std::string extension("_mdl.png");
#if defined(MATERIALX_BUILD_OIIO)
        extension = "_mdl.exr";
#endif
        // drop the `.genmdl` in order to have filenames supported by the image comparison
        std::string imageFilename = moduleToTest.substr(0, moduleToTest.size() - 7);

        mx::FilePath outputImageName = moduleToTestPath / (imageFilename + extension);

        renderCommand += " -o " + outputImageName.asString();
        mx::FilePath logFile = moduleToTestPath / (moduleToTest + ".mdl_render_log.txt");
        renderCommand += " --log_file " + logFile.asString();
        mx::FilePath errorLogFile = moduleToTestPath / (moduleToTest + ".mdl_render_errors.txt");
        int returnValue = std::system(renderCommand.c_str());
        std::ifstream logStream(errorLogFile);
        mx::StringVec result;
        std::string line;
        bool writeLogCode = false;
        while (std::getline(logStream, line))
        {
            if (!writeLogCode)
            {
                _logFile << renderCommand << std::endl;
                _logFile << "\tReturn code: " << std::to_string(returnValue) << std::endl;
                writeLogCode = true;
            }
            _logFile << "\tLog: " << line << std::endl;
        }

        CHECK(returnValue == 0);
    }
}

TEST_CASE("GenShader: MDL Shader Generation", "[genmdl]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();

    mx::FilePathVec testRootPaths;
    testRootPaths.push_back(searchPath.find("resources/Materials/TestSuite"));
    testRootPaths.push_back(searchPath.find("resources/Materials/Examples/StandardSurface"));

    const mx::FilePath logPath("genmdl_mdl_generate_test.txt");

    // Write shaders and try to compile only if mdlc exe specified.
    std::string mdlcExec(MATERIALX_MDLC_EXECUTABLE);
    bool writeShadersToDisk = !mdlcExec.empty();
    MdlShaderGeneratorTester tester(mx::MdlShaderGenerator::create(), testRootPaths, searchPath, logPath, writeShadersToDisk);
    tester.addSkipLibraryFiles();

    mx::GenOptions genOptions;
    genOptions.targetColorSpaceOverride = "lin_rec709";

    // Flipping the texture lookups for the test renderer only.
    // This is because OSL testrender does not allow to change the UV layout of their sphere (yet) and the MaterialX test suite
    // adopts the OSL behavior in order to produce comparable results. This means that raw texture coordinates, or procedurals
    // that use the texture coordinates, do not match what might be expected when reading the MaterialX spec:
    //    "[...] the image is mapped onto the geometry based on geometry UV coordinates, with the lower-left corner of an image 
    //     mapping to the (0,0) UV coordinate [...]"
    // This means for MDL: here, and only here in the test suite, we flip the UV coordinates of mesh using the `--uv_flip` option
    // of the renderer, and to correct the image orientation, we apply `fileTextureVerticalFlip`.
    // In regular MDL integrations this is not needed because MDL and MaterialX define the texture space equally with the origin
    // at the bottom left.
    genOptions.fileTextureVerticalFlip = true;

    mx::FilePath optionsFilePath = searchPath.find("resources/Materials/TestSuite/_options.mtlx");

    // Specify the MDL target version to be the latest which is also the default.
    mx::GenMdlOptionsPtr genMdlOptions = std::make_shared<mx::GenMdlOptions>();
    genMdlOptions->targetVersion = mx::GenMdlOptions::MdlVersion::MDL_LATEST;
    tester.addUserData(mx::GenMdlOptions::GEN_CONTEXT_USER_DATA_KEY, genMdlOptions);

    tester.validate(genOptions, optionsFilePath);
}
