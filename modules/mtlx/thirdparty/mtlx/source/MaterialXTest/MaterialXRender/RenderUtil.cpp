//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>
#include <MaterialXTest/MaterialXRender/RenderUtil.h>

#include <MaterialXFormat/Util.h>

namespace mx = MaterialX;

namespace RenderUtil
{

ShaderRenderTester::ShaderRenderTester(mx::ShaderGeneratorPtr shaderGenerator) :
    _shaderGenerator(shaderGenerator),
    _resolveImageFilenames(false),
    _emitColorTransforms(true)
{
}

ShaderRenderTester::~ShaderRenderTester()
{
}

// Create a list of generation options based on unit test options
// These options will override the original generation context options.
void ShaderRenderTester::getGenerationOptions(const GenShaderUtil::TestSuiteOptions& testOptions,
                                              const mx::GenOptions& originalOptions,
                                              std::vector<mx::GenOptions>& optionsList)
{
    optionsList.clear();
    if (testOptions.shaderInterfaces & 1)
    {
        mx::GenOptions reducedOption = originalOptions;
        reducedOption.shaderInterfaceType = mx::SHADER_INTERFACE_REDUCED;
        optionsList.push_back(reducedOption);
    }
    // Alway fallback to complete if no options specified.
    if ((testOptions.shaderInterfaces & 2) || optionsList.empty())
    {
        mx::GenOptions completeOption = originalOptions;
        completeOption.shaderInterfaceType = mx::SHADER_INTERFACE_COMPLETE;
        optionsList.push_back(completeOption);
    }
}

void ShaderRenderTester::printRunLog(const RenderProfileTimes &profileTimes,
                                     const GenShaderUtil::TestSuiteOptions& options,
                                     std::ostream& stream,
                                     mx::DocumentPtr dependLib)
{
    profileTimes.print(stream);

    stream << "---------------------------------------" << std::endl;
    options.print(stream);
}

void ShaderRenderTester::loadDependentLibraries(GenShaderUtil::TestSuiteOptions options, mx::FileSearchPath searchPath, mx::DocumentPtr& dependLib)
{
    dependLib = mx::createDocument();

    mx::loadLibraries({ "libraries" }, searchPath, dependLib);
    for (size_t i = 0; i < options.extraLibraryPaths.size(); i++)
    {
        const mx::FilePath& libraryPath = options.extraLibraryPaths[i];
        for (const mx::FilePath& libraryFile : libraryPath.getFilesInDirectory("mtlx"))
        {
            std::cout << "Extra library path: " << (libraryPath / libraryFile).asString() << std::endl;
            mx::loadLibrary((libraryPath / libraryFile), dependLib);
        }
    }

    // Load any addition per renderer libraries
    loadAdditionalLibraries(dependLib, options);
}

bool ShaderRenderTester::validate(const mx::FilePath optionsFilePath)
{
#ifdef LOG_TO_FILE
    std::ofstream logfile(_shaderGenerator->getTarget() + "_render_log.txt");
    std::ostream& log(logfile);
    std::string docValidLogFilename = _shaderGenerator->getTarget() + "_render_doc_validation_log.txt";
    std::ofstream docValidLogFile(docValidLogFilename);
    std::ostream& docValidLog(docValidLogFile);
    std::ofstream profilingLogfile(_shaderGenerator->getTarget() + "_render_profiling_log.txt");
    std::ostream& profilingLog(profilingLogfile);
#else
    std::ostream& log(std::cout);
    std::string docValidLogFilename = "std::cout";
    std::ostream& docValidLog(std::cout);
    std::ostream& profilingLog(std::cout);
#endif

    // Test has been turned off so just do nothing.
    // Check for an option file
    GenShaderUtil::TestSuiteOptions options;
    if (!options.readOptions(optionsFilePath))
    {
        log << "Can't find options file. Skip test." << std::endl;
        return false;
    }
    if (!runTest(options))
    {
        log << "Target: " << _shaderGenerator->getTarget() << " not set to run. Skip test." << std::endl;
        return false;
    }

    // Profiling times
    RenderUtil::RenderProfileTimes profileTimes;
    // Global setup timer
    mx::ScopedTimer totalTime(&profileTimes.totalTime);

    // Add files to override the files in the test suite to be tested.
    mx::StringSet testfileOverride;
    for (const auto& filterFile : options.overrideFiles)
    {
        testfileOverride.insert(filterFile);
    }

    // Data search path
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();

    mx::ScopedTimer ioTimer(&profileTimes.ioTime);
    mx::FilePathVec dirs;
    for (const auto& root : options.renderTestPaths)
    {
        mx::FilePathVec testRootDirs = searchPath.find(root).getSubDirectories();
        dirs.insert(std::end(dirs), std::begin(testRootDirs), std::end(testRootDirs));
    }
    ioTimer.endTimer();

    // Load in the library dependencies once
    // This will be imported in each test document below
    ioTimer.startTimer();
    mx::DocumentPtr dependLib;
    loadDependentLibraries(options, searchPath, dependLib);
    ioTimer.endTimer();

    // Create renderers and generators
    mx::ScopedTimer setupTime(&profileTimes.languageTimes.setupTime);

    createRenderer(log);

    mx::ColorManagementSystemPtr colorManagementSystem = mx::DefaultColorManagementSystem::create(_shaderGenerator->getTarget());
    colorManagementSystem->loadLibrary(dependLib);
    _shaderGenerator->setColorManagementSystem(colorManagementSystem);

    // Setup Unit system and working space
    mx::UnitSystemPtr unitSystem = mx::UnitSystem::create(_shaderGenerator->getTarget());
    _shaderGenerator->setUnitSystem(unitSystem);
    mx::UnitConverterRegistryPtr registry = mx::UnitConverterRegistry::create();
    mx::UnitTypeDefPtr distanceTypeDef = dependLib->getUnitTypeDef("distance");
    registry->addUnitConverter(distanceTypeDef, mx::LinearUnitConverter::create(distanceTypeDef));
    mx::UnitTypeDefPtr angleTypeDef = dependLib->getUnitTypeDef("angle");
    registry->addUnitConverter(angleTypeDef, mx::LinearUnitConverter::create(angleTypeDef));
    _shaderGenerator->getUnitSystem()->loadLibrary(dependLib);
    _shaderGenerator->getUnitSystem()->setUnitConverterRegistry(registry);

    mx::GenContext context(_shaderGenerator);
    context.registerSourceCodeSearchPath(searchPath);
    context.registerSourceCodeSearchPath(searchPath.find("libraries/stdlib/genosl/include"));

    // Set target unit space
    context.getOptions().targetDistanceUnit = "meter";

    // Set whether to emit colorspace transforms
    context.getOptions().emitColorTransforms = _emitColorTransforms;

    // Register shader metadata defined in the libraries.
    _shaderGenerator->registerShaderMetadata(dependLib, context);

    setupTime.endTimer();

    if (!options.enableDirectLighting)
    {
        context.getOptions().hwMaxActiveLightSources = 0;
    }
    registerLights(dependLib, options, context);

    // Map to replace "/" in Element path and ":" in namespaced names with "_".
    mx::StringMap pathMap;
    pathMap["/"] = "_";
    pathMap[":"] = "_";

    mx::ScopedTimer validateTimer(&profileTimes.validateTime);
    mx::ScopedTimer renderableSearchTimer(&profileTimes.renderableSearchTime);

    mx::StringSet usedImpls;

    const std::string MTLX_EXTENSION("mtlx");
    for (const auto& dir : dirs)
    {
        ioTimer.startTimer();
        mx::FilePathVec files;
        files = dir.getFilesInDirectory(MTLX_EXTENSION);
        ioTimer.endTimer();

        for (const mx::FilePath& file : files)
        {
            ioTimer.startTimer();
            // Check if a file override set is used and ignore all files
            // not part of the override set
            if (testfileOverride.size() && testfileOverride.count(file) == 0)
            {
                ioTimer.endTimer();
                continue;
            }

            const mx::FilePath filename = mx::FilePath(dir) / mx::FilePath(file);
            mx::DocumentPtr doc = mx::createDocument();
            try
            {
                mx::FileSearchPath readSearchPath(searchPath);
                readSearchPath.append(dir);
                mx::readFromXmlFile(doc, filename, readSearchPath);
            }
            catch (mx::Exception& e)
            {
                docValidLog << "Failed to load in file: " << filename.asString() << ". Error: " << e.what() << std::endl;
                WARN("Failed to load in file: " + filename.asString() + "See: " + docValidLogFilename + " for details.");
            }

            // For each new file clear the implementation cache.
            // Since the new file might contain implementations with names
            // colliding with implementations in previous test cases.
            context.clearNodeImplementations();

            doc->importLibrary(dependLib);
            ioTimer.endTimer();

            validateTimer.startTimer();
            log << "MTLX Filename: " << filename.asString() << std::endl;

            // Validate the test document
            std::string validationErrors;
            bool validDoc = doc->validate(&validationErrors);
            if (!validDoc)
            {
                docValidLog << filename.asString() << std::endl;
                docValidLog << validationErrors << std::endl;
            }
            validateTimer.endTimer();
            CHECK(validDoc);

            mx::FileSearchPath imageSearchPath(dir);
            imageSearchPath.append(searchPath);
            
            // Resolve file names if specified
            if (_resolveImageFilenames)
            {
                mx::flattenFilenames(doc, imageSearchPath, _customFilenameResolver);
            }

            mx::FilePath outputPath = mx::FilePath(dir) / file;
            outputPath.removeExtension();

            renderableSearchTimer.startTimer();
            std::vector<mx::TypedElementPtr> elements;
            try
            {
                elements = mx::findRenderableElements(doc);
            }
            catch (mx::Exception& e)
            {
                docValidLog << e.what() << std::endl;
                WARN("Shader generation error in " + filename.asString() + ": " + e.what());
            }
            renderableSearchTimer.endTimer();

            for (const auto& element : elements)
            {
                mx::string elementName = mx::createValidName(mx::replaceSubstrings(element->getNamePath(), pathMap));
                runRenderer(elementName, element, context, doc, log, options, profileTimes, imageSearchPath, outputPath, nullptr);
            }
        }
    }

    // Dump out profiling information
    totalTime.endTimer();
    printRunLog(profileTimes, options, profilingLog, dependLib);

    return true;
}

void ShaderRenderTester::addAdditionalTestStreams(mx::MeshPtr mesh)
{
    size_t vertexCount = mesh->getVertexCount();
    if (vertexCount < 1)
    {
        return;
    }

    const std::string TEXCOORD_STREAM0_NAME("i_" + mx::MeshStream::TEXCOORD_ATTRIBUTE + "_0");
    mx::MeshStreamPtr texCoordStream1 = mesh->getStream(TEXCOORD_STREAM0_NAME);
    mx::MeshFloatBuffer uv = texCoordStream1->getData();

    const std::string TEXCOORD_STREAM1_NAME("i_" + mx::MeshStream::TEXCOORD_ATTRIBUTE + "_1");
    mx::MeshFloatBuffer* texCoordData2 = nullptr;
    if (!mesh->getStream(TEXCOORD_STREAM1_NAME))
    {
        mx::MeshStreamPtr texCoordStream2 = mx::MeshStream::create(TEXCOORD_STREAM1_NAME, mx::MeshStream::TEXCOORD_ATTRIBUTE, 1);
        texCoordStream2->setStride(2);
        texCoordData2 = &(texCoordStream2->getData());
        texCoordData2->resize(vertexCount * 2);
        mesh->addStream(texCoordStream2);
    }

    const std::string COLOR_STREAM0_NAME("i_" + mx::MeshStream::COLOR_ATTRIBUTE + "_0");
    mx::MeshFloatBuffer* colorData1 = nullptr;
    if (!mesh->getStream(COLOR_STREAM0_NAME))
    {
        mx::MeshStreamPtr colorStream1 = mx::MeshStream::create(COLOR_STREAM0_NAME, mx::MeshStream::COLOR_ATTRIBUTE, 0);
        colorData1 = &(colorStream1->getData());
        colorStream1->setStride(4);
        colorData1->resize(vertexCount * 4);
        mesh->addStream(colorStream1);
    }

    const std::string COLOR_STREAM1_NAME("i_" + mx::MeshStream::COLOR_ATTRIBUTE + "_1");
    mx::MeshFloatBuffer* colorData2 = nullptr;
    if (!mesh->getStream(COLOR_STREAM1_NAME))
    {
        mx::MeshStreamPtr colorStream2 = mx::MeshStream::create(COLOR_STREAM1_NAME, mx::MeshStream::COLOR_ATTRIBUTE, 1);
        colorData2 = &(colorStream2->getData());
        colorStream2->setStride(4);
        colorData2->resize(vertexCount * 4);
        mesh->addStream(colorStream2);
    }

    const std::string GEOM_INT_STREAM_NAME("i_" + mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE + "_geompropvalue_integer");
    int32_t* geomIntData = nullptr;
    if (!mesh->getStream(GEOM_INT_STREAM_NAME))
    {
        mx::MeshStreamPtr geomIntStream = mx::MeshStream::create(GEOM_INT_STREAM_NAME, mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE, 0);
        geomIntStream->setStride(1);
        geomIntStream->getData().resize(vertexCount);
        mesh->addStream(geomIntStream);
        // Float and int32 have same size.
        geomIntData = reinterpret_cast<int32_t*>(geomIntStream->getData().data());
    }

    const std::string GEOM_FLOAT_STREAM_NAME("i_" + mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE + "_geompropvalue_float");
    mx::MeshFloatBuffer* geomFloatData = nullptr;
    if (!mesh->getStream(GEOM_FLOAT_STREAM_NAME))
    {
        mx::MeshStreamPtr geomFloatStream = mx::MeshStream::create(GEOM_FLOAT_STREAM_NAME, mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE, 1);
        geomFloatData = &(geomFloatStream->getData());
        geomFloatStream->setStride(1);
        geomFloatData->resize(vertexCount);
        mesh->addStream(geomFloatStream);
    }

    const std::string GEOM_VECTOR2_STREAM_NAME("i_" + mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE + "_geompropvalue_vector2");
    mx::MeshFloatBuffer* geomVector2Data = nullptr;
    if (!mesh->getStream(GEOM_VECTOR2_STREAM_NAME))
    {
        mx::MeshStreamPtr geomVector2Stream = mx::MeshStream::create(GEOM_VECTOR2_STREAM_NAME, mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE, 1);
        geomVector2Data = &(geomVector2Stream->getData());
        geomVector2Stream->setStride(2);
        geomVector2Data->resize(vertexCount * 2);
        mesh->addStream(geomVector2Stream);
    }

    const std::string GEOM_VECTOR3_STREAM_NAME("i_" + mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE + "_geompropvalue_vector3");
    mx::MeshFloatBuffer* geomVector3Data = nullptr;
    if (!mesh->getStream(GEOM_VECTOR3_STREAM_NAME))
    {
        mx::MeshStreamPtr geomVector3Stream = mx::MeshStream::create(GEOM_VECTOR3_STREAM_NAME, mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE, 1);
        geomVector3Data = &(geomVector3Stream->getData());
        geomVector3Stream->setStride(3);
        geomVector3Data->resize(vertexCount * 3);
        mesh->addStream(geomVector3Stream);
    }

    const std::string GEOM_VECTOR4_STREAM_NAME("i_" + mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE + "_geompropvalue_vector4");
    mx::MeshFloatBuffer* geomVector4Data = nullptr;
    if (!mesh->getStream(GEOM_VECTOR4_STREAM_NAME))
    {
        mx::MeshStreamPtr geomVector4Stream = mx::MeshStream::create(GEOM_VECTOR4_STREAM_NAME, mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE, 1);
        geomVector4Data = &(geomVector4Stream->getData());
        geomVector4Stream->setStride(4);
        geomVector4Data->resize(vertexCount * 4);
        mesh->addStream(geomVector4Stream);
    }

    const std::string GEOM_COLOR2_STREAM_NAME("i_" + mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE + "_geompropvalue_color2");
    mx::MeshFloatBuffer* geomColor2Data = nullptr;
    if (!mesh->getStream(GEOM_COLOR2_STREAM_NAME))
    {
        mx::MeshStreamPtr geomColor2Stream = mx::MeshStream::create(GEOM_COLOR2_STREAM_NAME, mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE, 1);
        geomColor2Data = &(geomColor2Stream->getData());
        geomColor2Stream->setStride(2);
        geomColor2Data->resize(vertexCount * 2);
        mesh->addStream(geomColor2Stream);
    }

    const std::string GEOM_COLOR3_STREAM_NAME("i_" + mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE + "_geompropvalue_color3");
    mx::MeshFloatBuffer* geomColor3Data = nullptr;
    if (!mesh->getStream(GEOM_COLOR3_STREAM_NAME))
    {
        mx::MeshStreamPtr geomColor3Stream = mx::MeshStream::create(GEOM_COLOR3_STREAM_NAME, mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE, 1);
        geomColor3Data = &(geomColor3Stream->getData());
        geomColor3Stream->setStride(3);
        geomColor3Data->resize(vertexCount * 3);
        mesh->addStream(geomColor3Stream);
    }

    const std::string GEOM_COLOR4_STREAM_NAME("i_" + mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE + "_geompropvalue_color4");
    mx::MeshFloatBuffer* geomColor4Data = nullptr;
    if (!mesh->getStream(GEOM_COLOR4_STREAM_NAME))
    {
        mx::MeshStreamPtr geomColor4Stream = mx::MeshStream::create(GEOM_COLOR4_STREAM_NAME, mx::MeshStream::GEOMETRY_PROPERTY_ATTRIBUTE, 1);
        geomColor4Data = &(geomColor4Stream->getData());
        geomColor4Stream->setStride(4);
        geomColor4Data->resize(vertexCount * 4);
        mesh->addStream(geomColor4Stream);
    }

    auto sineData = [](float uv, float freq){
        const float PI = std::acos(-1.0f);
        float angle = uv * 2 * PI * freq;
        return std::sin(angle) / 2.0f + 1.0f;
    };
    if (!uv.empty())
    {
        for (size_t i = 0; i < vertexCount; i++)
        {
            const size_t i2 = 2 * i;
            const size_t i21 = i2 + 1;
            const size_t i3 = 3 * i;
            const size_t i4 = 4 * i;

            // Fake second set of texture coordinates
            if (texCoordData2)
            {
                (*texCoordData2)[i2] = uv[i21];
                (*texCoordData2)[i21] = uv[i2];
            }
            if (colorData1)
            {
                // Fake some colors
                (*colorData1)[i4] = uv[i2];
                (*colorData1)[i4 + 1] = uv[i21];
                (*colorData1)[i4 + 2] = 1.0f;
                (*colorData1)[i4 + 3] = 1.0f;
            }
            if (colorData2)
            {
                (*colorData2)[i4] = 1.0f;
                (*colorData2)[i4 + 1] = uv[i2];
                (*colorData2)[i4 + 2] = uv[i21];
                (*colorData2)[i4 + 3] = 1.0f;
            }
            if (geomIntData)
            {
                geomIntData[i] = static_cast<int32_t>(uv[i21] * 5);
            }
            if (geomFloatData)
            {
                (*geomFloatData)[i] = sineData(uv[i21], 12.0f);
            }
            if (geomVector2Data)
            {
                (*geomVector2Data)[i2] = sineData(uv[i21], 6.0f);
                (*geomVector2Data)[i21] = 0.0f;
            }
            if (geomVector3Data)
            {
                (*geomVector3Data)[i3] = 0.0f;
                (*geomVector3Data)[i3 + 1] = sineData(uv[i21], 8.0f);
                (*geomVector3Data)[i3 + 2] = 0.0f;
            }
            if (geomVector4Data)
            {
                (*geomVector4Data)[i4] = 0.0f;
                (*geomVector4Data)[i4 + 1] = 0.0f;
                (*geomVector4Data)[i4 + 2] = sineData(uv[i21], 10.0f);
                (*geomVector4Data)[i4 + 3] = 1.0f;
            }

            if (geomColor2Data)
            {
                (*geomColor2Data)[i2] = sineData(uv[i2], 10.0f);
                (*geomColor2Data)[i21] = 0.0f;
            }
            if (geomColor3Data)
            {
                (*geomColor3Data)[i3] = 0.0f;
                (*geomColor3Data)[i3 + 1] = sineData(uv[i2], 8.0f);
                (*geomColor3Data)[i3 + 2] = 0.0f;
            }
            if (geomColor4Data)
            {
                (*geomColor4Data)[i4] = 0.0f;
                (*geomColor4Data)[i4 + 1] = 0.0f;
                (*geomColor4Data)[i4 + 2] = sineData(uv[i2], 6.0f);
                (*geomColor4Data)[i4 + 3] = 1.0f;
            }
        }
    }
}

} // namespace RenderUtil
