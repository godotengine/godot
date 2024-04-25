//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>
#include <MaterialXTest/MaterialXRender/RenderUtil.h>

#include <MaterialXRender/ShaderRenderer.h>
#include <MaterialXRender/StbImageLoader.h>
#include <MaterialXRender/TinyObjLoader.h>
#include <MaterialXRender/Types.h>

#include <MaterialXFormat/Util.h>

#ifdef MATERIALX_BUILD_OIIO
#include <MaterialXRender/OiioImageLoader.h>
#endif

#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_set>

namespace mx = MaterialX;

TEST_CASE("Render: Half Float", "[rendercore]")
{
    const std::vector<float> exactValues =
    {
        0.0f, 0.25f, 0.5f, 0.75f,
        1.0f, 8.0f, 64.0f, 512.0f,
        std::numeric_limits<float>::infinity()
    };
    const std::vector<float> nearValues =
    {
        1.0f / 3.0f, 1.0f / 5.0f, 1.0f / 7.0f,
        std::numeric_limits<float>::denorm_min()
    };
    const std::vector<float> signs = { 1.0f, -1.0f };

    // Test values with exact equivalence as float and half.
    for (float value : exactValues)
    {
        for (float sign : signs)
        {
            float f(value * sign);
            mx::Half h(f);
            REQUIRE(h == f);
            REQUIRE(h + mx::Half(1.0f) == f + 1.0f);
            REQUIRE(h - mx::Half(1.0f) == f - 1.0f);
            REQUIRE(h * mx::Half(2.0f) == f * 2.0f);
            REQUIRE(h / mx::Half(2.0f) == f / 2.0f);
            REQUIRE((h += mx::Half(3.0f)) == (f += 3.0f));
            REQUIRE((h -= mx::Half(3.0f)) == (f -= 3.0f));
            REQUIRE((h *= mx::Half(4.0f)) == (f *= 4.0f));
            REQUIRE((h /= mx::Half(4.0f)) == (f /= 4.0f));
            REQUIRE(-h == -f);
        }
    }

    // Test values with near equivalence as float and half.
    const float EPSILON = 0.001f;
    for (float value : nearValues)
    {
        for (float sign : signs)
        {
            float f(value * sign);
            mx::Half h(f);
            REQUIRE(h != f);
            REQUIRE(std::abs(h - f) < EPSILON);
        }
    }
}

struct GeomHandlerTestOptions
{
    mx::GeometryHandlerPtr geomHandler;
    std::ofstream* logFile;

    mx::StringSet testExtensions;
    mx::StringVec skipExtensions;
};

void testGeomHandler(GeomHandlerTestOptions& options)
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePath imagePath = searchPath.find("resources/Geometry/");
    mx::FilePathVec files;

    unsigned int loadFailed = 0;
    for (const std::string& extension : options.testExtensions)
    {
        if (options.skipExtensions.end() != std::find(options.skipExtensions.begin(), options.skipExtensions.end(), extension))
        {
            continue;
        }
        files = imagePath.getFilesInDirectory(extension);
        for (const mx::FilePath& file : files)
        {
            const mx::FilePath filePath = imagePath / file;
            bool loaded = options.geomHandler->loadGeometry(filePath);
            if (options.logFile)
            {
                *(options.logFile) << "Loaded image: " << filePath.asString() << ". Loaded: " << loaded << std::endl;
            }
            if (!loaded)
            {
                loadFailed++;
            }
        }
    }
    CHECK(loadFailed == 0);
}

TEST_CASE("Render: Geometry Handler Load", "[rendercore]")
{
    std::ofstream geomHandlerLog;
    geomHandlerLog.open("render_geom_handler_test.txt");
    bool geomLoaded = false;
    try
    {
        geomHandlerLog << "** Test TinyOBJ geom loader **" << std::endl;
        mx::TinyObjLoaderPtr loader = mx::TinyObjLoader::create();
        mx::GeometryHandlerPtr handler = mx::GeometryHandler::create();
        handler->addLoader(loader);

        GeomHandlerTestOptions options;
        options.logFile = &geomHandlerLog;
        options.geomHandler = handler;
        handler->supportedExtensions(options.testExtensions);
        testGeomHandler(options);

        geomLoaded = true;
    }
    catch (mx::ExceptionRenderError& e)
    {
        for (const auto& error : e.errorLog())
        {
            geomHandlerLog << e.what() << " " << error << std::endl;
        }
    }
    catch (mx::Exception& e)
    {
        std::cout << e.what();
    }
    CHECK(geomLoaded);
    geomHandlerLog.close();
}

struct ImageHandlerTestOptions
{
    mx::ImageHandlerPtr imageHandler;
    std::ofstream* logFile;

    mx::StringSet testExtensions;
    mx::StringVec skipExtensions;
};

void testImageHandler(ImageHandlerTestOptions& options)
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePath imagePath = searchPath.find("resources/Images/");
    mx::FilePathVec files;

    unsigned int loadFailed = 0;
    for (const std::string& extension : options.testExtensions)
    {
        if (options.skipExtensions.end() != std::find(options.skipExtensions.begin(), options.skipExtensions.end(), extension))
        {
            continue;
        }
        files = imagePath.getFilesInDirectory(extension);
        for (const mx::FilePath& file : files)
        {
            const mx::FilePath filePath = imagePath / file;
            mx::ImagePtr image = options.imageHandler->acquireImage(filePath);
            CHECK(image);
            image->releaseResourceBuffer();
            CHECK(!image->getResourceBuffer());
            if (options.logFile)
            {
                *(options.logFile) << "Loaded image: " << filePath.asString() << ". Loaded: " << (bool) image << std::endl;
            }
            if (!image)
            {
                loadFailed++;
            }
        }
    }
    CHECK(loadFailed == 0);
}

TEST_CASE("Render: Image Handler Load", "[rendercore]")
{
    std::ofstream imageHandlerLog;
    imageHandlerLog.open("render_image_handler_test.txt");
    bool imagesLoaded = false;
    try
    {
        mx::Color4 color(1.0f, 0.0f, 0.0f, 1.0f);
        mx::ImagePtr uniformImage = createUniformImage(1, 1, 4, mx::Image::BaseType::UINT8, color);
        CHECK(uniformImage->getWidth() == 1);
        CHECK(uniformImage->getHeight() == 1);
        CHECK(uniformImage->getMaxMipCount() == 1);
        CHECK(uniformImage->getTexelColor(0, 0) == color);

        mx::ImageHandlerPtr imageHandler = mx::ImageHandler::create(nullptr);
        ImageHandlerTestOptions options;
        options.logFile = &imageHandlerLog;

        imageHandlerLog << "** Test STB image loader **" << std::endl;
        mx::StbImageLoaderPtr stbLoader = mx::StbImageLoader::create();
        imageHandler->addLoader(stbLoader);
        options.testExtensions = stbLoader->supportedExtensions();
        options.imageHandler = imageHandler;
        testImageHandler(options);

#if defined(MATERIALX_BUILD_OIIO)
        imageHandlerLog << "** Test OpenImageIO image loader **" << std::endl;
        mx::OiioImageLoaderPtr oiioLoader = mx::OiioImageLoader::create();
        mx::ImageHandlerPtr imageHandler3 = mx::ImageHandler::create(nullptr);
        imageHandler3->addLoader(oiioLoader);
        options.testExtensions = oiioLoader->supportedExtensions();
        options.imageHandler = imageHandler3;
        // Getting libpng warning: iCCP: known incorrect sRGB profile for some reason. TBD.
        options.skipExtensions.push_back("gif");
        testImageHandler(options);
#endif
        imagesLoaded = true;
    }
    catch (mx::ExceptionRenderError& e)
    {
        for (const auto& error : e.errorLog())
        {
            imageHandlerLog << e.what() << " " << error << std::endl;
        }
    }
    catch (mx::Exception& e)
    {
        std::cout << e.what();
    }
    CHECK(imagesLoaded);
    imageHandlerLog.close();
}
