//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

namespace mx = MaterialX;

TEST_CASE("Syntactic operations", "[file]")
{
    using InputPair = std::pair<std::string, mx::FilePath::Format>;
    std::vector<InputPair> inputPairs =
    {
        {"D:\\Assets\\Materials\\Robot.mtlx", mx::FilePath::FormatWindows},
        {"\\\\Show\\Assets\\Materials\\Robot.mtlx", mx::FilePath::FormatWindows},
        {"Materials\\Robot.mtlx", mx::FilePath::FormatWindows},
        {"/Assets/Materials/Robot.mtlx", mx::FilePath::FormatPosix},
        {"Assets/Materials/Robot.mtlx", mx::FilePath::FormatPosix},
        {"Materials/Robot.mtlx", mx::FilePath::FormatPosix}
    };

    for (const InputPair& pair : inputPairs)
    {
        mx::FilePath path(pair.first);
        REQUIRE(path.asString(pair.second) == pair.first);
    }
}

TEST_CASE("File system operations", "[file]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePathVec examplePaths =
    {
        "libraries/stdlib/stdlib_defs.mtlx",
        "resources/Materials/Examples/StandardSurface/standard_surface_brass_tiled.mtlx",
        "resources/Materials/Examples/StandardSurface/standard_surface_marble_solid.mtlx",
    };
    for (const mx::FilePath& path : examplePaths)
    {
        REQUIRE(searchPath.find(path).exists());
    }

    REQUIRE(mx::FilePath::getCurrentPath().exists());
    REQUIRE(mx::FilePath::getModulePath().exists());
}

TEST_CASE("File search path operations", "[file]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    searchPath.append(searchPath.find("libraries/stdlib"));
    searchPath.append(searchPath.find("resources/Materials/Examples/StandardSurface"));

    mx::FilePathVec filenames =
    {
        "stdlib_defs.mtlx",
        "standard_surface_brass_tiled.mtlx",
        "standard_surface_marble_solid.mtlx",
    };

    for (const mx::FilePath& filename : filenames)
    {
        REQUIRE(searchPath.find(filename).exists());
    }
}

TEST_CASE("Flatten filenames", "[file]")
{
    const mx::FilePath TEST_FILE_PREFIX_STRING("resources\\Images\\");
    const mx::FilePath TEST_IMAGE_STRING1("brass_roughness.jpg");
    const mx::FilePath TEST_IMAGE_STRING2("brass_color.jpg");

    mx::DocumentPtr doc1 = mx::createDocument();

    // Set up document
    mx::NodeGraphPtr nodeGraph = doc1->addNodeGraph();
    nodeGraph->setFilePrefix(TEST_FILE_PREFIX_STRING.asString() + "\\"); // Note this is required as filepath->string strips out last separator
    mx::NodePtr image1 = nodeGraph->addNode("image");
    image1->setInputValue("file", "brass_roughness.jpg", mx::FILENAME_TYPE_STRING);
    mx::NodePtr image2 = nodeGraph->addNode("image");
    image2->setInputValue("file", "brass_color.jpg", mx::FILENAME_TYPE_STRING);

    // 1. Test resolving fileprefix
    mx::flattenFilenames(doc1);
    REQUIRE(nodeGraph->getFilePrefix() == mx::EMPTY_STRING);
    mx::FilePath resolvedPath(image1->getInputValue("file")->getValueString());
    REQUIRE(resolvedPath == (TEST_FILE_PREFIX_STRING / TEST_IMAGE_STRING1));
    resolvedPath = image2->getInputValue("file")->getValueString();
    REQUIRE(resolvedPath == (TEST_FILE_PREFIX_STRING / TEST_IMAGE_STRING2));

    // Reset document
    nodeGraph->setFilePrefix(TEST_FILE_PREFIX_STRING.asString() + "\\");
    image1->setInputValue("file", "brass_roughness.jpg", mx::FILENAME_TYPE_STRING);
    image2->setInputValue("file", "brass_color.jpg", mx::FILENAME_TYPE_STRING);

    // 2. Test resolving to absolute paths
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePath rootPath = searchPath.isEmpty() ? mx::FilePath() : searchPath[0];

    mx::flattenFilenames(doc1, searchPath);    
    REQUIRE(nodeGraph->getFilePrefix() == mx::EMPTY_STRING);
    resolvedPath = image1->getInputValue("file")->getValueString();
    REQUIRE(resolvedPath.asString() == (rootPath / TEST_FILE_PREFIX_STRING / TEST_IMAGE_STRING1).asString());
    resolvedPath = image2->getInputValue("file")->getValueString();
    REQUIRE(resolvedPath.asString() == (rootPath / TEST_FILE_PREFIX_STRING / TEST_IMAGE_STRING2).asString());

    // Reset document
    nodeGraph->setFilePrefix(TEST_FILE_PREFIX_STRING.asString() + "\\");
    image1->setInputValue("file", "brass_roughness.jpg", mx::FILENAME_TYPE_STRING);
    image2->setInputValue("file", "brass_color.jpg", mx::FILENAME_TYPE_STRING);

    // 3. Test with additional resolvers
    // - Create resolver to replace all Windows separators with POSIX ones
    mx::StringResolverPtr separatorReplacer = mx::StringResolver::create();
    separatorReplacer->setFilenameSubstitution("\\\\", "/");
    separatorReplacer->setFilenameSubstitution("\\", "/");

    mx::flattenFilenames(doc1, searchPath, separatorReplacer);
    REQUIRE(nodeGraph->getFilePrefix() == mx::EMPTY_STRING);
    std::string resolvedPathString = image1->getInputValue("file")->getValueString();
    REQUIRE(resolvedPathString == (rootPath / TEST_FILE_PREFIX_STRING / TEST_IMAGE_STRING1).asString(mx::FilePath::FormatPosix));
    resolvedPathString = image2->getInputValue("file")->getValueString();
    REQUIRE(resolvedPathString == (rootPath / TEST_FILE_PREFIX_STRING / TEST_IMAGE_STRING2).asString(mx::FilePath::FormatPosix));

    // 4. Test with pre-resolved filenames
    nodeGraph->setFilePrefix(TEST_FILE_PREFIX_STRING.asString() + "\\");
    mx::flattenFilenames(doc1, searchPath, separatorReplacer);
    REQUIRE(nodeGraph->getFilePrefix() == mx::EMPTY_STRING);
    resolvedPathString = image1->getInputValue("file")->getValueString();
    REQUIRE(resolvedPathString == (rootPath / TEST_FILE_PREFIX_STRING / TEST_IMAGE_STRING1).asString(mx::FilePath::FormatPosix));
    resolvedPathString = image2->getInputValue("file")->getValueString();
    REQUIRE(resolvedPathString == (rootPath / TEST_FILE_PREFIX_STRING / TEST_IMAGE_STRING2).asString(mx::FilePath::FormatPosix));
}

TEST_CASE("Path normalization test", "[file]")
{
    const mx::FilePath REFERENCE_REL_PATH("a/b");
    const mx::FilePath REFERENCE_ABS_PREFIX("/assets");

    std::vector<mx::FilePath> examplePaths =
    {
        "a/./b",
        "././a/b",
        "c/../d/../a/b",
        "a/b/./c/d/../.."
    };

    for (const mx::FilePath& path : examplePaths)
    {
        REQUIRE(path.getNormalized() == REFERENCE_REL_PATH);
        REQUIRE((REFERENCE_ABS_PREFIX / path).getNormalized() == (REFERENCE_ABS_PREFIX / REFERENCE_REL_PATH));
    }
}
