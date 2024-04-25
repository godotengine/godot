//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXFormat/Environ.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>

namespace mx = MaterialX;

TEST_CASE("Load content", "[xmlio]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePath libraryPath = searchPath.find("libraries/stdlib");
    mx::FilePath examplesPath = searchPath.find("resources/Materials/Examples/StandardSurface");
    searchPath.append(libraryPath);
    searchPath.append(examplesPath);

    // Read the standard library.
    std::vector<mx::DocumentPtr> libs;
    for (const mx::FilePath& filename : libraryPath.getFilesInDirectory(mx::MTLX_EXTENSION))
    {
        mx::DocumentPtr lib = mx::createDocument();
        mx::readFromXmlFile(lib, filename, searchPath);
        libs.push_back(lib);
    }

    // Read and validate each example document.
    for (const mx::FilePath& filename : examplesPath.getFilesInDirectory(mx::MTLX_EXTENSION))
    {
        mx::DocumentPtr doc = mx::createDocument();
        mx::readFromXmlFile(doc, filename, searchPath);
        for (mx::DocumentPtr lib : libs)
        {
            doc->importLibrary(lib);
        }
        std::string message;
        bool docValid = doc->validate(&message);
        if (!docValid)
        {
            WARN("[" + filename.asString() + "] " + message);
        }
        REQUIRE(docValid);

        // Traverse the document tree
        int valueElementCount = 0;
        for (mx::ElementPtr elem : doc->traverseTree())
        {
            if (elem->isA<mx::ValueElement>())
            {
                valueElementCount++;
            }
        }
        REQUIRE(valueElementCount > 0);

        // Serialize to XML.
        mx::XmlWriteOptions writeOptions;
        writeOptions.writeXIncludeEnable = false;
        std::string xmlString = mx::writeToXmlString(doc, &writeOptions);

        // Verify that the serialized document is identical.
        mx::DocumentPtr writtenDoc = mx::createDocument();
        mx::readFromXmlString(writtenDoc, xmlString);
        REQUIRE(*writtenDoc == *doc);
        mx::readFromXmlBuffer(writtenDoc, xmlString.c_str());
        REQUIRE(*writtenDoc == *doc);

        // Flatten all subgraphs.
        doc->flattenSubgraphs();
        for (mx::NodeGraphPtr graph : doc->getNodeGraphs())
        {
            if (graph->getActiveSourceUri() == doc->getSourceUri())
            {
                graph->flattenSubgraphs();
            }
        }
        REQUIRE(doc->validate());

        // Verify that all referenced types and nodes are declared.
        bool referencesValid = true;
        for (mx::ElementPtr elem : doc->traverseTree())
        {
            if (elem->getActiveSourceUri() != doc->getSourceUri())
            {
                continue;
            }

            mx::TypedElementPtr typedElem = elem->asA<mx::TypedElement>();
            if (typedElem && typedElem->hasType() && !typedElem->isMultiOutputType())
            {
                if (!typedElem->getTypeDef())
                {
                    WARN("[" + typedElem->getActiveSourceUri() + "] TypedElement " + typedElem->getName() + " has no matching TypeDef");
                    referencesValid = false;
                }
            }
            mx::NodePtr node = elem->asA<mx::Node>();
            if (node)
            {
                if (!node->getNodeDefString().empty() && !node->getNodeDef())
                {
                    WARN("[" + node->getActiveSourceUri() + "] Node " + node->getName() + " has no matching NodeDef for " + node->getNodeDefString());
                    referencesValid = false;
                }
            }
        }
        REQUIRE(referencesValid);
    }

    // Read the same document twice and verify that duplicate elements
    // are skipped.
    mx::DocumentPtr doc = mx::createDocument();
    std::string filename = "standard_surface_look_brass_tiled.mtlx";
    mx::readFromXmlFile(doc, filename, searchPath);
    mx::readFromXmlFile(doc, filename, searchPath);
    REQUIRE(doc->validate());

    // Import libraries twice and verify that duplicate elements are
    // skipped.
    mx::DocumentPtr libDoc = doc->copy();
    for (mx::DocumentPtr lib : libs)
    {
        libDoc->importLibrary(lib);
        libDoc->importLibrary(lib);
    }
    REQUIRE(libDoc->validate());

    // Read document with conflicting elements.
    mx::DocumentPtr conflictDoc = doc->copy();
    for (mx::ElementPtr elem : conflictDoc->traverseTree())
    {
        if (elem->isA<mx::Node>("image"))
        {
            elem->setFilePrefix("differentFolder/");
        }
    }
    mx::readFromXmlFile(conflictDoc, filename, searchPath);
    REQUIRE(conflictDoc->validate());

    // Reread in clean document
    doc = mx::createDocument();
    mx::readFromXmlFile(doc, filename, searchPath);

    // Read document without XIncludes.
    mx::DocumentPtr flatDoc = mx::createDocument();
    mx::XmlReadOptions readOptions;
    readOptions.readXIncludeFunction = nullptr;
    mx::readFromXmlFile(flatDoc, filename, searchPath, &readOptions);
    readOptions.readXIncludeFunction = mx::readFromXmlFile;
    REQUIRE(*flatDoc != *doc);

    // Read document using environment search path.
    mx::setEnviron(mx::MATERIALX_SEARCH_PATH_ENV_VAR, searchPath.asString());
    mx::DocumentPtr envDoc = mx::createDocument();
    mx::readFromXmlFile(envDoc, filename, mx::FileSearchPath(), &readOptions);
    REQUIRE(*doc == *envDoc);
    mx::removeEnviron(mx::MATERIALX_SEARCH_PATH_ENV_VAR);
    REQUIRE_THROWS_AS(mx::readFromXmlFile(envDoc, filename, mx::FileSearchPath(), &readOptions), mx::ExceptionFileMissing);

    // Serialize to XML with a custom predicate that skips images.
    auto skipImages = [](mx::ConstElementPtr elem)
    {
        return !elem->isA<mx::Node>("image");
    };
    mx::XmlWriteOptions writeOptions;
    writeOptions.writeXIncludeEnable = false;
    writeOptions.elementPredicate = skipImages;
    std::string xmlString = mx::writeToXmlString(doc, &writeOptions);

    // Reconstruct and verify that the document contains no images.
    mx::DocumentPtr writtenDoc = mx::createDocument();
    mx::readFromXmlString(writtenDoc, xmlString, mx::FileSearchPath(), &readOptions);
    REQUIRE(*writtenDoc != *doc);
    unsigned imageElementCount = 0;
    for (mx::ElementPtr elem : writtenDoc->traverseTree())
    {
        if (elem->isA<mx::Node>("image"))
        {
            imageElementCount++;
        }
    }
    REQUIRE(imageElementCount == 0);

    // Serialize to XML with a custom predicate to remove XIncludes.
    auto skipLibIncludes = [libs](mx::ConstElementPtr elem)
    {
        if (elem->hasSourceUri())
        {
            for (auto lib : libs)
            {
                if (lib->getSourceUri() == elem->getSourceUri())
                {
                    return false;
                }
            }
        }
        return true;
    };
    writeOptions.writeXIncludeEnable = true;
    writeOptions.elementPredicate = skipLibIncludes;
    xmlString = mx::writeToXmlString(writtenDoc, &writeOptions);

    // Verify that the document contains no XIncludes.
    writtenDoc = mx::createDocument();
    mx::readFromXmlString(writtenDoc, xmlString, mx::FileSearchPath(), &readOptions);
    bool hasSourceUri = false;
    for (mx::ElementPtr elem : writtenDoc->traverseTree())
    {
        if (elem->hasSourceUri())
        {
            hasSourceUri = true;
            break;
        }
    }
    REQUIRE(!hasSourceUri);

    // Read a document with a relative XInclude.
    mx::DocumentPtr parentDoc = mx::createDocument();
    mx::readFromXmlFile(parentDoc, "resources/Materials/TestSuite/libraries/metal/brass_wire_mesh.mtlx", searchPath);
    REQUIRE(parentDoc->getNodeDef("ND_TestMetal") != nullptr);

    // Read a string with a relative Xinclude
    parentDoc = mx::createDocument();
    std::string includeTest = 
        "<?xml version=\"1.0\"?> " 
        "<materialx version=\"1.38\">" 
        "<xi:include href=\"standard_surface_brass_tiled.mtlx\" />"  
        "<xi:include href=\"standard_surface_greysphere_calibration.mtlx\" />" 
        "</materialx>";
    mx::readFromXmlString(parentDoc, includeTest, searchPath);
    REQUIRE(parentDoc->getNodeGraph("NG_brass1") != nullptr);
    REQUIRE(parentDoc->getNodeGraph("NG_Greysphere_Calibration") != nullptr);

    // Read a non-existent document.
    mx::DocumentPtr nonExistentDoc = mx::createDocument();
    REQUIRE_THROWS_AS(mx::readFromXmlFile(nonExistentDoc, "NonExistent.mtlx", mx::FileSearchPath(), &readOptions), mx::ExceptionFileMissing);
}

TEST_CASE("Comments and newlines", "[xmlio]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePath testPath = searchPath.find("resources/Materials/Examples/StandardSurface/standard_surface_chess_set.mtlx");

    // Read the example file into an XML string buffer.
    std::string origXml = mx::readFile(testPath);

    // Convert the string to a document with comments and newlines preserved.
    mx::DocumentPtr doc = mx::createDocument();
    mx::XmlReadOptions readOptions;
    readOptions.readComments = true;
    readOptions.readNewlines = true;
    mx::readFromXmlString(doc, origXml, mx::FileSearchPath(), &readOptions);

    // Write the document to a new XML string buffer.
    std::string newXml = mx::writeToXmlString(doc);

    // Verify that the XML string buffers are identical.
    REQUIRE(origXml == newXml);
}

TEST_CASE("Fuzz testing", "[xmlio]")
{
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePath examplesPath = searchPath.find("resources/Materials/Examples/StandardSurface");

    std::mt19937 rng(0);
    std::uniform_int_distribution<size_t> randChar(0, 255);

    for (const mx::FilePath& filename : examplesPath.getFilesInDirectory(mx::MTLX_EXTENSION))
    {
        // Read the example file into an XML string buffer.
        const std::string origString = mx::readFile(examplesPath / filename);
        REQUIRE(origString.size() > 0);
        std::uniform_int_distribution<size_t> randPos(0, origString.size() - 1);

        // Iterate over test runs.
        for (size_t testRun = 0; testRun < 256; testRun++)
        {
            std::string editString = origString;

            // Iterate over string edits.
            for (size_t editIndex = 0; editIndex < 32; editIndex++)
            {
                // Randomly alter one character in the document.
                size_t charIndex = randPos(rng);
                size_t newChar = randChar(rng);
                editString[charIndex] = (char) newChar;

                // Attempt to interpret the edited string as a document, allowing only MaterialX exceptions.
                mx::DocumentPtr doc = mx::createDocument();
                try
                {
                    mx::readFromXmlString(doc, editString, searchPath);
                    doc->validate();
                }
                catch (const mx::Exception&)
                {
                    // On a MaterialX exception, proceed to the next test run.
                    break;
                }
            }
        }
    }
}

TEST_CASE("Locale region testing", "[xmlio]")
{
    // In the United States, the thousands separator is a comma, while in Germany it is a period.
    // Thus, one thousand twenty five is displayed as 1,025 in the United States and 1.025 in Germany.
    //
    // In a MaterialX document, a vector3 value of "1,1.5,2.0" should be interpreted as (1.0f, 1.5f, 2.0f).

    // Set the locale to Germany, if supported on this platform, and note the original locale.
    std::locale origLocale;
    try
    {
        origLocale = std::locale::global(std::locale("de_DE"));
    }
    catch (const std::runtime_error& e)
    {
        WARN("Unable to change locale " << e.what());
        return;
    }

    // Read and validate each example document.
    mx::FilePath testPath("resources/Materials/TestSuite/locale");
    for (const mx::FilePath& filename : testPath.getFilesInDirectory(mx::MTLX_EXTENSION))
    {
        mx::DocumentPtr doc = mx::createDocument();
        mx::readFromXmlFile(doc, testPath / filename);
        REQUIRE(doc->validate());

        for (mx::ElementPtr elem : doc->traverseTree())
        {
            if (elem->isA<mx::ValueElement>())
            {
                if (elem->hasAttribute("uiname"))
                {
                    REQUIRE(!elem->getAttribute("uiname").empty());
                }
            }
        }
    }

    // Restore the original locale.
    std::locale::global(origLocale);
}
