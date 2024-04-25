//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXFormat/Util.h>

#include <fstream>
#include <iostream>
#include <sstream>

MATERIALX_NAMESPACE_BEGIN

string readFile(const FilePath& filePath)
{
    std::ifstream file(filePath.asString(), std::ios::in);
    if (file)
    {
        std::stringstream stream;
        stream << file.rdbuf();
        file.close();
        if (stream)
        {
            return stream.str();
        }
    }
    return EMPTY_STRING;
}

void getSubdirectories(const FilePathVec& rootDirectories, const FileSearchPath& searchPath, FilePathVec& subDirectories)
{
    for (const FilePath& root : rootDirectories)
    {
        FilePath rootPath = searchPath.find(root);
        if (rootPath.exists())
        {
            FilePathVec childDirectories = rootPath.getSubDirectories();
            subDirectories.insert(std::end(subDirectories), std::begin(childDirectories), std::end(childDirectories));
        }
    }
}

void loadDocuments(const FilePath& rootPath, const FileSearchPath& searchPath, const StringSet& skipFiles,
                   const StringSet& includeFiles, vector<DocumentPtr>& documents, StringVec& documentsPaths,
                   const XmlReadOptions* readOptions, StringVec* errors)
{
    for (const FilePath& dir : rootPath.getSubDirectories())
    {
        for (const FilePath& file : dir.getFilesInDirectory(MTLX_EXTENSION))
        {
            if (!skipFiles.count(file) &&
                (includeFiles.empty() || includeFiles.count(file)))
            {
                DocumentPtr doc = createDocument();
                const FilePath filePath = dir / file;
                try
                {
                    FileSearchPath readSearchPath(searchPath);
                    readSearchPath.append(dir);
                    readFromXmlFile(doc, filePath, readSearchPath, readOptions);
                    documents.push_back(doc);
                    documentsPaths.push_back(filePath.asString());
                }
                catch (Exception& e)
                {
                    if (errors)
                    {
                        errors->push_back("Failed to load: " + filePath.asString() + ". Error: " + e.what());
                    }
                }
            }
        }
    }
}

void loadLibrary(const FilePath& file, DocumentPtr doc, const FileSearchPath& searchPath, const XmlReadOptions* readOptions)
{
    DocumentPtr libDoc = createDocument();
    readFromXmlFile(libDoc, file, searchPath, readOptions);
    doc->importLibrary(libDoc);
}

StringSet loadLibraries(const FilePathVec& libraryFolders,
                        const FileSearchPath& searchPath,
                        DocumentPtr doc,
                        const StringSet& excludeFiles,
                        const XmlReadOptions* readOptions)
{
    // Append environment path to the specified search path.
    FileSearchPath librarySearchPath = searchPath;
    librarySearchPath.append(getEnvironmentPath());

    StringSet loadedLibraries;
    if (libraryFolders.empty())
    {
        // No libraries specified so scan in all search paths
        for (const FilePath& libraryPath : librarySearchPath)
        {
            for (const FilePath& path : libraryPath.getSubDirectories())
            {
                for (const FilePath& filename : path.getFilesInDirectory(MTLX_EXTENSION))
                {
                    if (!excludeFiles.count(filename))
                    {
                        const FilePath& file = path / filename;
                        if (loadedLibraries.count(file) == 0)
                        {
                            loadLibrary(file, doc, searchPath, readOptions);
                            loadedLibraries.insert(file.asString());
                        }
                    }
                }
            }
        }
    }
    else
    {
        // Look for specific library folders in the search paths
        for (const FilePath& libraryName : libraryFolders)
        {
            FilePath libraryPath = librarySearchPath.find(libraryName);
            for (const FilePath& path : libraryPath.getSubDirectories())
            {
                for (const FilePath& filename : path.getFilesInDirectory(MTLX_EXTENSION))
                {
                    if (!excludeFiles.count(filename))
                    {
                        const FilePath& file = path / filename;
                        if (loadedLibraries.count(file) == 0)
                        {
                            loadLibrary(file, doc, searchPath, readOptions);
                            loadedLibraries.insert(file.asString());
                        }
                    }
                }
            }
        }
    }
    return loadedLibraries;
}

void flattenFilenames(DocumentPtr doc, const FileSearchPath& searchPath, StringResolverPtr customResolver)
{
    for (ElementPtr elem : doc->traverseTree())
    {
        ValueElementPtr valueElem = elem->asA<ValueElement>();
        if (!valueElem || valueElem->getType() != FILENAME_TYPE_STRING)
        {
            continue;
        }

        FilePath unresolvedValue(valueElem->getValueString());
        if (unresolvedValue.isEmpty())
        {
            continue;
        }
        StringResolverPtr elementResolver = elem->createStringResolver();
        // If the path is already absolute then don't allow an additional prefix
        // as this would make the path invalid.
        if (unresolvedValue.isAbsolute())
        {
            elementResolver->setFilePrefix(EMPTY_STRING);
        }
        string resolvedString = valueElem->getResolvedValueString(elementResolver);

        // Convert relative to absolute pathing if the file is not already found
        if (!searchPath.isEmpty())
        {
            FilePath resolvedValue(resolvedString);
            if (!resolvedValue.isAbsolute())
            {
                for (size_t i = 0; i < searchPath.size(); i++)
                {
                    FilePath testPath = searchPath[i] / resolvedValue;
                    testPath = testPath.getNormalized();
                    if (testPath.exists())
                    {
                        resolvedString = testPath.asString();
                        break;
                    }
                }
            }
        }

        // Apply any custom filename resolver
        if (customResolver && customResolver->isResolvedType(FILENAME_TYPE_STRING))
        {
            resolvedString = customResolver->resolve(resolvedString, FILENAME_TYPE_STRING);
        }

        valueElem->setValueString(resolvedString);
    }

    // Remove any file prefix attributes
    for (ElementPtr elem : doc->traverseTree())
    {
        if (elem->hasFilePrefix())
        {
            elem->removeAttribute(Element::FILE_PREFIX_ATTRIBUTE);
        }
    }
}

FileSearchPath getSourceSearchPath(ConstDocumentPtr doc)
{
    StringSet pathSet;
    for (ConstElementPtr elem : doc->traverseTree())
    {
        if (elem->hasSourceUri())
        {
            FilePath sourceFilename = FilePath(elem->getSourceUri());
            pathSet.insert(sourceFilename.getParentPath());
        }
    }

    FileSearchPath searchPath;
    for (FilePath path : pathSet)
    {
        searchPath.append(path);
    }

    return searchPath;
}

FileSearchPath getDefaultDataSearchPath()
{
    const FilePath REQUIRED_LIBRARY_FOLDER("libraries/targets");
    FilePath currentPath = FilePath::getModulePath();
    while (!currentPath.isEmpty())
    {
        if ((currentPath / REQUIRED_LIBRARY_FOLDER).exists())
        {
            return FileSearchPath(currentPath);
        }
        currentPath = currentPath.getParentPath();
    }
    return FileSearchPath();
}

MATERIALX_NAMESPACE_END
