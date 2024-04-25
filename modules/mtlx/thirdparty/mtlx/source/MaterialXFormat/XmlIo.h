//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_XMLIO_H
#define MATERIALX_XMLIO_H

/// @file
/// Support for the MTLX file format

#include <MaterialXCore/Library.h>

#include <MaterialXCore/Document.h>

#include <MaterialXFormat/Export.h>
#include <MaterialXFormat/File.h>

MATERIALX_NAMESPACE_BEGIN

class XmlReadOptions;

extern MX_FORMAT_API const string MTLX_EXTENSION;

/// A standard function that reads from an XML file into a Document, with
/// optional search path and read options.
using XmlReadFunction = std::function<void(DocumentPtr, const FilePath&, const FileSearchPath&, const XmlReadOptions*)>;

/// @class XmlReadOptions
/// A set of options for controlling the behavior of XML read functions.
class MX_FORMAT_API XmlReadOptions
{
  public:
    XmlReadOptions();
    ~XmlReadOptions() { }

    /// If true, then XML comments will be read into documents as comment elements.
    /// Defaults to false.
    bool readComments;

    /// If true, then XML newlines will be read into documents as newline elements.
    /// Defaults to false.
    bool readNewlines;

    /// If true, then documents from earlier versions of MaterialX will be upgraded
    /// to the current version.  Defaults to true.
    bool upgradeVersion;

    /// If provided, this function will be invoked when an XInclude reference
    /// needs to be read into a document.  Defaults to readFromXmlFile.
    XmlReadFunction readXIncludeFunction;

    /// The vector of parent XIncludes at the scope of the current document.
    /// Defaults to an empty vector.
    StringVec parentXIncludes;
};

/// @class XmlWriteOptions
/// A set of options for controlling the behavior of XML write functions.
class MX_FORMAT_API XmlWriteOptions
{
  public:
    XmlWriteOptions();
    ~XmlWriteOptions() { }

    /// If true, elements with source file markings will be written as
    /// XIncludes rather than explicit data.  Defaults to true.
    bool writeXIncludeEnable;

    /// If provided, this function will be used to exclude specific elements
    /// (those returning false) from the write operation.  Defaults to nullptr.
    ElementPredicate elementPredicate;
};

/// @class ExceptionParseError
/// An exception that is thrown when a requested document cannot be parsed.
class MX_FORMAT_API ExceptionParseError : public Exception
{
  public:
    using Exception::Exception;
};

/// @class ExceptionFileMissing
/// An exception that is thrown when a requested file cannot be opened.
class MX_FORMAT_API ExceptionFileMissing : public Exception
{
  public:
    using Exception::Exception;
};

/// @name Read Functions
/// @{

/// Read a Document as XML from the given character buffer.
/// @param doc The Document into which data is read.
/// @param buffer The character buffer from which data is read.
/// @param searchPath An optional sequence of file paths that will be applied
///    in order when searching for the given file and its includes.  This
///    argument can be supplied either as a FileSearchPath, or as a standard
///    string with paths separated by the PATH_SEPARATOR character.
/// @param readOptions An optional pointer to an XmlReadOptions object.
///    If provided, then the given options will affect the behavior of the
///    read function.  Defaults to a null pointer.
/// @throws ExceptionParseError if the document cannot be parsed.
MX_FORMAT_API void readFromXmlBuffer(DocumentPtr doc, const char* buffer, FileSearchPath searchPath = FileSearchPath(), const XmlReadOptions* readOptions = nullptr);

/// Read a Document as XML from the given input stream.
/// @param doc The Document into which data is read.
/// @param stream The input stream from which data is read.
/// @param searchPath An optional sequence of file paths that will be applied
///    in order when searching for the given file and its includes.  This
///    argument can be supplied either as a FileSearchPath, or as a standard
///    string with paths separated by the PATH_SEPARATOR character.
/// @param readOptions An optional pointer to an XmlReadOptions object.
///    If provided, then the given options will affect the behavior of the
///    read function.  Defaults to a null pointer.
/// @throws ExceptionParseError if the document cannot be parsed.
MX_FORMAT_API void readFromXmlStream(DocumentPtr doc, std::istream& stream, FileSearchPath searchPath = FileSearchPath(), const XmlReadOptions* readOptions = nullptr);

/// Read a Document as XML from the given filename.
/// @param doc The Document into which data is read.
/// @param filename The filename from which data is read.  This argument can
///    be supplied either as a FilePath or a standard string.
/// @param searchPath An optional sequence of file paths that will be applied
///    in order when searching for the given file and its includes.  This
///    argument can be supplied either as a FileSearchPath, or as a standard
///    string with paths separated by the PATH_SEPARATOR character.
/// @param readOptions An optional pointer to an XmlReadOptions object.
///    If provided, then the given options will affect the behavior of the read
///    function.  Defaults to a null pointer.
/// @throws ExceptionParseError if the document cannot be parsed.
/// @throws ExceptionFileMissing if the file cannot be opened.
MX_FORMAT_API void readFromXmlFile(DocumentPtr doc,
                                   FilePath filename,
                                   FileSearchPath searchPath = FileSearchPath(),
                                   const XmlReadOptions* readOptions = nullptr);

/// Read a Document as XML from the given string.
/// @param doc The Document into which data is read.
/// @param str The string from which data is read.
/// @param searchPath An optional sequence of file paths that will be applied
///    in order when searching for the given file and its includes.  This
///    argument can be supplied either as a FileSearchPath, or as a standard
///    string with paths separated by the PATH_SEPARATOR character.
/// @param readOptions An optional pointer to an XmlReadOptions object.
///    If provided, then the given options will affect the behavior of the
///    read function.  Defaults to a null pointer.
/// @throws ExceptionParseError if the document cannot be parsed.
MX_FORMAT_API void readFromXmlString(DocumentPtr doc, const string& str, const FileSearchPath& searchPath = FileSearchPath(), const XmlReadOptions* readOptions = nullptr);

/// @}
/// @name Write Functions
/// @{

/// Write a Document as XML to the given output stream.
/// @param doc The Document to be written.
/// @param stream The output stream to which data is written
/// @param writeOptions An optional pointer to an XmlWriteOptions object.
///    If provided, then the given options will affect the behavior of the
///    write function.  Defaults to a null pointer.
MX_FORMAT_API void writeToXmlStream(DocumentPtr doc, std::ostream& stream, const XmlWriteOptions* writeOptions = nullptr);

/// Write a Document as XML to the given filename.
/// @param doc The Document to be written.
/// @param filename The filename to which data is written.  This argument can
///    be supplied either as a FilePath or a standard string.
/// @param writeOptions An optional pointer to an XmlWriteOptions object.
///    If provided, then the given options will affect the behavior of the
///    write function.  Defaults to a null pointer.
MX_FORMAT_API void writeToXmlFile(DocumentPtr doc, const FilePath& filename, const XmlWriteOptions* writeOptions = nullptr);

/// Write a Document as XML to a new string, returned by value.
/// @param doc The Document to be written.
/// @param writeOptions An optional pointer to an XmlWriteOptions object.
///    If provided, then the given options will affect the behavior of the
///    write function.  Defaults to a null pointer.
/// @return The output string, returned by value
MX_FORMAT_API string writeToXmlString(DocumentPtr doc, const XmlWriteOptions* writeOptions = nullptr);

/// @}
/// @name Edit Functions
/// @{

/// Add an XInclude reference to the top of a Document, creating a generic
/// element to hold the reference filename.
/// @param doc The Document to be modified.
/// @param filename The filename of the XInclude reference to be added.
MX_FORMAT_API void prependXInclude(DocumentPtr doc, const FilePath& filename);

/// @}

MATERIALX_NAMESPACE_END

#endif
