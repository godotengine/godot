//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXFormat/XmlIo.h>

#include <MaterialXFormat/External/PugiXML/pugixml.hpp>

#include <MaterialXCore/Types.h>

#include <cstring>
#include <fstream>
#include <sstream>

using namespace pugi;

MATERIALX_NAMESPACE_BEGIN

const string MTLX_EXTENSION = "mtlx";

namespace
{

const string XINCLUDE_TAG = "xi:include";
const string XINCLUDE_NAMESPACE = "xmlns:xi";
const string XINCLUDE_URL = "http://www.w3.org/2001/XInclude";

void elementFromXml(const xml_node& xmlNode, ElementPtr elem, const XmlReadOptions* readOptions)
{
    // Store attributes in element.
    for (const xml_attribute& xmlAttr : xmlNode.attributes())
    {
        if (xmlAttr.name() != Element::NAME_ATTRIBUTE)
        {
            elem->setAttribute(xmlAttr.name(), xmlAttr.value());
        }
    }

    // Create child elements and recurse.
    for (const xml_node& xmlChild : xmlNode.children())
    {
        string category = xmlChild.name();
        string name;
        for (const xml_attribute& xmlAttr : xmlChild.attributes())
        {
            if (xmlAttr.name() == Element::NAME_ATTRIBUTE)
            {
                name = xmlAttr.value();
                break;
            }
        }

        // Check for duplicate elements.
        ConstElementPtr previous = elem->getChild(name);
        if (previous)
        {
            continue;
        }

        // Create the new element.
        ElementPtr child = elem->addChildOfCategory(category, name);
        elementFromXml(xmlChild, child, readOptions);

        // Handle the interpretation of XML comments and newlines.
        if (readOptions && category.empty())
        {
            if (readOptions->readComments && xmlChild.type() == node_comment)
            {
                child = elem->changeChildCategory(child, CommentElement::CATEGORY);
                child->setDocString(xmlChild.value());
            }
            else if (readOptions->readNewlines && xmlChild.type() == node_newline)
            {
                child = elem->changeChildCategory(child, NewlineElement::CATEGORY);
            }
        }
    }
}

void elementToXml(ConstElementPtr elem, xml_node& xmlNode, const XmlWriteOptions* writeOptions)
{
    bool writeXIncludeEnable = writeOptions ? writeOptions->writeXIncludeEnable : true;
    ElementPredicate elementPredicate = writeOptions ? writeOptions->elementPredicate : nullptr;

    // Store attributes in XML.
    if (!elem->getName().empty())
    {
        xmlNode.append_attribute(Element::NAME_ATTRIBUTE.c_str()) = elem->getName().c_str();
    }
    for (const string& attrName : elem->getAttributeNames())
    {
        xml_attribute xmlAttr = xmlNode.append_attribute(attrName.c_str());
        xmlAttr.set_value(elem->getAttribute(attrName).c_str());
    }

    // Create child nodes and recurse.
    StringSet writtenSourceFiles;
    for (auto child : elem->getChildren())
    {
        if (elementPredicate && !elementPredicate(child))
        {
            continue;
        }

        // Write XInclude references if requested.
        if (writeXIncludeEnable && child->hasSourceUri())
        {
            string sourceUri = child->getSourceUri();
            if (sourceUri != elem->getDocument()->getSourceUri())
            {
                if (!writtenSourceFiles.count(sourceUri))
                {
                    if (!xmlNode.attribute(XINCLUDE_NAMESPACE.c_str()))
                    {
                        xmlNode.append_attribute(XINCLUDE_NAMESPACE.c_str()) = XINCLUDE_URL.c_str();
                    }
                    xml_node includeNode = xmlNode.append_child(XINCLUDE_TAG.c_str());
                    xml_attribute includeAttr = includeNode.append_attribute("href");
                    FilePath includePath(sourceUri);

                    // Write relative include paths in Posix format, and absolute
                    // include paths in native format.
                    FilePath::Format includeFormat = includePath.isAbsolute() ? FilePath::FormatNative : FilePath::FormatPosix;
                    includeAttr.set_value(includePath.asString(includeFormat).c_str());

                    writtenSourceFiles.insert(sourceUri);
                }
                continue;
            }
        }

        // Write XML comments.
        if (child->getCategory() == CommentElement::CATEGORY)
        {
            xml_node xmlChild = xmlNode.append_child(node_comment);
            xmlChild.set_value(child->getAttribute(Element::DOC_ATTRIBUTE).c_str());
            continue;
        }

        // Write XML newlines.
        if (child->getCategory() == NewlineElement::CATEGORY)
        {
            xml_node xmlChild = xmlNode.append_child(node_newline);
            xmlChild.set_value("\n");
            continue;
        }

        xml_node xmlChild = xmlNode.append_child(child->getCategory().c_str());
        elementToXml(child, xmlChild, writeOptions);
    }
}

void processXIncludes(DocumentPtr doc, xml_node& xmlNode, const FileSearchPath& searchPath, const XmlReadOptions* readOptions)
{
    // Search path for includes. Set empty and then evaluated once in the iteration through xml includes.
    FileSearchPath includeSearchPath;

    XmlReadFunction readXIncludeFunction = readOptions ? readOptions->readXIncludeFunction : readFromXmlFile;
    xml_node xmlChild = xmlNode.first_child();
    while (xmlChild)
    {
        if (xmlChild.name() == XINCLUDE_TAG)
        {
            // Read XInclude references if requested.
            if (readXIncludeFunction)
            {
                string filename = xmlChild.attribute("href").value();

                // Check for XInclude cycles.
                if (readOptions)
                {
                    const StringVec& parents = readOptions->parentXIncludes;
                    if (std::find(parents.begin(), parents.end(), filename) != parents.end())
                    {
                        throw ExceptionParseError("XInclude cycle detected.");
                    }
                }

                // Read the included file into a library document.
                DocumentPtr library = createDocument();
                XmlReadOptions xiReadOptions = readOptions ? *readOptions : XmlReadOptions();
                xiReadOptions.parentXIncludes.push_back(filename);

                // Prepend the directory of the parent to accommodate
                // includes relative to the parent file location.
                if (includeSearchPath.isEmpty())
                {
                    string parentUri = doc->getSourceUri();
                    if (!parentUri.empty())
                    {
                        FilePath filePath = searchPath.find(parentUri);
                        if (!filePath.isEmpty())
                        {
                            // Remove the file name from the path as we want the path to the containing folder.
                            includeSearchPath = searchPath;
                            includeSearchPath.prepend(filePath.getParentPath());
                        }
                    }
                    // Set default search path if no parent path found
                    if (includeSearchPath.isEmpty())
                    {
                        includeSearchPath = searchPath;
                    }
                }
                readXIncludeFunction(library, filename, includeSearchPath, &xiReadOptions);

                // Import the library document.
                doc->importLibrary(library);
            }

            // Remove include directive.
            xml_node includeNode = xmlChild;
            xmlChild = xmlChild.next_sibling();
            xmlNode.remove_child(includeNode);
        }
        else
        {
            xmlChild = xmlChild.next_sibling();
        }
    }
}

void documentFromXml(DocumentPtr doc,
                     const xml_document& xmlDoc,
                     const FileSearchPath& searchPath = FileSearchPath(),
                     const XmlReadOptions* readOptions = nullptr)
{
    xml_node xmlRoot = xmlDoc.child(Document::CATEGORY.c_str());
    if (xmlRoot)
    {
        processXIncludes(doc, xmlRoot, searchPath, readOptions);
        elementFromXml(xmlRoot, doc, readOptions);
    }

    if (!readOptions || readOptions->upgradeVersion)
    {
        doc->upgradeVersion();
    }
}

void validateParseResult(const xml_parse_result& result, const FilePath& filename = FilePath())
{
    if (result)
    {
        return;
    }

    if (result.status == xml_parse_status::status_file_not_found ||
        result.status == xml_parse_status::status_io_error ||
        result.status == xml_parse_status::status_out_of_memory)
    {
        throw ExceptionFileMissing("Failed to open file for reading: " + filename.asString());
    }

    string desc = result.description();
    string offset = std::to_string(result.offset);
    string message = "XML parse error";
    if (!filename.isEmpty())
    {
        message += " in " + filename.asString();
    }
    message += " (" + desc + " at character " + offset + ")";

    throw ExceptionParseError(message);
}

unsigned int getParseOptions(const XmlReadOptions* readOptions)
{
    unsigned int parseOptions = parse_default;
    if (readOptions)
    {
        if (readOptions->readComments)
        {
            parseOptions |= parse_comments;
        }
        if (readOptions->readNewlines)
        {
            parseOptions |= parse_newlines;
        }
    }
    return parseOptions;
}

} // anonymous namespace

//
// XmlReadOptions methods
//

XmlReadOptions::XmlReadOptions() :
    readComments(false),
    readNewlines(false),
    upgradeVersion(true),
    readXIncludeFunction(readFromXmlFile)
{
}

//
// XmlWriteOptions methods
//

XmlWriteOptions::XmlWriteOptions() :
    writeXIncludeEnable(true)
{
}

//
// Reading
//

void readFromXmlBuffer(DocumentPtr doc, const char* buffer, FileSearchPath searchPath, const XmlReadOptions* readOptions)
{
    searchPath.append(getEnvironmentPath());

    xml_document xmlDoc;
    xml_parse_result result = xmlDoc.load_string(buffer, getParseOptions(readOptions));
    validateParseResult(result);

    documentFromXml(doc, xmlDoc, searchPath, readOptions);
}

void readFromXmlStream(DocumentPtr doc, std::istream& stream, FileSearchPath searchPath, const XmlReadOptions* readOptions)
{
    searchPath.append(getEnvironmentPath());

    xml_document xmlDoc;
    xml_parse_result result = xmlDoc.load(stream, getParseOptions(readOptions));
    validateParseResult(result);

    documentFromXml(doc, xmlDoc, searchPath, readOptions);
}

void readFromXmlFile(DocumentPtr doc, FilePath filename, FileSearchPath searchPath, const XmlReadOptions* readOptions)
{
    searchPath.append(getEnvironmentPath());
    filename = searchPath.find(filename);

    xml_document xmlDoc;
    xml_parse_result result = xmlDoc.load_file(filename.asString().c_str(), getParseOptions(readOptions));
    validateParseResult(result, filename);

    // This must be done before parsing the XML as the source URI
    // is used for searching for include files.
    if (readOptions && !readOptions->parentXIncludes.empty())
    {
        doc->setSourceUri(readOptions->parentXIncludes[0]);
    }
    else
    {
        doc->setSourceUri(filename);
    }
    documentFromXml(doc, xmlDoc, searchPath, readOptions);
}

void readFromXmlString(DocumentPtr doc, const string& str, const FileSearchPath& searchPath, const XmlReadOptions* readOptions)
{
    std::istringstream stream(str);
    readFromXmlStream(doc, stream, searchPath, readOptions);
}

//
// Writing
//

void writeToXmlStream(DocumentPtr doc, std::ostream& stream, const XmlWriteOptions* writeOptions)
{
    xml_document xmlDoc;
    xml_node xmlRoot = xmlDoc.append_child("materialx");
    elementToXml(doc, xmlRoot, writeOptions);
    xmlDoc.save(stream, "  ");
}

void writeToXmlFile(DocumentPtr doc, const FilePath& filename, const XmlWriteOptions* writeOptions)
{
    std::ofstream ofs(filename.asString());
    writeToXmlStream(doc, ofs, writeOptions);
}

string writeToXmlString(DocumentPtr doc, const XmlWriteOptions* writeOptions)
{
    std::ostringstream stream;
    writeToXmlStream(doc, stream, writeOptions);
    return stream.str();
}

void prependXInclude(DocumentPtr doc, const FilePath& filename)
{
    if (!filename.isEmpty())
    {
        ElementPtr elem = doc->addNode("xinclude");
        elem->setSourceUri(filename.asString());
        doc->setChildIndex(elem->getName(), 0);
    }
}

MATERIALX_NAMESPACE_END
