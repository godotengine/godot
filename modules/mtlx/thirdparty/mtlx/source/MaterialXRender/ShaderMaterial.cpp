//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/ShaderMaterial.h>
#include <MaterialXFormat/XmlIo.h>

MATERIALX_NAMESPACE_BEGIN

ShaderMaterial::ShaderMaterial() : _hasTransparency(false) { }
ShaderMaterial::~ShaderMaterial() { }

void ShaderMaterial::setDocument(DocumentPtr doc)
{
    _doc = doc;
}

DocumentPtr ShaderMaterial::getDocument() const
{
    return _doc;
}

void ShaderMaterial::setElement(TypedElementPtr val)
{
    _elem = val;
}

TypedElementPtr ShaderMaterial::getElement() const
{
    return _elem;
}

void ShaderMaterial::setMaterialNode(NodePtr node)
{
    _materialNode = node;
}

NodePtr ShaderMaterial::getMaterialNode() const
{
    return _materialNode;
}

void ShaderMaterial::setUdim(const std::string& val)
{
    _udim = val;
}

const std::string& ShaderMaterial::getUdim()
{
    return _udim;
}

ShaderPtr ShaderMaterial::getShader() const
{
    return _hwShader;
}

bool ShaderMaterial::hasTransparency() const
{
    return _hasTransparency;
}

bool ShaderMaterial::generateEnvironmentShader(GenContext& context,
                                               const FilePath& filename,
                                               DocumentPtr stdLib,
                                               const FilePath& imagePath)
{
    // Read in the environment nodegraph.
    DocumentPtr doc = createDocument();
    doc->importLibrary(stdLib);
    DocumentPtr envDoc = createDocument();
    readFromXmlFile(envDoc, filename);
    doc->importLibrary(envDoc);

    NodeGraphPtr envGraph = doc->getNodeGraph("envMap");
    if (!envGraph)
    {
        return false;
    }
    NodePtr image = envGraph->getNode("envImage");
    if (!image)
    {
        return false;
    }
    image->setInputValue("file", imagePath.asString(), FILENAME_TYPE_STRING);
    OutputPtr output = envGraph->getOutput("out");
    if (!output)
    {
        return false;
    }

    // Create the shader.
    std::string shaderName = "__ENV_SHADER__";
    _hwShader = createShader(shaderName, context, output);
    if (!_hwShader)
    {
        return false;
    }
    return generateShader(_hwShader);
}

MATERIALX_NAMESPACE_END
