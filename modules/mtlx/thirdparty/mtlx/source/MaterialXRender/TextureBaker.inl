//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/OiioImageLoader.h>
#include <MaterialXRender/StbImageLoader.h>
#include <MaterialXRender/Util.h>

#include <MaterialXGenShader/DefaultColorManagementSystem.h>

#include <MaterialXFormat/XmlIo.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const string SRGB_TEXTURE = "srgb_texture";
const string LIN_REC709 = "lin_rec709";
const string SHADER_PREFIX = "SR_";
const string DEFAULT_UDIM_PREFIX = "_";

} // anonymous namespace

template <typename Renderer, typename ShaderGen>
string TextureBaker<Renderer, ShaderGen>::getValueStringFromColor(const Color4& color, const string& type)
{
    if (type == "color4" || type == "vector4")
    {
        return toValueString(color);
    }
    if (type == "color3" || type == "vector3")
    {
        return toValueString(Vector3(color[0], color[1], color[2]));
    }
    if (type == "vector2")
    {
        return toValueString(Vector2(color[0], color[1]));
    }
    if (type == "float")
    {
        return toValueString(color[0]);
    }
    return EMPTY_STRING;
}

template <typename Renderer, typename ShaderGen>
TextureBaker<Renderer, ShaderGen>::TextureBaker(unsigned int width, unsigned int height, Image::BaseType baseType, bool flipSavedImage) :
    Renderer(width, height, baseType),
    _distanceUnit("meter"),
    _averageImages(false),
    _optimizeConstants(true),
    _bakedGraphName("NG_baked"),
    _bakedGeomInfoName("GI_baked"),
    _textureFilenameTemplate("$MATERIAL_$SHADINGMODEL_$INPUT$UDIMPREFIX$UDIM.$EXTENSION"),
    _outputStream(&std::cout),
    _hashImageNames(false),
    _textureSpaceMin(0.0f),
    _textureSpaceMax(1.0f),
    _generator(ShaderGen::create()),
    _permittedOverrides({ "$ASSET", "$MATERIAL", "$UDIMPREFIX" }),
    _flipSavedImage(flipSavedImage),
    _writeDocumentPerMaterial(true),
    _bakedTextureDoc(nullptr)
{
    if (baseType == Image::BaseType::UINT8)
    {
#if MATERIALX_BUILD_OIIO
        _extension = ImageLoader::TIFF_EXTENSION;
#else
        _extension = ImageLoader::PNG_EXTENSION;
#endif
        _colorSpace = SRGB_TEXTURE;
    }
    else
    {
#if MATERIALX_BUILD_OIIO
        _extension = ImageLoader::EXR_EXTENSION;
#else
        _extension = ImageLoader::HDR_EXTENSION;
#endif
        _colorSpace = LIN_REC709;
    }

    // Initialize our base renderer.
    Renderer::initialize();

    // Initialize our image handler.
    Renderer::_imageHandler = Renderer::createImageHandler(StbImageLoader::create());
#if MATERIALX_BUILD_OIIO
    Renderer::_imageHandler->addLoader(OiioImageLoader::create());
#endif

    // Create our dedicated frame capture image.
    _frameCaptureImage = Image::create(width, height, 4, baseType);
    _frameCaptureImage->createResourceBuffer();
}

template <typename Renderer, typename ShaderGen>
size_t TextureBaker<Renderer, ShaderGen>::findVarInTemplate(const string& filename, const string& var, size_t start)
{
    size_t i = filename.find(var, start);
    if (var == "$UDIM" && i != string::npos)
    {
        size_t udimPrefix = filename.find("$UDIMPREFIX", start);
        if (i == udimPrefix)
        {
            i = filename.find(var, i + 1);
        }
    }
    return i;
}

template <typename Renderer, typename ShaderGen>
FilePath TextureBaker<Renderer, ShaderGen>::generateTextureFilename(const StringMap& filenameTemplateMap)
{
    string bakedImageName = _textureFilenameTemplate;

    for (auto& pair : filenameTemplateMap)
    {
        string replacement = (_texTemplateOverrides.count(pair.first)) ? _texTemplateOverrides[pair.first] : pair.second;
        replacement = (filenameTemplateMap.at("$UDIM").empty() && pair.first == "$UDIMPREFIX") ? EMPTY_STRING : replacement;

        for (size_t i = 0; (i = findVarInTemplate(bakedImageName, pair.first, i)) != string::npos; i++)
        {
            bakedImageName.replace(i, pair.first.length(), replacement);
        }
    }

    if (_hashImageNames)
    {
        std::stringstream hashStream;
        hashStream << std::hash<std::string>{}(bakedImageName);
        hashStream << "." + getExtension();
        bakedImageName = hashStream.str();
    }
    return _outputImagePath / bakedImageName;
}

template <typename Renderer, typename ShaderGen>
StringMap TextureBaker<Renderer, ShaderGen>::initializeFileTemplateMap(InputPtr input, NodePtr shader, const string& udim)
{
    FilePath assetPath = FilePath(shader->getActiveSourceUri());
    assetPath.removeExtension();
    StringMap filenameTemplateMap;
    filenameTemplateMap["$ASSET"] = assetPath.getBaseName();
    filenameTemplateMap["$INPUT"] = _bakedInputMap[input->getName()];
    filenameTemplateMap["$EXTENSION"] = _extension;
    filenameTemplateMap["$MATERIAL"] = _material->getName();
    filenameTemplateMap["$SHADINGMODEL"] = shader->getCategory();
    filenameTemplateMap["$UDIM"] = udim;
    filenameTemplateMap["$UDIMPREFIX"] = DEFAULT_UDIM_PREFIX;
    return filenameTemplateMap;
}

template <typename Renderer, typename ShaderGen>
bool TextureBaker<Renderer, ShaderGen>::writeBakedImage(const BakedImage& baked, ImagePtr image)
{
    if (!Renderer::_imageHandler->saveImage(baked.filename, image, _flipSavedImage))
    {
        if (_outputStream)
        {
            *_outputStream << "Failed to write baked image: " << baked.filename.asString() << std::endl;
        }
        return false;
    }

    if (_outputStream)
    {
        *_outputStream << "Wrote baked image: " << baked.filename.asString() << std::endl;
    }

    return true;
}

template <typename Renderer, typename ShaderGen>
void TextureBaker<Renderer, ShaderGen>::bakeShaderInputs(NodePtr material, NodePtr shader, GenContext& context, const string& udim)
{
    _material = material;

    if (!shader)
    {
        return;
    }

    std::unordered_map<OutputPtr, InputPtr> bakedOutputMap;
    for (InputPtr input : shader->getInputs())
    {
        OutputPtr output = input->getConnectedOutput();
        if (output && !bakedOutputMap.count(output))
        {
            bakedOutputMap[output] = input;
            _bakedInputMap[input->getName()] = input->getName();

            // When possible, nodes with world-space outputs are applied outside of the baking process.
            NodePtr worldSpaceNode = connectsToWorldSpaceNode(output);
            if (worldSpaceNode)
            {
                output->setConnectedNode(worldSpaceNode->getConnectedNode("in"));
                _worldSpaceNodes[input->getName()] = worldSpaceNode;
            }
            StringMap filenameTemplateMap = initializeFileTemplateMap(input, shader, udim);
            bakeGraphOutput(output, context, filenameTemplateMap);
        }
        else if (bakedOutputMap.count(output))
        {
            // When the input shares the same output as a previously baked input, we use the already baked input.
            _bakedInputMap[input->getName()] = bakedOutputMap[output]->getName();
        }
    }

    // Release all images used to generate this set of shader inputs.
    Renderer::_imageHandler->clearImageCache();
}

template <typename Renderer, typename ShaderGen>
void TextureBaker<Renderer, ShaderGen>::bakeGraphOutput(OutputPtr output, GenContext& context, const StringMap& filenameTemplateMap)
{
    if (!output)
    {
        return;
    }

    bool encodeSrgb = _colorSpace == SRGB_TEXTURE && output->isColorType();
    Renderer::getFramebuffer()->setEncodeSrgb(encodeSrgb);

    ShaderPtr shader = _generator->generate("BakingShader", output, context);
    Renderer::createProgram(shader);

    // Render and capture the requested image.
    Renderer::renderTextureSpace(getTextureSpaceMin(), getTextureSpaceMax());
    string texturefilepath = generateTextureFilename(filenameTemplateMap);
    Renderer::captureImage(_frameCaptureImage);

    // Construct a baked image record.
    BakedImage baked;
    baked.filename = texturefilepath;
    if (_averageImages)
    {
        baked.uniformColor = _frameCaptureImage->getAverageColor();
        baked.isUniform = true;
    }
    else if (_frameCaptureImage->isUniformColor(&baked.uniformColor))
    {
        baked.isUniform = true;
    }
    _bakedImageMap[output].push_back(baked);

    // TODO: Write images to memory rather than to disk.
    // Write non-uniform images to disk.
    if (!baked.isUniform)
    {
        writeBakedImage(baked, _frameCaptureImage);
    }
}

template <typename Renderer, typename ShaderGen>
void TextureBaker<Renderer, ShaderGen>::optimizeBakedTextures(NodePtr shader)
{
    if (!shader)
    {
        return;
    }

    // Check for fully uniform outputs.
    for (auto& pair : _bakedImageMap)
    {
        bool outputIsUniform = true;
        for (BakedImage& baked : pair.second)
        {
            if (!baked.isUniform || baked.uniformColor != pair.second[0].uniformColor)
            {
                outputIsUniform = false;
                continue;
            }
        }
        if (outputIsUniform)
        {
            BakedConstant bakedConstant;
            bakedConstant.color = pair.second[0].uniformColor;
            _bakedConstantMap[pair.first] = bakedConstant;
        }
    }

    // Check for uniform outputs at their default values.
    NodeDefPtr shaderNodeDef = shader->getNodeDef();
    if (shaderNodeDef)
    {
        for (InputPtr shaderInput : shader->getInputs())
        {
            OutputPtr output = shaderInput->getConnectedOutput();
            if (output && _bakedConstantMap.count(output))
            {
                InputPtr input = shaderNodeDef->getInput(shaderInput->getName());
                if (input)
                {
                    Color4 uniformColor = _bakedConstantMap[output].color;
                    string uniformColorString = getValueStringFromColor(uniformColor, input->getType());
                    string defaultValueString = input->hasValue() ? input->getValue()->getValueString() : EMPTY_STRING;
                    if (uniformColorString == defaultValueString)
                    {
                        _bakedConstantMap[output].isDefault = true;
                    }
                }
            }
        }
    }

    // Remove baked images that have been replaced by constant values.
    for (auto& pair : _bakedConstantMap)
    {
        if (pair.second.isDefault || _optimizeConstants || _averageImages)
        {
            _bakedImageMap.erase(pair.first);
        }
    }
}

template <typename Renderer, typename ShaderGen>
DocumentPtr TextureBaker<Renderer, ShaderGen>::generateNewDocumentFromShader(NodePtr shader, const StringVec& udimSet)
{
    if (!shader)
    {
        return nullptr;
    }

    // Create document.
    if (!_bakedTextureDoc || _writeDocumentPerMaterial)
    {
        _bakedTextureDoc = createDocument();
    }
    if (shader->getDocument()->hasColorSpace())
    {
        _bakedTextureDoc->setColorSpace(shader->getDocument()->getColorSpace());
    }

    // Create node graph and geometry info.
    NodeGraphPtr bakedNodeGraph;
    if (!_bakedImageMap.empty())
    {
        _bakedGraphName = _bakedTextureDoc->createValidChildName(_bakedGraphName);
        bakedNodeGraph = _bakedTextureDoc->addNodeGraph(_bakedGraphName);
        bakedNodeGraph->setColorSpace(_colorSpace);
    }
    _bakedGeomInfoName = _bakedTextureDoc->createValidChildName(_bakedGeomInfoName);
    GeomInfoPtr bakedGeom = !udimSet.empty() ? _bakedTextureDoc->addGeomInfo(_bakedGeomInfoName) : nullptr;
    if (bakedGeom)
    {
        bakedGeom->setGeomPropValue(UDIM_SET_PROPERTY, udimSet, "stringarray");
    }

    // Create a shader node.
    NodePtr bakedShader = _bakedTextureDoc->addNode(shader->getCategory(), shader->getName(), shader->getType());

    // Optionally create a material node, connecting it to the new shader node.
    if (_material)
    {
        string materialName = (_texTemplateOverrides.count("$MATERIAL")) ? _texTemplateOverrides["$MATERIAL"] : _material->getName();
        NodePtr bakedMaterial = _bakedTextureDoc->addNode(_material->getCategory(), materialName, _material->getType());
        for (auto sourceMaterialInput : _material->getInputs())
        {
            const string& sourceMaterialInputName = sourceMaterialInput->getName();
            NodePtr upstreamShader = sourceMaterialInput->getConnectedNode();
            if (upstreamShader && (upstreamShader->getNamePath() == shader->getNamePath()))
            {
                InputPtr bakedMaterialInput = bakedMaterial->getInput(sourceMaterialInputName);
                if (!bakedMaterialInput)
                {
                    bakedMaterialInput = bakedMaterial->addInput(sourceMaterialInputName, sourceMaterialInput->getType());
                }
                bakedMaterialInput->setNodeName(bakedShader->getName());
            }
        }
    }

    // Create and connect inputs on the new shader node.
    for (ValueElementPtr valueElem : shader->getChildrenOfType<ValueElement>())
    {
        // Get the source input and its connected output.
        InputPtr sourceInput = valueElem->asA<Input>();
        if (!sourceInput)
        {
            continue;
        }

        OutputPtr output = sourceInput->getConnectedOutput();

        // Skip uniform outputs at their default values.
        if (output && _bakedConstantMap.count(output) && _bakedConstantMap[output].isDefault)
        {
            continue;
        }

        // Find or create the baked input.
        const string& sourceName = sourceInput->getName();
        const string& sourceType = sourceInput->getType();
        InputPtr bakedInput = bakedShader->getInput(sourceName);
        if (!bakedInput)
        {
            bakedInput = bakedShader->addInput(sourceName, sourceType);
        }

        // Assign image or constant data to the baked input.
        if (output)
        {
            // Store a constant value for uniform outputs.
            if (_optimizeConstants && _bakedConstantMap.count(output))
            {
                Color4 uniformColor = _bakedConstantMap[output].color;
                string uniformColorString = getValueStringFromColor(uniformColor, bakedInput->getType());
                bakedInput->setValueString(uniformColorString);
                if (bakedInput->isColorType())
                {
                    bakedInput->setColorSpace(_colorSpace);
                }
                continue;
            }

            if (!_bakedImageMap.empty())
            {
                // Add the image node.
                NodePtr bakedImage = bakedNodeGraph->addNode("image", sourceName, sourceType);
                InputPtr input = bakedImage->addInput("file", "filename");
                StringMap filenameTemplateMap = initializeFileTemplateMap(bakedInput, shader, udimSet.empty() ? EMPTY_STRING : UDIM_TOKEN);
                input->setValueString(generateTextureFilename(filenameTemplateMap));

                // Reconstruct any world-space nodes that were excluded from the baking process.
                auto worldSpacePair = _worldSpaceNodes.find(sourceInput->getName());
                if (worldSpacePair != _worldSpaceNodes.end())
                {
                    NodePtr origWorldSpaceNode = worldSpacePair->second;
                    if (origWorldSpaceNode)
                    {
                        NodePtr newWorldSpaceNode = bakedNodeGraph->addNode(origWorldSpaceNode->getCategory(), sourceName + "_map", sourceType);
                        newWorldSpaceNode->copyContentFrom(origWorldSpaceNode);
                        InputPtr mapInput = newWorldSpaceNode->getInput("in");
                        if (mapInput)
                        {
                            mapInput->setNodeName(bakedImage->getName());
                        }
                        bakedImage = newWorldSpaceNode;
                    }
                }

                // Add the graph output.
                OutputPtr bakedOutput = bakedNodeGraph->addOutput(sourceName + "_output", sourceType);
                bakedOutput->setConnectedNode(bakedImage);
                bakedInput->setConnectedOutput(bakedOutput);
            }
        }
        else
        {
            bakedInput->copyContentFrom(sourceInput);
        }
    }

    // Generate uniform images and write to disk.
    ImagePtr uniformImage = createUniformImage(4, 4, 4, Renderer::_baseType, Color4());
    for (const auto& pair : _bakedImageMap)
    {
        for (const BakedImage& baked : pair.second)
        {
            if (baked.isUniform)
            {
                uniformImage->setUniformColor(baked.uniformColor);
                writeBakedImage(baked, uniformImage);
            }
        }
    }

    // Clear cached information after each material bake
    _bakedImageMap.clear();
    _bakedConstantMap.clear();
    _worldSpaceNodes.clear();
    _bakedInputMap.clear();
    _material = nullptr;

    // Return the baked document on success.
    return _bakedTextureDoc;
}

template <typename Renderer, typename ShaderGen>
DocumentPtr TextureBaker<Renderer, ShaderGen>::bakeMaterialToDoc(DocumentPtr doc, const FileSearchPath& searchPath, const string& materialPath,
                                                                 const StringVec& udimSet, string& documentName)
{
    if (_outputStream)
    {
        *_outputStream << "Processing material: " << materialPath << std::endl;
    }

    // Set up generator context for material
    GenContext genContext(_generator);
    genContext.getOptions().targetColorSpaceOverride = LIN_REC709;
    genContext.getOptions().fileTextureVerticalFlip = true;
    genContext.getOptions().targetDistanceUnit = _distanceUnit;

    DefaultColorManagementSystemPtr cms = DefaultColorManagementSystem::create(genContext.getShaderGenerator().getTarget());
    cms->loadLibrary(doc);
    genContext.registerSourceCodeSearchPath(searchPath);
    genContext.getShaderGenerator().setColorManagementSystem(cms);

    // Compute the material tag set.
    StringVec materialTags = udimSet;
    if (materialTags.empty())
    {
        materialTags.push_back(EMPTY_STRING);
    }

    ElementPtr elem = doc->getDescendant(materialPath);
    if (!elem || !elem->isA<Node>())
    {
        return nullptr;
    }
    NodePtr materialNode = elem->asA<Node>();

    vector<NodePtr> shaderNodes = getShaderNodes(materialNode);
    NodePtr shaderNode = shaderNodes.empty() ? nullptr : shaderNodes[0];
    if (!shaderNode)
    {
        return nullptr;
    }

    StringResolverPtr resolver = StringResolver::create();

    // Iterate over material tags.
    for (const string& tag : materialTags)
    {
        // Always clear any cached implementations before generation.
        genContext.clearNodeImplementations();

        ShaderPtr hwShader = createShader("Shader", genContext, shaderNode);
        if (!hwShader)
        {
            continue;
        }
        Renderer::_imageHandler->setSearchPath(searchPath);
        resolver->setUdimString(tag);
        Renderer::_imageHandler->setFilenameResolver(resolver);
        bakeShaderInputs(materialNode, shaderNode, genContext, tag);

        // Optimize baked textures.
        optimizeBakedTextures(shaderNode);
    }

    // Link the baked material and textures in a MaterialX document.
    documentName = shaderNode->getName();
    return generateNewDocumentFromShader(shaderNode, udimSet);
}

template <typename Renderer, typename ShaderGen>
void TextureBaker<Renderer, ShaderGen>::bakeAllMaterials(DocumentPtr doc, const FileSearchPath& searchPath, const FilePath& outputFilename)
{
    if (_outputImagePath.isEmpty())
    {
        _outputImagePath = outputFilename.getParentPath();
        if (!_outputImagePath.exists())
        {
            _outputImagePath.createDirectory();
        }
    }

    std::vector<TypedElementPtr> renderableMaterials = findRenderableElements(doc);

    // Compute the UDIM set.
    ValuePtr udimSetValue = doc->getGeomPropValue(UDIM_SET_PROPERTY);
    StringVec udimSet;
    if (udimSetValue && udimSetValue->isA<StringVec>())
    {
        udimSet = udimSetValue->asA<StringVec>();
    }

    // Bake all materials in documents to memory.
    BakedDocumentVec bakedDocuments;
    for (size_t i = 0; i < renderableMaterials.size(); i++)
    {
        if (_outputStream && i > 0)
        {
            *_outputStream << std::endl;
        }

        const TypedElementPtr& element = renderableMaterials[i];
        string documentName;
        DocumentPtr bakedMaterialDoc = bakeMaterialToDoc(doc, searchPath, element->getNamePath(), udimSet, documentName);
        if (_writeDocumentPerMaterial && bakedMaterialDoc)
        {
            bakedDocuments.push_back(make_pair(documentName, bakedMaterialDoc));
        }
    }

    if (_writeDocumentPerMaterial)
    {
        // Write documents in memory to disk.
        size_t bakeCount = bakedDocuments.size();
        for (size_t i = 0; i < bakeCount; i++)
        {
            if (bakedDocuments[i].second)
            {
                FilePath writeFilename = outputFilename;

                // Add additional filename decorations if there are multiple documents.
                if (bakedDocuments.size() > 1)
                {
                    const string extension = writeFilename.getExtension();
                    writeFilename.removeExtension();
                    string filenameSeparator = writeFilename.isDirectory() ? EMPTY_STRING : "_";
                    writeFilename = FilePath(writeFilename.asString() + filenameSeparator + bakedDocuments[i].first + "." + extension);
                }

                writeToXmlFile(bakedDocuments[i].second, writeFilename);
                if (_outputStream)
                {
                    *_outputStream << "Wrote baked document: " << writeFilename.asString() << std::endl;
                }
            }
        }
    }
    else if (_bakedTextureDoc)
    {
        writeToXmlFile(_bakedTextureDoc, outputFilename);
        if (_outputStream)
        {
            *_outputStream << "Wrote baked document: " << outputFilename.asString() << std::endl;
        }
    }
}

template <typename Renderer, typename ShaderGen>
void TextureBaker<Renderer, ShaderGen>::setupUnitSystem(DocumentPtr unitDefinitions)
{
    UnitTypeDefPtr distanceTypeDef = unitDefinitions ? unitDefinitions->getUnitTypeDef("distance") : nullptr;
    UnitTypeDefPtr angleTypeDef = unitDefinitions ? unitDefinitions->getUnitTypeDef("angle") : nullptr;
    if (!distanceTypeDef && !angleTypeDef)
    {
        return;
    }

    UnitSystemPtr unitSystem = UnitSystem::create(_generator->getTarget());
    if (!unitSystem)
    {
        return;
    }
    _generator->setUnitSystem(unitSystem);
    UnitConverterRegistryPtr registry = UnitConverterRegistry::create();
    registry->addUnitConverter(distanceTypeDef, LinearUnitConverter::create(distanceTypeDef));
    registry->addUnitConverter(angleTypeDef, LinearUnitConverter::create(angleTypeDef));
    _generator->getUnitSystem()->loadLibrary(unitDefinitions);
    _generator->getUnitSystem()->setUnitConverterRegistry(registry);
}

MATERIALX_NAMESPACE_END
