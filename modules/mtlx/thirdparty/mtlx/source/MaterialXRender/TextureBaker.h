//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_TEXTUREBAKER
#define MATERIALX_TEXTUREBAKER

/// @file
/// Texture baking functionality

#include <iostream>

#include <MaterialXCore/Unit.h>

#include <MaterialXRender/Export.h>
#include <MaterialXFormat/File.h>
#include <MaterialXRender/ImageHandler.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

/// A vector of baked documents with their associated names.
using BakedDocumentVec = std::vector<std::pair<std::string, DocumentPtr>>;

/// @class TextureBaker
/// A helper class for baking procedural material content to textures.
/// TODO: Add support for graphs containing geometric nodes such as position
///       and normal.
template <typename Renderer, typename ShaderGen>
class TextureBaker : public Renderer
{
  public:
    /// Set the file extension for baked textures.
    void setExtension(const string& extension)
    {
        _extension = extension;
    }

    /// Return the file extension for baked textures.
    const string& getExtension() const
    {
        return _extension;
    }

    /// Set the color space in which color textures are encoded.
    ///
    /// By default, this color space is srgb_texture, and color inputs are
    /// automatically transformed to this space by the baker.  If another color
    /// space is set, then the input graph is responsible for transforming
    /// colors to this space.
    void setColorSpace(const string& colorSpace)
    {
        _colorSpace = colorSpace;
    }

    /// Return the color space in which color textures are encoded.
    const string& getColorSpace() const
    {
        return _colorSpace;
    }

    /// Set the distance unit to which textures are baked.  Defaults to meters.
    void setDistanceUnit(const string& unitSpace)
    {
        _distanceUnit = unitSpace;
    }

    /// Return the distance unit to which textures are baked.
    const string& getDistanceUnit() const
    {
        return _distanceUnit;
    }

    /// Set whether images should be averaged to generate constants.  Defaults to false.
    void setAverageImages(bool enable)
    {
        _averageImages = enable;
    }

    /// Return whether images should be averaged to generate constants.
    bool getAverageImages() const
    {
        return _averageImages;
    }

    /// Set whether uniform textures should be stored as constants.  Defaults to true.
    void setOptimizeConstants(bool enable)
    {
        _optimizeConstants = enable;
    }

    /// Return whether uniform textures should be stored as constants.
    bool getOptimizeConstants() const
    {
        return _optimizeConstants;
    }

    /// Set the output location for baked texture images.  Defaults to the root folder
    /// of the destination material.
    void setOutputImagePath(const FilePath& outputImagePath)
    {
        _outputImagePath = outputImagePath;
    }

    /// Get the current output location for baked texture images.
    const FilePath& getOutputImagePath()
    {
        return _outputImagePath;
    }

    /// Set the name of the baked graph element.
    void setBakedGraphName(const string& name)
    {
        _bakedGraphName = name;
    }

    /// Return the name of the baked graph element.
    const string& getBakedGraphName() const
    {
        return _bakedGraphName;
    }

    /// Set the name of the baked geometry info element.
    void setBakedGeomInfoName(const string& name)
    {
        _bakedGeomInfoName = name;
    }

    /// Return the name of the baked geometry info element.
    const string& getBakedGeomInfoName() const
    {
        return _bakedGeomInfoName;
    }

    /// Get the texture filename template.
    const string& getTextureFilenameTemplate() const
    {
        return _textureFilenameTemplate;
    }

    /// Set the texture filename template.
    void setTextureFilenameTemplate(const string& filenameTemplate)
    {
        _textureFilenameTemplate = (filenameTemplate.find("$EXTENSION") == string::npos) ?
            filenameTemplate + ".$EXTENSION" : filenameTemplate;
    }

    /// Set texFilenameOverrides if template variable exists.
    void setFilenameTemplateVarOverride(const string& key, const string& value)
    {
        if (_permittedOverrides.count(key))
        {
            _texTemplateOverrides[key] = value;
        }
    }

    /// Set the output stream for reporting progress and warnings.  Defaults to std::cout.
    void setOutputStream(std::ostream* outputStream)
    {
        _outputStream = outputStream;
    }

    /// Return the output stream for reporting progress and warnings.
    std::ostream* getOutputStream() const
    {
        return _outputStream;
    }

    /// Set whether to create a short name for baked images by hashing the baked image filenames
    /// This is useful for file systems which may have a maximum limit on filename size.
    /// By default names are not hashed.
    void setHashImageNames(bool enable)
    {
        _hashImageNames = enable;
    }

    /// Return whether automatic baked texture resolution is set.
    bool getHashImageNames() const
    {
        return _hashImageNames;
    }

    /// Set the minimum texcoords used in texture baking.  Defaults to 0, 0.
    void setTextureSpaceMin(const Vector2& min)
    {
        _textureSpaceMin = min;
    }

    /// Return the minimum texcoords used in texture baking.
    Vector2 getTextureSpaceMin() const
    {
        return _textureSpaceMin;
    }

    /// Set the maximum texcoords used in texture baking.  Defaults to 1, 1.
    void setTextureSpaceMax(const Vector2& max)
    {
        _textureSpaceMax = max;
    }

    /// Return the maximum texcoords used in texture baking.
    Vector2 getTextureSpaceMax() const
    {
        return _textureSpaceMax;
    }

    /// Set up the unit definitions to be used in baking.
    void setupUnitSystem(DocumentPtr unitDefinitions);

    /// Bake textures for all graph inputs of the given shader.
    void bakeShaderInputs(NodePtr material, NodePtr shader, GenContext& context, const string& udim = EMPTY_STRING);

    /// Bake a texture for the given graph output.
    void bakeGraphOutput(OutputPtr output, GenContext& context, const StringMap& filenameTemplateMap);

    /// Optimize baked textures before writing.
    void optimizeBakedTextures(NodePtr shader);

    /// Bake material to document in memory and write baked textures to disk.
    DocumentPtr bakeMaterialToDoc(DocumentPtr doc, const FileSearchPath& searchPath, const string& materialPath,
                                  const StringVec& udimSet, std::string& documentName);

    /// Bake materials in the given document and write them to disk.  If multiple documents are written,
    /// then the given output filename will be used as a template.
    void bakeAllMaterials(DocumentPtr doc, const FileSearchPath& searchPath, const FilePath& outputFileName);

    /// Set whether to write a separate document per material when calling bakeAllMaterials.
    /// By default separate documents are written.
    void writeDocumentPerMaterial(bool value)
    {
        _writeDocumentPerMaterial = value;
    }

    string getValueStringFromColor(const Color4& color, const string& type);

  protected:
    class BakedImage
    {
      public:
        FilePath filename;
        Color4 uniformColor;
        bool isUniform = false;
    };
    class BakedConstant
    {
      public:
        Color4 color;
        bool isDefault = false;
    };
    using BakedImageVec = vector<BakedImage>;
    using BakedImageMap = std::unordered_map<OutputPtr, BakedImageVec>;
    using BakedConstantMap = std::unordered_map<OutputPtr, BakedConstant>;

  protected:
    TextureBaker(unsigned int width, unsigned int height, Image::BaseType baseType, bool flipSavedImage);

    // Populate file template variable naming map
    StringMap initializeFileTemplateMap(InputPtr input, NodePtr shader, const string& udim = EMPTY_STRING);

    // Find first occurence of variable in filename from start index onwards
    size_t findVarInTemplate(const string& filename, const string& var, size_t start = 0);

    // Generate a texture filename for the given graph output.
    FilePath generateTextureFilename(const StringMap& fileTemplateMap);

    // Create document that links shader outputs to a material.
    DocumentPtr generateNewDocumentFromShader(NodePtr shader, const StringVec& udimSet);

    // Write a baked image to disk, returning true if the write was successful.
    bool writeBakedImage(const BakedImage& baked, ImagePtr image);

  protected:
    string _extension;
    string _colorSpace;
    string _distanceUnit;
    bool _averageImages;
    bool _optimizeConstants;
    FilePath _outputImagePath;
    string _bakedGraphName;
    string _bakedGeomInfoName;
    string _textureFilenameTemplate;
    std::ostream* _outputStream;
    bool _hashImageNames;
    Vector2 _textureSpaceMin;
    Vector2 _textureSpaceMax;

    ShaderGeneratorPtr _generator;
    ConstNodePtr _material;
    ImagePtr _frameCaptureImage;
    BakedImageMap _bakedImageMap;
    BakedConstantMap _bakedConstantMap;
    StringSet _permittedOverrides;
    StringMap _texTemplateOverrides;
    StringMap _bakedInputMap;

    std::unordered_map<string, NodePtr> _worldSpaceNodes;

    bool _flipSavedImage;

    bool _writeDocumentPerMaterial;
    DocumentPtr _bakedTextureDoc;
};

MATERIALX_NAMESPACE_END

#include <MaterialXRender/TextureBaker.inl>

#endif
