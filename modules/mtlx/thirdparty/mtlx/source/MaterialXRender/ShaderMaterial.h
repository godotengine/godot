//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_RENDER_SHADER_MATERIAL_H
#define MATERIALX_RENDER_SHADER_MATERIAL_H

/// @file
/// ShaderMaterial helper classes

#include <MaterialXCore/Generated.h>
#include <MaterialXRender/Image.h>
#include <MaterialXRender/Mesh.h>
#include <MaterialXRender/ImageHandler.h>
#include <MaterialXRender/LightHandler.h>
#include <MaterialXRender/Camera.h>
#include <MaterialXGenShader/UnitSystem.h>

#include <memory>

MATERIALX_NAMESPACE_BEGIN

using MaterialPtr = std::shared_ptr<class ShaderMaterial>;

/// @class ShadowState
/// Helper class representing shadow rendering state
class MX_RENDER_API ShadowState
{
  public:
    ImagePtr shadowMap;
    Matrix44 shadowMatrix;
    ImagePtr ambientOcclusionMap;
    float ambientOcclusionGain = 0.0f;
};

/// @class ShaderMaterial
/// Abstract class for shader generation and rendering of a ShaderMaterial
class MX_RENDER_API ShaderMaterial
{
  public:
    ShaderMaterial();
    virtual ~ShaderMaterial();

    /// Set the renderable element associated with this ShaderMaterial
    void setDocument(DocumentPtr doc);

    /// Return the document associated with this ShaderMaterial
    DocumentPtr getDocument() const;

    /// Set the renderable element associated with this ShaderMaterial
    void setElement(TypedElementPtr val);

    /// Return the renderable element associated with this ShaderMaterial
    TypedElementPtr getElement() const;

    /// Set the ShaderMaterial node associated with this ShaderMaterial
    void setMaterialNode(NodePtr node);

    /// Return the ShaderMaterial node associated with this ShaderMaterial
    NodePtr getMaterialNode() const;

    /// Set udim identifier
    void setUdim(const std::string& val);

    /// Get any associated udim identifier
    const std::string& getUdim();

    /// Load shader source from file.
    virtual bool loadSource(const FilePath& vertexShaderFile,
                            const FilePath& pixelShaderFile,
                            bool hasTransparency) = 0;

    /// Generate a shader from our currently stored element and
    /// the given generator context.
    virtual bool generateShader(GenContext& context) = 0;

    /// Copies shader and API specific generated program from ShaderMaterial to this one.
    virtual void copyShader(MaterialPtr ShaderMaterial) = 0;

    /// Generate a shader from the given hardware shader.
    virtual bool generateShader(ShaderPtr hwShader) = 0;

    /// Generate an environment background shader
    virtual bool generateEnvironmentShader(GenContext& context,
                                           const FilePath& filename,
                                           DocumentPtr stdLib,
                                           const FilePath& imagePath);

    /// Return the underlying hardware shader.
    ShaderPtr getShader() const;

    /// Return true if this ShaderMaterial has transparency.
    bool hasTransparency() const;

    /// Bind shader
    virtual bool bindShader() const = 0;

    /// Bind viewing information for this ShaderMaterial.
    virtual void bindViewInformation(CameraPtr camera) = 0;

    /// Bind all images for this ShaderMaterial.
    virtual void bindImages(ImageHandlerPtr imageHandler,
                            const FileSearchPath& searchPath,
                            bool enableMipmaps = true) = 0;

    /// Unbbind all images for this ShaderMaterial.
    virtual void unbindImages(ImageHandlerPtr imageHandler) = 0;

    /// Bind a single image.
    virtual ImagePtr bindImage(const FilePath& filePath,
                               const std::string& uniformName,
                               ImageHandlerPtr imageHandler,
                               const ImageSamplingProperties& samplingProperties) = 0;

    /// Bind lights to shader.
    virtual void bindLighting(LightHandlerPtr lightHandler,
                              ImageHandlerPtr imageHandler,
                              const ShadowState& shadowState) = 0;

    /// Bind the given mesh to this ShaderMaterial.
    virtual void bindMesh(MeshPtr mesh) = 0;

    /// Bind a mesh partition to this ShaderMaterial.
    virtual bool bindPartition(MeshPartitionPtr part) const = 0;

    /// Draw the given mesh partition.
    virtual void drawPartition(MeshPartitionPtr part) const = 0;

    /// Unbind all geometry from this ShaderMaterial.
    virtual void unbindGeometry() = 0;

    /// Return the block of public uniforms for this ShaderMaterial.
    virtual VariableBlock* getPublicUniforms() const = 0;

    /// Find a public uniform from its MaterialX path.
    virtual ShaderPort* findUniform(const std::string& path) const = 0;

    /// Modify the value of the uniform with the given path.
    virtual void modifyUniform(const std::string& path,
                               ConstValuePtr value,
                               std::string valueString = EMPTY_STRING) = 0;

  protected:
    virtual void clearShader() = 0;

  protected:
    ShaderPtr _hwShader;
    MeshPtr _boundMesh;

    DocumentPtr _doc;
    TypedElementPtr _elem;
    NodePtr _materialNode;

    std::string _udim;
    bool _hasTransparency;

    ImageVec _boundImages;
};

MATERIALX_NAMESPACE_END

#endif
