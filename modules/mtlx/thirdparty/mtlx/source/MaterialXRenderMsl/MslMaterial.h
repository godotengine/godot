//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_MSLMATERIAL_H
#define MATERIALX_MSLMATERIAL_H

/// @file
/// GLSL material helper classes

#include <MaterialXRenderMsl/Export.h>
#include <MaterialXRender/ShaderMaterial.h>

#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <MaterialXGenShader/UnitSystem.h>

MATERIALX_NAMESPACE_BEGIN

using GeometryHandlerPtr = std::shared_ptr<class GeometryHandler>;
using MslProgramPtr = std::shared_ptr<class MslProgram>;
using MslMaterialPtr = std::shared_ptr<class MslMaterial>;

/// @class MslMaterial
/// Helper class for MSL generation and rendering of a material
class MX_RENDERMSL_API MslMaterial : public ShaderMaterial
{
  public:
    MslMaterial() : ShaderMaterial()
    {
    }
    ~MslMaterial() { }

    static MslMaterialPtr create()
    {
        return std::make_shared<MslMaterial>();
    }

    /// Load shader source from file.
    bool loadSource(const FilePath& vertexShaderFile,
                    const FilePath& pixelShaderFile,
                    bool hasTransparency) override;

    /// Generate a shader from our currently stored element and
    /// the given generator context.
    bool generateShader(GenContext& context) override;

    /// Generate a shader from the given hardware shader.
    bool generateShader(ShaderPtr hwShader) override;
    
    /// Copy shader from one material to this one
    void copyShader(MaterialPtr material) override
    {
        _hwShader = std::static_pointer_cast<MslMaterial>(material)->_hwShader;
        _glProgram = std::static_pointer_cast<MslMaterial>(material)->_glProgram;
    }

    /// Return the underlying MSL program.
    MslProgramPtr getProgram() const
    {
        return _glProgram;
    }

    /// Bind shader
    bool bindShader() const override;

    /// Bind viewing information for this material.
    void bindViewInformation(CameraPtr camera) override;

    /// Bind all images for this material.
    void bindImages(ImageHandlerPtr imageHandler,
                    const FileSearchPath& searchPath,
                    bool enableMipmaps = true) override;

    /// Unbbind all images for this material.
    void unbindImages(ImageHandlerPtr imageHandler) override;

    /// Bind a single image.
    ImagePtr bindImage(const FilePath& filePath,
                       const std::string& uniformName,
                       ImageHandlerPtr imageHandler,
                       const ImageSamplingProperties& samplingProperties) override;

    /// Bind lights to shader.
    void bindLighting(LightHandlerPtr lightHandler,
                      ImageHandlerPtr imageHandler,
                      const ShadowState& shadowState) override;

    /// Bind the given mesh to this material.
    void bindMesh(MeshPtr mesh) override;

    /// Bind a mesh partition to this material.
    bool bindPartition(MeshPartitionPtr part) const override;

    /// Draw the given mesh partition.
    void drawPartition(MeshPartitionPtr part) const override;

    /// Unbind all geometry from this material.
    void unbindGeometry() override;

    /// Return the block of public uniforms for this material.
    VariableBlock* getPublicUniforms() const override;

    /// Find a public uniform from its MaterialX path.
    ShaderPort* findUniform(const std::string& path) const override;

    /// Modify the value of the uniform with the given path.
    void modifyUniform(const std::string& path,
                       ConstValuePtr value,
                       std::string valueString = EMPTY_STRING) override;

    void prepareUsedResources(CameraPtr cam,
                              GeometryHandlerPtr geometryHandler,
                              ImageHandlerPtr imageHandler,
                              LightHandlerPtr lightHandler);
  protected:
    void clearShader() override;
    
  protected:
    MslProgramPtr _glProgram;
};

MATERIALX_NAMESPACE_END

#endif
