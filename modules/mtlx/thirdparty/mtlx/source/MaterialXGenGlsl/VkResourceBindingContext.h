//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_VKRESOURCEBINDING_H
#define MATERIALX_VKRESOURCEBINDING_H

/// @file
/// Vulkan GLSL resource binding context

#include <MaterialXGenGlsl/Export.h>

#include <MaterialXGenShader/HwShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Shared pointer to a VkResourceBindingContext
using VkResourceBindingContextPtr = shared_ptr<class VkResourceBindingContext>;

/// @class VkResourceBindingContext
/// Class representing a resource binding for Vulkan Glsl shader resources.
class MX_GENGLSL_API VkResourceBindingContext : public HwResourceBindingContext
{
  public:
    VkResourceBindingContext(size_t uniformBindingLocation);

    static VkResourceBindingContextPtr create(size_t uniformBindingLocation = 0)
    {
        return std::make_shared<VkResourceBindingContext>(uniformBindingLocation);
    }

    // Initialize the context before generation starts.
    void initialize() override;

    // Emit directives for stage
    void emitDirectives(GenContext& context, ShaderStage& stage) override;

    // Emit uniforms with binding information
    void emitResourceBindings(GenContext& context, const VariableBlock& uniforms, ShaderStage& stage) override;

    // Emit structured uniforms with binding information and align members where possible
    void emitStructuredResourceBindings(GenContext& context, const VariableBlock& uniforms,
                                        ShaderStage& stage, const std::string& structInstanceName,
                                        const std::string& arraySuffix) override;

  protected:
    // Binding location for uniform blocks
    size_t _hwUniformBindLocation = 0;

    // Initial value of uniform binding location
    size_t _hwInitUniformBindLocation = 0;
};

MATERIALX_NAMESPACE_END

#endif
