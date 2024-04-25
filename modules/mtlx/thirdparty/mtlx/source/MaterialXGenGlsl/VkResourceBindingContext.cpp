//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/VkResourceBindingContext.h>

MATERIALX_NAMESPACE_BEGIN

//
// VkResourceBindingContext methods
//

VkResourceBindingContext::VkResourceBindingContext(size_t uniformBindingLocation) :
    _hwInitUniformBindLocation(uniformBindingLocation)
{
}

void VkResourceBindingContext::initialize()
{
    // Reset bind location counter.
    _hwUniformBindLocation = _hwInitUniformBindLocation;
}

void VkResourceBindingContext::emitDirectives(GenContext& context, ShaderStage& stage)
{
    const ShaderGenerator& generator = context.getShaderGenerator();

    // Write shader stage directives for Vulkan compliance
    std::string shaderStage;
    if (stage.getName() == Stage::VERTEX)
    {
        shaderStage = "vertex";
    }
    else if (stage.getName() == Stage::PIXEL)
    {
        shaderStage = "fragment";
    }

    if (!shaderStage.empty())
    {
        generator.emitLine("#pragma shader_stage(" + shaderStage + ")", stage, false);
    }
}

void VkResourceBindingContext::emitResourceBindings(GenContext& context, const VariableBlock& uniforms, ShaderStage& stage)
{
    const ShaderGenerator& generator = context.getShaderGenerator();
    const Syntax& syntax = generator.getSyntax();

    // First, emit all value uniforms in a block with single layout binding
    bool hasValueUniforms = false;
    for (auto uniform : uniforms.getVariableOrder())
    {
        if (uniform->getType() != Type::FILENAME)
        {
            hasValueUniforms = true;
            break;
        }
    }
    if (hasValueUniforms)
    {
        generator.emitLine("layout (std140, binding=" + std::to_string(_hwUniformBindLocation++) + ") " +
                               syntax.getUniformQualifier() + " " + uniforms.getName() + "_" + stage.getName(),
                           stage, false);
        generator.emitScopeBegin(stage);
        for (auto uniform : uniforms.getVariableOrder())
        {
            if (uniform->getType() != Type::FILENAME)
            {
                generator.emitLineBegin(stage);
                generator.emitVariableDeclaration(uniform, EMPTY_STRING, context, stage, false);
                generator.emitString(Syntax::SEMICOLON, stage);
                generator.emitLineEnd(stage, false);
            }
        }
        generator.emitScopeEnd(stage, true);
    }

    // Second, emit all sampler uniforms as separate uniforms with separate layout bindings
    for (auto uniform : uniforms.getVariableOrder())
    {
        if (*uniform->getType() == *Type::FILENAME)
        {
            generator.emitString("layout (binding=" + std::to_string(_hwUniformBindLocation++) + ") " + syntax.getUniformQualifier() + " ", stage);
            generator.emitVariableDeclaration(uniform, EMPTY_STRING, context, stage, false);
            generator.emitLineEnd(stage, true);
        }
    }

    generator.emitLineBreak(stage);
}

void VkResourceBindingContext::emitStructuredResourceBindings(GenContext& context, const VariableBlock& uniforms,
                                                              ShaderStage& stage, const std::string& structInstanceName,
                                                              const std::string& arraySuffix)
{
    const ShaderGenerator& generator = context.getShaderGenerator();
    const Syntax& syntax = generator.getSyntax();

    // Glsl structures need to be aligned. We make a best effort to base align struct members and add
    // padding if required.
    // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_uniform_buffer_object.txt

    const size_t baseAlignment = 16;
    std::unordered_map<const TypeDesc*, size_t> alignmentMap({ { Type::FLOAT, baseAlignment / 4 },
                                                               { Type::INTEGER, baseAlignment / 4 },
                                                               { Type::BOOLEAN, baseAlignment / 4 },
                                                               { Type::COLOR3, baseAlignment },
                                                               { Type::COLOR4, baseAlignment },
                                                               { Type::VECTOR2, baseAlignment },
                                                               { Type::VECTOR3, baseAlignment },
                                                               { Type::VECTOR4, baseAlignment },
                                                               { Type::MATRIX33, baseAlignment * 4 },
                                                               { Type::MATRIX44, baseAlignment * 4 } });

    // Get struct alignment and size
    // alignment, uniform member index
    vector<std::pair<size_t, size_t>> memberOrder;
    size_t structSize = 0;
    for (size_t i = 0; i < uniforms.size(); ++i)
    {
        auto it = alignmentMap.find(uniforms[i]->getType());
        if (it == alignmentMap.end())
        {
            structSize += baseAlignment;
            memberOrder.push_back(std::make_pair(baseAlignment, i));
        }
        else
        {
            structSize += it->second;
            memberOrder.push_back(std::make_pair(it->second, i));
        }
    }

    // Align up and determine number of padding floats to add
    const size_t numPaddingfloats =
        (((structSize + (baseAlignment - 1)) & ~(baseAlignment - 1)) - structSize) / 4;

    // Sort order from largest to smallest
    std::sort(memberOrder.begin(), memberOrder.end(),
              [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b)
              {
        return a.first > b.first;
    });

    // Emit the struct
    generator.emitLine("struct " + uniforms.getName(), stage, false);
    generator.emitScopeBegin(stage);

    for (size_t i = 0; i < uniforms.size(); ++i)
    {
        size_t variableIndex = memberOrder[i].second;
        generator.emitLineBegin(stage);
        generator.emitVariableDeclaration(
            uniforms[variableIndex], EMPTY_STRING, context, stage, false);
        generator.emitString(Syntax::SEMICOLON, stage);
        generator.emitLineEnd(stage, false);
    }

    // Emit padding
    for (size_t i = 0; i < numPaddingfloats; ++i)
    {
        generator.emitLine("float pad" + std::to_string(i), stage, true);
    }
    generator.emitScopeEnd(stage, true);

    // Emit binding information
    generator.emitLineBreak(stage);
    generator.emitLine("layout (std140, binding=" +
                           std::to_string(_hwUniformBindLocation++) + ") " +
                           syntax.getUniformQualifier() + " " + uniforms.getName() + "_" +
                           stage.getName(),
                       stage, false);
    generator.emitScopeBegin(stage);
    generator.emitLine(uniforms.getName() + " " + structInstanceName + arraySuffix, stage);
    generator.emitScopeEnd(stage, true);
}

MATERIALX_NAMESPACE_END
