//
// Copyright (c) 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// GlslangUtils: Wrapper for Khronos's glslang compiler.
//

#include "libANGLE/renderer/metal/mtl_glslang_utils.h"

#include <regex>

#include <spirv_msl.hpp>

#include "common/apple_platform_utils.h"
#include "compiler/translator/TranslatorMetal.h"
#include "libANGLE/renderer/glslang_wrapper_utils.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"

namespace rx
{
namespace mtl
{
namespace
{

constexpr uint32_t kGlslangTextureDescSet              = 0;
constexpr uint32_t kGlslangDefaultUniformAndXfbDescSet = 1;
constexpr uint32_t kGlslangDriverUniformsDescSet       = 2;
constexpr uint32_t kGlslangShaderResourceDescSet       = 3;
constexpr uint32_t kGlslangXfbStartingIndex            = 1;

constexpr char kShadowSamplerCompareModesVarName[] = "ANGLEShadowCompareModes";

// Original mapping of front end from sampler name to multiple sampler slots (in form of
// slot:count pair)
using OriginalSamplerBindingMap =
    std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint32_t>>>;

angle::Result HandleError(ErrorHandler *context, GlslangError)
{
    ANGLE_MTL_TRY(context, false);
    return angle::Result::Stop;
}

GlslangSourceOptions CreateSourceOptions()
{
    GlslangSourceOptions options;
    // These are binding options passed to glslang. The actual binding might be changed later
    // by spirv-cross.
    options.uniformsAndXfbDescriptorSetIndex = kGlslangDefaultUniformAndXfbDescSet;
    options.textureDescriptorSetIndex        = kGlslangTextureDescSet;
    options.driverUniformsDescriptorSetIndex = kGlslangDriverUniformsDescSet;
    options.shaderResourceDescriptorSetIndex = kGlslangShaderResourceDescSet;

    options.xfbBindingIndexStart = kGlslangXfbStartingIndex;

    static_assert(kDefaultUniformsBindingIndex != 0, "kDefaultUniformsBindingIndex must not be 0");
    static_assert(kDriverUniformsBindingIndex != 0, "kDriverUniformsBindingIndex must not be 0");

    return options;
}

spv::ExecutionModel ShaderTypeToSpvExecutionModel(gl::ShaderType shaderType)
{
    switch (shaderType)
    {
        case gl::ShaderType::Vertex:
            return spv::ExecutionModelVertex;
        case gl::ShaderType::Fragment:
            return spv::ExecutionModelFragment;
        default:
            UNREACHABLE();
            return spv::ExecutionModelMax;
    }
}

void BindBuffers(spirv_cross::CompilerMSL *compiler,
                 const spirv_cross::SmallVector<spirv_cross::Resource> &resources,
                 gl::ShaderType shaderType,
                 const std::unordered_map<std::string, uint32_t> &uboOriginalBindings,
                 std::array<uint32_t, kMaxGLUBOBindings> *uboBindingsRemapOut,
                 std::array<uint32_t, kMaxShaderXFBs> *xfbBindingRemapOut,
                 bool *uboArgumentBufferUsed)
{
    auto &compilerMsl = *compiler;

    uint32_t totalUniformBufferSlots = 0;
    uint32_t totalXfbSlots           = 0;
    struct UniformBufferVar
    {
        const char *name = nullptr;
        spirv_cross::MSLResourceBinding resBinding;
        uint32_t arraySize;
    };
    std::vector<UniformBufferVar> uniformBufferBindings;

    for (const spirv_cross::Resource &resource : resources)
    {
        spirv_cross::MSLResourceBinding resBinding;
        resBinding.stage = ShaderTypeToSpvExecutionModel(shaderType);

        if (compilerMsl.has_decoration(resource.id, spv::DecorationDescriptorSet))
        {
            resBinding.desc_set =
                compilerMsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
        }

        if (!compilerMsl.has_decoration(resource.id, spv::DecorationBinding))
        {
            continue;
        }

        resBinding.binding = compilerMsl.get_decoration(resource.id, spv::DecorationBinding);

        uint32_t bindingPoint = 0;
        // NOTE(hqle): We use separate discrete binding point for now, in future, we should use
        // one argument buffer for each descriptor set.
        switch (resBinding.desc_set)
        {
            case kGlslangTextureDescSet:
                // Texture binding point is ignored. We let spirv-cross automatically assign it and
                // retrieve it later
                continue;
            case kGlslangDriverUniformsDescSet:
                bindingPoint = mtl::kDriverUniformsBindingIndex;
                break;
            case kGlslangDefaultUniformAndXfbDescSet:
                if (shaderType != gl::ShaderType::Vertex || resBinding.binding == 0)
                {
                    bindingPoint = mtl::kDefaultUniformsBindingIndex;
                }
                else if (resBinding.binding >= kGlslangXfbStartingIndex)
                {
                    totalXfbSlots++;
                    uint32_t bufferGlslangZeroIdx = resBinding.binding - kGlslangXfbStartingIndex;
                    // XFB buffer is allocated slot starting from last discrete Metal buffer slot.
                    bindingPoint = kMaxShaderBuffers - 1 - bufferGlslangZeroIdx;

                    xfbBindingRemapOut->at(bufferGlslangZeroIdx) = bindingPoint;
                }
                else
                {
                    continue;
                }
                break;
            case kGlslangShaderResourceDescSet: {
                UniformBufferVar uboVar;
                uboVar.name                       = resource.name.c_str();
                uboVar.resBinding                 = resBinding;
                const spirv_cross::SPIRType &type = compilerMsl.get_type_from_variable(resource.id);
                if (!type.array.empty())
                {
                    uboVar.arraySize = type.array[0];
                }
                else
                {
                    uboVar.arraySize = 1;
                }
                totalUniformBufferSlots += uboVar.arraySize;
                uniformBufferBindings.push_back(uboVar);
            }
                continue;
            default:
                // We don't support this descriptor set.
                continue;
        }

        resBinding.msl_buffer = bindingPoint;

        compilerMsl.add_msl_resource_binding(resBinding);
    }  // for (resources)

    if (totalUniformBufferSlots == 0)
    {
        return;
    }

    // Remap the uniform buffers bindings. glslang allows uniform buffers array to use exactly
    // one slot in the descriptor set. However, metal enforces that the uniform buffers array
    // use (n) slots where n=array size.
    uint32_t currentSlot = 0;
    uint32_t maxUBODiscreteSlots =
        kMaxShaderBuffers - totalXfbSlots - kUBOArgumentBufferBindingIndex;

    if (totalUniformBufferSlots > maxUBODiscreteSlots)
    {
        // If shader uses more than kMaxUBODiscreteBindingSlots number of UBOs, encode them all into
        // an argument buffer. Each buffer will be assigned [[id(n)]] attribute.
        *uboArgumentBufferUsed = true;
    }
    else
    {
        // Use discrete buffer binding slot for UBOs which translates each slot to [[buffer(n)]]
        *uboArgumentBufferUsed = false;
        // Discrete buffer binding slot starts at kUBOArgumentBufferBindingIndex
        currentSlot += kUBOArgumentBufferBindingIndex;
    }

    for (UniformBufferVar &uboVar : uniformBufferBindings)
    {
        spirv_cross::MSLResourceBinding &resBinding = uboVar.resBinding;
        resBinding.msl_buffer                       = currentSlot;

        uint32_t originalBinding = uboOriginalBindings.at(uboVar.name);

        for (uint32_t i = 0; i < uboVar.arraySize; ++i, ++currentSlot)
        {
            // Use consecutive slot for member in array
            uboBindingsRemapOut->at(originalBinding + i) = currentSlot;
        }

        compilerMsl.add_msl_resource_binding(resBinding);
    }
}

void GetAssignedSamplerBindings(const spirv_cross::CompilerMSL &compilerMsl,
                                const OriginalSamplerBindingMap &originalBindings,
                                std::array<SamplerBinding, mtl::kMaxGLSamplerBindings> *bindings)
{
    for (const spirv_cross::Resource &resource : compilerMsl.get_shader_resources().sampled_images)
    {
        uint32_t descriptorSet = 0;
        if (compilerMsl.has_decoration(resource.id, spv::DecorationDescriptorSet))
        {
            descriptorSet = compilerMsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
        }

        // We already assigned descriptor set 0 to textures. Just to double check.
        ASSERT(descriptorSet == kGlslangTextureDescSet);
        ASSERT(compilerMsl.has_decoration(resource.id, spv::DecorationBinding));

        uint32_t actualTextureSlot = compilerMsl.get_automatic_msl_resource_binding(resource.id);
        uint32_t actualSamplerSlot =
            compilerMsl.get_automatic_msl_resource_binding_secondary(resource.id);

        // Assign sequential index for subsequent array elements
        const std::vector<std::pair<uint32_t, uint32_t>> &resOrignalBindings =
            originalBindings.at(resource.name);
        uint32_t currentTextureSlot = actualTextureSlot;
        uint32_t currentSamplerSlot = actualSamplerSlot;
        for (const std::pair<uint32_t, uint32_t> &originalBindingRange : resOrignalBindings)
        {
            SamplerBinding &actualBinding = bindings->at(originalBindingRange.first);
            actualBinding.textureBinding  = currentTextureSlot;
            actualBinding.samplerBinding  = currentSamplerSlot;

            currentTextureSlot += originalBindingRange.second;
            currentSamplerSlot += originalBindingRange.second;
        }
    }
}

std::string PostProcessTranslatedMsl(bool hasDepthSampler, const std::string &translatedSource)
{
    std::string source;
    if (hasDepthSampler)
    {
        // Add ANGLEShadowCompareModes variable to main(), We need to add here because it is the
        // only way without modifying spirv-cross.
        std::regex mainDeclareRegex(
            R"(((vertex|fragment|kernel)\s+[_a-zA-Z0-9<>]+\s+main[^\(]*\())");
        std::string mainDeclareReplaceStr = std::string("$1constant uniform<uint> *") +
                                            kShadowSamplerCompareModesVarName + "[[buffer(" +
                                            Str(kShadowSamplerCompareModesBindingIndex) + ")]], ";
        source = std::regex_replace(translatedSource, mainDeclareRegex, mainDeclareReplaceStr);
    }
    else
    {
        source = translatedSource;
    }

    // Add function_constant attribute to gl_SampleMask.
    // Even though this varying is only used when ANGLECoverageMaskEnabled is true,
    // the spirv-cross doesn't assign function_constant attribute to it. Thus it won't be dead-code
    // removed when ANGLECoverageMaskEnabled=false.
    std::string sampleMaskReplaceStr = std::string("[[sample_mask, function_constant(") +
                                       sh::TranslatorMetal::GetCoverageMaskEnabledConstName() +
                                       ")]]";

    // This replaces "gl_SampleMask [[sample_mask]]"
    //          with "gl_SampleMask [[sample_mask, function_constant(ANGLECoverageMaskEnabled)]]"
    std::regex sampleMaskDeclareRegex(R"(\[\s*\[\s*sample_mask\s*\]\s*\])");
    return std::regex_replace(source, sampleMaskDeclareRegex, sampleMaskReplaceStr);
}

// Customized spirv-cross compiler
class SpirvToMslCompiler : public spirv_cross::CompilerMSL
{
  public:
    SpirvToMslCompiler(Context *context, std::vector<uint32_t> &&spriv)
        : spirv_cross::CompilerMSL(spriv), mContext(context)
    {}

    void compileEx(gl::ShaderType shaderType,
                   const std::unordered_map<std::string, uint32_t> &uboOriginalBindings,
                   const OriginalSamplerBindingMap &originalSamplerBindings,
                   TranslatedShaderInfo *mslShaderInfoOut)
    {
        spirv_cross::CompilerMSL::Options compOpt;

#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
        compOpt.platform = spirv_cross::CompilerMSL::Options::macOS;
#else
        compOpt.platform = spirv_cross::CompilerMSL::Options::iOS;
#endif

        if (ANGLE_APPLE_AVAILABLE_XCI(10.14, 13.0, 12))
        {
            // Use Metal 2.1
            compOpt.set_msl_version(2, 1);
        }
        else
        {
            // Always use at least Metal 2.0.
            compOpt.set_msl_version(2);
        }

        compOpt.pad_fragment_output_components = true;

        // Tell spirv-cross to map default & driver uniform & storage blocks as we want
        spirv_cross::ShaderResources mslRes = spirv_cross::CompilerMSL::get_shader_resources();

        spirv_cross::SmallVector<spirv_cross::Resource> buffers = std::move(mslRes.uniform_buffers);
        buffers.insert(buffers.end(), mslRes.storage_buffers.begin(), mslRes.storage_buffers.end());

        BindBuffers(this, buffers, shaderType, uboOriginalBindings,
                    &mslShaderInfoOut->actualUBOBindings, &mslShaderInfoOut->actualXFBBindings,
                    &mslShaderInfoOut->hasUBOArgumentBuffer);

        if (mslShaderInfoOut->hasUBOArgumentBuffer)
        {
            // Enable argument buffer.
            compOpt.argument_buffers = true;

            // Force UBO argument buffer binding to start at kUBOArgumentBufferBindingIndex.
            spirv_cross::MSLResourceBinding argBufferBinding = {};
            argBufferBinding.stage    = ShaderTypeToSpvExecutionModel(shaderType);
            argBufferBinding.desc_set = kGlslangShaderResourceDescSet;
            argBufferBinding.binding =
                spirv_cross::kArgumentBufferBinding;  // spirv-cross built-in binding.
            argBufferBinding.msl_buffer = kUBOArgumentBufferBindingIndex;  // Actual binding.
            spirv_cross::CompilerMSL::add_msl_resource_binding(argBufferBinding);

            // Force discrete slot bindings for textures, default uniforms & driver uniforms
            // instead of using argument buffer.
            spirv_cross::CompilerMSL::add_discrete_descriptor_set(kGlslangTextureDescSet);
            spirv_cross::CompilerMSL::add_discrete_descriptor_set(
                kGlslangDefaultUniformAndXfbDescSet);
            spirv_cross::CompilerMSL::add_discrete_descriptor_set(kGlslangDriverUniformsDescSet);
        }
        else
        {
            // Disable argument buffer generation for uniform buffers
            compOpt.argument_buffers = false;
        }

        spirv_cross::CompilerMSL::set_msl_options(compOpt);

        addBuiltInResources();
        analyzeShaderVariables();

        // Actual compilation
        mslShaderInfoOut->metalShaderSource =
            PostProcessTranslatedMsl(mHasDepthSampler, spirv_cross::CompilerMSL::compile());

        // Retrieve automatic texture slot assignments
        GetAssignedSamplerBindings(*this, originalSamplerBindings,
                                   &mslShaderInfoOut->actualSamplerBindings);
    }

  private:
    // Override CompilerMSL
    void emit_header() override
    {
        spirv_cross::CompilerMSL::emit_header();
        if (!mHasDepthSampler)
        {
            return;
        }
        // Work around code for these issues:
        // - spriv_cross always translates shadow texture's sampling to sample_compare() and doesn't
        // take into account GL_TEXTURE_COMPARE_MODE=GL_NONE.
        // - on macOS, explicit level of detail parameter is not supported in sample_compare().
        // - on devices prior to iOS GPU family 3, changing sampler's compare mode outside shader is
        // not supported.
        if (!mContext->getDisplay()->getFeatures().allowRuntimeSamplerCompareMode.enabled)
        {
            statement("#define ANGLE_MTL_NO_SAMPLER_RUNTIME_COMPARE_MODE");
        }

        statement("enum class ANGLECompareMode : uint");
        statement("{");
        statement("    None = 0,");
        statement("    Less,");
        statement("    LessEqual,");
        statement("    Greater,");
        statement("    GreaterEqual,");
        statement("    Never,");
        statement("    Always,");
        statement("    Equal,");
        statement("    NotEqual,");
        statement("};");
        statement("");

        statement("template <typename T, typename UniformOrUInt>");
        statement("inline T ANGLEcompare(T depth, T dref, UniformOrUInt compareMode)");
        statement("{");
        statement("   ANGLECompareMode mode = static_cast<ANGLECompareMode>(compareMode);");
        statement("   switch (mode)");
        statement("   {");
        statement("        case ANGLECompareMode::Less:");
        statement("            return dref < depth;");
        statement("        case ANGLECompareMode::LessEqual:");
        statement("            return dref <= depth;");
        statement("        case ANGLECompareMode::Greater:");
        statement("            return dref > depth;");
        statement("        case ANGLECompareMode::GreaterEqual:");
        statement("            return dref >= depth;");
        statement("        case ANGLECompareMode::Never:");
        statement("            return 0;");
        statement("        case ANGLECompareMode::Always:");
        statement("            return 1;");
        statement("        case ANGLECompareMode::Equal:");
        statement("            return dref == depth;");
        statement("        case ANGLECompareMode::NotEqual:");
        statement("            return dref != depth;");
        statement("        default:");
        statement("            return 1;");
        statement("   }");
        statement("}");
        statement("");

        statement("// Wrapper functions for shadow texture functions");
        // 2D PCF sampling
        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtexturePCF(depth2d<T> texture, sampler s, float2 coord, float "
                  "compare_value, Opt options, int2 offset, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("#if defined(__METAL_MACOS__)");
        statement("    float2 dims = float2(texture.get_width(), texture.get_height());");
        statement("    float2 imgCoord = coord * dims;");
        statement("    float2 texelSize = 1.0 / dims;");
        statement("    float2 weight = fract(imgCoord);");
        statement("    float tl = ANGLEcompare(texture.sample(s, coord, options, offset), "
                  "compare_value, shadowCompareMode);");
        statement("    float tr = ANGLEcompare(texture.sample(s, coord + float2(texelSize.x, 0.0), "
                  "options, offset), compare_value, shadowCompareMode);");
        statement("    float bl = ANGLEcompare(texture.sample(s, coord + float2(0.0, texelSize.y), "
                  "options, offset), compare_value, shadowCompareMode);");
        statement("    float br = ANGLEcompare(texture.sample(s, coord + texelSize, options, "
                  "offset), compare_value, shadowCompareMode);");
        statement("    float top = mix(tl, tr, weight.x);");
        statement("    float bottom = mix(bl, br, weight.x);");
        statement("    return mix(top, bottom, weight.y);");
        statement("#else  // if defined(__METAL_MACOS__)");
        statement("    return ANGLEcompare(texture.sample(s, coord, options, offset), "
                  "compare_value, shadowCompareMode);");
        statement("#endif  // if defined(__METAL_MACOS__)");
        statement("}");
        statement("");

        // Cube PCF sampling
        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtexturePCF(depthcube<T> texture, sampler s, float3 coord, float "
                  "compare_value, Opt options, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    // NOTE(hqle): to implement");
        statement("    return ANGLEcompare(texture.sample(s, coord, options), compare_value, "
                  "shadowCompareMode);");
        statement("}");
        statement("");

        // 2D array PCF sampling
        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement(
            "inline T ANGLEtexturePCF(depth2d_array<T> texture, sampler s, float2 coord, uint "
            "array, float compare_value, Opt options, int2 offset, UniformOrUInt "
            "shadowCompareMode)");
        statement("{");
        statement("#if defined(__METAL_MACOS__)");
        statement("    float2 dims = float2(texture.get_width(), texture.get_height());");
        statement("    float2 imgCoord = coord * dims;");
        statement("    float2 texelSize = 1.0 / dims;");
        statement("    float2 weight = fract(imgCoord);");
        statement("    float tl = ANGLEcompare(texture.sample(s, coord, array, options, offset), "
                  "compare_value, shadowCompareMode);");
        statement("    float tr = ANGLEcompare(texture.sample(s, coord + float2(texelSize.x, 0.0), "
                  "array, options, offset), compare_value, shadowCompareMode);");
        statement("    float bl = ANGLEcompare(texture.sample(s, coord + float2(0.0, texelSize.y), "
                  "array, options, offset), compare_value, shadowCompareMode);");
        statement("    float br = ANGLEcompare(texture.sample(s, coord + texelSize, array, "
                  "options, offset), compare_value, shadowCompareMode);");
        statement("    float top = mix(tl, tr, weight.x);");
        statement("    float bottom = mix(bl, br, weight.x);");
        statement("    return mix(top, bottom, weight.y);");
        statement("#else  // if defined(__METAL_MACOS__)");
        statement("    return ANGLEcompare(texture.sample(s, coord, array, options, offset), "
                  "compare_value, shadowCompareMode);");
        statement("#endif  // if defined(__METAL_MACOS__)");
        statement("}");
        statement("");

        // 2D texture's sample_compare() wrapper
        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtextureCompare(depth2d<T> texture, sampler s, float2 coord, float "
                  "compare_value, Opt options, int2 offset, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement(
            "#if defined(__METAL_MACOS__) || defined(ANGLE_MTL_NO_SAMPLER_RUNTIME_COMPARE_MODE)");
        statement("    return ANGLEtexturePCF(texture, s, coord, compare_value, options, offset, "
                  "shadowCompareMode);");
        statement("#else");
        statement("    return texture.sample_compare(s, coord, compare_value, options, offset);");
        statement("#endif");
        statement("}");
        statement("");

        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtextureCompare(depth2d<T> texture, sampler s, float2 coord, float "
                  "compare_value, Opt options, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    return ANGLEtextureCompare(texture, s, coord, compare_value, options, "
                  "int2(0), shadowCompareMode);");
        statement("}");
        statement("");

        statement("template <typename T, typename UniformOrUInt>");
        statement("inline T ANGLEtextureCompare(depth2d<T> texture, sampler s, float2 coord, float "
                  "compare_value, int2 offset, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("#if defined(ANGLE_MTL_NO_SAMPLER_RUNTIME_COMPARE_MODE)");
        statement("    return ANGLEtexturePCF(texture, s, coord, compare_value, level(0), offset, "
                  "shadowCompareMode);");
        statement("#else");
        statement("    return texture.sample_compare(s, coord, compare_value, offset);");
        statement("#endif");
        statement("}");
        statement("");

        // Cube texture's sample_compare() wrapper
        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement(
            "inline T ANGLEtextureCompare(depthcube<T> texture, sampler s, float3 coord, float "
            "compare_value, Opt options, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement(
            "#if defined(__METAL_MACOS__) || defined(ANGLE_MTL_NO_SAMPLER_RUNTIME_COMPARE_MODE)");
        statement("    return ANGLEtexturePCF(texture, s, coord, compare_value, options, "
                  "shadowCompareMode);");
        statement("#else");
        statement("    return texture.sample_compare(s, coord, compare_value, options);");
        statement("#endif");
        statement("}");
        statement("");

        statement("template <typename T, typename UniformOrUInt>");
        statement(
            "inline T ANGLEtextureCompare(depthcube<T> texture, sampler s, float3 coord, float "
            "compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("#if defined(ANGLE_MTL_NO_SAMPLER_RUNTIME_COMPARE_MODE)");
        statement("    return ANGLEtexturePCF(texture, s, coord, compare_value, level(0), "
                  "shadowCompareMode);");
        statement("#else");
        statement("    return texture.sample_compare(s, coord, compare_value);");
        statement("#endif");
        statement("}");
        statement("");

        // 2D array texture's sample_compare() wrapper
        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtextureCompare(depth2d_array<T> texture, sampler s, float2 coord, "
                  "uint array, float compare_value, Opt options, int2 offset, UniformOrUInt "
                  "shadowCompareMode)");
        statement("{");
        statement(
            "#if defined(__METAL_MACOS__) || defined(ANGLE_MTL_NO_SAMPLER_RUNTIME_COMPARE_MODE)");
        statement("    return ANGLEtexturePCF(texture, s, coord, array, compare_value, options, "
                  "offset, shadowCompareMode);");
        statement("#else");
        statement(
            "    return texture.sample_compare(s, coord, array, compare_value, options, offset);");
        statement("#endif");
        statement("}");
        statement("");

        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtextureCompare(depth2d_array<T> texture, sampler s, float2 coord, "
                  "uint array, float compare_value, Opt options, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    return ANGLEtextureCompare(texture, s, coord, array, compare_value, "
                  "options, int2(0), shadowCompareMode);");
        statement("}");
        statement("");

        statement("template <typename T, typename UniformOrUInt>");
        statement("inline T ANGLEtextureCompare(depth2d_array<T> texture, sampler s, float2 coord, "
                  "uint array, float compare_value, int2 offset, UniformOrUInt "
                  "shadowCompareMode)");
        statement("{");
        statement("#if defined(ANGLE_MTL_NO_SAMPLER_RUNTIME_COMPARE_MODE)");
        statement("    return ANGLEtexturePCF(texture, s, coord, array, compare_value, level(0), "
                  "offset, shadowCompareMode);");
        statement("#else");
        statement("    return texture.sample_compare(s, coord, array, compare_value, offset);");
        statement("#endif");
        statement("}");
        statement("");

        // 2D texture's generic sampling function
        statement("// Wrapper functions for shadow texture functions");
        statement("template <typename T, typename UniformOrUInt>");
        statement("inline T ANGLEtexture(depth2d<T> texture, sampler s, float2 coord, int2 offset, "
                  "float compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    if (shadowCompareMode)");
        statement("    {");
        statement("        return ANGLEtextureCompare(texture, s, coord, compare_value, offset, "
                  "shadowCompareMode);");
        statement("    }");
        statement("    else");
        statement("    {");
        statement("        return texture.sample(s, coord, offset);");
        statement("    }");
        statement("}");
        statement("");

        statement("template <typename T, typename UniformOrUInt>");
        statement("inline T ANGLEtexture(depth2d<T> texture, sampler s, float2 coord, float "
                  "compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    return ANGLEtexture(texture, s, coord, int2(0), compare_value, "
                  "shadowCompareMode);");
        statement("}");
        statement("");

        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtexture(depth2d<T> texture, sampler s, float2 coord, Opt options, "
                  "int2 offset, float compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    if (shadowCompareMode)");
        statement("    {");
        statement("        return ANGLEtextureCompare(texture, s, coord, compare_value, options, "
                  "offset, shadowCompareMode);");
        statement("    }");
        statement("    else");
        statement("    {");
        statement("        return texture.sample(s, coord, options, offset);");
        statement("    }");
        statement("}");
        statement("");

        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtexture(depth2d<T> texture, sampler s, float2 coord, Opt options, "
                  "float compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    return ANGLEtexture(texture, s, coord, options, int2(0), compare_value, "
                  "shadowCompareMode);");
        statement("}");
        statement("");

        // Cube texture's generic sampling function
        statement("template <typename T, typename UniformOrUInt>");
        statement("inline T ANGLEtexture(depthcube<T> texture, sampler s, float3 coord, float "
                  "compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    if (shadowCompareMode)");
        statement("    {");
        statement("        return ANGLEtextureCompare(texture, s, coord, compare_value, "
                  "shadowCompareMode);");
        statement("    }");
        statement("    else");
        statement("    {");
        statement("        return texture.sample(s, coord);");
        statement("    }");
        statement("}");
        statement("");

        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtexture(depthcube<T> texture, sampler s, float2 coord, Opt "
                  "options, float compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    if (shadowCompareMode)");
        statement("    {");
        statement("        return ANGLEtextureCompare(texture, s, coord, compare_value, options, "
                  "shadowCompareMode);");
        statement("    }");
        statement("    else");
        statement("    {");
        statement("        return texture.sample(s, coord, options);");
        statement("    }");
        statement("}");
        statement("");

        // 2D array texture's generic sampling function
        statement("template <typename T, typename UniformOrUInt>");
        statement("inline T ANGLEtexture(depth2d_array<T> texture, sampler s, float2 coord, uint "
                  "array, int2 offset, "
                  "float compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    if (shadowCompareMode)");
        statement("    {");
        statement("        return ANGLEtextureCompare(texture, s, coord, array, compare_value, "
                  "offset, shadowCompareMode);");
        statement("    }");
        statement("    else");
        statement("    {");
        statement("        return texture.sample(s, coord, array, offset);");
        statement("    }");
        statement("}");
        statement("");

        statement("template <typename T, typename UniformOrUInt>");
        statement("inline T ANGLEtexture(depth2d_array<T> texture, sampler s, float2 coord, uint "
                  "array, float compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    return ANGLEtexture(texture, s, coord, array, int2(0), compare_value, "
                  "shadowCompareMode);");
        statement("}");
        statement("");

        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtexture(depth2d_array<T> texture, sampler s, float2 coord, uint "
                  "array, Opt options, int2 offset, "
                  "float compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    if (shadowCompareMode)");
        statement("    {");
        statement("        return ANGLEtextureCompare(texture, s, coord, array, compare_value, "
                  "options, offset, shadowCompareMode);");
        statement("    }");
        statement("    else");
        statement("    {");
        statement("        return texture.sample(s, coord, array, options, offset);");
        statement("    }");
        statement("}");
        statement("");

        statement("template <typename T, typename Opt, typename UniformOrUInt>");
        statement("inline T ANGLEtexture(depth2d_array<T> texture, sampler s, float2 coord, uint "
                  "array, Opt options, float compare_value, UniformOrUInt shadowCompareMode)");
        statement("{");
        statement("    return ANGLEtexture(texture, s, coord, array, options, int2(0), "
                  "compare_value, shadowCompareMode);");
        statement("}");
        statement("");
    }

    std::string to_function_name(spirv_cross::VariableID img,
                                 const spirv_cross::SPIRType &imgType,
                                 bool isFetch,
                                 bool isGather,
                                 bool isProj,
                                 bool hasArrayOffsets,
                                 bool hasOffset,
                                 bool hasGrad,
                                 bool hasDref,
                                 uint32_t lod,
                                 uint32_t minLod) override
    {
        if (!hasDref)
        {
            return spirv_cross::CompilerMSL::to_function_name(img, imgType, isFetch, isGather,
                                                              isProj, hasArrayOffsets, hasOffset,
                                                              hasGrad, hasDref, lod, minLod);
        }

        // Use custom ANGLEtexture function instead of using built-in sample_compare()
        return "ANGLEtexture";
    }

    std::string to_function_args(spirv_cross::VariableID img,
                                 const spirv_cross::SPIRType &imgType,
                                 bool isFetch,
                                 bool isGather,
                                 bool isProj,
                                 uint32_t coord,
                                 uint32_t coordComponents,
                                 uint32_t dref,
                                 uint32_t gradX,
                                 uint32_t gradY,
                                 uint32_t lod,
                                 uint32_t coffset,
                                 uint32_t offset,
                                 uint32_t bias,
                                 uint32_t comp,
                                 uint32_t sample,
                                 uint32_t minlod,
                                 bool *pForward) override
    {
        bool forward;
        std::string argsWithoutDref = spirv_cross::CompilerMSL::to_function_args(
            img, imgType, isFetch, isGather, isProj, coord, coordComponents, 0, gradX, gradY, lod,
            coffset, offset, bias, comp, sample, minlod, &forward);

        if (!dref)
        {
            if (pForward)
            {
                *pForward = forward;
            }
            return argsWithoutDref;
        }
        // Convert to arguments to ANGLEtexture.
        std::string args = to_expression(img);
        args += ", ";
        args += argsWithoutDref;
        args += ", ";

        forward                               = forward && should_forward(dref);
        const spirv_cross::SPIRType &drefType = expression_type(dref);
        std::string drefExpr;
        uint32_t altCoordComponent = 0;
        switch (imgType.image.dim)
        {
            case spv::Dim2D:
                altCoordComponent = 2;
                break;
            case spv::Dim3D:
            case spv::DimCube:
                altCoordComponent = 3;
                break;
            default:
                UNREACHABLE();
                break;
        }
        if (isProj)
            drefExpr = spirv_cross::join(to_enclosed_expression(dref), " / ",
                                         to_extract_component_expression(coord, altCoordComponent));
        else
            drefExpr = to_expression(dref);

        if (drefType.basetype == spirv_cross::SPIRType::Half)
            drefExpr = convert_to_f32(drefExpr, 1);

        args += drefExpr;
        args += ", ";
        args += toShadowCompareModeExpression(img);

        if (pForward)
        {
            *pForward = forward;
        }

        return args;
    }

    // Override function prototype emitter to insert shadow compare mode flag to come
    // together with the shadow sampler. NOTE(hqle): This is just 90% copy of spirv_msl's code.
    // The better way is modifying and creating a PR on spirv-cross repo directly. But this should
    // be a work around solution for now.
    void emit_function_prototype(spirv_cross::SPIRFunction &func,
                                 const spirv_cross::Bitset &) override
    {
        // Turn off clang-format to easier compare with original code
        // clang-format off
        using namespace spirv_cross;
        using namespace spv;
        using namespace std;

        if (func.self != ir.default_entry_point)
            add_function_overload(func);

        local_variable_names = resource_names;
        string decl;

        processing_entry_point = func.self == ir.default_entry_point;

        // Metal helper functions must be static force-inline otherwise they will cause problems when linked together in a single Metallib.
        if (!processing_entry_point)
            statement("static inline __attribute__((always_inline))");

        auto &type = get<SPIRType>(func.return_type);

        if (!type.array.empty() && msl_options.force_native_arrays)
        {
            // We cannot return native arrays in MSL, so "return" through an out variable.
            decl += "void";
        }
        else
        {
            decl += func_type_decl(type);
        }

        decl += " ";
        decl += to_name(func.self);
        decl += "(";

        if (!type.array.empty() && msl_options.force_native_arrays)
        {
            // Fake arrays returns by writing to an out array instead.
            decl += "thread ";
            decl += type_to_glsl(type);
            decl += " (&SPIRV_Cross_return_value)";
            decl += type_to_array_glsl(type);
            if (!func.arguments.empty())
                decl += ", ";
        }

        if (processing_entry_point)
        {
            if (msl_options.argument_buffers)
                decl += entry_point_args_argument_buffer(!func.arguments.empty());
            else
                decl += entry_point_args_classic(!func.arguments.empty());

            // If entry point function has variables that require early declaration,
            // ensure they each have an empty initializer, creating one if needed.
            // This is done at this late stage because the initialization expression
            // is cleared after each compilation pass.
            for (auto var_id : vars_needing_early_declaration)
            {
                auto &ed_var = get<SPIRVariable>(var_id);
                ID &initializer = ed_var.initializer;
                if (!initializer)
                    initializer = ir.increase_bound_by(1);

                // Do not override proper initializers.
                if (ir.ids[initializer].get_type() == TypeNone || ir.ids[initializer].get_type() == TypeExpression)
                    set<SPIRExpression>(ed_var.initializer, "{}", ed_var.basetype, true);
            }
        }

        for (auto &arg : func.arguments)
        {
            uint32_t name_id = arg.id;

            auto *var = maybe_get<SPIRVariable>(arg.id);
            if (var)
            {
                // If we need to modify the name of the variable, make sure we modify the original variable.
                // Our alias is just a shadow variable.
                if (arg.alias_global_variable && var->basevariable)
                    name_id = var->basevariable;

                var->parameter = &arg; // Hold a pointer to the parameter so we can invalidate the readonly field if needed.
            }

            add_local_variable_name(name_id);

            decl += argument_decl(arg);

            bool is_dynamic_img_sampler = has_extended_decoration(arg.id, SPIRVCrossDecorationDynamicImageSampler);

            auto &arg_type = get<SPIRType>(arg.type);
            if (arg_type.basetype == SPIRType::SampledImage && !is_dynamic_img_sampler)
            {
                // Manufacture automatic plane args for multiplanar texture
                uint32_t planes = 1;
                if (auto *constexpr_sampler = find_constexpr_sampler(name_id))
                    if (constexpr_sampler->ycbcr_conversion_enable)
                        planes = constexpr_sampler->planes;
                for (uint32_t i = 1; i < planes; i++)
                    decl += join(", ", argument_decl(arg), plane_name_suffix, i);

                // Manufacture automatic sampler arg for SampledImage texture
                if (arg_type.image.dim != DimBuffer)
                    decl += join(", thread const ", sampler_type(arg_type), " ", to_sampler_expression(arg.id));
            }

            // Manufacture automatic swizzle arg.
            if (msl_options.swizzle_texture_samples && has_sampled_images && is_sampled_image_type(arg_type) &&
                !is_dynamic_img_sampler)
            {
                bool arg_is_array = !arg_type.array.empty();
                decl += join(", constant uint", arg_is_array ? "* " : "& ", to_swizzle_expression(arg.id));
            }

            if (buffers_requiring_array_length.count(name_id))
            {
                bool arg_is_array = !arg_type.array.empty();
                decl += join(", constant uint", arg_is_array ? "* " : "& ", to_buffer_size_expression(name_id));
            }

            if (is_sampled_image_type(arg_type) && arg_type.image.depth)
            {
                bool arg_is_array = !arg_type.array.empty();
                // Insert shadow compare mode flag to the argument sequence:
                decl += join(", constant uniform<uint>", arg_is_array ? "* " : "& ",
                             toShadowCompareModeExpression(arg.id));
            }

            if (&arg != &func.arguments.back())
                decl += ", ";
        }

        decl += ")";
        statement(decl);

        // clang-format on
    }

    // Override function call arguments passing generator to insert shadow compare mode flag to come
    // together with the shadow sampler
    std::string to_func_call_arg(const spirv_cross::SPIRFunction::Parameter &arg,
                                 uint32_t id) override
    {
        std::string arg_str = spirv_cross::CompilerMSL::to_func_call_arg(arg, id);

        const spirv_cross::SPIRType &type = expression_type(id);

        if (is_sampled_image_type(type) && type.image.depth)
        {
            // Insert shadow compare mode flag to the argument sequence:
            // Need to check the base variable in case we need to apply a qualified alias.
            uint32_t var_id = 0;
            auto *var       = maybe_get<spirv_cross::SPIRVariable>(id);
            if (var)
                var_id = var->basevariable;

            arg_str += ", " + toShadowCompareModeExpression(var_id ? var_id : id);
        }
        return arg_str;
    }

    // Additional functions
    void addBuiltInResources()
    {
        uint32_t varId = build_constant_uint_array_pointer();
        set_name(varId, kShadowSamplerCompareModesVarName);
        // This should never match anything.
        set_decoration(varId, spv::DecorationDescriptorSet, kShadowSamplerCompareModesBindingIndex);
        set_decoration(varId, spv::DecorationBinding, 0);
        set_extended_decoration(varId, spirv_cross::SPIRVCrossDecorationResourceIndexPrimary, 0);
        mANGLEShadowCompareModesVarId = varId;
    }

    void analyzeShaderVariables()
    {
        ir.for_each_typed_id<spirv_cross::SPIRVariable>([this](uint32_t,
                                                               spirv_cross::SPIRVariable &var) {
            const spirv_cross::SPIRType &type = get_variable_data_type(var);
            uint32_t varId                    = var.self;

            if (var.storage == spv::StorageClassUniformConstant && !is_hidden_variable(var))
            {
                if (is_sampled_image_type(type) && type.image.depth)
                {
                    mHasDepthSampler = true;

                    auto &entry_func = this->get<spirv_cross::SPIRFunction>(ir.default_entry_point);
                    entry_func.fixup_hooks_in.push_back([this, &type, &var, varId]() {
                        bool isArrayType = !type.array.empty();

                        statement("constant uniform<uint>", isArrayType ? "* " : "& ",
                                  toShadowCompareModeExpression(varId),
                                  isArrayType ? " = &" : " = ",
                                  to_name(mANGLEShadowCompareModesVarId), "[",
                                  spirv_cross::convert_to_string(
                                      get_metal_resource_index(var, spirv_cross::SPIRType::Image)),
                                  "];");
                    });
                }
            }
        });
    }

    std::string toShadowCompareModeExpression(uint32_t id)
    {
        constexpr char kCompareModeSuffix[] = "_CompMode";
        auto *combined                      = maybe_get<spirv_cross::SPIRCombinedImageSampler>(id);

        std::string expr = to_expression(combined ? combined->image : spirv_cross::VariableID(id));
        auto index       = expr.find_first_of('[');

        if (index == std::string::npos)
            return expr + kCompareModeSuffix;
        else
        {
            auto imageExpr = expr.substr(0, index);
            auto arrayExpr = expr.substr(index);
            return imageExpr + kCompareModeSuffix + arrayExpr;
        }
    }

    Context *mContext;
    uint32_t mANGLEShadowCompareModesVarId = 0;
    bool mHasDepthSampler                  = false;
};

angle::Result ConvertSpirvToMsl(
    Context *context,
    gl::ShaderType shaderType,
    const std::unordered_map<std::string, uint32_t> &uboOriginalBindings,
    const OriginalSamplerBindingMap &originalSamplerBindings,
    std::vector<uint32_t> *sprivCode,
    TranslatedShaderInfo *translatedShaderInfoOut)
{
    if (!sprivCode || sprivCode->empty())
    {
        return angle::Result::Continue;
    }

    SpirvToMslCompiler compilerMsl(context, std::move(*sprivCode));

    // NOTE(hqle): spirv-cross uses exceptions to report error, what should we do here
    // in case of error?
    compilerMsl.compileEx(shaderType, uboOriginalBindings, originalSamplerBindings,
                          translatedShaderInfoOut);
    if (translatedShaderInfoOut->metalShaderSource.size() == 0)
    {
        ANGLE_MTL_CHECK(context, false, GL_INVALID_OPERATION);
    }

    return angle::Result::Continue;
}

}  // namespace

void TranslatedShaderInfo::reset()
{
    metalShaderSource.clear();
    metalLibrary         = nil;
    hasUBOArgumentBuffer = false;
    for (mtl::SamplerBinding &binding : actualSamplerBindings)
    {
        binding.textureBinding = mtl::kMaxShaderSamplers;
    }

    for (uint32_t &binding : actualUBOBindings)
    {
        binding = mtl::kMaxShaderBuffers;
    }

    for (uint32_t &binding : actualXFBBindings)
    {
        binding = mtl::kMaxShaderBuffers;
    }
}

void TranslatedShaderInfo::save(gl::BinaryOutputStream *stream)
{
    stream->writeString(metalShaderSource);
    stream->writeInt<int>(hasUBOArgumentBuffer);
    for (const mtl::SamplerBinding &binding : actualSamplerBindings)
    {
        stream->writeInt<uint32_t>(binding.textureBinding);
        stream->writeInt<uint32_t>(binding.samplerBinding);
    }

    for (uint32_t uboBinding : actualUBOBindings)
    {
        stream->writeInt<uint32_t>(uboBinding);
    }

    for (uint32_t xfbBinding : actualXFBBindings)
    {
        stream->writeInt<uint32_t>(xfbBinding);
    }
}

void TranslatedShaderInfo::load(gl::BinaryInputStream *stream)
{
    stream->readString(&metalShaderSource);

    hasUBOArgumentBuffer = stream->readInt<int>() != 0;

    for (mtl::SamplerBinding &binding : actualSamplerBindings)
    {
        binding.textureBinding = stream->readInt<uint32_t>();
        binding.samplerBinding = stream->readInt<uint32_t>();
    }

    for (uint32_t &uboBinding : actualUBOBindings)
    {
        uboBinding = stream->readInt<uint32_t>();
    }

    for (uint32_t &xfbBinding : actualXFBBindings)
    {
        xfbBinding = stream->readInt<uint32_t>();
    }
}

void GlslangGetShaderSource(const gl::ProgramState &programState,
                            const gl::ProgramLinkedResources &resources,
                            gl::ShaderMap<std::string> *shaderSourcesOut)
{
    rx::GlslangGetShaderSource(CreateSourceOptions(), false, programState, resources,
                               shaderSourcesOut);
}

angle::Result GlslangGetShaderSpirvCode(ErrorHandler *context,
                                        const gl::Caps &glCaps,
                                        const gl::ProgramState &programState,
                                        bool enableLineRasterEmulation,
                                        const gl::ShaderMap<std::string> &shaderSources,
                                        gl::ShaderMap<std::vector<uint32_t>> *shaderCodeOut,
                                        std::vector<uint32_t> *xfbOnlyShaderCodeOut /** nullable */)
{
    // Normal version without XFB emulation
    ANGLE_TRY(rx::GlslangGetShaderSpirvCode(
        [context](GlslangError error) { return HandleError(context, error); }, glCaps,
        enableLineRasterEmulation, /* enableXfbEmulation */ false, shaderSources, shaderCodeOut));

    // Metal doesn't allow vertex shader to write to both buffers and stage output. So need a
    // special version with only XFB emulation.
    if (xfbOnlyShaderCodeOut && !programState.getLinkedTransformFeedbackVaryings().empty())
    {
        gl::ShaderMap<std::string> vsOnlySrcMap;
        gl::ShaderMap<std::vector<uint32_t>> vsOnlyCodeMap;
        vsOnlySrcMap[gl::ShaderType::Vertex] = shaderSources[gl::ShaderType::Vertex];

        ANGLE_TRY(rx::GlslangGetShaderSpirvCode(
            [context](GlslangError error) { return HandleError(context, error); }, glCaps,
            enableLineRasterEmulation, /* enableXfbEmulation */ true, vsOnlySrcMap,
            &vsOnlyCodeMap));
        *xfbOnlyShaderCodeOut = std::move(vsOnlyCodeMap[gl::ShaderType::Vertex]);
    }

    return angle::Result::Continue;
}

angle::Result SpirvCodeToMsl(Context *context,
                             const gl::ProgramState &programState,
                             gl::ShaderMap<std::vector<uint32_t>> *spirvShaderCode,
                             std::vector<uint32_t> *xfbOnlySpirvCode /** nullable */,
                             gl::ShaderMap<TranslatedShaderInfo> *mslShaderInfoOut,
                             TranslatedShaderInfo *mslXfbOnlyShaderInfoOut /** nullable */)
{
    // Retrieve original uniform buffer bindings generated by front end. We will need to do a remap.
    std::unordered_map<std::string, uint32_t> uboOriginalBindings;
    const std::vector<gl::InterfaceBlock> &blocks = programState.getUniformBlocks();
    for (uint32_t bufferIdx = 0; bufferIdx < blocks.size(); ++bufferIdx)
    {
        const gl::InterfaceBlock &block = blocks[bufferIdx];
        if (!uboOriginalBindings.count(block.mappedName))
        {
            uboOriginalBindings[block.mappedName] = bufferIdx;
        }
    }
    // Retrieve original sampler bindings produced by front end.
    OriginalSamplerBindingMap originalSamplerBindings;
    const std::vector<gl::SamplerBinding> &samplerBindings = programState.getSamplerBindings();
    const std::vector<gl::LinkedUniform> &uniforms         = programState.getUniforms();

    for (uint32_t textureIndex = 0; textureIndex < samplerBindings.size(); ++textureIndex)
    {
        const gl::SamplerBinding &samplerBinding = samplerBindings[textureIndex];
        uint32_t uniformIndex = programState.getUniformIndexFromSamplerIndex(textureIndex);
        const gl::LinkedUniform &samplerUniform = uniforms[uniformIndex];
        bool isSamplerInStruct = samplerUniform.name.find('.') != std::string::npos;
        std::string mappedSamplerName =
            isSamplerInStruct ? GlslangGetMappedSamplerName(samplerUniform.name)
                              : GlslangGetMappedSamplerName(samplerUniform.mappedName);
        originalSamplerBindings[mappedSamplerName].push_back(
            {textureIndex, static_cast<uint32_t>(samplerBinding.boundTextureUnits.size())});
    }

    // Do the actual translation
    for (gl::ShaderType shaderType : gl::AllGLES2ShaderTypes())
    {
        std::vector<uint32_t> &sprivCode = spirvShaderCode->at(shaderType);
        ANGLE_TRY(ConvertSpirvToMsl(context, shaderType, uboOriginalBindings,
                                    originalSamplerBindings, &sprivCode,
                                    &mslShaderInfoOut->at(shaderType)));
    }  // for (gl::ShaderType shaderType

    // Special version of XFB only
    if (xfbOnlySpirvCode && !programState.getLinkedTransformFeedbackVaryings().empty())
    {
        ANGLE_TRY(ConvertSpirvToMsl(context, gl::ShaderType::Vertex, uboOriginalBindings,
                                    originalSamplerBindings, xfbOnlySpirvCode,
                                    mslXfbOnlyShaderInfoOut));
    }

    return angle::Result::Continue;
}

uint MslGetShaderShadowCompareMode(GLenum mode, GLenum func)
{
    // See SpirvToMslCompiler::emit_header()
    if (mode == GL_NONE)
    {
        return 0;
    }
    else
    {
        switch (func)
        {
            case GL_LESS:
                return 1;
            case GL_LEQUAL:
                return 2;
            case GL_GREATER:
                return 3;
            case GL_GEQUAL:
                return 4;
            case GL_NEVER:
                return 5;
            case GL_ALWAYS:
                return 6;
            case GL_EQUAL:
                return 7;
            case GL_NOTEQUAL:
                return 8;
            default:
                UNREACHABLE();
                return 1;
        }
    }
}

}  // namespace mtl
}  // namespace rx
