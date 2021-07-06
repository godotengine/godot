//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef LIBANGLE_CAPS_H_
#define LIBANGLE_CAPS_H_

#include "angle_gl.h"
#include "libANGLE/Version.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/Format.h"

#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace gl
{

struct Extensions;

struct TextureCaps
{
    TextureCaps();
    TextureCaps(const TextureCaps &other);
    ~TextureCaps();

    // Supports for basic texturing: glTexImage, glTexSubImage, etc
    bool texturable = false;

    // Support for linear or anisotropic filtering
    bool filterable = false;

    // Support for being used as a framebuffer attachment, i.e. glFramebufferTexture2D
    bool textureAttachment = false;

    // Support for being used as a renderbuffer format, i.e. glFramebufferRenderbuffer
    bool renderbuffer = false;

    // Set of supported sample counts, only guaranteed to be valid in ES3.
    SupportedSampleSet sampleCounts;

    // Get the maximum number of samples supported
    GLuint getMaxSamples() const;

    // Get the number of supported samples that is at least as many as requested.  Returns 0 if
    // there are no sample counts available
    GLuint getNearestSamples(GLuint requestedSamples) const;
};

TextureCaps GenerateMinimumTextureCaps(GLenum internalFormat,
                                       const Version &clientVersion,
                                       const Extensions &extensions);

class TextureCapsMap final : angle::NonCopyable
{
  public:
    TextureCapsMap();
    ~TextureCapsMap();

    // These methods are deprecated. Please use angle::Format for new features.
    void insert(GLenum internalFormat, const TextureCaps &caps);
    const TextureCaps &get(GLenum internalFormat) const;

    void clear();

    // Prefer using angle::Format methods.
    const TextureCaps &get(angle::FormatID formatID) const;
    void set(angle::FormatID formatID, const TextureCaps &caps);

  private:
    TextureCaps &get(angle::FormatID formatID);

    // Indexed by angle::FormatID
    std::array<TextureCaps, angle::kNumANGLEFormats> mFormatData;
};

void InitMinimumTextureCapsMap(const Version &clientVersion,
                               const Extensions &extensions,
                               TextureCapsMap *capsMap);

// Returns true if all the formats required to support GL_ANGLE_compressed_texture_etc are
// present. Does not determine if they are natively supported without decompression.
bool DetermineCompressedTextureETCSupport(const TextureCapsMap &textureCaps);

struct Extensions
{
    Extensions();
    Extensions(const Extensions &other);

    // Generate a vector of supported extension strings
    std::vector<std::string> getStrings() const;

    // Set all texture related extension support based on the supported textures.
    // Determines support for:
    // GL_OES_packed_depth_stencil
    // GL_OES_rgb8_rgba8
    // GL_EXT_texture_format_BGRA8888
    // GL_EXT_color_buffer_half_float,
    // GL_OES_texture_half_float, GL_OES_texture_half_float_linear
    // GL_OES_texture_float, GL_OES_texture_float_linear
    // GL_EXT_texture_rg
    // GL_EXT_texture_compression_dxt1, GL_ANGLE_texture_compression_dxt3,
    // GL_ANGLE_texture_compression_dxt5
    // GL_KHR_texture_compression_astc_ldr, GL_OES_texture_compression_astc.
    //     NOTE: GL_KHR_texture_compression_astc_hdr must be enabled separately. Support for the
    //           HDR profile cannot be determined from the format enums alone.
    // GL_OES_compressed_ETC1_RGB8_texture
    // GL_EXT_sRGB
    // GL_ANGLE_depth_texture, GL_OES_depth32
    // GL_EXT_color_buffer_float
    // GL_EXT_texture_norm16
    // GL_EXT_texture_compression_bptc
    void setTextureExtensionSupport(const TextureCapsMap &textureCaps);

    // indicate if any depth texture extension is available
    bool depthTextureAny() const { return (depthTextureANGLE || depthTextureOES); }

    // ES2 Extension support

    // GL_OES_element_index_uint
    bool elementIndexUint = false;

    // GL_OES_packed_depth_stencil
    bool packedDepthStencil = false;

    // GL_OES_get_program_binary
    bool getProgramBinary = false;

    // GL_OES_rgb8_rgba8
    // Implies that TextureCaps for GL_RGB8 and GL_RGBA8 exist
    bool rgb8rgba8 = false;

    // GL_EXT_texture_format_BGRA8888
    // Implies that TextureCaps for GL_BGRA8 exist
    bool textureFormatBGRA8888 = false;

    // GL_EXT_read_format_bgra
    bool readFormatBGRA = false;

    // GL_NV_pixel_buffer_object
    bool pixelBufferObject = false;

    // GL_OES_mapbuffer and GL_EXT_map_buffer_range
    bool mapBuffer      = false;
    bool mapBufferRange = false;

    // GL_EXT_color_buffer_half_float
    // Together with GL_OES_texture_half_float in a GLES 2.0 context, implies that half-float
    // textures are renderable.
    bool colorBufferHalfFloat = false;

    // GL_OES_texture_half_float and GL_OES_texture_half_float_linear
    // Implies that TextureCaps for GL_RGB16F, GL_RGBA16F, GL_ALPHA32F_EXT, GL_LUMINANCE32F_EXT and
    // GL_LUMINANCE_ALPHA32F_EXT exist
    bool textureHalfFloat       = false;
    bool textureHalfFloatLinear = false;

    // GL_EXT_texture_type_2_10_10_10_REV
    bool textureFormat2101010REV = false;

    // GL_OES_texture_float and GL_OES_texture_float_linear
    // Implies that TextureCaps for GL_RGB32F, GL_RGBA32F, GL_ALPHA16F_EXT, GL_LUMINANCE16F_EXT and
    // GL_LUMINANCE_ALPHA16F_EXT exist
    bool textureFloat       = false;
    bool textureFloatLinear = false;

    // GL_EXT_texture_rg
    // Implies that TextureCaps for GL_R8, GL_RG8 (and floating point R/RG texture formats if
    // floating point extensions are also present) exist
    bool textureRG = false;

    // GL_EXT_texture_compression_dxt1, GL_ANGLE_texture_compression_dxt3 and
    // GL_ANGLE_texture_compression_dxt5 Implies that TextureCaps exist for
    // GL_COMPRESSED_RGB_S3TC_DXT1_EXT, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
    // GL_COMPRESSED_RGBA_S3TC_DXT3_ANGLE and GL_COMPRESSED_RGBA_S3TC_DXT5_ANGLE
    bool textureCompressionDXT1 = false;
    bool textureCompressionDXT3 = false;
    bool textureCompressionDXT5 = false;

    // GL_EXT_texture_compression_s3tc_srgb
    // Implies that TextureCaps exist for GL_COMPRESSED_SRGB_S3TC_DXT1_EXT,
    // GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, and
    // GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT
    bool textureCompressionS3TCsRGB = false;

    // GL_KHR_texture_compression_astc_ldr
    bool textureCompressionASTCLDRKHR = false;

    // GL_KHR_texture_compression_astc_hdr
    bool textureCompressionASTCHDRKHR = false;

    // GL_OES_texture_compression_astc
    bool textureCompressionASTCOES = false;

    // GL_EXT_texture_compression_bptc
    bool textureCompressionBPTC = false;

    // GL_OES_compressed_ETC1_RGB8_texture
    // Implies that TextureCaps for GL_ETC1_RGB8_OES exist
    bool compressedETC1RGB8Texture = false;

    // OES_compressed_ETC2_RGB8_texture
    bool compressedETC2RGB8Texture = false;

    // OES_compressed_ETC2_sRGB8_texture
    bool compressedETC2sRGB8Texture = false;

    // OES_compressed_ETC2_punchthroughA_RGBA8_texture
    bool compressedETC2PunchthroughARGB8Texture = false;

    // OES_compressed_ETC2_punchthroughA_sRGB8_alpha_texture
    bool compressedETC2PunchthroughAsRGB8AlphaTexture = false;

    // OES_compressed_ETC2_RGBA8_texture
    bool compressedETC2RGBA8Texture = false;

    // OES_compressed_ETC2_sRGB8_alpha8_texture
    bool compressedETC2sRGB8Alpha8Texture = false;

    // OES_compressed_EAC_R11_unsigned_texture
    bool compressedEACR11UnsignedTexture = false;

    // OES_compressed_EAC_R11_signed_texture
    bool compressedEACR11SignedTexture = false;

    // OES_compressed_EAC_RG11_unsigned_texture
    bool compressedEACRG11UnsignedTexture = false;

    // OES_compressed_EAC_RG11_signed_texture
    bool compressedEACRG11SignedTexture = false;

    // ANGLE_compressed_texture_etc
    // ONLY exposed if ETC texture formats are natively supported without decompression
    // Backends should enable this extension explicitly. It is not enabled with
    // setTextureExtensionSupport, use DetermineCompressedTextureETCSupport to check if all of the
    // individual formats required to support this extension are available.
    bool compressedTextureETC = false;

    // GL_IMG_texture_compression_pvrtc
    bool compressedTexturePVRTC = false;

    // GL_EXT_pvrtc_sRGB
    bool compressedTexturePVRTCsRGB = false;

    // GL_EXT_sRGB
    // Implies that TextureCaps for GL_SRGB8_ALPHA8 and GL_SRGB8 exist
    // TODO: Don't advertise this extension in ES3
    bool sRGB = false;

    // GL_ANGLE_depth_texture
    bool depthTextureANGLE = false;

    // OES_depth_texture
    bool depthTextureOES = false;

    // GL_OES_depth24
    // Allows DEPTH_COMPONENT24_OES as a valid Renderbuffer format.
    bool depth24OES = false;

    // GL_OES_depth32
    // Allows DEPTH_COMPONENT32_OES as a valid Renderbuffer format.
    bool depth32 = false;

    // GL_OES_texture_3D
    bool texture3DOES = false;

    // GL_EXT_texture_storage
    bool textureStorage = false;

    // GL_OES_texture_npot
    bool textureNPOT = false;

    // GL_EXT_draw_buffers
    bool drawBuffers = false;

    // GL_EXT_texture_filter_anisotropic
    bool textureFilterAnisotropic = false;
    GLfloat maxTextureAnisotropy  = 0.0f;

    // GL_EXT_occlusion_query_boolean
    bool occlusionQueryBoolean = false;

    // GL_NV_fence
    bool fence = false;

    // GL_EXT_disjoint_timer_query
    bool disjointTimerQuery            = false;
    GLuint queryCounterBitsTimeElapsed = 0;
    GLuint queryCounterBitsTimestamp   = 0;

    // GL_EXT_robustness
    bool robustness = false;

    // GL_KHR_robust_buffer_access_behavior
    bool robustBufferAccessBehavior = false;

    // GL_EXT_blend_minmax
    bool blendMinMax = false;

    // GL_ANGLE_framebuffer_blit
    bool framebufferBlit = false;

    // GL_ANGLE_framebuffer_multisample
    bool framebufferMultisample = false;

    // GL_EXT_multisampled_render_to_texture
    bool multisampledRenderToTexture = false;

    // GL_ANGLE_instanced_arrays
    bool instancedArraysANGLE = false;
    // GL_EXT_instanced_arrays
    bool instancedArraysEXT = false;
    // Any version of the instanced arrays extension
    bool instancedArraysAny() const { return (instancedArraysANGLE || instancedArraysEXT); }

    // GL_ANGLE_pack_reverse_row_order
    bool packReverseRowOrder = false;

    // GL_OES_standard_derivatives
    bool standardDerivatives = false;

    // GL_EXT_shader_texture_lod
    bool shaderTextureLOD = false;

    // GL_EXT_frag_depth
    bool fragDepth = false;

    // OVR_multiview
    bool multiview  = false;
    GLuint maxViews = 1;

    // OVR_multiview2
    bool multiview2 = false;

    // GL_ANGLE_texture_usage
    bool textureUsage = false;

    // GL_ANGLE_translated_shader_source
    bool translatedShaderSource = false;

    // GL_OES_fbo_render_mipmap
    bool fboRenderMipmap = false;

    // GL_EXT_discard_framebuffer
    bool discardFramebuffer = false;

    // EXT_debug_marker
    bool debugMarker = false;

    // GL_OES_EGL_image
    bool eglImage = false;

    // GL_OES_EGL_image_external
    bool eglImageExternal = false;

    // GL_OES_EGL_image_external_essl3
    bool eglImageExternalEssl3 = false;

    // GL_MGL_EGL_image_cube
    bool eglImageCubeMGL = false;

    // GL_OES_EGL_sync
    bool eglSync = false;

    // GL_EXT_memory_object
    bool memoryObject = false;

    // GL_EXT_memory_object_fd
    bool memoryObjectFd = false;

    // GL_EXT_semaphore
    bool semaphore = false;

    // GL_EXT_semaphore_fd
    bool semaphoreFd = false;

    // NV_EGL_stream_consumer_external
    bool eglStreamConsumerExternal = false;

    // EXT_unpack_subimage
    bool unpackSubimage = false;

    // NV_pack_subimage
    bool packSubimage = false;

    // GL_OES_vertex_half_float
    bool vertexHalfFloat = false;

    // GL_OES_vertex_array_object
    bool vertexArrayObject = false;

    // GL_OES_vertex_type_10_10_10_2
    bool vertexAttribType1010102 = false;

    // GL_KHR_debug
    bool debug                     = false;
    GLuint maxDebugMessageLength   = 0;
    GLuint maxDebugLoggedMessages  = 0;
    GLuint maxDebugGroupStackDepth = 0;
    GLuint maxLabelLength          = 0;

    // KHR_no_error
    bool noError = false;

    // GL_ANGLE_lossy_etc_decode
    bool lossyETCDecode = false;

    // GL_CHROMIUM_bind_uniform_location
    bool bindUniformLocation = false;

    // GL_CHROMIUM_sync_query
    bool syncQuery = false;

    // GL_CHROMIUM_copy_texture
    bool copyTexture = false;

    // GL_CHROMIUM_copy_compressed_texture
    bool copyCompressedTexture = false;

    // GL_ANGLE_copy_texture_3d
    bool copyTexture3d = false;

    // GL_ANGLE_webgl_compatibility
    bool webglCompatibility = false;

    // GL_ANGLE_request_extension
    bool requestExtension = false;

    // GL_CHROMIUM_bind_generates_resource
    bool bindGeneratesResource = false;

    // GL_ANGLE_robust_client_memory
    bool robustClientMemory = false;

    // GL_OES_texture_border_clamp
    bool textureBorderClamp = false;

    // GL_EXT_texture_sRGB_decode
    bool textureSRGBDecode = false;

    // GL_EXT_sRGB_write_control
    bool sRGBWriteControl = false;

    // GL_CHROMIUM_color_buffer_float_rgb
    bool colorBufferFloatRGB = false;

    // GL_CHROMIUM_color_buffer_float_rgba
    bool colorBufferFloatRGBA = false;

    // ES3 Extension support

    // GL_EXT_color_buffer_float
    bool colorBufferFloat = false;

    // GL_EXT_multisample_compatibility.
    // written against ES 3.1 but can apply to earlier versions.
    bool multisampleCompatibility = false;

    // GL_CHROMIUM_framebuffer_mixed_samples
    bool framebufferMixedSamples = false;

    // GL_EXT_texture_norm16
    // written against ES 3.1 but can apply to ES 3.0 as well.
    bool textureNorm16 = false;

    // GL_CHROMIUM_path_rendering
    bool pathRendering = false;

    // GL_OES_surfaceless_context
    bool surfacelessContext = false;

    // GL_ANGLE_client_arrays
    bool clientArrays = false;

    // GL_ANGLE_robust_resource_initialization
    bool robustResourceInitialization = false;

    // GL_ANGLE_program_cache_control
    bool programCacheControl = false;

    // GL_ANGLE_texture_rectangle
    bool textureRectangle = false;

    // GL_EXT_geometry_shader
    bool geometryShader = false;

    // GLES1 emulation: GLES1 extensions
    // GL_OES_point_size_array
    bool pointSizeArray = false;

    // GL_OES_texture_cube_map
    bool textureCubeMap = false;

    // GL_OES_point_sprite
    bool pointSprite = false;

    // GL_OES_draw_texture
    bool drawTexture = false;

    // EGL_ANGLE_explicit_context GL subextensions
    // GL_ANGLE_explicit_context_gles1
    bool explicitContextGles1 = false;
    // GL_ANGLE_explicit_context
    bool explicitContext = false;

    // GL_KHR_parallel_shader_compile
    bool parallelShaderCompile = false;

    // GL_OES_texture_storage_multisample_2d_array
    bool textureStorageMultisample2DArray = false;

    // GL_ANGLE_multiview_multisample
    bool multiviewMultisample = false;

    // GL_EXT_blend_func_extended
    bool blendFuncExtended          = false;
    GLuint maxDualSourceDrawBuffers = 0;

    // GL_EXT_float_blend
    bool floatBlend = false;

    // GL_ANGLE_memory_size
    bool memorySize = false;

    // GL_ANGLE_texture_multisample
    bool textureMultisample = false;

    // GL_ANGLE_multi_draw
    bool multiDraw = false;

    // GL_ANGLE_provoking_vertex
    bool provokingVertex = false;

    // GL_CHROMIUM_lose_context
    bool loseContextCHROMIUM = false;

    // GL_ANGLE_texture_external_update
    bool textureExternalUpdateANGLE = false;

    // GL_ANGLE_base_vertex_base_instance
    bool baseVertexBaseInstance = false;

    // GL_APPLE_clip_distance
    bool clipDistanceAPPLE = false;
};

struct ExtensionInfo
{
    // If this extension can be enabled with glRequestExtension (GL_ANGLE_request_extension)
    bool Requestable = false;

    // Pointer to a boolean member of the Extensions struct
    typedef bool(Extensions::*ExtensionBool);
    ExtensionBool ExtensionsMember = nullptr;
};

using ExtensionInfoMap = std::map<std::string, ExtensionInfo>;
const ExtensionInfoMap &GetExtensionInfoMap();

struct Limitations
{
    Limitations();

    // Renderer doesn't support gl_FrontFacing in fragment shaders
    bool noFrontFacingSupport = false;

    // Renderer doesn't support GL_SAMPLE_ALPHA_TO_COVERAGE
    bool noSampleAlphaToCoverageSupport = false;

    // In glVertexAttribDivisorANGLE, attribute zero must have a zero divisor
    bool attributeZeroRequiresZeroDivisorInEXT = false;

    // Unable to support different values for front and back faces for stencil refs and masks
    bool noSeparateStencilRefsAndMasks = false;

    // Renderer doesn't support non-constant indexing loops in fragment shader
    bool shadersRequireIndexedLoopValidation = false;

    // Renderer doesn't support Simultaneous use of GL_CONSTANT_ALPHA/GL_ONE_MINUS_CONSTANT_ALPHA
    // and GL_CONSTANT_COLOR/GL_ONE_MINUS_CONSTANT_COLOR blend functions.
    bool noSimultaneousConstantColorAndAlphaBlendFunc = false;

    // D3D9 does not support flexible varying register packing.
    bool noFlexibleVaryingPacking = false;

    // D3D does not support having multiple transform feedback outputs go to the same buffer.
    bool noDoubleBoundTransformFeedbackBuffers = false;

    // D3D does not support vertex attribute aliasing
    bool noVertexAttributeAliasing = false;
};

struct TypePrecision
{
    TypePrecision();
    TypePrecision(const TypePrecision &other);

    void setIEEEFloat();
    void setTwosComplementInt(unsigned int bits);
    void setSimulatedFloat(unsigned int range, unsigned int precision);
    void setSimulatedInt(unsigned int range);

    void get(GLint *returnRange, GLint *returnPrecision) const;

    std::array<GLint, 2> range = {0, 0};
    GLint precision            = 0;
};

struct Caps
{
    Caps();
    Caps(const Caps &other);
    ~Caps();

    // ES 3.1 (April 29, 2015) 20.39: implementation dependent values
    GLuint64 maxElementIndex       = 0;
    GLuint max3DTextureSize        = 0;
    GLuint max2DTextureSize        = 0;
    GLuint maxRectangleTextureSize = 0;
    GLuint maxArrayTextureLayers   = 0;
    GLfloat maxLODBias             = 0.0f;
    GLuint maxCubeMapTextureSize   = 0;
    GLuint maxRenderbufferSize     = 0;
    GLfloat minAliasedPointSize    = 1.0f;
    GLfloat maxAliasedPointSize    = 1.0f;
    GLfloat minAliasedLineWidth    = 0.0f;
    GLfloat maxAliasedLineWidth    = 0.0f;

    // ES 3.1 (April 29, 2015) 20.40: implementation dependent values (cont.)
    GLuint maxDrawBuffers         = 0;
    GLuint maxFramebufferWidth    = 0;
    GLuint maxFramebufferHeight   = 0;
    GLuint maxFramebufferSamples  = 0;
    GLuint maxColorAttachments    = 0;
    GLuint maxViewportWidth       = 0;
    GLuint maxViewportHeight      = 0;
    GLuint maxSampleMaskWords     = 0;
    GLuint maxColorTextureSamples = 0;
    GLuint maxDepthTextureSamples = 0;
    GLuint maxIntegerSamples      = 0;
    GLuint64 maxServerWaitTimeout = 0;

    // ES 3.1 (April 29, 2015) Table 20.41: Implementation dependent values (cont.)
    GLint maxVertexAttribRelativeOffset = 0;
    GLuint maxVertexAttribBindings      = 0;
    GLint maxVertexAttribStride         = 0;
    GLuint maxElementsIndices           = 0;
    GLuint maxElementsVertices          = 0;
    std::vector<GLenum> compressedTextureFormats;
    std::vector<GLenum> programBinaryFormats;
    std::vector<GLenum> shaderBinaryFormats;
    TypePrecision vertexHighpFloat;
    TypePrecision vertexMediumpFloat;
    TypePrecision vertexLowpFloat;
    TypePrecision vertexHighpInt;
    TypePrecision vertexMediumpInt;
    TypePrecision vertexLowpInt;
    TypePrecision fragmentHighpFloat;
    TypePrecision fragmentMediumpFloat;
    TypePrecision fragmentLowpFloat;
    TypePrecision fragmentHighpInt;
    TypePrecision fragmentMediumpInt;
    TypePrecision fragmentLowpInt;

    // Implementation dependent limits required on all shader types.
    // TODO(jiawei.shao@intel.com): organize all such limits into ShaderMap.
    // ES 3.1 (April 29, 2015) Table 20.43: Implementation dependent Vertex shader limits
    // ES 3.1 (April 29, 2015) Table 20.44: Implementation dependent Fragment shader limits
    // ES 3.1 (April 29, 2015) Table 20.45: implementation dependent compute shader limits
    // GL_EXT_geometry_shader (May 31, 2016) Table 20.43gs: Implementation dependent geometry shader
    // limits
    // GL_EXT_geometry_shader (May 31, 2016) Table 20.46: Implementation dependent aggregate shader
    // limits
    ShaderMap<GLuint> maxShaderUniformBlocks        = {};
    ShaderMap<GLuint> maxShaderTextureImageUnits    = {};
    ShaderMap<GLuint> maxShaderStorageBlocks        = {};
    ShaderMap<GLuint> maxShaderUniformComponents    = {};
    ShaderMap<GLuint> maxShaderAtomicCounterBuffers = {};
    ShaderMap<GLuint> maxShaderAtomicCounters       = {};
    ShaderMap<GLuint> maxShaderImageUniforms        = {};
    // Note that we can query MAX_COMPUTE_UNIFORM_COMPONENTS and MAX_GEOMETRY_UNIFORM_COMPONENTS_EXT
    // by GetIntegerv, but we can only use GetInteger64v on MAX_VERTEX_UNIFORM_COMPONENTS and
    // MAX_FRAGMENT_UNIFORM_COMPONENTS. Currently we use GLuint64 to store all these values so that
    // we can put them together into one ShaderMap.
    ShaderMap<GLuint64> maxCombinedShaderUniformComponents = {};

    // ES 3.1 (April 29, 2015) Table 20.43: Implementation dependent Vertex shader limits
    GLuint maxVertexAttributes       = 0;
    GLuint maxVertexUniformVectors   = 0;
    GLuint maxVertexOutputComponents = 0;

    // ES 3.1 (April 29, 2015) Table 20.44: Implementation dependent Fragment shader limits
    GLuint maxFragmentUniformVectors     = 0;
    GLuint maxFragmentInputComponents    = 0;
    GLint minProgramTextureGatherOffset  = 0;
    GLuint maxProgramTextureGatherOffset = 0;
    GLint minProgramTexelOffset          = 0;
    GLint maxProgramTexelOffset          = 0;

    // ES 3.1 (April 29, 2015) Table 20.45: implementation dependent compute shader limits
    std::array<GLuint, 3> maxComputeWorkGroupCount = {0, 0, 0};
    std::array<GLuint, 3> maxComputeWorkGroupSize  = {0, 0, 0};
    GLuint maxComputeWorkGroupInvocations          = 0;
    GLuint maxComputeSharedMemorySize              = 0;

    // ES 3.1 (April 29, 2015) Table 20.46: implementation dependent aggregate shader limits
    GLuint maxUniformBufferBindings         = 0;
    GLuint64 maxUniformBlockSize            = 0;
    GLuint uniformBufferOffsetAlignment     = 0;
    GLuint maxCombinedUniformBlocks         = 0;
    GLuint maxVaryingComponents             = 0;
    GLuint maxVaryingVectors                = 0;
    GLuint maxCombinedTextureImageUnits     = 0;
    GLuint maxCombinedShaderOutputResources = 0;

    // ES 3.1 (April 29, 2015) Table 20.47: implementation dependent aggregate shader limits (cont.)
    GLuint maxUniformLocations                = 0;
    GLuint maxAtomicCounterBufferBindings     = 0;
    GLuint maxAtomicCounterBufferSize         = 0;
    GLuint maxCombinedAtomicCounterBuffers    = 0;
    GLuint maxCombinedAtomicCounters          = 0;
    GLuint maxImageUnits                      = 0;
    GLuint maxCombinedImageUniforms           = 0;
    GLuint maxShaderStorageBufferBindings     = 0;
    GLuint64 maxShaderStorageBlockSize        = 0;
    GLuint maxCombinedShaderStorageBlocks     = 0;
    GLuint shaderStorageBufferOffsetAlignment = 0;

    // ES 3.1 (April 29, 2015) Table 20.48: implementation dependent transform feedback limits
    GLuint maxTransformFeedbackInterleavedComponents = 0;
    GLuint maxTransformFeedbackSeparateAttributes    = 0;
    GLuint maxTransformFeedbackSeparateComponents    = 0;

    // ES 3.1 (April 29, 2015) Table 20.49: Framebuffer Dependent Values
    GLuint maxSamples = 0;

    // GL_EXT_geometry_shader (May 31, 2016) Table 20.40: Implementation-Dependent Values (cont.)
    GLuint maxFramebufferLayers = 0;
    GLuint layerProvokingVertex = 0;

    // GL_EXT_geometry_shader (May 31, 2016) Table 20.43gs: Implementation dependent geometry shader
    // limits
    GLuint maxGeometryInputComponents       = 0;
    GLuint maxGeometryOutputComponents      = 0;
    GLuint maxGeometryOutputVertices        = 0;
    GLuint maxGeometryTotalOutputComponents = 0;
    GLuint maxGeometryShaderInvocations     = 0;

    GLuint subPixelBits = 4;

    // GL_APPLE_clip_distance/GL_EXT_clip_cull_distance
    GLuint maxClipDistances = 0;

    // GLES1 emulation: Caps for ES 1.1. Taken from Table 6.20 / 6.22 in the OpenGL ES 1.1 spec.
    GLuint maxMultitextureUnits                 = 0;
    GLuint maxClipPlanes                        = 0;
    GLuint maxLights                            = 0;
    static constexpr int GlobalMatrixStackDepth = 16;
    GLuint maxModelviewMatrixStackDepth         = 0;
    GLuint maxProjectionMatrixStackDepth        = 0;
    GLuint maxTextureMatrixStackDepth           = 0;
    GLfloat minSmoothPointSize                  = 0.0f;
    GLfloat maxSmoothPointSize                  = 0.0f;
    GLfloat minSmoothLineWidth                  = 0.0f;
    GLfloat maxSmoothLineWidth                  = 0.0f;
};

Caps GenerateMinimumCaps(const Version &clientVersion, const Extensions &extensions);
}  // namespace gl

namespace egl
{

struct Caps
{
    Caps();

    // Support for NPOT surfaces
    bool textureNPOT;
};

struct DisplayExtensions
{
    DisplayExtensions();

    // Generate a vector of supported extension strings
    std::vector<std::string> getStrings() const;

    // EGL_EXT_create_context_robustness
    bool createContextRobustness = false;

    // EGL_ANGLE_d3d_share_handle_client_buffer
    bool d3dShareHandleClientBuffer = false;

    // EGL_ANGLE_d3d_texture_client_buffer
    bool d3dTextureClientBuffer = false;

    // EGL_ANGLE_surface_d3d_texture_2d_share_handle
    bool surfaceD3DTexture2DShareHandle = false;

    // EGL_ANGLE_query_surface_pointer
    bool querySurfacePointer = false;

    // EGL_ANGLE_window_fixed_size
    bool windowFixedSize = false;

    // EGL_ANGLE_keyed_mutex
    bool keyedMutex = false;

    // EGL_ANGLE_surface_orientation
    bool surfaceOrientation = false;

    // EGL_NV_post_sub_buffer
    bool postSubBuffer = false;

    // EGL_KHR_create_context
    bool createContext = false;

    // EGL_EXT_device_query
    bool deviceQuery = false;

    // EGL_KHR_image
    bool image = false;

    // EGL_KHR_image_base
    bool imageBase = false;

    // EGL_KHR_image_pixmap
    bool imagePixmap = false;

    // EGL_KHR_gl_texture_2D_image
    bool glTexture2DImage = false;

    // EGL_KHR_gl_texture_cubemap_image
    bool glTextureCubemapImage = false;

    // EGL_KHR_gl_texture_3D_image
    bool glTexture3DImage = false;

    // EGL_KHR_gl_renderbuffer_image
    bool glRenderbufferImage = false;

    // EGL_KHR_get_all_proc_addresses
    bool getAllProcAddresses = false;

    // EGL_ANGLE_flexible_surface_compatibility
    bool flexibleSurfaceCompatibility = false;

    // EGL_ANGLE_direct_composition
    bool directComposition = false;

    // EGL_ANGLE_windows_ui_composition
    bool windowsUIComposition = false;

    // KHR_create_context_no_error
    bool createContextNoError = false;

    // EGL_KHR_stream
    bool stream = false;

    // EGL_KHR_stream_consumer_gltexture
    bool streamConsumerGLTexture = false;

    // EGL_NV_stream_consumer_gltexture_yuv
    bool streamConsumerGLTextureYUV = false;

    // EGL_ANGLE_stream_producer_d3d_texture
    bool streamProducerD3DTexture = false;

    // EGL_KHR_fence_sync
    bool fenceSync = false;

    // EGL_KHR_wait_sync
    bool waitSync = false;

    // EGL_ANGLE_create_context_webgl_compatibility
    bool createContextWebGLCompatibility = false;

    // EGL_CHROMIUM_create_context_bind_generates_resource
    bool createContextBindGeneratesResource = false;

    // EGL_CHROMIUM_get_sync_values
    bool getSyncValues = false;

    // EGL_KHR_swap_buffers_with_damage
    bool swapBuffersWithDamage = false;

    // EGL_EXT_pixel_format_float
    bool pixelFormatFloat = false;

    // EGL_KHR_surfaceless_context
    bool surfacelessContext = false;

    // EGL_ANGLE_display_texture_share_group
    bool displayTextureShareGroup = false;

    // EGL_ANGLE_create_context_client_arrays
    bool createContextClientArrays = false;

    // EGL_ANGLE_program_cache_control
    bool programCacheControl = false;

    // EGL_ANGLE_robust_resource_initialization
    bool robustResourceInitialization = false;

    // EGL_ANGLE_iosurface_client_buffer
    bool iosurfaceClientBuffer = false;

    // EGL_mtl_texture_client_buffer
    bool mtlTextureClientBuffer = false;

    // EGL_ANGLE_create_context_extensions_enabled
    bool createContextExtensionsEnabled = false;

    // EGL_ANDROID_presentation_time
    bool presentationTime = false;

    // EGL_ANDROID_blob_cache
    bool blobCache = false;

    // EGL_ANDROID_image_native_buffer
    bool imageNativeBuffer = false;

    // EGL_ANDROID_get_frame_timestamps
    bool getFrameTimestamps = false;

    // EGL_ANDROID_recordable
    bool recordable = false;

    // EGL_ANGLE_power_preference
    bool powerPreference = false;

    // EGL_ANGLE_image_d3d11_texture
    bool imageD3D11Texture = false;

    // EGL_ANDROID_get_native_client_buffer
    bool getNativeClientBufferANDROID = false;

    // EGL_ANDROID_native_fence_sync
    bool nativeFenceSyncANDROID = false;

    // EGL_ANGLE_create_context_backwards_compatible
    bool createContextBackwardsCompatible = false;

    // EGL_KHR_no_config_context
    bool noConfigContext = false;

    // EGL_KHR_gl_colorspace
    bool glColorspace = false;

    // EGL_EXT_gl_colorspace_display_p3_linear
    bool glColorspaceDisplayP3Linear = false;

    // EGL_EXT_gl_colorspace_display_p3
    bool glColorspaceDisplayP3 = false;

    // EGL_EXT_gl_colorspace_scrgb
    bool glColorspaceScrgb = false;

    // EGL_EXT_gl_colorspace_scrgb_linear
    bool glColorspaceScrgbLinear = false;

    // EGL_EXT_gl_colorspace_display_p3_passthrough
    bool glColorspaceDisplayP3Passthrough = false;
};

struct DeviceExtensions
{
    DeviceExtensions();

    // Generate a vector of supported extension strings
    std::vector<std::string> getStrings() const;

    // EGL_ANGLE_device_d3d
    bool deviceD3D = false;

    // EGL_ANGLE_device_cgl
    bool deviceCGL = false;

    // EGL_ANGLE_device_mtl
    bool deviceMTL = false;
};

struct ClientExtensions
{
    ClientExtensions();
    ClientExtensions(const ClientExtensions &other);

    // Generate a vector of supported extension strings
    std::vector<std::string> getStrings() const;

    // EGL_EXT_client_extensions
    bool clientExtensions = false;

    // EGL_EXT_platform_base
    bool platformBase = false;

    // EGL_EXT_platform_device
    bool platformDevice = false;

    // EGL_ANGLE_platform_angle
    bool platformANGLE = false;

    // EGL_ANGLE_platform_angle_d3d
    bool platformANGLED3D = false;

    // EGL_ANGLE_platform_angle_opengl
    bool platformANGLEOpenGL = false;

    // EGL_ANGLE_platform_angle_null
    bool platformANGLENULL = false;

    // EGL_ANGLE_platform_angle_vulkan
    bool platformANGLEVulkan = false;

    // EGL_ANGLE_platform_angle_metal
    bool platformANGLEMetal = false;

    // EGL_ANGLE_platform_angle_context_virtualization
    bool platformANGLEContextVirtualization = false;

    // EGL_ANGLE_device_creation
    bool deviceCreation = false;

    // EGL_ANGLE_device_creation_d3d11
    bool deviceCreationD3D11 = false;

    // EGL_ANGLE_x11_visual
    bool x11Visual = false;

    // EGL_ANGLE_experimental_present_path
    bool experimentalPresentPath = false;

    // EGL_KHR_client_get_all_proc_addresses
    bool clientGetAllProcAddresses = false;

    // EGL_KHR_debug
    bool debug = false;

    // EGL_ANGLE_explicit_context
    bool explicitContext = false;

    // EGL_ANGLE_feature_control
    bool featureControlANGLE = false;

    // EGL_ANGLE_platform_angle_device_type_swiftshader
    bool platformANGLEDeviceTypeSwiftShader = false;
};

}  // namespace egl

#endif  // LIBANGLE_CAPS_H_
