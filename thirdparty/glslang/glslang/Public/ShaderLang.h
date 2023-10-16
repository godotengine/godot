//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013-2016 LunarG, Inc.
// Copyright (C) 2015-2018 Google, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
#ifndef _COMPILER_INTERFACE_INCLUDED_
#define _COMPILER_INTERFACE_INCLUDED_

#include "../Include/ResourceLimits.h"
#include "../MachineIndependent/Versions.h"

#include <cstring>
#include <vector>

#ifdef _WIN32
    #define C_DECL __cdecl
#else
    #define C_DECL
#endif

#ifdef GLSLANG_IS_SHARED_LIBRARY
    #ifdef _WIN32
        #ifdef GLSLANG_EXPORTING
            #define GLSLANG_EXPORT __declspec(dllexport)
        #else
            #define GLSLANG_EXPORT __declspec(dllimport)
        #endif
    #elif __GNUC__ >= 4
        #define GLSLANG_EXPORT __attribute__((visibility("default")))
    #endif
#endif // GLSLANG_IS_SHARED_LIBRARY

#ifndef GLSLANG_EXPORT
#define GLSLANG_EXPORT
#endif

//
// This is the platform independent interface between an OGL driver
// and the shading language compiler/linker.
//

#ifdef __cplusplus
    extern "C" {
#endif

//
// Call before doing any other compiler/linker operations.
//
// (Call once per process, not once per thread.)
//
GLSLANG_EXPORT int ShInitialize();

//
// Call this at process shutdown to clean up memory.
//
GLSLANG_EXPORT int ShFinalize();

//
// Types of languages the compiler can consume.
//
typedef enum {
    EShLangVertex,
    EShLangTessControl,
    EShLangTessEvaluation,
    EShLangGeometry,
    EShLangFragment,
    EShLangCompute,
    EShLangRayGen,
    EShLangRayGenNV = EShLangRayGen,
    EShLangIntersect,
    EShLangIntersectNV = EShLangIntersect,
    EShLangAnyHit,
    EShLangAnyHitNV = EShLangAnyHit,
    EShLangClosestHit,
    EShLangClosestHitNV = EShLangClosestHit,
    EShLangMiss,
    EShLangMissNV = EShLangMiss,
    EShLangCallable,
    EShLangCallableNV = EShLangCallable,
    EShLangTask,
    EShLangTaskNV = EShLangTask,
    EShLangMesh,
    EShLangMeshNV = EShLangMesh,
    LAST_ELEMENT_MARKER(EShLangCount),
} EShLanguage;         // would be better as stage, but this is ancient now

typedef enum : unsigned {
    EShLangVertexMask         = (1 << EShLangVertex),
    EShLangTessControlMask    = (1 << EShLangTessControl),
    EShLangTessEvaluationMask = (1 << EShLangTessEvaluation),
    EShLangGeometryMask       = (1 << EShLangGeometry),
    EShLangFragmentMask       = (1 << EShLangFragment),
    EShLangComputeMask        = (1 << EShLangCompute),
    EShLangRayGenMask         = (1 << EShLangRayGen),
    EShLangRayGenNVMask       = EShLangRayGenMask,
    EShLangIntersectMask      = (1 << EShLangIntersect),
    EShLangIntersectNVMask    = EShLangIntersectMask,
    EShLangAnyHitMask         = (1 << EShLangAnyHit),
    EShLangAnyHitNVMask       = EShLangAnyHitMask,
    EShLangClosestHitMask     = (1 << EShLangClosestHit),
    EShLangClosestHitNVMask   = EShLangClosestHitMask,
    EShLangMissMask           = (1 << EShLangMiss),
    EShLangMissNVMask         = EShLangMissMask,
    EShLangCallableMask       = (1 << EShLangCallable),
    EShLangCallableNVMask     = EShLangCallableMask,
    EShLangTaskMask           = (1 << EShLangTask),
    EShLangTaskNVMask         = EShLangTaskMask,
    EShLangMeshMask           = (1 << EShLangMesh),
    EShLangMeshNVMask         = EShLangMeshMask,
    LAST_ELEMENT_MARKER(EShLanguageMaskCount),
} EShLanguageMask;

namespace glslang {

class TType;

typedef enum {
    EShSourceNone,
    EShSourceGlsl,               // GLSL, includes ESSL (OpenGL ES GLSL)
    EShSourceHlsl,               // HLSL
    LAST_ELEMENT_MARKER(EShSourceCount),
} EShSource;                     // if EShLanguage were EShStage, this could be EShLanguage instead

typedef enum {
    EShClientNone,               // use when there is no client, e.g. for validation
    EShClientVulkan,             // as GLSL dialect, specifies KHR_vulkan_glsl extension
    EShClientOpenGL,             // as GLSL dialect, specifies ARB_gl_spirv extension
    LAST_ELEMENT_MARKER(EShClientCount),
} EShClient;

typedef enum {
    EShTargetNone,
    EShTargetSpv,                 // SPIR-V (preferred spelling)
    EshTargetSpv = EShTargetSpv,  // legacy spelling
    LAST_ELEMENT_MARKER(EShTargetCount),
} EShTargetLanguage;

typedef enum {
    EShTargetVulkan_1_0 = (1 << 22),                  // Vulkan 1.0
    EShTargetVulkan_1_1 = (1 << 22) | (1 << 12),      // Vulkan 1.1
    EShTargetVulkan_1_2 = (1 << 22) | (2 << 12),      // Vulkan 1.2
    EShTargetVulkan_1_3 = (1 << 22) | (3 << 12),      // Vulkan 1.3
    EShTargetOpenGL_450 = 450,                        // OpenGL
    LAST_ELEMENT_MARKER(EShTargetClientVersionCount = 5),
} EShTargetClientVersion;

typedef EShTargetClientVersion EshTargetClientVersion;

typedef enum {
    EShTargetSpv_1_0 = (1 << 16),                     // SPIR-V 1.0
    EShTargetSpv_1_1 = (1 << 16) | (1 << 8),          // SPIR-V 1.1
    EShTargetSpv_1_2 = (1 << 16) | (2 << 8),          // SPIR-V 1.2
    EShTargetSpv_1_3 = (1 << 16) | (3 << 8),          // SPIR-V 1.3
    EShTargetSpv_1_4 = (1 << 16) | (4 << 8),          // SPIR-V 1.4
    EShTargetSpv_1_5 = (1 << 16) | (5 << 8),          // SPIR-V 1.5
    EShTargetSpv_1_6 = (1 << 16) | (6 << 8),          // SPIR-V 1.6
    LAST_ELEMENT_MARKER(EShTargetLanguageVersionCount = 7),
} EShTargetLanguageVersion;

struct TInputLanguage {
    EShSource languageFamily; // redundant information with other input, this one overrides when not EShSourceNone
    EShLanguage stage;        // redundant information with other input, this one overrides when not EShSourceNone
    EShClient dialect;
    int dialectVersion;       // version of client's language definition, not the client (when not EShClientNone)
    bool vulkanRulesRelaxed;
};

struct TClient {
    EShClient client;
    EShTargetClientVersion version;   // version of client itself (not the client's input dialect)
};

struct TTarget {
    EShTargetLanguage language;
    EShTargetLanguageVersion version; // version to target, if SPIR-V, defined by "word 1" of the SPIR-V header
    bool hlslFunctionality1;          // can target hlsl_functionality1 extension(s)
};

// All source/client/target versions and settings.
// Can override previous methods of setting, when items are set here.
// Expected to grow, as more are added, rather than growing parameter lists.
struct TEnvironment {
    TInputLanguage input;     // definition of the input language
    TClient client;           // what client is the overall compilation being done for?
    TTarget target;           // what to generate
};

GLSLANG_EXPORT const char* StageName(EShLanguage);

} // end namespace glslang

//
// Types of output the linker will create.
//
typedef enum {
    EShExVertexFragment,
    EShExFragment
} EShExecutable;

//
// Optimization level for the compiler.
//
typedef enum {
    EShOptNoGeneration,
    EShOptNone,
    EShOptSimple,       // Optimizations that can be done quickly
    EShOptFull,         // Optimizations that will take more time
    LAST_ELEMENT_MARKER(EshOptLevelCount),
} EShOptimizationLevel;

//
// Texture and Sampler transformation mode.
//
typedef enum {
    EShTexSampTransKeep,   // keep textures and samplers as is (default)
    EShTexSampTransUpgradeTextureRemoveSampler,  // change texture w/o embeded sampler into sampled texture and throw away all samplers
    LAST_ELEMENT_MARKER(EShTexSampTransCount),
} EShTextureSamplerTransformMode;

//
// Message choices for what errors and warnings are given.
//
enum EShMessages : unsigned {
    EShMsgDefault          = 0,         // default is to give all required errors and extra warnings
    EShMsgRelaxedErrors    = (1 << 0),  // be liberal in accepting input
    EShMsgSuppressWarnings = (1 << 1),  // suppress all warnings, except those required by the specification
    EShMsgAST              = (1 << 2),  // print the AST intermediate representation
    EShMsgSpvRules         = (1 << 3),  // issue messages for SPIR-V generation
    EShMsgVulkanRules      = (1 << 4),  // issue messages for Vulkan-requirements of GLSL for SPIR-V
    EShMsgOnlyPreprocessor = (1 << 5),  // only print out errors produced by the preprocessor
    EShMsgReadHlsl         = (1 << 6),  // use HLSL parsing rules and semantics
    EShMsgCascadingErrors  = (1 << 7),  // get cascading errors; risks error-recovery issues, instead of an early exit
    EShMsgKeepUncalled     = (1 << 8),  // for testing, don't eliminate uncalled functions
    EShMsgHlslOffsets      = (1 << 9),  // allow block offsets to follow HLSL rules instead of GLSL rules
    EShMsgDebugInfo        = (1 << 10), // save debug information
    EShMsgHlslEnable16BitTypes  = (1 << 11), // enable use of 16-bit types in SPIR-V for HLSL
    EShMsgHlslLegalization  = (1 << 12), // enable HLSL Legalization messages
    EShMsgHlslDX9Compatible = (1 << 13), // enable HLSL DX9 compatible mode (for samplers and semantics)
    EShMsgBuiltinSymbolTable = (1 << 14), // print the builtin symbol table
    EShMsgEnhanced         = (1 << 15), // enhanced message readability
    LAST_ELEMENT_MARKER(EShMsgCount),
};

//
// Options for building reflection
//
typedef enum {
    EShReflectionDefault            = 0,        // default is original behaviour before options were added
    EShReflectionStrictArraySuffix  = (1 << 0), // reflection will follow stricter rules for array-of-structs suffixes
    EShReflectionBasicArraySuffix   = (1 << 1), // arrays of basic types will be appended with [0] as in GL reflection
    EShReflectionIntermediateIO     = (1 << 2), // reflect inputs and outputs to program, even with no vertex shader
    EShReflectionSeparateBuffers    = (1 << 3), // buffer variables and buffer blocks are reflected separately
    EShReflectionAllBlockVariables  = (1 << 4), // reflect all variables in blocks, even if they are inactive
    EShReflectionUnwrapIOBlocks     = (1 << 5), // unwrap input/output blocks the same as with uniform blocks
    EShReflectionAllIOVariables     = (1 << 6), // reflect all input/output variables, even if they are inactive
    EShReflectionSharedStd140SSBO   = (1 << 7), // Apply std140/shared rules for ubo to ssbo
    EShReflectionSharedStd140UBO    = (1 << 8), // Apply std140/shared rules for ubo to ssbo
    LAST_ELEMENT_MARKER(EShReflectionCount),
} EShReflectionOptions;

//
// Build a table for bindings.  This can be used for locating
// attributes, uniforms, globals, etc., as needed.
//
typedef struct {
    const char* name;
    int binding;
} ShBinding;

typedef struct {
    int numBindings;
    ShBinding* bindings;  // array of bindings
} ShBindingTable;

//
// ShHandle held by but opaque to the driver.  It is allocated,
// managed, and de-allocated by the compiler/linker. Its contents
// are defined by and used by the compiler and linker.  For example,
// symbol table information and object code passed from the compiler
// to the linker can be stored where ShHandle points.
//
// If handle creation fails, 0 will be returned.
//
typedef void* ShHandle;

//
// Driver calls these to create and destroy compiler/linker
// objects.
//
GLSLANG_EXPORT ShHandle ShConstructCompiler(const EShLanguage, int debugOptions);  // one per shader
GLSLANG_EXPORT ShHandle ShConstructLinker(const EShExecutable, int debugOptions);  // one per shader pair
GLSLANG_EXPORT ShHandle ShConstructUniformMap();                 // one per uniform namespace (currently entire program object)
GLSLANG_EXPORT void ShDestruct(ShHandle);

//
// The return value of ShCompile is boolean, non-zero indicating
// success.
//
// The info-log should be written by ShCompile into
// ShHandle, so it can answer future queries.
//
GLSLANG_EXPORT int ShCompile(
    const ShHandle,
    const char* const shaderStrings[],
    const int numStrings,
    const int* lengths,
    const EShOptimizationLevel,
    const TBuiltInResource *resources,
    int debugOptions,
    int defaultVersion = 110,            // use 100 for ES environment, overridden by #version in shader
    bool forwardCompatible = false,      // give errors for use of deprecated features
    EShMessages messages = EShMsgDefault // warnings and errors
    );

GLSLANG_EXPORT int ShLinkExt(
    const ShHandle,               // linker object
    const ShHandle h[],           // compiler objects to link together
    const int numHandles);

//
// ShSetEncrpytionMethod is a place-holder for specifying
// how source code is encrypted.
//
GLSLANG_EXPORT void ShSetEncryptionMethod(ShHandle);

//
// All the following return 0 if the information is not
// available in the object passed down, or the object is bad.
//
GLSLANG_EXPORT const char* ShGetInfoLog(const ShHandle);
GLSLANG_EXPORT const void* ShGetExecutable(const ShHandle);
GLSLANG_EXPORT int ShSetVirtualAttributeBindings(const ShHandle, const ShBindingTable*);   // to detect user aliasing
GLSLANG_EXPORT int ShSetFixedAttributeBindings(const ShHandle, const ShBindingTable*);     // to force any physical mappings
//
// Tell the linker to never assign a vertex attribute to this list of physical attributes
//
GLSLANG_EXPORT int ShExcludeAttributes(const ShHandle, int *attributes, int count);

//
// Returns the location ID of the named uniform.
// Returns -1 if error.
//
GLSLANG_EXPORT int ShGetUniformLocation(const ShHandle uniformMap, const char* name);

#ifdef __cplusplus
    }  // end extern "C"
#endif

////////////////////////////////////////////////////////////////////////////////////////////
//
// Deferred-Lowering C++ Interface
// -----------------------------------
//
// Below is a new alternate C++ interface, which deprecates the above
// opaque handle-based interface.
//
// The below is further designed to handle multiple compilation units per stage, where
// the intermediate results, including the parse tree, are preserved until link time,
// rather than the above interface which is designed to have each compilation unit
// lowered at compile time.  In the above model, linking occurs on the lowered results,
// whereas in this model intra-stage linking can occur at the parse tree
// (treeRoot in TIntermediate) level, and then a full stage can be lowered.
//

#include <list>
#include <string>
#include <utility>

class TCompiler;
class TInfoSink;

namespace glslang {

struct Version {
    int major;
    int minor;
    int patch;
    const char* flavor;
};

GLSLANG_EXPORT Version GetVersion();
GLSLANG_EXPORT const char* GetEsslVersionString();
GLSLANG_EXPORT const char* GetGlslVersionString();
GLSLANG_EXPORT int GetKhronosToolId();

class TIntermediate;
class TProgram;
class TPoolAllocator;

// Call this exactly once per process before using anything else
GLSLANG_EXPORT bool InitializeProcess();

// Call once per process to tear down everything
GLSLANG_EXPORT void FinalizeProcess();

// Resource type for IO resolver
enum TResourceType {
    EResSampler,
    EResTexture,
    EResImage,
    EResUbo,
    EResSsbo,
    EResUav,
    EResCount
};

enum TBlockStorageClass
{
    EbsUniform = 0,
    EbsStorageBuffer,
    EbsPushConstant,
    EbsNone,    // not a uniform or buffer variable
    EbsCount,
};

// Make one TShader per shader that you will link into a program. Then
//  - provide the shader through setStrings() or setStringsWithLengths()
//  - optionally call setEnv*(), see below for more detail
//  - optionally use setPreamble() to set a special shader string that will be
//    processed before all others but won't affect the validity of #version
//  - optionally call addProcesses() for each setting/transform,
//    see comment for class TProcesses
//  - call parse(): source language and target environment must be selected
//    either by correct setting of EShMessages sent to parse(), or by
//    explicitly calling setEnv*()
//  - query the info logs
//
// N.B.: Does not yet support having the same TShader instance being linked into
// multiple programs.
//
// N.B.: Destruct a linked program *before* destructing the shaders linked into it.
//
class TShader {
public:
    GLSLANG_EXPORT explicit TShader(EShLanguage);
    GLSLANG_EXPORT virtual ~TShader();
    GLSLANG_EXPORT void setStrings(const char* const* s, int n);
    GLSLANG_EXPORT void setStringsWithLengths(
        const char* const* s, const int* l, int n);
    GLSLANG_EXPORT void setStringsWithLengthsAndNames(
        const char* const* s, const int* l, const char* const* names, int n);
    void setPreamble(const char* s) { preamble = s; }
    GLSLANG_EXPORT void setEntryPoint(const char* entryPoint);
    GLSLANG_EXPORT void setSourceEntryPoint(const char* sourceEntryPointName);
    GLSLANG_EXPORT void addProcesses(const std::vector<std::string>&);
    GLSLANG_EXPORT void setUniqueId(unsigned long long id);
    GLSLANG_EXPORT void setOverrideVersion(int version);
    GLSLANG_EXPORT void setDebugInfo(bool debugInfo);

    // IO resolver binding data: see comments in ShaderLang.cpp
    GLSLANG_EXPORT void setShiftBinding(TResourceType res, unsigned int base);
    GLSLANG_EXPORT void setShiftSamplerBinding(unsigned int base);  // DEPRECATED: use setShiftBinding
    GLSLANG_EXPORT void setShiftTextureBinding(unsigned int base);  // DEPRECATED: use setShiftBinding
    GLSLANG_EXPORT void setShiftImageBinding(unsigned int base);    // DEPRECATED: use setShiftBinding
    GLSLANG_EXPORT void setShiftUboBinding(unsigned int base);      // DEPRECATED: use setShiftBinding
    GLSLANG_EXPORT void setShiftUavBinding(unsigned int base);      // DEPRECATED: use setShiftBinding
    GLSLANG_EXPORT void setShiftCbufferBinding(unsigned int base);  // synonym for setShiftUboBinding
    GLSLANG_EXPORT void setShiftSsboBinding(unsigned int base);     // DEPRECATED: use setShiftBinding
    GLSLANG_EXPORT void setShiftBindingForSet(TResourceType res, unsigned int base, unsigned int set);
    GLSLANG_EXPORT void setResourceSetBinding(const std::vector<std::string>& base);
    GLSLANG_EXPORT void setAutoMapBindings(bool map);
    GLSLANG_EXPORT void setAutoMapLocations(bool map);
    GLSLANG_EXPORT void addUniformLocationOverride(const char* name, int loc);
    GLSLANG_EXPORT void setUniformLocationBase(int base);
    GLSLANG_EXPORT void setInvertY(bool invert);
    GLSLANG_EXPORT void setDxPositionW(bool dxPosW);
    GLSLANG_EXPORT void setEnhancedMsgs();
#ifdef ENABLE_HLSL
    GLSLANG_EXPORT void setHlslIoMapping(bool hlslIoMap);
    GLSLANG_EXPORT void setFlattenUniformArrays(bool flatten);
#endif
    GLSLANG_EXPORT void setNoStorageFormat(bool useUnknownFormat);
    GLSLANG_EXPORT void setNanMinMaxClamp(bool nanMinMaxClamp);
    GLSLANG_EXPORT void setTextureSamplerTransformMode(EShTextureSamplerTransformMode mode);
    GLSLANG_EXPORT void addBlockStorageOverride(const char* nameStr, glslang::TBlockStorageClass backing);

    GLSLANG_EXPORT void setGlobalUniformBlockName(const char* name);
    GLSLANG_EXPORT void setAtomicCounterBlockName(const char* name);
    GLSLANG_EXPORT void setGlobalUniformSet(unsigned int set);
    GLSLANG_EXPORT void setGlobalUniformBinding(unsigned int binding);
    GLSLANG_EXPORT void setAtomicCounterBlockSet(unsigned int set);
    GLSLANG_EXPORT void setAtomicCounterBlockBinding(unsigned int binding);

    // For setting up the environment (cleared to nothingness in the constructor).
    // These must be called so that parsing is done for the right source language and
    // target environment, either indirectly through TranslateEnvironment() based on
    // EShMessages et. al., or directly by the user.
    //
    // setEnvInput:    The input source language and stage. If generating code for a
    //                 specific client, the input client semantics to use and the
    //                 version of that client's input semantics to use, otherwise
    //                 use EShClientNone and version of 0, e.g. for validation mode.
    //                 Note 'version' does not describe the target environment,
    //                 just the version of the source dialect to compile under.
    //                 For example, to choose the Vulkan dialect of GLSL defined by
    //                 version 100 of the KHR_vulkan_glsl extension: lang = EShSourceGlsl,
    //                 dialect = EShClientVulkan, and version = 100.
    //
    //                 See the definitions of TEnvironment, EShSource, EShLanguage,
    //                 and EShClient for choices and more detail.
    //
    // setEnvClient:   The client that will be hosting the execution, and its version.
    //                 Note 'version' is not the version of the languages involved, but
    //                 the version of the client environment.
    //                 Use EShClientNone and version of 0 if there is no client, e.g.
    //                 for validation mode.
    //
    //                 See EShTargetClientVersion for choices.
    //
    // setEnvTarget:   The language to translate to when generating code, and that
    //                 language's version.
    //                 Use EShTargetNone and version of 0 if there is no client, e.g.
    //                 for validation mode.
    //
    void setEnvInput(EShSource lang, EShLanguage envStage, EShClient client, int version)
    {
        environment.input.languageFamily = lang;
        environment.input.stage = envStage;
        environment.input.dialect = client;
        environment.input.dialectVersion = version;
    }
    void setEnvClient(EShClient client, EShTargetClientVersion version)
    {
        environment.client.client = client;
        environment.client.version = version;
    }
    void setEnvTarget(EShTargetLanguage lang, EShTargetLanguageVersion version)
    {
        environment.target.language = lang;
        environment.target.version = version;
    }

    void getStrings(const char* const* &s, int& n) { s = strings; n = numStrings; }

#ifdef ENABLE_HLSL
    void setEnvTargetHlslFunctionality1() { environment.target.hlslFunctionality1 = true; }
    bool getEnvTargetHlslFunctionality1() const { return environment.target.hlslFunctionality1; }
#else
    bool getEnvTargetHlslFunctionality1() const { return false; }
#endif

    void setEnvInputVulkanRulesRelaxed() { environment.input.vulkanRulesRelaxed = true; }
    bool getEnvInputVulkanRulesRelaxed() const { return environment.input.vulkanRulesRelaxed; }

    // Interface to #include handlers.
    //
    // To support #include, a client of Glslang does the following:
    // 1. Call setStringsWithNames to set the source strings and associated
    //    names.  For example, the names could be the names of the files
    //    containing the shader sources.
    // 2. Call parse with an Includer.
    //
    // When the Glslang parser encounters an #include directive, it calls
    // the Includer's include method with the requested include name
    // together with the current string name.  The returned IncludeResult
    // contains the fully resolved name of the included source, together
    // with the source text that should replace the #include directive
    // in the source stream.  After parsing that source, Glslang will
    // release the IncludeResult object.
    class Includer {
    public:
        // An IncludeResult contains the resolved name and content of a source
        // inclusion.
        struct IncludeResult {
            IncludeResult(const std::string& headerName, const char* const headerData, const size_t headerLength, void* userData) :
                headerName(headerName), headerData(headerData), headerLength(headerLength), userData(userData) { }
            // For a successful inclusion, the fully resolved name of the requested
            // include.  For example, in a file system-based includer, full resolution
            // should convert a relative path name into an absolute path name.
            // For a failed inclusion, this is an empty string.
            const std::string headerName;
            // The content and byte length of the requested inclusion.  The
            // Includer producing this IncludeResult retains ownership of the
            // storage.
            // For a failed inclusion, the header
            // field points to a string containing error details.
            const char* const headerData;
            const size_t headerLength;
            // Include resolver's context.
            void* userData;
        protected:
            IncludeResult& operator=(const IncludeResult&);
            IncludeResult();
        };

        // For both include methods below:
        //
        // Resolves an inclusion request by name, current source name,
        // and include depth.
        // On success, returns an IncludeResult containing the resolved name
        // and content of the include.
        // On failure, returns a nullptr, or an IncludeResult
        // with an empty string for the headerName and error details in the
        // header field.
        // The Includer retains ownership of the contents
        // of the returned IncludeResult value, and those contents must
        // remain valid until the releaseInclude method is called on that
        // IncludeResult object.
        //
        // Note "local" vs. "system" is not an "either/or": "local" is an
        // extra thing to do over "system". Both might get called, as per
        // the C++ specification.

        // For the "system" or <>-style includes; search the "system" paths.
        virtual IncludeResult* includeSystem(const char* /*headerName*/,
                                             const char* /*includerName*/,
                                             size_t /*inclusionDepth*/) { return nullptr; }

        // For the "local"-only aspect of a "" include. Should not search in the
        // "system" paths, because on returning a failure, the parser will
        // call includeSystem() to look in the "system" locations.
        virtual IncludeResult* includeLocal(const char* /*headerName*/,
                                            const char* /*includerName*/,
                                            size_t /*inclusionDepth*/) { return nullptr; }

        // Signals that the parser will no longer use the contents of the
        // specified IncludeResult.
        virtual void releaseInclude(IncludeResult*) = 0;
        virtual ~Includer() {}
    };

    // Fail all Includer searches
    class ForbidIncluder : public Includer {
    public:
        virtual void releaseInclude(IncludeResult*) override { }
    };

    GLSLANG_EXPORT bool parse(
        const TBuiltInResource*, int defaultVersion, EProfile defaultProfile,
        bool forceDefaultVersionAndProfile, bool forwardCompatible,
        EShMessages, Includer&);

    bool parse(const TBuiltInResource* res, int defaultVersion, EProfile defaultProfile, bool forceDefaultVersionAndProfile,
               bool forwardCompatible, EShMessages messages)
    {
        TShader::ForbidIncluder includer;
        return parse(res, defaultVersion, defaultProfile, forceDefaultVersionAndProfile, forwardCompatible, messages, includer);
    }

    // Equivalent to parse() without a default profile and without forcing defaults.
    bool parse(const TBuiltInResource* builtInResources, int defaultVersion, bool forwardCompatible, EShMessages messages)
    {
        return parse(builtInResources, defaultVersion, ENoProfile, false, forwardCompatible, messages);
    }

    bool parse(const TBuiltInResource* builtInResources, int defaultVersion, bool forwardCompatible, EShMessages messages,
               Includer& includer)
    {
        return parse(builtInResources, defaultVersion, ENoProfile, false, forwardCompatible, messages, includer);
    }

    // NOTE: Doing just preprocessing to obtain a correct preprocessed shader string
    // is not an officially supported or fully working path.
    GLSLANG_EXPORT bool preprocess(
        const TBuiltInResource* builtInResources, int defaultVersion,
        EProfile defaultProfile, bool forceDefaultVersionAndProfile,
        bool forwardCompatible, EShMessages message, std::string* outputString,
        Includer& includer);

    GLSLANG_EXPORT const char* getInfoLog();
    GLSLANG_EXPORT const char* getInfoDebugLog();
    EShLanguage getStage() const { return stage; }
    TIntermediate* getIntermediate() const { return intermediate; }

protected:
    TPoolAllocator* pool;
    EShLanguage stage;
    TCompiler* compiler;
    TIntermediate* intermediate;
    TInfoSink* infoSink;
    // strings and lengths follow the standard for glShaderSource:
    //     strings is an array of numStrings pointers to string data.
    //     lengths can be null, but if not it is an array of numStrings
    //         integers containing the length of the associated strings.
    //         if lengths is null or lengths[n] < 0  the associated strings[n] is
    //         assumed to be null-terminated.
    // stringNames is the optional names for all the strings. If stringNames
    // is null, then none of the strings has name. If a certain element in
    // stringNames is null, then the corresponding string does not have name.
    const char* const* strings;      // explicit code to compile, see previous comment
    const int* lengths;
    const char* const* stringNames;
    int numStrings;                  // size of the above arrays
    const char* preamble;            // string of implicit code to compile before the explicitly provided code

    // a function in the source string can be renamed FROM this TO the name given in setEntryPoint.
    std::string sourceEntryPointName;

    // overrides #version in shader source or default version if #version isn't present
    int overrideVersion;

    TEnvironment environment;

    friend class TProgram;

private:
    TShader& operator=(TShader&);
};

//
// A reflection database and its interface, consistent with the OpenGL API reflection queries.
//

// Data needed for just a single object at the granularity exchanged by the reflection API
class TObjectReflection {
public:
    GLSLANG_EXPORT TObjectReflection(const std::string& pName, const TType& pType, int pOffset, int pGLDefineType, int pSize, int pIndex);

    const TType* getType() const { return type; }
    GLSLANG_EXPORT int getBinding() const;
    GLSLANG_EXPORT void dump() const;
    static TObjectReflection badReflection() { return TObjectReflection(); }

    std::string name;
    int offset;
    int glDefineType;
    int size;                   // data size in bytes for a block, array size for a (non-block) object that's an array
    int index;
    int counterIndex;
    int numMembers;
    int arrayStride;            // stride of an array variable
    int topLevelArraySize;      // size of the top-level variable in a storage buffer member
    int topLevelArrayStride;    // stride of the top-level variable in a storage buffer member
    EShLanguageMask stages;

protected:
    TObjectReflection()
        : offset(-1), glDefineType(-1), size(-1), index(-1), counterIndex(-1), numMembers(-1), arrayStride(0),
          topLevelArrayStride(0), stages(EShLanguageMask(0)), type(nullptr)
    {
    }

    const TType* type;
};

class  TReflection;
class  TIoMapper;
struct TVarEntryInfo;

// Allows to customize the binding layout after linking.
// All used uniform variables will invoke at least validateBinding.
// If validateBinding returned true then the other resolveBinding,
// resolveSet, and resolveLocation are invoked to resolve the binding
// and descriptor set index respectively.
//
// Invocations happen in a particular order:
// 1) all shader inputs
// 2) all shader outputs
// 3) all uniforms with binding and set already defined
// 4) all uniforms with binding but no set defined
// 5) all uniforms with set but no binding defined
// 6) all uniforms with no binding and no set defined
//
// mapIO will use this resolver in two phases. The first
// phase is a notification phase, calling the corresponging
// notifiy callbacks, this phase ends with a call to endNotifications.
// Phase two starts directly after the call to endNotifications
// and calls all other callbacks to validate and to get the
// bindings, sets, locations, component and color indices.
//
// NOTE: that still limit checks are applied to bindings and sets
// and may result in an error.
class TIoMapResolver
{
public:
    virtual ~TIoMapResolver() {}

    // Should return true if the resulting/current binding would be okay.
    // Basic idea is to do aliasing binding checks with this.
    virtual bool validateBinding(EShLanguage stage, TVarEntryInfo& ent) = 0;
    // Should return a value >= 0 if the current binding should be overridden.
    // Return -1 if the current binding (including no binding) should be kept.
    virtual int resolveBinding(EShLanguage stage, TVarEntryInfo& ent) = 0;
    // Should return a value >= 0 if the current set should be overridden.
    // Return -1 if the current set (including no set) should be kept.
    virtual int resolveSet(EShLanguage stage, TVarEntryInfo& ent) = 0;
    // Should return a value >= 0 if the current location should be overridden.
    // Return -1 if the current location (including no location) should be kept.
    virtual int resolveUniformLocation(EShLanguage stage, TVarEntryInfo& ent) = 0;
    // Should return true if the resulting/current setup would be okay.
    // Basic idea is to do aliasing checks and reject invalid semantic names.
    virtual bool validateInOut(EShLanguage stage, TVarEntryInfo& ent) = 0;
    // Should return a value >= 0 if the current location should be overridden.
    // Return -1 if the current location (including no location) should be kept.
    virtual int resolveInOutLocation(EShLanguage stage, TVarEntryInfo& ent) = 0;
    // Should return a value >= 0 if the current component index should be overridden.
    // Return -1 if the current component index (including no index) should be kept.
    virtual int resolveInOutComponent(EShLanguage stage, TVarEntryInfo& ent) = 0;
    // Should return a value >= 0 if the current color index should be overridden.
    // Return -1 if the current color index (including no index) should be kept.
    virtual int resolveInOutIndex(EShLanguage stage, TVarEntryInfo& ent) = 0;
    // Notification of a uniform variable
    virtual void notifyBinding(EShLanguage stage, TVarEntryInfo& ent) = 0;
    // Notification of a in or out variable
    virtual void notifyInOut(EShLanguage stage, TVarEntryInfo& ent) = 0;
    // Called by mapIO when it starts its notify pass for the given stage
    virtual void beginNotifications(EShLanguage stage) = 0;
    // Called by mapIO when it has finished the notify pass
    virtual void endNotifications(EShLanguage stage) = 0;
    // Called by mipIO when it starts its resolve pass for the given stage
    virtual void beginResolve(EShLanguage stage) = 0;
    // Called by mapIO when it has finished the resolve pass
    virtual void endResolve(EShLanguage stage) = 0;
    // Called by mapIO when it starts its symbol collect for teh given stage
    virtual void beginCollect(EShLanguage stage) = 0;
    // Called by mapIO when it has finished the symbol collect
    virtual void endCollect(EShLanguage stage) = 0;
    // Called by TSlotCollector to resolve storage locations or bindings
    virtual void reserverStorageSlot(TVarEntryInfo& ent, TInfoSink& infoSink) = 0;
    // Called by TSlotCollector to resolve resource locations or bindings
    virtual void reserverResourceSlot(TVarEntryInfo& ent, TInfoSink& infoSink) = 0;
    // Called by mapIO.addStage to set shader stage mask to mark a stage be added to this pipeline
    virtual void addStage(EShLanguage stage, TIntermediate& stageIntermediate) = 0;
};

// Make one TProgram per set of shaders that will get linked together.  Add all
// the shaders that are to be linked together.  After calling shader.parse()
// for all shaders, call link().
//
// N.B.: Destruct a linked program *before* destructing the shaders linked into it.
//
class TProgram {
public:
    GLSLANG_EXPORT TProgram();
    GLSLANG_EXPORT virtual ~TProgram();
    void addShader(TShader* shader) { stages[shader->stage].push_back(shader); }
    std::list<TShader*>& getShaders(EShLanguage stage) { return stages[stage]; }
    // Link Validation interface
    GLSLANG_EXPORT bool link(EShMessages);
    GLSLANG_EXPORT const char* getInfoLog();
    GLSLANG_EXPORT const char* getInfoDebugLog();

    TIntermediate* getIntermediate(EShLanguage stage) const { return intermediate[stage]; }

    // Reflection Interface

    // call first, to do liveness analysis, index mapping, etc.; returns false on failure
    GLSLANG_EXPORT bool buildReflection(int opts = EShReflectionDefault);
    GLSLANG_EXPORT unsigned getLocalSize(int dim) const;                  // return dim'th local size
    GLSLANG_EXPORT int getReflectionIndex(const char *name) const;
    GLSLANG_EXPORT int getReflectionPipeIOIndex(const char* name, const bool inOrOut) const;
    GLSLANG_EXPORT int getNumUniformVariables() const;
    GLSLANG_EXPORT const TObjectReflection& getUniform(int index) const;
    GLSLANG_EXPORT int getNumUniformBlocks() const;
    GLSLANG_EXPORT const TObjectReflection& getUniformBlock(int index) const;
    GLSLANG_EXPORT int getNumPipeInputs() const;
    GLSLANG_EXPORT const TObjectReflection& getPipeInput(int index) const;
    GLSLANG_EXPORT int getNumPipeOutputs() const;
    GLSLANG_EXPORT const TObjectReflection& getPipeOutput(int index) const;
    GLSLANG_EXPORT int getNumBufferVariables() const;
    GLSLANG_EXPORT const TObjectReflection& getBufferVariable(int index) const;
    GLSLANG_EXPORT int getNumBufferBlocks() const;
    GLSLANG_EXPORT const TObjectReflection& getBufferBlock(int index) const;
    GLSLANG_EXPORT int getNumAtomicCounters() const;
    GLSLANG_EXPORT const TObjectReflection& getAtomicCounter(int index) const;

    // Legacy Reflection Interface - expressed in terms of above interface

    // can be used for glGetProgramiv(GL_ACTIVE_UNIFORMS)
    int getNumLiveUniformVariables() const             { return getNumUniformVariables(); }

    // can be used for glGetProgramiv(GL_ACTIVE_UNIFORM_BLOCKS)
    int getNumLiveUniformBlocks() const                { return getNumUniformBlocks(); }

    // can be used for glGetProgramiv(GL_ACTIVE_ATTRIBUTES)
    int getNumLiveAttributes() const                   { return getNumPipeInputs(); }

    // can be used for glGetUniformIndices()
    int getUniformIndex(const char *name) const        { return getReflectionIndex(name); }

    int getPipeIOIndex(const char *name, const bool inOrOut) const
                                                       { return getReflectionPipeIOIndex(name, inOrOut); }

    // can be used for "name" part of glGetActiveUniform()
    const char *getUniformName(int index) const        { return getUniform(index).name.c_str(); }

    // returns the binding number
    int getUniformBinding(int index) const             { return getUniform(index).getBinding(); }

    // returns Shaders Stages where a Uniform is present
    EShLanguageMask getUniformStages(int index) const  { return getUniform(index).stages; }

    // can be used for glGetActiveUniformsiv(GL_UNIFORM_BLOCK_INDEX)
    int getUniformBlockIndex(int index) const          { return getUniform(index).index; }

    // can be used for glGetActiveUniformsiv(GL_UNIFORM_TYPE)
    int getUniformType(int index) const                { return getUniform(index).glDefineType; }

    // can be used for glGetActiveUniformsiv(GL_UNIFORM_OFFSET)
    int getUniformBufferOffset(int index) const        { return getUniform(index).offset; }

    // can be used for glGetActiveUniformsiv(GL_UNIFORM_SIZE)
    int getUniformArraySize(int index) const           { return getUniform(index).size; }

    // returns a TType*
    const TType *getUniformTType(int index) const      { return getUniform(index).getType(); }

    // can be used for glGetActiveUniformBlockName()
    const char *getUniformBlockName(int index) const   { return getUniformBlock(index).name.c_str(); }

    // can be used for glGetActiveUniformBlockiv(UNIFORM_BLOCK_DATA_SIZE)
    int getUniformBlockSize(int index) const           { return getUniformBlock(index).size; }

    // returns the block binding number
    int getUniformBlockBinding(int index) const        { return getUniformBlock(index).getBinding(); }

    // returns block index of associated counter.
    int getUniformBlockCounterIndex(int index) const   { return getUniformBlock(index).counterIndex; }

    // returns a TType*
    const TType *getUniformBlockTType(int index) const { return getUniformBlock(index).getType(); }

    // can be used for glGetActiveAttrib()
    const char *getAttributeName(int index) const      { return getPipeInput(index).name.c_str(); }

    // can be used for glGetActiveAttrib()
    int getAttributeType(int index) const              { return getPipeInput(index).glDefineType; }

    // returns a TType*
    const TType *getAttributeTType(int index) const    { return getPipeInput(index).getType(); }

    GLSLANG_EXPORT void dumpReflection();
    // I/O mapping: apply base offsets and map live unbound variables
    // If resolver is not provided it uses the previous approach
    // and respects auto assignment and offsets.
    GLSLANG_EXPORT bool mapIO(TIoMapResolver* pResolver = nullptr, TIoMapper* pIoMapper = nullptr);

protected:
    GLSLANG_EXPORT bool linkStage(EShLanguage, EShMessages);
    GLSLANG_EXPORT bool crossStageCheck(EShMessages);

    TPoolAllocator* pool;
    std::list<TShader*> stages[EShLangCount];
    TIntermediate* intermediate[EShLangCount];
    bool newedIntermediate[EShLangCount];      // track which intermediate were "new" versus reusing a singleton unit in a stage
    TInfoSink* infoSink;
    TReflection* reflection;
    bool linked;

private:
    TProgram(TProgram&);
    TProgram& operator=(TProgram&);
};

} // end namespace glslang

#endif // _COMPILER_INTERFACE_INCLUDED_
