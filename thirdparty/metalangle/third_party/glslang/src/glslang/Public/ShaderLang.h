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
//#ifdef SH_EXPORTING
//    #define SH_IMPORT_EXPORT __declspec(dllexport)
//#else
//    #define SH_IMPORT_EXPORT __declspec(dllimport)
//#endif
#define SH_IMPORT_EXPORT
#else
#define SH_IMPORT_EXPORT
#define C_DECL
#endif

//
// This is the platform independent interface between an OGL driver
// and the shading language compiler/linker.
//

#ifdef __cplusplus
    extern "C" {
#endif

// This should always increase, as some paths to do not consume
// a more major number.
// It should increment by one when new functionality is added.
#define GLSLANG_MINOR_VERSION 13

//
// Call before doing any other compiler/linker operations.
//
// (Call once per process, not once per thread.)
//
SH_IMPORT_EXPORT int ShInitialize();

//
// Call this at process shutdown to clean up memory.
//
SH_IMPORT_EXPORT int ShFinalize();

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
    EShLangTaskNV,
    EShLangMeshNV,
    LAST_ELEMENT_MARKER(EShLangCount),
} EShLanguage;         // would be better as stage, but this is ancient now

typedef enum {
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
    EShLangTaskNVMask         = (1 << EShLangTaskNV),
    EShLangMeshNVMask         = (1 << EShLangMeshNV),
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
    EShClientVulkan,
    EShClientOpenGL,
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
    EShTargetOpenGL_450 = 450,                        // OpenGL
    LAST_ELEMENT_MARKER(EShTargetClientVersionCount),
} EShTargetClientVersion;

typedef EShTargetClientVersion EshTargetClientVersion;

typedef enum {
    EShTargetSpv_1_0 = (1 << 16),                     // SPIR-V 1.0
    EShTargetSpv_1_1 = (1 << 16) | (1 << 8),          // SPIR-V 1.1
    EShTargetSpv_1_2 = (1 << 16) | (2 << 8),          // SPIR-V 1.2
    EShTargetSpv_1_3 = (1 << 16) | (3 << 8),          // SPIR-V 1.3
    EShTargetSpv_1_4 = (1 << 16) | (4 << 8),          // SPIR-V 1.4
    EShTargetSpv_1_5 = (1 << 16) | (5 << 8),          // SPIR-V 1.5
    LAST_ELEMENT_MARKER(EShTargetLanguageVersionCount),
} EShTargetLanguageVersion;

struct TInputLanguage {
    EShSource languageFamily; // redundant information with other input, this one overrides when not EShSourceNone
    EShLanguage stage;        // redundant information with other input, this one overrides when not EShSourceNone
    EShClient dialect;
    int dialectVersion;       // version of client's language definition, not the client (when not EShClientNone)
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

const char* StageName(EShLanguage);

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
enum EShMessages {
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
    EShMsgHlslDX9Compatible = (1 << 13), // enable HLSL DX9 compatible mode (right now only for samplers)
    EShMsgBuiltinSymbolTable = (1 << 14), // print the builtin symbol table
    LAST_ELEMENT_MARKER(EShMsgCount),
};

//
// Options for building reflection
//
typedef enum {
    EShReflectionDefault           = 0,        // default is original behaviour before options were added
    EShReflectionStrictArraySuffix = (1 << 0), // reflection will follow stricter rules for array-of-structs suffixes
    EShReflectionBasicArraySuffix  = (1 << 1), // arrays of basic types will be appended with [0] as in GL reflection
    EShReflectionIntermediateIO    = (1 << 2), // reflect inputs and outputs to program, even with no vertex shader
    EShReflectionSeparateBuffers   = (1 << 3), // buffer variables and buffer blocks are reflected separately
    EShReflectionAllBlockVariables = (1 << 4), // reflect all variables in blocks, even if they are inactive
    EShReflectionUnwrapIOBlocks    = (1 << 5), // unwrap input/output blocks the same as with uniform blocks
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
// managed, and de-allocated by the compiler/linker. It's contents
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
SH_IMPORT_EXPORT ShHandle ShConstructCompiler(const EShLanguage, int debugOptions);  // one per shader
SH_IMPORT_EXPORT ShHandle ShConstructLinker(const EShExecutable, int debugOptions);  // one per shader pair
SH_IMPORT_EXPORT ShHandle ShConstructUniformMap();                 // one per uniform namespace (currently entire program object)
SH_IMPORT_EXPORT void ShDestruct(ShHandle);

//
// The return value of ShCompile is boolean, non-zero indicating
// success.
//
// The info-log should be written by ShCompile into
// ShHandle, so it can answer future queries.
//
SH_IMPORT_EXPORT int ShCompile(
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

SH_IMPORT_EXPORT int ShLinkExt(
    const ShHandle,               // linker object
    const ShHandle h[],           // compiler objects to link together
    const int numHandles);

//
// ShSetEncrpytionMethod is a place-holder for specifying
// how source code is encrypted.
//
SH_IMPORT_EXPORT void ShSetEncryptionMethod(ShHandle);

//
// All the following return 0 if the information is not
// available in the object passed down, or the object is bad.
//
SH_IMPORT_EXPORT const char* ShGetInfoLog(const ShHandle);
SH_IMPORT_EXPORT const void* ShGetExecutable(const ShHandle);
SH_IMPORT_EXPORT int ShSetVirtualAttributeBindings(const ShHandle, const ShBindingTable*);   // to detect user aliasing
SH_IMPORT_EXPORT int ShSetFixedAttributeBindings(const ShHandle, const ShBindingTable*);     // to force any physical mappings
//
// Tell the linker to never assign a vertex attribute to this list of physical attributes
//
SH_IMPORT_EXPORT int ShExcludeAttributes(const ShHandle, int *attributes, int count);

//
// Returns the location ID of the named uniform.
// Returns -1 if error.
//
SH_IMPORT_EXPORT int ShGetUniformLocation(const ShHandle uniformMap, const char* name);

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

const char* GetEsslVersionString();
const char* GetGlslVersionString();
int GetKhronosToolId();

class TIntermediate;
class TProgram;
class TPoolAllocator;

// Call this exactly once per process before using anything else
bool InitializeProcess();

// Call once per process to tear down everything
void FinalizeProcess();

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
    explicit TShader(EShLanguage);
    virtual ~TShader();
    void setStrings(const char* const* s, int n);
    void setStringsWithLengths(const char* const* s, const int* l, int n);
    void setStringsWithLengthsAndNames(
        const char* const* s, const int* l, const char* const* names, int n);
    void setPreamble(const char* s) { preamble = s; }
    void setEntryPoint(const char* entryPoint);
    void setSourceEntryPoint(const char* sourceEntryPointName);
    void addProcesses(const std::vector<std::string>&);

    // IO resolver binding data: see comments in ShaderLang.cpp
    void setShiftBinding(TResourceType res, unsigned int base);
    void setShiftSamplerBinding(unsigned int base);  // DEPRECATED: use setShiftBinding
    void setShiftTextureBinding(unsigned int base);  // DEPRECATED: use setShiftBinding
    void setShiftImageBinding(unsigned int base);    // DEPRECATED: use setShiftBinding
    void setShiftUboBinding(unsigned int base);      // DEPRECATED: use setShiftBinding
    void setShiftUavBinding(unsigned int base);      // DEPRECATED: use setShiftBinding
    void setShiftCbufferBinding(unsigned int base);  // synonym for setShiftUboBinding
    void setShiftSsboBinding(unsigned int base);     // DEPRECATED: use setShiftBinding
    void setShiftBindingForSet(TResourceType res, unsigned int base, unsigned int set);
    void setResourceSetBinding(const std::vector<std::string>& base);
    void setAutoMapBindings(bool map);
    void setAutoMapLocations(bool map);
    void addUniformLocationOverride(const char* name, int loc);
    void setUniformLocationBase(int base);
    void setInvertY(bool invert);
#ifdef ENABLE_HLSL
    void setHlslIoMapping(bool hlslIoMap);
    void setFlattenUniformArrays(bool flatten);
#endif
    void setNoStorageFormat(bool useUnknownFormat);
    void setNanMinMaxClamp(bool nanMinMaxClamp);
    void setTextureSamplerTransformMode(EShTextureSamplerTransformMode mode);

    // For setting up the environment (cleared to nothingness in the constructor).
    // These must be called so that parsing is done for the right source language and
    // target environment, either indirectly through TranslateEnvironment() based on
    // EShMessages et. al., or directly by the user.
    //
    // setEnvInput:    The input source language and stage. If generating code for a
    //                 specific client, the input client semantics to use and the
    //                 version of the that client's input semantics to use, otherwise
    //                 use EShClientNone and version of 0, e.g. for validation mode.
    //                 Note 'version' does not describe the target environment,
    //                 just the version of the source dialect to compile under.
    //
    //                 See the definitions of TEnvironment, EShSource, EShLanguage,
    //                 and EShClient for choices and more detail.
    //
    // setEnvClient:   The client that will be hosting the execution, and it's version.
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

    bool parse(const TBuiltInResource*, int defaultVersion, EProfile defaultProfile, bool forceDefaultVersionAndProfile,
               bool forwardCompatible, EShMessages, Includer&);

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
    bool preprocess(const TBuiltInResource* builtInResources,
                    int defaultVersion, EProfile defaultProfile, bool forceDefaultVersionAndProfile,
                    bool forwardCompatible, EShMessages message, std::string* outputString,
                    Includer& includer);

    const char* getInfoLog();
    const char* getInfoDebugLog();
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

    TEnvironment environment;

    friend class TProgram;

private:
    TShader& operator=(TShader&);
};

#ifndef GLSLANG_WEB

//
// A reflection database and its interface, consistent with the OpenGL API reflection queries.
//

// Data needed for just a single object at the granularity exchanged by the reflection API
class TObjectReflection {
public:
    TObjectReflection(const std::string& pName, const TType& pType, int pOffset, int pGLDefineType, int pSize, int pIndex);

    const TType* getType() const { return type; }
    int getBinding() const;
    void dump() const;
    static TObjectReflection badReflection() { return TObjectReflection(); }

    std::string name;
    int offset;
    int glDefineType;
    int size;                   // data size in bytes for a block, array size for a (non-block) object that's an array
    int index;
    int counterIndex;
    int numMembers;
    int arrayStride;            // stride of an array variable
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
    virtual void addStage(EShLanguage stage) = 0;
};

#endif // GLSLANG_WEB

// Make one TProgram per set of shaders that will get linked together.  Add all
// the shaders that are to be linked together.  After calling shader.parse()
// for all shaders, call link().
//
// N.B.: Destruct a linked program *before* destructing the shaders linked into it.
//
class TProgram {
public:
    TProgram();
    virtual ~TProgram();
    void addShader(TShader* shader) { stages[shader->stage].push_back(shader); }
    std::list<TShader*>& getShaders(EShLanguage stage) { return stages[stage]; }
    // Link Validation interface
    bool link(EShMessages);
    const char* getInfoLog();
    const char* getInfoDebugLog();

    TIntermediate* getIntermediate(EShLanguage stage) const { return intermediate[stage]; }

#ifndef GLSLANG_WEB

    // Reflection Interface

    // call first, to do liveness analysis, index mapping, etc.; returns false on failure
    bool buildReflection(int opts = EShReflectionDefault);
    unsigned getLocalSize(int dim) const;                  // return dim'th local size
    int getReflectionIndex(const char *name) const;
    int getReflectionPipeIOIndex(const char* name, const bool inOrOut) const;
    int getNumUniformVariables() const;
    const TObjectReflection& getUniform(int index) const;
    int getNumUniformBlocks() const;
    const TObjectReflection& getUniformBlock(int index) const;
    int getNumPipeInputs() const;
    const TObjectReflection& getPipeInput(int index) const;
    int getNumPipeOutputs() const;
    const TObjectReflection& getPipeOutput(int index) const;
    int getNumBufferVariables() const;
    const TObjectReflection& getBufferVariable(int index) const;
    int getNumBufferBlocks() const;
    const TObjectReflection& getBufferBlock(int index) const;
    int getNumAtomicCounters() const;
    const TObjectReflection& getAtomicCounter(int index) const;

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

    void dumpReflection();
    // I/O mapping: apply base offsets and map live unbound variables
    // If resolver is not provided it uses the previous approach
    // and respects auto assignment and offsets.
    bool mapIO(TIoMapResolver* pResolver = nullptr, TIoMapper* pIoMapper = nullptr);
#endif

protected:
    bool linkStage(EShLanguage, EShMessages);

    TPoolAllocator* pool;
    std::list<TShader*> stages[EShLangCount];
    TIntermediate* intermediate[EShLangCount];
    bool newedIntermediate[EShLangCount];      // track which intermediate were "new" versus reusing a singleton unit in a stage
    TInfoSink* infoSink;
#ifndef GLSLANG_WEB
    TReflection* reflection;
#endif
    bool linked;

private:
    TProgram(TProgram&);
    TProgram& operator=(TProgram&);
};

} // end namespace glslang

#endif // _COMPILER_INTERFACE_INCLUDED_
