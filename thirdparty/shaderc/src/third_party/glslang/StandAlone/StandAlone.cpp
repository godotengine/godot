//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013-2016 LunarG, Inc.
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

// this only applies to the standalone wrapper, not the front end in general
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ResourceLimits.h"
#include "Worklist.h"
#include "DirStackFileIncluder.h"
#include "./../glslang/Include/ShHandle.h"
#include "./../glslang/Include/revision.h"
#include "./../glslang/Public/ShaderLang.h"
#include "../SPIRV/GlslangToSpv.h"
#include "../SPIRV/GLSL.std.450.h"
#include "../SPIRV/doc.h"
#include "../SPIRV/disassemble.h"

#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <array>
#include <map>
#include <memory>
#include <thread>

#include "../glslang/OSDependent/osinclude.h"

extern "C" {
    SH_IMPORT_EXPORT void ShOutputHtml();
}

// Command-line options
enum TOptions {
    EOptionNone                 = 0,
    EOptionIntermediate         = (1 <<  0),
    EOptionSuppressInfolog      = (1 <<  1),
    EOptionMemoryLeakMode       = (1 <<  2),
    EOptionRelaxedErrors        = (1 <<  3),
    EOptionGiveWarnings         = (1 <<  4),
    EOptionLinkProgram          = (1 <<  5),
    EOptionMultiThreaded        = (1 <<  6),
    EOptionDumpConfig           = (1 <<  7),
    EOptionDumpReflection       = (1 <<  8),
    EOptionSuppressWarnings     = (1 <<  9),
    EOptionDumpVersions         = (1 << 10),
    EOptionSpv                  = (1 << 11),
    EOptionHumanReadableSpv     = (1 << 12),
    EOptionVulkanRules          = (1 << 13),
    EOptionDefaultDesktop       = (1 << 14),
    EOptionOutputPreprocessed   = (1 << 15),
    EOptionOutputHexadecimal    = (1 << 16),
    EOptionReadHlsl             = (1 << 17),
    EOptionCascadingErrors      = (1 << 18),
    EOptionAutoMapBindings      = (1 << 19),
    EOptionFlattenUniformArrays = (1 << 20),
    EOptionNoStorageFormat      = (1 << 21),
    EOptionKeepUncalled         = (1 << 22),
    EOptionHlslOffsets          = (1 << 23),
    EOptionHlslIoMapping        = (1 << 24),
    EOptionAutoMapLocations     = (1 << 25),
    EOptionDebug                = (1 << 26),
    EOptionStdin                = (1 << 27),
    EOptionOptimizeDisable      = (1 << 28),
    EOptionOptimizeSize         = (1 << 29),
    EOptionInvertY              = (1 << 30),
    EOptionDumpBareVersion      = (1 << 31),
};
bool targetHlslFunctionality1 = false;
bool SpvToolsDisassembler = false;
bool SpvToolsValidate = false;

//
// Return codes from main/exit().
//
enum TFailCode {
    ESuccess = 0,
    EFailUsage,
    EFailCompile,
    EFailLink,
    EFailCompilerCreate,
    EFailThreadCreate,
    EFailLinkerCreate
};

//
// Forward declarations.
//
EShLanguage FindLanguage(const std::string& name, bool parseSuffix=true);
void CompileFile(const char* fileName, ShHandle);
void usage();
char* ReadFileData(const char* fileName);
void FreeFileData(char* data);
void InfoLogMsg(const char* msg, const char* name, const int num);

// Globally track if any compile or link failure.
bool CompileFailed = false;
bool LinkFailed = false;

// array of unique places to leave the shader names and infologs for the asynchronous compiles
std::vector<std::unique_ptr<glslang::TWorkItem>> WorkItems;

TBuiltInResource Resources;
std::string ConfigFile;

//
// Parse either a .conf file provided by the user or the default from glslang::DefaultTBuiltInResource
//
void ProcessConfigFile()
{
    if (ConfigFile.size() == 0)
        Resources = glslang::DefaultTBuiltInResource;
    else {
        char* configString = ReadFileData(ConfigFile.c_str());
        glslang::DecodeResourceLimits(&Resources,  configString);
        FreeFileData(configString);
    }
}

int ReflectOptions = EShReflectionDefault;
int Options = 0;
const char* ExecutableName = nullptr;
const char* binaryFileName = nullptr;
const char* entryPointName = nullptr;
const char* sourceEntryPointName = nullptr;
const char* shaderStageName = nullptr;
const char* variableName = nullptr;
bool HlslEnable16BitTypes = false;
bool HlslDX9compatible = false;
std::vector<std::string> IncludeDirectoryList;

// Source environment
// (source 'Client' is currently the same as target 'Client')
int ClientInputSemanticsVersion = 100;

// Target environment
glslang::EShClient Client = glslang::EShClientNone;  // will stay EShClientNone if only validating
glslang::EShTargetClientVersion ClientVersion;       // not valid until Client is set
glslang::EShTargetLanguage TargetLanguage = glslang::EShTargetNone;
glslang::EShTargetLanguageVersion TargetVersion;     // not valid until TargetLanguage is set

std::vector<std::string> Processes;                     // what should be recorded by OpModuleProcessed, or equivalent

// Per descriptor-set binding base data
typedef std::map<unsigned int, unsigned int> TPerSetBaseBinding;

std::vector<std::pair<std::string, int>> uniformLocationOverrides;
int uniformBase = 0;

std::array<std::array<unsigned int, EShLangCount>, glslang::EResCount> baseBinding;
std::array<std::array<TPerSetBaseBinding, EShLangCount>, glslang::EResCount> baseBindingForSet;
std::array<std::vector<std::string>, EShLangCount> baseResourceSetBinding;

// Add things like "#define ..." to a preamble to use in the beginning of the shader.
class TPreamble {
public:
    TPreamble() { }

    bool isSet() const { return text.size() > 0; }
    const char* get() const { return text.c_str(); }

    // #define...
    void addDef(std::string def)
    {
        text.append("#define ");
        fixLine(def);

        Processes.push_back("D");
        Processes.back().append(def);

        // The first "=" needs to turn into a space
        const size_t equal = def.find_first_of("=");
        if (equal != def.npos)
            def[equal] = ' ';

        text.append(def);
        text.append("\n");
    }

    // #undef...
    void addUndef(std::string undef)
    {
        text.append("#undef ");
        fixLine(undef);

        Processes.push_back("U");
        Processes.back().append(undef);

        text.append(undef);
        text.append("\n");
    }

protected:
    void fixLine(std::string& line)
    {
        // Can't go past a newline in the line
        const size_t end = line.find_first_of("\n");
        if (end != line.npos)
            line = line.substr(0, end);
    }

    std::string text;  // contents of preamble
};

TPreamble UserPreamble;

//
// Create the default name for saving a binary if -o is not provided.
//
const char* GetBinaryName(EShLanguage stage)
{
    const char* name;
    if (binaryFileName == nullptr) {
        switch (stage) {
        case EShLangVertex:          name = "vert.spv";    break;
        case EShLangTessControl:     name = "tesc.spv";    break;
        case EShLangTessEvaluation:  name = "tese.spv";    break;
        case EShLangGeometry:        name = "geom.spv";    break;
        case EShLangFragment:        name = "frag.spv";    break;
        case EShLangCompute:         name = "comp.spv";    break;
#ifdef NV_EXTENSIONS
        case EShLangRayGenNV:        name = "rgen.spv";    break;
        case EShLangIntersectNV:     name = "rint.spv";    break;
        case EShLangAnyHitNV:        name = "rahit.spv";   break;
        case EShLangClosestHitNV:    name = "rchit.spv";   break;
        case EShLangMissNV:          name = "rmiss.spv";   break;
        case EShLangCallableNV:      name = "rcall.spv";   break;
        case EShLangMeshNV:          name = "mesh.spv";    break;
        case EShLangTaskNV:          name = "task.spv";    break;
#endif
        default:                     name = "unknown";     break;
        }
    } else
        name = binaryFileName;

    return name;
}

//
// *.conf => this is a config file that can set limits/resources
//
bool SetConfigFile(const std::string& name)
{
    if (name.size() < 5)
        return false;

    if (name.compare(name.size() - 5, 5, ".conf") == 0) {
        ConfigFile = name;
        return true;
    }

    return false;
}

//
// Give error and exit with failure code.
//
void Error(const char* message)
{
    fprintf(stderr, "%s: Error %s (use -h for usage)\n", ExecutableName, message);
    exit(EFailUsage);
}

//
// Process an optional binding base of one the forms:
//   --argname [stage] base            // base for stage (if given) or all stages (if not)
//   --argname [stage] [base set]...   // set/base pairs: set the base for given binding set.

// Where stage is one of the forms accepted by FindLanguage, and base is an integer
//
void ProcessBindingBase(int& argc, char**& argv, glslang::TResourceType res)
{
    if (argc < 2)
        usage();

    EShLanguage lang = EShLangCount;
    int singleBase = 0;
    TPerSetBaseBinding perSetBase;
    int arg = 1;

    // Parse stage, if given
    if (!isdigit(argv[arg][0])) {
        if (argc < 3) // this form needs one more argument
            usage();

        lang = FindLanguage(argv[arg++], false);
    }

    if ((argc - arg) > 2 && isdigit(argv[arg+0][0]) && isdigit(argv[arg+1][0])) {
        // Parse a per-set binding base
        while ((argc - arg) > 2 && isdigit(argv[arg+0][0]) && isdigit(argv[arg+1][0])) {
            const int baseNum = atoi(argv[arg++]);
            const int setNum = atoi(argv[arg++]);
            perSetBase[setNum] = baseNum;
        }
    } else {
        // Parse single binding base
        singleBase = atoi(argv[arg++]);
    }

    argc -= (arg-1);
    argv += (arg-1);

    // Set one or all languages
    const int langMin = (lang < EShLangCount) ? lang+0 : 0;
    const int langMax = (lang < EShLangCount) ? lang+1 : EShLangCount;

    for (int lang = langMin; lang < langMax; ++lang) {
        if (!perSetBase.empty())
            baseBindingForSet[res][lang].insert(perSetBase.begin(), perSetBase.end());
        else
            baseBinding[res][lang] = singleBase;
    }
}

void ProcessResourceSetBindingBase(int& argc, char**& argv, std::array<std::vector<std::string>, EShLangCount>& base)
{
    if (argc < 2)
        usage();

    if (!isdigit(argv[1][0])) {
        if (argc < 3) // this form needs one more argument
            usage();

        // Parse form: --argname stage [regname set base...], or:
        //             --argname stage set
        const EShLanguage lang = FindLanguage(argv[1], false);

        argc--;
        argv++;

        while (argc > 1 && argv[1] != nullptr && argv[1][0] != '-') {
            base[lang].push_back(argv[1]);

            argc--;
            argv++;
        }

        // Must have one arg, or a multiple of three (for [regname set binding] triples)
        if (base[lang].size() != 1 && (base[lang].size() % 3) != 0)
            usage();

    } else {
        // Parse form: --argname set
        for (int lang=0; lang<EShLangCount; ++lang)
            base[lang].push_back(argv[1]);

        argc--;
        argv++;
    }
}

//
// Do all command-line argument parsing.  This includes building up the work-items
// to be processed later, and saving all the command-line options.
//
// Does not return (it exits) if command-line is fatally flawed.
//
void ProcessArguments(std::vector<std::unique_ptr<glslang::TWorkItem>>& workItems, int argc, char* argv[])
{
    for (int res = 0; res < glslang::EResCount; ++res)
        baseBinding[res].fill(0);

    ExecutableName = argv[0];
    workItems.reserve(argc);

    const auto bumpArg = [&]() {
        if (argc > 0) {
            argc--;
            argv++;
        }
    };

    // read a string directly attached to a single-letter option
    const auto getStringOperand = [&](const char* desc) {
        if (argv[0][2] == 0) {
            printf("%s must immediately follow option (no spaces)\n", desc);
            exit(EFailUsage);
        }
        return argv[0] + 2;
    };

    // read a number attached to a single-letter option
    const auto getAttachedNumber = [&](const char* desc) {
        int num = atoi(argv[0] + 2);
        if (num == 0) {
            printf("%s: expected attached non-0 number\n", desc);
            exit(EFailUsage);
        }
        return num;
    };

    // minimum needed (without overriding something else) to target Vulkan SPIR-V
    const auto setVulkanSpv = []() {
        if (Client == glslang::EShClientNone)
            ClientVersion = glslang::EShTargetVulkan_1_0;
        Client = glslang::EShClientVulkan;
        Options |= EOptionSpv;
        Options |= EOptionVulkanRules;
        Options |= EOptionLinkProgram;
    };

    // minimum needed (without overriding something else) to target OpenGL SPIR-V
    const auto setOpenGlSpv = []() {
        if (Client == glslang::EShClientNone)
            ClientVersion = glslang::EShTargetOpenGL_450;
        Client = glslang::EShClientOpenGL;
        Options |= EOptionSpv;
        Options |= EOptionLinkProgram;
        // undo a -H default to Vulkan
        Options &= ~EOptionVulkanRules;
    };

    const auto getUniformOverride = [getStringOperand]() {
        const char *arg = getStringOperand("-u<name>:<location>");
        const char *split = strchr(arg, ':');
        if (split == NULL) {
            printf("%s: missing location\n", arg);
            exit(EFailUsage);
        }
        errno = 0;
        int location = ::strtol(split + 1, NULL, 10);
        if (errno) {
            printf("%s: invalid location\n", arg);
            exit(EFailUsage);
        }
        return std::make_pair(std::string(arg, split - arg), location);
    };

    for (bumpArg(); argc >= 1; bumpArg()) {
        if (argv[0][0] == '-') {
            switch (argv[0][1]) {
            case '-':
                {
                    std::string lowerword(argv[0]+2);
                    std::transform(lowerword.begin(), lowerword.end(), lowerword.begin(), ::tolower);

                    // handle --word style options
                    if (lowerword == "auto-map-bindings" ||  // synonyms
                        lowerword == "auto-map-binding"  ||
                        lowerword == "amb") {
                        Options |= EOptionAutoMapBindings;
                    } else if (lowerword == "auto-map-locations" || // synonyms
                               lowerword == "aml") {
                        Options |= EOptionAutoMapLocations;
                    } else if (lowerword == "uniform-base") {
                        if (argc <= 1)
                            Error("no <base> provided for --uniform-base");
                        uniformBase = ::strtol(argv[1], NULL, 10);
                        bumpArg();
                        break;
                    } else if (lowerword == "client") {
                        if (argc > 1) {
                            if (strcmp(argv[1], "vulkan100") == 0)
                                setVulkanSpv();
                            else if (strcmp(argv[1], "opengl100") == 0)
                                setOpenGlSpv();
                            else
                                Error("--client expects vulkan100 or opengl100");
                        }
                        bumpArg();
                    } else if (lowerword == "entry-point") {
                        entryPointName = argv[1];
                        if (argc <= 1)
                            Error("no <name> provided for --entry-point");
                        bumpArg();
                    } else if (lowerword == "flatten-uniform-arrays" || // synonyms
                               lowerword == "flatten-uniform-array"  ||
                               lowerword == "fua") {
                        Options |= EOptionFlattenUniformArrays;
                    } else if (lowerword == "hlsl-offsets") {
                        Options |= EOptionHlslOffsets;
                    } else if (lowerword == "hlsl-iomap" ||
                               lowerword == "hlsl-iomapper" ||
                               lowerword == "hlsl-iomapping") {
                        Options |= EOptionHlslIoMapping;
                    } else if (lowerword == "hlsl-enable-16bit-types") {
                        HlslEnable16BitTypes = true;
                    } else if (lowerword == "hlsl-dx9-compatible") {
                        HlslDX9compatible = true;
                    } else if (lowerword == "invert-y" ||  // synonyms
                               lowerword == "iy") {
                        Options |= EOptionInvertY;
                    } else if (lowerword == "keep-uncalled" || // synonyms
                               lowerword == "ku") {
                        Options |= EOptionKeepUncalled;
                    } else if (lowerword == "no-storage-format" || // synonyms
                               lowerword == "nsf") {
                        Options |= EOptionNoStorageFormat;
                    } else if (lowerword == "relaxed-errors") {
                        Options |= EOptionRelaxedErrors;
                    } else if (lowerword == "reflect-strict-array-suffix") {
                        ReflectOptions |= EShReflectionStrictArraySuffix;
                    } else if (lowerword == "reflect-basic-array-suffix") {
                        ReflectOptions |= EShReflectionBasicArraySuffix;
                    } else if (lowerword == "reflect-intermediate-io") {
                        ReflectOptions |= EShReflectionIntermediateIO;
                    } else if (lowerword == "reflect-separate-buffers") {
                        ReflectOptions |= EShReflectionSeparateBuffers;
                    } else if (lowerword == "reflect-all-block-variables") {
                        ReflectOptions |= EShReflectionAllBlockVariables;
                    } else if (lowerword == "reflect-unwrap-io-blocks") {
                        ReflectOptions |= EShReflectionUnwrapIOBlocks;
                    } else if (lowerword == "resource-set-bindings" ||  // synonyms
                               lowerword == "resource-set-binding"  ||
                               lowerword == "rsb") {
                        ProcessResourceSetBindingBase(argc, argv, baseResourceSetBinding);
                    } else if (lowerword == "shift-image-bindings" ||  // synonyms
                               lowerword == "shift-image-binding"  ||
                               lowerword == "sib") {
                        ProcessBindingBase(argc, argv, glslang::EResImage);
                    } else if (lowerword == "shift-sampler-bindings" || // synonyms
                               lowerword == "shift-sampler-binding"  ||
                               lowerword == "ssb") {
                        ProcessBindingBase(argc, argv, glslang::EResSampler);
                    } else if (lowerword == "shift-uav-bindings" ||  // synonyms
                               lowerword == "shift-uav-binding"  ||
                               lowerword == "suavb") {
                        ProcessBindingBase(argc, argv, glslang::EResUav);
                    } else if (lowerword == "shift-texture-bindings" ||  // synonyms
                               lowerword == "shift-texture-binding"  ||
                               lowerword == "stb") {
                        ProcessBindingBase(argc, argv, glslang::EResTexture);
                    } else if (lowerword == "shift-ubo-bindings" ||  // synonyms
                               lowerword == "shift-ubo-binding"  ||
                               lowerword == "shift-cbuffer-bindings" ||
                               lowerword == "shift-cbuffer-binding"  ||
                               lowerword == "sub" ||
                               lowerword == "scb") {
                        ProcessBindingBase(argc, argv, glslang::EResUbo);
                    } else if (lowerword == "shift-ssbo-bindings" ||  // synonyms
                               lowerword == "shift-ssbo-binding"  ||
                               lowerword == "sbb") {
                        ProcessBindingBase(argc, argv, glslang::EResSsbo);
                    } else if (lowerword == "source-entrypoint" || // synonyms
                               lowerword == "sep") {
                        if (argc <= 1)
                            Error("no <entry-point> provided for --source-entrypoint");
                        sourceEntryPointName = argv[1];
                        bumpArg();
                        break;
                    } else if (lowerword == "spirv-dis") {
                        SpvToolsDisassembler = true;
                    } else if (lowerword == "spirv-val") {
                        SpvToolsValidate = true;
                    } else if (lowerword == "stdin") {
                        Options |= EOptionStdin;
                        shaderStageName = argv[1];
                    } else if (lowerword == "suppress-warnings") {
                        Options |= EOptionSuppressWarnings;
                    } else if (lowerword == "target-env") {
                        if (argc > 1) {
                            if (strcmp(argv[1], "vulkan1.0") == 0) {
                                setVulkanSpv();
                                ClientVersion = glslang::EShTargetVulkan_1_0;
                            } else if (strcmp(argv[1], "vulkan1.1") == 0) {
                                setVulkanSpv();
                                ClientVersion = glslang::EShTargetVulkan_1_1;
                            } else if (strcmp(argv[1], "opengl") == 0) {
                                setOpenGlSpv();
                                ClientVersion = glslang::EShTargetOpenGL_450;
                            } else if (strcmp(argv[1], "spirv1.0") == 0) {
                                TargetLanguage = glslang::EShTargetSpv;
                                TargetVersion = glslang::EShTargetSpv_1_0;
                            } else if (strcmp(argv[1], "spirv1.1") == 0) {
                                TargetLanguage = glslang::EShTargetSpv;
                                TargetVersion = glslang::EShTargetSpv_1_1;
                            } else if (strcmp(argv[1], "spirv1.2") == 0) {
                                TargetLanguage = glslang::EShTargetSpv;
                                TargetVersion = glslang::EShTargetSpv_1_2;
                            } else if (strcmp(argv[1], "spirv1.3") == 0) {
                                TargetLanguage = glslang::EShTargetSpv;
                                TargetVersion = glslang::EShTargetSpv_1_3;
                            } else if (strcmp(argv[1], "spirv1.4") == 0) {
                                TargetLanguage = glslang::EShTargetSpv;
                                TargetVersion = glslang::EShTargetSpv_1_4;
                            } else
                                Error("--target-env expected one of: vulkan1.0, vulkan1.1, opengl, spirv1.0, spirv1.1, spirv1.2, or spirv1.3");
                        }
                        bumpArg();
                    } else if (lowerword == "variable-name" || // synonyms
                               lowerword == "vn") {
                        Options |= EOptionOutputHexadecimal;
                        if (argc <= 1)
                            Error("no <C-variable-name> provided for --variable-name");
                        variableName = argv[1];
                        bumpArg();
                        break;
                    } else if (lowerword == "version") {
                        Options |= EOptionDumpVersions;
                    } else {
                        usage();
                    }
                }
                break;
            case 'C':
                Options |= EOptionCascadingErrors;
                break;
            case 'D':
                if (argv[0][2] == 0)
                    Options |= EOptionReadHlsl;
                else
                    UserPreamble.addDef(getStringOperand("-D<macro> macro name"));
                break;
            case 'u':
                uniformLocationOverrides.push_back(getUniformOverride());
                break;
            case 'E':
                Options |= EOptionOutputPreprocessed;
                break;
            case 'G':
                // OpenGL client
                setOpenGlSpv();
                if (argv[0][2] != 0)
                    ClientInputSemanticsVersion = getAttachedNumber("-G<num> client input semantics");
                break;
            case 'H':
                Options |= EOptionHumanReadableSpv;
                if ((Options & EOptionSpv) == 0) {
                    // default to Vulkan
                    setVulkanSpv();
                }
                break;
            case 'I':
                IncludeDirectoryList.push_back(getStringOperand("-I<dir> include path"));
                break;
            case 'O':
                if (argv[0][2] == 'd')
                    Options |= EOptionOptimizeDisable;
                else if (argv[0][2] == 's')
#if ENABLE_OPT
                    Options |= EOptionOptimizeSize;
#else
                    Error("-Os not available; optimizer not linked");
#endif
                else
                    Error("unknown -O option");
                break;
            case 'S':
                if (argc <= 1)
                    Error("no <stage> specified for -S");
                shaderStageName = argv[1];
                bumpArg();
                break;
            case 'U':
                UserPreamble.addUndef(getStringOperand("-U<macro>: macro name"));
                break;
            case 'V':
                setVulkanSpv();
                if (argv[0][2] != 0)
                    ClientInputSemanticsVersion = getAttachedNumber("-V<num> client input semantics");
                break;
            case 'c':
                Options |= EOptionDumpConfig;
                break;
            case 'd':
                if (strncmp(&argv[0][1], "dumpversion", strlen(&argv[0][1]) + 1) == 0 ||
                    strncmp(&argv[0][1], "dumpfullversion", strlen(&argv[0][1]) + 1) == 0)
                    Options |= EOptionDumpBareVersion;
                else
                    Options |= EOptionDefaultDesktop;
                break;
            case 'e':
                entryPointName = argv[1];
                if (argc <= 1)
                    Error("no <name> provided for -e");
                bumpArg();
                break;
            case 'f':
                if (strcmp(&argv[0][2], "hlsl_functionality1") == 0)
                    targetHlslFunctionality1 = true;
                else
                    Error("-f: expected hlsl_functionality1");
                break;
            case 'g':
                Options |= EOptionDebug;
                break;
            case 'h':
                usage();
                break;
            case 'i':
                Options |= EOptionIntermediate;
                break;
            case 'l':
                Options |= EOptionLinkProgram;
                break;
            case 'm':
                Options |= EOptionMemoryLeakMode;
                break;
            case 'o':
                if (argc <= 1)
                    Error("no <file> provided for -o");
                binaryFileName = argv[1];
                bumpArg();
                break;
            case 'q':
                Options |= EOptionDumpReflection;
                break;
            case 'r':
                Options |= EOptionRelaxedErrors;
                break;
            case 's':
                Options |= EOptionSuppressInfolog;
                break;
            case 't':
                Options |= EOptionMultiThreaded;
                break;
            case 'v':
                Options |= EOptionDumpVersions;
                break;
            case 'w':
                Options |= EOptionSuppressWarnings;
                break;
            case 'x':
                Options |= EOptionOutputHexadecimal;
                break;
            default:
                usage();
                break;
            }
        } else {
            std::string name(argv[0]);
            if (! SetConfigFile(name)) {
                workItems.push_back(std::unique_ptr<glslang::TWorkItem>(new glslang::TWorkItem(name)));
            }
        }
    }

    // Make sure that -S is always specified if --stdin is specified
    if ((Options & EOptionStdin) && shaderStageName == nullptr)
        Error("must provide -S when --stdin is given");

    // Make sure that -E is not specified alongside linking (which includes SPV generation)
    if ((Options & EOptionOutputPreprocessed) && (Options & EOptionLinkProgram))
        Error("can't use -E when linking is selected");

    // -o or -x makes no sense if there is no target binary
    if (binaryFileName && (Options & EOptionSpv) == 0)
        Error("no binary generation requested (e.g., -V)");

    if ((Options & EOptionFlattenUniformArrays) != 0 &&
        (Options & EOptionReadHlsl) == 0)
        Error("uniform array flattening only valid when compiling HLSL source.");

    // rationalize client and target language
    if (TargetLanguage == glslang::EShTargetNone) {
        switch (ClientVersion) {
        case glslang::EShTargetVulkan_1_0:
            TargetLanguage = glslang::EShTargetSpv;
            TargetVersion = glslang::EShTargetSpv_1_0;
            break;
        case glslang::EShTargetVulkan_1_1:
            TargetLanguage = glslang::EShTargetSpv;
            TargetVersion = glslang::EShTargetSpv_1_3;
            break;
        case glslang::EShTargetOpenGL_450:
            TargetLanguage = glslang::EShTargetSpv;
            TargetVersion = glslang::EShTargetSpv_1_0;
            break;
        default:
            break;
        }
    }
    if (TargetLanguage != glslang::EShTargetNone && Client == glslang::EShClientNone)
        Error("To generate SPIR-V, also specify client semantics. See -G and -V.");
}

//
// Translate the meaningful subset of command-line options to parser-behavior options.
//
void SetMessageOptions(EShMessages& messages)
{
    if (Options & EOptionRelaxedErrors)
        messages = (EShMessages)(messages | EShMsgRelaxedErrors);
    if (Options & EOptionIntermediate)
        messages = (EShMessages)(messages | EShMsgAST);
    if (Options & EOptionSuppressWarnings)
        messages = (EShMessages)(messages | EShMsgSuppressWarnings);
    if (Options & EOptionSpv)
        messages = (EShMessages)(messages | EShMsgSpvRules);
    if (Options & EOptionVulkanRules)
        messages = (EShMessages)(messages | EShMsgVulkanRules);
    if (Options & EOptionOutputPreprocessed)
        messages = (EShMessages)(messages | EShMsgOnlyPreprocessor);
    if (Options & EOptionReadHlsl)
        messages = (EShMessages)(messages | EShMsgReadHlsl);
    if (Options & EOptionCascadingErrors)
        messages = (EShMessages)(messages | EShMsgCascadingErrors);
    if (Options & EOptionKeepUncalled)
        messages = (EShMessages)(messages | EShMsgKeepUncalled);
    if (Options & EOptionHlslOffsets)
        messages = (EShMessages)(messages | EShMsgHlslOffsets);
    if (Options & EOptionDebug)
        messages = (EShMessages)(messages | EShMsgDebugInfo);
    if (HlslEnable16BitTypes)
        messages = (EShMessages)(messages | EShMsgHlslEnable16BitTypes);
    if ((Options & EOptionOptimizeDisable) || !ENABLE_OPT)
        messages = (EShMessages)(messages | EShMsgHlslLegalization);
    if (HlslDX9compatible)
        messages = (EShMessages)(messages | EShMsgHlslDX9Compatible);
}

//
// Thread entry point, for non-linking asynchronous mode.
//
void CompileShaders(glslang::TWorklist& worklist)
{
    if (Options & EOptionDebug)
        Error("cannot generate debug information unless linking to generate code");

    glslang::TWorkItem* workItem;
    if (Options & EOptionStdin) {
        if (worklist.remove(workItem)) {
            ShHandle compiler = ShConstructCompiler(FindLanguage("stdin"), Options);
            if (compiler == nullptr)
                return;

            CompileFile("stdin", compiler);

            if (! (Options & EOptionSuppressInfolog))
                workItem->results = ShGetInfoLog(compiler);

            ShDestruct(compiler);
        }
    } else {
        while (worklist.remove(workItem)) {
            ShHandle compiler = ShConstructCompiler(FindLanguage(workItem->name), Options);
            if (compiler == 0)
                return;

            CompileFile(workItem->name.c_str(), compiler);

            if (! (Options & EOptionSuppressInfolog))
                workItem->results = ShGetInfoLog(compiler);

            ShDestruct(compiler);
        }
    }
}

// Outputs the given string, but only if it is non-null and non-empty.
// This prevents erroneous newlines from appearing.
void PutsIfNonEmpty(const char* str)
{
    if (str && str[0]) {
        puts(str);
    }
}

// Outputs the given string to stderr, but only if it is non-null and non-empty.
// This prevents erroneous newlines from appearing.
void StderrIfNonEmpty(const char* str)
{
    if (str && str[0])
        fprintf(stderr, "%s\n", str);
}

// Simple bundling of what makes a compilation unit for ease in passing around,
// and separation of handling file IO versus API (programmatic) compilation.
struct ShaderCompUnit {
    EShLanguage stage;
    static const int maxCount = 1;
    int count;                          // live number of strings/names
    const char* text[maxCount];         // memory owned/managed externally
    std::string fileName[maxCount];     // hold's the memory, but...
    const char* fileNameList[maxCount]; // downstream interface wants pointers

    ShaderCompUnit(EShLanguage stage) : stage(stage), count(0) { }

    ShaderCompUnit(const ShaderCompUnit& rhs)
    {
        stage = rhs.stage;
        count = rhs.count;
        for (int i = 0; i < count; ++i) {
            fileName[i] = rhs.fileName[i];
            text[i] = rhs.text[i];
            fileNameList[i] = rhs.fileName[i].c_str();
        }
    }

    void addString(std::string& ifileName, const char* itext)
    {
        assert(count < maxCount);
        fileName[count] = ifileName;
        text[count] = itext;
        fileNameList[count] = fileName[count].c_str();
        ++count;
    }
};

//
// For linking mode: Will independently parse each compilation unit, but then put them
// in the same program and link them together, making at most one linked module per
// pipeline stage.
//
// Uses the new C++ interface instead of the old handle-based interface.
//

void CompileAndLinkShaderUnits(std::vector<ShaderCompUnit> compUnits)
{
    // keep track of what to free
    std::list<glslang::TShader*> shaders;

    EShMessages messages = EShMsgDefault;
    SetMessageOptions(messages);

    //
    // Per-shader processing...
    //

    glslang::TProgram& program = *new glslang::TProgram;
    for (auto it = compUnits.cbegin(); it != compUnits.cend(); ++it) {
        const auto &compUnit = *it;
        glslang::TShader* shader = new glslang::TShader(compUnit.stage);
        shader->setStringsWithLengthsAndNames(compUnit.text, NULL, compUnit.fileNameList, compUnit.count);
        if (entryPointName)
            shader->setEntryPoint(entryPointName);
        if (sourceEntryPointName) {
            if (entryPointName == nullptr)
                printf("Warning: Changing source entry point name without setting an entry-point name.\n"
                       "Use '-e <name>'.\n");
            shader->setSourceEntryPoint(sourceEntryPointName);
        }
        if (UserPreamble.isSet())
            shader->setPreamble(UserPreamble.get());
        shader->addProcesses(Processes);

        // Set IO mapper binding shift values
        for (int r = 0; r < glslang::EResCount; ++r) {
            const glslang::TResourceType res = glslang::TResourceType(r);

            // Set base bindings
            shader->setShiftBinding(res, baseBinding[res][compUnit.stage]);
            
            // Set bindings for particular resource sets
            // TODO: use a range based for loop here, when available in all environments.
            for (auto i = baseBindingForSet[res][compUnit.stage].begin();
                 i != baseBindingForSet[res][compUnit.stage].end(); ++i)
                shader->setShiftBindingForSet(res, i->second, i->first);
        }

        shader->setFlattenUniformArrays((Options & EOptionFlattenUniformArrays) != 0);
        shader->setNoStorageFormat((Options & EOptionNoStorageFormat) != 0);
        shader->setResourceSetBinding(baseResourceSetBinding[compUnit.stage]);

        if (Options & EOptionHlslIoMapping)
            shader->setHlslIoMapping(true);

        if (Options & EOptionAutoMapBindings)
            shader->setAutoMapBindings(true);

        if (Options & EOptionAutoMapLocations)
            shader->setAutoMapLocations(true);

        if (Options & EOptionInvertY)
            shader->setInvertY(true);

        for (auto& uniOverride : uniformLocationOverrides) {
            shader->addUniformLocationOverride(uniOverride.first.c_str(),
                                               uniOverride.second);
        }

        shader->setUniformLocationBase(uniformBase);

        // Set up the environment, some subsettings take precedence over earlier
        // ways of setting things.
        if (Options & EOptionSpv) {
            shader->setEnvInput((Options & EOptionReadHlsl) ? glslang::EShSourceHlsl
                                                            : glslang::EShSourceGlsl,
                                compUnit.stage, Client, ClientInputSemanticsVersion);
            shader->setEnvClient(Client, ClientVersion);
            shader->setEnvTarget(TargetLanguage, TargetVersion);
            if (targetHlslFunctionality1)
                shader->setEnvTargetHlslFunctionality1();
        }

        shaders.push_back(shader);

        const int defaultVersion = Options & EOptionDefaultDesktop ? 110 : 100;

        DirStackFileIncluder includer;
        std::for_each(IncludeDirectoryList.rbegin(), IncludeDirectoryList.rend(), [&includer](const std::string& dir) {
            includer.pushExternalLocalDirectory(dir); });
        if (Options & EOptionOutputPreprocessed) {
            std::string str;
            if (shader->preprocess(&Resources, defaultVersion, ENoProfile, false, false, messages, &str, includer)) {
                PutsIfNonEmpty(str.c_str());
            } else {
                CompileFailed = true;
            }
            StderrIfNonEmpty(shader->getInfoLog());
            StderrIfNonEmpty(shader->getInfoDebugLog());
            continue;
        }

        if (! shader->parse(&Resources, defaultVersion, false, messages, includer))
            CompileFailed = true;

        program.addShader(shader);

        if (! (Options & EOptionSuppressInfolog) &&
            ! (Options & EOptionMemoryLeakMode)) {
            PutsIfNonEmpty(compUnit.fileName[0].c_str());
            PutsIfNonEmpty(shader->getInfoLog());
            PutsIfNonEmpty(shader->getInfoDebugLog());
        }
    }

    //
    // Program-level processing...
    //

    // Link
    if (! (Options & EOptionOutputPreprocessed) && ! program.link(messages))
        LinkFailed = true;

    // Map IO
    if (Options & EOptionSpv) {
        if (!program.mapIO())
            LinkFailed = true;
    }

    // Report
    if (! (Options & EOptionSuppressInfolog) &&
        ! (Options & EOptionMemoryLeakMode)) {
        PutsIfNonEmpty(program.getInfoLog());
        PutsIfNonEmpty(program.getInfoDebugLog());
    }

    // Reflect
    if (Options & EOptionDumpReflection) {
        program.buildReflection(ReflectOptions);
        program.dumpReflection();
    }

    // Dump SPIR-V
    if (Options & EOptionSpv) {
        if (CompileFailed || LinkFailed)
            printf("SPIR-V is not generated for failed compile or link\n");
        else {
            for (int stage = 0; stage < EShLangCount; ++stage) {
                if (program.getIntermediate((EShLanguage)stage)) {
                    std::vector<unsigned int> spirv;
                    std::string warningsErrors;
                    spv::SpvBuildLogger logger;
                    glslang::SpvOptions spvOptions;
                    if (Options & EOptionDebug)
                        spvOptions.generateDebugInfo = true;
                    spvOptions.disableOptimizer = (Options & EOptionOptimizeDisable) != 0;
                    spvOptions.optimizeSize = (Options & EOptionOptimizeSize) != 0;
                    spvOptions.disassemble = SpvToolsDisassembler;
                    spvOptions.validate = SpvToolsValidate;
                    glslang::GlslangToSpv(*program.getIntermediate((EShLanguage)stage), spirv, &logger, &spvOptions);

                    // Dump the spv to a file or stdout, etc., but only if not doing
                    // memory/perf testing, as it's not internal to programmatic use.
                    if (! (Options & EOptionMemoryLeakMode)) {
                        printf("%s", logger.getAllMessages().c_str());
                        if (Options & EOptionOutputHexadecimal) {
                            glslang::OutputSpvHex(spirv, GetBinaryName((EShLanguage)stage), variableName);
                        } else {
                            glslang::OutputSpvBin(spirv, GetBinaryName((EShLanguage)stage));
                        }
                        if (!SpvToolsDisassembler && (Options & EOptionHumanReadableSpv))
                            spv::Disassemble(std::cout, spirv);
                    }
                }
            }
        }
    }

    // Free everything up, program has to go before the shaders
    // because it might have merged stuff from the shaders, and
    // the stuff from the shaders has to have its destructors called
    // before the pools holding the memory in the shaders is freed.
    delete &program;
    while (shaders.size() > 0) {
        delete shaders.back();
        shaders.pop_back();
    }
}

//
// Do file IO part of compile and link, handing off the pure
// API/programmatic mode to CompileAndLinkShaderUnits(), which can
// be put in a loop for testing memory footprint and performance.
//
// This is just for linking mode: meaning all the shaders will be put into the
// the same program linked together.
//
// This means there are a limited number of work items (not multi-threading mode)
// and that the point is testing at the linking level. Hence, to enable
// performance and memory testing, the actual compile/link can be put in
// a loop, independent of processing the work items and file IO.
//
void CompileAndLinkShaderFiles(glslang::TWorklist& Worklist)
{
    std::vector<ShaderCompUnit> compUnits;

    // If this is using stdin, we can't really detect multiple different file
    // units by input type. We need to assume that we're just being given one
    // file of a certain type.
    if ((Options & EOptionStdin) != 0) {
        ShaderCompUnit compUnit(FindLanguage("stdin"));
        std::istreambuf_iterator<char> begin(std::cin), end;
        std::string tempString(begin, end);
        char* fileText = strdup(tempString.c_str());
        std::string fileName = "stdin";
        compUnit.addString(fileName, fileText);
        compUnits.push_back(compUnit);
    } else {
        // Transfer all the work items from to a simple list of
        // of compilation units.  (We don't care about the thread
        // work-item distribution properties in this path, which
        // is okay due to the limited number of shaders, know since
        // they are all getting linked together.)
        glslang::TWorkItem* workItem;
        while (Worklist.remove(workItem)) {
            ShaderCompUnit compUnit(FindLanguage(workItem->name));
            char* fileText = ReadFileData(workItem->name.c_str());
            if (fileText == nullptr)
                usage();
            compUnit.addString(workItem->name, fileText);
            compUnits.push_back(compUnit);
        }
    }

    // Actual call to programmatic processing of compile and link,
    // in a loop for testing memory and performance.  This part contains
    // all the perf/memory that a programmatic consumer will care about.
    for (int i = 0; i < ((Options & EOptionMemoryLeakMode) ? 100 : 1); ++i) {
        for (int j = 0; j < ((Options & EOptionMemoryLeakMode) ? 100 : 1); ++j)
           CompileAndLinkShaderUnits(compUnits);

        if (Options & EOptionMemoryLeakMode)
            glslang::OS_DumpMemoryCounters();
    }

    // free memory from ReadFileData, which got stored in a const char*
    // as the first string above
    for (auto it = compUnits.begin(); it != compUnits.end(); ++it)
        FreeFileData(const_cast<char*>(it->text[0]));
}

int singleMain()
{
    glslang::TWorklist workList;
    std::for_each(WorkItems.begin(), WorkItems.end(), [&workList](std::unique_ptr<glslang::TWorkItem>& item) {
        assert(item);
        workList.add(item.get());
    });

    if (Options & EOptionDumpConfig) {
        printf("%s", glslang::GetDefaultTBuiltInResourceString().c_str());
        if (workList.empty())
            return ESuccess;
    }

    if (Options & EOptionDumpBareVersion) {
        printf("%d.%d.%d\n",
            glslang::GetSpirvGeneratorVersion(), GLSLANG_MINOR_VERSION, GLSLANG_PATCH_LEVEL);
        if (workList.empty())
            return ESuccess;
    } else if (Options & EOptionDumpVersions) {
        printf("Glslang Version: %d.%d.%d\n",
            glslang::GetSpirvGeneratorVersion(), GLSLANG_MINOR_VERSION, GLSLANG_PATCH_LEVEL);
        printf("ESSL Version: %s\n", glslang::GetEsslVersionString());
        printf("GLSL Version: %s\n", glslang::GetGlslVersionString());
        std::string spirvVersion;
        glslang::GetSpirvVersion(spirvVersion);
        printf("SPIR-V Version %s\n", spirvVersion.c_str());
        printf("GLSL.std.450 Version %d, Revision %d\n", GLSLstd450Version, GLSLstd450Revision);
        printf("Khronos Tool ID %d\n", glslang::GetKhronosToolId());
        printf("SPIR-V Generator Version %d\n", glslang::GetSpirvGeneratorVersion());
        printf("GL_KHR_vulkan_glsl version %d\n", 100);
        printf("ARB_GL_gl_spirv version %d\n", 100);
        if (workList.empty())
            return ESuccess;
    }

    if (workList.empty() && ((Options & EOptionStdin) == 0)) {
        usage();
    }

    if (Options & EOptionStdin) {
        WorkItems.push_back(std::unique_ptr<glslang::TWorkItem>{new glslang::TWorkItem("stdin")});
        workList.add(WorkItems.back().get());
    }

    ProcessConfigFile();

    if ((Options & EOptionReadHlsl) && !((Options & EOptionOutputPreprocessed) || (Options & EOptionSpv)))
        Error("ERROR: HLSL requires SPIR-V code generation (or preprocessing only)");

    //
    // Two modes:
    // 1) linking all arguments together, single-threaded, new C++ interface
    // 2) independent arguments, can be tackled by multiple asynchronous threads, for testing thread safety, using the old handle interface
    //
    if (Options & (EOptionLinkProgram | EOptionOutputPreprocessed)) {
        glslang::InitializeProcess();
        glslang::InitializeProcess();  // also test reference counting of users
        glslang::InitializeProcess();  // also test reference counting of users
        glslang::FinalizeProcess();    // also test reference counting of users
        glslang::FinalizeProcess();    // also test reference counting of users
        CompileAndLinkShaderFiles(workList);
        glslang::FinalizeProcess();
    } else {
        ShInitialize();
        ShInitialize();  // also test reference counting of users
        ShFinalize();    // also test reference counting of users

        bool printShaderNames = workList.size() > 1;

        if (Options & EOptionMultiThreaded) {
            std::array<std::thread, 16> threads;
            for (unsigned int t = 0; t < threads.size(); ++t) {
                threads[t] = std::thread(CompileShaders, std::ref(workList));
                if (threads[t].get_id() == std::thread::id()) {
                    fprintf(stderr, "Failed to create thread\n");
                    return EFailThreadCreate;
                }
            }

            std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
        } else
            CompileShaders(workList);

        // Print out all the resulting infologs
        for (size_t w = 0; w < WorkItems.size(); ++w) {
            if (WorkItems[w]) {
                if (printShaderNames || WorkItems[w]->results.size() > 0)
                    PutsIfNonEmpty(WorkItems[w]->name.c_str());
                PutsIfNonEmpty(WorkItems[w]->results.c_str());
            }
        }

        ShFinalize();
    }

    if (CompileFailed)
        return EFailCompile;
    if (LinkFailed)
        return EFailLink;

    return 0;
}

int C_DECL main(int argc, char* argv[])
{
    ProcessArguments(WorkItems, argc, argv);

    int ret = 0;

    // Loop over the entire init/finalize cycle to watch memory changes
    const int iterations = 1;
    if (iterations > 1)
        glslang::OS_DumpMemoryCounters();
    for (int i = 0; i < iterations; ++i) {
        ret = singleMain();
        if (iterations > 1)
            glslang::OS_DumpMemoryCounters();
    }

    return ret;
}

//
//   Deduce the language from the filename.  Files must end in one of the
//   following extensions:
//
//   .vert = vertex
//   .tesc = tessellation control
//   .tese = tessellation evaluation
//   .geom = geometry
//   .frag = fragment
//   .comp = compute
//   .rgen = ray generation
//   .rint = ray intersection
//   .rahit = ray any hit
//   .rchit = ray closest hit
//   .rmiss = ray miss
//   .rcall = ray callable
//   .mesh  = mesh
//   .task  = task
//   Additionally, the file names may end in .<stage>.glsl and .<stage>.hlsl
//   where <stage> is one of the stages listed above.
//
EShLanguage FindLanguage(const std::string& name, bool parseStageName)
{
    std::string stageName;
    if (shaderStageName)
        stageName = shaderStageName;
    else if (parseStageName) {
        // Note: "first" extension means "first from the end", i.e.
        // if the file is named foo.vert.glsl, then "glsl" is first,
        // "vert" is second.
        size_t firstExtStart = name.find_last_of(".");
        bool hasFirstExt = firstExtStart != std::string::npos;
        size_t secondExtStart = hasFirstExt ? name.find_last_of(".", firstExtStart - 1) : std::string::npos;
        bool hasSecondExt = secondExtStart != std::string::npos;
        std::string firstExt = name.substr(firstExtStart + 1, std::string::npos);
        bool usesUnifiedExt = hasFirstExt && (firstExt == "glsl" || firstExt == "hlsl");
        if (usesUnifiedExt && firstExt == "hlsl")
            Options |= EOptionReadHlsl;
        if (hasFirstExt && !usesUnifiedExt)
            stageName = firstExt;
        else if (usesUnifiedExt && hasSecondExt)
            stageName = name.substr(secondExtStart + 1, firstExtStart - secondExtStart - 1);
        else {
            usage();
            return EShLangVertex;
        }
    } else
        stageName = name;

    if (stageName == "vert")
        return EShLangVertex;
    else if (stageName == "tesc")
        return EShLangTessControl;
    else if (stageName == "tese")
        return EShLangTessEvaluation;
    else if (stageName == "geom")
        return EShLangGeometry;
    else if (stageName == "frag")
        return EShLangFragment;
    else if (stageName == "comp")
        return EShLangCompute;
#ifdef NV_EXTENSIONS
    else if (stageName == "rgen")
        return EShLangRayGenNV;
    else if (stageName == "rint")
        return EShLangIntersectNV;
    else if (stageName == "rahit")
        return EShLangAnyHitNV;
    else if (stageName == "rchit")
        return EShLangClosestHitNV;
    else if (stageName == "rmiss")
        return EShLangMissNV;
    else if (stageName == "rcall")
        return EShLangCallableNV;
    else if (stageName == "mesh")
        return EShLangMeshNV;
    else if (stageName == "task")
        return EShLangTaskNV;
#endif

    usage();
    return EShLangVertex;
}

//
// Read a file's data into a string, and compile it using the old interface ShCompile,
// for non-linkable results.
//
void CompileFile(const char* fileName, ShHandle compiler)
{
    int ret = 0;
    char* shaderString;
    if ((Options & EOptionStdin) != 0) {
        std::istreambuf_iterator<char> begin(std::cin), end;
        std::string tempString(begin, end);
        shaderString = strdup(tempString.c_str());
    } else {
        shaderString = ReadFileData(fileName);
    }

    // move to length-based strings, rather than null-terminated strings
    int* lengths = new int[1];
    lengths[0] = (int)strlen(shaderString);

    EShMessages messages = EShMsgDefault;
    SetMessageOptions(messages);

    if (UserPreamble.isSet())
        Error("-D and -U options require -l (linking)\n");

    for (int i = 0; i < ((Options & EOptionMemoryLeakMode) ? 100 : 1); ++i) {
        for (int j = 0; j < ((Options & EOptionMemoryLeakMode) ? 100 : 1); ++j) {
            // ret = ShCompile(compiler, shaderStrings, NumShaderStrings, lengths, EShOptNone, &Resources, Options, (Options & EOptionDefaultDesktop) ? 110 : 100, false, messages);
            ret = ShCompile(compiler, &shaderString, 1, nullptr, EShOptNone, &Resources, Options, (Options & EOptionDefaultDesktop) ? 110 : 100, false, messages);
            // const char* multi[12] = { "# ve", "rsion", " 300 e", "s", "\n#err",
            //                         "or should be l", "ine 1", "string 5\n", "float glo", "bal",
            //                         ";\n#error should be line 2\n void main() {", "global = 2.3;}" };
            // const char* multi[7] = { "/", "/", "\\", "\n", "\n", "#", "version 300 es" };
            // ret = ShCompile(compiler, multi, 7, nullptr, EShOptNone, &Resources, Options, (Options & EOptionDefaultDesktop) ? 110 : 100, false, messages);
        }

        if (Options & EOptionMemoryLeakMode)
            glslang::OS_DumpMemoryCounters();
    }

    delete [] lengths;
    FreeFileData(shaderString);

    if (ret == 0)
        CompileFailed = true;
}

//
//   print usage to stdout
//
void usage()
{
    printf("Usage: glslangValidator [option]... [file]...\n"
           "\n"
           "'file' can end in .<stage> for auto-stage classification, where <stage> is:\n"
           "    .conf   to provide a config file that replaces the default configuration\n"
           "            (see -c option below for generating a template)\n"
           "    .vert   for a vertex shader\n"
           "    .tesc   for a tessellation control shader\n"
           "    .tese   for a tessellation evaluation shader\n"
           "    .geom   for a geometry shader\n"
           "    .frag   for a fragment shader\n"
           "    .comp   for a compute shader\n"
#ifdef NV_EXTENSIONS
           "    .mesh   for a mesh shader\n"
           "    .task   for a task shader\n"
           "    .rgen    for a ray generation shader\n"
           "    .rint    for a ray intersection shader\n"
           "    .rahit   for a ray any hit shader\n"
           "    .rchit   for a ray closest hit shader\n"
           "    .rmiss   for a ray miss shader\n"
           "    .rcall   for a ray callable shader\n"
#endif
           "    .glsl   for .vert.glsl, .tesc.glsl, ..., .comp.glsl compound suffixes\n"
           "    .hlsl   for .vert.hlsl, .tesc.hlsl, ..., .comp.hlsl compound suffixes\n"
           "\n"
           "Options:\n"
           "  -C          cascading errors; risk crash from accumulation of error recoveries\n"
           "  -D          input is HLSL (this is the default when any suffix is .hlsl)\n"
           "  -D<macro=def>\n"
           "  -D<macro>   define a pre-processor macro\n"
           "  -E          print pre-processed GLSL; cannot be used with -l;\n"
           "              errors will appear on stderr\n"
           "  -G[ver]     create SPIR-V binary, under OpenGL semantics; turns on -l;\n"
           "              default file name is <stage>.spv (-o overrides this);\n"
           "              'ver', when present, is the version of the input semantics,\n"
           "              which will appear in #define GL_SPIRV ver;\n"
           "              '--client opengl100' is the same as -G100;\n"
           "              a '--target-env' for OpenGL will also imply '-G'\n"
           "  -H          print human readable form of SPIR-V; turns on -V\n"
           "  -I<dir>     add dir to the include search path; includer's directory\n"
           "              is searched first, followed by left-to-right order of -I\n"
           "  -Od         disables optimization; may cause illegal SPIR-V for HLSL\n"
           "  -Os         optimizes SPIR-V to minimize size\n"
           "  -S <stage>  uses specified stage rather than parsing the file extension\n"
           "              choices for <stage> are vert, tesc, tese, geom, frag, or comp\n"
           "  -U<macro>   undefine a pre-processor macro\n"
           "  -V[ver]     create SPIR-V binary, under Vulkan semantics; turns on -l;\n"
           "              default file name is <stage>.spv (-o overrides this)\n"
           "              'ver', when present, is the version of the input semantics,\n"
           "              which will appear in #define VULKAN ver\n"
           "              '--client vulkan100' is the same as -V100\n"
           "              a '--target-env' for Vulkan will also imply '-V'\n"
           "  -c          configuration dump;\n"
           "              creates the default configuration file (redirect to a .conf file)\n"
           "  -d          default to desktop (#version 110) when there is no shader #version\n"
           "              (default is ES version 100)\n"
           "  -e <name> | --entry-point <name>\n"
           "              specify <name> as the entry-point function name\n"
           "  -f{hlsl_functionality1}\n"
           "              'hlsl_functionality1' enables use of the\n"
           "              SPV_GOOGLE_hlsl_functionality1 extension\n"
           "  -g          generate debug information\n"
           "  -h          print this usage message\n"
           "  -i          intermediate tree (glslang AST) is printed out\n"
           "  -l          link all input files together to form a single module\n"
           "  -m          memory leak mode\n"
           "  -o <file>   save binary to <file>, requires a binary option (e.g., -V)\n"
           "  -q          dump reflection query database\n"
           "  -r | --relaxed-errors"
           "              relaxed GLSL semantic error-checking mode\n"
           "  -s          silence syntax and semantic error reporting\n"
           "  -t          multi-threaded mode\n"
           "  -v | --version\n"
           "              print version strings\n"
           "  -w | --suppress-warnings\n"
           "              suppress GLSL warnings, except as required by \"#extension : warn\"\n"
           "  -x          save binary output as text-based 32-bit hexadecimal numbers\n"
           "  -u<name>:<loc> specify a uniform location override for --aml\n"
           "  --uniform-base <base> set a base to use for generated uniform locations\n"
           "  --auto-map-bindings | --amb       automatically bind uniform variables\n"
           "                                    without explicit bindings\n"
           "  --auto-map-locations | --aml      automatically locate input/output lacking\n"
           "                                    'location' (fragile, not cross stage)\n"
           "  --client {vulkan<ver>|opengl<ver>} see -V and -G\n"
           "  -dumpfullversion | -dumpversion   print bare major.minor.patchlevel\n"
           "  --flatten-uniform-arrays | --fua  flatten uniform texture/sampler arrays to\n"
           "                                    scalars\n"
           "  --hlsl-offsets                    allow block offsets to follow HLSL rules\n"
           "                                    works independently of source language\n"
           "  --hlsl-iomap                      perform IO mapping in HLSL register space\n"
           "  --hlsl-enable-16bit-types         allow 16-bit types in SPIR-V for HLSL\n"
           "  --hlsl-dx9-compatible             interprets sampler declarations as a texture/sampler combo like DirectX9 would."
           "  --invert-y | --iy                 invert position.Y output in vertex shader\n"
           "  --keep-uncalled | --ku            don't eliminate uncalled functions\n"
           "  --no-storage-format | --nsf       use Unknown image format\n"
           "  --reflect-strict-array-suffix     use strict array suffix rules when\n"
           "                                    reflecting\n"
           "  --reflect-basic-array-suffix      arrays of basic types will have trailing [0]\n"
           "  --reflect-intermediate-io         reflection includes inputs/outputs of linked\n"
           "                                    shaders rather than just vertex/fragment\n"
           "  --reflect-separate-buffers        reflect buffer variables and blocks\n"
           "                                    separately to uniforms\n"
           "  --reflect-all-block-variables     reflect all variables in blocks, whether\n"
           "                                    inactive or active\n"
           "  --reflect-unwrap-io-blocks        unwrap input/output blocks the same as\n"
           "                                    uniform blocks\n"
           "  --resource-set-binding [stage] name set binding\n"
           "                                    set descriptor set and binding for\n"
           "                                    individual resources\n"
           "  --resource-set-binding [stage] set\n"
           "                                    set descriptor set for all resources\n"
           "  --rsb                             synonym for --resource-set-binding\n"
           "  --shift-image-binding [stage] num\n"
           "                                    base binding number for images (uav)\n"
           "  --shift-image-binding [stage] [num set]...\n"
           "                                    per-descriptor-set shift values\n"
           "  --sib                             synonym for --shift-image-binding\n"
           "  --shift-sampler-binding [stage] num\n"
           "                                    base binding number for samplers\n"
           "  --shift-sampler-binding [stage] [num set]...\n"
           "                                    per-descriptor-set shift values\n"
           "  --ssb                             synonym for --shift-sampler-binding\n"
           "  --shift-ssbo-binding [stage] num  base binding number for SSBOs\n"
           "  --shift-ssbo-binding [stage] [num set]...\n"
           "                                    per-descriptor-set shift values\n"
           "  --sbb                             synonym for --shift-ssbo-binding\n"
           "  --shift-texture-binding [stage] num\n"
           "                                    base binding number for textures\n"
           "  --shift-texture-binding [stage] [num set]...\n"
           "                                    per-descriptor-set shift values\n"
           "  --stb                             synonym for --shift-texture-binding\n"
           "  --shift-uav-binding [stage] num   base binding number for UAVs\n"
           "  --shift-uav-binding [stage] [num set]...\n"
           "                                    per-descriptor-set shift values\n"
           "  --suavb                           synonym for --shift-uav-binding\n"
           "  --shift-UBO-binding [stage] num   base binding number for UBOs\n"
           "  --shift-UBO-binding [stage] [num set]...\n"
           "                                    per-descriptor-set shift values\n"
           "  --sub                             synonym for --shift-UBO-binding\n"
           "  --shift-cbuffer-binding | --scb   synonyms for --shift-UBO-binding\n"
           "  --spirv-dis                       output standard-form disassembly; works only\n"
           "                                    when a SPIR-V generation option is also used\n"
           "  --spirv-val                       execute the SPIRV-Tools validator\n"
           "  --source-entrypoint <name>        the given shader source function is\n"
           "                                    renamed to be the <name> given in -e\n"
           "  --sep                             synonym for --source-entrypoint\n"
           "  --stdin                           read from stdin instead of from a file;\n"
           "                                    requires providing the shader stage using -S\n"
           "  --target-env {vulkan1.0 | vulkan1.1 | opengl | \n"
           "                spirv1.0 | spirv1.1 | spirv1.2 | spirv1.3}\n"
           "                                    set execution environment that emitted code\n"
           "                                    will execute in (versus source language\n"
           "                                    semantics selected by --client) defaults:\n"
           "                                     * 'vulkan1.0' under '--client vulkan<ver>'\n"
           "                                     * 'opengl' under '--client opengl<ver>'\n"
           "                                     * 'spirv1.0' under --target-env vulkan1.0\n"
           "                                     * 'spirv1.3' under --target-env vulkan1.1\n"
           "                                    multiple --targen-env can be specified.\n"
           "  --variable-name <name>\n"
           "  --vn <name>                       creates a C header file that contains a\n"
           "                                    uint32_t array named <name>\n"
           "                                    initialized with the shader binary code\n"
           );

    exit(EFailUsage);
}

#if !defined _MSC_VER && !defined MINGW_HAS_SECURE_API

#include <errno.h>

int fopen_s(
   FILE** pFile,
   const char* filename,
   const char* mode
)
{
   if (!pFile || !filename || !mode) {
      return EINVAL;
   }

   FILE* f = fopen(filename, mode);
   if (! f) {
      if (errno != 0) {
         return errno;
      } else {
         return ENOENT;
      }
   }
   *pFile = f;

   return 0;
}

#endif

//
//   Malloc a string of sufficient size and read a string into it.
//
char* ReadFileData(const char* fileName)
{
    FILE *in = nullptr;
    int errorCode = fopen_s(&in, fileName, "r");
    if (errorCode || in == nullptr)
        Error("unable to open input file");

    int count = 0;
    while (fgetc(in) != EOF)
        count++;

    fseek(in, 0, SEEK_SET);

    char* return_data = (char*)malloc(count + 1);  // freed in FreeFileData()
    if ((int)fread(return_data, 1, count, in) != count) {
        free(return_data);
        Error("can't read input file");
    }

    return_data[count] = '\0';
    fclose(in);

    return return_data;
}

void FreeFileData(char* data)
{
    free(data);
}

void InfoLogMsg(const char* msg, const char* name, const int num)
{
    if (num >= 0 )
        printf("#### %s %s %d INFO LOG ####\n", msg, name, num);
    else
        printf("#### %s %s INFO LOG ####\n", msg, name);
}
