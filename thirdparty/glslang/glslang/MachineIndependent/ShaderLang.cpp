//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013-2016 LunarG, Inc.
// Copyright (C) 2015-2020 Google, Inc.
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

//
// Implement the top-level of interface to the compiler/linker,
// as defined in ShaderLang.h
// This is the platform independent interface between an OGL driver
// and the shading language compiler/linker.
//
#include <cstring>
#include <iostream>
#include <sstream>
#include <memory>
#include <mutex>
#include "SymbolTable.h"
#include "ParseHelper.h"
#include "Scan.h"
#include "ScanContext.h"

#ifdef ENABLE_HLSL
#include "../HLSL/hlslParseHelper.h"
#include "../HLSL/hlslParseables.h"
#include "../HLSL/hlslScanContext.h"
#endif

#include "../Include/ShHandle.h"

#include "preprocessor/PpContext.h"

#define SH_EXPORTING
#include "../Public/ShaderLang.h"
#include "reflection.h"
#include "iomapper.h"
#include "Initialize.h"

// TODO: this really shouldn't be here, it is only because of the trial addition
// of printing pre-processed tokens, which requires knowing the string literal
// token to print ", but none of that seems appropriate for this file.
#include "preprocessor/PpTokens.h"

// Build-time generated includes
#include "glslang/build_info.h"

namespace { // anonymous namespace for file-local functions and symbols

// Total number of successful initializers of glslang: a refcount
// Shared global; access should be protected by a global mutex/critical section.
int NumberOfClients = 0;

// global initialization lock
std::mutex init_lock;

using namespace glslang;

// Create a language specific version of parseables.
TBuiltInParseables* CreateBuiltInParseables(TInfoSink& infoSink, EShSource source)
{
    switch (source) {
    case EShSourceGlsl: return new TBuiltIns();              // GLSL builtIns
#ifdef ENABLE_HLSL
    case EShSourceHlsl: return new TBuiltInParseablesHlsl(); // HLSL intrinsics
#endif

    default:
        infoSink.info.message(EPrefixInternalError, "Unable to determine source language");
        return nullptr;
    }
}

// Create a language specific version of a parse context.
TParseContextBase* CreateParseContext(TSymbolTable& symbolTable, TIntermediate& intermediate,
                                      int version, EProfile profile, EShSource source,
                                      EShLanguage language, TInfoSink& infoSink,
                                      SpvVersion spvVersion, bool forwardCompatible, EShMessages messages,
                                      bool parsingBuiltIns, std::string sourceEntryPointName = "")
{
    switch (source) {
    case EShSourceGlsl: {
        if (sourceEntryPointName.size() == 0)
            intermediate.setEntryPointName("main");
        TString entryPoint = sourceEntryPointName.c_str();
        return new TParseContext(symbolTable, intermediate, parsingBuiltIns, version, profile, spvVersion,
                                 language, infoSink, forwardCompatible, messages, &entryPoint);
    }
#ifdef ENABLE_HLSL
    case EShSourceHlsl:
        return new HlslParseContext(symbolTable, intermediate, parsingBuiltIns, version, profile, spvVersion,
                                    language, infoSink, sourceEntryPointName.c_str(), forwardCompatible, messages);
#endif
    default:
        infoSink.info.message(EPrefixInternalError, "Unable to determine source language");
        return nullptr;
    }
}

// Local mapping functions for making arrays of symbol tables....

const int VersionCount = 17;  // index range in MapVersionToIndex

int MapVersionToIndex(int version)
{
    int index = 0;

    switch (version) {
    case 100: index =  0; break;
    case 110: index =  1; break;
    case 120: index =  2; break;
    case 130: index =  3; break;
    case 140: index =  4; break;
    case 150: index =  5; break;
    case 300: index =  6; break;
    case 330: index =  7; break;
    case 400: index =  8; break;
    case 410: index =  9; break;
    case 420: index = 10; break;
    case 430: index = 11; break;
    case 440: index = 12; break;
    case 310: index = 13; break;
    case 450: index = 14; break;
    case 500: index =  0; break; // HLSL
    case 320: index = 15; break;
    case 460: index = 16; break;
    default:  assert(0);  break;
    }

    assert(index < VersionCount);

    return index;
}

const int SpvVersionCount = 4;  // index range in MapSpvVersionToIndex

int MapSpvVersionToIndex(const SpvVersion& spvVersion)
{
    int index = 0;

    if (spvVersion.openGl > 0)
        index = 1;
    else if (spvVersion.vulkan > 0) {
        if (!spvVersion.vulkanRelaxed)
            index = 2;
        else
            index = 3;
    }

    assert(index < SpvVersionCount);

    return index;
}

const int ProfileCount = 4;   // index range in MapProfileToIndex

int MapProfileToIndex(EProfile profile)
{
    int index = 0;

    switch (profile) {
    case ENoProfile:            index = 0; break;
    case ECoreProfile:          index = 1; break;
    case ECompatibilityProfile: index = 2; break;
    case EEsProfile:            index = 3; break;
    default:                               break;
    }

    assert(index < ProfileCount);

    return index;
}

const int SourceCount = 2;

int MapSourceToIndex(EShSource source)
{
    int index = 0;

    switch (source) {
    case EShSourceGlsl: index = 0; break;
    case EShSourceHlsl: index = 1; break;
    default:                       break;
    }

    assert(index < SourceCount);

    return index;
}

// only one of these needed for non-ES; ES needs 2 for different precision defaults of built-ins
enum EPrecisionClass {
    EPcGeneral,
    EPcFragment,
    EPcCount
};

// A process-global symbol table per version per profile for built-ins common
// to multiple stages (languages), and a process-global symbol table per version
// per profile per stage for built-ins unique to each stage.  They will be sparsely
// populated, so they will only be generated as needed.
//
// Each has a different set of built-ins, and we want to preserve that from
// compile to compile.
//
TSymbolTable* CommonSymbolTable[VersionCount][SpvVersionCount][ProfileCount][SourceCount][EPcCount] = {};
TSymbolTable* SharedSymbolTables[VersionCount][SpvVersionCount][ProfileCount][SourceCount][EShLangCount] = {};

TPoolAllocator* PerProcessGPA = nullptr;

//
// Parse and add to the given symbol table the content of the given shader string.
//
bool InitializeSymbolTable(const TString& builtIns, int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language,
                           EShSource source, TInfoSink& infoSink, TSymbolTable& symbolTable)
{
    TIntermediate intermediate(language, version, profile);

    intermediate.setSource(source);

    std::unique_ptr<TParseContextBase> parseContext(CreateParseContext(symbolTable, intermediate, version, profile, source,
                                                                       language, infoSink, spvVersion, true, EShMsgDefault,
                                                                       true));

    TShader::ForbidIncluder includer;
    TPpContext ppContext(*parseContext, "", includer);
    TScanContext scanContext(*parseContext);
    parseContext->setScanContext(&scanContext);
    parseContext->setPpContext(&ppContext);

    //
    // Push the symbol table to give it an initial scope.  This
    // push should not have a corresponding pop, so that built-ins
    // are preserved, and the test for an empty table fails.
    //

    symbolTable.push();

    const char* builtInShaders[2];
    size_t builtInLengths[2];
    builtInShaders[0] = builtIns.c_str();
    builtInLengths[0] = builtIns.size();

    if (builtInLengths[0] == 0)
        return true;

    TInputScanner input(1, builtInShaders, builtInLengths);
    if (! parseContext->parseShaderStrings(ppContext, input) != 0) {
        infoSink.info.message(EPrefixInternalError, "Unable to parse built-ins");
        printf("Unable to parse built-ins\n%s\n", infoSink.info.c_str());
        printf("%s\n", builtInShaders[0]);

        return false;
    }

    return true;
}

int CommonIndex(EProfile profile, EShLanguage language)
{
    return (profile == EEsProfile && language == EShLangFragment) ? EPcFragment : EPcGeneral;
}

//
// To initialize per-stage shared tables, with the common table already complete.
//
void InitializeStageSymbolTable(TBuiltInParseables& builtInParseables, int version, EProfile profile, const SpvVersion& spvVersion,
                                EShLanguage language, EShSource source, TInfoSink& infoSink, TSymbolTable** commonTable,
                                TSymbolTable** symbolTables)
{
    (*symbolTables[language]).adoptLevels(*commonTable[CommonIndex(profile, language)]);
    InitializeSymbolTable(builtInParseables.getStageString(language), version, profile, spvVersion, language, source,
                          infoSink, *symbolTables[language]);
    builtInParseables.identifyBuiltIns(version, profile, spvVersion, language, *symbolTables[language]);
    if (profile == EEsProfile && version >= 300)
        (*symbolTables[language]).setNoBuiltInRedeclarations();
    if (version == 110)
        (*symbolTables[language]).setSeparateNameSpaces();
}

//
// Initialize the full set of shareable symbol tables;
// The common (cross-stage) and those shareable per-stage.
//
bool InitializeSymbolTables(TInfoSink& infoSink, TSymbolTable** commonTable,  TSymbolTable** symbolTables, int version, EProfile profile, const SpvVersion& spvVersion, EShSource source)
{
    std::unique_ptr<TBuiltInParseables> builtInParseables(CreateBuiltInParseables(infoSink, source));

    if (builtInParseables == nullptr)
        return false;

    builtInParseables->initialize(version, profile, spvVersion);

    // do the common tables
    InitializeSymbolTable(builtInParseables->getCommonString(), version, profile, spvVersion, EShLangVertex, source,
                          infoSink, *commonTable[EPcGeneral]);
    if (profile == EEsProfile)
        InitializeSymbolTable(builtInParseables->getCommonString(), version, profile, spvVersion, EShLangFragment, source,
                              infoSink, *commonTable[EPcFragment]);

    // do the per-stage tables

    // always have vertex and fragment
    InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangVertex, source,
                               infoSink, commonTable, symbolTables);
    InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangFragment, source,
                               infoSink, commonTable, symbolTables);

    // check for tessellation
    if ((profile != EEsProfile && version >= 150) ||
        (profile == EEsProfile && version >= 310)) {
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangTessControl, source,
                                   infoSink, commonTable, symbolTables);
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangTessEvaluation, source,
                                   infoSink, commonTable, symbolTables);
    }

    // check for geometry
    if ((profile != EEsProfile && version >= 150) ||
        (profile == EEsProfile && version >= 310))
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangGeometry, source,
                                   infoSink, commonTable, symbolTables);

    // check for compute
    if ((profile != EEsProfile && version >= 420) ||
        (profile == EEsProfile && version >= 310))
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangCompute, source,
                                   infoSink, commonTable, symbolTables);

    // check for ray tracing stages
    if (profile != EEsProfile && version >= 450) {
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangRayGen, source,
            infoSink, commonTable, symbolTables);
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangIntersect, source,
            infoSink, commonTable, symbolTables);
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangAnyHit, source,
            infoSink, commonTable, symbolTables);
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangClosestHit, source,
            infoSink, commonTable, symbolTables);
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangMiss, source,
            infoSink, commonTable, symbolTables);
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangCallable, source,
            infoSink, commonTable, symbolTables);
    }

    // check for mesh
    if ((profile != EEsProfile && version >= 450) ||
        (profile == EEsProfile && version >= 320))
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangMesh, source,
                                   infoSink, commonTable, symbolTables);

    // check for task
    if ((profile != EEsProfile && version >= 450) ||
        (profile == EEsProfile && version >= 320))
        InitializeStageSymbolTable(*builtInParseables, version, profile, spvVersion, EShLangTask, source,
                                   infoSink, commonTable, symbolTables);

    return true;
}

bool AddContextSpecificSymbols(const TBuiltInResource* resources, TInfoSink& infoSink, TSymbolTable& symbolTable, int version,
                               EProfile profile, const SpvVersion& spvVersion, EShLanguage language, EShSource source)
{
    std::unique_ptr<TBuiltInParseables> builtInParseables(CreateBuiltInParseables(infoSink, source));

    if (builtInParseables == nullptr)
        return false;

    builtInParseables->initialize(*resources, version, profile, spvVersion, language);
    InitializeSymbolTable(builtInParseables->getCommonString(), version, profile, spvVersion, language, source, infoSink, symbolTable);
    builtInParseables->identifyBuiltIns(version, profile, spvVersion, language, symbolTable, *resources);

    return true;
}

//
// To do this on the fly, we want to leave the current state of our thread's
// pool allocator intact, so:
//  - Switch to a new pool for parsing the built-ins
//  - Do the parsing, which builds the symbol table, using the new pool
//  - Switch to the process-global pool to save a copy of the resulting symbol table
//  - Free up the new pool used to parse the built-ins
//  - Switch back to the original thread's pool
//
// This only gets done the first time any thread needs a particular symbol table
// (lazy evaluation).
//
void SetupBuiltinSymbolTable(int version, EProfile profile, const SpvVersion& spvVersion, EShSource source)
{
    TInfoSink infoSink;

    // Make sure only one thread tries to do this at a time
    const std::lock_guard<std::mutex> lock(init_lock);

    // See if it's already been done for this version/profile combination
    int versionIndex = MapVersionToIndex(version);
    int spvVersionIndex = MapSpvVersionToIndex(spvVersion);
    int profileIndex = MapProfileToIndex(profile);
    int sourceIndex = MapSourceToIndex(source);
    if (CommonSymbolTable[versionIndex][spvVersionIndex][profileIndex][sourceIndex][EPcGeneral])
        return;

    // Switch to a new pool
    TPoolAllocator& previousAllocator = GetThreadPoolAllocator();
    TPoolAllocator* builtInPoolAllocator = new TPoolAllocator;
    SetThreadPoolAllocator(builtInPoolAllocator);

    // Dynamically allocate the local symbol tables so we can control when they are deallocated WRT when the pool is popped.
    TSymbolTable* commonTable[EPcCount];
    TSymbolTable* stageTables[EShLangCount];
    for (int precClass = 0; precClass < EPcCount; ++precClass)
        commonTable[precClass] = new TSymbolTable;
    for (int stage = 0; stage < EShLangCount; ++stage)
        stageTables[stage] = new TSymbolTable;

    // Generate the local symbol tables using the new pool
    InitializeSymbolTables(infoSink, commonTable, stageTables, version, profile, spvVersion, source);

    // Switch to the process-global pool
    SetThreadPoolAllocator(PerProcessGPA);

    // Copy the local symbol tables from the new pool to the global tables using the process-global pool
    for (int precClass = 0; precClass < EPcCount; ++precClass) {
        if (! commonTable[precClass]->isEmpty()) {
            CommonSymbolTable[versionIndex][spvVersionIndex][profileIndex][sourceIndex][precClass] = new TSymbolTable;
            CommonSymbolTable[versionIndex][spvVersionIndex][profileIndex][sourceIndex][precClass]->copyTable(*commonTable[precClass]);
            CommonSymbolTable[versionIndex][spvVersionIndex][profileIndex][sourceIndex][precClass]->readOnly();
        }
    }
    for (int stage = 0; stage < EShLangCount; ++stage) {
        if (! stageTables[stage]->isEmpty()) {
            SharedSymbolTables[versionIndex][spvVersionIndex][profileIndex][sourceIndex][stage] = new TSymbolTable;
            SharedSymbolTables[versionIndex][spvVersionIndex][profileIndex][sourceIndex][stage]->adoptLevels(*CommonSymbolTable
                              [versionIndex][spvVersionIndex][profileIndex][sourceIndex][CommonIndex(profile, (EShLanguage)stage)]);
            SharedSymbolTables[versionIndex][spvVersionIndex][profileIndex][sourceIndex][stage]->copyTable(*stageTables[stage]);
            SharedSymbolTables[versionIndex][spvVersionIndex][profileIndex][sourceIndex][stage]->readOnly();
        }
    }

    // Clean up the local tables before deleting the pool they used.
    for (int precClass = 0; precClass < EPcCount; ++precClass)
        delete commonTable[precClass];
    for (int stage = 0; stage < EShLangCount; ++stage)
        delete stageTables[stage];

    delete builtInPoolAllocator;
    SetThreadPoolAllocator(&previousAllocator);
}

// Function to Print all builtins
void DumpBuiltinSymbolTable(TInfoSink& infoSink, const TSymbolTable& symbolTable)
{
    infoSink.debug << "BuiltinSymbolTable {\n";

    symbolTable.dump(infoSink, true);

    infoSink.debug << "}\n";
}

// Return true if the shader was correctly specified for version/profile/stage.
bool DeduceVersionProfile(TInfoSink& infoSink, EShLanguage stage, bool versionNotFirst, int defaultVersion,
                          EShSource source, int& version, EProfile& profile, const SpvVersion& spvVersion)
{
    const int FirstProfileVersion = 150;
    bool correct = true;

    if (source == EShSourceHlsl) {
        version = 500;          // shader model; currently a characteristic of glslang, not the input
        profile = ECoreProfile; // allow doubles in prototype parsing
        return correct;
    }

    // Get a version...
    if (version == 0) {
        version = defaultVersion;
        // infoSink.info.message(EPrefixWarning, "#version: statement missing; use #version on first line of shader");
    }

    // Get a good profile...
    if (profile == ENoProfile) {
        if (version == 300 || version == 310 || version == 320) {
            correct = false;
            infoSink.info.message(EPrefixError, "#version: versions 300, 310, and 320 require specifying the 'es' profile");
            profile = EEsProfile;
        } else if (version == 100)
            profile = EEsProfile;
        else if (version >= FirstProfileVersion)
            profile = ECoreProfile;
        else
            profile = ENoProfile;
    } else {
        // a profile was provided...
        if (version < 150) {
            correct = false;
            infoSink.info.message(EPrefixError, "#version: versions before 150 do not allow a profile token");
            if (version == 100)
                profile = EEsProfile;
            else
                profile = ENoProfile;
        } else if (version == 300 || version == 310 || version == 320) {
            if (profile != EEsProfile) {
                correct = false;
                infoSink.info.message(EPrefixError, "#version: versions 300, 310, and 320 support only the es profile");
            }
            profile = EEsProfile;
        } else {
            if (profile == EEsProfile) {
                correct = false;
                infoSink.info.message(EPrefixError, "#version: only version 300, 310, and 320 support the es profile");
                if (version >= FirstProfileVersion)
                    profile = ECoreProfile;
                else
                    profile = ENoProfile;
            }
            // else: typical desktop case... e.g., "#version 410 core"
        }
    }

    // Fix version...
    switch (version) {
    // ES versions
    case 100: break;
    case 300: break;
    case 310: break;
    case 320: break;

    // desktop versions
    case 110: break;
    case 120: break;
    case 130: break;
    case 140: break;
    case 150: break;
    case 330: break;
    case 400: break;
    case 410: break;
    case 420: break;
    case 430: break;
    case 440: break;
    case 450: break;
    case 460: break;

    // unknown version
    default:
        correct = false;
        infoSink.info.message(EPrefixError, "version not supported");
        if (profile == EEsProfile)
            version = 310;
        else {
            version = 450;
            profile = ECoreProfile;
        }
        break;
    }

    // Correct for stage type...
    switch (stage) {
    case EShLangGeometry:
        if ((profile == EEsProfile && version < 310) ||
            (profile != EEsProfile && version < 150)) {
            correct = false;
            infoSink.info.message(EPrefixError, "#version: geometry shaders require es profile with version 310 or non-es profile with version 150 or above");
            version = (profile == EEsProfile) ? 310 : 150;
            if (profile == EEsProfile || profile == ENoProfile)
                profile = ECoreProfile;
        }
        break;
    case EShLangTessControl:
    case EShLangTessEvaluation:
        if ((profile == EEsProfile && version < 310) ||
            (profile != EEsProfile && version < 150)) {
            correct = false;
            infoSink.info.message(EPrefixError, "#version: tessellation shaders require es profile with version 310 or non-es profile with version 150 or above");
            version = (profile == EEsProfile) ? 310 : 400; // 150 supports the extension, correction is to 400 which does not
            if (profile == EEsProfile || profile == ENoProfile)
                profile = ECoreProfile;
        }
        break;
    case EShLangCompute:
        if ((profile == EEsProfile && version < 310) ||
            (profile != EEsProfile && version < 420)) {
            correct = false;
            infoSink.info.message(EPrefixError, "#version: compute shaders require es profile with version 310 or above, or non-es profile with version 420 or above");
            version = profile == EEsProfile ? 310 : 420;
        }
        break;
    case EShLangRayGen:
    case EShLangIntersect:
    case EShLangAnyHit:
    case EShLangClosestHit:
    case EShLangMiss:
    case EShLangCallable:
        if (profile == EEsProfile || version < 460) {
            correct = false;
            infoSink.info.message(EPrefixError, "#version: ray tracing shaders require non-es profile with version 460 or above");
            version = 460;
        }
        break;
    case EShLangMesh:
    case EShLangTask:
        if ((profile == EEsProfile && version < 320) ||
            (profile != EEsProfile && version < 450)) {
            correct = false;
            infoSink.info.message(EPrefixError, "#version: mesh/task shaders require es profile with version 320 or above, or non-es profile with version 450 or above");
            version = profile == EEsProfile ? 320 : 450;
        }
        break;
    default:
        break;
    }

    if (profile == EEsProfile && version >= 300 && versionNotFirst) {
        correct = false;
        infoSink.info.message(EPrefixError, "#version: statement must appear first in es-profile shader; before comments or newlines");
    }

    // Check for SPIR-V compatibility
    if (spvVersion.spv != 0) {
        switch (profile) {
        case EEsProfile:
            if (version < 310) {
                correct = false;
                infoSink.info.message(EPrefixError, "#version: ES shaders for SPIR-V require version 310 or higher");
                version = 310;
            }
            break;
        case ECompatibilityProfile:
            infoSink.info.message(EPrefixError, "#version: compilation for SPIR-V does not support the compatibility profile");
            break;
        default:
            if (spvVersion.vulkan > 0 && version < 140) {
                correct = false;
                infoSink.info.message(EPrefixError, "#version: Desktop shaders for Vulkan SPIR-V require version 140 or higher");
                version = 140;
            }
            if (spvVersion.openGl >= 100 && version < 330) {
                correct = false;
                infoSink.info.message(EPrefixError, "#version: Desktop shaders for OpenGL SPIR-V require version 330 or higher");
                version = 330;
            }
            break;
        }
    }

    return correct;
}

// There are multiple paths in for setting environment stuff.
// TEnvironment takes precedence, for what it sets, so sort all this out.
// Ideally, the internal code could be made to use TEnvironment, but for
// now, translate it to the historically used parameters.
void TranslateEnvironment(const TEnvironment* environment, EShMessages& messages, EShSource& source,
                          EShLanguage& stage, SpvVersion& spvVersion)
{
    // Set up environmental defaults, first ignoring 'environment'.
    if (messages & EShMsgSpvRules)
        spvVersion.spv = EShTargetSpv_1_0;
    if (messages & EShMsgVulkanRules) {
        spvVersion.vulkan = EShTargetVulkan_1_0;
        spvVersion.vulkanGlsl = 100;
    } else if (spvVersion.spv != 0)
        spvVersion.openGl = 100;

    // Now, override, based on any content set in 'environment'.
    // 'environment' must be cleared to ESh*None settings when items
    // are not being set.
    if (environment != nullptr) {
        // input language
        if (environment->input.languageFamily != EShSourceNone) {
            stage = environment->input.stage;
            switch (environment->input.dialect) {
            case EShClientNone:
                break;
            case EShClientVulkan:
                spvVersion.vulkanGlsl = environment->input.dialectVersion;
                spvVersion.vulkanRelaxed = environment->input.vulkanRulesRelaxed;
                break;
            case EShClientOpenGL:
                spvVersion.openGl = environment->input.dialectVersion;
                break;
            case EShClientCount:
                assert(0);
                break;
            }
            switch (environment->input.languageFamily) {
            case EShSourceNone:
                break;
            case EShSourceGlsl:
                source = EShSourceGlsl;
                messages = static_cast<EShMessages>(messages & ~EShMsgReadHlsl);
                break;
            case EShSourceHlsl:
                source = EShSourceHlsl;
                messages = static_cast<EShMessages>(messages | EShMsgReadHlsl);
                break;
            case EShSourceCount:
                assert(0);
                break;
            }
        }

        // client
        switch (environment->client.client) {
        case EShClientVulkan:
            spvVersion.vulkan = environment->client.version;
            break;
        default:
            break;
        }

        // generated code
        switch (environment->target.language) {
        case EshTargetSpv:
            spvVersion.spv = environment->target.version;
            break;
        default:
            break;
        }
    }
}

// Most processes are recorded when set in the intermediate representation,
// These are the few that are not.
void RecordProcesses(TIntermediate& intermediate, EShMessages messages, const std::string& sourceEntryPointName)
{
    if ((messages & EShMsgRelaxedErrors) != 0)
        intermediate.addProcess("relaxed-errors");
    if ((messages & EShMsgSuppressWarnings) != 0)
        intermediate.addProcess("suppress-warnings");
    if ((messages & EShMsgKeepUncalled) != 0)
        intermediate.addProcess("keep-uncalled");
    if (sourceEntryPointName.size() > 0) {
        intermediate.addProcess("source-entrypoint");
        intermediate.addProcessArgument(sourceEntryPointName);
    }
}

// This is the common setup and cleanup code for PreprocessDeferred and
// CompileDeferred.
// It takes any callable with a signature of
//  bool (TParseContextBase& parseContext, TPpContext& ppContext,
//                  TInputScanner& input, bool versionWillBeError,
//                  TSymbolTable& , TIntermediate& ,
//                  EShOptimizationLevel , EShMessages );
// Which returns false if a failure was detected and true otherwise.
//
template<typename ProcessingContext>
bool ProcessDeferred(
    TCompiler* compiler,
    const char* const shaderStrings[],
    const int numStrings,
    const int* inputLengths,
    const char* const stringNames[],
    const char* customPreamble,
    const EShOptimizationLevel optLevel,
    const TBuiltInResource* resources,
    int defaultVersion,  // use 100 for ES environment, 110 for desktop; this is the GLSL version, not SPIR-V or Vulkan
    EProfile defaultProfile,
    // set version/profile to defaultVersion/defaultProfile regardless of the #version
    // directive in the source code
    bool forceDefaultVersionAndProfile,
    int overrideVersion, // overrides version specified by #verison or default version
    bool forwardCompatible,     // give errors for use of deprecated features
    EShMessages messages,       // warnings/errors/AST; things to print out
    TIntermediate& intermediate, // returned tree, etc.
    ProcessingContext& processingContext,
    bool requireNonempty,
    TShader::Includer& includer,
    const std::string sourceEntryPointName = "",
    const TEnvironment* environment = nullptr,  // optional way of fully setting all versions, overriding the above
    bool compileOnly = false)
{
    // This must be undone (.pop()) by the caller, after it finishes consuming the created tree.
    GetThreadPoolAllocator().push();

    if (numStrings == 0)
        return true;

    // Move to length-based strings, rather than null-terminated strings.
    // Also, add strings to include the preamble and to ensure the shader is not null,
    // which lets the grammar accept what was a null (post preprocessing) shader.
    //
    // Shader will look like
    //   string 0:                system preamble
    //   string 1:                custom preamble
    //   string 2...numStrings+1: user's shader
    //   string numStrings+2:     "int;"
    const int numPre = 2;
    const int numPost = requireNonempty? 1 : 0;
    const int numTotal = numPre + numStrings + numPost;
    std::unique_ptr<size_t[]> lengths(new size_t[numTotal]);
    std::unique_ptr<const char*[]> strings(new const char*[numTotal]);
    std::unique_ptr<const char*[]> names(new const char*[numTotal]);
    for (int s = 0; s < numStrings; ++s) {
        strings[s + numPre] = shaderStrings[s];
        if (inputLengths == nullptr || inputLengths[s] < 0)
            lengths[s + numPre] = strlen(shaderStrings[s]);
        else
            lengths[s + numPre] = inputLengths[s];
    }
    if (stringNames != nullptr) {
        for (int s = 0; s < numStrings; ++s)
            names[s + numPre] = stringNames[s];
    } else {
        for (int s = 0; s < numStrings; ++s)
            names[s + numPre] = nullptr;
    }

    // Get all the stages, languages, clients, and other environment
    // stuff sorted out.
    EShSource sourceGuess = (messages & EShMsgReadHlsl) != 0 ? EShSourceHlsl : EShSourceGlsl;
    SpvVersion spvVersion;
    EShLanguage stage = compiler->getLanguage();
    TranslateEnvironment(environment, messages, sourceGuess, stage, spvVersion);
#ifdef ENABLE_HLSL
    EShSource source = sourceGuess;
    if (environment != nullptr && environment->target.hlslFunctionality1)
        intermediate.setHlslFunctionality1();
#else
    const EShSource source = EShSourceGlsl;
#endif
    // First, without using the preprocessor or parser, find the #version, so we know what
    // symbol tables, processing rules, etc. to set up.  This does not need the extra strings
    // outlined above, just the user shader, after the system and user preambles.
    glslang::TInputScanner userInput(numStrings, &strings[numPre], &lengths[numPre]);
    int version = 0;
    EProfile profile = ENoProfile;
    bool versionNotFirstToken = false;
    bool versionNotFirst = (source == EShSourceHlsl)
                                ? true
                                : userInput.scanVersion(version, profile, versionNotFirstToken);
    bool versionNotFound = version == 0;
    if (forceDefaultVersionAndProfile && source == EShSourceGlsl) {
        if (! (messages & EShMsgSuppressWarnings) && ! versionNotFound &&
            (version != defaultVersion || profile != defaultProfile)) {
            compiler->infoSink.info << "Warning, (version, profile) forced to be ("
                                    << defaultVersion << ", " << ProfileName(defaultProfile)
                                    << "), while in source code it is ("
                                    << version << ", " << ProfileName(profile) << ")\n";
        }

        if (versionNotFound) {
            versionNotFirstToken = false;
            versionNotFirst = false;
            versionNotFound = false;
        }
        version = defaultVersion;
        profile = defaultProfile;
    }
    if (source == EShSourceGlsl && overrideVersion != 0) {
        version = overrideVersion;
    }

    bool goodVersion = DeduceVersionProfile(compiler->infoSink, stage,
                                            versionNotFirst, defaultVersion, source, version, profile, spvVersion);
    bool versionWillBeError = (versionNotFound || (profile == EEsProfile && version >= 300 && versionNotFirst));
    bool warnVersionNotFirst = false;
    if (! versionWillBeError && versionNotFirstToken) {
        if (messages & EShMsgRelaxedErrors)
            warnVersionNotFirst = true;
        else
            versionWillBeError = true;
    }

    intermediate.setSource(source);
    intermediate.setVersion(version);
    intermediate.setProfile(profile);
    intermediate.setSpv(spvVersion);
    RecordProcesses(intermediate, messages, sourceEntryPointName);
    if (spvVersion.vulkan > 0)
        intermediate.setOriginUpperLeft();
#ifdef ENABLE_HLSL
    if ((messages & EShMsgHlslOffsets) || source == EShSourceHlsl)
        intermediate.setHlslOffsets();
#endif
    if (messages & EShMsgDebugInfo) {
        intermediate.setSourceFile(names[numPre]);
        for (int s = 0; s < numStrings; ++s) {
            // The string may not be null-terminated, so make sure we provide
            // the length along with the string.
            intermediate.addSourceText(strings[numPre + s], lengths[numPre + s]);
        }
    }
    SetupBuiltinSymbolTable(version, profile, spvVersion, source);

    TSymbolTable* cachedTable = SharedSymbolTables[MapVersionToIndex(version)]
                                                  [MapSpvVersionToIndex(spvVersion)]
                                                  [MapProfileToIndex(profile)]
                                                  [MapSourceToIndex(source)]
                                                  [stage];

    // Dynamically allocate the symbol table so we can control when it is deallocated WRT the pool.
    std::unique_ptr<TSymbolTable> symbolTable(new TSymbolTable);
    if (cachedTable)
        symbolTable->adoptLevels(*cachedTable);

    if (intermediate.getUniqueId() != 0)
        symbolTable->overwriteUniqueId(intermediate.getUniqueId());

    // Add built-in symbols that are potentially context dependent;
    // they get popped again further down.
    if (! AddContextSpecificSymbols(resources, compiler->infoSink, *symbolTable, version, profile, spvVersion,
                                    stage, source)) {
        return false;
    }

    if (messages & EShMsgBuiltinSymbolTable)
        DumpBuiltinSymbolTable(compiler->infoSink, *symbolTable);

    //
    // Now we can process the full shader under proper symbols and rules.
    //

    std::unique_ptr<TParseContextBase> parseContext(CreateParseContext(*symbolTable, intermediate, version, profile, source,
                                                    stage, compiler->infoSink,
                                                    spvVersion, forwardCompatible, messages, false, sourceEntryPointName));
    parseContext->compileOnly = compileOnly;
    TPpContext ppContext(*parseContext, names[numPre] ? names[numPre] : "", includer);

    // only GLSL (bison triggered, really) needs an externally set scan context
    glslang::TScanContext scanContext(*parseContext);
    if (source == EShSourceGlsl)
        parseContext->setScanContext(&scanContext);

    parseContext->setPpContext(&ppContext);
    parseContext->setLimits(*resources);
    if (! goodVersion)
        parseContext->addError();
    if (warnVersionNotFirst) {
        TSourceLoc loc;
        loc.init();
        parseContext->warn(loc, "Illegal to have non-comment, non-whitespace tokens before #version", "#version", "");
    }

    parseContext->initializeExtensionBehavior();

    // Fill in the strings as outlined above.
    std::string preamble;
    parseContext->getPreamble(preamble);
    strings[0] = preamble.c_str();
    lengths[0] = strlen(strings[0]);
    names[0] = nullptr;
    strings[1] = customPreamble;
    lengths[1] = strlen(strings[1]);
    names[1] = nullptr;
    assert(2 == numPre);
    if (requireNonempty) {
        const int postIndex = numStrings + numPre;
        strings[postIndex] = "\n int;";
        lengths[postIndex] = strlen(strings[numStrings + numPre]);
        names[postIndex] = nullptr;
    }
    TInputScanner fullInput(numStrings + numPre + numPost, strings.get(), lengths.get(), names.get(), numPre, numPost);

    // Push a new symbol allocation scope that will get used for the shader's globals.
    symbolTable->push();

    bool success = processingContext(*parseContext, ppContext, fullInput,
                                     versionWillBeError, *symbolTable,
                                     intermediate, optLevel, messages);
    intermediate.setUniqueId(symbolTable->getMaxSymbolId());
    return success;
}

// Responsible for keeping track of the most recent source string and line in
// the preprocessor and outputting newlines appropriately if the source string
// or line changes.
class SourceLineSynchronizer {
public:
    SourceLineSynchronizer(const std::function<int()>& lastSourceIndex,
                           std::string* output)
      : getLastSourceIndex(lastSourceIndex), output(output), lastSource(-1), lastLine(0) {}
//    SourceLineSynchronizer(const SourceLineSynchronizer&) = delete;
//    SourceLineSynchronizer& operator=(const SourceLineSynchronizer&) = delete;

    // Sets the internally tracked source string index to that of the most
    // recently read token. If we switched to a new source string, returns
    // true and inserts a newline. Otherwise, returns false and outputs nothing.
    bool syncToMostRecentString() {
        if (getLastSourceIndex() != lastSource) {
            // After switching to a new source string, we need to reset lastLine
            // because line number resets every time a new source string is
            // used. We also need to output a newline to separate the output
            // from the previous source string (if there is one).
            if (lastSource != -1 || lastLine != 0)
                *output += '\n';
            lastSource = getLastSourceIndex();
            lastLine = -1;
            return true;
        }
        return false;
    }

    // Calls syncToMostRecentString() and then sets the internally tracked line
    // number to tokenLine. If we switched to a new line, returns true and inserts
    // newlines appropriately. Otherwise, returns false and outputs nothing.
    bool syncToLine(int tokenLine) {
        syncToMostRecentString();
        const bool newLineStarted = lastLine < tokenLine;
        for (; lastLine < tokenLine; ++lastLine) {
            if (lastLine > 0) *output += '\n';
        }
        return newLineStarted;
    }

    // Sets the internally tracked line number to newLineNum.
    void setLineNum(int newLineNum) { lastLine = newLineNum; }

private:
    SourceLineSynchronizer& operator=(const SourceLineSynchronizer&);

    // A function for getting the index of the last valid source string we've
    // read tokens from.
    const std::function<int()> getLastSourceIndex;
    // output string for newlines.
    std::string* output;
    // lastSource is the source string index (starting from 0) of the last token
    // processed. It is tracked in order for newlines to be inserted when a new
    // source string starts. -1 means we haven't started processing any source
    // string.
    int lastSource;
    // lastLine is the line number (starting from 1) of the last token processed.
    // It is tracked in order for newlines to be inserted when a token appears
    // on a new line. 0 means we haven't started processing any line in the
    // current source string.
    int lastLine;
};

// DoPreprocessing is a valid ProcessingContext template argument,
// which only performs the preprocessing step of compilation.
// It places the result in the "string" argument to its constructor.
//
// This is not an officially supported or fully working path.
struct DoPreprocessing {
    explicit DoPreprocessing(std::string* string): outputString(string) {}
    bool operator()(TParseContextBase& parseContext, TPpContext& ppContext,
                    TInputScanner& input, bool versionWillBeError,
                    TSymbolTable&, TIntermediate&,
                    EShOptimizationLevel, EShMessages)
    {
        // This is a list of tokens that do not require a space before or after.
        static const std::string noNeededSpaceBeforeTokens = ";)[].,";
        static const std::string noNeededSpaceAfterTokens = ".([";
        glslang::TPpToken ppToken;

        parseContext.setScanner(&input);
        ppContext.setInput(input, versionWillBeError);

        std::string outputBuffer;
        SourceLineSynchronizer lineSync(
            std::bind(&TInputScanner::getLastValidSourceIndex, &input), &outputBuffer);

        parseContext.setExtensionCallback([&lineSync, &outputBuffer](
            int line, const char* extension, const char* behavior) {
                lineSync.syncToLine(line);
                outputBuffer += "#extension ";
                outputBuffer += extension;
                outputBuffer += " : ";
                outputBuffer += behavior;
        });

        parseContext.setLineCallback([&lineSync, &outputBuffer, &parseContext](
            int curLineNum, int newLineNum, bool hasSource, int sourceNum, const char* sourceName) {
            // SourceNum is the number of the source-string that is being parsed.
            lineSync.syncToLine(curLineNum);
            outputBuffer += "#line ";
            outputBuffer += std::to_string(newLineNum);
            if (hasSource) {
                outputBuffer += ' ';
                if (sourceName != nullptr) {
                    outputBuffer += '\"';
                    outputBuffer += sourceName;
                    outputBuffer += '\"';
                } else {
                    outputBuffer += std::to_string(sourceNum);
                }
            }
            if (parseContext.lineDirectiveShouldSetNextLine()) {
                // newLineNum is the new line number for the line following the #line
                // directive. So the new line number for the current line is
                newLineNum -= 1;
            }
            outputBuffer += '\n';
            // And we are at the next line of the #line directive now.
            lineSync.setLineNum(newLineNum + 1);
        });

        parseContext.setVersionCallback(
            [&lineSync, &outputBuffer](int line, int version, const char* str) {
                lineSync.syncToLine(line);
                outputBuffer += "#version ";
                outputBuffer += std::to_string(version);
                if (str) {
                    outputBuffer += ' ';
                    outputBuffer += str;
                }
            });

        parseContext.setPragmaCallback([&lineSync, &outputBuffer](
            int line, const glslang::TVector<glslang::TString>& ops) {
                lineSync.syncToLine(line);
                outputBuffer += "#pragma ";
                for(size_t i = 0; i < ops.size(); ++i) {
                    outputBuffer += ops[i].c_str();
                }
        });

        parseContext.setErrorCallback([&lineSync, &outputBuffer](
            int line, const char* errorMessage) {
                lineSync.syncToLine(line);
                outputBuffer += "#error ";
                outputBuffer += errorMessage;
        });

        int lastToken = EndOfInput; // lastToken records the last token processed.
        std::string lastTokenName;
        do {
            int token = ppContext.tokenize(ppToken);
            if (token == EndOfInput)
                break;

            bool isNewString = lineSync.syncToMostRecentString();
            bool isNewLine = lineSync.syncToLine(ppToken.loc.line);

            if (isNewLine) {
                // Don't emit whitespace onto empty lines.
                // Copy any whitespace characters at the start of a line
                // from the input to the output.
                outputBuffer += std::string(ppToken.loc.column - 1, ' ');
            }

            // Output a space in between tokens, but not at the start of a line,
            // and also not around special tokens. This helps with readability
            // and consistency.
            if (!isNewString && !isNewLine && lastToken != EndOfInput) {
                // left parenthesis need a leading space, except it is in a function-call-like context.
                // examples: `for (xxx)`, `a * (b + c)`, `vec(2.0)`, `foo(x, y, z)`
                if (token == '(') {
                    if (lastToken != PpAtomIdentifier ||
                        lastTokenName == "if" ||
                        lastTokenName == "for" ||
                        lastTokenName == "while" ||
                        lastTokenName == "switch")
                        outputBuffer += ' ';
                } else if ((noNeededSpaceBeforeTokens.find((char)token) == std::string::npos) &&
                    (noNeededSpaceAfterTokens.find((char)lastToken) == std::string::npos)) {
                    outputBuffer += ' ';
                }
            }
            if (token == PpAtomIdentifier)
                lastTokenName = ppToken.name;
            lastToken = token;
            if (token == PpAtomConstString)
                outputBuffer += "\"";
            outputBuffer += ppToken.name;
            if (token == PpAtomConstString)
                outputBuffer += "\"";
        } while (true);
        outputBuffer += '\n';
        *outputString = std::move(outputBuffer);

        bool success = true;
        if (parseContext.getNumErrors() > 0) {
            success = false;
            parseContext.infoSink.info.prefix(EPrefixError);
            parseContext.infoSink.info << parseContext.getNumErrors() << " compilation errors.  No code generated.\n\n";
        }
        return success;
    }
    std::string* outputString;
};

// DoFullParse is a valid ProcessingConext template argument for fully
// parsing the shader.  It populates the "intermediate" with the AST.
struct DoFullParse{
  bool operator()(TParseContextBase& parseContext, TPpContext& ppContext,
                  TInputScanner& fullInput, bool versionWillBeError,
                  TSymbolTable&, TIntermediate& intermediate,
                  EShOptimizationLevel optLevel, EShMessages messages)
    {
        bool success = true;
        // Parse the full shader.
        if (! parseContext.parseShaderStrings(ppContext, fullInput, versionWillBeError))
            success = false;

        if (success && intermediate.getTreeRoot()) {
            if (optLevel == EShOptNoGeneration)
                parseContext.infoSink.info.message(EPrefixNone, "No errors.  No code generation or linking was requested.");
            else
                success = intermediate.postProcess(intermediate.getTreeRoot(), parseContext.getLanguage());
        } else if (! success) {
            parseContext.infoSink.info.prefix(EPrefixError);
            parseContext.infoSink.info << parseContext.getNumErrors() << " compilation errors.  No code generated.\n\n";
        }

        if (messages & EShMsgAST)
            intermediate.output(parseContext.infoSink, true);

        return success;
    }
};

// Take a single compilation unit, and run the preprocessor on it.
// Return: True if there were no issues found in preprocessing,
//         False if during preprocessing any unknown version, pragmas or
//         extensions were found.
//
// NOTE: Doing just preprocessing to obtain a correct preprocessed shader string
// is not an officially supported or fully working path.
bool PreprocessDeferred(
    TCompiler* compiler,
    const char* const shaderStrings[],
    const int numStrings,
    const int* inputLengths,
    const char* const stringNames[],
    const char* preamble,
    const EShOptimizationLevel optLevel,
    const TBuiltInResource* resources,
    int defaultVersion,         // use 100 for ES environment, 110 for desktop
    EProfile defaultProfile,
    bool forceDefaultVersionAndProfile,
    int overrideVersion,        // use 0 if not overriding GLSL version
    bool forwardCompatible,     // give errors for use of deprecated features
    EShMessages messages,       // warnings/errors/AST; things to print out
    TShader::Includer& includer,
    TIntermediate& intermediate, // returned tree, etc.
    std::string* outputString,
    TEnvironment* environment = nullptr)
{
    DoPreprocessing parser(outputString);
    return ProcessDeferred(compiler, shaderStrings, numStrings, inputLengths, stringNames,
                           preamble, optLevel, resources, defaultVersion,
                           defaultProfile, forceDefaultVersionAndProfile, overrideVersion,
                           forwardCompatible, messages, intermediate, parser,
                           false, includer, "", environment);
}

//
// do a partial compile on the given strings for a single compilation unit
// for a potential deferred link into a single stage (and deferred full compile of that
// stage through machine-dependent compilation).
//
// all preprocessing, parsing, semantic checks, etc. for a single compilation unit
// are done here.
//
// return:  the tree and other information is filled into the intermediate argument,
//          and true is returned by the function for success.
//
bool CompileDeferred(
    TCompiler* compiler,
    const char* const shaderStrings[],
    const int numStrings,
    const int* inputLengths,
    const char* const stringNames[],
    const char* preamble,
    const EShOptimizationLevel optLevel,
    const TBuiltInResource* resources,
    int defaultVersion,         // use 100 for ES environment, 110 for desktop
    EProfile defaultProfile,
    bool forceDefaultVersionAndProfile,
    int overrideVersion,        // use 0 if not overriding GLSL version
    bool forwardCompatible,     // give errors for use of deprecated features
    EShMessages messages,       // warnings/errors/AST; things to print out
    TIntermediate& intermediate,// returned tree, etc.
    TShader::Includer& includer,
    const std::string sourceEntryPointName = "",
    TEnvironment* environment = nullptr,
    bool compileOnly = false)
{
    DoFullParse parser;
    return ProcessDeferred(compiler, shaderStrings, numStrings, inputLengths, stringNames,
                           preamble, optLevel, resources, defaultVersion,
                           defaultProfile, forceDefaultVersionAndProfile, overrideVersion,
                           forwardCompatible, messages, intermediate, parser,
                           true, includer, sourceEntryPointName, environment, compileOnly);
}

} // end anonymous namespace for local functions

//
// ShInitialize() should be called exactly once per process, not per thread.
//
int ShInitialize()
{
    const std::lock_guard<std::mutex> lock(init_lock);
    ++NumberOfClients;

    if (PerProcessGPA == nullptr)
        PerProcessGPA = new TPoolAllocator();

    glslang::TScanContext::fillInKeywordMap();
#ifdef ENABLE_HLSL
    glslang::HlslScanContext::fillInKeywordMap();
#endif

    return 1;
}

//
// Driver calls these to create and destroy compiler/linker
// objects.
//

ShHandle ShConstructCompiler(const EShLanguage language, int /*debugOptions unused*/)
{
    TShHandleBase* base = static_cast<TShHandleBase*>(ConstructCompiler(language, 0));

    return reinterpret_cast<void*>(base);
}

ShHandle ShConstructLinker(const EShExecutable executable, int /*debugOptions unused*/)
{
    TShHandleBase* base = static_cast<TShHandleBase*>(ConstructLinker(executable, 0));

    return reinterpret_cast<void*>(base);
}

ShHandle ShConstructUniformMap()
{
    TShHandleBase* base = static_cast<TShHandleBase*>(ConstructUniformMap());

    return reinterpret_cast<void*>(base);
}

void ShDestruct(ShHandle handle)
{
    if (handle == nullptr)
        return;

    TShHandleBase* base = static_cast<TShHandleBase*>(handle);

    if (base->getAsCompiler())
        DeleteCompiler(base->getAsCompiler());
    else if (base->getAsLinker())
        DeleteLinker(base->getAsLinker());
    else if (base->getAsUniformMap())
        DeleteUniformMap(base->getAsUniformMap());
}

//
// Cleanup symbol tables
//
int ShFinalize()
{
    const std::lock_guard<std::mutex> lock(init_lock);
    --NumberOfClients;
    assert(NumberOfClients >= 0);
    if (NumberOfClients > 0)
        return 1;

    for (int version = 0; version < VersionCount; ++version) {
        for (int spvVersion = 0; spvVersion < SpvVersionCount; ++spvVersion) {
            for (int p = 0; p < ProfileCount; ++p) {
                for (int source = 0; source < SourceCount; ++source) {
                    for (int stage = 0; stage < EShLangCount; ++stage) {
                        delete SharedSymbolTables[version][spvVersion][p][source][stage];
                        SharedSymbolTables[version][spvVersion][p][source][stage] = nullptr;
                    }
                }
            }
        }
    }

    for (int version = 0; version < VersionCount; ++version) {
        for (int spvVersion = 0; spvVersion < SpvVersionCount; ++spvVersion) {
            for (int p = 0; p < ProfileCount; ++p) {
                for (int source = 0; source < SourceCount; ++source) {
                    for (int pc = 0; pc < EPcCount; ++pc) {
                        delete CommonSymbolTable[version][spvVersion][p][source][pc];
                        CommonSymbolTable[version][spvVersion][p][source][pc] = nullptr;
                    }
                }
            }
        }
    }

    if (PerProcessGPA != nullptr) {
        delete PerProcessGPA;
        PerProcessGPA = nullptr;
    }

    glslang::TScanContext::deleteKeywordMap();
#ifdef ENABLE_HLSL
    glslang::HlslScanContext::deleteKeywordMap();
#endif

    return 1;
}

//
// Do a full compile on the given strings for a single compilation unit
// forming a complete stage.  The result of the machine dependent compilation
// is left in the provided compile object.
//
// Return:  The return value is really boolean, indicating
// success (1) or failure (0).
//
int ShCompile(
    const ShHandle handle,
    const char* const shaderStrings[],
    const int numStrings,
    const int* inputLengths,
    const EShOptimizationLevel optLevel,
    const TBuiltInResource* resources,
    int /*debugOptions*/,
    int defaultVersion,        // use 100 for ES environment, 110 for desktop
    bool forwardCompatible,    // give errors for use of deprecated features
    EShMessages messages,       // warnings/errors/AST; things to print out,
    const char *shaderFileName // the filename
    )
{
    // Map the generic handle to the C++ object
    if (handle == nullptr)
        return 0;

    TShHandleBase* base = reinterpret_cast<TShHandleBase*>(handle);
    TCompiler* compiler = base->getAsCompiler();
    if (compiler == nullptr)
        return 0;

    SetThreadPoolAllocator(compiler->getPool());

    compiler->infoSink.info.erase();
    compiler->infoSink.debug.erase();
    compiler->infoSink.info.setShaderFileName(shaderFileName);
    compiler->infoSink.debug.setShaderFileName(shaderFileName);


    TIntermediate intermediate(compiler->getLanguage());
    TShader::ForbidIncluder includer;
    bool success = CompileDeferred(compiler, shaderStrings, numStrings, inputLengths, nullptr,
                                   "", optLevel, resources, defaultVersion, ENoProfile, false, 0,
                                   forwardCompatible, messages, intermediate, includer);

    //
    // Call the machine dependent compiler
    //
    if (success && intermediate.getTreeRoot() && optLevel != EShOptNoGeneration)
        success = compiler->compile(intermediate.getTreeRoot(), intermediate.getVersion(), intermediate.getProfile());

    intermediate.removeTree();

    // Throw away all the temporary memory used by the compilation process.
    // The push was done in the CompileDeferred() call above.
    GetThreadPoolAllocator().pop();

    return success ? 1 : 0;
}

//
// Link the given compile objects.
//
// Return:  The return value of is really boolean, indicating
// success or failure.
//
int ShLinkExt(
    const ShHandle linkHandle,
    const ShHandle compHandles[],
    const int numHandles)
{
    if (linkHandle == nullptr || numHandles == 0)
        return 0;

    THandleList cObjects;

    for (int i = 0; i < numHandles; ++i) {
        if (compHandles[i] == nullptr)
            return 0;
        TShHandleBase* base = reinterpret_cast<TShHandleBase*>(compHandles[i]);
        if (base->getAsLinker()) {
            cObjects.push_back(base->getAsLinker());
        }
        if (base->getAsCompiler())
            cObjects.push_back(base->getAsCompiler());

        if (cObjects[i] == nullptr)
            return 0;
    }

    TShHandleBase* base = reinterpret_cast<TShHandleBase*>(linkHandle);
    TLinker* linker = static_cast<TLinker*>(base->getAsLinker());

    if (linker == nullptr)
        return 0;
    
    SetThreadPoolAllocator(linker->getPool());
    linker->infoSink.info.erase();

    for (int i = 0; i < numHandles; ++i) {
        if (cObjects[i]->getAsCompiler()) {
            if (! cObjects[i]->getAsCompiler()->linkable()) {
                linker->infoSink.info.message(EPrefixError, "Not all shaders have valid object code.");
                return 0;
            }
        }
    }

    bool ret = linker->link(cObjects);

    return ret ? 1 : 0;
}

//
// ShSetEncrpytionMethod is a place-holder for specifying
// how source code is encrypted.
//
void ShSetEncryptionMethod(ShHandle handle)
{
    if (handle == nullptr)
        return;
}

//
// Return any compiler/linker/uniformmap log of messages for the application.
//
const char* ShGetInfoLog(const ShHandle handle)
{
    if (handle == nullptr)
        return nullptr;

    TShHandleBase* base = static_cast<TShHandleBase*>(handle);
    TInfoSink* infoSink;

    if (base->getAsCompiler())
        infoSink = &(base->getAsCompiler()->getInfoSink());
    else if (base->getAsLinker())
        infoSink = &(base->getAsLinker()->getInfoSink());
    else
        return nullptr;

    infoSink->info << infoSink->debug.c_str();
    return infoSink->info.c_str();
}

//
// Return the resulting binary code from the link process.  Structure
// is machine dependent.
//
const void* ShGetExecutable(const ShHandle handle)
{
    if (handle == nullptr)
        return nullptr;

    TShHandleBase* base = reinterpret_cast<TShHandleBase*>(handle);

    TLinker* linker = static_cast<TLinker*>(base->getAsLinker());
    if (linker == nullptr)
        return nullptr;

    return linker->getObjectCode();
}

//
// Let the linker know where the application said it's attributes are bound.
// The linker does not use these values, they are remapped by the ICD or
// hardware.  It just needs them to know what's aliased.
//
// Return:  The return value of is really boolean, indicating
// success or failure.
//
int ShSetVirtualAttributeBindings(const ShHandle handle, const ShBindingTable* table)
{
    if (handle == nullptr)
        return 0;

    TShHandleBase* base = reinterpret_cast<TShHandleBase*>(handle);
    TLinker* linker = static_cast<TLinker*>(base->getAsLinker());

    if (linker == nullptr)
        return 0;

    linker->setAppAttributeBindings(table);

    return 1;
}

//
// Let the linker know where the predefined attributes have to live.
//
int ShSetFixedAttributeBindings(const ShHandle handle, const ShBindingTable* table)
{
    if (handle == nullptr)
        return 0;

    TShHandleBase* base = reinterpret_cast<TShHandleBase*>(handle);
    TLinker* linker = static_cast<TLinker*>(base->getAsLinker());

    if (linker == nullptr)
        return 0;

    linker->setFixedAttributeBindings(table);
    return 1;
}

//
// Some attribute locations are off-limits to the linker...
//
int ShExcludeAttributes(const ShHandle handle, int *attributes, int count)
{
    if (handle == nullptr)
        return 0;

    TShHandleBase* base = reinterpret_cast<TShHandleBase*>(handle);
    TLinker* linker = static_cast<TLinker*>(base->getAsLinker());
    if (linker == nullptr)
        return 0;

    linker->setExcludedAttributes(attributes, count);

    return 1;
}

//
// Return the index for OpenGL to use for knowing where a uniform lives.
//
// Return:  The return value of is really boolean, indicating
// success or failure.
//
int ShGetUniformLocation(const ShHandle handle, const char* name)
{
    if (handle == nullptr)
        return -1;

    TShHandleBase* base = reinterpret_cast<TShHandleBase*>(handle);
    TUniformMap* uniformMap= base->getAsUniformMap();
    if (uniformMap == nullptr)
        return -1;

    return uniformMap->getLocation(name);
}

////////////////////////////////////////////////////////////////////////////////////////////
//
// Deferred-Lowering C++ Interface
// -----------------------------------
//
// Below is a new alternate C++ interface that might potentially replace the above
// opaque handle-based interface.
//
// See more detailed comment in ShaderLang.h
//

namespace glslang {

Version GetVersion()
{
    Version version;
    version.major = GLSLANG_VERSION_MAJOR;
    version.minor = GLSLANG_VERSION_MINOR;
    version.patch = GLSLANG_VERSION_PATCH;
    version.flavor = GLSLANG_VERSION_FLAVOR;
    return version;
}

#define QUOTE(s) #s
#define STR(n) QUOTE(n)

const char* GetEsslVersionString()
{
    return "OpenGL ES GLSL 3.20 glslang Khronos. " STR(GLSLANG_VERSION_MAJOR) "." STR(GLSLANG_VERSION_MINOR) "." STR(
        GLSLANG_VERSION_PATCH) GLSLANG_VERSION_FLAVOR;
}

const char* GetGlslVersionString()
{
    return "4.60 glslang Khronos. " STR(GLSLANG_VERSION_MAJOR) "." STR(GLSLANG_VERSION_MINOR) "." STR(
        GLSLANG_VERSION_PATCH) GLSLANG_VERSION_FLAVOR;
}

int GetKhronosToolId()
{
    return 8;
}

bool InitializeProcess()
{
    return ShInitialize() != 0;
}

void FinalizeProcess()
{
    ShFinalize();
}

class TDeferredCompiler : public TCompiler {
public:
    TDeferredCompiler(EShLanguage s, TInfoSink& i) : TCompiler(s, i) { }
    virtual bool compile(TIntermNode*, int = 0, EProfile = ENoProfile) { return true; }
};

TShader::TShader(EShLanguage s)
    : stage(s), lengths(nullptr), stringNames(nullptr), preamble(""), overrideVersion(0)
{
    pool = new TPoolAllocator;
    infoSink = new TInfoSink;
    compiler = new TDeferredCompiler(stage, *infoSink);
    intermediate = new TIntermediate(s);

    // clear environment (avoid constructors in them for use in a C interface)
    environment.input.languageFamily = EShSourceNone;
    environment.input.dialect = EShClientNone;
    environment.input.vulkanRulesRelaxed = false;
    environment.client.client = EShClientNone;
    environment.target.language = EShTargetNone;
    environment.target.hlslFunctionality1 = false;
}

TShader::~TShader()
{
    delete infoSink;
    delete compiler;
    delete intermediate;
    delete pool;
}

void TShader::setStrings(const char* const* s, int n)
{
    strings = s;
    numStrings = n;
    lengths = nullptr;
}

void TShader::setStringsWithLengths(const char* const* s, const int* l, int n)
{
    strings = s;
    numStrings = n;
    lengths = l;
}

void TShader::setStringsWithLengthsAndNames(
    const char* const* s, const int* l, const char* const* names, int n)
{
    strings = s;
    numStrings = n;
    lengths = l;
    stringNames = names;
}

void TShader::setEntryPoint(const char* entryPoint)
{
    intermediate->setEntryPointName(entryPoint);
}

void TShader::setSourceEntryPoint(const char* name)
{
    sourceEntryPointName = name;
}

// Log initial settings and transforms.
// See comment for class TProcesses.
void TShader::addProcesses(const std::vector<std::string>& p)
{
    intermediate->addProcesses(p);
}

void  TShader::setUniqueId(unsigned long long id)
{
    intermediate->setUniqueId(id);
}

void TShader::setOverrideVersion(int version)
{
    overrideVersion = version;
}

void TShader::setDebugInfo(bool debugInfo)              { intermediate->setDebugInfo(debugInfo); }
void TShader::setInvertY(bool invert)                   { intermediate->setInvertY(invert); }
void TShader::setDxPositionW(bool invert)               { intermediate->setDxPositionW(invert); }
void TShader::setEnhancedMsgs()                         { intermediate->setEnhancedMsgs(); }
void TShader::setNanMinMaxClamp(bool useNonNan)         { intermediate->setNanMinMaxClamp(useNonNan); }

// Set binding base for given resource type
void TShader::setShiftBinding(TResourceType res, unsigned int base) {
    intermediate->setShiftBinding(res, base);
}

// Set binding base for given resource type for a given binding set.
void TShader::setShiftBindingForSet(TResourceType res, unsigned int base, unsigned int set) {
    intermediate->setShiftBindingForSet(res, base, set);
}

// Set binding base for sampler types
void TShader::setShiftSamplerBinding(unsigned int base) { setShiftBinding(EResSampler, base); }
// Set binding base for texture types (SRV)
void TShader::setShiftTextureBinding(unsigned int base) { setShiftBinding(EResTexture, base); }
// Set binding base for image types
void TShader::setShiftImageBinding(unsigned int base)   { setShiftBinding(EResImage, base); }
// Set binding base for uniform buffer objects (CBV)
void TShader::setShiftUboBinding(unsigned int base)     { setShiftBinding(EResUbo, base); }
// Synonym for setShiftUboBinding, to match HLSL language.
void TShader::setShiftCbufferBinding(unsigned int base) { setShiftBinding(EResUbo, base); }
// Set binding base for UAV (unordered access view)
void TShader::setShiftUavBinding(unsigned int base)     { setShiftBinding(EResUav, base); }
// Set binding base for SSBOs
void TShader::setShiftSsboBinding(unsigned int base)    { setShiftBinding(EResSsbo, base); }
// Enables binding automapping using TIoMapper
void TShader::setAutoMapBindings(bool map)              { intermediate->setAutoMapBindings(map); }
// Enables position.Y output negation in vertex shader

// Fragile: currently within one stage: simple auto-assignment of location
void TShader::setAutoMapLocations(bool map)             { intermediate->setAutoMapLocations(map); }
void TShader::addUniformLocationOverride(const char* name, int loc)
{
    intermediate->addUniformLocationOverride(name, loc);
}
void TShader::setUniformLocationBase(int base)
{
    intermediate->setUniformLocationBase(base);
}
void TShader::setNoStorageFormat(bool useUnknownFormat) { intermediate->setNoStorageFormat(useUnknownFormat); }
void TShader::setResourceSetBinding(const std::vector<std::string>& base)   { intermediate->setResourceSetBinding(base); }
void TShader::setTextureSamplerTransformMode(EShTextureSamplerTransformMode mode) { intermediate->setTextureSamplerTransformMode(mode); }

void TShader::addBlockStorageOverride(const char* nameStr, TBlockStorageClass backing) { intermediate->addBlockStorageOverride(nameStr, backing); }

void TShader::setGlobalUniformBlockName(const char* name) { intermediate->setGlobalUniformBlockName(name); }
void TShader::setGlobalUniformSet(unsigned int set) { intermediate->setGlobalUniformSet(set); }
void TShader::setGlobalUniformBinding(unsigned int binding) { intermediate->setGlobalUniformBinding(binding); }

void TShader::setAtomicCounterBlockName(const char* name) { intermediate->setAtomicCounterBlockName(name); }
void TShader::setAtomicCounterBlockSet(unsigned int set) { intermediate->setAtomicCounterBlockSet(set); }

#ifdef ENABLE_HLSL
// See comment above TDefaultHlslIoMapper in iomapper.cpp:
void TShader::setHlslIoMapping(bool hlslIoMap)          { intermediate->setHlslIoMapping(hlslIoMap); }
void TShader::setFlattenUniformArrays(bool flatten)     { intermediate->setFlattenUniformArrays(flatten); }
#endif

//
// Turn the shader strings into a parse tree in the TIntermediate.
//
// Returns true for success.
//
bool TShader::parse(const TBuiltInResource* builtInResources, int defaultVersion, EProfile defaultProfile, bool forceDefaultVersionAndProfile,
                    bool forwardCompatible, EShMessages messages, Includer& includer)
{
    SetThreadPoolAllocator(pool);

    if (! preamble)
        preamble = "";

    return CompileDeferred(compiler, strings, numStrings, lengths, stringNames,
                           preamble, EShOptNone, builtInResources, defaultVersion,
                           defaultProfile, forceDefaultVersionAndProfile, overrideVersion,
                           forwardCompatible, messages, *intermediate, includer, sourceEntryPointName,
                           &environment, compileOnly);
}

// Fill in a string with the result of preprocessing ShaderStrings
// Returns true if all extensions, pragmas and version strings were valid.
//
// NOTE: Doing just preprocessing to obtain a correct preprocessed shader string
// is not an officially supported or fully working path.
bool TShader::preprocess(const TBuiltInResource* builtInResources,
                         int defaultVersion, EProfile defaultProfile,
                         bool forceDefaultVersionAndProfile,
                         bool forwardCompatible, EShMessages message,
                         std::string* output_string,
                         Includer& includer)
{
    SetThreadPoolAllocator(pool);

    if (! preamble)
        preamble = "";

    return PreprocessDeferred(compiler, strings, numStrings, lengths, stringNames, preamble,
                              EShOptNone, builtInResources, defaultVersion,
                              defaultProfile, forceDefaultVersionAndProfile, overrideVersion,
                              forwardCompatible, message, includer, *intermediate, output_string,
                              &environment);
}

const char* TShader::getInfoLog()
{
    return infoSink->info.c_str();
}

const char* TShader::getInfoDebugLog()
{
    return infoSink->debug.c_str();
}

TProgram::TProgram() : reflection(nullptr), linked(false)
{
    pool = new TPoolAllocator;
    infoSink = new TInfoSink;
    for (int s = 0; s < EShLangCount; ++s) {
        intermediate[s] = nullptr;
        newedIntermediate[s] = false;
    }
}

TProgram::~TProgram()
{
    delete infoSink;
    delete reflection;

    for (int s = 0; s < EShLangCount; ++s)
        if (newedIntermediate[s])
            delete intermediate[s];

    delete pool;
}

//
// Merge the compilation units within each stage into a single TIntermediate.
// All starting compilation units need to be the result of calling TShader::parse().
//
// Return true for success.
//
bool TProgram::link(EShMessages messages)
{
    if (linked)
        return false;
    linked = true;

    bool error = false;

    SetThreadPoolAllocator(pool);

    for (int s = 0; s < EShLangCount; ++s) {
        if (! linkStage((EShLanguage)s, messages))
            error = true;
    }

    if (!error) {
        if (! crossStageCheck(messages))
            error = true;
    }

    return ! error;
}

//
// Merge the compilation units within the given stage into a single TIntermediate.
//
// Return true for success.
//
bool TProgram::linkStage(EShLanguage stage, EShMessages messages)
{
    if (stages[stage].size() == 0)
        return true;

    int numEsShaders = 0, numNonEsShaders = 0;
    for (auto it = stages[stage].begin(); it != stages[stage].end(); ++it) {
        if ((*it)->intermediate->getProfile() == EEsProfile) {
            numEsShaders++;
        } else {
            numNonEsShaders++;
        }
    }

    if (numEsShaders > 0 && numNonEsShaders > 0) {
        infoSink->info.message(EPrefixError, "Cannot mix ES profile with non-ES profile shaders");
        return false;
    } else if (numEsShaders > 1) {
        infoSink->info.message(EPrefixError, "Cannot attach multiple ES shaders of the same type to a single program");
        return false;
    }

    //
    // Be efficient for the common single compilation unit per stage case,
    // reusing it's TIntermediate instead of merging into a new one.
    //
    TIntermediate *firstIntermediate = stages[stage].front()->intermediate;
    if (stages[stage].size() == 1)
        intermediate[stage] = firstIntermediate;
    else {
        intermediate[stage] = new TIntermediate(stage,
                                                firstIntermediate->getVersion(),
                                                firstIntermediate->getProfile());
        intermediate[stage]->setLimits(firstIntermediate->getLimits());
        if (firstIntermediate->getEnhancedMsgs())
            intermediate[stage]->setEnhancedMsgs();

        // The new TIntermediate must use the same origin as the original TIntermediates.
        // Otherwise linking will fail due to different coordinate systems.
        if (firstIntermediate->getOriginUpperLeft()) {
            intermediate[stage]->setOriginUpperLeft();
        }
        intermediate[stage]->setSpv(firstIntermediate->getSpv());

        newedIntermediate[stage] = true;
    }

    if (messages & EShMsgAST)
        infoSink->info << "\nLinked " << StageName(stage) << " stage:\n\n";

    if (stages[stage].size() > 1) {
        std::list<TShader*>::const_iterator it;
        for (it = stages[stage].begin(); it != stages[stage].end(); ++it)
            intermediate[stage]->merge(*infoSink, *(*it)->intermediate);
    }
    intermediate[stage]->finalCheck(*infoSink, (messages & EShMsgKeepUncalled) != 0);

    if (messages & EShMsgAST)
        intermediate[stage]->output(*infoSink, true);

    return intermediate[stage]->getNumErrors() == 0;
}

//
// Check that there are no errors in linker objects accross stages
//
// Return true if no errors.
//
bool TProgram::crossStageCheck(EShMessages) {

    // make temporary intermediates to hold the linkage symbols for each linking interface
    // while we do the checks
    // Independent interfaces are:
    //                  all uniform variables and blocks
    //                  all buffer blocks
    //                  all in/out on a stage boundary

    TVector<TIntermediate*> activeStages;
    for (int s = 0; s < EShLangCount; ++s) {
        if (intermediate[s])
            activeStages.push_back(intermediate[s]);
    }

    // no extra linking if there is only one stage
    if (! (activeStages.size() > 1))
        return true;

    // setup temporary tree to hold unfirom objects from different stages
    TIntermediate* firstIntermediate = activeStages.front();
    TIntermediate uniforms(EShLangCount,
                           firstIntermediate->getVersion(),
                           firstIntermediate->getProfile());
    uniforms.setSpv(firstIntermediate->getSpv());

    TIntermAggregate uniformObjects(EOpLinkerObjects);
    TIntermAggregate root(EOpSequence);
    root.getSequence().push_back(&uniformObjects);
    uniforms.setTreeRoot(&root);

    bool error = false;

    // merge uniforms from all stages into a single intermediate
    for (unsigned int i = 0; i < activeStages.size(); ++i) {
        uniforms.mergeUniformObjects(*infoSink, *activeStages[i]);
    }
    error |= uniforms.getNumErrors() != 0;

    // copy final definition of global block back into each stage
    for (unsigned int i = 0; i < activeStages.size(); ++i) {
        // We only want to merge into already existing global uniform blocks.
        // A stage that doesn't already know about the global doesn't care about it's content.
        // Otherwise we end up pointing to the same object between different stages
        // and that will break binding/set remappings
        bool mergeExistingOnly = true;
        activeStages[i]->mergeGlobalUniformBlocks(*infoSink, uniforms, mergeExistingOnly);
    }

    // compare cross stage symbols for each stage boundary
    for (unsigned int i = 1; i < activeStages.size(); ++i) {
        activeStages[i - 1]->checkStageIO(*infoSink, *activeStages[i]);
        error |= (activeStages[i - 1]->getNumErrors() != 0);
    }

    return !error;
}

const char* TProgram::getInfoLog()
{
    return infoSink->info.c_str();
}

const char* TProgram::getInfoDebugLog()
{
    return infoSink->debug.c_str();
}

//
// Reflection implementation.
//

bool TProgram::buildReflection(int opts)
{
    if (! linked || reflection != nullptr)
        return false;

    int firstStage = EShLangVertex, lastStage = EShLangFragment;

    if (opts & EShReflectionIntermediateIO) {
        // if we're reflecting intermediate I/O, determine the first and last stage linked and use those as the
        // boundaries for which stages generate pipeline inputs/outputs
        firstStage = EShLangCount;
        lastStage = 0;
        for (int s = 0; s < EShLangCount; ++s) {
            if (intermediate[s]) {
                firstStage = std::min(firstStage, s);
                lastStage = std::max(lastStage, s);
            }
        }
    }

    reflection = new TReflection((EShReflectionOptions)opts, (EShLanguage)firstStage, (EShLanguage)lastStage);

    for (int s = 0; s < EShLangCount; ++s) {
        if (intermediate[s]) {
            if (! reflection->addStage((EShLanguage)s, *intermediate[s]))
                return false;
        }
    }

    return true;
}

unsigned TProgram::getLocalSize(int dim) const                        { return reflection->getLocalSize(dim); }
int TProgram::getReflectionIndex(const char* name) const              { return reflection->getIndex(name); }
int TProgram::getReflectionPipeIOIndex(const char* name, const bool inOrOut) const
                                                                      { return reflection->getPipeIOIndex(name, inOrOut); }

int TProgram::getNumUniformVariables() const                          { return reflection->getNumUniforms(); }
const TObjectReflection& TProgram::getUniform(int index) const        { return reflection->getUniform(index); }
int TProgram::getNumUniformBlocks() const                             { return reflection->getNumUniformBlocks(); }
const TObjectReflection& TProgram::getUniformBlock(int index) const   { return reflection->getUniformBlock(index); }
int TProgram::getNumPipeInputs() const                                { return reflection->getNumPipeInputs(); }
const TObjectReflection& TProgram::getPipeInput(int index) const      { return reflection->getPipeInput(index); }
int TProgram::getNumPipeOutputs() const                               { return reflection->getNumPipeOutputs(); }
const TObjectReflection& TProgram::getPipeOutput(int index) const     { return reflection->getPipeOutput(index); }
int TProgram::getNumBufferVariables() const                           { return reflection->getNumBufferVariables(); }
const TObjectReflection& TProgram::getBufferVariable(int index) const { return reflection->getBufferVariable(index); }
int TProgram::getNumBufferBlocks() const                              { return reflection->getNumStorageBuffers(); }
const TObjectReflection& TProgram::getBufferBlock(int index) const    { return reflection->getStorageBufferBlock(index); }
int TProgram::getNumAtomicCounters() const                            { return reflection->getNumAtomicCounters(); }
const TObjectReflection& TProgram::getAtomicCounter(int index) const  { return reflection->getAtomicCounter(index); }
void TProgram::dumpReflection() { if (reflection != nullptr) reflection->dump(); }

//
// I/O mapping implementation.
//
bool TProgram::mapIO(TIoMapResolver* pResolver, TIoMapper* pIoMapper)
{
    if (! linked)
        return false;
    TIoMapper* ioMapper = nullptr;
    TIoMapper defaultIOMapper;
    if (pIoMapper == nullptr)
        ioMapper = &defaultIOMapper;
    else
        ioMapper = pIoMapper;
    for (int s = 0; s < EShLangCount; ++s) {
        if (intermediate[s]) {
            if (! ioMapper->addStage((EShLanguage)s, *intermediate[s], *infoSink, pResolver))
                return false;
        }
    }

    return ioMapper->doMap(pResolver, *infoSink);
}

} // end namespace glslang
