//
// Copyright (C) 2016-2017 LunarG, Inc.
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

#include "../Include/Common.h"
#include "../Include/InfoSink.h"
#include "../Include/Types.h"

#include "gl_types.h"
#include "iomapper.h"
#include "SymbolTable.h"

//
// Map IO bindings.
//
// High-level algorithm for one stage:
//
// 1. Traverse all code (live+dead) to find the explicitly provided bindings.
//
// 2. Traverse (just) the live code to determine which non-provided bindings
//    require auto-numbering.  We do not auto-number dead ones.
//
// 3. Traverse all the code to apply the bindings:
//    a. explicitly given bindings are offset according to their type
//    b. implicit live bindings are auto-numbered into the holes, using
//       any open binding slot.
//    c. implicit dead bindings are left un-bound.
//

namespace glslang {

class TVarGatherTraverser : public TLiveTraverser {
public:
    TVarGatherTraverser(const TIntermediate& i, bool traverseDeadCode, TVarLiveMap& inList, TVarLiveMap& outList, TVarLiveMap& uniformList)
      : TLiveTraverser(i, traverseDeadCode, true, true, false)
      , inputList(inList)
      , outputList(outList)
      , uniformList(uniformList)
    {
    }

    virtual void visitSymbol(TIntermSymbol* base)
    {
        TVarLiveMap* target = nullptr;
        if (base->getQualifier().storage == EvqVaryingIn)
            target = &inputList;
        else if (base->getQualifier().storage == EvqVaryingOut)
            target = &outputList;
        else if (base->getQualifier().isUniformOrBuffer() && !base->getQualifier().isPushConstant() && !base->getQualifier().isShaderRecord())
            target = &uniformList;
        // If a global is being visited, then we should also traverse it incase it's evaluation
        // ends up visiting inputs we want to tag as live
        else if (base->getQualifier().storage == EvqGlobal)
            addGlobalReference(base->getAccessName());

        if (target) {
            TVarEntryInfo ent = {base->getId(), base, ! traverseAll, {}, {}, {}, {}, {}, {}, {}};
            ent.stage = intermediate.getStage();
            TVarLiveMap::iterator at = target->find(
                ent.symbol->getAccessName()); // std::lower_bound(target->begin(), target->end(), ent, TVarEntryInfo::TOrderById());
            if (at != target->end() && at->second.id == ent.id)
                at->second.live = at->second.live || ! traverseAll; // update live state
            else
                (*target)[ent.symbol->getAccessName()] = ent;
        }
    }

private:
    TVarLiveMap&    inputList;
    TVarLiveMap&    outputList;
    TVarLiveMap&    uniformList;
};

class TVarSetTraverser : public TLiveTraverser
{
public:
    TVarSetTraverser(const TIntermediate& i, const TVarLiveMap& inList, const TVarLiveMap& outList, const TVarLiveMap& uniformList)
      : TLiveTraverser(i, true, true, true, false)
      , inputList(inList)
      , outputList(outList)
      , uniformList(uniformList)
    {
    }

    virtual void visitSymbol(TIntermSymbol* base) {
        const TVarLiveMap* source;
        if (base->getQualifier().storage == EvqVaryingIn)
            source = &inputList;
        else if (base->getQualifier().storage == EvqVaryingOut)
            source = &outputList;
        else if (base->getQualifier().isUniformOrBuffer())
            source = &uniformList;
        else
            return;

        TVarEntryInfo ent = { base->getId(), {}, {}, {}, {}, {}, {}, {}, {}, {} };
        // Fix a defect, when block has no instance name, we need to find its block name
        TVarLiveMap::const_iterator at = source->find(base->getAccessName());
        if (at == source->end())
            return;

        if (at->second.id != ent.id)
            return;

        if (at->second.newBinding != -1)
            base->getWritableType().getQualifier().layoutBinding = at->second.newBinding;
        if (at->second.newSet != -1)
            base->getWritableType().getQualifier().layoutSet = at->second.newSet;
        if (at->second.newLocation != -1)
            base->getWritableType().getQualifier().layoutLocation = at->second.newLocation;
        if (at->second.newComponent != -1)
            base->getWritableType().getQualifier().layoutComponent = at->second.newComponent;
        if (at->second.newIndex != -1)
            base->getWritableType().getQualifier().layoutIndex = at->second.newIndex;
        if (at->second.upgradedToPushConstant)
            base->getWritableType().getQualifier().layoutPushConstant = true;
    }

  private:
    const TVarLiveMap&    inputList;
    const TVarLiveMap&    outputList;
    const TVarLiveMap&    uniformList;
};

struct TNotifyUniformAdaptor
{
    EShLanguage stage;
    TIoMapResolver& resolver;
    inline TNotifyUniformAdaptor(EShLanguage s, TIoMapResolver& r)
      : stage(s)
      , resolver(r)
    {
    }

    inline void operator()(std::pair<const TString, TVarEntryInfo>& entKey)
    {
        resolver.notifyBinding(stage, entKey.second);
    }

private:
    TNotifyUniformAdaptor& operator=(TNotifyUniformAdaptor&) = delete;
};

struct TNotifyInOutAdaptor
{
    EShLanguage stage;
    TIoMapResolver& resolver;
    inline TNotifyInOutAdaptor(EShLanguage s, TIoMapResolver& r) 
      : stage(s)
      , resolver(r)
    {
    }

    inline void operator()(std::pair<const TString, TVarEntryInfo>& entKey)
    {
        resolver.notifyInOut(entKey.second.stage, entKey.second);
    }

private:
    TNotifyInOutAdaptor& operator=(TNotifyInOutAdaptor&) = delete;
};

struct TResolverUniformAdaptor {
    TResolverUniformAdaptor(EShLanguage s, TIoMapResolver& r, TVarLiveMap* uniform[EShLangCount], TInfoSink& i, bool& e)
      : stage(s)
      , resolver(r)
      , infoSink(i)
      , error(e)
    {
        memcpy(uniformVarMap, uniform, EShLangCount * (sizeof(TVarLiveMap*)));
    }

    inline void operator()(std::pair<const TString, TVarEntryInfo>& entKey) {
        TVarEntryInfo& ent = entKey.second;
        ent.clearNewAssignments();
        const bool isValid = resolver.validateBinding(stage, ent);
        if (isValid) {
            resolver.resolveSet(ent.stage, ent);
            resolver.resolveBinding(ent.stage, ent);
            resolver.resolveUniformLocation(ent.stage, ent);

            if (ent.newBinding != -1) {
                if (ent.newBinding >= int(TQualifier::layoutBindingEnd)) {
                    TString err = "mapped binding out of range: " + entKey.first;

                    infoSink.info.message(EPrefixInternalError, err.c_str());
                    error = true;
                }

                if (ent.symbol->getQualifier().hasBinding()) {
                    for (uint32_t idx = EShLangVertex; idx < EShLangCount; ++idx) {
                        if (idx == ent.stage || uniformVarMap[idx] == nullptr)
                            continue;
                        auto entKey2 = uniformVarMap[idx]->find(entKey.first);
                        if (entKey2 != uniformVarMap[idx]->end()) {
                            entKey2->second.newBinding = ent.newBinding;
                        }
                    }
                }
            }
            if (ent.newSet != -1) {
                if (ent.newSet >= int(TQualifier::layoutSetEnd)) {
                    TString err = "mapped set out of range: " + entKey.first;

                    infoSink.info.message(EPrefixInternalError, err.c_str());
                    error = true;
                }
                if (ent.symbol->getQualifier().hasSet()) {
                    for (uint32_t idx = EShLangVertex; idx < EShLangCount; ++idx) {
                        if ((idx == stage) || (uniformVarMap[idx] == nullptr))
                            continue;
                        auto entKey2 = uniformVarMap[idx]->find(entKey.first);
                        if (entKey2 != uniformVarMap[idx]->end()) {
                            entKey2->second.newSet = ent.newSet;
                        }
                    }
                }
            }
        } else {
            TString errorMsg = "Invalid binding: " + entKey.first;
            infoSink.info.message(EPrefixInternalError, errorMsg.c_str());
            error = true;
        }
    }

    inline void setStage(EShLanguage s) { stage = s; }

    EShLanguage     stage;
    TIoMapResolver& resolver;
    TInfoSink&      infoSink;
    bool&           error;
    TVarLiveMap*    uniformVarMap[EShLangCount];
private:
    TResolverUniformAdaptor& operator=(TResolverUniformAdaptor&) = delete;
};

struct TResolverInOutAdaptor {
    TResolverInOutAdaptor(EShLanguage s, TIoMapResolver& r, TInfoSink& i, bool& e)
      : stage(s)
      , resolver(r)
      , infoSink(i)
      , error(e)
    {
    }

    inline void operator()(std::pair<const TString, TVarEntryInfo>& entKey)
    {
        TVarEntryInfo& ent = entKey.second;
        ent.clearNewAssignments();
        const bool isValid = resolver.validateInOut(ent.stage, ent);
        if (isValid) {
            resolver.resolveInOutLocation(stage, ent);
            resolver.resolveInOutComponent(stage, ent);
            resolver.resolveInOutIndex(stage, ent);
        } else {
            TString errorMsg;
            if (ent.symbol->getType().getQualifier().semanticName != nullptr) {
                errorMsg = "Invalid shader In/Out variable semantic: ";
                errorMsg += ent.symbol->getType().getQualifier().semanticName;
            } else {
                errorMsg = "Invalid shader In/Out variable: ";
                errorMsg += ent.symbol->getName();
            }
            infoSink.info.message(EPrefixInternalError, errorMsg.c_str());
            error = true;
        }
    }

    inline void setStage(EShLanguage s) { stage = s; }

    EShLanguage     stage;
    TIoMapResolver& resolver;
    TInfoSink&      infoSink;
    bool&           error;

private:
    TResolverInOutAdaptor& operator=(TResolverInOutAdaptor&) = delete;
};

// The class is used for reserving explicit uniform locations and ubo/ssbo/opaque bindings
// xxTODO: maybe this logic should be moved into the resolver's "validateInOut" and "validateUniform"

struct TSymbolValidater
{
    TSymbolValidater(TIoMapResolver& r, TInfoSink& i, TVarLiveMap* in[EShLangCount], TVarLiveMap* out[EShLangCount],
                     TVarLiveMap* uniform[EShLangCount], bool& hadError, EProfile profile, int version)
        : resolver(r)
        , infoSink(i)
        , hadError(hadError)
        , profile(profile)
        , version(version)
    {
        memcpy(inVarMaps, in, EShLangCount * (sizeof(TVarLiveMap*)));
        memcpy(outVarMaps, out, EShLangCount * (sizeof(TVarLiveMap*)));
        memcpy(uniformVarMap, uniform, EShLangCount * (sizeof(TVarLiveMap*)));

        std::map<TString, TString> anonymousMemberMap;
        std::vector<TRange> usedUniformLocation;
        std::vector<TString> usedUniformName;
        usedUniformLocation.clear();
        usedUniformName.clear();
        for (int i = 0; i < EShLangCount; i++) {
            if (uniformVarMap[i]) {
                for (auto uniformVar : *uniformVarMap[i])
                {
                    TIntermSymbol* pSymbol = uniformVar.second.symbol;
                    TQualifier qualifier = uniformVar.second.symbol->getQualifier();
                    TString symbolName = pSymbol->getAccessName();

                    // All the uniform needs multi-stage location check (block/default)
                    int uniformLocation = qualifier.layoutLocation;

                    if (uniformLocation != TQualifier::layoutLocationEnd) {
                        // Total size of current uniform, could be block, struct or other types.
                        int size = TIntermediate::computeTypeUniformLocationSize(pSymbol->getType());

                        TRange locationRange(uniformLocation, uniformLocation + size - 1);

                        // Combine location and component ranges
                        int overlapLocation = -1;
                        bool diffLocation = false;

                        // Check for collisions, except for vertex inputs on desktop targeting OpenGL
                        overlapLocation = checkLocationOverlap(locationRange, usedUniformLocation, symbolName, usedUniformName, diffLocation);

                        // Overlap locations of uniforms, regardless of components (multi stages)
                        if (overlapLocation == -1) {
                            usedUniformLocation.push_back(locationRange);
                            usedUniformName.push_back(symbolName);
                        }
                        else if (overlapLocation >= 0) {
                            if (diffLocation == true) {
                                TString err = ("Uniform location should be equal for same uniforms: " +std::to_string(overlapLocation)).c_str();
                                infoSink.info.message(EPrefixInternalError, err.c_str());
                                hadError = true;
                                break;
                            }
                            else {
                                TString err = ("Uniform location overlaps across stages: " + std::to_string(overlapLocation)).c_str();
                                infoSink.info.message(EPrefixInternalError, err.c_str());
                                hadError = true;
                                break;
                            }
                        }
                    }

                    if ((uniformVar.second.symbol->getBasicType() == EbtBlock) &&
                        IsAnonymous(uniformVar.second.symbol->getName()))
                    {
                        auto blockType = uniformVar.second.symbol->getType().getStruct();
                        for (size_t memberIdx = 0; memberIdx < blockType->size(); ++memberIdx) {
                            auto memberName = (*blockType)[memberIdx].type->getFieldName();
                            if (anonymousMemberMap.find(memberName) != anonymousMemberMap.end())
                            {
                                if (anonymousMemberMap[memberName] != uniformVar.second.symbol->getType().getTypeName())
                                {
                                    TString err = "Invalid block member name: " + memberName;
                                    infoSink.info.message(EPrefixInternalError, err.c_str());
                                    hadError = true;
                                    break;
                                }
                            }
                            else
                            {
                                anonymousMemberMap[memberName] = uniformVar.second.symbol->getType().getTypeName();
                            }
                        }
                    }
                    if (hadError)
                        break;
                }
            }
        }
    }

    // In case we need to new an intermediate, which costs too much
    int checkLocationOverlap(const TRange& locationRange, std::vector<TRange>& usedUniformLocation, const TString symbolName, std::vector<TString>& usedUniformName, bool& diffLocation)
    {
        for (size_t r = 0; r < usedUniformLocation.size(); ++r) {
            if (usedUniformName[r] == symbolName) {
                diffLocation = true;
                return (usedUniformLocation[r].start == locationRange.start &&
                        usedUniformLocation[r].last == locationRange.last)
                       ? -2 : std::max(locationRange.start, usedUniformLocation[r].start);
            }
            if (locationRange.overlap(usedUniformLocation[r])) {
                // there is a collision; pick one
                return std::max(locationRange.start, usedUniformLocation[r].start);
            }
        }

        return -1; // no collision
    }

    inline void operator()(std::pair<const TString, TVarEntryInfo>& entKey) {
        TVarEntryInfo& ent1 = entKey.second;
        TIntermSymbol* base = ent1.symbol;
        const TType& type = ent1.symbol->getType();
        const TString& name = entKey.first;
        TString mangleName1, mangleName2;
        EShLanguage stage = ent1.stage;
        EShLanguage preStage, currentStage, nextStage;

        preStage = EShLangCount;
        for (int i = stage - 1; i >= 0; i--) {
            if (inVarMaps[i] != nullptr) {
                preStage = static_cast<EShLanguage>(i);
                break;
            }
        }
        currentStage = stage;
        nextStage = EShLangCount;
        for (int i = stage + 1; i < EShLangCount; i++) {
            if (inVarMaps[i] != nullptr) {
                nextStage = static_cast<EShLanguage>(i);
                break;
            }
        }

        if (type.getQualifier().isArrayedIo(stage)) {
            TType subType(type, 0);
            subType.appendMangledName(mangleName1);
        } else {
            type.appendMangledName(mangleName1);
        }


        // basic checking that symbols match
        // more extensive checking in the link stage
        if (base->getQualifier().storage == EvqVaryingIn) {
            // validate stage in;
            if (preStage == EShLangCount)
                return;
            if (TSymbolTable::isBuiltInSymbol(base->getId()))
                return;
            if (outVarMaps[preStage] != nullptr) {
                auto ent2 = outVarMaps[preStage]->find(name);
                uint32_t location = base->getType().getQualifier().layoutLocation;
                if (ent2 == outVarMaps[preStage]->end() &&
                    location != glslang::TQualifier::layoutLocationEnd) {
                    for (auto var = outVarMaps[preStage]->begin(); var != ent2; var++) {
                        if (var->second.symbol->getType().getQualifier().layoutLocation == location) {
                            ent2 = var;
                            break;
                        }
                    }
                }
                if (ent2 != outVarMaps[preStage]->end()) {
                    auto& type1 = base->getType();
                    auto& type2 = ent2->second.symbol->getType();
                    hadError = hadError || typeCheck(&type1, &type2, name.c_str(), false);
                    if (ent2->second.symbol->getType().getQualifier().isArrayedIo(preStage)) {
                        TType subType(ent2->second.symbol->getType(), 0);
                        subType.appendMangledName(mangleName2);
                    } else {
                        ent2->second.symbol->getType().appendMangledName(mangleName2);
                    }

                    if (mangleName1 == mangleName2) {
                        // For ES 3.0 only, other versions have no such restrictions
                        // According to ES 3.0 spec: The type and presence of the interpolation qualifiers and
                        // storage qualifiers  of variables with the same name declared in all linked shaders must
                        // match, otherwise the link command will fail.
                        if (profile == EEsProfile && version == 300) {
                            // Don't need to check smooth qualifier, as it uses the default interpolation mode
                            if (ent1.stage == EShLangFragment && type1.isBuiltIn() == false) {
                                if (type1.getQualifier().flat != type2.getQualifier().flat ||
                                    type1.getQualifier().nopersp != type2.getQualifier().nopersp) {
                                    TString err = "Interpolation qualifier mismatch : " + entKey.first;
                                    infoSink.info.message(EPrefixInternalError, err.c_str());
                                    hadError = true;
                                }
                            }
                        }
                        return;
                    }
                    else {
                        // Deal with input/output pairs where one is a block member but the other is loose,
                        // e.g. with ARB_separate_shader_objects
                        if (type1.getBasicType() == EbtBlock &&
                            type1.isStruct() && !type2.isStruct()) {
                            // Iterate through block members tracking layout
                            glslang::TString name;
                            type1.getStruct()->begin()->type->appendMangledName(name);
                            if (name == mangleName2
                                && type1.getQualifier().layoutLocation == type2.getQualifier().layoutLocation) return;
                        }
                        if (type2.getBasicType() == EbtBlock &&
                            type2.isStruct() && !type1.isStruct()) {
                            // Iterate through block members tracking layout
                            glslang::TString name;
                            type2.getStruct()->begin()->type->appendMangledName(name);
                            if (name == mangleName1
                                && type1.getQualifier().layoutLocation == type2.getQualifier().layoutLocation) return;
                        }
                        TString err = "Invalid In/Out variable type : " + entKey.first;
                        infoSink.info.message(EPrefixInternalError, err.c_str());
                        hadError = true;
                    }
                }
                else if (!base->getType().isBuiltIn()) {
                    // According to spec: A link error is generated if any statically referenced input variable
                    // or block does not have a matching output
                    if (profile == EEsProfile && ent1.live) {
                        hadError = true;
                        TString errorStr = name + ": not been declare as a output variable in pre shader stage.";
                        infoSink.info.message(EPrefixError, errorStr.c_str());
                    }
                }
                return;
            }
        } else if (base->getQualifier().storage == EvqVaryingOut) {
            // validate stage out;
            if (nextStage == EShLangCount)
                return;
            if (TSymbolTable::isBuiltInSymbol(base->getId()))
                return;
            if (inVarMaps[nextStage] != nullptr) {
                auto ent2 = inVarMaps[nextStage]->find(name);
                if (ent2 != inVarMaps[nextStage]->end()) {
                    if (ent2->second.symbol->getType().getQualifier().isArrayedIo(nextStage)) {
                        TType subType(ent2->second.symbol->getType(), 0);
                        subType.appendMangledName(mangleName2);
                    } else {
                        ent2->second.symbol->getType().appendMangledName(mangleName2);
                    }
                    if (mangleName1 == mangleName2)
                        return;
                    else {
                        TString err = "Invalid In/Out variable type : " + entKey.first;
                        infoSink.info.message(EPrefixInternalError, err.c_str());
                        hadError = true;
                    }
                }
                return;
            }
        } else if (base->getQualifier().isUniformOrBuffer() && !base->getQualifier().isPushConstant()) {
            // validate uniform type;
            for (int i = 0; i < EShLangCount; i++) {
                if (i != currentStage && outVarMaps[i] != nullptr) {
                    auto ent2 = uniformVarMap[i]->find(name);
                    if (ent2 != uniformVarMap[i]->end()) {
                        ent2->second.symbol->getType().appendMangledName(mangleName2);
                        if (mangleName1 != mangleName2) {
                            ent2->second.symbol->getType().sameElementType(type);
                            TString err = "Invalid Uniform variable type : " + entKey.first;
                            infoSink.info.message(EPrefixInternalError, err.c_str());
                            hadError = true;
                        }
                        mangleName2.clear();

                        // validate instance name of blocks
                        if (hadError == false &&
                            base->getType().getBasicType() == EbtBlock &&
                            IsAnonymous(base->getName()) != IsAnonymous(ent2->second.symbol->getName())) {
                            TString err = "Matched uniform block names must also either all be lacking "
                                          "an instance name or all having an instance name: " + entKey.first;
                            infoSink.info.message(EPrefixInternalError, err.c_str());
                            hadError = true;
                        }

                        // validate uniform block member qualifier and member names
                        auto& type1 = base->getType();
                        auto& type2 = ent2->second.symbol->getType();
                        if (hadError == false && base->getType().getBasicType() == EbtBlock) {
                            hadError = hadError || typeCheck(&type1, &type2, name.c_str(), true);
                        }
                        else {
                            hadError = hadError || typeCheck(&type1, &type2, name.c_str(), false);
                        }
                    }
                    else if (base->getBasicType() == EbtBlock)
                    {
                        if (IsAnonymous(base->getName()))
                        {
                            // The name of anonymous block member can't same with default uniform variable.
                            auto blockType1 = base->getType().getStruct();
                            for (size_t memberIdx = 0; memberIdx < blockType1->size(); ++memberIdx) {
                                auto memberName = (*blockType1)[memberIdx].type->getFieldName();
                                if (uniformVarMap[i]->find(memberName) != uniformVarMap[i]->end())
                                {
                                    TString err = "Invalid Uniform variable name : " + memberName;
                                    infoSink.info.message(EPrefixInternalError, err.c_str());
                                    hadError = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    TVarLiveMap *inVarMaps[EShLangCount], *outVarMaps[EShLangCount], *uniformVarMap[EShLangCount];

    // Use for mark current shader stage for resolver
    TIoMapResolver& resolver;
    TInfoSink& infoSink;
    bool& hadError;
    EProfile profile;
    int version;

private:
    TSymbolValidater& operator=(TSymbolValidater&) = delete;

    bool qualifierCheck(const TType* const type1, const TType* const type2, const std::string& name, bool isBlock)
    {
        bool hasError = false;
        const TQualifier& qualifier1 = type1->getQualifier();
        const TQualifier& qualifier2 = type2->getQualifier();

        if (((isBlock == false) &&
            (type1->getQualifier().storage == EvqUniform && type2->getQualifier().storage == EvqUniform)) ||
            (type1->getQualifier().storage == EvqGlobal && type2->getQualifier().storage == EvqGlobal)) {
            if (qualifier1.precision != qualifier2.precision) {
                hasError = true;
                std::string errorStr = name + ": have precision conflict cross stage.";
                infoSink.info.message(EPrefixError, errorStr.c_str());
            }
            if (qualifier1.hasFormat() && qualifier2.hasFormat()) {
                if (qualifier1.layoutFormat != qualifier2.layoutFormat) {
                    hasError = true;
                    std::string errorStr = name + ": have layout format conflict cross stage.";
                    infoSink.info.message(EPrefixError, errorStr.c_str());
                }

            }
        }

        if (isBlock == true) {
            if (qualifier1.layoutPacking != qualifier2.layoutPacking) {
                hasError = true;
                std::string errorStr = name + ": have layoutPacking conflict cross stage.";
                infoSink.info.message(EPrefixError, errorStr.c_str());
            }
            if (qualifier1.layoutMatrix != qualifier2.layoutMatrix) {
                hasError = true;
                std::string errorStr = name + ": have layoutMatrix conflict cross stage.";
                infoSink.info.message(EPrefixError, errorStr.c_str());
            }
            if (qualifier1.layoutOffset != qualifier2.layoutOffset) {
                hasError = true;
                std::string errorStr = name + ": have layoutOffset conflict cross stage.";
                infoSink.info.message(EPrefixError, errorStr.c_str());
            }
            if (qualifier1.layoutAlign != qualifier2.layoutAlign) {
                hasError = true;
                std::string errorStr = name + ": have layoutAlign conflict cross stage.";
                infoSink.info.message(EPrefixError, errorStr.c_str());
            }
        }

        return hasError;
    }

    bool typeCheck(const TType* const type1, const TType* const type2, const std::string& name, bool isBlock)
    {
        bool hasError = false;
        if (!(type1->isStruct() && type2->isStruct())) {
            hasError = hasError || qualifierCheck(type1, type2, name, isBlock);
        }
        else {
            if (type1->getBasicType() == EbtBlock && type2->getBasicType() == EbtBlock)
                isBlock = true;
            const TTypeList* typeList1 = type1->getStruct();
            const TTypeList* typeList2 = type2->getStruct();

            std::string newName = name;
            size_t memberCount = typeList1->size();
            size_t index2 = 0;
            for (size_t index = 0; index < memberCount; index++, index2++) {
                // Skip inactive member
                if (typeList1->at(index).type->getBasicType() == EbtVoid)
                    continue;
                while (index2 < typeList2->size() && typeList2->at(index2).type->getBasicType() == EbtVoid) {
                    ++index2;
                }

                // TypeList1 has more members in list
                if (index2 == typeList2->size()) {
                    std::string errorStr = name + ": struct mismatch.";
                    infoSink.info.message(EPrefixError, errorStr.c_str());
                    hasError = true;
                    break;
                }

                if (typeList1->at(index).type->getFieldName() != typeList2->at(index2).type->getFieldName()) {
                    std::string errorStr = name + ": member name mismatch.";
                    infoSink.info.message(EPrefixError, errorStr.c_str());
                    hasError = true;
                }
                else {
                    newName = typeList1->at(index).type->getFieldName().c_str();
                }
                hasError = hasError || typeCheck(typeList1->at(index).type, typeList2->at(index2).type, newName, isBlock);
            }

            while (index2 < typeList2->size())
            {
                // TypeList2 has more members
                if (typeList2->at(index2).type->getBasicType() != EbtVoid) {
                    std::string errorStr = name + ": struct mismatch.";
                    infoSink.info.message(EPrefixError, errorStr.c_str());
                    hasError = true;
                    break;
                }
                ++index2;
            }
        }
        return hasError;
    }
};

struct TSlotCollector {
    TSlotCollector(TIoMapResolver& r, TInfoSink& i) : resolver(r), infoSink(i) { }

    inline void operator()(std::pair<const TString, TVarEntryInfo>& entKey) {
        resolver.reserverStorageSlot(entKey.second, infoSink);
        resolver.reserverResourceSlot(entKey.second, infoSink);
    }
    TIoMapResolver& resolver;
    TInfoSink& infoSink;

private:
    TSlotCollector& operator=(TSlotCollector&) = delete;
};

TDefaultIoResolverBase::TDefaultIoResolverBase(const TIntermediate& intermediate)
    : referenceIntermediate(intermediate)
    , nextUniformLocation(intermediate.getUniformLocationBase())
    , nextInputLocation(0)
    , nextOutputLocation(0)
{
    memset(stageMask, false, sizeof(bool) * (EShLangCount + 1));
    memset(stageIntermediates, 0, sizeof(TIntermediate*) * (EShLangCount));
    stageIntermediates[intermediate.getStage()] = &intermediate;
}

int TDefaultIoResolverBase::getBaseBinding(EShLanguage stage, TResourceType res, unsigned int set) const {
    return stageIntermediates[stage] ? selectBaseBinding(stageIntermediates[stage]->getShiftBinding(res), stageIntermediates[stage]->getShiftBindingForSet(res, set))
                                     : selectBaseBinding(referenceIntermediate.getShiftBinding(res), referenceIntermediate.getShiftBindingForSet(res, set));
}

const std::vector<std::string>& TDefaultIoResolverBase::getResourceSetBinding(EShLanguage stage) const {
    return stageIntermediates[stage] ? stageIntermediates[stage]->getResourceSetBinding()
                                     : referenceIntermediate.getResourceSetBinding();
}

bool TDefaultIoResolverBase::doAutoBindingMapping() const { return referenceIntermediate.getAutoMapBindings(); }

bool TDefaultIoResolverBase::doAutoLocationMapping() const { return referenceIntermediate.getAutoMapLocations(); }

TDefaultIoResolverBase::TSlotSet::iterator TDefaultIoResolverBase::findSlot(int set, int slot) {
    return std::lower_bound(slots[set].begin(), slots[set].end(), slot);
}

bool TDefaultIoResolverBase::checkEmpty(int set, int slot) {
    TSlotSet::iterator at = findSlot(set, slot);
    return ! (at != slots[set].end() && *at == slot);
}

int TDefaultIoResolverBase::reserveSlot(int set, int slot, int size) {
    TSlotSet::iterator at = findSlot(set, slot);
    // tolerate aliasing, by not double-recording aliases
    // (policy about appropriateness of the alias is higher up)
    for (int i = 0; i < size; i++) {
        if (at == slots[set].end() || *at != slot + i)
            at = slots[set].insert(at, slot + i);
        ++at;
    }
    return slot;
}

int TDefaultIoResolverBase::getFreeSlot(int set, int base, int size) {
    TSlotSet::iterator at = findSlot(set, base);
    if (at == slots[set].end())
        return reserveSlot(set, base, size);
    // look for a big enough gap
    for (; at != slots[set].end(); ++at) {
        if (*at - base >= size)
            break;
        base = *at + 1;
    }
    return reserveSlot(set, base, size);
}

int TDefaultIoResolverBase::resolveSet(EShLanguage stage, TVarEntryInfo& ent) {
    const TType& type = ent.symbol->getType();
    if (type.getQualifier().hasSet()) {
        return ent.newSet = type.getQualifier().layoutSet;
    }
    // If a command line or API option requested a single descriptor set, use that (if not overrided by spaceN)
    if (getResourceSetBinding(stage).size() == 1) {
        return ent.newSet = atoi(getResourceSetBinding(stage)[0].c_str());
    }
    return ent.newSet = 0;
}

int TDefaultIoResolverBase::resolveUniformLocation(EShLanguage /*stage*/, TVarEntryInfo& ent) {
    const TType& type = ent.symbol->getType();
    const char* name =  ent.symbol->getAccessName().c_str();
    // kick out of not doing this
    if (! doAutoLocationMapping()) {
        return ent.newLocation = -1;
    }
    // no locations added if already present, a built-in variable, a block, or an opaque
    if (type.getQualifier().hasLocation() || type.isBuiltIn() || type.getBasicType() == EbtBlock ||
        type.isAtomic() || type.isSpirvType() || (type.containsOpaque() && referenceIntermediate.getSpv().openGl == 0)) {
        return ent.newLocation = -1;
    }
    // no locations on blocks of built-in variables
    if (type.isStruct()) {
        if (type.getStruct()->size() < 1) {
            return ent.newLocation = -1;
        }
        if ((*type.getStruct())[0].type->isBuiltIn()) {
            return ent.newLocation = -1;
        }
    }
    int location = referenceIntermediate.getUniformLocationOverride(name);
    if (location != -1) {
        return ent.newLocation = location;
    }
    location = nextUniformLocation;
    nextUniformLocation += TIntermediate::computeTypeUniformLocationSize(type);
    return ent.newLocation = location;
}

int TDefaultIoResolverBase::resolveInOutLocation(EShLanguage stage, TVarEntryInfo& ent) {
    const TType& type = ent.symbol->getType();
    // kick out of not doing this
    if (! doAutoLocationMapping()) {
        return ent.newLocation = -1;
    }

    // no locations added if already present, a built-in variable, or a variable with SPIR-V decorate
    if (type.getQualifier().hasLocation() || type.isBuiltIn() || type.getQualifier().hasSpirvDecorate()) {
        return ent.newLocation = -1;
    }

    // no locations on blocks of built-in variables
    if (type.isStruct()) {
        if (type.getStruct()->size() < 1) {
            return ent.newLocation = -1;
        }
        if ((*type.getStruct())[0].type->isBuiltIn()) {
            return ent.newLocation = -1;
        }
    }
    // point to the right input or output location counter
    int& nextLocation = type.getQualifier().isPipeInput() ? nextInputLocation : nextOutputLocation;
    // Placeholder. This does not do proper cross-stage lining up, nor
    // work with mixed location/no-location declarations.
    int location = nextLocation;
    int typeLocationSize;
    // Don’t take into account the outer-most array if the stage’s
    // interface is automatically an array.
    typeLocationSize = computeTypeLocationSize(type, stage);
    nextLocation += typeLocationSize;
    return ent.newLocation = location;
}

int TDefaultIoResolverBase::resolveInOutComponent(EShLanguage /*stage*/, TVarEntryInfo& ent) {
    return ent.newComponent = -1;
}

int TDefaultIoResolverBase::resolveInOutIndex(EShLanguage /*stage*/, TVarEntryInfo& ent) { return ent.newIndex = -1; }

uint32_t TDefaultIoResolverBase::computeTypeLocationSize(const TType& type, EShLanguage stage) {
    int typeLocationSize;
    // Don’t take into account the outer-most array if the stage’s
    // interface is automatically an array.
    if (type.getQualifier().isArrayedIo(stage)) {
        TType elementType(type, 0);
        typeLocationSize = TIntermediate::computeTypeLocationSize(elementType, stage);
    } else {
        typeLocationSize = TIntermediate::computeTypeLocationSize(type, stage);
    }
    return typeLocationSize;
}

//TDefaultGlslIoResolver
TResourceType TDefaultGlslIoResolver::getResourceType(const glslang::TType& type) {
    if (isImageType(type)) {
        return EResImage;
    }
    if (isTextureType(type)) {
        return EResTexture;
    }
    if (isSsboType(type)) {
        return EResSsbo;
    }
    if (isSamplerType(type)) {
        return EResSampler;
    }
    if (isUboType(type)) {
        return EResUbo;
    }
    return EResCount;
}

TDefaultGlslIoResolver::TDefaultGlslIoResolver(const TIntermediate& intermediate)
    : TDefaultIoResolverBase(intermediate)
    , preStage(EShLangCount)
    , currentStage(EShLangCount)
{ }

int TDefaultGlslIoResolver::resolveInOutLocation(EShLanguage stage, TVarEntryInfo& ent) {
    const TType& type = ent.symbol->getType();
    const TString& name = ent.symbol->getAccessName();
    if (currentStage != stage) {
        preStage = currentStage;
        currentStage = stage;
    }
    // kick out if not doing this
    if (! doAutoLocationMapping()) {
        return ent.newLocation = -1;
    }
    // expand the location to each element if the symbol is a struct or array
    if (type.getQualifier().hasLocation()) {
        return ent.newLocation = type.getQualifier().layoutLocation;
    }
    // no locations added if already present, a built-in variable, or a variable with SPIR-V decorate
    if (type.isBuiltIn() || type.getQualifier().hasSpirvDecorate()) {
        return ent.newLocation = -1;
    }
    // no locations on blocks of built-in variables
    if (type.isStruct()) {
        if (type.getStruct()->size() < 1) {
            return ent.newLocation = -1;
        }
        if ((*type.getStruct())[0].type->isBuiltIn()) {
            return ent.newLocation = -1;
        }
    }
    int typeLocationSize = computeTypeLocationSize(type, stage);
    int location = type.getQualifier().layoutLocation;
    bool hasLocation = false;
    EShLanguage keyStage(EShLangCount);
    TStorageQualifier storage;
    storage = EvqInOut;
    if (type.getQualifier().isPipeInput()) {
        // If this symbol is a input, search pre stage's out
        keyStage = preStage;
    }
    if (type.getQualifier().isPipeOutput()) {
        // If this symbol is a output, search next stage's in
        keyStage = currentStage;
    }
    // The in/out in current stage is not declared with location, but it is possible declared
    // with explicit location in other stages, find the storageSlotMap firstly to check whether
    // the in/out has location
    int resourceKey = buildStorageKey(keyStage, storage);
    if (! storageSlotMap[resourceKey].empty()) {
        TVarSlotMap::iterator iter = storageSlotMap[resourceKey].find(name);
        if (iter != storageSlotMap[resourceKey].end()) {
            // If interface resource be found, set it has location and this symbol's new location
            // equal the symbol's explicit location declaration in pre or next stage.
            //
            // vs:    out vec4 a;
            // fs:    layout(..., location = 3,...) in vec4 a;
            hasLocation = true;
            location = iter->second;
            // if we want deal like that:
            // vs:    layout(location=4) out vec4 a;
            //        out vec4 b;
            //
            // fs:    in vec4 a;
            //        layout(location = 4) in vec4 b;
            // we need retraverse the map.
        }
        if (! hasLocation) {
            // If interface resource note found, It's mean the location in two stage are both implicit declarat.
            // So we should find a new slot for this interface.
            //
            // vs: out vec4 a;
            // fs: in vec4 a;
            location = getFreeSlot(resourceKey, 0, typeLocationSize);
            storageSlotMap[resourceKey][name] = location;
        }
    } else {
        // the first interface declarated in a program.
        TVarSlotMap varSlotMap;
        location = getFreeSlot(resourceKey, 0, typeLocationSize);
        varSlotMap[name] = location;
        storageSlotMap[resourceKey] = varSlotMap;
    }
    //Update location
    return ent.newLocation = location;
}

int TDefaultGlslIoResolver::resolveUniformLocation(EShLanguage /*stage*/, TVarEntryInfo& ent) {
    const TType& type = ent.symbol->getType();
    const TString& name = ent.symbol->getAccessName();
    // kick out of not doing this
    if (! doAutoLocationMapping()) {
        return ent.newLocation = -1;
    }
    // expand the location to each element if the symbol is a struct or array
    if (type.getQualifier().hasLocation() && (type.isStruct() || type.isArray())) {
        return ent.newLocation = type.getQualifier().layoutLocation;
    } else {
        // no locations added if already present, a built-in variable, a block, or an opaque
        if (type.getQualifier().hasLocation() || type.isBuiltIn() || type.getBasicType() == EbtBlock ||
            type.isAtomic() || type.isSpirvType() ||
            (type.containsOpaque() && referenceIntermediate.getSpv().openGl == 0)) {
            return ent.newLocation = -1;
        }
        // no locations on blocks of built-in variables
        if (type.isStruct()) {
            if (type.getStruct()->size() < 1) {
                return ent.newLocation = -1;
            }
            if ((*type.getStruct())[0].type->isBuiltIn()) {
                return ent.newLocation = -1;
            }
        }
    }
    int location = referenceIntermediate.getUniformLocationOverride(name.c_str());
    if (location != -1) {
        return ent.newLocation = location;
    }

    int size = TIntermediate::computeTypeUniformLocationSize(type);

    // The uniform in current stage is not declared with location, but it is possible declared
    // with explicit location in other stages, find the storageSlotMap firstly to check whether
    // the uniform has location
    bool hasLocation = false;
    int resourceKey = buildStorageKey(EShLangCount, EvqUniform);
    TVarSlotMap& slotMap = storageSlotMap[resourceKey];
    // Check dose shader program has uniform resource
    if (! slotMap.empty()) {
        // If uniform resource not empty, try find a same name uniform
        TVarSlotMap::iterator iter = slotMap.find(name);
        if (iter != slotMap.end()) {
            // If uniform resource be found, set it has location and this symbol's new location
            // equal the uniform's explicit location declaration in other stage.
            //
            // vs:    uniform vec4 a;
            // fs:    layout(..., location = 3,...) uniform vec4 a;
            hasLocation = true;
            location = iter->second;
        }
        if (! hasLocation) {
            // No explicit location declaration in other stage.
            // So we should find a new slot for this uniform.
            //
            // vs:    uniform vec4 a;
            // fs:    uniform vec4 a;
            location = getFreeSlot(resourceKey, 0, computeTypeLocationSize(type, currentStage));
            storageSlotMap[resourceKey][name] = location;
        }
    } else {
        // the first uniform declaration in a program.
        TVarSlotMap varSlotMap;
        location = getFreeSlot(resourceKey, 0, size);
        varSlotMap[name] = location;
        storageSlotMap[resourceKey] = varSlotMap;
    }
    return ent.newLocation = location;
}

int TDefaultGlslIoResolver::resolveBinding(EShLanguage stage, TVarEntryInfo& ent) {
    const TType& type = ent.symbol->getType();
    const TString& name = ent.symbol->getAccessName();
    // On OpenGL arrays of opaque types take a separate binding for each element
    int numBindings = referenceIntermediate.getSpv().openGl != 0 && type.isSizedArray() ? type.getCumulativeArraySize() : 1;
    TResourceType resource = getResourceType(type);
    // don't need to handle uniform symbol, it will be handled in resolveUniformLocation
    if (resource == EResUbo && type.getBasicType() != EbtBlock) {
        return ent.newBinding = -1;
    }
    // There is no 'set' qualifier in OpenGL shading language, each resource has its own
    // binding name space, so remap the 'set' to resource type which make each resource
    // binding is valid from 0 to MAX_XXRESOURCE_BINDINGS
    int set = referenceIntermediate.getSpv().openGl != 0 ? resource : ent.newSet;
    int resourceKey = set;
    if (resource < EResCount) {
        if (type.getQualifier().hasBinding()) {
            int newBinding = reserveSlot(resourceKey, getBaseBinding(stage, resource, set) + type.getQualifier().layoutBinding, numBindings);
            return ent.newBinding = newBinding;

        } else {
            // The resource in current stage is not declared with binding, but it is possible declared
            // with explicit binding in other stages, find the resourceSlotMap firstly to check whether
            // the resource has binding, don't need to allocate if it already has a binding
            bool hasBinding = false;
            ent.newBinding = -1; // leave as -1 if it isn't set below

            if (! resourceSlotMap[resourceKey].empty()) {
                TVarSlotMap::iterator iter = resourceSlotMap[resourceKey].find(name);
                if (iter != resourceSlotMap[resourceKey].end()) {
                    hasBinding = true;
                    ent.newBinding = iter->second;
                }
            }
            if (!hasBinding && (ent.live && doAutoBindingMapping())) {
                // find free slot, the caller did make sure it passes all vars with binding
                // first and now all are passed that do not have a binding and needs one
                int binding = getFreeSlot(resourceKey, getBaseBinding(stage, resource, set), numBindings);
                resourceSlotMap[resourceKey][name] = binding;
                ent.newBinding = binding;
            }
            return ent.newBinding;
        }
    }
    return ent.newBinding = -1;
}

void TDefaultGlslIoResolver::beginResolve(EShLanguage stage) {
    // reset stage state
    if (stage == EShLangCount)
        preStage = currentStage = stage;
    // update stage state
    else if (currentStage != stage) {
        preStage = currentStage;
        currentStage = stage;
    }
}

void TDefaultGlslIoResolver::endResolve(EShLanguage /*stage*/) {
    // TODO nothing
}

void TDefaultGlslIoResolver::beginCollect(EShLanguage stage) {
    // reset stage state
    if (stage == EShLangCount)
        preStage = currentStage = stage;
    // update stage state
    else if (currentStage != stage) {
        preStage = currentStage;
        currentStage = stage;
    }
}

void TDefaultGlslIoResolver::endCollect(EShLanguage /*stage*/) {
    // TODO nothing
}

void TDefaultGlslIoResolver::reserverStorageSlot(TVarEntryInfo& ent, TInfoSink& infoSink) {
    const TType& type = ent.symbol->getType();
    const TString& name = ent.symbol->getAccessName();
    TStorageQualifier storage = type.getQualifier().storage;
    EShLanguage stage(EShLangCount);
    switch (storage) {
    case EvqUniform:
        if (type.getBasicType() != EbtBlock && type.getQualifier().hasLocation()) {
            //
            // Reserve the slots for the uniforms who has explicit location
            int storageKey = buildStorageKey(EShLangCount, EvqUniform);
            int location = type.getQualifier().layoutLocation;
            TVarSlotMap& varSlotMap = storageSlotMap[storageKey];
            TVarSlotMap::iterator iter = varSlotMap.find(name);
            if (iter == varSlotMap.end()) {
                int numLocations = TIntermediate::computeTypeUniformLocationSize(type);
                reserveSlot(storageKey, location, numLocations);
                varSlotMap[name] = location;
            } else {
                // Allocate location by name for OpenGL driver, so the uniform in different
                // stages should be declared with the same location
                if (iter->second != location) {
                    TString errorMsg = "Invalid location: " + name;
                    infoSink.info.message(EPrefixInternalError, errorMsg.c_str());
                    hasError = true;
                }
            }
        }
        break;
    case EvqVaryingIn:
    case EvqVaryingOut:
        //
        // Reserve the slots for the inout who has explicit location
        if (type.getQualifier().hasLocation()) {
            stage = storage == EvqVaryingIn ? preStage : stage;
            stage = storage == EvqVaryingOut ? currentStage : stage;
            int storageKey = buildStorageKey(stage, EvqInOut);
            int location = type.getQualifier().layoutLocation;
            TVarSlotMap& varSlotMap = storageSlotMap[storageKey];
            TVarSlotMap::iterator iter = varSlotMap.find(name);
            if (iter == varSlotMap.end()) {
                int numLocations = TIntermediate::computeTypeUniformLocationSize(type);
                reserveSlot(storageKey, location, numLocations);
                varSlotMap[name] = location;
            } else {
                // Allocate location by name for OpenGL driver, so the uniform in different
                // stages should be declared with the same location
                if (iter->second != location) {
                    TString errorMsg = "Invalid location: " + name;
                    infoSink.info.message(EPrefixInternalError, errorMsg.c_str());
                    hasError = true;
                }
            }
        }
        break;
    default:
        break;
    }
}

void TDefaultGlslIoResolver::reserverResourceSlot(TVarEntryInfo& ent, TInfoSink& infoSink) {
    const TType& type = ent.symbol->getType();
    const TString& name = ent.symbol->getAccessName();
    TResourceType resource = getResourceType(type);
    int set = referenceIntermediate.getSpv().openGl != 0 ? resource : resolveSet(ent.stage, ent);
    int resourceKey = set;

    if (type.getQualifier().hasBinding()) {
        TVarSlotMap& varSlotMap = resourceSlotMap[resourceKey];
        TVarSlotMap::iterator iter = varSlotMap.find(name);
        int binding = type.getQualifier().layoutBinding + getBaseBinding(ent.stage, resource, set);

        if (iter == varSlotMap.end()) {
            // Reserve the slots for the ubo, ssbo and opaques who has explicit binding
            int numBindings = referenceIntermediate.getSpv().openGl != 0 && type.isSizedArray() ? type.getCumulativeArraySize() : 1;
            varSlotMap[name] = binding;
            reserveSlot(resourceKey, binding, numBindings);
        } else {
            // Allocate binding by name for OpenGL driver, so the resource in different
            // stages should be declared with the same binding
            if (iter->second != binding) {
                TString errorMsg = "Invalid binding: " + name;
                infoSink.info.message(EPrefixInternalError, errorMsg.c_str());
                hasError = true;
            }
        }
    }
}

//TDefaultGlslIoResolver end

/*
 * Basic implementation of glslang::TIoMapResolver that replaces the
 * previous offset behavior.
 * It does the same, uses the offsets for the corresponding uniform
 * types. Also respects the EOptionAutoMapBindings flag and binds
 * them if needed.
 */
/*
 * Default resolver
 */
struct TDefaultIoResolver : public TDefaultIoResolverBase {
    TDefaultIoResolver(const TIntermediate& intermediate) : TDefaultIoResolverBase(intermediate) { }

    bool validateBinding(EShLanguage /*stage*/, TVarEntryInfo& /*ent*/) override { return true; }

    TResourceType getResourceType(const glslang::TType& type) override {
        if (isImageType(type)) {
            return EResImage;
        }
        if (isTextureType(type)) {
            return EResTexture;
        }
        if (isSsboType(type)) {
            return EResSsbo;
        }
        if (isSamplerType(type)) {
            return EResSampler;
        }
        if (isUboType(type)) {
            return EResUbo;
        }
        return EResCount;
    }

    int resolveBinding(EShLanguage stage, TVarEntryInfo& ent) override {
        const TType& type = ent.symbol->getType();
        const int set = getLayoutSet(type);
        // On OpenGL arrays of opaque types take a seperate binding for each element
        int numBindings = referenceIntermediate.getSpv().openGl != 0 && type.isSizedArray() ? type.getCumulativeArraySize() : 1;
        TResourceType resource = getResourceType(type);
        if (resource < EResCount) {
            if (type.getQualifier().hasBinding()) {
                return ent.newBinding = reserveSlot(
                           set, getBaseBinding(stage, resource, set) + type.getQualifier().layoutBinding, numBindings);
            } else if (ent.live && doAutoBindingMapping()) {
                // find free slot, the caller did make sure it passes all vars with binding
                // first and now all are passed that do not have a binding and needs one
                return ent.newBinding = getFreeSlot(set, getBaseBinding(stage, resource, set), numBindings);
            }
        }
        return ent.newBinding = -1;
    }
};

#ifdef ENABLE_HLSL
/********************************************************************************
The following IO resolver maps types in HLSL register space, as follows:

t - for shader resource views (SRV)
   TEXTURE1D
   TEXTURE1DARRAY
   TEXTURE2D
   TEXTURE2DARRAY
   TEXTURE3D
   TEXTURECUBE
   TEXTURECUBEARRAY
   TEXTURE2DMS
   TEXTURE2DMSARRAY
   STRUCTUREDBUFFER
   BYTEADDRESSBUFFER
   BUFFER
   TBUFFER

s - for samplers
   SAMPLER
   SAMPLER1D
   SAMPLER2D
   SAMPLER3D
   SAMPLERCUBE
   SAMPLERSTATE
   SAMPLERCOMPARISONSTATE

u - for unordered access views (UAV)
   RWBYTEADDRESSBUFFER
   RWSTRUCTUREDBUFFER
   APPENDSTRUCTUREDBUFFER
   CONSUMESTRUCTUREDBUFFER
   RWBUFFER
   RWTEXTURE1D
   RWTEXTURE1DARRAY
   RWTEXTURE2D
   RWTEXTURE2DARRAY
   RWTEXTURE3D

b - for constant buffer views (CBV)
   CBUFFER
   CONSTANTBUFFER
 ********************************************************************************/
struct TDefaultHlslIoResolver : public TDefaultIoResolverBase {
    TDefaultHlslIoResolver(const TIntermediate& intermediate) : TDefaultIoResolverBase(intermediate) { }

    bool validateBinding(EShLanguage /*stage*/, TVarEntryInfo& /*ent*/) override { return true; }

    TResourceType getResourceType(const glslang::TType& type) override {
        if (isUavType(type)) {
            return EResUav;
        }
        if (isSrvType(type)) {
            return EResTexture;
        }
        if (isSamplerType(type)) {
            return EResSampler;
        }
        if (isUboType(type)) {
            return EResUbo;
        }
        return EResCount;
    }

    int resolveBinding(EShLanguage stage, TVarEntryInfo& ent) override {
        const TType& type = ent.symbol->getType();
        const int set = getLayoutSet(type);
        TResourceType resource = getResourceType(type);
        if (resource < EResCount) {
            if (type.getQualifier().hasBinding()) {
                return ent.newBinding = reserveSlot(set, getBaseBinding(stage, resource, set) + type.getQualifier().layoutBinding);
            } else if (ent.live && doAutoBindingMapping()) {
                // find free slot, the caller did make sure it passes all vars with binding
                // first and now all are passed that do not have a binding and needs one
                return ent.newBinding = getFreeSlot(set, getBaseBinding(stage, resource, set));
            }
        }
        return ent.newBinding = -1;
    }
};
#endif

// Map I/O variables to provided offsets, and make bindings for
// unbound but live variables.
//
// Returns false if the input is too malformed to do this.
bool TIoMapper::addStage(EShLanguage stage, TIntermediate& intermediate, TInfoSink& infoSink, TIoMapResolver* resolver) {
    bool somethingToDo = ! intermediate.getResourceSetBinding().empty() || intermediate.getAutoMapBindings() ||
                         intermediate.getAutoMapLocations();
    // Restrict the stricter condition to further check 'somethingToDo' only if 'somethingToDo' has not been set, reduce
    // unnecessary or insignificant for-loop operation after 'somethingToDo' have been true.
    for (int res = 0; (res < EResCount && !somethingToDo); ++res) {
        somethingToDo = somethingToDo || (intermediate.getShiftBinding(TResourceType(res)) != 0) ||
                        intermediate.hasShiftBindingForSet(TResourceType(res));
    }
    if (! somethingToDo && resolver == nullptr)
        return true;
    if (intermediate.getNumEntryPoints() != 1 || intermediate.isRecursive())
        return false;
    TIntermNode* root = intermediate.getTreeRoot();
    if (root == nullptr)
        return false;
    // if no resolver is provided, use the default resolver with the given shifts and auto map settings
    TDefaultIoResolver defaultResolver(intermediate);
#ifdef ENABLE_HLSL
    TDefaultHlslIoResolver defaultHlslResolver(intermediate);
    if (resolver == nullptr) {
        // TODO: use a passed in IO mapper for this
        if (intermediate.usingHlslIoMapping())
            resolver = &defaultHlslResolver;
        else
            resolver = &defaultResolver;
    }
#else
    resolver = &defaultResolver;
#endif
    resolver->addStage(stage, intermediate);

    TVarLiveMap inVarMap, outVarMap, uniformVarMap;
    TVarLiveVector inVector, outVector, uniformVector;
    TVarGatherTraverser iter_binding_all(intermediate, true, inVarMap, outVarMap, uniformVarMap);
    TVarGatherTraverser iter_binding_live(intermediate, false, inVarMap, outVarMap, uniformVarMap);
    root->traverse(&iter_binding_all);
    iter_binding_live.pushFunction(intermediate.getEntryPointMangledName().c_str());
    while (! iter_binding_live.destinations.empty()) {
        TIntermNode* destination = iter_binding_live.destinations.back();
        iter_binding_live.destinations.pop_back();
        destination->traverse(&iter_binding_live);
    }

    // sort entries by priority. see TVarEntryInfo::TOrderByPriority for info.
    for (auto& var : inVarMap) { inVector.push_back(var); }
    std::sort(inVector.begin(), inVector.end(), [](const TVarLivePair& p1, const TVarLivePair& p2) -> bool {
        return TVarEntryInfo::TOrderByPriority()(p1.second, p2.second);
    });
    for (auto& var : outVarMap) { outVector.push_back(var); }
    std::sort(outVector.begin(), outVector.end(), [](const TVarLivePair& p1, const TVarLivePair& p2) -> bool {
        return TVarEntryInfo::TOrderByPriority()(p1.second, p2.second);
    });
    for (auto& var : uniformVarMap) { uniformVector.push_back(var); }
    std::sort(uniformVector.begin(), uniformVector.end(), [](const TVarLivePair& p1, const TVarLivePair& p2) -> bool {
        return TVarEntryInfo::TOrderByPriority()(p1.second, p2.second);
    });
    bool hadError = false;
    TVarLiveMap* dummyUniformVarMap[EShLangCount] = {};
    TNotifyInOutAdaptor inOutNotify(stage, *resolver);
    TNotifyUniformAdaptor uniformNotify(stage, *resolver);
    TResolverUniformAdaptor uniformResolve(stage, *resolver, dummyUniformVarMap, infoSink, hadError);
    TResolverInOutAdaptor inOutResolve(stage, *resolver, infoSink, hadError);
    resolver->beginNotifications(stage);
    std::for_each(inVector.begin(), inVector.end(), inOutNotify);
    std::for_each(outVector.begin(), outVector.end(), inOutNotify);
    std::for_each(uniformVector.begin(), uniformVector.end(), uniformNotify);
    resolver->endNotifications(stage);
    resolver->beginResolve(stage);
    for (auto& var : inVector) { inOutResolve(var); }
    std::for_each(inVector.begin(), inVector.end(), [&inVarMap](TVarLivePair p) {
        auto at = inVarMap.find(p.second.symbol->getAccessName());
        if (at != inVarMap.end() && p.second.id == at->second.id)
            at->second = p.second;
    });
    for (auto& var : outVector) { inOutResolve(var); }
    std::for_each(outVector.begin(), outVector.end(), [&outVarMap](TVarLivePair p) {
        auto at = outVarMap.find(p.second.symbol->getAccessName());
        if (at != outVarMap.end() && p.second.id == at->second.id)
            at->second = p.second;
    });
    std::for_each(uniformVector.begin(), uniformVector.end(), uniformResolve);
    std::for_each(uniformVector.begin(), uniformVector.end(), [&uniformVarMap](TVarLivePair p) {
        auto at = uniformVarMap.find(p.second.symbol->getAccessName());
        if (at != uniformVarMap.end() && p.second.id == at->second.id)
            at->second = p.second;
    });
    resolver->endResolve(stage);
    if (!hadError) {
        TVarSetTraverser iter_iomap(intermediate, inVarMap, outVarMap, uniformVarMap);
        root->traverse(&iter_iomap);
    }
    return !hadError;
}

// Map I/O variables to provided offsets, and make bindings for
// unbound but live variables.
//
// Returns false if the input is too malformed to do this.
bool TGlslIoMapper::addStage(EShLanguage stage, TIntermediate& intermediate, TInfoSink& infoSink, TIoMapResolver* resolver) {
    bool somethingToDo = !intermediate.getResourceSetBinding().empty() ||
        intermediate.getAutoMapBindings() ||
        intermediate.getAutoMapLocations();

    // Profile and version are use for symbol validate.
    profile = intermediate.getProfile();
    version = intermediate.getVersion();

    // Restrict the stricter condition to further check 'somethingToDo' only if 'somethingToDo' has not been set, reduce
    // unnecessary or insignificant for-loop operation after 'somethingToDo' have been true.
    for (int res = 0; (res < EResCount && !somethingToDo); ++res) {
        somethingToDo = somethingToDo || (intermediate.getShiftBinding(TResourceType(res)) != 0) ||
                        intermediate.hasShiftBindingForSet(TResourceType(res));
    }
    if (! somethingToDo && resolver == nullptr) {
        return true;
    }
    if (intermediate.getNumEntryPoints() != 1 || intermediate.isRecursive()) {
        return false;
    }
    TIntermNode* root = intermediate.getTreeRoot();
    if (root == nullptr) {
        return false;
    }
    // if no resolver is provided, use the default resolver with the given shifts and auto map settings
    TDefaultGlslIoResolver defaultResolver(intermediate);
#ifdef ENABLE_HLSL
    TDefaultHlslIoResolver defaultHlslResolver(intermediate);
    if (resolver == nullptr) {
        // TODO: use a passed in IO mapper for this
        if (intermediate.usingHlslIoMapping())
            resolver = &defaultHlslResolver;
        else
            resolver = &defaultResolver;
    }
#else
    if (resolver == nullptr) {
        resolver = &defaultResolver;
    }
#endif
    resolver->addStage(stage, intermediate);
    inVarMaps[stage] = new TVarLiveMap(); outVarMaps[stage] = new TVarLiveMap(); uniformVarMap[stage] = new TVarLiveMap();
    TVarGatherTraverser iter_binding_all(intermediate, true, *inVarMaps[stage], *outVarMaps[stage],
                                         *uniformVarMap[stage]);
    TVarGatherTraverser iter_binding_live(intermediate, false, *inVarMaps[stage], *outVarMaps[stage],
                                          *uniformVarMap[stage]);
    root->traverse(&iter_binding_all);
    iter_binding_live.pushFunction(intermediate.getEntryPointMangledName().c_str());
    while (! iter_binding_live.destinations.empty()) {
        TIntermNode* destination = iter_binding_live.destinations.back();
        iter_binding_live.destinations.pop_back();
        destination->traverse(&iter_binding_live);
    }

    TNotifyInOutAdaptor inOutNotify(stage, *resolver);
    TNotifyUniformAdaptor uniformNotify(stage, *resolver);
    // Resolve current stage input symbol location with previous stage output here,
    // uniform symbol, ubo, ssbo and opaque symbols are per-program resource,
    // will resolve uniform symbol location and ubo/ssbo/opaque binding in doMap()
    resolver->beginNotifications(stage);
    std::for_each(inVarMaps[stage]->begin(), inVarMaps[stage]->end(), inOutNotify);
    std::for_each(outVarMaps[stage]->begin(), outVarMaps[stage]->end(), inOutNotify);
    std::for_each(uniformVarMap[stage]->begin(), uniformVarMap[stage]->end(), uniformNotify);
    resolver->endNotifications(stage);
    TSlotCollector slotCollector(*resolver, infoSink);
    resolver->beginCollect(stage);
    std::for_each(inVarMaps[stage]->begin(), inVarMaps[stage]->end(), slotCollector);
    std::for_each(outVarMaps[stage]->begin(), outVarMaps[stage]->end(), slotCollector);
    std::for_each(uniformVarMap[stage]->begin(), uniformVarMap[stage]->end(), slotCollector);
    resolver->endCollect(stage);
    intermediates[stage] = &intermediate;
    return !hadError;
}

bool TGlslIoMapper::doMap(TIoMapResolver* resolver, TInfoSink& infoSink) {
    resolver->endResolve(EShLangCount);
    if (!hadError) {
        //Resolve uniform location, ubo/ssbo/opaque bindings across stages
        TResolverUniformAdaptor uniformResolve(EShLangCount, *resolver, uniformVarMap, infoSink, hadError);
        TResolverInOutAdaptor inOutResolve(EShLangCount, *resolver, infoSink, hadError);
        TSymbolValidater symbolValidater(*resolver, infoSink, inVarMaps,
                                         outVarMaps, uniformVarMap, hadError, profile, version);

        TVarLiveVector inVectors[EShLangCount];
        TVarLiveVector outVectors[EShLangCount];
        TVarLiveVector uniformVector;

        resolver->beginResolve(EShLangCount);
        for (int stage = EShLangVertex; stage < EShLangCount; stage++) {
            if (inVarMaps[stage] != nullptr) {
                inOutResolve.setStage(EShLanguage(stage));

                // copy vars into a sorted list
                std::for_each(inVarMaps[stage]->begin(), inVarMaps[stage]->end(),
                        [&inVectors, stage](TVarLivePair p) { inVectors[stage].push_back(p); });
                std::sort(inVectors[stage].begin(), inVectors[stage].end(),
                        [](const TVarLivePair& p1, const TVarLivePair& p2) -> bool {
                            return TVarEntryInfo::TOrderByPriority()(p1.second, p2.second);
                });

                std::for_each(outVarMaps[stage]->begin(), outVarMaps[stage]->end(),
                        [&outVectors, stage](TVarLivePair p) { outVectors[stage].push_back(p); });
                std::sort(outVectors[stage].begin(), outVectors[stage].end(),
                        [](const TVarLivePair& p1, const TVarLivePair& p2) -> bool {
                            return TVarEntryInfo::TOrderByPriority()(p1.second, p2.second);
                });

                for (auto& var : inVectors[stage]) { symbolValidater(var); }
                for (auto& var : inVectors[stage]) { inOutResolve(var); }
                for (auto& var : outVectors[stage]) { symbolValidater(var); }
                for (auto& var : outVectors[stage]) { inOutResolve(var); }

                // copy results back into maps
                std::for_each(inVectors[stage].begin(), inVectors[stage].end(),
                    [this, stage](TVarLivePair p) {
                        auto at = inVarMaps[stage]->find(p.first);
                        if (at != inVarMaps[stage]->end())
                            at->second = p.second;
                });

                std::for_each(outVectors[stage].begin(), outVectors[stage].end(),
                    [this, stage](TVarLivePair p) {
                        auto at = outVarMaps[stage]->find(p.first);
                        if (at != outVarMaps[stage]->end())
                            at->second = p.second;
                });

            }
            if (uniformVarMap[stage] != nullptr) {
                uniformResolve.setStage(EShLanguage(stage));
                for (auto& var : *(uniformVarMap[stage])) { uniformVector.push_back(var); }
            }
        }
        std::sort(uniformVector.begin(), uniformVector.end(), [](const TVarLivePair& p1, const TVarLivePair& p2) -> bool {
            return TVarEntryInfo::TOrderByPriorityAndLive()(p1.second, p2.second);
        });
        for (auto& var : uniformVector) { symbolValidater(var); }
        for (auto& var : uniformVector) { uniformResolve(var); }
        std::sort(uniformVector.begin(), uniformVector.end(), [](const TVarLivePair& p1, const TVarLivePair& p2) -> bool {
            return TVarEntryInfo::TOrderByPriority()(p1.second, p2.second);
        });
        resolver->endResolve(EShLangCount);
        if (autoPushConstantBlockName.length()) {
            bool upgraded = false;
            for (size_t stage = 0; stage < EShLangCount; stage++) {
                if (intermediates[stage] != nullptr) {
                    TVarLiveMap** pUniformVarMap = uniformResolve.uniformVarMap;
                    auto at = pUniformVarMap[stage]->find(autoPushConstantBlockName);
                    if (at == pUniformVarMap[stage]->end())
                        continue;
                    TQualifier& qualifier = at->second.symbol->getQualifier();
                    if (!qualifier.isUniform())
                        continue;
                    TType& t = at->second.symbol->getWritableType();
                    int size, stride;
                    TIntermediate::getBaseAlignment(t, size, stride, autoPushConstantBlockPacking,
                                                    qualifier.layoutMatrix == ElmRowMajor);
                    if (size <= int(autoPushConstantMaxSize)) {
                        qualifier.setBlockStorage(EbsPushConstant);
                        qualifier.layoutPacking = autoPushConstantBlockPacking;
                        // Push constants don't have set/binding etc. decorations, remove those.
                        qualifier.layoutSet = TQualifier::layoutSetEnd;
                        at->second.clearNewAssignments();

                        upgraded = true;
                    }
                }
            }
            // If it's been upgraded to push_constant, then set the flag so when its traversed
            // in the next for loop, all references to this symbol will get their flag changed.
            // so it doesn't get a set/binding assigned to it.
            if (upgraded) {
                std::for_each(uniformVector.begin(), uniformVector.end(),
                                       [this](TVarLivePair& p) {
                if (p.first == autoPushConstantBlockName) {
                        p.second.upgradedToPushConstant = true;
                    }
                });
            }
        }
        for (size_t stage = 0; stage < EShLangCount; stage++) {
            if (intermediates[stage] != nullptr) {
                // traverse each stage, set new location to each input/output and unifom symbol, set new binding to
                // ubo, ssbo and opaque symbols. Assign push_constant upgrades as well.
                TVarLiveMap** pUniformVarMap = uniformResolve.uniformVarMap;
                std::for_each(uniformVector.begin(), uniformVector.end(), [pUniformVarMap, stage](TVarLivePair p) {
                    auto at = pUniformVarMap[stage]->find(p.second.symbol->getAccessName());
                    if (at != pUniformVarMap[stage]->end() && at->second.id == p.second.id){
                        if (p.second.upgradedToPushConstant) {
                            at->second.upgradedToPushConstant = true;
                        } else {
                            int resolvedBinding = at->second.newBinding;
                            at->second = p.second;
                            if (resolvedBinding > 0)
                                at->second.newBinding = resolvedBinding;
                        }
                    }
                });
                TVarSetTraverser iter_iomap(*intermediates[stage], *inVarMaps[stage], *outVarMaps[stage],
                                            *uniformResolve.uniformVarMap[stage]);
                intermediates[stage]->getTreeRoot()->traverse(&iter_iomap);
            }
        }
        return !hadError;
    } else {
        return false;
    }
}

} // end namespace glslang
