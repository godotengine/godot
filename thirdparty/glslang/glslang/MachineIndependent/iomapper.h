//
// Copyright (C) 2016 LunarG, Inc.
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

#if !defined(GLSLANG_WEB) && !defined(GLSLANG_ANGLE)

#ifndef _IOMAPPER_INCLUDED
#define _IOMAPPER_INCLUDED

#include <cstdint>
#include "LiveTraverser.h"
#include <unordered_map>
#include <unordered_set>
//
// A reflection database and its interface, consistent with the OpenGL API reflection queries.
//

class TInfoSink;

namespace glslang {

class TIntermediate;
struct TVarEntryInfo {
    int id;
    TIntermSymbol* symbol;
    bool live;
    int newBinding;
    int newSet;
    int newLocation;
    int newComponent;
    int newIndex;
    EShLanguage stage;
    struct TOrderById {
        inline bool operator()(const TVarEntryInfo& l, const TVarEntryInfo& r) { return l.id < r.id; }
    };

    struct TOrderByPriority {
        // ordering:
        // 1) has both binding and set
        // 2) has binding but no set
        // 3) has no binding but set
        // 4) has no binding and no set
        inline bool operator()(const TVarEntryInfo& l, const TVarEntryInfo& r) {
            const TQualifier& lq = l.symbol->getQualifier();
            const TQualifier& rq = r.symbol->getQualifier();

            // simple rules:
            // has binding gives 2 points
            // has set gives 1 point
            // who has the most points is more important.
            int lPoints = (lq.hasBinding() ? 2 : 0) + (lq.hasSet() ? 1 : 0);
            int rPoints = (rq.hasBinding() ? 2 : 0) + (rq.hasSet() ? 1 : 0);

            if (lPoints == rPoints)
                return l.id < r.id;
            return lPoints > rPoints;
        }
    };
};

// Base class for shared TIoMapResolver services, used by several derivations.
struct TDefaultIoResolverBase : public glslang::TIoMapResolver {
public:
    TDefaultIoResolverBase(const TIntermediate& intermediate);
    typedef std::vector<int> TSlotSet;
    typedef std::unordered_map<int, TSlotSet> TSlotSetMap;

    // grow the reflection stage by stage
    void notifyBinding(EShLanguage, TVarEntryInfo& /*ent*/) override {}
    void notifyInOut(EShLanguage, TVarEntryInfo& /*ent*/) override {}
    void beginNotifications(EShLanguage) override {}
    void endNotifications(EShLanguage) override {}
    void beginResolve(EShLanguage) override {}
    void endResolve(EShLanguage) override {}
    void beginCollect(EShLanguage) override {}
    void endCollect(EShLanguage) override {}
    void reserverResourceSlot(TVarEntryInfo& /*ent*/, TInfoSink& /*infoSink*/) override {}
    void reserverStorageSlot(TVarEntryInfo& /*ent*/, TInfoSink& /*infoSink*/) override {}
    int getBaseBinding(TResourceType res, unsigned int set) const;
    const std::vector<std::string>& getResourceSetBinding() const;
    virtual TResourceType getResourceType(const glslang::TType& type) = 0;
    bool doAutoBindingMapping() const;
    bool doAutoLocationMapping() const;
    TSlotSet::iterator findSlot(int set, int slot);
    bool checkEmpty(int set, int slot);
    bool validateInOut(EShLanguage /*stage*/, TVarEntryInfo& /*ent*/) override { return true; }
    int reserveSlot(int set, int slot, int size = 1);
    int getFreeSlot(int set, int base, int size = 1);
    int resolveSet(EShLanguage /*stage*/, TVarEntryInfo& ent) override;
    int resolveUniformLocation(EShLanguage /*stage*/, TVarEntryInfo& ent) override;
    int resolveInOutLocation(EShLanguage stage, TVarEntryInfo& ent) override;
    int resolveInOutComponent(EShLanguage /*stage*/, TVarEntryInfo& ent) override;
    int resolveInOutIndex(EShLanguage /*stage*/, TVarEntryInfo& ent) override;
    void addStage(EShLanguage stage) override {
        if (stage < EShLangCount)
            stageMask[stage] = true;
    }
    uint32_t computeTypeLocationSize(const TType& type, EShLanguage stage);

    TSlotSetMap slots;
    bool hasError = false;

protected:
    TDefaultIoResolverBase(TDefaultIoResolverBase&);
    TDefaultIoResolverBase& operator=(TDefaultIoResolverBase&);
    const TIntermediate& intermediate;
    int nextUniformLocation;
    int nextInputLocation;
    int nextOutputLocation;
    bool stageMask[EShLangCount + 1];
    // Return descriptor set specific base if there is one, and the generic base otherwise.
    int selectBaseBinding(int base, int descriptorSetBase) const {
        return descriptorSetBase != -1 ? descriptorSetBase : base;
    }

    static int getLayoutSet(const glslang::TType& type) {
        if (type.getQualifier().hasSet())
            return type.getQualifier().layoutSet;
        else
            return 0;
    }

    static bool isSamplerType(const glslang::TType& type) {
        return type.getBasicType() == glslang::EbtSampler && type.getSampler().isPureSampler();
    }

    static bool isTextureType(const glslang::TType& type) {
        return (type.getBasicType() == glslang::EbtSampler &&
                (type.getSampler().isTexture() || type.getSampler().isSubpass()));
    }

    static bool isUboType(const glslang::TType& type) {
        return type.getQualifier().storage == EvqUniform;
    }

    static bool isImageType(const glslang::TType& type) {
        return type.getBasicType() == glslang::EbtSampler && type.getSampler().isImage();
    }

    static bool isSsboType(const glslang::TType& type) {
        return type.getQualifier().storage == EvqBuffer;
    }

    // Return true if this is a SRV (shader resource view) type:
    static bool isSrvType(const glslang::TType& type) {
        return isTextureType(type) || type.getQualifier().storage == EvqBuffer;
    }

    // Return true if this is a UAV (unordered access view) type:
    static bool isUavType(const glslang::TType& type) {
        if (type.getQualifier().isReadOnly())
            return false;
        return (type.getBasicType() == glslang::EbtSampler && type.getSampler().isImage()) ||
                (type.getQualifier().storage == EvqBuffer);
    }
};

// Default I/O resolver for OpenGL
struct TDefaultGlslIoResolver : public TDefaultIoResolverBase {
public:
    typedef std::map<TString, int> TVarSlotMap;  // <resourceName, location/binding>
    typedef std::map<int, TVarSlotMap> TSlotMap; // <resourceKey, TVarSlotMap>
    TDefaultGlslIoResolver(const TIntermediate& intermediate);
    bool validateBinding(EShLanguage /*stage*/, TVarEntryInfo& /*ent*/) override { return true; }
    TResourceType getResourceType(const glslang::TType& type) override;
    int resolveInOutLocation(EShLanguage stage, TVarEntryInfo& ent) override;
    int resolveUniformLocation(EShLanguage /*stage*/, TVarEntryInfo& ent) override;
    int resolveBinding(EShLanguage /*stage*/, TVarEntryInfo& ent) override;
    void beginResolve(EShLanguage /*stage*/) override;
    void endResolve(EShLanguage stage) override;
    void beginCollect(EShLanguage) override;
    void endCollect(EShLanguage) override;
    void reserverStorageSlot(TVarEntryInfo& ent, TInfoSink& infoSink) override;
    void reserverResourceSlot(TVarEntryInfo& ent, TInfoSink& infoSink) override;
    // in/out symbol and uniform symbol are stored in the same resourceSlotMap, the storage key is used to identify each type of symbol.
    // We use stage and storage qualifier to construct a storage key. it can help us identify the same storage resource used in different stage.
    // if a resource is a program resource and we don't need know it usage stage, we can use same stage to build storage key.
    // Note: both stage and type must less then 0xffff.
    int buildStorageKey(EShLanguage stage, TStorageQualifier type) {
        assert(static_cast<uint32_t>(stage) <= 0x0000ffff && static_cast<uint32_t>(type) <= 0x0000ffff);
        return (stage << 16) | type;
    }

protected:
    // Use for mark pre stage, to get more interface symbol information.
    EShLanguage preStage;
    // Use for mark current shader stage for resolver
    EShLanguage currentStage;
    // Slot map for storage resource(location of uniform and interface symbol) It's a program share slot
    TSlotMap resourceSlotMap;
    // Slot map for other resource(image, ubo, ssbo), It's a program share slot.
    TSlotMap storageSlotMap;
};

typedef std::map<TString, TVarEntryInfo> TVarLiveMap;

// override function "operator=", if a vector<const _Kty, _Ty> being sort,
// when use vc++, the sort function will call :
// pair& operator=(const pair<_Other1, _Other2>& _Right)
// {
//     first = _Right.first;
//     second = _Right.second;
//     return (*this);
// }
// that will make a const type handing on left.
// override this function can avoid a compiler error.
// In the future, if the vc++ compiler can handle such a situation,
// this part of the code will be removed.
struct TVarLivePair : std::pair<const TString, TVarEntryInfo> {
    TVarLivePair(const std::pair<const TString, TVarEntryInfo>& _Right) : pair(_Right.first, _Right.second) {}
    TVarLivePair& operator=(const TVarLivePair& _Right) {
        const_cast<TString&>(first) = _Right.first;
        second = _Right.second;
        return (*this);
    }
    TVarLivePair(const TVarLivePair& src) : pair(src) { }
};
typedef std::vector<TVarLivePair> TVarLiveVector;

// I/O mapper
class TIoMapper {
public:
    TIoMapper() {}
    virtual ~TIoMapper() {}
    // grow the reflection stage by stage
    bool virtual addStage(EShLanguage, TIntermediate&, TInfoSink&, TIoMapResolver*);
    bool virtual doMap(TIoMapResolver*, TInfoSink&) { return true; }
};

// I/O mapper for OpenGL
class TGlslIoMapper : public TIoMapper {
public:
    TGlslIoMapper() {
        memset(inVarMaps,     0, sizeof(TVarLiveMap*)   * (EShLangCount + 1));
        memset(outVarMaps,    0, sizeof(TVarLiveMap*)   * (EShLangCount + 1));
        memset(uniformVarMap, 0, sizeof(TVarLiveMap*)   * (EShLangCount + 1));
        memset(intermediates, 0, sizeof(TIntermediate*) * (EShLangCount + 1));
        profile = ENoProfile;
        version = 0;
    }
    virtual ~TGlslIoMapper() {
        for (size_t stage = 0; stage < EShLangCount; stage++) {
            if (inVarMaps[stage] != nullptr) {
                delete inVarMaps[stage];
                inVarMaps[stage] = nullptr;
            }
            if (outVarMaps[stage] != nullptr) {
                delete outVarMaps[stage];
                outVarMaps[stage] = nullptr;
            }
            if (uniformVarMap[stage] != nullptr) {
                delete uniformVarMap[stage];
                uniformVarMap[stage] = nullptr;
            }
            if (intermediates[stage] != nullptr)
                intermediates[stage] = nullptr;
        }
    }
    // grow the reflection stage by stage
    bool addStage(EShLanguage, TIntermediate&, TInfoSink&, TIoMapResolver*) override;
    bool doMap(TIoMapResolver*, TInfoSink&) override;
    TVarLiveMap *inVarMaps[EShLangCount], *outVarMaps[EShLangCount],
                *uniformVarMap[EShLangCount];
    TIntermediate* intermediates[EShLangCount];
    bool hadError = false;
    EProfile profile;
    int version;
};

} // end namespace glslang

#endif // _IOMAPPER_INCLUDED

#endif // !GLSLANG_WEB && !GLSLANG_ANGLE
