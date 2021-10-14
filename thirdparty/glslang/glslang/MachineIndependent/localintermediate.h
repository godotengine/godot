//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2016 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
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

#ifndef _LOCAL_INTERMEDIATE_INCLUDED_
#define _LOCAL_INTERMEDIATE_INCLUDED_

#include "../Include/intermediate.h"
#include "../Public/ShaderLang.h"
#include "Versions.h"

#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <array>

class TInfoSink;

namespace glslang {

struct TMatrixSelector {
    int coord1;  // stay agnostic about column/row; this is parse order
    int coord2;
};

typedef int TVectorSelector;

const int MaxSwizzleSelectors = 4;

template<typename selectorType>
class TSwizzleSelectors {
public:
    TSwizzleSelectors() : size_(0) { }

    void push_back(selectorType comp)
    {
        if (size_ < MaxSwizzleSelectors)
            components[size_++] = comp;
    }
    void resize(int s)
    {
        assert(s <= size_);
        size_ = s;
    }
    int size() const { return size_; }
    selectorType operator[](int i) const
    {
        assert(i < MaxSwizzleSelectors);
        return components[i];
    }

private:
    int size_;
    selectorType components[MaxSwizzleSelectors];
};

//
// Some helper structures for TIntermediate.  Their contents are encapsulated
// by TIntermediate.
//

// Used for call-graph algorithms for detecting recursion, missing bodies, and dead bodies.
// A "call" is a pair: <caller, callee>.
// There can be duplicates. General assumption is the list is small.
struct TCall {
    TCall(const TString& pCaller, const TString& pCallee) : caller(pCaller), callee(pCallee) { }
    TString caller;
    TString callee;
    bool visited;
    bool currentPath;
    bool errorGiven;
    int calleeBodyPosition;
};

// A generic 1-D range.
struct TRange {
    TRange(int start, int last) : start(start), last(last) { }
    bool overlap(const TRange& rhs) const
    {
        return last >= rhs.start && start <= rhs.last;
    }
    int start;
    int last;
};

// An IO range is a 3-D rectangle; the set of (location, component, index) triples all lying
// within the same location range, component range, and index value.  Locations don't alias unless
// all other dimensions of their range overlap.
struct TIoRange {
    TIoRange(TRange location, TRange component, TBasicType basicType, int index)
        : location(location), component(component), basicType(basicType), index(index) { }
    bool overlap(const TIoRange& rhs) const
    {
        return location.overlap(rhs.location) && component.overlap(rhs.component) && index == rhs.index;
    }
    TRange location;
    TRange component;
    TBasicType basicType;
    int index;
};

// An offset range is a 2-D rectangle; the set of (binding, offset) pairs all lying
// within the same binding and offset range.
struct TOffsetRange {
    TOffsetRange(TRange binding, TRange offset)
        : binding(binding), offset(offset) { }
    bool overlap(const TOffsetRange& rhs) const
    {
        return binding.overlap(rhs.binding) && offset.overlap(rhs.offset);
    }
    TRange binding;
    TRange offset;
};

#ifndef GLSLANG_WEB
// Things that need to be tracked per xfb buffer.
struct TXfbBuffer {
    TXfbBuffer() : stride(TQualifier::layoutXfbStrideEnd), implicitStride(0), contains64BitType(false),
                   contains32BitType(false), contains16BitType(false) { }
    std::vector<TRange> ranges;  // byte offsets that have already been assigned
    unsigned int stride;
    unsigned int implicitStride;
    bool contains64BitType;
    bool contains32BitType;
    bool contains16BitType;
};
#endif

// Track a set of strings describing how the module was processed.
// This includes command line options, transforms, etc., ideally inclusive enough
// to reproduce the steps used to transform the input source to the output.
// E.g., see SPIR-V OpModuleProcessed.
// Each "process" or "transform" uses is expressed in the form:
//   process arg0 arg1 arg2 ...
//   process arg0 arg1 arg2 ...
// where everything is textual, and there can be zero or more arguments
class TProcesses {
public:
    TProcesses() {}
    ~TProcesses() {}

    void addProcess(const char* process)
    {
        processes.push_back(process);
    }
    void addProcess(const std::string& process)
    {
        processes.push_back(process);
    }
    void addArgument(int arg)
    {
        processes.back().append(" ");
        std::string argString = std::to_string(arg);
        processes.back().append(argString);
    }
    void addArgument(const char* arg)
    {
        processes.back().append(" ");
        processes.back().append(arg);
    }
    void addArgument(const std::string& arg)
    {
        processes.back().append(" ");
        processes.back().append(arg);
    }
    void addIfNonZero(const char* process, int value)
    {
        if (value != 0) {
            addProcess(process);
            addArgument(value);
        }
    }

    const std::vector<std::string>& getProcesses() const { return processes; }

private:
    std::vector<std::string> processes;
};

class TSymbolTable;
class TSymbol;
class TVariable;

//
// Texture and Sampler transformation mode.
//
enum ComputeDerivativeMode {
    LayoutDerivativeNone,         // default layout as SPV_NV_compute_shader_derivatives not enabled
    LayoutDerivativeGroupQuads,   // derivative_group_quadsNV
    LayoutDerivativeGroupLinear,  // derivative_group_linearNV
};

class TIdMaps {
public:
    TMap<TString, long long>& operator[](long long i) { return maps[i]; }
    const TMap<TString, long long>& operator[](long long i) const { return maps[i]; }
private:
    TMap<TString, long long> maps[EsiCount];
};

class TNumericFeatures {
public:
    TNumericFeatures() : features(0) { }
    TNumericFeatures(const TNumericFeatures&) = delete;
    TNumericFeatures& operator=(const TNumericFeatures&) = delete;
    typedef enum : unsigned int {
        shader_explicit_arithmetic_types          = 1 << 0,
        shader_explicit_arithmetic_types_int8     = 1 << 1,
        shader_explicit_arithmetic_types_int16    = 1 << 2,
        shader_explicit_arithmetic_types_int32    = 1 << 3,
        shader_explicit_arithmetic_types_int64    = 1 << 4,
        shader_explicit_arithmetic_types_float16  = 1 << 5,
        shader_explicit_arithmetic_types_float32  = 1 << 6,
        shader_explicit_arithmetic_types_float64  = 1 << 7,
        shader_implicit_conversions               = 1 << 8,
        gpu_shader_fp64                           = 1 << 9,
        gpu_shader_int16                          = 1 << 10,
        gpu_shader_half_float                     = 1 << 11,
    } feature;
    void insert(feature f) { features |= f; }
    void erase(feature f) { features &= ~f; }
    bool contains(feature f) const { return (features & f) != 0; }
private:
    unsigned int features;
};

// MustBeAssigned wraps a T, asserting that it has been assigned with 
// operator =() before attempting to read with operator T() or operator ->().
// Used to catch cases where fields are read before they have been assigned.
template<typename T>
class MustBeAssigned
{
public:
    MustBeAssigned() = default;
    MustBeAssigned(const T& v) : value(v) {}
    operator const T&() const { assert(isSet); return value; }
    const T* operator ->() const { assert(isSet); return &value; }
    MustBeAssigned& operator = (const T& v) { value = v; isSet = true; return *this; } 
private:
    T value;
    bool isSet = false;
};

//
// Set of helper functions to help parse and build the tree.
//
class TIntermediate {
public:
    explicit TIntermediate(EShLanguage l, int v = 0, EProfile p = ENoProfile) :
        language(l),
#ifndef GLSLANG_ANGLE
        profile(p), version(v),
#endif
        treeRoot(0),
        resources(TBuiltInResource{}),
        numEntryPoints(0), numErrors(0), numPushConstants(0), recursive(false),
        invertY(false),
        useStorageBuffer(false),
        invariantAll(false),
        nanMinMaxClamp(false),
        depthReplacing(false),
        uniqueId(0),
        globalUniformBlockName(""),
        atomicCounterBlockName(""),
        globalUniformBlockSet(TQualifier::layoutSetEnd),
        globalUniformBlockBinding(TQualifier::layoutBindingEnd),
        atomicCounterBlockSet(TQualifier::layoutSetEnd)
#ifndef GLSLANG_WEB
        ,
        implicitThisName("@this"), implicitCounterName("@count"),
        source(EShSourceNone),
        useVulkanMemoryModel(false),
        invocations(TQualifier::layoutNotSet), vertices(TQualifier::layoutNotSet),
        inputPrimitive(ElgNone), outputPrimitive(ElgNone),
        pixelCenterInteger(false), originUpperLeft(false),
        vertexSpacing(EvsNone), vertexOrder(EvoNone), interlockOrdering(EioNone), pointMode(false), earlyFragmentTests(false),
        postDepthCoverage(false), depthLayout(EldNone),
        hlslFunctionality1(false),
        blendEquations(0), xfbMode(false), multiStream(false),
        layoutOverrideCoverage(false),
        geoPassthroughEXT(false),
        numShaderRecordBlocks(0),
        computeDerivativeMode(LayoutDerivativeNone),
        primitives(TQualifier::layoutNotSet),
        numTaskNVBlocks(0),
        layoutPrimitiveCulling(false),
        autoMapBindings(false),
        autoMapLocations(false),
        flattenUniformArrays(false),
        useUnknownFormat(false),
        hlslOffsets(false),
        hlslIoMapping(false),
        useVariablePointers(false),
        textureSamplerTransformMode(EShTexSampTransKeep),
        needToLegalize(false),
        binaryDoubleOutput(false),
        subgroupUniformControlFlow(false),
        usePhysicalStorageBuffer(false),
        spirvRequirement(nullptr),
        spirvExecutionMode(nullptr),
        uniformLocationBase(0)
#endif
    {
        localSize[0] = 1;
        localSize[1] = 1;
        localSize[2] = 1;
        localSizeNotDefault[0] = false;
        localSizeNotDefault[1] = false;
        localSizeNotDefault[2] = false;
        localSizeSpecId[0] = TQualifier::layoutNotSet;
        localSizeSpecId[1] = TQualifier::layoutNotSet;
        localSizeSpecId[2] = TQualifier::layoutNotSet;
#ifndef GLSLANG_WEB
        xfbBuffers.resize(TQualifier::layoutXfbBufferEnd);
        shiftBinding.fill(0);
#endif
    }

    void setVersion(int v)
    {
#ifndef GLSLANG_ANGLE
        version = v;
#endif
    }
    void setProfile(EProfile p)
    {
#ifndef GLSLANG_ANGLE
        profile = p;
#endif
    }

    int getVersion() const { return version; }
    EProfile getProfile() const { return profile; }
    void setSpv(const SpvVersion& s)
    {
        spvVersion = s;

        // client processes
        if (spvVersion.vulkan > 0)
            processes.addProcess("client vulkan100");
        if (spvVersion.openGl > 0)
            processes.addProcess("client opengl100");

        // target SPV
        switch (spvVersion.spv) {
        case 0:
            break;
        case EShTargetSpv_1_0:
            break;
        case EShTargetSpv_1_1:
            processes.addProcess("target-env spirv1.1");
            break;
        case EShTargetSpv_1_2:
            processes.addProcess("target-env spirv1.2");
            break;
        case EShTargetSpv_1_3:
            processes.addProcess("target-env spirv1.3");
            break;
        case EShTargetSpv_1_4:
            processes.addProcess("target-env spirv1.4");
            break;
        case EShTargetSpv_1_5:
            processes.addProcess("target-env spirv1.5");
            break;
        default:
            processes.addProcess("target-env spirvUnknown");
            break;
        }

        // target-environment processes
        switch (spvVersion.vulkan) {
        case 0:
            break;
        case EShTargetVulkan_1_0:
            processes.addProcess("target-env vulkan1.0");
            break;
        case EShTargetVulkan_1_1:
            processes.addProcess("target-env vulkan1.1");
            break;
        case EShTargetVulkan_1_2:
            processes.addProcess("target-env vulkan1.2");
            break;
        default:
            processes.addProcess("target-env vulkanUnknown");
            break;
        }
        if (spvVersion.openGl > 0)
            processes.addProcess("target-env opengl");
    }
    const SpvVersion& getSpv() const { return spvVersion; }
    EShLanguage getStage() const { return language; }
    void addRequestedExtension(const char* extension) { requestedExtensions.insert(extension); }
    const std::set<std::string>& getRequestedExtensions() const { return requestedExtensions; }
    bool isRayTracingStage() const {
        return language >= EShLangRayGen && language <= EShLangCallableNV;
    }

    void setTreeRoot(TIntermNode* r) { treeRoot = r; }
    TIntermNode* getTreeRoot() const { return treeRoot; }
    void incrementEntryPointCount() { ++numEntryPoints; }
    int getNumEntryPoints() const { return numEntryPoints; }
    int getNumErrors() const { return numErrors; }
    void addPushConstantCount() { ++numPushConstants; }
    void setLimits(const TBuiltInResource& r) { resources = r; }
    const TBuiltInResource& getLimits() const { return resources; }

    bool postProcess(TIntermNode*, EShLanguage);
    void removeTree();

    void setEntryPointName(const char* ep)
    {
        entryPointName = ep;
        processes.addProcess("entry-point");
        processes.addArgument(entryPointName);
    }
    void setEntryPointMangledName(const char* ep) { entryPointMangledName = ep; }
    const std::string& getEntryPointName() const { return entryPointName; }
    const std::string& getEntryPointMangledName() const { return entryPointMangledName; }

    void setInvertY(bool invert)
    {
        invertY = invert;
        if (invertY)
            processes.addProcess("invert-y");
    }
    bool getInvertY() const { return invertY; }

#ifdef ENABLE_HLSL
    void setSource(EShSource s) { source = s; }
    EShSource getSource() const { return source; }
#else
    void setSource(EShSource s) { assert(s == EShSourceGlsl); (void)s; }
    EShSource getSource() const { return EShSourceGlsl; }
#endif

    bool isRecursive() const { return recursive; }

    TIntermSymbol* addSymbol(const TVariable&);
    TIntermSymbol* addSymbol(const TVariable&, const TSourceLoc&);
    TIntermSymbol* addSymbol(const TType&, const TSourceLoc&);
    TIntermSymbol* addSymbol(const TIntermSymbol&);
    TIntermTyped* addConversion(TOperator, const TType&, TIntermTyped*);
    std::tuple<TIntermTyped*, TIntermTyped*> addPairConversion(TOperator op, TIntermTyped* node0, TIntermTyped* node1);
    TIntermTyped* addUniShapeConversion(TOperator, const TType&, TIntermTyped*);
    TIntermTyped* addConversion(TBasicType convertTo, TIntermTyped* node) const;
    void addBiShapeConversion(TOperator, TIntermTyped*& lhsNode, TIntermTyped*& rhsNode);
    TIntermTyped* addShapeConversion(const TType&, TIntermTyped*);
    TIntermTyped* addBinaryMath(TOperator, TIntermTyped* left, TIntermTyped* right, const TSourceLoc&);
    TIntermTyped* addAssign(TOperator op, TIntermTyped* left, TIntermTyped* right, const TSourceLoc&);
    TIntermTyped* addIndex(TOperator op, TIntermTyped* base, TIntermTyped* index, const TSourceLoc&);
    TIntermTyped* addUnaryMath(TOperator, TIntermTyped* child, const TSourceLoc&);
    TIntermTyped* addBuiltInFunctionCall(const TSourceLoc& line, TOperator, bool unary, TIntermNode*, const TType& returnType);
    bool canImplicitlyPromote(TBasicType from, TBasicType to, TOperator op = EOpNull) const;
    bool isIntegralPromotion(TBasicType from, TBasicType to) const;
    bool isFPPromotion(TBasicType from, TBasicType to) const;
    bool isIntegralConversion(TBasicType from, TBasicType to) const;
    bool isFPConversion(TBasicType from, TBasicType to) const;
    bool isFPIntegralConversion(TBasicType from, TBasicType to) const;
    TOperator mapTypeToConstructorOp(const TType&) const;
    TIntermAggregate* growAggregate(TIntermNode* left, TIntermNode* right);
    TIntermAggregate* growAggregate(TIntermNode* left, TIntermNode* right, const TSourceLoc&);
    TIntermAggregate* makeAggregate(TIntermNode* node);
    TIntermAggregate* makeAggregate(TIntermNode* node, const TSourceLoc&);
    TIntermAggregate* makeAggregate(const TSourceLoc&);
    TIntermTyped* setAggregateOperator(TIntermNode*, TOperator, const TType& type, const TSourceLoc&);
    bool areAllChildConst(TIntermAggregate* aggrNode);
    TIntermSelection* addSelection(TIntermTyped* cond, TIntermNodePair code, const TSourceLoc&);
    TIntermTyped* addSelection(TIntermTyped* cond, TIntermTyped* trueBlock, TIntermTyped* falseBlock, const TSourceLoc&);
    TIntermTyped* addComma(TIntermTyped* left, TIntermTyped* right, const TSourceLoc&);
    TIntermTyped* addMethod(TIntermTyped*, const TType&, const TString*, const TSourceLoc&);
    TIntermConstantUnion* addConstantUnion(const TConstUnionArray&, const TType&, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(signed char, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(unsigned char, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(signed short, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(unsigned short, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(int, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(unsigned int, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(long long, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(unsigned long long, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(bool, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(double, TBasicType, const TSourceLoc&, bool literal = false) const;
    TIntermConstantUnion* addConstantUnion(const TString*, const TSourceLoc&, bool literal = false) const;
    TIntermTyped* promoteConstantUnion(TBasicType, TIntermConstantUnion*) const;
    bool parseConstTree(TIntermNode*, TConstUnionArray, TOperator, const TType&, bool singleConstantParam = false);
    TIntermLoop* addLoop(TIntermNode*, TIntermTyped*, TIntermTyped*, bool testFirst, const TSourceLoc&);
    TIntermAggregate* addForLoop(TIntermNode*, TIntermNode*, TIntermTyped*, TIntermTyped*, bool testFirst,
        const TSourceLoc&, TIntermLoop*&);
    TIntermBranch* addBranch(TOperator, const TSourceLoc&);
    TIntermBranch* addBranch(TOperator, TIntermTyped*, const TSourceLoc&);
    template<typename selectorType> TIntermTyped* addSwizzle(TSwizzleSelectors<selectorType>&, const TSourceLoc&);

    // Low level functions to add nodes (no conversions or other higher level transformations)
    // If a type is provided, the node's type will be set to it.
    TIntermBinary* addBinaryNode(TOperator op, TIntermTyped* left, TIntermTyped* right, const TSourceLoc&) const;
    TIntermBinary* addBinaryNode(TOperator op, TIntermTyped* left, TIntermTyped* right, const TSourceLoc&,
        const TType&) const;
    TIntermUnary* addUnaryNode(TOperator op, TIntermTyped* child, const TSourceLoc&) const;
    TIntermUnary* addUnaryNode(TOperator op, TIntermTyped* child, const TSourceLoc&, const TType&) const;

    // Constant folding (in Constant.cpp)
    TIntermTyped* fold(TIntermAggregate* aggrNode);
    TIntermTyped* foldConstructor(TIntermAggregate* aggrNode);
    TIntermTyped* foldDereference(TIntermTyped* node, int index, const TSourceLoc&);
    TIntermTyped* foldSwizzle(TIntermTyped* node, TSwizzleSelectors<TVectorSelector>& fields, const TSourceLoc&);

    // Tree ops
    static const TIntermTyped* findLValueBase(const TIntermTyped*, bool swizzleOkay , bool BufferReferenceOk = false);

    // Linkage related
    void addSymbolLinkageNodes(TIntermAggregate*& linkage, EShLanguage, TSymbolTable&);
    void addSymbolLinkageNode(TIntermAggregate*& linkage, const TSymbol&);
    TIntermAggregate* findLinkerObjects() const;

    void setGlobalUniformBlockName(const char* name) { globalUniformBlockName = std::string(name); }
    const char* getGlobalUniformBlockName() const { return globalUniformBlockName.c_str(); }
    void setGlobalUniformSet(unsigned int set) { globalUniformBlockSet = set; }
    unsigned int getGlobalUniformSet() const { return globalUniformBlockSet; }
    void setGlobalUniformBinding(unsigned int binding) { globalUniformBlockBinding = binding; }
    unsigned int getGlobalUniformBinding() const { return globalUniformBlockBinding; }

    void setAtomicCounterBlockName(const char* name) { atomicCounterBlockName = std::string(name); }
    const char* getAtomicCounterBlockName() const { return atomicCounterBlockName.c_str(); }
    void setAtomicCounterBlockSet(unsigned int set) { atomicCounterBlockSet = set; }
    unsigned int getAtomicCounterBlockSet() const { return atomicCounterBlockSet; }


    void setUseStorageBuffer() { useStorageBuffer = true; }
    bool usingStorageBuffer() const { return useStorageBuffer; }
    void setInvariantAll() { invariantAll = true; }
    bool isInvariantAll() const { return invariantAll; }
    void setDepthReplacing() { depthReplacing = true; }
    bool isDepthReplacing() const { return depthReplacing; }
    bool setLocalSize(int dim, int size)
    {
        if (localSizeNotDefault[dim])
            return size == localSize[dim];
        localSizeNotDefault[dim] = true;
        localSize[dim] = size;
        return true;
    }
    unsigned int getLocalSize(int dim) const { return localSize[dim]; }
    bool isLocalSizeSet() const
    {
        // Return true if any component has been set (i.e. any component is not default).
        return localSizeNotDefault[0] || localSizeNotDefault[1] || localSizeNotDefault[2];
    }
    bool setLocalSizeSpecId(int dim, int id)
    {
        if (localSizeSpecId[dim] != TQualifier::layoutNotSet)
            return id == localSizeSpecId[dim];
        localSizeSpecId[dim] = id;
        return true;
    }
    int getLocalSizeSpecId(int dim) const { return localSizeSpecId[dim]; }
    bool isLocalSizeSpecialized() const
    {
        // Return true if any component has been specialized.
        return localSizeSpecId[0] != TQualifier::layoutNotSet ||
               localSizeSpecId[1] != TQualifier::layoutNotSet ||
               localSizeSpecId[2] != TQualifier::layoutNotSet;
    }
#ifdef GLSLANG_WEB
    void output(TInfoSink&, bool tree) { }

    bool isEsProfile() const { return false; }
    bool getXfbMode() const { return false; }
    bool isMultiStream() const { return false; }
    TLayoutGeometry getOutputPrimitive() const { return ElgNone; }
    bool getPostDepthCoverage() const { return false; }
    bool getEarlyFragmentTests() const { return false; }
    TLayoutDepth getDepth() const { return EldNone; }
    bool getPixelCenterInteger() const { return false; }
    void setOriginUpperLeft() { }
    bool getOriginUpperLeft() const { return true; }
    TInterlockOrdering getInterlockOrdering() const { return EioNone; }

    bool getAutoMapBindings() const { return false; }
    bool getAutoMapLocations() const { return false; }
    int getNumPushConstants() const { return 0; }
    void addShaderRecordCount() { }
    void addTaskNVCount() { }
    void setUseVulkanMemoryModel() { }
    bool usingVulkanMemoryModel() const { return false; }
    bool usingPhysicalStorageBuffer() const { return false; }
    bool usingVariablePointers() const { return false; }
    unsigned getXfbStride(int buffer) const { return 0; }
    bool hasLayoutDerivativeModeNone() const { return false; }
    ComputeDerivativeMode getLayoutDerivativeModeNone() const { return LayoutDerivativeNone; }
#else
    void output(TInfoSink&, bool tree);

    bool isEsProfile() const { return profile == EEsProfile; }

    void setShiftBinding(TResourceType res, unsigned int shift)
    {
        shiftBinding[res] = shift;

        const char* name = getResourceName(res);
        if (name != nullptr)
            processes.addIfNonZero(name, shift);
    }

    unsigned int getShiftBinding(TResourceType res) const { return shiftBinding[res]; }

    void setShiftBindingForSet(TResourceType res, unsigned int shift, unsigned int set)
    {
        if (shift == 0) // ignore if there's no shift: it's a no-op.
            return;

        shiftBindingForSet[res][set] = shift;

        const char* name = getResourceName(res);
        if (name != nullptr) {
            processes.addProcess(name);
            processes.addArgument(shift);
            processes.addArgument(set);
        }
    }

    int getShiftBindingForSet(TResourceType res, unsigned int set) const
    {
        const auto shift = shiftBindingForSet[res].find(set);
        return shift == shiftBindingForSet[res].end() ? -1 : shift->second;
    }
    bool hasShiftBindingForSet(TResourceType res) const { return !shiftBindingForSet[res].empty(); }

    void setResourceSetBinding(const std::vector<std::string>& shift)
    {
        resourceSetBinding = shift;
        if (shift.size() > 0) {
            processes.addProcess("resource-set-binding");
            for (int s = 0; s < (int)shift.size(); ++s)
                processes.addArgument(shift[s]);
        }
    }
    const std::vector<std::string>& getResourceSetBinding() const { return resourceSetBinding; }
    void setAutoMapBindings(bool map)
    {
        autoMapBindings = map;
        if (autoMapBindings)
            processes.addProcess("auto-map-bindings");
    }
    bool getAutoMapBindings() const { return autoMapBindings; }
    void setAutoMapLocations(bool map)
    {
        autoMapLocations = map;
        if (autoMapLocations)
            processes.addProcess("auto-map-locations");
    }
    bool getAutoMapLocations() const { return autoMapLocations; }

#ifdef ENABLE_HLSL
    void setFlattenUniformArrays(bool flatten)
    {
        flattenUniformArrays = flatten;
        if (flattenUniformArrays)
            processes.addProcess("flatten-uniform-arrays");
    }
    bool getFlattenUniformArrays() const { return flattenUniformArrays; }
#endif
    void setNoStorageFormat(bool b)
    {
        useUnknownFormat = b;
        if (useUnknownFormat)
            processes.addProcess("no-storage-format");
    }
    bool getNoStorageFormat() const { return useUnknownFormat; }
    void setUseVulkanMemoryModel()
    {
        useVulkanMemoryModel = true;
        processes.addProcess("use-vulkan-memory-model");
    }
    bool usingVulkanMemoryModel() const { return useVulkanMemoryModel; }
    void setUsePhysicalStorageBuffer()
    {
        usePhysicalStorageBuffer = true;
    }
    bool usingPhysicalStorageBuffer() const { return usePhysicalStorageBuffer; }
    void setUseVariablePointers()
    {
        useVariablePointers = true;
        processes.addProcess("use-variable-pointers");
    }
    bool usingVariablePointers() const { return useVariablePointers; }

#ifdef ENABLE_HLSL
    template<class T> T addCounterBufferName(const T& name) const { return name + implicitCounterName; }
    bool hasCounterBufferName(const TString& name) const {
        size_t len = strlen(implicitCounterName);
        return name.size() > len &&
               name.compare(name.size() - len, len, implicitCounterName) == 0;
    }
#endif

    void setTextureSamplerTransformMode(EShTextureSamplerTransformMode mode) { textureSamplerTransformMode = mode; }
    int getNumPushConstants() const { return numPushConstants; }
    void addShaderRecordCount() { ++numShaderRecordBlocks; }
    void addTaskNVCount() { ++numTaskNVBlocks; }

    bool setInvocations(int i)
    {
        if (invocations != TQualifier::layoutNotSet)
            return invocations == i;
        invocations = i;
        return true;
    }
    int getInvocations() const { return invocations; }
    bool setVertices(int m)
    {
        if (vertices != TQualifier::layoutNotSet)
            return vertices == m;
        vertices = m;
        return true;
    }
    int getVertices() const { return vertices; }
    bool setInputPrimitive(TLayoutGeometry p)
    {
        if (inputPrimitive != ElgNone)
            return inputPrimitive == p;
        inputPrimitive = p;
        return true;
    }
    TLayoutGeometry getInputPrimitive() const { return inputPrimitive; }
    bool setVertexSpacing(TVertexSpacing s)
    {
        if (vertexSpacing != EvsNone)
            return vertexSpacing == s;
        vertexSpacing = s;
        return true;
    }
    TVertexSpacing getVertexSpacing() const { return vertexSpacing; }
    bool setVertexOrder(TVertexOrder o)
    {
        if (vertexOrder != EvoNone)
            return vertexOrder == o;
        vertexOrder = o;
        return true;
    }
    TVertexOrder getVertexOrder() const { return vertexOrder; }
    void setPointMode() { pointMode = true; }
    bool getPointMode() const { return pointMode; }

    bool setInterlockOrdering(TInterlockOrdering o)
    {
        if (interlockOrdering != EioNone)
            return interlockOrdering == o;
        interlockOrdering = o;
        return true;
    }
    TInterlockOrdering getInterlockOrdering() const { return interlockOrdering; }

    void setXfbMode() { xfbMode = true; }
    bool getXfbMode() const { return xfbMode; }
    void setMultiStream() { multiStream = true; }
    bool isMultiStream() const { return multiStream; }
    bool setOutputPrimitive(TLayoutGeometry p)
    {
        if (outputPrimitive != ElgNone)
            return outputPrimitive == p;
        outputPrimitive = p;
        return true;
    }
    TLayoutGeometry getOutputPrimitive() const { return outputPrimitive; }
    void setPostDepthCoverage() { postDepthCoverage = true; }
    bool getPostDepthCoverage() const { return postDepthCoverage; }
    void setEarlyFragmentTests() { earlyFragmentTests = true; }
    bool getEarlyFragmentTests() const { return earlyFragmentTests; }
    bool setDepth(TLayoutDepth d)
    {
        if (depthLayout != EldNone)
            return depthLayout == d;
        depthLayout = d;
        return true;
    }
    TLayoutDepth getDepth() const { return depthLayout; }
    void setOriginUpperLeft() { originUpperLeft = true; }
    bool getOriginUpperLeft() const { return originUpperLeft; }
    void setPixelCenterInteger() { pixelCenterInteger = true; }
    bool getPixelCenterInteger() const { return pixelCenterInteger; }
    void addBlendEquation(TBlendEquationShift b) { blendEquations |= (1 << b); }
    unsigned int getBlendEquations() const { return blendEquations; }
    bool setXfbBufferStride(int buffer, unsigned stride)
    {
        if (xfbBuffers[buffer].stride != TQualifier::layoutXfbStrideEnd)
            return xfbBuffers[buffer].stride == stride;
        xfbBuffers[buffer].stride = stride;
        return true;
    }
    unsigned getXfbStride(int buffer) const { return xfbBuffers[buffer].stride; }
    int addXfbBufferOffset(const TType&);
    unsigned int computeTypeXfbSize(const TType&, bool& contains64BitType, bool& contains32BitType, bool& contains16BitType) const;
    unsigned int computeTypeXfbSize(const TType&, bool& contains64BitType) const;
    void setLayoutOverrideCoverage() { layoutOverrideCoverage = true; }
    bool getLayoutOverrideCoverage() const { return layoutOverrideCoverage; }
    void setGeoPassthroughEXT() { geoPassthroughEXT = true; }
    bool getGeoPassthroughEXT() const { return geoPassthroughEXT; }
    void setLayoutDerivativeMode(ComputeDerivativeMode mode) { computeDerivativeMode = mode; }
    bool hasLayoutDerivativeModeNone() const { return computeDerivativeMode != LayoutDerivativeNone; }
    ComputeDerivativeMode getLayoutDerivativeModeNone() const { return computeDerivativeMode; }
    void setLayoutPrimitiveCulling() { layoutPrimitiveCulling = true; }
    bool getLayoutPrimitiveCulling() const { return layoutPrimitiveCulling; }
    bool setPrimitives(int m)
    {
        if (primitives != TQualifier::layoutNotSet)
            return primitives == m;
        primitives = m;
        return true;
    }
    int getPrimitives() const { return primitives; }
    const char* addSemanticName(const TString& name)
    {
        return semanticNameSet.insert(name).first->c_str();
    }
    void addUniformLocationOverride(const char* nameStr, int location)
    {
        std::string name = nameStr;
        uniformLocationOverrides[name] = location;
    }

    int getUniformLocationOverride(const char* nameStr) const
    {
        std::string name = nameStr;
        auto pos = uniformLocationOverrides.find(name);
        if (pos == uniformLocationOverrides.end())
            return -1;
        else
            return pos->second;
    }

    void setUniformLocationBase(int base) { uniformLocationBase = base; }
    int getUniformLocationBase() const { return uniformLocationBase; }

    void setNeedsLegalization() { needToLegalize = true; }
    bool needsLegalization() const { return needToLegalize; }

    void setBinaryDoubleOutput() { binaryDoubleOutput = true; }
    bool getBinaryDoubleOutput() { return binaryDoubleOutput; }

    void setSubgroupUniformControlFlow() { subgroupUniformControlFlow = true; }
    bool getSubgroupUniformControlFlow() const { return subgroupUniformControlFlow; }

    // GL_EXT_spirv_intrinsics
    void insertSpirvRequirement(const TSpirvRequirement* spirvReq);
    bool hasSpirvRequirement() const { return spirvRequirement != nullptr; }
    const TSpirvRequirement& getSpirvRequirement() const { return *spirvRequirement; }
    void insertSpirvExecutionMode(int executionMode, const TIntermAggregate* args = nullptr);
    void insertSpirvExecutionModeId(int executionMode, const TIntermAggregate* args);
    bool hasSpirvExecutionMode() const { return spirvExecutionMode != nullptr; }
    const TSpirvExecutionMode& getSpirvExecutionMode() const { return *spirvExecutionMode; }
#endif // GLSLANG_WEB

    void addBlockStorageOverride(const char* nameStr, TBlockStorageClass backing)
    {
        std::string name(nameStr);
        blockBackingOverrides[name] = backing;
    }
    TBlockStorageClass getBlockStorageOverride(const char* nameStr) const
    {
        std::string name = nameStr;
        auto pos = blockBackingOverrides.find(name);
        if (pos == blockBackingOverrides.end())
            return EbsNone;
        else
            return pos->second;
    }
#ifdef ENABLE_HLSL
    void setHlslFunctionality1() { hlslFunctionality1 = true; }
    bool getHlslFunctionality1() const { return hlslFunctionality1; }
    void setHlslOffsets()
    {
        hlslOffsets = true;
        if (hlslOffsets)
            processes.addProcess("hlsl-offsets");
    }
    bool usingHlslOffsets() const { return hlslOffsets; }
    void setHlslIoMapping(bool b)
    {
        hlslIoMapping = b;
        if (hlslIoMapping)
            processes.addProcess("hlsl-iomap");
    }
    bool usingHlslIoMapping() { return hlslIoMapping; }
#else
    bool getHlslFunctionality1() const { return false; }
    bool usingHlslOffsets() const { return false; }
    bool usingHlslIoMapping() { return false; }
#endif

    bool usingScalarBlockLayout() const {
        for (auto extIt = requestedExtensions.begin(); extIt != requestedExtensions.end(); ++extIt) {
            if (*extIt == E_GL_EXT_scalar_block_layout)
                return true;
        }
        return false;
    }

    bool IsRequestedExtension(const char* extension) const
    {
        return (requestedExtensions.find(extension) != requestedExtensions.end());
    }

    void addToCallGraph(TInfoSink&, const TString& caller, const TString& callee);
    void merge(TInfoSink&, TIntermediate&);
    void finalCheck(TInfoSink&, bool keepUncalled);

    void mergeGlobalUniformBlocks(TInfoSink& infoSink, TIntermediate& unit, bool mergeExistingOnly);
    void mergeUniformObjects(TInfoSink& infoSink, TIntermediate& unit);
    void checkStageIO(TInfoSink&, TIntermediate&);

    bool buildConvertOp(TBasicType dst, TBasicType src, TOperator& convertOp) const;
    TIntermTyped* createConversion(TBasicType convertTo, TIntermTyped* node) const;

    void addIoAccessed(const TString& name) { ioAccessed.insert(name); }
    bool inIoAccessed(const TString& name) const { return ioAccessed.find(name) != ioAccessed.end(); }

    int addUsedLocation(const TQualifier&, const TType&, bool& typeCollision);
    int checkLocationRange(int set, const TIoRange& range, const TType&, bool& typeCollision);
    int checkLocationRT(int set, int location);
    int addUsedOffsets(int binding, int offset, int numOffsets);
    bool addUsedConstantId(int id);
    static int computeTypeLocationSize(const TType&, EShLanguage);
    static int computeTypeUniformLocationSize(const TType&);

    static int getBaseAlignmentScalar(const TType&, int& size);
    static int getBaseAlignment(const TType&, int& size, int& stride, TLayoutPacking layoutPacking, bool rowMajor);
    static int getScalarAlignment(const TType&, int& size, int& stride, bool rowMajor);
    static int getMemberAlignment(const TType&, int& size, int& stride, TLayoutPacking layoutPacking, bool rowMajor);
    static bool improperStraddle(const TType& type, int size, int offset);
    static void updateOffset(const TType& parentType, const TType& memberType, int& offset, int& memberSize);
    static int getOffset(const TType& type, int index);
    static int getBlockSize(const TType& blockType);
    static int computeBufferReferenceTypeSize(const TType&);
    static bool isIoResizeArray(const TType& type, EShLanguage language);

    bool promote(TIntermOperator*);
    void setNanMinMaxClamp(bool setting) { nanMinMaxClamp = setting; }
    bool getNanMinMaxClamp() const { return nanMinMaxClamp; }

    void setSourceFile(const char* file) { if (file != nullptr) sourceFile = file; }
    const std::string& getSourceFile() const { return sourceFile; }
    void addSourceText(const char* text, size_t len) { sourceText.append(text, len); }
    const std::string& getSourceText() const { return sourceText; }
    const std::map<std::string, std::string>& getIncludeText() const { return includeText; }
    void addIncludeText(const char* name, const char* text, size_t len) { includeText[name].assign(text,len); }
    void addProcesses(const std::vector<std::string>& p)
    {
        for (int i = 0; i < (int)p.size(); ++i)
            processes.addProcess(p[i]);
    }
    void addProcess(const std::string& process) { processes.addProcess(process); }
    void addProcessArgument(const std::string& arg) { processes.addArgument(arg); }
    const std::vector<std::string>& getProcesses() const { return processes.getProcesses(); }
    unsigned long long getUniqueId() const { return uniqueId; }
    void setUniqueId(unsigned long long id) { uniqueId = id; }

    // Certain explicit conversions are allowed conditionally
#ifdef GLSLANG_WEB
    bool getArithemeticInt8Enabled() const { return false; }
    bool getArithemeticInt16Enabled() const { return false; }
    bool getArithemeticFloat16Enabled() const { return false; }
    void updateNumericFeature(TNumericFeatures::feature f, bool on) { }
#else
    bool getArithemeticInt8Enabled() const {
        return numericFeatures.contains(TNumericFeatures::shader_explicit_arithmetic_types) ||
               numericFeatures.contains(TNumericFeatures::shader_explicit_arithmetic_types_int8);
    }
    bool getArithemeticInt16Enabled() const {
        return numericFeatures.contains(TNumericFeatures::shader_explicit_arithmetic_types) ||
               numericFeatures.contains(TNumericFeatures::gpu_shader_int16) ||
               numericFeatures.contains(TNumericFeatures::shader_explicit_arithmetic_types_int16);
    }

    bool getArithemeticFloat16Enabled() const {
        return numericFeatures.contains(TNumericFeatures::shader_explicit_arithmetic_types) ||
               numericFeatures.contains(TNumericFeatures::gpu_shader_half_float) ||
               numericFeatures.contains(TNumericFeatures::shader_explicit_arithmetic_types_float16);
    }
    void updateNumericFeature(TNumericFeatures::feature f, bool on)
        { on ? numericFeatures.insert(f) : numericFeatures.erase(f); }
#endif

protected:
    TIntermSymbol* addSymbol(long long Id, const TString&, const TType&, const TConstUnionArray&, TIntermTyped* subtree, const TSourceLoc&);
    void error(TInfoSink& infoSink, const char*);
    void warn(TInfoSink& infoSink, const char*);
    void mergeCallGraphs(TInfoSink&, TIntermediate&);
    void mergeModes(TInfoSink&, TIntermediate&);
    void mergeTrees(TInfoSink&, TIntermediate&);
    void seedIdMap(TIdMaps& idMaps, long long& IdShift);
    void remapIds(const TIdMaps& idMaps, long long idShift, TIntermediate&);
    void mergeBodies(TInfoSink&, TIntermSequence& globals, const TIntermSequence& unitGlobals);
    void mergeLinkerObjects(TInfoSink&, TIntermSequence& linkerObjects, const TIntermSequence& unitLinkerObjects, EShLanguage);
    void mergeBlockDefinitions(TInfoSink&, TIntermSymbol* block, TIntermSymbol* unitBlock, TIntermediate* unitRoot);
    void mergeImplicitArraySizes(TType&, const TType&);
    void mergeErrorCheck(TInfoSink&, const TIntermSymbol&, const TIntermSymbol&, EShLanguage);
    void checkCallGraphCycles(TInfoSink&);
    void checkCallGraphBodies(TInfoSink&, bool keepUncalled);
    void inOutLocationCheck(TInfoSink&);
    void sharedBlockCheck(TInfoSink&);
    bool userOutputUsed() const;
    bool isSpecializationOperation(const TIntermOperator&) const;
    bool isNonuniformPropagating(TOperator) const;
    bool promoteUnary(TIntermUnary&);
    bool promoteBinary(TIntermBinary&);
    void addSymbolLinkageNode(TIntermAggregate*& linkage, TSymbolTable&, const TString&);
    bool promoteAggregate(TIntermAggregate&);
    void pushSelector(TIntermSequence&, const TVectorSelector&, const TSourceLoc&);
    void pushSelector(TIntermSequence&, const TMatrixSelector&, const TSourceLoc&);
    bool specConstantPropagates(const TIntermTyped&, const TIntermTyped&);
    void performTextureUpgradeAndSamplerRemovalTransformation(TIntermNode* root);
    bool isConversionAllowed(TOperator op, TIntermTyped* node) const;
    std::tuple<TBasicType, TBasicType> getConversionDestinationType(TBasicType type0, TBasicType type1, TOperator op) const;

    static const char* getResourceName(TResourceType);

    const EShLanguage language;  // stage, known at construction time
    std::string entryPointName;
    std::string entryPointMangledName;
    typedef std::list<TCall> TGraph;
    TGraph callGraph;

#ifdef GLSLANG_ANGLE
    const EProfile profile = ECoreProfile;
    const int version = 450;
#else
    EProfile profile;                           // source profile
    int version;                                // source version
#endif
    SpvVersion spvVersion;
    TIntermNode* treeRoot;
    std::set<std::string> requestedExtensions;  // cumulation of all enabled or required extensions; not connected to what subset of the shader used them
    MustBeAssigned<TBuiltInResource> resources;
    int numEntryPoints;
    int numErrors;
    int numPushConstants;
    bool recursive;
    bool invertY;
    bool useStorageBuffer;
    bool invariantAll;
    bool nanMinMaxClamp;            // true if desiring min/max/clamp to favor non-NaN over NaN
    bool depthReplacing;
    int localSize[3];
    bool localSizeNotDefault[3];
    int localSizeSpecId[3];
    unsigned long long uniqueId;

    std::string globalUniformBlockName;
    std::string atomicCounterBlockName;
    unsigned int globalUniformBlockSet;
    unsigned int globalUniformBlockBinding;
    unsigned int atomicCounterBlockSet;

#ifndef GLSLANG_WEB
public:
    const char* const implicitThisName;
    const char* const implicitCounterName;
protected:
    EShSource source;            // source language, known a bit later
    bool useVulkanMemoryModel;
    int invocations;
    int vertices;
    TLayoutGeometry inputPrimitive;
    TLayoutGeometry outputPrimitive;
    bool pixelCenterInteger;
    bool originUpperLeft;
    TVertexSpacing vertexSpacing;
    TVertexOrder vertexOrder;
    TInterlockOrdering interlockOrdering;
    bool pointMode;
    bool earlyFragmentTests;
    bool postDepthCoverage;
    TLayoutDepth depthLayout;
    bool hlslFunctionality1;
    int blendEquations;        // an 'or'ing of masks of shifts of TBlendEquationShift
    bool xfbMode;
    std::vector<TXfbBuffer> xfbBuffers;     // all the data we need to track per xfb buffer
    bool multiStream;
    bool layoutOverrideCoverage;
    bool geoPassthroughEXT;
    int numShaderRecordBlocks;
    ComputeDerivativeMode computeDerivativeMode;
    int primitives;
    int numTaskNVBlocks;
    bool layoutPrimitiveCulling;

    // Base shift values
    std::array<unsigned int, EResCount> shiftBinding;

    // Per-descriptor-set shift values
    std::array<std::map<int, int>, EResCount> shiftBindingForSet;

    std::vector<std::string> resourceSetBinding;
    bool autoMapBindings;
    bool autoMapLocations;
    bool flattenUniformArrays;
    bool useUnknownFormat;
    bool hlslOffsets;
    bool hlslIoMapping;
    bool useVariablePointers;

    std::set<TString> semanticNameSet;

    EShTextureSamplerTransformMode textureSamplerTransformMode;

    bool needToLegalize;
    bool binaryDoubleOutput;
    bool subgroupUniformControlFlow;
    bool usePhysicalStorageBuffer;

    TSpirvRequirement* spirvRequirement;
    TSpirvExecutionMode* spirvExecutionMode;

    std::unordered_map<std::string, int> uniformLocationOverrides;
    int uniformLocationBase;
    TNumericFeatures numericFeatures;
#endif
    std::unordered_map<std::string, TBlockStorageClass> blockBackingOverrides;

    std::unordered_set<int> usedConstantId; // specialization constant ids used
    std::vector<TOffsetRange> usedAtomics;  // sets of bindings used by atomic counters
    std::vector<TIoRange> usedIo[4];        // sets of used locations, one for each of in, out, uniform, and buffers
    std::vector<TRange> usedIoRT[2];        // sets of used location, one for rayPayload/rayPayloadIN and other
                                            // for callableData/callableDataIn
    // set of names of statically read/written I/O that might need extra checking
    std::set<TString> ioAccessed;
    // source code of shader, useful as part of debug information
    std::string sourceFile;
    std::string sourceText;

    // Included text. First string is a name, second is the included text
    std::map<std::string, std::string> includeText;

    // for OpModuleProcessed, or equivalent
    TProcesses processes;

private:
    void operator=(TIntermediate&); // prevent assignments
};

} // end namespace glslang

#endif // _LOCAL_INTERMEDIATE_INCLUDED_
