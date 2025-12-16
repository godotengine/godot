//
// Copyright (C) 2014-2016 LunarG, Inc.
// Copyright (C) 2015-2020 Google, Inc.
// Copyright (C) 2017, 2022-2025 Arm Limited.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
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
// Visit the nodes in the glslang intermediate tree representation to
// translate them to SPIR-V.
//

#include "spirv.hpp11"
#include "GlslangToSpv.h"
#include "SpvBuilder.h"
#include "SpvTools.h"
#include "spvUtil.h"

namespace spv {
    #include "GLSL.std.450.h"
    #include "GLSL.ext.KHR.h"
    #include "GLSL.ext.EXT.h"
    #include "GLSL.ext.AMD.h"
    #include "GLSL.ext.NV.h"
    #include "GLSL.ext.ARM.h"
    #include "GLSL.ext.QCOM.h"
    #include "NonSemanticDebugPrintf.h"
}

// Glslang includes
#include "../glslang/MachineIndependent/localintermediate.h"
#include "../glslang/MachineIndependent/SymbolTable.h"
#include "../glslang/Include/Common.h"

// Build-time generated includes
#include "glslang/build_info.h"

#include <fstream>
#include <iomanip>
#include <list>
#include <map>
#include <optional>
#include <stack>
#include <string>
#include <vector>

namespace {

namespace {
class SpecConstantOpModeGuard {
public:
    SpecConstantOpModeGuard(spv::Builder* builder)
        : builder_(builder) {
        previous_flag_ = builder->isInSpecConstCodeGenMode();
    }
    ~SpecConstantOpModeGuard() {
        previous_flag_ ? builder_->setToSpecConstCodeGenMode()
                       : builder_->setToNormalCodeGenMode();
    }
    void turnOnSpecConstantOpMode() {
        builder_->setToSpecConstCodeGenMode();
    }

private:
    spv::Builder* builder_;
    bool previous_flag_;
};

struct OpDecorations {
    public:
        OpDecorations(spv::Decoration precision, spv::Decoration noContraction, spv::Decoration nonUniform) :
            precision(precision)
            ,
            noContraction(noContraction),
            nonUniform(nonUniform)
        { }

    spv::Decoration precision;

        void addNoContraction(spv::Builder& builder, spv::Id t) { builder.addDecoration(t, noContraction); }
        void addNonUniform(spv::Builder& builder, spv::Id t)  { builder.addDecoration(t, nonUniform); }
    protected:
        spv::Decoration noContraction;
        spv::Decoration nonUniform;
};

} // namespace

//
// The main holder of information for translating glslang to SPIR-V.
//
// Derives from the AST walking base class.
//
class TGlslangToSpvTraverser : public glslang::TIntermTraverser {
public:
    TGlslangToSpvTraverser(unsigned int spvVersion, const glslang::TIntermediate*, spv::SpvBuildLogger* logger,
        glslang::SpvOptions& options);
    virtual ~TGlslangToSpvTraverser() { }

    bool visitAggregate(glslang::TVisit, glslang::TIntermAggregate*) override;
    bool visitBinary(glslang::TVisit, glslang::TIntermBinary*) override;
    void visitConstantUnion(glslang::TIntermConstantUnion*) override;
    bool visitSelection(glslang::TVisit, glslang::TIntermSelection*) override;
    bool visitSwitch(glslang::TVisit, glslang::TIntermSwitch*) override;
    void visitSymbol(glslang::TIntermSymbol* symbol) override;
    bool visitUnary(glslang::TVisit, glslang::TIntermUnary*) override;
    bool visitLoop(glslang::TVisit, glslang::TIntermLoop*) override;
    bool visitBranch(glslang::TVisit visit, glslang::TIntermBranch*) override;
    bool visitVariableDecl(glslang::TVisit, glslang::TIntermVariableDecl*) override;

    void finishSpv(bool compileOnly);
    void dumpSpv(std::vector<unsigned int>& out);

protected:
    TGlslangToSpvTraverser(TGlslangToSpvTraverser&);
    TGlslangToSpvTraverser& operator=(TGlslangToSpvTraverser&);

    spv::Decoration TranslateInterpolationDecoration(const glslang::TQualifier& qualifier);
    spv::Decoration TranslateAuxiliaryStorageDecoration(const glslang::TQualifier& qualifier);
    spv::Decoration TranslateNonUniformDecoration(const glslang::TQualifier& qualifier);
    spv::Decoration TranslateNonUniformDecoration(const spv::Builder::AccessChain::CoherentFlags& coherentFlags);
    spv::Builder::AccessChain::CoherentFlags TranslateCoherent(const glslang::TType& type);
    spv::MemoryAccessMask TranslateMemoryAccess(const spv::Builder::AccessChain::CoherentFlags &coherentFlags);
    spv::ImageOperandsMask TranslateImageOperands(const spv::Builder::AccessChain::CoherentFlags &coherentFlags);
    spv::Scope TranslateMemoryScope(const spv::Builder::AccessChain::CoherentFlags &coherentFlags);
    spv::BuiltIn TranslateBuiltInDecoration(glslang::TBuiltInVariable, bool memberDeclaration);
    spv::ImageFormat TranslateImageFormat(const glslang::TType& type);
    spv::SelectionControlMask TranslateSelectionControl(const glslang::TIntermSelection&) const;
    spv::SelectionControlMask TranslateSwitchControl(const glslang::TIntermSwitch&) const;
    spv::LoopControlMask TranslateLoopControl(const glslang::TIntermLoop&, std::vector<unsigned int>& operands) const;
    spv::StorageClass TranslateStorageClass(const glslang::TType&);
    void TranslateLiterals(const glslang::TVector<const glslang::TIntermConstantUnion*>&, std::vector<unsigned>&) const;
    void addIndirectionIndexCapabilities(const glslang::TType& baseType, const glslang::TType& indexType);
    spv::Id createSpvVariable(const glslang::TIntermSymbol*, spv::Id forcedType);
    spv::Id getSampledType(const glslang::TSampler&);
    spv::Id getInvertedSwizzleType(const glslang::TIntermTyped&);
    spv::Id createInvertedSwizzle(spv::Decoration precision, const glslang::TIntermTyped&, spv::Id parentResult);
    void convertSwizzle(const glslang::TIntermAggregate&, std::vector<unsigned>& swizzle);
    spv::Id convertGlslangToSpvType(const glslang::TType& type, bool forwardReferenceOnly = false);
    spv::Id convertGlslangToSpvType(const glslang::TType& type, glslang::TLayoutPacking, const glslang::TQualifier&,
        bool lastBufferBlockMember, bool forwardReferenceOnly = false);
    void applySpirvDecorate(const glslang::TType& type, spv::Id id, std::optional<int> member);
    bool filterMember(const glslang::TType& member);
    spv::Id convertGlslangStructToSpvType(const glslang::TType&, const glslang::TTypeList* glslangStruct,
                                          glslang::TLayoutPacking, const glslang::TQualifier&);
    spv::LinkageType convertGlslangLinkageToSpv(glslang::TLinkType glslangLinkType);
    void decorateStructType(const glslang::TType&, const glslang::TTypeList* glslangStruct, glslang::TLayoutPacking,
                            const glslang::TQualifier&, spv::Id, const std::vector<spv::Id>& spvMembers);
    spv::Id makeArraySizeId(const glslang::TArraySizes&, int dim, bool allowZero = false, bool boolType = false);
    spv::Id accessChainLoad(const glslang::TType& type);
    void    accessChainStore(const glslang::TType& type, spv::Id rvalue);
    void multiTypeStore(const glslang::TType&, spv::Id rValue);
    spv::Id convertLoadedBoolInUniformToUint(const glslang::TType& type, spv::Id nominalTypeId, spv::Id loadedId);
    glslang::TLayoutPacking getExplicitLayout(const glslang::TType& type) const;
    int getArrayStride(const glslang::TType& arrayType, glslang::TLayoutPacking, glslang::TLayoutMatrix);
    int getMatrixStride(const glslang::TType& matrixType, glslang::TLayoutPacking, glslang::TLayoutMatrix);
    void updateMemberOffset(const glslang::TType& structType, const glslang::TType& memberType, int& currentOffset,
                            int& nextOffset, glslang::TLayoutPacking, glslang::TLayoutMatrix);
    void declareUseOfStructMember(const glslang::TTypeList& members, int glslangMember);

    bool isShaderEntryPoint(const glslang::TIntermAggregate* node);
    bool writableParam(glslang::TStorageQualifier) const;
    bool originalParam(glslang::TStorageQualifier, const glslang::TType&, bool implicitThisParam);
    void makeFunctions(const glslang::TIntermSequence&);
    void makeGlobalInitializers(const glslang::TIntermSequence&);
    void collectRayTracingLinkerObjects();
    void visitFunctions(const glslang::TIntermSequence&);
    void translateArguments(const glslang::TIntermAggregate& node, std::vector<spv::Id>& arguments,
        spv::Builder::AccessChain::CoherentFlags &lvalueCoherentFlags);
    void translateArguments(glslang::TIntermUnary& node, std::vector<spv::Id>& arguments);
    spv::Id createImageTextureFunctionCall(glslang::TIntermOperator* node);
    spv::Id handleUserFunctionCall(const glslang::TIntermAggregate*);

    spv::Id createBinaryOperation(glslang::TOperator op, OpDecorations&, spv::Id typeId, spv::Id left, spv::Id right,
                                  glslang::TBasicType typeProxy, bool reduceComparison = true);
    spv::Id createBinaryMatrixOperation(spv::Op, OpDecorations&, spv::Id typeId, spv::Id left, spv::Id right);
    spv::Id createUnaryOperation(glslang::TOperator op, OpDecorations&, spv::Id typeId, spv::Id operand,
                                 glslang::TBasicType typeProxy,
                                 const spv::Builder::AccessChain::CoherentFlags &lvalueCoherentFlags,
                                 const glslang::TType &opType);
    spv::Id createUnaryMatrixOperation(spv::Op op, OpDecorations&, spv::Id typeId, spv::Id operand,
                                       glslang::TBasicType typeProxy);
    spv::Id createConversion(glslang::TOperator op, OpDecorations&, spv::Id destTypeId, spv::Id operand,
                             glslang::TBasicType resultBasicType, glslang::TBasicType operandBasicType);
    spv::Id createIntWidthConversion(spv::Id operand, int vectorSize, spv::Id destType,
                                     glslang::TBasicType resultBasicType, glslang::TBasicType operandBasicType);
    spv::Id makeSmearedConstant(spv::Id constant, int vectorSize);
    spv::Id createAtomicOperation(glslang::TOperator op, spv::Decoration precision, spv::Id typeId,
        std::vector<spv::Id>& operands, glslang::TBasicType typeProxy,
        const spv::Builder::AccessChain::CoherentFlags &lvalueCoherentFlags,
        const glslang::TType &opType);
    spv::Id createInvocationsOperation(glslang::TOperator op, spv::Id typeId, std::vector<spv::Id>& operands,
        glslang::TBasicType typeProxy);
    spv::Id CreateInvocationsVectorOperation(spv::Op op, spv::GroupOperation groupOperation,
        spv::Id typeId, std::vector<spv::Id>& operands);
    spv::Id createSubgroupOperation(glslang::TOperator op, spv::Id typeId, std::vector<spv::Id>& operands,
        glslang::TBasicType typeProxy);
    spv::Id createMiscOperation(glslang::TOperator op, spv::Decoration precision, spv::Id typeId,
        std::vector<spv::Id>& operands, glslang::TBasicType typeProxy);
    spv::Id createNoArgOperation(glslang::TOperator op, spv::Decoration precision, spv::Id typeId);
    spv::Id getSymbolId(const glslang::TIntermSymbol* node);
    void addMeshNVDecoration(spv::Id id, int member, const glslang::TQualifier & qualifier);
    bool hasQCOMImageProceessingDecoration(spv::Id id, spv::Decoration decor);
    void addImageProcessingQCOMDecoration(spv::Id id, spv::Decoration decor);
    void addImageProcessing2QCOMDecoration(spv::Id id, bool isForGather);
    spv::Id createSpvConstant(const glslang::TIntermTyped&);
    spv::Id createSpvConstantFromConstUnionArray(const glslang::TType& type, const glslang::TConstUnionArray&,
        int& nextConst, bool specConstant);
    bool isTrivialLeaf(const glslang::TIntermTyped* node);
    bool isTrivial(const glslang::TIntermTyped* node);
    spv::Id createShortCircuit(glslang::TOperator, glslang::TIntermTyped& left, glslang::TIntermTyped& right);
    spv::Id getExtBuiltins(const char* name);
    std::pair<spv::Id, spv::Id> getForcedType(glslang::TBuiltInVariable builtIn, const glslang::TType&);
    spv::Id translateForcedType(spv::Id object);
    spv::Id createCompositeConstruct(spv::Id typeId, std::vector<spv::Id> constituents);

    glslang::SpvOptions& options;
    spv::Function* shaderEntry;
    spv::Function* currentFunction;
    spv::Instruction* entryPoint;
    int sequenceDepth;

    spv::SpvBuildLogger* logger;

    // There is a 1:1 mapping between a spv builder and a module; this is thread safe
    spv::Builder builder;
    bool inEntryPoint;
    bool entryPointTerminated;
    bool linkageOnly;                  // true when visiting the set of objects in the AST present only for
                                       // establishing interface, whether or not they were statically used
    std::set<spv::Id> iOSet;           // all input/output variables from either static use or declaration of interface
    const glslang::TIntermediate* glslangIntermediate;
    bool nanMinMaxClamp;               // true if use NMin/NMax/NClamp instead of FMin/FMax/FClamp
    spv::Id stdBuiltins;
    spv::Id nonSemanticDebugPrintf;
    std::unordered_map<std::string, spv::Id> extBuiltinMap;

    std::unordered_map<long long, spv::Id> symbolValues;
    std::unordered_map<uint32_t, spv::Id> builtInVariableIds;
    std::unordered_set<long long> rValueParameters;  // set of formal function parameters passed as rValues,
                                               // rather than a pointer
    std::unordered_map<std::string, spv::Function*> functionMap;
    std::unordered_map<const glslang::TTypeList*, spv::Id> structMap[glslang::ElpCount][glslang::ElmCount];
    // for mapping glslang block indices to spv indices (e.g., due to hidden members):
    std::unordered_map<long long, std::vector<int>> memberRemapper;
    // for mapping glslang symbol struct to symbol Id
    std::unordered_map<const glslang::TTypeList*, long long> glslangTypeToIdMap;
    std::stack<bool> breakForLoop;  // false means break for switch
    std::unordered_map<std::string, const glslang::TIntermSymbol*> counterOriginator;
    // Map pointee types for EbtReference to their forward pointers
    std::map<const glslang::TType *, spv::Id> forwardPointers;
    // Type forcing, for when SPIR-V wants a different type than the AST,
    // requiring local translation to and from SPIR-V type on every access.
    // Maps <builtin-variable-id -> AST-required-type-id>
    std::unordered_map<spv::Id, spv::Id> forceType;
    // Used by Task shader while generating opearnds for OpEmitMeshTasksEXT
    spv::Id taskPayloadID;
    // Used later for generating OpTraceKHR/OpExecuteCallableKHR/OpHitObjectRecordHit*/OpHitObjectGetShaderBindingTableData
    std::unordered_map<unsigned int, glslang::TIntermSymbol *> locationToSymbol[4];
    std::unordered_map<spv::Id, std::vector<spv::Decoration> > idToQCOMDecorations;
};

//
// Helper functions for translating glslang representations to SPIR-V enumerants.
//

// Translate glslang profile to SPIR-V source language.
spv::SourceLanguage TranslateSourceLanguage(glslang::EShSource source, EProfile profile)
{
    switch (source) {
    case glslang::EShSourceGlsl:
        switch (profile) {
        case ENoProfile:
        case ECoreProfile:
        case ECompatibilityProfile:
            return spv::SourceLanguage::GLSL;
        case EEsProfile:
            return spv::SourceLanguage::ESSL;
        default:
            return spv::SourceLanguage::Unknown;
        }
    case glslang::EShSourceHlsl:
        return spv::SourceLanguage::HLSL;
    default:
        return spv::SourceLanguage::Unknown;
    }
}

// Translate glslang language (stage) to SPIR-V execution model.
spv::ExecutionModel TranslateExecutionModel(EShLanguage stage, bool isMeshShaderEXT = false)
{
    switch (stage) {
    case EShLangVertex:           return spv::ExecutionModel::Vertex;
    case EShLangFragment:         return spv::ExecutionModel::Fragment;
    case EShLangCompute:          return spv::ExecutionModel::GLCompute;
    case EShLangTessControl:      return spv::ExecutionModel::TessellationControl;
    case EShLangTessEvaluation:   return spv::ExecutionModel::TessellationEvaluation;
    case EShLangGeometry:         return spv::ExecutionModel::Geometry;
    case EShLangRayGen:           return spv::ExecutionModel::RayGenerationKHR;
    case EShLangIntersect:        return spv::ExecutionModel::IntersectionKHR;
    case EShLangAnyHit:           return spv::ExecutionModel::AnyHitKHR;
    case EShLangClosestHit:       return spv::ExecutionModel::ClosestHitKHR;
    case EShLangMiss:             return spv::ExecutionModel::MissKHR;
    case EShLangCallable:         return spv::ExecutionModel::CallableKHR;
    case EShLangTask:             return (isMeshShaderEXT)? spv::ExecutionModel::TaskEXT : spv::ExecutionModel::TaskNV;
    case EShLangMesh:             return (isMeshShaderEXT)? spv::ExecutionModel::MeshEXT : spv::ExecutionModel::MeshNV;
    default:
        assert(0);
        return spv::ExecutionModel::Fragment;
    }
}

// Translate glslang sampler type to SPIR-V dimensionality.
spv::Dim TranslateDimensionality(const glslang::TSampler& sampler)
{
    switch (sampler.dim) {
    case glslang::Esd1D:      return spv::Dim::Dim1D;
    case glslang::Esd2D:      return spv::Dim::Dim2D;
    case glslang::Esd3D:      return spv::Dim::Dim3D;
    case glslang::EsdCube:    return spv::Dim::Cube;
    case glslang::EsdRect:    return spv::Dim::Rect;
    case glslang::EsdBuffer:  return spv::Dim::Buffer;
    case glslang::EsdSubpass: return spv::Dim::SubpassData;
    case glslang::EsdAttachmentEXT: return spv::Dim::TileImageDataEXT;
    default:
        assert(0);
        return spv::Dim::Dim2D;
    }
}

// Translate glslang precision to SPIR-V precision decorations.
spv::Decoration TranslatePrecisionDecoration(glslang::TPrecisionQualifier glslangPrecision)
{
    switch (glslangPrecision) {
    case glslang::EpqLow:    return spv::Decoration::RelaxedPrecision;
    case glslang::EpqMedium: return spv::Decoration::RelaxedPrecision;
    default:
        return spv::NoPrecision;
    }
}

// Translate glslang type to SPIR-V precision decorations.
spv::Decoration TranslatePrecisionDecoration(const glslang::TType& type)
{
    return TranslatePrecisionDecoration(type.getQualifier().precision);
}

// Translate glslang type to SPIR-V block decorations.
spv::Decoration TranslateBlockDecoration(const glslang::TStorageQualifier storage, bool useStorageBuffer)
{
    switch (storage) {
    case glslang::EvqUniform:      return spv::Decoration::Block;
    case glslang::EvqBuffer:       return useStorageBuffer ? spv::Decoration::Block : spv::Decoration::BufferBlock;
    case glslang::EvqVaryingIn:    return spv::Decoration::Block;
    case glslang::EvqVaryingOut:   return spv::Decoration::Block;
    case glslang::EvqShared:       return spv::Decoration::Block;
    case glslang::EvqPayload:      return spv::Decoration::Block;
    case glslang::EvqPayloadIn:    return spv::Decoration::Block;
    case glslang::EvqHitAttr:      return spv::Decoration::Block;
    case glslang::EvqCallableData:   return spv::Decoration::Block;
    case glslang::EvqCallableDataIn: return spv::Decoration::Block;
    case glslang::EvqHitObjectAttrNV: return spv::Decoration::Block;
    case glslang::EvqHitObjectAttrEXT: return spv::Decoration::Block;
    default:
        assert(0);
        break;
    }

    return spv::Decoration::Max;
}

// Translate glslang type to SPIR-V memory decorations.
void TranslateMemoryDecoration(const glslang::TQualifier& qualifier, std::vector<spv::Decoration>& memory,
    bool useVulkanMemoryModel)
{
    if (!useVulkanMemoryModel) {
        if (qualifier.isVolatile()) {
            memory.push_back(spv::Decoration::Volatile);
            memory.push_back(spv::Decoration::Coherent);
        } else if (qualifier.isCoherent()) {
            memory.push_back(spv::Decoration::Coherent);
        }
    }
    if (qualifier.isRestrict())
        memory.push_back(spv::Decoration::Restrict);
    if (qualifier.isReadOnly())
        memory.push_back(spv::Decoration::NonWritable);
    if (qualifier.isWriteOnly())
        memory.push_back(spv::Decoration::NonReadable);
}

// Translate glslang type to SPIR-V layout decorations.
spv::Decoration TranslateLayoutDecoration(const glslang::TType& type, glslang::TLayoutMatrix matrixLayout)
{
    if (type.isMatrix()) {
        switch (matrixLayout) {
        case glslang::ElmRowMajor:
            return spv::Decoration::RowMajor;
        case glslang::ElmColumnMajor:
            return spv::Decoration::ColMajor;
        default:
            // opaque layouts don't need a majorness
            return spv::Decoration::Max;
        }
    } else {
        switch (type.getBasicType()) {
        default:
            return spv::Decoration::Max;
            break;
        case glslang::EbtBlock:
            switch (type.getQualifier().storage) {
            case glslang::EvqShared:
            case glslang::EvqUniform:
            case glslang::EvqBuffer:
                switch (type.getQualifier().layoutPacking) {
                case glslang::ElpShared:  return spv::Decoration::GLSLShared;
                case glslang::ElpPacked:  return spv::Decoration::GLSLPacked;
                default:
                    return spv::Decoration::Max;
                }
            case glslang::EvqVaryingIn:
            case glslang::EvqVaryingOut:
                if (type.getQualifier().isTaskMemory()) {
                    switch (type.getQualifier().layoutPacking) {
                    case glslang::ElpShared:  return spv::Decoration::GLSLShared;
                    case glslang::ElpPacked:  return spv::Decoration::GLSLPacked;
                    default: break;
                    }
                } else {
                    assert(type.getQualifier().layoutPacking == glslang::ElpNone);
                }
                return spv::Decoration::Max;
            case glslang::EvqPayload:
            case glslang::EvqPayloadIn:
            case glslang::EvqHitAttr:
            case glslang::EvqCallableData:
            case glslang::EvqCallableDataIn:
            case glslang::EvqHitObjectAttrNV:
            case glslang::EvqHitObjectAttrEXT:
                return spv::Decoration::Max;
            default:
                assert(0);
                return spv::Decoration::Max;
            }
        }
    }
}

// Translate glslang type to SPIR-V interpolation decorations.
// Returns spv::Decoration::Max when no decoration
// should be applied.
spv::Decoration TGlslangToSpvTraverser::TranslateInterpolationDecoration(const glslang::TQualifier& qualifier)
{
    if (qualifier.smooth)
        // Smooth decoration doesn't exist in SPIR-V 1.0
        return spv::Decoration::Max;
    else if (qualifier.isNonPerspective())
        return spv::Decoration::NoPerspective;
    else if (qualifier.flat)
        return spv::Decoration::Flat;
    else if (qualifier.isExplicitInterpolation()) {
        builder.addExtension(spv::E_SPV_AMD_shader_explicit_vertex_parameter);
        return spv::Decoration::ExplicitInterpAMD;
    }
    else
        return spv::Decoration::Max;
}

// Translate glslang type to SPIR-V auxiliary storage decorations.
// Returns spv::Decoration::Max when no decoration
// should be applied.
spv::Decoration TGlslangToSpvTraverser::TranslateAuxiliaryStorageDecoration(const glslang::TQualifier& qualifier)
{
    if (qualifier.centroid)
        return spv::Decoration::Centroid;
    else if (qualifier.patch)
        return spv::Decoration::Patch;
    else if (qualifier.sample) {
        builder.addCapability(spv::Capability::SampleRateShading);
        return spv::Decoration::Sample;
    }

    return spv::Decoration::Max;
}

// If glslang type is invariant, return SPIR-V invariant decoration.
spv::Decoration TranslateInvariantDecoration(const glslang::TQualifier& qualifier)
{
    if (qualifier.invariant)
        return spv::Decoration::Invariant;
    else
        return spv::Decoration::Max;
}

// If glslang type is noContraction, return SPIR-V NoContraction decoration.
spv::Decoration TranslateNoContractionDecoration(const glslang::TQualifier& qualifier)
{
    if (qualifier.isNoContraction())
        return spv::Decoration::NoContraction;
    else
        return spv::Decoration::Max;
}

// If glslang type is nonUniform, return SPIR-V NonUniform decoration.
spv::Decoration TGlslangToSpvTraverser::TranslateNonUniformDecoration(const glslang::TQualifier& qualifier)
{
    if (qualifier.isNonUniform()) {
        builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
        builder.addCapability(spv::Capability::ShaderNonUniformEXT);
        return spv::Decoration::NonUniformEXT;
    } else
        return spv::Decoration::Max;
}

// If lvalue flags contains nonUniform, return SPIR-V NonUniform decoration.
spv::Decoration TGlslangToSpvTraverser::TranslateNonUniformDecoration(
    const spv::Builder::AccessChain::CoherentFlags& coherentFlags)
{
    if (coherentFlags.isNonUniform()) {
        builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
        builder.addCapability(spv::Capability::ShaderNonUniformEXT);
        return spv::Decoration::NonUniformEXT;
    } else
        return spv::Decoration::Max;
}

spv::MemoryAccessMask TGlslangToSpvTraverser::TranslateMemoryAccess(
    const spv::Builder::AccessChain::CoherentFlags &coherentFlags)
{
    spv::MemoryAccessMask mask = spv::MemoryAccessMask::MaskNone;

    if (!glslangIntermediate->usingVulkanMemoryModel() || coherentFlags.isImage)
        return mask;

    if (coherentFlags.isVolatile() || coherentFlags.anyCoherent()) {
        mask = mask | spv::MemoryAccessMask::MakePointerAvailableKHR | 
                      spv::MemoryAccessMask::MakePointerVisibleKHR;
    }

    if (coherentFlags.nonprivate) {
        mask = mask | spv::MemoryAccessMask::NonPrivatePointerKHR;
    }
    if (coherentFlags.volatil) {
        mask = mask | spv::MemoryAccessMask::Volatile;
    }
    if (coherentFlags.nontemporal) {
        mask = mask | spv::MemoryAccessMask::Nontemporal;
    }
    if (mask != spv::MemoryAccessMask::MaskNone) {
        builder.addCapability(spv::Capability::VulkanMemoryModelKHR);
    }

    return mask;
}

spv::ImageOperandsMask TGlslangToSpvTraverser::TranslateImageOperands(
    const spv::Builder::AccessChain::CoherentFlags &coherentFlags)
{
    spv::ImageOperandsMask mask = spv::ImageOperandsMask::MaskNone;

    if (!glslangIntermediate->usingVulkanMemoryModel())
        return mask;

    if (coherentFlags.volatil ||
        coherentFlags.anyCoherent()) {
        mask = mask | spv::ImageOperandsMask::MakeTexelAvailableKHR |
                      spv::ImageOperandsMask::MakeTexelVisibleKHR;
    }
    if (coherentFlags.nonprivate) {
        mask = mask | spv::ImageOperandsMask::NonPrivateTexelKHR;
    }
    if (coherentFlags.volatil) {
        mask = mask | spv::ImageOperandsMask::VolatileTexelKHR;
    }
    if (coherentFlags.nontemporal && builder.getSpvVersion() >= spv::Spv_1_6) {
        mask = mask | spv::ImageOperandsMask::Nontemporal;
    }
    if (mask != spv::ImageOperandsMask::MaskNone) {
        builder.addCapability(spv::Capability::VulkanMemoryModelKHR);
    }

    return mask;
}

spv::Builder::AccessChain::CoherentFlags TGlslangToSpvTraverser::TranslateCoherent(const glslang::TType& type)
{
    spv::Builder::AccessChain::CoherentFlags flags = {};
    flags.coherent = type.getQualifier().coherent;
    flags.devicecoherent = type.getQualifier().devicecoherent;
    flags.queuefamilycoherent = type.getQualifier().queuefamilycoherent;
    // shared variables are implicitly workgroupcoherent in GLSL.
    flags.workgroupcoherent = type.getQualifier().workgroupcoherent ||
                              type.getQualifier().storage == glslang::EvqShared;
    flags.subgroupcoherent = type.getQualifier().subgroupcoherent;
    flags.shadercallcoherent = type.getQualifier().shadercallcoherent;
    flags.volatil = type.getQualifier().volatil;
    flags.nontemporal = type.getQualifier().nontemporal;
    // *coherent variables are implicitly nonprivate in GLSL
    flags.nonprivate = type.getQualifier().nonprivate ||
                       flags.anyCoherent() ||
                       flags.volatil;
    flags.isImage = type.getBasicType() == glslang::EbtSampler;
    flags.nonUniform = type.getQualifier().nonUniform;
    return flags;
}

spv::Scope TGlslangToSpvTraverser::TranslateMemoryScope(
    const spv::Builder::AccessChain::CoherentFlags &coherentFlags)
{
    spv::Scope scope = spv::Scope::Max;

    if (coherentFlags.volatil || coherentFlags.coherent) {
        // coherent defaults to Device scope in the old model, QueueFamilyKHR scope in the new model
        scope = glslangIntermediate->usingVulkanMemoryModel() ? spv::Scope::QueueFamilyKHR : spv::Scope::Device;
    } else if (coherentFlags.devicecoherent) {
        scope = spv::Scope::Device;
    } else if (coherentFlags.queuefamilycoherent) {
        scope = spv::Scope::QueueFamilyKHR;
    } else if (coherentFlags.workgroupcoherent) {
        scope = spv::Scope::Workgroup;
    } else if (coherentFlags.subgroupcoherent) {
        scope = spv::Scope::Subgroup;
    } else if (coherentFlags.shadercallcoherent) {
        scope = spv::Scope::ShaderCallKHR;
    }
    if (glslangIntermediate->usingVulkanMemoryModel() && scope == spv::Scope::Device) {
        builder.addCapability(spv::Capability::VulkanMemoryModelDeviceScopeKHR);
    }

    return scope;
}

// Translate a glslang built-in variable to a SPIR-V built in decoration.  Also generate
// associated capabilities when required.  For some built-in variables, a capability
// is generated only when using the variable in an executable instruction, but not when
// just declaring a struct member variable with it.  This is true for PointSize,
// ClipDistance, and CullDistance.
spv::BuiltIn TGlslangToSpvTraverser::TranslateBuiltInDecoration(glslang::TBuiltInVariable builtIn,
    bool memberDeclaration)
{
    switch (builtIn) {
    case glslang::EbvPointSize:
        // Defer adding the capability until the built-in is actually used.
        if (! memberDeclaration) {
            switch (glslangIntermediate->getStage()) {
            case EShLangGeometry:
                builder.addCapability(spv::Capability::GeometryPointSize);
                break;
            case EShLangTessControl:
            case EShLangTessEvaluation:
                builder.addCapability(spv::Capability::TessellationPointSize);
                break;
            default:
                break;
            }
        }
        return spv::BuiltIn::PointSize;

    case glslang::EbvPosition:             return spv::BuiltIn::Position;
    case glslang::EbvVertexId:             return spv::BuiltIn::VertexId;
    case glslang::EbvInstanceId:           return spv::BuiltIn::InstanceId;
    case glslang::EbvVertexIndex:          return spv::BuiltIn::VertexIndex;
    case glslang::EbvInstanceIndex:        return spv::BuiltIn::InstanceIndex;

    case glslang::EbvFragCoord:            return spv::BuiltIn::FragCoord;
    case glslang::EbvPointCoord:           return spv::BuiltIn::PointCoord;
    case glslang::EbvFace:                 return spv::BuiltIn::FrontFacing;
    case glslang::EbvFragDepth:            return spv::BuiltIn::FragDepth;

    case glslang::EbvNumWorkGroups:        return spv::BuiltIn::NumWorkgroups;
    case glslang::EbvWorkGroupSize:        return spv::BuiltIn::WorkgroupSize;
    case glslang::EbvWorkGroupId:          return spv::BuiltIn::WorkgroupId;
    case glslang::EbvLocalInvocationId:    return spv::BuiltIn::LocalInvocationId;
    case glslang::EbvLocalInvocationIndex: return spv::BuiltIn::LocalInvocationIndex;
    case glslang::EbvGlobalInvocationId:   return spv::BuiltIn::GlobalInvocationId;

    // These *Distance capabilities logically belong here, but if the member is declared and
    // then never used, consumers of SPIR-V prefer the capability not be declared.
    // They are now generated when used, rather than here when declared.
    // Potentially, the specification should be more clear what the minimum
    // use needed is to trigger the capability.
    //
    case glslang::EbvClipDistance:
        if (!memberDeclaration)
            builder.addCapability(spv::Capability::ClipDistance);
        return spv::BuiltIn::ClipDistance;

    case glslang::EbvCullDistance:
        if (!memberDeclaration)
            builder.addCapability(spv::Capability::CullDistance);
        return spv::BuiltIn::CullDistance;

    case glslang::EbvViewportIndex:
        if (glslangIntermediate->getStage() == EShLangGeometry ||
            glslangIntermediate->getStage() == EShLangFragment) {
            builder.addCapability(spv::Capability::MultiViewport);
        }
        if (glslangIntermediate->getStage() == EShLangVertex ||
            glslangIntermediate->getStage() == EShLangTessControl ||
            glslangIntermediate->getStage() == EShLangTessEvaluation) {

            if (builder.getSpvVersion() < spv::Spv_1_5) {
                builder.addIncorporatedExtension(spv::E_SPV_EXT_shader_viewport_index_layer, spv::Spv_1_5);
                builder.addCapability(spv::Capability::ShaderViewportIndexLayerEXT);
            }
            else
                builder.addCapability(spv::Capability::ShaderViewportIndex);
        }
        return spv::BuiltIn::ViewportIndex;

    case glslang::EbvSampleId:
        builder.addCapability(spv::Capability::SampleRateShading);
        return spv::BuiltIn::SampleId;

    case glslang::EbvSamplePosition:
        builder.addCapability(spv::Capability::SampleRateShading);
        return spv::BuiltIn::SamplePosition;

    case glslang::EbvSampleMask:
        return spv::BuiltIn::SampleMask;

    case glslang::EbvLayer:
        if (glslangIntermediate->getStage() == EShLangMesh) {
            return spv::BuiltIn::Layer;
        }
        if (glslangIntermediate->getStage() == EShLangGeometry ||
            glslangIntermediate->getStage() == EShLangFragment) {
            builder.addCapability(spv::Capability::Geometry);
        }
        if (glslangIntermediate->getStage() == EShLangVertex ||
            glslangIntermediate->getStage() == EShLangTessControl ||
            glslangIntermediate->getStage() == EShLangTessEvaluation) {

            if (builder.getSpvVersion() < spv::Spv_1_5) {
                builder.addIncorporatedExtension(spv::E_SPV_EXT_shader_viewport_index_layer, spv::Spv_1_5);
                builder.addCapability(spv::Capability::ShaderViewportIndexLayerEXT);
            } else
                builder.addCapability(spv::Capability::ShaderLayer);
        }
        return spv::BuiltIn::Layer;

    case glslang::EbvBaseVertex:
        builder.addIncorporatedExtension(spv::E_SPV_KHR_shader_draw_parameters, spv::Spv_1_3);
        builder.addCapability(spv::Capability::DrawParameters);
        return spv::BuiltIn::BaseVertex;

    case glslang::EbvBaseInstance:
        builder.addIncorporatedExtension(spv::E_SPV_KHR_shader_draw_parameters, spv::Spv_1_3);
        builder.addCapability(spv::Capability::DrawParameters);
        return spv::BuiltIn::BaseInstance;

    case glslang::EbvDrawId:
        builder.addIncorporatedExtension(spv::E_SPV_KHR_shader_draw_parameters, spv::Spv_1_3);
        builder.addCapability(spv::Capability::DrawParameters);
        return spv::BuiltIn::DrawIndex;

    case glslang::EbvPrimitiveId:
        if (glslangIntermediate->getStage() == EShLangFragment)
            builder.addCapability(spv::Capability::Geometry);
        return spv::BuiltIn::PrimitiveId;

    case glslang::EbvFragStencilRef:
        builder.addExtension(spv::E_SPV_EXT_shader_stencil_export);
        builder.addCapability(spv::Capability::StencilExportEXT);
        return spv::BuiltIn::FragStencilRefEXT;

    case glslang::EbvShadingRateKHR:
        builder.addExtension(spv::E_SPV_KHR_fragment_shading_rate);
        builder.addCapability(spv::Capability::FragmentShadingRateKHR);
        return spv::BuiltIn::ShadingRateKHR;

    case glslang::EbvPrimitiveShadingRateKHR:
        builder.addExtension(spv::E_SPV_KHR_fragment_shading_rate);
        builder.addCapability(spv::Capability::FragmentShadingRateKHR);
        return spv::BuiltIn::PrimitiveShadingRateKHR;

    case glslang::EbvInvocationId:         return spv::BuiltIn::InvocationId;
    case glslang::EbvTessLevelInner:       return spv::BuiltIn::TessLevelInner;
    case glslang::EbvTessLevelOuter:       return spv::BuiltIn::TessLevelOuter;
    case glslang::EbvTessCoord:            return spv::BuiltIn::TessCoord;
    case glslang::EbvPatchVertices:        return spv::BuiltIn::PatchVertices;
    case glslang::EbvHelperInvocation:     return spv::BuiltIn::HelperInvocation;

    case glslang::EbvSubGroupSize:
        builder.addExtension(spv::E_SPV_KHR_shader_ballot);
        builder.addCapability(spv::Capability::SubgroupBallotKHR);
        return spv::BuiltIn::SubgroupSize;

    case glslang::EbvSubGroupInvocation:
        builder.addExtension(spv::E_SPV_KHR_shader_ballot);
        builder.addCapability(spv::Capability::SubgroupBallotKHR);
        return spv::BuiltIn::SubgroupLocalInvocationId;

    case glslang::EbvSubGroupEqMask:
        builder.addExtension(spv::E_SPV_KHR_shader_ballot);
        builder.addCapability(spv::Capability::SubgroupBallotKHR);
        return spv::BuiltIn::SubgroupEqMask;

    case glslang::EbvSubGroupGeMask:
        builder.addExtension(spv::E_SPV_KHR_shader_ballot);
        builder.addCapability(spv::Capability::SubgroupBallotKHR);
        return spv::BuiltIn::SubgroupGeMask;

    case glslang::EbvSubGroupGtMask:
        builder.addExtension(spv::E_SPV_KHR_shader_ballot);
        builder.addCapability(spv::Capability::SubgroupBallotKHR);
        return spv::BuiltIn::SubgroupGtMask;

    case glslang::EbvSubGroupLeMask:
        builder.addExtension(spv::E_SPV_KHR_shader_ballot);
        builder.addCapability(spv::Capability::SubgroupBallotKHR);
        return spv::BuiltIn::SubgroupLeMask;

    case glslang::EbvSubGroupLtMask:
        builder.addExtension(spv::E_SPV_KHR_shader_ballot);
        builder.addCapability(spv::Capability::SubgroupBallotKHR);
        return spv::BuiltIn::SubgroupLtMask;

    case glslang::EbvNumSubgroups:
        builder.addCapability(spv::Capability::GroupNonUniform);
        return spv::BuiltIn::NumSubgroups;

    case glslang::EbvSubgroupID:
        builder.addCapability(spv::Capability::GroupNonUniform);
        return spv::BuiltIn::SubgroupId;

    case glslang::EbvSubgroupSize2:
        builder.addCapability(spv::Capability::GroupNonUniform);
        return spv::BuiltIn::SubgroupSize;

    case glslang::EbvSubgroupInvocation2:
        builder.addCapability(spv::Capability::GroupNonUniform);
        return spv::BuiltIn::SubgroupLocalInvocationId;

    case glslang::EbvSubgroupEqMask2:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformBallot);
        return spv::BuiltIn::SubgroupEqMask;

    case glslang::EbvSubgroupGeMask2:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformBallot);
        return spv::BuiltIn::SubgroupGeMask;

    case glslang::EbvSubgroupGtMask2:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformBallot);
        return spv::BuiltIn::SubgroupGtMask;

    case glslang::EbvSubgroupLeMask2:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformBallot);
        return spv::BuiltIn::SubgroupLeMask;

    case glslang::EbvSubgroupLtMask2:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformBallot);
        return spv::BuiltIn::SubgroupLtMask;

    case glslang::EbvBaryCoordNoPersp:
        builder.addExtension(spv::E_SPV_AMD_shader_explicit_vertex_parameter);
        return spv::BuiltIn::BaryCoordNoPerspAMD;

    case glslang::EbvBaryCoordNoPerspCentroid:
        builder.addExtension(spv::E_SPV_AMD_shader_explicit_vertex_parameter);
        return spv::BuiltIn::BaryCoordNoPerspCentroidAMD;

    case glslang::EbvBaryCoordNoPerspSample:
        builder.addExtension(spv::E_SPV_AMD_shader_explicit_vertex_parameter);
        return spv::BuiltIn::BaryCoordNoPerspSampleAMD;

    case glslang::EbvBaryCoordSmooth:
        builder.addExtension(spv::E_SPV_AMD_shader_explicit_vertex_parameter);
        return spv::BuiltIn::BaryCoordSmoothAMD;

    case glslang::EbvBaryCoordSmoothCentroid:
        builder.addExtension(spv::E_SPV_AMD_shader_explicit_vertex_parameter);
        return spv::BuiltIn::BaryCoordSmoothCentroidAMD;

    case glslang::EbvBaryCoordSmoothSample:
        builder.addExtension(spv::E_SPV_AMD_shader_explicit_vertex_parameter);
        return spv::BuiltIn::BaryCoordSmoothSampleAMD;

    case glslang::EbvBaryCoordPullModel:
        builder.addExtension(spv::E_SPV_AMD_shader_explicit_vertex_parameter);
        return spv::BuiltIn::BaryCoordPullModelAMD;

    case glslang::EbvDeviceIndex:
        builder.addIncorporatedExtension(spv::E_SPV_KHR_device_group, spv::Spv_1_3);
        builder.addCapability(spv::Capability::DeviceGroup);
        return spv::BuiltIn::DeviceIndex;

    case glslang::EbvViewIndex:
        builder.addIncorporatedExtension(spv::E_SPV_KHR_multiview, spv::Spv_1_3);
        builder.addCapability(spv::Capability::MultiView);
        return spv::BuiltIn::ViewIndex;

    case glslang::EbvFragSizeEXT:
        builder.addExtension(spv::E_SPV_EXT_fragment_invocation_density);
        builder.addCapability(spv::Capability::FragmentDensityEXT);
        return spv::BuiltIn::FragSizeEXT;

    case glslang::EbvFragInvocationCountEXT:
        builder.addExtension(spv::E_SPV_EXT_fragment_invocation_density);
        builder.addCapability(spv::Capability::FragmentDensityEXT);
        return spv::BuiltIn::FragInvocationCountEXT;

    case glslang::EbvViewportMaskNV:
        if (!memberDeclaration) {
            builder.addExtension(spv::E_SPV_NV_viewport_array2);
            builder.addCapability(spv::Capability::ShaderViewportMaskNV);
        }
        return spv::BuiltIn::ViewportMaskNV;
    case glslang::EbvSecondaryPositionNV:
        if (!memberDeclaration) {
            builder.addExtension(spv::E_SPV_NV_stereo_view_rendering);
            builder.addCapability(spv::Capability::ShaderStereoViewNV);
        }
        return spv::BuiltIn::SecondaryPositionNV;
    case glslang::EbvSecondaryViewportMaskNV:
        if (!memberDeclaration) {
            builder.addExtension(spv::E_SPV_NV_stereo_view_rendering);
            builder.addCapability(spv::Capability::ShaderStereoViewNV);
        }
        return spv::BuiltIn::SecondaryViewportMaskNV;
    case glslang::EbvPositionPerViewNV:
        if (!memberDeclaration) {
            builder.addExtension(spv::E_SPV_NVX_multiview_per_view_attributes);
            builder.addCapability(spv::Capability::PerViewAttributesNV);
        }
        return spv::BuiltIn::PositionPerViewNV;
    case glslang::EbvViewportMaskPerViewNV:
        if (!memberDeclaration) {
            builder.addExtension(spv::E_SPV_NVX_multiview_per_view_attributes);
            builder.addCapability(spv::Capability::PerViewAttributesNV);
        }
        return spv::BuiltIn::ViewportMaskPerViewNV;
    case glslang::EbvFragFullyCoveredNV:
        builder.addExtension(spv::E_SPV_EXT_fragment_fully_covered);
        builder.addCapability(spv::Capability::FragmentFullyCoveredEXT);
        return spv::BuiltIn::FullyCoveredEXT;
    case glslang::EbvFragmentSizeNV:
        builder.addExtension(spv::E_SPV_NV_shading_rate);
        builder.addCapability(spv::Capability::ShadingRateNV);
        return spv::BuiltIn::FragmentSizeNV;
    case glslang::EbvInvocationsPerPixelNV:
        builder.addExtension(spv::E_SPV_NV_shading_rate);
        builder.addCapability(spv::Capability::ShadingRateNV);
        return spv::BuiltIn::InvocationsPerPixelNV;

    // ray tracing
    case glslang::EbvLaunchId:
        return spv::BuiltIn::LaunchIdKHR;
    case glslang::EbvLaunchSize:
        return spv::BuiltIn::LaunchSizeKHR;
    case glslang::EbvWorldRayOrigin:
        return spv::BuiltIn::WorldRayOriginKHR;
    case glslang::EbvWorldRayDirection:
        return spv::BuiltIn::WorldRayDirectionKHR;
    case glslang::EbvObjectRayOrigin:
        return spv::BuiltIn::ObjectRayOriginKHR;
    case glslang::EbvObjectRayDirection:
        return spv::BuiltIn::ObjectRayDirectionKHR;
    case glslang::EbvRayTmin:
        return spv::BuiltIn::RayTminKHR;
    case glslang::EbvRayTmax:
        return spv::BuiltIn::RayTmaxKHR;
    case glslang::EbvCullMask:
        return spv::BuiltIn::CullMaskKHR;
    case glslang::EbvPositionFetch:
        return spv::BuiltIn::HitTriangleVertexPositionsKHR;
    case glslang::EbvInstanceCustomIndex:
        return spv::BuiltIn::InstanceCustomIndexKHR;
    case glslang::EbvHitKind:
        return spv::BuiltIn::HitKindKHR;
    case glslang::EbvObjectToWorld:
    case glslang::EbvObjectToWorld3x4:
        return spv::BuiltIn::ObjectToWorldKHR;
    case glslang::EbvWorldToObject:
    case glslang::EbvWorldToObject3x4:
        return spv::BuiltIn::WorldToObjectKHR;
    case glslang::EbvIncomingRayFlags:
        return spv::BuiltIn::IncomingRayFlagsKHR;
    case glslang::EbvGeometryIndex:
        return spv::BuiltIn::RayGeometryIndexKHR;
    case glslang::EbvCurrentRayTimeNV:
        builder.addExtension(spv::E_SPV_NV_ray_tracing_motion_blur);
        builder.addCapability(spv::Capability::RayTracingMotionBlurNV);
        return spv::BuiltIn::CurrentRayTimeNV;
    case glslang::EbvMicroTrianglePositionNV:
        builder.addCapability(spv::Capability::RayTracingDisplacementMicromapNV);
        builder.addExtension("SPV_NV_displacement_micromap");
        return spv::BuiltIn::HitMicroTriangleVertexPositionsNV;
    case glslang::EbvMicroTriangleBaryNV:
        builder.addCapability(spv::Capability::RayTracingDisplacementMicromapNV);
        builder.addExtension("SPV_NV_displacement_micromap");
        return spv::BuiltIn::HitMicroTriangleVertexBarycentricsNV;
    case glslang::EbvHitKindFrontFacingMicroTriangleNV:
        builder.addCapability(spv::Capability::RayTracingDisplacementMicromapNV);
        builder.addExtension("SPV_NV_displacement_micromap");
        return spv::BuiltIn::HitKindFrontFacingMicroTriangleNV;
    case glslang::EbvHitKindBackFacingMicroTriangleNV:
        builder.addCapability(spv::Capability::RayTracingDisplacementMicromapNV);
        builder.addExtension("SPV_NV_displacement_micromap");
        return spv::BuiltIn::HitKindBackFacingMicroTriangleNV;
    case glslang::EbvClusterIDNV:
        builder.addCapability(spv::Capability::RayTracingClusterAccelerationStructureNV);
        builder.addExtension("SPV_NV_cluster_acceleration_structure");
        return spv::BuiltIn::ClusterIDNV;
    case glslang::EbvHitIsSphereNV:
        builder.addCapability(spv::Capability::RayTracingSpheresGeometryNV);
        builder.addExtension("SPV_NV_linear_swept_spheres");
        return spv::BuiltIn::HitIsSphereNV;
    case glslang::EbvHitIsLSSNV:
        builder.addCapability(spv::Capability::RayTracingLinearSweptSpheresGeometryNV);
        builder.addExtension("SPV_NV_linear_swept_spheres");
        return spv::BuiltIn::HitIsLSSNV;
    case glslang::EbvHitSpherePositionNV:
        builder.addCapability(spv::Capability::RayTracingSpheresGeometryNV);
        builder.addExtension("SPV_NV_linear_swept_spheres");
        return spv::BuiltIn::HitSpherePositionNV;
    case glslang::EbvHitSphereRadiusNV:
        builder.addCapability(spv::Capability::RayTracingSpheresGeometryNV);
        builder.addExtension("SPV_NV_linear_swept_spheres");
        return spv::BuiltIn::HitSphereRadiusNV;
    case glslang::EbvHitLSSPositionsNV:
        builder.addCapability(spv::Capability::RayTracingLinearSweptSpheresGeometryNV);
        builder.addExtension("SPV_NV_linear_swept_spheres");
        return spv::BuiltIn::HitLSSPositionsNV;
    case glslang::EbvHitLSSRadiiNV:
        builder.addCapability(spv::Capability::RayTracingLinearSweptSpheresGeometryNV);
        builder.addExtension("SPV_NV_linear_swept_spheres");
        return spv::BuiltIn::HitLSSRadiiNV;

    // barycentrics
    case glslang::EbvBaryCoordNV:
        builder.addExtension(spv::E_SPV_NV_fragment_shader_barycentric);
        builder.addCapability(spv::Capability::FragmentBarycentricNV);
        return spv::BuiltIn::BaryCoordNV;
    case glslang::EbvBaryCoordNoPerspNV:
        builder.addExtension(spv::E_SPV_NV_fragment_shader_barycentric);
        builder.addCapability(spv::Capability::FragmentBarycentricNV);
        return spv::BuiltIn::BaryCoordNoPerspNV;

    case glslang::EbvBaryCoordEXT:
        builder.addExtension(spv::E_SPV_KHR_fragment_shader_barycentric);
        builder.addCapability(spv::Capability::FragmentBarycentricKHR);
        return spv::BuiltIn::BaryCoordKHR;
    case glslang::EbvBaryCoordNoPerspEXT:
        builder.addExtension(spv::E_SPV_KHR_fragment_shader_barycentric);
        builder.addCapability(spv::Capability::FragmentBarycentricKHR);
        return spv::BuiltIn::BaryCoordNoPerspKHR;

    // mesh shaders
    case glslang::EbvTaskCountNV:
        return spv::BuiltIn::TaskCountNV;
    case glslang::EbvPrimitiveCountNV:
        return spv::BuiltIn::PrimitiveCountNV;
    case glslang::EbvPrimitiveIndicesNV:
        return spv::BuiltIn::PrimitiveIndicesNV;
    case glslang::EbvClipDistancePerViewNV:
        return spv::BuiltIn::ClipDistancePerViewNV;
    case glslang::EbvCullDistancePerViewNV:
        return spv::BuiltIn::CullDistancePerViewNV;
    case glslang::EbvLayerPerViewNV:
        return spv::BuiltIn::LayerPerViewNV;
    case glslang::EbvMeshViewCountNV:
        return spv::BuiltIn::MeshViewCountNV;
    case glslang::EbvMeshViewIndicesNV:
        return spv::BuiltIn::MeshViewIndicesNV;

    // SPV_EXT_mesh_shader
    case glslang::EbvPrimitivePointIndicesEXT:
        return spv::BuiltIn::PrimitivePointIndicesEXT;
    case glslang::EbvPrimitiveLineIndicesEXT:
        return spv::BuiltIn::PrimitiveLineIndicesEXT;
    case glslang::EbvPrimitiveTriangleIndicesEXT:
        return spv::BuiltIn::PrimitiveTriangleIndicesEXT;
    case glslang::EbvCullPrimitiveEXT:
        return spv::BuiltIn::CullPrimitiveEXT;

    // sm builtins
    case glslang::EbvWarpsPerSM:
        builder.addExtension(spv::E_SPV_NV_shader_sm_builtins);
        builder.addCapability(spv::Capability::ShaderSMBuiltinsNV);
        return spv::BuiltIn::WarpsPerSMNV;
    case glslang::EbvSMCount:
        builder.addExtension(spv::E_SPV_NV_shader_sm_builtins);
        builder.addCapability(spv::Capability::ShaderSMBuiltinsNV);
        return spv::BuiltIn::SMCountNV;
    case glslang::EbvWarpID:
        builder.addExtension(spv::E_SPV_NV_shader_sm_builtins);
        builder.addCapability(spv::Capability::ShaderSMBuiltinsNV);
        return spv::BuiltIn::WarpIDNV;
    case glslang::EbvSMID:
        builder.addExtension(spv::E_SPV_NV_shader_sm_builtins);
        builder.addCapability(spv::Capability::ShaderSMBuiltinsNV);
        return spv::BuiltIn::SMIDNV;

   // ARM builtins
    case glslang::EbvCoreCountARM:
        builder.addExtension(spv::E_SPV_ARM_core_builtins);
        builder.addCapability(spv::Capability::CoreBuiltinsARM);
        return spv::BuiltIn::CoreCountARM;
    case glslang::EbvCoreIDARM:
        builder.addExtension(spv::E_SPV_ARM_core_builtins);
        builder.addCapability(spv::Capability::CoreBuiltinsARM);
        return spv::BuiltIn::CoreIDARM;
    case glslang::EbvCoreMaxIDARM:
        builder.addExtension(spv::E_SPV_ARM_core_builtins);
        builder.addCapability(spv::Capability::CoreBuiltinsARM);
        return spv::BuiltIn::CoreMaxIDARM;
    case glslang::EbvWarpIDARM:
        builder.addExtension(spv::E_SPV_ARM_core_builtins);
        builder.addCapability(spv::Capability::CoreBuiltinsARM);
        return spv::BuiltIn::WarpIDARM;
    case glslang::EbvWarpMaxIDARM:
        builder.addExtension(spv::E_SPV_ARM_core_builtins);
        builder.addCapability(spv::Capability::CoreBuiltinsARM);
        return spv::BuiltIn::WarpMaxIDARM;

    // QCOM builtins
    case glslang::EbvTileOffsetQCOM:
        builder.addExtension(spv::E_SPV_QCOM_tile_shading);
        return spv::BuiltIn::TileOffsetQCOM;
    case glslang::EbvTileDimensionQCOM:
        builder.addExtension(spv::E_SPV_QCOM_tile_shading);
        return spv::BuiltIn::TileDimensionQCOM;
    case glslang::EbvTileApronSizeQCOM:
        builder.addExtension(spv::E_SPV_QCOM_tile_shading);
        return spv::BuiltIn::TileApronSizeQCOM;

    default:
        return spv::BuiltIn::Max;
    }
}

// Translate glslang image layout format to SPIR-V image format.
spv::ImageFormat TGlslangToSpvTraverser::TranslateImageFormat(const glslang::TType& type)
{
    assert(type.getBasicType() == glslang::EbtSampler);

    // Check for capabilities
    switch (type.getQualifier().getFormat()) {
    case glslang::ElfRg32f:
    case glslang::ElfRg16f:
    case glslang::ElfR11fG11fB10f:
    case glslang::ElfR16f:
    case glslang::ElfRgba16:
    case glslang::ElfRgb10A2:
    case glslang::ElfRg16:
    case glslang::ElfRg8:
    case glslang::ElfR16:
    case glslang::ElfR8:
    case glslang::ElfRgba16Snorm:
    case glslang::ElfRg16Snorm:
    case glslang::ElfRg8Snorm:
    case glslang::ElfR16Snorm:
    case glslang::ElfR8Snorm:

    case glslang::ElfRg32i:
    case glslang::ElfRg16i:
    case glslang::ElfRg8i:
    case glslang::ElfR16i:
    case glslang::ElfR8i:

    case glslang::ElfRgb10a2ui:
    case glslang::ElfRg32ui:
    case glslang::ElfRg16ui:
    case glslang::ElfRg8ui:
    case glslang::ElfR16ui:
    case glslang::ElfR8ui:
        builder.addCapability(spv::Capability::StorageImageExtendedFormats);
        break;

    case glslang::ElfR64ui:
    case glslang::ElfR64i:
        builder.addExtension(spv::E_SPV_EXT_shader_image_int64);
        builder.addCapability(spv::Capability::Int64ImageEXT);
        break;
    default:
        break;
    }

    // do the translation
    switch (type.getQualifier().getFormat()) {
    case glslang::ElfNone:          return spv::ImageFormat::Unknown;
    case glslang::ElfRgba32f:       return spv::ImageFormat::Rgba32f;
    case glslang::ElfRgba16f:       return spv::ImageFormat::Rgba16f;
    case glslang::ElfR32f:          return spv::ImageFormat::R32f;
    case glslang::ElfRgba8:         return spv::ImageFormat::Rgba8;
    case glslang::ElfRgba8Snorm:    return spv::ImageFormat::Rgba8Snorm;
    case glslang::ElfRg32f:         return spv::ImageFormat::Rg32f;
    case glslang::ElfRg16f:         return spv::ImageFormat::Rg16f;
    case glslang::ElfR11fG11fB10f:  return spv::ImageFormat::R11fG11fB10f;
    case glslang::ElfR16f:          return spv::ImageFormat::R16f;
    case glslang::ElfRgba16:        return spv::ImageFormat::Rgba16;
    case glslang::ElfRgb10A2:       return spv::ImageFormat::Rgb10A2;
    case glslang::ElfRg16:          return spv::ImageFormat::Rg16;
    case glslang::ElfRg8:           return spv::ImageFormat::Rg8;
    case glslang::ElfR16:           return spv::ImageFormat::R16;
    case glslang::ElfR8:            return spv::ImageFormat::R8;
    case glslang::ElfRgba16Snorm:   return spv::ImageFormat::Rgba16Snorm;
    case glslang::ElfRg16Snorm:     return spv::ImageFormat::Rg16Snorm;
    case glslang::ElfRg8Snorm:      return spv::ImageFormat::Rg8Snorm;
    case glslang::ElfR16Snorm:      return spv::ImageFormat::R16Snorm;
    case glslang::ElfR8Snorm:       return spv::ImageFormat::R8Snorm;
    case glslang::ElfRgba32i:       return spv::ImageFormat::Rgba32i;
    case glslang::ElfRgba16i:       return spv::ImageFormat::Rgba16i;
    case glslang::ElfRgba8i:        return spv::ImageFormat::Rgba8i;
    case glslang::ElfR32i:          return spv::ImageFormat::R32i;
    case glslang::ElfRg32i:         return spv::ImageFormat::Rg32i;
    case glslang::ElfRg16i:         return spv::ImageFormat::Rg16i;
    case glslang::ElfRg8i:          return spv::ImageFormat::Rg8i;
    case glslang::ElfR16i:          return spv::ImageFormat::R16i;
    case glslang::ElfR8i:           return spv::ImageFormat::R8i;
    case glslang::ElfRgba32ui:      return spv::ImageFormat::Rgba32ui;
    case glslang::ElfRgba16ui:      return spv::ImageFormat::Rgba16ui;
    case glslang::ElfRgba8ui:       return spv::ImageFormat::Rgba8ui;
    case glslang::ElfR32ui:         return spv::ImageFormat::R32ui;
    case glslang::ElfRg32ui:        return spv::ImageFormat::Rg32ui;
    case glslang::ElfRg16ui:        return spv::ImageFormat::Rg16ui;
    case glslang::ElfRgb10a2ui:     return spv::ImageFormat::Rgb10a2ui;
    case glslang::ElfRg8ui:         return spv::ImageFormat::Rg8ui;
    case glslang::ElfR16ui:         return spv::ImageFormat::R16ui;
    case glslang::ElfR8ui:          return spv::ImageFormat::R8ui;
    case glslang::ElfR64ui:         return spv::ImageFormat::R64ui;
    case glslang::ElfR64i:          return spv::ImageFormat::R64i;
    default:                        return spv::ImageFormat::Max;
    }
}

spv::SelectionControlMask TGlslangToSpvTraverser::TranslateSelectionControl(
    const glslang::TIntermSelection& selectionNode) const
{
    if (selectionNode.getFlatten())
        return spv::SelectionControlMask::Flatten;
    if (selectionNode.getDontFlatten())
        return spv::SelectionControlMask::DontFlatten;
    return spv::SelectionControlMask::MaskNone;
}

spv::SelectionControlMask TGlslangToSpvTraverser::TranslateSwitchControl(const glslang::TIntermSwitch& switchNode)
    const
{
    if (switchNode.getFlatten())
        return spv::SelectionControlMask::Flatten;
    if (switchNode.getDontFlatten())
        return spv::SelectionControlMask::DontFlatten;
    return spv::SelectionControlMask::MaskNone;
}

// return a non-0 dependency if the dependency argument must be set
spv::LoopControlMask TGlslangToSpvTraverser::TranslateLoopControl(const glslang::TIntermLoop& loopNode,
    std::vector<unsigned int>& operands) const
{
    spv::LoopControlMask control = spv::LoopControlMask::MaskNone;

    if (loopNode.getDontUnroll())
        control = control | spv::LoopControlMask::DontUnroll;
    if (loopNode.getUnroll())
        control = control | spv::LoopControlMask::Unroll;
    if (unsigned(loopNode.getLoopDependency()) == glslang::TIntermLoop::dependencyInfinite)
        control = control | spv::LoopControlMask::DependencyInfinite;
    else if (loopNode.getLoopDependency() > 0) {
        control = control | spv::LoopControlMask::DependencyLength;
        operands.push_back((unsigned int)loopNode.getLoopDependency());
    }
    if (glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_4) {
        if (loopNode.getMinIterations() > 0) {
            control = control | spv::LoopControlMask::MinIterations;
            operands.push_back(loopNode.getMinIterations());
        }
        if (loopNode.getMaxIterations() < glslang::TIntermLoop::iterationsInfinite) {
            control = control | spv::LoopControlMask::MaxIterations;
            operands.push_back(loopNode.getMaxIterations());
        }
        if (loopNode.getIterationMultiple() > 1) {
            control = control | spv::LoopControlMask::IterationMultiple;
            operands.push_back(loopNode.getIterationMultiple());
        }
        if (loopNode.getPeelCount() > 0) {
            control = control | spv::LoopControlMask::PeelCount;
            operands.push_back(loopNode.getPeelCount());
        }
        if (loopNode.getPartialCount() > 0) {
            control = control | spv::LoopControlMask::PartialCount;
            operands.push_back(loopNode.getPartialCount());
        }
    }

    return control;
}

// Translate glslang type to SPIR-V storage class.
spv::StorageClass TGlslangToSpvTraverser::TranslateStorageClass(const glslang::TType& type)
{
    if (type.getBasicType() == glslang::EbtRayQuery || type.getBasicType() == glslang::EbtHitObjectNV 
        || type.getBasicType() == glslang::EbtHitObjectEXT)
        return spv::StorageClass::Private;
    if (type.getQualifier().isSpirvByReference()) {
        if (type.getQualifier().isParamInput() || type.getQualifier().isParamOutput())
            return spv::StorageClass::Function;
    }
    if (type.getQualifier().isPipeInput())
        return spv::StorageClass::Input;
    if (type.getQualifier().isPipeOutput())
        return spv::StorageClass::Output;
    if (type.getQualifier().storage == glslang::EvqTileImageEXT || type.isAttachmentEXT()) {
        builder.addExtension(spv::E_SPV_EXT_shader_tile_image);
        builder.addCapability(spv::Capability::TileImageColorReadAccessEXT);
        return spv::StorageClass::TileImageEXT;
    }

    if (type.getQualifier().isTileAttachmentQCOM()) {
        builder.addExtension(spv::E_SPV_QCOM_tile_shading);
        builder.addCapability(spv::Capability::TileShadingQCOM);
        return spv::StorageClass::TileAttachmentQCOM;
    }

    if (glslangIntermediate->getSource() != glslang::EShSourceHlsl ||
            type.getQualifier().storage == glslang::EvqUniform) {
        if (type.isAtomic())
            return spv::StorageClass::AtomicCounter;
        if (type.containsOpaque() && !glslangIntermediate->getBindlessMode())
            return spv::StorageClass::UniformConstant;
    }

    if (type.getQualifier().isUniformOrBuffer() &&
        type.getQualifier().isShaderRecord()) {
        return spv::StorageClass::ShaderRecordBufferKHR;
    }

    if (glslangIntermediate->usingStorageBuffer() && type.getQualifier().storage == glslang::EvqBuffer) {
        builder.addIncorporatedExtension(spv::E_SPV_KHR_storage_buffer_storage_class, spv::Spv_1_3);
        return spv::StorageClass::StorageBuffer;
    }

    if (type.getQualifier().isUniformOrBuffer()) {
        if (type.getQualifier().isPushConstant())
            return spv::StorageClass::PushConstant;
        if (type.getBasicType() == glslang::EbtBlock)
            return spv::StorageClass::Uniform;
        return spv::StorageClass::UniformConstant;
    }

    if (type.getQualifier().storage == glslang::EvqShared && type.getBasicType() == glslang::EbtBlock) {
        builder.addExtension(spv::E_SPV_KHR_workgroup_memory_explicit_layout);
        builder.addCapability(spv::Capability::WorkgroupMemoryExplicitLayoutKHR);
        return spv::StorageClass::Workgroup;
    }

    switch (type.getQualifier().storage) {
    case glslang::EvqGlobal:        return spv::StorageClass::Private;
    case glslang::EvqConstReadOnly: return spv::StorageClass::Function;
    case glslang::EvqTemporary:     return spv::StorageClass::Function;
    case glslang::EvqShared:           return spv::StorageClass::Workgroup;
    case glslang::EvqPayload:        return spv::StorageClass::RayPayloadKHR;
    case glslang::EvqPayloadIn:      return spv::StorageClass::IncomingRayPayloadKHR;
    case glslang::EvqHitAttr:        return spv::StorageClass::HitAttributeKHR;
    case glslang::EvqCallableData:   return spv::StorageClass::CallableDataKHR;
    case glslang::EvqCallableDataIn: return spv::StorageClass::IncomingCallableDataKHR;
    case glslang::EvqtaskPayloadSharedEXT : return spv::StorageClass::TaskPayloadWorkgroupEXT;
    case glslang::EvqHitObjectAttrNV: return spv::StorageClass::HitObjectAttributeNV;
    case glslang::EvqHitObjectAttrEXT: return spv::StorageClass::HitObjectAttributeEXT;
    case glslang::EvqSpirvStorageClass: return static_cast<spv::StorageClass>(type.getQualifier().spirvStorageClass);
    default:
        assert(0);
        break;
    }

    return spv::StorageClass::Function;
}

// Translate glslang constants to SPIR-V literals
void TGlslangToSpvTraverser::TranslateLiterals(const glslang::TVector<const glslang::TIntermConstantUnion*>& constants,
                                               std::vector<unsigned>& literals) const
{
    for (auto constant : constants) {
        if (constant->getBasicType() == glslang::EbtFloat) {
            float floatValue = static_cast<float>(constant->getConstArray()[0].getDConst());
            unsigned literal;
            static_assert(sizeof(literal) == sizeof(floatValue), "sizeof(unsigned) != sizeof(float)");
            memcpy(&literal, &floatValue, sizeof(literal));
            literals.push_back(literal);
        } else if (constant->getBasicType() == glslang::EbtInt) {
            unsigned literal = constant->getConstArray()[0].getIConst();
            literals.push_back(literal);
        } else if (constant->getBasicType() == glslang::EbtUint) {
            unsigned literal = constant->getConstArray()[0].getUConst();
            literals.push_back(literal);
        } else if (constant->getBasicType() == glslang::EbtBool) {
            unsigned literal = constant->getConstArray()[0].getBConst();
            literals.push_back(literal);
        } else if (constant->getBasicType() == glslang::EbtString) {
            auto str = constant->getConstArray()[0].getSConst()->c_str();
            unsigned literal = 0;
            char* literalPtr = reinterpret_cast<char*>(&literal);
            unsigned charCount = 0;
            char ch = 0;
            do {
                ch = *(str++);
                *(literalPtr++) = ch;
                ++charCount;
                if (charCount == 4) {
                    literals.push_back(literal);
                    literalPtr = reinterpret_cast<char*>(&literal);
                    charCount = 0;
                }
            } while (ch != 0);

            // Partial literal is padded with 0
            if (charCount > 0) {
                for (; charCount < 4; ++charCount)
                    *(literalPtr++) = 0;
                literals.push_back(literal);
            }
        } else
            assert(0); // Unexpected type
    }
}

// Add capabilities pertaining to how an array is indexed.
void TGlslangToSpvTraverser::addIndirectionIndexCapabilities(const glslang::TType& baseType,
                                                             const glslang::TType& indexType)
{
    if (indexType.getQualifier().isNonUniform()) {
        // deal with an asserted non-uniform index
        // SPV_EXT_descriptor_indexing already added in TranslateNonUniformDecoration
        if (baseType.getBasicType() == glslang::EbtSampler) {
            if (baseType.getQualifier().hasAttachment())
                builder.addCapability(spv::Capability::InputAttachmentArrayNonUniformIndexingEXT);
            else if (baseType.isImage() && baseType.getSampler().isBuffer())
                builder.addCapability(spv::Capability::StorageTexelBufferArrayNonUniformIndexingEXT);
            else if (baseType.isTexture() && baseType.getSampler().isBuffer())
                builder.addCapability(spv::Capability::UniformTexelBufferArrayNonUniformIndexingEXT);
            else if (baseType.isImage())
                builder.addCapability(spv::Capability::StorageImageArrayNonUniformIndexingEXT);
            else if (baseType.isTexture())
                builder.addCapability(spv::Capability::SampledImageArrayNonUniformIndexingEXT);
        } else if (baseType.getBasicType() == glslang::EbtBlock) {
            if (baseType.getQualifier().storage == glslang::EvqBuffer)
                builder.addCapability(spv::Capability::StorageBufferArrayNonUniformIndexingEXT);
            else if (baseType.getQualifier().storage == glslang::EvqUniform)
                builder.addCapability(spv::Capability::UniformBufferArrayNonUniformIndexingEXT);
        }
    } else {
        // assume a dynamically uniform index
        if (baseType.getBasicType() == glslang::EbtSampler) {
            if (baseType.getQualifier().hasAttachment()) {
                builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                builder.addCapability(spv::Capability::InputAttachmentArrayDynamicIndexingEXT);
            } else if (baseType.isImage() && baseType.getSampler().isBuffer()) {
                builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                builder.addCapability(spv::Capability::StorageTexelBufferArrayDynamicIndexingEXT);
            } else if (baseType.isTexture() && baseType.getSampler().isBuffer()) {
                builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                builder.addCapability(spv::Capability::UniformTexelBufferArrayDynamicIndexingEXT);
            }
        }
    }
}

// Return whether or not the given type is something that should be tied to a
// descriptor set.
bool IsDescriptorResource(const glslang::TType& type)
{
    // uniform and buffer blocks are included, unless it is a push_constant
    if (type.getBasicType() == glslang::EbtBlock)
        return type.getQualifier().isUniformOrBuffer() &&
        ! type.getQualifier().isShaderRecord() &&
        ! type.getQualifier().isPushConstant();

    // non block...
    // basically samplerXXX/subpass/sampler/texture are all included
    // if they are the global-scope-class, not the function parameter
    // (or local, if they ever exist) class.
    if (type.getBasicType() == glslang::EbtSampler ||
        type.getBasicType() == glslang::EbtAccStruct)
        return type.getQualifier().isUniformOrBuffer();

    // Tensors are tied to a descriptor.
    if (type.isTensorARM())
        return true;

    // None of the above.
    return false;
}

void InheritQualifiers(glslang::TQualifier& child, const glslang::TQualifier& parent)
{
    if (child.layoutMatrix == glslang::ElmNone)
        child.layoutMatrix = parent.layoutMatrix;

    if (parent.invariant)
        child.invariant = true;
    if (parent.flat)
        child.flat = true;
    if (parent.centroid)
        child.centroid = true;
    if (parent.nopersp)
        child.nopersp = true;
    if (parent.explicitInterp)
        child.explicitInterp = true;
    if (parent.perPrimitiveNV)
        child.perPrimitiveNV = true;
    if (parent.perViewNV)
        child.perViewNV = true;
    if (parent.perTaskNV)
        child.perTaskNV = true;
    if (parent.storage == glslang::EvqtaskPayloadSharedEXT)
        child.storage = glslang::EvqtaskPayloadSharedEXT;
    if (parent.patch)
        child.patch = true;
    if (parent.sample)
        child.sample = true;
    if (parent.coherent)
        child.coherent = true;
    if (parent.devicecoherent)
        child.devicecoherent = true;
    if (parent.queuefamilycoherent)
        child.queuefamilycoherent = true;
    if (parent.workgroupcoherent)
        child.workgroupcoherent = true;
    if (parent.subgroupcoherent)
        child.subgroupcoherent = true;
    if (parent.shadercallcoherent)
        child.shadercallcoherent = true;
    if (parent.nonprivate)
        child.nonprivate = true;
    if (parent.volatil)
        child.volatil = true;
    if (parent.nontemporal)
        child.nontemporal = true;
    if (parent.restrict)
        child.restrict = true;
    if (parent.readonly)
        child.readonly = true;
    if (parent.writeonly)
        child.writeonly = true;
    if (parent.nonUniform)
        child.nonUniform = true;
}

bool HasNonLayoutQualifiers(const glslang::TType& type, const glslang::TQualifier& qualifier)
{
    // This should list qualifiers that simultaneous satisfy:
    // - struct members might inherit from a struct declaration
    //     (note that non-block structs don't explicitly inherit,
    //      only implicitly, meaning no decoration involved)
    // - affect decorations on the struct members
    //     (note smooth does not, and expecting something like volatile
    //      to effect the whole object)
    // - are not part of the offset/st430/etc or row/column-major layout
    return qualifier.invariant || (qualifier.hasLocation() && type.getBasicType() == glslang::EbtBlock);
}

//
// Implement the TGlslangToSpvTraverser class.
//

TGlslangToSpvTraverser::TGlslangToSpvTraverser(unsigned int spvVersion,
    const glslang::TIntermediate* glslangIntermediate,
    spv::SpvBuildLogger* buildLogger, glslang::SpvOptions& options) :
        TIntermTraverser(true, false, true),
        options(options),
        shaderEntry(nullptr), currentFunction(nullptr),
        sequenceDepth(0), logger(buildLogger),
        builder(spvVersion, (glslang::GetKhronosToolId() << 16) | glslang::GetSpirvGeneratorVersion(), logger),
        inEntryPoint(false), entryPointTerminated(false), linkageOnly(false),
        glslangIntermediate(glslangIntermediate),
        nanMinMaxClamp(glslangIntermediate->getNanMinMaxClamp()),
        nonSemanticDebugPrintf(0),
        taskPayloadID(0)
{
    bool isMeshShaderExt = (glslangIntermediate->getRequestedExtensions().find(glslang::E_GL_EXT_mesh_shader) !=
                            glslangIntermediate->getRequestedExtensions().end());
    spv::ExecutionModel executionModel = TranslateExecutionModel(glslangIntermediate->getStage(), isMeshShaderExt);

    builder.clearAccessChain();
    builder.setSource(TranslateSourceLanguage(glslangIntermediate->getSource(), glslangIntermediate->getProfile()),
                      glslangIntermediate->getVersion());

    if (options.emitNonSemanticShaderDebugSource)
            this->options.emitNonSemanticShaderDebugInfo = true;
    if (options.emitNonSemanticShaderDebugInfo)
            this->options.generateDebugInfo = true;

    if (this->options.generateDebugInfo) {
        if (this->options.emitNonSemanticShaderDebugInfo) {
            builder.setEmitNonSemanticShaderDebugInfo(this->options.emitNonSemanticShaderDebugSource);
        }
        else {
            builder.setEmitSpirvDebugInfo();
        }
        builder.setDebugMainSourceFile(glslangIntermediate->getSourceFile());

        // Set the source shader's text. If for SPV version 1.0, include
        // a preamble in comments stating the OpModuleProcessed instructions.
        // Otherwise, emit those as actual instructions.
        std::string text;
        const std::vector<std::string>& processes = glslangIntermediate->getProcesses();
        for (int p = 0; p < (int)processes.size(); ++p) {
            if (glslangIntermediate->getSpv().spv < glslang::EShTargetSpv_1_1) {
                text.append("// OpModuleProcessed ");
                text.append(processes[p]);
                text.append("\n");
            } else
                builder.addModuleProcessed(processes[p]);
        }
        if (glslangIntermediate->getSpv().spv < glslang::EShTargetSpv_1_1 && (int)processes.size() > 0)
            text.append("#line 1\n");
        text.append(glslangIntermediate->getSourceText());
        builder.setSourceText(text);
        // Pass name and text for all included files
        const std::map<std::string, std::string>& include_txt = glslangIntermediate->getIncludeText();
        for (auto iItr = include_txt.begin(); iItr != include_txt.end(); ++iItr)
            builder.addInclude(iItr->first, iItr->second);
    }

    builder.setUseReplicatedComposites(glslangIntermediate->usingReplicatedComposites());

    stdBuiltins = builder.import("GLSL.std.450");

    spv::AddressingModel addressingModel = spv::AddressingModel::Logical;
    spv::MemoryModel memoryModel = spv::MemoryModel::GLSL450;

    if (glslangIntermediate->usingPhysicalStorageBuffer()) {
        addressingModel = spv::AddressingModel::PhysicalStorageBuffer64EXT;
        builder.addIncorporatedExtension(spv::E_SPV_KHR_physical_storage_buffer, spv::Spv_1_5);
        builder.addCapability(spv::Capability::PhysicalStorageBufferAddressesEXT);
    }
    if (glslangIntermediate->usingVulkanMemoryModel()) {
        memoryModel = spv::MemoryModel::VulkanKHR;
        builder.addCapability(spv::Capability::VulkanMemoryModelKHR);
        builder.addIncorporatedExtension(spv::E_SPV_KHR_vulkan_memory_model, spv::Spv_1_5);
    }
    builder.setMemoryModel(addressingModel, memoryModel);

    if (glslangIntermediate->usingVariablePointers()) {
        builder.addCapability(spv::Capability::VariablePointers);
    }

    // If not linking, there is no entry point
    if (!options.compileOnly) {
        shaderEntry = builder.makeEntryPoint(glslangIntermediate->getEntryPointName().c_str());
        entryPoint =
            builder.addEntryPoint(executionModel, shaderEntry, glslangIntermediate->getEntryPointName().c_str());
    }

    // Add the source extensions
    const auto& sourceExtensions = glslangIntermediate->getRequestedExtensions();
    for (auto it = sourceExtensions.begin(); it != sourceExtensions.end(); ++it)
        builder.addSourceExtension(it->c_str());

    // Add the top-level modes for this shader.

    if (glslangIntermediate->getXfbMode()) {
        builder.addCapability(spv::Capability::TransformFeedback);
        builder.addExecutionMode(shaderEntry, spv::ExecutionMode::Xfb);
    }

    if (glslangIntermediate->getLayoutPrimitiveCulling()) {
        builder.addCapability(spv::Capability::RayTraversalPrimitiveCullingKHR);
    }

    if (glslangIntermediate->getSubgroupUniformControlFlow()) {
        builder.addExtension(spv::E_SPV_KHR_subgroup_uniform_control_flow);
        builder.addExecutionMode(shaderEntry, spv::ExecutionMode::SubgroupUniformControlFlowKHR);
    }
    if (glslangIntermediate->getMaximallyReconverges()) {
        builder.addExtension(spv::E_SPV_KHR_maximal_reconvergence);
        builder.addExecutionMode(shaderEntry, spv::ExecutionMode::MaximallyReconvergesKHR);
    }

    if (glslangIntermediate->getQuadDerivMode())
    {
        builder.addCapability(spv::Capability::QuadControlKHR);
        builder.addExtension(spv::E_SPV_KHR_quad_control);
        builder.addExecutionMode(shaderEntry, spv::ExecutionMode::QuadDerivativesKHR);
    }

    if (glslangIntermediate->getReqFullQuadsMode())
    {
        builder.addCapability(spv::Capability::QuadControlKHR);
        builder.addExtension(spv::E_SPV_KHR_quad_control);
        builder.addExecutionMode(shaderEntry, spv::ExecutionMode::RequireFullQuadsKHR);
    }

    if (glslangIntermediate->usingShader64BitIndexing())
    {
        builder.addCapability(spv::Capability::Shader64BitIndexingEXT);
        builder.addExtension(spv::E_SPV_EXT_shader_64bit_indexing);
        builder.addExecutionMode(shaderEntry, spv::ExecutionMode::Shader64BitIndexingEXT);
    }

    spv::ExecutionMode mode;
    switch (glslangIntermediate->getStage()) {
    case EShLangVertex:
        builder.addCapability(spv::Capability::Shader);
        break;

    case EShLangFragment:
        builder.addCapability(spv::Capability::Shader);
        if (glslangIntermediate->getPixelCenterInteger())
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::PixelCenterInteger);

        if (glslangIntermediate->getOriginUpperLeft())
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::OriginUpperLeft);
        else
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::OriginLowerLeft);

        if (glslangIntermediate->getEarlyFragmentTests())
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::EarlyFragmentTests);

        if (glslangIntermediate->getEarlyAndLateFragmentTestsAMD())
        {
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::EarlyAndLateFragmentTestsAMD);
            builder.addExtension(spv::E_SPV_AMD_shader_early_and_late_fragment_tests);
        }

        if (glslangIntermediate->getPostDepthCoverage()) {
            builder.addCapability(spv::Capability::SampleMaskPostDepthCoverage);
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::PostDepthCoverage);
            builder.addExtension(spv::E_SPV_KHR_post_depth_coverage);
        }

        if (glslangIntermediate->getNonCoherentColorAttachmentReadEXT()) {
            builder.addCapability(spv::Capability::TileImageColorReadAccessEXT);
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::NonCoherentColorAttachmentReadEXT);
            builder.addExtension(spv::E_SPV_EXT_shader_tile_image);
        }

        if (glslangIntermediate->getNonCoherentDepthAttachmentReadEXT()) {
            builder.addCapability(spv::Capability::TileImageDepthReadAccessEXT);
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::NonCoherentDepthAttachmentReadEXT);
            builder.addExtension(spv::E_SPV_EXT_shader_tile_image);
        }

        if (glslangIntermediate->getNonCoherentStencilAttachmentReadEXT()) {
            builder.addCapability(spv::Capability::TileImageStencilReadAccessEXT);
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::NonCoherentStencilAttachmentReadEXT);
            builder.addExtension(spv::E_SPV_EXT_shader_tile_image);
        }

        if (glslangIntermediate->getNonCoherentTileAttachmentReadQCOM()) {
            builder.addCapability(spv::Capability::TileShadingQCOM);
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::NonCoherentTileAttachmentReadQCOM);
            builder.addExtension(spv::E_SPV_QCOM_tile_shading);
        }

        if (glslangIntermediate->isDepthReplacing())
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::DepthReplacing);

        if (glslangIntermediate->isStencilReplacing())
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::StencilRefReplacingEXT);

        switch(glslangIntermediate->getDepth()) {
        case glslang::EldGreater:   mode = spv::ExecutionMode::DepthGreater;   break;
        case glslang::EldLess:      mode = spv::ExecutionMode::DepthLess;      break;
        case glslang::EldUnchanged: mode = spv::ExecutionMode::DepthUnchanged; break;
        default:                    mode = spv::ExecutionMode::Max;            break;
        }

        if (mode != spv::ExecutionMode::Max)
            builder.addExecutionMode(shaderEntry, mode);

        switch (glslangIntermediate->getStencil()) {
        case glslang::ElsRefUnchangedFrontAMD:  mode = spv::ExecutionMode::StencilRefUnchangedFrontAMD; break;
        case glslang::ElsRefGreaterFrontAMD:    mode = spv::ExecutionMode::StencilRefGreaterFrontAMD;   break;
        case glslang::ElsRefLessFrontAMD:       mode = spv::ExecutionMode::StencilRefLessFrontAMD;      break;
        case glslang::ElsRefUnchangedBackAMD:   mode = spv::ExecutionMode::StencilRefUnchangedBackAMD;  break;
        case glslang::ElsRefGreaterBackAMD:     mode = spv::ExecutionMode::StencilRefGreaterBackAMD;    break;
        case glslang::ElsRefLessBackAMD:        mode = spv::ExecutionMode::StencilRefLessBackAMD;       break;
        default:                       mode = spv::ExecutionMode::Max;                         break;
        }

        if (mode != spv::ExecutionMode::Max)
            builder.addExecutionMode(shaderEntry, (spv::ExecutionMode)mode);
        switch (glslangIntermediate->getInterlockOrdering()) {
        case glslang::EioPixelInterlockOrdered:         mode = spv::ExecutionMode::PixelInterlockOrderedEXT;
            break;
        case glslang::EioPixelInterlockUnordered:       mode = spv::ExecutionMode::PixelInterlockUnorderedEXT;
            break;
        case glslang::EioSampleInterlockOrdered:        mode = spv::ExecutionMode::SampleInterlockOrderedEXT;
            break;
        case glslang::EioSampleInterlockUnordered:      mode = spv::ExecutionMode::SampleInterlockUnorderedEXT;
            break;
        case glslang::EioShadingRateInterlockOrdered:   mode = spv::ExecutionMode::ShadingRateInterlockOrderedEXT;
            break;
        case glslang::EioShadingRateInterlockUnordered: mode = spv::ExecutionMode::ShadingRateInterlockUnorderedEXT;
            break;
        default:                                        mode = spv::ExecutionMode::Max;
            break;
        }
        if (mode != spv::ExecutionMode::Max) {
            builder.addExecutionMode(shaderEntry, (spv::ExecutionMode)mode);
            if (mode == spv::ExecutionMode::ShadingRateInterlockOrderedEXT ||
                mode == spv::ExecutionMode::ShadingRateInterlockUnorderedEXT) {
                builder.addCapability(spv::Capability::FragmentShaderShadingRateInterlockEXT);
            } else if (mode == spv::ExecutionMode::PixelInterlockOrderedEXT ||
                       mode == spv::ExecutionMode::PixelInterlockUnorderedEXT) {
                builder.addCapability(spv::Capability::FragmentShaderPixelInterlockEXT);
            } else {
                builder.addCapability(spv::Capability::FragmentShaderSampleInterlockEXT);
            }
            builder.addExtension(spv::E_SPV_EXT_fragment_shader_interlock);
        }
    break;

    case EShLangCompute: {
        builder.addCapability(spv::Capability::Shader);
        bool needSizeId = false;
        for (int dim = 0; dim < 3; ++dim) {
            if ((glslangIntermediate->getLocalSizeSpecId(dim) != glslang::TQualifier::layoutNotSet)) {
                needSizeId = true;
                break;
            }
        }
        if (glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_6 && needSizeId) {
            std::vector<spv::Id> dimConstId;
            for (int dim = 0; dim < 3; ++dim) {
                bool specConst = (glslangIntermediate->getLocalSizeSpecId(dim) != glslang::TQualifier::layoutNotSet);
                dimConstId.push_back(builder.makeUintConstant(glslangIntermediate->getLocalSize(dim), specConst));
                if (specConst) {
                    builder.addDecoration(dimConstId.back(), spv::Decoration::SpecId,
                                          glslangIntermediate->getLocalSizeSpecId(dim));
                    needSizeId = true;
                }
            }
            builder.addExecutionModeId(shaderEntry, spv::ExecutionMode::LocalSizeId, dimConstId);
        } else {
            if (glslangIntermediate->getTileShadingRateQCOM(0) >= 1 || glslangIntermediate->getTileShadingRateQCOM(1) >= 1 || glslangIntermediate->getTileShadingRateQCOM(2) >= 1) {
                auto rate_x = glslangIntermediate->getTileShadingRateQCOM(0);
                auto rate_y = glslangIntermediate->getTileShadingRateQCOM(1);
                auto rate_z = glslangIntermediate->getTileShadingRateQCOM(2);
                rate_x = ( rate_x == 0 ? 1 : rate_x );
                rate_y = ( rate_y == 0 ? 1 : rate_y );
                rate_z = ( rate_z == 0 ? 1 : rate_z );
                builder.addExecutionMode(shaderEntry, spv::ExecutionMode::TileShadingRateQCOM, rate_x, rate_y, rate_z);
            } else {
                builder.addExecutionMode(shaderEntry, spv::ExecutionMode::LocalSize, glslangIntermediate->getLocalSize(0),
                                                                                   glslangIntermediate->getLocalSize(1),
                                                                                   glslangIntermediate->getLocalSize(2));
            }
        }
        if (glslangIntermediate->getLayoutDerivativeModeNone() == glslang::LayoutDerivativeGroupQuads) {
            builder.addCapability(spv::Capability::ComputeDerivativeGroupQuadsNV);
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::DerivativeGroupQuadsNV);
            builder.addExtension(spv::E_SPV_NV_compute_shader_derivatives);
        } else if (glslangIntermediate->getLayoutDerivativeModeNone() == glslang::LayoutDerivativeGroupLinear) {
            builder.addCapability(spv::Capability::ComputeDerivativeGroupLinearNV);
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::DerivativeGroupLinearNV);
            builder.addExtension(spv::E_SPV_NV_compute_shader_derivatives);
        }

        if (glslangIntermediate->getNonCoherentTileAttachmentReadQCOM()) {
            builder.addCapability(spv::Capability::TileShadingQCOM);
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::NonCoherentTileAttachmentReadQCOM);
            builder.addExtension(spv::E_SPV_QCOM_tile_shading);
        }

        break;
    }
    case EShLangTessEvaluation:
    case EShLangTessControl:
        builder.addCapability(spv::Capability::Tessellation);

        glslang::TLayoutGeometry primitive;

        if (glslangIntermediate->getStage() == EShLangTessControl) {
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::OutputVertices,
                glslangIntermediate->getVertices());
            primitive = glslangIntermediate->getOutputPrimitive();
        } else {
            primitive = glslangIntermediate->getInputPrimitive();
        }

        switch (primitive) {
        case glslang::ElgTriangles:           mode = spv::ExecutionMode::Triangles;     break;
        case glslang::ElgQuads:               mode = spv::ExecutionMode::Quads;         break;
        case glslang::ElgIsolines:            mode = spv::ExecutionMode::Isolines;      break;
        default:                              mode = spv::ExecutionMode::Max;           break;
        }
        if (mode != spv::ExecutionMode::Max)
            builder.addExecutionMode(shaderEntry, mode);

        switch (glslangIntermediate->getVertexSpacing()) {
        case glslang::EvsEqual:            mode = spv::ExecutionMode::SpacingEqual;          break;
        case glslang::EvsFractionalEven:   mode = spv::ExecutionMode::SpacingFractionalEven; break;
        case glslang::EvsFractionalOdd:    mode = spv::ExecutionMode::SpacingFractionalOdd;  break;
        default:                           mode = spv::ExecutionMode::Max;                   break;
        }
        if (mode != spv::ExecutionMode::Max)
            builder.addExecutionMode(shaderEntry, mode);

        switch (glslangIntermediate->getVertexOrder()) {
        case glslang::EvoCw:     mode = spv::ExecutionMode::VertexOrderCw;  break;
        case glslang::EvoCcw:    mode = spv::ExecutionMode::VertexOrderCcw; break;
        default:                 mode = spv::ExecutionMode::Max;            break;
        }
        if (mode != spv::ExecutionMode::Max)
            builder.addExecutionMode(shaderEntry, mode);

        if (glslangIntermediate->getPointMode())
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::PointMode);
        break;

    case EShLangGeometry:
        builder.addCapability(spv::Capability::Geometry);
        switch (glslangIntermediate->getInputPrimitive()) {
        case glslang::ElgPoints:             mode = spv::ExecutionMode::InputPoints;             break;
        case glslang::ElgLines:              mode = spv::ExecutionMode::InputLines;              break;
        case glslang::ElgLinesAdjacency:     mode = spv::ExecutionMode::InputLinesAdjacency;     break;
        case glslang::ElgTriangles:          mode = spv::ExecutionMode::Triangles;               break;
        case glslang::ElgTrianglesAdjacency: mode = spv::ExecutionMode::InputTrianglesAdjacency; break;
        default:                             mode = spv::ExecutionMode::Max;                     break;
        }
        if (mode != spv::ExecutionMode::Max)
            builder.addExecutionMode(shaderEntry, mode);

        builder.addExecutionMode(shaderEntry, spv::ExecutionMode::Invocations, glslangIntermediate->getInvocations());

        switch (glslangIntermediate->getOutputPrimitive()) {
        case glslang::ElgPoints:        mode = spv::ExecutionMode::OutputPoints;                 break;
        case glslang::ElgLineStrip:     mode = spv::ExecutionMode::OutputLineStrip;              break;
        case glslang::ElgTriangleStrip: mode = spv::ExecutionMode::OutputTriangleStrip;          break;
        default:                        mode = spv::ExecutionMode::Max;                          break;
        }
        if (mode != spv::ExecutionMode::Max)
            builder.addExecutionMode(shaderEntry, mode);
        builder.addExecutionMode(shaderEntry, spv::ExecutionMode::OutputVertices, glslangIntermediate->getVertices());
        break;

    case EShLangRayGen:
    case EShLangIntersect:
    case EShLangAnyHit:
    case EShLangClosestHit:
    case EShLangMiss:
    case EShLangCallable:
    {
        auto& extensions = glslangIntermediate->getRequestedExtensions();
        if (extensions.find("GL_EXT_opacity_micromap") != extensions.end()) {
            builder.addCapability(spv::Capability::RayTracingOpacityMicromapEXT);
            builder.addExtension("SPV_EXT_opacity_micromap");
        }
        if (extensions.find("GL_NV_ray_tracing") == extensions.end()) {
            builder.addCapability(spv::Capability::RayTracingKHR);
            builder.addExtension("SPV_KHR_ray_tracing");
        }
        else {
            builder.addCapability(spv::Capability::RayTracingNV);
            builder.addExtension("SPV_NV_ray_tracing");
        }
        if (glslangIntermediate->getStage() != EShLangRayGen && glslangIntermediate->getStage() != EShLangCallable) {
            if (extensions.find("GL_EXT_ray_cull_mask") != extensions.end()) {
                builder.addCapability(spv::Capability::RayCullMaskKHR);
                builder.addExtension("SPV_KHR_ray_cull_mask");
            }
            if (extensions.find("GL_EXT_ray_tracing_position_fetch") != extensions.end()) {
                builder.addCapability(spv::Capability::RayTracingPositionFetchKHR);
                builder.addExtension("SPV_KHR_ray_tracing_position_fetch");
            }
        }
        break;
    }
    case EShLangTask:
    case EShLangMesh:
        if(isMeshShaderExt) {
            builder.addCapability(spv::Capability::MeshShadingEXT);
            builder.addExtension(spv::E_SPV_EXT_mesh_shader);
        } else {
            builder.addCapability(spv::Capability::MeshShadingNV);
            builder.addExtension(spv::E_SPV_NV_mesh_shader);
        }
        if (glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_6) {
            std::vector<spv::Id> dimConstId;
            for (int dim = 0; dim < 3; ++dim) {
                bool specConst = (glslangIntermediate->getLocalSizeSpecId(dim) != glslang::TQualifier::layoutNotSet);
                dimConstId.push_back(builder.makeUintConstant(glslangIntermediate->getLocalSize(dim), specConst));
                if (specConst) {
                    builder.addDecoration(dimConstId.back(), spv::Decoration::SpecId,
                                          glslangIntermediate->getLocalSizeSpecId(dim));
                }
            }
            builder.addExecutionModeId(shaderEntry, spv::ExecutionMode::LocalSizeId, dimConstId);
        } else {
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::LocalSize, glslangIntermediate->getLocalSize(0),
                                                                               glslangIntermediate->getLocalSize(1),
                                                                               glslangIntermediate->getLocalSize(2));
        }
        if (glslangIntermediate->getStage() == EShLangMesh) {
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::OutputVertices,
                glslangIntermediate->getVertices());
            builder.addExecutionMode(shaderEntry, spv::ExecutionMode::OutputPrimitivesNV,
                glslangIntermediate->getPrimitives());

            switch (glslangIntermediate->getOutputPrimitive()) {
            case glslang::ElgPoints:        mode = spv::ExecutionMode::OutputPoints;      break;
            case glslang::ElgLines:         mode = spv::ExecutionMode::OutputLinesNV;     break;
            case glslang::ElgTriangles:     mode = spv::ExecutionMode::OutputTrianglesNV; break;
            default:                        mode = spv::ExecutionMode::Max;               break;
            }
            if (mode != spv::ExecutionMode::Max)
                builder.addExecutionMode(shaderEntry, (spv::ExecutionMode)mode);
        }
        break;

    default:
        break;
    }

    //
    // Add SPIR-V requirements (GL_EXT_spirv_intrinsics)
    //
    if (glslangIntermediate->hasSpirvRequirement()) {
        const glslang::TSpirvRequirement& spirvRequirement = glslangIntermediate->getSpirvRequirement();

        // Add SPIR-V extension requirement
        for (auto& extension : spirvRequirement.extensions)
            builder.addExtension(extension.c_str());

        // Add SPIR-V capability requirement
        for (auto capability : spirvRequirement.capabilities)
            builder.addCapability(static_cast<spv::Capability>(capability));
    }

    //
    // Add SPIR-V execution mode qualifiers (GL_EXT_spirv_intrinsics)
    //
    if (glslangIntermediate->hasSpirvExecutionMode()) {
        const glslang::TSpirvExecutionMode spirvExecutionMode = glslangIntermediate->getSpirvExecutionMode();

        // Add spirv_execution_mode
        for (auto& mode : spirvExecutionMode.modes) {
            if (!mode.second.empty()) {
                std::vector<unsigned> literals;
                TranslateLiterals(mode.second, literals);
                builder.addExecutionMode(shaderEntry, static_cast<spv::ExecutionMode>(mode.first), literals);
            } else
                builder.addExecutionMode(shaderEntry, static_cast<spv::ExecutionMode>(mode.first));
        }

        // Add spirv_execution_mode_id
        for (auto& modeId : spirvExecutionMode.modeIds) {
            std::vector<spv::Id> operandIds;
            assert(!modeId.second.empty());
            for (auto extraOperand : modeId.second) {
                if (extraOperand->getType().getQualifier().isSpecConstant())
                    operandIds.push_back(getSymbolId(extraOperand->getAsSymbolNode()));
                else
                    operandIds.push_back(createSpvConstant(*extraOperand));
            }
            builder.addExecutionModeId(shaderEntry, static_cast<spv::ExecutionMode>(modeId.first), operandIds);
        }
    }
}

// Finish creating SPV, after the traversal is complete.
void TGlslangToSpvTraverser::finishSpv(bool compileOnly)
{
    // If not linking, an entry point is not expected
    if (!compileOnly) {
        // Finish the entry point function
        if (!entryPointTerminated) {
            builder.setBuildPoint(shaderEntry->getLastBlock());
            builder.leaveFunction();
        }

        // finish off the entry-point SPV instruction by adding the Input/Output <id>
        entryPoint->reserveOperands(iOSet.size());
        for (auto id : iOSet)
            entryPoint->addIdOperand(id);
    }

    // Add capabilities, extensions, remove unneeded decorations, etc.,
    // based on the resulting SPIR-V.
    // Note: WebGPU code generation must have the opportunity to aggressively
    // prune unreachable merge blocks and continue targets.
    builder.postProcess(compileOnly);
}

// Write the SPV into 'out'.
void TGlslangToSpvTraverser::dumpSpv(std::vector<unsigned int>& out)
{
    builder.dump(out);
}

//
// Implement the traversal functions.
//
// Return true from interior nodes to have the external traversal
// continue on to children.  Return false if children were
// already processed.
//

//
// Symbols can turn into
//  - uniform/input reads
//  - output writes
//  - complex lvalue base setups:  foo.bar[3]....  , where we see foo and start up an access chain
//  - something simple that degenerates into the last bullet
//
void TGlslangToSpvTraverser::visitSymbol(glslang::TIntermSymbol* symbol)
{
    // We update the line information even though no code might be generated here
    // This is helpful to yield correct lines for control flow instructions
    if (!linkageOnly) {
        builder.setDebugSourceLocation(symbol->getLoc().line, symbol->getLoc().getFilename());
    }

    if (symbol->getBasicType() == glslang::EbtFunction) {
        return;
    }

    SpecConstantOpModeGuard spec_constant_op_mode_setter(&builder);
    if (symbol->getType().isStruct())
        glslangTypeToIdMap[symbol->getType().getStruct()] = symbol->getId();

    if (symbol->getType().getQualifier().isSpecConstant())
        spec_constant_op_mode_setter.turnOnSpecConstantOpMode();
#ifdef ENABLE_HLSL
    // Skip symbol handling if it is string-typed
    if (symbol->getBasicType() == glslang::EbtString)
        return;
#endif

    // getSymbolId() will set up all the IO decorations on the first call.
    // Formal function parameters were mapped during makeFunctions().
    spv::Id id = getSymbolId(symbol);

    if (symbol->getType().getQualifier().isTaskPayload())
        taskPayloadID = id; // cache the taskPayloadID to be used it as operand for OpEmitMeshTasksEXT

    if (builder.isPointer(id)) {
        if (!symbol->getType().getQualifier().isParamInput() &&
            !symbol->getType().getQualifier().isParamOutput()) {
            // Include all "static use" and "linkage only" interface variables on the OpEntryPoint instruction
            // Consider adding to the OpEntryPoint interface list.
            // Only looking at structures if they have at least one member.
            if (!symbol->getType().isStruct() || symbol->getType().getStruct()->size() > 0) {
                spv::StorageClass sc = builder.getStorageClass(id);
                // Before SPIR-V 1.4, we only want to include Input and Output.
                // Starting with SPIR-V 1.4, we want all globals.
                if ((glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_4 && builder.isGlobalVariable(id)) ||
                    (sc == spv::StorageClass::Input || sc == spv::StorageClass::Output)) {
                    iOSet.insert(id);
                }
            }
        }

        // If the SPIR-V type is required to be different than the AST type
        // (for ex SubgroupMasks or 3x4 ObjectToWorld/WorldToObject matrices),
        // translate now from the SPIR-V type to the AST type, for the consuming
        // operation.
        // Note this turns it from an l-value to an r-value.
        // Currently, all symbols needing this are inputs; avoid the map lookup when non-input.
        if (symbol->getType().getQualifier().storage == glslang::EvqVaryingIn)
            id = translateForcedType(id);
    }

    // Only process non-linkage-only nodes for generating actual static uses
    if (! linkageOnly || symbol->getQualifier().isSpecConstant()) {
        // Prepare to generate code for the access

        // L-value chains will be computed left to right.  We're on the symbol now,
        // which is the left-most part of the access chain, so now is "clear" time,
        // followed by setting the base.
        builder.clearAccessChain();

        // For now, we consider all user variables as being in memory, so they are pointers,
        // except for
        // A) R-Value arguments to a function, which are an intermediate object.
        //    See comments in handleUserFunctionCall().
        // B) Specialization constants (normal constants don't even come in as a variable),
        //    These are also pure R-values.
        // C) R-Values from type translation, see above call to translateForcedType()
        glslang::TQualifier qualifier = symbol->getQualifier();
        if (qualifier.isSpecConstant() || rValueParameters.find(symbol->getId()) != rValueParameters.end() ||
            !builder.isPointerType(builder.getTypeId(id)))
            builder.setAccessChainRValue(id);
        else
            builder.setAccessChainLValue(id);
    }

#ifdef ENABLE_HLSL
    // Process linkage-only nodes for any special additional interface work.
    if (linkageOnly) {
        if (glslangIntermediate->getHlslFunctionality1()) {
            // Map implicit counter buffers to their originating buffers, which should have been
            // seen by now, given earlier pruning of unused counters, and preservation of order
            // of declaration.
            if (symbol->getType().getQualifier().isUniformOrBuffer()) {
                if (!glslangIntermediate->hasCounterBufferName(symbol->getName())) {
                    // Save possible originating buffers for counter buffers, keyed by
                    // making the potential counter-buffer name.
                    std::string keyName = symbol->getName().c_str();
                    keyName = glslangIntermediate->addCounterBufferName(keyName);
                    counterOriginator[keyName] = symbol;
                } else {
                    // Handle a counter buffer, by finding the saved originating buffer.
                    std::string keyName = symbol->getName().c_str();
                    auto it = counterOriginator.find(keyName);
                    if (it != counterOriginator.end()) {
                        id = getSymbolId(it->second);
                        if (id != spv::NoResult) {
                            spv::Id counterId = getSymbolId(symbol);
                            if (counterId != spv::NoResult) {
                                builder.addExtension("SPV_GOOGLE_hlsl_functionality1");
                                builder.addDecorationId(id, spv::Decoration::HlslCounterBufferGOOGLE, counterId);
                            }
                        }
                    }
                }
            }
        }
    }
#endif
}

bool TGlslangToSpvTraverser::visitBinary(glslang::TVisit /* visit */, glslang::TIntermBinary* node)
{
    builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
    if (node->getLeft()->getAsSymbolNode() != nullptr && node->getLeft()->getType().isStruct()) {
        glslangTypeToIdMap[node->getLeft()->getType().getStruct()] = node->getLeft()->getAsSymbolNode()->getId();
    }
    if (node->getRight()->getAsSymbolNode() != nullptr && node->getRight()->getType().isStruct()) {
        glslangTypeToIdMap[node->getRight()->getType().getStruct()] = node->getRight()->getAsSymbolNode()->getId();
    }

    SpecConstantOpModeGuard spec_constant_op_mode_setter(&builder);
    if (node->getType().getQualifier().isSpecConstant())
        spec_constant_op_mode_setter.turnOnSpecConstantOpMode();

    // First, handle special cases
    switch (node->getOp()) {
    case glslang::EOpAssign:
    case glslang::EOpAddAssign:
    case glslang::EOpSubAssign:
    case glslang::EOpMulAssign:
    case glslang::EOpVectorTimesMatrixAssign:
    case glslang::EOpVectorTimesScalarAssign:
    case glslang::EOpMatrixTimesScalarAssign:
    case glslang::EOpMatrixTimesMatrixAssign:
    case glslang::EOpDivAssign:
    case glslang::EOpModAssign:
    case glslang::EOpAndAssign:
    case glslang::EOpInclusiveOrAssign:
    case glslang::EOpExclusiveOrAssign:
    case glslang::EOpLeftShiftAssign:
    case glslang::EOpRightShiftAssign:
        // A bin-op assign "a += b" means the same thing as "a = a + b"
        // where a is evaluated before b. For a simple assignment, GLSL
        // says to evaluate the left before the right.  So, always, left
        // node then right node.
        {
            // get the left l-value, save it away
            builder.clearAccessChain();
            node->getLeft()->traverse(this);
            spv::Builder::AccessChain lValue = builder.getAccessChain();

            // evaluate the right
            builder.clearAccessChain();
            node->getRight()->traverse(this);
            spv::Id rValue = accessChainLoad(node->getRight()->getType());

            // reset line number for assignment
            builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());

            if (node->getOp() != glslang::EOpAssign) {
                // the left is also an r-value
                builder.setAccessChain(lValue);
                spv::Id leftRValue = accessChainLoad(node->getLeft()->getType());

                // do the operation
                spv::Builder::AccessChain::CoherentFlags coherentFlags = TranslateCoherent(node->getLeft()->getType());
                coherentFlags |= TranslateCoherent(node->getRight()->getType());
                OpDecorations decorations = { TranslatePrecisionDecoration(node->getOperationPrecision()),
                                              TranslateNoContractionDecoration(node->getType().getQualifier()),
                                              TranslateNonUniformDecoration(coherentFlags) };
                rValue = createBinaryOperation(node->getOp(), decorations,
                                               convertGlslangToSpvType(node->getType()), leftRValue, rValue,
                                               node->getType().getBasicType());

                // these all need their counterparts in createBinaryOperation()
                assert(rValue != spv::NoResult);
            }

            // store the result
            builder.setAccessChain(lValue);
            multiTypeStore(node->getLeft()->getType(), rValue);

            // assignments are expressions having an rValue after they are evaluated...
            builder.clearAccessChain();
            builder.setAccessChainRValue(rValue);
        }
        return false;
    case glslang::EOpIndexDirect:
    case glslang::EOpIndexDirectStruct:
        {
            // Structure, array, matrix, or vector indirection with statically known index.
            // Get the left part of the access chain.
            node->getLeft()->traverse(this);

            // Add the next element in the chain

            const int glslangIndex = node->getRight()->getAsConstantUnion()->getConstArray()[0].getIConst();
            if (! node->getLeft()->getType().isArray() &&
                node->getLeft()->getType().isVector() &&
                node->getOp() == glslang::EOpIndexDirect) {
                // Swizzle is uniform so propagate uniform into access chain
                spv::Builder::AccessChain::CoherentFlags coherentFlags = TranslateCoherent(node->getLeft()->getType());
                coherentFlags.nonUniform = 0;
                // This is essentially a hard-coded vector swizzle of size 1,
                // so short circuit the access-chain stuff with a swizzle.
                std::vector<unsigned> swizzle;
                swizzle.push_back(glslangIndex);
                int dummySize;
                builder.accessChainPushSwizzle(swizzle, convertGlslangToSpvType(node->getLeft()->getType()),
                                               coherentFlags,
                                               glslangIntermediate->getBaseAlignmentScalar(
                                                   node->getLeft()->getType(), dummySize));
            } else {

                // Load through a block reference is performed with a dot operator that
                // is mapped to EOpIndexDirectStruct. When we get to the actual reference,
                // do a load and reset the access chain.
                if (node->getLeft()->isReference() &&
                    !node->getLeft()->getType().isArray() &&
                    node->getOp() == glslang::EOpIndexDirectStruct)
                {
                    spv::Id left = accessChainLoad(node->getLeft()->getType());
                    builder.clearAccessChain();
                    builder.setAccessChainLValue(left);
                }

                int spvIndex = glslangIndex;
                if (node->getLeft()->getBasicType() == glslang::EbtBlock &&
                    node->getOp() == glslang::EOpIndexDirectStruct)
                {
                    // This may be, e.g., an anonymous block-member selection, which generally need
                    // index remapping due to hidden members in anonymous blocks.
                    long long glslangId = glslangTypeToIdMap[node->getLeft()->getType().getStruct()];
                    if (memberRemapper.find(glslangId) != memberRemapper.end()) {
                        std::vector<int>& remapper = memberRemapper[glslangId];
                        assert(remapper.size() > 0);
                        spvIndex = remapper[glslangIndex];
                    }
                }

                // Struct reference propagates uniform lvalue
                spv::Builder::AccessChain::CoherentFlags coherentFlags =
                        TranslateCoherent(node->getLeft()->getType());
                coherentFlags.nonUniform = 0;

                // normal case for indexing array or structure or block
                if ((node->getRight()->getType().getBasicType() == glslang::EbtUint && glslangIntermediate->usingPromoteUint32Indices()) ||
                     node->getRight()->getType().contains64BitInt()) {
                    int64_t idx = node->getRight()->getType().contains64BitInt() ?
                                    node->getRight()->getAsConstantUnion()->getConstArray()[0].getI64Const() :
                                    node->getRight()->getAsConstantUnion()->getConstArray()[0].getUConst();
                    builder.accessChainPush(builder.makeInt64Constant(idx),
                            coherentFlags,
                            node->getLeft()->getType().getBufferReferenceAlignment());

                } else {
                    builder.accessChainPush(builder.makeIntConstant(spvIndex),
                            coherentFlags,
                            node->getLeft()->getType().getBufferReferenceAlignment());
                }
                // Add capabilities here for accessing PointSize and clip/cull distance.
                // We have deferred generation of associated capabilities until now.
                if (node->getLeft()->getType().isStruct() && ! node->getLeft()->getType().isArray())
                    declareUseOfStructMember(*(node->getLeft()->getType().getStruct()), glslangIndex);
            }
        }
        return false;
    case glslang::EOpIndexIndirect:
        {
            // Array, matrix, or vector indirection with variable index.
            // Will use native SPIR-V access-chain for and array indirection;
            // matrices are arrays of vectors, so will also work for a matrix.
            // Will use the access chain's 'component' for variable index into a vector.

            // This adapter is building access chains left to right.
            // Set up the access chain to the left.
            node->getLeft()->traverse(this);

            // save it so that computing the right side doesn't trash it
            spv::Builder::AccessChain partial = builder.getAccessChain();

            // compute the next index in the chain
            builder.clearAccessChain();
            node->getRight()->traverse(this);
            spv::Id index = accessChainLoad(node->getRight()->getType());

            // Zero-extend smaller unsigned integer types for array indexing.
            // SPIR-V OpAccessChain treats indices as signed, so we need to zero-extend
            // unsigned types to preserve their values (signed types are fine as-is).
            spv::Id indexType = builder.getTypeId(index);
            if (builder.isUintType(indexType) && builder.getScalarTypeWidth(indexType) < 32) {
                // Zero-extend unsigned types to preserve their values
                spv::Id uintType = builder.makeUintType(32);
                index = builder.createUnaryOp(spv::Op::OpUConvert, uintType, index);
            }

            addIndirectionIndexCapabilities(node->getLeft()->getType(), node->getRight()->getType());

            // restore the saved access chain
            builder.setAccessChain(partial);

            // Only if index is nonUniform should we propagate nonUniform into access chain
            spv::Builder::AccessChain::CoherentFlags index_flags = TranslateCoherent(node->getRight()->getType());
            spv::Builder::AccessChain::CoherentFlags coherent_flags = TranslateCoherent(node->getLeft()->getType());
            coherent_flags.nonUniform = index_flags.nonUniform;

            if (! node->getLeft()->getType().isArray() && node->getLeft()->getType().isVector()) {
                int dummySize;
                builder.accessChainPushComponent(
                    index, convertGlslangToSpvType(node->getLeft()->getType()), coherent_flags,
                                                glslangIntermediate->getBaseAlignmentScalar(node->getLeft()->getType(),
                                                dummySize));
            } else {
                if (glslangIntermediate->usingPromoteUint32Indices() &&
                    node->getRight()->getType().getBasicType() == glslang::EbtUint) {
                    index = createIntWidthConversion(index, 0, builder.makeIntegerType(64, true), glslang::EbtInt64, node->getRight()->getType().getBasicType());
                }

                builder.accessChainPush(index, coherent_flags,
                                        node->getLeft()->getType().getBufferReferenceAlignment());
            }
        }
        return false;
    case glslang::EOpVectorSwizzle:
        {
            node->getLeft()->traverse(this);
            std::vector<unsigned> swizzle;
            convertSwizzle(*node->getRight()->getAsAggregate(), swizzle);
            int dummySize;
            builder.accessChainPushSwizzle(swizzle, convertGlslangToSpvType(node->getLeft()->getType()),
                                           TranslateCoherent(node->getLeft()->getType()),
                                           glslangIntermediate->getBaseAlignmentScalar(node->getLeft()->getType(),
                                               dummySize));
        }
        return false;
    case glslang::EOpMatrixSwizzle:
        logger->missingFunctionality("matrix swizzle");
        return true;
    case glslang::EOpLogicalOr:
    case glslang::EOpLogicalAnd:
        {

            // These may require short circuiting, but can sometimes be done as straight
            // binary operations.  The right operand must be short circuited if it has
            // side effects, and should probably be if it is complex.
            if (isTrivial(node->getRight()->getAsTyped()))
                break; // handle below as a normal binary operation
            // otherwise, we need to do dynamic short circuiting on the right operand
            spv::Id result = createShortCircuit(node->getOp(), *node->getLeft()->getAsTyped(),
                *node->getRight()->getAsTyped());
            builder.clearAccessChain();
            builder.setAccessChainRValue(result);
        }
        return false;
    default:
        break;
    }

    // Assume generic binary op...

    // get right operand
    builder.clearAccessChain();
    node->getLeft()->traverse(this);
    spv::Id left = accessChainLoad(node->getLeft()->getType());

    // get left operand
    builder.clearAccessChain();
    node->getRight()->traverse(this);
    spv::Id right = accessChainLoad(node->getRight()->getType());

    // get result
    OpDecorations decorations = { TranslatePrecisionDecoration(node->getOperationPrecision()),
                                  TranslateNoContractionDecoration(node->getType().getQualifier()),
                                  TranslateNonUniformDecoration(node->getType().getQualifier()) };
    spv::Id result = createBinaryOperation(node->getOp(), decorations,
                                           convertGlslangToSpvType(node->getType()), left, right,
                                           node->getLeft()->getType().getBasicType());

    builder.clearAccessChain();
    if (! result) {
        logger->missingFunctionality("unknown glslang binary operation");
        return true;  // pick up a child as the place-holder result
    } else {
        builder.setAccessChainRValue(result);
        return false;
    }
}

spv::Id TGlslangToSpvTraverser::convertLoadedBoolInUniformToUint(const glslang::TType& type,
                                                                 spv::Id nominalTypeId,
                                                                 spv::Id loadedId)
{
    if (builder.isScalarType(nominalTypeId)) {
        // Conversion for bool
        spv::Id boolType = builder.makeBoolType();
        if (nominalTypeId != boolType)
            return builder.createBinOp(spv::Op::OpINotEqual, boolType, loadedId, builder.makeUintConstant(0));
    } else if (builder.isVectorType(nominalTypeId)) {
        // Conversion for bvec
        int vecSize = builder.getNumTypeComponents(nominalTypeId);
        spv::Id bvecType = builder.makeVectorType(builder.makeBoolType(), vecSize);
        if (nominalTypeId != bvecType)
            loadedId = builder.createBinOp(spv::Op::OpINotEqual, bvecType, loadedId,
                makeSmearedConstant(builder.makeUintConstant(0), vecSize));
    } else if (builder.isArrayType(nominalTypeId)) {
        // Conversion for bool array
        spv::Id boolArrayTypeId = convertGlslangToSpvType(type);
        if (nominalTypeId != boolArrayTypeId)
        {
            // Use OpCopyLogical from SPIR-V 1.4 if available.
            if (glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_4)
                return builder.createUnaryOp(spv::Op::OpCopyLogical, boolArrayTypeId, loadedId);

            glslang::TType glslangElementType(type, 0);
            spv::Id elementNominalTypeId = builder.getContainedTypeId(nominalTypeId);
            std::vector<spv::Id> constituents;
            for (int index = 0; index < type.getOuterArraySize(); ++index) {
                // get the element
                spv::Id elementValue = builder.createCompositeExtract(loadedId, elementNominalTypeId, index);

                // recursively convert it
                spv::Id elementConvertedValue = convertLoadedBoolInUniformToUint(glslangElementType, elementNominalTypeId, elementValue);
                constituents.push_back(elementConvertedValue);
            }
            return builder.createCompositeConstruct(boolArrayTypeId, constituents);
        }
    }

    return loadedId;
}

// Figure out what, if any, type changes are needed when accessing a specific built-in.
// Returns <the type SPIR-V requires for declarion, the type to translate to on use>.
// Also see comment for 'forceType', regarding tracking SPIR-V-required types.
std::pair<spv::Id, spv::Id> TGlslangToSpvTraverser::getForcedType(glslang::TBuiltInVariable glslangBuiltIn,
    const glslang::TType& glslangType)
{
    switch(glslangBuiltIn)
    {
        case glslang::EbvSubGroupEqMask:
        case glslang::EbvSubGroupGeMask:
        case glslang::EbvSubGroupGtMask:
        case glslang::EbvSubGroupLeMask:
        case glslang::EbvSubGroupLtMask: {
            // these require changing a 64-bit scaler -> a vector of 32-bit components
            if (glslangType.isVector())
                break;
            spv::Id ivec4_type = builder.makeVectorType(builder.makeUintType(32), 4);
            spv::Id uint64_type = builder.makeUintType(64);
            std::pair<spv::Id, spv::Id> ret(ivec4_type, uint64_type);
            return ret;
        }
        // There are no SPIR-V builtins defined for these and map onto original non-transposed
        // builtins. During visitBinary we insert a transpose
        case glslang::EbvWorldToObject3x4:
        case glslang::EbvObjectToWorld3x4: {
            spv::Id mat43 = builder.makeMatrixType(builder.makeFloatType(32), 4, 3);
            spv::Id mat34 = builder.makeMatrixType(builder.makeFloatType(32), 3, 4);
            std::pair<spv::Id, spv::Id> ret(mat43, mat34);
            return ret;
        }
        default:
            break;
    }

    std::pair<spv::Id, spv::Id> ret(spv::NoType, spv::NoType);
    return ret;
}

// For an object previously identified (see getForcedType() and forceType)
// as needing type translations, do the translation needed for a load, turning
// an L-value into in R-value.
spv::Id TGlslangToSpvTraverser::translateForcedType(spv::Id object)
{
    const auto forceIt = forceType.find(object);
    if (forceIt == forceType.end())
        return object;

    spv::Id desiredTypeId = forceIt->second;
    spv::Id objectTypeId = builder.getTypeId(object);
    assert(builder.isPointerType(objectTypeId));
    objectTypeId = builder.getContainedTypeId(objectTypeId);
    if (builder.isVectorType(objectTypeId) &&
        builder.getScalarTypeWidth(builder.getContainedTypeId(objectTypeId)) == 32) {
        if (builder.getScalarTypeWidth(desiredTypeId) == 64) {
            // handle 32-bit v.xy* -> 64-bit
            builder.clearAccessChain();
            builder.setAccessChainLValue(object);
            object = builder.accessChainLoad(spv::NoPrecision, spv::Decoration::Max, spv::Decoration::Max, objectTypeId);
            std::vector<spv::Id> components;
            components.push_back(builder.createCompositeExtract(object, builder.getContainedTypeId(objectTypeId), 0));
            components.push_back(builder.createCompositeExtract(object, builder.getContainedTypeId(objectTypeId), 1));

            spv::Id vecType = builder.makeVectorType(builder.getContainedTypeId(objectTypeId), 2);
            return builder.createUnaryOp(spv::Op::OpBitcast, desiredTypeId,
                                         builder.createCompositeConstruct(vecType, components));
        } else {
            logger->missingFunctionality("forcing 32-bit vector type to non 64-bit scalar");
        }
    } else if (builder.isMatrixType(objectTypeId)) {
            // There are no SPIR-V builtins defined for 3x4 variants of ObjectToWorld/WorldToObject
            // and we insert a transpose after loading the original non-transposed builtins
            builder.clearAccessChain();
            builder.setAccessChainLValue(object);
            object = builder.accessChainLoad(spv::NoPrecision, spv::Decoration::Max, spv::Decoration::Max, objectTypeId);
            return builder.createUnaryOp(spv::Op::OpTranspose, desiredTypeId, object);

    } else  {
        logger->missingFunctionality("forcing non 32-bit vector type");
    }

    return object;
}

bool TGlslangToSpvTraverser::visitUnary(glslang::TVisit /* visit */, glslang::TIntermUnary* node)
{
    builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());

    SpecConstantOpModeGuard spec_constant_op_mode_setter(&builder);
    if (node->getType().getQualifier().isSpecConstant())
        spec_constant_op_mode_setter.turnOnSpecConstantOpMode();

    spv::Id result = spv::NoResult;

    // try texturing first
    result = createImageTextureFunctionCall(node);
    if (result != spv::NoResult) {
        builder.clearAccessChain();
        builder.setAccessChainRValue(result);

        return false; // done with this node
    }

    // Non-texturing.

    if (node->getOp() == glslang::EOpArrayLength) {
        // Quite special; won't want to evaluate the operand.

        // Currently, the front-end does not allow .length() on an array until it is sized,
        // except for the last block membeor of an SSBO.
        // TODO: If this changes, link-time sized arrays might show up here, and need their
        // size extracted.

        // Normal .length() would have been constant folded by the front-end.
        // So, this has to be block.lastMember.length().
        // SPV wants "block" and member number as the operands, go get them.

        uint32_t bits = node->getType().contains64BitInt() ? 64 : 32;

        spv::Id length;
        if (node->getOperand()->getType().isCoopMat()) {
            spv::Id typeId = convertGlslangToSpvType(node->getOperand()->getType());
            assert(builder.isCooperativeMatrixType(typeId));

            if (node->getOperand()->getType().isCoopMatKHR()) {
                length = builder.createCooperativeMatrixLengthKHR(typeId);
            } else {
                spec_constant_op_mode_setter.turnOnSpecConstantOpMode();
                length = builder.createCooperativeMatrixLengthNV(typeId);
            }
        } else if (node->getOperand()->getType().isCoopVecNV()) {
            spv::Id typeId = convertGlslangToSpvType(node->getOperand()->getType());
            length = builder.getCooperativeVectorNumComponents(typeId);
        } else {
            glslang::TIntermTyped* block = node->getOperand()->getAsBinaryNode()->getLeft();
            block->traverse(this);
            unsigned int member = node->getOperand()->getAsBinaryNode()->getRight()->getAsConstantUnion()
                ->getConstArray()[0].getUConst();
            length = builder.createArrayLength(builder.accessChainGetLValue(), member, bits);
        }

        // GLSL semantics say the result of .length() is an int, while SPIR-V says
        // signedness must be 0. So, convert from SPIR-V unsigned back to GLSL's
        // AST expectation of a signed result.
        if (glslangIntermediate->getSource() == glslang::EShSourceGlsl) {
            if (builder.isInSpecConstCodeGenMode()) {
                length = builder.createBinOp(spv::Op::OpIAdd, builder.makeIntType(bits), length, builder.makeIntConstant(0));
            } else {
                length = builder.createUnaryOp(spv::Op::OpBitcast, builder.makeIntType(bits), length);
            }
        }

        builder.clearAccessChain();
        builder.setAccessChainRValue(length);

        return false;
    }

    // Force variable declaration - Debug Mode Only
    if (node->getOp() == glslang::EOpDeclare) {
        builder.clearAccessChain();
        node->getOperand()->traverse(this);
        builder.clearAccessChain();
        return false;
    }

    // Start by evaluating the operand

    // Does it need a swizzle inversion?  If so, evaluation is inverted;
    // operate first on the swizzle base, then apply the swizzle.
    spv::Id invertedType = spv::NoType;
    auto resultType = [&invertedType, &node, this](){ return invertedType != spv::NoType ?
        invertedType : convertGlslangToSpvType(node->getType()); };
    if (node->getOp() == glslang::EOpInterpolateAtCentroid)
        invertedType = getInvertedSwizzleType(*node->getOperand());

    builder.clearAccessChain();
    TIntermNode *operandNode;
    if (invertedType != spv::NoType)
        operandNode = node->getOperand()->getAsBinaryNode()->getLeft();
    else
        operandNode = node->getOperand();

    operandNode->traverse(this);

    spv::Id operand = spv::NoResult;

    spv::Builder::AccessChain::CoherentFlags lvalueCoherentFlags;

    const auto hitObjectOpsWithLvalue = [](glslang::TOperator op) {
        switch(op) {
            case glslang::EOpReorderThreadNV:
            case glslang::EOpHitObjectGetCurrentTimeNV:
            case glslang::EOpHitObjectGetHitKindNV:
            case glslang::EOpHitObjectGetPrimitiveIndexNV:
            case glslang::EOpHitObjectGetGeometryIndexNV:
            case glslang::EOpHitObjectGetInstanceIdNV:
            case glslang::EOpHitObjectGetInstanceCustomIndexNV:
            case glslang::EOpHitObjectGetObjectRayDirectionNV:
            case glslang::EOpHitObjectGetObjectRayOriginNV:
            case glslang::EOpHitObjectGetWorldRayDirectionNV:
            case glslang::EOpHitObjectGetWorldRayOriginNV:
            case glslang::EOpHitObjectGetWorldToObjectNV:
            case glslang::EOpHitObjectGetObjectToWorldNV:
            case glslang::EOpHitObjectGetRayTMaxNV:
            case glslang::EOpHitObjectGetRayTMinNV:
            case glslang::EOpHitObjectIsEmptyNV:
            case glslang::EOpHitObjectIsHitNV:
            case glslang::EOpHitObjectIsMissNV:
            case glslang::EOpHitObjectRecordEmptyNV:
            case glslang::EOpHitObjectGetShaderBindingTableRecordIndexNV:
            case glslang::EOpHitObjectGetShaderRecordBufferHandleNV:
            case glslang::EOpHitObjectGetClusterIdNV:
            case glslang::EOpHitObjectGetSpherePositionNV:
            case glslang::EOpHitObjectGetSphereRadiusNV:
            case glslang::EOpHitObjectIsSphereHitNV:
            case glslang::EOpHitObjectIsLSSHitNV:
            case glslang::EOpReorderThreadEXT:
            case glslang::EOpHitObjectGetCurrentTimeEXT:
            case glslang::EOpHitObjectGetHitKindEXT:
            case glslang::EOpHitObjectGetPrimitiveIndexEXT:
            case glslang::EOpHitObjectGetGeometryIndexEXT:
            case glslang::EOpHitObjectGetInstanceIdEXT:
            case glslang::EOpHitObjectGetInstanceCustomIndexEXT:
            case glslang::EOpHitObjectGetObjectRayDirectionEXT:
            case glslang::EOpHitObjectGetObjectRayOriginEXT:
            case glslang::EOpHitObjectGetWorldRayDirectionEXT:
            case glslang::EOpHitObjectGetWorldRayOriginEXT:
            case glslang::EOpHitObjectGetWorldToObjectEXT:
            case glslang::EOpHitObjectGetObjectToWorldEXT:
            case glslang::EOpHitObjectGetRayTMaxEXT:
            case glslang::EOpHitObjectGetRayTMinEXT:
            case glslang::EOpHitObjectGetRayFlagsEXT:
            case glslang::EOpHitObjectIsEmptyEXT:
            case glslang::EOpHitObjectIsHitEXT:
            case glslang::EOpHitObjectIsMissEXT:
            case glslang::EOpHitObjectRecordEmptyEXT:
            case glslang::EOpHitObjectGetShaderBindingTableRecordIndexEXT:
            case glslang::EOpHitObjectGetShaderRecordBufferHandleEXT:
                return true;
            default:
                return false;
        }
    };

    if (node->getOp() == glslang::EOpAtomicCounterIncrement ||
        node->getOp() == glslang::EOpAtomicCounterDecrement ||
        node->getOp() == glslang::EOpAtomicCounter          ||
        (node->getOp() == glslang::EOpInterpolateAtCentroid &&
          glslangIntermediate->getSource() != glslang::EShSourceHlsl)  ||
        node->getOp() == glslang::EOpRayQueryProceed        ||
        node->getOp() == glslang::EOpRayQueryGetRayTMin     ||
        node->getOp() == glslang::EOpRayQueryGetRayFlags    ||
        node->getOp() == glslang::EOpRayQueryGetWorldRayOrigin ||
        node->getOp() == glslang::EOpRayQueryGetWorldRayDirection ||
        node->getOp() == glslang::EOpRayQueryGetIntersectionCandidateAABBOpaque ||
        node->getOp() == glslang::EOpRayQueryTerminate ||
        node->getOp() == glslang::EOpRayQueryConfirmIntersection ||
        (node->getOp() == glslang::EOpSpirvInst && operandNode->getAsTyped()->getQualifier().isSpirvByReference()) ||
        hitObjectOpsWithLvalue(node->getOp())) {
        operand = builder.accessChainGetLValue(); // Special case l-value operands
        lvalueCoherentFlags = builder.getAccessChain().coherentFlags;
        lvalueCoherentFlags |= TranslateCoherent(operandNode->getAsTyped()->getType());
    } else if (operandNode->getAsTyped()->getQualifier().isSpirvLiteral()) {
        // Will be translated to a literal value, make a placeholder here
        operand = spv::NoResult;
    } else {
        operand = accessChainLoad(node->getOperand()->getType());
    }

    OpDecorations decorations = { TranslatePrecisionDecoration(node->getOperationPrecision()),
                                  TranslateNoContractionDecoration(node->getType().getQualifier()),
                                  TranslateNonUniformDecoration(node->getType().getQualifier()) };

    // it could be a conversion
    if (! result) {
        result = createConversion(node->getOp(), decorations, resultType(), operand,
            node->getType().getBasicType(), node->getOperand()->getBasicType());
        if (result) {
            if (node->getType().isCoopMatKHR() && node->getOperand()->getAsTyped()->getType().isCoopMatKHR() &&
                !node->getAsTyped()->getType().sameCoopMatUse(node->getOperand()->getAsTyped()->getType())) {
                // Conversions that change use need CapabilityCooperativeMatrixConversionsNV
                builder.addCapability(spv::Capability::CooperativeMatrixConversionsNV);
                builder.addExtension(spv::E_SPV_NV_cooperative_matrix2);
            }
        }
    }

    // if not, then possibly an operation
    if (! result)
        result = createUnaryOperation(node->getOp(), decorations, resultType(), operand,
            node->getOperand()->getBasicType(), lvalueCoherentFlags, node->getType());

    // it could be attached to a SPIR-V intruction
    if (!result) {
        if (node->getOp() == glslang::EOpSpirvInst) {
            const auto& spirvInst = node->getSpirvInstruction();
            if (spirvInst.set == "") {
                spv::IdImmediate idImmOp = {true, operand};
                if (operandNode->getAsTyped()->getQualifier().isSpirvLiteral()) {
                    // Translate the constant to a literal value
                    std::vector<unsigned> literals;
                    glslang::TVector<const glslang::TIntermConstantUnion*> constants;
                    constants.push_back(operandNode->getAsConstantUnion());
                    TranslateLiterals(constants, literals);
                    idImmOp = {false, literals[0]};
                }

                if (node->getBasicType() == glslang::EbtVoid)
                    builder.createNoResultOp(static_cast<spv::Op>(spirvInst.id), {idImmOp});
                else
                    result = builder.createOp(static_cast<spv::Op>(spirvInst.id), resultType(), {idImmOp});
            } else {
                result = builder.createBuiltinCall(
                    resultType(), spirvInst.set == "GLSL.std.450" ? stdBuiltins : getExtBuiltins(spirvInst.set.c_str()),
                    spirvInst.id, {operand});
            }

            if (node->getBasicType() == glslang::EbtVoid)
                return false; // done with this node
        }
    }

    if (result) {
        if (invertedType) {
            result = createInvertedSwizzle(decorations.precision, *node->getOperand(), result);
            decorations.addNonUniform(builder, result);
        }

        builder.clearAccessChain();
        builder.setAccessChainRValue(result);

        return false; // done with this node
    }

    // it must be a special case, check...
    switch (node->getOp()) {
    case glslang::EOpPostIncrement:
    case glslang::EOpPostDecrement:
    case glslang::EOpPreIncrement:
    case glslang::EOpPreDecrement:
        {
            // we need the integer value "1" or the floating point "1.0" to add/subtract
            spv::Id one = 0;
            if (node->getBasicType() == glslang::EbtFloat)
                one = builder.makeFloatConstant(1.0F);
            else if (node->getBasicType() == glslang::EbtDouble)
                one = builder.makeDoubleConstant(1.0);
            else if (node->getBasicType() == glslang::EbtFloat16)
                one = builder.makeFloat16Constant(1.0F);
            else if (node->getBasicType() == glslang::EbtBFloat16)
                one = builder.makeBFloat16Constant(1.0F);
            else if (node->getBasicType() == glslang::EbtFloatE5M2)
                one = builder.makeFloatE5M2Constant(1.0F);
            else if (node->getBasicType() == glslang::EbtFloatE4M3)
                one = builder.makeFloatE4M3Constant(1.0F);
            else if (node->getBasicType() == glslang::EbtInt8  || node->getBasicType() == glslang::EbtUint8)
                one = builder.makeInt8Constant(1);
            else if (node->getBasicType() == glslang::EbtInt16 || node->getBasicType() == glslang::EbtUint16)
                one = builder.makeInt16Constant(1);
            else if (node->getBasicType() == glslang::EbtInt64 || node->getBasicType() == glslang::EbtUint64)
                one = builder.makeInt64Constant(1);
            else
                one = builder.makeIntConstant(1);
            glslang::TOperator op;
            if (node->getOp() == glslang::EOpPreIncrement ||
                node->getOp() == glslang::EOpPostIncrement)
                op = glslang::EOpAdd;
            else
                op = glslang::EOpSub;

            spv::Id result = createBinaryOperation(op, decorations,
                                                   convertGlslangToSpvType(node->getType()), operand, one,
                                                   node->getType().getBasicType());
            assert(result != spv::NoResult);

            // The result of operation is always stored, but conditionally the
            // consumed result.  The consumed result is always an r-value.
            builder.accessChainStore(result,
                                     TranslateNonUniformDecoration(builder.getAccessChain().coherentFlags));
            builder.clearAccessChain();
            if (node->getOp() == glslang::EOpPreIncrement ||
                node->getOp() == glslang::EOpPreDecrement)
                builder.setAccessChainRValue(result);
            else
                builder.setAccessChainRValue(operand);
        }

        return false;

    case glslang::EOpAssumeEXT:
        builder.addCapability(spv::Capability::ExpectAssumeKHR);
        builder.addExtension(spv::E_SPV_KHR_expect_assume);
        builder.createNoResultOp(spv::Op::OpAssumeTrueKHR, operand);
        return false;
    case glslang::EOpEmitStreamVertex:
        builder.createNoResultOp(spv::Op::OpEmitStreamVertex, operand);
        return false;
    case glslang::EOpEndStreamPrimitive:
        builder.createNoResultOp(spv::Op::OpEndStreamPrimitive, operand);
        return false;
    case glslang::EOpRayQueryTerminate:
        builder.createNoResultOp(spv::Op::OpRayQueryTerminateKHR, operand);
        return false;
    case glslang::EOpRayQueryConfirmIntersection:
        builder.createNoResultOp(spv::Op::OpRayQueryConfirmIntersectionKHR, operand);
        return false;
    case glslang::EOpReorderThreadNV:
        builder.createNoResultOp(spv::Op::OpReorderThreadWithHitObjectNV, operand);
        return false;
    case glslang::EOpReorderThreadEXT:
        builder.createNoResultOp(spv::Op::OpReorderThreadWithHitObjectEXT, operand);
        return false;
    case glslang::EOpHitObjectRecordEmptyNV:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordEmptyNV, operand);
        return false;
    case glslang::EOpHitObjectRecordEmptyEXT:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordEmptyEXT, operand);
        return false;

    case glslang::EOpCreateTensorLayoutNV:
        result = builder.createOp(spv::Op::OpCreateTensorLayoutNV, resultType(), std::vector<spv::Id>{});
        builder.clearAccessChain();
        builder.setAccessChainRValue(result);
        return false;

    case glslang::EOpCreateTensorViewNV:
        result = builder.createOp(spv::Op::OpCreateTensorViewNV, resultType(), std::vector<spv::Id>{});
        builder.clearAccessChain();
        builder.setAccessChainRValue(result);
        return false;

    default:
        logger->missingFunctionality("unknown glslang unary");
        return true;  // pick up operand as placeholder result
    }
}

// Construct a composite object, recursively copying members if their types don't match
spv::Id TGlslangToSpvTraverser::createCompositeConstruct(spv::Id resultTypeId, std::vector<spv::Id> constituents)
{
    for (int c = 0; c < (int)constituents.size(); ++c) {
        spv::Id& constituent = constituents[c];
        spv::Id lType = builder.getContainedTypeId(resultTypeId, c);
        spv::Id rType = builder.getTypeId(constituent);
        if (lType != rType) {
            if (glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_4) {
                constituent = builder.createUnaryOp(spv::Op::OpCopyLogical, lType, constituent);
            } else if (builder.isStructType(rType)) {
                std::vector<spv::Id> rTypeConstituents;
                int numrTypeConstituents = builder.getNumTypeConstituents(rType);
                for (int i = 0; i < numrTypeConstituents; ++i) {
                    rTypeConstituents.push_back(builder.createCompositeExtract(constituent,
                        builder.getContainedTypeId(rType, i), i));
                }
                constituents[c] = createCompositeConstruct(lType, rTypeConstituents);
            } else {
                assert(builder.isArrayType(rType));
                std::vector<spv::Id> rTypeConstituents;
                int numrTypeConstituents = builder.getNumTypeConstituents(rType);

                spv::Id elementRType = builder.getContainedTypeId(rType);
                for (int i = 0; i < numrTypeConstituents; ++i) {
                    rTypeConstituents.push_back(builder.createCompositeExtract(constituent, elementRType, i));
                }
                constituents[c] = createCompositeConstruct(lType, rTypeConstituents);
            }
        }
    }
    return builder.createCompositeConstruct(resultTypeId, constituents);
}

bool TGlslangToSpvTraverser::visitAggregate(glslang::TVisit visit, glslang::TIntermAggregate* node)
{
    SpecConstantOpModeGuard spec_constant_op_mode_setter(&builder);
    if (node->getType().getQualifier().isSpecConstant())
        spec_constant_op_mode_setter.turnOnSpecConstantOpMode();

    spv::Id result = spv::NoResult;
    spv::Id invertedType = spv::NoType;                     // to use to override the natural type of the node
    std::vector<spv::Builder::AccessChain> complexLvalues;  // for holding swizzling l-values too complex for
                                                            // SPIR-V, for an out parameter
    std::vector<spv::Id> temporaryLvalues;                  // temporaries to pass, as proxies for complexLValues

    auto resultType = [&invertedType, &node, this](){
        if (invertedType != spv::NoType) {
            return invertedType;
        } else {
            auto ret = convertGlslangToSpvType(node->getType());
            // convertGlslangToSpvType may clobber the debug location, reset it
            builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
            return ret;
        }
    };

    // try texturing
    result = createImageTextureFunctionCall(node);
    if (result != spv::NoResult) {
        builder.clearAccessChain();
        builder.setAccessChainRValue(result);

        return false;
    } else if (node->getOp() == glslang::EOpImageStore ||
        node->getOp() == glslang::EOpImageStoreLod ||
        node->getOp() == glslang::EOpImageAtomicStore) {
        // "imageStore" is a special case, which has no result
        return false;
    }

    glslang::TOperator binOp = glslang::EOpNull;
    bool reduceComparison = true;
    bool isMatrix = false;
    bool noReturnValue = false;
    bool atomic = false;

    spv::Builder::AccessChain::CoherentFlags lvalueCoherentFlags;

    assert(node->getOp());

    spv::Decoration precision = TranslatePrecisionDecoration(node->getOperationPrecision());

    switch (node->getOp()) {
    case glslang::EOpScope:
    case glslang::EOpSequence:
    {
        if (visit == glslang::EvPreVisit) {
            ++sequenceDepth;
            if (sequenceDepth == 1) {
                // If this is the parent node of all the functions, we want to see them
                // early, so all call points have actual SPIR-V functions to reference.
                // In all cases, still let the traverser visit the children for us.
                makeFunctions(node->getAsAggregate()->getSequence());

                // Global initializers is specific to the shader entry point, which does not exist in compile-only mode
                if (!options.compileOnly) {
                    // Also, we want all globals initializers to go into the beginning of the entry point, before
                    // anything else gets there, so visit out of order, doing them all now.
                    makeGlobalInitializers(node->getAsAggregate()->getSequence());
                }

                //Pre process linker objects for ray tracing stages
                if (glslangIntermediate->isRayTracingStage())
                  collectRayTracingLinkerObjects();

                // Initializers are done, don't want to visit again, but functions and link objects need to be processed,
                // so do them manually.
                visitFunctions(node->getAsAggregate()->getSequence());

                return false;
            } else {
                if (node->getOp() == glslang::EOpScope) {
                    auto loc = node->getLoc();
                    builder.enterLexicalBlock(loc.line, loc.column);
                }
            }
        } else {
            if (sequenceDepth > 1 && node->getOp() == glslang::EOpScope)
                builder.leaveLexicalBlock();
            --sequenceDepth;
        }

        return true;
    }
    case glslang::EOpLinkerObjects:
    {
        if (visit == glslang::EvPreVisit)
            linkageOnly = true;
        else
            linkageOnly = false;

        return true;
    }
    case glslang::EOpComma:
    {
        // processing from left to right naturally leaves the right-most
        // lying around in the access chain
        glslang::TIntermSequence& glslangOperands = node->getSequence();
        for (int i = 0; i < (int)glslangOperands.size(); ++i)
            glslangOperands[i]->traverse(this);

        return false;
    }
    case glslang::EOpFunction:
        if (visit == glslang::EvPreVisit) {
            if (options.generateDebugInfo) {
                builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
            }
            if (isShaderEntryPoint(node)) {
                inEntryPoint = true;
                builder.setBuildPoint(shaderEntry->getLastBlock());
                builder.enterFunction(shaderEntry);
                currentFunction = shaderEntry;
            } else {
                // SPIR-V functions should already be in the functionMap from the prepass
                // that called makeFunctions().
                currentFunction = functionMap[node->getName().c_str()];
                spv::Block* functionBlock = currentFunction->getEntryBlock();
                builder.setBuildPoint(functionBlock);
                builder.enterFunction(currentFunction);
            }
            if (options.generateDebugInfo && !options.emitNonSemanticShaderDebugInfo) {
                const auto& loc = node->getLoc();
                const char* sourceFileName = loc.getFilename();
                spv::Id sourceFileId = sourceFileName ? builder.getStringId(sourceFileName) : builder.getMainFileId();
                currentFunction->setDebugLineInfo(sourceFileId, loc.line, loc.column);
            }
        } else {
            // Here we have finished visiting the function (post-visit). Finalize it.
            if (options.generateDebugInfo) {
                if (glslangIntermediate->getSource() == glslang::EShSourceGlsl && node->getSequence().size() > 1) {
                    auto endLoc = node->getSequence()[1]->getAsAggregate()->getEndLoc();
                    builder.setDebugSourceLocation(endLoc.line, endLoc.getFilename());
                }
            }
            if (inEntryPoint)
                entryPointTerminated = true;
            builder.leaveFunction();
            inEntryPoint = false;
            currentFunction = nullptr;
        }

        return true;
    case glslang::EOpParameters:
        // Parameters will have been consumed by EOpFunction processing, but not
        // the body, so we still visited the function node's children, making this
        // child redundant.
        return false;
    case glslang::EOpFunctionCall:
    {
        builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
        if (node->isUserDefined())
            result = handleUserFunctionCall(node);
        if (result) {
            builder.clearAccessChain();
            builder.setAccessChainRValue(result);
        } else
            logger->missingFunctionality("missing user function; linker needs to catch that");

        return false;
    }
    case glslang::EOpConstructMat2x2:
    case glslang::EOpConstructMat2x3:
    case glslang::EOpConstructMat2x4:
    case glslang::EOpConstructMat3x2:
    case glslang::EOpConstructMat3x3:
    case glslang::EOpConstructMat3x4:
    case glslang::EOpConstructMat4x2:
    case glslang::EOpConstructMat4x3:
    case glslang::EOpConstructMat4x4:
    case glslang::EOpConstructDMat2x2:
    case glslang::EOpConstructDMat2x3:
    case glslang::EOpConstructDMat2x4:
    case glslang::EOpConstructDMat3x2:
    case glslang::EOpConstructDMat3x3:
    case glslang::EOpConstructDMat3x4:
    case glslang::EOpConstructDMat4x2:
    case glslang::EOpConstructDMat4x3:
    case glslang::EOpConstructDMat4x4:
    case glslang::EOpConstructIMat2x2:
    case glslang::EOpConstructIMat2x3:
    case glslang::EOpConstructIMat2x4:
    case glslang::EOpConstructIMat3x2:
    case glslang::EOpConstructIMat3x3:
    case glslang::EOpConstructIMat3x4:
    case glslang::EOpConstructIMat4x2:
    case glslang::EOpConstructIMat4x3:
    case glslang::EOpConstructIMat4x4:
    case glslang::EOpConstructUMat2x2:
    case glslang::EOpConstructUMat2x3:
    case glslang::EOpConstructUMat2x4:
    case glslang::EOpConstructUMat3x2:
    case glslang::EOpConstructUMat3x3:
    case glslang::EOpConstructUMat3x4:
    case glslang::EOpConstructUMat4x2:
    case glslang::EOpConstructUMat4x3:
    case glslang::EOpConstructUMat4x4:
    case glslang::EOpConstructBMat2x2:
    case glslang::EOpConstructBMat2x3:
    case glslang::EOpConstructBMat2x4:
    case glslang::EOpConstructBMat3x2:
    case glslang::EOpConstructBMat3x3:
    case glslang::EOpConstructBMat3x4:
    case glslang::EOpConstructBMat4x2:
    case glslang::EOpConstructBMat4x3:
    case glslang::EOpConstructBMat4x4:
    case glslang::EOpConstructF16Mat2x2:
    case glslang::EOpConstructF16Mat2x3:
    case glslang::EOpConstructF16Mat2x4:
    case glslang::EOpConstructF16Mat3x2:
    case glslang::EOpConstructF16Mat3x3:
    case glslang::EOpConstructF16Mat3x4:
    case glslang::EOpConstructF16Mat4x2:
    case glslang::EOpConstructF16Mat4x3:
    case glslang::EOpConstructF16Mat4x4:
        isMatrix = true;
        [[fallthrough]];
    case glslang::EOpConstructFloat:
    case glslang::EOpConstructVec2:
    case glslang::EOpConstructVec3:
    case glslang::EOpConstructVec4:
    case glslang::EOpConstructDouble:
    case glslang::EOpConstructDVec2:
    case glslang::EOpConstructDVec3:
    case glslang::EOpConstructDVec4:
    case glslang::EOpConstructFloat16:
    case glslang::EOpConstructF16Vec2:
    case glslang::EOpConstructF16Vec3:
    case glslang::EOpConstructF16Vec4:
    case glslang::EOpConstructBFloat16:
    case glslang::EOpConstructBF16Vec2:
    case glslang::EOpConstructBF16Vec3:
    case glslang::EOpConstructBF16Vec4:
    case glslang::EOpConstructFloatE5M2:
    case glslang::EOpConstructFloatE5M2Vec2:
    case glslang::EOpConstructFloatE5M2Vec3:
    case glslang::EOpConstructFloatE5M2Vec4:
    case glslang::EOpConstructFloatE4M3:
    case glslang::EOpConstructFloatE4M3Vec2:
    case glslang::EOpConstructFloatE4M3Vec3:
    case glslang::EOpConstructFloatE4M3Vec4:
    case glslang::EOpConstructBool:
    case glslang::EOpConstructBVec2:
    case glslang::EOpConstructBVec3:
    case glslang::EOpConstructBVec4:
    case glslang::EOpConstructInt8:
    case glslang::EOpConstructI8Vec2:
    case glslang::EOpConstructI8Vec3:
    case glslang::EOpConstructI8Vec4:
    case glslang::EOpConstructUint8:
    case glslang::EOpConstructU8Vec2:
    case glslang::EOpConstructU8Vec3:
    case glslang::EOpConstructU8Vec4:
    case glslang::EOpConstructInt16:
    case glslang::EOpConstructI16Vec2:
    case glslang::EOpConstructI16Vec3:
    case glslang::EOpConstructI16Vec4:
    case glslang::EOpConstructUint16:
    case glslang::EOpConstructU16Vec2:
    case glslang::EOpConstructU16Vec3:
    case glslang::EOpConstructU16Vec4:
    case glslang::EOpConstructInt:
    case glslang::EOpConstructIVec2:
    case glslang::EOpConstructIVec3:
    case glslang::EOpConstructIVec4:
    case glslang::EOpConstructUint:
    case glslang::EOpConstructUVec2:
    case glslang::EOpConstructUVec3:
    case glslang::EOpConstructUVec4:
    case glslang::EOpConstructInt64:
    case glslang::EOpConstructI64Vec2:
    case glslang::EOpConstructI64Vec3:
    case glslang::EOpConstructI64Vec4:
    case glslang::EOpConstructUint64:
    case glslang::EOpConstructU64Vec2:
    case glslang::EOpConstructU64Vec3:
    case glslang::EOpConstructU64Vec4:
    case glslang::EOpConstructStruct:
    case glslang::EOpConstructTextureSampler:
    case glslang::EOpConstructReference:
    case glslang::EOpConstructCooperativeMatrixNV:
    case glslang::EOpConstructCooperativeMatrixKHR:
    case glslang::EOpConstructCooperativeVectorNV:
    case glslang::EOpConstructSaturated:
    {
        builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
        std::vector<spv::Id> arguments;
        translateArguments(*node, arguments, lvalueCoherentFlags);
        spv::Id constructed;
        if (node->getOp() == glslang::EOpConstructTextureSampler) {
            const glslang::TType& texType = node->getSequence()[0]->getAsTyped()->getType();
            if (glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_6 &&
                texType.getSampler().isBuffer()) {
                // SamplerBuffer is not supported in spirv1.6 so
                // `samplerBuffer(textureBuffer, sampler)` is a no-op
                // and textureBuffer is the result going forward
                constructed = arguments[0];
            } else
                constructed = builder.createOp(spv::Op::OpSampledImage, resultType(), arguments);
        } else if (node->getOp() == glslang::EOpConstructCooperativeMatrixKHR &&
                   node->getType().isCoopMatKHR() && node->getSequence()[0]->getAsTyped()->getType().isCoopMatKHR()) {
            builder.addCapability(spv::Capability::CooperativeMatrixConversionsNV);
            builder.addExtension(spv::E_SPV_NV_cooperative_matrix2);
            constructed = builder.createCooperativeMatrixConversion(resultType(), arguments[0]);
        } else if (node->getOp() == glslang::EOpConstructCooperativeVectorNV &&
                   arguments.size() == 1 &&
                   builder.getTypeId(arguments[0]) == resultType()) {
            constructed = arguments[0];
        } else if (node->getOp() == glslang::EOpConstructStruct ||
                 node->getOp() == glslang::EOpConstructCooperativeMatrixNV ||
                 node->getOp() == glslang::EOpConstructCooperativeMatrixKHR ||
                 node->getType().isArray() ||
                 // Handle constructing coopvec from one component here, to avoid the component
                 // getting smeared
                 (node->getOp() == glslang::EOpConstructCooperativeVectorNV && arguments.size() == 1 && builder.isScalar(arguments[0]))) {
            std::vector<spv::Id> constituents;
            for (int c = 0; c < (int)arguments.size(); ++c)
                constituents.push_back(arguments[c]);
            constructed = createCompositeConstruct(resultType(), constituents);
        } else if (isMatrix)
            constructed = builder.createMatrixConstructor(precision, arguments, resultType());
        else if (node->getOp() == glslang::EOpConstructSaturated) {
            OpDecorations decorations = { TranslatePrecisionDecoration(node->getOperationPrecision()),
                                          TranslateNoContractionDecoration(node->getType().getQualifier()),
                                          TranslateNonUniformDecoration(lvalueCoherentFlags) };

            constructed = createConversion(node->getOp(), decorations, resultType(), arguments[1],
                                           node->getType().getBasicType(), node->getSequence()[1]->getAsTyped()->getBasicType());
            builder.addDecoration(constructed, spv::Decoration::SaturatedToLargestFloat8NormalConversionEXT);
            builder.createStore(constructed, arguments[0]);
        }
        else
            constructed = builder.createConstructor(precision, arguments, resultType());

        if (node->getType().getQualifier().isNonUniform()) {
            builder.addDecoration(constructed, spv::Decoration::NonUniformEXT);
        }

        builder.clearAccessChain();
        builder.setAccessChainRValue(constructed);

        return false;
    }

    // These six are component-wise compares with component-wise results.
    // Forward on to createBinaryOperation(), requesting a vector result.
    case glslang::EOpLessThan:
    case glslang::EOpGreaterThan:
    case glslang::EOpLessThanEqual:
    case glslang::EOpGreaterThanEqual:
    case glslang::EOpVectorEqual:
    case glslang::EOpVectorNotEqual:
    {
        // Map the operation to a binary
        binOp = node->getOp();
        reduceComparison = false;
        switch (node->getOp()) {
        case glslang::EOpVectorEqual:     binOp = glslang::EOpVectorEqual;      break;
        case glslang::EOpVectorNotEqual:  binOp = glslang::EOpVectorNotEqual;   break;
        default:                          binOp = node->getOp();                break;
        }

        break;
    }
    case glslang::EOpMul:
        // component-wise matrix multiply
        binOp = glslang::EOpMul;
        break;
    case glslang::EOpOuterProduct:
        // two vectors multiplied to make a matrix
        binOp = glslang::EOpOuterProduct;
        break;
    case glslang::EOpDot:
    {
        // for scalar dot product, use multiply
        glslang::TIntermSequence& glslangOperands = node->getSequence();
        if (glslangOperands[0]->getAsTyped()->getVectorSize() == 1)
            binOp = glslang::EOpMul;
        break;
    }
    case glslang::EOpMod:
        // when an aggregate, this is the floating-point mod built-in function,
        // which can be emitted by the one in createBinaryOperation()
        binOp = glslang::EOpMod;
        break;

    case glslang::EOpEmitVertex:
    case glslang::EOpEndPrimitive:
    case glslang::EOpBarrier:
    case glslang::EOpMemoryBarrier:
    case glslang::EOpMemoryBarrierAtomicCounter:
    case glslang::EOpMemoryBarrierBuffer:
    case glslang::EOpMemoryBarrierImage:
    case glslang::EOpMemoryBarrierShared:
    case glslang::EOpGroupMemoryBarrier:
    case glslang::EOpDeviceMemoryBarrier:
    case glslang::EOpAllMemoryBarrierWithGroupSync:
    case glslang::EOpDeviceMemoryBarrierWithGroupSync:
    case glslang::EOpWorkgroupMemoryBarrier:
    case glslang::EOpWorkgroupMemoryBarrierWithGroupSync:
    case glslang::EOpSubgroupBarrier:
    case glslang::EOpSubgroupMemoryBarrier:
    case glslang::EOpSubgroupMemoryBarrierBuffer:
    case glslang::EOpSubgroupMemoryBarrierImage:
    case glslang::EOpSubgroupMemoryBarrierShared:
        noReturnValue = true;
        // These all have 0 operands and will naturally finish up in the code below for 0 operands
        break;

    case glslang::EOpAtomicAdd:
    case glslang::EOpAtomicSubtract:
    case glslang::EOpAtomicMin:
    case glslang::EOpAtomicMax:
    case glslang::EOpAtomicAnd:
    case glslang::EOpAtomicOr:
    case glslang::EOpAtomicXor:
    case glslang::EOpAtomicExchange:
    case glslang::EOpAtomicCompSwap:
        atomic = true;
        break;

    case glslang::EOpAtomicStore:
        noReturnValue = true;
        [[fallthrough]];
    case glslang::EOpAtomicLoad:
        atomic = true;
        break;

    case glslang::EOpAtomicCounterAdd:
    case glslang::EOpAtomicCounterSubtract:
    case glslang::EOpAtomicCounterMin:
    case glslang::EOpAtomicCounterMax:
    case glslang::EOpAtomicCounterAnd:
    case glslang::EOpAtomicCounterOr:
    case glslang::EOpAtomicCounterXor:
    case glslang::EOpAtomicCounterExchange:
    case glslang::EOpAtomicCounterCompSwap:
        builder.addExtension("SPV_KHR_shader_atomic_counter_ops");
        builder.addCapability(spv::Capability::AtomicStorageOps);
        atomic = true;
        break;

    case glslang::EOpAbsDifference:
    case glslang::EOpAddSaturate:
    case glslang::EOpSubSaturate:
    case glslang::EOpAverage:
    case glslang::EOpAverageRounded:
    case glslang::EOpMul32x16:
        builder.addCapability(spv::Capability::IntegerFunctions2INTEL);
        builder.addExtension("SPV_INTEL_shader_integer_functions2");
        binOp = node->getOp();
        break;

    case glslang::EOpExpectEXT:
        builder.addCapability(spv::Capability::ExpectAssumeKHR);
        builder.addExtension(spv::E_SPV_KHR_expect_assume);
        binOp = node->getOp();
        break;

    case glslang::EOpIgnoreIntersectionNV:
    case glslang::EOpTerminateRayNV:
    case glslang::EOpTraceNV:
    case glslang::EOpTraceRayMotionNV:
    case glslang::EOpTraceKHR:
    case glslang::EOpExecuteCallableNV:
    case glslang::EOpExecuteCallableKHR:
    case glslang::EOpWritePackedPrimitiveIndices4x8NV:
    case glslang::EOpEmitMeshTasksEXT:
    case glslang::EOpSetMeshOutputsEXT:
        noReturnValue = true;
        break;
    case glslang::EOpRayQueryInitialize:
    case glslang::EOpRayQueryTerminate:
    case glslang::EOpRayQueryGenerateIntersection:
    case glslang::EOpRayQueryConfirmIntersection:
        builder.addExtension("SPV_KHR_ray_query");
        builder.addCapability(spv::Capability::RayQueryKHR);
        noReturnValue = true;
        break;
    case glslang::EOpRayQueryProceed:
    case glslang::EOpRayQueryGetIntersectionType:
    case glslang::EOpRayQueryGetRayTMin:
    case glslang::EOpRayQueryGetRayFlags:
    case glslang::EOpRayQueryGetIntersectionT:
    case glslang::EOpRayQueryGetIntersectionInstanceCustomIndex:
    case glslang::EOpRayQueryGetIntersectionInstanceId:
    case glslang::EOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffset:
    case glslang::EOpRayQueryGetIntersectionGeometryIndex:
    case glslang::EOpRayQueryGetIntersectionPrimitiveIndex:
    case glslang::EOpRayQueryGetIntersectionBarycentrics:
    case glslang::EOpRayQueryGetIntersectionFrontFace:
    case glslang::EOpRayQueryGetIntersectionCandidateAABBOpaque:
    case glslang::EOpRayQueryGetIntersectionObjectRayDirection:
    case glslang::EOpRayQueryGetIntersectionObjectRayOrigin:
    case glslang::EOpRayQueryGetWorldRayDirection:
    case glslang::EOpRayQueryGetWorldRayOrigin:
    case glslang::EOpRayQueryGetIntersectionObjectToWorld:
    case glslang::EOpRayQueryGetIntersectionWorldToObject:
        builder.addExtension("SPV_KHR_ray_query");
        builder.addCapability(spv::Capability::RayQueryKHR);
        break;
    case glslang::EOpCooperativeMatrixLoad:
    case glslang::EOpCooperativeMatrixStore:
    case glslang::EOpCooperativeMatrixLoadNV:
    case glslang::EOpCooperativeMatrixStoreNV:
    case glslang::EOpCooperativeMatrixLoadTensorNV:
    case glslang::EOpCooperativeMatrixStoreTensorNV:
    case glslang::EOpCooperativeMatrixReduceNV:
    case glslang::EOpCooperativeMatrixPerElementOpNV:
    case glslang::EOpCooperativeMatrixTransposeNV:
    case glslang::EOpCooperativeVectorMatMulNV:
    case glslang::EOpCooperativeVectorMatMulAddNV:
    case glslang::EOpCooperativeVectorLoadNV:
    case glslang::EOpCooperativeVectorStoreNV:
    case glslang::EOpCooperativeVectorOuterProductAccumulateNV:
    case glslang::EOpCooperativeVectorReduceSumAccumulateNV:
        noReturnValue = true;
        break;
    case glslang::EOpBeginInvocationInterlock:
    case glslang::EOpEndInvocationInterlock:
        builder.addExtension(spv::E_SPV_EXT_fragment_shader_interlock);
        noReturnValue = true;
        break;

    case glslang::EOpHitObjectTraceRayNV:
    case glslang::EOpHitObjectTraceRayMotionNV:
    case glslang::EOpHitObjectGetAttributesNV:
    case glslang::EOpHitObjectExecuteShaderNV:
    case glslang::EOpHitObjectRecordEmptyNV:
    case glslang::EOpHitObjectRecordMissNV:
    case glslang::EOpHitObjectRecordMissMotionNV:
    case glslang::EOpHitObjectRecordHitNV:
    case glslang::EOpHitObjectRecordHitMotionNV:
    case glslang::EOpHitObjectRecordHitWithIndexNV:
    case glslang::EOpHitObjectRecordHitWithIndexMotionNV:
    case glslang::EOpReorderThreadNV:
        noReturnValue = true;
        [[fallthrough]];
    case glslang::EOpHitObjectIsEmptyNV:
    case glslang::EOpHitObjectIsMissNV:
    case glslang::EOpHitObjectIsHitNV:
    case glslang::EOpHitObjectGetRayTMinNV:
    case glslang::EOpHitObjectGetRayTMaxNV:
    case glslang::EOpHitObjectGetObjectRayOriginNV:
    case glslang::EOpHitObjectGetObjectRayDirectionNV:
    case glslang::EOpHitObjectGetWorldRayOriginNV:
    case glslang::EOpHitObjectGetWorldRayDirectionNV:
    case glslang::EOpHitObjectGetObjectToWorldNV:
    case glslang::EOpHitObjectGetWorldToObjectNV:
    case glslang::EOpHitObjectGetInstanceCustomIndexNV:
    case glslang::EOpHitObjectGetInstanceIdNV:
    case glslang::EOpHitObjectGetGeometryIndexNV:
    case glslang::EOpHitObjectGetPrimitiveIndexNV:
    case glslang::EOpHitObjectGetHitKindNV:
    case glslang::EOpHitObjectGetCurrentTimeNV:
    case glslang::EOpHitObjectGetShaderBindingTableRecordIndexNV:
    case glslang::EOpHitObjectGetShaderRecordBufferHandleNV:
        builder.addExtension(spv::E_SPV_NV_shader_invocation_reorder);
        builder.addCapability(spv::Capability::ShaderInvocationReorderNV);
        break;

    case glslang::EOpHitObjectGetLSSPositionsNV:
    case glslang::EOpHitObjectGetLSSRadiiNV:
        builder.addExtension(spv::E_SPV_NV_linear_swept_spheres);
        builder.addCapability(spv::Capability::ShaderInvocationReorderNV);
        builder.addCapability(spv::Capability::RayTracingLinearSweptSpheresGeometryNV);
        noReturnValue = true;
        break;

    case glslang::EOpRayQueryGetIntersectionLSSPositionsNV:
    case glslang::EOpRayQueryGetIntersectionLSSRadiiNV:
        builder.addExtension(spv::E_SPV_NV_linear_swept_spheres);
        builder.addCapability(spv::Capability::RayQueryKHR);
        builder.addCapability(spv::Capability::RayTracingLinearSweptSpheresGeometryNV);
        noReturnValue = true;
        break;

    case glslang::EOpRayQueryGetIntersectionSpherePositionNV:
    case glslang::EOpRayQueryGetIntersectionSphereRadiusNV:
    case glslang::EOpRayQueryIsSphereHitNV:
        builder.addExtension(spv::E_SPV_NV_linear_swept_spheres);
        builder.addCapability(spv::Capability::RayQueryKHR);
        builder.addCapability(spv::Capability::RayTracingSpheresGeometryNV);
        builder.addCapability(spv::Capability::RayTracingLinearSweptSpheresGeometryNV);
        break;

    case glslang::EOpRayQueryGetIntersectionLSSHitValueNV:
    case glslang::EOpRayQueryIsLSSHitNV:
        builder.addExtension(spv::E_SPV_NV_linear_swept_spheres);
        builder.addCapability(spv::Capability::RayQueryKHR);
        builder.addCapability(spv::Capability::RayTracingLinearSweptSpheresGeometryNV);
        break;

    case glslang::EOpHitObjectTraceRayEXT:
    case glslang::EOpHitObjectTraceRayMotionEXT:
    case glslang::EOpHitObjectGetAttributesEXT:
    case glslang::EOpHitObjectExecuteShaderEXT:
    case glslang::EOpHitObjectRecordEmptyEXT:
    case glslang::EOpHitObjectRecordMissEXT:
    case glslang::EOpHitObjectRecordMissMotionEXT:
    case glslang::EOpReorderThreadEXT:
    case glslang::EOpHitObjectSetShaderBindingTableRecordIndexEXT:
    case glslang::EOpHitObjectReorderExecuteEXT:
    case glslang::EOpHitObjectTraceReorderExecuteEXT:
    case glslang::EOpHitObjectTraceMotionReorderExecuteEXT:
    case glslang::EOpHitObjectRecordFromQueryEXT:
    case glslang::EOpHitObjectGetIntersectionTriangleVertexPositionsEXT:
        noReturnValue = true;
        [[fallthrough]];
    case glslang::EOpHitObjectIsEmptyEXT:
    case glslang::EOpHitObjectIsMissEXT:
    case glslang::EOpHitObjectIsHitEXT:
    case glslang::EOpHitObjectGetRayTMinEXT:
    case glslang::EOpHitObjectGetRayTMaxEXT:
    case glslang::EOpHitObjectGetRayFlagsEXT:
    case glslang::EOpHitObjectGetObjectRayOriginEXT:
    case glslang::EOpHitObjectGetObjectRayDirectionEXT:
    case glslang::EOpHitObjectGetWorldRayOriginEXT:
    case glslang::EOpHitObjectGetWorldRayDirectionEXT:
    case glslang::EOpHitObjectGetObjectToWorldEXT:
    case glslang::EOpHitObjectGetWorldToObjectEXT:
    case glslang::EOpHitObjectGetInstanceCustomIndexEXT:
    case glslang::EOpHitObjectGetInstanceIdEXT:
    case glslang::EOpHitObjectGetGeometryIndexEXT:
    case glslang::EOpHitObjectGetPrimitiveIndexEXT:
    case glslang::EOpHitObjectGetHitKindEXT:
    case glslang::EOpHitObjectGetCurrentTimeEXT:
    case glslang::EOpHitObjectGetShaderBindingTableRecordIndexEXT:
    case glslang::EOpHitObjectGetShaderRecordBufferHandleEXT:
        builder.addExtension(spv::E_SPV_EXT_shader_invocation_reorder);
        builder.addCapability(spv::Capability::ShaderInvocationReorderEXT);
        break;

    case glslang::EOpRayQueryGetIntersectionTriangleVertexPositionsEXT:
        builder.addExtension(spv::E_SPV_KHR_ray_tracing_position_fetch);
        builder.addCapability(spv::Capability::RayQueryPositionFetchKHR);
        noReturnValue = true;
        break;
    case glslang::EOpImageSampleWeightedQCOM:
        builder.addCapability(spv::Capability::TextureSampleWeightedQCOM);
        builder.addExtension(spv::E_SPV_QCOM_image_processing);
        break;
    case glslang::EOpImageBoxFilterQCOM:
        builder.addCapability(spv::Capability::TextureBoxFilterQCOM);
        builder.addExtension(spv::E_SPV_QCOM_image_processing);
        break;
    case glslang::EOpImageBlockMatchSADQCOM:
    case glslang::EOpImageBlockMatchSSDQCOM:
        builder.addCapability(spv::Capability::TextureBlockMatchQCOM);
        builder.addExtension(spv::E_SPV_QCOM_image_processing);
        break;
    case glslang::EOpTensorWriteARM:
        noReturnValue = true;
        break;

    case glslang::EOpImageBlockMatchWindowSSDQCOM:
    case glslang::EOpImageBlockMatchWindowSADQCOM:
        builder.addCapability(spv::Capability::TextureBlockMatchQCOM);
        builder.addExtension(spv::E_SPV_QCOM_image_processing);
        builder.addCapability(spv::Capability::TextureBlockMatch2QCOM);
        builder.addExtension(spv::E_SPV_QCOM_image_processing2);
        break;

    case glslang::EOpImageBlockMatchGatherSSDQCOM:
    case glslang::EOpImageBlockMatchGatherSADQCOM:
        builder.addCapability(spv::Capability::TextureBlockMatchQCOM);
        builder.addExtension(spv::E_SPV_QCOM_image_processing);
        builder.addCapability(spv::Capability::TextureBlockMatch2QCOM);
        builder.addExtension(spv::E_SPV_QCOM_image_processing2);
        break;

    case glslang::EOpFetchMicroTriangleVertexPositionNV:
    case glslang::EOpFetchMicroTriangleVertexBarycentricNV:
        builder.addExtension(spv::E_SPV_NV_displacement_micromap);
        builder.addCapability(spv::Capability::DisplacementMicromapNV);
        break;

    case glslang::EOpRayQueryGetIntersectionClusterIdNV:
        builder.addExtension(spv::E_SPV_NV_cluster_acceleration_structure);
        builder.addCapability(spv::Capability::RayQueryKHR);
        builder.addCapability(spv::Capability::RayTracingClusterAccelerationStructureNV);
        break;

    case glslang::EOpDebugPrintf:
        noReturnValue = true;
        break;

    default:
        break;
    }

    //
    // See if it maps to a regular operation.
    //
    if (binOp != glslang::EOpNull) {
        glslang::TIntermTyped* left = node->getSequence()[0]->getAsTyped();
        glslang::TIntermTyped* right = node->getSequence()[1]->getAsTyped();
        assert(left && right);

        builder.clearAccessChain();
        left->traverse(this);
        spv::Id leftId = accessChainLoad(left->getType());

        builder.clearAccessChain();
        right->traverse(this);
        spv::Id rightId = accessChainLoad(right->getType());

        builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
        OpDecorations decorations = { precision,
                                      TranslateNoContractionDecoration(node->getType().getQualifier()),
                                      TranslateNonUniformDecoration(node->getType().getQualifier()) };
        result = createBinaryOperation(binOp, decorations,
                                       resultType(), leftId, rightId,
                                       left->getType().getBasicType(), reduceComparison);

        // code above should only make binOp that exists in createBinaryOperation
        assert(result != spv::NoResult);
        builder.clearAccessChain();
        builder.setAccessChainRValue(result);

        return false;
    }

    //
    // Create the list of operands.
    //
    glslang::TIntermSequence& glslangOperands = node->getSequence();
    std::vector<spv::Id> operands;
    std::vector<spv::IdImmediate> memoryAccessOperands;
    for (int arg = 0; arg < (int)glslangOperands.size(); ++arg) {
        // special case l-value operands; there are just a few
        bool lvalue = false;
        switch (node->getOp()) {
        case glslang::EOpModf:
            if (arg == 1)
                lvalue = true;
            break;



        case glslang::EOpHitObjectRecordFromQueryEXT:
        case glslang::EOpHitObjectGetIntersectionTriangleVertexPositionsEXT:
            if (arg == 0 || arg == 1)
                lvalue = true;
            break;

        case glslang::EOpHitObjectRecordHitNV:
        case glslang::EOpHitObjectRecordHitMotionNV:
        case glslang::EOpHitObjectRecordHitWithIndexNV:
        case glslang::EOpHitObjectRecordHitWithIndexMotionNV:
        case glslang::EOpHitObjectTraceRayNV:
        case glslang::EOpHitObjectTraceRayMotionNV:
        case glslang::EOpHitObjectExecuteShaderNV:
        case glslang::EOpHitObjectRecordMissNV:
        case glslang::EOpHitObjectRecordMissMotionNV:
        case glslang::EOpHitObjectGetAttributesNV:
        case glslang::EOpHitObjectGetClusterIdNV:
        case glslang::EOpHitObjectTraceRayEXT:
        case glslang::EOpHitObjectTraceRayMotionEXT:
        case glslang::EOpHitObjectExecuteShaderEXT:
        case glslang::EOpHitObjectRecordMissEXT:
        case glslang::EOpHitObjectRecordMissMotionEXT:
        case glslang::EOpHitObjectGetAttributesEXT:
        case glslang::EOpHitObjectSetShaderBindingTableRecordIndexEXT:
        case glslang::EOpHitObjectReorderExecuteEXT:
        case glslang::EOpHitObjectTraceReorderExecuteEXT:
        case glslang::EOpHitObjectTraceMotionReorderExecuteEXT:
            if (arg == 0)
                lvalue = true;
            break;

        case glslang::EOpHitObjectGetLSSPositionsNV:
        case glslang::EOpHitObjectGetLSSRadiiNV:
            lvalue = true;
            break;

        case glslang::EOpRayQueryInitialize:
        case glslang::EOpRayQueryTerminate:
        case glslang::EOpRayQueryConfirmIntersection:
        case glslang::EOpRayQueryProceed:
        case glslang::EOpRayQueryGenerateIntersection:
        case glslang::EOpRayQueryGetIntersectionType:
        case glslang::EOpRayQueryGetIntersectionT:
        case glslang::EOpRayQueryGetIntersectionInstanceCustomIndex:
        case glslang::EOpRayQueryGetIntersectionInstanceId:
        case glslang::EOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffset:
        case glslang::EOpRayQueryGetIntersectionGeometryIndex:
        case glslang::EOpRayQueryGetIntersectionPrimitiveIndex:
        case glslang::EOpRayQueryGetIntersectionBarycentrics:
        case glslang::EOpRayQueryGetIntersectionFrontFace:
        case glslang::EOpRayQueryGetIntersectionObjectRayDirection:
        case glslang::EOpRayQueryGetIntersectionObjectRayOrigin:
        case glslang::EOpRayQueryGetIntersectionObjectToWorld:
        case glslang::EOpRayQueryGetIntersectionWorldToObject:
        case glslang::EOpRayQueryGetIntersectionClusterIdNV:
        case glslang::EOpRayQueryGetIntersectionSpherePositionNV:
        case glslang::EOpRayQueryGetIntersectionSphereRadiusNV:
        case glslang::EOpRayQueryGetIntersectionLSSHitValueNV:
        case glslang::EOpRayQueryIsSphereHitNV:
        case glslang::EOpRayQueryIsLSSHitNV:
            if (arg == 0)
                lvalue = true;
            break;

        case glslang::EOpAtomicAdd:
        case glslang::EOpAtomicSubtract:
        case glslang::EOpAtomicMin:
        case glslang::EOpAtomicMax:
        case glslang::EOpAtomicAnd:
        case glslang::EOpAtomicOr:
        case glslang::EOpAtomicXor:
        case glslang::EOpAtomicExchange:
        case glslang::EOpAtomicCompSwap:
            if (arg == 0)
                lvalue = true;
            break;

        case glslang::EOpFrexp:
            if (arg == 1)
                lvalue = true;
            break;
        case glslang::EOpInterpolateAtSample:
        case glslang::EOpInterpolateAtOffset:
        case glslang::EOpInterpolateAtVertex:
            if (arg == 0) {
                // If GLSL, use the address of the interpolant argument.
                // If HLSL, use an internal version of OpInterolates that takes
                // the rvalue of the interpolant. A fixup pass in spirv-opt
                // legalization will remove the OpLoad and convert to an lvalue.
                // Had to do this because legalization will only propagate a
                // builtin into an rvalue.
                lvalue = glslangIntermediate->getSource() != glslang::EShSourceHlsl;

                // Does it need a swizzle inversion?  If so, evaluation is inverted;
                // operate first on the swizzle base, then apply the swizzle.
                // That is, we transform
                //
                //    interpolate(v.zy)  ->  interpolate(v).zy
                //
                if (glslangOperands[0]->getAsOperator() &&
                    glslangOperands[0]->getAsOperator()->getOp() == glslang::EOpVectorSwizzle)
                    invertedType = convertGlslangToSpvType(
                        glslangOperands[0]->getAsBinaryNode()->getLeft()->getType());
            }
            break;
        case glslang::EOpAtomicLoad:
        case glslang::EOpAtomicStore:
        case glslang::EOpAtomicCounterAdd:
        case glslang::EOpAtomicCounterSubtract:
        case glslang::EOpAtomicCounterMin:
        case glslang::EOpAtomicCounterMax:
        case glslang::EOpAtomicCounterAnd:
        case glslang::EOpAtomicCounterOr:
        case glslang::EOpAtomicCounterXor:
        case glslang::EOpAtomicCounterExchange:
        case glslang::EOpAtomicCounterCompSwap:
            if (arg == 0)
                lvalue = true;
            break;
        case glslang::EOpAddCarry:
        case glslang::EOpSubBorrow:
            if (arg == 2)
                lvalue = true;
            break;
        case glslang::EOpUMulExtended:
        case glslang::EOpIMulExtended:
            if (arg >= 2)
                lvalue = true;
            break;
        case glslang::EOpCooperativeMatrixLoad:
        case glslang::EOpCooperativeMatrixLoadNV:
        case glslang::EOpCooperativeMatrixLoadTensorNV:
        case glslang::EOpCooperativeVectorLoadNV:
            if (arg == 0 || arg == 1)
                lvalue = true;
            break;
        case glslang::EOpCooperativeMatrixStore:
        case glslang::EOpCooperativeMatrixStoreNV:
        case glslang::EOpCooperativeMatrixStoreTensorNV:
        case glslang::EOpCooperativeVectorStoreNV:
            if (arg == 1)
                lvalue = true;
            break;
        case glslang::EOpCooperativeVectorMatMulNV:
            if (arg == 0 || arg == 3)
                lvalue = true;
            break;
        case glslang::EOpCooperativeVectorMatMulAddNV:
            if (arg == 0 || arg == 3 || arg == 6)
                lvalue = true;
            break;
        case glslang::EOpCooperativeVectorOuterProductAccumulateNV:
            if (arg == 2)
                lvalue = true;
            break;
        case glslang::EOpCooperativeVectorReduceSumAccumulateNV:
            if (arg == 1)
                lvalue = true;
            break;
        case glslang::EOpCooperativeMatrixReduceNV:
        case glslang::EOpCooperativeMatrixPerElementOpNV:
        case glslang::EOpCooperativeMatrixTransposeNV:
            if (arg == 0)
                lvalue = true;
            break;
        case glslang::EOpSpirvInst:
            if (glslangOperands[arg]->getAsTyped()->getQualifier().isSpirvByReference())
                lvalue = true;
            break;
        case glslang::EOpReorderThreadNV:
        case glslang::EOpReorderThreadEXT:
            //Three variants of reorderThreadNV, two of them use hitObjectNV
            if (arg == 0 && glslangOperands.size() != 2)
                lvalue = true;
            break;
        case glslang::EOpRayQueryGetIntersectionTriangleVertexPositionsEXT:
        case glslang::EOpRayQueryGetIntersectionLSSPositionsNV:
        case glslang::EOpRayQueryGetIntersectionLSSRadiiNV:
            if (arg == 0 || arg == 2)
                lvalue = true;
            break;
        case glslang::EOpTensorReadARM:
            if (arg == 2)
                lvalue = true;
            break;
        default:
            break;
        }
        builder.clearAccessChain();
        if (invertedType != spv::NoType && arg == 0)
            glslangOperands[0]->getAsBinaryNode()->getLeft()->traverse(this);
        else
            glslangOperands[arg]->traverse(this);

        bool isCoopMat = node->getOp() == glslang::EOpCooperativeMatrixLoad ||
                         node->getOp() == glslang::EOpCooperativeMatrixStore ||
                         node->getOp() == glslang::EOpCooperativeMatrixLoadNV ||
                         node->getOp() == glslang::EOpCooperativeMatrixStoreNV ||
                         node->getOp() == glslang::EOpCooperativeMatrixLoadTensorNV ||
                         node->getOp() == glslang::EOpCooperativeMatrixStoreTensorNV;
        bool isCoopVec = node->getOp() == glslang::EOpCooperativeVectorLoadNV ||
                         node->getOp() == glslang::EOpCooperativeVectorStoreNV;
        if (isCoopMat || isCoopVec) {

            if (arg == 1) {
                spv::Builder::AccessChain::CoherentFlags coherentFlags {};
                unsigned int alignment {};
                if (isCoopMat) {
                    // fold "element" parameter into the access chain
                    spv::Builder::AccessChain save = builder.getAccessChain();
                    builder.clearAccessChain();
                    glslangOperands[2]->traverse(this);

                    spv::Id elementId = accessChainLoad(glslangOperands[2]->getAsTyped()->getType());

                    builder.setAccessChain(save);

                    // Point to the first element of the array.
                    builder.accessChainPush(elementId,
                        TranslateCoherent(glslangOperands[arg]->getAsTyped()->getType()),
                                          glslangOperands[arg]->getAsTyped()->getType().getBufferReferenceAlignment());
                    coherentFlags = builder.getAccessChain().coherentFlags;
                    alignment = builder.getAccessChain().alignment;
                } else {
                    coherentFlags = builder.getAccessChain().coherentFlags;
                    coherentFlags |= TranslateCoherent(glslangOperands[arg]->getAsTyped()->getType());
                    alignment = 16;
                }

                spv::MemoryAccessMask memoryAccess = TranslateMemoryAccess(coherentFlags);
                if (node->getOp() == glslang::EOpCooperativeMatrixLoad ||
                    node->getOp() == glslang::EOpCooperativeMatrixLoadNV ||
                    node->getOp() == glslang::EOpCooperativeMatrixLoadTensorNV ||
                    node->getOp() == glslang::EOpCooperativeVectorLoadNV)
                    memoryAccess = (memoryAccess & ~spv::MemoryAccessMask::MakePointerAvailableKHR);
                if (node->getOp() == glslang::EOpCooperativeMatrixStore ||
                    node->getOp() == glslang::EOpCooperativeMatrixStoreNV ||
                    node->getOp() == glslang::EOpCooperativeMatrixStoreTensorNV ||
                    node->getOp() == glslang::EOpCooperativeVectorStoreNV)
                    memoryAccess = (memoryAccess & ~spv::MemoryAccessMask::MakePointerVisibleKHR);
                if (builder.getStorageClass(builder.getAccessChain().base) ==
                    spv::StorageClass::PhysicalStorageBufferEXT) {
                    memoryAccess = (spv::MemoryAccessMask)(memoryAccess | spv::MemoryAccessMask::Aligned);
                }

                memoryAccessOperands.push_back(spv::IdImmediate(false, memoryAccess));

                if (anySet(memoryAccess, spv::MemoryAccessMask::Aligned)) {
                    memoryAccessOperands.push_back(spv::IdImmediate(false, alignment));
                }

                if (anySet(memoryAccess,
                    spv::MemoryAccessMask::MakePointerAvailableKHR | spv::MemoryAccessMask::MakePointerVisibleKHR)) {
                    memoryAccessOperands.push_back(spv::IdImmediate(true,
                        builder.makeUintConstant(TranslateMemoryScope(coherentFlags))));
                }
            } else if (isCoopMat && arg == 2) {
                continue;
            }
        }

        // for l-values, pass the address, for r-values, pass the value
        if (lvalue) {
            if (invertedType == spv::NoType && !builder.isSpvLvalue()) {
                // SPIR-V cannot represent an l-value containing a swizzle that doesn't
                // reduce to a simple access chain.  So, we need a temporary vector to
                // receive the result, and must later swizzle that into the original
                // l-value.
                complexLvalues.push_back(builder.getAccessChain());
                temporaryLvalues.push_back(builder.createVariable(
                    spv::NoPrecision, spv::StorageClass::Function,
                    builder.accessChainGetInferredType(), "swizzleTemp"));
                operands.push_back(temporaryLvalues.back());
            } else {
                operands.push_back(builder.accessChainGetLValue());
            }
            lvalueCoherentFlags = builder.getAccessChain().coherentFlags;
            lvalueCoherentFlags |= TranslateCoherent(glslangOperands[arg]->getAsTyped()->getType());
        } else {
            builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
            glslang::TOperator glslangOp = node->getOp();
            if (arg == 1 &&
                (glslangOp == glslang::EOpRayQueryGetIntersectionType ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionT ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionInstanceCustomIndex ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionInstanceId ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffset ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionGeometryIndex ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionPrimitiveIndex ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionBarycentrics ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionFrontFace ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionObjectRayDirection ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionObjectRayOrigin ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionObjectToWorld ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionWorldToObject ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionTriangleVertexPositionsEXT ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionClusterIdNV ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionSpherePositionNV ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionSphereRadiusNV ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionLSSHitValueNV ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionLSSPositionsNV ||
                 glslangOp == glslang::EOpRayQueryGetIntersectionLSSRadiiNV ||
                 glslangOp == glslang::EOpRayQueryIsLSSHitNV ||
                 glslangOp == glslang::EOpRayQueryIsSphereHitNV
                    )) {
                bool cond = glslangOperands[arg]->getAsConstantUnion()->getConstArray()[0].getBConst();
                operands.push_back(builder.makeIntConstant(cond ? 1 : 0));
             } else if ((arg == 10 && glslangOp == glslang::EOpTraceKHR) ||
                        (arg == 11 && glslangOp == glslang::EOpTraceRayMotionNV) ||
                        (arg == 1  && glslangOp == glslang::EOpExecuteCallableKHR) ||
                        (arg == 1  && glslangOp == glslang::EOpHitObjectExecuteShaderNV) ||
                        (arg == 1  && glslangOp == glslang::EOpHitObjectExecuteShaderEXT) ||
                        (arg == 11 && glslangOp == glslang::EOpHitObjectTraceRayNV) ||
                        (arg == 11 && glslangOp == glslang::EOpHitObjectTraceRayEXT) ||
                        (arg == 12 && glslangOp == glslang::EOpHitObjectTraceRayMotionNV) ||
                        (arg == 12 && glslangOp == glslang::EOpHitObjectTraceRayMotionEXT) ||
                        (arg == 12 && glslangOp == glslang::EOpHitObjectTraceMotionReorderExecuteEXT && glslangOperands.size() == 13) ||
                        (arg == 14 && glslangOp == glslang::EOpHitObjectTraceMotionReorderExecuteEXT && glslangOperands.size() == 15) ||
                        (arg == 11 && glslangOp == glslang::EOpHitObjectTraceReorderExecuteEXT && glslangOperands.size() == 12) ||
                        (arg == 13 && glslangOp == glslang::EOpHitObjectTraceReorderExecuteEXT && glslangOperands.size() == 14) ||
                        (arg == 1  && glslangOp == glslang::EOpHitObjectReorderExecuteEXT && glslangOperands.size() == 2) ||
                        (arg == 3  && glslangOp == glslang::EOpHitObjectReorderExecuteEXT && glslangOperands.size() == 4)) {
                 const int set = glslangOp == glslang::EOpExecuteCallableKHR ? 1 : 0;
                 const int location = glslangOperands[arg]->getAsConstantUnion()->getConstArray()[0].getUConst();
                 auto itNode = locationToSymbol[set].find(location);
                 visitSymbol(itNode->second);
                 spv::Id symId = getSymbolId(itNode->second);
                 operands.push_back(symId);
            } else if ((arg == 12 && glslangOp == glslang::EOpHitObjectRecordHitNV) ||
                       (arg == 13 && glslangOp == glslang::EOpHitObjectRecordHitMotionNV) ||
                       (arg == 11 && glslangOp == glslang::EOpHitObjectRecordHitWithIndexNV) ||
                       (arg == 12 && glslangOp == glslang::EOpHitObjectRecordHitWithIndexMotionNV) ||
                       (arg == 3  && glslangOp == glslang::EOpHitObjectRecordFromQueryEXT) ||
                       (arg == 1  && glslangOp == glslang::EOpHitObjectGetAttributesEXT) ||
                       (arg == 1  && glslangOp == glslang::EOpHitObjectGetAttributesNV)) {
                 const int location = glslangOperands[arg]->getAsConstantUnion()->getConstArray()[0].getUConst();
                 const int set = 2;
                 auto itNode = locationToSymbol[set].find(location);
                 visitSymbol(itNode->second);
                 spv::Id symId = getSymbolId(itNode->second);
                 operands.push_back(symId);
            } else if (glslangOperands[arg]->getAsTyped()->getQualifier().isSpirvLiteral()) {
                // Will be translated to a literal value, make a placeholder here
                operands.push_back(spv::NoResult);
            } else if (glslangOperands[arg]->getAsTyped()->getBasicType() == glslang::EbtFunction) {
                spv::Function* function = functionMap[glslangOperands[arg]->getAsSymbolNode()->getMangledName().c_str()];
                assert(function);
                operands.push_back(function->getId());
            } else  {
               operands.push_back(accessChainLoad(glslangOperands[arg]->getAsTyped()->getType()));
            }
        }
    }

    builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
    if (node->getOp() == glslang::EOpCooperativeMatrixLoadTensorNV) {
        std::vector<spv::IdImmediate> idImmOps;

        builder.addCapability(spv::Capability::CooperativeMatrixTensorAddressingNV);
        builder.addExtension(spv::E_SPV_NV_cooperative_matrix2);

        spv::Id object = builder.createLoad(operands[0], spv::NoPrecision);

        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // Pointer
        idImmOps.push_back(spv::IdImmediate(true, object)); // Object
        idImmOps.push_back(spv::IdImmediate(true, operands[2])); // tensorLayout

        idImmOps.insert(idImmOps.end(), memoryAccessOperands.begin(), memoryAccessOperands.end()); // memoryaccess

        // initialize tensor operands to zero, then OR in flags based on the operands
        size_t tensorOpIdx = idImmOps.size();
        idImmOps.push_back(spv::IdImmediate(false, 0));

        for (uint32_t i = 3; i < operands.size(); ++i) {
            if (builder.isTensorView(operands[i])) {
                addMask(idImmOps[tensorOpIdx].word, spv::TensorAddressingOperandsMask::TensorView);
            } else {
                // must be the decode func
                addMask(idImmOps[tensorOpIdx].word, spv::TensorAddressingOperandsMask::DecodeFunc);
                builder.addCapability(spv::Capability::CooperativeMatrixBlockLoadsNV);
            }
            idImmOps.push_back(spv::IdImmediate(true, operands[i])); // tensorView or decodeFunc
        }

        // get the pointee type
        spv::Id typeId = builder.getContainedTypeId(builder.getTypeId(operands[0]));
        assert(builder.isCooperativeMatrixType(typeId));
        // do the op
        spv::Id result = builder.createOp(spv::Op::OpCooperativeMatrixLoadTensorNV, typeId, idImmOps);
        // store the result to the pointer (out param 'm')
        builder.createStore(result, operands[0]);
        result = 0;
    } else if (node->getOp() == glslang::EOpCooperativeMatrixLoad ||
               node->getOp() == glslang::EOpCooperativeMatrixLoadNV) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // buf
        if (node->getOp() == glslang::EOpCooperativeMatrixLoad) {
            idImmOps.push_back(spv::IdImmediate(true, operands[3])); // matrixLayout
            auto layout = (spv::CooperativeMatrixLayout)builder.getConstantScalar(operands[3]);
            if (layout == spv::CooperativeMatrixLayout::RowBlockedInterleavedARM ||
                layout == spv::CooperativeMatrixLayout::ColumnBlockedInterleavedARM) {
                builder.addExtension(spv::E_SPV_ARM_cooperative_matrix_layouts);
                builder.addCapability(spv::Capability::CooperativeMatrixLayoutsARM);
            }
            idImmOps.push_back(spv::IdImmediate(true, operands[2])); // stride
        } else {
            idImmOps.push_back(spv::IdImmediate(true, operands[2])); // stride
            idImmOps.push_back(spv::IdImmediate(true, operands[3])); // colMajor
        }
        idImmOps.insert(idImmOps.end(), memoryAccessOperands.begin(), memoryAccessOperands.end());
        // get the pointee type
        spv::Id typeId = builder.getContainedTypeId(builder.getTypeId(operands[0]));
        assert(builder.isCooperativeMatrixType(typeId));
        // do the op
        spv::Id result = node->getOp() == glslang::EOpCooperativeMatrixLoad
                       ? builder.createOp(spv::Op::OpCooperativeMatrixLoadKHR, typeId, idImmOps)
                       : builder.createOp(spv::Op::OpCooperativeMatrixLoadNV, typeId, idImmOps);
        // store the result to the pointer (out param 'm')
        builder.createStore(result, operands[0]);
        result = 0;
    } else if (node->getOp() == glslang::EOpCooperativeMatrixStoreTensorNV) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // buf
        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // object

        builder.addCapability(spv::Capability::CooperativeMatrixTensorAddressingNV);
        builder.addExtension(spv::E_SPV_NV_cooperative_matrix2);

        idImmOps.push_back(spv::IdImmediate(true, operands[2])); // tensorLayout

        idImmOps.insert(idImmOps.end(), memoryAccessOperands.begin(), memoryAccessOperands.end()); // memoryaccess

        if (operands.size() > 3) {
            idImmOps.push_back(spv::IdImmediate(false, spv::TensorAddressingOperandsMask::TensorView));
            idImmOps.push_back(spv::IdImmediate(true, operands[3])); // tensorView
        } else {
            idImmOps.push_back(spv::IdImmediate(false, 0));
        }

        builder.createNoResultOp(spv::Op::OpCooperativeMatrixStoreTensorNV, idImmOps);
        result = 0;
    } else if (node->getOp() == glslang::EOpCooperativeMatrixStore ||
               node->getOp() == glslang::EOpCooperativeMatrixStoreNV) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // buf
        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // object
        if (node->getOp() == glslang::EOpCooperativeMatrixStore) {
            idImmOps.push_back(spv::IdImmediate(true, operands[3])); // matrixLayout
            auto layout = (spv::CooperativeMatrixLayout)builder.getConstantScalar(operands[3]);
            if (layout == spv::CooperativeMatrixLayout::RowBlockedInterleavedARM ||
                layout == spv::CooperativeMatrixLayout::ColumnBlockedInterleavedARM) {
                builder.addExtension(spv::E_SPV_ARM_cooperative_matrix_layouts);
                builder.addCapability(spv::Capability::CooperativeMatrixLayoutsARM);
            }
            idImmOps.push_back(spv::IdImmediate(true, operands[2])); // stride
        } else {
            idImmOps.push_back(spv::IdImmediate(true, operands[2])); // stride
            idImmOps.push_back(spv::IdImmediate(true, operands[3])); // colMajor
        }
        idImmOps.insert(idImmOps.end(), memoryAccessOperands.begin(), memoryAccessOperands.end());

        if (node->getOp() == glslang::EOpCooperativeMatrixStore)
            builder.createNoResultOp(spv::Op::OpCooperativeMatrixStoreKHR, idImmOps);
        else
            builder.createNoResultOp(spv::Op::OpCooperativeMatrixStoreNV, idImmOps);
        result = 0;
    } else if (node->getOp() == glslang::EOpRayQueryGetIntersectionTriangleVertexPositionsEXT) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // q
        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // committed

        spv::Id typeId = builder.makeArrayType(builder.makeVectorType(builder.makeFloatType(32), 3),
                                               builder.makeUintConstant(3), 0);
        // do the op

        spv::Op spvOp = spv::Op::OpRayQueryGetIntersectionTriangleVertexPositionsKHR;

        spv::Id result = builder.createOp(spvOp, typeId, idImmOps);
        // store the result to the pointer (out param 'm')
        builder.createStore(result, operands[2]);
        result = 0;
    } else if (node->getOp() == glslang::EOpRayQueryGetIntersectionLSSPositionsNV) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // q
        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // committed

        spv::Id typeId = builder.makeArrayType(builder.makeVectorType(builder.makeFloatType(32), 3),
                                               builder.makeUintConstant(2), 0);
        // do the op

        spv::Op spvOp = spv::Op::OpRayQueryGetIntersectionLSSPositionsNV;

        spv::Id result = builder.createOp(spvOp, typeId, idImmOps);
        // store the result to the pointer (out param 'm')
        builder.createStore(result, operands[2]);
        result = 0;
    } else if (node->getOp() == glslang::EOpRayQueryGetIntersectionLSSRadiiNV) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // q
        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // committed

        spv::Id typeId = builder.makeArrayType(builder.makeFloatType(32),
                                               builder.makeUintConstant(2), 0);
        // do the op

        spv::Op spvOp = spv::Op::OpRayQueryGetIntersectionLSSRadiiNV;

        spv::Id result = builder.createOp(spvOp, typeId, idImmOps);
        // store the result to the pointer (out param 'm')
        builder.createStore(result, operands[2]);
        result = 0;
    } else if (node->getOp() == glslang::EOpHitObjectGetLSSPositionsNV) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // hitObject

        spv::Op spvOp = spv::Op::OpHitObjectGetLSSPositionsNV;
        spv::Id typeId = builder.makeArrayType(builder.makeVectorType(builder.makeFloatType(32), 3),
                                               builder.makeUintConstant(2), 0);

        spv::Id result = builder.createOp(spvOp, typeId, idImmOps);
        // store the result to the pointer (out param 'm')
        builder.createStore(result, operands[1]);
        result = 0;
    } else if (node->getOp() == glslang::EOpHitObjectGetLSSRadiiNV) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // hitObject

        spv::Op spvOp = spv::Op::OpHitObjectGetLSSRadiiNV;
        spv::Id typeId = builder.makeArrayType(builder.makeFloatType(32),
                                               builder.makeUintConstant(2), 0);

        spv::Id result = builder.createOp(spvOp, typeId, idImmOps);
        // store the result to the pointer (out param 'm')
        builder.createStore(result, operands[1]);
        result = 0;
    } else if (node->getOp() == glslang::EOpHitObjectGetIntersectionTriangleVertexPositionsEXT) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // hitObject

        spv::Op spvOp = spv::Op::OpHitObjectGetIntersectionTriangleVertexPositionsEXT;
        spv::Id typeId = builder.makeArrayType(builder.makeVectorType(builder.makeFloatType(32), 3),
                                               builder.makeUintConstant(3), 0);

        spv::Id result = builder.createOp(spvOp, typeId, idImmOps);
        // store the result to the pointer (out param 'm')
        builder.createStore(result, operands[1]);
        result = 0;
    } else if (node->getOp() == glslang::EOpCooperativeMatrixMulAdd) {
        auto matrixOperands = spv::CooperativeMatrixOperandsMask::MaskNone;

        // If the optional operand is present, initialize matrixOperands to that value.
        if (glslangOperands.size() == 4 && glslangOperands[3]->getAsConstantUnion()) {
            matrixOperands = (spv::CooperativeMatrixOperandsMask)glslangOperands[3]->getAsConstantUnion()->getConstArray()[0].getIConst();
        }

        // Determine Cooperative Matrix Operands bits from the signedness of the types.
        if (isTypeSignedInt(glslangOperands[0]->getAsTyped()->getBasicType()))
            addMask(matrixOperands, spv::CooperativeMatrixOperandsMask::MatrixASignedComponentsKHR);
        if (isTypeSignedInt(glslangOperands[1]->getAsTyped()->getBasicType()))
            addMask(matrixOperands, spv::CooperativeMatrixOperandsMask::MatrixBSignedComponentsKHR);
        if (isTypeSignedInt(glslangOperands[2]->getAsTyped()->getBasicType()))
            addMask(matrixOperands, spv::CooperativeMatrixOperandsMask::MatrixCSignedComponentsKHR);
        if (isTypeSignedInt(node->getBasicType()))
            addMask(matrixOperands, spv::CooperativeMatrixOperandsMask::MatrixResultSignedComponentsKHR);

        std::vector<spv::IdImmediate> idImmOps;
        idImmOps.push_back(spv::IdImmediate(true, operands[0]));
        idImmOps.push_back(spv::IdImmediate(true, operands[1]));
        idImmOps.push_back(spv::IdImmediate(true, operands[2]));
        if (matrixOperands != spv::CooperativeMatrixOperandsMask::MaskNone)
            idImmOps.push_back(spv::IdImmediate(false, matrixOperands));

        result = builder.createOp(spv::Op::OpCooperativeMatrixMulAddKHR, resultType(), idImmOps);
    } else if (node->getOp() == glslang::EOpCooperativeMatrixReduceNV) {
        builder.addCapability(spv::Capability::CooperativeMatrixReductionsNV);
        builder.addExtension(spv::E_SPV_NV_cooperative_matrix2);

        spv::Op opcode = spv::Op::OpCooperativeMatrixReduceNV;
        unsigned mask = glslangOperands[2]->getAsConstantUnion()->getConstArray()[0].getUConst();

        spv::Id typeId = builder.getContainedTypeId(builder.getTypeId(operands[0]));
        assert(builder.isCooperativeMatrixType(typeId));

        result = builder.createCooperativeMatrixReduce(opcode, typeId, operands[1], mask, operands[3]);
        // store the result to the pointer (out param 'm')
        builder.createStore(result, operands[0]);
        result = 0;
    } else if (node->getOp() == glslang::EOpCooperativeMatrixPerElementOpNV) {
        builder.addCapability(spv::Capability::CooperativeMatrixPerElementOperationsNV);
        builder.addExtension(spv::E_SPV_NV_cooperative_matrix2);

        spv::Id typeId = builder.getContainedTypeId(builder.getTypeId(operands[0]));
        assert(builder.isCooperativeMatrixType(typeId));

        result = builder.createCooperativeMatrixPerElementOp(typeId, operands);
        // store the result to the pointer
        builder.createStore(result, operands[0]);
        result = 0;
    } else if (node->getOp() == glslang::EOpCooperativeMatrixTransposeNV) {

        builder.addCapability(spv::Capability::CooperativeMatrixConversionsNV);
        builder.addExtension(spv::E_SPV_NV_cooperative_matrix2);

        spv::Id typeId = builder.getContainedTypeId(builder.getTypeId(operands[0]));
        assert(builder.isCooperativeMatrixType(typeId));

        result = builder.createUnaryOp(spv::Op::OpCooperativeMatrixTransposeNV, typeId, operands[1]);
        // store the result to the pointer
        builder.createStore(result, operands[0]);
        result = 0;
    } else if (node->getOp() == glslang::EOpBitCastArrayQCOM) {
        builder.addCapability(spv::Capability::CooperativeMatrixConversionQCOM);
        builder.addExtension(spv::E_SPV_QCOM_cooperative_matrix_conversion);
        result = builder.createUnaryOp(spv::Op::OpBitCastArrayQCOM, resultType(), operands[0]);
    } else if (node->getOp() == glslang::EOpCompositeConstructCoopMatQCOM) {
        builder.addCapability(spv::Capability::CooperativeMatrixConversionQCOM);
        builder.addExtension(spv::E_SPV_QCOM_cooperative_matrix_conversion);
        result = builder.createUnaryOp(spv::Op::OpCompositeConstructCoopMatQCOM, resultType(), operands[0]);
    } else if (node->getOp() == glslang::EOpCompositeExtractCoopMatQCOM) {
        builder.addCapability(spv::Capability::CooperativeMatrixConversionQCOM);
        builder.addExtension(spv::E_SPV_QCOM_cooperative_matrix_conversion);
        result = builder.createUnaryOp(spv::Op::OpCompositeExtractCoopMatQCOM, resultType(), operands[0]);
    } else if (node->getOp() == glslang::EOpExtractSubArrayQCOM) {
        builder.addCapability(spv::Capability::CooperativeMatrixConversionQCOM);
        builder.addExtension(spv::E_SPV_QCOM_cooperative_matrix_conversion);

        std::vector<spv::Id> arguments { operands[0], operands[1] };;
        result = builder.createOp(spv::Op::OpExtractSubArrayQCOM, resultType(), arguments);
    } else if (node->getOp() == glslang::EOpCooperativeVectorMatMulNV ||
               node->getOp() == glslang::EOpCooperativeVectorMatMulAddNV) {
        auto matrixOperands = spv::CooperativeMatrixOperandsMask::MaskNone;

        bool isMulAdd = node->getOp() == glslang::EOpCooperativeVectorMatMulAddNV;

        // Determine Cooperative Matrix Operands bits from the signedness of the types.

        if (isTypeSignedInt(glslangOperands[1]->getAsTyped()->getBasicType()))
            addMask(matrixOperands, spv::CooperativeMatrixOperandsMask::MatrixBSignedComponentsKHR);
        if (isTypeSignedInt(glslangOperands[0]->getAsTyped()->getBasicType()))
            addMask(matrixOperands, spv::CooperativeMatrixOperandsMask::MatrixResultSignedComponentsKHR);

        uint32_t opIdx = 1;
        std::vector<spv::IdImmediate> idImmOps;
        idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // Input
        idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // InputInterpretation
        idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // Matrix
        idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // MatrixOffset
        idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // MatrixInterpretation
        if (isMulAdd) {
            idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // Bias
            idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // BiasOffset
            idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // BiasInterpretation
        }
        idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // M
        idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // K
        idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // MemoryLayout
        idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // Transpose
        idImmOps.push_back(spv::IdImmediate(true, operands[opIdx++])); // MatrixStride
        if (matrixOperands != spv::CooperativeMatrixOperandsMask::MaskNone)
            idImmOps.push_back(spv::IdImmediate(false, matrixOperands));  // Cooperative Matrix Operands

        // get the pointee type
        spv::Id typeId = builder.getContainedTypeId(builder.getTypeId(operands[0]));
        assert(builder.isCooperativeVectorType(typeId));
        // do the op
        spv::Id result = builder.createOp(isMulAdd ? spv::Op::OpCooperativeVectorMatrixMulAddNV : spv::Op::OpCooperativeVectorMatrixMulNV, typeId, idImmOps);
        // store the result to the pointer (out param 'res')
        builder.createStore(result, operands[0]);
        result = 0;
    } else if (node->getOp() == glslang::EOpCooperativeVectorLoadNV) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // buf
        idImmOps.push_back(spv::IdImmediate(true, operands[2])); // offset
        idImmOps.insert(idImmOps.end(), memoryAccessOperands.begin(), memoryAccessOperands.end());
        // get the pointee type
        spv::Id typeId = builder.getContainedTypeId(builder.getTypeId(operands[0]));
        assert(builder.isCooperativeVectorType(typeId));
        // do the op
        spv::Id result = builder.createOp(spv::Op::OpCooperativeVectorLoadNV, typeId, idImmOps);
        // store the result to the pointer (out param 'v')
        builder.createStore(result, operands[0]);
        result = 0;
    } else if (node->getOp() == glslang::EOpCooperativeVectorStoreNV) {
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // buf
        idImmOps.push_back(spv::IdImmediate(true, operands[2])); // offset
        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // object
        idImmOps.insert(idImmOps.end(), memoryAccessOperands.begin(), memoryAccessOperands.end());
        builder.createNoResultOp(spv::Op::OpCooperativeVectorStoreNV, idImmOps);
        result = 0;
    } else if (node->getOp() == glslang::EOpCooperativeVectorOuterProductAccumulateNV) {
        builder.addCapability(spv::Capability::CooperativeVectorTrainingNV);
        builder.addExtension(spv::E_SPV_NV_cooperative_vector);

        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[2])); // Matrix
        idImmOps.push_back(spv::IdImmediate(true, operands[3])); // Offset
        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // A
        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // B
        idImmOps.push_back(spv::IdImmediate(true, operands[5])); // MemoryLayout
        idImmOps.push_back(spv::IdImmediate(true, operands[6])); // MatrixInterpretation
        idImmOps.push_back(spv::IdImmediate(true, operands[4])); // Stride
        builder.createNoResultOp(spv::Op::OpCooperativeVectorOuterProductAccumulateNV, idImmOps);
        result = 0;
    } else if (node->getOp() == glslang::EOpCooperativeVectorReduceSumAccumulateNV) {
        builder.addCapability(spv::Capability::CooperativeVectorTrainingNV);
        builder.addExtension(spv::E_SPV_NV_cooperative_vector);

        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // Buf
        idImmOps.push_back(spv::IdImmediate(true, operands[2])); // Offset
        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // A
        builder.createNoResultOp(spv::Op::OpCooperativeVectorReduceSumAccumulateNV, idImmOps);
        result = 0;
    } else if (node->getOp() == glslang::EOpTensorReadARM ||
               node->getOp() == glslang::EOpTensorWriteARM) {
        const bool isWrite = node->getOp() == glslang::EOpTensorWriteARM;
        const unsigned int tensorMinOperandCount = 3;
        assert(operands.size() >= tensorMinOperandCount);
        std::vector<spv::IdImmediate> idImmOps;

        idImmOps.push_back(spv::IdImmediate(true, operands[0])); // tensor
        idImmOps.push_back(spv::IdImmediate(true, operands[1])); // coords
        if (isWrite) {
            idImmOps.push_back(spv::IdImmediate(true, operands[2])); // value
        }

        // Analyze the tensor operands
        spv::IdImmediate tensorOperands = { false, uint32_t(spv::TensorOperandsMask::MaskNone) };
        bool pushExtraArg = false;
        if (operands.size() > tensorMinOperandCount) {
            auto enumVal = builder.getConstantScalar(operands[tensorMinOperandCount]);

            if (enumVal & uint32_t(spv::TensorOperandsMask::NontemporalARM)) {
                tensorOperands.word |= uint32_t(spv::TensorOperandsMask::NontemporalARM);
            }
            if (enumVal & uint32_t(spv::TensorOperandsMask::OutOfBoundsValueARM)) {
                tensorOperands.word |= uint32_t(spv::TensorOperandsMask::OutOfBoundsValueARM);
                assert(operands.size() >= tensorMinOperandCount + 2 &&
                    "TensorOperandsOutOfBoundsValueMask requires an additional value");
                pushExtraArg = true;
            }
        }

        // Append optional tensor operands if the mask was non-zero.
        if (tensorOperands.word) {
            idImmOps.push_back(tensorOperands);
            if (pushExtraArg)
                idImmOps.push_back(spv::IdImmediate(true, operands[tensorMinOperandCount + 1]));
        }

        if (isWrite) {
            builder.createNoResultOp(spv::Op::OpTensorWriteARM, idImmOps);
            result = 0;
        } else {
            // Use the result argument type as the OpTensorReadARM result type.
            const glslang::TType &resArgType = glslangOperands[2]->getAsTyped()->getType();
            spv::Id retType = convertGlslangToSpvType(resArgType);
            result = builder.createOp(spv::Op::OpTensorReadARM, retType, idImmOps);
            // Store the result to the result argument.
            builder.createStore(result, operands[2]);
        }
    } else if (node->getOp() == glslang::EOpTensorSizeARM) {
        // Expected operands are (tensor, dimension)
        assert(operands.size() == 2);

        spv::Id tensorOp = operands[0];
        spv::Id dimOp = operands[1];
        assert(builder.isTensorTypeARM(builder.getTypeId(tensorOp)) && "operand #0 must be a tensor");

        std::vector<spv::IdImmediate> idImmOps;
        idImmOps.push_back(spv::IdImmediate(true, tensorOp));
        idImmOps.push_back(spv::IdImmediate(true, dimOp));
        result = builder.createOp(spv::Op::OpTensorQuerySizeARM, resultType(), idImmOps);
    } else if (atomic) {
        // Handle all atomics
        glslang::TBasicType typeProxy = (node->getOp() == glslang::EOpAtomicStore)
            ? node->getSequence()[0]->getAsTyped()->getBasicType() : node->getBasicType();
        result = createAtomicOperation(node->getOp(), precision, resultType(), operands, typeProxy,
            lvalueCoherentFlags, node->getType());
    } else if (node->getOp() == glslang::EOpSpirvInst) {
        const auto& spirvInst = node->getSpirvInstruction();
        if (spirvInst.set == "") {
            std::vector<spv::IdImmediate> idImmOps;
            for (unsigned int i = 0; i < glslangOperands.size(); ++i) {
                if (glslangOperands[i]->getAsTyped()->getQualifier().isSpirvLiteral()) {
                    // Translate the constant to a literal value
                    std::vector<unsigned> literals;
                    glslang::TVector<const glslang::TIntermConstantUnion*> constants;
                    constants.push_back(glslangOperands[i]->getAsConstantUnion());
                    TranslateLiterals(constants, literals);
                    idImmOps.push_back({false, literals[0]});
                } else
                    idImmOps.push_back({true, operands[i]});
            }

            if (node->getBasicType() == glslang::EbtVoid)
                builder.createNoResultOp(static_cast<spv::Op>(spirvInst.id), idImmOps);
            else
                result = builder.createOp(static_cast<spv::Op>(spirvInst.id), resultType(), idImmOps);
        } else {
            result = builder.createBuiltinCall(
                resultType(), spirvInst.set == "GLSL.std.450" ? stdBuiltins : getExtBuiltins(spirvInst.set.c_str()),
                spirvInst.id, operands);
        }
        noReturnValue = node->getBasicType() == glslang::EbtVoid;
    } else if (node->getOp() == glslang::EOpDebugPrintf) {
        if (!nonSemanticDebugPrintf) {
            nonSemanticDebugPrintf = builder.import("NonSemantic.DebugPrintf");
        }
        result = builder.createBuiltinCall(builder.makeVoidType(), nonSemanticDebugPrintf, spv::NonSemanticDebugPrintfDebugPrintf, operands);
        builder.addExtension(spv::E_SPV_KHR_non_semantic_info);
    } else {
        // Pass through to generic operations.
        switch (glslangOperands.size()) {
        case 0:
            result = createNoArgOperation(node->getOp(), precision, resultType());
            break;
        case 1:
            {
                OpDecorations decorations = { precision,
                                              TranslateNoContractionDecoration(node->getType().getQualifier()),
                                              TranslateNonUniformDecoration(node->getType().getQualifier()) };
                result = createUnaryOperation(
                    node->getOp(), decorations,
                    resultType(), operands.front(),
                    glslangOperands[0]->getAsTyped()->getBasicType(), lvalueCoherentFlags, node->getType());
            }
            break;
        default:
            result = createMiscOperation(node->getOp(), precision, resultType(), operands, node->getBasicType());
            break;
        }

        if (invertedType != spv::NoResult)
            result = createInvertedSwizzle(precision, *glslangOperands[0]->getAsBinaryNode(), result);

        for (unsigned int i = 0; i < temporaryLvalues.size(); ++i) {
            builder.setAccessChain(complexLvalues[i]);
            builder.accessChainStore(builder.createLoad(temporaryLvalues[i], spv::NoPrecision),
                TranslateNonUniformDecoration(complexLvalues[i].coherentFlags));
        }
    }

    if (noReturnValue)
        return false;

    if (! result) {
        logger->missingFunctionality("unknown glslang aggregate");
        return true;  // pick up a child as a placeholder operand
    } else {
        builder.clearAccessChain();
        builder.setAccessChainRValue(result);
        return false;
    }
}

// This path handles both if-then-else and ?:
// The if-then-else has a node type of void, while
// ?: has either a void or a non-void node type
//
// Leaving the result, when not void:
// GLSL only has r-values as the result of a :?, but
// if we have an l-value, that can be more efficient if it will
// become the base of a complex r-value expression, because the
// next layer copies r-values into memory to use the access-chain mechanism
bool TGlslangToSpvTraverser::visitSelection(glslang::TVisit /* visit */, glslang::TIntermSelection* node)
{
    // see if OpSelect can handle it
    const auto isOpSelectable = [&]() {
        if (node->getBasicType() == glslang::EbtVoid)
            return false;
        // OpSelect can do all other types starting with SPV 1.4
        if (glslangIntermediate->getSpv().spv < glslang::EShTargetSpv_1_4) {
            // pre-1.4, only scalars and vectors can be handled
            if ((!node->getType().isScalar() && !node->getType().isVector()))
                return false;
        }
        return true;
    };

    // See if it simple and safe, or required, to execute both sides.
    // Crucially, side effects must be either semantically required or avoided,
    // and there are performance trade-offs.
    // Return true if required or a good idea (and safe) to execute both sides,
    // false otherwise.
    const auto bothSidesPolicy = [&]() -> bool {
        // do we have both sides?
        if (node->getTrueBlock()  == nullptr ||
            node->getFalseBlock() == nullptr)
            return false;

        // required? (unless we write additional code to look for side effects
        // and make performance trade-offs if none are present)
        if (!node->getShortCircuit())
            return true;

        // if not required to execute both, decide based on performance/practicality...

        if (!isOpSelectable())
            return false;

        assert(node->getType() == node->getTrueBlock() ->getAsTyped()->getType() &&
               node->getType() == node->getFalseBlock()->getAsTyped()->getType());

        // return true if a single operand to ? : is okay for OpSelect
        const auto operandOkay = [](glslang::TIntermTyped* node) {
            return node->getAsSymbolNode() || node->getType().getQualifier().isConstant();
        };

        return operandOkay(node->getTrueBlock() ->getAsTyped()) &&
               operandOkay(node->getFalseBlock()->getAsTyped());
    };

    spv::Id result = spv::NoResult; // upcoming result selecting between trueValue and falseValue
    // emit the condition before doing anything with selection
    node->getCondition()->traverse(this);
    spv::Id condition = accessChainLoad(node->getCondition()->getType());

    // Find a way of executing both sides and selecting the right result.
    const auto executeBothSides = [&]() -> void {
        // execute both sides
        spv::Id resultType = convertGlslangToSpvType(node->getType());
        node->getTrueBlock()->traverse(this);
        spv::Id trueValue = accessChainLoad(node->getTrueBlock()->getAsTyped()->getType());
        node->getFalseBlock()->traverse(this);
        spv::Id falseValue = accessChainLoad(node->getFalseBlock()->getAsTyped()->getType());

        builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());

        // done if void
        if (node->getBasicType() == glslang::EbtVoid)
            return;

        // emit code to select between trueValue and falseValue
        // see if OpSelect can handle the result type, and that the SPIR-V types
        // of the inputs match the result type.
        if (isOpSelectable()) {
            // Emit OpSelect for this selection.

            // smear condition to vector, if necessary (AST is always scalar)
            // Before 1.4, smear like for mix(), starting with 1.4, keep it scalar
            if (glslangIntermediate->getSpv().spv < glslang::EShTargetSpv_1_4 && builder.isVector(trueValue)) {
                condition = builder.smearScalar(spv::NoPrecision, condition,
                                                builder.makeVectorType(builder.makeBoolType(),
                                                                       builder.getNumComponents(trueValue)));
            }

            // If the types do not match, it is because of mismatched decorations on aggregates.
            // Since isOpSelectable only lets us get here for SPIR-V >= 1.4, we can use OpCopyObject
            // to get matching types.
            if (builder.getTypeId(trueValue) != resultType) {
                trueValue = builder.createUnaryOp(spv::Op::OpCopyLogical, resultType, trueValue);
            }
            if (builder.getTypeId(falseValue) != resultType) {
                falseValue = builder.createUnaryOp(spv::Op::OpCopyLogical, resultType, falseValue);
            }

            // OpSelect
            result = builder.createTriOp(spv::Op::OpSelect, resultType, condition, trueValue, falseValue);

            builder.clearAccessChain();
            builder.setAccessChainRValue(result);
        } else {
            // We need control flow to select the result.
            // TODO: Once SPIR-V OpSelect allows arbitrary types, eliminate this path.
            result = builder.createVariable(TranslatePrecisionDecoration(node->getType()),
                spv::StorageClass::Function, resultType);

            // Selection control:
            const spv::SelectionControlMask control = TranslateSelectionControl(*node);

            // make an "if" based on the value created by the condition
            spv::Builder::If ifBuilder(condition, control, builder);

            // emit the "then" statement
            builder.clearAccessChain();
            builder.setAccessChainLValue(result);
            multiTypeStore(node->getType(), trueValue);

            ifBuilder.makeBeginElse();
            // emit the "else" statement
            builder.clearAccessChain();
            builder.setAccessChainLValue(result);
            multiTypeStore(node->getType(), falseValue);

            // finish off the control flow
            ifBuilder.makeEndIf();

            builder.clearAccessChain();
            builder.setAccessChainLValue(result);
        }
    };

    // Execute the one side needed, as per the condition
    const auto executeOneSide = [&]() {
        // Always emit control flow.
        if (node->getBasicType() != glslang::EbtVoid) {
            result = builder.createVariable(TranslatePrecisionDecoration(node->getType()), spv::StorageClass::Function,
                convertGlslangToSpvType(node->getType()));
        }

        // Selection control:
        const spv::SelectionControlMask control = TranslateSelectionControl(*node);

        // make an "if" based on the value created by the condition
        spv::Builder::If ifBuilder(condition, control, builder);

        // emit the "then" statement
        if (node->getTrueBlock() != nullptr) {
            node->getTrueBlock()->traverse(this);
            if (result != spv::NoResult) {
                spv::Id load = accessChainLoad(node->getTrueBlock()->getAsTyped()->getType());

                builder.clearAccessChain();
                builder.setAccessChainLValue(result);
                multiTypeStore(node->getType(), load);
            }
        }

        if (node->getFalseBlock() != nullptr) {
            ifBuilder.makeBeginElse();
            // emit the "else" statement
            node->getFalseBlock()->traverse(this);
            if (result != spv::NoResult) {
                spv::Id load = accessChainLoad(node->getFalseBlock()->getAsTyped()->getType());

                builder.clearAccessChain();
                builder.setAccessChainLValue(result);
                multiTypeStore(node->getType(), load);
            }
        }

        // finish off the control flow
        ifBuilder.makeEndIf();

        if (result != spv::NoResult) {
            builder.clearAccessChain();
            builder.setAccessChainLValue(result);
        }
    };

    // Try for OpSelect (or a requirement to execute both sides)
    if (bothSidesPolicy()) {
        SpecConstantOpModeGuard spec_constant_op_mode_setter(&builder);
        if (node->getType().getQualifier().isSpecConstant())
            spec_constant_op_mode_setter.turnOnSpecConstantOpMode();
        executeBothSides();
    } else
        executeOneSide();

    return false;
}

bool TGlslangToSpvTraverser::visitSwitch(glslang::TVisit /* visit */, glslang::TIntermSwitch* node)
{
    // emit and get the condition before doing anything with switch
    node->getCondition()->traverse(this);
    spv::Id selector = accessChainLoad(node->getCondition()->getAsTyped()->getType());

    // Selection control:
    const spv::SelectionControlMask control = TranslateSwitchControl(*node);

    // browse the children to sort out code segments
    int defaultSegment = -1;
    std::vector<TIntermNode*> codeSegments;
    glslang::TIntermSequence& sequence = node->getBody()->getSequence();
    std::vector<int> caseValues;
    std::vector<int> valueIndexToSegment(sequence.size());  // note: probably not all are used, it is an overestimate
    for (glslang::TIntermSequence::iterator c = sequence.begin(); c != sequence.end(); ++c) {
        TIntermNode* child = *c;
        if (child->getAsBranchNode() && child->getAsBranchNode()->getFlowOp() == glslang::EOpDefault)
            defaultSegment = (int)codeSegments.size();
        else if (child->getAsBranchNode() && child->getAsBranchNode()->getFlowOp() == glslang::EOpCase) {
            valueIndexToSegment[caseValues.size()] = (int)codeSegments.size();
            caseValues.push_back(child->getAsBranchNode()->getExpression()->getAsConstantUnion()
                ->getConstArray()[0].getIConst());
        } else
            codeSegments.push_back(child);
    }

    // handle the case where the last code segment is missing, due to no code
    // statements between the last case and the end of the switch statement
    if ((caseValues.size() && (int)codeSegments.size() == valueIndexToSegment[caseValues.size() - 1]) ||
        (int)codeSegments.size() == defaultSegment)
        codeSegments.push_back(nullptr);

    // make the switch statement
    std::vector<spv::Block*> segmentBlocks; // returned, as the blocks allocated in the call
    builder.makeSwitch(selector, control, (int)codeSegments.size(), caseValues, valueIndexToSegment, defaultSegment,
        segmentBlocks);

    // emit all the code in the segments
    breakForLoop.push(false);
    for (unsigned int s = 0; s < codeSegments.size(); ++s) {
        builder.nextSwitchSegment(segmentBlocks, s);
        if (codeSegments[s])
            codeSegments[s]->traverse(this);
        else
            builder.addSwitchBreak(true);
    }
    breakForLoop.pop();

    builder.endSwitch(segmentBlocks);

    return false;
}

void TGlslangToSpvTraverser::visitConstantUnion(glslang::TIntermConstantUnion* node)
{
    if (node->getQualifier().isSpirvLiteral())
        return; // Translated to a literal value, skip further processing

    int nextConst = 0;
    spv::Id constant = createSpvConstantFromConstUnionArray(node->getType(), node->getConstArray(), nextConst, false);

    builder.clearAccessChain();
    builder.setAccessChainRValue(constant);
}

bool TGlslangToSpvTraverser::visitLoop(glslang::TVisit /* visit */, glslang::TIntermLoop* node)
{
    auto blocks = builder.makeNewLoop();
    builder.createBranch(true, &blocks.head);

    // Loop control:
    std::vector<unsigned int> operands;
    const spv::LoopControlMask control = TranslateLoopControl(*node, operands);

    // Spec requires back edges to target header blocks, and every header block
    // must dominate its merge block.  Make a header block first to ensure these
    // conditions are met.  By definition, it will contain OpLoopMerge, followed
    // by a block-ending branch.  But we don't want to put any other body/test
    // instructions in it, since the body/test may have arbitrary instructions,
    // including merges of its own.
    builder.setBuildPoint(&blocks.head);
    builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
    builder.createLoopMerge(&blocks.merge, &blocks.continue_target, control, operands);
    if (node->testFirst() && node->getTest()) {
        spv::Block& test = builder.makeNewBlock();
        builder.createBranch(true, &test);

        builder.setBuildPoint(&test);
        node->getTest()->traverse(this);
        spv::Id condition = accessChainLoad(node->getTestExpr()->getType());
        builder.createConditionalBranch(condition, &blocks.body, &blocks.merge);

        builder.setBuildPoint(&blocks.body);
        breakForLoop.push(true);
        if (node->getBody())
            node->getBody()->traverse(this);
        builder.createBranch(true, &blocks.continue_target);
        breakForLoop.pop();

        builder.setBuildPoint(&blocks.continue_target);
        if (node->getTerminal())
            node->getTerminal()->traverse(this);
        builder.createBranch(true, &blocks.head);
    } else {
        builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());
        builder.createBranch(true, &blocks.body);

        breakForLoop.push(true);
        builder.setBuildPoint(&blocks.body);
        if (node->getBody())
            node->getBody()->traverse(this);
        builder.createBranch(true, &blocks.continue_target);
        breakForLoop.pop();

        builder.setBuildPoint(&blocks.continue_target);
        if (node->getTerminal())
            node->getTerminal()->traverse(this);
        if (node->getTest()) {
            node->getTest()->traverse(this);
            spv::Id condition =
                accessChainLoad(node->getTestExpr()->getType());
            builder.createConditionalBranch(condition, &blocks.head, &blocks.merge);
        } else {
            // TODO: unless there was a break/return/discard instruction
            // somewhere in the body, this is an infinite loop, so we should
            // issue a warning.
            builder.createBranch(true, &blocks.head);
        }
    }
    builder.setBuildPoint(&blocks.merge);
    builder.closeLoop();
    return false;
}

bool TGlslangToSpvTraverser::visitBranch(glslang::TVisit /* visit */, glslang::TIntermBranch* node)
{
    if (node->getExpression())
        node->getExpression()->traverse(this);

    builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());

    switch (node->getFlowOp()) {
    case glslang::EOpKill:
        if (glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_6) {
            builder.addCapability(spv::Capability::DemoteToHelperInvocation);
            builder.createNoResultOp(spv::Op::OpDemoteToHelperInvocationEXT);
        } else {
            builder.makeStatementTerminator(spv::Op::OpKill, "post-discard");
        }
        break;
    case glslang::EOpTerminateInvocation:
        builder.addExtension(spv::E_SPV_KHR_terminate_invocation);
        builder.makeStatementTerminator(spv::Op::OpTerminateInvocation, "post-terminate-invocation");
        break;
    case glslang::EOpBreak:
        if (breakForLoop.top())
            builder.createLoopExit();
        else
            builder.addSwitchBreak(false);
        break;
    case glslang::EOpContinue:
        builder.createLoopContinue();
        break;
    case glslang::EOpReturn:
        if (node->getExpression() != nullptr) {
            const glslang::TType& glslangReturnType = node->getExpression()->getType();
            spv::Id returnId = accessChainLoad(glslangReturnType);
            if (builder.getTypeId(returnId) != currentFunction->getReturnType() ||
                TranslatePrecisionDecoration(glslangReturnType) != currentFunction->getReturnPrecision()) {
                builder.clearAccessChain();
                spv::Id copyId = builder.createVariable(currentFunction->getReturnPrecision(),
                    spv::StorageClass::Function, currentFunction->getReturnType());
                builder.setAccessChainLValue(copyId);
                multiTypeStore(glslangReturnType, returnId);
                returnId = builder.createLoad(copyId, currentFunction->getReturnPrecision());
            }
            builder.makeReturn(false, returnId);
        } else
            builder.makeReturn(false);

        builder.clearAccessChain();
        break;

    case glslang::EOpDemote:
        builder.createNoResultOp(spv::Op::OpDemoteToHelperInvocationEXT);
        builder.addExtension(spv::E_SPV_EXT_demote_to_helper_invocation);
        builder.addCapability(spv::Capability::DemoteToHelperInvocationEXT);
        break;
    case glslang::EOpTerminateRayKHR:
        builder.makeStatementTerminator(spv::Op::OpTerminateRayKHR, "post-terminateRayKHR");
        break;
    case glslang::EOpIgnoreIntersectionKHR:
        builder.makeStatementTerminator(spv::Op::OpIgnoreIntersectionKHR, "post-ignoreIntersectionKHR");
        break;

    default:
        assert(0);
        break;
    }

    return false;
}

bool TGlslangToSpvTraverser::visitVariableDecl(glslang::TVisit visit, glslang::TIntermVariableDecl* node)
{
    if (visit == glslang::EvPreVisit) {
        builder.setDebugSourceLocation(node->getDeclSymbol()->getLoc().line, node->getDeclSymbol()->getLoc().getFilename());
        // We touch the symbol once here to create the debug info.
        getSymbolId(node->getDeclSymbol());
    }

    return true;
}


spv::Id TGlslangToSpvTraverser::createSpvVariable(const glslang::TIntermSymbol* node, spv::Id forcedType)
{
    // First, steer off constants, which are not SPIR-V variables, but
    // can still have a mapping to a SPIR-V Id.
    // This includes specialization constants.
    if (node->getQualifier().isConstant()) {
        spv::Id result = createSpvConstant(*node);
        if (result != spv::NoResult) {
            auto name = node->getAsSymbolNode()->getAccessName().c_str();
            auto typeId = convertGlslangToSpvType(node->getType());
            builder.createConstVariable(typeId, name, result, currentFunction == nullptr);
            return result;
        }
    }

    // Now, handle actual variables
    spv::StorageClass storageClass = TranslateStorageClass(node->getType());
    spv::Id spvType = forcedType == spv::NoType ? convertGlslangToSpvType(node->getType())
                                                : forcedType;

    const bool contains16BitType = node->getType().contains16BitFloat() ||
                                   node->getType().contains16BitInt();
    if (contains16BitType) {
        switch (storageClass) {
        case spv::StorageClass::Input:
        case spv::StorageClass::Output:
            builder.addIncorporatedExtension(spv::E_SPV_KHR_16bit_storage, spv::Spv_1_3);
            builder.addCapability(spv::Capability::StorageInputOutput16);
            break;
        case spv::StorageClass::Uniform:
            builder.addIncorporatedExtension(spv::E_SPV_KHR_16bit_storage, spv::Spv_1_3);
            if (node->getType().getQualifier().storage == glslang::EvqBuffer)
                builder.addCapability(spv::Capability::StorageUniformBufferBlock16);
            else
                builder.addCapability(spv::Capability::StorageUniform16);
            break;
        case spv::StorageClass::PushConstant:
            builder.addIncorporatedExtension(spv::E_SPV_KHR_16bit_storage, spv::Spv_1_3);
            builder.addCapability(spv::Capability::StoragePushConstant16);
            break;
        case spv::StorageClass::StorageBuffer:
        case spv::StorageClass::PhysicalStorageBufferEXT:
            builder.addIncorporatedExtension(spv::E_SPV_KHR_16bit_storage, spv::Spv_1_3);
            builder.addCapability(spv::Capability::StorageUniformBufferBlock16);
            break;
        case spv::StorageClass::TileAttachmentQCOM:
            builder.addCapability(spv::Capability::TileShadingQCOM);
            break;
        default:
            if (storageClass == spv::StorageClass::Workgroup &&
                node->getType().getBasicType() == glslang::EbtBlock) {
                builder.addCapability(spv::Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR);
                break;
            }
            if (node->getType().contains16BitFloat())
                builder.addCapability(spv::Capability::Float16);
            if (node->getType().contains16BitInt())
                builder.addCapability(spv::Capability::Int16);
            break;
        }
    }

    if (node->getType().contains8BitInt()) {
        if (storageClass == spv::StorageClass::PushConstant) {
            builder.addIncorporatedExtension(spv::E_SPV_KHR_8bit_storage, spv::Spv_1_5);
            builder.addCapability(spv::Capability::StoragePushConstant8);
        } else if (storageClass == spv::StorageClass::Uniform) {
            builder.addIncorporatedExtension(spv::E_SPV_KHR_8bit_storage, spv::Spv_1_5);
            builder.addCapability(spv::Capability::UniformAndStorageBuffer8BitAccess);
        } else if (storageClass == spv::StorageClass::StorageBuffer) {
            builder.addIncorporatedExtension(spv::E_SPV_KHR_8bit_storage, spv::Spv_1_5);
            builder.addCapability(spv::Capability::StorageBuffer8BitAccess);
        } else if (storageClass == spv::StorageClass::Workgroup &&
                   node->getType().getBasicType() == glslang::EbtBlock) {
            builder.addCapability(spv::Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR);
        } else {
            builder.addCapability(spv::Capability::Int8);
        }
    }

    const char* name = node->getName().c_str();
    if (glslang::IsAnonymous(name))
        name = "";

    spv::Id initializer = spv::NoResult;

    if (node->getType().getQualifier().storage == glslang::EvqUniform && !node->getConstArray().empty()) {
        int nextConst = 0;
        initializer = createSpvConstantFromConstUnionArray(node->getType(),
                                                           node->getConstArray(),
                                                           nextConst,
                                                           false /* specConst */);
    } else if (node->getType().getQualifier().isNullInit()) {
        initializer = builder.makeNullConstant(spvType);
    }

    spv::Id var = builder.createVariable(spv::NoPrecision, storageClass, spvType, name, initializer, false);

    if (options.emitNonSemanticShaderDebugInfo && storageClass != spv::StorageClass::Function) {
        // Create variable alias for retargeted symbols if any.
        // Notably, this is only applicable to built-in variables so that it is okay to only use name as the key.
        auto [itBegin, itEnd] = glslangIntermediate->getBuiltinAliasLookup().equal_range(name);
        for (auto it = itBegin; it != itEnd; ++it) {
            builder.createDebugGlobalVariable(builder.getDebugType(spvType), it->second.c_str(), var);
        }
    }

    std::vector<spv::Decoration> topLevelDecorations;
    glslang::TQualifier typeQualifier = node->getType().getQualifier();
    TranslateMemoryDecoration(typeQualifier, topLevelDecorations, glslangIntermediate->usingVulkanMemoryModel());
    for (auto deco : topLevelDecorations) {
        builder.addDecoration(var, deco);
    }
    return var;
}

// Return type Id of the sampled type.
spv::Id TGlslangToSpvTraverser::getSampledType(const glslang::TSampler& sampler)
{
    switch (sampler.type) {
        case glslang::EbtInt:      return builder.makeIntType(32);
        case glslang::EbtUint:     return builder.makeUintType(32);
        case glslang::EbtFloat:    return builder.makeFloatType(32);
        case glslang::EbtFloat16:
            builder.addExtension(spv::E_SPV_AMD_gpu_shader_half_float_fetch);
            builder.addCapability(spv::Capability::Float16ImageAMD);
            return builder.makeFloatType(16);
        case glslang::EbtInt64:
            builder.addExtension(spv::E_SPV_EXT_shader_image_int64);
            builder.addCapability(spv::Capability::Int64ImageEXT);
            return builder.makeIntType(64);
        case glslang::EbtUint64:
            builder.addExtension(spv::E_SPV_EXT_shader_image_int64);
            builder.addCapability(spv::Capability::Int64ImageEXT);
            return builder.makeUintType(64);
        default:
            assert(0);
            return builder.makeFloatType(32);
    }
}

// If node is a swizzle operation, return the type that should be used if
// the swizzle base is first consumed by another operation, before the swizzle
// is applied.
spv::Id TGlslangToSpvTraverser::getInvertedSwizzleType(const glslang::TIntermTyped& node)
{
    if (node.getAsOperator() &&
        node.getAsOperator()->getOp() == glslang::EOpVectorSwizzle)
        return convertGlslangToSpvType(node.getAsBinaryNode()->getLeft()->getType());
    else
        return spv::NoType;
}

// When inverting a swizzle with a parent op, this function
// will apply the swizzle operation to a completed parent operation.
spv::Id TGlslangToSpvTraverser::createInvertedSwizzle(spv::Decoration precision, const glslang::TIntermTyped& node,
    spv::Id parentResult)
{
    std::vector<unsigned> swizzle;
    convertSwizzle(*node.getAsBinaryNode()->getRight()->getAsAggregate(), swizzle);
    return builder.createRvalueSwizzle(precision, convertGlslangToSpvType(node.getType()), parentResult, swizzle);
}

// Convert a glslang AST swizzle node to a swizzle vector for building SPIR-V.
void TGlslangToSpvTraverser::convertSwizzle(const glslang::TIntermAggregate& node, std::vector<unsigned>& swizzle)
{
    const glslang::TIntermSequence& swizzleSequence = node.getSequence();
    for (int i = 0; i < (int)swizzleSequence.size(); ++i)
        swizzle.push_back(swizzleSequence[i]->getAsConstantUnion()->getConstArray()[0].getIConst());
}

// Convert from a glslang type to an SPV type, by calling into a
// recursive version of this function. This establishes the inherited
// layout state rooted from the top-level type.
spv::Id TGlslangToSpvTraverser::convertGlslangToSpvType(const glslang::TType& type, bool forwardReferenceOnly)
{
    return convertGlslangToSpvType(type, getExplicitLayout(type), type.getQualifier(), false, forwardReferenceOnly);
}

spv::LinkageType TGlslangToSpvTraverser::convertGlslangLinkageToSpv(glslang::TLinkType linkType)
{
    switch (linkType) {
    case glslang::ELinkExport:
        return spv::LinkageType::Export;
    default:
        return spv::LinkageType::Max;
    }
}

// Do full recursive conversion of an arbitrary glslang type to a SPIR-V Id.
// explicitLayout can be kept the same throughout the hierarchical recursive walk.
// Mutually recursive with convertGlslangStructToSpvType().
spv::Id TGlslangToSpvTraverser::convertGlslangToSpvType(const glslang::TType& type,
    glslang::TLayoutPacking explicitLayout, const glslang::TQualifier& qualifier,
    bool lastBufferBlockMember, bool forwardReferenceOnly)
{
    spv::Id spvType = spv::NoResult;

    switch (type.getBasicType()) {
    case glslang::EbtVoid:
        spvType = builder.makeVoidType();
        assert (! type.isArray());
        break;
    case glslang::EbtBool:
        // "transparent" bool doesn't exist in SPIR-V.  The GLSL convention is
        // a 32-bit int where non-0 means true.
        if (explicitLayout != glslang::ElpNone)
            spvType = builder.makeUintType(32);
        else
            spvType = builder.makeBoolType();
        break;
    case glslang::EbtInt:
        spvType = builder.makeIntType(32);
        break;
    case glslang::EbtUint:
        spvType = builder.makeUintType(32);
        break;
    case glslang::EbtFloat:
        spvType = builder.makeFloatType(32);
        break;
    case glslang::EbtDouble:
        spvType = builder.makeFloatType(64);
        break;
    case glslang::EbtFloat16:
        spvType = builder.makeFloatType(16);
        break;
    case glslang::EbtBFloat16:
        spvType = builder.makeBFloat16Type();
        break;
    case glslang::EbtFloatE5M2:
        spvType = builder.makeFloatE5M2Type();
        break;
    case glslang::EbtFloatE4M3:
        spvType = builder.makeFloatE4M3Type();
        break;
    case glslang::EbtInt8:
        spvType = builder.makeIntType(8);
        break;
    case glslang::EbtUint8:
        spvType = builder.makeUintType(8);
        break;
    case glslang::EbtInt16:
        spvType = builder.makeIntType(16);
        break;
    case glslang::EbtUint16:
        spvType = builder.makeUintType(16);
        break;
    case glslang::EbtInt64:
        spvType = builder.makeIntType(64);
        break;
    case glslang::EbtUint64:
        spvType = builder.makeUintType(64);
        break;
    case glslang::EbtAtomicUint:
        builder.addCapability(spv::Capability::AtomicStorage);
        spvType = builder.makeUintType(32);
        break;
    case glslang::EbtAccStruct:
        switch (glslangIntermediate->getStage()) {
        case EShLangRayGen:
        case EShLangIntersect:
        case EShLangAnyHit:
        case EShLangClosestHit:
        case EShLangMiss:
        case EShLangCallable:
            // these all should have the RayTracingNV/KHR capability already
            break;
        default:
            {
                auto& extensions = glslangIntermediate->getRequestedExtensions();
                if (extensions.find("GL_EXT_ray_query") != extensions.end()) {
                    builder.addExtension(spv::E_SPV_KHR_ray_query);
                    builder.addCapability(spv::Capability::RayQueryKHR);
                }
            }
            break;
        }
        spvType = builder.makeAccelerationStructureType();
        break;
    case glslang::EbtRayQuery:
        {
            auto& extensions = glslangIntermediate->getRequestedExtensions();
            if (extensions.find("GL_EXT_ray_query") != extensions.end()) {
                builder.addExtension(spv::E_SPV_KHR_ray_query);
                builder.addCapability(spv::Capability::RayQueryKHR);
            }
            spvType = builder.makeRayQueryType();
        }
        break;
    case glslang::EbtReference:
        {
            // Make the forward pointer, then recurse to convert the structure type, then
            // patch up the forward pointer with a real pointer type.
            if (forwardPointers.find(type.getReferentType()) == forwardPointers.end()) {
                spv::Id forwardId = builder.makeForwardPointer(spv::StorageClass::PhysicalStorageBufferEXT);
                forwardPointers[type.getReferentType()] = forwardId;
            }
            spvType = forwardPointers[type.getReferentType()];
            if (!forwardReferenceOnly) {
                spv::Id referentType = convertGlslangToSpvType(*type.getReferentType());
                builder.makePointerFromForwardPointer(spv::StorageClass::PhysicalStorageBufferEXT,
                                                      forwardPointers[type.getReferentType()],
                                                      referentType);
            }
        }
        break;
    case glslang::EbtSampler:
        {
            const glslang::TSampler& sampler = type.getSampler();
            std::string debugName;

            if (sampler.isPureSampler()) {
                if (options.emitNonSemanticShaderDebugInfo) {
                    if (glslangIntermediate->getSource() == glslang::EShSourceGlsl) {
                        debugName = sampler.getString();
                    }
                    else {
                        debugName = "type.sampler";
                    }
                }
                spvType = builder.makeSamplerType(debugName.c_str());
            } else {
                // an image is present, make its type
                if (options.emitNonSemanticShaderDebugInfo) {
                    if (glslangIntermediate->getSource() == glslang::EShSourceGlsl) {
                        debugName = sampler.removeCombined().getString();
                    }
                    else {
                        switch (sampler.dim) {
                        case glslang::Esd1D:           debugName = "type.1d.image"; break;
                        case glslang::Esd2D:           debugName = "type.2d.image"; break;
                        case glslang::Esd3D:           debugName = "type.3d.image"; break;
                        case glslang::EsdCube:         debugName = "type.cube.image"; break;
                        default:                       debugName = "type.image"; break;
                        }
                    }
                }
                spvType = builder.makeImageType(getSampledType(sampler), TranslateDimensionality(sampler),
                                                sampler.isShadow(), sampler.isArrayed(), sampler.isMultiSample(),
                                                sampler.isImageClass() ? 2 : 1, TranslateImageFormat(type), debugName.c_str());
                if (sampler.isCombined() &&
                    (!sampler.isBuffer() || glslangIntermediate->getSpv().spv < glslang::EShTargetSpv_1_6)) {
                    // Already has both image and sampler, make the combined type. Only combine sampler to
                    // buffer if before SPIR-V 1.6.
                    if (options.emitNonSemanticShaderDebugInfo) {
                        if (glslangIntermediate->getSource() == glslang::EShSourceGlsl) {
                            debugName = sampler.getString();
                        }
                        else {
                            debugName = "type.sampled.image";
                        }
                    }
                    spvType = builder.makeSampledImageType(spvType, debugName.c_str());
                }
            }
        }
        break;
    case glslang::EbtStruct:
    case glslang::EbtBlock:
        {
            // If we've seen this struct type, return it
            const glslang::TTypeList* glslangMembers = type.getStruct();

            // Try to share structs for different layouts, but not yet for other
            // kinds of qualification (primarily not yet including interpolant qualification).
            if (! HasNonLayoutQualifiers(type, qualifier))
                spvType = structMap[explicitLayout][qualifier.layoutMatrix][glslangMembers];
            if (spvType != spv::NoResult)
                break;

            // else, we haven't seen it...
            if (type.getBasicType() == glslang::EbtBlock)
                memberRemapper[glslangTypeToIdMap[glslangMembers]].resize(glslangMembers->size());
            spvType = convertGlslangStructToSpvType(type, glslangMembers, explicitLayout, qualifier);
        }
        break;
    case glslang::EbtString:
        // no type used for OpString
        return 0;

    case glslang::EbtHitObjectNV: {
        builder.addExtension(spv::E_SPV_NV_shader_invocation_reorder);
        builder.addCapability(spv::Capability::ShaderInvocationReorderNV);
        spvType = builder.makeHitObjectNVType();
    }
    break;

    case glslang::EbtHitObjectEXT: {
        builder.addExtension(spv::E_SPV_EXT_shader_invocation_reorder);
        builder.addCapability(spv::Capability::ShaderInvocationReorderEXT);
        spvType = builder.makeHitObjectEXTType();
    }
    break;
    case glslang::EbtSpirvType: {
        // GL_EXT_spirv_intrinsics
        const auto& spirvType = type.getSpirvType();
        const auto& spirvInst = spirvType.spirvInst;

        std::vector<spv::IdImmediate> operands;
        for (const auto& typeParam : spirvType.typeParams) {
            if (typeParam.getAsConstant() != nullptr) {
                // Constant expression
                auto constant = typeParam.getAsConstant();
                if (constant->isLiteral()) {
                    if (constant->getBasicType() == glslang::EbtFloat) {
                        float floatValue = static_cast<float>(constant->getConstArray()[0].getDConst());
                        unsigned literal;
                        static_assert(sizeof(literal) == sizeof(floatValue), "sizeof(unsigned) != sizeof(float)");
                        memcpy(&literal, &floatValue, sizeof(literal));
                        operands.push_back({false, literal});
                    } else if (constant->getBasicType() == glslang::EbtInt) {
                        unsigned literal = constant->getConstArray()[0].getIConst();
                        operands.push_back({false, literal});
                    } else if (constant->getBasicType() == glslang::EbtUint) {
                        unsigned literal = constant->getConstArray()[0].getUConst();
                        operands.push_back({false, literal});
                    } else if (constant->getBasicType() == glslang::EbtBool) {
                        unsigned literal = constant->getConstArray()[0].getBConst();
                        operands.push_back({false, literal});
                    } else if (constant->getBasicType() == glslang::EbtString) {
                        auto str = constant->getConstArray()[0].getSConst()->c_str();
                        unsigned literal = 0;
                        char* literalPtr = reinterpret_cast<char*>(&literal);
                        unsigned charCount = 0;
                        char ch = 0;
                        do {
                            ch = *(str++);
                            *(literalPtr++) = ch;
                            ++charCount;
                            if (charCount == 4) {
                                operands.push_back({false, literal});
                                literalPtr = reinterpret_cast<char*>(&literal);
                                charCount = 0;
                            }
                        } while (ch != 0);

                        // Partial literal is padded with 0
                        if (charCount > 0) {
                            for (; charCount < 4; ++charCount)
                                *(literalPtr++) = 0;
                            operands.push_back({false, literal});
                        }
                    } else
                        assert(0); // Unexpected type
                } else
                    operands.push_back({true, createSpvConstant(*constant)});
            } else {
                // Type specifier
                assert(typeParam.getAsType() != nullptr);
                operands.push_back({true, convertGlslangToSpvType(*typeParam.getAsType())});
            }
        }

        assert(spirvInst.set == ""); // Currently, couldn't be extended instructions.
        spvType = builder.makeGenericType(static_cast<spv::Op>(spirvInst.id), operands);

        break;
    }
    case glslang::EbtTensorLayoutNV:
    {
        builder.addCapability(spv::Capability::TensorAddressingNV);
        builder.addExtension(spv::E_SPV_NV_tensor_addressing);

        std::vector<spv::IdImmediate> operands;
        for (uint32_t i = 0; i < 2; ++i) {
            operands.push_back({true, makeArraySizeId(*type.getTypeParameters()->arraySizes, i, true)});
        }
        spvType = builder.makeGenericType(spv::Op::OpTypeTensorLayoutNV, operands);
        break;
    }
    case glslang::EbtTensorViewNV:
    {
        builder.addCapability(spv::Capability::TensorAddressingNV);
        builder.addExtension(spv::E_SPV_NV_tensor_addressing);

        uint32_t dim = type.getTypeParameters()->arraySizes->getDimSize(0);
        assert(dim >= 1 && dim <= 5);
        std::vector<spv::IdImmediate> operands;
        for (uint32_t i = 0; i < dim + 2; ++i) {
            operands.push_back({true, makeArraySizeId(*type.getTypeParameters()->arraySizes, i, true, i==1)});
        }
        spvType = builder.makeGenericType(spv::Op::OpTypeTensorViewNV, operands);
        break;
    }
    default:
        assert(0);
        break;
    }

    if (type.isMatrix())
        spvType = builder.makeMatrixType(spvType, type.getMatrixCols(), type.getMatrixRows());
    else {
        // If this variable has a vector element count greater than 1, create a SPIR-V vector
        if (type.getVectorSize() > 1)
            spvType = builder.makeVectorType(spvType, type.getVectorSize());
    }

    if (type.isCoopMatNV()) {
        builder.addCapability(spv::Capability::CooperativeMatrixNV);
        builder.addExtension(spv::E_SPV_NV_cooperative_matrix);

        if (type.getBasicType() == glslang::EbtFloat16)
            builder.addCapability(spv::Capability::Float16);
        if (type.getBasicType() == glslang::EbtUint8 ||
            type.getBasicType() == glslang::EbtInt8) {
            builder.addCapability(spv::Capability::Int8);
        }

        spv::Id scope = makeArraySizeId(*type.getTypeParameters()->arraySizes, 1);
        spv::Id rows = makeArraySizeId(*type.getTypeParameters()->arraySizes, 2);
        spv::Id cols = makeArraySizeId(*type.getTypeParameters()->arraySizes, 3);

        spvType = builder.makeCooperativeMatrixTypeNV(spvType, scope, rows, cols);
    }

    if (type.isCoopMatKHR()) {
        builder.addCapability(spv::Capability::CooperativeMatrixKHR);
        builder.addExtension(spv::E_SPV_KHR_cooperative_matrix);

        if (type.getBasicType() == glslang::EbtBFloat16) {
            builder.addExtension(spv::E_SPV_KHR_bfloat16);
            builder.addCapability(spv::Capability::BFloat16CooperativeMatrixKHR);
        }

        if (type.getBasicType() == glslang::EbtFloatE5M2 || type.getBasicType() == glslang::EbtFloatE4M3) {
            builder.addExtension(spv::E_SPV_EXT_float8);
            builder.addCapability(spv::Capability::Float8CooperativeMatrixEXT);
        }

        if (type.getBasicType() == glslang::EbtFloat16)
            builder.addCapability(spv::Capability::Float16);
        if (type.getBasicType() == glslang::EbtUint8 || type.getBasicType() == glslang::EbtInt8) {
            builder.addCapability(spv::Capability::Int8);
        }

        spv::Id scope = makeArraySizeId(*type.getTypeParameters()->arraySizes, 0);
        spv::Id rows = makeArraySizeId(*type.getTypeParameters()->arraySizes, 1);
        spv::Id cols = makeArraySizeId(*type.getTypeParameters()->arraySizes, 2);
        spv::Id use = makeArraySizeId(*type.getTypeParameters()->arraySizes, 3, true);

        spvType = builder.makeCooperativeMatrixTypeKHR(spvType, scope, rows, cols, use);
    }
    else if (type.isTensorARM()) {
        builder.addCapability(spv::Capability::TensorsARM);
        builder.addExtension(spv::E_SPV_ARM_tensors);
        if (type.getBasicType() == glslang::EbtInt8 || type.getBasicType() == glslang::EbtUint8) {
            builder.addCapability(spv::Capability::Int8);
        } else if (type.getBasicType() == glslang::EbtInt16 ||
                   type.getBasicType() == glslang::EbtUint16) {
            builder.addCapability(spv::Capability::Int16);
        } else if (type.getBasicType() == glslang::EbtInt64 ||
                   type.getBasicType() == glslang::EbtUint64) {
            builder.addCapability(spv::Capability::Int64);
        } else if (type.getBasicType() == glslang::EbtFloat16) {
            builder.addCapability(spv::Capability::Float16);
        }

        spv::Id rank = makeArraySizeId(*type.getTypeParameters()->arraySizes, 0);

        spvType = builder.makeTensorTypeARM(spvType, rank);
    }

    if (type.isCoopVecNV()) {
        builder.addCapability(spv::Capability::CooperativeVectorNV);
        builder.addExtension(spv::E_SPV_NV_cooperative_vector);

        if (type.getBasicType() == glslang::EbtFloat16)
            builder.addCapability(spv::Capability::Float16);
        if (type.getBasicType() == glslang::EbtUint8 || type.getBasicType() == glslang::EbtInt8) {
            builder.addCapability(spv::Capability::Int8);
        }

        spv::Id components = makeArraySizeId(*type.getTypeParameters()->arraySizes, 0);

        spvType = builder.makeCooperativeVectorTypeNV(spvType, components);
    }

    if (type.isArray()) {
        int stride = 0;  // keep this 0 unless doing an explicit layout; 0 will mean no decoration, no stride

        // Do all but the outer dimension
        if (type.getArraySizes()->getNumDims() > 1) {
            // We need to decorate array strides for types needing explicit layout, except blocks.
            if (explicitLayout != glslang::ElpNone && type.getBasicType() != glslang::EbtBlock) {
                // Use a dummy glslang type for querying internal strides of
                // arrays of arrays, but using just a one-dimensional array.
                glslang::TType simpleArrayType(type, 0); // deference type of the array
                while (simpleArrayType.getArraySizes()->getNumDims() > 1)
                    simpleArrayType.getArraySizes()->dereference();

                // Will compute the higher-order strides here, rather than making a whole
                // pile of types and doing repetitive recursion on their contents.
                stride = getArrayStride(simpleArrayType, explicitLayout, qualifier.layoutMatrix);
            }

            // make the arrays
            for (int dim = type.getArraySizes()->getNumDims() - 1; dim > 0; --dim) {
                spvType = builder.makeArrayType(spvType, makeArraySizeId(*type.getArraySizes(), dim), stride);
                if (stride > 0)
                    builder.addDecoration(spvType, spv::Decoration::ArrayStride, stride);
                stride *= type.getArraySizes()->getDimSize(dim);
            }
        } else {
            // single-dimensional array, and don't yet have stride

            // We need to decorate array strides for types needing explicit layout, except blocks.
            if (explicitLayout != glslang::ElpNone && type.getBasicType() != glslang::EbtBlock)
                stride = getArrayStride(type, explicitLayout, qualifier.layoutMatrix);
        }

        // Do the outer dimension, which might not be known for a runtime-sized array.
        // (Unsized arrays that survive through linking will be runtime-sized arrays)
        if (type.isSizedArray())
            spvType = builder.makeArrayType(spvType, makeArraySizeId(*type.getArraySizes(), 0), stride);
        else {
            // If we see an runtime array in a buffer_reference, it is not a descriptor
            if (!lastBufferBlockMember && type.getBasicType() != glslang::EbtReference) {
                builder.addIncorporatedExtension("SPV_EXT_descriptor_indexing", spv::Spv_1_5);
                builder.addCapability(spv::Capability::RuntimeDescriptorArrayEXT);
            }
            spvType = builder.makeRuntimeArray(spvType);
        }
        if (stride > 0)
            builder.addDecoration(spvType, spv::Decoration::ArrayStride, stride);
    }

    return spvType;
}

// Apply SPIR-V decorations to the SPIR-V object (provided by SPIR-V ID). If member index is provided, the
// decorations are applied to this member.
void TGlslangToSpvTraverser::applySpirvDecorate(const glslang::TType& type, spv::Id id, std::optional<int> member)
{
    assert(type.getQualifier().hasSpirvDecorate());

    const glslang::TSpirvDecorate& spirvDecorate = type.getQualifier().getSpirvDecorate();

    // Add spirv_decorate
    for (auto& decorate : spirvDecorate.decorates) {
        if (!decorate.second.empty()) {
            std::vector<unsigned> literals;
            TranslateLiterals(decorate.second, literals);
            if (member.has_value())
                builder.addMemberDecoration(id, *member, static_cast<spv::Decoration>(decorate.first), literals);
            else
                builder.addDecoration(id, static_cast<spv::Decoration>(decorate.first), literals);
        } else {
            if (member.has_value())
                builder.addMemberDecoration(id, *member, static_cast<spv::Decoration>(decorate.first));
            else
                builder.addDecoration(id, static_cast<spv::Decoration>(decorate.first));
        }
    }

    // Add spirv_decorate_id
    if (member.has_value()) {
        // spirv_decorate_id not applied to members
        assert(spirvDecorate.decorateIds.empty());
    } else {
        for (auto& decorateId : spirvDecorate.decorateIds) {
            std::vector<spv::Id> operandIds;
            assert(!decorateId.second.empty());
            for (auto extraOperand : decorateId.second) {
                if (extraOperand->getQualifier().isFrontEndConstant())
                    operandIds.push_back(createSpvConstant(*extraOperand));
                else
                    operandIds.push_back(getSymbolId(extraOperand->getAsSymbolNode()));
            }
            builder.addDecorationId(id, static_cast<spv::Decoration>(decorateId.first), operandIds);
        }
    }

    // Add spirv_decorate_string
    for (auto& decorateString : spirvDecorate.decorateStrings) {
        std::vector<const char*> strings;
        assert(!decorateString.second.empty());
        for (auto extraOperand : decorateString.second) {
            const char* string = extraOperand->getConstArray()[0].getSConst()->c_str();
            strings.push_back(string);
        }
        if (member.has_value())
            builder.addMemberDecoration(id, *member, static_cast<spv::Decoration>(decorateString.first), strings);
        else
            builder.addDecoration(id, static_cast<spv::Decoration>(decorateString.first), strings);
    }
}

// TODO: this functionality should exist at a higher level, in creating the AST
//
// Identify interface members that don't have their required extension turned on.
//
bool TGlslangToSpvTraverser::filterMember(const glslang::TType& member)
{
    auto& extensions = glslangIntermediate->getRequestedExtensions();

    if (member.getFieldName() == "gl_SecondaryViewportMaskNV" &&
        extensions.find("GL_NV_stereo_view_rendering") == extensions.end())
        return true;
    if (member.getFieldName() == "gl_SecondaryPositionNV" &&
        extensions.find("GL_NV_stereo_view_rendering") == extensions.end())
        return true;

    if (glslangIntermediate->getStage() == EShLangMesh) {
        if (member.getFieldName() == "gl_PrimitiveShadingRateEXT" &&
            extensions.find("GL_EXT_fragment_shading_rate") == extensions.end())
            return true;
    }

    if (glslangIntermediate->getStage() != EShLangMesh) {
        if (member.getFieldName() == "gl_ViewportMask" &&
            extensions.find("GL_NV_viewport_array2") == extensions.end())
            return true;
        if (member.getFieldName() == "gl_PositionPerViewNV" &&
            extensions.find("GL_NVX_multiview_per_view_attributes") == extensions.end())
            return true;
        if (member.getFieldName() == "gl_ViewportMaskPerViewNV" &&
            extensions.find("GL_NVX_multiview_per_view_attributes") == extensions.end())
            return true;
    }

    return false;
}

// Do full recursive conversion of a glslang structure (or block) type to a SPIR-V Id.
// explicitLayout can be kept the same throughout the hierarchical recursive walk.
// Mutually recursive with convertGlslangToSpvType().
spv::Id TGlslangToSpvTraverser::convertGlslangStructToSpvType(const glslang::TType& type,
                                                              const glslang::TTypeList* glslangMembers,
                                                              glslang::TLayoutPacking explicitLayout,
                                                              const glslang::TQualifier& qualifier)
{
    // Create a vector of struct types for SPIR-V to consume
    std::vector<spv::Id> spvMembers;
    int memberDelta = 0;  // how much the member's index changes from glslang to SPIR-V, normally 0,
                          // except sometimes for blocks
    std::vector<std::pair<glslang::TType*, glslang::TQualifier> > deferredForwardPointers;
    std::vector<spv::StructMemberDebugInfo> memberDebugInfo;
    for (int i = 0; i < (int)glslangMembers->size(); i++) {
        auto& glslangMember = (*glslangMembers)[i];
        if (glslangMember.type->hiddenMember()) {
            ++memberDelta;
            if (type.getBasicType() == glslang::EbtBlock)
                memberRemapper[glslangTypeToIdMap[glslangMembers]][i] = -1;
        } else {
            if (type.getBasicType() == glslang::EbtBlock) {
                if (filterMember(*glslangMember.type)) {
                    memberDelta++;
                    memberRemapper[glslangTypeToIdMap[glslangMembers]][i] = -1;
                    continue;
                }
                memberRemapper[glslangTypeToIdMap[glslangMembers]][i] = i - memberDelta;
            }
            // modify just this child's view of the qualifier
            glslang::TQualifier memberQualifier = glslangMember.type->getQualifier();
            InheritQualifiers(memberQualifier, qualifier);

            // manually inherit location
            if (! memberQualifier.hasLocation() && qualifier.hasLocation())
                memberQualifier.layoutLocation = qualifier.layoutLocation;

            // recurse
            bool lastBufferBlockMember = qualifier.storage == glslang::EvqBuffer &&
                                         i == (int)glslangMembers->size() - 1;

            // Make forward pointers for any pointer members.
            if (glslangMember.type->isReference() &&
                forwardPointers.find(glslangMember.type->getReferentType()) == forwardPointers.end()) {
                deferredForwardPointers.push_back(std::make_pair(glslangMember.type, memberQualifier));
            }

            // Create the member type.
            auto const spvMember = convertGlslangToSpvType(*glslangMember.type, explicitLayout, memberQualifier, lastBufferBlockMember,
                glslangMember.type->isReference());
            spvMembers.push_back(spvMember);

            // Update the builder with the type's location so that we can create debug types for the structure members.
            // There doesn't exist a "clean" entry point for this information to be passed along to the builder so, for now,
            // it is stored in the builder and consumed during the construction of composite debug types.
            // TODO: This probably warrants further investigation. This approach was decided to be the least ugly of the
            // quick and dirty approaches that were tried.
            // Advantages of this approach:
            //  + Relatively clean. No direct calls into debug type system.
            //  + Handles nested recursive structures.
            // Disadvantages of this approach:
            //  + Not as clean as desired. Traverser queries/sets persistent state. This is fragile.
            //  + Table lookup during creation of composite debug types. This really shouldn't be necessary.
            if(options.emitNonSemanticShaderDebugInfo) {
                spv::StructMemberDebugInfo debugInfo{};
                debugInfo.name = glslangMember.type->getFieldName();
                debugInfo.line = glslangMember.loc.line;
                debugInfo.column = glslangMember.loc.column;

                // Per the GLSL spec, bool variables inside of a uniform or buffer block are generated as uint.
                // But for debug info, we want to represent them as bool because that is the original type in
                // the source code. The bool type can be nested within a vector or a multidimensional array,
                // so we must construct the chain of types up from the scalar bool.
                if (glslangIntermediate->getSource() == glslang::EShSourceGlsl && explicitLayout != glslang::ElpNone &&
                    glslangMember.type->getBasicType() == glslang::EbtBool) {
                    auto typeId = builder.makeBoolType();
                    if (glslangMember.type->isVector()) {
                        typeId = builder.makeVectorType(typeId, glslangMember.type->getVectorSize());
                    }
                    if (glslangMember.type->isArray()) {
                        const auto* arraySizes = glslangMember.type->getArraySizes();
                        int dims = arraySizes->getNumDims();
                        for (int i = dims - 1; i >= 0; --i) {
                            spv::Id size = builder.makeIntConstant(arraySizes->getDimSize(i));
                            typeId = builder.makeArrayType(typeId, size, 0);
                        }
                    }
                    debugInfo.debugTypeOverride = builder.getDebugType(typeId);
                }

                memberDebugInfo.push_back(debugInfo);
            }
        }
    }

    // Make the SPIR-V type
    spv::Id spvType = builder.makeStructType(spvMembers, memberDebugInfo, type.getTypeName().c_str(), false);
    if (! HasNonLayoutQualifiers(type, qualifier))
        structMap[explicitLayout][qualifier.layoutMatrix][glslangMembers] = spvType;

    // Decorate it
    decorateStructType(type, glslangMembers, explicitLayout, qualifier, spvType, spvMembers);

    for (int i = 0; i < (int)deferredForwardPointers.size(); ++i) {
        auto it = deferredForwardPointers[i];
        convertGlslangToSpvType(*it.first, explicitLayout, it.second, false);
    }

    return spvType;
}

void TGlslangToSpvTraverser::decorateStructType(const glslang::TType& type,
                                                const glslang::TTypeList* glslangMembers,
                                                glslang::TLayoutPacking explicitLayout,
                                                const glslang::TQualifier& qualifier,
                                                spv::Id spvType,
                                                const std::vector<spv::Id>& spvMembers)
{
    // Name and decorate the non-hidden members
    int offset = -1;
    bool memberLocationInvalid = type.isArrayOfArrays() ||
        (type.isArray() && (type.getQualifier().isArrayedIo(glslangIntermediate->getStage()) == false));
    for (int i = 0; i < (int)glslangMembers->size(); i++) {
        glslang::TType& glslangMember = *(*glslangMembers)[i].type;
        int member = i;
        if (type.getBasicType() == glslang::EbtBlock) {
            member = memberRemapper[glslangTypeToIdMap[glslangMembers]][i];
            if (filterMember(glslangMember))
                continue;
        }

        // modify just this child's view of the qualifier
        glslang::TQualifier memberQualifier = glslangMember.getQualifier();
        InheritQualifiers(memberQualifier, qualifier);

        // using -1 above to indicate a hidden member
        if (member < 0)
            continue;

        builder.addMemberName(spvType, member, glslangMember.getFieldName().c_str());
        builder.addMemberDecoration(spvType, member,
                                    TranslateLayoutDecoration(glslangMember, memberQualifier.layoutMatrix));
        builder.addMemberDecoration(spvType, member, TranslatePrecisionDecoration(glslangMember));
        // Add interpolation and auxiliary storage decorations only to
        // top-level members of Input and Output storage classes
        if (type.getQualifier().storage == glslang::EvqVaryingIn ||
            type.getQualifier().storage == glslang::EvqVaryingOut) {
            if (type.getBasicType() == glslang::EbtBlock ||
                glslangIntermediate->getSource() == glslang::EShSourceHlsl) {
                builder.addMemberDecoration(spvType, member, TranslateInterpolationDecoration(memberQualifier));
                builder.addMemberDecoration(spvType, member, TranslateAuxiliaryStorageDecoration(memberQualifier));
                addMeshNVDecoration(spvType, member, memberQualifier);
            }
        }
        builder.addMemberDecoration(spvType, member, TranslateInvariantDecoration(memberQualifier));

        if (type.getBasicType() == glslang::EbtBlock &&
            qualifier.storage == glslang::EvqBuffer) {
            // Add memory decorations only to top-level members of shader storage block
            std::vector<spv::Decoration> memory;
            TranslateMemoryDecoration(memberQualifier, memory, glslangIntermediate->usingVulkanMemoryModel());
            for (unsigned int i = 0; i < memory.size(); ++i)
                builder.addMemberDecoration(spvType, member, memory[i]);
        }

        // Location assignment was already completed correctly by the front end,
        // just track whether a member needs to be decorated.
        // Ignore member locations if the container is an array, as that's
        // ill-specified and decisions have been made to not allow this.
        if (!memberLocationInvalid && memberQualifier.hasLocation())
            builder.addMemberDecoration(spvType, member, spv::Decoration::Location, memberQualifier.layoutLocation);

        // component, XFB, others
        if (glslangMember.getQualifier().hasComponent())
            builder.addMemberDecoration(spvType, member, spv::Decoration::Component,
                                        glslangMember.getQualifier().layoutComponent);
        if (glslangMember.getQualifier().hasXfbOffset())
            builder.addMemberDecoration(spvType, member, spv::Decoration::Offset,
                                        glslangMember.getQualifier().layoutXfbOffset);
        else if (explicitLayout != glslang::ElpNone) {
            // figure out what to do with offset, which is accumulating
            int nextOffset;
            updateMemberOffset(type, glslangMember, offset, nextOffset, explicitLayout, memberQualifier.layoutMatrix);
            if (offset >= 0)
                builder.addMemberDecoration(spvType, member, spv::Decoration::Offset, offset);
            offset = nextOffset;
        }

        if (glslangMember.isMatrix() && explicitLayout != glslang::ElpNone)
            builder.addMemberDecoration(spvType, member, spv::Decoration::MatrixStride,
                                        getMatrixStride(glslangMember, explicitLayout, memberQualifier.layoutMatrix));

        // built-in variable decorations
        spv::BuiltIn builtIn = TranslateBuiltInDecoration(glslangMember.getQualifier().builtIn, true);
        if (builtIn != spv::BuiltIn::Max)
            builder.addMemberDecoration(spvType, member, spv::Decoration::BuiltIn, (int)builtIn);

        // nonuniform
        builder.addMemberDecoration(spvType, member, TranslateNonUniformDecoration(glslangMember.getQualifier()));

        if (glslangIntermediate->getHlslFunctionality1() && memberQualifier.semanticName != nullptr) {
            builder.addExtension("SPV_GOOGLE_hlsl_functionality1");
            builder.addMemberDecoration(spvType, member, spv::Decoration::HlslSemanticGOOGLE,
                                        memberQualifier.semanticName);
        }

        if (builtIn == spv::BuiltIn::Layer) {
            // SPV_NV_viewport_array2 extension
            if (glslangMember.getQualifier().layoutViewportRelative){
                builder.addMemberDecoration(spvType, member, spv::Decoration::ViewportRelativeNV);
                builder.addCapability(spv::Capability::ShaderViewportMaskNV);
                builder.addExtension(spv::E_SPV_NV_viewport_array2);
            }
            if (glslangMember.getQualifier().layoutSecondaryViewportRelativeOffset != -2048){
                builder.addMemberDecoration(spvType, member,
                                            spv::Decoration::SecondaryViewportRelativeNV,
                                            glslangMember.getQualifier().layoutSecondaryViewportRelativeOffset);
                builder.addCapability(spv::Capability::ShaderStereoViewNV);
                builder.addExtension(spv::E_SPV_NV_stereo_view_rendering);
            }
        }
        if (glslangMember.getQualifier().layoutPassthrough) {
            builder.addMemberDecoration(spvType, member, spv::Decoration::PassthroughNV);
            builder.addCapability(spv::Capability::GeometryShaderPassthroughNV);
            builder.addExtension(spv::E_SPV_NV_geometry_shader_passthrough);
        }

        // Add SPIR-V decorations (GL_EXT_spirv_intrinsics)
        if (glslangMember.getQualifier().hasSpirvDecorate())
            applySpirvDecorate(glslangMember, spvType, member);
    }

    // Decorate the structure
    builder.addDecoration(spvType, TranslateLayoutDecoration(type, qualifier.layoutMatrix));
    const auto basicType = type.getBasicType();
    const auto typeStorageQualifier = type.getQualifier().storage;
    if (basicType == glslang::EbtBlock) {
        builder.addDecoration(spvType, TranslateBlockDecoration(typeStorageQualifier, glslangIntermediate->usingStorageBuffer()));
    } else if (basicType == glslang::EbtStruct && glslangIntermediate->getSpv().vulkan > 0) {
        const auto hasRuntimeArray = !spvMembers.empty() && builder.getOpCode(spvMembers.back()) == spv::Op::OpTypeRuntimeArray;
        if (hasRuntimeArray) {
            builder.addDecoration(spvType, TranslateBlockDecoration(typeStorageQualifier, glslangIntermediate->usingStorageBuffer()));
        }
    }

    if (qualifier.hasHitObjectShaderRecordNV())
        builder.addDecoration(spvType, spv::Decoration::HitObjectShaderRecordBufferNV);
    if (qualifier.hasHitObjectShaderRecordEXT())
        builder.addDecoration(spvType, spv::Decoration::HitObjectShaderRecordBufferEXT);
}

// Turn the expression forming the array size into an id.
// This is not quite trivial, because of specialization constants.
// Sometimes, a raw constant is turned into an Id, and sometimes
// a specialization constant expression is.
spv::Id TGlslangToSpvTraverser::makeArraySizeId(const glslang::TArraySizes& arraySizes, int dim, bool allowZero, bool boolType)
{
    // First, see if this is sized with a node, meaning a specialization constant:
    glslang::TIntermTyped* specNode = arraySizes.getDimNode(dim);
    if (specNode != nullptr) {
        builder.clearAccessChain();
        SpecConstantOpModeGuard spec_constant_op_mode_setter(&builder);
        spec_constant_op_mode_setter.turnOnSpecConstantOpMode();
        specNode->traverse(this);
        return accessChainLoad(specNode->getAsTyped()->getType());
    }

    // Otherwise, need a compile-time (front end) size, get it:
    int size = arraySizes.getDimSize(dim);

    if (!allowZero)
        assert(size > 0);

    if (boolType) {
        return builder.makeBoolConstant(size);
    } else {
        return builder.makeUintConstant(size);
    }
}

// Wrap the builder's accessChainLoad to:
//  - localize handling of RelaxedPrecision
//  - use the SPIR-V inferred type instead of another conversion of the glslang type
//    (avoids unnecessary work and possible type punning for structures)
//  - do conversion of concrete to abstract type
spv::Id TGlslangToSpvTraverser::accessChainLoad(const glslang::TType& type)
{
    spv::Id nominalTypeId = builder.accessChainGetInferredType();

    spv::Builder::AccessChain::CoherentFlags coherentFlags = builder.getAccessChain().coherentFlags;
    coherentFlags |= TranslateCoherent(type);

    spv::MemoryAccessMask accessMask = spv::MemoryAccessMask(TranslateMemoryAccess(coherentFlags) & ~spv::MemoryAccessMask::MakePointerAvailableKHR);
    // If the value being loaded is HelperInvocation, SPIR-V 1.6 is being generated (so that
    // SPV_EXT_demote_to_helper_invocation is in core) and the memory model is in use, add
    // the Volatile MemoryAccess semantic.
    if (type.getQualifier().builtIn == glslang::EbvHelperInvocation &&
        glslangIntermediate->usingVulkanMemoryModel() &&
        glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_6) {
        accessMask = spv::MemoryAccessMask(accessMask | spv::MemoryAccessMask::Volatile);
    }

    unsigned int alignment = builder.getAccessChain().alignment;
    alignment |= type.getBufferReferenceAlignment();

    spv::Id loadedId = builder.accessChainLoad(TranslatePrecisionDecoration(type),
        TranslateNonUniformDecoration(builder.getAccessChain().coherentFlags),
        TranslateNonUniformDecoration(type.getQualifier()),
        nominalTypeId,
        accessMask,
        TranslateMemoryScope(coherentFlags),
        alignment);

    // Need to convert to abstract types when necessary
    if (type.getBasicType() == glslang::EbtBool) {
        loadedId = convertLoadedBoolInUniformToUint(type, nominalTypeId, loadedId);
    }

    return loadedId;
}

// Wrap the builder's accessChainStore to:
//  - do conversion of concrete to abstract type
//
// Implicitly uses the existing builder.accessChain as the storage target.
void TGlslangToSpvTraverser::accessChainStore(const glslang::TType& type, spv::Id rvalue)
{
    // Need to convert to abstract types when necessary
    if (type.getBasicType() == glslang::EbtBool) {
        spv::Id nominalTypeId = builder.accessChainGetInferredType();

        if (builder.isScalarType(nominalTypeId)) {
            // Conversion for bool
            spv::Id boolType = builder.makeBoolType();
            if (nominalTypeId != boolType) {
                // keep these outside arguments, for determinant order-of-evaluation
                spv::Id one = builder.makeUintConstant(1);
                spv::Id zero = builder.makeUintConstant(0);
                rvalue = builder.createTriOp(spv::Op::OpSelect, nominalTypeId, rvalue, one, zero);
            } else if (builder.getTypeId(rvalue) != boolType)
                rvalue = builder.createBinOp(spv::Op::OpINotEqual, boolType, rvalue, builder.makeUintConstant(0));
        } else if (builder.isVectorType(nominalTypeId)) {
            // Conversion for bvec
            int vecSize = builder.getNumTypeComponents(nominalTypeId);
            spv::Id bvecType = builder.makeVectorType(builder.makeBoolType(), vecSize);
            if (nominalTypeId != bvecType) {
                // keep these outside arguments, for determinant order-of-evaluation
                spv::Id one = makeSmearedConstant(builder.makeUintConstant(1), vecSize);
                spv::Id zero = makeSmearedConstant(builder.makeUintConstant(0), vecSize);
                rvalue = builder.createTriOp(spv::Op::OpSelect, nominalTypeId, rvalue, one, zero);
            } else if (builder.getTypeId(rvalue) != bvecType)
                rvalue = builder.createBinOp(spv::Op::OpINotEqual, bvecType, rvalue,
                                             makeSmearedConstant(builder.makeUintConstant(0), vecSize));
        }
    }

    spv::Builder::AccessChain::CoherentFlags coherentFlags = builder.getAccessChain().coherentFlags;
    coherentFlags |= TranslateCoherent(type);

    unsigned int alignment = builder.getAccessChain().alignment;
    alignment |= type.getBufferReferenceAlignment();

    builder.accessChainStore(rvalue, TranslateNonUniformDecoration(builder.getAccessChain().coherentFlags),
                             spv::MemoryAccessMask(TranslateMemoryAccess(coherentFlags) &
                                ~spv::MemoryAccessMask::MakePointerVisibleKHR),
                             TranslateMemoryScope(coherentFlags), alignment);
}

// For storing when types match at the glslang level, but not might match at the
// SPIR-V level.
//
// This especially happens when a single glslang type expands to multiple
// SPIR-V types, like a struct that is used in a member-undecorated way as well
// as in a member-decorated way.
//
// NOTE: This function can handle any store request; if it's not special it
// simplifies to a simple OpStore.
//
// Implicitly uses the existing builder.accessChain as the storage target.
void TGlslangToSpvTraverser::multiTypeStore(const glslang::TType& type, spv::Id rValue)
{
    // we only do the complex path here if it's an aggregate
    if (! type.isStruct() && ! type.isArray()) {
        accessChainStore(type, rValue);
        return;
    }

    // and, it has to be a case of type aliasing
    spv::Id rType = builder.getTypeId(rValue);
    spv::Id lValue = builder.accessChainGetLValue();
    spv::Id lType = builder.getContainedTypeId(builder.getTypeId(lValue));
    if (lType == rType) {
        accessChainStore(type, rValue);
        return;
    }

    // Recursively (as needed) copy an aggregate type to a different aggregate type,
    // where the two types were the same type in GLSL. This requires member
    // by member copy, recursively.

    // SPIR-V 1.4 added an instruction to do help do this.
    if (glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_4) {
        // However, bool in uniform space is changed to int, so
        // OpCopyLogical does not work for that.
        // TODO: It would be more robust to do a full recursive verification of the types satisfying SPIR-V rules.
        bool rBool = builder.containsType(builder.getTypeId(rValue), spv::Op::OpTypeBool, 0);
        bool lBool = builder.containsType(lType, spv::Op::OpTypeBool, 0);
        if (lBool == rBool) {
            spv::Id logicalCopy = builder.createUnaryOp(spv::Op::OpCopyLogical, lType, rValue);
            accessChainStore(type, logicalCopy);
            return;
        }
    }

    // If an array, copy element by element.
    if (type.isArray()) {
        glslang::TType glslangElementType(type, 0);
        spv::Id elementRType = builder.getContainedTypeId(rType);
        for (int index = 0; index < type.getOuterArraySize(); ++index) {
            // get the source member
            spv::Id elementRValue = builder.createCompositeExtract(rValue, elementRType, index);

            // set up the target storage
            builder.clearAccessChain();
            builder.setAccessChainLValue(lValue);
            builder.accessChainPush(builder.makeIntConstant(index), TranslateCoherent(type),
                type.getBufferReferenceAlignment());

            // store the member
            multiTypeStore(glslangElementType, elementRValue);
        }
    } else {
        assert(type.isStruct());

        // loop over structure members
        const glslang::TTypeList& members = *type.getStruct();
        for (int m = 0; m < (int)members.size(); ++m) {
            const glslang::TType& glslangMemberType = *members[m].type;

            // get the source member
            spv::Id memberRType = builder.getContainedTypeId(rType, m);
            spv::Id memberRValue = builder.createCompositeExtract(rValue, memberRType, m);

            // set up the target storage
            builder.clearAccessChain();
            builder.setAccessChainLValue(lValue);
            builder.accessChainPush(builder.makeIntConstant(m), TranslateCoherent(type),
                type.getBufferReferenceAlignment());

            // store the member
            multiTypeStore(glslangMemberType, memberRValue);
        }
    }
}

// Decide whether or not this type should be
// decorated with offsets and strides, and if so
// whether std140 or std430 rules should be applied.
glslang::TLayoutPacking TGlslangToSpvTraverser::getExplicitLayout(const glslang::TType& type) const
{
    // has to be a block
    if (type.getBasicType() != glslang::EbtBlock)
        return glslang::ElpNone;

    // has to be a uniform or buffer block or task in/out blocks
    if (type.getQualifier().storage != glslang::EvqUniform &&
        type.getQualifier().storage != glslang::EvqBuffer &&
        type.getQualifier().storage != glslang::EvqShared &&
        !type.getQualifier().isTaskMemory())
        return glslang::ElpNone;

    // return the layout to use
    switch (type.getQualifier().layoutPacking) {
    case glslang::ElpStd140:
    case glslang::ElpStd430:
    case glslang::ElpScalar:
        return type.getQualifier().layoutPacking;
    default:
        return glslang::ElpNone;
    }
}

// Given an array type, returns the integer stride required for that array
int TGlslangToSpvTraverser::getArrayStride(const glslang::TType& arrayType, glslang::TLayoutPacking explicitLayout,
    glslang::TLayoutMatrix matrixLayout)
{
    int size;
    int stride;
    glslangIntermediate->getMemberAlignment(arrayType, size, stride, explicitLayout,
        matrixLayout == glslang::ElmRowMajor);

    return stride;
}

// Given a matrix type, or array (of array) of matrixes type, returns the integer stride required for that matrix
// when used as a member of an interface block
int TGlslangToSpvTraverser::getMatrixStride(const glslang::TType& matrixType, glslang::TLayoutPacking explicitLayout,
    glslang::TLayoutMatrix matrixLayout)
{
    glslang::TType elementType;
    elementType.shallowCopy(matrixType);
    elementType.clearArraySizes();

    int size;
    int stride;
    glslangIntermediate->getMemberAlignment(elementType, size, stride, explicitLayout,
        matrixLayout == glslang::ElmRowMajor);

    return stride;
}

// Given a member type of a struct, realign the current offset for it, and compute
// the next (not yet aligned) offset for the next member, which will get aligned
// on the next call.
// 'currentOffset' should be passed in already initialized, ready to modify, and reflecting
// the migration of data from nextOffset -> currentOffset.  It should be -1 on the first call.
// -1 means a non-forced member offset (no decoration needed).
void TGlslangToSpvTraverser::updateMemberOffset(const glslang::TType& structType, const glslang::TType& memberType,
    int& currentOffset, int& nextOffset, glslang::TLayoutPacking explicitLayout, glslang::TLayoutMatrix matrixLayout)
{
    // this will get a positive value when deemed necessary
    nextOffset = -1;

    // override anything in currentOffset with user-set offset
    if (memberType.getQualifier().hasOffset())
        currentOffset = memberType.getQualifier().layoutOffset;

    // It could be that current linker usage in glslang updated all the layoutOffset,
    // in which case the following code does not matter.  But, that's not quite right
    // once cross-compilation unit GLSL validation is done, as the original user
    // settings are needed in layoutOffset, and then the following will come into play.

    if (explicitLayout == glslang::ElpNone) {
        if (! memberType.getQualifier().hasOffset())
            currentOffset = -1;

        return;
    }

    // Getting this far means we need explicit offsets
    if (currentOffset < 0)
        currentOffset = 0;

    // Now, currentOffset is valid (either 0, or from a previous nextOffset),
    // but possibly not yet correctly aligned.

    int memberSize;
    int dummyStride;
    int memberAlignment = glslangIntermediate->getMemberAlignment(memberType, memberSize, dummyStride, explicitLayout,
        matrixLayout == glslang::ElmRowMajor);

    bool isVectorLike = memberType.isVector();
    if (memberType.isMatrix()) {
        if (matrixLayout == glslang::ElmRowMajor)
            isVectorLike = memberType.getMatrixRows() == 1;
        else
            isVectorLike = memberType.getMatrixCols() == 1;
    }

    // Adjust alignment for HLSL rules
    // TODO: make this consistent in early phases of code:
    //       adjusting this late means inconsistencies with earlier code, which for reflection is an issue
    // Until reflection is brought in sync with these adjustments, don't apply to $Global,
    // which is the most likely to rely on reflection, and least likely to rely implicit layouts
    if (glslangIntermediate->usingHlslOffsets() &&
        ! memberType.isStruct() && structType.getTypeName().compare("$Global") != 0) {
        int componentSize;
        int componentAlignment = glslangIntermediate->getBaseAlignmentScalar(memberType, componentSize);
        if (! memberType.isArray() && isVectorLike && componentAlignment <= 4)
            memberAlignment = componentAlignment;

        // Don't add unnecessary padding after this member
        // (undo std140 bumping size to a mutliple of vec4)
        if (explicitLayout == glslang::ElpStd140) {
            if (memberType.isMatrix()) {
                if (matrixLayout == glslang::ElmRowMajor)
                    memberSize -= componentSize * (4 - memberType.getMatrixCols());
                else
                    memberSize -= componentSize * (4 - memberType.getMatrixRows());
            } else if (memberType.isArray())
                memberSize -= componentSize * (4 - memberType.getVectorSize());
        }
    }

    // Bump up to member alignment
    glslang::RoundToPow2(currentOffset, memberAlignment);

    // Bump up to vec4 if there is a bad straddle
    if (explicitLayout != glslang::ElpScalar && glslangIntermediate->improperStraddle(memberType, memberSize,
        currentOffset, isVectorLike))
        glslang::RoundToPow2(currentOffset, 16);

    nextOffset = currentOffset + memberSize;
}

void TGlslangToSpvTraverser::declareUseOfStructMember(const glslang::TTypeList& members, int glslangMember)
{
    const glslang::TBuiltInVariable glslangBuiltIn = members[glslangMember].type->getQualifier().builtIn;
    switch (glslangBuiltIn)
    {
    case glslang::EbvPointSize:
    case glslang::EbvClipDistance:
    case glslang::EbvCullDistance:
    case glslang::EbvViewportMaskNV:
    case glslang::EbvSecondaryPositionNV:
    case glslang::EbvSecondaryViewportMaskNV:
    case glslang::EbvPositionPerViewNV:
    case glslang::EbvViewportMaskPerViewNV:
    case glslang::EbvTaskCountNV:
    case glslang::EbvPrimitiveCountNV:
    case glslang::EbvPrimitiveIndicesNV:
    case glslang::EbvClipDistancePerViewNV:
    case glslang::EbvCullDistancePerViewNV:
    case glslang::EbvLayerPerViewNV:
    case glslang::EbvMeshViewCountNV:
    case glslang::EbvMeshViewIndicesNV:
        // Generate the associated capability.  Delegate to TranslateBuiltInDecoration.
        // Alternately, we could just call this for any glslang built-in, since the
        // capability already guards against duplicates.
        TranslateBuiltInDecoration(glslangBuiltIn, false);
        break;
    default:
        // Capabilities were already generated when the struct was declared.
        break;
    }
}

bool TGlslangToSpvTraverser::isShaderEntryPoint(const glslang::TIntermAggregate* node)
{
    return node->getName().compare(glslangIntermediate->getEntryPointMangledName().c_str()) == 0;
}

// Does parameter need a place to keep writes, separate from the original?
// Assumes called after originalParam(), which filters out block/buffer/opaque-based
// qualifiers such that we should have only in/out/inout/constreadonly here.
bool TGlslangToSpvTraverser::writableParam(glslang::TStorageQualifier qualifier) const
{
    assert(qualifier == glslang::EvqIn ||
           qualifier == glslang::EvqOut ||
           qualifier == glslang::EvqInOut ||
           qualifier == glslang::EvqUniform ||
           qualifier == glslang::EvqConstReadOnly);
    return qualifier != glslang::EvqConstReadOnly &&
           qualifier != glslang::EvqUniform;
}

// Is parameter pass-by-original?
bool TGlslangToSpvTraverser::originalParam(glslang::TStorageQualifier qualifier, const glslang::TType& paramType,
                                           bool implicitThisParam)
{
    if (implicitThisParam)                                                                     // implicit this
        return true;
    if (glslangIntermediate->getSource() == glslang::EShSourceHlsl)
        return paramType.getBasicType() == glslang::EbtBlock;
    return (paramType.containsOpaque() && !glslangIntermediate->getBindlessMode()) ||       // sampler, etc.
           paramType.getQualifier().isSpirvByReference() ||                                    // spirv_by_reference
           (paramType.getBasicType() == glslang::EbtBlock && qualifier == glslang::EvqBuffer); // SSBO
}

// Make all the functions, skeletally, without actually visiting their bodies.
void TGlslangToSpvTraverser::makeFunctions(const glslang::TIntermSequence& glslFunctions)
{
    const auto getParamDecorations = [&](std::vector<spv::Decoration>& decorations, const glslang::TType& type,
        bool useVulkanMemoryModel) {
        spv::Decoration paramPrecision = TranslatePrecisionDecoration(type);
        if (paramPrecision != spv::NoPrecision)
            decorations.push_back(paramPrecision);
        TranslateMemoryDecoration(type.getQualifier(), decorations, useVulkanMemoryModel);
        if (type.isReference()) {
            // Original and non-writable params pass the pointer directly and
            // use restrict/aliased, others are stored to a pointer in Function
            // memory and use RestrictPointer/AliasedPointer.
            if (originalParam(type.getQualifier().storage, type, false) ||
                !writableParam(type.getQualifier().storage)) {
                // TranslateMemoryDecoration added Restrict decoration already.
                if (!type.getQualifier().isRestrict()) {
                    decorations.push_back(spv::Decoration::Aliased);
                }
            } else {
                decorations.push_back(type.getQualifier().isRestrict() ? spv::Decoration::RestrictPointerEXT :
                                                                         spv::Decoration::AliasedPointerEXT);
            }
        }
    };

    for (int f = 0; f < (int)glslFunctions.size(); ++f) {
        glslang::TIntermAggregate* glslFunction = glslFunctions[f]->getAsAggregate();
        if (! glslFunction || glslFunction->getOp() != glslang::EOpFunction)
            continue;

        builder.setDebugSourceLocation(glslFunction->getLoc().line, glslFunction->getLoc().getFilename());

        if (isShaderEntryPoint(glslFunction)) {
            // For HLSL, the entry function is actually a compiler generated function to resolve the difference of
            // entry function signature between HLSL and SPIR-V. So we don't emit debug information for that.
            if (glslangIntermediate->getSource() != glslang::EShSourceHlsl) {
                builder.setupFunctionDebugInfo(shaderEntry, glslangIntermediate->getEntryPointMangledName().c_str(),
                                               std::vector<spv::Id>(), // main function has no param
                                               std::vector<char const*>());
            }
            continue;
        }
        // We're on a user function.  Set up the basic interface for the function now,
        // so that it's available to call.  Translating the body will happen later.
        //
        // Typically (except for a "const in" parameter), an address will be passed to the
        // function.  What it is an address of varies:
        //
        // - "in" parameters not marked as "const" can be written to without modifying the calling
        //   argument so that write needs to be to a copy, hence the address of a copy works.
        //
        // - "const in" parameters can just be the r-value, as no writes need occur.
        //
        // - "out" and "inout" arguments can't be done as pointers to the calling argument, because
        //   GLSL has copy-in/copy-out semantics.  They can be handled though with a pointer to a copy.

        std::vector<spv::Id> paramTypes;
        std::vector<char const*> paramNames;
        std::vector<std::vector<spv::Decoration>> paramDecorations; // list of decorations per parameter
        glslang::TIntermSequence& parameters = glslFunction->getSequence()[0]->getAsAggregate()->getSequence();

#ifdef ENABLE_HLSL
        bool implicitThis = (int)parameters.size() > 0 && parameters[0]->getAsSymbolNode()->getName() ==
                                                          glslangIntermediate->implicitThisName;
#else
        bool implicitThis = false;
#endif

        paramDecorations.resize(parameters.size());
        for (int p = 0; p < (int)parameters.size(); ++p) {
            const glslang::TType& paramType = parameters[p]->getAsTyped()->getType();
            spv::Id typeId = convertGlslangToSpvType(paramType);
            if (originalParam(paramType.getQualifier().storage, paramType, implicitThis && p == 0))
                typeId = builder.makePointer(TranslateStorageClass(paramType), typeId);
            else if (writableParam(paramType.getQualifier().storage))
                typeId = builder.makePointer(spv::StorageClass::Function, typeId);
            else
                rValueParameters.insert(parameters[p]->getAsSymbolNode()->getId());
            getParamDecorations(paramDecorations[p], paramType, glslangIntermediate->usingVulkanMemoryModel());
            paramTypes.push_back(typeId);
        }

        for (auto const parameter:parameters) {
            paramNames.push_back(parameter->getAsSymbolNode()->getName().c_str());
        }

        spv::Block* functionBlock;
        spv::Function* function = builder.makeFunctionEntry(
            TranslatePrecisionDecoration(glslFunction->getType()), convertGlslangToSpvType(glslFunction->getType()),
            glslFunction->getName().c_str(), convertGlslangLinkageToSpv(glslFunction->getLinkType()), paramTypes,
            paramDecorations, &functionBlock);
        builder.setupFunctionDebugInfo(function, glslFunction->getName().c_str(), paramTypes, paramNames);
        if (implicitThis)
            function->setImplicitThis();

        // Track function to emit/call later
        functionMap[glslFunction->getName().c_str()] = function;

        // Set the parameter id's
        for (int p = 0; p < (int)parameters.size(); ++p) {
            symbolValues[parameters[p]->getAsSymbolNode()->getId()] = function->getParamId(p);
            // give a name too
            builder.addName(function->getParamId(p), parameters[p]->getAsSymbolNode()->getName().c_str());

            const glslang::TType& paramType = parameters[p]->getAsTyped()->getType();
            if (paramType.contains8BitInt())
                builder.addCapability(spv::Capability::Int8);
            if (paramType.contains16BitInt())
                builder.addCapability(spv::Capability::Int16);
            if (paramType.contains16BitFloat())
                builder.addCapability(spv::Capability::Float16);
        }
    }
}

// Process all the initializers, while skipping the functions and link objects
void TGlslangToSpvTraverser::makeGlobalInitializers(const glslang::TIntermSequence& initializers)
{
    builder.setBuildPoint(shaderEntry->getLastBlock());
    for (int i = 0; i < (int)initializers.size(); ++i) {
        glslang::TIntermAggregate* initializer = initializers[i]->getAsAggregate();
        if (initializer && initializer->getOp() != glslang::EOpFunction && initializer->getOp() !=
            glslang::EOpLinkerObjects) {

            // We're on a top-level node that's not a function.  Treat as an initializer, whose
            // code goes into the beginning of the entry point.
            initializer->traverse(this);
        }
    }
}
// Walk over all linker objects to create a map for payload and callable data linker objects
// and their location to be used during codegen for OpTraceKHR and OpExecuteCallableKHR
// This is done here since it is possible that these linker objects are not be referenced in the AST
void TGlslangToSpvTraverser::collectRayTracingLinkerObjects()
{
    glslang::TIntermAggregate* linkerObjects = glslangIntermediate->findLinkerObjects();
    for (auto& objSeq : linkerObjects->getSequence()) {
        auto objNode = objSeq->getAsSymbolNode();
        if (objNode != nullptr) {
            if (objNode->getQualifier().hasLocation()) {
                unsigned int location = objNode->getQualifier().layoutLocation;
                auto st = objNode->getQualifier().storage;
                int set;
                switch (st)
                {
                case glslang::EvqPayload:
                case glslang::EvqPayloadIn:
                    set = 0;
                    break;
                case glslang::EvqCallableData:
                case glslang::EvqCallableDataIn:
                    set = 1;
                    break;

                case glslang::EvqHitObjectAttrNV:
                case glslang::EvqHitObjectAttrEXT:
                    set = 2;
                    break;

                default:
                    set = -1;
                }
                if (set != -1)
                    locationToSymbol[set].insert(std::make_pair(location, objNode));
            }
        }
    }
}
// Process all the functions, while skipping initializers.
void TGlslangToSpvTraverser::visitFunctions(const glslang::TIntermSequence& glslFunctions)
{
    for (int f = 0; f < (int)glslFunctions.size(); ++f) {
        glslang::TIntermAggregate* node = glslFunctions[f]->getAsAggregate();
        if (node && (node->getOp() == glslang::EOpFunction || node->getOp() == glslang::EOpLinkerObjects))
            node->traverse(this);
    }
}

void TGlslangToSpvTraverser::translateArguments(const glslang::TIntermAggregate& node, std::vector<spv::Id>& arguments,
    spv::Builder::AccessChain::CoherentFlags &lvalueCoherentFlags)
{
    const glslang::TIntermSequence& glslangArguments = node.getSequence();

    glslang::TSampler sampler = {};
    bool cubeCompare = false;
    bool f16ShadowCompare = false;
    if (node.isTexture() || node.isImage()) {
        sampler = glslangArguments[0]->getAsTyped()->getType().getSampler();
        cubeCompare = sampler.dim == glslang::EsdCube && sampler.arrayed && sampler.shadow;
        f16ShadowCompare = sampler.shadow &&
            glslangArguments[1]->getAsTyped()->getType().getBasicType() == glslang::EbtFloat16;
    }

    for (int i = 0; i < (int)glslangArguments.size(); ++i) {
        builder.clearAccessChain();
        glslangArguments[i]->traverse(this);

        // Special case l-value operands
        bool lvalue = false;
        switch (node.getOp()) {
        case glslang::EOpImageAtomicAdd:
        case glslang::EOpImageAtomicMin:
        case glslang::EOpImageAtomicMax:
        case glslang::EOpImageAtomicAnd:
        case glslang::EOpImageAtomicOr:
        case glslang::EOpImageAtomicXor:
        case glslang::EOpImageAtomicExchange:
        case glslang::EOpImageAtomicCompSwap:
        case glslang::EOpImageAtomicLoad:
        case glslang::EOpImageAtomicStore:
            if (i == 0)
                lvalue = true;
            break;
        case glslang::EOpSparseImageLoad:
            if ((sampler.ms && i == 3) || (! sampler.ms && i == 2))
                lvalue = true;
            break;
        case glslang::EOpSparseTexture:
            if (((cubeCompare || f16ShadowCompare) && i == 3) || (! (cubeCompare || f16ShadowCompare) && i == 2))
                lvalue = true;
            break;
        case glslang::EOpSparseTextureClamp:
            if (((cubeCompare || f16ShadowCompare) && i == 4) || (! (cubeCompare || f16ShadowCompare) && i == 3))
                lvalue = true;
            break;
        case glslang::EOpSparseTextureLod:
        case glslang::EOpSparseTextureOffset:
            if  ((f16ShadowCompare && i == 4) || (! f16ShadowCompare && i == 3))
                lvalue = true;
            break;
        case glslang::EOpSparseTextureFetch:
            if ((sampler.dim != glslang::EsdRect && i == 3) || (sampler.dim == glslang::EsdRect && i == 2))
                lvalue = true;
            break;
        case glslang::EOpSparseTextureFetchOffset:
            if ((sampler.dim != glslang::EsdRect && i == 4) || (sampler.dim == glslang::EsdRect && i == 3))
                lvalue = true;
            break;
        case glslang::EOpSparseTextureLodOffset:
        case glslang::EOpSparseTextureGrad:
        case glslang::EOpSparseTextureOffsetClamp:
            if ((f16ShadowCompare && i == 5) || (! f16ShadowCompare && i == 4))
                lvalue = true;
            break;
        case glslang::EOpSparseTextureGradOffset:
        case glslang::EOpSparseTextureGradClamp:
            if ((f16ShadowCompare && i == 6) || (! f16ShadowCompare && i == 5))
                lvalue = true;
            break;
        case glslang::EOpSparseTextureGradOffsetClamp:
            if ((f16ShadowCompare && i == 7) || (! f16ShadowCompare && i == 6))
                lvalue = true;
            break;
        case glslang::EOpSparseTextureGather:
            if ((sampler.shadow && i == 3) || (! sampler.shadow && i == 2))
                lvalue = true;
            break;
        case glslang::EOpSparseTextureGatherOffset:
        case glslang::EOpSparseTextureGatherOffsets:
            if ((sampler.shadow && i == 4) || (! sampler.shadow && i == 3))
                lvalue = true;
            break;
        case glslang::EOpSparseTextureGatherLod:
            if (i == 3)
                lvalue = true;
            break;
        case glslang::EOpSparseTextureGatherLodOffset:
        case glslang::EOpSparseTextureGatherLodOffsets:
            if (i == 4)
                lvalue = true;
            break;
        case glslang::EOpSparseImageLoadLod:
            if (i == 3)
                lvalue = true;
            break;
        case glslang::EOpImageSampleFootprintNV:
            if (i == 4)
                lvalue = true;
            break;
        case glslang::EOpImageSampleFootprintClampNV:
        case glslang::EOpImageSampleFootprintLodNV:
            if (i == 5)
                lvalue = true;
            break;
        case glslang::EOpImageSampleFootprintGradNV:
            if (i == 6)
                lvalue = true;
            break;
        case glslang::EOpImageSampleFootprintGradClampNV:
            if (i == 7)
                lvalue = true;
            break;
        case glslang::EOpRayQueryGetIntersectionTriangleVertexPositionsEXT:
        case glslang::EOpRayQueryGetIntersectionLSSPositionsNV:
        case glslang::EOpRayQueryGetIntersectionLSSRadiiNV:
            if (i == 2)
                lvalue = true;
            break;
        case glslang::EOpConstructSaturated:
            if (i == 0)
                lvalue = true;
            break;
        default:
            break;
        }

        if (lvalue) {
            spv::Id lvalue_id = builder.accessChainGetLValue();
            arguments.push_back(lvalue_id);
            lvalueCoherentFlags = builder.getAccessChain().coherentFlags;
            builder.addDecoration(lvalue_id, TranslateNonUniformDecoration(lvalueCoherentFlags));
            lvalueCoherentFlags |= TranslateCoherent(glslangArguments[i]->getAsTyped()->getType());
        } else {
            if (i > 0 &&
                glslangArguments[i]->getAsSymbolNode() && glslangArguments[i-1]->getAsSymbolNode() &&
                glslangArguments[i]->getAsSymbolNode()->getId() == glslangArguments[i-1]->getAsSymbolNode()->getId()) {
                // Reuse the id if possible
                arguments.push_back(arguments[i-1]);
            } else {
                arguments.push_back(accessChainLoad(glslangArguments[i]->getAsTyped()->getType()));
            }
        }
    }
}

void TGlslangToSpvTraverser::translateArguments(glslang::TIntermUnary& node, std::vector<spv::Id>& arguments)
{
    builder.clearAccessChain();
    node.getOperand()->traverse(this);
    arguments.push_back(accessChainLoad(node.getOperand()->getType()));
}

spv::Id TGlslangToSpvTraverser::createImageTextureFunctionCall(glslang::TIntermOperator* node)
{
    if (! node->isImage() && ! node->isTexture())
        return spv::NoResult;

    builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());

    // Process a GLSL texturing op (will be SPV image)

    const glslang::TType &imageType = node->getAsAggregate()
                                        ? node->getAsAggregate()->getSequence()[0]->getAsTyped()->getType()
                                        : node->getAsUnaryNode()->getOperand()->getAsTyped()->getType();
    const glslang::TSampler sampler = imageType.getSampler();
    bool f16ShadowCompare = (sampler.shadow && node->getAsAggregate())
            ? node->getAsAggregate()->getSequence()[1]->getAsTyped()->getType().getBasicType() == glslang::EbtFloat16
            : false;

    const auto signExtensionMask = [&]() {
        if (builder.getSpvVersion() >= spv::Spv_1_4) {
            if (sampler.type == glslang::EbtUint)
                return spv::ImageOperandsMask::ZeroExtend;
            else if (sampler.type == glslang::EbtInt)
                return spv::ImageOperandsMask::SignExtend;
        }
        return spv::ImageOperandsMask::MaskNone;
    };

    spv::Builder::AccessChain::CoherentFlags lvalueCoherentFlags;

    std::vector<spv::Id> arguments;
    if (node->getAsAggregate())
        translateArguments(*node->getAsAggregate(), arguments, lvalueCoherentFlags);
    else
        translateArguments(*node->getAsUnaryNode(), arguments);
    spv::Decoration precision = TranslatePrecisionDecoration(node->getType());

    spv::Builder::TextureParameters params = { };
    params.sampler = arguments[0];

    glslang::TCrackedTextureOp cracked;
    node->crackTexture(sampler, cracked);

    const bool isUnsignedResult = node->getType().getBasicType() == glslang::EbtUint;

    if (builder.isSampledImage(params.sampler) &&
        ((cracked.query && node->getOp() != glslang::EOpTextureQueryLod) || cracked.fragMask || cracked.fetch)) {
        params.sampler = builder.createUnaryOp(spv::Op::OpImage, builder.getImageType(params.sampler), params.sampler);
        if (imageType.getQualifier().isNonUniform()) {
            builder.addDecoration(params.sampler, spv::Decoration::NonUniformEXT);
        }
    }
    // Check for queries
    if (cracked.query) {
        switch (node->getOp()) {
        case glslang::EOpImageQuerySize:
        case glslang::EOpTextureQuerySize:
            if (arguments.size() > 1) {
                params.lod = arguments[1];
                return builder.createTextureQueryCall(spv::Op::OpImageQuerySizeLod, params, isUnsignedResult);
            } else
                return builder.createTextureQueryCall(spv::Op::OpImageQuerySize, params, isUnsignedResult);
        case glslang::EOpImageQuerySamples:
        case glslang::EOpTextureQuerySamples:
            return builder.createTextureQueryCall(spv::Op::OpImageQuerySamples, params, isUnsignedResult);
        case glslang::EOpTextureQueryLod:
            params.coords = arguments[1];
            return builder.createTextureQueryCall(spv::Op::OpImageQueryLod, params, isUnsignedResult);
        case glslang::EOpTextureQueryLevels:
            return builder.createTextureQueryCall(spv::Op::OpImageQueryLevels, params, isUnsignedResult);
        case glslang::EOpSparseTexelsResident:
            return builder.createUnaryOp(spv::Op::OpImageSparseTexelsResident, builder.makeBoolType(), arguments[0]);
        default:
            assert(0);
            break;
        }
    }

    int components = node->getType().getVectorSize();

    if (node->getOp() == glslang::EOpImageLoad ||
        node->getOp() == glslang::EOpImageLoadLod ||
        node->getOp() == glslang::EOpTextureFetch ||
        node->getOp() == glslang::EOpTextureFetchOffset) {
        // These must produce 4 components, per SPIR-V spec.  We'll add a conversion constructor if needed.
        // This will only happen through the HLSL path for operator[], so we do not have to handle e.g.
        // the EOpTexture/Proj/Lod/etc family.  It would be harmless to do so, but would need more logic
        // here around e.g. which ones return scalars or other types.
        components = 4;
    }

    glslang::TType returnType(node->getType().getBasicType(), glslang::EvqTemporary, components);

    auto resultType = [&returnType,this]{ return convertGlslangToSpvType(returnType); };

    // Check for image functions other than queries
    if (node->isImage()) {
        std::vector<spv::IdImmediate> operands;
        auto opIt = arguments.begin();
        spv::IdImmediate image = { true, *(opIt++) };
        operands.push_back(image);

        // Handle subpass operations
        // TODO: GLSL should change to have the "MS" only on the type rather than the
        // built-in function.
        if (cracked.subpass) {
            // add on the (0,0) coordinate
            spv::Id zero = builder.makeIntConstant(0);
            std::vector<spv::Id> comps;
            comps.push_back(zero);
            comps.push_back(zero);
            spv::IdImmediate coord = { true,
                builder.makeCompositeConstant(builder.makeVectorType(builder.makeIntType(32), 2), comps) };
            operands.push_back(coord);
            spv::IdImmediate imageOperands = { false, spv::ImageOperandsMask::MaskNone };
            imageOperands.word = imageOperands.word | (unsigned)signExtensionMask();
            if (sampler.isMultiSample()) {
                imageOperands.word = imageOperands.word | (unsigned)spv::ImageOperandsMask::Sample;
            }
            if (imageOperands.word != (unsigned)spv::ImageOperandsMask::MaskNone) {
                operands.push_back(imageOperands);
                if (sampler.isMultiSample()) {
                    spv::IdImmediate imageOperand = { true, *(opIt++) };
                    operands.push_back(imageOperand);
                }
            }
            spv::Id result = builder.createOp(spv::Op::OpImageRead, resultType(), operands);
            builder.setPrecision(result, precision);
            return result;
        }

        if (cracked.attachmentEXT) {
            if (opIt != arguments.end()) {
                spv::IdImmediate sample = { true, *opIt };
                operands.push_back(sample);
            }
            spv::Id result = builder.createOp(spv::Op::OpColorAttachmentReadEXT, resultType(), operands);
            builder.addExtension(spv::E_SPV_EXT_shader_tile_image);
            builder.setPrecision(result, precision);
            return result;
        }

        spv::IdImmediate coord = { true, *(opIt++) };
        operands.push_back(coord);
        if (node->getOp() == glslang::EOpImageLoad || node->getOp() == glslang::EOpImageLoadLod) {
            spv::ImageOperandsMask mask = spv::ImageOperandsMask::MaskNone;
            if (sampler.isMultiSample()) {
                mask = mask | spv::ImageOperandsMask::Sample;
            }
            if (cracked.lod) {
                builder.addExtension(spv::E_SPV_AMD_shader_image_load_store_lod);
                builder.addCapability(spv::Capability::ImageReadWriteLodAMD);
                mask = mask | spv::ImageOperandsMask::Lod;
            }
            mask = mask | TranslateImageOperands(TranslateCoherent(imageType));
            mask = (spv::ImageOperandsMask)(mask & ~spv::ImageOperandsMask::MakeTexelAvailableKHR);
            mask = mask | signExtensionMask();
            if (mask != spv::ImageOperandsMask::MaskNone) {
                spv::IdImmediate imageOperands = { false, (unsigned int)mask };
                operands.push_back(imageOperands);
            }
            if (anySet(mask, spv::ImageOperandsMask::Sample)) {
                spv::IdImmediate imageOperand = { true, *opIt++ };
                operands.push_back(imageOperand);
            }
            if (anySet(mask, spv::ImageOperandsMask::Lod)) {
                spv::IdImmediate imageOperand = { true, *opIt++ };
                operands.push_back(imageOperand);
            }
            if (anySet(mask, spv::ImageOperandsMask::MakeTexelVisibleKHR)) {
                spv::IdImmediate imageOperand = { true,
                                    builder.makeUintConstant(TranslateMemoryScope(TranslateCoherent(imageType))) };
                operands.push_back(imageOperand);
            }

            if (builder.getImageTypeFormat(builder.getImageType(operands.front().word)) == spv::ImageFormat::Unknown)
                builder.addCapability(spv::Capability::StorageImageReadWithoutFormat);

            std::vector<spv::Id> result(1, builder.createOp(spv::Op::OpImageRead, resultType(), operands));
            builder.setPrecision(result[0], precision);

            // If needed, add a conversion constructor to the proper size.
            if (components != node->getType().getVectorSize())
                result[0] = builder.createConstructor(precision, result, convertGlslangToSpvType(node->getType()));

            return result[0];
        } else if (node->getOp() == glslang::EOpImageStore || node->getOp() == glslang::EOpImageStoreLod) {

            // Push the texel value before the operands
            if (sampler.isMultiSample() || cracked.lod) {
                spv::IdImmediate texel = { true, *(opIt + 1) };
                operands.push_back(texel);
            } else {
                spv::IdImmediate texel = { true, *opIt };
                operands.push_back(texel);
            }

            spv::ImageOperandsMask mask = spv::ImageOperandsMask::MaskNone;
            if (sampler.isMultiSample()) {
                mask = mask | spv::ImageOperandsMask::Sample;
            }
            if (cracked.lod) {
                builder.addExtension(spv::E_SPV_AMD_shader_image_load_store_lod);
                builder.addCapability(spv::Capability::ImageReadWriteLodAMD);
                mask = mask | spv::ImageOperandsMask::Lod;
            }
            mask = mask | TranslateImageOperands(TranslateCoherent(imageType));
            mask = (spv::ImageOperandsMask)(mask & ~spv::ImageOperandsMask::MakeTexelVisibleKHR);
            mask = mask | signExtensionMask();
            if (mask != spv::ImageOperandsMask::MaskNone) {
                spv::IdImmediate imageOperands = { false, (unsigned int)mask };
                operands.push_back(imageOperands);
            }
            if (anySet(mask, spv::ImageOperandsMask::Sample)) {
                spv::IdImmediate imageOperand = { true, *opIt++ };
                operands.push_back(imageOperand);
            }
            if (anySet(mask, spv::ImageOperandsMask::Lod)) {
                spv::IdImmediate imageOperand = { true, *opIt++ };
                operands.push_back(imageOperand);
            }
            if (anySet(mask, spv::ImageOperandsMask::MakeTexelAvailableKHR)) {
                spv::IdImmediate imageOperand = { true,
                    builder.makeUintConstant(TranslateMemoryScope(TranslateCoherent(imageType))) };
                operands.push_back(imageOperand);
            }

            builder.createNoResultOp(spv::Op::OpImageWrite, operands);
            if (builder.getImageTypeFormat(builder.getImageType(operands.front().word)) == spv::ImageFormat::Unknown)
                builder.addCapability(spv::Capability::StorageImageWriteWithoutFormat);
            return spv::NoResult;
        } else if (node->getOp() == glslang::EOpSparseImageLoad ||
                   node->getOp() == glslang::EOpSparseImageLoadLod) {
            builder.addCapability(spv::Capability::SparseResidency);
            if (builder.getImageTypeFormat(builder.getImageType(operands.front().word)) == spv::ImageFormat::Unknown)
                builder.addCapability(spv::Capability::StorageImageReadWithoutFormat);

            spv::ImageOperandsMask mask = spv::ImageOperandsMask::MaskNone;
            if (sampler.isMultiSample()) {
                mask = mask | spv::ImageOperandsMask::Sample;
            }
            if (cracked.lod) {
                builder.addExtension(spv::E_SPV_AMD_shader_image_load_store_lod);
                builder.addCapability(spv::Capability::ImageReadWriteLodAMD);

                mask = mask | spv::ImageOperandsMask::Lod;
            }
            mask = mask | TranslateImageOperands(TranslateCoherent(imageType));
            mask = (spv::ImageOperandsMask)(mask & ~spv::ImageOperandsMask::MakeTexelAvailableKHR);
            mask = mask | signExtensionMask();
            if (mask != spv::ImageOperandsMask::MaskNone) {
                spv::IdImmediate imageOperands = { false, (unsigned int)mask };
                operands.push_back(imageOperands);
            }
            if (anySet(mask, spv::ImageOperandsMask::Sample)) {
                spv::IdImmediate imageOperand = { true, *opIt++ };
                operands.push_back(imageOperand);
            }
            if (anySet(mask, spv::ImageOperandsMask::Lod)) {
                spv::IdImmediate imageOperand = { true, *opIt++ };
                operands.push_back(imageOperand);
            }
            if (anySet(mask, spv::ImageOperandsMask::MakeTexelVisibleKHR)) {
                spv::IdImmediate imageOperand = { true, builder.makeUintConstant(TranslateMemoryScope(
                    TranslateCoherent(imageType))) };
                operands.push_back(imageOperand);
            }

            // Create the return type that was a special structure
            spv::Id texelOut = *opIt;
            spv::Id typeId0 = resultType();
            spv::Id typeId1 = builder.getDerefTypeId(texelOut);
            spv::Id resultTypeId = builder.makeStructResultType(typeId0, typeId1);

            spv::Id resultId = builder.createOp(spv::Op::OpImageSparseRead, resultTypeId, operands);

            // Decode the return type
            builder.createStore(builder.createCompositeExtract(resultId, typeId1, 1), texelOut);
            return builder.createCompositeExtract(resultId, typeId0, 0);
        } else {
            // Process image atomic operations

            // GLSL "IMAGE_PARAMS" will involve in constructing an image texel pointer and this pointer,
            // as the first source operand, is required by SPIR-V atomic operations.
            // For non-MS, the sample value should be 0
            spv::IdImmediate sample = { true, sampler.isMultiSample() ? *(opIt++) : builder.makeUintConstant(0) };
            operands.push_back(sample);

            spv::Id resultTypeId;
            glslang::TBasicType typeProxy = node->getBasicType();
            // imageAtomicStore has a void return type so base the pointer type on
            // the type of the value operand.
            if (node->getOp() == glslang::EOpImageAtomicStore) {
                resultTypeId = builder.makePointer(spv::StorageClass::Image, builder.getTypeId(*opIt));
                typeProxy = node->getAsAggregate()->getSequence()[0]->getAsTyped()->getType().getSampler().type;
            } else {
                resultTypeId = builder.makePointer(spv::StorageClass::Image, resultType());
            }
            spv::Id pointer = builder.createOp(spv::Op::OpImageTexelPointer, resultTypeId, operands);
            if (imageType.getQualifier().nonUniform) {
                builder.addDecoration(pointer, spv::Decoration::NonUniformEXT);
            }

            std::vector<spv::Id> operands;
            operands.push_back(pointer);
            for (; opIt != arguments.end(); ++opIt)
                operands.push_back(*opIt);

            return createAtomicOperation(node->getOp(), precision, resultType(), operands, typeProxy,
                lvalueCoherentFlags, node->getType());
        }
    }

    // Check for fragment mask functions other than queries
    if (cracked.fragMask) {
        assert(sampler.ms);

        auto opIt = arguments.begin();
        std::vector<spv::Id> operands;

        operands.push_back(params.sampler);
        ++opIt;

        if (sampler.isSubpass()) {
            // add on the (0,0) coordinate
            spv::Id zero = builder.makeIntConstant(0);
            std::vector<spv::Id> comps;
            comps.push_back(zero);
            comps.push_back(zero);
            operands.push_back(builder.makeCompositeConstant(
                builder.makeVectorType(builder.makeIntType(32), 2), comps));
        }

        for (; opIt != arguments.end(); ++opIt)
            operands.push_back(*opIt);

        spv::Op fragMaskOp = spv::Op::OpNop;
        if (node->getOp() == glslang::EOpFragmentMaskFetch)
            fragMaskOp = spv::Op::OpFragmentMaskFetchAMD;
        else if (node->getOp() == glslang::EOpFragmentFetch)
            fragMaskOp = spv::Op::OpFragmentFetchAMD;

        builder.addExtension(spv::E_SPV_AMD_shader_fragment_mask);
        builder.addCapability(spv::Capability::FragmentMaskAMD);
        return builder.createOp(fragMaskOp, resultType(), operands);
    }

    // Check for texture functions other than queries
    bool sparse = node->isSparseTexture();
    bool imageFootprint = node->isImageFootprint();
    bool cubeCompare = sampler.dim == glslang::EsdCube && sampler.isArrayed() && sampler.isShadow();

    // check for bias argument
    bool bias = false;
    if (! cracked.lod && ! cracked.grad && ! cracked.fetch && ! cubeCompare) {
        int nonBiasArgCount = 2;
        if (cracked.gather)
            ++nonBiasArgCount; // comp argument should be present when bias argument is present

        if (f16ShadowCompare)
            ++nonBiasArgCount;
        if (cracked.offset)
            ++nonBiasArgCount;
        else if (cracked.offsets)
            ++nonBiasArgCount;
        if (cracked.grad)
            nonBiasArgCount += 2;
        if (cracked.lodClamp)
            ++nonBiasArgCount;
        if (sparse)
            ++nonBiasArgCount;
        if (imageFootprint)
            //Following three extra arguments
            // int granularity, bool coarse, out gl_TextureFootprint2DNV footprint
            nonBiasArgCount += 3;
        if ((int)arguments.size() > nonBiasArgCount)
            bias = true;
    }

    if (cracked.gather) {
        const auto& sourceExtensions = glslangIntermediate->getRequestedExtensions();
        if (bias || cracked.lod ||
            sourceExtensions.find(glslang::E_GL_AMD_texture_gather_bias_lod) != sourceExtensions.end()) {
            builder.addExtension(spv::E_SPV_AMD_texture_gather_bias_lod);
            builder.addCapability(spv::Capability::ImageGatherBiasLodAMD);
        }
    }

    // set the rest of the arguments

    params.coords = arguments[1];
    int extraArgs = 0;
    bool noImplicitLod = false;

    // sort out where Dref is coming from
    if (cubeCompare || f16ShadowCompare) {
        params.Dref = arguments[2];
        ++extraArgs;
    } else if (sampler.shadow && cracked.gather) {
        params.Dref = arguments[2];
        ++extraArgs;
    } else if (sampler.shadow) {
        std::vector<spv::Id> indexes;
        int dRefComp;
        if (cracked.proj)
            dRefComp = 2;  // "The resulting 3rd component of P in the shadow forms is used as Dref"
        else
            dRefComp = builder.getNumComponents(params.coords) - 1;
        indexes.push_back(dRefComp);
        params.Dref = builder.createCompositeExtract(params.coords,
            builder.getScalarTypeId(builder.getTypeId(params.coords)), indexes);
    }

    // lod
    if (cracked.lod) {
        params.lod = arguments[2 + extraArgs];
        ++extraArgs;
    } else if (glslangIntermediate->getStage() != EShLangFragment &&
               !(glslangIntermediate->getStage() == EShLangCompute &&
                 glslangIntermediate->hasLayoutDerivativeModeNone())) {
        // we need to invent the default lod for an explicit lod instruction for a non-fragment stage
        noImplicitLod = true;
    }

    // multisample
    if (sampler.isMultiSample()) {
        params.sample = arguments[2 + extraArgs]; // For MS, "sample" should be specified
        ++extraArgs;
    }

    // gradient
    if (cracked.grad) {
        params.gradX = arguments[2 + extraArgs];
        params.gradY = arguments[3 + extraArgs];
        extraArgs += 2;
    }

    // offset and offsets
    if (cracked.offset) {
        params.offset = arguments[2 + extraArgs];
        ++extraArgs;
    } else if (cracked.offsets) {
        params.offsets = arguments[2 + extraArgs];
        ++extraArgs;
    }

    // lod clamp
    if (cracked.lodClamp) {
        params.lodClamp = arguments[2 + extraArgs];
        ++extraArgs;
    }
    // sparse
    if (sparse) {
        params.texelOut = arguments[2 + extraArgs];
        ++extraArgs;
    }
    // gather component
    if (cracked.gather && ! sampler.shadow) {
        // default component is 0, if missing, otherwise an argument
        if (2 + extraArgs < (int)arguments.size()) {
            params.component = arguments[2 + extraArgs];
            ++extraArgs;
        } else
            params.component = builder.makeIntConstant(0);
    }
    spv::Id  resultStruct = spv::NoResult;
    if (imageFootprint) {
        //Following three extra arguments
        // int granularity, bool coarse, out gl_TextureFootprint2DNV footprint
        params.granularity = arguments[2 + extraArgs];
        params.coarse = arguments[3 + extraArgs];
        resultStruct = arguments[4 + extraArgs];
        extraArgs += 3;
    }

    // bias
    if (bias) {
        params.bias = arguments[2 + extraArgs];
        ++extraArgs;
    }

    if (imageFootprint) {
        builder.addExtension(spv::E_SPV_NV_shader_image_footprint);
        builder.addCapability(spv::Capability::ImageFootprintNV);


        //resultStructType(OpenGL type) contains 5 elements:
        //struct gl_TextureFootprint2DNV {
        //    uvec2 anchor;
        //    uvec2 offset;
        //    uvec2 mask;
        //    uint  lod;
        //    uint  granularity;
        //};
        //or
        //struct gl_TextureFootprint3DNV {
        //    uvec3 anchor;
        //    uvec3 offset;
        //    uvec2 mask;
        //    uint  lod;
        //    uint  granularity;
        //};
        spv::Id resultStructType = builder.getContainedTypeId(builder.getTypeId(resultStruct));
        assert(builder.isStructType(resultStructType));

        //resType (SPIR-V type) contains 6 elements:
        //Member 0 must be a Boolean type scalar(LOD),
        //Member 1 must be a vector of integer type, whose Signedness operand is 0(anchor),
        //Member 2 must be a vector of integer type, whose Signedness operand is 0(offset),
        //Member 3 must be a vector of integer type, whose Signedness operand is 0(mask),
        //Member 4 must be a scalar of integer type, whose Signedness operand is 0(lod),
        //Member 5 must be a scalar of integer type, whose Signedness operand is 0(granularity).
        std::vector<spv::Id> members;
        members.push_back(resultType());
        for (int i = 0; i < 5; i++) {
            members.push_back(builder.getContainedTypeId(resultStructType, i));
        }
        spv::Id resType = builder.makeStructType(members, {}, "ResType");

        //call ImageFootprintNV
        spv::Id res = builder.createTextureCall(precision, resType, sparse, cracked.fetch, cracked.proj,
                                                cracked.gather, noImplicitLod, params, signExtensionMask());

        //copy resType (SPIR-V type) to resultStructType(OpenGL type)
        for (int i = 0; i < 5; i++) {
            builder.clearAccessChain();
            builder.setAccessChainLValue(resultStruct);

            //Accessing to a struct we created, no coherent flag is set
            spv::Builder::AccessChain::CoherentFlags flags;
            flags.clear();

            builder.accessChainPush(builder.makeIntConstant(i), flags, 0);
            builder.accessChainStore(builder.createCompositeExtract(res, builder.getContainedTypeId(resType, i+1),
                i+1), TranslateNonUniformDecoration(imageType.getQualifier()));
        }
        return builder.createCompositeExtract(res, resultType(), 0);
    }

    // projective component (might not to move)
    // GLSL: "The texture coordinates consumed from P, not including the last component of P,
    //       are divided by the last component of P."
    // SPIR-V:  "... (u [, v] [, w], q)... It may be a vector larger than needed, but all
    //          unused components will appear after all used components."
    if (cracked.proj) {
        int projSourceComp = builder.getNumComponents(params.coords) - 1;
        int projTargetComp;
        switch (sampler.dim) {
        case glslang::Esd1D:   projTargetComp = 1;              break;
        case glslang::Esd2D:   projTargetComp = 2;              break;
        case glslang::EsdRect: projTargetComp = 2;              break;
        default:               projTargetComp = projSourceComp; break;
        }
        // copy the projective coordinate if we have to
        if (projTargetComp != projSourceComp) {
            spv::Id projComp = builder.createCompositeExtract(params.coords,
                                    builder.getScalarTypeId(builder.getTypeId(params.coords)), projSourceComp);
            params.coords = builder.createCompositeInsert(projComp, params.coords,
                                    builder.getTypeId(params.coords), projTargetComp);
        }
    }

    // nonprivate
    if (imageType.getQualifier().nonprivate) {
        params.nonprivate = true;
    }

    // volatile
    if (imageType.getQualifier().volatil) {
        params.volatil = true;
    }

    if (imageType.getQualifier().nontemporal) {
        params.nontemporal = true;
    }

    std::vector<spv::Id> result( 1,
        builder.createTextureCall(precision, resultType(), sparse, cracked.fetch, cracked.proj, cracked.gather,
                                  noImplicitLod, params, signExtensionMask())
    );

    if (components != node->getType().getVectorSize())
        result[0] = builder.createConstructor(precision, result, convertGlslangToSpvType(node->getType()));

    return result[0];
}

spv::Id TGlslangToSpvTraverser::handleUserFunctionCall(const glslang::TIntermAggregate* node)
{
    // Grab the function's pointer from the previously created function
    spv::Function* function = functionMap[node->getName().c_str()];
    if (! function)
        return 0;

    const glslang::TIntermSequence& glslangArgs = node->getSequence();
    const glslang::TQualifierList& qualifiers = node->getQualifierList();

    //  See comments in makeFunctions() for details about the semantics for parameter passing.
    //
    // These imply we need a four step process:
    // 1. Evaluate the arguments
    // 2. Allocate and make copies of in, out, and inout arguments
    // 3. Make the call
    // 4. Copy back the results

    // 1. Evaluate the arguments and their types
    std::vector<spv::Builder::AccessChain> lValues;
    std::vector<spv::Id> rValues;
    std::vector<const glslang::TType*> argTypes;
    for (int a = 0; a < (int)glslangArgs.size(); ++a) {
        argTypes.push_back(&glslangArgs[a]->getAsTyped()->getType());
        // build l-value
        builder.clearAccessChain();
        glslangArgs[a]->traverse(this);
        // keep outputs and pass-by-originals as l-values, evaluate others as r-values
        if (originalParam(qualifiers[a], *argTypes[a], function->hasImplicitThis() && a == 0) ||
            writableParam(qualifiers[a])) {
            // save l-value
            lValues.push_back(builder.getAccessChain());
        } else {
            // process r-value
            rValues.push_back(accessChainLoad(*argTypes.back()));
        }
    }

    // Reset source location to the function call location after argument evaluation
    builder.setDebugSourceLocation(node->getLoc().line, node->getLoc().getFilename());

    // 2. Allocate space for anything needing a copy, and if it's "in" or "inout"
    // copy the original into that space.
    //
    // Also, build up the list of actual arguments to pass in for the call
    int lValueCount = 0;
    int rValueCount = 0;
    std::vector<spv::Id> spvArgs;
    for (int a = 0; a < (int)glslangArgs.size(); ++a) {
        spv::Id arg;
        if (originalParam(qualifiers[a], *argTypes[a], function->hasImplicitThis() && a == 0)) {
            builder.setAccessChain(lValues[lValueCount]);
            arg = builder.accessChainGetLValue();
            ++lValueCount;
        } else if (writableParam(qualifiers[a])) {
            // need space to hold the copy
            arg = builder.createVariable(function->getParamPrecision(a), spv::StorageClass::Function,
                builder.getContainedTypeId(function->getParamType(a)), "param");
            if (qualifiers[a] == glslang::EvqIn || qualifiers[a] == glslang::EvqInOut) {
                // need to copy the input into output space
                builder.setAccessChain(lValues[lValueCount]);
                spv::Id copy = accessChainLoad(*argTypes[a]);
                builder.clearAccessChain();
                builder.setAccessChainLValue(arg);
                multiTypeStore(*argTypes[a], copy);
            }
            ++lValueCount;
        } else {
            // process r-value, which involves a copy for a type mismatch
            if (function->getParamType(a) != builder.getTypeId(rValues[rValueCount]) ||
                TranslatePrecisionDecoration(*argTypes[a]) != function->getParamPrecision(a))
            {
                spv::Id argCopy = builder.createVariable(function->getParamPrecision(a), spv::StorageClass::Function, function->getParamType(a), "arg");
                builder.clearAccessChain();
                builder.setAccessChainLValue(argCopy);
                multiTypeStore(*argTypes[a], rValues[rValueCount]);
                arg = builder.createLoad(argCopy, function->getParamPrecision(a));
            } else
                arg = rValues[rValueCount];
            ++rValueCount;
        }
        spvArgs.push_back(arg);
    }

    // 3. Make the call.
    spv::Id result = builder.createFunctionCall(function, spvArgs);
    builder.setPrecision(result, TranslatePrecisionDecoration(node->getType()));
    builder.addDecoration(result, TranslateNonUniformDecoration(node->getType().getQualifier()));

    // 4. Copy back out an "out" arguments.
    lValueCount = 0;
    for (int a = 0; a < (int)glslangArgs.size(); ++a) {
        if (originalParam(qualifiers[a], *argTypes[a], function->hasImplicitThis() && a == 0))
            ++lValueCount;
        else if (writableParam(qualifiers[a])) {
            if (qualifiers[a] == glslang::EvqOut || qualifiers[a] == glslang::EvqInOut) {
                spv::Id copy = builder.createLoad(spvArgs[a], spv::NoPrecision);
                builder.addDecoration(copy, TranslateNonUniformDecoration(argTypes[a]->getQualifier()));
                builder.setAccessChain(lValues[lValueCount]);
                multiTypeStore(*argTypes[a], copy);
            }
            ++lValueCount;
        }
    }

    return result;
}

// Translate AST operation to SPV operation, already having SPV-based operands/types.
spv::Id TGlslangToSpvTraverser::createBinaryOperation(glslang::TOperator op, OpDecorations& decorations,
                                                      spv::Id typeId, spv::Id left, spv::Id right,
                                                      glslang::TBasicType typeProxy, bool reduceComparison)
{
    bool isUnsigned = isTypeUnsignedInt(typeProxy);
    bool isFloat = isTypeFloat(typeProxy);
    bool isBool = typeProxy == glslang::EbtBool;

    spv::Op binOp = spv::Op::OpNop;
    bool needMatchingVectors = true;  // for non-matrix ops, would a scalar need to smear to match a vector?
    bool comparison = false;

    switch (op) {
    case glslang::EOpAdd:
    case glslang::EOpAddAssign:
        if (isFloat)
            binOp = spv::Op::OpFAdd;
        else
            binOp = spv::Op::OpIAdd;
        break;
    case glslang::EOpSub:
    case glslang::EOpSubAssign:
        if (isFloat)
            binOp = spv::Op::OpFSub;
        else
            binOp = spv::Op::OpISub;
        break;
    case glslang::EOpMul:
    case glslang::EOpMulAssign:
        if (isFloat)
            binOp = spv::Op::OpFMul;
        else
            binOp = spv::Op::OpIMul;
        break;
    case glslang::EOpVectorTimesScalar:
    case glslang::EOpVectorTimesScalarAssign:
        if (isFloat && (builder.isVector(left) || builder.isVector(right) || builder.isCooperativeVector(left) || builder.isCooperativeVector(right))) {
            if (builder.isVector(right) || builder.isCooperativeVector(right))
                std::swap(left, right);
            assert(builder.isScalar(right));
            needMatchingVectors = false;
            binOp = spv::Op::OpVectorTimesScalar;
        } else if (isFloat) {
            binOp = spv::Op::OpFMul;
        } else if (builder.isCooperativeVector(left) || builder.isCooperativeVector(right)) {
            if (builder.isCooperativeVector(right))
                std::swap(left, right);
            assert(builder.isScalar(right));
            // Construct a cooperative vector from the scalar
            right = builder.createCompositeConstruct(builder.getTypeId(left), { right });
            binOp = spv::Op::OpIMul;
        } else {
            binOp = spv::Op::OpIMul;
        }
        break;
    case glslang::EOpVectorTimesMatrix:
    case glslang::EOpVectorTimesMatrixAssign:
        binOp = spv::Op::OpVectorTimesMatrix;
        break;
    case glslang::EOpMatrixTimesVector:
        binOp = spv::Op::OpMatrixTimesVector;
        break;
    case glslang::EOpMatrixTimesScalar:
    case glslang::EOpMatrixTimesScalarAssign:
        binOp = spv::Op::OpMatrixTimesScalar;
        break;
    case glslang::EOpMatrixTimesMatrix:
    case glslang::EOpMatrixTimesMatrixAssign:
        binOp = spv::Op::OpMatrixTimesMatrix;
        break;
    case glslang::EOpOuterProduct:
        binOp = spv::Op::OpOuterProduct;
        needMatchingVectors = false;
        break;

    case glslang::EOpDiv:
    case glslang::EOpDivAssign:
        if (isFloat)
            binOp = spv::Op::OpFDiv;
        else if (isUnsigned)
            binOp = spv::Op::OpUDiv;
        else
            binOp = spv::Op::OpSDiv;
        break;
    case glslang::EOpMod:
    case glslang::EOpModAssign:
        if (isFloat)
            binOp = spv::Op::OpFMod;
        else if (isUnsigned)
            binOp = spv::Op::OpUMod;
        else
            binOp = spv::Op::OpSMod;
        break;
    case glslang::EOpRightShift:
    case glslang::EOpRightShiftAssign:
        if (isUnsigned)
            binOp = spv::Op::OpShiftRightLogical;
        else
            binOp = spv::Op::OpShiftRightArithmetic;
        break;
    case glslang::EOpLeftShift:
    case glslang::EOpLeftShiftAssign:
        binOp = spv::Op::OpShiftLeftLogical;
        break;
    case glslang::EOpAnd:
    case glslang::EOpAndAssign:
        binOp = spv::Op::OpBitwiseAnd;
        break;
    case glslang::EOpLogicalAnd:
        needMatchingVectors = false;
        binOp = spv::Op::OpLogicalAnd;
        break;
    case glslang::EOpInclusiveOr:
    case glslang::EOpInclusiveOrAssign:
        binOp = spv::Op::OpBitwiseOr;
        break;
    case glslang::EOpLogicalOr:
        needMatchingVectors = false;
        binOp = spv::Op::OpLogicalOr;
        break;
    case glslang::EOpExclusiveOr:
    case glslang::EOpExclusiveOrAssign:
        binOp = spv::Op::OpBitwiseXor;
        break;
    case glslang::EOpLogicalXor:
        needMatchingVectors = false;
        binOp = spv::Op::OpLogicalNotEqual;
        break;

    case glslang::EOpAbsDifference:
        binOp = isUnsigned ? spv::Op::OpAbsUSubINTEL : spv::Op::OpAbsISubINTEL;
        break;

    case glslang::EOpAddSaturate:
        binOp = isUnsigned ? spv::Op::OpUAddSatINTEL : spv::Op::OpIAddSatINTEL;
        break;

    case glslang::EOpSubSaturate:
        binOp = isUnsigned ? spv::Op::OpUSubSatINTEL : spv::Op::OpISubSatINTEL;
        break;

    case glslang::EOpAverage:
        binOp = isUnsigned ? spv::Op::OpUAverageINTEL : spv::Op::OpIAverageINTEL;
        break;

    case glslang::EOpAverageRounded:
        binOp = isUnsigned ? spv::Op::OpUAverageRoundedINTEL : spv::Op::OpIAverageRoundedINTEL;
        break;

    case glslang::EOpMul32x16:
        binOp = isUnsigned ? spv::Op::OpUMul32x16INTEL : spv::Op::OpIMul32x16INTEL;
        break;

    case glslang::EOpExpectEXT:
        binOp = spv::Op::OpExpectKHR;
        break;

    case glslang::EOpLessThan:
    case glslang::EOpGreaterThan:
    case glslang::EOpLessThanEqual:
    case glslang::EOpGreaterThanEqual:
    case glslang::EOpEqual:
    case glslang::EOpNotEqual:
    case glslang::EOpVectorEqual:
    case glslang::EOpVectorNotEqual:
        comparison = true;
        break;
    default:
        break;
    }

    // handle mapped binary operations (should be non-comparison)
    if (binOp != spv::Op::OpNop) {
        assert(comparison == false);
        if (builder.isMatrix(left) || builder.isMatrix(right) ||
            builder.isCooperativeMatrix(left) || builder.isCooperativeMatrix(right))
            return createBinaryMatrixOperation(binOp, decorations, typeId, left, right);

        // No matrix involved; make both operands be the same number of components, if needed
        if (needMatchingVectors)
            builder.promoteScalar(decorations.precision, left, right);

        spv::Id result = builder.createBinOp(binOp, typeId, left, right);
        decorations.addNoContraction(builder, result);
        decorations.addNonUniform(builder, result);
        return builder.setPrecision(result, decorations.precision);
    }

    if (! comparison)
        return 0;

    // Handle comparison instructions

    if (reduceComparison && (op == glslang::EOpEqual || op == glslang::EOpNotEqual)
                         && (builder.isVector(left) || builder.isMatrix(left) || builder.isAggregate(left))) {
        spv::Id result = builder.createCompositeCompare(decorations.precision, left, right, op == glslang::EOpEqual);
        decorations.addNonUniform(builder, result);
        return result;
    }

    switch (op) {
    case glslang::EOpLessThan:
        if (isFloat)
            binOp = spv::Op::OpFOrdLessThan;
        else if (isUnsigned)
            binOp = spv::Op::OpULessThan;
        else
            binOp = spv::Op::OpSLessThan;
        break;
    case glslang::EOpGreaterThan:
        if (isFloat)
            binOp = spv::Op::OpFOrdGreaterThan;
        else if (isUnsigned)
            binOp = spv::Op::OpUGreaterThan;
        else
            binOp = spv::Op::OpSGreaterThan;
        break;
    case glslang::EOpLessThanEqual:
        if (isFloat)
            binOp = spv::Op::OpFOrdLessThanEqual;
        else if (isUnsigned)
            binOp = spv::Op::OpULessThanEqual;
        else
            binOp = spv::Op::OpSLessThanEqual;
        break;
    case glslang::EOpGreaterThanEqual:
        if (isFloat)
            binOp = spv::Op::OpFOrdGreaterThanEqual;
        else if (isUnsigned)
            binOp = spv::Op::OpUGreaterThanEqual;
        else
            binOp = spv::Op::OpSGreaterThanEqual;
        break;
    case glslang::EOpEqual:
    case glslang::EOpVectorEqual:
        if (isFloat)
            binOp = spv::Op::OpFOrdEqual;
        else if (isBool)
            binOp = spv::Op::OpLogicalEqual;
        else
            binOp = spv::Op::OpIEqual;
        break;
    case glslang::EOpNotEqual:
    case glslang::EOpVectorNotEqual:
        if (isFloat)
            binOp = spv::Op::OpFUnordNotEqual;
        else if (isBool)
            binOp = spv::Op::OpLogicalNotEqual;
        else
            binOp = spv::Op::OpINotEqual;
        break;
    default:
        break;
    }

    if (binOp != spv::Op::OpNop) {
        spv::Id result = builder.createBinOp(binOp, typeId, left, right);
        decorations.addNoContraction(builder, result);
        decorations.addNonUniform(builder, result);
        return builder.setPrecision(result, decorations.precision);
    }

    return 0;
}

//
// Translate AST matrix operation to SPV operation, already having SPV-based operands/types.
// These can be any of:
//
//   matrix * scalar
//   scalar * matrix
//   matrix * matrix     linear algebraic
//   matrix * vector
//   vector * matrix
//   matrix * matrix     componentwise
//   matrix op matrix    op in {+, -, /}
//   matrix op scalar    op in {+, -, /}
//   scalar op matrix    op in {+, -, /}
//
spv::Id TGlslangToSpvTraverser::createBinaryMatrixOperation(spv::Op op, OpDecorations& decorations, spv::Id typeId,
                                                            spv::Id left, spv::Id right)
{
    bool firstClass = true;

    // First, handle first-class matrix operations (* and matrix/scalar)
    switch (op) {
    case spv::Op::OpFDiv:
        if (builder.isMatrix(left) && builder.isScalar(right)) {
            // turn matrix / scalar into a multiply...
            spv::Id resultType = builder.getTypeId(right);
            right = builder.createBinOp(spv::Op::OpFDiv, resultType, builder.makeFpConstant(resultType, 1.0), right);
            op = spv::Op::OpMatrixTimesScalar;
        } else
            firstClass = false;
        break;
    case spv::Op::OpMatrixTimesScalar:
        if (builder.isMatrix(right) || builder.isCooperativeMatrix(right))
            std::swap(left, right);
        assert(builder.isScalar(right));
        break;
    case spv::Op::OpVectorTimesMatrix:
        assert(builder.isVector(left));
        assert(builder.isMatrix(right));
        break;
    case spv::Op::OpMatrixTimesVector:
        assert(builder.isMatrix(left));
        assert(builder.isVector(right));
        break;
    case spv::Op::OpMatrixTimesMatrix:
        assert(builder.isMatrix(left));
        assert(builder.isMatrix(right));
        break;
    default:
        firstClass = false;
        break;
    }

    if (builder.isCooperativeMatrix(left) || builder.isCooperativeMatrix(right))
        firstClass = true;

    if (firstClass) {
        spv::Id result = builder.createBinOp(op, typeId, left, right);
        decorations.addNoContraction(builder, result);
        decorations.addNonUniform(builder, result);
        return builder.setPrecision(result, decorations.precision);
    }

    // Handle component-wise +, -, *, %, and / for all combinations of type.
    // The result type of all of them is the same type as the (a) matrix operand.
    // The algorithm is to:
    //   - break the matrix(es) into vectors
    //   - smear any scalar to a vector
    //   - do vector operations
    //   - make a matrix out the vector results
    switch (op) {
    case spv::Op::OpFAdd:
    case spv::Op::OpFSub:
    case spv::Op::OpFDiv:
    case spv::Op::OpFMod:
    case spv::Op::OpFMul:
    {
        // one time set up...
        bool  leftMat = builder.isMatrix(left);
        bool rightMat = builder.isMatrix(right);
        unsigned int numCols = leftMat ? builder.getNumColumns(left) : builder.getNumColumns(right);
        int numRows = leftMat ? builder.getNumRows(left) : builder.getNumRows(right);
        spv::Id scalarType = builder.getScalarTypeId(typeId);
        spv::Id vecType = builder.makeVectorType(scalarType, numRows);
        std::vector<spv::Id> results;
        spv::Id smearVec = spv::NoResult;
        if (builder.isScalar(left))
            smearVec = builder.smearScalar(decorations.precision, left, vecType);
        else if (builder.isScalar(right))
            smearVec = builder.smearScalar(decorations.precision, right, vecType);

        // do each vector op
        for (unsigned int c = 0; c < numCols; ++c) {
            std::vector<unsigned int> indexes;
            indexes.push_back(c);
            spv::Id  leftVec =  leftMat ? builder.createCompositeExtract( left, vecType, indexes) : smearVec;
            spv::Id rightVec = rightMat ? builder.createCompositeExtract(right, vecType, indexes) : smearVec;
            spv::Id result = builder.createBinOp(op, vecType, leftVec, rightVec);
            decorations.addNoContraction(builder, result);
            decorations.addNonUniform(builder, result);
            results.push_back(builder.setPrecision(result, decorations.precision));
        }

        // put the pieces together
        spv::Id result = builder.setPrecision(builder.createCompositeConstruct(typeId, results), decorations.precision);
        decorations.addNonUniform(builder, result);
        return result;
    }
    default:
        assert(0);
        return spv::NoResult;
    }
}

spv::Id TGlslangToSpvTraverser::createUnaryOperation(glslang::TOperator op, OpDecorations& decorations, spv::Id typeId,
    spv::Id operand, glslang::TBasicType typeProxy, const spv::Builder::AccessChain::CoherentFlags &lvalueCoherentFlags,
    const glslang::TType &opType)
{
    spv::Op unaryOp = spv::Op::OpNop;
    int extBuiltins = -1;
    int libCall = -1;
    bool isUnsigned = isTypeUnsignedInt(typeProxy);
    bool isFloat = isTypeFloat(typeProxy);

    switch (op) {
    case glslang::EOpNegative:
        if (isFloat) {
            unaryOp = spv::Op::OpFNegate;
            if (builder.isMatrixType(typeId))
                return createUnaryMatrixOperation(unaryOp, decorations, typeId, operand, typeProxy);
        } else
            unaryOp = spv::Op::OpSNegate;
        break;

    case glslang::EOpLogicalNot:
    case glslang::EOpVectorLogicalNot:
        unaryOp = spv::Op::OpLogicalNot;
        break;
    case glslang::EOpBitwiseNot:
        unaryOp = spv::Op::OpNot;
        break;

    case glslang::EOpDeterminant:
        libCall = spv::GLSLstd450Determinant;
        break;
    case glslang::EOpMatrixInverse:
        libCall = spv::GLSLstd450MatrixInverse;
        break;
    case glslang::EOpTranspose:
        unaryOp = spv::Op::OpTranspose;
        break;

    case glslang::EOpRadians:
        libCall = spv::GLSLstd450Radians;
        break;
    case glslang::EOpDegrees:
        libCall = spv::GLSLstd450Degrees;
        break;
    case glslang::EOpSin:
        libCall = spv::GLSLstd450Sin;
        break;
    case glslang::EOpCos:
        libCall = spv::GLSLstd450Cos;
        break;
    case glslang::EOpTan:
        libCall = spv::GLSLstd450Tan;
        break;
    case glslang::EOpAcos:
        libCall = spv::GLSLstd450Acos;
        break;
    case glslang::EOpAsin:
        libCall = spv::GLSLstd450Asin;
        break;
    case glslang::EOpAtan:
        libCall = spv::GLSLstd450Atan;
        break;

    case glslang::EOpAcosh:
        libCall = spv::GLSLstd450Acosh;
        break;
    case glslang::EOpAsinh:
        libCall = spv::GLSLstd450Asinh;
        break;
    case glslang::EOpAtanh:
        libCall = spv::GLSLstd450Atanh;
        break;
    case glslang::EOpTanh:
        libCall = spv::GLSLstd450Tanh;
        break;
    case glslang::EOpCosh:
        libCall = spv::GLSLstd450Cosh;
        break;
    case glslang::EOpSinh:
        libCall = spv::GLSLstd450Sinh;
        break;

    case glslang::EOpLength:
        libCall = spv::GLSLstd450Length;
        break;
    case glslang::EOpNormalize:
        libCall = spv::GLSLstd450Normalize;
        break;

    case glslang::EOpExp:
        libCall = spv::GLSLstd450Exp;
        break;
    case glslang::EOpLog:
        libCall = spv::GLSLstd450Log;
        break;
    case glslang::EOpExp2:
        libCall = spv::GLSLstd450Exp2;
        break;
    case glslang::EOpLog2:
        libCall = spv::GLSLstd450Log2;
        break;
    case glslang::EOpSqrt:
        libCall = spv::GLSLstd450Sqrt;
        break;
    case glslang::EOpInverseSqrt:
        libCall = spv::GLSLstd450InverseSqrt;
        break;

    case glslang::EOpFloor:
        libCall = spv::GLSLstd450Floor;
        break;
    case glslang::EOpTrunc:
        libCall = spv::GLSLstd450Trunc;
        break;
    case glslang::EOpRound:
        libCall = spv::GLSLstd450Round;
        break;
    case glslang::EOpRoundEven:
        libCall = spv::GLSLstd450RoundEven;
        break;
    case glslang::EOpCeil:
        libCall = spv::GLSLstd450Ceil;
        break;
    case glslang::EOpFract:
        libCall = spv::GLSLstd450Fract;
        break;

    case glslang::EOpIsNan:
        unaryOp = spv::Op::OpIsNan;
        break;
    case glslang::EOpIsInf:
        unaryOp = spv::Op::OpIsInf;
        break;
    case glslang::EOpIsFinite:
        unaryOp = spv::Op::OpIsFinite;
        break;

    case glslang::EOpFloatBitsToInt:
    case glslang::EOpFloatBitsToUint:
    case glslang::EOpIntBitsToFloat:
    case glslang::EOpUintBitsToFloat:
    case glslang::EOpDoubleBitsToInt64:
    case glslang::EOpDoubleBitsToUint64:
    case glslang::EOpInt64BitsToDouble:
    case glslang::EOpUint64BitsToDouble:
    case glslang::EOpFloat16BitsToInt16:
    case glslang::EOpFloat16BitsToUint16:
    case glslang::EOpInt16BitsToFloat16:
    case glslang::EOpUint16BitsToFloat16:
        unaryOp = spv::Op::OpBitcast;
        break;

    case glslang::EOpPackSnorm2x16:
        libCall = spv::GLSLstd450PackSnorm2x16;
        break;
    case glslang::EOpUnpackSnorm2x16:
        libCall = spv::GLSLstd450UnpackSnorm2x16;
        break;
    case glslang::EOpPackUnorm2x16:
        libCall = spv::GLSLstd450PackUnorm2x16;
        break;
    case glslang::EOpUnpackUnorm2x16:
        libCall = spv::GLSLstd450UnpackUnorm2x16;
        break;
    case glslang::EOpPackHalf2x16:
        libCall = spv::GLSLstd450PackHalf2x16;
        break;
    case glslang::EOpUnpackHalf2x16:
        libCall = spv::GLSLstd450UnpackHalf2x16;
        break;
    case glslang::EOpPackSnorm4x8:
        libCall = spv::GLSLstd450PackSnorm4x8;
        break;
    case glslang::EOpUnpackSnorm4x8:
        libCall = spv::GLSLstd450UnpackSnorm4x8;
        break;
    case glslang::EOpPackUnorm4x8:
        libCall = spv::GLSLstd450PackUnorm4x8;
        break;
    case glslang::EOpUnpackUnorm4x8:
        libCall = spv::GLSLstd450UnpackUnorm4x8;
        break;
    case glslang::EOpPackDouble2x32:
        libCall = spv::GLSLstd450PackDouble2x32;
        break;
    case glslang::EOpUnpackDouble2x32:
        libCall = spv::GLSLstd450UnpackDouble2x32;
        break;

    case glslang::EOpPackInt2x32:
    case glslang::EOpUnpackInt2x32:
    case glslang::EOpPackUint2x32:
    case glslang::EOpUnpackUint2x32:
    case glslang::EOpPack16:
    case glslang::EOpPack32:
    case glslang::EOpPack64:
    case glslang::EOpUnpack32:
    case glslang::EOpUnpack16:
    case glslang::EOpUnpack8:
    case glslang::EOpPackInt2x16:
    case glslang::EOpUnpackInt2x16:
    case glslang::EOpPackUint2x16:
    case glslang::EOpUnpackUint2x16:
    case glslang::EOpPackInt4x16:
    case glslang::EOpUnpackInt4x16:
    case glslang::EOpPackUint4x16:
    case glslang::EOpUnpackUint4x16:
    case glslang::EOpPackFloat2x16:
    case glslang::EOpUnpackFloat2x16:
        unaryOp = spv::Op::OpBitcast;
        break;

    case glslang::EOpDPdx:
        unaryOp = spv::Op::OpDPdx;
        break;
    case glslang::EOpDPdy:
        unaryOp = spv::Op::OpDPdy;
        break;
    case glslang::EOpFwidth:
        unaryOp = spv::Op::OpFwidth;
        break;

    case glslang::EOpAny:
        unaryOp = spv::Op::OpAny;
        break;
    case glslang::EOpAll:
        unaryOp = spv::Op::OpAll;
        break;

    case glslang::EOpAbs:
        if (isFloat)
            libCall = spv::GLSLstd450FAbs;
        else
            libCall = spv::GLSLstd450SAbs;
        break;
    case glslang::EOpSign:
        if (isFloat)
            libCall = spv::GLSLstd450FSign;
        else
            libCall = spv::GLSLstd450SSign;
        break;

    case glslang::EOpDPdxFine:
        unaryOp = spv::Op::OpDPdxFine;
        break;
    case glslang::EOpDPdyFine:
        unaryOp = spv::Op::OpDPdyFine;
        break;
    case glslang::EOpFwidthFine:
        unaryOp = spv::Op::OpFwidthFine;
        break;
    case glslang::EOpDPdxCoarse:
        unaryOp = spv::Op::OpDPdxCoarse;
        break;
    case glslang::EOpDPdyCoarse:
        unaryOp = spv::Op::OpDPdyCoarse;
        break;
    case glslang::EOpFwidthCoarse:
        unaryOp = spv::Op::OpFwidthCoarse;
        break;
    case glslang::EOpRayQueryProceed:
        unaryOp = spv::Op::OpRayQueryProceedKHR;
        break;
    case glslang::EOpRayQueryGetRayTMin:
        unaryOp = spv::Op::OpRayQueryGetRayTMinKHR;
        break;
    case glslang::EOpRayQueryGetRayFlags:
        unaryOp = spv::Op::OpRayQueryGetRayFlagsKHR;
        break;
    case glslang::EOpRayQueryGetWorldRayOrigin:
        unaryOp = spv::Op::OpRayQueryGetWorldRayOriginKHR;
        break;
    case glslang::EOpRayQueryGetWorldRayDirection:
        unaryOp = spv::Op::OpRayQueryGetWorldRayDirectionKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionCandidateAABBOpaque:
        unaryOp = spv::Op::OpRayQueryGetIntersectionCandidateAABBOpaqueKHR;
        break;
    case glslang::EOpInterpolateAtCentroid:
        if (typeProxy == glslang::EbtFloat16)
            builder.addExtension(spv::E_SPV_AMD_gpu_shader_half_float);
        libCall = spv::GLSLstd450InterpolateAtCentroid;
        break;
    case glslang::EOpAtomicCounterIncrement:
    case glslang::EOpAtomicCounterDecrement:
    case glslang::EOpAtomicCounter:
    {
        // Handle all of the atomics in one place, in createAtomicOperation()
        std::vector<spv::Id> operands;
        operands.push_back(operand);
        return createAtomicOperation(op, decorations.precision, typeId, operands, typeProxy, lvalueCoherentFlags, opType);
    }

    case glslang::EOpBitFieldReverse:
        unaryOp = spv::Op::OpBitReverse;
        break;
    case glslang::EOpBitCount:
        unaryOp = spv::Op::OpBitCount;
        break;
    case glslang::EOpFindLSB:
        libCall = spv::GLSLstd450FindILsb;
        break;
    case glslang::EOpFindMSB:
        if (isUnsigned)
            libCall = spv::GLSLstd450FindUMsb;
        else
            libCall = spv::GLSLstd450FindSMsb;
        break;

    case glslang::EOpCountLeadingZeros:
        builder.addCapability(spv::Capability::IntegerFunctions2INTEL);
        builder.addExtension("SPV_INTEL_shader_integer_functions2");
        unaryOp = spv::Op::OpUCountLeadingZerosINTEL;
        break;

    case glslang::EOpCountTrailingZeros:
        builder.addCapability(spv::Capability::IntegerFunctions2INTEL);
        builder.addExtension("SPV_INTEL_shader_integer_functions2");
        unaryOp = spv::Op::OpUCountTrailingZerosINTEL;
        break;

    case glslang::EOpBallot:
    case glslang::EOpReadFirstInvocation:
    case glslang::EOpAnyInvocation:
    case glslang::EOpAllInvocations:
    case glslang::EOpAllInvocationsEqual:
    case glslang::EOpMinInvocations:
    case glslang::EOpMaxInvocations:
    case glslang::EOpAddInvocations:
    case glslang::EOpMinInvocationsNonUniform:
    case glslang::EOpMaxInvocationsNonUniform:
    case glslang::EOpAddInvocationsNonUniform:
    case glslang::EOpMinInvocationsInclusiveScan:
    case glslang::EOpMaxInvocationsInclusiveScan:
    case glslang::EOpAddInvocationsInclusiveScan:
    case glslang::EOpMinInvocationsInclusiveScanNonUniform:
    case glslang::EOpMaxInvocationsInclusiveScanNonUniform:
    case glslang::EOpAddInvocationsInclusiveScanNonUniform:
    case glslang::EOpMinInvocationsExclusiveScan:
    case glslang::EOpMaxInvocationsExclusiveScan:
    case glslang::EOpAddInvocationsExclusiveScan:
    case glslang::EOpMinInvocationsExclusiveScanNonUniform:
    case glslang::EOpMaxInvocationsExclusiveScanNonUniform:
    case glslang::EOpAddInvocationsExclusiveScanNonUniform:
    {
        std::vector<spv::Id> operands;
        operands.push_back(operand);
        return createInvocationsOperation(op, typeId, operands, typeProxy);
    }
    case glslang::EOpSubgroupAll:
    case glslang::EOpSubgroupAny:
    case glslang::EOpSubgroupAllEqual:
    case glslang::EOpSubgroupBroadcastFirst:
    case glslang::EOpSubgroupBallot:
    case glslang::EOpSubgroupInverseBallot:
    case glslang::EOpSubgroupBallotBitCount:
    case glslang::EOpSubgroupBallotInclusiveBitCount:
    case glslang::EOpSubgroupBallotExclusiveBitCount:
    case glslang::EOpSubgroupBallotFindLSB:
    case glslang::EOpSubgroupBallotFindMSB:
    case glslang::EOpSubgroupAdd:
    case glslang::EOpSubgroupMul:
    case glslang::EOpSubgroupMin:
    case glslang::EOpSubgroupMax:
    case glslang::EOpSubgroupAnd:
    case glslang::EOpSubgroupOr:
    case glslang::EOpSubgroupXor:
    case glslang::EOpSubgroupInclusiveAdd:
    case glslang::EOpSubgroupInclusiveMul:
    case glslang::EOpSubgroupInclusiveMin:
    case glslang::EOpSubgroupInclusiveMax:
    case glslang::EOpSubgroupInclusiveAnd:
    case glslang::EOpSubgroupInclusiveOr:
    case glslang::EOpSubgroupInclusiveXor:
    case glslang::EOpSubgroupExclusiveAdd:
    case glslang::EOpSubgroupExclusiveMul:
    case glslang::EOpSubgroupExclusiveMin:
    case glslang::EOpSubgroupExclusiveMax:
    case glslang::EOpSubgroupExclusiveAnd:
    case glslang::EOpSubgroupExclusiveOr:
    case glslang::EOpSubgroupExclusiveXor:
    case glslang::EOpSubgroupQuadSwapHorizontal:
    case glslang::EOpSubgroupQuadSwapVertical:
    case glslang::EOpSubgroupQuadSwapDiagonal:
    case glslang::EOpSubgroupQuadAll:
    case glslang::EOpSubgroupQuadAny: {
        std::vector<spv::Id> operands;
        operands.push_back(operand);
        return createSubgroupOperation(op, typeId, operands, typeProxy);
    }
    case glslang::EOpMbcnt:
        extBuiltins = getExtBuiltins(spv::E_SPV_AMD_shader_ballot);
        libCall = spv::MbcntAMD;
        break;

    case glslang::EOpCubeFaceIndex:
        extBuiltins = getExtBuiltins(spv::E_SPV_AMD_gcn_shader);
        libCall = spv::CubeFaceIndexAMD;
        break;

    case glslang::EOpCubeFaceCoord:
        extBuiltins = getExtBuiltins(spv::E_SPV_AMD_gcn_shader);
        libCall = spv::CubeFaceCoordAMD;
        break;
    case glslang::EOpSubgroupPartition:
        unaryOp = spv::Op::OpGroupNonUniformPartitionNV;
        break;
    case glslang::EOpConstructReference:
        unaryOp = spv::Op::OpBitcast;
        break;

    case glslang::EOpConvUint64ToAccStruct:
    case glslang::EOpConvUvec2ToAccStruct:
        unaryOp = spv::Op::OpConvertUToAccelerationStructureKHR;
        break;

    case glslang::EOpHitObjectIsEmptyNV:
        unaryOp = spv::Op::OpHitObjectIsEmptyNV;
        break;

    case glslang::EOpHitObjectIsEmptyEXT:
        unaryOp = spv::Op::OpHitObjectIsEmptyEXT;
        break;

    case glslang::EOpHitObjectIsMissNV:
        unaryOp = spv::Op::OpHitObjectIsMissNV;
        break;

    case glslang::EOpHitObjectIsMissEXT:
        unaryOp = spv::Op::OpHitObjectIsMissEXT;
        break;

    case glslang::EOpHitObjectIsHitNV:
        unaryOp = spv::Op::OpHitObjectIsHitNV;
        break;

    case glslang::EOpHitObjectIsHitEXT:
        unaryOp = spv::Op::OpHitObjectIsHitEXT;
        break;

    case glslang::EOpHitObjectGetObjectRayOriginNV:
        unaryOp = spv::Op::OpHitObjectGetObjectRayOriginNV;
        break;

    case glslang::EOpHitObjectGetObjectRayOriginEXT:
        unaryOp = spv::Op::OpHitObjectGetObjectRayOriginEXT;
        break;

    case glslang::EOpHitObjectGetObjectRayDirectionNV:
        unaryOp = spv::Op::OpHitObjectGetObjectRayDirectionNV;
        break;

    case glslang::EOpHitObjectGetObjectRayDirectionEXT:
        unaryOp = spv::Op::OpHitObjectGetObjectRayDirectionEXT;
        break;

    case glslang::EOpHitObjectGetWorldRayOriginNV:
        unaryOp = spv::Op::OpHitObjectGetWorldRayOriginNV;
        break;

    case glslang::EOpHitObjectGetWorldRayOriginEXT:
        unaryOp = spv::Op::OpHitObjectGetWorldRayOriginEXT;
        break;

    case glslang::EOpHitObjectGetWorldRayDirectionNV:
        unaryOp = spv::Op::OpHitObjectGetWorldRayDirectionNV;
        break;

    case glslang::EOpHitObjectGetWorldRayDirectionEXT:
        unaryOp = spv::Op::OpHitObjectGetWorldRayDirectionEXT;
        break;

    case glslang::EOpHitObjectGetObjectToWorldNV:
        unaryOp = spv::Op::OpHitObjectGetObjectToWorldNV;
        break;

    case glslang::EOpHitObjectGetObjectToWorldEXT:
        unaryOp = spv::Op::OpHitObjectGetObjectToWorldEXT;
        break;

    case glslang::EOpHitObjectGetWorldToObjectNV:
        unaryOp = spv::Op::OpHitObjectGetWorldToObjectNV;
        break;

    case glslang::EOpHitObjectGetWorldToObjectEXT:
        unaryOp = spv::Op::OpHitObjectGetWorldToObjectEXT;
        break;

    case glslang::EOpHitObjectGetRayTMinNV:
        unaryOp = spv::Op::OpHitObjectGetRayTMinNV;
        break;

    case glslang::EOpHitObjectGetRayTMinEXT:
        unaryOp = spv::Op::OpHitObjectGetRayTMinEXT;
        break;

    case glslang::EOpHitObjectGetRayTMaxNV:
        unaryOp = spv::Op::OpHitObjectGetRayTMaxNV;
        break;

    case glslang::EOpHitObjectGetRayTMaxEXT:
        unaryOp = spv::Op::OpHitObjectGetRayTMaxEXT;
        break;

    case glslang::EOpHitObjectGetRayFlagsEXT:
        unaryOp = spv::Op::OpHitObjectGetRayFlagsEXT;
        break;

    case glslang::EOpHitObjectGetPrimitiveIndexNV:
        unaryOp = spv::Op::OpHitObjectGetPrimitiveIndexNV;
        break;

    case glslang::EOpHitObjectGetPrimitiveIndexEXT:
        unaryOp = spv::Op::OpHitObjectGetPrimitiveIndexEXT;
        break;

    case glslang::EOpHitObjectGetInstanceIdNV:
        unaryOp = spv::Op::OpHitObjectGetInstanceIdNV;
        break;

    case glslang::EOpHitObjectGetInstanceIdEXT:
        unaryOp = spv::Op::OpHitObjectGetInstanceIdEXT;
        break;

    case glslang::EOpHitObjectGetInstanceCustomIndexNV:
        unaryOp = spv::Op::OpHitObjectGetInstanceCustomIndexNV;
        break;

    case glslang::EOpHitObjectGetInstanceCustomIndexEXT:
        unaryOp = spv::Op::OpHitObjectGetInstanceCustomIndexEXT;
        break;

    case glslang::EOpHitObjectGetGeometryIndexNV:
        unaryOp = spv::Op::OpHitObjectGetGeometryIndexNV;
        break;

    case glslang::EOpHitObjectGetGeometryIndexEXT:
        unaryOp = spv::Op::OpHitObjectGetGeometryIndexEXT;
        break;

    case glslang::EOpHitObjectGetHitKindNV:
        unaryOp = spv::Op::OpHitObjectGetHitKindNV;
        break;

    case glslang::EOpHitObjectGetHitKindEXT:
        unaryOp = spv::Op::OpHitObjectGetHitKindEXT;
        break;

    case glslang::EOpHitObjectGetCurrentTimeNV:
        unaryOp = spv::Op::OpHitObjectGetCurrentTimeNV;
        break;

    case glslang::EOpHitObjectGetCurrentTimeEXT:
        unaryOp = spv::Op::OpHitObjectGetCurrentTimeEXT;
        break;

    case glslang::EOpHitObjectGetShaderBindingTableRecordIndexNV:
        unaryOp = spv::Op::OpHitObjectGetShaderBindingTableRecordIndexNV;
        break;

    case glslang::EOpHitObjectGetShaderBindingTableRecordIndexEXT:
        unaryOp = spv::Op::OpHitObjectGetShaderBindingTableRecordIndexEXT;
        break;

    case glslang::EOpHitObjectGetShaderRecordBufferHandleNV:
        unaryOp = spv::Op::OpHitObjectGetShaderRecordBufferHandleNV;
        break;

    case glslang::EOpHitObjectGetClusterIdNV:
        unaryOp = spv::Op::OpHitObjectGetClusterIdNV;
        builder.addExtension(spv::E_SPV_NV_cluster_acceleration_structure);
        builder.addCapability(spv::Capability::ShaderInvocationReorderNV);
        builder.addCapability(spv::Capability::RayTracingClusterAccelerationStructureNV);
        break;

    case glslang::EOpHitObjectGetSpherePositionNV:
        unaryOp = spv::Op::OpHitObjectGetSpherePositionNV;
        builder.addExtension(spv::E_SPV_NV_linear_swept_spheres);
        builder.addCapability(spv::Capability::ShaderInvocationReorderNV);
        builder.addCapability(spv::Capability::RayTracingSpheresGeometryNV);
        break;

    case glslang::EOpHitObjectGetSphereRadiusNV:
        unaryOp = spv::Op::OpHitObjectGetSphereRadiusNV;
        builder.addExtension(spv::E_SPV_NV_linear_swept_spheres);
        builder.addCapability(spv::Capability::ShaderInvocationReorderNV);
        builder.addCapability(spv::Capability::RayTracingSpheresGeometryNV);
        break;

    case glslang::EOpHitObjectIsSphereHitNV:
        unaryOp = spv::Op::OpHitObjectIsSphereHitNV;
        builder.addExtension(spv::E_SPV_NV_linear_swept_spheres);
        builder.addCapability(spv::Capability::ShaderInvocationReorderNV);
        builder.addCapability(spv::Capability::RayTracingSpheresGeometryNV);
        break;

    case glslang::EOpHitObjectIsLSSHitNV:
        unaryOp = spv::Op::OpHitObjectIsLSSHitNV;
        builder.addExtension(spv::E_SPV_NV_linear_swept_spheres);
        builder.addCapability(spv::Capability::ShaderInvocationReorderNV);
        builder.addCapability(spv::Capability::RayTracingLinearSweptSpheresGeometryNV);
        break;

    case glslang::EOpHitObjectGetShaderRecordBufferHandleEXT:
        unaryOp = spv::Op::OpHitObjectGetShaderRecordBufferHandleEXT;
        break;

    case glslang::EOpFetchMicroTriangleVertexPositionNV:
        unaryOp = spv::Op::OpFetchMicroTriangleVertexPositionNV;
        break;

    case glslang::EOpFetchMicroTriangleVertexBarycentricNV:
        unaryOp = spv::Op::OpFetchMicroTriangleVertexBarycentricNV;
        break;

    case glslang::EOpCopyObject:
        unaryOp = spv::Op::OpCopyObject;
        break;

    case glslang::EOpDepthAttachmentReadEXT:
        builder.addExtension(spv::E_SPV_EXT_shader_tile_image);
        builder.addCapability(spv::Capability::TileImageDepthReadAccessEXT);
        unaryOp = spv::Op::OpDepthAttachmentReadEXT;
        decorations.precision = spv::NoPrecision;
        break;
    case glslang::EOpStencilAttachmentReadEXT:
        builder.addExtension(spv::E_SPV_EXT_shader_tile_image);
        builder.addCapability(spv::Capability::TileImageStencilReadAccessEXT);
        unaryOp = spv::Op::OpStencilAttachmentReadEXT;
        decorations.precision = spv::Decoration::RelaxedPrecision;
        break;

    default:
        return 0;
    }

    spv::Id id;
    if (libCall >= 0) {
        std::vector<spv::Id> args;
        args.push_back(operand);
        id = builder.createBuiltinCall(typeId, extBuiltins >= 0 ? extBuiltins : stdBuiltins, libCall, args);
    } else {
        id = builder.createUnaryOp(unaryOp, typeId, operand);
    }

    decorations.addNoContraction(builder, id);
    decorations.addNonUniform(builder, id);
    return builder.setPrecision(id, decorations.precision);
}

// Create a unary operation on a matrix
spv::Id TGlslangToSpvTraverser::createUnaryMatrixOperation(spv::Op op, OpDecorations& decorations, spv::Id typeId,
                                                           spv::Id operand, glslang::TBasicType /* typeProxy */)
{
    // Handle unary operations vector by vector.
    // The result type is the same type as the original type.
    // The algorithm is to:
    //   - break the matrix into vectors
    //   - apply the operation to each vector
    //   - make a matrix out the vector results

    // get the types sorted out
    int numCols = builder.getNumColumns(operand);
    int numRows = builder.getNumRows(operand);
    spv::Id srcVecType  = builder.makeVectorType(builder.getScalarTypeId(builder.getTypeId(operand)), numRows);
    spv::Id destVecType = builder.makeVectorType(builder.getScalarTypeId(typeId), numRows);
    std::vector<spv::Id> results;

    // do each vector op
    for (int c = 0; c < numCols; ++c) {
        std::vector<unsigned int> indexes;
        indexes.push_back(c);
        spv::Id srcVec  = builder.createCompositeExtract(operand, srcVecType, indexes);
        spv::Id destVec = builder.createUnaryOp(op, destVecType, srcVec);
        decorations.addNoContraction(builder, destVec);
        decorations.addNonUniform(builder, destVec);
        results.push_back(builder.setPrecision(destVec, decorations.precision));
    }

    // put the pieces together
    spv::Id result = builder.setPrecision(builder.createCompositeConstruct(typeId, results), decorations.precision);
    decorations.addNonUniform(builder, result);
    return result;
}

// For converting integers where both the bitwidth and the signedness could
// change, but only do the width change here. The caller is still responsible
// for the signedness conversion.
// destType is the final type that will be converted to, but this function
// may only be doing part of that conversion.
spv::Id TGlslangToSpvTraverser::createIntWidthConversion(spv::Id operand, int vectorSize, spv::Id destType,
                                                         glslang::TBasicType resultBasicType, glslang::TBasicType operandBasicType)
{
    // Get the result type width, based on the type to convert to.
    int width = GetNumBits(resultBasicType);

    // Get the conversion operation and result type,
    // based on the target width, but the source type.
    spv::Id type = spv::NoType;
    spv::Op convOp = spv::Op::OpNop;
    if (isTypeSignedInt(operandBasicType)) {
        convOp = spv::Op::OpSConvert;
        type = builder.makeIntType(width);
    } else {
        convOp = spv::Op::OpUConvert;
        type = builder.makeUintType(width);
    }

    if (builder.getOpCode(destType) == spv::Op::OpTypeCooperativeVectorNV) {
        type = builder.makeCooperativeVectorTypeNV(type, builder.getCooperativeVectorNumComponents(destType));
    } else if (vectorSize > 0)
        type = builder.makeVectorType(type, vectorSize);
    else if (builder.getOpCode(destType) == spv::Op::OpTypeCooperativeMatrixKHR ||
             builder.getOpCode(destType) == spv::Op::OpTypeCooperativeMatrixNV) {

        type = builder.makeCooperativeMatrixTypeWithSameShape(type, destType);
    }

    return builder.createUnaryOp(convOp, type, operand);
}

spv::Id TGlslangToSpvTraverser::createConversion(glslang::TOperator op, OpDecorations& decorations, spv::Id destType,
                                                 spv::Id operand, glslang::TBasicType resultBasicType, glslang::TBasicType operandBasicType)
{
    spv::Op convOp = spv::Op::OpNop;
    spv::Id zero = 0;
    spv::Id one = 0;

    int vectorSize = builder.isVectorType(destType) ? builder.getNumTypeComponents(destType) : 0;

    if (IsOpNumericConv(op) || op == glslang::EOpConstructSaturated) {
        if (isTypeSignedInt(operandBasicType) && isTypeFloat(resultBasicType)) {
            convOp = spv::Op::OpConvertSToF;
        }
        if (isTypeUnsignedInt(operandBasicType) && isTypeFloat(resultBasicType)) {
            convOp = spv::Op::OpConvertUToF;
        }
        if (isTypeFloat(operandBasicType) && isTypeSignedInt(resultBasicType)) {
            convOp = spv::Op::OpConvertFToS;
        }
        if (isTypeFloat(operandBasicType) && isTypeUnsignedInt(resultBasicType)) {
            convOp = spv::Op::OpConvertFToU;
        }
        if (isTypeSignedInt(operandBasicType) && isTypeSignedInt(resultBasicType)) {
            convOp = spv::Op::OpSConvert;
        }
        if (isTypeUnsignedInt(operandBasicType) && isTypeUnsignedInt(resultBasicType)) {
            convOp = spv::Op::OpUConvert;
        }
        if (isTypeFloat(operandBasicType) && isTypeFloat(resultBasicType)) {
            convOp = spv::Op::OpFConvert;
            if (builder.isMatrixType(destType))
                return createUnaryMatrixOperation(convOp, decorations, destType, operand, operandBasicType);
        }
        if (isTypeInt(operandBasicType) && isTypeInt(resultBasicType) &&
            isTypeUnsignedInt(operandBasicType) != isTypeUnsignedInt(resultBasicType)) {

            if (GetNumBits(operandBasicType) != GetNumBits(resultBasicType)) {
                // OpSConvert/OpUConvert + OpBitCast
                operand = createIntWidthConversion(operand, vectorSize, destType, resultBasicType, operandBasicType);
            }

            if (builder.isInSpecConstCodeGenMode()) {
                uint32_t bits = GetNumBits(resultBasicType);
                spv::Id zeroType = builder.makeUintType(bits);
                if (bits == 64) {
                    zero = builder.makeInt64Constant(zeroType, 0, false);
                } else {
                    zero = builder.makeIntConstant(zeroType, 0, false);
                }
                zero = makeSmearedConstant(zero, vectorSize);
                // Use OpIAdd, instead of OpBitcast to do the conversion when
                // generating for OpSpecConstantOp instruction.
                return builder.createBinOp(spv::Op::OpIAdd, destType, operand, zero);
            }
            // For normal run-time conversion instruction, use OpBitcast.
            convOp = spv::Op::OpBitcast;
        }
        if (resultBasicType == glslang::EbtBool) {
            uint32_t bits = GetNumBits(operandBasicType);
            if (isTypeInt(operandBasicType)) {
                spv::Id zeroType = builder.makeUintType(bits);
                if (bits == 64) {
                    zero = builder.makeInt64Constant(zeroType, 0, false);
                } else {
                    zero = builder.makeIntConstant(zeroType, 0, false);
                }
                zero = makeSmearedConstant(zero, vectorSize);
                return builder.createBinOp(spv::Op::OpINotEqual, destType, operand, zero);
            } else {
                assert(isTypeFloat(operandBasicType));
                if (bits == 64) {
                    zero = builder.makeDoubleConstant(0.0);
                } else if (bits == 32) {
                    zero = builder.makeFloatConstant(0.0);
                } else {
                    assert(bits == 16);
                    zero = builder.makeFloat16Constant(0.0);
                }
                zero = makeSmearedConstant(zero, vectorSize);
                return builder.createBinOp(spv::Op::OpFUnordNotEqual, destType, operand, zero);
            }
        }
        if (operandBasicType == glslang::EbtBool) {
            uint32_t bits = GetNumBits(resultBasicType);
            convOp = spv::Op::OpSelect;
            if (isTypeInt(resultBasicType)) {
                spv::Id zeroType = isTypeSignedInt(resultBasicType) ? builder.makeIntType(bits) : builder.makeUintType(bits);
                if (bits == 64) {
                    zero = builder.makeInt64Constant(zeroType, 0, false);
                    one = builder.makeInt64Constant(zeroType, 1, false);
                } else {
                    zero = builder.makeIntConstant(zeroType, 0, false);
                    one = builder.makeIntConstant(zeroType, 1, false);
                }
            } else {
                assert(isTypeFloat(resultBasicType));
                if (bits == 64) {
                    zero = builder.makeDoubleConstant(0.0);
                    one = builder.makeDoubleConstant(1.0);
                } else if (bits == 32) {
                    zero = builder.makeFloatConstant(0.0);
                    one = builder.makeFloatConstant(1.0);
                } else {
                    assert(bits == 16);
                    zero = builder.makeFloat16Constant(0.0);
                    one = builder.makeFloat16Constant(1.0);
                }
            }
        }
    }

    if (convOp == spv::Op::OpNop) {
        switch (op) {
        case glslang::EOpConvUint64ToPtr:
            convOp = spv::Op::OpConvertUToPtr;
            break;
        case glslang::EOpConvPtrToUint64:
            convOp = spv::Op::OpConvertPtrToU;
            break;
        case glslang::EOpConvPtrToUvec2:
        case glslang::EOpConvUvec2ToPtr:
            convOp = spv::Op::OpBitcast;
            break;

        default:
            break;
        }
    }

    spv::Id result = 0;
    if (convOp == spv::Op::OpNop)
        return result;

    if (convOp == spv::Op::OpSelect) {
        zero = makeSmearedConstant(zero, vectorSize);
        one  = makeSmearedConstant(one, vectorSize);
        result = builder.createTriOp(convOp, destType, operand, one, zero);
    } else
        result = builder.createUnaryOp(convOp, destType, operand);

    result = builder.setPrecision(result, decorations.precision);
    decorations.addNonUniform(builder, result);
    return result;
}

spv::Id TGlslangToSpvTraverser::makeSmearedConstant(spv::Id constant, int vectorSize)
{
    if (vectorSize == 0)
        return constant;

    spv::Id vectorTypeId = builder.makeVectorType(builder.getTypeId(constant), vectorSize);
    std::vector<spv::Id> components;
    for (int c = 0; c < vectorSize; ++c)
        components.push_back(constant);
    return builder.makeCompositeConstant(vectorTypeId, components);
}

// For glslang ops that map to SPV atomic opCodes
spv::Id TGlslangToSpvTraverser::createAtomicOperation(glslang::TOperator op, spv::Decoration /*precision*/,
    spv::Id typeId, std::vector<spv::Id>& operands, glslang::TBasicType typeProxy,
    const spv::Builder::AccessChain::CoherentFlags &lvalueCoherentFlags, const glslang::TType &opType)
{
    spv::Op opCode = spv::Op::OpNop;

    switch (op) {
    case glslang::EOpAtomicAdd:
    case glslang::EOpImageAtomicAdd:
    case glslang::EOpAtomicCounterAdd:
        opCode = spv::Op::OpAtomicIAdd;
        if (typeProxy == glslang::EbtFloat16 || typeProxy == glslang::EbtFloat || typeProxy == glslang::EbtDouble) {
            opCode = spv::Op::OpAtomicFAddEXT;
            if (typeProxy == glslang::EbtFloat16 &&
                (opType.getVectorSize() == 2 || opType.getVectorSize() == 4)) {
                builder.addExtension(spv::E_SPV_NV_shader_atomic_fp16_vector);
                builder.addCapability(spv::Capability::AtomicFloat16VectorNV);
            } else {
                builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_add);
                if (typeProxy == glslang::EbtFloat16) {
                    builder.addExtension(spv::E_SPV_EXT_shader_atomic_float16_add);
                    builder.addCapability(spv::Capability::AtomicFloat16AddEXT);
                } else if (typeProxy == glslang::EbtFloat) {
                    builder.addCapability(spv::Capability::AtomicFloat32AddEXT);
                } else {
                    builder.addCapability(spv::Capability::AtomicFloat64AddEXT);
                }
            }
        }
        break;
    case glslang::EOpAtomicSubtract:
    case glslang::EOpAtomicCounterSubtract:
        opCode = spv::Op::OpAtomicISub;
        break;
    case glslang::EOpAtomicMin:
    case glslang::EOpImageAtomicMin:
    case glslang::EOpAtomicCounterMin:
        if (typeProxy == glslang::EbtFloat16 || typeProxy == glslang::EbtFloat || typeProxy == glslang::EbtDouble) {
            opCode = spv::Op::OpAtomicFMinEXT;
            if (typeProxy == glslang::EbtFloat16 &&
                (opType.getVectorSize() == 2 || opType.getVectorSize() == 4)) {
                builder.addExtension(spv::E_SPV_NV_shader_atomic_fp16_vector);
                builder.addCapability(spv::Capability::AtomicFloat16VectorNV);
            } else {
                builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_min_max);
                if (typeProxy == glslang::EbtFloat16)
                    builder.addCapability(spv::Capability::AtomicFloat16MinMaxEXT);
                else if (typeProxy == glslang::EbtFloat)
                    builder.addCapability(spv::Capability::AtomicFloat32MinMaxEXT);
                else
                    builder.addCapability(spv::Capability::AtomicFloat64MinMaxEXT);
            }
        } else if (typeProxy == glslang::EbtUint || typeProxy == glslang::EbtUint64) {
            opCode = spv::Op::OpAtomicUMin;
        } else {
            opCode = spv::Op::OpAtomicSMin;
        }
        break;
    case glslang::EOpAtomicMax:
    case glslang::EOpImageAtomicMax:
    case glslang::EOpAtomicCounterMax:
        if (typeProxy == glslang::EbtFloat16 || typeProxy == glslang::EbtFloat || typeProxy == glslang::EbtDouble) {
            opCode = spv::Op::OpAtomicFMaxEXT;
            if (typeProxy == glslang::EbtFloat16 &&
                (opType.getVectorSize() == 2 || opType.getVectorSize() == 4)) {
                builder.addExtension(spv::E_SPV_NV_shader_atomic_fp16_vector);
                builder.addCapability(spv::Capability::AtomicFloat16VectorNV);
            } else {
                builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_min_max);
                if (typeProxy == glslang::EbtFloat16)
                    builder.addCapability(spv::Capability::AtomicFloat16MinMaxEXT);
                else if (typeProxy == glslang::EbtFloat)
                    builder.addCapability(spv::Capability::AtomicFloat32MinMaxEXT);
                else
                    builder.addCapability(spv::Capability::AtomicFloat64MinMaxEXT);
            }
        } else if (typeProxy == glslang::EbtUint || typeProxy == glslang::EbtUint64) {
            opCode = spv::Op::OpAtomicUMax;
        } else {
            opCode = spv::Op::OpAtomicSMax;
        }
        break;
    case glslang::EOpAtomicAnd:
    case glslang::EOpImageAtomicAnd:
    case glslang::EOpAtomicCounterAnd:
        opCode = spv::Op::OpAtomicAnd;
        break;
    case glslang::EOpAtomicOr:
    case glslang::EOpImageAtomicOr:
    case glslang::EOpAtomicCounterOr:
        opCode = spv::Op::OpAtomicOr;
        break;
    case glslang::EOpAtomicXor:
    case glslang::EOpImageAtomicXor:
    case glslang::EOpAtomicCounterXor:
        opCode = spv::Op::OpAtomicXor;
        break;
    case glslang::EOpAtomicExchange:
    case glslang::EOpImageAtomicExchange:
    case glslang::EOpAtomicCounterExchange:
        if ((typeProxy == glslang::EbtFloat16) && 
            (opType.getVectorSize() == 2 || opType.getVectorSize() == 4)) {
                builder.addExtension(spv::E_SPV_NV_shader_atomic_fp16_vector);
                builder.addCapability(spv::Capability::AtomicFloat16VectorNV);
        }

        opCode = spv::Op::OpAtomicExchange;
        break;
    case glslang::EOpAtomicCompSwap:
    case glslang::EOpImageAtomicCompSwap:
    case glslang::EOpAtomicCounterCompSwap:
        opCode = spv::Op::OpAtomicCompareExchange;
        break;
    case glslang::EOpAtomicCounterIncrement:
        opCode = spv::Op::OpAtomicIIncrement;
        break;
    case glslang::EOpAtomicCounterDecrement:
        opCode = spv::Op::OpAtomicIDecrement;
        break;
    case glslang::EOpAtomicCounter:
    case glslang::EOpImageAtomicLoad:
    case glslang::EOpAtomicLoad:
        opCode = spv::Op::OpAtomicLoad;
        break;
    case glslang::EOpAtomicStore:
    case glslang::EOpImageAtomicStore:
        opCode = spv::Op::OpAtomicStore;
        break;
    default:
        assert(0);
        break;
    }

    if (typeProxy == glslang::EbtInt64 || typeProxy == glslang::EbtUint64)
        builder.addCapability(spv::Capability::Int64Atomics);

    // Sort out the operands
    //  - mapping from glslang -> SPV
    //  - there are extra SPV operands that are optional in glslang
    //  - compare-exchange swaps the value and comparator
    //  - compare-exchange has an extra memory semantics
    //  - EOpAtomicCounterDecrement needs a post decrement
    spv::Id pointerId = 0, compareId = 0, valueId = 0;
    // scope defaults to Device in the old model, QueueFamilyKHR in the new model
    spv::Id scopeId;
    if (glslangIntermediate->usingVulkanMemoryModel()) {
        scopeId = builder.makeUintConstant(spv::Scope::QueueFamilyKHR);
    } else {
        scopeId = builder.makeUintConstant(spv::Scope::Device);
    }
    // semantics default to relaxed
    spv::Id semanticsId = builder.makeUintConstant(lvalueCoherentFlags.isVolatile() &&
        glslangIntermediate->usingVulkanMemoryModel() ?
                                                    spv::MemorySemanticsMask::Volatile :
                                                    spv::MemorySemanticsMask::MaskNone);
    spv::Id semanticsId2 = semanticsId;

    pointerId = operands[0];
    if (opCode == spv::Op::OpAtomicIIncrement || opCode == spv::Op::OpAtomicIDecrement) {
        // no additional operands
    } else if (opCode == spv::Op::OpAtomicCompareExchange) {
        compareId = operands[1];
        valueId = operands[2];
        if (operands.size() > 3) {
            scopeId = operands[3];
            semanticsId = builder.makeUintConstant(
                builder.getConstantScalar(operands[4]) | builder.getConstantScalar(operands[5]));
            semanticsId2 = builder.makeUintConstant(
                builder.getConstantScalar(operands[6]) | builder.getConstantScalar(operands[7]));
        }
    } else if (opCode == spv::Op::OpAtomicLoad) {
        if (operands.size() > 1) {
            scopeId = operands[1];
            semanticsId = builder.makeUintConstant(
                builder.getConstantScalar(operands[2]) | builder.getConstantScalar(operands[3]));
        }
    } else {
        // atomic store or RMW
        valueId = operands[1];
        if (operands.size() > 2) {
            scopeId = operands[2];
            semanticsId = builder.makeUintConstant
                (builder.getConstantScalar(operands[3]) | builder.getConstantScalar(operands[4]));
        }
    }

    // Check for capabilities
    auto const semanticsImmediate = (spv::MemorySemanticsMask)(builder.getConstantScalar(semanticsId) | builder.getConstantScalar(semanticsId2));
    if (anySet(semanticsImmediate, spv::MemorySemanticsMask::MakeAvailableKHR |
                                   spv::MemorySemanticsMask::MakeVisibleKHR |
                                   spv::MemorySemanticsMask::OutputMemoryKHR |
                                   spv::MemorySemanticsMask::Volatile)) {
        builder.addCapability(spv::Capability::VulkanMemoryModelKHR);
    }

    auto const scope = (spv::Scope)builder.getConstantScalar(scopeId);
    if (scope == spv::Scope::QueueFamily) {
        builder.addCapability(spv::Capability::VulkanMemoryModelKHR);
    }

    if (glslangIntermediate->usingVulkanMemoryModel() && scope == spv::Scope::Device) {
        builder.addCapability(spv::Capability::VulkanMemoryModelDeviceScopeKHR);
    }

    std::vector<spv::Id> spvAtomicOperands;  // hold the spv operands
    spvAtomicOperands.reserve(6);
    spvAtomicOperands.push_back(pointerId);
    spvAtomicOperands.push_back(scopeId);
    spvAtomicOperands.push_back(semanticsId);
    if (opCode == spv::Op::OpAtomicCompareExchange) {
        spvAtomicOperands.push_back(semanticsId2);
        spvAtomicOperands.push_back(valueId);
        spvAtomicOperands.push_back(compareId);
    } else if (opCode != spv::Op::OpAtomicLoad && opCode != spv::Op::OpAtomicIIncrement && opCode != spv::Op::OpAtomicIDecrement) {
        spvAtomicOperands.push_back(valueId);
    }

    if (opCode == spv::Op::OpAtomicStore) {
        builder.createNoResultOp(opCode, spvAtomicOperands);
        return 0;
    } else {
        spv::Id resultId = builder.createOp(opCode, typeId, spvAtomicOperands);

        // GLSL and HLSL atomic-counter decrement return post-decrement value,
        // while SPIR-V returns pre-decrement value. Translate between these semantics.
        if (op == glslang::EOpAtomicCounterDecrement)
            resultId = builder.createBinOp(spv::Op::OpISub, typeId, resultId, builder.makeIntConstant(1));

        return resultId;
    }
}

// Create group invocation operations.
spv::Id TGlslangToSpvTraverser::createInvocationsOperation(glslang::TOperator op, spv::Id typeId,
    std::vector<spv::Id>& operands, glslang::TBasicType typeProxy)
{
    bool isUnsigned = isTypeUnsignedInt(typeProxy);
    bool isFloat = isTypeFloat(typeProxy);

    spv::Op opCode = spv::Op::OpNop;
    std::vector<spv::IdImmediate> spvGroupOperands;
    spv::GroupOperation groupOperation = spv::GroupOperation::Max;

    if (op == glslang::EOpBallot || op == glslang::EOpReadFirstInvocation ||
        op == glslang::EOpReadInvocation) {
        builder.addExtension(spv::E_SPV_KHR_shader_ballot);
        builder.addCapability(spv::Capability::SubgroupBallotKHR);
    } else if (op == glslang::EOpAnyInvocation ||
        op == glslang::EOpAllInvocations ||
        op == glslang::EOpAllInvocationsEqual) {
        builder.addExtension(spv::E_SPV_KHR_subgroup_vote);
        builder.addCapability(spv::Capability::SubgroupVoteKHR);
    } else {
        builder.addCapability(spv::Capability::Groups);
        if (op == glslang::EOpMinInvocationsNonUniform ||
            op == glslang::EOpMaxInvocationsNonUniform ||
            op == glslang::EOpAddInvocationsNonUniform ||
            op == glslang::EOpMinInvocationsInclusiveScanNonUniform ||
            op == glslang::EOpMaxInvocationsInclusiveScanNonUniform ||
            op == glslang::EOpAddInvocationsInclusiveScanNonUniform ||
            op == glslang::EOpMinInvocationsExclusiveScanNonUniform ||
            op == glslang::EOpMaxInvocationsExclusiveScanNonUniform ||
            op == glslang::EOpAddInvocationsExclusiveScanNonUniform)
            builder.addExtension(spv::E_SPV_AMD_shader_ballot);

        switch (op) {
        case glslang::EOpMinInvocations:
        case glslang::EOpMaxInvocations:
        case glslang::EOpAddInvocations:
        case glslang::EOpMinInvocationsNonUniform:
        case glslang::EOpMaxInvocationsNonUniform:
        case glslang::EOpAddInvocationsNonUniform:
            groupOperation = spv::GroupOperation::Reduce;
            break;
        case glslang::EOpMinInvocationsInclusiveScan:
        case glslang::EOpMaxInvocationsInclusiveScan:
        case glslang::EOpAddInvocationsInclusiveScan:
        case glslang::EOpMinInvocationsInclusiveScanNonUniform:
        case glslang::EOpMaxInvocationsInclusiveScanNonUniform:
        case glslang::EOpAddInvocationsInclusiveScanNonUniform:
            groupOperation = spv::GroupOperation::InclusiveScan;
            break;
        case glslang::EOpMinInvocationsExclusiveScan:
        case glslang::EOpMaxInvocationsExclusiveScan:
        case glslang::EOpAddInvocationsExclusiveScan:
        case glslang::EOpMinInvocationsExclusiveScanNonUniform:
        case glslang::EOpMaxInvocationsExclusiveScanNonUniform:
        case glslang::EOpAddInvocationsExclusiveScanNonUniform:
            groupOperation = spv::GroupOperation::ExclusiveScan;
            break;
        default:
            break;
        }
        spv::IdImmediate scope = { true, builder.makeUintConstant(spv::Scope::Subgroup) };
        spvGroupOperands.push_back(scope);
        if (groupOperation != spv::GroupOperation::Max) {
            spv::IdImmediate groupOp = { false, (unsigned)groupOperation };
            spvGroupOperands.push_back(groupOp);
        }
    }

    for (auto opIt = operands.begin(); opIt != operands.end(); ++opIt) {
        spv::IdImmediate op = { true, *opIt };
        spvGroupOperands.push_back(op);
    }

    switch (op) {
    case glslang::EOpAnyInvocation:
        opCode = spv::Op::OpSubgroupAnyKHR;
        break;
    case glslang::EOpAllInvocations:
        opCode = spv::Op::OpSubgroupAllKHR;
        break;
    case glslang::EOpAllInvocationsEqual:
        opCode = spv::Op::OpSubgroupAllEqualKHR;
        break;
    case glslang::EOpReadInvocation:
        opCode = spv::Op::OpSubgroupReadInvocationKHR;
        if (builder.isVectorType(typeId))
            return CreateInvocationsVectorOperation(opCode, groupOperation, typeId, operands);
        break;
    case glslang::EOpReadFirstInvocation:
        opCode = spv::Op::OpSubgroupFirstInvocationKHR;
        if (builder.isVectorType(typeId))
            return CreateInvocationsVectorOperation(opCode, groupOperation, typeId, operands);
        break;
    case glslang::EOpBallot:
    {
        // NOTE: According to the spec, the result type of "OpSubgroupBallotKHR" must be a 4 component vector of 32
        // bit integer types. The GLSL built-in function "ballotARB()" assumes the maximum number of invocations in
        // a subgroup is 64. Thus, we have to convert uvec4.xy to uint64_t as follow:
        //
        //     result = Bitcast(SubgroupBallotKHR(Predicate).xy)
        //
        spv::Id uintType  = builder.makeUintType(32);
        spv::Id uvec4Type = builder.makeVectorType(uintType, 4);
        spv::Id result = builder.createOp(spv::Op::OpSubgroupBallotKHR, uvec4Type, spvGroupOperands);

        std::vector<spv::Id> components;
        components.push_back(builder.createCompositeExtract(result, uintType, 0));
        components.push_back(builder.createCompositeExtract(result, uintType, 1));

        spv::Id uvec2Type = builder.makeVectorType(uintType, 2);
        return builder.createUnaryOp(spv::Op::OpBitcast, typeId,
                                     builder.createCompositeConstruct(uvec2Type, components));
    }

    case glslang::EOpMinInvocations:
    case glslang::EOpMaxInvocations:
    case glslang::EOpAddInvocations:
    case glslang::EOpMinInvocationsInclusiveScan:
    case glslang::EOpMaxInvocationsInclusiveScan:
    case glslang::EOpAddInvocationsInclusiveScan:
    case glslang::EOpMinInvocationsExclusiveScan:
    case glslang::EOpMaxInvocationsExclusiveScan:
    case glslang::EOpAddInvocationsExclusiveScan:
        if (op == glslang::EOpMinInvocations ||
            op == glslang::EOpMinInvocationsInclusiveScan ||
            op == glslang::EOpMinInvocationsExclusiveScan) {
            if (isFloat)
                opCode = spv::Op::OpGroupFMin;
            else {
                if (isUnsigned)
                    opCode = spv::Op::OpGroupUMin;
                else
                    opCode = spv::Op::OpGroupSMin;
            }
        } else if (op == glslang::EOpMaxInvocations ||
                   op == glslang::EOpMaxInvocationsInclusiveScan ||
                   op == glslang::EOpMaxInvocationsExclusiveScan) {
            if (isFloat)
                opCode = spv::Op::OpGroupFMax;
            else {
                if (isUnsigned)
                    opCode = spv::Op::OpGroupUMax;
                else
                    opCode = spv::Op::OpGroupSMax;
            }
        } else {
            if (isFloat)
                opCode = spv::Op::OpGroupFAdd;
            else
                opCode = spv::Op::OpGroupIAdd;
        }

        if (builder.isVectorType(typeId))
            return CreateInvocationsVectorOperation(opCode, groupOperation, typeId, operands);

        break;
    case glslang::EOpMinInvocationsNonUniform:
    case glslang::EOpMaxInvocationsNonUniform:
    case glslang::EOpAddInvocationsNonUniform:
    case glslang::EOpMinInvocationsInclusiveScanNonUniform:
    case glslang::EOpMaxInvocationsInclusiveScanNonUniform:
    case glslang::EOpAddInvocationsInclusiveScanNonUniform:
    case glslang::EOpMinInvocationsExclusiveScanNonUniform:
    case glslang::EOpMaxInvocationsExclusiveScanNonUniform:
    case glslang::EOpAddInvocationsExclusiveScanNonUniform:
        if (op == glslang::EOpMinInvocationsNonUniform ||
            op == glslang::EOpMinInvocationsInclusiveScanNonUniform ||
            op == glslang::EOpMinInvocationsExclusiveScanNonUniform) {
            if (isFloat)
                opCode = spv::Op::OpGroupFMinNonUniformAMD;
            else {
                if (isUnsigned)
                    opCode = spv::Op::OpGroupUMinNonUniformAMD;
                else
                    opCode = spv::Op::OpGroupSMinNonUniformAMD;
            }
        }
        else if (op == glslang::EOpMaxInvocationsNonUniform ||
                 op == glslang::EOpMaxInvocationsInclusiveScanNonUniform ||
                 op == glslang::EOpMaxInvocationsExclusiveScanNonUniform) {
            if (isFloat)
                opCode = spv::Op::OpGroupFMaxNonUniformAMD;
            else {
                if (isUnsigned)
                    opCode = spv::Op::OpGroupUMaxNonUniformAMD;
                else
                    opCode = spv::Op::OpGroupSMaxNonUniformAMD;
            }
        }
        else {
            if (isFloat)
                opCode = spv::Op::OpGroupFAddNonUniformAMD;
            else
                opCode = spv::Op::OpGroupIAddNonUniformAMD;
        }

        if (builder.isVectorType(typeId))
            return CreateInvocationsVectorOperation(opCode, groupOperation, typeId, operands);

        break;
    default:
        logger->missingFunctionality("invocation operation");
        return spv::NoResult;
    }

    assert(opCode != spv::Op::OpNop);
    return builder.createOp(opCode, typeId, spvGroupOperands);
}

// Create group invocation operations on a vector
spv::Id TGlslangToSpvTraverser::CreateInvocationsVectorOperation(spv::Op op, spv::GroupOperation groupOperation,
    spv::Id typeId, std::vector<spv::Id>& operands)
{
    assert(op == spv::Op::OpGroupFMin || op == spv::Op::OpGroupUMin || op == spv::Op::OpGroupSMin ||
           op == spv::Op::OpGroupFMax || op == spv::Op::OpGroupUMax || op == spv::Op::OpGroupSMax ||
           op == spv::Op::OpGroupFAdd || op == spv::Op::OpGroupIAdd || op == spv::Op::OpGroupBroadcast ||
           op == spv::Op::OpSubgroupReadInvocationKHR || op == spv::Op::OpSubgroupFirstInvocationKHR ||
           op == spv::Op::OpGroupFMinNonUniformAMD || op == spv::Op::OpGroupUMinNonUniformAMD ||
           op == spv::Op::OpGroupSMinNonUniformAMD ||
           op == spv::Op::OpGroupFMaxNonUniformAMD || op == spv::Op::OpGroupUMaxNonUniformAMD ||
           op == spv::Op::OpGroupSMaxNonUniformAMD ||
           op == spv::Op::OpGroupFAddNonUniformAMD || op == spv::Op::OpGroupIAddNonUniformAMD);

    // Handle group invocation operations scalar by scalar.
    // The result type is the same type as the original type.
    // The algorithm is to:
    //   - break the vector into scalars
    //   - apply the operation to each scalar
    //   - make a vector out the scalar results

    // get the types sorted out
    int numComponents = builder.getNumComponents(operands[0]);
    spv::Id scalarType = builder.getScalarTypeId(builder.getTypeId(operands[0]));
    std::vector<spv::Id> results;

    // do each scalar op
    for (int comp = 0; comp < numComponents; ++comp) {
        std::vector<unsigned int> indexes;
        indexes.push_back(comp);
        spv::IdImmediate scalar = { true, builder.createCompositeExtract(operands[0], scalarType, indexes) };
        std::vector<spv::IdImmediate> spvGroupOperands;
        if (op == spv::Op::OpSubgroupReadInvocationKHR) {
            spvGroupOperands.push_back(scalar);
            spv::IdImmediate operand = { true, operands[1] };
            spvGroupOperands.push_back(operand);
        } else if (op == spv::Op::OpSubgroupFirstInvocationKHR) {
            spvGroupOperands.push_back(scalar);
        } else if (op == spv::Op::OpGroupBroadcast) {
            spv::IdImmediate scope = { true, builder.makeUintConstant(spv::Scope::Subgroup) };
            spvGroupOperands.push_back(scope);
            spvGroupOperands.push_back(scalar);
            spv::IdImmediate operand = { true, operands[1] };
            spvGroupOperands.push_back(operand);
        } else {
            spv::IdImmediate scope = { true, builder.makeUintConstant(spv::Scope::Subgroup) };
            spvGroupOperands.push_back(scope);
            spv::IdImmediate groupOp = { false, (unsigned)groupOperation };
            spvGroupOperands.push_back(groupOp);
            spvGroupOperands.push_back(scalar);
        }

        results.push_back(builder.createOp(op, scalarType, spvGroupOperands));
    }

    // put the pieces together
    return builder.createCompositeConstruct(typeId, results);
}

// Create subgroup invocation operations.
spv::Id TGlslangToSpvTraverser::createSubgroupOperation(glslang::TOperator op, spv::Id typeId,
    std::vector<spv::Id>& operands, glslang::TBasicType typeProxy)
{
    // Add the required capabilities.
    switch (op) {
    case glslang::EOpSubgroupElect:
        builder.addCapability(spv::Capability::GroupNonUniform);
        break;
    case glslang::EOpSubgroupQuadAll:
    case glslang::EOpSubgroupQuadAny:
        builder.addExtension(spv::E_SPV_KHR_quad_control);
        builder.addCapability(spv::Capability::QuadControlKHR);
        [[fallthrough]];
    case glslang::EOpSubgroupAll:
    case glslang::EOpSubgroupAny:
    case glslang::EOpSubgroupAllEqual:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformVote);
        break;
    case glslang::EOpSubgroupBroadcast:
    case glslang::EOpSubgroupBroadcastFirst:
    case glslang::EOpSubgroupBallot:
    case glslang::EOpSubgroupInverseBallot:
    case glslang::EOpSubgroupBallotBitExtract:
    case glslang::EOpSubgroupBallotBitCount:
    case glslang::EOpSubgroupBallotInclusiveBitCount:
    case glslang::EOpSubgroupBallotExclusiveBitCount:
    case glslang::EOpSubgroupBallotFindLSB:
    case glslang::EOpSubgroupBallotFindMSB:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformBallot);
        break;
    case glslang::EOpSubgroupRotate:
    case glslang::EOpSubgroupClusteredRotate:
        builder.addExtension(spv::E_SPV_KHR_subgroup_rotate);
        builder.addCapability(spv::Capability::GroupNonUniformRotateKHR);
        break;
    case glslang::EOpSubgroupShuffle:
    case glslang::EOpSubgroupShuffleXor:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformShuffle);
        break;
    case glslang::EOpSubgroupShuffleUp:
    case glslang::EOpSubgroupShuffleDown:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformShuffleRelative);
        break;
    case glslang::EOpSubgroupAdd:
    case glslang::EOpSubgroupMul:
    case glslang::EOpSubgroupMin:
    case glslang::EOpSubgroupMax:
    case glslang::EOpSubgroupAnd:
    case glslang::EOpSubgroupOr:
    case glslang::EOpSubgroupXor:
    case glslang::EOpSubgroupInclusiveAdd:
    case glslang::EOpSubgroupInclusiveMul:
    case glslang::EOpSubgroupInclusiveMin:
    case glslang::EOpSubgroupInclusiveMax:
    case glslang::EOpSubgroupInclusiveAnd:
    case glslang::EOpSubgroupInclusiveOr:
    case glslang::EOpSubgroupInclusiveXor:
    case glslang::EOpSubgroupExclusiveAdd:
    case glslang::EOpSubgroupExclusiveMul:
    case glslang::EOpSubgroupExclusiveMin:
    case glslang::EOpSubgroupExclusiveMax:
    case glslang::EOpSubgroupExclusiveAnd:
    case glslang::EOpSubgroupExclusiveOr:
    case glslang::EOpSubgroupExclusiveXor:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformArithmetic);
        break;
    case glslang::EOpSubgroupClusteredAdd:
    case glslang::EOpSubgroupClusteredMul:
    case glslang::EOpSubgroupClusteredMin:
    case glslang::EOpSubgroupClusteredMax:
    case glslang::EOpSubgroupClusteredAnd:
    case glslang::EOpSubgroupClusteredOr:
    case glslang::EOpSubgroupClusteredXor:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformClustered);
        break;
    case glslang::EOpSubgroupQuadBroadcast:
    case glslang::EOpSubgroupQuadSwapHorizontal:
    case glslang::EOpSubgroupQuadSwapVertical:
    case glslang::EOpSubgroupQuadSwapDiagonal:
        builder.addCapability(spv::Capability::GroupNonUniform);
        builder.addCapability(spv::Capability::GroupNonUniformQuad);
        break;
    case glslang::EOpSubgroupPartitionedAdd:
    case glslang::EOpSubgroupPartitionedMul:
    case glslang::EOpSubgroupPartitionedMin:
    case glslang::EOpSubgroupPartitionedMax:
    case glslang::EOpSubgroupPartitionedAnd:
    case glslang::EOpSubgroupPartitionedOr:
    case glslang::EOpSubgroupPartitionedXor:
    case glslang::EOpSubgroupPartitionedInclusiveAdd:
    case glslang::EOpSubgroupPartitionedInclusiveMul:
    case glslang::EOpSubgroupPartitionedInclusiveMin:
    case glslang::EOpSubgroupPartitionedInclusiveMax:
    case glslang::EOpSubgroupPartitionedInclusiveAnd:
    case glslang::EOpSubgroupPartitionedInclusiveOr:
    case glslang::EOpSubgroupPartitionedInclusiveXor:
    case glslang::EOpSubgroupPartitionedExclusiveAdd:
    case glslang::EOpSubgroupPartitionedExclusiveMul:
    case glslang::EOpSubgroupPartitionedExclusiveMin:
    case glslang::EOpSubgroupPartitionedExclusiveMax:
    case glslang::EOpSubgroupPartitionedExclusiveAnd:
    case glslang::EOpSubgroupPartitionedExclusiveOr:
    case glslang::EOpSubgroupPartitionedExclusiveXor:
        builder.addExtension(spv::E_SPV_NV_shader_subgroup_partitioned);
        builder.addCapability(spv::Capability::GroupNonUniformPartitionedNV);
        break;
    default: assert(0 && "Unhandled subgroup operation!");
    }


    const bool isUnsigned = isTypeUnsignedInt(typeProxy);
    const bool isFloat = isTypeFloat(typeProxy);
    const bool isBool = typeProxy == glslang::EbtBool;

    spv::Op opCode = spv::Op::OpNop;

    // Figure out which opcode to use.
    switch (op) {
    case glslang::EOpSubgroupElect:                   opCode = spv::Op::OpGroupNonUniformElect; break;
    case glslang::EOpSubgroupQuadAll:                 opCode = spv::Op::OpGroupNonUniformQuadAllKHR; break;
    case glslang::EOpSubgroupAll:                     opCode = spv::Op::OpGroupNonUniformAll; break;
    case glslang::EOpSubgroupQuadAny:                 opCode = spv::Op::OpGroupNonUniformQuadAnyKHR; break;
    case glslang::EOpSubgroupAny:                     opCode = spv::Op::OpGroupNonUniformAny; break;
    case glslang::EOpSubgroupAllEqual:                opCode = spv::Op::OpGroupNonUniformAllEqual; break;
    case glslang::EOpSubgroupBroadcast:               opCode = spv::Op::OpGroupNonUniformBroadcast; break;
    case glslang::EOpSubgroupBroadcastFirst:          opCode = spv::Op::OpGroupNonUniformBroadcastFirst; break;
    case glslang::EOpSubgroupBallot:                  opCode = spv::Op::OpGroupNonUniformBallot; break;
    case glslang::EOpSubgroupInverseBallot:           opCode = spv::Op::OpGroupNonUniformInverseBallot; break;
    case glslang::EOpSubgroupBallotBitExtract:        opCode = spv::Op::OpGroupNonUniformBallotBitExtract; break;
    case glslang::EOpSubgroupBallotBitCount:
    case glslang::EOpSubgroupBallotInclusiveBitCount:
    case glslang::EOpSubgroupBallotExclusiveBitCount: opCode = spv::Op::OpGroupNonUniformBallotBitCount; break;
    case glslang::EOpSubgroupBallotFindLSB:           opCode = spv::Op::OpGroupNonUniformBallotFindLSB; break;
    case glslang::EOpSubgroupBallotFindMSB:           opCode = spv::Op::OpGroupNonUniformBallotFindMSB; break;
    case glslang::EOpSubgroupShuffle:                 opCode = spv::Op::OpGroupNonUniformShuffle; break;
    case glslang::EOpSubgroupShuffleXor:              opCode = spv::Op::OpGroupNonUniformShuffleXor; break;
    case glslang::EOpSubgroupShuffleUp:               opCode = spv::Op::OpGroupNonUniformShuffleUp; break;
    case glslang::EOpSubgroupShuffleDown:             opCode = spv::Op::OpGroupNonUniformShuffleDown; break;
    case glslang::EOpSubgroupRotate:
    case glslang::EOpSubgroupClusteredRotate:         opCode = spv::Op::OpGroupNonUniformRotateKHR; break;
    case glslang::EOpSubgroupAdd:
    case glslang::EOpSubgroupInclusiveAdd:
    case glslang::EOpSubgroupExclusiveAdd:
    case glslang::EOpSubgroupClusteredAdd:
    case glslang::EOpSubgroupPartitionedAdd:
    case glslang::EOpSubgroupPartitionedInclusiveAdd:
    case glslang::EOpSubgroupPartitionedExclusiveAdd:
        if (isFloat) {
            opCode = spv::Op::OpGroupNonUniformFAdd;
        } else {
            opCode = spv::Op::OpGroupNonUniformIAdd;
        }
        break;
    case glslang::EOpSubgroupMul:
    case glslang::EOpSubgroupInclusiveMul:
    case glslang::EOpSubgroupExclusiveMul:
    case glslang::EOpSubgroupClusteredMul:
    case glslang::EOpSubgroupPartitionedMul:
    case glslang::EOpSubgroupPartitionedInclusiveMul:
    case glslang::EOpSubgroupPartitionedExclusiveMul:
        if (isFloat) {
            opCode = spv::Op::OpGroupNonUniformFMul;
        } else {
            opCode = spv::Op::OpGroupNonUniformIMul;
        }
        break;
    case glslang::EOpSubgroupMin:
    case glslang::EOpSubgroupInclusiveMin:
    case glslang::EOpSubgroupExclusiveMin:
    case glslang::EOpSubgroupClusteredMin:
    case glslang::EOpSubgroupPartitionedMin:
    case glslang::EOpSubgroupPartitionedInclusiveMin:
    case glslang::EOpSubgroupPartitionedExclusiveMin:
        if (isFloat) {
            opCode = spv::Op::OpGroupNonUniformFMin;
        } else if (isUnsigned) {
            opCode = spv::Op::OpGroupNonUniformUMin;
        } else {
            opCode = spv::Op::OpGroupNonUniformSMin;
        }
        break;
    case glslang::EOpSubgroupMax:
    case glslang::EOpSubgroupInclusiveMax:
    case glslang::EOpSubgroupExclusiveMax:
    case glslang::EOpSubgroupClusteredMax:
    case glslang::EOpSubgroupPartitionedMax:
    case glslang::EOpSubgroupPartitionedInclusiveMax:
    case glslang::EOpSubgroupPartitionedExclusiveMax:
        if (isFloat) {
            opCode = spv::Op::OpGroupNonUniformFMax;
        } else if (isUnsigned) {
            opCode = spv::Op::OpGroupNonUniformUMax;
        } else {
            opCode = spv::Op::OpGroupNonUniformSMax;
        }
        break;
    case glslang::EOpSubgroupAnd:
    case glslang::EOpSubgroupInclusiveAnd:
    case glslang::EOpSubgroupExclusiveAnd:
    case glslang::EOpSubgroupClusteredAnd:
    case glslang::EOpSubgroupPartitionedAnd:
    case glslang::EOpSubgroupPartitionedInclusiveAnd:
    case glslang::EOpSubgroupPartitionedExclusiveAnd:
        if (isBool) {
            opCode = spv::Op::OpGroupNonUniformLogicalAnd;
        } else {
            opCode = spv::Op::OpGroupNonUniformBitwiseAnd;
        }
        break;
    case glslang::EOpSubgroupOr:
    case glslang::EOpSubgroupInclusiveOr:
    case glslang::EOpSubgroupExclusiveOr:
    case glslang::EOpSubgroupClusteredOr:
    case glslang::EOpSubgroupPartitionedOr:
    case glslang::EOpSubgroupPartitionedInclusiveOr:
    case glslang::EOpSubgroupPartitionedExclusiveOr:
        if (isBool) {
            opCode = spv::Op::OpGroupNonUniformLogicalOr;
        } else {
            opCode = spv::Op::OpGroupNonUniformBitwiseOr;
        }
        break;
    case glslang::EOpSubgroupXor:
    case glslang::EOpSubgroupInclusiveXor:
    case glslang::EOpSubgroupExclusiveXor:
    case glslang::EOpSubgroupClusteredXor:
    case glslang::EOpSubgroupPartitionedXor:
    case glslang::EOpSubgroupPartitionedInclusiveXor:
    case glslang::EOpSubgroupPartitionedExclusiveXor:
        if (isBool) {
            opCode = spv::Op::OpGroupNonUniformLogicalXor;
        } else {
            opCode = spv::Op::OpGroupNonUniformBitwiseXor;
        }
        break;
    case glslang::EOpSubgroupQuadBroadcast:      opCode = spv::Op::OpGroupNonUniformQuadBroadcast; break;
    case glslang::EOpSubgroupQuadSwapHorizontal:
    case glslang::EOpSubgroupQuadSwapVertical:
    case glslang::EOpSubgroupQuadSwapDiagonal:   opCode = spv::Op::OpGroupNonUniformQuadSwap; break;
    default: assert(0 && "Unhandled subgroup operation!");
    }

    // get the right Group Operation
    spv::GroupOperation groupOperation = spv::GroupOperation::Max;
    switch (op) {
    default:
        break;
    case glslang::EOpSubgroupBallotBitCount:
    case glslang::EOpSubgroupAdd:
    case glslang::EOpSubgroupMul:
    case glslang::EOpSubgroupMin:
    case glslang::EOpSubgroupMax:
    case glslang::EOpSubgroupAnd:
    case glslang::EOpSubgroupOr:
    case glslang::EOpSubgroupXor:
        groupOperation = spv::GroupOperation::Reduce;
        break;
    case glslang::EOpSubgroupBallotInclusiveBitCount:
    case glslang::EOpSubgroupInclusiveAdd:
    case glslang::EOpSubgroupInclusiveMul:
    case glslang::EOpSubgroupInclusiveMin:
    case glslang::EOpSubgroupInclusiveMax:
    case glslang::EOpSubgroupInclusiveAnd:
    case glslang::EOpSubgroupInclusiveOr:
    case glslang::EOpSubgroupInclusiveXor:
        groupOperation = spv::GroupOperation::InclusiveScan;
        break;
    case glslang::EOpSubgroupBallotExclusiveBitCount:
    case glslang::EOpSubgroupExclusiveAdd:
    case glslang::EOpSubgroupExclusiveMul:
    case glslang::EOpSubgroupExclusiveMin:
    case glslang::EOpSubgroupExclusiveMax:
    case glslang::EOpSubgroupExclusiveAnd:
    case glslang::EOpSubgroupExclusiveOr:
    case glslang::EOpSubgroupExclusiveXor:
        groupOperation = spv::GroupOperation::ExclusiveScan;
        break;
    case glslang::EOpSubgroupClusteredAdd:
    case glslang::EOpSubgroupClusteredMul:
    case glslang::EOpSubgroupClusteredMin:
    case glslang::EOpSubgroupClusteredMax:
    case glslang::EOpSubgroupClusteredAnd:
    case glslang::EOpSubgroupClusteredOr:
    case glslang::EOpSubgroupClusteredXor:
        groupOperation = spv::GroupOperation::ClusteredReduce;
        break;
    case glslang::EOpSubgroupPartitionedAdd:
    case glslang::EOpSubgroupPartitionedMul:
    case glslang::EOpSubgroupPartitionedMin:
    case glslang::EOpSubgroupPartitionedMax:
    case glslang::EOpSubgroupPartitionedAnd:
    case glslang::EOpSubgroupPartitionedOr:
    case glslang::EOpSubgroupPartitionedXor:
        groupOperation = spv::GroupOperation::PartitionedReduceNV;
        break;
    case glslang::EOpSubgroupPartitionedInclusiveAdd:
    case glslang::EOpSubgroupPartitionedInclusiveMul:
    case glslang::EOpSubgroupPartitionedInclusiveMin:
    case glslang::EOpSubgroupPartitionedInclusiveMax:
    case glslang::EOpSubgroupPartitionedInclusiveAnd:
    case glslang::EOpSubgroupPartitionedInclusiveOr:
    case glslang::EOpSubgroupPartitionedInclusiveXor:
        groupOperation = spv::GroupOperation::PartitionedInclusiveScanNV;
        break;
    case glslang::EOpSubgroupPartitionedExclusiveAdd:
    case glslang::EOpSubgroupPartitionedExclusiveMul:
    case glslang::EOpSubgroupPartitionedExclusiveMin:
    case glslang::EOpSubgroupPartitionedExclusiveMax:
    case glslang::EOpSubgroupPartitionedExclusiveAnd:
    case glslang::EOpSubgroupPartitionedExclusiveOr:
    case glslang::EOpSubgroupPartitionedExclusiveXor:
        groupOperation = spv::GroupOperation::PartitionedExclusiveScanNV;
        break;
    }

    // build the instruction
    std::vector<spv::IdImmediate> spvGroupOperands;

    // Every operation begins with the Execution Scope operand.
    spv::IdImmediate executionScope = { true, builder.makeUintConstant(spv::Scope::Subgroup) };
    // All other ops need the execution scope. Quad Control Ops don't need scope, it's always Quad.
    if (opCode != spv::Op::OpGroupNonUniformQuadAllKHR && opCode != spv::Op::OpGroupNonUniformQuadAnyKHR) {
        spvGroupOperands.push_back(executionScope);
    }

    // Next, for all operations that use a Group Operation, push that as an operand.
    if (groupOperation != spv::GroupOperation::Max) {
        spv::IdImmediate groupOperand = { false, (unsigned)groupOperation };
        spvGroupOperands.push_back(groupOperand);
    }

    // Push back the operands next.
    for (auto opIt = operands.cbegin(); opIt != operands.cend(); ++opIt) {
        spv::IdImmediate operand = { true, *opIt };
        spvGroupOperands.push_back(operand);
    }

    // Some opcodes have additional operands.
    spv::Id directionId = spv::NoResult;
    switch (op) {
    default: break;
    case glslang::EOpSubgroupQuadSwapHorizontal: directionId = builder.makeUintConstant(0); break;
    case glslang::EOpSubgroupQuadSwapVertical:   directionId = builder.makeUintConstant(1); break;
    case glslang::EOpSubgroupQuadSwapDiagonal:   directionId = builder.makeUintConstant(2); break;
    }
    if (directionId != spv::NoResult) {
        spv::IdImmediate direction = { true, directionId };
        spvGroupOperands.push_back(direction);
    }

    return builder.createOp(opCode, typeId, spvGroupOperands);
}

spv::Id TGlslangToSpvTraverser::createMiscOperation(glslang::TOperator op, spv::Decoration precision,
    spv::Id typeId, std::vector<spv::Id>& operands, glslang::TBasicType typeProxy)
{
    bool isUnsigned = isTypeUnsignedInt(typeProxy);
    bool isFloat = isTypeFloat(typeProxy);

    spv::Op opCode = spv::Op::OpNop;
    int extBuiltins = -1;
    int libCall = -1;
    size_t consumedOperands = operands.size();
    spv::Id typeId0 = 0;
    if (consumedOperands > 0)
        typeId0 = builder.getTypeId(operands[0]);
    spv::Id typeId1 = 0;
    if (consumedOperands > 1)
        typeId1 = builder.getTypeId(operands[1]);
    spv::Id frexpIntType = 0;

    switch (op) {
    case glslang::EOpMin:
        if (isFloat)
            libCall = nanMinMaxClamp ? spv::GLSLstd450NMin : spv::GLSLstd450FMin;
        else if (isUnsigned)
            libCall = spv::GLSLstd450UMin;
        else
            libCall = spv::GLSLstd450SMin;
        builder.promoteScalar(precision, operands.front(), operands.back());
        break;
    case glslang::EOpModf:
        {
            libCall = spv::GLSLstd450ModfStruct;
            assert(builder.isFloatType(builder.getScalarTypeId(typeId0)));
            // The returned struct has two members of the same type as the first argument
            typeId = builder.makeStructResultType(typeId0, typeId0);
            consumedOperands = 1;
        }
        break;
    case glslang::EOpMax:
        if (isFloat)
            libCall = nanMinMaxClamp ? spv::GLSLstd450NMax : spv::GLSLstd450FMax;
        else if (isUnsigned)
            libCall = spv::GLSLstd450UMax;
        else
            libCall = spv::GLSLstd450SMax;
        builder.promoteScalar(precision, operands.front(), operands.back());
        break;
    case glslang::EOpPow:
        libCall = spv::GLSLstd450Pow;
        break;
    case glslang::EOpDot:
    case glslang::EOpDotPackedEXT:
    case glslang::EOpDotAccSatEXT:
    case glslang::EOpDotPackedAccSatEXT:
        {
            if (builder.isFloatType(builder.getScalarTypeId(typeId0)) ||
                // HLSL supports dot(int,int) which is just a multiply
                glslangIntermediate->getSource() == glslang::EShSourceHlsl) {
                if (typeProxy == glslang::EbtBFloat16) {
                    builder.addExtension(spv::E_SPV_KHR_bfloat16);
                    builder.addCapability(spv::Capability::BFloat16DotProductKHR);
                }
                opCode = spv::Op::OpDot;
            } else {
                builder.addExtension(spv::E_SPV_KHR_integer_dot_product);
                builder.addCapability(spv::Capability::DotProductKHR);
                const unsigned int vectorSize = builder.getNumComponents(operands[0]);
                if (op == glslang::EOpDotPackedEXT || op == glslang::EOpDotPackedAccSatEXT) {
                    builder.addCapability(spv::Capability::DotProductInput4x8BitPackedKHR);
                } else if (vectorSize == 4 && builder.getScalarTypeWidth(typeId0) == 8) {
                    builder.addCapability(spv::Capability::DotProductInput4x8BitKHR);
                } else {
                    builder.addCapability(spv::Capability::DotProductInputAllKHR);
                }
                const bool type0isSigned = builder.isIntType(builder.getScalarTypeId(typeId0));
                const bool type1isSigned = builder.isIntType(builder.getScalarTypeId(typeId1));
                const bool accSat = (op == glslang::EOpDotAccSatEXT || op == glslang::EOpDotPackedAccSatEXT);
                if (!type0isSigned && !type1isSigned) {
                    opCode = accSat ? spv::Op::OpUDotAccSatKHR : spv::Op::OpUDotKHR;
                } else if (type0isSigned && type1isSigned) {
                    opCode = accSat ? spv::Op::OpSDotAccSatKHR : spv::Op::OpSDotKHR;
                } else {
                    opCode = accSat ? spv::Op::OpSUDotAccSatKHR : spv::Op::OpSUDotKHR;
                    // the spir-v opcode assumes the operands to be "signed, unsigned" in that order, so swap if needed
                    if (type1isSigned) {
                        std::swap(operands[0], operands[1]);
                    }
                }
                std::vector<spv::IdImmediate> operands2;
                for (auto &o : operands) {
                    operands2.push_back({true, o});
                }
                if (op == glslang::EOpDotPackedEXT || op == glslang::EOpDotPackedAccSatEXT) {
                    operands2.push_back({false, 0});
                }
                return builder.createOp(opCode, typeId, operands2);
            }
        }
        break;
    case glslang::EOpAtan:
        libCall = spv::GLSLstd450Atan2;
        break;

    case glslang::EOpClamp:
        if (isFloat)
            libCall = nanMinMaxClamp ? spv::GLSLstd450NClamp : spv::GLSLstd450FClamp;
        else if (isUnsigned)
            libCall = spv::GLSLstd450UClamp;
        else
            libCall = spv::GLSLstd450SClamp;
        builder.promoteScalar(precision, operands.front(), operands[1]);
        builder.promoteScalar(precision, operands.front(), operands[2]);
        break;
    case glslang::EOpMix:
        if (! builder.isBoolType(builder.getScalarTypeId(builder.getTypeId(operands.back())))) {
            assert(isFloat);
            libCall = spv::GLSLstd450FMix;
        } else {
            opCode = spv::Op::OpSelect;
            std::swap(operands.front(), operands.back());
        }
        builder.promoteScalar(precision, operands.front(), operands.back());
        break;
    case glslang::EOpStep:
        libCall = spv::GLSLstd450Step;
        builder.promoteScalar(precision, operands.front(), operands.back());
        break;
    case glslang::EOpSmoothStep:
        libCall = spv::GLSLstd450SmoothStep;
        builder.promoteScalar(precision, operands[0], operands[2]);
        builder.promoteScalar(precision, operands[1], operands[2]);
        break;

    case glslang::EOpDistance:
        libCall = spv::GLSLstd450Distance;
        break;
    case glslang::EOpCross:
        libCall = spv::GLSLstd450Cross;
        break;
    case glslang::EOpFaceForward:
        libCall = spv::GLSLstd450FaceForward;
        break;
    case glslang::EOpReflect:
        libCall = spv::GLSLstd450Reflect;
        break;
    case glslang::EOpRefract:
        libCall = spv::GLSLstd450Refract;
        break;
    case glslang::EOpBarrier:
        {
            // This is for the extended controlBarrier function, with four operands.
            // The unextended barrier() goes through createNoArgOperation.
            assert(operands.size() == 4);
            auto const executionScope = (spv::Scope)builder.getConstantScalar(operands[0]);
            auto const memoryScope = (spv::Scope)builder.getConstantScalar(operands[1]);
            auto const semantics = (spv::MemorySemanticsMask)(builder.getConstantScalar(operands[2]) | builder.getConstantScalar(operands[3]));
            builder.createControlBarrier(executionScope, memoryScope,
                semantics);
            if (anySet(semantics, spv::MemorySemanticsMask::MakeAvailableKHR |
                                  spv::MemorySemanticsMask::MakeVisibleKHR |
                                  spv::MemorySemanticsMask::OutputMemoryKHR |
                                  spv::MemorySemanticsMask::Volatile)) {
                builder.addCapability(spv::Capability::VulkanMemoryModelKHR);
            }
            if (glslangIntermediate->usingVulkanMemoryModel() && (executionScope == spv::Scope::Device ||
                memoryScope == spv::Scope::Device)) {
                builder.addCapability(spv::Capability::VulkanMemoryModelDeviceScopeKHR);
            }
            return 0;
        }
        break;
    case glslang::EOpMemoryBarrier:
        {
            // This is for the extended memoryBarrier function, with three operands.
            // The unextended memoryBarrier() goes through createNoArgOperation.
            assert(operands.size() == 3);
            auto const memoryScope = (spv::Scope)builder.getConstantScalar(operands[0]);
            auto const semantics = (spv::MemorySemanticsMask)(builder.getConstantScalar(operands[1]) | builder.getConstantScalar(operands[2]));
            builder.createMemoryBarrier(memoryScope, semantics);
            if (anySet(semantics, spv::MemorySemanticsMask::MakeAvailableKHR |
                                  spv::MemorySemanticsMask::MakeVisibleKHR |
                                  spv::MemorySemanticsMask::OutputMemoryKHR |
                                  spv::MemorySemanticsMask::Volatile)) {
                builder.addCapability(spv::Capability::VulkanMemoryModelKHR);
            }
            if (glslangIntermediate->usingVulkanMemoryModel() && memoryScope == spv::Scope::Device) {
                builder.addCapability(spv::Capability::VulkanMemoryModelDeviceScopeKHR);
            }
            return 0;
        }
        break;

    case glslang::EOpInterpolateAtSample:
        if (typeProxy == glslang::EbtFloat16)
            builder.addExtension(spv::E_SPV_AMD_gpu_shader_half_float);
        libCall = spv::GLSLstd450InterpolateAtSample;
        break;
    case glslang::EOpInterpolateAtOffset:
        if (typeProxy == glslang::EbtFloat16)
            builder.addExtension(spv::E_SPV_AMD_gpu_shader_half_float);
        libCall = spv::GLSLstd450InterpolateAtOffset;
        break;
    case glslang::EOpAddCarry:
        opCode = spv::Op::OpIAddCarry;
        typeId = builder.makeStructResultType(typeId0, typeId0);
        consumedOperands = 2;
        break;
    case glslang::EOpSubBorrow:
        opCode = spv::Op::OpISubBorrow;
        typeId = builder.makeStructResultType(typeId0, typeId0);
        consumedOperands = 2;
        break;
    case glslang::EOpUMulExtended:
        opCode = spv::Op::OpUMulExtended;
        typeId = builder.makeStructResultType(typeId0, typeId0);
        consumedOperands = 2;
        break;
    case glslang::EOpIMulExtended:
        opCode = spv::Op::OpSMulExtended;
        typeId = builder.makeStructResultType(typeId0, typeId0);
        consumedOperands = 2;
        break;
    case glslang::EOpBitfieldExtract:
        if (isUnsigned)
            opCode = spv::Op::OpBitFieldUExtract;
        else
            opCode = spv::Op::OpBitFieldSExtract;
        break;
    case glslang::EOpBitfieldInsert:
        opCode = spv::Op::OpBitFieldInsert;
        break;

    case glslang::EOpFma:
        libCall = spv::GLSLstd450Fma;
        break;
    case glslang::EOpFrexp:
        {
            libCall = spv::GLSLstd450FrexpStruct;
            assert(builder.isPointerType(typeId1));
            typeId1 = builder.getContainedTypeId(typeId1);
            int width = builder.getScalarTypeWidth(typeId1);
            if (width == 16)
                // Using 16-bit exp operand, enable extension SPV_AMD_gpu_shader_int16
                builder.addExtension(spv::E_SPV_AMD_gpu_shader_int16);
            if (builder.getNumComponents(operands[0]) == 1)
                frexpIntType = builder.makeIntegerType(width, true);
            else
                frexpIntType = builder.makeVectorType(builder.makeIntegerType(width, true),
                    builder.getNumComponents(operands[0]));
            typeId = builder.makeStructResultType(typeId0, frexpIntType);
            consumedOperands = 1;
        }
        break;
    case glslang::EOpLdexp:
        libCall = spv::GLSLstd450Ldexp;
        break;

    case glslang::EOpReadInvocation:
        return createInvocationsOperation(op, typeId, operands, typeProxy);

    case glslang::EOpSubgroupBroadcast:
    case glslang::EOpSubgroupBallotBitExtract:
    case glslang::EOpSubgroupShuffle:
    case glslang::EOpSubgroupShuffleXor:
    case glslang::EOpSubgroupShuffleUp:
    case glslang::EOpSubgroupShuffleDown:
    case glslang::EOpSubgroupRotate:
    case glslang::EOpSubgroupClusteredRotate:
    case glslang::EOpSubgroupClusteredAdd:
    case glslang::EOpSubgroupClusteredMul:
    case glslang::EOpSubgroupClusteredMin:
    case glslang::EOpSubgroupClusteredMax:
    case glslang::EOpSubgroupClusteredAnd:
    case glslang::EOpSubgroupClusteredOr:
    case glslang::EOpSubgroupClusteredXor:
    case glslang::EOpSubgroupQuadBroadcast:
    case glslang::EOpSubgroupPartitionedAdd:
    case glslang::EOpSubgroupPartitionedMul:
    case glslang::EOpSubgroupPartitionedMin:
    case glslang::EOpSubgroupPartitionedMax:
    case glslang::EOpSubgroupPartitionedAnd:
    case glslang::EOpSubgroupPartitionedOr:
    case glslang::EOpSubgroupPartitionedXor:
    case glslang::EOpSubgroupPartitionedInclusiveAdd:
    case glslang::EOpSubgroupPartitionedInclusiveMul:
    case glslang::EOpSubgroupPartitionedInclusiveMin:
    case glslang::EOpSubgroupPartitionedInclusiveMax:
    case glslang::EOpSubgroupPartitionedInclusiveAnd:
    case glslang::EOpSubgroupPartitionedInclusiveOr:
    case glslang::EOpSubgroupPartitionedInclusiveXor:
    case glslang::EOpSubgroupPartitionedExclusiveAdd:
    case glslang::EOpSubgroupPartitionedExclusiveMul:
    case glslang::EOpSubgroupPartitionedExclusiveMin:
    case glslang::EOpSubgroupPartitionedExclusiveMax:
    case glslang::EOpSubgroupPartitionedExclusiveAnd:
    case glslang::EOpSubgroupPartitionedExclusiveOr:
    case glslang::EOpSubgroupPartitionedExclusiveXor:
        return createSubgroupOperation(op, typeId, operands, typeProxy);

    case glslang::EOpSwizzleInvocations:
        extBuiltins = getExtBuiltins(spv::E_SPV_AMD_shader_ballot);
        libCall = spv::SwizzleInvocationsAMD;
        break;
    case glslang::EOpSwizzleInvocationsMasked:
        extBuiltins = getExtBuiltins(spv::E_SPV_AMD_shader_ballot);
        libCall = spv::SwizzleInvocationsMaskedAMD;
        break;
    case glslang::EOpWriteInvocation:
        extBuiltins = getExtBuiltins(spv::E_SPV_AMD_shader_ballot);
        libCall = spv::WriteInvocationAMD;
        break;

    case glslang::EOpMin3:
        extBuiltins = getExtBuiltins(spv::E_SPV_AMD_shader_trinary_minmax);
        if (isFloat)
            libCall = spv::FMin3AMD;
        else {
            if (isUnsigned)
                libCall = spv::UMin3AMD;
            else
                libCall = spv::SMin3AMD;
        }
        break;
    case glslang::EOpMax3:
        extBuiltins = getExtBuiltins(spv::E_SPV_AMD_shader_trinary_minmax);
        if (isFloat)
            libCall = spv::FMax3AMD;
        else {
            if (isUnsigned)
                libCall = spv::UMax3AMD;
            else
                libCall = spv::SMax3AMD;
        }
        break;
    case glslang::EOpMid3:
        extBuiltins = getExtBuiltins(spv::E_SPV_AMD_shader_trinary_minmax);
        if (isFloat)
            libCall = spv::FMid3AMD;
        else {
            if (isUnsigned)
                libCall = spv::UMid3AMD;
            else
                libCall = spv::SMid3AMD;
        }
        break;

    case glslang::EOpInterpolateAtVertex:
        if (typeProxy == glslang::EbtFloat16)
            builder.addExtension(spv::E_SPV_AMD_gpu_shader_half_float);
        extBuiltins = getExtBuiltins(spv::E_SPV_AMD_shader_explicit_vertex_parameter);
        libCall = spv::InterpolateAtVertexAMD;
        break;

    case glslang::EOpReportIntersection:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpReportIntersectionKHR;
        break;
    case glslang::EOpTraceNV:
        builder.createNoResultOp(spv::Op::OpTraceNV, operands);
        return 0;
    case glslang::EOpTraceRayMotionNV:
        builder.addExtension(spv::E_SPV_NV_ray_tracing_motion_blur);
        builder.addCapability(spv::Capability::RayTracingMotionBlurNV);
        builder.createNoResultOp(spv::Op::OpTraceRayMotionNV, operands);
        return 0;
    case glslang::EOpTraceKHR:
        builder.createNoResultOp(spv::Op::OpTraceRayKHR, operands);
        return 0;
    case glslang::EOpExecuteCallableNV:
        builder.createNoResultOp(spv::Op::OpExecuteCallableNV, operands);
        return 0;
    case glslang::EOpExecuteCallableKHR:
        builder.createNoResultOp(spv::Op::OpExecuteCallableKHR, operands);
        return 0;

    case glslang::EOpRayQueryInitialize:
        builder.createNoResultOp(spv::Op::OpRayQueryInitializeKHR, operands);
        return 0;
    case glslang::EOpRayQueryTerminate:
        builder.createNoResultOp(spv::Op::OpRayQueryTerminateKHR, operands);
        return 0;
    case glslang::EOpRayQueryGenerateIntersection:
        builder.createNoResultOp(spv::Op::OpRayQueryGenerateIntersectionKHR, operands);
        return 0;
    case glslang::EOpRayQueryConfirmIntersection:
        builder.createNoResultOp(spv::Op::OpRayQueryConfirmIntersectionKHR, operands);
        return 0;
    case glslang::EOpRayQueryProceed:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpRayQueryProceedKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionType:
        typeId = builder.makeUintType(32);
        opCode = spv::Op::OpRayQueryGetIntersectionTypeKHR;
        break;
    case glslang::EOpRayQueryGetRayTMin:
        typeId = builder.makeFloatType(32);
        opCode = spv::Op::OpRayQueryGetRayTMinKHR;
        break;
    case glslang::EOpRayQueryGetRayFlags:
        typeId = builder.makeIntType(32);
        opCode = spv::Op::OpRayQueryGetRayFlagsKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionT:
        typeId = builder.makeFloatType(32);
        opCode = spv::Op::OpRayQueryGetIntersectionTKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionInstanceCustomIndex:
        typeId = builder.makeIntType(32);
        opCode = spv::Op::OpRayQueryGetIntersectionInstanceCustomIndexKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionInstanceId:
        typeId = builder.makeIntType(32);
        opCode = spv::Op::OpRayQueryGetIntersectionInstanceIdKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffset:
        typeId = builder.makeUintType(32);
        opCode = spv::Op::OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionGeometryIndex:
        typeId = builder.makeIntType(32);
        opCode = spv::Op::OpRayQueryGetIntersectionGeometryIndexKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionPrimitiveIndex:
        typeId = builder.makeIntType(32);
        opCode = spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionBarycentrics:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 2);
        opCode = spv::Op::OpRayQueryGetIntersectionBarycentricsKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionFrontFace:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpRayQueryGetIntersectionFrontFaceKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionCandidateAABBOpaque:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpRayQueryGetIntersectionCandidateAABBOpaqueKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionObjectRayDirection:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpRayQueryGetIntersectionObjectRayDirectionKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionObjectRayOrigin:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpRayQueryGetIntersectionObjectRayOriginKHR;
        break;
    case glslang::EOpRayQueryGetWorldRayDirection:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpRayQueryGetWorldRayDirectionKHR;
        break;
    case glslang::EOpRayQueryGetWorldRayOrigin:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpRayQueryGetWorldRayOriginKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionObjectToWorld:
        typeId = builder.makeMatrixType(builder.makeFloatType(32), 4, 3);
        opCode = spv::Op::OpRayQueryGetIntersectionObjectToWorldKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionClusterIdNV:
        typeId = builder.makeIntegerType(32, 1);
        opCode = spv::Op::OpRayQueryGetClusterIdNV;
        break;
    case glslang::EOpRayQueryGetIntersectionWorldToObject:
        typeId = builder.makeMatrixType(builder.makeFloatType(32), 4, 3);
        opCode = spv::Op::OpRayQueryGetIntersectionWorldToObjectKHR;
        break;
    case glslang::EOpRayQueryGetIntersectionSpherePositionNV:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpRayQueryGetIntersectionSpherePositionNV;
        break;
    case glslang::EOpRayQueryGetIntersectionSphereRadiusNV:
        typeId = builder.makeFloatType(32);
        opCode = spv::Op::OpRayQueryGetIntersectionSphereRadiusNV;
        break;
    case glslang::EOpRayQueryGetIntersectionLSSHitValueNV:
        typeId = builder.makeFloatType(32);
        opCode = spv::Op::OpRayQueryGetIntersectionLSSHitValueNV;
        break;
    case glslang::EOpRayQueryIsSphereHitNV:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpRayQueryIsSphereHitNV;
        break;
    case glslang::EOpRayQueryIsLSSHitNV:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpRayQueryIsLSSHitNV;
        break;
    case glslang::EOpWritePackedPrimitiveIndices4x8NV:
        builder.createNoResultOp(spv::Op::OpWritePackedPrimitiveIndices4x8NV, operands);
        return 0;
    case glslang::EOpEmitMeshTasksEXT:
        if (taskPayloadID)
            operands.push_back(taskPayloadID);
        // As per SPV_EXT_mesh_shader make it a terminating instruction in the current block
        builder.makeStatementTerminator(spv::Op::OpEmitMeshTasksEXT, operands, "post-OpEmitMeshTasksEXT");
        return 0;
    case glslang::EOpSetMeshOutputsEXT:
        builder.createNoResultOp(spv::Op::OpSetMeshOutputsEXT, operands);
        return 0;
    case glslang::EOpCooperativeMatrixMulAddNV:
        opCode = spv::Op::OpCooperativeMatrixMulAddNV;
        break;
    case glslang::EOpHitObjectTraceRayNV:
        builder.createNoResultOp(spv::Op::OpHitObjectTraceRayNV, operands);
        return 0;
    case glslang::EOpHitObjectTraceRayEXT:
        builder.createNoResultOp(spv::Op::OpHitObjectTraceRayEXT, operands);
        return 0;
    case glslang::EOpHitObjectTraceRayMotionNV:
        builder.createNoResultOp(spv::Op::OpHitObjectTraceRayMotionNV, operands);
        return 0;
    case glslang::EOpHitObjectTraceRayMotionEXT:
        builder.createNoResultOp(spv::Op::OpHitObjectTraceRayMotionEXT, operands);
        return 0;
    case glslang::EOpHitObjectRecordHitNV:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordHitNV, operands);
        return 0;
    case glslang::EOpHitObjectRecordHitMotionNV:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordHitMotionNV, operands);
        return 0;
    case glslang::EOpHitObjectRecordHitWithIndexNV:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordHitWithIndexNV, operands);
        return 0;
    case glslang::EOpHitObjectRecordHitWithIndexMotionNV:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordHitWithIndexMotionNV, operands);
        return 0;
    case glslang::EOpHitObjectRecordMissNV:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordMissNV, operands);
        return 0;
    case glslang::EOpHitObjectRecordMissEXT:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordMissEXT, operands);
        return 0;
    case glslang::EOpHitObjectRecordMissMotionNV:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordMissMotionNV, operands);
        return 0;
    case glslang::EOpHitObjectRecordMissMotionEXT:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordMissMotionEXT, operands);
        return 0;
    case glslang::EOpHitObjectExecuteShaderNV:
        builder.createNoResultOp(spv::Op::OpHitObjectExecuteShaderNV, operands);
        return 0;
    case glslang::EOpHitObjectExecuteShaderEXT:
        builder.createNoResultOp(spv::Op::OpHitObjectExecuteShaderEXT, operands);
        return 0;
    case glslang::EOpHitObjectIsEmptyNV:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpHitObjectIsEmptyNV;
        break;
    case glslang::EOpHitObjectIsEmptyEXT:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpHitObjectIsEmptyEXT;
        break;
    case glslang::EOpHitObjectIsMissNV:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpHitObjectIsMissNV;
        break;
    case glslang::EOpHitObjectIsMissEXT:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpHitObjectIsMissEXT;
        break;
    case glslang::EOpHitObjectIsHitNV:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpHitObjectIsHitNV;
        break;
    case glslang::EOpHitObjectIsSphereHitNV:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpHitObjectIsSphereHitNV;
        break;
    case glslang::EOpHitObjectIsLSSHitNV:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpHitObjectIsLSSHitNV;
        break;
    case glslang::EOpHitObjectIsHitEXT:
        typeId = builder.makeBoolType();
        opCode = spv::Op::OpHitObjectIsHitEXT;
        break;
    case glslang::EOpHitObjectGetRayTMinNV:
        typeId = builder.makeFloatType(32);
        opCode = spv::Op::OpHitObjectGetRayTMinNV;
        break;
    case glslang::EOpHitObjectGetRayTMinEXT:
        typeId = builder.makeFloatType(32);
        opCode = spv::Op::OpHitObjectGetRayTMinEXT;
        break;
    case glslang::EOpHitObjectGetRayTMaxNV:
        typeId = builder.makeFloatType(32);
        opCode = spv::Op::OpHitObjectGetRayTMaxNV;
        break;
    case glslang::EOpHitObjectGetRayTMaxEXT:
        typeId = builder.makeFloatType(32);
        opCode = spv::Op::OpHitObjectGetRayTMaxEXT;
        break;
    case glslang::EOpHitObjectGetRayFlagsEXT:
        typeId = builder.makeIntegerType(32, 0);
        opCode = spv::Op::OpHitObjectGetRayFlagsEXT;
        break;
    case glslang::EOpHitObjectGetObjectRayOriginNV:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpHitObjectGetObjectRayOriginNV;
        break;
    case glslang::EOpHitObjectGetObjectRayOriginEXT:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpHitObjectGetObjectRayOriginEXT;
        break;
    case glslang::EOpHitObjectGetObjectRayDirectionNV:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpHitObjectGetObjectRayDirectionNV;
        break;
    case glslang::EOpHitObjectGetObjectRayDirectionEXT:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpHitObjectGetObjectRayDirectionEXT;
        break;
    case glslang::EOpHitObjectGetWorldRayOriginNV:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpHitObjectGetWorldRayOriginNV;
        break;
    case glslang::EOpHitObjectGetWorldRayOriginEXT:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpHitObjectGetWorldRayOriginEXT;
        break;
    case glslang::EOpHitObjectGetWorldRayDirectionNV:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpHitObjectGetWorldRayDirectionNV;
        break;
    case glslang::EOpHitObjectGetWorldRayDirectionEXT:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpHitObjectGetWorldRayDirectionEXT;
        break;
    case glslang::EOpHitObjectGetWorldToObjectNV:
        typeId = builder.makeMatrixType(builder.makeFloatType(32), 4, 3);
        opCode = spv::Op::OpHitObjectGetWorldToObjectNV;
        break;
    case glslang::EOpHitObjectGetWorldToObjectEXT:
        typeId = builder.makeMatrixType(builder.makeFloatType(32), 4, 3);
        opCode = spv::Op::OpHitObjectGetWorldToObjectEXT;
        break;
    case glslang::EOpHitObjectGetObjectToWorldNV:
        typeId = builder.makeMatrixType(builder.makeFloatType(32), 4, 3);
        opCode = spv::Op::OpHitObjectGetObjectToWorldNV;
        break;
    case glslang::EOpHitObjectGetObjectToWorldEXT:
        typeId = builder.makeMatrixType(builder.makeFloatType(32), 4, 3);
        opCode = spv::Op::OpHitObjectGetObjectToWorldEXT;
        break;
    case glslang::EOpHitObjectGetInstanceCustomIndexNV:
        typeId = builder.makeIntegerType(32, 1);
        opCode = spv::Op::OpHitObjectGetInstanceCustomIndexNV;
        break;
    case glslang::EOpHitObjectGetInstanceCustomIndexEXT:
        typeId = builder.makeIntegerType(32, 1);
        opCode = spv::Op::OpHitObjectGetInstanceCustomIndexEXT;
        break;
    case glslang::EOpHitObjectGetInstanceIdNV:
        typeId = builder.makeIntegerType(32, 1);
        opCode = spv::Op::OpHitObjectGetInstanceIdNV;
        break;
    case glslang::EOpHitObjectGetInstanceIdEXT:
        typeId = builder.makeIntegerType(32, 1);
        opCode = spv::Op::OpHitObjectGetInstanceIdEXT;
        break;
    case glslang::EOpHitObjectGetGeometryIndexNV:
        typeId = builder.makeIntegerType(32, 1);
        opCode = spv::Op::OpHitObjectGetGeometryIndexNV;
        break;
    case glslang::EOpHitObjectGetGeometryIndexEXT:
        typeId = builder.makeIntegerType(32, 1);
        opCode = spv::Op::OpHitObjectGetGeometryIndexEXT;
        break;
    case glslang::EOpHitObjectGetPrimitiveIndexNV:
        typeId = builder.makeIntegerType(32, 1);
        opCode = spv::Op::OpHitObjectGetPrimitiveIndexNV;
        break;
    case glslang::EOpHitObjectGetPrimitiveIndexEXT:
        typeId = builder.makeIntegerType(32, 1);
        opCode = spv::Op::OpHitObjectGetPrimitiveIndexEXT;
        break;
    case glslang::EOpHitObjectGetHitKindNV:
        typeId = builder.makeIntegerType(32, 0);
        opCode = spv::Op::OpHitObjectGetHitKindNV;
        break;
    case glslang::EOpHitObjectGetHitKindEXT:
        typeId = builder.makeIntegerType(32, 0);
        opCode = spv::Op::OpHitObjectGetHitKindEXT;
        break;
    case glslang::EOpHitObjectGetCurrentTimeNV:
        typeId = builder.makeFloatType(32);
        opCode = spv::Op::OpHitObjectGetCurrentTimeNV;
        break;
    case glslang::EOpHitObjectGetCurrentTimeEXT:
        typeId = builder.makeFloatType(32);
        opCode = spv::Op::OpHitObjectGetCurrentTimeEXT;
        break;
    case glslang::EOpHitObjectGetShaderBindingTableRecordIndexNV:
        typeId = builder.makeIntegerType(32, 0);
        opCode = spv::Op::OpHitObjectGetShaderBindingTableRecordIndexNV;
        return 0;
    case glslang::EOpHitObjectGetShaderBindingTableRecordIndexEXT:
        typeId = builder.makeIntegerType(32, 0);
        opCode = spv::Op::OpHitObjectGetShaderBindingTableRecordIndexEXT;
        return 0;
    case glslang::EOpHitObjectGetAttributesNV:
        builder.createNoResultOp(spv::Op::OpHitObjectGetAttributesNV, operands);
        return 0;
    case glslang::EOpHitObjectGetAttributesEXT:
        builder.createNoResultOp(spv::Op::OpHitObjectGetAttributesEXT, operands);
        return 0;
    case glslang::EOpHitObjectRecordFromQueryEXT:
        builder.createNoResultOp(spv::Op::OpHitObjectRecordFromQueryEXT, operands);
        return 0;
    case glslang::EOpHitObjectGetShaderRecordBufferHandleNV:
        typeId = builder.makeVectorType(builder.makeUintType(32), 2);
        opCode = spv::Op::OpHitObjectGetShaderRecordBufferHandleNV;
        break;
    case glslang::EOpHitObjectGetClusterIdNV:
        typeId = builder.makeIntegerType(32, 1);
        opCode = spv::Op::OpHitObjectGetClusterIdNV;
        break;
    case glslang::EOpHitObjectGetShaderRecordBufferHandleEXT:
        typeId = builder.makeVectorType(builder.makeUintType(32), 2);
        opCode = spv::Op::OpHitObjectGetShaderRecordBufferHandleEXT;
        break;
    case glslang::EOpHitObjectSetShaderBindingTableRecordIndexEXT:
        builder.createNoResultOp(spv::Op::OpHitObjectSetShaderBindingTableRecordIndexEXT, operands);
        return 0;
    case glslang::EOpReorderThreadNV: {
        if (operands.size() == 2) {
            builder.createNoResultOp(spv::Op::OpReorderThreadWithHintNV, operands);
        } else {
            builder.createNoResultOp(spv::Op::OpReorderThreadWithHitObjectNV, operands);
        }
        return 0;
    }
    case glslang::EOpReorderThreadEXT: {
        if (operands.size() == 2) {
            builder.createNoResultOp(spv::Op::OpReorderThreadWithHintEXT, operands);
        } else {
            builder.createNoResultOp(spv::Op::OpReorderThreadWithHitObjectEXT, operands);
        }
        return 0;
    }

    case glslang::EOpHitObjectReorderExecuteEXT: {
        if (operands.size() == 2) {
            builder.createNoResultOp(spv::Op::OpHitObjectReorderExecuteShaderEXT, operands);
        } else {
            // GLSL intrinsic is
            // hitObjectReorderExecuteEXT(hitObjectEXT hitObject, uint hint, uint bits,int payload) while
            // SPIRV is hitObject id , payload id, optional hint id, optional bits id hence reorder operands
            builder.createNoResultOp(spv::Op::OpHitObjectReorderExecuteShaderEXT, {operands[0], operands[3], operands[1], operands[2]});
        }
        return 0;
    }

    case glslang::EOpHitObjectTraceReorderExecuteEXT: {
        if (operands.size() == 12) {
            builder.createNoResultOp(spv::Op::OpHitObjectTraceReorderExecuteEXT, operands);
        } else {
            std::vector<spv::Id> argOperands;
            std::copy(operands.begin(), operands.begin() + 11, std::back_inserter(argOperands));
            argOperands.push_back(operands[13]);
            argOperands.push_back(operands[11]);
            argOperands.push_back(operands[12]);
            builder.createNoResultOp(spv::Op::OpHitObjectTraceReorderExecuteEXT, argOperands);
        }
        return 0;
    }
    case glslang::EOpHitObjectTraceMotionReorderExecuteEXT: {
        if (operands.size() == 13) {
            builder.createNoResultOp(spv::Op::OpHitObjectTraceMotionReorderExecuteEXT, operands);
        } else {
            std::vector<spv::Id> argOperands;
            std::copy(operands.begin(), operands.begin() + 12, std::back_inserter(argOperands));
            argOperands.push_back(operands[14]);
            argOperands.push_back(operands[12]);
            argOperands.push_back(operands[13]);
            builder.createNoResultOp(spv::Op::OpHitObjectTraceMotionReorderExecuteEXT, argOperands);
        }
        return 0;
    }
    case glslang::EOpImageSampleWeightedQCOM:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 4);
        opCode = spv::Op::OpImageSampleWeightedQCOM;
        addImageProcessingQCOMDecoration(operands[2], spv::Decoration::WeightTextureQCOM);
        break;
    case glslang::EOpImageBoxFilterQCOM:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 4);
        opCode = spv::Op::OpImageBoxFilterQCOM;
        break;
    case glslang::EOpImageBlockMatchSADQCOM:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 4);
        opCode = spv::Op::OpImageBlockMatchSADQCOM;
        addImageProcessingQCOMDecoration(operands[0], spv::Decoration::BlockMatchTextureQCOM);
        addImageProcessingQCOMDecoration(operands[2], spv::Decoration::BlockMatchTextureQCOM);
        break;
    case glslang::EOpImageBlockMatchSSDQCOM:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 4);
        opCode = spv::Op::OpImageBlockMatchSSDQCOM;
        addImageProcessingQCOMDecoration(operands[0], spv::Decoration::BlockMatchTextureQCOM);
        addImageProcessingQCOMDecoration(operands[2], spv::Decoration::BlockMatchTextureQCOM);
        break;

    case glslang::EOpFetchMicroTriangleVertexBarycentricNV:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 2);
        opCode = spv::Op::OpFetchMicroTriangleVertexBarycentricNV;
        break;

    case glslang::EOpFetchMicroTriangleVertexPositionNV:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 3);
        opCode = spv::Op::OpFetchMicroTriangleVertexPositionNV;
        break;

    case glslang::EOpImageBlockMatchWindowSSDQCOM:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 4);
        opCode = spv::Op::OpImageBlockMatchWindowSSDQCOM;
        addImageProcessing2QCOMDecoration(operands[0], false);
        addImageProcessing2QCOMDecoration(operands[2], false);
        break;
    case glslang::EOpImageBlockMatchWindowSADQCOM:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 4);
        opCode = spv::Op::OpImageBlockMatchWindowSADQCOM;
        addImageProcessing2QCOMDecoration(operands[0], false);
        addImageProcessing2QCOMDecoration(operands[2], false);
        break;
    case glslang::EOpImageBlockMatchGatherSSDQCOM:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 4);
        opCode = spv::Op::OpImageBlockMatchGatherSSDQCOM;
        addImageProcessing2QCOMDecoration(operands[0], true);
        addImageProcessing2QCOMDecoration(operands[2], true);
        break;
    case glslang::EOpImageBlockMatchGatherSADQCOM:
        typeId = builder.makeVectorType(builder.makeFloatType(32), 4);
        opCode = spv::Op::OpImageBlockMatchGatherSADQCOM;
        addImageProcessing2QCOMDecoration(operands[0], true);
        addImageProcessing2QCOMDecoration(operands[2], true);
        break;
    case glslang::EOpCreateTensorLayoutNV:
        return builder.createOp(spv::Op::OpCreateTensorLayoutNV, typeId, std::vector<spv::Id>{});
    case glslang::EOpCreateTensorViewNV:
        return builder.createOp(spv::Op::OpCreateTensorViewNV, typeId, std::vector<spv::Id>{});
    case glslang::EOpTensorLayoutSetBlockSizeNV:
        opCode = spv::Op::OpTensorLayoutSetBlockSizeNV;
        break;
    case glslang::EOpTensorLayoutSetDimensionNV:
        opCode = spv::Op::OpTensorLayoutSetDimensionNV;
        break;
    case glslang::EOpTensorLayoutSetStrideNV:
        opCode = spv::Op::OpTensorLayoutSetStrideNV;
        break;
    case glslang::EOpTensorLayoutSliceNV:
        opCode = spv::Op::OpTensorLayoutSliceNV;
        break;
    case glslang::EOpTensorLayoutSetClampValueNV:
        opCode = spv::Op::OpTensorLayoutSetClampValueNV;
        break;
    case glslang::EOpTensorViewSetDimensionNV:
        opCode = spv::Op::OpTensorViewSetDimensionNV;
        break;
    case glslang::EOpTensorViewSetStrideNV:
        opCode = spv::Op::OpTensorViewSetStrideNV;
        break;
    case glslang::EOpTensorViewSetClipNV:
        opCode = spv::Op::OpTensorViewSetClipNV;
        break;
    default:
        return 0;
    }

    spv::Id id = 0;
    if (libCall >= 0) {
        // Use an extended instruction from the standard library.
        // Construct the call arguments, without modifying the original operands vector.
        // We might need the remaining arguments, e.g. in the EOpFrexp case.
        std::vector<spv::Id> callArguments(operands.begin(), operands.begin() + consumedOperands);
        id = builder.createBuiltinCall(typeId, extBuiltins >= 0 ? extBuiltins : stdBuiltins, libCall, callArguments);
    } else if (opCode == spv::Op::OpDot && !isFloat) {
        // int dot(int, int)
        // NOTE: never called for scalar/vector1, this is turned into simple mul before this can be reached
        const int componentCount = builder.getNumComponents(operands[0]);
        spv::Id mulOp = builder.createBinOp(spv::Op::OpIMul, builder.getTypeId(operands[0]), operands[0], operands[1]);
        builder.setPrecision(mulOp, precision);
        id = builder.createCompositeExtract(mulOp, typeId, 0);
        for (int i = 1; i < componentCount; ++i) {
            builder.setPrecision(id, precision);
            id = builder.createBinOp(spv::Op::OpIAdd, typeId, id, builder.createCompositeExtract(mulOp, typeId, i));
        }
    } else {
        switch (consumedOperands) {
        case 0:
            // should all be handled by visitAggregate and createNoArgOperation
            assert(0);
            return 0;
        case 1:
            // should all be handled by createUnaryOperation
            assert(0);
            return 0;
        case 2:
            id = builder.createBinOp(opCode, typeId, operands[0], operands[1]);
            break;
        default:
            // anything 3 or over doesn't have l-value operands, so all should be consumed
            assert(consumedOperands == operands.size());
            id = builder.createOp(opCode, typeId, operands);
            break;
        }
    }

    // Decode the return types that were structures
    switch (op) {
    case glslang::EOpAddCarry:
    case glslang::EOpSubBorrow:
        builder.createStore(builder.createCompositeExtract(id, typeId0, 1), operands[2]);
        id = builder.createCompositeExtract(id, typeId0, 0);
        break;
    case glslang::EOpUMulExtended:
    case glslang::EOpIMulExtended:
        builder.createStore(builder.createCompositeExtract(id, typeId0, 0), operands[3]);
        builder.createStore(builder.createCompositeExtract(id, typeId0, 1), operands[2]);
        break;
    case glslang::EOpModf:
        {
            assert(operands.size() == 2);
            builder.createStore(builder.createCompositeExtract(id, typeId0, 1), operands[1]);
            id = builder.createCompositeExtract(id, typeId0, 0);
        }
        break;
    case glslang::EOpFrexp:
        {
            assert(operands.size() == 2);
            if (builder.isFloatType(builder.getScalarTypeId(typeId1))) {
                // "exp" is floating-point type (from HLSL intrinsic)
                spv::Id member1 = builder.createCompositeExtract(id, frexpIntType, 1);
                member1 = builder.createUnaryOp(spv::Op::OpConvertSToF, typeId1, member1);
                builder.createStore(member1, operands[1]);
            } else
                // "exp" is integer type (from GLSL built-in function)
                builder.createStore(builder.createCompositeExtract(id, frexpIntType, 1), operands[1]);
            id = builder.createCompositeExtract(id, typeId0, 0);
        }
        break;
    default:
        break;
    }

    return builder.setPrecision(id, precision);
}

// Intrinsics with no arguments (or no return value, and no precision).
spv::Id TGlslangToSpvTraverser::createNoArgOperation(glslang::TOperator op, spv::Decoration precision, spv::Id typeId)
{
    // GLSL memory barriers use queuefamily scope in new model, device scope in old model
    spv::Scope memoryBarrierScope = glslangIntermediate->usingVulkanMemoryModel() ?
        spv::Scope::QueueFamilyKHR : spv::Scope::Device;

    switch (op) {
    case glslang::EOpBarrier:
        if (glslangIntermediate->getStage() == EShLangTessControl) {
            if (glslangIntermediate->usingVulkanMemoryModel()) {
                builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Workgroup,
                                             spv::MemorySemanticsMask::OutputMemoryKHR |
                                             spv::MemorySemanticsMask::AcquireRelease);
                builder.addCapability(spv::Capability::VulkanMemoryModelKHR);
            } else {
                builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Invocation, spv::MemorySemanticsMask::MaskNone);
            }
        } else {
            builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Workgroup,
                                            spv::MemorySemanticsMask::WorkgroupMemory |
                                            spv::MemorySemanticsMask::AcquireRelease);
        }
        return 0;
    case glslang::EOpMemoryBarrier:
        builder.createMemoryBarrier(memoryBarrierScope, spv::MemorySemanticsAllMemory |
                                                        spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpMemoryBarrierBuffer:
        builder.createMemoryBarrier(memoryBarrierScope, spv::MemorySemanticsMask::UniformMemory |
                                                        spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpMemoryBarrierShared:
        builder.createMemoryBarrier(memoryBarrierScope, spv::MemorySemanticsMask::WorkgroupMemory |
                                                        spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpGroupMemoryBarrier:
        builder.createMemoryBarrier(spv::Scope::Workgroup, spv::MemorySemanticsAllMemory |
                                                           spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpMemoryBarrierAtomicCounter:
        builder.createMemoryBarrier(memoryBarrierScope, spv::MemorySemanticsMask::AtomicCounterMemory |
                                                        spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpMemoryBarrierImage:
        builder.createMemoryBarrier(memoryBarrierScope, spv::MemorySemanticsMask::ImageMemory |
                                                        spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpAllMemoryBarrierWithGroupSync:
        builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Device,
                                        spv::MemorySemanticsAllMemory |
                                        spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpDeviceMemoryBarrier:
        builder.createMemoryBarrier(spv::Scope::Device, spv::MemorySemanticsMask::UniformMemory |
                                                        spv::MemorySemanticsMask::ImageMemory |
                                                        spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpDeviceMemoryBarrierWithGroupSync:
        builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Device, spv::MemorySemanticsMask::UniformMemory |
                                                                                spv::MemorySemanticsMask::ImageMemory |
                                                                                spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpWorkgroupMemoryBarrier:
        builder.createMemoryBarrier(spv::Scope::Workgroup, spv::MemorySemanticsMask::WorkgroupMemory |
                                                           spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpWorkgroupMemoryBarrierWithGroupSync:
        builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Workgroup,
                                        spv::MemorySemanticsMask::WorkgroupMemory |
                                        spv::MemorySemanticsMask::AcquireRelease);
        return 0;
    case glslang::EOpSubgroupBarrier:
        builder.createControlBarrier(spv::Scope::Subgroup, spv::Scope::Subgroup, spv::MemorySemanticsAllMemory |
                                                                                 spv::MemorySemanticsMask::AcquireRelease);
        return spv::NoResult;
    case glslang::EOpSubgroupMemoryBarrier:
        builder.createMemoryBarrier(spv::Scope::Subgroup, spv::MemorySemanticsAllMemory |
                                                          spv::MemorySemanticsMask::AcquireRelease);
        return spv::NoResult;
    case glslang::EOpSubgroupMemoryBarrierBuffer:
        builder.createMemoryBarrier(spv::Scope::Subgroup, spv::MemorySemanticsMask::UniformMemory |
                                                          spv::MemorySemanticsMask::AcquireRelease);
        return spv::NoResult;
    case glslang::EOpSubgroupMemoryBarrierImage:
        builder.createMemoryBarrier(spv::Scope::Subgroup, spv::MemorySemanticsMask::ImageMemory |
                                                          spv::MemorySemanticsMask::AcquireRelease);
        return spv::NoResult;
    case glslang::EOpSubgroupMemoryBarrierShared:
        builder.createMemoryBarrier(spv::Scope::Subgroup, spv::MemorySemanticsMask::WorkgroupMemory |
                                                          spv::MemorySemanticsMask::AcquireRelease);
        return spv::NoResult;

    case glslang::EOpEmitVertex:
        builder.createNoResultOp(spv::Op::OpEmitVertex);
        return 0;
    case glslang::EOpEndPrimitive:
        builder.createNoResultOp(spv::Op::OpEndPrimitive);
        return 0;

    case glslang::EOpSubgroupElect: {
        std::vector<spv::Id> operands;
        return createSubgroupOperation(op, typeId, operands, glslang::EbtVoid);
    }
    case glslang::EOpTime:
    {
        std::vector<spv::Id> args; // Dummy arguments
        spv::Id id = builder.createBuiltinCall(typeId, getExtBuiltins(spv::E_SPV_AMD_gcn_shader), spv::TimeAMD, args);
        return builder.setPrecision(id, precision);
    }
    case glslang::EOpIgnoreIntersectionNV:
        builder.createNoResultOp(spv::Op::OpIgnoreIntersectionNV);
        return 0;
    case glslang::EOpTerminateRayNV:
        builder.createNoResultOp(spv::Op::OpTerminateRayNV);
        return 0;
    case glslang::EOpRayQueryInitialize:
        builder.createNoResultOp(spv::Op::OpRayQueryInitializeKHR);
        return 0;
    case glslang::EOpRayQueryTerminate:
        builder.createNoResultOp(spv::Op::OpRayQueryTerminateKHR);
        return 0;
    case glslang::EOpRayQueryGenerateIntersection:
        builder.createNoResultOp(spv::Op::OpRayQueryGenerateIntersectionKHR);
        return 0;
    case glslang::EOpRayQueryConfirmIntersection:
        builder.createNoResultOp(spv::Op::OpRayQueryConfirmIntersectionKHR);
        return 0;
    case glslang::EOpBeginInvocationInterlock:
        builder.createNoResultOp(spv::Op::OpBeginInvocationInterlockEXT);
        return 0;
    case glslang::EOpEndInvocationInterlock:
        builder.createNoResultOp(spv::Op::OpEndInvocationInterlockEXT);
        return 0;

    case glslang::EOpIsHelperInvocation:
    {
        std::vector<spv::Id> args; // Dummy arguments
        builder.addExtension(spv::E_SPV_EXT_demote_to_helper_invocation);
        builder.addCapability(spv::Capability::DemoteToHelperInvocationEXT);
        return builder.createOp(spv::Op::OpIsHelperInvocationEXT, typeId, args);
    }

    case glslang::EOpReadClockSubgroupKHR: {
        std::vector<spv::Id> args;
        args.push_back(builder.makeUintConstant(spv::Scope::Subgroup));
        builder.addExtension(spv::E_SPV_KHR_shader_clock);
        builder.addCapability(spv::Capability::ShaderClockKHR);
        return builder.createOp(spv::Op::OpReadClockKHR, typeId, args);
    }

    case glslang::EOpReadClockDeviceKHR: {
        std::vector<spv::Id> args;
        args.push_back(builder.makeUintConstant(spv::Scope::Device));
        builder.addExtension(spv::E_SPV_KHR_shader_clock);
        builder.addCapability(spv::Capability::ShaderClockKHR);
        return builder.createOp(spv::Op::OpReadClockKHR, typeId, args);
    }
    case glslang::EOpStencilAttachmentReadEXT:
    case glslang::EOpDepthAttachmentReadEXT:
    {
        builder.addExtension(spv::E_SPV_EXT_shader_tile_image);

        spv::Decoration precision;
        spv::Op spv_op;
        if (op == glslang::EOpStencilAttachmentReadEXT)
        {
            precision = spv::Decoration::RelaxedPrecision;
            spv_op = spv::Op::OpStencilAttachmentReadEXT;
            builder.addCapability(spv::Capability::TileImageStencilReadAccessEXT);
        }
        else
        {
            precision = spv::NoPrecision;
            spv_op = spv::Op::OpDepthAttachmentReadEXT;
            builder.addCapability(spv::Capability::TileImageDepthReadAccessEXT);
        }

        std::vector<spv::Id> args; // Dummy args
        spv::Id result = builder.createOp(spv_op, typeId, args);
        return builder.setPrecision(result, precision);
    }
    default:
        break;
    }

    logger->missingFunctionality("unknown operation with no arguments");

    return 0;
}

spv::Id TGlslangToSpvTraverser::getSymbolId(const glslang::TIntermSymbol* symbol)
{
    auto iter = symbolValues.find(symbol->getId());
    spv::Id id;
    if (symbolValues.end() != iter) {
        id = iter->second;
        return id;
    }

    // it was not found, create it
    spv::BuiltIn builtIn = TranslateBuiltInDecoration(symbol->getQualifier().builtIn, false);
    auto forcedType = getForcedType(symbol->getQualifier().builtIn, symbol->getType());

    // There are pairs of symbols that map to the same SPIR-V built-in:
    // gl_ObjectToWorldEXT and gl_ObjectToWorld3x4EXT, and gl_WorldToObjectEXT
    // and gl_WorldToObject3x4EXT. SPIR-V forbids having two OpVariables
    // with the same BuiltIn in the same storage class, so we must re-use one.
    const bool mayNeedToReuseBuiltIn =
        builtIn == spv::BuiltIn::ObjectToWorldKHR ||
        builtIn == spv::BuiltIn::WorldToObjectKHR;

    if (mayNeedToReuseBuiltIn) {
        auto iter = builtInVariableIds.find(uint32_t(builtIn));
        if (builtInVariableIds.end() != iter) {
            id = iter->second;
            symbolValues[symbol->getId()] = id;
            if (forcedType.second != spv::NoType)
                forceType[id] = forcedType.second;
            return id;
        }
    }

    if (symbol->getBasicType() == glslang::EbtFunction) {
        return 0;
    }

    id = createSpvVariable(symbol, forcedType.first);

    if (mayNeedToReuseBuiltIn) {
        builtInVariableIds.insert({uint32_t(builtIn), id});
    }

    symbolValues[symbol->getId()] = id;
    if (forcedType.second != spv::NoType)
        forceType[id] = forcedType.second;

    if (symbol->getBasicType() != glslang::EbtBlock) {
        builder.addDecoration(id, TranslatePrecisionDecoration(symbol->getType()));
        builder.addDecoration(id, TranslateInterpolationDecoration(symbol->getType().getQualifier()));
        builder.addDecoration(id, TranslateAuxiliaryStorageDecoration(symbol->getType().getQualifier()));
        addMeshNVDecoration(id, /*member*/ -1, symbol->getType().getQualifier());
        if (symbol->getQualifier().hasComponent())
            builder.addDecoration(id, spv::Decoration::Component, symbol->getQualifier().layoutComponent);
        if (symbol->getQualifier().hasIndex())
            builder.addDecoration(id, spv::Decoration::Index, symbol->getQualifier().layoutIndex);
        if (symbol->getType().getQualifier().hasSpecConstantId())
            builder.addDecoration(id, spv::Decoration::SpecId, symbol->getType().getQualifier().layoutSpecConstantId);
        // atomic counters use this:
        if (symbol->getQualifier().hasOffset())
            builder.addDecoration(id, spv::Decoration::Offset, symbol->getQualifier().layoutOffset);
    }

    if (symbol->getQualifier().hasLocation()) {
        if (!(glslangIntermediate->isRayTracingStage() &&
              (glslangIntermediate->IsRequestedExtension(glslang::E_GL_EXT_ray_tracing) ||
               glslangIntermediate->IsRequestedExtension(glslang::E_GL_NV_shader_invocation_reorder) ||
               glslangIntermediate->IsRequestedExtension(glslang::E_GL_EXT_shader_invocation_reorder))
              && (builder.getStorageClass(id) == spv::StorageClass::RayPayloadKHR ||
                  builder.getStorageClass(id) == spv::StorageClass::IncomingRayPayloadKHR ||
                  builder.getStorageClass(id) == spv::StorageClass::CallableDataKHR ||
                  builder.getStorageClass(id) == spv::StorageClass::IncomingCallableDataKHR ||
                  builder.getStorageClass(id) == spv::StorageClass::HitObjectAttributeEXT ||
                  builder.getStorageClass(id) == spv::StorageClass::HitObjectAttributeNV))) {
            // Location values are used to link TraceRayKHR/ExecuteCallableKHR/HitObjectGetAttributesNV
            // to corresponding variables but are not valid in SPIRV since they are supported only
            // for Input/Output Storage classes.
            builder.addDecoration(id, spv::Decoration::Location, symbol->getQualifier().layoutLocation);
        }
    }

    builder.addDecoration(id, TranslateInvariantDecoration(symbol->getType().getQualifier()));
    if (symbol->getQualifier().hasStream() && glslangIntermediate->isMultiStream()) {
        builder.addCapability(spv::Capability::GeometryStreams);
        builder.addDecoration(id, spv::Decoration::Stream, symbol->getQualifier().layoutStream);
    }
    if (symbol->getQualifier().hasSet())
        builder.addDecoration(id, spv::Decoration::DescriptorSet, symbol->getQualifier().layoutSet);
    else if (IsDescriptorResource(symbol->getType())) {
        // default to 0
        builder.addDecoration(id, spv::Decoration::DescriptorSet, 0);
    }
    if (symbol->getQualifier().hasBinding())
        builder.addDecoration(id, spv::Decoration::Binding, symbol->getQualifier().layoutBinding);
    else if (IsDescriptorResource(symbol->getType())) {
        // default to 0
        builder.addDecoration(id, spv::Decoration::Binding, 0);
    }
    if (symbol->getQualifier().hasAttachment())
        builder.addDecoration(id, spv::Decoration::InputAttachmentIndex, symbol->getQualifier().layoutAttachment);
    if (glslangIntermediate->getXfbMode()) {
        builder.addCapability(spv::Capability::TransformFeedback);
        if (symbol->getQualifier().hasXfbBuffer()) {
            builder.addDecoration(id, spv::Decoration::XfbBuffer, symbol->getQualifier().layoutXfbBuffer);
            unsigned stride = glslangIntermediate->getXfbStride(symbol->getQualifier().layoutXfbBuffer);
            if (stride != glslang::TQualifier::layoutXfbStrideEnd)
                builder.addDecoration(id, spv::Decoration::XfbStride, stride);
        }
        if (symbol->getQualifier().hasXfbOffset())
            builder.addDecoration(id, spv::Decoration::Offset, symbol->getQualifier().layoutXfbOffset);
    }

    // add built-in variable decoration
    if (builtIn != spv::BuiltIn::Max) {
        // WorkgroupSize deprecated in spirv1.6
        if (glslangIntermediate->getSpv().spv < glslang::EShTargetSpv_1_6 ||
            builtIn != spv::BuiltIn::WorkgroupSize)
            builder.addDecoration(id, spv::Decoration::BuiltIn, (int)builtIn);
    }

    // Add volatile decoration to HelperInvocation for spirv1.6 and beyond
    if (builtIn == spv::BuiltIn::HelperInvocation &&
        !glslangIntermediate->usingVulkanMemoryModel() &&
        glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_6) {
        builder.addDecoration(id, spv::Decoration::Volatile);
    }

    // Subgroup builtins which have input storage class are volatile for ray tracing stages.
    if (symbol->getType().isImage() || symbol->getQualifier().isPipeInput()) {
        std::vector<spv::Decoration> memory;
        TranslateMemoryDecoration(symbol->getType().getQualifier(), memory,
            glslangIntermediate->usingVulkanMemoryModel());
        for (unsigned int i = 0; i < memory.size(); ++i)
            builder.addDecoration(id, memory[i]);
    }

    if (builtIn == spv::BuiltIn::SampleMask) {
          spv::Decoration decoration;
          // GL_NV_sample_mask_override_coverage extension
          if (glslangIntermediate->getLayoutOverrideCoverage())
              decoration = spv::Decoration::OverrideCoverageNV;
          else
              decoration = spv::Decoration::Max;
        builder.addDecoration(id, decoration);
        if (decoration != spv::Decoration::Max) {
            builder.addCapability(spv::Capability::SampleMaskOverrideCoverageNV);
            builder.addExtension(spv::E_SPV_NV_sample_mask_override_coverage);
        }
    }
    else if (builtIn == spv::BuiltIn::Layer) {
        // SPV_NV_viewport_array2 extension
        if (symbol->getQualifier().layoutViewportRelative) {
            builder.addDecoration(id, spv::Decoration::ViewportRelativeNV);
            builder.addCapability(spv::Capability::ShaderViewportMaskNV);
            builder.addExtension(spv::E_SPV_NV_viewport_array2);
        }
        if (symbol->getQualifier().layoutSecondaryViewportRelativeOffset != -2048) {
            builder.addDecoration(id, spv::Decoration::SecondaryViewportRelativeNV,
                                  symbol->getQualifier().layoutSecondaryViewportRelativeOffset);
            builder.addCapability(spv::Capability::ShaderStereoViewNV);
            builder.addExtension(spv::E_SPV_NV_stereo_view_rendering);
        }
    }

    if (symbol->getQualifier().layoutPassthrough) {
        builder.addDecoration(id, spv::Decoration::PassthroughNV);
        builder.addCapability(spv::Capability::GeometryShaderPassthroughNV);
        builder.addExtension(spv::E_SPV_NV_geometry_shader_passthrough);
    }
    if (symbol->getQualifier().pervertexNV) {
        builder.addDecoration(id, spv::Decoration::PerVertexNV);
        builder.addCapability(spv::Capability::FragmentBarycentricNV);
        builder.addExtension(spv::E_SPV_NV_fragment_shader_barycentric);
    }

    if (symbol->getQualifier().pervertexEXT) {
        builder.addDecoration(id, spv::Decoration::PerVertexKHR);
        builder.addCapability(spv::Capability::FragmentBarycentricKHR);
        builder.addExtension(spv::E_SPV_KHR_fragment_shader_barycentric);
    }

    if (glslangIntermediate->getHlslFunctionality1() && symbol->getType().getQualifier().semanticName != nullptr) {
        builder.addExtension("SPV_GOOGLE_hlsl_functionality1");
        builder.addDecoration(id, spv::Decoration::HlslSemanticGOOGLE,
                              symbol->getType().getQualifier().semanticName);
    }

    if (symbol->isReference()) {
        builder.addDecoration(id, symbol->getType().getQualifier().restrict ?
            spv::Decoration::RestrictPointerEXT : spv::Decoration::AliasedPointerEXT);
    }

    // Add SPIR-V decorations (GL_EXT_spirv_intrinsics)
    if (symbol->getType().getQualifier().hasSpirvDecorate())
        applySpirvDecorate(symbol->getType(), id, {});

    return id;
}

// add per-primitive, per-view. per-task decorations to a struct member (member >= 0) or an object
void TGlslangToSpvTraverser::addMeshNVDecoration(spv::Id id, int member, const glslang::TQualifier& qualifier)
{
    bool isMeshShaderExt = (glslangIntermediate->getRequestedExtensions().find(glslang::E_GL_EXT_mesh_shader) !=
                            glslangIntermediate->getRequestedExtensions().end());

    if (member >= 0) {
        if (qualifier.perPrimitiveNV) {
            // Need to add capability/extension for fragment shader.
            // Mesh shader already adds this by default.
            if (glslangIntermediate->getStage() == EShLangFragment) {
                if(isMeshShaderExt) {
                    builder.addCapability(spv::Capability::MeshShadingEXT);
                    builder.addExtension(spv::E_SPV_EXT_mesh_shader);
                } else {
                    builder.addCapability(spv::Capability::MeshShadingNV);
                    builder.addExtension(spv::E_SPV_NV_mesh_shader);
                }
            }
            builder.addMemberDecoration(id, (unsigned)member, spv::Decoration::PerPrimitiveNV);
        }
        if (qualifier.perViewNV)
            builder.addMemberDecoration(id, (unsigned)member, spv::Decoration::PerViewNV);
        if (qualifier.perTaskNV)
            builder.addMemberDecoration(id, (unsigned)member, spv::Decoration::PerTaskNV);
    } else {
        if (qualifier.perPrimitiveNV) {
            // Need to add capability/extension for fragment shader.
            // Mesh shader already adds this by default.
            if (glslangIntermediate->getStage() == EShLangFragment) {
                if(isMeshShaderExt) {
                    builder.addCapability(spv::Capability::MeshShadingEXT);
                    builder.addExtension(spv::E_SPV_EXT_mesh_shader);
                } else {
                    builder.addCapability(spv::Capability::MeshShadingNV);
                    builder.addExtension(spv::E_SPV_NV_mesh_shader);
                }
            }
            builder.addDecoration(id, spv::Decoration::PerPrimitiveNV);
        }
        if (qualifier.perViewNV)
            builder.addDecoration(id, spv::Decoration::PerViewNV);
        if (qualifier.perTaskNV)
            builder.addDecoration(id, spv::Decoration::PerTaskNV);
    }
}

bool TGlslangToSpvTraverser::hasQCOMImageProceessingDecoration(spv::Id id, spv::Decoration decor)
{
  std::vector<spv::Decoration> &decoVec = idToQCOMDecorations[id];
  for ( auto d : decoVec ) {
    if ( d == decor )
      return true;
  }
  return false;
}

void TGlslangToSpvTraverser::addImageProcessingQCOMDecoration(spv::Id id, spv::Decoration decor)
{
  spv::Op opc = builder.getOpCode(id);
  if (opc == spv::Op::OpSampledImage) {
    id  = builder.getIdOperand(id, 0);
    opc = builder.getOpCode(id);
  }

  if (opc == spv::Op::OpLoad) {
    spv::Id texid = builder.getIdOperand(id, 0);
    if (!hasQCOMImageProceessingDecoration(texid, decor)) {//
      builder.addDecoration(texid, decor);
      idToQCOMDecorations[texid].push_back(decor);
    }
  }
}

void TGlslangToSpvTraverser::addImageProcessing2QCOMDecoration(spv::Id id, bool isForGather)
{
  if (isForGather) {
    return addImageProcessingQCOMDecoration(id, spv::Decoration::BlockMatchTextureQCOM);
  }

  auto addDecor =
    [this](spv::Id id, spv::Decoration decor) {
      spv::Op tsopc = this->builder.getOpCode(id);
      if (tsopc == spv::Op::OpLoad) {
        spv::Id tsid = this->builder.getIdOperand(id, 0);
        if (this->glslangIntermediate->getSpv().spv >= glslang::EShTargetSpv_1_4) {
          assert(iOSet.count(tsid) > 0);
        }
        if (!hasQCOMImageProceessingDecoration(tsid, decor)) {
          this->builder.addDecoration(tsid, decor);
          idToQCOMDecorations[tsid].push_back(decor);
        }
      }
    };

  spv::Op opc = builder.getOpCode(id);
  bool isInterfaceObject = (opc != spv::Op::OpSampledImage);

  if (!isInterfaceObject) {
    addDecor(builder.getIdOperand(id, 0), spv::Decoration::BlockMatchTextureQCOM);
    addDecor(builder.getIdOperand(id, 1), spv::Decoration::BlockMatchSamplerQCOM);
  } else {
    addDecor(id, spv::Decoration::BlockMatchTextureQCOM);
    addDecor(id, spv::Decoration::BlockMatchSamplerQCOM);
  }
}

// Make a full tree of instructions to build a SPIR-V specialization constant,
// or regular constant if possible.
//
// TBD: this is not yet done, nor verified to be the best design, it does do the leaf symbols though
//
// Recursively walk the nodes.  The nodes form a tree whose leaves are
// regular constants, which themselves are trees that createSpvConstant()
// recursively walks.  So, this function walks the "top" of the tree:
//  - emit specialization constant-building instructions for specConstant
//  - when running into a non-spec-constant, switch to createSpvConstant()
spv::Id TGlslangToSpvTraverser::createSpvConstant(const glslang::TIntermTyped& node)
{
    assert(node.getQualifier().isConstant());

    // Handle front-end constants first (non-specialization constants).
    if (! node.getQualifier().specConstant) {
        // hand off to the non-spec-constant path
        assert(node.getAsConstantUnion() != nullptr || node.getAsSymbolNode() != nullptr);
        int nextConst = 0;
        return createSpvConstantFromConstUnionArray(node.getType(), node.getAsConstantUnion() ?
            node.getAsConstantUnion()->getConstArray() : node.getAsSymbolNode()->getConstArray(),
            nextConst, false);
    }

    // We now know we have a specialization constant to build

    // Extra capabilities may be needed.
    if (node.getType().contains8BitInt())
        builder.addCapability(spv::Capability::Int8);
    if (node.getType().contains16BitFloat())
        builder.addCapability(spv::Capability::Float16);
    if (node.getType().contains16BitInt())
        builder.addCapability(spv::Capability::Int16);
    if (node.getType().contains64BitInt())
        builder.addCapability(spv::Capability::Int64);
    if (node.getType().containsDouble())
        builder.addCapability(spv::Capability::Float64);

    // gl_WorkGroupSize is a special case until the front-end handles hierarchical specialization constants,
    // even then, it's specialization ids are handled by special case syntax in GLSL: layout(local_size_x = ...
    if (node.getType().getQualifier().builtIn == glslang::EbvWorkGroupSize) {
        std::vector<spv::Id> dimConstId;
        for (int dim = 0; dim < 3; ++dim) {
            bool specConst = (glslangIntermediate->getLocalSizeSpecId(dim) != glslang::TQualifier::layoutNotSet);
            dimConstId.push_back(builder.makeUintConstant(glslangIntermediate->getLocalSize(dim), specConst));
            if (specConst) {
                builder.addDecoration(dimConstId.back(), spv::Decoration::SpecId,
                                      glslangIntermediate->getLocalSizeSpecId(dim));
            }
        }
        return builder.makeCompositeConstant(builder.makeVectorType(builder.makeUintType(32), 3), dimConstId, true);
    }

    // An AST node labelled as specialization constant should be a symbol node.
    // Its initializer should either be a sub tree with constant nodes, or a constant union array.
    if (auto* sn = node.getAsSymbolNode()) {
        spv::Id result;
        if (auto* sub_tree = sn->getConstSubtree()) {
            // Traverse the constant constructor sub tree like generating normal run-time instructions.
            // During the AST traversal, if the node is marked as 'specConstant', SpecConstantOpModeGuard
            // will set the builder into spec constant op instruction generating mode.
            sub_tree->traverse(this);
            result = accessChainLoad(sub_tree->getType());
        } else if (auto* const_union_array = &sn->getConstArray()) {
            int nextConst = 0;
            result = createSpvConstantFromConstUnionArray(sn->getType(), *const_union_array, nextConst, true);
        } else {
            logger->missingFunctionality("Invalid initializer for spec constant.");
            return spv::NoResult;
        }
        builder.addName(result, sn->getName().c_str());
        return result;
    }

    // Neither a front-end constant node, nor a specialization constant node with constant union array or
    // constant sub tree as initializer.
    logger->missingFunctionality("Neither a front-end constant nor a spec constant.");
    return spv::NoResult;
}

// Use 'consts' as the flattened glslang source of scalar constants to recursively
// build the aggregate SPIR-V constant.
//
// If there are not enough elements present in 'consts', 0 will be substituted;
// an empty 'consts' can be used to create a fully zeroed SPIR-V constant.
//
spv::Id TGlslangToSpvTraverser::createSpvConstantFromConstUnionArray(const glslang::TType& glslangType,
    const glslang::TConstUnionArray& consts, int& nextConst, bool specConstant)
{
    // vector of constants for SPIR-V
    std::vector<spv::Id> spvConsts;

    // Type is used for struct and array constants
    spv::Id typeId = convertGlslangToSpvType(glslangType);

    if (glslangType.isArray()) {
        glslang::TType elementType(glslangType, 0);
        for (int i = 0; i < glslangType.getOuterArraySize(); ++i)
            spvConsts.push_back(createSpvConstantFromConstUnionArray(elementType, consts, nextConst, false));
    } else if (glslangType.isMatrix()) {
        glslang::TType vectorType(glslangType, 0);
        for (int col = 0; col < glslangType.getMatrixCols(); ++col)
            spvConsts.push_back(createSpvConstantFromConstUnionArray(vectorType, consts, nextConst, false));
    } else if (glslangType.isCoopMat()) {
        glslang::TType componentType(glslangType.getBasicType());
        spvConsts.push_back(createSpvConstantFromConstUnionArray(componentType, consts, nextConst, false));
    } else if (glslangType.isStruct()) {
        glslang::TVector<glslang::TTypeLoc>::const_iterator iter;
        for (iter = glslangType.getStruct()->begin(); iter != glslangType.getStruct()->end(); ++iter)
            spvConsts.push_back(createSpvConstantFromConstUnionArray(*iter->type, consts, nextConst, false));
    } else if (glslangType.getVectorSize() > 1 || glslangType.isCoopVecNV()) {
        unsigned int numComponents = glslangType.isCoopVecNV() ? glslangType.getTypeParameters()->arraySizes->getDimSize(0) : glslangType.getVectorSize();
        for (unsigned int i = 0; i < numComponents; ++i) {
            bool zero = nextConst >= consts.size();
            switch (glslangType.getBasicType()) {
            case glslang::EbtInt:
                spvConsts.push_back(builder.makeIntConstant(zero ? 0 : consts[nextConst].getIConst()));
                break;
            case glslang::EbtUint:
                spvConsts.push_back(builder.makeUintConstant(zero ? 0 : consts[nextConst].getUConst()));
                break;
            case glslang::EbtFloat:
                spvConsts.push_back(builder.makeFloatConstant(zero ? 0.0F : (float)consts[nextConst].getDConst()));
                break;
            case glslang::EbtBool:
                spvConsts.push_back(builder.makeBoolConstant(zero ? false : consts[nextConst].getBConst()));
                break;
            case glslang::EbtInt8:
                builder.addCapability(spv::Capability::Int8);
                spvConsts.push_back(builder.makeInt8Constant(zero ? 0 : consts[nextConst].getI8Const()));
                break;
            case glslang::EbtUint8:
                builder.addCapability(spv::Capability::Int8);
                spvConsts.push_back(builder.makeUint8Constant(zero ? 0 : consts[nextConst].getU8Const()));
                break;
            case glslang::EbtInt16:
                builder.addCapability(spv::Capability::Int16);
                spvConsts.push_back(builder.makeInt16Constant(zero ? 0 : consts[nextConst].getI16Const()));
                break;
            case glslang::EbtUint16:
                builder.addCapability(spv::Capability::Int16);
                spvConsts.push_back(builder.makeUint16Constant(zero ? 0 : consts[nextConst].getU16Const()));
                break;
            case glslang::EbtInt64:
                spvConsts.push_back(builder.makeInt64Constant(zero ? 0 : consts[nextConst].getI64Const()));
                break;
            case glslang::EbtUint64:
                spvConsts.push_back(builder.makeUint64Constant(zero ? 0 : consts[nextConst].getU64Const()));
                break;
            case glslang::EbtDouble:
                spvConsts.push_back(builder.makeDoubleConstant(zero ? 0.0 : consts[nextConst].getDConst()));
                break;
            case glslang::EbtFloat16:
                builder.addCapability(spv::Capability::Float16);
                spvConsts.push_back(builder.makeFloat16Constant(zero ? 0.0F : (float)consts[nextConst].getDConst()));
                break;
            case glslang::EbtBFloat16:
                spvConsts.push_back(builder.makeBFloat16Constant(zero ? 0.0F : (float)consts[nextConst].getDConst()));
                break;
            case glslang::EbtFloatE5M2:
                spvConsts.push_back(builder.makeFloatE5M2Constant(zero ? 0.0F : (float)consts[nextConst].getDConst()));
                break;
            case glslang::EbtFloatE4M3:
                spvConsts.push_back(builder.makeFloatE4M3Constant(zero ? 0.0F : (float)consts[nextConst].getDConst()));
                break;
            default:
                assert(0);
                break;
            }
            ++nextConst;
        }
    } else {
        // we have a non-aggregate (scalar) constant
        bool zero = nextConst >= consts.size();
        spv::Id scalar = 0;
        switch (glslangType.getBasicType()) {
        case glslang::EbtInt:
            scalar = builder.makeIntConstant(zero ? 0 : consts[nextConst].getIConst(), specConstant);
            break;
        case glslang::EbtUint:
            scalar = builder.makeUintConstant(zero ? 0 : consts[nextConst].getUConst(), specConstant);
            break;
        case glslang::EbtFloat:
            scalar = builder.makeFloatConstant(zero ? 0.0F : (float)consts[nextConst].getDConst(), specConstant);
            break;
        case glslang::EbtBool:
            scalar = builder.makeBoolConstant(zero ? false : consts[nextConst].getBConst(), specConstant);
            break;
        case glslang::EbtInt8:
            builder.addCapability(spv::Capability::Int8);
            scalar = builder.makeInt8Constant(zero ? 0 : consts[nextConst].getI8Const(), specConstant);
            break;
        case glslang::EbtUint8:
            builder.addCapability(spv::Capability::Int8);
            scalar = builder.makeUint8Constant(zero ? 0 : consts[nextConst].getU8Const(), specConstant);
            break;
        case glslang::EbtInt16:
            builder.addCapability(spv::Capability::Int16);
            scalar = builder.makeInt16Constant(zero ? 0 : consts[nextConst].getI16Const(), specConstant);
            break;
        case glslang::EbtUint16:
            builder.addCapability(spv::Capability::Int16);
            scalar = builder.makeUint16Constant(zero ? 0 : consts[nextConst].getU16Const(), specConstant);
            break;
        case glslang::EbtInt64:
            scalar = builder.makeInt64Constant(zero ? 0 : consts[nextConst].getI64Const(), specConstant);
            break;
        case glslang::EbtUint64:
            scalar = builder.makeUint64Constant(zero ? 0 : consts[nextConst].getU64Const(), specConstant);
            break;
        case glslang::EbtDouble:
            scalar = builder.makeDoubleConstant(zero ? 0.0 : consts[nextConst].getDConst(), specConstant);
            break;
        case glslang::EbtFloat16:
            builder.addCapability(spv::Capability::Float16);
            scalar = builder.makeFloat16Constant(zero ? 0.0F : (float)consts[nextConst].getDConst(), specConstant);
            break;
        case glslang::EbtBFloat16:
            scalar = builder.makeBFloat16Constant(zero ? 0.0F : (float)consts[nextConst].getDConst(), specConstant);
            break;
        case glslang::EbtFloatE5M2:
            scalar = builder.makeFloatE5M2Constant(zero ? 0.0F : (float)consts[nextConst].getDConst(), specConstant);
            break;
        case glslang::EbtFloatE4M3:
            scalar = builder.makeFloatE4M3Constant(zero ? 0.0F : (float)consts[nextConst].getDConst(), specConstant);
            break;
        case glslang::EbtReference:
            scalar = builder.makeUint64Constant(zero ? 0 : consts[nextConst].getU64Const(), specConstant);
            scalar = builder.createUnaryOp(spv::Op::OpBitcast, typeId, scalar);
            break;
        case glslang::EbtString:
            scalar = builder.getStringId(consts[nextConst].getSConst()->c_str());
            break;
        default:
            assert(0);
            break;
        }
        ++nextConst;
        return scalar;
    }

    return builder.makeCompositeConstant(typeId, spvConsts);
}

// Return true if the node is a constant or symbol whose reading has no
// non-trivial observable cost or effect.
bool TGlslangToSpvTraverser::isTrivialLeaf(const glslang::TIntermTyped* node)
{
    // don't know what this is
    if (node == nullptr)
        return false;

    // a constant is safe
    if (node->getAsConstantUnion() != nullptr)
        return true;

    // not a symbol means non-trivial
    if (node->getAsSymbolNode() == nullptr)
        return false;

    // a symbol, depends on what's being read
    switch (node->getType().getQualifier().storage) {
    case glslang::EvqTemporary:
    case glslang::EvqGlobal:
    case glslang::EvqIn:
    case glslang::EvqInOut:
    case glslang::EvqConst:
    case glslang::EvqConstReadOnly:
    case glslang::EvqUniform:
        return true;
    default:
        return false;
    }
}

// A node is trivial if it is a single operation with no side effects.
// HLSL (and/or vectors) are always trivial, as it does not short circuit.
// Otherwise, error on the side of saying non-trivial.
// Return true if trivial.
bool TGlslangToSpvTraverser::isTrivial(const glslang::TIntermTyped* node)
{
    if (node == nullptr)
        return false;

    // count non scalars as trivial, as well as anything coming from HLSL
    if (! node->getType().isScalarOrVec1() || glslangIntermediate->getSource() == glslang::EShSourceHlsl)
        return true;

    // symbols and constants are trivial
    if (isTrivialLeaf(node))
        return true;

    // otherwise, it needs to be a simple operation or one or two leaf nodes

    // not a simple operation
    const glslang::TIntermBinary* binaryNode = node->getAsBinaryNode();
    const glslang::TIntermUnary* unaryNode = node->getAsUnaryNode();
    if (binaryNode == nullptr && unaryNode == nullptr)
        return false;

    // not on leaf nodes
    if (binaryNode && (! isTrivialLeaf(binaryNode->getLeft()) || ! isTrivialLeaf(binaryNode->getRight())))
        return false;

    if (unaryNode && ! isTrivialLeaf(unaryNode->getOperand())) {
        return false;
    }

    if (IsOpNumericConv(node->getAsOperator()->getOp()) &&
        node->getType().getBasicType() == glslang::EbtBool) {
        return true;
    }

    switch (node->getAsOperator()->getOp()) {
    case glslang::EOpLogicalNot:
    case glslang::EOpEqual:
    case glslang::EOpNotEqual:
    case glslang::EOpLessThan:
    case glslang::EOpGreaterThan:
    case glslang::EOpLessThanEqual:
    case glslang::EOpGreaterThanEqual:
    case glslang::EOpIndexDirect:
    case glslang::EOpIndexDirectStruct:
    case glslang::EOpLogicalXor:
    case glslang::EOpAny:
    case glslang::EOpAll:
        return true;
    default:
        return false;
    }
}

// Emit short-circuiting code, where 'right' is never evaluated unless
// the left side is true (for &&) or false (for ||).
spv::Id TGlslangToSpvTraverser::createShortCircuit(glslang::TOperator op, glslang::TIntermTyped& left,
    glslang::TIntermTyped& right)
{
    spv::Id boolTypeId = builder.makeBoolType();

    // emit left operand
    builder.clearAccessChain();
    left.traverse(this);
    spv::Id leftId = accessChainLoad(left.getType());

    // Operands to accumulate OpPhi operands
    std::vector<spv::Id> phiOperands;
    phiOperands.reserve(4);
    // accumulate left operand's phi information
    phiOperands.push_back(leftId);
    phiOperands.push_back(builder.getBuildPoint()->getId());

    // Make the two kinds of operation symmetric with a "!"
    //   || => emit "if (! left) result = right"
    //   && => emit "if (  left) result = right"
    //
    // TODO: this runtime "not" for || could be avoided by adding functionality
    // to 'builder' to have an "else" without an "then"
    if (op == glslang::EOpLogicalOr)
        leftId = builder.createUnaryOp(spv::Op::OpLogicalNot, boolTypeId, leftId);

    // make an "if" based on the left value
    spv::Builder::If ifBuilder(leftId, spv::SelectionControlMask::MaskNone, builder);

    // emit right operand as the "then" part of the "if"
    builder.clearAccessChain();
    right.traverse(this);
    spv::Id rightId = accessChainLoad(right.getType());

    // accumulate left operand's phi information
    phiOperands.push_back(rightId);
    phiOperands.push_back(builder.getBuildPoint()->getId());

    // finish the "if"
    ifBuilder.makeEndIf();

    // phi together the two results
    return builder.createOp(spv::Op::OpPhi, boolTypeId, phiOperands);
}

// Return type Id of the imported set of extended instructions corresponds to the name.
// Import this set if it has not been imported yet.
spv::Id TGlslangToSpvTraverser::getExtBuiltins(const char* name)
{
    if (extBuiltinMap.find(name) != extBuiltinMap.end())
        return extBuiltinMap[name];
    else {
        spv::Id extBuiltins = builder.import(name);
        extBuiltinMap[name] = extBuiltins;
        return extBuiltins;
    }
}

} // end anonymous namespace

namespace glslang {

void GetSpirvVersion(std::string& version)
{
    const int bufSize = 100;
    char buf[bufSize];
    snprintf(buf, bufSize, "0x%08x, Revision %d", spv::Version, spv::Revision);
    version = buf;
}

// For low-order part of the generator's magic number. Bump up
// when there is a change in the style (e.g., if SSA form changes,
// or a different instruction sequence to do something gets used).
int GetSpirvGeneratorVersion()
{
    // return 1; // start
    // return 2; // EOpAtomicCounterDecrement gets a post decrement, to map between GLSL -> SPIR-V
    // return 3; // change/correct barrier-instruction operands, to match memory model group decisions
    // return 4; // some deeper access chains: for dynamic vector component, and local Boolean component
    // return 5; // make OpArrayLength result type be an int with signedness of 0
    // return 6; // revert version 5 change, which makes a different (new) kind of incorrect code,
                 // versions 4 and 6 each generate OpArrayLength as it has long been done
    // return 7; // GLSL volatile keyword maps to both SPIR-V decorations Volatile and Coherent
    // return 8; // switch to new dead block eliminator; use OpUnreachable
    // return 9; // don't include opaque function parameters in OpEntryPoint global's operand list
    // return 10; // Generate OpFUnordNotEqual for != comparisons
    return 11; // Make OpEmitMeshTasksEXT a terminal instruction
}

// Write SPIR-V out to a binary file
bool OutputSpvBin(const std::vector<unsigned int>& spirv, const char* baseName)
{
    std::ofstream out;
    out.open(baseName, std::ios::binary | std::ios::out);
    if (out.fail()) {
        printf("ERROR: Failed to open file: %s\n", baseName);
        return false;
    }
    for (int i = 0; i < (int)spirv.size(); ++i) {
        unsigned int word = spirv[i];
        out.write((const char*)&word, 4);
    }
    out.close();
    return true;
}

// Write SPIR-V out to a text file with 32-bit hexadecimal words
bool OutputSpvHex(const std::vector<unsigned int>& spirv, const char* baseName, const char* varName)
{
    std::ofstream out;
    out.open(baseName, std::ios::binary | std::ios::out);
    if (out.fail()) {
        printf("ERROR: Failed to open file: %s\n", baseName);
        return false;
    }
    out << "\t// " <<
        GetSpirvGeneratorVersion() <<
        GLSLANG_VERSION_MAJOR << "." << GLSLANG_VERSION_MINOR << "." << GLSLANG_VERSION_PATCH <<
        GLSLANG_VERSION_FLAVOR << std::endl;
    if (varName != nullptr) {
        out << "\t #pragma once" << std::endl;
        out << "const uint32_t " << varName << "[] = {" << std::endl;
    }
    const int WORDS_PER_LINE = 8;
    for (int i = 0; i < (int)spirv.size(); i += WORDS_PER_LINE) {
        out << "\t";
        for (int j = 0; j < WORDS_PER_LINE && i + j < (int)spirv.size(); ++j) {
            const unsigned int word = spirv[i + j];
            out << "0x" << std::hex << std::setw(8) << std::setfill('0') << word;
            if (i + j + 1 < (int)spirv.size()) {
                out << ",";
            }
        }
        out << std::endl;
    }
    if (varName != nullptr) {
        out << "};";
        out << std::endl;
    }
    out.close();
    return true;
}

//
// Set up the glslang traversal
//
void GlslangToSpv(const TIntermediate& intermediate, std::vector<unsigned int>& spirv, SpvOptions* options)
{
    spv::SpvBuildLogger logger;
    GlslangToSpv(intermediate, spirv, &logger, options);
}

void GlslangToSpv(const TIntermediate& intermediate, std::vector<unsigned int>& spirv,
                  spv::SpvBuildLogger* logger, SpvOptions* options)
{
    TIntermNode* root = intermediate.getTreeRoot();

    if (root == nullptr)
        return;

    SpvOptions defaultOptions;
    if (options == nullptr)
        options = &defaultOptions;

    GetThreadPoolAllocator().push();

    TGlslangToSpvTraverser it(intermediate.getSpv().spv, &intermediate, logger, *options);
    root->traverse(&it);
    it.finishSpv(options->compileOnly);
    it.dumpSpv(spirv);

#if ENABLE_OPT
    // If from HLSL, run spirv-opt to "legalize" the SPIR-V for Vulkan
    // eg. forward and remove memory writes of opaque types.
    bool prelegalization = intermediate.getSource() == EShSourceHlsl;
    if ((prelegalization || options->optimizeSize) && !options->disableOptimizer) {
        SpirvToolsTransform(intermediate, spirv, logger, options);
        prelegalization = false;
    }
    else if (options->stripDebugInfo) {
        // Strip debug info even if optimization is disabled.
        SpirvToolsStripDebugInfo(intermediate, spirv, logger);
    }

    if (options->validate)
        SpirvToolsValidate(intermediate, spirv, logger, prelegalization);

    if (options->disassemble)
        SpirvToolsDisassemble(std::cout, spirv);

#endif

    GetThreadPoolAllocator().pop();
}

} // end namespace glslang
