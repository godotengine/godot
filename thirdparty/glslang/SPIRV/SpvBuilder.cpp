//
// Copyright (C) 2014-2015 LunarG, Inc.
// Copyright (C) 2015-2018 Google, Inc.
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
// Helper for making SPIR-V IR.  Generally, this is documented in the header
// SpvBuilder.h.
//

#include <cassert>
#include <cstdlib>

#include <unordered_set>
#include <algorithm>

#include "SpvBuilder.h"
#include "spvUtil.h"
#include "hex_float.h"

#ifndef _WIN32
    #include <cstdio>
#endif

namespace spv {

Builder::Builder(unsigned int spvVersion, unsigned int magicNumber, SpvBuildLogger* buildLogger) :
    spvVersion(spvVersion),
    sourceLang(SourceLanguage::Unknown),
    sourceVersion(0),
    addressModel(AddressingModel::Logical),
    memoryModel(MemoryModel::GLSL450),
    builderNumber(magicNumber),
    buildPoint(nullptr),
    uniqueId(0),
    entryPointFunction(nullptr),
    generatingOpCodeForSpecConst(false),
    logger(buildLogger)
{
    clearAccessChain();
}

Builder::~Builder()
{
}

Id Builder::import(const char* name)
{
    Instruction* import = new Instruction(getUniqueId(), NoType, Op::OpExtInstImport);
    import->addStringOperand(name);
    module.mapInstruction(import);

    imports.push_back(std::unique_ptr<Instruction>(import));
    return import->getResultId();
}

// For creating new groupedTypes (will return old type if the requested one was already made).
Id Builder::makeVoidType()
{
    Instruction* type;
    if (groupedTypes[enumCast(Op::OpTypeVoid)].size() == 0) {
        Id typeId = getUniqueId();
        type = new Instruction(typeId, NoType, Op::OpTypeVoid);
        groupedTypes[enumCast(Op::OpTypeVoid)].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
        // Core OpTypeVoid used for debug void type
        if (emitNonSemanticShaderDebugInfo)
            debugTypeIdLookup[typeId] = typeId;
    } else
        type = groupedTypes[enumCast(Op::OpTypeVoid)].back();

    return type->getResultId();
}

Id Builder::makeBoolType()
{
    Instruction* type;
    if (groupedTypes[enumCast(Op::OpTypeBool)].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, Op::OpTypeBool);
        groupedTypes[enumCast(Op::OpTypeBool)].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);

        if (emitNonSemanticShaderDebugInfo) {
            auto const debugResultId = makeBoolDebugType(32);
            debugTypeIdLookup[type->getResultId()] = debugResultId;
        }

    } else
        type = groupedTypes[enumCast(Op::OpTypeBool)].back();


    return type->getResultId();
}

Id Builder::makeSamplerType(const char* debugName)
{
    Instruction* type;
    if (groupedTypes[enumCast(Op::OpTypeSampler)].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, Op::OpTypeSampler);
        groupedTypes[enumCast(Op::OpTypeSampler)].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
    } else
        type = groupedTypes[enumCast(Op::OpTypeSampler)].back();

    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeOpaqueDebugType(debugName);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makePointer(StorageClass storageClass, Id pointee)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypePointer)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypePointer)][t];
        if (type->getImmediateOperand(0) == (unsigned)storageClass &&
            type->getIdOperand(1) == pointee)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypePointer);
    type->reserveOperands(2);
    type->addImmediateOperand(storageClass);
    type->addIdOperand(pointee);
    groupedTypes[enumCast(Op::OpTypePointer)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    if (emitNonSemanticShaderDebugInfo) {
        const Id debugResultId = makePointerDebugType(storageClass, pointee);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makeForwardPointer(StorageClass storageClass)
{
    // Caching/uniquifying doesn't work here, because we don't know the
    // pointee type and there can be multiple forward pointers of the same
    // storage type. Somebody higher up in the stack must keep track.
    Instruction* type = new Instruction(getUniqueId(), NoType, Op::OpTypeForwardPointer);
    type->addImmediateOperand(storageClass);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    if (emitNonSemanticShaderDebugInfo) {
        const Id debugResultId = makeForwardPointerDebugType(storageClass);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }
    return type->getResultId();
}

Id Builder::makePointerFromForwardPointer(StorageClass storageClass, Id forwardPointerType, Id pointee)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypePointer)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypePointer)][t];
        if (type->getImmediateOperand(0) == (unsigned)storageClass &&
            type->getIdOperand(1) == pointee)
            return type->getResultId();
    }

    type = new Instruction(forwardPointerType, NoType, Op::OpTypePointer);
    type->reserveOperands(2);
    type->addImmediateOperand(storageClass);
    type->addIdOperand(pointee);
    groupedTypes[enumCast(Op::OpTypePointer)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    // If we are emitting nonsemantic debuginfo, we need to patch the debug pointer type
    // that was emitted alongside the forward pointer, now that we have a pointee debug
    // type for it to point to.
    if (emitNonSemanticShaderDebugInfo) {
        Instruction *debugForwardPointer = module.getInstruction(getDebugType(forwardPointerType));
        assert(getDebugType(pointee));
        debugForwardPointer->setIdOperand(2, getDebugType(pointee));
    }

    return type->getResultId();
}

Id Builder::makeIntegerType(int width, bool hasSign)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeInt)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeInt)][t];
        if (type->getImmediateOperand(0) == (unsigned)width &&
            type->getImmediateOperand(1) == (hasSign ? 1u : 0u))
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeInt);
    type->reserveOperands(2);
    type->addImmediateOperand(width);
    type->addImmediateOperand(hasSign ? 1 : 0);
    groupedTypes[enumCast(Op::OpTypeInt)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    // deal with capabilities
    switch (width) {
    case 8:
    case 16:
        // these are currently handled by storage-type declarations and post processing
        break;
    case 64:
        addCapability(Capability::Int64);
        break;
    default:
        break;
    }

    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeIntegerDebugType(width, hasSign);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makeFloatType(int width)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeFloat)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeFloat)][t];
        if (type->getNumOperands() != 1) {
            continue;
        }
        if (type->getImmediateOperand(0) == (unsigned)width)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeFloat);
    type->addImmediateOperand(width);
    groupedTypes[enumCast(Op::OpTypeFloat)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    // deal with capabilities
    switch (width) {
    case 16:
        // currently handled by storage-type declarations and post processing
        break;
    case 64:
        addCapability(Capability::Float64);
        break;
    default:
        break;
    }

    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeFloatDebugType(width);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makeBFloat16Type()
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeFloat)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeFloat)][t];
        if (type->getNumOperands() != 2) {
            continue;
        }
        if (type->getImmediateOperand(0) == (unsigned)16 &&
            type->getImmediateOperand(1) == FPEncoding::BFloat16KHR)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeFloat);
    type->addImmediateOperand(16);
    type->addImmediateOperand(FPEncoding::BFloat16KHR);
    groupedTypes[enumCast(Op::OpTypeFloat)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    addExtension(spv::E_SPV_KHR_bfloat16);
    addCapability(Capability::BFloat16TypeKHR);

#if 0
    // XXX not supported
    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeFloatDebugType(width);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }
#endif

    return type->getResultId();
}

Id Builder::makeFloatE5M2Type()
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeFloat)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeFloat)][t];
        if (type->getNumOperands() != 2) {
            continue;
        }
        if (type->getImmediateOperand(0) == (unsigned)8 &&
            type->getImmediateOperand(1) == FPEncoding::Float8E5M2EXT)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeFloat);
    type->addImmediateOperand(8);
    type->addImmediateOperand(FPEncoding::Float8E5M2EXT);
    groupedTypes[enumCast(Op::OpTypeFloat)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    addExtension(spv::E_SPV_EXT_float8);
    addCapability(Capability::Float8EXT);

#if 0
    // XXX not supported
    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeFloatDebugType(width);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }
#endif

    return type->getResultId();
}

Id Builder::makeFloatE4M3Type()
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeFloat)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeFloat)][t];
        if (type->getNumOperands() != 2) {
            continue;
        }
        if (type->getImmediateOperand(0) == (unsigned)8 &&
            type->getImmediateOperand(1) == FPEncoding::Float8E4M3EXT)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeFloat);
    type->addImmediateOperand(8);
    type->addImmediateOperand(FPEncoding::Float8E4M3EXT);
    groupedTypes[enumCast(Op::OpTypeFloat)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    addExtension(spv::E_SPV_EXT_float8);
    addCapability(Capability::Float8EXT);

#if 0
    // XXX not supported
    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeFloatDebugType(width);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }
#endif

    return type->getResultId();
}

// Make a struct without checking for duplication.
// See makeStructResultType() for non-decorated structs
// needed as the result of some instructions, which does
// check for duplicates.
// For compiler-generated structs, debug info is ignored.
Id Builder::makeStructType(const std::vector<Id>& members, const std::vector<spv::StructMemberDebugInfo>& memberDebugInfo,
                           const char* name, bool const compilerGenerated)
{
    // Don't look for previous one, because in the general case,
    // structs can be duplicated except for decorations.

    // not found, make it
    Instruction* type = new Instruction(getUniqueId(), NoType, Op::OpTypeStruct);
    for (int op = 0; op < (int)members.size(); ++op)
        type->addIdOperand(members[op]);
    groupedTypes[enumCast(Op::OpTypeStruct)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);
    addName(type->getResultId(), name);

    if (emitNonSemanticShaderDebugInfo && !compilerGenerated) {
        assert(members.size() == memberDebugInfo.size());
        auto const debugResultId =
            makeCompositeDebugType(members, memberDebugInfo, name, NonSemanticShaderDebugInfo100Structure);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

// Make a struct for the simple results of several instructions,
// checking for duplication.
Id Builder::makeStructResultType(Id type0, Id type1)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeStruct)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeStruct)][t];
        if (type->getNumOperands() != 2)
            continue;
        if (type->getIdOperand(0) != type0 ||
            type->getIdOperand(1) != type1)
            continue;
        return type->getResultId();
    }

    // not found, make it
    std::vector<spv::Id> members;
    members.push_back(type0);
    members.push_back(type1);

    return makeStructType(members, {}, "ResType");
}

Id Builder::makeVectorType(Id component, int size)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeVector)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeVector)][t];
        if (type->getIdOperand(0) == component &&
            type->getImmediateOperand(1) == (unsigned)size)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeVector);
    type->reserveOperands(2);
    type->addIdOperand(component);
    type->addImmediateOperand(size);
    groupedTypes[enumCast(Op::OpTypeVector)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeVectorDebugType(component, size);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makeMatrixType(Id component, int cols, int rows)
{
    assert(cols <= maxMatrixSize && rows <= maxMatrixSize);

    Id column = makeVectorType(component, rows);

    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeMatrix)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeMatrix)][t];
        if (type->getIdOperand(0) == column &&
            type->getImmediateOperand(1) == (unsigned)cols)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeMatrix);
    type->reserveOperands(2);
    type->addIdOperand(column);
    type->addImmediateOperand(cols);
    groupedTypes[enumCast(Op::OpTypeMatrix)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeMatrixDebugType(column, cols);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makeCooperativeMatrixTypeKHR(Id component, Id scope, Id rows, Id cols, Id use)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeCooperativeMatrixKHR)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeCooperativeMatrixKHR)][t];
        if (type->getIdOperand(0) == component &&
            type->getIdOperand(1) == scope &&
            type->getIdOperand(2) == rows &&
            type->getIdOperand(3) == cols &&
            type->getIdOperand(4) == use)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeCooperativeMatrixKHR);
    type->reserveOperands(5);
    type->addIdOperand(component);
    type->addIdOperand(scope);
    type->addIdOperand(rows);
    type->addIdOperand(cols);
    type->addIdOperand(use);
    groupedTypes[enumCast(Op::OpTypeCooperativeMatrixKHR)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    if (emitNonSemanticShaderDebugInfo)
    {
        // Find a name for one of the parameters. It can either come from debuginfo for another
        // type, or an OpName from a constant.
        auto const findName = [&](Id id) {
            Id id2 = getDebugType(id);
            for (auto &t : groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeBasic]) {
                if (t->getResultId() == id2) {
                    for (auto &s : strings) {
                        if (s->getResultId() == t->getIdOperand(2)) {
                            return s->getNameString();
                        }
                    }
                }
            }
            for (auto &t : names) {
                if (t->getIdOperand(0) == id) {
                    return t->getNameString();
                }
            }
            return "unknown";
        };
        std::string debugName = "coopmat<";
        debugName += std::string(findName(component)) + ", ";
        if (isConstantScalar(scope)) {
            debugName += std::string("gl_Scope") + std::string(spv::ScopeToString((spv::Scope)getConstantScalar(scope))) + ", ";
        } else {
            debugName += std::string(findName(scope)) + ", ";
        }
        debugName += std::string(findName(rows)) + ", ";
        debugName += std::string(findName(cols)) + ">";
        // There's no nonsemantic debug info instruction for cooperative matrix types,
        // use opaque composite instead.
        auto const debugResultId = makeOpaqueDebugType(debugName.c_str());
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makeCooperativeMatrixTypeNV(Id component, Id scope, Id rows, Id cols)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeCooperativeMatrixNV)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeCooperativeMatrixNV)][t];
        if (type->getIdOperand(0) == component && type->getIdOperand(1) == scope && type->getIdOperand(2) == rows &&
            type->getIdOperand(3) == cols)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeCooperativeMatrixNV);
    type->reserveOperands(4);
    type->addIdOperand(component);
    type->addIdOperand(scope);
    type->addIdOperand(rows);
    type->addIdOperand(cols);
    groupedTypes[enumCast(Op::OpTypeCooperativeMatrixNV)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeCooperativeMatrixTypeWithSameShape(Id component, Id otherType)
{
    Instruction* instr = module.getInstruction(otherType);
    if (instr->getOpCode() == Op::OpTypeCooperativeMatrixNV) {
        return makeCooperativeMatrixTypeNV(component, instr->getIdOperand(1), instr->getIdOperand(2), instr->getIdOperand(3));
    } else {
        assert(instr->getOpCode() == Op::OpTypeCooperativeMatrixKHR);
        return makeCooperativeMatrixTypeKHR(component, instr->getIdOperand(1), instr->getIdOperand(2), instr->getIdOperand(3), instr->getIdOperand(4));
    }
}

Id Builder::makeCooperativeVectorTypeNV(Id componentType, Id components)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeCooperativeVectorNV)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeCooperativeVectorNV)][t];
        if (type->getIdOperand(0) == componentType &&
            type->getIdOperand(1) == components)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeCooperativeVectorNV);
    type->addIdOperand(componentType);
    type->addIdOperand(components);
    groupedTypes[enumCast(Op::OpTypeCooperativeVectorNV)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeTensorTypeARM(Id elementType, Id rank)
{
    // See if an OpTypeTensorARM with same element type and rank already exists.
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeTensorARM)].size(); ++t) {
        const Instruction *type = groupedTypes[enumCast(Op::OpTypeTensorARM)][t];
        if (type->getIdOperand(0) == elementType && type->getIdOperand(1) == rank)
            return type->getResultId();
    }

    // Not found, make it.
    std::unique_ptr<Instruction> type(new Instruction(getUniqueId(), NoType, Op::OpTypeTensorARM));
    type->addIdOperand(elementType);
    type->addIdOperand(rank);
    groupedTypes[enumCast(Op::OpTypeTensorARM)].push_back(type.get());
    module.mapInstruction(type.get());
    Id resultID = type->getResultId();
    constantsTypesGlobals.push_back(std::move(type));
    return resultID;
}

Id Builder::makeGenericType(spv::Op opcode, std::vector<spv::IdImmediate>& operands)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(opcode)].size(); ++t) {
        type = groupedTypes[enumCast(opcode)][t];
        if (static_cast<size_t>(type->getNumOperands()) != operands.size())
            continue; // Number mismatch, find next

        bool match = true;
        for (int op = 0; match && op < (int)operands.size(); ++op) {
            match = (operands[op].isId ? type->getIdOperand(op) : type->getImmediateOperand(op)) == operands[op].word;
        }
        if (match)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, opcode);
    type->reserveOperands(operands.size());
    for (size_t op = 0; op < operands.size(); ++op) {
        if (operands[op].isId)
            type->addIdOperand(operands[op].word);
        else
            type->addImmediateOperand(operands[op].word);
    }
    groupedTypes[enumCast(opcode)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

// TODO: performance: track arrays per stride
// If a stride is supplied (non-zero) make an array.
// If no stride (0), reuse previous array types.
// 'size' is an Id of a constant or specialization constant of the array size
Id Builder::makeArrayType(Id element, Id sizeId, int stride)
{
    Instruction* type;
    if (stride == 0) {
        // try to find existing type
        for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeArray)].size(); ++t) {
            type = groupedTypes[enumCast(Op::OpTypeArray)][t];
            if (type->getIdOperand(0) == element &&
                type->getIdOperand(1) == sizeId &&
                explicitlyLaidOut.find(type->getResultId()) == explicitlyLaidOut.end())
                return type->getResultId();
        }
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeArray);
    type->reserveOperands(2);
    type->addIdOperand(element);
    type->addIdOperand(sizeId);
    groupedTypes[enumCast(Op::OpTypeArray)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    if (stride != 0) {
        explicitlyLaidOut.insert(type->getResultId());
    }

    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeArrayDebugType(element, sizeId);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makeRuntimeArray(Id element)
{
    Instruction* type = new Instruction(getUniqueId(), NoType, Op::OpTypeRuntimeArray);
    type->addIdOperand(element);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeArrayDebugType(element, makeUintConstant(0));
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makeFunctionType(Id returnType, const std::vector<Id>& paramTypes)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeFunction)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeFunction)][t];
        if (type->getIdOperand(0) != returnType || (int)paramTypes.size() != type->getNumOperands() - 1)
            continue;
        bool mismatch = false;
        for (int p = 0; p < (int)paramTypes.size(); ++p) {
            if (paramTypes[p] != type->getIdOperand(p + 1)) {
                mismatch = true;
                break;
            }
        }
        if (! mismatch)
        {
            // If compiling HLSL, glslang will create a wrapper function around the entrypoint. Accordingly, a void(void)
            // function type is created for the wrapper function. However, nonsemantic shader debug information is disabled
            // while creating the HLSL wrapper. Consequently, if we encounter another void(void) function, we need to create
            // the associated debug function type if it hasn't been created yet.
            if(emitNonSemanticShaderDebugInfo && getDebugType(type->getResultId()) == NoType) {
                assert(sourceLang == spv::SourceLanguage::HLSL);
                assert(getTypeClass(returnType) == Op::OpTypeVoid && paramTypes.size() == 0);

                Id id = makeDebugFunctionType(returnType, {});
                debugTypeIdLookup[type->getResultId()] = id;
            }
            return type->getResultId();
        }
    }

    // not found, make it
    Id typeId = getUniqueId();
    type = new Instruction(typeId, NoType, Op::OpTypeFunction);
    type->reserveOperands(paramTypes.size() + 1);
    type->addIdOperand(returnType);
    for (int p = 0; p < (int)paramTypes.size(); ++p)
        type->addIdOperand(paramTypes[p]);
    groupedTypes[enumCast(Op::OpTypeFunction)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    // make debug type and map it
    if (emitNonSemanticShaderDebugInfo) {
        Id debugTypeId = makeDebugFunctionType(returnType, paramTypes);
        debugTypeIdLookup[typeId] = debugTypeId;
    }

    return type->getResultId();
}

Id Builder::makeDebugFunctionType(Id returnType, const std::vector<Id>& paramTypes)
{
    assert(getDebugType(returnType) != NoType);

    Id typeId = getUniqueId();
    auto type = new Instruction(typeId, makeVoidType(), Op::OpExtInst);
    type->reserveOperands(paramTypes.size() + 4);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugTypeFunction);
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100FlagIsPublic));
    type->addIdOperand(getDebugType(returnType));
    for (auto const paramType : paramTypes) {
        if (isPointerType(paramType) || isArrayType(paramType)) {
            type->addIdOperand(getDebugType(getContainedTypeId(paramType)));
        }
        else {
            type->addIdOperand(getDebugType(paramType));
        }
    }
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);
    return typeId;
}

Id Builder::makeImageType(Id sampledType, Dim dim, bool depth, bool arrayed, bool ms, unsigned sampled,
    ImageFormat format, const char* debugName)
{
    assert(sampled == 1 || sampled == 2);

    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeImage)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeImage)][t];
        if (type->getIdOperand(0) == sampledType &&
            type->getImmediateOperand(1) == (unsigned int)dim &&
            type->getImmediateOperand(2) == (  depth ? 1u : 0u) &&
            type->getImmediateOperand(3) == (arrayed ? 1u : 0u) &&
            type->getImmediateOperand(4) == (     ms ? 1u : 0u) &&
            type->getImmediateOperand(5) == sampled &&
            type->getImmediateOperand(6) == (unsigned int)format)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeImage);
    type->reserveOperands(7);
    type->addIdOperand(sampledType);
    type->addImmediateOperand(   dim);
    type->addImmediateOperand(  depth ? 1 : 0);
    type->addImmediateOperand(arrayed ? 1 : 0);
    type->addImmediateOperand(     ms ? 1 : 0);
    type->addImmediateOperand(sampled);
    type->addImmediateOperand((unsigned int)format);

    groupedTypes[enumCast(Op::OpTypeImage)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    // deal with capabilities
    switch (dim) {
    case Dim::Buffer:
        if (sampled == 1)
            addCapability(Capability::SampledBuffer);
        else
            addCapability(Capability::ImageBuffer);
        break;
    case Dim::Dim1D:
        if (sampled == 1)
            addCapability(Capability::Sampled1D);
        else
            addCapability(Capability::Image1D);
        break;
    case Dim::Cube:
        if (arrayed) {
            if (sampled == 1)
                addCapability(Capability::SampledCubeArray);
            else
                addCapability(Capability::ImageCubeArray);
        }
        break;
    case Dim::Rect:
        if (sampled == 1)
            addCapability(Capability::SampledRect);
        else
            addCapability(Capability::ImageRect);
        break;
    case Dim::SubpassData:
        addCapability(Capability::InputAttachment);
        break;
    default:
        break;
    }

    if (ms) {
        if (sampled == 2) {
            // Images used with subpass data are not storage
            // images, so don't require the capability for them.
            if (dim != Dim::SubpassData)
                addCapability(Capability::StorageImageMultisample);
            if (arrayed)
                addCapability(Capability::ImageMSArray);
        }
    }

    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeOpaqueDebugType(debugName);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makeSampledImageType(Id imageType, const char* debugName)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypeSampledImage)].size(); ++t) {
        type = groupedTypes[enumCast(Op::OpTypeSampledImage)][t];
        if (type->getIdOperand(0) == imageType)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, Op::OpTypeSampledImage);
    type->addIdOperand(imageType);

    groupedTypes[enumCast(Op::OpTypeSampledImage)].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    if (emitNonSemanticShaderDebugInfo)
    {
        auto const debugResultId = makeOpaqueDebugType(debugName);
        debugTypeIdLookup[type->getResultId()] = debugResultId;
    }

    return type->getResultId();
}

Id Builder::makeDebugInfoNone()
{
    if (debugInfoNone != 0)
        return debugInfoNone;

    Instruction* inst = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    inst->reserveOperands(2);
    inst->addIdOperand(nonSemanticShaderDebugInfo);
    inst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugInfoNone);

    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(inst));
    module.mapInstruction(inst);

    debugInfoNone = inst->getResultId();

    return debugInfoNone;
}

Id Builder::makeBoolDebugType(int const size)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeBasic].size(); ++t) {
        type = groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeBasic][t];
        if (type->getIdOperand(0) == getStringId("bool") &&
            type->getIdOperand(1) == static_cast<unsigned int>(size) &&
            type->getIdOperand(2) == NonSemanticShaderDebugInfo100Boolean)
            return type->getResultId();
    }

    type = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    type->reserveOperands(6);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugTypeBasic);

    type->addIdOperand(getStringId("bool")); // name id
    type->addIdOperand(makeUintConstant(size)); // size id
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100Boolean)); // encoding id
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100None)); // flags id

    groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeBasic].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeIntegerDebugType(int const width, bool const hasSign)
{
    const char* typeName = nullptr;
    switch (width) {
        case 8:  typeName = hasSign ? "int8_t" : "uint8_t"; break;
        case 16: typeName = hasSign ? "int16_t" : "uint16_t"; break;
        case 64: typeName = hasSign ? "int64_t" : "uint64_t"; break;
        default: typeName = hasSign ? "int" : "uint";
    }
    auto nameId = getStringId(typeName);
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeBasic].size(); ++t) {
        type = groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeBasic][t];
        if (type->getIdOperand(0) == nameId &&
            type->getIdOperand(1) == static_cast<unsigned int>(width) &&
            type->getIdOperand(2) == (hasSign ? NonSemanticShaderDebugInfo100Signed : NonSemanticShaderDebugInfo100Unsigned))
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    type->reserveOperands(6);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugTypeBasic);
    type->addIdOperand(nameId); // name id
    type->addIdOperand(makeUintConstant(width)); // size id
    if(hasSign == true) {
        type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100Signed)); // encoding id
    } else {
        type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100Unsigned)); // encoding id
    }
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100None)); // flags id

    groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeBasic].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeFloatDebugType(int const width)
{
    const char* typeName = nullptr;
    switch (width) {
        case 16: typeName = "float16_t"; break;
        case 64: typeName = "double"; break;
        default: typeName = "float"; break;
    }
    auto nameId = getStringId(typeName);
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeBasic].size(); ++t) {
        type = groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeBasic][t];
        if (type->getIdOperand(0) == nameId &&
            type->getIdOperand(1) == static_cast<unsigned int>(width) &&
            type->getIdOperand(2) == NonSemanticShaderDebugInfo100Float)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    type->reserveOperands(6);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugTypeBasic);
    type->addIdOperand(nameId); // name id
    type->addIdOperand(makeUintConstant(width)); // size id
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100Float)); // encoding id
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100None)); // flags id

    groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeBasic].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeSequentialDebugType(Id const baseType, Id const componentCount, NonSemanticShaderDebugInfo100Instructions const sequenceType)
{
    assert(sequenceType == NonSemanticShaderDebugInfo100DebugTypeArray ||
        sequenceType == NonSemanticShaderDebugInfo100DebugTypeVector);

    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedDebugTypes[sequenceType].size(); ++t) {
        type = groupedDebugTypes[sequenceType][t];
        if (type->getIdOperand(0) == baseType &&
            type->getIdOperand(1) == makeUintConstant(componentCount))
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    type->reserveOperands(4);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(sequenceType);
    type->addIdOperand(getDebugType(baseType)); // base type
    type->addIdOperand(componentCount); // component count

    groupedDebugTypes[sequenceType].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeArrayDebugType(Id const baseType, Id const componentCount)
{
    return makeSequentialDebugType(baseType, componentCount, NonSemanticShaderDebugInfo100DebugTypeArray);
}

Id Builder::makeVectorDebugType(Id const baseType, int const componentCount)
{
    return makeSequentialDebugType(baseType, makeUintConstant(componentCount), NonSemanticShaderDebugInfo100DebugTypeVector);
}

Id Builder::makeMatrixDebugType(Id const vectorType, int const vectorCount, bool columnMajor)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeMatrix].size(); ++t) {
        type = groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeMatrix][t];
        if (type->getIdOperand(0) == vectorType &&
            type->getIdOperand(1) == makeUintConstant(vectorCount))
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    type->reserveOperands(5);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugTypeMatrix);
    type->addIdOperand(getDebugType(vectorType)); // vector type id
    type->addIdOperand(makeUintConstant(vectorCount)); // component count id
    type->addIdOperand(makeBoolConstant(columnMajor)); // column-major id

    groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeMatrix].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeMemberDebugType(Id const memberType, StructMemberDebugInfo const& debugTypeLoc)
{
    assert(getDebugType(memberType) != NoType);

    Instruction* type = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    type->reserveOperands(10);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugTypeMember);
    type->addIdOperand(getStringId(debugTypeLoc.name)); // name id
    type->addIdOperand(debugTypeLoc.debugTypeOverride != 0 ? debugTypeLoc.debugTypeOverride
                                                           : getDebugType(memberType)); // type id
    type->addIdOperand(makeDebugSource(currentFileId));                            // source id
    type->addIdOperand(makeUintConstant(debugTypeLoc.line));   // line id TODO: currentLine is always zero
    type->addIdOperand(makeUintConstant(debugTypeLoc.column)); // TODO: column id
    type->addIdOperand(makeUintConstant(0));                   // TODO: offset id
    type->addIdOperand(makeUintConstant(0));                   // TODO: size id
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100FlagIsPublic)); // flags id

    groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeMember].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeCompositeDebugType(std::vector<Id> const& memberTypes, std::vector<StructMemberDebugInfo> const& memberDebugInfo,
                                   char const* const name, NonSemanticShaderDebugInfo100DebugCompositeType const tag)
{
    // Create the debug member types.
    std::vector<Id> memberDebugTypes;
    assert(memberTypes.size() == memberDebugInfo.size());
    for (size_t i = 0; i < memberTypes.size(); i++) {
        if (getDebugType(memberTypes[i]) != NoType) {
            memberDebugTypes.emplace_back(makeMemberDebugType(memberTypes[i], memberDebugInfo[i]));
        }
    }

    // Create The structure debug type.
    Instruction* type = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    type->reserveOperands(memberDebugTypes.size() + 11);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugTypeComposite);
    type->addIdOperand(getStringId(name)); // name id
    type->addIdOperand(makeUintConstant(tag)); // tag id
    type->addIdOperand(makeDebugSource(currentFileId)); // source id
    type->addIdOperand(makeUintConstant(currentLine)); // line id TODO: currentLine always zero?
    type->addIdOperand(makeUintConstant(0)); // TODO: column id
    type->addIdOperand(makeDebugCompilationUnit()); // scope id
    type->addIdOperand(getStringId(name)); // linkage name id
    type->addIdOperand(makeUintConstant(0)); // TODO: size id
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100FlagIsPublic)); // flags id
    for(auto const memberDebugType : memberDebugTypes) {
        type->addIdOperand(memberDebugType);
    }

    groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeComposite].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

// The NonSemantic Shader Debug Info doesn't really have a dedicated opcode for opaque types. Instead, we use DebugTypeComposite.
// To represent a source language opaque type, this instruction must have no Members operands, Size operand must be
// DebugInfoNone, and Name must start with @ to avoid clashes with user defined names.
Id Builder::makeOpaqueDebugType(char const* const name)
{
    // Create The structure debug type.
    Instruction* type = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    type->reserveOperands(11);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugTypeComposite);
    type->addIdOperand(getStringId(name)); // name id
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100Structure)); // tag id
    type->addIdOperand(makeDebugSource(currentFileId)); // source id
    type->addIdOperand(makeUintConstant(currentLine)); // line id TODO: currentLine always zero?
    type->addIdOperand(makeUintConstant(0)); // TODO: column id
    type->addIdOperand(makeDebugCompilationUnit()); // scope id
    // Prepend '@' to opaque types.
    type->addIdOperand(getStringId('@' + std::string(name))); // linkage name id
    type->addIdOperand(makeDebugInfoNone()); // size id
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100FlagIsPublic)); // flags id

    groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypeComposite].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makePointerDebugType(StorageClass storageClass, Id const baseType)
{
    const Id debugBaseType = getDebugType(baseType);
    if (!debugBaseType) {
        return makeDebugInfoNone();
    }
    const Id scID = makeUintConstant(storageClass);
    for (Instruction* otherType : groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypePointer]) {
        if (otherType->getIdOperand(2) == debugBaseType &&
            otherType->getIdOperand(3) == scID) {
            return otherType->getResultId();
        }
    }

    Instruction* type = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    type->reserveOperands(5);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugTypePointer);
    type->addIdOperand(debugBaseType);
    type->addIdOperand(scID);
    type->addIdOperand(makeUintConstant(0));

    groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypePointer].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

// Emit a OpExtInstWithForwardRefsKHR nonsemantic instruction for a pointer debug type
// where we don't have the pointee yet. Since we don't have the pointee yet, it just
// points to itself and we rely on patching it later.
Id Builder::makeForwardPointerDebugType(StorageClass storageClass)
{
    const Id scID = makeUintConstant(storageClass);

    this->addExtension(spv::E_SPV_KHR_relaxed_extended_instruction);

    Instruction *type = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInstWithForwardRefsKHR);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugTypePointer);
    type->addIdOperand(type->getResultId());
    type->addIdOperand(scID);
    type->addIdOperand(makeUintConstant(0));

    groupedDebugTypes[NonSemanticShaderDebugInfo100DebugTypePointer].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeDebugSource(const Id fileName) {
    if (debugSourceId.find(fileName) != debugSourceId.end())
        return debugSourceId[fileName];
    spv::Id resultId = getUniqueId();
    Instruction* sourceInst = new Instruction(resultId, makeVoidType(), Op::OpExtInst);
    sourceInst->reserveOperands(3);
    sourceInst->addIdOperand(nonSemanticShaderDebugInfo);
    sourceInst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugSource);
    sourceInst->addIdOperand(fileName);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(sourceInst));
    module.mapInstruction(sourceInst);
    if (emitNonSemanticShaderDebugSource) {
        const int maxWordCount = 0xFFFF;
        const int opSourceWordCount = 4;
        const int nonNullBytesPerInstruction = 4 * (maxWordCount - opSourceWordCount) - 1;
        auto processDebugSource = [&](std::string source) {
            if (source.size() > 0) {
                int nextByte = 0;
                while ((int)source.size() - nextByte > 0) {
                    auto subString = source.substr(nextByte, nonNullBytesPerInstruction);
                    auto sourceId = getStringId(subString);
                    if (nextByte == 0) {
                        // DebugSource
                        sourceInst->addIdOperand(sourceId);
                    } else {
                        // DebugSourceContinued
                        Instruction* sourceContinuedInst = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
                        sourceContinuedInst->reserveOperands(2);
                        sourceContinuedInst->addIdOperand(nonSemanticShaderDebugInfo);
                        sourceContinuedInst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugSourceContinued);
                        sourceContinuedInst->addIdOperand(sourceId);
                        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(sourceContinuedInst));
                        module.mapInstruction(sourceContinuedInst);
                    }
                    nextByte += nonNullBytesPerInstruction;
                }
            } else {
                auto sourceId = getStringId(source);
                sourceInst->addIdOperand(sourceId);
            }
        };
        if (fileName == mainFileId) {
            processDebugSource(sourceText);
        } else {
            auto incItr = includeFiles.find(fileName);
            if (incItr != includeFiles.end()) {
                processDebugSource(*incItr->second);
            } else {
                // We omit the optional source text item if not available in glslang
            }
        }
    }
    debugSourceId[fileName] = resultId;
    return resultId;
}

Id Builder::makeDebugCompilationUnit() {
    if (nonSemanticShaderCompilationUnitId != 0)
        return nonSemanticShaderCompilationUnitId;
    spv::Id resultId = getUniqueId();
    Instruction* sourceInst = new Instruction(resultId, makeVoidType(), Op::OpExtInst);
    sourceInst->reserveOperands(6);
    sourceInst->addIdOperand(nonSemanticShaderDebugInfo);
    sourceInst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugCompilationUnit);
    sourceInst->addIdOperand(makeUintConstant(1)); // TODO(greg-lunarg): Get rid of magic number
    sourceInst->addIdOperand(makeUintConstant(4)); // TODO(greg-lunarg): Get rid of magic number
    sourceInst->addIdOperand(makeDebugSource(mainFileId));
    sourceInst->addIdOperand(makeUintConstant(sourceLang));
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(sourceInst));
    module.mapInstruction(sourceInst);
    nonSemanticShaderCompilationUnitId = resultId;

    // We can reasonably assume that makeDebugCompilationUnit will be called before any of
    // debug-scope stack. Function scopes and lexical scopes will occur afterward.
    assert(currentDebugScopeId.empty());
    currentDebugScopeId.push(nonSemanticShaderCompilationUnitId);

    return resultId;
}

Id Builder::createDebugGlobalVariable(Id const type, char const*const name, Id const variable)
{
    assert(type != 0);

    Instruction* inst = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    inst->reserveOperands(11);
    inst->addIdOperand(nonSemanticShaderDebugInfo);
    inst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugGlobalVariable);
    inst->addIdOperand(getStringId(name)); // name id
    inst->addIdOperand(type); // type id
    inst->addIdOperand(makeDebugSource(currentFileId)); // source id
    inst->addIdOperand(makeUintConstant(currentLine)); // line id TODO: currentLine always zero?
    inst->addIdOperand(makeUintConstant(0)); // TODO: column id
    inst->addIdOperand(makeDebugCompilationUnit()); // scope id
    inst->addIdOperand(getStringId(name)); // linkage name id
    inst->addIdOperand(variable); // variable id
    inst->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100FlagIsDefinition)); // flags id

    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(inst));
    module.mapInstruction(inst);

    return inst->getResultId();
}

Id Builder::createDebugLocalVariable(Id type, char const*const name, size_t const argNumber)
{
    assert(name != nullptr);
    assert(!currentDebugScopeId.empty());

    Instruction* inst = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    inst->reserveOperands(9);
    inst->addIdOperand(nonSemanticShaderDebugInfo);
    inst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugLocalVariable);
    inst->addIdOperand(getStringId(name)); // name id
    inst->addIdOperand(type); // type id
    inst->addIdOperand(makeDebugSource(currentFileId)); // source id
    inst->addIdOperand(makeUintConstant(currentLine)); // line id
    inst->addIdOperand(makeUintConstant(0)); // TODO: column id
    inst->addIdOperand(currentDebugScopeId.top()); // scope id
    inst->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100FlagIsLocal)); // flags id
    if(argNumber != 0) {
        inst->addIdOperand(makeUintConstant(static_cast<unsigned int>(argNumber)));
    }

    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(inst));
    module.mapInstruction(inst);

    return inst->getResultId();
}

Id Builder::makeDebugExpression()
{
    if (debugExpression != 0)
        return debugExpression;

    Instruction* inst = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    inst->reserveOperands(2);
    inst->addIdOperand(nonSemanticShaderDebugInfo);
    inst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugExpression);

    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(inst));
    module.mapInstruction(inst);

    debugExpression = inst->getResultId();

    return debugExpression;
}

Id Builder::makeDebugDeclare(Id const debugLocalVariable, Id const pointer)
{
    Instruction* inst = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    inst->reserveOperands(5);
    inst->addIdOperand(nonSemanticShaderDebugInfo);
    inst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugDeclare);
    inst->addIdOperand(debugLocalVariable); // debug local variable id
    inst->addIdOperand(pointer); // pointer to local variable id
    inst->addIdOperand(makeDebugExpression()); // expression id
    addInstruction(std::unique_ptr<Instruction>(inst));

    return inst->getResultId();
}

Id Builder::makeDebugValue(Id const debugLocalVariable, Id const value)
{
    Instruction* inst = new Instruction(getUniqueId(), makeVoidType(), Op::OpExtInst);
    inst->reserveOperands(5);
    inst->addIdOperand(nonSemanticShaderDebugInfo);
    inst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugValue);
    inst->addIdOperand(debugLocalVariable); // debug local variable id
    inst->addIdOperand(value); // value of local variable id
    inst->addIdOperand(makeDebugExpression()); // expression id
    addInstruction(std::unique_ptr<Instruction>(inst));

    return inst->getResultId();
}

Id Builder::makeAccelerationStructureType()
{
    Instruction *type;
    if (groupedTypes[enumCast(Op::OpTypeAccelerationStructureKHR)].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, Op::OpTypeAccelerationStructureKHR);
        groupedTypes[enumCast(Op::OpTypeAccelerationStructureKHR)].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
        if (emitNonSemanticShaderDebugInfo) {
            spv::Id debugType = makeOpaqueDebugType("accelerationStructure");
            debugTypeIdLookup[type->getResultId()] = debugType;
        }
    } else {
        type = groupedTypes[enumCast(Op::OpTypeAccelerationStructureKHR)].back();
    }

    return type->getResultId();
}

Id Builder::makeRayQueryType()
{
    Instruction *type;
    if (groupedTypes[enumCast(Op::OpTypeRayQueryKHR)].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, Op::OpTypeRayQueryKHR);
        groupedTypes[enumCast(Op::OpTypeRayQueryKHR)].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
        if (emitNonSemanticShaderDebugInfo) {
            spv::Id debugType = makeOpaqueDebugType("rayQuery");
            debugTypeIdLookup[type->getResultId()] = debugType;
        }
    } else {
        type = groupedTypes[enumCast(Op::OpTypeRayQueryKHR)].back();
    }

    return type->getResultId();
}

Id Builder::makeHitObjectEXTType()
{
    Instruction *type;
    if (groupedTypes[enumCast(Op::OpTypeHitObjectEXT)].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, Op::OpTypeHitObjectEXT);
        groupedTypes[enumCast(Op::OpTypeHitObjectEXT)].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
    } else {
        type = groupedTypes[enumCast(Op::OpTypeHitObjectEXT)].back();
    }

    return type->getResultId();
}
Id Builder::makeHitObjectNVType()
{
    Instruction *type;
    if (groupedTypes[enumCast(Op::OpTypeHitObjectNV)].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, Op::OpTypeHitObjectNV);
        groupedTypes[enumCast(Op::OpTypeHitObjectNV)].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
        if (emitNonSemanticShaderDebugInfo) {
            spv::Id debugType = makeOpaqueDebugType("hitObjectNV");
            debugTypeIdLookup[type->getResultId()] = debugType;
        }
    } else {
        type = groupedTypes[enumCast(Op::OpTypeHitObjectNV)].back();
    }

    return type->getResultId();
}

Id Builder::getDerefTypeId(Id resultId) const
{
    Id typeId = getTypeId(resultId);
    assert(isPointerType(typeId));

    return module.getInstruction(typeId)->getIdOperand(1);
}

Op Builder::getMostBasicTypeClass(Id typeId) const
{
    Instruction* instr = module.getInstruction(typeId);

    Op typeClass = instr->getOpCode();
    switch (typeClass)
    {
    case Op::OpTypeVector:
    case Op::OpTypeMatrix:
    case Op::OpTypeArray:
    case Op::OpTypeRuntimeArray:
        return getMostBasicTypeClass(instr->getIdOperand(0));
    case Op::OpTypePointer:
        return getMostBasicTypeClass(instr->getIdOperand(1));
    default:
        return typeClass;
    }
}

unsigned int Builder::getNumTypeConstituents(Id typeId) const
{
    Instruction* instr = module.getInstruction(typeId);

    switch (instr->getOpCode())
    {
    case Op::OpTypeBool:
    case Op::OpTypeInt:
    case Op::OpTypeFloat:
    case Op::OpTypePointer:
        return 1;
    case Op::OpTypeVector:
    case Op::OpTypeMatrix:
        return instr->getImmediateOperand(1);
    case Op::OpTypeCooperativeVectorNV:
    case Op::OpTypeArray:
    {
        Id lengthId = instr->getIdOperand(1);
        return module.getInstruction(lengthId)->getImmediateOperand(0);
    }
    case Op::OpTypeStruct:
        return instr->getNumOperands();
    case Op::OpTypeCooperativeMatrixKHR:
    case Op::OpTypeCooperativeMatrixNV:
        // has only one constituent when used with OpCompositeConstruct.
        return 1;
    default:
        assert(0);
        return 1;
    }
}

// Return the lowest-level type of scalar that an homogeneous composite is made out of.
// Typically, this is just to find out if something is made out of ints or floats.
// However, it includes returning a structure, if say, it is an array of structure.
Id Builder::getScalarTypeId(Id typeId) const
{
    Instruction* instr = module.getInstruction(typeId);

    Op typeClass = instr->getOpCode();
    switch (typeClass)
    {
    case Op::OpTypeVoid:
    case Op::OpTypeBool:
    case Op::OpTypeInt:
    case Op::OpTypeFloat:
    case Op::OpTypeStruct:
        return instr->getResultId();
    case Op::OpTypeVector:
    case Op::OpTypeMatrix:
    case Op::OpTypeArray:
    case Op::OpTypeRuntimeArray:
    case Op::OpTypePointer:
    case Op::OpTypeCooperativeVectorNV:
        return getScalarTypeId(getContainedTypeId(typeId));
    default:
        assert(0);
        return NoResult;
    }
}

// Return the type of 'member' of a composite.
Id Builder::getContainedTypeId(Id typeId, int member) const
{
    Instruction* instr = module.getInstruction(typeId);

    Op typeClass = instr->getOpCode();
    switch (typeClass)
    {
    case Op::OpTypeVector:
    case Op::OpTypeMatrix:
    case Op::OpTypeArray:
    case Op::OpTypeRuntimeArray:
    case Op::OpTypeCooperativeMatrixKHR:
    case Op::OpTypeCooperativeMatrixNV:
    case Op::OpTypeCooperativeVectorNV:
        return instr->getIdOperand(0);
    case Op::OpTypePointer:
        return instr->getIdOperand(1);
    case Op::OpTypeStruct:
        return instr->getIdOperand(member);
    default:
        assert(0);
        return NoResult;
    }
}

// Figure out the final resulting type of the access chain.
Id Builder::getResultingAccessChainType() const
{
    assert(accessChain.base != NoResult);
    Id typeId = getTypeId(accessChain.base);

    assert(isPointerType(typeId));
    typeId = getContainedTypeId(typeId);

    for (int i = 0; i < (int)accessChain.indexChain.size(); ++i) {
        if (isStructType(typeId)) {
            assert(isConstantScalar(accessChain.indexChain[i]));
            typeId = getContainedTypeId(typeId, getConstantScalar(accessChain.indexChain[i]));
        } else
            typeId = getContainedTypeId(typeId, accessChain.indexChain[i]);
    }

    return typeId;
}

// Return the immediately contained type of a given composite type.
Id Builder::getContainedTypeId(Id typeId) const
{
    return getContainedTypeId(typeId, 0);
}

// Returns true if 'typeId' is or contains a scalar type declared with 'typeOp'
// of width 'width'. The 'width' is only consumed for int and float types.
// Returns false otherwise.
bool Builder::containsType(Id typeId, spv::Op typeOp, unsigned int width) const
{
    const Instruction& instr = *module.getInstruction(typeId);

    Op typeClass = instr.getOpCode();
    switch (typeClass)
    {
    case Op::OpTypeInt:
    case Op::OpTypeFloat:
        return typeClass == typeOp && instr.getImmediateOperand(0) == width;
    case Op::OpTypeStruct:
        for (int m = 0; m < instr.getNumOperands(); ++m) {
            if (containsType(instr.getIdOperand(m), typeOp, width))
                return true;
        }
        return false;
    case Op::OpTypePointer:
        return false;
    case Op::OpTypeVector:
    case Op::OpTypeMatrix:
    case Op::OpTypeArray:
    case Op::OpTypeRuntimeArray:
        return containsType(getContainedTypeId(typeId), typeOp, width);
    default:
        return typeClass == typeOp;
    }
}

// return true if the type is a pointer to PhysicalStorageBufferEXT or an
// contains such a pointer. These require restrict/aliased decorations.
bool Builder::containsPhysicalStorageBufferOrArray(Id typeId) const
{
    const Instruction& instr = *module.getInstruction(typeId);

    Op typeClass = instr.getOpCode();
    switch (typeClass)
    {
    case Op::OpTypePointer:
        return getTypeStorageClass(typeId) == StorageClass::PhysicalStorageBufferEXT;
    case Op::OpTypeArray:
        return containsPhysicalStorageBufferOrArray(getContainedTypeId(typeId));
    case Op::OpTypeStruct:
        for (int m = 0; m < instr.getNumOperands(); ++m) {
            if (containsPhysicalStorageBufferOrArray(instr.getIdOperand(m)))
                return true;
        }
        return false;
    default:
        return false;
    }
}

// See if a scalar constant of this type has already been created, so it
// can be reused rather than duplicated.  (Required by the specification).
Id Builder::findScalarConstant(Op typeClass, Op opcode, Id typeId, unsigned value)
{
    ScalarConstantKey key{ enumCast(typeClass), enumCast(opcode), typeId, value, 0 };
    auto it = groupedScalarConstantResultIDs.find(key);
    return (it != groupedScalarConstantResultIDs.end()) ? it->second : 0;
}

// Version of findScalarConstant (see above) for scalars that take two operands (e.g. a 'double' or 'int64').
Id Builder::findScalarConstant(Op typeClass, Op opcode, Id typeId, unsigned v1, unsigned v2)
{
    ScalarConstantKey key{ enumCast(typeClass), enumCast(opcode), typeId, v1, v2 };
    auto it = groupedScalarConstantResultIDs.find(key);
    return (it != groupedScalarConstantResultIDs.end()) ? it->second : 0;
}

// Return true if consuming 'opcode' means consuming a constant.
// "constant" here means after final transform to executable code,
// the value consumed will be a constant, so includes specialization.
bool Builder::isConstantOpCode(Op opcode) const
{
    switch (opcode) {
    case Op::OpUndef:
    case Op::OpConstantTrue:
    case Op::OpConstantFalse:
    case Op::OpConstant:
    case Op::OpConstantComposite:
    case Op::OpConstantCompositeReplicateEXT:
    case Op::OpConstantSampler:
    case Op::OpConstantNull:
    case Op::OpSpecConstantTrue:
    case Op::OpSpecConstantFalse:
    case Op::OpSpecConstant:
    case Op::OpSpecConstantComposite:
    case Op::OpSpecConstantCompositeReplicateEXT:
    case Op::OpSpecConstantOp:
        return true;
    default:
        return false;
    }
}

// Return true if consuming 'opcode' means consuming a specialization constant.
bool Builder::isSpecConstantOpCode(Op opcode) const
{
    switch (opcode) {
    case Op::OpSpecConstantTrue:
    case Op::OpSpecConstantFalse:
    case Op::OpSpecConstant:
    case Op::OpSpecConstantComposite:
    case Op::OpSpecConstantOp:
    case Op::OpSpecConstantCompositeReplicateEXT:
        return true;
    default:
        return false;
    }
}

Id Builder::makeNullConstant(Id typeId)
{
    Instruction* constant;

    // See if we already made it.
    Id existing = NoResult;
    for (int i = 0; i < (int)nullConstants.size(); ++i) {
        constant = nullConstants[i];
        if (constant->getTypeId() == typeId)
            existing = constant->getResultId();
    }

    if (existing != NoResult)
        return existing;

    // Make it
    Instruction* c = new Instruction(getUniqueId(), typeId, Op::OpConstantNull);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    nullConstants.push_back(c);
    module.mapInstruction(c);

    return c->getResultId();
}

Id Builder::makeBoolConstant(bool b, bool specConstant)
{
    Id typeId = makeBoolType();
    Op opcode = specConstant ? (b ? Op::OpSpecConstantTrue : Op::OpSpecConstantFalse) : (b ? Op::OpConstantTrue : Op::OpConstantFalse);

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (!specConstant) {
        Id existing = findScalarConstant(Op::OpTypeBool, opcode, typeId, 0);
        if (existing)
            return existing;
    }

    // Make it
    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    module.mapInstruction(c);

    Id resultId = c->getResultId();
    if (!specConstant) {
        ScalarConstantKey key{enumCast(Op::OpTypeBool), enumCast(opcode), typeId, 0, 0};
        groupedScalarConstantResultIDs[key] = resultId;
    }
    return resultId;
}

Id Builder::makeIntConstant(Id typeId, unsigned value, bool specConstant)
{
    Op opcode = specConstant ? Op::OpSpecConstant : Op::OpConstant;

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (! specConstant) {
        Id existing = findScalarConstant(Op::OpTypeInt, opcode, typeId, value);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(value);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    module.mapInstruction(c);

    Id resultId = c->getResultId();
    if (!specConstant) {
        ScalarConstantKey key{ enumCast(Op::OpTypeInt), enumCast(opcode), typeId, value, 0 };
        groupedScalarConstantResultIDs[key] = resultId;
    }
    return resultId;
}

Id Builder::makeInt64Constant(Id typeId, unsigned long long value, bool specConstant)
{
    Op opcode = specConstant ? Op::OpSpecConstant : Op::OpConstant;

    unsigned op1 = value & 0xFFFFFFFF;
    unsigned op2 = value >> 32;

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (! specConstant) {
        Id existing = findScalarConstant(Op::OpTypeInt, opcode, typeId, op1, op2);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->reserveOperands(2);
    c->addImmediateOperand(op1);
    c->addImmediateOperand(op2);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    module.mapInstruction(c);

    Id resultId = c->getResultId();
    if (!specConstant) {
        ScalarConstantKey key{ enumCast(Op::OpTypeInt), enumCast(opcode), typeId, op1, op2 };
        groupedScalarConstantResultIDs[key] = resultId;
    }
    return resultId;
}

Id Builder::makeFloatConstant(float f, bool specConstant)
{
    Op opcode = specConstant ? Op::OpSpecConstant : Op::OpConstant;
    Id typeId = makeFloatType(32);
    union { float fl; unsigned int ui; } u;
    u.fl = f;
    unsigned value = u.ui;

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (! specConstant) {
        Id existing = findScalarConstant(Op::OpTypeFloat, opcode, typeId, value);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(value);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    module.mapInstruction(c);

    Id resultId = c->getResultId();
    if (!specConstant) {
        ScalarConstantKey key{ enumCast(Op::OpTypeFloat), enumCast(opcode), typeId, value, 0 };
        groupedScalarConstantResultIDs[key] = resultId;
    }
    return resultId;
}

Id Builder::makeDoubleConstant(double d, bool specConstant)
{
    Op opcode = specConstant ? Op::OpSpecConstant : Op::OpConstant;
    Id typeId = makeFloatType(64);
    union { double db; unsigned long long ull; } u;
    u.db = d;
    unsigned long long value = u.ull;
    unsigned op1 = value & 0xFFFFFFFF;
    unsigned op2 = value >> 32;

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (! specConstant) {
        Id existing = findScalarConstant(Op::OpTypeFloat, opcode, typeId, op1, op2);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->reserveOperands(2);
    c->addImmediateOperand(op1);
    c->addImmediateOperand(op2);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    module.mapInstruction(c);

    Id resultId = c->getResultId();
    if (!specConstant) {
        ScalarConstantKey key{ enumCast(Op::OpTypeFloat), enumCast(opcode), typeId, op1, op2 };
        groupedScalarConstantResultIDs[key] = resultId;
    }
    return resultId;
}

Id Builder::makeFloat16Constant(float f16, bool specConstant)
{
    Op opcode = specConstant ? Op::OpSpecConstant : Op::OpConstant;
    Id typeId = makeFloatType(16);

    spvutils::HexFloat<spvutils::FloatProxy<float>> fVal(f16);
    spvutils::HexFloat<spvutils::FloatProxy<spvutils::Float16>> f16Val(0);
    fVal.castTo(f16Val, spvutils::kRoundToZero);

    unsigned value = f16Val.value().getAsFloat().get_value();

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (!specConstant) {
        Id existing = findScalarConstant(Op::OpTypeFloat, opcode, typeId, value);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(value);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    module.mapInstruction(c);

    Id resultId = c->getResultId();
    if (!specConstant) {
        ScalarConstantKey key{ enumCast(Op::OpTypeFloat), enumCast(opcode), typeId, value, 0 };
        groupedScalarConstantResultIDs[key] = resultId;
    }
    return resultId;
}

Id Builder::makeBFloat16Constant(float bf16, bool specConstant)
{
    Op opcode = specConstant ? Op::OpSpecConstant : Op::OpConstant;
    Id typeId = makeBFloat16Type();

    union {
        float f;
        uint32_t u;
    } un;
    un.f = bf16;

    // take high 16b of fp32 value. This is effectively round-to-zero, other than certain NaNs.
    unsigned value = un.u >> 16;

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (!specConstant) {
        Id existing = findScalarConstant(Op::OpTypeFloat, opcode, typeId, value);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(value);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    module.mapInstruction(c);

    Id resultId = c->getResultId();
    if (!specConstant) {
        ScalarConstantKey key{ enumCast(Op::OpTypeFloat), enumCast(opcode), typeId, value, 0 };
        groupedScalarConstantResultIDs[key] = resultId;
    }
    return resultId;
}

Id Builder::makeFloatE5M2Constant(float fe5m2, bool specConstant)
{
    Op opcode = specConstant ? Op::OpSpecConstant : Op::OpConstant;
    Id typeId = makeFloatE5M2Type();

    spvutils::HexFloat<spvutils::FloatProxy<float>> fVal(fe5m2);
    spvutils::HexFloat<spvutils::FloatProxy<spvutils::FloatE5M2>> fe5m2Val(0);
    fVal.castTo(fe5m2Val, spvutils::kRoundToZero);

    unsigned value = fe5m2Val.value().getAsFloat().get_value();

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (!specConstant) {
        Id existing = findScalarConstant(Op::OpTypeFloat, opcode, typeId, value);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(value);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    module.mapInstruction(c);

    Id resultId = c->getResultId();
    if (!specConstant) {
        ScalarConstantKey key{enumCast(Op::OpTypeFloat), enumCast(opcode), typeId, value, 0};
        groupedScalarConstantResultIDs[key] = resultId;
    }
    return resultId;
}

Id Builder::makeFloatE4M3Constant(float fe4m3, bool specConstant)
{
    Op opcode = specConstant ? Op::OpSpecConstant : Op::OpConstant;
    Id typeId = makeFloatE4M3Type();

    spvutils::HexFloat<spvutils::FloatProxy<float>> fVal(fe4m3);
    spvutils::HexFloat<spvutils::FloatProxy<spvutils::FloatE4M3>> fe4m3Val(0);
    fVal.castTo(fe4m3Val, spvutils::kRoundToZero);

    unsigned value = fe4m3Val.value().getAsFloat().get_value();

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (!specConstant) {
        Id existing = findScalarConstant(Op::OpTypeFloat, opcode, typeId, value);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(value);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    module.mapInstruction(c);

    Id resultId = c->getResultId();
    if (!specConstant) {
        ScalarConstantKey key{enumCast(Op::OpTypeFloat), enumCast(opcode), typeId, value, 0};
        groupedScalarConstantResultIDs[key] = resultId;
    }
    return resultId;
}

Id Builder::makeFpConstant(Id type, double d, bool specConstant)
{
    const int width = getScalarTypeWidth(type);

    assert(isFloatType(type));

    switch (width) {
    case 16:
            return makeFloat16Constant((float)d, specConstant);
    case 32:
            return makeFloatConstant((float)d, specConstant);
    case 64:
            return makeDoubleConstant(d, specConstant);
    default:
            break;
    }

    assert(false);
    return NoResult;
}

Id Builder::importNonSemanticShaderDebugInfoInstructions()
{
    assert(emitNonSemanticShaderDebugInfo == true);

    if(nonSemanticShaderDebugInfo == 0)
    {
        this->addExtension(spv::E_SPV_KHR_non_semantic_info);
        nonSemanticShaderDebugInfo = this->import("NonSemantic.Shader.DebugInfo.100");
    }

    return nonSemanticShaderDebugInfo;
}

Id Builder::findCompositeConstant(Op typeClass, Op opcode, Id typeId, const std::vector<Id>& comps, size_t numMembers)
{
    Instruction* constant = nullptr;
    bool found = false;
    for (int i = 0; i < (int)groupedCompositeConstants[enumCast(typeClass)].size(); ++i) {
        constant = groupedCompositeConstants[enumCast(typeClass)][i];

        if (constant->getTypeId() != typeId)
            continue;

        if (constant->getOpCode() != opcode) {
            continue;
        }

        if (constant->getNumOperands() != (int)numMembers)
            continue;

        // same contents?
        bool mismatch = false;
        for (int op = 0; op < constant->getNumOperands(); ++op) {
            if (constant->getIdOperand(op) != comps[op]) {
                mismatch = true;
                break;
            }
        }
        if (! mismatch) {
            found = true;
            break;
        }
    }

    return found ? constant->getResultId() : NoResult;
}

Id Builder::findStructConstant(Id typeId, const std::vector<Id>& comps)
{
    Instruction* constant = nullptr;
    bool found = false;
    for (int i = 0; i < (int)groupedStructConstants[typeId].size(); ++i) {
        constant = groupedStructConstants[typeId][i];

        // same contents?
        bool mismatch = false;
        for (int op = 0; op < constant->getNumOperands(); ++op) {
            if (constant->getIdOperand(op) != comps[op]) {
                mismatch = true;
                break;
            }
        }
        if (! mismatch) {
            found = true;
            break;
        }
    }

    return found ? constant->getResultId() : NoResult;
}

// Comments in header
Id Builder::makeCompositeConstant(Id typeId, const std::vector<Id>& members, bool specConstant)
{
    assert(typeId);
    Op typeClass = getTypeClass(typeId);

    bool replicate = false;
    size_t numMembers = members.size();
    if (useReplicatedComposites || typeClass == Op::OpTypeCooperativeVectorNV) {
        // use replicate if all members are the same
        replicate = numMembers > 0 &&
            std::equal(members.begin() + 1, members.end(), members.begin());

        if (replicate) {
            numMembers = 1;
            addCapability(spv::Capability::ReplicatedCompositesEXT);
            addExtension(spv::E_SPV_EXT_replicated_composites);
        }
    }

    Op opcode = replicate ?
        (specConstant ? Op::OpSpecConstantCompositeReplicateEXT : Op::OpConstantCompositeReplicateEXT) :
        (specConstant ? Op::OpSpecConstantComposite : Op::OpConstantComposite);

    switch (typeClass) {
    case Op::OpTypeVector:
    case Op::OpTypeArray:
    case Op::OpTypeMatrix:
    case Op::OpTypeCooperativeMatrixKHR:
    case Op::OpTypeCooperativeMatrixNV:
    case Op::OpTypeCooperativeVectorNV:
        if (! specConstant) {
            Id existing = findCompositeConstant(typeClass, opcode, typeId, members, numMembers);
            if (existing)
                return existing;
        }
        break;
    case Op::OpTypeStruct:
        if (! specConstant) {
            Id existing = findStructConstant(typeId, members);
            if (existing)
                return existing;
        }
        break;
    default:
        assert(0);
        return makeFloatConstant(0.0);
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->reserveOperands(members.size());
    for (size_t op = 0; op < numMembers; ++op)
        c->addIdOperand(members[op]);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    if (typeClass == Op::OpTypeStruct)
        groupedStructConstants[typeId].push_back(c);
    else
        groupedCompositeConstants[enumCast(typeClass)].push_back(c);
    module.mapInstruction(c);

    return c->getResultId();
}

Instruction* Builder::addEntryPoint(ExecutionModel model, Function* function, const char* name)
{
    Instruction* entryPoint = new Instruction(Op::OpEntryPoint);
    entryPoint->reserveOperands(3);
    entryPoint->addImmediateOperand(model);
    entryPoint->addIdOperand(function->getId());
    entryPoint->addStringOperand(name);

    entryPoints.push_back(std::unique_ptr<Instruction>(entryPoint));

    return entryPoint;
}

// Currently relying on the fact that all 'value' of interest are small non-negative values.
void Builder::addExecutionMode(Function* entryPoint, ExecutionMode mode, int value1, int value2, int value3)
{
    // entryPoint can be null if we are in compile-only mode
    if (!entryPoint)
        return;

    Instruction* instr = new Instruction(Op::OpExecutionMode);
    instr->reserveOperands(3);
    instr->addIdOperand(entryPoint->getId());
    instr->addImmediateOperand(mode);
    if (value1 >= 0)
        instr->addImmediateOperand(value1);
    if (value2 >= 0)
        instr->addImmediateOperand(value2);
    if (value3 >= 0)
        instr->addImmediateOperand(value3);

    executionModes.push_back(std::unique_ptr<Instruction>(instr));
}

void Builder::addExecutionMode(Function* entryPoint, ExecutionMode mode, const std::vector<unsigned>& literals)
{
    // entryPoint can be null if we are in compile-only mode
    if (!entryPoint)
        return;

    Instruction* instr = new Instruction(Op::OpExecutionMode);
    instr->reserveOperands(literals.size() + 2);
    instr->addIdOperand(entryPoint->getId());
    instr->addImmediateOperand(mode);
    for (auto literal : literals)
        instr->addImmediateOperand(literal);

    executionModes.push_back(std::unique_ptr<Instruction>(instr));
}

void Builder::addExecutionModeId(Function* entryPoint, ExecutionMode mode, const std::vector<Id>& operandIds)
{
    // entryPoint can be null if we are in compile-only mode
    if (!entryPoint)
        return;

    Instruction* instr = new Instruction(Op::OpExecutionModeId);
    instr->reserveOperands(operandIds.size() + 2);
    instr->addIdOperand(entryPoint->getId());
    instr->addImmediateOperand(mode);
    for (auto operandId : operandIds)
        instr->addIdOperand(operandId);

    executionModes.push_back(std::unique_ptr<Instruction>(instr));
}

void Builder::addName(Id id, const char* string)
{
    Instruction* name = new Instruction(Op::OpName);
    name->reserveOperands(2);
    name->addIdOperand(id);
    name->addStringOperand(string);

    names.push_back(std::unique_ptr<Instruction>(name));
}

void Builder::addMemberName(Id id, int memberNumber, const char* string)
{
    Instruction* name = new Instruction(Op::OpMemberName);
    name->reserveOperands(3);
    name->addIdOperand(id);
    name->addImmediateOperand(memberNumber);
    name->addStringOperand(string);

    names.push_back(std::unique_ptr<Instruction>(name));
}

void Builder::addDecoration(Id id, Decoration decoration, int num)
{
    if (decoration == spv::Decoration::Max)
        return;

    Instruction* dec = new Instruction(Op::OpDecorate);
    dec->reserveOperands(2);
    dec->addIdOperand(id);
    dec->addImmediateOperand(decoration);
    if (num >= 0)
        dec->addImmediateOperand(num);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addDecoration(Id id, Decoration decoration, const char* s)
{
    if (decoration == spv::Decoration::Max)
        return;

    Instruction* dec = new Instruction(Op::OpDecorateString);
    dec->reserveOperands(3);
    dec->addIdOperand(id);
    dec->addImmediateOperand(decoration);
    dec->addStringOperand(s);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addDecoration(Id id, Decoration decoration, const std::vector<unsigned>& literals)
{
    if (decoration == spv::Decoration::Max)
        return;

    Instruction* dec = new Instruction(Op::OpDecorate);
    dec->reserveOperands(literals.size() + 2);
    dec->addIdOperand(id);
    dec->addImmediateOperand(decoration);
    for (auto literal : literals)
        dec->addImmediateOperand(literal);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addDecoration(Id id, Decoration decoration, const std::vector<const char*>& strings)
{
    if (decoration == spv::Decoration::Max)
        return;

    Instruction* dec = new Instruction(Op::OpDecorateString);
    dec->reserveOperands(strings.size() + 2);
    dec->addIdOperand(id);
    dec->addImmediateOperand(decoration);
    for (auto string : strings)
        dec->addStringOperand(string);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addLinkageDecoration(Id id, const char* name, spv::LinkageType linkType) {
    Instruction* dec = new Instruction(Op::OpDecorate);
    dec->reserveOperands(4);
    dec->addIdOperand(id);
    dec->addImmediateOperand(spv::Decoration::LinkageAttributes);
    dec->addStringOperand(name);
    dec->addImmediateOperand(linkType);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addDecorationId(Id id, Decoration decoration, Id idDecoration)
{
    if (decoration == spv::Decoration::Max)
        return;

    Instruction* dec = new Instruction(Op::OpDecorateId);
    dec->reserveOperands(3);
    dec->addIdOperand(id);
    dec->addImmediateOperand(decoration);
    dec->addIdOperand(idDecoration);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addDecorationId(Id id, Decoration decoration, const std::vector<Id>& operandIds)
{
    if(decoration == spv::Decoration::Max)
        return;

    Instruction* dec = new Instruction(Op::OpDecorateId);
    dec->reserveOperands(operandIds.size() + 2);
    dec->addIdOperand(id);
    dec->addImmediateOperand(decoration);

    for (auto operandId : operandIds)
        dec->addIdOperand(operandId);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addMemberDecoration(Id id, unsigned int member, Decoration decoration, int num)
{
    if (decoration == spv::Decoration::Max)
        return;

    Instruction* dec = new Instruction(Op::OpMemberDecorate);
    dec->reserveOperands(3);
    dec->addIdOperand(id);
    dec->addImmediateOperand(member);
    dec->addImmediateOperand(decoration);
    if (num >= 0)
        dec->addImmediateOperand(num);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addMemberDecoration(Id id, unsigned int member, Decoration decoration, const char *s)
{
    if (decoration == spv::Decoration::Max)
        return;

    Instruction* dec = new Instruction(Op::OpMemberDecorateStringGOOGLE);
    dec->reserveOperands(4);
    dec->addIdOperand(id);
    dec->addImmediateOperand(member);
    dec->addImmediateOperand(decoration);
    dec->addStringOperand(s);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addMemberDecoration(Id id, unsigned int member, Decoration decoration, const std::vector<unsigned>& literals)
{
    if (decoration == spv::Decoration::Max)
        return;

    Instruction* dec = new Instruction(Op::OpMemberDecorate);
    dec->reserveOperands(literals.size() + 3);
    dec->addIdOperand(id);
    dec->addImmediateOperand(member);
    dec->addImmediateOperand(decoration);
    for (auto literal : literals)
        dec->addImmediateOperand(literal);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addMemberDecoration(Id id, unsigned int member, Decoration decoration, const std::vector<const char*>& strings)
{
    if (decoration == spv::Decoration::Max)
        return;

    Instruction* dec = new Instruction(Op::OpMemberDecorateString);
    dec->reserveOperands(strings.size() + 3);
    dec->addIdOperand(id);
    dec->addImmediateOperand(member);
    dec->addImmediateOperand(decoration);
    for (auto string : strings)
        dec->addStringOperand(string);

    decorations.insert(std::unique_ptr<Instruction>(dec));
}

void Builder::addInstruction(std::unique_ptr<Instruction> inst) {
    // Phis must appear first in their block, don't insert line tracking instructions
    // in front of them, just add the OpPhi and return.
    if (inst->getOpCode() == Op::OpPhi) {
        buildPoint->addInstruction(std::move(inst));
        return;
    }
    // Optionally insert OpDebugScope
    if (emitNonSemanticShaderDebugInfo && dirtyScopeTracker) {
        if (buildPoint->updateDebugScope(currentDebugScopeId.top())) {
            auto scopeInst = std::make_unique<Instruction>(getUniqueId(), makeVoidType(), Op::OpExtInst);
            scopeInst->reserveOperands(3);
            scopeInst->addIdOperand(nonSemanticShaderDebugInfo);
            scopeInst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugScope);
            scopeInst->addIdOperand(currentDebugScopeId.top());
            buildPoint->addInstruction(std::move(scopeInst));
        }

        dirtyScopeTracker = false;
    }

    // Insert OpLine/OpDebugLine if the debug source location has changed
    if (trackDebugInfo && dirtyLineTracker) {
        if (buildPoint->updateDebugSourceLocation(currentLine, 0, currentFileId)) {
            if (emitSpirvDebugInfo) {
                auto lineInst = std::make_unique<Instruction>(Op::OpLine);
                lineInst->reserveOperands(3);
                lineInst->addIdOperand(currentFileId);
                lineInst->addImmediateOperand(currentLine);
                lineInst->addImmediateOperand(0);
                buildPoint->addInstruction(std::move(lineInst));
            }
            if (emitNonSemanticShaderDebugInfo) {
                auto lineInst = std::make_unique<Instruction>(getUniqueId(), makeVoidType(), Op::OpExtInst);
                lineInst->reserveOperands(7);
                lineInst->addIdOperand(nonSemanticShaderDebugInfo);
                lineInst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugLine);
                lineInst->addIdOperand(makeDebugSource(currentFileId));
                lineInst->addIdOperand(makeUintConstant(currentLine));
                lineInst->addIdOperand(makeUintConstant(currentLine));
                lineInst->addIdOperand(makeUintConstant(0));
                lineInst->addIdOperand(makeUintConstant(0));
                buildPoint->addInstruction(std::move(lineInst));
            }
        }

        dirtyLineTracker = false;
    }

    buildPoint->addInstruction(std::move(inst));
}

void Builder::addInstructionNoDebugInfo(std::unique_ptr<Instruction> inst) {
    buildPoint->addInstruction(std::move(inst));
}

// Comments in header
Function* Builder::makeEntryPoint(const char* entryPoint)
{
    assert(! entryPointFunction);

    auto const returnType = makeVoidType();

    restoreNonSemanticShaderDebugInfo = emitNonSemanticShaderDebugInfo;
    if(sourceLang == spv::SourceLanguage::HLSL) {
        emitNonSemanticShaderDebugInfo = false;
    }

    Block* entry = nullptr;
    entryPointFunction = makeFunctionEntry(NoPrecision, returnType, entryPoint, LinkageType::Max, {}, {}, &entry);

    emitNonSemanticShaderDebugInfo = restoreNonSemanticShaderDebugInfo;

    return entryPointFunction;
}

// Comments in header
Function* Builder::makeFunctionEntry(Decoration precision, Id returnType, const char* name, LinkageType linkType,
                                     const std::vector<Id>& paramTypes,
                                     const std::vector<std::vector<Decoration>>& decorations, Block** entry)
{
    // Make the function and initial instructions in it
    Id typeId = makeFunctionType(returnType, paramTypes);
    Id firstParamId = paramTypes.size() == 0 ? 0 : getUniqueIds((int)paramTypes.size());
    Id funcId = getUniqueId();
    Function* function = new Function(funcId, returnType, typeId, firstParamId, linkType, name, module);

    // Set up the precisions
    setPrecision(function->getId(), precision);
    function->setReturnPrecision(precision);
    for (unsigned p = 0; p < (unsigned)decorations.size(); ++p) {
        for (int d = 0; d < (int)decorations[p].size(); ++d) {
            addDecoration(firstParamId + p, decorations[p][d]);
            function->addParamPrecision(p, decorations[p][d]);
        }
    }

    // reset last debug scope
    if (emitNonSemanticShaderDebugInfo) {
        dirtyScopeTracker = true;
    }

    // CFG
    assert(entry != nullptr);
    *entry = new Block(getUniqueId(), *function);
    function->addBlock(*entry);
    setBuildPoint(*entry);

    if (name)
        addName(function->getId(), name);

    functions.push_back(std::unique_ptr<Function>(function));

    return function;
}

void Builder::setupFunctionDebugInfo(Function* function, const char* name, const std::vector<Id>& paramTypes,
                                     const std::vector<char const*>& paramNames)
{

    if (!emitNonSemanticShaderDebugInfo)
        return;

    Id nameId = getStringId(unmangleFunctionName(name));
    Id funcTypeId = function->getFuncTypeId();
    assert(getDebugType(funcTypeId) != NoType);
    Id funcId = function->getId();

    assert(funcId != 0);

    // Make the debug function instruction
    Id debugFuncId = makeDebugFunction(function, nameId, funcTypeId);
    debugFuncIdLookup[funcId] = debugFuncId;
    currentDebugScopeId.push(debugFuncId);

    // DebugScope and DebugLine for parameter DebugDeclares
    assert(paramTypes.size() == paramNames.size());
    if ((int)paramTypes.size() > 0) {
        Id firstParamId = function->getParamId(0);

        for (size_t p = 0; p < paramTypes.size(); ++p) {
            bool passByRef = false;
            Id paramTypeId = paramTypes[p];

            // For pointer-typed parameters, they are actually passed by reference and we need unwrap the pointer to get the actual parameter type.
            if (isPointerType(paramTypeId) || isArrayType(paramTypeId)) {
                passByRef = true;
                paramTypeId = getContainedTypeId(paramTypeId);
            }

            auto const& paramName = paramNames[p];
            auto const debugLocalVariableId = createDebugLocalVariable(getDebugType(paramTypeId), paramName, p + 1);
            auto const paramId = static_cast<Id>(firstParamId + p);

            if (passByRef) {
                makeDebugDeclare(debugLocalVariableId, paramId);
            } else {
                makeDebugValue(debugLocalVariableId, paramId);
            }
        }
    }

    // Clear debug scope stack
    if (emitNonSemanticShaderDebugInfo)
        currentDebugScopeId.pop();
}

Id Builder::makeDebugFunction([[maybe_unused]] Function* function, Id nameId, Id funcTypeId)
{
    assert(function != nullptr);
    assert(nameId != 0);
    assert(funcTypeId != 0);
    assert(getDebugType(funcTypeId) != NoType);

    Id funcId = getUniqueId();
    auto type = new Instruction(funcId, makeVoidType(), Op::OpExtInst);
    type->reserveOperands(11);
    type->addIdOperand(nonSemanticShaderDebugInfo);
    type->addImmediateOperand(NonSemanticShaderDebugInfo100DebugFunction);
    type->addIdOperand(nameId);
    type->addIdOperand(getDebugType(funcTypeId));
    type->addIdOperand(makeDebugSource(currentFileId)); // TODO: This points to file of definition instead of declaration
    type->addIdOperand(makeUintConstant(currentLine)); // TODO: This points to line of definition instead of declaration
    type->addIdOperand(makeUintConstant(0)); // column
    type->addIdOperand(makeDebugCompilationUnit()); // scope
    type->addIdOperand(nameId); // linkage name
    type->addIdOperand(makeUintConstant(NonSemanticShaderDebugInfo100FlagIsPublic));
    type->addIdOperand(makeUintConstant(currentLine));
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);
    return funcId;
}

Id Builder::makeDebugLexicalBlock(uint32_t line, uint32_t column) {
    assert(!currentDebugScopeId.empty());

    Id lexId = getUniqueId();
    auto lex = new Instruction(lexId, makeVoidType(), Op::OpExtInst);
    lex->reserveOperands(6);
    lex->addIdOperand(nonSemanticShaderDebugInfo);
    lex->addImmediateOperand(NonSemanticShaderDebugInfo100DebugLexicalBlock);
    lex->addIdOperand(makeDebugSource(currentFileId));
    lex->addIdOperand(makeUintConstant(line));
    lex->addIdOperand(makeUintConstant(column)); // column
    lex->addIdOperand(currentDebugScopeId.top()); // scope
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(lex));
    module.mapInstruction(lex);
    return lexId;
}

std::string Builder::unmangleFunctionName(std::string const& name) const
{
    assert(name.length() > 0);

    if(name.rfind('(') != std::string::npos) {
        return name.substr(0, name.rfind('('));
    } else {
        return name;
    }
}

// Comments in header
void Builder::makeReturn(bool implicit, Id retVal)
{
    if (retVal) {
        Instruction* inst = new Instruction(NoResult, NoType, Op::OpReturnValue);
        inst->addIdOperand(retVal);
        addInstruction(std::unique_ptr<Instruction>(inst));
    } else
        addInstruction(std::unique_ptr<Instruction>(new Instruction(NoResult, NoType, Op::OpReturn)));

    if (! implicit)
        createAndSetNoPredecessorBlock("post-return");
}

// Comments in header
void Builder::enterLexicalBlock(uint32_t line, uint32_t column)
{
    if (!emitNonSemanticShaderDebugInfo) {
        return;
    }

    // Generate new lexical scope debug instruction
    Id lexId = makeDebugLexicalBlock(line, column);
    currentDebugScopeId.push(lexId);
    dirtyScopeTracker = true;
}

// Comments in header
void Builder::leaveLexicalBlock()
{
    if (!emitNonSemanticShaderDebugInfo) {
        return;
    }

    // Pop current scope from stack and clear current scope
    currentDebugScopeId.pop();
    dirtyScopeTracker = true;
}

// Comments in header
void Builder::enterFunction(Function const* function)
{
    currentFunction = function;

    // Save and disable debugInfo for HLSL entry point function. It is a wrapper
    // function with no user code in it.
    restoreNonSemanticShaderDebugInfo = emitNonSemanticShaderDebugInfo;
    if (sourceLang == spv::SourceLanguage::HLSL && function == entryPointFunction) {
        emitNonSemanticShaderDebugInfo = false;
    }

    if (emitNonSemanticShaderDebugInfo) {
        // Initialize scope state
        Id funcId = function->getFuncId();
        Id debugFuncId = getDebugFunction(funcId);
        currentDebugScopeId.push(debugFuncId);
        // Create DebugFunctionDefinition
        spv::Id resultId = getUniqueId();
        Instruction* defInst = new Instruction(resultId, makeVoidType(), Op::OpExtInst);
        defInst->reserveOperands(4);
        defInst->addIdOperand(nonSemanticShaderDebugInfo);
        defInst->addImmediateOperand(NonSemanticShaderDebugInfo100DebugFunctionDefinition);
        defInst->addIdOperand(debugFuncId);
        defInst->addIdOperand(funcId);
        addInstruction(std::unique_ptr<Instruction>(defInst));
    }

    if (auto linkType = function->getLinkType(); linkType != LinkageType::Max) {
        Id funcId = function->getFuncId();
        addCapability(Capability::Linkage);
        addLinkageDecoration(funcId, function->getExportName(), linkType);
    }
}

// Comments in header
void Builder::leaveFunction()
{
    Block* block = buildPoint;
    Function& function = buildPoint->getParent();
    assert(block);

    // If our function did not contain a return, add a return void now.
    if (! block->isTerminated()) {
        if (function.getReturnType() == makeVoidType())
            makeReturn(true);
        else {
            makeReturn(true, createUndefined(function.getReturnType()));
        }
    }

    // Clear function scope from debug scope stack
    if (emitNonSemanticShaderDebugInfo)
        currentDebugScopeId.pop();

    emitNonSemanticShaderDebugInfo = restoreNonSemanticShaderDebugInfo;

    // Clear current function record
    currentFunction = nullptr;
}

// Comments in header
void Builder::makeStatementTerminator(spv::Op opcode, const char *name)
{
    addInstruction(std::unique_ptr<Instruction>(new Instruction(opcode)));
    createAndSetNoPredecessorBlock(name);
}

// Comments in header
void Builder::makeStatementTerminator(spv::Op opcode, const std::vector<Id>& operands, const char* name)
{
    // It's assumed that the terminator instruction is always of void return type
    // However in future if there is a need for non void return type, new helper
    // methods can be created.
    createNoResultOp(opcode, operands);
    createAndSetNoPredecessorBlock(name);
}

void Builder::createConstVariable(Id type, const char* name, Id constant, bool isGlobal)
{
    if (emitNonSemanticShaderDebugInfo) {
        Id debugType = getDebugType(type);
        if (isGlobal) {
            createDebugGlobalVariable(debugType, name, constant);
        }
        else {
            auto debugLocal = createDebugLocalVariable(debugType, name);
            makeDebugValue(debugLocal, constant);
        }
    }
}

// Comments in header
Id Builder::createVariable(Decoration precision, StorageClass storageClass, Id type, const char* name, Id initializer,
    bool const compilerGenerated)
{
    Id pointerType = makePointer(storageClass, type);
    Instruction* inst = new Instruction(getUniqueId(), pointerType, Op::OpVariable);
    inst->addImmediateOperand(storageClass);
    if (initializer != NoResult)
        inst->addIdOperand(initializer);

    if (storageClass == StorageClass::Function) {
        // Validation rules require the declaration in the entry block
        buildPoint->getParent().addLocalVariable(std::unique_ptr<Instruction>(inst));
    }
    else {
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(inst));
        module.mapInstruction(inst);
    }

    if (emitNonSemanticShaderDebugInfo && !compilerGenerated)
    {
        // For debug info, we prefer respecting how the variable is declared in source code.
        // We may emulate some local variables as global variable with private storage in SPIR-V, but we still want to
        // treat them as local variables in debug info.
        if (storageClass == StorageClass::Function || (currentFunction && storageClass == StorageClass::Private)) {
            auto const debugLocalVariableId = createDebugLocalVariable(getDebugType(type), name);
            makeDebugDeclare(debugLocalVariableId, inst->getResultId());
        }
        else {
            createDebugGlobalVariable(getDebugType(type), name, inst->getResultId());
        }
    }

    if (name)
        addName(inst->getResultId(), name);
    setPrecision(inst->getResultId(), precision);

    return inst->getResultId();
}

// Comments in header
Id Builder::createUndefined(Id type)
{
  Instruction* inst = new Instruction(getUniqueId(), type, Op::OpUndef);
  addInstruction(std::unique_ptr<Instruction>(inst));
  return inst->getResultId();
}

// av/vis/nonprivate are unnecessary and illegal for some storage classes.
spv::MemoryAccessMask Builder::sanitizeMemoryAccessForStorageClass(spv::MemoryAccessMask memoryAccess, StorageClass sc)
    const
{
    switch (sc) {
    case spv::StorageClass::Uniform:
    case spv::StorageClass::Workgroup:
    case spv::StorageClass::StorageBuffer:
    case spv::StorageClass::PhysicalStorageBufferEXT:
        break;
    default:
        memoryAccess = spv::MemoryAccessMask(memoryAccess &
                        ~(spv::MemoryAccessMask::MakePointerAvailableKHR |
                          spv::MemoryAccessMask::MakePointerVisibleKHR |
                          spv::MemoryAccessMask::NonPrivatePointerKHR));
        break;
    }
    return memoryAccess;
}

// Comments in header
void Builder::createStore(Id rValue, Id lValue, spv::MemoryAccessMask memoryAccess, spv::Scope scope,
    unsigned int alignment)
{
    Instruction* store = new Instruction(Op::OpStore);
    store->reserveOperands(2);
    store->addIdOperand(lValue);
    store->addIdOperand(rValue);

    memoryAccess = sanitizeMemoryAccessForStorageClass(memoryAccess, getStorageClass(lValue));

    if (memoryAccess != MemoryAccessMask::MaskNone) {
        store->addImmediateOperand(memoryAccess);
        if (anySet(memoryAccess, spv::MemoryAccessMask::Aligned)) {
            store->addImmediateOperand(alignment);
        }
        if (anySet(memoryAccess, spv::MemoryAccessMask::MakePointerAvailableKHR)) {
            store->addIdOperand(makeUintConstant(scope));
        }
    }

    addInstruction(std::unique_ptr<Instruction>(store));
}

// Comments in header
Id Builder::createLoad(Id lValue, spv::Decoration precision, spv::MemoryAccessMask memoryAccess,
    spv::Scope scope, unsigned int alignment)
{
    Instruction* load = new Instruction(getUniqueId(), getDerefTypeId(lValue), Op::OpLoad);
    load->addIdOperand(lValue);

    memoryAccess = sanitizeMemoryAccessForStorageClass(memoryAccess, getStorageClass(lValue));

    if (memoryAccess != MemoryAccessMask::MaskNone) {
        load->addImmediateOperand(memoryAccess);
        if (anySet(memoryAccess, spv::MemoryAccessMask::Aligned)) {
            load->addImmediateOperand(alignment);
        }
        if (anySet(memoryAccess, spv::MemoryAccessMask::MakePointerVisibleKHR)) {
            load->addIdOperand(makeUintConstant(scope));
        }
    }

    addInstruction(std::unique_ptr<Instruction>(load));
    setPrecision(load->getResultId(), precision);

    return load->getResultId();
}

// Comments in header
Id Builder::createAccessChain(StorageClass storageClass, Id base, const std::vector<Id>& offsets)
{
    // Figure out the final resulting type.
    Id typeId = getResultingAccessChainType();
    typeId = makePointer(storageClass, typeId);

    // Make the instruction
    Instruction* chain = new Instruction(getUniqueId(), typeId, Op::OpAccessChain);
    chain->reserveOperands(offsets.size() + 1);
    chain->addIdOperand(base);
    for (int i = 0; i < (int)offsets.size(); ++i)
        chain->addIdOperand(offsets[i]);
    addInstruction(std::unique_ptr<Instruction>(chain));

    return chain->getResultId();
}

Id Builder::createArrayLength(Id base, unsigned int member, unsigned int bits)
{
    spv::Id intType = makeUintType(bits);
    Instruction* length = new Instruction(getUniqueId(), intType, Op::OpArrayLength);
    length->reserveOperands(2);
    length->addIdOperand(base);
    length->addImmediateOperand(member);
    addInstruction(std::unique_ptr<Instruction>(length));

    return length->getResultId();
}

Id Builder::createCooperativeMatrixLengthKHR(Id type)
{
    spv::Id intType = makeUintType(32);

    // Generate code for spec constants if in spec constant operation
    // generation mode.
    if (generatingOpCodeForSpecConst) {
        return createSpecConstantOp(Op::OpCooperativeMatrixLengthKHR, intType, std::vector<Id>(1, type), std::vector<Id>());
    }

    Instruction* length = new Instruction(getUniqueId(), intType, Op::OpCooperativeMatrixLengthKHR);
    length->addIdOperand(type);
    addInstruction(std::unique_ptr<Instruction>(length));

    return length->getResultId();
}

Id Builder::createCooperativeMatrixLengthNV(Id type)
{
    spv::Id intType = makeUintType(32);

    // Generate code for spec constants if in spec constant operation
    // generation mode.
    if (generatingOpCodeForSpecConst) {
        return createSpecConstantOp(Op::OpCooperativeMatrixLengthNV, intType, std::vector<Id>(1, type), std::vector<Id>());
    }

    Instruction* length = new Instruction(getUniqueId(), intType, Op::OpCooperativeMatrixLengthNV);
    length->addIdOperand(type);
    addInstruction(std::unique_ptr<Instruction>(length));

    return length->getResultId();
}

Id Builder::createCompositeExtract(Id composite, Id typeId, unsigned index)
{
    // Generate code for spec constants if in spec constant operation
    // generation mode.
    if (generatingOpCodeForSpecConst) {
        return createSpecConstantOp(Op::OpCompositeExtract, typeId, std::vector<Id>(1, composite),
            std::vector<Id>(1, index));
    }
    Instruction* extract = new Instruction(getUniqueId(), typeId, Op::OpCompositeExtract);
    extract->reserveOperands(2);
    extract->addIdOperand(composite);
    extract->addImmediateOperand(index);
    addInstruction(std::unique_ptr<Instruction>(extract));

    return extract->getResultId();
}

Id Builder::createCompositeExtract(Id composite, Id typeId, const std::vector<unsigned>& indexes)
{
    // Generate code for spec constants if in spec constant operation
    // generation mode.
    if (generatingOpCodeForSpecConst) {
        return createSpecConstantOp(Op::OpCompositeExtract, typeId, std::vector<Id>(1, composite), indexes);
    }
    Instruction* extract = new Instruction(getUniqueId(), typeId, Op::OpCompositeExtract);
    extract->reserveOperands(indexes.size() + 1);
    extract->addIdOperand(composite);
    for (int i = 0; i < (int)indexes.size(); ++i)
        extract->addImmediateOperand(indexes[i]);
    addInstruction(std::unique_ptr<Instruction>(extract));

    return extract->getResultId();
}

Id Builder::createCompositeInsert(Id object, Id composite, Id typeId, unsigned index)
{
    Instruction* insert = new Instruction(getUniqueId(), typeId, Op::OpCompositeInsert);
    insert->reserveOperands(3);
    insert->addIdOperand(object);
    insert->addIdOperand(composite);
    insert->addImmediateOperand(index);
    addInstruction(std::unique_ptr<Instruction>(insert));

    return insert->getResultId();
}

Id Builder::createCompositeInsert(Id object, Id composite, Id typeId, const std::vector<unsigned>& indexes)
{
    Instruction* insert = new Instruction(getUniqueId(), typeId, Op::OpCompositeInsert);
    insert->reserveOperands(indexes.size() + 2);
    insert->addIdOperand(object);
    insert->addIdOperand(composite);
    for (int i = 0; i < (int)indexes.size(); ++i)
        insert->addImmediateOperand(indexes[i]);
    addInstruction(std::unique_ptr<Instruction>(insert));

    return insert->getResultId();
}

Id Builder::createVectorExtractDynamic(Id vector, Id typeId, Id componentIndex)
{
    Instruction* extract = new Instruction(getUniqueId(), typeId, Op::OpVectorExtractDynamic);
    extract->reserveOperands(2);
    extract->addIdOperand(vector);
    extract->addIdOperand(componentIndex);
    addInstruction(std::unique_ptr<Instruction>(extract));

    return extract->getResultId();
}

Id Builder::createVectorInsertDynamic(Id vector, Id typeId, Id component, Id componentIndex)
{
    Instruction* insert = new Instruction(getUniqueId(), typeId, Op::OpVectorInsertDynamic);
    insert->reserveOperands(3);
    insert->addIdOperand(vector);
    insert->addIdOperand(component);
    insert->addIdOperand(componentIndex);
    addInstruction(std::unique_ptr<Instruction>(insert));

    return insert->getResultId();
}

// An opcode that has no operands, no result id, and no type
void Builder::createNoResultOp(Op opCode)
{
    Instruction* op = new Instruction(opCode);
    addInstruction(std::unique_ptr<Instruction>(op));
}

// An opcode that has one id operand, no result id, and no type
void Builder::createNoResultOp(Op opCode, Id operand)
{
    Instruction* op = new Instruction(opCode);
    op->addIdOperand(operand);
    addInstruction(std::unique_ptr<Instruction>(op));
}

// An opcode that has one or more operands, no result id, and no type
void Builder::createNoResultOp(Op opCode, const std::vector<Id>& operands)
{
    Instruction* op = new Instruction(opCode);
    op->reserveOperands(operands.size());
    for (auto id : operands) {
        op->addIdOperand(id);
    }
    addInstruction(std::unique_ptr<Instruction>(op));
}

// An opcode that has multiple operands, no result id, and no type
void Builder::createNoResultOp(Op opCode, const std::vector<IdImmediate>& operands)
{
    Instruction* op = new Instruction(opCode);
    op->reserveOperands(operands.size());
    for (auto it = operands.cbegin(); it != operands.cend(); ++it) {
        if (it->isId)
            op->addIdOperand(it->word);
        else
            op->addImmediateOperand(it->word);
    }
    addInstruction(std::unique_ptr<Instruction>(op));
}

void Builder::createControlBarrier(Scope execution, Scope memory, MemorySemanticsMask semantics)
{
    Instruction* op = new Instruction(Op::OpControlBarrier);
    op->reserveOperands(3);
    op->addIdOperand(makeUintConstant(execution));
    op->addIdOperand(makeUintConstant(memory));
    op->addIdOperand(makeUintConstant(semantics));
    addInstruction(std::unique_ptr<Instruction>(op));
}

void Builder::createMemoryBarrier(Scope executionScope, MemorySemanticsMask memorySemantics)
{
    Instruction* op = new Instruction(Op::OpMemoryBarrier);
    op->reserveOperands(2);
    op->addIdOperand(makeUintConstant((unsigned)executionScope));
    op->addIdOperand(makeUintConstant((unsigned)memorySemantics));
    addInstruction(std::unique_ptr<Instruction>(op));
}

// An opcode that has one operands, a result id, and a type
Id Builder::createUnaryOp(Op opCode, Id typeId, Id operand)
{
    // Generate code for spec constants if in spec constant operation
    // generation mode.
    if (generatingOpCodeForSpecConst) {
        return createSpecConstantOp(opCode, typeId, std::vector<Id>(1, operand), std::vector<Id>());
    }
    Instruction* op = new Instruction(getUniqueId(), typeId, opCode);
    op->addIdOperand(operand);
    addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

Id Builder::createBinOp(Op opCode, Id typeId, Id left, Id right)
{
    // Generate code for spec constants if in spec constant operation
    // generation mode.
    if (generatingOpCodeForSpecConst) {
        std::vector<Id> operands(2);
        operands[0] = left; operands[1] = right;
        return createSpecConstantOp(opCode, typeId, operands, std::vector<Id>());
    }
    Instruction* op = new Instruction(getUniqueId(), typeId, opCode);
    op->reserveOperands(2);
    op->addIdOperand(left);
    op->addIdOperand(right);
    addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

Id Builder::createTriOp(Op opCode, Id typeId, Id op1, Id op2, Id op3)
{
    // Generate code for spec constants if in spec constant operation
    // generation mode.
    if (generatingOpCodeForSpecConst) {
        std::vector<Id> operands(3);
        operands[0] = op1;
        operands[1] = op2;
        operands[2] = op3;
        return createSpecConstantOp(
            opCode, typeId, operands, std::vector<Id>());
    }
    Instruction* op = new Instruction(getUniqueId(), typeId, opCode);
    op->reserveOperands(3);
    op->addIdOperand(op1);
    op->addIdOperand(op2);
    op->addIdOperand(op3);
    addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

Id Builder::createOp(Op opCode, Id typeId, const std::vector<Id>& operands)
{
    Instruction* op = new Instruction(getUniqueId(), typeId, opCode);
    op->reserveOperands(operands.size());
    for (auto id : operands)
        op->addIdOperand(id);
    addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

Id Builder::createOp(Op opCode, Id typeId, const std::vector<IdImmediate>& operands)
{
    Instruction* op = new Instruction(getUniqueId(), typeId, opCode);
    op->reserveOperands(operands.size());
    for (auto it = operands.cbegin(); it != operands.cend(); ++it) {
        if (it->isId)
            op->addIdOperand(it->word);
        else
            op->addImmediateOperand(it->word);
    }
    addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

Id Builder::createSpecConstantOp(Op opCode, Id typeId, const std::vector<Id>& operands,
    const std::vector<unsigned>& literals)
{
    Instruction* op = new Instruction(getUniqueId(), typeId, Op::OpSpecConstantOp);
    op->reserveOperands(operands.size() + literals.size() + 1);
    op->addImmediateOperand((unsigned) opCode);
    for (auto it = operands.cbegin(); it != operands.cend(); ++it)
        op->addIdOperand(*it);
    for (auto it = literals.cbegin(); it != literals.cend(); ++it)
        op->addImmediateOperand(*it);
    module.mapInstruction(op);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(op));

    // OpSpecConstantOp's using 8 or 16 bit types require the associated capability
    if (containsType(typeId, Op::OpTypeInt, 8))
        addCapability(Capability::Int8);
    if (containsType(typeId, Op::OpTypeInt, 16))
        addCapability(Capability::Int16);
    if (containsType(typeId, Op::OpTypeFloat, 16))
        addCapability(Capability::Float16);

    return op->getResultId();
}

Id Builder::createFunctionCall(spv::Function* function, const std::vector<spv::Id>& args)
{
    Instruction* op = new Instruction(getUniqueId(), function->getReturnType(), Op::OpFunctionCall);
    op->reserveOperands(args.size() + 1);
    op->addIdOperand(function->getId());
    for (int a = 0; a < (int)args.size(); ++a)
        op->addIdOperand(args[a]);
    addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

// Comments in header
Id Builder::createRvalueSwizzle(Decoration precision, Id typeId, Id source, const std::vector<unsigned>& channels)
{
    if (channels.size() == 1)
        return setPrecision(createCompositeExtract(source, typeId, channels.front()), precision);

    if (generatingOpCodeForSpecConst) {
        std::vector<Id> operands(2);
        operands[0] = operands[1] = source;
        return setPrecision(createSpecConstantOp(Op::OpVectorShuffle, typeId, operands, channels), precision);
    }
    Instruction* swizzle = new Instruction(getUniqueId(), typeId, Op::OpVectorShuffle);
    assert(isVector(source));
    swizzle->reserveOperands(channels.size() + 2);
    swizzle->addIdOperand(source);
    swizzle->addIdOperand(source);
    for (int i = 0; i < (int)channels.size(); ++i)
        swizzle->addImmediateOperand(channels[i]);
    addInstruction(std::unique_ptr<Instruction>(swizzle));

    return setPrecision(swizzle->getResultId(), precision);
}

// Comments in header
Id Builder::createLvalueSwizzle(Id typeId, Id target, Id source, const std::vector<unsigned>& channels)
{
    if (channels.size() == 1 && getNumComponents(source) == 1)
        return createCompositeInsert(source, target, typeId, channels.front());

    Instruction* swizzle = new Instruction(getUniqueId(), typeId, Op::OpVectorShuffle);

    assert(isVector(target));
    swizzle->reserveOperands(2);
    swizzle->addIdOperand(target);

    assert(getNumComponents(source) == channels.size());
    assert(isVector(source));
    swizzle->addIdOperand(source);

    // Set up an identity shuffle from the base value to the result value
    unsigned int components[4];
    int numTargetComponents = getNumComponents(target);
    for (int i = 0; i < numTargetComponents; ++i)
        components[i] = i;

    // Punch in the l-value swizzle
    for (int i = 0; i < (int)channels.size(); ++i)
        components[channels[i]] = numTargetComponents + i;

    // finish the instruction with these components selectors
    swizzle->reserveOperands(numTargetComponents);
    for (int i = 0; i < numTargetComponents; ++i)
        swizzle->addImmediateOperand(components[i]);
    addInstruction(std::unique_ptr<Instruction>(swizzle));

    return swizzle->getResultId();
}

// Comments in header
void Builder::promoteScalar(Decoration precision, Id& left, Id& right)
{
    int direction = getNumComponents(right) - getNumComponents(left);

    if (direction > 0)
        left = smearScalar(precision, left, makeVectorType(getTypeId(left), getNumComponents(right)));
    else if (direction < 0)
        right = smearScalar(precision, right, makeVectorType(getTypeId(right), getNumComponents(left)));

    return;
}

// Comments in header
Id Builder::smearScalar(Decoration precision, Id scalar, Id vectorType)
{
    assert(getNumComponents(scalar) == 1);
    assert(getTypeId(scalar) == getScalarTypeId(vectorType));

    int numComponents = getNumTypeComponents(vectorType);
    if (numComponents == 1 && !isCooperativeVectorType(vectorType))
        return scalar;

    Instruction* smear = nullptr;
    if (generatingOpCodeForSpecConst) {
        auto members = std::vector<spv::Id>(numComponents, scalar);
        // Sometime even in spec-constant-op mode, the temporary vector created by
        // promoting a scalar might not be a spec constant. This should depend on
        // the scalar.
        // e.g.:
        //  const vec2 spec_const_result = a_spec_const_vec2 + a_front_end_const_scalar;
        // In such cases, the temporary vector created from a_front_end_const_scalar
        // is not a spec constant vector, even though the binary operation node is marked
        // as 'specConstant' and we are in spec-constant-op mode.
        auto result_id = makeCompositeConstant(vectorType, members, isSpecConstant(scalar));
        smear = module.getInstruction(result_id);
    } else {
        bool replicate = (useReplicatedComposites || isCooperativeVectorType(vectorType)) && (numComponents > 0);

        if (replicate) {
            numComponents = 1;
            addCapability(spv::Capability::ReplicatedCompositesEXT);
            addExtension(spv::E_SPV_EXT_replicated_composites);
        }

        Op opcode = replicate ? Op::OpCompositeConstructReplicateEXT : Op::OpCompositeConstruct;

        smear = new Instruction(getUniqueId(), vectorType, opcode);
        smear->reserveOperands(numComponents);
        for (int c = 0; c < numComponents; ++c)
            smear->addIdOperand(scalar);
        addInstruction(std::unique_ptr<Instruction>(smear));
    }

    return setPrecision(smear->getResultId(), precision);
}

// Comments in header
Id Builder::createBuiltinCall(Id resultType, Id builtins, int entryPoint, const std::vector<Id>& args)
{
    Instruction* inst = new Instruction(getUniqueId(), resultType, Op::OpExtInst);
    inst->reserveOperands(args.size() + 2);
    inst->addIdOperand(builtins);
    inst->addImmediateOperand(entryPoint);
    for (int arg = 0; arg < (int)args.size(); ++arg)
        inst->addIdOperand(args[arg]);

    addInstruction(std::unique_ptr<Instruction>(inst));

    return inst->getResultId();
}

// Accept all parameters needed to create a texture instruction.
// Create the correct instruction based on the inputs, and make the call.
Id Builder::createTextureCall(Decoration precision, Id resultType, bool sparse, bool fetch, bool proj, bool gather,
    bool noImplicitLod, const TextureParameters& parameters, ImageOperandsMask signExtensionMask)
{
    std::vector<Id> texArgs;

    //
    // Set up the fixed arguments
    //
    bool explicitLod = false;
    texArgs.push_back(parameters.sampler);
    texArgs.push_back(parameters.coords);
    if (parameters.Dref != NoResult)
        texArgs.push_back(parameters.Dref);
    if (parameters.component != NoResult)
        texArgs.push_back(parameters.component);

    if (parameters.granularity != NoResult)
        texArgs.push_back(parameters.granularity);
    if (parameters.coarse != NoResult)
        texArgs.push_back(parameters.coarse);

    //
    // Set up the optional arguments
    //
    size_t optArgNum = texArgs.size(); // the position of the mask for the optional arguments, if any.
    ImageOperandsMask mask = ImageOperandsMask::MaskNone; // the mask operand
    if (parameters.bias) {
        mask = (ImageOperandsMask)(mask | ImageOperandsMask::Bias);
        texArgs.push_back(parameters.bias);
    }
    if (parameters.lod) {
        mask = (ImageOperandsMask)(mask | ImageOperandsMask::Lod);
        texArgs.push_back(parameters.lod);
        explicitLod = true;
    } else if (parameters.gradX) {
        mask = (ImageOperandsMask)(mask | ImageOperandsMask::Grad);
        texArgs.push_back(parameters.gradX);
        texArgs.push_back(parameters.gradY);
        explicitLod = true;
    } else if (noImplicitLod && ! fetch && ! gather) {
        // have to explicitly use lod of 0 if not allowed to have them be implicit, and
        // we would otherwise be about to issue an implicit instruction
        mask = (ImageOperandsMask)(mask | ImageOperandsMask::Lod);
        texArgs.push_back(makeFloatConstant(0.0));
        explicitLod = true;
    }
    if (parameters.offset) {
        if (isConstant(parameters.offset))
            mask = (ImageOperandsMask)(mask | ImageOperandsMask::ConstOffset);
        else {
            addCapability(Capability::ImageGatherExtended);
            mask = (ImageOperandsMask)(mask | ImageOperandsMask::Offset);
        }
        texArgs.push_back(parameters.offset);
    }
    if (parameters.offsets) {
        if (!isConstant(parameters.offsets) && sourceLang == spv::SourceLanguage::GLSL) {
            mask = (ImageOperandsMask)(mask | ImageOperandsMask::Offsets);
        } else {
            addCapability(Capability::ImageGatherExtended);
            mask = (ImageOperandsMask)(mask | ImageOperandsMask::ConstOffsets);
        }
        texArgs.push_back(parameters.offsets);
    }
    if (parameters.sample) {
        mask = (ImageOperandsMask)(mask | ImageOperandsMask::Sample);
        texArgs.push_back(parameters.sample);
    }
    if (parameters.lodClamp) {
        // capability if this bit is used
        addCapability(Capability::MinLod);

        mask = (ImageOperandsMask)(mask | ImageOperandsMask::MinLod);
        texArgs.push_back(parameters.lodClamp);
    }
    if (parameters.nonprivate) {
        mask = mask | ImageOperandsMask::NonPrivateTexelKHR;
    }
    if (parameters.volatil) {
        mask = mask | ImageOperandsMask::VolatileTexelKHR;
    }
    if (parameters.nontemporal) {
        mask = mask | ImageOperandsMask::Nontemporal;
    }
    mask = mask | signExtensionMask;
    // insert the operand for the mask, if any bits were set.
    if (mask != ImageOperandsMask::MaskNone)
        texArgs.insert(texArgs.begin() + optArgNum, (Id)mask);

    //
    // Set up the instruction
    //
    Op opCode = Op::OpNop;  // All paths below need to set this
    if (fetch) {
        if (sparse)
            opCode = Op::OpImageSparseFetch;
        else
            opCode = Op::OpImageFetch;
    } else if (parameters.granularity && parameters.coarse) {
        opCode = Op::OpImageSampleFootprintNV;
    } else if (gather) {
        if (parameters.Dref)
            if (sparse)
                opCode = Op::OpImageSparseDrefGather;
            else
                opCode = Op::OpImageDrefGather;
        else
            if (sparse)
                opCode = Op::OpImageSparseGather;
            else
                opCode = Op::OpImageGather;
    } else if (explicitLod) {
        if (parameters.Dref) {
            if (proj)
                if (sparse)
                    opCode = Op::OpImageSparseSampleProjDrefExplicitLod;
                else
                    opCode = Op::OpImageSampleProjDrefExplicitLod;
            else
                if (sparse)
                    opCode = Op::OpImageSparseSampleDrefExplicitLod;
                else
                    opCode = Op::OpImageSampleDrefExplicitLod;
        } else {
            if (proj)
                if (sparse)
                    opCode = Op::OpImageSparseSampleProjExplicitLod;
                else
                    opCode = Op::OpImageSampleProjExplicitLod;
            else
                if (sparse)
                    opCode = Op::OpImageSparseSampleExplicitLod;
                else
                    opCode = Op::OpImageSampleExplicitLod;
        }
    } else {
        if (parameters.Dref) {
            if (proj)
                if (sparse)
                    opCode = Op::OpImageSparseSampleProjDrefImplicitLod;
                else
                    opCode = Op::OpImageSampleProjDrefImplicitLod;
            else
                if (sparse)
                    opCode = Op::OpImageSparseSampleDrefImplicitLod;
                else
                    opCode = Op::OpImageSampleDrefImplicitLod;
        } else {
            if (proj)
                if (sparse)
                    opCode = Op::OpImageSparseSampleProjImplicitLod;
                else
                    opCode = Op::OpImageSampleProjImplicitLod;
            else
                if (sparse)
                    opCode = Op::OpImageSparseSampleImplicitLod;
                else
                    opCode = Op::OpImageSampleImplicitLod;
        }
    }

    // See if the result type is expecting a smeared result.
    // This happens when a legacy shadow*() call is made, which
    // gets a vec4 back instead of a float.
    Id smearedType = resultType;
    if (! isScalarType(resultType)) {
        switch (opCode) {
        case Op::OpImageSampleDrefImplicitLod:
        case Op::OpImageSampleDrefExplicitLod:
        case Op::OpImageSampleProjDrefImplicitLod:
        case Op::OpImageSampleProjDrefExplicitLod:
            resultType = getScalarTypeId(resultType);
            break;
        default:
            break;
        }
    }

    Id typeId0 = 0;
    Id typeId1 = 0;

    if (sparse) {
        typeId0 = resultType;
        typeId1 = getDerefTypeId(parameters.texelOut);
        resultType = makeStructResultType(typeId0, typeId1);
    }

    // Build the SPIR-V instruction
    Instruction* textureInst = new Instruction(getUniqueId(), resultType, opCode);
    textureInst->reserveOperands(optArgNum + (texArgs.size() - (optArgNum + 1)));
    for (size_t op = 0; op < optArgNum; ++op)
        textureInst->addIdOperand(texArgs[op]);
    if (optArgNum < texArgs.size())
        textureInst->addImmediateOperand(texArgs[optArgNum]);
    for (size_t op = optArgNum + 1; op < texArgs.size(); ++op)
        textureInst->addIdOperand(texArgs[op]);
    setPrecision(textureInst->getResultId(), precision);
    addInstruction(std::unique_ptr<Instruction>(textureInst));

    Id resultId = textureInst->getResultId();

    if (sparse) {
        // set capability
        addCapability(Capability::SparseResidency);

        // Decode the return type that was a special structure
        createStore(createCompositeExtract(resultId, typeId1, 1), parameters.texelOut);
        resultId = createCompositeExtract(resultId, typeId0, 0);
        setPrecision(resultId, precision);
    } else {
        // When a smear is needed, do it, as per what was computed
        // above when resultType was changed to a scalar type.
        if (resultType != smearedType)
            resultId = smearScalar(precision, resultId, smearedType);
    }

    return resultId;
}

// Comments in header
Id Builder::createTextureQueryCall(Op opCode, const TextureParameters& parameters, bool isUnsignedResult)
{
    // Figure out the result type
    Id resultType = 0;
    switch (opCode) {
    case Op::OpImageQuerySize:
    case Op::OpImageQuerySizeLod:
    {
        int numComponents = 0;
        switch (getTypeDimensionality(getImageType(parameters.sampler))) {
        case Dim::Dim1D:
        case Dim::Buffer:
            numComponents = 1;
            break;
        case Dim::Dim2D:
        case Dim::Cube:
        case Dim::Rect:
        case Dim::SubpassData:
            numComponents = 2;
            break;
        case Dim::Dim3D:
            numComponents = 3;
            break;

        default:
            assert(0);
            break;
        }
        if (isArrayedImageType(getImageType(parameters.sampler)))
            ++numComponents;

        Id intType = isUnsignedResult ? makeUintType(32) : makeIntType(32);
        if (numComponents == 1)
            resultType = intType;
        else
            resultType = makeVectorType(intType, numComponents);

        break;
    }
    case Op::OpImageQueryLod:
        resultType = makeVectorType(getScalarTypeId(getTypeId(parameters.coords)), 2);
        break;
    case Op::OpImageQueryLevels:
    case Op::OpImageQuerySamples:
        resultType = isUnsignedResult ? makeUintType(32) : makeIntType(32);
        break;
    default:
        assert(0);
        break;
    }

    Instruction* query = new Instruction(getUniqueId(), resultType, opCode);
    query->addIdOperand(parameters.sampler);
    if (parameters.coords)
        query->addIdOperand(parameters.coords);
    if (parameters.lod)
        query->addIdOperand(parameters.lod);
    addInstruction(std::unique_ptr<Instruction>(query));
    addCapability(Capability::ImageQuery);

    return query->getResultId();
}

// External comments in header.
// Operates recursively to visit the composite's hierarchy.
Id Builder::createCompositeCompare(Decoration precision, Id value1, Id value2, bool equal)
{
    Id boolType = makeBoolType();
    Id valueType = getTypeId(value1);

    Id resultId = NoResult;

    int numConstituents = getNumTypeConstituents(valueType);

    // Scalars and Vectors

    if (isScalarType(valueType) || isVectorType(valueType)) {
        assert(valueType == getTypeId(value2));
        // These just need a single comparison, just have
        // to figure out what it is.
        Op op;
        switch (getMostBasicTypeClass(valueType)) {
        case Op::OpTypeFloat:
            op = equal ? Op::OpFOrdEqual : Op::OpFUnordNotEqual;
            break;
        case Op::OpTypeInt:
        default:
            op = equal ? Op::OpIEqual : Op::OpINotEqual;
            break;
        case Op::OpTypeBool:
            op = equal ? Op::OpLogicalEqual : Op::OpLogicalNotEqual;
            precision = NoPrecision;
            break;
        }

        if (isScalarType(valueType)) {
            // scalar
            resultId = createBinOp(op, boolType, value1, value2);
        } else {
            // vector
            resultId = createBinOp(op, makeVectorType(boolType, numConstituents), value1, value2);
            setPrecision(resultId, precision);
            // reduce vector compares...
            resultId = createUnaryOp(equal ? Op::OpAll : Op::OpAny, boolType, resultId);
        }

        return setPrecision(resultId, precision);
    }

    // Only structs, arrays, and matrices should be left.
    // They share in common the reduction operation across their constituents.
    assert(isAggregateType(valueType) || isMatrixType(valueType));

    // Compare each pair of constituents
    for (int constituent = 0; constituent < numConstituents; ++constituent) {
        std::vector<unsigned> indexes(1, constituent);
        Id constituentType1 = getContainedTypeId(getTypeId(value1), constituent);
        Id constituentType2 = getContainedTypeId(getTypeId(value2), constituent);
        Id constituent1 = createCompositeExtract(value1, constituentType1, indexes);
        Id constituent2 = createCompositeExtract(value2, constituentType2, indexes);

        Id subResultId = createCompositeCompare(precision, constituent1, constituent2, equal);

        if (constituent == 0)
            resultId = subResultId;
        else
            resultId = setPrecision(createBinOp(equal ? Op::OpLogicalAnd : Op::OpLogicalOr, boolType, resultId, subResultId),
                                    precision);
    }

    return resultId;
}

// OpCompositeConstruct
Id Builder::createCompositeConstruct(Id typeId, const std::vector<Id>& constituents)
{
    assert(isAggregateType(typeId) || (getNumTypeConstituents(typeId) > 1 &&
           getNumTypeConstituents(typeId) == constituents.size()) ||
           (isCooperativeVectorType(typeId) && constituents.size() == 1));

    if (generatingOpCodeForSpecConst) {
        // Sometime, even in spec-constant-op mode, the constant composite to be
        // constructed may not be a specialization constant.
        // e.g.:
        //  const mat2 m2 = mat2(a_spec_const, a_front_end_const, another_front_end_const, third_front_end_const);
        // The first column vector should be a spec constant one, as a_spec_const is a spec constant.
        // The second column vector should NOT be spec constant, as it does not contain any spec constants.
        // To handle such cases, we check the constituents of the constant vector to determine whether this
        // vector should be created as a spec constant.
        return makeCompositeConstant(typeId, constituents,
                                     std::any_of(constituents.begin(), constituents.end(),
                                                 [&](spv::Id id) { return isSpecConstant(id); }));
    }

    bool replicate = false;
    size_t numConstituents = constituents.size();

    if (useReplicatedComposites || isCooperativeVectorType(typeId)) {
        replicate = numConstituents > 0 &&
            std::equal(constituents.begin() + 1, constituents.end(), constituents.begin());
    }

    if (replicate) {
        numConstituents = 1;
        addCapability(spv::Capability::ReplicatedCompositesEXT);
        addExtension(spv::E_SPV_EXT_replicated_composites);
    }

    Op opcode = replicate ? Op::OpCompositeConstructReplicateEXT : Op::OpCompositeConstruct;

    Instruction* op = new Instruction(getUniqueId(), typeId, opcode);
    op->reserveOperands(constituents.size());
    for (size_t c = 0; c < numConstituents; ++c)
        op->addIdOperand(constituents[c]);
    addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

// coopmat conversion
Id Builder::createCooperativeMatrixConversion(Id typeId, Id source)
{
    Instruction* op = new Instruction(getUniqueId(), typeId, Op::OpCooperativeMatrixConvertNV);
    op->addIdOperand(source);
    addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

// coopmat reduce
Id Builder::createCooperativeMatrixReduce(Op opcode, Id typeId, Id source, unsigned int mask, Id func)
{
    Instruction* op = new Instruction(getUniqueId(), typeId, opcode);
    op->addIdOperand(source);
    op->addImmediateOperand(mask);
    op->addIdOperand(func);
    addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

// coopmat per-element operation
Id Builder::createCooperativeMatrixPerElementOp(Id typeId, const std::vector<Id>& operands)
{
    Instruction* op = new Instruction(getUniqueId(), typeId, spv::Op::OpCooperativeMatrixPerElementOpNV);
    // skip operand[0], which is where the result is stored
    for (uint32_t i = 1; i < operands.size(); ++i) {
        op->addIdOperand(operands[i]);
    }
    addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

// Vector or scalar constructor
Id Builder::createConstructor(Decoration precision, const std::vector<Id>& sources, Id resultTypeId)
{
    Id result = NoResult;
    unsigned int numTargetComponents = getNumTypeComponents(resultTypeId);
    unsigned int targetComponent = 0;

    // Special case: when calling a vector constructor with a single scalar
    // argument, smear the scalar
    if (sources.size() == 1 && isScalar(sources[0]) && (numTargetComponents > 1 || isCooperativeVectorType(resultTypeId)))
        return smearScalar(precision, sources[0], resultTypeId);

    // Special case: 2 vectors of equal size
    if (sources.size() == 1 && isVector(sources[0]) && numTargetComponents == getNumComponents(sources[0])) {
        assert(resultTypeId == getTypeId(sources[0]));
        return sources[0];
    }

    // accumulate the arguments for OpCompositeConstruct
    std::vector<Id> constituents;
    Id scalarTypeId = getScalarTypeId(resultTypeId);

    // lambda to store the result of visiting an argument component
    const auto latchResult = [&](Id comp) {
        if (numTargetComponents > 1)
            constituents.push_back(comp);
        else
            result = comp;
        ++targetComponent;
    };

    // lambda to visit a vector argument's components
    const auto accumulateVectorConstituents = [&](Id sourceArg) {
        unsigned int sourceSize = getNumComponents(sourceArg);
        unsigned int sourcesToUse = sourceSize;
        if (sourcesToUse + targetComponent > numTargetComponents)
            sourcesToUse = numTargetComponents - targetComponent;

        for (unsigned int s = 0; s < sourcesToUse; ++s) {
            std::vector<unsigned> swiz;
            swiz.push_back(s);
            latchResult(createRvalueSwizzle(precision, scalarTypeId, sourceArg, swiz));
        }
    };

    // lambda to visit a matrix argument's components
    const auto accumulateMatrixConstituents = [&](Id sourceArg) {
        unsigned int sourceSize = getNumColumns(sourceArg) * getNumRows(sourceArg);
        unsigned int sourcesToUse = sourceSize;
        if (sourcesToUse + targetComponent > numTargetComponents)
            sourcesToUse = numTargetComponents - targetComponent;

        unsigned int col = 0;
        unsigned int row = 0;
        for (unsigned int s = 0; s < sourcesToUse; ++s) {
            if (row >= getNumRows(sourceArg)) {
                row = 0;
                col++;
            }
            std::vector<Id> indexes;
            indexes.push_back(col);
            indexes.push_back(row);
            latchResult(createCompositeExtract(sourceArg, scalarTypeId, indexes));
            row++;
        }
    };

    // Go through the source arguments, each one could have either
    // a single or multiple components to contribute.
    for (unsigned int i = 0; i < sources.size(); ++i) {

        if (isScalar(sources[i]) || isPointer(sources[i]))
            latchResult(sources[i]);
        else if (isVector(sources[i]) || isCooperativeVector(sources[i]))
            accumulateVectorConstituents(sources[i]);
        else if (isMatrix(sources[i]))
            accumulateMatrixConstituents(sources[i]);
        else
            assert(0);

        if (targetComponent >= numTargetComponents)
            break;
    }

    // If the result is a vector, make it from the gathered constituents.
    if (constituents.size() > 0) {
        result = createCompositeConstruct(resultTypeId, constituents);
        return setPrecision(result, precision);
    } else {
      // Precision was set when generating this component.
      return result;
    }
}

// Comments in header
Id Builder::createMatrixConstructor(Decoration precision, const std::vector<Id>& sources, Id resultTypeId)
{
    Id componentTypeId = getScalarTypeId(resultTypeId);
    unsigned int numCols = getTypeNumColumns(resultTypeId);
    unsigned int numRows = getTypeNumRows(resultTypeId);

    Instruction* instr = module.getInstruction(componentTypeId);
    const unsigned bitCount = instr->getImmediateOperand(0);

    // Optimize matrix constructed from a bigger matrix
    if (isMatrix(sources[0]) && getNumColumns(sources[0]) >= numCols && getNumRows(sources[0]) >= numRows) {
        // To truncate the matrix to a smaller number of rows/columns, we need to:
        // 1. For each column, extract the column and truncate it to the required size using shuffle
        // 2. Assemble the resulting matrix from all columns
        Id matrix = sources[0];
        Id columnTypeId = getContainedTypeId(resultTypeId);
        Id sourceColumnTypeId = getContainedTypeId(getTypeId(matrix));

        std::vector<unsigned> channels;
        for (unsigned int row = 0; row < numRows; ++row)
            channels.push_back(row);

        std::vector<Id> matrixColumns;
        for (unsigned int col = 0; col < numCols; ++col) {
            std::vector<unsigned> indexes;
            indexes.push_back(col);
            Id colv = createCompositeExtract(matrix, sourceColumnTypeId, indexes);
            setPrecision(colv, precision);

            if (numRows != getNumRows(matrix)) {
                matrixColumns.push_back(createRvalueSwizzle(precision, columnTypeId, colv, channels));
            } else {
                matrixColumns.push_back(colv);
            }
        }

        return setPrecision(createCompositeConstruct(resultTypeId, matrixColumns), precision);
    }

    // Detect a matrix being constructed from a repeated vector of the correct size.
    // Create the composite directly from it.
    if (sources.size() == numCols && isVector(sources[0]) && getNumComponents(sources[0]) == numRows &&
        std::equal(sources.begin() + 1, sources.end(), sources.begin())) {
        return setPrecision(createCompositeConstruct(resultTypeId, sources), precision);
    }

    // Otherwise, will use a two step process
    // 1. make a compile-time 2D array of values
    // 2. construct a matrix from that array

    // Step 1.

    // initialize the array to the identity matrix
    Id ids[maxMatrixSize][maxMatrixSize];
    Id  one = (bitCount == 64 ? makeDoubleConstant(1.0) : makeFloatConstant(1.0));
    Id zero = (bitCount == 64 ? makeDoubleConstant(0.0) : makeFloatConstant(0.0));
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            if (col == row)
                ids[col][row] = one;
            else
                ids[col][row] = zero;
        }
    }

    // modify components as dictated by the arguments
    if (sources.size() == 1 && isScalar(sources[0])) {
        // a single scalar; resets the diagonals
        for (int col = 0; col < 4; ++col)
            ids[col][col] = sources[0];
    } else if (isMatrix(sources[0])) {
        // constructing from another matrix; copy over the parts that exist in both the argument and constructee
        Id matrix = sources[0];
        unsigned int minCols = std::min(numCols, getNumColumns(matrix));
        unsigned int minRows = std::min(numRows, getNumRows(matrix));
        for (unsigned int col = 0; col < minCols; ++col) {
            std::vector<unsigned> indexes;
            indexes.push_back(col);
            for (unsigned int row = 0; row < minRows; ++row) {
                indexes.push_back(row);
                ids[col][row] = createCompositeExtract(matrix, componentTypeId, indexes);
                indexes.pop_back();
                setPrecision(ids[col][row], precision);
            }
        }
    } else {
        // fill in the matrix in column-major order with whatever argument components are available
        unsigned int row = 0;
        unsigned int col = 0;

        for (unsigned int arg = 0; arg < sources.size() && col < numCols; ++arg) {
            Id argComp = sources[arg];
            for (unsigned int comp = 0; comp < getNumComponents(sources[arg]); ++comp) {
                if (getNumComponents(sources[arg]) > 1) {
                    argComp = createCompositeExtract(sources[arg], componentTypeId, comp);
                    setPrecision(argComp, precision);
                }
                ids[col][row++] = argComp;
                if (row == numRows) {
                    row = 0;
                    col++;
                }
                if (col == numCols) {
                    // If more components are provided than fit the matrix, discard the rest.
                    break;
                }
            }
        }
    }

    // Step 2:  Construct a matrix from that array.
    // First make the column vectors, then make the matrix.

    // make the column vectors
    Id columnTypeId = getContainedTypeId(resultTypeId);
    std::vector<Id> matrixColumns;
    for (unsigned int col = 0; col < numCols; ++col) {
        std::vector<Id> vectorComponents;
        for (unsigned int row = 0; row < numRows; ++row)
            vectorComponents.push_back(ids[col][row]);
        Id column = createCompositeConstruct(columnTypeId, vectorComponents);
        setPrecision(column, precision);
        matrixColumns.push_back(column);
    }

    // make the matrix
    return setPrecision(createCompositeConstruct(resultTypeId, matrixColumns), precision);
}

// Comments in header
Builder::If::If(Id cond, SelectionControlMask ctrl, Builder& gb) :
    builder(gb),
    condition(cond),
    control(ctrl),
    elseBlock(nullptr)
{
    function = &builder.getBuildPoint()->getParent();

    // make the blocks, but only put the then-block into the function,
    // the else-block and merge-block will be added later, in order, after
    // earlier code is emitted
    thenBlock = new Block(builder.getUniqueId(), *function);
    mergeBlock = new Block(builder.getUniqueId(), *function);

    // Save the current block, so that we can add in the flow control split when
    // makeEndIf is called.
    headerBlock = builder.getBuildPoint();
    builder.createSelectionMerge(mergeBlock, control);

    function->addBlock(thenBlock);
    builder.setBuildPoint(thenBlock);
}

// Comments in header
void Builder::If::makeBeginElse()
{
    // Close out the "then" by having it jump to the mergeBlock
    builder.createBranch(true, mergeBlock);

    // Make the first else block and add it to the function
    elseBlock = new Block(builder.getUniqueId(), *function);
    function->addBlock(elseBlock);

    // Start building the else block
    builder.setBuildPoint(elseBlock);
}

// Comments in header
void Builder::If::makeEndIf()
{
    // jump to the merge block
    builder.createBranch(true, mergeBlock);

    // Go back to the headerBlock and make the flow control split
    builder.setBuildPoint(headerBlock);
    if (elseBlock)
        builder.createConditionalBranch(condition, thenBlock, elseBlock);
    else
        builder.createConditionalBranch(condition, thenBlock, mergeBlock);

    // add the merge block to the function
    function->addBlock(mergeBlock);
    builder.setBuildPoint(mergeBlock);
}

// Comments in header
void Builder::makeSwitch(Id selector, SelectionControlMask control, int numSegments, const std::vector<int>& caseValues,
                         const std::vector<int>& valueIndexToSegment, int defaultSegment,
                         std::vector<Block*>& segmentBlocks)
{
    Function& function = buildPoint->getParent();

    // make all the blocks
    for (int s = 0; s < numSegments; ++s)
        segmentBlocks.push_back(new Block(getUniqueId(), function));

    Block* mergeBlock = new Block(getUniqueId(), function);

    // make and insert the switch's selection-merge instruction
    createSelectionMerge(mergeBlock, control);

    // make the switch instruction
    Instruction* switchInst = new Instruction(NoResult, NoType, Op::OpSwitch);
    switchInst->reserveOperands((caseValues.size() * 2) + 2);
    switchInst->addIdOperand(selector);
    auto defaultOrMerge = (defaultSegment >= 0) ? segmentBlocks[defaultSegment] : mergeBlock;
    switchInst->addIdOperand(defaultOrMerge->getId());
    defaultOrMerge->addPredecessor(buildPoint);
    for (int i = 0; i < (int)caseValues.size(); ++i) {
        switchInst->addImmediateOperand(caseValues[i]);
        switchInst->addIdOperand(segmentBlocks[valueIndexToSegment[i]]->getId());
        segmentBlocks[valueIndexToSegment[i]]->addPredecessor(buildPoint);
    }
    addInstruction(std::unique_ptr<Instruction>(switchInst));

    // push the merge block
    switchMerges.push(mergeBlock);
}

// Comments in header
void Builder::addSwitchBreak(bool implicit)
{
    // branch to the top of the merge block stack
    createBranch(implicit, switchMerges.top());
    createAndSetNoPredecessorBlock("post-switch-break");
}

// Comments in header
void Builder::nextSwitchSegment(std::vector<Block*>& segmentBlock, int nextSegment)
{
    int lastSegment = nextSegment - 1;
    if (lastSegment >= 0) {
        // Close out previous segment by jumping, if necessary, to next segment
        if (! buildPoint->isTerminated())
            createBranch(true, segmentBlock[nextSegment]);
    }
    Block* block = segmentBlock[nextSegment];
    block->getParent().addBlock(block);
    setBuildPoint(block);
}

// Comments in header
void Builder::endSwitch(std::vector<Block*>& /*segmentBlock*/)
{
    // Close out previous segment by jumping, if necessary, to next segment
    if (! buildPoint->isTerminated())
        addSwitchBreak(true);

    switchMerges.top()->getParent().addBlock(switchMerges.top());
    setBuildPoint(switchMerges.top());

    switchMerges.pop();
}

Block& Builder::makeNewBlock()
{
    Function& function = buildPoint->getParent();
    auto block = new Block(getUniqueId(), function);
    function.addBlock(block);
    return *block;
}

Builder::LoopBlocks& Builder::makeNewLoop()
{
    // This verbosity is needed to simultaneously get the same behavior
    // everywhere (id's in the same order), have a syntax that works
    // across lots of versions of C++, have no warnings from pedantic
    // compilation modes, and leave the rest of the code alone.
    Block& head            = makeNewBlock();
    Block& body            = makeNewBlock();
    Block& merge           = makeNewBlock();
    Block& continue_target = makeNewBlock();
    LoopBlocks blocks(head, body, merge, continue_target);
    loops.push(blocks);
    return loops.top();
}

void Builder::createLoopContinue()
{
    createBranch(false, &loops.top().continue_target);
    // Set up a block for dead code.
    createAndSetNoPredecessorBlock("post-loop-continue");
}

void Builder::createLoopExit()
{
    createBranch(false, &loops.top().merge);
    // Set up a block for dead code.
    createAndSetNoPredecessorBlock("post-loop-break");
}

void Builder::closeLoop()
{
    loops.pop();
}

void Builder::clearAccessChain()
{
    accessChain.base = NoResult;
    accessChain.indexChain.clear();
    accessChain.instr = NoResult;
    accessChain.swizzle.clear();
    accessChain.component = NoResult;
    accessChain.preSwizzleBaseType = NoType;
    accessChain.isRValue = false;
    accessChain.coherentFlags.clear();
    accessChain.alignment = 0;
}

// Comments in header
void Builder::accessChainPushSwizzle(std::vector<unsigned>& swizzle, Id preSwizzleBaseType,
    AccessChain::CoherentFlags coherentFlags, unsigned int alignment)
{
    accessChain.coherentFlags |= coherentFlags;
    accessChain.alignment |= alignment;

    // swizzles can be stacked in GLSL, but simplified to a single
    // one here; the base type doesn't change
    if (accessChain.preSwizzleBaseType == NoType)
        accessChain.preSwizzleBaseType = preSwizzleBaseType;

    // if needed, propagate the swizzle for the current access chain
    if (accessChain.swizzle.size() > 0) {
        std::vector<unsigned> oldSwizzle = accessChain.swizzle;
        accessChain.swizzle.resize(0);
        for (unsigned int i = 0; i < swizzle.size(); ++i) {
            assert(swizzle[i] < oldSwizzle.size());
            accessChain.swizzle.push_back(oldSwizzle[swizzle[i]]);
        }
    } else
        accessChain.swizzle = swizzle;

    // determine if we need to track this swizzle anymore
    simplifyAccessChainSwizzle();
}

// Comments in header
void Builder::accessChainStore(Id rvalue, Decoration nonUniform, spv::MemoryAccessMask memoryAccess, spv::Scope scope, unsigned int alignment)
{
    assert(accessChain.isRValue == false);

    transferAccessChainSwizzle(true);

    // If a swizzle exists and is not full and is not dynamic, then the swizzle will be broken into individual stores.
    if (accessChain.swizzle.size() > 0 &&
        getNumTypeComponents(getResultingAccessChainType()) != accessChain.swizzle.size() &&
        accessChain.component == NoResult) {
        for (unsigned int i = 0; i < accessChain.swizzle.size(); ++i) {
            accessChain.indexChain.push_back(makeUintConstant(accessChain.swizzle[i]));
            accessChain.instr = NoResult;

            Id base = collapseAccessChain();
            addDecoration(base, nonUniform);

            accessChain.indexChain.pop_back();
            accessChain.instr = NoResult;

            // dynamic component should be gone
            assert(accessChain.component == NoResult);

            Id source = createCompositeExtract(rvalue, getContainedTypeId(getTypeId(rvalue)), i);

            // take LSB of alignment
            alignment = alignment & ~(alignment & (alignment-1));
            if (getStorageClass(base) == StorageClass::PhysicalStorageBufferEXT) {
                memoryAccess = (spv::MemoryAccessMask)(memoryAccess | spv::MemoryAccessMask::Aligned);
            }

            createStore(source, base, memoryAccess, scope, alignment);
        }
    }
    else {
        Id base = collapseAccessChain();
        addDecoration(base, nonUniform);

        Id source = rvalue;

        // dynamic component should be gone
        assert(accessChain.component == NoResult);

        // If swizzle still exists, it may be out-of-order, we must load the target vector,
        // extract and insert elements to perform writeMask and/or swizzle.
        if (accessChain.swizzle.size() > 0) {
            Id tempBaseId = createLoad(base, spv::NoPrecision);
            source = createLvalueSwizzle(getTypeId(tempBaseId), tempBaseId, source, accessChain.swizzle);
        }

        // take LSB of alignment
        alignment = alignment & ~(alignment & (alignment-1));
        if (getStorageClass(base) == StorageClass::PhysicalStorageBufferEXT) {
            memoryAccess = (spv::MemoryAccessMask)(memoryAccess | spv::MemoryAccessMask::Aligned);
        }

        createStore(source, base, memoryAccess, scope, alignment);
    }
}

// Comments in header
Id Builder::accessChainLoad(Decoration precision, Decoration l_nonUniform,
    Decoration r_nonUniform, Id resultType, spv::MemoryAccessMask memoryAccess,
    spv::Scope scope, unsigned int alignment)
{
    Id id;

    if (accessChain.isRValue) {
        // transfer access chain, but try to stay in registers
        transferAccessChainSwizzle(false);
        if (accessChain.indexChain.size() > 0) {
            Id swizzleBase = accessChain.preSwizzleBaseType != NoType ? accessChain.preSwizzleBaseType : resultType;

            // if all the accesses are constants, we can use OpCompositeExtract
            std::vector<unsigned> indexes;
            bool constant = true;
            for (int i = 0; i < (int)accessChain.indexChain.size(); ++i) {
                if (isConstantScalar(accessChain.indexChain[i]))
                    indexes.push_back(getConstantScalar(accessChain.indexChain[i]));
                else {
                    constant = false;
                    break;
                }
            }

            if (constant) {
                id = createCompositeExtract(accessChain.base, swizzleBase, indexes);
                setPrecision(id, precision);
            } else if (isCooperativeVector(accessChain.base)) {
                assert(accessChain.indexChain.size() == 1);
                id = createVectorExtractDynamic(accessChain.base, resultType, accessChain.indexChain[0]);
            } else {
                Id lValue = NoResult;
                if (spvVersion >= Spv_1_4 && isValidInitializer(accessChain.base)) {
                    // make a new function variable for this r-value, using an initializer,
                    // and mark it as NonWritable so that downstream it can be detected as a lookup
                    // table
                    lValue = createVariable(NoPrecision, StorageClass::Function, getTypeId(accessChain.base),
                        "indexable", accessChain.base);
                    addDecoration(lValue, Decoration::NonWritable);
                } else {
                    lValue = createVariable(NoPrecision, StorageClass::Function, getTypeId(accessChain.base),
                        "indexable");
                    // store into it
                    createStore(accessChain.base, lValue);
                }
                // move base to the new variable
                accessChain.base = lValue;
                accessChain.isRValue = false;

                // load through the access chain
                id = createLoad(collapseAccessChain(), precision);
            }
        } else
            id = accessChain.base;  // no precision, it was set when this was defined
    } else {
        transferAccessChainSwizzle(true);

        // take LSB of alignment
        alignment = alignment & ~(alignment & (alignment-1));
        if (getStorageClass(accessChain.base) == StorageClass::PhysicalStorageBufferEXT) {
            memoryAccess = (spv::MemoryAccessMask)(memoryAccess | spv::MemoryAccessMask::Aligned);
        }

        // load through the access chain
        id = collapseAccessChain();
        // Apply nonuniform both to the access chain and the loaded value.
        // Buffer accesses need the access chain decorated, and this is where
        // loaded image types get decorated. TODO: This should maybe move to
        // createImageTextureFunctionCall.
        addDecoration(id, l_nonUniform);
        id = createLoad(id, precision, memoryAccess, scope, alignment);
        addDecoration(id, r_nonUniform);
    }

    // Done, unless there are swizzles to do
    if (accessChain.swizzle.size() == 0 && accessChain.component == NoResult)
        return id;

    // Do remaining swizzling

    // Do the basic swizzle
    if (accessChain.swizzle.size() > 0) {
        Id swizzledType = getScalarTypeId(getTypeId(id));
        if (accessChain.swizzle.size() > 1)
            swizzledType = makeVectorType(swizzledType, (int)accessChain.swizzle.size());
        id = createRvalueSwizzle(precision, swizzledType, id, accessChain.swizzle);
    }

    // Do the dynamic component
    if (accessChain.component != NoResult)
        id = setPrecision(createVectorExtractDynamic(id, resultType, accessChain.component), precision);

    addDecoration(id, r_nonUniform);
    return id;
}

Id Builder::accessChainGetLValue()
{
    assert(accessChain.isRValue == false);

    transferAccessChainSwizzle(true);
    Id lvalue = collapseAccessChain();

    // If swizzle exists, it is out-of-order or not full, we must load the target vector,
    // extract and insert elements to perform writeMask and/or swizzle.  This does not
    // go with getting a direct l-value pointer.
    assert(accessChain.swizzle.size() == 0);
    assert(accessChain.component == NoResult);

    return lvalue;
}

// comment in header
Id Builder::accessChainGetInferredType()
{
    // anything to operate on?
    if (accessChain.base == NoResult)
        return NoType;
    Id type = getTypeId(accessChain.base);

    // do initial dereference
    if (! accessChain.isRValue)
        type = getContainedTypeId(type);

    // dereference each index
    for (auto it = accessChain.indexChain.cbegin(); it != accessChain.indexChain.cend(); ++it) {
        if (isStructType(type))
            type = getContainedTypeId(type, getConstantScalar(*it));
        else
            type = getContainedTypeId(type);
    }

    // dereference swizzle
    if (accessChain.swizzle.size() == 1)
        type = getContainedTypeId(type);
    else if (accessChain.swizzle.size() > 1)
        type = makeVectorType(getContainedTypeId(type), (int)accessChain.swizzle.size());

    // dereference component selection
    if (accessChain.component)
        type = getContainedTypeId(type);

    return type;
}

void Builder::dump(std::vector<unsigned int>& out) const
{
    // Header, before first instructions:
    out.push_back(MagicNumber);
    out.push_back(spvVersion);
    out.push_back(builderNumber);
    out.push_back(uniqueId + 1);
    out.push_back(0);

    // Capabilities
    for (auto it = capabilities.cbegin(); it != capabilities.cend(); ++it) {
        Instruction capInst(0, 0, Op::OpCapability);
        capInst.addImmediateOperand(*it);
        capInst.dump(out);
    }

    for (auto it = extensions.cbegin(); it != extensions.cend(); ++it) {
        Instruction extInst(0, 0, Op::OpExtension);
        extInst.addStringOperand(it->c_str());
        extInst.dump(out);
    }

    dumpInstructions(out, imports);
    Instruction memInst(0, 0, Op::OpMemoryModel);
    memInst.addImmediateOperand(addressModel);
    memInst.addImmediateOperand(memoryModel);
    memInst.dump(out);

    // Instructions saved up while building:
    dumpInstructions(out, entryPoints);
    dumpInstructions(out, executionModes);

    // Debug instructions
    dumpInstructions(out, strings);
    dumpSourceInstructions(out);
    for (int e = 0; e < (int)sourceExtensions.size(); ++e) {
        Instruction sourceExtInst(0, 0, Op::OpSourceExtension);
        sourceExtInst.addStringOperand(sourceExtensions[e]);
        sourceExtInst.dump(out);
    }
    dumpInstructions(out, names);
    dumpModuleProcesses(out);

    // Annotation instructions
    dumpInstructions(out, decorations);

    dumpInstructions(out, constantsTypesGlobals);
    dumpInstructions(out, externals);

    // The functions
    module.dump(out);
}

//
// Protected methods.
//

// Turn the described access chain in 'accessChain' into an instruction(s)
// computing its address.  This *cannot* include complex swizzles, which must
// be handled after this is called.
//
// Can generate code.
Id Builder::collapseAccessChain()
{
    assert(accessChain.isRValue == false);

    // did we already emit an access chain for this?
    if (accessChain.instr != NoResult)
        return accessChain.instr;

    // If we have a dynamic component, we can still transfer
    // that into a final operand to the access chain.  We need to remap the
    // dynamic component through the swizzle to get a new dynamic component to
    // update.
    //
    // This was not done in transferAccessChainSwizzle() because it might
    // generate code.
    remapDynamicSwizzle();
    if (accessChain.component != NoResult) {
        // transfer the dynamic component to the access chain
        accessChain.indexChain.push_back(accessChain.component);
        accessChain.component = NoResult;
    }

    // note that non-trivial swizzling is left pending

    // do we have an access chain?
    if (accessChain.indexChain.size() == 0)
        return accessChain.base;

    // emit the access chain
    StorageClass storageClass = (StorageClass)module.getStorageClass(getTypeId(accessChain.base));
    accessChain.instr = createAccessChain(storageClass, accessChain.base, accessChain.indexChain);

    return accessChain.instr;
}

// For a dynamic component selection of a swizzle.
//
// Turn the swizzle and dynamic component into just a dynamic component.
//
// Generates code.
void Builder::remapDynamicSwizzle()
{
    // do we have a swizzle to remap a dynamic component through?
    if (accessChain.component != NoResult && accessChain.swizzle.size() > 1) {
        // build a vector of the swizzle for the component to map into
        std::vector<Id> components;
        for (int c = 0; c < (int)accessChain.swizzle.size(); ++c)
            components.push_back(makeUintConstant(accessChain.swizzle[c]));
        Id mapType = makeVectorType(makeUintType(32), (int)accessChain.swizzle.size());
        Id map = makeCompositeConstant(mapType, components);

        // use it
        accessChain.component = createVectorExtractDynamic(map, makeUintType(32), accessChain.component);
        accessChain.swizzle.clear();
    }
}

// clear out swizzle if it is redundant, that is reselecting the same components
// that would be present without the swizzle.
void Builder::simplifyAccessChainSwizzle()
{
    // If the swizzle has fewer components than the vector, it is subsetting, and must stay
    // to preserve that fact.
    if (getNumTypeComponents(accessChain.preSwizzleBaseType) > accessChain.swizzle.size())
        return;

    // if components are out of order, it is a swizzle
    for (unsigned int i = 0; i < accessChain.swizzle.size(); ++i) {
        if (i != accessChain.swizzle[i])
            return;
    }

    // otherwise, there is no need to track this swizzle
    accessChain.swizzle.clear();
    if (accessChain.component == NoResult)
        accessChain.preSwizzleBaseType = NoType;
}

// To the extent any swizzling can become part of the chain
// of accesses instead of a post operation, make it so.
// If 'dynamic' is true, include transferring the dynamic component,
// otherwise, leave it pending.
//
// Does not generate code. just updates the access chain.
void Builder::transferAccessChainSwizzle(bool dynamic)
{
    // non existent?
    if (accessChain.swizzle.size() == 0 && accessChain.component == NoResult)
        return;

    // too complex?
    // (this requires either a swizzle, or generating code for a dynamic component)
    if (accessChain.swizzle.size() > 1)
        return;

    // single component, either in the swizzle and/or dynamic component
    if (accessChain.swizzle.size() == 1) {
        assert(accessChain.component == NoResult);
        // handle static component selection
        accessChain.indexChain.push_back(makeUintConstant(accessChain.swizzle.front()));
        accessChain.swizzle.clear();
        accessChain.preSwizzleBaseType = NoType;
    } else if (dynamic && accessChain.component != NoResult) {
        assert(accessChain.swizzle.size() == 0);
        // handle dynamic component
        accessChain.indexChain.push_back(accessChain.component);
        accessChain.preSwizzleBaseType = NoType;
        accessChain.component = NoResult;
    }
}

// Utility method for creating a new block and setting the insert point to
// be in it. This is useful for flow-control operations that need a "dummy"
// block proceeding them (e.g. instructions after a discard, etc).
void Builder::createAndSetNoPredecessorBlock(const char* /*name*/)
{
    Block* block = new Block(getUniqueId(), buildPoint->getParent());
    block->setUnreachable();
    buildPoint->getParent().addBlock(block);
    setBuildPoint(block);

    // if (name)
    //    addName(block->getId(), name);
}

// Comments in header
void Builder::createBranch(bool implicit, Block* block)
{
    Instruction* branch = new Instruction(Op::OpBranch);
    branch->addIdOperand(block->getId());
    if (implicit) {
        addInstructionNoDebugInfo(std::unique_ptr<Instruction>(branch));
    }
    else {
        addInstruction(std::unique_ptr<Instruction>(branch));
    }
    block->addPredecessor(buildPoint);
}

void Builder::createSelectionMerge(Block* mergeBlock, SelectionControlMask control)
{
    Instruction* merge = new Instruction(Op::OpSelectionMerge);
    merge->reserveOperands(2);
    merge->addIdOperand(mergeBlock->getId());
    merge->addImmediateOperand(control);
    addInstruction(std::unique_ptr<Instruction>(merge));
}

void Builder::createLoopMerge(Block* mergeBlock, Block* continueBlock, LoopControlMask control,
                              const std::vector<unsigned int>& operands)
{
    Instruction* merge = new Instruction(Op::OpLoopMerge);
    merge->reserveOperands(operands.size() + 3);
    merge->addIdOperand(mergeBlock->getId());
    merge->addIdOperand(continueBlock->getId());
    merge->addImmediateOperand(control);
    for (int op = 0; op < (int)operands.size(); ++op)
        merge->addImmediateOperand(operands[op]);
    addInstruction(std::unique_ptr<Instruction>(merge));
}

void Builder::createConditionalBranch(Id condition, Block* thenBlock, Block* elseBlock)
{
    Instruction* branch = new Instruction(Op::OpBranchConditional);
    branch->reserveOperands(3);
    branch->addIdOperand(condition);
    branch->addIdOperand(thenBlock->getId());
    branch->addIdOperand(elseBlock->getId());

    // A conditional branch is always attached to a condition expression
    addInstructionNoDebugInfo(std::unique_ptr<Instruction>(branch));

    thenBlock->addPredecessor(buildPoint);
    elseBlock->addPredecessor(buildPoint);
}

// OpSource
// [OpSourceContinued]
// ...
void Builder::dumpSourceInstructions(const spv::Id fileId, const std::string& text,
                                     std::vector<unsigned int>& out) const
{
    const int maxWordCount = 0xFFFF;
    const int opSourceWordCount = 4;
    const int nonNullBytesPerInstruction = 4 * (maxWordCount - opSourceWordCount) - 1;

    if (sourceLang != SourceLanguage::Unknown) {
        // OpSource Language Version File Source
        Instruction sourceInst(NoResult, NoType, Op::OpSource);
        sourceInst.reserveOperands(3);
        sourceInst.addImmediateOperand(sourceLang);
        sourceInst.addImmediateOperand(sourceVersion);
        // File operand
        if (fileId != NoResult) {
            sourceInst.addIdOperand(fileId);
            // Source operand
            if (text.size() > 0) {
                int nextByte = 0;
                std::string subString;
                while ((int)text.size() - nextByte > 0) {
                    subString = text.substr(nextByte, nonNullBytesPerInstruction);
                    if (nextByte == 0) {
                        // OpSource
                        sourceInst.addStringOperand(subString.c_str());
                        sourceInst.dump(out);
                    } else {
                        // OpSourcContinued
                        Instruction sourceContinuedInst(Op::OpSourceContinued);
                        sourceContinuedInst.addStringOperand(subString.c_str());
                        sourceContinuedInst.dump(out);
                    }
                    nextByte += nonNullBytesPerInstruction;
                }
            } else
                sourceInst.dump(out);
        } else
            sourceInst.dump(out);
    }
}

// Dump an OpSource[Continued] sequence for the source and every include file
void Builder::dumpSourceInstructions(std::vector<unsigned int>& out) const
{
    if (emitNonSemanticShaderDebugInfo) return;
    dumpSourceInstructions(mainFileId, sourceText, out);
    for (auto iItr = includeFiles.begin(); iItr != includeFiles.end(); ++iItr)
        dumpSourceInstructions(iItr->first, *iItr->second, out);
}

template <class Range> void Builder::dumpInstructions(std::vector<unsigned int>& out, const Range& instructions) const
{
    for (const auto& inst : instructions) {
        inst->dump(out);
    }
}

void Builder::dumpModuleProcesses(std::vector<unsigned int>& out) const
{
    for (int i = 0; i < (int)moduleProcesses.size(); ++i) {
        Instruction moduleProcessed(Op::OpModuleProcessed);
        moduleProcessed.addStringOperand(moduleProcesses[i]);
        moduleProcessed.dump(out);
    }
}

bool Builder::DecorationInstructionLessThan::operator()(const std::unique_ptr<Instruction>& lhs,
                                                        const std::unique_ptr<Instruction>& rhs) const
{
    // Order by the id to which the decoration applies first. This is more intuitive.
    assert(lhs->isIdOperand(0) && rhs->isIdOperand(0));
    if (lhs->getIdOperand(0) != rhs->getIdOperand(0)) {
        return lhs->getIdOperand(0) < rhs->getIdOperand(0);
    }

    if (lhs->getOpCode() != rhs->getOpCode())
        return lhs->getOpCode() < rhs->getOpCode();

    // Now compare the operands.
    int minSize = std::min(lhs->getNumOperands(), rhs->getNumOperands());
    for (int i = 1; i < minSize; ++i) {
        if (lhs->isIdOperand(i) != rhs->isIdOperand(i)) {
            return lhs->isIdOperand(i) < rhs->isIdOperand(i);
        }

        if (lhs->isIdOperand(i)) {
            if (lhs->getIdOperand(i) != rhs->getIdOperand(i)) {
                return lhs->getIdOperand(i) < rhs->getIdOperand(i);
            }
        } else {
            if (lhs->getImmediateOperand(i) != rhs->getImmediateOperand(i)) {
                return lhs->getImmediateOperand(i) < rhs->getImmediateOperand(i);
            }
        }
    }

    if (lhs->getNumOperands() != rhs->getNumOperands())
        return lhs->getNumOperands() < rhs->getNumOperands();

    // In this case they are equal.
    return false;
}
} // end spv namespace
