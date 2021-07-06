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

#ifndef GLSLANG_WEB
#include "hex_float.h"
#endif

#ifndef _WIN32
    #include <cstdio>
#endif

namespace spv {

Builder::Builder(unsigned int spvVersion, unsigned int magicNumber, SpvBuildLogger* buildLogger) :
    spvVersion(spvVersion),
    source(SourceLanguageUnknown),
    sourceVersion(0),
    sourceFileStringId(NoResult),
    currentLine(0),
    currentFile(nullptr),
    emitOpLines(false),
    addressModel(AddressingModelLogical),
    memoryModel(MemoryModelGLSL450),
    builderNumber(magicNumber),
    buildPoint(0),
    uniqueId(0),
    entryPointFunction(0),
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
    Instruction* import = new Instruction(getUniqueId(), NoType, OpExtInstImport);
    import->addStringOperand(name);
    module.mapInstruction(import);

    imports.push_back(std::unique_ptr<Instruction>(import));
    return import->getResultId();
}

// Emit instruction for non-filename-based #line directives (ie. no filename
// seen yet): emit an OpLine if we've been asked to emit OpLines and the line
// number has changed since the last time, and is a valid line number.
void Builder::setLine(int lineNum)
{
    if (lineNum != 0 && lineNum != currentLine) {
        currentLine = lineNum;
        if (emitOpLines)
            addLine(sourceFileStringId, currentLine, 0);
    }
}

// If no filename, do non-filename-based #line emit. Else do filename-based emit.
// Emit OpLine if we've been asked to emit OpLines and the line number or filename
// has changed since the last time, and line number is valid.
void Builder::setLine(int lineNum, const char* filename)
{
    if (filename == nullptr) {
        setLine(lineNum);
        return;
    }
    if ((lineNum != 0 && lineNum != currentLine) || currentFile == nullptr ||
            strncmp(filename, currentFile, strlen(currentFile) + 1) != 0) {
        currentLine = lineNum;
        currentFile = filename;
        if (emitOpLines) {
            spv::Id strId = getStringId(filename);
            addLine(strId, currentLine, 0);
        }
    }
}

void Builder::addLine(Id fileName, int lineNum, int column)
{
    Instruction* line = new Instruction(OpLine);
    line->addIdOperand(fileName);
    line->addImmediateOperand(lineNum);
    line->addImmediateOperand(column);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(line));
}

// For creating new groupedTypes (will return old type if the requested one was already made).
Id Builder::makeVoidType()
{
    Instruction* type;
    if (groupedTypes[OpTypeVoid].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, OpTypeVoid);
        groupedTypes[OpTypeVoid].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
    } else
        type = groupedTypes[OpTypeVoid].back();

    return type->getResultId();
}

Id Builder::makeBoolType()
{
    Instruction* type;
    if (groupedTypes[OpTypeBool].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, OpTypeBool);
        groupedTypes[OpTypeBool].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
    } else
        type = groupedTypes[OpTypeBool].back();

    return type->getResultId();
}

Id Builder::makeSamplerType()
{
    Instruction* type;
    if (groupedTypes[OpTypeSampler].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, OpTypeSampler);
        groupedTypes[OpTypeSampler].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
    } else
        type = groupedTypes[OpTypeSampler].back();

    return type->getResultId();
}

Id Builder::makePointer(StorageClass storageClass, Id pointee)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypePointer].size(); ++t) {
        type = groupedTypes[OpTypePointer][t];
        if (type->getImmediateOperand(0) == (unsigned)storageClass &&
            type->getIdOperand(1) == pointee)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, OpTypePointer);
    type->addImmediateOperand(storageClass);
    type->addIdOperand(pointee);
    groupedTypes[OpTypePointer].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeForwardPointer(StorageClass storageClass)
{
    // Caching/uniquifying doesn't work here, because we don't know the
    // pointee type and there can be multiple forward pointers of the same
    // storage type. Somebody higher up in the stack must keep track.
    Instruction* type = new Instruction(getUniqueId(), NoType, OpTypeForwardPointer);
    type->addImmediateOperand(storageClass);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makePointerFromForwardPointer(StorageClass storageClass, Id forwardPointerType, Id pointee)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypePointer].size(); ++t) {
        type = groupedTypes[OpTypePointer][t];
        if (type->getImmediateOperand(0) == (unsigned)storageClass &&
            type->getIdOperand(1) == pointee)
            return type->getResultId();
    }

    type = new Instruction(forwardPointerType, NoType, OpTypePointer);
    type->addImmediateOperand(storageClass);
    type->addIdOperand(pointee);
    groupedTypes[OpTypePointer].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeIntegerType(int width, bool hasSign)
{
#ifdef GLSLANG_WEB
    assert(width == 32);
    width = 32;
#endif

    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypeInt].size(); ++t) {
        type = groupedTypes[OpTypeInt][t];
        if (type->getImmediateOperand(0) == (unsigned)width &&
            type->getImmediateOperand(1) == (hasSign ? 1u : 0u))
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, OpTypeInt);
    type->addImmediateOperand(width);
    type->addImmediateOperand(hasSign ? 1 : 0);
    groupedTypes[OpTypeInt].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    // deal with capabilities
    switch (width) {
    case 8:
    case 16:
        // these are currently handled by storage-type declarations and post processing
        break;
    case 64:
        addCapability(CapabilityInt64);
        break;
    default:
        break;
    }

    return type->getResultId();
}

Id Builder::makeFloatType(int width)
{
#ifdef GLSLANG_WEB
    assert(width == 32);
    width = 32;
#endif

    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypeFloat].size(); ++t) {
        type = groupedTypes[OpTypeFloat][t];
        if (type->getImmediateOperand(0) == (unsigned)width)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, OpTypeFloat);
    type->addImmediateOperand(width);
    groupedTypes[OpTypeFloat].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    // deal with capabilities
    switch (width) {
    case 16:
        // currently handled by storage-type declarations and post processing
        break;
    case 64:
        addCapability(CapabilityFloat64);
        break;
    default:
        break;
    }

    return type->getResultId();
}

// Make a struct without checking for duplication.
// See makeStructResultType() for non-decorated structs
// needed as the result of some instructions, which does
// check for duplicates.
Id Builder::makeStructType(const std::vector<Id>& members, const char* name)
{
    // Don't look for previous one, because in the general case,
    // structs can be duplicated except for decorations.

    // not found, make it
    Instruction* type = new Instruction(getUniqueId(), NoType, OpTypeStruct);
    for (int op = 0; op < (int)members.size(); ++op)
        type->addIdOperand(members[op]);
    groupedTypes[OpTypeStruct].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);
    addName(type->getResultId(), name);

    return type->getResultId();
}

// Make a struct for the simple results of several instructions,
// checking for duplication.
Id Builder::makeStructResultType(Id type0, Id type1)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypeStruct].size(); ++t) {
        type = groupedTypes[OpTypeStruct][t];
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

    return makeStructType(members, "ResType");
}

Id Builder::makeVectorType(Id component, int size)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypeVector].size(); ++t) {
        type = groupedTypes[OpTypeVector][t];
        if (type->getIdOperand(0) == component &&
            type->getImmediateOperand(1) == (unsigned)size)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, OpTypeVector);
    type->addIdOperand(component);
    type->addImmediateOperand(size);
    groupedTypes[OpTypeVector].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeMatrixType(Id component, int cols, int rows)
{
    assert(cols <= maxMatrixSize && rows <= maxMatrixSize);

    Id column = makeVectorType(component, rows);

    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypeMatrix].size(); ++t) {
        type = groupedTypes[OpTypeMatrix][t];
        if (type->getIdOperand(0) == column &&
            type->getImmediateOperand(1) == (unsigned)cols)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, OpTypeMatrix);
    type->addIdOperand(column);
    type->addImmediateOperand(cols);
    groupedTypes[OpTypeMatrix].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeCooperativeMatrixType(Id component, Id scope, Id rows, Id cols)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypeCooperativeMatrixNV].size(); ++t) {
        type = groupedTypes[OpTypeCooperativeMatrixNV][t];
        if (type->getIdOperand(0) == component &&
            type->getIdOperand(1) == scope &&
            type->getIdOperand(2) == rows &&
            type->getIdOperand(3) == cols)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, OpTypeCooperativeMatrixNV);
    type->addIdOperand(component);
    type->addIdOperand(scope);
    type->addIdOperand(rows);
    type->addIdOperand(cols);
    groupedTypes[OpTypeCooperativeMatrixNV].push_back(type);
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
        for (int t = 0; t < (int)groupedTypes[OpTypeArray].size(); ++t) {
            type = groupedTypes[OpTypeArray][t];
            if (type->getIdOperand(0) == element &&
                type->getIdOperand(1) == sizeId)
                return type->getResultId();
        }
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, OpTypeArray);
    type->addIdOperand(element);
    type->addIdOperand(sizeId);
    groupedTypes[OpTypeArray].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeRuntimeArray(Id element)
{
    Instruction* type = new Instruction(getUniqueId(), NoType, OpTypeRuntimeArray);
    type->addIdOperand(element);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeFunctionType(Id returnType, const std::vector<Id>& paramTypes)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypeFunction].size(); ++t) {
        type = groupedTypes[OpTypeFunction][t];
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
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, OpTypeFunction);
    type->addIdOperand(returnType);
    for (int p = 0; p < (int)paramTypes.size(); ++p)
        type->addIdOperand(paramTypes[p]);
    groupedTypes[OpTypeFunction].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

Id Builder::makeImageType(Id sampledType, Dim dim, bool depth, bool arrayed, bool ms, unsigned sampled,
    ImageFormat format)
{
    assert(sampled == 1 || sampled == 2);

    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypeImage].size(); ++t) {
        type = groupedTypes[OpTypeImage][t];
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
    type = new Instruction(getUniqueId(), NoType, OpTypeImage);
    type->addIdOperand(sampledType);
    type->addImmediateOperand(   dim);
    type->addImmediateOperand(  depth ? 1 : 0);
    type->addImmediateOperand(arrayed ? 1 : 0);
    type->addImmediateOperand(     ms ? 1 : 0);
    type->addImmediateOperand(sampled);
    type->addImmediateOperand((unsigned int)format);

    groupedTypes[OpTypeImage].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

#ifndef GLSLANG_WEB
    // deal with capabilities
    switch (dim) {
    case DimBuffer:
        if (sampled == 1)
            addCapability(CapabilitySampledBuffer);
        else
            addCapability(CapabilityImageBuffer);
        break;
    case Dim1D:
        if (sampled == 1)
            addCapability(CapabilitySampled1D);
        else
            addCapability(CapabilityImage1D);
        break;
    case DimCube:
        if (arrayed) {
            if (sampled == 1)
                addCapability(CapabilitySampledCubeArray);
            else
                addCapability(CapabilityImageCubeArray);
        }
        break;
    case DimRect:
        if (sampled == 1)
            addCapability(CapabilitySampledRect);
        else
            addCapability(CapabilityImageRect);
        break;
    case DimSubpassData:
        addCapability(CapabilityInputAttachment);
        break;
    default:
        break;
    }

    if (ms) {
        if (sampled == 2) {
            // Images used with subpass data are not storage
            // images, so don't require the capability for them.
            if (dim != Dim::DimSubpassData)
                addCapability(CapabilityStorageImageMultisample);
            if (arrayed)
                addCapability(CapabilityImageMSArray);
        }
    }
#endif

    return type->getResultId();
}

Id Builder::makeSampledImageType(Id imageType)
{
    // try to find it
    Instruction* type;
    for (int t = 0; t < (int)groupedTypes[OpTypeSampledImage].size(); ++t) {
        type = groupedTypes[OpTypeSampledImage][t];
        if (type->getIdOperand(0) == imageType)
            return type->getResultId();
    }

    // not found, make it
    type = new Instruction(getUniqueId(), NoType, OpTypeSampledImage);
    type->addIdOperand(imageType);

    groupedTypes[OpTypeSampledImage].push_back(type);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
    module.mapInstruction(type);

    return type->getResultId();
}

#ifndef GLSLANG_WEB
Id Builder::makeAccelerationStructureType()
{
    Instruction *type;
    if (groupedTypes[OpTypeAccelerationStructureKHR].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, OpTypeAccelerationStructureKHR);
        groupedTypes[OpTypeAccelerationStructureKHR].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
    } else {
        type = groupedTypes[OpTypeAccelerationStructureKHR].back();
    }

    return type->getResultId();
}

Id Builder::makeRayQueryType()
{
    Instruction *type;
    if (groupedTypes[OpTypeRayQueryProvisionalKHR].size() == 0) {
        type = new Instruction(getUniqueId(), NoType, OpTypeRayQueryProvisionalKHR);
        groupedTypes[OpTypeRayQueryProvisionalKHR].push_back(type);
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(type));
        module.mapInstruction(type);
    } else {
        type = groupedTypes[OpTypeRayQueryProvisionalKHR].back();
    }

    return type->getResultId();
}
#endif

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
    case OpTypeVector:
    case OpTypeMatrix:
    case OpTypeArray:
    case OpTypeRuntimeArray:
        return getMostBasicTypeClass(instr->getIdOperand(0));
    case OpTypePointer:
        return getMostBasicTypeClass(instr->getIdOperand(1));
    default:
        return typeClass;
    }
}

int Builder::getNumTypeConstituents(Id typeId) const
{
    Instruction* instr = module.getInstruction(typeId);

    switch (instr->getOpCode())
    {
    case OpTypeBool:
    case OpTypeInt:
    case OpTypeFloat:
    case OpTypePointer:
        return 1;
    case OpTypeVector:
    case OpTypeMatrix:
        return instr->getImmediateOperand(1);
    case OpTypeArray:
    {
        Id lengthId = instr->getIdOperand(1);
        return module.getInstruction(lengthId)->getImmediateOperand(0);
    }
    case OpTypeStruct:
        return instr->getNumOperands();
    case OpTypeCooperativeMatrixNV:
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
    case OpTypeVoid:
    case OpTypeBool:
    case OpTypeInt:
    case OpTypeFloat:
    case OpTypeStruct:
        return instr->getResultId();
    case OpTypeVector:
    case OpTypeMatrix:
    case OpTypeArray:
    case OpTypeRuntimeArray:
    case OpTypePointer:
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
    case OpTypeVector:
    case OpTypeMatrix:
    case OpTypeArray:
    case OpTypeRuntimeArray:
    case OpTypeCooperativeMatrixNV:
        return instr->getIdOperand(0);
    case OpTypePointer:
        return instr->getIdOperand(1);
    case OpTypeStruct:
        return instr->getIdOperand(member);
    default:
        assert(0);
        return NoResult;
    }
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
    case OpTypeInt:
    case OpTypeFloat:
        return typeClass == typeOp && instr.getImmediateOperand(0) == width;
    case OpTypeStruct:
        for (int m = 0; m < instr.getNumOperands(); ++m) {
            if (containsType(instr.getIdOperand(m), typeOp, width))
                return true;
        }
        return false;
    case OpTypePointer:
        return false;
    case OpTypeVector:
    case OpTypeMatrix:
    case OpTypeArray:
    case OpTypeRuntimeArray:
        return containsType(getContainedTypeId(typeId), typeOp, width);
    default:
        return typeClass == typeOp;
    }
}

// return true if the type is a pointer to PhysicalStorageBufferEXT or an
// array of such pointers. These require restrict/aliased decorations.
bool Builder::containsPhysicalStorageBufferOrArray(Id typeId) const
{
    const Instruction& instr = *module.getInstruction(typeId);

    Op typeClass = instr.getOpCode();
    switch (typeClass)
    {
    case OpTypePointer:
        return getTypeStorageClass(typeId) == StorageClassPhysicalStorageBufferEXT;
    case OpTypeArray:
        return containsPhysicalStorageBufferOrArray(getContainedTypeId(typeId));
    default:
        return false;
    }
}

// See if a scalar constant of this type has already been created, so it
// can be reused rather than duplicated.  (Required by the specification).
Id Builder::findScalarConstant(Op typeClass, Op opcode, Id typeId, unsigned value)
{
    Instruction* constant;
    for (int i = 0; i < (int)groupedConstants[typeClass].size(); ++i) {
        constant = groupedConstants[typeClass][i];
        if (constant->getOpCode() == opcode &&
            constant->getTypeId() == typeId &&
            constant->getImmediateOperand(0) == value)
            return constant->getResultId();
    }

    return 0;
}

// Version of findScalarConstant (see above) for scalars that take two operands (e.g. a 'double' or 'int64').
Id Builder::findScalarConstant(Op typeClass, Op opcode, Id typeId, unsigned v1, unsigned v2)
{
    Instruction* constant;
    for (int i = 0; i < (int)groupedConstants[typeClass].size(); ++i) {
        constant = groupedConstants[typeClass][i];
        if (constant->getOpCode() == opcode &&
            constant->getTypeId() == typeId &&
            constant->getImmediateOperand(0) == v1 &&
            constant->getImmediateOperand(1) == v2)
            return constant->getResultId();
    }

    return 0;
}

// Return true if consuming 'opcode' means consuming a constant.
// "constant" here means after final transform to executable code,
// the value consumed will be a constant, so includes specialization.
bool Builder::isConstantOpCode(Op opcode) const
{
    switch (opcode) {
    case OpUndef:
    case OpConstantTrue:
    case OpConstantFalse:
    case OpConstant:
    case OpConstantComposite:
    case OpConstantSampler:
    case OpConstantNull:
    case OpSpecConstantTrue:
    case OpSpecConstantFalse:
    case OpSpecConstant:
    case OpSpecConstantComposite:
    case OpSpecConstantOp:
        return true;
    default:
        return false;
    }
}

// Return true if consuming 'opcode' means consuming a specialization constant.
bool Builder::isSpecConstantOpCode(Op opcode) const
{
    switch (opcode) {
    case OpSpecConstantTrue:
    case OpSpecConstantFalse:
    case OpSpecConstant:
    case OpSpecConstantComposite:
    case OpSpecConstantOp:
        return true;
    default:
        return false;
    }
}

Id Builder::makeBoolConstant(bool b, bool specConstant)
{
    Id typeId = makeBoolType();
    Instruction* constant;
    Op opcode = specConstant ? (b ? OpSpecConstantTrue : OpSpecConstantFalse) : (b ? OpConstantTrue : OpConstantFalse);

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (! specConstant) {
        Id existing = 0;
        for (int i = 0; i < (int)groupedConstants[OpTypeBool].size(); ++i) {
            constant = groupedConstants[OpTypeBool][i];
            if (constant->getTypeId() == typeId && constant->getOpCode() == opcode)
                existing = constant->getResultId();
        }

        if (existing)
            return existing;
    }

    // Make it
    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    groupedConstants[OpTypeBool].push_back(c);
    module.mapInstruction(c);

    return c->getResultId();
}

Id Builder::makeIntConstant(Id typeId, unsigned value, bool specConstant)
{
    Op opcode = specConstant ? OpSpecConstant : OpConstant;

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (! specConstant) {
        Id existing = findScalarConstant(OpTypeInt, opcode, typeId, value);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(value);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    groupedConstants[OpTypeInt].push_back(c);
    module.mapInstruction(c);

    return c->getResultId();
}

Id Builder::makeInt64Constant(Id typeId, unsigned long long value, bool specConstant)
{
    Op opcode = specConstant ? OpSpecConstant : OpConstant;

    unsigned op1 = value & 0xFFFFFFFF;
    unsigned op2 = value >> 32;

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (! specConstant) {
        Id existing = findScalarConstant(OpTypeInt, opcode, typeId, op1, op2);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(op1);
    c->addImmediateOperand(op2);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    groupedConstants[OpTypeInt].push_back(c);
    module.mapInstruction(c);

    return c->getResultId();
}

Id Builder::makeFloatConstant(float f, bool specConstant)
{
    Op opcode = specConstant ? OpSpecConstant : OpConstant;
    Id typeId = makeFloatType(32);
    union { float fl; unsigned int ui; } u;
    u.fl = f;
    unsigned value = u.ui;

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (! specConstant) {
        Id existing = findScalarConstant(OpTypeFloat, opcode, typeId, value);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(value);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    groupedConstants[OpTypeFloat].push_back(c);
    module.mapInstruction(c);

    return c->getResultId();
}

Id Builder::makeDoubleConstant(double d, bool specConstant)
{
#ifdef GLSLANG_WEB
    assert(0);
    return NoResult;
#else
    Op opcode = specConstant ? OpSpecConstant : OpConstant;
    Id typeId = makeFloatType(64);
    union { double db; unsigned long long ull; } u;
    u.db = d;
    unsigned long long value = u.ull;
    unsigned op1 = value & 0xFFFFFFFF;
    unsigned op2 = value >> 32;

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (! specConstant) {
        Id existing = findScalarConstant(OpTypeFloat, opcode, typeId, op1, op2);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(op1);
    c->addImmediateOperand(op2);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    groupedConstants[OpTypeFloat].push_back(c);
    module.mapInstruction(c);

    return c->getResultId();
#endif
}

Id Builder::makeFloat16Constant(float f16, bool specConstant)
{
#ifdef GLSLANG_WEB
    assert(0);
    return NoResult;
#else
    Op opcode = specConstant ? OpSpecConstant : OpConstant;
    Id typeId = makeFloatType(16);

    spvutils::HexFloat<spvutils::FloatProxy<float>> fVal(f16);
    spvutils::HexFloat<spvutils::FloatProxy<spvutils::Float16>> f16Val(0);
    fVal.castTo(f16Val, spvutils::kRoundToZero);

    unsigned value = f16Val.value().getAsFloat().get_value();

    // See if we already made it. Applies only to regular constants, because specialization constants
    // must remain distinct for the purpose of applying a SpecId decoration.
    if (!specConstant) {
        Id existing = findScalarConstant(OpTypeFloat, opcode, typeId, value);
        if (existing)
            return existing;
    }

    Instruction* c = new Instruction(getUniqueId(), typeId, opcode);
    c->addImmediateOperand(value);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    groupedConstants[OpTypeFloat].push_back(c);
    module.mapInstruction(c);

    return c->getResultId();
#endif
}

Id Builder::makeFpConstant(Id type, double d, bool specConstant)
{
#ifdef GLSLANG_WEB
    const int width = 32;
    assert(width == getScalarTypeWidth(type));
#else
    const int width = getScalarTypeWidth(type);
#endif

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

Id Builder::findCompositeConstant(Op typeClass, Id typeId, const std::vector<Id>& comps)
{
    Instruction* constant = 0;
    bool found = false;
    for (int i = 0; i < (int)groupedConstants[typeClass].size(); ++i) {
        constant = groupedConstants[typeClass][i];

        if (constant->getTypeId() != typeId)
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
    Instruction* constant = 0;
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
    Op opcode = specConstant ? OpSpecConstantComposite : OpConstantComposite;
    assert(typeId);
    Op typeClass = getTypeClass(typeId);

    switch (typeClass) {
    case OpTypeVector:
    case OpTypeArray:
    case OpTypeMatrix:
    case OpTypeCooperativeMatrixNV:
        if (! specConstant) {
            Id existing = findCompositeConstant(typeClass, typeId, members);
            if (existing)
                return existing;
        }
        break;
    case OpTypeStruct:
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
    for (int op = 0; op < (int)members.size(); ++op)
        c->addIdOperand(members[op]);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(c));
    if (typeClass == OpTypeStruct)
        groupedStructConstants[typeId].push_back(c);
    else
        groupedConstants[typeClass].push_back(c);
    module.mapInstruction(c);

    return c->getResultId();
}

Instruction* Builder::addEntryPoint(ExecutionModel model, Function* function, const char* name)
{
    Instruction* entryPoint = new Instruction(OpEntryPoint);
    entryPoint->addImmediateOperand(model);
    entryPoint->addIdOperand(function->getId());
    entryPoint->addStringOperand(name);

    entryPoints.push_back(std::unique_ptr<Instruction>(entryPoint));

    return entryPoint;
}

// Currently relying on the fact that all 'value' of interest are small non-negative values.
void Builder::addExecutionMode(Function* entryPoint, ExecutionMode mode, int value1, int value2, int value3)
{
    Instruction* instr = new Instruction(OpExecutionMode);
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

void Builder::addName(Id id, const char* string)
{
    Instruction* name = new Instruction(OpName);
    name->addIdOperand(id);
    name->addStringOperand(string);

    names.push_back(std::unique_ptr<Instruction>(name));
}

void Builder::addMemberName(Id id, int memberNumber, const char* string)
{
    Instruction* name = new Instruction(OpMemberName);
    name->addIdOperand(id);
    name->addImmediateOperand(memberNumber);
    name->addStringOperand(string);

    names.push_back(std::unique_ptr<Instruction>(name));
}

void Builder::addDecoration(Id id, Decoration decoration, int num)
{
    if (decoration == spv::DecorationMax)
        return;

    Instruction* dec = new Instruction(OpDecorate);
    dec->addIdOperand(id);
    dec->addImmediateOperand(decoration);
    if (num >= 0)
        dec->addImmediateOperand(num);

    decorations.push_back(std::unique_ptr<Instruction>(dec));
}

void Builder::addDecoration(Id id, Decoration decoration, const char* s)
{
    if (decoration == spv::DecorationMax)
        return;

    Instruction* dec = new Instruction(OpDecorateStringGOOGLE);
    dec->addIdOperand(id);
    dec->addImmediateOperand(decoration);
    dec->addStringOperand(s);

    decorations.push_back(std::unique_ptr<Instruction>(dec));
}

void Builder::addDecorationId(Id id, Decoration decoration, Id idDecoration)
{
    if (decoration == spv::DecorationMax)
        return;

    Instruction* dec = new Instruction(OpDecorateId);
    dec->addIdOperand(id);
    dec->addImmediateOperand(decoration);
    dec->addIdOperand(idDecoration);

    decorations.push_back(std::unique_ptr<Instruction>(dec));
}

void Builder::addMemberDecoration(Id id, unsigned int member, Decoration decoration, int num)
{
    if (decoration == spv::DecorationMax)
        return;

    Instruction* dec = new Instruction(OpMemberDecorate);
    dec->addIdOperand(id);
    dec->addImmediateOperand(member);
    dec->addImmediateOperand(decoration);
    if (num >= 0)
        dec->addImmediateOperand(num);

    decorations.push_back(std::unique_ptr<Instruction>(dec));
}

void Builder::addMemberDecoration(Id id, unsigned int member, Decoration decoration, const char *s)
{
    if (decoration == spv::DecorationMax)
        return;

    Instruction* dec = new Instruction(OpMemberDecorateStringGOOGLE);
    dec->addIdOperand(id);
    dec->addImmediateOperand(member);
    dec->addImmediateOperand(decoration);
    dec->addStringOperand(s);

    decorations.push_back(std::unique_ptr<Instruction>(dec));
}

// Comments in header
Function* Builder::makeEntryPoint(const char* entryPoint)
{
    assert(! entryPointFunction);

    Block* entry;
    std::vector<Id> params;
    std::vector<std::vector<Decoration>> decorations;

    entryPointFunction = makeFunctionEntry(NoPrecision, makeVoidType(), entryPoint, params, decorations, &entry);

    return entryPointFunction;
}

// Comments in header
Function* Builder::makeFunctionEntry(Decoration precision, Id returnType, const char* name,
                                     const std::vector<Id>& paramTypes,
                                     const std::vector<std::vector<Decoration>>& decorations, Block **entry)
{
    // Make the function and initial instructions in it
    Id typeId = makeFunctionType(returnType, paramTypes);
    Id firstParamId = paramTypes.size() == 0 ? 0 : getUniqueIds((int)paramTypes.size());
    Function* function = new Function(getUniqueId(), returnType, typeId, firstParamId, module);

    // Set up the precisions
    setPrecision(function->getId(), precision);
    for (unsigned p = 0; p < (unsigned)decorations.size(); ++p) {
        for (int d = 0; d < (int)decorations[p].size(); ++d)
            addDecoration(firstParamId + p, decorations[p][d]);
    }

    // CFG
    if (entry) {
        *entry = new Block(getUniqueId(), *function);
        function->addBlock(*entry);
        setBuildPoint(*entry);
    }

    if (name)
        addName(function->getId(), name);

    functions.push_back(std::unique_ptr<Function>(function));

    return function;
}

// Comments in header
void Builder::makeReturn(bool implicit, Id retVal)
{
    if (retVal) {
        Instruction* inst = new Instruction(NoResult, NoType, OpReturnValue);
        inst->addIdOperand(retVal);
        buildPoint->addInstruction(std::unique_ptr<Instruction>(inst));
    } else
        buildPoint->addInstruction(std::unique_ptr<Instruction>(new Instruction(NoResult, NoType, OpReturn)));

    if (! implicit)
        createAndSetNoPredecessorBlock("post-return");
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
}

// Comments in header
void Builder::makeDiscard()
{
    buildPoint->addInstruction(std::unique_ptr<Instruction>(new Instruction(OpKill)));
    createAndSetNoPredecessorBlock("post-discard");
}

// Comments in header
Id Builder::createVariable(StorageClass storageClass, Id type, const char* name, Id initializer)
{
    Id pointerType = makePointer(storageClass, type);
    Instruction* inst = new Instruction(getUniqueId(), pointerType, OpVariable);
    inst->addImmediateOperand(storageClass);
    if (initializer != NoResult)
        inst->addIdOperand(initializer);

    switch (storageClass) {
    case StorageClassFunction:
        // Validation rules require the declaration in the entry block
        buildPoint->getParent().addLocalVariable(std::unique_ptr<Instruction>(inst));
        break;

    default:
        constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(inst));
        module.mapInstruction(inst);
        break;
    }

    if (name)
        addName(inst->getResultId(), name);

    return inst->getResultId();
}

// Comments in header
Id Builder::createUndefined(Id type)
{
  Instruction* inst = new Instruction(getUniqueId(), type, OpUndef);
  buildPoint->addInstruction(std::unique_ptr<Instruction>(inst));
  return inst->getResultId();
}

// av/vis/nonprivate are unnecessary and illegal for some storage classes.
spv::MemoryAccessMask Builder::sanitizeMemoryAccessForStorageClass(spv::MemoryAccessMask memoryAccess, StorageClass sc)
    const
{
    switch (sc) {
    case spv::StorageClassUniform:
    case spv::StorageClassWorkgroup:
    case spv::StorageClassStorageBuffer:
    case spv::StorageClassPhysicalStorageBufferEXT:
        break;
    default:
        memoryAccess = spv::MemoryAccessMask(memoryAccess & 
                        ~(spv::MemoryAccessMakePointerAvailableKHRMask |
                          spv::MemoryAccessMakePointerVisibleKHRMask |
                          spv::MemoryAccessNonPrivatePointerKHRMask));
        break;
    }
    return memoryAccess;
}

// Comments in header
void Builder::createStore(Id rValue, Id lValue, spv::MemoryAccessMask memoryAccess, spv::Scope scope,
    unsigned int alignment)
{
    Instruction* store = new Instruction(OpStore);
    store->addIdOperand(lValue);
    store->addIdOperand(rValue);

    memoryAccess = sanitizeMemoryAccessForStorageClass(memoryAccess, getStorageClass(lValue));

    if (memoryAccess != MemoryAccessMaskNone) {
        store->addImmediateOperand(memoryAccess);
        if (memoryAccess & spv::MemoryAccessAlignedMask) {
            store->addImmediateOperand(alignment);
        }
        if (memoryAccess & spv::MemoryAccessMakePointerAvailableKHRMask) {
            store->addIdOperand(makeUintConstant(scope));
        }
    }

    buildPoint->addInstruction(std::unique_ptr<Instruction>(store));
}

// Comments in header
Id Builder::createLoad(Id lValue, spv::MemoryAccessMask memoryAccess, spv::Scope scope, unsigned int alignment)
{
    Instruction* load = new Instruction(getUniqueId(), getDerefTypeId(lValue), OpLoad);
    load->addIdOperand(lValue);

    memoryAccess = sanitizeMemoryAccessForStorageClass(memoryAccess, getStorageClass(lValue));

    if (memoryAccess != MemoryAccessMaskNone) {
        load->addImmediateOperand(memoryAccess);
        if (memoryAccess & spv::MemoryAccessAlignedMask) {
            load->addImmediateOperand(alignment);
        }
        if (memoryAccess & spv::MemoryAccessMakePointerVisibleKHRMask) {
            load->addIdOperand(makeUintConstant(scope));
        }
    }

    buildPoint->addInstruction(std::unique_ptr<Instruction>(load));

    return load->getResultId();
}

// Comments in header
Id Builder::createAccessChain(StorageClass storageClass, Id base, const std::vector<Id>& offsets)
{
    // Figure out the final resulting type.
    spv::Id typeId = getTypeId(base);
    assert(isPointerType(typeId) && offsets.size() > 0);
    typeId = getContainedTypeId(typeId);
    for (int i = 0; i < (int)offsets.size(); ++i) {
        if (isStructType(typeId)) {
            assert(isConstantScalar(offsets[i]));
            typeId = getContainedTypeId(typeId, getConstantScalar(offsets[i]));
        } else
            typeId = getContainedTypeId(typeId, offsets[i]);
    }
    typeId = makePointer(storageClass, typeId);

    // Make the instruction
    Instruction* chain = new Instruction(getUniqueId(), typeId, OpAccessChain);
    chain->addIdOperand(base);
    for (int i = 0; i < (int)offsets.size(); ++i)
        chain->addIdOperand(offsets[i]);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(chain));

    return chain->getResultId();
}

Id Builder::createArrayLength(Id base, unsigned int member)
{
    spv::Id intType = makeUintType(32);
    Instruction* length = new Instruction(getUniqueId(), intType, OpArrayLength);
    length->addIdOperand(base);
    length->addImmediateOperand(member);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(length));

    return length->getResultId();
}

Id Builder::createCooperativeMatrixLength(Id type)
{
    spv::Id intType = makeUintType(32);

    // Generate code for spec constants if in spec constant operation
    // generation mode.
    if (generatingOpCodeForSpecConst) {
        return createSpecConstantOp(OpCooperativeMatrixLengthNV, intType, std::vector<Id>(1, type), std::vector<Id>());
    }

    Instruction* length = new Instruction(getUniqueId(), intType, OpCooperativeMatrixLengthNV);
    length->addIdOperand(type);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(length));

    return length->getResultId();
}

Id Builder::createCompositeExtract(Id composite, Id typeId, unsigned index)
{
    // Generate code for spec constants if in spec constant operation
    // generation mode.
    if (generatingOpCodeForSpecConst) {
        return createSpecConstantOp(OpCompositeExtract, typeId, std::vector<Id>(1, composite),
            std::vector<Id>(1, index));
    }
    Instruction* extract = new Instruction(getUniqueId(), typeId, OpCompositeExtract);
    extract->addIdOperand(composite);
    extract->addImmediateOperand(index);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(extract));

    return extract->getResultId();
}

Id Builder::createCompositeExtract(Id composite, Id typeId, const std::vector<unsigned>& indexes)
{
    // Generate code for spec constants if in spec constant operation
    // generation mode.
    if (generatingOpCodeForSpecConst) {
        return createSpecConstantOp(OpCompositeExtract, typeId, std::vector<Id>(1, composite), indexes);
    }
    Instruction* extract = new Instruction(getUniqueId(), typeId, OpCompositeExtract);
    extract->addIdOperand(composite);
    for (int i = 0; i < (int)indexes.size(); ++i)
        extract->addImmediateOperand(indexes[i]);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(extract));

    return extract->getResultId();
}

Id Builder::createCompositeInsert(Id object, Id composite, Id typeId, unsigned index)
{
    Instruction* insert = new Instruction(getUniqueId(), typeId, OpCompositeInsert);
    insert->addIdOperand(object);
    insert->addIdOperand(composite);
    insert->addImmediateOperand(index);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(insert));

    return insert->getResultId();
}

Id Builder::createCompositeInsert(Id object, Id composite, Id typeId, const std::vector<unsigned>& indexes)
{
    Instruction* insert = new Instruction(getUniqueId(), typeId, OpCompositeInsert);
    insert->addIdOperand(object);
    insert->addIdOperand(composite);
    for (int i = 0; i < (int)indexes.size(); ++i)
        insert->addImmediateOperand(indexes[i]);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(insert));

    return insert->getResultId();
}

Id Builder::createVectorExtractDynamic(Id vector, Id typeId, Id componentIndex)
{
    Instruction* extract = new Instruction(getUniqueId(), typeId, OpVectorExtractDynamic);
    extract->addIdOperand(vector);
    extract->addIdOperand(componentIndex);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(extract));

    return extract->getResultId();
}

Id Builder::createVectorInsertDynamic(Id vector, Id typeId, Id component, Id componentIndex)
{
    Instruction* insert = new Instruction(getUniqueId(), typeId, OpVectorInsertDynamic);
    insert->addIdOperand(vector);
    insert->addIdOperand(component);
    insert->addIdOperand(componentIndex);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(insert));

    return insert->getResultId();
}

// An opcode that has no operands, no result id, and no type
void Builder::createNoResultOp(Op opCode)
{
    Instruction* op = new Instruction(opCode);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));
}

// An opcode that has one id operand, no result id, and no type
void Builder::createNoResultOp(Op opCode, Id operand)
{
    Instruction* op = new Instruction(opCode);
    op->addIdOperand(operand);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));
}

// An opcode that has one or more operands, no result id, and no type
void Builder::createNoResultOp(Op opCode, const std::vector<Id>& operands)
{
    Instruction* op = new Instruction(opCode);
    for (auto it = operands.cbegin(); it != operands.cend(); ++it) {
        op->addIdOperand(*it);
    }
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));
}

// An opcode that has multiple operands, no result id, and no type
void Builder::createNoResultOp(Op opCode, const std::vector<IdImmediate>& operands)
{
    Instruction* op = new Instruction(opCode);
    for (auto it = operands.cbegin(); it != operands.cend(); ++it) {
        if (it->isId)
            op->addIdOperand(it->word);
        else
            op->addImmediateOperand(it->word);
    }
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));
}

void Builder::createControlBarrier(Scope execution, Scope memory, MemorySemanticsMask semantics)
{
    Instruction* op = new Instruction(OpControlBarrier);
    op->addIdOperand(makeUintConstant(execution));
    op->addIdOperand(makeUintConstant(memory));
    op->addIdOperand(makeUintConstant(semantics));
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));
}

void Builder::createMemoryBarrier(unsigned executionScope, unsigned memorySemantics)
{
    Instruction* op = new Instruction(OpMemoryBarrier);
    op->addIdOperand(makeUintConstant(executionScope));
    op->addIdOperand(makeUintConstant(memorySemantics));
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));
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
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));

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
    op->addIdOperand(left);
    op->addIdOperand(right);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));

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
    op->addIdOperand(op1);
    op->addIdOperand(op2);
    op->addIdOperand(op3);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

Id Builder::createOp(Op opCode, Id typeId, const std::vector<Id>& operands)
{
    Instruction* op = new Instruction(getUniqueId(), typeId, opCode);
    for (auto it = operands.cbegin(); it != operands.cend(); ++it)
        op->addIdOperand(*it);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

Id Builder::createOp(Op opCode, Id typeId, const std::vector<IdImmediate>& operands)
{
    Instruction* op = new Instruction(getUniqueId(), typeId, opCode);
    for (auto it = operands.cbegin(); it != operands.cend(); ++it) {
        if (it->isId)
            op->addIdOperand(it->word);
        else
            op->addImmediateOperand(it->word);
    }
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

Id Builder::createSpecConstantOp(Op opCode, Id typeId, const std::vector<Id>& operands,
    const std::vector<unsigned>& literals)
{
    Instruction* op = new Instruction(getUniqueId(), typeId, OpSpecConstantOp);
    op->addImmediateOperand((unsigned) opCode);
    for (auto it = operands.cbegin(); it != operands.cend(); ++it)
        op->addIdOperand(*it);
    for (auto it = literals.cbegin(); it != literals.cend(); ++it)
        op->addImmediateOperand(*it);
    module.mapInstruction(op);
    constantsTypesGlobals.push_back(std::unique_ptr<Instruction>(op));

    return op->getResultId();
}

Id Builder::createFunctionCall(spv::Function* function, const std::vector<spv::Id>& args)
{
    Instruction* op = new Instruction(getUniqueId(), function->getReturnType(), OpFunctionCall);
    op->addIdOperand(function->getId());
    for (int a = 0; a < (int)args.size(); ++a)
        op->addIdOperand(args[a]);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));

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
        return setPrecision(createSpecConstantOp(OpVectorShuffle, typeId, operands, channels), precision);
    }
    Instruction* swizzle = new Instruction(getUniqueId(), typeId, OpVectorShuffle);
    assert(isVector(source));
    swizzle->addIdOperand(source);
    swizzle->addIdOperand(source);
    for (int i = 0; i < (int)channels.size(); ++i)
        swizzle->addImmediateOperand(channels[i]);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(swizzle));

    return setPrecision(swizzle->getResultId(), precision);
}

// Comments in header
Id Builder::createLvalueSwizzle(Id typeId, Id target, Id source, const std::vector<unsigned>& channels)
{
    if (channels.size() == 1 && getNumComponents(source) == 1)
        return createCompositeInsert(source, target, typeId, channels.front());

    Instruction* swizzle = new Instruction(getUniqueId(), typeId, OpVectorShuffle);

    assert(isVector(target));
    swizzle->addIdOperand(target);

    assert(getNumComponents(source) == (int)channels.size());
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
    for (int i = 0; i < numTargetComponents; ++i)
        swizzle->addImmediateOperand(components[i]);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(swizzle));

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
    if (numComponents == 1)
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
        smear = new Instruction(getUniqueId(), vectorType, OpCompositeConstruct);
        for (int c = 0; c < numComponents; ++c)
            smear->addIdOperand(scalar);
        buildPoint->addInstruction(std::unique_ptr<Instruction>(smear));
    }

    return setPrecision(smear->getResultId(), precision);
}

// Comments in header
Id Builder::createBuiltinCall(Id resultType, Id builtins, int entryPoint, const std::vector<Id>& args)
{
    Instruction* inst = new Instruction(getUniqueId(), resultType, OpExtInst);
    inst->addIdOperand(builtins);
    inst->addImmediateOperand(entryPoint);
    for (int arg = 0; arg < (int)args.size(); ++arg)
        inst->addIdOperand(args[arg]);

    buildPoint->addInstruction(std::unique_ptr<Instruction>(inst));

    return inst->getResultId();
}

// Accept all parameters needed to create a texture instruction.
// Create the correct instruction based on the inputs, and make the call.
Id Builder::createTextureCall(Decoration precision, Id resultType, bool sparse, bool fetch, bool proj, bool gather,
    bool noImplicitLod, const TextureParameters& parameters, ImageOperandsMask signExtensionMask)
{
    static const int maxTextureArgs = 10;
    Id texArgs[maxTextureArgs] = {};

    //
    // Set up the fixed arguments
    //
    int numArgs = 0;
    bool explicitLod = false;
    texArgs[numArgs++] = parameters.sampler;
    texArgs[numArgs++] = parameters.coords;
    if (parameters.Dref != NoResult)
        texArgs[numArgs++] = parameters.Dref;
    if (parameters.component != NoResult)
        texArgs[numArgs++] = parameters.component;

#ifndef GLSLANG_WEB
    if (parameters.granularity != NoResult)
        texArgs[numArgs++] = parameters.granularity;
    if (parameters.coarse != NoResult)
        texArgs[numArgs++] = parameters.coarse;
#endif 

    //
    // Set up the optional arguments
    //
    int optArgNum = numArgs;    // track which operand, if it exists, is the mask of optional arguments
    ++numArgs;                  // speculatively make room for the mask operand
    ImageOperandsMask mask = ImageOperandsMaskNone; // the mask operand
    if (parameters.bias) {
        mask = (ImageOperandsMask)(mask | ImageOperandsBiasMask);
        texArgs[numArgs++] = parameters.bias;
    }
    if (parameters.lod) {
        mask = (ImageOperandsMask)(mask | ImageOperandsLodMask);
        texArgs[numArgs++] = parameters.lod;
        explicitLod = true;
    } else if (parameters.gradX) {
        mask = (ImageOperandsMask)(mask | ImageOperandsGradMask);
        texArgs[numArgs++] = parameters.gradX;
        texArgs[numArgs++] = parameters.gradY;
        explicitLod = true;
    } else if (noImplicitLod && ! fetch && ! gather) {
        // have to explicitly use lod of 0 if not allowed to have them be implicit, and
        // we would otherwise be about to issue an implicit instruction
        mask = (ImageOperandsMask)(mask | ImageOperandsLodMask);
        texArgs[numArgs++] = makeFloatConstant(0.0);
        explicitLod = true;
    }
    if (parameters.offset) {
        if (isConstant(parameters.offset))
            mask = (ImageOperandsMask)(mask | ImageOperandsConstOffsetMask);
        else {
            addCapability(CapabilityImageGatherExtended);
            mask = (ImageOperandsMask)(mask | ImageOperandsOffsetMask);
        }
        texArgs[numArgs++] = parameters.offset;
    }
    if (parameters.offsets) {
        addCapability(CapabilityImageGatherExtended);
        mask = (ImageOperandsMask)(mask | ImageOperandsConstOffsetsMask);
        texArgs[numArgs++] = parameters.offsets;
    }
#ifndef GLSLANG_WEB
    if (parameters.sample) {
        mask = (ImageOperandsMask)(mask | ImageOperandsSampleMask);
        texArgs[numArgs++] = parameters.sample;
    }
    if (parameters.lodClamp) {
        // capability if this bit is used
        addCapability(CapabilityMinLod);

        mask = (ImageOperandsMask)(mask | ImageOperandsMinLodMask);
        texArgs[numArgs++] = parameters.lodClamp;
    }
    if (parameters.nonprivate) {
        mask = mask | ImageOperandsNonPrivateTexelKHRMask;
    }
    if (parameters.volatil) {
        mask = mask | ImageOperandsVolatileTexelKHRMask;
    }
#endif
    mask = mask | signExtensionMask;
    if (mask == ImageOperandsMaskNone)
        --numArgs;  // undo speculative reservation for the mask argument
    else
        texArgs[optArgNum] = mask;

    //
    // Set up the instruction
    //
    Op opCode = OpNop;  // All paths below need to set this
    if (fetch) {
        if (sparse)
            opCode = OpImageSparseFetch;
        else
            opCode = OpImageFetch;
#ifndef GLSLANG_WEB
    } else if (parameters.granularity && parameters.coarse) {
        opCode = OpImageSampleFootprintNV;
    } else if (gather) {
        if (parameters.Dref)
            if (sparse)
                opCode = OpImageSparseDrefGather;
            else
                opCode = OpImageDrefGather;
        else
            if (sparse)
                opCode = OpImageSparseGather;
            else
                opCode = OpImageGather;
#endif
    } else if (explicitLod) {
        if (parameters.Dref) {
            if (proj)
                if (sparse)
                    opCode = OpImageSparseSampleProjDrefExplicitLod;
                else
                    opCode = OpImageSampleProjDrefExplicitLod;
            else
                if (sparse)
                    opCode = OpImageSparseSampleDrefExplicitLod;
                else
                    opCode = OpImageSampleDrefExplicitLod;
        } else {
            if (proj)
                if (sparse)
                    opCode = OpImageSparseSampleProjExplicitLod;
                else
                    opCode = OpImageSampleProjExplicitLod;
            else
                if (sparse)
                    opCode = OpImageSparseSampleExplicitLod;
                else
                    opCode = OpImageSampleExplicitLod;
        }
    } else {
        if (parameters.Dref) {
            if (proj)
                if (sparse)
                    opCode = OpImageSparseSampleProjDrefImplicitLod;
                else
                    opCode = OpImageSampleProjDrefImplicitLod;
            else
                if (sparse)
                    opCode = OpImageSparseSampleDrefImplicitLod;
                else
                    opCode = OpImageSampleDrefImplicitLod;
        } else {
            if (proj)
                if (sparse)
                    opCode = OpImageSparseSampleProjImplicitLod;
                else
                    opCode = OpImageSampleProjImplicitLod;
            else
                if (sparse)
                    opCode = OpImageSparseSampleImplicitLod;
                else
                    opCode = OpImageSampleImplicitLod;
        }
    }

    // See if the result type is expecting a smeared result.
    // This happens when a legacy shadow*() call is made, which
    // gets a vec4 back instead of a float.
    Id smearedType = resultType;
    if (! isScalarType(resultType)) {
        switch (opCode) {
        case OpImageSampleDrefImplicitLod:
        case OpImageSampleDrefExplicitLod:
        case OpImageSampleProjDrefImplicitLod:
        case OpImageSampleProjDrefExplicitLod:
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
    for (int op = 0; op < optArgNum; ++op)
        textureInst->addIdOperand(texArgs[op]);
    if (optArgNum < numArgs)
        textureInst->addImmediateOperand(texArgs[optArgNum]);
    for (int op = optArgNum + 1; op < numArgs; ++op)
        textureInst->addIdOperand(texArgs[op]);
    setPrecision(textureInst->getResultId(), precision);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(textureInst));

    Id resultId = textureInst->getResultId();

    if (sparse) {
        // set capability
        addCapability(CapabilitySparseResidency);

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
    case OpImageQuerySize:
    case OpImageQuerySizeLod:
    {
        int numComponents = 0;
        switch (getTypeDimensionality(getImageType(parameters.sampler))) {
        case Dim1D:
        case DimBuffer:
            numComponents = 1;
            break;
        case Dim2D:
        case DimCube:
        case DimRect:
        case DimSubpassData:
            numComponents = 2;
            break;
        case Dim3D:
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
    case OpImageQueryLod:
        resultType = makeVectorType(getScalarTypeId(getTypeId(parameters.coords)), 2);
        break;
    case OpImageQueryLevels:
    case OpImageQuerySamples:
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
    buildPoint->addInstruction(std::unique_ptr<Instruction>(query));
    addCapability(CapabilityImageQuery);

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
        case OpTypeFloat:
            op = equal ? OpFOrdEqual : OpFOrdNotEqual;
            break;
        case OpTypeInt:
        default:
            op = equal ? OpIEqual : OpINotEqual;
            break;
        case OpTypeBool:
            op = equal ? OpLogicalEqual : OpLogicalNotEqual;
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
            resultId = createUnaryOp(equal ? OpAll : OpAny, boolType, resultId);
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
            resultId = setPrecision(createBinOp(equal ? OpLogicalAnd : OpLogicalOr, boolType, resultId, subResultId),
                                    precision);
    }

    return resultId;
}

// OpCompositeConstruct
Id Builder::createCompositeConstruct(Id typeId, const std::vector<Id>& constituents)
{
    assert(isAggregateType(typeId) || (getNumTypeConstituents(typeId) > 1 &&
           getNumTypeConstituents(typeId) == (int)constituents.size()));

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

    Instruction* op = new Instruction(getUniqueId(), typeId, OpCompositeConstruct);
    for (int c = 0; c < (int)constituents.size(); ++c)
        op->addIdOperand(constituents[c]);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(op));

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
    if (sources.size() == 1 && isScalar(sources[0]) && numTargetComponents > 1)
        return smearScalar(precision, sources[0], resultTypeId);

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

        int col = 0;
        int row = 0;
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
        else if (isVector(sources[i]))
            accumulateVectorConstituents(sources[i]);
        else if (isMatrix(sources[i]))
            accumulateMatrixConstituents(sources[i]);
        else
            assert(0);

        if (targetComponent >= numTargetComponents)
            break;
    }

    // If the result is a vector, make it from the gathered constituents.
    if (constituents.size() > 0)
        result = createCompositeConstruct(resultTypeId, constituents);

    return setPrecision(result, precision);
}

// Comments in header
Id Builder::createMatrixConstructor(Decoration precision, const std::vector<Id>& sources, Id resultTypeId)
{
    Id componentTypeId = getScalarTypeId(resultTypeId);
    int numCols = getTypeNumColumns(resultTypeId);
    int numRows = getTypeNumRows(resultTypeId);

    Instruction* instr = module.getInstruction(componentTypeId);
#ifdef GLSLANG_WEB
    const unsigned bitCount = 32;
    assert(bitCount == instr->getImmediateOperand(0));
#else
    const unsigned bitCount = instr->getImmediateOperand(0);
#endif

    // Optimize matrix constructed from a bigger matrix
    if (isMatrix(sources[0]) && getNumColumns(sources[0]) >= numCols && getNumRows(sources[0]) >= numRows) {
        // To truncate the matrix to a smaller number of rows/columns, we need to:
        // 1. For each column, extract the column and truncate it to the required size using shuffle
        // 2. Assemble the resulting matrix from all columns
        Id matrix = sources[0];
        Id columnTypeId = getContainedTypeId(resultTypeId);
        Id sourceColumnTypeId = getContainedTypeId(getTypeId(matrix));

        std::vector<unsigned> channels;
        for (int row = 0; row < numRows; ++row)
            channels.push_back(row);

        std::vector<Id> matrixColumns;
        for (int col = 0; col < numCols; ++col) {
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
        int minCols = std::min(numCols, getNumColumns(matrix));
        int minRows = std::min(numRows, getNumRows(matrix));
        for (int col = 0; col < minCols; ++col) {
            std::vector<unsigned> indexes;
            indexes.push_back(col);
            for (int row = 0; row < minRows; ++row) {
                indexes.push_back(row);
                ids[col][row] = createCompositeExtract(matrix, componentTypeId, indexes);
                indexes.pop_back();
                setPrecision(ids[col][row], precision);
            }
        }
    } else {
        // fill in the matrix in column-major order with whatever argument components are available
        int row = 0;
        int col = 0;

        for (int arg = 0; arg < (int)sources.size(); ++arg) {
            Id argComp = sources[arg];
            for (int comp = 0; comp < getNumComponents(sources[arg]); ++comp) {
                if (getNumComponents(sources[arg]) > 1) {
                    argComp = createCompositeExtract(sources[arg], componentTypeId, comp);
                    setPrecision(argComp, precision);
                }
                ids[col][row++] = argComp;
                if (row == numRows) {
                    row = 0;
                    col++;
                }
            }
        }
    }

    // Step 2:  Construct a matrix from that array.
    // First make the column vectors, then make the matrix.

    // make the column vectors
    Id columnTypeId = getContainedTypeId(resultTypeId);
    std::vector<Id> matrixColumns;
    for (int col = 0; col < numCols; ++col) {
        std::vector<Id> vectorComponents;
        for (int row = 0; row < numRows; ++row)
            vectorComponents.push_back(ids[col][row]);
        Id column = createCompositeConstruct(columnTypeId, vectorComponents);
        setPrecision(column, precision);
        matrixColumns.push_back(column);
    }

    // make the matrix
    return setPrecision(createCompositeConstruct(resultTypeId, matrixColumns), precision);
}

// Comments in header
Builder::If::If(Id cond, unsigned int ctrl, Builder& gb) :
    builder(gb),
    condition(cond),
    control(ctrl),
    elseBlock(0)
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

    function->addBlock(thenBlock);
    builder.setBuildPoint(thenBlock);
}

// Comments in header
void Builder::If::makeBeginElse()
{
    // Close out the "then" by having it jump to the mergeBlock
    builder.createBranch(mergeBlock);

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
    builder.createBranch(mergeBlock);

    // Go back to the headerBlock and make the flow control split
    builder.setBuildPoint(headerBlock);
    builder.createSelectionMerge(mergeBlock, control);
    if (elseBlock)
        builder.createConditionalBranch(condition, thenBlock, elseBlock);
    else
        builder.createConditionalBranch(condition, thenBlock, mergeBlock);

    // add the merge block to the function
    function->addBlock(mergeBlock);
    builder.setBuildPoint(mergeBlock);
}

// Comments in header
void Builder::makeSwitch(Id selector, unsigned int control, int numSegments, const std::vector<int>& caseValues,
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
    Instruction* switchInst = new Instruction(NoResult, NoType, OpSwitch);
    switchInst->addIdOperand(selector);
    auto defaultOrMerge = (defaultSegment >= 0) ? segmentBlocks[defaultSegment] : mergeBlock;
    switchInst->addIdOperand(defaultOrMerge->getId());
    defaultOrMerge->addPredecessor(buildPoint);
    for (int i = 0; i < (int)caseValues.size(); ++i) {
        switchInst->addImmediateOperand(caseValues[i]);
        switchInst->addIdOperand(segmentBlocks[valueIndexToSegment[i]]->getId());
        segmentBlocks[valueIndexToSegment[i]]->addPredecessor(buildPoint);
    }
    buildPoint->addInstruction(std::unique_ptr<Instruction>(switchInst));

    // push the merge block
    switchMerges.push(mergeBlock);
}

// Comments in header
void Builder::addSwitchBreak()
{
    // branch to the top of the merge block stack
    createBranch(switchMerges.top());
    createAndSetNoPredecessorBlock("post-switch-break");
}

// Comments in header
void Builder::nextSwitchSegment(std::vector<Block*>& segmentBlock, int nextSegment)
{
    int lastSegment = nextSegment - 1;
    if (lastSegment >= 0) {
        // Close out previous segment by jumping, if necessary, to next segment
        if (! buildPoint->isTerminated())
            createBranch(segmentBlock[nextSegment]);
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
        addSwitchBreak();

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
    createBranch(&loops.top().continue_target);
    // Set up a block for dead code.
    createAndSetNoPredecessorBlock("post-loop-continue");
}

void Builder::createLoopExit()
{
    createBranch(&loops.top().merge);
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
void Builder::accessChainStore(Id rvalue, spv::MemoryAccessMask memoryAccess, spv::Scope scope, unsigned int alignment)
{
    assert(accessChain.isRValue == false);

    transferAccessChainSwizzle(true);
    Id base = collapseAccessChain();
    Id source = rvalue;

    // dynamic component should be gone
    assert(accessChain.component == NoResult);

    // If swizzle still exists, it is out-of-order or not full, we must load the target vector,
    // extract and insert elements to perform writeMask and/or swizzle.
    if (accessChain.swizzle.size() > 0) {
        Id tempBaseId = createLoad(base);
        source = createLvalueSwizzle(getTypeId(tempBaseId), tempBaseId, source, accessChain.swizzle);
    }

    // take LSB of alignment
    alignment = alignment & ~(alignment & (alignment-1));
    if (getStorageClass(base) == StorageClassPhysicalStorageBufferEXT) {
        memoryAccess = (spv::MemoryAccessMask)(memoryAccess | spv::MemoryAccessAlignedMask);
    }

    createStore(source, base, memoryAccess, scope, alignment);
}

// Comments in header
Id Builder::accessChainLoad(Decoration precision, Decoration nonUniform, Id resultType,
    spv::MemoryAccessMask memoryAccess, spv::Scope scope, unsigned int alignment)
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
            } else {
                Id lValue = NoResult;
                if (spvVersion >= Spv_1_4) {
                    // make a new function variable for this r-value, using an initializer,
                    // and mark it as NonWritable so that downstream it can be detected as a lookup
                    // table
                    lValue = createVariable(StorageClassFunction, getTypeId(accessChain.base), "indexable",
                        accessChain.base);
                    addDecoration(lValue, DecorationNonWritable);
                } else {
                    lValue = createVariable(StorageClassFunction, getTypeId(accessChain.base), "indexable");
                    // store into it
                    createStore(accessChain.base, lValue);
                }
                // move base to the new variable
                accessChain.base = lValue;
                accessChain.isRValue = false;

                // load through the access chain
                id = createLoad(collapseAccessChain());
            }
            setPrecision(id, precision);
        } else
            id = accessChain.base;  // no precision, it was set when this was defined
    } else {
        transferAccessChainSwizzle(true);

        // take LSB of alignment
        alignment = alignment & ~(alignment & (alignment-1));
        if (getStorageClass(accessChain.base) == StorageClassPhysicalStorageBufferEXT) {
            memoryAccess = (spv::MemoryAccessMask)(memoryAccess | spv::MemoryAccessAlignedMask);
        }

        // load through the access chain
        id = collapseAccessChain();
        // Apply nonuniform both to the access chain and the loaded value.
        // Buffer accesses need the access chain decorated, and this is where
        // loaded image types get decorated. TODO: This should maybe move to
        // createImageTextureFunctionCall.
        addDecoration(id, nonUniform);
        id = createLoad(id, memoryAccess, scope, alignment);
        setPrecision(id, precision);
        addDecoration(id, nonUniform);
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

    addDecoration(id, nonUniform);
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
        Instruction capInst(0, 0, OpCapability);
        capInst.addImmediateOperand(*it);
        capInst.dump(out);
    }

    for (auto it = extensions.cbegin(); it != extensions.cend(); ++it) {
        Instruction extInst(0, 0, OpExtension);
        extInst.addStringOperand(it->c_str());
        extInst.dump(out);
    }

    dumpInstructions(out, imports);
    Instruction memInst(0, 0, OpMemoryModel);
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
        Instruction sourceExtInst(0, 0, OpSourceExtension);
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
    if (getNumTypeComponents(accessChain.preSwizzleBaseType) > (int)accessChain.swizzle.size())
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
void Builder::createBranch(Block* block)
{
    Instruction* branch = new Instruction(OpBranch);
    branch->addIdOperand(block->getId());
    buildPoint->addInstruction(std::unique_ptr<Instruction>(branch));
    block->addPredecessor(buildPoint);
}

void Builder::createSelectionMerge(Block* mergeBlock, unsigned int control)
{
    Instruction* merge = new Instruction(OpSelectionMerge);
    merge->addIdOperand(mergeBlock->getId());
    merge->addImmediateOperand(control);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(merge));
}

void Builder::createLoopMerge(Block* mergeBlock, Block* continueBlock, unsigned int control,
                              const std::vector<unsigned int>& operands)
{
    Instruction* merge = new Instruction(OpLoopMerge);
    merge->addIdOperand(mergeBlock->getId());
    merge->addIdOperand(continueBlock->getId());
    merge->addImmediateOperand(control);
    for (int op = 0; op < (int)operands.size(); ++op)
        merge->addImmediateOperand(operands[op]);
    buildPoint->addInstruction(std::unique_ptr<Instruction>(merge));
}

void Builder::createConditionalBranch(Id condition, Block* thenBlock, Block* elseBlock)
{
    Instruction* branch = new Instruction(OpBranchConditional);
    branch->addIdOperand(condition);
    branch->addIdOperand(thenBlock->getId());
    branch->addIdOperand(elseBlock->getId());
    buildPoint->addInstruction(std::unique_ptr<Instruction>(branch));
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

    if (source != SourceLanguageUnknown) {
        // OpSource Language Version File Source
        Instruction sourceInst(NoResult, NoType, OpSource);
        sourceInst.addImmediateOperand(source);
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
                        Instruction sourceContinuedInst(OpSourceContinued);
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
    dumpSourceInstructions(sourceFileStringId, sourceText, out);
    for (auto iItr = includeFiles.begin(); iItr != includeFiles.end(); ++iItr)
        dumpSourceInstructions(iItr->first, *iItr->second, out);
}

void Builder::dumpInstructions(std::vector<unsigned int>& out,
    const std::vector<std::unique_ptr<Instruction> >& instructions) const
{
    for (int i = 0; i < (int)instructions.size(); ++i) {
        instructions[i]->dump(out);
    }
}

void Builder::dumpModuleProcesses(std::vector<unsigned int>& out) const
{
    for (int i = 0; i < (int)moduleProcesses.size(); ++i) {
        Instruction moduleProcessed(OpModuleProcessed);
        moduleProcessed.addStringOperand(moduleProcesses[i]);
        moduleProcessed.dump(out);
    }
}

}; // end spv namespace
