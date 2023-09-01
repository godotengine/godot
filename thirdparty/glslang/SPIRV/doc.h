//
// Copyright (C) 2014-2015 LunarG, Inc.
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
// Parameterize the SPIR-V enumerants.
//

#pragma once

#include "spirv.hpp"

#include <vector>

namespace spv {

// Fill in all the parameters
void Parameterize();

// Return the English names of all the enums.
const char* SourceString(int);
const char* AddressingString(int);
const char* MemoryString(int);
const char* ExecutionModelString(int);
const char* ExecutionModeString(int);
const char* StorageClassString(int);
const char* DecorationString(int);
const char* BuiltInString(int);
const char* DimensionString(int);
const char* SelectControlString(int);
const char* LoopControlString(int);
const char* FunctionControlString(int);
const char* SamplerAddressingModeString(int);
const char* SamplerFilterModeString(int);
const char* ImageFormatString(int);
const char* ImageChannelOrderString(int);
const char* ImageChannelTypeString(int);
const char* ImageChannelDataTypeString(int type);
const char* ImageOperandsString(int format);
const char* ImageOperands(int);
const char* FPFastMathString(int);
const char* FPRoundingModeString(int);
const char* LinkageTypeString(int);
const char* FuncParamAttrString(int);
const char* AccessQualifierString(int);
const char* MemorySemanticsString(int);
const char* MemoryAccessString(int);
const char* ExecutionScopeString(int);
const char* GroupOperationString(int);
const char* KernelEnqueueFlagsString(int);
const char* KernelProfilingInfoString(int);
const char* CapabilityString(int);
const char* OpcodeString(int);
const char* ScopeString(int mem);

// For grouping opcodes into subsections
enum OpcodeClass {
    OpClassMisc,
    OpClassDebug,
    OpClassAnnotate,
    OpClassExtension,
    OpClassMode,
    OpClassType,
    OpClassConstant,
    OpClassMemory,
    OpClassFunction,
    OpClassImage,
    OpClassConvert,
    OpClassComposite,
    OpClassArithmetic,
    OpClassBit,
    OpClassRelationalLogical,
    OpClassDerivative,
    OpClassFlowControl,
    OpClassAtomic,
    OpClassPrimitive,
    OpClassBarrier,
    OpClassGroup,
    OpClassDeviceSideEnqueue,
    OpClassPipe,

    OpClassCount,
    OpClassMissing             // all instructions start out as missing
};

// For parameterizing operands.
enum OperandClass {
    OperandNone,
    OperandId,
    OperandVariableIds,
    OperandOptionalLiteral,
    OperandOptionalLiteralString,
    OperandVariableLiterals,
    OperandVariableIdLiteral,
    OperandVariableLiteralId,
    OperandLiteralNumber,
    OperandLiteralString,
    OperandVariableLiteralStrings,
    OperandSource,
    OperandExecutionModel,
    OperandAddressing,
    OperandMemory,
    OperandExecutionMode,
    OperandStorage,
    OperandDimensionality,
    OperandSamplerAddressingMode,
    OperandSamplerFilterMode,
    OperandSamplerImageFormat,
    OperandImageChannelOrder,
    OperandImageChannelDataType,
    OperandImageOperands,
    OperandFPFastMath,
    OperandFPRoundingMode,
    OperandLinkageType,
    OperandAccessQualifier,
    OperandFuncParamAttr,
    OperandDecoration,
    OperandBuiltIn,
    OperandSelect,
    OperandLoop,
    OperandFunction,
    OperandMemorySemantics,
    OperandMemoryAccess,
    OperandScope,
    OperandGroupOperation,
    OperandKernelEnqueueFlags,
    OperandKernelProfilingInfo,
    OperandCapability,
    OperandCooperativeMatrixOperands,

    OperandOpcode,

    OperandCount
};

// Any specific enum can have a set of capabilities that allow it:
typedef std::vector<Capability> EnumCaps;

// Parameterize a set of operands with their OperandClass(es) and descriptions.
class OperandParameters {
public:
    OperandParameters() { }
    void push(OperandClass oc, const char* d, bool opt = false)
    {
        opClass.push_back(oc);
        desc.push_back(d);
        optional.push_back(opt);
    }
    void setOptional();
    OperandClass getClass(int op) const { return opClass[op]; }
    const char* getDesc(int op) const { return desc[op]; }
    bool isOptional(int op) const { return optional[op]; }
    int getNum() const { return (int)opClass.size(); }

protected:
    std::vector<OperandClass> opClass;
    std::vector<const char*> desc;
    std::vector<bool> optional;
};

// Parameterize an enumerant
class EnumParameters {
public:
    EnumParameters() : desc(nullptr) { }
    const char* desc;
};

// Parameterize a set of enumerants that form an enum
class EnumDefinition : public EnumParameters {
public:
    EnumDefinition() :
        ceiling(0), bitmask(false), getName(nullptr), enumParams(nullptr), operandParams(nullptr) { }
    void set(int ceil, const char* (*name)(int), EnumParameters* ep, bool mask = false)
    {
        ceiling = ceil;
        getName = name;
        bitmask = mask;
        enumParams = ep;
    }
    void setOperands(OperandParameters* op) { operandParams = op; }
    int ceiling;   // ceiling of enumerants
    bool bitmask;  // true if these enumerants combine into a bitmask
    const char* (*getName)(int);      // a function that returns the name for each enumerant value (or shift)
    EnumParameters* enumParams;       // parameters for each individual enumerant
    OperandParameters* operandParams; // sets of operands
};

// Parameterize an instruction's logical format, including its known set of operands,
// per OperandParameters above.
class InstructionParameters {
public:
    InstructionParameters() :
        opDesc("TBD"),
        opClass(OpClassMissing),
        typePresent(true),         // most normal, only exceptions have to be spelled out
        resultPresent(true)        // most normal, only exceptions have to be spelled out
    { }

    void setResultAndType(bool r, bool t)
    {
        resultPresent = r;
        typePresent = t;
    }

    bool hasResult() const { return resultPresent != 0; }
    bool hasType()   const { return typePresent != 0; }

    const char* opDesc;
    OpcodeClass opClass;
    OperandParameters operands;

protected:
    int typePresent   : 1;
    int resultPresent : 1;
};

// The set of objects that hold all the instruction/operand
// parameterization information.
extern InstructionParameters InstructionDesc[];

// These hold definitions of the enumerants used for operands
extern EnumDefinition OperandClassParams[];

const char* GetOperandDesc(OperandClass operand);
void PrintImmediateRow(int imm, const char* name, const EnumParameters* enumParams, bool caps, bool hex = false);
const char* AccessQualifierString(int attr);

void PrintOperands(const OperandParameters& operands, int reservedOperands);

}  // end namespace spv
