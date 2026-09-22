//
// Copyright(C) 2021 Advanced Micro Devices, Inc.
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
// GL_EXT_spirv_intrinsics
//
#include "../Include/intermediate.h"
#include "../Include/SpirvIntrinsics.h"
#include "../Include/Types.h"
#include "ParseHelper.h"

namespace glslang {

bool TSpirvTypeParameter::operator==(const TSpirvTypeParameter& rhs) const
{
    if (getAsConstant() != nullptr)
        return getAsConstant()->getConstArray() == rhs.getAsConstant()->getConstArray();

    assert(getAsType() != nullptr);
    return *getAsType() == *rhs.getAsType();
}

//
// Handle SPIR-V requirements
//
TSpirvRequirement* TParseContext::makeSpirvRequirement(const TSourceLoc& loc, const TString& name,
                                                       const TIntermAggregate* extensions,
                                                       const TIntermAggregate* capabilities)
{
    TSpirvRequirement* spirvReq = new TSpirvRequirement;

    if (name == "extensions") {
        assert(extensions);
        for (auto extension : extensions->getSequence()) {
            assert(extension->getAsConstantUnion());
            spirvReq->extensions.insert(*extension->getAsConstantUnion()->getConstArray()[0].getSConst());
        }
    } else if (name == "capabilities") {
        assert(capabilities);
        for (auto capability : capabilities->getSequence()) {
            assert(capability->getAsConstantUnion());
            spirvReq->capabilities.insert(capability->getAsConstantUnion()->getConstArray()[0].getIConst());
        }
    } else
        error(loc, "unknown SPIR-V requirement", name.c_str(), "");

    return spirvReq;
}

TSpirvRequirement* TParseContext::mergeSpirvRequirements(const TSourceLoc& loc, TSpirvRequirement* spirvReq1,
                                                         TSpirvRequirement* spirvReq2)
{
    // Merge the second SPIR-V requirement to the first one
    if (!spirvReq2->extensions.empty()) {
        if (spirvReq1->extensions.empty())
            spirvReq1->extensions = spirvReq2->extensions;
        else
            error(loc, "too many SPIR-V requirements", "extensions", "");
    }

    if (!spirvReq2->capabilities.empty()) {
        if (spirvReq1->capabilities.empty())
            spirvReq1->capabilities = spirvReq2->capabilities;
        else
            error(loc, "too many SPIR-V requirements", "capabilities", "");
    }

    return spirvReq1;
}

void TIntermediate::insertSpirvRequirement(const TSpirvRequirement* spirvReq)
{
    if (!spirvRequirement)
        spirvRequirement = new TSpirvRequirement;

    for (auto extension : spirvReq->extensions)
        spirvRequirement->extensions.insert(extension);

    for (auto capability : spirvReq->capabilities)
        spirvRequirement->capabilities.insert(capability);
}

//
// Handle SPIR-V execution modes
//
void TIntermediate::insertSpirvExecutionMode(int executionMode, const TIntermAggregate* args)
{
    if (!spirvExecutionMode)
        spirvExecutionMode = new TSpirvExecutionMode;

    TVector<const TIntermConstantUnion*> extraOperands;
    if (args) {
        for (auto arg : args->getSequence()) {
            auto extraOperand = arg->getAsConstantUnion();
            assert(extraOperand != nullptr);
            extraOperands.push_back(extraOperand);
        }
    }
    spirvExecutionMode->modes[executionMode] = extraOperands;
}

void TIntermediate::insertSpirvExecutionModeId(int executionMode, const TIntermAggregate* args)
{
    if (!spirvExecutionMode)
        spirvExecutionMode = new TSpirvExecutionMode;

    assert(args);
    TVector<const TIntermTyped*> extraOperands;

    for (auto arg : args->getSequence()) {
        auto extraOperand = arg->getAsTyped();
        assert(extraOperand != nullptr && extraOperand->getQualifier().isConstant());
        extraOperands.push_back(extraOperand);
    }
    spirvExecutionMode->modeIds[executionMode] = extraOperands;
}

//
// Handle SPIR-V decorate qualifiers
//
void TQualifier::setSpirvDecorate(int decoration, const TIntermAggregate* args)
{
    if (!spirvDecorate)
        spirvDecorate = new TSpirvDecorate;

    TVector<const TIntermConstantUnion*> extraOperands;
    if (args) {
        for (auto arg : args->getSequence()) {
            auto extraOperand = arg->getAsConstantUnion();
            assert(extraOperand != nullptr);
            extraOperands.push_back(extraOperand);
        }
    }
    spirvDecorate->decorates[decoration] = extraOperands;
}

void TQualifier::setSpirvDecorateId(int decoration, const TIntermAggregate* args)
{
    if (!spirvDecorate)
        spirvDecorate = new TSpirvDecorate;

    assert(args);
    TVector<const TIntermTyped*> extraOperands;
    for (auto arg : args->getSequence()) {
        auto extraOperand = arg->getAsTyped();
        assert(extraOperand != nullptr);
        extraOperands.push_back(extraOperand);
    }
    spirvDecorate->decorateIds[decoration] = extraOperands;
}

void TQualifier::setSpirvDecorateString(int decoration, const TIntermAggregate* args)
{
    if (!spirvDecorate)
        spirvDecorate = new TSpirvDecorate;

    assert(args);
    TVector<const TIntermConstantUnion*> extraOperands;
    for (auto arg : args->getSequence()) {
        auto extraOperand = arg->getAsConstantUnion();
        assert(extraOperand != nullptr);
        extraOperands.push_back(extraOperand);
    }
    spirvDecorate->decorateStrings[decoration] = extraOperands;
}

TString TQualifier::getSpirvDecorateQualifierString() const
{
    assert(spirvDecorate);

    TString qualifierString;

    const auto appendFloat = [&](float f) { qualifierString.append(std::to_string(f).c_str()); };
    const auto appendInt = [&](int i) { qualifierString.append(std::to_string(i).c_str()); };
    const auto appendUint = [&](unsigned int u) { qualifierString.append(std::to_string(u).c_str()); };
    const auto appendBool = [&](bool b) { qualifierString.append(std::to_string(b).c_str()); };
    const auto appendStr = [&](const char* s) { qualifierString.append(s); };

    const auto appendDecorate = [&](const TIntermTyped* constant) {
        if (constant->getAsConstantUnion()) {
            auto& constArray = constant->getAsConstantUnion()->getConstArray();
            if (constant->getBasicType() == EbtFloat) {
                float value = static_cast<float>(constArray[0].getDConst());
                appendFloat(value);
            } else if (constant->getBasicType() == EbtInt) {
                int value = constArray[0].getIConst();
                appendInt(value);
            } else if (constant->getBasicType() == EbtUint) {
                unsigned value = constArray[0].getUConst();
                appendUint(value);
            } else if (constant->getBasicType() == EbtBool) {
                bool value = constArray[0].getBConst();
                appendBool(value);
            } else if (constant->getBasicType() == EbtString) {
                const TString* value = constArray[0].getSConst();
                appendStr(value->c_str());
            } else
                assert(0);
        } else {
            assert(constant->getAsSymbolNode());
            appendStr(constant->getAsSymbolNode()->getName().c_str());
        }
    };

    for (auto& decorate : spirvDecorate->decorates) {
        appendStr("spirv_decorate(");
        appendInt(decorate.first);
        for (auto extraOperand : decorate.second) {
            appendStr(", ");
            appendDecorate(extraOperand);
        }
        appendStr(") ");
    }

    for (auto& decorateId : spirvDecorate->decorateIds) {
        appendStr("spirv_decorate_id(");
        appendInt(decorateId.first);
        for (auto extraOperand : decorateId.second) {
            appendStr(", ");
            appendDecorate(extraOperand);
        }
        appendStr(") ");
    }

    for (auto& decorateString : spirvDecorate->decorateStrings) {
        appendStr("spirv_decorate_string(");
        appendInt(decorateString.first);
        for (auto extraOperand : decorateString.second) {
            appendStr(", ");
            appendDecorate(extraOperand);
        }
        appendStr(") ");
    }

    return qualifierString;
}

//
// Handle SPIR-V type specifiers
//
void TPublicType::setSpirvType(const TSpirvInstruction& spirvInst, const TSpirvTypeParameters* typeParams)
{
    if (!spirvType)
        spirvType = new TSpirvType;

    basicType = EbtSpirvType;
    spirvType->spirvInst = spirvInst;
    if (typeParams)
        spirvType->typeParams = *typeParams;
}

TSpirvTypeParameters* TParseContext::makeSpirvTypeParameters(const TSourceLoc& loc, const TIntermConstantUnion* constant)
{
    TSpirvTypeParameters* spirvTypeParams = new TSpirvTypeParameters;
    if (constant->getBasicType() != EbtFloat &&
        constant->getBasicType() != EbtInt &&
        constant->getBasicType() != EbtUint &&
        constant->getBasicType() != EbtBool &&
        constant->getBasicType() != EbtString)
        error(loc, "this type not allowed", constant->getType().getBasicString(), "");
    else
        spirvTypeParams->push_back(TSpirvTypeParameter(constant));

    return spirvTypeParams;
}

TSpirvTypeParameters* TParseContext::makeSpirvTypeParameters(const TSourceLoc& /* loc */,
                                                             const TPublicType& type)
{
    TSpirvTypeParameters* spirvTypeParams = new TSpirvTypeParameters;
    spirvTypeParams->push_back(TSpirvTypeParameter(new TType(type)));
    return spirvTypeParams;
}

TSpirvTypeParameters* TParseContext::mergeSpirvTypeParameters(TSpirvTypeParameters* spirvTypeParams1, TSpirvTypeParameters* spirvTypeParams2)
{
    // Merge SPIR-V type parameters of the second one to the first one
    for (const auto& spirvTypeParam : *spirvTypeParams2)
        spirvTypeParams1->push_back(spirvTypeParam);
    return spirvTypeParams1;
}

//
// Handle SPIR-V instruction qualifiers
//
TSpirvInstruction* TParseContext::makeSpirvInstruction(const TSourceLoc& loc, const TString& name, const TString& value)
{
    TSpirvInstruction* spirvInst = new TSpirvInstruction;
    if (name == "set")
        spirvInst->set = value;
    else
        error(loc, "unknown SPIR-V instruction qualifier", name.c_str(), "");

    return spirvInst;
}

TSpirvInstruction* TParseContext::makeSpirvInstruction(const TSourceLoc& loc, const TString& name, int value)
{
    TSpirvInstruction* spirvInstuction = new TSpirvInstruction;
    if (name == "id")
        spirvInstuction->id = value;
    else
        error(loc, "unknown SPIR-V instruction qualifier", name.c_str(), "");

    return spirvInstuction;
}

TSpirvInstruction* TParseContext::mergeSpirvInstruction(const TSourceLoc& loc, TSpirvInstruction* spirvInst1, TSpirvInstruction* spirvInst2)
{
    // Merge qualifiers of the second SPIR-V instruction to those of the first one
    if (!spirvInst2->set.empty()) {
        if (spirvInst1->set.empty())
            spirvInst1->set = spirvInst2->set;
        else
            error(loc, "too many SPIR-V instruction qualifiers", "spirv_instruction", "(set)");
    }

    if (spirvInst2->id != -1) {
        if (spirvInst1->id == -1)
            spirvInst1->id = spirvInst2->id;
        else
            error(loc, "too many SPIR-V instruction qualifiers", "spirv_instruction", "(id)");
    }

    return spirvInst1;
}

} // end namespace glslang
