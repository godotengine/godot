// Copyright (c) 2014-2019 The Khronos Group Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and/or associated documentation files (the "Materials"),
// to deal in the Materials without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Materials, and to permit persons to whom the
// Materials are furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Materials.
// 
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS KHRONOS
// STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS SPECIFICATIONS AND
// HEADER INFORMATION ARE LOCATED AT https://www.khronos.org/registry/ 
// 
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM,OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS
// IN THE MATERIALS.

#include <assert.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <fstream>

#include "jsoncpp/dist/json/json.h"

#include "jsonToSpirv.h"

namespace spv {

// The set of objects that hold all the instruction/operand
// parameterization information.
InstructionValues InstructionDesc;

// Note: There is no entry for OperandOpcode. Use InstructionDesc instead.
EnumDefinition OperandClassParams[OperandOpcode];
EnumValues SourceLanguageParams;
EnumValues ExecutionModelParams;
EnumValues AddressingParams;
EnumValues MemoryParams;
EnumValues ExecutionModeParams;
EnumValues StorageParams;
EnumValues SamplerAddressingModeParams;
EnumValues SamplerFilterModeParams;
EnumValues ImageFormatParams;
EnumValues ImageChannelOrderParams;
EnumValues ImageChannelDataTypeParams;
EnumValues ImageOperandsParams;
EnumValues FPFastMathParams;
EnumValues FPRoundingModeParams;
EnumValues LinkageTypeParams;
EnumValues DecorationParams;
EnumValues BuiltInParams;
EnumValues DimensionalityParams;
EnumValues FuncParamAttrParams;
EnumValues AccessQualifierParams;
EnumValues GroupOperationParams;
EnumValues LoopControlParams;
EnumValues SelectionControlParams;
EnumValues FunctionControlParams;
EnumValues MemorySemanticsParams;
EnumValues MemoryAccessParams;
EnumValues ScopeParams;
EnumValues KernelEnqueueFlagsParams;
EnumValues KernelProfilingInfoParams;
EnumValues CapabilityParams;

std::pair<bool, std::string> ReadFile(const std::string& path)
{
    std::ifstream fstream(path, std::ios::in);
    if (fstream) {
        std::string contents;
        fstream.seekg(0, std::ios::end);
        contents.reserve((unsigned int)fstream.tellg());
        fstream.seekg(0, std::ios::beg);
        contents.assign((std::istreambuf_iterator<char>(fstream)),
                        std::istreambuf_iterator<char>());
        return std::make_pair(true, contents);
    }
    return std::make_pair(false, "");
}

struct ClassOptionality {
    OperandClass type;
    bool optional;
};

// Converts the |operandKind| and |quantifier| pair used to describe operands
// in the JSON grammar to OperandClass and optionality used in this repo.
ClassOptionality ToOperandClassAndOptionality(const std::string& operandKind, const std::string& quantifier)
{
    assert(quantifier.empty() || quantifier == "?" || quantifier == "*");

    if (operandKind == "IdRef") {
        if (quantifier.empty())
            return {OperandId, false};
        else if (quantifier == "?")
            return {OperandId, true};
        else
            return {OperandVariableIds, false};
    } else if (operandKind == "LiteralInteger") {
        if (quantifier.empty())
            return {OperandLiteralNumber, false};
        if (quantifier == "?")
            return {OperandOptionalLiteral, true};
        else
            return {OperandVariableLiterals, false};
    } else if (operandKind == "LiteralString") {
        if (quantifier.empty())
            return {OperandLiteralString, false};
        else if (quantifier == "?")
            return {OperandLiteralString, true};
        else {
            assert(0 && "this case should not exist");
            return {OperandNone, false};
        }
    } else if (operandKind == "PairLiteralIntegerIdRef") {
        // Used by OpSwitch in the grammar
        return {OperandVariableLiteralId, false};
    } else if (operandKind == "PairIdRefLiteralInteger") {
        // Used by OpGroupMemberDecorate in the grammar
        return {OperandVariableIdLiteral, false};
    } else if (operandKind == "PairIdRefIdRef") {
        // Used by OpPhi in the grammar
        return {OperandVariableIds, false};
    } else {
        OperandClass type = OperandNone;
        if (operandKind == "IdMemorySemantics" || operandKind == "MemorySemantics") {
            type = OperandMemorySemantics;
        } else if (operandKind == "IdScope" || operandKind == "Scope") {
            type = OperandScope;
        } else if (operandKind == "LiteralExtInstInteger") {
            type = OperandLiteralNumber;
        } else if (operandKind == "LiteralSpecConstantOpInteger") {
            type = OperandLiteralNumber;
        } else if (operandKind == "LiteralContextDependentNumber") {
            type = OperandVariableLiterals;
        } else if (operandKind == "SourceLanguage") {
            type = OperandSource;
        } else if (operandKind == "ExecutionModel") {
            type = OperandExecutionModel;
        } else if (operandKind == "AddressingModel") {
            type = OperandAddressing;
        } else if (operandKind == "MemoryModel") {
            type = OperandMemory;
        } else if (operandKind == "ExecutionMode") {
            type = OperandExecutionMode;
        } else if (operandKind == "StorageClass") {
            type = OperandStorage;
        } else if (operandKind == "Dim") {
            type = OperandDimensionality;
        } else if (operandKind == "SamplerAddressingMode") {
            type = OperandSamplerAddressingMode;
        } else if (operandKind == "SamplerFilterMode") {
            type = OperandSamplerFilterMode;
        } else if (operandKind == "ImageFormat") {
            type = OperandSamplerImageFormat;
        } else if (operandKind == "ImageChannelOrder") {
            type = OperandImageChannelOrder;
        } else if (operandKind == "ImageChannelDataType") {
            type = OperandImageChannelDataType;
        } else if (operandKind == "FPRoundingMode") {
            type = OperandFPRoundingMode;
        } else if (operandKind == "LinkageType") {
            type = OperandLinkageType;
        } else if (operandKind == "AccessQualifier") {
            type = OperandAccessQualifier;
        } else if (operandKind == "FunctionParameterAttribute") {
            type = OperandFuncParamAttr;
        } else if (operandKind == "Decoration") {
            type = OperandDecoration;
        } else if (operandKind == "BuiltIn") {
            type = OperandBuiltIn;
        } else if (operandKind == "GroupOperation") {
            type = OperandGroupOperation;
        } else if (operandKind == "KernelEnqueueFlags") {
            type = OperandKernelEnqueueFlags;
        } else if (operandKind == "KernelProfilingInfo") {
            type = OperandKernelProfilingInfo;
        } else if (operandKind == "Capability") {
            type = OperandCapability;
        } else if (operandKind == "ImageOperands") {
            type = OperandImageOperands;
        } else if (operandKind == "FPFastMathMode") {
            type = OperandFPFastMath;
        } else if (operandKind == "SelectionControl") {
            type = OperandSelect;
        } else if (operandKind == "LoopControl") {
            type = OperandLoop;
        } else if (operandKind == "FunctionControl") {
            type = OperandFunction;
        } else if (operandKind == "MemoryAccess") {
            type = OperandMemoryAccess;
        }

        if (type == OperandNone) {
            std::cerr << "Unhandled operand kind found: " << operandKind << std::endl;
            exit(1);
        }
        return {type, !quantifier.empty()};
    }
}

bool IsTypeOrResultId(const std::string& str, bool* isType, bool* isResult)
{
    if (str == "IdResultType")
        return *isType = true;
    if (str == "IdResult")
        return *isResult = true;
    return false;
}

// Given a number string, returns the position of the only bits set in the number.
// So it requires the number is a power of two.
unsigned int NumberStringToBit(const std::string& str)
{
    char* parseEnd;
    unsigned int value = (unsigned int)std::strtol(str.c_str(), &parseEnd, 16);
    assert(!(value & (value - 1)) && "input number is not a power of 2");
    unsigned int bit = 0;
    for (; value; value >>= 1) ++bit;
    return bit;
}

bool ExcludeInstruction(unsigned op, bool buildingHeaders)
{
    // Some instructions in the grammar don't need to be reflected
    // in the specification.

    if (buildingHeaders)
        return false;

    if (op >= 5699 /* OpVmeImageINTEL */ && op <= 5816 /* OpSubgroupAvcSicGetInterRawSadsINTEL */)
        return true;

    return false;
}

void jsonToSpirv(const std::string& jsonPath, bool buildingHeaders)
{
    // only do this once.
    static bool initialized = false;
    if (initialized)
        return;
    initialized = true;

    // Read the JSON grammar file.
    bool fileReadOk = false;
    std::string content;
    std::tie(fileReadOk, content) = ReadFile(jsonPath);
    if (!fileReadOk) {
        std::cerr << "Failed to read JSON grammar file: "
                  << jsonPath << std::endl;
        exit(1);
    }

    // Decode the JSON grammar file.
    Json::Reader reader;
    Json::Value root;
    if (!reader.parse(content, root)) {
        std::cerr << "Failed to parse JSON grammar:\n"
                  << reader.getFormattedErrorMessages();
        exit(1);
    }

    // Layouts for all instructions.

    // A lambda for returning capabilities from a JSON object as strings.
    const auto getCaps = [](const Json::Value& object) {
        EnumCaps result;
        const auto& caps = object["capabilities"];
        if (!caps.empty()) {
            assert(caps.isArray());
            for (const auto& cap : caps) {
                result.emplace_back(cap.asString());
            }
        }
        return result;
    };

    // A lambda for returning extensions from a JSON object as strings.
    const auto getExts = [](const Json::Value& object) {
        Extensions result;
        const auto& exts = object["extensions"];
        if (!exts.empty()) {
            assert(exts.isArray());
            for (const auto& ext : exts) {
                result.emplace_back(ext.asString());
            }
        }
        return result;
    };

    const Json::Value insts = root["instructions"];
    for (const auto& inst : insts) {
        const unsigned int opcode = inst["opcode"].asUInt();
        if (ExcludeInstruction(opcode, buildingHeaders))
            continue;
        const std::string name = inst["opname"].asString();
        EnumCaps caps = getCaps(inst);
        std::string version = inst["version"].asString();
        Extensions exts = getExts(inst);
        OperandParameters operands;
        bool defResultId = false;
        bool defTypeId = false;
        for (const auto& operand : inst["operands"]) {
            const std::string kind = operand["kind"].asString();
            const std::string quantifier = operand.get("quantifier", "").asString();
            const std::string doc = operand.get("name", "").asString();
            if (!IsTypeOrResultId(kind, &defTypeId, &defResultId)) {
                const auto p = ToOperandClassAndOptionality(kind, quantifier);
                operands.push(p.type, doc, p.optional);
            }
        }
        InstructionDesc.emplace_back(
            std::move(EnumValue(opcode, name,
                                std::move(caps), std::move(version), std::move(exts),
                                std::move(operands))),
            defTypeId, defResultId);
    }

    // Specific additional context-dependent operands

    // Populate dest with EnumValue objects constructed from source.
    const auto populateEnumValues = [&getCaps,&getExts](EnumValues* dest, const Json::Value& source, bool bitEnum) {
        // A lambda for determining the numeric value to be used for a given
        // enumerant in JSON form, and whether that value is a 0 in a bitfield.
        auto getValue = [&bitEnum](const Json::Value& enumerant) {
            std::pair<unsigned, bool> result{0u,false};
            if (!bitEnum) {
                result.first = enumerant["value"].asUInt();
            } else {
                const unsigned int bit = NumberStringToBit(enumerant["value"].asString());
                if (bit == 0)
                    result.second = true;
                else
                    result.first = bit - 1;  // This is the *shift* amount.
            }
            return result;
        };

        for (const auto& enumerant : source["enumerants"]) {
            unsigned value;
            bool skip_zero_in_bitfield;
            std::tie(value, skip_zero_in_bitfield) = getValue(enumerant);
            if (skip_zero_in_bitfield)
                continue;
            EnumCaps caps(getCaps(enumerant));
            std::string version = enumerant["version"].asString();
            Extensions exts(getExts(enumerant));
            OperandParameters params;
            const Json::Value& paramsJson = enumerant["parameters"];
            if (!paramsJson.empty()) {  // This enumerant has parameters.
                assert(paramsJson.isArray());
                for (const auto& param : paramsJson) {
                    const std::string kind = param["kind"].asString();
                    const std::string doc = param.get("name", "").asString();
                    const auto p = ToOperandClassAndOptionality(kind, ""); // All parameters are required!
                    params.push(p.type, doc);
                }
            }
            dest->emplace_back(
                value, enumerant["enumerant"].asString(),
                std::move(caps), std::move(version), std::move(exts), std::move(params));
        }
    };

    const auto establishOperandClass = [&populateEnumValues](
            const std::string& enumName, spv::OperandClass operandClass,
            spv::EnumValues* enumValues, const Json::Value& operandEnum, const std::string& category) {
        assert(category == "BitEnum" || category == "ValueEnum");
        bool bitEnum = (category == "BitEnum");
        populateEnumValues(enumValues, operandEnum, bitEnum);
        OperandClassParams[operandClass].set(enumName, enumValues, bitEnum);
    };

    const Json::Value operandEnums = root["operand_kinds"];
    for (const auto& operandEnum : operandEnums) {
        const std::string enumName = operandEnum["kind"].asString();
        const std::string category = operandEnum["category"].asString();
        if (enumName == "SourceLanguage") {
            establishOperandClass(enumName, OperandSource, &SourceLanguageParams, operandEnum, category);
        } else if (enumName == "Decoration") {
            establishOperandClass(enumName, OperandDecoration, &DecorationParams, operandEnum, category);
        } else if (enumName == "ExecutionMode") {
            establishOperandClass(enumName, OperandExecutionMode, &ExecutionModeParams, operandEnum, category);
        } else if (enumName == "Capability") {
            establishOperandClass(enumName, OperandCapability, &CapabilityParams, operandEnum, category);
        } else if (enumName == "AddressingModel") {
            establishOperandClass(enumName, OperandAddressing, &AddressingParams, operandEnum, category);
        } else if (enumName == "MemoryModel") {
            establishOperandClass(enumName, OperandMemory, &MemoryParams, operandEnum, category);
        } else if (enumName == "MemorySemantics") {
            establishOperandClass(enumName, OperandMemorySemantics, &MemorySemanticsParams, operandEnum, category);
        } else if (enumName == "ExecutionModel") {
            establishOperandClass(enumName, OperandExecutionModel, &ExecutionModelParams, operandEnum, category);
        } else if (enumName == "StorageClass") {
            establishOperandClass(enumName, OperandStorage, &StorageParams, operandEnum, category);
        } else if (enumName == "SamplerAddressingMode") {
            establishOperandClass(enumName, OperandSamplerAddressingMode, &SamplerAddressingModeParams, operandEnum, category);
        } else if (enumName == "SamplerFilterMode") {
            establishOperandClass(enumName, OperandSamplerFilterMode, &SamplerFilterModeParams, operandEnum, category);
        } else if (enumName == "ImageFormat") {
            establishOperandClass(enumName, OperandSamplerImageFormat, &ImageFormatParams, operandEnum, category);
        } else if (enumName == "ImageChannelOrder") {
            establishOperandClass(enumName, OperandImageChannelOrder, &ImageChannelOrderParams, operandEnum, category);
        } else if (enumName == "ImageChannelDataType") {
            establishOperandClass(enumName, OperandImageChannelDataType, &ImageChannelDataTypeParams, operandEnum, category);
        } else if (enumName == "ImageOperands") {
            establishOperandClass(enumName, OperandImageOperands, &ImageOperandsParams, operandEnum, category);
        } else if (enumName == "FPFastMathMode") {
            establishOperandClass(enumName, OperandFPFastMath, &FPFastMathParams, operandEnum, category);
        } else if (enumName == "FPRoundingMode") {
            establishOperandClass(enumName, OperandFPRoundingMode, &FPRoundingModeParams, operandEnum, category);
        } else if (enumName == "LinkageType") {
            establishOperandClass(enumName, OperandLinkageType, &LinkageTypeParams, operandEnum, category);
        } else if (enumName == "FunctionParameterAttribute") {
            establishOperandClass(enumName, OperandFuncParamAttr, &FuncParamAttrParams, operandEnum, category);
        } else if (enumName == "AccessQualifier") {
            establishOperandClass(enumName, OperandAccessQualifier, &AccessQualifierParams, operandEnum, category);
        } else if (enumName == "BuiltIn") {
            establishOperandClass(enumName, OperandBuiltIn, &BuiltInParams, operandEnum, category);
        } else if (enumName == "SelectionControl") {
            establishOperandClass(enumName, OperandSelect, &SelectionControlParams, operandEnum, category);
        } else if (enumName == "LoopControl") {
            establishOperandClass(enumName, OperandLoop, &LoopControlParams, operandEnum, category);
        } else if (enumName == "FunctionControl") {
            establishOperandClass(enumName, OperandFunction, &FunctionControlParams, operandEnum, category);
        } else if (enumName == "Dim") {
            establishOperandClass(enumName, OperandDimensionality, &DimensionalityParams, operandEnum, category);
        } else if (enumName == "MemoryAccess") {
            establishOperandClass(enumName, OperandMemoryAccess, &MemoryAccessParams, operandEnum, category);
        } else if (enumName == "Scope") {
            establishOperandClass(enumName, OperandScope, &ScopeParams, operandEnum, category);
        } else if (enumName == "GroupOperation") {
            establishOperandClass(enumName, OperandGroupOperation, &GroupOperationParams, operandEnum, category);
        } else if (enumName == "KernelEnqueueFlags") {
            establishOperandClass(enumName, OperandKernelEnqueueFlags, &KernelEnqueueFlagsParams, operandEnum, category);
        } else if (enumName == "KernelProfilingInfo") {
            establishOperandClass(enumName, OperandKernelProfilingInfo, &KernelProfilingInfoParams, operandEnum, category);
        }
    }
}

};  // end namespace spv
