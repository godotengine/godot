// Copyright (c) 2014-2024 The Khronos Group Inc.
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
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <fstream>

#include "jsoncpp/dist/json/json.h"

#include "jsonToSpirv.h"

namespace {
// Returns true if the given string is a valid SPIR-V version.
bool validSpirvVersionString(const std::string s) {
  return
  s == "1.0" ||
  s == "1.1" ||
  s == "1.2" ||
  s == "1.3" ||
  s == "1.4" ||
  s == "1.5" ||
  s == "1.6";
}

// Returns true if the given string is a valid version
// specifier in the grammar file.
bool validSpirvVersionStringSpecifier(const std::string s) {
  return s == "None" || validSpirvVersionString(s);
}
}  // anonymous namespace

namespace spv {

bool IsLegacyDoublyEnabledInstruction(const std::string& instruction) {
  static std::unordered_set<std::string> allowed = {
      "OpSubgroupBallotKHR",
      "OpSubgroupFirstInvocationKHR",
      "OpSubgroupAllKHR",
      "OpSubgroupAnyKHR",
      "OpSubgroupAllEqualKHR",
      "OpSubgroupReadInvocationKHR",
      "OpTraceRayKHR",
      "OpExecuteCallableKHR",
      "OpConvertUToAccelerationStructureKHR",
      "OpIgnoreIntersectionKHR",
      "OpTerminateRayKHR",
      "OpTypeRayQueryKHR",
      "OpRayQueryInitializeKHR",
      "OpRayQueryTerminateKHR",
      "OpRayQueryGenerateIntersectionKHR",
      "OpRayQueryConfirmIntersectionKHR",
      "OpRayQueryProceedKHR",
      "OpRayQueryGetIntersectionTypeKHR",
      "OpGroupIAddNonUniformAMD",
      "OpGroupFAddNonUniformAMD",
      "OpGroupFMinNonUniformAMD",
      "OpGroupUMinNonUniformAMD",
      "OpGroupSMinNonUniformAMD",
      "OpGroupFMaxNonUniformAMD",
      "OpGroupUMaxNonUniformAMD",
      "OpGroupSMaxNonUniformAMD",
      "OpFragmentMaskFetchAMD",
      "OpFragmentFetchAMD",
      "OpImageSampleFootprintNV",
      "OpGroupNonUniformPartitionNV",
      "OpWritePackedPrimitiveIndices4x8NV",
      "OpReportIntersectionNV",
      "OpReportIntersectionKHR",
      "OpIgnoreIntersectionNV",
      "OpTerminateRayNV",
      "OpTraceNV",
      "OpTraceMotionNV",
      "OpTraceRayMotionNV",
      "OpTypeAccelerationStructureNV",
      "OpTypeAccelerationStructureKHR",
      "OpExecuteCallableNV",
      "OpTypeCooperativeMatrixNV",
      "OpCooperativeMatrixLoadNV",
      "OpCooperativeMatrixStoreNV",
      "OpCooperativeMatrixMulAddNV",
      "OpCooperativeMatrixLengthNV",
      "OpBeginInvocationInterlockEXT",
      "OpEndInvocationInterlockEXT",
      "OpIsHelperInvocationEXT",
      "OpConstantFunctionPointerINTEL",
      "OpFunctionPointerCallINTEL",
      "OpAssumeTrueKHR",
      "OpExpectKHR",
      "OpLoopControlINTEL",
      "OpAliasDomainDeclINTEL",
      "OpAliasScopeDeclINTEL",
      "OpAliasScopeListDeclINTEL",
      "OpReadPipeBlockingINTEL",
      "OpWritePipeBlockingINTEL",
      "OpFPGARegINTEL",
      "OpRayQueryGetRayTMinKHR",
      "OpRayQueryGetRayFlagsKHR",
      "OpRayQueryGetIntersectionTKHR",
      "OpRayQueryGetIntersectionInstanceCustomIndexKHR",
      "OpRayQueryGetIntersectionInstanceIdKHR",
      "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR",
      "OpRayQueryGetIntersectionGeometryIndexKHR",
      "OpRayQueryGetIntersectionPrimitiveIndexKHR",
      "OpRayQueryGetIntersectionBarycentricsKHR",
      "OpRayQueryGetIntersectionFrontFaceKHR",
      "OpRayQueryGetIntersectionCandidateAABBOpaqueKHR",
      "OpRayQueryGetIntersectionObjectRayDirectionKHR",
      "OpRayQueryGetIntersectionObjectRayOriginKHR",
      "OpRayQueryGetWorldRayDirectionKHR",
      "OpRayQueryGetWorldRayOriginKHR",
      "OpRayQueryGetIntersectionObjectToWorldKHR",
      "OpRayQueryGetIntersectionWorldToObjectKHR",
      "OpAtomicFAddEXT",
  };
  return allowed.count(instruction) != 0;
}

bool EnumValue::IsValid(OperandClass oc, const std::string& context) const
{
  bool result = true;
  if (firstVersion.empty()) {
    std::cerr << "Error: " << context << " " << name << " \"version\" must be set, probably to \"None\"" << std::endl;
    result = false;
  } else if (!validSpirvVersionStringSpecifier(firstVersion)) {
    std::cerr << "Error: " << context << " " << name << " \"version\" is invalid: " << firstVersion << std::endl;
    result = false;
  }
  if (!lastVersion.empty() && !validSpirvVersionString(lastVersion)) {
    std::cerr << "Error: " << context << " " << name << " \"lastVersion\" is invalid: " << lastVersion << std::endl;
    result = false;
  }

  // When a feature is introduced by an extension, the firstVersion is set to
  // "None". There are three cases:
  // -  A new capability should be guarded/enabled by the extension
  // -  A new instruction should be:
  //      - Guarded/enabled by a new capability.
  //      - Not enabled by *both* a capability and an extension.
  //        There are many existing instructions that are already like this,
  //        and we grandparent them as allowed.
  // -  Other enums fall into two cases:
  //    1. The enum is part of a new operand kind introduced by the extension.
  //       In this case we rely on transitivity: The use of the operand occurs
  //       in a new instruction that itself is guarded; or as the operand of
  //       another operand that itself is (recursively) guarded.
  //    2. The enum is a new case in an existing operand kind.  This case
  //       should be guarded by a capability.  However, we do not check this
  //       here.  Checking it requires more context than we have here.
  if (oc == OperandOpcode) {
    const bool instruction_unusable =
        (firstVersion == "None") && extensions.empty() && capabilities.empty();
    if (instruction_unusable) {
      std::cerr << "Error: " << context << " " << name << " is not usable: "
                << "its version is set to \"None\", and it is not enabled by a "
                << "capability or extension. Guard it with a capability."
                << std::endl;
      result = false;
    }
    // Complain if an instruction is not in any core version and also enabled by
    // both an extension and a capability.
    // It's important to check the "not in any core version" case, because,
    // for example, OpTerminateInvocation is in SPIR-V 1.6 *and* enabled by an
    // extension, and guarded by the Shader capability.
    const bool instruction_doubly_enabled = (firstVersion == "None") &&
                                            !extensions.empty() &&
                                            !capabilities.empty();
    if (instruction_doubly_enabled && !IsLegacyDoublyEnabledInstruction(name)) {
      std::cerr << "Error: " << context << " " << name << " is doubly-enabled: "
                << "it is enabled by both a capability and an extension. "
                << "Guard it with a capability only." << std::endl;
      result = false;
    }
  }
  if (oc == OperandCapability) {
    // If capability X lists capabilities Y and Z, then Y and Z are *enabled*
    // when X is enabled. They are not *guards* on X's use.
    // Only versions and extensions can guard a capability.
    const bool capability_unusable =
        (firstVersion == "None") && extensions.empty();
    if (capability_unusable) {
      std::cerr << "Error: " << context << " " << name << " is not usable: "
                << "its version is set to \"None\", and it is not enabled by "
                << "an extension. Guard it with an extension." << std::endl;
      result = false;
    }
  }

  return result;
}

// The set of objects that hold all the instruction/operand
// parameterization information.
InstructionValues InstructionDesc;

// The ordered list (in printing order) of printing classes
// (specification subsections).
PrintingClasses InstructionPrintingClasses;

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
EnumValues FPDenormModeParams;
EnumValues FPOperationModeParams;
EnumValues QuantizationModesParams;
EnumValues OverflowModesParams;
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
EnumValues RayFlagsParams;
EnumValues RayQueryIntersectionParams;
EnumValues RayQueryCommittedIntersectionTypeParams;
EnumValues RayQueryCandidateIntersectionTypeParams;
EnumValues FragmentShadingRateParams;
EnumValues PackedVectorFormatParams;
EnumValues CooperativeMatrixOperandsParams;
EnumValues CooperativeMatrixLayoutParams;
EnumValues CooperativeMatrixUseParams;
EnumValues InitializationModeQualifierParams;
EnumValues HostAccessQualifierParams;
EnumValues LoadCacheControlParams;
EnumValues StoreCacheControlParams;
EnumValues NamedMaximumNumberOfRegistersParams;
EnumValues RawAccessChainOperandsParams;
EnumValues FPEncodingParams;

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
            return {OperandOptionalLiteralStrings, false};
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
            type = OperandAnySizeLiteralNumber;
        } else if (operandKind == "LiteralFloat") {
            type = OperandLiteralNumber;
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
        } else if (operandKind == "FPDenormMode") {
            type = OperandFPDenormMode;
        } else if (operandKind == "FPOperationMode") {
            type = OperandFPOperationMode;
        } else if (operandKind == "QuantizationModes") {
            type = OperandQuantizationModes;
        } else if (operandKind == "OverflowModes") {
            type = OperandOverflowModes;
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
            type = OperandMemoryOperands;
        } else if (operandKind == "RayFlags") {
            type = OperandRayFlags;
        } else if (operandKind == "RayQueryIntersection") {
            type = OperandRayQueryIntersection;
        } else if (operandKind == "RayQueryCommittedIntersectionType") {
            type = OperandRayQueryCommittedIntersectionType;
        } else if (operandKind == "RayQueryCandidateIntersectionType") {
            type = OperandRayQueryCandidateIntersectionType;
        } else if (operandKind == "FragmentShadingRate") {
            type = OperandFragmentShadingRate;
        } else if (operandKind == "PackedVectorFormat") {
            type = OperandPackedVectorFormat;
        } else if (operandKind == "CooperativeMatrixOperands") {
            type = OperandCooperativeMatrixOperands;
        } else if (operandKind == "CooperativeMatrixLayout") {
            type = OperandCooperativeMatrixLayout;
        } else if (operandKind == "CooperativeMatrixUse") {
            type = OperandCooperativeMatrixUse;
        } else if (operandKind == "InitializationModeQualifier") {
            type = OperandInitializationModeQualifier;
        } else if (operandKind == "HostAccessQualifier") {
            type = OperandHostAccessQualifier;
        } else if (operandKind == "LoadCacheControl") {
            type = OperandLoadCacheControl;
        } else if (operandKind == "StoreCacheControl") {
            type = OperandStoreCacheControl;
        } else if (operandKind == "NamedMaximumNumberOfRegisters") {
            type = OperandNamedMaximumNumberOfRegisters;
        } else if (operandKind == "RawAccessChainOperands") {
            type = OperandRawAccessChainOperands;
        } else if (operandKind == "FPEncoding") {
            type = OperandFPEncoding;
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

// Given two pairs (name and in core) compares if the order is correct for naming
// conventions. The conventions are:
// * Core
// * KHR
// * EXT
// * Vendor (no preference between vendors)
//
// Returns true if the order is valid.
bool SuffixComparison(const std::string& prev, bool prevCore,
                      const std::string& cur, bool curCore)
{
  // Duplicate entry
  if (prev == cur) return false;

  if (prevCore) return true;
  if (curCore) return false;

  // Both are suffixed names.
  const bool prevKHR = prev.substr(prev.size() - 3) == "KHR";
  const bool prevEXT = prev.substr(prev.size() - 3) == "EXT";
  const bool curKHR = cur.substr(cur.size() - 3) == "KHR";
  const bool curEXT = cur.substr(cur.size() - 3) == "EXT";

  if (prevKHR) return true;
  if (curKHR) return false;
  if (prevEXT) return true;
  if (curEXT) return false;

  return true;
}

void jsonToSpirv(const std::string& jsonPath, bool buildingHeaders)
{
    // only do this once.
    static bool initialized = false;
    if (initialized)
        return;
    initialized = true;

    size_t errorCount = 0;

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

    // set up the printing classes
    std::unordered_set<std::string> tags;  // short-lived local for error checking below
    const Json::Value printingClasses = root["instruction_printing_class"];
    for (const auto& printingClass : printingClasses) {
        if (printingClass["tag"].asString().size() > 0)
            tags.insert(printingClass["tag"].asString()); // just for error checking
        else {
            std::cerr << "Error: each instruction_printing_class requires a non-empty \"tag\"" << std::endl;
            std::exit(1);
        }
        if (buildingHeaders || printingClass["tag"].asString() != "@exclude") {
            InstructionPrintingClasses.push_back({printingClass["tag"].asString(),
                                                  printingClass["heading"].asString()});
        }
    }

    // process the instructions
    const Json::Value insts = root["instructions"];
    unsigned maxOpcode = 0;
    std::string maxName = "";
    bool maxCore = false;
    bool firstOpcode = true;
    for (const auto& inst : insts) {
        const auto printingClass = inst["class"].asString();
        if (printingClass.size() == 0) {
            std::cerr << "Error: " << inst["opname"].asString()
                      << " requires a non-empty printing \"class\" tag" << std::endl;
            std::exit(1);
        }
        if (!buildingHeaders && printingClass == "@exclude")
            continue;
        if (tags.find(printingClass) == tags.end()) {
            std::cerr << "Error: " << inst["opname"].asString()
                      << " requires a \"class\" declared as a \"tag\" in \"instruction printing_class\""
                      << std::endl;
            std::exit(1);
        }
        const auto opcode = inst["opcode"].asUInt();
        const std::string name = inst["opname"].asString();
        std::string version = inst["version"].asString();
        if (firstOpcode) {
          maxOpcode = opcode;
          maxName = name;
          maxCore = version != "None";
          firstOpcode = false;
        } else {
          if (maxOpcode > opcode) {
            std::cerr << "Error: " << name
                      << " is out of order. It follows the instruction with opcode " << maxOpcode
                      << std::endl;
            std::exit(1);
          } else if (maxOpcode == opcode &&
                     !SuffixComparison(maxName, maxCore, name,
                                       version != "None")) {
            std::cerr << "Error: " << name
                      << " is out of order. It follows alias " << maxName
                      << std::endl;
            std::exit(1);
          } else {
            maxOpcode = opcode;
            maxName = name;
            maxCore = version != "None";
          }
        }
        EnumCaps caps = getCaps(inst);
        std::string lastVersion = inst["lastVersion"].asString();
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
                                std::move(caps), std::move(version), std::move(lastVersion), std::move(exts),
                                std::move(operands))),
             printingClass, defTypeId, defResultId);
        if (!InstructionDesc.back().IsValid(OperandOpcode, "instruction")) {
          errorCount++;
        }
    }

    // Specific additional context-dependent operands

    // Populate dest with EnumValue objects constructed from source.
    const auto populateEnumValues = [&getCaps,&getExts,&errorCount](EnumValues* dest, const Json::Value& source, bool bitEnum) {
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

        unsigned maxValue = 0;
        std::string maxName = "";
        bool maxCore = false;
        bool firstValue = true;
        for (const auto& enumerant : source["enumerants"]) {
            unsigned value;
            bool skip_zero_in_bitfield;
            std::tie(value, skip_zero_in_bitfield) = getValue(enumerant);
            std::string name = enumerant["enumerant"].asString();
            std::string version = enumerant["version"].asString();
            if (skip_zero_in_bitfield)
                continue;
            if (firstValue) {
              maxValue = value;
              maxName = name;
              maxCore = version != "None";
              firstValue = false;
            } else {
              if (maxValue > value) {
                std::cerr << "Error: " << source["kind"] << " enumerant " << name
                          << " is out of order. It has value " <<  value
                          << " but follows the enumerant with value " << maxValue << std::endl;
                std::exit(1);
              } else if (maxValue == value &&
                         !SuffixComparison(maxName, maxCore, name,
                                           version != "None")) {
                std::cerr << "Error: " << source["kind"] << " enumerant " << name
                          << " is out of order. It follows alias " << maxName << std::endl;
                std::exit(1);
              } else {
                maxValue = value;
                maxName = name;
                maxCore = version != "None";
              }
            }
            EnumCaps caps(getCaps(enumerant));
            std::string lastVersion = enumerant["lastVersion"].asString();
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
                std::move(caps), std::move(version), std::move(lastVersion), std::move(exts), std::move(params));
        }
    };

    const auto establishOperandClass = [&populateEnumValues,&errorCount](
            const std::string& enumName, spv::OperandClass operandClass,
            spv::EnumValues* enumValues, const Json::Value& operandEnum, const std::string& category) {
        assert(category == "BitEnum" || category == "ValueEnum");
        bool bitEnum = (category == "BitEnum");
        if (!operandEnum["version"].empty()) {
          std::cerr << "Error: container for " << enumName << " operand_kind must not have a version field" << std::endl;
          errorCount++;
        }
        populateEnumValues(enumValues, operandEnum, bitEnum);
        const std::string errContext = "enum " + enumName;
        for (const auto& e: *enumValues) {
          if (!e.IsValid(operandClass, errContext)) {
            errorCount++;
          }
        }
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
        } else if (enumName == "FPDenormMode") {
            establishOperandClass(enumName, OperandFPDenormMode, &FPDenormModeParams, operandEnum, category);
        } else if (enumName == "FPOperationMode") {
            establishOperandClass(enumName, OperandFPOperationMode, &FPOperationModeParams, operandEnum, category);
        } else if (enumName == "QuantizationModes") {
            establishOperandClass(enumName, OperandQuantizationModes, &QuantizationModesParams, operandEnum, category);
        } else if (enumName == "OverflowModes") {
            establishOperandClass(enumName, OperandOverflowModes, &OverflowModesParams, operandEnum, category);
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
            establishOperandClass(enumName, OperandMemoryOperands, &MemoryAccessParams, operandEnum, category);
        } else if (enumName == "Scope") {
            establishOperandClass(enumName, OperandScope, &ScopeParams, operandEnum, category);
        } else if (enumName == "GroupOperation") {
            establishOperandClass(enumName, OperandGroupOperation, &GroupOperationParams, operandEnum, category);
        } else if (enumName == "KernelEnqueueFlags") {
            establishOperandClass(enumName, OperandKernelEnqueueFlags, &KernelEnqueueFlagsParams, operandEnum, category);
        } else if (enumName == "KernelProfilingInfo") {
            establishOperandClass(enumName, OperandKernelProfilingInfo, &KernelProfilingInfoParams, operandEnum, category);
        } else if (enumName == "RayFlags") {
            establishOperandClass(enumName, OperandRayFlags, &RayFlagsParams, operandEnum, category);
        } else if (enumName == "RayQueryIntersection") {
            establishOperandClass(enumName, OperandRayQueryIntersection, &RayQueryIntersectionParams, operandEnum, category);
        } else if (enumName == "RayQueryCommittedIntersectionType") {
            establishOperandClass(enumName, OperandRayQueryCommittedIntersectionType, &RayQueryCommittedIntersectionTypeParams, operandEnum, category);
        } else if (enumName == "RayQueryCandidateIntersectionType") {
            establishOperandClass(enumName, OperandRayQueryCandidateIntersectionType, &RayQueryCandidateIntersectionTypeParams, operandEnum, category);
        } else if (enumName == "FragmentShadingRate") {
            establishOperandClass(enumName, OperandFragmentShadingRate, &FragmentShadingRateParams, operandEnum, category);
        } else if (enumName == "PackedVectorFormat") {
            establishOperandClass(enumName, OperandPackedVectorFormat, &PackedVectorFormatParams, operandEnum, category);
        } else if (enumName == "CooperativeMatrixOperands") {
            establishOperandClass(enumName, OperandCooperativeMatrixOperands, &CooperativeMatrixOperandsParams, operandEnum, category);
        } else if (enumName == "CooperativeMatrixLayout") {
            establishOperandClass(enumName, OperandCooperativeMatrixLayout, &CooperativeMatrixLayoutParams, operandEnum, category);
        } else if (enumName == "CooperativeMatrixUse") {
            establishOperandClass(enumName, OperandCooperativeMatrixUse, &CooperativeMatrixUseParams, operandEnum, category);
        } else if (enumName == "InitializationModeQualifier") {
            establishOperandClass(enumName, OperandInitializationModeQualifier, &InitializationModeQualifierParams, operandEnum, category);
        } else if (enumName == "HostAccessQualifier") {
            establishOperandClass(enumName, OperandHostAccessQualifier, &HostAccessQualifierParams, operandEnum, category);
        } else if (enumName == "LoadCacheControl") {
            establishOperandClass(enumName, OperandLoadCacheControl, &LoadCacheControlParams, operandEnum, category);
        } else if (enumName == "StoreCacheControl") {
            establishOperandClass(enumName, OperandStoreCacheControl, &StoreCacheControlParams, operandEnum, category);
        } else if (enumName == "NamedMaximumNumberOfRegisters") {
            establishOperandClass(enumName, OperandNamedMaximumNumberOfRegisters, &NamedMaximumNumberOfRegistersParams, operandEnum, category);
        } else if (enumName == "RawAccessChainOperands") {
            establishOperandClass(enumName, OperandRawAccessChainOperands, &RawAccessChainOperandsParams, operandEnum, category);
        } else if (enumName == "FPEncoding") {
            establishOperandClass(enumName, OperandFPEncoding, &FPEncodingParams, operandEnum, category);
        }
    }

    if (errorCount > 0) {
      std::exit(1);
    }
}

};  // end namespace spv
