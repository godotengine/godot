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

#pragma once
#ifndef JSON_TO_SPIRV
#define JSON_TO_SPIRV

#include <algorithm>
#include <string>
#include <vector>
#include <assert.h>

namespace spv {

    // Reads the file in the given |path|. Returns true and the contents of the
// file on success; otherwise, returns false and an empty string.
std::pair<bool, std::string> ReadFile(const std::string& path);

// Fill in all the parameters
void jsonToSpirv(const std::string& jsonPath, bool buildingHeaders);

// For parameterizing operands.
// The ordering here affects the printing order in the SPIR-V specification.
// Please add new operand classes at the end.
enum OperandClass {
    OperandNone,
    OperandId,
    OperandVariableIds,
    OperandOptionalLiteral,
    OperandOptionalLiteralString,
    OperandOptionalLiteralStrings,
    OperandVariableLiterals,
    OperandVariableIdLiteral,
    OperandVariableLiteralId,
    OperandAnySizeLiteralNumber,
    OperandLiteralNumber,
    OperandLiteralString,
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
    OperandMemoryOperands,
    OperandScope,
    OperandGroupOperation,
    OperandKernelEnqueueFlags,
    OperandKernelProfilingInfo,
    OperandCapability,
    OperandRayFlags,
    OperandRayQueryIntersection,
    OperandRayQueryCommittedIntersectionType,
    OperandRayQueryCandidateIntersectionType,
    OperandFragmentShadingRate,
    OperandFPDenormMode,
    OperandFPOperationMode,
    OperandQuantizationModes,
    OperandOverflowModes,
    OperandPackedVectorFormat,
    OperandCooperativeMatrixOperands,
    OperandCooperativeMatrixLayout,
    OperandCooperativeMatrixUse,
    OperandInitializationModeQualifier,
    OperandHostAccessQualifier,
    OperandLoadCacheControl,
    OperandStoreCacheControl,
    OperandNamedMaximumNumberOfRegisters,
    OperandRawAccessChainOperands,
    OperandFPEncoding,

    OperandOpcode,

    OperandCount
};

// For direct representation of the JSON grammar "instruction_printing_class".
struct PrintingClass {
    std::string tag;
    std::string heading;
};
using PrintingClasses = std::vector<PrintingClass>;

// Any specific enum can have a set of capabilities that allow it:
typedef std::vector<std::string> EnumCaps;

// A set of extensions.
typedef std::vector<std::string> Extensions;

// Parameterize a set of operands with their OperandClass(es) and descriptions.
class OperandParameters {
public:
    OperandParameters() { }
    void push(OperandClass oc, const std::string& d, bool opt = false)
    {
        opClass.push_back(oc);
        desc.push_back(d);
        optional.push_back(opt);
    }
    void setOptional();
    OperandClass getClass(int op) const { return opClass[op]; }
    const char* getDesc(int op) const { return desc[op].c_str(); }
    bool isOptional(int op) const { return optional[op]; }
    int getNum() const { return (int)opClass.size(); }

protected:
    std::vector<OperandClass> opClass;
    std::vector<std::string> desc;
    std::vector<bool> optional;
};

// An ordered sequence of EValue.  We'll preserve the order found in the
// JSON file.  You can look up a value by enum or by name.  If there are
// duplicate values, then take the first.  We assume names are unique.
// The EValue must have an unsigned |value| field and a string |name| field.
template <typename EValue>
class EnumValuesContainer {
public:
    using ContainerType = std::vector<EValue>;
    using iterator = typename ContainerType::iterator;
    using const_iterator = typename ContainerType::const_iterator;

    EnumValuesContainer() {}

    // Constructs an EValue in place as a new element at the end of the
    // sequence.
    template <typename... Args>
    void emplace_back(Args&&... args) {
        values.emplace_back(std::forward<Args>(args)...);
    }

    // Returns the first EValue in the sequence with the given value.
    // More than one EValue might have the same value.
    EValue& operator[](unsigned value) {
        auto where = std::find_if(begin(), end(), [&value](const EValue& e) {
           return value == e.value;
        });
        assert((where != end()) && "Could not find enum in the enum list");
        return *where;
    }
    // gets *all* entries for the value, including the first one
    void gatherAliases(unsigned value, std::vector<EValue*>& aliases) {
        std::for_each(begin(), end(), [&](EValue& e) {
            if (value == e.value)
                aliases.push_back(&e);});
    }
    // Returns the EValue with the given name.  We assume uniqueness
    // by name.
    EValue& at(std::string name) {
        auto where = std::find_if(begin(), end(), [&name](const EValue& e) {
           return name == e.name;
        });
        assert((where != end()) && "Could not find name in the enum list");
        return *where;
    }

    iterator begin() { return values.begin(); }
    iterator end() { return values.end(); }
    EValue& back() { return values.back(); }

private:
    ContainerType values;
};

// A single enumerant value.  Corresponds to a row in an enumeration table
// in the spec.
class EnumValue {
public:
    EnumValue() : value(0), desc(nullptr) {}
    EnumValue(unsigned int the_value, const std::string& the_name, EnumCaps&& the_caps,
        const std::string& the_firstVersion, const std::string& the_lastVersion,
        Extensions&& the_extensions, OperandParameters&& the_operands) :
      value(the_value), name(the_name), capabilities(std::move(the_caps)),
      firstVersion(std::move(the_firstVersion)), lastVersion(std::move(the_lastVersion)),
      extensions(std::move(the_extensions)), operands(std::move(the_operands)), desc(nullptr) { }

    // For ValueEnum, the value from the JSON file.
    // For BitEnum, the index of the bit position represented by this mask.
    // (That is, what you shift 1 by to get the mask.)
    unsigned value;
    std::string name;
    EnumCaps capabilities;
    std::string firstVersion;
    std::string lastVersion;
    // A feature only be enabled by certain extensions.
    // An empty list means the feature does not require an extension.
    // Normally, only Capability enums are enabled by extension.  In turn,
    // other enums and instructions are enabled by those capabilities.
    Extensions extensions;
    OperandParameters operands;
    const char* desc;

    // Returns true if this enum is valid, in isolation.
    // Otherwise emits a diagnostic to std::cerr and returns false.
    bool IsValid(OperandClass oc, const std::string& context) const;
};

using EnumValues = EnumValuesContainer<EnumValue>;

// Parameterize a set of enumerants that form an enum
class EnumDefinition {
public:
    EnumDefinition() :
        desc(0), bitmask(false), enumValues(nullptr) { }
    void set(const std::string& enumName, EnumValues* enumValuesArg, bool mask = false)
    {
        codeName = enumName;
        bitmask = mask;
        enumValues = enumValuesArg;
    }
    // Returns the first EnumValue in the sequence with the given value.
    // More than one EnumValue might have the same value.  Only valid
    // if enumValues has been populated.
    EnumValue& operator[](unsigned value) {
        assert(enumValues != nullptr);
        return (*enumValues)[value];
    }
    // Returns the name of the first EnumValue with the given value.
    // Assumes enumValues has been populated.
    const char* getName(unsigned value) {
        return (*this)[value].name.c_str();
    }

    using iterator = EnumValues::iterator;
    iterator begin() { return enumValues->begin(); }
    iterator end() { return enumValues->end(); }

    std::string codeName; // name to use when declaring headers for code
    const char* desc;
    bool bitmask;  // true if these enumerants combine into a bitmask
    EnumValues* enumValues; // parameters for each individual enumerant
};

// Parameterize an instruction's logical format, including its known set of operands,
// per OperandParameters above.
class InstructionValue : public EnumValue {
public:
    InstructionValue(EnumValue&& e, const std::string& printClass, bool has_type, bool has_result)
     : EnumValue(std::move(e)),
       printingClass(printClass),
       opDesc("TBD."),
       typePresent(has_type),
       resultPresent(has_result),
       alias(this) { }
    InstructionValue(const InstructionValue& v)
    {
        *this = v;
        alias = this;
    }

    bool hasResult() const { return resultPresent != 0; }
    bool hasType()   const { return typePresent != 0; }
    void setAlias(const InstructionValue& a) { alias = &a; }
    const InstructionValue& getAlias() const { return *alias; }
    bool isAlias() const { return alias != this; }

    std::string printingClass;
    const char* opDesc;

protected:
    int typePresent   : 1;
    int resultPresent : 1;
    const InstructionValue* alias;    // correct only after discovering the aliases; otherwise points to this
};

using InstructionValues = EnumValuesContainer<InstructionValue>;

// Parameterization info for all instructions.
extern InstructionValues InstructionDesc;
extern PrintingClasses InstructionPrintingClasses;

// These hold definitions of the enumerants used for operands.
// This is indexed by OperandClass, but not including OperandOpcode.
extern EnumDefinition OperandClassParams[];

};  // end namespace spv

#endif // JSON_TO_SPIRV
