//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLLibrary.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "../Foundation/Foundation.hpp"
#include "MTLDataType.hpp"
#include "MTLDefines.hpp"
#include "MTLFunctionDescriptor.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
class Argument;
class ArgumentEncoder;
class Attribute;
class CompileOptions;
class Device;
class Function;
class FunctionConstant;
class FunctionConstantValues;
class FunctionDescriptor;
class FunctionReflection;
class IntersectionFunctionDescriptor;
class VertexAttribute;
_MTL_ENUM(NS::UInteger, PatchType) {
    PatchTypeNone = 0,
    PatchTypeTriangle = 1,
    PatchTypeQuad = 2,
};

_MTL_ENUM(NS::UInteger, FunctionType) {
    FunctionTypeVertex = 1,
    FunctionTypeFragment = 2,
    FunctionTypeKernel = 3,
    FunctionTypeVisible = 5,
    FunctionTypeIntersection = 6,
    FunctionTypeMesh = 7,
    FunctionTypeObject = 8,
};

_MTL_ENUM(NS::UInteger, LanguageVersion) {
    LanguageVersion1_0 = 65536,
    LanguageVersion1_1 = 65537,
    LanguageVersion1_2 = 65538,
    LanguageVersion2_0 = 131072,
    LanguageVersion2_1 = 131073,
    LanguageVersion2_2 = 131074,
    LanguageVersion2_3 = 131075,
    LanguageVersion2_4 = 131076,
    LanguageVersion3_0 = 196608,
    LanguageVersion3_1 = 196609,
    LanguageVersion3_2 = 196610,
    LanguageVersion4_0 = 262144,
};

_MTL_ENUM(NS::Integer, LibraryType) {
    LibraryTypeExecutable = 0,
    LibraryTypeDynamic = 1,
};

_MTL_ENUM(NS::Integer, LibraryOptimizationLevel) {
    LibraryOptimizationLevelDefault = 0,
    LibraryOptimizationLevelSize = 1,
};

_MTL_ENUM(NS::Integer, CompileSymbolVisibility) {
    CompileSymbolVisibilityDefault = 0,
    CompileSymbolVisibilityHidden = 1,
};

_MTL_ENUM(NS::Integer, MathMode) {
    MathModeSafe = 0,
    MathModeRelaxed = 1,
    MathModeFast = 2,
};

_MTL_ENUM(NS::Integer, MathFloatingPointFunctions) {
    MathFloatingPointFunctionsFast = 0,
    MathFloatingPointFunctionsPrecise = 1,
};

_MTL_ENUM(NS::UInteger, LibraryError) {
    LibraryErrorUnsupported = 1,
    LibraryErrorInternal = 2,
    LibraryErrorCompileFailure = 3,
    LibraryErrorCompileWarning = 4,
    LibraryErrorFunctionNotFound = 5,
    LibraryErrorFileNotFound = 6,
};

using AutoreleasedArgument = MTL::Argument*;
using FunctionCompletionHandlerFunction = std::function<void(MTL::Function* pFunction, NS::Error* pError)>;

class VertexAttribute : public NS::Referencing<VertexAttribute>
{
public:
    [[deprecated("please use isActive instead")]]
    bool                    active() const;

    static VertexAttribute* alloc();

    NS::UInteger            attributeIndex() const;

    DataType                attributeType() const;

    VertexAttribute*        init();

    bool                    isActive() const;

    bool                    isPatchControlPointData() const;

    bool                    isPatchData() const;

    NS::String*             name() const;

    [[deprecated("please use isPatchControlPointData instead")]]
    bool patchControlPointData() const;

    [[deprecated("please use isPatchData instead")]]
    bool patchData() const;
};
class Attribute : public NS::Referencing<Attribute>
{
public:
    [[deprecated("please use isActive instead")]]
    bool              active() const;

    static Attribute* alloc();

    NS::UInteger      attributeIndex() const;

    DataType          attributeType() const;

    Attribute*        init();

    bool              isActive() const;

    bool              isPatchControlPointData() const;

    bool              isPatchData() const;

    NS::String*       name() const;

    [[deprecated("please use isPatchControlPointData instead")]]
    bool patchControlPointData() const;

    [[deprecated("please use isPatchData instead")]]
    bool patchData() const;
};
class FunctionConstant : public NS::Referencing<FunctionConstant>
{
public:
    static FunctionConstant* alloc();

    NS::UInteger             index() const;

    FunctionConstant*        init();

    NS::String*              name() const;

    bool                     required() const;

    DataType                 type() const;
};
class Function : public NS::Referencing<Function>
{
public:
    Device*          device() const;

    NS::Dictionary*  functionConstantsDictionary() const;

    FunctionType     functionType() const;

    NS::String*      label() const;

    NS::String*      name() const;

    ArgumentEncoder* newArgumentEncoder(NS::UInteger bufferIndex);
    ArgumentEncoder* newArgumentEncoder(NS::UInteger bufferIndex, const MTL::AutoreleasedArgument* reflection);

    FunctionOptions  options() const;

    NS::Integer      patchControlPointCount() const;

    PatchType        patchType() const;

    void             setLabel(const NS::String* label);

    NS::Array*       stageInputAttributes() const;

    NS::Array*       vertexAttributes() const;
};
class CompileOptions : public NS::Copying<CompileOptions>
{
public:
    static CompileOptions*     alloc();

    bool                       allowReferencingUndefinedSymbols() const;

    CompileSymbolVisibility    compileSymbolVisibility() const;

    bool                       enableLogging() const;

    bool                       fastMathEnabled() const;

    CompileOptions*            init();

    NS::String*                installName() const;

    LanguageVersion            languageVersion() const;

    NS::Array*                 libraries() const;

    LibraryType                libraryType() const;

    MathFloatingPointFunctions mathFloatingPointFunctions() const;

    MathMode                   mathMode() const;

    NS::UInteger               maxTotalThreadsPerThreadgroup() const;

    LibraryOptimizationLevel   optimizationLevel() const;

    NS::Dictionary*            preprocessorMacros() const;

    bool                       preserveInvariance() const;

    Size                       requiredThreadsPerThreadgroup() const;

    void                       setAllowReferencingUndefinedSymbols(bool allowReferencingUndefinedSymbols);

    void                       setCompileSymbolVisibility(MTL::CompileSymbolVisibility compileSymbolVisibility);

    void                       setEnableLogging(bool enableLogging);

    void                       setFastMathEnabled(bool fastMathEnabled);

    void                       setInstallName(const NS::String* installName);

    void                       setLanguageVersion(MTL::LanguageVersion languageVersion);

    void                       setLibraries(const NS::Array* libraries);

    void                       setLibraryType(MTL::LibraryType libraryType);

    void                       setMathFloatingPointFunctions(MTL::MathFloatingPointFunctions mathFloatingPointFunctions);

    void                       setMathMode(MTL::MathMode mathMode);

    void                       setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup);

    void                       setOptimizationLevel(MTL::LibraryOptimizationLevel optimizationLevel);

    void                       setPreprocessorMacros(const NS::Dictionary* preprocessorMacros);

    void                       setPreserveInvariance(bool preserveInvariance);

    void                       setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup);
};
class FunctionReflection : public NS::Referencing<FunctionReflection>
{
public:
    static FunctionReflection* alloc();

    NS::Array*                 bindings() const;

    FunctionReflection*        init();
};
class Library : public NS::Referencing<Library>
{
public:
    Device*             device() const;

    NS::Array*          functionNames() const;

    NS::String*         installName() const;

    NS::String*         label() const;

    Function*           newFunction(const NS::String* functionName);
    Function*           newFunction(const NS::String* name, const MTL::FunctionConstantValues* constantValues, NS::Error** error);
    void                newFunction(const NS::String* name, const MTL::FunctionConstantValues* constantValues, void (^completionHandler)(MTL::Function*, NS::Error*));
    void                newFunction(const MTL::FunctionDescriptor* descriptor, void (^completionHandler)(MTL::Function*, NS::Error*));
    Function*           newFunction(const MTL::FunctionDescriptor* descriptor, NS::Error** error);
    void                newFunction(const NS::String* pFunctionName, const MTL::FunctionConstantValues* pConstantValues, const MTL::FunctionCompletionHandlerFunction& completionHandler);
    void                newFunction(const MTL::FunctionDescriptor* pDescriptor, const MTL::FunctionCompletionHandlerFunction& completionHandler);

    void                newIntersectionFunction(const MTL::IntersectionFunctionDescriptor* descriptor, void (^completionHandler)(MTL::Function*, NS::Error*));
    Function*           newIntersectionFunction(const MTL::IntersectionFunctionDescriptor* descriptor, NS::Error** error);
    void                newIntersectionFunction(const MTL::IntersectionFunctionDescriptor* pDescriptor, const MTL::FunctionCompletionHandlerFunction& completionHandler);

    FunctionReflection* reflectionForFunction(const NS::String* functionName);

    void                setLabel(const NS::String* label);

    LibraryType         type() const;
};

}
_MTL_INLINE bool MTL::VertexAttribute::active() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isActive));
}

_MTL_INLINE MTL::VertexAttribute* MTL::VertexAttribute::alloc()
{
    return NS::Object::alloc<MTL::VertexAttribute>(_MTL_PRIVATE_CLS(MTLVertexAttribute));
}

_MTL_INLINE NS::UInteger MTL::VertexAttribute::attributeIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(attributeIndex));
}

_MTL_INLINE MTL::DataType MTL::VertexAttribute::attributeType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(attributeType));
}

_MTL_INLINE MTL::VertexAttribute* MTL::VertexAttribute::init()
{
    return NS::Object::init<MTL::VertexAttribute>();
}

_MTL_INLINE bool MTL::VertexAttribute::isActive() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isActive));
}

_MTL_INLINE bool MTL::VertexAttribute::isPatchControlPointData() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isPatchControlPointData));
}

_MTL_INLINE bool MTL::VertexAttribute::isPatchData() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isPatchData));
}

_MTL_INLINE NS::String* MTL::VertexAttribute::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE bool MTL::VertexAttribute::patchControlPointData() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isPatchControlPointData));
}

_MTL_INLINE bool MTL::VertexAttribute::patchData() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isPatchData));
}

_MTL_INLINE bool MTL::Attribute::active() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isActive));
}

_MTL_INLINE MTL::Attribute* MTL::Attribute::alloc()
{
    return NS::Object::alloc<MTL::Attribute>(_MTL_PRIVATE_CLS(MTLAttribute));
}

_MTL_INLINE NS::UInteger MTL::Attribute::attributeIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(attributeIndex));
}

_MTL_INLINE MTL::DataType MTL::Attribute::attributeType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(attributeType));
}

_MTL_INLINE MTL::Attribute* MTL::Attribute::init()
{
    return NS::Object::init<MTL::Attribute>();
}

_MTL_INLINE bool MTL::Attribute::isActive() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isActive));
}

_MTL_INLINE bool MTL::Attribute::isPatchControlPointData() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isPatchControlPointData));
}

_MTL_INLINE bool MTL::Attribute::isPatchData() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isPatchData));
}

_MTL_INLINE NS::String* MTL::Attribute::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE bool MTL::Attribute::patchControlPointData() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isPatchControlPointData));
}

_MTL_INLINE bool MTL::Attribute::patchData() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isPatchData));
}

_MTL_INLINE MTL::FunctionConstant* MTL::FunctionConstant::alloc()
{
    return NS::Object::alloc<MTL::FunctionConstant>(_MTL_PRIVATE_CLS(MTLFunctionConstant));
}

_MTL_INLINE NS::UInteger MTL::FunctionConstant::index() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(index));
}

_MTL_INLINE MTL::FunctionConstant* MTL::FunctionConstant::init()
{
    return NS::Object::init<MTL::FunctionConstant>();
}

_MTL_INLINE NS::String* MTL::FunctionConstant::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE bool MTL::FunctionConstant::required() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(required));
}

_MTL_INLINE MTL::DataType MTL::FunctionConstant::type() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(type));
}

_MTL_INLINE MTL::Device* MTL::Function::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::Dictionary* MTL::Function::functionConstantsDictionary() const
{
    return Object::sendMessage<NS::Dictionary*>(this, _MTL_PRIVATE_SEL(functionConstantsDictionary));
}

_MTL_INLINE MTL::FunctionType MTL::Function::functionType() const
{
    return Object::sendMessage<MTL::FunctionType>(this, _MTL_PRIVATE_SEL(functionType));
}

_MTL_INLINE NS::String* MTL::Function::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::String* MTL::Function::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE MTL::ArgumentEncoder* MTL::Function::newArgumentEncoder(NS::UInteger bufferIndex)
{
    return Object::sendMessage<MTL::ArgumentEncoder*>(this, _MTL_PRIVATE_SEL(newArgumentEncoderWithBufferIndex_), bufferIndex);
}

_MTL_INLINE MTL::ArgumentEncoder* MTL::Function::newArgumentEncoder(NS::UInteger bufferIndex, const MTL::AutoreleasedArgument* reflection)
{
    return Object::sendMessage<MTL::ArgumentEncoder*>(this, _MTL_PRIVATE_SEL(newArgumentEncoderWithBufferIndex_reflection_), bufferIndex, reflection);
}

_MTL_INLINE MTL::FunctionOptions MTL::Function::options() const
{
    return Object::sendMessage<MTL::FunctionOptions>(this, _MTL_PRIVATE_SEL(options));
}

_MTL_INLINE NS::Integer MTL::Function::patchControlPointCount() const
{
    return Object::sendMessage<NS::Integer>(this, _MTL_PRIVATE_SEL(patchControlPointCount));
}

_MTL_INLINE MTL::PatchType MTL::Function::patchType() const
{
    return Object::sendMessage<MTL::PatchType>(this, _MTL_PRIVATE_SEL(patchType));
}

_MTL_INLINE void MTL::Function::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE NS::Array* MTL::Function::stageInputAttributes() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(stageInputAttributes));
}

_MTL_INLINE NS::Array* MTL::Function::vertexAttributes() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(vertexAttributes));
}

_MTL_INLINE MTL::CompileOptions* MTL::CompileOptions::alloc()
{
    return NS::Object::alloc<MTL::CompileOptions>(_MTL_PRIVATE_CLS(MTLCompileOptions));
}

_MTL_INLINE bool MTL::CompileOptions::allowReferencingUndefinedSymbols() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(allowReferencingUndefinedSymbols));
}

_MTL_INLINE MTL::CompileSymbolVisibility MTL::CompileOptions::compileSymbolVisibility() const
{
    return Object::sendMessage<MTL::CompileSymbolVisibility>(this, _MTL_PRIVATE_SEL(compileSymbolVisibility));
}

_MTL_INLINE bool MTL::CompileOptions::enableLogging() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(enableLogging));
}

_MTL_INLINE bool MTL::CompileOptions::fastMathEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(fastMathEnabled));
}

_MTL_INLINE MTL::CompileOptions* MTL::CompileOptions::init()
{
    return NS::Object::init<MTL::CompileOptions>();
}

_MTL_INLINE NS::String* MTL::CompileOptions::installName() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(installName));
}

_MTL_INLINE MTL::LanguageVersion MTL::CompileOptions::languageVersion() const
{
    return Object::sendMessage<MTL::LanguageVersion>(this, _MTL_PRIVATE_SEL(languageVersion));
}

_MTL_INLINE NS::Array* MTL::CompileOptions::libraries() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(libraries));
}

_MTL_INLINE MTL::LibraryType MTL::CompileOptions::libraryType() const
{
    return Object::sendMessage<MTL::LibraryType>(this, _MTL_PRIVATE_SEL(libraryType));
}

_MTL_INLINE MTL::MathFloatingPointFunctions MTL::CompileOptions::mathFloatingPointFunctions() const
{
    return Object::sendMessage<MTL::MathFloatingPointFunctions>(this, _MTL_PRIVATE_SEL(mathFloatingPointFunctions));
}

_MTL_INLINE MTL::MathMode MTL::CompileOptions::mathMode() const
{
    return Object::sendMessage<MTL::MathMode>(this, _MTL_PRIVATE_SEL(mathMode));
}

_MTL_INLINE NS::UInteger MTL::CompileOptions::maxTotalThreadsPerThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerThreadgroup));
}

_MTL_INLINE MTL::LibraryOptimizationLevel MTL::CompileOptions::optimizationLevel() const
{
    return Object::sendMessage<MTL::LibraryOptimizationLevel>(this, _MTL_PRIVATE_SEL(optimizationLevel));
}

_MTL_INLINE NS::Dictionary* MTL::CompileOptions::preprocessorMacros() const
{
    return Object::sendMessage<NS::Dictionary*>(this, _MTL_PRIVATE_SEL(preprocessorMacros));
}

_MTL_INLINE bool MTL::CompileOptions::preserveInvariance() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(preserveInvariance));
}

_MTL_INLINE MTL::Size MTL::CompileOptions::requiredThreadsPerThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerThreadgroup));
}

_MTL_INLINE void MTL::CompileOptions::setAllowReferencingUndefinedSymbols(bool allowReferencingUndefinedSymbols)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAllowReferencingUndefinedSymbols_), allowReferencingUndefinedSymbols);
}

_MTL_INLINE void MTL::CompileOptions::setCompileSymbolVisibility(MTL::CompileSymbolVisibility compileSymbolVisibility)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCompileSymbolVisibility_), compileSymbolVisibility);
}

_MTL_INLINE void MTL::CompileOptions::setEnableLogging(bool enableLogging)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setEnableLogging_), enableLogging);
}

_MTL_INLINE void MTL::CompileOptions::setFastMathEnabled(bool fastMathEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFastMathEnabled_), fastMathEnabled);
}

_MTL_INLINE void MTL::CompileOptions::setInstallName(const NS::String* installName)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstallName_), installName);
}

_MTL_INLINE void MTL::CompileOptions::setLanguageVersion(MTL::LanguageVersion languageVersion)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLanguageVersion_), languageVersion);
}

_MTL_INLINE void MTL::CompileOptions::setLibraries(const NS::Array* libraries)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLibraries_), libraries);
}

_MTL_INLINE void MTL::CompileOptions::setLibraryType(MTL::LibraryType libraryType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLibraryType_), libraryType);
}

_MTL_INLINE void MTL::CompileOptions::setMathFloatingPointFunctions(MTL::MathFloatingPointFunctions mathFloatingPointFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMathFloatingPointFunctions_), mathFloatingPointFunctions);
}

_MTL_INLINE void MTL::CompileOptions::setMathMode(MTL::MathMode mathMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMathMode_), mathMode);
}

_MTL_INLINE void MTL::CompileOptions::setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadsPerThreadgroup_), maxTotalThreadsPerThreadgroup);
}

_MTL_INLINE void MTL::CompileOptions::setOptimizationLevel(MTL::LibraryOptimizationLevel optimizationLevel)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOptimizationLevel_), optimizationLevel);
}

_MTL_INLINE void MTL::CompileOptions::setPreprocessorMacros(const NS::Dictionary* preprocessorMacros)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPreprocessorMacros_), preprocessorMacros);
}

_MTL_INLINE void MTL::CompileOptions::setPreserveInvariance(bool preserveInvariance)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPreserveInvariance_), preserveInvariance);
}

_MTL_INLINE void MTL::CompileOptions::setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRequiredThreadsPerThreadgroup_), requiredThreadsPerThreadgroup);
}

_MTL_INLINE MTL::FunctionReflection* MTL::FunctionReflection::alloc()
{
    return NS::Object::alloc<MTL::FunctionReflection>(_MTL_PRIVATE_CLS(MTLFunctionReflection));
}

_MTL_INLINE NS::Array* MTL::FunctionReflection::bindings() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(bindings));
}

_MTL_INLINE MTL::FunctionReflection* MTL::FunctionReflection::init()
{
    return NS::Object::init<MTL::FunctionReflection>();
}

_MTL_INLINE MTL::Device* MTL::Library::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::Array* MTL::Library::functionNames() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(functionNames));
}

_MTL_INLINE NS::String* MTL::Library::installName() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(installName));
}

_MTL_INLINE NS::String* MTL::Library::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL::Function* MTL::Library::newFunction(const NS::String* functionName)
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(newFunctionWithName_), functionName);
}

_MTL_INLINE MTL::Function* MTL::Library::newFunction(const NS::String* name, const MTL::FunctionConstantValues* constantValues, NS::Error** error)
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(newFunctionWithName_constantValues_error_), name, constantValues, error);
}

_MTL_INLINE void MTL::Library::newFunction(const NS::String* name, const MTL::FunctionConstantValues* constantValues, void (^completionHandler)(MTL::Function*, NS::Error*))
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(newFunctionWithName_constantValues_completionHandler_), name, constantValues, completionHandler);
}

_MTL_INLINE void MTL::Library::newFunction(const MTL::FunctionDescriptor* descriptor, void (^completionHandler)(MTL::Function*, NS::Error*))
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(newFunctionWithDescriptor_completionHandler_), descriptor, completionHandler);
}

_MTL_INLINE MTL::Function* MTL::Library::newFunction(const MTL::FunctionDescriptor* descriptor, NS::Error** error)
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(newFunctionWithDescriptor_error_), descriptor, error);
}

_MTL_INLINE void MTL::Library::newFunction(const NS::String* pFunctionName, const MTL::FunctionConstantValues* pConstantValues, const MTL::FunctionCompletionHandlerFunction& completionHandler)
{
    __block MTL::FunctionCompletionHandlerFunction blockCompletionHandler = completionHandler;
    newFunction(pFunctionName, pConstantValues, ^(MTL::Function* pFunction, NS::Error* pError) { blockCompletionHandler(pFunction, pError); });
}

_MTL_INLINE void MTL::Library::newFunction(const MTL::FunctionDescriptor* pDescriptor, const MTL::FunctionCompletionHandlerFunction& completionHandler)
{
    __block MTL::FunctionCompletionHandlerFunction blockCompletionHandler = completionHandler;
    newFunction(pDescriptor, ^(MTL::Function* pFunction, NS::Error* pError) { blockCompletionHandler(pFunction, pError); });
}

_MTL_INLINE void MTL::Library::newIntersectionFunction(const MTL::IntersectionFunctionDescriptor* descriptor, void (^completionHandler)(MTL::Function*, NS::Error*))
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(newIntersectionFunctionWithDescriptor_completionHandler_), descriptor, completionHandler);
}

_MTL_INLINE MTL::Function* MTL::Library::newIntersectionFunction(const MTL::IntersectionFunctionDescriptor* descriptor, NS::Error** error)
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(newIntersectionFunctionWithDescriptor_error_), descriptor, error);
}

_MTL_INLINE void MTL::Library::newIntersectionFunction(const MTL::IntersectionFunctionDescriptor* pDescriptor, const MTL::FunctionCompletionHandlerFunction& completionHandler)
{
    __block MTL::FunctionCompletionHandlerFunction blockCompletionHandler = completionHandler;
    newIntersectionFunction(pDescriptor, ^(MTL::Function* pFunction, NS::Error* pError) { blockCompletionHandler(pFunction, pError); });
}

_MTL_INLINE MTL::FunctionReflection* MTL::Library::reflectionForFunction(const NS::String* functionName)
{
    return Object::sendMessage<MTL::FunctionReflection*>(this, _MTL_PRIVATE_SEL(reflectionForFunctionWithName_), functionName);
}

_MTL_INLINE void MTL::Library::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE MTL::LibraryType MTL::Library::type() const
{
    return Object::sendMessage<MTL::LibraryType>(this, _MTL_PRIVATE_SEL(type));
}
