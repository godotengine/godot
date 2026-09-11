#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class ArgumentEncoder;
    class Device;
    class FunctionConstantValues;
    class FunctionDescriptor;
    class IntersectionFunctionDescriptor;
    enum DataType : NS::UInteger;
    using FunctionOptions = NS::UInteger;
}
namespace NS {
    class Array;
    class Dictionary;
    class Error;
    class String;
}

namespace MTL
{

extern NS::ErrorDomain const LibraryErrorDomain __asm__("_MTLLibraryErrorDomain");
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
    LanguageVersion1_0 = (1 << 16),
    LanguageVersion1_1 = (1 << 16) + 1,
    LanguageVersion1_2 = (1 << 16) + 2,
    LanguageVersion2_0 = (2 << 16),
    LanguageVersion2_1 = (2 << 16) + 1,
    LanguageVersion2_2 = (2 << 16) + 2,
    LanguageVersion2_3 = (2 << 16) + 3,
    LanguageVersion2_4 = (2 << 16) + 4,
    LanguageVersion3_0 = (3 << 16) + 0,
    LanguageVersion3_1 = (3 << 16) + 1,
    LanguageVersion3_2 = (3 << 16) + 2,
    LanguageVersion4_0 = (4 << 16) + 0,
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


class VertexAttribute;
class Attribute;
class FunctionConstant;
class Function;
class CompileOptions;
class FunctionReflection;
class Library;

class VertexAttribute : public NS::Referencing<VertexAttribute>
{
public:
    static VertexAttribute* alloc();
    VertexAttribute*        init() const;

    bool          active() const;
    NS::UInteger  attributeIndex() const;
    MTL::DataType attributeType() const;
    bool          isActive();
    bool          isPatchControlPointData();
    bool          isPatchData();
    NS::String*   name() const;
    bool          patchControlPointData() const;
    bool          patchData() const;

};

class Attribute : public NS::Referencing<Attribute>
{
public:
    static Attribute* alloc();
    Attribute*        init() const;

    bool          active() const;
    NS::UInteger  attributeIndex() const;
    MTL::DataType attributeType() const;
    bool          isActive();
    bool          isPatchControlPointData();
    bool          isPatchData();
    NS::String*   name() const;
    bool          patchControlPointData() const;
    bool          patchData() const;

};

class FunctionConstant : public NS::Referencing<FunctionConstant>
{
public:
    static FunctionConstant* alloc();
    FunctionConstant*        init() const;

    NS::UInteger  index() const;
    NS::String*   name() const;
    bool          required() const;
    MTL::DataType type() const;

};

class Function : public NS::Referencing<Function>
{
public:
    MTL::Device*          device() const;
    NS::Dictionary*       functionConstantsDictionary() const;
    MTL::FunctionType     functionType() const;
    NS::String*           label() const;
    NS::String*           name() const;
    MTL::ArgumentEncoder* newArgumentEncoder(NS::UInteger bufferIndex);
    MTL::ArgumentEncoder* newArgumentEncoder(NS::UInteger bufferIndex, MTL::AutoreleasedArgument* reflection);
    MTL::FunctionOptions  options() const;
    NS::Integer           patchControlPointCount() const;
    MTL::PatchType        patchType() const;
    void                  setLabel(NS::String* label);
    NS::Array*            stageInputAttributes() const;
    NS::Array*            vertexAttributes() const;

};

class CompileOptions : public NS::Copying<CompileOptions>
{
public:
    static CompileOptions* alloc();
    CompileOptions*        init() const;

    bool                            allowReferencingUndefinedSymbols() const;
    MTL::CompileSymbolVisibility    compileSymbolVisibility() const;
    bool                            enableLogging() const;
    bool                            fastMathEnabled() const;
    NS::String*                     installName() const;
    MTL::LanguageVersion            languageVersion() const;
    NS::Array*                      libraries() const;
    MTL::LibraryType                libraryType() const;
    MTL::MathFloatingPointFunctions mathFloatingPointFunctions() const;
    MTL::MathMode                   mathMode() const;
    NS::UInteger                    maxTotalThreadsPerThreadgroup() const;
    MTL::LibraryOptimizationLevel   optimizationLevel() const;
    NS::Dictionary*                 preprocessorMacros() const;
    bool                            preserveInvariance() const;
    MTL::Size                       requiredThreadsPerThreadgroup() const;
    void                            setAllowReferencingUndefinedSymbols(bool allowReferencingUndefinedSymbols);
    void                            setCompileSymbolVisibility(MTL::CompileSymbolVisibility compileSymbolVisibility);
    void                            setEnableLogging(bool enableLogging);
    void                            setFastMathEnabled(bool fastMathEnabled);
    void                            setInstallName(NS::String* installName);
    void                            setLanguageVersion(MTL::LanguageVersion languageVersion);
    void                            setLibraries(NS::Array* libraries);
    void                            setLibraryType(MTL::LibraryType libraryType);
    void                            setMathFloatingPointFunctions(MTL::MathFloatingPointFunctions mathFloatingPointFunctions);
    void                            setMathMode(MTL::MathMode mathMode);
    void                            setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup);
    void                            setOptimizationLevel(MTL::LibraryOptimizationLevel optimizationLevel);
    void                            setPreprocessorMacros(NS::Dictionary* preprocessorMacros);
    void                            setPreserveInvariance(bool preserveInvariance);
    void                            setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup);

};

class FunctionReflection : public NS::Referencing<FunctionReflection>
{
public:
    static FunctionReflection* alloc();
    FunctionReflection*        init() const;

    NS::Array* bindings() const;

};

class Library : public NS::Referencing<Library>
{
public:
    MTL::Device*             device() const;
    NS::Array*               functionNames() const;
    NS::String*              installName() const;
    NS::String*              label() const;
    MTL::Function*           newFunction(NS::String* functionName);
    MTL::Function*           newFunction(NS::String* name, MTL::FunctionConstantValues* constantValues, NS::Error** error);
    void                     newFunction(NS::String* name, MTL::FunctionConstantValues* constantValues, MTL::NewFunctionBlock completionHandler);
    void                     newFunction(NS::String* name, MTL::FunctionConstantValues* constantValues, const MTL::NewFunctionFunction& completionHandler);
    void                     newFunction(MTL::FunctionDescriptor* descriptor, MTL::NewFunctionBlock completionHandler);
    void                     newFunction(MTL::FunctionDescriptor* descriptor, const MTL::NewFunctionFunction& completionHandler);
    MTL::Function*           newFunction(MTL::FunctionDescriptor* descriptor, NS::Error** error);
    void                     newIntersectionFunction(MTL::IntersectionFunctionDescriptor* descriptor, MTL::NewFunctionBlock completionHandler);
    void                     newIntersectionFunction(MTL::IntersectionFunctionDescriptor* descriptor, const MTL::NewFunctionFunction& completionHandler);
    MTL::Function*           newIntersectionFunction(MTL::IntersectionFunctionDescriptor* descriptor, NS::Error** error);
    MTL::FunctionReflection* reflection(NS::String* functionName);
    void                     setLabel(NS::String* label);
    MTL::LibraryType         type() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLVertexAttribute;
extern "C" void *OBJC_CLASS_$_MTLAttribute;
extern "C" void *OBJC_CLASS_$_MTLFunctionConstant;
extern "C" void *OBJC_CLASS_$_MTLFunction;
extern "C" void *OBJC_CLASS_$_MTLCompileOptions;
extern "C" void *OBJC_CLASS_$_MTLFunctionReflection;
extern "C" void *OBJC_CLASS_$_MTLLibrary;

_MTL_INLINE MTL::VertexAttribute* MTL::VertexAttribute::alloc()
{
    return _MTL_msg_MTL__VertexAttributep_alloc((const void*)&OBJC_CLASS_$_MTLVertexAttribute, nullptr);
}

_MTL_INLINE MTL::VertexAttribute* MTL::VertexAttribute::init() const
{
    return _MTL_msg_MTL__VertexAttributep_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::VertexAttribute::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::VertexAttribute::attributeIndex() const
{
    return _MTL_msg_NS__UInteger_attributeIndex((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::VertexAttribute::attributeType() const
{
    return _MTL_msg_MTL__DataType_attributeType((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::VertexAttribute::active() const
{
    return _MTL_msg_bool_active((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::VertexAttribute::patchData() const
{
    return _MTL_msg_bool_patchData((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::VertexAttribute::patchControlPointData() const
{
    return _MTL_msg_bool_patchControlPointData((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::VertexAttribute::isActive()
{
    return _MTL_msg_bool_isActive((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::VertexAttribute::isPatchData()
{
    return _MTL_msg_bool_isPatchData((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::VertexAttribute::isPatchControlPointData()
{
    return _MTL_msg_bool_isPatchControlPointData((const void*)this, nullptr);
}

_MTL_INLINE MTL::Attribute* MTL::Attribute::alloc()
{
    return _MTL_msg_MTL__Attributep_alloc((const void*)&OBJC_CLASS_$_MTLAttribute, nullptr);
}

_MTL_INLINE MTL::Attribute* MTL::Attribute::init() const
{
    return _MTL_msg_MTL__Attributep_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::Attribute::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Attribute::attributeIndex() const
{
    return _MTL_msg_NS__UInteger_attributeIndex((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::Attribute::attributeType() const
{
    return _MTL_msg_MTL__DataType_attributeType((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Attribute::active() const
{
    return _MTL_msg_bool_active((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Attribute::patchData() const
{
    return _MTL_msg_bool_patchData((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Attribute::patchControlPointData() const
{
    return _MTL_msg_bool_patchControlPointData((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Attribute::isActive()
{
    return _MTL_msg_bool_isActive((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Attribute::isPatchData()
{
    return _MTL_msg_bool_isPatchData((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Attribute::isPatchControlPointData()
{
    return _MTL_msg_bool_isPatchControlPointData((const void*)this, nullptr);
}

_MTL_INLINE MTL::FunctionConstant* MTL::FunctionConstant::alloc()
{
    return _MTL_msg_MTL__FunctionConstantp_alloc((const void*)&OBJC_CLASS_$_MTLFunctionConstant, nullptr);
}

_MTL_INLINE MTL::FunctionConstant* MTL::FunctionConstant::init() const
{
    return _MTL_msg_MTL__FunctionConstantp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::FunctionConstant::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::FunctionConstant::type() const
{
    return _MTL_msg_MTL__DataType_type((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::FunctionConstant::index() const
{
    return _MTL_msg_NS__UInteger_index((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::FunctionConstant::required() const
{
    return _MTL_msg_bool_required((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::Function::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Function::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Device* MTL::Function::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE MTL::FunctionType MTL::Function::functionType() const
{
    return _MTL_msg_MTL__FunctionType_functionType((const void*)this, nullptr);
}

_MTL_INLINE MTL::PatchType MTL::Function::patchType() const
{
    return _MTL_msg_MTL__PatchType_patchType((const void*)this, nullptr);
}

_MTL_INLINE NS::Integer MTL::Function::patchControlPointCount() const
{
    return _MTL_msg_NS__Integer_patchControlPointCount((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::Function::vertexAttributes() const
{
    return _MTL_msg_NS__Arrayp_vertexAttributes((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::Function::stageInputAttributes() const
{
    return _MTL_msg_NS__Arrayp_stageInputAttributes((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::Function::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE NS::Dictionary* MTL::Function::functionConstantsDictionary() const
{
    return _MTL_msg_NS__Dictionaryp_functionConstantsDictionary((const void*)this, nullptr);
}

_MTL_INLINE MTL::FunctionOptions MTL::Function::options() const
{
    return _MTL_msg_MTL__FunctionOptions_options((const void*)this, nullptr);
}

_MTL_INLINE MTL::ArgumentEncoder* MTL::Function::newArgumentEncoder(NS::UInteger bufferIndex)
{
    return _MTL_msg_MTL__ArgumentEncoderp_newArgumentEncoderWithBufferIndex__NS__UInteger((const void*)this, nullptr, bufferIndex);
}

_MTL_INLINE MTL::ArgumentEncoder* MTL::Function::newArgumentEncoder(NS::UInteger bufferIndex, MTL::AutoreleasedArgument* reflection)
{
    return _MTL_msg_MTL__ArgumentEncoderp_newArgumentEncoderWithBufferIndex_reflection__NS__UInteger_MTL__Argumentpp((const void*)this, nullptr, bufferIndex, reflection);
}

_MTL_INLINE MTL::CompileOptions* MTL::CompileOptions::alloc()
{
    return _MTL_msg_MTL__CompileOptionsp_alloc((const void*)&OBJC_CLASS_$_MTLCompileOptions, nullptr);
}

_MTL_INLINE MTL::CompileOptions* MTL::CompileOptions::init() const
{
    return _MTL_msg_MTL__CompileOptionsp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::Dictionary* MTL::CompileOptions::preprocessorMacros() const
{
    return _MTL_msg_NS__Dictionaryp_preprocessorMacros((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setPreprocessorMacros(NS::Dictionary* preprocessorMacros)
{
    _MTL_msg_v_setPreprocessorMacros__NS__Dictionaryp((const void*)this, nullptr, preprocessorMacros);
}

_MTL_INLINE bool MTL::CompileOptions::fastMathEnabled() const
{
    return _MTL_msg_bool_fastMathEnabled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setFastMathEnabled(bool fastMathEnabled)
{
    _MTL_msg_v_setFastMathEnabled__bool((const void*)this, nullptr, fastMathEnabled);
}

_MTL_INLINE MTL::MathMode MTL::CompileOptions::mathMode() const
{
    return _MTL_msg_MTL__MathMode_mathMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setMathMode(MTL::MathMode mathMode)
{
    _MTL_msg_v_setMathMode__MTL__MathMode((const void*)this, nullptr, mathMode);
}

_MTL_INLINE MTL::MathFloatingPointFunctions MTL::CompileOptions::mathFloatingPointFunctions() const
{
    return _MTL_msg_MTL__MathFloatingPointFunctions_mathFloatingPointFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setMathFloatingPointFunctions(MTL::MathFloatingPointFunctions mathFloatingPointFunctions)
{
    _MTL_msg_v_setMathFloatingPointFunctions__MTL__MathFloatingPointFunctions((const void*)this, nullptr, mathFloatingPointFunctions);
}

_MTL_INLINE MTL::LanguageVersion MTL::CompileOptions::languageVersion() const
{
    return _MTL_msg_MTL__LanguageVersion_languageVersion((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setLanguageVersion(MTL::LanguageVersion languageVersion)
{
    _MTL_msg_v_setLanguageVersion__MTL__LanguageVersion((const void*)this, nullptr, languageVersion);
}

_MTL_INLINE MTL::LibraryType MTL::CompileOptions::libraryType() const
{
    return _MTL_msg_MTL__LibraryType_libraryType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setLibraryType(MTL::LibraryType libraryType)
{
    _MTL_msg_v_setLibraryType__MTL__LibraryType((const void*)this, nullptr, libraryType);
}

_MTL_INLINE NS::String* MTL::CompileOptions::installName() const
{
    return _MTL_msg_NS__Stringp_installName((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setInstallName(NS::String* installName)
{
    _MTL_msg_v_setInstallName__NS__Stringp((const void*)this, nullptr, installName);
}

_MTL_INLINE NS::Array* MTL::CompileOptions::libraries() const
{
    return _MTL_msg_NS__Arrayp_libraries((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setLibraries(NS::Array* libraries)
{
    _MTL_msg_v_setLibraries__NS__Arrayp((const void*)this, nullptr, libraries);
}

_MTL_INLINE bool MTL::CompileOptions::preserveInvariance() const
{
    return _MTL_msg_bool_preserveInvariance((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setPreserveInvariance(bool preserveInvariance)
{
    _MTL_msg_v_setPreserveInvariance__bool((const void*)this, nullptr, preserveInvariance);
}

_MTL_INLINE MTL::LibraryOptimizationLevel MTL::CompileOptions::optimizationLevel() const
{
    return _MTL_msg_MTL__LibraryOptimizationLevel_optimizationLevel((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setOptimizationLevel(MTL::LibraryOptimizationLevel optimizationLevel)
{
    _MTL_msg_v_setOptimizationLevel__MTL__LibraryOptimizationLevel((const void*)this, nullptr, optimizationLevel);
}

_MTL_INLINE MTL::CompileSymbolVisibility MTL::CompileOptions::compileSymbolVisibility() const
{
    return _MTL_msg_MTL__CompileSymbolVisibility_compileSymbolVisibility((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setCompileSymbolVisibility(MTL::CompileSymbolVisibility compileSymbolVisibility)
{
    _MTL_msg_v_setCompileSymbolVisibility__MTL__CompileSymbolVisibility((const void*)this, nullptr, compileSymbolVisibility);
}

_MTL_INLINE bool MTL::CompileOptions::allowReferencingUndefinedSymbols() const
{
    return _MTL_msg_bool_allowReferencingUndefinedSymbols((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setAllowReferencingUndefinedSymbols(bool allowReferencingUndefinedSymbols)
{
    _MTL_msg_v_setAllowReferencingUndefinedSymbols__bool((const void*)this, nullptr, allowReferencingUndefinedSymbols);
}

_MTL_INLINE NS::UInteger MTL::CompileOptions::maxTotalThreadsPerThreadgroup() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup)
{
    _MTL_msg_v_setMaxTotalThreadsPerThreadgroup__NS__UInteger((const void*)this, nullptr, maxTotalThreadsPerThreadgroup);
}

_MTL_INLINE MTL::Size MTL::CompileOptions::requiredThreadsPerThreadgroup() const
{
    return _MTL_msg_MTL__Size_requiredThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup)
{
    _MTL_msg_v_setRequiredThreadsPerThreadgroup__MTL__Size((const void*)this, nullptr, requiredThreadsPerThreadgroup);
}

_MTL_INLINE bool MTL::CompileOptions::enableLogging() const
{
    return _MTL_msg_bool_enableLogging((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CompileOptions::setEnableLogging(bool enableLogging)
{
    _MTL_msg_v_setEnableLogging__bool((const void*)this, nullptr, enableLogging);
}

_MTL_INLINE MTL::FunctionReflection* MTL::FunctionReflection::alloc()
{
    return _MTL_msg_MTL__FunctionReflectionp_alloc((const void*)&OBJC_CLASS_$_MTLFunctionReflection, nullptr);
}

_MTL_INLINE MTL::FunctionReflection* MTL::FunctionReflection::init() const
{
    return _MTL_msg_MTL__FunctionReflectionp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::FunctionReflection::bindings() const
{
    return _MTL_msg_NS__Arrayp_bindings((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::Library::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Library::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Device* MTL::Library::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::Library::functionNames() const
{
    return _MTL_msg_NS__Arrayp_functionNames((const void*)this, nullptr);
}

_MTL_INLINE MTL::LibraryType MTL::Library::type() const
{
    return _MTL_msg_MTL__LibraryType_type((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::Library::installName() const
{
    return _MTL_msg_NS__Stringp_installName((const void*)this, nullptr);
}

_MTL_INLINE MTL::Function* MTL::Library::newFunction(NS::String* functionName)
{
    return _MTL_msg_MTL__Functionp_newFunctionWithName__NS__Stringp((const void*)this, nullptr, functionName);
}

_MTL_INLINE MTL::Function* MTL::Library::newFunction(NS::String* name, MTL::FunctionConstantValues* constantValues, NS::Error** error)
{
    return _MTL_msg_MTL__Functionp_newFunctionWithName_constantValues_error__NS__Stringp_MTL__FunctionConstantValuesp_NS__Errorpp((const void*)this, nullptr, name, constantValues, error);
}

_MTL_INLINE void MTL::Library::newFunction(NS::String* name, MTL::FunctionConstantValues* constantValues, MTL::NewFunctionBlock completionHandler)
{
    _MTL_msg_v_newFunctionWithName_constantValues_completionHandler__NS__Stringp_MTL__FunctionConstantValuesp_MTL__NewFunctionBlock((const void*)this, nullptr, name, constantValues, completionHandler);
}

_MTL_INLINE void MTL::Library::newFunction(NS::String* name, MTL::FunctionConstantValues* constantValues, const MTL::NewFunctionFunction& completionHandler)
{
    __block MTL::NewFunctionFunction blockFunction = completionHandler;
    newFunction(name, constantValues, ^(MTL::Function* x0, NS::Error* x1) { blockFunction(x0, x1); });
}

_MTL_INLINE MTL::FunctionReflection* MTL::Library::reflection(NS::String* functionName)
{
    return _MTL_msg_MTL__FunctionReflectionp_reflectionForFunctionWithName__NS__Stringp((const void*)this, nullptr, functionName);
}

_MTL_INLINE void MTL::Library::newFunction(MTL::FunctionDescriptor* descriptor, MTL::NewFunctionBlock completionHandler)
{
    _MTL_msg_v_newFunctionWithDescriptor_completionHandler__MTL__FunctionDescriptorp_MTL__NewFunctionBlock((const void*)this, nullptr, descriptor, completionHandler);
}

_MTL_INLINE void MTL::Library::newFunction(MTL::FunctionDescriptor* descriptor, const MTL::NewFunctionFunction& completionHandler)
{
    __block MTL::NewFunctionFunction blockFunction = completionHandler;
    newFunction(descriptor, ^(MTL::Function* x0, NS::Error* x1) { blockFunction(x0, x1); });
}

_MTL_INLINE MTL::Function* MTL::Library::newFunction(MTL::FunctionDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL__Functionp_newFunctionWithDescriptor_error__MTL__FunctionDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE void MTL::Library::newIntersectionFunction(MTL::IntersectionFunctionDescriptor* descriptor, MTL::NewFunctionBlock completionHandler)
{
    _MTL_msg_v_newIntersectionFunctionWithDescriptor_completionHandler__MTL__IntersectionFunctionDescriptorp_MTL__NewFunctionBlock((const void*)this, nullptr, descriptor, completionHandler);
}

_MTL_INLINE void MTL::Library::newIntersectionFunction(MTL::IntersectionFunctionDescriptor* descriptor, const MTL::NewFunctionFunction& completionHandler)
{
    __block MTL::NewFunctionFunction blockFunction = completionHandler;
    newIntersectionFunction(descriptor, ^(MTL::Function* x0, NS::Error* x1) { blockFunction(x0, x1); });
}

_MTL_INLINE MTL::Function* MTL::Library::newIntersectionFunction(MTL::IntersectionFunctionDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL__Functionp_newIntersectionFunctionWithDescriptor_error__MTL__IntersectionFunctionDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}
