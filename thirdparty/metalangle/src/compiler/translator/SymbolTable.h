//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_SYMBOLTABLE_H_
#define COMPILER_TRANSLATOR_SYMBOLTABLE_H_

//
// Symbol table for parsing.  Has these design characteristics:
//
// * Same symbol table can be used to compile many shaders, to preserve
//   effort of creating and loading with the large numbers of built-in
//   symbols.
//
// * Name mangling will be used to give each function a unique name
//   so that symbol table lookups are never ambiguous.  This allows
//   a simpler symbol table structure.
//
// * Pushing and popping of scope, so symbol table will really be a stack
//   of symbol tables.  Searched from the top, with new inserts going into
//   the top.
//
// * Constants:  Compile time constant symbols will keep their values
//   in the symbol table.  The parser can substitute constants at parse
//   time, including doing constant folding and constant propagation.
//
// * No temporaries:  Temporaries made from operations (+, --, .xy, etc.)
//   are tracked in the intermediate representation, not the symbol table.
//

#include <limits>
#include <memory>
#include <set>

#include "common/angleutils.h"
#include "compiler/translator/ExtensionBehavior.h"
#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/InfoSink.h"
#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/SymbolTable_autogen.h"

enum class Shader
{
    ALL,
    FRAGMENT,      // GL_FRAGMENT_SHADER
    VERTEX,        // GL_VERTEX_SHADER
    COMPUTE,       // GL_COMPUTE_SHADER
    GEOMETRY,      // GL_GEOMETRY_SHADER
    GEOMETRY_EXT,  // GL_GEOMETRY_SHADER_EXT
    NOT_COMPUTE
};

namespace sh
{

struct UnmangledBuiltIn
{
    constexpr UnmangledBuiltIn(TExtension extension) : extension(extension) {}

    TExtension extension;
};

using VarPointer        = TSymbol *(TSymbolTableBase::*);
using ValidateExtension = int(ShBuiltInResources::*);

enum class Spec
{
    GLSL,
    ESSL
};

constexpr uint16_t kESSL1Only = 100;

static_assert(offsetof(ShBuiltInResources, OES_standard_derivatives) != 0,
              "Update SymbolTable extension logic");

#define EXT_INDEX(Ext) (offsetof(ShBuiltInResources, Ext) / sizeof(int))

class SymbolRule
{
  public:
    const TSymbol *get(ShShaderSpec shaderSpec,
                       int shaderVersion,
                       sh::GLenum shaderType,
                       const ShBuiltInResources &resources,
                       const TSymbolTableBase &symbolTable) const;

    template <Spec spec, int version, Shader shaders, size_t extensionIndex, typename T>
    constexpr static SymbolRule Get(T value);

  private:
    constexpr SymbolRule(Spec spec,
                         int version,
                         Shader shaders,
                         size_t extensionIndex,
                         const TSymbol *symbol);

    constexpr SymbolRule(Spec spec,
                         int version,
                         Shader shaders,
                         size_t extensionIndex,
                         VarPointer resourceVar);

    union SymbolOrVar
    {
        constexpr SymbolOrVar(const TSymbol *symbolIn) : symbol(symbolIn) {}
        constexpr SymbolOrVar(VarPointer varIn) : var(varIn) {}

        const TSymbol *symbol;
        VarPointer var;
    };

    uint16_t mIsDesktop : 1;
    uint16_t mIsVar : 1;
    uint16_t mVersion : 14;
    uint8_t mShaders;
    uint8_t mExtensionIndex;
    SymbolOrVar mSymbolOrVar;
};

constexpr SymbolRule::SymbolRule(Spec spec,
                                 int version,
                                 Shader shaders,
                                 size_t extensionIndex,
                                 const TSymbol *symbol)
    : mIsDesktop(spec == Spec::GLSL ? 1u : 0u),
      mIsVar(0u),
      mVersion(static_cast<uint16_t>(version)),
      mShaders(static_cast<uint8_t>(shaders)),
      mExtensionIndex(extensionIndex),
      mSymbolOrVar(symbol)
{}

constexpr SymbolRule::SymbolRule(Spec spec,
                                 int version,
                                 Shader shaders,
                                 size_t extensionIndex,
                                 VarPointer resourceVar)
    : mIsDesktop(spec == Spec::GLSL ? 1u : 0u),
      mIsVar(1u),
      mVersion(static_cast<uint16_t>(version)),
      mShaders(static_cast<uint8_t>(shaders)),
      mExtensionIndex(extensionIndex),
      mSymbolOrVar(resourceVar)
{}

template <Spec spec, int version, Shader shaders, size_t extensionIndex, typename T>
// static
constexpr SymbolRule SymbolRule::Get(T value)
{
    static_assert(version < 0x4000u, "version OOR");
    static_assert(static_cast<uint8_t>(shaders) < 0xFFu, "shaders OOR");
    static_assert(static_cast<uint8_t>(extensionIndex) < 0xFF, "extensionIndex OOR");
    return SymbolRule(spec, version, shaders, extensionIndex, value);
}

const TSymbol *FindMangledBuiltIn(ShShaderSpec shaderSpec,
                                  int shaderVersion,
                                  sh::GLenum shaderType,
                                  const ShBuiltInResources &resources,
                                  const TSymbolTableBase &symbolTable,
                                  const SymbolRule *rules,
                                  uint16_t startIndex,
                                  uint16_t endIndex);

class UnmangledEntry
{
  public:
    constexpr UnmangledEntry(const char *name,
                             TExtension esslExtension,
                             TExtension glslExtension,
                             int esslVersion,
                             int glslVersion,
                             Shader shaderType);

    bool matches(const ImmutableString &name,
                 ShShaderSpec shaderSpec,
                 int shaderVersion,
                 sh::GLenum shaderType,
                 const TExtensionBehavior &extensions) const;

  private:
    const char *mName;
    uint8_t mESSLExtension;
    uint8_t mGLSLExtension;
    uint8_t mShaderType;
    uint16_t mESSLVersion;
    uint16_t mGLSLVersion;
};

constexpr UnmangledEntry::UnmangledEntry(const char *name,
                                         TExtension esslExtension,
                                         TExtension glslExtension,
                                         int esslVersion,
                                         int glslVersion,
                                         Shader shaderType)
    : mName(name),
      mESSLExtension(static_cast<uint8_t>(esslExtension)),
      mGLSLExtension(static_cast<uint8_t>(glslExtension)),
      mShaderType(static_cast<uint8_t>(shaderType)),
      mESSLVersion(esslVersion < 0 ? std::numeric_limits<uint16_t>::max()
                                   : static_cast<uint16_t>(esslVersion)),
      mGLSLVersion(glslVersion < 0 ? std::numeric_limits<uint16_t>::max()
                                   : static_cast<uint16_t>(glslVersion))
{}

class TSymbolTable : angle::NonCopyable, TSymbolTableBase
{
  public:
    TSymbolTable();
    // To start using the symbol table after construction:
    // * initializeBuiltIns() needs to be called.
    // * push() needs to be called to push the global level.

    ~TSymbolTable();

    bool isEmpty() const;
    bool atGlobalLevel() const;

    void push();
    void pop();

    // Declare a non-function symbol at the current scope. Return true in case the declaration was
    // successful, and false if the declaration failed due to redefinition.
    bool declare(TSymbol *symbol);

    // Only used to declare internal variables.
    bool declareInternal(TSymbol *symbol);

    // Functions are always declared at global scope.
    void declareUserDefinedFunction(TFunction *function, bool insertUnmangledName);

    // These return the TFunction pointer to keep using to refer to this function.
    const TFunction *markFunctionHasPrototypeDeclaration(const ImmutableString &mangledName,
                                                         bool *hadPrototypeDeclarationOut) const;
    const TFunction *setFunctionParameterNamesFromDefinition(const TFunction *function,
                                                             bool *wasDefinedOut) const;

    // Return false if the gl_in array size has already been initialized with a mismatching value.
    bool setGlInArraySize(unsigned int inputArraySize);
    TVariable *getGlInVariableWithArraySize() const;

    const TVariable *gl_FragData() const;
    const TVariable *gl_SecondaryFragDataEXT() const;

    void markStaticRead(const TVariable &variable);
    void markStaticWrite(const TVariable &variable);

    // Note: Should not call this for constant variables.
    bool isStaticallyUsed(const TVariable &variable) const;

    // find() is guaranteed not to retain a reference to the ImmutableString, so an ImmutableString
    // with a reference to a short-lived char * is fine to pass here.
    const TSymbol *find(const ImmutableString &name, int shaderVersion) const;

    const TSymbol *findUserDefined(const ImmutableString &name) const;

    TFunction *findUserDefinedFunction(const ImmutableString &name) const;

    const TSymbol *findGlobal(const ImmutableString &name) const;
    const TSymbol *findGlobalWithConversion(const std::vector<ImmutableString> &names) const;

    const TSymbol *findBuiltIn(const ImmutableString &name, int shaderVersion) const;
    const TSymbol *findBuiltInWithConversion(const std::vector<ImmutableString> &names,
                                             int shaderVersion) const;

    void setDefaultPrecision(TBasicType type, TPrecision prec);

    // Searches down the precisionStack for a precision qualifier
    // for the specified TBasicType
    TPrecision getDefaultPrecision(TBasicType type) const;

    // This records invariant varyings declared through "invariant varying_name;".
    void addInvariantVarying(const TVariable &variable);

    // If this returns false, the varying could still be invariant if it is set as invariant during
    // the varying variable declaration - this piece of information is stored in the variable's
    // type, not here.
    bool isVaryingInvariant(const TVariable &variable) const;

    void setGlobalInvariant(bool invariant);

    const TSymbolUniqueId nextUniqueId() { return TSymbolUniqueId(this); }

    // Gets the built-in accessible by a shader with the specified version, if any.
    bool isUnmangledBuiltInName(const ImmutableString &name,
                                int shaderVersion,
                                const TExtensionBehavior &extensions) const;

    void initializeBuiltIns(sh::GLenum type,
                            ShShaderSpec spec,
                            const ShBuiltInResources &resources);
    void clearCompilationResults();

  private:
    friend class TSymbolUniqueId;

    struct VariableMetadata
    {
        VariableMetadata();
        bool staticRead;
        bool staticWrite;
        bool invariant;
    };

    int nextUniqueIdValue();

    class TSymbolTableLevel;

    void initSamplerDefaultPrecision(TBasicType samplerType);

    void initializeBuiltInVariables(sh::GLenum shaderType,
                                    ShShaderSpec spec,
                                    const ShBuiltInResources &resources);

    VariableMetadata *getOrCreateVariableMetadata(const TVariable &variable);

    std::vector<std::unique_ptr<TSymbolTableLevel>> mTable;

    // There's one precision stack level for predefined precisions and then one level for each scope
    // in table.
    typedef TMap<TBasicType, TPrecision> PrecisionStackLevel;
    std::vector<std::unique_ptr<PrecisionStackLevel>> mPrecisionStack;

    bool mGlobalInvariant;

    int mUniqueIdCounter;

    static const int kLastBuiltInId;

    sh::GLenum mShaderType;
    ShShaderSpec mShaderSpec;
    ShBuiltInResources mResources;

    // Indexed by unique id. Map instead of vector since the variables are fairly sparse.
    std::map<int, VariableMetadata> mVariableMetadata;

    // Store gl_in variable with its array size once the array size can be determined. The array
    // size can also be checked against latter input primitive type declaration.
    TVariable *mGlInVariableWithArraySize;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_SYMBOLTABLE_H_
