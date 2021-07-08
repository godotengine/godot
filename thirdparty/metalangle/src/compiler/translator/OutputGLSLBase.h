//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_OUTPUTGLSLBASE_H_
#define COMPILER_TRANSLATOR_OUTPUTGLSLBASE_H_

#include <set>

#include "compiler/translator/HashNames.h"
#include "compiler/translator/InfoSink.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

class TOutputGLSLBase : public TIntermTraverser
{
  public:
    TOutputGLSLBase(TInfoSinkBase &objSink,
                    ShArrayIndexClampingStrategy clampingStrategy,
                    ShHashFunction64 hashFunction,
                    NameMap &nameMap,
                    TSymbolTable *symbolTable,
                    sh::GLenum shaderType,
                    int shaderVersion,
                    ShShaderOutput output,
                    ShCompileOptions compileOptions);

    ShShaderOutput getShaderOutput() const { return mOutput; }

    // Return the original name if hash function pointer is NULL;
    // otherwise return the hashed name. Has special handling for internal names and built-ins,
    // which are not hashed.
    ImmutableString hashName(const TSymbol *symbol);

  protected:
    TInfoSinkBase &objSink() { return mObjSink; }
    void writeFloat(TInfoSinkBase &out, float f);
    void writeTriplet(Visit visit, const char *preStr, const char *inStr, const char *postStr);
    std::string getCommonLayoutQualifiers(TIntermTyped *variable);
    std::string getMemoryQualifiers(const TType &type);
    virtual void writeLayoutQualifier(TIntermTyped *variable);
    virtual void writeFieldLayoutQualifier(const TField *field);
    void writeInvariantQualifier(const TType &type);
    virtual void writeVariableType(const TType &type, const TSymbol *symbol);
    virtual bool writeVariablePrecision(TPrecision precision) = 0;
    void writeFunctionParameters(const TFunction *func);
    const TConstantUnion *writeConstantUnion(const TType &type, const TConstantUnion *pConstUnion);
    void writeConstructorTriplet(Visit visit, const TType &type);
    ImmutableString getTypeName(const TType &type);

    void visitSymbol(TIntermSymbol *node) override;
    void visitConstantUnion(TIntermConstantUnion *node) override;
    bool visitSwizzle(Visit visit, TIntermSwizzle *node) override;
    bool visitBinary(Visit visit, TIntermBinary *node) override;
    bool visitUnary(Visit visit, TIntermUnary *node) override;
    bool visitTernary(Visit visit, TIntermTernary *node) override;
    bool visitIfElse(Visit visit, TIntermIfElse *node) override;
    bool visitSwitch(Visit visit, TIntermSwitch *node) override;
    bool visitCase(Visit visit, TIntermCase *node) override;
    void visitFunctionPrototype(TIntermFunctionPrototype *node) override;
    bool visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    bool visitBlock(Visit visit, TIntermBlock *node) override;
    bool visitInvariantDeclaration(Visit visit, TIntermInvariantDeclaration *node) override;
    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override;
    bool visitLoop(Visit visit, TIntermLoop *node) override;
    bool visitBranch(Visit visit, TIntermBranch *node) override;
    void visitPreprocessorDirective(TIntermPreprocessorDirective *node) override;

    void visitCodeBlock(TIntermBlock *node);

    ImmutableString hashFieldName(const TField *field);
    // Same as hashName(), but without hashing "main".
    ImmutableString hashFunctionNameIfNeeded(const TFunction *func);
    // Used to translate function names for differences between ESSL and GLSL
    virtual ImmutableString translateTextureFunction(const ImmutableString &name) { return name; }

    void declareStruct(const TStructure *structure);
    virtual void writeQualifier(TQualifier qualifier, const TType &type, const TSymbol *symbol);
    bool structDeclared(const TStructure *structure) const;

  private:
    void declareInterfaceBlockLayout(const TInterfaceBlock *interfaceBlock);
    void declareInterfaceBlock(const TInterfaceBlock *interfaceBlock);

    void writeBuiltInFunctionTriplet(Visit visit, TOperator op, bool useEmulatedFunction);

    const char *mapQualifierToString(TQualifier qualifier);

    TInfoSinkBase &mObjSink;
    bool mDeclaringVariable;

    // This set contains all the ids of the structs from every scope.
    std::set<int> mDeclaredStructs;

    ShArrayIndexClampingStrategy mClampingStrategy;

    // name hashing.
    ShHashFunction64 mHashFunction;

    NameMap &mNameMap;

    sh::GLenum mShaderType;

    const int mShaderVersion;

    ShShaderOutput mOutput;

    ShCompileOptions mCompileOptions;
};

void WriteGeometryShaderLayoutQualifiers(TInfoSinkBase &out,
                                         sh::TLayoutPrimitiveType inputPrimitive,
                                         int invocations,
                                         sh::TLayoutPrimitiveType outputPrimitive,
                                         int maxVertices);

bool NeedsToWriteLayoutQualifier(const TType &type);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_OUTPUTGLSLBASE_H_
