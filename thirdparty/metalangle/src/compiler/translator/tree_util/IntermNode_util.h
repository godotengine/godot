//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// IntermNode_util.h: High-level utilities for creating AST nodes and node hierarchies. Mostly meant
// to be used in AST transforms.

#ifndef COMPILER_TRANSLATOR_INTERMNODEUTIL_H_
#define COMPILER_TRANSLATOR_INTERMNODEUTIL_H_

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/tree_util/FindFunction.h"

namespace sh
{

class TSymbolTable;
class TVariable;

TIntermFunctionPrototype *CreateInternalFunctionPrototypeNode(const TFunction &func);
TIntermFunctionDefinition *CreateInternalFunctionDefinitionNode(const TFunction &func,
                                                                TIntermBlock *functionBody);

TIntermTyped *CreateZeroNode(const TType &type);
TIntermConstantUnion *CreateFloatNode(float value);
TIntermConstantUnion *CreateIndexNode(int index);
TIntermConstantUnion *CreateUIntNode(unsigned int value);
TIntermConstantUnion *CreateBoolNode(bool value);

TVariable *CreateTempVariable(TSymbolTable *symbolTable, const TType *type);
TVariable *CreateTempVariable(TSymbolTable *symbolTable, const TType *type, TQualifier qualifier);

TIntermSymbol *CreateTempSymbolNode(const TVariable *tempVariable);
TIntermDeclaration *CreateTempDeclarationNode(const TVariable *tempVariable);
TIntermDeclaration *CreateTempInitDeclarationNode(const TVariable *tempVariable,
                                                  TIntermTyped *initializer);
TIntermBinary *CreateTempAssignmentNode(const TVariable *tempVariable, TIntermTyped *rightNode);

TVariable *DeclareTempVariable(TSymbolTable *symbolTable,
                               const TType *type,
                               TQualifier qualifier,
                               TIntermDeclaration **declarationOut);
TVariable *DeclareTempVariable(TSymbolTable *symbolTable,
                               TIntermTyped *initializer,
                               TQualifier qualifier,
                               TIntermDeclaration **declarationOut);
const TVariable *DeclareInterfaceBlock(TIntermBlock *root,
                                       TSymbolTable *symbolTable,
                                       TFieldList *fieldList,
                                       TQualifier qualifier,
                                       const TMemoryQualifier &memoryQualifier,
                                       uint32_t arraySize,
                                       const ImmutableString &blockTypeName,
                                       const ImmutableString &blockVariableName);

// If the input node is nullptr, return nullptr.
// If the input node is a block node, return it.
// If the input node is not a block node, put it inside a block node and return that.
TIntermBlock *EnsureBlock(TIntermNode *node);

// Should be called from inside Compiler::compileTreeImpl() where the global level is in scope.
TIntermSymbol *ReferenceGlobalVariable(const ImmutableString &name,
                                       const TSymbolTable &symbolTable);

// Note: this can't access desktop GLSL built-ins. Those can only be accessed directly through
// BuiltIn.h.
TIntermSymbol *ReferenceBuiltInVariable(const ImmutableString &name,
                                        const TSymbolTable &symbolTable,
                                        int shaderVersion);

TIntermTyped *CreateBuiltInFunctionCallNode(const char *name,
                                            TIntermSequence *arguments,
                                            const TSymbolTable &symbolTable,
                                            int shaderVersion);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_INTERMNODEUTIL_H_
