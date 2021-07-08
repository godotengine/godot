//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
#ifndef COMPILER_TRANSLATOR_PARSECONTEXT_H_
#define COMPILER_TRANSLATOR_PARSECONTEXT_H_

#include "compiler/preprocessor/Preprocessor.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/Declarator.h"
#include "compiler/translator/Diagnostics.h"
#include "compiler/translator/DirectiveHandler.h"
#include "compiler/translator/FunctionLookup.h"
#include "compiler/translator/QualifierTypes.h"
#include "compiler/translator/SymbolTable.h"

namespace sh
{

struct TMatrixFields
{
    bool wholeRow;
    bool wholeCol;
    int row;
    int col;
};

//
// The following are extra variables needed during parsing, grouped together so
// they can be passed to the parser without needing a global.
//
class TParseContext : angle::NonCopyable
{
  public:
    TParseContext(TSymbolTable &symt,
                  TExtensionBehavior &ext,
                  sh::GLenum type,
                  ShShaderSpec spec,
                  ShCompileOptions options,
                  bool checksPrecErrors,
                  TDiagnostics *diagnostics,
                  const ShBuiltInResources &resources,
                  ShShaderOutput outputType);
    ~TParseContext();

    bool anyMultiviewExtensionAvailable();
    const angle::pp::Preprocessor &getPreprocessor() const { return mPreprocessor; }
    angle::pp::Preprocessor &getPreprocessor() { return mPreprocessor; }
    void *getScanner() const { return mScanner; }
    void setScanner(void *scanner) { mScanner = scanner; }
    int getShaderVersion() const { return mShaderVersion; }
    sh::GLenum getShaderType() const { return mShaderType; }
    ShShaderSpec getShaderSpec() const { return mShaderSpec; }
    int numErrors() const { return mDiagnostics->numErrors(); }
    void error(const TSourceLoc &loc, const char *reason, const char *token);
    void error(const TSourceLoc &loc, const char *reason, const ImmutableString &token);
    void warning(const TSourceLoc &loc, const char *reason, const char *token);

    // If isError is false, a warning will be reported instead.
    void outOfRangeError(bool isError,
                         const TSourceLoc &loc,
                         const char *reason,
                         const char *token);

    TIntermBlock *getTreeRoot() const { return mTreeRoot; }
    void setTreeRoot(TIntermBlock *treeRoot) { mTreeRoot = treeRoot; }

    bool getFragmentPrecisionHigh() const
    {
        return mFragmentPrecisionHighOnESSL1 || mShaderVersion >= 300;
    }
    void setFragmentPrecisionHighOnESSL1(bool fragmentPrecisionHigh)
    {
        mFragmentPrecisionHighOnESSL1 = fragmentPrecisionHigh;
    }

    void setLoopNestingLevel(int loopNestintLevel) { mLoopNestingLevel = loopNestintLevel; }

    void incrLoopNestingLevel() { ++mLoopNestingLevel; }
    void decrLoopNestingLevel() { --mLoopNestingLevel; }

    void incrSwitchNestingLevel() { ++mSwitchNestingLevel; }
    void decrSwitchNestingLevel() { --mSwitchNestingLevel; }

    bool isComputeShaderLocalSizeDeclared() const { return mComputeShaderLocalSizeDeclared; }
    sh::WorkGroupSize getComputeShaderLocalSize() const;

    int getNumViews() const { return mNumViews; }

    void enterFunctionDeclaration() { mDeclaringFunction = true; }

    void exitFunctionDeclaration() { mDeclaringFunction = false; }

    bool declaringFunction() const { return mDeclaringFunction; }

    TIntermConstantUnion *addScalarLiteral(const TConstantUnion *constantUnion,
                                           const TSourceLoc &line);

    // This method is guaranteed to succeed, even if no variable with 'name' exists.
    const TVariable *getNamedVariable(const TSourceLoc &location,
                                      const ImmutableString &name,
                                      const TSymbol *symbol);
    TIntermTyped *parseVariableIdentifier(const TSourceLoc &location,
                                          const ImmutableString &name,
                                          const TSymbol *symbol);

    // Look at a '.' field selector string and change it into offsets for a vector.
    bool parseVectorFields(const TSourceLoc &line,
                           const ImmutableString &compString,
                           int vecSize,
                           TVector<int> *fieldOffsets);

    void assignError(const TSourceLoc &line, const char *op, const TType &left, const TType &right);
    void unaryOpError(const TSourceLoc &line, const char *op, const TType &operand);
    void binaryOpError(const TSourceLoc &line,
                       const char *op,
                       const TType &left,
                       const TType &right);

    // Check functions - the ones that return bool return false if an error was generated.

    bool checkIsNotReserved(const TSourceLoc &line, const ImmutableString &identifier);
    void checkPrecisionSpecified(const TSourceLoc &line, TPrecision precision, TBasicType type);
    bool checkCanBeLValue(const TSourceLoc &line, const char *op, TIntermTyped *node);
    void checkIsConst(TIntermTyped *node);
    void checkIsScalarInteger(TIntermTyped *node, const char *token);
    bool checkIsAtGlobalLevel(const TSourceLoc &line, const char *token);
    bool checkConstructorArguments(const TSourceLoc &line,
                                   const TIntermSequence &arguments,
                                   const TType &type);

    // Returns a sanitized array size to use (the size is at least 1).
    unsigned int checkIsValidArraySize(const TSourceLoc &line, TIntermTyped *expr);
    bool checkIsValidQualifierForArray(const TSourceLoc &line, const TPublicType &elementQualifier);
    bool checkArrayElementIsNotArray(const TSourceLoc &line, const TPublicType &elementType);
    bool checkIsNonVoid(const TSourceLoc &line,
                        const ImmutableString &identifier,
                        const TBasicType &type);
    bool checkIsScalarBool(const TSourceLoc &line, const TIntermTyped *type);
    void checkIsScalarBool(const TSourceLoc &line, const TPublicType &pType);
    bool checkIsNotOpaqueType(const TSourceLoc &line,
                              const TTypeSpecifierNonArray &pType,
                              const char *reason);
    void checkDeclaratorLocationIsNotSpecified(const TSourceLoc &line, const TPublicType &pType);
    void checkLocationIsNotSpecified(const TSourceLoc &location,
                                     const TLayoutQualifier &layoutQualifier);
    void checkStd430IsForShaderStorageBlock(const TSourceLoc &location,
                                            const TLayoutBlockStorage &blockStorage,
                                            const TQualifier &qualifier);
    void checkIsParameterQualifierValid(const TSourceLoc &line,
                                        const TTypeQualifierBuilder &typeQualifierBuilder,
                                        TType *type);

    // Check if at least one of the specified extensions can be used, and generate error/warning as
    // appropriate according to the spec.
    // This function is only needed for a few different small constant sizes of extension array, and
    // we want to avoid unnecessary dynamic allocations. That's why checkCanUseOneOfExtensions is a
    // template function rather than one taking a vector.
    template <size_t size>
    bool checkCanUseOneOfExtensions(const TSourceLoc &line,
                                    const std::array<TExtension, size> &extensions);
    bool checkCanUseExtension(const TSourceLoc &line, TExtension extension);

    // Done for all declarations, whether empty or not.
    void declarationQualifierErrorCheck(const sh::TQualifier qualifier,
                                        const sh::TLayoutQualifier &layoutQualifier,
                                        const TSourceLoc &location);
    // Done for the first non-empty declarator in a declaration.
    void nonEmptyDeclarationErrorCheck(const TPublicType &publicType,
                                       const TSourceLoc &identifierLocation);
    // Done only for empty declarations.
    void emptyDeclarationErrorCheck(const TType &type, const TSourceLoc &location);

    void checkLayoutQualifierSupported(const TSourceLoc &location,
                                       const ImmutableString &layoutQualifierName,
                                       int versionRequired);
    bool checkWorkGroupSizeIsNotSpecified(const TSourceLoc &location,
                                          const TLayoutQualifier &layoutQualifier);
    void functionCallRValueLValueErrorCheck(const TFunction *fnCandidate, TIntermAggregate *fnCall);
    void checkInvariantVariableQualifier(bool invariant,
                                         const TQualifier qualifier,
                                         const TSourceLoc &invariantLocation);
    void checkInputOutputTypeIsValidES3(const TQualifier qualifier,
                                        const TPublicType &type,
                                        const TSourceLoc &qualifierLocation);
    void checkLocalVariableConstStorageQualifier(const TQualifierWrapperBase &qualifier);
    const TPragma &pragma() const { return mDirectiveHandler.pragma(); }
    const TExtensionBehavior &extensionBehavior() const
    {
        return mDirectiveHandler.extensionBehavior();
    }

    bool isExtensionEnabled(TExtension extension) const;
    void handleExtensionDirective(const TSourceLoc &loc, const char *extName, const char *behavior);
    void handlePragmaDirective(const TSourceLoc &loc,
                               const char *name,
                               const char *value,
                               bool stdgl);

    // Returns true on success. *initNode may still be nullptr on success in case the initialization
    // is not needed in the AST.
    bool executeInitializer(const TSourceLoc &line,
                            const ImmutableString &identifier,
                            TType *type,
                            TIntermTyped *initializer,
                            TIntermBinary **initNode);
    TIntermNode *addConditionInitializer(const TPublicType &pType,
                                         const ImmutableString &identifier,
                                         TIntermTyped *initializer,
                                         const TSourceLoc &loc);
    TIntermNode *addLoop(TLoopType type,
                         TIntermNode *init,
                         TIntermNode *cond,
                         TIntermTyped *expr,
                         TIntermNode *body,
                         const TSourceLoc &loc);

    // For "if" test nodes. There are three children: a condition, a true path, and a false path.
    // The two paths are in TIntermNodePair code.
    TIntermNode *addIfElse(TIntermTyped *cond, TIntermNodePair code, const TSourceLoc &loc);

    void addFullySpecifiedType(TPublicType *typeSpecifier);
    TPublicType addFullySpecifiedType(const TTypeQualifierBuilder &typeQualifierBuilder,
                                      const TPublicType &typeSpecifier);

    TIntermDeclaration *parseSingleDeclaration(TPublicType &publicType,
                                               const TSourceLoc &identifierOrTypeLocation,
                                               const ImmutableString &identifier);
    TIntermDeclaration *parseSingleArrayDeclaration(TPublicType &elementType,
                                                    const TSourceLoc &identifierLocation,
                                                    const ImmutableString &identifier,
                                                    const TSourceLoc &indexLocation,
                                                    const TVector<unsigned int> &arraySizes);
    TIntermDeclaration *parseSingleInitDeclaration(const TPublicType &publicType,
                                                   const TSourceLoc &identifierLocation,
                                                   const ImmutableString &identifier,
                                                   const TSourceLoc &initLocation,
                                                   TIntermTyped *initializer);

    // Parse a declaration like "type a[n] = initializer"
    // Note that this does not apply to declarations like "type[n] a = initializer"
    TIntermDeclaration *parseSingleArrayInitDeclaration(TPublicType &elementType,
                                                        const TSourceLoc &identifierLocation,
                                                        const ImmutableString &identifier,
                                                        const TSourceLoc &indexLocation,
                                                        const TVector<unsigned int> &arraySizes,
                                                        const TSourceLoc &initLocation,
                                                        TIntermTyped *initializer);

    TIntermInvariantDeclaration *parseInvariantDeclaration(
        const TTypeQualifierBuilder &typeQualifierBuilder,
        const TSourceLoc &identifierLoc,
        const ImmutableString &identifier,
        const TSymbol *symbol);

    void parseDeclarator(TPublicType &publicType,
                         const TSourceLoc &identifierLocation,
                         const ImmutableString &identifier,
                         TIntermDeclaration *declarationOut);
    void parseArrayDeclarator(TPublicType &elementType,
                              const TSourceLoc &identifierLocation,
                              const ImmutableString &identifier,
                              const TSourceLoc &arrayLocation,
                              const TVector<unsigned int> &arraySizes,
                              TIntermDeclaration *declarationOut);
    void parseInitDeclarator(const TPublicType &publicType,
                             const TSourceLoc &identifierLocation,
                             const ImmutableString &identifier,
                             const TSourceLoc &initLocation,
                             TIntermTyped *initializer,
                             TIntermDeclaration *declarationOut);

    // Parse a declarator like "a[n] = initializer"
    void parseArrayInitDeclarator(const TPublicType &elementType,
                                  const TSourceLoc &identifierLocation,
                                  const ImmutableString &identifier,
                                  const TSourceLoc &indexLocation,
                                  const TVector<unsigned int> &arraySizes,
                                  const TSourceLoc &initLocation,
                                  TIntermTyped *initializer,
                                  TIntermDeclaration *declarationOut);

    TIntermNode *addEmptyStatement(const TSourceLoc &location);

    void parseDefaultPrecisionQualifier(const TPrecision precision,
                                        const TPublicType &type,
                                        const TSourceLoc &loc);
    void parseGlobalLayoutQualifier(const TTypeQualifierBuilder &typeQualifierBuilder);

    TIntermFunctionPrototype *addFunctionPrototypeDeclaration(const TFunction &parsedFunction,
                                                              const TSourceLoc &location);
    TIntermFunctionDefinition *addFunctionDefinition(TIntermFunctionPrototype *functionPrototype,
                                                     TIntermBlock *functionBody,
                                                     const TSourceLoc &location);
    void parseFunctionDefinitionHeader(const TSourceLoc &location,
                                       const TFunction *function,
                                       TIntermFunctionPrototype **prototypeOut);
    TFunction *parseFunctionDeclarator(const TSourceLoc &location, TFunction *function);
    TFunction *parseFunctionHeader(const TPublicType &type,
                                   const ImmutableString &name,
                                   const TSourceLoc &location);

    TFunctionLookup *addNonConstructorFunc(const ImmutableString &name, const TSymbol *symbol);
    TFunctionLookup *addConstructorFunc(const TPublicType &publicType);

    TParameter parseParameterDeclarator(const TPublicType &publicType,
                                        const ImmutableString &name,
                                        const TSourceLoc &nameLoc);

    TParameter parseParameterArrayDeclarator(const ImmutableString &name,
                                             const TSourceLoc &nameLoc,
                                             const TVector<unsigned int> &arraySizes,
                                             const TSourceLoc &arrayLoc,
                                             TPublicType *elementType);

    TIntermTyped *addIndexExpression(TIntermTyped *baseExpression,
                                     const TSourceLoc &location,
                                     TIntermTyped *indexExpression);
    TIntermTyped *addFieldSelectionExpression(TIntermTyped *baseExpression,
                                              const TSourceLoc &dotLocation,
                                              const ImmutableString &fieldString,
                                              const TSourceLoc &fieldLocation);

    // Parse declarator for a single field
    TDeclarator *parseStructDeclarator(const ImmutableString &identifier, const TSourceLoc &loc);
    TDeclarator *parseStructArrayDeclarator(const ImmutableString &identifier,
                                            const TSourceLoc &loc,
                                            const TVector<unsigned int> *arraySizes);

    void checkDoesNotHaveDuplicateFieldName(const TFieldList::const_iterator begin,
                                            const TFieldList::const_iterator end,
                                            const ImmutableString &name,
                                            const TSourceLoc &location);
    TFieldList *addStructFieldList(TFieldList *fields, const TSourceLoc &location);
    TFieldList *combineStructFieldLists(TFieldList *processedFields,
                                        const TFieldList *newlyAddedFields,
                                        const TSourceLoc &location);
    TFieldList *addStructDeclaratorListWithQualifiers(
        const TTypeQualifierBuilder &typeQualifierBuilder,
        TPublicType *typeSpecifier,
        const TDeclaratorList *declaratorList);
    TFieldList *addStructDeclaratorList(const TPublicType &typeSpecifier,
                                        const TDeclaratorList *declaratorList);
    TTypeSpecifierNonArray addStructure(const TSourceLoc &structLine,
                                        const TSourceLoc &nameLine,
                                        const ImmutableString &structName,
                                        TFieldList *fieldList);

    TIntermDeclaration *addInterfaceBlock(const TTypeQualifierBuilder &typeQualifierBuilder,
                                          const TSourceLoc &nameLine,
                                          const ImmutableString &blockName,
                                          TFieldList *fieldList,
                                          const ImmutableString &instanceName,
                                          const TSourceLoc &instanceLine,
                                          TIntermTyped *arrayIndex,
                                          const TSourceLoc &arrayIndexLine);

    void parseLocalSize(const ImmutableString &qualifierType,
                        const TSourceLoc &qualifierTypeLine,
                        int intValue,
                        const TSourceLoc &intValueLine,
                        const std::string &intValueString,
                        size_t index,
                        sh::WorkGroupSize *localSize);
    void parseNumViews(int intValue,
                       const TSourceLoc &intValueLine,
                       const std::string &intValueString,
                       int *numViews);
    void parseInvocations(int intValue,
                          const TSourceLoc &intValueLine,
                          const std::string &intValueString,
                          int *numInvocations);
    void parseMaxVertices(int intValue,
                          const TSourceLoc &intValueLine,
                          const std::string &intValueString,
                          int *numMaxVertices);
    void parseIndexLayoutQualifier(int intValue,
                                   const TSourceLoc &intValueLine,
                                   const std::string &intValueString,
                                   int *index);
    TLayoutQualifier parseLayoutQualifier(const ImmutableString &qualifierType,
                                          const TSourceLoc &qualifierTypeLine);
    TLayoutQualifier parseLayoutQualifier(const ImmutableString &qualifierType,
                                          const TSourceLoc &qualifierTypeLine,
                                          int intValue,
                                          const TSourceLoc &intValueLine);
    TTypeQualifierBuilder *createTypeQualifierBuilder(const TSourceLoc &loc);
    TStorageQualifierWrapper *parseGlobalStorageQualifier(TQualifier qualifier,
                                                          const TSourceLoc &loc);
    TStorageQualifierWrapper *parseVaryingQualifier(const TSourceLoc &loc);
    TStorageQualifierWrapper *parseInQualifier(const TSourceLoc &loc);
    TStorageQualifierWrapper *parseOutQualifier(const TSourceLoc &loc);
    TStorageQualifierWrapper *parseInOutQualifier(const TSourceLoc &loc);
    TLayoutQualifier joinLayoutQualifiers(TLayoutQualifier leftQualifier,
                                          TLayoutQualifier rightQualifier,
                                          const TSourceLoc &rightQualifierLocation);

    // Performs an error check for embedded struct declarations.
    void enterStructDeclaration(const TSourceLoc &line, const ImmutableString &identifier);
    void exitStructDeclaration();

    void checkIsBelowStructNestingLimit(const TSourceLoc &line, const TField &field);

    TIntermSwitch *addSwitch(TIntermTyped *init,
                             TIntermBlock *statementList,
                             const TSourceLoc &loc);
    TIntermCase *addCase(TIntermTyped *condition, const TSourceLoc &loc);
    TIntermCase *addDefault(const TSourceLoc &loc);

    TIntermTyped *addUnaryMath(TOperator op, TIntermTyped *child, const TSourceLoc &loc);
    TIntermTyped *addUnaryMathLValue(TOperator op, TIntermTyped *child, const TSourceLoc &loc);
    TIntermTyped *addBinaryMath(TOperator op,
                                TIntermTyped *left,
                                TIntermTyped *right,
                                const TSourceLoc &loc);
    TIntermTyped *addBinaryMathBooleanResult(TOperator op,
                                             TIntermTyped *left,
                                             TIntermTyped *right,
                                             const TSourceLoc &loc);
    TIntermTyped *addAssign(TOperator op,
                            TIntermTyped *left,
                            TIntermTyped *right,
                            const TSourceLoc &loc);

    TIntermTyped *addComma(TIntermTyped *left, TIntermTyped *right, const TSourceLoc &loc);

    TIntermBranch *addBranch(TOperator op, const TSourceLoc &loc);
    TIntermBranch *addBranch(TOperator op, TIntermTyped *expression, const TSourceLoc &loc);

    void appendStatement(TIntermBlock *block, TIntermNode *statement);

    void checkTextureGather(TIntermAggregate *functionCall);
    void checkTextureOffsetConst(TIntermAggregate *functionCall);
    void checkImageMemoryAccessForBuiltinFunctions(TIntermAggregate *functionCall);
    void checkImageMemoryAccessForUserDefinedFunctions(const TFunction *functionDefinition,
                                                       const TIntermAggregate *functionCall);
    void checkAtomicMemoryBuiltinFunctions(TIntermAggregate *functionCall);

    // fnCall is only storing the built-in op, and function name or constructor type. arguments
    // has the arguments.
    TIntermTyped *addFunctionCallOrMethod(TFunctionLookup *fnCall, const TSourceLoc &loc);

    TIntermTyped *addTernarySelection(TIntermTyped *cond,
                                      TIntermTyped *trueExpression,
                                      TIntermTyped *falseExpression,
                                      const TSourceLoc &line);

    int getGeometryShaderMaxVertices() const { return mGeometryShaderMaxVertices; }
    int getGeometryShaderInvocations() const
    {
        return (mGeometryShaderInvocations > 0) ? mGeometryShaderInvocations : 1;
    }
    TLayoutPrimitiveType getGeometryShaderInputPrimitiveType() const
    {
        return mGeometryShaderInputPrimitiveType;
    }
    TLayoutPrimitiveType getGeometryShaderOutputPrimitiveType() const
    {
        return mGeometryShaderOutputPrimitiveType;
    }

    ShShaderOutput getOutputType() const { return mOutputType; }

    // TODO(jmadill): make this private
    TSymbolTable &symbolTable;  // symbol table that goes with the language currently being parsed

  private:
    class AtomicCounterBindingState;
    constexpr static size_t kAtomicCounterSize = 4;
    // UNIFORM_ARRAY_STRIDE for atomic counter arrays is an implementation-dependent value which
    // can be queried after a program is linked according to ES 3.10 section 7.7.1. This is
    // controversial with the offset inheritance as described in ESSL 3.10 section 4.4.6. Currently
    // we treat it as always 4 in favour of the original interpretation in
    // "ARB_shader_atomic_counters".
    // TODO(jie.a.chen@intel.com): Double check this once the spec vagueness is resolved.
    // Note that there may be tests in AtomicCounter_test that will need to be updated as well.
    constexpr static size_t kAtomicCounterArrayStride = 4;

    void markStaticReadIfSymbol(TIntermNode *node);

    // Returns a clamped index. If it prints out an error message, the token is "[]".
    int checkIndexLessThan(bool outOfRangeIndexIsError,
                           const TSourceLoc &location,
                           int index,
                           int arraySize,
                           const char *reason);

    bool declareVariable(const TSourceLoc &line,
                         const ImmutableString &identifier,
                         const TType *type,
                         TVariable **variable);

    void checkCanBeDeclaredWithoutInitializer(const TSourceLoc &line,
                                              const ImmutableString &identifier,
                                              TType *type);

    TParameter parseParameterDeclarator(TType *type,
                                        const ImmutableString &name,
                                        const TSourceLoc &nameLoc);

    bool checkIsValidTypeAndQualifierForArray(const TSourceLoc &indexLocation,
                                              const TPublicType &elementType);
    // Done for all atomic counter declarations, whether empty or not.
    void atomicCounterQualifierErrorCheck(const TPublicType &publicType,
                                          const TSourceLoc &location);

    // Assumes that multiplication op has already been set based on the types.
    bool isMultiplicationTypeCombinationValid(TOperator op, const TType &left, const TType &right);

    void checkOutParameterIsNotOpaqueType(const TSourceLoc &line,
                                          TQualifier qualifier,
                                          const TType &type);

    void checkInternalFormatIsNotSpecified(const TSourceLoc &location,
                                           TLayoutImageInternalFormat internalFormat);
    void checkMemoryQualifierIsNotSpecified(const TMemoryQualifier &memoryQualifier,
                                            const TSourceLoc &location);
    void checkAtomicCounterOffsetDoesNotOverlap(bool forceAppend,
                                                const TSourceLoc &loc,
                                                TType *type);
    void checkAtomicCounterOffsetAlignment(const TSourceLoc &location, const TType &type);

    void checkIndexIsNotSpecified(const TSourceLoc &location, int index);
    void checkBindingIsValid(const TSourceLoc &identifierLocation, const TType &type);
    void checkBindingIsNotSpecified(const TSourceLoc &location, int binding);
    void checkOffsetIsNotSpecified(const TSourceLoc &location, int offset);
    void checkImageBindingIsValid(const TSourceLoc &location,
                                  int binding,
                                  int arrayTotalElementCount);
    void checkSamplerBindingIsValid(const TSourceLoc &location,
                                    int binding,
                                    int arrayTotalElementCount);
    void checkBlockBindingIsValid(const TSourceLoc &location,
                                  const TQualifier &qualifier,
                                  int binding,
                                  int arraySize);
    void checkAtomicCounterBindingIsValid(const TSourceLoc &location, int binding);

    void checkUniformLocationInRange(const TSourceLoc &location,
                                     int objectLocationCount,
                                     const TLayoutQualifier &layoutQualifier);

    void checkYuvIsNotSpecified(const TSourceLoc &location, bool yuv);

    bool checkUnsizedArrayConstructorArgumentDimensionality(const TIntermSequence &arguments,
                                                            TType type,
                                                            const TSourceLoc &line);

    // Will set the size of the outermost array according to geometry shader input layout.
    void checkGeometryShaderInputAndSetArraySize(const TSourceLoc &location,
                                                 const ImmutableString &token,
                                                 TType *type);

    // Will size any unsized array type so unsized arrays won't need to be taken into account
    // further along the line in parsing.
    void checkIsNotUnsizedArray(const TSourceLoc &line,
                                const char *errorMessage,
                                const ImmutableString &token,
                                TType *arrayType);

    TIntermTyped *addBinaryMathInternal(TOperator op,
                                        TIntermTyped *left,
                                        TIntermTyped *right,
                                        const TSourceLoc &loc);
    TIntermTyped *createUnaryMath(TOperator op,
                                  TIntermTyped *child,
                                  const TSourceLoc &loc,
                                  const TFunction *func);

    TIntermTyped *addMethod(TFunctionLookup *fnCall, const TSourceLoc &loc);
    TIntermTyped *addConstructor(TFunctionLookup *fnCall, const TSourceLoc &line);
    TIntermTyped *addNonConstructorFunctionCall(TFunctionLookup *fnCall, const TSourceLoc &loc);

    // Return either the original expression or the folded version of the expression in case the
    // folded node will validate the same way during subsequent parsing.
    TIntermTyped *expressionOrFoldedResult(TIntermTyped *expression);

    // Return true if the checks pass
    bool binaryOpCommonCheck(TOperator op,
                             TIntermTyped *left,
                             TIntermTyped *right,
                             const TSourceLoc &loc);

    TIntermFunctionPrototype *createPrototypeNodeFromFunction(const TFunction &function,
                                                              const TSourceLoc &location,
                                                              bool insertParametersToSymbolTable);

    void setAtomicCounterBindingDefaultOffset(const TPublicType &declaration,
                                              const TSourceLoc &location);

    bool checkPrimitiveTypeMatchesTypeQualifier(const TTypeQualifier &typeQualifier);
    bool parseGeometryShaderInputLayoutQualifier(const TTypeQualifier &typeQualifier);
    bool parseGeometryShaderOutputLayoutQualifier(const TTypeQualifier &typeQualifier);
    void setGeometryShaderInputArraySize(unsigned int inputArraySize, const TSourceLoc &line);

    // Set to true when the last/current declarator list was started with an empty declaration. The
    // non-empty declaration error check will need to be performed if the empty declaration is
    // followed by a declarator.
    bool mDeferredNonEmptyDeclarationErrorCheck;

    sh::GLenum mShaderType;    // vertex or fragment language (future: pack or unpack)
    ShShaderSpec mShaderSpec;  // The language specification compiler conforms to - GLES2 or WebGL.
    ShCompileOptions mCompileOptions;  // Options passed to TCompiler
    int mShaderVersion;
    TIntermBlock *mTreeRoot;  // root of parse tree being created
    int mLoopNestingLevel;    // 0 if outside all loops
    int mStructNestingLevel;  // incremented while parsing a struct declaration
    int mSwitchNestingLevel;  // 0 if outside all switch statements
    const TType
        *mCurrentFunctionType;    // the return type of the function that's currently being parsed
    bool mFunctionReturnsValue;   // true if a non-void function has a return
    bool mChecksPrecisionErrors;  // true if an error will be generated when a variable is declared
                                  // without precision, explicit or implicit.
    bool mFragmentPrecisionHighOnESSL1;  // true if highp precision is supported when compiling
                                         // ESSL1.
    TLayoutMatrixPacking mDefaultUniformMatrixPacking;
    TLayoutBlockStorage mDefaultUniformBlockStorage;
    TLayoutMatrixPacking mDefaultBufferMatrixPacking;
    TLayoutBlockStorage mDefaultBufferBlockStorage;
    TString mHashErrMsg;
    TDiagnostics *mDiagnostics;
    TDirectiveHandler mDirectiveHandler;
    angle::pp::Preprocessor mPreprocessor;
    void *mScanner;
    int mMinProgramTexelOffset;
    int mMaxProgramTexelOffset;

    int mMinProgramTextureGatherOffset;
    int mMaxProgramTextureGatherOffset;

    // keep track of local group size declared in layout. It should be declared only once.
    bool mComputeShaderLocalSizeDeclared;
    sh::WorkGroupSize mComputeShaderLocalSize;
    // keep track of number of views declared in layout.
    int mNumViews;
    int mMaxNumViews;
    int mMaxImageUnits;
    int mMaxCombinedTextureImageUnits;
    int mMaxUniformLocations;
    int mMaxUniformBufferBindings;
    int mMaxAtomicCounterBindings;
    int mMaxShaderStorageBufferBindings;

    // keeps track whether we are declaring / defining a function
    bool mDeclaringFunction;

    // Track the state of each atomic counter binding.
    std::map<int, AtomicCounterBindingState> mAtomicCounterBindingStates;

    // Track the geometry shader global parameters declared in layout.
    TLayoutPrimitiveType mGeometryShaderInputPrimitiveType;
    TLayoutPrimitiveType mGeometryShaderOutputPrimitiveType;
    int mGeometryShaderInvocations;
    int mGeometryShaderMaxVertices;
    int mMaxGeometryShaderInvocations;
    int mMaxGeometryShaderMaxVertices;

    // Track when we add new scope for func body in ESSL 1.00 spec
    bool mFunctionBodyNewScope;

    ShShaderOutput mOutputType;
};

int PaParseStrings(size_t count,
                   const char *const string[],
                   const int length[],
                   TParseContext *context);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_PARSECONTEXT_H_
