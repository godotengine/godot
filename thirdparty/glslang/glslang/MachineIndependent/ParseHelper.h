//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2013 LunarG, Inc.
// Copyright (C) 2015-2018 Google, Inc.
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
// This header defines a two-level parse-helper hierarchy, derived from
// TParseVersions:
//  - TParseContextBase:  sharable across multiple parsers
//  - TParseContext:      GLSL specific helper
//

#ifndef _PARSER_HELPER_INCLUDED_
#define _PARSER_HELPER_INCLUDED_

#include <cstdarg>
#include <functional>

#include "parseVersions.h"
#include "../Include/ShHandle.h"
#include "SymbolTable.h"
#include "localintermediate.h"
#include "Scan.h"
#include "attribute.h"

namespace glslang {

struct TPragma {
    TPragma(bool o, bool d) : optimize(o), debug(d) { }
    bool optimize;
    bool debug;
    TPragmaTable pragmaTable;
};

class TScanContext;
class TPpContext;

typedef std::set<int> TIdSetType;
typedef std::map<const TTypeList*, std::map<size_t, const TTypeList*>> TStructRecord;

//
// Sharable code (as well as what's in TParseVersions) across
// parse helpers.
//
class TParseContextBase : public TParseVersions {
public:
    TParseContextBase(TSymbolTable& symbolTable, TIntermediate& interm, bool parsingBuiltins, int version,
                      EProfile profile, const SpvVersion& spvVersion, EShLanguage language,
                      TInfoSink& infoSink, bool forwardCompatible, EShMessages messages,
                      const TString* entryPoint = nullptr)
          : TParseVersions(interm, version, profile, spvVersion, language, infoSink, forwardCompatible, messages),
            scopeMangler("::"),
            symbolTable(symbolTable),
            statementNestingLevel(0), loopNestingLevel(0), structNestingLevel(0), controlFlowNestingLevel(0),
            currentFunctionType(nullptr),
            postEntryPointReturn(false),
            contextPragma(true, false),
            beginInvocationInterlockCount(0), endInvocationInterlockCount(0),
            parsingBuiltins(parsingBuiltins), scanContext(nullptr), ppContext(nullptr),
            limits(resources.limits),
            globalUniformBlock(nullptr),
            globalUniformBinding(TQualifier::layoutBindingEnd),
            globalUniformSet(TQualifier::layoutSetEnd)
    {
        if (entryPoint != nullptr)
            sourceEntryPointName = *entryPoint;
    }
    virtual ~TParseContextBase() { }

#if !defined(GLSLANG_WEB) || defined(GLSLANG_WEB_DEVEL)
    virtual void C_DECL   error(const TSourceLoc&, const char* szReason, const char* szToken,
                                const char* szExtraInfoFormat, ...);
    virtual void C_DECL    warn(const TSourceLoc&, const char* szReason, const char* szToken,
                                const char* szExtraInfoFormat, ...);
    virtual void C_DECL ppError(const TSourceLoc&, const char* szReason, const char* szToken,
                                const char* szExtraInfoFormat, ...);
    virtual void C_DECL  ppWarn(const TSourceLoc&, const char* szReason, const char* szToken,
                                const char* szExtraInfoFormat, ...);
#endif

    virtual void setLimits(const TBuiltInResource&) = 0;

    void checkIndex(const TSourceLoc&, const TType&, int& index);

    EShLanguage getLanguage() const { return language; }
    void setScanContext(TScanContext* c) { scanContext = c; }
    TScanContext* getScanContext() const { return scanContext; }
    void setPpContext(TPpContext* c) { ppContext = c; }
    TPpContext* getPpContext() const { return ppContext; }

    virtual void setLineCallback(const std::function<void(int, int, bool, int, const char*)>& func) { lineCallback = func; }
    virtual void setExtensionCallback(const std::function<void(int, const char*, const char*)>& func) { extensionCallback = func; }
    virtual void setVersionCallback(const std::function<void(int, int, const char*)>& func) { versionCallback = func; }
    virtual void setPragmaCallback(const std::function<void(int, const TVector<TString>&)>& func) { pragmaCallback = func; }
    virtual void setErrorCallback(const std::function<void(int, const char*)>& func) { errorCallback = func; }

    virtual void reservedPpErrorCheck(const TSourceLoc&, const char* name, const char* op) = 0;
    virtual bool lineContinuationCheck(const TSourceLoc&, bool endOfComment) = 0;
    virtual bool lineDirectiveShouldSetNextLine() const = 0;
    virtual void handlePragma(const TSourceLoc&, const TVector<TString>&) = 0;

    virtual bool parseShaderStrings(TPpContext&, TInputScanner& input, bool versionWillBeError = false) = 0;

    virtual void notifyVersion(int line, int version, const char* type_string)
    {
        if (versionCallback)
            versionCallback(line, version, type_string);
    }
    virtual void notifyErrorDirective(int line, const char* error_message)
    {
        if (errorCallback)
            errorCallback(line, error_message);
    }
    virtual void notifyLineDirective(int curLineNo, int newLineNo, bool hasSource, int sourceNum, const char* sourceName)
    {
        if (lineCallback)
            lineCallback(curLineNo, newLineNo, hasSource, sourceNum, sourceName);
    }
    virtual void notifyExtensionDirective(int line, const char* extension, const char* behavior)
    {
        if (extensionCallback)
            extensionCallback(line, extension, behavior);
    }

#ifdef ENABLE_HLSL
    // Manage the global uniform block (default uniforms in GLSL, $Global in HLSL)
    virtual void growGlobalUniformBlock(const TSourceLoc&, TType&, const TString& memberName, TTypeList* typeList = nullptr);
#endif

    // Potentially rename shader entry point function
    void renameShaderFunction(TString*& name) const
    {
        // Replace the entry point name given in the shader with the real entry point name,
        // if there is a substitution.
        if (name != nullptr && *name == sourceEntryPointName && intermediate.getEntryPointName().size() > 0)
            name = NewPoolTString(intermediate.getEntryPointName().c_str());
    }

    virtual bool lValueErrorCheck(const TSourceLoc&, const char* op, TIntermTyped*);
    virtual void rValueErrorCheck(const TSourceLoc&, const char* op, TIntermTyped*);

    const char* const scopeMangler;

    // Basic parsing state, easily accessible to the grammar

    TSymbolTable& symbolTable;        // symbol table that goes with the current language, version, and profile
    int statementNestingLevel;        // 0 if outside all flow control or compound statements
    int loopNestingLevel;             // 0 if outside all loops
    int structNestingLevel;           // 0 if outside blocks and structures
    int controlFlowNestingLevel;      // 0 if outside all flow control
    const TType* currentFunctionType; // the return type of the function that's currently being parsed
    bool functionReturnsValue;        // true if a non-void function has a return
    // if inside a function, true if the function is the entry point and this is after a return statement
    bool postEntryPointReturn;
    // case, node, case, case, node, ...; ensure only one node between cases;   stack of them for nesting
    TList<TIntermSequence*> switchSequenceStack;
    // the statementNestingLevel the current switch statement is at, which must match the level of its case statements
    TList<int> switchLevel;
    struct TPragma contextPragma;
    int beginInvocationInterlockCount;
    int endInvocationInterlockCount;

protected:
    TParseContextBase(TParseContextBase&);
    TParseContextBase& operator=(TParseContextBase&);

    const bool parsingBuiltins;       // true if parsing built-in symbols/functions
    TVector<TSymbol*> linkageSymbols; // will be transferred to 'linkage', after all editing is done, order preserving
    TScanContext* scanContext;
    TPpContext* ppContext;
    TBuiltInResource resources;
    TLimits& limits;
    TString sourceEntryPointName;

    // These, if set, will be called when a line, pragma ... is preprocessed.
    // They will be called with any parameters to the original directive.
    std::function<void(int, int, bool, int, const char*)> lineCallback;
    std::function<void(int, const TVector<TString>&)> pragmaCallback;
    std::function<void(int, int, const char*)> versionCallback;
    std::function<void(int, const char*, const char*)> extensionCallback;
    std::function<void(int, const char*)> errorCallback;

    // see implementation for detail
    const TFunction* selectFunction(const TVector<const TFunction*>, const TFunction&,
        std::function<bool(const TType&, const TType&, TOperator, int arg)>,
        std::function<bool(const TType&, const TType&, const TType&)>,
        /* output */ bool& tie);

    virtual void parseSwizzleSelector(const TSourceLoc&, const TString&, int size,
                                      TSwizzleSelectors<TVectorSelector>&);

    // Manage the global uniform block (default uniforms in GLSL, $Global in HLSL)
    TVariable* globalUniformBlock;     // the actual block, inserted into the symbol table
    unsigned int globalUniformBinding; // the block's binding number
    unsigned int globalUniformSet;     // the block's set number
    int firstNewMember;                // the index of the first member not yet inserted into the symbol table
    // override this to set the language-specific name
    virtual const char* getGlobalUniformBlockName() const { return ""; }
    virtual void setUniformBlockDefaults(TType&) const { }
    virtual void finalizeGlobalUniformBlockLayout(TVariable&) { }
    virtual void outputMessage(const TSourceLoc&, const char* szReason, const char* szToken,
                               const char* szExtraInfoFormat, TPrefixType prefix,
                               va_list args);
    virtual void trackLinkage(TSymbol& symbol);
    virtual void makeEditable(TSymbol*&);
    virtual TVariable* getEditableVariable(const char* name);
    virtual void finish();
};

//
// Manage the state for when to respect precision qualifiers and when to warn about
// the defaults being different than might be expected.
//
class TPrecisionManager {
public:
    TPrecisionManager() : obey(false), warn(false), explicitIntDefault(false), explicitFloatDefault(false){ }
    virtual ~TPrecisionManager() {}

    void respectPrecisionQualifiers() { obey = true; }
    bool respectingPrecisionQualifiers() const { return obey; }
    bool shouldWarnAboutDefaults() const { return warn; }
    void defaultWarningGiven() { warn = false; }
    void warnAboutDefaults() { warn = true; }
    void explicitIntDefaultSeen()
    {
        explicitIntDefault = true;
        if (explicitFloatDefault)
            warn = false;
    }
    void explicitFloatDefaultSeen()
    {
        explicitFloatDefault = true;
        if (explicitIntDefault)
            warn = false;
    }

protected:
    bool obey;                  // respect precision qualifiers
    bool warn;                  // need to give a warning about the defaults
    bool explicitIntDefault;    // user set the default for int/uint
    bool explicitFloatDefault;  // user set the default for float
};

//
// GLSL-specific parse helper.  Should have GLSL in the name, but that's
// too big of a change for comparing branches at the moment, and perhaps
// impacts downstream consumers as well.
//
class TParseContext : public TParseContextBase {
public:
    TParseContext(TSymbolTable&, TIntermediate&, bool parsingBuiltins, int version, EProfile, const SpvVersion& spvVersion, EShLanguage, TInfoSink&,
                  bool forwardCompatible = false, EShMessages messages = EShMsgDefault,
                  const TString* entryPoint = nullptr);
    virtual ~TParseContext();

    bool obeyPrecisionQualifiers() const { return precisionManager.respectingPrecisionQualifiers(); }
    void setPrecisionDefaults();

    void setLimits(const TBuiltInResource&) override;
    bool parseShaderStrings(TPpContext&, TInputScanner& input, bool versionWillBeError = false) override;
    void parserError(const char* s);     // for bison's yyerror

    void reservedErrorCheck(const TSourceLoc&, const TString&);
    void reservedPpErrorCheck(const TSourceLoc&, const char* name, const char* op) override;
    bool lineContinuationCheck(const TSourceLoc&, bool endOfComment) override;
    bool lineDirectiveShouldSetNextLine() const override;
    bool builtInName(const TString&);

    void handlePragma(const TSourceLoc&, const TVector<TString>&) override;
    TIntermTyped* handleVariable(const TSourceLoc&, TSymbol* symbol, const TString* string);
    TIntermTyped* handleBracketDereference(const TSourceLoc&, TIntermTyped* base, TIntermTyped* index);
    void handleIndexLimits(const TSourceLoc&, TIntermTyped* base, TIntermTyped* index);

#ifndef GLSLANG_WEB
    void makeEditable(TSymbol*&) override;
    void ioArrayCheck(const TSourceLoc&, const TType&, const TString& identifier);
#endif
    bool isIoResizeArray(const TType&) const;
    void fixIoArraySize(const TSourceLoc&, TType&);
    void handleIoResizeArrayAccess(const TSourceLoc&, TIntermTyped* base);
    void checkIoArraysConsistency(const TSourceLoc&, bool tailOnly = false);
    int getIoArrayImplicitSize(const TQualifier&, TString* featureString = nullptr) const;
    void checkIoArrayConsistency(const TSourceLoc&, int requiredSize, const char* feature, TType&, const TString&);

    TIntermTyped* handleBinaryMath(const TSourceLoc&, const char* str, TOperator op, TIntermTyped* left, TIntermTyped* right);
    TIntermTyped* handleUnaryMath(const TSourceLoc&, const char* str, TOperator op, TIntermTyped* childNode);
    TIntermTyped* handleDotDereference(const TSourceLoc&, TIntermTyped* base, const TString& field);
    TIntermTyped* handleDotSwizzle(const TSourceLoc&, TIntermTyped* base, const TString& field);
    void blockMemberExtensionCheck(const TSourceLoc&, const TIntermTyped* base, int member, const TString& memberName);
    TFunction* handleFunctionDeclarator(const TSourceLoc&, TFunction& function, bool prototype);
    TIntermAggregate* handleFunctionDefinition(const TSourceLoc&, TFunction&);
    TIntermTyped* handleFunctionCall(const TSourceLoc&, TFunction*, TIntermNode*);
    TIntermTyped* handleBuiltInFunctionCall(TSourceLoc, TIntermNode* arguments, const TFunction& function);
    void computeBuiltinPrecisions(TIntermTyped&, const TFunction&);
    TIntermNode* handleReturnValue(const TSourceLoc&, TIntermTyped*);
    void checkLocation(const TSourceLoc&, TOperator);
    TIntermTyped* handleLengthMethod(const TSourceLoc&, TFunction*, TIntermNode*);
    void addInputArgumentConversions(const TFunction&, TIntermNode*&) const;
    TIntermTyped* addOutputArgumentConversions(const TFunction&, TIntermAggregate&) const;
    TIntermTyped* addAssign(const TSourceLoc&, TOperator op, TIntermTyped* left, TIntermTyped* right);
    void builtInOpCheck(const TSourceLoc&, const TFunction&, TIntermOperator&);
    void nonOpBuiltInCheck(const TSourceLoc&, const TFunction&, TIntermAggregate&);
    void userFunctionCallCheck(const TSourceLoc&, TIntermAggregate&);
    void samplerConstructorLocationCheck(const TSourceLoc&, const char* token, TIntermNode*);
    TFunction* handleConstructorCall(const TSourceLoc&, const TPublicType&);
    void handlePrecisionQualifier(const TSourceLoc&, TQualifier&, TPrecisionQualifier);
    void checkPrecisionQualifier(const TSourceLoc&, TPrecisionQualifier);
    void memorySemanticsCheck(const TSourceLoc&, const TFunction&, const TIntermOperator& callNode);

    void assignError(const TSourceLoc&, const char* op, TString left, TString right);
    void unaryOpError(const TSourceLoc&, const char* op, TString operand);
    void binaryOpError(const TSourceLoc&, const char* op, TString left, TString right);
    void variableCheck(TIntermTyped*& nodePtr);
    bool lValueErrorCheck(const TSourceLoc&, const char* op, TIntermTyped*) override;
    void rValueErrorCheck(const TSourceLoc&, const char* op, TIntermTyped*) override;
    void constantValueCheck(TIntermTyped* node, const char* token);
    void integerCheck(const TIntermTyped* node, const char* token);
    void globalCheck(const TSourceLoc&, const char* token);
    bool constructorError(const TSourceLoc&, TIntermNode*, TFunction&, TOperator, TType&);
    bool constructorTextureSamplerError(const TSourceLoc&, const TFunction&);
    void arraySizeCheck(const TSourceLoc&, TIntermTyped* expr, TArraySize&, const char *sizeType);
    bool arrayQualifierError(const TSourceLoc&, const TQualifier&);
    bool arrayError(const TSourceLoc&, const TType&);
    void arraySizeRequiredCheck(const TSourceLoc&, const TArraySizes&);
    void structArrayCheck(const TSourceLoc&, const TType& structure);
    void arraySizesCheck(const TSourceLoc&, const TQualifier&, TArraySizes*, const TIntermTyped* initializer, bool lastMember);
    void arrayOfArrayVersionCheck(const TSourceLoc&, const TArraySizes*);
    bool voidErrorCheck(const TSourceLoc&, const TString&, TBasicType);
    void boolCheck(const TSourceLoc&, const TIntermTyped*);
    void boolCheck(const TSourceLoc&, const TPublicType&);
    void samplerCheck(const TSourceLoc&, const TType&, const TString& identifier, TIntermTyped* initializer);
    void atomicUintCheck(const TSourceLoc&, const TType&, const TString& identifier);
    void accStructCheck(const TSourceLoc & loc, const TType & type, const TString & identifier);
    void transparentOpaqueCheck(const TSourceLoc&, const TType&, const TString& identifier);
    void memberQualifierCheck(glslang::TPublicType&);
    void globalQualifierFixCheck(const TSourceLoc&, TQualifier&);
    void globalQualifierTypeCheck(const TSourceLoc&, const TQualifier&, const TPublicType&);
    bool structQualifierErrorCheck(const TSourceLoc&, const TPublicType& pType);
    void mergeQualifiers(const TSourceLoc&, TQualifier& dst, const TQualifier& src, bool force);
    void setDefaultPrecision(const TSourceLoc&, TPublicType&, TPrecisionQualifier);
    int computeSamplerTypeIndex(TSampler&);
    TPrecisionQualifier getDefaultPrecision(TPublicType&);
    void precisionQualifierCheck(const TSourceLoc&, TBasicType, TQualifier&);
    void parameterTypeCheck(const TSourceLoc&, TStorageQualifier qualifier, const TType& type);
    bool containsFieldWithBasicType(const TType& type ,TBasicType basicType);
    TSymbol* redeclareBuiltinVariable(const TSourceLoc&, const TString&, const TQualifier&, const TShaderQualifiers&);
    void redeclareBuiltinBlock(const TSourceLoc&, TTypeList& typeList, const TString& blockName, const TString* instanceName, TArraySizes* arraySizes);
    void paramCheckFixStorage(const TSourceLoc&, const TStorageQualifier&, TType& type);
    void paramCheckFix(const TSourceLoc&, const TQualifier&, TType& type);
    void nestedBlockCheck(const TSourceLoc&);
    void nestedStructCheck(const TSourceLoc&);
    void arrayObjectCheck(const TSourceLoc&, const TType&, const char* op);
    void opaqueCheck(const TSourceLoc&, const TType&, const char* op);
    void referenceCheck(const TSourceLoc&, const TType&, const char* op);
    void storage16BitAssignmentCheck(const TSourceLoc&, const TType&, const char* op);
    void specializationCheck(const TSourceLoc&, const TType&, const char* op);
    void structTypeCheck(const TSourceLoc&, TPublicType&);
    void inductiveLoopCheck(const TSourceLoc&, TIntermNode* init, TIntermLoop* loop);
    void arrayLimitCheck(const TSourceLoc&, const TString&, int size);
    void limitCheck(const TSourceLoc&, int value, const char* limit, const char* feature);

    void inductiveLoopBodyCheck(TIntermNode*, int loopIndexId, TSymbolTable&);
    void constantIndexExpressionCheck(TIntermNode*);

    void setLayoutQualifier(const TSourceLoc&, TPublicType&, TString&);
    void setLayoutQualifier(const TSourceLoc&, TPublicType&, TString&, const TIntermTyped*);
    void mergeObjectLayoutQualifiers(TQualifier& dest, const TQualifier& src, bool inheritOnly);
    void layoutObjectCheck(const TSourceLoc&, const TSymbol&);
    void layoutMemberLocationArrayCheck(const TSourceLoc&, bool memberWithLocation, TArraySizes* arraySizes);
    void layoutTypeCheck(const TSourceLoc&, const TType&);
    void layoutQualifierCheck(const TSourceLoc&, const TQualifier&);
    void checkNoShaderLayouts(const TSourceLoc&, const TShaderQualifiers&);
    void fixOffset(const TSourceLoc&, TSymbol&);

    const TFunction* findFunction(const TSourceLoc& loc, const TFunction& call, bool& builtIn);
    const TFunction* findFunctionExact(const TSourceLoc& loc, const TFunction& call, bool& builtIn);
    const TFunction* findFunction120(const TSourceLoc& loc, const TFunction& call, bool& builtIn);
    const TFunction* findFunction400(const TSourceLoc& loc, const TFunction& call, bool& builtIn);
    const TFunction* findFunctionExplicitTypes(const TSourceLoc& loc, const TFunction& call, bool& builtIn);
    void declareTypeDefaults(const TSourceLoc&, const TPublicType&);
    TIntermNode* declareVariable(const TSourceLoc&, TString& identifier, const TPublicType&, TArraySizes* typeArray = 0, TIntermTyped* initializer = 0);
    TIntermTyped* addConstructor(const TSourceLoc&, TIntermNode*, const TType&);
    TIntermTyped* constructAggregate(TIntermNode*, const TType&, int, const TSourceLoc&);
    TIntermTyped* constructBuiltIn(const TType&, TOperator, TIntermTyped*, const TSourceLoc&, bool subset);
    void inheritMemoryQualifiers(const TQualifier& from, TQualifier& to);
    void declareBlock(const TSourceLoc&, TTypeList& typeList, const TString* instanceName = 0, TArraySizes* arraySizes = 0);
    void blockStageIoCheck(const TSourceLoc&, const TQualifier&);
    void blockQualifierCheck(const TSourceLoc&, const TQualifier&, bool instanceName);
    void fixBlockLocations(const TSourceLoc&, TQualifier&, TTypeList&, bool memberWithLocation, bool memberWithoutLocation);
    void fixXfbOffsets(TQualifier&, TTypeList&);
    void fixBlockUniformOffsets(TQualifier&, TTypeList&);
    void fixBlockUniformLayoutMatrix(TQualifier&, TTypeList*, TTypeList*);
    void fixBlockUniformLayoutPacking(TQualifier&, TTypeList*, TTypeList*);
    void addQualifierToExisting(const TSourceLoc&, TQualifier, const TString& identifier);
    void addQualifierToExisting(const TSourceLoc&, TQualifier, TIdentifierList&);
    void invariantCheck(const TSourceLoc&, const TQualifier&);
    void updateStandaloneQualifierDefaults(const TSourceLoc&, const TPublicType&);
    void wrapupSwitchSubsequence(TIntermAggregate* statements, TIntermNode* branchNode);
    TIntermNode* addSwitch(const TSourceLoc&, TIntermTyped* expression, TIntermAggregate* body);
    const TTypeList* recordStructCopy(TStructRecord&, const TType*, const TType*);

#ifndef GLSLANG_WEB
    TAttributeType attributeFromName(const TString& name) const;
    TAttributes* makeAttributes(const TString& identifier) const;
    TAttributes* makeAttributes(const TString& identifier, TIntermNode* node) const;
    TAttributes* mergeAttributes(TAttributes*, TAttributes*) const;

    // Determine selection control from attributes
    void handleSelectionAttributes(const TAttributes& attributes, TIntermNode*);
    void handleSwitchAttributes(const TAttributes& attributes, TIntermNode*);
    // Determine loop control from attributes
    void handleLoopAttributes(const TAttributes& attributes, TIntermNode*);
#endif

    void checkAndResizeMeshViewDim(const TSourceLoc&, TType&, bool isBlockMember);

protected:
    void nonInitConstCheck(const TSourceLoc&, TString& identifier, TType& type);
    void inheritGlobalDefaults(TQualifier& dst) const;
    TVariable* makeInternalVariable(const char* name, const TType&) const;
    TVariable* declareNonArray(const TSourceLoc&, const TString& identifier, const TType&);
    void declareArray(const TSourceLoc&, const TString& identifier, const TType&, TSymbol*&);
    void checkRuntimeSizable(const TSourceLoc&, const TIntermTyped&);
    bool isRuntimeLength(const TIntermTyped&) const;
    TIntermNode* executeInitializer(const TSourceLoc&, TIntermTyped* initializer, TVariable* variable);
    TIntermTyped* convertInitializerList(const TSourceLoc&, const TType&, TIntermTyped* initializer);
#ifndef GLSLANG_WEB
    void finish() override;
#endif

public:
    //
    // Generally, bison productions, the scanner, and the PP need read/write access to these; just give them direct access
    //

    // Current state of parsing
    bool inMain;                 // if inside a function, true if the function is main
    const TString* blockName;
    TQualifier currentBlockQualifier;
    TPrecisionQualifier defaultPrecision[EbtNumTypes];
    TBuiltInResource resources;
    TLimits& limits;

protected:
    TParseContext(TParseContext&);
    TParseContext& operator=(TParseContext&);

    static const int maxSamplerIndex = EsdNumDims * (EbtNumTypes * (2 * 2 * 2 * 2 * 2)); // see computeSamplerTypeIndex()
    TPrecisionQualifier defaultSamplerPrecision[maxSamplerIndex];
    TPrecisionManager precisionManager;
    TQualifier globalBufferDefaults;
    TQualifier globalUniformDefaults;
    TQualifier globalInputDefaults;
    TQualifier globalOutputDefaults;
    TString currentCaller;        // name of last function body entered (not valid when at global scope)
#ifndef GLSLANG_WEB
    int* atomicUintOffsets;       // to become an array of the right size to hold an offset per binding point
    bool anyIndexLimits;
    TIdSetType inductiveLoopIds;
    TVector<TIntermTyped*> needsIndexLimitationChecking;
    TStructRecord matrixFixRecord;
    TStructRecord packingFixRecord;

    //
    // Geometry shader input arrays:
    //  - array sizing is based on input primitive and/or explicit size
    //
    // Tessellation control output arrays:
    //  - array sizing is based on output layout(vertices=...) and/or explicit size
    //
    // Both:
    //  - array sizing is retroactive
    //  - built-in block redeclarations interact with this
    //
    // Design:
    //  - use a per-context "resize-list", a list of symbols whose array sizes
    //    can be fixed
    //
    //  - the resize-list starts empty at beginning of user-shader compilation, it does
    //    not have built-ins in it
    //
    //  - on built-in array use: copyUp() symbol and add it to the resize-list
    //
    //  - on user array declaration: add it to the resize-list
    //
    //  - on block redeclaration: copyUp() symbol and add it to the resize-list
    //     * note, that appropriately gives an error if redeclaring a block that
    //       was already used and hence already copied-up
    //
    //  - on seeing a layout declaration that sizes the array, fix everything in the
    //    resize-list, giving errors for mismatch
    //
    //  - on seeing an array size declaration, give errors on mismatch between it and previous
    //    array-sizing declarations
    //
    TVector<TSymbol*> ioArraySymbolResizeList;
#endif
};

} // end namespace glslang

#endif // _PARSER_HELPER_INCLUDED_
