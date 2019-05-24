//
// Copyright (C) 2016-2018 Google, Inc.
// Copyright (C) 2016 LunarG, Inc.
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
#ifndef HLSL_PARSE_INCLUDED_
#define HLSL_PARSE_INCLUDED_

#include "../glslang/MachineIndependent/parseVersions.h"
#include "../glslang/MachineIndependent/ParseHelper.h"
#include "../glslang/MachineIndependent/attribute.h"

#include <array>

namespace glslang {

class TFunctionDeclarator;

class HlslParseContext : public TParseContextBase {
public:
    HlslParseContext(TSymbolTable&, TIntermediate&, bool parsingBuiltins,
                     int version, EProfile, const SpvVersion& spvVersion, EShLanguage, TInfoSink&,
                     const TString sourceEntryPointName,
                     bool forwardCompatible = false, EShMessages messages = EShMsgDefault);
    virtual ~HlslParseContext();
    void initializeExtensionBehavior() override;

    void setLimits(const TBuiltInResource&) override;
    bool parseShaderStrings(TPpContext&, TInputScanner& input, bool versionWillBeError = false) override;
    virtual const char* getGlobalUniformBlockName() const override { return "$Global"; }
    virtual void setUniformBlockDefaults(TType& block) const override
    {
        block.getQualifier().layoutPacking = ElpStd140;
        block.getQualifier().layoutMatrix = ElmRowMajor;
    }

    void reservedPpErrorCheck(const TSourceLoc&, const char* /*name*/, const char* /*op*/) override { }
    bool lineContinuationCheck(const TSourceLoc&, bool /*endOfComment*/) override { return true; }
    bool lineDirectiveShouldSetNextLine() const override { return true; }
    bool builtInName(const TString&);

    void handlePragma(const TSourceLoc&, const TVector<TString>&) override;
    TIntermTyped* handleVariable(const TSourceLoc&, const TString* string);
    TIntermTyped* handleBracketDereference(const TSourceLoc&, TIntermTyped* base, TIntermTyped* index);
    TIntermTyped* handleBracketOperator(const TSourceLoc&, TIntermTyped* base, TIntermTyped* index);

    TIntermTyped* handleBinaryMath(const TSourceLoc&, const char* str, TOperator op, TIntermTyped* left, TIntermTyped* right);
    TIntermTyped* handleUnaryMath(const TSourceLoc&, const char* str, TOperator op, TIntermTyped* childNode);
    TIntermTyped* handleDotDereference(const TSourceLoc&, TIntermTyped* base, const TString& field);
    bool isBuiltInMethod(const TSourceLoc&, TIntermTyped* base, const TString& field);
    void assignToInterface(TVariable& variable);
    void handleFunctionDeclarator(const TSourceLoc&, TFunction& function, bool prototype);
    TIntermAggregate* handleFunctionDefinition(const TSourceLoc&, TFunction&, const TAttributes&, TIntermNode*& entryPointTree);
    TIntermNode* transformEntryPoint(const TSourceLoc&, TFunction&, const TAttributes&);
    void handleEntryPointAttributes(const TSourceLoc&, const TAttributes&);
    void transferTypeAttributes(const TSourceLoc&, const TAttributes&, TType&, bool allowEntry = false);
    void handleFunctionBody(const TSourceLoc&, TFunction&, TIntermNode* functionBody, TIntermNode*& node);
    void remapEntryPointIO(TFunction& function, TVariable*& returnValue, TVector<TVariable*>& inputs, TVector<TVariable*>& outputs);
    void remapNonEntryPointIO(TFunction& function);
    TIntermNode* handleReturnValue(const TSourceLoc&, TIntermTyped*);
    void handleFunctionArgument(TFunction*, TIntermTyped*& arguments, TIntermTyped* newArg);
    TIntermTyped* handleAssign(const TSourceLoc&, TOperator, TIntermTyped* left, TIntermTyped* right);
    TIntermTyped* handleAssignToMatrixSwizzle(const TSourceLoc&, TOperator, TIntermTyped* left, TIntermTyped* right);
    TIntermTyped* handleFunctionCall(const TSourceLoc&, TFunction*, TIntermTyped*);
    TIntermAggregate* assignClipCullDistance(const TSourceLoc&, TOperator, int semanticId, TIntermTyped* left, TIntermTyped* right);
    TIntermTyped* assignPosition(const TSourceLoc&, TOperator, TIntermTyped* left, TIntermTyped* right);
    void decomposeIntrinsic(const TSourceLoc&, TIntermTyped*& node, TIntermNode* arguments);
    void decomposeSampleMethods(const TSourceLoc&, TIntermTyped*& node, TIntermNode* arguments);
    void decomposeStructBufferMethods(const TSourceLoc&, TIntermTyped*& node, TIntermNode* arguments);
    void decomposeGeometryMethods(const TSourceLoc&, TIntermTyped*& node, TIntermNode* arguments);
    void pushFrontArguments(TIntermTyped* front, TIntermTyped*& arguments);
    void addInputArgumentConversions(const TFunction&, TIntermTyped*&);
    void expandArguments(const TSourceLoc&, const TFunction&, TIntermTyped*&);
    TIntermTyped* addOutputArgumentConversions(const TFunction&, TIntermOperator&);
    void builtInOpCheck(const TSourceLoc&, const TFunction&, TIntermOperator&);
    TFunction* makeConstructorCall(const TSourceLoc&, const TType&);
    void handleSemantic(TSourceLoc, TQualifier&, TBuiltInVariable, const TString& upperCase);
    void handlePackOffset(const TSourceLoc&, TQualifier&, const glslang::TString& location,
                          const glslang::TString* component);
    void handleRegister(const TSourceLoc&, TQualifier&, const glslang::TString* profile, const glslang::TString& desc,
                        int subComponent, const glslang::TString*);
    TIntermTyped* convertConditionalExpression(const TSourceLoc&, TIntermTyped*, bool mustBeScalar = true);
    TIntermAggregate* handleSamplerTextureCombine(const TSourceLoc& loc, TIntermTyped* argTex, TIntermTyped* argSampler);

    bool parseMatrixSwizzleSelector(const TSourceLoc&, const TString&, int cols, int rows, TSwizzleSelectors<TMatrixSelector>&);
    int getMatrixComponentsColumn(int rows, const TSwizzleSelectors<TMatrixSelector>&);
    void assignError(const TSourceLoc&, const char* op, TString left, TString right);
    void unaryOpError(const TSourceLoc&, const char* op, TString operand);
    void binaryOpError(const TSourceLoc&, const char* op, TString left, TString right);
    void variableCheck(TIntermTyped*& nodePtr);
    void constantValueCheck(TIntermTyped* node, const char* token);
    void integerCheck(const TIntermTyped* node, const char* token);
    void globalCheck(const TSourceLoc&, const char* token);
    bool constructorError(const TSourceLoc&, TIntermNode*, TFunction&, TOperator, TType&);
    void arraySizeCheck(const TSourceLoc&, TIntermTyped* expr, TArraySize&);
    void arraySizeRequiredCheck(const TSourceLoc&, const TArraySizes&);
    void structArrayCheck(const TSourceLoc&, const TType& structure);
    bool voidErrorCheck(const TSourceLoc&, const TString&, TBasicType);
    void globalQualifierFix(const TSourceLoc&, TQualifier&);
    bool structQualifierErrorCheck(const TSourceLoc&, const TPublicType& pType);
    void mergeQualifiers(TQualifier& dst, const TQualifier& src);
    int computeSamplerTypeIndex(TSampler&);
    TSymbol* redeclareBuiltinVariable(const TSourceLoc&, const TString&, const TQualifier&, const TShaderQualifiers&);
    void paramFix(TType& type);
    void specializationCheck(const TSourceLoc&, const TType&, const char* op);

    void setLayoutQualifier(const TSourceLoc&, TQualifier&, TString&);
    void setLayoutQualifier(const TSourceLoc&, TQualifier&, TString&, const TIntermTyped*);
    void setSpecConstantId(const TSourceLoc&, TQualifier&, int value);
    void mergeObjectLayoutQualifiers(TQualifier& dest, const TQualifier& src, bool inheritOnly);
    void checkNoShaderLayouts(const TSourceLoc&, const TShaderQualifiers&);

    const TFunction* findFunction(const TSourceLoc& loc, TFunction& call, bool& builtIn, int& thisDepth, TIntermTyped*& args);
    void addGenMulArgumentConversion(const TSourceLoc& loc, TFunction& call, TIntermTyped*& args);
    void declareTypedef(const TSourceLoc&, const TString& identifier, const TType&);
    void declareStruct(const TSourceLoc&, TString& structName, TType&);
    TSymbol* lookupUserType(const TString&, TType&);
    TIntermNode* declareVariable(const TSourceLoc&, const TString& identifier, TType&, TIntermTyped* initializer = 0);
    void lengthenList(const TSourceLoc&, TIntermSequence& list, int size, TIntermTyped* scalarInit);
    TIntermTyped* handleConstructor(const TSourceLoc&, TIntermTyped*, const TType&);
    TIntermTyped* addConstructor(const TSourceLoc&, TIntermTyped*, const TType&);
    TIntermTyped* convertArray(TIntermTyped*, const TType&);
    TIntermTyped* constructAggregate(TIntermNode*, const TType&, int, const TSourceLoc&);
    TIntermTyped* constructBuiltIn(const TType&, TOperator, TIntermTyped*, const TSourceLoc&, bool subset);
    void declareBlock(const TSourceLoc&, TType&, const TString* instanceName = 0);
    void declareStructBufferCounter(const TSourceLoc& loc, const TType& bufferType, const TString& name);
    void fixBlockLocations(const TSourceLoc&, TQualifier&, TTypeList&, bool memberWithLocation, bool memberWithoutLocation);
    void fixXfbOffsets(TQualifier&, TTypeList&);
    void fixBlockUniformOffsets(const TQualifier&, TTypeList&);
    void addQualifierToExisting(const TSourceLoc&, TQualifier, const TString& identifier);
    void addQualifierToExisting(const TSourceLoc&, TQualifier, TIdentifierList&);
    void updateStandaloneQualifierDefaults(const TSourceLoc&, const TPublicType&);
    void wrapupSwitchSubsequence(TIntermAggregate* statements, TIntermNode* branchNode);
    TIntermNode* addSwitch(const TSourceLoc&, TIntermTyped* expression, TIntermAggregate* body, const TAttributes&);

    void nestLooping()       { ++loopNestingLevel; }
    void unnestLooping()     { --loopNestingLevel; }
    void nestAnnotations()   { ++annotationNestingLevel; }
    void unnestAnnotations() { --annotationNestingLevel; }
    int getAnnotationNestingLevel() { return annotationNestingLevel; }
    void pushScope()         { symbolTable.push(); }
    void popScope()          { symbolTable.pop(0); }

    void pushThisScope(const TType&, const TVector<TFunctionDeclarator>&);
    void popThisScope()      { symbolTable.pop(0); }

    void pushImplicitThis(TVariable* thisParameter) { implicitThisStack.push_back(thisParameter); }
    void popImplicitThis() { implicitThisStack.pop_back(); }
    TVariable* getImplicitThis(int thisDepth) const { return implicitThisStack[implicitThisStack.size() - thisDepth]; }

    void pushNamespace(const TString& name);
    void popNamespace();
    void getFullNamespaceName(TString*&) const;
    void addScopeMangler(TString&);

    void pushSwitchSequence(TIntermSequence* sequence) { switchSequenceStack.push_back(sequence); }
    void popSwitchSequence() { switchSequenceStack.pop_back(); }

    virtual void growGlobalUniformBlock(const TSourceLoc&, TType&, const TString& memberName,
        TTypeList* typeList = nullptr) override;

    // Apply L-value conversions.  E.g, turning a write to a RWTexture into an ImageStore.
    TIntermTyped* handleLvalue(const TSourceLoc&, const char* op, TIntermTyped*& node);
    bool lValueErrorCheck(const TSourceLoc&, const char* op, TIntermTyped*) override;

    TLayoutFormat getLayoutFromTxType(const TSourceLoc&, const TType&);

    bool handleOutputGeometry(const TSourceLoc&, const TLayoutGeometry& geometry);
    bool handleInputGeometry(const TSourceLoc&, const TLayoutGeometry& geometry);

    // Determine selection control from attributes
    void handleSelectionAttributes(const TSourceLoc& loc, TIntermSelection*, const TAttributes& attributes);
    void handleSwitchAttributes(const TSourceLoc& loc, TIntermSwitch*, const TAttributes& attributes);

    // Determine loop control from attributes
    void handleLoopAttributes(const TSourceLoc& loc, TIntermLoop*, const TAttributes& attributes);

    // Share struct buffer deep types
    void shareStructBufferType(TType&);

    // Set texture return type of the given sampler.  Returns success (not all types are valid).
    bool setTextureReturnType(TSampler& sampler, const TType& retType, const TSourceLoc& loc);

    // Obtain the sampler return type of the given sampler in retType.
    void getTextureReturnType(const TSampler& sampler, TType& retType) const;

    TAttributeType attributeFromName(const TString& nameSpace, const TString& name) const;

protected:
    struct TFlattenData {
        TFlattenData() : nextBinding(TQualifier::layoutBindingEnd),
                         nextLocation(TQualifier::layoutLocationEnd) { }
        TFlattenData(int nb, int nl) : nextBinding(nb), nextLocation(nl) { }

        TVector<TVariable*> members;     // individual flattened variables
        TVector<int> offsets;            // offset to next tree level
        unsigned int nextBinding;        // next binding to use.
        unsigned int nextLocation;       // next location to use
    };

    void fixConstInit(const TSourceLoc&, const TString& identifier, TType& type, TIntermTyped*& initializer);
    void inheritGlobalDefaults(TQualifier& dst) const;
    TVariable* makeInternalVariable(const char* name, const TType&) const;
    TVariable* makeInternalVariable(const TString& name, const TType& type) const {
        return makeInternalVariable(name.c_str(), type);
    }
    TIntermSymbol* makeInternalVariableNode(const TSourceLoc&, const char* name, const TType&) const;
    TVariable* declareNonArray(const TSourceLoc&, const TString& identifier, const TType&, bool track);
    void declareArray(const TSourceLoc&, const TString& identifier, const TType&, TSymbol*&, bool track);
    TIntermNode* executeInitializer(const TSourceLoc&, TIntermTyped* initializer, TVariable* variable);
    TIntermTyped* convertInitializerList(const TSourceLoc&, const TType&, TIntermTyped* initializer, TIntermTyped* scalarInit);
    bool isScalarConstructor(const TIntermNode*);
    TOperator mapAtomicOp(const TSourceLoc& loc, TOperator op, bool isImage);

    // Return true if this node requires L-value conversion (e.g, to an imageStore).
    bool shouldConvertLValue(const TIntermNode*) const;

    // Array and struct flattening
    TIntermTyped* flattenAccess(TIntermTyped* base, int member);
    TIntermTyped* flattenAccess(int uniqueId, int member, TStorageQualifier outerStorage, const TType&, int subset = -1);
    int findSubtreeOffset(const TIntermNode&) const;
    int findSubtreeOffset(const TType&, int subset, const TVector<int>& offsets) const;
    bool shouldFlatten(const TType&, TStorageQualifier, bool topLevel) const;
    bool wasFlattened(const TIntermTyped* node) const;
    bool wasFlattened(int id) const { return flattenMap.find(id) != flattenMap.end(); }
    int  addFlattenedMember(const TVariable&, const TType&, TFlattenData&, const TString& name, bool linkage,
                            const TQualifier& outerQualifier, const TArraySizes* builtInArraySizes);

    // Structure splitting (splits interstage built-in types into its own struct)
    void split(const TVariable&);
    void splitBuiltIn(const TString& baseName, const TType& memberType, const TArraySizes*, const TQualifier&);
    const TType& split(const TType& type, const TString& name, const TQualifier&);
    bool wasSplit(const TIntermTyped* node) const;
    bool wasSplit(int id) const { return splitNonIoVars.find(id) != splitNonIoVars.end(); }
    TVariable* getSplitNonIoVar(int id) const;
    void addPatchConstantInvocation();
    void fixTextureShadowModes();
    void finalizeAppendMethods();
    TIntermTyped* makeIntegerIndex(TIntermTyped*);

    void fixBuiltInIoType(TType&);

    void flatten(const TVariable& variable, bool linkage);
    int flatten(const TVariable& variable, const TType&, TFlattenData&, TString name, bool linkage,
                const TQualifier& outerQualifier, const TArraySizes* builtInArraySizes);
    int flattenStruct(const TVariable& variable, const TType&, TFlattenData&, TString name, bool linkage,
                      const TQualifier& outerQualifier, const TArraySizes* builtInArraySizes);
    int flattenArray(const TVariable& variable, const TType&, TFlattenData&, TString name, bool linkage,
                     const TQualifier& outerQualifier);

    bool hasUniform(const TQualifier& qualifier) const;
    void clearUniform(TQualifier& qualifier);
    bool isInputBuiltIn(const TQualifier& qualifier) const;
    bool hasInput(const TQualifier& qualifier) const;
    void correctOutput(TQualifier& qualifier);
    bool isOutputBuiltIn(const TQualifier& qualifier) const;
    bool hasOutput(const TQualifier& qualifier) const;
    void correctInput(TQualifier& qualifier);
    void correctUniform(TQualifier& qualifier);
    void clearUniformInputOutput(TQualifier& qualifier);

    // Test method names
    bool isStructBufferMethod(const TString& name) const;
    void counterBufferType(const TSourceLoc& loc, TType& type);

    // Return standard sample position array
    TIntermConstantUnion* getSamplePosArray(int count);

    TType* getStructBufferContentType(const TType& type) const;
    bool isStructBufferType(const TType& type) const { return getStructBufferContentType(type) != nullptr; }
    TIntermTyped* indexStructBufferContent(const TSourceLoc& loc, TIntermTyped* buffer) const;
    TIntermTyped* getStructBufferCounter(const TSourceLoc& loc, TIntermTyped* buffer);
    TString getStructBuffCounterName(const TString&) const;
    void addStructBuffArguments(const TSourceLoc& loc, TIntermAggregate*&);
    void addStructBufferHiddenCounterParam(const TSourceLoc& loc, TParameter&, TIntermAggregate*&);

    // Return true if this type is a reference.  This is not currently a type method in case that's
    // a language specific answer.
    bool isReference(const TType& type) const { return isStructBufferType(type); }

    // Return true if this a buffer type that has an associated counter buffer.
    bool hasStructBuffCounter(const TType&) const;

    // Finalization step: remove unused buffer blocks from linkage (we don't know until the
    // shader is entirely compiled)
    void removeUnusedStructBufferCounters();
 
    static bool isClipOrCullDistance(TBuiltInVariable);
    static bool isClipOrCullDistance(const TQualifier& qual) { return isClipOrCullDistance(qual.builtIn); }
    static bool isClipOrCullDistance(const TType& type) { return isClipOrCullDistance(type.getQualifier()); }

    // Find the patch constant function (issues error, returns nullptr if not found)
    const TFunction* findPatchConstantFunction(const TSourceLoc& loc);

    // Pass through to base class after remembering built-in mappings.
    using TParseContextBase::trackLinkage;
    void trackLinkage(TSymbol& variable) override;

    void finish() override; // post-processing

    // Linkage symbol helpers
    TIntermSymbol* findTessLinkageSymbol(TBuiltInVariable biType) const;

    // Current state of parsing
    int annotationNestingLevel;  // 0 if outside all annotations

    HlslParseContext(HlslParseContext&);
    HlslParseContext& operator=(HlslParseContext&);

    static const int maxSamplerIndex = EsdNumDims * (EbtNumTypes * (2 * 2 * 2)); // see computeSamplerTypeIndex()
    TQualifier globalBufferDefaults;
    TQualifier globalUniformDefaults;
    TQualifier globalInputDefaults;
    TQualifier globalOutputDefaults;
    TString currentCaller;        // name of last function body entered (not valid when at global scope)
    TIdSetType inductiveLoopIds;
    TVector<TIntermTyped*> needsIndexLimitationChecking;

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

    TMap<int, TFlattenData> flattenMap;

    // IO-type map. Maps a pure symbol-table form of a structure-member list into
    // each of the (up to) three kinds of IO, as each as different allowed decorations,
    // but HLSL allows mixing all in the same structure.
    struct tIoKinds {
        TTypeList* input;
        TTypeList* output;
        TTypeList* uniform;
    };
    TMap<const TTypeList*, tIoKinds> ioTypeMap;

    // Structure splitting data:
    TMap<int, TVariable*> splitNonIoVars;  // variables with the built-in interstage IO removed, indexed by unique ID.

    // Structuredbuffer shared types.  Typically there are only a few.
    TVector<TType*> structBufferTypes;

    // This tracks texture sample user structure return types.  Only a limited number are supported, as
    // may fit in TSampler::structReturnIndex.
    TVector<TTypeList*> textureReturnStruct;
    
    TMap<TString, bool> structBufferCounter;  // true if counter buffer is in use

    // The built-in interstage IO map considers e.g, EvqPosition on input and output separately, so that we
    // can build the linkage correctly if position appears on both sides.  Otherwise, multiple positions
    // are considered identical.
    struct tInterstageIoData {
        tInterstageIoData(TBuiltInVariable bi, TStorageQualifier q) :
            builtIn(bi), storage(q) { }

        TBuiltInVariable  builtIn;
        TStorageQualifier storage;

        // ordering for maps
        bool operator<(const tInterstageIoData d) const {
            return (builtIn != d.builtIn) ? (builtIn < d.builtIn) : (storage < d.storage);
        }
    };

    TMap<tInterstageIoData, TVariable*> splitBuiltIns; // split built-ins, indexed by built-in type.
    TVariable* inputPatch; // input patch is special for PCF: it's the only non-builtin PCF input,
                           // and is handled as a pseudo-builtin.

    unsigned int nextInLocation;
    unsigned int nextOutLocation;

    TFunction* entryPointFunction;
    TIntermNode* entryPointFunctionBody;

    TString patchConstantFunctionName; // hull shader patch constant function name, from function level attribute.
    TMap<TBuiltInVariable, TSymbol*> builtInTessLinkageSymbols; // used for tessellation, finding declared built-ins

    TVector<TString> currentTypePrefix;      // current scoping prefix for nested structures
    TVector<TVariable*> implicitThisStack;   // currently active 'this' variables for nested structures

    TVariable* gsStreamOutput;               // geometry shader stream outputs, for emit (Append method)

    TVariable* clipDistanceOutput;           // synthesized clip distance out variable (shader might have >1)
    TVariable* cullDistanceOutput;           // synthesized cull distance out variable (shader might have >1)
    TVariable* clipDistanceInput;            // synthesized clip distance in variable (shader might have >1)
    TVariable* cullDistanceInput;            // synthesized cull distance in variable (shader might have >1)

    static const int maxClipCullRegs = 2;
    std::array<int, maxClipCullRegs> clipSemanticNSizeIn;  // vector, indexed by clip semantic ID
    std::array<int, maxClipCullRegs> cullSemanticNSizeIn;  // vector, indexed by cull semantic ID
    std::array<int, maxClipCullRegs> clipSemanticNSizeOut; // vector, indexed by clip semantic ID
    std::array<int, maxClipCullRegs> cullSemanticNSizeOut; // vector, indexed by cull semantic ID

    // This tracks the first (mip level) argument to the .mips[][] operator.  Since this can be nested as
    // in tx.mips[tx.mips[0][1].x][2], we need a stack.  We also track the TSourceLoc for error reporting 
    // purposes.
    struct tMipsOperatorData {
        tMipsOperatorData(TSourceLoc l, TIntermTyped* m) : loc(l), mipLevel(m) { }
        TSourceLoc loc;
        TIntermTyped* mipLevel;
    };

    TVector<tMipsOperatorData> mipsOperatorMipArg;

    // The geometry output stream is not copied out from the entry point as a typical output variable
    // is.  It's written via EmitVertex (hlsl=Append), which may happen in arbitrary control flow.
    // For this we need the real output symbol.  Since it may not be known at the time and Append()
    // method is parsed, the sequence will be patched during finalization.
    struct tGsAppendData {
        TIntermAggregate* node;
        TSourceLoc loc;
    };

    TVector<tGsAppendData> gsAppends;

    // A texture object may be used with shadow and non-shadow samplers, but both may not be
    // alive post-DCE in the same shader.  We do not know at compilation time which are alive: that's
    // only known post-DCE.  If a texture is used both ways, we create two textures, and
    // leave the elimiation of one to the optimizer.  This maps the shader variant to
    // the shadow variant.
    //
    // This can be removed if and when the texture shadow code in
    // HlslParseContext::handleSamplerTextureCombine is removed.
    struct tShadowTextureSymbols {
        tShadowTextureSymbols() { symId.fill(-1); }

        void set(bool shadow, int id) { symId[int(shadow)] = id; }
        int get(bool shadow) const { return symId[int(shadow)]; }

        // True if this texture has been seen with both shadow and non-shadow modes
        bool overloaded() const { return symId[0] != -1 && symId[1] != -1; }
        bool isShadowId(int id) const { return symId[1] == id; }

    private:
        std::array<int, 2> symId;
    };

    TMap<int, tShadowTextureSymbols*> textureShadowVariant;
};

// This is the prefix we use for built-in methods to avoid namespace collisions with
// global scope user functions.
// TODO: this would be better as a nonparseable character, but that would
// require changing the scanner.
#define BUILTIN_PREFIX "__BI_"

} // end namespace glslang

#endif // HLSL_PARSE_INCLUDED_
