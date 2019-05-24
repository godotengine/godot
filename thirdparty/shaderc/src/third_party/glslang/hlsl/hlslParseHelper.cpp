//
// Copyright (C) 2017-2018 Google, Inc.
// Copyright (C) 2017 LunarG, Inc.
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

#include "hlslParseHelper.h"
#include "hlslScanContext.h"
#include "hlslGrammar.h"
#include "hlslAttributes.h"

#include "../glslang/Include/Common.h"
#include "../glslang/MachineIndependent/Scan.h"
#include "../glslang/MachineIndependent/preprocessor/PpContext.h"

#include "../glslang/OSDependent/osinclude.h"

#include <algorithm>
#include <functional>
#include <cctype>
#include <array>
#include <set>

namespace glslang {

HlslParseContext::HlslParseContext(TSymbolTable& symbolTable, TIntermediate& interm, bool parsingBuiltins,
                                   int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language,
                                   TInfoSink& infoSink,
                                   const TString sourceEntryPointName,
                                   bool forwardCompatible, EShMessages messages) :
    TParseContextBase(symbolTable, interm, parsingBuiltins, version, profile, spvVersion, language, infoSink,
                      forwardCompatible, messages, &sourceEntryPointName),
    annotationNestingLevel(0),
    inputPatch(nullptr),
    nextInLocation(0), nextOutLocation(0),
    entryPointFunction(nullptr),
    entryPointFunctionBody(nullptr),
    gsStreamOutput(nullptr),
    clipDistanceOutput(nullptr),
    cullDistanceOutput(nullptr),
    clipDistanceInput(nullptr),
    cullDistanceInput(nullptr)
{
    globalUniformDefaults.clear();
    globalUniformDefaults.layoutMatrix = ElmRowMajor;
    globalUniformDefaults.layoutPacking = ElpStd140;

    globalBufferDefaults.clear();
    globalBufferDefaults.layoutMatrix = ElmRowMajor;
    globalBufferDefaults.layoutPacking = ElpStd430;

    globalInputDefaults.clear();
    globalOutputDefaults.clear();

    clipSemanticNSizeIn.fill(0);
    cullSemanticNSizeIn.fill(0);
    clipSemanticNSizeOut.fill(0);
    cullSemanticNSizeOut.fill(0);

    // "Shaders in the transform
    // feedback capturing mode have an initial global default of
    //     layout(xfb_buffer = 0) out;"
    if (language == EShLangVertex ||
        language == EShLangTessControl ||
        language == EShLangTessEvaluation ||
        language == EShLangGeometry)
        globalOutputDefaults.layoutXfbBuffer = 0;

    if (language == EShLangGeometry)
        globalOutputDefaults.layoutStream = 0;
}

HlslParseContext::~HlslParseContext()
{
}

void HlslParseContext::initializeExtensionBehavior()
{
    TParseContextBase::initializeExtensionBehavior();

    // HLSL allows #line by default.
    extensionBehavior[E_GL_GOOGLE_cpp_style_line_directive] = EBhEnable;
}

void HlslParseContext::setLimits(const TBuiltInResource& r)
{
    resources = r;
    intermediate.setLimits(resources);
}

//
// Parse an array of strings using the parser in HlslRules.
//
// Returns true for successful acceptance of the shader, false if any errors.
//
bool HlslParseContext::parseShaderStrings(TPpContext& ppContext, TInputScanner& input, bool versionWillBeError)
{
    currentScanner = &input;
    ppContext.setInput(input, versionWillBeError);

    HlslScanContext scanContext(*this, ppContext);
    HlslGrammar grammar(scanContext, *this);
    if (!grammar.parse()) {
        // Print a message formated such that if you click on the message it will take you right to
        // the line through most UIs.
        const glslang::TSourceLoc& sourceLoc = input.getSourceLoc();
        infoSink.info << sourceLoc.getFilenameStr() << "(" << sourceLoc.line << "): error at column " << sourceLoc.column
                      << ", HLSL parsing failed.\n";
        ++numErrors;
        return false;
    }

    finish();

    return numErrors == 0;
}

//
// Return true if this l-value node should be converted in some manner.
// For instance: turning a load aggregate into a store in an l-value.
//
bool HlslParseContext::shouldConvertLValue(const TIntermNode* node) const
{
    if (node == nullptr || node->getAsTyped() == nullptr)
        return false;

    const TIntermAggregate* lhsAsAggregate = node->getAsAggregate();
    const TIntermBinary* lhsAsBinary = node->getAsBinaryNode();

    // If it's a swizzled/indexed aggregate, look at the left node instead.
    if (lhsAsBinary != nullptr &&
        (lhsAsBinary->getOp() == EOpVectorSwizzle || lhsAsBinary->getOp() == EOpIndexDirect))
        lhsAsAggregate = lhsAsBinary->getLeft()->getAsAggregate();
    if (lhsAsAggregate != nullptr && lhsAsAggregate->getOp() == EOpImageLoad)
        return true;

    return false;
}

void HlslParseContext::growGlobalUniformBlock(const TSourceLoc& loc, TType& memberType, const TString& memberName,
                                              TTypeList* newTypeList)
{
    newTypeList = nullptr;
    correctUniform(memberType.getQualifier());
    if (memberType.isStruct()) {
        auto it = ioTypeMap.find(memberType.getStruct());
        if (it != ioTypeMap.end() && it->second.uniform)
            newTypeList = it->second.uniform;
    }
    TParseContextBase::growGlobalUniformBlock(loc, memberType, memberName, newTypeList);
}

//
// Return a TLayoutFormat corresponding to the given texture type.
//
TLayoutFormat HlslParseContext::getLayoutFromTxType(const TSourceLoc& loc, const TType& txType)
{
    if (txType.isStruct()) {
        // TODO: implement.
        error(loc, "unimplemented: structure type in image or buffer", "", "");
        return ElfNone;
    }

    const int components = txType.getVectorSize();
    const TBasicType txBasicType = txType.getBasicType();

    const auto selectFormat = [this,&components](TLayoutFormat v1, TLayoutFormat v2, TLayoutFormat v4) -> TLayoutFormat {
        if (intermediate.getNoStorageFormat())
            return ElfNone;

        return components == 1 ? v1 :
               components == 2 ? v2 : v4;
    };

    switch (txBasicType) {
    case EbtFloat: return selectFormat(ElfR32f,  ElfRg32f,  ElfRgba32f);
    case EbtInt:   return selectFormat(ElfR32i,  ElfRg32i,  ElfRgba32i);
    case EbtUint:  return selectFormat(ElfR32ui, ElfRg32ui, ElfRgba32ui);
    default:
        error(loc, "unknown basic type in image format", "", "");
        return ElfNone;
    }
}

//
// Both test and if necessary, spit out an error, to see if the node is really
// an l-value that can be operated on this way.
//
// Returns true if there was an error.
//
bool HlslParseContext::lValueErrorCheck(const TSourceLoc& loc, const char* op, TIntermTyped* node)
{
    if (shouldConvertLValue(node)) {
        // if we're writing to a texture, it must be an RW form.

        TIntermAggregate* lhsAsAggregate = node->getAsAggregate();
        TIntermTyped* object = lhsAsAggregate->getSequence()[0]->getAsTyped();

        if (!object->getType().getSampler().isImage()) {
            error(loc, "operator[] on a non-RW texture must be an r-value", "", "");
            return true;
        }
    }

    // We tolerate samplers as l-values, even though they are nominally
    // illegal, because we expect a later optimization to eliminate them.
    if (node->getType().getBasicType() == EbtSampler) {
        intermediate.setNeedsLegalization();
        return false;
    }

    // Let the base class check errors
    return TParseContextBase::lValueErrorCheck(loc, op, node);
}

//
// This function handles l-value conversions and verifications.  It uses, but is not synonymous
// with lValueErrorCheck.  That function accepts an l-value directly, while this one must be
// given the surrounding tree - e.g, with an assignment, so we can convert the assign into a
// series of other image operations.
//
// Most things are passed through unmodified, except for error checking.
//
TIntermTyped* HlslParseContext::handleLvalue(const TSourceLoc& loc, const char* op, TIntermTyped*& node)
{
    if (node == nullptr)
        return nullptr;

    TIntermBinary* nodeAsBinary = node->getAsBinaryNode();
    TIntermUnary* nodeAsUnary = node->getAsUnaryNode();
    TIntermAggregate* sequence = nullptr;

    TIntermTyped* lhs = nodeAsUnary  ? nodeAsUnary->getOperand() :
                        nodeAsBinary ? nodeAsBinary->getLeft() :
                        nullptr;

    // Early bail out if there is no conversion to apply
    if (!shouldConvertLValue(lhs)) {
        if (lhs != nullptr)
            if (lValueErrorCheck(loc, op, lhs))
                return nullptr;
        return node;
    }

    // *** If we get here, we're going to apply some conversion to an l-value.

    // Helper to create a load.
    const auto makeLoad = [&](TIntermSymbol* rhsTmp, TIntermTyped* object, TIntermTyped* coord, const TType& derefType) {
        TIntermAggregate* loadOp = new TIntermAggregate(EOpImageLoad);
        loadOp->setLoc(loc);
        loadOp->getSequence().push_back(object);
        loadOp->getSequence().push_back(intermediate.addSymbol(*coord->getAsSymbolNode()));
        loadOp->setType(derefType);

        sequence = intermediate.growAggregate(sequence,
                                              intermediate.addAssign(EOpAssign, rhsTmp, loadOp, loc),
                                              loc);
    };

    // Helper to create a store.
    const auto makeStore = [&](TIntermTyped* object, TIntermTyped* coord, TIntermSymbol* rhsTmp) {
        TIntermAggregate* storeOp = new TIntermAggregate(EOpImageStore);
        storeOp->getSequence().push_back(object);
        storeOp->getSequence().push_back(coord);
        storeOp->getSequence().push_back(intermediate.addSymbol(*rhsTmp));
        storeOp->setLoc(loc);
        storeOp->setType(TType(EbtVoid));

        sequence = intermediate.growAggregate(sequence, storeOp);
    };

    // Helper to create an assign.
    const auto makeBinary = [&](TOperator op, TIntermTyped* lhs, TIntermTyped* rhs) {
        sequence = intermediate.growAggregate(sequence,
                                              intermediate.addBinaryNode(op, lhs, rhs, loc, lhs->getType()),
                                              loc);
    };

    // Helper to complete sequence by adding trailing variable, so we evaluate to the right value.
    const auto finishSequence = [&](TIntermSymbol* rhsTmp, const TType& derefType) -> TIntermAggregate* {
        // Add a trailing use of the temp, so the sequence returns the proper value.
        sequence = intermediate.growAggregate(sequence, intermediate.addSymbol(*rhsTmp));
        sequence->setOperator(EOpSequence);
        sequence->setLoc(loc);
        sequence->setType(derefType);

        return sequence;
    };

    // Helper to add unary op
    const auto makeUnary = [&](TOperator op, TIntermSymbol* rhsTmp) {
        sequence = intermediate.growAggregate(sequence,
                                              intermediate.addUnaryNode(op, intermediate.addSymbol(*rhsTmp), loc,
                                                                        rhsTmp->getType()),
                                              loc);
    };

    // Return true if swizzle or index writes all components of the given variable.
    const auto writesAllComponents = [&](TIntermSymbol* var, TIntermBinary* swizzle) -> bool {
        if (swizzle == nullptr)  // not a swizzle or index
            return true;

        // Track which components are being set.
        std::array<bool, 4> compIsSet;
        compIsSet.fill(false);

        const TIntermConstantUnion* asConst     = swizzle->getRight()->getAsConstantUnion();
        const TIntermAggregate*     asAggregate = swizzle->getRight()->getAsAggregate();

        // This could be either a direct index, or a swizzle.
        if (asConst) {
            compIsSet[asConst->getConstArray()[0].getIConst()] = true;
        } else if (asAggregate) {
            const TIntermSequence& seq = asAggregate->getSequence();
            for (int comp=0; comp<int(seq.size()); ++comp)
                compIsSet[seq[comp]->getAsConstantUnion()->getConstArray()[0].getIConst()] = true;
        } else {
            assert(0);
        }

        // Return true if all components are being set by the index or swizzle
        return std::all_of(compIsSet.begin(), compIsSet.begin() + var->getType().getVectorSize(),
                           [](bool isSet) { return isSet; } );
    };

    // Create swizzle matching input swizzle
    const auto addSwizzle = [&](TIntermSymbol* var, TIntermBinary* swizzle) -> TIntermTyped* {
        if (swizzle)
            return intermediate.addBinaryNode(swizzle->getOp(), var, swizzle->getRight(), loc, swizzle->getType());
        else
            return var;
    };

    TIntermBinary*    lhsAsBinary    = lhs->getAsBinaryNode();
    TIntermAggregate* lhsAsAggregate = lhs->getAsAggregate();
    bool lhsIsSwizzle = false;

    // If it's a swizzled L-value, remember the swizzle, and use the LHS.
    if (lhsAsBinary != nullptr && (lhsAsBinary->getOp() == EOpVectorSwizzle || lhsAsBinary->getOp() == EOpIndexDirect)) {
        lhsAsAggregate = lhsAsBinary->getLeft()->getAsAggregate();
        lhsIsSwizzle = true;
    }

    TIntermTyped* object = lhsAsAggregate->getSequence()[0]->getAsTyped();
    TIntermTyped* coord  = lhsAsAggregate->getSequence()[1]->getAsTyped();

    const TSampler& texSampler = object->getType().getSampler();

    TType objDerefType;
    getTextureReturnType(texSampler, objDerefType);

    if (nodeAsBinary) {
        TIntermTyped* rhs = nodeAsBinary->getRight();
        const TOperator assignOp = nodeAsBinary->getOp();

        bool isModifyOp = false;

        switch (assignOp) {
        case EOpAddAssign:
        case EOpSubAssign:
        case EOpMulAssign:
        case EOpVectorTimesMatrixAssign:
        case EOpVectorTimesScalarAssign:
        case EOpMatrixTimesScalarAssign:
        case EOpMatrixTimesMatrixAssign:
        case EOpDivAssign:
        case EOpModAssign:
        case EOpAndAssign:
        case EOpInclusiveOrAssign:
        case EOpExclusiveOrAssign:
        case EOpLeftShiftAssign:
        case EOpRightShiftAssign:
            isModifyOp = true;
            // fall through...
        case EOpAssign:
            {
                // Since this is an lvalue, we'll convert an image load to a sequence like this
                // (to still provide the value):
                //   OpSequence
                //      OpImageStore(object, lhs, rhs)
                //      rhs
                // But if it's not a simple symbol RHS (say, a fn call), we don't want to duplicate the RHS,
                // so we'll convert instead to this:
                //   OpSequence
                //      rhsTmp = rhs
                //      OpImageStore(object, coord, rhsTmp)
                //      rhsTmp
                // If this is a read-modify-write op, like +=, we issue:
                //   OpSequence
                //      coordtmp = load's param1
                //      rhsTmp = OpImageLoad(object, coordTmp)
                //      rhsTmp op= rhs
                //      OpImageStore(object, coordTmp, rhsTmp)
                //      rhsTmp
                //
                // If the lvalue is swizzled, we apply that when writing the temp variable, like so:
                //    ...
                //    rhsTmp.some_swizzle = ...
                // For partial writes, an error is generated.

                TIntermSymbol* rhsTmp = rhs->getAsSymbolNode();
                TIntermTyped* coordTmp = coord;

                if (rhsTmp == nullptr || isModifyOp || lhsIsSwizzle) {
                    rhsTmp = makeInternalVariableNode(loc, "storeTemp", objDerefType);

                    // Partial updates not yet supported
                    if (!writesAllComponents(rhsTmp, lhsAsBinary)) {
                        error(loc, "unimplemented: partial image updates", "", "");
                    }

                    // Assign storeTemp = rhs
                    if (isModifyOp) {
                        // We have to make a temp var for the coordinate, to avoid evaluating it twice.
                        coordTmp = makeInternalVariableNode(loc, "coordTemp", coord->getType());
                        makeBinary(EOpAssign, coordTmp, coord); // coordtmp = load[param1]
                        makeLoad(rhsTmp, object, coordTmp, objDerefType); // rhsTmp = OpImageLoad(object, coordTmp)
                    }

                    // rhsTmp op= rhs.
                    makeBinary(assignOp, addSwizzle(intermediate.addSymbol(*rhsTmp), lhsAsBinary), rhs);
                }

                makeStore(object, coordTmp, rhsTmp);         // add a store
                return finishSequence(rhsTmp, objDerefType); // return rhsTmp from sequence
            }

        default:
            break;
        }
    }

    if (nodeAsUnary) {
        const TOperator assignOp = nodeAsUnary->getOp();

        switch (assignOp) {
        case EOpPreIncrement:
        case EOpPreDecrement:
            {
                // We turn this into:
                //   OpSequence
                //      coordtmp = load's param1
                //      rhsTmp = OpImageLoad(object, coordTmp)
                //      rhsTmp op
                //      OpImageStore(object, coordTmp, rhsTmp)
                //      rhsTmp

                TIntermSymbol* rhsTmp = makeInternalVariableNode(loc, "storeTemp", objDerefType);
                TIntermTyped* coordTmp = makeInternalVariableNode(loc, "coordTemp", coord->getType());

                makeBinary(EOpAssign, coordTmp, coord);           // coordtmp = load[param1]
                makeLoad(rhsTmp, object, coordTmp, objDerefType); // rhsTmp = OpImageLoad(object, coordTmp)
                makeUnary(assignOp, rhsTmp);                      // op rhsTmp
                makeStore(object, coordTmp, rhsTmp);              // OpImageStore(object, coordTmp, rhsTmp)
                return finishSequence(rhsTmp, objDerefType);      // return rhsTmp from sequence
            }

        case EOpPostIncrement:
        case EOpPostDecrement:
            {
                // We turn this into:
                //   OpSequence
                //      coordtmp = load's param1
                //      rhsTmp1 = OpImageLoad(object, coordTmp)
                //      rhsTmp2 = rhsTmp1
                //      rhsTmp2 op
                //      OpImageStore(object, coordTmp, rhsTmp2)
                //      rhsTmp1 (pre-op value)
                TIntermSymbol* rhsTmp1 = makeInternalVariableNode(loc, "storeTempPre",  objDerefType);
                TIntermSymbol* rhsTmp2 = makeInternalVariableNode(loc, "storeTempPost", objDerefType);
                TIntermTyped* coordTmp = makeInternalVariableNode(loc, "coordTemp", coord->getType());

                makeBinary(EOpAssign, coordTmp, coord);            // coordtmp = load[param1]
                makeLoad(rhsTmp1, object, coordTmp, objDerefType); // rhsTmp1 = OpImageLoad(object, coordTmp)
                makeBinary(EOpAssign, rhsTmp2, rhsTmp1);           // rhsTmp2 = rhsTmp1
                makeUnary(assignOp, rhsTmp2);                      // rhsTmp op
                makeStore(object, coordTmp, rhsTmp2);              // OpImageStore(object, coordTmp, rhsTmp2)
                return finishSequence(rhsTmp1, objDerefType);      // return rhsTmp from sequence
            }

        default:
            break;
        }
    }

    if (lhs)
        if (lValueErrorCheck(loc, op, lhs))
            return nullptr;

    return node;
}

void HlslParseContext::handlePragma(const TSourceLoc& loc, const TVector<TString>& tokens)
{
    if (pragmaCallback)
        pragmaCallback(loc.line, tokens);

    if (tokens.size() == 0)
        return;

    // These pragmas are case insensitive in HLSL, so we'll compare in lower case.
    TVector<TString> lowerTokens = tokens;

    for (auto it = lowerTokens.begin(); it != lowerTokens.end(); ++it)
        std::transform(it->begin(), it->end(), it->begin(), ::tolower);

    // Handle pack_matrix
    if (tokens.size() == 4 && lowerTokens[0] == "pack_matrix" && tokens[1] == "(" && tokens[3] == ")") {
        // Note that HLSL semantic order is Mrc, not Mcr like SPIR-V, so we reverse the sense.
        // Row major becomes column major and vice versa.

        if (lowerTokens[2] == "row_major") {
            globalUniformDefaults.layoutMatrix = globalBufferDefaults.layoutMatrix = ElmColumnMajor;
        } else if (lowerTokens[2] == "column_major") {
            globalUniformDefaults.layoutMatrix = globalBufferDefaults.layoutMatrix = ElmRowMajor;
        } else {
            // unknown majorness strings are treated as (HLSL column major)==(SPIR-V row major)
            warn(loc, "unknown pack_matrix pragma value", tokens[2].c_str(), "");
            globalUniformDefaults.layoutMatrix = globalBufferDefaults.layoutMatrix = ElmRowMajor;
        }
        return;
    }

    // Handle once
    if (lowerTokens[0] == "once") {
        warn(loc, "not implemented", "#pragma once", "");
        return;
    }
}

//
// Look at a '.' matrix selector string and change it into components
// for a matrix. There are two types:
//
//   _21    second row, first column (one based)
//   _m21   third row, second column (zero based)
//
// Returns true if there is no error.
//
bool HlslParseContext::parseMatrixSwizzleSelector(const TSourceLoc& loc, const TString& fields, int cols, int rows,
                                                  TSwizzleSelectors<TMatrixSelector>& components)
{
    int startPos[MaxSwizzleSelectors];
    int numComps = 0;
    TString compString = fields;

    // Find where each component starts,
    // recording the first character position after the '_'.
    for (size_t c = 0; c < compString.size(); ++c) {
        if (compString[c] == '_') {
            if (numComps >= MaxSwizzleSelectors) {
                error(loc, "matrix component swizzle has too many components", compString.c_str(), "");
                return false;
            }
            if (c > compString.size() - 3 ||
                    ((compString[c+1] == 'm' || compString[c+1] == 'M') && c > compString.size() - 4)) {
                error(loc, "matrix component swizzle missing", compString.c_str(), "");
                return false;
            }
            startPos[numComps++] = (int)c + 1;
        }
    }

    // Process each component
    for (int i = 0; i < numComps; ++i) {
        int pos = startPos[i];
        int bias = -1;
        if (compString[pos] == 'm' || compString[pos] == 'M') {
            bias = 0;
            ++pos;
        }
        TMatrixSelector comp;
        comp.coord1 = compString[pos+0] - '0' + bias;
        comp.coord2 = compString[pos+1] - '0' + bias;
        if (comp.coord1 < 0 || comp.coord1 >= cols) {
            error(loc, "matrix row component out of range", compString.c_str(), "");
            return false;
        }
        if (comp.coord2 < 0 || comp.coord2 >= rows) {
            error(loc, "matrix column component out of range", compString.c_str(), "");
            return false;
        }
        components.push_back(comp);
    }

    return true;
}

// If the 'comps' express a column of a matrix,
// return the column.  Column means the first coords all match.
//
// Otherwise, return -1.
//
int HlslParseContext::getMatrixComponentsColumn(int rows, const TSwizzleSelectors<TMatrixSelector>& selector)
{
    int col = -1;

    // right number of comps?
    if (selector.size() != rows)
        return -1;

    // all comps in the same column?
    // rows in order?
    col = selector[0].coord1;
    for (int i = 0; i < rows; ++i) {
        if (col != selector[i].coord1)
            return -1;
        if (i != selector[i].coord2)
            return -1;
    }

    return col;
}

//
// Handle seeing a variable identifier in the grammar.
//
TIntermTyped* HlslParseContext::handleVariable(const TSourceLoc& loc, const TString* string)
{
    int thisDepth;
    TSymbol* symbol = symbolTable.find(*string, thisDepth);
    if (symbol && symbol->getAsVariable() && symbol->getAsVariable()->isUserType()) {
        error(loc, "expected symbol, not user-defined type", string->c_str(), "");
        return nullptr;
    }

    const TVariable* variable = nullptr;
    const TAnonMember* anon = symbol ? symbol->getAsAnonMember() : nullptr;
    TIntermTyped* node = nullptr;
    if (anon) {
        // It was a member of an anonymous container, which could be a 'this' structure.

        // Create a subtree for its dereference.
        if (thisDepth > 0) {
            variable = getImplicitThis(thisDepth);
            if (variable == nullptr)
                error(loc, "cannot access member variables (static member function?)", "this", "");
        }
        if (variable == nullptr)
            variable = anon->getAnonContainer().getAsVariable();

        TIntermTyped* container = intermediate.addSymbol(*variable, loc);
        TIntermTyped* constNode = intermediate.addConstantUnion(anon->getMemberNumber(), loc);
        node = intermediate.addIndex(EOpIndexDirectStruct, container, constNode, loc);

        node->setType(*(*variable->getType().getStruct())[anon->getMemberNumber()].type);
        if (node->getType().hiddenMember())
            error(loc, "member of nameless block was not redeclared", string->c_str(), "");
    } else {
        // Not a member of an anonymous container.

        // The symbol table search was done in the lexical phase.
        // See if it was a variable.
        variable = symbol ? symbol->getAsVariable() : nullptr;
        if (variable) {
            if ((variable->getType().getBasicType() == EbtBlock ||
                variable->getType().getBasicType() == EbtStruct) && variable->getType().getStruct() == nullptr) {
                error(loc, "cannot be used (maybe an instance name is needed)", string->c_str(), "");
                variable = nullptr;
            }
        } else {
            if (symbol)
                error(loc, "variable name expected", string->c_str(), "");
        }

        // Recovery, if it wasn't found or was not a variable.
        if (variable == nullptr) {
            error(loc, "unknown variable", string->c_str(), "");
            variable = new TVariable(string, TType(EbtVoid));
        }

        if (variable->getType().getQualifier().isFrontEndConstant())
            node = intermediate.addConstantUnion(variable->getConstArray(), variable->getType(), loc);
        else
            node = intermediate.addSymbol(*variable, loc);
    }

    if (variable->getType().getQualifier().isIo())
        intermediate.addIoAccessed(*string);

    return node;
}

//
// Handle operator[] on any objects it applies to.  Currently:
//    Textures
//    Buffers
//
TIntermTyped* HlslParseContext::handleBracketOperator(const TSourceLoc& loc, TIntermTyped* base, TIntermTyped* index)
{
    // handle r-value operator[] on textures and images.  l-values will be processed later.
    if (base->getType().getBasicType() == EbtSampler && !base->isArray()) {
        const TSampler& sampler = base->getType().getSampler();
        if (sampler.isImage() || sampler.isTexture()) {
            if (! mipsOperatorMipArg.empty() && mipsOperatorMipArg.back().mipLevel == nullptr) {
                // The first operator[] to a .mips[] sequence is the mip level.  We'll remember it.
                mipsOperatorMipArg.back().mipLevel = index;
                return base;  // next [] index is to the same base.
            } else {
                TIntermAggregate* load = new TIntermAggregate(sampler.isImage() ? EOpImageLoad : EOpTextureFetch);

                TType sampReturnType;
                getTextureReturnType(sampler, sampReturnType);

                load->setType(sampReturnType);
                load->setLoc(loc);
                load->getSequence().push_back(base);
                load->getSequence().push_back(index);

                // Textures need a MIP.  If we saw one go by, use it.  Otherwise, use zero.
                if (sampler.isTexture()) {
                    if (! mipsOperatorMipArg.empty()) {
                        load->getSequence().push_back(mipsOperatorMipArg.back().mipLevel);
                        mipsOperatorMipArg.pop_back();
                    } else {
                        load->getSequence().push_back(intermediate.addConstantUnion(0, loc, true));
                    }
                }

                return load;
            }
        }
    }

    // Handle operator[] on structured buffers: this indexes into the array element of the buffer.
    // indexStructBufferContent returns nullptr if it isn't a structuredbuffer (SSBO).
    TIntermTyped* sbArray = indexStructBufferContent(loc, base);
    if (sbArray != nullptr) {
        if (sbArray == nullptr)
            return nullptr;

        // Now we'll apply the [] index to that array
        const TOperator idxOp = (index->getQualifier().storage == EvqConst) ? EOpIndexDirect : EOpIndexIndirect;

        TIntermTyped* element = intermediate.addIndex(idxOp, sbArray, index, loc);
        const TType derefType(sbArray->getType(), 0);
        element->setType(derefType);
        return element;
    }

    return nullptr;
}

//
// Cast index value to a uint if it isn't already (for operator[], load indexes, etc)
TIntermTyped* HlslParseContext::makeIntegerIndex(TIntermTyped* index)
{
    const TBasicType indexBasicType = index->getType().getBasicType();
    const int vecSize = index->getType().getVectorSize();

    // We can use int types directly as the index
    if (indexBasicType == EbtInt || indexBasicType == EbtUint ||
        indexBasicType == EbtInt64 || indexBasicType == EbtUint64)
        return index;

    // Cast index to unsigned integer if it isn't one.
    return intermediate.addConversion(EOpConstructUint, TType(EbtUint, EvqTemporary, vecSize), index);
}

//
// Handle seeing a base[index] dereference in the grammar.
//
TIntermTyped* HlslParseContext::handleBracketDereference(const TSourceLoc& loc, TIntermTyped* base, TIntermTyped* index)
{
    index = makeIntegerIndex(index);

    if (index == nullptr) {
        error(loc, " unknown index type ", "", "");
        return nullptr;
    }

    TIntermTyped* result = handleBracketOperator(loc, base, index);

    if (result != nullptr)
        return result;  // it was handled as an operator[]

    bool flattened = false;
    int indexValue = 0;
    if (index->getQualifier().isFrontEndConstant())
        indexValue = index->getAsConstantUnion()->getConstArray()[0].getIConst();

    variableCheck(base);
    if (! base->isArray() && ! base->isMatrix() && ! base->isVector()) {
        if (base->getAsSymbolNode())
            error(loc, " left of '[' is not of type array, matrix, or vector ",
                  base->getAsSymbolNode()->getName().c_str(), "");
        else
            error(loc, " left of '[' is not of type array, matrix, or vector ", "expression", "");
    } else if (base->getType().getQualifier().storage == EvqConst && index->getQualifier().storage == EvqConst) {
        // both base and index are front-end constants
        checkIndex(loc, base->getType(), indexValue);
        return intermediate.foldDereference(base, indexValue, loc);
    } else {
        // at least one of base and index is variable...

        if (index->getQualifier().isFrontEndConstant())
            checkIndex(loc, base->getType(), indexValue);

        if (base->getType().isScalarOrVec1())
            result = base;
        else if (base->getAsSymbolNode() && wasFlattened(base)) {
            if (index->getQualifier().storage != EvqConst)
                error(loc, "Invalid variable index to flattened array", base->getAsSymbolNode()->getName().c_str(), "");

            result = flattenAccess(base, indexValue);
            flattened = (result != base);
        } else {
            if (index->getQualifier().isFrontEndConstant()) {
                if (base->getType().isUnsizedArray())
                    base->getWritableType().updateImplicitArraySize(indexValue + 1);
                else
                    checkIndex(loc, base->getType(), indexValue);
                result = intermediate.addIndex(EOpIndexDirect, base, index, loc);
            } else
                result = intermediate.addIndex(EOpIndexIndirect, base, index, loc);
        }
    }

    if (result == nullptr) {
        // Insert dummy error-recovery result
        result = intermediate.addConstantUnion(0.0, EbtFloat, loc);
    } else {
        // If the array reference was flattened, it has the correct type.  E.g, if it was
        // a uniform array, it was flattened INTO a set of scalar uniforms, not scalar temps.
        // In that case, we preserve the qualifiers.
        if (!flattened) {
            // Insert valid dereferenced result
            TType newType(base->getType(), 0);  // dereferenced type
            if (base->getType().getQualifier().storage == EvqConst && index->getQualifier().storage == EvqConst)
                newType.getQualifier().storage = EvqConst;
            else
                newType.getQualifier().storage = EvqTemporary;
            result->setType(newType);
        }
    }

    return result;
}

// Handle seeing a binary node with a math operation.
TIntermTyped* HlslParseContext::handleBinaryMath(const TSourceLoc& loc, const char* str, TOperator op,
                                                 TIntermTyped* left, TIntermTyped* right)
{
    TIntermTyped* result = intermediate.addBinaryMath(op, left, right, loc);
    if (result == nullptr)
        binaryOpError(loc, str, left->getCompleteString(), right->getCompleteString());

    return result;
}

// Handle seeing a unary node with a math operation.
TIntermTyped* HlslParseContext::handleUnaryMath(const TSourceLoc& loc, const char* str, TOperator op,
                                                TIntermTyped* childNode)
{
    TIntermTyped* result = intermediate.addUnaryMath(op, childNode, loc);

    if (result)
        return result;
    else
        unaryOpError(loc, str, childNode->getCompleteString());

    return childNode;
}
//
// Return true if the name is a struct buffer method
//
bool HlslParseContext::isStructBufferMethod(const TString& name) const
{
    return
        name == "GetDimensions"              ||
        name == "Load"                       ||
        name == "Load2"                      ||
        name == "Load3"                      ||
        name == "Load4"                      ||
        name == "Store"                      ||
        name == "Store2"                     ||
        name == "Store3"                     ||
        name == "Store4"                     ||
        name == "InterlockedAdd"             ||
        name == "InterlockedAnd"             ||
        name == "InterlockedCompareExchange" ||
        name == "InterlockedCompareStore"    ||
        name == "InterlockedExchange"        ||
        name == "InterlockedMax"             ||
        name == "InterlockedMin"             ||
        name == "InterlockedOr"              ||
        name == "InterlockedXor"             ||
        name == "IncrementCounter"           ||
        name == "DecrementCounter"           ||
        name == "Append"                     ||
        name == "Consume";
}

//
// Handle seeing a base.field dereference in the grammar, where 'field' is a
// swizzle or member variable.
//
TIntermTyped* HlslParseContext::handleDotDereference(const TSourceLoc& loc, TIntermTyped* base, const TString& field)
{
    variableCheck(base);

    if (base->isArray()) {
        error(loc, "cannot apply to an array:", ".", field.c_str());
        return base;
    }

    TIntermTyped* result = base;

    if (base->getType().getBasicType() == EbtSampler) {
        // Handle .mips[mipid][pos] operation on textures
        const TSampler& sampler = base->getType().getSampler();
        if (sampler.isTexture() && field == "mips") {
            // Push a null to signify that we expect a mip level under operator[] next.
            mipsOperatorMipArg.push_back(tMipsOperatorData(loc, nullptr));
            // Keep 'result' pointing to 'base', since we expect an operator[] to go by next.
        } else {
            if (field == "mips")
                error(loc, "unexpected texture type for .mips[][] operator:",
                      base->getType().getCompleteString().c_str(), "");
            else
                error(loc, "unexpected operator on texture type:", field.c_str(),
                      base->getType().getCompleteString().c_str());
        }
    } else if (base->isVector() || base->isScalar()) {
        TSwizzleSelectors<TVectorSelector> selectors;
        parseSwizzleSelector(loc, field, base->getVectorSize(), selectors);

        if (base->isScalar()) {
            if (selectors.size() == 1)
                return result;
            else {
                TType type(base->getBasicType(), EvqTemporary, selectors.size());
                return addConstructor(loc, base, type);
            }
        }
        if (base->getVectorSize() == 1) {
            TType scalarType(base->getBasicType(), EvqTemporary, 1);
            if (selectors.size() == 1)
                return addConstructor(loc, base, scalarType);
            else {
                TType vectorType(base->getBasicType(), EvqTemporary, selectors.size());
                return addConstructor(loc, addConstructor(loc, base, scalarType), vectorType);
            }
        }

        if (base->getType().getQualifier().isFrontEndConstant())
            result = intermediate.foldSwizzle(base, selectors, loc);
        else {
            if (selectors.size() == 1) {
                TIntermTyped* index = intermediate.addConstantUnion(selectors[0], loc);
                result = intermediate.addIndex(EOpIndexDirect, base, index, loc);
                result->setType(TType(base->getBasicType(), EvqTemporary));
            } else {
                TIntermTyped* index = intermediate.addSwizzle(selectors, loc);
                result = intermediate.addIndex(EOpVectorSwizzle, base, index, loc);
                result->setType(TType(base->getBasicType(), EvqTemporary, base->getType().getQualifier().precision,
                                selectors.size()));
            }
        }
    } else if (base->isMatrix()) {
        TSwizzleSelectors<TMatrixSelector> selectors;
        if (! parseMatrixSwizzleSelector(loc, field, base->getMatrixCols(), base->getMatrixRows(), selectors))
            return result;

        if (selectors.size() == 1) {
            // Representable by m[c][r]
            if (base->getType().getQualifier().isFrontEndConstant()) {
                result = intermediate.foldDereference(base, selectors[0].coord1, loc);
                result = intermediate.foldDereference(result, selectors[0].coord2, loc);
            } else {
                result = intermediate.addIndex(EOpIndexDirect, base,
                                               intermediate.addConstantUnion(selectors[0].coord1, loc),
                                               loc);
                TType dereferencedCol(base->getType(), 0);
                result->setType(dereferencedCol);
                result = intermediate.addIndex(EOpIndexDirect, result,
                                               intermediate.addConstantUnion(selectors[0].coord2, loc),
                                               loc);
                TType dereferenced(dereferencedCol, 0);
                result->setType(dereferenced);
            }
        } else {
            int column = getMatrixComponentsColumn(base->getMatrixRows(), selectors);
            if (column >= 0) {
                // Representable by m[c]
                if (base->getType().getQualifier().isFrontEndConstant())
                    result = intermediate.foldDereference(base, column, loc);
                else {
                    result = intermediate.addIndex(EOpIndexDirect, base, intermediate.addConstantUnion(column, loc),
                                                   loc);
                    TType dereferenced(base->getType(), 0);
                    result->setType(dereferenced);
                }
            } else {
                // general case, not a column, not a single component
                TIntermTyped* index = intermediate.addSwizzle(selectors, loc);
                result = intermediate.addIndex(EOpMatrixSwizzle, base, index, loc);
                result->setType(TType(base->getBasicType(), EvqTemporary, base->getType().getQualifier().precision,
                                      selectors.size()));
           }
        }
    } else if (base->getBasicType() == EbtStruct || base->getBasicType() == EbtBlock) {
        const TTypeList* fields = base->getType().getStruct();
        bool fieldFound = false;
        int member;
        for (member = 0; member < (int)fields->size(); ++member) {
            if ((*fields)[member].type->getFieldName() == field) {
                fieldFound = true;
                break;
            }
        }
        if (fieldFound) {
            if (base->getAsSymbolNode() && wasFlattened(base)) {
                result = flattenAccess(base, member);
            } else {
                if (base->getType().getQualifier().storage == EvqConst)
                    result = intermediate.foldDereference(base, member, loc);
                else {
                    TIntermTyped* index = intermediate.addConstantUnion(member, loc);
                    result = intermediate.addIndex(EOpIndexDirectStruct, base, index, loc);
                    result->setType(*(*fields)[member].type);
                }
            }
        } else
            error(loc, "no such field in structure", field.c_str(), "");
    } else
        error(loc, "does not apply to this type:", field.c_str(), base->getType().getCompleteString().c_str());

    return result;
}

//
// Return true if the field should be treated as a built-in method.
// Return false otherwise.
//
bool HlslParseContext::isBuiltInMethod(const TSourceLoc&, TIntermTyped* base, const TString& field)
{
    if (base == nullptr)
        return false;

    variableCheck(base);

    if (base->getType().getBasicType() == EbtSampler) {
        return true;
    } else if (isStructBufferType(base->getType()) && isStructBufferMethod(field)) {
        return true;
    } else if (field == "Append" ||
               field == "RestartStrip") {
        // We cannot check the type here: it may be sanitized if we're not compiling a geometry shader, but
        // the code is around in the shader source.
        return true;
    } else
        return false;
}

// Independently establish a built-in that is a member of a structure.
// 'arraySizes' are what's desired for the independent built-in, whatever
// the higher-level source/expression of them was.
void HlslParseContext::splitBuiltIn(const TString& baseName, const TType& memberType, const TArraySizes* arraySizes,
                                    const TQualifier& outerQualifier)
{
    // Because of arrays of structs, we might be asked more than once,
    // but the arraySizes passed in should have captured the whole thing
    // the first time.
    // However, clip/cull rely on multiple updates.
    if (!isClipOrCullDistance(memberType))
        if (splitBuiltIns.find(tInterstageIoData(memberType.getQualifier().builtIn, outerQualifier.storage)) !=
            splitBuiltIns.end())
            return;

    TVariable* ioVar = makeInternalVariable(baseName + "." + memberType.getFieldName(), memberType);

    if (arraySizes != nullptr && !memberType.isArray())
        ioVar->getWritableType().copyArraySizes(*arraySizes);

    splitBuiltIns[tInterstageIoData(memberType.getQualifier().builtIn, outerQualifier.storage)] = ioVar;
    if (!isClipOrCullDistance(ioVar->getType()))
        trackLinkage(*ioVar);

    // Merge qualifier from the user structure
    mergeQualifiers(ioVar->getWritableType().getQualifier(), outerQualifier);

    // Fix the builtin type if needed (e.g, some types require fixed array sizes, no matter how the
    // shader declared them).  This is done after mergeQualifiers(), in case fixBuiltInIoType looks
    // at the qualifier to determine e.g, in or out qualifications.
    fixBuiltInIoType(ioVar->getWritableType());

    // But, not location, we're losing that
    ioVar->getWritableType().getQualifier().layoutLocation = TQualifier::layoutLocationEnd;
}

// Split a type into
//   1. a struct of non-I/O members
//   2. a collection of independent I/O variables
void HlslParseContext::split(const TVariable& variable)
{
    // Create a new variable:
    const TType& clonedType = *variable.getType().clone();
    const TType& splitType = split(clonedType, variable.getName(), clonedType.getQualifier());
    splitNonIoVars[variable.getUniqueId()] = makeInternalVariable(variable.getName(), splitType);
}

// Recursive implementation of split().
// Returns reference to the modified type.
const TType& HlslParseContext::split(const TType& type, const TString& name, const TQualifier& outerQualifier)
{
    if (type.isStruct()) {
        TTypeList* userStructure = type.getWritableStruct();
        for (auto ioType = userStructure->begin(); ioType != userStructure->end(); ) {
            if (ioType->type->isBuiltIn()) {
                // move out the built-in
                splitBuiltIn(name, *ioType->type, type.getArraySizes(), outerQualifier);
                ioType = userStructure->erase(ioType);
            } else {
                split(*ioType->type, name + "." + ioType->type->getFieldName(), outerQualifier);
                ++ioType;
            }
        }
    }

    return type;
}

// Is this an aggregate that should be flattened?
// Can be applied to intermediate levels of type in a hierarchy.
// Some things like flattening uniform arrays are only about the top level
// of the aggregate, triggered on 'topLevel'.
bool HlslParseContext::shouldFlatten(const TType& type, TStorageQualifier qualifier, bool topLevel) const
{
    switch (qualifier) {
    case EvqVaryingIn:
    case EvqVaryingOut:
        return type.isStruct() || type.isArray();
    case EvqUniform:
        return (type.isArray() && intermediate.getFlattenUniformArrays() && topLevel) ||
               (type.isStruct() && type.containsOpaque());
    default:
        return false;
    };
}

// Top level variable flattening: construct data
void HlslParseContext::flatten(const TVariable& variable, bool linkage)
{
    const TType& type = variable.getType();

    // If it's a standalone built-in, there is nothing to flatten
    if (type.isBuiltIn() && !type.isStruct())
        return;

    auto entry = flattenMap.insert(std::make_pair(variable.getUniqueId(),
                                                  TFlattenData(type.getQualifier().layoutBinding,
                                                               type.getQualifier().layoutLocation)));

    // the item is a map pair, so first->second is the TFlattenData itself.
    flatten(variable, type, entry.first->second, variable.getName(), linkage, type.getQualifier(), nullptr);
}

// Recursively flatten the given variable at the provided type, building the flattenData as we go.
//
// This is mutually recursive with flattenStruct and flattenArray.
// We are going to flatten an arbitrarily nested composite structure into a linear sequence of
// members, and later on, we want to turn a path through the tree structure into a final
// location in this linear sequence.
//
// If the tree was N-ary, that can be directly calculated.  However, we are dealing with
// arbitrary numbers - perhaps a struct of 7 members containing an array of 3.  Thus, we must
// build a data structure to allow the sequence of bracket and dot operators on arrays and
// structs to arrive at the proper member.
//
// To avoid storing a tree with pointers, we are going to flatten the tree into a vector of integers.
// The leaves are the indexes into the flattened member array.
// Each level will have the next location for the Nth item stored sequentially, so for instance:
//
// struct { float2 a[2]; int b; float4 c[3] };
//
// This will produce the following flattened tree:
// Pos: 0  1   2    3  4    5  6   7     8   9  10   11  12 13
//     (3, 7,  8,   5, 6,   0, 1,  2,   11, 12, 13,   3,  4, 5}
//
// Given a reference to mystruct.c[1], the access chain is (2,1), so we traverse:
//   (0+2) = 8  -->  (8+1) = 12 -->   12 = 4
//
// so the 4th flattened member in traversal order is ours.
//
int HlslParseContext::flatten(const TVariable& variable, const TType& type,
                              TFlattenData& flattenData, TString name, bool linkage,
                              const TQualifier& outerQualifier,
                              const TArraySizes* builtInArraySizes)
{
    // If something is an arrayed struct, the array flattener will recursively call flatten()
    // to then flatten the struct, so this is an "if else": we don't do both.
    if (type.isArray())
        return flattenArray(variable, type, flattenData, name, linkage, outerQualifier);
    else if (type.isStruct())
        return flattenStruct(variable, type, flattenData, name, linkage, outerQualifier, builtInArraySizes);
    else {
        assert(0); // should never happen
        return -1;
    }
}

// Add a single flattened member to the flattened data being tracked for the composite
// Returns true for the final flattening level.
int HlslParseContext::addFlattenedMember(const TVariable& variable, const TType& type, TFlattenData& flattenData,
                                         const TString& memberName, bool linkage,
                                         const TQualifier& outerQualifier,
                                         const TArraySizes* builtInArraySizes)
{
    if (!shouldFlatten(type, outerQualifier.storage, false)) {
        // This is as far as we flatten.  Insert the variable.
        TVariable* memberVariable = makeInternalVariable(memberName, type);
        mergeQualifiers(memberVariable->getWritableType().getQualifier(), variable.getType().getQualifier());

        if (flattenData.nextBinding != TQualifier::layoutBindingEnd)
            memberVariable->getWritableType().getQualifier().layoutBinding = flattenData.nextBinding++;

        if (memberVariable->getType().isBuiltIn()) {
            // inherited locations are nonsensical for built-ins (TODO: what if semantic had a number)
            memberVariable->getWritableType().getQualifier().layoutLocation = TQualifier::layoutLocationEnd;
        } else {
            // inherited locations must be auto bumped, not replicated
            if (flattenData.nextLocation != TQualifier::layoutLocationEnd) {
                memberVariable->getWritableType().getQualifier().layoutLocation = flattenData.nextLocation;
                flattenData.nextLocation += intermediate.computeTypeLocationSize(memberVariable->getType(), language);
                nextOutLocation = std::max(nextOutLocation, flattenData.nextLocation);
            }
        }

        flattenData.offsets.push_back(static_cast<int>(flattenData.members.size()));
        flattenData.members.push_back(memberVariable);

        if (linkage)
            trackLinkage(*memberVariable);

        return static_cast<int>(flattenData.offsets.size()) - 1; // location of the member reference
    } else {
        // Further recursion required
        return flatten(variable, type, flattenData, memberName, linkage, outerQualifier, builtInArraySizes);
    }
}

// Figure out the mapping between an aggregate's top members and an
// equivalent set of individual variables.
//
// Assumes shouldFlatten() or equivalent was called first.
int HlslParseContext::flattenStruct(const TVariable& variable, const TType& type,
                                    TFlattenData& flattenData, TString name, bool linkage,
                                    const TQualifier& outerQualifier,
                                    const TArraySizes* builtInArraySizes)
{
    assert(type.isStruct());

    auto members = *type.getStruct();

    // Reserve space for this tree level.
    int start = static_cast<int>(flattenData.offsets.size());
    int pos = start;
    flattenData.offsets.resize(int(pos + members.size()), -1);

    for (int member = 0; member < (int)members.size(); ++member) {
        TType& dereferencedType = *members[member].type;
        if (dereferencedType.isBuiltIn())
            splitBuiltIn(variable.getName(), dereferencedType, builtInArraySizes, outerQualifier);
        else {
            const int mpos = addFlattenedMember(variable, dereferencedType, flattenData,
                                                name + "." + dereferencedType.getFieldName(),
                                                linkage, outerQualifier,
                                                builtInArraySizes == nullptr && dereferencedType.isArray()
                                                                       ? dereferencedType.getArraySizes()
                                                                       : builtInArraySizes);
            flattenData.offsets[pos++] = mpos;
        }
    }

    return start;
}

// Figure out mapping between an array's members and an
// equivalent set of individual variables.
//
// Assumes shouldFlatten() or equivalent was called first.
int HlslParseContext::flattenArray(const TVariable& variable, const TType& type,
                                   TFlattenData& flattenData, TString name, bool linkage,
                                   const TQualifier& outerQualifier)
{
    assert(type.isSizedArray());

    const int size = type.getOuterArraySize();
    const TType dereferencedType(type, 0);

    if (name.empty())
        name = variable.getName();

    // Reserve space for this tree level.
    int start = static_cast<int>(flattenData.offsets.size());
    int pos   = start;
    flattenData.offsets.resize(int(pos + size), -1);

    for (int element=0; element < size; ++element) {
        char elementNumBuf[20];  // sufficient for MAXINT
        snprintf(elementNumBuf, sizeof(elementNumBuf)-1, "[%d]", element);
        const int mpos = addFlattenedMember(variable, dereferencedType, flattenData,
                                            name + elementNumBuf, linkage, outerQualifier,
                                            type.getArraySizes());

        flattenData.offsets[pos++] = mpos;
    }

    return start;
}

// Return true if we have flattened this node.
bool HlslParseContext::wasFlattened(const TIntermTyped* node) const
{
    return node != nullptr && node->getAsSymbolNode() != nullptr &&
           wasFlattened(node->getAsSymbolNode()->getId());
}

// Return true if we have split this structure
bool HlslParseContext::wasSplit(const TIntermTyped* node) const
{
    return node != nullptr && node->getAsSymbolNode() != nullptr &&
           wasSplit(node->getAsSymbolNode()->getId());
}

// Turn an access into an aggregate that was flattened to instead be
// an access to the individual variable the member was flattened to.
// Assumes wasFlattened() or equivalent was called first.
TIntermTyped* HlslParseContext::flattenAccess(TIntermTyped* base, int member)
{
    const TType dereferencedType(base->getType(), member);  // dereferenced type
    const TIntermSymbol& symbolNode = *base->getAsSymbolNode();
    TIntermTyped* flattened = flattenAccess(symbolNode.getId(), member, base->getQualifier().storage,
                                            dereferencedType, symbolNode.getFlattenSubset());

    return flattened ? flattened : base;
}
TIntermTyped* HlslParseContext::flattenAccess(int uniqueId, int member, TStorageQualifier outerStorage,
    const TType& dereferencedType, int subset)
{
    const auto flattenData = flattenMap.find(uniqueId);

    if (flattenData == flattenMap.end())
        return nullptr;

    // Calculate new cumulative offset from the packed tree
    int newSubset = flattenData->second.offsets[subset >= 0 ? subset + member : member];

    TIntermSymbol* subsetSymbol;
    if (!shouldFlatten(dereferencedType, outerStorage, false)) {
        // Finished flattening: create symbol for variable
        member = flattenData->second.offsets[newSubset];
        const TVariable* memberVariable = flattenData->second.members[member];
        subsetSymbol = intermediate.addSymbol(*memberVariable);
        subsetSymbol->setFlattenSubset(-1);
    } else {

        // If this is not the final flattening, accumulate the position and return
        // an object of the partially dereferenced type.
        subsetSymbol = new TIntermSymbol(uniqueId, "flattenShadow", dereferencedType);
        subsetSymbol->setFlattenSubset(newSubset);
    }

    return subsetSymbol;
}

// For finding where the first leaf is in a subtree of a multi-level aggregate
// that is just getting a subset assigned. Follows the same logic as flattenAccess,
// but logically going down the "left-most" tree branch each step of the way.
//
// Returns the offset into the first leaf of the subset.
int HlslParseContext::findSubtreeOffset(const TIntermNode& node) const
{
    const TIntermSymbol* sym = node.getAsSymbolNode();
    if (sym == nullptr)
        return 0;
    if (!sym->isArray() && !sym->isStruct())
        return 0;
    int subset = sym->getFlattenSubset();
    if (subset == -1)
        return 0;

    // Getting this far means a partial aggregate is identified by the flatten subset.
    // Find the first leaf of the subset.

    const auto flattenData = flattenMap.find(sym->getId());
    if (flattenData == flattenMap.end())
        return 0;

    return findSubtreeOffset(sym->getType(), subset, flattenData->second.offsets);

    do {
        subset = flattenData->second.offsets[subset];
    } while (true);
}
// Recursively do the desent
int HlslParseContext::findSubtreeOffset(const TType& type, int subset, const TVector<int>& offsets) const
{
    if (!type.isArray() && !type.isStruct())
        return offsets[subset];
    TType derefType(type, 0);
    return findSubtreeOffset(derefType, offsets[subset], offsets);
};

// Find and return the split IO TVariable for id, or nullptr if none.
TVariable* HlslParseContext::getSplitNonIoVar(int id) const
{
    const auto splitNonIoVar = splitNonIoVars.find(id);
    if (splitNonIoVar == splitNonIoVars.end())
        return nullptr;

    return splitNonIoVar->second;
}

// Pass through to base class after remembering built-in mappings.
void HlslParseContext::trackLinkage(TSymbol& symbol)
{
    TBuiltInVariable biType = symbol.getType().getQualifier().builtIn;

    if (biType != EbvNone)
        builtInTessLinkageSymbols[biType] = symbol.clone();

    TParseContextBase::trackLinkage(symbol);
}


// Returns true if the built-in is a clip or cull distance variable.
bool HlslParseContext::isClipOrCullDistance(TBuiltInVariable builtIn)
{
    return builtIn == EbvClipDistance || builtIn == EbvCullDistance;
}

// Some types require fixed array sizes in SPIR-V, but can be scalars or
// arrays of sizes SPIR-V doesn't allow.  For example, tessellation factors.
// This creates the right size.  A conversion is performed when the internal
// type is copied to or from the external type.  This corrects the externally
// facing input or output type to abide downstream semantics.
void HlslParseContext::fixBuiltInIoType(TType& type)
{
    int requiredArraySize = 0;
    int requiredVectorSize = 0;

    switch (type.getQualifier().builtIn) {
    case EbvTessLevelOuter: requiredArraySize = 4; break;
    case EbvTessLevelInner: requiredArraySize = 2; break;

    case EbvSampleMask:
        {
            // Promote scalar to array of size 1.  Leave existing arrays alone.
            if (!type.isArray())
                requiredArraySize = 1;
            break;
        }

    case EbvWorkGroupId:        requiredVectorSize = 3; break;
    case EbvGlobalInvocationId: requiredVectorSize = 3; break;
    case EbvLocalInvocationId:  requiredVectorSize = 3; break;
    case EbvTessCoord:          requiredVectorSize = 3; break;

    default:
        if (isClipOrCullDistance(type)) {
            const int loc = type.getQualifier().layoutLocation;

            if (type.getQualifier().builtIn == EbvClipDistance) {
                if (type.getQualifier().storage == EvqVaryingIn)
                    clipSemanticNSizeIn[loc] = type.getVectorSize();
                else
                    clipSemanticNSizeOut[loc] = type.getVectorSize();
            } else {
                if (type.getQualifier().storage == EvqVaryingIn)
                    cullSemanticNSizeIn[loc] = type.getVectorSize();
                else
                    cullSemanticNSizeOut[loc] = type.getVectorSize();
            }
        }

        return;
    }

    // Alter or set vector size as needed.
    if (requiredVectorSize > 0) {
        TType newType(type.getBasicType(), type.getQualifier().storage, requiredVectorSize);
        newType.getQualifier() = type.getQualifier();

        type.shallowCopy(newType);
    }

    // Alter or set array size as needed.
    if (requiredArraySize > 0) {
        if (!type.isArray() || type.getOuterArraySize() != requiredArraySize) {
            TArraySizes* arraySizes = new TArraySizes;
            arraySizes->addInnerSize(requiredArraySize);
            type.transferArraySizes(arraySizes);
        }
    }
}

// Variables that correspond to the user-interface in and out of a stage
// (not the built-in interface) are
//  - assigned locations
//  - registered as a linkage node (part of the stage's external interface).
// Assumes it is called in the order in which locations should be assigned.
void HlslParseContext::assignToInterface(TVariable& variable)
{
    const auto assignLocation = [&](TVariable& variable) {
        TType& type = variable.getWritableType();
        if (!type.isStruct() || type.getStruct()->size() > 0) {
            TQualifier& qualifier = type.getQualifier();
            if (qualifier.storage == EvqVaryingIn || qualifier.storage == EvqVaryingOut) {
                if (qualifier.builtIn == EbvNone && !qualifier.hasLocation()) {
                    // Strip off the outer array dimension for those having an extra one.
                    int size;
                    if (type.isArray() && qualifier.isArrayedIo(language)) {
                        TType elementType(type, 0);
                        size = intermediate.computeTypeLocationSize(elementType, language);
                    } else
                        size = intermediate.computeTypeLocationSize(type, language);

                    if (qualifier.storage == EvqVaryingIn) {
                        variable.getWritableType().getQualifier().layoutLocation = nextInLocation;
                        nextInLocation += size;
                    } else {
                        variable.getWritableType().getQualifier().layoutLocation = nextOutLocation;
                        nextOutLocation += size;
                    }
                }
                trackLinkage(variable);
            }
        }
    };

    if (wasFlattened(variable.getUniqueId())) {
        auto& memberList = flattenMap[variable.getUniqueId()].members;
        for (auto member = memberList.begin(); member != memberList.end(); ++member)
            assignLocation(**member);
    } else if (wasSplit(variable.getUniqueId())) {
        TVariable* splitIoVar = getSplitNonIoVar(variable.getUniqueId());
        assignLocation(*splitIoVar);
    } else {
        assignLocation(variable);
    }
}

//
// Handle seeing a function declarator in the grammar.  This is the precursor
// to recognizing a function prototype or function definition.
//
void HlslParseContext::handleFunctionDeclarator(const TSourceLoc& loc, TFunction& function, bool prototype)
{
    //
    // Multiple declarations of the same function name are allowed.
    //
    // If this is a definition, the definition production code will check for redefinitions
    // (we don't know at this point if it's a definition or not).
    //
    bool builtIn;
    TSymbol* symbol = symbolTable.find(function.getMangledName(), &builtIn);
    const TFunction* prevDec = symbol ? symbol->getAsFunction() : 0;

    if (prototype) {
        // All built-in functions are defined, even though they don't have a body.
        // Count their prototype as a definition instead.
        if (symbolTable.atBuiltInLevel())
            function.setDefined();
        else {
            if (prevDec && ! builtIn)
                symbol->getAsFunction()->setPrototyped();  // need a writable one, but like having prevDec as a const
            function.setPrototyped();
        }
    }

    // This insert won't actually insert it if it's a duplicate signature, but it will still check for
    // other forms of name collisions.
    if (! symbolTable.insert(function))
        error(loc, "function name is redeclaration of existing name", function.getName().c_str(), "");
}

// For struct buffers with counters, we must pass the counter buffer as hidden parameter.
// This adds the hidden parameter to the parameter list in 'paramNodes' if needed.
// Otherwise, it's a no-op
void HlslParseContext::addStructBufferHiddenCounterParam(const TSourceLoc& loc, TParameter& param,
                                                         TIntermAggregate*& paramNodes)
{
    if (! hasStructBuffCounter(*param.type))
        return;

    const TString counterBlockName(intermediate.addCounterBufferName(*param.name));

    TType counterType;
    counterBufferType(loc, counterType);
    TVariable *variable = makeInternalVariable(counterBlockName, counterType);

    if (! symbolTable.insert(*variable))
        error(loc, "redefinition", variable->getName().c_str(), "");

    paramNodes = intermediate.growAggregate(paramNodes,
                                            intermediate.addSymbol(*variable, loc),
                                            loc);
}

//
// Handle seeing the function prototype in front of a function definition in the grammar.
// The body is handled after this function returns.
//
// Returns an aggregate of parameter-symbol nodes.
//
TIntermAggregate* HlslParseContext::handleFunctionDefinition(const TSourceLoc& loc, TFunction& function,
                                                             const TAttributes& attributes,
                                                             TIntermNode*& entryPointTree)
{
    currentCaller = function.getMangledName();
    TSymbol* symbol = symbolTable.find(function.getMangledName());
    TFunction* prevDec = symbol ? symbol->getAsFunction() : nullptr;

    if (prevDec == nullptr)
        error(loc, "can't find function", function.getName().c_str(), "");
    // Note:  'prevDec' could be 'function' if this is the first time we've seen function
    // as it would have just been put in the symbol table.  Otherwise, we're looking up
    // an earlier occurrence.

    if (prevDec && prevDec->isDefined()) {
        // Then this function already has a body.
        error(loc, "function already has a body", function.getName().c_str(), "");
    }
    if (prevDec && ! prevDec->isDefined()) {
        prevDec->setDefined();

        // Remember the return type for later checking for RETURN statements.
        currentFunctionType = &(prevDec->getType());
    } else
        currentFunctionType = new TType(EbtVoid);
    functionReturnsValue = false;

    // Entry points need different I/O and other handling, transform it so the
    // rest of this function doesn't care.
    entryPointTree = transformEntryPoint(loc, function, attributes);

    //
    // New symbol table scope for body of function plus its arguments
    //
    pushScope();

    //
    // Insert parameters into the symbol table.
    // If the parameter has no name, it's not an error, just don't insert it
    // (could be used for unused args).
    //
    // Also, accumulate the list of parameters into the AST, so lower level code
    // knows where to find parameters.
    //
    TIntermAggregate* paramNodes = new TIntermAggregate;
    for (int i = 0; i < function.getParamCount(); i++) {
        TParameter& param = function[i];
        if (param.name != nullptr) {
            TVariable *variable = new TVariable(param.name, *param.type);

            if (i == 0 && function.hasImplicitThis()) {
                // Anonymous 'this' members are already in a symbol-table level,
                // and we need to know what function parameter to map them to.
                symbolTable.makeInternalVariable(*variable);
                pushImplicitThis(variable);
            }

            // Insert the parameters with name in the symbol table.
            if (! symbolTable.insert(*variable))
                error(loc, "redefinition", variable->getName().c_str(), "");

            // Add parameters to the AST list.
            if (shouldFlatten(variable->getType(), variable->getType().getQualifier().storage, true)) {
                // Expand the AST parameter nodes (but not the name mangling or symbol table view)
                // for structures that need to be flattened.
                flatten(*variable, false);
                const TTypeList* structure = variable->getType().getStruct();
                for (int mem = 0; mem < (int)structure->size(); ++mem) {
                    paramNodes = intermediate.growAggregate(paramNodes,
                                                            flattenAccess(variable->getUniqueId(), mem,
                                                                          variable->getType().getQualifier().storage,
                                                                          *(*structure)[mem].type),
                                                            loc);
                }
            } else {
                // Add the parameter to the AST
                paramNodes = intermediate.growAggregate(paramNodes,
                                                        intermediate.addSymbol(*variable, loc),
                                                        loc);
            }

            // Add hidden AST parameter for struct buffer counters, if needed.
            addStructBufferHiddenCounterParam(loc, param, paramNodes);
        } else
            paramNodes = intermediate.growAggregate(paramNodes, intermediate.addSymbol(*param.type, loc), loc);
    }
    if (function.hasIllegalImplicitThis())
        pushImplicitThis(nullptr);

    intermediate.setAggregateOperator(paramNodes, EOpParameters, TType(EbtVoid), loc);
    loopNestingLevel = 0;
    controlFlowNestingLevel = 0;
    postEntryPointReturn = false;

    return paramNodes;
}

// Handle all [attrib] attribute for the shader entry point
void HlslParseContext::handleEntryPointAttributes(const TSourceLoc& loc, const TAttributes& attributes)
{
    for (auto it = attributes.begin(); it != attributes.end(); ++it) {
        switch (it->name) {
        case EatNumThreads:
        {
            const TIntermSequence& sequence = it->args->getSequence();
            for (int lid = 0; lid < int(sequence.size()); ++lid)
                intermediate.setLocalSize(lid, sequence[lid]->getAsConstantUnion()->getConstArray()[0].getIConst());
            break;
        }
        case EatMaxVertexCount:
        {
            int maxVertexCount;

            if (! it->getInt(maxVertexCount)) {
                error(loc, "invalid maxvertexcount", "", "");
            } else {
                if (! intermediate.setVertices(maxVertexCount))
                    error(loc, "cannot change previously set maxvertexcount attribute", "", "");
            }
            break;
        }
        case EatPatchConstantFunc:
        {
            TString pcfName;
            if (! it->getString(pcfName, 0, false)) {
                error(loc, "invalid patch constant function", "", "");
            } else {
                patchConstantFunctionName = pcfName;
            }
            break;
        }
        case EatDomain:
        {
            // Handle [domain("...")]
            TString domainStr;
            if (! it->getString(domainStr)) {
                error(loc, "invalid domain", "", "");
            } else {
                TLayoutGeometry domain = ElgNone;

                if (domainStr == "tri") {
                    domain = ElgTriangles;
                } else if (domainStr == "quad") {
                    domain = ElgQuads;
                } else if (domainStr == "isoline") {
                    domain = ElgIsolines;
                } else {
                    error(loc, "unsupported domain type", domainStr.c_str(), "");
                }

                if (language == EShLangTessEvaluation) {
                    if (! intermediate.setInputPrimitive(domain))
                        error(loc, "cannot change previously set domain", TQualifier::getGeometryString(domain), "");
                } else {
                    if (! intermediate.setOutputPrimitive(domain))
                        error(loc, "cannot change previously set domain", TQualifier::getGeometryString(domain), "");
                }
            }
            break;
        }
        case EatOutputTopology:
        {
            // Handle [outputtopology("...")]
            TString topologyStr;
            if (! it->getString(topologyStr)) {
                error(loc, "invalid outputtopology", "", "");
            } else {
                TVertexOrder vertexOrder = EvoNone;
                TLayoutGeometry primitive = ElgNone;

                if (topologyStr == "point") {
                    intermediate.setPointMode();
                } else if (topologyStr == "line") {
                    primitive = ElgIsolines;
                } else if (topologyStr == "triangle_cw") {
                    vertexOrder = EvoCw;
                    primitive = ElgTriangles;
                } else if (topologyStr == "triangle_ccw") {
                    vertexOrder = EvoCcw;
                    primitive = ElgTriangles;
                } else {
                    error(loc, "unsupported outputtopology type", topologyStr.c_str(), "");
                }

                if (vertexOrder != EvoNone) {
                    if (! intermediate.setVertexOrder(vertexOrder)) {
                        error(loc, "cannot change previously set outputtopology",
                              TQualifier::getVertexOrderString(vertexOrder), "");
                    }
                }
                if (primitive != ElgNone)
                    intermediate.setOutputPrimitive(primitive);
            }
            break;
        }
        case EatPartitioning:
        {
            // Handle [partitioning("...")]
            TString partitionStr;
            if (! it->getString(partitionStr)) {
                error(loc, "invalid partitioning", "", "");
            } else {
                TVertexSpacing partitioning = EvsNone;
                
                if (partitionStr == "integer") {
                    partitioning = EvsEqual;
                } else if (partitionStr == "fractional_even") {
                    partitioning = EvsFractionalEven;
                } else if (partitionStr == "fractional_odd") {
                    partitioning = EvsFractionalOdd;
                    //} else if (partition == "pow2") { // TODO: currently nothing to map this to.
                } else {
                    error(loc, "unsupported partitioning type", partitionStr.c_str(), "");
                }

                if (! intermediate.setVertexSpacing(partitioning))
                    error(loc, "cannot change previously set partitioning",
                          TQualifier::getVertexSpacingString(partitioning), "");
            }
            break;
        }
        case EatOutputControlPoints:
        {
            // Handle [outputcontrolpoints("...")]
            int ctrlPoints;
            if (! it->getInt(ctrlPoints)) {
                error(loc, "invalid outputcontrolpoints", "", "");
            } else {
                if (! intermediate.setVertices(ctrlPoints)) {
                    error(loc, "cannot change previously set outputcontrolpoints attribute", "", "");
                }
            }
            break;
        }
        case EatEarlyDepthStencil:
            intermediate.setEarlyFragmentTests();
            break;
        case EatBuiltIn:
        case EatLocation:
            // tolerate these because of dual use of entrypoint and type attributes
            break;
        default:
            warn(loc, "attribute does not apply to entry point", "", "");
            break;
        }
    }
}

// Update the given type with any type-like attribute information in the
// attributes.
void HlslParseContext::transferTypeAttributes(const TSourceLoc& loc, const TAttributes& attributes, TType& type,
    bool allowEntry)
{
    if (attributes.size() == 0)
        return;

    int value;
    TString builtInString;
    for (auto it = attributes.begin(); it != attributes.end(); ++it) {
        switch (it->name) {
        case EatLocation:
            // location
            if (it->getInt(value))
                type.getQualifier().layoutLocation = value;
            break;
        case EatBinding:
            // binding
            if (it->getInt(value)) {
                type.getQualifier().layoutBinding = value;
                type.getQualifier().layoutSet = 0;
            }
            // set
            if (it->getInt(value, 1))
                type.getQualifier().layoutSet = value;
            break;
        case EatGlobalBinding:
            // global cbuffer binding
            if (it->getInt(value))
                globalUniformBinding = value;
            // global cbuffer binding
            if (it->getInt(value, 1))
                globalUniformSet = value;
            break;
        case EatInputAttachment:
            // input attachment
            if (it->getInt(value))
                type.getQualifier().layoutAttachment = value;
            break;
        case EatBuiltIn:
            // PointSize built-in
            if (it->getString(builtInString, 0, false)) {
                if (builtInString == "PointSize")
                    type.getQualifier().builtIn = EbvPointSize;
            }
            break;
        case EatPushConstant:
            // push_constant
            type.getQualifier().layoutPushConstant = true;
            break;
        case EatConstantId:
            // specialization constant
            if (it->getInt(value)) {
                TSourceLoc loc;
                loc.init();
                setSpecConstantId(loc, type.getQualifier(), value);
            }
            break;
        default:
            if (! allowEntry)
                warn(loc, "attribute does not apply to a type", "", "");
            break;
        }
    }
}

//
// Do all special handling for the entry point, including wrapping
// the shader's entry point with the official entry point that will call it.
//
// The following:
//
//    retType shaderEntryPoint(args...) // shader declared entry point
//    { body }
//
// Becomes
//
//    out retType ret;
//    in iargs<that are input>...;
//    out oargs<that are output> ...;
//
//    void shaderEntryPoint()    // synthesized, but official, entry point
//    {
//        args<that are input> = iargs...;
//        ret = @shaderEntryPoint(args...);
//        oargs = args<that are output>...;
//    }
//    retType @shaderEntryPoint(args...)
//    { body }
//
// The symbol table will still map the original entry point name to the
// the modified function and its new name:
//
//    symbol table:  shaderEntryPoint  ->   @shaderEntryPoint
//
// Returns nullptr if no entry-point tree was built, otherwise, returns
// a subtree that creates the entry point.
//
TIntermNode* HlslParseContext::transformEntryPoint(const TSourceLoc& loc, TFunction& userFunction,
                                                   const TAttributes& attributes)
{
    // Return true if this is a tessellation patch constant function input to a domain shader.
    const auto isDsPcfInput = [this](const TType& type) {
        return language == EShLangTessEvaluation &&
        type.contains([](const TType* t) {
                return t->getQualifier().builtIn == EbvTessLevelOuter ||
                       t->getQualifier().builtIn == EbvTessLevelInner;
            });
    };

    // if we aren't in the entry point, fix the IO as such and exit
    if (userFunction.getName().compare(intermediate.getEntryPointName().c_str()) != 0) {
        remapNonEntryPointIO(userFunction);
        return nullptr;
    }

    entryPointFunction = &userFunction; // needed in finish()

    // Handle entry point attributes
    handleEntryPointAttributes(loc, attributes);

    // entry point logic...

    // Move parameters and return value to shader in/out
    TVariable* entryPointOutput; // gets created in remapEntryPointIO
    TVector<TVariable*> inputs;
    TVector<TVariable*> outputs;
    remapEntryPointIO(userFunction, entryPointOutput, inputs, outputs);

    // Further this return/in/out transform by flattening, splitting, and assigning locations
    const auto makeVariableInOut = [&](TVariable& variable) {
        if (variable.getType().isStruct()) {
            if (variable.getType().getQualifier().isArrayedIo(language)) {
                if (variable.getType().containsBuiltIn())
                    split(variable);
            } else if (shouldFlatten(variable.getType(), EvqVaryingIn /* not assigned yet, but close enough */, true))
                flatten(variable, false /* don't track linkage here, it will be tracked in assignToInterface() */);
        }
        // TODO: flatten arrays too
        // TODO: flatten everything in I/O
        // TODO: replace all split with flatten, make all paths can create flattened I/O, then split code can be removed

        // For clip and cull distance, multiple output variables potentially get merged
        // into one in assignClipCullDistance.  That code in assignClipCullDistance
        // handles the interface logic, so we avoid it here in that case.
        if (!isClipOrCullDistance(variable.getType()))
            assignToInterface(variable);
    };
    if (entryPointOutput != nullptr)
        makeVariableInOut(*entryPointOutput);
    for (auto it = inputs.begin(); it != inputs.end(); ++it)
        if (!isDsPcfInput((*it)->getType()))  // wait until the end for PCF input (see comment below)
            makeVariableInOut(*(*it));
    for (auto it = outputs.begin(); it != outputs.end(); ++it)
        makeVariableInOut(*(*it));

    // In the domain shader, PCF input must be at the end of the linkage.  That's because in the
    // hull shader there is no ordering: the output comes from the separate PCF, which does not
    // participate in the argument list.  That is always put at the end of the HS linkage, so the
    // input side of the DS must match.  The argument may be in any position in the DS argument list
    // however, so this ensures the linkage is built in the correct order regardless of argument order.
    if (language == EShLangTessEvaluation) {
        for (auto it = inputs.begin(); it != inputs.end(); ++it)
            if (isDsPcfInput((*it)->getType()))
                makeVariableInOut(*(*it));
    }

    // Synthesize the call

    pushScope(); // matches the one in handleFunctionBody()

    // new signature
    TType voidType(EbtVoid);
    TFunction synthEntryPoint(&userFunction.getName(), voidType);
    TIntermAggregate* synthParams = new TIntermAggregate();
    intermediate.setAggregateOperator(synthParams, EOpParameters, voidType, loc);
    intermediate.setEntryPointMangledName(synthEntryPoint.getMangledName().c_str());
    intermediate.incrementEntryPointCount();
    TFunction callee(&userFunction.getName(), voidType); // call based on old name, which is still in the symbol table

    // change original name
    userFunction.addPrefix("@");                         // change the name in the function, but not in the symbol table

    // Copy inputs (shader-in -> calling arg), while building up the call node
    TVector<TVariable*> argVars;
    TIntermAggregate* synthBody = new TIntermAggregate();
    auto inputIt = inputs.begin();
    TIntermTyped* callingArgs = nullptr;

    for (int i = 0; i < userFunction.getParamCount(); i++) {
        TParameter& param = userFunction[i];
        argVars.push_back(makeInternalVariable(*param.name, *param.type));
        argVars.back()->getWritableType().getQualifier().makeTemporary();

        // Track the input patch, which is the only non-builtin supported by hull shader PCF.
        if (param.getDeclaredBuiltIn() == EbvInputPatch)
            inputPatch = argVars.back();

        TIntermSymbol* arg = intermediate.addSymbol(*argVars.back());
        handleFunctionArgument(&callee, callingArgs, arg);
        if (param.type->getQualifier().isParamInput()) {
            intermediate.growAggregate(synthBody, handleAssign(loc, EOpAssign, arg,
                                                               intermediate.addSymbol(**inputIt)));
            inputIt++;
        }
    }

    // Call
    currentCaller = synthEntryPoint.getMangledName();
    TIntermTyped* callReturn = handleFunctionCall(loc, &callee, callingArgs);
    currentCaller = userFunction.getMangledName();

    // Return value
    if (entryPointOutput) {
        TIntermTyped* returnAssign;

        // For hull shaders, the wrapped entry point return value is written to
        // an array element as indexed by invocation ID, which we might have to make up.
        // This is required to match SPIR-V semantics.
        if (language == EShLangTessControl) {
            TIntermSymbol* invocationIdSym = findTessLinkageSymbol(EbvInvocationId);

            // If there is no user declared invocation ID, we must make one.
            if (invocationIdSym == nullptr) {
                TType invocationIdType(EbtUint, EvqIn, 1);
                TString* invocationIdName = NewPoolTString("InvocationId");
                invocationIdType.getQualifier().builtIn = EbvInvocationId;

                TVariable* variable = makeInternalVariable(*invocationIdName, invocationIdType);

                globalQualifierFix(loc, variable->getWritableType().getQualifier());
                trackLinkage(*variable);

                invocationIdSym = intermediate.addSymbol(*variable);
            }

            TIntermTyped* element = intermediate.addIndex(EOpIndexIndirect, intermediate.addSymbol(*entryPointOutput),
                                                          invocationIdSym, loc);

            // Set the type of the array element being dereferenced
            const TType derefElementType(entryPointOutput->getType(), 0);
            element->setType(derefElementType);

            returnAssign = handleAssign(loc, EOpAssign, element, callReturn);
        } else {
            returnAssign = handleAssign(loc, EOpAssign, intermediate.addSymbol(*entryPointOutput), callReturn);
        }
        intermediate.growAggregate(synthBody, returnAssign);
    } else
        intermediate.growAggregate(synthBody, callReturn);

    // Output copies
    auto outputIt = outputs.begin();
    for (int i = 0; i < userFunction.getParamCount(); i++) {
        TParameter& param = userFunction[i];

        // GS outputs are via emit, so we do not copy them here.
        if (param.type->getQualifier().isParamOutput()) {
            if (param.getDeclaredBuiltIn() == EbvGsOutputStream) {
                // GS output stream does not assign outputs here: it's the Append() method
                // which writes to the output, probably multiple times separated by Emit.
                // We merely remember the output to use, here.
                gsStreamOutput = *outputIt;
            } else {
                intermediate.growAggregate(synthBody, handleAssign(loc, EOpAssign,
                                                                   intermediate.addSymbol(**outputIt),
                                                                   intermediate.addSymbol(*argVars[i])));
            }

            outputIt++;
        }
    }

    // Put the pieces together to form a full function subtree
    // for the synthesized entry point.
    synthBody->setOperator(EOpSequence);
    TIntermNode* synthFunctionDef = synthParams;
    handleFunctionBody(loc, synthEntryPoint, synthBody, synthFunctionDef);

    entryPointFunctionBody = synthBody;

    return synthFunctionDef;
}

void HlslParseContext::handleFunctionBody(const TSourceLoc& loc, TFunction& function, TIntermNode* functionBody,
                                          TIntermNode*& node)
{
    node = intermediate.growAggregate(node, functionBody);
    intermediate.setAggregateOperator(node, EOpFunction, function.getType(), loc);
    node->getAsAggregate()->setName(function.getMangledName().c_str());

    popScope();
    if (function.hasImplicitThis())
        popImplicitThis();

    if (function.getType().getBasicType() != EbtVoid && ! functionReturnsValue)
        error(loc, "function does not return a value:", "", function.getName().c_str());
}

// AST I/O is done through shader globals declared in the 'in' or 'out'
// storage class.  An HLSL entry point has a return value, input parameters
// and output parameters.  These need to get remapped to the AST I/O.
void HlslParseContext::remapEntryPointIO(TFunction& function, TVariable*& returnValue,
    TVector<TVariable*>& inputs, TVector<TVariable*>& outputs)
{
    // We might have in input structure type with no decorations that caused it
    // to look like an input type, yet it has (e.g.) interpolation types that
    // must be modified that turn it into an input type.
    // Hence, a missing ioTypeMap for 'input' might need to be synthesized.
    const auto synthesizeEditedInput = [this](TType& type) {
        // True if a type needs to be 'flat'
        const auto needsFlat = [](const TType& type) {
            return type.containsBasicType(EbtInt) ||
                    type.containsBasicType(EbtUint) ||
                    type.containsBasicType(EbtInt64) ||
                    type.containsBasicType(EbtUint64) ||
                    type.containsBasicType(EbtBool) ||
                    type.containsBasicType(EbtDouble);
        };

        if (language == EShLangFragment && needsFlat(type)) {
            if (type.isStruct()) {
                TTypeList* finalList = nullptr;
                auto it = ioTypeMap.find(type.getStruct());
                if (it == ioTypeMap.end() || it->second.input == nullptr) {
                    // Getting here means we have no input struct, but we need one.
                    auto list = new TTypeList;
                    for (auto member = type.getStruct()->begin(); member != type.getStruct()->end(); ++member) {
                        TType* newType = new TType;
                        newType->shallowCopy(*member->type);
                        TTypeLoc typeLoc = { newType, member->loc };
                        list->push_back(typeLoc);
                    }
                    // install the new input type
                    if (it == ioTypeMap.end()) {
                        tIoKinds newLists = { list, nullptr, nullptr };
                        ioTypeMap[type.getStruct()] = newLists;
                    } else
                        it->second.input = list;
                    finalList = list;
                } else
                    finalList = it->second.input;
                // edit for 'flat'
                for (auto member = finalList->begin(); member != finalList->end(); ++member) {
                    if (needsFlat(*member->type)) {
                        member->type->getQualifier().clearInterpolation();
                        member->type->getQualifier().flat = true;
                    }
                }
            } else {
                type.getQualifier().clearInterpolation();
                type.getQualifier().flat = true;
            }
        }
    };

    // Do the actual work to make a type be a shader input or output variable,
    // and clear the original to be non-IO (for use as a normal function parameter/return).
    const auto makeIoVariable = [this](const char* name, TType& type, TStorageQualifier storage) -> TVariable* {
        TVariable* ioVariable = makeInternalVariable(name, type);
        clearUniformInputOutput(type.getQualifier());
        if (type.isStruct()) {
            auto newLists = ioTypeMap.find(ioVariable->getType().getStruct());
            if (newLists != ioTypeMap.end()) {
                if (storage == EvqVaryingIn && newLists->second.input)
                    ioVariable->getWritableType().setStruct(newLists->second.input);
                else if (storage == EvqVaryingOut && newLists->second.output)
                    ioVariable->getWritableType().setStruct(newLists->second.output);
            }
        }
        if (storage == EvqVaryingIn) {
            correctInput(ioVariable->getWritableType().getQualifier());
            if (language == EShLangTessEvaluation)
                if (!ioVariable->getType().isArray())
                    ioVariable->getWritableType().getQualifier().patch = true;
        } else {
            correctOutput(ioVariable->getWritableType().getQualifier());
        }
        ioVariable->getWritableType().getQualifier().storage = storage;

        fixBuiltInIoType(ioVariable->getWritableType());

        return ioVariable;
    };

    // return value is actually a shader-scoped output (out)
    if (function.getType().getBasicType() == EbtVoid) {
        returnValue = nullptr;
    } else {
        if (language == EShLangTessControl) {
            // tessellation evaluation in HLSL writes a per-ctrl-pt value, but it needs to be an
            // array in SPIR-V semantics.  We'll write to it indexed by invocation ID.

            returnValue = makeIoVariable("@entryPointOutput", function.getWritableType(), EvqVaryingOut);

            TType outputType;
            outputType.shallowCopy(function.getType());

            // vertices has necessarily already been set when handling entry point attributes.
            TArraySizes* arraySizes = new TArraySizes;
            arraySizes->addInnerSize(intermediate.getVertices());
            outputType.transferArraySizes(arraySizes);

            clearUniformInputOutput(function.getWritableType().getQualifier());
            returnValue = makeIoVariable("@entryPointOutput", outputType, EvqVaryingOut);
        } else {
            returnValue = makeIoVariable("@entryPointOutput", function.getWritableType(), EvqVaryingOut);
        }
    }

    // parameters are actually shader-scoped inputs and outputs (in or out)
    for (int i = 0; i < function.getParamCount(); i++) {
        TType& paramType = *function[i].type;
        if (paramType.getQualifier().isParamInput()) {
            synthesizeEditedInput(paramType);
            TVariable* argAsGlobal = makeIoVariable(function[i].name->c_str(), paramType, EvqVaryingIn);
            inputs.push_back(argAsGlobal);
        }
        if (paramType.getQualifier().isParamOutput()) {
            TVariable* argAsGlobal = makeIoVariable(function[i].name->c_str(), paramType, EvqVaryingOut);
            outputs.push_back(argAsGlobal);
        }
    }
}

// An HLSL function that looks like an entry point, but is not,
// declares entry point IO built-ins, but these have to be undone.
void HlslParseContext::remapNonEntryPointIO(TFunction& function)
{
    // return value
    if (function.getType().getBasicType() != EbtVoid)
        clearUniformInputOutput(function.getWritableType().getQualifier());

    // parameters.
    // References to structuredbuffer types are left unmodified
    for (int i = 0; i < function.getParamCount(); i++)
        if (!isReference(*function[i].type))
            clearUniformInputOutput(function[i].type->getQualifier());
}

// Handle function returns, including type conversions to the function return type
// if necessary.
TIntermNode* HlslParseContext::handleReturnValue(const TSourceLoc& loc, TIntermTyped* value)
{
    functionReturnsValue = true;

    if (currentFunctionType->getBasicType() == EbtVoid) {
        error(loc, "void function cannot return a value", "return", "");
        return intermediate.addBranch(EOpReturn, loc);
    } else if (*currentFunctionType != value->getType()) {
        value = intermediate.addConversion(EOpReturn, *currentFunctionType, value);
        if (value && *currentFunctionType != value->getType())
            value = intermediate.addUniShapeConversion(EOpReturn, *currentFunctionType, value);
        if (value == nullptr || *currentFunctionType != value->getType()) {
            error(loc, "type does not match, or is not convertible to, the function's return type", "return", "");
            return value;
        }
    }

    return intermediate.addBranch(EOpReturn, value, loc);
}

void HlslParseContext::handleFunctionArgument(TFunction* function,
                                              TIntermTyped*& arguments, TIntermTyped* newArg)
{
    TParameter param = { 0, new TType, nullptr };
    param.type->shallowCopy(newArg->getType());

    function->addParameter(param);
    if (arguments)
        arguments = intermediate.growAggregate(arguments, newArg);
    else
        arguments = newArg;
}

// Position may require special handling: we can optionally invert Y.
// See: https://github.com/KhronosGroup/glslang/issues/1173
//      https://github.com/KhronosGroup/glslang/issues/494
TIntermTyped* HlslParseContext::assignPosition(const TSourceLoc& loc, TOperator op,
                                               TIntermTyped* left, TIntermTyped* right)
{
    // If we are not asked for Y inversion, use a plain old assign.
    if (!intermediate.getInvertY())
        return intermediate.addAssign(op, left, right, loc);

    // If we get here, we should invert Y.
    TIntermAggregate* assignList = nullptr;

    // If this is a complex rvalue, we don't want to dereference it many times.  Create a temporary.
    TVariable* rhsTempVar = nullptr;
    rhsTempVar = makeInternalVariable("@position", right->getType());
    rhsTempVar->getWritableType().getQualifier().makeTemporary();

    {
        TIntermTyped* rhsTempSym = intermediate.addSymbol(*rhsTempVar, loc);
        assignList = intermediate.growAggregate(assignList,
                                                intermediate.addAssign(EOpAssign, rhsTempSym, right, loc), loc);
    }

    // pos.y = -pos.y
    {
        const int Y = 1;

        TIntermTyped* tempSymL = intermediate.addSymbol(*rhsTempVar, loc);
        TIntermTyped* tempSymR = intermediate.addSymbol(*rhsTempVar, loc);
        TIntermTyped* index = intermediate.addConstantUnion(Y, loc);

        TIntermTyped* lhsElement = intermediate.addIndex(EOpIndexDirect, tempSymL, index, loc);
        TIntermTyped* rhsElement = intermediate.addIndex(EOpIndexDirect, tempSymR, index, loc);

        const TType derefType(right->getType(), 0);
    
        lhsElement->setType(derefType);
        rhsElement->setType(derefType);

        TIntermTyped* yNeg = intermediate.addUnaryMath(EOpNegative, rhsElement, loc);

        assignList = intermediate.growAggregate(assignList, intermediate.addAssign(EOpAssign, lhsElement, yNeg, loc));
    }

    // Assign the rhs temp (now with Y inversion) to the final output
    {
        TIntermTyped* rhsTempSym = intermediate.addSymbol(*rhsTempVar, loc);
        assignList = intermediate.growAggregate(assignList, intermediate.addAssign(op, left, rhsTempSym, loc));
    }

    assert(assignList != nullptr);
    assignList->setOperator(EOpSequence);

    return assignList;
}
    
// Clip and cull distance require special handling due to a semantic mismatch.  In HLSL,
// these can be float scalar, float vector, or arrays of float scalar or float vector.
// In SPIR-V, they are arrays of scalar floats in all cases.  We must copy individual components
// (e.g, both x and y components of a float2) out into the destination float array.
//
// The values are assigned to sequential members of the output array.  The inner dimension
// is vector components.  The outer dimension is array elements.
TIntermAggregate* HlslParseContext::assignClipCullDistance(const TSourceLoc& loc, TOperator op, int semanticId,
                                                           TIntermTyped* left, TIntermTyped* right)
{
    switch (language) {
    case EShLangFragment:
    case EShLangVertex:
    case EShLangGeometry:
        break;
    default:
        error(loc, "unimplemented: clip/cull not currently implemented for this stage", "", "");
        return nullptr;
    }

    TVariable** clipCullVar = nullptr;

    // Figure out if we are assigning to, or from, clip or cull distance.
    const bool isOutput = isClipOrCullDistance(left->getType());

    // This is the rvalue or lvalue holding the clip or cull distance.
    TIntermTyped* clipCullNode = isOutput ? left : right;
    // This is the value going into or out of the clip or cull distance.
    TIntermTyped* internalNode = isOutput ? right : left;

    const TBuiltInVariable builtInType = clipCullNode->getQualifier().builtIn;

    decltype(clipSemanticNSizeIn)* semanticNSize = nullptr;

    // Refer to either the clip or the cull distance, depending on semantic.
    switch (builtInType) {
    case EbvClipDistance:
        clipCullVar = isOutput ? &clipDistanceOutput : &clipDistanceInput;
        semanticNSize = isOutput ? &clipSemanticNSizeOut : &clipSemanticNSizeIn;
        break;
    case EbvCullDistance:
        clipCullVar = isOutput ? &cullDistanceOutput : &cullDistanceInput;
        semanticNSize = isOutput ? &cullSemanticNSizeOut : &cullSemanticNSizeIn;
        break;

    // called invalidly: we expected a clip or a cull distance.
    // static compile time problem: should not happen.
    default: assert(0); return nullptr;
    }

    // This is the offset in the destination array of a given semantic's data
    std::array<int, maxClipCullRegs> semanticOffset;

    // Calculate offset of variable of semantic N in destination array
    int arrayLoc = 0;
    int vecItems = 0;

    for (int x = 0; x < maxClipCullRegs; ++x) {
        // See if we overflowed the vec4 packing
        if ((vecItems + (*semanticNSize)[x]) > 4) {
            arrayLoc = (arrayLoc + 3) & (~0x3); // round up to next multiple of 4
            vecItems = 0;
        }

        semanticOffset[x] = arrayLoc;
        vecItems += (*semanticNSize)[x];
        arrayLoc += (*semanticNSize)[x];
    }
 

    // It can have up to 2 array dimensions (in the case of geometry shader inputs)
    const TArraySizes* const internalArraySizes = internalNode->getType().getArraySizes();
    const int internalArrayDims = internalNode->getType().isArray() ? internalArraySizes->getNumDims() : 0;
    // vector sizes:
    const int internalVectorSize = internalNode->getType().getVectorSize();
    // array sizes, or 1 if it's not an array:
    const int internalInnerArraySize = (internalArrayDims > 0 ? internalArraySizes->getDimSize(internalArrayDims-1) : 1);
    const int internalOuterArraySize = (internalArrayDims > 1 ? internalArraySizes->getDimSize(0) : 1);

    // The created type may be an array of arrays, e.g, for geometry shader inputs.
    const bool isImplicitlyArrayed = (language == EShLangGeometry && !isOutput);

    // If we haven't created the output already, create it now.
    if (*clipCullVar == nullptr) {
        // ClipDistance and CullDistance are handled specially in the entry point input/output copy
        // algorithm, because they may need to be unpacked from components of vectors (or a scalar)
        // into a float array, or vice versa.  Here, we make the array the right size and type,
        // which depends on the incoming data, which has several potential dimensions:
        //    * Semantic ID
        //    * vector size 
        //    * array size
        // Of those, semantic ID and array size cannot appear simultaneously.
        //
        // Also to note: for implicitly arrayed forms (e.g, geometry shader inputs), we need to create two
        // array dimensions.  The shader's declaration may have one or two array dimensions.  One is always
        // the geometry's dimension.

        const bool useInnerSize = internalArrayDims > 1 || !isImplicitlyArrayed;

        const int requiredInnerArraySize = arrayLoc * (useInnerSize ? internalInnerArraySize : 1);
        const int requiredOuterArraySize = (internalArrayDims > 0) ? internalArraySizes->getDimSize(0) : 1;

        TType clipCullType(EbtFloat, clipCullNode->getType().getQualifier().storage, 1);
        clipCullType.getQualifier() = clipCullNode->getType().getQualifier();

        // Create required array dimension
        TArraySizes* arraySizes = new TArraySizes;
        if (isImplicitlyArrayed)
            arraySizes->addInnerSize(requiredOuterArraySize);
        arraySizes->addInnerSize(requiredInnerArraySize);
        clipCullType.transferArraySizes(arraySizes);

        // Obtain symbol name: we'll use that for the symbol we introduce.
        TIntermSymbol* sym = clipCullNode->getAsSymbolNode();
        assert(sym != nullptr);

        // We are moving the semantic ID from the layout location, so it is no longer needed or
        // desired there.
        clipCullType.getQualifier().layoutLocation = TQualifier::layoutLocationEnd;

        // Create variable and track its linkage
        *clipCullVar = makeInternalVariable(sym->getName().c_str(), clipCullType);

        trackLinkage(**clipCullVar);
    }

    // Create symbol for the clip or cull variable.
    TIntermSymbol* clipCullSym = intermediate.addSymbol(**clipCullVar);

    // vector sizes:
    const int clipCullVectorSize = clipCullSym->getType().getVectorSize();

    // array sizes, or 1 if it's not an array:
    const TArraySizes* const clipCullArraySizes = clipCullSym->getType().getArraySizes();
    const int clipCullOuterArraySize = isImplicitlyArrayed ? clipCullArraySizes->getDimSize(0) : 1;
    const int clipCullInnerArraySize = clipCullArraySizes->getDimSize(isImplicitlyArrayed ? 1 : 0);

    // clipCullSym has got to be an array of scalar floats, per SPIR-V semantics.
    // fixBuiltInIoType() should have handled that upstream.
    assert(clipCullSym->getType().isArray());
    assert(clipCullSym->getType().getVectorSize() == 1);
    assert(clipCullSym->getType().getBasicType() == EbtFloat);

    // We may be creating multiple sub-assignments.  This is an aggregate to hold them.
    // TODO: it would be possible to be clever sometimes and avoid the sequence node if not needed.
    TIntermAggregate* assignList = nullptr;

    // Holds individual component assignments as we make them.
    TIntermTyped* clipCullAssign = nullptr;

    // If the types are homomorphic, use a simple assign.  No need to mess about with 
    // individual components.
    if (clipCullSym->getType().isArray() == internalNode->getType().isArray() &&
        clipCullInnerArraySize == internalInnerArraySize &&
        clipCullOuterArraySize == internalOuterArraySize &&
        clipCullVectorSize == internalVectorSize) {

        if (isOutput)
            clipCullAssign = intermediate.addAssign(op, clipCullSym, internalNode, loc);
        else
            clipCullAssign = intermediate.addAssign(op, internalNode, clipCullSym, loc);

        assignList = intermediate.growAggregate(assignList, clipCullAssign);
        assignList->setOperator(EOpSequence);

        return assignList;
    }

    // We are going to copy each component of the internal (per array element if indicated) to sequential
    // array elements of the clipCullSym.  This tracks the lhs element we're writing to as we go along.
    // We may be starting in the middle - e.g, for a non-zero semantic ID calculated above.
    int clipCullInnerArrayPos = semanticOffset[semanticId];
    int clipCullOuterArrayPos = 0;

    // Lambda to add an index to a node, set the type of the result, and return the new node.
    const auto addIndex = [this, &loc](TIntermTyped* node, int pos) -> TIntermTyped* {
        const TType derefType(node->getType(), 0);
        node = intermediate.addIndex(EOpIndexDirect, node, intermediate.addConstantUnion(pos, loc), loc);
        node->setType(derefType);
        return node;
    };

    // Loop through every component of every element of the internal, and copy to or from the matching external.
    for (int internalOuterArrayPos = 0; internalOuterArrayPos < internalOuterArraySize; ++internalOuterArrayPos) {
        for (int internalInnerArrayPos = 0; internalInnerArrayPos < internalInnerArraySize; ++internalInnerArrayPos) {
            for (int internalComponent = 0; internalComponent < internalVectorSize; ++internalComponent) {
                // clip/cull array member to read from / write to:
                TIntermTyped* clipCullMember = clipCullSym;

                // If implicitly arrayed, there is an outer array dimension involved
                if (isImplicitlyArrayed)
                    clipCullMember = addIndex(clipCullMember, clipCullOuterArrayPos);

                // Index into proper array position for clip cull member
                clipCullMember = addIndex(clipCullMember, clipCullInnerArrayPos++);

                // if needed, start over with next outer array slice.
                if (isImplicitlyArrayed && clipCullInnerArrayPos >= clipCullInnerArraySize) {
                    clipCullInnerArrayPos = semanticOffset[semanticId];
                    ++clipCullOuterArrayPos;
                }

                // internal member to read from / write to:
                TIntermTyped* internalMember = internalNode;

                // If internal node has outer array dimension, index appropriately.
                if (internalArrayDims > 1)
                    internalMember = addIndex(internalMember, internalOuterArrayPos);

                // If internal node has inner array dimension, index appropriately.
                if (internalArrayDims > 0)
                    internalMember = addIndex(internalMember, internalInnerArrayPos);

                // If internal node is a vector, extract the component of interest.
                if (internalNode->getType().isVector())
                    internalMember = addIndex(internalMember, internalComponent);

                // Create an assignment: output from internal to clip cull, or input from clip cull to internal.
                if (isOutput)
                    clipCullAssign = intermediate.addAssign(op, clipCullMember, internalMember, loc);
                else
                    clipCullAssign = intermediate.addAssign(op, internalMember, clipCullMember, loc);

                // Track assignment in the sequence.
                assignList = intermediate.growAggregate(assignList, clipCullAssign);
            }
        }
    }

    assert(assignList != nullptr);
    assignList->setOperator(EOpSequence);

    return assignList;
}

// Some simple source assignments need to be flattened to a sequence
// of AST assignments. Catch these and flatten, otherwise, pass through
// to intermediate.addAssign().
//
// Also, assignment to matrix swizzles requires multiple component assignments,
// intercept those as well.
TIntermTyped* HlslParseContext::handleAssign(const TSourceLoc& loc, TOperator op, TIntermTyped* left,
                                             TIntermTyped* right)
{
    if (left == nullptr || right == nullptr)
        return nullptr;

    // writing to opaques will require fixing transforms
    if (left->getType().containsOpaque())
        intermediate.setNeedsLegalization();

    if (left->getAsOperator() && left->getAsOperator()->getOp() == EOpMatrixSwizzle)
        return handleAssignToMatrixSwizzle(loc, op, left, right);

    // Return true if the given node is an index operation into a split variable.
    const auto indexesSplit = [this](const TIntermTyped* node) -> bool {
        const TIntermBinary* binaryNode = node->getAsBinaryNode();

        if (binaryNode == nullptr)
            return false;

        return (binaryNode->getOp() == EOpIndexDirect || binaryNode->getOp() == EOpIndexIndirect) && 
               wasSplit(binaryNode->getLeft());
    };

    // Return true if this stage assigns clip position with potentially inverted Y
    const auto assignsClipPos = [this](const TIntermTyped* node) -> bool {
        return node->getType().getQualifier().builtIn == EbvPosition &&
               (language == EShLangVertex || language == EShLangGeometry || language == EShLangTessEvaluation);
    };

    const bool isSplitLeft    = wasSplit(left) || indexesSplit(left);
    const bool isSplitRight   = wasSplit(right) || indexesSplit(right);

    const bool isFlattenLeft  = wasFlattened(left);
    const bool isFlattenRight = wasFlattened(right);

    // OK to do a single assign if neither side is split or flattened.  Otherwise, 
    // fall through to a member-wise copy.
    if (!isFlattenLeft && !isFlattenRight && !isSplitLeft && !isSplitRight) {
        // Clip and cull distance requires more processing.  See comment above assignClipCullDistance.
        if (isClipOrCullDistance(left->getType()) || isClipOrCullDistance(right->getType())) {
            const bool isOutput = isClipOrCullDistance(left->getType());

            const int semanticId = (isOutput ? left : right)->getType().getQualifier().layoutLocation;
            return assignClipCullDistance(loc, op, semanticId, left, right);
        } else if (assignsClipPos(left)) {
            // Position can require special handling: see comment above assignPosition
            return assignPosition(loc, op, left, right);
        } else if (left->getQualifier().builtIn == EbvSampleMask) {
            // Certain builtins are required to be arrayed outputs in SPIR-V, but may internally be scalars
            // in the shader.  Copy the scalar RHS into the LHS array element zero, if that happens.
            if (left->isArray() && !right->isArray()) {
                const TType derefType(left->getType(), 0);
                left = intermediate.addIndex(EOpIndexDirect, left, intermediate.addConstantUnion(0, loc), loc);
                left->setType(derefType);
                // Fall through to add assign.
            }
        }

        return intermediate.addAssign(op, left, right, loc);
    }

    TIntermAggregate* assignList = nullptr;
    const TVector<TVariable*>* leftVariables = nullptr;
    const TVector<TVariable*>* rightVariables = nullptr;

    // A temporary to store the right node's value, so we don't keep indirecting into it
    // if it's not a simple symbol.
    TVariable* rhsTempVar = nullptr;

    // If the RHS is a simple symbol node, we'll copy it for each member.
    TIntermSymbol* cloneSymNode = nullptr;

    int memberCount = 0;

    // Track how many items there are to copy.
    if (left->getType().isStruct())
        memberCount = (int)left->getType().getStruct()->size();
    if (left->getType().isArray())
        memberCount = left->getType().getCumulativeArraySize();

    if (isFlattenLeft)
        leftVariables = &flattenMap.find(left->getAsSymbolNode()->getId())->second.members;

    if (isFlattenRight) {
        rightVariables = &flattenMap.find(right->getAsSymbolNode()->getId())->second.members;
    } else {
        // The RHS is not flattened.  There are several cases:
        // 1. 1 item to copy:  Use the RHS directly.
        // 2. >1 item, simple symbol RHS: we'll create a new TIntermSymbol node for each, but no assign to temp.
        // 3. >1 item, complex RHS: assign it to a new temp variable, and create a TIntermSymbol for each member.

        if (memberCount <= 1) {
            // case 1: we'll use the symbol directly below.  Nothing to do.
        } else {
            if (right->getAsSymbolNode() != nullptr) {
                // case 2: we'll copy the symbol per iteration below.
                cloneSymNode = right->getAsSymbolNode();
            } else {
                // case 3: assign to a temp, and indirect into that.
                rhsTempVar = makeInternalVariable("flattenTemp", right->getType());
                rhsTempVar->getWritableType().getQualifier().makeTemporary();
                TIntermTyped* noFlattenRHS = intermediate.addSymbol(*rhsTempVar, loc);

                // Add this to the aggregate being built.
                assignList = intermediate.growAggregate(assignList,
                                                        intermediate.addAssign(op, noFlattenRHS, right, loc), loc);
            }
        }
    }

    // When dealing with split arrayed structures of built-ins, the arrayness is moved to the extracted built-in
    // variables, which is awkward when copying between split and unsplit structures.  This variable tracks
    // array indirections so they can be percolated from outer structs to inner variables.
    std::vector <int> arrayElement;

    TStorageQualifier leftStorage = left->getType().getQualifier().storage;
    TStorageQualifier rightStorage = right->getType().getQualifier().storage;

    int leftOffset = findSubtreeOffset(*left);
    int rightOffset = findSubtreeOffset(*right);

    const auto getMember = [&](bool isLeft, const TType& type, int member, TIntermTyped* splitNode, int splitMember,
                               bool flattened)
                           -> TIntermTyped * {
        const bool split     = isLeft ? isSplitLeft   : isSplitRight;

        TIntermTyped* subTree;
        const TType derefType(type, member);
        const TVariable* builtInVar = nullptr;
        if ((flattened || split) && derefType.isBuiltIn()) {
            auto splitPair = splitBuiltIns.find(HlslParseContext::tInterstageIoData(
                                                   derefType.getQualifier().builtIn,
                                                   isLeft ? leftStorage : rightStorage));
            if (splitPair != splitBuiltIns.end())
                builtInVar = splitPair->second;
        }
        if (builtInVar != nullptr) {
            // copy from interstage IO built-in if needed
            subTree = intermediate.addSymbol(*builtInVar);

            if (subTree->getType().isArray()) {
                // Arrayness of builtIn symbols isn't handled by the normal recursion:
                // it's been extracted and moved to the built-in.
                if (!arrayElement.empty()) {
                    const TType splitDerefType(subTree->getType(), arrayElement.back());
                    subTree = intermediate.addIndex(EOpIndexDirect, subTree,
                                                    intermediate.addConstantUnion(arrayElement.back(), loc), loc);
                    subTree->setType(splitDerefType);
                } else if (splitNode->getAsOperator() != nullptr && (splitNode->getAsOperator()->getOp() == EOpIndexIndirect)) {
                    // This might also be a stage with arrayed outputs, in which case there's an index
                    // operation we should transfer to the output builtin.

                    const TType splitDerefType(subTree->getType(), 0);
                    subTree = intermediate.addIndex(splitNode->getAsOperator()->getOp(), subTree,
                                                    splitNode->getAsBinaryNode()->getRight(), loc);
                    subTree->setType(splitDerefType);
                }
            }
        } else if (flattened && !shouldFlatten(derefType, isLeft ? leftStorage : rightStorage, false)) {
            if (isLeft)
                subTree = intermediate.addSymbol(*(*leftVariables)[leftOffset++]);
            else
                subTree = intermediate.addSymbol(*(*rightVariables)[rightOffset++]);
        } else {
            // Index operator if it's an aggregate, else EOpNull
            const TOperator accessOp = type.isArray()  ? EOpIndexDirect
                                     : type.isStruct() ? EOpIndexDirectStruct
                                     : EOpNull;
            if (accessOp == EOpNull) {
                subTree = splitNode;
            } else {
                subTree = intermediate.addIndex(accessOp, splitNode, intermediate.addConstantUnion(splitMember, loc),
                                                loc);
                const TType splitDerefType(splitNode->getType(), splitMember);
                subTree->setType(splitDerefType);
            }
        }

        return subTree;
    };

    // Use the proper RHS node: a new symbol from a TVariable, copy
    // of an TIntermSymbol node, or sometimes the right node directly.
    right = rhsTempVar != nullptr   ? intermediate.addSymbol(*rhsTempVar, loc) :
            cloneSymNode != nullptr ? intermediate.addSymbol(*cloneSymNode) :
            right;

    // Cannot use auto here, because this is recursive, and auto can't work out the type without seeing the
    // whole thing.  So, we'll resort to an explicit type via std::function.
    const std::function<void(TIntermTyped* left, TIntermTyped* right, TIntermTyped* splitLeft, TIntermTyped* splitRight,
                             bool topLevel)>
    traverse = [&](TIntermTyped* left, TIntermTyped* right, TIntermTyped* splitLeft, TIntermTyped* splitRight,
                   bool topLevel) -> void {
        // If we get here, we are assigning to or from a whole array or struct that must be
        // flattened, so have to do member-by-member assignment:

        bool shouldFlattenSubsetLeft = isFlattenLeft && shouldFlatten(left->getType(), leftStorage, topLevel);
        bool shouldFlattenSubsetRight = isFlattenRight && shouldFlatten(right->getType(), rightStorage, topLevel);

        if ((left->getType().isArray() || right->getType().isArray()) &&
              (shouldFlattenSubsetLeft  || isSplitLeft ||
               shouldFlattenSubsetRight || isSplitRight)) {
            const int elementsL = left->getType().isArray()  ? left->getType().getOuterArraySize()  : 1;
            const int elementsR = right->getType().isArray() ? right->getType().getOuterArraySize() : 1;

            // The arrays might not be the same size,
            // e.g., if the size has been forced for EbvTessLevelInner/Outer.
            const int elementsToCopy = std::min(elementsL, elementsR);

            // array case
            for (int element = 0; element < elementsToCopy; ++element) {
                arrayElement.push_back(element);

                // Add a new AST symbol node if we have a temp variable holding a complex RHS.
                TIntermTyped* subLeft  = getMember(true,  left->getType(),  element, left, element,
                                                   shouldFlattenSubsetLeft);
                TIntermTyped* subRight = getMember(false, right->getType(), element, right, element,
                                                   shouldFlattenSubsetRight);

                TIntermTyped* subSplitLeft =  isSplitLeft  ? getMember(true,  left->getType(),  element, splitLeft,
                                                                       element, shouldFlattenSubsetLeft)
                                                           : subLeft;
                TIntermTyped* subSplitRight = isSplitRight ? getMember(false, right->getType(), element, splitRight,
                                                                       element, shouldFlattenSubsetRight)
                                                           : subRight;

                traverse(subLeft, subRight, subSplitLeft, subSplitRight, false);

                arrayElement.pop_back();
            }
        } else if (left->getType().isStruct() && (shouldFlattenSubsetLeft  || isSplitLeft ||
                                                  shouldFlattenSubsetRight || isSplitRight)) {
            // struct case
            const auto& membersL = *left->getType().getStruct();
            const auto& membersR = *right->getType().getStruct();

            // These track the members in the split structures corresponding to the same in the unsplit structures,
            // which we traverse in parallel.
            int memberL = 0;
            int memberR = 0;

            // Handle empty structure assignment
            if (int(membersL.size()) == 0 && int(membersR.size()) == 0)
                assignList = intermediate.growAggregate(assignList, intermediate.addAssign(op, left, right, loc), loc);

            for (int member = 0; member < int(membersL.size()); ++member) {
                const TType& typeL = *membersL[member].type;
                const TType& typeR = *membersR[member].type;

                TIntermTyped* subLeft  = getMember(true,  left->getType(), member, left, member,
                                                   shouldFlattenSubsetLeft);
                TIntermTyped* subRight = getMember(false, right->getType(), member, right, member,
                                                   shouldFlattenSubsetRight);

                // If there is no splitting, use the same values to avoid inefficiency.
                TIntermTyped* subSplitLeft =  isSplitLeft  ? getMember(true,  left->getType(),  member, splitLeft,
                                                                       memberL, shouldFlattenSubsetLeft)
                                                           : subLeft;
                TIntermTyped* subSplitRight = isSplitRight ? getMember(false, right->getType(), member, splitRight,
                                                                       memberR, shouldFlattenSubsetRight)
                                                           : subRight;

                if (isClipOrCullDistance(subSplitLeft->getType()) || isClipOrCullDistance(subSplitRight->getType())) {
                    // Clip and cull distance built-in assignment is complex in its own right, and is handled in
                    // a separate function dedicated to that task.  See comment above assignClipCullDistance;

                    const bool isOutput = isClipOrCullDistance(subSplitLeft->getType());

                    // Since all clip/cull semantics boil down to the same built-in type, we need to get the
                    // semantic ID from the dereferenced type's layout location, to avoid an N-1 mapping.
                    const TType derefType((isOutput ? left : right)->getType(), member);
                    const int semanticId = derefType.getQualifier().layoutLocation;

                    TIntermAggregate* clipCullAssign = assignClipCullDistance(loc, op, semanticId,
                                                                              subSplitLeft, subSplitRight);

                    assignList = intermediate.growAggregate(assignList, clipCullAssign, loc);
                } else if (assignsClipPos(subSplitLeft)) {
                    // Position can require special handling: see comment above assignPosition
                    TIntermTyped* positionAssign = assignPosition(loc, op, subSplitLeft, subSplitRight);
                    assignList = intermediate.growAggregate(assignList, positionAssign, loc);
                } else if (!shouldFlattenSubsetLeft && !shouldFlattenSubsetRight &&
                           !typeL.containsBuiltIn() && !typeR.containsBuiltIn()) {
                    // If this is the final flattening (no nested types below to flatten)
                    // we'll copy the member, else recurse into the type hierarchy.
                    // However, if splitting the struct, that means we can copy a whole
                    // subtree here IFF it does not itself contain any interstage built-in
                    // IO variables, so we only have to recurse into it if there's something
                    // for splitting to do.  That can save a lot of AST verbosity for
                    // a bunch of memberwise copies.

                    assignList = intermediate.growAggregate(assignList,
                                                            intermediate.addAssign(op, subSplitLeft, subSplitRight, loc),
                                                            loc);
                } else {
                    traverse(subLeft, subRight, subSplitLeft, subSplitRight, false);
                }

                memberL += (typeL.isBuiltIn() ? 0 : 1);
                memberR += (typeR.isBuiltIn() ? 0 : 1);
            }
        } else {
            // Member copy
            assignList = intermediate.growAggregate(assignList, intermediate.addAssign(op, left, right, loc), loc);
        }

    };

    TIntermTyped* splitLeft  = left;
    TIntermTyped* splitRight = right;

    // If either left or right was a split structure, we must read or write it, but still have to
    // parallel-recurse through the unsplit structure to identify the built-in IO vars.
    // The left can be either a symbol, or an index into a symbol (e.g, array reference)
    if (isSplitLeft) {
        if (indexesSplit(left)) {
            // Index case: Refer to the indexed symbol, if the left is an index operator.
            const TIntermSymbol* symNode = left->getAsBinaryNode()->getLeft()->getAsSymbolNode();

            TIntermTyped* splitLeftNonIo = intermediate.addSymbol(*getSplitNonIoVar(symNode->getId()), loc);

            splitLeft = intermediate.addIndex(left->getAsBinaryNode()->getOp(), splitLeftNonIo,
                                              left->getAsBinaryNode()->getRight(), loc);

            const TType derefType(splitLeftNonIo->getType(), 0);
            splitLeft->setType(derefType);
        } else {
            // Symbol case: otherwise, if not indexed, we have the symbol directly.
            const TIntermSymbol* symNode = left->getAsSymbolNode();
            splitLeft = intermediate.addSymbol(*getSplitNonIoVar(symNode->getId()), loc);
        }
    }

    if (isSplitRight)
        splitRight = intermediate.addSymbol(*getSplitNonIoVar(right->getAsSymbolNode()->getId()), loc);

    // This makes the whole assignment, recursing through subtypes as needed.
    traverse(left, right, splitLeft, splitRight, true);

    assert(assignList != nullptr);
    assignList->setOperator(EOpSequence);

    return assignList;
}

// An assignment to matrix swizzle must be decomposed into individual assignments.
// These must be selected component-wise from the RHS and stored component-wise
// into the LHS.
TIntermTyped* HlslParseContext::handleAssignToMatrixSwizzle(const TSourceLoc& loc, TOperator op, TIntermTyped* left,
                                                            TIntermTyped* right)
{
    assert(left->getAsOperator() && left->getAsOperator()->getOp() == EOpMatrixSwizzle);

    if (op != EOpAssign)
        error(loc, "only simple assignment to non-simple matrix swizzle is supported", "assign", "");

    // isolate the matrix and swizzle nodes
    TIntermTyped* matrix = left->getAsBinaryNode()->getLeft()->getAsTyped();
    const TIntermSequence& swizzle = left->getAsBinaryNode()->getRight()->getAsAggregate()->getSequence();

    // if the RHS isn't already a simple vector, let's store into one
    TIntermSymbol* vector = right->getAsSymbolNode();
    TIntermTyped* vectorAssign = nullptr;
    if (vector == nullptr) {
        // create a new intermediate vector variable to assign to
        TType vectorType(matrix->getBasicType(), EvqTemporary, matrix->getQualifier().precision, (int)swizzle.size()/2);
        vector = intermediate.addSymbol(*makeInternalVariable("intermVec", vectorType), loc);

        // assign the right to the new vector
        vectorAssign = handleAssign(loc, op, vector, right);
    }

    // Assign the vector components to the matrix components.
    // Store this as a sequence, so a single aggregate node represents this
    // entire operation.
    TIntermAggregate* result = intermediate.makeAggregate(vectorAssign);
    TType columnType(matrix->getType(), 0);
    TType componentType(columnType, 0);
    TType indexType(EbtInt);
    for (int i = 0; i < (int)swizzle.size(); i += 2) {
        // the right component, single index into the RHS vector
        TIntermTyped* rightComp = intermediate.addIndex(EOpIndexDirect, vector,
                                    intermediate.addConstantUnion(i/2, loc), loc);

        // the left component, double index into the LHS matrix
        TIntermTyped* leftComp = intermediate.addIndex(EOpIndexDirect, matrix,
                                    intermediate.addConstantUnion(swizzle[i]->getAsConstantUnion()->getConstArray(),
                                                                  indexType, loc),
                                    loc);
        leftComp->setType(columnType);
        leftComp = intermediate.addIndex(EOpIndexDirect, leftComp,
                                    intermediate.addConstantUnion(swizzle[i+1]->getAsConstantUnion()->getConstArray(),
                                                                  indexType, loc),
                                    loc);
        leftComp->setType(componentType);

        // Add the assignment to the aggregate
        result = intermediate.growAggregate(result, intermediate.addAssign(op, leftComp, rightComp, loc));
    }

    result->setOp(EOpSequence);

    return result;
}

//
// HLSL atomic operations have slightly different arguments than
// GLSL/AST/SPIRV.  The semantics are converted below in decomposeIntrinsic.
// This provides the post-decomposition equivalent opcode.
//
TOperator HlslParseContext::mapAtomicOp(const TSourceLoc& loc, TOperator op, bool isImage)
{
    switch (op) {
    case EOpInterlockedAdd:             return isImage ? EOpImageAtomicAdd      : EOpAtomicAdd;
    case EOpInterlockedAnd:             return isImage ? EOpImageAtomicAnd      : EOpAtomicAnd;
    case EOpInterlockedCompareExchange: return isImage ? EOpImageAtomicCompSwap : EOpAtomicCompSwap;
    case EOpInterlockedMax:             return isImage ? EOpImageAtomicMax      : EOpAtomicMax;
    case EOpInterlockedMin:             return isImage ? EOpImageAtomicMin      : EOpAtomicMin;
    case EOpInterlockedOr:              return isImage ? EOpImageAtomicOr       : EOpAtomicOr;
    case EOpInterlockedXor:             return isImage ? EOpImageAtomicXor      : EOpAtomicXor;
    case EOpInterlockedExchange:        return isImage ? EOpImageAtomicExchange : EOpAtomicExchange;
    case EOpInterlockedCompareStore:  // TODO: ...
    default:
        error(loc, "unknown atomic operation", "unknown op", "");
        return EOpNull;
    }
}

//
// Create a combined sampler/texture from separate sampler and texture.
//
TIntermAggregate* HlslParseContext::handleSamplerTextureCombine(const TSourceLoc& loc, TIntermTyped* argTex,
                                                                TIntermTyped* argSampler)
{
    TIntermAggregate* txcombine = new TIntermAggregate(EOpConstructTextureSampler);

    txcombine->getSequence().push_back(argTex);
    txcombine->getSequence().push_back(argSampler);

    TSampler samplerType = argTex->getType().getSampler();
    samplerType.combined = true;

    // TODO:
    // This block exists until the spec no longer requires shadow modes on texture objects.
    // It can be deleted after that, along with the shadowTextureVariant member.
    {
        const bool shadowMode = argSampler->getType().getSampler().shadow;

        TIntermSymbol* texSymbol = argTex->getAsSymbolNode();

        if (texSymbol == nullptr)
            texSymbol = argTex->getAsBinaryNode()->getLeft()->getAsSymbolNode();

        if (texSymbol == nullptr) {
            error(loc, "unable to find texture symbol", "", "");
            return nullptr;
        }

        // This forces the texture's shadow state to be the sampler's
        // shadow state.  This depends on downstream optimization to
        // DCE one variant in [shadow, nonshadow] if both are present,
        // or the SPIR-V module would be invalid.
        int newId = texSymbol->getId();

        // Check to see if this texture has been given a shadow mode already.
        // If so, look up the one we already have.
        const auto textureShadowEntry = textureShadowVariant.find(texSymbol->getId());

        if (textureShadowEntry != textureShadowVariant.end())
            newId = textureShadowEntry->second->get(shadowMode);
        else
            textureShadowVariant[texSymbol->getId()] = NewPoolObject(tShadowTextureSymbols(), 1);

        // Sometimes we have to create another symbol (if this texture has been seen before,
        // and we haven't created the form for this shadow mode).
        if (newId == -1) {
            TType texType;
            texType.shallowCopy(argTex->getType());
            texType.getSampler().shadow = shadowMode;  // set appropriate shadow mode.
            globalQualifierFix(loc, texType.getQualifier());

            TVariable* newTexture = makeInternalVariable(texSymbol->getName(), texType);

            trackLinkage(*newTexture);

            newId = newTexture->getUniqueId();
        }

        assert(newId != -1);

        if (textureShadowVariant.find(newId) == textureShadowVariant.end())
            textureShadowVariant[newId] = textureShadowVariant[texSymbol->getId()];

        textureShadowVariant[newId]->set(shadowMode, newId);

        // Remember this shadow mode in the texture and the merged type.
        argTex->getWritableType().getSampler().shadow = shadowMode;
        samplerType.shadow = shadowMode;

        texSymbol->switchId(newId);
    }

    txcombine->setType(TType(samplerType, EvqTemporary));
    txcombine->setLoc(loc);

    return txcombine;
}

// Return true if this a buffer type that has an associated counter buffer.
bool HlslParseContext::hasStructBuffCounter(const TType& type) const
{
    switch (type.getQualifier().declaredBuiltIn) {
    case EbvAppendConsume:       // fall through...
    case EbvRWStructuredBuffer:  // ...
        return true;
    default:
        return false; // the other structuredbuffer types do not have a counter.
    }
}

void HlslParseContext::counterBufferType(const TSourceLoc& loc, TType& type)
{
    // Counter type
    TType* counterType = new TType(EbtUint, EvqBuffer);
    counterType->setFieldName(intermediate.implicitCounterName);

    TTypeList* blockStruct = new TTypeList;
    TTypeLoc  member = { counterType, loc };
    blockStruct->push_back(member);

    TType blockType(blockStruct, "", counterType->getQualifier());
    blockType.getQualifier().storage = EvqBuffer;

    type.shallowCopy(blockType);
    shareStructBufferType(type);
}

// declare counter for a structured buffer type
void HlslParseContext::declareStructBufferCounter(const TSourceLoc& loc, const TType& bufferType, const TString& name)
{
    // Bail out if not a struct buffer
    if (! isStructBufferType(bufferType))
        return;

    if (! hasStructBuffCounter(bufferType))
        return;

    TType blockType;
    counterBufferType(loc, blockType);

    TString* blockName = NewPoolTString(intermediate.addCounterBufferName(name).c_str());

    // Counter buffer is not yet in use
    structBufferCounter[*blockName] = false;

    shareStructBufferType(blockType);
    declareBlock(loc, blockType, blockName);
}

// return the counter that goes with a given structuredbuffer
TIntermTyped* HlslParseContext::getStructBufferCounter(const TSourceLoc& loc, TIntermTyped* buffer)
{
    // Bail out if not a struct buffer
    if (buffer == nullptr || ! isStructBufferType(buffer->getType()))
        return nullptr;

    const TString counterBlockName(intermediate.addCounterBufferName(buffer->getAsSymbolNode()->getName()));

    // Mark the counter as being used
    structBufferCounter[counterBlockName] = true;

    TIntermTyped* counterVar = handleVariable(loc, &counterBlockName);  // find the block structure
    TIntermTyped* index = intermediate.addConstantUnion(0, loc); // index to counter inside block struct

    TIntermTyped* counterMember = intermediate.addIndex(EOpIndexDirectStruct, counterVar, index, loc);
    counterMember->setType(TType(EbtUint));
    return counterMember;
}

//
// Decompose structure buffer methods into AST
//
void HlslParseContext::decomposeStructBufferMethods(const TSourceLoc& loc, TIntermTyped*& node, TIntermNode* arguments)
{
    if (node == nullptr || node->getAsOperator() == nullptr || arguments == nullptr)
        return;

    const TOperator op  = node->getAsOperator()->getOp();
    TIntermAggregate* argAggregate = arguments->getAsAggregate();

    // Buffer is the object upon which method is called, so always arg 0
    TIntermTyped* bufferObj = nullptr;

    // The parameters can be an aggregate, or just a the object as a symbol if there are no fn params.
    if (argAggregate) {
        if (argAggregate->getSequence().empty())
            return;
        if (argAggregate->getSequence()[0])
            bufferObj = argAggregate->getSequence()[0]->getAsTyped();
    } else {
        bufferObj = arguments->getAsSymbolNode();
    }

    if (bufferObj == nullptr || bufferObj->getAsSymbolNode() == nullptr)
        return;

    // Some methods require a hidden internal counter, obtained via getStructBufferCounter().
    // This lambda adds something to it and returns the old value.
    const auto incDecCounter = [&](int incval) -> TIntermTyped* {
        TIntermTyped* incrementValue = intermediate.addConstantUnion(static_cast<unsigned int>(incval), loc, true);
        TIntermTyped* counter = getStructBufferCounter(loc, bufferObj); // obtain the counter member

        if (counter == nullptr)
            return nullptr;

        TIntermAggregate* counterIncrement = new TIntermAggregate(EOpAtomicAdd);
        counterIncrement->setType(TType(EbtUint, EvqTemporary));
        counterIncrement->setLoc(loc);
        counterIncrement->getSequence().push_back(counter);
        counterIncrement->getSequence().push_back(incrementValue);

        return counterIncrement;
    };

    // Index to obtain the runtime sized array out of the buffer.
    TIntermTyped* argArray = indexStructBufferContent(loc, bufferObj);
    if (argArray == nullptr)
        return;  // It might not be a struct buffer method.

    switch (op) {
    case EOpMethodLoad:
        {
            TIntermTyped* argIndex = makeIntegerIndex(argAggregate->getSequence()[1]->getAsTyped());  // index

            const TType& bufferType = bufferObj->getType();

            const TBuiltInVariable builtInType = bufferType.getQualifier().declaredBuiltIn;

            // Byte address buffers index in bytes (only multiples of 4 permitted... not so much a byte address
            // buffer then, but that's what it calls itself.
            const bool isByteAddressBuffer = (builtInType == EbvByteAddressBuffer   || 
                                              builtInType == EbvRWByteAddressBuffer);
                

            if (isByteAddressBuffer)
                argIndex = intermediate.addBinaryNode(EOpRightShift, argIndex,
                                                      intermediate.addConstantUnion(2, loc, true),
                                                      loc, TType(EbtInt));

            // Index into the array to find the item being loaded.
            const TOperator idxOp = (argIndex->getQualifier().storage == EvqConst) ? EOpIndexDirect : EOpIndexIndirect;

            node = intermediate.addIndex(idxOp, argArray, argIndex, loc);

            const TType derefType(argArray->getType(), 0);
            node->setType(derefType);
        }
        
        break;

    case EOpMethodLoad2:
    case EOpMethodLoad3:
    case EOpMethodLoad4:
        {
            TIntermTyped* argIndex = makeIntegerIndex(argAggregate->getSequence()[1]->getAsTyped());  // index

            TOperator constructOp = EOpNull;
            int size = 0;

            switch (op) {
            case EOpMethodLoad2: size = 2; constructOp = EOpConstructVec2; break;
            case EOpMethodLoad3: size = 3; constructOp = EOpConstructVec3; break;
            case EOpMethodLoad4: size = 4; constructOp = EOpConstructVec4; break;
            default: assert(0);
            }

            TIntermTyped* body = nullptr;

            // First, we'll store the address in a variable to avoid multiple shifts
            // (we must convert the byte address to an item address)
            TIntermTyped* byteAddrIdx = intermediate.addBinaryNode(EOpRightShift, argIndex,
                                                                   intermediate.addConstantUnion(2, loc, true),
                                                                   loc, TType(EbtInt));

            TVariable* byteAddrSym = makeInternalVariable("byteAddrTemp", TType(EbtInt, EvqTemporary));
            TIntermTyped* byteAddrIdxVar = intermediate.addSymbol(*byteAddrSym, loc);

            body = intermediate.growAggregate(body, intermediate.addAssign(EOpAssign, byteAddrIdxVar, byteAddrIdx, loc));

            TIntermTyped* vec = nullptr;

            // These are only valid on (rw)byteaddressbuffers, so we can always perform the >>2
            // address conversion.
            for (int idx=0; idx<size; ++idx) {
                TIntermTyped* offsetIdx = byteAddrIdxVar;

                // add index offset
                if (idx != 0)
                    offsetIdx = intermediate.addBinaryNode(EOpAdd, offsetIdx,
                                                           intermediate.addConstantUnion(idx, loc, true),
                                                           loc, TType(EbtInt));

                const TOperator idxOp = (offsetIdx->getQualifier().storage == EvqConst) ? EOpIndexDirect
                                                                                        : EOpIndexIndirect;

                TIntermTyped* indexVal = intermediate.addIndex(idxOp, argArray, offsetIdx, loc);

                TType derefType(argArray->getType(), 0);
                derefType.getQualifier().makeTemporary();
                indexVal->setType(derefType);

                vec = intermediate.growAggregate(vec, indexVal);
            }

            vec->setType(TType(argArray->getBasicType(), EvqTemporary, size));
            vec->getAsAggregate()->setOperator(constructOp);

            body = intermediate.growAggregate(body, vec);
            body->setType(vec->getType());
            body->getAsAggregate()->setOperator(EOpSequence);

            node = body;
        }

        break;

    case EOpMethodStore:
    case EOpMethodStore2:
    case EOpMethodStore3:
    case EOpMethodStore4:
        {
            TIntermTyped* argIndex = makeIntegerIndex(argAggregate->getSequence()[1]->getAsTyped());  // index
            TIntermTyped* argValue = argAggregate->getSequence()[2]->getAsTyped();  // value

            // Index into the array to find the item being loaded.
            // Byte address buffers index in bytes (only multiples of 4 permitted... not so much a byte address
            // buffer then, but that's what it calls itself).

            int size = 0;

            switch (op) {
            case EOpMethodStore:  size = 1; break;
            case EOpMethodStore2: size = 2; break;
            case EOpMethodStore3: size = 3; break;
            case EOpMethodStore4: size = 4; break;
            default: assert(0);
            }

            TIntermAggregate* body = nullptr;

            // First, we'll store the address in a variable to avoid multiple shifts
            // (we must convert the byte address to an item address)
            TIntermTyped* byteAddrIdx = intermediate.addBinaryNode(EOpRightShift, argIndex,
                                                                   intermediate.addConstantUnion(2, loc, true), loc, TType(EbtInt));

            TVariable* byteAddrSym = makeInternalVariable("byteAddrTemp", TType(EbtInt, EvqTemporary));
            TIntermTyped* byteAddrIdxVar = intermediate.addSymbol(*byteAddrSym, loc);

            body = intermediate.growAggregate(body, intermediate.addAssign(EOpAssign, byteAddrIdxVar, byteAddrIdx, loc));

            for (int idx=0; idx<size; ++idx) {
                TIntermTyped* offsetIdx = byteAddrIdxVar;
                TIntermTyped* idxConst = intermediate.addConstantUnion(idx, loc, true);

                // add index offset
                if (idx != 0)
                    offsetIdx = intermediate.addBinaryNode(EOpAdd, offsetIdx, idxConst, loc, TType(EbtInt));

                const TOperator idxOp = (offsetIdx->getQualifier().storage == EvqConst) ? EOpIndexDirect
                                                                                        : EOpIndexIndirect;

                TIntermTyped* lValue = intermediate.addIndex(idxOp, argArray, offsetIdx, loc);
                const TType derefType(argArray->getType(), 0);
                lValue->setType(derefType);

                TIntermTyped* rValue;
                if (size == 1) {
                    rValue = argValue;
                } else {
                    rValue = intermediate.addIndex(EOpIndexDirect, argValue, idxConst, loc);
                    const TType indexType(argValue->getType(), 0);
                    rValue->setType(indexType);
                }
                    
                TIntermTyped* assign = intermediate.addAssign(EOpAssign, lValue, rValue, loc); 

                body = intermediate.growAggregate(body, assign);
            }

            body->setOperator(EOpSequence);
            node = body;
        }

        break;

    case EOpMethodGetDimensions:
        {
            const int numArgs = (int)argAggregate->getSequence().size();
            TIntermTyped* argNumItems = argAggregate->getSequence()[1]->getAsTyped();  // out num items
            TIntermTyped* argStride   = numArgs > 2 ? argAggregate->getSequence()[2]->getAsTyped() : nullptr;  // out stride

            TIntermAggregate* body = nullptr;

            // Length output:
            if (argArray->getType().isSizedArray()) {
                const int length = argArray->getType().getOuterArraySize();
                TIntermTyped* assign = intermediate.addAssign(EOpAssign, argNumItems,
                                                              intermediate.addConstantUnion(length, loc, true), loc);
                body = intermediate.growAggregate(body, assign, loc);
            } else {
                TIntermTyped* lengthCall = intermediate.addBuiltInFunctionCall(loc, EOpArrayLength, true, argArray,
                                                                               argNumItems->getType());
                TIntermTyped* assign = intermediate.addAssign(EOpAssign, argNumItems, lengthCall, loc);
                body = intermediate.growAggregate(body, assign, loc);
            }

            // Stride output:
            if (argStride != nullptr) {
                int size;
                int stride;
                intermediate.getMemberAlignment(argArray->getType(), size, stride, argArray->getType().getQualifier().layoutPacking,
                                                argArray->getType().getQualifier().layoutMatrix == ElmRowMajor);

                TIntermTyped* assign = intermediate.addAssign(EOpAssign, argStride,
                                                              intermediate.addConstantUnion(stride, loc, true), loc);

                body = intermediate.growAggregate(body, assign);
            }

            body->setOperator(EOpSequence);
            node = body;
        }

        break;

    case EOpInterlockedAdd:
    case EOpInterlockedAnd:
    case EOpInterlockedExchange:
    case EOpInterlockedMax:
    case EOpInterlockedMin:
    case EOpInterlockedOr:
    case EOpInterlockedXor:
    case EOpInterlockedCompareExchange:
    case EOpInterlockedCompareStore:
        {
            // We'll replace the first argument with the block dereference, and let
            // downstream decomposition handle the rest.

            TIntermSequence& sequence = argAggregate->getSequence();

            TIntermTyped* argIndex     = makeIntegerIndex(sequence[1]->getAsTyped());  // index
            argIndex = intermediate.addBinaryNode(EOpRightShift, argIndex, intermediate.addConstantUnion(2, loc, true),
                                                  loc, TType(EbtInt));

            const TOperator idxOp = (argIndex->getQualifier().storage == EvqConst) ? EOpIndexDirect : EOpIndexIndirect;
            TIntermTyped* element = intermediate.addIndex(idxOp, argArray, argIndex, loc);

            const TType derefType(argArray->getType(), 0);
            element->setType(derefType);

            // Replace the numeric byte offset parameter with array reference.
            sequence[1] = element;
            sequence.erase(sequence.begin(), sequence.begin()+1);
        }
        break;

    case EOpMethodIncrementCounter:
        {
            node = incDecCounter(1);
            break;
        }

    case EOpMethodDecrementCounter:
        {
            TIntermTyped* preIncValue = incDecCounter(-1); // result is original value
            node = intermediate.addBinaryNode(EOpAdd, preIncValue, intermediate.addConstantUnion(-1, loc, true), loc,
                                              preIncValue->getType());
            break;
        }

    case EOpMethodAppend:
        {
            TIntermTyped* oldCounter = incDecCounter(1);

            TIntermTyped* lValue = intermediate.addIndex(EOpIndexIndirect, argArray, oldCounter, loc);
            TIntermTyped* rValue = argAggregate->getSequence()[1]->getAsTyped();

            const TType derefType(argArray->getType(), 0);
            lValue->setType(derefType);

            node = intermediate.addAssign(EOpAssign, lValue, rValue, loc);

            break;
        }

    case EOpMethodConsume:
        {
            TIntermTyped* oldCounter = incDecCounter(-1);

            TIntermTyped* newCounter = intermediate.addBinaryNode(EOpAdd, oldCounter,
                                                                  intermediate.addConstantUnion(-1, loc, true), loc,
                                                                  oldCounter->getType());

            node = intermediate.addIndex(EOpIndexIndirect, argArray, newCounter, loc);

            const TType derefType(argArray->getType(), 0);
            node->setType(derefType);

            break;
        }

    default:
        break; // most pass through unchanged
    }
}

// Create array of standard sample positions for given sample count.
// TODO: remove when a real method to query sample pos exists in SPIR-V.
TIntermConstantUnion* HlslParseContext::getSamplePosArray(int count)
{
    struct tSamplePos { float x, y; };

    static const tSamplePos pos1[] = {
        { 0.0/16.0,  0.0/16.0 },
    };

    // standard sample positions for 2, 4, 8, and 16 samples.
    static const tSamplePos pos2[] = {
        { 4.0/16.0,  4.0/16.0 }, {-4.0/16.0, -4.0/16.0 },
    };

    static const tSamplePos pos4[] = {
        {-2.0/16.0, -6.0/16.0 }, { 6.0/16.0, -2.0/16.0 }, {-6.0/16.0,  2.0/16.0 }, { 2.0/16.0,  6.0/16.0 },
    };

    static const tSamplePos pos8[] = {
        { 1.0/16.0, -3.0/16.0 }, {-1.0/16.0,  3.0/16.0 }, { 5.0/16.0,  1.0/16.0 }, {-3.0/16.0, -5.0/16.0 },
        {-5.0/16.0,  5.0/16.0 }, {-7.0/16.0, -1.0/16.0 }, { 3.0/16.0,  7.0/16.0 }, { 7.0/16.0, -7.0/16.0 },
    };

    static const tSamplePos pos16[] = {
        { 1.0/16.0,  1.0/16.0 }, {-1.0/16.0, -3.0/16.0 }, {-3.0/16.0,  2.0/16.0 }, { 4.0/16.0, -1.0/16.0 },
        {-5.0/16.0, -2.0/16.0 }, { 2.0/16.0,  5.0/16.0 }, { 5.0/16.0,  3.0/16.0 }, { 3.0/16.0, -5.0/16.0 },
        {-2.0/16.0,  6.0/16.0 }, { 0.0/16.0, -7.0/16.0 }, {-4.0/16.0, -6.0/16.0 }, {-6.0/16.0,  4.0/16.0 },
        {-8.0/16.0,  0.0/16.0 }, { 7.0/16.0, -4.0/16.0 }, { 6.0/16.0,  7.0/16.0 }, {-7.0/16.0, -8.0/16.0 },
    };

    const tSamplePos* sampleLoc = nullptr;
    int numSamples = count;

    switch (count) {
    case 2:  sampleLoc = pos2;  break;
    case 4:  sampleLoc = pos4;  break;
    case 8:  sampleLoc = pos8;  break;
    case 16: sampleLoc = pos16; break;
    default:
        sampleLoc = pos1;
        numSamples = 1;
    }

    TConstUnionArray* values = new TConstUnionArray(numSamples*2);
    
    for (int pos=0; pos<count; ++pos) {
        TConstUnion x, y;
        x.setDConst(sampleLoc[pos].x);
        y.setDConst(sampleLoc[pos].y);

        (*values)[pos*2+0] = x;
        (*values)[pos*2+1] = y;
    }

    TType retType(EbtFloat, EvqConst, 2);

    if (numSamples != 1) {
        TArraySizes* arraySizes = new TArraySizes;
        arraySizes->addInnerSize(numSamples);
        retType.transferArraySizes(arraySizes);
    }

    return new TIntermConstantUnion(*values, retType);
}

//
// Decompose DX9 and DX10 sample intrinsics & object methods into AST
//
void HlslParseContext::decomposeSampleMethods(const TSourceLoc& loc, TIntermTyped*& node, TIntermNode* arguments)
{
    if (node == nullptr || !node->getAsOperator())
        return;

    // Sampler return must always be a vec4, but we can construct a shorter vector or a structure from it.
    const auto convertReturn = [&loc, &node, this](TIntermTyped* result, const TSampler& sampler) -> TIntermTyped* {
        result->setType(TType(node->getType().getBasicType(), EvqTemporary, node->getVectorSize()));

        TIntermTyped* convertedResult = nullptr;
        
        TType retType;
        getTextureReturnType(sampler, retType);

        if (retType.isStruct()) {
            // For type convenience, conversionAggregate points to the convertedResult (we know it's an aggregate here)
            TIntermAggregate* conversionAggregate = new TIntermAggregate;
            convertedResult = conversionAggregate;

            // Convert vector output to return structure.  We will need a temp symbol to copy the results to.
            TVariable* structVar = makeInternalVariable("@sampleStructTemp", retType);

            // We also need a temp symbol to hold the result of the texture.  We don't want to re-fetch the
            // sample each time we'll index into the result, so we'll copy to this, and index into the copy.
            TVariable* sampleShadow = makeInternalVariable("@sampleResultShadow", result->getType());

            // Initial copy from texture to our sample result shadow.
            TIntermTyped* shadowCopy = intermediate.addAssign(EOpAssign, intermediate.addSymbol(*sampleShadow, loc),
                                                              result, loc);

            conversionAggregate->getSequence().push_back(shadowCopy);

            unsigned vec4Pos = 0;

            for (unsigned m = 0; m < unsigned(retType.getStruct()->size()); ++m) {
                const TType memberType(retType, m); // dereferenced type of the member we're about to assign.
                
                // Check for bad struct members.  This should have been caught upstream.  Complain, because
                // wwe don't know what to do with it.  This algorithm could be generalized to handle
                // other things, e.g, sub-structures, but HLSL doesn't allow them.
                if (!memberType.isVector() && !memberType.isScalar()) {
                    error(loc, "expected: scalar or vector type in texture structure", "", "");
                    return nullptr;
                }
                    
                // Index into the struct variable to find the member to assign.
                TIntermTyped* structMember = intermediate.addIndex(EOpIndexDirectStruct,
                                                                   intermediate.addSymbol(*structVar, loc),
                                                                   intermediate.addConstantUnion(m, loc), loc);

                structMember->setType(memberType);

                // Assign each component of (possible) vector in struct member.
                for (int component = 0; component < memberType.getVectorSize(); ++component) {
                    TIntermTyped* vec4Member = intermediate.addIndex(EOpIndexDirect,
                                                                     intermediate.addSymbol(*sampleShadow, loc),
                                                                     intermediate.addConstantUnion(vec4Pos++, loc), loc);
                    vec4Member->setType(TType(memberType.getBasicType(), EvqTemporary, 1));

                    TIntermTyped* memberAssign = nullptr;

                    if (memberType.isVector()) {
                        // Vector member: we need to create an access chain to the vector component.

                        TIntermTyped* structVecComponent = intermediate.addIndex(EOpIndexDirect, structMember,
                                                                                 intermediate.addConstantUnion(component, loc), loc);
                        
                        memberAssign = intermediate.addAssign(EOpAssign, structVecComponent, vec4Member, loc);
                    } else {
                        // Scalar member: we can assign to it directly.
                        memberAssign = intermediate.addAssign(EOpAssign, structMember, vec4Member, loc);
                    }

                    
                    conversionAggregate->getSequence().push_back(memberAssign);
                }
            }

            // Add completed variable so the expression results in the whole struct value we just built.
            conversionAggregate->getSequence().push_back(intermediate.addSymbol(*structVar, loc));

            // Make it a sequence.
            intermediate.setAggregateOperator(conversionAggregate, EOpSequence, retType, loc);
        } else {
            // vector clamp the output if template vector type is smaller than sample result.
            if (retType.getVectorSize() < node->getVectorSize()) {
                // Too many components.  Construct shorter vector from it.
                const TOperator op = intermediate.mapTypeToConstructorOp(retType);

                convertedResult = constructBuiltIn(retType, op, result, loc, false);
            } else {
                // Enough components.  Use directly.
                convertedResult = result;
            }
        }

        convertedResult->setLoc(loc);
        return convertedResult;
    };

    const TOperator op  = node->getAsOperator()->getOp();
    const TIntermAggregate* argAggregate = arguments ? arguments->getAsAggregate() : nullptr;

    // Bail out if not a sampler method.
    // Note though this is odd to do before checking the op, because the op
    // could be something that takes the arguments, and the function in question
    // takes the result of the op.  So, this is not the final word.
    if (arguments != nullptr) {
        if (argAggregate == nullptr) {
            if (arguments->getAsTyped()->getBasicType() != EbtSampler)
                return;
        } else {
            if (argAggregate->getSequence().size() == 0 || 
                argAggregate->getSequence()[0] == nullptr ||
                argAggregate->getSequence()[0]->getAsTyped()->getBasicType() != EbtSampler)
                return;
        }
    }

    switch (op) {
    // **** DX9 intrinsics: ****
    case EOpTexture:
        {
            // Texture with ddx & ddy is really gradient form in HLSL
            if (argAggregate->getSequence().size() == 4)
                node->getAsAggregate()->setOperator(EOpTextureGrad);

            break;
        }
    case EOpTextureLod: //is almost EOpTextureBias (only args & operations are different)
        {
            TIntermTyped *argSamp = argAggregate->getSequence()[0]->getAsTyped();   // sampler
            TIntermTyped *argCoord = argAggregate->getSequence()[1]->getAsTyped();  // coord

            assert(argCoord->getVectorSize() == 4);
            TIntermTyped *w = intermediate.addConstantUnion(3, loc, true);
            TIntermTyped *argLod = intermediate.addIndex(EOpIndexDirect, argCoord, w, loc);

            TOperator constructOp = EOpNull;
            const TSampler &sampler = argSamp->getType().getSampler();
            int coordSize = 0;

            switch (sampler.dim)
            {
            case Esd1D:   constructOp = EOpConstructFloat; coordSize = 1; break; // 1D
            case Esd2D:   constructOp = EOpConstructVec2;  coordSize = 2; break; // 2D
            case Esd3D:   constructOp = EOpConstructVec3;  coordSize = 3; break; // 3D
            case EsdCube: constructOp = EOpConstructVec3;  coordSize = 3; break; // also 3D
            default:
                break;
            }

            TIntermAggregate *constructCoord = new TIntermAggregate(constructOp);
            constructCoord->getSequence().push_back(argCoord);
            constructCoord->setLoc(loc);
            constructCoord->setType(TType(argCoord->getBasicType(), EvqTemporary, coordSize));

            TIntermAggregate *tex = new TIntermAggregate(EOpTextureLod);
            tex->getSequence().push_back(argSamp);        // sampler
            tex->getSequence().push_back(constructCoord); // coordinate
            tex->getSequence().push_back(argLod);         // lod

            node = convertReturn(tex, sampler);

            break;
        }

    case EOpTextureBias:
        {
            TIntermTyped* arg0 = argAggregate->getSequence()[0]->getAsTyped();  // sampler
            TIntermTyped* arg1 = argAggregate->getSequence()[1]->getAsTyped();  // coord

            // HLSL puts bias in W component of coordinate.  We extract it and add it to
            // the argument list, instead
            TIntermTyped* w = intermediate.addConstantUnion(3, loc, true);
            TIntermTyped* bias = intermediate.addIndex(EOpIndexDirect, arg1, w, loc);

            TOperator constructOp = EOpNull;
            const TSampler& sampler = arg0->getType().getSampler();

            switch (sampler.dim) {
            case Esd1D:   constructOp = EOpConstructFloat; break; // 1D
            case Esd2D:   constructOp = EOpConstructVec2;  break; // 2D
            case Esd3D:   constructOp = EOpConstructVec3;  break; // 3D
            case EsdCube: constructOp = EOpConstructVec3;  break; // also 3D
            default: break;
            }

            TIntermAggregate* constructCoord = new TIntermAggregate(constructOp);
            constructCoord->getSequence().push_back(arg1);
            constructCoord->setLoc(loc);

            // The input vector should never be less than 2, since there's always a bias.
            // The max is for safety, and should be a no-op.
            constructCoord->setType(TType(arg1->getBasicType(), EvqTemporary, std::max(arg1->getVectorSize() - 1, 0)));

            TIntermAggregate* tex = new TIntermAggregate(EOpTexture);
            tex->getSequence().push_back(arg0);           // sampler
            tex->getSequence().push_back(constructCoord); // coordinate
            tex->getSequence().push_back(bias);           // bias

            node = convertReturn(tex, sampler);

            break;
        }

    // **** DX10 methods: ****
    case EOpMethodSample:     // fall through
    case EOpMethodSampleBias: // ...
        {
            TIntermTyped* argTex    = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* argSamp   = argAggregate->getSequence()[1]->getAsTyped();
            TIntermTyped* argCoord  = argAggregate->getSequence()[2]->getAsTyped();
            TIntermTyped* argBias   = nullptr;
            TIntermTyped* argOffset = nullptr;
            const TSampler& sampler = argTex->getType().getSampler();

            int nextArg = 3;

            if (op == EOpMethodSampleBias)  // SampleBias has a bias arg
                argBias = argAggregate->getSequence()[nextArg++]->getAsTyped();

            TOperator textureOp = EOpTexture;

            if ((int)argAggregate->getSequence().size() == (nextArg+1)) { // last parameter is offset form
                textureOp = EOpTextureOffset;
                argOffset = argAggregate->getSequence()[nextArg++]->getAsTyped();
            }

            TIntermAggregate* txcombine = handleSamplerTextureCombine(loc, argTex, argSamp);

            TIntermAggregate* txsample = new TIntermAggregate(textureOp);
            txsample->getSequence().push_back(txcombine);
            txsample->getSequence().push_back(argCoord);

            if (argBias != nullptr)
                txsample->getSequence().push_back(argBias);

            if (argOffset != nullptr)
                txsample->getSequence().push_back(argOffset);

            node = convertReturn(txsample, sampler);

            break;
        }

    case EOpMethodSampleGrad: // ...
        {
            TIntermTyped* argTex    = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* argSamp   = argAggregate->getSequence()[1]->getAsTyped();
            TIntermTyped* argCoord  = argAggregate->getSequence()[2]->getAsTyped();
            TIntermTyped* argDDX    = argAggregate->getSequence()[3]->getAsTyped();
            TIntermTyped* argDDY    = argAggregate->getSequence()[4]->getAsTyped();
            TIntermTyped* argOffset = nullptr;
            const TSampler& sampler = argTex->getType().getSampler();

            TOperator textureOp = EOpTextureGrad;

            if (argAggregate->getSequence().size() == 6) { // last parameter is offset form
                textureOp = EOpTextureGradOffset;
                argOffset = argAggregate->getSequence()[5]->getAsTyped();
            }

            TIntermAggregate* txcombine = handleSamplerTextureCombine(loc, argTex, argSamp);

            TIntermAggregate* txsample = new TIntermAggregate(textureOp);
            txsample->getSequence().push_back(txcombine);
            txsample->getSequence().push_back(argCoord);
            txsample->getSequence().push_back(argDDX);
            txsample->getSequence().push_back(argDDY);

            if (argOffset != nullptr)
                txsample->getSequence().push_back(argOffset);

            node = convertReturn(txsample, sampler);

            break;
        }

    case EOpMethodGetDimensions:
        {
            // AST returns a vector of results, which we break apart component-wise into
            // separate values to assign to the HLSL method's outputs, ala:
            //  tx . GetDimensions(width, height);
            //      float2 sizeQueryTemp = EOpTextureQuerySize
            //      width = sizeQueryTemp.X;
            //      height = sizeQueryTemp.Y;

            TIntermTyped* argTex = argAggregate->getSequence()[0]->getAsTyped();
            const TType& texType = argTex->getType();

            assert(texType.getBasicType() == EbtSampler);

            const TSampler& sampler = texType.getSampler();
            const TSamplerDim dim = sampler.dim;
            const bool isImage = sampler.isImage();
            const bool isMs = sampler.isMultiSample();
            const int numArgs = (int)argAggregate->getSequence().size();

            int numDims = 0;

            switch (dim) {
            case Esd1D:     numDims = 1; break; // W
            case Esd2D:     numDims = 2; break; // W, H
            case Esd3D:     numDims = 3; break; // W, H, D
            case EsdCube:   numDims = 2; break; // W, H (cube)
            case EsdBuffer: numDims = 1; break; // W (buffers)
            case EsdRect:   numDims = 2; break; // W, H (rect)
            default:
                assert(0 && "unhandled texture dimension");
            }

            // Arrayed adds another dimension for the number of array elements
            if (sampler.isArrayed())
                ++numDims;

            // Establish whether the method itself is querying mip levels.  This can be false even
            // if the underlying query requires a MIP level, due to the available HLSL method overloads.
            const bool mipQuery = (numArgs > (numDims + 1 + (isMs ? 1 : 0)));

            // Establish whether we must use the LOD form of query (even if the method did not supply a mip level to query).
            // True if:
            //   1. 1D/2D/3D/Cube AND multisample==0 AND NOT image (those can be sent to the non-LOD query)
            // or,
            //   2. There is a LOD (because the non-LOD query cannot be used in that case, per spec)
            const bool mipRequired =
                ((dim == Esd1D || dim == Esd2D || dim == Esd3D || dim == EsdCube) && !isMs && !isImage) || // 1...
                mipQuery; // 2...

            // AST assumes integer return.  Will be converted to float if required.
            TIntermAggregate* sizeQuery = new TIntermAggregate(isImage ? EOpImageQuerySize : EOpTextureQuerySize);
            sizeQuery->getSequence().push_back(argTex);

            // If we're building an LOD query, add the LOD.
            if (mipRequired) {
                // If the base HLSL query had no MIP level given, use level 0.
                TIntermTyped* queryLod = mipQuery ? argAggregate->getSequence()[1]->getAsTyped() :
                    intermediate.addConstantUnion(0, loc, true);
                sizeQuery->getSequence().push_back(queryLod);
            }

            sizeQuery->setType(TType(EbtUint, EvqTemporary, numDims));
            sizeQuery->setLoc(loc);

            // Return value from size query
            TVariable* tempArg = makeInternalVariable("sizeQueryTemp", sizeQuery->getType());
            tempArg->getWritableType().getQualifier().makeTemporary();
            TIntermTyped* sizeQueryAssign = intermediate.addAssign(EOpAssign,
                                                                   intermediate.addSymbol(*tempArg, loc),
                                                                   sizeQuery, loc);

            // Compound statement for assigning outputs
            TIntermAggregate* compoundStatement = intermediate.makeAggregate(sizeQueryAssign, loc);
            // Index of first output parameter
            const int outParamBase = mipQuery ? 2 : 1;

            for (int compNum = 0; compNum < numDims; ++compNum) {
                TIntermTyped* indexedOut = nullptr;
                TIntermSymbol* sizeQueryReturn = intermediate.addSymbol(*tempArg, loc);

                if (numDims > 1) {
                    TIntermTyped* component = intermediate.addConstantUnion(compNum, loc, true);
                    indexedOut = intermediate.addIndex(EOpIndexDirect, sizeQueryReturn, component, loc);
                    indexedOut->setType(TType(EbtUint, EvqTemporary, 1));
                    indexedOut->setLoc(loc);
                } else {
                    indexedOut = sizeQueryReturn;
                }

                TIntermTyped* outParam = argAggregate->getSequence()[outParamBase + compNum]->getAsTyped();
                TIntermTyped* compAssign = intermediate.addAssign(EOpAssign, outParam, indexedOut, loc);

                compoundStatement = intermediate.growAggregate(compoundStatement, compAssign);
            }

            // handle mip level parameter
            if (mipQuery) {
                TIntermTyped* outParam = argAggregate->getSequence()[outParamBase + numDims]->getAsTyped();

                TIntermAggregate* levelsQuery = new TIntermAggregate(EOpTextureQueryLevels);
                levelsQuery->getSequence().push_back(argTex);
                levelsQuery->setType(TType(EbtUint, EvqTemporary, 1));
                levelsQuery->setLoc(loc);

                TIntermTyped* compAssign = intermediate.addAssign(EOpAssign, outParam, levelsQuery, loc);
                compoundStatement = intermediate.growAggregate(compoundStatement, compAssign);
            }

            // 2DMS formats query # samples, which needs a different query op
            if (sampler.isMultiSample()) {
                TIntermTyped* outParam = argAggregate->getSequence()[outParamBase + numDims]->getAsTyped();

                TIntermAggregate* samplesQuery = new TIntermAggregate(EOpImageQuerySamples);
                samplesQuery->getSequence().push_back(argTex);
                samplesQuery->setType(TType(EbtUint, EvqTemporary, 1));
                samplesQuery->setLoc(loc);

                TIntermTyped* compAssign = intermediate.addAssign(EOpAssign, outParam, samplesQuery, loc);
                compoundStatement = intermediate.growAggregate(compoundStatement, compAssign);
            }

            compoundStatement->setOperator(EOpSequence);
            compoundStatement->setLoc(loc);
            compoundStatement->setType(TType(EbtVoid));

            node = compoundStatement;

            break;
        }

    case EOpMethodSampleCmp:  // fall through...
    case EOpMethodSampleCmpLevelZero:
        {
            TIntermTyped* argTex    = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* argSamp   = argAggregate->getSequence()[1]->getAsTyped();
            TIntermTyped* argCoord  = argAggregate->getSequence()[2]->getAsTyped();
            TIntermTyped* argCmpVal = argAggregate->getSequence()[3]->getAsTyped();
            TIntermTyped* argOffset = nullptr;

            // Sampler argument should be a sampler.
            if (argSamp->getType().getBasicType() != EbtSampler) {
                error(loc, "expected: sampler type", "", "");
                return;
            }

            // Sampler should be a SamplerComparisonState
            if (! argSamp->getType().getSampler().isShadow()) {
                error(loc, "expected: SamplerComparisonState", "", "");
                return;
            }

            // optional offset value
            if (argAggregate->getSequence().size() > 4)
                argOffset = argAggregate->getSequence()[4]->getAsTyped();

            const int coordDimWithCmpVal = argCoord->getType().getVectorSize() + 1; // +1 for cmp

            // AST wants comparison value as one of the texture coordinates
            TOperator constructOp = EOpNull;
            switch (coordDimWithCmpVal) {
            // 1D can't happen: there's always at least 1 coordinate dimension + 1 cmp val
            case 2: constructOp = EOpConstructVec2;  break;
            case 3: constructOp = EOpConstructVec3;  break;
            case 4: constructOp = EOpConstructVec4;  break;
            case 5: constructOp = EOpConstructVec4;  break; // cubeArrayShadow, cmp value is separate arg.
            default: assert(0); break;
            }

            TIntermAggregate* coordWithCmp = new TIntermAggregate(constructOp);
            coordWithCmp->getSequence().push_back(argCoord);
            if (coordDimWithCmpVal != 5) // cube array shadow is special.
                coordWithCmp->getSequence().push_back(argCmpVal);
            coordWithCmp->setLoc(loc);
            coordWithCmp->setType(TType(argCoord->getBasicType(), EvqTemporary, std::min(coordDimWithCmpVal, 4)));

            TOperator textureOp = (op == EOpMethodSampleCmpLevelZero ? EOpTextureLod : EOpTexture);
            if (argOffset != nullptr)
                textureOp = (op == EOpMethodSampleCmpLevelZero ? EOpTextureLodOffset : EOpTextureOffset);

            // Create combined sampler & texture op
            TIntermAggregate* txcombine = handleSamplerTextureCombine(loc, argTex, argSamp);
            TIntermAggregate* txsample = new TIntermAggregate(textureOp);
            txsample->getSequence().push_back(txcombine);
            txsample->getSequence().push_back(coordWithCmp);

            if (coordDimWithCmpVal == 5) // cube array shadow is special: cmp val follows coord.
                txsample->getSequence().push_back(argCmpVal);

            // the LevelZero form uses 0 as an explicit LOD
            if (op == EOpMethodSampleCmpLevelZero)
                txsample->getSequence().push_back(intermediate.addConstantUnion(0.0, EbtFloat, loc, true));

            // Add offset if present
            if (argOffset != nullptr)
                txsample->getSequence().push_back(argOffset);

            txsample->setType(node->getType());
            txsample->setLoc(loc);
            node = txsample;

            break;
        }

    case EOpMethodLoad:
        {
            TIntermTyped* argTex    = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* argCoord  = argAggregate->getSequence()[1]->getAsTyped();
            TIntermTyped* argOffset = nullptr;
            TIntermTyped* lodComponent = nullptr;
            TIntermTyped* coordSwizzle = nullptr;

            const TSampler& sampler = argTex->getType().getSampler();
            const bool isMS = sampler.isMultiSample();
            const bool isBuffer = sampler.dim == EsdBuffer;
            const bool isImage = sampler.isImage();
            const TBasicType coordBaseType = argCoord->getType().getBasicType();

            // Last component of coordinate is the mip level, for non-MS.  we separate them here:
            if (isMS || isBuffer || isImage) {
                // MS, Buffer, and Image have no LOD
                coordSwizzle = argCoord;
            } else {
                // Extract coordinate
                int swizzleSize = argCoord->getType().getVectorSize() - (isMS ? 0 : 1);
                TSwizzleSelectors<TVectorSelector> coordFields;
                for (int i = 0; i < swizzleSize; ++i)
                    coordFields.push_back(i);
                TIntermTyped* coordIdx = intermediate.addSwizzle(coordFields, loc);
                coordSwizzle = intermediate.addIndex(EOpVectorSwizzle, argCoord, coordIdx, loc);
                coordSwizzle->setType(TType(coordBaseType, EvqTemporary, coordFields.size()));

                // Extract LOD
                TIntermTyped* lodIdx = intermediate.addConstantUnion(coordFields.size(), loc, true);
                lodComponent = intermediate.addIndex(EOpIndexDirect, argCoord, lodIdx, loc);
                lodComponent->setType(TType(coordBaseType, EvqTemporary, 1));
            }

            const int numArgs    = (int)argAggregate->getSequence().size();
            const bool hasOffset = ((!isMS && numArgs == 3) || (isMS && numArgs == 4));

            // Create texel fetch
            const TOperator fetchOp = (isImage   ? EOpImageLoad :
                                       hasOffset ? EOpTextureFetchOffset :
                                       EOpTextureFetch);
            TIntermAggregate* txfetch = new TIntermAggregate(fetchOp);

            // Build up the fetch
            txfetch->getSequence().push_back(argTex);
            txfetch->getSequence().push_back(coordSwizzle);

            if (isMS) {
                // add 2DMS sample index
                TIntermTyped* argSampleIdx  = argAggregate->getSequence()[2]->getAsTyped();
                txfetch->getSequence().push_back(argSampleIdx);
            } else if (isBuffer) {
                // Nothing else to do for buffers.
            } else if (isImage) {
                // Nothing else to do for images.
            } else {
                // 2DMS and buffer have no LOD, but everything else does.
                txfetch->getSequence().push_back(lodComponent);
            }

            // Obtain offset arg, if there is one.
            if (hasOffset) {
                const int offsetPos  = (isMS ? 3 : 2);
                argOffset = argAggregate->getSequence()[offsetPos]->getAsTyped();
                txfetch->getSequence().push_back(argOffset);
            }

            node = convertReturn(txfetch, sampler);

            break;
        }

    case EOpMethodSampleLevel:
        {
            TIntermTyped* argTex    = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* argSamp   = argAggregate->getSequence()[1]->getAsTyped();
            TIntermTyped* argCoord  = argAggregate->getSequence()[2]->getAsTyped();
            TIntermTyped* argLod    = argAggregate->getSequence()[3]->getAsTyped();
            TIntermTyped* argOffset = nullptr;
            const TSampler& sampler = argTex->getType().getSampler();

            const int  numArgs = (int)argAggregate->getSequence().size();

            if (numArgs == 5) // offset, if present
                argOffset = argAggregate->getSequence()[4]->getAsTyped();

            const TOperator textureOp = (argOffset == nullptr ? EOpTextureLod : EOpTextureLodOffset);
            TIntermAggregate* txsample = new TIntermAggregate(textureOp);

            TIntermAggregate* txcombine = handleSamplerTextureCombine(loc, argTex, argSamp);

            txsample->getSequence().push_back(txcombine);
            txsample->getSequence().push_back(argCoord);
            txsample->getSequence().push_back(argLod);

            if (argOffset != nullptr)
                txsample->getSequence().push_back(argOffset);

            node = convertReturn(txsample, sampler);

            break;
        }

    case EOpMethodGather:
        {
            TIntermTyped* argTex    = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* argSamp   = argAggregate->getSequence()[1]->getAsTyped();
            TIntermTyped* argCoord  = argAggregate->getSequence()[2]->getAsTyped();
            TIntermTyped* argOffset = nullptr;

            // Offset is optional
            if (argAggregate->getSequence().size() > 3)
                argOffset = argAggregate->getSequence()[3]->getAsTyped();

            const TOperator textureOp = (argOffset == nullptr ? EOpTextureGather : EOpTextureGatherOffset);
            TIntermAggregate* txgather = new TIntermAggregate(textureOp);

            TIntermAggregate* txcombine = handleSamplerTextureCombine(loc, argTex, argSamp);

            txgather->getSequence().push_back(txcombine);
            txgather->getSequence().push_back(argCoord);
            // Offset if not given is implicitly channel 0 (red)

            if (argOffset != nullptr)
                txgather->getSequence().push_back(argOffset);

            txgather->setType(node->getType());
            txgather->setLoc(loc);
            node = txgather;

            break;
        }

    case EOpMethodGatherRed:      // fall through...
    case EOpMethodGatherGreen:    // ...
    case EOpMethodGatherBlue:     // ...
    case EOpMethodGatherAlpha:    // ...
    case EOpMethodGatherCmpRed:   // ...
    case EOpMethodGatherCmpGreen: // ...
    case EOpMethodGatherCmpBlue:  // ...
    case EOpMethodGatherCmpAlpha: // ...
        {
            int channel = 0;    // the channel we are gathering
            int cmpValues = 0;  // 1 if there is a compare value (handier than a bool below)

            switch (op) {
            case EOpMethodGatherCmpRed:   cmpValues = 1;  // fall through
            case EOpMethodGatherRed:      channel = 0; break;
            case EOpMethodGatherCmpGreen: cmpValues = 1;  // fall through
            case EOpMethodGatherGreen:    channel = 1; break;
            case EOpMethodGatherCmpBlue:  cmpValues = 1;  // fall through
            case EOpMethodGatherBlue:     channel = 2; break;
            case EOpMethodGatherCmpAlpha: cmpValues = 1;  // fall through
            case EOpMethodGatherAlpha:    channel = 3; break;
            default:                      assert(0);   break;
            }

            // For now, we have nothing to map the component-wise comparison forms
            // to, because neither GLSL nor SPIR-V has such an opcode.  Issue an
            // unimplemented error instead.  Most of the machinery is here if that
            // should ever become available.  However, red can be passed through
            // to OpImageDrefGather.  G/B/A cannot, because that opcode does not
            // accept a component.
            if (cmpValues != 0 && op != EOpMethodGatherCmpRed) {
                error(loc, "unimplemented: component-level gather compare", "", "");
                return;
            }

            int arg = 0;

            TIntermTyped* argTex        = argAggregate->getSequence()[arg++]->getAsTyped();
            TIntermTyped* argSamp       = argAggregate->getSequence()[arg++]->getAsTyped();
            TIntermTyped* argCoord      = argAggregate->getSequence()[arg++]->getAsTyped();
            TIntermTyped* argOffset     = nullptr;
            TIntermTyped* argOffsets[4] = { nullptr, nullptr, nullptr, nullptr };
            // TIntermTyped* argStatus     = nullptr; // TODO: residency
            TIntermTyped* argCmp        = nullptr;

            const TSamplerDim dim = argTex->getType().getSampler().dim;

            const int  argSize = (int)argAggregate->getSequence().size();
            bool hasStatus     = (argSize == (5+cmpValues) || argSize == (8+cmpValues));
            bool hasOffset1    = false;
            bool hasOffset4    = false;

            // Sampler argument should be a sampler.
            if (argSamp->getType().getBasicType() != EbtSampler) {
                error(loc, "expected: sampler type", "", "");
                return;
            }

            // Cmp forms require SamplerComparisonState
            if (cmpValues > 0 && ! argSamp->getType().getSampler().isShadow()) {
                error(loc, "expected: SamplerComparisonState", "", "");
                return;
            }

            // Only 2D forms can have offsets.  Discover if we have 0, 1 or 4 offsets.
            if (dim == Esd2D) {
                hasOffset1 = (argSize == (4+cmpValues) || argSize == (5+cmpValues));
                hasOffset4 = (argSize == (7+cmpValues) || argSize == (8+cmpValues));
            }

            assert(!(hasOffset1 && hasOffset4));

            TOperator textureOp = EOpTextureGather;

            // Compare forms have compare value
            if (cmpValues != 0)
                argCmp = argOffset = argAggregate->getSequence()[arg++]->getAsTyped();

            // Some forms have single offset
            if (hasOffset1) {
                textureOp = EOpTextureGatherOffset;   // single offset form
                argOffset = argAggregate->getSequence()[arg++]->getAsTyped();
            }

            // Some forms have 4 gather offsets
            if (hasOffset4) {
                textureOp = EOpTextureGatherOffsets;  // note plural, for 4 offset form
                for (int offsetNum = 0; offsetNum < 4; ++offsetNum)
                    argOffsets[offsetNum] = argAggregate->getSequence()[arg++]->getAsTyped();
            }

            // Residency status
            if (hasStatus) {
                // argStatus = argAggregate->getSequence()[arg++]->getAsTyped();
                error(loc, "unimplemented: residency status", "", "");
                return;
            }

            TIntermAggregate* txgather = new TIntermAggregate(textureOp);
            TIntermAggregate* txcombine = handleSamplerTextureCombine(loc, argTex, argSamp);

            TIntermTyped* argChannel = intermediate.addConstantUnion(channel, loc, true);

            txgather->getSequence().push_back(txcombine);
            txgather->getSequence().push_back(argCoord);

            // AST wants an array of 4 offsets, where HLSL has separate args.  Here
            // we construct an array from the separate args.
            if (hasOffset4) {
                TType arrayType(EbtInt, EvqTemporary, 2);
                TArraySizes* arraySizes = new TArraySizes;
                arraySizes->addInnerSize(4);
                arrayType.transferArraySizes(arraySizes);

                TIntermAggregate* initList = new TIntermAggregate(EOpNull);

                for (int offsetNum = 0; offsetNum < 4; ++offsetNum)
                    initList->getSequence().push_back(argOffsets[offsetNum]);

                argOffset = addConstructor(loc, initList, arrayType);
            }

            // Add comparison value if we have one
            if (argCmp != nullptr)
                txgather->getSequence().push_back(argCmp);

            // Add offset (either 1, or an array of 4) if we have one
            if (argOffset != nullptr)
                txgather->getSequence().push_back(argOffset);

            // Add channel value if the sampler is not shadow
            if (! argSamp->getType().getSampler().isShadow())
                txgather->getSequence().push_back(argChannel);

            txgather->setType(node->getType());
            txgather->setLoc(loc);
            node = txgather;

            break;
        }

    case EOpMethodCalculateLevelOfDetail:
    case EOpMethodCalculateLevelOfDetailUnclamped:
        {
            TIntermTyped* argTex    = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* argSamp   = argAggregate->getSequence()[1]->getAsTyped();
            TIntermTyped* argCoord  = argAggregate->getSequence()[2]->getAsTyped();

            TIntermAggregate* txquerylod = new TIntermAggregate(EOpTextureQueryLod);

            TIntermAggregate* txcombine = handleSamplerTextureCombine(loc, argTex, argSamp);
            txquerylod->getSequence().push_back(txcombine);
            txquerylod->getSequence().push_back(argCoord);

            TIntermTyped* lodComponent = intermediate.addConstantUnion(
                op == EOpMethodCalculateLevelOfDetail ? 0 : 1,
                loc, true);
            TIntermTyped* lodComponentIdx = intermediate.addIndex(EOpIndexDirect, txquerylod, lodComponent, loc);
            lodComponentIdx->setType(TType(EbtFloat, EvqTemporary, 1));
            node = lodComponentIdx;

            break;
        }

    case EOpMethodGetSamplePosition:
        {
            // TODO: this entire decomposition exists because there is not yet a way to query
            // the sample position directly through SPIR-V.  Instead, we return fixed sample
            // positions for common cases.  *** If the sample positions are set differently,
            // this will be wrong. ***

            TIntermTyped* argTex     = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* argSampIdx = argAggregate->getSequence()[1]->getAsTyped();

            TIntermAggregate* samplesQuery = new TIntermAggregate(EOpImageQuerySamples);
            samplesQuery->getSequence().push_back(argTex);
            samplesQuery->setType(TType(EbtUint, EvqTemporary, 1));
            samplesQuery->setLoc(loc);

            TIntermAggregate* compoundStatement = nullptr;

            TVariable* outSampleCount = makeInternalVariable("@sampleCount", TType(EbtUint));
            outSampleCount->getWritableType().getQualifier().makeTemporary();
            TIntermTyped* compAssign = intermediate.addAssign(EOpAssign, intermediate.addSymbol(*outSampleCount, loc),
                                                              samplesQuery, loc);
            compoundStatement = intermediate.growAggregate(compoundStatement, compAssign);

            TIntermTyped* idxtest[4];

            // Create tests against 2, 4, 8, and 16 sample values
            int count = 0;
            for (int val = 2; val <= 16; val *= 2)
                idxtest[count++] =
                    intermediate.addBinaryNode(EOpEqual, 
                                               intermediate.addSymbol(*outSampleCount, loc),
                                               intermediate.addConstantUnion(val, loc),
                                               loc, TType(EbtBool));

            const TOperator idxOp = (argSampIdx->getQualifier().storage == EvqConst) ? EOpIndexDirect : EOpIndexIndirect;
            
            // Create index ops into position arrays given sample index.
            // TODO: should it be clamped?
            TIntermTyped* index[4];
            count = 0;
            for (int val = 2; val <= 16; val *= 2) {
                index[count] = intermediate.addIndex(idxOp, getSamplePosArray(val), argSampIdx, loc);
                index[count++]->setType(TType(EbtFloat, EvqTemporary, 2));
            }

            // Create expression as:
            // (sampleCount == 2)  ? pos2[idx] :
            // (sampleCount == 4)  ? pos4[idx] :
            // (sampleCount == 8)  ? pos8[idx] :
            // (sampleCount == 16) ? pos16[idx] : float2(0,0);
            TIntermTyped* test = 
                intermediate.addSelection(idxtest[0], index[0], 
                    intermediate.addSelection(idxtest[1], index[1], 
                        intermediate.addSelection(idxtest[2], index[2],
                            intermediate.addSelection(idxtest[3], index[3], 
                                                      getSamplePosArray(1), loc), loc), loc), loc);
                                         
            compoundStatement = intermediate.growAggregate(compoundStatement, test);
            compoundStatement->setOperator(EOpSequence);
            compoundStatement->setLoc(loc);
            compoundStatement->setType(TType(EbtFloat, EvqTemporary, 2));

            node = compoundStatement;

            break;
        }

    case EOpSubpassLoad:
        {
            const TIntermTyped* argSubpass = 
                argAggregate ? argAggregate->getSequence()[0]->getAsTyped() :
                arguments->getAsTyped();

            const TSampler& sampler = argSubpass->getType().getSampler();

            // subpass load: the multisample form is overloaded.  Here, we convert that to
            // the EOpSubpassLoadMS opcode.
            if (argAggregate != nullptr && argAggregate->getSequence().size() > 1)
                node->getAsOperator()->setOp(EOpSubpassLoadMS);

            node = convertReturn(node, sampler);

            break;
        }
        

    default:
        break; // most pass through unchanged
    }
}

//
// Decompose geometry shader methods
//
void HlslParseContext::decomposeGeometryMethods(const TSourceLoc& loc, TIntermTyped*& node, TIntermNode* arguments)
{
    if (node == nullptr || !node->getAsOperator())
        return;

    const TOperator op  = node->getAsOperator()->getOp();
    const TIntermAggregate* argAggregate = arguments ? arguments->getAsAggregate() : nullptr;

    switch (op) {
    case EOpMethodAppend:
        if (argAggregate) {
            // Don't emit these for non-GS stage, since we won't have the gsStreamOutput symbol.
            if (language != EShLangGeometry) {
                node = nullptr;
                return;
            }

            TIntermAggregate* sequence = nullptr;
            TIntermAggregate* emit = new TIntermAggregate(EOpEmitVertex);

            emit->setLoc(loc);
            emit->setType(TType(EbtVoid));

            TIntermTyped* data = argAggregate->getSequence()[1]->getAsTyped();

            // This will be patched in finalization during finalizeAppendMethods()
            sequence = intermediate.growAggregate(sequence, data, loc);
            sequence = intermediate.growAggregate(sequence, emit);

            sequence->setOperator(EOpSequence);
            sequence->setLoc(loc);
            sequence->setType(TType(EbtVoid));

            gsAppends.push_back({sequence, loc});

            node = sequence;
        }
        break;

    case EOpMethodRestartStrip:
        {
            // Don't emit these for non-GS stage, since we won't have the gsStreamOutput symbol.
            if (language != EShLangGeometry) {
                node = nullptr;
                return;
            }

            TIntermAggregate* cut = new TIntermAggregate(EOpEndPrimitive);
            cut->setLoc(loc);
            cut->setType(TType(EbtVoid));
            node = cut;
        }
        break;

    default:
        break; // most pass through unchanged
    }
}

//
// Optionally decompose intrinsics to AST opcodes.
//
void HlslParseContext::decomposeIntrinsic(const TSourceLoc& loc, TIntermTyped*& node, TIntermNode* arguments)
{
    // Helper to find image data for image atomics:
    // OpImageLoad(image[idx])
    // We take the image load apart and add its params to the atomic op aggregate node
    const auto imageAtomicParams = [this, &loc, &node](TIntermAggregate* atomic, TIntermTyped* load) {
        TIntermAggregate* loadOp = load->getAsAggregate();
        if (loadOp == nullptr) {
            error(loc, "unknown image type in atomic operation", "", "");
            node = nullptr;
            return;
        }

        atomic->getSequence().push_back(loadOp->getSequence()[0]);
        atomic->getSequence().push_back(loadOp->getSequence()[1]);
    };

    // Return true if this is an imageLoad, which we will change to an image atomic.
    const auto isImageParam = [](TIntermTyped* image) -> bool {
        TIntermAggregate* imageAggregate = image->getAsAggregate();
        return imageAggregate != nullptr && imageAggregate->getOp() == EOpImageLoad;
    };

    const auto lookupBuiltinVariable = [&](const char* name, TBuiltInVariable builtin, TType& type) -> TIntermTyped* {
        TSymbol* symbol = symbolTable.find(name);
        if (nullptr == symbol) {
            type.getQualifier().builtIn = builtin;

            TVariable* variable = new TVariable(NewPoolTString(name), type);

            symbolTable.insert(*variable);

            symbol = symbolTable.find(name);
            assert(symbol && "Inserted symbol could not be found!");
        }

        return intermediate.addSymbol(*(symbol->getAsVariable()), loc);
    };

    // HLSL intrinsics can be pass through to native AST opcodes, or decomposed here to existing AST
    // opcodes for compatibility with existing software stacks.
    static const bool decomposeHlslIntrinsics = true;

    if (!decomposeHlslIntrinsics || !node || !node->getAsOperator())
        return;

    const TIntermAggregate* argAggregate = arguments ? arguments->getAsAggregate() : nullptr;
    TIntermUnary* fnUnary = node->getAsUnaryNode();
    const TOperator op  = node->getAsOperator()->getOp();

    switch (op) {
    case EOpGenMul:
        {
            // mul(a,b) -> MatrixTimesMatrix, MatrixTimesVector, MatrixTimesScalar, VectorTimesScalar, Dot, Mul
            // Since we are treating HLSL rows like GLSL columns (the first matrix indirection),
            // we must reverse the operand order here.  Hence, arg0 gets sequence[1], etc.
            TIntermTyped* arg0 = argAggregate->getSequence()[1]->getAsTyped();
            TIntermTyped* arg1 = argAggregate->getSequence()[0]->getAsTyped();

            if (arg0->isVector() && arg1->isVector()) {  // vec * vec
                node->getAsAggregate()->setOperator(EOpDot);
            } else {
                node = handleBinaryMath(loc, "mul", EOpMul, arg0, arg1);
            }

            break;
        }

    case EOpRcp:
        {
            // rcp(a) -> 1 / a
            TIntermTyped* arg0 = fnUnary->getOperand();
            TBasicType   type0 = arg0->getBasicType();
            TIntermTyped* one  = intermediate.addConstantUnion(1, type0, loc, true);
            node  = handleBinaryMath(loc, "rcp", EOpDiv, one, arg0);

            break;
        }

    case EOpAny: // fall through
    case EOpAll:
        {
            TIntermTyped* typedArg = arguments->getAsTyped();

            // HLSL allows float/etc types here, and the SPIR-V opcode requires a bool.
            // We'll convert here.  Note that for efficiency, we could add a smarter
            // decomposition for some type cases, e.g, maybe by decomposing a dot product.
            if (typedArg->getType().getBasicType() != EbtBool) {
                const TType boolType(EbtBool, EvqTemporary,
                                     typedArg->getVectorSize(),
                                     typedArg->getMatrixCols(),
                                     typedArg->getMatrixRows(),
                                     typedArg->isVector());

                typedArg = intermediate.addConversion(EOpConstructBool, boolType, typedArg);
                node->getAsUnaryNode()->setOperand(typedArg);
            }

            break;
        }

    case EOpSaturate:
        {
            // saturate(a) -> clamp(a,0,1)
            TIntermTyped* arg0 = fnUnary->getOperand();
            TBasicType   type0 = arg0->getBasicType();
            TIntermAggregate* clamp = new TIntermAggregate(EOpClamp);

            clamp->getSequence().push_back(arg0);
            clamp->getSequence().push_back(intermediate.addConstantUnion(0, type0, loc, true));
            clamp->getSequence().push_back(intermediate.addConstantUnion(1, type0, loc, true));
            clamp->setLoc(loc);
            clamp->setType(node->getType());
            clamp->getWritableType().getQualifier().makeTemporary();
            node = clamp;

            break;
        }

    case EOpSinCos:
        {
            // sincos(a,b,c) -> b = sin(a), c = cos(a)
            TIntermTyped* arg0 = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* arg1 = argAggregate->getSequence()[1]->getAsTyped();
            TIntermTyped* arg2 = argAggregate->getSequence()[2]->getAsTyped();

            TIntermTyped* sinStatement = handleUnaryMath(loc, "sin", EOpSin, arg0);
            TIntermTyped* cosStatement = handleUnaryMath(loc, "cos", EOpCos, arg0);
            TIntermTyped* sinAssign    = intermediate.addAssign(EOpAssign, arg1, sinStatement, loc);
            TIntermTyped* cosAssign    = intermediate.addAssign(EOpAssign, arg2, cosStatement, loc);

            TIntermAggregate* compoundStatement = intermediate.makeAggregate(sinAssign, loc);
            compoundStatement = intermediate.growAggregate(compoundStatement, cosAssign);
            compoundStatement->setOperator(EOpSequence);
            compoundStatement->setLoc(loc);
            compoundStatement->setType(TType(EbtVoid));

            node = compoundStatement;

            break;
        }

    case EOpClip:
        {
            // clip(a) -> if (any(a<0)) discard;
            TIntermTyped*  arg0 = fnUnary->getOperand();
            TBasicType     type0 = arg0->getBasicType();
            TIntermTyped*  compareNode = nullptr;

            // For non-scalars: per experiment with FXC compiler, discard if any component < 0.
            if (!arg0->isScalar()) {
                // component-wise compare: a < 0
                TIntermAggregate* less = new TIntermAggregate(EOpLessThan);
                less->getSequence().push_back(arg0);
                less->setLoc(loc);

                // make vec or mat of bool matching dimensions of input
                less->setType(TType(EbtBool, EvqTemporary,
                                    arg0->getType().getVectorSize(),
                                    arg0->getType().getMatrixCols(),
                                    arg0->getType().getMatrixRows(),
                                    arg0->getType().isVector()));

                // calculate # of components for comparison const
                const int constComponentCount =
                    std::max(arg0->getType().getVectorSize(), 1) *
                    std::max(arg0->getType().getMatrixCols(), 1) *
                    std::max(arg0->getType().getMatrixRows(), 1);

                TConstUnion zero;
                if (arg0->getType().isIntegerDomain())
                    zero.setDConst(0);
                else
                    zero.setDConst(0.0);
                TConstUnionArray zeros(constComponentCount, zero);

                less->getSequence().push_back(intermediate.addConstantUnion(zeros, arg0->getType(), loc, true));

                compareNode = intermediate.addBuiltInFunctionCall(loc, EOpAny, true, less, TType(EbtBool));
            } else {
                TIntermTyped* zero;
                if (arg0->getType().isIntegerDomain())
                    zero = intermediate.addConstantUnion(0, loc, true);
                else
                    zero = intermediate.addConstantUnion(0.0, type0, loc, true);
                compareNode = handleBinaryMath(loc, "clip", EOpLessThan, arg0, zero);
            }

            TIntermBranch* killNode = intermediate.addBranch(EOpKill, loc);

            node = new TIntermSelection(compareNode, killNode, nullptr);
            node->setLoc(loc);

            break;
        }

    case EOpLog10:
        {
            // log10(a) -> log2(a) * 0.301029995663981  (== 1/log2(10))
            TIntermTyped* arg0 = fnUnary->getOperand();
            TIntermTyped* log2 = handleUnaryMath(loc, "log2", EOpLog2, arg0);
            TIntermTyped* base = intermediate.addConstantUnion(0.301029995663981f, EbtFloat, loc, true);

            node  = handleBinaryMath(loc, "mul", EOpMul, log2, base);

            break;
        }

    case EOpDst:
        {
            // dest.x = 1;
            // dest.y = src0.y * src1.y;
            // dest.z = src0.z;
            // dest.w = src1.w;

            TIntermTyped* arg0 = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* arg1 = argAggregate->getSequence()[1]->getAsTyped();

            TIntermTyped* y = intermediate.addConstantUnion(1, loc, true);
            TIntermTyped* z = intermediate.addConstantUnion(2, loc, true);
            TIntermTyped* w = intermediate.addConstantUnion(3, loc, true);

            TIntermTyped* src0y = intermediate.addIndex(EOpIndexDirect, arg0, y, loc);
            TIntermTyped* src1y = intermediate.addIndex(EOpIndexDirect, arg1, y, loc);
            TIntermTyped* src0z = intermediate.addIndex(EOpIndexDirect, arg0, z, loc);
            TIntermTyped* src1w = intermediate.addIndex(EOpIndexDirect, arg1, w, loc);

            TIntermAggregate* dst = new TIntermAggregate(EOpConstructVec4);

            dst->getSequence().push_back(intermediate.addConstantUnion(1.0, EbtFloat, loc, true));
            dst->getSequence().push_back(handleBinaryMath(loc, "mul", EOpMul, src0y, src1y));
            dst->getSequence().push_back(src0z);
            dst->getSequence().push_back(src1w);
            dst->setType(TType(EbtFloat, EvqTemporary, 4));
            dst->setLoc(loc);
            node = dst;

            break;
        }

    case EOpInterlockedAdd: // optional last argument (if present) is assigned from return value
    case EOpInterlockedMin: // ...
    case EOpInterlockedMax: // ...
    case EOpInterlockedAnd: // ...
    case EOpInterlockedOr:  // ...
    case EOpInterlockedXor: // ...
    case EOpInterlockedExchange: // always has output arg
        {
            TIntermTyped* arg0 = argAggregate->getSequence()[0]->getAsTyped();  // dest
            TIntermTyped* arg1 = argAggregate->getSequence()[1]->getAsTyped();  // value
            TIntermTyped* arg2 = nullptr;

            if (argAggregate->getSequence().size() > 2)
                arg2 = argAggregate->getSequence()[2]->getAsTyped();

            const bool isImage = isImageParam(arg0);
            const TOperator atomicOp = mapAtomicOp(loc, op, isImage);
            TIntermAggregate* atomic = new TIntermAggregate(atomicOp);
            atomic->setType(arg0->getType());
            atomic->getWritableType().getQualifier().makeTemporary();
            atomic->setLoc(loc);

            if (isImage) {
                // orig_value = imageAtomicOp(image, loc, data)
                imageAtomicParams(atomic, arg0);
                atomic->getSequence().push_back(arg1);

                if (argAggregate->getSequence().size() > 2) {
                    node = intermediate.addAssign(EOpAssign, arg2, atomic, loc);
                } else {
                    node = atomic; // no assignment needed, as there was no out var.
                }
            } else {
                // Normal memory variable:
                // arg0 = mem, arg1 = data, arg2(optional,out) = orig_value
                if (argAggregate->getSequence().size() > 2) {
                    // optional output param is present.  return value goes to arg2.
                    atomic->getSequence().push_back(arg0);
                    atomic->getSequence().push_back(arg1);

                    node = intermediate.addAssign(EOpAssign, arg2, atomic, loc);
                } else {
                    // Set the matching operator.  Since output is absent, this is all we need to do.
                    node->getAsAggregate()->setOperator(atomicOp);
                    node->setType(atomic->getType());
                }
            }

            break;
        }

    case EOpInterlockedCompareExchange:
        {
            TIntermTyped* arg0 = argAggregate->getSequence()[0]->getAsTyped();  // dest
            TIntermTyped* arg1 = argAggregate->getSequence()[1]->getAsTyped();  // cmp
            TIntermTyped* arg2 = argAggregate->getSequence()[2]->getAsTyped();  // value
            TIntermTyped* arg3 = argAggregate->getSequence()[3]->getAsTyped();  // orig

            const bool isImage = isImageParam(arg0);
            TIntermAggregate* atomic = new TIntermAggregate(mapAtomicOp(loc, op, isImage));
            atomic->setLoc(loc);
            atomic->setType(arg2->getType());
            atomic->getWritableType().getQualifier().makeTemporary();

            if (isImage) {
                imageAtomicParams(atomic, arg0);
            } else {
                atomic->getSequence().push_back(arg0);
            }

            atomic->getSequence().push_back(arg1);
            atomic->getSequence().push_back(arg2);
            node = intermediate.addAssign(EOpAssign, arg3, atomic, loc);

            break;
        }

    case EOpEvaluateAttributeSnapped:
        {
            // SPIR-V InterpolateAtOffset uses float vec2 offset in pixels
            // HLSL uses int2 offset on a 16x16 grid in [-8..7] on x & y:
            //   iU = (iU<<28)>>28
            //   fU = ((float)iU)/16
            // Targets might handle this natively, in which case they can disable
            // decompositions.

            TIntermTyped* arg0 = argAggregate->getSequence()[0]->getAsTyped();  // value
            TIntermTyped* arg1 = argAggregate->getSequence()[1]->getAsTyped();  // offset

            TIntermTyped* i28 = intermediate.addConstantUnion(28, loc, true);
            TIntermTyped* iU = handleBinaryMath(loc, ">>", EOpRightShift,
                                                handleBinaryMath(loc, "<<", EOpLeftShift, arg1, i28),
                                                i28);

            TIntermTyped* recip16 = intermediate.addConstantUnion((1.0/16.0), EbtFloat, loc, true);
            TIntermTyped* floatOffset = handleBinaryMath(loc, "mul", EOpMul,
                                                         intermediate.addConversion(EOpConstructFloat,
                                                                                    TType(EbtFloat, EvqTemporary, 2), iU),
                                                         recip16);

            TIntermAggregate* interp = new TIntermAggregate(EOpInterpolateAtOffset);
            interp->getSequence().push_back(arg0);
            interp->getSequence().push_back(floatOffset);
            interp->setLoc(loc);
            interp->setType(arg0->getType());
            interp->getWritableType().getQualifier().makeTemporary();

            node = interp;

            break;
        }

    case EOpLit:
        {
            TIntermTyped* n_dot_l = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* n_dot_h = argAggregate->getSequence()[1]->getAsTyped();
            TIntermTyped* m = argAggregate->getSequence()[2]->getAsTyped();

            TIntermAggregate* dst = new TIntermAggregate(EOpConstructVec4);

            // Ambient
            dst->getSequence().push_back(intermediate.addConstantUnion(1.0, EbtFloat, loc, true));

            // Diffuse:
            TIntermTyped* zero = intermediate.addConstantUnion(0.0, EbtFloat, loc, true);
            TIntermAggregate* diffuse = new TIntermAggregate(EOpMax);
            diffuse->getSequence().push_back(n_dot_l);
            diffuse->getSequence().push_back(zero);
            diffuse->setLoc(loc);
            diffuse->setType(TType(EbtFloat));
            dst->getSequence().push_back(diffuse);

            // Specular:
            TIntermAggregate* min_ndot = new TIntermAggregate(EOpMin);
            min_ndot->getSequence().push_back(n_dot_l);
            min_ndot->getSequence().push_back(n_dot_h);
            min_ndot->setLoc(loc);
            min_ndot->setType(TType(EbtFloat));

            TIntermTyped* compare = handleBinaryMath(loc, "<", EOpLessThan, min_ndot, zero);
            TIntermTyped* n_dot_h_m = handleBinaryMath(loc, "mul", EOpMul, n_dot_h, m);  // n_dot_h * m

            dst->getSequence().push_back(intermediate.addSelection(compare, zero, n_dot_h_m, loc));

            // One:
            dst->getSequence().push_back(intermediate.addConstantUnion(1.0, EbtFloat, loc, true));

            dst->setLoc(loc);
            dst->setType(TType(EbtFloat, EvqTemporary, 4));
            node = dst;
            break;
        }

    case EOpAsDouble:
        {
            // asdouble accepts two 32 bit ints.  we can use EOpUint64BitsToDouble, but must
            // first construct a uint64.
            TIntermTyped* arg0 = argAggregate->getSequence()[0]->getAsTyped();
            TIntermTyped* arg1 = argAggregate->getSequence()[1]->getAsTyped();

            if (arg0->getType().isVector()) { // TODO: ...
                error(loc, "double2 conversion not implemented", "asdouble", "");
                break;
            }

            TIntermAggregate* uint64 = new TIntermAggregate(EOpConstructUVec2);

            uint64->getSequence().push_back(arg0);
            uint64->getSequence().push_back(arg1);
            uint64->setType(TType(EbtUint, EvqTemporary, 2));  // convert 2 uints to a uint2
            uint64->setLoc(loc);

            // bitcast uint2 to a double
            TIntermTyped* convert = new TIntermUnary(EOpUint64BitsToDouble);
            convert->getAsUnaryNode()->setOperand(uint64);
            convert->setLoc(loc);
            convert->setType(TType(EbtDouble, EvqTemporary));
            node = convert;

            break;
        }

    case EOpF16tof32:
        {
            // input uvecN with low 16 bits of each component holding a float16.  convert to float32.
            TIntermTyped* argValue = node->getAsUnaryNode()->getOperand();
            TIntermTyped* zero = intermediate.addConstantUnion(0, loc, true);
            const int vecSize = argValue->getType().getVectorSize();

            TOperator constructOp = EOpNull;
            switch (vecSize) {
            case 1: constructOp = EOpNull;          break; // direct use, no construct needed
            case 2: constructOp = EOpConstructVec2; break;
            case 3: constructOp = EOpConstructVec3; break;
            case 4: constructOp = EOpConstructVec4; break;
            default: assert(0); break;
            }

            // For scalar case, we don't need to construct another type.
            TIntermAggregate* result = (vecSize > 1) ? new TIntermAggregate(constructOp) : nullptr;

            if (result) {
                result->setType(TType(EbtFloat, EvqTemporary, vecSize));
                result->setLoc(loc);
            }

            for (int idx = 0; idx < vecSize; ++idx) {
                TIntermTyped* idxConst = intermediate.addConstantUnion(idx, loc, true);
                TIntermTyped* component = argValue->getType().isVector() ? 
                    intermediate.addIndex(EOpIndexDirect, argValue, idxConst, loc) : argValue;

                if (component != argValue)
                    component->setType(TType(argValue->getBasicType(), EvqTemporary));

                TIntermTyped* unpackOp  = new TIntermUnary(EOpUnpackHalf2x16);
                unpackOp->setType(TType(EbtFloat, EvqTemporary, 2));
                unpackOp->getAsUnaryNode()->setOperand(component);
                unpackOp->setLoc(loc);

                TIntermTyped* lowOrder  = intermediate.addIndex(EOpIndexDirect, unpackOp, zero, loc);
                
                if (result != nullptr) {
                    result->getSequence().push_back(lowOrder);
                    node = result;
                } else {
                    node = lowOrder;
                }
            }
            
            break;
        }

    case EOpF32tof16:
        {
            // input floatN converted to 16 bit float in low order bits of each component of uintN
            TIntermTyped* argValue = node->getAsUnaryNode()->getOperand();

            TIntermTyped* zero = intermediate.addConstantUnion(0.0, EbtFloat, loc, true);
            const int vecSize = argValue->getType().getVectorSize();

            TOperator constructOp = EOpNull;
            switch (vecSize) {
            case 1: constructOp = EOpNull;           break; // direct use, no construct needed
            case 2: constructOp = EOpConstructUVec2; break;
            case 3: constructOp = EOpConstructUVec3; break;
            case 4: constructOp = EOpConstructUVec4; break;
            default: assert(0); break;
            }

            // For scalar case, we don't need to construct another type.
            TIntermAggregate* result = (vecSize > 1) ? new TIntermAggregate(constructOp) : nullptr;

            if (result) {
                result->setType(TType(EbtUint, EvqTemporary, vecSize));
                result->setLoc(loc);
            }

            for (int idx = 0; idx < vecSize; ++idx) {
                TIntermTyped* idxConst = intermediate.addConstantUnion(idx, loc, true);
                TIntermTyped* component = argValue->getType().isVector() ? 
                    intermediate.addIndex(EOpIndexDirect, argValue, idxConst, loc) : argValue;

                if (component != argValue)
                    component->setType(TType(argValue->getBasicType(), EvqTemporary));

                TIntermAggregate* vec2ComponentAndZero = new TIntermAggregate(EOpConstructVec2);
                vec2ComponentAndZero->getSequence().push_back(component);
                vec2ComponentAndZero->getSequence().push_back(zero);
                vec2ComponentAndZero->setType(TType(EbtFloat, EvqTemporary, 2));
                vec2ComponentAndZero->setLoc(loc);
                
                TIntermTyped* packOp = new TIntermUnary(EOpPackHalf2x16);
                packOp->getAsUnaryNode()->setOperand(vec2ComponentAndZero);
                packOp->setLoc(loc);
                packOp->setType(TType(EbtUint, EvqTemporary));

                if (result != nullptr) {
                    result->getSequence().push_back(packOp);
                    node = result;
                } else {
                    node = packOp;
                }
            }

            break;
        }

    case EOpD3DCOLORtoUBYTE4:
        {
            // ivec4 ( x.zyxw * 255.001953 );
            TIntermTyped* arg0 = node->getAsUnaryNode()->getOperand();
            TSwizzleSelectors<TVectorSelector> selectors;
            selectors.push_back(2);
            selectors.push_back(1);
            selectors.push_back(0);
            selectors.push_back(3);
            TIntermTyped* swizzleIdx = intermediate.addSwizzle(selectors, loc);
            TIntermTyped* swizzled = intermediate.addIndex(EOpVectorSwizzle, arg0, swizzleIdx, loc);
            swizzled->setType(arg0->getType());
            swizzled->getWritableType().getQualifier().makeTemporary();

            TIntermTyped* conversion = intermediate.addConstantUnion(255.001953f, EbtFloat, loc, true);
            TIntermTyped* rangeConverted = handleBinaryMath(loc, "mul", EOpMul, conversion, swizzled);
            rangeConverted->setType(arg0->getType());
            rangeConverted->getWritableType().getQualifier().makeTemporary();

            node = intermediate.addConversion(EOpConstructInt, TType(EbtInt, EvqTemporary, 4), rangeConverted);
            node->setLoc(loc);
            node->setType(TType(EbtInt, EvqTemporary, 4));
            break;
        }

    case EOpIsFinite:
        {
            // Since OPIsFinite in SPIR-V is only supported with the Kernel capability, we translate
            // it to !isnan && !isinf

            TIntermTyped* arg0 = node->getAsUnaryNode()->getOperand();

            // We'll make a temporary in case the RHS is cmoplex
            TVariable* tempArg = makeInternalVariable("@finitetmp", arg0->getType());
            tempArg->getWritableType().getQualifier().makeTemporary();

            TIntermTyped* tmpArgAssign = intermediate.addAssign(EOpAssign,
                                                                intermediate.addSymbol(*tempArg, loc),
                                                                arg0, loc);

            TIntermAggregate* compoundStatement = intermediate.makeAggregate(tmpArgAssign, loc);

            const TType boolType(EbtBool, EvqTemporary, arg0->getVectorSize(), arg0->getMatrixCols(),
                                 arg0->getMatrixRows());

            TIntermTyped* isnan = handleUnaryMath(loc, "isnan", EOpIsNan, intermediate.addSymbol(*tempArg, loc));
            isnan->setType(boolType);

            TIntermTyped* notnan = handleUnaryMath(loc, "!", EOpLogicalNot, isnan);
            notnan->setType(boolType);

            TIntermTyped* isinf = handleUnaryMath(loc, "isinf", EOpIsInf, intermediate.addSymbol(*tempArg, loc));
            isinf->setType(boolType);

            TIntermTyped* notinf = handleUnaryMath(loc, "!", EOpLogicalNot, isinf);
            notinf->setType(boolType);
            
            TIntermTyped* andNode = handleBinaryMath(loc, "and", EOpLogicalAnd, notnan, notinf);
            andNode->setType(boolType);

            compoundStatement = intermediate.growAggregate(compoundStatement, andNode);
            compoundStatement->setOperator(EOpSequence);
            compoundStatement->setLoc(loc);
            compoundStatement->setType(boolType);

            node = compoundStatement;

            break;
        }
    case EOpWaveGetLaneCount:
        {
            // Mapped to gl_SubgroupSize builtin (We preprend @ to the symbol
            // so that it inhabits the symbol table, but has a user-invalid name
            // in-case some source HLSL defined the symbol also).
            TType type(EbtUint, EvqVaryingIn);
            node = lookupBuiltinVariable("@gl_SubgroupSize", EbvSubgroupSize2, type);
            break;
        }
    case EOpWaveGetLaneIndex:
        {
            // Mapped to gl_SubgroupInvocationID builtin (We preprend @ to the
            // symbol so that it inhabits the symbol table, but has a
            // user-invalid name in-case some source HLSL defined the symbol
            // also).
            TType type(EbtUint, EvqVaryingIn);
            node = lookupBuiltinVariable("@gl_SubgroupInvocationID", EbvSubgroupInvocation2, type);
            break;
        }
    case EOpWaveActiveCountBits:
        {
            // Mapped to subgroupBallotBitCount(subgroupBallot()) builtin

            // uvec4 type.
            TType uvec4Type(EbtUint, EvqTemporary, 4);

            // Get the uvec4 return from subgroupBallot().
            TIntermTyped* res = intermediate.addBuiltInFunctionCall(loc,
                EOpSubgroupBallot, true, arguments, uvec4Type);

            // uint type.
            TType uintType(EbtUint, EvqTemporary);

            node = intermediate.addBuiltInFunctionCall(loc,
                EOpSubgroupBallotBitCount, true, res, uintType);

            break;
        }
    case EOpWavePrefixCountBits:
        {
            // Mapped to subgroupBallotInclusiveBitCount(subgroupBallot())
            // builtin

            // uvec4 type.
            TType uvec4Type(EbtUint, EvqTemporary, 4);

            // Get the uvec4 return from subgroupBallot().
            TIntermTyped* res = intermediate.addBuiltInFunctionCall(loc,
                EOpSubgroupBallot, true, arguments, uvec4Type);

            // uint type.
            TType uintType(EbtUint, EvqTemporary);

            node = intermediate.addBuiltInFunctionCall(loc,
                EOpSubgroupBallotInclusiveBitCount, true, res, uintType);

            break;
        }

    default:
        break; // most pass through unchanged
    }
}

//
// Handle seeing function call syntax in the grammar, which could be any of
//  - .length() method
//  - constructor
//  - a call to a built-in function mapped to an operator
//  - a call to a built-in function that will remain a function call (e.g., texturing)
//  - user function
//  - subroutine call (not implemented yet)
//
TIntermTyped* HlslParseContext::handleFunctionCall(const TSourceLoc& loc, TFunction* function, TIntermTyped* arguments)
{
    TIntermTyped* result = nullptr;

    TOperator op = function->getBuiltInOp();
    if (op != EOpNull) {
        //
        // Then this should be a constructor.
        // Don't go through the symbol table for constructors.
        // Their parameters will be verified algorithmically.
        //
        TType type(EbtVoid);  // use this to get the type back
        if (! constructorError(loc, arguments, *function, op, type)) {
            //
            // It's a constructor, of type 'type'.
            //
            result = handleConstructor(loc, arguments, type);
            if (result == nullptr) {
                error(loc, "cannot construct with these arguments", type.getCompleteString().c_str(), "");
                return nullptr;
            }
        }
    } else {
        //
        // Find it in the symbol table.
        //
        const TFunction* fnCandidate = nullptr;
        bool builtIn = false;
        int thisDepth = 0;

        // For mat mul, the situation is unusual: we have to compare vector sizes to mat row or col sizes,
        // and clamp the opposite arg.  Since that's complex, we farm it off to a separate method.
        // It doesn't naturally fall out of processing an argument at a time in isolation.
        if (function->getName() == "mul")
            addGenMulArgumentConversion(loc, *function, arguments);

        TIntermAggregate* aggregate = arguments ? arguments->getAsAggregate() : nullptr;

        // TODO: this needs improvement: there's no way at present to look up a signature in
        // the symbol table for an arbitrary type.  This is a temporary hack until that ability exists.
        // It will have false positives, since it doesn't check arg counts or types.
        if (arguments) {
            // Check if first argument is struct buffer type.  It may be an aggregate or a symbol, so we
            // look for either case.

            TIntermTyped* arg0 = nullptr;

            if (aggregate && aggregate->getSequence().size() > 0 && aggregate->getSequence()[0])
                arg0 = aggregate->getSequence()[0]->getAsTyped();
            else if (arguments->getAsSymbolNode())
                arg0 = arguments->getAsSymbolNode();

            if (arg0 != nullptr && isStructBufferType(arg0->getType())) {
                static const int methodPrefixSize = sizeof(BUILTIN_PREFIX)-1;

                if (function->getName().length() > methodPrefixSize &&
                    isStructBufferMethod(function->getName().substr(methodPrefixSize))) {
                    const TString mangle = function->getName() + "(";
                    TSymbol* symbol = symbolTable.find(mangle, &builtIn);

                    if (symbol)
                        fnCandidate = symbol->getAsFunction();
                }
            }
        }

        if (fnCandidate == nullptr)
            fnCandidate = findFunction(loc, *function, builtIn, thisDepth, arguments);

        if (fnCandidate) {
            // This is a declared function that might map to
            //  - a built-in operator,
            //  - a built-in function not mapped to an operator, or
            //  - a user function.

            // turn an implicit member-function resolution into an explicit call
            TString callerName;
            if (thisDepth == 0)
                callerName = fnCandidate->getMangledName();
            else {
                // get the explicit (full) name of the function
                callerName = currentTypePrefix[currentTypePrefix.size() - thisDepth];
                callerName += fnCandidate->getMangledName();
                // insert the implicit calling argument
                pushFrontArguments(intermediate.addSymbol(*getImplicitThis(thisDepth)), arguments);
            }

            // Convert 'in' arguments, so that types match.
            // However, skip those that need expansion, that is covered next.
            if (arguments)
                addInputArgumentConversions(*fnCandidate, arguments);

            // Expand arguments.  Some arguments must physically expand to a different set
            // than what the shader declared and passes.
            if (arguments && !builtIn)
                expandArguments(loc, *fnCandidate, arguments);

            // Expansion may have changed the form of arguments
            aggregate = arguments ? arguments->getAsAggregate() : nullptr;

            op = fnCandidate->getBuiltInOp();
            if (builtIn && op != EOpNull) {
                // A function call mapped to a built-in operation.
                result = intermediate.addBuiltInFunctionCall(loc, op, fnCandidate->getParamCount() == 1, arguments,
                                                             fnCandidate->getType());
                if (result == nullptr)  {
                    error(arguments->getLoc(), " wrong operand type", "Internal Error",
                        "built in unary operator function.  Type: %s",
                        static_cast<TIntermTyped*>(arguments)->getCompleteString().c_str());
                } else if (result->getAsOperator()) {
                    builtInOpCheck(loc, *fnCandidate, *result->getAsOperator());
                }
            } else {
                // This is a function call not mapped to built-in operator.
                // It could still be a built-in function, but only if PureOperatorBuiltins == false.
                result = intermediate.setAggregateOperator(arguments, EOpFunctionCall, fnCandidate->getType(), loc);
                TIntermAggregate* call = result->getAsAggregate();
                call->setName(callerName);

                // this is how we know whether the given function is a built-in function or a user-defined function
                // if builtIn == false, it's a userDefined -> could be an overloaded built-in function also
                // if builtIn == true, it's definitely a built-in function with EOpNull
                if (! builtIn) {
                    call->setUserDefined();
                    intermediate.addToCallGraph(infoSink, currentCaller, callerName);
                }
            }

            // for decompositions, since we want to operate on the function node, not the aggregate holding
            // output conversions.
            const TIntermTyped* fnNode = result;

            decomposeStructBufferMethods(loc, result, arguments); // HLSL->AST struct buffer method decompositions
            decomposeIntrinsic(loc, result, arguments);           // HLSL->AST intrinsic decompositions
            decomposeSampleMethods(loc, result, arguments);       // HLSL->AST sample method decompositions
            decomposeGeometryMethods(loc, result, arguments);     // HLSL->AST geometry method decompositions

            // Create the qualifier list, carried in the AST for the call.
            // Because some arguments expand to multiple arguments, the qualifier list will
            // be longer than the formal parameter list.
            if (result == fnNode && result->getAsAggregate()) {
                TQualifierList& qualifierList = result->getAsAggregate()->getQualifierList();
                for (int i = 0; i < fnCandidate->getParamCount(); ++i) {
                    TStorageQualifier qual = (*fnCandidate)[i].type->getQualifier().storage;
                    if (hasStructBuffCounter(*(*fnCandidate)[i].type)) {
                        // add buffer and counter buffer argument qualifier
                        qualifierList.push_back(qual);
                        qualifierList.push_back(qual);
                    } else if (shouldFlatten(*(*fnCandidate)[i].type, (*fnCandidate)[i].type->getQualifier().storage,
                                             true)) {
                        // add structure member expansion
                        for (int memb = 0; memb < (int)(*fnCandidate)[i].type->getStruct()->size(); ++memb)
                            qualifierList.push_back(qual);
                    } else {
                        // Normal 1:1 case
                        qualifierList.push_back(qual);
                    }
                }
            }

            // Convert 'out' arguments.  If it was a constant folded built-in, it won't be an aggregate anymore.
            // Built-ins with a single argument aren't called with an aggregate, but they also don't have an output.
            // Also, build the qualifier list for user function calls, which are always called with an aggregate.
            // We don't do this is if there has been a decomposition, which will have added its own conversions
            // for output parameters.
            if (result == fnNode && result->getAsAggregate())
                result = addOutputArgumentConversions(*fnCandidate, *result->getAsOperator());
        }
    }

    // generic error recovery
    // TODO: simplification: localize all the error recoveries that look like this, and taking type into account to
    //       reduce cascades
    if (result == nullptr)
        result = intermediate.addConstantUnion(0.0, EbtFloat, loc);

    return result;
}

// An initial argument list is difficult: it can be null, or a single node,
// or an aggregate if more than one argument.  Add one to the front, maintaining
// this lack of uniformity.
void HlslParseContext::pushFrontArguments(TIntermTyped* front, TIntermTyped*& arguments)
{
    if (arguments == nullptr)
        arguments = front;
    else if (arguments->getAsAggregate() != nullptr)
        arguments->getAsAggregate()->getSequence().insert(arguments->getAsAggregate()->getSequence().begin(), front);
    else
        arguments = intermediate.growAggregate(front, arguments);
}

//
// HLSL allows mismatched dimensions on vec*mat, mat*vec, vec*vec, and mat*mat.  This is a
// situation not well suited to resolution in intrinsic selection, but we can do so here, since we
// can look at both arguments insert explicit shape changes if required.
//
void HlslParseContext::addGenMulArgumentConversion(const TSourceLoc& loc, TFunction& call, TIntermTyped*& args)
{
    TIntermAggregate* argAggregate = args ? args->getAsAggregate() : nullptr;

    if (argAggregate == nullptr || argAggregate->getSequence().size() != 2) {
        // It really ought to have two arguments.
        error(loc, "expected: mul arguments", "", "");
        return;
    }

    TIntermTyped* arg0 = argAggregate->getSequence()[0]->getAsTyped();
    TIntermTyped* arg1 = argAggregate->getSequence()[1]->getAsTyped();

    if (arg0->isVector() && arg1->isVector()) {
        // For:
        //    vec * vec: it's handled during intrinsic selection, so while we could do it here,
        //               we can also ignore it, which is easier.
    } else if (arg0->isVector() && arg1->isMatrix()) {
        // vec * mat: we clamp the vec if the mat col is smaller, else clamp the mat col.
        if (arg0->getVectorSize() < arg1->getMatrixCols()) {
            // vec is smaller, so truncate larger mat dimension
            const TType truncType(arg1->getBasicType(), arg1->getQualifier().storage, arg1->getQualifier().precision,
                                  0, arg0->getVectorSize(), arg1->getMatrixRows());
            arg1 = addConstructor(loc, arg1, truncType);
        } else if (arg0->getVectorSize() > arg1->getMatrixCols()) {
            // vec is larger, so truncate vec to mat size
            const TType truncType(arg0->getBasicType(), arg0->getQualifier().storage, arg0->getQualifier().precision,
                                  arg1->getMatrixCols());
            arg0 = addConstructor(loc, arg0, truncType);
        }
    } else if (arg0->isMatrix() && arg1->isVector()) {
        // mat * vec: we clamp the vec if the mat col is smaller, else clamp the mat col.
        if (arg1->getVectorSize() < arg0->getMatrixRows()) {
            // vec is smaller, so truncate larger mat dimension
            const TType truncType(arg0->getBasicType(), arg0->getQualifier().storage, arg0->getQualifier().precision,
                                  0, arg0->getMatrixCols(), arg1->getVectorSize());
            arg0 = addConstructor(loc, arg0, truncType);
        } else if (arg1->getVectorSize() > arg0->getMatrixRows()) {
            // vec is larger, so truncate vec to mat size
            const TType truncType(arg1->getBasicType(), arg1->getQualifier().storage, arg1->getQualifier().precision,
                                  arg0->getMatrixRows());
            arg1 = addConstructor(loc, arg1, truncType);
        }
    } else if (arg0->isMatrix() && arg1->isMatrix()) {
        // mat * mat: we clamp the smaller inner dimension to match the other matrix size.
        // Remember, HLSL Mrc = GLSL/SPIRV Mcr.
        if (arg0->getMatrixRows() > arg1->getMatrixCols()) {
            const TType truncType(arg0->getBasicType(), arg0->getQualifier().storage, arg0->getQualifier().precision,
                                  0, arg0->getMatrixCols(), arg1->getMatrixCols());
            arg0 = addConstructor(loc, arg0, truncType);
        } else if (arg0->getMatrixRows() < arg1->getMatrixCols()) {
            const TType truncType(arg1->getBasicType(), arg1->getQualifier().storage, arg1->getQualifier().precision,
                                  0, arg0->getMatrixRows(), arg1->getMatrixRows());
            arg1 = addConstructor(loc, arg1, truncType);
        }
    } else {
        // It's something with scalars: we'll just leave it alone.  Function selection will handle it
        // downstream.
    }

    // Warn if we altered one of the arguments
    if (arg0 != argAggregate->getSequence()[0] || arg1 != argAggregate->getSequence()[1])
        warn(loc, "mul() matrix size mismatch", "", "");

    // Put arguments back.  (They might be unchanged, in which case this is harmless).
    argAggregate->getSequence()[0] = arg0;
    argAggregate->getSequence()[1] = arg1;

    call[0].type = &arg0->getWritableType();
    call[1].type = &arg1->getWritableType();
}

//
// Add any needed implicit conversions for function-call arguments to input parameters.
//
void HlslParseContext::addInputArgumentConversions(const TFunction& function, TIntermTyped*& arguments)
{
    TIntermAggregate* aggregate = arguments->getAsAggregate();

    // Replace a single argument with a single argument.
    const auto setArg = [&](int paramNum, TIntermTyped* arg) {
        if (function.getParamCount() == 1)
            arguments = arg;
        else {
            if (aggregate == nullptr)
                arguments = arg;
            else
                aggregate->getSequence()[paramNum] = arg;
        }
    };

    // Process each argument's conversion
    for (int param = 0; param < function.getParamCount(); ++param) {
        if (! function[param].type->getQualifier().isParamInput())
            continue;

        // At this early point there is a slight ambiguity between whether an aggregate 'arguments'
        // is the single argument itself or its children are the arguments.  Only one argument
        // means take 'arguments' itself as the one argument.
        TIntermTyped* arg = function.getParamCount() == 1
                                   ? arguments->getAsTyped()
                                   : (aggregate ? 
                                        aggregate->getSequence()[param]->getAsTyped() :
                                        arguments->getAsTyped());
        if (*function[param].type != arg->getType()) {
            // In-qualified arguments just need an extra node added above the argument to
            // convert to the correct type.
            TIntermTyped* convArg = intermediate.addConversion(EOpFunctionCall, *function[param].type, arg);
            if (convArg != nullptr)
                convArg = intermediate.addUniShapeConversion(EOpFunctionCall, *function[param].type, convArg);
            if (convArg != nullptr)
                setArg(param, convArg);
            else
                error(arg->getLoc(), "cannot convert input argument, argument", "", "%d", param);
        } else {
            if (wasFlattened(arg)) {
                // If both formal and calling arg are to be flattened, leave that to argument
                // expansion, not conversion.
                if (!shouldFlatten(*function[param].type, function[param].type->getQualifier().storage, true)) {
                    // Will make a two-level subtree.
                    // The deepest will copy member-by-member to build the structure to pass.
                    // The level above that will be a two-operand EOpComma sequence that follows the copy by the
                    // object itself.
                    TVariable* internalAggregate = makeInternalVariable("aggShadow", *function[param].type);
                    internalAggregate->getWritableType().getQualifier().makeTemporary();
                    TIntermSymbol* internalSymbolNode = new TIntermSymbol(internalAggregate->getUniqueId(),
                                                                          internalAggregate->getName(),
                                                                          internalAggregate->getType());
                    internalSymbolNode->setLoc(arg->getLoc());
                    // This makes the deepest level, the member-wise copy
                    TIntermAggregate* assignAgg = handleAssign(arg->getLoc(), EOpAssign,
                                                               internalSymbolNode, arg)->getAsAggregate();

                    // Now, pair that with the resulting aggregate.
                    assignAgg = intermediate.growAggregate(assignAgg, internalSymbolNode, arg->getLoc());
                    assignAgg->setOperator(EOpComma);
                    assignAgg->setType(internalAggregate->getType());
                    setArg(param, assignAgg);
                }
            }
        }
    }
}

//
// Add any needed implicit expansion of calling arguments from what the shader listed to what's
// internally needed for the AST (given the constraints downstream).
//
void HlslParseContext::expandArguments(const TSourceLoc& loc, const TFunction& function, TIntermTyped*& arguments)
{
    TIntermAggregate* aggregate = arguments->getAsAggregate();
    int functionParamNumberOffset = 0;

    // Replace a single argument with a single argument.
    const auto setArg = [&](int paramNum, TIntermTyped* arg) {
        if (function.getParamCount() + functionParamNumberOffset == 1)
            arguments = arg;
        else {
            if (aggregate == nullptr)
                arguments = arg;
            else
                aggregate->getSequence()[paramNum] = arg;
        }
    };

    // Replace a single argument with a list of arguments
    const auto setArgList = [&](int paramNum, const TVector<TIntermTyped*>& args) {
        if (args.size() == 1)
            setArg(paramNum, args.front());
        else if (args.size() > 1) {
            if (function.getParamCount() + functionParamNumberOffset == 1) {
                arguments = intermediate.makeAggregate(args.front());
                std::for_each(args.begin() + 1, args.end(), 
                    [&](TIntermTyped* arg) {
                        arguments = intermediate.growAggregate(arguments, arg);
                    });
            } else {
                auto it = aggregate->getSequence().erase(aggregate->getSequence().begin() + paramNum);
                aggregate->getSequence().insert(it, args.begin(), args.end());
            }
            functionParamNumberOffset += (int)(args.size() - 1);
        }
    };

    // Process each argument's conversion
    for (int param = 0; param < function.getParamCount(); ++param) {
        // At this early point there is a slight ambiguity between whether an aggregate 'arguments'
        // is the single argument itself or its children are the arguments.  Only one argument
        // means take 'arguments' itself as the one argument.
        TIntermTyped* arg = function.getParamCount() == 1
                                   ? arguments->getAsTyped()
                                   : (aggregate ? 
                                        aggregate->getSequence()[param + functionParamNumberOffset]->getAsTyped() :
                                        arguments->getAsTyped());

        if (wasFlattened(arg) && shouldFlatten(*function[param].type, function[param].type->getQualifier().storage, true)) {
            // Need to pass the structure members instead of the structure.
            TVector<TIntermTyped*> memberArgs;
            for (int memb = 0; memb < (int)arg->getType().getStruct()->size(); ++memb)
                memberArgs.push_back(flattenAccess(arg, memb));
            setArgList(param + functionParamNumberOffset, memberArgs);
        }
    }

    // TODO: if we need both hidden counter args (below) and struct expansion (above)
    // the two algorithms need to be merged: Each assumes the list starts out 1:1 between
    // parameters and arguments.

    // If any argument is a pass-by-reference struct buffer with an associated counter
    // buffer, we have to add another hidden parameter for that counter.
    if (aggregate)
        addStructBuffArguments(loc, aggregate);
}

//
// Add any needed implicit output conversions for function-call arguments.  This
// can require a new tree topology, complicated further by whether the function
// has a return value.
//
// Returns a node of a subtree that evaluates to the return value of the function.
//
TIntermTyped* HlslParseContext::addOutputArgumentConversions(const TFunction& function, TIntermOperator& intermNode)
{
    assert (intermNode.getAsAggregate() != nullptr || intermNode.getAsUnaryNode() != nullptr);

    const TSourceLoc& loc = intermNode.getLoc();

    TIntermSequence argSequence; // temp sequence for unary node args

    if (intermNode.getAsUnaryNode())
        argSequence.push_back(intermNode.getAsUnaryNode()->getOperand());

    TIntermSequence& arguments = argSequence.empty() ? intermNode.getAsAggregate()->getSequence() : argSequence;

    const auto needsConversion = [&](int argNum) {
        return function[argNum].type->getQualifier().isParamOutput() &&
               (*function[argNum].type != arguments[argNum]->getAsTyped()->getType() ||
                shouldConvertLValue(arguments[argNum]) ||
                wasFlattened(arguments[argNum]->getAsTyped()));
    };

    // Will there be any output conversions?
    bool outputConversions = false;
    for (int i = 0; i < function.getParamCount(); ++i) {
        if (needsConversion(i)) {
            outputConversions = true;
            break;
        }
    }

    if (! outputConversions)
        return &intermNode;

    // Setup for the new tree, if needed:
    //
    // Output conversions need a different tree topology.
    // Out-qualified arguments need a temporary of the correct type, with the call
    // followed by an assignment of the temporary to the original argument:
    //     void: function(arg, ...)  ->        (          function(tempArg, ...), arg = tempArg, ...)
    //     ret = function(arg, ...)  ->  ret = (tempRet = function(tempArg, ...), arg = tempArg, ..., tempRet)
    // Where the "tempArg" type needs no conversion as an argument, but will convert on assignment.
    TIntermTyped* conversionTree = nullptr;
    TVariable* tempRet = nullptr;
    if (intermNode.getBasicType() != EbtVoid) {
        // do the "tempRet = function(...), " bit from above
        tempRet = makeInternalVariable("tempReturn", intermNode.getType());
        TIntermSymbol* tempRetNode = intermediate.addSymbol(*tempRet, loc);
        conversionTree = intermediate.addAssign(EOpAssign, tempRetNode, &intermNode, loc);
    } else
        conversionTree = &intermNode;

    conversionTree = intermediate.makeAggregate(conversionTree);

    // Process each argument's conversion
    for (int i = 0; i < function.getParamCount(); ++i) {
        if (needsConversion(i)) {
            // Out-qualified arguments needing conversion need to use the topology setup above.
            // Do the " ...(tempArg, ...), arg = tempArg" bit from above.

            // Make a temporary for what the function expects the argument to look like.
            TVariable* tempArg = makeInternalVariable("tempArg", *function[i].type);
            tempArg->getWritableType().getQualifier().makeTemporary();
            TIntermSymbol* tempArgNode = intermediate.addSymbol(*tempArg, loc);

            // This makes the deepest level, the member-wise copy
            TIntermTyped* tempAssign = handleAssign(arguments[i]->getLoc(), EOpAssign, arguments[i]->getAsTyped(),
                                                    tempArgNode);
            tempAssign = handleLvalue(arguments[i]->getLoc(), "assign", tempAssign);
            conversionTree = intermediate.growAggregate(conversionTree, tempAssign, arguments[i]->getLoc());

            // replace the argument with another node for the same tempArg variable
            arguments[i] = intermediate.addSymbol(*tempArg, loc);
        }
    }

    // Finalize the tree topology (see bigger comment above).
    if (tempRet) {
        // do the "..., tempRet" bit from above
        TIntermSymbol* tempRetNode = intermediate.addSymbol(*tempRet, loc);
        conversionTree = intermediate.growAggregate(conversionTree, tempRetNode, loc);
    }

    conversionTree = intermediate.setAggregateOperator(conversionTree, EOpComma, intermNode.getType(), loc);

    return conversionTree;
}

//
// Add any needed "hidden" counter buffer arguments for function calls.
//
// Modifies the 'aggregate' argument if needed.  Otherwise, is no-op.
//
void HlslParseContext::addStructBuffArguments(const TSourceLoc& loc, TIntermAggregate*& aggregate)
{
    // See if there are any SB types with counters.
    const bool hasStructBuffArg =
        std::any_of(aggregate->getSequence().begin(),
                    aggregate->getSequence().end(),
                    [this](const TIntermNode* node) {
                        return (node && node->getAsTyped() != nullptr) && hasStructBuffCounter(node->getAsTyped()->getType());
                    });

    // Nothing to do, if we didn't find one.
    if (! hasStructBuffArg)
        return;

    TIntermSequence argsWithCounterBuffers;

    for (int param = 0; param < int(aggregate->getSequence().size()); ++param) {
        argsWithCounterBuffers.push_back(aggregate->getSequence()[param]);

        if (hasStructBuffCounter(aggregate->getSequence()[param]->getAsTyped()->getType())) {
            const TIntermSymbol* blockSym = aggregate->getSequence()[param]->getAsSymbolNode();
            if (blockSym != nullptr) {
                TType counterType;
                counterBufferType(loc, counterType);

                const TString counterBlockName(intermediate.addCounterBufferName(blockSym->getName()));

                TVariable* variable = makeInternalVariable(counterBlockName, counterType);

                // Mark this buffer's counter block as being in use
                structBufferCounter[counterBlockName] = true;

                TIntermSymbol* sym = intermediate.addSymbol(*variable, loc);
                argsWithCounterBuffers.push_back(sym);
            }
        }
    }

    // Swap with the temp list we've built up.
    aggregate->getSequence().swap(argsWithCounterBuffers);
}


//
// Do additional checking of built-in function calls that is not caught
// by normal semantic checks on argument type, extension tagging, etc.
//
// Assumes there has been a semantically correct match to a built-in function prototype.
//
void HlslParseContext::builtInOpCheck(const TSourceLoc& loc, const TFunction& fnCandidate, TIntermOperator& callNode)
{
    // Set up convenience accessors to the argument(s).  There is almost always
    // multiple arguments for the cases below, but when there might be one,
    // check the unaryArg first.
    const TIntermSequence* argp = nullptr;   // confusing to use [] syntax on a pointer, so this is to help get a reference
    const TIntermTyped* unaryArg = nullptr;
    const TIntermTyped* arg0 = nullptr;
    if (callNode.getAsAggregate()) {
        argp = &callNode.getAsAggregate()->getSequence();
        if (argp->size() > 0)
            arg0 = (*argp)[0]->getAsTyped();
    } else {
        assert(callNode.getAsUnaryNode());
        unaryArg = callNode.getAsUnaryNode()->getOperand();
        arg0 = unaryArg;
    }
    const TIntermSequence& aggArgs = *argp;  // only valid when unaryArg is nullptr

    switch (callNode.getOp()) {
    case EOpTextureGather:
    case EOpTextureGatherOffset:
    case EOpTextureGatherOffsets:
    {
        // Figure out which variants are allowed by what extensions,
        // and what arguments must be constant for which situations.

        TString featureString = fnCandidate.getName() + "(...)";
        const char* feature = featureString.c_str();
        int compArg = -1;  // track which argument, if any, is the constant component argument
        switch (callNode.getOp()) {
        case EOpTextureGather:
            // More than two arguments needs gpu_shader5, and rectangular or shadow needs gpu_shader5,
            // otherwise, need GL_ARB_texture_gather.
            if (fnCandidate.getParamCount() > 2 || fnCandidate[0].type->getSampler().dim == EsdRect ||
                fnCandidate[0].type->getSampler().shadow) {
                if (! fnCandidate[0].type->getSampler().shadow)
                    compArg = 2;
            }
            break;
        case EOpTextureGatherOffset:
            // GL_ARB_texture_gather is good enough for 2D non-shadow textures with no component argument
            if (! fnCandidate[0].type->getSampler().shadow)
                compArg = 3;
            break;
        case EOpTextureGatherOffsets:
            if (! fnCandidate[0].type->getSampler().shadow)
                compArg = 3;
            break;
        default:
            break;
        }

        if (compArg > 0 && compArg < fnCandidate.getParamCount()) {
            if (aggArgs[compArg]->getAsConstantUnion()) {
                int value = aggArgs[compArg]->getAsConstantUnion()->getConstArray()[0].getIConst();
                if (value < 0 || value > 3)
                    error(loc, "must be 0, 1, 2, or 3:", feature, "component argument");
            } else
                error(loc, "must be a compile-time constant:", feature, "component argument");
        }

        break;
    }

    case EOpTextureOffset:
    case EOpTextureFetchOffset:
    case EOpTextureProjOffset:
    case EOpTextureLodOffset:
    case EOpTextureProjLodOffset:
    case EOpTextureGradOffset:
    case EOpTextureProjGradOffset:
    {
        // Handle texture-offset limits checking
        // Pick which argument has to hold constant offsets
        int arg = -1;
        switch (callNode.getOp()) {
        case EOpTextureOffset:          arg = 2;  break;
        case EOpTextureFetchOffset:     arg = (arg0->getType().getSampler().dim != EsdRect) ? 3 : 2; break;
        case EOpTextureProjOffset:      arg = 2;  break;
        case EOpTextureLodOffset:       arg = 3;  break;
        case EOpTextureProjLodOffset:   arg = 3;  break;
        case EOpTextureGradOffset:      arg = 4;  break;
        case EOpTextureProjGradOffset:  arg = 4;  break;
        default:
            assert(0);
            break;
        }

        if (arg > 0) {
            if (aggArgs[arg]->getAsConstantUnion() == nullptr)
                error(loc, "argument must be compile-time constant", "texel offset", "");
            else {
                const TType& type = aggArgs[arg]->getAsTyped()->getType();
                for (int c = 0; c < type.getVectorSize(); ++c) {
                    int offset = aggArgs[arg]->getAsConstantUnion()->getConstArray()[c].getIConst();
                    if (offset > resources.maxProgramTexelOffset || offset < resources.minProgramTexelOffset)
                        error(loc, "value is out of range:", "texel offset",
                              "[gl_MinProgramTexelOffset, gl_MaxProgramTexelOffset]");
                }
            }
        }

        break;
    }

    case EOpTextureQuerySamples:
    case EOpImageQuerySamples:
        break;

    case EOpImageAtomicAdd:
    case EOpImageAtomicMin:
    case EOpImageAtomicMax:
    case EOpImageAtomicAnd:
    case EOpImageAtomicOr:
    case EOpImageAtomicXor:
    case EOpImageAtomicExchange:
    case EOpImageAtomicCompSwap:
        break;

    case EOpInterpolateAtCentroid:
    case EOpInterpolateAtSample:
    case EOpInterpolateAtOffset:
        // Make sure the first argument is an interpolant, or an array element of an interpolant
        if (arg0->getType().getQualifier().storage != EvqVaryingIn) {
            // It might still be an array element.
            //
            // We could check more, but the semantics of the first argument are already met; the
            // only way to turn an array into a float/vec* is array dereference and swizzle.
            //
            // ES and desktop 4.3 and earlier:  swizzles may not be used
            // desktop 4.4 and later: swizzles may be used
            const TIntermTyped* base = TIntermediate::findLValueBase(arg0, true);
            if (base == nullptr || base->getType().getQualifier().storage != EvqVaryingIn)
                error(loc, "first argument must be an interpolant, or interpolant-array element",
                      fnCandidate.getName().c_str(), "");
        }
        break;

    default:
        break;
    }
}

//
// Handle seeing something in a grammar production that can be done by calling
// a constructor.
//
// The constructor still must be "handled" by handleFunctionCall(), which will
// then call handleConstructor().
//
TFunction* HlslParseContext::makeConstructorCall(const TSourceLoc& loc, const TType& type)
{
    TOperator op = intermediate.mapTypeToConstructorOp(type);

    if (op == EOpNull) {
        error(loc, "cannot construct this type", type.getBasicString(), "");
        return nullptr;
    }

    TString empty("");

    return new TFunction(&empty, type, op);
}

//
// Handle seeing a "COLON semantic" at the end of a type declaration,
// by updating the type according to the semantic.
//
void HlslParseContext::handleSemantic(TSourceLoc loc, TQualifier& qualifier, TBuiltInVariable builtIn,
                                      const TString& upperCase)
{
    // Parse and return semantic number.  If limit is 0, it will be ignored.  Otherwise, if the parsed
    // semantic number is >= limit, errorMsg is issued and 0 is returned.
    // TODO: it would be nicer if limit and errorMsg had default parameters, but some compilers don't yet
    // accept those in lambda functions.
    const auto getSemanticNumber = [this, loc](const TString& semantic, unsigned int limit, const char* errorMsg) -> unsigned int {
        size_t pos = semantic.find_last_not_of("0123456789");
        if (pos == std::string::npos)
            return 0u;

        unsigned int semanticNum = (unsigned int)atoi(semantic.c_str() + pos + 1);

        if (limit != 0 && semanticNum >= limit) {
            error(loc, errorMsg, semantic.c_str(), "");
            return 0u;
        }

        return semanticNum;
    };

    switch(builtIn) {
    case EbvNone:
        // Get location numbers from fragment outputs, instead of
        // auto-assigning them.
        if (language == EShLangFragment && upperCase.compare(0, 9, "SV_TARGET") == 0) {
            qualifier.layoutLocation = getSemanticNumber(upperCase, 0, nullptr);
            nextOutLocation = std::max(nextOutLocation, qualifier.layoutLocation + 1u);
        } else if (upperCase.compare(0, 15, "SV_CLIPDISTANCE") == 0) {
            builtIn = EbvClipDistance;
            qualifier.layoutLocation = getSemanticNumber(upperCase, maxClipCullRegs, "invalid clip semantic");
        } else if (upperCase.compare(0, 15, "SV_CULLDISTANCE") == 0) {
            builtIn = EbvCullDistance;
            qualifier.layoutLocation = getSemanticNumber(upperCase, maxClipCullRegs, "invalid cull semantic");
        }
        break;
    case EbvPosition:
        // adjust for stage in/out
        if (language == EShLangFragment)
            builtIn = EbvFragCoord;
        break;
    case EbvFragStencilRef:
        error(loc, "unimplemented; need ARB_shader_stencil_export", "SV_STENCILREF", "");
        break;
    case EbvTessLevelInner:
    case EbvTessLevelOuter:
        qualifier.patch = true;
        break;
    default:
        break;
    }

    if (qualifier.builtIn == EbvNone)
        qualifier.builtIn = builtIn;
    qualifier.semanticName = intermediate.addSemanticName(upperCase);
}

//
// Handle seeing something like "PACKOFFSET LEFT_PAREN c[Subcomponent][.component] RIGHT_PAREN"
//
// 'location' has the "c[Subcomponent]" part.
// 'component' points to the "component" part, or nullptr if not present.
//
void HlslParseContext::handlePackOffset(const TSourceLoc& loc, TQualifier& qualifier, const glslang::TString& location,
                                        const glslang::TString* component)
{
    if (location.size() == 0 || location[0] != 'c') {
        error(loc, "expected 'c'", "packoffset", "");
        return;
    }
    if (location.size() == 1)
        return;
    if (! isdigit(location[1])) {
        error(loc, "expected number after 'c'", "packoffset", "");
        return;
    }

    qualifier.layoutOffset = 16 * atoi(location.substr(1, location.size()).c_str());
    if (component != nullptr) {
        int componentOffset = 0;
        switch ((*component)[0]) {
        case 'x': componentOffset =  0; break;
        case 'y': componentOffset =  4; break;
        case 'z': componentOffset =  8; break;
        case 'w': componentOffset = 12; break;
        default:
            componentOffset = -1;
            break;
        }
        if (componentOffset < 0 || component->size() > 1) {
            error(loc, "expected {x, y, z, w} for component", "packoffset", "");
            return;
        }
        qualifier.layoutOffset += componentOffset;
    }
}

//
// Handle seeing something like "REGISTER LEFT_PAREN [shader_profile,] Type# RIGHT_PAREN"
//
// 'profile' points to the shader_profile part, or nullptr if not present.
// 'desc' is the type# part.
//
void HlslParseContext::handleRegister(const TSourceLoc& loc, TQualifier& qualifier, const glslang::TString* profile,
                                      const glslang::TString& desc, int subComponent, const glslang::TString* spaceDesc)
{
    if (profile != nullptr)
        warn(loc, "ignoring shader_profile", "register", "");

    if (desc.size() < 1) {
        error(loc, "expected register type", "register", "");
        return;
    }

    int regNumber = 0;
    if (desc.size() > 1) {
        if (isdigit(desc[1]))
            regNumber = atoi(desc.substr(1, desc.size()).c_str());
        else {
            error(loc, "expected register number after register type", "register", "");
            return;
        }
    }

    // more information about register types see
    // https://docs.microsoft.com/en-us/windows/desktop/direct3dhlsl/dx-graphics-hlsl-variable-register
    const std::vector<std::string>& resourceInfo = intermediate.getResourceSetBinding();
    switch (std::tolower(desc[0])) {
    case 'c':
        // c register is the register slot in the global const buffer
        // each slot is a vector of 4 32 bit components
        qualifier.layoutOffset = regNumber * 4 * 4;
        break;
        // const buffer register slot
    case 'b':
        // textrues and structured buffers
    case 't':
        // samplers
    case 's':
        // uav resources
    case 'u':
        // if nothing else has set the binding, do so now
        // (other mechanisms override this one)
        if (!qualifier.hasBinding())
            qualifier.layoutBinding = regNumber + subComponent;

        // This handles per-register layout sets numbers.  For the global mode which sets
        // every symbol to the same value, see setLinkageLayoutSets().
        if ((resourceInfo.size() % 3) == 0) {
            // Apply per-symbol resource set and binding.
            for (auto it = resourceInfo.cbegin(); it != resourceInfo.cend(); it = it + 3) {
                if (strcmp(desc.c_str(), it[0].c_str()) == 0) {
                    qualifier.layoutSet = atoi(it[1].c_str());
                    qualifier.layoutBinding = atoi(it[2].c_str()) + subComponent;
                    break;
                }
            }
        }
        break;
    default:
        warn(loc, "ignoring unrecognized register type", "register", "%c", desc[0]);
        break;
    }

    // space
    unsigned int setNumber;
    const auto crackSpace = [&]() -> bool {
        const int spaceLen = 5;
        if (spaceDesc->size() < spaceLen + 1)
            return false;
        if (spaceDesc->compare(0, spaceLen, "space") != 0)
            return false;
        if (! isdigit((*spaceDesc)[spaceLen]))
            return false;
        setNumber = atoi(spaceDesc->substr(spaceLen, spaceDesc->size()).c_str());
        return true;
    };

    // if nothing else has set the set, do so now
    // (other mechanisms override this one)
    if (spaceDesc && !qualifier.hasSet()) {
        if (! crackSpace()) {
            error(loc, "expected spaceN", "register", "");
            return;
        }
        qualifier.layoutSet = setNumber;
    }
}

// Convert to a scalar boolean, or if not allowed by HLSL semantics,
// report an error and return nullptr.
TIntermTyped* HlslParseContext::convertConditionalExpression(const TSourceLoc& loc, TIntermTyped* condition,
                                                             bool mustBeScalar)
{
    if (mustBeScalar && !condition->getType().isScalarOrVec1()) {
        error(loc, "requires a scalar", "conditional expression", "");
        return nullptr;
    }

    return intermediate.addConversion(EOpConstructBool, TType(EbtBool, EvqTemporary, condition->getVectorSize()),
                                      condition);
}

//
// Same error message for all places assignments don't work.
//
void HlslParseContext::assignError(const TSourceLoc& loc, const char* op, TString left, TString right)
{
    error(loc, "", op, "cannot convert from '%s' to '%s'",
        right.c_str(), left.c_str());
}

//
// Same error message for all places unary operations don't work.
//
void HlslParseContext::unaryOpError(const TSourceLoc& loc, const char* op, TString operand)
{
    error(loc, " wrong operand type", op,
        "no operation '%s' exists that takes an operand of type %s (or there is no acceptable conversion)",
        op, operand.c_str());
}

//
// Same error message for all binary operations don't work.
//
void HlslParseContext::binaryOpError(const TSourceLoc& loc, const char* op, TString left, TString right)
{
    error(loc, " wrong operand types:", op,
        "no operation '%s' exists that takes a left-hand operand of type '%s' and "
        "a right operand of type '%s' (or there is no acceptable conversion)",
        op, left.c_str(), right.c_str());
}

//
// A basic type of EbtVoid is a key that the name string was seen in the source, but
// it was not found as a variable in the symbol table.  If so, give the error
// message and insert a dummy variable in the symbol table to prevent future errors.
//
void HlslParseContext::variableCheck(TIntermTyped*& nodePtr)
{
    TIntermSymbol* symbol = nodePtr->getAsSymbolNode();
    if (! symbol)
        return;

    if (symbol->getType().getBasicType() == EbtVoid) {
        error(symbol->getLoc(), "undeclared identifier", symbol->getName().c_str(), "");

        // Add to symbol table to prevent future error messages on the same name
        if (symbol->getName().size() > 0) {
            TVariable* fakeVariable = new TVariable(&symbol->getName(), TType(EbtFloat));
            symbolTable.insert(*fakeVariable);

            // substitute a symbol node for this new variable
            nodePtr = intermediate.addSymbol(*fakeVariable, symbol->getLoc());
        }
    }
}

//
// Both test, and if necessary spit out an error, to see if the node is really
// a constant.
//
void HlslParseContext::constantValueCheck(TIntermTyped* node, const char* token)
{
    if (node->getQualifier().storage != EvqConst)
        error(node->getLoc(), "constant expression required", token, "");
}

//
// Both test, and if necessary spit out an error, to see if the node is really
// an integer.
//
void HlslParseContext::integerCheck(const TIntermTyped* node, const char* token)
{
    if ((node->getBasicType() == EbtInt || node->getBasicType() == EbtUint) && node->isScalar())
        return;

    error(node->getLoc(), "scalar integer expression required", token, "");
}

//
// Both test, and if necessary spit out an error, to see if we are currently
// globally scoped.
//
void HlslParseContext::globalCheck(const TSourceLoc& loc, const char* token)
{
    if (! symbolTable.atGlobalLevel())
        error(loc, "not allowed in nested scope", token, "");
}

bool HlslParseContext::builtInName(const TString& /*identifier*/)
{
    return false;
}

//
// Make sure there is enough data and not too many arguments provided to the
// constructor to build something of the type of the constructor.  Also returns
// the type of the constructor.
//
// Returns true if there was an error in construction.
//
bool HlslParseContext::constructorError(const TSourceLoc& loc, TIntermNode* node, TFunction& function,
                                        TOperator op, TType& type)
{
    type.shallowCopy(function.getType());

    bool constructingMatrix = false;
    switch (op) {
    case EOpConstructTextureSampler:
        error(loc, "unhandled texture constructor", "constructor", "");
        return true;
    case EOpConstructMat2x2:
    case EOpConstructMat2x3:
    case EOpConstructMat2x4:
    case EOpConstructMat3x2:
    case EOpConstructMat3x3:
    case EOpConstructMat3x4:
    case EOpConstructMat4x2:
    case EOpConstructMat4x3:
    case EOpConstructMat4x4:
    case EOpConstructDMat2x2:
    case EOpConstructDMat2x3:
    case EOpConstructDMat2x4:
    case EOpConstructDMat3x2:
    case EOpConstructDMat3x3:
    case EOpConstructDMat3x4:
    case EOpConstructDMat4x2:
    case EOpConstructDMat4x3:
    case EOpConstructDMat4x4:
    case EOpConstructIMat2x2:
    case EOpConstructIMat2x3:
    case EOpConstructIMat2x4:
    case EOpConstructIMat3x2:
    case EOpConstructIMat3x3:
    case EOpConstructIMat3x4:
    case EOpConstructIMat4x2:
    case EOpConstructIMat4x3:
    case EOpConstructIMat4x4:
    case EOpConstructUMat2x2:
    case EOpConstructUMat2x3:
    case EOpConstructUMat2x4:
    case EOpConstructUMat3x2:
    case EOpConstructUMat3x3:
    case EOpConstructUMat3x4:
    case EOpConstructUMat4x2:
    case EOpConstructUMat4x3:
    case EOpConstructUMat4x4:
    case EOpConstructBMat2x2:
    case EOpConstructBMat2x3:
    case EOpConstructBMat2x4:
    case EOpConstructBMat3x2:
    case EOpConstructBMat3x3:
    case EOpConstructBMat3x4:
    case EOpConstructBMat4x2:
    case EOpConstructBMat4x3:
    case EOpConstructBMat4x4:
        constructingMatrix = true;
        break;
    default:
        break;
    }

    //
    // Walk the arguments for first-pass checks and collection of information.
    //

    int size = 0;
    bool constType = true;
    bool full = false;
    bool overFull = false;
    bool matrixInMatrix = false;
    bool arrayArg = false;
    for (int arg = 0; arg < function.getParamCount(); ++arg) {
        if (function[arg].type->isArray()) {
            if (function[arg].type->isUnsizedArray()) {
                // Can't construct from an unsized array.
                error(loc, "array argument must be sized", "constructor", "");
                return true;
            }
            arrayArg = true;
        }
        if (constructingMatrix && function[arg].type->isMatrix())
            matrixInMatrix = true;

        // 'full' will go to true when enough args have been seen.  If we loop
        // again, there is an extra argument.
        if (full) {
            // For vectors and matrices, it's okay to have too many components
            // available, but not okay to have unused arguments.
            overFull = true;
        }

        size += function[arg].type->computeNumComponents();
        if (op != EOpConstructStruct && ! type.isArray() && size >= type.computeNumComponents())
            full = true;

        if (function[arg].type->getQualifier().storage != EvqConst)
            constType = false;
    }

    if (constType)
        type.getQualifier().storage = EvqConst;

    if (type.isArray()) {
        if (function.getParamCount() == 0) {
            error(loc, "array constructor must have at least one argument", "constructor", "");
            return true;
        }

        if (type.isUnsizedArray()) {
            // auto adapt the constructor type to the number of arguments
            type.changeOuterArraySize(function.getParamCount());
        } else if (type.getOuterArraySize() != function.getParamCount() && type.computeNumComponents() > size) {
            error(loc, "array constructor needs one argument per array element", "constructor", "");
            return true;
        }

        if (type.isArrayOfArrays()) {
            // Types have to match, but we're still making the type.
            // Finish making the type, and the comparison is done later
            // when checking for conversion.
            TArraySizes& arraySizes = *type.getArraySizes();

            // At least the dimensionalities have to match.
            if (! function[0].type->isArray() ||
                arraySizes.getNumDims() != function[0].type->getArraySizes()->getNumDims() + 1) {
                error(loc, "array constructor argument not correct type to construct array element", "constructor", "");
                return true;
            }

            if (arraySizes.isInnerUnsized()) {
                // "Arrays of arrays ..., and the size for any dimension is optional"
                // That means we need to adopt (from the first argument) the other array sizes into the type.
                for (int d = 1; d < arraySizes.getNumDims(); ++d) {
                    if (arraySizes.getDimSize(d) == UnsizedArraySize) {
                        arraySizes.setDimSize(d, function[0].type->getArraySizes()->getDimSize(d - 1));
                    }
                }
            }
        }
    }

    // Some array -> array type casts are okay
    if (arrayArg && function.getParamCount() == 1 && op != EOpConstructStruct && type.isArray() &&
        !type.isArrayOfArrays() && !function[0].type->isArrayOfArrays() &&
        type.getVectorSize() >= 1 && function[0].type->getVectorSize() >= 1)
        return false;

    if (arrayArg && op != EOpConstructStruct && ! type.isArrayOfArrays()) {
        error(loc, "constructing non-array constituent from array argument", "constructor", "");
        return true;
    }

    if (matrixInMatrix && ! type.isArray()) {
        return false;
    }

    if (overFull) {
        error(loc, "too many arguments", "constructor", "");
        return true;
    }

    if (op == EOpConstructStruct && ! type.isArray()) {
        if (isScalarConstructor(node))
            return false;

        // Self-type construction: e.g, we can construct a struct from a single identically typed object.
        if (function.getParamCount() == 1 && type == *function[0].type)
            return false;

        if ((int)type.getStruct()->size() != function.getParamCount()) {
            error(loc, "Number of constructor parameters does not match the number of structure fields", "constructor", "");
            return true;
        }
    }

    if ((op != EOpConstructStruct && size != 1 && size < type.computeNumComponents()) ||
        (op == EOpConstructStruct && size < type.computeNumComponents())) {
        error(loc, "not enough data provided for construction", "constructor", "");
        return true;
    }

    return false;
}

// See if 'node', in the context of constructing aggregates, is a scalar argument
// to a constructor.
//
bool HlslParseContext::isScalarConstructor(const TIntermNode* node)
{
    // Obviously, it must be a scalar, but an aggregate node might not be fully
    // completed yet: holding a sequence of initializers under an aggregate
    // would not yet be typed, so don't check it's type.  This corresponds to
    // the aggregate operator also not being set yet. (An aggregate operation
    // that legitimately yields a scalar will have a getOp() of that operator,
    // not EOpNull.)

    return node->getAsTyped() != nullptr &&
           node->getAsTyped()->isScalar() &&
           (node->getAsAggregate() == nullptr || node->getAsAggregate()->getOp() != EOpNull);
}

// Checks to see if a void variable has been declared and raise an error message for such a case
//
// returns true in case of an error
//
bool HlslParseContext::voidErrorCheck(const TSourceLoc& loc, const TString& identifier, const TBasicType basicType)
{
    if (basicType == EbtVoid) {
        error(loc, "illegal use of type 'void'", identifier.c_str(), "");
        return true;
    }

    return false;
}

//
// Fix just a full qualifier (no variables or types yet, but qualifier is complete) at global level.
//
void HlslParseContext::globalQualifierFix(const TSourceLoc&, TQualifier& qualifier)
{
    // move from parameter/unknown qualifiers to pipeline in/out qualifiers
    switch (qualifier.storage) {
    case EvqIn:
        qualifier.storage = EvqVaryingIn;
        break;
    case EvqOut:
        qualifier.storage = EvqVaryingOut;
        break;
    default:
        break;
    }
}

//
// Merge characteristics of the 'src' qualifier into the 'dst'.
// If there is duplication, issue error messages, unless 'force'
// is specified, which means to just override default settings.
//
// Also, when force is false, it will be assumed that 'src' follows
// 'dst', for the purpose of error checking order for versions
// that require specific orderings of qualifiers.
//
void HlslParseContext::mergeQualifiers(TQualifier& dst, const TQualifier& src)
{
    // Storage qualification
    if (dst.storage == EvqTemporary || dst.storage == EvqGlobal)
        dst.storage = src.storage;
    else if ((dst.storage == EvqIn  && src.storage == EvqOut) ||
             (dst.storage == EvqOut && src.storage == EvqIn))
        dst.storage = EvqInOut;
    else if ((dst.storage == EvqIn    && src.storage == EvqConst) ||
             (dst.storage == EvqConst && src.storage == EvqIn))
        dst.storage = EvqConstReadOnly;

    // Layout qualifiers
    mergeObjectLayoutQualifiers(dst, src, false);

    // individual qualifiers
    bool repeated = false;
#define MERGE_SINGLETON(field) repeated |= dst.field && src.field; dst.field |= src.field;
    MERGE_SINGLETON(invariant);
    MERGE_SINGLETON(noContraction);
    MERGE_SINGLETON(centroid);
    MERGE_SINGLETON(smooth);
    MERGE_SINGLETON(flat);
    MERGE_SINGLETON(nopersp);
    MERGE_SINGLETON(patch);
    MERGE_SINGLETON(sample);
    MERGE_SINGLETON(coherent);
    MERGE_SINGLETON(volatil);
    MERGE_SINGLETON(restrict);
    MERGE_SINGLETON(readonly);
    MERGE_SINGLETON(writeonly);
    MERGE_SINGLETON(specConstant);
    MERGE_SINGLETON(nonUniform);
}

// used to flatten the sampler type space into a single dimension
// correlates with the declaration of defaultSamplerPrecision[]
int HlslParseContext::computeSamplerTypeIndex(TSampler& sampler)
{
    int arrayIndex = sampler.arrayed ? 1 : 0;
    int shadowIndex = sampler.shadow ? 1 : 0;
    int externalIndex = sampler.external ? 1 : 0;

    return EsdNumDims *
           (EbtNumTypes * (2 * (2 * arrayIndex + shadowIndex) + externalIndex) + sampler.type) + sampler.dim;
}

//
// Do size checking for an array type's size.
//
void HlslParseContext::arraySizeCheck(const TSourceLoc& loc, TIntermTyped* expr, TArraySize& sizePair)
{
    bool isConst = false;
    sizePair.size = 1;
    sizePair.node = nullptr;

    TIntermConstantUnion* constant = expr->getAsConstantUnion();
    if (constant) {
        // handle true (non-specialization) constant
        sizePair.size = constant->getConstArray()[0].getIConst();
        isConst = true;
    } else {
        // see if it's a specialization constant instead
        if (expr->getQualifier().isSpecConstant()) {
            isConst = true;
            sizePair.node = expr;
            TIntermSymbol* symbol = expr->getAsSymbolNode();
            if (symbol && symbol->getConstArray().size() > 0)
                sizePair.size = symbol->getConstArray()[0].getIConst();
        }
    }

    if (! isConst || (expr->getBasicType() != EbtInt && expr->getBasicType() != EbtUint)) {
        error(loc, "array size must be a constant integer expression", "", "");
        return;
    }

    if (sizePair.size <= 0) {
        error(loc, "array size must be a positive integer", "", "");
        return;
    }
}

//
// Require array to be completely sized
//
void HlslParseContext::arraySizeRequiredCheck(const TSourceLoc& loc, const TArraySizes& arraySizes)
{
    if (arraySizes.hasUnsized())
        error(loc, "array size required", "", "");
}

void HlslParseContext::structArrayCheck(const TSourceLoc& /*loc*/, const TType& type)
{
    const TTypeList& structure = *type.getStruct();
    for (int m = 0; m < (int)structure.size(); ++m) {
        const TType& member = *structure[m].type;
        if (member.isArray())
            arraySizeRequiredCheck(structure[m].loc, *member.getArraySizes());
    }
}

//
// Do all the semantic checking for declaring or redeclaring an array, with and
// without a size, and make the right changes to the symbol table.
//
void HlslParseContext::declareArray(const TSourceLoc& loc, const TString& identifier, const TType& type,
                                    TSymbol*& symbol, bool track)
{
    if (symbol == nullptr) {
        bool currentScope;
        symbol = symbolTable.find(identifier, nullptr, &currentScope);

        if (symbol && builtInName(identifier) && ! symbolTable.atBuiltInLevel()) {
            // bad shader (errors already reported) trying to redeclare a built-in name as an array
            return;
        }
        if (symbol == nullptr || ! currentScope) {
            //
            // Successfully process a new definition.
            // (Redeclarations have to take place at the same scope; otherwise they are hiding declarations)
            //
            symbol = new TVariable(&identifier, type);
            symbolTable.insert(*symbol);
            if (track && symbolTable.atGlobalLevel())
                trackLinkage(*symbol);

            return;
        }
        if (symbol->getAsAnonMember()) {
            error(loc, "cannot redeclare a user-block member array", identifier.c_str(), "");
            symbol = nullptr;
            return;
        }
    }

    //
    // Process a redeclaration.
    //

    if (symbol == nullptr) {
        error(loc, "array variable name expected", identifier.c_str(), "");
        return;
    }

    // redeclareBuiltinVariable() should have already done the copyUp()
    TType& existingType = symbol->getWritableType();

    if (existingType.isSizedArray()) {
        // be more lenient for input arrays to geometry shaders and tessellation control outputs,
        // where the redeclaration is the same size
        return;
    }

    existingType.updateArraySizes(type);
}

//
// Enforce non-initializer type/qualifier rules.
//
void HlslParseContext::fixConstInit(const TSourceLoc& loc, const TString& identifier, TType& type,
                                    TIntermTyped*& initializer)
{
    //
    // Make the qualifier make sense, given that there is an initializer.
    //
    if (initializer == nullptr) {
        if (type.getQualifier().storage == EvqConst ||
            type.getQualifier().storage == EvqConstReadOnly) {
            initializer = intermediate.makeAggregate(loc);
            warn(loc, "variable with qualifier 'const' not initialized; zero initializing", identifier.c_str(), "");
        }
    }
}

//
// See if the identifier is a built-in symbol that can be redeclared, and if so,
// copy the symbol table's read-only built-in variable to the current
// global level, where it can be modified based on the passed in type.
//
// Returns nullptr if no redeclaration took place; meaning a normal declaration still
// needs to occur for it, not necessarily an error.
//
// Returns a redeclared and type-modified variable if a redeclared occurred.
//
TSymbol* HlslParseContext::redeclareBuiltinVariable(const TSourceLoc& /*loc*/, const TString& identifier,
                                                    const TQualifier& /*qualifier*/,
                                                    const TShaderQualifiers& /*publicType*/)
{
    if (! builtInName(identifier) || symbolTable.atBuiltInLevel() || ! symbolTable.atGlobalLevel())
        return nullptr;

    return nullptr;
}

//
// Generate index to the array element in a structure buffer (SSBO)
//
TIntermTyped* HlslParseContext::indexStructBufferContent(const TSourceLoc& loc, TIntermTyped* buffer) const
{
    // Bail out if not a struct buffer
    if (buffer == nullptr || ! isStructBufferType(buffer->getType()))
        return nullptr;

    // Runtime sized array is always the last element.
    const TTypeList* bufferStruct = buffer->getType().getStruct();
    TIntermTyped* arrayPosition = intermediate.addConstantUnion(unsigned(bufferStruct->size()-1), loc);

    TIntermTyped* argArray = intermediate.addIndex(EOpIndexDirectStruct, buffer, arrayPosition, loc);
    argArray->setType(*(*bufferStruct)[bufferStruct->size()-1].type);

    return argArray;
}

//
// IFF type is a structuredbuffer/byteaddressbuffer type, return the content
// (template) type.   E.g, StructuredBuffer<MyType> -> MyType.  Else return nullptr.
//
TType* HlslParseContext::getStructBufferContentType(const TType& type) const
{
    if (type.getBasicType() != EbtBlock || type.getQualifier().storage != EvqBuffer)
        return nullptr;

    const int memberCount = (int)type.getStruct()->size();
    assert(memberCount > 0);

    TType* contentType = (*type.getStruct())[memberCount-1].type;

    return contentType->isUnsizedArray() ? contentType : nullptr;
}

//
// If an existing struct buffer has a sharable type, then share it.
//
void HlslParseContext::shareStructBufferType(TType& type)
{
    // PackOffset must be equivalent to share types on a per-member basis.
    // Note: cannot use auto type due to recursion.  Thus, this is a std::function.
    const std::function<bool(TType& lhs, TType& rhs)>
    compareQualifiers = [&](TType& lhs, TType& rhs) -> bool {
        if (lhs.getQualifier().layoutOffset != rhs.getQualifier().layoutOffset)
            return false;

        if (lhs.isStruct() != rhs.isStruct())
            return false;

        if (lhs.isStruct() && rhs.isStruct()) {
            if (lhs.getStruct()->size() != rhs.getStruct()->size())
                return false;

            for (int i = 0; i < int(lhs.getStruct()->size()); ++i)
                if (!compareQualifiers(*(*lhs.getStruct())[i].type, *(*rhs.getStruct())[i].type))
                    return false;
        }

        return true;
    };

    // We need to compare certain qualifiers in addition to the type.
    const auto typeEqual = [compareQualifiers](TType& lhs, TType& rhs) -> bool {
        if (lhs.getQualifier().readonly != rhs.getQualifier().readonly)
            return false;

        // If both are structures, recursively look for packOffset equality
        // as well as type equality.
        return compareQualifiers(lhs, rhs) && lhs == rhs;
    };

    // This is an exhaustive O(N) search, but real world shaders have
    // only a small number of these.
    for (int idx = 0; idx < int(structBufferTypes.size()); ++idx) {
        // If the deep structure matches, modulo qualifiers, use it
        if (typeEqual(*structBufferTypes[idx], type)) {
            type.shallowCopy(*structBufferTypes[idx]);
            return;
        }
    }

    // Otherwise, remember it:
    TType* typeCopy = new TType;
    typeCopy->shallowCopy(type);
    structBufferTypes.push_back(typeCopy);
}

void HlslParseContext::paramFix(TType& type)
{
    switch (type.getQualifier().storage) {
    case EvqConst:
        type.getQualifier().storage = EvqConstReadOnly;
        break;
    case EvqGlobal:
    case EvqUniform:
    case EvqTemporary:
        type.getQualifier().storage = EvqIn;
        break;
    case EvqBuffer:
        {
            // SSBO parameter.  These do not go through the declareBlock path since they are fn parameters.
            correctUniform(type.getQualifier());
            TQualifier bufferQualifier = globalBufferDefaults;
            mergeObjectLayoutQualifiers(bufferQualifier, type.getQualifier(), true);
            bufferQualifier.storage = type.getQualifier().storage;
            bufferQualifier.readonly = type.getQualifier().readonly;
            bufferQualifier.coherent = type.getQualifier().coherent;
            bufferQualifier.declaredBuiltIn = type.getQualifier().declaredBuiltIn;
            type.getQualifier() = bufferQualifier;
            break;
        }
    default:
        break;
    }
}

void HlslParseContext::specializationCheck(const TSourceLoc& loc, const TType& type, const char* op)
{
    if (type.containsSpecializationSize())
        error(loc, "can't use with types containing arrays sized with a specialization constant", op, "");
}

//
// Layout qualifier stuff.
//

// Put the id's layout qualification into the public type, for qualifiers not having a number set.
// This is before we know any type information for error checking.
void HlslParseContext::setLayoutQualifier(const TSourceLoc& loc, TQualifier& qualifier, TString& id)
{
    std::transform(id.begin(), id.end(), id.begin(), ::tolower);

    if (id == TQualifier::getLayoutMatrixString(ElmColumnMajor)) {
        qualifier.layoutMatrix = ElmRowMajor;
        return;
    }
    if (id == TQualifier::getLayoutMatrixString(ElmRowMajor)) {
        qualifier.layoutMatrix = ElmColumnMajor;
        return;
    }
    if (id == "push_constant") {
        requireVulkan(loc, "push_constant");
        qualifier.layoutPushConstant = true;
        return;
    }
    if (language == EShLangGeometry || language == EShLangTessEvaluation) {
        if (id == TQualifier::getGeometryString(ElgTriangles)) {
            // publicType.shaderQualifiers.geometry = ElgTriangles;
            warn(loc, "ignored", id.c_str(), "");
            return;
        }
        if (language == EShLangGeometry) {
            if (id == TQualifier::getGeometryString(ElgPoints)) {
                // publicType.shaderQualifiers.geometry = ElgPoints;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == TQualifier::getGeometryString(ElgLineStrip)) {
                // publicType.shaderQualifiers.geometry = ElgLineStrip;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == TQualifier::getGeometryString(ElgLines)) {
                // publicType.shaderQualifiers.geometry = ElgLines;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == TQualifier::getGeometryString(ElgLinesAdjacency)) {
                // publicType.shaderQualifiers.geometry = ElgLinesAdjacency;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == TQualifier::getGeometryString(ElgTrianglesAdjacency)) {
                // publicType.shaderQualifiers.geometry = ElgTrianglesAdjacency;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == TQualifier::getGeometryString(ElgTriangleStrip)) {
                // publicType.shaderQualifiers.geometry = ElgTriangleStrip;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
        } else {
            assert(language == EShLangTessEvaluation);

            // input primitive
            if (id == TQualifier::getGeometryString(ElgTriangles)) {
                // publicType.shaderQualifiers.geometry = ElgTriangles;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == TQualifier::getGeometryString(ElgQuads)) {
                // publicType.shaderQualifiers.geometry = ElgQuads;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == TQualifier::getGeometryString(ElgIsolines)) {
                // publicType.shaderQualifiers.geometry = ElgIsolines;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }

            // vertex spacing
            if (id == TQualifier::getVertexSpacingString(EvsEqual)) {
                // publicType.shaderQualifiers.spacing = EvsEqual;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == TQualifier::getVertexSpacingString(EvsFractionalEven)) {
                // publicType.shaderQualifiers.spacing = EvsFractionalEven;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == TQualifier::getVertexSpacingString(EvsFractionalOdd)) {
                // publicType.shaderQualifiers.spacing = EvsFractionalOdd;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }

            // triangle order
            if (id == TQualifier::getVertexOrderString(EvoCw)) {
                // publicType.shaderQualifiers.order = EvoCw;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == TQualifier::getVertexOrderString(EvoCcw)) {
                // publicType.shaderQualifiers.order = EvoCcw;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }

            // point mode
            if (id == "point_mode") {
                // publicType.shaderQualifiers.pointMode = true;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
        }
    }
    if (language == EShLangFragment) {
        if (id == "origin_upper_left") {
            // publicType.shaderQualifiers.originUpperLeft = true;
            warn(loc, "ignored", id.c_str(), "");
            return;
        }
        if (id == "pixel_center_integer") {
            // publicType.shaderQualifiers.pixelCenterInteger = true;
            warn(loc, "ignored", id.c_str(), "");
            return;
        }
        if (id == "early_fragment_tests") {
            // publicType.shaderQualifiers.earlyFragmentTests = true;
            warn(loc, "ignored", id.c_str(), "");
            return;
        }
        for (TLayoutDepth depth = (TLayoutDepth)(EldNone + 1); depth < EldCount; depth = (TLayoutDepth)(depth + 1)) {
            if (id == TQualifier::getLayoutDepthString(depth)) {
                // publicType.shaderQualifiers.layoutDepth = depth;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
        }
        if (id.compare(0, 13, "blend_support") == 0) {
            bool found = false;
            for (TBlendEquationShift be = (TBlendEquationShift)0; be < EBlendCount; be = (TBlendEquationShift)(be + 1)) {
                if (id == TQualifier::getBlendEquationString(be)) {
                    requireExtensions(loc, 1, &E_GL_KHR_blend_equation_advanced, "blend equation");
                    intermediate.addBlendEquation(be);
                    // publicType.shaderQualifiers.blendEquation = true;
                    warn(loc, "ignored", id.c_str(), "");
                    found = true;
                    break;
                }
            }
            if (! found)
                error(loc, "unknown blend equation", "blend_support", "");
            return;
        }
    }
    error(loc, "unrecognized layout identifier, or qualifier requires assignment (e.g., binding = 4)", id.c_str(), "");
}

// Put the id's layout qualifier value into the public type, for qualifiers having a number set.
// This is before we know any type information for error checking.
void HlslParseContext::setLayoutQualifier(const TSourceLoc& loc, TQualifier& qualifier, TString& id,
                                          const TIntermTyped* node)
{
    const char* feature = "layout-id value";
    // const char* nonLiteralFeature = "non-literal layout-id value";

    integerCheck(node, feature);
    const TIntermConstantUnion* constUnion = node->getAsConstantUnion();
    int value = 0;
    if (constUnion) {
        value = constUnion->getConstArray()[0].getIConst();
    }

    std::transform(id.begin(), id.end(), id.begin(), ::tolower);

    if (id == "offset") {
        qualifier.layoutOffset = value;
        return;
    } else if (id == "align") {
        // "The specified alignment must be a power of 2, or a compile-time error results."
        if (! IsPow2(value))
            error(loc, "must be a power of 2", "align", "");
        else
            qualifier.layoutAlign = value;
        return;
    } else if (id == "location") {
        if ((unsigned int)value >= TQualifier::layoutLocationEnd)
            error(loc, "location is too large", id.c_str(), "");
        else
            qualifier.layoutLocation = value;
        return;
    } else if (id == "set") {
        if ((unsigned int)value >= TQualifier::layoutSetEnd)
            error(loc, "set is too large", id.c_str(), "");
        else
            qualifier.layoutSet = value;
        return;
    } else if (id == "binding") {
        if ((unsigned int)value >= TQualifier::layoutBindingEnd)
            error(loc, "binding is too large", id.c_str(), "");
        else
            qualifier.layoutBinding = value;
        return;
    } else if (id == "component") {
        if ((unsigned)value >= TQualifier::layoutComponentEnd)
            error(loc, "component is too large", id.c_str(), "");
        else
            qualifier.layoutComponent = value;
        return;
    } else if (id.compare(0, 4, "xfb_") == 0) {
        // "Any shader making any static use (after preprocessing) of any of these
        // *xfb_* qualifiers will cause the shader to be in a transform feedback
        // capturing mode and hence responsible for describing the transform feedback
        // setup."
        intermediate.setXfbMode();
        if (id == "xfb_buffer") {
            // "It is a compile-time error to specify an *xfb_buffer* that is greater than
            // the implementation-dependent constant gl_MaxTransformFeedbackBuffers."
            if (value >= resources.maxTransformFeedbackBuffers)
                error(loc, "buffer is too large:", id.c_str(), "gl_MaxTransformFeedbackBuffers is %d",
                      resources.maxTransformFeedbackBuffers);
            if (value >= (int)TQualifier::layoutXfbBufferEnd)
                error(loc, "buffer is too large:", id.c_str(), "internal max is %d", TQualifier::layoutXfbBufferEnd - 1);
            else
                qualifier.layoutXfbBuffer = value;
            return;
        } else if (id == "xfb_offset") {
            if (value >= (int)TQualifier::layoutXfbOffsetEnd)
                error(loc, "offset is too large:", id.c_str(), "internal max is %d", TQualifier::layoutXfbOffsetEnd - 1);
            else
                qualifier.layoutXfbOffset = value;
            return;
        } else if (id == "xfb_stride") {
            // "The resulting stride (implicit or explicit), when divided by 4, must be less than or equal to the
            // implementation-dependent constant gl_MaxTransformFeedbackInterleavedComponents."
            if (value > 4 * resources.maxTransformFeedbackInterleavedComponents)
                error(loc, "1/4 stride is too large:", id.c_str(), "gl_MaxTransformFeedbackInterleavedComponents is %d",
                      resources.maxTransformFeedbackInterleavedComponents);
            else if (value >= (int)TQualifier::layoutXfbStrideEnd)
                error(loc, "stride is too large:", id.c_str(), "internal max is %d", TQualifier::layoutXfbStrideEnd - 1);
            if (value < (int)TQualifier::layoutXfbStrideEnd)
                qualifier.layoutXfbStride = value;
            return;
        }
    }

    if (id == "input_attachment_index") {
        requireVulkan(loc, "input_attachment_index");
        if (value >= (int)TQualifier::layoutAttachmentEnd)
            error(loc, "attachment index is too large", id.c_str(), "");
        else
            qualifier.layoutAttachment = value;
        return;
    }
    if (id == "constant_id") {
        setSpecConstantId(loc, qualifier, value);
        return;
    }

    switch (language) {
    case EShLangVertex:
        break;

    case EShLangTessControl:
        if (id == "vertices") {
            if (value == 0)
                error(loc, "must be greater than 0", "vertices", "");
            else
                // publicType.shaderQualifiers.vertices = value;
                warn(loc, "ignored", id.c_str(), "");
            return;
        }
        break;

    case EShLangTessEvaluation:
        break;

    case EShLangGeometry:
        if (id == "invocations") {
            if (value == 0)
                error(loc, "must be at least 1", "invocations", "");
            else
                // publicType.shaderQualifiers.invocations = value;
                warn(loc, "ignored", id.c_str(), "");
            return;
        }
        if (id == "max_vertices") {
            // publicType.shaderQualifiers.vertices = value;
            warn(loc, "ignored", id.c_str(), "");
            if (value > resources.maxGeometryOutputVertices)
                error(loc, "too large, must be less than gl_MaxGeometryOutputVertices", "max_vertices", "");
            return;
        }
        if (id == "stream") {
            qualifier.layoutStream = value;
            return;
        }
        break;

    case EShLangFragment:
        if (id == "index") {
            qualifier.layoutIndex = value;
            return;
        }
        break;

    case EShLangCompute:
        if (id.compare(0, 11, "local_size_") == 0) {
            if (id == "local_size_x") {
                // publicType.shaderQualifiers.localSize[0] = value;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == "local_size_y") {
                // publicType.shaderQualifiers.localSize[1] = value;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (id == "local_size_z") {
                // publicType.shaderQualifiers.localSize[2] = value;
                warn(loc, "ignored", id.c_str(), "");
                return;
            }
            if (spvVersion.spv != 0) {
                if (id == "local_size_x_id") {
                    // publicType.shaderQualifiers.localSizeSpecId[0] = value;
                    warn(loc, "ignored", id.c_str(), "");
                    return;
                }
                if (id == "local_size_y_id") {
                    // publicType.shaderQualifiers.localSizeSpecId[1] = value;
                    warn(loc, "ignored", id.c_str(), "");
                    return;
                }
                if (id == "local_size_z_id") {
                    // publicType.shaderQualifiers.localSizeSpecId[2] = value;
                    warn(loc, "ignored", id.c_str(), "");
                    return;
                }
            }
        }
        break;

    default:
        break;
    }

    error(loc, "there is no such layout identifier for this stage taking an assigned value", id.c_str(), "");
}

void HlslParseContext::setSpecConstantId(const TSourceLoc& loc, TQualifier& qualifier, int value)
{
    if (value >= (int)TQualifier::layoutSpecConstantIdEnd) {
        error(loc, "specialization-constant id is too large", "constant_id", "");
    } else {
        qualifier.layoutSpecConstantId = value;
        qualifier.specConstant = true;
        if (! intermediate.addUsedConstantId(value))
            error(loc, "specialization-constant id already used", "constant_id", "");
    }
    return;
}

// Merge any layout qualifier information from src into dst, leaving everything else in dst alone
//
// "More than one layout qualifier may appear in a single declaration.
// Additionally, the same layout-qualifier-name can occur multiple times
// within a layout qualifier or across multiple layout qualifiers in the
// same declaration. When the same layout-qualifier-name occurs
// multiple times, in a single declaration, the last occurrence overrides
// the former occurrence(s).  Further, if such a layout-qualifier-name
// will effect subsequent declarations or other observable behavior, it
// is only the last occurrence that will have any effect, behaving as if
// the earlier occurrence(s) within the declaration are not present.
// This is also true for overriding layout-qualifier-names, where one
// overrides the other (e.g., row_major vs. column_major); only the last
// occurrence has any effect."
//
void HlslParseContext::mergeObjectLayoutQualifiers(TQualifier& dst, const TQualifier& src, bool inheritOnly)
{
    if (src.hasMatrix())
        dst.layoutMatrix = src.layoutMatrix;
    if (src.hasPacking())
        dst.layoutPacking = src.layoutPacking;

    if (src.hasStream())
        dst.layoutStream = src.layoutStream;

    if (src.hasFormat())
        dst.layoutFormat = src.layoutFormat;

    if (src.hasXfbBuffer())
        dst.layoutXfbBuffer = src.layoutXfbBuffer;

    if (src.hasAlign())
        dst.layoutAlign = src.layoutAlign;

    if (! inheritOnly) {
        if (src.hasLocation())
            dst.layoutLocation = src.layoutLocation;
        if (src.hasComponent())
            dst.layoutComponent = src.layoutComponent;
        if (src.hasIndex())
            dst.layoutIndex = src.layoutIndex;

        if (src.hasOffset())
            dst.layoutOffset = src.layoutOffset;

        if (src.hasSet())
            dst.layoutSet = src.layoutSet;
        if (src.layoutBinding != TQualifier::layoutBindingEnd)
            dst.layoutBinding = src.layoutBinding;

        if (src.hasXfbStride())
            dst.layoutXfbStride = src.layoutXfbStride;
        if (src.hasXfbOffset())
            dst.layoutXfbOffset = src.layoutXfbOffset;
        if (src.hasAttachment())
            dst.layoutAttachment = src.layoutAttachment;
        if (src.hasSpecConstantId())
            dst.layoutSpecConstantId = src.layoutSpecConstantId;

        if (src.layoutPushConstant)
            dst.layoutPushConstant = true;
    }
}


//
// Look up a function name in the symbol table, and make sure it is a function.
//
// First, look for an exact match.  If there is none, use the generic selector
// TParseContextBase::selectFunction() to find one, parameterized by the
// convertible() and better() predicates defined below.
//
// Return the function symbol if found, otherwise nullptr.
//
const TFunction* HlslParseContext::findFunction(const TSourceLoc& loc, TFunction& call, bool& builtIn, int& thisDepth,
                                                TIntermTyped*& args)
{
    if (symbolTable.isFunctionNameVariable(call.getName())) {
        error(loc, "can't use function syntax on variable", call.getName().c_str(), "");
        return nullptr;
    }

    // first, look for an exact match
    bool dummyScope;
    TSymbol* symbol = symbolTable.find(call.getMangledName(), &builtIn, &dummyScope, &thisDepth);
    if (symbol)
        return symbol->getAsFunction();

    // no exact match, use the generic selector, parameterized by the GLSL rules

    // create list of candidates to send
    TVector<const TFunction*> candidateList;
    symbolTable.findFunctionNameList(call.getMangledName(), candidateList, builtIn);

    // These built-in ops can accept any type, so we bypass the argument selection
    if (candidateList.size() == 1 && builtIn &&
        (candidateList[0]->getBuiltInOp() == EOpMethodAppend ||
         candidateList[0]->getBuiltInOp() == EOpMethodRestartStrip ||
         candidateList[0]->getBuiltInOp() == EOpMethodIncrementCounter ||
         candidateList[0]->getBuiltInOp() == EOpMethodDecrementCounter ||
         candidateList[0]->getBuiltInOp() == EOpMethodAppend ||
         candidateList[0]->getBuiltInOp() == EOpMethodConsume)) {
        return candidateList[0];
    }

    bool allowOnlyUpConversions = true;

    // can 'from' convert to 'to'?
    const auto convertible = [&](const TType& from, const TType& to, TOperator op, int arg) -> bool {
        if (from == to)
            return true;

        // no aggregate conversions
        if (from.isArray()  || to.isArray() ||
            from.isStruct() || to.isStruct())
            return false;

        switch (op) {
        case EOpInterlockedAdd:
        case EOpInterlockedAnd:
        case EOpInterlockedCompareExchange:
        case EOpInterlockedCompareStore:
        case EOpInterlockedExchange:
        case EOpInterlockedMax:
        case EOpInterlockedMin:
        case EOpInterlockedOr:
        case EOpInterlockedXor:
            // We do not promote the texture or image type for these ocodes.  Normally that would not
            // be an issue because it's a buffer, but we haven't decomposed the opcode yet, and at this
            // stage it's merely e.g, a basic integer type.
            //
            // Instead, we want to promote other arguments, but stay within the same family.  In other
            // words, InterlockedAdd(RWBuffer<int>, ...) will always use the int flavor, never the uint flavor,
            // but it is allowed to promote its other arguments.
            if (arg == 0)
                return false;
            break;
        case EOpMethodSample:
        case EOpMethodSampleBias:
        case EOpMethodSampleCmp:
        case EOpMethodSampleCmpLevelZero:
        case EOpMethodSampleGrad:
        case EOpMethodSampleLevel:
        case EOpMethodLoad:
        case EOpMethodGetDimensions:
        case EOpMethodGetSamplePosition:
        case EOpMethodGather:
        case EOpMethodCalculateLevelOfDetail:
        case EOpMethodCalculateLevelOfDetailUnclamped:
        case EOpMethodGatherRed:
        case EOpMethodGatherGreen:
        case EOpMethodGatherBlue:
        case EOpMethodGatherAlpha:
        case EOpMethodGatherCmp:
        case EOpMethodGatherCmpRed:
        case EOpMethodGatherCmpGreen:
        case EOpMethodGatherCmpBlue:
        case EOpMethodGatherCmpAlpha:
        case EOpMethodAppend:
        case EOpMethodRestartStrip:
            // those are method calls, the object type can not be changed
            // they are equal if the dim and type match (is dim sufficient?)
            if (arg == 0)
                return from.getSampler().type == to.getSampler().type &&
                       from.getSampler().arrayed == to.getSampler().arrayed &&
                       from.getSampler().shadow == to.getSampler().shadow &&
                       from.getSampler().ms == to.getSampler().ms &&
                       from.getSampler().dim == to.getSampler().dim;
            break;
        default:
            break;
        }

        // basic types have to be convertible
        if (allowOnlyUpConversions)
            if (! intermediate.canImplicitlyPromote(from.getBasicType(), to.getBasicType(), EOpFunctionCall))
                return false;

        // shapes have to be convertible
        if ((from.isScalarOrVec1() && to.isScalarOrVec1()) ||
            (from.isScalarOrVec1() && to.isVector())    ||
            (from.isScalarOrVec1() && to.isMatrix())    ||
            (from.isVector() && to.isVector() && from.getVectorSize() >= to.getVectorSize()))
            return true;

        // TODO: what are the matrix rules? they go here

        return false;
    };

    // Is 'to2' a better conversion than 'to1'?
    // Ties should not be considered as better.
    // Assumes 'convertible' already said true.
    const auto better = [](const TType& from, const TType& to1, const TType& to2) -> bool {
        // exact match is always better than mismatch
        if (from == to2)
            return from != to1;
        if (from == to1)
            return false;

        // shape changes are always worse
        if (from.isScalar() || from.isVector()) {
            if (from.getVectorSize() == to2.getVectorSize() &&
                from.getVectorSize() != to1.getVectorSize())
                return true;
            if (from.getVectorSize() == to1.getVectorSize() &&
                from.getVectorSize() != to2.getVectorSize())
                return false;
        }

        // Handle sampler betterness: An exact sampler match beats a non-exact match.
        // (If we just looked at basic type, all EbtSamplers would look the same).
        // If any type is not a sampler, just use the linearize function below.
        if (from.getBasicType() == EbtSampler && to1.getBasicType() == EbtSampler && to2.getBasicType() == EbtSampler) {
            // We can ignore the vector size in the comparison.
            TSampler to1Sampler = to1.getSampler();
            TSampler to2Sampler = to2.getSampler();

            to1Sampler.vectorSize = to2Sampler.vectorSize = from.getSampler().vectorSize;

            if (from.getSampler() == to2Sampler)
                return from.getSampler() != to1Sampler;
            if (from.getSampler() == to1Sampler)
                return false;
        }

        // Might or might not be changing shape, which means basic type might
        // or might not match, so within that, the question is how big a
        // basic-type conversion is being done.
        //
        // Use a hierarchy of domains, translated to order of magnitude
        // in a linearized view:
        //   - floating-point vs. integer
        //     - 32 vs. 64 bit (or width in general)
        //       - bool vs. non bool
        //         - signed vs. not signed
        const auto linearize = [](const TBasicType& basicType) -> int {
            switch (basicType) {
            case EbtBool:     return 1;
            case EbtInt:      return 10;
            case EbtUint:     return 11;
            case EbtInt64:    return 20;
            case EbtUint64:   return 21;
            case EbtFloat:    return 100;
            case EbtDouble:   return 110;
            default:          return 0;
            }
        };

        return abs(linearize(to2.getBasicType()) - linearize(from.getBasicType())) <
               abs(linearize(to1.getBasicType()) - linearize(from.getBasicType()));
    };

    // for ambiguity reporting
    bool tie = false;

    // send to the generic selector
    const TFunction* bestMatch = selectFunction(candidateList, call, convertible, better, tie);

    if (bestMatch == nullptr) {
        // If there is nothing selected by allowing only up-conversions (to a larger linearize() value),
        // we instead try down-conversions, which are valid in HLSL, but not preferred if there are any
        // upconversions possible.
        allowOnlyUpConversions = false;
        bestMatch = selectFunction(candidateList, call, convertible, better, tie);
    }

    if (bestMatch == nullptr) {
        error(loc, "no matching overloaded function found", call.getName().c_str(), "");
        return nullptr;
    }

    // For built-ins, we can convert across the arguments.  This will happen in several steps:
    // Step 1:  If there's an exact match, use it.
    // Step 2a: Otherwise, get the operator from the best match and promote arguments:
    // Step 2b: reconstruct the TFunction based on the new arg types
    // Step 3:  Re-select after type promotion is applied, to find proper candidate.
    if (builtIn) {
        // Step 1: If there's an exact match, use it.
        if (call.getMangledName() == bestMatch->getMangledName())
            return bestMatch;

        // Step 2a: Otherwise, get the operator from the best match and promote arguments as if we
        // are that kind of operator.
        if (args != nullptr) {
            // The arg list can be a unary node, or an aggregate.  We have to handle both.
            // We will use the normal promote() facilities, which require an interm node.
            TIntermOperator* promote = nullptr;

            if (call.getParamCount() == 1) {
                promote = new TIntermUnary(bestMatch->getBuiltInOp());
                promote->getAsUnaryNode()->setOperand(args->getAsTyped());
            } else {
                promote = new TIntermAggregate(bestMatch->getBuiltInOp());
                promote->getAsAggregate()->getSequence().swap(args->getAsAggregate()->getSequence());
            }

            if (! intermediate.promote(promote))
                return nullptr;

            // Obtain the promoted arg list.
            if (call.getParamCount() == 1) {
                args = promote->getAsUnaryNode()->getOperand();
            } else {
                promote->getAsAggregate()->getSequence().swap(args->getAsAggregate()->getSequence());
            }
        }

        // Step 2b: reconstruct the TFunction based on the new arg types
        TFunction convertedCall(&call.getName(), call.getType(), call.getBuiltInOp());

        if (args->getAsAggregate()) {
            // Handle aggregates: put all args into the new function call
            for (int arg=0; arg<int(args->getAsAggregate()->getSequence().size()); ++arg) {
                // TODO: But for constness, we could avoid the new & shallowCopy, and use the pointer directly.
                TParameter param = { 0, new TType, nullptr };
                param.type->shallowCopy(args->getAsAggregate()->getSequence()[arg]->getAsTyped()->getType());
                convertedCall.addParameter(param);
            }
        } else if (args->getAsUnaryNode()) {
            // Handle unaries: put all args into the new function call
            TParameter param = { 0, new TType, nullptr };
            param.type->shallowCopy(args->getAsUnaryNode()->getOperand()->getAsTyped()->getType());
            convertedCall.addParameter(param);
        } else if (args->getAsTyped()) {
            // Handle bare e.g, floats, not in an aggregate.
            TParameter param = { 0, new TType, nullptr };
            param.type->shallowCopy(args->getAsTyped()->getType());
            convertedCall.addParameter(param);
        } else {
            assert(0); // unknown argument list.
            return nullptr;
        }

        // Step 3: Re-select after type promotion, to find proper candidate
        // send to the generic selector
        bestMatch = selectFunction(candidateList, convertedCall, convertible, better, tie);

        // At this point, there should be no tie.
    }

    if (tie)
        error(loc, "ambiguous best function under implicit type conversion", call.getName().c_str(), "");

    // Append default parameter values if needed
    if (!tie && bestMatch != nullptr) {
        for (int defParam = call.getParamCount(); defParam < bestMatch->getParamCount(); ++defParam) {
            handleFunctionArgument(&call, args, (*bestMatch)[defParam].defaultValue);
        }
    }

    return bestMatch;
}

//
// Do everything necessary to handle a typedef declaration, for a single symbol.
//
// 'parseType' is the type part of the declaration (to the left)
// 'arraySizes' is the arrayness tagged on the identifier (to the right)
//
void HlslParseContext::declareTypedef(const TSourceLoc& loc, const TString& identifier, const TType& parseType)
{
    TVariable* typeSymbol = new TVariable(&identifier, parseType, true);
    if (! symbolTable.insert(*typeSymbol))
        error(loc, "name already defined", "typedef", identifier.c_str());
}

// Do everything necessary to handle a struct declaration, including
// making IO aliases because HLSL allows mixed IO in a struct that specializes
// based on the usage (input, output, uniform, none).
void HlslParseContext::declareStruct(const TSourceLoc& loc, TString& structName, TType& type)
{
    // If it was named, which means the type can be reused later, add
    // it to the symbol table.  (Unless it's a block, in which
    // case the name is not a type.)
    if (type.getBasicType() == EbtBlock || structName.size() == 0)
        return;

    TVariable* userTypeDef = new TVariable(&structName, type, true);
    if (! symbolTable.insert(*userTypeDef)) {
        error(loc, "redefinition", structName.c_str(), "struct");
        return;
    }

    // See if we need IO aliases for the structure typeList

    const auto condAlloc = [](bool pred, TTypeList*& list) {
        if (pred && list == nullptr)
            list = new TTypeList;
    };

    tIoKinds newLists = { nullptr, nullptr, nullptr }; // allocate for each kind found
    for (auto member = type.getStruct()->begin(); member != type.getStruct()->end(); ++member) {
        condAlloc(hasUniform(member->type->getQualifier()), newLists.uniform);
        condAlloc(  hasInput(member->type->getQualifier()), newLists.input);
        condAlloc( hasOutput(member->type->getQualifier()), newLists.output);

        if (member->type->isStruct()) {
            auto it = ioTypeMap.find(member->type->getStruct());
            if (it != ioTypeMap.end()) {
                condAlloc(it->second.uniform != nullptr, newLists.uniform);
                condAlloc(it->second.input   != nullptr, newLists.input);
                condAlloc(it->second.output  != nullptr, newLists.output);
            }
        }
    }
    if (newLists.uniform == nullptr &&
        newLists.input   == nullptr &&
        newLists.output  == nullptr) {
        // Won't do any IO caching, clear up the type and get out now.
        for (auto member = type.getStruct()->begin(); member != type.getStruct()->end(); ++member)
            clearUniformInputOutput(member->type->getQualifier());
        return;
    }

    // We have IO involved.

    // Make a pure typeList for the symbol table, and cache side copies of IO versions.
    for (auto member = type.getStruct()->begin(); member != type.getStruct()->end(); ++member) {
        const auto inheritStruct = [&](TTypeList* s, TTypeLoc& ioMember) {
            if (s != nullptr) {
                ioMember.type = new TType;
                ioMember.type->shallowCopy(*member->type);
                ioMember.type->setStruct(s);
            }
        };
        const auto newMember = [&](TTypeLoc& m) {
            if (m.type == nullptr) {
                m.type = new TType;
                m.type->shallowCopy(*member->type);
            }
        };

        TTypeLoc newUniformMember = { nullptr, member->loc };
        TTypeLoc newInputMember   = { nullptr, member->loc };
        TTypeLoc newOutputMember  = { nullptr, member->loc };
        if (member->type->isStruct()) {
            // swap in an IO child if there is one
            auto it = ioTypeMap.find(member->type->getStruct());
            if (it != ioTypeMap.end()) {
                inheritStruct(it->second.uniform, newUniformMember);
                inheritStruct(it->second.input,   newInputMember);
                inheritStruct(it->second.output,  newOutputMember);
            }
        }
        if (newLists.uniform) {
            newMember(newUniformMember);

            // inherit default matrix layout (changeable via #pragma pack_matrix), if none given.
            if (member->type->isMatrix() && member->type->getQualifier().layoutMatrix == ElmNone)
                newUniformMember.type->getQualifier().layoutMatrix = globalUniformDefaults.layoutMatrix;

            correctUniform(newUniformMember.type->getQualifier());
            newLists.uniform->push_back(newUniformMember);
        }
        if (newLists.input) {
            newMember(newInputMember);
            correctInput(newInputMember.type->getQualifier());
            newLists.input->push_back(newInputMember);
        }
        if (newLists.output) {
            newMember(newOutputMember);
            correctOutput(newOutputMember.type->getQualifier());
            newLists.output->push_back(newOutputMember);
        }

        // make original pure
        clearUniformInputOutput(member->type->getQualifier());
    }
    ioTypeMap[type.getStruct()] = newLists;
}

// Lookup a user-type by name.
// If found, fill in the type and return the defining symbol.
// If not found, return nullptr.
TSymbol* HlslParseContext::lookupUserType(const TString& typeName, TType& type)
{
    TSymbol* symbol = symbolTable.find(typeName);
    if (symbol && symbol->getAsVariable() && symbol->getAsVariable()->isUserType()) {
        type.shallowCopy(symbol->getType());
        return symbol;
    } else
        return nullptr;
}

//
// Do everything necessary to handle a variable (non-block) declaration.
// Either redeclaring a variable, or making a new one, updating the symbol
// table, and all error checking.
//
// Returns a subtree node that computes an initializer, if needed.
// Returns nullptr if there is no code to execute for initialization.
//
// 'parseType' is the type part of the declaration (to the left)
// 'arraySizes' is the arrayness tagged on the identifier (to the right)
//
TIntermNode* HlslParseContext::declareVariable(const TSourceLoc& loc, const TString& identifier, TType& type,
                                               TIntermTyped* initializer)
{
    if (voidErrorCheck(loc, identifier, type.getBasicType()))
        return nullptr;

    // Global consts with initializers that are non-const act like EvqGlobal in HLSL.
    // This test is implicitly recursive, because initializers propagate constness
    // up the aggregate node tree during creation.  E.g, for:
    //    { { 1, 2 }, { 3, 4 } }
    // the initializer list is marked EvqConst at the top node, and remains so here.  However:
    //    { 1, { myvar, 2 }, 3 }
    // is not a const intializer, and still becomes EvqGlobal here.

    const bool nonConstInitializer = (initializer != nullptr && initializer->getQualifier().storage != EvqConst);

    if (type.getQualifier().storage == EvqConst && symbolTable.atGlobalLevel() && nonConstInitializer) {
        // Force to global
        type.getQualifier().storage = EvqGlobal;
    }

    // make const and initialization consistent
    fixConstInit(loc, identifier, type, initializer);

    // Check for redeclaration of built-ins and/or attempting to declare a reserved name
    TSymbol* symbol = nullptr;

    inheritGlobalDefaults(type.getQualifier());

    const bool flattenVar = shouldFlatten(type, type.getQualifier().storage, true);

    // correct IO in the type
    switch (type.getQualifier().storage) {
    case EvqGlobal:
    case EvqTemporary:
        clearUniformInputOutput(type.getQualifier());
        break;
    case EvqUniform:
    case EvqBuffer:
        correctUniform(type.getQualifier());
        if (type.isStruct()) {
            auto it = ioTypeMap.find(type.getStruct());
            if (it != ioTypeMap.end())
                type.setStruct(it->second.uniform);
        }

        break;
    default:
        break;
    }

    // Declare the variable
    if (type.isArray()) {
        // array case
        declareArray(loc, identifier, type, symbol, !flattenVar);
    } else {
        // non-array case
        if (symbol == nullptr)
            symbol = declareNonArray(loc, identifier, type, !flattenVar);
        else if (type != symbol->getType())
            error(loc, "cannot change the type of", "redeclaration", symbol->getName().c_str());
    }

    if (symbol == nullptr)
        return nullptr;

    if (flattenVar)
        flatten(*symbol->getAsVariable(), symbolTable.atGlobalLevel());

    if (initializer == nullptr)
        return nullptr;

    // Deal with initializer
    TVariable* variable = symbol->getAsVariable();
    if (variable == nullptr) {
        error(loc, "initializer requires a variable, not a member", identifier.c_str(), "");
        return nullptr;
    }
    return executeInitializer(loc, initializer, variable);
}

// Pick up global defaults from the provide global defaults into dst.
void HlslParseContext::inheritGlobalDefaults(TQualifier& dst) const
{
    if (dst.storage == EvqVaryingOut) {
        if (! dst.hasStream() && language == EShLangGeometry)
            dst.layoutStream = globalOutputDefaults.layoutStream;
        if (! dst.hasXfbBuffer())
            dst.layoutXfbBuffer = globalOutputDefaults.layoutXfbBuffer;
    }
}

//
// Make an internal-only variable whose name is for debug purposes only
// and won't be searched for.  Callers will only use the return value to use
// the variable, not the name to look it up.  It is okay if the name
// is the same as other names; there won't be any conflict.
//
TVariable* HlslParseContext::makeInternalVariable(const char* name, const TType& type) const
{
    TString* nameString = NewPoolTString(name);
    TVariable* variable = new TVariable(nameString, type);
    symbolTable.makeInternalVariable(*variable);

    return variable;
}

// Make a symbol node holding a new internal temporary variable.
TIntermSymbol* HlslParseContext::makeInternalVariableNode(const TSourceLoc& loc, const char* name,
                                                          const TType& type) const
{
    TVariable* tmpVar = makeInternalVariable(name, type);
    tmpVar->getWritableType().getQualifier().makeTemporary();

    return intermediate.addSymbol(*tmpVar, loc);
}

//
// Declare a non-array variable, the main point being there is no redeclaration
// for resizing allowed.
//
// Return the successfully declared variable.
//
TVariable* HlslParseContext::declareNonArray(const TSourceLoc& loc, const TString& identifier, const TType& type,
                                             bool track)
{
    // make a new variable
    TVariable* variable = new TVariable(&identifier, type);

    // add variable to symbol table
    if (symbolTable.insert(*variable)) {
        if (track && symbolTable.atGlobalLevel())
            trackLinkage(*variable);
        return variable;
    }

    error(loc, "redefinition", variable->getName().c_str(), "");
    return nullptr;
}

//
// Handle all types of initializers from the grammar.
//
// Returning nullptr just means there is no code to execute to handle the
// initializer, which will, for example, be the case for constant initializers.
//
// Returns a subtree that accomplished the initialization.
//
TIntermNode* HlslParseContext::executeInitializer(const TSourceLoc& loc, TIntermTyped* initializer, TVariable* variable)
{
    //
    // Identifier must be of type constant, a global, or a temporary, and
    // starting at version 120, desktop allows uniforms to have initializers.
    //
    TStorageQualifier qualifier = variable->getType().getQualifier().storage;

    //
    // If the initializer was from braces { ... }, we convert the whole subtree to a
    // constructor-style subtree, allowing the rest of the code to operate
    // identically for both kinds of initializers.
    //
    //
    // Type can't be deduced from the initializer list, so a skeletal type to
    // follow has to be passed in.  Constness and specialization-constness
    // should be deduced bottom up, not dictated by the skeletal type.
    //
    TType skeletalType;
    skeletalType.shallowCopy(variable->getType());
    skeletalType.getQualifier().makeTemporary();
    if (initializer->getAsAggregate() && initializer->getAsAggregate()->getOp() == EOpNull)
        initializer = convertInitializerList(loc, skeletalType, initializer, nullptr);
    if (initializer == nullptr) {
        // error recovery; don't leave const without constant values
        if (qualifier == EvqConst)
            variable->getWritableType().getQualifier().storage = EvqTemporary;
        return nullptr;
    }

    // Fix outer arrayness if variable is unsized, getting size from the initializer
    if (initializer->getType().isSizedArray() && variable->getType().isUnsizedArray())
        variable->getWritableType().changeOuterArraySize(initializer->getType().getOuterArraySize());

    // Inner arrayness can also get set by an initializer
    if (initializer->getType().isArrayOfArrays() && variable->getType().isArrayOfArrays() &&
        initializer->getType().getArraySizes()->getNumDims() ==
        variable->getType().getArraySizes()->getNumDims()) {
        // adopt unsized sizes from the initializer's sizes
        for (int d = 1; d < variable->getType().getArraySizes()->getNumDims(); ++d) {
            if (variable->getType().getArraySizes()->getDimSize(d) == UnsizedArraySize) {
                variable->getWritableType().getArraySizes()->setDimSize(d,
                    initializer->getType().getArraySizes()->getDimSize(d));
            }
        }
    }

    // Uniform and global consts require a constant initializer
    if (qualifier == EvqUniform && initializer->getType().getQualifier().storage != EvqConst) {
        error(loc, "uniform initializers must be constant", "=", "'%s'", variable->getType().getCompleteString().c_str());
        variable->getWritableType().getQualifier().storage = EvqTemporary;
        return nullptr;
    }

    // Const variables require a constant initializer
    if (qualifier == EvqConst) {
        if (initializer->getType().getQualifier().storage != EvqConst) {
            variable->getWritableType().getQualifier().storage = EvqConstReadOnly;
            qualifier = EvqConstReadOnly;
        }
    }

    if (qualifier == EvqConst || qualifier == EvqUniform) {
        // Compile-time tagging of the variable with its constant value...

        initializer = intermediate.addConversion(EOpAssign, variable->getType(), initializer);
        if (initializer != nullptr && variable->getType() != initializer->getType())
            initializer = intermediate.addUniShapeConversion(EOpAssign, variable->getType(), initializer);
        if (initializer == nullptr || !initializer->getAsConstantUnion() ||
                                      variable->getType() != initializer->getType()) {
            error(loc, "non-matching or non-convertible constant type for const initializer",
                variable->getType().getStorageQualifierString(), "");
            variable->getWritableType().getQualifier().storage = EvqTemporary;
            return nullptr;
        }

        variable->setConstArray(initializer->getAsConstantUnion()->getConstArray());
    } else {
        // normal assigning of a value to a variable...
        specializationCheck(loc, initializer->getType(), "initializer");
        TIntermSymbol* intermSymbol = intermediate.addSymbol(*variable, loc);
        TIntermNode* initNode = handleAssign(loc, EOpAssign, intermSymbol, initializer);
        if (initNode == nullptr)
            assignError(loc, "=", intermSymbol->getCompleteString(), initializer->getCompleteString());
        return initNode;
    }

    return nullptr;
}

//
// Reprocess any initializer-list { ... } parts of the initializer.
// Need to hierarchically assign correct types and implicit
// conversions. Will do this mimicking the same process used for
// creating a constructor-style initializer, ensuring we get the
// same form.
//
// Returns a node representing an expression for the initializer list expressed
// as the correct type.
//
// Returns nullptr if there is an error.
//
TIntermTyped* HlslParseContext::convertInitializerList(const TSourceLoc& loc, const TType& type,
                                                       TIntermTyped* initializer, TIntermTyped* scalarInit)
{
    // Will operate recursively.  Once a subtree is found that is constructor style,
    // everything below it is already good: Only the "top part" of the initializer
    // can be an initializer list, where "top part" can extend for several (or all) levels.

    // see if we have bottomed out in the tree within the initializer-list part
    TIntermAggregate* initList = initializer->getAsAggregate();
    if (initList == nullptr || initList->getOp() != EOpNull) {
        // We don't have a list, but if it's a scalar and the 'type' is a
        // composite, we need to lengthen below to make it useful.
        // Otherwise, this is an already formed object to initialize with.
        if (type.isScalar() || !initializer->getType().isScalar())
            return initializer;
        else
            initList = intermediate.makeAggregate(initializer);
    }

    // Of the initializer-list set of nodes, need to process bottom up,
    // so recurse deep, then process on the way up.

    // Go down the tree here...
    if (type.isArray()) {
        // The type's array might be unsized, which could be okay, so base sizes on the size of the aggregate.
        // Later on, initializer execution code will deal with array size logic.
        TType arrayType;
        arrayType.shallowCopy(type);                     // sharing struct stuff is fine
        arrayType.copyArraySizes(*type.getArraySizes()); // but get a fresh copy of the array information, to edit below

        // edit array sizes to fill in unsized dimensions
        if (type.isUnsizedArray())
            arrayType.changeOuterArraySize((int)initList->getSequence().size());

        // set unsized array dimensions that can be derived from the initializer's first element
        if (arrayType.isArrayOfArrays() && initList->getSequence().size() > 0) {
            TIntermTyped* firstInit = initList->getSequence()[0]->getAsTyped();
            if (firstInit->getType().isArray() &&
                arrayType.getArraySizes()->getNumDims() == firstInit->getType().getArraySizes()->getNumDims() + 1) {
                for (int d = 1; d < arrayType.getArraySizes()->getNumDims(); ++d) {
                    if (arrayType.getArraySizes()->getDimSize(d) == UnsizedArraySize)
                        arrayType.getArraySizes()->setDimSize(d, firstInit->getType().getArraySizes()->getDimSize(d - 1));
                }
            }
        }

        // lengthen list to be long enough
        lengthenList(loc, initList->getSequence(), arrayType.getOuterArraySize(), scalarInit);

        // recursively process each element
        TType elementType(arrayType, 0); // dereferenced type
        for (int i = 0; i < arrayType.getOuterArraySize(); ++i) {
            initList->getSequence()[i] = convertInitializerList(loc, elementType,
                                                                initList->getSequence()[i]->getAsTyped(), scalarInit);
            if (initList->getSequence()[i] == nullptr)
                return nullptr;
        }

        return addConstructor(loc, initList, arrayType);
    } else if (type.isStruct()) {
        // do we have implicit assignments to opaques?
        for (size_t i = initList->getSequence().size(); i < type.getStruct()->size(); ++i) {
            if ((*type.getStruct())[i].type->containsOpaque()) {
                error(loc, "cannot implicitly initialize opaque members", "initializer list", "");
                return nullptr;
            }
        }

        // lengthen list to be long enough
        lengthenList(loc, initList->getSequence(), static_cast<int>(type.getStruct()->size()), scalarInit);

        if (type.getStruct()->size() != initList->getSequence().size()) {
            error(loc, "wrong number of structure members", "initializer list", "");
            return nullptr;
        }
        for (size_t i = 0; i < type.getStruct()->size(); ++i) {
            initList->getSequence()[i] = convertInitializerList(loc, *(*type.getStruct())[i].type,
                                                                initList->getSequence()[i]->getAsTyped(), scalarInit);
            if (initList->getSequence()[i] == nullptr)
                return nullptr;
        }
    } else if (type.isMatrix()) {
        if (type.computeNumComponents() == (int)initList->getSequence().size()) {
            // This means the matrix is initialized component-wise, rather than as
            // a series of rows and columns.  We can just use the list directly as
            // a constructor; no further processing needed.
        } else {
            // lengthen list to be long enough
            lengthenList(loc, initList->getSequence(), type.getMatrixCols(), scalarInit);

            if (type.getMatrixCols() != (int)initList->getSequence().size()) {
                error(loc, "wrong number of matrix columns:", "initializer list", type.getCompleteString().c_str());
                return nullptr;
            }
            TType vectorType(type, 0); // dereferenced type
            for (int i = 0; i < type.getMatrixCols(); ++i) {
                initList->getSequence()[i] = convertInitializerList(loc, vectorType,
                                                                    initList->getSequence()[i]->getAsTyped(), scalarInit);
                if (initList->getSequence()[i] == nullptr)
                    return nullptr;
            }
        }
    } else if (type.isVector()) {
        // lengthen list to be long enough
        lengthenList(loc, initList->getSequence(), type.getVectorSize(), scalarInit);

        // error check; we're at bottom, so work is finished below
        if (type.getVectorSize() != (int)initList->getSequence().size()) {
            error(loc, "wrong vector size (or rows in a matrix column):", "initializer list",
                  type.getCompleteString().c_str());
            return nullptr;
        }
    } else if (type.isScalar()) {
        // lengthen list to be long enough
        lengthenList(loc, initList->getSequence(), 1, scalarInit);

        if ((int)initList->getSequence().size() != 1) {
            error(loc, "scalar expected one element:", "initializer list", type.getCompleteString().c_str());
            return nullptr;
        }
    } else {
        error(loc, "unexpected initializer-list type:", "initializer list", type.getCompleteString().c_str());
        return nullptr;
    }

    // Now that the subtree is processed, process this node as if the
    // initializer list is a set of arguments to a constructor.
    TIntermTyped* emulatedConstructorArguments;
    if (initList->getSequence().size() == 1)
        emulatedConstructorArguments = initList->getSequence()[0]->getAsTyped();
    else
        emulatedConstructorArguments = initList;

    return addConstructor(loc, emulatedConstructorArguments, type);
}

// Lengthen list to be long enough to cover any gap from the current list size
// to 'size'. If the list is longer, do nothing.
// The value to lengthen with is the default for short lists.
//
// By default, lists that are too short due to lack of initializers initialize to zero.
// Alternatively, it could be a scalar initializer for a structure. Both cases are handled,
// based on whether something is passed in as 'scalarInit'.
//
// 'scalarInit' must be safe to use each time this is called (no side effects replication).
//
void HlslParseContext::lengthenList(const TSourceLoc& loc, TIntermSequence& list, int size, TIntermTyped* scalarInit)
{
    for (int c = (int)list.size(); c < size; ++c) {
        if (scalarInit == nullptr)
            list.push_back(intermediate.addConstantUnion(0, loc));
        else
            list.push_back(scalarInit);
    }
}

//
// Test for the correctness of the parameters passed to various constructor functions
// and also convert them to the right data type, if allowed and required.
//
// Returns nullptr for an error or the constructed node (aggregate or typed) for no error.
//
TIntermTyped* HlslParseContext::handleConstructor(const TSourceLoc& loc, TIntermTyped* node, const TType& type)
{
    if (node == nullptr)
        return nullptr;

    // Construct identical type
    if (type == node->getType())
        return node;

    // Handle the idiom "(struct type)<scalar value>"
    if (type.isStruct() && isScalarConstructor(node)) {
        // 'node' will almost always get used multiple times, so should not be used directly,
        // it would create a DAG instead of a tree, which might be okay (would
        // like to formalize that for constants and symbols), but if it has
        // side effects, they would get executed multiple times, which is not okay.
        if (node->getAsConstantUnion() == nullptr && node->getAsSymbolNode() == nullptr) {
            TIntermAggregate* seq = intermediate.makeAggregate(loc);
            TIntermSymbol* copy = makeInternalVariableNode(loc, "scalarCopy", node->getType());
            seq = intermediate.growAggregate(seq, intermediate.addBinaryNode(EOpAssign, copy, node, loc));
            seq = intermediate.growAggregate(seq, convertInitializerList(loc, type, intermediate.makeAggregate(loc), copy));
            seq->setOp(EOpComma);
            seq->setType(type);
            return seq;
        } else
            return convertInitializerList(loc, type, intermediate.makeAggregate(loc), node);
    }

    return addConstructor(loc, node, type);
}

// Add a constructor, either from the grammar, or other programmatic reasons.
//
// 'node' is what to construct from.
// 'type' is what type to construct.
//
// Returns the constructed object.
// Return nullptr if it can't be done.
//
TIntermTyped* HlslParseContext::addConstructor(const TSourceLoc& loc, TIntermTyped* node, const TType& type)
{
    TIntermAggregate* aggrNode = node->getAsAggregate();
    TOperator op = intermediate.mapTypeToConstructorOp(type);

    if (op == EOpConstructTextureSampler)
        return intermediate.setAggregateOperator(aggrNode, op, type, loc);

    TTypeList::const_iterator memberTypes;
    if (op == EOpConstructStruct)
        memberTypes = type.getStruct()->begin();

    TType elementType;
    if (type.isArray()) {
        TType dereferenced(type, 0);
        elementType.shallowCopy(dereferenced);
    } else
        elementType.shallowCopy(type);

    bool singleArg;
    if (aggrNode != nullptr) {
        if (aggrNode->getOp() != EOpNull)
            singleArg = true;
        else
            singleArg = false;
    } else
        singleArg = true;

    TIntermTyped *newNode;
    if (singleArg) {
        // Handle array -> array conversion
        // Constructing an array of one type from an array of another type is allowed,
        // assuming there are enough components available (semantic-checked earlier).
        if (type.isArray() && node->isArray())
            newNode = convertArray(node, type);

        // If structure constructor or array constructor is being called
        // for only one parameter inside the aggregate, we need to call constructAggregate function once.
        else if (type.isArray())
            newNode = constructAggregate(node, elementType, 1, node->getLoc());
        else if (op == EOpConstructStruct)
            newNode = constructAggregate(node, *(*memberTypes).type, 1, node->getLoc());
        else {
            // shape conversion for matrix constructor from scalar.  HLSL semantics are: scalar
            // is replicated into every element of the matrix (not just the diagnonal), so
            // that is handled specially here.
            if (type.isMatrix() && node->getType().isScalarOrVec1())
                node = intermediate.addShapeConversion(type, node);

            newNode = constructBuiltIn(type, op, node, node->getLoc(), false);
        }

        if (newNode && (type.isArray() || op == EOpConstructStruct))
            newNode = intermediate.setAggregateOperator(newNode, EOpConstructStruct, type, loc);

        return newNode;
    }

    //
    // Handle list of arguments.
    //
    TIntermSequence& sequenceVector = aggrNode->getSequence();    // Stores the information about the parameter to the constructor
    // if the structure constructor contains more than one parameter, then construct
    // each parameter

    int paramCount = 0;  // keeps a track of the constructor parameter number being checked

    // for each parameter to the constructor call, check to see if the right type is passed or convert them
    // to the right type if possible (and allowed).
    // for structure constructors, just check if the right type is passed, no conversion is allowed.

    for (TIntermSequence::iterator p = sequenceVector.begin();
        p != sequenceVector.end(); p++, paramCount++) {
        if (type.isArray())
            newNode = constructAggregate(*p, elementType, paramCount + 1, node->getLoc());
        else if (op == EOpConstructStruct)
            newNode = constructAggregate(*p, *(memberTypes[paramCount]).type, paramCount + 1, node->getLoc());
        else
            newNode = constructBuiltIn(type, op, (*p)->getAsTyped(), node->getLoc(), true);

        if (newNode)
            *p = newNode;
        else
            return nullptr;
    }

    TIntermTyped* constructor = intermediate.setAggregateOperator(aggrNode, op, type, loc);

    return constructor;
}

// Function for constructor implementation. Calls addUnaryMath with appropriate EOp value
// for the parameter to the constructor (passed to this function). Essentially, it converts
// the parameter types correctly. If a constructor expects an int (like ivec2) and is passed a
// float, then float is converted to int.
//
// Returns nullptr for an error or the constructed node.
//
TIntermTyped* HlslParseContext::constructBuiltIn(const TType& type, TOperator op, TIntermTyped* node,
                                                 const TSourceLoc& loc, bool subset)
{
    TIntermTyped* newNode;
    TOperator basicOp;

    //
    // First, convert types as needed.
    //
    switch (op) {
    case EOpConstructF16Vec2:
    case EOpConstructF16Vec3:
    case EOpConstructF16Vec4:
    case EOpConstructF16Mat2x2:
    case EOpConstructF16Mat2x3:
    case EOpConstructF16Mat2x4:
    case EOpConstructF16Mat3x2:
    case EOpConstructF16Mat3x3:
    case EOpConstructF16Mat3x4:
    case EOpConstructF16Mat4x2:
    case EOpConstructF16Mat4x3:
    case EOpConstructF16Mat4x4:
    case EOpConstructFloat16:
        basicOp = EOpConstructFloat16;
        break;

    case EOpConstructVec2:
    case EOpConstructVec3:
    case EOpConstructVec4:
    case EOpConstructMat2x2:
    case EOpConstructMat2x3:
    case EOpConstructMat2x4:
    case EOpConstructMat3x2:
    case EOpConstructMat3x3:
    case EOpConstructMat3x4:
    case EOpConstructMat4x2:
    case EOpConstructMat4x3:
    case EOpConstructMat4x4:
    case EOpConstructFloat:
        basicOp = EOpConstructFloat;
        break;

    case EOpConstructDVec2:
    case EOpConstructDVec3:
    case EOpConstructDVec4:
    case EOpConstructDMat2x2:
    case EOpConstructDMat2x3:
    case EOpConstructDMat2x4:
    case EOpConstructDMat3x2:
    case EOpConstructDMat3x3:
    case EOpConstructDMat3x4:
    case EOpConstructDMat4x2:
    case EOpConstructDMat4x3:
    case EOpConstructDMat4x4:
    case EOpConstructDouble:
        basicOp = EOpConstructDouble;
        break;

    case EOpConstructI16Vec2:
    case EOpConstructI16Vec3:
    case EOpConstructI16Vec4:
    case EOpConstructInt16:
        basicOp = EOpConstructInt16;
        break;

    case EOpConstructIVec2:
    case EOpConstructIVec3:
    case EOpConstructIVec4:
    case EOpConstructIMat2x2:
    case EOpConstructIMat2x3:
    case EOpConstructIMat2x4:
    case EOpConstructIMat3x2:
    case EOpConstructIMat3x3:
    case EOpConstructIMat3x4:
    case EOpConstructIMat4x2:
    case EOpConstructIMat4x3:
    case EOpConstructIMat4x4:
    case EOpConstructInt:
        basicOp = EOpConstructInt;
        break;

    case EOpConstructU16Vec2:
    case EOpConstructU16Vec3:
    case EOpConstructU16Vec4:
    case EOpConstructUint16:
        basicOp = EOpConstructUint16;
        break;

    case EOpConstructUVec2:
    case EOpConstructUVec3:
    case EOpConstructUVec4:
    case EOpConstructUMat2x2:
    case EOpConstructUMat2x3:
    case EOpConstructUMat2x4:
    case EOpConstructUMat3x2:
    case EOpConstructUMat3x3:
    case EOpConstructUMat3x4:
    case EOpConstructUMat4x2:
    case EOpConstructUMat4x3:
    case EOpConstructUMat4x4:
    case EOpConstructUint:
        basicOp = EOpConstructUint;
        break;

    case EOpConstructBVec2:
    case EOpConstructBVec3:
    case EOpConstructBVec4:
    case EOpConstructBMat2x2:
    case EOpConstructBMat2x3:
    case EOpConstructBMat2x4:
    case EOpConstructBMat3x2:
    case EOpConstructBMat3x3:
    case EOpConstructBMat3x4:
    case EOpConstructBMat4x2:
    case EOpConstructBMat4x3:
    case EOpConstructBMat4x4:
    case EOpConstructBool:
        basicOp = EOpConstructBool;
        break;

    default:
        error(loc, "unsupported construction", "", "");

        return nullptr;
    }
    newNode = intermediate.addUnaryMath(basicOp, node, node->getLoc());
    if (newNode == nullptr) {
        error(loc, "can't convert", "constructor", "");
        return nullptr;
    }

    //
    // Now, if there still isn't an operation to do the construction, and we need one, add one.
    //

    // Otherwise, skip out early.
    if (subset || (newNode != node && newNode->getType() == type))
        return newNode;

    // setAggregateOperator will insert a new node for the constructor, as needed.
    return intermediate.setAggregateOperator(newNode, op, type, loc);
}

// Convert the array in node to the requested type, which is also an array.
// Returns nullptr on failure, otherwise returns aggregate holding the list of
// elements needed to construct the array.
TIntermTyped* HlslParseContext::convertArray(TIntermTyped* node, const TType& type)
{
    assert(node->isArray() && type.isArray());
    if (node->getType().computeNumComponents() < type.computeNumComponents())
        return nullptr;

    // TODO: write an argument replicator, for the case the argument should not be
    // executed multiple times, yet multiple copies are needed.

    TIntermTyped* constructee = node->getAsTyped();
    // track where we are in consuming the argument
    int constructeeElement = 0;
    int constructeeComponent = 0;

    // bump up to the next component to consume
    const auto getNextComponent = [&]() {
        TIntermTyped* component;
        component = handleBracketDereference(node->getLoc(), constructee, 
                                             intermediate.addConstantUnion(constructeeElement, node->getLoc()));
        if (component->isVector())
            component = handleBracketDereference(node->getLoc(), component,
                                                 intermediate.addConstantUnion(constructeeComponent, node->getLoc()));
        // bump component pointer up
        ++constructeeComponent;
        if (constructeeComponent == constructee->getVectorSize()) {
            constructeeComponent = 0;
            ++constructeeElement;
        }
        return component;
    };

    // make one subnode per constructed array element
    TIntermAggregate* constructor = nullptr;
    TType derefType(type, 0);
    TType speculativeComponentType(derefType, 0);
    TType* componentType = derefType.isVector() ? &speculativeComponentType : &derefType;
    TOperator componentOp = intermediate.mapTypeToConstructorOp(*componentType);
    TType crossType(node->getBasicType(), EvqTemporary, type.getVectorSize());
    for (int e = 0; e < type.getOuterArraySize(); ++e) {
        // construct an element
        TIntermTyped* elementArg;
        if (type.getVectorSize() == constructee->getVectorSize()) {
            // same element shape
            elementArg = handleBracketDereference(node->getLoc(), constructee,
                                                  intermediate.addConstantUnion(e, node->getLoc()));
        } else {
            // mismatched element shapes
            if (type.getVectorSize() == 1)
                elementArg = getNextComponent();
            else {
                // make a vector
                TIntermAggregate* elementConstructee = nullptr;
                for (int c = 0; c < type.getVectorSize(); ++c)
                    elementConstructee = intermediate.growAggregate(elementConstructee, getNextComponent());
                elementArg = addConstructor(node->getLoc(), elementConstructee, crossType);
            }
        }
        // convert basic types
        elementArg = intermediate.addConversion(componentOp, derefType, elementArg);
        if (elementArg == nullptr)
            return nullptr;
        // combine with top-level constructor
        constructor = intermediate.growAggregate(constructor, elementArg);
    }

    return constructor;
}

// This function tests for the type of the parameters to the structure or array constructor. Raises
// an error message if the expected type does not match the parameter passed to the constructor.
//
// Returns nullptr for an error or the input node itself if the expected and the given parameter types match.
//
TIntermTyped* HlslParseContext::constructAggregate(TIntermNode* node, const TType& type, int paramCount,
                                                   const TSourceLoc& loc)
{
    // Handle cases that map more 1:1 between constructor arguments and constructed.
    TIntermTyped* converted = intermediate.addConversion(EOpConstructStruct, type, node->getAsTyped());
    if (converted == nullptr || converted->getType() != type) {
        error(loc, "", "constructor", "cannot convert parameter %d from '%s' to '%s'", paramCount,
            node->getAsTyped()->getType().getCompleteString().c_str(), type.getCompleteString().c_str());

        return nullptr;
    }

    return converted;
}

//
// Do everything needed to add an interface block.
//
void HlslParseContext::declareBlock(const TSourceLoc& loc, TType& type, const TString* instanceName)
{
    assert(type.getWritableStruct() != nullptr);

    // Clean up top-level decorations that don't belong.
    switch (type.getQualifier().storage) {
    case EvqUniform:
    case EvqBuffer:
        correctUniform(type.getQualifier());
        break;
    case EvqVaryingIn:
        correctInput(type.getQualifier());
        break;
    case EvqVaryingOut:
        correctOutput(type.getQualifier());
        break;
    default:
        break;
    }

    TTypeList& typeList = *type.getWritableStruct();
    // fix and check for member storage qualifiers and types that don't belong within a block
    for (unsigned int member = 0; member < typeList.size(); ++member) {
        TType& memberType = *typeList[member].type;
        TQualifier& memberQualifier = memberType.getQualifier();
        const TSourceLoc& memberLoc = typeList[member].loc;
        globalQualifierFix(memberLoc, memberQualifier);
        memberQualifier.storage = type.getQualifier().storage;

        if (memberType.isStruct()) {
            // clean up and pick up the right set of decorations
            auto it = ioTypeMap.find(memberType.getStruct());
            switch (type.getQualifier().storage) {
            case EvqUniform:
            case EvqBuffer:
                correctUniform(type.getQualifier());
                if (it != ioTypeMap.end() && it->second.uniform)
                    memberType.setStruct(it->second.uniform);
                break;
            case EvqVaryingIn:
                correctInput(type.getQualifier());
                if (it != ioTypeMap.end() && it->second.input)
                    memberType.setStruct(it->second.input);
                break;
            case EvqVaryingOut:
                correctOutput(type.getQualifier());
                if (it != ioTypeMap.end() && it->second.output)
                    memberType.setStruct(it->second.output);
                break;
            default:
                break;
            }
        }
    }

    // Make default block qualification, and adjust the member qualifications

    TQualifier defaultQualification;
    switch (type.getQualifier().storage) {
    case EvqUniform:    defaultQualification = globalUniformDefaults;    break;
    case EvqBuffer:     defaultQualification = globalBufferDefaults;     break;
    case EvqVaryingIn:  defaultQualification = globalInputDefaults;      break;
    case EvqVaryingOut: defaultQualification = globalOutputDefaults;     break;
    default:            defaultQualification.clear();                    break;
    }

    // Special case for "push_constant uniform", which has a default of std430,
    // contrary to normal uniform defaults, and can't have a default tracked for it.
    if (type.getQualifier().layoutPushConstant && ! type.getQualifier().hasPacking())
        type.getQualifier().layoutPacking = ElpStd430;

    // fix and check for member layout qualifiers

    mergeObjectLayoutQualifiers(defaultQualification, type.getQualifier(), true);

    bool memberWithLocation = false;
    bool memberWithoutLocation = false;
    for (unsigned int member = 0; member < typeList.size(); ++member) {
        TQualifier& memberQualifier = typeList[member].type->getQualifier();
        const TSourceLoc& memberLoc = typeList[member].loc;
        if (memberQualifier.hasStream()) {
            if (defaultQualification.layoutStream != memberQualifier.layoutStream)
                error(memberLoc, "member cannot contradict block", "stream", "");
        }

        // "This includes a block's inheritance of the
        // current global default buffer, a block member's inheritance of the block's
        // buffer, and the requirement that any *xfb_buffer* declared on a block
        // member must match the buffer inherited from the block."
        if (memberQualifier.hasXfbBuffer()) {
            if (defaultQualification.layoutXfbBuffer != memberQualifier.layoutXfbBuffer)
                error(memberLoc, "member cannot contradict block (or what block inherited from global)", "xfb_buffer", "");
        }

        if (memberQualifier.hasLocation()) {
            switch (type.getQualifier().storage) {
            case EvqVaryingIn:
            case EvqVaryingOut:
                memberWithLocation = true;
                break;
            default:
                break;
            }
        } else
            memberWithoutLocation = true;

        TQualifier newMemberQualification = defaultQualification;
        mergeQualifiers(newMemberQualification, memberQualifier);
        memberQualifier = newMemberQualification;
    }

    // Process the members
    fixBlockLocations(loc, type.getQualifier(), typeList, memberWithLocation, memberWithoutLocation);
    fixXfbOffsets(type.getQualifier(), typeList);
    fixBlockUniformOffsets(type.getQualifier(), typeList);

    // reverse merge, so that currentBlockQualifier now has all layout information
    // (can't use defaultQualification directly, it's missing other non-layout-default-class qualifiers)
    mergeObjectLayoutQualifiers(type.getQualifier(), defaultQualification, true);

    //
    // Build and add the interface block as a new type named 'blockName'
    //

    // Use the instance name as the interface name if one exists, else the block name.
    const TString& interfaceName = (instanceName && !instanceName->empty()) ? *instanceName : type.getTypeName();

    TType blockType(&typeList, interfaceName, type.getQualifier());
    if (type.isArray())
        blockType.transferArraySizes(type.getArraySizes());

    // Add the variable, as anonymous or named instanceName.
    // Make an anonymous variable if no name was provided.
    if (instanceName == nullptr)
        instanceName = NewPoolTString("");

    TVariable& variable = *new TVariable(instanceName, blockType);
    if (! symbolTable.insert(variable)) {
        if (*instanceName == "")
            error(loc, "nameless block contains a member that already has a name at global scope",
                  "" /* blockName->c_str() */, "");
        else
            error(loc, "block instance name redefinition", variable.getName().c_str(), "");

        return;
    }

    // Save it in the AST for linker use.
    if (symbolTable.atGlobalLevel())
        trackLinkage(variable);
}

//
// "For a block, this process applies to the entire block, or until the first member
// is reached that has a location layout qualifier. When a block member is declared with a location
// qualifier, its location comes from that qualifier: The member's location qualifier overrides the block-level
// declaration. Subsequent members are again assigned consecutive locations, based on the newest location,
// until the next member declared with a location qualifier. The values used for locations do not have to be
// declared in increasing order."
void HlslParseContext::fixBlockLocations(const TSourceLoc& loc, TQualifier& qualifier, TTypeList& typeList, bool memberWithLocation, bool memberWithoutLocation)
{
    // "If a block has no block-level location layout qualifier, it is required that either all or none of its members
    // have a location layout qualifier, or a compile-time error results."
    if (! qualifier.hasLocation() && memberWithLocation && memberWithoutLocation)
        error(loc, "either the block needs a location, or all members need a location, or no members have a location", "location", "");
    else {
        if (memberWithLocation) {
            // remove any block-level location and make it per *every* member
            int nextLocation = 0;  // by the rule above, initial value is not relevant
            if (qualifier.hasAnyLocation()) {
                nextLocation = qualifier.layoutLocation;
                qualifier.layoutLocation = TQualifier::layoutLocationEnd;
                if (qualifier.hasComponent()) {
                    // "It is a compile-time error to apply the *component* qualifier to a ... block"
                    error(loc, "cannot apply to a block", "component", "");
                }
                if (qualifier.hasIndex()) {
                    error(loc, "cannot apply to a block", "index", "");
                }
            }
            for (unsigned int member = 0; member < typeList.size(); ++member) {
                TQualifier& memberQualifier = typeList[member].type->getQualifier();
                const TSourceLoc& memberLoc = typeList[member].loc;
                if (! memberQualifier.hasLocation()) {
                    if (nextLocation >= (int)TQualifier::layoutLocationEnd)
                        error(memberLoc, "location is too large", "location", "");
                    memberQualifier.layoutLocation = nextLocation;
                    memberQualifier.layoutComponent = 0;
                }
                nextLocation = memberQualifier.layoutLocation +
                               intermediate.computeTypeLocationSize(*typeList[member].type, language);
            }
        }
    }
}

void HlslParseContext::fixXfbOffsets(TQualifier& qualifier, TTypeList& typeList)
{
    // "If a block is qualified with xfb_offset, all its
    // members are assigned transform feedback buffer offsets. If a block is not qualified with xfb_offset, any
    // members of that block not qualified with an xfb_offset will not be assigned transform feedback buffer
    // offsets."

    if (! qualifier.hasXfbBuffer() || ! qualifier.hasXfbOffset())
        return;

    int nextOffset = qualifier.layoutXfbOffset;
    for (unsigned int member = 0; member < typeList.size(); ++member) {
        TQualifier& memberQualifier = typeList[member].type->getQualifier();
        bool contains64BitType = false;
#ifdef AMD_EXTENSIONS
        bool contains32BitType = false;
        bool contains16BitType = false;
        int memberSize = intermediate.computeTypeXfbSize(*typeList[member].type, contains64BitType, contains32BitType, contains16BitType);
#else
        int memberSize = intermediate.computeTypeXfbSize(*typeList[member].type, contains64BitType);
#endif
        // see if we need to auto-assign an offset to this member
        if (! memberQualifier.hasXfbOffset()) {
            // "if applied to an aggregate containing a double or 64-bit integer, the offset must also be a multiple of 8"
            if (contains64BitType)
                RoundToPow2(nextOffset, 8);
#ifdef AMD_EXTENSIONS
            else if (contains32BitType)
                RoundToPow2(nextOffset, 4);
            // "if applied to an aggregate containing a half float or 16-bit integer, the offset must also be a multiple of 2"
            else if (contains16BitType)
                RoundToPow2(nextOffset, 2);
#endif
            memberQualifier.layoutXfbOffset = nextOffset;
        } else
            nextOffset = memberQualifier.layoutXfbOffset;
        nextOffset += memberSize;
    }

    // The above gave all block members an offset, so we can take it off the block now,
    // which will avoid double counting the offset usage.
    qualifier.layoutXfbOffset = TQualifier::layoutXfbOffsetEnd;
}

// Calculate and save the offset of each block member, using the recursively
// defined block offset rules and the user-provided offset and align.
//
// Also, compute and save the total size of the block. For the block's size, arrayness
// is not taken into account, as each element is backed by a separate buffer.
//
void HlslParseContext::fixBlockUniformOffsets(const TQualifier& qualifier, TTypeList& typeList)
{
    if (! qualifier.isUniformOrBuffer())
        return;
    if (qualifier.layoutPacking != ElpStd140 && qualifier.layoutPacking != ElpStd430 && qualifier.layoutPacking != ElpScalar)
        return;

    int offset = 0;
    int memberSize;
    for (unsigned int member = 0; member < typeList.size(); ++member) {
        TQualifier& memberQualifier = typeList[member].type->getQualifier();
        const TSourceLoc& memberLoc = typeList[member].loc;

        // "When align is applied to an array, it effects only the start of the array, not the array's internal stride."

        // modify just the children's view of matrix layout, if there is one for this member
        TLayoutMatrix subMatrixLayout = typeList[member].type->getQualifier().layoutMatrix;
        int dummyStride;
        int memberAlignment = intermediate.getMemberAlignment(*typeList[member].type, memberSize, dummyStride,
                                                              qualifier.layoutPacking,
                                                              subMatrixLayout != ElmNone
                                                                  ? subMatrixLayout == ElmRowMajor
                                                                  : qualifier.layoutMatrix == ElmRowMajor);
        if (memberQualifier.hasOffset()) {
            // "The specified offset must be a multiple
            // of the base alignment of the type of the block member it qualifies, or a compile-time error results."
            if (! IsMultipleOfPow2(memberQualifier.layoutOffset, memberAlignment))
                error(memberLoc, "must be a multiple of the member's alignment", "offset", "");

            // "The offset qualifier forces the qualified member to start at or after the specified
            // integral-constant expression, which will be its byte offset from the beginning of the buffer.
            // "The actual offset of a member is computed as
            // follows: If offset was declared, start with that offset, otherwise start with the next available offset."
            offset = std::max(offset, memberQualifier.layoutOffset);
        }

        // "The actual alignment of a member will be the greater of the specified align alignment and the standard
        // (e.g., std140) base alignment for the member's type."
        if (memberQualifier.hasAlign())
            memberAlignment = std::max(memberAlignment, memberQualifier.layoutAlign);

        // "If the resulting offset is not a multiple of the actual alignment,
        // increase it to the first offset that is a multiple of
        // the actual alignment."
        RoundToPow2(offset, memberAlignment);
        typeList[member].type->getQualifier().layoutOffset = offset;
        offset += memberSize;
    }
}

// For an identifier that is already declared, add more qualification to it.
void HlslParseContext::addQualifierToExisting(const TSourceLoc& loc, TQualifier qualifier, const TString& identifier)
{
    TSymbol* symbol = symbolTable.find(identifier);
    if (symbol == nullptr) {
        error(loc, "identifier not previously declared", identifier.c_str(), "");
        return;
    }
    if (symbol->getAsFunction()) {
        error(loc, "cannot re-qualify a function name", identifier.c_str(), "");
        return;
    }

    if (qualifier.isAuxiliary() ||
        qualifier.isMemory() ||
        qualifier.isInterpolation() ||
        qualifier.hasLayout() ||
        qualifier.storage != EvqTemporary ||
        qualifier.precision != EpqNone) {
        error(loc, "cannot add storage, auxiliary, memory, interpolation, layout, or precision qualifier to an existing variable", identifier.c_str(), "");
        return;
    }

    // For read-only built-ins, add a new symbol for holding the modified qualifier.
    // This will bring up an entire block, if a block type has to be modified (e.g., gl_Position inside a block)
    if (symbol->isReadOnly())
        symbol = symbolTable.copyUp(symbol);

    if (qualifier.invariant) {
        if (intermediate.inIoAccessed(identifier))
            error(loc, "cannot change qualification after use", "invariant", "");
        symbol->getWritableType().getQualifier().invariant = true;
    } else if (qualifier.noContraction) {
        if (intermediate.inIoAccessed(identifier))
            error(loc, "cannot change qualification after use", "precise", "");
        symbol->getWritableType().getQualifier().noContraction = true;
    } else if (qualifier.specConstant) {
        symbol->getWritableType().getQualifier().makeSpecConstant();
        if (qualifier.hasSpecConstantId())
            symbol->getWritableType().getQualifier().layoutSpecConstantId = qualifier.layoutSpecConstantId;
    } else
        warn(loc, "unknown requalification", "", "");
}

void HlslParseContext::addQualifierToExisting(const TSourceLoc& loc, TQualifier qualifier, TIdentifierList& identifiers)
{
    for (unsigned int i = 0; i < identifiers.size(); ++i)
        addQualifierToExisting(loc, qualifier, *identifiers[i]);
}

//
// Update the intermediate for the given input geometry
//
bool HlslParseContext::handleInputGeometry(const TSourceLoc& loc, const TLayoutGeometry& geometry)
{
    switch (geometry) {
    case ElgPoints:             // fall through
    case ElgLines:              // ...
    case ElgTriangles:          // ...
    case ElgLinesAdjacency:     // ...
    case ElgTrianglesAdjacency: // ...
        if (! intermediate.setInputPrimitive(geometry)) {
            error(loc, "input primitive geometry redefinition", TQualifier::getGeometryString(geometry), "");
            return false;
        }
        break;

    default:
        error(loc, "cannot apply to 'in'", TQualifier::getGeometryString(geometry), "");
        return false;
    }

    return true;
}

//
// Update the intermediate for the given output geometry
//
bool HlslParseContext::handleOutputGeometry(const TSourceLoc& loc, const TLayoutGeometry& geometry)
{
    // If this is not a geometry shader, ignore.  It might be a mixed shader including several stages.
    // Since that's an OK situation, return true for success.
    if (language != EShLangGeometry)
        return true;

    switch (geometry) {
    case ElgPoints:
    case ElgLineStrip:
    case ElgTriangleStrip:
        if (! intermediate.setOutputPrimitive(geometry)) {
            error(loc, "output primitive geometry redefinition", TQualifier::getGeometryString(geometry), "");
            return false;
        }
        break;
    default:
        error(loc, "cannot apply to 'out'", TQualifier::getGeometryString(geometry), "");
        return false;
    }

    return true;
}

//
// Selection attributes
//
void HlslParseContext::handleSelectionAttributes(const TSourceLoc& loc, TIntermSelection* selection,
    const TAttributes& attributes)
{
    if (selection == nullptr)
        return;

    for (auto it = attributes.begin(); it != attributes.end(); ++it) {
        switch (it->name) {
        case EatFlatten:
            selection->setFlatten();
            break;
        case EatBranch:
            selection->setDontFlatten();
            break;
        default:
            warn(loc, "attribute does not apply to a selection", "", "");
            break;
        }
    }
}

//
// Switch attributes
//
void HlslParseContext::handleSwitchAttributes(const TSourceLoc& loc, TIntermSwitch* selection,
    const TAttributes& attributes)
{
    if (selection == nullptr)
        return;

    for (auto it = attributes.begin(); it != attributes.end(); ++it) {
        switch (it->name) {
        case EatFlatten:
            selection->setFlatten();
            break;
        case EatBranch:
            selection->setDontFlatten();
            break;
        default:
            warn(loc, "attribute does not apply to a switch", "", "");
            break;
        }
    }
}

//
// Loop attributes
//
void HlslParseContext::handleLoopAttributes(const TSourceLoc& loc, TIntermLoop* loop,
    const TAttributes& attributes)
{
    if (loop == nullptr)
        return;

    for (auto it = attributes.begin(); it != attributes.end(); ++it) {
        switch (it->name) {
        case EatUnroll:
            loop->setUnroll();
            break;
        case EatLoop:
            loop->setDontUnroll();
            break;
        default:
            warn(loc, "attribute does not apply to a loop", "", "");
            break;
        }
    }
}

//
// Updating default qualifier for the case of a declaration with just a qualifier,
// no type, block, or identifier.
//
void HlslParseContext::updateStandaloneQualifierDefaults(const TSourceLoc& loc, const TPublicType& publicType)
{
    if (publicType.shaderQualifiers.vertices != TQualifier::layoutNotSet) {
        assert(language == EShLangTessControl || language == EShLangGeometry);
        // const char* id = (language == EShLangTessControl) ? "vertices" : "max_vertices";
    }
    if (publicType.shaderQualifiers.invocations != TQualifier::layoutNotSet) {
        if (! intermediate.setInvocations(publicType.shaderQualifiers.invocations))
            error(loc, "cannot change previously set layout value", "invocations", "");
    }
    if (publicType.shaderQualifiers.geometry != ElgNone) {
        if (publicType.qualifier.storage == EvqVaryingIn) {
            switch (publicType.shaderQualifiers.geometry) {
            case ElgPoints:
            case ElgLines:
            case ElgLinesAdjacency:
            case ElgTriangles:
            case ElgTrianglesAdjacency:
            case ElgQuads:
            case ElgIsolines:
                break;
            default:
                error(loc, "cannot apply to input", TQualifier::getGeometryString(publicType.shaderQualifiers.geometry),
                      "");
            }
        } else if (publicType.qualifier.storage == EvqVaryingOut) {
            handleOutputGeometry(loc, publicType.shaderQualifiers.geometry);
        } else
            error(loc, "cannot apply to:", TQualifier::getGeometryString(publicType.shaderQualifiers.geometry),
                  GetStorageQualifierString(publicType.qualifier.storage));
    }
    if (publicType.shaderQualifiers.spacing != EvsNone)
        intermediate.setVertexSpacing(publicType.shaderQualifiers.spacing);
    if (publicType.shaderQualifiers.order != EvoNone)
        intermediate.setVertexOrder(publicType.shaderQualifiers.order);
    if (publicType.shaderQualifiers.pointMode)
        intermediate.setPointMode();
    for (int i = 0; i < 3; ++i) {
        if (publicType.shaderQualifiers.localSize[i] > 1) {
            int max = 0;
            switch (i) {
            case 0: max = resources.maxComputeWorkGroupSizeX; break;
            case 1: max = resources.maxComputeWorkGroupSizeY; break;
            case 2: max = resources.maxComputeWorkGroupSizeZ; break;
            default: break;
            }
            if (intermediate.getLocalSize(i) > (unsigned int)max)
                error(loc, "too large; see gl_MaxComputeWorkGroupSize", "local_size", "");

            // Fix the existing constant gl_WorkGroupSize with this new information.
            TVariable* workGroupSize = getEditableVariable("gl_WorkGroupSize");
            workGroupSize->getWritableConstArray()[i].setUConst(intermediate.getLocalSize(i));
        }
        if (publicType.shaderQualifiers.localSizeSpecId[i] != TQualifier::layoutNotSet) {
            intermediate.setLocalSizeSpecId(i, publicType.shaderQualifiers.localSizeSpecId[i]);
            // Set the workgroup built-in variable as a specialization constant
            TVariable* workGroupSize = getEditableVariable("gl_WorkGroupSize");
            workGroupSize->getWritableType().getQualifier().specConstant = true;
        }
    }
    if (publicType.shaderQualifiers.earlyFragmentTests)
        intermediate.setEarlyFragmentTests();

    const TQualifier& qualifier = publicType.qualifier;

    switch (qualifier.storage) {
    case EvqUniform:
        if (qualifier.hasMatrix())
            globalUniformDefaults.layoutMatrix = qualifier.layoutMatrix;
        if (qualifier.hasPacking())
            globalUniformDefaults.layoutPacking = qualifier.layoutPacking;
        break;
    case EvqBuffer:
        if (qualifier.hasMatrix())
            globalBufferDefaults.layoutMatrix = qualifier.layoutMatrix;
        if (qualifier.hasPacking())
            globalBufferDefaults.layoutPacking = qualifier.layoutPacking;
        break;
    case EvqVaryingIn:
        break;
    case EvqVaryingOut:
        if (qualifier.hasStream())
            globalOutputDefaults.layoutStream = qualifier.layoutStream;
        if (qualifier.hasXfbBuffer())
            globalOutputDefaults.layoutXfbBuffer = qualifier.layoutXfbBuffer;
        if (globalOutputDefaults.hasXfbBuffer() && qualifier.hasXfbStride()) {
            if (! intermediate.setXfbBufferStride(globalOutputDefaults.layoutXfbBuffer, qualifier.layoutXfbStride))
                error(loc, "all stride settings must match for xfb buffer", "xfb_stride", "%d",
                      qualifier.layoutXfbBuffer);
        }
        break;
    default:
        error(loc, "default qualifier requires 'uniform', 'buffer', 'in', or 'out' storage qualification", "", "");
        return;
    }
}

//
// Take the sequence of statements that has been built up since the last case/default,
// put it on the list of top-level nodes for the current (inner-most) switch statement,
// and follow that by the case/default we are on now.  (See switch topology comment on
// TIntermSwitch.)
//
void HlslParseContext::wrapupSwitchSubsequence(TIntermAggregate* statements, TIntermNode* branchNode)
{
    TIntermSequence* switchSequence = switchSequenceStack.back();

    if (statements) {
        statements->setOperator(EOpSequence);
        switchSequence->push_back(statements);
    }
    if (branchNode) {
        // check all previous cases for the same label (or both are 'default')
        for (unsigned int s = 0; s < switchSequence->size(); ++s) {
            TIntermBranch* prevBranch = (*switchSequence)[s]->getAsBranchNode();
            if (prevBranch) {
                TIntermTyped* prevExpression = prevBranch->getExpression();
                TIntermTyped* newExpression = branchNode->getAsBranchNode()->getExpression();
                if (prevExpression == nullptr && newExpression == nullptr)
                    error(branchNode->getLoc(), "duplicate label", "default", "");
                else if (prevExpression != nullptr &&
                    newExpression != nullptr &&
                    prevExpression->getAsConstantUnion() &&
                    newExpression->getAsConstantUnion() &&
                    prevExpression->getAsConstantUnion()->getConstArray()[0].getIConst() ==
                    newExpression->getAsConstantUnion()->getConstArray()[0].getIConst())
                    error(branchNode->getLoc(), "duplicated value", "case", "");
            }
        }
        switchSequence->push_back(branchNode);
    }
}

//
// Turn the top-level node sequence built up of wrapupSwitchSubsequence
// into a switch node.
//
TIntermNode* HlslParseContext::addSwitch(const TSourceLoc& loc, TIntermTyped* expression,
                                         TIntermAggregate* lastStatements, const TAttributes& attributes)
{
    wrapupSwitchSubsequence(lastStatements, nullptr);

    if (expression == nullptr ||
        (expression->getBasicType() != EbtInt && expression->getBasicType() != EbtUint) ||
        expression->getType().isArray() || expression->getType().isMatrix() || expression->getType().isVector())
        error(loc, "condition must be a scalar integer expression", "switch", "");

    // If there is nothing to do, drop the switch but still execute the expression
    TIntermSequence* switchSequence = switchSequenceStack.back();
    if (switchSequence->size() == 0)
        return expression;

    if (lastStatements == nullptr) {
        // emulate a break for error recovery
        lastStatements = intermediate.makeAggregate(intermediate.addBranch(EOpBreak, loc));
        lastStatements->setOperator(EOpSequence);
        switchSequence->push_back(lastStatements);
    }

    TIntermAggregate* body = new TIntermAggregate(EOpSequence);
    body->getSequence() = *switchSequenceStack.back();
    body->setLoc(loc);

    TIntermSwitch* switchNode = new TIntermSwitch(expression, body);
    switchNode->setLoc(loc);
    handleSwitchAttributes(loc, switchNode, attributes);

    return switchNode;
}

// Make a new symbol-table level that is made out of the members of a structure.
// This should be done as an anonymous struct (name is "") so that the symbol table
// finds the members with no explicit reference to a 'this' variable.
void HlslParseContext::pushThisScope(const TType& thisStruct, const TVector<TFunctionDeclarator>& functionDeclarators)
{
    // member variables
    TVariable& thisVariable = *new TVariable(NewPoolTString(""), thisStruct);
    symbolTable.pushThis(thisVariable);

    // member functions
    for (auto it = functionDeclarators.begin(); it != functionDeclarators.end(); ++it) {
        // member should have a prefix matching currentTypePrefix.back()
        // but, symbol lookup within the class scope will just use the
        // unprefixed name. Hence, there are two: one fully prefixed and
        // one with no prefix.
        TFunction& member = *it->function->clone();
        member.removePrefix(currentTypePrefix.back());
        symbolTable.insert(member);
    }
}

// Track levels of class/struct/namespace nesting with a prefix string using
// the type names separated by the scoping operator. E.g., two levels
// would look like:
//
//   outer::inner
//
// The string is empty when at normal global level.
//
void HlslParseContext::pushNamespace(const TString& typeName)
{
    // make new type prefix
    TString newPrefix;
    if (currentTypePrefix.size() > 0)
        newPrefix = currentTypePrefix.back();
    newPrefix.append(typeName);
    newPrefix.append(scopeMangler);
    currentTypePrefix.push_back(newPrefix);
}

// Opposite of pushNamespace(), see above
void HlslParseContext::popNamespace()
{
    currentTypePrefix.pop_back();
}

// Use the class/struct nesting string to create a global name for
// a member of a class/struct.
void HlslParseContext::getFullNamespaceName(TString*& name) const
{
    if (currentTypePrefix.size() == 0)
        return;

    TString* fullName = NewPoolTString(currentTypePrefix.back().c_str());
    fullName->append(*name);
    name = fullName;
}

// Helper function to add the namespace scope mangling syntax to a string.
void HlslParseContext::addScopeMangler(TString& name)
{
    name.append(scopeMangler);
}

// Return true if this has uniform-interface like decorations.
bool HlslParseContext::hasUniform(const TQualifier& qualifier) const
{
    return qualifier.hasUniformLayout() ||
           qualifier.layoutPushConstant;
}

// Potentially not the opposite of hasUniform(), as if some characteristic is
// ever used for more than one thing (e.g., uniform or input), hasUniform() should
// say it exists, but clearUniform() should leave it in place.
void HlslParseContext::clearUniform(TQualifier& qualifier)
{
    qualifier.clearUniformLayout();
    qualifier.layoutPushConstant = false;
}

// Return false if builtIn by itself doesn't force this qualifier to be an input qualifier.
bool HlslParseContext::isInputBuiltIn(const TQualifier& qualifier) const
{
    switch (qualifier.builtIn) {
    case EbvPosition:
    case EbvPointSize:
        return language != EShLangVertex && language != EShLangCompute && language != EShLangFragment;
    case EbvClipDistance:
    case EbvCullDistance:
        return language != EShLangVertex && language != EShLangCompute;
    case EbvFragCoord:
    case EbvFace:
    case EbvHelperInvocation:
    case EbvLayer:
    case EbvPointCoord:
    case EbvSampleId:
    case EbvSampleMask:
    case EbvSamplePosition:
    case EbvViewportIndex:
        return language == EShLangFragment;
    case EbvGlobalInvocationId:
    case EbvLocalInvocationIndex:
    case EbvLocalInvocationId:
    case EbvNumWorkGroups:
    case EbvWorkGroupId:
    case EbvWorkGroupSize:
        return language == EShLangCompute;
    case EbvInvocationId:
        return language == EShLangTessControl || language == EShLangTessEvaluation || language == EShLangGeometry;
    case EbvPatchVertices:
        return language == EShLangTessControl || language == EShLangTessEvaluation;
    case EbvInstanceId:
    case EbvInstanceIndex:
    case EbvVertexId:
    case EbvVertexIndex:
        return language == EShLangVertex;
    case EbvPrimitiveId:
        return language == EShLangGeometry || language == EShLangFragment || language == EShLangTessControl;
    case EbvTessLevelInner:
    case EbvTessLevelOuter:
        return language == EShLangTessEvaluation;
    case EbvTessCoord:
        return language == EShLangTessEvaluation;
    default:
        return false;
    }
}

// Return true if there are decorations to preserve for input-like storage.
bool HlslParseContext::hasInput(const TQualifier& qualifier) const
{
    if (qualifier.hasAnyLocation())
        return true;

    if (language == EShLangFragment && (qualifier.isInterpolation() || qualifier.centroid || qualifier.sample))
        return true;

    if (language == EShLangTessEvaluation && qualifier.patch)
        return true;

    if (isInputBuiltIn(qualifier))
        return true;

    return false;
}

// Return false if builtIn by itself doesn't force this qualifier to be an output qualifier.
bool HlslParseContext::isOutputBuiltIn(const TQualifier& qualifier) const
{
    switch (qualifier.builtIn) {
    case EbvPosition:
    case EbvPointSize:
    case EbvClipVertex:
    case EbvClipDistance:
    case EbvCullDistance:
        return language != EShLangFragment && language != EShLangCompute;
    case EbvFragDepth:
    case EbvFragDepthGreater:
    case EbvFragDepthLesser:
    case EbvSampleMask:
        return language == EShLangFragment;
    case EbvLayer:
    case EbvViewportIndex:
        return language == EShLangGeometry || language == EShLangVertex;
    case EbvPrimitiveId:
        return language == EShLangGeometry;
    case EbvTessLevelInner:
    case EbvTessLevelOuter:
        return language == EShLangTessControl;
    default:
        return false;
    }
}

// Return true if there are decorations to preserve for output-like storage.
bool HlslParseContext::hasOutput(const TQualifier& qualifier) const
{
    if (qualifier.hasAnyLocation())
        return true;

    if (language != EShLangFragment && language != EShLangCompute && qualifier.hasXfb())
        return true;

    if (language == EShLangTessControl && qualifier.patch)
        return true;

    if (language == EShLangGeometry && qualifier.hasStream())
        return true;

    if (isOutputBuiltIn(qualifier))
        return true;

    return false;
}

// Make the IO decorations etc. be appropriate only for an input interface.
void HlslParseContext::correctInput(TQualifier& qualifier)
{
    clearUniform(qualifier);
    if (language == EShLangVertex)
        qualifier.clearInterstage();
    if (language != EShLangTessEvaluation)
        qualifier.patch = false;
    if (language != EShLangFragment) {
        qualifier.clearInterpolation();
        qualifier.sample = false;
    }

    qualifier.clearStreamLayout();
    qualifier.clearXfbLayout();

    if (! isInputBuiltIn(qualifier))
        qualifier.builtIn = EbvNone;
}

// Make the IO decorations etc. be appropriate only for an output interface.
void HlslParseContext::correctOutput(TQualifier& qualifier)
{
    clearUniform(qualifier);
    if (language == EShLangFragment)
        qualifier.clearInterstage();
    if (language != EShLangGeometry)
        qualifier.clearStreamLayout();
    if (language == EShLangFragment)
        qualifier.clearXfbLayout();
    if (language != EShLangTessControl)
        qualifier.patch = false;

    switch (qualifier.builtIn) {
    case EbvFragDepth:
        intermediate.setDepthReplacing();
        intermediate.setDepth(EldAny);
        break;
    case EbvFragDepthGreater:
        intermediate.setDepthReplacing();
        intermediate.setDepth(EldGreater);
        qualifier.builtIn = EbvFragDepth;
        break;
    case EbvFragDepthLesser:
        intermediate.setDepthReplacing();
        intermediate.setDepth(EldLess);
        qualifier.builtIn = EbvFragDepth;
        break;
    default:
        break;
    }

    if (! isOutputBuiltIn(qualifier))
        qualifier.builtIn = EbvNone;
}

// Make the IO decorations etc. be appropriate only for uniform type interfaces.
void HlslParseContext::correctUniform(TQualifier& qualifier)
{
    if (qualifier.declaredBuiltIn == EbvNone)
        qualifier.declaredBuiltIn = qualifier.builtIn;

    qualifier.builtIn = EbvNone;
    qualifier.clearInterstage();
    qualifier.clearInterstageLayout();
}

// Clear out all IO/Uniform stuff, so this has nothing to do with being an IO interface.
void HlslParseContext::clearUniformInputOutput(TQualifier& qualifier)
{
    clearUniform(qualifier);
    correctUniform(qualifier);
}


// Set texture return type.  Returns success (not all types are valid).
bool HlslParseContext::setTextureReturnType(TSampler& sampler, const TType& retType, const TSourceLoc& loc)
{
    // Seed the output with an invalid index.  We will set it to a valid one if we can.
    sampler.structReturnIndex = TSampler::noReturnStruct;

    // Arrays aren't supported.
    if (retType.isArray()) {
        error(loc, "Arrays not supported in texture template types", "", "");
        return false;
    }

    // If return type is a vector, remember the vector size in the sampler, and return.
    if (retType.isVector() || retType.isScalar()) {
        sampler.vectorSize = retType.getVectorSize();
        return true;
    }

    // If it wasn't a vector, it must be a struct meeting certain requirements.  The requirements
    // are checked below: just check for struct-ness here.
    if (!retType.isStruct()) {
        error(loc, "Invalid texture template type", "", "");
        return false;
    }

    // TODO: Subpass doesn't handle struct returns, due to some oddities with fn overloading.
    if (sampler.isSubpass()) {
        error(loc, "Unimplemented: structure template type in subpass input", "", "");
        return false;
    }

    TTypeList* members = retType.getWritableStruct();

    // Check for too many or not enough structure members.
    if (members->size() > 4 || members->size() == 0) {
        error(loc, "Invalid member count in texture template structure", "", "");
        return false;
    }

    // Error checking: We must have <= 4 total components, all of the same basic type.
    unsigned totalComponents = 0;
    for (unsigned m = 0; m < members->size(); ++m) {
        // Check for bad member types
        if (!(*members)[m].type->isScalar() && !(*members)[m].type->isVector()) {
            error(loc, "Invalid texture template struct member type", "", "");
            return false;
        }

        const unsigned memberVectorSize = (*members)[m].type->getVectorSize();
        totalComponents += memberVectorSize;

        // too many total member components
        if (totalComponents > 4) {
            error(loc, "Too many components in texture template structure type", "", "");
            return false;
        }

        // All members must be of a common basic type
        if ((*members)[m].type->getBasicType() != (*members)[0].type->getBasicType()) {
            error(loc, "Texture template structure members must same basic type", "", "");
            return false;
        }
    }

    // If the structure in the return type already exists in the table, we'll use it.  Otherwise, we'll make
    // a new entry.  This is a linear search, but it hardly ever happens, and the list cannot be very large.
    for (unsigned int idx = 0; idx < textureReturnStruct.size(); ++idx) {
        if (textureReturnStruct[idx] == members) {
            sampler.structReturnIndex = idx;
            return true;
        }
    }

    // It wasn't found as an existing entry.  See if we have room for a new one.
    if (textureReturnStruct.size() >= TSampler::structReturnSlots) {
        error(loc, "Texture template struct return slots exceeded", "", "");
        return false;
    }

    // Insert it in the vector that tracks struct return types.
    sampler.structReturnIndex = unsigned(textureReturnStruct.size());
    textureReturnStruct.push_back(members);
    
    // Success!
    return true;
}

// Return the sampler return type in retType.
void HlslParseContext::getTextureReturnType(const TSampler& sampler, TType& retType) const
{
    if (sampler.hasReturnStruct()) {
        assert(textureReturnStruct.size() >= sampler.structReturnIndex);

        // We land here if the texture return is a structure.
        TTypeList* blockStruct = textureReturnStruct[sampler.structReturnIndex];

        const TType resultType(blockStruct, "");
        retType.shallowCopy(resultType);
    } else {
        // We land here if the texture return is a vector or scalar.
        const TType resultType(sampler.type, EvqTemporary, sampler.getVectorSize());
        retType.shallowCopy(resultType);
    }
}


// Return a symbol for the tessellation linkage variable of the given TBuiltInVariable type
TIntermSymbol* HlslParseContext::findTessLinkageSymbol(TBuiltInVariable biType) const
{
    const auto it = builtInTessLinkageSymbols.find(biType);
    if (it == builtInTessLinkageSymbols.end())  // if it wasn't declared by the user, return nullptr
        return nullptr;

    return intermediate.addSymbol(*it->second->getAsVariable());
}

// Find the patch constant function (issues error, returns nullptr if not found)
const TFunction* HlslParseContext::findPatchConstantFunction(const TSourceLoc& loc)
{
    if (symbolTable.isFunctionNameVariable(patchConstantFunctionName)) {
        error(loc, "can't use variable in patch constant function", patchConstantFunctionName.c_str(), "");
        return nullptr;
    }

    const TString mangledName = patchConstantFunctionName + "(";

    // create list of PCF candidates
    TVector<const TFunction*> candidateList;
    bool builtIn;
    symbolTable.findFunctionNameList(mangledName, candidateList, builtIn);
    
    // We have to have one and only one, or we don't know which to pick: the patchconstantfunc does not
    // allow any disambiguation of overloads.
    if (candidateList.empty()) {
        error(loc, "patch constant function not found", patchConstantFunctionName.c_str(), "");
        return nullptr;
    }

    // Based on directed experiments, it appears that if there are overloaded patchconstantfunctions,
    // HLSL picks the last one in shader source order.  Since that isn't yet implemented here, error
    // out if there is more than one candidate.
    if (candidateList.size() > 1) {
        error(loc, "ambiguous patch constant function", patchConstantFunctionName.c_str(), "");
        return nullptr;
    }

    return candidateList[0];
}

// Finalization step: Add patch constant function invocation
void HlslParseContext::addPatchConstantInvocation()
{
    TSourceLoc loc;
    loc.init();

    // If there's no patch constant function, or we're not a HS, do nothing.
    if (patchConstantFunctionName.empty() || language != EShLangTessControl)
        return;

    // Look for built-in variables in a function's parameter list.
    const auto findBuiltIns = [&](const TFunction& function, std::set<tInterstageIoData>& builtIns) {
        for (int p=0; p<function.getParamCount(); ++p) {
            TStorageQualifier storage = function[p].type->getQualifier().storage;

            if (storage == EvqConstReadOnly) // treated identically to input
                storage = EvqIn;

            if (function[p].getDeclaredBuiltIn() != EbvNone)
                builtIns.insert(HlslParseContext::tInterstageIoData(function[p].getDeclaredBuiltIn(), storage));
            else
                builtIns.insert(HlslParseContext::tInterstageIoData(function[p].type->getQualifier().builtIn, storage));
        }
    };

    // If we synthesize a built-in interface variable, we must add it to the linkage.
    const auto addToLinkage = [&](const TType& type, const TString* name, TIntermSymbol** symbolNode) {
        if (name == nullptr) {
            error(loc, "unable to locate patch function parameter name", "", "");
            return;
        } else {
            TVariable& variable = *new TVariable(name, type);
            if (! symbolTable.insert(variable)) {
                error(loc, "unable to declare patch constant function interface variable", name->c_str(), "");
                return;
            }

            globalQualifierFix(loc, variable.getWritableType().getQualifier());

            if (symbolNode != nullptr)
                *symbolNode = intermediate.addSymbol(variable);

            trackLinkage(variable);
        }
    };

    const auto isOutputPatch = [](TFunction& patchConstantFunction, int param) {
        const TType& type = *patchConstantFunction[param].type;
        const TBuiltInVariable biType = patchConstantFunction[param].getDeclaredBuiltIn();

        return type.isSizedArray() && biType == EbvOutputPatch;
    };
    
    // We will perform these steps.  Each is in a scoped block for separation: they could
    // become separate functions to make addPatchConstantInvocation shorter.
    // 
    // 1. Union the interfaces, and create built-ins for anything present in the PCF and
    //    declared as a built-in variable that isn't present in the entry point's signature.
    //
    // 2. Synthesizes a call to the patchconstfunction using built-in variables from either main,
    //    or the ones we created.  Matching is based on built-in type.  We may use synthesized
    //    variables from (1) above.
    // 
    // 2B: Synthesize per control point invocations of wrapped entry point if the PCF requires them.
    //
    // 3. Create a return sequence: copy the return value (if any) from the PCF to a
    //    (non-sanitized) output variable.  In case this may involve multiple copies, such as for
    //    an arrayed variable, a temporary copy of the PCF output is created to avoid multiple
    //    indirections into a complex R-value coming from the call to the PCF.
    // 
    // 4. Create a barrier.
    // 
    // 5/5B. Call the PCF inside an if test for (invocation id == 0).

    TFunction* patchConstantFunctionPtr = const_cast<TFunction*>(findPatchConstantFunction(loc));

    if (patchConstantFunctionPtr == nullptr)
        return;

    TFunction& patchConstantFunction = *patchConstantFunctionPtr;

    const int pcfParamCount = patchConstantFunction.getParamCount();
    TIntermSymbol* invocationIdSym = findTessLinkageSymbol(EbvInvocationId);
    TIntermSequence& epBodySeq = entryPointFunctionBody->getAsAggregate()->getSequence();

    int outPatchParam = -1; // -1 means there isn't one.

    // ================ Step 1A: Union Interfaces ================
    // Our patch constant function.
    {
        std::set<tInterstageIoData> pcfBuiltIns;  // patch constant function built-ins
        std::set<tInterstageIoData> epfBuiltIns;  // entry point function built-ins

        assert(entryPointFunction);
        assert(entryPointFunctionBody);

        findBuiltIns(patchConstantFunction, pcfBuiltIns);
        findBuiltIns(*entryPointFunction,   epfBuiltIns);

        // Find the set of built-ins in the PCF that are not present in the entry point.
        std::set<tInterstageIoData> notInEntryPoint;

        notInEntryPoint = pcfBuiltIns;

        // std::set_difference not usable on unordered containers
        for (auto bi = epfBuiltIns.begin(); bi != epfBuiltIns.end(); ++bi)
            notInEntryPoint.erase(*bi);

        // Now we'll add those to the entry and to the linkage.
        for (int p=0; p<pcfParamCount; ++p) {
            const TBuiltInVariable biType   = patchConstantFunction[p].getDeclaredBuiltIn();
            TStorageQualifier storage = patchConstantFunction[p].type->getQualifier().storage;

            // Track whether there is an output patch param
            if (isOutputPatch(patchConstantFunction, p)) {
                if (outPatchParam >= 0) {
                    // Presently we only support one per ctrl pt input.
                    error(loc, "unimplemented: multiple output patches in patch constant function", "", "");
                    return;
                }
                outPatchParam = p;
            }

            if (biType != EbvNone) {
                TType* paramType = patchConstantFunction[p].type->clone();

                if (storage == EvqConstReadOnly) // treated identically to input
                    storage = EvqIn;

                // Presently, the only non-built-in we support is InputPatch, which is treated as
                // a pseudo-built-in.
                if (biType == EbvInputPatch) {
                    builtInTessLinkageSymbols[biType] = inputPatch;
                } else if (biType == EbvOutputPatch) {
                    // Nothing...
                } else {
                    // Use the original declaration type for the linkage
                    paramType->getQualifier().builtIn = biType;

                    if (notInEntryPoint.count(tInterstageIoData(biType, storage)) == 1)
                        addToLinkage(*paramType, patchConstantFunction[p].name, nullptr);
                }
            }
        }

        // If we didn't find it because the shader made one, add our own.
        if (invocationIdSym == nullptr) {
            TType invocationIdType(EbtUint, EvqIn, 1);
            TString* invocationIdName = NewPoolTString("InvocationId");
            invocationIdType.getQualifier().builtIn = EbvInvocationId;
            addToLinkage(invocationIdType, invocationIdName, &invocationIdSym);
        }

        assert(invocationIdSym);
    }

    TIntermTyped* pcfArguments = nullptr;
    TVariable* perCtrlPtVar = nullptr;

    // ================ Step 1B: Argument synthesis ================
    // Create pcfArguments for synthesis of patchconstantfunction invocation
    {
        for (int p=0; p<pcfParamCount; ++p) {
            TIntermTyped* inputArg = nullptr;

            if (p == outPatchParam) {
                if (perCtrlPtVar == nullptr) {
                    perCtrlPtVar = makeInternalVariable(*patchConstantFunction[outPatchParam].name,
                                                        *patchConstantFunction[outPatchParam].type);

                    perCtrlPtVar->getWritableType().getQualifier().makeTemporary();
                }
                inputArg = intermediate.addSymbol(*perCtrlPtVar, loc);
            } else {
                // find which built-in it is
                const TBuiltInVariable biType = patchConstantFunction[p].getDeclaredBuiltIn();
                
                if (biType == EbvInputPatch && inputPatch == nullptr) {
                    error(loc, "unimplemented: PCF input patch without entry point input patch parameter", "", "");
                    return;
                }

                inputArg = findTessLinkageSymbol(biType);

                if (inputArg == nullptr) {
                    error(loc, "unable to find patch constant function built-in variable", "", "");
                    return;
                }
            }

            if (pcfParamCount == 1)
                pcfArguments = inputArg;
            else
                pcfArguments = intermediate.growAggregate(pcfArguments, inputArg);
        }
    }

    // ================ Step 2: Synthesize call to PCF ================
    TIntermAggregate* pcfCallSequence = nullptr;
    TIntermTyped* pcfCall = nullptr;

    {
        // Create a function call to the patchconstantfunction
        if (pcfArguments)
            addInputArgumentConversions(patchConstantFunction, pcfArguments);

        // Synthetic call.
        pcfCall = intermediate.setAggregateOperator(pcfArguments, EOpFunctionCall, patchConstantFunction.getType(), loc);
        pcfCall->getAsAggregate()->setUserDefined();
        pcfCall->getAsAggregate()->setName(patchConstantFunction.getMangledName());
        intermediate.addToCallGraph(infoSink, intermediate.getEntryPointMangledName().c_str(),
                                    patchConstantFunction.getMangledName());

        if (pcfCall->getAsAggregate()) {
            TQualifierList& qualifierList = pcfCall->getAsAggregate()->getQualifierList();
            for (int i = 0; i < patchConstantFunction.getParamCount(); ++i) {
                TStorageQualifier qual = patchConstantFunction[i].type->getQualifier().storage;
                qualifierList.push_back(qual);
            }
            pcfCall = addOutputArgumentConversions(patchConstantFunction, *pcfCall->getAsOperator());
        }
    }

    // ================ Step 2B: Per Control Point synthesis ================
    // If there is per control point data, we must either emulate that with multiple
    // invocations of the entry point to build up an array, or (TODO:) use a yet
    // unavailable extension to look across the SIMD lanes.  This is the former
    // as a placeholder for the latter.
    if (outPatchParam >= 0) {
        // We must introduce a local temp variable of the type wanted by the PCF input.
        const int arraySize = patchConstantFunction[outPatchParam].type->getOuterArraySize();

        if (entryPointFunction->getType().getBasicType() == EbtVoid) {
            error(loc, "entry point must return a value for use with patch constant function", "", "");
            return;
        }

        // Create calls to wrapped main to fill in the array.  We will substitute fixed values
        // of invocation ID when calling the wrapped main.

        // This is the type of the each member of the per ctrl point array.
        const TType derefType(perCtrlPtVar->getType(), 0);

        for (int cpt = 0; cpt < arraySize; ++cpt) {
            // TODO: improve.  substr(1) here is to avoid the '@' that was grafted on but isn't in the symtab
            // for this function.
            const TString origName = entryPointFunction->getName().substr(1);
            TFunction callee(&origName, TType(EbtVoid));
            TIntermTyped* callingArgs = nullptr;

            for (int i = 0; i < entryPointFunction->getParamCount(); i++) {
                TParameter& param = (*entryPointFunction)[i];
                TType& paramType = *param.type;

                if (paramType.getQualifier().isParamOutput()) {
                    error(loc, "unimplemented: entry point outputs in patch constant function invocation", "", "");
                    return;
                }

                if (paramType.getQualifier().isParamInput())  {
                    TIntermTyped* arg = nullptr;
                    if ((*entryPointFunction)[i].getDeclaredBuiltIn() == EbvInvocationId) {
                        // substitute invocation ID with the array element ID
                        arg = intermediate.addConstantUnion(cpt, loc);
                    } else {
                        TVariable* argVar = makeInternalVariable(*param.name, *param.type);
                        argVar->getWritableType().getQualifier().makeTemporary();
                        arg = intermediate.addSymbol(*argVar);
                    }

                    handleFunctionArgument(&callee, callingArgs, arg);
                }
            }

            // Call and assign to per ctrl point variable
            currentCaller = intermediate.getEntryPointMangledName().c_str();
            TIntermTyped* callReturn = handleFunctionCall(loc, &callee, callingArgs);
            TIntermTyped* index = intermediate.addConstantUnion(cpt, loc);
            TIntermSymbol* perCtrlPtSym = intermediate.addSymbol(*perCtrlPtVar, loc);
            TIntermTyped* element = intermediate.addIndex(EOpIndexDirect, perCtrlPtSym, index, loc);
            element->setType(derefType);
            element->setLoc(loc);

            pcfCallSequence = intermediate.growAggregate(pcfCallSequence, 
                                                         handleAssign(loc, EOpAssign, element, callReturn));
        }
    }

    // ================ Step 3: Create return Sequence ================
    // Return sequence: copy PCF result to a temporary, then to shader output variable.
    if (pcfCall->getBasicType() != EbtVoid) {
        const TType* retType = &patchConstantFunction.getType();  // return type from the PCF
        TType outType; // output type that goes with the return type.
        outType.shallowCopy(*retType);

        // substitute the output type
        const auto newLists = ioTypeMap.find(retType->getStruct());
        if (newLists != ioTypeMap.end())
            outType.setStruct(newLists->second.output);

        // Substitute the top level type's built-in type
        if (patchConstantFunction.getDeclaredBuiltInType() != EbvNone)
            outType.getQualifier().builtIn = patchConstantFunction.getDeclaredBuiltInType();

        outType.getQualifier().patch = true; // make it a per-patch variable

        TVariable* pcfOutput = makeInternalVariable("@patchConstantOutput", outType);
        pcfOutput->getWritableType().getQualifier().storage = EvqVaryingOut;

        if (pcfOutput->getType().containsBuiltIn())
            split(*pcfOutput);

        assignToInterface(*pcfOutput);

        TIntermSymbol* pcfOutputSym = intermediate.addSymbol(*pcfOutput, loc);

        // The call to the PCF is a complex R-value: we want to store it in a temp to avoid
        // repeated calls to the PCF:
        TVariable* pcfCallResult = makeInternalVariable("@patchConstantResult", *retType);
        pcfCallResult->getWritableType().getQualifier().makeTemporary();

        TIntermSymbol* pcfResultVar = intermediate.addSymbol(*pcfCallResult, loc);
        TIntermNode* pcfResultAssign = handleAssign(loc, EOpAssign, pcfResultVar, pcfCall);
        TIntermNode* pcfResultToOut = handleAssign(loc, EOpAssign, pcfOutputSym,
                                                   intermediate.addSymbol(*pcfCallResult, loc));

        pcfCallSequence = intermediate.growAggregate(pcfCallSequence, pcfResultAssign);
        pcfCallSequence = intermediate.growAggregate(pcfCallSequence, pcfResultToOut);
    } else {
        pcfCallSequence = intermediate.growAggregate(pcfCallSequence, pcfCall);
    }

    // ================ Step 4: Barrier ================    
    TIntermTyped* barrier = new TIntermAggregate(EOpBarrier);
    barrier->setLoc(loc);
    barrier->setType(TType(EbtVoid));
    epBodySeq.insert(epBodySeq.end(), barrier);

    // ================ Step 5: Test on invocation ID ================
    TIntermTyped* zero = intermediate.addConstantUnion(0, loc, true);
    TIntermTyped* cmp =  intermediate.addBinaryNode(EOpEqual, invocationIdSym, zero, loc, TType(EbtBool));


    // ================ Step 5B: Create if statement on Invocation ID == 0 ================
    intermediate.setAggregateOperator(pcfCallSequence, EOpSequence, TType(EbtVoid), loc);
    TIntermTyped* invocationIdTest = new TIntermSelection(cmp, pcfCallSequence, nullptr);
    invocationIdTest->setLoc(loc);

    // add our test sequence before the return.
    epBodySeq.insert(epBodySeq.end(), invocationIdTest);
}

// Finalization step: remove unused buffer blocks from linkage (we don't know until the
// shader is entirely compiled).
// Preserve order of remaining symbols.
void HlslParseContext::removeUnusedStructBufferCounters()
{
    const auto endIt = std::remove_if(linkageSymbols.begin(), linkageSymbols.end(),
                                      [this](const TSymbol* sym) {
                                          const auto sbcIt = structBufferCounter.find(sym->getName());
                                          return sbcIt != structBufferCounter.end() && !sbcIt->second;
                                      });

    linkageSymbols.erase(endIt, linkageSymbols.end());
}

// Finalization step: patch texture shadow modes to match samplers they were combined with
void HlslParseContext::fixTextureShadowModes()
{
    for (auto symbol = linkageSymbols.begin(); symbol != linkageSymbols.end(); ++symbol) {
        TSampler& sampler = (*symbol)->getWritableType().getSampler();

        if (sampler.isTexture()) {
            const auto shadowMode = textureShadowVariant.find((*symbol)->getUniqueId());
            if (shadowMode != textureShadowVariant.end()) {

                if (shadowMode->second->overloaded())
                    // Texture needs legalization if it's been seen with both shadow and non-shadow modes.
                    intermediate.setNeedsLegalization();

                sampler.shadow = shadowMode->second->isShadowId((*symbol)->getUniqueId());
            }
        }
    }
}

// Finalization step: patch append methods to use proper stream output, which isn't known until
// main is parsed, which could happen after the append method is parsed.
void HlslParseContext::finalizeAppendMethods()
{
    TSourceLoc loc;
    loc.init();

    // Nothing to do: bypass test for valid stream output.
    if (gsAppends.empty())
        return;

    if (gsStreamOutput == nullptr) {
        error(loc, "unable to find output symbol for Append()", "", "");
        return;
    }

    // Patch append sequences, now that we know the stream output symbol.
    for (auto append = gsAppends.begin(); append != gsAppends.end(); ++append) {
        append->node->getSequence()[0] = 
            handleAssign(append->loc, EOpAssign,
                         intermediate.addSymbol(*gsStreamOutput, append->loc),
                         append->node->getSequence()[0]->getAsTyped());
    }
}

// post-processing
void HlslParseContext::finish()
{
    // Error check: There was a dangling .mips operator.  These are not nested constructs in the grammar, so
    // cannot be detected there.  This is not strictly needed in a non-validating parser; it's just helpful.
    if (! mipsOperatorMipArg.empty()) {
        error(mipsOperatorMipArg.back().loc, "unterminated mips operator:", "", "");
    }

    removeUnusedStructBufferCounters();
    addPatchConstantInvocation();
    fixTextureShadowModes();
    finalizeAppendMethods();

    // Communicate out (esp. for command line) that we formed AST that will make
    // illegal AST SPIR-V and it needs transforms to legalize it.
    if (intermediate.needsLegalization() && (messages & EShMsgHlslLegalization))
        infoSink.info << "WARNING: AST will form illegal SPIR-V; need to transform to legalize";

    TParseContextBase::finish();
}

} // end namespace glslang
