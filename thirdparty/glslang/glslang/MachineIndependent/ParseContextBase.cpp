//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2016 Google, Inc.
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

// Implement the TParseContextBase class.

#include <cstdarg>

#include "ParseHelper.h"

extern int yyparse(glslang::TParseContext*);

namespace glslang {

//
// Used to output syntax, parsing, and semantic errors.
//

void TParseContextBase::outputMessage(const TSourceLoc& loc, const char* szReason,
                                      const char* szToken,
                                      const char* szExtraInfoFormat,
                                      TPrefixType prefix, va_list args)
{
    const int maxSize = MaxTokenLength + 200;
    char szExtraInfo[maxSize];

    safe_vsprintf(szExtraInfo, maxSize, szExtraInfoFormat, args);

    infoSink.info.prefix(prefix);
    infoSink.info.location(loc);
    infoSink.info << "'" << szToken <<  "' : " << szReason << " " << szExtraInfo << "\n";

    if (prefix == EPrefixError) {
        ++numErrors;
    }
}

#if !defined(GLSLANG_WEB) || defined(GLSLANG_WEB_DEVEL)

void C_DECL TParseContextBase::error(const TSourceLoc& loc, const char* szReason, const char* szToken,
                                     const char* szExtraInfoFormat, ...)
{
    if (messages & EShMsgOnlyPreprocessor)
        return;
    va_list args;
    va_start(args, szExtraInfoFormat);
    outputMessage(loc, szReason, szToken, szExtraInfoFormat, EPrefixError, args);
    va_end(args);

    if ((messages & EShMsgCascadingErrors) == 0)
        currentScanner->setEndOfInput();
}

void C_DECL TParseContextBase::warn(const TSourceLoc& loc, const char* szReason, const char* szToken,
                                    const char* szExtraInfoFormat, ...)
{
    if (suppressWarnings())
        return;
    va_list args;
    va_start(args, szExtraInfoFormat);
    outputMessage(loc, szReason, szToken, szExtraInfoFormat, EPrefixWarning, args);
    va_end(args);
}

void C_DECL TParseContextBase::ppError(const TSourceLoc& loc, const char* szReason, const char* szToken,
                                       const char* szExtraInfoFormat, ...)
{
    va_list args;
    va_start(args, szExtraInfoFormat);
    outputMessage(loc, szReason, szToken, szExtraInfoFormat, EPrefixError, args);
    va_end(args);

    if ((messages & EShMsgCascadingErrors) == 0)
        currentScanner->setEndOfInput();
}

void C_DECL TParseContextBase::ppWarn(const TSourceLoc& loc, const char* szReason, const char* szToken,
                                      const char* szExtraInfoFormat, ...)
{
    va_list args;
    va_start(args, szExtraInfoFormat);
    outputMessage(loc, szReason, szToken, szExtraInfoFormat, EPrefixWarning, args);
    va_end(args);
}

#endif

//
// Both test and if necessary, spit out an error, to see if the node is really
// an l-value that can be operated on this way.
//
// Returns true if there was an error.
//
bool TParseContextBase::lValueErrorCheck(const TSourceLoc& loc, const char* op, TIntermTyped* node)
{
    TIntermBinary* binaryNode = node->getAsBinaryNode();

    const char* symbol = nullptr;
    TIntermSymbol* symNode = node->getAsSymbolNode();
    if (symNode != nullptr)
        symbol = symNode->getName().c_str();

    const char* message = nullptr;
    switch (node->getQualifier().storage) {
    case EvqConst:          message = "can't modify a const";        break;
    case EvqConstReadOnly:  message = "can't modify a const";        break;
    case EvqUniform:        message = "can't modify a uniform";      break;
#ifndef GLSLANG_WEB
    case EvqBuffer:
        if (node->getQualifier().isReadOnly())
            message = "can't modify a readonly buffer";
        if (node->getQualifier().isShaderRecord())
            message = "can't modify a shaderrecordnv qualified buffer";
        break;
    case EvqHitAttr:
        if (language != EShLangIntersect)
            message = "cannot modify hitAttributeNV in this stage";
        break;
#endif

    default:
        //
        // Type that can't be written to?
        //
        switch (node->getBasicType()) {
        case EbtSampler:
            message = "can't modify a sampler";
            break;
        case EbtVoid:
            message = "can't modify void";
            break;
#ifndef GLSLANG_WEB
        case EbtAtomicUint:
            message = "can't modify an atomic_uint";
            break;
        case EbtAccStruct:
            message = "can't modify accelerationStructureNV";
            break;
        case EbtRayQuery:
            message = "can't modify rayQueryEXT";
            break;
#endif
        default:
            break;
        }
    }

    if (message == nullptr && binaryNode == nullptr && symNode == nullptr) {
        error(loc, " l-value required", op, "", "");

        return true;
    }

    //
    // Everything else is okay, no error.
    //
    if (message == nullptr)
    {
        if (binaryNode) {
            switch (binaryNode->getOp()) {
            case EOpIndexDirect:
            case EOpIndexIndirect:     // fall through
            case EOpIndexDirectStruct: // fall through
            case EOpVectorSwizzle:
            case EOpMatrixSwizzle:
                return lValueErrorCheck(loc, op, binaryNode->getLeft());
            default:
                break;
            }
            error(loc, " l-value required", op, "", "");

            return true;
        }
        return false;
    }

    //
    // If we get here, we have an error and a message.
    //
    const TIntermTyped* leftMostTypeNode = TIntermediate::findLValueBase(node, true);

    if (symNode)
        error(loc, " l-value required", op, "\"%s\" (%s)", symbol, message);
    else
        if (binaryNode && binaryNode->getAsOperator()->getOp() == EOpIndexDirectStruct)
            if(IsAnonymous(leftMostTypeNode->getAsSymbolNode()->getName()))
                error(loc, " l-value required", op, "\"%s\" (%s)", leftMostTypeNode->getAsSymbolNode()->getAccessName().c_str(), message);
            else
                error(loc, " l-value required", op, "\"%s\" (%s)", leftMostTypeNode->getAsSymbolNode()->getName().c_str(), message);
        else
            error(loc, " l-value required", op, "(%s)", message);

    return true;
}

// Test for and give an error if the node can't be read from.
void TParseContextBase::rValueErrorCheck(const TSourceLoc& loc, const char* op, TIntermTyped* node)
{
    TIntermBinary* binaryNode = node->getAsBinaryNode();
    const TIntermSymbol* symNode = node->getAsSymbolNode();

    if (! node)
        return;

    if (node->getQualifier().isWriteOnly()) {
        const TIntermTyped* leftMostTypeNode = TIntermediate::findLValueBase(node, true);

        if (symNode != nullptr)
            error(loc, "can't read from writeonly object: ", op, symNode->getName().c_str());
        else if (binaryNode &&
                (binaryNode->getAsOperator()->getOp() == EOpIndexDirectStruct ||
                 binaryNode->getAsOperator()->getOp() == EOpIndexDirect))
            if(IsAnonymous(leftMostTypeNode->getAsSymbolNode()->getName()))
                error(loc, "can't read from writeonly object: ", op, leftMostTypeNode->getAsSymbolNode()->getAccessName().c_str());
            else
                error(loc, "can't read from writeonly object: ", op, leftMostTypeNode->getAsSymbolNode()->getName().c_str());
        else
            error(loc, "can't read from writeonly object: ", op, "");

    } else {
        if (binaryNode) {
            switch (binaryNode->getOp()) {
            case EOpIndexDirect:
            case EOpIndexIndirect:
            case EOpIndexDirectStruct:
            case EOpVectorSwizzle:
            case EOpMatrixSwizzle:
                rValueErrorCheck(loc, op, binaryNode->getLeft());
            default:
                break;
            }
        }
    }
}

// Add 'symbol' to the list of deferred linkage symbols, which
// are later processed in finish(), at which point the symbol
// must still be valid.
// It is okay if the symbol's type will be subsequently edited;
// the modifications will be tracked.
// Order is preserved, to avoid creating novel forward references.
void TParseContextBase::trackLinkage(TSymbol& symbol)
{
    if (!parsingBuiltins)
        linkageSymbols.push_back(&symbol);
}

// Ensure index is in bounds, correct if necessary.
// Give an error if not.
void TParseContextBase::checkIndex(const TSourceLoc& loc, const TType& type, int& index)
{
    const auto sizeIsSpecializationExpression = [&type]() {
        return type.containsSpecializationSize() &&
               type.getArraySizes()->getOuterNode() != nullptr &&
               type.getArraySizes()->getOuterNode()->getAsSymbolNode() == nullptr; };

    if (index < 0) {
        error(loc, "", "[", "index out of range '%d'", index);
        index = 0;
    } else if (type.isArray()) {
        if (type.isSizedArray() && !sizeIsSpecializationExpression() &&
            index >= type.getOuterArraySize()) {
            error(loc, "", "[", "array index out of range '%d'", index);
            index = type.getOuterArraySize() - 1;
        }
    } else if (type.isVector()) {
        if (index >= type.getVectorSize()) {
            error(loc, "", "[", "vector index out of range '%d'", index);
            index = type.getVectorSize() - 1;
        }
    } else if (type.isMatrix()) {
        if (index >= type.getMatrixCols()) {
            error(loc, "", "[", "matrix index out of range '%d'", index);
            index = type.getMatrixCols() - 1;
        }
    }
}

// Make a shared symbol have a non-shared version that can be edited by the current
// compile, such that editing its type will not change the shared version and will
// effect all nodes already sharing it (non-shallow type),
// or adopting its full type after being edited (shallow type).
void TParseContextBase::makeEditable(TSymbol*& symbol)
{
    // copyUp() does a deep copy of the type.
    symbol = symbolTable.copyUp(symbol);

    // Save it (deferred, so it can be edited first) in the AST for linker use.
    if (symbol)
        trackLinkage(*symbol);
}

// Return a writable version of the variable 'name'.
//
// Return nullptr if 'name' is not found.  This should mean
// something is seriously wrong (e.g., compiler asking self for
// built-in that doesn't exist).
TVariable* TParseContextBase::getEditableVariable(const char* name)
{
    bool builtIn;
    TSymbol* symbol = symbolTable.find(name, &builtIn);

    assert(symbol != nullptr);
    if (symbol == nullptr)
        return nullptr;

    if (builtIn)
        makeEditable(symbol);

    return symbol->getAsVariable();
}

// Select the best matching function for 'call' from 'candidateList'.
//
// Assumptions
//
// There is no exact match, so a selection algorithm needs to run. That is, the
// language-specific handler should check for exact match first, to
// decide what to do, before calling this selector.
//
// Input
//
//  * list of candidate signatures to select from
//  * the call
//  * a predicate function convertible(from, to) that says whether or not type
//    'from' can implicitly convert to type 'to' (it includes the case of what
//    the calling language would consider a matching type with no conversion
//    needed)
//  * a predicate function better(from1, from2, to1, to2) that says whether or
//    not a conversion from <-> to2 is considered better than a conversion
//    from <-> to1 (both in and out directions need testing, as declared by the
//    formal parameter)
//
// Output
//
//  * best matching candidate (or none, if no viable candidates found)
//  * whether there was a tie for the best match (ambiguous overload selection,
//    caller's choice for how to report)
//
const TFunction* TParseContextBase::selectFunction(
    const TVector<const TFunction*> candidateList,
    const TFunction& call,
    std::function<bool(const TType& from, const TType& to, TOperator op, int arg)> convertible,
    std::function<bool(const TType& from, const TType& to1, const TType& to2)> better,
    /* output */ bool& tie)
{
//
// Operation
//
// 1. Prune the input list of candidates down to a list of viable candidates,
// where each viable candidate has
//
//  * at least as many parameters as there are calling arguments, with any
//    remaining parameters being optional or having default values
//  * each parameter is true under convertible(A, B), where A is the calling
//    type for in and B is the formal type, and in addition, for out B is the
//    calling type and A is the formal type
//
// 2. If there are no viable candidates, return with no match.
//
// 3. If there is only one viable candidate, it is the best match.
//
// 4. If there are multiple viable candidates, select the first viable candidate
// as the incumbent. Compare the incumbent to the next viable candidate, and if
// that candidate is better (bullets below), make it the incumbent. Repeat, with
// a linear walk through the viable candidate list. The final incumbent will be
// returned as the best match. A viable candidate is better than the incumbent if
//
//  * it has a function argument with a better(...) conversion than the incumbent,
//    for all directions needed by in and out
//  * the incumbent has no argument with a better(...) conversion then the
//    candidate, for either in or out (as needed)
//
// 5. Check for ambiguity by comparing the best match against all other viable
// candidates. If any other viable candidate has a function argument with a
// better(...) conversion than the best candidate (for either in or out
// directions), return that there was a tie for best.
//

    tie = false;

    // 1. prune to viable...
    TVector<const TFunction*> viableCandidates;
    for (auto it = candidateList.begin(); it != candidateList.end(); ++it) {
        const TFunction& candidate = *(*it);

        // to even be a potential match, number of arguments must be >= the number of
        // fixed (non-default) parameters, and <= the total (including parameter with defaults).
        if (call.getParamCount() < candidate.getFixedParamCount() ||
            call.getParamCount() > candidate.getParamCount())
            continue;

        // see if arguments are convertible
        bool viable = true;

        // The call can have fewer parameters than the candidate, if some have defaults.
        const int paramCount = std::min(call.getParamCount(), candidate.getParamCount());
        for (int param = 0; param < paramCount; ++param) {
            if (candidate[param].type->getQualifier().isParamInput()) {
                if (! convertible(*call[param].type, *candidate[param].type, candidate.getBuiltInOp(), param)) {
                    viable = false;
                    break;
                }
            }
            if (candidate[param].type->getQualifier().isParamOutput()) {
                if (! convertible(*candidate[param].type, *call[param].type, candidate.getBuiltInOp(), param)) {
                    viable = false;
                    break;
                }
            }
        }

        if (viable)
            viableCandidates.push_back(&candidate);
    }

    // 2. none viable...
    if (viableCandidates.size() == 0)
        return nullptr;

    // 3. only one viable...
    if (viableCandidates.size() == 1)
        return viableCandidates.front();

    // 4. find best...
    const auto betterParam = [&call, &better](const TFunction& can1, const TFunction& can2) -> bool {
        // is call -> can2 better than call -> can1 for any parameter
        bool hasBetterParam = false;
        for (int param = 0; param < call.getParamCount(); ++param) {
            if (better(*call[param].type, *can1[param].type, *can2[param].type)) {
                hasBetterParam = true;
                break;
            }
        }
        return hasBetterParam;
    };

    const auto equivalentParams = [&call, &better](const TFunction& can1, const TFunction& can2) -> bool {
        // is call -> can2 equivalent to call -> can1 for all the call parameters?
        for (int param = 0; param < call.getParamCount(); ++param) {
            if (better(*call[param].type, *can1[param].type, *can2[param].type) ||
                better(*call[param].type, *can2[param].type, *can1[param].type))
                return false;
        }
        return true;
    };

    const TFunction* incumbent = viableCandidates.front();
    for (auto it = viableCandidates.begin() + 1; it != viableCandidates.end(); ++it) {
        const TFunction& candidate = *(*it);
        if (betterParam(*incumbent, candidate) && ! betterParam(candidate, *incumbent))
            incumbent = &candidate;
    }

    // 5. ambiguity...
    for (auto it = viableCandidates.begin(); it != viableCandidates.end(); ++it) {
        if (incumbent == *it)
            continue;
        const TFunction& candidate = *(*it);

        // In the case of default parameters, it may have an identical initial set, which is
        // also ambiguous
        if (betterParam(*incumbent, candidate) || equivalentParams(*incumbent, candidate))
            tie = true;
    }

    return incumbent;
}

//
// Look at a '.' field selector string and change it into numerical selectors
// for a vector or scalar.
//
// Always return some form of swizzle, so the result is always usable.
//
void TParseContextBase::parseSwizzleSelector(const TSourceLoc& loc, const TString& compString, int vecSize,
                                             TSwizzleSelectors<TVectorSelector>& selector)
{
    // Too long?
    if (compString.size() > MaxSwizzleSelectors)
        error(loc, "vector swizzle too long", compString.c_str(), "");

    // Use this to test that all swizzle characters are from the same swizzle-namespace-set
    enum {
        exyzw,
        ergba,
        estpq,
    } fieldSet[MaxSwizzleSelectors];

    // Decode the swizzle string.
    int size = std::min(MaxSwizzleSelectors, (int)compString.size());
    for (int i = 0; i < size; ++i) {
        switch (compString[i])  {
        case 'x':
            selector.push_back(0);
            fieldSet[i] = exyzw;
            break;
        case 'r':
            selector.push_back(0);
            fieldSet[i] = ergba;
            break;
        case 's':
            selector.push_back(0);
            fieldSet[i] = estpq;
            break;

        case 'y':
            selector.push_back(1);
            fieldSet[i] = exyzw;
            break;
        case 'g':
            selector.push_back(1);
            fieldSet[i] = ergba;
            break;
        case 't':
            selector.push_back(1);
            fieldSet[i] = estpq;
            break;

        case 'z':
            selector.push_back(2);
            fieldSet[i] = exyzw;
            break;
        case 'b':
            selector.push_back(2);
            fieldSet[i] = ergba;
            break;
        case 'p':
            selector.push_back(2);
            fieldSet[i] = estpq;
            break;

        case 'w':
            selector.push_back(3);
            fieldSet[i] = exyzw;
            break;
        case 'a':
            selector.push_back(3);
            fieldSet[i] = ergba;
            break;
        case 'q':
            selector.push_back(3);
            fieldSet[i] = estpq;
            break;

        default:
            error(loc, "unknown swizzle selection", compString.c_str(), "");
            break;
        }
    }

    // Additional error checking.
    for (int i = 0; i < selector.size(); ++i) {
        if (selector[i] >= vecSize) {
            error(loc, "vector swizzle selection out of range",  compString.c_str(), "");
            selector.resize(i);
            break;
        }

        if (i > 0 && fieldSet[i] != fieldSet[i-1]) {
            error(loc, "vector swizzle selectors not from the same set", compString.c_str(), "");
            selector.resize(i);
            break;
        }
    }

    // Ensure it is valid.
    if (selector.size() == 0)
        selector.push_back(0);
}

//
// Make the passed-in variable information become a member of the
// global uniform block.  If this doesn't exist yet, make it.
//
void TParseContextBase::growGlobalUniformBlock(const TSourceLoc& loc, TType& memberType, const TString& memberName, TTypeList* typeList)
{
    // Make the global block, if not yet made.
    if (globalUniformBlock == nullptr) {
        TQualifier blockQualifier;
        blockQualifier.clear();
        blockQualifier.storage = EvqUniform;
        TType blockType(new TTypeList, *NewPoolTString(getGlobalUniformBlockName()), blockQualifier);
        setUniformBlockDefaults(blockType);
        globalUniformBlock = new TVariable(NewPoolTString(""), blockType, true);
        firstNewMember = 0;
    }

    // Update with binding and set
    globalUniformBlock->getWritableType().getQualifier().layoutBinding = globalUniformBinding;
    globalUniformBlock->getWritableType().getQualifier().layoutSet = globalUniformSet;

    // Add the requested member as a member to the global block.
    TType* type = new TType;
    type->shallowCopy(memberType);
    type->setFieldName(memberName);
    if (typeList)
        type->setStruct(typeList);
    TTypeLoc typeLoc = {type, loc};
    globalUniformBlock->getType().getWritableStruct()->push_back(typeLoc);

    // Insert into the symbol table.
    if (firstNewMember == 0) {
        // This is the first request; we need a normal symbol table insert
        if (symbolTable.insert(*globalUniformBlock))
            trackLinkage(*globalUniformBlock);
        else
            error(loc, "failed to insert the global constant buffer", "uniform", "");
    } else {
        // This is a follow-on request; we need to amend the first insert
        symbolTable.amend(*globalUniformBlock, firstNewMember);
    }

    ++firstNewMember;
}

void TParseContextBase::growAtomicCounterBlock(int binding, const TSourceLoc& loc, TType& memberType, const TString& memberName, TTypeList* typeList) {
    // Make the atomic counter block, if not yet made.
    const auto &at  = atomicCounterBuffers.find(binding);
    if (at == atomicCounterBuffers.end()) {
        atomicCounterBuffers.insert({binding, (TVariable*)nullptr });
        atomicCounterBlockFirstNewMember.insert({binding, 0});
    }

    TVariable*& atomicCounterBuffer = atomicCounterBuffers[binding];
    int& bufferNewMember = atomicCounterBlockFirstNewMember[binding];

    if (atomicCounterBuffer == nullptr) {
        TQualifier blockQualifier;
        blockQualifier.clear();
        blockQualifier.storage = EvqBuffer;
        
        char charBuffer[512];
        if (binding != TQualifier::layoutBindingEnd) {
            snprintf(charBuffer, 512, "%s_%d", getAtomicCounterBlockName(), binding);
        } else {
            snprintf(charBuffer, 512, "%s_0", getAtomicCounterBlockName());
        }
        
        TType blockType(new TTypeList, *NewPoolTString(charBuffer), blockQualifier);
        setUniformBlockDefaults(blockType);
        blockType.getQualifier().layoutPacking = ElpStd430;
        atomicCounterBuffer = new TVariable(NewPoolTString(""), blockType, true);
        // If we arn't auto mapping bindings then set the block to use the same
        // binding as what the atomic was set to use
        if (!intermediate.getAutoMapBindings()) {
            atomicCounterBuffer->getWritableType().getQualifier().layoutBinding = binding;
        }
        bufferNewMember = 0;

        atomicCounterBuffer->getWritableType().getQualifier().layoutSet = atomicCounterBlockSet;
    }

    // Add the requested member as a member to the global block.
    TType* type = new TType;
    type->shallowCopy(memberType);
    type->setFieldName(memberName);
    if (typeList)
        type->setStruct(typeList);
    TTypeLoc typeLoc = {type, loc};
    atomicCounterBuffer->getType().getWritableStruct()->push_back(typeLoc);

    // Insert into the symbol table.
    if (bufferNewMember == 0) {
        // This is the first request; we need a normal symbol table insert
        if (symbolTable.insert(*atomicCounterBuffer))
            trackLinkage(*atomicCounterBuffer);
        else
            error(loc, "failed to insert the global constant buffer", "buffer", "");
    } else {
        // This is a follow-on request; we need to amend the first insert
        symbolTable.amend(*atomicCounterBuffer, bufferNewMember);
    }

    ++bufferNewMember;
}

void TParseContextBase::finish()
{
    if (parsingBuiltins)
        return;

    // Transfer the linkage symbols to AST nodes, preserving order.
    TIntermAggregate* linkage = new TIntermAggregate;
    for (auto i = linkageSymbols.begin(); i != linkageSymbols.end(); ++i)
        intermediate.addSymbolLinkageNode(linkage, **i);
    intermediate.addSymbolLinkageNodes(linkage, getLanguage(), symbolTable);
}

} // end namespace glslang
