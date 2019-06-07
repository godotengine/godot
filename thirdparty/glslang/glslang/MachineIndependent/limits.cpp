//
// Copyright (C) 2013 LunarG, Inc.
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
// Do sub tree walks for
// 1) inductive loop bodies to see if the inductive variable is modified
// 2) array-index expressions to see if they are "constant-index-expression"
//
// These are per Appendix A of ES 2.0:
//
// "Within the body of the loop, the loop index is not statically assigned to nor is it used as the
// argument to a function out or inout parameter."
//
// "The following are constant-index-expressions:
//  - Constant expressions
//  - Loop indices as defined in section 4
//  - Expressions composed of both of the above"
//
// N.B.: assuming the last rule excludes function calls
//

#include "ParseHelper.h"

namespace glslang {

//
// The inductive loop-body traverser.
//
// Just look at things that might modify the loop index.
//

class TInductiveTraverser : public TIntermTraverser {
public:
    TInductiveTraverser(int id, TSymbolTable& st)
    : loopId(id), symbolTable(st), bad(false)  { }

    virtual bool visitBinary(TVisit, TIntermBinary* node);
    virtual bool visitUnary(TVisit, TIntermUnary* node);
    virtual bool visitAggregate(TVisit, TIntermAggregate* node);

    int loopId;           // unique ID of the symbol that's the loop inductive variable
    TSymbolTable& symbolTable;
    bool bad;
    TSourceLoc badLoc;

protected:
    TInductiveTraverser(TInductiveTraverser&);
    TInductiveTraverser& operator=(TInductiveTraverser&);
};

// check binary operations for those modifying the loop index
bool TInductiveTraverser::visitBinary(TVisit /* visit */, TIntermBinary* node)
{
    if (node->modifiesState() && node->getLeft()->getAsSymbolNode() &&
                                 node->getLeft()->getAsSymbolNode()->getId() == loopId) {
        bad = true;
        badLoc = node->getLoc();
    }

    return true;
}

// check unary operations for those modifying the loop index
bool TInductiveTraverser::visitUnary(TVisit /* visit */, TIntermUnary* node)
{
    if (node->modifiesState() && node->getOperand()->getAsSymbolNode() &&
                                 node->getOperand()->getAsSymbolNode()->getId() == loopId) {
        bad = true;
        badLoc = node->getLoc();
    }

    return true;
}

// check function calls for arguments modifying the loop index
bool TInductiveTraverser::visitAggregate(TVisit /* visit */, TIntermAggregate* node)
{
    if (node->getOp() == EOpFunctionCall) {
        // see if an out or inout argument is the loop index
        const TIntermSequence& args = node->getSequence();
        for (int i = 0; i < (int)args.size(); ++i) {
            if (args[i]->getAsSymbolNode() && args[i]->getAsSymbolNode()->getId() == loopId) {
                TSymbol* function = symbolTable.find(node->getName());
                const TType* type = (*function->getAsFunction())[i].type;
                if (type->getQualifier().storage == EvqOut ||
                    type->getQualifier().storage == EvqInOut) {
                    bad = true;
                    badLoc = node->getLoc();
                }
            }
        }
    }

    return true;
}

//
// External function to call for loop check.
//
void TParseContext::inductiveLoopBodyCheck(TIntermNode* body, int loopId, TSymbolTable& symbolTable)
{
    TInductiveTraverser it(loopId, symbolTable);

    if (body == nullptr)
        return;

    body->traverse(&it);

    if (it.bad)
        error(it.badLoc, "inductive loop index modified", "limitations", "");
}

//
// The "constant-index-expression" tranverser.
//
// Just look at things that can form an index.
//

class TIndexTraverser : public TIntermTraverser {
public:
    TIndexTraverser(const TIdSetType& ids) : inductiveLoopIds(ids), bad(false) { }
    virtual void visitSymbol(TIntermSymbol* symbol);
    virtual bool visitAggregate(TVisit, TIntermAggregate* node);
    const TIdSetType& inductiveLoopIds;
    bool bad;
    TSourceLoc badLoc;

protected:
    TIndexTraverser(TIndexTraverser&);
    TIndexTraverser& operator=(TIndexTraverser&);
};

// make sure symbols are inductive-loop indexes
void TIndexTraverser::visitSymbol(TIntermSymbol* symbol)
{
    if (inductiveLoopIds.find(symbol->getId()) == inductiveLoopIds.end()) {
        bad = true;
        badLoc = symbol->getLoc();
    }
}

// check for function calls, assuming they are bad; spec. doesn't really say
bool TIndexTraverser::visitAggregate(TVisit /* visit */, TIntermAggregate* node)
{
    if (node->getOp() == EOpFunctionCall) {
        bad = true;
        badLoc = node->getLoc();
    }

    return true;
}

//
// External function to call for loop check.
//
void TParseContext::constantIndexExpressionCheck(TIntermNode* index)
{
    TIndexTraverser it(inductiveLoopIds);

    index->traverse(&it);

    if (it.bad)
        error(it.badLoc, "Non-constant-index-expression", "limitations", "");
}

} // end namespace glslang
