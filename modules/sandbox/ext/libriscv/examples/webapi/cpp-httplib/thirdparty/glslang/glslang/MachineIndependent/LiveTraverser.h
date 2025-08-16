//
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

#pragma once

#include "../Include/Common.h"
#include "reflection.h"
#include "localintermediate.h"

#include "gl_types.h"

#include <list>
#include <unordered_set>

namespace glslang {

//
// The traverser: mostly pass through, except
//  - processing function-call nodes to push live functions onto the stack of functions to process
//  - processing selection nodes to trim semantically dead code
//
// This is in the glslang namespace directly so it can be a friend of TReflection.
// This can be derived from to implement reflection database traversers or
// binding mappers: anything that wants to traverse the live subset of the tree.
//

class TLiveTraverser : public TIntermTraverser {
public:
    TLiveTraverser(const TIntermediate& i, bool traverseAll = false,
                   bool preVisit = true, bool inVisit = false, bool postVisit = false) :
        TIntermTraverser(preVisit, inVisit, postVisit),
        intermediate(i), traverseAll(traverseAll)
    { }

    //
    // Given a function name, find its subroot in the tree, and push it onto the stack of
    // functions left to process.
    //
    void pushFunction(const TString& name)
    {
        TIntermSequence& globals = intermediate.getTreeRoot()->getAsAggregate()->getSequence();
        for (unsigned int f = 0; f < globals.size(); ++f) {
            TIntermAggregate* candidate = globals[f]->getAsAggregate();
            if (candidate && candidate->getOp() == EOpFunction && candidate->getName() == name) {
                destinations.push_back(candidate);
                break;
            }
        }
    }

    void pushGlobalReference(const TString& name)
    {
        TIntermSequence& globals = intermediate.getTreeRoot()->getAsAggregate()->getSequence();
        for (unsigned int f = 0; f < globals.size(); ++f) {
            TIntermAggregate* candidate = globals[f]->getAsAggregate();
            if (candidate && candidate->getOp() == EOpSequence &&
                candidate->getSequence().size() == 1 &&
                candidate->getSequence()[0]->getAsBinaryNode()) {
                TIntermBinary* binary = candidate->getSequence()[0]->getAsBinaryNode();
                TIntermSymbol* symbol = binary->getLeft()->getAsSymbolNode();
                if (symbol && symbol->getQualifier().storage == EvqGlobal &&
                    symbol->getName() == name) {
                    destinations.push_back(candidate);
                    break;
                }
            }
        }
    }

    typedef std::list<TIntermAggregate*> TDestinationStack;
    TDestinationStack destinations;

protected:
    // To catch which function calls are not dead, and hence which functions must be visited.
    virtual bool visitAggregate(TVisit, TIntermAggregate* node)
    {
        if (!traverseAll)
            if (node->getOp() == EOpFunctionCall)
                addFunctionCall(node);

        return true; // traverse this subtree
    }

    // To prune semantically dead paths.
    virtual bool visitSelection(TVisit /* visit */,  TIntermSelection* node)
    {
        if (traverseAll)
            return true; // traverse all code

        TIntermConstantUnion* constant = node->getCondition()->getAsConstantUnion();
        if (constant) {
            // cull the path that is dead
            if (constant->getConstArray()[0].getBConst() == true && node->getTrueBlock())
                node->getTrueBlock()->traverse(this);
            if (constant->getConstArray()[0].getBConst() == false && node->getFalseBlock())
                node->getFalseBlock()->traverse(this);

            return false; // don't traverse any more, we did it all above
        } else
            return true; // traverse the whole subtree
    }

    // Track live functions as well as uniforms, so that we don't visit dead functions
    // and only visit each function once.
    void addFunctionCall(TIntermAggregate* call)
    {
        // just use the map to ensure we process each function at most once
        if (liveFunctions.find(call->getName()) == liveFunctions.end()) {
            liveFunctions.insert(call->getName());
            pushFunction(call->getName());
        }
    }

    void addGlobalReference(const TString& name)
    {
        // just use the map to ensure we process each global at most once
        if (liveGlobals.find(name) == liveGlobals.end()) {
            liveGlobals.insert(name);
            pushGlobalReference(name);
        }
    }

    const TIntermediate& intermediate;
    typedef std::unordered_set<TString> TLiveFunctions;
    TLiveFunctions liveFunctions;
    typedef std::unordered_set<TString> TLiveGlobals;
    TLiveGlobals liveGlobals;
    bool traverseAll;

private:
    // prevent copy & copy construct
    TLiveTraverser(TLiveTraverser&);
    TLiveTraverser& operator=(TLiveTraverser&);
};

} // namespace glslang
