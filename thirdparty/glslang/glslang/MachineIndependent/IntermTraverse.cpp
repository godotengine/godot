//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013 LunarG, Inc.
// Copyright (c) 2002-2010 The ANGLE Project Authors.
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

#include "../Include/intermediate.h"

namespace glslang {

//
// Traverse the intermediate representation tree, and
// call a node type specific function for each node.
// Done recursively through the member function Traverse().
// Node types can be skipped if their function to call is 0,
// but their subtree will still be traversed.
// Nodes with children can have their whole subtree skipped
// if preVisit is turned on and the type specific function
// returns false.
//
// preVisit, postVisit, and rightToLeft control what order
// nodes are visited in.
//

//
// Traversal functions for terminals are straightforward....
//
void TIntermMethod::traverse(TIntermTraverser*)
{
    // Tree should always resolve all methods as a non-method.
}

void TIntermSymbol::traverse(TIntermTraverser *it)
{
    it->visitSymbol(this);
}

void TIntermConstantUnion::traverse(TIntermTraverser *it)
{
    it->visitConstantUnion(this);
}

const TString& TIntermSymbol::getAccessName() const {
    if (getBasicType() == EbtBlock)
        return getType().getTypeName();
    else
        return getName();
}

//
// Traverse a binary node.
//
void TIntermBinary::traverse(TIntermTraverser *it)
{
    bool visit = true;

    //
    // visit the node before children if pre-visiting.
    //
    if (it->preVisit)
        visit = it->visitBinary(EvPreVisit, this);

    //
    // Visit the children, in the right order.
    //
    if (visit) {
        it->incrementDepth(this);

        if (it->rightToLeft) {
            if (right)
                right->traverse(it);

            if (it->inVisit)
                visit = it->visitBinary(EvInVisit, this);

            if (visit && left)
                left->traverse(it);
        } else {
            if (left)
                left->traverse(it);

            if (it->inVisit)
                visit = it->visitBinary(EvInVisit, this);

            if (visit && right)
                right->traverse(it);
        }

        it->decrementDepth();
    }

    //
    // Visit the node after the children, if requested and the traversal
    // hasn't been canceled yet.
    //
    if (visit && it->postVisit)
        it->visitBinary(EvPostVisit, this);
}

//
// Traverse a unary node.  Same comments in binary node apply here.
//
void TIntermUnary::traverse(TIntermTraverser *it)
{
    bool visit = true;

    if (it->preVisit)
        visit = it->visitUnary(EvPreVisit, this);

    if (visit) {
        it->incrementDepth(this);
        operand->traverse(it);
        it->decrementDepth();
    }

    if (visit && it->postVisit)
        it->visitUnary(EvPostVisit, this);
}

//
// Traverse an aggregate node.  Same comments in binary node apply here.
//
void TIntermAggregate::traverse(TIntermTraverser *it)
{
    bool visit = true;

    if (it->preVisit)
        visit = it->visitAggregate(EvPreVisit, this);

    if (visit) {
        it->incrementDepth(this);

        if (it->rightToLeft) {
            for (TIntermSequence::reverse_iterator sit = sequence.rbegin(); sit != sequence.rend(); sit++) {
                (*sit)->traverse(it);

                if (visit && it->inVisit) {
                    if (*sit != sequence.front())
                        visit = it->visitAggregate(EvInVisit, this);
                }
            }
        } else {
            for (TIntermSequence::iterator sit = sequence.begin(); sit != sequence.end(); sit++) {
                (*sit)->traverse(it);

                if (visit && it->inVisit) {
                    if (*sit != sequence.back())
                        visit = it->visitAggregate(EvInVisit, this);
                }
            }
        }

        it->decrementDepth();
    }

    if (visit && it->postVisit)
        it->visitAggregate(EvPostVisit, this);
}

//
// Traverse a selection node.  Same comments in binary node apply here.
//
void TIntermSelection::traverse(TIntermTraverser *it)
{
    bool visit = true;

    if (it->preVisit)
        visit = it->visitSelection(EvPreVisit, this);

    if (visit) {
        it->incrementDepth(this);
        if (it->rightToLeft) {
            if (falseBlock)
                falseBlock->traverse(it);
            if (trueBlock)
                trueBlock->traverse(it);
            condition->traverse(it);
        } else {
            condition->traverse(it);
            if (trueBlock)
                trueBlock->traverse(it);
            if (falseBlock)
                falseBlock->traverse(it);
        }
        it->decrementDepth();
    }

    if (visit && it->postVisit)
        it->visitSelection(EvPostVisit, this);
}

//
// Traverse a loop node.  Same comments in binary node apply here.
//
void TIntermLoop::traverse(TIntermTraverser *it)
{
    bool visit = true;

    if (it->preVisit)
        visit = it->visitLoop(EvPreVisit, this);

    if (visit) {
        it->incrementDepth(this);

        if (it->rightToLeft) {
            if (terminal)
                terminal->traverse(it);

            if (body)
                body->traverse(it);

            if (test)
                test->traverse(it);
        } else {
            if (test)
                test->traverse(it);

            if (body)
                body->traverse(it);

            if (terminal)
                terminal->traverse(it);
        }

        it->decrementDepth();
    }

    if (visit && it->postVisit)
        it->visitLoop(EvPostVisit, this);
}

//
// Traverse a branch node.  Same comments in binary node apply here.
//
void TIntermBranch::traverse(TIntermTraverser *it)
{
    bool visit = true;

    if (it->preVisit)
        visit = it->visitBranch(EvPreVisit, this);

    if (visit && expression) {
        it->incrementDepth(this);
        expression->traverse(it);
        it->decrementDepth();
    }

    if (visit && it->postVisit)
        it->visitBranch(EvPostVisit, this);
}

//
// Traverse a switch node.
//
void TIntermSwitch::traverse(TIntermTraverser* it)
{
    bool visit = true;

    if (it->preVisit)
        visit = it->visitSwitch(EvPreVisit, this);

    if (visit) {
        it->incrementDepth(this);
        if (it->rightToLeft) {
            body->traverse(it);
            condition->traverse(it);
        } else {
            condition->traverse(it);
            body->traverse(it);
        }
        it->decrementDepth();
    }

    if (visit && it->postVisit)
        it->visitSwitch(EvPostVisit, this);
}

//
// Traverse a variable declaration.
//
void TIntermVariableDecl::traverse(TIntermTraverser *it)
{
    bool visit = true;

    if (it->preVisit)
        visit = it->visitVariableDecl(EvPreVisit, this);

    if (visit) {
        it->incrementDepth(this);
        if (it->rightToLeft) {
            if (it->includeDeclSymbol)
                declSymbol->traverse(it);
            if (initNode)
                initNode->traverse(it);
        }
        else {
            if (initNode)
                initNode->traverse(it);
            if (it->includeDeclSymbol)
                declSymbol->traverse(it);
        }
        it->decrementDepth();
    }

    if (visit && it->postVisit)
        it->visitVariableDecl(EvPostVisit, this);
}

} // end namespace glslang
