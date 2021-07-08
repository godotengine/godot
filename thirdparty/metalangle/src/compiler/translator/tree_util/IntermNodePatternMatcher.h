//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// IntermNodePatternMatcher is a helper class for matching node trees to given patterns.
// It can be used whenever the same checks for certain node structures are common to multiple AST
// traversers.
//

#ifndef COMPILER_TRANSLATOR_TREEUTIL_INTERMNODEPATTERNMATCHER_H_
#define COMPILER_TRANSLATOR_TREEUTIL_INTERMNODEPATTERNMATCHER_H_

namespace sh
{

class TIntermAggregate;
class TIntermBinary;
class TIntermDeclaration;
class TIntermNode;
class TIntermTernary;
class TIntermUnary;

class IntermNodePatternMatcher
{
  public:
    static bool IsDynamicIndexingOfNonSSBOVectorOrMatrix(TIntermBinary *node);
    static bool IsDynamicIndexingOfVectorOrMatrix(TIntermBinary *node);

    enum PatternType
    {
        // Matches expressions that are unfolded to if statements by UnfoldShortCircuitToIf
        kUnfoldedShortCircuitExpression = 0x0001,

        // Matches expressions that return arrays with the exception of simple statements where a
        // constructor or function call result is assigned.
        kExpressionReturningArray = 0x0001 << 1,

        // Matches dynamic indexing of vectors or matrices in l-values.
        kDynamicIndexingOfVectorOrMatrixInLValue = 0x0001 << 2,

        // Matches declarations with more than one declared variables.
        kMultiDeclaration = 0x0001 << 3,

        // Matches declarations of arrays.
        kArrayDeclaration = 0x0001 << 4,

        // Matches declarations of structs where the struct type does not have a name.
        kNamelessStructDeclaration = 0x0001 << 5,

        // Matches array length() method.
        kArrayLengthMethod = 0x0001 << 6,

        // Matches a vector or matrix constructor whose arguments are scalarized by the
        // SH_SCALARIZE_VEC_OR_MAT_CONSTRUCTOR_ARGUMENTS workaround.
        kScalarizedVecOrMatConstructor = 0x0001 << 7
    };
    IntermNodePatternMatcher(const unsigned int mask);

    bool match(TIntermUnary *node);

    bool match(TIntermBinary *node, TIntermNode *parentNode);

    // Use this version for checking binary node matches in case you're using flag
    // kDynamicIndexingOfVectorOrMatrixInLValue.
    bool match(TIntermBinary *node, TIntermNode *parentNode, bool isLValueRequiredHere);

    bool match(TIntermAggregate *node, TIntermNode *parentNode);
    bool match(TIntermTernary *node);
    bool match(TIntermDeclaration *node);

  private:
    const unsigned int mMask;

    bool matchInternal(TIntermBinary *node, TIntermNode *parentNode);
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEUTIL_INTERMNODEPATTERNMATCHER_H_
