/*
 * Copyright (C) 2012 Apple Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY APPLE, INC. ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL APPLE, INC. OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "third_party/compiler/ArrayBoundsClamper.h"

#include "compiler/translator/tree_util/IntermTraverse.h"

// The built-in 'clamp' instruction only accepts floats and returns a float.  I
// iterated a few times with our driver team who examined the output from our
// compiler - they said the multiple casts generates more code than a single
// function call.  An inline ternary operator might have been better, but since
// the index value might be an expression itself, we'd have to make temporary
// variables to avoid evaluating the expression multiple times.  And making
// temporary variables was difficult because ANGLE would then need to make more
// brutal changes to the expression tree.

const char *kIntClampBegin = "// BEGIN: Generated code for array bounds clamping\n\n";
const char *kIntClampEnd   = "// END: Generated code for array bounds clamping\n\n";
const char *kIntClampDefinition =
    "int webgl_int_clamp(int value, int minValue, int maxValue) { return ((value < minValue) ? "
    "minValue : ((value > maxValue) ? maxValue : value)); }\n\n";

namespace sh
{

namespace
{

class ArrayBoundsClamperMarker : public TIntermTraverser
{
  public:
    ArrayBoundsClamperMarker() : TIntermTraverser(true, false, false), mNeedsClamp(false) {}

    bool visitBinary(Visit visit, TIntermBinary *node) override
    {
        if (node->getOp() == EOpIndexIndirect)
        {
            TIntermTyped *left = node->getLeft();
            if (left->isArray() || left->isVector() || left->isMatrix())
            {
                node->setAddIndexClamp();
                mNeedsClamp = true;
            }
        }
        return true;
    }

    bool GetNeedsClamp() { return mNeedsClamp; }

  private:
    bool mNeedsClamp;
};

}  // anonymous namespace

ArrayBoundsClamper::ArrayBoundsClamper()
    : mClampingStrategy(SH_CLAMP_WITH_CLAMP_INTRINSIC), mArrayBoundsClampDefinitionNeeded(false)
{}

void ArrayBoundsClamper::SetClampingStrategy(ShArrayIndexClampingStrategy clampingStrategy)
{
    mClampingStrategy = clampingStrategy;
}

void ArrayBoundsClamper::MarkIndirectArrayBoundsForClamping(TIntermNode *root)
{
    ASSERT(root);

    ArrayBoundsClamperMarker clamper;
    root->traverse(&clamper);
    if (clamper.GetNeedsClamp())
    {
        SetArrayBoundsClampDefinitionNeeded();
    }
}

void ArrayBoundsClamper::OutputClampingFunctionDefinition(TInfoSinkBase &out) const
{
    if (!mArrayBoundsClampDefinitionNeeded)
    {
        return;
    }
    if (mClampingStrategy != SH_CLAMP_WITH_USER_DEFINED_INT_CLAMP_FUNCTION)
    {
        return;
    }
    out << kIntClampBegin << kIntClampDefinition << kIntClampEnd;
}

}  // namespace sh
