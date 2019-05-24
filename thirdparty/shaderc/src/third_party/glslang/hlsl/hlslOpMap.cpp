//
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
//    Neither the name of Google, Inc., nor the names of its
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

// Map from physical token form (e.g. '-') to logical operator
// form (e.g., binary subtract or unary negate).

#include "hlslOpMap.h"

namespace glslang {

// Map parsing tokens that could be assignments into assignment operators.
TOperator HlslOpMap::assignment(EHlslTokenClass op)
{
    switch (op) {
    case EHTokAssign:      return EOpAssign;
    case EHTokMulAssign:   return EOpMulAssign;
    case EHTokDivAssign:   return EOpDivAssign;
    case EHTokAddAssign:   return EOpAddAssign;
    case EHTokModAssign:   return EOpModAssign;
    case EHTokLeftAssign:  return EOpLeftShiftAssign;
    case EHTokRightAssign: return EOpRightShiftAssign;
    case EHTokAndAssign:   return EOpAndAssign;
    case EHTokXorAssign:   return EOpExclusiveOrAssign;
    case EHTokOrAssign:    return EOpInclusiveOrAssign;
    case EHTokSubAssign:   return EOpSubAssign;

    default:
        return EOpNull;
    }
}

// Map parsing tokens that could be binary operations into binary operators.
TOperator HlslOpMap::binary(EHlslTokenClass op)
{
    switch (op) {
    case EHTokPlus:        return EOpAdd;
    case EHTokDash:        return EOpSub;
    case EHTokStar:        return EOpMul;
    case EHTokSlash:       return EOpDiv;
    case EHTokPercent:     return EOpMod;
    case EHTokRightOp:     return EOpRightShift;
    case EHTokLeftOp:      return EOpLeftShift;
    case EHTokAmpersand:   return EOpAnd;
    case EHTokVerticalBar: return EOpInclusiveOr;
    case EHTokCaret:       return EOpExclusiveOr;
    case EHTokEqOp:        return EOpEqual;
    case EHTokNeOp:        return EOpNotEqual;
    case EHTokLeftAngle:   return EOpLessThan;
    case EHTokRightAngle:  return EOpGreaterThan;
    case EHTokLeOp:        return EOpLessThanEqual;
    case EHTokGeOp:        return EOpGreaterThanEqual;
    case EHTokOrOp:        return EOpLogicalOr;
    case EHTokXorOp:       return EOpLogicalXor;
    case EHTokAndOp:       return EOpLogicalAnd;

    default:
        return EOpNull;
    }
}

// Map parsing tokens that could be unary operations into unary operators.
// These are just the ones that can appear in front of its operand.
TOperator HlslOpMap::preUnary(EHlslTokenClass op)
{
    switch (op) {
    case EHTokPlus:       return EOpAdd;        // means no-op, but still a unary op was present
    case EHTokDash:       return EOpNegative;
    case EHTokBang:       return EOpLogicalNot;
    case EHTokTilde:      return EOpBitwiseNot;

    case EHTokIncOp:      return EOpPreIncrement;
    case EHTokDecOp:      return EOpPreDecrement;

    default:              return EOpNull;       // means not a pre-unary op
    }
}

// Map parsing tokens that could be unary operations into unary operators.
// These are just the ones that can appear behind its operand.
TOperator HlslOpMap::postUnary(EHlslTokenClass op)
{
    switch (op) {
    case EHTokDot:         return EOpIndexDirectStruct;
    case EHTokLeftBracket: return EOpIndexIndirect;

    case EHTokIncOp:       return EOpPostIncrement;
    case EHTokDecOp:       return EOpPostDecrement;

    case EHTokColonColon:  return EOpScoping;

    default:               return EOpNull;             // means not a post-unary op
    }
}

// Map operators into their level of precedence.
PrecedenceLevel HlslOpMap::precedenceLevel(TOperator op)
{
    switch (op) {
    case EOpLogicalOr:
        return PlLogicalOr;
    case EOpLogicalXor:
        return PlLogicalXor;
    case EOpLogicalAnd:
        return PlLogicalAnd;

    case EOpInclusiveOr:
        return PlBitwiseOr;
    case EOpExclusiveOr:
        return PlBitwiseXor;
    case EOpAnd:
        return PlBitwiseAnd;

    case EOpEqual:
    case EOpNotEqual:
        return PlEquality;

    case EOpLessThan:
    case EOpGreaterThan:
    case EOpLessThanEqual:
    case EOpGreaterThanEqual:
        return PlRelational;

    case EOpRightShift:
    case EOpLeftShift:
        return PlShift;

    case EOpAdd:
    case EOpSub:
        return PlAdd;

    case EOpMul:
    case EOpDiv:
    case EOpMod:
        return PlMul;

    default:
        return PlBad;
    }
}

} // end namespace glslang
