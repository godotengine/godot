//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2016 LunarG, Inc.
// Copyright (C) 2015-2020 Google, Inc.
// Copyright (C) 2017, 2022-2024 Arm Limited.
// Modifications Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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
// Create strings that declare built-in definitions, add built-ins programmatically
// that cannot be expressed in the strings, and establish mappings between
// built-in functions and operators.
//
// Where to put a built-in:
//   TBuiltIns::initialize(version,profile)       context-independent textual built-ins; add them to the right string
//   TBuiltIns::initialize(resources,...)         context-dependent textual built-ins; add them to the right string
//   TBuiltIns::identifyBuiltIns(...,symbolTable) context-independent programmatic additions/mappings to the symbol table,
//                                                including identifying what extensions are needed if a version does not allow a symbol
//   TBuiltIns::identifyBuiltIns(...,symbolTable, resources) context-dependent programmatic additions/mappings to the symbol table,
//                                                including identifying what extensions are needed if a version does not allow a symbol
//

#include <array>
#include "Initialize.h"
#include "span.h"

namespace glslang {

// TODO: ARB_Compatability: do full extension support
const bool ARBCompatibility = true;

const bool ForwardCompatibility = false;

namespace {

//
// A set of definitions for tabling of the built-in functions.
//

// Order matters here, as does correlation with the subsequent
// "const int ..." declarations and the ArgType enumerants.
const char* TypeString[] = {
   "bool",  "bvec2", "bvec3", "bvec4",
   "float",  "vec2",  "vec3",  "vec4",
   "int",   "ivec2", "ivec3", "ivec4",
   "uint",  "uvec2", "uvec3", "uvec4",
};
const int TypeStringCount = sizeof(TypeString) / sizeof(char*); // number of entries in 'TypeString'
const int TypeStringRowShift = 2;                               // shift amount to go downe one row in 'TypeString'
const int TypeStringColumnMask = (1 << TypeStringRowShift) - 1; // reduce type to its column number in 'TypeString'
const int TypeStringScalarMask = ~TypeStringColumnMask;         // take type to its scalar column in 'TypeString'

enum ArgType {
    // numbers hardcoded to correspond to 'TypeString'; order and value matter
    TypeB    = 1 << 0,  // Boolean
    TypeF    = 1 << 1,  // float 32
    TypeI    = 1 << 2,  // int 32
    TypeU    = 1 << 3,  // uint 32
    TypeF16  = 1 << 4,  // float 16
    TypeF64  = 1 << 5,  // float 64
    TypeI8   = 1 << 6,  // int 8
    TypeI16  = 1 << 7,  // int 16
    TypeI64  = 1 << 8,  // int 64
    TypeU8   = 1 << 9,  // uint 8
    TypeU16  = 1 << 10, // uint 16
    TypeU64  = 1 << 11, // uint 64
};
// Mixtures of the above, to help the function tables
const ArgType TypeFI  = static_cast<ArgType>(TypeF | TypeI);
const ArgType TypeFIB = static_cast<ArgType>(TypeF | TypeI | TypeB);
const ArgType TypeIU  = static_cast<ArgType>(TypeI | TypeU);

// The relationships between arguments and return type, whether anything is
// output, or other unusual situations.
enum ArgClass {
    ClassRegular     = 0,  // nothing special, just all vector widths with matching return type; traditional arithmetic
    ClassLS     = 1 << 0,  // the last argument is also held fixed as a (type-matched) scalar while the others cycle
    ClassXLS    = 1 << 1,  // the last argument is exclusively a (type-matched) scalar while the others cycle
    ClassLS2    = 1 << 2,  // the last two arguments are held fixed as a (type-matched) scalar while the others cycle
    ClassFS     = 1 << 3,  // the first argument is held fixed as a (type-matched) scalar while the others cycle
    ClassFS2    = 1 << 4,  // the first two arguments are held fixed as a (type-matched) scalar while the others cycle
    ClassLO     = 1 << 5,  // the last argument is an output
    ClassB      = 1 << 6,  // return type cycles through only bool/bvec, matching vector width of args
    ClassLB     = 1 << 7,  // last argument cycles through only bool/bvec, matching vector width of args
    ClassV1     = 1 << 8,  // scalar only
    ClassFIO    = 1 << 9,  // first argument is inout
    ClassRS     = 1 << 10, // the return is held scalar as the arguments cycle
    ClassNS     = 1 << 11, // no scalar prototype
    ClassCV     = 1 << 12, // first argument is 'coherent volatile'
    ClassFO     = 1 << 13, // first argument is output
    ClassV3     = 1 << 14, // vec3 only
};
// Mixtures of the above, to help the function tables
const ArgClass ClassV1FIOCV = (ArgClass)(ClassV1 | ClassFIO | ClassCV);
const ArgClass ClassBNS     = (ArgClass)(ClassB  | ClassNS);
const ArgClass ClassRSNS    = (ArgClass)(ClassRS | ClassNS);

// A descriptor, for a single profile, of when something is available.
// If the current profile does not match 'profile' mask below, the other fields
// do not apply (nor validate).
// profiles == EBadProfile is the end of an array of these
struct Versioning {
    EProfile profiles;       // the profile(s) (mask) that the following fields are valid for
    int minExtendedVersion;  // earliest version when extensions are enabled; ignored if numExtensions is 0
    int minCoreVersion;      // earliest version function is in core; 0 means never
    int numExtensions;       // how many extensions are in the 'extensions' list
    const char** extensions; // list of extension names enabling the function
};

EProfile EDesktopProfile = static_cast<EProfile>(ENoProfile | ECoreProfile | ECompatibilityProfile);

// Declare pointers to put into the table for versioning.
    const std::array Es300Desktop130Version = { Versioning{ EEsProfile,      0, 300, 0, nullptr },
                                                Versioning{ EDesktopProfile, 0, 130, 0, nullptr },
                                              };

    const std::array Es310Desktop400Version = { Versioning{ EEsProfile,      0, 310, 0, nullptr },
                                                Versioning{ EDesktopProfile, 0, 400, 0, nullptr },
                                              };

    const std::array Es310Desktop450Version = { Versioning{ EEsProfile,      0, 310, 0, nullptr },
                                                Versioning{ EDesktopProfile, 0, 450, 0, nullptr },
                                              };

// The main descriptor of what a set of function prototypes can look like, and
// a pointer to extra versioning information, when needed.
struct BuiltInFunction {
    TOperator op;                 // operator to map the name to
    const char* name;             // function name
    int numArguments;             // number of arguments (overloads with varying arguments need different entries)
    ArgType types;                // ArgType mask
    ArgClass classes;             // the ways this particular function entry manifests
    const span<const Versioning> versioning; // An empty span means always a valid version
};

// The tables can have the same built-in function name more than one time,
// but the exact same prototype must be indicated at most once.
// The prototypes that get declared are the union of all those indicated.
// This is important when different releases add new prototypes for the same name.
// It also also congnitively simpler tiling of the prototype space.
// In practice, most names can be fully represented with one entry.
//
// Table is terminated by an OpNull TOperator.

const std::array BaseFunctions = {
//    TOperator,           name,       arg-count,   ArgType,   ArgClass,     versioning
//    ---------            ----        ---------    -------    --------      ----------
    BuiltInFunction{ EOpRadians,          "radians",          1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpDegrees,          "degrees",          1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpSin,              "sin",              1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpCos,              "cos",              1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpTan,              "tan",              1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpAsin,             "asin",             1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpAcos,             "acos",             1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpAtan,             "atan",             2,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpAtan,             "atan",             1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpPow,              "pow",              2,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpExp,              "exp",              1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpLog,              "log",              1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpExp2,             "exp2",             1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpLog2,             "log2",             1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpSqrt,             "sqrt",             1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpInverseSqrt,      "inversesqrt",      1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpAbs,              "abs",              1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpSign,             "sign",             1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpFloor,            "floor",            1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpCeil,             "ceil",             1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpFract,            "fract",            1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpMod,              "mod",              2,   TypeF,     ClassLS,      {} },
    BuiltInFunction{ EOpMin,              "min",              2,   TypeF,     ClassLS,      {} },
    BuiltInFunction{ EOpMax,              "max",              2,   TypeF,     ClassLS,      {} },
    BuiltInFunction{ EOpClamp,            "clamp",            3,   TypeF,     ClassLS2,     {} },
    BuiltInFunction{ EOpMix,              "mix",              3,   TypeF,     ClassLS,      {} },
    BuiltInFunction{ EOpStep,             "step",             2,   TypeF,     ClassFS,      {} },
    BuiltInFunction{ EOpSmoothStep,       "smoothstep",       3,   TypeF,     ClassFS2,     {} },
    BuiltInFunction{ EOpNormalize,        "normalize",        1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpFaceForward,      "faceforward",      3,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpReflect,          "reflect",          2,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpRefract,          "refract",          3,   TypeF,     ClassXLS,     {} },
    BuiltInFunction{ EOpLength,           "length",           1,   TypeF,     ClassRS,      {} },
    BuiltInFunction{ EOpDistance,         "distance",         2,   TypeF,     ClassRS,      {} },
    BuiltInFunction{ EOpDot,              "dot",              2,   TypeF,     ClassRS,      {} },
    BuiltInFunction{ EOpCross,            "cross",            2,   TypeF,     ClassV3,      {} },
    BuiltInFunction{ EOpLessThan,         "lessThan",         2,   TypeFI,    ClassBNS,     {} },
    BuiltInFunction{ EOpLessThanEqual,    "lessThanEqual",    2,   TypeFI,    ClassBNS,     {} },
    BuiltInFunction{ EOpGreaterThan,      "greaterThan",      2,   TypeFI,    ClassBNS,     {} },
    BuiltInFunction{ EOpGreaterThanEqual, "greaterThanEqual", 2,   TypeFI,    ClassBNS,     {} },
    BuiltInFunction{ EOpVectorEqual,      "equal",            2,   TypeFIB,   ClassBNS,     {} },
    BuiltInFunction{ EOpVectorNotEqual,   "notEqual",         2,   TypeFIB,   ClassBNS,     {} },
    BuiltInFunction{ EOpAny,              "any",              1,   TypeB,     ClassRSNS,    {} },
    BuiltInFunction{ EOpAll,              "all",              1,   TypeB,     ClassRSNS,    {} },
    BuiltInFunction{ EOpVectorLogicalNot, "not",              1,   TypeB,     ClassNS,      {} },
    BuiltInFunction{ EOpSinh,             "sinh",             1,   TypeF,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpCosh,             "cosh",             1,   TypeF,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpTanh,             "tanh",             1,   TypeF,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpAsinh,            "asinh",            1,   TypeF,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpAcosh,            "acosh",            1,   TypeF,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpAtanh,            "atanh",            1,   TypeF,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpAbs,              "abs",              1,   TypeI,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpSign,             "sign",             1,   TypeI,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpTrunc,            "trunc",            1,   TypeF,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpRound,            "round",            1,   TypeF,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpRoundEven,        "roundEven",        1,   TypeF,     ClassRegular, {Es300Desktop130Version} },
    BuiltInFunction{ EOpModf,             "modf",             2,   TypeF,     ClassLO,      {Es300Desktop130Version} },
    BuiltInFunction{ EOpMin,              "min",              2,   TypeIU,    ClassLS,      {Es300Desktop130Version} },
    BuiltInFunction{ EOpMax,              "max",              2,   TypeIU,    ClassLS,      {Es300Desktop130Version} },
    BuiltInFunction{ EOpClamp,            "clamp",            3,   TypeIU,    ClassLS2,     {Es300Desktop130Version} },
    BuiltInFunction{ EOpMix,              "mix",              3,   TypeF,     ClassLB,      {Es300Desktop130Version} },
    BuiltInFunction{ EOpIsInf,            "isinf",            1,   TypeF,     ClassB,       {Es300Desktop130Version} },
    BuiltInFunction{ EOpIsNan,            "isnan",            1,   TypeF,     ClassB,       {Es300Desktop130Version} },
    BuiltInFunction{ EOpLessThan,         "lessThan",         2,   TypeU,     ClassBNS,     {Es300Desktop130Version} },
    BuiltInFunction{ EOpLessThanEqual,    "lessThanEqual",    2,   TypeU,     ClassBNS,     {Es300Desktop130Version} },
    BuiltInFunction{ EOpGreaterThan,      "greaterThan",      2,   TypeU,     ClassBNS,     {Es300Desktop130Version} },
    BuiltInFunction{ EOpGreaterThanEqual, "greaterThanEqual", 2,   TypeU,     ClassBNS,     {Es300Desktop130Version} },
    BuiltInFunction{ EOpVectorEqual,      "equal",            2,   TypeU,     ClassBNS,     {Es300Desktop130Version} },
    BuiltInFunction{ EOpVectorNotEqual,   "notEqual",         2,   TypeU,     ClassBNS,     {Es300Desktop130Version} },
    BuiltInFunction{ EOpAtomicAdd,        "atomicAdd",        2,   TypeIU,    ClassV1FIOCV, {Es310Desktop400Version} },
    BuiltInFunction{ EOpAtomicMin,        "atomicMin",        2,   TypeIU,    ClassV1FIOCV, {Es310Desktop400Version} },
    BuiltInFunction{ EOpAtomicMax,        "atomicMax",        2,   TypeIU,    ClassV1FIOCV, {Es310Desktop400Version} },
    BuiltInFunction{ EOpAtomicAnd,        "atomicAnd",        2,   TypeIU,    ClassV1FIOCV, {Es310Desktop400Version} },
    BuiltInFunction{ EOpAtomicOr,         "atomicOr",         2,   TypeIU,    ClassV1FIOCV, {Es310Desktop400Version} },
    BuiltInFunction{ EOpAtomicXor,        "atomicXor",        2,   TypeIU,    ClassV1FIOCV, {Es310Desktop400Version} },
    BuiltInFunction{ EOpAtomicExchange,   "atomicExchange",   2,   TypeIU,    ClassV1FIOCV, {Es310Desktop400Version} },
    BuiltInFunction{ EOpAtomicCompSwap,   "atomicCompSwap",   3,   TypeIU,    ClassV1FIOCV, {Es310Desktop400Version} },
    BuiltInFunction{ EOpMix,              "mix",              3,   TypeB,     ClassRegular, {Es310Desktop450Version} },
    BuiltInFunction{ EOpMix,              "mix",              3,   TypeIU,    ClassLB,      {Es310Desktop450Version} },
};

const std::array DerivativeFunctions = {
    BuiltInFunction{ EOpDPdx,             "dFdx",             1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpDPdy,             "dFdy",             1,   TypeF,     ClassRegular, {} },
    BuiltInFunction{ EOpFwidth,           "fwidth",           1,   TypeF,     ClassRegular, {} },
};

// For functions declared some other way, but still use the table to relate to operator.
struct CustomFunction {
    TOperator op;                 // operator to map the name to
    const char* name;             // function name
    const span<const Versioning> versioning; // An empty span means always a valid version
};

const CustomFunction CustomFunctions[] = {
    { EOpBarrier,             "barrier",             {} },
    { EOpMemoryBarrierShared, "memoryBarrierShared", {} },
    { EOpGroupMemoryBarrier,  "groupMemoryBarrier",  {} },
    { EOpMemoryBarrier,       "memoryBarrier",       {} },
    { EOpMemoryBarrierBuffer, "memoryBarrierBuffer", {} },

    { EOpPackSnorm2x16,       "packSnorm2x16",       {} },
    { EOpUnpackSnorm2x16,     "unpackSnorm2x16",     {} },
    { EOpPackUnorm2x16,       "packUnorm2x16",       {} },
    { EOpUnpackUnorm2x16,     "unpackUnorm2x16",     {} },
    { EOpPackHalf2x16,        "packHalf2x16",        {} },
    { EOpUnpackHalf2x16,      "unpackHalf2x16",      {} },

    { EOpMul,                 "matrixCompMult",      {} },
    { EOpOuterProduct,        "outerProduct",        {} },
    { EOpTranspose,           "transpose",           {} },
    { EOpDeterminant,         "determinant",         {} },
    { EOpMatrixInverse,       "inverse",             {} },
    { EOpFloatBitsToInt,      "floatBitsToInt",      {} },
    { EOpFloatBitsToUint,     "floatBitsToUint",     {} },
    { EOpIntBitsToFloat,      "intBitsToFloat",      {} },
    { EOpUintBitsToFloat,     "uintBitsToFloat",     {} },

    { EOpTextureQuerySize,      "textureSize",           {} },
    { EOpTextureQueryLod,       "textureQueryLod",       {} },
    { EOpTextureQueryLod,       "textureQueryLOD",       {} }, // extension GL_ARB_texture_query_lod
    { EOpTextureQueryLevels,    "textureQueryLevels",    {} },
    { EOpTextureQuerySamples,   "textureSamples",        {} },
    { EOpTexture,               "texture",               {} },
    { EOpTextureProj,           "textureProj",           {} },
    { EOpTextureLod,            "textureLod",            {} },
    { EOpTextureOffset,         "textureOffset",         {} },
    { EOpTextureFetch,          "texelFetch",            {} },
    { EOpTextureFetchOffset,    "texelFetchOffset",      {} },
    { EOpTextureProjOffset,     "textureProjOffset",     {} },
    { EOpTextureLodOffset,      "textureLodOffset",      {} },
    { EOpTextureProjLod,        "textureProjLod",        {} },
    { EOpTextureProjLodOffset,  "textureProjLodOffset",  {} },
    { EOpTextureGrad,           "textureGrad",           {} },
    { EOpTextureGradOffset,     "textureGradOffset",     {} },
    { EOpTextureProjGrad,       "textureProjGrad",       {} },
    { EOpTextureProjGradOffset, "textureProjGradOffset", {} },
};

// For the given table of functions, add all the indicated prototypes for each
// one, to be returned in the passed in decls.
void AddTabledBuiltin(TString& decls, const BuiltInFunction& function)
{
    const auto isScalarType = [](int type) { return (type & TypeStringColumnMask) == 0; };

    // loop across these two:
    //  0: the varying arg set, and
    //  1: the fixed scalar args
    const ArgClass ClassFixed = (ArgClass)(ClassLS | ClassXLS | ClassLS2 | ClassFS | ClassFS2);
    for (int fixed = 0; fixed < ((function.classes & ClassFixed) > 0 ? 2 : 1); ++fixed) {

        if (fixed == 0 && (function.classes & ClassXLS))
            continue;

        // walk the type strings in TypeString[]
        for (int type = 0; type < TypeStringCount; ++type) {
            // skip types not selected: go from type to row number to type bit
            if ((function.types & (1 << (type >> TypeStringRowShift))) == 0)
                continue;

            // if we aren't on a scalar, and should be, skip
            if ((function.classes & ClassV1) && !isScalarType(type))
                continue;

            // if we aren't on a 3-vector, and should be, skip
            if ((function.classes & ClassV3) && (type & TypeStringColumnMask) != 2)
                continue;

            // skip replication of all arg scalars between the varying arg set and the fixed args
            if (fixed == 1 && type == (type & TypeStringScalarMask) && (function.classes & ClassXLS) == 0)
                continue;

            // skip scalars when we are told to
            if ((function.classes & ClassNS) && isScalarType(type))
                continue;

            // return type
            if (function.classes & ClassB)
                decls.append(TypeString[type & TypeStringColumnMask]);
            else if (function.classes & ClassRS)
                decls.append(TypeString[type & TypeStringScalarMask]);
            else
                decls.append(TypeString[type]);
            decls.append(" ");
            decls.append(function.name);
            decls.append("(");

            // arguments
            for (int arg = 0; arg < function.numArguments; ++arg) {
                if (arg == function.numArguments - 1 && (function.classes & ClassLO))
                    decls.append("out ");
                if (arg == 0) {
                    if (function.classes & ClassCV)
                        decls.append("coherent volatile ");
                    if (function.classes & ClassFIO)
                        decls.append("inout ");
                    if (function.classes & ClassFO)
                        decls.append("out ");
                }
                if ((function.classes & ClassLB) && arg == function.numArguments - 1)
                    decls.append(TypeString[type & TypeStringColumnMask]);
                else if (fixed && ((arg == function.numArguments - 1 && (function.classes & (ClassLS | ClassXLS |
                                                                                                       ClassLS2))) ||
                                   (arg == function.numArguments - 2 && (function.classes & ClassLS2))             ||
                                   (arg == 0                         && (function.classes & (ClassFS | ClassFS2))) ||
                                   (arg == 1                         && (function.classes & ClassFS2))))
                    decls.append(TypeString[type & TypeStringScalarMask]);
                else
                    decls.append(TypeString[type]);
                if (arg < function.numArguments - 1)
                    decls.append(",");
            }
            decls.append(");\n");
        }
    }
}

// See if the tabled versioning information allows the current version.
bool ValidVersion(const BuiltInFunction& function, int version, EProfile profile, const SpvVersion& /* spVersion */)
{
    // nullptr means always valid
    if (function.versioning.empty())
        return true;

    // check for what is said about our current profile
    for (const auto& v : function.versioning) {
        if ((v.profiles & profile) != 0) {
            if (v.minCoreVersion <= version || (v.numExtensions > 0 && v.minExtendedVersion <= version))
                return true;
        }
    }

    return false;
}

// Relate a single table of built-ins to their AST operator.
// This can get called redundantly (especially for the common built-ins, when
// called once per stage). This is a performance issue only, not a correctness
// concern.  It is done for quality arising from simplicity, as there are subtleties
// to get correct if instead trying to do it surgically.
template<class FunctionContainer>
void RelateTabledBuiltins(const FunctionContainer& functions, TSymbolTable& symbolTable)
{
    for (const auto& fn : functions) {
        symbolTable.relateToOperator(fn.name, fn.op);
    }
}

} // end anonymous namespace

// Add declarations for all tables of built-in functions.
void TBuiltIns::addTabledBuiltins(int version, EProfile profile, const SpvVersion& spvVersion)
{
    const auto forEachFunction = [&](TString& decls, const span<const BuiltInFunction>& functions) {
        for (const auto& fn : functions) {
            if (ValidVersion(fn, version, profile, spvVersion))
                AddTabledBuiltin(decls, fn);
        }
    };

    forEachFunction(commonBuiltins, BaseFunctions);
    forEachFunction(stageBuiltins[EShLangFragment], DerivativeFunctions);

    if ((profile == EEsProfile && version >= 320) || (profile != EEsProfile && version >= 450))
        forEachFunction(stageBuiltins[EShLangCompute], DerivativeFunctions);
}

// Relate all tables of built-ins to the AST operators.
void TBuiltIns::relateTabledBuiltins(int /* version */, EProfile /* profile */, const SpvVersion& /* spvVersion */, EShLanguage /* stage */, TSymbolTable& symbolTable)
{
    RelateTabledBuiltins(BaseFunctions, symbolTable);
    RelateTabledBuiltins(DerivativeFunctions, symbolTable);
    RelateTabledBuiltins(CustomFunctions, symbolTable);
}

inline bool IncludeLegacy(int version, EProfile profile, const SpvVersion& spvVersion)
{
    return profile != EEsProfile && (version <= 130 || (spvVersion.spv == 0 && version == 140 && ARBCompatibility) ||
           profile == ECompatibilityProfile);
}

// Construct TBuiltInParseables base class.  This can be used for language-common constructs.
TBuiltInParseables::TBuiltInParseables()
{
}

// Destroy TBuiltInParseables.
TBuiltInParseables::~TBuiltInParseables()
{
}

TBuiltIns::TBuiltIns()
{
    // Set up textual representations for making all the permutations
    // of texturing/imaging functions.
    prefixes[EbtFloat] =  "";
    prefixes[EbtInt]   = "i";
    prefixes[EbtUint]  = "u";
    prefixes[EbtFloat16] = "f16";
    prefixes[EbtInt8]  = "i8";
    prefixes[EbtUint8] = "u8";
    prefixes[EbtInt16]  = "i16";
    prefixes[EbtUint16] = "u16";
    prefixes[EbtInt64]  = "i64";
    prefixes[EbtUint64] = "u64";

    postfixes[2] = "2";
    postfixes[3] = "3";
    postfixes[4] = "4";

    // Map from symbolic class of texturing dimension to numeric dimensions.
    dimMap[Esd2D] = 2;
    dimMap[Esd3D] = 3;
    dimMap[EsdCube] = 3;
    dimMap[Esd1D] = 1;
    dimMap[EsdRect] = 2;
    dimMap[EsdBuffer] = 1;
    dimMap[EsdSubpass] = 2;  // potentially unused for now
    dimMap[EsdAttachmentEXT] = 2;  // potentially unused for now
}

TBuiltIns::~TBuiltIns()
{
}


//
// Add all context-independent built-in functions and variables that are present
// for the given version and profile.  Share common ones across stages, otherwise
// make stage-specific entries.
//
// Most built-ins variables can be added as simple text strings.  Some need to
// be added programmatically, which is done later in IdentifyBuiltIns() below.
//
void TBuiltIns::initialize(int version, EProfile profile, const SpvVersion& spvVersion)
{
    addTabledBuiltins(version, profile, spvVersion);

    //============================================================================
    //
    // Prototypes for built-in functions used repeatly by different shaders
    //
    //============================================================================

    //
    // Derivatives Functions.
    //
    TString derivativeControls (
        "float dFdxFine(float p);"
        "vec2  dFdxFine(vec2  p);"
        "vec3  dFdxFine(vec3  p);"
        "vec4  dFdxFine(vec4  p);"

        "float dFdyFine(float p);"
        "vec2  dFdyFine(vec2  p);"
        "vec3  dFdyFine(vec3  p);"
        "vec4  dFdyFine(vec4  p);"

        "float fwidthFine(float p);"
        "vec2  fwidthFine(vec2  p);"
        "vec3  fwidthFine(vec3  p);"
        "vec4  fwidthFine(vec4  p);"

        "float dFdxCoarse(float p);"
        "vec2  dFdxCoarse(vec2  p);"
        "vec3  dFdxCoarse(vec3  p);"
        "vec4  dFdxCoarse(vec4  p);"

        "float dFdyCoarse(float p);"
        "vec2  dFdyCoarse(vec2  p);"
        "vec3  dFdyCoarse(vec3  p);"
        "vec4  dFdyCoarse(vec4  p);"

        "float fwidthCoarse(float p);"
        "vec2  fwidthCoarse(vec2  p);"
        "vec3  fwidthCoarse(vec3  p);"
        "vec4  fwidthCoarse(vec4  p);"
    );

    TString derivativesAndControl16bits (
        "float16_t dFdx(float16_t);"
        "f16vec2   dFdx(f16vec2);"
        "f16vec3   dFdx(f16vec3);"
        "f16vec4   dFdx(f16vec4);"

        "float16_t dFdy(float16_t);"
        "f16vec2   dFdy(f16vec2);"
        "f16vec3   dFdy(f16vec3);"
        "f16vec4   dFdy(f16vec4);"

        "float16_t dFdxFine(float16_t);"
        "f16vec2   dFdxFine(f16vec2);"
        "f16vec3   dFdxFine(f16vec3);"
        "f16vec4   dFdxFine(f16vec4);"

        "float16_t dFdyFine(float16_t);"
        "f16vec2   dFdyFine(f16vec2);"
        "f16vec3   dFdyFine(f16vec3);"
        "f16vec4   dFdyFine(f16vec4);"

        "float16_t dFdxCoarse(float16_t);"
        "f16vec2   dFdxCoarse(f16vec2);"
        "f16vec3   dFdxCoarse(f16vec3);"
        "f16vec4   dFdxCoarse(f16vec4);"

        "float16_t dFdyCoarse(float16_t);"
        "f16vec2   dFdyCoarse(f16vec2);"
        "f16vec3   dFdyCoarse(f16vec3);"
        "f16vec4   dFdyCoarse(f16vec4);"

        "float16_t fwidth(float16_t);"
        "f16vec2   fwidth(f16vec2);"
        "f16vec3   fwidth(f16vec3);"
        "f16vec4   fwidth(f16vec4);"

        "float16_t fwidthFine(float16_t);"
        "f16vec2   fwidthFine(f16vec2);"
        "f16vec3   fwidthFine(f16vec3);"
        "f16vec4   fwidthFine(f16vec4);"

        "float16_t fwidthCoarse(float16_t);"
        "f16vec2   fwidthCoarse(f16vec2);"
        "f16vec3   fwidthCoarse(f16vec3);"
        "f16vec4   fwidthCoarse(f16vec4);"
    );

    TString derivativesAndControl64bits (
        "float64_t dFdx(float64_t);"
        "f64vec2   dFdx(f64vec2);"
        "f64vec3   dFdx(f64vec3);"
        "f64vec4   dFdx(f64vec4);"

        "float64_t dFdy(float64_t);"
        "f64vec2   dFdy(f64vec2);"
        "f64vec3   dFdy(f64vec3);"
        "f64vec4   dFdy(f64vec4);"

        "float64_t dFdxFine(float64_t);"
        "f64vec2   dFdxFine(f64vec2);"
        "f64vec3   dFdxFine(f64vec3);"
        "f64vec4   dFdxFine(f64vec4);"

        "float64_t dFdyFine(float64_t);"
        "f64vec2   dFdyFine(f64vec2);"
        "f64vec3   dFdyFine(f64vec3);"
        "f64vec4   dFdyFine(f64vec4);"

        "float64_t dFdxCoarse(float64_t);"
        "f64vec2   dFdxCoarse(f64vec2);"
        "f64vec3   dFdxCoarse(f64vec3);"
        "f64vec4   dFdxCoarse(f64vec4);"

        "float64_t dFdyCoarse(float64_t);"
        "f64vec2   dFdyCoarse(f64vec2);"
        "f64vec3   dFdyCoarse(f64vec3);"
        "f64vec4   dFdyCoarse(f64vec4);"

        "float64_t fwidth(float64_t);"
        "f64vec2   fwidth(f64vec2);"
        "f64vec3   fwidth(f64vec3);"
        "f64vec4   fwidth(f64vec4);"

        "float64_t fwidthFine(float64_t);"
        "f64vec2   fwidthFine(f64vec2);"
        "f64vec3   fwidthFine(f64vec3);"
        "f64vec4   fwidthFine(f64vec4);"

        "float64_t fwidthCoarse(float64_t);"
        "f64vec2   fwidthCoarse(f64vec2);"
        "f64vec3   fwidthCoarse(f64vec3);"
        "f64vec4   fwidthCoarse(f64vec4);"
    );

    //============================================================================
    //
    // Prototypes for built-in functions seen by both vertex and fragment shaders.
    //
    //============================================================================

    //
    // double functions added to desktop 4.00, but not fma, frexp, ldexp, or pack/unpack
    //
    if (profile != EEsProfile && version >= 150) {  // ARB_gpu_shader_fp64
        commonBuiltins.append(

            "double sqrt(double);"
            "dvec2  sqrt(dvec2);"
            "dvec3  sqrt(dvec3);"
            "dvec4  sqrt(dvec4);"

            "double inversesqrt(double);"
            "dvec2  inversesqrt(dvec2);"
            "dvec3  inversesqrt(dvec3);"
            "dvec4  inversesqrt(dvec4);"

            "double abs(double);"
            "dvec2  abs(dvec2);"
            "dvec3  abs(dvec3);"
            "dvec4  abs(dvec4);"

            "double sign(double);"
            "dvec2  sign(dvec2);"
            "dvec3  sign(dvec3);"
            "dvec4  sign(dvec4);"

            "double floor(double);"
            "dvec2  floor(dvec2);"
            "dvec3  floor(dvec3);"
            "dvec4  floor(dvec4);"

            "double trunc(double);"
            "dvec2  trunc(dvec2);"
            "dvec3  trunc(dvec3);"
            "dvec4  trunc(dvec4);"

            "double round(double);"
            "dvec2  round(dvec2);"
            "dvec3  round(dvec3);"
            "dvec4  round(dvec4);"

            "double roundEven(double);"
            "dvec2  roundEven(dvec2);"
            "dvec3  roundEven(dvec3);"
            "dvec4  roundEven(dvec4);"

            "double ceil(double);"
            "dvec2  ceil(dvec2);"
            "dvec3  ceil(dvec3);"
            "dvec4  ceil(dvec4);"

            "double fract(double);"
            "dvec2  fract(dvec2);"
            "dvec3  fract(dvec3);"
            "dvec4  fract(dvec4);"

            "double mod(double, double);"
            "dvec2  mod(dvec2 , double);"
            "dvec3  mod(dvec3 , double);"
            "dvec4  mod(dvec4 , double);"
            "dvec2  mod(dvec2 , dvec2);"
            "dvec3  mod(dvec3 , dvec3);"
            "dvec4  mod(dvec4 , dvec4);"

            "double modf(double, out double);"
            "dvec2  modf(dvec2,  out dvec2);"
            "dvec3  modf(dvec3,  out dvec3);"
            "dvec4  modf(dvec4,  out dvec4);"

            "double min(double, double);"
            "dvec2  min(dvec2,  double);"
            "dvec3  min(dvec3,  double);"
            "dvec4  min(dvec4,  double);"
            "dvec2  min(dvec2,  dvec2);"
            "dvec3  min(dvec3,  dvec3);"
            "dvec4  min(dvec4,  dvec4);"

            "double max(double, double);"
            "dvec2  max(dvec2 , double);"
            "dvec3  max(dvec3 , double);"
            "dvec4  max(dvec4 , double);"
            "dvec2  max(dvec2 , dvec2);"
            "dvec3  max(dvec3 , dvec3);"
            "dvec4  max(dvec4 , dvec4);"

            "double clamp(double, double, double);"
            "dvec2  clamp(dvec2 , double, double);"
            "dvec3  clamp(dvec3 , double, double);"
            "dvec4  clamp(dvec4 , double, double);"
            "dvec2  clamp(dvec2 , dvec2 , dvec2);"
            "dvec3  clamp(dvec3 , dvec3 , dvec3);"
            "dvec4  clamp(dvec4 , dvec4 , dvec4);"

            "double mix(double, double, double);"
            "dvec2  mix(dvec2,  dvec2,  double);"
            "dvec3  mix(dvec3,  dvec3,  double);"
            "dvec4  mix(dvec4,  dvec4,  double);"
            "dvec2  mix(dvec2,  dvec2,  dvec2);"
            "dvec3  mix(dvec3,  dvec3,  dvec3);"
            "dvec4  mix(dvec4,  dvec4,  dvec4);"
            "double mix(double, double, bool);"
            "dvec2  mix(dvec2,  dvec2,  bvec2);"
            "dvec3  mix(dvec3,  dvec3,  bvec3);"
            "dvec4  mix(dvec4,  dvec4,  bvec4);"

            "double step(double, double);"
            "dvec2  step(dvec2 , dvec2);"
            "dvec3  step(dvec3 , dvec3);"
            "dvec4  step(dvec4 , dvec4);"
            "dvec2  step(double, dvec2);"
            "dvec3  step(double, dvec3);"
            "dvec4  step(double, dvec4);"

            "double smoothstep(double, double, double);"
            "dvec2  smoothstep(dvec2 , dvec2 , dvec2);"
            "dvec3  smoothstep(dvec3 , dvec3 , dvec3);"
            "dvec4  smoothstep(dvec4 , dvec4 , dvec4);"
            "dvec2  smoothstep(double, double, dvec2);"
            "dvec3  smoothstep(double, double, dvec3);"
            "dvec4  smoothstep(double, double, dvec4);"

            "bool  isnan(double);"
            "bvec2 isnan(dvec2);"
            "bvec3 isnan(dvec3);"
            "bvec4 isnan(dvec4);"

            "bool  isinf(double);"
            "bvec2 isinf(dvec2);"
            "bvec3 isinf(dvec3);"
            "bvec4 isinf(dvec4);"

            "double length(double);"
            "double length(dvec2);"
            "double length(dvec3);"
            "double length(dvec4);"

            "double distance(double, double);"
            "double distance(dvec2 , dvec2);"
            "double distance(dvec3 , dvec3);"
            "double distance(dvec4 , dvec4);"

            "double dot(double, double);"
            "double dot(dvec2 , dvec2);"
            "double dot(dvec3 , dvec3);"
            "double dot(dvec4 , dvec4);"

            "dvec3 cross(dvec3, dvec3);"

            "double normalize(double);"
            "dvec2  normalize(dvec2);"
            "dvec3  normalize(dvec3);"
            "dvec4  normalize(dvec4);"

            "double faceforward(double, double, double);"
            "dvec2  faceforward(dvec2,  dvec2,  dvec2);"
            "dvec3  faceforward(dvec3,  dvec3,  dvec3);"
            "dvec4  faceforward(dvec4,  dvec4,  dvec4);"

            "double reflect(double, double);"
            "dvec2  reflect(dvec2 , dvec2 );"
            "dvec3  reflect(dvec3 , dvec3 );"
            "dvec4  reflect(dvec4 , dvec4 );"

            "double refract(double, double, double);"
            "dvec2  refract(dvec2 , dvec2 , double);"
            "dvec3  refract(dvec3 , dvec3 , double);"
            "dvec4  refract(dvec4 , dvec4 , double);"

            "dmat2 matrixCompMult(dmat2, dmat2);"
            "dmat3 matrixCompMult(dmat3, dmat3);"
            "dmat4 matrixCompMult(dmat4, dmat4);"
            "dmat2x3 matrixCompMult(dmat2x3, dmat2x3);"
            "dmat2x4 matrixCompMult(dmat2x4, dmat2x4);"
            "dmat3x2 matrixCompMult(dmat3x2, dmat3x2);"
            "dmat3x4 matrixCompMult(dmat3x4, dmat3x4);"
            "dmat4x2 matrixCompMult(dmat4x2, dmat4x2);"
            "dmat4x3 matrixCompMult(dmat4x3, dmat4x3);"

            "dmat2   outerProduct(dvec2, dvec2);"
            "dmat3   outerProduct(dvec3, dvec3);"
            "dmat4   outerProduct(dvec4, dvec4);"
            "dmat2x3 outerProduct(dvec3, dvec2);"
            "dmat3x2 outerProduct(dvec2, dvec3);"
            "dmat2x4 outerProduct(dvec4, dvec2);"
            "dmat4x2 outerProduct(dvec2, dvec4);"
            "dmat3x4 outerProduct(dvec4, dvec3);"
            "dmat4x3 outerProduct(dvec3, dvec4);"

            "dmat2   transpose(dmat2);"
            "dmat3   transpose(dmat3);"
            "dmat4   transpose(dmat4);"
            "dmat2x3 transpose(dmat3x2);"
            "dmat3x2 transpose(dmat2x3);"
            "dmat2x4 transpose(dmat4x2);"
            "dmat4x2 transpose(dmat2x4);"
            "dmat3x4 transpose(dmat4x3);"
            "dmat4x3 transpose(dmat3x4);"

            "double determinant(dmat2);"
            "double determinant(dmat3);"
            "double determinant(dmat4);"

            "dmat2 inverse(dmat2);"
            "dmat3 inverse(dmat3);"
            "dmat4 inverse(dmat4);"

            "bvec2 lessThan(dvec2, dvec2);"
            "bvec3 lessThan(dvec3, dvec3);"
            "bvec4 lessThan(dvec4, dvec4);"

            "bvec2 lessThanEqual(dvec2, dvec2);"
            "bvec3 lessThanEqual(dvec3, dvec3);"
            "bvec4 lessThanEqual(dvec4, dvec4);"

            "bvec2 greaterThan(dvec2, dvec2);"
            "bvec3 greaterThan(dvec3, dvec3);"
            "bvec4 greaterThan(dvec4, dvec4);"

            "bvec2 greaterThanEqual(dvec2, dvec2);"
            "bvec3 greaterThanEqual(dvec3, dvec3);"
            "bvec4 greaterThanEqual(dvec4, dvec4);"

            "bvec2 equal(dvec2, dvec2);"
            "bvec3 equal(dvec3, dvec3);"
            "bvec4 equal(dvec4, dvec4);"

            "bvec2 notEqual(dvec2, dvec2);"
            "bvec3 notEqual(dvec3, dvec3);"
            "bvec4 notEqual(dvec4, dvec4);"

            "\n");
    }

    if (profile == EEsProfile && version >= 310) {  // Explicit Types
      commonBuiltins.append(

        "float64_t sqrt(float64_t);"
        "f64vec2  sqrt(f64vec2);"
        "f64vec3  sqrt(f64vec3);"
        "f64vec4  sqrt(f64vec4);"

        "float64_t inversesqrt(float64_t);"
        "f64vec2  inversesqrt(f64vec2);"
        "f64vec3  inversesqrt(f64vec3);"
        "f64vec4  inversesqrt(f64vec4);"

        "float64_t abs(float64_t);"
        "f64vec2  abs(f64vec2);"
        "f64vec3  abs(f64vec3);"
        "f64vec4  abs(f64vec4);"

        "float64_t sign(float64_t);"
        "f64vec2  sign(f64vec2);"
        "f64vec3  sign(f64vec3);"
        "f64vec4  sign(f64vec4);"

        "float64_t floor(float64_t);"
        "f64vec2  floor(f64vec2);"
        "f64vec3  floor(f64vec3);"
        "f64vec4  floor(f64vec4);"

        "float64_t trunc(float64_t);"
        "f64vec2  trunc(f64vec2);"
        "f64vec3  trunc(f64vec3);"
        "f64vec4  trunc(f64vec4);"

        "float64_t round(float64_t);"
        "f64vec2  round(f64vec2);"
        "f64vec3  round(f64vec3);"
        "f64vec4  round(f64vec4);"

        "float64_t roundEven(float64_t);"
        "f64vec2  roundEven(f64vec2);"
        "f64vec3  roundEven(f64vec3);"
        "f64vec4  roundEven(f64vec4);"

        "float64_t ceil(float64_t);"
        "f64vec2  ceil(f64vec2);"
        "f64vec3  ceil(f64vec3);"
        "f64vec4  ceil(f64vec4);"

        "float64_t fract(float64_t);"
        "f64vec2  fract(f64vec2);"
        "f64vec3  fract(f64vec3);"
        "f64vec4  fract(f64vec4);"

        "float64_t mod(float64_t, float64_t);"
        "f64vec2  mod(f64vec2 , float64_t);"
        "f64vec3  mod(f64vec3 , float64_t);"
        "f64vec4  mod(f64vec4 , float64_t);"
        "f64vec2  mod(f64vec2 , f64vec2);"
        "f64vec3  mod(f64vec3 , f64vec3);"
        "f64vec4  mod(f64vec4 , f64vec4);"

        "float64_t modf(float64_t, out float64_t);"
        "f64vec2  modf(f64vec2,  out f64vec2);"
        "f64vec3  modf(f64vec3,  out f64vec3);"
        "f64vec4  modf(f64vec4,  out f64vec4);"

        "float64_t min(float64_t, float64_t);"
        "f64vec2  min(f64vec2,  float64_t);"
        "f64vec3  min(f64vec3,  float64_t);"
        "f64vec4  min(f64vec4,  float64_t);"
        "f64vec2  min(f64vec2,  f64vec2);"
        "f64vec3  min(f64vec3,  f64vec3);"
        "f64vec4  min(f64vec4,  f64vec4);"

        "float64_t max(float64_t, float64_t);"
        "f64vec2  max(f64vec2 , float64_t);"
        "f64vec3  max(f64vec3 , float64_t);"
        "f64vec4  max(f64vec4 , float64_t);"
        "f64vec2  max(f64vec2 , f64vec2);"
        "f64vec3  max(f64vec3 , f64vec3);"
        "f64vec4  max(f64vec4 , f64vec4);"

        "float64_t clamp(float64_t, float64_t, float64_t);"
        "f64vec2  clamp(f64vec2 , float64_t, float64_t);"
        "f64vec3  clamp(f64vec3 , float64_t, float64_t);"
        "f64vec4  clamp(f64vec4 , float64_t, float64_t);"
        "f64vec2  clamp(f64vec2 , f64vec2 , f64vec2);"
        "f64vec3  clamp(f64vec3 , f64vec3 , f64vec3);"
        "f64vec4  clamp(f64vec4 , f64vec4 , f64vec4);"

        "float64_t mix(float64_t, float64_t, float64_t);"
        "f64vec2  mix(f64vec2,  f64vec2,  float64_t);"
        "f64vec3  mix(f64vec3,  f64vec3,  float64_t);"
        "f64vec4  mix(f64vec4,  f64vec4,  float64_t);"
        "f64vec2  mix(f64vec2,  f64vec2,  f64vec2);"
        "f64vec3  mix(f64vec3,  f64vec3,  f64vec3);"
        "f64vec4  mix(f64vec4,  f64vec4,  f64vec4);"
        "float64_t mix(float64_t, float64_t, bool);"
        "f64vec2  mix(f64vec2,  f64vec2,  bvec2);"
        "f64vec3  mix(f64vec3,  f64vec3,  bvec3);"
        "f64vec4  mix(f64vec4,  f64vec4,  bvec4);"

        "float64_t step(float64_t, float64_t);"
        "f64vec2  step(f64vec2 , f64vec2);"
        "f64vec3  step(f64vec3 , f64vec3);"
        "f64vec4  step(f64vec4 , f64vec4);"
        "f64vec2  step(float64_t, f64vec2);"
        "f64vec3  step(float64_t, f64vec3);"
        "f64vec4  step(float64_t, f64vec4);"

        "float64_t smoothstep(float64_t, float64_t, float64_t);"
        "f64vec2  smoothstep(f64vec2 , f64vec2 , f64vec2);"
        "f64vec3  smoothstep(f64vec3 , f64vec3 , f64vec3);"
        "f64vec4  smoothstep(f64vec4 , f64vec4 , f64vec4);"
        "f64vec2  smoothstep(float64_t, float64_t, f64vec2);"
        "f64vec3  smoothstep(float64_t, float64_t, f64vec3);"
        "f64vec4  smoothstep(float64_t, float64_t, f64vec4);"

        "float64_t length(float64_t);"
        "float64_t length(f64vec2);"
        "float64_t length(f64vec3);"
        "float64_t length(f64vec4);"

        "float64_t distance(float64_t, float64_t);"
        "float64_t distance(f64vec2 , f64vec2);"
        "float64_t distance(f64vec3 , f64vec3);"
        "float64_t distance(f64vec4 , f64vec4);"

        "float64_t dot(float64_t, float64_t);"
        "float64_t dot(f64vec2 , f64vec2);"
        "float64_t dot(f64vec3 , f64vec3);"
        "float64_t dot(f64vec4 , f64vec4);"

        "f64vec3 cross(f64vec3, f64vec3);"

        "float64_t normalize(float64_t);"
        "f64vec2  normalize(f64vec2);"
        "f64vec3  normalize(f64vec3);"
        "f64vec4  normalize(f64vec4);"

        "float64_t faceforward(float64_t, float64_t, float64_t);"
        "f64vec2  faceforward(f64vec2,  f64vec2,  f64vec2);"
        "f64vec3  faceforward(f64vec3,  f64vec3,  f64vec3);"
        "f64vec4  faceforward(f64vec4,  f64vec4,  f64vec4);"

        "float64_t reflect(float64_t, float64_t);"
        "f64vec2  reflect(f64vec2 , f64vec2 );"
        "f64vec3  reflect(f64vec3 , f64vec3 );"
        "f64vec4  reflect(f64vec4 , f64vec4 );"

        "float64_t refract(float64_t, float64_t, float64_t);"
        "f64vec2  refract(f64vec2 , f64vec2 , float64_t);"
        "f64vec3  refract(f64vec3 , f64vec3 , float64_t);"
        "f64vec4  refract(f64vec4 , f64vec4 , float64_t);"

        "f64mat2 matrixCompMult(f64mat2, f64mat2);"
        "f64mat3 matrixCompMult(f64mat3, f64mat3);"
        "f64mat4 matrixCompMult(f64mat4, f64mat4);"
        "f64mat2x3 matrixCompMult(f64mat2x3, f64mat2x3);"
        "f64mat2x4 matrixCompMult(f64mat2x4, f64mat2x4);"
        "f64mat3x2 matrixCompMult(f64mat3x2, f64mat3x2);"
        "f64mat3x4 matrixCompMult(f64mat3x4, f64mat3x4);"
        "f64mat4x2 matrixCompMult(f64mat4x2, f64mat4x2);"
        "f64mat4x3 matrixCompMult(f64mat4x3, f64mat4x3);"

        "f64mat2   outerProduct(f64vec2, f64vec2);"
        "f64mat3   outerProduct(f64vec3, f64vec3);"
        "f64mat4   outerProduct(f64vec4, f64vec4);"
        "f64mat2x3 outerProduct(f64vec3, f64vec2);"
        "f64mat3x2 outerProduct(f64vec2, f64vec3);"
        "f64mat2x4 outerProduct(f64vec4, f64vec2);"
        "f64mat4x2 outerProduct(f64vec2, f64vec4);"
        "f64mat3x4 outerProduct(f64vec4, f64vec3);"
        "f64mat4x3 outerProduct(f64vec3, f64vec4);"

        "f64mat2   transpose(f64mat2);"
        "f64mat3   transpose(f64mat3);"
        "f64mat4   transpose(f64mat4);"
        "f64mat2x3 transpose(f64mat3x2);"
        "f64mat3x2 transpose(f64mat2x3);"
        "f64mat2x4 transpose(f64mat4x2);"
        "f64mat4x2 transpose(f64mat2x4);"
        "f64mat3x4 transpose(f64mat4x3);"
        "f64mat4x3 transpose(f64mat3x4);"

        "float64_t determinant(f64mat2);"
        "float64_t determinant(f64mat3);"
        "float64_t determinant(f64mat4);"

        "f64mat2 inverse(f64mat2);"
        "f64mat3 inverse(f64mat3);"
        "f64mat4 inverse(f64mat4);"

        "\n");
    }

    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 310)) {
        commonBuiltins.append(

            "int64_t abs(int64_t);"
            "i64vec2 abs(i64vec2);"
            "i64vec3 abs(i64vec3);"
            "i64vec4 abs(i64vec4);"

            "int64_t sign(int64_t);"
            "i64vec2 sign(i64vec2);"
            "i64vec3 sign(i64vec3);"
            "i64vec4 sign(i64vec4);"

            "int64_t  min(int64_t,  int64_t);"
            "i64vec2  min(i64vec2,  int64_t);"
            "i64vec3  min(i64vec3,  int64_t);"
            "i64vec4  min(i64vec4,  int64_t);"
            "i64vec2  min(i64vec2,  i64vec2);"
            "i64vec3  min(i64vec3,  i64vec3);"
            "i64vec4  min(i64vec4,  i64vec4);"
            "uint64_t min(uint64_t, uint64_t);"
            "u64vec2  min(u64vec2,  uint64_t);"
            "u64vec3  min(u64vec3,  uint64_t);"
            "u64vec4  min(u64vec4,  uint64_t);"
            "u64vec2  min(u64vec2,  u64vec2);"
            "u64vec3  min(u64vec3,  u64vec3);"
            "u64vec4  min(u64vec4,  u64vec4);"

            "int64_t  max(int64_t,  int64_t);"
            "i64vec2  max(i64vec2,  int64_t);"
            "i64vec3  max(i64vec3,  int64_t);"
            "i64vec4  max(i64vec4,  int64_t);"
            "i64vec2  max(i64vec2,  i64vec2);"
            "i64vec3  max(i64vec3,  i64vec3);"
            "i64vec4  max(i64vec4,  i64vec4);"
            "uint64_t max(uint64_t, uint64_t);"
            "u64vec2  max(u64vec2,  uint64_t);"
            "u64vec3  max(u64vec3,  uint64_t);"
            "u64vec4  max(u64vec4,  uint64_t);"
            "u64vec2  max(u64vec2,  u64vec2);"
            "u64vec3  max(u64vec3,  u64vec3);"
            "u64vec4  max(u64vec4,  u64vec4);"

            "int64_t  clamp(int64_t,  int64_t,  int64_t);"
            "i64vec2  clamp(i64vec2,  int64_t,  int64_t);"
            "i64vec3  clamp(i64vec3,  int64_t,  int64_t);"
            "i64vec4  clamp(i64vec4,  int64_t,  int64_t);"
            "i64vec2  clamp(i64vec2,  i64vec2,  i64vec2);"
            "i64vec3  clamp(i64vec3,  i64vec3,  i64vec3);"
            "i64vec4  clamp(i64vec4,  i64vec4,  i64vec4);"
            "uint64_t clamp(uint64_t, uint64_t, uint64_t);"
            "u64vec2  clamp(u64vec2,  uint64_t, uint64_t);"
            "u64vec3  clamp(u64vec3,  uint64_t, uint64_t);"
            "u64vec4  clamp(u64vec4,  uint64_t, uint64_t);"
            "u64vec2  clamp(u64vec2,  u64vec2,  u64vec2);"
            "u64vec3  clamp(u64vec3,  u64vec3,  u64vec3);"
            "u64vec4  clamp(u64vec4,  u64vec4,  u64vec4);"

            "int64_t  mix(int64_t,  int64_t,  bool);"
            "i64vec2  mix(i64vec2,  i64vec2,  bvec2);"
            "i64vec3  mix(i64vec3,  i64vec3,  bvec3);"
            "i64vec4  mix(i64vec4,  i64vec4,  bvec4);"
            "uint64_t mix(uint64_t, uint64_t, bool);"
            "u64vec2  mix(u64vec2,  u64vec2,  bvec2);"
            "u64vec3  mix(u64vec3,  u64vec3,  bvec3);"
            "u64vec4  mix(u64vec4,  u64vec4,  bvec4);"

            "int64_t doubleBitsToInt64(float64_t);"
            "i64vec2 doubleBitsToInt64(f64vec2);"
            "i64vec3 doubleBitsToInt64(f64vec3);"
            "i64vec4 doubleBitsToInt64(f64vec4);"

            "uint64_t doubleBitsToUint64(float64_t);"
            "u64vec2  doubleBitsToUint64(f64vec2);"
            "u64vec3  doubleBitsToUint64(f64vec3);"
            "u64vec4  doubleBitsToUint64(f64vec4);"

            "float64_t int64BitsToDouble(int64_t);"
            "f64vec2  int64BitsToDouble(i64vec2);"
            "f64vec3  int64BitsToDouble(i64vec3);"
            "f64vec4  int64BitsToDouble(i64vec4);"

            "float64_t uint64BitsToDouble(uint64_t);"
            "f64vec2  uint64BitsToDouble(u64vec2);"
            "f64vec3  uint64BitsToDouble(u64vec3);"
            "f64vec4  uint64BitsToDouble(u64vec4);"

            "int64_t  packInt2x32(ivec2);"
            "uint64_t packUint2x32(uvec2);"
            "ivec2    unpackInt2x32(int64_t);"
            "uvec2    unpackUint2x32(uint64_t);"

            "bvec2 lessThan(i64vec2, i64vec2);"
            "bvec3 lessThan(i64vec3, i64vec3);"
            "bvec4 lessThan(i64vec4, i64vec4);"
            "bvec2 lessThan(u64vec2, u64vec2);"
            "bvec3 lessThan(u64vec3, u64vec3);"
            "bvec4 lessThan(u64vec4, u64vec4);"

            "bvec2 lessThanEqual(i64vec2, i64vec2);"
            "bvec3 lessThanEqual(i64vec3, i64vec3);"
            "bvec4 lessThanEqual(i64vec4, i64vec4);"
            "bvec2 lessThanEqual(u64vec2, u64vec2);"
            "bvec3 lessThanEqual(u64vec3, u64vec3);"
            "bvec4 lessThanEqual(u64vec4, u64vec4);"

            "bvec2 greaterThan(i64vec2, i64vec2);"
            "bvec3 greaterThan(i64vec3, i64vec3);"
            "bvec4 greaterThan(i64vec4, i64vec4);"
            "bvec2 greaterThan(u64vec2, u64vec2);"
            "bvec3 greaterThan(u64vec3, u64vec3);"
            "bvec4 greaterThan(u64vec4, u64vec4);"

            "bvec2 greaterThanEqual(i64vec2, i64vec2);"
            "bvec3 greaterThanEqual(i64vec3, i64vec3);"
            "bvec4 greaterThanEqual(i64vec4, i64vec4);"
            "bvec2 greaterThanEqual(u64vec2, u64vec2);"
            "bvec3 greaterThanEqual(u64vec3, u64vec3);"
            "bvec4 greaterThanEqual(u64vec4, u64vec4);"

            "bvec2 equal(i64vec2, i64vec2);"
            "bvec3 equal(i64vec3, i64vec3);"
            "bvec4 equal(i64vec4, i64vec4);"
            "bvec2 equal(u64vec2, u64vec2);"
            "bvec3 equal(u64vec3, u64vec3);"
            "bvec4 equal(u64vec4, u64vec4);"

            "bvec2 notEqual(i64vec2, i64vec2);"
            "bvec3 notEqual(i64vec3, i64vec3);"
            "bvec4 notEqual(i64vec4, i64vec4);"
            "bvec2 notEqual(u64vec2, u64vec2);"
            "bvec3 notEqual(u64vec3, u64vec3);"
            "bvec4 notEqual(u64vec4, u64vec4);"

            "int64_t bitCount(int64_t);"
            "i64vec2 bitCount(i64vec2);"
            "i64vec3 bitCount(i64vec3);"
            "i64vec4 bitCount(i64vec4);"

            "int64_t bitCount(uint64_t);"
            "i64vec2 bitCount(u64vec2);"
            "i64vec3 bitCount(u64vec3);"
            "i64vec4 bitCount(u64vec4);"

            "int64_t findLSB(int64_t);"
            "i64vec2 findLSB(i64vec2);"
            "i64vec3 findLSB(i64vec3);"
            "i64vec4 findLSB(i64vec4);"

            "int64_t findLSB(uint64_t);"
            "i64vec2 findLSB(u64vec2);"
            "i64vec3 findLSB(u64vec3);"
            "i64vec4 findLSB(u64vec4);"

            "int64_t findMSB(int64_t);"
            "i64vec2 findMSB(i64vec2);"
            "i64vec3 findMSB(i64vec3);"
            "i64vec4 findMSB(i64vec4);"

            "int64_t findMSB(uint64_t);"
            "i64vec2 findMSB(u64vec2);"
            "i64vec3 findMSB(u64vec3);"
            "i64vec4 findMSB(u64vec4);"

            "\n"
        );
    }

    // GL_AMD_shader_trinary_minmax
    if (profile != EEsProfile && version >= 430) {
        commonBuiltins.append(
            "float min3(float, float, float);"
            "vec2  min3(vec2,  vec2,  vec2);"
            "vec3  min3(vec3,  vec3,  vec3);"
            "vec4  min3(vec4,  vec4,  vec4);"

            "int   min3(int,   int,   int);"
            "ivec2 min3(ivec2, ivec2, ivec2);"
            "ivec3 min3(ivec3, ivec3, ivec3);"
            "ivec4 min3(ivec4, ivec4, ivec4);"

            "uint  min3(uint,  uint,  uint);"
            "uvec2 min3(uvec2, uvec2, uvec2);"
            "uvec3 min3(uvec3, uvec3, uvec3);"
            "uvec4 min3(uvec4, uvec4, uvec4);"

            "float max3(float, float, float);"
            "vec2  max3(vec2,  vec2,  vec2);"
            "vec3  max3(vec3,  vec3,  vec3);"
            "vec4  max3(vec4,  vec4,  vec4);"

            "int   max3(int,   int,   int);"
            "ivec2 max3(ivec2, ivec2, ivec2);"
            "ivec3 max3(ivec3, ivec3, ivec3);"
            "ivec4 max3(ivec4, ivec4, ivec4);"

            "uint  max3(uint,  uint,  uint);"
            "uvec2 max3(uvec2, uvec2, uvec2);"
            "uvec3 max3(uvec3, uvec3, uvec3);"
            "uvec4 max3(uvec4, uvec4, uvec4);"

            "float mid3(float, float, float);"
            "vec2  mid3(vec2,  vec2,  vec2);"
            "vec3  mid3(vec3,  vec3,  vec3);"
            "vec4  mid3(vec4,  vec4,  vec4);"

            "int   mid3(int,   int,   int);"
            "ivec2 mid3(ivec2, ivec2, ivec2);"
            "ivec3 mid3(ivec3, ivec3, ivec3);"
            "ivec4 mid3(ivec4, ivec4, ivec4);"

            "uint  mid3(uint,  uint,  uint);"
            "uvec2 mid3(uvec2, uvec2, uvec2);"
            "uvec3 mid3(uvec3, uvec3, uvec3);"
            "uvec4 mid3(uvec4, uvec4, uvec4);"

            "float16_t min3(float16_t, float16_t, float16_t);"
            "f16vec2   min3(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   min3(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   min3(f16vec4,   f16vec4,   f16vec4);"

            "float16_t max3(float16_t, float16_t, float16_t);"
            "f16vec2   max3(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   max3(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   max3(f16vec4,   f16vec4,   f16vec4);"

            "float16_t mid3(float16_t, float16_t, float16_t);"
            "f16vec2   mid3(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   mid3(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   mid3(f16vec4,   f16vec4,   f16vec4);"

            "int16_t   min3(int16_t,   int16_t,   int16_t);"
            "i16vec2   min3(i16vec2,   i16vec2,   i16vec2);"
            "i16vec3   min3(i16vec3,   i16vec3,   i16vec3);"
            "i16vec4   min3(i16vec4,   i16vec4,   i16vec4);"

            "int16_t   max3(int16_t,   int16_t,   int16_t);"
            "i16vec2   max3(i16vec2,   i16vec2,   i16vec2);"
            "i16vec3   max3(i16vec3,   i16vec3,   i16vec3);"
            "i16vec4   max3(i16vec4,   i16vec4,   i16vec4);"

            "int16_t   mid3(int16_t,   int16_t,   int16_t);"
            "i16vec2   mid3(i16vec2,   i16vec2,   i16vec2);"
            "i16vec3   mid3(i16vec3,   i16vec3,   i16vec3);"
            "i16vec4   mid3(i16vec4,   i16vec4,   i16vec4);"

            "uint16_t  min3(uint16_t,  uint16_t,  uint16_t);"
            "u16vec2   min3(u16vec2,   u16vec2,   u16vec2);"
            "u16vec3   min3(u16vec3,   u16vec3,   u16vec3);"
            "u16vec4   min3(u16vec4,   u16vec4,   u16vec4);"

            "uint16_t  max3(uint16_t,  uint16_t,  uint16_t);"
            "u16vec2   max3(u16vec2,   u16vec2,   u16vec2);"
            "u16vec3   max3(u16vec3,   u16vec3,   u16vec3);"
            "u16vec4   max3(u16vec4,   u16vec4,   u16vec4);"

            "uint16_t  mid3(uint16_t,  uint16_t,  uint16_t);"
            "u16vec2   mid3(u16vec2,   u16vec2,   u16vec2);"
            "u16vec3   mid3(u16vec3,   u16vec3,   u16vec3);"
            "u16vec4   mid3(u16vec4,   u16vec4,   u16vec4);"

            "\n"
        );
    }

    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 430)) {
        commonBuiltins.append(
            "uint atomicAdd(coherent volatile inout uint, uint, int, int, int);"
            " int atomicAdd(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicMin(coherent volatile inout uint, uint, int, int, int);"
            " int atomicMin(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicMax(coherent volatile inout uint, uint, int, int, int);"
            " int atomicMax(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicAnd(coherent volatile inout uint, uint, int, int, int);"
            " int atomicAnd(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicOr (coherent volatile inout uint, uint, int, int, int);"
            " int atomicOr (coherent volatile inout  int,  int, int, int, int);"

            "uint atomicXor(coherent volatile inout uint, uint, int, int, int);"
            " int atomicXor(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicExchange(coherent volatile inout uint, uint, int, int, int);"
            " int atomicExchange(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicCompSwap(coherent volatile inout uint, uint, uint, int, int, int, int, int);"
            " int atomicCompSwap(coherent volatile inout  int,  int,  int, int, int, int, int, int);"

            "uint atomicLoad(coherent volatile in uint, int, int, int);"
            " int atomicLoad(coherent volatile in  int, int, int, int);"

            "void atomicStore(coherent volatile out uint, uint, int, int, int);"
            "void atomicStore(coherent volatile out  int,  int, int, int, int);"

            "\n");
    }

    if (profile != EEsProfile && version >= 440) {
        commonBuiltins.append(
            "uint64_t atomicMin(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicMin(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicMin(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicMin(coherent volatile inout  int64_t,  int64_t, int, int, int);"
            "float16_t atomicMin(coherent volatile inout float16_t, float16_t);"
            "float16_t atomicMin(coherent volatile inout float16_t, float16_t, int, int, int);"
            "   float atomicMin(coherent volatile inout float, float);"
            "   float atomicMin(coherent volatile inout float, float, int, int, int);"
            "  double atomicMin(coherent volatile inout double, double);"
            "  double atomicMin(coherent volatile inout double, double, int, int, int);"

            "uint64_t atomicMax(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicMax(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicMax(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicMax(coherent volatile inout  int64_t,  int64_t, int, int, int);"
            "float16_t atomicMax(coherent volatile inout float16_t, float16_t);"
            "float16_t atomicMax(coherent volatile inout float16_t, float16_t, int, int, int);"
            "   float atomicMax(coherent volatile inout float, float);"
            "   float atomicMax(coherent volatile inout float, float, int, int, int);"
            "  double atomicMax(coherent volatile inout double, double);"
            "  double atomicMax(coherent volatile inout double, double, int, int, int);"

            "uint64_t atomicAnd(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicAnd(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicAnd(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicAnd(coherent volatile inout  int64_t,  int64_t, int, int, int);"

            "uint64_t atomicOr (coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicOr (coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicOr (coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicOr (coherent volatile inout  int64_t,  int64_t, int, int, int);"

            "uint64_t atomicXor(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicXor(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicXor(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicXor(coherent volatile inout  int64_t,  int64_t, int, int, int);"

            "uint64_t atomicAdd(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicAdd(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicAdd(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicAdd(coherent volatile inout  int64_t,  int64_t, int, int, int);"
            "float16_t atomicAdd(coherent volatile inout float16_t, float16_t);"
            "float16_t atomicAdd(coherent volatile inout float16_t, float16_t, int, int, int);"
            "   float atomicAdd(coherent volatile inout float, float);"
            "   float atomicAdd(coherent volatile inout float, float, int, int, int);"
            "  double atomicAdd(coherent volatile inout double, double);"
            "  double atomicAdd(coherent volatile inout double, double, int, int, int);"

            "uint64_t atomicExchange(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicExchange(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicExchange(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicExchange(coherent volatile inout  int64_t,  int64_t, int, int, int);"
            "float16_t atomicExchange(coherent volatile inout float16_t, float16_t);"
            "float16_t atomicExchange(coherent volatile inout float16_t, float16_t, int, int, int);"
            "   float atomicExchange(coherent volatile inout float, float);"
            "   float atomicExchange(coherent volatile inout float, float, int, int, int);"
            "  double atomicExchange(coherent volatile inout double, double);"
            "  double atomicExchange(coherent volatile inout double, double, int, int, int);"

            "uint64_t atomicCompSwap(coherent volatile inout uint64_t, uint64_t, uint64_t);"
            " int64_t atomicCompSwap(coherent volatile inout  int64_t,  int64_t,  int64_t);"
            "uint64_t atomicCompSwap(coherent volatile inout uint64_t, uint64_t, uint64_t, int, int, int, int, int);"
            " int64_t atomicCompSwap(coherent volatile inout  int64_t,  int64_t,  int64_t, int, int, int, int, int);"

            "uint64_t atomicLoad(coherent volatile in uint64_t, int, int, int);"
            " int64_t atomicLoad(coherent volatile in  int64_t, int, int, int);"
            "float16_t atomicLoad(coherent volatile in float16_t, int, int, int);"
            "   float atomicLoad(coherent volatile in float, int, int, int);"
            "  double atomicLoad(coherent volatile in double, int, int, int);"

            "void atomicStore(coherent volatile out uint64_t, uint64_t, int, int, int);"
            "void atomicStore(coherent volatile out  int64_t,  int64_t, int, int, int);"
            "void atomicStore(coherent volatile out float16_t, float16_t, int, int, int);"
            "void atomicStore(coherent volatile out float, float, int, int, int);"
            "void atomicStore(coherent volatile out double, double, int, int, int);"
            "\n");
    }

    // NV_shader_atomic_fp16_vector
    if (profile != EEsProfile && version >= 430) {
        commonBuiltins.append(
            "f16vec2 atomicAdd(coherent volatile inout f16vec2, f16vec2);"
            "f16vec4 atomicAdd(coherent volatile inout f16vec4, f16vec4);"
            "f16vec2 atomicMin(coherent volatile inout f16vec2, f16vec2);"
            "f16vec4 atomicMin(coherent volatile inout f16vec4, f16vec4);"
            "f16vec2 atomicMax(coherent volatile inout f16vec2, f16vec2);"
            "f16vec4 atomicMax(coherent volatile inout f16vec4, f16vec4);"
            "f16vec2 atomicExchange(coherent volatile inout f16vec2, f16vec2);"
            "f16vec4 atomicExchange(coherent volatile inout f16vec4, f16vec4);"
            "\n");
    }

    if ((profile == EEsProfile && version >= 300) ||
        (profile != EEsProfile && version >= 150)) { // GL_ARB_shader_bit_encoding
        commonBuiltins.append(
            "int   floatBitsToInt(highp float value);"
            "ivec2 floatBitsToInt(highp vec2  value);"
            "ivec3 floatBitsToInt(highp vec3  value);"
            "ivec4 floatBitsToInt(highp vec4  value);"

            "uint  floatBitsToUint(highp float value);"
            "uvec2 floatBitsToUint(highp vec2  value);"
            "uvec3 floatBitsToUint(highp vec3  value);"
            "uvec4 floatBitsToUint(highp vec4  value);"

            "float intBitsToFloat(highp int   value);"
            "vec2  intBitsToFloat(highp ivec2 value);"
            "vec3  intBitsToFloat(highp ivec3 value);"
            "vec4  intBitsToFloat(highp ivec4 value);"

            "float uintBitsToFloat(highp uint  value);"
            "vec2  uintBitsToFloat(highp uvec2 value);"
            "vec3  uintBitsToFloat(highp uvec3 value);"
            "vec4  uintBitsToFloat(highp uvec4 value);"

            "\n");
    }

    if ((profile != EEsProfile && version >= 400) ||
        (profile == EEsProfile && version >= 310)) {    // GL_OES_gpu_shader5

        commonBuiltins.append(
            "float  fma(float,  float,  float );"
            "vec2   fma(vec2,   vec2,   vec2  );"
            "vec3   fma(vec3,   vec3,   vec3  );"
            "vec4   fma(vec4,   vec4,   vec4  );"
            "\n");
    }

    if (profile != EEsProfile && version >= 150) {  // ARB_gpu_shader_fp64
            commonBuiltins.append(
                "double fma(double, double, double);"
                "dvec2  fma(dvec2,  dvec2,  dvec2 );"
                "dvec3  fma(dvec3,  dvec3,  dvec3 );"
                "dvec4  fma(dvec4,  dvec4,  dvec4 );"
                "\n");
    }

    if (profile == EEsProfile && version >= 310) {  // ARB_gpu_shader_fp64
            commonBuiltins.append(
                "float64_t fma(float64_t, float64_t, float64_t);"
                "f64vec2  fma(f64vec2,  f64vec2,  f64vec2 );"
                "f64vec3  fma(f64vec3,  f64vec3,  f64vec3 );"
                "f64vec4  fma(f64vec4,  f64vec4,  f64vec4 );"
                "\n");
    }

    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 400)) {
        commonBuiltins.append(
            "float frexp(highp float, out highp int);"
            "vec2  frexp(highp vec2,  out highp ivec2);"
            "vec3  frexp(highp vec3,  out highp ivec3);"
            "vec4  frexp(highp vec4,  out highp ivec4);"

            "float ldexp(highp float, highp int);"
            "vec2  ldexp(highp vec2,  highp ivec2);"
            "vec3  ldexp(highp vec3,  highp ivec3);"
            "vec4  ldexp(highp vec4,  highp ivec4);"

            "\n");
    }

    if (profile != EEsProfile && version >= 150) { // ARB_gpu_shader_fp64
        commonBuiltins.append(
            "double frexp(double, out int);"
            "dvec2  frexp( dvec2, out ivec2);"
            "dvec3  frexp( dvec3, out ivec3);"
            "dvec4  frexp( dvec4, out ivec4);"

            "double ldexp(double, int);"
            "dvec2  ldexp( dvec2, ivec2);"
            "dvec3  ldexp( dvec3, ivec3);"
            "dvec4  ldexp( dvec4, ivec4);"

            "double packDouble2x32(uvec2);"
            "uvec2 unpackDouble2x32(double);"

            "\n");
    }

    if (profile == EEsProfile && version >= 310) { // ARB_gpu_shader_fp64
        commonBuiltins.append(
            "float64_t frexp(float64_t, out int);"
            "f64vec2  frexp( f64vec2, out ivec2);"
            "f64vec3  frexp( f64vec3, out ivec3);"
            "f64vec4  frexp( f64vec4, out ivec4);"

            "float64_t ldexp(float64_t, int);"
            "f64vec2  ldexp( f64vec2, ivec2);"
            "f64vec3  ldexp( f64vec3, ivec3);"
            "f64vec4  ldexp( f64vec4, ivec4);"

            "\n");
    }

    if ((profile == EEsProfile && version >= 300) ||
        (profile != EEsProfile && version >= 150)) {
        commonBuiltins.append(
            "highp uint packUnorm2x16(vec2);"
                  "vec2 unpackUnorm2x16(highp uint);"
            "\n");
    }

    if ((profile == EEsProfile && version >= 300) ||
        (profile != EEsProfile && version >= 150)) {
        commonBuiltins.append(
            "highp uint packSnorm2x16(vec2);"
            "      vec2 unpackSnorm2x16(highp uint);"
            "highp uint packHalf2x16(vec2);"
            "\n");
    }

    if (profile == EEsProfile && version >= 300) {
        commonBuiltins.append(
            "mediump vec2 unpackHalf2x16(highp uint);"
            "\n");
    } else if (profile != EEsProfile && version >= 150) {
        commonBuiltins.append(
            "        vec2 unpackHalf2x16(highp uint);"
            "\n");
    }

    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 150)) {
        commonBuiltins.append(
            "highp uint packSnorm4x8(vec4);"
            "highp uint packUnorm4x8(vec4);"
            "\n");
    }

    if (profile == EEsProfile && version >= 310) {
        commonBuiltins.append(
            "mediump vec4 unpackSnorm4x8(highp uint);"
            "mediump vec4 unpackUnorm4x8(highp uint);"
            "\n");
    } else if (profile != EEsProfile && version >= 150) {
        commonBuiltins.append(
                    "vec4 unpackSnorm4x8(highp uint);"
                    "vec4 unpackUnorm4x8(highp uint);"
            "\n");
    }

    //
    // Matrix Functions.
    //
    commonBuiltins.append(
        "mat2 matrixCompMult(mat2 x, mat2 y);"
        "mat3 matrixCompMult(mat3 x, mat3 y);"
        "mat4 matrixCompMult(mat4 x, mat4 y);"

        "\n");

    // 120 is correct for both ES and desktop
    if (version >= 120) {
        commonBuiltins.append(
            "mat2   outerProduct(vec2 c, vec2 r);"
            "mat3   outerProduct(vec3 c, vec3 r);"
            "mat4   outerProduct(vec4 c, vec4 r);"
            "mat2x3 outerProduct(vec3 c, vec2 r);"
            "mat3x2 outerProduct(vec2 c, vec3 r);"
            "mat2x4 outerProduct(vec4 c, vec2 r);"
            "mat4x2 outerProduct(vec2 c, vec4 r);"
            "mat3x4 outerProduct(vec4 c, vec3 r);"
            "mat4x3 outerProduct(vec3 c, vec4 r);"

            "mat2   transpose(mat2   m);"
            "mat3   transpose(mat3   m);"
            "mat4   transpose(mat4   m);"
            "mat2x3 transpose(mat3x2 m);"
            "mat3x2 transpose(mat2x3 m);"
            "mat2x4 transpose(mat4x2 m);"
            "mat4x2 transpose(mat2x4 m);"
            "mat3x4 transpose(mat4x3 m);"
            "mat4x3 transpose(mat3x4 m);"

            "mat2x3 matrixCompMult(mat2x3, mat2x3);"
            "mat2x4 matrixCompMult(mat2x4, mat2x4);"
            "mat3x2 matrixCompMult(mat3x2, mat3x2);"
            "mat3x4 matrixCompMult(mat3x4, mat3x4);"
            "mat4x2 matrixCompMult(mat4x2, mat4x2);"
            "mat4x3 matrixCompMult(mat4x3, mat4x3);"

            "\n");

        // 150 is correct for both ES and desktop
        if (version >= 150) {
            commonBuiltins.append(
                "float determinant(mat2 m);"
                "float determinant(mat3 m);"
                "float determinant(mat4 m);"

                "mat2 inverse(mat2 m);"
                "mat3 inverse(mat3 m);"
                "mat4 inverse(mat4 m);"

                "\n");
        }
    }

    //
    // Original-style texture functions existing in all stages.
    // (Per-stage functions below.)
    //
    if ((profile == EEsProfile && version == 100) ||
         profile == ECompatibilityProfile ||
        (profile == ECoreProfile && version < 420) ||
         profile == ENoProfile) {
        if (spvVersion.spv == 0) {
            commonBuiltins.append(
                "vec4 texture2D(sampler2D, vec2);"

                "vec4 texture2DProj(sampler2D, vec3);"
                "vec4 texture2DProj(sampler2D, vec4);"

                "vec4 texture3D(sampler3D, vec3);"     // OES_texture_3D, but caught by keyword check
                "vec4 texture3DProj(sampler3D, vec4);" // OES_texture_3D, but caught by keyword check

                "vec4 textureCube(samplerCube, vec3);"

                "\n");
        }
    }

    if ( profile == ECompatibilityProfile ||
        (profile == ECoreProfile && version < 420) ||
         profile == ENoProfile) {
        if (spvVersion.spv == 0) {
            commonBuiltins.append(
                "vec4 texture1D(sampler1D, float);"

                "vec4 texture1DProj(sampler1D, vec2);"
                "vec4 texture1DProj(sampler1D, vec4);"

                "vec4 shadow1D(sampler1DShadow, vec3);"
                "vec4 shadow2D(sampler2DShadow, vec3);"
                "vec4 shadow1DProj(sampler1DShadow, vec4);"
                "vec4 shadow2DProj(sampler2DShadow, vec4);"

                "vec4 texture2DRect(sampler2DRect, vec2);"          // GL_ARB_texture_rectangle, caught by keyword check
                "vec4 texture2DRectProj(sampler2DRect, vec3);"      // GL_ARB_texture_rectangle, caught by keyword check
                "vec4 texture2DRectProj(sampler2DRect, vec4);"      // GL_ARB_texture_rectangle, caught by keyword check
                "vec4 shadow2DRect(sampler2DRectShadow, vec3);"     // GL_ARB_texture_rectangle, caught by keyword check
                "vec4 shadow2DRectProj(sampler2DRectShadow, vec4);" // GL_ARB_texture_rectangle, caught by keyword check

                "vec4 texture1DArray(sampler1DArray, vec2);"      // GL_EXT_texture_array
                "vec4 texture2DArray(sampler2DArray, vec3);"      // GL_EXT_texture_array
                "vec4 shadow1DArray(sampler1DArrayShadow, vec3);" // GL_EXT_texture_array
                "vec4 shadow2DArray(sampler2DArrayShadow, vec4);" // GL_EXT_texture_array
                "vec4 texture1DArray(sampler1DArray, vec2, float);"                // GL_EXT_texture_array
                "vec4 texture2DArray(sampler2DArray, vec3, float);"                // GL_EXT_texture_array
                "vec4 shadow1DArray(sampler1DArrayShadow, vec3, float);"           // GL_EXT_texture_array
                "vec4 texture1DArrayLod(sampler1DArray, vec2, float);"      // GL_EXT_texture_array
                "vec4 texture2DArrayLod(sampler2DArray, vec3, float);"      // GL_EXT_texture_array
                "vec4 shadow1DArrayLod(sampler1DArrayShadow, vec3, float);" // GL_EXT_texture_array
                "\n");
        }
    }

    if (profile == EEsProfile) {
        if (spvVersion.spv == 0) {
            if (version < 300) {
                commonBuiltins.append(
                    "vec4 texture2D(samplerExternalOES, vec2 coord);" // GL_OES_EGL_image_external
                    "vec4 texture2DProj(samplerExternalOES, vec3);"   // GL_OES_EGL_image_external
                    "vec4 texture2DProj(samplerExternalOES, vec4);"   // GL_OES_EGL_image_external
                "\n");
            } else {
                commonBuiltins.append(
                    "highp ivec2 textureSize(samplerExternalOES, int lod);"   // GL_OES_EGL_image_external_essl3
                    "vec4 texture(samplerExternalOES, vec2);"                 // GL_OES_EGL_image_external_essl3
                    "vec4 texture(samplerExternalOES, vec2, float bias);"     // GL_OES_EGL_image_external_essl3
                    "vec4 textureProj(samplerExternalOES, vec3);"             // GL_OES_EGL_image_external_essl3
                    "vec4 textureProj(samplerExternalOES, vec3, float bias);" // GL_OES_EGL_image_external_essl3
                    "vec4 textureProj(samplerExternalOES, vec4);"             // GL_OES_EGL_image_external_essl3
                    "vec4 textureProj(samplerExternalOES, vec4, float bias);" // GL_OES_EGL_image_external_essl3
                    "vec4 texelFetch(samplerExternalOES, ivec2, int lod);"    // GL_OES_EGL_image_external_essl3
                "\n");
            }
            commonBuiltins.append(
                "highp ivec2 textureSize(__samplerExternal2DY2YEXT, int lod);" // GL_EXT_YUV_target
                "vec4 texture(__samplerExternal2DY2YEXT, vec2);"               // GL_EXT_YUV_target
                "vec4 texture(__samplerExternal2DY2YEXT, vec2, float bias);"   // GL_EXT_YUV_target
                "vec4 textureProj(__samplerExternal2DY2YEXT, vec3);"           // GL_EXT_YUV_target
                "vec4 textureProj(__samplerExternal2DY2YEXT, vec3, float bias);" // GL_EXT_YUV_target
                "vec4 textureProj(__samplerExternal2DY2YEXT, vec4);"           // GL_EXT_YUV_target
                "vec4 textureProj(__samplerExternal2DY2YEXT, vec4, float bias);" // GL_EXT_YUV_target
                "vec4 texelFetch(__samplerExternal2DY2YEXT sampler, ivec2, int lod);" // GL_EXT_YUV_target
                "\n");
            commonBuiltins.append(
                "vec4 texture2DGradEXT(sampler2D, vec2, vec2, vec2);"      // GL_EXT_shader_texture_lod
                "vec4 texture2DProjGradEXT(sampler2D, vec3, vec2, vec2);"  // GL_EXT_shader_texture_lod
                "vec4 texture2DProjGradEXT(sampler2D, vec4, vec2, vec2);"  // GL_EXT_shader_texture_lod
                "vec4 textureCubeGradEXT(samplerCube, vec3, vec3, vec3);"  // GL_EXT_shader_texture_lod

                "float shadow2DEXT(sampler2DShadow, vec3);"     // GL_EXT_shadow_samplers
                "float shadow2DProjEXT(sampler2DShadow, vec4);" // GL_EXT_shadow_samplers

                "\n");
        }
    }

    //
    // Noise functions.
    //
    if (spvVersion.spv == 0 && profile != EEsProfile) {
        commonBuiltins.append(
            "float noise1(float x);"
            "float noise1(vec2  x);"
            "float noise1(vec3  x);"
            "float noise1(vec4  x);"

            "vec2 noise2(float x);"
            "vec2 noise2(vec2  x);"
            "vec2 noise2(vec3  x);"
            "vec2 noise2(vec4  x);"

            "vec3 noise3(float x);"
            "vec3 noise3(vec2  x);"
            "vec3 noise3(vec3  x);"
            "vec3 noise3(vec4  x);"

            "vec4 noise4(float x);"
            "vec4 noise4(vec2  x);"
            "vec4 noise4(vec3  x);"
            "vec4 noise4(vec4  x);"

            "\n");
    }

    if (spvVersion.vulkan == 0) {
        //
        // Atomic counter functions.
        //
        if ((profile != EEsProfile && version >= 300) ||
            (profile == EEsProfile && version >= 310)) {
            commonBuiltins.append(
                "uint atomicCounterIncrement(atomic_uint);"
                "uint atomicCounterDecrement(atomic_uint);"
                "uint atomicCounter(atomic_uint);"

                "\n");
        }
        if (profile != EEsProfile && version == 450) {
            commonBuiltins.append(
                "uint atomicCounterAddARB(atomic_uint, uint);"
                "uint atomicCounterSubtractARB(atomic_uint, uint);"
                "uint atomicCounterMinARB(atomic_uint, uint);"
                "uint atomicCounterMaxARB(atomic_uint, uint);"
                "uint atomicCounterAndARB(atomic_uint, uint);"
                "uint atomicCounterOrARB(atomic_uint, uint);"
                "uint atomicCounterXorARB(atomic_uint, uint);"
                "uint atomicCounterExchangeARB(atomic_uint, uint);"
                "uint atomicCounterCompSwapARB(atomic_uint, uint, uint);"

                "\n");
        }


        if (profile != EEsProfile && version >= 460) {
            commonBuiltins.append(
                "uint atomicCounterAdd(atomic_uint, uint);"
                "uint atomicCounterSubtract(atomic_uint, uint);"
                "uint atomicCounterMin(atomic_uint, uint);"
                "uint atomicCounterMax(atomic_uint, uint);"
                "uint atomicCounterAnd(atomic_uint, uint);"
                "uint atomicCounterOr(atomic_uint, uint);"
                "uint atomicCounterXor(atomic_uint, uint);"
                "uint atomicCounterExchange(atomic_uint, uint);"
                "uint atomicCounterCompSwap(atomic_uint, uint, uint);"

                "\n");
        }
    }
    else if (spvVersion.vulkanRelaxed) {
        //
        // Atomic counter functions act as aliases to normal atomic functions.
        // replace definitions to take 'volatile coherent uint' instead of 'atomic_uint'
        // and map to equivalent non-counter atomic op
        //
        if ((profile != EEsProfile && version >= 300) ||
            (profile == EEsProfile && version >= 310)) {
            commonBuiltins.append(
                "uint atomicCounterIncrement(volatile coherent uint);"
                "uint atomicCounterDecrement(volatile coherent uint);"
                "uint atomicCounter(volatile coherent uint);"

                "\n");
        }
        if (profile != EEsProfile && version >= 460) {
            commonBuiltins.append(
                "uint atomicCounterAdd(volatile coherent uint, uint);"
                "uint atomicCounterSubtract(volatile coherent uint, uint);"
                "uint atomicCounterMin(volatile coherent uint, uint);"
                "uint atomicCounterMax(volatile coherent uint, uint);"
                "uint atomicCounterAnd(volatile coherent uint, uint);"
                "uint atomicCounterOr(volatile coherent uint, uint);"
                "uint atomicCounterXor(volatile coherent uint, uint);"
                "uint atomicCounterExchange(volatile coherent uint, uint);"
                "uint atomicCounterCompSwap(volatile coherent uint, uint, uint);"

                "\n");
        }
    }

    // Bitfield
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 400)) {
        commonBuiltins.append(
            "  int bitfieldExtract(  int, int, int);"
            "ivec2 bitfieldExtract(ivec2, int, int);"
            "ivec3 bitfieldExtract(ivec3, int, int);"
            "ivec4 bitfieldExtract(ivec4, int, int);"

            " uint bitfieldExtract( uint, int, int);"
            "uvec2 bitfieldExtract(uvec2, int, int);"
            "uvec3 bitfieldExtract(uvec3, int, int);"
            "uvec4 bitfieldExtract(uvec4, int, int);"

            "  int bitfieldInsert(  int base,   int, int, int);"
            "ivec2 bitfieldInsert(ivec2 base, ivec2, int, int);"
            "ivec3 bitfieldInsert(ivec3 base, ivec3, int, int);"
            "ivec4 bitfieldInsert(ivec4 base, ivec4, int, int);"

            " uint bitfieldInsert( uint base,  uint, int, int);"
            "uvec2 bitfieldInsert(uvec2 base, uvec2, int, int);"
            "uvec3 bitfieldInsert(uvec3 base, uvec3, int, int);"
            "uvec4 bitfieldInsert(uvec4 base, uvec4, int, int);"

            "\n");
    }

    if (profile != EEsProfile && version >= 400) {
        commonBuiltins.append(
            "  int findLSB(  int);"
            "ivec2 findLSB(ivec2);"
            "ivec3 findLSB(ivec3);"
            "ivec4 findLSB(ivec4);"

            "  int findLSB( uint);"
            "ivec2 findLSB(uvec2);"
            "ivec3 findLSB(uvec3);"
            "ivec4 findLSB(uvec4);"

            "\n");
    } else if (profile == EEsProfile && version >= 310) {
        commonBuiltins.append(
            "lowp   int findLSB(  int);"
            "lowp ivec2 findLSB(ivec2);"
            "lowp ivec3 findLSB(ivec3);"
            "lowp ivec4 findLSB(ivec4);"

            "lowp   int findLSB( uint);"
            "lowp ivec2 findLSB(uvec2);"
            "lowp ivec3 findLSB(uvec3);"
            "lowp ivec4 findLSB(uvec4);"

            "\n");
    }

    if (profile != EEsProfile && version >= 400) {
        commonBuiltins.append(
            "  int bitCount(  int);"
            "ivec2 bitCount(ivec2);"
            "ivec3 bitCount(ivec3);"
            "ivec4 bitCount(ivec4);"

            "  int bitCount( uint);"
            "ivec2 bitCount(uvec2);"
            "ivec3 bitCount(uvec3);"
            "ivec4 bitCount(uvec4);"

            "  int findMSB(highp   int);"
            "ivec2 findMSB(highp ivec2);"
            "ivec3 findMSB(highp ivec3);"
            "ivec4 findMSB(highp ivec4);"

            "  int findMSB(highp  uint);"
            "ivec2 findMSB(highp uvec2);"
            "ivec3 findMSB(highp uvec3);"
            "ivec4 findMSB(highp uvec4);"

            "\n");
    }

    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 400)) {
        commonBuiltins.append(
            " uint uaddCarry(highp  uint, highp  uint, out lowp  uint carry);"
            "uvec2 uaddCarry(highp uvec2, highp uvec2, out lowp uvec2 carry);"
            "uvec3 uaddCarry(highp uvec3, highp uvec3, out lowp uvec3 carry);"
            "uvec4 uaddCarry(highp uvec4, highp uvec4, out lowp uvec4 carry);"

            " uint usubBorrow(highp  uint, highp  uint, out lowp  uint borrow);"
            "uvec2 usubBorrow(highp uvec2, highp uvec2, out lowp uvec2 borrow);"
            "uvec3 usubBorrow(highp uvec3, highp uvec3, out lowp uvec3 borrow);"
            "uvec4 usubBorrow(highp uvec4, highp uvec4, out lowp uvec4 borrow);"

            "void umulExtended(highp  uint, highp  uint, out highp  uint, out highp  uint lsb);"
            "void umulExtended(highp uvec2, highp uvec2, out highp uvec2, out highp uvec2 lsb);"
            "void umulExtended(highp uvec3, highp uvec3, out highp uvec3, out highp uvec3 lsb);"
            "void umulExtended(highp uvec4, highp uvec4, out highp uvec4, out highp uvec4 lsb);"

            "void imulExtended(highp   int, highp   int, out highp   int, out highp   int lsb);"
            "void imulExtended(highp ivec2, highp ivec2, out highp ivec2, out highp ivec2 lsb);"
            "void imulExtended(highp ivec3, highp ivec3, out highp ivec3, out highp ivec3 lsb);"
            "void imulExtended(highp ivec4, highp ivec4, out highp ivec4, out highp ivec4 lsb);"

            "  int bitfieldReverse(highp   int);"
            "ivec2 bitfieldReverse(highp ivec2);"
            "ivec3 bitfieldReverse(highp ivec3);"
            "ivec4 bitfieldReverse(highp ivec4);"

            " uint bitfieldReverse(highp  uint);"
            "uvec2 bitfieldReverse(highp uvec2);"
            "uvec3 bitfieldReverse(highp uvec3);"
            "uvec4 bitfieldReverse(highp uvec4);"

            "\n");
    }

    if (profile == EEsProfile && version >= 310) {
        commonBuiltins.append(
            "lowp   int bitCount(  int);"
            "lowp ivec2 bitCount(ivec2);"
            "lowp ivec3 bitCount(ivec3);"
            "lowp ivec4 bitCount(ivec4);"

            "lowp   int bitCount( uint);"
            "lowp ivec2 bitCount(uvec2);"
            "lowp ivec3 bitCount(uvec3);"
            "lowp ivec4 bitCount(uvec4);"

            "lowp   int findMSB(highp   int);"
            "lowp ivec2 findMSB(highp ivec2);"
            "lowp ivec3 findMSB(highp ivec3);"
            "lowp ivec4 findMSB(highp ivec4);"

            "lowp   int findMSB(highp  uint);"
            "lowp ivec2 findMSB(highp uvec2);"
            "lowp ivec3 findMSB(highp uvec3);"
            "lowp ivec4 findMSB(highp uvec4);"

            "\n");
    }

    // GL_ARB_shader_ballot
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(
            "uint64_t ballotARB(bool);"

            "float readInvocationARB(float, uint);"
            "vec2  readInvocationARB(vec2,  uint);"
            "vec3  readInvocationARB(vec3,  uint);"
            "vec4  readInvocationARB(vec4,  uint);"

            "int   readInvocationARB(int,   uint);"
            "ivec2 readInvocationARB(ivec2, uint);"
            "ivec3 readInvocationARB(ivec3, uint);"
            "ivec4 readInvocationARB(ivec4, uint);"

            "uint  readInvocationARB(uint,  uint);"
            "uvec2 readInvocationARB(uvec2, uint);"
            "uvec3 readInvocationARB(uvec3, uint);"
            "uvec4 readInvocationARB(uvec4, uint);"

            "float readFirstInvocationARB(float);"
            "vec2  readFirstInvocationARB(vec2);"
            "vec3  readFirstInvocationARB(vec3);"
            "vec4  readFirstInvocationARB(vec4);"

            "int   readFirstInvocationARB(int);"
            "ivec2 readFirstInvocationARB(ivec2);"
            "ivec3 readFirstInvocationARB(ivec3);"
            "ivec4 readFirstInvocationARB(ivec4);"

            "uint  readFirstInvocationARB(uint);"
            "uvec2 readFirstInvocationARB(uvec2);"
            "uvec3 readFirstInvocationARB(uvec3);"
            "uvec4 readFirstInvocationARB(uvec4);"

            "\n");
    }

    // GL_ARB_shader_group_vote
    if (profile != EEsProfile && version >= 430) {
        commonBuiltins.append(
            "bool anyInvocationARB(bool);"
            "bool allInvocationsARB(bool);"
            "bool allInvocationsEqualARB(bool);"

            "\n");
    }

    // GL_KHR_shader_subgroup
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 140)) {
        commonBuiltins.append(
            "void subgroupBarrier();"
            "void subgroupMemoryBarrier();"
            "void subgroupMemoryBarrierBuffer();"
            "void subgroupMemoryBarrierImage();"
            "bool subgroupElect();"

            "bool   subgroupAll(bool);\n"
            "bool   subgroupAny(bool);\n"
            "uvec4  subgroupBallot(bool);\n"
            "bool   subgroupInverseBallot(uvec4);\n"
            "bool   subgroupBallotBitExtract(uvec4, uint);\n"
            "uint   subgroupBallotBitCount(uvec4);\n"
            "uint   subgroupBallotInclusiveBitCount(uvec4);\n"
            "uint   subgroupBallotExclusiveBitCount(uvec4);\n"
            "uint   subgroupBallotFindLSB(uvec4);\n"
            "uint   subgroupBallotFindMSB(uvec4);\n"
            );

        // Generate all flavors of subgroup ops.
        static const char *subgroupOps[] = 
        {
            "bool   subgroupAllEqual(%s);\n",
            "%s     subgroupBroadcast(%s, uint);\n",
            "%s     subgroupBroadcastFirst(%s);\n",
            "%s     subgroupShuffle(%s, uint);\n",
            "%s     subgroupShuffleXor(%s, uint);\n",
            "%s     subgroupShuffleUp(%s, uint delta);\n",
            "%s     subgroupShuffleDown(%s, uint delta);\n",
            "%s     subgroupRotate(%s, uint);\n",
            "%s     subgroupClusteredRotate(%s, uint, uint);\n",
            "%s     subgroupAdd(%s);\n",
            "%s     subgroupMul(%s);\n",
            "%s     subgroupMin(%s);\n",
            "%s     subgroupMax(%s);\n",
            "%s     subgroupAnd(%s);\n",
            "%s     subgroupOr(%s);\n",
            "%s     subgroupXor(%s);\n",
            "%s     subgroupInclusiveAdd(%s);\n",
            "%s     subgroupInclusiveMul(%s);\n",
            "%s     subgroupInclusiveMin(%s);\n",
            "%s     subgroupInclusiveMax(%s);\n",
            "%s     subgroupInclusiveAnd(%s);\n",
            "%s     subgroupInclusiveOr(%s);\n",
            "%s     subgroupInclusiveXor(%s);\n",
            "%s     subgroupExclusiveAdd(%s);\n",
            "%s     subgroupExclusiveMul(%s);\n",
            "%s     subgroupExclusiveMin(%s);\n",
            "%s     subgroupExclusiveMax(%s);\n",
            "%s     subgroupExclusiveAnd(%s);\n",
            "%s     subgroupExclusiveOr(%s);\n",
            "%s     subgroupExclusiveXor(%s);\n",
            "%s     subgroupClusteredAdd(%s, uint);\n",
            "%s     subgroupClusteredMul(%s, uint);\n",
            "%s     subgroupClusteredMin(%s, uint);\n",
            "%s     subgroupClusteredMax(%s, uint);\n",
            "%s     subgroupClusteredAnd(%s, uint);\n",
            "%s     subgroupClusteredOr(%s, uint);\n",
            "%s     subgroupClusteredXor(%s, uint);\n",
            "%s     subgroupQuadBroadcast(%s, uint);\n",
            "%s     subgroupQuadSwapHorizontal(%s);\n",
            "%s     subgroupQuadSwapVertical(%s);\n",
            "%s     subgroupQuadSwapDiagonal(%s);\n",
            "uvec4  subgroupPartitionNV(%s);\n",
            "%s     subgroupPartitionedAddNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedMulNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedMinNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedMaxNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedAndNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedOrNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedXorNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveAddNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveMulNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveMinNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveMaxNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveAndNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveOrNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveXorNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveAddNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveMulNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveMinNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveMaxNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveAndNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveOrNV(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveXorNV(%s, uvec4 ballot);\n",
        };

        static const char *floatTypes[] = { 
            "float", "vec2", "vec3", "vec4", 
            "float16_t", "f16vec2", "f16vec3", "f16vec4", 
        };
        static const char *doubleTypes[] = { 
            "double", "dvec2", "dvec3", "dvec4", 
        };
        static const char *intTypes[] = { 
            "int8_t", "i8vec2", "i8vec3", "i8vec4", 
            "int16_t", "i16vec2", "i16vec3", "i16vec4", 
            "int", "ivec2", "ivec3", "ivec4", 
            "int64_t", "i64vec2", "i64vec3", "i64vec4", 
            "uint8_t", "u8vec2", "u8vec3", "u8vec4", 
            "uint16_t", "u16vec2", "u16vec3", "u16vec4", 
            "uint", "uvec2", "uvec3", "uvec4", 
            "uint64_t", "u64vec2", "u64vec3", "u64vec4", 
        };
        static const char *boolTypes[] = { 
            "bool", "bvec2", "bvec3", "bvec4", 
        };

        for (size_t i = 0; i < sizeof(subgroupOps)/sizeof(subgroupOps[0]); ++i) {
            const char *op = subgroupOps[i];

            // Logical operations don't support float
            bool logicalOp = strstr(op, "Or") || strstr(op, "And") ||
                             (strstr(op, "Xor") && !strstr(op, "ShuffleXor"));
            // Math operations don't support bool
            bool mathOp = strstr(op, "Add") || strstr(op, "Mul") || strstr(op, "Min") || strstr(op, "Max");

            const int bufSize = 256;
            char buf[bufSize];

            if (!logicalOp) {
                for (size_t j = 0; j < sizeof(floatTypes)/sizeof(floatTypes[0]); ++j) {
                    snprintf(buf, bufSize, op, floatTypes[j], floatTypes[j]);
                    commonBuiltins.append(buf);
                }
                if (profile != EEsProfile && version >= 400) {
                    for (size_t j = 0; j < sizeof(doubleTypes)/sizeof(doubleTypes[0]); ++j) {
                        snprintf(buf, bufSize, op, doubleTypes[j], doubleTypes[j]);
                        commonBuiltins.append(buf);
                    }
                }
            }
            if (!mathOp) {
                for (size_t j = 0; j < sizeof(boolTypes)/sizeof(boolTypes[0]); ++j) {
                    snprintf(buf, bufSize, op, boolTypes[j], boolTypes[j]);
                    commonBuiltins.append(buf);
                }
            }
            for (size_t j = 0; j < sizeof(intTypes)/sizeof(intTypes[0]); ++j) {
                snprintf(buf, bufSize, op, intTypes[j], intTypes[j]);
                commonBuiltins.append(buf);
            }
        }

        stageBuiltins[EShLangCompute].append(
            "void subgroupMemoryBarrierShared();"

            "\n"
            );
        stageBuiltins[EShLangMesh].append(
            "void subgroupMemoryBarrierShared();"
            "\n"
            );
        stageBuiltins[EShLangTask].append(
            "void subgroupMemoryBarrierShared();"
            "\n"
            );
    }

    // GL_EXT_shader_quad_control
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 140)) {
        commonBuiltins.append(
            "bool subgroupQuadAll(bool);\n"
            "bool subgroupQuadAny(bool);\n"
            );
    }

    if (profile != EEsProfile && version >= 460) {
        commonBuiltins.append(
            "bool anyInvocation(bool);"
            "bool allInvocations(bool);"
            "bool allInvocationsEqual(bool);"

            "\n");
    }

    // GL_AMD_shader_ballot
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(
            "float minInvocationsAMD(float);"
            "vec2  minInvocationsAMD(vec2);"
            "vec3  minInvocationsAMD(vec3);"
            "vec4  minInvocationsAMD(vec4);"

            "int   minInvocationsAMD(int);"
            "ivec2 minInvocationsAMD(ivec2);"
            "ivec3 minInvocationsAMD(ivec3);"
            "ivec4 minInvocationsAMD(ivec4);"

            "uint  minInvocationsAMD(uint);"
            "uvec2 minInvocationsAMD(uvec2);"
            "uvec3 minInvocationsAMD(uvec3);"
            "uvec4 minInvocationsAMD(uvec4);"

            "double minInvocationsAMD(double);"
            "dvec2  minInvocationsAMD(dvec2);"
            "dvec3  minInvocationsAMD(dvec3);"
            "dvec4  minInvocationsAMD(dvec4);"

            "int64_t minInvocationsAMD(int64_t);"
            "i64vec2 minInvocationsAMD(i64vec2);"
            "i64vec3 minInvocationsAMD(i64vec3);"
            "i64vec4 minInvocationsAMD(i64vec4);"

            "uint64_t minInvocationsAMD(uint64_t);"
            "u64vec2  minInvocationsAMD(u64vec2);"
            "u64vec3  minInvocationsAMD(u64vec3);"
            "u64vec4  minInvocationsAMD(u64vec4);"

            "float16_t minInvocationsAMD(float16_t);"
            "f16vec2   minInvocationsAMD(f16vec2);"
            "f16vec3   minInvocationsAMD(f16vec3);"
            "f16vec4   minInvocationsAMD(f16vec4);"

            "int16_t minInvocationsAMD(int16_t);"
            "i16vec2 minInvocationsAMD(i16vec2);"
            "i16vec3 minInvocationsAMD(i16vec3);"
            "i16vec4 minInvocationsAMD(i16vec4);"

            "uint16_t minInvocationsAMD(uint16_t);"
            "u16vec2  minInvocationsAMD(u16vec2);"
            "u16vec3  minInvocationsAMD(u16vec3);"
            "u16vec4  minInvocationsAMD(u16vec4);"

            "float minInvocationsInclusiveScanAMD(float);"
            "vec2  minInvocationsInclusiveScanAMD(vec2);"
            "vec3  minInvocationsInclusiveScanAMD(vec3);"
            "vec4  minInvocationsInclusiveScanAMD(vec4);"

            "int   minInvocationsInclusiveScanAMD(int);"
            "ivec2 minInvocationsInclusiveScanAMD(ivec2);"
            "ivec3 minInvocationsInclusiveScanAMD(ivec3);"
            "ivec4 minInvocationsInclusiveScanAMD(ivec4);"

            "uint  minInvocationsInclusiveScanAMD(uint);"
            "uvec2 minInvocationsInclusiveScanAMD(uvec2);"
            "uvec3 minInvocationsInclusiveScanAMD(uvec3);"
            "uvec4 minInvocationsInclusiveScanAMD(uvec4);"

            "double minInvocationsInclusiveScanAMD(double);"
            "dvec2  minInvocationsInclusiveScanAMD(dvec2);"
            "dvec3  minInvocationsInclusiveScanAMD(dvec3);"
            "dvec4  minInvocationsInclusiveScanAMD(dvec4);"

            "int64_t minInvocationsInclusiveScanAMD(int64_t);"
            "i64vec2 minInvocationsInclusiveScanAMD(i64vec2);"
            "i64vec3 minInvocationsInclusiveScanAMD(i64vec3);"
            "i64vec4 minInvocationsInclusiveScanAMD(i64vec4);"

            "uint64_t minInvocationsInclusiveScanAMD(uint64_t);"
            "u64vec2  minInvocationsInclusiveScanAMD(u64vec2);"
            "u64vec3  minInvocationsInclusiveScanAMD(u64vec3);"
            "u64vec4  minInvocationsInclusiveScanAMD(u64vec4);"

            "float16_t minInvocationsInclusiveScanAMD(float16_t);"
            "f16vec2   minInvocationsInclusiveScanAMD(f16vec2);"
            "f16vec3   minInvocationsInclusiveScanAMD(f16vec3);"
            "f16vec4   minInvocationsInclusiveScanAMD(f16vec4);"

            "int16_t minInvocationsInclusiveScanAMD(int16_t);"
            "i16vec2 minInvocationsInclusiveScanAMD(i16vec2);"
            "i16vec3 minInvocationsInclusiveScanAMD(i16vec3);"
            "i16vec4 minInvocationsInclusiveScanAMD(i16vec4);"

            "uint16_t minInvocationsInclusiveScanAMD(uint16_t);"
            "u16vec2  minInvocationsInclusiveScanAMD(u16vec2);"
            "u16vec3  minInvocationsInclusiveScanAMD(u16vec3);"
            "u16vec4  minInvocationsInclusiveScanAMD(u16vec4);"

            "float minInvocationsExclusiveScanAMD(float);"
            "vec2  minInvocationsExclusiveScanAMD(vec2);"
            "vec3  minInvocationsExclusiveScanAMD(vec3);"
            "vec4  minInvocationsExclusiveScanAMD(vec4);"

            "int   minInvocationsExclusiveScanAMD(int);"
            "ivec2 minInvocationsExclusiveScanAMD(ivec2);"
            "ivec3 minInvocationsExclusiveScanAMD(ivec3);"
            "ivec4 minInvocationsExclusiveScanAMD(ivec4);"

            "uint  minInvocationsExclusiveScanAMD(uint);"
            "uvec2 minInvocationsExclusiveScanAMD(uvec2);"
            "uvec3 minInvocationsExclusiveScanAMD(uvec3);"
            "uvec4 minInvocationsExclusiveScanAMD(uvec4);"

            "double minInvocationsExclusiveScanAMD(double);"
            "dvec2  minInvocationsExclusiveScanAMD(dvec2);"
            "dvec3  minInvocationsExclusiveScanAMD(dvec3);"
            "dvec4  minInvocationsExclusiveScanAMD(dvec4);"

            "int64_t minInvocationsExclusiveScanAMD(int64_t);"
            "i64vec2 minInvocationsExclusiveScanAMD(i64vec2);"
            "i64vec3 minInvocationsExclusiveScanAMD(i64vec3);"
            "i64vec4 minInvocationsExclusiveScanAMD(i64vec4);"

            "uint64_t minInvocationsExclusiveScanAMD(uint64_t);"
            "u64vec2  minInvocationsExclusiveScanAMD(u64vec2);"
            "u64vec3  minInvocationsExclusiveScanAMD(u64vec3);"
            "u64vec4  minInvocationsExclusiveScanAMD(u64vec4);"

            "float16_t minInvocationsExclusiveScanAMD(float16_t);"
            "f16vec2   minInvocationsExclusiveScanAMD(f16vec2);"
            "f16vec3   minInvocationsExclusiveScanAMD(f16vec3);"
            "f16vec4   minInvocationsExclusiveScanAMD(f16vec4);"

            "int16_t minInvocationsExclusiveScanAMD(int16_t);"
            "i16vec2 minInvocationsExclusiveScanAMD(i16vec2);"
            "i16vec3 minInvocationsExclusiveScanAMD(i16vec3);"
            "i16vec4 minInvocationsExclusiveScanAMD(i16vec4);"

            "uint16_t minInvocationsExclusiveScanAMD(uint16_t);"
            "u16vec2  minInvocationsExclusiveScanAMD(u16vec2);"
            "u16vec3  minInvocationsExclusiveScanAMD(u16vec3);"
            "u16vec4  minInvocationsExclusiveScanAMD(u16vec4);"

            "float maxInvocationsAMD(float);"
            "vec2  maxInvocationsAMD(vec2);"
            "vec3  maxInvocationsAMD(vec3);"
            "vec4  maxInvocationsAMD(vec4);"

            "int   maxInvocationsAMD(int);"
            "ivec2 maxInvocationsAMD(ivec2);"
            "ivec3 maxInvocationsAMD(ivec3);"
            "ivec4 maxInvocationsAMD(ivec4);"

            "uint  maxInvocationsAMD(uint);"
            "uvec2 maxInvocationsAMD(uvec2);"
            "uvec3 maxInvocationsAMD(uvec3);"
            "uvec4 maxInvocationsAMD(uvec4);"

            "double maxInvocationsAMD(double);"
            "dvec2  maxInvocationsAMD(dvec2);"
            "dvec3  maxInvocationsAMD(dvec3);"
            "dvec4  maxInvocationsAMD(dvec4);"

            "int64_t maxInvocationsAMD(int64_t);"
            "i64vec2 maxInvocationsAMD(i64vec2);"
            "i64vec3 maxInvocationsAMD(i64vec3);"
            "i64vec4 maxInvocationsAMD(i64vec4);"

            "uint64_t maxInvocationsAMD(uint64_t);"
            "u64vec2  maxInvocationsAMD(u64vec2);"
            "u64vec3  maxInvocationsAMD(u64vec3);"
            "u64vec4  maxInvocationsAMD(u64vec4);"

            "float16_t maxInvocationsAMD(float16_t);"
            "f16vec2   maxInvocationsAMD(f16vec2);"
            "f16vec3   maxInvocationsAMD(f16vec3);"
            "f16vec4   maxInvocationsAMD(f16vec4);"

            "int16_t maxInvocationsAMD(int16_t);"
            "i16vec2 maxInvocationsAMD(i16vec2);"
            "i16vec3 maxInvocationsAMD(i16vec3);"
            "i16vec4 maxInvocationsAMD(i16vec4);"

            "uint16_t maxInvocationsAMD(uint16_t);"
            "u16vec2  maxInvocationsAMD(u16vec2);"
            "u16vec3  maxInvocationsAMD(u16vec3);"
            "u16vec4  maxInvocationsAMD(u16vec4);"

            "float maxInvocationsInclusiveScanAMD(float);"
            "vec2  maxInvocationsInclusiveScanAMD(vec2);"
            "vec3  maxInvocationsInclusiveScanAMD(vec3);"
            "vec4  maxInvocationsInclusiveScanAMD(vec4);"

            "int   maxInvocationsInclusiveScanAMD(int);"
            "ivec2 maxInvocationsInclusiveScanAMD(ivec2);"
            "ivec3 maxInvocationsInclusiveScanAMD(ivec3);"
            "ivec4 maxInvocationsInclusiveScanAMD(ivec4);"

            "uint  maxInvocationsInclusiveScanAMD(uint);"
            "uvec2 maxInvocationsInclusiveScanAMD(uvec2);"
            "uvec3 maxInvocationsInclusiveScanAMD(uvec3);"
            "uvec4 maxInvocationsInclusiveScanAMD(uvec4);"

            "double maxInvocationsInclusiveScanAMD(double);"
            "dvec2  maxInvocationsInclusiveScanAMD(dvec2);"
            "dvec3  maxInvocationsInclusiveScanAMD(dvec3);"
            "dvec4  maxInvocationsInclusiveScanAMD(dvec4);"

            "int64_t maxInvocationsInclusiveScanAMD(int64_t);"
            "i64vec2 maxInvocationsInclusiveScanAMD(i64vec2);"
            "i64vec3 maxInvocationsInclusiveScanAMD(i64vec3);"
            "i64vec4 maxInvocationsInclusiveScanAMD(i64vec4);"

            "uint64_t maxInvocationsInclusiveScanAMD(uint64_t);"
            "u64vec2  maxInvocationsInclusiveScanAMD(u64vec2);"
            "u64vec3  maxInvocationsInclusiveScanAMD(u64vec3);"
            "u64vec4  maxInvocationsInclusiveScanAMD(u64vec4);"

            "float16_t maxInvocationsInclusiveScanAMD(float16_t);"
            "f16vec2   maxInvocationsInclusiveScanAMD(f16vec2);"
            "f16vec3   maxInvocationsInclusiveScanAMD(f16vec3);"
            "f16vec4   maxInvocationsInclusiveScanAMD(f16vec4);"

            "int16_t maxInvocationsInclusiveScanAMD(int16_t);"
            "i16vec2 maxInvocationsInclusiveScanAMD(i16vec2);"
            "i16vec3 maxInvocationsInclusiveScanAMD(i16vec3);"
            "i16vec4 maxInvocationsInclusiveScanAMD(i16vec4);"

            "uint16_t maxInvocationsInclusiveScanAMD(uint16_t);"
            "u16vec2  maxInvocationsInclusiveScanAMD(u16vec2);"
            "u16vec3  maxInvocationsInclusiveScanAMD(u16vec3);"
            "u16vec4  maxInvocationsInclusiveScanAMD(u16vec4);"

            "float maxInvocationsExclusiveScanAMD(float);"
            "vec2  maxInvocationsExclusiveScanAMD(vec2);"
            "vec3  maxInvocationsExclusiveScanAMD(vec3);"
            "vec4  maxInvocationsExclusiveScanAMD(vec4);"

            "int   maxInvocationsExclusiveScanAMD(int);"
            "ivec2 maxInvocationsExclusiveScanAMD(ivec2);"
            "ivec3 maxInvocationsExclusiveScanAMD(ivec3);"
            "ivec4 maxInvocationsExclusiveScanAMD(ivec4);"

            "uint  maxInvocationsExclusiveScanAMD(uint);"
            "uvec2 maxInvocationsExclusiveScanAMD(uvec2);"
            "uvec3 maxInvocationsExclusiveScanAMD(uvec3);"
            "uvec4 maxInvocationsExclusiveScanAMD(uvec4);"

            "double maxInvocationsExclusiveScanAMD(double);"
            "dvec2  maxInvocationsExclusiveScanAMD(dvec2);"
            "dvec3  maxInvocationsExclusiveScanAMD(dvec3);"
            "dvec4  maxInvocationsExclusiveScanAMD(dvec4);"

            "int64_t maxInvocationsExclusiveScanAMD(int64_t);"
            "i64vec2 maxInvocationsExclusiveScanAMD(i64vec2);"
            "i64vec3 maxInvocationsExclusiveScanAMD(i64vec3);"
            "i64vec4 maxInvocationsExclusiveScanAMD(i64vec4);"

            "uint64_t maxInvocationsExclusiveScanAMD(uint64_t);"
            "u64vec2  maxInvocationsExclusiveScanAMD(u64vec2);"
            "u64vec3  maxInvocationsExclusiveScanAMD(u64vec3);"
            "u64vec4  maxInvocationsExclusiveScanAMD(u64vec4);"

            "float16_t maxInvocationsExclusiveScanAMD(float16_t);"
            "f16vec2   maxInvocationsExclusiveScanAMD(f16vec2);"
            "f16vec3   maxInvocationsExclusiveScanAMD(f16vec3);"
            "f16vec4   maxInvocationsExclusiveScanAMD(f16vec4);"

            "int16_t maxInvocationsExclusiveScanAMD(int16_t);"
            "i16vec2 maxInvocationsExclusiveScanAMD(i16vec2);"
            "i16vec3 maxInvocationsExclusiveScanAMD(i16vec3);"
            "i16vec4 maxInvocationsExclusiveScanAMD(i16vec4);"

            "uint16_t maxInvocationsExclusiveScanAMD(uint16_t);"
            "u16vec2  maxInvocationsExclusiveScanAMD(u16vec2);"
            "u16vec3  maxInvocationsExclusiveScanAMD(u16vec3);"
            "u16vec4  maxInvocationsExclusiveScanAMD(u16vec4);"

            "float addInvocationsAMD(float);"
            "vec2  addInvocationsAMD(vec2);"
            "vec3  addInvocationsAMD(vec3);"
            "vec4  addInvocationsAMD(vec4);"

            "int   addInvocationsAMD(int);"
            "ivec2 addInvocationsAMD(ivec2);"
            "ivec3 addInvocationsAMD(ivec3);"
            "ivec4 addInvocationsAMD(ivec4);"

            "uint  addInvocationsAMD(uint);"
            "uvec2 addInvocationsAMD(uvec2);"
            "uvec3 addInvocationsAMD(uvec3);"
            "uvec4 addInvocationsAMD(uvec4);"

            "double  addInvocationsAMD(double);"
            "dvec2   addInvocationsAMD(dvec2);"
            "dvec3   addInvocationsAMD(dvec3);"
            "dvec4   addInvocationsAMD(dvec4);"

            "int64_t addInvocationsAMD(int64_t);"
            "i64vec2 addInvocationsAMD(i64vec2);"
            "i64vec3 addInvocationsAMD(i64vec3);"
            "i64vec4 addInvocationsAMD(i64vec4);"

            "uint64_t addInvocationsAMD(uint64_t);"
            "u64vec2  addInvocationsAMD(u64vec2);"
            "u64vec3  addInvocationsAMD(u64vec3);"
            "u64vec4  addInvocationsAMD(u64vec4);"

            "float16_t addInvocationsAMD(float16_t);"
            "f16vec2   addInvocationsAMD(f16vec2);"
            "f16vec3   addInvocationsAMD(f16vec3);"
            "f16vec4   addInvocationsAMD(f16vec4);"

            "int16_t addInvocationsAMD(int16_t);"
            "i16vec2 addInvocationsAMD(i16vec2);"
            "i16vec3 addInvocationsAMD(i16vec3);"
            "i16vec4 addInvocationsAMD(i16vec4);"

            "uint16_t addInvocationsAMD(uint16_t);"
            "u16vec2  addInvocationsAMD(u16vec2);"
            "u16vec3  addInvocationsAMD(u16vec3);"
            "u16vec4  addInvocationsAMD(u16vec4);"

            "float addInvocationsInclusiveScanAMD(float);"
            "vec2  addInvocationsInclusiveScanAMD(vec2);"
            "vec3  addInvocationsInclusiveScanAMD(vec3);"
            "vec4  addInvocationsInclusiveScanAMD(vec4);"

            "int   addInvocationsInclusiveScanAMD(int);"
            "ivec2 addInvocationsInclusiveScanAMD(ivec2);"
            "ivec3 addInvocationsInclusiveScanAMD(ivec3);"
            "ivec4 addInvocationsInclusiveScanAMD(ivec4);"

            "uint  addInvocationsInclusiveScanAMD(uint);"
            "uvec2 addInvocationsInclusiveScanAMD(uvec2);"
            "uvec3 addInvocationsInclusiveScanAMD(uvec3);"
            "uvec4 addInvocationsInclusiveScanAMD(uvec4);"

            "double  addInvocationsInclusiveScanAMD(double);"
            "dvec2   addInvocationsInclusiveScanAMD(dvec2);"
            "dvec3   addInvocationsInclusiveScanAMD(dvec3);"
            "dvec4   addInvocationsInclusiveScanAMD(dvec4);"

            "int64_t addInvocationsInclusiveScanAMD(int64_t);"
            "i64vec2 addInvocationsInclusiveScanAMD(i64vec2);"
            "i64vec3 addInvocationsInclusiveScanAMD(i64vec3);"
            "i64vec4 addInvocationsInclusiveScanAMD(i64vec4);"

            "uint64_t addInvocationsInclusiveScanAMD(uint64_t);"
            "u64vec2  addInvocationsInclusiveScanAMD(u64vec2);"
            "u64vec3  addInvocationsInclusiveScanAMD(u64vec3);"
            "u64vec4  addInvocationsInclusiveScanAMD(u64vec4);"

            "float16_t addInvocationsInclusiveScanAMD(float16_t);"
            "f16vec2   addInvocationsInclusiveScanAMD(f16vec2);"
            "f16vec3   addInvocationsInclusiveScanAMD(f16vec3);"
            "f16vec4   addInvocationsInclusiveScanAMD(f16vec4);"

            "int16_t addInvocationsInclusiveScanAMD(int16_t);"
            "i16vec2 addInvocationsInclusiveScanAMD(i16vec2);"
            "i16vec3 addInvocationsInclusiveScanAMD(i16vec3);"
            "i16vec4 addInvocationsInclusiveScanAMD(i16vec4);"

            "uint16_t addInvocationsInclusiveScanAMD(uint16_t);"
            "u16vec2  addInvocationsInclusiveScanAMD(u16vec2);"
            "u16vec3  addInvocationsInclusiveScanAMD(u16vec3);"
            "u16vec4  addInvocationsInclusiveScanAMD(u16vec4);"

            "float addInvocationsExclusiveScanAMD(float);"
            "vec2  addInvocationsExclusiveScanAMD(vec2);"
            "vec3  addInvocationsExclusiveScanAMD(vec3);"
            "vec4  addInvocationsExclusiveScanAMD(vec4);"

            "int   addInvocationsExclusiveScanAMD(int);"
            "ivec2 addInvocationsExclusiveScanAMD(ivec2);"
            "ivec3 addInvocationsExclusiveScanAMD(ivec3);"
            "ivec4 addInvocationsExclusiveScanAMD(ivec4);"

            "uint  addInvocationsExclusiveScanAMD(uint);"
            "uvec2 addInvocationsExclusiveScanAMD(uvec2);"
            "uvec3 addInvocationsExclusiveScanAMD(uvec3);"
            "uvec4 addInvocationsExclusiveScanAMD(uvec4);"

            "double  addInvocationsExclusiveScanAMD(double);"
            "dvec2   addInvocationsExclusiveScanAMD(dvec2);"
            "dvec3   addInvocationsExclusiveScanAMD(dvec3);"
            "dvec4   addInvocationsExclusiveScanAMD(dvec4);"

            "int64_t addInvocationsExclusiveScanAMD(int64_t);"
            "i64vec2 addInvocationsExclusiveScanAMD(i64vec2);"
            "i64vec3 addInvocationsExclusiveScanAMD(i64vec3);"
            "i64vec4 addInvocationsExclusiveScanAMD(i64vec4);"

            "uint64_t addInvocationsExclusiveScanAMD(uint64_t);"
            "u64vec2  addInvocationsExclusiveScanAMD(u64vec2);"
            "u64vec3  addInvocationsExclusiveScanAMD(u64vec3);"
            "u64vec4  addInvocationsExclusiveScanAMD(u64vec4);"

            "float16_t addInvocationsExclusiveScanAMD(float16_t);"
            "f16vec2   addInvocationsExclusiveScanAMD(f16vec2);"
            "f16vec3   addInvocationsExclusiveScanAMD(f16vec3);"
            "f16vec4   addInvocationsExclusiveScanAMD(f16vec4);"

            "int16_t addInvocationsExclusiveScanAMD(int16_t);"
            "i16vec2 addInvocationsExclusiveScanAMD(i16vec2);"
            "i16vec3 addInvocationsExclusiveScanAMD(i16vec3);"
            "i16vec4 addInvocationsExclusiveScanAMD(i16vec4);"

            "uint16_t addInvocationsExclusiveScanAMD(uint16_t);"
            "u16vec2  addInvocationsExclusiveScanAMD(u16vec2);"
            "u16vec3  addInvocationsExclusiveScanAMD(u16vec3);"
            "u16vec4  addInvocationsExclusiveScanAMD(u16vec4);"

            "float minInvocationsNonUniformAMD(float);"
            "vec2  minInvocationsNonUniformAMD(vec2);"
            "vec3  minInvocationsNonUniformAMD(vec3);"
            "vec4  minInvocationsNonUniformAMD(vec4);"

            "int   minInvocationsNonUniformAMD(int);"
            "ivec2 minInvocationsNonUniformAMD(ivec2);"
            "ivec3 minInvocationsNonUniformAMD(ivec3);"
            "ivec4 minInvocationsNonUniformAMD(ivec4);"

            "uint  minInvocationsNonUniformAMD(uint);"
            "uvec2 minInvocationsNonUniformAMD(uvec2);"
            "uvec3 minInvocationsNonUniformAMD(uvec3);"
            "uvec4 minInvocationsNonUniformAMD(uvec4);"

            "double minInvocationsNonUniformAMD(double);"
            "dvec2  minInvocationsNonUniformAMD(dvec2);"
            "dvec3  minInvocationsNonUniformAMD(dvec3);"
            "dvec4  minInvocationsNonUniformAMD(dvec4);"

            "int64_t minInvocationsNonUniformAMD(int64_t);"
            "i64vec2 minInvocationsNonUniformAMD(i64vec2);"
            "i64vec3 minInvocationsNonUniformAMD(i64vec3);"
            "i64vec4 minInvocationsNonUniformAMD(i64vec4);"

            "uint64_t minInvocationsNonUniformAMD(uint64_t);"
            "u64vec2  minInvocationsNonUniformAMD(u64vec2);"
            "u64vec3  minInvocationsNonUniformAMD(u64vec3);"
            "u64vec4  minInvocationsNonUniformAMD(u64vec4);"

            "float16_t minInvocationsNonUniformAMD(float16_t);"
            "f16vec2   minInvocationsNonUniformAMD(f16vec2);"
            "f16vec3   minInvocationsNonUniformAMD(f16vec3);"
            "f16vec4   minInvocationsNonUniformAMD(f16vec4);"

            "int16_t minInvocationsNonUniformAMD(int16_t);"
            "i16vec2 minInvocationsNonUniformAMD(i16vec2);"
            "i16vec3 minInvocationsNonUniformAMD(i16vec3);"
            "i16vec4 minInvocationsNonUniformAMD(i16vec4);"

            "uint16_t minInvocationsNonUniformAMD(uint16_t);"
            "u16vec2  minInvocationsNonUniformAMD(u16vec2);"
            "u16vec3  minInvocationsNonUniformAMD(u16vec3);"
            "u16vec4  minInvocationsNonUniformAMD(u16vec4);"

            "float minInvocationsInclusiveScanNonUniformAMD(float);"
            "vec2  minInvocationsInclusiveScanNonUniformAMD(vec2);"
            "vec3  minInvocationsInclusiveScanNonUniformAMD(vec3);"
            "vec4  minInvocationsInclusiveScanNonUniformAMD(vec4);"

            "int   minInvocationsInclusiveScanNonUniformAMD(int);"
            "ivec2 minInvocationsInclusiveScanNonUniformAMD(ivec2);"
            "ivec3 minInvocationsInclusiveScanNonUniformAMD(ivec3);"
            "ivec4 minInvocationsInclusiveScanNonUniformAMD(ivec4);"

            "uint  minInvocationsInclusiveScanNonUniformAMD(uint);"
            "uvec2 minInvocationsInclusiveScanNonUniformAMD(uvec2);"
            "uvec3 minInvocationsInclusiveScanNonUniformAMD(uvec3);"
            "uvec4 minInvocationsInclusiveScanNonUniformAMD(uvec4);"

            "double minInvocationsInclusiveScanNonUniformAMD(double);"
            "dvec2  minInvocationsInclusiveScanNonUniformAMD(dvec2);"
            "dvec3  minInvocationsInclusiveScanNonUniformAMD(dvec3);"
            "dvec4  minInvocationsInclusiveScanNonUniformAMD(dvec4);"

            "int64_t minInvocationsInclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 minInvocationsInclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 minInvocationsInclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 minInvocationsInclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t minInvocationsInclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  minInvocationsInclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  minInvocationsInclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  minInvocationsInclusiveScanNonUniformAMD(u64vec4);"

            "float16_t minInvocationsInclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   minInvocationsInclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   minInvocationsInclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   minInvocationsInclusiveScanNonUniformAMD(f16vec4);"

            "int16_t minInvocationsInclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 minInvocationsInclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 minInvocationsInclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 minInvocationsInclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t minInvocationsInclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  minInvocationsInclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  minInvocationsInclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  minInvocationsInclusiveScanNonUniformAMD(u16vec4);"

            "float minInvocationsExclusiveScanNonUniformAMD(float);"
            "vec2  minInvocationsExclusiveScanNonUniformAMD(vec2);"
            "vec3  minInvocationsExclusiveScanNonUniformAMD(vec3);"
            "vec4  minInvocationsExclusiveScanNonUniformAMD(vec4);"

            "int   minInvocationsExclusiveScanNonUniformAMD(int);"
            "ivec2 minInvocationsExclusiveScanNonUniformAMD(ivec2);"
            "ivec3 minInvocationsExclusiveScanNonUniformAMD(ivec3);"
            "ivec4 minInvocationsExclusiveScanNonUniformAMD(ivec4);"

            "uint  minInvocationsExclusiveScanNonUniformAMD(uint);"
            "uvec2 minInvocationsExclusiveScanNonUniformAMD(uvec2);"
            "uvec3 minInvocationsExclusiveScanNonUniformAMD(uvec3);"
            "uvec4 minInvocationsExclusiveScanNonUniformAMD(uvec4);"

            "double minInvocationsExclusiveScanNonUniformAMD(double);"
            "dvec2  minInvocationsExclusiveScanNonUniformAMD(dvec2);"
            "dvec3  minInvocationsExclusiveScanNonUniformAMD(dvec3);"
            "dvec4  minInvocationsExclusiveScanNonUniformAMD(dvec4);"

            "int64_t minInvocationsExclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 minInvocationsExclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 minInvocationsExclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 minInvocationsExclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t minInvocationsExclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  minInvocationsExclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  minInvocationsExclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  minInvocationsExclusiveScanNonUniformAMD(u64vec4);"

            "float16_t minInvocationsExclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   minInvocationsExclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   minInvocationsExclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   minInvocationsExclusiveScanNonUniformAMD(f16vec4);"

            "int16_t minInvocationsExclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 minInvocationsExclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 minInvocationsExclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 minInvocationsExclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t minInvocationsExclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  minInvocationsExclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  minInvocationsExclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  minInvocationsExclusiveScanNonUniformAMD(u16vec4);"

            "float maxInvocationsNonUniformAMD(float);"
            "vec2  maxInvocationsNonUniformAMD(vec2);"
            "vec3  maxInvocationsNonUniformAMD(vec3);"
            "vec4  maxInvocationsNonUniformAMD(vec4);"

            "int   maxInvocationsNonUniformAMD(int);"
            "ivec2 maxInvocationsNonUniformAMD(ivec2);"
            "ivec3 maxInvocationsNonUniformAMD(ivec3);"
            "ivec4 maxInvocationsNonUniformAMD(ivec4);"

            "uint  maxInvocationsNonUniformAMD(uint);"
            "uvec2 maxInvocationsNonUniformAMD(uvec2);"
            "uvec3 maxInvocationsNonUniformAMD(uvec3);"
            "uvec4 maxInvocationsNonUniformAMD(uvec4);"

            "double maxInvocationsNonUniformAMD(double);"
            "dvec2  maxInvocationsNonUniformAMD(dvec2);"
            "dvec3  maxInvocationsNonUniformAMD(dvec3);"
            "dvec4  maxInvocationsNonUniformAMD(dvec4);"

            "int64_t maxInvocationsNonUniformAMD(int64_t);"
            "i64vec2 maxInvocationsNonUniformAMD(i64vec2);"
            "i64vec3 maxInvocationsNonUniformAMD(i64vec3);"
            "i64vec4 maxInvocationsNonUniformAMD(i64vec4);"

            "uint64_t maxInvocationsNonUniformAMD(uint64_t);"
            "u64vec2  maxInvocationsNonUniformAMD(u64vec2);"
            "u64vec3  maxInvocationsNonUniformAMD(u64vec3);"
            "u64vec4  maxInvocationsNonUniformAMD(u64vec4);"

            "float16_t maxInvocationsNonUniformAMD(float16_t);"
            "f16vec2   maxInvocationsNonUniformAMD(f16vec2);"
            "f16vec3   maxInvocationsNonUniformAMD(f16vec3);"
            "f16vec4   maxInvocationsNonUniformAMD(f16vec4);"

            "int16_t maxInvocationsNonUniformAMD(int16_t);"
            "i16vec2 maxInvocationsNonUniformAMD(i16vec2);"
            "i16vec3 maxInvocationsNonUniformAMD(i16vec3);"
            "i16vec4 maxInvocationsNonUniformAMD(i16vec4);"

            "uint16_t maxInvocationsNonUniformAMD(uint16_t);"
            "u16vec2  maxInvocationsNonUniformAMD(u16vec2);"
            "u16vec3  maxInvocationsNonUniformAMD(u16vec3);"
            "u16vec4  maxInvocationsNonUniformAMD(u16vec4);"

            "float maxInvocationsInclusiveScanNonUniformAMD(float);"
            "vec2  maxInvocationsInclusiveScanNonUniformAMD(vec2);"
            "vec3  maxInvocationsInclusiveScanNonUniformAMD(vec3);"
            "vec4  maxInvocationsInclusiveScanNonUniformAMD(vec4);"

            "int   maxInvocationsInclusiveScanNonUniformAMD(int);"
            "ivec2 maxInvocationsInclusiveScanNonUniformAMD(ivec2);"
            "ivec3 maxInvocationsInclusiveScanNonUniformAMD(ivec3);"
            "ivec4 maxInvocationsInclusiveScanNonUniformAMD(ivec4);"

            "uint  maxInvocationsInclusiveScanNonUniformAMD(uint);"
            "uvec2 maxInvocationsInclusiveScanNonUniformAMD(uvec2);"
            "uvec3 maxInvocationsInclusiveScanNonUniformAMD(uvec3);"
            "uvec4 maxInvocationsInclusiveScanNonUniformAMD(uvec4);"

            "double maxInvocationsInclusiveScanNonUniformAMD(double);"
            "dvec2  maxInvocationsInclusiveScanNonUniformAMD(dvec2);"
            "dvec3  maxInvocationsInclusiveScanNonUniformAMD(dvec3);"
            "dvec4  maxInvocationsInclusiveScanNonUniformAMD(dvec4);"

            "int64_t maxInvocationsInclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 maxInvocationsInclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 maxInvocationsInclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 maxInvocationsInclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t maxInvocationsInclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  maxInvocationsInclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  maxInvocationsInclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  maxInvocationsInclusiveScanNonUniformAMD(u64vec4);"

            "float16_t maxInvocationsInclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   maxInvocationsInclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   maxInvocationsInclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   maxInvocationsInclusiveScanNonUniformAMD(f16vec4);"

            "int16_t maxInvocationsInclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 maxInvocationsInclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 maxInvocationsInclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 maxInvocationsInclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t maxInvocationsInclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  maxInvocationsInclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  maxInvocationsInclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  maxInvocationsInclusiveScanNonUniformAMD(u16vec4);"

            "float maxInvocationsExclusiveScanNonUniformAMD(float);"
            "vec2  maxInvocationsExclusiveScanNonUniformAMD(vec2);"
            "vec3  maxInvocationsExclusiveScanNonUniformAMD(vec3);"
            "vec4  maxInvocationsExclusiveScanNonUniformAMD(vec4);"

            "int   maxInvocationsExclusiveScanNonUniformAMD(int);"
            "ivec2 maxInvocationsExclusiveScanNonUniformAMD(ivec2);"
            "ivec3 maxInvocationsExclusiveScanNonUniformAMD(ivec3);"
            "ivec4 maxInvocationsExclusiveScanNonUniformAMD(ivec4);"

            "uint  maxInvocationsExclusiveScanNonUniformAMD(uint);"
            "uvec2 maxInvocationsExclusiveScanNonUniformAMD(uvec2);"
            "uvec3 maxInvocationsExclusiveScanNonUniformAMD(uvec3);"
            "uvec4 maxInvocationsExclusiveScanNonUniformAMD(uvec4);"

            "double maxInvocationsExclusiveScanNonUniformAMD(double);"
            "dvec2  maxInvocationsExclusiveScanNonUniformAMD(dvec2);"
            "dvec3  maxInvocationsExclusiveScanNonUniformAMD(dvec3);"
            "dvec4  maxInvocationsExclusiveScanNonUniformAMD(dvec4);"

            "int64_t maxInvocationsExclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 maxInvocationsExclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 maxInvocationsExclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 maxInvocationsExclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t maxInvocationsExclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  maxInvocationsExclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  maxInvocationsExclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  maxInvocationsExclusiveScanNonUniformAMD(u64vec4);"

            "float16_t maxInvocationsExclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   maxInvocationsExclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   maxInvocationsExclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   maxInvocationsExclusiveScanNonUniformAMD(f16vec4);"

            "int16_t maxInvocationsExclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 maxInvocationsExclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 maxInvocationsExclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 maxInvocationsExclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t maxInvocationsExclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  maxInvocationsExclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  maxInvocationsExclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  maxInvocationsExclusiveScanNonUniformAMD(u16vec4);"

            "float addInvocationsNonUniformAMD(float);"
            "vec2  addInvocationsNonUniformAMD(vec2);"
            "vec3  addInvocationsNonUniformAMD(vec3);"
            "vec4  addInvocationsNonUniformAMD(vec4);"

            "int   addInvocationsNonUniformAMD(int);"
            "ivec2 addInvocationsNonUniformAMD(ivec2);"
            "ivec3 addInvocationsNonUniformAMD(ivec3);"
            "ivec4 addInvocationsNonUniformAMD(ivec4);"

            "uint  addInvocationsNonUniformAMD(uint);"
            "uvec2 addInvocationsNonUniformAMD(uvec2);"
            "uvec3 addInvocationsNonUniformAMD(uvec3);"
            "uvec4 addInvocationsNonUniformAMD(uvec4);"

            "double addInvocationsNonUniformAMD(double);"
            "dvec2  addInvocationsNonUniformAMD(dvec2);"
            "dvec3  addInvocationsNonUniformAMD(dvec3);"
            "dvec4  addInvocationsNonUniformAMD(dvec4);"

            "int64_t addInvocationsNonUniformAMD(int64_t);"
            "i64vec2 addInvocationsNonUniformAMD(i64vec2);"
            "i64vec3 addInvocationsNonUniformAMD(i64vec3);"
            "i64vec4 addInvocationsNonUniformAMD(i64vec4);"

            "uint64_t addInvocationsNonUniformAMD(uint64_t);"
            "u64vec2  addInvocationsNonUniformAMD(u64vec2);"
            "u64vec3  addInvocationsNonUniformAMD(u64vec3);"
            "u64vec4  addInvocationsNonUniformAMD(u64vec4);"

            "float16_t addInvocationsNonUniformAMD(float16_t);"
            "f16vec2   addInvocationsNonUniformAMD(f16vec2);"
            "f16vec3   addInvocationsNonUniformAMD(f16vec3);"
            "f16vec4   addInvocationsNonUniformAMD(f16vec4);"

            "int16_t addInvocationsNonUniformAMD(int16_t);"
            "i16vec2 addInvocationsNonUniformAMD(i16vec2);"
            "i16vec3 addInvocationsNonUniformAMD(i16vec3);"
            "i16vec4 addInvocationsNonUniformAMD(i16vec4);"

            "uint16_t addInvocationsNonUniformAMD(uint16_t);"
            "u16vec2  addInvocationsNonUniformAMD(u16vec2);"
            "u16vec3  addInvocationsNonUniformAMD(u16vec3);"
            "u16vec4  addInvocationsNonUniformAMD(u16vec4);"

            "float addInvocationsInclusiveScanNonUniformAMD(float);"
            "vec2  addInvocationsInclusiveScanNonUniformAMD(vec2);"
            "vec3  addInvocationsInclusiveScanNonUniformAMD(vec3);"
            "vec4  addInvocationsInclusiveScanNonUniformAMD(vec4);"

            "int   addInvocationsInclusiveScanNonUniformAMD(int);"
            "ivec2 addInvocationsInclusiveScanNonUniformAMD(ivec2);"
            "ivec3 addInvocationsInclusiveScanNonUniformAMD(ivec3);"
            "ivec4 addInvocationsInclusiveScanNonUniformAMD(ivec4);"

            "uint  addInvocationsInclusiveScanNonUniformAMD(uint);"
            "uvec2 addInvocationsInclusiveScanNonUniformAMD(uvec2);"
            "uvec3 addInvocationsInclusiveScanNonUniformAMD(uvec3);"
            "uvec4 addInvocationsInclusiveScanNonUniformAMD(uvec4);"

            "double addInvocationsInclusiveScanNonUniformAMD(double);"
            "dvec2  addInvocationsInclusiveScanNonUniformAMD(dvec2);"
            "dvec3  addInvocationsInclusiveScanNonUniformAMD(dvec3);"
            "dvec4  addInvocationsInclusiveScanNonUniformAMD(dvec4);"

            "int64_t addInvocationsInclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 addInvocationsInclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 addInvocationsInclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 addInvocationsInclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t addInvocationsInclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  addInvocationsInclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  addInvocationsInclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  addInvocationsInclusiveScanNonUniformAMD(u64vec4);"

            "float16_t addInvocationsInclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   addInvocationsInclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   addInvocationsInclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   addInvocationsInclusiveScanNonUniformAMD(f16vec4);"

            "int16_t addInvocationsInclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 addInvocationsInclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 addInvocationsInclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 addInvocationsInclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t addInvocationsInclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  addInvocationsInclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  addInvocationsInclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  addInvocationsInclusiveScanNonUniformAMD(u16vec4);"

            "float addInvocationsExclusiveScanNonUniformAMD(float);"
            "vec2  addInvocationsExclusiveScanNonUniformAMD(vec2);"
            "vec3  addInvocationsExclusiveScanNonUniformAMD(vec3);"
            "vec4  addInvocationsExclusiveScanNonUniformAMD(vec4);"

            "int   addInvocationsExclusiveScanNonUniformAMD(int);"
            "ivec2 addInvocationsExclusiveScanNonUniformAMD(ivec2);"
            "ivec3 addInvocationsExclusiveScanNonUniformAMD(ivec3);"
            "ivec4 addInvocationsExclusiveScanNonUniformAMD(ivec4);"

            "uint  addInvocationsExclusiveScanNonUniformAMD(uint);"
            "uvec2 addInvocationsExclusiveScanNonUniformAMD(uvec2);"
            "uvec3 addInvocationsExclusiveScanNonUniformAMD(uvec3);"
            "uvec4 addInvocationsExclusiveScanNonUniformAMD(uvec4);"

            "double addInvocationsExclusiveScanNonUniformAMD(double);"
            "dvec2  addInvocationsExclusiveScanNonUniformAMD(dvec2);"
            "dvec3  addInvocationsExclusiveScanNonUniformAMD(dvec3);"
            "dvec4  addInvocationsExclusiveScanNonUniformAMD(dvec4);"

            "int64_t addInvocationsExclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 addInvocationsExclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 addInvocationsExclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 addInvocationsExclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t addInvocationsExclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  addInvocationsExclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  addInvocationsExclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  addInvocationsExclusiveScanNonUniformAMD(u64vec4);"

            "float16_t addInvocationsExclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   addInvocationsExclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   addInvocationsExclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   addInvocationsExclusiveScanNonUniformAMD(f16vec4);"

            "int16_t addInvocationsExclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 addInvocationsExclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 addInvocationsExclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 addInvocationsExclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t addInvocationsExclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  addInvocationsExclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  addInvocationsExclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  addInvocationsExclusiveScanNonUniformAMD(u16vec4);"

            "float swizzleInvocationsAMD(float, uvec4);"
            "vec2  swizzleInvocationsAMD(vec2,  uvec4);"
            "vec3  swizzleInvocationsAMD(vec3,  uvec4);"
            "vec4  swizzleInvocationsAMD(vec4,  uvec4);"

            "int   swizzleInvocationsAMD(int,   uvec4);"
            "ivec2 swizzleInvocationsAMD(ivec2, uvec4);"
            "ivec3 swizzleInvocationsAMD(ivec3, uvec4);"
            "ivec4 swizzleInvocationsAMD(ivec4, uvec4);"

            "uint  swizzleInvocationsAMD(uint,  uvec4);"
            "uvec2 swizzleInvocationsAMD(uvec2, uvec4);"
            "uvec3 swizzleInvocationsAMD(uvec3, uvec4);"
            "uvec4 swizzleInvocationsAMD(uvec4, uvec4);"

            "float swizzleInvocationsMaskedAMD(float, uvec3);"
            "vec2  swizzleInvocationsMaskedAMD(vec2,  uvec3);"
            "vec3  swizzleInvocationsMaskedAMD(vec3,  uvec3);"
            "vec4  swizzleInvocationsMaskedAMD(vec4,  uvec3);"

            "int   swizzleInvocationsMaskedAMD(int,   uvec3);"
            "ivec2 swizzleInvocationsMaskedAMD(ivec2, uvec3);"
            "ivec3 swizzleInvocationsMaskedAMD(ivec3, uvec3);"
            "ivec4 swizzleInvocationsMaskedAMD(ivec4, uvec3);"

            "uint  swizzleInvocationsMaskedAMD(uint,  uvec3);"
            "uvec2 swizzleInvocationsMaskedAMD(uvec2, uvec3);"
            "uvec3 swizzleInvocationsMaskedAMD(uvec3, uvec3);"
            "uvec4 swizzleInvocationsMaskedAMD(uvec4, uvec3);"

            "float writeInvocationAMD(float, float, uint);"
            "vec2  writeInvocationAMD(vec2,  vec2,  uint);"
            "vec3  writeInvocationAMD(vec3,  vec3,  uint);"
            "vec4  writeInvocationAMD(vec4,  vec4,  uint);"

            "int   writeInvocationAMD(int,   int,   uint);"
            "ivec2 writeInvocationAMD(ivec2, ivec2, uint);"
            "ivec3 writeInvocationAMD(ivec3, ivec3, uint);"
            "ivec4 writeInvocationAMD(ivec4, ivec4, uint);"

            "uint  writeInvocationAMD(uint,  uint,  uint);"
            "uvec2 writeInvocationAMD(uvec2, uvec2, uint);"
            "uvec3 writeInvocationAMD(uvec3, uvec3, uint);"
            "uvec4 writeInvocationAMD(uvec4, uvec4, uint);"

            "uint mbcntAMD(uint64_t);"

            "\n");
    }

    // GL_AMD_gcn_shader
    if (profile != EEsProfile && version >= 440) {
        commonBuiltins.append(
            "float cubeFaceIndexAMD(vec3);"
            "vec2  cubeFaceCoordAMD(vec3);"
            "uint64_t timeAMD();"

            "in int gl_SIMDGroupSizeAMD;"
            "\n");
    }

    // GL_AMD_shader_fragment_mask
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(
            "uint fragmentMaskFetchAMD(sampler2DMS,       ivec2);"
            "uint fragmentMaskFetchAMD(isampler2DMS,      ivec2);"
            "uint fragmentMaskFetchAMD(usampler2DMS,      ivec2);"

            "uint fragmentMaskFetchAMD(sampler2DMSArray,  ivec3);"
            "uint fragmentMaskFetchAMD(isampler2DMSArray, ivec3);"
            "uint fragmentMaskFetchAMD(usampler2DMSArray, ivec3);"

            "vec4  fragmentFetchAMD(sampler2DMS,       ivec2, uint);"
            "ivec4 fragmentFetchAMD(isampler2DMS,      ivec2, uint);"
            "uvec4 fragmentFetchAMD(usampler2DMS,      ivec2, uint);"

            "vec4  fragmentFetchAMD(sampler2DMSArray,  ivec3, uint);"
            "ivec4 fragmentFetchAMD(isampler2DMSArray, ivec3, uint);"
            "uvec4 fragmentFetchAMD(usampler2DMSArray, ivec3, uint);"

            "\n");
    }

    if ((profile != EEsProfile && version >= 130) ||
        (profile == EEsProfile && version >= 300)) {
        commonBuiltins.append(
            "uint countLeadingZeros(uint);"
            "uvec2 countLeadingZeros(uvec2);"
            "uvec3 countLeadingZeros(uvec3);"
            "uvec4 countLeadingZeros(uvec4);"

            "uint countTrailingZeros(uint);"
            "uvec2 countTrailingZeros(uvec2);"
            "uvec3 countTrailingZeros(uvec3);"
            "uvec4 countTrailingZeros(uvec4);"

            "uint absoluteDifference(int, int);"
            "uvec2 absoluteDifference(ivec2, ivec2);"
            "uvec3 absoluteDifference(ivec3, ivec3);"
            "uvec4 absoluteDifference(ivec4, ivec4);"

            "uint16_t absoluteDifference(int16_t, int16_t);"
            "u16vec2 absoluteDifference(i16vec2, i16vec2);"
            "u16vec3 absoluteDifference(i16vec3, i16vec3);"
            "u16vec4 absoluteDifference(i16vec4, i16vec4);"

            "uint64_t absoluteDifference(int64_t, int64_t);"
            "u64vec2 absoluteDifference(i64vec2, i64vec2);"
            "u64vec3 absoluteDifference(i64vec3, i64vec3);"
            "u64vec4 absoluteDifference(i64vec4, i64vec4);"

            "uint absoluteDifference(uint, uint);"
            "uvec2 absoluteDifference(uvec2, uvec2);"
            "uvec3 absoluteDifference(uvec3, uvec3);"
            "uvec4 absoluteDifference(uvec4, uvec4);"

            "uint16_t absoluteDifference(uint16_t, uint16_t);"
            "u16vec2 absoluteDifference(u16vec2, u16vec2);"
            "u16vec3 absoluteDifference(u16vec3, u16vec3);"
            "u16vec4 absoluteDifference(u16vec4, u16vec4);"

            "uint64_t absoluteDifference(uint64_t, uint64_t);"
            "u64vec2 absoluteDifference(u64vec2, u64vec2);"
            "u64vec3 absoluteDifference(u64vec3, u64vec3);"
            "u64vec4 absoluteDifference(u64vec4, u64vec4);"

            "int addSaturate(int, int);"
            "ivec2 addSaturate(ivec2, ivec2);"
            "ivec3 addSaturate(ivec3, ivec3);"
            "ivec4 addSaturate(ivec4, ivec4);"

            "int16_t addSaturate(int16_t, int16_t);"
            "i16vec2 addSaturate(i16vec2, i16vec2);"
            "i16vec3 addSaturate(i16vec3, i16vec3);"
            "i16vec4 addSaturate(i16vec4, i16vec4);"

            "int64_t addSaturate(int64_t, int64_t);"
            "i64vec2 addSaturate(i64vec2, i64vec2);"
            "i64vec3 addSaturate(i64vec3, i64vec3);"
            "i64vec4 addSaturate(i64vec4, i64vec4);"

            "uint addSaturate(uint, uint);"
            "uvec2 addSaturate(uvec2, uvec2);"
            "uvec3 addSaturate(uvec3, uvec3);"
            "uvec4 addSaturate(uvec4, uvec4);"

            "uint16_t addSaturate(uint16_t, uint16_t);"
            "u16vec2 addSaturate(u16vec2, u16vec2);"
            "u16vec3 addSaturate(u16vec3, u16vec3);"
            "u16vec4 addSaturate(u16vec4, u16vec4);"

            "uint64_t addSaturate(uint64_t, uint64_t);"
            "u64vec2 addSaturate(u64vec2, u64vec2);"
            "u64vec3 addSaturate(u64vec3, u64vec3);"
            "u64vec4 addSaturate(u64vec4, u64vec4);"

            "int subtractSaturate(int, int);"
            "ivec2 subtractSaturate(ivec2, ivec2);"
            "ivec3 subtractSaturate(ivec3, ivec3);"
            "ivec4 subtractSaturate(ivec4, ivec4);"

            "int16_t subtractSaturate(int16_t, int16_t);"
            "i16vec2 subtractSaturate(i16vec2, i16vec2);"
            "i16vec3 subtractSaturate(i16vec3, i16vec3);"
            "i16vec4 subtractSaturate(i16vec4, i16vec4);"

            "int64_t subtractSaturate(int64_t, int64_t);"
            "i64vec2 subtractSaturate(i64vec2, i64vec2);"
            "i64vec3 subtractSaturate(i64vec3, i64vec3);"
            "i64vec4 subtractSaturate(i64vec4, i64vec4);"

            "uint subtractSaturate(uint, uint);"
            "uvec2 subtractSaturate(uvec2, uvec2);"
            "uvec3 subtractSaturate(uvec3, uvec3);"
            "uvec4 subtractSaturate(uvec4, uvec4);"

            "uint16_t subtractSaturate(uint16_t, uint16_t);"
            "u16vec2 subtractSaturate(u16vec2, u16vec2);"
            "u16vec3 subtractSaturate(u16vec3, u16vec3);"
            "u16vec4 subtractSaturate(u16vec4, u16vec4);"

            "uint64_t subtractSaturate(uint64_t, uint64_t);"
            "u64vec2 subtractSaturate(u64vec2, u64vec2);"
            "u64vec3 subtractSaturate(u64vec3, u64vec3);"
            "u64vec4 subtractSaturate(u64vec4, u64vec4);"

            "int average(int, int);"
            "ivec2 average(ivec2, ivec2);"
            "ivec3 average(ivec3, ivec3);"
            "ivec4 average(ivec4, ivec4);"

            "int16_t average(int16_t, int16_t);"
            "i16vec2 average(i16vec2, i16vec2);"
            "i16vec3 average(i16vec3, i16vec3);"
            "i16vec4 average(i16vec4, i16vec4);"

            "int64_t average(int64_t, int64_t);"
            "i64vec2 average(i64vec2, i64vec2);"
            "i64vec3 average(i64vec3, i64vec3);"
            "i64vec4 average(i64vec4, i64vec4);"

            "uint average(uint, uint);"
            "uvec2 average(uvec2, uvec2);"
            "uvec3 average(uvec3, uvec3);"
            "uvec4 average(uvec4, uvec4);"

            "uint16_t average(uint16_t, uint16_t);"
            "u16vec2 average(u16vec2, u16vec2);"
            "u16vec3 average(u16vec3, u16vec3);"
            "u16vec4 average(u16vec4, u16vec4);"

            "uint64_t average(uint64_t, uint64_t);"
            "u64vec2 average(u64vec2, u64vec2);"
            "u64vec3 average(u64vec3, u64vec3);"
            "u64vec4 average(u64vec4, u64vec4);"

            "int averageRounded(int, int);"
            "ivec2 averageRounded(ivec2, ivec2);"
            "ivec3 averageRounded(ivec3, ivec3);"
            "ivec4 averageRounded(ivec4, ivec4);"

            "int16_t averageRounded(int16_t, int16_t);"
            "i16vec2 averageRounded(i16vec2, i16vec2);"
            "i16vec3 averageRounded(i16vec3, i16vec3);"
            "i16vec4 averageRounded(i16vec4, i16vec4);"

            "int64_t averageRounded(int64_t, int64_t);"
            "i64vec2 averageRounded(i64vec2, i64vec2);"
            "i64vec3 averageRounded(i64vec3, i64vec3);"
            "i64vec4 averageRounded(i64vec4, i64vec4);"

            "uint averageRounded(uint, uint);"
            "uvec2 averageRounded(uvec2, uvec2);"
            "uvec3 averageRounded(uvec3, uvec3);"
            "uvec4 averageRounded(uvec4, uvec4);"

            "uint16_t averageRounded(uint16_t, uint16_t);"
            "u16vec2 averageRounded(u16vec2, u16vec2);"
            "u16vec3 averageRounded(u16vec3, u16vec3);"
            "u16vec4 averageRounded(u16vec4, u16vec4);"

            "uint64_t averageRounded(uint64_t, uint64_t);"
            "u64vec2 averageRounded(u64vec2, u64vec2);"
            "u64vec3 averageRounded(u64vec3, u64vec3);"
            "u64vec4 averageRounded(u64vec4, u64vec4);"

            "int multiply32x16(int, int);"
            "ivec2 multiply32x16(ivec2, ivec2);"
            "ivec3 multiply32x16(ivec3, ivec3);"
            "ivec4 multiply32x16(ivec4, ivec4);"

            "uint multiply32x16(uint, uint);"
            "uvec2 multiply32x16(uvec2, uvec2);"
            "uvec3 multiply32x16(uvec3, uvec3);"
            "uvec4 multiply32x16(uvec4, uvec4);"
            "\n");
    }

    if ((profile != EEsProfile && version >= 450) ||
        (profile == EEsProfile && version >= 320)) {
        commonBuiltins.append(
            "struct gl_TextureFootprint2DNV {"
                "uvec2 anchor;"
                "uvec2 offset;"
                "uvec2 mask;"
                "uint lod;"
                "uint granularity;"
            "};"

            "struct gl_TextureFootprint3DNV {"
                "uvec3 anchor;"
                "uvec3 offset;"
                "uvec2 mask;"
                "uint lod;"
                "uint granularity;"
            "};"
            "bool textureFootprintNV(sampler2D, vec2, int, bool, out gl_TextureFootprint2DNV);"
            "bool textureFootprintNV(sampler3D, vec3, int, bool, out gl_TextureFootprint3DNV);"
            "bool textureFootprintNV(sampler2D, vec2, int, bool, out gl_TextureFootprint2DNV, float);"
            "bool textureFootprintNV(sampler3D, vec3, int, bool, out gl_TextureFootprint3DNV, float);"
            "bool textureFootprintClampNV(sampler2D, vec2, float, int, bool, out gl_TextureFootprint2DNV);"
            "bool textureFootprintClampNV(sampler3D, vec3, float, int, bool, out gl_TextureFootprint3DNV);"
            "bool textureFootprintClampNV(sampler2D, vec2, float, int, bool, out gl_TextureFootprint2DNV, float);"
            "bool textureFootprintClampNV(sampler3D, vec3, float, int, bool, out gl_TextureFootprint3DNV, float);"
            "bool textureFootprintLodNV(sampler2D, vec2, float, int, bool, out gl_TextureFootprint2DNV);"
            "bool textureFootprintLodNV(sampler3D, vec3, float, int, bool, out gl_TextureFootprint3DNV);"
            "bool textureFootprintGradNV(sampler2D, vec2, vec2, vec2, int, bool, out gl_TextureFootprint2DNV);"
            "bool textureFootprintGradClampNV(sampler2D, vec2, vec2, vec2, float, int, bool, out gl_TextureFootprint2DNV);"
            "\n");
    }

    if ((profile == EEsProfile && version >= 300 && version < 310) ||
        (profile != EEsProfile && version >= 150 && version < 450)) { // GL_EXT_shader_integer_mix
        commonBuiltins.append("int mix(int, int, bool);"
                              "ivec2 mix(ivec2, ivec2, bvec2);"
                              "ivec3 mix(ivec3, ivec3, bvec3);"
                              "ivec4 mix(ivec4, ivec4, bvec4);"
                              "uint  mix(uint,  uint,  bool );"
                              "uvec2 mix(uvec2, uvec2, bvec2);"
                              "uvec3 mix(uvec3, uvec3, bvec3);"
                              "uvec4 mix(uvec4, uvec4, bvec4);"
                              "bool  mix(bool,  bool,  bool );"
                              "bvec2 mix(bvec2, bvec2, bvec2);"
                              "bvec3 mix(bvec3, bvec3, bvec3);"
                              "bvec4 mix(bvec4, bvec4, bvec4);"

                              "\n");
    }

    // GL_AMD_gpu_shader_half_float/Explicit types
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 310)) {
        commonBuiltins.append(
            "float16_t radians(float16_t);"
            "f16vec2   radians(f16vec2);"
            "f16vec3   radians(f16vec3);"
            "f16vec4   radians(f16vec4);"

            "float16_t degrees(float16_t);"
            "f16vec2   degrees(f16vec2);"
            "f16vec3   degrees(f16vec3);"
            "f16vec4   degrees(f16vec4);"

            "float16_t sin(float16_t);"
            "f16vec2   sin(f16vec2);"
            "f16vec3   sin(f16vec3);"
            "f16vec4   sin(f16vec4);"

            "float16_t cos(float16_t);"
            "f16vec2   cos(f16vec2);"
            "f16vec3   cos(f16vec3);"
            "f16vec4   cos(f16vec4);"

            "float16_t tan(float16_t);"
            "f16vec2   tan(f16vec2);"
            "f16vec3   tan(f16vec3);"
            "f16vec4   tan(f16vec4);"

            "float16_t asin(float16_t);"
            "f16vec2   asin(f16vec2);"
            "f16vec3   asin(f16vec3);"
            "f16vec4   asin(f16vec4);"

            "float16_t acos(float16_t);"
            "f16vec2   acos(f16vec2);"
            "f16vec3   acos(f16vec3);"
            "f16vec4   acos(f16vec4);"

            "float16_t atan(float16_t, float16_t);"
            "f16vec2   atan(f16vec2,   f16vec2);"
            "f16vec3   atan(f16vec3,   f16vec3);"
            "f16vec4   atan(f16vec4,   f16vec4);"

            "float16_t atan(float16_t);"
            "f16vec2   atan(f16vec2);"
            "f16vec3   atan(f16vec3);"
            "f16vec4   atan(f16vec4);"

            "float16_t sinh(float16_t);"
            "f16vec2   sinh(f16vec2);"
            "f16vec3   sinh(f16vec3);"
            "f16vec4   sinh(f16vec4);"

            "float16_t cosh(float16_t);"
            "f16vec2   cosh(f16vec2);"
            "f16vec3   cosh(f16vec3);"
            "f16vec4   cosh(f16vec4);"

            "float16_t tanh(float16_t);"
            "f16vec2   tanh(f16vec2);"
            "f16vec3   tanh(f16vec3);"
            "f16vec4   tanh(f16vec4);"

            "float16_t asinh(float16_t);"
            "f16vec2   asinh(f16vec2);"
            "f16vec3   asinh(f16vec3);"
            "f16vec4   asinh(f16vec4);"

            "float16_t acosh(float16_t);"
            "f16vec2   acosh(f16vec2);"
            "f16vec3   acosh(f16vec3);"
            "f16vec4   acosh(f16vec4);"

            "float16_t atanh(float16_t);"
            "f16vec2   atanh(f16vec2);"
            "f16vec3   atanh(f16vec3);"
            "f16vec4   atanh(f16vec4);"

            "float16_t pow(float16_t, float16_t);"
            "f16vec2   pow(f16vec2,   f16vec2);"
            "f16vec3   pow(f16vec3,   f16vec3);"
            "f16vec4   pow(f16vec4,   f16vec4);"

            "float16_t exp(float16_t);"
            "f16vec2   exp(f16vec2);"
            "f16vec3   exp(f16vec3);"
            "f16vec4   exp(f16vec4);"

            "float16_t log(float16_t);"
            "f16vec2   log(f16vec2);"
            "f16vec3   log(f16vec3);"
            "f16vec4   log(f16vec4);"

            "float16_t exp2(float16_t);"
            "f16vec2   exp2(f16vec2);"
            "f16vec3   exp2(f16vec3);"
            "f16vec4   exp2(f16vec4);"

            "float16_t log2(float16_t);"
            "f16vec2   log2(f16vec2);"
            "f16vec3   log2(f16vec3);"
            "f16vec4   log2(f16vec4);"

            "float16_t sqrt(float16_t);"
            "f16vec2   sqrt(f16vec2);"
            "f16vec3   sqrt(f16vec3);"
            "f16vec4   sqrt(f16vec4);"

            "float16_t inversesqrt(float16_t);"
            "f16vec2   inversesqrt(f16vec2);"
            "f16vec3   inversesqrt(f16vec3);"
            "f16vec4   inversesqrt(f16vec4);"

            "float16_t abs(float16_t);"
            "f16vec2   abs(f16vec2);"
            "f16vec3   abs(f16vec3);"
            "f16vec4   abs(f16vec4);"

            "float16_t sign(float16_t);"
            "f16vec2   sign(f16vec2);"
            "f16vec3   sign(f16vec3);"
            "f16vec4   sign(f16vec4);"

            "float16_t floor(float16_t);"
            "f16vec2   floor(f16vec2);"
            "f16vec3   floor(f16vec3);"
            "f16vec4   floor(f16vec4);"

            "float16_t trunc(float16_t);"
            "f16vec2   trunc(f16vec2);"
            "f16vec3   trunc(f16vec3);"
            "f16vec4   trunc(f16vec4);"

            "float16_t round(float16_t);"
            "f16vec2   round(f16vec2);"
            "f16vec3   round(f16vec3);"
            "f16vec4   round(f16vec4);"

            "float16_t roundEven(float16_t);"
            "f16vec2   roundEven(f16vec2);"
            "f16vec3   roundEven(f16vec3);"
            "f16vec4   roundEven(f16vec4);"

            "float16_t ceil(float16_t);"
            "f16vec2   ceil(f16vec2);"
            "f16vec3   ceil(f16vec3);"
            "f16vec4   ceil(f16vec4);"

            "float16_t fract(float16_t);"
            "f16vec2   fract(f16vec2);"
            "f16vec3   fract(f16vec3);"
            "f16vec4   fract(f16vec4);"

            "float16_t mod(float16_t, float16_t);"
            "f16vec2   mod(f16vec2,   float16_t);"
            "f16vec3   mod(f16vec3,   float16_t);"
            "f16vec4   mod(f16vec4,   float16_t);"
            "f16vec2   mod(f16vec2,   f16vec2);"
            "f16vec3   mod(f16vec3,   f16vec3);"
            "f16vec4   mod(f16vec4,   f16vec4);"

            "float16_t modf(float16_t, out float16_t);"
            "f16vec2   modf(f16vec2,   out f16vec2);"
            "f16vec3   modf(f16vec3,   out f16vec3);"
            "f16vec4   modf(f16vec4,   out f16vec4);"

            "float16_t min(float16_t, float16_t);"
            "f16vec2   min(f16vec2,   float16_t);"
            "f16vec3   min(f16vec3,   float16_t);"
            "f16vec4   min(f16vec4,   float16_t);"
            "f16vec2   min(f16vec2,   f16vec2);"
            "f16vec3   min(f16vec3,   f16vec3);"
            "f16vec4   min(f16vec4,   f16vec4);"

            "float16_t max(float16_t, float16_t);"
            "f16vec2   max(f16vec2,   float16_t);"
            "f16vec3   max(f16vec3,   float16_t);"
            "f16vec4   max(f16vec4,   float16_t);"
            "f16vec2   max(f16vec2,   f16vec2);"
            "f16vec3   max(f16vec3,   f16vec3);"
            "f16vec4   max(f16vec4,   f16vec4);"

            "float16_t clamp(float16_t, float16_t, float16_t);"
            "f16vec2   clamp(f16vec2,   float16_t, float16_t);"
            "f16vec3   clamp(f16vec3,   float16_t, float16_t);"
            "f16vec4   clamp(f16vec4,   float16_t, float16_t);"
            "f16vec2   clamp(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   clamp(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   clamp(f16vec4,   f16vec4,   f16vec4);"

            "float16_t mix(float16_t, float16_t, float16_t);"
            "f16vec2   mix(f16vec2,   f16vec2,   float16_t);"
            "f16vec3   mix(f16vec3,   f16vec3,   float16_t);"
            "f16vec4   mix(f16vec4,   f16vec4,   float16_t);"
            "f16vec2   mix(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   mix(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   mix(f16vec4,   f16vec4,   f16vec4);"
            "float16_t mix(float16_t, float16_t, bool);"
            "f16vec2   mix(f16vec2,   f16vec2,   bvec2);"
            "f16vec3   mix(f16vec3,   f16vec3,   bvec3);"
            "f16vec4   mix(f16vec4,   f16vec4,   bvec4);"

            "float16_t step(float16_t, float16_t);"
            "f16vec2   step(f16vec2,   f16vec2);"
            "f16vec3   step(f16vec3,   f16vec3);"
            "f16vec4   step(f16vec4,   f16vec4);"
            "f16vec2   step(float16_t, f16vec2);"
            "f16vec3   step(float16_t, f16vec3);"
            "f16vec4   step(float16_t, f16vec4);"

            "float16_t smoothstep(float16_t, float16_t, float16_t);"
            "f16vec2   smoothstep(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   smoothstep(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   smoothstep(f16vec4,   f16vec4,   f16vec4);"
            "f16vec2   smoothstep(float16_t, float16_t, f16vec2);"
            "f16vec3   smoothstep(float16_t, float16_t, f16vec3);"
            "f16vec4   smoothstep(float16_t, float16_t, f16vec4);"

            "bool  isnan(float16_t);"
            "bvec2 isnan(f16vec2);"
            "bvec3 isnan(f16vec3);"
            "bvec4 isnan(f16vec4);"

            "bool  isinf(float16_t);"
            "bvec2 isinf(f16vec2);"
            "bvec3 isinf(f16vec3);"
            "bvec4 isinf(f16vec4);"

            "float16_t fma(float16_t, float16_t, float16_t);"
            "f16vec2   fma(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   fma(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   fma(f16vec4,   f16vec4,   f16vec4);"

            "float16_t frexp(float16_t, out int);"
            "f16vec2   frexp(f16vec2,   out ivec2);"
            "f16vec3   frexp(f16vec3,   out ivec3);"
            "f16vec4   frexp(f16vec4,   out ivec4);"

            "float16_t ldexp(float16_t, in int);"
            "f16vec2   ldexp(f16vec2,   in ivec2);"
            "f16vec3   ldexp(f16vec3,   in ivec3);"
            "f16vec4   ldexp(f16vec4,   in ivec4);"

            "uint    packFloat2x16(f16vec2);"
            "f16vec2 unpackFloat2x16(uint);"

            "float16_t length(float16_t);"
            "float16_t length(f16vec2);"
            "float16_t length(f16vec3);"
            "float16_t length(f16vec4);"

            "float16_t distance(float16_t, float16_t);"
            "float16_t distance(f16vec2,   f16vec2);"
            "float16_t distance(f16vec3,   f16vec3);"
            "float16_t distance(f16vec4,   f16vec4);"

            "float16_t dot(float16_t, float16_t);"
            "float16_t dot(f16vec2,   f16vec2);"
            "float16_t dot(f16vec3,   f16vec3);"
            "float16_t dot(f16vec4,   f16vec4);"

            "f16vec3 cross(f16vec3, f16vec3);"

            "float16_t normalize(float16_t);"
            "f16vec2   normalize(f16vec2);"
            "f16vec3   normalize(f16vec3);"
            "f16vec4   normalize(f16vec4);"

            "float16_t faceforward(float16_t, float16_t, float16_t);"
            "f16vec2   faceforward(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   faceforward(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   faceforward(f16vec4,   f16vec4,   f16vec4);"

            "float16_t reflect(float16_t, float16_t);"
            "f16vec2   reflect(f16vec2,   f16vec2);"
            "f16vec3   reflect(f16vec3,   f16vec3);"
            "f16vec4   reflect(f16vec4,   f16vec4);"

            "float16_t refract(float16_t, float16_t, float16_t);"
            "f16vec2   refract(f16vec2,   f16vec2,   float16_t);"
            "f16vec3   refract(f16vec3,   f16vec3,   float16_t);"
            "f16vec4   refract(f16vec4,   f16vec4,   float16_t);"

            "f16mat2   matrixCompMult(f16mat2,   f16mat2);"
            "f16mat3   matrixCompMult(f16mat3,   f16mat3);"
            "f16mat4   matrixCompMult(f16mat4,   f16mat4);"
            "f16mat2x3 matrixCompMult(f16mat2x3, f16mat2x3);"
            "f16mat2x4 matrixCompMult(f16mat2x4, f16mat2x4);"
            "f16mat3x2 matrixCompMult(f16mat3x2, f16mat3x2);"
            "f16mat3x4 matrixCompMult(f16mat3x4, f16mat3x4);"
            "f16mat4x2 matrixCompMult(f16mat4x2, f16mat4x2);"
            "f16mat4x3 matrixCompMult(f16mat4x3, f16mat4x3);"

            "f16mat2   outerProduct(f16vec2, f16vec2);"
            "f16mat3   outerProduct(f16vec3, f16vec3);"
            "f16mat4   outerProduct(f16vec4, f16vec4);"
            "f16mat2x3 outerProduct(f16vec3, f16vec2);"
            "f16mat3x2 outerProduct(f16vec2, f16vec3);"
            "f16mat2x4 outerProduct(f16vec4, f16vec2);"
            "f16mat4x2 outerProduct(f16vec2, f16vec4);"
            "f16mat3x4 outerProduct(f16vec4, f16vec3);"
            "f16mat4x3 outerProduct(f16vec3, f16vec4);"

            "f16mat2   transpose(f16mat2);"
            "f16mat3   transpose(f16mat3);"
            "f16mat4   transpose(f16mat4);"
            "f16mat2x3 transpose(f16mat3x2);"
            "f16mat3x2 transpose(f16mat2x3);"
            "f16mat2x4 transpose(f16mat4x2);"
            "f16mat4x2 transpose(f16mat2x4);"
            "f16mat3x4 transpose(f16mat4x3);"
            "f16mat4x3 transpose(f16mat3x4);"

            "float16_t determinant(f16mat2);"
            "float16_t determinant(f16mat3);"
            "float16_t determinant(f16mat4);"

            "f16mat2 inverse(f16mat2);"
            "f16mat3 inverse(f16mat3);"
            "f16mat4 inverse(f16mat4);"

            "bvec2 lessThan(f16vec2, f16vec2);"
            "bvec3 lessThan(f16vec3, f16vec3);"
            "bvec4 lessThan(f16vec4, f16vec4);"

            "bvec2 lessThanEqual(f16vec2, f16vec2);"
            "bvec3 lessThanEqual(f16vec3, f16vec3);"
            "bvec4 lessThanEqual(f16vec4, f16vec4);"

            "bvec2 greaterThan(f16vec2, f16vec2);"
            "bvec3 greaterThan(f16vec3, f16vec3);"
            "bvec4 greaterThan(f16vec4, f16vec4);"

            "bvec2 greaterThanEqual(f16vec2, f16vec2);"
            "bvec3 greaterThanEqual(f16vec3, f16vec3);"
            "bvec4 greaterThanEqual(f16vec4, f16vec4);"

            "bvec2 equal(f16vec2, f16vec2);"
            "bvec3 equal(f16vec3, f16vec3);"
            "bvec4 equal(f16vec4, f16vec4);"

            "bvec2 notEqual(f16vec2, f16vec2);"
            "bvec3 notEqual(f16vec3, f16vec3);"
            "bvec4 notEqual(f16vec4, f16vec4);"

            "\n");
    }

    // Explicit types
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 310)) {
        commonBuiltins.append(
            "int8_t abs(int8_t);"
            "i8vec2 abs(i8vec2);"
            "i8vec3 abs(i8vec3);"
            "i8vec4 abs(i8vec4);"

            "int8_t sign(int8_t);"
            "i8vec2 sign(i8vec2);"
            "i8vec3 sign(i8vec3);"
            "i8vec4 sign(i8vec4);"

            "int8_t min(int8_t x, int8_t y);"
            "i8vec2 min(i8vec2 x, int8_t y);"
            "i8vec3 min(i8vec3 x, int8_t y);"
            "i8vec4 min(i8vec4 x, int8_t y);"
            "i8vec2 min(i8vec2 x, i8vec2 y);"
            "i8vec3 min(i8vec3 x, i8vec3 y);"
            "i8vec4 min(i8vec4 x, i8vec4 y);"

            "uint8_t min(uint8_t x, uint8_t y);"
            "u8vec2 min(u8vec2 x, uint8_t y);"
            "u8vec3 min(u8vec3 x, uint8_t y);"
            "u8vec4 min(u8vec4 x, uint8_t y);"
            "u8vec2 min(u8vec2 x, u8vec2 y);"
            "u8vec3 min(u8vec3 x, u8vec3 y);"
            "u8vec4 min(u8vec4 x, u8vec4 y);"

            "int8_t max(int8_t x, int8_t y);"
            "i8vec2 max(i8vec2 x, int8_t y);"
            "i8vec3 max(i8vec3 x, int8_t y);"
            "i8vec4 max(i8vec4 x, int8_t y);"
            "i8vec2 max(i8vec2 x, i8vec2 y);"
            "i8vec3 max(i8vec3 x, i8vec3 y);"
            "i8vec4 max(i8vec4 x, i8vec4 y);"

            "uint8_t max(uint8_t x, uint8_t y);"
            "u8vec2 max(u8vec2 x, uint8_t y);"
            "u8vec3 max(u8vec3 x, uint8_t y);"
            "u8vec4 max(u8vec4 x, uint8_t y);"
            "u8vec2 max(u8vec2 x, u8vec2 y);"
            "u8vec3 max(u8vec3 x, u8vec3 y);"
            "u8vec4 max(u8vec4 x, u8vec4 y);"

            "int8_t    clamp(int8_t x, int8_t minVal, int8_t maxVal);"
            "i8vec2  clamp(i8vec2  x, int8_t minVal, int8_t maxVal);"
            "i8vec3  clamp(i8vec3  x, int8_t minVal, int8_t maxVal);"
            "i8vec4  clamp(i8vec4  x, int8_t minVal, int8_t maxVal);"
            "i8vec2  clamp(i8vec2  x, i8vec2  minVal, i8vec2  maxVal);"
            "i8vec3  clamp(i8vec3  x, i8vec3  minVal, i8vec3  maxVal);"
            "i8vec4  clamp(i8vec4  x, i8vec4  minVal, i8vec4  maxVal);"

            "uint8_t   clamp(uint8_t x, uint8_t minVal, uint8_t maxVal);"
            "u8vec2  clamp(u8vec2  x, uint8_t minVal, uint8_t maxVal);"
            "u8vec3  clamp(u8vec3  x, uint8_t minVal, uint8_t maxVal);"
            "u8vec4  clamp(u8vec4  x, uint8_t minVal, uint8_t maxVal);"
            "u8vec2  clamp(u8vec2  x, u8vec2  minVal, u8vec2  maxVal);"
            "u8vec3  clamp(u8vec3  x, u8vec3  minVal, u8vec3  maxVal);"
            "u8vec4  clamp(u8vec4  x, u8vec4  minVal, u8vec4  maxVal);"

            "int8_t  mix(int8_t,  int8_t,  bool);"
            "i8vec2  mix(i8vec2,  i8vec2,  bvec2);"
            "i8vec3  mix(i8vec3,  i8vec3,  bvec3);"
            "i8vec4  mix(i8vec4,  i8vec4,  bvec4);"
            "uint8_t mix(uint8_t, uint8_t, bool);"
            "u8vec2  mix(u8vec2,  u8vec2,  bvec2);"
            "u8vec3  mix(u8vec3,  u8vec3,  bvec3);"
            "u8vec4  mix(u8vec4,  u8vec4,  bvec4);"

            "bvec2 lessThan(i8vec2, i8vec2);"
            "bvec3 lessThan(i8vec3, i8vec3);"
            "bvec4 lessThan(i8vec4, i8vec4);"
            "bvec2 lessThan(u8vec2, u8vec2);"
            "bvec3 lessThan(u8vec3, u8vec3);"
            "bvec4 lessThan(u8vec4, u8vec4);"

            "bvec2 lessThanEqual(i8vec2, i8vec2);"
            "bvec3 lessThanEqual(i8vec3, i8vec3);"
            "bvec4 lessThanEqual(i8vec4, i8vec4);"
            "bvec2 lessThanEqual(u8vec2, u8vec2);"
            "bvec3 lessThanEqual(u8vec3, u8vec3);"
            "bvec4 lessThanEqual(u8vec4, u8vec4);"

            "bvec2 greaterThan(i8vec2, i8vec2);"
            "bvec3 greaterThan(i8vec3, i8vec3);"
            "bvec4 greaterThan(i8vec4, i8vec4);"
            "bvec2 greaterThan(u8vec2, u8vec2);"
            "bvec3 greaterThan(u8vec3, u8vec3);"
            "bvec4 greaterThan(u8vec4, u8vec4);"

            "bvec2 greaterThanEqual(i8vec2, i8vec2);"
            "bvec3 greaterThanEqual(i8vec3, i8vec3);"
            "bvec4 greaterThanEqual(i8vec4, i8vec4);"
            "bvec2 greaterThanEqual(u8vec2, u8vec2);"
            "bvec3 greaterThanEqual(u8vec3, u8vec3);"
            "bvec4 greaterThanEqual(u8vec4, u8vec4);"

            "bvec2 equal(i8vec2, i8vec2);"
            "bvec3 equal(i8vec3, i8vec3);"
            "bvec4 equal(i8vec4, i8vec4);"
            "bvec2 equal(u8vec2, u8vec2);"
            "bvec3 equal(u8vec3, u8vec3);"
            "bvec4 equal(u8vec4, u8vec4);"

            "bvec2 notEqual(i8vec2, i8vec2);"
            "bvec3 notEqual(i8vec3, i8vec3);"
            "bvec4 notEqual(i8vec4, i8vec4);"
            "bvec2 notEqual(u8vec2, u8vec2);"
            "bvec3 notEqual(u8vec3, u8vec3);"
            "bvec4 notEqual(u8vec4, u8vec4);"

            "  int8_t bitfieldExtract(  int8_t, int8_t, int8_t);"
            "i8vec2 bitfieldExtract(i8vec2, int8_t, int8_t);"
            "i8vec3 bitfieldExtract(i8vec3, int8_t, int8_t);"
            "i8vec4 bitfieldExtract(i8vec4, int8_t, int8_t);"

            " uint8_t bitfieldExtract( uint8_t, int8_t, int8_t);"
            "u8vec2 bitfieldExtract(u8vec2, int8_t, int8_t);"
            "u8vec3 bitfieldExtract(u8vec3, int8_t, int8_t);"
            "u8vec4 bitfieldExtract(u8vec4, int8_t, int8_t);"

            "  int8_t bitfieldInsert(  int8_t base,   int8_t, int8_t, int8_t);"
            "i8vec2 bitfieldInsert(i8vec2 base, i8vec2, int8_t, int8_t);"
            "i8vec3 bitfieldInsert(i8vec3 base, i8vec3, int8_t, int8_t);"
            "i8vec4 bitfieldInsert(i8vec4 base, i8vec4, int8_t, int8_t);"

            " uint8_t bitfieldInsert( uint8_t base,  uint8_t, int8_t, int8_t);"
            "u8vec2 bitfieldInsert(u8vec2 base, u8vec2, int8_t, int8_t);"
            "u8vec3 bitfieldInsert(u8vec3 base, u8vec3, int8_t, int8_t);"
            "u8vec4 bitfieldInsert(u8vec4 base, u8vec4, int8_t, int8_t);"

            "  int8_t bitCount(  int8_t);"
            "i8vec2 bitCount(i8vec2);"
            "i8vec3 bitCount(i8vec3);"
            "i8vec4 bitCount(i8vec4);"

            "  int8_t bitCount( uint8_t);"
            "i8vec2 bitCount(u8vec2);"
            "i8vec3 bitCount(u8vec3);"
            "i8vec4 bitCount(u8vec4);"

            "  int8_t findLSB(  int8_t);"
            "i8vec2 findLSB(i8vec2);"
            "i8vec3 findLSB(i8vec3);"
            "i8vec4 findLSB(i8vec4);"

            "  int8_t findLSB( uint8_t);"
            "i8vec2 findLSB(u8vec2);"
            "i8vec3 findLSB(u8vec3);"
            "i8vec4 findLSB(u8vec4);"

            "  int8_t findMSB(  int8_t);"
            "i8vec2 findMSB(i8vec2);"
            "i8vec3 findMSB(i8vec3);"
            "i8vec4 findMSB(i8vec4);"

            "  int8_t findMSB( uint8_t);"
            "i8vec2 findMSB(u8vec2);"
            "i8vec3 findMSB(u8vec3);"
            "i8vec4 findMSB(u8vec4);"

            "int16_t abs(int16_t);"
            "i16vec2 abs(i16vec2);"
            "i16vec3 abs(i16vec3);"
            "i16vec4 abs(i16vec4);"

            "int16_t sign(int16_t);"
            "i16vec2 sign(i16vec2);"
            "i16vec3 sign(i16vec3);"
            "i16vec4 sign(i16vec4);"

            "int16_t min(int16_t x, int16_t y);"
            "i16vec2 min(i16vec2 x, int16_t y);"
            "i16vec3 min(i16vec3 x, int16_t y);"
            "i16vec4 min(i16vec4 x, int16_t y);"
            "i16vec2 min(i16vec2 x, i16vec2 y);"
            "i16vec3 min(i16vec3 x, i16vec3 y);"
            "i16vec4 min(i16vec4 x, i16vec4 y);"

            "uint16_t min(uint16_t x, uint16_t y);"
            "u16vec2 min(u16vec2 x, uint16_t y);"
            "u16vec3 min(u16vec3 x, uint16_t y);"
            "u16vec4 min(u16vec4 x, uint16_t y);"
            "u16vec2 min(u16vec2 x, u16vec2 y);"
            "u16vec3 min(u16vec3 x, u16vec3 y);"
            "u16vec4 min(u16vec4 x, u16vec4 y);"

            "int16_t max(int16_t x, int16_t y);"
            "i16vec2 max(i16vec2 x, int16_t y);"
            "i16vec3 max(i16vec3 x, int16_t y);"
            "i16vec4 max(i16vec4 x, int16_t y);"
            "i16vec2 max(i16vec2 x, i16vec2 y);"
            "i16vec3 max(i16vec3 x, i16vec3 y);"
            "i16vec4 max(i16vec4 x, i16vec4 y);"

            "uint16_t max(uint16_t x, uint16_t y);"
            "u16vec2 max(u16vec2 x, uint16_t y);"
            "u16vec3 max(u16vec3 x, uint16_t y);"
            "u16vec4 max(u16vec4 x, uint16_t y);"
            "u16vec2 max(u16vec2 x, u16vec2 y);"
            "u16vec3 max(u16vec3 x, u16vec3 y);"
            "u16vec4 max(u16vec4 x, u16vec4 y);"

            "int16_t    clamp(int16_t x, int16_t minVal, int16_t maxVal);"
            "i16vec2  clamp(i16vec2  x, int16_t minVal, int16_t maxVal);"
            "i16vec3  clamp(i16vec3  x, int16_t minVal, int16_t maxVal);"
            "i16vec4  clamp(i16vec4  x, int16_t minVal, int16_t maxVal);"
            "i16vec2  clamp(i16vec2  x, i16vec2  minVal, i16vec2  maxVal);"
            "i16vec3  clamp(i16vec3  x, i16vec3  minVal, i16vec3  maxVal);"
            "i16vec4  clamp(i16vec4  x, i16vec4  minVal, i16vec4  maxVal);"

            "uint16_t   clamp(uint16_t x, uint16_t minVal, uint16_t maxVal);"
            "u16vec2  clamp(u16vec2  x, uint16_t minVal, uint16_t maxVal);"
            "u16vec3  clamp(u16vec3  x, uint16_t minVal, uint16_t maxVal);"
            "u16vec4  clamp(u16vec4  x, uint16_t minVal, uint16_t maxVal);"
            "u16vec2  clamp(u16vec2  x, u16vec2  minVal, u16vec2  maxVal);"
            "u16vec3  clamp(u16vec3  x, u16vec3  minVal, u16vec3  maxVal);"
            "u16vec4  clamp(u16vec4  x, u16vec4  minVal, u16vec4  maxVal);"

            "int16_t  mix(int16_t,  int16_t,  bool);"
            "i16vec2  mix(i16vec2,  i16vec2,  bvec2);"
            "i16vec3  mix(i16vec3,  i16vec3,  bvec3);"
            "i16vec4  mix(i16vec4,  i16vec4,  bvec4);"
            "uint16_t mix(uint16_t, uint16_t, bool);"
            "u16vec2  mix(u16vec2,  u16vec2,  bvec2);"
            "u16vec3  mix(u16vec3,  u16vec3,  bvec3);"
            "u16vec4  mix(u16vec4,  u16vec4,  bvec4);"

            "float16_t frexp(float16_t, out int16_t);"
            "f16vec2   frexp(f16vec2,   out i16vec2);"
            "f16vec3   frexp(f16vec3,   out i16vec3);"
            "f16vec4   frexp(f16vec4,   out i16vec4);"

            "float16_t ldexp(float16_t, int16_t);"
            "f16vec2   ldexp(f16vec2,   i16vec2);"
            "f16vec3   ldexp(f16vec3,   i16vec3);"
            "f16vec4   ldexp(f16vec4,   i16vec4);"

            "int16_t halfBitsToInt16(float16_t);"
            "i16vec2 halfBitsToInt16(f16vec2);"
            "i16vec3 halhBitsToInt16(f16vec3);"
            "i16vec4 halfBitsToInt16(f16vec4);"

            "uint16_t halfBitsToUint16(float16_t);"
            "u16vec2  halfBitsToUint16(f16vec2);"
            "u16vec3  halfBitsToUint16(f16vec3);"
            "u16vec4  halfBitsToUint16(f16vec4);"

            "int16_t float16BitsToInt16(float16_t);"
            "i16vec2 float16BitsToInt16(f16vec2);"
            "i16vec3 float16BitsToInt16(f16vec3);"
            "i16vec4 float16BitsToInt16(f16vec4);"

            "uint16_t float16BitsToUint16(float16_t);"
            "u16vec2  float16BitsToUint16(f16vec2);"
            "u16vec3  float16BitsToUint16(f16vec3);"
            "u16vec4  float16BitsToUint16(f16vec4);"

            "float16_t int16BitsToFloat16(int16_t);"
            "f16vec2   int16BitsToFloat16(i16vec2);"
            "f16vec3   int16BitsToFloat16(i16vec3);"
            "f16vec4   int16BitsToFloat16(i16vec4);"

            "float16_t uint16BitsToFloat16(uint16_t);"
            "f16vec2   uint16BitsToFloat16(u16vec2);"
            "f16vec3   uint16BitsToFloat16(u16vec3);"
            "f16vec4   uint16BitsToFloat16(u16vec4);"

            "float16_t int16BitsToHalf(int16_t);"
            "f16vec2   int16BitsToHalf(i16vec2);"
            "f16vec3   int16BitsToHalf(i16vec3);"
            "f16vec4   int16BitsToHalf(i16vec4);"

            "float16_t uint16BitsToHalf(uint16_t);"
            "f16vec2   uint16BitsToHalf(u16vec2);"
            "f16vec3   uint16BitsToHalf(u16vec3);"
            "f16vec4   uint16BitsToHalf(u16vec4);"

            "int      packInt2x16(i16vec2);"
            "uint     packUint2x16(u16vec2);"
            "int64_t  packInt4x16(i16vec4);"
            "uint64_t packUint4x16(u16vec4);"
            "i16vec2  unpackInt2x16(int);"
            "u16vec2  unpackUint2x16(uint);"
            "i16vec4  unpackInt4x16(int64_t);"
            "u16vec4  unpackUint4x16(uint64_t);"

            "bvec2 lessThan(i16vec2, i16vec2);"
            "bvec3 lessThan(i16vec3, i16vec3);"
            "bvec4 lessThan(i16vec4, i16vec4);"
            "bvec2 lessThan(u16vec2, u16vec2);"
            "bvec3 lessThan(u16vec3, u16vec3);"
            "bvec4 lessThan(u16vec4, u16vec4);"

            "bvec2 lessThanEqual(i16vec2, i16vec2);"
            "bvec3 lessThanEqual(i16vec3, i16vec3);"
            "bvec4 lessThanEqual(i16vec4, i16vec4);"
            "bvec2 lessThanEqual(u16vec2, u16vec2);"
            "bvec3 lessThanEqual(u16vec3, u16vec3);"
            "bvec4 lessThanEqual(u16vec4, u16vec4);"

            "bvec2 greaterThan(i16vec2, i16vec2);"
            "bvec3 greaterThan(i16vec3, i16vec3);"
            "bvec4 greaterThan(i16vec4, i16vec4);"
            "bvec2 greaterThan(u16vec2, u16vec2);"
            "bvec3 greaterThan(u16vec3, u16vec3);"
            "bvec4 greaterThan(u16vec4, u16vec4);"

            "bvec2 greaterThanEqual(i16vec2, i16vec2);"
            "bvec3 greaterThanEqual(i16vec3, i16vec3);"
            "bvec4 greaterThanEqual(i16vec4, i16vec4);"
            "bvec2 greaterThanEqual(u16vec2, u16vec2);"
            "bvec3 greaterThanEqual(u16vec3, u16vec3);"
            "bvec4 greaterThanEqual(u16vec4, u16vec4);"

            "bvec2 equal(i16vec2, i16vec2);"
            "bvec3 equal(i16vec3, i16vec3);"
            "bvec4 equal(i16vec4, i16vec4);"
            "bvec2 equal(u16vec2, u16vec2);"
            "bvec3 equal(u16vec3, u16vec3);"
            "bvec4 equal(u16vec4, u16vec4);"

            "bvec2 notEqual(i16vec2, i16vec2);"
            "bvec3 notEqual(i16vec3, i16vec3);"
            "bvec4 notEqual(i16vec4, i16vec4);"
            "bvec2 notEqual(u16vec2, u16vec2);"
            "bvec3 notEqual(u16vec3, u16vec3);"
            "bvec4 notEqual(u16vec4, u16vec4);"

            "  int16_t bitfieldExtract(  int16_t, int16_t, int16_t);"
            "i16vec2 bitfieldExtract(i16vec2, int16_t, int16_t);"
            "i16vec3 bitfieldExtract(i16vec3, int16_t, int16_t);"
            "i16vec4 bitfieldExtract(i16vec4, int16_t, int16_t);"

            " uint16_t bitfieldExtract( uint16_t, int16_t, int16_t);"
            "u16vec2 bitfieldExtract(u16vec2, int16_t, int16_t);"
            "u16vec3 bitfieldExtract(u16vec3, int16_t, int16_t);"
            "u16vec4 bitfieldExtract(u16vec4, int16_t, int16_t);"

            "  int16_t bitfieldInsert(  int16_t base,   int16_t, int16_t, int16_t);"
            "i16vec2 bitfieldInsert(i16vec2 base, i16vec2, int16_t, int16_t);"
            "i16vec3 bitfieldInsert(i16vec3 base, i16vec3, int16_t, int16_t);"
            "i16vec4 bitfieldInsert(i16vec4 base, i16vec4, int16_t, int16_t);"

            " uint16_t bitfieldInsert( uint16_t base,  uint16_t, int16_t, int16_t);"
            "u16vec2 bitfieldInsert(u16vec2 base, u16vec2, int16_t, int16_t);"
            "u16vec3 bitfieldInsert(u16vec3 base, u16vec3, int16_t, int16_t);"
            "u16vec4 bitfieldInsert(u16vec4 base, u16vec4, int16_t, int16_t);"

            "  int16_t bitCount(  int16_t);"
            "i16vec2 bitCount(i16vec2);"
            "i16vec3 bitCount(i16vec3);"
            "i16vec4 bitCount(i16vec4);"

            "  int16_t bitCount( uint16_t);"
            "i16vec2 bitCount(u16vec2);"
            "i16vec3 bitCount(u16vec3);"
            "i16vec4 bitCount(u16vec4);"

            "  int16_t findLSB(  int16_t);"
            "i16vec2 findLSB(i16vec2);"
            "i16vec3 findLSB(i16vec3);"
            "i16vec4 findLSB(i16vec4);"

            "  int16_t findLSB( uint16_t);"
            "i16vec2 findLSB(u16vec2);"
            "i16vec3 findLSB(u16vec3);"
            "i16vec4 findLSB(u16vec4);"

            "  int16_t findMSB(  int16_t);"
            "i16vec2 findMSB(i16vec2);"
            "i16vec3 findMSB(i16vec3);"
            "i16vec4 findMSB(i16vec4);"

            "  int16_t findMSB( uint16_t);"
            "i16vec2 findMSB(u16vec2);"
            "i16vec3 findMSB(u16vec3);"
            "i16vec4 findMSB(u16vec4);"

            "int16_t  pack16(i8vec2);"
            "uint16_t pack16(u8vec2);"
            "int32_t  pack32(i8vec4);"
            "uint32_t pack32(u8vec4);"
            "int32_t  pack32(i16vec2);"
            "uint32_t pack32(u16vec2);"
            "int64_t  pack64(i16vec4);"
            "uint64_t pack64(u16vec4);"
            "int64_t  pack64(i32vec2);"
            "uint64_t pack64(u32vec2);"

            "i8vec2   unpack8(int16_t);"
            "u8vec2   unpack8(uint16_t);"
            "i8vec4   unpack8(int32_t);"
            "u8vec4   unpack8(uint32_t);"
            "i16vec2  unpack16(int32_t);"
            "u16vec2  unpack16(uint32_t);"
            "i16vec4  unpack16(int64_t);"
            "u16vec4  unpack16(uint64_t);"
            "i32vec2  unpack32(int64_t);"
            "u32vec2  unpack32(uint64_t);"

            // GL_EXT_expect_assume
            "int8_t expectEXT(int8_t, int8_t);"
            "i8vec2 expectEXT(i8vec2, i8vec2);"
            "i8vec3 expectEXT(i8vec3, i8vec3);"
            "i8vec4 expectEXT(i8vec4, i8vec4);"

            "uint8_t expectEXT(uint8_t, uint8_t);"
            "u8vec2 expectEXT(u8vec2, u8vec2);"
            "u8vec3 expectEXT(u8vec3, u8vec3);"
            "u8vec4 expectEXT(u8vec4, u8vec4);"

            "int16_t expectEXT(int16_t, int16_t);"
            "i16vec2 expectEXT(i16vec2, i16vec2);"
            "i16vec3 expectEXT(i16vec3, i16vec3);"
            "i16vec4 expectEXT(i16vec4, i16vec4);"

            "uint16_t expectEXT(uint16_t, uint16_t);"
            "u16vec2 expectEXT(u16vec2, u16vec2);"
            "u16vec3 expectEXT(u16vec3, u16vec3);"
            "u16vec4 expectEXT(u16vec4, u16vec4);"

            "int64_t expectEXT(int64_t, int64_t);"
            "i64vec2 expectEXT(i64vec2, i64vec2);"
            "i64vec3 expectEXT(i64vec3, i64vec3);"
            "i64vec4 expectEXT(i64vec4, i64vec4);"

            "uint64_t expectEXT(uint64_t, uint64_t);"
            "u64vec2 expectEXT(u64vec2, u64vec2);"
            "u64vec3 expectEXT(u64vec3, u64vec3);"
            "u64vec4 expectEXT(u64vec4, u64vec4);"
            "\n");
    }

    // Builtins for GL_EXT_texture_shadow_lod
    if ((profile == EEsProfile && version >= 300) || ((profile != EEsProfile && version >= 130))) { 
        commonBuiltins.append(
            "float texture(sampler2DArrayShadow, vec4, float);"
            "float texture(samplerCubeArrayShadow, vec4, float, float);"
            "float textureLod(sampler2DArrayShadow, vec4, float);"
            "float textureLod(samplerCubeShadow, vec4, float);"
            "float textureLod(samplerCubeArrayShadow, vec4, float, float);"
            "float textureLodOffset(sampler2DArrayShadow, vec4, float, ivec2);"
            "float textureOffset(sampler2DArrayShadow, vec4 , ivec2, float);"
            "\n");
    }

    if (profile != EEsProfile && version >= 450) {
        stageBuiltins[EShLangFragment].append(derivativesAndControl64bits);
        stageBuiltins[EShLangFragment].append(
            "float64_t interpolateAtCentroid(float64_t);"
            "f64vec2   interpolateAtCentroid(f64vec2);"
            "f64vec3   interpolateAtCentroid(f64vec3);"
            "f64vec4   interpolateAtCentroid(f64vec4);"

            "float64_t interpolateAtSample(float64_t, int);"
            "f64vec2   interpolateAtSample(f64vec2,   int);"
            "f64vec3   interpolateAtSample(f64vec3,   int);"
            "f64vec4   interpolateAtSample(f64vec4,   int);"

            "float64_t interpolateAtOffset(float64_t, f64vec2);"
            "f64vec2   interpolateAtOffset(f64vec2,   f64vec2);"
            "f64vec3   interpolateAtOffset(f64vec3,   f64vec2);"
            "f64vec4   interpolateAtOffset(f64vec4,   f64vec2);"

            "\n");

    }

    // GL_EXT_expect_assume
    if ((profile == EEsProfile && version >= 310) ||
         ((profile != EEsProfile && version >= 140))) {
        commonBuiltins.append(
            "void assumeEXT(bool);"

            "bool expectEXT(bool, bool);"
            "bvec2 expectEXT(bvec2, bvec2);"
            "bvec3 expectEXT(bvec3, bvec3);"
            "bvec4 expectEXT(bvec4, bvec4);"

            "int expectEXT(int, int);"
            "ivec2 expectEXT(ivec2, ivec2);"
            "ivec3 expectEXT(ivec3, ivec3);"
            "ivec4 expectEXT(ivec4, ivec4);"

            "uint expectEXT(uint, uint);"
            "uvec2 expectEXT(uvec2, uvec2);"
            "uvec3 expectEXT(uvec3, uvec3);"
            "uvec4 expectEXT(uvec4, uvec4);"
            "\n");
    }

    // QCOM_image_processing
    if ((profile == EEsProfile && version >= 310) ||
         (profile != EEsProfile && version >= 140)) {
        commonBuiltins.append(
           "vec4 textureWeightedQCOM(sampler2D, vec2, sampler2DArray);"
           "vec4 textureWeightedQCOM(sampler2D, vec2, sampler1DArray);"
           "vec4 textureBoxFilterQCOM(sampler2D, vec2, vec2);"
           "vec4 textureBlockMatchSADQCOM(sampler2D, uvec2, sampler2D, uvec2, uvec2);"
           "vec4 textureBlockMatchSSDQCOM(sampler2D, uvec2, sampler2D, uvec2, uvec2);"

           "vec4 textureBlockMatchWindowSSDQCOM(sampler2D, uvec2, sampler2D, uvec2, uvec2);"
           "vec4 textureBlockMatchWindowSADQCOM(sampler2D, uvec2, sampler2D, uvec2, uvec2);"
           "vec4 textureBlockMatchGatherSSDQCOM(sampler2D, uvec2, sampler2D, uvec2, uvec2);"
           "vec4 textureBlockMatchGatherSADQCOM(sampler2D, uvec2, sampler2D, uvec2, uvec2);"
           "\n");
    }

    //============================================================================
    //
    // Prototypes for built-in functions seen by vertex shaders only.
    // (Except legacy lod functions, where it depends which release they are
    // vertex only.)
    //
    //============================================================================

    //
    // Geometric Functions.
    //
    if (spvVersion.vulkan == 0 && IncludeLegacy(version, profile, spvVersion))
        stageBuiltins[EShLangVertex].append("vec4 ftransform();");

    //
    // Original-style texture Functions with lod.
    //
    TString* s;
    if (version == 100)
        s = &stageBuiltins[EShLangVertex];
    else
        s = &commonBuiltins;
    if ((profile == EEsProfile && version == 100) ||
         profile == ECompatibilityProfile ||
        (profile == ECoreProfile && version < 420) ||
         profile == ENoProfile) {
        if (spvVersion.spv == 0) {
            s->append(
                "vec4 texture2DLod(sampler2D, vec2, float);"         // GL_ARB_shader_texture_lod
                "vec4 texture2DProjLod(sampler2D, vec3, float);"     // GL_ARB_shader_texture_lod
                "vec4 texture2DProjLod(sampler2D, vec4, float);"     // GL_ARB_shader_texture_lod
                "vec4 texture3DLod(sampler3D, vec3, float);"         // GL_ARB_shader_texture_lod  // OES_texture_3D, but caught by keyword check
                "vec4 texture3DProjLod(sampler3D, vec4, float);"     // GL_ARB_shader_texture_lod  // OES_texture_3D, but caught by keyword check
                "vec4 textureCubeLod(samplerCube, vec3, float);"     // GL_ARB_shader_texture_lod

                "\n");
        }
    }
    if ( profile == ECompatibilityProfile ||
        (profile == ECoreProfile && version < 420) ||
         profile == ENoProfile) {
        if (spvVersion.spv == 0) {
            s->append(
                "vec4 texture1DLod(sampler1D, float, float);"                          // GL_ARB_shader_texture_lod
                "vec4 texture1DProjLod(sampler1D, vec2, float);"                       // GL_ARB_shader_texture_lod
                "vec4 texture1DProjLod(sampler1D, vec4, float);"                       // GL_ARB_shader_texture_lod
                "vec4 shadow1DLod(sampler1DShadow, vec3, float);"                      // GL_ARB_shader_texture_lod
                "vec4 shadow2DLod(sampler2DShadow, vec3, float);"                      // GL_ARB_shader_texture_lod
                "vec4 shadow1DProjLod(sampler1DShadow, vec4, float);"                  // GL_ARB_shader_texture_lod
                "vec4 shadow2DProjLod(sampler2DShadow, vec4, float);"                  // GL_ARB_shader_texture_lod

                "vec4 texture1DGradARB(sampler1D, float, float, float);"               // GL_ARB_shader_texture_lod
                "vec4 texture1DProjGradARB(sampler1D, vec2, float, float);"            // GL_ARB_shader_texture_lod
                "vec4 texture1DProjGradARB(sampler1D, vec4, float, float);"            // GL_ARB_shader_texture_lod
                "vec4 texture2DGradARB(sampler2D, vec2, vec2, vec2);"                  // GL_ARB_shader_texture_lod
                "vec4 texture2DProjGradARB(sampler2D, vec3, vec2, vec2);"              // GL_ARB_shader_texture_lod
                "vec4 texture2DProjGradARB(sampler2D, vec4, vec2, vec2);"              // GL_ARB_shader_texture_lod
                "vec4 texture3DGradARB(sampler3D, vec3, vec3, vec3);"                  // GL_ARB_shader_texture_lod
                "vec4 texture3DProjGradARB(sampler3D, vec4, vec3, vec3);"              // GL_ARB_shader_texture_lod
                "vec4 textureCubeGradARB(samplerCube, vec3, vec3, vec3);"              // GL_ARB_shader_texture_lod
                "vec4 shadow1DGradARB(sampler1DShadow, vec3, float, float);"           // GL_ARB_shader_texture_lod
                "vec4 shadow1DProjGradARB( sampler1DShadow, vec4, float, float);"      // GL_ARB_shader_texture_lod
                "vec4 shadow2DGradARB(sampler2DShadow, vec3, vec2, vec2);"             // GL_ARB_shader_texture_lod
                "vec4 shadow2DProjGradARB( sampler2DShadow, vec4, vec2, vec2);"        // GL_ARB_shader_texture_lod
                "vec4 texture2DRectGradARB(sampler2DRect, vec2, vec2, vec2);"          // GL_ARB_shader_texture_lod
                "vec4 texture2DRectProjGradARB( sampler2DRect, vec3, vec2, vec2);"     // GL_ARB_shader_texture_lod
                "vec4 texture2DRectProjGradARB( sampler2DRect, vec4, vec2, vec2);"     // GL_ARB_shader_texture_lod
                "vec4 shadow2DRectGradARB( sampler2DRectShadow, vec3, vec2, vec2);"    // GL_ARB_shader_texture_lod
                "vec4 shadow2DRectProjGradARB(sampler2DRectShadow, vec4, vec2, vec2);" // GL_ARB_shader_texture_lod

                "\n");
        }
    }

    if ((profile != EEsProfile && version >= 150) ||
        (profile == EEsProfile && version >= 310)) {
        //============================================================================
        //
        // Prototypes for built-in functions seen by geometry shaders only.
        //
        //============================================================================

        if (profile != EEsProfile && (version >= 400 || version == 150)) {
            stageBuiltins[EShLangGeometry].append(
                "void EmitStreamVertex(int);"
                "void EndStreamPrimitive(int);"
                );
        }
        stageBuiltins[EShLangGeometry].append(
            "void EmitVertex();"
            "void EndPrimitive();"
            "\n");
    }

    //============================================================================
    //
    // Prototypes for all control functions.
    //
    //============================================================================
    bool esBarrier = (profile == EEsProfile && version >= 310);
    if ((profile != EEsProfile && version >= 150) || esBarrier)
        stageBuiltins[EShLangTessControl].append(
            "void barrier();"
            );
    if ((profile != EEsProfile && version >= 420) || esBarrier)
        stageBuiltins[EShLangCompute].append(
            "void barrier();"
            );
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        stageBuiltins[EShLangMesh].append(
            "void barrier();"
            );
        stageBuiltins[EShLangTask].append(
            "void barrier();"
            );
    }
    if ((profile != EEsProfile && version >= 130) || esBarrier)
        commonBuiltins.append(
            "void memoryBarrier();"
            );
    if ((profile != EEsProfile && version >= 420) || esBarrier) {
        commonBuiltins.append(
            "void memoryBarrierBuffer();"
            );
        stageBuiltins[EShLangCompute].append(
            "void memoryBarrierShared();"
            "void groupMemoryBarrier();"
            );
    }
    if ((profile != EEsProfile && version >= 420) || esBarrier) {
        if (spvVersion.vulkan == 0 || spvVersion.vulkanRelaxed) {
            commonBuiltins.append("void memoryBarrierAtomicCounter();");
        }
        commonBuiltins.append("void memoryBarrierImage();");
    }
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        stageBuiltins[EShLangMesh].append(
            "void memoryBarrierShared();"
            "void groupMemoryBarrier();"
        );
        stageBuiltins[EShLangTask].append(
            "void memoryBarrierShared();"
            "void groupMemoryBarrier();"
        );
    }

    commonBuiltins.append("void controlBarrier(int, int, int, int);\n"
                          "void memoryBarrier(int, int, int);\n");

    commonBuiltins.append("void debugPrintfEXT();\n");

    if (profile != EEsProfile && version >= 450) {
        // coopMatStoreNV perhaps ought to have "out" on the buf parameter, but
        // adding it introduces undesirable tempArgs on the stack. What we want
        // is more like "buf" thought of as a pointer value being an in parameter.
        stageBuiltins[EShLangCompute].append(
            "void coopMatLoadNV(out fcoopmatNV m, volatile coherent float16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out fcoopmatNV m, volatile coherent float[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out fcoopmatNV m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out fcoopmatNV m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out fcoopmatNV m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out fcoopmatNV m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out fcoopmatNV m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out fcoopmatNV m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "void coopMatStoreNV(fcoopmatNV m, volatile coherent float16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(fcoopmatNV m, volatile coherent float[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(fcoopmatNV m, volatile coherent float64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(fcoopmatNV m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(fcoopmatNV m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(fcoopmatNV m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(fcoopmatNV m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(fcoopmatNV m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(fcoopmatNV m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "fcoopmatNV coopMatMulAddNV(fcoopmatNV A, fcoopmatNV B, fcoopmatNV C);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent int8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent int16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent int[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent int64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent ivec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent ivec4[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out icoopmatNV m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent int8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent int16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent int[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent int64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent ivec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent ivec4[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadNV(out ucoopmatNV m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "void coopMatStoreNV(icoopmatNV m, volatile coherent int8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent int16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent int[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent int64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent ivec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent ivec4[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(icoopmatNV m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "void coopMatStoreNV(ucoopmatNV m, volatile coherent int8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent int16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent int[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent int64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent ivec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent ivec4[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreNV(ucoopmatNV m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "icoopmatNV coopMatMulAddNV(icoopmatNV A, icoopmatNV B, icoopmatNV C);\n"
            "ucoopmatNV coopMatMulAddNV(ucoopmatNV A, ucoopmatNV B, ucoopmatNV C);\n"
            );

        std::string cooperativeMatrixFuncs =
            "void coopMatLoad(out coopmat m, volatile coherent int8_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent int16_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent int32_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent int64_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent uint8_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent uint16_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent uint32_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent uint64_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent float16_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent float[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent float64_t[] buf, uint element, uint stride, int matrixLayout);\n"

            "void coopMatLoad(out coopmat m, volatile coherent i8vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent i16vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent i32vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent i64vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent u8vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent u16vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent u32vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent u64vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent f16vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent f32vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent f64vec2[] buf, uint element, uint stride, int matrixLayout);\n"

            "void coopMatLoad(out coopmat m, volatile coherent i8vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent i16vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent i32vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent i64vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent u8vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent u16vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent u32vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent u64vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent f16vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent f32vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatLoad(out coopmat m, volatile coherent f64vec4[] buf, uint element, uint stride, int matrixLayout);\n"

            "void coopMatStore(coopmat m, volatile coherent int8_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent int16_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent int32_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent int64_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent uint8_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent uint16_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent uint32_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent uint64_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent float16_t[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent float[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent float64_t[] buf, uint element, uint stride, int matrixLayout);\n"

            "void coopMatStore(coopmat m, volatile coherent i8vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent i16vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent i32vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent i64vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent u8vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent u16vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent u32vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent u64vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent f16vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent f32vec2[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent f64vec2[] buf, uint element, uint stride, int matrixLayout);\n"

            "void coopMatStore(coopmat m, volatile coherent i8vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent i16vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent i32vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent i64vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent u8vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent u16vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent u32vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent u64vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent f16vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent f32vec4[] buf, uint element, uint stride, int matrixLayout);\n"
            "void coopMatStore(coopmat m, volatile coherent f64vec4[] buf, uint element, uint stride, int matrixLayout);\n"

            "coopmat coopMatMulAdd(coopmat A, coopmat B, coopmat C);\n"
            "coopmat coopMatMulAdd(coopmat A, coopmat B, coopmat C, int matrixOperands);\n";

        commonBuiltins.append(cooperativeMatrixFuncs.c_str());

        commonBuiltins.append(
            "const int gl_MatrixUseA = 0;\n"
            "const int gl_MatrixUseB = 1;\n"
            "const int gl_MatrixUseAccumulator = 2;\n"
            "const int gl_MatrixOperandsSaturatingAccumulation = 0x10;\n"
            "const int gl_CooperativeMatrixLayoutRowMajor = 0;\n"
            "const int gl_CooperativeMatrixLayoutColumnMajor = 1;\n"
            "\n"
            );
    }

    //============================================================================
    //
    // Prototypes for built-in functions seen by fragment shaders only.
    //
    //============================================================================

    //
    // Original-style texture Functions with bias.
    //
    if (spvVersion.spv == 0 && (profile != EEsProfile || version == 100)) {
        stageBuiltins[EShLangFragment].append(
            "vec4 texture2D(sampler2D, vec2, float);"
            "vec4 texture2DProj(sampler2D, vec3, float);"
            "vec4 texture2DProj(sampler2D, vec4, float);"
            "vec4 texture3D(sampler3D, vec3, float);"        // OES_texture_3D
            "vec4 texture3DProj(sampler3D, vec4, float);"    // OES_texture_3D
            "vec4 textureCube(samplerCube, vec3, float);"

            "\n");
    }
    if (spvVersion.spv == 0 && (profile != EEsProfile && version > 100)) {
        stageBuiltins[EShLangFragment].append(
            "vec4 texture1D(sampler1D, float, float);"
            "vec4 texture1DProj(sampler1D, vec2, float);"
            "vec4 texture1DProj(sampler1D, vec4, float);"
            "vec4 shadow1D(sampler1DShadow, vec3, float);"
            "vec4 shadow2D(sampler2DShadow, vec3, float);"
            "vec4 shadow1DProj(sampler1DShadow, vec4, float);"
            "vec4 shadow2DProj(sampler2DShadow, vec4, float);"

            "\n");
    }
    if (spvVersion.spv == 0 && profile == EEsProfile) {
        stageBuiltins[EShLangFragment].append(
            "vec4 texture2DLodEXT(sampler2D, vec2, float);"      // GL_EXT_shader_texture_lod
            "vec4 texture2DProjLodEXT(sampler2D, vec3, float);"  // GL_EXT_shader_texture_lod
            "vec4 texture2DProjLodEXT(sampler2D, vec4, float);"  // GL_EXT_shader_texture_lod
            "vec4 textureCubeLodEXT(samplerCube, vec3, float);"  // GL_EXT_shader_texture_lod

            "\n");
    }

    // GL_EXT_shader_tile_image
    if (spvVersion.vulkan > 0) {
        stageBuiltins[EShLangFragment].append(
            "lowp uint stencilAttachmentReadEXT();"
            "lowp uint stencilAttachmentReadEXT(int);"
            "highp float depthAttachmentReadEXT();"
            "highp float depthAttachmentReadEXT(int);"
            "\n");
        stageBuiltins[EShLangFragment].append(
            "vec4 colorAttachmentReadEXT(attachmentEXT);"
            "vec4 colorAttachmentReadEXT(attachmentEXT, int);"
            "ivec4 colorAttachmentReadEXT(iattachmentEXT);"
            "ivec4 colorAttachmentReadEXT(iattachmentEXT, int);"
            "uvec4 colorAttachmentReadEXT(uattachmentEXT);"
            "uvec4 colorAttachmentReadEXT(uattachmentEXT, int);"
            "\n");
    }

    // GL_ARB_derivative_control
    if (profile != EEsProfile && version >= 400) {
        stageBuiltins[EShLangFragment].append(derivativeControls);
        stageBuiltins[EShLangFragment].append("\n");
    }

    // GL_OES_shader_multisample_interpolation
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 400)) {
        stageBuiltins[EShLangFragment].append(
            "float interpolateAtCentroid(float);"
            "vec2  interpolateAtCentroid(vec2);"
            "vec3  interpolateAtCentroid(vec3);"
            "vec4  interpolateAtCentroid(vec4);"

            "float interpolateAtSample(float, int);"
            "vec2  interpolateAtSample(vec2,  int);"
            "vec3  interpolateAtSample(vec3,  int);"
            "vec4  interpolateAtSample(vec4,  int);"

            "float interpolateAtOffset(float, vec2);"
            "vec2  interpolateAtOffset(vec2,  vec2);"
            "vec3  interpolateAtOffset(vec3,  vec2);"
            "vec4  interpolateAtOffset(vec4,  vec2);"

            "\n");
    }

    stageBuiltins[EShLangFragment].append(
        "void beginInvocationInterlockARB(void);"
        "void endInvocationInterlockARB(void);");

    stageBuiltins[EShLangFragment].append(
        "bool helperInvocationEXT();"
        "\n");

    // GL_AMD_shader_explicit_vertex_parameter
    if (profile != EEsProfile && version >= 450) {
        stageBuiltins[EShLangFragment].append(
            "float interpolateAtVertexAMD(float, uint);"
            "vec2  interpolateAtVertexAMD(vec2,  uint);"
            "vec3  interpolateAtVertexAMD(vec3,  uint);"
            "vec4  interpolateAtVertexAMD(vec4,  uint);"

            "int   interpolateAtVertexAMD(int,   uint);"
            "ivec2 interpolateAtVertexAMD(ivec2, uint);"
            "ivec3 interpolateAtVertexAMD(ivec3, uint);"
            "ivec4 interpolateAtVertexAMD(ivec4, uint);"

            "uint  interpolateAtVertexAMD(uint,  uint);"
            "uvec2 interpolateAtVertexAMD(uvec2, uint);"
            "uvec3 interpolateAtVertexAMD(uvec3, uint);"
            "uvec4 interpolateAtVertexAMD(uvec4, uint);"

            "float16_t interpolateAtVertexAMD(float16_t, uint);"
            "f16vec2   interpolateAtVertexAMD(f16vec2,   uint);"
            "f16vec3   interpolateAtVertexAMD(f16vec3,   uint);"
            "f16vec4   interpolateAtVertexAMD(f16vec4,   uint);"

            "\n");
    }

    // GL_AMD_gpu_shader_half_float
    if (profile != EEsProfile && version >= 450) {
        stageBuiltins[EShLangFragment].append(derivativesAndControl16bits);
        stageBuiltins[EShLangFragment].append("\n");

        stageBuiltins[EShLangFragment].append(
            "float16_t interpolateAtCentroid(float16_t);"
            "f16vec2   interpolateAtCentroid(f16vec2);"
            "f16vec3   interpolateAtCentroid(f16vec3);"
            "f16vec4   interpolateAtCentroid(f16vec4);"

            "float16_t interpolateAtSample(float16_t, int);"
            "f16vec2   interpolateAtSample(f16vec2,   int);"
            "f16vec3   interpolateAtSample(f16vec3,   int);"
            "f16vec4   interpolateAtSample(f16vec4,   int);"

            "float16_t interpolateAtOffset(float16_t, f16vec2);"
            "f16vec2   interpolateAtOffset(f16vec2,   f16vec2);"
            "f16vec3   interpolateAtOffset(f16vec3,   f16vec2);"
            "f16vec4   interpolateAtOffset(f16vec4,   f16vec2);"

            "\n");
    }

    // GL_ARB_shader_clock& GL_EXT_shader_realtime_clock
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(
            "uvec2 clock2x32ARB();"
            "uint64_t clockARB();"
            "uvec2 clockRealtime2x32EXT();"
            "uint64_t clockRealtimeEXT();"
            "\n");
    }

    // GL_AMD_shader_fragment_mask
    if (profile != EEsProfile && version >= 450 && spvVersion.vulkan > 0) {
        stageBuiltins[EShLangFragment].append(
            "uint fragmentMaskFetchAMD(subpassInputMS);"
            "uint fragmentMaskFetchAMD(isubpassInputMS);"
            "uint fragmentMaskFetchAMD(usubpassInputMS);"

            "vec4  fragmentFetchAMD(subpassInputMS,  uint);"
            "ivec4 fragmentFetchAMD(isubpassInputMS, uint);"
            "uvec4 fragmentFetchAMD(usubpassInputMS, uint);"

            "\n");
    }

    // Builtins for GL_NV_ray_tracing/GL_NV_ray_tracing_motion_blur/GL_EXT_ray_tracing/GL_EXT_ray_query/
    // GL_NV_shader_invocation_reorder/GL_KHR_ray_tracing_position_Fetch
    if (profile != EEsProfile && version >= 460) {
         commonBuiltins.append("void rayQueryInitializeEXT(rayQueryEXT, accelerationStructureEXT, uint, uint, vec3, float, vec3, float);"
            "void rayQueryTerminateEXT(rayQueryEXT);"
            "void rayQueryGenerateIntersectionEXT(rayQueryEXT, float);"
            "void rayQueryConfirmIntersectionEXT(rayQueryEXT);"
            "bool rayQueryProceedEXT(rayQueryEXT);"
            "uint rayQueryGetIntersectionTypeEXT(rayQueryEXT, bool);"
            "float rayQueryGetRayTMinEXT(rayQueryEXT);"
            "uint rayQueryGetRayFlagsEXT(rayQueryEXT);"
            "vec3 rayQueryGetWorldRayOriginEXT(rayQueryEXT);"
            "vec3 rayQueryGetWorldRayDirectionEXT(rayQueryEXT);"
            "float rayQueryGetIntersectionTEXT(rayQueryEXT, bool);"
            "int rayQueryGetIntersectionInstanceCustomIndexEXT(rayQueryEXT, bool);"
            "int rayQueryGetIntersectionInstanceIdEXT(rayQueryEXT, bool);"
            "uint rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rayQueryEXT, bool);"
            "int rayQueryGetIntersectionGeometryIndexEXT(rayQueryEXT, bool);"
            "int rayQueryGetIntersectionPrimitiveIndexEXT(rayQueryEXT, bool);"
            "vec2 rayQueryGetIntersectionBarycentricsEXT(rayQueryEXT, bool);"
            "bool rayQueryGetIntersectionFrontFaceEXT(rayQueryEXT, bool);"
            "bool rayQueryGetIntersectionCandidateAABBOpaqueEXT(rayQueryEXT);"
            "vec3 rayQueryGetIntersectionObjectRayDirectionEXT(rayQueryEXT, bool);"
            "vec3 rayQueryGetIntersectionObjectRayOriginEXT(rayQueryEXT, bool);"
            "mat4x3 rayQueryGetIntersectionObjectToWorldEXT(rayQueryEXT, bool);"
            "mat4x3 rayQueryGetIntersectionWorldToObjectEXT(rayQueryEXT, bool);"
            "void rayQueryGetIntersectionTriangleVertexPositionsEXT(rayQueryEXT, bool, out vec3[3]);"
            "\n");

        stageBuiltins[EShLangRayGen].append(
            "void traceNV(accelerationStructureNV,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void traceRayMotionNV(accelerationStructureNV,uint,uint,uint,uint,uint,vec3,float,vec3,float,float,int);"
            "void traceRayEXT(accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void executeCallableNV(uint, int);"
            "void executeCallableEXT(uint, int);"
            "void hitObjectTraceRayNV(hitObjectNV,accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void hitObjectTraceRayMotionNV(hitObjectNV,accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,float,int);"
            "void hitObjectRecordHitNV(hitObjectNV,accelerationStructureEXT,int,int,int,uint,uint,uint,vec3,float,vec3,float,int);"
            "void hitObjectRecordHitMotionNV(hitObjectNV,accelerationStructureEXT,int,int,int,uint,uint,uint,vec3,float,vec3,float,float,int);"
            "void hitObjectRecordHitWithIndexNV(hitObjectNV, accelerationStructureEXT,int,int,int,uint,uint,vec3,float,vec3,float,int);"
            "void hitObjectRecordHitWithIndexMotionNV(hitObjectNV, accelerationStructureEXT,int,int,int,uint,uint,vec3,float,vec3,float,float,int);"
            "void hitObjectRecordMissNV(hitObjectNV,uint,vec3,float,vec3,float);"
            "void hitObjectRecordMissMotionNV(hitObjectNV,uint,vec3,float,vec3,float,float);"
            "void hitObjectRecordEmptyNV(hitObjectNV);"
            "void hitObjectExecuteShaderNV(hitObjectNV,int);"
            "bool hitObjectIsEmptyNV(hitObjectNV);"
            "bool hitObjectIsMissNV(hitObjectNV);"
            "bool hitObjectIsHitNV(hitObjectNV);"
            "float hitObjectGetRayTMinNV(hitObjectNV);"
            "float hitObjectGetRayTMaxNV(hitObjectNV);"
            "vec3 hitObjectGetWorldRayOriginNV(hitObjectNV);"
            "vec3 hitObjectGetWorldRayDirectionNV(hitObjectNV);"
            "vec3 hitObjectGetObjectRayOriginNV(hitObjectNV);"
            "vec3 hitObjectGetObjectRayDirectionNV(hitObjectNV);"
            "mat4x3 hitObjectGetWorldToObjectNV(hitObjectNV);"
            "mat4x3 hitObjectGetObjectToWorldNV(hitObjectNV);"
            "int hitObjectGetInstanceCustomIndexNV(hitObjectNV);"
            "int hitObjectGetInstanceIdNV(hitObjectNV);"
            "int hitObjectGetGeometryIndexNV(hitObjectNV);"
            "int hitObjectGetPrimitiveIndexNV(hitObjectNV);"
            "uint hitObjectGetHitKindNV(hitObjectNV);"
            "void hitObjectGetAttributesNV(hitObjectNV,int);"
            "float hitObjectGetCurrentTimeNV(hitObjectNV);"
            "uint hitObjectGetShaderBindingTableRecordIndexNV(hitObjectNV);"
            "uvec2 hitObjectGetShaderRecordBufferHandleNV(hitObjectNV);"
            "void reorderThreadNV(uint, uint);"
            "void reorderThreadNV(hitObjectNV);"
            "void reorderThreadNV(hitObjectNV, uint, uint);"
            "vec3 fetchMicroTriangleVertexPositionNV(accelerationStructureEXT, int, int, int, ivec2);"
            "vec2 fetchMicroTriangleVertexBarycentricNV(accelerationStructureEXT, int, int, int, ivec2);"
            "\n");
        stageBuiltins[EShLangIntersect].append(
            "bool reportIntersectionNV(float, uint);"
            "bool reportIntersectionEXT(float, uint);"
            "\n");
        stageBuiltins[EShLangAnyHit].append(
            "void ignoreIntersectionNV();"
            "void terminateRayNV();"
            "\n");
        stageBuiltins[EShLangClosestHit].append(
            "void traceNV(accelerationStructureNV,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void traceRayMotionNV(accelerationStructureNV,uint,uint,uint,uint,uint,vec3,float,vec3,float,float,int);"
            "void traceRayEXT(accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void executeCallableNV(uint, int);"
            "void executeCallableEXT(uint, int);"
            "void hitObjectTraceRayNV(hitObjectNV,accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void hitObjectTraceRayMotionNV(hitObjectNV,accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,float,int);"
            "void hitObjectRecordHitNV(hitObjectNV,accelerationStructureEXT,int,int,int,uint,uint,uint,vec3,float,vec3,float,int);"
            "void hitObjectRecordHitMotionNV(hitObjectNV,accelerationStructureEXT,int,int,int,uint,uint,uint,vec3,float,vec3,float,float,int);"
            "void hitObjectRecordHitWithIndexNV(hitObjectNV,accelerationStructureEXT,int,int,int,uint,uint,vec3,float,vec3,float,int);"
            "void hitObjectRecordHitWithIndexMotionNV(hitObjectNV, accelerationStructureEXT,int,int,int,uint,uint,vec3,float,vec3,float,float,int);"
            "void hitObjectRecordMissNV(hitObjectNV, uint, vec3, float, vec3, float);"
            "void hitObjectRecordMissMotionNV(hitObjectNV,uint,vec3,float,vec3,float,float);"
            "void hitObjectRecordEmptyNV(hitObjectNV);"
            "void hitObjectExecuteShaderNV(hitObjectNV, int);"
            "bool hitObjectIsEmptyNV(hitObjectNV);"
            "bool hitObjectIsMissNV(hitObjectNV);"
            "bool hitObjectIsHitNV(hitObjectNV);"
            "float hitObjectGetRayTMinNV(hitObjectNV);"
            "float hitObjectGetRayTMaxNV(hitObjectNV);"
            "vec3 hitObjectGetWorldRayOriginNV(hitObjectNV);"
            "vec3 hitObjectGetWorldRayDirectionNV(hitObjectNV);"
            "vec3 hitObjectGetObjectRayOriginNV(hitObjectNV);"
            "vec3 hitObjectGetObjectRayDirectionNV(hitObjectNV);"
            "mat4x3 hitObjectGetWorldToObjectNV(hitObjectNV);"
            "mat4x3 hitObjectGetObjectToWorldNV(hitObjectNV);"
            "int hitObjectGetInstanceCustomIndexNV(hitObjectNV);"
            "int hitObjectGetInstanceIdNV(hitObjectNV);"
            "int hitObjectGetGeometryIndexNV(hitObjectNV);"
            "int hitObjectGetPrimitiveIndexNV(hitObjectNV);"
            "uint hitObjectGetHitKindNV(hitObjectNV);"
            "void hitObjectGetAttributesNV(hitObjectNV,int);"
            "float hitObjectGetCurrentTimeNV(hitObjectNV);"
            "uint hitObjectGetShaderBindingTableRecordIndexNV(hitObjectNV);"
            "uvec2 hitObjectGetShaderRecordBufferHandleNV(hitObjectNV);"
            "\n");
        stageBuiltins[EShLangMiss].append(
            "void traceNV(accelerationStructureNV,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void traceRayMotionNV(accelerationStructureNV,uint,uint,uint,uint,uint,vec3,float,vec3,float,float,int);"
            "void traceRayEXT(accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void executeCallableNV(uint, int);"
            "void executeCallableEXT(uint, int);"
            "void hitObjectTraceRayNV(hitObjectNV,accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void hitObjectTraceRayMotionNV(hitObjectNV,accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,float,int);"
            "void hitObjectRecordHitNV(hitObjectNV,accelerationStructureEXT,int,int,int,uint,uint,uint,vec3,float,vec3,float,int);"
            "void hitObjectRecordHitMotionNV(hitObjectNV,accelerationStructureEXT,int,int,int,uint,uint,uint,vec3,float,vec3,float,float,int);"
            "void hitObjectRecordHitWithIndexNV(hitObjectNV,accelerationStructureEXT,int,int,int,uint,uint,vec3,float,vec3,float,int);"
            "void hitObjectRecordHitWithIndexMotionNV(hitObjectNV, accelerationStructureEXT,int,int,int,uint,uint,vec3,float,vec3,float,float,int);"
            "void hitObjectRecordMissNV(hitObjectNV, uint, vec3, float, vec3, float);"
            "void hitObjectRecordMissMotionNV(hitObjectNV,uint,vec3,float,vec3,float,float);"
            "void hitObjectRecordEmptyNV(hitObjectNV);"
            "void hitObjectExecuteShaderNV(hitObjectNV, int);"
            "bool hitObjectIsEmptyNV(hitObjectNV);"
            "bool hitObjectIsMissNV(hitObjectNV);"
            "bool hitObjectIsHitNV(hitObjectNV);"
            "float hitObjectGetRayTMinNV(hitObjectNV);"
            "float hitObjectGetRayTMaxNV(hitObjectNV);"
            "vec3 hitObjectGetWorldRayOriginNV(hitObjectNV);"
            "vec3 hitObjectGetWorldRayDirectionNV(hitObjectNV);"
            "vec3 hitObjectGetObjectRayOriginNV(hitObjectNV);"
            "vec3 hitObjectGetObjectRayDirectionNV(hitObjectNV);"
            "mat4x3 hitObjectGetWorldToObjectNV(hitObjectNV);"
            "mat4x3 hitObjectGetObjectToWorldNV(hitObjectNV);"
            "int hitObjectGetInstanceCustomIndexNV(hitObjectNV);"
            "int hitObjectGetInstanceIdNV(hitObjectNV);"
            "int hitObjectGetGeometryIndexNV(hitObjectNV);"
            "int hitObjectGetPrimitiveIndexNV(hitObjectNV);"
            "uint hitObjectGetHitKindNV(hitObjectNV);"
            "void hitObjectGetAttributesNV(hitObjectNV,int);"
            "float hitObjectGetCurrentTimeNV(hitObjectNV);"
            "uint hitObjectGetShaderBindingTableRecordIndexNV(hitObjectNV);"
            "uvec2 hitObjectGetShaderRecordBufferHandleNV(hitObjectNV);"
            "\n");
        stageBuiltins[EShLangCallable].append(
            "void executeCallableNV(uint, int);"
            "void executeCallableEXT(uint, int);"
            "\n");
    }

    //E_SPV_NV_compute_shader_derivatives
    if ((profile == EEsProfile && version >= 320) || (profile != EEsProfile && version >= 450)) {
        stageBuiltins[EShLangCompute].append(derivativeControls);
        stageBuiltins[EShLangCompute].append("\n");
    }
    if (profile != EEsProfile && version >= 450) {
        stageBuiltins[EShLangCompute].append(derivativesAndControl16bits);
        stageBuiltins[EShLangCompute].append(derivativesAndControl64bits);
        stageBuiltins[EShLangCompute].append("\n");
    }

    // Builtins for GL_NV_mesh_shader
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        stageBuiltins[EShLangMesh].append(
            "void writePackedPrimitiveIndices4x8NV(uint, uint);"
            "\n");
    }
    // Builtins for GL_EXT_mesh_shader
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        // Builtins for GL_EXT_mesh_shader
        stageBuiltins[EShLangTask].append(
            "void EmitMeshTasksEXT(uint, uint, uint);"
            "\n");

        stageBuiltins[EShLangMesh].append(
            "void SetMeshOutputsEXT(uint, uint);"
            "\n");
    }
    // Builtins for GL_NV_displacement_micromap
    if ((profile != EEsProfile && version >= 460) || (profile == EEsProfile && version >= 320)) {
        stageBuiltins[EShLangMesh].append(
            "vec3 fetchMicroTriangleVertexPositionNV(accelerationStructureEXT, int, int, int, ivec2);"
            "vec2 fetchMicroTriangleVertexBarycentricNV(accelerationStructureEXT, int, int, int, ivec2);"
            "\n");

        stageBuiltins[EShLangCompute].append(
            "vec3 fetchMicroTriangleVertexPositionNV(accelerationStructureEXT, int, int, int, ivec2);"
            "vec2 fetchMicroTriangleVertexBarycentricNV(accelerationStructureEXT, int, int, int, ivec2);"
            "\n");

    }


    //============================================================================
    //
    // Standard Uniforms
    //
    //============================================================================

    //
    // Depth range in window coordinates, p. 33
    //
    if (spvVersion.spv == 0) {
        commonBuiltins.append(
            "struct gl_DepthRangeParameters {"
            );
        if (profile == EEsProfile) {
            commonBuiltins.append(
                "highp float near;"   // n
                "highp float far;"    // f
                "highp float diff;"   // f - n
                );
        } else {
            commonBuiltins.append(
                "float near;"  // n
                "float far;"   // f
                "float diff;"  // f - n
                );
        }

        commonBuiltins.append(
            "};"
            "uniform gl_DepthRangeParameters gl_DepthRange;"
            "\n");
    }

    if (spvVersion.spv == 0 && IncludeLegacy(version, profile, spvVersion)) {
        //
        // Matrix state. p. 31, 32, 37, 39, 40.
        //
        commonBuiltins.append(
            "uniform mat4  gl_ModelViewMatrix;"
            "uniform mat4  gl_ProjectionMatrix;"
            "uniform mat4  gl_ModelViewProjectionMatrix;"

            //
            // Derived matrix state that provides inverse and transposed versions
            // of the matrices above.
            //
            "uniform mat3  gl_NormalMatrix;"

            "uniform mat4  gl_ModelViewMatrixInverse;"
            "uniform mat4  gl_ProjectionMatrixInverse;"
            "uniform mat4  gl_ModelViewProjectionMatrixInverse;"

            "uniform mat4  gl_ModelViewMatrixTranspose;"
            "uniform mat4  gl_ProjectionMatrixTranspose;"
            "uniform mat4  gl_ModelViewProjectionMatrixTranspose;"

            "uniform mat4  gl_ModelViewMatrixInverseTranspose;"
            "uniform mat4  gl_ProjectionMatrixInverseTranspose;"
            "uniform mat4  gl_ModelViewProjectionMatrixInverseTranspose;"

            //
            // Normal scaling p. 39.
            //
            "uniform float gl_NormalScale;"

            //
            // Point Size, p. 66, 67.
            //
            "struct gl_PointParameters {"
                "float size;"
                "float sizeMin;"
                "float sizeMax;"
                "float fadeThresholdSize;"
                "float distanceConstantAttenuation;"
                "float distanceLinearAttenuation;"
                "float distanceQuadraticAttenuation;"
            "};"

            "uniform gl_PointParameters gl_Point;"

            //
            // Material State p. 50, 55.
            //
            "struct gl_MaterialParameters {"
                "vec4  emission;"    // Ecm
                "vec4  ambient;"     // Acm
                "vec4  diffuse;"     // Dcm
                "vec4  specular;"    // Scm
                "float shininess;"   // Srm
            "};"
            "uniform gl_MaterialParameters  gl_FrontMaterial;"
            "uniform gl_MaterialParameters  gl_BackMaterial;"

            //
            // Light State p 50, 53, 55.
            //
            "struct gl_LightSourceParameters {"
                "vec4  ambient;"             // Acli
                "vec4  diffuse;"             // Dcli
                "vec4  specular;"            // Scli
                "vec4  position;"            // Ppli
                "vec4  halfVector;"          // Derived: Hi
                "vec3  spotDirection;"       // Sdli
                "float spotExponent;"        // Srli
                "float spotCutoff;"          // Crli
                                                        // (range: [0.0,90.0], 180.0)
                "float spotCosCutoff;"       // Derived: cos(Crli)
                                                        // (range: [1.0,0.0],-1.0)
                "float constantAttenuation;" // K0
                "float linearAttenuation;"   // K1
                "float quadraticAttenuation;"// K2
            "};"

            "struct gl_LightModelParameters {"
                "vec4  ambient;"       // Acs
            "};"

            "uniform gl_LightModelParameters  gl_LightModel;"

            //
            // Derived state from products of light and material.
            //
            "struct gl_LightModelProducts {"
                "vec4  sceneColor;"     // Derived. Ecm + Acm * Acs
            "};"

            "uniform gl_LightModelProducts gl_FrontLightModelProduct;"
            "uniform gl_LightModelProducts gl_BackLightModelProduct;"

            "struct gl_LightProducts {"
                "vec4  ambient;"        // Acm * Acli
                "vec4  diffuse;"        // Dcm * Dcli
                "vec4  specular;"       // Scm * Scli
            "};"

            //
            // Fog p. 161
            //
            "struct gl_FogParameters {"
                "vec4  color;"
                "float density;"
                "float start;"
                "float end;"
                "float scale;"   //  1 / (gl_FogEnd - gl_FogStart)
            "};"

            "uniform gl_FogParameters gl_Fog;"

            "\n");
    }

    //============================================================================
    //
    // Define the interface to the compute shader.
    //
    //============================================================================

    if ((profile != EEsProfile && version >= 420) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangCompute].append(
            "in    highp uvec3 gl_NumWorkGroups;"
            "const highp uvec3 gl_WorkGroupSize = uvec3(1,1,1);"

            "in highp uvec3 gl_WorkGroupID;"
            "in highp uvec3 gl_LocalInvocationID;"

            "in highp uvec3 gl_GlobalInvocationID;"
            "in highp uint gl_LocalInvocationIndex;"

            "\n");
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangCompute].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "\n");
    }

    //============================================================================
    //
    // Define the interface to the mesh/task shader.
    //
    //============================================================================

    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        // per-vertex attributes
        stageBuiltins[EShLangMesh].append(
            "out gl_MeshPerVertexNV {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
                "float gl_CullDistance[];"
                "perviewNV vec4 gl_PositionPerViewNV[];"
                "perviewNV float gl_ClipDistancePerViewNV[][];"
                "perviewNV float gl_CullDistancePerViewNV[][];"
            "} gl_MeshVerticesNV[];"
        );

        // per-primitive attributes
        stageBuiltins[EShLangMesh].append(
            "perprimitiveNV out gl_MeshPerPrimitiveNV {"
                "int gl_PrimitiveID;"
                "int gl_Layer;"
                "int gl_ViewportIndex;"
                "int gl_ViewportMask[];"
                "perviewNV int gl_LayerPerViewNV[];"
                "perviewNV int gl_ViewportMaskPerViewNV[][];"
            "} gl_MeshPrimitivesNV[];"
        );

        stageBuiltins[EShLangMesh].append(
            "out uint gl_PrimitiveCountNV;"
            "out uint gl_PrimitiveIndicesNV[];"

            "in uint gl_MeshViewCountNV;"
            "in uint gl_MeshViewIndicesNV[4];"

            "const highp uvec3 gl_WorkGroupSize = uvec3(1,1,1);"

            "in highp uvec3 gl_WorkGroupID;"
            "in highp uvec3 gl_LocalInvocationID;"

            "in highp uvec3 gl_GlobalInvocationID;"
            "in highp uint gl_LocalInvocationIndex;"
            "\n");

        // GL_EXT_mesh_shader
        stageBuiltins[EShLangMesh].append(
            "out uint gl_PrimitivePointIndicesEXT[];"
            "out uvec2 gl_PrimitiveLineIndicesEXT[];"
            "out uvec3 gl_PrimitiveTriangleIndicesEXT[];"
            "in    highp uvec3 gl_NumWorkGroups;"
            "\n");

        // per-vertex attributes
        stageBuiltins[EShLangMesh].append(
            "out gl_MeshPerVertexEXT {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
                "float gl_CullDistance[];"
            "} gl_MeshVerticesEXT[];"
        );

        // per-primitive attributes
        stageBuiltins[EShLangMesh].append(
            "perprimitiveEXT out gl_MeshPerPrimitiveEXT {"
                "int gl_PrimitiveID;"
                "int gl_Layer;"
                "int gl_ViewportIndex;"
                "bool gl_CullPrimitiveEXT;"
                "int  gl_PrimitiveShadingRateEXT;"
            "} gl_MeshPrimitivesEXT[];"
        );

        stageBuiltins[EShLangTask].append(
            "out uint gl_TaskCountNV;"

            "const highp uvec3 gl_WorkGroupSize = uvec3(1,1,1);"

            "in highp uvec3 gl_WorkGroupID;"
            "in highp uvec3 gl_LocalInvocationID;"

            "in highp uvec3 gl_GlobalInvocationID;"
            "in highp uint gl_LocalInvocationIndex;"

            "in uint gl_MeshViewCountNV;"
            "in uint gl_MeshViewIndicesNV[4];"
            "in    highp uvec3 gl_NumWorkGroups;"
            "\n");
    }

    if (profile != EEsProfile && version >= 450) {
        stageBuiltins[EShLangMesh].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in int gl_DrawIDARB;"             // GL_ARB_shader_draw_parameters
            "in int gl_ViewIndex;"             // GL_EXT_multiview
            "\n");

        stageBuiltins[EShLangTask].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in int gl_DrawIDARB;"             // GL_ARB_shader_draw_parameters
            "\n");

        if (version >= 460) {
            stageBuiltins[EShLangMesh].append(
                "in int gl_DrawID;"
                "\n");

            stageBuiltins[EShLangTask].append(
                "in int gl_DrawID;"
                "\n");
        }
    }

    //============================================================================
    //
    // Define the interface to the vertex shader.
    //
    //============================================================================

    if (profile != EEsProfile) {
        if (version < 130) {
            stageBuiltins[EShLangVertex].append(
                "attribute vec4  gl_Color;"
                "attribute vec4  gl_SecondaryColor;"
                "attribute vec3  gl_Normal;"
                "attribute vec4  gl_Vertex;"
                "attribute vec4  gl_MultiTexCoord0;"
                "attribute vec4  gl_MultiTexCoord1;"
                "attribute vec4  gl_MultiTexCoord2;"
                "attribute vec4  gl_MultiTexCoord3;"
                "attribute vec4  gl_MultiTexCoord4;"
                "attribute vec4  gl_MultiTexCoord5;"
                "attribute vec4  gl_MultiTexCoord6;"
                "attribute vec4  gl_MultiTexCoord7;"
                "attribute float gl_FogCoord;"
                "\n");
        } else if (IncludeLegacy(version, profile, spvVersion)) {
            stageBuiltins[EShLangVertex].append(
                "in vec4  gl_Color;"
                "in vec4  gl_SecondaryColor;"
                "in vec3  gl_Normal;"
                "in vec4  gl_Vertex;"
                "in vec4  gl_MultiTexCoord0;"
                "in vec4  gl_MultiTexCoord1;"
                "in vec4  gl_MultiTexCoord2;"
                "in vec4  gl_MultiTexCoord3;"
                "in vec4  gl_MultiTexCoord4;"
                "in vec4  gl_MultiTexCoord5;"
                "in vec4  gl_MultiTexCoord6;"
                "in vec4  gl_MultiTexCoord7;"
                "in float gl_FogCoord;"
                "\n");
        }

        if (version < 150) {
            if (version < 130) {
                stageBuiltins[EShLangVertex].append(
                    "        vec4  gl_ClipVertex;"       // needs qualifier fixed later
                    "varying vec4  gl_FrontColor;"
                    "varying vec4  gl_BackColor;"
                    "varying vec4  gl_FrontSecondaryColor;"
                    "varying vec4  gl_BackSecondaryColor;"
                    "varying vec4  gl_TexCoord[];"
                    "varying float gl_FogFragCoord;"
                    "\n");
            } else if (IncludeLegacy(version, profile, spvVersion)) {
                stageBuiltins[EShLangVertex].append(
                    "    vec4  gl_ClipVertex;"       // needs qualifier fixed later
                    "out vec4  gl_FrontColor;"
                    "out vec4  gl_BackColor;"
                    "out vec4  gl_FrontSecondaryColor;"
                    "out vec4  gl_BackSecondaryColor;"
                    "out vec4  gl_TexCoord[];"
                    "out float gl_FogFragCoord;"
                    "\n");
            }
            stageBuiltins[EShLangVertex].append(
                "vec4 gl_Position;"   // needs qualifier fixed later
                "float gl_PointSize;" // needs qualifier fixed later
                );

            if (version == 130 || version == 140)
                stageBuiltins[EShLangVertex].append(
                    "out float gl_ClipDistance[];"
                    );
        } else {
            // version >= 150
            stageBuiltins[EShLangVertex].append(
                "out gl_PerVertex {"
                    "vec4 gl_Position;"     // needs qualifier fixed later
                    "float gl_PointSize;"   // needs qualifier fixed later
                    "float gl_ClipDistance[];"
                    );
            if (IncludeLegacy(version, profile, spvVersion))
                stageBuiltins[EShLangVertex].append(
                    "vec4 gl_ClipVertex;"   // needs qualifier fixed later
                    "vec4 gl_FrontColor;"
                    "vec4 gl_BackColor;"
                    "vec4 gl_FrontSecondaryColor;"
                    "vec4 gl_BackSecondaryColor;"
                    "vec4 gl_TexCoord[];"
                    "float gl_FogFragCoord;"
                    );
            if (version >= 450)
                stageBuiltins[EShLangVertex].append(
                    "float gl_CullDistance[];"
                    );
            stageBuiltins[EShLangVertex].append(
                "};"
                "\n");
        }
        if (version >= 130 && spvVersion.vulkan == 0)
            stageBuiltins[EShLangVertex].append(
                "int gl_VertexID;"            // needs qualifier fixed later
                );
        if (spvVersion.vulkan == 0)
            stageBuiltins[EShLangVertex].append(
                "int gl_InstanceID;"          // needs qualifier fixed later
                );
        if (spvVersion.vulkan > 0 && version >= 140)
            stageBuiltins[EShLangVertex].append(
                "in int gl_VertexIndex;"
                "in int gl_InstanceIndex;"
                );

        if (spvVersion.vulkan > 0 && version >= 140 && spvVersion.vulkanRelaxed)
            stageBuiltins[EShLangVertex].append(
                "in int gl_VertexID;"         // declare with 'in' qualifier
                "in int gl_InstanceID;"
                );

        if (version >= 440) {
            stageBuiltins[EShLangVertex].append(
                "in int gl_BaseVertexARB;"
                "in int gl_BaseInstanceARB;"
                "in int gl_DrawIDARB;"
                );
        }
        if (version >= 410) {
            stageBuiltins[EShLangVertex].append(
                "out int gl_ViewportIndex;"
                "out int gl_Layer;"
                );
        }
        if (version >= 460) {
            stageBuiltins[EShLangVertex].append(
                "in int gl_BaseVertex;"
                "in int gl_BaseInstance;"
                "in int gl_DrawID;"
                );
        }

        if (version >= 430)
            stageBuiltins[EShLangVertex].append(
                "out int gl_ViewportMask[];"             // GL_NV_viewport_array2
                );

        if (version >= 450)
            stageBuiltins[EShLangVertex].append(
                "out int gl_SecondaryViewportMaskNV[];"  // GL_NV_stereo_view_rendering
                "out vec4 gl_SecondaryPositionNV;"       // GL_NV_stereo_view_rendering
                "out vec4 gl_PositionPerViewNV[];"       // GL_NVX_multiview_per_view_attributes
                "out int  gl_ViewportMaskPerViewNV[];"   // GL_NVX_multiview_per_view_attributes
                );
    } else {
        // ES profile
        if (version == 100) {
            stageBuiltins[EShLangVertex].append(
                "highp   vec4  gl_Position;"  // needs qualifier fixed later
                "mediump float gl_PointSize;" // needs qualifier fixed later
                "highp int gl_InstanceID;" // needs qualifier fixed later
                );
        } else {
            if (spvVersion.vulkan == 0 || spvVersion.vulkanRelaxed)
                stageBuiltins[EShLangVertex].append(
                    "in highp int gl_VertexID;"      // needs qualifier fixed later
                    "in highp int gl_InstanceID;"    // needs qualifier fixed later
                    );
            if (spvVersion.vulkan > 0)
                stageBuiltins[EShLangVertex].append(
                    "in highp int gl_VertexIndex;"
                    "in highp int gl_InstanceIndex;"
                    );
            if (version < 310)
                stageBuiltins[EShLangVertex].append(
                    "highp vec4  gl_Position;"    // needs qualifier fixed later
                    "highp float gl_PointSize;"   // needs qualifier fixed later
                    );
            else
                stageBuiltins[EShLangVertex].append(
                    "out gl_PerVertex {"
                        "highp vec4  gl_Position;"    // needs qualifier fixed later
                        "highp float gl_PointSize;"   // needs qualifier fixed later
                    "};"
                    );
        }
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangVertex].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in highp int gl_ViewIndex;"       // GL_EXT_multiview
            "\n");
    }

    if (version >= 300 /* both ES and non-ES */) {
        stageBuiltins[EShLangVertex].append(
            "in highp uint gl_ViewID_OVR;"     // GL_OVR_multiview, GL_OVR_multiview2
            "\n");
    }

    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangVertex].append(
            "out highp int gl_PrimitiveShadingRateEXT;" // GL_EXT_fragment_shading_rate
            "\n");
    }

    //============================================================================
    //
    // Define the interface to the geometry shader.
    //
    //============================================================================

    if (profile == ECoreProfile || profile == ECompatibilityProfile) {
        stageBuiltins[EShLangGeometry].append(
            "in gl_PerVertex {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
                );
        if (profile == ECompatibilityProfile)
            stageBuiltins[EShLangGeometry].append(
                "vec4 gl_ClipVertex;"
                "vec4 gl_FrontColor;"
                "vec4 gl_BackColor;"
                "vec4 gl_FrontSecondaryColor;"
                "vec4 gl_BackSecondaryColor;"
                "vec4 gl_TexCoord[];"
                "float gl_FogFragCoord;"
                );
        if (version >= 450)
            stageBuiltins[EShLangGeometry].append(
                "float gl_CullDistance[];"
                "vec4 gl_SecondaryPositionNV;"   // GL_NV_stereo_view_rendering
                "vec4 gl_PositionPerViewNV[];"   // GL_NVX_multiview_per_view_attributes
                );
        stageBuiltins[EShLangGeometry].append(
            "} gl_in[];"

            "in int gl_PrimitiveIDIn;"
            "out gl_PerVertex {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
                "\n");
        if (profile == ECompatibilityProfile && version >= 400)
            stageBuiltins[EShLangGeometry].append(
                "vec4 gl_ClipVertex;"
                "vec4 gl_FrontColor;"
                "vec4 gl_BackColor;"
                "vec4 gl_FrontSecondaryColor;"
                "vec4 gl_BackSecondaryColor;"
                "vec4 gl_TexCoord[];"
                "float gl_FogFragCoord;"
                );
        if (version >= 450)
            stageBuiltins[EShLangGeometry].append(
                "float gl_CullDistance[];"
                );
        stageBuiltins[EShLangGeometry].append(
            "};"

            "out int gl_PrimitiveID;"
            "out int gl_Layer;");

        if (version >= 150)
            stageBuiltins[EShLangGeometry].append(
            "out int gl_ViewportIndex;"
            );

        if (profile == ECompatibilityProfile && version < 400)
            stageBuiltins[EShLangGeometry].append(
            "out vec4 gl_ClipVertex;"
            );

        if (version >= 400)
            stageBuiltins[EShLangGeometry].append(
            "in int gl_InvocationID;"
            );

        if (version >= 430)
            stageBuiltins[EShLangGeometry].append(
                "out int gl_ViewportMask[];"               // GL_NV_viewport_array2
            );

        if (version >= 450)
            stageBuiltins[EShLangGeometry].append(
                "out int gl_SecondaryViewportMaskNV[];"    // GL_NV_stereo_view_rendering
                "out vec4 gl_SecondaryPositionNV;"         // GL_NV_stereo_view_rendering
                "out vec4 gl_PositionPerViewNV[];"         // GL_NVX_multiview_per_view_attributes
                "out int  gl_ViewportMaskPerViewNV[];"     // GL_NVX_multiview_per_view_attributes
            );

        stageBuiltins[EShLangGeometry].append("\n");
    } else if (profile == EEsProfile && version >= 310) {
        stageBuiltins[EShLangGeometry].append(
            "in gl_PerVertex {"
                "highp vec4 gl_Position;"
                "highp float gl_PointSize;"
            "} gl_in[];"
            "\n"
            "in highp int gl_PrimitiveIDIn;"
            "in highp int gl_InvocationID;"
            "\n"
            "out gl_PerVertex {"
                "highp vec4 gl_Position;"
                "highp float gl_PointSize;"
            "};"
            "\n"
            "out highp int gl_PrimitiveID;"
            "out highp int gl_Layer;"
            "\n"
            );
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangGeometry].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in highp int gl_ViewIndex;"       // GL_EXT_multiview
            "\n");
    }

    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangGeometry].append(
            "out highp int gl_PrimitiveShadingRateEXT;" // GL_EXT_fragment_shading_rate
            "\n");
    }

    //============================================================================
    //
    // Define the interface to the tessellation control shader.
    //
    //============================================================================

    if (profile != EEsProfile && version >= 150) {
        // Note:  "in gl_PerVertex {...} gl_in[gl_MaxPatchVertices];" is declared in initialize() below,
        // as it depends on the resource sizing of gl_MaxPatchVertices.

        stageBuiltins[EShLangTessControl].append(
            "in int gl_PatchVerticesIn;"
            "in int gl_PrimitiveID;"
            "in int gl_InvocationID;"

            "out gl_PerVertex {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
                );
        if (profile == ECompatibilityProfile)
            stageBuiltins[EShLangTessControl].append(
                "vec4 gl_ClipVertex;"
                "vec4 gl_FrontColor;"
                "vec4 gl_BackColor;"
                "vec4 gl_FrontSecondaryColor;"
                "vec4 gl_BackSecondaryColor;"
                "vec4 gl_TexCoord[];"
                "float gl_FogFragCoord;"
                );
        if (version >= 450)
            stageBuiltins[EShLangTessControl].append(
                "float gl_CullDistance[];"
            );
        if (version >= 430)
            stageBuiltins[EShLangTessControl].append(
                "int  gl_ViewportMask[];"             // GL_NV_viewport_array2
            );
        if (version >= 450)
            stageBuiltins[EShLangTessControl].append(
                "vec4 gl_SecondaryPositionNV;"        // GL_NV_stereo_view_rendering
                "int  gl_SecondaryViewportMaskNV[];"  // GL_NV_stereo_view_rendering
                "vec4 gl_PositionPerViewNV[];"        // GL_NVX_multiview_per_view_attributes
                "int  gl_ViewportMaskPerViewNV[];"    // GL_NVX_multiview_per_view_attributes
                );
        stageBuiltins[EShLangTessControl].append(
            "} gl_out[];"

            "patch out float gl_TessLevelOuter[4];"
            "patch out float gl_TessLevelInner[2];"
            "\n");

        if (version >= 410)
            stageBuiltins[EShLangTessControl].append(
                "out int gl_ViewportIndex;"
                "out int gl_Layer;"
                "\n");

    } else {
        // Note:  "in gl_PerVertex {...} gl_in[gl_MaxPatchVertices];" is declared in initialize() below,
        // as it depends on the resource sizing of gl_MaxPatchVertices.

        stageBuiltins[EShLangTessControl].append(
            "in highp int gl_PatchVerticesIn;"
            "in highp int gl_PrimitiveID;"
            "in highp int gl_InvocationID;"

            "out gl_PerVertex {"
                "highp vec4 gl_Position;"
                "highp float gl_PointSize;"
                );
        stageBuiltins[EShLangTessControl].append(
            "} gl_out[];"

            "patch out highp float gl_TessLevelOuter[4];"
            "patch out highp float gl_TessLevelInner[2];"
            "patch out highp vec4 gl_BoundingBoxOES[2];"
            "patch out highp vec4 gl_BoundingBoxEXT[2];"
            "\n");
        if (profile == EEsProfile && version >= 320) {
            stageBuiltins[EShLangTessControl].append(
                "patch out highp vec4 gl_BoundingBox[2];"
                "\n"
            );
        }
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangTessControl].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in highp int gl_ViewIndex;"       // GL_EXT_multiview
            "\n");
    }

    //============================================================================
    //
    // Define the interface to the tessellation evaluation shader.
    //
    //============================================================================

    if (profile != EEsProfile && version >= 150) {
        // Note:  "in gl_PerVertex {...} gl_in[gl_MaxPatchVertices];" is declared in initialize() below,
        // as it depends on the resource sizing of gl_MaxPatchVertices.

        stageBuiltins[EShLangTessEvaluation].append(
            "in int gl_PatchVerticesIn;"
            "in int gl_PrimitiveID;"
            "in vec3 gl_TessCoord;"

            "patch in float gl_TessLevelOuter[4];"
            "patch in float gl_TessLevelInner[2];"

            "out gl_PerVertex {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
            );
        if (version >= 400 && profile == ECompatibilityProfile)
            stageBuiltins[EShLangTessEvaluation].append(
                "vec4 gl_ClipVertex;"
                "vec4 gl_FrontColor;"
                "vec4 gl_BackColor;"
                "vec4 gl_FrontSecondaryColor;"
                "vec4 gl_BackSecondaryColor;"
                "vec4 gl_TexCoord[];"
                "float gl_FogFragCoord;"
                );
        if (version >= 450)
            stageBuiltins[EShLangTessEvaluation].append(
                "float gl_CullDistance[];"
                );
        stageBuiltins[EShLangTessEvaluation].append(
            "};"
            "\n");

        if (version >= 410)
            stageBuiltins[EShLangTessEvaluation].append(
                "out int gl_ViewportIndex;"
                "out int gl_Layer;"
                "\n");

        if (version >= 430)
            stageBuiltins[EShLangTessEvaluation].append(
                "out int  gl_ViewportMask[];"             // GL_NV_viewport_array2
            );

        if (version >= 450)
            stageBuiltins[EShLangTessEvaluation].append(
                "out vec4 gl_SecondaryPositionNV;"        // GL_NV_stereo_view_rendering
                "out int  gl_SecondaryViewportMaskNV[];"  // GL_NV_stereo_view_rendering
                "out vec4 gl_PositionPerViewNV[];"        // GL_NVX_multiview_per_view_attributes
                "out int  gl_ViewportMaskPerViewNV[];"    // GL_NVX_multiview_per_view_attributes
                );

    } else if (profile == EEsProfile && version >= 310) {
        // Note:  "in gl_PerVertex {...} gl_in[gl_MaxPatchVertices];" is declared in initialize() below,
        // as it depends on the resource sizing of gl_MaxPatchVertices.

        stageBuiltins[EShLangTessEvaluation].append(
            "in highp int gl_PatchVerticesIn;"
            "in highp int gl_PrimitiveID;"
            "in highp vec3 gl_TessCoord;"

            "patch in highp float gl_TessLevelOuter[4];"
            "patch in highp float gl_TessLevelInner[2];"

            "out gl_PerVertex {"
                "highp vec4 gl_Position;"
                "highp float gl_PointSize;"
            );
        stageBuiltins[EShLangTessEvaluation].append(
            "};"
            "\n");
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangTessEvaluation].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in highp int gl_ViewIndex;"       // GL_EXT_multiview
            "\n");
    }

    //============================================================================
    //
    // Define the interface to the fragment shader.
    //
    //============================================================================

    if (profile != EEsProfile) {

        stageBuiltins[EShLangFragment].append(
            "vec4  gl_FragCoord;"   // needs qualifier fixed later
            "bool  gl_FrontFacing;" // needs qualifier fixed later
            "float gl_FragDepth;"   // needs qualifier fixed later
            );
        if (version >= 120)
            stageBuiltins[EShLangFragment].append(
                "vec2 gl_PointCoord;"  // needs qualifier fixed later
                );
        if (version >= 140)
            stageBuiltins[EShLangFragment].append(
                "out int gl_FragStencilRefARB;"
                );
        if (IncludeLegacy(version, profile, spvVersion) || (! ForwardCompatibility && version < 420))
            stageBuiltins[EShLangFragment].append(
                "vec4 gl_FragColor;"   // needs qualifier fixed later
                );

        if (version < 130) {
            stageBuiltins[EShLangFragment].append(
                "varying vec4  gl_Color;"
                "varying vec4  gl_SecondaryColor;"
                "varying vec4  gl_TexCoord[];"
                "varying float gl_FogFragCoord;"
                );
        } else {
            stageBuiltins[EShLangFragment].append(
                "in float gl_ClipDistance[];"
                );

            if (IncludeLegacy(version, profile, spvVersion)) {
                if (version < 150)
                    stageBuiltins[EShLangFragment].append(
                        "in float gl_FogFragCoord;"
                        "in vec4  gl_TexCoord[];"
                        "in vec4  gl_Color;"
                        "in vec4  gl_SecondaryColor;"
                        );
                else
                    stageBuiltins[EShLangFragment].append(
                        "in gl_PerFragment {"
                            "in float gl_FogFragCoord;"
                            "in vec4  gl_TexCoord[];"
                            "in vec4  gl_Color;"
                            "in vec4  gl_SecondaryColor;"
                        "};"
                        );
            }
        }

        if (version >= 150)
            stageBuiltins[EShLangFragment].append(
                "flat in int gl_PrimitiveID;"
                );

        if (version >= 130) { // ARB_sample_shading
            stageBuiltins[EShLangFragment].append(
                "flat in  int  gl_SampleID;"
                "     in  vec2 gl_SamplePosition;"
                "     out int  gl_SampleMask[];"
                );

            if (spvVersion.spv == 0) {
                stageBuiltins[EShLangFragment].append(
                    "uniform int gl_NumSamples;"
                );
            }
        }

        if (version >= 400)
            stageBuiltins[EShLangFragment].append(
                "flat in  int  gl_SampleMaskIn[];"
            );

        if (version >= 430)
            stageBuiltins[EShLangFragment].append(
                "flat in int gl_Layer;"
                "flat in int gl_ViewportIndex;"
                );

        if (version >= 450)
            stageBuiltins[EShLangFragment].append(
                "in float gl_CullDistance[];"
                "bool gl_HelperInvocation;"     // needs qualifier fixed later
                );

        if (version >= 450)
            stageBuiltins[EShLangFragment].append( // GL_EXT_fragment_invocation_density
                "flat in ivec2 gl_FragSizeEXT;"
                "flat in int   gl_FragInvocationCountEXT;"
                );

        if (version >= 450)
            stageBuiltins[EShLangFragment].append(
                "in vec2 gl_BaryCoordNoPerspAMD;"
                "in vec2 gl_BaryCoordNoPerspCentroidAMD;"
                "in vec2 gl_BaryCoordNoPerspSampleAMD;"
                "in vec2 gl_BaryCoordSmoothAMD;"
                "in vec2 gl_BaryCoordSmoothCentroidAMD;"
                "in vec2 gl_BaryCoordSmoothSampleAMD;"
                "in vec3 gl_BaryCoordPullModelAMD;"
                );

        if (version >= 430)
            stageBuiltins[EShLangFragment].append(
                "in bool gl_FragFullyCoveredNV;"
                );
        if (version >= 450)
            stageBuiltins[EShLangFragment].append(
                "flat in ivec2 gl_FragmentSizeNV;"          // GL_NV_shading_rate_image
                "flat in int   gl_InvocationsPerPixelNV;"
                "in vec3 gl_BaryCoordNV;"                   // GL_NV_fragment_shader_barycentric
                "in vec3 gl_BaryCoordNoPerspNV;"
                "in vec3 gl_BaryCoordEXT;"                  // GL_EXT_fragment_shader_barycentric
                "in vec3 gl_BaryCoordNoPerspEXT;"
                );

        if (version >= 450)
            stageBuiltins[EShLangFragment].append(
                "flat in int gl_ShadingRateEXT;" // GL_EXT_fragment_shading_rate
            );

    } else {
        // ES profile

        if (version == 100) {
            stageBuiltins[EShLangFragment].append(
                "mediump vec4 gl_FragCoord;"    // needs qualifier fixed later
                "        bool gl_FrontFacing;"  // needs qualifier fixed later
                "mediump vec4 gl_FragColor;"    // needs qualifier fixed later
                "mediump vec2 gl_PointCoord;"   // needs qualifier fixed later
                );
        }
        if (version >= 300) {
            stageBuiltins[EShLangFragment].append(
                "highp   vec4  gl_FragCoord;"    // needs qualifier fixed later
                "        bool  gl_FrontFacing;"  // needs qualifier fixed later
                "mediump vec2  gl_PointCoord;"   // needs qualifier fixed later
                "highp   float gl_FragDepth;"    // needs qualifier fixed later
                );
        }
        if (version >= 310) {
            stageBuiltins[EShLangFragment].append(
                "bool gl_HelperInvocation;"          // needs qualifier fixed later
                "flat in highp int gl_PrimitiveID;"  // needs qualifier fixed later
                "flat in highp int gl_Layer;"        // needs qualifier fixed later
                );

            stageBuiltins[EShLangFragment].append(  // GL_OES_sample_variables
                "flat  in lowp     int gl_SampleID;"
                "      in mediump vec2 gl_SamplePosition;"
                "flat  in highp    int gl_SampleMaskIn[];"
                "     out highp    int gl_SampleMask[];"
                );
            if (spvVersion.spv == 0)
                stageBuiltins[EShLangFragment].append(  // GL_OES_sample_variables
                    "uniform lowp int gl_NumSamples;"
                    );
        }
        stageBuiltins[EShLangFragment].append(
            "highp float gl_FragDepthEXT;"       // GL_EXT_frag_depth
            );

        if (version >= 310)
            stageBuiltins[EShLangFragment].append( // GL_EXT_fragment_invocation_density
                "flat in ivec2 gl_FragSizeEXT;"
                "flat in int   gl_FragInvocationCountEXT;"
            );
        if (version >= 320)
            stageBuiltins[EShLangFragment].append( // GL_NV_shading_rate_image
                "flat in ivec2 gl_FragmentSizeNV;"
                "flat in int   gl_InvocationsPerPixelNV;"
            );
        if (version >= 320)
            stageBuiltins[EShLangFragment].append(
                "in vec3 gl_BaryCoordNV;"
                "in vec3 gl_BaryCoordNoPerspNV;"
                "in vec3 gl_BaryCoordEXT;"
                "in vec3 gl_BaryCoordNoPerspEXT;"
            );
        if (version >= 310)
            stageBuiltins[EShLangFragment].append(
                "flat in highp int gl_ShadingRateEXT;" // GL_EXT_fragment_shading_rate
            );
    }

    stageBuiltins[EShLangFragment].append("\n");

    if (version >= 130)
        add2ndGenerationSamplingImaging(version, profile, spvVersion);

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangFragment].append(
            "flat in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "flat in highp int gl_ViewIndex;"       // GL_EXT_multiview
            "\n");
    }

    if (version >= 300 /* both ES and non-ES */) {
        stageBuiltins[EShLangFragment].append(
            "flat in highp uint gl_ViewID_OVR;"     // GL_OVR_multiview, GL_OVR_multiview2
            "\n");
    }

    // GL_ARB_shader_ballot
    if (profile != EEsProfile && version >= 450) {
        const char* ballotDecls =
            "uniform uint gl_SubGroupSizeARB;"
            "in uint     gl_SubGroupInvocationARB;"
            "in uint64_t gl_SubGroupEqMaskARB;"
            "in uint64_t gl_SubGroupGeMaskARB;"
            "in uint64_t gl_SubGroupGtMaskARB;"
            "in uint64_t gl_SubGroupLeMaskARB;"
            "in uint64_t gl_SubGroupLtMaskARB;"
            "\n";
        const char* rtBallotDecls =
            "uniform volatile uint gl_SubGroupSizeARB;"
            "in volatile uint     gl_SubGroupInvocationARB;"
            "in volatile uint64_t gl_SubGroupEqMaskARB;"
            "in volatile uint64_t gl_SubGroupGeMaskARB;"
            "in volatile uint64_t gl_SubGroupGtMaskARB;"
            "in volatile uint64_t gl_SubGroupLeMaskARB;"
            "in volatile uint64_t gl_SubGroupLtMaskARB;"
            "\n";
        const char* fragmentBallotDecls =
            "uniform uint gl_SubGroupSizeARB;"
            "flat in uint     gl_SubGroupInvocationARB;"
            "flat in uint64_t gl_SubGroupEqMaskARB;"
            "flat in uint64_t gl_SubGroupGeMaskARB;"
            "flat in uint64_t gl_SubGroupGtMaskARB;"
            "flat in uint64_t gl_SubGroupLeMaskARB;"
            "flat in uint64_t gl_SubGroupLtMaskARB;"
            "\n";
        stageBuiltins[EShLangVertex]        .append(ballotDecls);
        stageBuiltins[EShLangTessControl]   .append(ballotDecls);
        stageBuiltins[EShLangTessEvaluation].append(ballotDecls);
        stageBuiltins[EShLangGeometry]      .append(ballotDecls);
        stageBuiltins[EShLangCompute]       .append(ballotDecls);
        stageBuiltins[EShLangFragment]      .append(fragmentBallotDecls);
        stageBuiltins[EShLangMesh]        .append(ballotDecls);
        stageBuiltins[EShLangTask]        .append(ballotDecls);
        stageBuiltins[EShLangRayGen]        .append(rtBallotDecls);
        stageBuiltins[EShLangIntersect]     .append(rtBallotDecls);
        // No volatile qualifier on these builtins in any-hit
        stageBuiltins[EShLangAnyHit]        .append(ballotDecls);
        stageBuiltins[EShLangClosestHit]    .append(rtBallotDecls);
        stageBuiltins[EShLangMiss]          .append(rtBallotDecls);
        stageBuiltins[EShLangCallable]      .append(rtBallotDecls);
    }

    // GL_KHR_shader_subgroup
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 140)) {
        const char* subgroupDecls =
            "in mediump uint  gl_SubgroupSize;"
            "in mediump uint  gl_SubgroupInvocationID;"
            "in highp   uvec4 gl_SubgroupEqMask;"
            "in highp   uvec4 gl_SubgroupGeMask;"
            "in highp   uvec4 gl_SubgroupGtMask;"
            "in highp   uvec4 gl_SubgroupLeMask;"
            "in highp   uvec4 gl_SubgroupLtMask;"
            // GL_NV_shader_sm_builtins
            "in highp   uint  gl_WarpsPerSMNV;"
            "in highp   uint  gl_SMCountNV;"
            "in highp   uint  gl_WarpIDNV;"
            "in highp   uint  gl_SMIDNV;"
            // GL_ARM_shader_core_builtins
            "in highp   uint  gl_CoreIDARM;"
            "in highp   uint  gl_CoreCountARM;"
            "in highp   uint  gl_CoreMaxIDARM;"
            "in highp   uint  gl_WarpIDARM;"
            "in highp   uint  gl_WarpMaxIDARM;"
            "\n";
        const char* fragmentSubgroupDecls =
            "flat in mediump uint  gl_SubgroupSize;"
            "flat in mediump uint  gl_SubgroupInvocationID;"
            "flat in highp   uvec4 gl_SubgroupEqMask;"
            "flat in highp   uvec4 gl_SubgroupGeMask;"
            "flat in highp   uvec4 gl_SubgroupGtMask;"
            "flat in highp   uvec4 gl_SubgroupLeMask;"
            "flat in highp   uvec4 gl_SubgroupLtMask;"
            // GL_NV_shader_sm_builtins
            "flat in highp   uint  gl_WarpsPerSMNV;"
            "flat in highp   uint  gl_SMCountNV;"
            "flat in highp   uint  gl_WarpIDNV;"
            "flat in highp   uint  gl_SMIDNV;"
            // GL_ARM_shader_core_builtins
            "flat in highp   uint  gl_CoreIDARM;"
            "flat in highp   uint  gl_CoreCountARM;"
            "flat in highp   uint  gl_CoreMaxIDARM;"
            "flat in highp   uint  gl_WarpIDARM;"
            "flat in highp   uint  gl_WarpMaxIDARM;"
            "\n";
        const char* computeSubgroupDecls =
            "in highp   uint  gl_NumSubgroups;"
            "in highp   uint  gl_SubgroupID;"
            "\n";
        // These builtins are volatile for RT stages
        const char* rtSubgroupDecls =
            "in mediump volatile uint  gl_SubgroupSize;"
            "in mediump volatile uint  gl_SubgroupInvocationID;"
            "in highp   volatile uvec4 gl_SubgroupEqMask;"
            "in highp   volatile uvec4 gl_SubgroupGeMask;"
            "in highp   volatile uvec4 gl_SubgroupGtMask;"
            "in highp   volatile uvec4 gl_SubgroupLeMask;"
            "in highp   volatile uvec4 gl_SubgroupLtMask;"
            // GL_NV_shader_sm_builtins
            "in highp    uint  gl_WarpsPerSMNV;"
            "in highp    uint  gl_SMCountNV;"
            "in highp volatile uint  gl_WarpIDNV;"
            "in highp volatile uint  gl_SMIDNV;"
            // GL_ARM_shader_core_builtins
            "in highp   uint  gl_CoreIDARM;"
            "in highp   uint  gl_CoreCountARM;"
            "in highp   uint  gl_CoreMaxIDARM;"
            "in highp   uint  gl_WarpIDARM;"
            "in highp   uint  gl_WarpMaxIDARM;"
            "\n";

        stageBuiltins[EShLangVertex]        .append(subgroupDecls);
        stageBuiltins[EShLangTessControl]   .append(subgroupDecls);
        stageBuiltins[EShLangTessEvaluation].append(subgroupDecls);
        stageBuiltins[EShLangGeometry]      .append(subgroupDecls);
        stageBuiltins[EShLangCompute]       .append(subgroupDecls);
        stageBuiltins[EShLangCompute]       .append(computeSubgroupDecls);
        stageBuiltins[EShLangFragment]      .append(fragmentSubgroupDecls);
        stageBuiltins[EShLangMesh]        .append(subgroupDecls);
        stageBuiltins[EShLangMesh]        .append(computeSubgroupDecls);
        stageBuiltins[EShLangTask]        .append(subgroupDecls);
        stageBuiltins[EShLangTask]        .append(computeSubgroupDecls);
        stageBuiltins[EShLangRayGen]        .append(rtSubgroupDecls);
        stageBuiltins[EShLangIntersect]     .append(rtSubgroupDecls);
        // No volatile qualifier on these builtins in any-hit
        stageBuiltins[EShLangAnyHit]        .append(subgroupDecls);
        stageBuiltins[EShLangClosestHit]    .append(rtSubgroupDecls);
        stageBuiltins[EShLangMiss]          .append(rtSubgroupDecls);
        stageBuiltins[EShLangCallable]      .append(rtSubgroupDecls);
    }

    // GL_NV_ray_tracing/GL_EXT_ray_tracing
    if (profile != EEsProfile && version >= 460) {

        const char *constRayFlags =
            "const uint gl_RayFlagsNoneNV = 0U;"
            "const uint gl_RayFlagsNoneEXT = 0U;"
            "const uint gl_RayFlagsOpaqueNV = 1U;"
            "const uint gl_RayFlagsOpaqueEXT = 1U;"
            "const uint gl_RayFlagsNoOpaqueNV = 2U;"
            "const uint gl_RayFlagsNoOpaqueEXT = 2U;"
            "const uint gl_RayFlagsTerminateOnFirstHitNV = 4U;"
            "const uint gl_RayFlagsTerminateOnFirstHitEXT = 4U;"
            "const uint gl_RayFlagsSkipClosestHitShaderNV = 8U;"
            "const uint gl_RayFlagsSkipClosestHitShaderEXT = 8U;"
            "const uint gl_RayFlagsCullBackFacingTrianglesNV = 16U;"
            "const uint gl_RayFlagsCullBackFacingTrianglesEXT = 16U;"
            "const uint gl_RayFlagsCullFrontFacingTrianglesNV = 32U;"
            "const uint gl_RayFlagsCullFrontFacingTrianglesEXT = 32U;"
            "const uint gl_RayFlagsCullOpaqueNV = 64U;"
            "const uint gl_RayFlagsCullOpaqueEXT = 64U;"
            "const uint gl_RayFlagsCullNoOpaqueNV = 128U;"
            "const uint gl_RayFlagsCullNoOpaqueEXT = 128U;"
            "const uint gl_RayFlagsSkipTrianglesEXT = 256U;"
            "const uint gl_RayFlagsSkipAABBEXT = 512U;"
            "const uint gl_RayFlagsForceOpacityMicromap2StateEXT = 1024U;"
            "const uint gl_HitKindFrontFacingTriangleEXT = 254U;"
            "const uint gl_HitKindBackFacingTriangleEXT = 255U;"
            "in    uint gl_HitKindFrontFacingMicroTriangleNV;"
            "in    uint gl_HitKindBackFacingMicroTriangleNV;"
            "\n";

        const char *constRayQueryIntersection =
            "const uint gl_RayQueryCandidateIntersectionEXT = 0U;"
            "const uint gl_RayQueryCommittedIntersectionEXT = 1U;"
            "const uint gl_RayQueryCommittedIntersectionNoneEXT = 0U;"
            "const uint gl_RayQueryCommittedIntersectionTriangleEXT = 1U;"
            "const uint gl_RayQueryCommittedIntersectionGeneratedEXT = 2U;"
            "const uint gl_RayQueryCandidateIntersectionTriangleEXT = 0U;"
            "const uint gl_RayQueryCandidateIntersectionAABBEXT = 1U;"
            "\n";

        const char *rayGenDecls =
            "in    uvec3  gl_LaunchIDNV;"
            "in    uvec3  gl_LaunchIDEXT;"
            "in    uvec3  gl_LaunchSizeNV;"
            "in    uvec3  gl_LaunchSizeEXT;"
            "\n";
        const char *intersectDecls =
            "in    uvec3  gl_LaunchIDNV;"
            "in    uvec3  gl_LaunchIDEXT;"
            "in    uvec3  gl_LaunchSizeNV;"
            "in    uvec3  gl_LaunchSizeEXT;"
            "in     int   gl_PrimitiveID;"
            "in     int   gl_InstanceID;"
            "in     int   gl_InstanceCustomIndexNV;"
            "in     int   gl_InstanceCustomIndexEXT;"
            "in     int   gl_GeometryIndexEXT;"
            "in    vec3   gl_WorldRayOriginNV;"
            "in    vec3   gl_WorldRayOriginEXT;"
            "in    vec3   gl_WorldRayDirectionNV;"
            "in    vec3   gl_WorldRayDirectionEXT;"
            "in    vec3   gl_ObjectRayOriginNV;"
            "in    vec3   gl_ObjectRayOriginEXT;"
            "in    vec3   gl_ObjectRayDirectionNV;"
            "in    vec3   gl_ObjectRayDirectionEXT;"
            "in    float  gl_RayTminNV;"
            "in    float  gl_RayTminEXT;"
            "in    float  gl_RayTmaxNV;"
            "in volatile float gl_RayTmaxEXT;"
            "in    mat4x3 gl_ObjectToWorldNV;"
            "in    mat4x3 gl_ObjectToWorldEXT;"
            "in    mat3x4 gl_ObjectToWorld3x4EXT;"
            "in    mat4x3 gl_WorldToObjectNV;"
            "in    mat4x3 gl_WorldToObjectEXT;"
            "in    mat3x4 gl_WorldToObject3x4EXT;"
            "in    uint   gl_IncomingRayFlagsNV;"
            "in    uint   gl_IncomingRayFlagsEXT;"
            "in    float  gl_CurrentRayTimeNV;"
            "in    uint   gl_CullMaskEXT;"
            "\n";
        const char *hitDecls =
            "in    uvec3  gl_LaunchIDNV;"
            "in    uvec3  gl_LaunchIDEXT;"
            "in    uvec3  gl_LaunchSizeNV;"
            "in    uvec3  gl_LaunchSizeEXT;"
            "in     int   gl_PrimitiveID;"
            "in     int   gl_InstanceID;"
            "in     int   gl_InstanceCustomIndexNV;"
            "in     int   gl_InstanceCustomIndexEXT;"
            "in     int   gl_GeometryIndexEXT;"
            "in    vec3   gl_WorldRayOriginNV;"
            "in    vec3   gl_WorldRayOriginEXT;"
            "in    vec3   gl_WorldRayDirectionNV;"
            "in    vec3   gl_WorldRayDirectionEXT;"
            "in    vec3   gl_ObjectRayOriginNV;"
            "in    vec3   gl_ObjectRayOriginEXT;"
            "in    vec3   gl_ObjectRayDirectionNV;"
            "in    vec3   gl_ObjectRayDirectionEXT;"
            "in    float  gl_RayTminNV;"
            "in    float  gl_RayTminEXT;"
            "in    float  gl_RayTmaxNV;"
            "in    float  gl_RayTmaxEXT;"
            "in    float  gl_HitTNV;"
            "in    float  gl_HitTEXT;"
            "in    uint   gl_HitKindNV;"
            "in    uint   gl_HitKindEXT;"
            "in    mat4x3 gl_ObjectToWorldNV;"
            "in    mat4x3 gl_ObjectToWorldEXT;"
            "in    mat3x4 gl_ObjectToWorld3x4EXT;"
            "in    mat4x3 gl_WorldToObjectNV;"
            "in    mat4x3 gl_WorldToObjectEXT;"
            "in    mat3x4 gl_WorldToObject3x4EXT;"
            "in    uint   gl_IncomingRayFlagsNV;"
            "in    uint   gl_IncomingRayFlagsEXT;"
            "in    float  gl_CurrentRayTimeNV;"
            "in    uint   gl_CullMaskEXT;"
            "in    vec3   gl_HitTriangleVertexPositionsEXT[3];"
            "in    vec3   gl_HitMicroTriangleVertexPositionsNV[3];"
            "in    vec2   gl_HitMicroTriangleVertexBarycentricsNV[3];"
            "\n";

        const char *missDecls =
            "in    uvec3  gl_LaunchIDNV;"
            "in    uvec3  gl_LaunchIDEXT;"
            "in    uvec3  gl_LaunchSizeNV;"
            "in    uvec3  gl_LaunchSizeEXT;"
            "in    vec3   gl_WorldRayOriginNV;"
            "in    vec3   gl_WorldRayOriginEXT;"
            "in    vec3   gl_WorldRayDirectionNV;"
            "in    vec3   gl_WorldRayDirectionEXT;"
            "in    vec3   gl_ObjectRayOriginNV;"
            "in    vec3   gl_ObjectRayDirectionNV;"
            "in    float  gl_RayTminNV;"
            "in    float  gl_RayTminEXT;"
            "in    float  gl_RayTmaxNV;"
            "in    float  gl_RayTmaxEXT;"
            "in    uint   gl_IncomingRayFlagsNV;"
            "in    uint   gl_IncomingRayFlagsEXT;"
            "in    float  gl_CurrentRayTimeNV;"
            "in    uint   gl_CullMaskEXT;"
            "\n";

        const char *callableDecls =
            "in    uvec3  gl_LaunchIDNV;"
            "in    uvec3  gl_LaunchIDEXT;"
            "in    uvec3  gl_LaunchSizeNV;"
            "in    uvec3  gl_LaunchSizeEXT;"
            "\n";


        commonBuiltins.append(constRayQueryIntersection);
        commonBuiltins.append(constRayFlags);

        stageBuiltins[EShLangRayGen].append(rayGenDecls);
        stageBuiltins[EShLangIntersect].append(intersectDecls);
        stageBuiltins[EShLangAnyHit].append(hitDecls);
        stageBuiltins[EShLangClosestHit].append(hitDecls);
        stageBuiltins[EShLangMiss].append(missDecls);
        stageBuiltins[EShLangCallable].append(callableDecls);

    }

    if ((profile != EEsProfile && version >= 140)) {
        const char *deviceIndex =
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "\n";

        stageBuiltins[EShLangRayGen].append(deviceIndex);
        stageBuiltins[EShLangIntersect].append(deviceIndex);
        stageBuiltins[EShLangAnyHit].append(deviceIndex);
        stageBuiltins[EShLangClosestHit].append(deviceIndex);
        stageBuiltins[EShLangMiss].append(deviceIndex);
    }

    if ((profile != EEsProfile && version >= 420) ||
        (profile == EEsProfile && version >= 310)) {
        commonBuiltins.append("const int gl_ScopeDevice      = 1;\n");
        commonBuiltins.append("const int gl_ScopeWorkgroup   = 2;\n");
        commonBuiltins.append("const int gl_ScopeSubgroup    = 3;\n");
        commonBuiltins.append("const int gl_ScopeInvocation  = 4;\n");
        commonBuiltins.append("const int gl_ScopeQueueFamily = 5;\n");
        commonBuiltins.append("const int gl_ScopeShaderCallEXT = 6;\n");

        commonBuiltins.append("const int gl_SemanticsRelaxed         = 0x0;\n");
        commonBuiltins.append("const int gl_SemanticsAcquire         = 0x2;\n");
        commonBuiltins.append("const int gl_SemanticsRelease         = 0x4;\n");
        commonBuiltins.append("const int gl_SemanticsAcquireRelease  = 0x8;\n");
        commonBuiltins.append("const int gl_SemanticsMakeAvailable   = 0x2000;\n");
        commonBuiltins.append("const int gl_SemanticsMakeVisible     = 0x4000;\n");
        commonBuiltins.append("const int gl_SemanticsVolatile        = 0x8000;\n");

        commonBuiltins.append("const int gl_StorageSemanticsNone     = 0x0;\n");
        commonBuiltins.append("const int gl_StorageSemanticsBuffer   = 0x40;\n");
        commonBuiltins.append("const int gl_StorageSemanticsShared   = 0x100;\n");
        commonBuiltins.append("const int gl_StorageSemanticsImage    = 0x800;\n");
        commonBuiltins.append("const int gl_StorageSemanticsOutput   = 0x1000;\n");
    }

    // Adding these to common built-ins triggers an assert due to a memory corruption in related code when testing
    // So instead add to each stage individually, avoiding the GLSLang bug
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 310)) {
        for (int stage=EShLangVertex; stage<EShLangCount; stage++)
        {
            stageBuiltins[static_cast<EShLanguage>(stage)].append("const highp int gl_ShadingRateFlag2VerticalPixelsEXT       = 1;\n");
            stageBuiltins[static_cast<EShLanguage>(stage)].append("const highp int gl_ShadingRateFlag4VerticalPixelsEXT       = 2;\n");
            stageBuiltins[static_cast<EShLanguage>(stage)].append("const highp int gl_ShadingRateFlag2HorizontalPixelsEXT     = 4;\n");
            stageBuiltins[static_cast<EShLanguage>(stage)].append("const highp int gl_ShadingRateFlag4HorizontalPixelsEXT     = 8;\n");
        }
    }
    
    // GL_EXT_shader_image_int64
    if ((profile != EEsProfile && version >= 420) ||
        (profile == EEsProfile && version >= 310)) {
            
        const TBasicType bTypes[] = { EbtInt64, EbtUint64 };
        for (int ms = 0; ms <= 1; ++ms) { // loop over "bool" multisample or not
            for (int arrayed = 0; arrayed <= 1; ++arrayed) { // loop over "bool" arrayed or not
                for (int dim = Esd1D; dim < EsdSubpass; ++dim) { // 1D, ..., buffer
                    if ((dim == Esd1D || dim == EsdRect) && profile == EEsProfile)
                        continue;
                    
                    if ((dim == Esd3D || dim == EsdRect || dim == EsdBuffer) && arrayed)
                        continue;
                    
                    if (dim != Esd2D && ms)
                        continue;
                    
                    // Loop over the bTypes
                    for (size_t bType = 0; bType < sizeof(bTypes)/sizeof(TBasicType); ++bType) {
                        //
                        // Now, make all the function prototypes for the type we just built...
                        //
                        TSampler sampler;
                    
                        sampler.setImage(bTypes[bType], (TSamplerDim)dim, arrayed ? true : false,
                                                                          false,
                                                                          ms      ? true : false);

                        TString typeName = sampler.getString();

                        addQueryFunctions(sampler, typeName, version, profile);
                        addImageFunctions(sampler, typeName, version, profile);
                    }
                }
            }
        }
    }

    // printf("%s\n", commonBuiltins.c_str());
    // printf("%s\n", stageBuiltins[EShLangFragment].c_str());
}

//
// Helper function for initialize(), to add the second set of names for texturing,
// when adding context-independent built-in functions.
//
void TBuiltIns::add2ndGenerationSamplingImaging(int version, EProfile profile, const SpvVersion& spvVersion)
{
    //
    // In this function proper, enumerate the types, then calls the next set of functions
    // to enumerate all the uses for that type.
    //

    // enumerate all the types
    const TBasicType bTypes[] = { EbtFloat, EbtInt, EbtUint,
         EbtFloat16
    };
    bool skipBuffer = (profile == EEsProfile && version < 310) || (profile != EEsProfile && version < 140);
    bool skipCubeArrayed = (profile == EEsProfile && version < 310) || (profile != EEsProfile && version < 130);
    for (int image = 0; image <= 1; ++image) // loop over "bool" image vs sampler
    {
        for (int shadow = 0; shadow <= 1; ++shadow) { // loop over "bool" shadow or not
            for (int ms = 0; ms <= 1; ++ms) // loop over "bool" multisample or not
            {
                if ((ms || image) && shadow)
                    continue;
                if (ms && profile != EEsProfile && version < 140)
                    continue;
                if (ms && image && profile == EEsProfile)
                    continue;
                if (ms && profile == EEsProfile && version < 310)
                    continue;

                for (int arrayed = 0; arrayed <= 1; ++arrayed) { // loop over "bool" arrayed or not
                    for (int dim = Esd1D; dim < EsdNumDims; ++dim) { // 1D, ..., buffer, subpass
                        if (dim == EsdAttachmentEXT)
                            continue;
                        if (dim == EsdSubpass && spvVersion.vulkan == 0)
                            continue;
                        if (dim == EsdSubpass && (image || shadow || arrayed))
                            continue;
                        if ((dim == Esd1D || dim == EsdRect) && profile == EEsProfile)
                            continue;
                        if (dim == EsdSubpass && spvVersion.vulkan == 0)
                            continue;
                        if (dim == EsdSubpass && (image || shadow || arrayed))
                            continue;
                        if ((dim == Esd1D || dim == EsdRect) && profile == EEsProfile)
                            continue;
                        if (dim != Esd2D && dim != EsdSubpass && ms)
                            continue;
                        if (dim == EsdBuffer && skipBuffer)
                            continue;
                        if (dim == EsdBuffer && (shadow || arrayed || ms))
                            continue;
                        if (ms && arrayed && profile == EEsProfile && version < 310)
                            continue;
                        if (dim == Esd3D && shadow)
                            continue;
                        if (dim == EsdCube && arrayed && skipCubeArrayed)
                            continue;
                        if ((dim == Esd3D || dim == EsdRect) && arrayed)
                            continue;

                        // Loop over the bTypes
                        for (size_t bType = 0; bType < sizeof(bTypes)/sizeof(TBasicType); ++bType) {
                            if (bTypes[bType] == EbtFloat16 && (profile == EEsProfile || version < 450))
                                continue;
                            if (dim == EsdRect && version < 140 && bType > 0)
                                continue;
                            if (shadow && (bTypes[bType] == EbtInt || bTypes[bType] == EbtUint))
                                continue;
                            //
                            // Now, make all the function prototypes for the type we just built...
                            //
                            TSampler sampler;
                            if (dim == EsdSubpass) {
                                sampler.setSubpass(bTypes[bType], ms ? true : false);
                            } else if (dim == EsdAttachmentEXT) {
                                sampler.setAttachmentEXT(bTypes[bType]);
                            } else
                            if (image) {
                                sampler.setImage(bTypes[bType], (TSamplerDim)dim, arrayed ? true : false,
                                                                                  shadow  ? true : false,
                                                                                  ms      ? true : false);
                            } else {
                                sampler.set(bTypes[bType], (TSamplerDim)dim, arrayed ? true : false,
                                                                             shadow  ? true : false,
                                                                             ms      ? true : false);
                            }

                            TString typeName = sampler.getString();

                            if (dim == EsdSubpass) {
                                addSubpassSampling(sampler, typeName, version, profile);
                                continue;
                            }

                            addQueryFunctions(sampler, typeName, version, profile);

                            if (image)
                                addImageFunctions(sampler, typeName, version, profile);
                            else {
                                addSamplingFunctions(sampler, typeName, version, profile);
                                addGatherFunctions(sampler, typeName, version, profile);
                                if (spvVersion.vulkan > 0 && sampler.isCombined() && !sampler.shadow) {
                                    // Base Vulkan allows texelFetch() for
                                    // textureBuffer (i.e. without sampler).
                                    //
                                    // GL_EXT_samplerless_texture_functions
                                    // allows texelFetch() and query functions
                                    // (other than textureQueryLod()) for all
                                    // texture types.
                                    sampler.setTexture(sampler.type, sampler.dim, sampler.arrayed, sampler.shadow,
                                                       sampler.ms);
                                    TString textureTypeName = sampler.getString();
                                    addSamplingFunctions(sampler, textureTypeName, version, profile);
                                    addQueryFunctions(sampler, textureTypeName, version, profile);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    //
    // sparseTexelsResidentARB()
    //
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append("bool sparseTexelsResidentARB(int code);\n");
    }
}

//
// Helper function for add2ndGenerationSamplingImaging(),
// when adding context-independent built-in functions.
//
// Add all the query functions for the given type.
//
void TBuiltIns::addQueryFunctions(TSampler sampler, const TString& typeName, int version, EProfile profile)
{
    //
    // textureSize() and imageSize()
    //

    int sizeDims = dimMap[sampler.dim] + (sampler.arrayed ? 1 : 0) - (sampler.dim == EsdCube ? 1 : 0);

    if (sampler.isImage() && ((profile == EEsProfile && version < 310) || (profile != EEsProfile && version < 420)))
        return;

    if (profile == EEsProfile)
        commonBuiltins.append("highp ");
    if (sizeDims == 1)
        commonBuiltins.append("int");
    else {
        commonBuiltins.append("ivec");
        commonBuiltins.append(postfixes[sizeDims]);
    }
    if (sampler.isImage())
        commonBuiltins.append(" imageSize(readonly writeonly volatile coherent ");
    else
        commonBuiltins.append(" textureSize(");
    commonBuiltins.append(typeName);
    if (! sampler.isImage() && ! sampler.isRect() && ! sampler.isBuffer() && ! sampler.isMultiSample())
        commonBuiltins.append(",int);\n");
    else
        commonBuiltins.append(");\n");

    //
    // textureSamples() and imageSamples()
    //

    // GL_ARB_shader_texture_image_samples
    // TODO: spec issue? there are no memory qualifiers; how to query a writeonly/readonly image, etc?
    if (profile != EEsProfile && version >= 430 && sampler.isMultiSample()) {
        commonBuiltins.append("int ");
        if (sampler.isImage())
            commonBuiltins.append("imageSamples(readonly writeonly volatile coherent ");
        else
            commonBuiltins.append("textureSamples(");
        commonBuiltins.append(typeName);
        commonBuiltins.append(");\n");
    }

    //
    // textureQueryLod(), fragment stage only
    // Also enabled with extension GL_ARB_texture_query_lod
    // Extension GL_ARB_texture_query_lod says that textureQueryLOD() also exist at extension.

    if (profile != EEsProfile && version >= 150 && sampler.isCombined() && sampler.dim != EsdRect &&
        ! sampler.isMultiSample() && ! sampler.isBuffer()) {

        const TString funcName[2] = {"vec2 textureQueryLod(", "vec2 textureQueryLOD("};

        for (int i = 0; i < 2; ++i){
            for (int f16TexAddr = 0; f16TexAddr < 2; ++f16TexAddr) {
                if (f16TexAddr && sampler.type != EbtFloat16)
                    continue;
                stageBuiltins[EShLangFragment].append(funcName[i]);
                stageBuiltins[EShLangFragment].append(typeName);
                if (dimMap[sampler.dim] == 1)
                    if (f16TexAddr)
                        stageBuiltins[EShLangFragment].append(", float16_t");
                    else
                        stageBuiltins[EShLangFragment].append(", float");
                else {
                    if (f16TexAddr)
                        stageBuiltins[EShLangFragment].append(", f16vec");
                    else
                        stageBuiltins[EShLangFragment].append(", vec");
                    stageBuiltins[EShLangFragment].append(postfixes[dimMap[sampler.dim]]);
                }
                stageBuiltins[EShLangFragment].append(");\n");
            }

            stageBuiltins[EShLangCompute].append(funcName[i]);
            stageBuiltins[EShLangCompute].append(typeName);
            if (dimMap[sampler.dim] == 1)
                stageBuiltins[EShLangCompute].append(", float");
            else {
                stageBuiltins[EShLangCompute].append(", vec");
                stageBuiltins[EShLangCompute].append(postfixes[dimMap[sampler.dim]]);
            }
            stageBuiltins[EShLangCompute].append(");\n");
        }
    }

    //
    // textureQueryLevels()
    //

    if (profile != EEsProfile && version >= 430 && ! sampler.isImage() && sampler.dim != EsdRect &&
        ! sampler.isMultiSample() && ! sampler.isBuffer()) {
        commonBuiltins.append("int textureQueryLevels(");
        commonBuiltins.append(typeName);
        commonBuiltins.append(");\n");
    }
}

//
// Helper function for add2ndGenerationSamplingImaging(),
// when adding context-independent built-in functions.
//
// Add all the image access functions for the given type.
//
void TBuiltIns::addImageFunctions(TSampler sampler, const TString& typeName, int version, EProfile profile)
{
    int dims = dimMap[sampler.dim];
    // most things with an array add a dimension, except for cubemaps
    if (sampler.arrayed && sampler.dim != EsdCube)
        ++dims;

    TString imageParams = typeName;
    if (dims == 1)
        imageParams.append(", int");
    else {
        imageParams.append(", ivec");
        imageParams.append(postfixes[dims]);
    }
    if (sampler.isMultiSample())
        imageParams.append(", int");

    if (profile == EEsProfile)
        commonBuiltins.append("highp ");
    commonBuiltins.append(prefixes[sampler.type]);
    commonBuiltins.append("vec4 imageLoad(readonly volatile coherent ");
    commonBuiltins.append(imageParams);
    commonBuiltins.append(");\n");

    commonBuiltins.append("void imageStore(writeonly volatile coherent ");
    commonBuiltins.append(imageParams);
    commonBuiltins.append(", ");
    commonBuiltins.append(prefixes[sampler.type]);
    commonBuiltins.append("vec4);\n");

    if (! sampler.is1D() && ! sampler.isBuffer() && profile != EEsProfile && version >= 450) {
        commonBuiltins.append("int sparseImageLoadARB(readonly volatile coherent ");
        commonBuiltins.append(imageParams);
        commonBuiltins.append(", out ");
        commonBuiltins.append(prefixes[sampler.type]);
        commonBuiltins.append("vec4");
        commonBuiltins.append(");\n");
    }

    if ( profile != EEsProfile ||
        (profile == EEsProfile && version >= 310)) {
        if (sampler.type == EbtInt || sampler.type == EbtUint || sampler.type == EbtInt64 || sampler.type == EbtUint64 ) {
            
            const char* dataType;
            switch (sampler.type) {
                case(EbtInt): dataType = "highp int"; break;
                case(EbtUint): dataType = "highp uint"; break;
                case(EbtInt64): dataType = "highp int64_t"; break;
                case(EbtUint64): dataType = "highp uint64_t"; break;
                default: dataType = "";
            }

            const int numBuiltins = 7;

            static const char* atomicFunc[numBuiltins] = {
                " imageAtomicAdd(volatile coherent ",
                " imageAtomicMin(volatile coherent ",
                " imageAtomicMax(volatile coherent ",
                " imageAtomicAnd(volatile coherent ",
                " imageAtomicOr(volatile coherent ",
                " imageAtomicXor(volatile coherent ",
                " imageAtomicExchange(volatile coherent "
            };

            // Loop twice to add prototypes with/without scope/semantics
            for (int j = 0; j < 2; ++j) {
                for (size_t i = 0; i < numBuiltins; ++i) {
                    commonBuiltins.append(dataType);
                    commonBuiltins.append(atomicFunc[i]);
                    commonBuiltins.append(imageParams);
                    commonBuiltins.append(", ");
                    commonBuiltins.append(dataType);
                    if (j == 1) {
                        commonBuiltins.append(", int, int, int");
                    }
                    commonBuiltins.append(");\n");
                }

                commonBuiltins.append(dataType);
                commonBuiltins.append(" imageAtomicCompSwap(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", ");
                commonBuiltins.append(dataType);
                commonBuiltins.append(", ");
                commonBuiltins.append(dataType);
                if (j == 1) {
                    commonBuiltins.append(", int, int, int, int, int");
                }
                commonBuiltins.append(");\n");
            }

            commonBuiltins.append(dataType);
            commonBuiltins.append(" imageAtomicLoad(volatile coherent ");
            commonBuiltins.append(imageParams);
            commonBuiltins.append(", int, int, int);\n");

            commonBuiltins.append("void imageAtomicStore(volatile coherent ");
            commonBuiltins.append(imageParams);
            commonBuiltins.append(", ");
            commonBuiltins.append(dataType);
            commonBuiltins.append(", int, int, int);\n");

        } else {
            // not int or uint
            // GL_ARB_ES3_1_compatibility
            // TODO: spec issue: are there restrictions on the kind of layout() that can be used?  what about dropping memory qualifiers?
            if (profile == EEsProfile && version >= 310) {
                commonBuiltins.append("float imageAtomicExchange(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float);\n");
            }

            // GL_NV_shader_atomic_fp16_vector
            if (profile != EEsProfile && version >= 430) {
                const int numFp16Builtins = 4;
                const char* atomicFp16Func[numFp16Builtins] = {
                    " imageAtomicAdd(volatile coherent ",
                    " imageAtomicMin(volatile coherent ",
                    " imageAtomicMax(volatile coherent ",
                    " imageAtomicExchange(volatile coherent "
                };
                const int numFp16DataTypes = 2;
                const char* atomicFp16DataTypes[numFp16DataTypes] = {
                    "f16vec2",
                    "f16vec4"
                };
                // Loop twice to add prototypes with/without scope/semantics
                for (int j = 0; j < numFp16DataTypes; ++j) {
                    for (int i = 0; i < numFp16Builtins; ++i) {
                        commonBuiltins.append(atomicFp16DataTypes[j]);
                        commonBuiltins.append(atomicFp16Func[i]);
                        commonBuiltins.append(imageParams);
                        commonBuiltins.append(", ");
                        commonBuiltins.append(atomicFp16DataTypes[j]);
                        commonBuiltins.append(");\n");
                    }
                }
            }

            if (profile != EEsProfile && version >= 450) {
                commonBuiltins.append("float imageAtomicAdd(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float);\n");

                commonBuiltins.append("float imageAtomicAdd(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float");
                commonBuiltins.append(", int, int, int);\n");

                commonBuiltins.append("float imageAtomicExchange(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float);\n");

                commonBuiltins.append("float imageAtomicExchange(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float");
                commonBuiltins.append(", int, int, int);\n");

                commonBuiltins.append("float imageAtomicLoad(readonly volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", int, int, int);\n");

                commonBuiltins.append("void imageAtomicStore(writeonly volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float");
                commonBuiltins.append(", int, int, int);\n");

                commonBuiltins.append("float imageAtomicMin(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float);\n");

                commonBuiltins.append("float imageAtomicMin(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float");
                commonBuiltins.append(", int, int, int);\n");

                commonBuiltins.append("float imageAtomicMax(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float);\n");

                commonBuiltins.append("float imageAtomicMax(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float");
                commonBuiltins.append(", int, int, int);\n");
            }
        }
    }

    if (sampler.dim == EsdRect || sampler.dim == EsdBuffer || sampler.shadow || sampler.isMultiSample())
        return;

    if (profile == EEsProfile || version < 450)
        return;

    TString imageLodParams = typeName;
    if (dims == 1)
        imageLodParams.append(", int");
    else {
        imageLodParams.append(", ivec");
        imageLodParams.append(postfixes[dims]);
    }
    imageLodParams.append(", int");

    commonBuiltins.append(prefixes[sampler.type]);
    commonBuiltins.append("vec4 imageLoadLodAMD(readonly volatile coherent ");
    commonBuiltins.append(imageLodParams);
    commonBuiltins.append(");\n");

    commonBuiltins.append("void imageStoreLodAMD(writeonly volatile coherent ");
    commonBuiltins.append(imageLodParams);
    commonBuiltins.append(", ");
    commonBuiltins.append(prefixes[sampler.type]);
    commonBuiltins.append("vec4);\n");

    if (! sampler.is1D()) {
        commonBuiltins.append("int sparseImageLoadLodAMD(readonly volatile coherent ");
        commonBuiltins.append(imageLodParams);
        commonBuiltins.append(", out ");
        commonBuiltins.append(prefixes[sampler.type]);
        commonBuiltins.append("vec4");
        commonBuiltins.append(");\n");
    }
}

//
// Helper function for initialize(),
// when adding context-independent built-in functions.
//
// Add all the subpass access functions for the given type.
//
void TBuiltIns::addSubpassSampling(TSampler sampler, const TString& typeName, int /*version*/, EProfile /*profile*/)
{
    stageBuiltins[EShLangFragment].append(prefixes[sampler.type]);
    stageBuiltins[EShLangFragment].append("vec4 subpassLoad");
    stageBuiltins[EShLangFragment].append("(");
    stageBuiltins[EShLangFragment].append(typeName.c_str());
    if (sampler.isMultiSample())
        stageBuiltins[EShLangFragment].append(", int");
    stageBuiltins[EShLangFragment].append(");\n");
}

//
// Helper function for add2ndGenerationSamplingImaging(),
// when adding context-independent built-in functions.
//
// Add all the texture lookup functions for the given type.
//
void TBuiltIns::addSamplingFunctions(TSampler sampler, const TString& typeName, int version, EProfile profile)
{
    //
    // texturing
    //
    for (int proj = 0; proj <= 1; ++proj) { // loop over "bool" projective or not

        if (proj && (sampler.dim == EsdCube || sampler.isBuffer() || sampler.arrayed || sampler.isMultiSample()
            || !sampler.isCombined()))
            continue;

        for (int lod = 0; lod <= 1; ++lod) {

            if (lod && (sampler.isBuffer() || sampler.isRect() || sampler.isMultiSample() || !sampler.isCombined()))
                continue;
            if (lod && sampler.dim == Esd2D && sampler.arrayed && sampler.shadow)
                continue;
            if (lod && sampler.dim == EsdCube && sampler.shadow)
                continue;

            for (int bias = 0; bias <= 1; ++bias) {

                if (bias && (lod || sampler.isMultiSample() || !sampler.isCombined()))
                    continue;
                if (bias && (sampler.dim == Esd2D || sampler.dim == EsdCube) && sampler.shadow && sampler.arrayed)
                    continue;
                if (bias && (sampler.isRect() || sampler.isBuffer()))
                    continue;

                for (int offset = 0; offset <= 1; ++offset) { // loop over "bool" offset or not

                    if (proj + offset + bias + lod > 3)
                        continue;
                    if (offset && (sampler.dim == EsdCube || sampler.isBuffer() || sampler.isMultiSample()))
                        continue;

                    for (int fetch = 0; fetch <= 1; ++fetch) { // loop over "bool" fetch or not

                        if (proj + offset + fetch + bias + lod > 3)
                            continue;
                        if (fetch && (lod || bias))
                            continue;
                        if (fetch && (sampler.shadow || sampler.dim == EsdCube))
                            continue;
                        if (fetch == 0 && (sampler.isMultiSample() || sampler.isBuffer()
                            || !sampler.isCombined()))
                            continue;

                        for (int grad = 0; grad <= 1; ++grad) { // loop over "bool" grad or not

                            if (grad && (lod || bias || sampler.isMultiSample() || !sampler.isCombined()))
                                continue;
                            if (grad && sampler.isBuffer())
                                continue;
                            if (proj + offset + fetch + grad + bias + lod > 3)
                                continue;

                            for (int extraProj = 0; extraProj <= 1; ++extraProj) {
                                bool compare = false;
                                int totalDims = dimMap[sampler.dim] + (sampler.arrayed ? 1 : 0);
                                // skip dummy unused second component for 1D non-array shadows
                                if (sampler.shadow && totalDims < 2)
                                    totalDims = 2;
                                totalDims += (sampler.shadow ? 1 : 0) + proj;
                                if (totalDims > 4 && sampler.shadow) {
                                    compare = true;
                                    totalDims = 4;
                                }
                                assert(totalDims <= 4);

                                if (extraProj && ! proj)
                                    continue;
                                if (extraProj && (sampler.dim == Esd3D || sampler.shadow || !sampler.isCombined()))
                                    continue;

                                // loop over 16-bit floating-point texel addressing
                                for (int f16TexAddr = 0; f16TexAddr <= 1; ++f16TexAddr)
                                {
                                    if (f16TexAddr && sampler.type != EbtFloat16)
                                        continue;
                                    if (f16TexAddr && sampler.shadow && ! compare) {
                                        compare = true; // compare argument is always present
                                        totalDims--;
                                    }
                                    // loop over "bool" lod clamp
                                    for (int lodClamp = 0; lodClamp <= 1 ;++lodClamp)
                                    {
                                        if (lodClamp && (profile == EEsProfile || version < 450))
                                            continue;
                                        if (lodClamp && (proj || lod || fetch))
                                            continue;

                                        // loop over "bool" sparse or not
                                        for (int sparse = 0; sparse <= 1; ++sparse)
                                        {
                                            if (sparse && (profile == EEsProfile || version < 450))
                                                continue;
                                            // Sparse sampling is not for 1D/1D array texture, buffer texture, and
                                            // projective texture
                                            if (sparse && (sampler.is1D() || sampler.isBuffer() || proj))
                                                continue;

                                            TString s;

                                            // return type
                                            if (sparse)
                                                s.append("int ");
                                            else {
                                                if (sampler.shadow)
                                                    if (sampler.type == EbtFloat16)
                                                        s.append("float16_t ");
                                                    else
                                                        s.append("float ");
                                                else {
                                                    s.append(prefixes[sampler.type]);
                                                    s.append("vec4 ");
                                                }
                                            }

                                            // name
                                            if (sparse) {
                                                if (fetch)
                                                    s.append("sparseTexel");
                                                else
                                                    s.append("sparseTexture");
                                            }
                                            else {
                                                if (fetch)
                                                    s.append("texel");
                                                else
                                                    s.append("texture");
                                            }
                                            if (proj)
                                                s.append("Proj");
                                            if (lod)
                                                s.append("Lod");
                                            if (grad)
                                                s.append("Grad");
                                            if (fetch)
                                                s.append("Fetch");
                                            if (offset)
                                                s.append("Offset");
                                            if (lodClamp)
                                                s.append("Clamp");
                                            if (lodClamp != 0 || sparse)
                                                s.append("ARB");
                                            s.append("(");

                                            // sampler type
                                            s.append(typeName);
                                            // P coordinate
                                            if (extraProj) {
                                                if (f16TexAddr)
                                                    s.append(",f16vec4");
                                                else
                                                    s.append(",vec4");
                                            } else {
                                                s.append(",");
                                                TBasicType t = fetch ? EbtInt : (f16TexAddr ? EbtFloat16 : EbtFloat);
                                                if (totalDims == 1)
                                                    s.append(TType::getBasicString(t));
                                                else {
                                                    s.append(prefixes[t]);
                                                    s.append("vec");
                                                    s.append(postfixes[totalDims]);
                                                }
                                            }
                                            // non-optional compare
                                            if (compare)
                                                s.append(",float");

                                            // non-optional lod argument (lod that's not driven by lod loop) or sample
                                            if ((fetch && !sampler.isBuffer() &&
                                                 !sampler.isRect() && !sampler.isMultiSample())
                                                 || (sampler.isMultiSample() && fetch))
                                                s.append(",int");
                                            // non-optional lod
                                            if (lod) {
                                                if (f16TexAddr)
                                                    s.append(",float16_t");
                                                else
                                                    s.append(",float");
                                            }

                                            // gradient arguments
                                            if (grad) {
                                                if (dimMap[sampler.dim] == 1) {
                                                    if (f16TexAddr)
                                                        s.append(",float16_t,float16_t");
                                                    else
                                                        s.append(",float,float");
                                                } else {
                                                    if (f16TexAddr)
                                                        s.append(",f16vec");
                                                    else
                                                        s.append(",vec");
                                                    s.append(postfixes[dimMap[sampler.dim]]);
                                                    if (f16TexAddr)
                                                        s.append(",f16vec");
                                                    else
                                                        s.append(",vec");
                                                    s.append(postfixes[dimMap[sampler.dim]]);
                                                }
                                            }
                                            // offset
                                            if (offset) {
                                                if (dimMap[sampler.dim] == 1)
                                                    s.append(",int");
                                                else {
                                                    s.append(",ivec");
                                                    s.append(postfixes[dimMap[sampler.dim]]);
                                                }
                                            }

                                            // lod clamp
                                            if (lodClamp) {
                                                if (f16TexAddr)
                                                    s.append(",float16_t");
                                                else
                                                    s.append(",float");
                                            }
                                            // texel out (for sparse texture)
                                            if (sparse) {
                                                s.append(",out ");
                                                if (sampler.shadow)
                                                    if (sampler.type == EbtFloat16)
                                                        s.append("float16_t");
                                                    else
                                                        s.append("float");
                                                else {
                                                    s.append(prefixes[sampler.type]);
                                                    s.append("vec4");
                                                }
                                            }
                                            // optional bias
                                            if (bias) {
                                                if (f16TexAddr)
                                                    s.append(",float16_t");
                                                else
                                                    s.append(",float");
                                            }
                                            s.append(");\n");

                                            // Add to the per-language set of built-ins
                                            if (!grad && (bias || lodClamp != 0)) {
                                                stageBuiltins[EShLangFragment].append(s);
                                                stageBuiltins[EShLangCompute].append(s);
                                            } else
                                                commonBuiltins.append(s);

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//
// Helper function for add2ndGenerationSamplingImaging(),
// when adding context-independent built-in functions.
//
// Add all the texture gather functions for the given type.
//
void TBuiltIns::addGatherFunctions(TSampler sampler, const TString& typeName, int version, EProfile profile)
{
    switch (sampler.dim) {
    case Esd2D:
    case EsdRect:
    case EsdCube:
        break;
    default:
        return;
    }

    if (sampler.isMultiSample())
        return;

    if (version < 140 && sampler.dim == EsdRect && sampler.type != EbtFloat)
        return;

    for (int f16TexAddr = 0; f16TexAddr <= 1; ++f16TexAddr) { // loop over 16-bit floating-point texel addressing

        if (f16TexAddr && sampler.type != EbtFloat16)
            continue;
        for (int offset = 0; offset < 3; ++offset) { // loop over three forms of offset in the call name:  none, Offset, and Offsets

            for (int comp = 0; comp < 2; ++comp) { // loop over presence of comp argument

                if (comp > 0 && sampler.shadow)
                    continue;

                if (offset > 0 && sampler.dim == EsdCube)
                    continue;

                for (int sparse = 0; sparse <= 1; ++sparse) { // loop over "bool" sparse or not
                    if (sparse && (profile == EEsProfile || version < 450))
                        continue;

                    TString s;

                    // return type
                    if (sparse)
                        s.append("int ");
                    else {
                        s.append(prefixes[sampler.type]);
                        s.append("vec4 ");
                    }

                    // name
                    if (sparse)
                        s.append("sparseTextureGather");
                    else
                        s.append("textureGather");
                    switch (offset) {
                    case 1:
                        s.append("Offset");
                        break;
                    case 2:
                        s.append("Offsets");
                        break;
                    default:
                        break;
                    }
                    if (sparse)
                        s.append("ARB");
                    s.append("(");

                    // sampler type argument
                    s.append(typeName);

                    // P coordinate argument
                    if (f16TexAddr)
                        s.append(",f16vec");
                    else
                        s.append(",vec");
                    int totalDims = dimMap[sampler.dim] + (sampler.arrayed ? 1 : 0);
                    s.append(postfixes[totalDims]);

                    // refZ argument
                    if (sampler.shadow)
                        s.append(",float");

                    // offset argument
                    if (offset > 0) {
                        s.append(",ivec2");
                        if (offset == 2)
                            s.append("[4]");
                    }

                    // texel out (for sparse texture)
                    if (sparse) {
                        s.append(",out ");
                        s.append(prefixes[sampler.type]);
                        s.append("vec4 ");
                    }

                    // comp argument
                    if (comp)
                        s.append(",int");

                    s.append(");\n");
                    commonBuiltins.append(s);
                }
            }
        }
    }

    if (sampler.dim == EsdRect || sampler.shadow)
        return;

    if (profile == EEsProfile || version < 450)
        return;

    for (int bias = 0; bias < 2; ++bias) { // loop over presence of bias argument

        for (int lod = 0; lod < 2; ++lod) { // loop over presence of lod argument

            if ((lod && bias) || (lod == 0 && bias == 0))
                continue;

            for (int f16TexAddr = 0; f16TexAddr <= 1; ++f16TexAddr) { // loop over 16-bit floating-point texel addressing

                if (f16TexAddr && sampler.type != EbtFloat16)
                    continue;

                for (int offset = 0; offset < 3; ++offset) { // loop over three forms of offset in the call name:  none, Offset, and Offsets

                    for (int comp = 0; comp < 2; ++comp) { // loop over presence of comp argument

                        if (comp == 0 && bias)
                            continue;

                        if (offset > 0 && sampler.dim == EsdCube)
                            continue;

                        for (int sparse = 0; sparse <= 1; ++sparse) { // loop over "bool" sparse or not
                            if (sparse && (profile == EEsProfile || version < 450))
                                continue;

                            TString s;

                            // return type
                            if (sparse)
                                s.append("int ");
                            else {
                                s.append(prefixes[sampler.type]);
                                s.append("vec4 ");
                            }

                            // name
                            if (sparse)
                                s.append("sparseTextureGather");
                            else
                                s.append("textureGather");

                            if (lod)
                                s.append("Lod");

                            switch (offset) {
                            case 1:
                                s.append("Offset");
                                break;
                            case 2:
                                s.append("Offsets");
                                break;
                            default:
                                break;
                            }

                            if (lod)
                                s.append("AMD");
                            else if (sparse)
                                s.append("ARB");

                            s.append("(");

                            // sampler type argument
                            s.append(typeName);

                            // P coordinate argument
                            if (f16TexAddr)
                                s.append(",f16vec");
                            else
                                s.append(",vec");
                            int totalDims = dimMap[sampler.dim] + (sampler.arrayed ? 1 : 0);
                            s.append(postfixes[totalDims]);

                            // lod argument
                            if (lod) {
                                if (f16TexAddr)
                                    s.append(",float16_t");
                                else
                                    s.append(",float");
                            }

                            // offset argument
                            if (offset > 0) {
                                s.append(",ivec2");
                                if (offset == 2)
                                    s.append("[4]");
                            }

                            // texel out (for sparse texture)
                            if (sparse) {
                                s.append(",out ");
                                s.append(prefixes[sampler.type]);
                                s.append("vec4 ");
                            }

                            // comp argument
                            if (comp)
                                s.append(",int");

                            // bias argument
                            if (bias) {
                                if (f16TexAddr)
                                    s.append(",float16_t");
                                else
                                    s.append(",float");
                            }

                            s.append(");\n");
                            if (bias)
                                stageBuiltins[EShLangFragment].append(s);
                            else
                                commonBuiltins.append(s);
                        }
                    }
                }
            }
        }
    }
}

//
// Add context-dependent built-in functions and variables that are present
// for the given version and profile.  All the results are put into just the
// commonBuiltins, because it is called for just a specific stage.  So,
// add stage-specific entries to the commonBuiltins, and only if that stage
// was requested.
//
void TBuiltIns::initialize(const TBuiltInResource &resources, int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language)
{
    //
    // Initialize the context-dependent (resource-dependent) built-in strings for parsing.
    //

    //============================================================================
    //
    // Standard Uniforms
    //
    //============================================================================

    TString& s = commonBuiltins;
    const int maxSize = 200;
    char builtInConstant[maxSize];

    //
    // Build string of implementation dependent constants.
    //

    if (profile == EEsProfile) {
        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxVertexAttribs = %d;", resources.maxVertexAttribs);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxVertexUniformVectors = %d;", resources.maxVertexUniformVectors);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxVertexTextureImageUnits = %d;", resources.maxVertexTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxCombinedTextureImageUnits = %d;", resources.maxCombinedTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxTextureImageUnits = %d;", resources.maxTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxFragmentUniformVectors = %d;", resources.maxFragmentUniformVectors);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxDrawBuffers = %d;", resources.maxDrawBuffers);
        s.append(builtInConstant);

        if (version == 100) {
            snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxVaryingVectors = %d;", resources.maxVaryingVectors);
            s.append(builtInConstant);
        } else {
            snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxVertexOutputVectors = %d;", resources.maxVertexOutputVectors);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxFragmentInputVectors = %d;", resources.maxFragmentInputVectors);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const mediump int  gl_MinProgramTexelOffset = %d;", resources.minProgramTexelOffset);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxProgramTexelOffset = %d;", resources.maxProgramTexelOffset);
            s.append(builtInConstant);
        }

        if (version >= 310) {
            // geometry

            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryInputComponents = %d;", resources.maxGeometryInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryOutputComponents = %d;", resources.maxGeometryOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryImageUniforms = %d;", resources.maxGeometryImageUniforms);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryTextureImageUnits = %d;", resources.maxGeometryTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryOutputVertices = %d;", resources.maxGeometryOutputVertices);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryTotalOutputComponents = %d;", resources.maxGeometryTotalOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryUniformComponents = %d;", resources.maxGeometryUniformComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryAtomicCounters = %d;", resources.maxGeometryAtomicCounters);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryAtomicCounterBuffers = %d;", resources.maxGeometryAtomicCounterBuffers);
            s.append(builtInConstant);

            // tessellation

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlInputComponents = %d;", resources.maxTessControlInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlOutputComponents = %d;", resources.maxTessControlOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlTextureImageUnits = %d;", resources.maxTessControlTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlUniformComponents = %d;", resources.maxTessControlUniformComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlTotalOutputComponents = %d;", resources.maxTessControlTotalOutputComponents);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationInputComponents = %d;", resources.maxTessEvaluationInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationOutputComponents = %d;", resources.maxTessEvaluationOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationTextureImageUnits = %d;", resources.maxTessEvaluationTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationUniformComponents = %d;", resources.maxTessEvaluationUniformComponents);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessPatchComponents = %d;", resources.maxTessPatchComponents);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int gl_MaxPatchVertices = %d;", resources.maxPatchVertices);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessGenLevel = %d;", resources.maxTessGenLevel);
            s.append(builtInConstant);

            // this is here instead of with the others in initialize(version, profile) due to the dependence on gl_MaxPatchVertices
            if (language == EShLangTessControl || language == EShLangTessEvaluation) {
                s.append(
                    "in gl_PerVertex {"
                        "highp vec4 gl_Position;"
                        "highp float gl_PointSize;"
                        "highp vec4 gl_SecondaryPositionNV;"  // GL_NV_stereo_view_rendering
                        "highp vec4 gl_PositionPerViewNV[];"  // GL_NVX_multiview_per_view_attributes
                    "} gl_in[gl_MaxPatchVertices];"
                    "\n");
            }
        }

        if (version >= 320) {
            // tessellation

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlImageUniforms = %d;", resources.maxTessControlImageUniforms);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationImageUniforms = %d;", resources.maxTessEvaluationImageUniforms);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlAtomicCounters = %d;", resources.maxTessControlAtomicCounters);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationAtomicCounters = %d;", resources.maxTessEvaluationAtomicCounters);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlAtomicCounterBuffers = %d;", resources.maxTessControlAtomicCounterBuffers);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationAtomicCounterBuffers = %d;", resources.maxTessEvaluationAtomicCounterBuffers);
            s.append(builtInConstant);
        }

        if (version >= 100) {
            // GL_EXT_blend_func_extended
            snprintf(builtInConstant, maxSize, "const mediump int gl_MaxDualSourceDrawBuffersEXT = %d;", resources.maxDualSourceDrawBuffersEXT);
            s.append(builtInConstant);
            // this is here instead of with the others in initialize(version, profile) due to the dependence on gl_MaxDualSourceDrawBuffersEXT
            if (language == EShLangFragment) {
                s.append(
                    "mediump vec4 gl_SecondaryFragColorEXT;"
                    "mediump vec4 gl_SecondaryFragDataEXT[gl_MaxDualSourceDrawBuffersEXT];"
                    "\n");
            }
        }
    } else {
        // non-ES profile

        if (version > 400) {
            snprintf(builtInConstant, maxSize, "const int  gl_MaxVertexUniformVectors = %d;", resources.maxVertexUniformVectors);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int  gl_MaxFragmentUniformVectors = %d;", resources.maxFragmentUniformVectors);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int  gl_MaxVaryingVectors = %d;", resources.maxVaryingVectors);
            s.append(builtInConstant);
        }

        snprintf(builtInConstant, maxSize, "const int  gl_MaxVertexAttribs = %d;", resources.maxVertexAttribs);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxVertexTextureImageUnits = %d;", resources.maxVertexTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxCombinedTextureImageUnits = %d;", resources.maxCombinedTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxTextureImageUnits = %d;", resources.maxTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxDrawBuffers = %d;", resources.maxDrawBuffers);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxLights = %d;", resources.maxLights);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxClipPlanes = %d;", resources.maxClipPlanes);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxTextureUnits = %d;", resources.maxTextureUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxTextureCoords = %d;", resources.maxTextureCoords);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxVertexUniformComponents = %d;", resources.maxVertexUniformComponents);
        s.append(builtInConstant);

        // Moved from just being deprecated into compatibility profile only as of 4.20
        if (version < 420 || profile == ECompatibilityProfile) {
            snprintf(builtInConstant, maxSize, "const int  gl_MaxVaryingFloats = %d;", resources.maxVaryingFloats);
            s.append(builtInConstant);
        }

        snprintf(builtInConstant, maxSize, "const int  gl_MaxFragmentUniformComponents = %d;", resources.maxFragmentUniformComponents);
        s.append(builtInConstant);

        if (spvVersion.spv == 0 && IncludeLegacy(version, profile, spvVersion)) {
            //
            // OpenGL'uniform' state.  Page numbers are in reference to version
            // 1.4 of the OpenGL specification.
            //

            //
            // Matrix state. p. 31, 32, 37, 39, 40.
            //
            s.append("uniform mat4  gl_TextureMatrix[gl_MaxTextureCoords];"

            //
            // Derived matrix state that provides inverse and transposed versions
            // of the matrices above.
            //
                        "uniform mat4  gl_TextureMatrixInverse[gl_MaxTextureCoords];"

                        "uniform mat4  gl_TextureMatrixTranspose[gl_MaxTextureCoords];"

                        "uniform mat4  gl_TextureMatrixInverseTranspose[gl_MaxTextureCoords];"

            //
            // Clip planes p. 42.
            //
                        "uniform vec4  gl_ClipPlane[gl_MaxClipPlanes];"

            //
            // Light State p 50, 53, 55.
            //
                        "uniform gl_LightSourceParameters  gl_LightSource[gl_MaxLights];"

            //
            // Derived state from products of light.
            //
                        "uniform gl_LightProducts gl_FrontLightProduct[gl_MaxLights];"
                        "uniform gl_LightProducts gl_BackLightProduct[gl_MaxLights];"

            //
            // Texture Environment and Generation, p. 152, p. 40-42.
            //
                        "uniform vec4  gl_TextureEnvColor[gl_MaxTextureImageUnits];"
                        "uniform vec4  gl_EyePlaneS[gl_MaxTextureCoords];"
                        "uniform vec4  gl_EyePlaneT[gl_MaxTextureCoords];"
                        "uniform vec4  gl_EyePlaneR[gl_MaxTextureCoords];"
                        "uniform vec4  gl_EyePlaneQ[gl_MaxTextureCoords];"
                        "uniform vec4  gl_ObjectPlaneS[gl_MaxTextureCoords];"
                        "uniform vec4  gl_ObjectPlaneT[gl_MaxTextureCoords];"
                        "uniform vec4  gl_ObjectPlaneR[gl_MaxTextureCoords];"
                        "uniform vec4  gl_ObjectPlaneQ[gl_MaxTextureCoords];");
        }

        if (version >= 130) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxClipDistances = %d;", resources.maxClipDistances);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxVaryingComponents = %d;", resources.maxVaryingComponents);
            s.append(builtInConstant);

            // GL_ARB_shading_language_420pack
            snprintf(builtInConstant, maxSize, "const mediump int  gl_MinProgramTexelOffset = %d;", resources.minProgramTexelOffset);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxProgramTexelOffset = %d;", resources.maxProgramTexelOffset);
            s.append(builtInConstant);
        }

        // geometry
        if (version >= 150) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryInputComponents = %d;", resources.maxGeometryInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryOutputComponents = %d;", resources.maxGeometryOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryTextureImageUnits = %d;", resources.maxGeometryTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryOutputVertices = %d;", resources.maxGeometryOutputVertices);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryTotalOutputComponents = %d;", resources.maxGeometryTotalOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryUniformComponents = %d;", resources.maxGeometryUniformComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryVaryingComponents = %d;", resources.maxGeometryVaryingComponents);
            s.append(builtInConstant);

        }

        if (version >= 150) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxVertexOutputComponents = %d;", resources.maxVertexOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxFragmentInputComponents = %d;", resources.maxFragmentInputComponents);
            s.append(builtInConstant);
        }

        // tessellation
        if (version >= 150) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlInputComponents = %d;", resources.maxTessControlInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlOutputComponents = %d;", resources.maxTessControlOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlTextureImageUnits = %d;", resources.maxTessControlTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlUniformComponents = %d;", resources.maxTessControlUniformComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlTotalOutputComponents = %d;", resources.maxTessControlTotalOutputComponents);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationInputComponents = %d;", resources.maxTessEvaluationInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationOutputComponents = %d;", resources.maxTessEvaluationOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationTextureImageUnits = %d;", resources.maxTessEvaluationTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationUniformComponents = %d;", resources.maxTessEvaluationUniformComponents);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessPatchComponents = %d;", resources.maxTessPatchComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessGenLevel = %d;", resources.maxTessGenLevel);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxPatchVertices = %d;", resources.maxPatchVertices);
            s.append(builtInConstant);

            // this is here instead of with the others in initialize(version, profile) due to the dependence on gl_MaxPatchVertices
            if (language == EShLangTessControl || language == EShLangTessEvaluation) {
                s.append(
                    "in gl_PerVertex {"
                        "vec4 gl_Position;"
                        "float gl_PointSize;"
                        "float gl_ClipDistance[];"
                    );
                if (profile == ECompatibilityProfile)
                    s.append(
                        "vec4 gl_ClipVertex;"
                        "vec4 gl_FrontColor;"
                        "vec4 gl_BackColor;"
                        "vec4 gl_FrontSecondaryColor;"
                        "vec4 gl_BackSecondaryColor;"
                        "vec4 gl_TexCoord[];"
                        "float gl_FogFragCoord;"
                        );
                if (profile != EEsProfile && version >= 450)
                    s.append(
                        "float gl_CullDistance[];"
                        "vec4 gl_SecondaryPositionNV;"  // GL_NV_stereo_view_rendering
                        "vec4 gl_PositionPerViewNV[];"  // GL_NVX_multiview_per_view_attributes
                       );
                s.append(
                    "} gl_in[gl_MaxPatchVertices];"
                    "\n");
            }
        }

        if (version >= 150) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxViewports = %d;", resources.maxViewports);
            s.append(builtInConstant);
        }

        // images
        if (version >= 130) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedImageUnitsAndFragmentOutputs = %d;", resources.maxCombinedImageUnitsAndFragmentOutputs);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxImageSamples = %d;", resources.maxImageSamples);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlImageUniforms = %d;", resources.maxTessControlImageUniforms);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationImageUniforms = %d;", resources.maxTessEvaluationImageUniforms);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryImageUniforms = %d;", resources.maxGeometryImageUniforms);
            s.append(builtInConstant);
        }

        // enhanced layouts
        if (version >= 430) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxTransformFeedbackBuffers = %d;", resources.maxTransformFeedbackBuffers);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTransformFeedbackInterleavedComponents = %d;", resources.maxTransformFeedbackInterleavedComponents);
            s.append(builtInConstant);
        }
    }

    // compute
    if ((profile == EEsProfile && version >= 310) || (profile != EEsProfile && version >= 420)) {
        snprintf(builtInConstant, maxSize, "const ivec3 gl_MaxComputeWorkGroupCount = ivec3(%d,%d,%d);", resources.maxComputeWorkGroupCountX,
                                                                                                         resources.maxComputeWorkGroupCountY,
                                                                                                         resources.maxComputeWorkGroupCountZ);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const ivec3 gl_MaxComputeWorkGroupSize = ivec3(%d,%d,%d);", resources.maxComputeWorkGroupSizeX,
                                                                                                        resources.maxComputeWorkGroupSizeY,
                                                                                                        resources.maxComputeWorkGroupSizeZ);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int gl_MaxComputeUniformComponents = %d;", resources.maxComputeUniformComponents);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxComputeTextureImageUnits = %d;", resources.maxComputeTextureImageUnits);
        s.append(builtInConstant);

        s.append("\n");
    }

    // images (some in compute below)
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 130)) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxImageUnits = %d;", resources.maxImageUnits);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedShaderOutputResources = %d;", resources.maxCombinedShaderOutputResources);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxVertexImageUniforms = %d;", resources.maxVertexImageUniforms);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxFragmentImageUniforms = %d;", resources.maxFragmentImageUniforms);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedImageUniforms = %d;", resources.maxCombinedImageUniforms);
        s.append(builtInConstant);
    }

    // compute
    if ((profile == EEsProfile && version >= 310) || (profile != EEsProfile && version >= 420)) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxComputeImageUniforms = %d;", resources.maxComputeImageUniforms);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxComputeAtomicCounters = %d;", resources.maxComputeAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxComputeAtomicCounterBuffers = %d;", resources.maxComputeAtomicCounterBuffers);
        s.append(builtInConstant);

        s.append("\n");
    }

    // atomic counters (some in compute below)
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 420)) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxVertexAtomicCounters = %d;", resources.               maxVertexAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxFragmentAtomicCounters = %d;", resources.             maxFragmentAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedAtomicCounters = %d;", resources.             maxCombinedAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxAtomicCounterBindings = %d;", resources.              maxAtomicCounterBindings);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxVertexAtomicCounterBuffers = %d;", resources.         maxVertexAtomicCounterBuffers);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxFragmentAtomicCounterBuffers = %d;", resources.       maxFragmentAtomicCounterBuffers);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedAtomicCounterBuffers = %d;", resources.       maxCombinedAtomicCounterBuffers);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxAtomicCounterBufferSize = %d;", resources.            maxAtomicCounterBufferSize);
        s.append(builtInConstant);
    }
    if (profile != EEsProfile && version >= 420) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlAtomicCounters = %d;", resources.          maxTessControlAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationAtomicCounters = %d;", resources.       maxTessEvaluationAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryAtomicCounters = %d;", resources.             maxGeometryAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlAtomicCounterBuffers = %d;", resources.    maxTessControlAtomicCounterBuffers);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationAtomicCounterBuffers = %d;", resources. maxTessEvaluationAtomicCounterBuffers);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryAtomicCounterBuffers = %d;", resources.       maxGeometryAtomicCounterBuffers);
        s.append(builtInConstant);

        s.append("\n");
    }

    // GL_ARB_cull_distance
    if (profile != EEsProfile && version >= 450) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxCullDistances = %d;",                resources.maxCullDistances);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedClipAndCullDistances = %d;", resources.maxCombinedClipAndCullDistances);
        s.append(builtInConstant);
    }

    // GL_ARB_ES3_1_compatibility
    if ((profile != EEsProfile && version >= 450) ||
        (profile == EEsProfile && version >= 310)) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxSamples = %d;", resources.maxSamples);
        s.append(builtInConstant);
    }

    // SPV_NV_mesh_shader
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxMeshOutputVerticesNV = %d;", resources.maxMeshOutputVerticesNV);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int gl_MaxMeshOutputPrimitivesNV = %d;", resources.maxMeshOutputPrimitivesNV);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const ivec3 gl_MaxMeshWorkGroupSizeNV = ivec3(%d,%d,%d);", resources.maxMeshWorkGroupSizeX_NV,
                                                                                                       resources.maxMeshWorkGroupSizeY_NV,
                                                                                                       resources.maxMeshWorkGroupSizeZ_NV);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const ivec3 gl_MaxTaskWorkGroupSizeNV = ivec3(%d,%d,%d);", resources.maxTaskWorkGroupSizeX_NV,
                                                                                                       resources.maxTaskWorkGroupSizeY_NV,
                                                                                                       resources.maxTaskWorkGroupSizeZ_NV);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int gl_MaxMeshViewCountNV = %d;", resources.maxMeshViewCountNV);
        s.append(builtInConstant);

        s.append("\n");
    }

    s.append("\n");
}

//
// To support special built-ins that have a special qualifier that cannot be declared textually
// in a shader, like gl_Position.
//
// This lets the type of the built-in be declared textually, and then have just its qualifier be
// updated afterward.
//
// Safe to call even if name is not present.
//
// Only use this for built-in variables that have a special qualifier in TStorageQualifier.
// New built-in variables should use a generic (textually declarable) qualifier in
// TStoraregQualifier and only call BuiltInVariable().
//
static void SpecialQualifier(const char* name, TStorageQualifier qualifier, TBuiltInVariable builtIn, TSymbolTable& symbolTable)
{
    TSymbol* symbol = symbolTable.find(name);
    if (symbol == nullptr)
        return;

    TQualifier& symQualifier = symbol->getWritableType().getQualifier();
    symQualifier.storage = qualifier;
    symQualifier.builtIn = builtIn;
}

//
// Modify the symbol's flat decoration.
//
// Safe to call even if name is not present.
//
// Originally written to transform gl_SubGroupSizeARB from uniform to fragment input in Vulkan.
//
static void ModifyFlatDecoration(const char* name, bool flat, TSymbolTable& symbolTable)
{
    TSymbol* symbol = symbolTable.find(name);
    if (symbol == nullptr)
        return;

    TQualifier& symQualifier = symbol->getWritableType().getQualifier();
    symQualifier.flat = flat;
}

//
// To tag built-in variables with their TBuiltInVariable enum.  Use this when the
// normal declaration text already gets the qualifier right, and all that's needed
// is setting the builtIn field.  This should be the normal way for all new
// built-in variables.
//
// If SpecialQualifier() was called, this does not need to be called.
//
// Safe to call even if name is not present.
//
static void BuiltInVariable(const char* name, TBuiltInVariable builtIn, TSymbolTable& symbolTable)
{
    TSymbol* symbol = symbolTable.find(name);
    if (symbol == nullptr)
        return;

    TQualifier& symQualifier = symbol->getWritableType().getQualifier();
    symQualifier.builtIn = builtIn;
}

static void RetargetVariable(const char* from, const char* to, TSymbolTable& symbolTable)
{
    symbolTable.retargetSymbol(from, to);
}

//
// For built-in variables inside a named block.
// SpecialQualifier() won't ever go inside a block; their member's qualifier come
// from the qualification of the block.
//
// See comments above for other detail.
//
static void BuiltInVariable(const char* blockName, const char* name, TBuiltInVariable builtIn, TSymbolTable& symbolTable)
{
    TSymbol* symbol = symbolTable.find(blockName);
    if (symbol == nullptr)
        return;

    TTypeList& structure = *symbol->getWritableType().getWritableStruct();
    for (int i = 0; i < (int)structure.size(); ++i) {
        if (structure[i].type->getFieldName().compare(name) == 0) {
            structure[i].type->getQualifier().builtIn = builtIn;
            return;
        }
    }
}

//
// Finish adding/processing context-independent built-in symbols.
// 1) Programmatically add symbols that could not be added by simple text strings above.
// 2) Map built-in functions to operators, for those that will turn into an operation node
//    instead of remaining a function call.
// 3) Tag extension-related symbols added to their base version with their extensions, so
//    that if an early version has the extension turned off, there is an error reported on use.
//
void TBuiltIns::identifyBuiltIns(int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language, TSymbolTable& symbolTable)
{
    //
    // Tag built-in variables and functions with additional qualifier and extension information
    // that cannot be declared with the text strings.
    //

    // N.B.: a symbol should only be tagged once, and this function is called multiple times, once
    // per stage that's used for this profile.  So
    //  - generally, stick common ones in the fragment stage to ensure they are tagged exactly once
    //  - for ES, which has different precisions for different stages, the coarsest-grained tagging
    //    for a built-in used in many stages needs to be once for the fragment stage and once for
    //    the vertex stage

    switch(language) {
    case EShLangVertex:
        if (spvVersion.vulkan > 0) {
            BuiltInVariable("gl_VertexIndex",   EbvVertexIndex,   symbolTable);
            BuiltInVariable("gl_InstanceIndex", EbvInstanceIndex, symbolTable);
        }

        if (spvVersion.vulkan == 0) {
            SpecialQualifier("gl_VertexID",   EvqVertexId,   EbvVertexId,   symbolTable);
            SpecialQualifier("gl_InstanceID", EvqInstanceId, EbvInstanceId, symbolTable);
            if (version < 140)
                symbolTable.setVariableExtensions("gl_InstanceID", 1, &E_GL_EXT_draw_instanced);
        }

        if (spvVersion.vulkan > 0 && spvVersion.vulkanRelaxed) {
            // treat these built-ins as aliases of VertexIndex and InstanceIndex
            RetargetVariable("gl_InstanceID", "gl_InstanceIndex", symbolTable);
            RetargetVariable("gl_VertexID", "gl_VertexIndex", symbolTable);
        }

        if (profile != EEsProfile) {
            if (version >= 440) {
                symbolTable.setVariableExtensions("gl_BaseVertexARB",   1, &E_GL_ARB_shader_draw_parameters);
                symbolTable.setVariableExtensions("gl_BaseInstanceARB", 1, &E_GL_ARB_shader_draw_parameters);
                symbolTable.setVariableExtensions("gl_DrawIDARB",       1, &E_GL_ARB_shader_draw_parameters);
                BuiltInVariable("gl_BaseVertexARB",   EbvBaseVertex,   symbolTable);
                BuiltInVariable("gl_BaseInstanceARB", EbvBaseInstance, symbolTable);
                BuiltInVariable("gl_DrawIDARB",       EbvDrawId,       symbolTable);
            }
            if (version >= 460) {
                BuiltInVariable("gl_BaseVertex",   EbvBaseVertex,   symbolTable);
                BuiltInVariable("gl_BaseInstance", EbvBaseInstance, symbolTable);
                BuiltInVariable("gl_DrawID",       EbvDrawId,       symbolTable);
            }
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupInvocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            symbolTable.setFunctionExtensions("ballotARB",              1, &E_GL_ARB_shader_ballot);
            symbolTable.setFunctionExtensions("readInvocationARB",      1, &E_GL_ARB_shader_ballot);
            symbolTable.setFunctionExtensions("readFirstInvocationARB", 1, &E_GL_ARB_shader_ballot);

            if (version >= 430) {
                symbolTable.setFunctionExtensions("anyInvocationARB",       1, &E_GL_ARB_shader_group_vote);
                symbolTable.setFunctionExtensions("allInvocationsARB",      1, &E_GL_ARB_shader_group_vote);
                symbolTable.setFunctionExtensions("allInvocationsEqualARB", 1, &E_GL_ARB_shader_group_vote);
            }
        }


        if (profile != EEsProfile) {
            symbolTable.setFunctionExtensions("minInvocationsAMD",                1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxInvocationsAMD",                1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addInvocationsAMD",                1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("minInvocationsNonUniformAMD",      1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxInvocationsNonUniformAMD",      1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addInvocationsNonUniformAMD",      1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("swizzleInvocationsAMD",            1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("swizzleInvocationsWithPatternAMD", 1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("writeInvocationAMD",               1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("mbcntAMD",                         1, &E_GL_AMD_shader_ballot);

            symbolTable.setFunctionExtensions("minInvocationsInclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxInvocationsInclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addInvocationsInclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("minInvocationsInclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxInvocationsInclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addInvocationsInclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("minInvocationsExclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxInvocationsExclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addInvocationsExclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("minInvocationsExclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxInvocationsExclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addInvocationsExclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
        }

        if (profile != EEsProfile) {
            symbolTable.setFunctionExtensions("min3", 1, &E_GL_AMD_shader_trinary_minmax);
            symbolTable.setFunctionExtensions("max3", 1, &E_GL_AMD_shader_trinary_minmax);
            symbolTable.setFunctionExtensions("mid3", 1, &E_GL_AMD_shader_trinary_minmax);
        }

        if (profile != EEsProfile) {
            symbolTable.setVariableExtensions("gl_SIMDGroupSizeAMD", 1, &E_GL_AMD_gcn_shader);
            SpecialQualifier("gl_SIMDGroupSizeAMD", EvqVaryingIn, EbvSubGroupSize, symbolTable);

            symbolTable.setFunctionExtensions("cubeFaceIndexAMD", 1, &E_GL_AMD_gcn_shader);
            symbolTable.setFunctionExtensions("cubeFaceCoordAMD", 1, &E_GL_AMD_gcn_shader);
            symbolTable.setFunctionExtensions("timeAMD",          1, &E_GL_AMD_gcn_shader);
        }

        if (profile != EEsProfile) {
            symbolTable.setFunctionExtensions("fragmentMaskFetchAMD", 1, &E_GL_AMD_shader_fragment_mask);
            symbolTable.setFunctionExtensions("fragmentFetchAMD",     1, &E_GL_AMD_shader_fragment_mask);
        }

        symbolTable.setFunctionExtensions("countLeadingZeros",  1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("countTrailingZeros", 1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("absoluteDifference", 1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("addSaturate",        1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("subtractSaturate",   1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("average",            1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("averageRounded",     1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("multiply32x16",      1, &E_GL_INTEL_shader_integer_functions2);

        symbolTable.setFunctionExtensions("textureFootprintNV",          1, &E_GL_NV_shader_texture_footprint);
        symbolTable.setFunctionExtensions("textureFootprintClampNV",     1, &E_GL_NV_shader_texture_footprint);
        symbolTable.setFunctionExtensions("textureFootprintLodNV",       1, &E_GL_NV_shader_texture_footprint);
        symbolTable.setFunctionExtensions("textureFootprintGradNV",      1, &E_GL_NV_shader_texture_footprint);
        symbolTable.setFunctionExtensions("textureFootprintGradClampNV", 1, &E_GL_NV_shader_texture_footprint);
        // Compatibility variables, vertex only
        if (spvVersion.spv == 0) {
            BuiltInVariable("gl_Color",          EbvColor,          symbolTable);
            BuiltInVariable("gl_SecondaryColor", EbvSecondaryColor, symbolTable);
            BuiltInVariable("gl_Normal",         EbvNormal,         symbolTable);
            BuiltInVariable("gl_Vertex",         EbvVertex,         symbolTable);
            BuiltInVariable("gl_MultiTexCoord0", EbvMultiTexCoord0, symbolTable);
            BuiltInVariable("gl_MultiTexCoord1", EbvMultiTexCoord1, symbolTable);
            BuiltInVariable("gl_MultiTexCoord2", EbvMultiTexCoord2, symbolTable);
            BuiltInVariable("gl_MultiTexCoord3", EbvMultiTexCoord3, symbolTable);
            BuiltInVariable("gl_MultiTexCoord4", EbvMultiTexCoord4, symbolTable);
            BuiltInVariable("gl_MultiTexCoord5", EbvMultiTexCoord5, symbolTable);
            BuiltInVariable("gl_MultiTexCoord6", EbvMultiTexCoord6, symbolTable);
            BuiltInVariable("gl_MultiTexCoord7", EbvMultiTexCoord7, symbolTable);
            BuiltInVariable("gl_FogCoord",       EbvFogFragCoord,   symbolTable);
        }

        if (profile == EEsProfile) {
            if (spvVersion.spv == 0) {
                symbolTable.setFunctionExtensions("texture2DGradEXT",     1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DProjGradEXT", 1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("textureCubeGradEXT",   1, &E_GL_EXT_shader_texture_lod);
                if (version == 310)
                    symbolTable.setFunctionExtensions("textureGatherOffsets", Num_AEP_gpu_shader5, AEP_gpu_shader5);
            }
            if (version == 310)
                symbolTable.setFunctionExtensions("fma", Num_AEP_gpu_shader5, AEP_gpu_shader5);
        }

        if (profile == EEsProfile && version < 320) {
            symbolTable.setFunctionExtensions("imageAtomicAdd",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicMin",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicMax",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicAnd",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicOr",       1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicXor",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicExchange", 1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicCompSwap", 1, &E_GL_OES_shader_image_atomic);
        }

        if (version >= 300 /* both ES and non-ES */) {
            symbolTable.setVariableExtensions("gl_ViewID_OVR", Num_OVR_multiview_EXTs, OVR_multiview_EXTs);
            BuiltInVariable("gl_ViewID_OVR", EbvViewIndex, symbolTable);
        }

        if (profile == EEsProfile) {
            symbolTable.setFunctionExtensions("shadow2DEXT",        1, &E_GL_EXT_shadow_samplers);
            symbolTable.setFunctionExtensions("shadow2DProjEXT",    1, &E_GL_EXT_shadow_samplers);
        }

        // E_GL_EXT_texture_array
        if (profile != EEsProfile && spvVersion.spv == 0) {
            symbolTable.setFunctionExtensions("texture1DArray", 1, &E_GL_EXT_texture_array);
            symbolTable.setFunctionExtensions("texture2DArray", 1, &E_GL_EXT_texture_array);
            symbolTable.setFunctionExtensions("shadow1DArray", 1, &E_GL_EXT_texture_array);
            symbolTable.setFunctionExtensions("shadow2DArray", 1, &E_GL_EXT_texture_array);

            symbolTable.setFunctionExtensions("texture1DArrayLod", 1, &E_GL_EXT_texture_array);
            symbolTable.setFunctionExtensions("texture2DArrayLod", 1, &E_GL_EXT_texture_array);
            symbolTable.setFunctionExtensions("shadow1DArrayLod", 1, &E_GL_EXT_texture_array);
        }
        [[fallthrough]];

    case EShLangTessControl:
        if (profile == EEsProfile && version >= 310) {
            BuiltInVariable("gl_BoundingBoxEXT", EbvBoundingBox, symbolTable);
            symbolTable.setVariableExtensions("gl_BoundingBoxEXT", 1,
                                              &E_GL_EXT_primitive_bounding_box);
            BuiltInVariable("gl_BoundingBoxOES", EbvBoundingBox, symbolTable);
            symbolTable.setVariableExtensions("gl_BoundingBoxOES", 1,
                                              &E_GL_OES_primitive_bounding_box);

            if (version >= 320) {
                BuiltInVariable("gl_BoundingBox", EbvBoundingBox, symbolTable);
            }
        }
        [[fallthrough]];

    case EShLangTessEvaluation:
    case EShLangGeometry:
        SpecialQualifier("gl_Position",   EvqPosition,   EbvPosition,   symbolTable);
        SpecialQualifier("gl_PointSize",  EvqPointSize,  EbvPointSize,  symbolTable);

        BuiltInVariable("gl_in",  "gl_Position",     EbvPosition,     symbolTable);
        BuiltInVariable("gl_in",  "gl_PointSize",    EbvPointSize,    symbolTable);

        BuiltInVariable("gl_out", "gl_Position",     EbvPosition,     symbolTable);
        BuiltInVariable("gl_out", "gl_PointSize",    EbvPointSize,    symbolTable);

        SpecialQualifier("gl_ClipVertex", EvqClipVertex, EbvClipVertex, symbolTable);

        BuiltInVariable("gl_in",  "gl_ClipDistance", EbvClipDistance, symbolTable);
        BuiltInVariable("gl_in",  "gl_CullDistance", EbvCullDistance, symbolTable);

        BuiltInVariable("gl_out", "gl_ClipDistance", EbvClipDistance, symbolTable);
        BuiltInVariable("gl_out", "gl_CullDistance", EbvCullDistance, symbolTable);

        BuiltInVariable("gl_ClipDistance",    EbvClipDistance,   symbolTable);
        BuiltInVariable("gl_CullDistance",    EbvCullDistance,   symbolTable);
        BuiltInVariable("gl_PrimitiveIDIn",   EbvPrimitiveId,    symbolTable);
        BuiltInVariable("gl_PrimitiveID",     EbvPrimitiveId,    symbolTable);
        BuiltInVariable("gl_InvocationID",    EbvInvocationId,   symbolTable);
        BuiltInVariable("gl_Layer",           EbvLayer,          symbolTable);
        BuiltInVariable("gl_ViewportIndex",   EbvViewportIndex,  symbolTable);

        if (language != EShLangGeometry) {
            symbolTable.setVariableExtensions("gl_Layer",         Num_viewportEXTs, viewportEXTs);
            symbolTable.setVariableExtensions("gl_ViewportIndex", Num_viewportEXTs, viewportEXTs);
        }
        symbolTable.setVariableExtensions("gl_ViewportMask",            1, &E_GL_NV_viewport_array2);
        symbolTable.setVariableExtensions("gl_SecondaryPositionNV",     1, &E_GL_NV_stereo_view_rendering);
        symbolTable.setVariableExtensions("gl_SecondaryViewportMaskNV", 1, &E_GL_NV_stereo_view_rendering);
        symbolTable.setVariableExtensions("gl_PositionPerViewNV",       1, &E_GL_NVX_multiview_per_view_attributes);
        symbolTable.setVariableExtensions("gl_ViewportMaskPerViewNV",   1, &E_GL_NVX_multiview_per_view_attributes);

        BuiltInVariable("gl_ViewportMask",              EbvViewportMaskNV,          symbolTable);
        BuiltInVariable("gl_SecondaryPositionNV",       EbvSecondaryPositionNV,     symbolTable);
        BuiltInVariable("gl_SecondaryViewportMaskNV",   EbvSecondaryViewportMaskNV, symbolTable);
        BuiltInVariable("gl_PositionPerViewNV",         EbvPositionPerViewNV,       symbolTable);
        BuiltInVariable("gl_ViewportMaskPerViewNV",     EbvViewportMaskPerViewNV,   symbolTable);

        if (language == EShLangVertex || language == EShLangGeometry) {
            symbolTable.setVariableExtensions("gl_in", "gl_SecondaryPositionNV", 1, &E_GL_NV_stereo_view_rendering);
            symbolTable.setVariableExtensions("gl_in", "gl_PositionPerViewNV",   1, &E_GL_NVX_multiview_per_view_attributes);

            BuiltInVariable("gl_in", "gl_SecondaryPositionNV", EbvSecondaryPositionNV, symbolTable);
            BuiltInVariable("gl_in", "gl_PositionPerViewNV",   EbvPositionPerViewNV,   symbolTable);
        }
        symbolTable.setVariableExtensions("gl_out", "gl_ViewportMask",            1, &E_GL_NV_viewport_array2);
        symbolTable.setVariableExtensions("gl_out", "gl_SecondaryPositionNV",     1, &E_GL_NV_stereo_view_rendering);
        symbolTable.setVariableExtensions("gl_out", "gl_SecondaryViewportMaskNV", 1, &E_GL_NV_stereo_view_rendering);
        symbolTable.setVariableExtensions("gl_out", "gl_PositionPerViewNV",       1, &E_GL_NVX_multiview_per_view_attributes);
        symbolTable.setVariableExtensions("gl_out", "gl_ViewportMaskPerViewNV",   1, &E_GL_NVX_multiview_per_view_attributes);

        BuiltInVariable("gl_out", "gl_ViewportMask",            EbvViewportMaskNV,          symbolTable);
        BuiltInVariable("gl_out", "gl_SecondaryPositionNV",     EbvSecondaryPositionNV,     symbolTable);
        BuiltInVariable("gl_out", "gl_SecondaryViewportMaskNV", EbvSecondaryViewportMaskNV, symbolTable);
        BuiltInVariable("gl_out", "gl_PositionPerViewNV",       EbvPositionPerViewNV,       symbolTable);
        BuiltInVariable("gl_out", "gl_ViewportMaskPerViewNV",   EbvViewportMaskPerViewNV,   symbolTable);

        BuiltInVariable("gl_PatchVerticesIn", EbvPatchVertices,  symbolTable);
        BuiltInVariable("gl_TessLevelOuter",  EbvTessLevelOuter, symbolTable);
        BuiltInVariable("gl_TessLevelInner",  EbvTessLevelInner, symbolTable);
        BuiltInVariable("gl_TessCoord",       EbvTessCoord,      symbolTable);

        if (version < 410)
            symbolTable.setVariableExtensions("gl_ViewportIndex", 1, &E_GL_ARB_viewport_array);

        // Compatibility variables

        BuiltInVariable("gl_in", "gl_ClipVertex",          EbvClipVertex,          symbolTable);
        BuiltInVariable("gl_in", "gl_FrontColor",          EbvFrontColor,          symbolTable);
        BuiltInVariable("gl_in", "gl_BackColor",           EbvBackColor,           symbolTable);
        BuiltInVariable("gl_in", "gl_FrontSecondaryColor", EbvFrontSecondaryColor, symbolTable);
        BuiltInVariable("gl_in", "gl_BackSecondaryColor",  EbvBackSecondaryColor,  symbolTable);
        BuiltInVariable("gl_in", "gl_TexCoord",            EbvTexCoord,            symbolTable);
        BuiltInVariable("gl_in", "gl_FogFragCoord",        EbvFogFragCoord,        symbolTable);

        BuiltInVariable("gl_out", "gl_ClipVertex",          EbvClipVertex,          symbolTable);
        BuiltInVariable("gl_out", "gl_FrontColor",          EbvFrontColor,          symbolTable);
        BuiltInVariable("gl_out", "gl_BackColor",           EbvBackColor,           symbolTable);
        BuiltInVariable("gl_out", "gl_FrontSecondaryColor", EbvFrontSecondaryColor, symbolTable);
        BuiltInVariable("gl_out", "gl_BackSecondaryColor",  EbvBackSecondaryColor,  symbolTable);
        BuiltInVariable("gl_out", "gl_TexCoord",            EbvTexCoord,            symbolTable);
        BuiltInVariable("gl_out", "gl_FogFragCoord",        EbvFogFragCoord,        symbolTable);

        BuiltInVariable("gl_ClipVertex",          EbvClipVertex,          symbolTable);
        BuiltInVariable("gl_FrontColor",          EbvFrontColor,          symbolTable);
        BuiltInVariable("gl_BackColor",           EbvBackColor,           symbolTable);
        BuiltInVariable("gl_FrontSecondaryColor", EbvFrontSecondaryColor, symbolTable);
        BuiltInVariable("gl_BackSecondaryColor",  EbvBackSecondaryColor,  symbolTable);
        BuiltInVariable("gl_TexCoord",            EbvTexCoord,            symbolTable);
        BuiltInVariable("gl_FogFragCoord",        EbvFogFragCoord,        symbolTable);

        // gl_PointSize, when it needs to be tied to an extension, is always a member of a block.
        // (Sometimes with an instance name, sometimes anonymous).
        if (profile == EEsProfile) {
            if (language == EShLangGeometry) {
                symbolTable.setVariableExtensions("gl_PointSize", Num_AEP_geometry_point_size, AEP_geometry_point_size);
                symbolTable.setVariableExtensions("gl_in", "gl_PointSize", Num_AEP_geometry_point_size, AEP_geometry_point_size);
            } else if (language == EShLangTessEvaluation || language == EShLangTessControl) {
                // gl_in tessellation settings of gl_PointSize are in the context-dependent paths
                symbolTable.setVariableExtensions("gl_PointSize", Num_AEP_tessellation_point_size, AEP_tessellation_point_size);
                symbolTable.setVariableExtensions("gl_out", "gl_PointSize", Num_AEP_tessellation_point_size, AEP_tessellation_point_size);
            }
        }

        if ((profile != EEsProfile && version >= 140) ||
            (profile == EEsProfile && version >= 310)) {
            symbolTable.setVariableExtensions("gl_DeviceIndex",  1, &E_GL_EXT_device_group);
            BuiltInVariable("gl_DeviceIndex", EbvDeviceIndex, symbolTable);
            symbolTable.setVariableExtensions("gl_ViewIndex", 1, &E_GL_EXT_multiview);
            BuiltInVariable("gl_ViewIndex", EbvViewIndex, symbolTable);
        }

        if (profile != EEsProfile) {
            BuiltInVariable("gl_SubGroupInvocationARB", EbvSubGroupInvocation, symbolTable);
            BuiltInVariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0) {
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
                if (language == EShLangFragment)
                    ModifyFlatDecoration("gl_SubGroupSizeARB", true, symbolTable);
            }
            else
                BuiltInVariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupInvocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltInVariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltInVariable("gl_SubgroupInvocationID", EbvSubgroupInvocation2, symbolTable);
            BuiltInVariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            // GL_NV_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMNV",         1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountNV",            1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDNV",             1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDNV",               1, &E_GL_NV_shader_sm_builtins);
            BuiltInVariable("gl_WarpsPerSMNV",          EbvWarpsPerSM,      symbolTable);
            BuiltInVariable("gl_SMCountNV",             EbvSMCount,         symbolTable);
            BuiltInVariable("gl_WarpIDNV",              EbvWarpID,          symbolTable);
            BuiltInVariable("gl_SMIDNV",                EbvSMID,            symbolTable);

            // GL_ARM_shader_core_builtins
            symbolTable.setVariableExtensions("gl_CoreCountARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);

            BuiltInVariable("gl_CoreCountARM", EbvCoreCountARM, symbolTable);
            BuiltInVariable("gl_CoreIDARM",    EbvCoreIDARM, symbolTable);
            BuiltInVariable("gl_CoreMaxIDARM", EbvCoreMaxIDARM, symbolTable);
            BuiltInVariable("gl_WarpIDARM",    EbvWarpIDARM, symbolTable);
            BuiltInVariable("gl_WarpMaxIDARM", EbvWarpMaxIDARM, symbolTable);
        }

		if (language == EShLangGeometry || language == EShLangVertex) {
			if ((profile == EEsProfile && version >= 310) ||
				(profile != EEsProfile && version >= 450)) {
				symbolTable.setVariableExtensions("gl_PrimitiveShadingRateEXT", 1, &E_GL_EXT_fragment_shading_rate);
				BuiltInVariable("gl_PrimitiveShadingRateEXT", EbvPrimitiveShadingRateKHR, symbolTable);

				symbolTable.setVariableExtensions("gl_ShadingRateFlag2VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
				symbolTable.setVariableExtensions("gl_ShadingRateFlag4VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
				symbolTable.setVariableExtensions("gl_ShadingRateFlag2HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
				symbolTable.setVariableExtensions("gl_ShadingRateFlag4HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
			}
		}
        break;

    case EShLangFragment:
        SpecialQualifier("gl_FrontFacing",      EvqFace,       EbvFace,             symbolTable);
        SpecialQualifier("gl_FragCoord",        EvqFragCoord,  EbvFragCoord,        symbolTable);
        SpecialQualifier("gl_PointCoord",       EvqPointCoord, EbvPointCoord,       symbolTable);
        if (spvVersion.spv == 0)
            SpecialQualifier("gl_FragColor",    EvqFragColor,  EbvFragColor,        symbolTable);
        else {
            TSymbol* symbol = symbolTable.find("gl_FragColor");
            if (symbol) {
                symbol->getWritableType().getQualifier().storage = EvqVaryingOut;
                symbol->getWritableType().getQualifier().layoutLocation = 0;
            }
        }
        SpecialQualifier("gl_FragDepth",        EvqFragDepth,  EbvFragDepth,        symbolTable);
        SpecialQualifier("gl_FragDepthEXT",     EvqFragDepth,  EbvFragDepth,        symbolTable);
        SpecialQualifier("gl_FragStencilRefARB", EvqFragStencil, EbvFragStencilRef, symbolTable);
        SpecialQualifier("gl_HelperInvocation", EvqVaryingIn,  EbvHelperInvocation, symbolTable);

        BuiltInVariable("gl_ClipDistance",    EbvClipDistance,   symbolTable);
        BuiltInVariable("gl_CullDistance",    EbvCullDistance,   symbolTable);
        BuiltInVariable("gl_PrimitiveID",     EbvPrimitiveId,    symbolTable);

        if (profile != EEsProfile && version >= 140) {
            symbolTable.setVariableExtensions("gl_FragStencilRefARB", 1, &E_GL_ARB_shader_stencil_export);
            BuiltInVariable("gl_FragStencilRefARB", EbvFragStencilRef, symbolTable);
        }

        if (profile != EEsProfile && version < 400) {
            symbolTable.setFunctionExtensions("textureQueryLOD", 1, &E_GL_ARB_texture_query_lod);
        }

        if (profile != EEsProfile && version >= 460) {
            symbolTable.setFunctionExtensions("rayQueryInitializeEXT",                                            1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryTerminateEXT",                                             1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGenerateIntersectionEXT",                                  1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryConfirmIntersectionEXT",                                   1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryProceedEXT",                                               1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionTypeEXT",                                   1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionTEXT",                                      1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetRayFlagsEXT",                                           1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetRayTMinEXT",                                            1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionInstanceCustomIndexEXT",                    1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionInstanceIdEXT",                             1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT", 1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionGeometryIndexEXT",                          1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionPrimitiveIndexEXT",                         1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionBarycentricsEXT",                           1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionFrontFaceEXT",                              1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionCandidateAABBOpaqueEXT",                    1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionObjectRayDirectionEXT",                     1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionObjectRayOriginEXT",                        1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionObjectToWorldEXT",                          1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionWorldToObjectEXT",                          1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetWorldRayOriginEXT",                                     1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetWorldRayDirectionEXT",                                  1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionTriangleVertexPositionsEXT",                1, &E_GL_EXT_ray_tracing_position_fetch);
            symbolTable.setVariableExtensions("gl_RayFlagsSkipAABBEXT",                         1, &E_GL_EXT_ray_flags_primitive_culling);
            symbolTable.setVariableExtensions("gl_RayFlagsSkipTrianglesEXT",                    1, &E_GL_EXT_ray_flags_primitive_culling);
            symbolTable.setVariableExtensions("gl_RayFlagsForceOpacityMicromap2StateEXT",                  1, &E_GL_EXT_opacity_micromap);
        }

        if ((profile != EEsProfile && version >= 130) ||
            (profile == EEsProfile && version >= 310)) {
            BuiltInVariable("gl_SampleID",           EbvSampleId,       symbolTable);
            BuiltInVariable("gl_SamplePosition",     EbvSamplePosition, symbolTable);
            BuiltInVariable("gl_SampleMask",         EbvSampleMask,     symbolTable);

            if (profile != EEsProfile && version < 400) {
                BuiltInVariable("gl_NumSamples",     EbvSampleMask,     symbolTable);

                symbolTable.setVariableExtensions("gl_SampleMask",     1, &E_GL_ARB_sample_shading);
                symbolTable.setVariableExtensions("gl_SampleID",       1, &E_GL_ARB_sample_shading);
                symbolTable.setVariableExtensions("gl_SamplePosition", 1, &E_GL_ARB_sample_shading);
                symbolTable.setVariableExtensions("gl_NumSamples",     1, &E_GL_ARB_sample_shading);
            } else {
                BuiltInVariable("gl_SampleMaskIn",    EbvSampleMask,     symbolTable);

                if (profile == EEsProfile && version < 320) {
                    symbolTable.setVariableExtensions("gl_SampleID", 1, &E_GL_OES_sample_variables);
                    symbolTable.setVariableExtensions("gl_SamplePosition", 1, &E_GL_OES_sample_variables);
                    symbolTable.setVariableExtensions("gl_SampleMaskIn", 1, &E_GL_OES_sample_variables);
                    symbolTable.setVariableExtensions("gl_SampleMask", 1, &E_GL_OES_sample_variables);
                    symbolTable.setVariableExtensions("gl_NumSamples", 1, &E_GL_OES_sample_variables);
                }
            }
        }

        BuiltInVariable("gl_Layer",           EbvLayer,          symbolTable);
        BuiltInVariable("gl_ViewportIndex",   EbvViewportIndex,  symbolTable);

        // Compatibility variables

        BuiltInVariable("gl_in", "gl_FogFragCoord",   EbvFogFragCoord,   symbolTable);
        BuiltInVariable("gl_in", "gl_TexCoord",       EbvTexCoord,       symbolTable);
        BuiltInVariable("gl_in", "gl_Color",          EbvColor,          symbolTable);
        BuiltInVariable("gl_in", "gl_SecondaryColor", EbvSecondaryColor, symbolTable);

        BuiltInVariable("gl_FogFragCoord",   EbvFogFragCoord,   symbolTable);
        BuiltInVariable("gl_TexCoord",       EbvTexCoord,       symbolTable);
        BuiltInVariable("gl_Color",          EbvColor,          symbolTable);
        BuiltInVariable("gl_SecondaryColor", EbvSecondaryColor, symbolTable);

        // built-in functions

        if (profile == EEsProfile) {
            if (spvVersion.spv == 0) {
                symbolTable.setFunctionExtensions("texture2DLodEXT",      1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DProjLodEXT",  1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("textureCubeLodEXT",    1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DGradEXT",     1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DProjGradEXT", 1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("textureCubeGradEXT",   1, &E_GL_EXT_shader_texture_lod);
                if (version < 320)
                    symbolTable.setFunctionExtensions("textureGatherOffsets", Num_AEP_gpu_shader5, AEP_gpu_shader5);
            }
            if (version == 100) {
                symbolTable.setFunctionExtensions("dFdx",   1, &E_GL_OES_standard_derivatives);
                symbolTable.setFunctionExtensions("dFdy",   1, &E_GL_OES_standard_derivatives);
                symbolTable.setFunctionExtensions("fwidth", 1, &E_GL_OES_standard_derivatives);
            }
            if (version == 310) {
                symbolTable.setFunctionExtensions("fma", Num_AEP_gpu_shader5, AEP_gpu_shader5);
                symbolTable.setFunctionExtensions("interpolateAtCentroid", 1, &E_GL_OES_shader_multisample_interpolation);
                symbolTable.setFunctionExtensions("interpolateAtSample",   1, &E_GL_OES_shader_multisample_interpolation);
                symbolTable.setFunctionExtensions("interpolateAtOffset",   1, &E_GL_OES_shader_multisample_interpolation);
            }
        } else if (version < 130) {
            if (spvVersion.spv == 0) {
                symbolTable.setFunctionExtensions("texture1DLod",        1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DLod",        1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture3DLod",        1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("textureCubeLod",      1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture1DProjLod",    1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DProjLod",    1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture3DProjLod",    1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("shadow1DLod",         1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("shadow2DLod",         1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("shadow1DProjLod",     1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("shadow2DProjLod",     1, &E_GL_ARB_shader_texture_lod);
            }
        }

        // E_GL_ARB_shader_texture_lod functions usable only with the extension enabled
        if (profile != EEsProfile && spvVersion.spv == 0) {
            symbolTable.setFunctionExtensions("texture1DGradARB",         1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture1DProjGradARB",     1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture2DGradARB",         1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture2DProjGradARB",     1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture3DGradARB",         1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture3DProjGradARB",     1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("textureCubeGradARB",       1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow1DGradARB",          1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow1DProjGradARB",      1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow2DGradARB",          1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow2DProjGradARB",      1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture2DRectGradARB",     1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture2DRectProjGradARB", 1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow2DRectGradARB",      1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow2DRectProjGradARB",  1, &E_GL_ARB_shader_texture_lod);
        }

        // E_GL_ARB_shader_image_load_store
        if (profile != EEsProfile && version < 420)
            symbolTable.setFunctionExtensions("memoryBarrier", 1, &E_GL_ARB_shader_image_load_store);
        // All the image access functions are protected by checks on the type of the first argument.

        // E_GL_ARB_shader_atomic_counters
        if (profile != EEsProfile && version < 420) {
            symbolTable.setFunctionExtensions("atomicCounterIncrement", 1, &E_GL_ARB_shader_atomic_counters);
            symbolTable.setFunctionExtensions("atomicCounterDecrement", 1, &E_GL_ARB_shader_atomic_counters);
            symbolTable.setFunctionExtensions("atomicCounter"         , 1, &E_GL_ARB_shader_atomic_counters);
        }

        // E_GL_ARB_shader_atomic_counter_ops
        if (profile != EEsProfile && version == 450) {
            symbolTable.setFunctionExtensions("atomicCounterAddARB"     , 1, &E_GL_ARB_shader_atomic_counter_ops);
            symbolTable.setFunctionExtensions("atomicCounterSubtractARB", 1, &E_GL_ARB_shader_atomic_counter_ops);
            symbolTable.setFunctionExtensions("atomicCounterMinARB"     , 1, &E_GL_ARB_shader_atomic_counter_ops);
            symbolTable.setFunctionExtensions("atomicCounterMaxARB"     , 1, &E_GL_ARB_shader_atomic_counter_ops);
            symbolTable.setFunctionExtensions("atomicCounterAndARB"     , 1, &E_GL_ARB_shader_atomic_counter_ops);
            symbolTable.setFunctionExtensions("atomicCounterOrARB"      , 1, &E_GL_ARB_shader_atomic_counter_ops);
            symbolTable.setFunctionExtensions("atomicCounterXorARB"     , 1, &E_GL_ARB_shader_atomic_counter_ops);
            symbolTable.setFunctionExtensions("atomicCounterExchangeARB", 1, &E_GL_ARB_shader_atomic_counter_ops);
            symbolTable.setFunctionExtensions("atomicCounterCompSwapARB", 1, &E_GL_ARB_shader_atomic_counter_ops);
        }

        // E_GL_ARB_derivative_control
        if (profile != EEsProfile && version < 450) {
            symbolTable.setFunctionExtensions("dFdxFine",     1, &E_GL_ARB_derivative_control);
            symbolTable.setFunctionExtensions("dFdyFine",     1, &E_GL_ARB_derivative_control);
            symbolTable.setFunctionExtensions("fwidthFine",   1, &E_GL_ARB_derivative_control);
            symbolTable.setFunctionExtensions("dFdxCoarse",   1, &E_GL_ARB_derivative_control);
            symbolTable.setFunctionExtensions("dFdyCoarse",   1, &E_GL_ARB_derivative_control);
            symbolTable.setFunctionExtensions("fwidthCoarse", 1, &E_GL_ARB_derivative_control);
        }

        // E_GL_ARB_sparse_texture2
        if (profile != EEsProfile)
        {
            symbolTable.setFunctionExtensions("sparseTextureARB",              1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureLodARB",           1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureOffsetARB",        1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTexelFetchARB",           1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTexelFetchOffsetARB",     1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureLodOffsetARB",     1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureGradARB",          1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureGradOffsetARB",    1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureGatherARB",        1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureGatherOffsetARB",  1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureGatherOffsetsARB", 1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseImageLoadARB",            1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTexelsResident",          1, &E_GL_ARB_sparse_texture2);
        }

        // E_GL_ARB_sparse_texture_clamp
        if (profile != EEsProfile)
        {
            symbolTable.setFunctionExtensions("sparseTextureClampARB",              1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("sparseTextureOffsetClampARB",        1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("sparseTextureGradClampARB",          1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("sparseTextureGradOffsetClampARB",    1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("textureClampARB",                    1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("textureOffsetClampARB",              1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("textureGradClampARB",                1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("textureGradOffsetClampARB",          1, &E_GL_ARB_sparse_texture_clamp);
        }

        // E_GL_AMD_shader_explicit_vertex_parameter
        if (profile != EEsProfile) {
            symbolTable.setVariableExtensions("gl_BaryCoordNoPerspAMD",         1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordNoPerspCentroidAMD", 1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordNoPerspSampleAMD",   1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordSmoothAMD",          1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordSmoothCentroidAMD",  1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordSmoothSampleAMD",    1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordPullModelAMD",       1, &E_GL_AMD_shader_explicit_vertex_parameter);

            symbolTable.setFunctionExtensions("interpolateAtVertexAMD",         1, &E_GL_AMD_shader_explicit_vertex_parameter);

            BuiltInVariable("gl_BaryCoordNoPerspAMD",           EbvBaryCoordNoPersp,         symbolTable);
            BuiltInVariable("gl_BaryCoordNoPerspCentroidAMD",   EbvBaryCoordNoPerspCentroid, symbolTable);
            BuiltInVariable("gl_BaryCoordNoPerspSampleAMD",     EbvBaryCoordNoPerspSample,   symbolTable);
            BuiltInVariable("gl_BaryCoordSmoothAMD",            EbvBaryCoordSmooth,          symbolTable);
            BuiltInVariable("gl_BaryCoordSmoothCentroidAMD",    EbvBaryCoordSmoothCentroid,  symbolTable);
            BuiltInVariable("gl_BaryCoordSmoothSampleAMD",      EbvBaryCoordSmoothSample,    symbolTable);
            BuiltInVariable("gl_BaryCoordPullModelAMD",         EbvBaryCoordPullModel,       symbolTable);
        }

        // E_GL_AMD_texture_gather_bias_lod
        if (profile != EEsProfile) {
            symbolTable.setFunctionExtensions("textureGatherLodAMD",                1, &E_GL_AMD_texture_gather_bias_lod);
            symbolTable.setFunctionExtensions("textureGatherLodOffsetAMD",          1, &E_GL_AMD_texture_gather_bias_lod);
            symbolTable.setFunctionExtensions("textureGatherLodOffsetsAMD",         1, &E_GL_AMD_texture_gather_bias_lod);
            symbolTable.setFunctionExtensions("sparseTextureGatherLodAMD",          1, &E_GL_AMD_texture_gather_bias_lod);
            symbolTable.setFunctionExtensions("sparseTextureGatherLodOffsetAMD",    1, &E_GL_AMD_texture_gather_bias_lod);
            symbolTable.setFunctionExtensions("sparseTextureGatherLodOffsetsAMD",   1, &E_GL_AMD_texture_gather_bias_lod);
        }

        // E_GL_AMD_shader_image_load_store_lod
        if (profile != EEsProfile) {
            symbolTable.setFunctionExtensions("imageLoadLodAMD",        1, &E_GL_AMD_shader_image_load_store_lod);
            symbolTable.setFunctionExtensions("imageStoreLodAMD",       1, &E_GL_AMD_shader_image_load_store_lod);
            symbolTable.setFunctionExtensions("sparseImageLoadLodAMD",  1, &E_GL_AMD_shader_image_load_store_lod);
        }
        if (profile != EEsProfile && version >= 430) {
            symbolTable.setVariableExtensions("gl_FragFullyCoveredNV", 1, &E_GL_NV_conservative_raster_underestimation);
            BuiltInVariable("gl_FragFullyCoveredNV", EbvFragFullyCoveredNV, symbolTable);
        }
        if ((profile != EEsProfile && version >= 450) ||
            (profile == EEsProfile && version >= 320)) {
            symbolTable.setVariableExtensions("gl_FragmentSizeNV",        1, &E_GL_NV_shading_rate_image);
            symbolTable.setVariableExtensions("gl_InvocationsPerPixelNV", 1, &E_GL_NV_shading_rate_image);
            BuiltInVariable("gl_FragmentSizeNV",        EbvFragmentSizeNV, symbolTable);
            BuiltInVariable("gl_InvocationsPerPixelNV", EbvInvocationsPerPixelNV, symbolTable);
            symbolTable.setVariableExtensions("gl_BaryCoordNV",        1, &E_GL_NV_fragment_shader_barycentric);
            symbolTable.setVariableExtensions("gl_BaryCoordNoPerspNV", 1, &E_GL_NV_fragment_shader_barycentric);
            BuiltInVariable("gl_BaryCoordNV",        EbvBaryCoordNV,        symbolTable);
            BuiltInVariable("gl_BaryCoordNoPerspNV", EbvBaryCoordNoPerspNV, symbolTable);
            symbolTable.setVariableExtensions("gl_BaryCoordEXT",        1, &E_GL_EXT_fragment_shader_barycentric);
            symbolTable.setVariableExtensions("gl_BaryCoordNoPerspEXT", 1, &E_GL_EXT_fragment_shader_barycentric);
            BuiltInVariable("gl_BaryCoordEXT",        EbvBaryCoordEXT,        symbolTable);
            BuiltInVariable("gl_BaryCoordNoPerspEXT", EbvBaryCoordNoPerspEXT, symbolTable);
        }

        if ((profile != EEsProfile && version >= 450) ||
            (profile == EEsProfile && version >= 310)) {
            symbolTable.setVariableExtensions("gl_FragSizeEXT",            1, &E_GL_EXT_fragment_invocation_density);
            symbolTable.setVariableExtensions("gl_FragInvocationCountEXT", 1, &E_GL_EXT_fragment_invocation_density);
            BuiltInVariable("gl_FragSizeEXT",            EbvFragSizeEXT, symbolTable);
            BuiltInVariable("gl_FragInvocationCountEXT", EbvFragInvocationCountEXT, symbolTable);
        }

        symbolTable.setVariableExtensions("gl_FragDepthEXT", 1, &E_GL_EXT_frag_depth);

        symbolTable.setFunctionExtensions("clockARB",     1, &E_GL_ARB_shader_clock);
        symbolTable.setFunctionExtensions("clock2x32ARB", 1, &E_GL_ARB_shader_clock);

        symbolTable.setFunctionExtensions("clockRealtimeEXT", 1, &E_GL_EXT_shader_realtime_clock);
        symbolTable.setFunctionExtensions("clockRealtime2x32EXT", 1, &E_GL_EXT_shader_realtime_clock);

        if (profile == EEsProfile && version < 320) {
            symbolTable.setVariableExtensions("gl_PrimitiveID",  Num_AEP_geometry_shader, AEP_geometry_shader);
            symbolTable.setVariableExtensions("gl_Layer",        Num_AEP_geometry_shader, AEP_geometry_shader);
        }

        if (profile == EEsProfile && version < 320) {
            symbolTable.setFunctionExtensions("imageAtomicAdd",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicMin",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicMax",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicAnd",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicOr",       1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicXor",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicExchange", 1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicCompSwap", 1, &E_GL_OES_shader_image_atomic);
        }

        if (profile != EEsProfile && version < 330 ) {
            const char* bitsConvertExt[2] = {E_GL_ARB_shader_bit_encoding, E_GL_ARB_gpu_shader5};
            symbolTable.setFunctionExtensions("floatBitsToInt", 2, bitsConvertExt);
            symbolTable.setFunctionExtensions("floatBitsToUint", 2, bitsConvertExt);
            symbolTable.setFunctionExtensions("intBitsToFloat", 2, bitsConvertExt);
            symbolTable.setFunctionExtensions("uintBitsToFloat", 2, bitsConvertExt);
        }

        if (profile != EEsProfile && version < 430 ) {
            symbolTable.setFunctionExtensions("imageSize", 1, &E_GL_ARB_shader_image_size);
        }

        // GL_ARB_shader_storage_buffer_object
        if (profile != EEsProfile && version < 430 ) {
            symbolTable.setFunctionExtensions("atomicAdd", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicMin", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicMax", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicAnd", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicOr", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicXor", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicExchange", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicCompSwap", 1, &E_GL_ARB_shader_storage_buffer_object);
        }

        // GL_ARB_shading_language_packing
        if (profile != EEsProfile && version < 400 ) {
            symbolTable.setFunctionExtensions("packUnorm2x16", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("unpackUnorm2x16", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("packSnorm4x8", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("packUnorm4x8", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("unpackSnorm4x8", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("unpackUnorm4x8", 1, &E_GL_ARB_shading_language_packing);
        }
        if (profile != EEsProfile && version < 420 ) {
            symbolTable.setFunctionExtensions("packSnorm2x16", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("unpackSnorm2x16", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("unpackHalf2x16", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("packHalf2x16", 1, &E_GL_ARB_shading_language_packing);
        }

        symbolTable.setVariableExtensions("gl_DeviceIndex",  1, &E_GL_EXT_device_group);
        BuiltInVariable("gl_DeviceIndex", EbvDeviceIndex, symbolTable);
        symbolTable.setVariableExtensions("gl_ViewIndex", 1, &E_GL_EXT_multiview);
        BuiltInVariable("gl_ViewIndex", EbvViewIndex, symbolTable);
        if (version >= 300 /* both ES and non-ES */) {
            symbolTable.setVariableExtensions("gl_ViewID_OVR", Num_OVR_multiview_EXTs, OVR_multiview_EXTs);
            BuiltInVariable("gl_ViewID_OVR", EbvViewIndex, symbolTable);
        }

        // GL_ARB_shader_ballot
        if (profile != EEsProfile) {
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupInvocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            BuiltInVariable("gl_SubGroupInvocationARB", EbvSubGroupInvocation, symbolTable);
            BuiltInVariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0) {
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
                if (language == EShLangFragment)
                    ModifyFlatDecoration("gl_SubGroupSizeARB", true, symbolTable);
            }
            else
                BuiltInVariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);
        }

        // GL_EXT_expect_assume
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setFunctionExtensions("assumeEXT", 1, &E_GL_EXT_expect_assume);
            symbolTable.setFunctionExtensions("expectEXT", 1, &E_GL_EXT_expect_assume);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupInvocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltInVariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltInVariable("gl_SubgroupInvocationID", EbvSubgroupInvocation2, symbolTable);
            BuiltInVariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            symbolTable.setFunctionExtensions("subgroupBarrier",                 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setFunctionExtensions("subgroupMemoryBarrier",           1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setFunctionExtensions("subgroupMemoryBarrierBuffer",     1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setFunctionExtensions("subgroupMemoryBarrierImage",      1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setFunctionExtensions("subgroupElect",                   1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setFunctionExtensions("subgroupAll",                     1, &E_GL_KHR_shader_subgroup_vote);
            symbolTable.setFunctionExtensions("subgroupAny",                     1, &E_GL_KHR_shader_subgroup_vote);
            symbolTable.setFunctionExtensions("subgroupAllEqual",                1, &E_GL_KHR_shader_subgroup_vote);
            symbolTable.setFunctionExtensions("subgroupBroadcast",               1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBroadcastFirst",          1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallot",                  1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupInverseBallot",           1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotBitExtract",        1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotBitCount",          1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotInclusiveBitCount", 1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotExclusiveBitCount", 1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotFindLSB",           1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotFindMSB",           1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupShuffle",                 1, &E_GL_KHR_shader_subgroup_shuffle);
            symbolTable.setFunctionExtensions("subgroupShuffleXor",              1, &E_GL_KHR_shader_subgroup_shuffle);
            symbolTable.setFunctionExtensions("subgroupShuffleUp",               1, &E_GL_KHR_shader_subgroup_shuffle_relative);
            symbolTable.setFunctionExtensions("subgroupShuffleDown",             1, &E_GL_KHR_shader_subgroup_shuffle_relative);
            symbolTable.setFunctionExtensions("subgroupRotate",                  1, &E_GL_KHR_shader_subgroup_rotate);
            symbolTable.setFunctionExtensions("subgroupClusteredRotate",         1, &E_GL_KHR_shader_subgroup_rotate);
            symbolTable.setFunctionExtensions("subgroupAdd",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupMul",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupMin",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupMax",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupAnd",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupOr",                      1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupXor",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveAdd",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveMul",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveMin",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveMax",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveAnd",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveOr",             1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveXor",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveAdd",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveMul",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveMin",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveMax",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveAnd",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveOr",             1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveXor",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupClusteredAdd",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredMul",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredMin",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredMax",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredAnd",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredOr",             1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredXor",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupQuadBroadcast",           1, &E_GL_KHR_shader_subgroup_quad);
            symbolTable.setFunctionExtensions("subgroupQuadSwapHorizontal",      1, &E_GL_KHR_shader_subgroup_quad);
            symbolTable.setFunctionExtensions("subgroupQuadSwapVertical",        1, &E_GL_KHR_shader_subgroup_quad);
            symbolTable.setFunctionExtensions("subgroupQuadSwapDiagonal",        1, &E_GL_KHR_shader_subgroup_quad);
            symbolTable.setFunctionExtensions("subgroupPartitionNV",                          1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedAddNV",                     1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedMulNV",                     1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedMinNV",                     1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedMaxNV",                     1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedAndNV",                     1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedOrNV",                      1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedXorNV",                     1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveAddNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveMulNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveMinNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveMaxNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveAndNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveOrNV",             1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveXorNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveAddNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveMulNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveMinNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveMaxNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveAndNV",            1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveOrNV",             1, &E_GL_NV_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveXorNV",            1, &E_GL_NV_shader_subgroup_partitioned);

            // GL_NV_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMNV",         1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountNV",            1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDNV",             1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDNV",               1, &E_GL_NV_shader_sm_builtins);
            BuiltInVariable("gl_WarpsPerSMNV",          EbvWarpsPerSM,      symbolTable);
            BuiltInVariable("gl_SMCountNV",             EbvSMCount,         symbolTable);
            BuiltInVariable("gl_WarpIDNV",              EbvWarpID,          symbolTable);
            BuiltInVariable("gl_SMIDNV",                EbvSMID,            symbolTable);

            // GL_ARM_shader_core_builtins
            symbolTable.setVariableExtensions("gl_CoreCountARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);

            BuiltInVariable("gl_CoreCountARM", EbvCoreCountARM, symbolTable);
            BuiltInVariable("gl_CoreIDARM",    EbvCoreIDARM, symbolTable);
            BuiltInVariable("gl_CoreMaxIDARM", EbvCoreMaxIDARM, symbolTable);
            BuiltInVariable("gl_WarpIDARM",    EbvWarpIDARM, symbolTable);
            BuiltInVariable("gl_WarpMaxIDARM", EbvWarpMaxIDARM, symbolTable);
        }

        if (profile == EEsProfile) {
            symbolTable.setFunctionExtensions("shadow2DEXT",        1, &E_GL_EXT_shadow_samplers);
            symbolTable.setFunctionExtensions("shadow2DProjEXT",    1, &E_GL_EXT_shadow_samplers);
        }

        if (spvVersion.vulkan > 0) {
            symbolTable.setVariableExtensions("gl_ScopeDevice",             1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_ScopeWorkgroup",          1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_ScopeSubgroup",           1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_ScopeInvocation",         1, &E_GL_KHR_memory_scope_semantics);

            symbolTable.setVariableExtensions("gl_SemanticsRelaxed",        1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsAcquire",        1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsRelease",        1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsAcquireRelease", 1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsMakeAvailable",  1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsMakeVisible",    1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsVolatile",       1, &E_GL_KHR_memory_scope_semantics);

            symbolTable.setVariableExtensions("gl_StorageSemanticsNone",    1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_StorageSemanticsBuffer",  1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_StorageSemanticsShared",  1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_StorageSemanticsImage",   1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_StorageSemanticsOutput",  1, &E_GL_KHR_memory_scope_semantics);
        }

        symbolTable.setFunctionExtensions("helperInvocationEXT",            1, &E_GL_EXT_demote_to_helper_invocation);

        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 450)) {
            symbolTable.setVariableExtensions("gl_ShadingRateEXT", 1, &E_GL_EXT_fragment_shading_rate);
            BuiltInVariable("gl_ShadingRateEXT", EbvShadingRateKHR, symbolTable);

            symbolTable.setVariableExtensions("gl_ShadingRateFlag2VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag4VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag2HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag4HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
        }

        // GL_EXT_shader_quad_control
        if ((profile != EEsProfile && version >= 140) ||
            (profile == EEsProfile && version >= 310)) {
            symbolTable.setFunctionExtensions("subgroupQuadAll",                     1,  &E_GL_KHR_shader_subgroup_vote);
            symbolTable.setFunctionExtensions("subgroupQuadAny",                     1,  &E_GL_KHR_shader_subgroup_vote);
        }

        // GL_EXT_shader_tile_image
        symbolTable.setFunctionExtensions("stencilAttachmentReadEXT", 1, &E_GL_EXT_shader_tile_image);
        symbolTable.setFunctionExtensions("depthAttachmentReadEXT", 1, &E_GL_EXT_shader_tile_image);
        symbolTable.setFunctionExtensions("colorAttachmentReadEXT", 1, &E_GL_EXT_shader_tile_image);

        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {

            symbolTable.setFunctionExtensions("textureWeightedQCOM",      1, &E_GL_QCOM_image_processing);
            symbolTable.setFunctionExtensions("textureBoxFilterQCOM",     1, &E_GL_QCOM_image_processing);
            symbolTable.setFunctionExtensions("textureBlockMatchSADQCOM", 1, &E_GL_QCOM_image_processing);
            symbolTable.setFunctionExtensions("textureBlockMatchSSDQCOM", 1, &E_GL_QCOM_image_processing);

            symbolTable.setFunctionExtensions("textureBlockMatchWindowSSDQCOM", 1, &E_GL_QCOM_image_processing2);
            symbolTable.setFunctionExtensions("textureBlockMatchWindowSADQCOM", 1, &E_GL_QCOM_image_processing2);
            symbolTable.setFunctionExtensions("textureBlockMatchGatherSSDQCOM", 1, &E_GL_QCOM_image_processing2);
            symbolTable.setFunctionExtensions("textureBlockMatchGatherSADQCOM", 1, &E_GL_QCOM_image_processing2);
        }
        break;

    case EShLangCompute:
        BuiltInVariable("gl_NumWorkGroups",         EbvNumWorkGroups,        symbolTable);
        BuiltInVariable("gl_WorkGroupSize",         EbvWorkGroupSize,        symbolTable);
        BuiltInVariable("gl_WorkGroupID",           EbvWorkGroupId,          symbolTable);
        BuiltInVariable("gl_LocalInvocationID",     EbvLocalInvocationId,    symbolTable);
        BuiltInVariable("gl_GlobalInvocationID",    EbvGlobalInvocationId,   symbolTable);
        BuiltInVariable("gl_LocalInvocationIndex",  EbvLocalInvocationIndex, symbolTable);
        BuiltInVariable("gl_DeviceIndex",           EbvDeviceIndex,          symbolTable);
        BuiltInVariable("gl_ViewIndex",             EbvViewIndex,            symbolTable);

        if ((profile != EEsProfile && version >= 140) ||
            (profile == EEsProfile && version >= 310)) {
            symbolTable.setVariableExtensions("gl_DeviceIndex",  1, &E_GL_EXT_device_group);
            symbolTable.setVariableExtensions("gl_ViewIndex",    1, &E_GL_EXT_multiview);
        }

        if (profile != EEsProfile && version < 430) {
            symbolTable.setVariableExtensions("gl_NumWorkGroups",        1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_WorkGroupSize",        1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_WorkGroupID",          1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_LocalInvocationID",    1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_GlobalInvocationID",   1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_LocalInvocationIndex", 1, &E_GL_ARB_compute_shader);

            symbolTable.setVariableExtensions("gl_MaxComputeWorkGroupCount",       1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeWorkGroupSize",        1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeUniformComponents",    1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeTextureImageUnits",    1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeImageUniforms",        1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeAtomicCounters",       1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeAtomicCounterBuffers", 1, &E_GL_ARB_compute_shader);

            symbolTable.setFunctionExtensions("barrier",                    1, &E_GL_ARB_compute_shader);
            symbolTable.setFunctionExtensions("memoryBarrierAtomicCounter", 1, &E_GL_ARB_compute_shader);
            symbolTable.setFunctionExtensions("memoryBarrierBuffer",        1, &E_GL_ARB_compute_shader);
            symbolTable.setFunctionExtensions("memoryBarrierImage",         1, &E_GL_ARB_compute_shader);
            symbolTable.setFunctionExtensions("memoryBarrierShared",        1, &E_GL_ARB_compute_shader);
            symbolTable.setFunctionExtensions("groupMemoryBarrier",         1, &E_GL_ARB_compute_shader);
        }


        symbolTable.setFunctionExtensions("controlBarrier",                 1, &E_GL_KHR_memory_scope_semantics);
        symbolTable.setFunctionExtensions("debugPrintfEXT",                 1, &E_GL_EXT_debug_printf);

        // GL_ARB_shader_ballot
        if (profile != EEsProfile) {
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupInvocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            BuiltInVariable("gl_SubGroupInvocationARB", EbvSubGroupInvocation, symbolTable);
            BuiltInVariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0) {
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
                if (language == EShLangFragment)
                    ModifyFlatDecoration("gl_SubGroupSizeARB", true, symbolTable);
            }
            else
                BuiltInVariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupInvocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltInVariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltInVariable("gl_SubgroupInvocationID", EbvSubgroupInvocation2, symbolTable);
            BuiltInVariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            // GL_NV_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMNV",         1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountNV",            1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDNV",             1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDNV",               1, &E_GL_NV_shader_sm_builtins);
            BuiltInVariable("gl_WarpsPerSMNV",          EbvWarpsPerSM,      symbolTable);
            BuiltInVariable("gl_SMCountNV",             EbvSMCount,         symbolTable);
            BuiltInVariable("gl_WarpIDNV",              EbvWarpID,          symbolTable);
            BuiltInVariable("gl_SMIDNV",                EbvSMID,            symbolTable);

            // GL_ARM_shader_core_builtins
            symbolTable.setVariableExtensions("gl_CoreCountARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);

            BuiltInVariable("gl_CoreCountARM", EbvCoreCountARM, symbolTable);
            BuiltInVariable("gl_CoreIDARM",    EbvCoreIDARM, symbolTable);
            BuiltInVariable("gl_CoreMaxIDARM", EbvCoreMaxIDARM, symbolTable);
            BuiltInVariable("gl_WarpIDARM",    EbvWarpIDARM, symbolTable);
            BuiltInVariable("gl_WarpMaxIDARM", EbvWarpMaxIDARM, symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_NumSubgroups", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupID",   1, &E_GL_KHR_shader_subgroup_basic);

            BuiltInVariable("gl_NumSubgroups", EbvNumSubgroups, symbolTable);
            BuiltInVariable("gl_SubgroupID",   EbvSubgroupID,   symbolTable);

            symbolTable.setFunctionExtensions("subgroupMemoryBarrierShared", 1, &E_GL_KHR_shader_subgroup_basic);
        }

        {
            const char *coopExt[2] = { E_GL_NV_cooperative_matrix, E_GL_NV_integer_cooperative_matrix };
            symbolTable.setFunctionExtensions("coopMatLoadNV",   2, coopExt);
            symbolTable.setFunctionExtensions("coopMatStoreNV",  2, coopExt);
            symbolTable.setFunctionExtensions("coopMatMulAddNV", 2, coopExt);
        }

        {
            symbolTable.setFunctionExtensions("coopMatLoad",   1, &E_GL_KHR_cooperative_matrix);
            symbolTable.setFunctionExtensions("coopMatStore",  1, &E_GL_KHR_cooperative_matrix);
            symbolTable.setFunctionExtensions("coopMatMulAdd", 1, &E_GL_KHR_cooperative_matrix);
        }

        if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
            symbolTable.setFunctionExtensions("dFdx",                   1, &E_GL_NV_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("dFdy",                   1, &E_GL_NV_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("fwidth",                 1, &E_GL_NV_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("dFdxFine",               1, &E_GL_NV_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("dFdyFine",               1, &E_GL_NV_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("fwidthFine",             1, &E_GL_NV_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("dFdxCoarse",             1, &E_GL_NV_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("dFdyCoarse",             1, &E_GL_NV_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("fwidthCoarse",           1, &E_GL_NV_compute_shader_derivatives);
        }

        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 450)) {
            symbolTable.setVariableExtensions("gl_ShadingRateFlag2VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag4VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag2HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag4HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
        }

        if ((profile != EEsProfile && version >= 460)) {
            symbolTable.setFunctionExtensions("fetchMicroTriangleVertexPositionNV", 1, &E_GL_NV_displacement_micromap);
            symbolTable.setFunctionExtensions("fetchMicroTriangleVertexBarycentricNV", 1, &E_GL_NV_displacement_micromap);
        }
        break;

    case EShLangRayGen:
    case EShLangIntersect:
    case EShLangAnyHit:
    case EShLangClosestHit:
    case EShLangMiss:
    case EShLangCallable:
        if (profile != EEsProfile && version >= 460) {
            const char *rtexts[] = { E_GL_NV_ray_tracing, E_GL_EXT_ray_tracing };
            symbolTable.setVariableExtensions("gl_LaunchIDNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_LaunchIDEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_LaunchSizeNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_LaunchSizeEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_PrimitiveID", 2, rtexts);
            symbolTable.setVariableExtensions("gl_InstanceID", 2, rtexts);
            symbolTable.setVariableExtensions("gl_InstanceCustomIndexNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_InstanceCustomIndexEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_GeometryIndexEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldRayOriginNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldRayOriginEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldRayDirectionNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldRayDirectionEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectRayOriginNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectRayOriginEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectRayDirectionNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectRayDirectionEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_RayTminNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_RayTminEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_RayTmaxNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_RayTmaxEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_CullMaskEXT", 1, &E_GL_EXT_ray_cull_mask);
            symbolTable.setVariableExtensions("gl_HitTNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_HitTEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_HitKindNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_HitKindEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectToWorldNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectToWorldEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectToWorld3x4EXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldToObjectNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldToObjectEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldToObject3x4EXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_IncomingRayFlagsNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setVariableExtensions("gl_IncomingRayFlagsEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_CurrentRayTimeNV", 1, &E_GL_NV_ray_tracing_motion_blur);
            symbolTable.setVariableExtensions("gl_HitTriangleVertexPositionsEXT", 1, &E_GL_EXT_ray_tracing_position_fetch);
            symbolTable.setVariableExtensions("gl_HitMicroTriangleVertexPositionsNV", 1, &E_GL_NV_displacement_micromap);
            symbolTable.setVariableExtensions("gl_HitMicroTriangleVertexBarycentricsNV", 1, &E_GL_NV_displacement_micromap);

            symbolTable.setVariableExtensions("gl_DeviceIndex", 1, &E_GL_EXT_device_group);


            symbolTable.setFunctionExtensions("traceNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setFunctionExtensions("traceRayMotionNV", 1, &E_GL_NV_ray_tracing_motion_blur);
            symbolTable.setFunctionExtensions("traceRayEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setFunctionExtensions("reportIntersectionNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setFunctionExtensions("reportIntersectionEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setFunctionExtensions("ignoreIntersectionNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setFunctionExtensions("terminateRayNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setFunctionExtensions("executeCallableNV", 1, &E_GL_NV_ray_tracing);
            symbolTable.setFunctionExtensions("executeCallableEXT", 1, &E_GL_EXT_ray_tracing);

            symbolTable.setFunctionExtensions("hitObjectTraceRayNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectTraceRayMotionNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectRecordHitNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectRecordHitMotionNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectRecordHitWithIndexNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectRecordHitWithIndexMotionNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectRecordMissNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectRecordMissMotionNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectRecordEmptyNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectExecuteShaderNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectIsEmptyNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectIsMissNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectIsHitNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetRayTMinNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetRayTMaxNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetObjectRayOriginNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetObjectRayDirectionNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetWorldRayOriginNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetWorldRayDirectionNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetWorldToObjectNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetbjectToWorldNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetInstanceCustomIndexNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetInstanceIdNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetGeometryIndexNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetPrimitiveIndexNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetHitKindNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetAttributesNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetCurrentTimeNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetShaderBindingTableRecordIndexNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("hitObjectGetShaderRecordBufferHandleNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("reorderThreadNV", 1, &E_GL_NV_shader_invocation_reorder);
            symbolTable.setFunctionExtensions("fetchMicroTriangleVertexPositionNV", 1, &E_GL_NV_displacement_micromap);
            symbolTable.setFunctionExtensions("fetchMicroTriangleVertexBarycentricNV", 1, &E_GL_NV_displacement_micromap);


            BuiltInVariable("gl_LaunchIDNV",             EbvLaunchId,           symbolTable);
            BuiltInVariable("gl_LaunchIDEXT",            EbvLaunchId,           symbolTable);
            BuiltInVariable("gl_LaunchSizeNV",           EbvLaunchSize,         symbolTable);
            BuiltInVariable("gl_LaunchSizeEXT",          EbvLaunchSize,         symbolTable);
            BuiltInVariable("gl_PrimitiveID",            EbvPrimitiveId,        symbolTable);
            BuiltInVariable("gl_InstanceID",             EbvInstanceId,         symbolTable);
            BuiltInVariable("gl_InstanceCustomIndexNV",  EbvInstanceCustomIndex,symbolTable);
            BuiltInVariable("gl_InstanceCustomIndexEXT", EbvInstanceCustomIndex,symbolTable);
            BuiltInVariable("gl_GeometryIndexEXT",       EbvGeometryIndex,      symbolTable);
            BuiltInVariable("gl_WorldRayOriginNV",       EbvWorldRayOrigin,     symbolTable);
            BuiltInVariable("gl_WorldRayOriginEXT",      EbvWorldRayOrigin,     symbolTable);
            BuiltInVariable("gl_WorldRayDirectionNV",    EbvWorldRayDirection,  symbolTable);
            BuiltInVariable("gl_WorldRayDirectionEXT",   EbvWorldRayDirection,  symbolTable);
            BuiltInVariable("gl_ObjectRayOriginNV",      EbvObjectRayOrigin,    symbolTable);
            BuiltInVariable("gl_ObjectRayOriginEXT",     EbvObjectRayOrigin,    symbolTable);
            BuiltInVariable("gl_ObjectRayDirectionNV",   EbvObjectRayDirection, symbolTable);
            BuiltInVariable("gl_ObjectRayDirectionEXT",  EbvObjectRayDirection, symbolTable);
            BuiltInVariable("gl_RayTminNV",              EbvRayTmin,            symbolTable);
            BuiltInVariable("gl_RayTminEXT",             EbvRayTmin,            symbolTable);
            BuiltInVariable("gl_RayTmaxNV",              EbvRayTmax,            symbolTable);
            BuiltInVariable("gl_RayTmaxEXT",             EbvRayTmax,            symbolTable);
            BuiltInVariable("gl_CullMaskEXT",            EbvCullMask,           symbolTable);
            BuiltInVariable("gl_HitTNV",                 EbvHitT,               symbolTable);
            BuiltInVariable("gl_HitTEXT",                EbvHitT,               symbolTable);
            BuiltInVariable("gl_HitKindNV",              EbvHitKind,            symbolTable);
            BuiltInVariable("gl_HitKindEXT",             EbvHitKind,            symbolTable);
            BuiltInVariable("gl_ObjectToWorldNV",        EbvObjectToWorld,      symbolTable);
            BuiltInVariable("gl_ObjectToWorldEXT",       EbvObjectToWorld,      symbolTable);
            BuiltInVariable("gl_ObjectToWorld3x4EXT",    EbvObjectToWorld3x4,   symbolTable);
            BuiltInVariable("gl_WorldToObjectNV",        EbvWorldToObject,      symbolTable);
            BuiltInVariable("gl_WorldToObjectEXT",       EbvWorldToObject,      symbolTable);
            BuiltInVariable("gl_WorldToObject3x4EXT",    EbvWorldToObject3x4,   symbolTable);
            BuiltInVariable("gl_IncomingRayFlagsNV",     EbvIncomingRayFlags,   symbolTable);
            BuiltInVariable("gl_IncomingRayFlagsEXT",    EbvIncomingRayFlags,   symbolTable);
            BuiltInVariable("gl_DeviceIndex",            EbvDeviceIndex,        symbolTable);
            BuiltInVariable("gl_CurrentRayTimeNV",       EbvCurrentRayTimeNV,   symbolTable);
            BuiltInVariable("gl_HitTriangleVertexPositionsEXT", EbvPositionFetch, symbolTable);
            BuiltInVariable("gl_HitMicroTriangleVertexPositionsNV", EbvMicroTrianglePositionNV, symbolTable);
            BuiltInVariable("gl_HitMicroTriangleVertexBarycentricsNV", EbvMicroTriangleBaryNV, symbolTable);
            BuiltInVariable("gl_HitKindFrontFacingMicroTriangleNV", EbvHitKindFrontFacingMicroTriangleNV, symbolTable);
            BuiltInVariable("gl_HitKindBackFacingMicroTriangleNV", EbvHitKindBackFacingMicroTriangleNV, symbolTable);

            // GL_ARB_shader_ballot
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupInvocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            BuiltInVariable("gl_SubGroupInvocationARB", EbvSubGroupInvocation, symbolTable);
            BuiltInVariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0) {
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
                if (language == EShLangFragment)
                    ModifyFlatDecoration("gl_SubGroupSizeARB", true, symbolTable);
            }
            else
                BuiltInVariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);

            // GL_KHR_shader_subgroup
            symbolTable.setVariableExtensions("gl_NumSubgroups",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupID",           1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupInvocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltInVariable("gl_NumSubgroups",         EbvNumSubgroups,        symbolTable);
            BuiltInVariable("gl_SubgroupID",           EbvSubgroupID,          symbolTable);
            BuiltInVariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltInVariable("gl_SubgroupInvocationID", EbvSubgroupInvocation2, symbolTable);
            BuiltInVariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            // GL_NV_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMNV",         1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountNV",            1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDNV",             1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDNV",               1, &E_GL_NV_shader_sm_builtins);
            BuiltInVariable("gl_WarpsPerSMNV",          EbvWarpsPerSM,      symbolTable);
            BuiltInVariable("gl_SMCountNV",             EbvSMCount,         symbolTable);
            BuiltInVariable("gl_WarpIDNV",              EbvWarpID,          symbolTable);
            BuiltInVariable("gl_SMIDNV",                EbvSMID,            symbolTable);

            // GL_ARM_shader_core_builtins
            symbolTable.setVariableExtensions("gl_CoreCountARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);

            BuiltInVariable("gl_CoreCountARM", EbvCoreCountARM, symbolTable);
            BuiltInVariable("gl_CoreIDARM",    EbvCoreIDARM, symbolTable);
            BuiltInVariable("gl_CoreMaxIDARM", EbvCoreMaxIDARM, symbolTable);
            BuiltInVariable("gl_WarpIDARM",    EbvWarpIDARM, symbolTable);
            BuiltInVariable("gl_WarpMaxIDARM", EbvWarpMaxIDARM, symbolTable);
        }
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 450)) {
            symbolTable.setVariableExtensions("gl_ShadingRateFlag2VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag4VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag2HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag4HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
        }
        break;

    case EShLangMesh:
        if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
            // per-vertex builtins
            symbolTable.setVariableExtensions("gl_MeshVerticesNV", "gl_Position",     1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesNV", "gl_PointSize",    1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesNV", "gl_ClipDistance", 1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesNV", "gl_CullDistance", 1, &E_GL_NV_mesh_shader);

            BuiltInVariable("gl_MeshVerticesNV", "gl_Position",     EbvPosition,     symbolTable);
            BuiltInVariable("gl_MeshVerticesNV", "gl_PointSize",    EbvPointSize,    symbolTable);
            BuiltInVariable("gl_MeshVerticesNV", "gl_ClipDistance", EbvClipDistance, symbolTable);
            BuiltInVariable("gl_MeshVerticesNV", "gl_CullDistance", EbvCullDistance, symbolTable);

            symbolTable.setVariableExtensions("gl_MeshVerticesNV", "gl_PositionPerViewNV",     1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesNV", "gl_ClipDistancePerViewNV", 1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesNV", "gl_CullDistancePerViewNV", 1, &E_GL_NV_mesh_shader);

            BuiltInVariable("gl_MeshVerticesNV", "gl_PositionPerViewNV",     EbvPositionPerViewNV,     symbolTable);
            BuiltInVariable("gl_MeshVerticesNV", "gl_ClipDistancePerViewNV", EbvClipDistancePerViewNV, symbolTable);
            BuiltInVariable("gl_MeshVerticesNV", "gl_CullDistancePerViewNV", EbvCullDistancePerViewNV, symbolTable);

            // per-primitive builtins
            symbolTable.setVariableExtensions("gl_MeshPrimitivesNV", "gl_PrimitiveID",   1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesNV", "gl_Layer",         1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesNV", "gl_ViewportIndex", 1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesNV", "gl_ViewportMask",  1, &E_GL_NV_mesh_shader);

            BuiltInVariable("gl_MeshPrimitivesNV", "gl_PrimitiveID",   EbvPrimitiveId,    symbolTable);
            BuiltInVariable("gl_MeshPrimitivesNV", "gl_Layer",         EbvLayer,          symbolTable);
            BuiltInVariable("gl_MeshPrimitivesNV", "gl_ViewportIndex", EbvViewportIndex,  symbolTable);
            BuiltInVariable("gl_MeshPrimitivesNV", "gl_ViewportMask",  EbvViewportMaskNV, symbolTable);

            // per-view per-primitive builtins
            symbolTable.setVariableExtensions("gl_MeshPrimitivesNV", "gl_LayerPerViewNV",        1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesNV", "gl_ViewportMaskPerViewNV", 1, &E_GL_NV_mesh_shader);

            BuiltInVariable("gl_MeshPrimitivesNV", "gl_LayerPerViewNV",        EbvLayerPerViewNV,        symbolTable);
            BuiltInVariable("gl_MeshPrimitivesNV", "gl_ViewportMaskPerViewNV", EbvViewportMaskPerViewNV, symbolTable);

            // other builtins
            symbolTable.setVariableExtensions("gl_PrimitiveCountNV",     1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_PrimitiveIndicesNV",   1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshViewCountNV",      1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshViewIndicesNV",    1, &E_GL_NV_mesh_shader);
            if (profile != EEsProfile) {
                symbolTable.setVariableExtensions("gl_WorkGroupSize",        Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setVariableExtensions("gl_WorkGroupID",          Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setVariableExtensions("gl_LocalInvocationID",    Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setVariableExtensions("gl_GlobalInvocationID",   Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setVariableExtensions("gl_LocalInvocationIndex", Num_AEP_mesh_shader, AEP_mesh_shader);
            } else {
                symbolTable.setVariableExtensions("gl_WorkGroupSize",        1, &E_GL_NV_mesh_shader);
                symbolTable.setVariableExtensions("gl_WorkGroupID",          1, &E_GL_NV_mesh_shader);
                symbolTable.setVariableExtensions("gl_LocalInvocationID",    1, &E_GL_NV_mesh_shader);
                symbolTable.setVariableExtensions("gl_GlobalInvocationID",   1, &E_GL_NV_mesh_shader);
                symbolTable.setVariableExtensions("gl_LocalInvocationIndex", 1, &E_GL_NV_mesh_shader);
            }
            BuiltInVariable("gl_PrimitiveCountNV",     EbvPrimitiveCountNV,     symbolTable);
            BuiltInVariable("gl_PrimitiveIndicesNV",   EbvPrimitiveIndicesNV,   symbolTable);
            BuiltInVariable("gl_MeshViewCountNV",      EbvMeshViewCountNV,      symbolTable);
            BuiltInVariable("gl_MeshViewIndicesNV",    EbvMeshViewIndicesNV,    symbolTable);
            BuiltInVariable("gl_WorkGroupSize",        EbvWorkGroupSize,        symbolTable);
            BuiltInVariable("gl_WorkGroupID",          EbvWorkGroupId,          symbolTable);
            BuiltInVariable("gl_LocalInvocationID",    EbvLocalInvocationId,    symbolTable);
            BuiltInVariable("gl_GlobalInvocationID",   EbvGlobalInvocationId,   symbolTable);
            BuiltInVariable("gl_LocalInvocationIndex", EbvLocalInvocationIndex, symbolTable);

            // builtin constants
            symbolTable.setVariableExtensions("gl_MaxMeshOutputVerticesNV",   1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MaxMeshOutputPrimitivesNV", 1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MaxMeshWorkGroupSizeNV",    1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MaxMeshViewCountNV",        1, &E_GL_NV_mesh_shader);

            // builtin functions
            if (profile != EEsProfile) {
                symbolTable.setFunctionExtensions("barrier",                      Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setFunctionExtensions("memoryBarrierShared",          Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setFunctionExtensions("groupMemoryBarrier",           Num_AEP_mesh_shader, AEP_mesh_shader);
            } else {
                symbolTable.setFunctionExtensions("barrier",                      1, &E_GL_NV_mesh_shader);
                symbolTable.setFunctionExtensions("memoryBarrierShared",          1, &E_GL_NV_mesh_shader);
                symbolTable.setFunctionExtensions("groupMemoryBarrier",           1, &E_GL_NV_mesh_shader);
            }
            symbolTable.setFunctionExtensions("writePackedPrimitiveIndices4x8NV",  1, &E_GL_NV_mesh_shader);
        }

        if (profile != EEsProfile && version >= 450) {
            // GL_EXT_Mesh_shader
            symbolTable.setVariableExtensions("gl_PrimitivePointIndicesEXT",    1, &E_GL_EXT_mesh_shader);
            symbolTable.setVariableExtensions("gl_PrimitiveLineIndicesEXT",     1, &E_GL_EXT_mesh_shader);
            symbolTable.setVariableExtensions("gl_PrimitiveTriangleIndicesEXT", 1, &E_GL_EXT_mesh_shader);
            symbolTable.setVariableExtensions("gl_NumWorkGroups",               1, &E_GL_EXT_mesh_shader);

            BuiltInVariable("gl_PrimitivePointIndicesEXT",    EbvPrimitivePointIndicesEXT,    symbolTable);
            BuiltInVariable("gl_PrimitiveLineIndicesEXT",     EbvPrimitiveLineIndicesEXT,     symbolTable);
            BuiltInVariable("gl_PrimitiveTriangleIndicesEXT", EbvPrimitiveTriangleIndicesEXT, symbolTable);
            BuiltInVariable("gl_NumWorkGroups",        EbvNumWorkGroups,        symbolTable);

            symbolTable.setVariableExtensions("gl_MeshVerticesEXT", "gl_Position",     1, &E_GL_EXT_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesEXT", "gl_PointSize",    1, &E_GL_EXT_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesEXT", "gl_ClipDistance", 1, &E_GL_EXT_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesEXT", "gl_CullDistance", 1, &E_GL_EXT_mesh_shader);
            
            BuiltInVariable("gl_MeshVerticesEXT", "gl_Position",     EbvPosition,     symbolTable);
            BuiltInVariable("gl_MeshVerticesEXT", "gl_PointSize",    EbvPointSize,    symbolTable);
            BuiltInVariable("gl_MeshVerticesEXT", "gl_ClipDistance", EbvClipDistance, symbolTable);
            BuiltInVariable("gl_MeshVerticesEXT", "gl_CullDistance", EbvCullDistance, symbolTable);
            
            symbolTable.setVariableExtensions("gl_MeshPrimitivesEXT", "gl_PrimitiveID",             1, &E_GL_EXT_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesEXT", "gl_Layer",                   1, &E_GL_EXT_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesEXT", "gl_ViewportIndex",           1, &E_GL_EXT_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesEXT", "gl_CullPrimitiveEXT",        1, &E_GL_EXT_mesh_shader);

            // note: technically this member requires both GL_EXT_mesh_shader and GL_EXT_fragment_shading_rate
            // since setVariableExtensions only needs *one of* the extensions to validate, it's more useful to specify EXT_fragment_shading_rate
            // GL_EXT_mesh_shader will be required in practice by use of other fields of gl_MeshPrimitivesEXT
            symbolTable.setVariableExtensions("gl_MeshPrimitivesEXT", "gl_PrimitiveShadingRateEXT", 1, &E_GL_EXT_fragment_shading_rate);

            BuiltInVariable("gl_MeshPrimitivesEXT", "gl_PrimitiveID",              EbvPrimitiveId,    symbolTable);
            BuiltInVariable("gl_MeshPrimitivesEXT", "gl_Layer",                    EbvLayer,          symbolTable);
            BuiltInVariable("gl_MeshPrimitivesEXT", "gl_ViewportIndex",            EbvViewportIndex,  symbolTable);
            BuiltInVariable("gl_MeshPrimitivesEXT", "gl_CullPrimitiveEXT",         EbvCullPrimitiveEXT, symbolTable);
            BuiltInVariable("gl_MeshPrimitivesEXT", "gl_PrimitiveShadingRateEXT",  EbvPrimitiveShadingRateKHR, symbolTable);

            symbolTable.setFunctionExtensions("SetMeshOutputsEXT",  1, &E_GL_EXT_mesh_shader);

            // GL_EXT_device_group
            symbolTable.setVariableExtensions("gl_DeviceIndex", 1, &E_GL_EXT_device_group);
            BuiltInVariable("gl_DeviceIndex", EbvDeviceIndex, symbolTable);

            // GL_ARB_shader_draw_parameters
            symbolTable.setVariableExtensions("gl_DrawIDARB", 1, &E_GL_ARB_shader_draw_parameters);
            BuiltInVariable("gl_DrawIDARB", EbvDrawId, symbolTable);
            if (version >= 460) {
                BuiltInVariable("gl_DrawID", EbvDrawId, symbolTable);
            }
            // GL_EXT_multiview
            BuiltInVariable("gl_ViewIndex", EbvViewIndex, symbolTable);
            symbolTable.setVariableExtensions("gl_ViewIndex", 1, &E_GL_EXT_multiview);

            // GL_ARB_shader_ballot
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupInvocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            BuiltInVariable("gl_SubGroupInvocationARB", EbvSubGroupInvocation, symbolTable);
            BuiltInVariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0) {
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
                if (language == EShLangFragment)
                    ModifyFlatDecoration("gl_SubGroupSizeARB", true, symbolTable);
            }
            else
                BuiltInVariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_NumSubgroups",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupID",           1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupInvocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltInVariable("gl_NumSubgroups",         EbvNumSubgroups,        symbolTable);
            BuiltInVariable("gl_SubgroupID",           EbvSubgroupID,          symbolTable);
            BuiltInVariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltInVariable("gl_SubgroupInvocationID", EbvSubgroupInvocation2, symbolTable);
            BuiltInVariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            symbolTable.setFunctionExtensions("subgroupMemoryBarrierShared", 1, &E_GL_KHR_shader_subgroup_basic);

            // GL_NV_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMNV",         1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountNV",            1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDNV",             1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDNV",               1, &E_GL_NV_shader_sm_builtins);
            BuiltInVariable("gl_WarpsPerSMNV",          EbvWarpsPerSM,      symbolTable);
            BuiltInVariable("gl_SMCountNV",             EbvSMCount,         symbolTable);
            BuiltInVariable("gl_WarpIDNV",              EbvWarpID,          symbolTable);
            BuiltInVariable("gl_SMIDNV",                EbvSMID,            symbolTable);

            // GL_ARM_shader_core_builtins
            symbolTable.setVariableExtensions("gl_CoreCountARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);

            BuiltInVariable("gl_CoreCountARM", EbvCoreCountARM, symbolTable);
            BuiltInVariable("gl_CoreIDARM",    EbvCoreIDARM, symbolTable);
            BuiltInVariable("gl_CoreMaxIDARM", EbvCoreMaxIDARM, symbolTable);
            BuiltInVariable("gl_WarpIDARM",    EbvWarpIDARM, symbolTable);
            BuiltInVariable("gl_WarpMaxIDARM", EbvWarpMaxIDARM, symbolTable);
        }

        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 450)) {
            symbolTable.setVariableExtensions("gl_ShadingRateFlag2VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag4VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag2HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag4HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
        }

        // Builtins for GL_NV_displacment_micromap
        if ((profile != EEsProfile && version >= 460)) {
            symbolTable.setFunctionExtensions("fetchMicroTriangleVertexPositionNV", 1, &E_GL_NV_displacement_micromap);
            symbolTable.setFunctionExtensions("fetchMicroTriangleVertexBarycentricNV", 1, &E_GL_NV_displacement_micromap);
        }

        break;

    case EShLangTask:
        if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
            symbolTable.setVariableExtensions("gl_TaskCountNV",          1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshViewCountNV",      1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshViewIndicesNV",    1, &E_GL_NV_mesh_shader);
            if (profile != EEsProfile) {
                symbolTable.setVariableExtensions("gl_WorkGroupSize",        Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setVariableExtensions("gl_WorkGroupID",          Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setVariableExtensions("gl_LocalInvocationID",    Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setVariableExtensions("gl_GlobalInvocationID",   Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setVariableExtensions("gl_LocalInvocationIndex", Num_AEP_mesh_shader, AEP_mesh_shader);
            } else {
                symbolTable.setVariableExtensions("gl_WorkGroupSize",        1, &E_GL_NV_mesh_shader);
                symbolTable.setVariableExtensions("gl_WorkGroupID",          1, &E_GL_NV_mesh_shader);
                symbolTable.setVariableExtensions("gl_LocalInvocationID",    1, &E_GL_NV_mesh_shader);
                symbolTable.setVariableExtensions("gl_GlobalInvocationID",   1, &E_GL_NV_mesh_shader);
                symbolTable.setVariableExtensions("gl_LocalInvocationIndex", 1, &E_GL_NV_mesh_shader);
            }

            BuiltInVariable("gl_TaskCountNV",          EbvTaskCountNV,          symbolTable);
            BuiltInVariable("gl_WorkGroupSize",        EbvWorkGroupSize,        symbolTable);
            BuiltInVariable("gl_WorkGroupID",          EbvWorkGroupId,          symbolTable);
            BuiltInVariable("gl_LocalInvocationID",    EbvLocalInvocationId,    symbolTable);
            BuiltInVariable("gl_GlobalInvocationID",   EbvGlobalInvocationId,   symbolTable);
            BuiltInVariable("gl_LocalInvocationIndex", EbvLocalInvocationIndex, symbolTable);
            BuiltInVariable("gl_MeshViewCountNV",      EbvMeshViewCountNV,      symbolTable);
            BuiltInVariable("gl_MeshViewIndicesNV",    EbvMeshViewIndicesNV,    symbolTable);

            symbolTable.setVariableExtensions("gl_MaxTaskWorkGroupSizeNV", 1, &E_GL_NV_mesh_shader);
            symbolTable.setVariableExtensions("gl_MaxMeshViewCountNV",     1, &E_GL_NV_mesh_shader);

            if (profile != EEsProfile) {
                symbolTable.setFunctionExtensions("barrier",                   Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setFunctionExtensions("memoryBarrierShared",       Num_AEP_mesh_shader, AEP_mesh_shader);
                symbolTable.setFunctionExtensions("groupMemoryBarrier",        Num_AEP_mesh_shader, AEP_mesh_shader);
            } else {
                symbolTable.setFunctionExtensions("barrier",                   1, &E_GL_NV_mesh_shader);
                symbolTable.setFunctionExtensions("memoryBarrierShared",       1, &E_GL_NV_mesh_shader);
                symbolTable.setFunctionExtensions("groupMemoryBarrier",        1, &E_GL_NV_mesh_shader);
            }
        }

        if (profile != EEsProfile && version >= 450) {
            // GL_EXT_mesh_shader
            symbolTable.setFunctionExtensions("EmitMeshTasksEXT",          1, &E_GL_EXT_mesh_shader);
            symbolTable.setVariableExtensions("gl_NumWorkGroups",        1, &E_GL_EXT_mesh_shader);
            BuiltInVariable("gl_NumWorkGroups",        EbvNumWorkGroups,        symbolTable);

            // GL_EXT_device_group
            symbolTable.setVariableExtensions("gl_DeviceIndex", 1, &E_GL_EXT_device_group);
            BuiltInVariable("gl_DeviceIndex", EbvDeviceIndex, symbolTable);

            // GL_ARB_shader_draw_parameters
            symbolTable.setVariableExtensions("gl_DrawIDARB", 1, &E_GL_ARB_shader_draw_parameters);
            BuiltInVariable("gl_DrawIDARB", EbvDrawId, symbolTable);
            if (version >= 460) {
                BuiltInVariable("gl_DrawID", EbvDrawId, symbolTable);
            }

            // GL_ARB_shader_ballot
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupInvocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            BuiltInVariable("gl_SubGroupInvocationARB", EbvSubGroupInvocation, symbolTable);
            BuiltInVariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltInVariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0) {
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
                if (language == EShLangFragment)
                    ModifyFlatDecoration("gl_SubGroupSizeARB", true, symbolTable);
            }
            else
                BuiltInVariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_NumSubgroups",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupID",           1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupInvocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltInVariable("gl_NumSubgroups",         EbvNumSubgroups,        symbolTable);
            BuiltInVariable("gl_SubgroupID",           EbvSubgroupID,          symbolTable);
            BuiltInVariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltInVariable("gl_SubgroupInvocationID", EbvSubgroupInvocation2, symbolTable);
            BuiltInVariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltInVariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            symbolTable.setFunctionExtensions("subgroupMemoryBarrierShared", 1, &E_GL_KHR_shader_subgroup_basic);

            // GL_NV_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMNV",         1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountNV",            1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDNV",             1, &E_GL_NV_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDNV",               1, &E_GL_NV_shader_sm_builtins);
            BuiltInVariable("gl_WarpsPerSMNV",          EbvWarpsPerSM,      symbolTable);
            BuiltInVariable("gl_SMCountNV",             EbvSMCount,         symbolTable);
            BuiltInVariable("gl_WarpIDNV",              EbvWarpID,          symbolTable);
            BuiltInVariable("gl_SMIDNV",                EbvSMID,            symbolTable);

            // GL_ARM_shader_core_builtins
            symbolTable.setVariableExtensions("gl_CoreCountARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_CoreMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDARM",    1, &E_GL_ARM_shader_core_builtins);
            symbolTable.setVariableExtensions("gl_WarpMaxIDARM", 1, &E_GL_ARM_shader_core_builtins);

            BuiltInVariable("gl_CoreCountARM", EbvCoreCountARM, symbolTable);
            BuiltInVariable("gl_CoreIDARM",    EbvCoreIDARM, symbolTable);
            BuiltInVariable("gl_CoreMaxIDARM", EbvCoreMaxIDARM, symbolTable);
            BuiltInVariable("gl_WarpIDARM",    EbvWarpIDARM, symbolTable);
            BuiltInVariable("gl_WarpMaxIDARM", EbvWarpMaxIDARM, symbolTable);
        }
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 450)) {
            symbolTable.setVariableExtensions("gl_ShadingRateFlag2VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag4VerticalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag2HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
            symbolTable.setVariableExtensions("gl_ShadingRateFlag4HorizontalPixelsEXT", 1, &E_GL_EXT_fragment_shading_rate);
        }
        break;

    default:
        assert(false && "Language not supported");
        break;
    }

    //
    // Next, identify which built-ins have a mapping to an operator.
    // If PureOperatorBuiltins is false, those that are not identified as such are
    // expected to be resolved through a library of functions, versus as
    // operations.
    //

    relateTabledBuiltins(version, profile, spvVersion, language, symbolTable);

    symbolTable.relateToOperator("doubleBitsToInt64",  EOpDoubleBitsToInt64);
    symbolTable.relateToOperator("doubleBitsToUint64", EOpDoubleBitsToUint64);
    symbolTable.relateToOperator("int64BitsToDouble",  EOpInt64BitsToDouble);
    symbolTable.relateToOperator("uint64BitsToDouble", EOpUint64BitsToDouble);
    symbolTable.relateToOperator("halfBitsToInt16",  EOpFloat16BitsToInt16);
    symbolTable.relateToOperator("halfBitsToUint16", EOpFloat16BitsToUint16);
    symbolTable.relateToOperator("float16BitsToInt16",  EOpFloat16BitsToInt16);
    symbolTable.relateToOperator("float16BitsToUint16", EOpFloat16BitsToUint16);
    symbolTable.relateToOperator("int16BitsToFloat16",  EOpInt16BitsToFloat16);
    symbolTable.relateToOperator("uint16BitsToFloat16", EOpUint16BitsToFloat16);

    symbolTable.relateToOperator("int16BitsToHalf",  EOpInt16BitsToFloat16);
    symbolTable.relateToOperator("uint16BitsToHalf", EOpUint16BitsToFloat16);

    symbolTable.relateToOperator("packSnorm4x8",    EOpPackSnorm4x8);
    symbolTable.relateToOperator("unpackSnorm4x8",  EOpUnpackSnorm4x8);
    symbolTable.relateToOperator("packUnorm4x8",    EOpPackUnorm4x8);
    symbolTable.relateToOperator("unpackUnorm4x8",  EOpUnpackUnorm4x8);

    symbolTable.relateToOperator("packDouble2x32",    EOpPackDouble2x32);
    symbolTable.relateToOperator("unpackDouble2x32",  EOpUnpackDouble2x32);

    symbolTable.relateToOperator("packInt2x32",     EOpPackInt2x32);
    symbolTable.relateToOperator("unpackInt2x32",   EOpUnpackInt2x32);
    symbolTable.relateToOperator("packUint2x32",    EOpPackUint2x32);
    symbolTable.relateToOperator("unpackUint2x32",  EOpUnpackUint2x32);

    symbolTable.relateToOperator("packInt2x16",     EOpPackInt2x16);
    symbolTable.relateToOperator("unpackInt2x16",   EOpUnpackInt2x16);
    symbolTable.relateToOperator("packUint2x16",    EOpPackUint2x16);
    symbolTable.relateToOperator("unpackUint2x16",  EOpUnpackUint2x16);

    symbolTable.relateToOperator("packInt4x16",     EOpPackInt4x16);
    symbolTable.relateToOperator("unpackInt4x16",   EOpUnpackInt4x16);
    symbolTable.relateToOperator("packUint4x16",    EOpPackUint4x16);
    symbolTable.relateToOperator("unpackUint4x16",  EOpUnpackUint4x16);
    symbolTable.relateToOperator("packFloat2x16",   EOpPackFloat2x16);
    symbolTable.relateToOperator("unpackFloat2x16", EOpUnpackFloat2x16);

    symbolTable.relateToOperator("pack16",          EOpPack16);
    symbolTable.relateToOperator("pack32",          EOpPack32);
    symbolTable.relateToOperator("pack64",          EOpPack64);

    symbolTable.relateToOperator("unpack32",        EOpUnpack32);
    symbolTable.relateToOperator("unpack16",        EOpUnpack16);
    symbolTable.relateToOperator("unpack8",         EOpUnpack8);

    symbolTable.relateToOperator("controlBarrier",             EOpBarrier);
    symbolTable.relateToOperator("memoryBarrierAtomicCounter", EOpMemoryBarrierAtomicCounter);
    symbolTable.relateToOperator("memoryBarrierImage",         EOpMemoryBarrierImage);

    if (spvVersion.vulkanRelaxed) {
        //
        // functions signature have been replaced to take uint operations on buffer variables
        // remap atomic counter functions to atomic operations
        //
        symbolTable.relateToOperator("memoryBarrierAtomicCounter", EOpMemoryBarrierBuffer);
    }

    symbolTable.relateToOperator("atomicLoad",     EOpAtomicLoad);
    symbolTable.relateToOperator("atomicStore",    EOpAtomicStore);

    symbolTable.relateToOperator("atomicCounterIncrement", EOpAtomicCounterIncrement);
    symbolTable.relateToOperator("atomicCounterDecrement", EOpAtomicCounterDecrement);
    symbolTable.relateToOperator("atomicCounter",          EOpAtomicCounter);

    if (spvVersion.vulkanRelaxed) {
        //
        // functions signature have been replaced to take uint operations
        // remap atomic counter functions to atomic operations
        //
        // these atomic counter functions do not match signatures of glsl
        // atomic functions, so they will be remapped to semantically
        // equivalent functions in the parser
        //
        symbolTable.relateToOperator("atomicCounterIncrement", EOpNull);
        symbolTable.relateToOperator("atomicCounterDecrement", EOpNull);
        symbolTable.relateToOperator("atomicCounter", EOpNull);
    }

    symbolTable.relateToOperator("clockARB",     EOpReadClockSubgroupKHR);
    symbolTable.relateToOperator("clock2x32ARB", EOpReadClockSubgroupKHR);

    symbolTable.relateToOperator("clockRealtimeEXT",     EOpReadClockDeviceKHR);
    symbolTable.relateToOperator("clockRealtime2x32EXT", EOpReadClockDeviceKHR);

    if (profile != EEsProfile && version == 450) {
        symbolTable.relateToOperator("atomicCounterAddARB",      EOpAtomicCounterAdd);
        symbolTable.relateToOperator("atomicCounterSubtractARB", EOpAtomicCounterSubtract);
        symbolTable.relateToOperator("atomicCounterMinARB",      EOpAtomicCounterMin);
        symbolTable.relateToOperator("atomicCounterMaxARB",      EOpAtomicCounterMax);
        symbolTable.relateToOperator("atomicCounterAndARB",      EOpAtomicCounterAnd);
        symbolTable.relateToOperator("atomicCounterOrARB",       EOpAtomicCounterOr);
        symbolTable.relateToOperator("atomicCounterXorARB",      EOpAtomicCounterXor);
        symbolTable.relateToOperator("atomicCounterExchangeARB", EOpAtomicCounterExchange);
        symbolTable.relateToOperator("atomicCounterCompSwapARB", EOpAtomicCounterCompSwap);
    }

    if (profile != EEsProfile && version >= 460) {
        symbolTable.relateToOperator("atomicCounterAdd",      EOpAtomicCounterAdd);
        symbolTable.relateToOperator("atomicCounterSubtract", EOpAtomicCounterSubtract);
        symbolTable.relateToOperator("atomicCounterMin",      EOpAtomicCounterMin);
        symbolTable.relateToOperator("atomicCounterMax",      EOpAtomicCounterMax);
        symbolTable.relateToOperator("atomicCounterAnd",      EOpAtomicCounterAnd);
        symbolTable.relateToOperator("atomicCounterOr",       EOpAtomicCounterOr);
        symbolTable.relateToOperator("atomicCounterXor",      EOpAtomicCounterXor);
        symbolTable.relateToOperator("atomicCounterExchange", EOpAtomicCounterExchange);
        symbolTable.relateToOperator("atomicCounterCompSwap", EOpAtomicCounterCompSwap);
    }

    if (spvVersion.vulkanRelaxed) {
        //
        // functions signature have been replaced to take 'uint' instead of 'atomic_uint'
        // remap atomic counter functions to non-counter atomic ops so
        // functions act as aliases to non-counter atomic ops
        //
        symbolTable.relateToOperator("atomicCounterAdd", EOpAtomicAdd);
        symbolTable.relateToOperator("atomicCounterSubtract", EOpAtomicSubtract);
        symbolTable.relateToOperator("atomicCounterMin", EOpAtomicMin);
        symbolTable.relateToOperator("atomicCounterMax", EOpAtomicMax);
        symbolTable.relateToOperator("atomicCounterAnd", EOpAtomicAnd);
        symbolTable.relateToOperator("atomicCounterOr", EOpAtomicOr);
        symbolTable.relateToOperator("atomicCounterXor", EOpAtomicXor);
        symbolTable.relateToOperator("atomicCounterExchange", EOpAtomicExchange);
        symbolTable.relateToOperator("atomicCounterCompSwap", EOpAtomicCompSwap);
    }

    symbolTable.relateToOperator("fma",               EOpFma);
    symbolTable.relateToOperator("frexp",             EOpFrexp);
    symbolTable.relateToOperator("ldexp",             EOpLdexp);
    symbolTable.relateToOperator("uaddCarry",         EOpAddCarry);
    symbolTable.relateToOperator("usubBorrow",        EOpSubBorrow);
    symbolTable.relateToOperator("umulExtended",      EOpUMulExtended);
    symbolTable.relateToOperator("imulExtended",      EOpIMulExtended);
    symbolTable.relateToOperator("bitfieldExtract",   EOpBitfieldExtract);
    symbolTable.relateToOperator("bitfieldInsert",    EOpBitfieldInsert);
    symbolTable.relateToOperator("bitfieldReverse",   EOpBitFieldReverse);
    symbolTable.relateToOperator("bitCount",          EOpBitCount);
    symbolTable.relateToOperator("findLSB",           EOpFindLSB);
    symbolTable.relateToOperator("findMSB",           EOpFindMSB);

    symbolTable.relateToOperator("helperInvocationEXT",  EOpIsHelperInvocation);

    symbolTable.relateToOperator("countLeadingZeros",  EOpCountLeadingZeros);
    symbolTable.relateToOperator("countTrailingZeros", EOpCountTrailingZeros);
    symbolTable.relateToOperator("absoluteDifference", EOpAbsDifference);
    symbolTable.relateToOperator("addSaturate",        EOpAddSaturate);
    symbolTable.relateToOperator("subtractSaturate",   EOpSubSaturate);
    symbolTable.relateToOperator("average",            EOpAverage);
    symbolTable.relateToOperator("averageRounded",     EOpAverageRounded);
    symbolTable.relateToOperator("multiply32x16",      EOpMul32x16);
    symbolTable.relateToOperator("debugPrintfEXT",     EOpDebugPrintf);
    symbolTable.relateToOperator("assumeEXT",          EOpAssumeEXT);
    symbolTable.relateToOperator("expectEXT",          EOpExpectEXT);


    if (PureOperatorBuiltins) {
        symbolTable.relateToOperator("imageSize",               EOpImageQuerySize);
        symbolTable.relateToOperator("imageSamples",            EOpImageQuerySamples);
        symbolTable.relateToOperator("imageLoad",               EOpImageLoad);
        symbolTable.relateToOperator("imageStore",              EOpImageStore);
        symbolTable.relateToOperator("imageAtomicAdd",          EOpImageAtomicAdd);
        symbolTable.relateToOperator("imageAtomicMin",          EOpImageAtomicMin);
        symbolTable.relateToOperator("imageAtomicMax",          EOpImageAtomicMax);
        symbolTable.relateToOperator("imageAtomicAnd",          EOpImageAtomicAnd);
        symbolTable.relateToOperator("imageAtomicOr",           EOpImageAtomicOr);
        symbolTable.relateToOperator("imageAtomicXor",          EOpImageAtomicXor);
        symbolTable.relateToOperator("imageAtomicExchange",     EOpImageAtomicExchange);
        symbolTable.relateToOperator("imageAtomicCompSwap",     EOpImageAtomicCompSwap);
        symbolTable.relateToOperator("imageAtomicLoad",         EOpImageAtomicLoad);
        symbolTable.relateToOperator("imageAtomicStore",        EOpImageAtomicStore);

        symbolTable.relateToOperator("subpassLoad",             EOpSubpassLoad);
        symbolTable.relateToOperator("subpassLoadMS",           EOpSubpassLoadMS);

        symbolTable.relateToOperator("textureGather",           EOpTextureGather);
        symbolTable.relateToOperator("textureGatherOffset",     EOpTextureGatherOffset);
        symbolTable.relateToOperator("textureGatherOffsets",    EOpTextureGatherOffsets);

        symbolTable.relateToOperator("noise1", EOpNoise);
        symbolTable.relateToOperator("noise2", EOpNoise);
        symbolTable.relateToOperator("noise3", EOpNoise);
        symbolTable.relateToOperator("noise4", EOpNoise);

        symbolTable.relateToOperator("textureFootprintNV",          EOpImageSampleFootprintNV);
        symbolTable.relateToOperator("textureFootprintClampNV",     EOpImageSampleFootprintClampNV);
        symbolTable.relateToOperator("textureFootprintLodNV",       EOpImageSampleFootprintLodNV);
        symbolTable.relateToOperator("textureFootprintGradNV",      EOpImageSampleFootprintGradNV);
        symbolTable.relateToOperator("textureFootprintGradClampNV", EOpImageSampleFootprintGradClampNV);

        if (spvVersion.spv == 0 && IncludeLegacy(version, profile, spvVersion))
            symbolTable.relateToOperator("ftransform", EOpFtransform);

        if (spvVersion.spv == 0 && (IncludeLegacy(version, profile, spvVersion) ||
            (profile == EEsProfile && version == 100))) {

            symbolTable.relateToOperator("texture1D",                EOpTexture);
            symbolTable.relateToOperator("texture1DGradARB",         EOpTextureGrad);
            symbolTable.relateToOperator("texture1DProj",            EOpTextureProj);
            symbolTable.relateToOperator("texture1DProjGradARB",     EOpTextureProjGrad);
            symbolTable.relateToOperator("texture1DLod",             EOpTextureLod);
            symbolTable.relateToOperator("texture1DProjLod",         EOpTextureProjLod);

            symbolTable.relateToOperator("texture2DRect",            EOpTexture);
            symbolTable.relateToOperator("texture2DRectProj",        EOpTextureProj);
            symbolTable.relateToOperator("texture2DRectGradARB",     EOpTextureGrad);
            symbolTable.relateToOperator("texture2DRectProjGradARB", EOpTextureProjGrad);
            symbolTable.relateToOperator("shadow2DRect",             EOpTexture);
            symbolTable.relateToOperator("shadow2DRectProj",         EOpTextureProj);
            symbolTable.relateToOperator("shadow2DRectGradARB",      EOpTextureGrad);
            symbolTable.relateToOperator("shadow2DRectProjGradARB",  EOpTextureProjGrad);

            symbolTable.relateToOperator("texture2D",                EOpTexture);
            symbolTable.relateToOperator("texture2DProj",            EOpTextureProj);
            symbolTable.relateToOperator("texture2DGradEXT",         EOpTextureGrad);
            symbolTable.relateToOperator("texture2DGradARB",         EOpTextureGrad);
            symbolTable.relateToOperator("texture2DProjGradEXT",     EOpTextureProjGrad);
            symbolTable.relateToOperator("texture2DProjGradARB",     EOpTextureProjGrad);
            symbolTable.relateToOperator("texture2DLod",             EOpTextureLod);
            symbolTable.relateToOperator("texture2DLodEXT",          EOpTextureLod);
            symbolTable.relateToOperator("texture2DProjLod",         EOpTextureProjLod);
            symbolTable.relateToOperator("texture2DProjLodEXT",      EOpTextureProjLod);

            symbolTable.relateToOperator("texture3D",                EOpTexture);
            symbolTable.relateToOperator("texture3DGradARB",         EOpTextureGrad);
            symbolTable.relateToOperator("texture3DProj",            EOpTextureProj);
            symbolTable.relateToOperator("texture3DProjGradARB",     EOpTextureProjGrad);
            symbolTable.relateToOperator("texture3DLod",             EOpTextureLod);
            symbolTable.relateToOperator("texture3DProjLod",         EOpTextureProjLod);
            symbolTable.relateToOperator("textureCube",              EOpTexture);
            symbolTable.relateToOperator("textureCubeGradEXT",       EOpTextureGrad);
            symbolTable.relateToOperator("textureCubeGradARB",       EOpTextureGrad);
            symbolTable.relateToOperator("textureCubeLod",           EOpTextureLod);
            symbolTable.relateToOperator("textureCubeLodEXT",        EOpTextureLod);
            symbolTable.relateToOperator("shadow1D",                 EOpTexture);
            symbolTable.relateToOperator("shadow1DGradARB",          EOpTextureGrad);
            symbolTable.relateToOperator("shadow2D",                 EOpTexture);
            symbolTable.relateToOperator("shadow2DGradARB",          EOpTextureGrad);
            symbolTable.relateToOperator("shadow1DProj",             EOpTextureProj);
            symbolTable.relateToOperator("shadow2DProj",             EOpTextureProj);
            symbolTable.relateToOperator("shadow1DProjGradARB",      EOpTextureProjGrad);
            symbolTable.relateToOperator("shadow2DProjGradARB",      EOpTextureProjGrad);
            symbolTable.relateToOperator("shadow1DLod",              EOpTextureLod);
            symbolTable.relateToOperator("shadow2DLod",              EOpTextureLod);
            symbolTable.relateToOperator("shadow1DProjLod",          EOpTextureProjLod);
            symbolTable.relateToOperator("shadow2DProjLod",          EOpTextureProjLod);
        }

        if (profile != EEsProfile) {
            symbolTable.relateToOperator("sparseTextureARB",                EOpSparseTexture);
            symbolTable.relateToOperator("sparseTextureLodARB",             EOpSparseTextureLod);
            symbolTable.relateToOperator("sparseTextureOffsetARB",          EOpSparseTextureOffset);
            symbolTable.relateToOperator("sparseTexelFetchARB",             EOpSparseTextureFetch);
            symbolTable.relateToOperator("sparseTexelFetchOffsetARB",       EOpSparseTextureFetchOffset);
            symbolTable.relateToOperator("sparseTextureLodOffsetARB",       EOpSparseTextureLodOffset);
            symbolTable.relateToOperator("sparseTextureGradARB",            EOpSparseTextureGrad);
            symbolTable.relateToOperator("sparseTextureGradOffsetARB",      EOpSparseTextureGradOffset);
            symbolTable.relateToOperator("sparseTextureGatherARB",          EOpSparseTextureGather);
            symbolTable.relateToOperator("sparseTextureGatherOffsetARB",    EOpSparseTextureGatherOffset);
            symbolTable.relateToOperator("sparseTextureGatherOffsetsARB",   EOpSparseTextureGatherOffsets);
            symbolTable.relateToOperator("sparseImageLoadARB",              EOpSparseImageLoad);
            symbolTable.relateToOperator("sparseTexelsResidentARB",         EOpSparseTexelsResident);

            symbolTable.relateToOperator("sparseTextureClampARB",           EOpSparseTextureClamp);
            symbolTable.relateToOperator("sparseTextureOffsetClampARB",     EOpSparseTextureOffsetClamp);
            symbolTable.relateToOperator("sparseTextureGradClampARB",       EOpSparseTextureGradClamp);
            symbolTable.relateToOperator("sparseTextureGradOffsetClampARB", EOpSparseTextureGradOffsetClamp);
            symbolTable.relateToOperator("textureClampARB",                 EOpTextureClamp);
            symbolTable.relateToOperator("textureOffsetClampARB",           EOpTextureOffsetClamp);
            symbolTable.relateToOperator("textureGradClampARB",             EOpTextureGradClamp);
            symbolTable.relateToOperator("textureGradOffsetClampARB",       EOpTextureGradOffsetClamp);

            symbolTable.relateToOperator("ballotARB",                       EOpBallot);
            symbolTable.relateToOperator("readInvocationARB",               EOpReadInvocation);
            symbolTable.relateToOperator("readFirstInvocationARB",          EOpReadFirstInvocation);

            if (version >= 430) {
                symbolTable.relateToOperator("anyInvocationARB",            EOpAnyInvocation);
                symbolTable.relateToOperator("allInvocationsARB",           EOpAllInvocations);
                symbolTable.relateToOperator("allInvocationsEqualARB",      EOpAllInvocationsEqual);
            }
            if (version >= 460) {
                symbolTable.relateToOperator("anyInvocation",               EOpAnyInvocation);
                symbolTable.relateToOperator("allInvocations",              EOpAllInvocations);
                symbolTable.relateToOperator("allInvocationsEqual",         EOpAllInvocationsEqual);
            }
            symbolTable.relateToOperator("minInvocationsAMD",                           EOpMinInvocations);
            symbolTable.relateToOperator("maxInvocationsAMD",                           EOpMaxInvocations);
            symbolTable.relateToOperator("addInvocationsAMD",                           EOpAddInvocations);
            symbolTable.relateToOperator("minInvocationsNonUniformAMD",                 EOpMinInvocationsNonUniform);
            symbolTable.relateToOperator("maxInvocationsNonUniformAMD",                 EOpMaxInvocationsNonUniform);
            symbolTable.relateToOperator("addInvocationsNonUniformAMD",                 EOpAddInvocationsNonUniform);
            symbolTable.relateToOperator("minInvocationsInclusiveScanAMD",              EOpMinInvocationsInclusiveScan);
            symbolTable.relateToOperator("maxInvocationsInclusiveScanAMD",              EOpMaxInvocationsInclusiveScan);
            symbolTable.relateToOperator("addInvocationsInclusiveScanAMD",              EOpAddInvocationsInclusiveScan);
            symbolTable.relateToOperator("minInvocationsInclusiveScanNonUniformAMD",    EOpMinInvocationsInclusiveScanNonUniform);
            symbolTable.relateToOperator("maxInvocationsInclusiveScanNonUniformAMD",    EOpMaxInvocationsInclusiveScanNonUniform);
            symbolTable.relateToOperator("addInvocationsInclusiveScanNonUniformAMD",    EOpAddInvocationsInclusiveScanNonUniform);
            symbolTable.relateToOperator("minInvocationsExclusiveScanAMD",              EOpMinInvocationsExclusiveScan);
            symbolTable.relateToOperator("maxInvocationsExclusiveScanAMD",              EOpMaxInvocationsExclusiveScan);
            symbolTable.relateToOperator("addInvocationsExclusiveScanAMD",              EOpAddInvocationsExclusiveScan);
            symbolTable.relateToOperator("minInvocationsExclusiveScanNonUniformAMD",    EOpMinInvocationsExclusiveScanNonUniform);
            symbolTable.relateToOperator("maxInvocationsExclusiveScanNonUniformAMD",    EOpMaxInvocationsExclusiveScanNonUniform);
            symbolTable.relateToOperator("addInvocationsExclusiveScanNonUniformAMD",    EOpAddInvocationsExclusiveScanNonUniform);
            symbolTable.relateToOperator("swizzleInvocationsAMD",                       EOpSwizzleInvocations);
            symbolTable.relateToOperator("swizzleInvocationsMaskedAMD",                 EOpSwizzleInvocationsMasked);
            symbolTable.relateToOperator("writeInvocationAMD",                          EOpWriteInvocation);
            symbolTable.relateToOperator("mbcntAMD",                                    EOpMbcnt);

            symbolTable.relateToOperator("min3",    EOpMin3);
            symbolTable.relateToOperator("max3",    EOpMax3);
            symbolTable.relateToOperator("mid3",    EOpMid3);

            symbolTable.relateToOperator("cubeFaceIndexAMD",    EOpCubeFaceIndex);
            symbolTable.relateToOperator("cubeFaceCoordAMD",    EOpCubeFaceCoord);
            symbolTable.relateToOperator("timeAMD",             EOpTime);

            symbolTable.relateToOperator("textureGatherLodAMD",                 EOpTextureGatherLod);
            symbolTable.relateToOperator("textureGatherLodOffsetAMD",           EOpTextureGatherLodOffset);
            symbolTable.relateToOperator("textureGatherLodOffsetsAMD",          EOpTextureGatherLodOffsets);
            symbolTable.relateToOperator("sparseTextureGatherLodAMD",           EOpSparseTextureGatherLod);
            symbolTable.relateToOperator("sparseTextureGatherLodOffsetAMD",     EOpSparseTextureGatherLodOffset);
            symbolTable.relateToOperator("sparseTextureGatherLodOffsetsAMD",    EOpSparseTextureGatherLodOffsets);

            symbolTable.relateToOperator("imageLoadLodAMD",                     EOpImageLoadLod);
            symbolTable.relateToOperator("imageStoreLodAMD",                    EOpImageStoreLod);
            symbolTable.relateToOperator("sparseImageLoadLodAMD",               EOpSparseImageLoadLod);

            symbolTable.relateToOperator("fragmentMaskFetchAMD",                EOpFragmentMaskFetch);
            symbolTable.relateToOperator("fragmentFetchAMD",                    EOpFragmentFetch);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.relateToOperator("subgroupBarrier",                 EOpSubgroupBarrier);
            symbolTable.relateToOperator("subgroupMemoryBarrier",           EOpSubgroupMemoryBarrier);
            symbolTable.relateToOperator("subgroupMemoryBarrierBuffer",     EOpSubgroupMemoryBarrierBuffer);
            symbolTable.relateToOperator("subgroupMemoryBarrierImage",      EOpSubgroupMemoryBarrierImage);
            symbolTable.relateToOperator("subgroupElect",                   EOpSubgroupElect);
            symbolTable.relateToOperator("subgroupAll",                     EOpSubgroupAll);
            symbolTable.relateToOperator("subgroupAny",                     EOpSubgroupAny);
            symbolTable.relateToOperator("subgroupAllEqual",                EOpSubgroupAllEqual);
            symbolTable.relateToOperator("subgroupBroadcast",               EOpSubgroupBroadcast);
            symbolTable.relateToOperator("subgroupBroadcastFirst",          EOpSubgroupBroadcastFirst);
            symbolTable.relateToOperator("subgroupBallot",                  EOpSubgroupBallot);
            symbolTable.relateToOperator("subgroupInverseBallot",           EOpSubgroupInverseBallot);
            symbolTable.relateToOperator("subgroupBallotBitExtract",        EOpSubgroupBallotBitExtract);
            symbolTable.relateToOperator("subgroupBallotBitCount",          EOpSubgroupBallotBitCount);
            symbolTable.relateToOperator("subgroupBallotInclusiveBitCount", EOpSubgroupBallotInclusiveBitCount);
            symbolTable.relateToOperator("subgroupBallotExclusiveBitCount", EOpSubgroupBallotExclusiveBitCount);
            symbolTable.relateToOperator("subgroupBallotFindLSB",           EOpSubgroupBallotFindLSB);
            symbolTable.relateToOperator("subgroupBallotFindMSB",           EOpSubgroupBallotFindMSB);
            symbolTable.relateToOperator("subgroupShuffle",                 EOpSubgroupShuffle);
            symbolTable.relateToOperator("subgroupShuffleXor",              EOpSubgroupShuffleXor);
            symbolTable.relateToOperator("subgroupShuffleUp",               EOpSubgroupShuffleUp);
            symbolTable.relateToOperator("subgroupShuffleDown",             EOpSubgroupShuffleDown);
            symbolTable.relateToOperator("subgroupRotate",                  EOpSubgroupRotate);
            symbolTable.relateToOperator("subgroupClusteredRotate",         EOpSubgroupClusteredRotate);
            symbolTable.relateToOperator("subgroupAdd",                     EOpSubgroupAdd);
            symbolTable.relateToOperator("subgroupMul",                     EOpSubgroupMul);
            symbolTable.relateToOperator("subgroupMin",                     EOpSubgroupMin);
            symbolTable.relateToOperator("subgroupMax",                     EOpSubgroupMax);
            symbolTable.relateToOperator("subgroupAnd",                     EOpSubgroupAnd);
            symbolTable.relateToOperator("subgroupOr",                      EOpSubgroupOr);
            symbolTable.relateToOperator("subgroupXor",                     EOpSubgroupXor);
            symbolTable.relateToOperator("subgroupInclusiveAdd",            EOpSubgroupInclusiveAdd);
            symbolTable.relateToOperator("subgroupInclusiveMul",            EOpSubgroupInclusiveMul);
            symbolTable.relateToOperator("subgroupInclusiveMin",            EOpSubgroupInclusiveMin);
            symbolTable.relateToOperator("subgroupInclusiveMax",            EOpSubgroupInclusiveMax);
            symbolTable.relateToOperator("subgroupInclusiveAnd",            EOpSubgroupInclusiveAnd);
            symbolTable.relateToOperator("subgroupInclusiveOr",             EOpSubgroupInclusiveOr);
            symbolTable.relateToOperator("subgroupInclusiveXor",            EOpSubgroupInclusiveXor);
            symbolTable.relateToOperator("subgroupExclusiveAdd",            EOpSubgroupExclusiveAdd);
            symbolTable.relateToOperator("subgroupExclusiveMul",            EOpSubgroupExclusiveMul);
            symbolTable.relateToOperator("subgroupExclusiveMin",            EOpSubgroupExclusiveMin);
            symbolTable.relateToOperator("subgroupExclusiveMax",            EOpSubgroupExclusiveMax);
            symbolTable.relateToOperator("subgroupExclusiveAnd",            EOpSubgroupExclusiveAnd);
            symbolTable.relateToOperator("subgroupExclusiveOr",             EOpSubgroupExclusiveOr);
            symbolTable.relateToOperator("subgroupExclusiveXor",            EOpSubgroupExclusiveXor);
            symbolTable.relateToOperator("subgroupClusteredAdd",            EOpSubgroupClusteredAdd);
            symbolTable.relateToOperator("subgroupClusteredMul",            EOpSubgroupClusteredMul);
            symbolTable.relateToOperator("subgroupClusteredMin",            EOpSubgroupClusteredMin);
            symbolTable.relateToOperator("subgroupClusteredMax",            EOpSubgroupClusteredMax);
            symbolTable.relateToOperator("subgroupClusteredAnd",            EOpSubgroupClusteredAnd);
            symbolTable.relateToOperator("subgroupClusteredOr",             EOpSubgroupClusteredOr);
            symbolTable.relateToOperator("subgroupClusteredXor",            EOpSubgroupClusteredXor);
            symbolTable.relateToOperator("subgroupQuadBroadcast",           EOpSubgroupQuadBroadcast);
            symbolTable.relateToOperator("subgroupQuadSwapHorizontal",      EOpSubgroupQuadSwapHorizontal);
            symbolTable.relateToOperator("subgroupQuadSwapVertical",        EOpSubgroupQuadSwapVertical);
            symbolTable.relateToOperator("subgroupQuadSwapDiagonal",        EOpSubgroupQuadSwapDiagonal);

            symbolTable.relateToOperator("subgroupPartitionNV",                          EOpSubgroupPartition);
            symbolTable.relateToOperator("subgroupPartitionedAddNV",                     EOpSubgroupPartitionedAdd);
            symbolTable.relateToOperator("subgroupPartitionedMulNV",                     EOpSubgroupPartitionedMul);
            symbolTable.relateToOperator("subgroupPartitionedMinNV",                     EOpSubgroupPartitionedMin);
            symbolTable.relateToOperator("subgroupPartitionedMaxNV",                     EOpSubgroupPartitionedMax);
            symbolTable.relateToOperator("subgroupPartitionedAndNV",                     EOpSubgroupPartitionedAnd);
            symbolTable.relateToOperator("subgroupPartitionedOrNV",                      EOpSubgroupPartitionedOr);
            symbolTable.relateToOperator("subgroupPartitionedXorNV",                     EOpSubgroupPartitionedXor);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveAddNV",            EOpSubgroupPartitionedInclusiveAdd);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveMulNV",            EOpSubgroupPartitionedInclusiveMul);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveMinNV",            EOpSubgroupPartitionedInclusiveMin);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveMaxNV",            EOpSubgroupPartitionedInclusiveMax);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveAndNV",            EOpSubgroupPartitionedInclusiveAnd);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveOrNV",             EOpSubgroupPartitionedInclusiveOr);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveXorNV",            EOpSubgroupPartitionedInclusiveXor);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveAddNV",            EOpSubgroupPartitionedExclusiveAdd);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveMulNV",            EOpSubgroupPartitionedExclusiveMul);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveMinNV",            EOpSubgroupPartitionedExclusiveMin);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveMaxNV",            EOpSubgroupPartitionedExclusiveMax);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveAndNV",            EOpSubgroupPartitionedExclusiveAnd);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveOrNV",             EOpSubgroupPartitionedExclusiveOr);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveXorNV",            EOpSubgroupPartitionedExclusiveXor);
        }

        if (profile == EEsProfile) {
            symbolTable.relateToOperator("shadow2DEXT",              EOpTexture);
            symbolTable.relateToOperator("shadow2DProjEXT",          EOpTextureProj);
        }

        // GL_EXT_shader_quad_control
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.relateToOperator("subgroupQuadAll",                     EOpSubgroupQuadAll);
            symbolTable.relateToOperator("subgroupQuadAny",                     EOpSubgroupQuadAny);
        }

        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.relateToOperator("textureWeightedQCOM",      EOpImageSampleWeightedQCOM);
            symbolTable.relateToOperator("textureBoxFilterQCOM",     EOpImageBoxFilterQCOM);
            symbolTable.relateToOperator("textureBlockMatchSADQCOM", EOpImageBlockMatchSADQCOM);
            symbolTable.relateToOperator("textureBlockMatchSSDQCOM", EOpImageBlockMatchSSDQCOM);

            symbolTable.relateToOperator("textureBlockMatchWindowSSDQCOM", EOpImageBlockMatchWindowSSDQCOM);
            symbolTable.relateToOperator("textureBlockMatchWindowSADQCOM", EOpImageBlockMatchWindowSADQCOM);
            symbolTable.relateToOperator("textureBlockMatchGatherSSDQCOM", EOpImageBlockMatchGatherSSDQCOM);
            symbolTable.relateToOperator("textureBlockMatchGatherSADQCOM", EOpImageBlockMatchGatherSADQCOM);
        }

        if (profile != EEsProfile && spvVersion.spv == 0) {
            symbolTable.relateToOperator("texture1DArray", EOpTexture);
            symbolTable.relateToOperator("texture2DArray", EOpTexture);
            symbolTable.relateToOperator("shadow1DArray", EOpTexture);
            symbolTable.relateToOperator("shadow2DArray", EOpTexture);

            symbolTable.relateToOperator("texture1DArrayLod", EOpTextureLod);
            symbolTable.relateToOperator("texture2DArrayLod", EOpTextureLod);
            symbolTable.relateToOperator("shadow1DArrayLod", EOpTextureLod);
        }
    }

    switch(language) {
    case EShLangVertex:
        break;

    case EShLangTessControl:
    case EShLangTessEvaluation:
        break;

    case EShLangGeometry:
        symbolTable.relateToOperator("EmitStreamVertex",   EOpEmitStreamVertex);
        symbolTable.relateToOperator("EndStreamPrimitive", EOpEndStreamPrimitive);
        symbolTable.relateToOperator("EmitVertex",         EOpEmitVertex);
        symbolTable.relateToOperator("EndPrimitive",       EOpEndPrimitive);
        break;

    case EShLangFragment:
        if (profile != EEsProfile && version >= 400) {
            symbolTable.relateToOperator("dFdxFine",     EOpDPdxFine);
            symbolTable.relateToOperator("dFdyFine",     EOpDPdyFine);
            symbolTable.relateToOperator("fwidthFine",   EOpFwidthFine);
            symbolTable.relateToOperator("dFdxCoarse",   EOpDPdxCoarse);
            symbolTable.relateToOperator("dFdyCoarse",   EOpDPdyCoarse);
            symbolTable.relateToOperator("fwidthCoarse", EOpFwidthCoarse);
        }

        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("rayQueryInitializeEXT",                                             EOpRayQueryInitialize);
            symbolTable.relateToOperator("rayQueryTerminateEXT",                                              EOpRayQueryTerminate);
            symbolTable.relateToOperator("rayQueryGenerateIntersectionEXT",                                   EOpRayQueryGenerateIntersection);
            symbolTable.relateToOperator("rayQueryConfirmIntersectionEXT",                                    EOpRayQueryConfirmIntersection);
            symbolTable.relateToOperator("rayQueryProceedEXT",                                                EOpRayQueryProceed);
            symbolTable.relateToOperator("rayQueryGetIntersectionTypeEXT",                                    EOpRayQueryGetIntersectionType);
            symbolTable.relateToOperator("rayQueryGetRayTMinEXT",                                             EOpRayQueryGetRayTMin);
            symbolTable.relateToOperator("rayQueryGetRayFlagsEXT",                                            EOpRayQueryGetRayFlags);
            symbolTable.relateToOperator("rayQueryGetIntersectionTEXT",                                       EOpRayQueryGetIntersectionT);
            symbolTable.relateToOperator("rayQueryGetIntersectionInstanceCustomIndexEXT",                     EOpRayQueryGetIntersectionInstanceCustomIndex);
            symbolTable.relateToOperator("rayQueryGetIntersectionInstanceIdEXT",                              EOpRayQueryGetIntersectionInstanceId);
            symbolTable.relateToOperator("rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT",  EOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffset);
            symbolTable.relateToOperator("rayQueryGetIntersectionGeometryIndexEXT",                           EOpRayQueryGetIntersectionGeometryIndex);
            symbolTable.relateToOperator("rayQueryGetIntersectionPrimitiveIndexEXT",                          EOpRayQueryGetIntersectionPrimitiveIndex);
            symbolTable.relateToOperator("rayQueryGetIntersectionBarycentricsEXT",                            EOpRayQueryGetIntersectionBarycentrics);
            symbolTable.relateToOperator("rayQueryGetIntersectionFrontFaceEXT",                               EOpRayQueryGetIntersectionFrontFace);
            symbolTable.relateToOperator("rayQueryGetIntersectionCandidateAABBOpaqueEXT",                     EOpRayQueryGetIntersectionCandidateAABBOpaque);
            symbolTable.relateToOperator("rayQueryGetIntersectionObjectRayDirectionEXT",                      EOpRayQueryGetIntersectionObjectRayDirection);
            symbolTable.relateToOperator("rayQueryGetIntersectionObjectRayOriginEXT",                         EOpRayQueryGetIntersectionObjectRayOrigin);
            symbolTable.relateToOperator("rayQueryGetWorldRayDirectionEXT",                                   EOpRayQueryGetWorldRayDirection);
            symbolTable.relateToOperator("rayQueryGetWorldRayOriginEXT",                                      EOpRayQueryGetWorldRayOrigin);
            symbolTable.relateToOperator("rayQueryGetIntersectionObjectToWorldEXT",                           EOpRayQueryGetIntersectionObjectToWorld);
            symbolTable.relateToOperator("rayQueryGetIntersectionWorldToObjectEXT",                           EOpRayQueryGetIntersectionWorldToObject);
            symbolTable.relateToOperator("rayQueryGetIntersectionTriangleVertexPositionsEXT",                 EOpRayQueryGetIntersectionTriangleVertexPositionsEXT);
        }

        symbolTable.relateToOperator("interpolateAtCentroid", EOpInterpolateAtCentroid);
        symbolTable.relateToOperator("interpolateAtSample",   EOpInterpolateAtSample);
        symbolTable.relateToOperator("interpolateAtOffset",   EOpInterpolateAtOffset);

        if (profile != EEsProfile)
            symbolTable.relateToOperator("interpolateAtVertexAMD", EOpInterpolateAtVertex);

        symbolTable.relateToOperator("beginInvocationInterlockARB", EOpBeginInvocationInterlock);
        symbolTable.relateToOperator("endInvocationInterlockARB",   EOpEndInvocationInterlock);

        symbolTable.relateToOperator("stencilAttachmentReadEXT", EOpStencilAttachmentReadEXT);
        symbolTable.relateToOperator("depthAttachmentReadEXT",   EOpDepthAttachmentReadEXT);
        symbolTable.relateToOperator("colorAttachmentReadEXT",   EOpColorAttachmentReadEXT);

        break;

    case EShLangCompute:
        symbolTable.relateToOperator("subgroupMemoryBarrierShared", EOpSubgroupMemoryBarrierShared);
        if ((profile != EEsProfile && version >= 450) ||
            (profile == EEsProfile && version >= 320)) {
            symbolTable.relateToOperator("dFdx",        EOpDPdx);
            symbolTable.relateToOperator("dFdy",        EOpDPdy);
            symbolTable.relateToOperator("fwidth",      EOpFwidth);
            symbolTable.relateToOperator("dFdxFine",    EOpDPdxFine);
            symbolTable.relateToOperator("dFdyFine",    EOpDPdyFine);
            symbolTable.relateToOperator("fwidthFine",  EOpFwidthFine);
            symbolTable.relateToOperator("dFdxCoarse",  EOpDPdxCoarse);
            symbolTable.relateToOperator("dFdyCoarse",  EOpDPdyCoarse);
            symbolTable.relateToOperator("fwidthCoarse",EOpFwidthCoarse);
        }
        symbolTable.relateToOperator("coopMatLoadNV",              EOpCooperativeMatrixLoadNV);
        symbolTable.relateToOperator("coopMatStoreNV",             EOpCooperativeMatrixStoreNV);
        symbolTable.relateToOperator("coopMatMulAddNV",            EOpCooperativeMatrixMulAddNV);

        symbolTable.relateToOperator("coopMatLoad",                EOpCooperativeMatrixLoad);
        symbolTable.relateToOperator("coopMatStore",               EOpCooperativeMatrixStore);
        symbolTable.relateToOperator("coopMatMulAdd",              EOpCooperativeMatrixMulAdd);

        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("fetchMicroTriangleVertexPositionNV", EOpFetchMicroTriangleVertexPositionNV);
            symbolTable.relateToOperator("fetchMicroTriangleVertexBarycentricNV", EOpFetchMicroTriangleVertexBarycentricNV);
        }
        break;

    case EShLangRayGen:
        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("fetchMicroTriangleVertexPositionNV", EOpFetchMicroTriangleVertexPositionNV);
            symbolTable.relateToOperator("fetchMicroTriangleVertexBarycentricNV", EOpFetchMicroTriangleVertexBarycentricNV);
        }
        [[fallthrough]];
    case EShLangClosestHit:
    case EShLangMiss:
        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("traceNV", EOpTraceNV);
            symbolTable.relateToOperator("traceRayMotionNV", EOpTraceRayMotionNV);
            symbolTable.relateToOperator("traceRayEXT", EOpTraceKHR);
            symbolTable.relateToOperator("executeCallableNV", EOpExecuteCallableNV);
            symbolTable.relateToOperator("executeCallableEXT", EOpExecuteCallableKHR);

            symbolTable.relateToOperator("hitObjectTraceRayNV", EOpHitObjectTraceRayNV);
            symbolTable.relateToOperator("hitObjectTraceRayMotionNV", EOpHitObjectTraceRayMotionNV);
            symbolTable.relateToOperator("hitObjectRecordHitNV", EOpHitObjectRecordHitNV);
            symbolTable.relateToOperator("hitObjectRecordHitMotionNV", EOpHitObjectRecordHitMotionNV);
            symbolTable.relateToOperator("hitObjectRecordHitWithIndexNV", EOpHitObjectRecordHitWithIndexNV);
            symbolTable.relateToOperator("hitObjectRecordHitWithIndexMotionNV", EOpHitObjectRecordHitWithIndexMotionNV);
            symbolTable.relateToOperator("hitObjectRecordMissNV", EOpHitObjectRecordMissNV);
            symbolTable.relateToOperator("hitObjectRecordMissMotionNV", EOpHitObjectRecordMissMotionNV);
            symbolTable.relateToOperator("hitObjectRecordEmptyNV", EOpHitObjectRecordEmptyNV);
            symbolTable.relateToOperator("hitObjectExecuteShaderNV", EOpHitObjectExecuteShaderNV);
            symbolTable.relateToOperator("hitObjectIsEmptyNV", EOpHitObjectIsEmptyNV);
            symbolTable.relateToOperator("hitObjectIsMissNV", EOpHitObjectIsMissNV);
            symbolTable.relateToOperator("hitObjectIsHitNV", EOpHitObjectIsHitNV);
            symbolTable.relateToOperator("hitObjectGetRayTMinNV", EOpHitObjectGetRayTMinNV);
            symbolTable.relateToOperator("hitObjectGetRayTMaxNV", EOpHitObjectGetRayTMaxNV);
            symbolTable.relateToOperator("hitObjectGetObjectRayOriginNV", EOpHitObjectGetObjectRayOriginNV);
            symbolTable.relateToOperator("hitObjectGetObjectRayDirectionNV", EOpHitObjectGetObjectRayDirectionNV);
            symbolTable.relateToOperator("hitObjectGetWorldRayOriginNV", EOpHitObjectGetWorldRayOriginNV);
            symbolTable.relateToOperator("hitObjectGetWorldRayDirectionNV", EOpHitObjectGetWorldRayDirectionNV);
            symbolTable.relateToOperator("hitObjectGetWorldToObjectNV", EOpHitObjectGetWorldToObjectNV);
            symbolTable.relateToOperator("hitObjectGetObjectToWorldNV", EOpHitObjectGetObjectToWorldNV);
            symbolTable.relateToOperator("hitObjectGetInstanceCustomIndexNV", EOpHitObjectGetInstanceCustomIndexNV);
            symbolTable.relateToOperator("hitObjectGetInstanceIdNV", EOpHitObjectGetInstanceIdNV);
            symbolTable.relateToOperator("hitObjectGetGeometryIndexNV", EOpHitObjectGetGeometryIndexNV);
            symbolTable.relateToOperator("hitObjectGetPrimitiveIndexNV", EOpHitObjectGetPrimitiveIndexNV);
            symbolTable.relateToOperator("hitObjectGetHitKindNV", EOpHitObjectGetHitKindNV);
            symbolTable.relateToOperator("hitObjectGetAttributesNV", EOpHitObjectGetAttributesNV);
            symbolTable.relateToOperator("hitObjectGetCurrentTimeNV", EOpHitObjectGetCurrentTimeNV);
            symbolTable.relateToOperator("hitObjectGetShaderBindingTableRecordIndexNV", EOpHitObjectGetShaderBindingTableRecordIndexNV);
            symbolTable.relateToOperator("hitObjectGetShaderRecordBufferHandleNV", EOpHitObjectGetShaderRecordBufferHandleNV);
            symbolTable.relateToOperator("reorderThreadNV", EOpReorderThreadNV);
        }
        break;
    case EShLangIntersect:
        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("reportIntersectionNV", EOpReportIntersection);
            symbolTable.relateToOperator("reportIntersectionEXT", EOpReportIntersection);
        }
        break;
    case EShLangAnyHit:
        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("ignoreIntersectionNV", EOpIgnoreIntersectionNV);
            symbolTable.relateToOperator("terminateRayNV", EOpTerminateRayNV);
        }
        break;
    case EShLangCallable:
        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("executeCallableNV", EOpExecuteCallableNV);
            symbolTable.relateToOperator("executeCallableEXT", EOpExecuteCallableKHR);
        }
        break;
    case EShLangMesh:
        if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
            symbolTable.relateToOperator("writePackedPrimitiveIndices4x8NV", EOpWritePackedPrimitiveIndices4x8NV);
            symbolTable.relateToOperator("memoryBarrierShared", EOpMemoryBarrierShared);
            symbolTable.relateToOperator("groupMemoryBarrier", EOpGroupMemoryBarrier);
            symbolTable.relateToOperator("subgroupMemoryBarrierShared", EOpSubgroupMemoryBarrierShared);
        }

        if (profile != EEsProfile && version >= 450) {
            symbolTable.relateToOperator("SetMeshOutputsEXT", EOpSetMeshOutputsEXT);
        }

        if (profile != EEsProfile && version >= 460) {
            // Builtins for GL_NV_displacement_micromap.
            symbolTable.relateToOperator("fetchMicroTriangleVertexPositionNV", EOpFetchMicroTriangleVertexPositionNV);
            symbolTable.relateToOperator("fetchMicroTriangleVertexBarycentricNV", EOpFetchMicroTriangleVertexBarycentricNV);
        }
        break;
    case EShLangTask:
        if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
            symbolTable.relateToOperator("memoryBarrierShared", EOpMemoryBarrierShared);
            symbolTable.relateToOperator("groupMemoryBarrier", EOpGroupMemoryBarrier);
            symbolTable.relateToOperator("subgroupMemoryBarrierShared", EOpSubgroupMemoryBarrierShared);
        }
        if (profile != EEsProfile && version >= 450) {
            symbolTable.relateToOperator("EmitMeshTasksEXT", EOpEmitMeshTasksEXT);
        }
        break;

    default:
        assert(false && "Language not supported");
    }
}

//
// Add context-dependent (resource-specific) built-ins not handled by the above.  These
// would be ones that need to be programmatically added because they cannot
// be added by simple text strings.  For these, also
// 1) Map built-in functions to operators, for those that will turn into an operation node
//    instead of remaining a function call.
// 2) Tag extension-related symbols added to their base version with their extensions, so
//    that if an early version has the extension turned off, there is an error reported on use.
//
void TBuiltIns::identifyBuiltIns(int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language, TSymbolTable& symbolTable, const TBuiltInResource &resources)
{
    if (profile != EEsProfile && version >= 430 && version < 440) {
        symbolTable.setVariableExtensions("gl_MaxTransformFeedbackBuffers", 1, &E_GL_ARB_enhanced_layouts);
        symbolTable.setVariableExtensions("gl_MaxTransformFeedbackInterleavedComponents", 1, &E_GL_ARB_enhanced_layouts);
    }
    if (profile != EEsProfile && version >= 130 && version < 420) {
        symbolTable.setVariableExtensions("gl_MinProgramTexelOffset", 1, &E_GL_ARB_shading_language_420pack);
        symbolTable.setVariableExtensions("gl_MaxProgramTexelOffset", 1, &E_GL_ARB_shading_language_420pack);
    }
    if (profile != EEsProfile && version >= 150 && version < 410)
        symbolTable.setVariableExtensions("gl_MaxViewports", 1, &E_GL_ARB_viewport_array);

    switch(language) {
    case EShLangFragment:
        // Set up gl_FragData based on current array size.
        if (version == 100 || IncludeLegacy(version, profile, spvVersion) || (! ForwardCompatibility && profile != EEsProfile && version < 420)) {
            TPrecisionQualifier pq = profile == EEsProfile ? EpqMedium : EpqNone;
            TType fragData(EbtFloat, EvqFragColor, pq, 4);
            TArraySizes* arraySizes = new TArraySizes;
            arraySizes->addInnerSize(resources.maxDrawBuffers);
            fragData.transferArraySizes(arraySizes);
            symbolTable.insert(*new TVariable(NewPoolTString("gl_FragData"), fragData));
            SpecialQualifier("gl_FragData", EvqFragColor, EbvFragData, symbolTable);
        }

        // GL_EXT_blend_func_extended
        if (profile == EEsProfile && version >= 100) {
           symbolTable.setVariableExtensions("gl_MaxDualSourceDrawBuffersEXT",    1, &E_GL_EXT_blend_func_extended);
           symbolTable.setVariableExtensions("gl_SecondaryFragColorEXT",    1, &E_GL_EXT_blend_func_extended);
           symbolTable.setVariableExtensions("gl_SecondaryFragDataEXT",    1, &E_GL_EXT_blend_func_extended);
           SpecialQualifier("gl_SecondaryFragColorEXT", EvqVaryingOut, EbvSecondaryFragColorEXT, symbolTable);
           SpecialQualifier("gl_SecondaryFragDataEXT", EvqVaryingOut, EbvSecondaryFragDataEXT, symbolTable);
        }

        break;

    case EShLangTessControl:
    case EShLangTessEvaluation:
        // Because of the context-dependent array size (gl_MaxPatchVertices),
        // these variables were added later than the others and need to be mapped now.

        // standard members
        BuiltInVariable("gl_in", "gl_Position",     EbvPosition,     symbolTable);
        BuiltInVariable("gl_in", "gl_PointSize",    EbvPointSize,    symbolTable);
        BuiltInVariable("gl_in", "gl_ClipDistance", EbvClipDistance, symbolTable);
        BuiltInVariable("gl_in", "gl_CullDistance", EbvCullDistance, symbolTable);

        // compatibility members
        BuiltInVariable("gl_in", "gl_ClipVertex",          EbvClipVertex,          symbolTable);
        BuiltInVariable("gl_in", "gl_FrontColor",          EbvFrontColor,          symbolTable);
        BuiltInVariable("gl_in", "gl_BackColor",           EbvBackColor,           symbolTable);
        BuiltInVariable("gl_in", "gl_FrontSecondaryColor", EbvFrontSecondaryColor, symbolTable);
        BuiltInVariable("gl_in", "gl_BackSecondaryColor",  EbvBackSecondaryColor,  symbolTable);
        BuiltInVariable("gl_in", "gl_TexCoord",            EbvTexCoord,            symbolTable);
        BuiltInVariable("gl_in", "gl_FogFragCoord",        EbvFogFragCoord,        symbolTable);

        symbolTable.setVariableExtensions("gl_in", "gl_SecondaryPositionNV", 1, &E_GL_NV_stereo_view_rendering);
        symbolTable.setVariableExtensions("gl_in", "gl_PositionPerViewNV",   1, &E_GL_NVX_multiview_per_view_attributes);

        BuiltInVariable("gl_in", "gl_SecondaryPositionNV", EbvSecondaryPositionNV, symbolTable);
        BuiltInVariable("gl_in", "gl_PositionPerViewNV",   EbvPositionPerViewNV,   symbolTable);

        // extension requirements
        if (profile == EEsProfile) {
            symbolTable.setVariableExtensions("gl_in", "gl_PointSize", Num_AEP_tessellation_point_size, AEP_tessellation_point_size);
        }

        break;

    default:
        break;
    }
}

} // end namespace glslang
