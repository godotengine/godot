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

//
// Create strings that declare built-in definitions, add built-ins programmatically
// that cannot be expressed in the strings, and establish mappings between
// built-in functions and operators.
//
// Where to put a built-in:
//   TBuiltInParseablesHlsl::initialize(version,profile) context-independent textual built-ins; add them to the right string
//   TBuiltInParseablesHlsl::initialize(resources,...)   context-dependent textual built-ins; add them to the right string
//   TBuiltInParseablesHlsl::identifyBuiltIns(...,symbolTable) context-independent programmatic additions/mappings to the symbol table,
//                                                including identifying what extensions are needed if a version does not allow a symbol
//   TBuiltInParseablesHlsl::identifyBuiltIns(...,symbolTable, resources) context-dependent programmatic additions/mappings to the
//                                                symbol table, including identifying what extensions are needed if a version does
//                                                not allow a symbol
//

#include "hlslParseables.h"
#include "hlslParseHelper.h"
#include <cctype>
#include <utility>
#include <algorithm>

namespace {  // anonymous namespace functions

const bool UseHlslTypes = true;

const char* BaseTypeName(const char argOrder, const char* scalarName, const char* vecName, const char* matName)
{
    switch (argOrder) {
    case 'S': return scalarName;
    case 'V': return vecName;
    case 'M': return matName;
    default:  return "UNKNOWN_TYPE";
    }
}

// arg order queries
bool IsSamplerType(const char argType)     { return argType == 'S' || argType == 's'; }
bool IsArrayed(const char argOrder)        { return argOrder == '@' || argOrder == '&' || argOrder == '#'; }
bool IsTextureNonMS(const char argOrder)   { return argOrder == '%'; }
bool IsSubpassInput(const char argOrder)   { return argOrder == '[' || argOrder == ']'; }
bool IsArrayedTexture(const char argOrder) { return argOrder == '@'; }
bool IsTextureMS(const char argOrder)      { return argOrder == '$' || argOrder == '&'; }
bool IsMS(const char argOrder)             { return IsTextureMS(argOrder) || argOrder == ']'; }
bool IsBuffer(const char argOrder)         { return argOrder == '*' || argOrder == '~'; }
bool IsImage(const char argOrder)          { return argOrder == '!' || argOrder == '#' || argOrder == '~'; }

bool IsTextureType(const char argOrder)
{
    return IsTextureNonMS(argOrder) || IsArrayedTexture(argOrder) ||
           IsTextureMS(argOrder) || IsBuffer(argOrder) || IsImage(argOrder);
}

// Reject certain combinations that are illegal sample methods.  For example,
// 3D arrays.
bool IsIllegalSample(const glslang::TString& name, const char* argOrder, int dim0)
{
    const bool isArrayed = IsArrayed(*argOrder);
    const bool isMS      = IsTextureMS(*argOrder);
    const bool isBuffer  = IsBuffer(*argOrder);

    // there are no 3D arrayed textures, or 3D SampleCmp(LevelZero)
    if (dim0 == 3 && (isArrayed || name == "SampleCmp" || name == "SampleCmpLevelZero"))
        return true;

    const int numArgs = int(std::count(argOrder, argOrder + strlen(argOrder), ',')) + 1;

    // Reject invalid offset forms with cubemaps
    if (dim0 == 4) {
        if ((name == "Sample"             && numArgs >= 4) ||
            (name == "SampleBias"         && numArgs >= 5) ||
            (name == "SampleCmp"          && numArgs >= 5) ||
            (name == "SampleCmpLevelZero" && numArgs >= 5) ||
            (name == "SampleGrad"         && numArgs >= 6) ||
            (name == "SampleLevel"        && numArgs >= 5))
            return true;
    }

    const bool isGather =
        (name == "Gather" ||
         name == "GatherRed" ||
         name == "GatherGreen" ||
         name == "GatherBlue"  ||
         name == "GatherAlpha");

    const bool isGatherCmp =
        (name == "GatherCmp"      ||
         name == "GatherCmpRed"   ||
         name == "GatherCmpGreen" ||
         name == "GatherCmpBlue"  ||
         name == "GatherCmpAlpha");

    // Reject invalid Gathers
    if (isGather || isGatherCmp) {
        if (dim0 == 1 || dim0 == 3)   // there are no 1D or 3D gathers
            return true;

        // no offset on cube or cube array gathers
        if (dim0 == 4) {
            if ((isGather && numArgs > 3) || (isGatherCmp && numArgs > 4))
                return true;
        }
    }

    // Reject invalid Loads
    if (name == "Load" && dim0 == 4)
        return true; // Load does not support any cubemaps, arrayed or not.

    // Multisample formats are only 2D and 2Darray
    if (isMS && dim0 != 2)
        return true;

    // Buffer are only 1D
    if (isBuffer && dim0 != 1)
        return true;

    return false;
}

// Return the number of the coordinate arg, if any
int CoordinateArgPos(const glslang::TString& name, bool isTexture)
{
    if (!isTexture || (name == "GetDimensions"))
        return -1;  // has none
    else if (name == "Load")
        return 1;
    else
        return 2;  // other texture methods are 2
}

// Some texture methods use an addition coordinate dimension for the mip
bool HasMipInCoord(const glslang::TString& name, bool isMS, bool isBuffer, bool isImage)
{
    return name == "Load" && !isMS && !isBuffer && !isImage;
}

// LOD calculations don't pass the array level in the coordinate.
bool NoArrayCoord(const glslang::TString& name)
{
    return name == "CalculateLevelOfDetail" || name == "CalculateLevelOfDetailUnclamped";
}

// Handle IO params marked with > or <
const char* IoParam(glslang::TString& s, const char* nthArgOrder)
{
    if (*nthArgOrder == '>') {           // output params
        ++nthArgOrder;
        s.append("out ");
    } else if (*nthArgOrder == '<') {    // input params
        ++nthArgOrder;
        s.append("in ");
    }

    return nthArgOrder;
}

// Handle repeated args
void HandleRepeatArg(const char*& arg, const char*& prev, const char* current)
{
    if (*arg == ',' || *arg == '\0')
        arg = prev;
    else
        prev = current;
}

// Return true for the end of a single argument key, which can be the end of the string, or
// the comma separator.
inline bool IsEndOfArg(const char* arg)
{
    return arg == nullptr || *arg == '\0' || *arg == ',';
}

// If this is a fixed vector size, such as V3, return the size.  Else return 0.
int FixedVecSize(const char* arg)
{
    while (!IsEndOfArg(arg)) {
        if (isdigit(*arg))
            return *arg - '0';
        ++arg;
    }

    return 0; // none found.
}

// Create and return a type name.  This is done in GLSL, not HLSL conventions, until such
// time as builtins are parsed using the HLSL parser.
//
//    order:   S = scalar, V = vector, M = matrix
//    argType: F = float, D = double, I = int, U = uint, B = bool, S = sampler
//    dim0 = vector dimension, or matrix 1st dimension
//    dim1 = matrix 2nd dimension
glslang::TString& AppendTypeName(glslang::TString& s, const char* argOrder, const char* argType, int dim0, int dim1)
{
    const bool isTranspose = (argOrder[0] == '^');
    const bool isTexture   = IsTextureType(argOrder[0]);
    const bool isArrayed   = IsArrayed(argOrder[0]);
    const bool isSampler   = IsSamplerType(argType[0]);
    const bool isMS        = IsMS(argOrder[0]);
    const bool isBuffer    = IsBuffer(argOrder[0]);
    const bool isImage     = IsImage(argOrder[0]);
    const bool isSubpass   = IsSubpassInput(argOrder[0]);

    char type  = *argType;

    if (isTranspose) {  // Take transpose of matrix dimensions
        std::swap(dim0, dim1);
    } else if (isTexture || isSubpass) {
        if (type == 'F')       // map base type to texture of that type.
            type = 'T';        // e.g, int -> itexture, uint -> utexture, etc.
        else if (type == 'I')
            type = 'i';
        else if (type == 'U')
            type = 'u';
    }

    if (isTranspose)
        ++argOrder;

    char order = *argOrder;

    if (UseHlslTypes) {
        switch (type) {
        case '-': s += "void";                                break;
        case 'F': s += "float";                               break;
        case 'D': s += "double";                              break;
        case 'I': s += "int";                                 break;
        case 'U': s += "uint";                                break;
        case 'L': s += "int64_t";                             break;
        case 'M': s += "uint64_t";                            break;
        case 'B': s += "bool";                                break;
        case 'S': s += "sampler";                             break;
        case 's': s += "SamplerComparisonState";              break;
        case 'T': s += ((isBuffer && isImage) ? "RWBuffer" :
                        isSubpass ? "SubpassInput" :
                        isBuffer ? "Buffer" :
                        isImage  ? "RWTexture" : "Texture");  break;
        case 'i': s += ((isBuffer && isImage) ? "RWBuffer" :
                        isSubpass ? "SubpassInput" :
                        isBuffer ? "Buffer" :
                        isImage ? "RWTexture" : "Texture");   break;
        case 'u': s += ((isBuffer && isImage) ? "RWBuffer" :
                        isSubpass ? "SubpassInput" :
                        isBuffer ? "Buffer" :
                        isImage ? "RWTexture" : "Texture");   break;
        default:  s += "UNKNOWN_TYPE";                        break;
        }

        if (isSubpass && isMS)
            s += "MS";

    } else {
        switch (type) {
        case '-': s += "void"; break;
        case 'F': s += BaseTypeName(order, "float",  "vec",  "mat");  break;
        case 'D': s += BaseTypeName(order, "double", "dvec", "dmat"); break;
        case 'I': s += BaseTypeName(order, "int",    "ivec", "imat"); break;
        case 'U': s += BaseTypeName(order, "uint",   "uvec", "umat"); break;
        case 'B': s += BaseTypeName(order, "bool",   "bvec", "bmat"); break;
        case 'S': s += "sampler";                                     break;
        case 's': s += "samplerShadow";                               break;
        case 'T': // fall through
        case 'i': // ...
        case 'u': // ...
            if (type != 'T') // create itexture, utexture, etc
                s += type;

            s += ((isImage && isBuffer) ? "imageBuffer"   :
                  isSubpass             ? "subpassInput" :
                  isImage               ? "image"         :
                  isBuffer              ? "samplerBuffer" :
                  "texture");
            break;

        default:  s += "UNKNOWN_TYPE"; break;
        }
    }

    // handle fixed vector sizes, such as float3, and only ever 3.
    const int fixedVecSize = FixedVecSize(argOrder);
    if (fixedVecSize != 0)
        dim0 = dim1 = fixedVecSize;

    const char dim0Char = ('0' + char(dim0));
    const char dim1Char = ('0' + char(dim1));

    // Add sampler dimensions
    if (isSampler || isTexture) {
        if ((order == 'V' || isTexture) && !isBuffer) {
            switch (dim0) {
            case 1: s += "1D";                   break;
            case 2: s += (isMS ? "2DMS" : "2D"); break;
            case 3: s += "3D";                   break;
            case 4: s += "Cube";                 break;
            default: s += "UNKNOWN_SAMPLER";     break;
            }
        }
    } else {
        // Non-sampler type:
        // verify dimensions
        if (((order == 'V' || order == 'M') && (dim0 < 1 || dim0 > 4)) ||
            (order == 'M' && (dim1 < 1 || dim1 > 4))) {
            s += "UNKNOWN_DIMENSION";
            return s;
        }

        switch (order) {
        case '-': break;  // no dimensions for voids
        case 'S': break;  // no dimensions on scalars
        case 'V':
            s += dim0Char;
            break;
        case 'M':
            s += dim0Char;
            s += 'x';
            s += dim1Char;
            break;
        default:
            break;
        }
    }

    // handle arrayed textures
    if (isArrayed)
        s += "Array";

    // For HLSL, append return type for texture types
    if (UseHlslTypes) {
        switch (type) {
        case 'i': s += "<int";   s += dim0Char; s += ">"; break;
        case 'u': s += "<uint";  s += dim0Char; s += ">"; break;
        case 'T': s += "<float"; s += dim0Char; s += ">"; break;
        default: break;
        }
    }

    return s;
}

// The GLSL parser can be used to parse a subset of HLSL prototypes.  However, many valid HLSL prototypes
// are not valid GLSL prototypes.  This rejects the invalid ones.  Thus, there is a single switch below
// to enable creation of the entire HLSL space.
inline bool IsValid(const char* cname, char retOrder, char retType, char argOrder, char argType, int dim0, int dim1)
{
    const bool isVec = (argOrder == 'V');
    const bool isMat = (argOrder == 'M');

    const std::string name(cname);

    // these do not have vec1 versions
    if (dim0 == 1 && (name == "normalize" || name == "reflect" || name == "refract"))
        return false;

    if (!IsTextureType(argOrder) && (isVec && dim0 == 1)) // avoid vec1
        return false;

    if (UseHlslTypes) {
        // NO further restrictions for HLSL
    } else {
        // GLSL parser restrictions
        if ((isMat && (argType == 'I' || argType == 'U' || argType == 'B')) ||
            (retOrder == 'M' && (retType == 'I' || retType == 'U' || retType == 'B')))
            return false;

        if (isMat && dim0 == 1 && dim1 == 1)  // avoid mat1x1
            return false;

        if (isMat && dim1 == 1)  // TODO: avoid mat Nx1 until we find the right GLSL profile
            return false;

        if (name == "GetRenderTargetSamplePosition" ||
            name == "tex1D" ||
            name == "tex1Dgrad")
            return false;
    }

    return true;
}

// return position of end of argument specifier
inline const char* FindEndOfArg(const char* arg)
{
    while (!IsEndOfArg(arg))
        ++arg;

    return *arg == '\0' ? nullptr : arg;
}

// Return pointer to beginning of Nth argument specifier in the string.
inline const char* NthArg(const char* arg, int n)
{
    for (int x=0; x<n && arg; ++x)
        if ((arg = FindEndOfArg(arg)) != nullptr)
            ++arg;  // skip arg separator

    return arg;
}

inline void FindVectorMatrixBounds(const char* argOrder, int fixedVecSize, int& dim0Min, int& dim0Max, int& /*dim1Min*/, int& dim1Max)
{
    for (int arg = 0; ; ++arg) {
        const char* nthArgOrder(NthArg(argOrder, arg));
        if (nthArgOrder == nullptr)
            break;
        else if (*nthArgOrder == 'V' || IsSubpassInput(*nthArgOrder))
            dim0Max = 4;
        else if (*nthArgOrder == 'M')
            dim0Max = dim1Max = 4;
    }

    if (fixedVecSize > 0) // handle fixed sized vectors
        dim0Min = dim0Max = fixedVecSize;
}

} // end anonymous namespace

namespace glslang {

TBuiltInParseablesHlsl::TBuiltInParseablesHlsl()
{
}

//
// Handle creation of mat*mat specially, since it doesn't fall conveniently out of
// the generic prototype creation code below.
//
void TBuiltInParseablesHlsl::createMatTimesMat()
{
    TString& s = commonBuiltins;

    const int first = (UseHlslTypes ? 1 : 2);

    for (int xRows = first; xRows <=4; xRows++) {
        for (int xCols = first; xCols <=4; xCols++) {
            const int yRows = xCols;
            for (int yCols = first; yCols <=4; yCols++) {
                const int retRows = xRows;
                const int retCols = yCols;

                // Create a mat * mat of the appropriate dimensions
                AppendTypeName(s, "M", "F", retRows, retCols);  // add return type
                s.append(" ");                                  // space between type and name
                s.append("mul");                                // intrinsic name
                s.append("(");                                  // open paren

                AppendTypeName(s, "M", "F", xRows, xCols);      // add X input
                s.append(", ");
                AppendTypeName(s, "M", "F", yRows, yCols);      // add Y input

                s.append(");\n");                               // close paren
            }

            // Create M*V
            AppendTypeName(s, "V", "F", xRows, 1);          // add return type
            s.append(" ");                                  // space between type and name
            s.append("mul");                                // intrinsic name
            s.append("(");                                  // open paren

            AppendTypeName(s, "M", "F", xRows, xCols);      // add X input
            s.append(", ");
            AppendTypeName(s, "V", "F", xCols, 1);          // add Y input

            s.append(");\n");                               // close paren

            // Create V*M
            AppendTypeName(s, "V", "F", xCols, 1);          // add return type
            s.append(" ");                                  // space between type and name
            s.append("mul");                                // intrinsic name
            s.append("(");                                  // open paren

            AppendTypeName(s, "V", "F", xRows, 1);          // add Y input
            s.append(", ");
            AppendTypeName(s, "M", "F", xRows, xCols);      // add X input

            s.append(");\n");                               // close paren
        }
    }
}

//
// Add all context-independent built-in functions and variables that are present
// for the given version and profile.  Share common ones across stages, otherwise
// make stage-specific entries.
//
// Most built-ins variables can be added as simple text strings.  Some need to
// be added programmatically, which is done later in IdentifyBuiltIns() below.
//
void TBuiltInParseablesHlsl::initialize(int /*version*/, EProfile /*profile*/, const SpvVersion& /*spvVersion*/)
{
    static const EShLanguageMask EShLangAll    = EShLanguageMask(EShLangCount - 1);

    // These are the actual stage masks defined in the documentation, in case they are
    // needed for future validation.  For now, they are commented out, and set below
    // to EShLangAll, to allow any intrinsic to be used in any shader, which is legal
    // if it is not called.
    //
    // static const EShLanguageMask EShLangPSCS   = EShLanguageMask(EShLangFragmentMask | EShLangComputeMask);
    // static const EShLanguageMask EShLangVSPSGS = EShLanguageMask(EShLangVertexMask | EShLangFragmentMask | EShLangGeometryMask);
    // static const EShLanguageMask EShLangCS     = EShLangComputeMask;
    // static const EShLanguageMask EShLangPS     = EShLangFragmentMask;
    // static const EShLanguageMask EShLangHS     = EShLangTessControlMask;

    // This set uses EShLangAll for everything.
    static const EShLanguageMask EShLangPSCS   = EShLangAll;
    static const EShLanguageMask EShLangVSPSGS = EShLangAll;
    static const EShLanguageMask EShLangCS     = EShLangAll;
    static const EShLanguageMask EShLangPS     = EShLangAll;
    static const EShLanguageMask EShLangHS     = EShLangAll;
    static const EShLanguageMask EShLangGS     = EShLangAll;

    // This structure encodes the prototype information for each HLSL intrinsic.
    // Because explicit enumeration would be cumbersome, it's procedurally generated.
    // orderKey can be:
    //   S = scalar, V = vector, M = matrix, - = void
    // typekey can be:
    //   D = double, F = float, U = uint, I = int, B = bool, S = sampler, s = shadowSampler, M = uint64_t, L = int64_t
    // An empty order or type key repeats the first one.  E.g: SVM,, means 3 args each of SVM.
    // '>' as first letter of order creates an output parameter
    // '<' as first letter of order creates an input parameter
    // '^' as first letter of order takes transpose dimensions
    // '%' as first letter of order creates texture of given F/I/U type (texture, itexture, etc)
    // '@' as first letter of order creates arrayed texture of given type
    // '$' / '&' as first letter of order creates 2DMS / 2DMSArray textures
    // '*' as first letter of order creates buffer object
    // '!' as first letter of order creates image object
    // '#' as first letter of order creates arrayed image object
    // '~' as first letter of order creates an image buffer object
    // '[' / ']' as first letter of order creates a SubpassInput/SubpassInputMS object

    static const struct {
        const char*   name;      // intrinsic name
        const char*   retOrder;  // return type key: empty matches order of 1st argument
        const char*   retType;   // return type key: empty matches type of 1st argument
        const char*   argOrder;  // argument order key
        const char*   argType;   // argument type key
        unsigned int  stage;     // stage mask
        bool          method;    // true if it's a method.
    } hlslIntrinsics[] = {
        // name                               retOrd   retType    argOrder          argType          stage mask     method
        // ----------------------------------------------------------------------------------------------------------------
        { "abort",                            nullptr, nullptr,   "-",              "-",             EShLangAll,    false },
        { "abs",                              nullptr, nullptr,   "SVM",            "DFUI",          EShLangAll,    false },
        { "acos",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "all",                              "S",    "B",        "SVM",            "BFIU",          EShLangAll,    false },
        { "AllMemoryBarrier",                 nullptr, nullptr,   "-",              "-",             EShLangCS,     false },
        { "AllMemoryBarrierWithGroupSync",    nullptr, nullptr,   "-",              "-",             EShLangCS,     false },
        { "any",                              "S",     "B",       "SVM",            "BFIU",          EShLangAll,    false },
        { "asdouble",                         "S",     "D",       "S,",             "UI,",           EShLangAll,    false },
        { "asdouble",                         "V2",    "D",       "V2,",            "UI,",           EShLangAll,    false },
        { "asfloat",                          nullptr, "F",       "SVM",            "BFIU",          EShLangAll,    false },
        { "asin",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "asint",                            nullptr, "I",       "SVM",            "FIU",           EShLangAll,    false },
        { "asuint",                           nullptr, "U",       "SVM",            "FIU",           EShLangAll,    false },
        { "atan",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "atan2",                            nullptr, nullptr,   "SVM,",           "F,",            EShLangAll,    false },
        { "ceil",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "CheckAccessFullyMapped",           "S",     "B" ,      "S",              "U",             EShLangPSCS,   false },
        { "clamp",                            nullptr, nullptr,   "SVM,,",          "FUI,,",         EShLangAll,    false },
        { "clip",                             "-",     "-",       "SVM",            "FUI",           EShLangPS,     false },
        { "cos",                              nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "cosh",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "countbits",                        nullptr, nullptr,   "SV",             "UI",            EShLangAll,    false },
        { "cross",                            nullptr, nullptr,   "V3,",            "F,",            EShLangAll,    false },
        { "D3DCOLORtoUBYTE4",                 "V4",    "I",       "V4",             "F",             EShLangAll,    false },
        { "ddx",                              nullptr, nullptr,   "SVM",            "F",             EShLangPS,     false },
        { "ddx_coarse",                       nullptr, nullptr,   "SVM",            "F",             EShLangPS,     false },
        { "ddx_fine",                         nullptr, nullptr,   "SVM",            "F",             EShLangPS,     false },
        { "ddy",                              nullptr, nullptr,   "SVM",            "F",             EShLangPS,     false },
        { "ddy_coarse",                       nullptr, nullptr,   "SVM",            "F",             EShLangPS,     false },
        { "ddy_fine",                         nullptr, nullptr,   "SVM",            "F",             EShLangPS,     false },
        { "degrees",                          nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "determinant",                      "S",     "F",       "M",              "F",             EShLangAll,    false },
        { "DeviceMemoryBarrier",              nullptr, nullptr,   "-",              "-",             EShLangPSCS,   false },
        { "DeviceMemoryBarrierWithGroupSync", nullptr, nullptr,   "-",              "-",             EShLangCS,     false },
        { "distance",                         "S",     "F",       "SV,",            "F,",            EShLangAll,    false },
        { "dot",                              "S",     nullptr,   "SV,",            "FI,",           EShLangAll,    false },
        { "dst",                              nullptr, nullptr,   "V4,",            "F,",            EShLangAll,    false },
        // { "errorf",                           "-",     "-",       "",             "",             EShLangAll,    false }, TODO: varargs
        { "EvaluateAttributeAtCentroid",      nullptr, nullptr,   "SVM",            "F",             EShLangPS,     false },
        { "EvaluateAttributeAtSample",        nullptr, nullptr,   "SVM,S",          "F,U",           EShLangPS,     false },
        { "EvaluateAttributeSnapped",         nullptr, nullptr,   "SVM,V2",         "F,I",           EShLangPS,     false },
        { "exp",                              nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "exp2",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "f16tof32",                         nullptr, "F",       "SV",             "U",             EShLangAll,    false },
        { "f32tof16",                         nullptr, "U",       "SV",             "F",             EShLangAll,    false },
        { "faceforward",                      nullptr, nullptr,   "V,,",            "F,,",           EShLangAll,    false },
        { "firstbithigh",                     nullptr, nullptr,   "SV",             "UI",            EShLangAll,    false },
        { "firstbitlow",                      nullptr, nullptr,   "SV",             "UI",            EShLangAll,    false },
        { "floor",                            nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "fma",                              nullptr, nullptr,   "SVM,,",          "D,,",           EShLangAll,    false },
        { "fmod",                             nullptr, nullptr,   "SVM,",           "F,",            EShLangAll,    false },
        { "frac",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "frexp",                            nullptr, nullptr,   "SVM,",           "F,",            EShLangAll,    false },
        { "fwidth",                           nullptr, nullptr,   "SVM",            "F",             EShLangPS,     false },
        { "GetRenderTargetSampleCount",       "S",     "U",       "-",              "-",             EShLangAll,    false },
        { "GetRenderTargetSamplePosition",    "V2",    "F",       "V1",             "I",             EShLangAll,    false },
        { "GroupMemoryBarrier",               nullptr, nullptr,   "-",              "-",             EShLangCS,     false },
        { "GroupMemoryBarrierWithGroupSync",  nullptr, nullptr,   "-",              "-",             EShLangCS,     false },
        { "InterlockedAdd",                   "-",     "-",       "SVM,,>",         "UI,,",          EShLangPSCS,   false },
        { "InterlockedAdd",                   "-",     "-",       "SVM,",           "UI,",           EShLangPSCS,   false },
        { "InterlockedAnd",                   "-",     "-",       "SVM,,>",         "UI,,",          EShLangPSCS,   false },
        { "InterlockedAnd",                   "-",     "-",       "SVM,",           "UI,",           EShLangPSCS,   false },
        { "InterlockedCompareExchange",       "-",     "-",       "SVM,,,>",        "UI,,,",         EShLangPSCS,   false },
        { "InterlockedCompareStore",          "-",     "-",       "SVM,,",          "UI,,",          EShLangPSCS,   false },
        { "InterlockedExchange",              "-",     "-",       "SVM,,>",         "UI,,",          EShLangPSCS,   false },
        { "InterlockedMax",                   "-",     "-",       "SVM,,>",         "UI,,",          EShLangPSCS,   false },
        { "InterlockedMax",                   "-",     "-",       "SVM,",           "UI,",           EShLangPSCS,   false },
        { "InterlockedMin",                   "-",     "-",       "SVM,,>",         "UI,,",          EShLangPSCS,   false },
        { "InterlockedMin",                   "-",     "-",       "SVM,",           "UI,",           EShLangPSCS,   false },
        { "InterlockedOr",                    "-",     "-",       "SVM,,>",         "UI,,",          EShLangPSCS,   false },
        { "InterlockedOr",                    "-",     "-",       "SVM,",           "UI,",           EShLangPSCS,   false },
        { "InterlockedXor",                   "-",     "-",       "SVM,,>",         "UI,,",          EShLangPSCS,   false },
        { "InterlockedXor",                   "-",     "-",       "SVM,",           "UI,",           EShLangPSCS,   false },
        { "isfinite",                         nullptr, "B" ,      "SVM",            "F",             EShLangAll,    false },
        { "isinf",                            nullptr, "B" ,      "SVM",            "F",             EShLangAll,    false },
        { "isnan",                            nullptr, "B" ,      "SVM",            "F",             EShLangAll,    false },
        { "ldexp",                            nullptr, nullptr,   "SVM,",           "F,",            EShLangAll,    false },
        { "length",                           "S",     "F",       "SV",             "F",             EShLangAll,    false },
        { "lerp",                             nullptr, nullptr,   "VM,,",           "F,,",           EShLangAll,    false },
        { "lerp",                             nullptr, nullptr,   "SVM,,S",         "F,,",           EShLangAll,    false },
        { "lit",                              "V4",    "F",       "S,,",            "F,,",           EShLangAll,    false },
        { "log",                              nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "log10",                            nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "log2",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "mad",                              nullptr, nullptr,   "SVM,,",          "DFUI,,",        EShLangAll,    false },
        { "max",                              nullptr, nullptr,   "SVM,",           "FIU,",          EShLangAll,    false },
        { "min",                              nullptr, nullptr,   "SVM,",           "FIU,",          EShLangAll,    false },
        { "modf",                             nullptr, nullptr,   "SVM,>",          "FIU,",          EShLangAll,    false },
        { "msad4",                            "V4",    "U",       "S,V2,V4",        "U,,",           EShLangAll,    false },
        { "mul",                              "S",     nullptr,   "S,S",            "FI,",           EShLangAll,    false },
        { "mul",                              "V",     nullptr,   "S,V",            "FI,",           EShLangAll,    false },
        { "mul",                              "M",     nullptr,   "S,M",            "FI,",           EShLangAll,    false },
        { "mul",                              "V",     nullptr,   "V,S",            "FI,",           EShLangAll,    false },
        { "mul",                              "S",     nullptr,   "V,V",            "FI,",           EShLangAll,    false },
        { "mul",                              "M",     nullptr,   "M,S",            "FI,",           EShLangAll,    false },
        // mat*mat form of mul is handled in createMatTimesMat()
        { "noise",                            "S",     "F",       "V",              "F",             EShLangPS,     false },
        { "normalize",                        nullptr, nullptr,   "V",              "F",             EShLangAll,    false },
        { "pow",                              nullptr, nullptr,   "SVM,",           "F,",            EShLangAll,    false },
        // { "printf",                           "-",     "-",       "",            "",              EShLangAll,    false }, TODO: varargs
        { "Process2DQuadTessFactorsAvg",      "-",     "-",       "V4,V2,>V4,>V2,", "F,,,,",         EShLangHS,     false },
        { "Process2DQuadTessFactorsMax",      "-",     "-",       "V4,V2,>V4,>V2,", "F,,,,",         EShLangHS,     false },
        { "Process2DQuadTessFactorsMin",      "-",     "-",       "V4,V2,>V4,>V2,", "F,,,,",         EShLangHS,     false },
        { "ProcessIsolineTessFactors",        "-",     "-",       "S,,>,>",         "F,,,",          EShLangHS,     false },
        { "ProcessQuadTessFactorsAvg",        "-",     "-",       "V4,S,>V4,>V2,",  "F,,,,",         EShLangHS,     false },
        { "ProcessQuadTessFactorsMax",        "-",     "-",       "V4,S,>V4,>V2,",  "F,,,,",         EShLangHS,     false },
        { "ProcessQuadTessFactorsMin",        "-",     "-",       "V4,S,>V4,>V2,",  "F,,,,",         EShLangHS,     false },
        { "ProcessTriTessFactorsAvg",         "-",     "-",       "V3,S,>V3,>S,",   "F,,,,",         EShLangHS,     false },
        { "ProcessTriTessFactorsMax",         "-",     "-",       "V3,S,>V3,>S,",   "F,,,,",         EShLangHS,     false },
        { "ProcessTriTessFactorsMin",         "-",     "-",       "V3,S,>V3,>S,",   "F,,,,",         EShLangHS,     false },
        { "radians",                          nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "rcp",                              nullptr, nullptr,   "SVM",            "FD",            EShLangAll,    false },
        { "reflect",                          nullptr, nullptr,   "V,",             "F,",            EShLangAll,    false },
        { "refract",                          nullptr, nullptr,   "V,V,S",          "F,,",           EShLangAll,    false },
        { "reversebits",                      nullptr, nullptr,   "SV",             "UI",            EShLangAll,    false },
        { "round",                            nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "rsqrt",                            nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "saturate",                         nullptr, nullptr ,  "SVM",            "F",             EShLangAll,    false },
        { "sign",                             nullptr, nullptr,   "SVM",            "FI",            EShLangAll,    false },
        { "sin",                              nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "sincos",                           "-",     "-",       "SVM,>,>",        "F,,",           EShLangAll,    false },
        { "sinh",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "smoothstep",                       nullptr, nullptr,   "SVM,,",          "F,,",           EShLangAll,    false },
        { "sqrt",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "step",                             nullptr, nullptr,   "SVM,",           "F,",            EShLangAll,    false },
        { "tan",                              nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "tanh",                             nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },
        { "tex1D",                            "V4",    "F",       "S,S",            "S,F",           EShLangPS,     false },
        { "tex1D",                            "V4",    "F",       "S,S,V1,",        "S,F,,",         EShLangPS,     false },
        { "tex1Dbias",                        "V4",    "F",       "S,V4",           "S,F",           EShLangPS,     false },
        { "tex1Dgrad",                        "V4",    "F",       "S,,,",           "S,F,,",         EShLangPS,     false },
        { "tex1Dlod",                         "V4",    "F",       "S,V4",           "S,F",           EShLangPS,     false },
        { "tex1Dproj",                        "V4",    "F",       "S,V4",           "S,F",           EShLangPS,     false },
        { "tex2D",                            "V4",    "F",       "V2,",            "S,F",           EShLangPS,     false },
        { "tex2D",                            "V4",    "F",       "V2,,,",          "S,F,,",         EShLangPS,     false },
        { "tex2Dbias",                        "V4",    "F",       "V2,V4",          "S,F",           EShLangPS,     false },
        { "tex2Dgrad",                        "V4",    "F",       "V2,,,",          "S,F,,",         EShLangPS,     false },
        { "tex2Dlod",                         "V4",    "F",       "V2,V4",          "S,F",           EShLangAll,    false },
        { "tex2Dproj",                        "V4",    "F",       "V2,V4",          "S,F",           EShLangPS,     false },
        { "tex3D",                            "V4",    "F",       "V3,",            "S,F",           EShLangPS,     false },
        { "tex3D",                            "V4",    "F",       "V3,,,",          "S,F,,",         EShLangPS,     false },
        { "tex3Dbias",                        "V4",    "F",       "V3,V4",          "S,F",           EShLangPS,     false },
        { "tex3Dgrad",                        "V4",    "F",       "V3,,,",          "S,F,,",         EShLangPS,     false },
        { "tex3Dlod",                         "V4",    "F",       "V3,V4",          "S,F",           EShLangPS,     false },
        { "tex3Dproj",                        "V4",    "F",       "V3,V4",          "S,F",           EShLangPS,     false },
        { "texCUBE",                          "V4",    "F",       "V4,V3",          "S,F",           EShLangPS,     false },
        { "texCUBE",                          "V4",    "F",       "V4,V3,,",        "S,F,,",         EShLangPS,     false },
        { "texCUBEbias",                      "V4",    "F",       "V4,",            "S,F",           EShLangPS,     false },
        { "texCUBEgrad",                      "V4",    "F",       "V4,V3,,",        "S,F,,",         EShLangPS,     false },
        { "texCUBElod",                       "V4",    "F",       "V4,",            "S,F",           EShLangPS,     false },
        { "texCUBEproj",                      "V4",    "F",       "V4,",            "S,F",           EShLangPS,     false },
        { "transpose",                        "^M",    nullptr,   "M",              "FUIB",          EShLangAll,    false },
        { "trunc",                            nullptr, nullptr,   "SVM",            "F",             EShLangAll,    false },

        // Texture object methods.  Return type can be overridden by shader declaration.
        // !O = no offset, O = offset
        { "Sample",             /*!O*/        "V4",    nullptr,   "%@,S,V",         "FIU,S,F",        EShLangPS,    true },
        { "Sample",             /* O*/        "V4",    nullptr,   "%@,S,V,",        "FIU,S,F,I",      EShLangPS,    true },

        { "SampleBias",         /*!O*/        "V4",    nullptr,   "%@,S,V,S",       "FIU,S,F,",       EShLangPS,    true },
        { "SampleBias",         /* O*/        "V4",    nullptr,   "%@,S,V,S,V",     "FIU,S,F,,I",     EShLangPS,    true },

        // TODO: FXC accepts int/uint samplers here.  unclear what that means.
        { "SampleCmp",          /*!O*/        "S",     "F",       "%@,S,V,S",       "FIU,s,F,",       EShLangPS,    true },
        { "SampleCmp",          /* O*/        "S",     "F",       "%@,S,V,S,V",     "FIU,s,F,,I",     EShLangPS,    true },

        // TODO: FXC accepts int/uint samplers here.  unclear what that means.
        { "SampleCmpLevelZero", /*!O*/        "S",     "F",       "%@,S,V,S",       "FIU,s,F,F",      EShLangPS,    true },
        { "SampleCmpLevelZero", /* O*/        "S",     "F",       "%@,S,V,S,V",     "FIU,s,F,F,I",    EShLangPS,    true },

        { "SampleGrad",         /*!O*/        "V4",    nullptr,   "%@,S,V,,",       "FIU,S,F,,",      EShLangAll,   true },
        { "SampleGrad",         /* O*/        "V4",    nullptr,   "%@,S,V,,,",      "FIU,S,F,,,I",    EShLangAll,   true },

        { "SampleLevel",        /*!O*/        "V4",    nullptr,   "%@,S,V,S",       "FIU,S,F,",       EShLangAll,   true },
        { "SampleLevel",        /* O*/        "V4",    nullptr,   "%@,S,V,S,V",     "FIU,S,F,,I",     EShLangAll,   true },

        { "Load",               /*!O*/        "V4",    nullptr,   "%@,V",           "FIU,I",          EShLangAll,   true },
        { "Load",               /* O*/        "V4",    nullptr,   "%@,V,V",         "FIU,I,I",        EShLangAll,   true },
        { "Load", /* +sampleidex*/            "V4",    nullptr,   "$&,V,S",         "FIU,I,I",        EShLangAll,   true },
        { "Load", /* +samplindex, offset*/    "V4",    nullptr,   "$&,V,S,V",       "FIU,I,I,I",      EShLangAll,   true },

        // RWTexture loads
        { "Load",                             "V4",    nullptr,   "!#,V",           "FIU,I",          EShLangAll,   true },
        // (RW)Buffer loads
        { "Load",                             "V4",    nullptr,   "~*1,V",          "FIU,I",          EShLangAll,   true },

        { "Gather",             /*!O*/        "V4",    nullptr,   "%@,S,V",         "FIU,S,F",        EShLangAll,   true },
        { "Gather",             /* O*/        "V4",    nullptr,   "%@,S,V,V",       "FIU,S,F,I",      EShLangAll,   true },

        { "CalculateLevelOfDetail",           "S",     "F",       "%@,S,V",         "FUI,S,F",        EShLangPS,    true },
        { "CalculateLevelOfDetailUnclamped",  "S",     "F",       "%@,S,V",         "FUI,S,F",        EShLangPS,    true },

        { "GetSamplePosition",                "V2",    "F",       "$&2,S",          "FUI,I",          EShLangVSPSGS,true },

        //
        // UINT Width
        // UINT MipLevel, UINT Width, UINT NumberOfLevels
        { "GetDimensions",   /* 1D */         "-",     "-",       "%!~1,>S",        "FUI,U",          EShLangAll,   true },
        { "GetDimensions",   /* 1D */         "-",     "-",       "%!~1,>S",        "FUI,F",          EShLangAll,   true },
        { "GetDimensions",   /* 1D */         "-",     "-",       "%1,S,>S,",       "FUI,U,,",        EShLangAll,   true },
        { "GetDimensions",   /* 1D */         "-",     "-",       "%1,S,>S,",       "FUI,U,F,",       EShLangAll,   true },

        // UINT Width, UINT Elements
        // UINT MipLevel, UINT Width, UINT Elements, UINT NumberOfLevels
        { "GetDimensions",   /* 1DArray */    "-",     "-",       "@#1,>S,",        "FUI,U,",         EShLangAll,   true },
        { "GetDimensions",   /* 1DArray */    "-",     "-",       "@#1,>S,",        "FUI,F,",         EShLangAll,   true },
        { "GetDimensions",   /* 1DArray */    "-",     "-",       "@1,S,>S,,",      "FUI,U,,,",       EShLangAll,   true },
        { "GetDimensions",   /* 1DArray */    "-",     "-",       "@1,S,>S,,",      "FUI,U,F,,",      EShLangAll,   true },

        // UINT Width, UINT Height
        // UINT MipLevel, UINT Width, UINT Height, UINT NumberOfLevels
        { "GetDimensions",   /* 2D */         "-",     "-",       "%!2,>S,",        "FUI,U,",         EShLangAll,   true },
        { "GetDimensions",   /* 2D */         "-",     "-",       "%!2,>S,",        "FUI,F,",         EShLangAll,   true },
        { "GetDimensions",   /* 2D */         "-",     "-",       "%2,S,>S,,",      "FUI,U,,,",       EShLangAll,   true },
        { "GetDimensions",   /* 2D */         "-",     "-",       "%2,S,>S,,",      "FUI,U,F,,",      EShLangAll,   true },

        // UINT Width, UINT Height, UINT Elements
        // UINT MipLevel, UINT Width, UINT Height, UINT Elements, UINT NumberOfLevels
        { "GetDimensions",   /* 2DArray */    "-",     "-",       "@#2,>S,,",       "FUI,U,,",        EShLangAll,   true },
        { "GetDimensions",   /* 2DArray */    "-",     "-",       "@#2,>S,,",       "FUI,F,F,F",      EShLangAll,   true },
        { "GetDimensions",   /* 2DArray */    "-",     "-",       "@2,S,>S,,,",     "FUI,U,,,,",      EShLangAll,   true },
        { "GetDimensions",   /* 2DArray */    "-",     "-",       "@2,S,>S,,,",     "FUI,U,F,,,",     EShLangAll,   true },

        // UINT Width, UINT Height, UINT Depth
        // UINT MipLevel, UINT Width, UINT Height, UINT Depth, UINT NumberOfLevels
        { "GetDimensions",   /* 3D */         "-",     "-",       "%!3,>S,,",       "FUI,U,,",        EShLangAll,   true },
        { "GetDimensions",   /* 3D */         "-",     "-",       "%!3,>S,,",       "FUI,F,,",        EShLangAll,   true },
        { "GetDimensions",   /* 3D */         "-",     "-",       "%3,S,>S,,,",     "FUI,U,,,,",      EShLangAll,   true },
        { "GetDimensions",   /* 3D */         "-",     "-",       "%3,S,>S,,,",     "FUI,U,F,,,",     EShLangAll,   true },

        // UINT Width, UINT Height
        // UINT MipLevel, UINT Width, UINT Height, UINT NumberOfLevels
        { "GetDimensions",   /* Cube */       "-",     "-",       "%4,>S,",         "FUI,U,",         EShLangAll,   true },
        { "GetDimensions",   /* Cube */       "-",     "-",       "%4,>S,",         "FUI,F,",         EShLangAll,   true },
        { "GetDimensions",   /* Cube */       "-",     "-",       "%4,S,>S,,",      "FUI,U,,,",       EShLangAll,   true },
        { "GetDimensions",   /* Cube */       "-",     "-",       "%4,S,>S,,",      "FUI,U,F,,",      EShLangAll,   true },

        // UINT Width, UINT Height, UINT Elements
        // UINT MipLevel, UINT Width, UINT Height, UINT Elements, UINT NumberOfLevels
        { "GetDimensions",   /* CubeArray */  "-",     "-",       "@4,>S,,",        "FUI,U,,",        EShLangAll,   true },
        { "GetDimensions",   /* CubeArray */  "-",     "-",       "@4,>S,,",        "FUI,F,,",        EShLangAll,   true },
        { "GetDimensions",   /* CubeArray */  "-",     "-",       "@4,S,>S,,,",     "FUI,U,,,,",      EShLangAll,   true },
        { "GetDimensions",   /* CubeArray */  "-",     "-",       "@4,S,>S,,,",     "FUI,U,F,,,",     EShLangAll,   true },

        // UINT Width, UINT Height, UINT Samples
        // UINT Width, UINT Height, UINT Elements, UINT Samples
        { "GetDimensions",   /* 2DMS */       "-",     "-",       "$2,>S,,",        "FUI,U,,",        EShLangAll,   true },
        { "GetDimensions",   /* 2DMS */       "-",     "-",       "$2,>S,,",        "FUI,U,,",        EShLangAll,   true },
        { "GetDimensions",   /* 2DMSArray */  "-",     "-",       "&2,>S,,,",       "FUI,U,,,",       EShLangAll,   true },
        { "GetDimensions",   /* 2DMSArray */  "-",     "-",       "&2,>S,,,",       "FUI,U,,,",       EShLangAll,   true },

        // SM5 texture methods
        { "GatherRed",       /*!O*/           "V4",    nullptr,   "%@,S,V",         "FIU,S,F",        EShLangAll,   true },
        { "GatherRed",       /* O*/           "V4",    nullptr,   "%@,S,V,",        "FIU,S,F,I",      EShLangAll,   true },
        { "GatherRed",       /* O, status*/   "V4",    nullptr,   "%@,S,V,,>S",     "FIU,S,F,I,U",    EShLangAll,   true },
        { "GatherRed",       /* O-4 */        "V4",    nullptr,   "%@,S,V,,,,",     "FIU,S,F,I,,,",   EShLangAll,   true },
        { "GatherRed",       /* O-4, status */"V4",    nullptr,   "%@,S,V,,,,,S",   "FIU,S,F,I,,,,U", EShLangAll,   true },

        { "GatherGreen",     /*!O*/           "V4",    nullptr,   "%@,S,V",         "FIU,S,F",        EShLangAll,   true },
        { "GatherGreen",     /* O*/           "V4",    nullptr,   "%@,S,V,",        "FIU,S,F,I",      EShLangAll,   true },
        { "GatherGreen",     /* O, status*/   "V4",    nullptr,   "%@,S,V,,>S",     "FIU,S,F,I,U",    EShLangAll,   true },
        { "GatherGreen",     /* O-4 */        "V4",    nullptr,   "%@,S,V,,,,",     "FIU,S,F,I,,,",   EShLangAll,   true },
        { "GatherGreen",     /* O-4, status */"V4",    nullptr,   "%@,S,V,,,,,S",   "FIU,S,F,I,,,,U", EShLangAll,   true },

        { "GatherBlue",      /*!O*/           "V4",    nullptr,   "%@,S,V",         "FIU,S,F",        EShLangAll,   true },
        { "GatherBlue",      /* O*/           "V4",    nullptr,   "%@,S,V,",        "FIU,S,F,I",      EShLangAll,   true },
        { "GatherBlue",      /* O, status*/   "V4",    nullptr,   "%@,S,V,,>S",     "FIU,S,F,I,U",    EShLangAll,   true },
        { "GatherBlue",      /* O-4 */        "V4",    nullptr,   "%@,S,V,,,,",     "FIU,S,F,I,,,",   EShLangAll,   true },
        { "GatherBlue",      /* O-4, status */"V4",    nullptr,   "%@,S,V,,,,,S",   "FIU,S,F,I,,,,U", EShLangAll,   true },

        { "GatherAlpha",     /*!O*/           "V4",    nullptr,   "%@,S,V",         "FIU,S,F",        EShLangAll,   true },
        { "GatherAlpha",     /* O*/           "V4",    nullptr,   "%@,S,V,",        "FIU,S,F,I",      EShLangAll,   true },
        { "GatherAlpha",     /* O, status*/   "V4",    nullptr,   "%@,S,V,,>S",     "FIU,S,F,I,U",    EShLangAll,   true },
        { "GatherAlpha",     /* O-4 */        "V4",    nullptr,   "%@,S,V,,,,",     "FIU,S,F,I,,,",   EShLangAll,   true },
        { "GatherAlpha",     /* O-4, status */"V4",    nullptr,   "%@,S,V,,,,,S",   "FIU,S,F,I,,,,U", EShLangAll,   true },

        { "GatherCmp",       /*!O*/           "V4",    nullptr,   "%@,S,V,S",       "FIU,s,F,",       EShLangAll,   true },
        { "GatherCmp",       /* O*/           "V4",    nullptr,   "%@,S,V,S,V",     "FIU,s,F,,I",     EShLangAll,   true },
        { "GatherCmp",       /* O, status*/   "V4",    nullptr,   "%@,S,V,S,V,>S",  "FIU,s,F,,I,U",   EShLangAll,   true },
        { "GatherCmp",       /* O-4 */        "V4",    nullptr,   "%@,S,V,S,V,,,",  "FIU,s,F,,I,,,",  EShLangAll,   true },
        { "GatherCmp",       /* O-4, status */"V4",    nullptr,   "%@,S,V,S,V,,V,S","FIU,s,F,,I,,,,U",EShLangAll,   true },

        { "GatherCmpRed",    /*!O*/           "V4",    nullptr,   "%@,S,V,S",       "FIU,s,F,",       EShLangAll,   true },
        { "GatherCmpRed",    /* O*/           "V4",    nullptr,   "%@,S,V,S,V",     "FIU,s,F,,I",     EShLangAll,   true },
        { "GatherCmpRed",    /* O, status*/   "V4",    nullptr,   "%@,S,V,S,V,>S",  "FIU,s,F,,I,U",   EShLangAll,   true },
        { "GatherCmpRed",    /* O-4 */        "V4",    nullptr,   "%@,S,V,S,V,,,",  "FIU,s,F,,I,,,",  EShLangAll,   true },
        { "GatherCmpRed",    /* O-4, status */"V4",    nullptr,   "%@,S,V,S,V,,V,S","FIU,s,F,,I,,,,U",EShLangAll,   true },

        { "GatherCmpGreen",  /*!O*/           "V4",    nullptr,   "%@,S,V,S",       "FIU,s,F,",       EShLangAll,   true },
        { "GatherCmpGreen",  /* O*/           "V4",    nullptr,   "%@,S,V,S,V",     "FIU,s,F,,I",     EShLangAll,   true },
        { "GatherCmpGreen",  /* O, status*/   "V4",    nullptr,   "%@,S,V,S,V,>S",  "FIU,s,F,,I,U",   EShLangAll,   true },
        { "GatherCmpGreen",  /* O-4 */        "V4",    nullptr,   "%@,S,V,S,V,,,",  "FIU,s,F,,I,,,",  EShLangAll,   true },
        { "GatherCmpGreen",  /* O-4, status */"V4",    nullptr,   "%@,S,V,S,V,,,,S","FIU,s,F,,I,,,,U",EShLangAll,   true },

        { "GatherCmpBlue",   /*!O*/           "V4",    nullptr,   "%@,S,V,S",       "FIU,s,F,",       EShLangAll,   true },
        { "GatherCmpBlue",   /* O*/           "V4",    nullptr,   "%@,S,V,S,V",     "FIU,s,F,,I",     EShLangAll,   true },
        { "GatherCmpBlue",   /* O, status*/   "V4",    nullptr,   "%@,S,V,S,V,>S",  "FIU,s,F,,I,U",   EShLangAll,   true },
        { "GatherCmpBlue",   /* O-4 */        "V4",    nullptr,   "%@,S,V,S,V,,,",  "FIU,s,F,,I,,,",  EShLangAll,   true },
        { "GatherCmpBlue",   /* O-4, status */"V4",    nullptr,   "%@,S,V,S,V,,,,S","FIU,s,F,,I,,,,U",EShLangAll,   true },

        { "GatherCmpAlpha",  /*!O*/           "V4",    nullptr,   "%@,S,V,S",       "FIU,s,F,",       EShLangAll,   true },
        { "GatherCmpAlpha",  /* O*/           "V4",    nullptr,   "%@,S,V,S,V",     "FIU,s,F,,I",     EShLangAll,   true },
        { "GatherCmpAlpha",  /* O, status*/   "V4",    nullptr,   "%@,S,V,S,V,>S",  "FIU,s,F,,I,U",   EShLangAll,   true },
        { "GatherCmpAlpha",  /* O-4 */        "V4",    nullptr,   "%@,S,V,S,V,,,",  "FIU,s,F,,I,,,",  EShLangAll,   true },
        { "GatherCmpAlpha",  /* O-4, status */"V4",    nullptr,   "%@,S,V,S,V,,,,S","FIU,s,F,,I,,,,U",EShLangAll,   true },

        // geometry methods
        { "Append",                           "-",     "-",       "-",              "-",              EShLangGS ,   true },
        { "RestartStrip",                     "-",     "-",       "-",              "-",              EShLangGS ,   true },

        // Methods for structurebuffers.  TODO: wildcard type matching.
        { "Load",                             nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "Load2",                            nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "Load3",                            nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "Load4",                            nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "Store",                            nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "Store2",                           nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "Store3",                           nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "Store4",                           nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "GetDimensions",                    nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "InterlockedAdd",                   nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "InterlockedAnd",                   nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "InterlockedCompareExchange",       nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "InterlockedCompareStore",          nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "InterlockedExchange",              nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "InterlockedMax",                   nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "InterlockedMin",                   nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "InterlockedOr",                    nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "InterlockedXor",                   nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "IncrementCounter",                 nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "DecrementCounter",                 nullptr, nullptr,   "-",              "-",              EShLangAll,   true },
        { "Consume",                          nullptr, nullptr,   "-",              "-",              EShLangAll,   true },

        // SM 6.0

        { "WaveIsFirstLane",                  "S",     "B",       "-",              "-",              EShLangPSCS,  false},
        { "WaveGetLaneCount",                 "S",     "U",       "-",              "-",              EShLangPSCS,  false},
        { "WaveGetLaneIndex",                 "S",     "U",       "-",              "-",              EShLangPSCS,  false},
        { "WaveActiveAnyTrue",                "S",     "B",       "S",              "B",              EShLangPSCS,  false},
        { "WaveActiveAllTrue",                "S",     "B",       "S",              "B",              EShLangPSCS,  false},
        { "WaveActiveBallot",                 "V4",    "U",       "S",              "B",              EShLangPSCS,  false},
        { "WaveReadLaneAt",                   nullptr, nullptr,   "SV,S",           "DFUI,U",         EShLangPSCS,  false},
        { "WaveReadLaneFirst",                nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "WaveActiveAllEqual",               "S",     "B",       "SV",             "DFUI",           EShLangPSCS,  false},
        { "WaveActiveAllEqualBool",           "S",     "B",       "S",              "B",              EShLangPSCS,  false},
        { "WaveActiveCountBits",              "S",     "U",       "S",              "B",              EShLangPSCS,  false},
        
        { "WaveActiveSum",                    nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "WaveActiveProduct",                nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "WaveActiveBitAnd",                 nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "WaveActiveBitOr",                  nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "WaveActiveBitXor",                 nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "WaveActiveMin",                    nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "WaveActiveMax",                    nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "WavePrefixSum",                    nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "WavePrefixProduct",                nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "WavePrefixCountBits",              "S",     "U",       "S",              "B",              EShLangPSCS,  false},
        { "QuadReadAcrossX",                  nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "QuadReadAcrossY",                  nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "QuadReadAcrossDiagonal",           nullptr, nullptr,   "SV",             "DFUI",           EShLangPSCS,  false},
        { "QuadReadLaneAt",                   nullptr, nullptr,   "SV,S",           "DFUI,U",         EShLangPSCS,  false},

        // Methods for subpass input objects
        { "SubpassLoad",                      "V4",    nullptr,   "[",              "FIU",            EShLangPS,    true },
        { "SubpassLoad",                      "V4",    nullptr,   "],S",            "FIU,I",          EShLangPS,    true },

        // Mark end of list, since we want to avoid a range-based for, as some compilers don't handle it yet.
        { nullptr,                            nullptr, nullptr,   nullptr,      nullptr,  0, false },
    };

    // Create prototypes for the intrinsics.  TODO: Avoid ranged based for until all compilers can handle it.
    for (int icount = 0; hlslIntrinsics[icount].name; ++icount) {
        const auto& intrinsic = hlslIntrinsics[icount];

        for (int stage = 0; stage < EShLangCount; ++stage) {                                // for each stage...
            if ((intrinsic.stage & (1<<stage)) == 0) // skip inapplicable stages
                continue;

            // reference to either the common builtins, or stage specific builtins.
            TString& s = (intrinsic.stage == EShLangAll) ? commonBuiltins : stageBuiltins[stage];

            for (const char* argOrder = intrinsic.argOrder; !IsEndOfArg(argOrder); ++argOrder) { // for each order...
                const bool isTexture   = IsTextureType(*argOrder);
                const bool isArrayed   = IsArrayed(*argOrder);
                const bool isMS        = IsTextureMS(*argOrder);
                const bool isBuffer    = IsBuffer(*argOrder);
                const bool isImage     = IsImage(*argOrder);
                const bool mipInCoord  = HasMipInCoord(intrinsic.name, isMS, isBuffer, isImage);
                const int fixedVecSize = FixedVecSize(argOrder);
                const int coordArg     = CoordinateArgPos(intrinsic.name, isTexture);

                // calculate min and max vector and matrix dimensions
                int dim0Min = 1;
                int dim0Max = 1;
                int dim1Min = 1;
                int dim1Max = 1;

                FindVectorMatrixBounds(argOrder, fixedVecSize, dim0Min, dim0Max, dim1Min, dim1Max);

                for (const char* argType = intrinsic.argType; !IsEndOfArg(argType); ++argType) { // for each type...
                    for (int dim0 = dim0Min; dim0 <= dim0Max; ++dim0) {          // for each dim 0...
                        for (int dim1 = dim1Min; dim1 <= dim1Max; ++dim1) {      // for each dim 1...
                            const char* retOrder = intrinsic.retOrder ? intrinsic.retOrder : argOrder;
                            const char* retType  = intrinsic.retType  ? intrinsic.retType  : argType;

                            if (!IsValid(intrinsic.name, *retOrder, *retType, *argOrder, *argType, dim0, dim1))
                                continue;

                            // Reject some forms of sample methods that don't exist.
                            if (isTexture && IsIllegalSample(intrinsic.name, argOrder, dim0))
                                continue;

                            AppendTypeName(s, retOrder, retType, dim0, dim1);  // add return type
                            s.append(" ");                                     // space between type and name

                            // methods have a prefix.  TODO: it would be better as an invalid identifier character,
                            // but that requires a scanner change.
                            if (intrinsic.method)
                                s.append(BUILTIN_PREFIX);

                            s.append(intrinsic.name);                          // intrinsic name
                            s.append("(");                                     // open paren

                            const char* prevArgOrder = nullptr;
                            const char* prevArgType = nullptr;

                            // Append argument types, if any.
                            for (int arg = 0; ; ++arg) {
                                const char* nthArgOrder(NthArg(argOrder, arg));
                                const char* nthArgType(NthArg(argType, arg));

                                if (nthArgOrder == nullptr || nthArgType == nullptr)
                                    break;

                                // cube textures use vec3 coordinates
                                int argDim0 = isTexture && arg > 0 ? std::min(dim0, 3) : dim0;

                                s.append(arg > 0 ? ", ": "");  // comma separator if needed

                                const char* orderBegin = nthArgOrder;
                                nthArgOrder = IoParam(s, nthArgOrder);

                                // Comma means use the previous argument order and type.
                                HandleRepeatArg(nthArgOrder, prevArgOrder, orderBegin);
                                HandleRepeatArg(nthArgType,  prevArgType, nthArgType);

                                // In case the repeated arg has its own I/O marker
                                nthArgOrder = IoParam(s, nthArgOrder);

                                // arrayed textures have one extra coordinate dimension, except for
                                // the CalculateLevelOfDetail family.
                                if (isArrayed && arg == coordArg && !NoArrayCoord(intrinsic.name))
                                    argDim0++;

                                // Some texture methods use an addition arg dimension to hold mip
                                if (arg == coordArg && mipInCoord)
                                    argDim0++;

                                // For textures, the 1D case isn't a 1-vector, but a scalar.
                                if (isTexture && argDim0 == 1 && arg > 0 && *nthArgOrder == 'V')
                                    nthArgOrder = "S";

                                AppendTypeName(s, nthArgOrder, nthArgType, argDim0, dim1); // Add arguments
                            }

                            s.append(");\n");            // close paren and trailing semicolon
                        } // dim 1 loop
                    } // dim 0 loop
                } // arg type loop

                // skip over special characters
                if (isTexture && isalpha(argOrder[1]))
                    ++argOrder;
                if (isdigit(argOrder[1]))
                    ++argOrder;
            } // arg order loop

            if (intrinsic.stage == EShLangAll) // common builtins are only added once.
                break;
        }
    }

    createMatTimesMat(); // handle this case separately, for convenience

    // printf("Common:\n%s\n",   getCommonString().c_str());
    // printf("Frag:\n%s\n",     getStageString(EShLangFragment).c_str());
    // printf("Vertex:\n%s\n",   getStageString(EShLangVertex).c_str());
    // printf("Geo:\n%s\n",      getStageString(EShLangGeometry).c_str());
    // printf("TessCtrl:\n%s\n", getStageString(EShLangTessControl).c_str());
    // printf("TessEval:\n%s\n", getStageString(EShLangTessEvaluation).c_str());
    // printf("Compute:\n%s\n",  getStageString(EShLangCompute).c_str());
}

//
// Add context-dependent built-in functions and variables that are present
// for the given version and profile.  All the results are put into just the
// commonBuiltins, because it is called for just a specific stage.  So,
// add stage-specific entries to the commonBuiltins, and only if that stage
// was requested.
//
void TBuiltInParseablesHlsl::initialize(const TBuiltInResource& /*resources*/, int /*version*/, EProfile /*profile*/,
                                        const SpvVersion& /*spvVersion*/, EShLanguage /*language*/)
{
}

//
// Finish adding/processing context-independent built-in symbols.
// 1) Programmatically add symbols that could not be added by simple text strings above.
// 2) Map built-in functions to operators, for those that will turn into an operation node
//    instead of remaining a function call.
// 3) Tag extension-related symbols added to their base version with their extensions, so
//    that if an early version has the extension turned off, there is an error reported on use.
//
void TBuiltInParseablesHlsl::identifyBuiltIns(int /*version*/, EProfile /*profile*/, const SpvVersion& /*spvVersion*/, EShLanguage /*language*/,
                                              TSymbolTable& symbolTable)
{
    // symbolTable.relateToOperator("abort",                       EOpAbort);
    symbolTable.relateToOperator("abs",                         EOpAbs);
    symbolTable.relateToOperator("acos",                        EOpAcos);
    symbolTable.relateToOperator("all",                         EOpAll);
    symbolTable.relateToOperator("AllMemoryBarrier",            EOpMemoryBarrier);
    symbolTable.relateToOperator("AllMemoryBarrierWithGroupSync", EOpAllMemoryBarrierWithGroupSync);
    symbolTable.relateToOperator("any",                         EOpAny);
    symbolTable.relateToOperator("asdouble",                    EOpAsDouble);
    symbolTable.relateToOperator("asfloat",                     EOpIntBitsToFloat);
    symbolTable.relateToOperator("asin",                        EOpAsin);
    symbolTable.relateToOperator("asint",                       EOpFloatBitsToInt);
    symbolTable.relateToOperator("asuint",                      EOpFloatBitsToUint);
    symbolTable.relateToOperator("atan",                        EOpAtan);
    symbolTable.relateToOperator("atan2",                       EOpAtan);
    symbolTable.relateToOperator("ceil",                        EOpCeil);
    // symbolTable.relateToOperator("CheckAccessFullyMapped");
    symbolTable.relateToOperator("clamp",                       EOpClamp);
    symbolTable.relateToOperator("clip",                        EOpClip);
    symbolTable.relateToOperator("cos",                         EOpCos);
    symbolTable.relateToOperator("cosh",                        EOpCosh);
    symbolTable.relateToOperator("countbits",                   EOpBitCount);
    symbolTable.relateToOperator("cross",                       EOpCross);
    symbolTable.relateToOperator("D3DCOLORtoUBYTE4",            EOpD3DCOLORtoUBYTE4);
    symbolTable.relateToOperator("ddx",                         EOpDPdx);
    symbolTable.relateToOperator("ddx_coarse",                  EOpDPdxCoarse);
    symbolTable.relateToOperator("ddx_fine",                    EOpDPdxFine);
    symbolTable.relateToOperator("ddy",                         EOpDPdy);
    symbolTable.relateToOperator("ddy_coarse",                  EOpDPdyCoarse);
    symbolTable.relateToOperator("ddy_fine",                    EOpDPdyFine);
    symbolTable.relateToOperator("degrees",                     EOpDegrees);
    symbolTable.relateToOperator("determinant",                 EOpDeterminant);
    symbolTable.relateToOperator("DeviceMemoryBarrier",         EOpDeviceMemoryBarrier);
    symbolTable.relateToOperator("DeviceMemoryBarrierWithGroupSync", EOpDeviceMemoryBarrierWithGroupSync);
    symbolTable.relateToOperator("distance",                    EOpDistance);
    symbolTable.relateToOperator("dot",                         EOpDot);
    symbolTable.relateToOperator("dst",                         EOpDst);
    // symbolTable.relateToOperator("errorf",                      EOpErrorf);
    symbolTable.relateToOperator("EvaluateAttributeAtCentroid", EOpInterpolateAtCentroid);
    symbolTable.relateToOperator("EvaluateAttributeAtSample",   EOpInterpolateAtSample);
    symbolTable.relateToOperator("EvaluateAttributeSnapped",    EOpEvaluateAttributeSnapped);
    symbolTable.relateToOperator("exp",                         EOpExp);
    symbolTable.relateToOperator("exp2",                        EOpExp2);
    symbolTable.relateToOperator("f16tof32",                    EOpF16tof32);
    symbolTable.relateToOperator("f32tof16",                    EOpF32tof16);
    symbolTable.relateToOperator("faceforward",                 EOpFaceForward);
    symbolTable.relateToOperator("firstbithigh",                EOpFindMSB);
    symbolTable.relateToOperator("firstbitlow",                 EOpFindLSB);
    symbolTable.relateToOperator("floor",                       EOpFloor);
    symbolTable.relateToOperator("fma",                         EOpFma);
    symbolTable.relateToOperator("fmod",                        EOpMod);
    symbolTable.relateToOperator("frac",                        EOpFract);
    symbolTable.relateToOperator("frexp",                       EOpFrexp);
    symbolTable.relateToOperator("fwidth",                      EOpFwidth);
    // symbolTable.relateToOperator("GetRenderTargetSampleCount");
    // symbolTable.relateToOperator("GetRenderTargetSamplePosition");
    symbolTable.relateToOperator("GroupMemoryBarrier",          EOpWorkgroupMemoryBarrier);
    symbolTable.relateToOperator("GroupMemoryBarrierWithGroupSync", EOpWorkgroupMemoryBarrierWithGroupSync);
    symbolTable.relateToOperator("InterlockedAdd",              EOpInterlockedAdd);
    symbolTable.relateToOperator("InterlockedAnd",              EOpInterlockedAnd);
    symbolTable.relateToOperator("InterlockedCompareExchange",  EOpInterlockedCompareExchange);
    symbolTable.relateToOperator("InterlockedCompareStore",     EOpInterlockedCompareStore);
    symbolTable.relateToOperator("InterlockedExchange",         EOpInterlockedExchange);
    symbolTable.relateToOperator("InterlockedMax",              EOpInterlockedMax);
    symbolTable.relateToOperator("InterlockedMin",              EOpInterlockedMin);
    symbolTable.relateToOperator("InterlockedOr",               EOpInterlockedOr);
    symbolTable.relateToOperator("InterlockedXor",              EOpInterlockedXor);
    symbolTable.relateToOperator("isfinite",                    EOpIsFinite);
    symbolTable.relateToOperator("isinf",                       EOpIsInf);
    symbolTable.relateToOperator("isnan",                       EOpIsNan);
    symbolTable.relateToOperator("ldexp",                       EOpLdexp);
    symbolTable.relateToOperator("length",                      EOpLength);
    symbolTable.relateToOperator("lerp",                        EOpMix);
    symbolTable.relateToOperator("lit",                         EOpLit);
    symbolTable.relateToOperator("log",                         EOpLog);
    symbolTable.relateToOperator("log10",                       EOpLog10);
    symbolTable.relateToOperator("log2",                        EOpLog2);
    symbolTable.relateToOperator("mad",                         EOpFma);
    symbolTable.relateToOperator("max",                         EOpMax);
    symbolTable.relateToOperator("min",                         EOpMin);
    symbolTable.relateToOperator("modf",                        EOpModf);
    // symbolTable.relateToOperator("msad4",                       EOpMsad4);
    symbolTable.relateToOperator("mul",                         EOpGenMul);
    // symbolTable.relateToOperator("noise",                    EOpNoise); // TODO: check return type
    symbolTable.relateToOperator("normalize",                   EOpNormalize);
    symbolTable.relateToOperator("pow",                         EOpPow);
    // symbolTable.relateToOperator("printf",                     EOpPrintf);
    // symbolTable.relateToOperator("Process2DQuadTessFactorsAvg");
    // symbolTable.relateToOperator("Process2DQuadTessFactorsMax");
    // symbolTable.relateToOperator("Process2DQuadTessFactorsMin");
    // symbolTable.relateToOperator("ProcessIsolineTessFactors");
    // symbolTable.relateToOperator("ProcessQuadTessFactorsAvg");
    // symbolTable.relateToOperator("ProcessQuadTessFactorsMax");
    // symbolTable.relateToOperator("ProcessQuadTessFactorsMin");
    // symbolTable.relateToOperator("ProcessTriTessFactorsAvg");
    // symbolTable.relateToOperator("ProcessTriTessFactorsMax");
    // symbolTable.relateToOperator("ProcessTriTessFactorsMin");
    symbolTable.relateToOperator("radians",                     EOpRadians);
    symbolTable.relateToOperator("rcp",                         EOpRcp);
    symbolTable.relateToOperator("reflect",                     EOpReflect);
    symbolTable.relateToOperator("refract",                     EOpRefract);
    symbolTable.relateToOperator("reversebits",                 EOpBitFieldReverse);
    symbolTable.relateToOperator("round",                       EOpRoundEven);
    symbolTable.relateToOperator("rsqrt",                       EOpInverseSqrt);
    symbolTable.relateToOperator("saturate",                    EOpSaturate);
    symbolTable.relateToOperator("sign",                        EOpSign);
    symbolTable.relateToOperator("sin",                         EOpSin);
    symbolTable.relateToOperator("sincos",                      EOpSinCos);
    symbolTable.relateToOperator("sinh",                        EOpSinh);
    symbolTable.relateToOperator("smoothstep",                  EOpSmoothStep);
    symbolTable.relateToOperator("sqrt",                        EOpSqrt);
    symbolTable.relateToOperator("step",                        EOpStep);
    symbolTable.relateToOperator("tan",                         EOpTan);
    symbolTable.relateToOperator("tanh",                        EOpTanh);
    symbolTable.relateToOperator("tex1D",                       EOpTexture);
    symbolTable.relateToOperator("tex1Dbias",                   EOpTextureBias);
    symbolTable.relateToOperator("tex1Dgrad",                   EOpTextureGrad);
    symbolTable.relateToOperator("tex1Dlod",                    EOpTextureLod);
    symbolTable.relateToOperator("tex1Dproj",                   EOpTextureProj);
    symbolTable.relateToOperator("tex2D",                       EOpTexture);
    symbolTable.relateToOperator("tex2Dbias",                   EOpTextureBias);
    symbolTable.relateToOperator("tex2Dgrad",                   EOpTextureGrad);
    symbolTable.relateToOperator("tex2Dlod",                    EOpTextureLod);
    symbolTable.relateToOperator("tex2Dproj",                   EOpTextureProj);
    symbolTable.relateToOperator("tex3D",                       EOpTexture);
    symbolTable.relateToOperator("tex3Dbias",                   EOpTextureBias);
    symbolTable.relateToOperator("tex3Dgrad",                   EOpTextureGrad);
    symbolTable.relateToOperator("tex3Dlod",                    EOpTextureLod);
    symbolTable.relateToOperator("tex3Dproj",                   EOpTextureProj);
    symbolTable.relateToOperator("texCUBE",                     EOpTexture);
    symbolTable.relateToOperator("texCUBEbias",                 EOpTextureBias);
    symbolTable.relateToOperator("texCUBEgrad",                 EOpTextureGrad);
    symbolTable.relateToOperator("texCUBElod",                  EOpTextureLod);
    symbolTable.relateToOperator("texCUBEproj",                 EOpTextureProj);
    symbolTable.relateToOperator("transpose",                   EOpTranspose);
    symbolTable.relateToOperator("trunc",                       EOpTrunc);

    // Texture methods
    symbolTable.relateToOperator(BUILTIN_PREFIX "Sample",                      EOpMethodSample);
    symbolTable.relateToOperator(BUILTIN_PREFIX "SampleBias",                  EOpMethodSampleBias);
    symbolTable.relateToOperator(BUILTIN_PREFIX "SampleCmp",                   EOpMethodSampleCmp);
    symbolTable.relateToOperator(BUILTIN_PREFIX "SampleCmpLevelZero",          EOpMethodSampleCmpLevelZero);
    symbolTable.relateToOperator(BUILTIN_PREFIX "SampleGrad",                  EOpMethodSampleGrad);
    symbolTable.relateToOperator(BUILTIN_PREFIX "SampleLevel",                 EOpMethodSampleLevel);
    symbolTable.relateToOperator(BUILTIN_PREFIX "Load",                        EOpMethodLoad);
    symbolTable.relateToOperator(BUILTIN_PREFIX "GetDimensions",               EOpMethodGetDimensions);
    symbolTable.relateToOperator(BUILTIN_PREFIX "GetSamplePosition",           EOpMethodGetSamplePosition);
    symbolTable.relateToOperator(BUILTIN_PREFIX "Gather",                      EOpMethodGather);
    symbolTable.relateToOperator(BUILTIN_PREFIX "CalculateLevelOfDetail",      EOpMethodCalculateLevelOfDetail);
    symbolTable.relateToOperator(BUILTIN_PREFIX "CalculateLevelOfDetailUnclamped", EOpMethodCalculateLevelOfDetailUnclamped);

    // Structure buffer methods (excluding associations already made above for texture methods w/ same name)
    symbolTable.relateToOperator(BUILTIN_PREFIX "Load2",                       EOpMethodLoad2);
    symbolTable.relateToOperator(BUILTIN_PREFIX "Load3",                       EOpMethodLoad3);
    symbolTable.relateToOperator(BUILTIN_PREFIX "Load4",                       EOpMethodLoad4);
    symbolTable.relateToOperator(BUILTIN_PREFIX "Store",                       EOpMethodStore);
    symbolTable.relateToOperator(BUILTIN_PREFIX "Store2",                      EOpMethodStore2);
    symbolTable.relateToOperator(BUILTIN_PREFIX "Store3",                      EOpMethodStore3);
    symbolTable.relateToOperator(BUILTIN_PREFIX "Store4",                      EOpMethodStore4);
    symbolTable.relateToOperator(BUILTIN_PREFIX "IncrementCounter",            EOpMethodIncrementCounter);
    symbolTable.relateToOperator(BUILTIN_PREFIX "DecrementCounter",            EOpMethodDecrementCounter);
    // Append is also a GS method: we don't add it twice
    symbolTable.relateToOperator(BUILTIN_PREFIX "Consume",                     EOpMethodConsume);

    symbolTable.relateToOperator(BUILTIN_PREFIX "InterlockedAdd",              EOpInterlockedAdd);
    symbolTable.relateToOperator(BUILTIN_PREFIX "InterlockedAnd",              EOpInterlockedAnd);
    symbolTable.relateToOperator(BUILTIN_PREFIX "InterlockedCompareExchange",  EOpInterlockedCompareExchange);
    symbolTable.relateToOperator(BUILTIN_PREFIX "InterlockedCompareStore",     EOpInterlockedCompareStore);
    symbolTable.relateToOperator(BUILTIN_PREFIX "InterlockedExchange",         EOpInterlockedExchange);
    symbolTable.relateToOperator(BUILTIN_PREFIX "InterlockedMax",              EOpInterlockedMax);
    symbolTable.relateToOperator(BUILTIN_PREFIX "InterlockedMin",              EOpInterlockedMin);
    symbolTable.relateToOperator(BUILTIN_PREFIX "InterlockedOr",               EOpInterlockedOr);
    symbolTable.relateToOperator(BUILTIN_PREFIX "InterlockedXor",              EOpInterlockedXor);

    // SM5 Texture methods
    symbolTable.relateToOperator(BUILTIN_PREFIX "GatherRed",                   EOpMethodGatherRed);
    symbolTable.relateToOperator(BUILTIN_PREFIX "GatherGreen",                 EOpMethodGatherGreen);
    symbolTable.relateToOperator(BUILTIN_PREFIX "GatherBlue",                  EOpMethodGatherBlue);
    symbolTable.relateToOperator(BUILTIN_PREFIX "GatherAlpha",                 EOpMethodGatherAlpha);
    symbolTable.relateToOperator(BUILTIN_PREFIX "GatherCmp",                   EOpMethodGatherCmpRed); // alias
    symbolTable.relateToOperator(BUILTIN_PREFIX "GatherCmpRed",                EOpMethodGatherCmpRed);
    symbolTable.relateToOperator(BUILTIN_PREFIX "GatherCmpGreen",              EOpMethodGatherCmpGreen);
    symbolTable.relateToOperator(BUILTIN_PREFIX "GatherCmpBlue",               EOpMethodGatherCmpBlue);
    symbolTable.relateToOperator(BUILTIN_PREFIX "GatherCmpAlpha",              EOpMethodGatherCmpAlpha);

    // GS methods
    symbolTable.relateToOperator(BUILTIN_PREFIX "Append",                      EOpMethodAppend);
    symbolTable.relateToOperator(BUILTIN_PREFIX "RestartStrip",                EOpMethodRestartStrip);

    // Wave ops
    symbolTable.relateToOperator("WaveIsFirstLane",                            EOpSubgroupElect);
    symbolTable.relateToOperator("WaveGetLaneCount",                           EOpWaveGetLaneCount);
    symbolTable.relateToOperator("WaveGetLaneIndex",                           EOpWaveGetLaneIndex);
    symbolTable.relateToOperator("WaveActiveAnyTrue",                          EOpSubgroupAny);
    symbolTable.relateToOperator("WaveActiveAllTrue",                          EOpSubgroupAll);
    symbolTable.relateToOperator("WaveActiveBallot",                           EOpSubgroupBallot);
    symbolTable.relateToOperator("WaveReadLaneFirst",                          EOpSubgroupBroadcastFirst);
    symbolTable.relateToOperator("WaveReadLaneAt",                             EOpSubgroupShuffle);
    symbolTable.relateToOperator("WaveActiveAllEqual",                         EOpSubgroupAllEqual);
    symbolTable.relateToOperator("WaveActiveAllEqualBool",                     EOpSubgroupAllEqual);
    symbolTable.relateToOperator("WaveActiveCountBits",                        EOpWaveActiveCountBits);
    symbolTable.relateToOperator("WaveActiveSum",                              EOpSubgroupAdd);
    symbolTable.relateToOperator("WaveActiveProduct",                          EOpSubgroupMul);
    symbolTable.relateToOperator("WaveActiveBitAnd",                           EOpSubgroupAnd);
    symbolTable.relateToOperator("WaveActiveBitOr",                            EOpSubgroupOr);
    symbolTable.relateToOperator("WaveActiveBitXor",                           EOpSubgroupXor);
    symbolTable.relateToOperator("WaveActiveMin",                              EOpSubgroupMin);
    symbolTable.relateToOperator("WaveActiveMax",                              EOpSubgroupMax);
    symbolTable.relateToOperator("WavePrefixSum",                              EOpSubgroupInclusiveAdd);
    symbolTable.relateToOperator("WavePrefixProduct",                          EOpSubgroupInclusiveMul);
    symbolTable.relateToOperator("WavePrefixCountBits",                        EOpWavePrefixCountBits);
    symbolTable.relateToOperator("QuadReadAcrossX",                            EOpSubgroupQuadSwapHorizontal);
    symbolTable.relateToOperator("QuadReadAcrossY",                            EOpSubgroupQuadSwapVertical);
    symbolTable.relateToOperator("QuadReadAcrossDiagonal",                     EOpSubgroupQuadSwapDiagonal);
    symbolTable.relateToOperator("QuadReadLaneAt",                             EOpSubgroupQuadBroadcast);

    // Subpass input methods
    symbolTable.relateToOperator(BUILTIN_PREFIX "SubpassLoad",                 EOpSubpassLoad);
    symbolTable.relateToOperator(BUILTIN_PREFIX "SubpassLoadMS",               EOpSubpassLoadMS);
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
void TBuiltInParseablesHlsl::identifyBuiltIns(int /*version*/, EProfile /*profile*/, const SpvVersion& /*spvVersion*/, EShLanguage /*language*/,
                                              TSymbolTable& /*symbolTable*/, const TBuiltInResource& /*resources*/)
{
}

} // end namespace glslang
