//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
// Copyright (C) 2020 Google, Inc.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
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
// GLSL scanning, leveraging the scanning done by the preprocessor.
//

#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include "../Include/Types.h"
#include "SymbolTable.h"
#include "ParseHelper.h"
#include "attribute.h"
#include "glslang_tab.cpp.h"
#include "ScanContext.h"
#include "Scan.h"

// preprocessor includes
#include "preprocessor/PpContext.h"
#include "preprocessor/PpTokens.h"

// Required to avoid missing prototype warnings for some compilers
int yylex(YYSTYPE*, glslang::TParseContext&);

namespace glslang {

// read past any white space
void TInputScanner::consumeWhiteSpace(bool& foundNonSpaceTab)
{
    int c = peek();  // don't accidentally consume anything other than whitespace
    while (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        if (c == '\r' || c == '\n')
            foundNonSpaceTab = true;
        get();
        c = peek();
    }
}

// return true if a comment was actually consumed
bool TInputScanner::consumeComment()
{
    if (peek() != '/')
        return false;

    get();  // consume the '/'
    int c = peek();
    if (c == '/') {

        // a '//' style comment
        get();  // consume the second '/'
        c = get();
        do {
            while (c != EndOfInput && c != '\\' && c != '\r' && c != '\n')
                c = get();

            if (c == EndOfInput || c == '\r' || c == '\n') {
                while (c == '\r' || c == '\n')
                    c = get();

                // we reached the end of the comment
                break;
            } else {
                // it's a '\', so we need to keep going, after skipping what's escaped

                // read the skipped character
                c = get();

                // if it's a two-character newline, skip both characters
                if (c == '\r' && peek() == '\n')
                    get();
                c = get();
            }
        } while (true);

        // put back the last non-comment character
        if (c != EndOfInput)
            unget();

        return true;
    } else if (c == '*') {

        // a '/*' style comment
        get();  // consume the '*'
        c = get();
        do {
            while (c != EndOfInput && c != '*')
                c = get();
            if (c == '*') {
                c = get();
                if (c == '/')
                    break;  // end of comment
                // not end of comment
            } else // end of input
                break;
        } while (true);

        return true;
    } else {
        // it's not a comment, put the '/' back
        unget();

        return false;
    }
}

// skip whitespace, then skip a comment, rinse, repeat
void TInputScanner::consumeWhitespaceComment(bool& foundNonSpaceTab)
{
    do {
        consumeWhiteSpace(foundNonSpaceTab);

        // if not starting a comment now, then done
        int c = peek();
        if (c != '/' || c == EndOfInput)
            return;

        // skip potential comment
        foundNonSpaceTab = true;
        if (! consumeComment())
            return;

    } while (true);
}

// Returns true if there was non-white space (e.g., a comment, newline) before the #version
// or no #version was found; otherwise, returns false.  There is no error case, it always
// succeeds, but will leave version == 0 if no #version was found.
//
// Sets notFirstToken based on whether tokens (beyond white space and comments)
// appeared before the #version.
//
// N.B. does not attempt to leave input in any particular known state.  The assumption
// is that scanning will start anew, following the rules for the chosen version/profile,
// and with a corresponding parsing context.
//
bool TInputScanner::scanVersion(int& version, EProfile& profile, bool& notFirstToken)
{
    // This function doesn't have to get all the semantics correct,
    // just find the #version if there is a correct one present.
    // The preprocessor will have the responsibility of getting all the semantics right.

    bool versionNotFirst = false;  // means not first WRT comments and white space, nothing more
    notFirstToken = false;         // means not first WRT to real tokens
    version = 0;                   // means not found
    profile = ENoProfile;

    bool foundNonSpaceTab = false;
    bool lookingInMiddle = false;
    int c;
    do {
        if (lookingInMiddle) {
            notFirstToken = true;
            // make forward progress by finishing off the current line plus extra new lines
            if (peek() != '\n' && peek() != '\r') {
                do {
                    c = get();
                } while (c != EndOfInput && c != '\n' && c != '\r');
            }
            while (peek() == '\n' || peek() == '\r')
                get();
            if (peek() == EndOfInput)
                return true;
        }
        lookingInMiddle = true;

        // Nominal start, skipping the desktop allowed comments and white space, but tracking if
        // something else was found for ES:
        consumeWhitespaceComment(foundNonSpaceTab);
        if (foundNonSpaceTab)
            versionNotFirst = true;

        // "#"
        if (get() != '#') {
            versionNotFirst = true;
            continue;
        }

        // whitespace
        do {
            c = get();
        } while (c == ' ' || c == '\t');

        // "version"
        if (    c != 'v' ||
            get() != 'e' ||
            get() != 'r' ||
            get() != 's' ||
            get() != 'i' ||
            get() != 'o' ||
            get() != 'n') {
            versionNotFirst = true;
            continue;
        }

        // whitespace
        do {
            c = get();
        } while (c == ' ' || c == '\t');

        // version number
        while (c >= '0' && c <= '9') {
            version = 10 * version + (c - '0');
            c = get();
        }
        if (version == 0) {
            versionNotFirst = true;
            continue;
        }

        // whitespace
        while (c == ' ' || c == '\t')
            c = get();

        // profile
        const int maxProfileLength = 13;  // not including any 0
        char profileString[maxProfileLength];
        int profileLength;
        for (profileLength = 0; profileLength < maxProfileLength; ++profileLength) {
            if (c == EndOfInput || c == ' ' || c == '\t' || c == '\n' || c == '\r')
                break;
            profileString[profileLength] = (char)c;
            c = get();
        }
        if (c != EndOfInput && c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            versionNotFirst = true;
            continue;
        }

        if (profileLength == 2 && strncmp(profileString, "es", profileLength) == 0)
            profile = EEsProfile;
        else if (profileLength == 4 && strncmp(profileString, "core", profileLength) == 0)
            profile = ECoreProfile;
        else if (profileLength == 13 && strncmp(profileString, "compatibility", profileLength) == 0)
            profile = ECompatibilityProfile;

        return versionNotFirst;
    } while (true);
}

// Fill this in when doing glslang-level scanning, to hand back to the parser.
class TParserToken {
public:
    explicit TParserToken(YYSTYPE& b) : sType(b) { }

    YYSTYPE& sType;
protected:
    TParserToken(TParserToken&);
    TParserToken& operator=(TParserToken&);
};

} // end namespace glslang

// This is the function the glslang parser (i.e., bison) calls to get its next token
int yylex(YYSTYPE* glslangTokenDesc, glslang::TParseContext& parseContext)
{
    glslang::TParserToken token(*glslangTokenDesc);

    return parseContext.getScanContext()->tokenize(parseContext.getPpContext(), token);
}

namespace {

struct str_eq
{
    bool operator()(const char* lhs, const char* rhs) const
    {
        return strcmp(lhs, rhs) == 0;
    }
};

struct str_hash
{
    size_t operator()(const char* str) const
    {
        // djb2
        unsigned long hash = 5381;
        int c;

        while ((c = *str++) != 0)
            hash = ((hash << 5) + hash) + c;

        return hash;
    }
};

// A single global usable by all threads, by all versions, by all languages.
const std::unordered_map<const char*, int, str_hash, str_eq> KeywordMap {
    {"const",CONST},
    {"uniform",UNIFORM},
    {"tileImageEXT",TILEIMAGEEXT},
    {"buffer",BUFFER},
    {"in",IN},
    {"out",OUT},
    {"smooth",SMOOTH},
    {"flat",FLAT},
    {"centroid",CENTROID},
    {"invariant",INVARIANT},
    {"packed",PACKED},
    {"resource",RESOURCE},
    {"inout",INOUT},
    {"struct",STRUCT},
    {"break",BREAK},
    {"continue",CONTINUE},
    {"do",DO},
    {"for",FOR},
    {"while",WHILE},
    {"switch",SWITCH},
    {"case",CASE},
    {"default",DEFAULT},
    {"if",IF},
    {"else",ELSE},
    {"discard",DISCARD},
    {"terminateInvocation",TERMINATE_INVOCATION},
    {"terminateRayEXT",TERMINATE_RAY},
    {"ignoreIntersectionEXT",IGNORE_INTERSECTION},
    {"return",RETURN},
    {"void",VOID},
    {"bool",BOOL},
    {"float",FLOAT},
    {"int",INT},
    {"bvec2",BVEC2},
    {"bvec3",BVEC3},
    {"bvec4",BVEC4},
    {"vec2",VEC2},
    {"vec3",VEC3},
    {"vec4",VEC4},
    {"ivec2",IVEC2},
    {"ivec3",IVEC3},
    {"ivec4",IVEC4},
    {"mat2",MAT2},
    {"mat3",MAT3},
    {"mat4",MAT4},
    {"true",BOOLCONSTANT},
    {"false",BOOLCONSTANT},
    {"layout",LAYOUT},
    {"shared",SHARED},
    {"highp",HIGH_PRECISION},
    {"mediump",MEDIUM_PRECISION},
    {"lowp",LOW_PRECISION},
    {"superp",SUPERP},
    {"precision",PRECISION},
    {"mat2x2",MAT2X2},
    {"mat2x3",MAT2X3},
    {"mat2x4",MAT2X4},
    {"mat3x2",MAT3X2},
    {"mat3x3",MAT3X3},
    {"mat3x4",MAT3X4},
    {"mat4x2",MAT4X2},
    {"mat4x3",MAT4X3},
    {"mat4x4",MAT4X4},
    {"uint",UINT},
    {"uvec2",UVEC2},
    {"uvec3",UVEC3},
    {"uvec4",UVEC4},

    {"nonuniformEXT",NONUNIFORM},
    {"demote",DEMOTE},
    {"attribute",ATTRIBUTE},
    {"varying",VARYING},
    {"noperspective",NOPERSPECTIVE},
    {"coherent",COHERENT},
    {"devicecoherent",DEVICECOHERENT},
    {"queuefamilycoherent",QUEUEFAMILYCOHERENT},
    {"workgroupcoherent",WORKGROUPCOHERENT},
    {"subgroupcoherent",SUBGROUPCOHERENT},
    {"shadercallcoherent",SHADERCALLCOHERENT},
    {"nonprivate",NONPRIVATE},
    {"restrict",RESTRICT},
    {"readonly",READONLY},
    {"writeonly",WRITEONLY},
    {"atomic_uint",ATOMIC_UINT},
    {"volatile",VOLATILE},
    {"nontemporal",NONTEMPORAL},
    {"patch",PATCH},
    {"sample",SAMPLE},
    {"subroutine",SUBROUTINE},
    {"dmat2",DMAT2},
    {"dmat3",DMAT3},
    {"dmat4",DMAT4},
    {"dmat2x2",DMAT2X2},
    {"dmat2x3",DMAT2X3},
    {"dmat2x4",DMAT2X4},
    {"dmat3x2",DMAT3X2},
    {"dmat3x3",DMAT3X3},
    {"dmat3x4",DMAT3X4},
    {"dmat4x2",DMAT4X2},
    {"dmat4x3",DMAT4X3},
    {"dmat4x4",DMAT4X4},
    {"image1D",IMAGE1D},
    {"iimage1D",IIMAGE1D},
    {"uimage1D",UIMAGE1D},
    {"image2D",IMAGE2D},
    {"iimage2D",IIMAGE2D},
    {"uimage2D",UIMAGE2D},
    {"image3D",IMAGE3D},
    {"iimage3D",IIMAGE3D},
    {"uimage3D",UIMAGE3D},
    {"image2DRect",IMAGE2DRECT},
    {"iimage2DRect",IIMAGE2DRECT},
    {"uimage2DRect",UIMAGE2DRECT},
    {"imageCube",IMAGECUBE},
    {"iimageCube",IIMAGECUBE},
    {"uimageCube",UIMAGECUBE},
    {"imageBuffer",IMAGEBUFFER},
    {"iimageBuffer",IIMAGEBUFFER},
    {"uimageBuffer",UIMAGEBUFFER},
    {"image1DArray",IMAGE1DARRAY},
    {"iimage1DArray",IIMAGE1DARRAY},
    {"uimage1DArray",UIMAGE1DARRAY},
    {"image2DArray",IMAGE2DARRAY},
    {"iimage2DArray",IIMAGE2DARRAY},
    {"uimage2DArray",UIMAGE2DARRAY},
    {"imageCubeArray",IMAGECUBEARRAY},
    {"iimageCubeArray",IIMAGECUBEARRAY},
    {"uimageCubeArray",UIMAGECUBEARRAY},
    {"image2DMS",IMAGE2DMS},
    {"iimage2DMS",IIMAGE2DMS},
    {"uimage2DMS",UIMAGE2DMS},
    {"image2DMSArray",IMAGE2DMSARRAY},
    {"iimage2DMSArray",IIMAGE2DMSARRAY},
    {"uimage2DMSArray",UIMAGE2DMSARRAY},
    {"i64image1D",I64IMAGE1D},
    {"u64image1D",U64IMAGE1D},
    {"i64image2D",I64IMAGE2D},
    {"u64image2D",U64IMAGE2D},
    {"i64image3D",I64IMAGE3D},
    {"u64image3D",U64IMAGE3D},
    {"i64image2DRect",I64IMAGE2DRECT},
    {"u64image2DRect",U64IMAGE2DRECT},
    {"i64imageCube",I64IMAGECUBE},
    {"u64imageCube",U64IMAGECUBE},
    {"i64imageBuffer",I64IMAGEBUFFER},
    {"u64imageBuffer",U64IMAGEBUFFER},
    {"i64image1DArray",I64IMAGE1DARRAY},
    {"u64image1DArray",U64IMAGE1DARRAY},
    {"i64image2DArray",I64IMAGE2DARRAY},
    {"u64image2DArray",U64IMAGE2DARRAY},
    {"i64imageCubeArray",I64IMAGECUBEARRAY},
    {"u64imageCubeArray",U64IMAGECUBEARRAY},
    {"i64image2DMS",I64IMAGE2DMS},
    {"u64image2DMS",U64IMAGE2DMS},
    {"i64image2DMSArray",I64IMAGE2DMSARRAY},
    {"u64image2DMSArray",U64IMAGE2DMSARRAY},
    {"double",DOUBLE},
    {"dvec2",DVEC2},
    {"dvec3",DVEC3},
    {"dvec4",DVEC4},
    {"int64_t",INT64_T},
    {"uint64_t",UINT64_T},
    {"i64vec2",I64VEC2},
    {"i64vec3",I64VEC3},
    {"i64vec4",I64VEC4},
    {"u64vec2",U64VEC2},
    {"u64vec3",U64VEC3},
    {"u64vec4",U64VEC4},

    // GL_EXT_shader_explicit_arithmetic_types
    {"int8_t",INT8_T},
    {"i8vec2",I8VEC2},
    {"i8vec3",I8VEC3},
    {"i8vec4",I8VEC4},
    {"uint8_t",UINT8_T},
    {"u8vec2",U8VEC2},
    {"u8vec3",U8VEC3},
    {"u8vec4",U8VEC4},

    {"int16_t",INT16_T},
    {"i16vec2",I16VEC2},
    {"i16vec3",I16VEC3},
    {"i16vec4",I16VEC4},
    {"uint16_t",UINT16_T},
    {"u16vec2",U16VEC2},
    {"u16vec3",U16VEC3},
    {"u16vec4",U16VEC4},

    {"int32_t",INT32_T},
    {"i32vec2",I32VEC2},
    {"i32vec3",I32VEC3},
    {"i32vec4",I32VEC4},
    {"uint32_t",UINT32_T},
    {"u32vec2",U32VEC2},
    {"u32vec3",U32VEC3},
    {"u32vec4",U32VEC4},

    {"float16_t",FLOAT16_T},
    {"f16vec2",F16VEC2},
    {"f16vec3",F16VEC3},
    {"f16vec4",F16VEC4},
    {"f16mat2",F16MAT2},
    {"f16mat3",F16MAT3},
    {"f16mat4",F16MAT4},
    {"f16mat2x2",F16MAT2X2},
    {"f16mat2x3",F16MAT2X3},
    {"f16mat2x4",F16MAT2X4},
    {"f16mat3x2",F16MAT3X2},
    {"f16mat3x3",F16MAT3X3},
    {"f16mat3x4",F16MAT3X4},
    {"f16mat4x2",F16MAT4X2},
    {"f16mat4x3",F16MAT4X3},
    {"f16mat4x4",F16MAT4X4},

    {"bfloat16_t",BFLOAT16_T},
    {"bf16vec2",BF16VEC2},
    {"bf16vec3",BF16VEC3},
    {"bf16vec4",BF16VEC4},

    {"floate5m2_t",FLOATE5M2_T},
    {"fe5m2vec2",FE5M2VEC2},
    {"fe5m2vec3",FE5M2VEC3},
    {"fe5m2vec4",FE5M2VEC4},

    {"floate4m3_t",FLOATE4M3_T},
    {"fe4m3vec2",FE4M3VEC2},
    {"fe4m3vec3",FE4M3VEC3},
    {"fe4m3vec4",FE4M3VEC4},

    {"float32_t",FLOAT32_T},
    {"f32vec2",F32VEC2},
    {"f32vec3",F32VEC3},
    {"f32vec4",F32VEC4},
    {"f32mat2",F32MAT2},
    {"f32mat3",F32MAT3},
    {"f32mat4",F32MAT4},
    {"f32mat2x2",F32MAT2X2},
    {"f32mat2x3",F32MAT2X3},
    {"f32mat2x4",F32MAT2X4},
    {"f32mat3x2",F32MAT3X2},
    {"f32mat3x3",F32MAT3X3},
    {"f32mat3x4",F32MAT3X4},
    {"f32mat4x2",F32MAT4X2},
    {"f32mat4x3",F32MAT4X3},
    {"f32mat4x4",F32MAT4X4},
    {"float64_t",FLOAT64_T},
    {"f64vec2",F64VEC2},
    {"f64vec3",F64VEC3},
    {"f64vec4",F64VEC4},
    {"f64mat2",F64MAT2},
    {"f64mat3",F64MAT3},
    {"f64mat4",F64MAT4},
    {"f64mat2x2",F64MAT2X2},
    {"f64mat2x3",F64MAT2X3},
    {"f64mat2x4",F64MAT2X4},
    {"f64mat3x2",F64MAT3X2},
    {"f64mat3x3",F64MAT3X3},
    {"f64mat3x4",F64MAT3X4},
    {"f64mat4x2",F64MAT4X2},
    {"f64mat4x3",F64MAT4X3},
    {"f64mat4x4",F64MAT4X4},

    // GL_EXT_spirv_intrinsics
    {"spirv_instruction",SPIRV_INSTRUCTION},
    {"spirv_execution_mode",SPIRV_EXECUTION_MODE},
    {"spirv_execution_mode_id",SPIRV_EXECUTION_MODE_ID},
    {"spirv_decorate",SPIRV_DECORATE},
    {"spirv_decorate_id",SPIRV_DECORATE_ID},
    {"spirv_decorate_string",SPIRV_DECORATE_STRING},
    {"spirv_type",SPIRV_TYPE},
    {"spirv_storage_class",SPIRV_STORAGE_CLASS},
    {"spirv_by_reference",SPIRV_BY_REFERENCE},
    {"spirv_literal",SPIRV_LITERAL},

    {"sampler2D",SAMPLER2D},
    {"samplerCube",SAMPLERCUBE},
    {"samplerCubeShadow",SAMPLERCUBESHADOW},
    {"sampler2DArray",SAMPLER2DARRAY},
    {"sampler2DArrayShadow",SAMPLER2DARRAYSHADOW},
    {"isampler2D",ISAMPLER2D},
    {"isampler3D",ISAMPLER3D},
    {"isamplerCube",ISAMPLERCUBE},
    {"isampler2DArray",ISAMPLER2DARRAY},
    {"usampler2D",USAMPLER2D},
    {"usampler3D",USAMPLER3D},
    {"usamplerCube",USAMPLERCUBE},
    {"usampler2DArray",USAMPLER2DARRAY},
    {"sampler3D",SAMPLER3D},
    {"sampler2DShadow",SAMPLER2DSHADOW},

    {"texture2D",TEXTURE2D},
    {"textureCube",TEXTURECUBE},
    {"texture2DArray",TEXTURE2DARRAY},
    {"itexture2D",ITEXTURE2D},
    {"itexture3D",ITEXTURE3D},
    {"itextureCube",ITEXTURECUBE},
    {"itexture2DArray",ITEXTURE2DARRAY},
    {"utexture2D",UTEXTURE2D},
    {"utexture3D",UTEXTURE3D},
    {"utextureCube",UTEXTURECUBE},
    {"utexture2DArray",UTEXTURE2DARRAY},
    {"texture3D",TEXTURE3D},

    {"sampler",SAMPLER},
    {"samplerShadow",SAMPLERSHADOW},

    {"textureCubeArray",TEXTURECUBEARRAY},
    {"itextureCubeArray",ITEXTURECUBEARRAY},
    {"utextureCubeArray",UTEXTURECUBEARRAY},
    {"samplerCubeArray",SAMPLERCUBEARRAY},
    {"samplerCubeArrayShadow",SAMPLERCUBEARRAYSHADOW},
    {"isamplerCubeArray",ISAMPLERCUBEARRAY},
    {"usamplerCubeArray",USAMPLERCUBEARRAY},
    {"sampler1DArrayShadow",SAMPLER1DARRAYSHADOW},
    {"isampler1DArray",ISAMPLER1DARRAY},
    {"usampler1D",USAMPLER1D},
    {"isampler1D",ISAMPLER1D},
    {"usampler1DArray",USAMPLER1DARRAY},
    {"samplerBuffer",SAMPLERBUFFER},
    {"isampler2DRect",ISAMPLER2DRECT},
    {"usampler2DRect",USAMPLER2DRECT},
    {"isamplerBuffer",ISAMPLERBUFFER},
    {"usamplerBuffer",USAMPLERBUFFER},
    {"sampler2DMS",SAMPLER2DMS},
    {"isampler2DMS",ISAMPLER2DMS},
    {"usampler2DMS",USAMPLER2DMS},
    {"sampler2DMSArray",SAMPLER2DMSARRAY},
    {"isampler2DMSArray",ISAMPLER2DMSARRAY},
    {"usampler2DMSArray",USAMPLER2DMSARRAY},
    {"sampler1D",SAMPLER1D},
    {"sampler1DShadow",SAMPLER1DSHADOW},
    {"sampler2DRect",SAMPLER2DRECT},
    {"sampler2DRectShadow",SAMPLER2DRECTSHADOW},
    {"sampler1DArray",SAMPLER1DARRAY},

    {"samplerExternalOES",     SAMPLEREXTERNALOES}, // GL_OES_EGL_image_external
    {"__samplerExternal2DY2YEXT", SAMPLEREXTERNAL2DY2YEXT}, // GL_EXT_YUV_target

    {"itexture1DArray",ITEXTURE1DARRAY},
    {"utexture1D",UTEXTURE1D},
    {"itexture1D",ITEXTURE1D},
    {"utexture1DArray",UTEXTURE1DARRAY},
    {"textureBuffer",TEXTUREBUFFER},
    {"itexture2DRect",ITEXTURE2DRECT},
    {"utexture2DRect",UTEXTURE2DRECT},
    {"itextureBuffer",ITEXTUREBUFFER},
    {"utextureBuffer",UTEXTUREBUFFER},
    {"texture2DMS",TEXTURE2DMS},
    {"itexture2DMS",ITEXTURE2DMS},
    {"utexture2DMS",UTEXTURE2DMS},
    {"texture2DMSArray",TEXTURE2DMSARRAY},
    {"itexture2DMSArray",ITEXTURE2DMSARRAY},
    {"utexture2DMSArray",UTEXTURE2DMSARRAY},
    {"texture1D",TEXTURE1D},
    {"texture2DRect",TEXTURE2DRECT},
    {"texture1DArray",TEXTURE1DARRAY},

    {"attachmentEXT",ATTACHMENTEXT},
    {"iattachmentEXT",IATTACHMENTEXT},
    {"uattachmentEXT",UATTACHMENTEXT},

    {"subpassInput",SUBPASSINPUT},
    {"subpassInputMS",SUBPASSINPUTMS},
    {"isubpassInput",ISUBPASSINPUT},
    {"isubpassInputMS",ISUBPASSINPUTMS},
    {"usubpassInput",USUBPASSINPUT},
    {"usubpassInputMS",USUBPASSINPUTMS},

    {"f16sampler1D",F16SAMPLER1D},
    {"f16sampler2D",F16SAMPLER2D},
    {"f16sampler3D",F16SAMPLER3D},
    {"f16sampler2DRect",F16SAMPLER2DRECT},
    {"f16samplerCube",F16SAMPLERCUBE},
    {"f16sampler1DArray",F16SAMPLER1DARRAY},
    {"f16sampler2DArray",F16SAMPLER2DARRAY},
    {"f16samplerCubeArray",F16SAMPLERCUBEARRAY},
    {"f16samplerBuffer",F16SAMPLERBUFFER},
    {"f16sampler2DMS",F16SAMPLER2DMS},
    {"f16sampler2DMSArray",F16SAMPLER2DMSARRAY},
    {"f16sampler1DShadow",F16SAMPLER1DSHADOW},
    {"f16sampler2DShadow",F16SAMPLER2DSHADOW},
    {"f16sampler2DRectShadow",F16SAMPLER2DRECTSHADOW},
    {"f16samplerCubeShadow",F16SAMPLERCUBESHADOW},
    {"f16sampler1DArrayShadow",F16SAMPLER1DARRAYSHADOW},
    {"f16sampler2DArrayShadow",F16SAMPLER2DARRAYSHADOW},
    {"f16samplerCubeArrayShadow",F16SAMPLERCUBEARRAYSHADOW},

    {"f16image1D",F16IMAGE1D},
    {"f16image2D",F16IMAGE2D},
    {"f16image3D",F16IMAGE3D},
    {"f16image2DRect",F16IMAGE2DRECT},
    {"f16imageCube",F16IMAGECUBE},
    {"f16image1DArray",F16IMAGE1DARRAY},
    {"f16image2DArray",F16IMAGE2DARRAY},
    {"f16imageCubeArray",F16IMAGECUBEARRAY},
    {"f16imageBuffer",F16IMAGEBUFFER},
    {"f16image2DMS",F16IMAGE2DMS},
    {"f16image2DMSArray",F16IMAGE2DMSARRAY},

    {"f16texture1D",F16TEXTURE1D},
    {"f16texture2D",F16TEXTURE2D},
    {"f16texture3D",F16TEXTURE3D},
    {"f16texture2DRect",F16TEXTURE2DRECT},
    {"f16textureCube",F16TEXTURECUBE},
    {"f16texture1DArray",F16TEXTURE1DARRAY},
    {"f16texture2DArray",F16TEXTURE2DARRAY},
    {"f16textureCubeArray",F16TEXTURECUBEARRAY},
    {"f16textureBuffer",F16TEXTUREBUFFER},
    {"f16texture2DMS",F16TEXTURE2DMS},
    {"f16texture2DMSArray",F16TEXTURE2DMSARRAY},

    {"f16subpassInput",F16SUBPASSINPUT},
    {"f16subpassInputMS",F16SUBPASSINPUTMS},
    {"__explicitInterpAMD",EXPLICITINTERPAMD},
    {"pervertexNV",PERVERTEXNV},
    {"pervertexEXT",PERVERTEXEXT},
    {"precise",PRECISE},

    {"rayPayloadNV",PAYLOADNV},
    {"rayPayloadEXT",PAYLOADEXT},
    {"rayPayloadInNV",PAYLOADINNV},
    {"rayPayloadInEXT",PAYLOADINEXT},
    {"hitAttributeNV",HITATTRNV},
    {"hitAttributeEXT",HITATTREXT},
    {"callableDataNV",CALLDATANV},
    {"callableDataEXT",CALLDATAEXT},
    {"callableDataInNV",CALLDATAINNV},
    {"callableDataInEXT",CALLDATAINEXT},
    {"accelerationStructureNV",ACCSTRUCTNV},
    {"accelerationStructureEXT",ACCSTRUCTEXT},
    {"rayQueryEXT",RAYQUERYEXT},
    {"perprimitiveNV",PERPRIMITIVENV},
    {"perviewNV",PERVIEWNV},
    {"taskNV",PERTASKNV},
    {"perprimitiveEXT",PERPRIMITIVEEXT},
    {"taskPayloadSharedEXT",TASKPAYLOADWORKGROUPEXT},

    {"fcoopmatNV",FCOOPMATNV},
    {"icoopmatNV",ICOOPMATNV},
    {"ucoopmatNV",UCOOPMATNV},

    {"coopmat",COOPMAT},

    {"hitObjectNV",HITOBJECTNV},
    {"hitObjectAttributeNV",HITOBJECTATTRNV},

    {"tensorARM",TENSORARM},

    {"hitObjectEXT",HITOBJECTEXT},
    {"hitObjectAttributeEXT",HITOBJECTATTREXT},

    {"__function",FUNCTION},
    {"tensorLayoutNV",TENSORLAYOUTNV},
    {"tensorViewNV",TENSORVIEWNV},

    {"coopvecNV",COOPVECNV},
};
const std::unordered_set<const char*, str_hash, str_eq> ReservedSet {
    "common",
    "partition",
    "active",
    "asm",
    "class",
    "union",
    "enum",
    "typedef",
    "template",
    "this",
    "goto",
    "inline",
    "noinline",
    "public",
    "static",
    "extern",
    "external",
    "interface",
    "long",
    "short",
    "half",
    "fixed",
    "unsigned",
    "input",
    "output",
    "hvec2",
    "hvec3",
    "hvec4",
    "fvec2",
    "fvec3",
    "fvec4",
    "sampler3DRect",
    "filter",
    "sizeof",
    "cast",
    "namespace",
    "using",
};

}

namespace glslang {

// Called by yylex to get the next token.
// Returning 0 implies end of input.
int TScanContext::tokenize(TPpContext* pp, TParserToken& token)
{
    do {
        parserToken = &token;
        TPpToken ppToken;
        int token = pp->tokenize(ppToken);
        if (token == EndOfInput)
            return 0;

        tokenText = ppToken.name;
        loc = ppToken.loc;
        parserToken->sType.lex.loc = loc;
        switch (token) {
        case ';':  afterType = false; afterBuffer = false; inDeclaratorList = false; afterDeclarator = false; angleBracketDepth = 0; squareBracketDepth = 0; parenDepth = 0; return SEMICOLON;
        case ',':
            // If we just processed a declarator (identifier after a type), this comma
            // indicates that we're in a declarator list. Note that 'afterDeclarator' is
            // only set when we are not inside a template parameter list, array expression,
            // or function parameter list.
            if (afterDeclarator) {
                inDeclaratorList = true;
            }
            afterType = false;
            afterDeclarator = false;
            return COMMA;
        case ':':                       return COLON;
        case '=':  afterType = false; inDeclaratorList = false; afterDeclarator = false; return EQUAL;
        case '(':  afterType = false; inDeclaratorList = false; afterDeclarator = false; parenDepth++; return LEFT_PAREN;
        case ')':  afterType = false; inDeclaratorList = false; afterDeclarator = false; if (parenDepth > 0) parenDepth--; return RIGHT_PAREN;
        case '.':  field = true;        return DOT;
        case '!':                       return BANG;
        case '-':                       return DASH;
        case '~':                       return TILDE;
        case '+':                       return PLUS;
        case '*':                       return STAR;
        case '/':                       return SLASH;
        case '%':                       return PERCENT;
        case '<':                       angleBracketDepth++; return LEFT_ANGLE;
        case '>':                       if (angleBracketDepth > 0) angleBracketDepth--; return RIGHT_ANGLE;
        case '|':                       return VERTICAL_BAR;
        case '^':                       return CARET;
        case '&':                       return AMPERSAND;
        case '?':                       return QUESTION;
        case '[':                       squareBracketDepth++; return LEFT_BRACKET;
        case ']':                       if (squareBracketDepth > 0) squareBracketDepth--; return RIGHT_BRACKET;
        case '{':  afterStruct = false; afterBuffer = false; inDeclaratorList = false; afterDeclarator = false; angleBracketDepth = 0; squareBracketDepth = 0; parenDepth = 0; return LEFT_BRACE;
        case '}':  inDeclaratorList = false; afterDeclarator = false; angleBracketDepth = 0; squareBracketDepth = 0; parenDepth = 0; return RIGHT_BRACE;
        case '\\':
            parseContext.error(loc, "illegal use of escape character", "\\", "");
            break;

        case PPAtomAddAssign:          return ADD_ASSIGN;
        case PPAtomSubAssign:          return SUB_ASSIGN;
        case PPAtomMulAssign:          return MUL_ASSIGN;
        case PPAtomDivAssign:          return DIV_ASSIGN;
        case PPAtomModAssign:          return MOD_ASSIGN;

        case PpAtomRight:              return RIGHT_OP;
        case PpAtomLeft:               return LEFT_OP;

        case PpAtomRightAssign:        return RIGHT_ASSIGN;
        case PpAtomLeftAssign:         return LEFT_ASSIGN;
        case PpAtomAndAssign:          return AND_ASSIGN;
        case PpAtomOrAssign:           return OR_ASSIGN;
        case PpAtomXorAssign:          return XOR_ASSIGN;

        case PpAtomAnd:                return AND_OP;
        case PpAtomOr:                 return OR_OP;
        case PpAtomXor:                return XOR_OP;

        case PpAtomEQ:                 return EQ_OP;
        case PpAtomGE:                 return GE_OP;
        case PpAtomNE:                 return NE_OP;
        case PpAtomLE:                 return LE_OP;

        case PpAtomDecrement:          return DEC_OP;
        case PpAtomIncrement:          return INC_OP;

        case PpAtomColonColon:
            parseContext.error(loc, "not supported", "::", "");
            break;

        case PpAtomConstString:        parserToken->sType.lex.string = NewPoolTString(tokenText);     return STRING_LITERAL;
        case PpAtomConstInt:           parserToken->sType.lex.i    = ppToken.ival;       return INTCONSTANT;
        case PpAtomConstUint:          parserToken->sType.lex.i    = ppToken.ival;       return UINTCONSTANT;
        case PpAtomConstFloat:         parserToken->sType.lex.d    = ppToken.dval;       return FLOATCONSTANT;
        case PpAtomConstInt16:         parserToken->sType.lex.i    = ppToken.ival;       return INT16CONSTANT;
        case PpAtomConstUint16:        parserToken->sType.lex.i    = ppToken.ival;       return UINT16CONSTANT;
        case PpAtomConstInt64:         parserToken->sType.lex.i64  = ppToken.i64val;     return INT64CONSTANT;
        case PpAtomConstUint64:        parserToken->sType.lex.i64  = ppToken.i64val;     return UINT64CONSTANT;
        case PpAtomConstDouble:        parserToken->sType.lex.d    = ppToken.dval;       return DOUBLECONSTANT;
        case PpAtomConstFloat16:       parserToken->sType.lex.d    = ppToken.dval;       return FLOAT16CONSTANT;
        case PpAtomIdentifier:
        {
            int token = tokenizeIdentifier();
            field = false;
            return token;
        }

        case EndOfInput:               return 0;

        default:
            char buf[2];
            buf[0] = (char)token;
            buf[1] = 0;
            parseContext.error(loc, "unexpected token", buf, "");
            break;
        }
    } while (true);
}

int TScanContext::tokenizeIdentifier()
{
    if (ReservedSet.find(tokenText) != ReservedSet.end())
        return reservedWord();

    auto it = KeywordMap.find(tokenText);
    if (it == KeywordMap.end()) {
        // Should have an identifier of some sort
        return identifierOrType();
    }
    keyword = it->second;

    switch (keyword) {
    case CONST:
    case UNIFORM:
    case TILEIMAGEEXT:
    case IN:
    case OUT:
    case INOUT:
    case BREAK:
    case CONTINUE:
    case DO:
    case FOR:
    case WHILE:
    case IF:
    case ELSE:
    case DISCARD:
    case RETURN:
    case CASE:
        return keyword;

    case TERMINATE_INVOCATION:
        if (!parseContext.extensionTurnedOn(E_GL_EXT_terminate_invocation))
            return identifierOrType();
        return keyword;

    case TERMINATE_RAY:
    case IGNORE_INTERSECTION:
        if (!parseContext.extensionTurnedOn(E_GL_EXT_ray_tracing))
            return identifierOrType();
        return keyword;

    case BUFFER:
        afterBuffer = true;
        if ((parseContext.isEsProfile() && parseContext.version < 310) ||
            (!parseContext.isEsProfile() && (parseContext.version < 430 &&
            !parseContext.extensionTurnedOn(E_GL_ARB_shader_storage_buffer_object))))
            return identifierOrType();
        return keyword;

    case STRUCT:
        afterStruct = true;
        return keyword;

    case SWITCH:
    case DEFAULT:
        if ((parseContext.isEsProfile() && parseContext.version < 300) ||
            (!parseContext.isEsProfile() && parseContext.version < 130))
            reservedWord();
        return keyword;

    case VOID:
    case BOOL:
    case FLOAT:
    case INT:
    case BVEC2:
    case BVEC3:
    case BVEC4:
    case VEC2:
    case VEC3:
    case VEC4:
    case IVEC2:
    case IVEC3:
    case IVEC4:
    case MAT2:
    case MAT3:
    case MAT4:
    case SAMPLER2D:
    case SAMPLERCUBE:
        afterType = true;
        return keyword;

    case BOOLCONSTANT:
        if (strcmp("true", tokenText) == 0)
            parserToken->sType.lex.b = true;
        else
            parserToken->sType.lex.b = false;
        return keyword;

    case SMOOTH:
        if ((parseContext.isEsProfile() && parseContext.version < 300) ||
            (!parseContext.isEsProfile() && parseContext.version < 130))
            return identifierOrType();
        return keyword;
    case FLAT:
        if (parseContext.isEsProfile() && parseContext.version < 300)
            reservedWord();
        else if (!parseContext.isEsProfile() && parseContext.version < 130)
            return identifierOrType();
        return keyword;
    case CENTROID:
        if (parseContext.version < 120)
            return identifierOrType();
        return keyword;
    case INVARIANT:
        if (!parseContext.isEsProfile() && parseContext.version < 120)
            return identifierOrType();
        return keyword;
    case PACKED:
        if ((parseContext.isEsProfile() && parseContext.version < 300) ||
            (!parseContext.isEsProfile() && parseContext.version < 140))
            return reservedWord();
        return identifierOrType();

    case RESOURCE:
    {
        bool reserved = (parseContext.isEsProfile() && parseContext.version >= 300) ||
                        (!parseContext.isEsProfile() && parseContext.version >= 420);
        return identifierOrReserved(reserved);
    }
    case SUPERP:
    {
        bool reserved = parseContext.isEsProfile() || parseContext.version >= 130;
        return identifierOrReserved(reserved);
    }

    case NOPERSPECTIVE:
        if (parseContext.extensionTurnedOn(E_GL_NV_shader_noperspective_interpolation))
            return keyword;
        return es30ReservedFromGLSL(130);

    case NONUNIFORM:
        if (parseContext.extensionTurnedOn(E_GL_EXT_nonuniform_qualifier))
            return keyword;
        else
            return identifierOrType();
    case ATTRIBUTE:
    case VARYING:
        if (parseContext.isEsProfile() && parseContext.version >= 300)
            reservedWord();
        return keyword;
    case PAYLOADNV:
    case PAYLOADINNV:
    case HITATTRNV:
    case CALLDATANV:
    case CALLDATAINNV:
    case ACCSTRUCTNV:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_NV_ray_tracing))
            return keyword;
        return identifierOrType();
    case ACCSTRUCTEXT:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_ray_tracing) ||
            parseContext.extensionTurnedOn(E_GL_EXT_ray_query) ||
            parseContext.extensionTurnedOn(E_GL_NV_displacement_micromap))
            return keyword;
        return identifierOrType();
    case PAYLOADEXT:
    case PAYLOADINEXT:
    case HITATTREXT:
    case CALLDATAEXT:
    case CALLDATAINEXT:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_ray_tracing) ||
            parseContext.extensionTurnedOn(E_GL_EXT_ray_query))
            return keyword;
        return identifierOrType();
    case RAYQUERYEXT:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            (!parseContext.isEsProfile() && parseContext.version >= 460
                 && parseContext.extensionTurnedOn(E_GL_EXT_ray_query)))
            return keyword;
        return identifierOrType();
    case ATOMIC_UINT:
        if ((parseContext.isEsProfile() && parseContext.version >= 310) ||
            parseContext.extensionTurnedOn(E_GL_ARB_shader_atomic_counters))
            return keyword;
        return es30ReservedFromGLSL(420);

    case COHERENT:
    case DEVICECOHERENT:
    case QUEUEFAMILYCOHERENT:
    case WORKGROUPCOHERENT:
    case SUBGROUPCOHERENT:
    case SHADERCALLCOHERENT:
    case NONPRIVATE:
    case RESTRICT:
    case READONLY:
    case WRITEONLY:
        if (parseContext.isEsProfile() && parseContext.version >= 310)
            return keyword;
        return es30ReservedFromGLSL(parseContext.extensionTurnedOn(E_GL_ARB_shader_image_load_store) ? 130 : 420);
    case VOLATILE:
        if (parseContext.isEsProfile() && parseContext.version >= 310)
            return keyword;
        if (! parseContext.symbolTable.atBuiltInLevel() && (parseContext.isEsProfile() ||
            (parseContext.version < 420 && ! parseContext.extensionTurnedOn(E_GL_ARB_shader_image_load_store))))
            reservedWord();
        return keyword;
    case NONTEMPORAL:
        if (parseContext.symbolTable.atBuiltInLevel())
            return keyword;
        if (parseContext.extensionTurnedOn(E_GL_EXT_nontemporal_keyword)) {
            if (!parseContext.intermediate.usingVulkanMemoryModel())
                parseContext.warn(loc, "Nontemporal without the Vulkan Memory Model is ignored", tokenText, "");
            return keyword;
        }
        return identifierOrType();
    case PATCH:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            (parseContext.isEsProfile() &&
             (parseContext.version >= 320 ||
              parseContext.extensionsTurnedOn(Num_AEP_tessellation_shader, AEP_tessellation_shader))) ||
            (!parseContext.isEsProfile() && parseContext.extensionTurnedOn(E_GL_ARB_tessellation_shader)))
            return keyword;

        return es30ReservedFromGLSL(400);

    case SAMPLE: 
    {
        const int numLayoutExts = 3;
        const char* layoutExts[numLayoutExts] = {E_GL_OES_shader_multisample_interpolation, E_GL_ARB_gpu_shader5,
                                                 E_GL_NV_gpu_shader5};
        if ((parseContext.isEsProfile() && parseContext.version >= 320) ||
            parseContext.extensionsTurnedOn(numLayoutExts, layoutExts))
            return keyword;
        return es30ReservedFromGLSL(400);
    }
    case SUBROUTINE:
        return es30ReservedFromGLSL(400);

    case SHARED:
        if ((parseContext.isEsProfile() && parseContext.version < 300) ||
            (!parseContext.isEsProfile() && parseContext.version < 140))
            return identifierOrType();
        return keyword;
    case LAYOUT:
    {
        const int numLayoutExts = 2;
        const char* layoutExts[numLayoutExts] = { E_GL_ARB_shading_language_420pack,
                                                  E_GL_ARB_explicit_attrib_location };
        if ((parseContext.isEsProfile() && parseContext.version < 300) ||
            (!parseContext.isEsProfile() && parseContext.version < 140 &&
            ! parseContext.extensionsTurnedOn(numLayoutExts, layoutExts)))
            return identifierOrType();
        return keyword;
    }

    case HIGH_PRECISION:
    case MEDIUM_PRECISION:
    case LOW_PRECISION:
    case PRECISION:
        return precisionKeyword();

    case MAT2X2:
    case MAT2X3:
    case MAT2X4:
    case MAT3X2:
    case MAT3X3:
    case MAT3X4:
    case MAT4X2:
    case MAT4X3:
    case MAT4X4:
        return matNxM();

    case DMAT2:
    case DMAT3:
    case DMAT4:
    case DMAT2X2:
    case DMAT2X3:
    case DMAT2X4:
    case DMAT3X2:
    case DMAT3X3:
    case DMAT3X4:
    case DMAT4X2:
    case DMAT4X3:
    case DMAT4X4:
        return dMat();

    case IMAGE1D:
    case IIMAGE1D:
    case UIMAGE1D:
    case IMAGE1DARRAY:
    case IIMAGE1DARRAY:
    case UIMAGE1DARRAY:
    case IMAGE2DRECT:
    case IIMAGE2DRECT:
    case UIMAGE2DRECT:
        afterType = true;
        return firstGenerationImage(false);

    case I64IMAGE1D:
    case U64IMAGE1D:
    case I64IMAGE1DARRAY:
    case U64IMAGE1DARRAY:
    case I64IMAGE2DRECT:
    case U64IMAGE2DRECT:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_image_int64)) {
            return firstGenerationImage(false);
        }
        return identifierOrType();

    case IMAGEBUFFER:
    case IIMAGEBUFFER:
    case UIMAGEBUFFER:
        afterType = true;
        if ((parseContext.isEsProfile() && parseContext.version >= 320) ||
            parseContext.extensionsTurnedOn(Num_AEP_texture_buffer, AEP_texture_buffer))
            return keyword;
        return firstGenerationImage(false);
        
    case I64IMAGEBUFFER:
    case U64IMAGEBUFFER:
        afterType = true;        
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_image_int64)) {
            if ((parseContext.isEsProfile() && parseContext.version >= 320) ||
                parseContext.extensionsTurnedOn(Num_AEP_texture_buffer, AEP_texture_buffer))
                return keyword;
            return firstGenerationImage(false);
        }
        return identifierOrType();

    case IMAGE2D:
    case IIMAGE2D:
    case UIMAGE2D:
    case IMAGE3D:
    case IIMAGE3D:
    case UIMAGE3D:
    case IMAGECUBE:
    case IIMAGECUBE:
    case UIMAGECUBE:
    case IMAGE2DARRAY:
    case IIMAGE2DARRAY:
    case UIMAGE2DARRAY:
        afterType = true;
        return firstGenerationImage(true);

    case I64IMAGE2D:
    case U64IMAGE2D:
    case I64IMAGE3D:
    case U64IMAGE3D:
    case I64IMAGECUBE:
    case U64IMAGECUBE:
    case I64IMAGE2DARRAY:
    case U64IMAGE2DARRAY:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_image_int64))
            return firstGenerationImage(true);
        return identifierOrType();
        
    case IMAGECUBEARRAY:
    case IIMAGECUBEARRAY:
    case UIMAGECUBEARRAY:
        afterType = true;
        if ((parseContext.isEsProfile() && parseContext.version >= 320) ||
            parseContext.extensionsTurnedOn(Num_AEP_texture_cube_map_array, AEP_texture_cube_map_array))
            return keyword;
        return secondGenerationImage();
        
    case I64IMAGECUBEARRAY:
    case U64IMAGECUBEARRAY:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_image_int64)) {
            if ((parseContext.isEsProfile() && parseContext.version >= 320) ||
                parseContext.extensionsTurnedOn(Num_AEP_texture_cube_map_array, AEP_texture_cube_map_array))
                return keyword;
            return secondGenerationImage();
        }
        return identifierOrType();

    case IMAGE2DMS:
    case IIMAGE2DMS:
    case UIMAGE2DMS:
    case IMAGE2DMSARRAY:
    case IIMAGE2DMSARRAY:
    case UIMAGE2DMSARRAY:
        afterType = true;
        return secondGenerationImage();
        
    case I64IMAGE2DMS:
    case U64IMAGE2DMS:
    case I64IMAGE2DMSARRAY:
    case U64IMAGE2DMSARRAY:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_image_int64)) {
            return secondGenerationImage();
        }
        return identifierOrType();

    case DOUBLE:
    case DVEC2:
    case DVEC3:
    case DVEC4:
        afterType = true;
        if (parseContext.isEsProfile() || parseContext.version < 150 ||
            (!parseContext.symbolTable.atBuiltInLevel() &&
              (parseContext.version < 400 && !parseContext.extensionTurnedOn(E_GL_ARB_gpu_shader_fp64) &&
              (parseContext.version < 410 && !parseContext.extensionTurnedOn(E_GL_ARB_vertex_attrib_64bit)))))
            reservedWord();
        return keyword;

    case INT64_T:
    case UINT64_T:
    case I64VEC2:
    case I64VEC3:
    case I64VEC4:
    case U64VEC2:
    case U64VEC3:
    case U64VEC4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_ARB_gpu_shader_int64) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types) ||
            parseContext.extensionTurnedOn(E_GL_NV_gpu_shader5) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types_int64))
            return keyword;
        return identifierOrType();

    case INT8_T:
    case UINT8_T:
    case I8VEC2:
    case I8VEC3:
    case I8VEC4:
    case U8VEC2:
    case U8VEC3:
    case U8VEC4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_8bit_storage) ||
            parseContext.extensionTurnedOn(E_GL_NV_gpu_shader5) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types_int8))
            return keyword;
        return identifierOrType();

    case INT16_T:
    case UINT16_T:
    case I16VEC2:
    case I16VEC3:
    case I16VEC4:
    case U16VEC2:
    case U16VEC3:
    case U16VEC4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_AMD_gpu_shader_int16) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_16bit_storage) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types) ||
            parseContext.extensionTurnedOn(E_GL_NV_gpu_shader5) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types_int16))
            return keyword;
        return identifierOrType();
    case INT32_T:
    case UINT32_T:
    case I32VEC2:
    case I32VEC3:
    case I32VEC4:
    case U32VEC2:
    case U32VEC3:
    case U32VEC4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types) ||
            parseContext.extensionTurnedOn(E_GL_NV_gpu_shader5) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types_int32))
            return keyword;
        return identifierOrType();
    case FLOAT32_T:
    case F32VEC2:
    case F32VEC3:
    case F32VEC4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types) ||
            parseContext.extensionTurnedOn(E_GL_NV_gpu_shader5) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types_float32))
            return keyword;
        return identifierOrType();
    case F32MAT2:
    case F32MAT3:
    case F32MAT4:
    case F32MAT2X2:
    case F32MAT2X3:
    case F32MAT2X4:
    case F32MAT3X2:
    case F32MAT3X3:
    case F32MAT3X4:
    case F32MAT4X2:
    case F32MAT4X3:
    case F32MAT4X4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types_float32))
            return keyword;
        return identifierOrType();

    case FLOAT64_T:
    case F64VEC2:
    case F64VEC3:
    case F64VEC4:
    afterType = true;
    if (parseContext.symbolTable.atBuiltInLevel() ||
        parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types) ||
        (parseContext.extensionTurnedOn(E_GL_NV_gpu_shader5) && 
         parseContext.extensionTurnedOn(E_GL_ARB_gpu_shader_fp64)) ||
        parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types_float64))
        return keyword;
    return identifierOrType();
    case F64MAT2:
    case F64MAT3:
    case F64MAT4:
    case F64MAT2X2:
    case F64MAT2X3:
    case F64MAT2X4:
    case F64MAT3X2:
    case F64MAT3X3:
    case F64MAT3X4:
    case F64MAT4X2:
    case F64MAT4X3:
    case F64MAT4X4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types_float64))
            return keyword;
        return identifierOrType();

    case FLOAT16_T:
    case F16VEC2:
    case F16VEC3:
    case F16VEC4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_AMD_gpu_shader_half_float) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_16bit_storage) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types) ||
            parseContext.extensionTurnedOn(E_GL_NV_gpu_shader5) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types_float16))
            return keyword;

        return identifierOrType();

    case F16MAT2:
    case F16MAT3:
    case F16MAT4:
    case F16MAT2X2:
    case F16MAT2X3:
    case F16MAT2X4:
    case F16MAT3X2:
    case F16MAT3X3:
    case F16MAT3X4:
    case F16MAT4X2:
    case F16MAT4X3:
    case F16MAT4X4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_AMD_gpu_shader_half_float) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types) ||
            parseContext.extensionTurnedOn(E_GL_EXT_shader_explicit_arithmetic_types_float16))
            return keyword;

        return identifierOrType();

    case BFLOAT16_T:
    case BF16VEC2:
    case BF16VEC3:
    case BF16VEC4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_bfloat16))
            return keyword;

        return identifierOrType();

    case FLOATE5M2_T:
    case FE5M2VEC2:
    case FE5M2VEC3:
    case FE5M2VEC4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_float_e5m2))
            return keyword;

        return identifierOrType();

    case FLOATE4M3_T:
    case FE4M3VEC2:
    case FE4M3VEC3:
    case FE4M3VEC4:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_float_e4m3))
            return keyword;

        return identifierOrType();

    case SAMPLERCUBEARRAY:
    case SAMPLERCUBEARRAYSHADOW:
    case ISAMPLERCUBEARRAY:
    case USAMPLERCUBEARRAY:
        afterType = true;
        if ((parseContext.isEsProfile() && parseContext.version >= 320) ||
            parseContext.extensionsTurnedOn(Num_AEP_texture_cube_map_array, AEP_texture_cube_map_array))
            return keyword;
        if (parseContext.isEsProfile() || (parseContext.version < 400 &&
            ! parseContext.extensionTurnedOn(E_GL_ARB_texture_cube_map_array)
            && ! parseContext.extensionsTurnedOn(Num_AEP_core_gpu_shader5, AEP_core_gpu_shader5)))
            reservedWord();
        return keyword;

    case TEXTURECUBEARRAY:
    case ITEXTURECUBEARRAY:
    case UTEXTURECUBEARRAY:
        if (parseContext.spvVersion.vulkan > 0)
            return keyword;
        else
            return identifierOrType();

    case UINT:
    case UVEC2:
    case UVEC3:
    case UVEC4:
    case SAMPLERCUBESHADOW:
    case SAMPLER2DARRAY:
    case SAMPLER2DARRAYSHADOW:
    case ISAMPLER2D:
    case ISAMPLER3D:
    case ISAMPLERCUBE:
    case ISAMPLER2DARRAY:
    case USAMPLER2D:
    case USAMPLER3D:
    case USAMPLERCUBE:
    case USAMPLER2DARRAY:
        afterType = true;
        if (keyword == SAMPLER2DARRAY || keyword == SAMPLER2DARRAYSHADOW) {
            if (!parseContext.isEsProfile() &&
                (parseContext.extensionTurnedOn(E_GL_EXT_texture_array) || parseContext.symbolTable.atBuiltInLevel())) {
                return keyword;
            }
        }
        return nonreservedKeyword(300, 130);

    case SAMPLER3D:
        afterType = true;
        if (parseContext.isEsProfile() && parseContext.version < 300) {
            if (!parseContext.extensionTurnedOn(E_GL_OES_texture_3D))
                reservedWord();
        }
        return keyword;

    case SAMPLER2DSHADOW:
        afterType = true;
        if (parseContext.isEsProfile() && parseContext.version < 300) {
            if (!parseContext.extensionTurnedOn(E_GL_EXT_shadow_samplers))
                reservedWord();
        }
        return keyword;

    case TEXTURE2D:
    case TEXTURECUBE:
    case TEXTURE2DARRAY:
    case ITEXTURE2D:
    case ITEXTURE3D:
    case ITEXTURECUBE:
    case ITEXTURE2DARRAY:
    case UTEXTURE2D:
    case UTEXTURE3D:
    case UTEXTURECUBE:
    case UTEXTURE2DARRAY:
    case TEXTURE3D:
    case SAMPLER:
    case SAMPLERSHADOW:
        if (parseContext.spvVersion.vulkan > 0)
            return keyword;
        else
            return identifierOrType();

    case ISAMPLER1D:
    case ISAMPLER1DARRAY:
    case SAMPLER1DARRAYSHADOW:
    case USAMPLER1D:
    case USAMPLER1DARRAY:
        afterType = true;
        if (keyword == SAMPLER1DARRAYSHADOW) {
            if (!parseContext.isEsProfile() &&
                (parseContext.extensionTurnedOn(E_GL_EXT_texture_array) || parseContext.symbolTable.atBuiltInLevel())) {
                return keyword;
            }
        }
        return es30ReservedFromGLSL(130);
    case ISAMPLER2DRECT:
    case USAMPLER2DRECT:
        afterType = true;
        return es30ReservedFromGLSL(140);

    case SAMPLERBUFFER:
        afterType = true;
        if ((parseContext.isEsProfile() && parseContext.version >= 320) ||
            parseContext.extensionsTurnedOn(Num_AEP_texture_buffer, AEP_texture_buffer))
            return keyword;
        return es30ReservedFromGLSL(130);

    case ISAMPLERBUFFER:
    case USAMPLERBUFFER:
        afterType = true;
        if ((parseContext.isEsProfile() && parseContext.version >= 320) ||
            parseContext.extensionsTurnedOn(Num_AEP_texture_buffer, AEP_texture_buffer))
            return keyword;
        return es30ReservedFromGLSL(140);

    case SAMPLER2DMS:
    case ISAMPLER2DMS:
    case USAMPLER2DMS:
        afterType = true;
        if (parseContext.isEsProfile() && parseContext.version >= 310)
            return keyword;
        if (!parseContext.isEsProfile() && (parseContext.version > 140 ||
            (parseContext.version == 140 && parseContext.extensionsTurnedOn(1, &E_GL_ARB_texture_multisample))))
            return keyword;
        return es30ReservedFromGLSL(150);

    case SAMPLER2DMSARRAY:
    case ISAMPLER2DMSARRAY:
    case USAMPLER2DMSARRAY:
        afterType = true;
        if ((parseContext.isEsProfile() && parseContext.version >= 320) ||
            parseContext.extensionsTurnedOn(1, &E_GL_OES_texture_storage_multisample_2d_array))
            return keyword;
        if (!parseContext.isEsProfile() && (parseContext.version > 140 ||
            (parseContext.version == 140 && parseContext.extensionsTurnedOn(1, &E_GL_ARB_texture_multisample))))
            return keyword;
        return es30ReservedFromGLSL(150);

    case SAMPLER1D:
    case SAMPLER1DSHADOW:
        afterType = true;
        if (parseContext.isEsProfile())
            reservedWord();
        return keyword;

    case SAMPLER2DRECT:
    case SAMPLER2DRECTSHADOW:
        afterType = true;
        if (parseContext.isEsProfile())
            reservedWord();
        else if (parseContext.version < 140 && ! parseContext.symbolTable.atBuiltInLevel() && ! parseContext.extensionTurnedOn(E_GL_ARB_texture_rectangle)) {
            if (parseContext.relaxedErrors())
                parseContext.requireExtensions(loc, 1, &E_GL_ARB_texture_rectangle, "texture-rectangle sampler keyword");
            else
                reservedWord();
        }
        return keyword;

    case SAMPLER1DARRAY:
        afterType = true;
        if (parseContext.isEsProfile() && parseContext.version == 300)
            reservedWord();
        else if ((parseContext.isEsProfile() && parseContext.version < 300) ||
                 ((!parseContext.isEsProfile() && parseContext.version < 130) &&
                   !parseContext.symbolTable.atBuiltInLevel() &&
                   !parseContext.extensionTurnedOn(E_GL_EXT_texture_array)))
            return identifierOrType();
        return keyword;

    case SAMPLEREXTERNALOES:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_OES_EGL_image_external) ||
            parseContext.extensionTurnedOn(E_GL_OES_EGL_image_external_essl3))
            return keyword;
        return identifierOrType();

    case SAMPLEREXTERNAL2DY2YEXT:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_YUV_target))
            return keyword;
        return identifierOrType();

    case ITEXTURE1DARRAY:
    case UTEXTURE1D:
    case ITEXTURE1D:
    case UTEXTURE1DARRAY:
    case TEXTUREBUFFER:
    case ITEXTURE2DRECT:
    case UTEXTURE2DRECT:
    case ITEXTUREBUFFER:
    case UTEXTUREBUFFER:
    case TEXTURE2DMS:
    case ITEXTURE2DMS:
    case UTEXTURE2DMS:
    case TEXTURE2DMSARRAY:
    case ITEXTURE2DMSARRAY:
    case UTEXTURE2DMSARRAY:
    case TEXTURE1D:
    case TEXTURE2DRECT:
    case TEXTURE1DARRAY:
        if (parseContext.spvVersion.vulkan > 0)
            return keyword;
        else
            return identifierOrType();

    case SUBPASSINPUT:
    case SUBPASSINPUTMS:
    case ISUBPASSINPUT:
    case ISUBPASSINPUTMS:
    case USUBPASSINPUT:
    case USUBPASSINPUTMS:
    case ATTACHMENTEXT:
    case IATTACHMENTEXT:
    case UATTACHMENTEXT:
        if (parseContext.spvVersion.vulkan > 0)
            return keyword;
        else
            return identifierOrType();

    case F16SAMPLER1D:
    case F16SAMPLER2D:
    case F16SAMPLER3D:
    case F16SAMPLER2DRECT:
    case F16SAMPLERCUBE:
    case F16SAMPLER1DARRAY:
    case F16SAMPLER2DARRAY:
    case F16SAMPLERCUBEARRAY:
    case F16SAMPLERBUFFER:
    case F16SAMPLER2DMS:
    case F16SAMPLER2DMSARRAY:
    case F16SAMPLER1DSHADOW:
    case F16SAMPLER2DSHADOW:
    case F16SAMPLER1DARRAYSHADOW:
    case F16SAMPLER2DARRAYSHADOW:
    case F16SAMPLER2DRECTSHADOW:
    case F16SAMPLERCUBESHADOW:
    case F16SAMPLERCUBEARRAYSHADOW:

    case F16IMAGE1D:
    case F16IMAGE2D:
    case F16IMAGE3D:
    case F16IMAGE2DRECT:
    case F16IMAGECUBE:
    case F16IMAGE1DARRAY:
    case F16IMAGE2DARRAY:
    case F16IMAGECUBEARRAY:
    case F16IMAGEBUFFER:
    case F16IMAGE2DMS:
    case F16IMAGE2DMSARRAY:

    case F16TEXTURE1D:
    case F16TEXTURE2D:
    case F16TEXTURE3D:
    case F16TEXTURE2DRECT:
    case F16TEXTURECUBE:
    case F16TEXTURE1DARRAY:
    case F16TEXTURE2DARRAY:
    case F16TEXTURECUBEARRAY:
    case F16TEXTUREBUFFER:
    case F16TEXTURE2DMS:
    case F16TEXTURE2DMSARRAY:

    case F16SUBPASSINPUT:
    case F16SUBPASSINPUTMS:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_AMD_gpu_shader_half_float_fetch))
            return keyword;
        return identifierOrType();

    case EXPLICITINTERPAMD:
        if (parseContext.extensionTurnedOn(E_GL_AMD_shader_explicit_vertex_parameter))
            return keyword;
        return identifierOrType();

    case PERVERTEXNV:
        if ((!parseContext.isEsProfile() && parseContext.version >= 450) ||
            parseContext.extensionTurnedOn(E_GL_NV_fragment_shader_barycentric))
            return keyword;
        return identifierOrType();

    case PERVERTEXEXT:
        if ((!parseContext.isEsProfile() && parseContext.version >= 450) ||
            parseContext.extensionTurnedOn(E_GL_EXT_fragment_shader_barycentric))
            return keyword;
        return identifierOrType();

    case PRECISE:
        if ((parseContext.isEsProfile() &&
             (parseContext.version >= 320 || parseContext.extensionsTurnedOn(Num_AEP_gpu_shader5, AEP_gpu_shader5))) ||
            (!parseContext.isEsProfile() &&
             (parseContext.version >= 400 
             || parseContext.extensionsTurnedOn(Num_AEP_core_gpu_shader5, AEP_core_gpu_shader5))))
            return keyword;
        if (parseContext.isEsProfile() && parseContext.version == 310) {
            reservedWord();
            return keyword;
        }
        return identifierOrType();

    case PERPRIMITIVENV:
    case PERVIEWNV:
    case PERTASKNV:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_NV_mesh_shader))
            return keyword;
        return identifierOrType();

    case PERPRIMITIVEEXT:
    case TASKPAYLOADWORKGROUPEXT:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_mesh_shader))
            return keyword;
        return identifierOrType();

    case FCOOPMATNV:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_NV_cooperative_matrix))
            return keyword;
        return identifierOrType();

    case UCOOPMATNV:
    case ICOOPMATNV:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_NV_integer_cooperative_matrix))
            return keyword;
        return identifierOrType();
    case TENSORARM:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_ARM_tensors))
            return keyword;
        return identifierOrType();

    case COOPMAT:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_KHR_cooperative_matrix))
            return keyword;
        return identifierOrType();

    case COOPVECNV:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_NV_cooperative_vector))
            return keyword;
        return identifierOrType();

    case DEMOTE:
        if (parseContext.extensionTurnedOn(E_GL_EXT_demote_to_helper_invocation))
            return keyword;
        else
            return identifierOrType();

    case SPIRV_INSTRUCTION:
    case SPIRV_EXECUTION_MODE:
    case SPIRV_EXECUTION_MODE_ID:
    case SPIRV_DECORATE:
    case SPIRV_DECORATE_ID:
    case SPIRV_DECORATE_STRING:
    case SPIRV_TYPE:
    case SPIRV_STORAGE_CLASS:
    case SPIRV_BY_REFERENCE:
    case SPIRV_LITERAL:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_EXT_spirv_intrinsics))
            return keyword;
        return identifierOrType();

    case HITOBJECTNV:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            (!parseContext.isEsProfile() && parseContext.version >= 460
                 && parseContext.extensionTurnedOn(E_GL_NV_shader_invocation_reorder)))
            return keyword;
        return identifierOrType();

    case HITOBJECTEXT:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            (!parseContext.isEsProfile() && parseContext.version >= 460
                 && parseContext.extensionTurnedOn(E_GL_EXT_shader_invocation_reorder)))
            return keyword;
        return identifierOrType();

    case HITOBJECTATTRNV:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            (!parseContext.isEsProfile() && parseContext.version >= 460
                 && parseContext.extensionTurnedOn(E_GL_NV_shader_invocation_reorder)))
            return keyword;
        return identifierOrType();

    case HITOBJECTATTREXT:
        if (parseContext.symbolTable.atBuiltInLevel() ||
            (!parseContext.isEsProfile() && parseContext.version >= 460
                 && parseContext.extensionTurnedOn(E_GL_EXT_shader_invocation_reorder)))
            return keyword;
        return identifierOrType();

    case FUNCTION:
    case TENSORLAYOUTNV:
    case TENSORVIEWNV:
        afterType = true;
        if (parseContext.symbolTable.atBuiltInLevel() ||
            parseContext.extensionTurnedOn(E_GL_NV_cooperative_matrix2))
            return keyword;
        return identifierOrType();

    default:
        parseContext.infoSink.info.message(EPrefixInternalError, "Unknown glslang keyword", loc);
        return 0;
    }
}

int TScanContext::identifierOrType()
{
    parserToken->sType.lex.string = NewPoolTString(tokenText);
    if (field)
        return IDENTIFIER;

    // If we see an identifier right after a type, this might be a declarator.
    // But not in template parameters (inside angle brackets), array expressions (inside square brackets),
    // or function parameters (inside parentheses)
    if (afterType && angleBracketDepth == 0 && squareBracketDepth == 0 && parenDepth == 0) {
        afterDeclarator = true;
        afterType = false;
        return IDENTIFIER;
    }

    parserToken->sType.lex.symbol = parseContext.symbolTable.find(*parserToken->sType.lex.string);
    if ((afterType == false && afterStruct == false) && parserToken->sType.lex.symbol != nullptr) {
        if (const TVariable* variable = parserToken->sType.lex.symbol->getAsVariable()) {
            if (variable->isUserType() &&
                // treat redeclaration of forward-declared buffer/uniform reference as an identifier
                !(variable->getType().isReference() && afterBuffer)) {

                // If we're in a declarator list (like "float a, B;"), treat struct names as IDENTIFIER
                // to fix GitHub issue #3931
                if (inDeclaratorList) {
                    return IDENTIFIER;
                }
                
                afterType = true;
                return TYPE_NAME;
            }
        }
    }

    return IDENTIFIER;
}

// Give an error for use of a reserved symbol.
// However, allow built-in declarations to use reserved words, to allow
// extension support before the extension is enabled.
int TScanContext::reservedWord()
{
    if (! parseContext.symbolTable.atBuiltInLevel())
        parseContext.error(loc, "Reserved word.", tokenText, "", "");

    return 0;
}

int TScanContext::identifierOrReserved(bool reserved)
{
    if (reserved) {
        reservedWord();

        return 0;
    }

    if (parseContext.isForwardCompatible())
        parseContext.warn(loc, "using future reserved keyword", tokenText, "");

    return identifierOrType();
}

// For keywords that suddenly showed up on non-ES (not previously reserved)
// but then got reserved by ES 3.0.
int TScanContext::es30ReservedFromGLSL(int version)
{
    if (parseContext.symbolTable.atBuiltInLevel())
        return keyword;

    if ((parseContext.isEsProfile() && parseContext.version < 300) ||
        (!parseContext.isEsProfile() && parseContext.version < version)) {
            if (parseContext.isForwardCompatible())
                parseContext.warn(loc, "future reserved word in ES 300 and keyword in GLSL", tokenText, "");

            return identifierOrType();
    } else if (parseContext.isEsProfile() && parseContext.version >= 300)
        reservedWord();

    return keyword;
}

// For a keyword that was never reserved, until it suddenly
// showed up, both in an es version and a non-ES version.
int TScanContext::nonreservedKeyword(int esVersion, int nonEsVersion)
{
    if ((parseContext.isEsProfile() && parseContext.version < esVersion) ||
        (!parseContext.isEsProfile() && parseContext.version < nonEsVersion)) {
        if (parseContext.isForwardCompatible())
            parseContext.warn(loc, "using future keyword", tokenText, "");

        return identifierOrType();
    }

    return keyword;
}

int TScanContext::precisionKeyword()
{
    if (parseContext.isEsProfile() || parseContext.version >= 130)
        return keyword;

    if (parseContext.isForwardCompatible())
        parseContext.warn(loc, "using ES precision qualifier keyword", tokenText, "");

    return identifierOrType();
}

int TScanContext::matNxM()
{
    afterType = true;

    if (parseContext.version > 110)
        return keyword;

    if (parseContext.isForwardCompatible())
        parseContext.warn(loc, "using future non-square matrix type keyword", tokenText, "");

    return identifierOrType();
}

int TScanContext::dMat()
{
    afterType = true;

    if (parseContext.isEsProfile() && parseContext.version >= 300) {
        reservedWord();

        return keyword;
    }

    if (!parseContext.isEsProfile() && (parseContext.version >= 400 ||
        parseContext.symbolTable.atBuiltInLevel() ||
        (parseContext.version >= 150 && parseContext.extensionTurnedOn(E_GL_ARB_gpu_shader_fp64)) ||
        (parseContext.version >= 150 && parseContext.extensionTurnedOn(E_GL_ARB_vertex_attrib_64bit)
         && parseContext.language == EShLangVertex)))
        return keyword;

    if (parseContext.isForwardCompatible())
        parseContext.warn(loc, "using future type keyword", tokenText, "");

    return identifierOrType();
}

int TScanContext::firstGenerationImage(bool inEs310)
{
    if (parseContext.symbolTable.atBuiltInLevel() ||
        (!parseContext.isEsProfile() && (parseContext.version >= 420 ||
         parseContext.extensionTurnedOn(E_GL_ARB_shader_image_load_store))) ||
        (inEs310 && parseContext.isEsProfile() && parseContext.version >= 310))
        return keyword;

    if ((parseContext.isEsProfile() && parseContext.version >= 300) ||
        (!parseContext.isEsProfile() && parseContext.version >= 130)) {
        reservedWord();

        return keyword;
    }

    if (parseContext.isForwardCompatible())
        parseContext.warn(loc, "using future type keyword", tokenText, "");

    return identifierOrType();
}

int TScanContext::secondGenerationImage()
{
    if (parseContext.isEsProfile() && parseContext.version >= 310) {
        reservedWord();
        return keyword;
    }

    if (parseContext.symbolTable.atBuiltInLevel() ||
        (!parseContext.isEsProfile() &&
         (parseContext.version >= 420 || parseContext.extensionTurnedOn(E_GL_ARB_shader_image_load_store))))
        return keyword;

    if (parseContext.isForwardCompatible())
        parseContext.warn(loc, "using future type keyword", tokenText, "");

    return identifierOrType();
}

} // end namespace glslang
