// Copyright (c) 2017 The Khronos Group Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and/or associated documentation files (the "Materials"),
// to deal in the Materials without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Materials, and to permit persons to whom the
// Materials are furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Materials.
// 
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS KHRONOS
// STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS SPECIFICATIONS AND
// HEADER INFORMATION ARE LOCATED AT https://www.khronos.org/registry/ 
// 
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM,OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS
// IN THE MATERIALS.

#ifndef SPIRV_EXTINST_DebugInfo_H_
#define SPIRV_EXTINST_DebugInfo_H_

#ifdef __cplusplus
extern "C" {
#endif

enum { DebugInfoVersion = 100, DebugInfoVersion_BitWidthPadding = 0x7fffffff };
enum { DebugInfoRevision = 1, DebugInfoRevision_BitWidthPadding = 0x7fffffff };

enum DebugInfoInstructions {
    DebugInfoDebugInfoNone = 0,
    DebugInfoDebugCompilationUnit = 1,
    DebugInfoDebugTypeBasic = 2,
    DebugInfoDebugTypePointer = 3,
    DebugInfoDebugTypeQualifier = 4,
    DebugInfoDebugTypeArray = 5,
    DebugInfoDebugTypeVector = 6,
    DebugInfoDebugTypedef = 7,
    DebugInfoDebugTypeFunction = 8,
    DebugInfoDebugTypeEnum = 9,
    DebugInfoDebugTypeComposite = 10,
    DebugInfoDebugTypeMember = 11,
    DebugInfoDebugTypeInheritance = 12,
    DebugInfoDebugTypePtrToMember = 13,
    DebugInfoDebugTypeTemplate = 14,
    DebugInfoDebugTypeTemplateParameter = 15,
    DebugInfoDebugTypeTemplateTemplateParameter = 16,
    DebugInfoDebugTypeTemplateParameterPack = 17,
    DebugInfoDebugGlobalVariable = 18,
    DebugInfoDebugFunctionDeclaration = 19,
    DebugInfoDebugFunction = 20,
    DebugInfoDebugLexicalBlock = 21,
    DebugInfoDebugLexicalBlockDiscriminator = 22,
    DebugInfoDebugScope = 23,
    DebugInfoDebugNoScope = 24,
    DebugInfoDebugInlinedAt = 25,
    DebugInfoDebugLocalVariable = 26,
    DebugInfoDebugInlinedVariable = 27,
    DebugInfoDebugDeclare = 28,
    DebugInfoDebugValue = 29,
    DebugInfoDebugOperation = 30,
    DebugInfoDebugExpression = 31,
    DebugInfoDebugMacroDef = 32,
    DebugInfoDebugMacroUndef = 33,
    DebugInfoInstructionsMax = 0x7ffffff
};


enum DebugInfoDebugInfoFlags {
    DebugInfoNone = 0x0000,
    DebugInfoFlagIsProtected = 0x01,
    DebugInfoFlagIsPrivate = 0x02,
    DebugInfoFlagIsPublic = 0x03,
    DebugInfoFlagIsLocal = 0x04,
    DebugInfoFlagIsDefinition = 0x08,
    DebugInfoFlagFwdDecl = 0x10,
    DebugInfoFlagArtificial = 0x20,
    DebugInfoFlagExplicit = 0x40,
    DebugInfoFlagPrototyped = 0x80,
    DebugInfoFlagObjectPointer = 0x100,
    DebugInfoFlagStaticMember = 0x200,
    DebugInfoFlagIndirectVariable = 0x400,
    DebugInfoFlagLValueReference = 0x800,
    DebugInfoFlagRValueReference = 0x1000,
    DebugInfoFlagIsOptimized = 0x2000,
    DebugInfoDebugInfoFlagsMax = 0x7ffffff
};

enum DebugInfoDebugBaseTypeAttributeEncoding {
    DebugInfoUnspecified = 0,
    DebugInfoAddress = 1,
    DebugInfoBoolean = 2,
    DebugInfoFloat = 4,
    DebugInfoSigned = 5,
    DebugInfoSignedChar = 6,
    DebugInfoUnsigned = 7,
    DebugInfoUnsignedChar = 8,
    DebugInfoDebugBaseTypeAttributeEncodingMax = 0x7ffffff
};

enum DebugInfoDebugCompositeType {
    DebugInfoClass = 0,
    DebugInfoStructure = 1,
    DebugInfoUnion = 2,
    DebugInfoDebugCompositeTypeMax = 0x7ffffff
};

enum DebugInfoDebugTypeQualifier {
    DebugInfoConstType = 0,
    DebugInfoVolatileType = 1,
    DebugInfoRestrictType = 2,
    DebugInfoDebugTypeQualifierMax = 0x7ffffff
};

enum DebugInfoDebugOperation {
    DebugInfoDeref = 0,
    DebugInfoPlus = 1,
    DebugInfoMinus = 2,
    DebugInfoPlusUconst = 3,
    DebugInfoBitPiece = 4,
    DebugInfoSwap = 5,
    DebugInfoXderef = 6,
    DebugInfoStackValue = 7,
    DebugInfoConstu = 8,
    DebugInfoDebugOperationMax = 0x7ffffff
};


#ifdef __cplusplus
}
#endif

#endif // SPIRV_EXTINST_DebugInfo_H_