// Copyright (c) 2018 The Khronos Group Inc.
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

#ifndef SPIRV_EXTINST_OpenCLDebugInfo100_H_
#define SPIRV_EXTINST_OpenCLDebugInfo100_H_

#ifdef __cplusplus
extern "C" {
#endif

enum { OpenCLDebugInfo100Version = 200, OpenCLDebugInfo100Version_BitWidthPadding = 0x7fffffff };
enum { OpenCLDebugInfo100Revision = 2, OpenCLDebugInfo100Revision_BitWidthPadding = 0x7fffffff };

enum OpenCLDebugInfo100Instructions {
    OpenCLDebugInfo100DebugInfoNone = 0,
    OpenCLDebugInfo100DebugCompilationUnit = 1,
    OpenCLDebugInfo100DebugTypeBasic = 2,
    OpenCLDebugInfo100DebugTypePointer = 3,
    OpenCLDebugInfo100DebugTypeQualifier = 4,
    OpenCLDebugInfo100DebugTypeArray = 5,
    OpenCLDebugInfo100DebugTypeVector = 6,
    OpenCLDebugInfo100DebugTypedef = 7,
    OpenCLDebugInfo100DebugTypeFunction = 8,
    OpenCLDebugInfo100DebugTypeEnum = 9,
    OpenCLDebugInfo100DebugTypeComposite = 10,
    OpenCLDebugInfo100DebugTypeMember = 11,
    OpenCLDebugInfo100DebugTypeInheritance = 12,
    OpenCLDebugInfo100DebugTypePtrToMember = 13,
    OpenCLDebugInfo100DebugTypeTemplate = 14,
    OpenCLDebugInfo100DebugTypeTemplateParameter = 15,
    OpenCLDebugInfo100DebugTypeTemplateTemplateParameter = 16,
    OpenCLDebugInfo100DebugTypeTemplateParameterPack = 17,
    OpenCLDebugInfo100DebugGlobalVariable = 18,
    OpenCLDebugInfo100DebugFunctionDeclaration = 19,
    OpenCLDebugInfo100DebugFunction = 20,
    OpenCLDebugInfo100DebugLexicalBlock = 21,
    OpenCLDebugInfo100DebugLexicalBlockDiscriminator = 22,
    OpenCLDebugInfo100DebugScope = 23,
    OpenCLDebugInfo100DebugNoScope = 24,
    OpenCLDebugInfo100DebugInlinedAt = 25,
    OpenCLDebugInfo100DebugLocalVariable = 26,
    OpenCLDebugInfo100DebugInlinedVariable = 27,
    OpenCLDebugInfo100DebugDeclare = 28,
    OpenCLDebugInfo100DebugValue = 29,
    OpenCLDebugInfo100DebugOperation = 30,
    OpenCLDebugInfo100DebugExpression = 31,
    OpenCLDebugInfo100DebugMacroDef = 32,
    OpenCLDebugInfo100DebugMacroUndef = 33,
    OpenCLDebugInfo100DebugImportedEntity = 34,
    OpenCLDebugInfo100DebugSource = 35,
    OpenCLDebugInfo100DebugModuleINTEL = 36,
    OpenCLDebugInfo100InstructionsMax = 0x7ffffff
};


enum OpenCLDebugInfo100DebugInfoFlags {
    OpenCLDebugInfo100None = 0x0000,
    OpenCLDebugInfo100FlagIsProtected = 0x01,
    OpenCLDebugInfo100FlagIsPrivate = 0x02,
    OpenCLDebugInfo100FlagIsPublic = 0x03,
    OpenCLDebugInfo100FlagIsLocal = 0x04,
    OpenCLDebugInfo100FlagIsDefinition = 0x08,
    OpenCLDebugInfo100FlagFwdDecl = 0x10,
    OpenCLDebugInfo100FlagArtificial = 0x20,
    OpenCLDebugInfo100FlagExplicit = 0x40,
    OpenCLDebugInfo100FlagPrototyped = 0x80,
    OpenCLDebugInfo100FlagObjectPointer = 0x100,
    OpenCLDebugInfo100FlagStaticMember = 0x200,
    OpenCLDebugInfo100FlagIndirectVariable = 0x400,
    OpenCLDebugInfo100FlagLValueReference = 0x800,
    OpenCLDebugInfo100FlagRValueReference = 0x1000,
    OpenCLDebugInfo100FlagIsOptimized = 0x2000,
    OpenCLDebugInfo100FlagIsEnumClass = 0x4000,
    OpenCLDebugInfo100FlagTypePassByValue = 0x8000,
    OpenCLDebugInfo100FlagTypePassByReference = 0x10000,
    OpenCLDebugInfo100DebugInfoFlagsMax = 0x7ffffff
};

enum OpenCLDebugInfo100DebugBaseTypeAttributeEncoding {
    OpenCLDebugInfo100Unspecified = 0,
    OpenCLDebugInfo100Address = 1,
    OpenCLDebugInfo100Boolean = 2,
    OpenCLDebugInfo100Float = 3,
    OpenCLDebugInfo100Signed = 4,
    OpenCLDebugInfo100SignedChar = 5,
    OpenCLDebugInfo100Unsigned = 6,
    OpenCLDebugInfo100UnsignedChar = 7,
    OpenCLDebugInfo100DebugBaseTypeAttributeEncodingMax = 0x7ffffff
};

enum OpenCLDebugInfo100DebugCompositeType {
    OpenCLDebugInfo100Class = 0,
    OpenCLDebugInfo100Structure = 1,
    OpenCLDebugInfo100Union = 2,
    OpenCLDebugInfo100DebugCompositeTypeMax = 0x7ffffff
};

enum OpenCLDebugInfo100DebugTypeQualifier {
    OpenCLDebugInfo100ConstType = 0,
    OpenCLDebugInfo100VolatileType = 1,
    OpenCLDebugInfo100RestrictType = 2,
    OpenCLDebugInfo100AtomicType = 3,
    OpenCLDebugInfo100DebugTypeQualifierMax = 0x7ffffff
};

enum OpenCLDebugInfo100DebugOperation {
    OpenCLDebugInfo100Deref = 0,
    OpenCLDebugInfo100Plus = 1,
    OpenCLDebugInfo100Minus = 2,
    OpenCLDebugInfo100PlusUconst = 3,
    OpenCLDebugInfo100BitPiece = 4,
    OpenCLDebugInfo100Swap = 5,
    OpenCLDebugInfo100Xderef = 6,
    OpenCLDebugInfo100StackValue = 7,
    OpenCLDebugInfo100Constu = 8,
    OpenCLDebugInfo100Fragment = 9,
    OpenCLDebugInfo100DebugOperationMax = 0x7ffffff
};

enum OpenCLDebugInfo100DebugImportedEntity {
    OpenCLDebugInfo100ImportedModule = 0,
    OpenCLDebugInfo100ImportedDeclaration = 1,
    OpenCLDebugInfo100DebugImportedEntityMax = 0x7ffffff
};


#ifdef __cplusplus
}
#endif

#endif // SPIRV_EXTINST_OpenCLDebugInfo100_H_