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

#ifndef SPIRV_UNIFIED1_NonSemanticShaderDebugInfo100_H_
#define SPIRV_UNIFIED1_NonSemanticShaderDebugInfo100_H_

#ifdef __cplusplus
extern "C" {
#endif

enum {
    NonSemanticShaderDebugInfo100Version = 100,
    NonSemanticShaderDebugInfo100Version_BitWidthPadding = 0x7fffffff
};
enum {
    NonSemanticShaderDebugInfo100Revision = 6,
    NonSemanticShaderDebugInfo100Revision_BitWidthPadding = 0x7fffffff
};

enum NonSemanticShaderDebugInfo100Instructions {
    NonSemanticShaderDebugInfo100DebugInfoNone = 0,
    NonSemanticShaderDebugInfo100DebugCompilationUnit = 1,
    NonSemanticShaderDebugInfo100DebugTypeBasic = 2,
    NonSemanticShaderDebugInfo100DebugTypePointer = 3,
    NonSemanticShaderDebugInfo100DebugTypeQualifier = 4,
    NonSemanticShaderDebugInfo100DebugTypeArray = 5,
    NonSemanticShaderDebugInfo100DebugTypeVector = 6,
    NonSemanticShaderDebugInfo100DebugTypedef = 7,
    NonSemanticShaderDebugInfo100DebugTypeFunction = 8,
    NonSemanticShaderDebugInfo100DebugTypeEnum = 9,
    NonSemanticShaderDebugInfo100DebugTypeComposite = 10,
    NonSemanticShaderDebugInfo100DebugTypeMember = 11,
    NonSemanticShaderDebugInfo100DebugTypeInheritance = 12,
    NonSemanticShaderDebugInfo100DebugTypePtrToMember = 13,
    NonSemanticShaderDebugInfo100DebugTypeTemplate = 14,
    NonSemanticShaderDebugInfo100DebugTypeTemplateParameter = 15,
    NonSemanticShaderDebugInfo100DebugTypeTemplateTemplateParameter = 16,
    NonSemanticShaderDebugInfo100DebugTypeTemplateParameterPack = 17,
    NonSemanticShaderDebugInfo100DebugGlobalVariable = 18,
    NonSemanticShaderDebugInfo100DebugFunctionDeclaration = 19,
    NonSemanticShaderDebugInfo100DebugFunction = 20,
    NonSemanticShaderDebugInfo100DebugLexicalBlock = 21,
    NonSemanticShaderDebugInfo100DebugLexicalBlockDiscriminator = 22,
    NonSemanticShaderDebugInfo100DebugScope = 23,
    NonSemanticShaderDebugInfo100DebugNoScope = 24,
    NonSemanticShaderDebugInfo100DebugInlinedAt = 25,
    NonSemanticShaderDebugInfo100DebugLocalVariable = 26,
    NonSemanticShaderDebugInfo100DebugInlinedVariable = 27,
    NonSemanticShaderDebugInfo100DebugDeclare = 28,
    NonSemanticShaderDebugInfo100DebugValue = 29,
    NonSemanticShaderDebugInfo100DebugOperation = 30,
    NonSemanticShaderDebugInfo100DebugExpression = 31,
    NonSemanticShaderDebugInfo100DebugMacroDef = 32,
    NonSemanticShaderDebugInfo100DebugMacroUndef = 33,
    NonSemanticShaderDebugInfo100DebugImportedEntity = 34,
    NonSemanticShaderDebugInfo100DebugSource = 35,
    NonSemanticShaderDebugInfo100DebugFunctionDefinition = 101,
    NonSemanticShaderDebugInfo100DebugSourceContinued = 102,
    NonSemanticShaderDebugInfo100DebugLine = 103,
    NonSemanticShaderDebugInfo100DebugNoLine = 104,
    NonSemanticShaderDebugInfo100DebugBuildIdentifier = 105,
    NonSemanticShaderDebugInfo100DebugStoragePath = 106,
    NonSemanticShaderDebugInfo100DebugEntryPoint = 107,
    NonSemanticShaderDebugInfo100DebugTypeMatrix = 108,
    NonSemanticShaderDebugInfo100InstructionsMax = 0x7fffffff
};


enum NonSemanticShaderDebugInfo100DebugInfoFlags {
    NonSemanticShaderDebugInfo100None = 0x0000,
    NonSemanticShaderDebugInfo100FlagIsProtected = 0x01,
    NonSemanticShaderDebugInfo100FlagIsPrivate = 0x02,
    NonSemanticShaderDebugInfo100FlagIsPublic = 0x03,
    NonSemanticShaderDebugInfo100FlagIsLocal = 0x04,
    NonSemanticShaderDebugInfo100FlagIsDefinition = 0x08,
    NonSemanticShaderDebugInfo100FlagFwdDecl = 0x10,
    NonSemanticShaderDebugInfo100FlagArtificial = 0x20,
    NonSemanticShaderDebugInfo100FlagExplicit = 0x40,
    NonSemanticShaderDebugInfo100FlagPrototyped = 0x80,
    NonSemanticShaderDebugInfo100FlagObjectPointer = 0x100,
    NonSemanticShaderDebugInfo100FlagStaticMember = 0x200,
    NonSemanticShaderDebugInfo100FlagIndirectVariable = 0x400,
    NonSemanticShaderDebugInfo100FlagLValueReference = 0x800,
    NonSemanticShaderDebugInfo100FlagRValueReference = 0x1000,
    NonSemanticShaderDebugInfo100FlagIsOptimized = 0x2000,
    NonSemanticShaderDebugInfo100FlagIsEnumClass = 0x4000,
    NonSemanticShaderDebugInfo100FlagTypePassByValue = 0x8000,
    NonSemanticShaderDebugInfo100FlagTypePassByReference = 0x10000,
    NonSemanticShaderDebugInfo100FlagUnknownPhysicalLayout = 0x20000,
    NonSemanticShaderDebugInfo100DebugInfoFlagsMax = 0x7fffffff
};

enum NonSemanticShaderDebugInfo100BuildIdentifierFlags {
    NonSemanticShaderDebugInfo100IdentifierPossibleDuplicates = 0x01,
    NonSemanticShaderDebugInfo100BuildIdentifierFlagsMax = 0x7fffffff
};

enum NonSemanticShaderDebugInfo100DebugBaseTypeAttributeEncoding {
    NonSemanticShaderDebugInfo100Unspecified = 0,
    NonSemanticShaderDebugInfo100Address = 1,
    NonSemanticShaderDebugInfo100Boolean = 2,
    NonSemanticShaderDebugInfo100Float = 3,
    NonSemanticShaderDebugInfo100Signed = 4,
    NonSemanticShaderDebugInfo100SignedChar = 5,
    NonSemanticShaderDebugInfo100Unsigned = 6,
    NonSemanticShaderDebugInfo100UnsignedChar = 7,
    NonSemanticShaderDebugInfo100DebugBaseTypeAttributeEncodingMax = 0x7fffffff
};

enum NonSemanticShaderDebugInfo100DebugCompositeType {
    NonSemanticShaderDebugInfo100Class = 0,
    NonSemanticShaderDebugInfo100Structure = 1,
    NonSemanticShaderDebugInfo100Union = 2,
    NonSemanticShaderDebugInfo100DebugCompositeTypeMax = 0x7fffffff
};

enum NonSemanticShaderDebugInfo100DebugTypeQualifier {
    NonSemanticShaderDebugInfo100ConstType = 0,
    NonSemanticShaderDebugInfo100VolatileType = 1,
    NonSemanticShaderDebugInfo100RestrictType = 2,
    NonSemanticShaderDebugInfo100AtomicType = 3,
    NonSemanticShaderDebugInfo100DebugTypeQualifierMax = 0x7fffffff
};

enum NonSemanticShaderDebugInfo100DebugOperation {
    NonSemanticShaderDebugInfo100Deref = 0,
    NonSemanticShaderDebugInfo100Plus = 1,
    NonSemanticShaderDebugInfo100Minus = 2,
    NonSemanticShaderDebugInfo100PlusUconst = 3,
    NonSemanticShaderDebugInfo100BitPiece = 4,
    NonSemanticShaderDebugInfo100Swap = 5,
    NonSemanticShaderDebugInfo100Xderef = 6,
    NonSemanticShaderDebugInfo100StackValue = 7,
    NonSemanticShaderDebugInfo100Constu = 8,
    NonSemanticShaderDebugInfo100Fragment = 9,
    NonSemanticShaderDebugInfo100DebugOperationMax = 0x7fffffff
};

enum NonSemanticShaderDebugInfo100DebugImportedEntity {
    NonSemanticShaderDebugInfo100ImportedModule = 0,
    NonSemanticShaderDebugInfo100ImportedDeclaration = 1,
    NonSemanticShaderDebugInfo100DebugImportedEntityMax = 0x7fffffff
};


#ifdef __cplusplus
}
#endif

#endif // SPIRV_UNIFIED1_NonSemanticShaderDebugInfo100_H_
