///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// dxcisense.h                                                               //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides declarations for the DirectX Compiler IntelliSense component.    //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef __DXC_ISENSE__
#define __DXC_ISENSE__

#include "dxc/dxcapi.h"
#include "dxc/Support/WinAdapter.h"

typedef enum DxcGlobalOptions
{
  DxcGlobalOpt_None = 0x0,
  DxcGlobalOpt_ThreadBackgroundPriorityForIndexing = 0x1,
  DxcGlobalOpt_ThreadBackgroundPriorityForEditing = 0x2,
  DxcGlobalOpt_ThreadBackgroundPriorityForAll =
    DxcGlobalOpt_ThreadBackgroundPriorityForIndexing | DxcGlobalOpt_ThreadBackgroundPriorityForEditing
} DxcGlobalOptions;

typedef enum DxcTokenKind
{
  DxcTokenKind_Punctuation = 0, // A token that contains some kind of punctuation.
  DxcTokenKind_Keyword = 1,     // A language keyword.
  DxcTokenKind_Identifier = 2,  // An identifier (that is not a keyword).
  DxcTokenKind_Literal = 3,     // A numeric, string, or character literal.
  DxcTokenKind_Comment = 4,     // A comment.
  DxcTokenKind_Unknown = 5,     // An unknown token (possibly known to a future version).
  DxcTokenKind_BuiltInType = 6, // A built-in type like int, void or float3.
} DxcTokenKind;

typedef enum DxcTypeKind
{
  DxcTypeKind_Invalid = 0, // Reprents an invalid type (e.g., where no type is available).
  DxcTypeKind_Unexposed = 1, // A type whose specific kind is not exposed via this interface.
  // Builtin types
  DxcTypeKind_Void = 2,
  DxcTypeKind_Bool = 3,
  DxcTypeKind_Char_U = 4,
  DxcTypeKind_UChar = 5,
  DxcTypeKind_Char16 = 6,
  DxcTypeKind_Char32 = 7,
  DxcTypeKind_UShort = 8,
  DxcTypeKind_UInt = 9,
  DxcTypeKind_ULong = 10,
  DxcTypeKind_ULongLong = 11,
  DxcTypeKind_UInt128 = 12,
  DxcTypeKind_Char_S = 13,
  DxcTypeKind_SChar = 14,
  DxcTypeKind_WChar = 15,
  DxcTypeKind_Short = 16,
  DxcTypeKind_Int = 17,
  DxcTypeKind_Long = 18,
  DxcTypeKind_LongLong = 19,
  DxcTypeKind_Int128 = 20,
  DxcTypeKind_Float = 21,
  DxcTypeKind_Double = 22,
  DxcTypeKind_LongDouble = 23,
  DxcTypeKind_NullPtr = 24,
  DxcTypeKind_Overload = 25,
  DxcTypeKind_Dependent = 26,
  DxcTypeKind_ObjCId = 27,
  DxcTypeKind_ObjCClass = 28,
  DxcTypeKind_ObjCSel = 29,
  DxcTypeKind_FirstBuiltin = DxcTypeKind_Void,
  DxcTypeKind_LastBuiltin = DxcTypeKind_ObjCSel,

  DxcTypeKind_Complex = 100,
  DxcTypeKind_Pointer = 101,
  DxcTypeKind_BlockPointer = 102,
  DxcTypeKind_LValueReference = 103,
  DxcTypeKind_RValueReference = 104,
  DxcTypeKind_Record = 105,
  DxcTypeKind_Enum = 106,
  DxcTypeKind_Typedef = 107,
  DxcTypeKind_ObjCInterface = 108,
  DxcTypeKind_ObjCObjectPointer = 109,
  DxcTypeKind_FunctionNoProto = 110,
  DxcTypeKind_FunctionProto = 111,
  DxcTypeKind_ConstantArray = 112,
  DxcTypeKind_Vector = 113,
  DxcTypeKind_IncompleteArray = 114,
  DxcTypeKind_VariableArray = 115,
  DxcTypeKind_DependentSizedArray = 116,
  DxcTypeKind_MemberPointer = 117
} DxcTypeKind;

// Describes the severity of a particular diagnostic.
typedef enum DxcDiagnosticSeverity
{
  // A diagnostic that has been suppressed, e.g., by a command-line option.
  DxcDiagnostic_Ignored = 0,

  // This diagnostic is a note that should be attached to the previous (non-note) diagnostic.
  DxcDiagnostic_Note = 1,

  // This diagnostic indicates suspicious code that may not be wrong.
  DxcDiagnostic_Warning = 2,

  // This diagnostic indicates that the code is ill-formed.
  DxcDiagnostic_Error = 3,

  // This diagnostic indicates that the code is ill-formed such that future
  // parser rec unlikely to produce useful results.
  DxcDiagnostic_Fatal = 4

} DxcDiagnosticSeverity;

// Options to control the display of diagnostics.
typedef enum DxcDiagnosticDisplayOptions
{
  // Display the source-location information where the diagnostic was located.
  DxcDiagnostic_DisplaySourceLocation = 0x01,

  // If displaying the source-location information of the diagnostic,
  // also include the column number.
  DxcDiagnostic_DisplayColumn = 0x02,

  // If displaying the source-location information of the diagnostic,
  // also include information about source ranges in a machine-parsable format.
  DxcDiagnostic_DisplaySourceRanges = 0x04,

  // Display the option name associated with this diagnostic, if any.
  DxcDiagnostic_DisplayOption = 0x08,

  // Display the category number associated with this diagnostic, if any.
  DxcDiagnostic_DisplayCategoryId = 0x10,

  // Display the category name associated with this diagnostic, if any.
  DxcDiagnostic_DisplayCategoryName = 0x20,

  // Display the severity of the diagnostic message.
  DxcDiagnostic_DisplaySeverity = 0x200
} DxcDiagnosticDisplayOptions;

typedef enum DxcTranslationUnitFlags
{
  // Used to indicate that no special translation-unit options are needed.
  DxcTranslationUnitFlags_None = 0x0,

  // Used to indicate that the parser should construct a "detailed"
  // preprocessing record, including all macro definitions and instantiations.
  DxcTranslationUnitFlags_DetailedPreprocessingRecord = 0x01,

  // Used to indicate that the translation unit is incomplete.
  DxcTranslationUnitFlags_Incomplete = 0x02,

  // Used to indicate that the translation unit should be built with an
  // implicit precompiled header for the preamble.
  DxcTranslationUnitFlags_PrecompiledPreamble = 0x04,

  // Used to indicate that the translation unit should cache some
  // code-completion results with each reparse of the source file.
  DxcTranslationUnitFlags_CacheCompletionResults = 0x08,

  // Used to indicate that the translation unit will be serialized with
  // SaveTranslationUnit.
  DxcTranslationUnitFlags_ForSerialization = 0x10,

  // DEPRECATED
  DxcTranslationUnitFlags_CXXChainedPCH = 0x20,

  // Used to indicate that function/method bodies should be skipped while parsing.
  DxcTranslationUnitFlags_SkipFunctionBodies = 0x40,

  // Used to indicate that brief documentation comments should be
  // included into the set of code completions returned from this translation
  // unit.
  DxcTranslationUnitFlags_IncludeBriefCommentsInCodeCompletion = 0x80,

  // Used to indicate that compilation should occur on the caller's thread.
  DxcTranslationUnitFlags_UseCallerThread = 0x800
} DxcTranslationUnitFlags;

typedef enum DxcCursorFormatting
{
  DxcCursorFormatting_Default = 0x0,             // Default rules, language-insensitive formatting.
  DxcCursorFormatting_UseLanguageOptions = 0x1,  // Language-sensitive formatting.
  DxcCursorFormatting_SuppressSpecifiers = 0x2,  // Supresses type specifiers.
  DxcCursorFormatting_SuppressTagKeyword = 0x4,  // Suppressed tag keyword (eg, 'class').
  DxcCursorFormatting_IncludeNamespaceKeyword = 0x8,  // Include namespace keyword.
} DxcCursorFormatting;

enum DxcCursorKind {
  /* Declarations */
  DxcCursor_UnexposedDecl = 1, // A declaration whose specific kind is not exposed via this interface.
  DxcCursor_StructDecl = 2, // A C or C++ struct.
  DxcCursor_UnionDecl = 3, // A C or C++ union.
  DxcCursor_ClassDecl = 4, // A C++ class.
  DxcCursor_EnumDecl = 5, // An enumeration.
  DxcCursor_FieldDecl = 6, // A field (in C) or non-static data member (in C++) in a struct, union, or C++ class.
  DxcCursor_EnumConstantDecl = 7, // An enumerator constant.
  DxcCursor_FunctionDecl = 8, // A function.
  DxcCursor_VarDecl = 9, // A variable.
  DxcCursor_ParmDecl = 10, // A function or method parameter.
  DxcCursor_ObjCInterfaceDecl = 11, // An Objective-C interface.
  DxcCursor_ObjCCategoryDecl = 12, // An Objective-C interface for a category.
  DxcCursor_ObjCProtocolDecl = 13, // An Objective-C protocol declaration.
  DxcCursor_ObjCPropertyDecl = 14, // An Objective-C property declaration.
  DxcCursor_ObjCIvarDecl = 15, // An Objective-C instance variable.
  DxcCursor_ObjCInstanceMethodDecl = 16, // An Objective-C instance method.
  DxcCursor_ObjCClassMethodDecl = 17, // An Objective-C class method.
  DxcCursor_ObjCImplementationDecl = 18, // An Objective-C \@implementation.
  DxcCursor_ObjCCategoryImplDecl = 19, // An Objective-C \@implementation for a category.
  DxcCursor_TypedefDecl = 20, // A typedef
  DxcCursor_CXXMethod = 21, // A C++ class method.
  DxcCursor_Namespace = 22, // A C++ namespace.
  DxcCursor_LinkageSpec = 23, // A linkage specification, e.g. 'extern "C"'.
  DxcCursor_Constructor = 24, // A C++ constructor.
  DxcCursor_Destructor = 25, // A C++ destructor.
  DxcCursor_ConversionFunction = 26, // A C++ conversion function.
  DxcCursor_TemplateTypeParameter = 27, // A C++ template type parameter.
  DxcCursor_NonTypeTemplateParameter = 28, // A C++ non-type template parameter.
  DxcCursor_TemplateTemplateParameter = 29, // A C++ template template parameter.
  DxcCursor_FunctionTemplate = 30, // A C++ function template.
  DxcCursor_ClassTemplate = 31, // A C++ class template.
  DxcCursor_ClassTemplatePartialSpecialization = 32, // A C++ class template partial specialization.
  DxcCursor_NamespaceAlias = 33, // A C++ namespace alias declaration.
  DxcCursor_UsingDirective = 34, // A C++ using directive.
  DxcCursor_UsingDeclaration = 35, // A C++ using declaration.
  DxcCursor_TypeAliasDecl = 36, // A C++ alias declaration
  DxcCursor_ObjCSynthesizeDecl = 37, // An Objective-C \@synthesize definition.
  DxcCursor_ObjCDynamicDecl = 38, // An Objective-C \@dynamic definition.
  DxcCursor_CXXAccessSpecifier = 39, // An access specifier.

  DxcCursor_FirstDecl = DxcCursor_UnexposedDecl,
  DxcCursor_LastDecl = DxcCursor_CXXAccessSpecifier,

  /* References */
  DxcCursor_FirstRef = 40, /* Decl references */
  DxcCursor_ObjCSuperClassRef = 40,
  DxcCursor_ObjCProtocolRef = 41,
  DxcCursor_ObjCClassRef = 42,
  /**
  * \brief A reference to a type declaration.
  *
  * A type reference occurs anywhere where a type is named but not
  * declared. For example, given:
  *
  * \code
  * typedef unsigned size_type;
  * size_type size;
  * \endcode
  *
  * The typedef is a declaration of size_type (DxcCursor_TypedefDecl),
  * while the type of the variable "size" is referenced. The cursor
  * referenced by the type of size is the typedef for size_type.
  */
  DxcCursor_TypeRef = 43, // A reference to a type declaration.
  DxcCursor_CXXBaseSpecifier = 44,
  DxcCursor_TemplateRef = 45, // A reference to a class template, function template, template template parameter, or class template partial specialization.
  DxcCursor_NamespaceRef = 46, // A reference to a namespace or namespace alias.
  DxcCursor_MemberRef = 47, // A reference to a member of a struct, union, or class that occurs in some non-expression context, e.g., a designated initializer.
  /**
  * \brief A reference to a labeled statement.
  *
  * This cursor kind is used to describe the jump to "start_over" in the
  * goto statement in the following example:
  *
  * \code
  *   start_over:
  *     ++counter;
  *
  *     goto start_over;
  * \endcode
  *
  * A label reference cursor refers to a label statement.
  */
  DxcCursor_LabelRef = 48, // A reference to a labeled statement.

  // A reference to a set of overloaded functions or function templates
  // that has not yet been resolved to a specific function or function template.
  //
  // An overloaded declaration reference cursor occurs in C++ templates where
  // a dependent name refers to a function.
  DxcCursor_OverloadedDeclRef = 49,
  DxcCursor_VariableRef = 50, // A reference to a variable that occurs in some non-expression context, e.g., a C++ lambda capture list.

  DxcCursor_LastRef = DxcCursor_VariableRef,

  /* Error conditions */
  DxcCursor_FirstInvalid = 70,
  DxcCursor_InvalidFile = 70,
  DxcCursor_NoDeclFound = 71,
  DxcCursor_NotImplemented = 72,
  DxcCursor_InvalidCode = 73,
  DxcCursor_LastInvalid = DxcCursor_InvalidCode,

  /* Expressions */
  DxcCursor_FirstExpr = 100,

  /**
  * \brief An expression whose specific kind is not exposed via this
  * interface.
  *
  * Unexposed expressions have the same operations as any other kind
  * of expression; one can extract their location information,
  * spelling, children, etc. However, the specific kind of the
  * expression is not reported.
  */
  DxcCursor_UnexposedExpr = 100, // An expression whose specific kind is not exposed via this interface.
  DxcCursor_DeclRefExpr = 101, // An expression that refers to some value declaration, such as a function, varible, or enumerator.
  DxcCursor_MemberRefExpr = 102, // An expression that refers to a member of a struct, union, class, Objective-C class, etc.
  DxcCursor_CallExpr = 103, // An expression that calls a function.
  DxcCursor_ObjCMessageExpr = 104, // An expression that sends a message to an Objective-C object or class.
  DxcCursor_BlockExpr = 105, // An expression that represents a block literal.
  DxcCursor_IntegerLiteral = 106, // An integer literal.
  DxcCursor_FloatingLiteral = 107, // A floating point number literal.
  DxcCursor_ImaginaryLiteral = 108, // An imaginary number literal.
  DxcCursor_StringLiteral = 109, // A string literal.
  DxcCursor_CharacterLiteral = 110, // A character literal.
  DxcCursor_ParenExpr = 111, // A parenthesized expression, e.g. "(1)". This AST node is only formed if full location information is requested.
  DxcCursor_UnaryOperator = 112, // This represents the unary-expression's (except sizeof and alignof).
  DxcCursor_ArraySubscriptExpr = 113, // [C99 6.5.2.1] Array Subscripting.
  DxcCursor_BinaryOperator = 114, // A builtin binary operation expression such as "x + y" or "x <= y".
  DxcCursor_CompoundAssignOperator = 115, // Compound assignment such as "+=".
  DxcCursor_ConditionalOperator = 116, // The ?: ternary operator.
  DxcCursor_CStyleCastExpr = 117, // An explicit cast in C (C99 6.5.4) or a C-style cast in C++ (C++ [expr.cast]), which uses the syntax (Type)expr, eg: (int)f.
  DxcCursor_CompoundLiteralExpr = 118, // [C99 6.5.2.5]
  DxcCursor_InitListExpr = 119, // Describes an C or C++ initializer list.
  DxcCursor_AddrLabelExpr = 120, // The GNU address of label extension, representing &&label.
  DxcCursor_StmtExpr = 121, // This is the GNU Statement Expression extension: ({int X=4; X;})
  DxcCursor_GenericSelectionExpr = 122, // Represents a C11 generic selection.

  /** \brief Implements the GNU __null extension, which is a name for a null
  * pointer constant that has integral type (e.g., int or long) and is the same
  * size and alignment as a pointer.
  *
  * The __null extension is typically only used by system headers, which define
  * NULL as __null in C++ rather than using 0 (which is an integer that may not
  * match the size of a pointer).
  */
  DxcCursor_GNUNullExpr = 123,
  DxcCursor_CXXStaticCastExpr = 124, // C++'s static_cast<> expression.
  DxcCursor_CXXDynamicCastExpr = 125, // C++'s dynamic_cast<> expression.
  DxcCursor_CXXReinterpretCastExpr = 126, // C++'s reinterpret_cast<> expression.
  DxcCursor_CXXConstCastExpr = 127, // C++'s const_cast<> expression.

  /** \brief Represents an explicit C++ type conversion that uses "functional"
  * notion (C++ [expr.type.conv]).
  *
  * Example:
  * \code
  *   x = int(0.5);
  * \endcode
  */
  DxcCursor_CXXFunctionalCastExpr = 128,
  DxcCursor_CXXTypeidExpr = 129, // A C++ typeid expression (C++ [expr.typeid]).
  DxcCursor_CXXBoolLiteralExpr = 130, // [C++ 2.13.5] C++ Boolean Literal.
  DxcCursor_CXXNullPtrLiteralExpr = 131, // [C++0x 2.14.7] C++ Pointer Literal.
  DxcCursor_CXXThisExpr = 132, // Represents the "this" expression in C++
  DxcCursor_CXXThrowExpr = 133, // [C++ 15] C++ Throw Expression, both 'throw' and 'throw' assignment-expression.
  DxcCursor_CXXNewExpr = 134, // A new expression for memory allocation and constructor calls, e.g: "new CXXNewExpr(foo)".
  DxcCursor_CXXDeleteExpr = 135, // A delete expression for memory deallocation and destructor calls, e.g. "delete[] pArray".
  DxcCursor_UnaryExpr = 136, // A unary expression.
  DxcCursor_ObjCStringLiteral = 137, // An Objective-C string literal i.e. @"foo".
  DxcCursor_ObjCEncodeExpr = 138, // An Objective-C \@encode expression.
  DxcCursor_ObjCSelectorExpr = 139, // An Objective-C \@selector expression.
  DxcCursor_ObjCProtocolExpr = 140, // An Objective-C \@protocol expression.

  /** \brief An Objective-C "bridged" cast expression, which casts between
  * Objective-C pointers and C pointers, transferring ownership in the process.
  *
  * \code
  *   NSString *str = (__bridge_transfer NSString *)CFCreateString();
  * \endcode
  */
  DxcCursor_ObjCBridgedCastExpr = 141,

  /** \brief Represents a C++0x pack expansion that produces a sequence of
  * expressions.
  *
  * A pack expansion expression contains a pattern (which itself is an
  * expression) followed by an ellipsis. For example:
  *
  * \code
  * template<typename F, typename ...Types>
  * void forward(F f, Types &&...args) {
  *  f(static_cast<Types&&>(args)...);
  * }
  * \endcode
  */
  DxcCursor_PackExpansionExpr = 142,

  /** \brief Represents an expression that computes the length of a parameter
  * pack.
  *
  * \code
  * template<typename ...Types>
  * struct count {
  *   static const unsigned value = sizeof...(Types);
  * };
  * \endcode
  */
  DxcCursor_SizeOfPackExpr = 143,

  /* \brief Represents a C++ lambda expression that produces a local function
  * object.
  *
  * \code
  * void abssort(float *x, unsigned N) {
  *   std::sort(x, x + N,
  *             [](float a, float b) {
  *               return std::abs(a) < std::abs(b);
  *             });
  * }
  * \endcode
  */
  DxcCursor_LambdaExpr = 144,
  DxcCursor_ObjCBoolLiteralExpr = 145, // Objective-c Boolean Literal.
  DxcCursor_ObjCSelfExpr = 146, // Represents the "self" expression in a ObjC method.
  DxcCursor_LastExpr = DxcCursor_ObjCSelfExpr,

  /* Statements */
  DxcCursor_FirstStmt = 200,
  /**
  * \brief A statement whose specific kind is not exposed via this
  * interface.
  *
  * Unexposed statements have the same operations as any other kind of
  * statement; one can extract their location information, spelling,
  * children, etc. However, the specific kind of the statement is not
  * reported.
  */
  DxcCursor_UnexposedStmt = 200,

  /** \brief A labelled statement in a function.
  *
  * This cursor kind is used to describe the "start_over:" label statement in
  * the following example:
  *
  * \code
  *   start_over:
  *     ++counter;
  * \endcode
  *
  */
  DxcCursor_LabelStmt = 201,
  DxcCursor_CompoundStmt = 202, // A group of statements like { stmt stmt }. This cursor kind is used to describe compound statements, e.g. function bodies.
  DxcCursor_CaseStmt = 203, // A case statement.
  DxcCursor_DefaultStmt = 204, // A default statement.
  DxcCursor_IfStmt = 205, // An if statement
  DxcCursor_SwitchStmt = 206, // A switch statement.
  DxcCursor_WhileStmt = 207, // A while statement.
  DxcCursor_DoStmt = 208, // A do statement.
  DxcCursor_ForStmt = 209, // A for statement.
  DxcCursor_GotoStmt = 210, // A goto statement.
  DxcCursor_IndirectGotoStmt = 211, // An indirect goto statement.
  DxcCursor_ContinueStmt = 212, // A continue statement.
  DxcCursor_BreakStmt = 213, // A break statement.
  DxcCursor_ReturnStmt = 214, // A return statement.
  DxcCursor_GCCAsmStmt = 215, // A GCC inline assembly statement extension.
  DxcCursor_AsmStmt = DxcCursor_GCCAsmStmt,

  DxcCursor_ObjCAtTryStmt = 216, // Objective-C's overall \@try-\@catch-\@finally statement.
  DxcCursor_ObjCAtCatchStmt = 217, // Objective-C's \@catch statement.
  DxcCursor_ObjCAtFinallyStmt = 218, // Objective-C's \@finally statement.
  DxcCursor_ObjCAtThrowStmt = 219, // Objective-C's \@throw statement.
  DxcCursor_ObjCAtSynchronizedStmt = 220, // Objective-C's \@synchronized statement.
  DxcCursor_ObjCAutoreleasePoolStmt = 221, // Objective-C's autorelease pool statement.
  DxcCursor_ObjCForCollectionStmt = 222, // Objective-C's collection statement.

  DxcCursor_CXXCatchStmt = 223, // C++'s catch statement.
  DxcCursor_CXXTryStmt = 224, // C++'s try statement.
  DxcCursor_CXXForRangeStmt = 225, // C++'s for (* : *) statement.

  DxcCursor_SEHTryStmt = 226, // Windows Structured Exception Handling's try statement.
  DxcCursor_SEHExceptStmt = 227, // Windows Structured Exception Handling's except statement.
  DxcCursor_SEHFinallyStmt = 228, // Windows Structured Exception Handling's finally statement.

  DxcCursor_MSAsmStmt = 229, // A MS inline assembly statement extension.
  DxcCursor_NullStmt = 230, // The null satement ";": C99 6.8.3p3.
  DxcCursor_DeclStmt = 231, // Adaptor class for mixing declarations with statements and expressions.
  DxcCursor_OMPParallelDirective = 232, // OpenMP parallel directive.
  DxcCursor_OMPSimdDirective = 233,  // OpenMP SIMD directive.
  DxcCursor_OMPForDirective = 234,  // OpenMP for directive.
  DxcCursor_OMPSectionsDirective = 235,  // OpenMP sections directive.
  DxcCursor_OMPSectionDirective = 236,  // OpenMP section directive.
  DxcCursor_OMPSingleDirective = 237,  // OpenMP single directive.
  DxcCursor_OMPParallelForDirective = 238,  // OpenMP parallel for directive.
  DxcCursor_OMPParallelSectionsDirective = 239,  // OpenMP parallel sections directive.
  DxcCursor_OMPTaskDirective = 240,  // OpenMP task directive.
  DxcCursor_OMPMasterDirective = 241,  // OpenMP master directive.
  DxcCursor_OMPCriticalDirective = 242,  // OpenMP critical directive.
  DxcCursor_OMPTaskyieldDirective = 243,  // OpenMP taskyield directive.
  DxcCursor_OMPBarrierDirective = 244,  // OpenMP barrier directive.
  DxcCursor_OMPTaskwaitDirective = 245,  // OpenMP taskwait directive.
  DxcCursor_OMPFlushDirective = 246,  // OpenMP flush directive.
  DxcCursor_SEHLeaveStmt = 247,  // Windows Structured Exception Handling's leave statement.
  DxcCursor_OMPOrderedDirective = 248,  // OpenMP ordered directive.
  DxcCursor_OMPAtomicDirective = 249,  // OpenMP atomic directive.
  DxcCursor_OMPForSimdDirective = 250,  // OpenMP for SIMD directive.
  DxcCursor_OMPParallelForSimdDirective = 251,  // OpenMP parallel for SIMD directive.
  DxcCursor_OMPTargetDirective = 252,  // OpenMP target directive.
  DxcCursor_OMPTeamsDirective = 253,  // OpenMP teams directive.
  DxcCursor_OMPTaskgroupDirective = 254,  // OpenMP taskgroup directive.
  DxcCursor_OMPCancellationPointDirective = 255,  // OpenMP cancellation point directive.
  DxcCursor_OMPCancelDirective = 256,  // OpenMP cancel directive.
  DxcCursor_LastStmt = DxcCursor_OMPCancelDirective,

  DxcCursor_TranslationUnit = 300, // Cursor that represents the translation unit itself.

  /* Attributes */
  DxcCursor_FirstAttr = 400,
  /**
  * \brief An attribute whose specific kind is not exposed via this
  * interface.
  */
  DxcCursor_UnexposedAttr = 400,

  DxcCursor_IBActionAttr = 401,
  DxcCursor_IBOutletAttr = 402,
  DxcCursor_IBOutletCollectionAttr = 403,
  DxcCursor_CXXFinalAttr = 404,
  DxcCursor_CXXOverrideAttr = 405,
  DxcCursor_AnnotateAttr = 406,
  DxcCursor_AsmLabelAttr = 407,
  DxcCursor_PackedAttr = 408,
  DxcCursor_PureAttr = 409,
  DxcCursor_ConstAttr = 410,
  DxcCursor_NoDuplicateAttr = 411,
  DxcCursor_CUDAConstantAttr = 412,
  DxcCursor_CUDADeviceAttr = 413,
  DxcCursor_CUDAGlobalAttr = 414,
  DxcCursor_CUDAHostAttr = 415,
  DxcCursor_CUDASharedAttr = 416,
  DxcCursor_LastAttr = DxcCursor_CUDASharedAttr,

  /* Preprocessing */
  DxcCursor_PreprocessingDirective = 500,
  DxcCursor_MacroDefinition = 501,
  DxcCursor_MacroExpansion = 502,
  DxcCursor_MacroInstantiation = DxcCursor_MacroExpansion,
  DxcCursor_InclusionDirective = 503,
  DxcCursor_FirstPreprocessing = DxcCursor_PreprocessingDirective,
  DxcCursor_LastPreprocessing = DxcCursor_InclusionDirective,

  /* Extra Declarations */
  /**
  * \brief A module import declaration.
  */
  DxcCursor_ModuleImportDecl = 600,
  DxcCursor_FirstExtraDecl = DxcCursor_ModuleImportDecl,
  DxcCursor_LastExtraDecl = DxcCursor_ModuleImportDecl
};

enum DxcCursorKindFlags
{
  DxcCursorKind_None = 0,
  DxcCursorKind_Declaration = 0x1,
  DxcCursorKind_Reference = 0x2,
  DxcCursorKind_Expression = 0x4,
  DxcCursorKind_Statement = 0x8,
  DxcCursorKind_Attribute = 0x10,
  DxcCursorKind_Invalid = 0x20,
  DxcCursorKind_TranslationUnit = 0x40,
  DxcCursorKind_Preprocessing = 0x80,
  DxcCursorKind_Unexposed = 0x100,
};

enum DxcCodeCompleteFlags
{
  DxcCodeCompleteFlags_None = 0,
  DxcCodeCompleteFlags_IncludeMacros = 0x1,
  DxcCodeCompleteFlags_IncludeCodePatterns = 0x2,
  DxcCodeCompleteFlags_IncludeBriefComments = 0x4,
};

enum DxcCompletionChunkKind
{
  DxcCompletionChunk_Optional = 0,
  DxcCompletionChunk_TypedText = 1,
  DxcCompletionChunk_Text = 2,
  DxcCompletionChunk_Placeholder = 3,
  DxcCompletionChunk_Informative = 4,
  DxcCompletionChunk_CurrentParameter = 5,
  DxcCompletionChunk_LeftParen = 6,
  DxcCompletionChunk_RightParen = 7,
  DxcCompletionChunk_LeftBracket = 8,
  DxcCompletionChunk_RightBracket = 9,
  DxcCompletionChunk_LeftBrace = 10,
  DxcCompletionChunk_RightBrace = 11,
  DxcCompletionChunk_LeftAngle = 12,
  DxcCompletionChunk_RightAngle = 13,
  DxcCompletionChunk_Comma = 14,
  DxcCompletionChunk_ResultType = 15,
  DxcCompletionChunk_Colon = 16,
  DxcCompletionChunk_SemiColon = 17,
  DxcCompletionChunk_Equal = 18,
  DxcCompletionChunk_HorizontalSpace = 19,
  DxcCompletionChunk_VerticalSpace = 20,
};

struct IDxcCursor;
struct IDxcDiagnostic;
struct IDxcFile;
struct IDxcInclusion;
struct IDxcIntelliSense;
struct IDxcIndex;
struct IDxcSourceLocation;
struct IDxcSourceRange;
struct IDxcToken;
struct IDxcTranslationUnit;
struct IDxcType;
struct IDxcUnsavedFile;
struct IDxcCodeCompleteResults;
struct IDxcCompletionResult;
struct IDxcCompletionString;

CROSS_PLATFORM_UUIDOF(IDxcCursor, "1467b985-288d-4d2a-80c1-ef89c42c40bc")
struct IDxcCursor : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE GetExtent(_Outptr_result_nullonfailure_ IDxcSourceRange** pRange) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetLocation(_Outptr_result_nullonfailure_ IDxcSourceLocation** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetKind(_Out_ DxcCursorKind* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetKindFlags(_Out_ DxcCursorKindFlags* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetSemanticParent(_Outptr_result_nullonfailure_ IDxcCursor** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetLexicalParent(_Outptr_result_nullonfailure_ IDxcCursor** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetCursorType(_Outptr_result_nullonfailure_ IDxcType** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetNumArguments(_Out_ int* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetArgumentAt(int index, _Outptr_result_nullonfailure_ IDxcCursor** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetReferencedCursor(_Outptr_result_nullonfailure_ IDxcCursor** pResult) = 0;
  /// <summary>For a cursor that is either a reference to or a declaration of some entity, retrieve a cursor that describes the definition of that entity.</summary>
  /// <remarks>Some entities can be declared multiple times within a translation unit, but only one of those declarations can also be a definition.</remarks>
  /// <returns>A cursor to the definition of this entity; nullptr if there is no definition in this translation unit.</returns>
  virtual HRESULT STDMETHODCALLTYPE GetDefinitionCursor(_Outptr_result_nullonfailure_ IDxcCursor** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE FindReferencesInFile(
    _In_ IDxcFile* file, unsigned skip, unsigned top,
    _Out_ unsigned* pResultLength, _Outptr_result_buffer_maybenull_(*pResultLength) IDxcCursor*** pResult) = 0;
  /// <summary>Gets the name for the entity references by the cursor, e.g. foo for an 'int foo' variable.</summary>
  virtual HRESULT STDMETHODCALLTYPE GetSpelling(_Outptr_result_maybenull_ LPSTR* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE IsEqualTo(_In_ IDxcCursor* other, _Out_ BOOL* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE IsNull(_Out_ BOOL* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE IsDefinition(_Out_ BOOL* pResult) = 0;
  /// <summary>Gets the display name for the cursor, including e.g. parameter types for a function.</summary>
  virtual HRESULT STDMETHODCALLTYPE GetDisplayName(_Out_ BSTR* pResult) = 0;
  /// <summary>Gets the qualified name for the symbol the cursor refers to.</summary>
  virtual HRESULT STDMETHODCALLTYPE GetQualifiedName(BOOL includeTemplateArgs, _Outptr_result_maybenull_ BSTR* pResult) = 0;
  /// <summary>Gets a name for the cursor, applying the specified formatting flags.</summary>
  virtual HRESULT STDMETHODCALLTYPE GetFormattedName(DxcCursorFormatting formatting , _Outptr_result_maybenull_ BSTR* pResult) = 0;
  /// <summary>Gets children in pResult up to top elements.</summary>
  virtual HRESULT STDMETHODCALLTYPE GetChildren(
    unsigned skip, unsigned top,
    _Out_ unsigned* pResultLength, _Outptr_result_buffer_maybenull_(*pResultLength) IDxcCursor*** pResult) = 0;
  /// <summary>Gets the cursor following a location within a compound cursor.</summary>
  virtual HRESULT STDMETHODCALLTYPE GetSnappedChild(_In_ IDxcSourceLocation* location, _Outptr_result_maybenull_ IDxcCursor** pResult) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcDiagnostic, "4f76b234-3659-4d33-99b0-3b0db994b564")
struct IDxcDiagnostic : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE FormatDiagnostic(
    DxcDiagnosticDisplayOptions options,
    _Outptr_result_maybenull_ LPSTR* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetSeverity(_Out_ DxcDiagnosticSeverity* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetLocation(_Outptr_result_nullonfailure_ IDxcSourceLocation** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetSpelling(_Outptr_result_maybenull_ LPSTR* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetCategoryText(_Outptr_result_maybenull_ LPSTR* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetNumRanges(_Out_ unsigned* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetRangeAt(unsigned index, _Outptr_result_nullonfailure_ IDxcSourceRange** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetNumFixIts(_Out_ unsigned* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetFixItAt(unsigned index,
    _Outptr_result_nullonfailure_ IDxcSourceRange** pReplacementRange, _Outptr_result_maybenull_ LPSTR* pText) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcFile, "bb2fca9e-1478-47ba-b08c-2c502ada4895")
struct IDxcFile : public IUnknown
{
  /// <summary>Gets the file name for this file.</summary>
  virtual HRESULT STDMETHODCALLTYPE GetName(_Outptr_result_maybenull_ LPSTR* pResult) = 0;
  /// <summary>Checks whether this file is equal to the other specified file.</summary>
  virtual HRESULT STDMETHODCALLTYPE IsEqualTo(_In_ IDxcFile* other, _Out_ BOOL* pResult) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcInclusion, "0c364d65-df44-4412-888e-4e552fc5e3d6")
struct IDxcInclusion : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE GetIncludedFile(_Outptr_result_nullonfailure_ IDxcFile** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetStackLength(_Out_ unsigned *pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetStackItem(unsigned index, _Outptr_result_nullonfailure_ IDxcSourceLocation **pResult) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcIntelliSense, "b1f99513-46d6-4112-8169-dd0d6053f17d")
struct IDxcIntelliSense : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE CreateIndex(_Outptr_result_nullonfailure_ IDxcIndex** index) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetNullLocation(_Outptr_result_nullonfailure_ IDxcSourceLocation** location) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetNullRange(_Outptr_result_nullonfailure_ IDxcSourceRange** location) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetRange(
    _In_ IDxcSourceLocation* start,
    _In_ IDxcSourceLocation* end,
    _Outptr_result_nullonfailure_ IDxcSourceRange** location) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetDefaultDiagnosticDisplayOptions(
    _Out_ DxcDiagnosticDisplayOptions* pValue) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetDefaultEditingTUOptions(_Out_ DxcTranslationUnitFlags* pValue) = 0;
  virtual HRESULT STDMETHODCALLTYPE CreateUnsavedFile(_In_ LPCSTR fileName, _In_ LPCSTR contents, unsigned contentLength, _Outptr_result_nullonfailure_ IDxcUnsavedFile** pResult) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcIndex, "937824a0-7f5a-4815-9ba7-7fc0424f4173")
struct IDxcIndex : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE SetGlobalOptions(DxcGlobalOptions options) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetGlobalOptions(_Out_ DxcGlobalOptions* options) = 0;
  virtual HRESULT STDMETHODCALLTYPE ParseTranslationUnit(
      _In_z_ const char *source_filename,
      _In_count_(num_command_line_args) const char * const *command_line_args,
      int num_command_line_args,
      _In_count_(num_unsaved_files) IDxcUnsavedFile** unsaved_files,
      unsigned num_unsaved_files,
      DxcTranslationUnitFlags options,
      _Out_ IDxcTranslationUnit** pTranslationUnit) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcSourceLocation, "8e7ddf1c-d7d3-4d69-b286-85fccba1e0cf")
struct IDxcSourceLocation : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE IsEqualTo(_In_ IDxcSourceLocation* other, _Out_ BOOL* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetSpellingLocation(
    _Outptr_opt_ IDxcFile** pFile,
    _Out_opt_ unsigned* pLine,
    _Out_opt_ unsigned* pCol,
    _Out_opt_ unsigned* pOffset) = 0;
  virtual HRESULT STDMETHODCALLTYPE IsNull(_Out_ BOOL* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetPresumedLocation(
    _Outptr_opt_ LPSTR* pFilename,
    _Out_opt_ unsigned* pLine,
    _Out_opt_ unsigned* pCol) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcSourceRange, "f1359b36-a53f-4e81-b514-b6b84122a13f")
struct IDxcSourceRange : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE IsNull(_Out_ BOOL* pValue) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetStart(_Out_ IDxcSourceLocation** pValue) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetEnd(_Out_ IDxcSourceLocation** pValue) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetOffsets(_Out_ unsigned* startOffset, _Out_ unsigned* endOffset) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcToken, "7f90b9ff-a275-4932-97d8-3cfd234482a2")
struct IDxcToken : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE GetKind(_Out_ DxcTokenKind* pValue) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetLocation(_Out_ IDxcSourceLocation** pValue) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetExtent(_Out_ IDxcSourceRange** pValue) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetSpelling(_Out_ LPSTR* pValue) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcTranslationUnit, "9677dee0-c0e5-46a1-8b40-3db3168be63d")
struct IDxcTranslationUnit : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE GetCursor(_Out_ IDxcCursor** pCursor) = 0;
  virtual HRESULT STDMETHODCALLTYPE Tokenize(
    _In_ IDxcSourceRange* range,
    _Outptr_result_buffer_maybenull_(*pTokenCount) IDxcToken*** pTokens,
    _Out_ unsigned* pTokenCount) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetLocation(
    _In_ IDxcFile* file,
    unsigned line, unsigned column,
    _Outptr_result_nullonfailure_ IDxcSourceLocation** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetNumDiagnostics(_Out_ unsigned* pValue) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetDiagnostic(unsigned index, _Outptr_result_nullonfailure_ IDxcDiagnostic** pValue) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetFile(_In_ const char* name, _Outptr_result_nullonfailure_ IDxcFile** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetFileName(_Outptr_result_maybenull_ LPSTR* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE Reparse(
    _In_count_(num_unsaved_files) IDxcUnsavedFile** unsaved_files,
    unsigned num_unsaved_files) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetCursorForLocation(_In_ IDxcSourceLocation* location, _Outptr_result_nullonfailure_ IDxcCursor** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetLocationForOffset(_In_ IDxcFile* file, unsigned offset, _Outptr_result_nullonfailure_ IDxcSourceLocation** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetSkippedRanges(_In_ IDxcFile* file, _Out_ unsigned* pResultCount, _Outptr_result_buffer_(*pResultCount) IDxcSourceRange*** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetDiagnosticDetails(unsigned index, DxcDiagnosticDisplayOptions options,
    _Out_ unsigned* errorCode,
    _Out_ unsigned* errorLine,
    _Out_ unsigned* errorColumn,
    _Out_ BSTR* errorFile,
    _Out_ unsigned* errorOffset,
    _Out_ unsigned* errorLength,
    _Out_ BSTR* errorMessage) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetInclusionList(_Out_ unsigned* pResultCount, _Outptr_result_buffer_(*pResultCount) IDxcInclusion*** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE CodeCompleteAt(
      _In_ const char *fileName, unsigned line, unsigned column,
      _In_ IDxcUnsavedFile** pUnsavedFiles, unsigned numUnsavedFiles,
      _In_ DxcCodeCompleteFlags options,
      _Outptr_result_nullonfailure_ IDxcCodeCompleteResults **pResult) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcType, "2ec912fd-b144-4a15-ad0d-1c5439c81e46")
struct IDxcType : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE GetSpelling(_Outptr_result_z_ LPSTR* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE IsEqualTo(_In_ IDxcType* other, _Out_ BOOL* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetKind(_Out_ DxcTypeKind* pResult) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcUnsavedFile, "8ec00f98-07d0-4e60-9d7c-5a50b5b0017f")
struct IDxcUnsavedFile : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE GetFileName(_Outptr_result_z_ LPSTR* pFileName) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetContents(_Outptr_result_z_ LPSTR* pContents) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetLength(_Out_ unsigned* pLength) = 0;
};


CROSS_PLATFORM_UUIDOF(IDxcCodeCompleteResults, "1E06466A-FD8B-45F3-A78F-8A3F76EBB552")
struct IDxcCodeCompleteResults : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE GetNumResults(_Out_ unsigned* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetResultAt(unsigned index, _Outptr_result_nullonfailure_ IDxcCompletionResult** pResult) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcCompletionResult, "943C0588-22D0-4784-86FC-701F802AC2B6")
struct IDxcCompletionResult : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE GetCursorKind(_Out_ DxcCursorKind* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetCompletionString(_Outptr_result_nullonfailure_ IDxcCompletionString** pResult) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcCompletionString, "06B51E0F-A605-4C69-A110-CD6E14B58EEC")
struct IDxcCompletionString : public IUnknown
{
  virtual HRESULT STDMETHODCALLTYPE GetNumCompletionChunks(_Out_ unsigned* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetCompletionChunkKind(unsigned chunkNumber, _Out_ DxcCompletionChunkKind* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetCompletionChunkText(unsigned chunkNumber, _Out_ LPSTR* pResult) = 0;
};

// Fun fact: 'extern' is required because const is by default static in C++, so
// CLSID_DxcIntelliSense is not visible externally (this is OK in C, since const is
// not by default static in C)

#ifdef _MSC_VER
#define CLSID_SCOPE __declspec(selectany) extern
#else
#define CLSID_SCOPE
#endif

CLSID_SCOPE const CLSID
    CLSID_DxcIntelliSense = {/* 3047833c-d1c0-4b8e-9d40-102878605985 */
                             0x3047833c,
                             0xd1c0,
                             0x4b8e,
                             {0x9d, 0x40, 0x10, 0x28, 0x78, 0x60, 0x59, 0x85}};

#endif
