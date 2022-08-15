///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// CompilationResult.h                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file provides a class to parse a translation unit and return the     //
// diagnostic results and AST tree.                                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <atomic>

#include "dxc/Support/WinIncludes.h"

#include "dxc/dxcapi.h"
#include "dxc/dxcisense.h"
#include "dxc/Support/dxcapi.use.h"
#include "llvm/Support/Atomic.h"

inline HRESULT IFE(HRESULT hr) {
  if (FAILED(hr)) {
    throw std::runtime_error("COM call failed");
  }
  return hr;
}

inline HRESULT GetFirstChildFromCursor(IDxcCursor *cursor,
                                       IDxcCursor **pResult) {
  HRESULT hr;
  IDxcCursor **children = nullptr;
  unsigned childrenCount;
  hr = cursor->GetChildren(0, 1, &childrenCount, &children);
  if (SUCCEEDED(hr) && childrenCount == 1) {
    *pResult = children[0];
  } else {
    *pResult = nullptr;
    hr = E_FAIL;
  }
  CoTaskMemFree(children);
  return hr;
}

class TrivialDxcUnsavedFile : IDxcUnsavedFile
{
private:
  volatile std::atomic<llvm::sys::cas_flag> m_dwRef;
  LPCSTR m_fileName;
  LPCSTR m_contents;
  unsigned m_length;
public:
  TrivialDxcUnsavedFile(LPCSTR fileName, LPCSTR contents)
    : m_dwRef(0), m_fileName(fileName), m_contents(contents)
  {
    m_length = (unsigned)strlen(m_contents);
  }

  static HRESULT Create(LPCSTR fileName, LPCSTR contents, IDxcUnsavedFile** pResult)
  {
    CComPtr<TrivialDxcUnsavedFile> pNewValue = new TrivialDxcUnsavedFile(fileName, contents);
    return pNewValue.QueryInterface(pResult);
  }
  ULONG STDMETHODCALLTYPE AddRef() { return (ULONG)++m_dwRef; }
  ULONG STDMETHODCALLTYPE Release() { 
    ULONG result = (ULONG)--m_dwRef;
    if (result == 0) delete this;
    return result;
  }
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** ppvObject)
  {
    if (ppvObject == nullptr) return E_POINTER;
    if (IsEqualIID(iid, __uuidof(IUnknown)) ||
        IsEqualIID(iid, __uuidof(INoMarshal)) ||
        IsEqualIID(iid, __uuidof(IDxcUnsavedFile)))
    {
      *ppvObject = reinterpret_cast<IDxcUnsavedFile*>(this);
      reinterpret_cast<IDxcUnsavedFile*>(this)->AddRef();
      return S_OK;
    }

    return E_NOINTERFACE;
  }
  HRESULT STDMETHODCALLTYPE GetFileName(LPSTR* pFileName)
  {
    *pFileName = (LPSTR)CoTaskMemAlloc(1 + strlen(m_fileName));
    strcpy(*pFileName, m_fileName);
    return S_OK;
  }
  HRESULT STDMETHODCALLTYPE GetContents(LPSTR* pContents)
  {
    *pContents = (LPSTR)CoTaskMemAlloc(m_length + 1);
    memcpy(*pContents, m_contents, m_length + 1);
    return S_OK;
  }
  HRESULT STDMETHODCALLTYPE GetLength(unsigned* pLength)
  {
    *pLength = m_length;
    return S_OK;
  }
};

class HlslIntellisenseSupport : public dxc::DxcDllSupport {
public:
  HlslIntellisenseSupport() {}
  HlslIntellisenseSupport(HlslIntellisenseSupport &&other)
      : dxc::DxcDllSupport(std::move(other)) {}

  HRESULT CreateIntellisense(_Outptr_ IDxcIntelliSense **pResult) {
    return CreateInstance(CLSID_DxcIntelliSense, pResult);
  }
};

/// Summary of the results of a clang compilation operation.
class CompilationResult
{
private:
  // Keep Intellisense alive.
  std::shared_ptr<HlslIntellisenseSupport> IsenseSupport;

  /// The top-level Intellisense object.
  CComPtr<IDxcIntelliSense> Intellisense;
  /// The libclang index for the compilation operation.
  CComPtr<IDxcIndex> Index;

  /// The number of diagnostic messages emitted.
  unsigned NumDiagnostics;

  /// The number of diagnostic messages emitted that indicate errors.
  unsigned NumErrorDiagnostics;

  /// The diagnostic messages emitted.
  std::vector<std::string> DiagnosticMessages;

  /// The severity of diagnostic messages.
  std::vector<DxcDiagnosticSeverity> DiagnosticSeverities;

  // Hide the copy constructor.
  CompilationResult(const CompilationResult&);

public:
  CompilationResult(std::shared_ptr<HlslIntellisenseSupport> support, IDxcIntelliSense* isense, IDxcIndex* index, IDxcTranslationUnit* tu) :
    IsenseSupport(support),
    Intellisense(isense),
    Index(index),
    NumErrorDiagnostics(0),
    TU(tu)
  {
    if (tu) {
      IFE(tu->GetNumDiagnostics(&NumDiagnostics));
    } else {
      NumDiagnostics = 0;
    }

    DxcDiagnosticDisplayOptions diagnosticOptions;
    IFE(isense->GetDefaultDiagnosticDisplayOptions(&diagnosticOptions));
    for (unsigned i = 0; i < NumDiagnostics; i++) {
      CComPtr<IDxcDiagnostic> diagnostic;
      IFE(tu->GetDiagnostic(i, &diagnostic));

      LPSTR format;
      IFE(diagnostic->FormatDiagnostic(diagnosticOptions, &format));
      DiagnosticMessages.push_back(std::string(format));
      CoTaskMemFree(format);

      DxcDiagnosticSeverity severity;
      IFE(diagnostic->GetSeverity(&severity));
      DiagnosticSeverities.push_back(severity);
      if (IsErrorSeverity(severity)) {
        NumErrorDiagnostics++;
      }
    }

    assert(NumErrorDiagnostics <= NumDiagnostics && "else counting code in loop is incorrect");
    assert(DiagnosticMessages.size() == NumDiagnostics && "else some diagnostics have no message");
    assert(DiagnosticSeverities.size() == NumDiagnostics && "else some diagnostics have no severity");
  }
  
  CompilationResult(CompilationResult&& other) :
    IsenseSupport(std::move(other.IsenseSupport)) {
    // Allow move constructor.
    Intellisense = other.Intellisense; other.Intellisense = nullptr;
    Index = other.Index; other.Index = nullptr;
    TU = other.TU; other.TU = nullptr;
    NumDiagnostics = other.NumDiagnostics;
    NumErrorDiagnostics = other.NumErrorDiagnostics;
    DiagnosticMessages = std::move(other.DiagnosticMessages);
    DiagnosticSeverities = std::move(other.DiagnosticSeverities);
  }

  ~CompilationResult() { Dispose(); }

  /// The translation unit resulting from the compilation.
  CComPtr<IDxcTranslationUnit> TU;

  static const char *getDefaultFileName() { return "filename.hlsl"; }

  static std::shared_ptr<HlslIntellisenseSupport> DefaultHlslSupport;

  static std::shared_ptr<HlslIntellisenseSupport> GetHlslSupport() {
    if (DefaultHlslSupport.get() != nullptr) {
      return DefaultHlslSupport;
    }
    std::shared_ptr<HlslIntellisenseSupport> result = std::make_shared<HlslIntellisenseSupport>();
    IFE(result->Initialize());
    return result;
  }

  static CompilationResult CreateForProgramAndArgs(const char* text, size_t textLen,
    _In_count_(commandLineArgsCount) const char* commandLineArgs[],
    unsigned commandLineArgsCount,
    _In_opt_ DxcTranslationUnitFlags* options = nullptr)
  {
    std::shared_ptr<HlslIntellisenseSupport> support(GetHlslSupport());

    CComPtr<IDxcIntelliSense> isense;
    IFE(support->CreateIntellisense(&isense));
    CComPtr<IDxcIndex> tuIndex;
    CComPtr<IDxcTranslationUnit> tu;
    const char *fileName = getDefaultFileName();
    if (textLen == 0) textLen = strlen(text);

    IFE(isense->CreateIndex(&tuIndex));

    CComPtr<IDxcUnsavedFile> unsavedFile;
    IFE(TrivialDxcUnsavedFile::Create(fileName, text, &unsavedFile));

    DxcTranslationUnitFlags localOptions;
    if (options == nullptr)
    {
      IFE(isense->GetDefaultEditingTUOptions(&localOptions));
    }
    else
    {
      localOptions = *options;
    }
    IFE(tuIndex->ParseTranslationUnit(fileName, 
      commandLineArgs, commandLineArgsCount,
      &(unsavedFile.p), 1, localOptions, &tu));
    
    return CompilationResult(support, isense.p, tuIndex.p, tu.p);
  }

  static CompilationResult
  CreateForProgram(const char *text, size_t textLen,
                   _In_opt_ DxcTranslationUnitFlags *options = nullptr) {
    const char *commandLineArgs[] = {"-c", "-ferror-limit=200"};
    unsigned commandLineArgsCount = _countof(commandLineArgs);
    return CreateForProgramAndArgs(text, textLen, commandLineArgs,
                                   commandLineArgsCount, options);
  }

  static CompilationResult CreateForCommandLine(char* arguments, const char* fileName) {
    std::shared_ptr<HlslIntellisenseSupport> support(GetHlslSupport());
    return CreateForCommandLine(arguments, fileName, support);
  }

  static CompilationResult CreateForCommandLine(char* arguments, const char* fileName, std::shared_ptr<HlslIntellisenseSupport> support) {
    CComPtr<IDxcIntelliSense> isense;
    IFE(support->CreateIntellisense(&isense));
    CComPtr<IDxcIndex> tuIndex;
    CComPtr<IDxcTranslationUnit> tu;
    const char* commandLineArgs[32];
    unsigned commandLineArgsCount = 0;
    char* nextArg = arguments;

    // Set a very high number of errors to avoid giving up too early.
    commandLineArgs[commandLineArgsCount++] = "-ferror-limit=2000";

    // Turn on spell checking for compatibility. This produces error messages
    // that suggest corrected spellings.
    commandLineArgs[commandLineArgsCount++] = "-fspell-checking";

    // Turn off color diagnostics to avoid control characters in diagnostic
    // stream.
    commandLineArgs[commandLineArgsCount++] = "-fno-color-diagnostics";

    IFE(isense->CreateIndex(&tuIndex));

    // Split command line arguments by spaces
    if (nextArg) {
      // skip leading spaces
      while (*nextArg == ' ')
        nextArg++;
      commandLineArgs[commandLineArgsCount++] = nextArg;
      while ((*nextArg != '\0')) {
        if (*nextArg == ' ') {
          *nextArg = 0;
          commandLineArgs[commandLineArgsCount++] = nextArg + 1;
        }
        nextArg++;
      }
    }

    DxcTranslationUnitFlags options;
    IFE(isense->GetDefaultEditingTUOptions(&options));
    IFE(tuIndex->ParseTranslationUnit(fileName, commandLineArgs,
                                      commandLineArgsCount, nullptr, 0, options,
                                      &tu));

    return CompilationResult(support, isense.p, tuIndex.p, tu.p);
  }

  bool IsTUAvailable() const { return TU != nullptr; }
  bool ParseSucceeded() const { return TU != nullptr && NumErrorDiagnostics == 0; }

  static bool IsErrorSeverity(DxcDiagnosticSeverity severity)
  {
    return (severity == DxcDiagnostic_Error || severity == DxcDiagnostic_Fatal);
  }

  /// Gets a string with all error messages concatenated, one per line.
  std::string GetTextForErrors() const {
    assert(DiagnosticMessages.size() == NumDiagnostics && "otherwise some diagnostics have no message");
    assert(DiagnosticSeverities.size() == NumDiagnostics && "otherwise some diagnostics have no severity");

    std::stringstream ostr;
    for (size_t i = 0; i < DiagnosticMessages.size(); i++) {
      if (IsErrorSeverity(DiagnosticSeverities[i])) {
        ostr << DiagnosticMessages[i] << '\n';
      }
    }

    return ostr.str();
  }

  /// Releases resources, including the translation unit AST.
  void Dispose()
  {
    TU = nullptr;
    Index = nullptr;
    Intellisense = nullptr;
    IsenseSupport = nullptr;
  }

private:
#if SUPPORTS_CURSOR_WALK
  struct CursorStringData {
    std::stringstream &ostr;
    std::vector<CXCursor> cursors;
    CursorStringData(std::stringstream &the_ostr) : ostr(the_ostr) {}
  };

  static
  CXChildVisitResult AppendCursorStringCallback(
    CXCursor cursor, CXCursor parent, CXClientData client_data)
  {
    CursorStringData* d = (CursorStringData*)client_data;
    auto cursorsStart = std::begin(d->cursors);
    auto cursorsEnd = std::end(d->cursors);
    auto parentLocation = std::find(cursorsStart, cursorsEnd, parent);
    assert(parentLocation != cursorsEnd && "otherwise the parent was not visited previously");
    AppendCursorString(cursor, parentLocation - cursorsStart, d->ostr);
    d->cursors.resize(1 + (parentLocation - cursorsStart));
    d->cursors.push_back(cursor);
    return CXChildVisit_Recurse;
  }
#endif
  static
  void AppendCursorString(IDxcCursor* cursor, size_t indent, std::stringstream& ostr)
  {
    if (indent > 0) {
      std::streamsize prior = ostr.width();
      ostr.width(indent);
      ostr << ' ';
      ostr.width(prior);
    }

    CComPtr<IDxcType> type;
    DxcCursorKind kind;
    cursor->GetKind(&kind);
    cursor->GetCursorType(&type);
    switch (kind)
    {
    case DxcCursor_UnexposedDecl: ostr << "UnexposedDecl"; break;
    case DxcCursor_StructDecl: ostr << "StructDecl"; break;
    case DxcCursor_UnionDecl: ostr << "UnionDecl"; break; 
    case DxcCursor_ClassDecl: ostr << "ClassDecl"; break;
    case DxcCursor_EnumDecl: ostr << "EnumDecl"; break;
    case DxcCursor_FieldDecl: ostr << "FieldDecl"; break;
    case DxcCursor_EnumConstantDecl: ostr << "EnumConstantDecl"; break;
    case DxcCursor_FunctionDecl: ostr << "FunctionDecl"; break;
    case DxcCursor_VarDecl: ostr << "VarDecl"; break;
    case DxcCursor_ParmDecl: ostr << "ParmDecl"; break;
    case DxcCursor_ObjCInterfaceDecl: ostr << "ObjCInterfaceDecl"; break;
    case DxcCursor_ObjCCategoryDecl: ostr << "ObjCCategoryDecl"; break;
    case DxcCursor_ObjCProtocolDecl: ostr << "ObjCProtocolDecl"; break;
    case DxcCursor_ObjCPropertyDecl: ostr << "ObjCPropertyDecl"; break;
    case DxcCursor_ObjCIvarDecl: ostr << "ObjCIvarDecl"; break;
    case DxcCursor_ObjCInstanceMethodDecl: ostr << "ObjCInstanceMethodDecl"; break;
    case DxcCursor_ObjCClassMethodDecl: ostr << "ObjCClassMethodDecl"; break;
    case DxcCursor_ObjCImplementationDecl: ostr << "ObjCImplementationDecl"; break;
    case DxcCursor_ObjCCategoryImplDecl: ostr << "ObjCCategoryImplDecl"; break;
    case DxcCursor_TypedefDecl: ostr << "TypedefDecl"; break;
    case DxcCursor_CXXMethod: ostr << "CXXMethod"; break;
    case DxcCursor_Namespace: ostr << "Namespace"; break;
    case DxcCursor_LinkageSpec: ostr << "LinkageSpec"; break;
    case DxcCursor_Constructor: ostr << "Constructor"; break;
    case DxcCursor_Destructor: ostr << "Destructor"; break;
    case DxcCursor_ConversionFunction: ostr << "ConversionFunction"; break;
    case DxcCursor_TemplateTypeParameter: ostr << "TemplateTypeParameter"; break;
    case DxcCursor_NonTypeTemplateParameter: ostr << "NonTypeTemplateParameter"; break;
    case DxcCursor_TemplateTemplateParameter: ostr << "TemplateTemplateParameter"; break;
    case DxcCursor_FunctionTemplate: ostr << "FunctionTemplate"; break;
    case DxcCursor_ClassTemplate: ostr << "ClassTemplate"; break;
    case DxcCursor_ClassTemplatePartialSpecialization: ostr << "ClassTemplatePartialSpecialization"; break;
    case DxcCursor_NamespaceAlias: ostr << "NamespaceAlias"; break;
    case DxcCursor_UsingDirective: ostr << "UsingDirective"; break;
    case DxcCursor_UsingDeclaration: ostr << "UsingDeclaration"; break;
    case DxcCursor_TypeAliasDecl: ostr << "TypeAliasDecl"; break;
    case DxcCursor_ObjCSynthesizeDecl: ostr << "ObjCSynthesizeDecl"; break;
    case DxcCursor_ObjCDynamicDecl: ostr << "ObjCDynamicDecl"; break;
    case DxcCursor_CXXAccessSpecifier: ostr << "CXXAccessSpecifier"; break;
    case DxcCursor_ObjCSuperClassRef: ostr << "ObjCSuperClassRef"; break;
    case DxcCursor_ObjCProtocolRef: ostr << "ObjCProtocolRef"; break;
    case DxcCursor_ObjCClassRef: ostr << "ObjCClassRef"; break;
    case DxcCursor_TypeRef: ostr << "TypeRef"; break;
    case DxcCursor_CXXBaseSpecifier: ostr << "CXXBaseSpecifier"; break;
    case DxcCursor_TemplateRef: ostr << "TemplateRef"; break;
    case DxcCursor_NamespaceRef: ostr << "NamespaceRef"; break;
    case DxcCursor_MemberRef: ostr << "MemberRef"; break;
    case DxcCursor_LabelRef: ostr << "LabelRef"; break;
    case DxcCursor_OverloadedDeclRef: ostr << "OverloadedDeclRef"; break;
    case DxcCursor_VariableRef: ostr << "VariableRef"; break;
    case DxcCursor_InvalidFile: ostr << "InvalidFile"; break;
    case DxcCursor_NoDeclFound: ostr << "NoDeclFound"; break;
    case DxcCursor_NotImplemented: ostr << "NotImplemented"; break;
    case DxcCursor_InvalidCode: ostr << "InvalidCode"; break;
    case DxcCursor_UnexposedExpr: ostr << "UnexposedExpr"; break;
    case DxcCursor_DeclRefExpr: ostr << "DeclRefExpr"; break;
    case DxcCursor_MemberRefExpr: ostr << "MemberRefExpr"; break;
    case DxcCursor_CallExpr: ostr << "CallExpr"; break;
    case DxcCursor_ObjCMessageExpr: ostr << "ObjCMessageExpr"; break;
    case DxcCursor_BlockExpr: ostr << "BlockExpr"; break;
    case DxcCursor_IntegerLiteral: ostr << "IntegerLiteral"; break;
    case DxcCursor_FloatingLiteral: ostr << "FloatingLiteral"; break;
    case DxcCursor_ImaginaryLiteral: ostr << "ImaginaryLiteral"; break;
    case DxcCursor_StringLiteral: ostr << "StringLiteral"; break;
    case DxcCursor_CharacterLiteral: ostr << "CharacterLiteral"; break;
    case DxcCursor_ParenExpr: ostr << "ParenExpr"; break;
    case DxcCursor_UnaryOperator: ostr << "UnaryOperator"; break;
    case DxcCursor_ArraySubscriptExpr: ostr << "ArraySubscriptExpr"; break;
    case DxcCursor_BinaryOperator: ostr << "BinaryOperator"; break;
    case DxcCursor_CompoundAssignOperator: ostr << "CompoundAssignOperator"; break;
    case DxcCursor_ConditionalOperator: ostr << "ConditionalOperator"; break;
    case DxcCursor_CStyleCastExpr: ostr << "CStyleCastExpr"; break;
    case DxcCursor_CompoundLiteralExpr: ostr << "CompoundLiteralExpr"; break;
    case DxcCursor_InitListExpr: ostr << "InitListExpr"; break;
    case DxcCursor_AddrLabelExpr: ostr << "AddrLabelExpr"; break;
    case DxcCursor_StmtExpr: ostr << "StmtExpr"; break;
    case DxcCursor_GenericSelectionExpr: ostr << "GenericSelectionExpr"; break;
    case DxcCursor_GNUNullExpr: ostr << "GNUNullExpr"; break;
    case DxcCursor_CXXStaticCastExpr: ostr << "CXXStaticCastExpr"; break;
    case DxcCursor_CXXDynamicCastExpr: ostr << "CXXDynamicCastExpr"; break;
    case DxcCursor_CXXReinterpretCastExpr: ostr << "CXXReinterpretCastExpr"; break;
    case DxcCursor_CXXConstCastExpr: ostr << "CXXConstCastExpr"; break;
    case DxcCursor_CXXFunctionalCastExpr: ostr << "CXXFunctionalCastExpr"; break;
    case DxcCursor_CXXTypeidExpr: ostr << "CXXTypeidExpr"; break;
    case DxcCursor_CXXBoolLiteralExpr: ostr << "CXXBoolLiteralExpr"; break;
    case DxcCursor_CXXNullPtrLiteralExpr: ostr << "CXXNullPtrLiteralExpr"; break;
    case DxcCursor_CXXThisExpr: ostr << "CXXThisExpr"; break;
    case DxcCursor_CXXThrowExpr: ostr << "CXXThrowExpr"; break;
    case DxcCursor_CXXNewExpr: ostr << "CXXNewExpr"; break;
    case DxcCursor_CXXDeleteExpr: ostr << "CXXDeleteExpr"; break;
    case DxcCursor_UnaryExpr: ostr << "UnaryExpr"; break;
    case DxcCursor_ObjCStringLiteral: ostr << "ObjCStringLiteral"; break;
    case DxcCursor_ObjCEncodeExpr: ostr << "ObjCEncodeExpr"; break;
    case DxcCursor_ObjCSelectorExpr: ostr << "ObjCSelectorExpr"; break;
    case DxcCursor_ObjCProtocolExpr: ostr << "ObjCProtocolExpr"; break;
    case DxcCursor_ObjCBridgedCastExpr: ostr << "ObjCBridgedCastExpr"; break;
    case DxcCursor_PackExpansionExpr: ostr << "PackExpansionExpr"; break;
    case DxcCursor_SizeOfPackExpr: ostr << "SizeOfPackExpr"; break;
    case DxcCursor_LambdaExpr: ostr << "LambdaExpr"; break;
    case DxcCursor_ObjCBoolLiteralExpr: ostr << "ObjCBoolLiteralExpr"; break;
    case DxcCursor_ObjCSelfExpr: ostr << "ObjCSelfExpr"; break;
    case DxcCursor_UnexposedStmt: ostr << "UnexposedStmt"; break;
    case DxcCursor_LabelStmt: ostr << "LabelStmt"; break;
    case DxcCursor_CompoundStmt: ostr << "CompoundStmt"; break;
    case DxcCursor_CaseStmt: ostr << "CaseStmt"; break;
    case DxcCursor_DefaultStmt: ostr << "DefaultStmt"; break;
    case DxcCursor_IfStmt: ostr << "IfStmt"; break;
    case DxcCursor_SwitchStmt: ostr << "SwitchStmt"; break;
    case DxcCursor_WhileStmt: ostr << "WhileStmt"; break;
    case DxcCursor_DoStmt: ostr << "DoStmt"; break;
    case DxcCursor_ForStmt: ostr << "ForStmt"; break;
    case DxcCursor_GotoStmt: ostr << "GotoStmt"; break;
    case DxcCursor_IndirectGotoStmt: ostr << "IndirectGotoStmt"; break;
    case DxcCursor_ContinueStmt: ostr << "ContinueStmt"; break;
    case DxcCursor_BreakStmt: ostr << "BreakStmt"; break;
    case DxcCursor_ReturnStmt: ostr << "ReturnStmt"; break;
    case DxcCursor_GCCAsmStmt: ostr << "GCCAsmStmt"; break;
    case DxcCursor_ObjCAtTryStmt: ostr << "ObjCAtTryStmt"; break;
    case DxcCursor_ObjCAtCatchStmt: ostr << "ObjCAtCatchStmt"; break;
    case DxcCursor_ObjCAtFinallyStmt: ostr << "ObjCAtFinallyStmt"; break;
    case DxcCursor_ObjCAtThrowStmt: ostr << "ObjCAtThrowStmt"; break;
    case DxcCursor_ObjCAtSynchronizedStmt: ostr << "ObjCAtSynchronizedStmt"; break;
    case DxcCursor_ObjCAutoreleasePoolStmt: ostr << "ObjCAutoreleasePoolStmt"; break;
    case DxcCursor_ObjCForCollectionStmt: ostr << "ObjCForCollectionStmt"; break;
    case DxcCursor_CXXCatchStmt: ostr << "CXXCatchStmt"; break;
    case DxcCursor_CXXTryStmt: ostr << "CXXTryStmt"; break;
    case DxcCursor_CXXForRangeStmt: ostr << "CXXForRangeStmt"; break;
    case DxcCursor_SEHTryStmt: ostr << "SEHTryStmt"; break;
    case DxcCursor_SEHExceptStmt: ostr << "SEHExceptStmt"; break;
    case DxcCursor_SEHFinallyStmt: ostr << "SEHFinallyStmt"; break;
    case DxcCursor_MSAsmStmt: ostr << "MSAsmStmt"; break;
    case DxcCursor_NullStmt: ostr << "NullStmt"; break;
    case DxcCursor_DeclStmt: ostr << "DeclStmt"; break;
    case DxcCursor_OMPParallelDirective: ostr << "OMPParallelDirective"; break;
    case DxcCursor_TranslationUnit: ostr << "TranslationUnit"; break;
    case DxcCursor_UnexposedAttr: ostr << "UnexposedAttr"; break;
#if 0
  CXCursor_IBActionAttr                  = 401,
  CXCursor_IBOutletAttr                  = 402,
  CXCursor_IBOutletCollectionAttr        = 403,
  CXCursor_CXXFinalAttr                  = 404,
  CXCursor_CXXOverrideAttr               = 405,
  CXCursor_AnnotateAttr                  = 406,
  CXCursor_AsmLabelAttr                  = 407,
  CXCursor_PackedAttr                    = 408,
     
  /* Preprocessing */
  CXCursor_PreprocessingDirective        = 500,
  CXCursor_MacroDefinition               = 501,
  CXCursor_MacroExpansion                = 502,
  CXCursor_InclusionDirective            = 503,
    case CXCursor_FunctionDecl:
      break;
    case CXCursor_ClassDecl:
      ostr << "class decl";
      break;
    case CXCursor_TranslationUnit:
      ostr << "translation unit";
      break;
#endif 
    default:
      ostr << "unknown/unhandled cursor kind " << kind;
      break;
    }

    DxcTypeKind typeKind;
    if (type != nullptr && SUCCEEDED(type->GetKind(&typeKind)) &&
        typeKind != DxcTypeKind_Invalid) {
      LPSTR name;
      type->GetSpelling(&name);
      ostr << " [type " << name << "]";
      CoTaskMemFree(name);
    }

    ostr << '\n';

    // Recurse.
    IDxcCursor** children = nullptr;
    unsigned childrenCount;
    cursor->GetChildren(0, 64, &childrenCount, &children);
    for (unsigned i = 0; i < childrenCount; i++) {
      AppendCursorString(children[i], indent + 1, ostr);
      children[i]->Release();
    }
    CoTaskMemFree(children);
  }

public:
  std::string BuildASTString() {
    if (TU == nullptr) {
      return "<failed to build - TU is null>";
    }

    CComPtr<IDxcCursor> cursor;
    std::stringstream ostr;
    this->TU->GetCursor(&cursor);
    AppendCursorString(cursor, 0, ostr);
    return ostr.str();
  }
};
