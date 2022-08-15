///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcTestUtils.h                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file provides utility functions for testing dxc APIs.                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>
#include <vector>
#include <map>
#include "dxc/dxcapi.h"
#include "dxc/Support/dxcapi.use.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"

namespace hlsl {
namespace options {
class DxcOpts;
class MainArgs;
}
}

/// Use this class to run a FileCheck invocation in memory.
class FileCheckForTest {
public:
  FileCheckForTest();

  std::string CheckFilename;
  /// File to check (defaults to stdin)
  std::string InputFilename;
  /// Prefix to use from check file (defaults to 'CHECK')
  std::vector<std::string> CheckPrefixes;
  /// Do not treat all horizontal whitespace as equivalent
  bool NoCanonicalizeWhiteSpace;
  /// Add an implicit negative check with this pattern to every
  /// positive check. This can be used to ensure that no instances of
  /// this pattern occur which are not matched by a positive pattern
  std::vector<std::string> ImplicitCheckNot;
  /// Allow the input file to be empty. This is useful when making
  /// checks that some error message does not occur, for example.
  bool AllowEmptyInput;

  /// VariableTable - This holds all the current filecheck variables.
  llvm::StringMap<std::string> VariableTable;

  /// String to read in place of standard input.
  std::string InputForStdin;
  /// Output stream.
  std::string test_outs;
  /// Output stream.
  std::string test_errs;

  int Run();
};

// wstring because most uses need UTF-16: IDxcResult output names, include handler
typedef std::map<std::wstring, CComPtr<IDxcBlob>> FileMap;

// The result of running a single command in a run pipeline
struct FileRunCommandResult {
  CComPtr<IDxcOperationResult> OpResult; // The operation result, if any.
  std::string StdOut;
  std::string StdErr;
  int ExitCode = 0;
  bool AbortPipeline = false; // True to prevent running subsequent commands

  static inline FileRunCommandResult Success() {
    FileRunCommandResult result;
    result.ExitCode = 0;
    return result;
  }

  static inline FileRunCommandResult Success(std::string StdOut) {
    FileRunCommandResult result;
    result.ExitCode = 0;
    result.StdOut = std::move(StdOut);
    return result;
  }

  static inline FileRunCommandResult Error(int ExitCode, std::string StdErr) {
    FileRunCommandResult result;
    result.ExitCode = ExitCode;
    result.StdErr = std::move(StdErr);
    return result;
  }

  static inline FileRunCommandResult Error(std::string StdErr) {
    return Error(1, StdErr);
  }
};

typedef std::map<std::string, std::string> PluginToolsPaths;

class FileRunCommandPart {
public:
  FileRunCommandPart(const std::string &command, const std::string &arguments, LPCWSTR commandFileName);
  FileRunCommandPart(const FileRunCommandPart&) = default;
  FileRunCommandPart(FileRunCommandPart&&) = default;
  
  FileRunCommandResult Run(dxc::DxcDllSupport &DllSupport, const FileRunCommandResult *Prior, PluginToolsPaths *pPluginToolsPaths = nullptr, LPCWSTR dumpName = nullptr);
  FileRunCommandResult RunHashTests(dxc::DxcDllSupport &DllSupport);
  
  FileRunCommandResult ReadOptsForDxc(hlsl::options::MainArgs &argStrings, hlsl::options::DxcOpts &Opts, unsigned flagsToInclude = 0);

  std::string Command;      // Command to run, eg %dxc
  std::string Arguments;    // Arguments to command
  LPCWSTR CommandFileName;  // File name replacement for %s
  FileMap *pVFS = nullptr;  // Files in virtual file system

private:
  FileRunCommandResult RunFileChecker(const FileRunCommandResult *Prior, LPCWSTR dumpName = nullptr);
  FileRunCommandResult RunDxc(dxc::DxcDllSupport &DllSupport, const FileRunCommandResult *Prior);
  FileRunCommandResult RunDxv(dxc::DxcDllSupport &DllSupport, const FileRunCommandResult *Prior);
  FileRunCommandResult RunOpt(dxc::DxcDllSupport &DllSupport, const FileRunCommandResult *Prior);
  FileRunCommandResult RunListParts(dxc::DxcDllSupport &DllSupport, const FileRunCommandResult *Prior);
  FileRunCommandResult RunD3DReflect(dxc::DxcDllSupport &DllSupport, const FileRunCommandResult *Prior);
  FileRunCommandResult RunDxr(dxc::DxcDllSupport &DllSupport, const FileRunCommandResult *Prior);
  FileRunCommandResult RunLink(dxc::DxcDllSupport &DllSupport, const FileRunCommandResult *Prior);
  FileRunCommandResult RunTee(const FileRunCommandResult *Prior);
  FileRunCommandResult RunXFail(const FileRunCommandResult *Prior);
  FileRunCommandResult RunDxilVer(dxc::DxcDllSupport& DllSupport, const FileRunCommandResult* Prior);
  FileRunCommandResult RunDxcHashTest(dxc::DxcDllSupport &DllSupport);
  FileRunCommandResult RunFromPath(const std::string &path, const FileRunCommandResult *Prior);
  FileRunCommandResult RunFileCompareText(const FileRunCommandResult *Prior);
#ifdef _WIN32
  FileRunCommandResult RunFxc(dxc::DxcDllSupport &DllSupport, const FileRunCommandResult* Prior);
#endif

  void SubstituteFilenameVars(std::string &args);
#ifdef _WIN32
  bool ReadFileContentToString(HANDLE hFile, std::string &str);
#endif
};

void ParseCommandParts(LPCSTR commands, LPCWSTR fileName, std::vector<FileRunCommandPart> &parts);
void ParseCommandPartsFromFile(LPCWSTR fileName, std::vector<FileRunCommandPart> &parts);

class FileRunTestResult {
public:
  std::string ErrorMessage;
  int RunResult;
  static FileRunTestResult RunHashTestFromFileCommands(LPCWSTR fileName);
  static FileRunTestResult RunFromFileCommands(LPCWSTR fileName,
                                               PluginToolsPaths *pPluginToolsPaths = nullptr,
                                               LPCWSTR dumpName = nullptr);
  static FileRunTestResult RunFromFileCommands(LPCWSTR fileName,
                                               dxc::DxcDllSupport &dllSupport,
                                               PluginToolsPaths *pPluginToolsPaths = nullptr,
                                               LPCWSTR dumpName = nullptr);
};

void AssembleToContainer(dxc::DxcDllSupport &dllSupport, IDxcBlob *pModule, IDxcBlob **pContainer);
std::string BlobToUtf8(_In_ IDxcBlob *pBlob);
std::wstring BlobToWide(_In_ IDxcBlob *pBlob);
void CheckOperationSucceeded(IDxcOperationResult *pResult, IDxcBlob **ppBlob);
bool CheckOperationResultMsgs(IDxcOperationResult *pResult,
                              llvm::ArrayRef<LPCSTR> pErrorMsgs,
                              bool maySucceedAnyway, bool bRegex);
bool CheckOperationResultMsgs(IDxcOperationResult *pResult,
                              const LPCSTR *pErrorMsgs, size_t errorMsgCount,
                              bool maySucceedAnyway, bool bRegex);
bool CheckMsgs(const LPCSTR pText, size_t TextCount, const LPCSTR *pErrorMsgs,
               size_t errorMsgCount, bool bRegex);
bool CheckNotMsgs(const LPCSTR pText, size_t TextCount, const LPCSTR *pErrorMsgs,
               size_t errorMsgCount, bool bRegex);
void GetDxilPart(dxc::DxcDllSupport &dllSupport, IDxcBlob *pProgram, IDxcBlob **pDxilPart);
std::string DisassembleProgram(dxc::DxcDllSupport &dllSupport, IDxcBlob *pProgram);
void SplitPassList(LPWSTR pPassesBuffer, std::vector<LPCWSTR> &passes);
void MultiByteStringToBlob(dxc::DxcDllSupport &dllSupport, const std::string &val, UINT32 codePoint, _Outptr_ IDxcBlob **ppBlob);
void MultiByteStringToBlob(dxc::DxcDllSupport &dllSupport, const std::string &val, UINT32 codePoint, _Outptr_ IDxcBlobEncoding **ppBlob);
void Utf8ToBlob(dxc::DxcDllSupport &dllSupport, const std::string &val, _Outptr_ IDxcBlob **ppBlob);
void Utf8ToBlob(dxc::DxcDllSupport &dllSupport, const std::string &val, _Outptr_ IDxcBlobEncoding **ppBlob);
void Utf8ToBlob(dxc::DxcDllSupport &dllSupport, const char *pVal, _Outptr_ IDxcBlobEncoding **ppBlob);
void WideToBlob(dxc::DxcDllSupport &dllSupport, const std::wstring &val, _Outptr_ IDxcBlob **ppBlob);
void WideToBlob(dxc::DxcDllSupport &dllSupport, const std::wstring &val, _Outptr_ IDxcBlobEncoding **ppBlob);
void VerifyCompileOK(dxc::DxcDllSupport &dllSupport, LPCSTR pText,
                     LPWSTR pTargetProfile, LPCWSTR pArgs,
                     _Outptr_ IDxcBlob **ppResult);
void VerifyCompileOK(dxc::DxcDllSupport &dllSupport, LPCSTR pText,
                     LPWSTR pTargetProfile, std::vector<LPCWSTR> &args,
                     _Outptr_ IDxcBlob **ppResult);

HRESULT GetVersion(dxc::DxcDllSupport& DllSupport, REFCLSID clsid, unsigned &Major, unsigned &Minor);
bool ParseTargetProfile(llvm::StringRef targetProfile, llvm::StringRef &outStage, unsigned &outMajor, unsigned &outMinor);

class VersionSupportInfo {
private:
  bool m_CompilerIsDebugBuild;
public:
  bool m_InternalValidator;
  unsigned m_DxilMajor, m_DxilMinor;
  unsigned m_ValMajor, m_ValMinor;

  VersionSupportInfo();
  // Initialize version info structure.  TODO: add device shader model support
  void Initialize(dxc::DxcDllSupport &dllSupport);
  // Return true if IR sensitive test should be skipped, and log comment
  bool SkipIRSensitiveTest();
  // Return true if test requiring DXIL of given version should be skipped, and log comment
  bool SkipDxilVersion(unsigned major, unsigned minor);
  // Return true if out-of-memory test should be skipped, and log comment
  bool SkipOutOfMemoryTest();
};
