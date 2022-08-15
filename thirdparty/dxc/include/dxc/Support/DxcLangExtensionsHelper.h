///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcLangExtensionsHelper.h                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a helper class to implement language extensions to HLSL.         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef __DXCLANGEXTENSIONSHELPER_H__
#define __DXCLANGEXTENSIONSHELPER_H__

#include "dxc/Support/Unicode.h"
#include "dxc/Support/FileIOHelper.h"
#include "dxc/Support/DxcLangExtensionsCommonHelper.h"
#include <vector>

namespace llvm {
class raw_string_ostream;
class CallInst;
class Value;
}
namespace clang {
class CompilerInstance;
}

namespace hlsl {

class DxcLangExtensionsHelper : public DxcLangExtensionsCommonHelper, public DxcLangExtensionsHelperApply {
private:

public:
  void SetupSema(clang::Sema &S) override {
    clang::ExternalASTSource *astSource = S.getASTContext().getExternalSource();
    if (clang::ExternalSemaSource *externalSema =
            llvm::dyn_cast_or_null<clang::ExternalSemaSource>(astSource)) {
      for (auto &&table : GetIntrinsicTables()) {
        hlsl::RegisterIntrinsicTable(externalSema, table);
      }
    }
  }

  void SetupPreprocessorOptions(clang::PreprocessorOptions &PPOpts) override {
    for (const auto &define : GetDefines()) {
      PPOpts.addMacroDef(llvm::StringRef(define.c_str()));
    }
  }

  DxcLangExtensionsHelper *GetDxcLangExtensionsHelper() override {
    return this;
  }
 
  DxcLangExtensionsHelper() {}
};

// A parsed semantic define is a semantic define that has actually been
// parsed by the compiler. It has a name (required), a value (could be
// the empty string), and a location. We use an encoded clang::SourceLocation
// for the location to avoid a clang include dependency.
struct ParsedSemanticDefine{
  std::string Name;
  std::string Value;
  unsigned Location;
};
typedef std::vector<ParsedSemanticDefine> ParsedSemanticDefineList;

// Confirm that <name> matches the star pattern in <mask>
inline bool IsMacroMatch(StringRef name, const std::string &mask) {
  return Unicode::IsStarMatchUTF8(mask.c_str(), mask.size(), name.data(),
                                  name.size());
}

// Return the collection of semantic defines parsed by the compiler instance.
ParsedSemanticDefineList
  CollectSemanticDefinesParsedByCompiler(clang::CompilerInstance &compiler,
                                         _In_ DxcLangExtensionsHelper *helper);

} // namespace hlsl

#endif
