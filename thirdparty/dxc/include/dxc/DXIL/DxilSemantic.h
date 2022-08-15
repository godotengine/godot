///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSemantic.h                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation of HLSL parameter semantics.                               //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "llvm/ADT/StringRef.h"

#include "DxilConstants.h"
#include "DxilShaderModel.h"

namespace hlsl {

/// Use this class to represent HLSL parameter semantic.
class Semantic {
public:
  using Kind = DXIL::SemanticKind;

  static const int kUndefinedRow = -1;
  static const int kUndefinedCol = -1;

  static const Semantic *GetByName(llvm::StringRef name);
  static const Semantic *GetByName(llvm::StringRef Name, DXIL::SigPointKind sigPointKind,
    unsigned MajorVersion = ShaderModel::kHighestMajor, unsigned MinorVersion = ShaderModel::kHighestMinor);
  static const Semantic *Get(Kind kind);
  static const Semantic *Get(Kind kind, DXIL::SigPointKind sigPointKind,
    unsigned MajorVersion = ShaderModel::kHighestMajor, unsigned MinorVersion = ShaderModel::kHighestMinor);
  static const Semantic *GetInvalid();
  static const Semantic *GetArbitrary();
  static bool HasSVPrefix(llvm::StringRef Name);
  static void DecomposeNameAndIndex(llvm::StringRef FullName, llvm::StringRef *pName, unsigned *pIndex);

  Kind GetKind() const;
  const char *GetName() const;
  bool IsArbitrary() const;
  bool IsInvalid() const;

private:
  Kind m_Kind;                   // Semantic kind.
  const char *m_pszName;              // Canonical name (for system semantics).

  Semantic() = delete;
  Semantic(Kind Kind, const char *pszName);

  // Table of all semantic properties.
  static const unsigned kNumSemanticRecords = (unsigned)Kind::Invalid + 1;
  static const Semantic ms_SemanticTable[kNumSemanticRecords];

  friend class ShaderModel;
  friend class SignatureElement;
};

} // namespace hlsl
