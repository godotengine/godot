///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaSymbolsManager.h                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "dxc/Support/WinIncludes.h"

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/PointerUnion.h"

namespace llvm {
class DIDerivedType;
class DILocalVariable;
class DIScope;
class DITemplateTypeParameter;
class DIType;
class Instruction;
}  // namespace llvm

namespace dxil_dia {
class Session;
class Symbol;
class SymbolChildrenEnumerator;

class SymbolManager {
public:
  struct LiveRange {
    unsigned Start;
    unsigned Length;
  };

  class SymbolFactory {
  protected:
      SymbolFactory(DWORD ID, DWORD ParentID);

      DWORD m_ID;
      DWORD m_ParentID;

  public:
      virtual ~SymbolFactory();
      virtual HRESULT Create(Session *pSession, Symbol **) = 0;
  };

  using ScopeToIDMap = llvm::DenseMap<llvm::DIScope *, DWORD>;
  using IDToLiveRangeMap = std::unordered_map<DWORD, LiveRange>;
  using ParentToChildrenMap = std::unordered_multimap<DWORD, DWORD>;


  SymbolManager();
  SymbolManager(SymbolManager &&) = default;
  SymbolManager &operator =(SymbolManager &&) = default;
  ~SymbolManager();

  void Init(Session *pSes);

  size_t NumSymbols() const { return m_symbolCtors.size(); }
  HRESULT GetSymbolByID(size_t id, Symbol **ppSym) const;
  HRESULT GetLiveRangeOf(Symbol *pSym, LiveRange *LR) const;
  HRESULT GetGlobalScope(Symbol **ppSym) const;
  HRESULT ChildrenOf(Symbol *pSym, std::vector<CComPtr<Symbol>> *pChildren) const;
  HRESULT DbgScopeOf(const llvm::Instruction *instr, SymbolChildrenEnumerator **ppRet) const;

private:
  HRESULT ChildrenOf(DWORD ID, std::vector<CComPtr<Symbol>> *pChildren) const;

  // Not a CComPtr, and not AddRef'd - m_pSession is the owner of this.
  Session *m_pSession = nullptr;

  // Vector of factories for all symbols in the DXIL module.
  std::vector<std::unique_ptr<SymbolFactory>> m_symbolCtors;

  // Mapping from scope to its ID.
  ScopeToIDMap m_scopeToID;

  // Mapping from symbol ID to live range. Globals are live [0, end),
  // locals, [first dbg.declare, end of scope)
  // TODO: the live range information assumes structured dxil - which should hold
  // for non-optimized code - so we need something more robust. For now, this is
  // good enough.
  IDToLiveRangeMap m_symbolToLiveRange;

  ParentToChildrenMap m_parentToChildren;
};
}  // namespace dxil_dia
