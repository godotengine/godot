///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaSymbolsManager.cpp                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
#include "DxilDiaSymbolManager.h"

#include <cctype>
#include <functional>
#include <type_traits>

#include <comdef.h>

#include "dxc/DxilPIXPasses/DxilPIXVirtualRegisters.h"
#include "dxc/Support/Unicode.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "DxilDiaSession.h"
#include "DxilDiaTableSymbols.h"

static constexpr std::uint32_t kNullSymbolID = 0;

namespace dxil_dia {
namespace hlsl_symbols {

// HLSL Symbol Hierarchy
// ---- ------ ---------
// 
//                                  +---------------+
//                                  | Program (EXE) |                                   Global Scope
//                                  +------+--------+
//                                         |
//                                +--------^-----------+
//                                | Compiland (Shader) |                                Compilation Unit
//                                +--------+-----------+
//                                         |
//      +------------+------------+--------+-------+------------+--------------+
//      |            |            |        |       |            |              |        
// +----^----+   +---^---+   +----^---+    |   +---^---+   +----^----+   +-----^-----+
// | Details |   | Flags |   | Target |    |   | Entry |   | Defines |   | Arguments |  Synthetic Symbols
// +---------+   +-------+   +--------+    |   +-------+   +---------+   +-----------+
//                                         |
//                                         |
//       +---------------+------------+----+-----+-------------+-----------+
//       |               |            |          |             |           | 
// +-----^-----+   +-----^-----+   +--^--+   +---^--+      +---^--+     +--^--+
// | Function0 |   | Function1 |   | ... |   | UDT0 |      | UDT1 |     | ... |         Source Symbols
// +-----+-----+   +-----+-----+   +-----+   +---+--+      +---+--+     +-----+
//       |               |                       |             |
//  +----^----+     +----^----+             +----^----+   +----^----+
//  | Locals0 |     | Locals1 |             | Fields0 |   | Fields1 |
//  +---------+     +---------+             +---------+   +---------+

static const std::string & DxilEntryName(Session *pSession);

template <typename S, typename... C, typename = typename std::enable_if<!std::is_same<Symbol, S>::value>::type>
HRESULT AllocAndInit(
  IMalloc *pMalloc,
  Session *pSession,
  DWORD dwIndex,
  DWORD dwSymTag,
  S **ppSymbol,
  C... ctorArgs) {
  *ppSymbol = S::Alloc(pMalloc, ctorArgs...);
  if (*ppSymbol == nullptr) {
    return E_OUTOFMEMORY;
  }
  (*ppSymbol)->AddRef();
  (*ppSymbol)->Init(pSession, dwIndex, dwSymTag);
  return S_OK;
}

template <typename T, typename R>
T *dyn_cast_to_ditype(R ref) {
  return llvm::dyn_cast<T>((llvm::Metadata *) ref);
}

template <typename T, typename R>
T *dyn_cast_to_ditype_or_null(R ref) {
  return llvm::dyn_cast_or_null<T>((llvm::Metadata *) ref);
}

template <typename N>
struct DISymbol : public Symbol {
  DISymbol(IMalloc *M, N Node) : Symbol(M), m_pNode(Node) {}

  N m_pNode;
};

template <typename N>
struct TypedSymbol : public DISymbol<N> {
  TypedSymbol(IMalloc *M, N Node, DWORD dwTypeID, llvm::DIType *Type) : DISymbol(M, Node), m_dwTypeID(dwTypeID), m_pType(Type) {}

  STDMETHODIMP get_type(
    /* [retval][out] */ IDiaSymbol **ppRetVal) override {
    if (ppRetVal == nullptr) {
      return E_INVALIDARG;
    }
    *ppRetVal = false;

    if (m_pType == nullptr) {
      return S_FALSE;
    }

    Symbol *ret;
    IFR(m_pSession->SymMgr().GetSymbolByID(m_dwTypeID, &ret));

    *ppRetVal = ret;
    return S_OK;
  }

  const DWORD m_dwTypeID;
  llvm::DIType *m_pType;
};

struct GlobalScopeSymbol : public Symbol {
  DXC_MICROCOM_TM_ALLOC(GlobalScopeSymbol)
  explicit GlobalScopeSymbol(IMalloc *M) : Symbol(M) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, Symbol **ppSym);
  HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) override;
};

namespace symbol_factory {
class GlobalScope final : public SymbolManager::SymbolFactory {
public:
    GlobalScope(DWORD ID, DWORD ParentID)
        : SymbolManager::SymbolFactory(ID, ParentID) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        return hlsl_symbols::GlobalScopeSymbol::Create(pMalloc, pSession, ppRet);
    }
};
}  // namespace symbol_factory

struct CompilandSymbol : public DISymbol<llvm::DICompileUnit *> {
  DXC_MICROCOM_TM_ALLOC(CompilandSymbol)
  explicit CompilandSymbol(IMalloc *M, llvm::DICompileUnit *CU) : DISymbol<llvm::DICompileUnit *>(M, CU) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, llvm::DICompileUnit *CU, Symbol **ppSym);
  HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) override;
};

namespace symbol_factory {
class Compiland final : public SymbolManager::SymbolFactory {
public:
    Compiland(DWORD ID, DWORD ParentID, llvm::DICompileUnit *CU)
        : SymbolManager::SymbolFactory(ID, ParentID), m_CU(CU) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(hlsl_symbols::CompilandSymbol::Create(pMalloc, pSession, m_CU, ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        return S_OK;
    }

private:
    llvm::DICompileUnit *m_CU;
};
}  // namespace symbol_factory

struct CompilandDetailsSymbol : public Symbol {
  DXC_MICROCOM_TM_ALLOC(CompilandDetailsSymbol)
  explicit CompilandDetailsSymbol(IMalloc *M) : Symbol(M) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, Symbol **ppSym);
  HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) override;

#pragma region IDiaSymbol implementation
// DEFINE_SIMPLE_GETTER is used to generate the boilerplate needed for the
// property getters needed by this symbol. name is the property name (as
// defined in IDiaSymbol::get_<name>. There should be a static (non-const OK)
// function defined in this class as
//
//    <RetTy> <name>(CompilandDetailsSymbol * <this>)
//
// <RetTy> **must** match the property type in IDiaSymbol::get_<name>'s
// parameter; <this> is literally the this pointer. The function needs to
// be static (thus requiring the explicit <this> parameter) so that
// DEFINE_SIMPLE_GETTER can use decltype(name(nullptr)) in order to
// define the property parameter type.
#define DEFINE_SIMPLE_GETTER(name)                                        \
    STDMETHODIMP get_ ## name(decltype(name(nullptr)) *pValue) override { \
      if (pValue == nullptr) {                                            \
        return E_INVALIDARG;                                              \
      }                                                                   \
      *pValue = name(this);                                               \
      return S_OK;                                                        \
    }

  static constexpr DWORD platform(CompilandDetailsSymbol *) { return 256; }
  static constexpr DWORD language(CompilandDetailsSymbol *) { return 16; }
  static constexpr BOOL hasDebugInfo(CompilandDetailsSymbol *) { return true; }
  static BSTR compilerName(CompilandDetailsSymbol *) {
    CComBSTR retval;
    retval.Append("dxcompiler");
    return retval.Detach();
  }
  static DWORD frontEndMajor(CompilandDetailsSymbol *self) {
    return self->m_pSession->DxilModuleRef().GetShaderModel()->GetMajor();
  }
  static DWORD frontEndMinor(CompilandDetailsSymbol *self) {
    return self->m_pSession->DxilModuleRef().GetShaderModel()->GetMinor();
  }

  DEFINE_SIMPLE_GETTER(platform);
  DEFINE_SIMPLE_GETTER(language);
  DEFINE_SIMPLE_GETTER(frontEndMajor);
  DEFINE_SIMPLE_GETTER(frontEndMinor);
  DEFINE_SIMPLE_GETTER(hasDebugInfo);
  DEFINE_SIMPLE_GETTER(compilerName);
#undef DEFINE_SIMPLE_GETTER
#pragma endregion
};

namespace symbol_factory {
class CompilandDetails final : public SymbolManager::SymbolFactory {
public:
    CompilandDetails(DWORD ID, DWORD ParentID)
        : SymbolManager::SymbolFactory(ID, ParentID) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(hlsl_symbols::CompilandDetailsSymbol::Create(pMalloc, pSession, ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        return S_OK;
    }
};
}  // namespace symbol_factory

struct CompilandEnvSymbol : public Symbol {
  DXC_MICROCOM_TM_ALLOC(CompilandEnvSymbol)
  explicit CompilandEnvSymbol(IMalloc *M) : Symbol(M) {}
  static HRESULT CreateFlags(IMalloc *pMalloc, Session *pSession, Symbol **ppSym);
  static HRESULT CreateTarget(IMalloc *pMalloc, Session *pSession, Symbol **ppSym);
  static HRESULT CreateEntry(IMalloc *pMalloc, Session *pSession, Symbol **ppSym);
  static HRESULT CreateDefines(IMalloc *pMalloc, Session *pSession, Symbol **pSym);
  static HRESULT CreateArguments(IMalloc *pMalloc, Session *pSession, Symbol **ppSym);
  HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) override;
};

namespace symbol_factory {
using CompilandEnvCreateFn = HRESULT(IMalloc *, Session *, Symbol **);
template<CompilandEnvCreateFn C>
class CompilandEnv final : public SymbolManager::SymbolFactory {
public:
    CompilandEnv(DWORD ID, DWORD ParentID)
        : SymbolManager::SymbolFactory(ID, ParentID) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(C(pMalloc, pSession, ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        return S_OK;
    }
};
}  // namespace symbol_factory

struct FunctionSymbol : public TypedSymbol<llvm::DISubprogram *> {
  DXC_MICROCOM_TM_ALLOC(FunctionSymbol)
  FunctionSymbol(IMalloc *M, llvm::DISubprogram *Node, DWORD dwTypeID, llvm::DIType *Type) : TypedSymbol<llvm::DISubprogram *>(M, Node, dwTypeID, Type) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, DWORD dwID, llvm::DISubprogram *Node, DWORD dwTypeID, llvm::DIType *Type, Symbol **ppSym);
  HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) override;
};

namespace symbol_factory {
class Function final : public SymbolManager::SymbolFactory {
public:
    Function(DWORD ID, DWORD ParentID, llvm::DISubprogram *Node, DWORD TypeID)
        : SymbolManager::SymbolFactory(ID, ParentID),
          m_Node(Node), m_TypeID(TypeID) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(FunctionSymbol::Create(pMalloc, pSession, m_ID, m_Node, m_TypeID, m_Node->getType(), ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        (*ppRet)->SetName(CA2W(m_Node->getName().str().c_str(), CP_UTF8));
        return S_OK;
    }

private:
    llvm::DISubprogram *m_Node;
    DWORD m_TypeID;
};
}  // namespace symbol_factory

struct FunctionBlockSymbol : public Symbol {
  DXC_MICROCOM_TM_ALLOC(FunctionBlockSymbol)
  explicit FunctionBlockSymbol(IMalloc *M) : Symbol(M) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, DWORD dwID, Symbol **ppSym);
  HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) override;
};

namespace symbol_factory {
class FunctionBlock final : public SymbolManager::SymbolFactory {
public:
    FunctionBlock(DWORD ID, DWORD ParentID)
        : SymbolManager::SymbolFactory(ID, ParentID) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(FunctionBlockSymbol::Create(pMalloc, pSession, m_ID, ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        return S_OK;
    }
};
}  // namespace symbol_factory

struct TypeSymbol : public DISymbol<llvm::DIType *> {
  using LazySymbolName = std::function<HRESULT(Session *, std::string *)>;
  DXC_MICROCOM_TM_ALLOC(TypeSymbol)
  TypeSymbol(IMalloc *M, llvm::DIType *Node, LazySymbolName LazySymbolName) : DISymbol<llvm::DIType *>(M, Node), m_lazySymbolName(LazySymbolName) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, DWORD dwParentID, DWORD dwID, DWORD st, llvm::DIType *Node, LazySymbolName LazySymbolName, Symbol **ppSym);
  STDMETHODIMP get_name(
    /* [retval][out] */ BSTR *pRetVal) override;
  STDMETHODIMP get_baseType(
    /* [retval][out] */ DWORD *pRetVal) override;
  STDMETHODIMP get_length(
    /* [retval][out] */ ULONGLONG *pRetVal) override;

  HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) override;

  LazySymbolName m_lazySymbolName;
};

namespace symbol_factory {
class Type final : public SymbolManager::SymbolFactory {
public:
    Type(DWORD ID, DWORD ParentID, DWORD st, llvm::DIType *Node, TypeSymbol::LazySymbolName LazySymbolName)
        : SymbolManager::SymbolFactory(ID, ParentID),
          m_st(st), m_Node(Node), m_LazySymbolName(LazySymbolName) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(TypeSymbol::Create(pMalloc, pSession, m_ParentID, m_ID, m_st, m_Node, m_LazySymbolName, ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        return S_OK;
    }

private:
    DWORD m_st;
    llvm::DIType *m_Node;
    TypeSymbol::LazySymbolName m_LazySymbolName;
};
}  // namespace symbol_factory

struct TypedefTypeSymbol : public TypeSymbol {
  DXC_MICROCOM_TM_ALLOC(TypedefTypeSymbol)
  TypedefTypeSymbol(IMalloc *M, llvm::DIType *Node, DWORD dwBaseTypeID) : TypeSymbol(M, Node, nullptr), m_dwBaseTypeID(dwBaseTypeID) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, DWORD dwParentID, DWORD dwID, llvm::DIType *Node, DWORD dwBaseTypeID, Symbol **ppSym);

  STDMETHODIMP get_type(
    /* [retval][out] */ IDiaSymbol **ppRetVal) override;

  const DWORD m_dwBaseTypeID;
};

namespace symbol_factory {
class TypedefType final : public SymbolManager::SymbolFactory {
public:
    TypedefType(DWORD ID, DWORD ParentID, llvm::DIType *Node, DWORD BaseTypeID)
        : SymbolManager::SymbolFactory(ID, ParentID),
          m_Node(Node), m_BaseTypeID(BaseTypeID) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(TypedefTypeSymbol::Create(pMalloc, pSession, m_ParentID, m_ID, m_Node, m_BaseTypeID, ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        (*ppRet)->SetName(CA2W(m_Node->getName().str().c_str(), CP_UTF8));
        return S_OK;
    }

private:
    llvm::DIType *m_Node;
    DWORD m_BaseTypeID;
};
}  // namespace symbol_factory

struct VectorTypeSymbol : public TypeSymbol {
  DXC_MICROCOM_TM_ALLOC(VectorTypeSymbol)
  VectorTypeSymbol(IMalloc *M, llvm::DIType *Node, DWORD dwElemTyID, std::uint32_t NumElts) : TypeSymbol(M, Node, nullptr), m_ElemTyID(dwElemTyID), m_NumElts(NumElts) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, DWORD dwParentID, DWORD dwID, llvm::DIType *Node, DWORD dwElemTyID, std::uint32_t NumElts, Symbol **ppSym);

  STDMETHODIMP get_count(
    /* [retval][out] */ DWORD *pRetVal) override;
  STDMETHODIMP get_type(
    /* [retval][out] */ IDiaSymbol **ppRetVal) override;

  std::uint32_t m_ElemTyID;
  std::uint32_t m_NumElts;
};

namespace symbol_factory {
class VectorType final : public SymbolManager::SymbolFactory {
public:
    VectorType(DWORD ID, DWORD ParentID, llvm::DIType *Node, DWORD ElemTyID, std::uint32_t NumElts)
        : SymbolManager::SymbolFactory(ID, ParentID),
          m_Node(Node), m_ElemTyID(ElemTyID), m_NumElts(NumElts) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(VectorTypeSymbol::Create(pMalloc, pSession, m_ParentID, m_ID, m_Node, m_ElemTyID, m_NumElts, ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        (*ppRet)->SetName(CA2W(m_Node->getName().str().c_str(), CP_UTF8));
        return S_OK;
    }

private:
    llvm::DIType *m_Node;
    DWORD m_ElemTyID;
    std::uint32_t m_NumElts;
};
}  // namespace symbol_factory

struct UDTSymbol : public TypeSymbol {
  DXC_MICROCOM_TM_ALLOC(UDTSymbol)
  UDTSymbol(IMalloc *M, llvm::DICompositeType *Node, LazySymbolName LazyName) : TypeSymbol(M, Node, LazyName) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, DWORD dwParentID, DWORD dwID, llvm::DICompositeType *Node, LazySymbolName LazySymbolName, Symbol **ppSym);
};

namespace symbol_factory {
class UDT final : public SymbolManager::SymbolFactory {
public:
    UDT(DWORD ID, DWORD ParentID, llvm::DICompositeType *Node, TypeSymbol::LazySymbolName LazySymbolName)
        : SymbolManager::SymbolFactory(ID, ParentID),
          m_Node(Node), m_LazySymbolName(LazySymbolName) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(UDTSymbol::Create(pMalloc, pSession, m_ParentID, m_ID, m_Node, m_LazySymbolName, ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        return S_OK;
    }

private:
    llvm::DICompositeType *m_Node;
    TypeSymbol::LazySymbolName m_LazySymbolName;
};
}  // namespace symbol_factory

struct GlobalVariableSymbol : public TypedSymbol<llvm::DIGlobalVariable *> {
  DXC_MICROCOM_TM_ALLOC(GlobalVariableSymbol)
  GlobalVariableSymbol(IMalloc *M, llvm::DIGlobalVariable *GV, DWORD dwTypeID, llvm::DIType *Type) : TypedSymbol<llvm::DIGlobalVariable *>(M, GV, dwTypeID, Type) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, DWORD dwID, llvm::DIGlobalVariable *GV, DWORD dwTypeID, llvm::DIType *Type, Symbol **ppSym);
  HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) override;
};

namespace symbol_factory {
class GlobalVariable final : public SymbolManager::SymbolFactory {
public:
    GlobalVariable(DWORD ID, DWORD ParentID, llvm::DIGlobalVariable *GV, DWORD TypeID, llvm::DIType *Type)
        : SymbolManager::SymbolFactory(ID, ParentID),
          m_GV(GV), m_TypeID(TypeID), m_Type(Type) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(GlobalVariableSymbol::Create(pMalloc, pSession, m_ID, m_GV, m_TypeID, m_Type, ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        (*ppRet)->SetName(CA2W(m_GV->getName().str().c_str(), CP_UTF8));
        (*ppRet)->SetIsHLSLData(true);
        return S_OK;
    }

private:
    llvm::DIGlobalVariable *m_GV;
    DWORD m_TypeID;
    llvm::DIType *m_Type;
};
}  // namespace symbol_factory

struct LocalVariableSymbol : public TypedSymbol<llvm::DIVariable *> {
  DXC_MICROCOM_TM_ALLOC(LocalVariableSymbol)
  LocalVariableSymbol(IMalloc *M, llvm::DIVariable *Node, DWORD dwTypeID, llvm::DIType *Type, DWORD dwOffsetInUDT, DWORD dwDxilRegNum) : TypedSymbol<llvm::DIVariable *>(M, Node, dwTypeID, Type), m_dwOffsetInUDT(dwOffsetInUDT), m_dwDxilRegNum(dwDxilRegNum) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, DWORD dwID, llvm::DIVariable *Node, DWORD dwTypeID, llvm::DIType *Type, DWORD dwOffsetInUDT, DWORD m_dwDxilRegNum, Symbol **ppSym);
  STDMETHODIMP get_locationType(
    /* [retval][out] */ DWORD *pRetVal) override;
  STDMETHODIMP get_isAggregated(
    /* [retval][out] */ BOOL *pRetVal) override;
  STDMETHODIMP get_registerType(
    /* [retval][out] */ DWORD *pRetVal) override;
  STDMETHODIMP get_offsetInUdt(
    /* [retval][out] */ DWORD *pRetVal) override;
  STDMETHODIMP get_sizeInUdt(
    /* [retval][out] */ DWORD *pRetVal) override;
  STDMETHODIMP get_numberOfRegisterIndices(
    /* [retval][out] */ DWORD *pRetVal) override;
  STDMETHODIMP get_numericProperties(
    /* [in] */ DWORD cnt,
    /* [out] */ DWORD *pcnt,
    /* [size_is][out] */ DWORD *pProperties) override;

  HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) override;

  const DWORD m_dwOffsetInUDT;
  const DWORD m_dwDxilRegNum;
};

namespace symbol_factory {
class LocalVarInfo {
public:
  LocalVarInfo() = default;
  LocalVarInfo(const LocalVarInfo &) = delete;
  LocalVarInfo(LocalVarInfo &&) = default;

  DWORD GetVarID() const { return m_dwVarID; }
  DWORD GetOffsetInUDT() const { return m_dwOffsetInUDT; }
  DWORD GetDxilRegister() const { return m_dwDxilRegister; }

  void SetVarID(DWORD dwVarID) { m_dwVarID = dwVarID; }
  void SetOffsetInUDT(DWORD dwOffsetInUDT) { m_dwOffsetInUDT = dwOffsetInUDT; }
  void SetDxilRegister(DWORD dwDxilReg) { m_dwDxilRegister = dwDxilReg; }

private:
  DWORD m_dwVarID = 0;
  DWORD m_dwOffsetInUDT = 0;
  DWORD m_dwDxilRegister = 0;
};

class LocalVariable final : public SymbolManager::SymbolFactory {
public:
    LocalVariable(DWORD ID, DWORD ParentID, llvm::DIVariable *Node, DWORD TypeID, llvm::DIType *Type, std::shared_ptr<LocalVarInfo> VI)
        : SymbolManager::SymbolFactory(ID, ParentID),
          m_Node(Node), m_TypeID(TypeID), m_Type(Type), m_VI(VI) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(LocalVariableSymbol::Create(pMalloc, pSession, m_ID, m_Node, m_TypeID, m_Type, m_VI->GetOffsetInUDT(), m_VI->GetDxilRegister(), ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        (*ppRet)->SetName(CA2W(m_Node->getName().str().c_str(), CP_UTF8));
        (*ppRet)->SetDataKind(m_Node->getTag() == llvm::dwarf::DW_TAG_arg_variable ? DataIsParam : DataIsLocal);
        return S_OK;
    }

private:
    llvm::DIVariable *m_Node;
    DWORD m_TypeID;
    llvm::DIType *m_Type;
    std::shared_ptr<LocalVarInfo> m_VI;
};
}  // namespace symbol_factory

struct UDTFieldSymbol : public TypedSymbol<llvm::DIDerivedType *> {
  DXC_MICROCOM_TM_ALLOC(UDTFieldSymbol)
  UDTFieldSymbol(IMalloc *M, llvm::DIDerivedType *Node, DWORD dwTypeID, llvm::DIType *Type) : TypedSymbol<llvm::DIDerivedType *>(M, Node, dwTypeID, Type) {}
  static HRESULT Create(IMalloc *pMalloc, Session *pSession, DWORD dwID, llvm::DIDerivedType *Node, DWORD dwTypeID, llvm::DIType *Type, Symbol **ppSym);
  STDMETHODIMP get_offset(
    /* [retval][out] */ LONG *pRetVal) override;

  HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) override;
};

namespace symbol_factory {
class UDTField final : public SymbolManager::SymbolFactory {
public:
    UDTField(DWORD ID, DWORD ParentID, llvm::DIDerivedType *Node, DWORD TypeID, llvm::DIType *Type)
        : SymbolManager::SymbolFactory(ID, ParentID),
          m_Node(Node), m_TypeID(TypeID), m_Type(Type) {}

    virtual HRESULT Create(Session *pSession, Symbol **ppRet) override {
        IMalloc *pMalloc = pSession->GetMallocNoRef();
        IFR(UDTFieldSymbol::Create(pMalloc, pSession, m_ID, m_Node, m_TypeID, m_Type, ppRet));
        (*ppRet)->SetLexicalParent(m_ParentID);
        (*ppRet)->SetName(CA2W(m_Node->getName().str().c_str(), CP_UTF8));
        (*ppRet)->SetDataKind(m_Node->isStaticMember() ? DataIsStaticLocal : DataIsMember);
        return S_OK;
    }

private:
    llvm::DIDerivedType *m_Node;
    DWORD m_TypeID;
    llvm::DIType *m_Type;
};
}  // namespace symbol_factory

class SymbolManagerInit {
public:
  using SymbolCtor = std::function<HRESULT(Session *pSession, DWORD ID, Symbol **ppSym)>;

  using LazySymbolName = TypeSymbol::LazySymbolName;

  class TypeInfo {
  public:
    TypeInfo() = delete;
    TypeInfo(const TypeInfo &) = delete;
    TypeInfo(TypeInfo &&) = default;

    explicit TypeInfo(DWORD dwTypeID) : m_dwTypeID(dwTypeID) {}

    DWORD GetTypeID() const { return m_dwTypeID; }
    DWORD GetCurrentSizeInBytes() const { return m_dwCurrentSizeInBytes; }
    const std::vector<llvm::DIType *> &GetLayout() const { return m_Layout; }

    void Embed(const TypeInfo &TI);

    void AddBasicType(llvm::DIBasicType *BT);

  private:
    DWORD m_dwTypeID;
    std::vector<llvm::DIType *> m_Layout;
    DWORD m_dwCurrentSizeInBytes = 0;
  };
  using TypeToInfoMap = llvm::DenseMap<llvm::DIType *, std::unique_ptr<TypeInfo> >;

  // Because of the way the VarToID map is constructed, the
  // vector<LocalVarInfo> may need to grow. The Symbol Constructor for local
  // variable captures the LocalVarInfo for the local variable it creates, and
  // it needs access to the information on this map (thus a by-value capture is
  // not enough). We heap-allocate the VarInfos, and the local variables symbol
  // constructors capture the pointer - meaning everything should be fine
  // even if the vector is moved around.
  using LocalVarToIDMap = llvm::DenseMap<
      llvm::DILocalVariable *,
      std::vector<std::shared_ptr<symbol_factory::LocalVarInfo>>>;

  using UDTFieldToIDMap = llvm::DenseMap<llvm::DIDerivedType *, DWORD>;

  SymbolManagerInit(
    Session *pSession,
    std::vector<std::unique_ptr<SymbolManager::SymbolFactory>> *pSymCtors,
    SymbolManager::ScopeToIDMap *pScopeToSym,
    SymbolManager::IDToLiveRangeMap *pSymToLR);

  template <typename Factory, typename... Args>
  HRESULT AddSymbol(DWORD dwParentID, DWORD *pNewSymID, Args&&... args) {
      if (dwParentID > m_SymCtors.size()) {
          return E_FAIL;
      }

      const DWORD dwNewSymID = m_SymCtors.size() + 1;
      m_SymCtors.emplace_back(std::unique_ptr<Factory>(new Factory(dwNewSymID, dwParentID, std::forward<Args>(args)...)));
      *pNewSymID = dwNewSymID;
      IFR(AddParent(dwParentID));
      return S_OK;
  }

  HRESULT CreateFunctionsForAllCUs();
  HRESULT CreateGlobalVariablesForAllCUs();
  HRESULT CreateLocalVariables();
  HRESULT CreateLiveRanges();
  HRESULT IsDbgDeclareCall(llvm::Module *M, const llvm::Instruction *I,
                           DWORD *pReg, DWORD *pRegSize,
                           llvm::DILocalVariable **LV, uint64_t *pStartOffset,
                           uint64_t *pEndOffset,
                           dxil_dia::Session::RVA *pLowestUserRVA,
                           dxil_dia::Session::RVA *pHighestUserRVA);
  HRESULT GetDxilAllocaRegister(llvm::Instruction *I, DWORD *pRegNum, DWORD *pRegSize);
  HRESULT PopulateParentToChildrenIDMap(SymbolManager::ParentToChildrenMap *pParentToChildren);

private:
  HRESULT GetTypeInfo(llvm::DIType *T, TypeInfo **TI);

  template<typename Factory, typename... Args>
  HRESULT AddType(DWORD dwParentID, llvm::DIType *T, DWORD *pNewSymID, Args&&... args) {
      IFR(AddSymbol<Factory>(dwParentID, pNewSymID, std::forward<Args>(args)...));
      if (!m_TypeToInfo.insert(std::make_pair(T, llvm::make_unique<TypeInfo>(*pNewSymID))).second) {
          return E_FAIL;
      }
      return S_OK;
  }

  HRESULT AddParent(DWORD dwParentIndex);
  HRESULT CreateFunctionBlockForLocalScope(llvm::DILocalScope *LS, DWORD *pNewSymID);
  HRESULT CreateFunctionBlockForInstruction(llvm::Instruction *I);
  HRESULT CreateFunctionBlocksForFunction(llvm::Function *F);
  HRESULT CreateFunctionsForCU(llvm::DICompileUnit *CU);
  HRESULT CreateGlobalVariablesForCU(llvm::DICompileUnit *CU);
  HRESULT GetScopeID(llvm::DIScope *S, DWORD *pScopeID);
  HRESULT CreateType(llvm::DIType *T, DWORD *pNewTypeID);
  HRESULT CreateSubroutineType(DWORD dwParentID, llvm::DISubroutineType *ST, DWORD *pNewTypeID);
  HRESULT CreateBasicType(DWORD dwParentID, llvm::DIBasicType *VT, DWORD *pNewTypeID);
  HRESULT CreateCompositeType(DWORD dwParentID, llvm::DICompositeType *CT, DWORD *pNewTypeID);
  HRESULT CreateHLSLType(llvm::DICompositeType *T, DWORD *pNewTypeID);
  HRESULT IsHLSLVectorType(llvm::DICompositeType *T, DWORD *pEltTyID, std::uint32_t *pElemCnt);
  HRESULT CreateHLSLVectorType(llvm::DICompositeType *T, DWORD pEltTyID, std::uint32_t pElemCnt, DWORD *pNewTypeID);
  HRESULT HandleDerivedType(DWORD dwParentID, llvm::DIDerivedType *DT, DWORD *pNewTypeID);
  HRESULT CreateLocalVariable(DWORD dwParentID, llvm::DILocalVariable *LV);
  HRESULT GetTypeLayout(llvm::DIType *Ty, std::vector<DWORD> *pRet);
  HRESULT CreateUDTField(DWORD dwParentID, llvm::DIDerivedType *Field);

  Session &m_Session;
  std::vector<std::unique_ptr<SymbolManager::SymbolFactory>> &m_SymCtors;
  SymbolManager::ScopeToIDMap &m_ScopeToSym;
  SymbolManager::IDToLiveRangeMap &m_SymToLR;

  // vector of parents, i.e., for each i in parents[i], parents[i] is the
  // parent of m_symbol[i].
  std::vector<std::uint32_t> m_Parent;

  LocalVarToIDMap m_VarToID;

  UDTFieldToIDMap m_FieldToID;

  TypeToInfoMap m_TypeToInfo;

  TypeInfo &CurrentUDTInfo() { return *m_pCurUDT; }
  TypeInfo *m_pCurUDT = nullptr;

  struct UDTScope {
    UDTScope() = delete;
    UDTScope(const UDTScope &) = delete;
    UDTScope(UDTScope &&) = default;

    UDTScope(TypeInfo **pCur, TypeInfo *pNext) : m_pCur(pCur), m_pPrev(*pCur) {
      *pCur = pNext;
    }
    ~UDTScope() { *m_pCur = m_pPrev; }

    TypeInfo **m_pCur;
    TypeInfo *m_pPrev;
  };

  UDTScope BeginUDTScope(TypeInfo *pNext) {
    return UDTScope(&m_pCurUDT, pNext);
  }

};

}  // namespace hlsl_symbols
}  // namespace dxil_dia

STDMETHODIMP dxil_dia::hlsl_symbols::TypeSymbol::get_baseType(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = btNoType;

  if (auto *BT = llvm::dyn_cast<llvm::DIBasicType>(m_pNode)) {
    switch (BT->getEncoding()) {
    case llvm::dwarf::DW_ATE_boolean:
      *pRetVal = btBool; break;
    case llvm::dwarf::DW_ATE_unsigned:
      *pRetVal = btUInt; break;
    case llvm::dwarf::DW_ATE_signed:
      *pRetVal = btInt; break;
    case llvm::dwarf::DW_ATE_float:
      *pRetVal = btFloat; break;
    }
  }

  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::TypeSymbol::get_length(
  /* [retval][out] */ ULONGLONG *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = 0;

  if (auto *BT = llvm::dyn_cast<llvm::DIBasicType>(m_pNode)) {
    static constexpr DWORD kNumBitsPerByte = 8;
    const DWORD SizeInBits = BT->getSizeInBits();
    *pRetVal = SizeInBits / kNumBitsPerByte;
  }

  return S_OK;
}

static const std::string &dxil_dia::hlsl_symbols::DxilEntryName(Session *pSession) {
  return pSession->DxilModuleRef().GetEntryFunctionName();
}

HRESULT dxil_dia::hlsl_symbols::GlobalScopeSymbol::Create(IMalloc *pMalloc, Session *pSession, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, HlslProgramId, SymTagExe, (GlobalScopeSymbol**)ppSym));
  (*ppSym)->SetName(L"HLSL");

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::GlobalScopeSymbol::GetChildren(std::vector<CComPtr<Symbol>> *children) {
  return m_pSession->SymMgr().ChildrenOf(this, children);
}

HRESULT dxil_dia::hlsl_symbols::CompilandSymbol::Create(IMalloc *pMalloc, Session *pSession, llvm::DICompileUnit *CU, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, HlslCompilandId, SymTagCompiland, (CompilandSymbol**)ppSym, CU));
  (*ppSym)->SetName(L"main");
  if (pSession->MainFileName()) {
    llvm::StringRef strRef = llvm::dyn_cast<llvm::MDString>(pSession->MainFileName()->getOperand(0)->getOperand(0))->getString();
    std::string str(strRef.begin(), strRef.size()); // To make sure str is null terminated
    (*ppSym)->SetSourceFileName(_bstr_t(Unicode::UTF8ToWideStringOrThrow(str.data()).c_str()));
  }

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::CompilandSymbol::GetChildren(std::vector<CComPtr<Symbol>> *children) {
  return m_pSession->SymMgr().ChildrenOf(this, children);
}

HRESULT dxil_dia::hlsl_symbols::CompilandDetailsSymbol::Create(IMalloc *pMalloc, Session *pSession, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, HlslCompilandDetailsId, SymTagCompilandDetails, (CompilandDetailsSymbol**)ppSym));
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::CompilandDetailsSymbol::GetChildren(std::vector<CComPtr<Symbol>> *children) {
  children->clear();
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::CompilandEnvSymbol::CreateFlags(IMalloc *pMalloc, Session *pSession, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, HlslCompilandEnvFlagsId, SymTagCompilandEnv, (CompilandEnvSymbol**)ppSym));
  (*ppSym)->SetName(L"hlslFlags");

  const char *specialCases[] = { "/T", "-T", "-D", "/D", "-E", "/E", };

  llvm::MDNode *argsNode = pSession->Arguments()->getOperand(0);
  // Construct a double null terminated string for defines with L"\0" as a delimiter
  CComBSTR pBSTR;
  for (llvm::MDNode::op_iterator it = argsNode->op_begin(); it != argsNode->op_end(); ++it) {
    llvm::StringRef strRef = llvm::dyn_cast<llvm::MDString>(*it)->getString();

    bool skip = false;
    bool skipTwice = false;
    for (unsigned i = 0; i < _countof(specialCases); i++) {
      if (strRef == specialCases[i]) {
        skipTwice = true;
        skip = true;
        break;
      }
      else if (strRef.startswith(specialCases[i])) {
        skip = true;
        break;
      }
    }

    if (skip) {
      if (skipTwice)
        ++it;
      continue;
    }

    std::string str(strRef.begin(), strRef.size());
    CA2W cv(str.c_str(), CP_UTF8);
    pBSTR.Append(cv);
    pBSTR.Append(L"\0", 1);
  }
  pBSTR.Append(L"\0", 1);
  VARIANT Variant;
  Variant.bstrVal = pBSTR;
  Variant.vt = VARENUM::VT_BSTR;
  (*ppSym)->SetValue(&Variant);
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::CompilandEnvSymbol::CreateTarget(IMalloc *pMalloc, Session *pSession, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, HlslCompilandEnvTargetId, SymTagCompilandEnv, (CompilandEnvSymbol**)ppSym));
  (*ppSym)->SetName(L"hlslTarget");
  (*ppSym)->SetValue(pSession->DxilModuleRef().GetShaderModel()->GetName());
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::CompilandEnvSymbol::CreateEntry(IMalloc *pMalloc, Session *pSession, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, HlslCompilandEnvEntryId, SymTagCompilandEnv, (CompilandEnvSymbol**)ppSym));
  (*ppSym)->SetName(L"hlslEntry");
  (*ppSym)->SetValue(DxilEntryName(pSession).c_str());
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::CompilandEnvSymbol::CreateDefines(IMalloc *pMalloc, Session *pSession, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, HlslCompilandEnvDefinesId, SymTagCompilandEnv, (CompilandEnvSymbol**)ppSym));
  (*ppSym)->SetName(L"hlslDefines");
  llvm::MDNode *definesNode = pSession->Defines()->getOperand(0);
  // Construct a double null terminated string for defines with L"\0" as a delimiter
  CComBSTR pBSTR;
  for (llvm::MDNode::op_iterator it = definesNode->op_begin(); it != definesNode->op_end(); ++it) {
    llvm::StringRef strRef = llvm::dyn_cast<llvm::MDString>(*it)->getString();
    std::string str(strRef.begin(), strRef.size());
    CA2W cv(str.c_str(), CP_UTF8);
    pBSTR.Append(cv);
    pBSTR.Append(L"\0", 1);
  }
  pBSTR.Append(L"\0", 1);
  VARIANT Variant;
  Variant.bstrVal = pBSTR;
  Variant.vt = VARENUM::VT_BSTR;
  (*ppSym)->SetValue(&Variant);
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::CompilandEnvSymbol::CreateArguments(IMalloc *pMalloc, Session *pSession, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, HlslCompilandEnvArgumentsId, SymTagCompilandEnv, (CompilandEnvSymbol**)ppSym));
  (*ppSym)->SetName(L"hlslArguments");
  auto Arguments = pSession->Arguments()->getOperand(0);
  auto NumArguments = Arguments->getNumOperands();
  std::string args;
  for (unsigned i = 0; i < NumArguments; ++i) {
    llvm::StringRef strRef = llvm::dyn_cast<llvm::MDString>(Arguments->getOperand(i))->getString();
    if (!args.empty())
      args.push_back(' ');
    args = args + strRef.str();
  }
  (*ppSym)->SetValue(args.c_str());
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::CompilandEnvSymbol::GetChildren(std::vector<CComPtr<Symbol>> *children) {
  children->clear();
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::FunctionSymbol::Create(IMalloc *pMalloc, Session *pSession, DWORD dwID, llvm::DISubprogram *Node, DWORD dwTypeID, llvm::DIType *Type, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, dwID, SymTagFunction, (FunctionSymbol**)ppSym, Node, dwTypeID, Type));
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::FunctionSymbol::GetChildren(std::vector<CComPtr<Symbol>> *children) {
  return m_pSession->SymMgr().ChildrenOf(this, children);
}

HRESULT dxil_dia::hlsl_symbols::FunctionBlockSymbol::Create(IMalloc *pMalloc, Session *pSession, DWORD dwID, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, dwID, SymTagBlock, (FunctionBlockSymbol**)ppSym));
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::FunctionBlockSymbol::GetChildren(std::vector<CComPtr<Symbol>> *children) {
  return m_pSession->SymMgr().ChildrenOf(this, children);
}

HRESULT dxil_dia::hlsl_symbols::TypeSymbol::Create(IMalloc *pMalloc, Session *pSession, DWORD dwParentID, DWORD dwID, DWORD st, llvm::DIType *Node, LazySymbolName LazySymbolName, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, dwID, st, (TypeSymbol**)ppSym, Node, LazySymbolName));
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::TypeSymbol::get_name(
  /* [retval][out] */ BSTR *pRetVal) {
  DxcThreadMalloc TM(m_pSession->GetMallocNoRef());
  if (m_lazySymbolName != nullptr) {
    DXASSERT(!this->HasName(), "Setting type name multiple times.");
    std::string Name;
    IFR(m_lazySymbolName(m_pSession, &Name));
    this->SetName(CA2W(Name.c_str(), CP_UTF8));
    m_lazySymbolName = nullptr;
  }
  return Symbol::get_name(pRetVal);
}

HRESULT dxil_dia::hlsl_symbols::TypeSymbol::GetChildren(std::vector<CComPtr<Symbol>> *children) {
  return m_pSession->SymMgr().ChildrenOf(this, children);
}

HRESULT dxil_dia::hlsl_symbols::VectorTypeSymbol::Create(IMalloc *pMalloc, Session *pSession, DWORD dwParentID, DWORD dwID, llvm::DIType *Node, DWORD dwElemTyID, std::uint32_t NumElts, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, dwID, SymTagVectorType, (VectorTypeSymbol**)ppSym, Node, dwElemTyID, NumElts));
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::VectorTypeSymbol::get_count(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = m_NumElts;
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::VectorTypeSymbol::get_type(
  /* [retval][out] */ IDiaSymbol **ppRetVal) {
  if (ppRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *ppRetVal = false;

  Symbol *ret;
  IFR(m_pSession->SymMgr().GetSymbolByID(m_ElemTyID, &ret));

  *ppRetVal = ret;
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::UDTSymbol::Create(IMalloc *pMalloc, Session *pSession, DWORD dwParentID, DWORD dwID, llvm::DICompositeType *Node, LazySymbolName LazySymbolName, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, dwID, SymTagUDT, (UDTSymbol**)ppSym, Node, LazySymbolName));
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::TypedefTypeSymbol::Create(IMalloc *pMalloc, Session *pSession, DWORD dwParentID, DWORD dwID, llvm::DIType *Node, DWORD dwBaseTypeID, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, dwID, SymTagTypedef, (TypedefTypeSymbol**)ppSym, Node, dwBaseTypeID));
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::TypedefTypeSymbol::get_type(
  /* [retval][out] */ IDiaSymbol **ppRetVal) {
  if (ppRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *ppRetVal = nullptr;

  Symbol *ret = nullptr;
  IFR(m_pSession->SymMgr().GetSymbolByID(m_dwBaseTypeID, &ret));
  *ppRetVal = ret;

  return S_FALSE;
}

HRESULT dxil_dia::hlsl_symbols::GlobalVariableSymbol::Create(IMalloc *pMalloc, Session *pSession, DWORD dwID, llvm::DIGlobalVariable *GV, DWORD dwTypeID, llvm::DIType *Type, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, dwID, SymTagData, (GlobalVariableSymbol**)ppSym, GV, dwTypeID, Type));
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::GlobalVariableSymbol::GetChildren(std::vector<CComPtr<Symbol>> *children) {
  children->clear();
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::LocalVariableSymbol::Create(IMalloc *pMalloc, Session *pSession, DWORD dwID, llvm::DIVariable *Node, DWORD dwTypeID, llvm::DIType *Type, DWORD dwOffsetInUDT, DWORD dwDxilRegNum, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, dwID, SymTagData, (LocalVariableSymbol**)ppSym, Node, dwTypeID, Type, dwOffsetInUDT, dwDxilRegNum));
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::LocalVariableSymbol::get_locationType(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = LocIsEnregistered;
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::LocalVariableSymbol::get_isAggregated(
  /* [retval][out] */ BOOL *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = m_pType->getTag() == llvm::dwarf::DW_TAG_member;
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::LocalVariableSymbol::get_registerType(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  static constexpr DWORD kPixTraceVirtualRegister = 0xfe;
  *pRetVal = kPixTraceVirtualRegister;
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::LocalVariableSymbol::get_offsetInUdt(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = m_dwOffsetInUDT;
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::LocalVariableSymbol::get_sizeInUdt(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  static constexpr DWORD kBitsPerByte = 8;
  //auto *DT = llvm::cast<llvm::DIDerivedType>(m_pType);
  *pRetVal = 4; //DT->getSizeInBits() / kBitsPerByte;
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::LocalVariableSymbol::get_numberOfRegisterIndices(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = 1;
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::LocalVariableSymbol::get_numericProperties(
  /* [in] */ DWORD cnt,
  /* [out] */ DWORD *pcnt,
  /* [size_is][out] */ DWORD *pProperties) {
  if (pcnt == nullptr || pProperties == nullptr || cnt != 1) {
    return E_INVALIDARG;
  }

  pProperties[0] = m_dwDxilRegNum;
  *pcnt = 1;
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::LocalVariableSymbol::GetChildren(std::vector<CComPtr<Symbol>> *children) {
  children->clear();
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::UDTFieldSymbol::Create(IMalloc *pMalloc, Session *pSession, DWORD dwID, llvm::DIDerivedType *Node, DWORD dwTypeID, llvm::DIType *Type, Symbol **ppSym) {
  IFR(AllocAndInit(pMalloc, pSession, dwID, SymTagData, (UDTFieldSymbol**)ppSym, Node, dwTypeID, Type));
  return S_OK;
}

STDMETHODIMP dxil_dia::hlsl_symbols::UDTFieldSymbol::get_offset(
  /* [retval][out] */ LONG *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  static constexpr DWORD kBitsPerByte = 8;
  *pRetVal = m_pNode->getOffsetInBits() / kBitsPerByte;
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::UDTFieldSymbol::GetChildren(std::vector<CComPtr<Symbol>> *children) {
  children->clear();
  return S_OK;
}

dxil_dia::hlsl_symbols::SymbolManagerInit::SymbolManagerInit(
    Session *pSession,
    std::vector<std::unique_ptr<SymbolManager::SymbolFactory>> *pSymCtors,
    SymbolManager::ScopeToIDMap *pScopeToSym,
    SymbolManager::IDToLiveRangeMap *pSymToLR)
  : m_Session(*pSession),
    m_SymCtors(*pSymCtors),
    m_ScopeToSym(*pScopeToSym),
    m_SymToLR(*pSymToLR) {
  DXASSERT_ARGS(m_Parent.size() == m_SymCtors.size(),
                "parent and symbol array size mismatch: %d vs %d",
                m_Parent.size(),
                m_SymCtors.size());
}

void dxil_dia::hlsl_symbols::SymbolManagerInit::TypeInfo::Embed(const TypeInfo &TI) {
  for (const auto &E : TI.GetLayout()) {
    m_Layout.emplace_back(E);
  }
  m_dwCurrentSizeInBytes += TI.m_dwCurrentSizeInBytes;
}

void dxil_dia::hlsl_symbols::SymbolManagerInit::TypeInfo::AddBasicType(llvm::DIBasicType *BT) {
  m_Layout.emplace_back(BT);

  static constexpr DWORD kNumBitsPerByte = 8;
  m_dwCurrentSizeInBytes += BT->getSizeInBits() / kNumBitsPerByte;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::GetTypeInfo(llvm::DIType *T, TypeInfo **TI) {
  auto tyInfoIt = m_TypeToInfo.find(T);
  if (tyInfoIt == m_TypeToInfo.end()) {
    return E_FAIL;
  }

  *TI = tyInfoIt->second.get();
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::AddParent(DWORD dwParentIndex) {
  m_Parent.emplace_back(dwParentIndex);
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateFunctionBlockForLocalScope(llvm::DILocalScope *LS, DWORD *pNewSymID) {
  if (LS == nullptr) {
    return E_FAIL;
  }

  auto lsIT = m_ScopeToSym.find(LS);
  if (lsIT != m_ScopeToSym.end()) {
    *pNewSymID = lsIT->second;
    return S_OK;
  }

  llvm::DILocalScope *ParentLS = nullptr;
  if (auto *Location = llvm::dyn_cast<llvm::DILocation>(LS)) {
    ParentLS = Location->getInlinedAtScope();
    if (ParentLS == nullptr) {
      ParentLS = Location->getScope();
    }
  } else if (auto *Block = llvm::dyn_cast<llvm::DILexicalBlock>(LS)) {
    ParentLS = Block->getScope();
  } else if (auto *BlockFile = llvm::dyn_cast<llvm::DILexicalBlockFile>(LS)) {
    ParentLS = BlockFile->getScope();
  }

  if (ParentLS == nullptr) {
    return E_FAIL;
  }

  DWORD dwParentID;
  IFR(CreateFunctionBlockForLocalScope(ParentLS, &dwParentID));

  IFR(AddSymbol<symbol_factory::FunctionBlock>(dwParentID, pNewSymID));
  m_ScopeToSym.insert(std::make_pair(LS, *pNewSymID));

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateFunctionBlockForInstruction(llvm::Instruction *I) {
  const llvm::DebugLoc &DL = I->getDebugLoc();
  if (!DL) {
    return S_OK;
  }

  llvm::MDNode *LocalScope = DL.getInlinedAtScope();
  if (LocalScope == nullptr) {
    LocalScope = DL.getScope();
  }
  if (LocalScope == nullptr) {
    return S_OK;
  }
  auto *LS = llvm::dyn_cast<llvm::DILocalScope>(LocalScope);
  if (LS == nullptr) {
    return E_FAIL;
  }

  auto localScopeIt = m_ScopeToSym.find(LS);
  if (localScopeIt == m_ScopeToSym.end()) {
    DWORD dwUnusedNewSymID;
    IFR(CreateFunctionBlockForLocalScope(LS, &dwUnusedNewSymID));
  }

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateFunctionBlocksForFunction(llvm::Function *F) {
  for (llvm::BasicBlock &BB : *F) {
    for (llvm::Instruction &I : BB) {
      IFR(CreateFunctionBlockForInstruction(&I));
    }
  }

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateFunctionsForCU(llvm::DICompileUnit *CU) {
  bool FoundFunctions = false;
  for (llvm::DISubprogram *SubProgram : CU->getSubprograms()) {
    DWORD dwNewFunID;
    const DWORD dwParentID = SubProgram->isLocalToUnit() ? HlslCompilandId : HlslProgramId;
    DWORD dwSubprogramTypeID;
    IFR(CreateType(SubProgram->getType(), &dwSubprogramTypeID));
    IFR(AddSymbol<symbol_factory::Function>(dwParentID, &dwNewFunID, SubProgram, dwSubprogramTypeID));
    m_ScopeToSym.insert(std::make_pair(SubProgram, dwNewFunID));
  }

  for (llvm::DISubprogram* SubProgram : CU->getSubprograms()) {
    if (llvm::Function *F = SubProgram->getFunction()) {
      IFR(CreateFunctionBlocksForFunction(F));
      FoundFunctions = true;
    }
  }

  if (!FoundFunctions) {
    // This works around an old bug in dxcompiler whose effects are still
    // sometimes present in PIX users' traces. (The bug was that the subprogram(s)
    // weren't pointing to their contained function.)
    llvm::Module *M = &m_Session.ModuleRef();
    auto &DM = M->GetDxilModule();
    llvm::Function *EntryPoint = DM.GetEntryFunction();
    IFR(CreateFunctionBlocksForFunction(EntryPoint));
  }

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateFunctionsForAllCUs() {
  for (llvm::DICompileUnit *pCU : m_Session.InfoRef().compile_units()) {
    IFR(CreateFunctionsForCU(pCU));
  }
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateGlobalVariablesForCU(llvm::DICompileUnit *CU) {
  for (llvm::DIGlobalVariable *GlobalVariable : CU->getGlobalVariables()) {
    DWORD dwUnusedNewGVID;
    const DWORD dwParentID = GlobalVariable->isLocalToUnit() ? HlslCompilandId : HlslProgramId;
    auto *GVType = dyn_cast_to_ditype<llvm::DIType>(GlobalVariable->getType());
    DWORD dwGVTypeID;
    IFR(CreateType(GVType, &dwGVTypeID));
    IFR(AddSymbol<symbol_factory::GlobalVariable>(dwParentID, &dwUnusedNewGVID, GlobalVariable, dwGVTypeID, GVType));
  }

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateGlobalVariablesForAllCUs() {
  for (llvm::DICompileUnit *pCU : m_Session.InfoRef().compile_units()) {
    IFR(CreateGlobalVariablesForCU(pCU));
  }
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::GetScopeID(llvm::DIScope *S, DWORD *pScopeID) {
  auto ParentScopeIt = m_ScopeToSym.find(S);
  if (ParentScopeIt != m_ScopeToSym.end()) {
    *pScopeID = ParentScopeIt->second;
  } else {
    auto *ParentScopeTy = llvm::dyn_cast<llvm::DIType>(S);
    if (!ParentScopeTy) {
      // Any non-existing scope must be a type.
      return E_FAIL;
    }
    IFR(CreateType(ParentScopeTy, pScopeID));
  }
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateType(llvm::DIType *Type, DWORD *pNewTypeID) {
  if (Type == nullptr) {
    return E_FAIL;
  }

  auto lsIT = m_TypeToInfo.find(Type);
  if (lsIT != m_TypeToInfo.end()) {
    *pNewTypeID = lsIT->second->GetTypeID();
    return S_OK;
  }

  if (auto *ST = llvm::dyn_cast<llvm::DISubroutineType>(Type)) {
    IFR(CreateSubroutineType(HlslProgramId, ST, pNewTypeID));
    return S_OK;
  } else if (auto *BT = llvm::dyn_cast<llvm::DIBasicType>(Type)) {
    IFR(CreateBasicType(HlslProgramId, BT, pNewTypeID));
    return S_OK;
  } else if (auto *CT = llvm::dyn_cast<llvm::DICompositeType>(Type)) {
    DWORD dwParentID = HlslProgramId;
    if (auto *ParentScope = dyn_cast_to_ditype_or_null<llvm::DIScope>(CT->getScope())) {
      IFR(GetScopeID(ParentScope, &dwParentID));
    }
    IFR(CreateCompositeType(dwParentID, CT, pNewTypeID));
    return S_OK;
  } else if (auto *DT = llvm::dyn_cast<llvm::DIDerivedType>(Type)) {
    DWORD dwParentID = HlslProgramId;
    if (auto *ParentScope = dyn_cast_to_ditype_or_null<llvm::DIScope>(DT->getScope())) {
      IFR(GetScopeID(ParentScope, &dwParentID));
    }
    IFR(HandleDerivedType(dwParentID, DT, pNewTypeID));
    return S_OK;
  }

  return E_FAIL;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateSubroutineType(DWORD dwParentID, llvm::DISubroutineType *ST, DWORD *pNewTypeID) {
  LazySymbolName LazyName;

  llvm::DITypeRefArray Types = ST->getTypeArray();
  if (Types.size() > 0) {
    std::vector<DWORD> TypeIDs;
    TypeIDs.reserve(Types.size());

    for (llvm::Metadata *M : Types) {
      auto *Ty = dyn_cast_to_ditype_or_null<llvm::DIType>(M);
      if (Ty == nullptr) {
        TypeIDs.emplace_back(kNullSymbolID);
      } else {
        DWORD dwTyID;
        IFR(CreateType(Ty, &dwTyID));
        TypeIDs.emplace_back(dwTyID);
      }
    }

    LazyName = [TypeIDs](Session *pSession, std::string *Name) -> HRESULT {
      Name->clear();
      llvm::raw_string_ostream OS(*Name);
      OS.SetUnbuffered();

      bool first = true;
      bool firstArg = true;
      auto &SM = pSession->SymMgr();
      for (DWORD ID : TypeIDs) {
        if (!first && !firstArg) {
          OS << ", ";
        }
        if (ID == kNullSymbolID) {
          OS << "void";
        } else {
          CComPtr<Symbol> SymTy;
          IFR(SM.GetSymbolByID(ID, &SymTy));
          CComBSTR name;
          IFR(SymTy->get_name(&name));
          if (!name) {
            OS << "???";
          } else {
            OS << CW2A((BSTR)name, CP_UTF8);
          }
        }
        if (first) {
          OS << "(";
        }
        firstArg = first;
        first = false;
      }
      OS << ")";
      return S_OK;
    };
  }

  IFR(AddType<symbol_factory::Type>(dwParentID, ST, pNewTypeID, SymTagFunctionType, ST, LazyName));

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateBasicType(DWORD dwParentID, llvm::DIBasicType *BT, DWORD *pNewTypeID) {
  DXASSERT_ARGS(dwParentID == HlslProgramId,
                "%d vs %d",
                dwParentID,
                HlslProgramId);

  LazySymbolName LazyName = [BT](Session *pSession, std::string *Name) -> HRESULT {
    *Name = BT->getName();
    return S_OK;
  };

  IFR(AddType<symbol_factory::Type>(dwParentID, BT, pNewTypeID, SymTagBaseType, BT, LazyName));

  TypeInfo *TI;
  IFR(GetTypeInfo(BT, &TI));
  TI->AddBasicType(BT);

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateCompositeType(DWORD dwParentID, llvm::DICompositeType *CT, DWORD *pNewTypeID) {
  switch (CT->getTag()) {
  case llvm::dwarf::DW_TAG_array_type: {
    auto *BaseType = dyn_cast_to_ditype_or_null<llvm::DIType>(CT->getBaseType());
    if (BaseType == nullptr) {
      return E_FAIL;
    }

    DWORD dwBaseTypeID = kNullSymbolID;
    IFR(CreateType(BaseType, &dwBaseTypeID));

    auto LazyName = [CT, dwBaseTypeID](Session *pSession, std::string *Name) -> HRESULT {
      auto &SM = pSession->SymMgr();
      Name->clear();
      llvm::raw_string_ostream OS(*Name);
      OS.SetUnbuffered();

      auto *BaseTy = llvm::dyn_cast<llvm::DIType>(CT->getBaseType());
      if (BaseTy == nullptr) {
        return E_FAIL;
      }
      CComPtr<Symbol> SymTy;
      IFR(SM.GetSymbolByID(dwBaseTypeID, &SymTy));
      CComBSTR name;
      IFR(SymTy->get_name(&name));
      if (!name) {
        OS << "???";
      } else {
        OS << CW2A((BSTR)name, CP_UTF8);
      }

      OS << "[";
      bool first = true;
      for (llvm::DINode *N : CT->getElements()) {
        if (!first) {
          OS << "][";
        }
        first = false;
        if (N != nullptr) {
          if (auto *SubRange = llvm::dyn_cast<llvm::DISubrange>(N)) {
            OS << SubRange->getCount();
          } else {
            OS << "???";
          }
        }
      }
      OS << "]";
      return S_OK;
    };

    IFR(AddType<symbol_factory::Type>(dwParentID, CT, pNewTypeID, SymTagArrayType, CT, LazyName));
    TypeInfo *ctTI;
    IFR(GetTypeInfo(CT, &ctTI));
    TypeInfo *baseTI;
    IFR(GetTypeInfo(BaseType, &baseTI));
    int64_t embedCount = 1;
    for (llvm::DINode *N : CT->getElements()) {
      if (N != nullptr) {
        if (auto *SubRange = llvm::dyn_cast<llvm::DISubrange>(N)) {
          embedCount *= SubRange->getCount();
        } else {
          return E_FAIL;
        }
      }
    }
    for (int64_t i = 0; i < embedCount; ++i) {
      ctTI->Embed(*baseTI);
    }
    return S_OK;
  }
  case llvm::dwarf::DW_TAG_class_type: {
    HRESULT hr;
    IFR(hr = CreateHLSLType(CT, pNewTypeID));
    if (hr == S_OK) {
      return S_OK;
    }
    break;
  }
  }

  auto LazyName = [CT](Session *pSession, std::string *Name) -> HRESULT {
    *Name = CT->getName();
    return S_OK;
  };

  IFR(AddType<symbol_factory::UDT>(dwParentID, CT, pNewTypeID, CT, LazyName));

  TypeInfo *udtTI;
  IFR(GetTypeInfo(CT, &udtTI));
  auto udtScope = BeginUDTScope(udtTI);
  for (llvm::DINode *N : CT->getElements()) {
    if (auto *Field = llvm::dyn_cast<llvm::DIType>(N)) {
      DWORD dwUnusedFieldID;
      IFR(CreateType(Field, &dwUnusedFieldID));
    }
  }

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateHLSLType(llvm::DICompositeType *T, DWORD *pNewTypeID) {
  DWORD dwEltTyID;
  std::uint32_t ElemCnt;
  HRESULT hr;
  IFR(hr = IsHLSLVectorType(T, &dwEltTyID, &ElemCnt));
  if (hr == S_OK) {
    // e.g. float4, int2 etc
    return CreateHLSLVectorType(T, dwEltTyID, ElemCnt, pNewTypeID);
  }
  return S_FALSE;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::IsHLSLVectorType(llvm::DICompositeType *T, DWORD *pEltTyID, std::uint32_t *pElemCnt) {
  llvm::StringRef Name = T->getName();
  if (!Name.startswith("vector<")) {
    return S_FALSE;
  }

  llvm::DITemplateParameterArray Args = T->getTemplateParams();
  if (Args.size() != 2) {
    return E_FAIL;
  }

  auto *ElemTyParam = llvm::dyn_cast<llvm::DITemplateTypeParameter>(Args[0]);
  if (ElemTyParam == nullptr) {
    return E_FAIL;
  }
  auto *ElemTy = dyn_cast_to_ditype<llvm::DIType>(ElemTyParam->getType());
  if (ElemTy == nullptr) {
    return E_FAIL;
  }

  DWORD dwEltTyID;
  IFR(CreateType(ElemTy, &dwEltTyID));

  auto *ElemCntParam = llvm::dyn_cast<llvm::DITemplateValueParameter>(Args[1]);
  if (ElemCntParam == nullptr) {
    return E_FAIL;
  }
  auto *ElemCntMD = llvm::dyn_cast<llvm::ConstantAsMetadata>(ElemCntParam->getValue());
  auto *ElemCnt = llvm::dyn_cast_or_null<llvm::ConstantInt>(ElemCntMD->getValue());
  if (ElemCnt == nullptr) {
    return E_FAIL;
  }
  if (ElemCnt->getLimitedValue() > 4) {
    return E_FAIL;
  }

  *pEltTyID = dwEltTyID;
  *pElemCnt = ElemCnt->getLimitedValue();
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateHLSLVectorType(llvm::DICompositeType *T, DWORD pEltTyID, std::uint32_t pElemCnt, DWORD *pNewTypeID) {
  llvm::DITemplateParameterArray Args = T->getTemplateParams();
  if (Args.size() != 2) {
    return E_FAIL;
  }

  auto *ElemTyParam = llvm::dyn_cast<llvm::DITemplateTypeParameter>(Args[0]);
  if (ElemTyParam == nullptr) {
    return E_FAIL;
  }
  auto *ElemTy = dyn_cast_to_ditype<llvm::DIType>(ElemTyParam->getType());
  if (ElemTy == nullptr) {
    return E_FAIL;
  }

  DWORD dwElemTyID;
  IFT(CreateType(ElemTy, &dwElemTyID));

  auto *ElemCntParam = llvm::dyn_cast<llvm::DITemplateValueParameter>(Args[1]);
  if (ElemCntParam == nullptr) {
    return E_FAIL;
  }
  auto *ElemCntMD = llvm::dyn_cast<llvm::ConstantAsMetadata>(ElemCntParam->getValue());
  auto *ElemCnt = llvm::dyn_cast_or_null<llvm::ConstantInt>(ElemCntMD->getValue());
  if (ElemCnt == nullptr) {
    return E_FAIL;
  }
  if (ElemCnt->getLimitedValue() > 4) {
    return E_FAIL;
  }

  const DWORD dwParentID = HlslProgramId;
  IFR(AddType<symbol_factory::VectorType>(dwParentID, T, pNewTypeID, T, dwElemTyID, ElemCnt->getLimitedValue()));

  TypeInfo *vecTI;
  IFR(GetTypeInfo(T, &vecTI));
  TypeInfo *elemTI;
  IFR(GetTypeInfo(ElemTy, &elemTI));
  for (std::uint64_t i = 0; i < ElemCnt->getLimitedValue(); ++i) {
    vecTI->Embed(*elemTI);
  }

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::HandleDerivedType(DWORD dwParentID, llvm::DIDerivedType *DT, DWORD *pNewTypeID) {
  DWORD st;
  LazySymbolName LazyName;

  DWORD dwBaseTypeID = kNullSymbolID;
  auto *BaseTy = llvm::dyn_cast_or_null<llvm::DIType>(DT->getBaseType());
  if (BaseTy != nullptr) {
    IFR(CreateType(BaseTy, &dwBaseTypeID));
  }

  auto LazyNameWithQualifier = [dwBaseTypeID, DT](Session *pSession, std::string *Name, const char *Qualifier) -> HRESULT {
    auto &SM = pSession->SymMgr();
    Name->clear();
    llvm::raw_string_ostream OS(*Name);
    OS.SetUnbuffered();

    auto *BaseTy = llvm::dyn_cast<llvm::DIType>(DT->getBaseType());
    if (BaseTy == nullptr) {
      return E_FAIL;
    }
    CComPtr<Symbol> SymTy;
    IFR(SM.GetSymbolByID(dwBaseTypeID, &SymTy));
    CComBSTR name;
    IFR(SymTy->get_name(&name));
    if (!name) {
      OS << "???";
    } else {
      OS << CW2A((BSTR)name, CP_UTF8);
    }
    OS << Qualifier;
    return S_OK;
  };

  switch (DT->getTag()) {
  case llvm::dwarf::DW_TAG_member: {
    // Type is not really a type, but rather a struct member.
    IFR(CreateUDTField(dwParentID, DT));
    return S_OK;
  }
  default:
    st = SymTagBlock;
    LazyName = [](Session *pSession, std::string *Name) -> HRESULT {
      Name->clear();
      return S_OK;
    };
    break;
  case llvm::dwarf::DW_TAG_typedef: {
    if (dwBaseTypeID == kNullSymbolID) {
      return E_FAIL;
    }

    IFR(AddType<symbol_factory::TypedefType>(dwParentID, DT, pNewTypeID, DT, dwBaseTypeID));

    TypeInfo *dtTI;
    IFR(GetTypeInfo(DT, &dtTI));
    TypeInfo *baseTI;
    IFR(GetTypeInfo(BaseTy, &baseTI));
    dtTI->Embed(*baseTI);

    return S_OK;
  }
  case llvm::dwarf::DW_TAG_const_type: {
    if (dwBaseTypeID == kNullSymbolID) {
      return E_FAIL;
    }

    st = SymTagCustomType;
    LazyName = std::bind(LazyNameWithQualifier, std::placeholders::_1, std::placeholders::_2, " const");
    break;
  }
  case llvm::dwarf::DW_TAG_pointer_type: {
    if (dwBaseTypeID == kNullSymbolID) {
      return E_FAIL;
    }

    st = SymTagPointerType;
    LazyName = std::bind(LazyNameWithQualifier, std::placeholders::_1, std::placeholders::_2, " *");
    break;
  }
  case llvm::dwarf::DW_TAG_reference_type: {
    if (dwBaseTypeID == kNullSymbolID) {
      return E_FAIL;
    }

    st = SymTagCustomType;
    LazyName = std::bind(LazyNameWithQualifier, std::placeholders::_1, std::placeholders::_2, " &");
    break;
  }
  }

  IFR(AddType<symbol_factory::Type>(dwParentID, DT, pNewTypeID, st, DT, LazyName));

  if (DT->getTag() == llvm::dwarf::DW_TAG_const_type) {
    TypeInfo *dtTI;
    IFR(GetTypeInfo(DT, &dtTI));
    TypeInfo *baseTI;
    IFR(GetTypeInfo(BaseTy, &baseTI));
    dtTI->Embed(*baseTI);
  }

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateLocalVariable(DWORD dwParentID, llvm::DILocalVariable *LV) {
  auto *LVTy = dyn_cast_to_ditype<llvm::DIType>(LV->getType());
  if (LVTy == nullptr) {
    return E_FAIL;
  }

  if (m_VarToID.count(LV) != 0) {
    return S_OK;
  }

  DWORD dwLVTypeID;
  IFR(CreateType(LVTy, &dwLVTypeID));
  TypeInfo *varTI;
  IFR(GetTypeInfo(LVTy, &varTI));

  DWORD dwOffsetInUDT = 0;
  auto &newVars = m_VarToID[LV];
  std::vector<llvm::DIType *> Tys = varTI->GetLayout();
  for (llvm::DIType *Ty : Tys) {
    TypeInfo *TI;
    IFR(GetTypeInfo(Ty, &TI));
    DWORD dwNewLVID;
    newVars.emplace_back(std::make_shared<symbol_factory::LocalVarInfo>());
    std::shared_ptr<symbol_factory::LocalVarInfo> VI = newVars.back();
    IFR(AddSymbol<symbol_factory::LocalVariable>(dwParentID, &dwNewLVID, LV, dwLVTypeID, LVTy, VI));
    VI->SetVarID(dwNewLVID);
    VI->SetOffsetInUDT(dwOffsetInUDT);

    static constexpr DWORD kNumBitsPerByte = 8;
    dwOffsetInUDT += Ty->getSizeInBits() / kNumBitsPerByte;
  }
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::GetTypeLayout(llvm::DIType *Ty, std::vector<DWORD> *pRet) {
  pRet->clear();

  TypeInfo *TI;
  IFR(GetTypeInfo(Ty, &TI));
  for (llvm::DIType * T : TI->GetLayout()) {
    TypeInfo *eTI;
    IFR(GetTypeInfo(T, &eTI));
    pRet->emplace_back(eTI->GetTypeID());
  }
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateUDTField(DWORD dwParentID, llvm::DIDerivedType *Field) {
  auto *FieldTy = dyn_cast_to_ditype<llvm::DIType>(Field->getBaseType());
  if (FieldTy == nullptr) {
    return E_FAIL;
  }

  if (m_FieldToID.count(Field) != 0) {
    return S_OK;
  }

  DWORD dwLVTypeID;
  IFR(CreateType(FieldTy, &dwLVTypeID));
  if (m_pCurUDT != nullptr) {
    const DWORD dwOffsetInBytes = CurrentUDTInfo().GetCurrentSizeInBytes();
    DXASSERT_ARGS(dwOffsetInBytes == Field->getOffsetInBits() / 8,
      "%d vs %d",
      dwOffsetInBytes,
      Field->getOffsetInBits() / 8);
    TypeInfo *lvTI;
    IFR(GetTypeInfo(FieldTy, &lvTI));
    CurrentUDTInfo().Embed(*lvTI);
  }

  DWORD dwNewLVID;
  IFR(AddSymbol<symbol_factory::UDTField>(dwParentID, &dwNewLVID, Field, dwLVTypeID, FieldTy));
  m_FieldToID.insert(std::make_pair(Field, dwNewLVID));
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateLocalVariables() {
  llvm::Module *M = &m_Session.ModuleRef();

  llvm::Function *DbgDeclare = llvm::Intrinsic::getDeclaration(M, llvm::Intrinsic::dbg_declare);
  for (llvm::Value *U : DbgDeclare->users()) {
    auto *CI = llvm::dyn_cast<llvm::CallInst>(U);
    auto *LS = llvm::dyn_cast_or_null<llvm::DILocalScope>(CI->getDebugLoc()->getInlinedAtScope());
    auto SymIt = m_ScopeToSym.find(LS);
    if (SymIt == m_ScopeToSym.end()) {
        continue;
    }

    auto *LocalNameMetadata = llvm::dyn_cast<llvm::MetadataAsValue>(CI->getArgOperand(1));
    if (auto *LV = llvm::dyn_cast<llvm::DILocalVariable>(LocalNameMetadata->getMetadata())) {
      const DWORD dwParentID = SymIt->second;
      if (FAILED(CreateLocalVariable(dwParentID, LV))) {
          continue;
      }
    }
  }

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::CreateLiveRanges() {
  // Simple algorithm:
  //   live_range = map from SymbolID to SymbolManager.LiveRange
  //   end_of_scope = map from Scope to RVA
  //   for each I in reverse(pSession.InstructionsRef):
  //     scope = I.scope
  //     if scope not in end_of_scope:
  //       end_of_scope[scope] = rva(I)
  //     if I is dbg.declare:
  //       live_range[symbol of I] = SymbolManager.LiveRange[FirstUseRVA, end_of_scope[scope]]
  llvm::Module *M = &m_Session.ModuleRef();
  m_SymToLR.clear();
  const auto &Instrs = m_Session.InstructionsRef();
  llvm::DenseMap<llvm::DILocalScope *, Session::RVA> EndOfScope;
  for (auto It = Instrs.rbegin(); It != Instrs.rend(); ++It) {
    const Session::RVA RVA = It->first;
    const auto *I = It->second;
    const llvm::DebugLoc &DL = I->getDebugLoc();
    if (!DL) {
      continue;
    }
    llvm::MDNode *LocalScope = DL.getScope();
    if (LocalScope == nullptr) {
      continue;
    }
    auto *LS = llvm::dyn_cast<llvm::DILocalScope>(LocalScope);
    if (LS == nullptr) {
      return E_FAIL;
    }

    if (EndOfScope.count(LS) == 0) {
      EndOfScope.insert(std::make_pair(LS, RVA + 1));
    }
    auto endOfScopeRVA = EndOfScope.find(LS)->second;

    DWORD Reg;
    DWORD RegSize;
    llvm::DILocalVariable *LV;
    uint64_t StartOffset;
    uint64_t EndOffset;
    Session::RVA FirstUseRVA;
    Session::RVA LastUseRVA;
    HRESULT hr = IsDbgDeclareCall(M, I, &Reg, &RegSize, &LV, &StartOffset,
                                  &EndOffset, &FirstUseRVA, &LastUseRVA);
    if (hr != S_OK) {
      continue;
    }

    endOfScopeRVA = std::max<Session::RVA>(endOfScopeRVA, LastUseRVA);

    auto varIt = m_VarToID.find(LV);
    if (varIt == m_VarToID.end()) {
      // All variables should already have been seen and created.
      return E_FAIL;
    }

    for (auto &Var : varIt->second) {
      const DWORD dwOffsetInUDT = Var->GetOffsetInUDT();
      if (dwOffsetInUDT < StartOffset || dwOffsetInUDT >= EndOffset) {
        continue;
      }
      DXASSERT_ARGS((dwOffsetInUDT - StartOffset) % 4 == 0,
                    "Invalid byte offset %d into variable",
                    (dwOffsetInUDT - StartOffset));
      const DWORD dwRegIndex = (dwOffsetInUDT - StartOffset) / 4;
      if (dwRegIndex >= RegSize) {
        continue;
      }
      Var->SetDxilRegister(Reg + dwRegIndex);
      m_SymToLR[Var->GetVarID()] = SymbolManager::LiveRange{
        static_cast<uint32_t>(FirstUseRVA),
        endOfScopeRVA - static_cast<uint32_t>(FirstUseRVA)
      };
    }
  }
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::IsDbgDeclareCall(
    llvm::Module *M, const llvm::Instruction *I, DWORD *pReg, DWORD *pRegSize,
    llvm::DILocalVariable **LV, uint64_t *pStartOffset, uint64_t *pEndOffset,
    dxil_dia::Session::RVA *pLowestUserRVA,
    dxil_dia::Session::RVA *pHighestUserRVA) {
  auto *CI = llvm::dyn_cast<llvm::CallInst>(I);
  if (CI == nullptr) {
    return S_FALSE;
  }

  llvm::Function *DbgDeclare = llvm::Intrinsic::getDeclaration(M, llvm::Intrinsic::dbg_declare);
  if (CI->getCalledFunction() != DbgDeclare) {
    return S_FALSE;
  }

  *LV = nullptr;
  *pReg = *pRegSize = 0;
  *pStartOffset = *pEndOffset = 0;
  *pLowestUserRVA = 0;
  *pHighestUserRVA = 0;

  std::vector<dxil_dia::Session::RVA> usesRVAs;

  bool HasRegister = false;
  if (auto *RegMV = llvm::dyn_cast<llvm::MetadataAsValue>(CI->getArgOperand(0))) {
    if (auto *RegVM = llvm::dyn_cast<llvm::ValueAsMetadata>(RegMV->getMetadata())) {
      if (auto *Reg = llvm::dyn_cast<llvm::Instruction>(RegVM->getValue())) {
        HRESULT hr;
        IFR(hr = GetDxilAllocaRegister(Reg, pReg, pRegSize));
        if (hr != S_OK) {
          return hr;
        }
        HasRegister = true;
        llvm::iterator_range<llvm::Value::user_iterator> users = Reg->users();
        for (llvm::User *user : users) {
          auto *inst = llvm::dyn_cast<llvm::Instruction>(user);
          if (inst != nullptr) {
            auto rva = m_Session.RvaMapRef().find(inst);
            usesRVAs.push_back(rva->second);
          }
        }
      }
    }
  }
  if (!HasRegister) {
    return E_FAIL;
  }

  if (!usesRVAs.empty()) {
    *pLowestUserRVA = *std::min_element(usesRVAs.begin(), usesRVAs.end());
    *pHighestUserRVA = *std::max_element(usesRVAs.begin(), usesRVAs.end());
  }

  if (auto *LVMV = llvm::dyn_cast<llvm::MetadataAsValue>(CI->getArgOperand(1))) {
    *LV = llvm::dyn_cast<llvm::DILocalVariable>(LVMV->getMetadata());
    if (*LV == nullptr) {
      return E_FAIL;
    }
  }

  if (auto *FieldsMV = llvm::dyn_cast<llvm::MetadataAsValue>(CI->getArgOperand(2))) {
    auto *Fields = llvm::dyn_cast<llvm::DIExpression>(FieldsMV->getMetadata());
    if (Fields == nullptr) {
      return E_FAIL;
    }

    static constexpr uint64_t kNumBytesPerDword = 4;
    if (Fields->isBitPiece()) {
      const uint64_t BitPieceOffset = Fields->getBitPieceOffset();
      const uint64_t BitPieceSize = Fields->getBitPieceSize();

      // dxcompiler had a bug (fixed in
      // 4870297404a37269e24ddce7db3bd94a8110fff8) where the BitPieceSize
      // was defined in bytes, not bits. We use the register size in bits to
      // verify if Size is bits or bytes.
      if (*pRegSize * kNumBytesPerDword == BitPieceSize) {
        // Size is bytes.
        *pStartOffset = BitPieceOffset;
        *pEndOffset = *pStartOffset + BitPieceSize;
      } else {
        // Size is (should be) bits; pStartOffset/pEndOffset should be bytes.
        // We don't expect to encounter bit pieces more granular than bytes.
        static constexpr uint64_t kNumBitsPerByte = 8;
        (void)kNumBitsPerByte;
        assert(BitPieceOffset % kNumBitsPerByte == 0);
        assert(BitPieceSize % kNumBitsPerByte == 0);
        *pStartOffset = BitPieceOffset / kNumBitsPerByte;
        *pEndOffset = *pStartOffset + (BitPieceSize / kNumBitsPerByte);
      }
    } else {
      *pStartOffset = 0;
      *pEndOffset = *pRegSize * kNumBytesPerDword;
    }
  }

  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::GetDxilAllocaRegister(llvm::Instruction *I, DWORD *pRegNum, DWORD *pRegSize) {
  auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(I);
  if (Alloca == nullptr) {
    return S_FALSE;
  }

  std::uint32_t uRegNum;
  std::uint32_t uRegSize;
  if (!pix_dxil::PixAllocaReg::FromInst(Alloca, &uRegNum, &uRegSize)) {
    return S_FALSE;
  }

  *pRegNum = uRegNum;
  *pRegSize = uRegSize;
  return S_OK;
}

HRESULT dxil_dia::hlsl_symbols::SymbolManagerInit::PopulateParentToChildrenIDMap(SymbolManager::ParentToChildrenMap *pParentToChildren) {
  DXASSERT_ARGS(m_SymCtors.size() == m_Parent.size(),
                "parents vector must be the same size of symbols ctor vector: %d vs %d",
                m_SymCtors.size(),
                m_Parent.size());

  for (size_t i = 0; i < m_Parent.size(); ++i) {
#ifndef NDEBUG
    {
      CComPtr<Symbol> S;
      IFT(m_SymCtors[i]->Create(&m_Session, &S));
      DXASSERT_ARGS(S->GetID() == i + 1,
                    "Invalid symbol index %d for %d",
                    S->GetID(),
                   i + 1);
    }
#endif  // !NDEBUG

    DXASSERT_ARGS(m_Parent[i] != kNullSymbolID || (i + 1) == HlslProgramId,
                  "Parentless symbol %d", i + 1);
    if (m_Parent[i] != kNullSymbolID) {
      pParentToChildren->emplace(m_Parent[i], i + 1);
    }
  }

  return S_OK;
}

dxil_dia::SymbolManager::SymbolFactory::SymbolFactory(DWORD ID, DWORD ParentID)
    : m_ID(ID), m_ParentID(ParentID) {}

dxil_dia::SymbolManager::SymbolFactory::~SymbolFactory() = default;

dxil_dia::SymbolManager::SymbolManager() = default;

dxil_dia::SymbolManager::~SymbolManager() {
  m_pSession = nullptr;
}

void dxil_dia::SymbolManager::Init(Session *pSes) {
  DXASSERT(m_pSession == nullptr, "SymbolManager already initialized");
  m_pSession = pSes;
  m_symbolCtors.clear();
  m_parentToChildren.clear();

  llvm::DebugInfoFinder &DIFinder = pSes->InfoRef();
  if (DIFinder.compile_unit_count() != 1) {
    throw hlsl::Exception(E_FAIL);
  }
  llvm::DICompileUnit *ShaderCU = *DIFinder.compile_units().begin();

  hlsl_symbols::SymbolManagerInit SMI(pSes, &m_symbolCtors, &m_scopeToID, &m_symbolToLiveRange);

  DWORD dwHlslProgramID;
  IFT(SMI.AddSymbol<hlsl_symbols::symbol_factory::GlobalScope>(kNullSymbolID, &dwHlslProgramID));
  DXASSERT_ARGS(dwHlslProgramID == HlslProgramId,
                "%d vs %d",
                dwHlslProgramID,
                HlslProgramId);


  DWORD dwHlslCompilandID;
  IFT(SMI.AddSymbol<hlsl_symbols::symbol_factory::Compiland>(dwHlslProgramID, &dwHlslCompilandID, ShaderCU));
  m_scopeToID.insert(std::make_pair(ShaderCU, dwHlslCompilandID));
  DXASSERT_ARGS(dwHlslCompilandID == HlslCompilandId,
                "%d vs %d",
                dwHlslCompilandID,
                HlslCompilandId);


  DWORD dwHlslCompilandDetailsId;
  IFT(SMI.AddSymbol<hlsl_symbols::symbol_factory::CompilandDetails>(dwHlslCompilandID, &dwHlslCompilandDetailsId));
  DXASSERT_ARGS(dwHlslCompilandDetailsId == HlslCompilandDetailsId,
                "%d vs %d",
                dwHlslCompilandDetailsId,
                HlslCompilandDetailsId);


  DWORD dwHlslCompilandEnvFlagsID;
  IFT(SMI.AddSymbol<hlsl_symbols::symbol_factory::CompilandEnv<hlsl_symbols::CompilandEnvSymbol::CreateFlags>>(dwHlslCompilandID, &dwHlslCompilandEnvFlagsID));
  DXASSERT_ARGS(dwHlslCompilandEnvFlagsID == HlslCompilandEnvFlagsId,
                "%d vs %d",
                dwHlslCompilandEnvFlagsID,
                HlslCompilandEnvFlagsId);


  DWORD dwHlslCompilandEnvTargetID;
  IFT(SMI.AddSymbol<hlsl_symbols::symbol_factory::CompilandEnv<hlsl_symbols::CompilandEnvSymbol::CreateTarget>>(dwHlslCompilandID, &dwHlslCompilandEnvTargetID));
  DXASSERT_ARGS(dwHlslCompilandEnvTargetID == HlslCompilandEnvTargetId,
                "%d vs %d",
                dwHlslCompilandEnvTargetID,
                HlslCompilandEnvTargetId);


  DWORD dwHlslCompilandEnvEntryID;
  IFT(SMI.AddSymbol<hlsl_symbols::symbol_factory::CompilandEnv<hlsl_symbols::CompilandEnvSymbol::CreateEntry>>(dwHlslCompilandID, &dwHlslCompilandEnvEntryID));
  DXASSERT_ARGS(dwHlslCompilandEnvEntryID == HlslCompilandEnvEntryId,
                "%d vs %d",
                dwHlslCompilandEnvEntryID,
                HlslCompilandEnvEntryId);


  DWORD dwHlslCompilandEnvDefinesID;
  IFT(SMI.AddSymbol<hlsl_symbols::symbol_factory::CompilandEnv<hlsl_symbols::CompilandEnvSymbol::CreateDefines>>(dwHlslCompilandID, &dwHlslCompilandEnvDefinesID));
  DXASSERT_ARGS(dwHlslCompilandEnvDefinesID == HlslCompilandEnvDefinesId,
                "%d vs %d",
                dwHlslCompilandEnvDefinesID,
                HlslCompilandEnvDefinesId);


  DWORD dwHlslCompilandEnvArgumentsID;
  IFT(SMI.AddSymbol<hlsl_symbols::symbol_factory::CompilandEnv<hlsl_symbols::CompilandEnvSymbol::CreateArguments>>(dwHlslCompilandID, &dwHlslCompilandEnvArgumentsID));
  DXASSERT_ARGS(dwHlslCompilandEnvArgumentsID == HlslCompilandEnvArgumentsId,
                "%d vs %d",
                dwHlslCompilandEnvArgumentsID,
                HlslCompilandEnvArgumentsId);


  IFT(SMI.CreateFunctionsForAllCUs());
  IFT(SMI.CreateGlobalVariablesForAllCUs());
  IFT(SMI.CreateLocalVariables());
  IFT(SMI.CreateLiveRanges());
  IFT(SMI.PopulateParentToChildrenIDMap(&m_parentToChildren));
}

HRESULT dxil_dia::SymbolManager::GetSymbolByID(size_t id, Symbol **ppSym) const {
  if (ppSym == nullptr) {
    return E_INVALIDARG;
  }
  *ppSym = nullptr;

  if (m_pSession == nullptr) {
    return E_FAIL;
  }

  if (id <= 0) {
    return E_INVALIDARG;
  }
  if (id > m_symbolCtors.size()) {
    return S_FALSE;
  }

  DxcThreadMalloc TM(m_pSession->GetMallocNoRef());
  IFR(m_symbolCtors[id - 1]->Create(m_pSession, ppSym));
  return S_OK;
}

HRESULT dxil_dia::SymbolManager::GetLiveRangeOf(Symbol *pSym, LiveRange *LR) const {
  const DWORD dwSymID = pSym->GetID();
  if (dwSymID <= 0 || dwSymID > m_symbolCtors.size()) {
    return E_INVALIDARG;
  }

  auto symIt = m_symbolToLiveRange.find(dwSymID);
  if (symIt == m_symbolToLiveRange.end()) {
    return S_FALSE;
  }
  *LR = symIt->second;
  return S_OK;
}

HRESULT dxil_dia::SymbolManager::GetGlobalScope(Symbol **ppSym) const {
  return GetSymbolByID(HlslProgramId, ppSym);
}

HRESULT dxil_dia::SymbolManager::ChildrenOf(DWORD ID, std::vector<CComPtr<Symbol>> *pChildren) const {
  pChildren->clear();
  auto childrenList = m_parentToChildren.equal_range(ID);
  for (auto it = childrenList.first; it != childrenList.second; ++it) {
    CComPtr<Symbol> Child;
    IFR(GetSymbolByID(it->second, &Child));
    pChildren->emplace_back(Child);
  }
  return S_OK;
}

HRESULT dxil_dia::SymbolManager::ChildrenOf(Symbol *pSym, std::vector<CComPtr<Symbol>> *pChildren) const {
  const std::uint32_t pSymID = pSym->GetID();
  IFR(ChildrenOf(pSymID, pChildren));
  return S_OK;
}

HRESULT dxil_dia::SymbolManager::DbgScopeOf(const llvm::Instruction *instr, SymbolChildrenEnumerator **ppRet) const {
  *ppRet = nullptr;

  const llvm::DebugLoc &DL = instr->getDebugLoc();
  if (!DL) {
    return S_FALSE;
  }

  llvm::MDNode *LocalScope = DL.getInlinedAtScope();
  if (LocalScope == nullptr) {
    LocalScope = DL.getScope();
  }
  if (LocalScope == nullptr) {
    return S_FALSE;
  }
  auto *LS = llvm::dyn_cast<llvm::DILocalScope>(LocalScope);
  if (LS == nullptr) {
    // This is a failure as instructions should always live in a DILocalScope
    return E_FAIL;
  }

  auto scopeIt = m_scopeToID.find(LS);
  if (scopeIt == m_scopeToID.end()) {
    // This is a failure because all scopes should already exist in the symbol manager.
    return E_FAIL;
  }

  CComPtr<SymbolChildrenEnumerator> ret = SymbolChildrenEnumerator::Alloc(m_pSession->GetMallocNoRef());
  if (!ret) {
    return E_OUTOFMEMORY;
  }

  CComPtr<Symbol> s;
  IFR(GetSymbolByID(scopeIt->second, &s));
  std::vector<CComPtr<Symbol>> children{s};
  ret->Init(std::move(children));

  *ppRet = ret.Detach();
  return S_OK;
}
