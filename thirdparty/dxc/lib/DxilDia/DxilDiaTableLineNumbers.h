///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableLineNumbers.h                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"

#include <vector>

#include "dia2.h"

#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Instruction.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"

#include "DxilDia.h"
#include "DxilDiaTable.h"

namespace dxil_dia {
class Session;

class LineNumber : public IDiaLineNumber {
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<Session> m_pSession;
  const llvm::Instruction *m_inst;

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  STDMETHODIMP QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDiaLineNumber>(this, iid, ppvObject);
  }

  LineNumber(
    /* [in] */ IMalloc *pMalloc,
    /* [in] */ Session *pSession,
    /* [in] */ const llvm::Instruction * inst);

  const llvm::DebugLoc &DL() const;

  const llvm::Instruction *Inst() const { return m_inst; }

  STDMETHODIMP get_compiland(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_sourceFile(
    /* [retval][out] */ IDiaSourceFile **pRetVal) override;

  STDMETHODIMP get_lineNumber(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_lineNumberEnd(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_columnNumber(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_columnNumberEnd(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_addressSection(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_addressOffset(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_relativeVirtualAddress(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_virtualAddress(
    /* [retval][out] */ ULONGLONG *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_length(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_sourceFileId(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_statement(
    /* [retval][out] */ BOOL *pRetVal) override;

  STDMETHODIMP get_compilandId(
    /* [retval][out] */ DWORD *pRetVal) override;
};

class LineNumbersTable : public impl::TableBase<IDiaEnumLineNumbers, IDiaLineNumber> {
public:
  LineNumbersTable(
    /* [in] */ IMalloc *pMalloc,
    /* [in] */ Session *pSession);

  LineNumbersTable(
    /* [in] */ IMalloc *pMalloc,
    /* [in] */ Session *pSession,
    /* [in] */ std::vector<const llvm::Instruction*> &&instructions);

  HRESULT GetItem(
    /* [in] */ DWORD index, 
    /* [out] */ IDiaLineNumber **ppItem) override;

private:
  // Keep a reference to the instructions that contain the line numbers.
  const std::vector<const llvm::Instruction *> &m_instructions;

  // Provide storage space for instructions for when the table contains
  // a subset of all instructions.
  std::vector<const llvm::Instruction *> m_instructionsStorage;
};

}  // namespace dxil_dia