///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// ReducibilityAnalysis.h                                                    //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Implements reducibility analysis pass.                                    //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once


namespace llvm {
class Module;
class Function;
class PassRegistry;
class FunctionPass;

enum class IrreducibilityAction {
  ThrowException,
  PrintLog,
  Ignore,
};

extern char &ReducibilityAnalysisID;

llvm::FunctionPass *createReducibilityAnalysisPass(IrreducibilityAction Action = IrreducibilityAction::ThrowException);

void initializeReducibilityAnalysisPass(llvm::PassRegistry&);

bool IsReducible(const llvm::Module &M, IrreducibilityAction Action = IrreducibilityAction::ThrowException);
bool IsReducible(const llvm::Function &F, IrreducibilityAction Action = IrreducibilityAction::ThrowException);

}
