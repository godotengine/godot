//===-- llvm/CodeGen/WinEHFuncInfo.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Data structures and associated state for Windows exception handling schemes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_WINEHFUNCINFO_H
#define LLVM_CODEGEN_WINEHFUNCINFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class BasicBlock;
class Constant;
class Function;
class GlobalVariable;
class InvokeInst;
class IntrinsicInst;
class LandingPadInst;
class MCSymbol;
class Value;

enum ActionType { Catch, Cleanup };

class ActionHandler {
public:
  ActionHandler(BasicBlock *BB, ActionType Type)
      : StartBB(BB), Type(Type), EHState(-1), HandlerBlockOrFunc(nullptr) {}

  ActionType getType() const { return Type; }
  BasicBlock *getStartBlock() const { return StartBB; }

  bool hasBeenProcessed() { return HandlerBlockOrFunc != nullptr; }

  void setHandlerBlockOrFunc(Constant *F) { HandlerBlockOrFunc = F; }
  Constant *getHandlerBlockOrFunc() { return HandlerBlockOrFunc; }

  void setEHState(int State) { EHState = State; }
  int getEHState() const { return EHState; }

private:
  BasicBlock *StartBB;
  ActionType Type;
  int EHState;

  // Can be either a BlockAddress or a Function depending on the EH personality.
  Constant *HandlerBlockOrFunc;
};

class CatchHandler : public ActionHandler {
public:
  CatchHandler(BasicBlock *BB, Constant *Selector, BasicBlock *NextBB)
      : ActionHandler(BB, ActionType::Catch), Selector(Selector),
      NextBB(NextBB), ExceptionObjectVar(nullptr),
      ExceptionObjectIndex(-1) {}

  // Method for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ActionHandler *H) {
    return H->getType() == ActionType::Catch;
  }

  Constant *getSelector() const { return Selector; }
  BasicBlock *getNextBB() const { return NextBB; }

  const Value *getExceptionVar() { return ExceptionObjectVar; }
  TinyPtrVector<BasicBlock *> &getReturnTargets() { return ReturnTargets; }

  void setExceptionVar(const Value *Val) { ExceptionObjectVar = Val; }
  void setExceptionVarIndex(int Index) { ExceptionObjectIndex = Index;  }
  int getExceptionVarIndex() const { return ExceptionObjectIndex; }
  void setReturnTargets(TinyPtrVector<BasicBlock *> &Targets) {
    ReturnTargets = Targets;
  }

private:
  Constant *Selector;
  BasicBlock *NextBB;
  // While catch handlers are being outlined the ExceptionObjectVar field will
  // be populated with the instruction in the parent frame that corresponds
  // to the exception object (or nullptr if the catch does not use an
  // exception object) and the ExceptionObjectIndex field will be -1.
  // When the parseEHActions function is called to populate a vector of
  // instances of this class, the ExceptionObjectVar field will be nullptr
  // and the ExceptionObjectIndex will be the index of the exception object in
  // the parent function's localescape block.
  const Value *ExceptionObjectVar;
  int ExceptionObjectIndex;
  TinyPtrVector<BasicBlock *> ReturnTargets;
};

class CleanupHandler : public ActionHandler {
public:
  CleanupHandler(BasicBlock *BB) : ActionHandler(BB, ActionType::Cleanup) {}

  // Method for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ActionHandler *H) {
    return H->getType() == ActionType::Cleanup;
  }
};

void parseEHActions(const IntrinsicInst *II,
                    SmallVectorImpl<std::unique_ptr<ActionHandler>> &Actions);

// The following structs respresent the .xdata for functions using C++
// exceptions on Windows.

struct WinEHUnwindMapEntry {
  int ToState;
  Function *Cleanup;
};

struct WinEHHandlerType {
  int Adjectives;
  GlobalVariable *TypeDescriptor;
  int CatchObjRecoverIdx;
  Function *Handler;
};

struct WinEHTryBlockMapEntry {
  int TryLow;
  int TryHigh;
  SmallVector<WinEHHandlerType, 1> HandlerArray;
};

struct WinEHFuncInfo {
  DenseMap<const Function *, const LandingPadInst *> RootLPad;
  DenseMap<const Function *, const InvokeInst *> LastInvoke;
  DenseMap<const Function *, int> HandlerEnclosedState;
  DenseMap<const Function *, bool> LastInvokeVisited;
  DenseMap<const LandingPadInst *, int> LandingPadStateMap;
  DenseMap<const Function *, int> CatchHandlerParentFrameObjIdx;
  DenseMap<const Function *, int> CatchHandlerParentFrameObjOffset;
  DenseMap<const Function *, int> CatchHandlerMaxState;
  DenseMap<const Function *, int> HandlerBaseState;
  SmallVector<WinEHUnwindMapEntry, 4> UnwindMap;
  SmallVector<WinEHTryBlockMapEntry, 4> TryBlockMap;
  SmallVector<std::pair<MCSymbol *, int>, 4> IPToStateList;
  int UnwindHelpFrameIdx = INT_MAX;
  int UnwindHelpFrameOffset = -1;
  unsigned NumIPToStateFuncsVisited = 0;

  /// localescape index of the 32-bit EH registration node. Set by
  /// WinEHStatePass and used indirectly by SEH filter functions of the parent.
  int EHRegNodeEscapeIndex = INT_MAX;

  WinEHFuncInfo() {}
};

/// Analyze the IR in ParentFn and it's handlers to build WinEHFuncInfo, which
/// describes the state numbers and tables used by __CxxFrameHandler3. This
/// analysis assumes that WinEHPrepare has already been run.
void calculateWinCXXEHStateNumbers(const Function *ParentFn,
                                   WinEHFuncInfo &FuncInfo);

}
#endif // LLVM_CODEGEN_WINEHFUNCINFO_H
