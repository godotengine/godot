//===- MCWinEH.h - Windows Unwinding Support --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWINEH_H
#define LLVM_MC_MCWINEH_H

#include <vector>

namespace llvm {
class MCContext;
class MCSection;
class MCStreamer;
class MCSymbol;
class StringRef;

namespace WinEH {
struct Instruction {
  const MCSymbol *Label;
  const unsigned Offset;
  const unsigned Register;
  const unsigned Operation;

  Instruction(unsigned Op, MCSymbol *L, unsigned Reg, unsigned Off)
    : Label(L), Offset(Off), Register(Reg), Operation(Op) {}
};

struct FrameInfo {
  const MCSymbol *Begin;
  const MCSymbol *End;
  const MCSymbol *ExceptionHandler;
  const MCSymbol *Function;
  const MCSymbol *PrologEnd;
  const MCSymbol *Symbol;

  bool HandlesUnwind;
  bool HandlesExceptions;

  int LastFrameInst;
  const FrameInfo *ChainedParent;
  std::vector<Instruction> Instructions;

  FrameInfo()
    : Begin(nullptr), End(nullptr), ExceptionHandler(nullptr),
      Function(nullptr), PrologEnd(nullptr), Symbol(nullptr),
      HandlesUnwind(false), HandlesExceptions(false), LastFrameInst(-1),
      ChainedParent(nullptr), Instructions() {}
  FrameInfo(const MCSymbol *Function, const MCSymbol *BeginFuncEHLabel)
    : Begin(BeginFuncEHLabel), End(nullptr), ExceptionHandler(nullptr),
      Function(Function), PrologEnd(nullptr), Symbol(nullptr),
      HandlesUnwind(false), HandlesExceptions(false), LastFrameInst(-1),
      ChainedParent(nullptr), Instructions() {}
  FrameInfo(const MCSymbol *Function, const MCSymbol *BeginFuncEHLabel,
            const FrameInfo *ChainedParent)
    : Begin(BeginFuncEHLabel), End(nullptr), ExceptionHandler(nullptr),
      Function(Function), PrologEnd(nullptr), Symbol(nullptr),
      HandlesUnwind(false), HandlesExceptions(false), LastFrameInst(-1),
      ChainedParent(ChainedParent), Instructions() {}
};

class UnwindEmitter {
public:
  static MCSection *getPDataSection(const MCSymbol *Function,
                                    MCContext &Context);
  static MCSection *getXDataSection(const MCSymbol *Function,
                                    MCContext &Context);

  virtual ~UnwindEmitter() { }

  //
  // This emits the unwind info sections (.pdata and .xdata in PE/COFF).
  //
  virtual void Emit(MCStreamer &Streamer) const = 0;
  virtual void EmitUnwindInfo(MCStreamer &Streamer, FrameInfo *FI) const = 0;
};
}
}

#endif
