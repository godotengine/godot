//===--- Arg.cpp - Argument Implementations -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Option/Arg.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::opt;

Arg::Arg(const Option Opt, StringRef S, unsigned Index, const Arg *BaseArg)
    : Opt(Opt), BaseArg(BaseArg), Spelling(S), Index(Index), Claimed(false),
      OwnsValues(false) {}

Arg::Arg(const Option Opt, StringRef S, unsigned Index, const char *Value0,
         const Arg *BaseArg)
    : Opt(Opt), BaseArg(BaseArg), Spelling(S), Index(Index), Claimed(false),
      OwnsValues(false) {
  Values.push_back(Value0);
}

Arg::Arg(const Option Opt, StringRef S, unsigned Index, const char *Value0,
         const char *Value1, const Arg *BaseArg)
    : Opt(Opt), BaseArg(BaseArg), Spelling(S), Index(Index), Claimed(false),
      OwnsValues(false) {
  Values.push_back(Value0);
  Values.push_back(Value1);
}

Arg::~Arg() {
  if (OwnsValues) {
    for (unsigned i = 0, e = Values.size(); i != e; ++i)
      delete[] Values[i];
  }
}

void Arg::dump() const {
  llvm::errs() << "<";

  llvm::errs() << " Opt:";
  Opt.dump();

  llvm::errs() << " Index:" << Index;

  llvm::errs() << " Values: [";
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (i) llvm::errs() << ", ";
    llvm::errs() << "'" << Values[i] << "'";
  }

  llvm::errs() << "]>\n";
}

std::string Arg::getAsString(const ArgList &Args) const {
  SmallString<256> Res;
  llvm::raw_svector_ostream OS(Res);

  ArgStringList ASL;
  render(Args, ASL);
  for (ArgStringList::iterator
         it = ASL.begin(), ie = ASL.end(); it != ie; ++it) {
    if (it != ASL.begin())
      OS << ' ';
    OS << *it;
  }

  return OS.str();
}

void Arg::renderAsInput(const ArgList &Args, ArgStringList &Output) const {
  if (!getOption().hasNoOptAsInput()) {
    render(Args, Output);
    return;
  }

  Output.append(Values.begin(), Values.end());
}

void Arg::render(const ArgList &Args, ArgStringList &Output) const {
  switch (getOption().getRenderStyle()) {
  case Option::RenderValuesStyle:
    Output.append(Values.begin(), Values.end());
    break;

  case Option::RenderCommaJoinedStyle: {
    SmallString<256> Res;
    llvm::raw_svector_ostream OS(Res);
    OS << getSpelling();
    for (unsigned i = 0, e = getNumValues(); i != e; ++i) {
      if (i) OS << ',';
      OS << getValue(i);
    }
    Output.push_back(Args.MakeArgString(OS.str()));
    break;
  }

 case Option::RenderJoinedStyle:
    Output.push_back(Args.GetOrMakeJoinedArgString(
                       getIndex(), getSpelling(), getValue(0)));
    Output.append(Values.begin() + 1, Values.end());
    break;

  case Option::RenderSeparateStyle:
    Output.push_back(Args.MakeArgString(getSpelling()));
    Output.append(Values.begin(), Values.end());
    break;
  }
}
