//===--- ArgList.cpp - Argument List Management ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Option/ArgList.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::opt;

void arg_iterator::SkipToNextArg() {
  for (; Current != Args.end(); ++Current) {
    // Done if there are no filters.
    if (!Id0.isValid())
      break;

    // Otherwise require a match.
    const Option &O = (*Current)->getOption();
    if (O.matches(Id0) ||
        (Id1.isValid() && O.matches(Id1)) ||
        (Id2.isValid() && O.matches(Id2)))
      break;
  }
}

void ArgList::append(Arg *A) {
  Args.push_back(A);
}

void ArgList::eraseArg(OptSpecifier Id) {
  Args.erase(std::remove_if(begin(), end(),
                            [=](Arg *A) { return A->getOption().matches(Id); }),
             end());
}

Arg *ArgList::getLastArgNoClaim(OptSpecifier Id) const {
  // FIXME: Make search efficient?
  for (const_reverse_iterator it = rbegin(), ie = rend(); it != ie; ++it)
    if ((*it)->getOption().matches(Id))
      return *it;
  return nullptr;
}

Arg *ArgList::getLastArgNoClaim(OptSpecifier Id0, OptSpecifier Id1) const {
  // FIXME: Make search efficient?
  for (const_reverse_iterator it = rbegin(), ie = rend(); it != ie; ++it)
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1))
      return *it;
  return nullptr;
}

Arg *ArgList::getLastArgNoClaim(OptSpecifier Id0, OptSpecifier Id1,
                                OptSpecifier Id2) const {
  // FIXME: Make search efficient?
  for (const_reverse_iterator it = rbegin(), ie = rend(); it != ie; ++it)
    if ((*it)->getOption().matches(Id0) || (*it)->getOption().matches(Id1) ||
        (*it)->getOption().matches(Id2))
      return *it;
  return nullptr;
}

Arg *ArgList::getLastArgNoClaim(OptSpecifier Id0, OptSpecifier Id1,
                                OptSpecifier Id2, OptSpecifier Id3) const {
  // FIXME: Make search efficient?
  for (const_reverse_iterator it = rbegin(), ie = rend(); it != ie; ++it)
    if ((*it)->getOption().matches(Id0) || (*it)->getOption().matches(Id1) ||
        (*it)->getOption().matches(Id2) || (*it)->getOption().matches(Id3))
      return *it;
  return nullptr;
}

Arg *ArgList::getLastArg(OptSpecifier Id) const {
  Arg *Res = nullptr;
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id)) {
      Res = *it;
      Res->claim();
    }
  }

  return Res;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1) const {
  Arg *Res = nullptr;
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1)) {
      Res = *it;
      Res->claim();

    }
  }

  return Res;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1,
                         OptSpecifier Id2) const {
  Arg *Res = nullptr;
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1) ||
        (*it)->getOption().matches(Id2)) {
      Res = *it;
      Res->claim();
    }
  }

  return Res;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1,
                         OptSpecifier Id2, OptSpecifier Id3) const {
  Arg *Res = nullptr;
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1) ||
        (*it)->getOption().matches(Id2) ||
        (*it)->getOption().matches(Id3)) {
      Res = *it;
      Res->claim();
    }
  }

  return Res;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1,
                         OptSpecifier Id2, OptSpecifier Id3,
                         OptSpecifier Id4) const {
  Arg *Res = nullptr;
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1) ||
        (*it)->getOption().matches(Id2) ||
        (*it)->getOption().matches(Id3) ||
        (*it)->getOption().matches(Id4)) {
      Res = *it;
      Res->claim();
    }
  }

  return Res;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1,
                         OptSpecifier Id2, OptSpecifier Id3,
                         OptSpecifier Id4, OptSpecifier Id5) const {
  Arg *Res = nullptr;
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1) ||
        (*it)->getOption().matches(Id2) ||
        (*it)->getOption().matches(Id3) ||
        (*it)->getOption().matches(Id4) ||
        (*it)->getOption().matches(Id5)) {
      Res = *it;
      Res->claim();
    }
  }

  return Res;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1,
                         OptSpecifier Id2, OptSpecifier Id3,
                         OptSpecifier Id4, OptSpecifier Id5,
                         OptSpecifier Id6) const {
  Arg *Res = nullptr;
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1) ||
        (*it)->getOption().matches(Id2) ||
        (*it)->getOption().matches(Id3) ||
        (*it)->getOption().matches(Id4) ||
        (*it)->getOption().matches(Id5) ||
        (*it)->getOption().matches(Id6)) {
      Res = *it;
      Res->claim();
    }
  }

  return Res;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1,
                         OptSpecifier Id2, OptSpecifier Id3,
                         OptSpecifier Id4, OptSpecifier Id5,
                         OptSpecifier Id6, OptSpecifier Id7) const {
  Arg *Res = nullptr;
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1) ||
        (*it)->getOption().matches(Id2) ||
        (*it)->getOption().matches(Id3) ||
        (*it)->getOption().matches(Id4) ||
        (*it)->getOption().matches(Id5) ||
        (*it)->getOption().matches(Id6) ||
        (*it)->getOption().matches(Id7)) {
      Res = *it;
      Res->claim();
    }
  }

  return Res;
}

bool ArgList::hasFlag(OptSpecifier Pos, OptSpecifier Neg, bool Default) const {
  if (Arg *A = getLastArg(Pos, Neg))
    return A->getOption().matches(Pos);
  return Default;
}

bool ArgList::hasFlag(OptSpecifier Pos, OptSpecifier PosAlias, OptSpecifier Neg,
                      bool Default) const {
  if (Arg *A = getLastArg(Pos, PosAlias, Neg))
    return A->getOption().matches(Pos) || A->getOption().matches(PosAlias);
  return Default;
}

StringRef ArgList::getLastArgValue(OptSpecifier Id,
                                         StringRef Default) const {
  if (Arg *A = getLastArg(Id))
    return A->getValue();
  return Default;
}

std::vector<std::string> ArgList::getAllArgValues(OptSpecifier Id) const {
  SmallVector<const char *, 16> Values;
  AddAllArgValues(Values, Id);
  return std::vector<std::string>(Values.begin(), Values.end());
}

void ArgList::AddLastArg(ArgStringList &Output, OptSpecifier Id) const {
  if (Arg *A = getLastArg(Id)) {
    A->claim();
    A->render(*this, Output);
  }
}

void ArgList::AddLastArg(ArgStringList &Output, OptSpecifier Id0,
                         OptSpecifier Id1) const {
  if (Arg *A = getLastArg(Id0, Id1)) {
    A->claim();
    A->render(*this, Output);
  }
}

void ArgList::AddAllArgs(ArgStringList &Output, OptSpecifier Id0,
                         OptSpecifier Id1, OptSpecifier Id2) const {
  for (auto Arg: filtered(Id0, Id1, Id2)) {
    Arg->claim();
    Arg->render(*this, Output);
  }
}

void ArgList::AddAllArgValues(ArgStringList &Output, OptSpecifier Id0,
                              OptSpecifier Id1, OptSpecifier Id2) const {
  for (auto Arg : filtered(Id0, Id1, Id2)) {
    Arg->claim();
    const auto &Values = Arg->getValues();
    Output.append(Values.begin(), Values.end());
  }
}

void ArgList::AddAllArgsTranslated(ArgStringList &Output, OptSpecifier Id0,
                                   const char *Translation,
                                   bool Joined) const {
  for (auto Arg: filtered(Id0)) {
    Arg->claim();

    if (Joined) {
      Output.push_back(MakeArgString(StringRef(Translation) +
                                     Arg->getValue(0)));
    } else {
      Output.push_back(Translation);
      Output.push_back(Arg->getValue(0));
    }
  }
}

void ArgList::ClaimAllArgs(OptSpecifier Id0) const {
  for (auto Arg : filtered(Id0))
    Arg->claim();
}

void ArgList::ClaimAllArgs() const {
  for (const_iterator it = begin(), ie = end(); it != ie; ++it)
    if (!(*it)->isClaimed())
      (*it)->claim();
}

const char *ArgList::GetOrMakeJoinedArgString(unsigned Index,
                                              StringRef LHS,
                                              StringRef RHS) const {
  StringRef Cur = getArgString(Index);
  if (Cur.size() == LHS.size() + RHS.size() &&
      Cur.startswith(LHS) && Cur.endswith(RHS))
    return Cur.data();

  return MakeArgString(LHS + RHS);
}

//

void InputArgList::releaseMemory() {
  // An InputArgList always owns its arguments.
  for (Arg *A : *this)
    delete A;
}

InputArgList::InputArgList(const char* const *ArgBegin,
                           const char* const *ArgEnd)
  : NumInputArgStrings(ArgEnd - ArgBegin) {
  ArgStrings.append(ArgBegin, ArgEnd);
}

unsigned InputArgList::MakeIndex(StringRef String0) const {
  unsigned Index = ArgStrings.size();

  // Tuck away so we have a reliable const char *.
  SynthesizedStrings.push_back(String0);
  ArgStrings.push_back(SynthesizedStrings.back().c_str());

  return Index;
}

unsigned InputArgList::MakeIndex(StringRef String0,
                                 StringRef String1) const {
  unsigned Index0 = MakeIndex(String0);
  unsigned Index1 = MakeIndex(String1);
  assert(Index0 + 1 == Index1 && "Unexpected non-consecutive indices!");
  (void) Index1;
  return Index0;
}

const char *InputArgList::MakeArgStringRef(StringRef Str) const {
  return getArgString(MakeIndex(Str));
}

//

DerivedArgList::DerivedArgList(const InputArgList &BaseArgs)
    : BaseArgs(BaseArgs) {}

const char *DerivedArgList::MakeArgStringRef(StringRef Str) const {
  return BaseArgs.MakeArgString(Str);
}

void DerivedArgList::AddSynthesizedArg(Arg *A) {
  SynthesizedArgs.push_back(std::unique_ptr<Arg>(A));
}

Arg *DerivedArgList::MakeFlagArg(const Arg *BaseArg, const Option Opt) const {
  SynthesizedArgs.push_back(
      make_unique<Arg>(Opt, MakeArgString(Opt.getPrefix() + Opt.getName()),
                       BaseArgs.MakeIndex(Opt.getName()), BaseArg));
  return SynthesizedArgs.back().get();
}

Arg *DerivedArgList::MakePositionalArg(const Arg *BaseArg, const Option Opt,
                                       StringRef Value) const {
  unsigned Index = BaseArgs.MakeIndex(Value);
  SynthesizedArgs.push_back(
      make_unique<Arg>(Opt, MakeArgString(Opt.getPrefix() + Opt.getName()),
                       Index, BaseArgs.getArgString(Index), BaseArg));
  return SynthesizedArgs.back().get();
}

Arg *DerivedArgList::MakeSeparateArg(const Arg *BaseArg, const Option Opt,
                                     StringRef Value) const {
  unsigned Index = BaseArgs.MakeIndex(Opt.getName(), Value);
  SynthesizedArgs.push_back(
      make_unique<Arg>(Opt, MakeArgString(Opt.getPrefix() + Opt.getName()),
                       Index, BaseArgs.getArgString(Index + 1), BaseArg));
  return SynthesizedArgs.back().get();
}

Arg *DerivedArgList::MakeJoinedArg(const Arg *BaseArg, const Option Opt,
                                   StringRef Value) const {
  unsigned Index = BaseArgs.MakeIndex((Opt.getName() + Value).str());
  SynthesizedArgs.push_back(make_unique<Arg>(
      Opt, MakeArgString(Opt.getPrefix() + Opt.getName()), Index,
      BaseArgs.getArgString(Index) + Opt.getName().size(), BaseArg));
  return SynthesizedArgs.back().get();
}
