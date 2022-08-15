//===--- ArgList.h - Argument List Management -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPTION_ARGLIST_H
#define LLVM_OPTION_ARGLIST_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Option/Option.h"
#include <list>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
namespace opt {
class ArgList;
class Option;

/// arg_iterator - Iterates through arguments stored inside an ArgList.
class arg_iterator {
  /// The current argument.
  SmallVectorImpl<Arg*>::const_iterator Current;

  /// The argument list we are iterating over.
  const ArgList &Args;

  /// Optional filters on the arguments which will be match. Most clients
  /// should never want to iterate over arguments without filters, so we won't
  /// bother to factor this into two separate iterator implementations.
  //
  // FIXME: Make efficient; the idea is to provide efficient iteration over
  // all arguments which match a particular id and then just provide an
  // iterator combinator which takes multiple iterators which can be
  // efficiently compared and returns them in order.
  OptSpecifier Id0, Id1, Id2;

  void SkipToNextArg();

public:
  typedef Arg * const *                 value_type;
  typedef Arg * const &                 reference;
  typedef Arg * const *                 pointer;
  typedef std::forward_iterator_tag   iterator_category;
  typedef std::ptrdiff_t              difference_type;

  arg_iterator(SmallVectorImpl<Arg *>::const_iterator it, const ArgList &Args,
               OptSpecifier Id0 = 0U, OptSpecifier Id1 = 0U,
               OptSpecifier Id2 = 0U)
      : Current(it), Args(Args), Id0(Id0), Id1(Id1), Id2(Id2) {
    SkipToNextArg();
  }

  operator const Arg*() { return *Current; }
  reference operator*() const { return *Current; }
  pointer operator->() const { return Current; }

  arg_iterator &operator++() {
    ++Current;
    SkipToNextArg();
    return *this;
  }

  arg_iterator operator++(int) {
    arg_iterator tmp(*this);
    ++(*this);
    return tmp;
  }

  friend bool operator==(arg_iterator LHS, arg_iterator RHS) {
    return LHS.Current == RHS.Current;
  }
  friend bool operator!=(arg_iterator LHS, arg_iterator RHS) {
    return !(LHS == RHS);
  }
};

/// ArgList - Ordered collection of driver arguments.
///
/// The ArgList class manages a list of Arg instances as well as
/// auxiliary data and convenience methods to allow Tools to quickly
/// check for the presence of Arg instances for a particular Option
/// and to iterate over groups of arguments.
class ArgList {
public:
  typedef SmallVector<Arg*, 16> arglist_type;
  typedef arglist_type::iterator iterator;
  typedef arglist_type::const_iterator const_iterator;
  typedef arglist_type::reverse_iterator reverse_iterator;
  typedef arglist_type::const_reverse_iterator const_reverse_iterator;

private:
  /// The internal list of arguments.
  arglist_type Args;

protected:
  // Make the default special members protected so they won't be used to slice
  // derived objects, but can still be used by derived objects to implement
  // their own special members.
  ArgList() = default;
  // Explicit move operations to ensure the container is cleared post-move
  // otherwise it could lead to a double-delete in the case of moving of an
  // InputArgList which deletes the contents of the container. If we could fix
  // up the ownership here (delegate storage/ownership to the derived class so
  // it can be a container of unique_ptr) this would be simpler.
  ArgList(ArgList &&RHS) : Args(std::move(RHS.Args)) { RHS.Args.clear(); }
  ArgList &operator=(ArgList &&RHS) {
    Args = std::move(RHS.Args);
    RHS.Args.clear();
    return *this;
  }
  // Protect the dtor to ensure this type is never destroyed polymorphically.
  ~ArgList() = default;

public:

  /// @name Arg Access
  /// @{

  /// append - Append \p A to the arg list.
  void append(Arg *A);

  arglist_type &getArgs() { return Args; }
  const arglist_type &getArgs() const { return Args; }

  unsigned size() const { return Args.size(); }

  /// @}
  /// @name Arg Iteration
  /// @{

  iterator begin() { return Args.begin(); }
  iterator end() { return Args.end(); }

  reverse_iterator rbegin() { return Args.rbegin(); }
  reverse_iterator rend() { return Args.rend(); }

  const_iterator begin() const { return Args.begin(); }
  const_iterator end() const { return Args.end(); }

  const_reverse_iterator rbegin() const { return Args.rbegin(); }
  const_reverse_iterator rend() const { return Args.rend(); }

  arg_iterator filtered_begin(OptSpecifier Id0 = 0U, OptSpecifier Id1 = 0U,
                              OptSpecifier Id2 = 0U) const {
    return arg_iterator(Args.begin(), *this, Id0, Id1, Id2);
  }
  arg_iterator filtered_end() const {
    return arg_iterator(Args.end(), *this);
  }

  iterator_range<arg_iterator> filtered(OptSpecifier Id0 = 0U,
                                        OptSpecifier Id1 = 0U,
                                        OptSpecifier Id2 = 0U) const {
    return make_range(filtered_begin(Id0, Id1, Id2), filtered_end());
  }

  /// @}
  /// @name Arg Removal
  /// @{

  /// eraseArg - Remove any option matching \p Id.
  void eraseArg(OptSpecifier Id);

  /// @}
  /// @name Arg Access
  /// @{

  /// hasArg - Does the arg list contain any option matching \p Id.
  ///
  /// \p Claim Whether the argument should be claimed, if it exists.
  bool hasArgNoClaim(OptSpecifier Id) const {
    return getLastArgNoClaim(Id) != nullptr;
  }
  bool hasArg(OptSpecifier Id) const {
    return getLastArg(Id) != nullptr;
  }
  bool hasArg(OptSpecifier Id0, OptSpecifier Id1) const {
    return getLastArg(Id0, Id1) != nullptr;
  }
  bool hasArg(OptSpecifier Id0, OptSpecifier Id1, OptSpecifier Id2) const {
    return getLastArg(Id0, Id1, Id2) != nullptr;
  }

  /// getLastArg - Return the last argument matching \p Id, or null.
  ///
  /// \p Claim Whether the argument should be claimed, if it exists.
  Arg *getLastArgNoClaim(OptSpecifier Id) const;
  Arg *getLastArgNoClaim(OptSpecifier Id0, OptSpecifier Id1) const;
  Arg *getLastArgNoClaim(OptSpecifier Id0, OptSpecifier Id1,
                         OptSpecifier Id2) const;
  Arg *getLastArgNoClaim(OptSpecifier Id0, OptSpecifier Id1, OptSpecifier Id2,
                         OptSpecifier Id3) const;
  Arg *getLastArg(OptSpecifier Id) const;
  Arg *getLastArg(OptSpecifier Id0, OptSpecifier Id1) const;
  Arg *getLastArg(OptSpecifier Id0, OptSpecifier Id1, OptSpecifier Id2) const;
  Arg *getLastArg(OptSpecifier Id0, OptSpecifier Id1, OptSpecifier Id2,
                  OptSpecifier Id3) const;
  Arg *getLastArg(OptSpecifier Id0, OptSpecifier Id1, OptSpecifier Id2,
                  OptSpecifier Id3, OptSpecifier Id4) const;
  Arg *getLastArg(OptSpecifier Id0, OptSpecifier Id1, OptSpecifier Id2,
                  OptSpecifier Id3, OptSpecifier Id4, OptSpecifier Id5) const;
  Arg *getLastArg(OptSpecifier Id0, OptSpecifier Id1, OptSpecifier Id2,
                  OptSpecifier Id3, OptSpecifier Id4, OptSpecifier Id5,
                  OptSpecifier Id6) const;
  Arg *getLastArg(OptSpecifier Id0, OptSpecifier Id1, OptSpecifier Id2,
                  OptSpecifier Id3, OptSpecifier Id4, OptSpecifier Id5,
                  OptSpecifier Id6, OptSpecifier Id7) const;

  /// getArgString - Return the input argument string at \p Index.
  virtual const char *getArgString(unsigned Index) const = 0;

  /// getNumInputArgStrings - Return the number of original argument strings,
  /// which are guaranteed to be the first strings in the argument string
  /// list.
  virtual unsigned getNumInputArgStrings() const = 0;

  /// @}
  /// @name Argument Lookup Utilities
  /// @{

  /// getLastArgValue - Return the value of the last argument, or a default.
  StringRef getLastArgValue(OptSpecifier Id,
                                  StringRef Default = "") const;

  /// getAllArgValues - Get the values of all instances of the given argument
  /// as strings.
  std::vector<std::string> getAllArgValues(OptSpecifier Id) const;

  /// @}
  /// @name Translation Utilities
  /// @{

  /// hasFlag - Given an option \p Pos and its negative form \p Neg, return
  /// true if the option is present, false if the negation is present, and
  /// \p Default if neither option is given. If both the option and its
  /// negation are present, the last one wins.
  bool hasFlag(OptSpecifier Pos, OptSpecifier Neg, bool Default=true) const;

  /// hasFlag - Given an option \p Pos, an alias \p PosAlias and its negative
  /// form \p Neg, return true if the option or its alias is present, false if
  /// the negation is present, and \p Default if none of the options are
  /// given. If multiple options are present, the last one wins.
  bool hasFlag(OptSpecifier Pos, OptSpecifier PosAlias, OptSpecifier Neg,
               bool Default = true) const;

  /// AddLastArg - Render only the last argument match \p Id0, if present.
  void AddLastArg(ArgStringList &Output, OptSpecifier Id0) const;
  void AddLastArg(ArgStringList &Output, OptSpecifier Id0,
                  OptSpecifier Id1) const;

  /// AddAllArgs - Render all arguments matching the given ids.
  void AddAllArgs(ArgStringList &Output, OptSpecifier Id0,
                  OptSpecifier Id1 = 0U, OptSpecifier Id2 = 0U) const;

  /// AddAllArgValues - Render the argument values of all arguments
  /// matching the given ids.
  void AddAllArgValues(ArgStringList &Output, OptSpecifier Id0,
                        OptSpecifier Id1 = 0U, OptSpecifier Id2 = 0U) const;

  /// AddAllArgsTranslated - Render all the arguments matching the
  /// given ids, but forced to separate args and using the provided
  /// name instead of the first option value.
  ///
  /// \param Joined - If true, render the argument as joined with
  /// the option specifier.
  void AddAllArgsTranslated(ArgStringList &Output, OptSpecifier Id0,
                            const char *Translation,
                            bool Joined = false) const;

  /// ClaimAllArgs - Claim all arguments which match the given
  /// option id.
  void ClaimAllArgs(OptSpecifier Id0) const;

  /// ClaimAllArgs - Claim all arguments.
  ///
  void ClaimAllArgs() const;

  /// @}
  /// @name Arg Synthesis
  /// @{

  /// Construct a constant string pointer whose
  /// lifetime will match that of the ArgList.
  virtual const char *MakeArgStringRef(StringRef Str) const = 0;
  const char *MakeArgString(const Twine &Str) const {
    SmallString<256> Buf;
    return MakeArgStringRef(Str.toStringRef(Buf));
  }

  /// \brief Create an arg string for (\p LHS + \p RHS), reusing the
  /// string at \p Index if possible.
  const char *GetOrMakeJoinedArgString(unsigned Index, StringRef LHS,
                                        StringRef RHS) const;

  /// @}
};

class InputArgList final : public ArgList {
private:
  /// List of argument strings used by the contained Args.
  ///
  /// This is mutable since we treat the ArgList as being the list
  /// of Args, and allow routines to add new strings (to have a
  /// convenient place to store the memory) via MakeIndex.
  mutable ArgStringList ArgStrings;

  /// Strings for synthesized arguments.
  ///
  /// This is mutable since we treat the ArgList as being the list
  /// of Args, and allow routines to add new strings (to have a
  /// convenient place to store the memory) via MakeIndex.
  mutable std::list<std::string> SynthesizedStrings;

  /// The number of original input argument strings.
  unsigned NumInputArgStrings;

  /// Release allocated arguments.
  void releaseMemory();

public:
  InputArgList(const char* const *ArgBegin, const char* const *ArgEnd);
  InputArgList(InputArgList &&RHS)
      : ArgList(std::move(RHS)), ArgStrings(std::move(RHS.ArgStrings)),
        SynthesizedStrings(std::move(RHS.SynthesizedStrings)),
        NumInputArgStrings(RHS.NumInputArgStrings) {}
  InputArgList &operator=(InputArgList &&RHS) {
    releaseMemory();
    ArgList::operator=(std::move(RHS));
    ArgStrings = std::move(RHS.ArgStrings);
    SynthesizedStrings = std::move(RHS.SynthesizedStrings);
    NumInputArgStrings = RHS.NumInputArgStrings;
    return *this;
  }
  ~InputArgList() { releaseMemory(); }

  const char *getArgString(unsigned Index) const override {
    return ArgStrings[Index];
  }

  unsigned getNumInputArgStrings() const override {
    return NumInputArgStrings;
  }

  /// @name Arg Synthesis
  /// @{

public:
  /// MakeIndex - Get an index for the given string(s).
  unsigned MakeIndex(StringRef String0) const;
  unsigned MakeIndex(StringRef String0, StringRef String1) const;

  using ArgList::MakeArgString;
  const char *MakeArgStringRef(StringRef Str) const override;

  /// @}
};

/// DerivedArgList - An ordered collection of driver arguments,
/// whose storage may be in another argument list.
class DerivedArgList final : public ArgList {
  const InputArgList &BaseArgs;

  /// The list of arguments we synthesized.
  mutable SmallVector<std::unique_ptr<Arg>, 16> SynthesizedArgs;

public:
  /// Construct a new derived arg list from \p BaseArgs.
  DerivedArgList(const InputArgList &BaseArgs);

  const char *getArgString(unsigned Index) const override {
    return BaseArgs.getArgString(Index);
  }

  unsigned getNumInputArgStrings() const override {
    return BaseArgs.getNumInputArgStrings();
  }

  const InputArgList &getBaseArgs() const {
    return BaseArgs;
  }

  /// @name Arg Synthesis
  /// @{

  /// AddSynthesizedArg - Add a argument to the list of synthesized arguments
  /// (to be freed).
  void AddSynthesizedArg(Arg *A);

  using ArgList::MakeArgString;
  const char *MakeArgStringRef(StringRef Str) const override;

  /// AddFlagArg - Construct a new FlagArg for the given option \p Id and
  /// append it to the argument list.
  void AddFlagArg(const Arg *BaseArg, const Option Opt) {
    append(MakeFlagArg(BaseArg, Opt));
  }

  /// AddPositionalArg - Construct a new Positional arg for the given option
  /// \p Id, with the provided \p Value and append it to the argument
  /// list.
  void AddPositionalArg(const Arg *BaseArg, const Option Opt,
                        StringRef Value) {
    append(MakePositionalArg(BaseArg, Opt, Value));
  }


  /// AddSeparateArg - Construct a new Positional arg for the given option
  /// \p Id, with the provided \p Value and append it to the argument
  /// list.
  void AddSeparateArg(const Arg *BaseArg, const Option Opt,
                      StringRef Value) {
    append(MakeSeparateArg(BaseArg, Opt, Value));
  }


  /// AddJoinedArg - Construct a new Positional arg for the given option
  /// \p Id, with the provided \p Value and append it to the argument list.
  void AddJoinedArg(const Arg *BaseArg, const Option Opt,
                    StringRef Value) {
    append(MakeJoinedArg(BaseArg, Opt, Value));
  }


  /// MakeFlagArg - Construct a new FlagArg for the given option \p Id.
  Arg *MakeFlagArg(const Arg *BaseArg, const Option Opt) const;

  /// MakePositionalArg - Construct a new Positional arg for the
  /// given option \p Id, with the provided \p Value.
  Arg *MakePositionalArg(const Arg *BaseArg, const Option Opt,
                          StringRef Value) const;

  /// MakeSeparateArg - Construct a new Positional arg for the
  /// given option \p Id, with the provided \p Value.
  Arg *MakeSeparateArg(const Arg *BaseArg, const Option Opt,
                        StringRef Value) const;

  /// MakeJoinedArg - Construct a new Positional arg for the
  /// given option \p Id, with the provided \p Value.
  Arg *MakeJoinedArg(const Arg *BaseArg, const Option Opt,
                      StringRef Value) const;

  /// @}
};

} // end namespace opt
} // end namespace llvm

#endif
