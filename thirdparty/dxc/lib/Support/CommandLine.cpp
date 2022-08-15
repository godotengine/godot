//===-- CommandLine.cpp - Command line parser implementation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements a command line argument processor that is useful when
// creating a tool.  It provides a simple, minimalistic interface that is easily
// extensible and supports nonlocal (library) command line options.
//
// Note that rather than trying to figure out what this code does, you could try
// reading the library documentation located in docs/CommandLine.html
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm-c/Support.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/config.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <map>
using namespace llvm;
using namespace cl;

#define DEBUG_TYPE "commandline"

//===----------------------------------------------------------------------===//
// Template instantiations and anchors.
//
namespace llvm {
namespace cl {
template class basic_parser<bool>;
template class basic_parser<boolOrDefault>;
template class basic_parser<int>;
template class basic_parser<unsigned>;
template class basic_parser<unsigned long long>;
template class basic_parser<double>;
template class basic_parser<float>;
template class basic_parser<std::string>;
template class basic_parser<char>;

template class opt<unsigned>;
template class opt<int>;
template class opt<std::string>;
template class opt<char>;
template class opt<bool>;
}
} // end namespace llvm::cl

// Pin the vtables to this file.
void GenericOptionValue::anchor() {}
void OptionValue<boolOrDefault>::anchor() {}
void OptionValue<std::string>::anchor() {}
void Option::anchor() {}
void basic_parser_impl::anchor() {}
void parser<bool>::anchor() {}
void parser<boolOrDefault>::anchor() {}
void parser<int>::anchor() {}
void parser<unsigned>::anchor() {}
void parser<unsigned long long>::anchor() {}
void parser<double>::anchor() {}
void parser<float>::anchor() {}
void parser<std::string>::anchor() {}
void parser<char>::anchor() {}

//===----------------------------------------------------------------------===//

namespace {

class CommandLineParser {
public:
  // Globals for name and overview of program.  Program name is not a string to
  // avoid static ctor/dtor issues.
  std::string ProgramName;
  const char *ProgramOverview;

  // This collects additional help to be printed.
  std::vector<const char *> MoreHelp;

  SmallVector<Option *, 4> PositionalOpts;
  SmallVector<Option *, 4> SinkOpts;
  StringMap<Option *> OptionsMap;

  Option *ConsumeAfterOpt; // The ConsumeAfter option if it exists.

  // This collects the different option categories that have been registered.
  SmallPtrSet<OptionCategory *, 16> RegisteredOptionCategories;

  CommandLineParser() : ProgramOverview(nullptr), ConsumeAfterOpt(nullptr) {}

  void ParseCommandLineOptions(int argc, const char *const *argv,
                               const char *Overview);

  void addLiteralOption(Option &Opt, const char *Name) {
    if (!Opt.hasArgStr()) {
      if (!OptionsMap.insert(std::make_pair(Name, &Opt)).second) {
        errs() << ProgramName << ": CommandLine Error: Option '" << Name
               << "' registered more than once!\n";
        report_fatal_error("inconsistency in registered CommandLine options");
      }
    }
  }

  void addOption(Option *O) {
    bool HadErrors = false;
    if (O->hasArgStr()) {
      // Add argument to the argument map!
      if (!OptionsMap.insert(std::make_pair(O->ArgStr, O)).second) {
        errs() << ProgramName << ": CommandLine Error: Option '" << O->ArgStr
               << "' registered more than once!\n";
        HadErrors = true;
      }
    }

    // Remember information about positional options.
    if (O->getFormattingFlag() == cl::Positional)
      PositionalOpts.push_back(O);
    else if (O->getMiscFlags() & cl::Sink) // Remember sink options
      SinkOpts.push_back(O);
    else if (O->getNumOccurrencesFlag() == cl::ConsumeAfter) {
      if (ConsumeAfterOpt) {
        O->error("Cannot specify more than one option with cl::ConsumeAfter!");
        HadErrors = true;
      }
      ConsumeAfterOpt = O;
    }

    // Fail hard if there were errors. These are strictly unrecoverable and
    // indicate serious issues such as conflicting option names or an
    // incorrectly
    // linked LLVM distribution.
    if (HadErrors)
      report_fatal_error("inconsistency in registered CommandLine options");
  }

  void removeOption(Option *O) {
    SmallVector<const char *, 16> OptionNames;
    O->getExtraOptionNames(OptionNames);
    if (O->hasArgStr())
      OptionNames.push_back(O->ArgStr);
    for (auto Name : OptionNames)
      OptionsMap.erase(StringRef(Name));

    if (O->getFormattingFlag() == cl::Positional)
      for (auto Opt = PositionalOpts.begin(); Opt != PositionalOpts.end();
           ++Opt) {
        if (*Opt == O) {
          PositionalOpts.erase(Opt);
          break;
        }
      }
    else if (O->getMiscFlags() & cl::Sink)
      for (auto Opt = SinkOpts.begin(); Opt != SinkOpts.end(); ++Opt) {
        if (*Opt == O) {
          SinkOpts.erase(Opt);
          break;
        }
      }
    else if (O == ConsumeAfterOpt)
      ConsumeAfterOpt = nullptr;
  }

  bool hasOptions() {
    return (!OptionsMap.empty() || !PositionalOpts.empty() ||
            nullptr != ConsumeAfterOpt);
  }

  void updateArgStr(Option *O, const char *NewName) {
    if (!OptionsMap.insert(std::make_pair(NewName, O)).second) {
      errs() << ProgramName << ": CommandLine Error: Option '" << O->ArgStr
             << "' registered more than once!\n";
      report_fatal_error("inconsistency in registered CommandLine options");
    }
    OptionsMap.erase(StringRef(O->ArgStr));
  }

  void printOptionValues();

  void registerCategory(OptionCategory *cat) {
    assert(std::count_if(RegisteredOptionCategories.begin(),
                         RegisteredOptionCategories.end(),
                         [cat](const OptionCategory *Category) {
                           return cat->getName() == Category->getName();
                         }) == 0 &&
           "Duplicate option categories");

    RegisteredOptionCategories.insert(cat);
  }

private:
  Option *LookupOption(StringRef &Arg, StringRef &Value);
};

} // namespace

static ManagedStatic<CommandLineParser> GlobalParser;

void cl::AddLiteralOption(Option &O, const char *Name) {
  GlobalParser->addLiteralOption(O, Name);
}

extrahelp::extrahelp(const char *Help) : morehelp(Help) {
  GlobalParser->MoreHelp.push_back(Help);
}

void Option::addArgument() {
  GlobalParser->addOption(this);
  FullyInitialized = true;
}

void Option::removeArgument() { GlobalParser->removeOption(this); }

void Option::setArgStr(const char *S) {
  if (FullyInitialized)
    GlobalParser->updateArgStr(this, S);
  ArgStr = S;
}

// Initialise the general option category.
OptionCategory *llvm::cl::GeneralCategory; // HLSL Change - GeneralCategory is now a pointer

void OptionCategory::registerCategory() {
  GlobalParser->registerCategory(this);
}

//===----------------------------------------------------------------------===//
// Basic, shared command line option processing machinery.
//

/// LookupOption - Lookup the option specified by the specified option on the
/// command line.  If there is a value specified (after an equal sign) return
/// that as well.  This assumes that leading dashes have already been stripped.
Option *CommandLineParser::LookupOption(StringRef &Arg, StringRef &Value) {
  // Reject all dashes.
  if (Arg.empty())
    return nullptr;

  size_t EqualPos = Arg.find('=');

  // If we have an equals sign, remember the value.
  if (EqualPos == StringRef::npos) {
    // Look up the option.
    StringMap<Option *>::const_iterator I = OptionsMap.find(Arg);
    return I != OptionsMap.end() ? I->second : nullptr;
  }

  // If the argument before the = is a valid option name, we match.  If not,
  // return Arg unmolested.
  StringMap<Option *>::const_iterator I =
      OptionsMap.find(Arg.substr(0, EqualPos));
  if (I == OptionsMap.end())
    return nullptr;

  Value = Arg.substr(EqualPos + 1);
  Arg = Arg.substr(0, EqualPos);
  return I->second;
}

/// LookupNearestOption - Lookup the closest match to the option specified by
/// the specified option on the command line.  If there is a value specified
/// (after an equal sign) return that as well.  This assumes that leading dashes
/// have already been stripped.
static Option *LookupNearestOption(StringRef Arg,
                                   const StringMap<Option *> &OptionsMap,
                                   std::string &NearestString) {
  // Reject all dashes.
  if (Arg.empty())
    return nullptr;

  // Split on any equal sign.
  std::pair<StringRef, StringRef> SplitArg = Arg.split('=');
  StringRef &LHS = SplitArg.first; // LHS == Arg when no '=' is present.
  StringRef &RHS = SplitArg.second;

  // Find the closest match.
  Option *Best = nullptr;
  unsigned BestDistance = 0;
  for (StringMap<Option *>::const_iterator it = OptionsMap.begin(),
                                           ie = OptionsMap.end();
       it != ie; ++it) {
    Option *O = it->second;
    SmallVector<const char *, 16> OptionNames;
    O->getExtraOptionNames(OptionNames);
    if (O->ArgStr[0])
      OptionNames.push_back(O->ArgStr);

    bool PermitValue = O->getValueExpectedFlag() != cl::ValueDisallowed;
    StringRef Flag = PermitValue ? LHS : Arg;
    for (size_t i = 0, e = OptionNames.size(); i != e; ++i) {
      StringRef Name = OptionNames[i];
      unsigned Distance = StringRef(Name).edit_distance(
          Flag, /*AllowReplacements=*/true, /*MaxEditDistance=*/BestDistance);
      if (!Best || Distance < BestDistance) {
        Best = O;
        BestDistance = Distance;
        if (RHS.empty() || !PermitValue)
          NearestString = OptionNames[i];
        else
          NearestString = (Twine(OptionNames[i]) + "=" + RHS).str();
      }
    }
  }

  return Best;
}

/// CommaSeparateAndAddOccurrence - A wrapper around Handler->addOccurrence()
/// that does special handling of cl::CommaSeparated options.
static bool CommaSeparateAndAddOccurrence(Option *Handler, unsigned pos,
                                          StringRef ArgName, StringRef Value,
                                          bool MultiArg = false) {
  // Check to see if this option accepts a comma separated list of values.  If
  // it does, we have to split up the value into multiple values.
  if (Handler->getMiscFlags() & CommaSeparated) {
    StringRef Val(Value);
    StringRef::size_type Pos = Val.find(',');

    while (Pos != StringRef::npos) {
      // Process the portion before the comma.
      if (Handler->addOccurrence(pos, ArgName, Val.substr(0, Pos), MultiArg))
        return true;
      // Erase the portion before the comma, AND the comma.
      Val = Val.substr(Pos + 1);
      Value.substr(Pos + 1); // Increment the original value pointer as well.
      // Check for another comma.
      Pos = Val.find(',');
    }

    Value = Val;
  }

  if (Handler->addOccurrence(pos, ArgName, Value, MultiArg))
    return true;

  return false;
}

/// ProvideOption - For Value, this differentiates between an empty value ("")
/// and a null value (StringRef()).  The later is accepted for arguments that
/// don't allow a value (-foo) the former is rejected (-foo=).
static inline bool ProvideOption(Option *Handler, StringRef ArgName,
                                 StringRef Value, int argc,
                                 const char *const *argv, int &i) {
  // Is this a multi-argument option?
  unsigned NumAdditionalVals = Handler->getNumAdditionalVals();

  // Enforce value requirements
  switch (Handler->getValueExpectedFlag()) {
  case ValueRequired:
    if (!Value.data()) { // No value specified?
      if (i + 1 >= argc)
        return Handler->error("requires a value!");
      // Steal the next argument, like for '-o filename'
      assert(argv && "null check");
      Value = argv[++i];
    }
    break;
  case ValueDisallowed:
    if (NumAdditionalVals > 0)
      return Handler->error("multi-valued option specified"
                            " with ValueDisallowed modifier!");

    if (Value.data())
      return Handler->error("does not allow a value! '" + Twine(Value) +
                            "' specified.");
    break;
  case ValueOptional:
    break;
  }

  // If this isn't a multi-arg option, just run the handler.
  if (NumAdditionalVals == 0)
    return CommaSeparateAndAddOccurrence(Handler, i, ArgName, Value);

  // If it is, run the handle several times.
  bool MultiArg = false;

  if (Value.data()) {
    if (CommaSeparateAndAddOccurrence(Handler, i, ArgName, Value, MultiArg))
      return true;
    --NumAdditionalVals;
    MultiArg = true;
  }

  while (NumAdditionalVals > 0) {
    if (i + 1 >= argc)
      return Handler->error("not enough values!");
    assert(argv && "null check");
    Value = argv[++i];

    if (CommaSeparateAndAddOccurrence(Handler, i, ArgName, Value, MultiArg))
      return true;
    MultiArg = true;
    --NumAdditionalVals;
  }
  return false;
}

static bool ProvidePositionalOption(Option *Handler, StringRef Arg, int i) {
  int Dummy = i;
  return ProvideOption(Handler, Handler->ArgStr, Arg, 0, nullptr, Dummy);
}

// Option predicates...
static inline bool isGrouping(const Option *O) {
  return O->getFormattingFlag() == cl::Grouping;
}
static inline bool isPrefixedOrGrouping(const Option *O) {
  return isGrouping(O) || O->getFormattingFlag() == cl::Prefix;
}

// getOptionPred - Check to see if there are any options that satisfy the
// specified predicate with names that are the prefixes in Name.  This is
// checked by progressively stripping characters off of the name, checking to
// see if there options that satisfy the predicate.  If we find one, return it,
// otherwise return null.
//
static Option *getOptionPred(StringRef Name, size_t &Length,
                             bool (*Pred)(const Option *),
                             const StringMap<Option *> &OptionsMap) {

  StringMap<Option *>::const_iterator OMI = OptionsMap.find(Name);

  // Loop while we haven't found an option and Name still has at least two
  // characters in it (so that the next iteration will not be the empty
  // string.
  while (OMI == OptionsMap.end() && Name.size() > 1) {
    Name = Name.substr(0, Name.size() - 1); // Chop off the last character.
    OMI = OptionsMap.find(Name);
  }

  if (OMI != OptionsMap.end() && Pred(OMI->second)) {
    Length = Name.size();
    return OMI->second; // Found one!
  }
  return nullptr; // No option found!
}

/// HandlePrefixedOrGroupedOption - The specified argument string (which started
/// with at least one '-') does not fully match an available option.  Check to
/// see if this is a prefix or grouped option.  If so, split arg into output an
/// Arg/Value pair and return the Option to parse it with.
static Option *
HandlePrefixedOrGroupedOption(StringRef &Arg, StringRef &Value,
                              bool &ErrorParsing,
                              const StringMap<Option *> &OptionsMap) {
  if (Arg.size() == 1)
    return nullptr;

  // Do the lookup!
  size_t Length = 0;
  Option *PGOpt = getOptionPred(Arg, Length, isPrefixedOrGrouping, OptionsMap);
  if (!PGOpt)
    return nullptr;

  // If the option is a prefixed option, then the value is simply the
  // rest of the name...  so fall through to later processing, by
  // setting up the argument name flags and value fields.
  if (PGOpt->getFormattingFlag() == cl::Prefix) {
    Value = Arg.substr(Length);
    Arg = Arg.substr(0, Length);
    assert(OptionsMap.count(Arg) && OptionsMap.find(Arg)->second == PGOpt);
    return PGOpt;
  }

  // This must be a grouped option... handle them now.  Grouping options can't
  // have values.
  assert(isGrouping(PGOpt) && "Broken getOptionPred!");

  do {
    // Move current arg name out of Arg into OneArgName.
    StringRef OneArgName = Arg.substr(0, Length);
    Arg = Arg.substr(Length);

    // Because ValueRequired is an invalid flag for grouped arguments,
    // we don't need to pass argc/argv in.
    assert(PGOpt->getValueExpectedFlag() != cl::ValueRequired &&
           "Option can not be cl::Grouping AND cl::ValueRequired!");
    int Dummy = 0;
    ErrorParsing |=
        ProvideOption(PGOpt, OneArgName, StringRef(), 0, nullptr, Dummy);

    // Get the next grouping option.
    PGOpt = getOptionPred(Arg, Length, isGrouping, OptionsMap);
  } while (PGOpt && Length != Arg.size());

  // Return the last option with Arg cut down to just the last one.
  return PGOpt;
}

static bool RequiresValue(const Option *O) {
  return O->getNumOccurrencesFlag() == cl::Required ||
         O->getNumOccurrencesFlag() == cl::OneOrMore;
}

static bool EatsUnboundedNumberOfValues(const Option *O) {
  return O->getNumOccurrencesFlag() == cl::ZeroOrMore ||
         O->getNumOccurrencesFlag() == cl::OneOrMore;
}

static bool isWhitespace(char C) { return strchr(" \t\n\r\f\v", C); }

static bool isQuote(char C) { return C == '\"' || C == '\''; }

static bool isGNUSpecial(char C) { return strchr("\\\"\' ", C); }

void cl::TokenizeGNUCommandLine(StringRef Src, StringSaver &Saver,
                                SmallVectorImpl<const char *> &NewArgv,
                                bool MarkEOLs) {
  SmallString<128> Token;
  for (size_t I = 0, E = Src.size(); I != E; ++I) {
    // Consume runs of whitespace.
    if (Token.empty()) {
      while (I != E && isWhitespace(Src[I])) {
        // Mark the end of lines in response files
        if (MarkEOLs && Src[I] == '\n')
          NewArgv.push_back(nullptr);
        ++I;
      }
      if (I == E)
        break;
    }

    // Backslashes can escape backslashes, spaces, and other quotes.  Otherwise
    // they are literal.  This makes it much easier to read Windows file paths.
    if (I + 1 < E && Src[I] == '\\' && isGNUSpecial(Src[I + 1])) {
      ++I; // Skip the escape.
      Token.push_back(Src[I]);
      continue;
    }

    // Consume a quoted string.
    if (isQuote(Src[I])) {
      char Quote = Src[I++];
      while (I != E && Src[I] != Quote) {
        // Backslashes are literal, unless they escape a special character.
        if (Src[I] == '\\' && I + 1 != E && isGNUSpecial(Src[I + 1]))
          ++I;
        Token.push_back(Src[I]);
        ++I;
      }
      if (I == E)
        break;
      continue;
    }

    // End the token if this is whitespace.
    if (isWhitespace(Src[I])) {
      if (!Token.empty())
        NewArgv.push_back(Saver.save(Token.c_str()));
      Token.clear();
      continue;
    }

    // This is a normal character.  Append it.
    Token.push_back(Src[I]);
  }

  // Append the last token after hitting EOF with no whitespace.
  if (!Token.empty())
    NewArgv.push_back(Saver.save(Token.c_str()));
  // Mark the end of response files
  if (MarkEOLs)
    NewArgv.push_back(nullptr);
}

/// Backslashes are interpreted in a rather complicated way in the Windows-style
/// command line, because backslashes are used both to separate path and to
/// escape double quote. This method consumes runs of backslashes as well as the
/// following double quote if it's escaped.
///
///  * If an even number of backslashes is followed by a double quote, one
///    backslash is output for every pair of backslashes, and the last double
///    quote remains unconsumed. The double quote will later be interpreted as
///    the start or end of a quoted string in the main loop outside of this
///    function.
///
///  * If an odd number of backslashes is followed by a double quote, one
///    backslash is output for every pair of backslashes, and a double quote is
///    output for the last pair of backslash-double quote. The double quote is
///    consumed in this case.
///
///  * Otherwise, backslashes are interpreted literally.
static size_t parseBackslash(StringRef Src, size_t I, SmallString<128> &Token) {
  size_t E = Src.size();
  int BackslashCount = 0;
  // Skip the backslashes.
  do {
    ++I;
    ++BackslashCount;
  } while (I != E && Src[I] == '\\');

  bool FollowedByDoubleQuote = (I != E && Src[I] == '"');
  if (FollowedByDoubleQuote) {
    Token.append(BackslashCount / 2, '\\');
    if (BackslashCount % 2 == 0)
      return I - 1;
    Token.push_back('"');
    return I;
  }
  Token.append(BackslashCount, '\\');
  return I - 1;
}

void cl::TokenizeWindowsCommandLine(StringRef Src, StringSaver &Saver,
                                    SmallVectorImpl<const char *> &NewArgv,
                                    bool MarkEOLs) {
  SmallString<128> Token;

  // This is a small state machine to consume characters until it reaches the
  // end of the source string.
  enum { INIT, UNQUOTED, QUOTED } State = INIT;
  for (size_t I = 0, E = Src.size(); I != E; ++I) {
    // INIT state indicates that the current input index is at the start of
    // the string or between tokens.
    if (State == INIT) {
      if (isWhitespace(Src[I])) {
        // Mark the end of lines in response files
        if (MarkEOLs && Src[I] == '\n')
          NewArgv.push_back(nullptr);
        continue;
      }
      if (Src[I] == '"') {
        State = QUOTED;
        continue;
      }
      if (Src[I] == '\\') {
        I = parseBackslash(Src, I, Token);
        State = UNQUOTED;
        continue;
      }
      Token.push_back(Src[I]);
      State = UNQUOTED;
      continue;
    }

    // UNQUOTED state means that it's reading a token not quoted by double
    // quotes.
    if (State == UNQUOTED) {
      // Whitespace means the end of the token.
      if (isWhitespace(Src[I])) {
        NewArgv.push_back(Saver.save(Token.c_str()));
        Token.clear();
        State = INIT;
        // Mark the end of lines in response files
        if (MarkEOLs && Src[I] == '\n')
          NewArgv.push_back(nullptr);
        continue;
      }
      if (Src[I] == '"') {
        State = QUOTED;
        continue;
      }
      if (Src[I] == '\\') {
        I = parseBackslash(Src, I, Token);
        continue;
      }
      Token.push_back(Src[I]);
      continue;
    }

    // QUOTED state means that it's reading a token quoted by double quotes.
    if (State == QUOTED) {
      if (Src[I] == '"') {
        State = UNQUOTED;
        continue;
      }
      if (Src[I] == '\\') {
        I = parseBackslash(Src, I, Token);
        continue;
      }
      Token.push_back(Src[I]);
    }
  }
  // Append the last token after hitting EOF with no whitespace.
  if (!Token.empty())
    NewArgv.push_back(Saver.save(Token.c_str()));
  // Mark the end of response files
  if (MarkEOLs)
    NewArgv.push_back(nullptr);
}

// It is called byte order marker but the UTF-8 BOM is actually not affected
// by the host system's endianness.
static bool hasUTF8ByteOrderMark(ArrayRef<char> S) {
  return (S.size() >= 3 && S[0] == '\xef' && S[1] == '\xbb' && S[2] == '\xbf');
}

static bool ExpandResponseFile(const char *FName, StringSaver &Saver,
                               TokenizerCallback Tokenizer,
                               SmallVectorImpl<const char *> &NewArgv,
                               bool MarkEOLs = false) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MemBufOrErr =
      MemoryBuffer::getFile(FName);
  if (!MemBufOrErr)
    return false;
  MemoryBuffer &MemBuf = *MemBufOrErr.get();
  StringRef Str(MemBuf.getBufferStart(), MemBuf.getBufferSize());

  // If we have a UTF-16 byte order mark, convert to UTF-8 for parsing.
  ArrayRef<char> BufRef(MemBuf.getBufferStart(), MemBuf.getBufferEnd());
  std::string UTF8Buf;
  if (hasUTF16ByteOrderMark(BufRef)) {
    if (!convertUTF16ToUTF8String(BufRef, UTF8Buf))
      return false;
    Str = StringRef(UTF8Buf);
  }
  // If we see UTF-8 BOM sequence at the beginning of a file, we shall remove
  // these bytes before parsing.
  // Reference: http://en.wikipedia.org/wiki/UTF-8#Byte_order_mark
  else if (hasUTF8ByteOrderMark(BufRef))
    Str = StringRef(BufRef.data() + 3, BufRef.size() - 3);

  // Tokenize the contents into NewArgv.
  Tokenizer(Str, Saver, NewArgv, MarkEOLs);

  return true;
}

/// \brief Expand response files on a command line recursively using the given
/// StringSaver and tokenization strategy.
bool cl::ExpandResponseFiles(StringSaver &Saver, TokenizerCallback Tokenizer,
                             SmallVectorImpl<const char *> &Argv,
                             bool MarkEOLs) {
  unsigned RspFiles = 0;
  bool AllExpanded = true;

  // Don't cache Argv.size() because it can change.
  for (unsigned I = 0; I != Argv.size();) {
    const char *Arg = Argv[I];
    // Check if it is an EOL marker
    if (Arg == nullptr) {
      ++I;
      continue;
    }
    if (Arg[0] != '@') {
      ++I;
      continue;
    }

    // If we have too many response files, leave some unexpanded.  This avoids
    // crashing on self-referential response files.
    if (RspFiles++ > 20)
      return false;

    // Replace this response file argument with the tokenization of its
    // contents.  Nested response files are expanded in subsequent iterations.
    // FIXME: If a nested response file uses a relative path, is it relative to
    // the cwd of the process or the response file?
    SmallVector<const char *, 0> ExpandedArgv;
    if (!ExpandResponseFile(Arg + 1, Saver, Tokenizer, ExpandedArgv,
                            MarkEOLs)) {
      // We couldn't read this file, so we leave it in the argument stream and
      // move on.
      AllExpanded = false;
      ++I;
      continue;
    }
    Argv.erase(Argv.begin() + I);
    Argv.insert(Argv.begin() + I, ExpandedArgv.begin(), ExpandedArgv.end());
  }
  return AllExpanded;
}

/// ParseEnvironmentOptions - An alternative entry point to the
/// CommandLine library, which allows you to read the program's name
/// from the caller (as PROGNAME) and its command-line arguments from
/// an environment variable (whose name is given in ENVVAR).
///
void cl::ParseEnvironmentOptions(const char *progName, const char *envVar,
                                 const char *Overview) {
  // Check args.
  assert(progName && "Program name not specified");
  assert(envVar && "Environment variable name missing");

  // Get the environment variable they want us to parse options out of.
  const char *envValue = getenv(envVar);
  if (!envValue)
    return;

  // Get program's "name", which we wouldn't know without the caller
  // telling us.
  SmallVector<const char *, 20> newArgv;
  BumpPtrAllocator A;
  BumpPtrStringSaver Saver(A);
  newArgv.push_back(Saver.save(progName));

  // Parse the value of the environment variable into a "command line"
  // and hand it off to ParseCommandLineOptions().
  TokenizeGNUCommandLine(envValue, Saver, newArgv);
  int newArgc = static_cast<int>(newArgv.size());
  ParseCommandLineOptions(newArgc, &newArgv[0], Overview);
}

void cl::ParseCommandLineOptions(int argc, const char *const *argv,
                                 const char *Overview) {
  GlobalParser->ParseCommandLineOptions(argc, argv, Overview);
}

void CommandLineParser::ParseCommandLineOptions(int argc,
                                                const char *const *argv,
                                                const char *Overview) {
  // assert(hasOptions() && "No options specified!"); // HLSL Change - it's valid to have no options for the DLL build

  // Expand response files.
  SmallVector<const char *, 20> newArgv(argv, argv + argc);
  BumpPtrAllocator A;
  BumpPtrStringSaver Saver(A);
  ExpandResponseFiles(Saver, TokenizeGNUCommandLine, newArgv);
  argv = &newArgv[0];
  argc = static_cast<int>(newArgv.size());

  // Copy the program name into ProgName, making sure not to overflow it.
  ProgramName = sys::path::filename(argv[0]);

  ProgramOverview = Overview;
  bool ErrorParsing = false;

  // Check out the positional arguments to collect information about them.
  unsigned NumPositionalRequired = 0;

  // Determine whether or not there are an unlimited number of positionals
  bool HasUnlimitedPositionals = false;

  if (ConsumeAfterOpt) {
    assert(PositionalOpts.size() > 0 &&
           "Cannot specify cl::ConsumeAfter without a positional argument!");
  }
  if (!PositionalOpts.empty()) {

    // Calculate how many positional values are _required_.
    bool UnboundedFound = false;
    for (size_t i = 0, e = PositionalOpts.size(); i != e; ++i) {
      Option *Opt = PositionalOpts[i];
      if (RequiresValue(Opt))
        ++NumPositionalRequired;
      else if (ConsumeAfterOpt) {
        // ConsumeAfter cannot be combined with "optional" positional options
        // unless there is only one positional argument...
        if (PositionalOpts.size() > 1)
          ErrorParsing |= Opt->error(
              "error - this positional option will never be matched, "
              "because it does not Require a value, and a "
              "cl::ConsumeAfter option is active!");
      } else if (UnboundedFound && !Opt->hasArgStr()) {
        // Print entire command line into error output
        std::string CmdLine;
        for (int argi = 0; argi < argc; ++argi) {
          if (argv[argi] != nullptr) {
            if (!CmdLine.empty())
              CmdLine += ' ';
            CmdLine += argv[argi];
          }
        }

        // This option does not "require" a value...  Make sure this option is
        // not specified after an option that eats all extra arguments, or this
        // one will never get any!
        //
        ErrorParsing |= Opt->error("error - option can never match, because "
                                   "another positional argument will match an "
                                   "unbounded number of values, and this option"
                                   " does not require a value!");
        errs() << ProgramName << ": CommandLine Error: Command line '"
               << CmdLine << "' is all messed up!\n";
        errs() << PositionalOpts.size();
      }
      UnboundedFound |= EatsUnboundedNumberOfValues(Opt);
    }
    HasUnlimitedPositionals = UnboundedFound || ConsumeAfterOpt;
  }

  // PositionalVals - A vector of "positional" arguments we accumulate into
  // the process at the end.
  //
  SmallVector<std::pair<StringRef, unsigned>, 4> PositionalVals;

  // If the program has named positional arguments, and the name has been run
  // across, keep track of which positional argument was named.  Otherwise put
  // the positional args into the PositionalVals list...
  Option *ActivePositionalArg = nullptr;

  // Loop over all of the arguments... processing them.
  bool DashDashFound = false; // Have we read '--'?
  for (int i = 1; i < argc; ++i) {
    Option *Handler = nullptr;
    Option *NearestHandler = nullptr;
    std::string NearestHandlerString;
    StringRef Value;
    StringRef ArgName = "";

    // Check to see if this is a positional argument.  This argument is
    // considered to be positional if it doesn't start with '-', if it is "-"
    // itself, or if we have seen "--" already.
    //
    if (argv[i][0] != '-' || argv[i][1] == 0 || DashDashFound) {
      // Positional argument!
      if (ActivePositionalArg) {
        ProvidePositionalOption(ActivePositionalArg, argv[i], i);
        continue; // We are done!
      }

      if (!PositionalOpts.empty()) {
        PositionalVals.push_back(std::make_pair(argv[i], i));

        // All of the positional arguments have been fulfulled, give the rest to
        // the consume after option... if it's specified...
        //
        if (PositionalVals.size() >= NumPositionalRequired && ConsumeAfterOpt) {
          for (++i; i < argc; ++i)
            PositionalVals.push_back(std::make_pair(argv[i], i));
          break; // Handle outside of the argument processing loop...
        }

        // Delay processing positional arguments until the end...
        continue;
      }
    } else if (argv[i][0] == '-' && argv[i][1] == '-' && argv[i][2] == 0 &&
               !DashDashFound) {
      DashDashFound = true; // This is the mythical "--"?
      continue;             // Don't try to process it as an argument itself.
    } else if (ActivePositionalArg &&
               (ActivePositionalArg->getMiscFlags() & PositionalEatsArgs)) {
      // If there is a positional argument eating options, check to see if this
      // option is another positional argument.  If so, treat it as an argument,
      // otherwise feed it to the eating positional.
      ArgName = argv[i] + 1;
      // Eat leading dashes.
      while (!ArgName.empty() && ArgName[0] == '-')
        ArgName = ArgName.substr(1);

      Handler = LookupOption(ArgName, Value);
      if (!Handler || Handler->getFormattingFlag() != cl::Positional) {
        ProvidePositionalOption(ActivePositionalArg, argv[i], i);
        continue; // We are done!
      }

    } else { // We start with a '-', must be an argument.
      ArgName = argv[i] + 1;
      // Eat leading dashes.
      while (!ArgName.empty() && ArgName[0] == '-')
        ArgName = ArgName.substr(1);

      Handler = LookupOption(ArgName, Value);

      // Check to see if this "option" is really a prefixed or grouped argument.
      if (!Handler)
        Handler = HandlePrefixedOrGroupedOption(ArgName, Value, ErrorParsing,
                                                OptionsMap);

      // Otherwise, look for the closest available option to report to the user
      // in the upcoming error.
      if (!Handler && SinkOpts.empty())
        NearestHandler =
            LookupNearestOption(ArgName, OptionsMap, NearestHandlerString);
    }

    if (!Handler) {
      if (SinkOpts.empty()) {
        errs() << ProgramName << ": Unknown command line argument '" << argv[i]
               << "'.  Try: '" << argv[0] << " -help'\n";

        if (NearestHandler) {
          // If we know a near match, report it as well.
          errs() << ProgramName << ": Did you mean '-" << NearestHandlerString
                 << "'?\n";
        }

        ErrorParsing = true;
      } else {
        for (SmallVectorImpl<Option *>::iterator I = SinkOpts.begin(),
                                                 E = SinkOpts.end();
             I != E; ++I)
          (*I)->addOccurrence(i, "", argv[i]);
      }
      continue;
    }

    // If this is a named positional argument, just remember that it is the
    // active one...
    if (Handler->getFormattingFlag() == cl::Positional)
      ActivePositionalArg = Handler;
    else
      ErrorParsing |= ProvideOption(Handler, ArgName, Value, argc, argv, i);
  }

  // Check and handle positional arguments now...
  if (NumPositionalRequired > PositionalVals.size()) {
    errs() << ProgramName
           << ": Not enough positional command line arguments specified!\n"
           << "Must specify at least " << NumPositionalRequired
           << " positional arguments: See: " << argv[0] << " -help\n";

    ErrorParsing = true;
  } else if (!HasUnlimitedPositionals &&
             PositionalVals.size() > PositionalOpts.size()) {
    errs() << ProgramName << ": Too many positional arguments specified!\n"
           << "Can specify at most " << PositionalOpts.size()
           << " positional arguments: See: " << argv[0] << " -help\n";
    ErrorParsing = true;

  } else if (!ConsumeAfterOpt) {
    // Positional args have already been handled if ConsumeAfter is specified.
    unsigned ValNo = 0, NumVals = static_cast<unsigned>(PositionalVals.size());
    for (size_t i = 0, e = PositionalOpts.size(); i != e; ++i) {
      if (RequiresValue(PositionalOpts[i])) {
        ProvidePositionalOption(PositionalOpts[i], PositionalVals[ValNo].first,
                                PositionalVals[ValNo].second);
        ValNo++;
        --NumPositionalRequired; // We fulfilled our duty...
      }

      // If we _can_ give this option more arguments, do so now, as long as we
      // do not give it values that others need.  'Done' controls whether the
      // option even _WANTS_ any more.
      //
      bool Done = PositionalOpts[i]->getNumOccurrencesFlag() == cl::Required;
      while (NumVals - ValNo > NumPositionalRequired && !Done) {
        switch (PositionalOpts[i]->getNumOccurrencesFlag()) {
        case cl::Optional:
          Done = true; // Optional arguments want _at most_ one value
        // FALL THROUGH
        case cl::ZeroOrMore: // Zero or more will take all they can get...
        case cl::OneOrMore:  // One or more will take all they can get...
          ProvidePositionalOption(PositionalOpts[i],
                                  PositionalVals[ValNo].first,
                                  PositionalVals[ValNo].second);
          ValNo++;
          break;
        default:
          llvm_unreachable("Internal error, unexpected NumOccurrences flag in "
                           "positional argument processing!");
        }
      }
    }
  } else {
    assert(ConsumeAfterOpt && NumPositionalRequired <= PositionalVals.size());
    unsigned ValNo = 0;
    for (size_t j = 1, e = PositionalOpts.size(); j != e; ++j)
      if (RequiresValue(PositionalOpts[j])) {
        ErrorParsing |= ProvidePositionalOption(PositionalOpts[j],
                                                PositionalVals[ValNo].first,
                                                PositionalVals[ValNo].second);
        ValNo++;
      }

    // Handle the case where there is just one positional option, and it's
    // optional.  In this case, we want to give JUST THE FIRST option to the
    // positional option and keep the rest for the consume after.  The above
    // loop would have assigned no values to positional options in this case.
    //
    if (PositionalOpts.size() == 1 && ValNo == 0 && !PositionalVals.empty()) {
      ErrorParsing |= ProvidePositionalOption(PositionalOpts[0],
                                              PositionalVals[ValNo].first,
                                              PositionalVals[ValNo].second);
      ValNo++;
    }

    // Handle over all of the rest of the arguments to the
    // cl::ConsumeAfter command line option...
    for (; ValNo != PositionalVals.size(); ++ValNo)
      ErrorParsing |=
          ProvidePositionalOption(ConsumeAfterOpt, PositionalVals[ValNo].first,
                                  PositionalVals[ValNo].second);
  }

  // Loop over args and make sure all required args are specified!
  for (const auto &Opt : OptionsMap) {
    switch (Opt.second->getNumOccurrencesFlag()) {
    case Required:
    case OneOrMore:
      if (Opt.second->getNumOccurrences() == 0) {
        Opt.second->error("must be specified at least once!");
        ErrorParsing = true;
      }
    // Fall through
    default:
      break;
    }
  }

  // Now that we know if -debug is specified, we can use it.
  // Note that if ReadResponseFiles == true, this must be done before the
  // memory allocated for the expanded command line is free()d below.
  DEBUG(dbgs() << "Args: ";
        for (int i = 0; i < argc; ++i) dbgs() << argv[i] << ' ';
        dbgs() << '\n';);

  // Free all of the memory allocated to the map.  Command line options may only
  // be processed once!
  MoreHelp.clear();

  // If we had an error processing our arguments, don't let the program execute
  if (ErrorParsing)
    exit(1);
}

//===----------------------------------------------------------------------===//
// Option Base class implementation
//

bool Option::error(const Twine &Message, StringRef ArgName) {
  if (!ArgName.data())
    ArgName = ArgStr;
  if (ArgName.empty())
    errs() << HelpStr; // Be nice for positional arguments
  else
    errs() << GlobalParser->ProgramName << ": for the -" << ArgName;

  errs() << " option: " << Message << "\n";
  return true;
}

bool Option::addOccurrence(unsigned pos, StringRef ArgName, StringRef Value,
                           bool MultiArg) {
  if (!MultiArg)
    NumOccurrences++; // Increment the number of times we have been seen

  switch (getNumOccurrencesFlag()) {
  case Optional:
    if (NumOccurrences > 1)
      return error("may only occur zero or one times!", ArgName);
    break;
  case Required:
    if (NumOccurrences > 1)
      return error("must occur exactly one time!", ArgName);
  // Fall through
  case OneOrMore:
  case ZeroOrMore:
  case ConsumeAfter:
    break;
  }

  return handleOccurrence(pos, ArgName, Value);
}

// getValueStr - Get the value description string, using "DefaultMsg" if nothing
// has been specified yet.
//
static const char *getValueStr(const Option &O, const char *DefaultMsg) {
  if (O.ValueStr[0] == 0)
    return DefaultMsg;
  return O.ValueStr;
}

//===----------------------------------------------------------------------===//
// cl::alias class implementation
//

// Return the width of the option tag for printing...
size_t alias::getOptionWidth() const { return std::strlen(ArgStr) + 6; }

static void printHelpStr(StringRef HelpStr, size_t Indent,
                         size_t FirstLineIndentedBy) {
  std::pair<StringRef, StringRef> Split = HelpStr.split('\n');
  outs().indent(Indent - FirstLineIndentedBy) << " - " << Split.first << "\n";
  while (!Split.second.empty()) {
    Split = Split.second.split('\n');
    outs().indent(Indent) << Split.first << "\n";
  }
}

// Print out the option for the alias.
void alias::printOptionInfo(size_t GlobalWidth) const {
  outs() << "  -" << ArgStr;
  printHelpStr(HelpStr, GlobalWidth, std::strlen(ArgStr) + 6);
}

//===----------------------------------------------------------------------===//
// Parser Implementation code...
//

// basic_parser implementation
//

// Return the width of the option tag for printing...
size_t basic_parser_impl::getOptionWidth(const Option &O) const {
  size_t Len = std::strlen(O.ArgStr);
  if (const char *ValName = getValueName())
    Len += std::strlen(getValueStr(O, ValName)) + 3;

  return Len + 6;
}

// printOptionInfo - Print out information about this option.  The
// to-be-maintained width is specified.
//
void basic_parser_impl::printOptionInfo(const Option &O,
                                        size_t GlobalWidth) const {
  outs() << "  -" << O.ArgStr;

  if (const char *ValName = getValueName())
    outs() << "=<" << getValueStr(O, ValName) << '>';

  printHelpStr(O.HelpStr, GlobalWidth, getOptionWidth(O));
}

void basic_parser_impl::printOptionName(const Option &O,
                                        size_t GlobalWidth) const {
  outs() << "  -" << O.ArgStr;
  outs().indent(GlobalWidth - std::strlen(O.ArgStr));
}

// parser<bool> implementation
//
bool parser<bool>::parse(Option &O, StringRef ArgName, StringRef Arg,
                         bool &Value) {
  if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" ||
      Arg == "1") {
    Value = true;
    return false;
  }

  if (Arg == "false" || Arg == "FALSE" || Arg == "False" || Arg == "0") {
    Value = false;
    return false;
  }
  return O.error("'" + Arg +
                 "' is invalid value for boolean argument! Try 0 or 1");
}

// parser<boolOrDefault> implementation
//
bool parser<boolOrDefault>::parse(Option &O, StringRef ArgName, StringRef Arg,
                                  boolOrDefault &Value) {
  if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" ||
      Arg == "1") {
    Value = BOU_TRUE;
    return false;
  }
  if (Arg == "false" || Arg == "FALSE" || Arg == "False" || Arg == "0") {
    Value = BOU_FALSE;
    return false;
  }

  return O.error("'" + Arg +
                 "' is invalid value for boolean argument! Try 0 or 1");
}

// parser<int> implementation
//
bool parser<int>::parse(Option &O, StringRef ArgName, StringRef Arg,
                        int &Value) {
  if (Arg.getAsInteger(0, Value))
    return O.error("'" + Arg + "' value invalid for integer argument!");
  return false;
}

// parser<unsigned> implementation
//
bool parser<unsigned>::parse(Option &O, StringRef ArgName, StringRef Arg,
                             unsigned &Value) {

  if (Arg.getAsInteger(0, Value))
    return O.error("'" + Arg + "' value invalid for uint argument!");
  return false;
}

// parser<unsigned long long> implementation
//
bool parser<unsigned long long>::parse(Option &O, StringRef ArgName,
                                       StringRef Arg,
                                       unsigned long long &Value) {

  if (Arg.getAsInteger(0, Value))
    return O.error("'" + Arg + "' value invalid for uint argument!");
  return false;
}

// parser<double>/parser<float> implementation
//
static bool parseDouble(Option &O, StringRef Arg, double &Value) {
  SmallString<32> TmpStr(Arg.begin(), Arg.end());
  const char *ArgStart = TmpStr.c_str();
  char *End;
  Value = strtod(ArgStart, &End);
  if (*End != 0)
    return O.error("'" + Arg + "' value invalid for floating point argument!");
  return false;
}

bool parser<double>::parse(Option &O, StringRef ArgName, StringRef Arg,
                           double &Val) {
  return parseDouble(O, Arg, Val);
}

bool parser<float>::parse(Option &O, StringRef ArgName, StringRef Arg,
                          float &Val) {
  double dVal;
  if (parseDouble(O, Arg, dVal))
    return true;
  Val = (float)dVal;
  return false;
}

// generic_parser_base implementation
//

// findOption - Return the option number corresponding to the specified
// argument string.  If the option is not found, getNumOptions() is returned.
//
unsigned generic_parser_base::findOption(const char *Name) {
  unsigned e = getNumOptions();

  for (unsigned i = 0; i != e; ++i) {
    if (strcmp(getOption(i), Name) == 0)
      return i;
  }
  return e;
}

// Return the width of the option tag for printing...
size_t generic_parser_base::getOptionWidth(const Option &O) const {
  if (O.hasArgStr()) {
    size_t Size = std::strlen(O.ArgStr) + 6;
    for (unsigned i = 0, e = getNumOptions(); i != e; ++i)
      Size = std::max(Size, std::strlen(getOption(i)) + 8);
    return Size;
  } else {
    size_t BaseSize = 0;
    for (unsigned i = 0, e = getNumOptions(); i != e; ++i)
      BaseSize = std::max(BaseSize, std::strlen(getOption(i)) + 8);
    return BaseSize;
  }
}

// printOptionInfo - Print out information about this option.  The
// to-be-maintained width is specified.
//
void generic_parser_base::printOptionInfo(const Option &O,
                                          size_t GlobalWidth) const {
  if (O.hasArgStr()) {
    outs() << "  -" << O.ArgStr;
    printHelpStr(O.HelpStr, GlobalWidth, std::strlen(O.ArgStr) + 6);

    for (unsigned i = 0, e = getNumOptions(); i != e; ++i) {
      size_t NumSpaces = GlobalWidth - strlen(getOption(i)) - 8;
      outs() << "    =" << getOption(i);
      outs().indent(NumSpaces) << " -   " << getDescription(i) << '\n';
    }
  } else {
    if (O.HelpStr[0])
      outs() << "  " << O.HelpStr << '\n';
    for (unsigned i = 0, e = getNumOptions(); i != e; ++i) {
      const char *Option = getOption(i);
      outs() << "    -" << Option;
      printHelpStr(getDescription(i), GlobalWidth, std::strlen(Option) + 8);
    }
  }
}

static const size_t MaxOptWidth = 8; // arbitrary spacing for printOptionDiff

// printGenericOptionDiff - Print the value of this option and it's default.
//
// "Generic" options have each value mapped to a name.
void generic_parser_base::printGenericOptionDiff(
    const Option &O, const GenericOptionValue &Value,
    const GenericOptionValue &Default, size_t GlobalWidth) const {
  outs() << "  -" << O.ArgStr;
  outs().indent(GlobalWidth - std::strlen(O.ArgStr));

  unsigned NumOpts = getNumOptions();
  for (unsigned i = 0; i != NumOpts; ++i) {
    if (Value.compare(getOptionValue(i)))
      continue;

    outs() << "= " << getOption(i);
    size_t L = std::strlen(getOption(i));
    size_t NumSpaces = MaxOptWidth > L ? MaxOptWidth - L : 0;
    outs().indent(NumSpaces) << " (default: ";
    for (unsigned j = 0; j != NumOpts; ++j) {
      if (Default.compare(getOptionValue(j)))
        continue;
      outs() << getOption(j);
      break;
    }
    outs() << ")\n";
    return;
  }
  outs() << "= *unknown option value*\n";
}

// printOptionDiff - Specializations for printing basic value types.
//
#define PRINT_OPT_DIFF(T)                                                      \
  void parser<T>::printOptionDiff(const Option &O, T V, OptionValue<T> D,      \
                                  size_t GlobalWidth) const {                  \
    printOptionName(O, GlobalWidth);                                           \
    std::string Str;                                                           \
    {                                                                          \
      raw_string_ostream SS(Str);                                              \
      SS << V;                                                                 \
    }                                                                          \
    outs() << "= " << Str;                                                     \
    size_t NumSpaces =                                                         \
        MaxOptWidth > Str.size() ? MaxOptWidth - Str.size() : 0;               \
    outs().indent(NumSpaces) << " (default: ";                                 \
    if (D.hasValue())                                                          \
      outs() << D.getValue();                                                  \
    else                                                                       \
      outs() << "*no default*";                                                \
    outs() << ")\n";                                                           \
  }

PRINT_OPT_DIFF(bool)
PRINT_OPT_DIFF(boolOrDefault)
PRINT_OPT_DIFF(int)
PRINT_OPT_DIFF(unsigned)
PRINT_OPT_DIFF(unsigned long long)
PRINT_OPT_DIFF(double)
PRINT_OPT_DIFF(float)
PRINT_OPT_DIFF(char)

void parser<std::string>::printOptionDiff(const Option &O, StringRef V,
                                          OptionValue<std::string> D,
                                          size_t GlobalWidth) const {
  printOptionName(O, GlobalWidth);
  outs() << "= " << V;
  size_t NumSpaces = MaxOptWidth > V.size() ? MaxOptWidth - V.size() : 0;
  outs().indent(NumSpaces) << " (default: ";
  if (D.hasValue())
    outs() << D.getValue();
  else
    outs() << "*no default*";
  outs() << ")\n";
}

// Print a placeholder for options that don't yet support printOptionDiff().
void basic_parser_impl::printOptionNoValue(const Option &O,
                                           size_t GlobalWidth) const {
  printOptionName(O, GlobalWidth);
  outs() << "= *cannot print option value*\n";
}

//===----------------------------------------------------------------------===//
// -help and -help-hidden option implementation
//

// HLSL Change - __cdecl
static int __cdecl OptNameCompare(const std::pair<const char *, Option *> *LHS,
                          const std::pair<const char *, Option *> *RHS) {
  return strcmp(LHS->first, RHS->first);
}

// Copy Options into a vector so we can sort them as we like.
static void sortOpts(StringMap<Option *> &OptMap,
                     SmallVectorImpl<std::pair<const char *, Option *>> &Opts,
                     bool ShowHidden) {
  SmallPtrSet<Option *, 128> OptionSet; // Duplicate option detection.

  for (StringMap<Option *>::iterator I = OptMap.begin(), E = OptMap.end();
       I != E; ++I) {
    // Ignore really-hidden options.
    if (I->second->getOptionHiddenFlag() == ReallyHidden)
      continue;

    // Unless showhidden is set, ignore hidden flags.
    if (I->second->getOptionHiddenFlag() == Hidden && !ShowHidden)
      continue;

    // If we've already seen this option, don't add it to the list again.
    if (!OptionSet.insert(I->second).second)
      continue;

    Opts.push_back(
        std::pair<const char *, Option *>(I->getKey().data(), I->second));
  }

  // Sort the options list alphabetically.
  array_pod_sort(Opts.begin(), Opts.end(), OptNameCompare);
}

namespace {

class HelpPrinter {
protected:
  const bool ShowHidden;
  typedef SmallVector<std::pair<const char *, Option *>, 128>
      StrOptionPairVector;
  // Print the options. Opts is assumed to be alphabetically sorted.
  virtual void printOptions(StrOptionPairVector &Opts, size_t MaxArgLen) {
    for (size_t i = 0, e = Opts.size(); i != e; ++i)
      Opts[i].second->printOptionInfo(MaxArgLen);
  }

public:
  explicit HelpPrinter(bool showHidden) : ShowHidden(showHidden) {}
  virtual ~HelpPrinter() {}

  // Invoke the printer.
  void operator=(bool Value) {
    if (!Value)
      return;

    StrOptionPairVector Opts;
    sortOpts(GlobalParser->OptionsMap, Opts, ShowHidden);

    if (GlobalParser->ProgramOverview)
      outs() << "OVERVIEW: " << GlobalParser->ProgramOverview << "\n";

    outs() << "USAGE: " << GlobalParser->ProgramName << " [options]";

    for (auto Opt : GlobalParser->PositionalOpts) {
      if (Opt->ArgStr[0])
        outs() << " --" << Opt->ArgStr;
      outs() << " " << Opt->HelpStr;
    }

    // Print the consume after option info if it exists...
    if (GlobalParser->ConsumeAfterOpt)
      outs() << " " << GlobalParser->ConsumeAfterOpt->HelpStr;

    outs() << "\n\n";

    // Compute the maximum argument length...
    size_t MaxArgLen = 0;
    for (size_t i = 0, e = Opts.size(); i != e; ++i)
      MaxArgLen = std::max(MaxArgLen, Opts[i].second->getOptionWidth());

    outs() << "OPTIONS:\n";
    printOptions(Opts, MaxArgLen);

    // Print any extra help the user has declared.
    for (auto I : GlobalParser->MoreHelp)
      outs() << I;
    GlobalParser->MoreHelp.clear();

    // Halt the program since help information was printed
    exit(0);
  }
};

class CategorizedHelpPrinter : public HelpPrinter {
public:
  explicit CategorizedHelpPrinter(bool showHidden) : HelpPrinter(showHidden) {}

  // Helper function for printOptions().
  // It shall return a negative value if A's name should be lexicographically
  // ordered before B's name. It returns a value greater equal zero otherwise.
  static int __cdecl OptionCategoryCompare(OptionCategory *const *A,
                                   OptionCategory *const *B) {
    return strcmp((*A)->getName(), (*B)->getName());
  }

  // Make sure we inherit our base class's operator=()
  using HelpPrinter::operator=;

protected:
  void printOptions(StrOptionPairVector &Opts, size_t MaxArgLen) override {
    std::vector<OptionCategory *> SortedCategories;
    std::map<OptionCategory *, std::vector<Option *>> CategorizedOptions;

    // Collect registered option categories into vector in preparation for
    // sorting.
    for (auto I = GlobalParser->RegisteredOptionCategories.begin(),
              E = GlobalParser->RegisteredOptionCategories.end();
         I != E; ++I) {
      SortedCategories.push_back(*I);
    }

    // Sort the different option categories alphabetically.
    assert(SortedCategories.size() > 0 && "No option categories registered!");
    array_pod_sort(SortedCategories.begin(), SortedCategories.end(),
                   OptionCategoryCompare);

    // Create map to empty vectors.
    for (std::vector<OptionCategory *>::const_iterator
             I = SortedCategories.begin(),
             E = SortedCategories.end();
         I != E; ++I)
      CategorizedOptions[*I] = std::vector<Option *>();

    // Walk through pre-sorted options and assign into categories.
    // Because the options are already alphabetically sorted the
    // options within categories will also be alphabetically sorted.
    for (size_t I = 0, E = Opts.size(); I != E; ++I) {
      Option *Opt = Opts[I].second;
      assert(CategorizedOptions.count(Opt->Category) > 0 &&
             "Option has an unregistered category");
      CategorizedOptions[Opt->Category].push_back(Opt);
    }

    // Now do printing.
    for (std::vector<OptionCategory *>::const_iterator
             Category = SortedCategories.begin(),
             E = SortedCategories.end();
         Category != E; ++Category) {
      // Hide empty categories for -help, but show for -help-hidden.
      bool IsEmptyCategory = CategorizedOptions[*Category].size() == 0;
      if (!ShowHidden && IsEmptyCategory)
        continue;

      // Print category information.
      outs() << "\n";
      outs() << (*Category)->getName() << ":\n";

      // Check if description is set.
      if ((*Category)->getDescription() != nullptr)
        outs() << (*Category)->getDescription() << "\n\n";
      else
        outs() << "\n";

      // When using -help-hidden explicitly state if the category has no
      // options associated with it.
      if (IsEmptyCategory) {
        outs() << "  This option category has no options.\n";
        continue;
      }
      // Loop over the options in the category and print.
      for (std::vector<Option *>::const_iterator
               Opt = CategorizedOptions[*Category].begin(),
               E = CategorizedOptions[*Category].end();
           Opt != E; ++Opt)
        (*Opt)->printOptionInfo(MaxArgLen);
    }
  }
};

// This wraps the Uncategorizing and Categorizing printers and decides
// at run time which should be invoked.
class HelpPrinterWrapper {
private:
#if 0 // HLSL Change Starts
  HelpPrinter &UncategorizedPrinter;
  CategorizedHelpPrinter &CategorizedPrinter;
#endif // HLSL Change Ends
public:
  explicit HelpPrinterWrapper(HelpPrinter &UncategorizedPrinter,
                              CategorizedHelpPrinter &CategorizedPrinter)
#if 0 // HLSL Change Starts
      : UncategorizedPrinter(UncategorizedPrinter),
        CategorizedPrinter(CategorizedPrinter)
#endif // HLSL Change Starts
  {}

  // Invoke the printer.
  void operator=(bool Value);
};

} // End anonymous namespace

// Declare the four HelpPrinter instances that are used to print out help, or
// help-hidden as an uncategorized list or in categories.
static HelpPrinter UncategorizedNormalPrinter(false);
static HelpPrinter UncategorizedHiddenPrinter(true);
static CategorizedHelpPrinter CategorizedNormalPrinter(false);
static CategorizedHelpPrinter CategorizedHiddenPrinter(true);

// Declare HelpPrinter wrappers that will decide whether or not to invoke
// a categorizing help printer
static HelpPrinterWrapper WrappedNormalPrinter(UncategorizedNormalPrinter,
                                               CategorizedNormalPrinter);
static HelpPrinterWrapper WrappedHiddenPrinter(UncategorizedHiddenPrinter,
                                               CategorizedHiddenPrinter);

#if 0 // HLSL Change Starts

// Define a category for generic options that all tools should have.
static cl::OptionCategory GenericCategory("Generic Options");

// Define uncategorized help printers.
// -help-list is hidden by default because if Option categories are being used
// then -help behaves the same as -help-list.
static cl::opt<HelpPrinter, true, parser<bool>> HLOp(
    "help-list",
    cl::desc("Display list of available options (-help-list-hidden for more)"),
    cl::location(UncategorizedNormalPrinter), cl::Hidden, cl::ValueDisallowed,
    cl::cat(GenericCategory));

static cl::opt<HelpPrinter, true, parser<bool>>
    HLHOp("help-list-hidden", cl::desc("Display list of all available options"),
          cl::location(UncategorizedHiddenPrinter), cl::Hidden,
          cl::ValueDisallowed, cl::cat(GenericCategory));

// Define uncategorized/categorized help printers. These printers change their
// behaviour at runtime depending on whether one or more Option categories have
// been declared.
static cl::opt<HelpPrinterWrapper, true, parser<bool>>
    HOp("help", cl::desc("Display available options (-help-hidden for more)"),
        cl::location(WrappedNormalPrinter), cl::ValueDisallowed,
        cl::cat(GenericCategory));

static cl::opt<HelpPrinterWrapper, true, parser<bool>>
    HHOp("help-hidden", cl::desc("Display all available options"),
         cl::location(WrappedHiddenPrinter), cl::Hidden, cl::ValueDisallowed,
         cl::cat(GenericCategory));

static cl::opt<bool> PrintOptions(
    "print-options",
    cl::desc("Print non-default options after command line parsing"),
    cl::Hidden, cl::init(false), cl::cat(GenericCategory));

static cl::opt<bool> PrintAllOptions(
    "print-all-options",
    cl::desc("Print all option values after command line parsing"), cl::Hidden,
    cl::init(false), cl::cat(GenericCategory));

void HelpPrinterWrapper::operator=(bool Value) {
  if (!Value)
    return;

  // Decide which printer to invoke. If more than one option category is
  // registered then it is useful to show the categorized help instead of
  // uncategorized help.
  if (GlobalParser->RegisteredOptionCategories.size() > 1) {
    // unhide -help-list option so user can have uncategorized output if they
    // want it.
    HLOp.setHiddenFlag(NotHidden);

    CategorizedPrinter = true; // Invoke categorized printer
  } else
    UncategorizedPrinter = true; // Invoke uncategorized printer
}

#else
static const bool PrintOptions = false;
static const bool PrintAllOptions = false;

#endif // HLSL Change Ends

// Print the value of each option.
void cl::PrintOptionValues() { GlobalParser->printOptionValues(); }

void CommandLineParser::printOptionValues() {
  if (!PrintOptions && !PrintAllOptions)
    return;

  SmallVector<std::pair<const char *, Option *>, 128> Opts;
  sortOpts(OptionsMap, Opts, /*ShowHidden*/ true);

  // Compute the maximum argument length...
  size_t MaxArgLen = 0;
  for (size_t i = 0, e = Opts.size(); i != e; ++i)
    MaxArgLen = std::max(MaxArgLen, Opts[i].second->getOptionWidth());

  for (size_t i = 0, e = Opts.size(); i != e; ++i)
    Opts[i].second->printOptionValue(MaxArgLen, PrintAllOptions);
}

static void (*OverrideVersionPrinter)() = nullptr;

static std::vector<void (*)()> *ExtraVersionPrinters = nullptr;

namespace {
class VersionPrinter {
public:
  void print() {
    raw_ostream &OS = outs();
    OS << "LLVM (http://llvm.org/):\n"
       << "  " << PACKAGE_NAME << " version " << PACKAGE_VERSION;
#ifdef LLVM_VERSION_INFO
    OS << " " << LLVM_VERSION_INFO;
#endif
    OS << "\n  ";
#ifndef __OPTIMIZE__
    OS << "DEBUG build";
#else
    OS << "Optimized build";
#endif
#ifndef NDEBUG
    OS << " with assertions";
#endif
    std::string CPU = sys::getHostCPUName();
    if (CPU == "generic")
      CPU = "(unknown)";
    OS << ".\n"
#if (ENABLE_TIMESTAMPS == 1)
// HLSL Change - remove any possibility that __DATE__ or __TIME__ are used (build cache)
//       << "  Built " << __DATE__ << " (" << __TIME__ << ").\n"
#endif
       << "  Default target: " << sys::getDefaultTargetTriple() << '\n'
       << "  Host CPU: " << CPU << '\n';
  }
  void operator=(bool OptionWasSpecified) {
    if (!OptionWasSpecified)
      return;

    if (OverrideVersionPrinter != nullptr) {
      (*OverrideVersionPrinter)();
      exit(0);
    }
    print();

    // Iterate over any registered extra printers and call them to add further
    // information.
    if (ExtraVersionPrinters != nullptr) {
      outs() << '\n';
      for (std::vector<void (*)()>::iterator I = ExtraVersionPrinters->begin(),
                                             E = ExtraVersionPrinters->end();
           I != E; ++I)
        (*I)();
    }

    exit(0);
  }
};
} // End anonymous namespace

// Define the --version option that prints out the LLVM version for the tool
static VersionPrinter VersionPrinterInstance;

#if 0 // HLSL Change Starts
static cl::opt<VersionPrinter, true, parser<bool>>
    VersOp("version", cl::desc("Display the version of this program"),
           cl::location(VersionPrinterInstance), cl::ValueDisallowed,
           cl::cat(GenericCategory));
#else
static const OptionCategory *GenericCategory;
#endif // HLSL Change Ends

// Utility function for printing the help message.
void cl::PrintHelpMessage(bool Hidden, bool Categorized) {
  // This looks weird, but it actually prints the help message. The Printers are
  // types of HelpPrinter and the help gets printed when its operator= is
  // invoked. That's because the "normal" usages of the help printer is to be
  // assigned true/false depending on whether -help or -help-hidden was given or
  // not.  Since we're circumventing that we have to make it look like -help or
  // -help-hidden were given, so we assign true.

  if (!Hidden && !Categorized)
    UncategorizedNormalPrinter = true;
  else if (!Hidden && Categorized)
    CategorizedNormalPrinter = true;
  else if (Hidden && !Categorized)
    UncategorizedHiddenPrinter = true;
  else
    CategorizedHiddenPrinter = true;
}

/// Utility function for printing version number.
void cl::PrintVersionMessage() { VersionPrinterInstance.print(); }

void cl::SetVersionPrinter(void (*func)()) { OverrideVersionPrinter = func; }

void cl::AddExtraVersionPrinter(void (*func)()) {
  if (!ExtraVersionPrinters)
    ExtraVersionPrinters = new std::vector<void (*)()>;

  ExtraVersionPrinters->push_back(func);
}

StringMap<Option *> &cl::getRegisteredOptions() {
  return GlobalParser->OptionsMap;
}

void cl::HideUnrelatedOptions(cl::OptionCategory &Category) {
  for (auto &I : GlobalParser->OptionsMap) {
    if (I.second->Category != &Category &&
        I.second->Category != GenericCategory) // HLSL Change - use pointer
      I.second->setHiddenFlag(cl::ReallyHidden);
  }
}

void cl::HideUnrelatedOptions(ArrayRef<const cl::OptionCategory *> Categories) {
  auto CategoriesBegin = Categories.begin();
  auto CategoriesEnd = Categories.end();
  for (auto &I : GlobalParser->OptionsMap) {
    if (std::find(CategoriesBegin, CategoriesEnd, I.second->Category) ==
            CategoriesEnd &&
        I.second->Category != GenericCategory) // HLSL Change - use pointer
      I.second->setHiddenFlag(cl::ReallyHidden);
  }
}

void LLVMParseCommandLineOptions(int argc, const char *const *argv,
                                 const char *Overview) {
  llvm::cl::ParseCommandLineOptions(argc, argv, Overview);
}
