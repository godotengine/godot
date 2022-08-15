//===- SymbolRewriter.cpp - Symbol Rewriter ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SymbolRewriter is a LLVM pass which can rewrite symbols transparently within
// existing code.  It is implemented as a compiler pass and is configured via a
// YAML configuration file.
//
// The YAML configuration file format is as follows:
//
// RewriteMapFile := RewriteDescriptors
// RewriteDescriptors := RewriteDescriptor | RewriteDescriptors
// RewriteDescriptor := RewriteDescriptorType ':' '{' RewriteDescriptorFields '}'
// RewriteDescriptorFields := RewriteDescriptorField | RewriteDescriptorFields
// RewriteDescriptorField := FieldIdentifier ':' FieldValue ','
// RewriteDescriptorType := Identifier
// FieldIdentifier := Identifier
// FieldValue := Identifier
// Identifier := [0-9a-zA-Z]+
//
// Currently, the following descriptor types are supported:
//
// - function:          (function rewriting)
//      + Source        (original name of the function)
//      + Target        (explicit transformation)
//      + Transform     (pattern transformation)
//      + Naked         (boolean, whether the function is undecorated)
// - global variable:   (external linkage global variable rewriting)
//      + Source        (original name of externally visible variable)
//      + Target        (explicit transformation)
//      + Transform     (pattern transformation)
// - global alias:      (global alias rewriting)
//      + Source        (original name of the aliased name)
//      + Target        (explicit transformation)
//      + Transform     (pattern transformation)
//
// Note that source and exactly one of [Target, Transform] must be provided
//
// New rewrite descriptors can be created.  Addding a new rewrite descriptor
// involves:
//
//  a) extended the rewrite descriptor kind enumeration
//     (<anonymous>::RewriteDescriptor::RewriteDescriptorType)
//  b) implementing the new descriptor
//     (c.f. <anonymous>::ExplicitRewriteFunctionDescriptor)
//  c) extending the rewrite map parser
//     (<anonymous>::RewriteMapParser::parseEntry)
//
//  Specify to rewrite the symbols using the `-rewrite-symbols` option, and
//  specify the map file to use for the rewriting via the `-rewrite-map-file`
//  option.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "symbol-rewriter"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/SymbolRewriter.h"

using namespace llvm;
using namespace SymbolRewriter;

#if 0 // HLSL Change Starts - option pending
static cl::list<std::string> RewriteMapFiles("rewrite-map-file",
                                             cl::desc("Symbol Rewrite Map"),
                                             cl::value_desc("filename"));
#endif // HLSL Change Ends

static void rewriteComdat(Module &M, GlobalObject *GO,
                          const std::string &Source,
                          const std::string &Target) {
  if (Comdat *CD = GO->getComdat()) {
    auto &Comdats = M.getComdatSymbolTable();

    Comdat *C = M.getOrInsertComdat(Target);
    C->setSelectionKind(CD->getSelectionKind());
    GO->setComdat(C);

    Comdats.erase(Comdats.find(Source));
  }
}

namespace {
template <RewriteDescriptor::Type DT, typename ValueType,
          ValueType *(llvm::Module::*Get)(StringRef) const>
class ExplicitRewriteDescriptor : public RewriteDescriptor {
public:
  const std::string Source;
  const std::string Target;

  ExplicitRewriteDescriptor(StringRef S, StringRef T, const bool Naked)
      : RewriteDescriptor(DT), Source(Naked ? StringRef("\01" + S.str()) : S),
        Target(T) {}

  bool performOnModule(Module &M) override;

  static bool classof(const RewriteDescriptor *RD) {
    return RD->getType() == DT;
  }
};

template <RewriteDescriptor::Type DT, typename ValueType,
          ValueType *(llvm::Module::*Get)(StringRef) const>
bool ExplicitRewriteDescriptor<DT, ValueType, Get>::performOnModule(Module &M) {
  bool Changed = false;
  if (ValueType *S = (M.*Get)(Source)) {
    if (GlobalObject *GO = dyn_cast<GlobalObject>(S))
      rewriteComdat(M, GO, Source, Target);

    if (Value *T = (M.*Get)(Target))
      S->setValueName(T->getValueName());
    else
      S->setName(Target);

    Changed = true;
  }
  return Changed;
}

template <RewriteDescriptor::Type DT, typename ValueType,
          ValueType *(llvm::Module::*Get)(StringRef) const,
          iterator_range<typename iplist<ValueType>::iterator>
          (llvm::Module::*Iterator)()>
class PatternRewriteDescriptor : public RewriteDescriptor {
public:
  const std::string Pattern;
  const std::string Transform;

  PatternRewriteDescriptor(StringRef P, StringRef T)
    : RewriteDescriptor(DT), Pattern(P), Transform(T) { }

  bool performOnModule(Module &M) override;

  static bool classof(const RewriteDescriptor *RD) {
    return RD->getType() == DT;
  }
};

template <RewriteDescriptor::Type DT, typename ValueType,
          ValueType *(llvm::Module::*Get)(StringRef) const,
          iterator_range<typename iplist<ValueType>::iterator>
          (llvm::Module::*Iterator)()>
bool PatternRewriteDescriptor<DT, ValueType, Get, Iterator>::
performOnModule(Module &M) {
  bool Changed = false;
  for (auto &C : (M.*Iterator)()) {
    std::string Error;

    std::string Name = Regex(Pattern).sub(Transform, C.getName(), &Error);
    if (!Error.empty())
      report_fatal_error("unable to transforn " + C.getName() + " in " +
                         M.getModuleIdentifier() + ": " + Error);

    if (C.getName() == Name)
      continue;

    if (GlobalObject *GO = dyn_cast<GlobalObject>(&C))
      rewriteComdat(M, GO, C.getName(), Name);

    if (Value *V = (M.*Get)(Name))
      C.setValueName(V->getValueName());
    else
      C.setName(Name);

    Changed = true;
  }
  return Changed;
}

/// Represents a rewrite for an explicitly named (function) symbol.  Both the
/// source function name and target function name of the transformation are
/// explicitly spelt out.
typedef ExplicitRewriteDescriptor<RewriteDescriptor::Type::Function,
                                  llvm::Function, &llvm::Module::getFunction>
    ExplicitRewriteFunctionDescriptor;

/// Represents a rewrite for an explicitly named (global variable) symbol.  Both
/// the source variable name and target variable name are spelt out.  This
/// applies only to module level variables.
typedef ExplicitRewriteDescriptor<RewriteDescriptor::Type::GlobalVariable,
                                  llvm::GlobalVariable,
                                  &llvm::Module::getGlobalVariable>
    ExplicitRewriteGlobalVariableDescriptor;

/// Represents a rewrite for an explicitly named global alias.  Both the source
/// and target name are explicitly spelt out.
typedef ExplicitRewriteDescriptor<RewriteDescriptor::Type::NamedAlias,
                                  llvm::GlobalAlias,
                                  &llvm::Module::getNamedAlias>
    ExplicitRewriteNamedAliasDescriptor;

/// Represents a rewrite for a regular expression based pattern for functions.
/// A pattern for the function name is provided and a transformation for that
/// pattern to determine the target function name create the rewrite rule.
typedef PatternRewriteDescriptor<RewriteDescriptor::Type::Function,
                                 llvm::Function, &llvm::Module::getFunction,
                                 &llvm::Module::functions>
    PatternRewriteFunctionDescriptor;

/// Represents a rewrite for a global variable based upon a matching pattern.
/// Each global variable matching the provided pattern will be transformed as
/// described in the transformation pattern for the target.  Applies only to
/// module level variables.
typedef PatternRewriteDescriptor<RewriteDescriptor::Type::GlobalVariable,
                                 llvm::GlobalVariable,
                                 &llvm::Module::getGlobalVariable,
                                 &llvm::Module::globals>
    PatternRewriteGlobalVariableDescriptor;

/// PatternRewriteNamedAliasDescriptor - represents a rewrite for global
/// aliases which match a given pattern.  The provided transformation will be
/// applied to each of the matching names.
typedef PatternRewriteDescriptor<RewriteDescriptor::Type::NamedAlias,
                                 llvm::GlobalAlias,
                                 &llvm::Module::getNamedAlias,
                                 &llvm::Module::aliases>
    PatternRewriteNamedAliasDescriptor;
} // namespace

bool RewriteMapParser::parse(const std::string &MapFile,
                             RewriteDescriptorList *DL) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Mapping =
      MemoryBuffer::getFile(MapFile);

  if (!Mapping)
    report_fatal_error("unable to read rewrite map '" + MapFile + "': " +
                       Mapping.getError().message());

  if (!parse(*Mapping, DL))
    report_fatal_error("unable to parse rewrite map '" + MapFile + "'");

  return true;
}

bool RewriteMapParser::parse(std::unique_ptr<MemoryBuffer> &MapFile,
                             RewriteDescriptorList *DL) {
  SourceMgr SM;
  yaml::Stream YS(MapFile->getBuffer(), SM);

  for (auto &Document : YS) {
    yaml::MappingNode *DescriptorList;

    // ignore empty documents
    if (isa<yaml::NullNode>(Document.getRoot()))
      continue;

    DescriptorList = dyn_cast<yaml::MappingNode>(Document.getRoot());
    if (!DescriptorList) {
      YS.printError(Document.getRoot(), "DescriptorList node must be a map");
      return false;
    }

    for (auto &Descriptor : *DescriptorList)
      if (!parseEntry(YS, Descriptor, DL))
        return false;
  }

  return true;
}

bool RewriteMapParser::parseEntry(yaml::Stream &YS, yaml::KeyValueNode &Entry,
                                  RewriteDescriptorList *DL) {
  yaml::ScalarNode *Key;
  yaml::MappingNode *Value;
  SmallString<32> KeyStorage;
  StringRef RewriteType;

  Key = dyn_cast<yaml::ScalarNode>(Entry.getKey());
  if (!Key) {
    YS.printError(Entry.getKey(), "rewrite type must be a scalar");
    return false;
  }

  Value = dyn_cast<yaml::MappingNode>(Entry.getValue());
  if (!Value) {
    YS.printError(Entry.getValue(), "rewrite descriptor must be a map");
    return false;
  }

  RewriteType = Key->getValue(KeyStorage);
  if (RewriteType.equals("function"))
    return parseRewriteFunctionDescriptor(YS, Key, Value, DL);
  else if (RewriteType.equals("global variable"))
    return parseRewriteGlobalVariableDescriptor(YS, Key, Value, DL);
  else if (RewriteType.equals("global alias"))
    return parseRewriteGlobalAliasDescriptor(YS, Key, Value, DL);

  YS.printError(Entry.getKey(), "unknown rewrite type");
  return false;
}

bool RewriteMapParser::
parseRewriteFunctionDescriptor(yaml::Stream &YS, yaml::ScalarNode *K,
                               yaml::MappingNode *Descriptor,
                               RewriteDescriptorList *DL) {
  bool Naked = false;
  std::string Source;
  std::string Target;
  std::string Transform;

  for (auto &Field : *Descriptor) {
    yaml::ScalarNode *Key;
    yaml::ScalarNode *Value;
    SmallString<32> KeyStorage;
    SmallString<32> ValueStorage;
    StringRef KeyValue;

    Key = dyn_cast<yaml::ScalarNode>(Field.getKey());
    if (!Key) {
      YS.printError(Field.getKey(), "descriptor key must be a scalar");
      return false;
    }

    Value = dyn_cast<yaml::ScalarNode>(Field.getValue());
    if (!Value) {
      YS.printError(Field.getValue(), "descriptor value must be a scalar");
      return false;
    }

    KeyValue = Key->getValue(KeyStorage);
    if (KeyValue.equals("source")) {
      std::string Error;

      Source = Value->getValue(ValueStorage);
      if (!Regex(Source).isValid(Error)) {
        YS.printError(Field.getKey(), "invalid regex: " + Error);
        return false;
      }
    } else if (KeyValue.equals("target")) {
      Target = Value->getValue(ValueStorage);
    } else if (KeyValue.equals("transform")) {
      Transform = Value->getValue(ValueStorage);
    } else if (KeyValue.equals("naked")) {
      std::string Undecorated;

      Undecorated = Value->getValue(ValueStorage);
      Naked = StringRef(Undecorated).lower() == "true" || Undecorated == "1";
    } else {
      YS.printError(Field.getKey(), "unknown key for function");
      return false;
    }
  }

  if (Transform.empty() == Target.empty()) {
    YS.printError(Descriptor,
                  "exactly one of transform or target must be specified");
    return false;
  }

  // TODO see if there is a more elegant solution to selecting the rewrite
  // descriptor type
  if (!Target.empty())
    DL->push_back(new ExplicitRewriteFunctionDescriptor(Source, Target, Naked));
  else
    DL->push_back(new PatternRewriteFunctionDescriptor(Source, Transform));

  return true;
}

bool RewriteMapParser::
parseRewriteGlobalVariableDescriptor(yaml::Stream &YS, yaml::ScalarNode *K,
                                     yaml::MappingNode *Descriptor,
                                     RewriteDescriptorList *DL) {
  std::string Source;
  std::string Target;
  std::string Transform;

  for (auto &Field : *Descriptor) {
    yaml::ScalarNode *Key;
    yaml::ScalarNode *Value;
    SmallString<32> KeyStorage;
    SmallString<32> ValueStorage;
    StringRef KeyValue;

    Key = dyn_cast<yaml::ScalarNode>(Field.getKey());
    if (!Key) {
      YS.printError(Field.getKey(), "descriptor Key must be a scalar");
      return false;
    }

    Value = dyn_cast<yaml::ScalarNode>(Field.getValue());
    if (!Value) {
      YS.printError(Field.getValue(), "descriptor value must be a scalar");
      return false;
    }

    KeyValue = Key->getValue(KeyStorage);
    if (KeyValue.equals("source")) {
      std::string Error;

      Source = Value->getValue(ValueStorage);
      if (!Regex(Source).isValid(Error)) {
        YS.printError(Field.getKey(), "invalid regex: " + Error);
        return false;
      }
    } else if (KeyValue.equals("target")) {
      Target = Value->getValue(ValueStorage);
    } else if (KeyValue.equals("transform")) {
      Transform = Value->getValue(ValueStorage);
    } else {
      YS.printError(Field.getKey(), "unknown Key for Global Variable");
      return false;
    }
  }

  if (Transform.empty() == Target.empty()) {
    YS.printError(Descriptor,
                  "exactly one of transform or target must be specified");
    return false;
  }

  if (!Target.empty())
    DL->push_back(new ExplicitRewriteGlobalVariableDescriptor(Source, Target,
                                                              /*Naked*/false));
  else
    DL->push_back(new PatternRewriteGlobalVariableDescriptor(Source,
                                                             Transform));

  return true;
}

bool RewriteMapParser::
parseRewriteGlobalAliasDescriptor(yaml::Stream &YS, yaml::ScalarNode *K,
                                  yaml::MappingNode *Descriptor,
                                  RewriteDescriptorList *DL) {
  std::string Source;
  std::string Target;
  std::string Transform;

  for (auto &Field : *Descriptor) {
    yaml::ScalarNode *Key;
    yaml::ScalarNode *Value;
    SmallString<32> KeyStorage;
    SmallString<32> ValueStorage;
    StringRef KeyValue;

    Key = dyn_cast<yaml::ScalarNode>(Field.getKey());
    if (!Key) {
      YS.printError(Field.getKey(), "descriptor key must be a scalar");
      return false;
    }

    Value = dyn_cast<yaml::ScalarNode>(Field.getValue());
    if (!Value) {
      YS.printError(Field.getValue(), "descriptor value must be a scalar");
      return false;
    }

    KeyValue = Key->getValue(KeyStorage);
    if (KeyValue.equals("source")) {
      std::string Error;

      Source = Value->getValue(ValueStorage);
      if (!Regex(Source).isValid(Error)) {
        YS.printError(Field.getKey(), "invalid regex: " + Error);
        return false;
      }
    } else if (KeyValue.equals("target")) {
      Target = Value->getValue(ValueStorage);
    } else if (KeyValue.equals("transform")) {
      Transform = Value->getValue(ValueStorage);
    } else {
      YS.printError(Field.getKey(), "unknown key for Global Alias");
      return false;
    }
  }

  if (Transform.empty() == Target.empty()) {
    YS.printError(Descriptor,
                  "exactly one of transform or target must be specified");
    return false;
  }

  if (!Target.empty())
    DL->push_back(new ExplicitRewriteNamedAliasDescriptor(Source, Target,
                                                          /*Naked*/false));
  else
    DL->push_back(new PatternRewriteNamedAliasDescriptor(Source, Transform));

  return true;
}

namespace {
class RewriteSymbols : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid

  RewriteSymbols();
  RewriteSymbols(SymbolRewriter::RewriteDescriptorList &DL);

  bool runOnModule(Module &M) override;

private:
  void loadAndParseMapFiles();

  SymbolRewriter::RewriteDescriptorList Descriptors;
};

char RewriteSymbols::ID = 0;

RewriteSymbols::RewriteSymbols() : ModulePass(ID) {
  initializeRewriteSymbolsPass(*PassRegistry::getPassRegistry());
  loadAndParseMapFiles();
}

RewriteSymbols::RewriteSymbols(SymbolRewriter::RewriteDescriptorList &DL)
    : ModulePass(ID) {
  Descriptors.splice(Descriptors.begin(), DL);
}

bool RewriteSymbols::runOnModule(Module &M) {
  bool Changed;

  Changed = false;
  for (auto &Descriptor : Descriptors)
    Changed |= Descriptor.performOnModule(M);

  return Changed;
}

void RewriteSymbols::loadAndParseMapFiles() {
  const std::vector<std::string> MapFiles; // HLSL Change - do not init from a global RewriteMapFiles
  SymbolRewriter::RewriteMapParser parser;

  for (const auto &MapFile : MapFiles)
    parser.parse(MapFile, &Descriptors);
}
}

INITIALIZE_PASS(RewriteSymbols, "rewrite-symbols", "Rewrite Symbols", false,
                false)

ModulePass *llvm::createRewriteSymbolsPass() { return new RewriteSymbols(); }

ModulePass *
llvm::createRewriteSymbolsPass(SymbolRewriter::RewriteDescriptorList &DL) {
  return new RewriteSymbols(DL);
}
