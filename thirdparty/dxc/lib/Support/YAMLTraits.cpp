//===- lib/Support/YAMLTraits.cpp -----------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/YAMLTraits.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstring>
using namespace llvm;
using namespace yaml;

//===----------------------------------------------------------------------===//
//  IO
//===----------------------------------------------------------------------===//

IO::IO(void *Context) : Ctxt(Context) {
}

IO::~IO() {
}

void *IO::getContext() {
  return Ctxt;
}

void IO::setContext(void *Context) {
  Ctxt = Context;
}

//===----------------------------------------------------------------------===//
//  Input
//===----------------------------------------------------------------------===//

Input::Input(StringRef InputContent,
             void *Ctxt,
             SourceMgr::DiagHandlerTy DiagHandler,
             void *DiagHandlerCtxt)
  : IO(Ctxt),
    Strm(new Stream(InputContent, SrcMgr)),
    CurrentNode(nullptr) {
  if (DiagHandler)
    SrcMgr.setDiagHandler(DiagHandler, DiagHandlerCtxt);
  DocIterator = Strm->begin();
}

Input::~Input() {
}

std::error_code Input::error() { return EC; }

// Pin the vtables to this file.
void Input::HNode::anchor() {}
void Input::EmptyHNode::anchor() {}
void Input::ScalarHNode::anchor() {}
void Input::MapHNode::anchor() {}
void Input::SequenceHNode::anchor() {}

bool Input::outputting() {
  return false;
}

bool Input::setCurrentDocument() {
  if (DocIterator != Strm->end()) {
    Node *N = DocIterator->getRoot();
    if (!N) {
      assert(Strm->failed() && "Root is NULL iff parsing failed");
      EC = make_error_code(errc::invalid_argument);
      return false;
    }

    if (isa<NullNode>(N)) {
      // Empty files are allowed and ignored
      ++DocIterator;
      return setCurrentDocument();
    }
    TopNode = this->createHNodes(N);
    CurrentNode = TopNode.get();
    return true;
  }
  return false;
}

bool Input::nextDocument() {
  return ++DocIterator != Strm->end();
}

const Node *Input::getCurrentNode() const {
  return CurrentNode ? CurrentNode->_node : nullptr;
}

bool Input::mapTag(StringRef Tag, bool Default) {
  std::string foundTag = CurrentNode->_node->getVerbatimTag();
  if (foundTag.empty()) {
    // If no tag found and 'Tag' is the default, say it was found.
    return Default;
  }
  // Return true iff found tag matches supplied tag.
  return Tag.equals(foundTag);
}

void Input::beginMapping() {
  if (EC)
    return;
  // CurrentNode can be null if the document is empty.
  MapHNode *MN = dyn_cast_or_null<MapHNode>(CurrentNode);
  if (MN) {
    MN->ValidKeys.clear();
  }
}

bool Input::preflightKey(const char *Key, bool Required, bool, bool &UseDefault,
                         void *&SaveInfo) {
  UseDefault = false;
  if (EC)
    return false;

  // CurrentNode is null for empty documents, which is an error in case required
  // nodes are present.
  if (!CurrentNode) {
    if (Required)
      EC = make_error_code(errc::invalid_argument);
    return false;
  }

  MapHNode *MN = dyn_cast<MapHNode>(CurrentNode);
  if (!MN) {
    setError(CurrentNode, "not a mapping");
    return false;
  }
  MN->ValidKeys.push_back(Key);
  HNode *Value = MN->Mapping[Key].get();
  if (!Value) {
    if (Required)
      setError(CurrentNode, Twine("missing required key '") + Key + "'");
    else
      UseDefault = true;
    return false;
  }
  SaveInfo = CurrentNode;
  CurrentNode = Value;
  return true;
}

void Input::postflightKey(void *saveInfo) {
  CurrentNode = reinterpret_cast<HNode *>(saveInfo);
}

void Input::endMapping() {
  if (EC)
    return;
  // CurrentNode can be null if the document is empty.
  MapHNode *MN = dyn_cast_or_null<MapHNode>(CurrentNode);
  if (!MN)
    return;
  for (const auto &NN : MN->Mapping) {
    if (!MN->isValidKey(NN.first())) {
      setError(NN.second.get(), Twine("unknown key '") + NN.first() + "'");
      break;
    }
  }
}

void Input::beginFlowMapping() { beginMapping(); }

void Input::endFlowMapping() { endMapping(); }

unsigned Input::beginSequence() {
  if (SequenceHNode *SQ = dyn_cast<SequenceHNode>(CurrentNode))
    return SQ->Entries.size();
  if (isa<EmptyHNode>(CurrentNode))
    return 0;
  // Treat case where there's a scalar "null" value as an empty sequence.
  if (ScalarHNode *SN = dyn_cast<ScalarHNode>(CurrentNode)) {
    if (isNull(SN->value()))
      return 0;
  }
  // Any other type of HNode is an error.
  setError(CurrentNode, "not a sequence");
  return 0;
}

void Input::endSequence() {
}

bool Input::preflightElement(unsigned Index, void *&SaveInfo) {
  if (EC)
    return false;
  if (SequenceHNode *SQ = dyn_cast<SequenceHNode>(CurrentNode)) {
    SaveInfo = CurrentNode;
    CurrentNode = SQ->Entries[Index].get();
    return true;
  }
  return false;
}

void Input::postflightElement(void *SaveInfo) {
  CurrentNode = reinterpret_cast<HNode *>(SaveInfo);
}

unsigned Input::beginFlowSequence() { return beginSequence(); }

bool Input::preflightFlowElement(unsigned index, void *&SaveInfo) {
  if (EC)
    return false;
  if (SequenceHNode *SQ = dyn_cast<SequenceHNode>(CurrentNode)) {
    SaveInfo = CurrentNode;
    CurrentNode = SQ->Entries[index].get();
    return true;
  }
  return false;
}

void Input::postflightFlowElement(void *SaveInfo) {
  CurrentNode = reinterpret_cast<HNode *>(SaveInfo);
}

void Input::endFlowSequence() {
}

void Input::beginEnumScalar() {
  ScalarMatchFound = false;
}

bool Input::matchEnumScalar(const char *Str, bool) {
  if (ScalarMatchFound)
    return false;
  if (ScalarHNode *SN = dyn_cast<ScalarHNode>(CurrentNode)) {
    if (SN->value().equals(Str)) {
      ScalarMatchFound = true;
      return true;
    }
  }
  return false;
}

bool Input::matchEnumFallback() {
  if (ScalarMatchFound)
    return false;
  ScalarMatchFound = true;
  return true;
}

void Input::endEnumScalar() {
  if (!ScalarMatchFound) {
    setError(CurrentNode, "unknown enumerated scalar");
  }
}

bool Input::beginBitSetScalar(bool &DoClear) {
  BitValuesUsed.clear();
  if (SequenceHNode *SQ = dyn_cast<SequenceHNode>(CurrentNode)) {
    BitValuesUsed.insert(BitValuesUsed.begin(), SQ->Entries.size(), false);
  } else {
    setError(CurrentNode, "expected sequence of bit values");
  }
  DoClear = true;
  return true;
}

bool Input::bitSetMatch(const char *Str, bool) {
  if (EC)
    return false;
  if (SequenceHNode *SQ = dyn_cast<SequenceHNode>(CurrentNode)) {
    unsigned Index = 0;
    for (auto &N : SQ->Entries) {
      if (ScalarHNode *SN = dyn_cast<ScalarHNode>(N.get())) {
        if (SN->value().equals(Str)) {
          BitValuesUsed[Index] = true;
          return true;
        }
      } else {
        setError(CurrentNode, "unexpected scalar in sequence of bit values");
      }
      ++Index;
    }
  } else {
    setError(CurrentNode, "expected sequence of bit values");
  }
  return false;
}

void Input::endBitSetScalar() {
  if (EC)
    return;
  if (SequenceHNode *SQ = dyn_cast<SequenceHNode>(CurrentNode)) {
    assert(BitValuesUsed.size() == SQ->Entries.size());
    for (unsigned i = 0; i < SQ->Entries.size(); ++i) {
      if (!BitValuesUsed[i]) {
        setError(SQ->Entries[i].get(), "unknown bit value");
        return;
      }
    }
  }
}

void Input::scalarString(StringRef &S, bool) {
  if (ScalarHNode *SN = dyn_cast<ScalarHNode>(CurrentNode)) {
    S = SN->value();
  } else {
    setError(CurrentNode, "unexpected scalar");
  }
}

void Input::blockScalarString(StringRef &S) { scalarString(S, false); }

void Input::setError(HNode *hnode, const Twine &message) {
  assert(hnode && "HNode must not be NULL");
  this->setError(hnode->_node, message);
}

void Input::setError(Node *node, const Twine &message) {
  Strm->printError(node, message);
  EC = make_error_code(errc::invalid_argument);
}

std::unique_ptr<Input::HNode> Input::createHNodes(Node *N) {
  SmallString<128> StringStorage;
  if (ScalarNode *SN = dyn_cast<ScalarNode>(N)) {
    StringRef KeyStr = SN->getValue(StringStorage);
    if (!StringStorage.empty()) {
      // Copy string to permanent storage
      unsigned Len = StringStorage.size();
      char *Buf = StringAllocator.Allocate<char>(Len);
      memcpy(Buf, &StringStorage[0], Len);
      KeyStr = StringRef(Buf, Len);
    }
    return llvm::make_unique<ScalarHNode>(N, KeyStr);
  } else if (BlockScalarNode *BSN = dyn_cast<BlockScalarNode>(N)) {
    StringRef Value = BSN->getValue();
    char *Buf = StringAllocator.Allocate<char>(Value.size());
    memcpy(Buf, Value.data(), Value.size());
    return llvm::make_unique<ScalarHNode>(N, StringRef(Buf, Value.size()));
  } else if (SequenceNode *SQ = dyn_cast<SequenceNode>(N)) {
    auto SQHNode = llvm::make_unique<SequenceHNode>(N);
    for (Node &SN : *SQ) {
      auto Entry = this->createHNodes(&SN);
      if (EC)
        break;
      SQHNode->Entries.push_back(std::move(Entry));
    }
    return std::move(SQHNode);
  } else if (MappingNode *Map = dyn_cast<MappingNode>(N)) {
    auto mapHNode = llvm::make_unique<MapHNode>(N);
    for (KeyValueNode &KVN : *Map) {
      Node *KeyNode = KVN.getKey();
      ScalarNode *KeyScalar = dyn_cast<ScalarNode>(KeyNode);
      if (!KeyScalar) {
        setError(KeyNode, "Map key must be a scalar");
        break;
      }
      StringStorage.clear();
      StringRef KeyStr = KeyScalar->getValue(StringStorage);
      if (!StringStorage.empty()) {
        // Copy string to permanent storage
        unsigned Len = StringStorage.size();
        char *Buf = StringAllocator.Allocate<char>(Len);
        memcpy(Buf, &StringStorage[0], Len);
        KeyStr = StringRef(Buf, Len);
      }
      auto ValueHNode = this->createHNodes(KVN.getValue());
      if (EC)
        break;
      mapHNode->Mapping[KeyStr] = std::move(ValueHNode);
    }
    return std::move(mapHNode);
  } else if (isa<NullNode>(N)) {
    return llvm::make_unique<EmptyHNode>(N);
  } else {
    setError(N, "unknown node kind");
    return nullptr;
  }
}

bool Input::MapHNode::isValidKey(StringRef Key) {
  for (const char *K : ValidKeys) {
    if (Key.equals(K))
      return true;
  }
  return false;
}

void Input::setError(const Twine &Message) {
  this->setError(CurrentNode, Message);
}

bool Input::canElideEmptySequence() {
  return false;
}

//===----------------------------------------------------------------------===//
//  Output
//===----------------------------------------------------------------------===//

Output::Output(raw_ostream &yout, void *context, int WrapColumn)
    : IO(context),
      Out(yout),
      WrapColumn(WrapColumn),
      Column(0),
      ColumnAtFlowStart(0),
      ColumnAtMapFlowStart(0),
      NeedBitValueComma(false),
      NeedFlowSequenceComma(false),
      EnumerationMatchFound(false),
      NeedsNewLine(false) {
}

Output::~Output() {
}

bool Output::outputting() {
  return true;
}

void Output::beginMapping() {
  StateStack.push_back(inMapFirstKey);
  NeedsNewLine = true;
}

bool Output::mapTag(StringRef Tag, bool Use) {
  if (Use) {
    this->output(" ");
    this->output(Tag);
  }
  return Use;
}

void Output::endMapping() {
  StateStack.pop_back();
}

bool Output::preflightKey(const char *Key, bool Required, bool SameAsDefault,
                          bool &UseDefault, void *&) {
  UseDefault = false;
  if (Required || !SameAsDefault) {
    auto State = StateStack.back();
    if (State == inFlowMapFirstKey || State == inFlowMapOtherKey) {
      flowKey(Key);
    } else {
      this->newLineCheck();
      this->paddedKey(Key);
    }
    return true;
  }
  return false;
}

void Output::postflightKey(void *) {
  if (StateStack.back() == inMapFirstKey) {
    StateStack.pop_back();
    StateStack.push_back(inMapOtherKey);
  } else if (StateStack.back() == inFlowMapFirstKey) {
    StateStack.pop_back();
    StateStack.push_back(inFlowMapOtherKey);
  }
}

void Output::beginFlowMapping() {
  StateStack.push_back(inFlowMapFirstKey);
  this->newLineCheck();
  ColumnAtMapFlowStart = Column;
  output("{ ");
}

void Output::endFlowMapping() {
  StateStack.pop_back();
  this->outputUpToEndOfLine(" }");
}

void Output::beginDocuments() {
  this->outputUpToEndOfLine("---");
}

bool Output::preflightDocument(unsigned index) {
  if (index > 0)
    this->outputUpToEndOfLine("\n---");
  return true;
}

void Output::postflightDocument() {
}

void Output::endDocuments() {
  output("\n...\n");
}

unsigned Output::beginSequence() {
  StateStack.push_back(inSeq);
  NeedsNewLine = true;
  return 0;
}

void Output::endSequence() {
  StateStack.pop_back();
}

bool Output::preflightElement(unsigned, void *&) {
  return true;
}

void Output::postflightElement(void *) {
}

unsigned Output::beginFlowSequence() {
  StateStack.push_back(inFlowSeq);
  this->newLineCheck();
  ColumnAtFlowStart = Column;
  output("[ ");
  NeedFlowSequenceComma = false;
  return 0;
}

void Output::endFlowSequence() {
  StateStack.pop_back();
  this->outputUpToEndOfLine(" ]");
}

bool Output::preflightFlowElement(unsigned, void *&) {
  if (NeedFlowSequenceComma)
    output(", ");
  if (WrapColumn && Column > WrapColumn) {
    output("\n");
    for (int i = 0; i < ColumnAtFlowStart; ++i)
      output(" ");
    Column = ColumnAtFlowStart;
    output("  ");
  }
  return true;
}

void Output::postflightFlowElement(void *) {
  NeedFlowSequenceComma = true;
}

void Output::beginEnumScalar() {
  EnumerationMatchFound = false;
}

bool Output::matchEnumScalar(const char *Str, bool Match) {
  if (Match && !EnumerationMatchFound) {
    this->newLineCheck();
    this->outputUpToEndOfLine(Str);
    EnumerationMatchFound = true;
  }
  return false;
}

bool Output::matchEnumFallback() {
  if (EnumerationMatchFound)
    return false;
  EnumerationMatchFound = true;
  return true;
}

void Output::endEnumScalar() {
  if (!EnumerationMatchFound)
    llvm_unreachable("bad runtime enum value");
}

bool Output::beginBitSetScalar(bool &DoClear) {
  this->newLineCheck();
  output("[ ");
  NeedBitValueComma = false;
  DoClear = false;
  return true;
}

bool Output::bitSetMatch(const char *Str, bool Matches) {
  if (Matches) {
    if (NeedBitValueComma)
      output(", ");
    this->output(Str);
    NeedBitValueComma = true;
  }
  return false;
}

void Output::endBitSetScalar() {
  this->outputUpToEndOfLine(" ]");
}

void Output::scalarString(StringRef &S, bool MustQuote) {
  this->newLineCheck();
  if (S.empty()) {
    // Print '' for the empty string because leaving the field empty is not
    // allowed.
    this->outputUpToEndOfLine("''");
    return;
  }
  if (!MustQuote) {
    // Only quote if we must.
    this->outputUpToEndOfLine(S);
    return;
  }
  unsigned i = 0;
  unsigned j = 0;
  unsigned End = S.size();
  output("'"); // Starting single quote.
  const char *Base = S.data();
  while (j < End) {
    // Escape a single quote by doubling it.
    if (S[j] == '\'') {
      output(StringRef(&Base[i], j - i + 1));
      output("'");
      i = j + 1;
    }
    ++j;
  }
  output(StringRef(&Base[i], j - i));
  this->outputUpToEndOfLine("'"); // Ending single quote.
}

void Output::blockScalarString(StringRef &S) {
  if (!StateStack.empty())
    newLineCheck();
  output(" |");
  outputNewLine();

  unsigned Indent = StateStack.empty() ? 1 : StateStack.size();

  auto Buffer = MemoryBuffer::getMemBuffer(S, "", false);
  for (line_iterator Lines(*Buffer, false); !Lines.is_at_end(); ++Lines) {
    for (unsigned I = 0; I < Indent; ++I) {
      output("  ");
    }
    output(*Lines);
    outputNewLine();
  }
}

void Output::setError(const Twine &message) {
}

bool Output::canElideEmptySequence() {
  // Normally, with an optional key/value where the value is an empty sequence,
  // the whole key/value can be not written.  But, that produces wrong yaml
  // if the key/value is the only thing in the map and the map is used in
  // a sequence.  This detects if the this sequence is the first key/value
  // in map that itself is embedded in a sequnce.
  if (StateStack.size() < 2)
    return true;
  if (StateStack.back() != inMapFirstKey)
    return true;
  return (StateStack[StateStack.size()-2] != inSeq);
}

void Output::output(StringRef s) {
  Column += s.size();
  Out << s;
}

void Output::outputUpToEndOfLine(StringRef s) {
  this->output(s);
  if (StateStack.empty() || (StateStack.back() != inFlowSeq &&
                             StateStack.back() != inFlowMapFirstKey &&
                             StateStack.back() != inFlowMapOtherKey))
    NeedsNewLine = true;
}

void Output::outputNewLine() {
  Out << "\n";
  Column = 0;
}

// if seq at top, indent as if map, then add "- "
// if seq in middle, use "- " if firstKey, else use "  "
//

void Output::newLineCheck() {
  if (!NeedsNewLine)
    return;
  NeedsNewLine = false;

  this->outputNewLine();

  assert(StateStack.size() > 0);
  unsigned Indent = StateStack.size() - 1;
  bool OutputDash = false;

  if (StateStack.back() == inSeq) {
    OutputDash = true;
  } else if ((StateStack.size() > 1) && ((StateStack.back() == inMapFirstKey) ||
             (StateStack.back() == inFlowSeq) ||
             (StateStack.back() == inFlowMapFirstKey)) &&
             (StateStack[StateStack.size() - 2] == inSeq)) {
    --Indent;
    OutputDash = true;
  }

  for (unsigned i = 0; i < Indent; ++i) {
    output("  ");
  }
  if (OutputDash) {
    output("- ");
  }

}

void Output::paddedKey(StringRef key) {
  output(key);
  output(":");
  const char *spaces = "                ";
  if (key.size() < strlen(spaces))
    output(&spaces[key.size()]);
  else
    output(" ");
}

void Output::flowKey(StringRef Key) {
  if (StateStack.back() == inFlowMapOtherKey)
    output(", ");
  if (WrapColumn && Column > WrapColumn) {
    output("\n");
    for (int I = 0; I < ColumnAtMapFlowStart; ++I)
      output(" ");
    Column = ColumnAtMapFlowStart;
    output("  ");
  }
  output(Key);
  output(": ");
}

//===----------------------------------------------------------------------===//
//  traits for built-in types
//===----------------------------------------------------------------------===//

void ScalarTraits<bool>::output(const bool &Val, void *, raw_ostream &Out) {
  Out << (Val ? "true" : "false");
}

StringRef ScalarTraits<bool>::input(StringRef Scalar, void *, bool &Val) {
  if (Scalar.equals("true")) {
    Val = true;
    return StringRef();
  } else if (Scalar.equals("false")) {
    Val = false;
    return StringRef();
  }
  return "invalid boolean";
}

void ScalarTraits<StringRef>::output(const StringRef &Val, void *,
                                     raw_ostream &Out) {
  Out << Val;
}

StringRef ScalarTraits<StringRef>::input(StringRef Scalar, void *,
                                         StringRef &Val) {
  Val = Scalar;
  return StringRef();
}

void ScalarTraits<std::string>::output(const std::string &Val, void *,
                                     raw_ostream &Out) {
  Out << Val;
}

StringRef ScalarTraits<std::string>::input(StringRef Scalar, void *,
                                         std::string &Val) {
  Val = Scalar.str();
  return StringRef();
}

void ScalarTraits<uint8_t>::output(const uint8_t &Val, void *,
                                   raw_ostream &Out) {
  // use temp uin32_t because ostream thinks uint8_t is a character
  uint32_t Num = Val;
  Out << Num;
}

StringRef ScalarTraits<uint8_t>::input(StringRef Scalar, void *, uint8_t &Val) {
  unsigned long long n;
  if (getAsUnsignedInteger(Scalar, 0, n))
    return "invalid number";
  if (n > 0xFF)
    return "out of range number";
  Val = n;
  return StringRef();
}

void ScalarTraits<uint16_t>::output(const uint16_t &Val, void *,
                                    raw_ostream &Out) {
  Out << Val;
}

StringRef ScalarTraits<uint16_t>::input(StringRef Scalar, void *,
                                        uint16_t &Val) {
  unsigned long long n;
  if (getAsUnsignedInteger(Scalar, 0, n))
    return "invalid number";
  if (n > 0xFFFF)
    return "out of range number";
  Val = n;
  return StringRef();
}

void ScalarTraits<uint32_t>::output(const uint32_t &Val, void *,
                                    raw_ostream &Out) {
  Out << Val;
}

StringRef ScalarTraits<uint32_t>::input(StringRef Scalar, void *,
                                        uint32_t &Val) {
  unsigned long long n;
  if (getAsUnsignedInteger(Scalar, 0, n))
    return "invalid number";
  if (n > 0xFFFFFFFFUL)
    return "out of range number";
  Val = n;
  return StringRef();
}

void ScalarTraits<uint64_t>::output(const uint64_t &Val, void *,
                                    raw_ostream &Out) {
  Out << Val;
}

StringRef ScalarTraits<uint64_t>::input(StringRef Scalar, void *,
                                        uint64_t &Val) {
  unsigned long long N;
  if (getAsUnsignedInteger(Scalar, 0, N))
    return "invalid number";
  Val = N;
  return StringRef();
}

void ScalarTraits<int8_t>::output(const int8_t &Val, void *, raw_ostream &Out) {
  // use temp in32_t because ostream thinks int8_t is a character
  int32_t Num = Val;
  Out << Num;
}

StringRef ScalarTraits<int8_t>::input(StringRef Scalar, void *, int8_t &Val) {
  long long N;
  if (getAsSignedInteger(Scalar, 0, N))
    return "invalid number";
  if ((N > 127) || (N < -128))
    return "out of range number";
  Val = N;
  return StringRef();
}

void ScalarTraits<int16_t>::output(const int16_t &Val, void *,
                                   raw_ostream &Out) {
  Out << Val;
}

StringRef ScalarTraits<int16_t>::input(StringRef Scalar, void *, int16_t &Val) {
  long long N;
  if (getAsSignedInteger(Scalar, 0, N))
    return "invalid number";
  if ((N > INT16_MAX) || (N < INT16_MIN))
    return "out of range number";
  Val = N;
  return StringRef();
}

void ScalarTraits<int32_t>::output(const int32_t &Val, void *,
                                   raw_ostream &Out) {
  Out << Val;
}

StringRef ScalarTraits<int32_t>::input(StringRef Scalar, void *, int32_t &Val) {
  long long N;
  if (getAsSignedInteger(Scalar, 0, N))
    return "invalid number";
  if ((N > INT32_MAX) || (N < INT32_MIN))
    return "out of range number";
  Val = N;
  return StringRef();
}

void ScalarTraits<int64_t>::output(const int64_t &Val, void *,
                                   raw_ostream &Out) {
  Out << Val;
}

StringRef ScalarTraits<int64_t>::input(StringRef Scalar, void *, int64_t &Val) {
  long long N;
  if (getAsSignedInteger(Scalar, 0, N))
    return "invalid number";
  Val = N;
  return StringRef();
}

void ScalarTraits<double>::output(const double &Val, void *, raw_ostream &Out) {
  Out << format("%g", Val);
}

StringRef ScalarTraits<double>::input(StringRef Scalar, void *, double &Val) {
  SmallString<32> buff(Scalar.begin(), Scalar.end());
  char *end;
  Val = strtod(buff.c_str(), &end);
  if (*end != '\0')
    return "invalid floating point number";
  return StringRef();
}

void ScalarTraits<float>::output(const float &Val, void *, raw_ostream &Out) {
  Out << format("%g", Val);
}

StringRef ScalarTraits<float>::input(StringRef Scalar, void *, float &Val) {
  SmallString<32> buff(Scalar.begin(), Scalar.end());
  char *end;
  Val = strtod(buff.c_str(), &end);
  if (*end != '\0')
    return "invalid floating point number";
  return StringRef();
}

void ScalarTraits<Hex8>::output(const Hex8 &Val, void *, raw_ostream &Out) {
  uint8_t Num = Val;
  Out << format("0x%02X", Num);
}

StringRef ScalarTraits<Hex8>::input(StringRef Scalar, void *, Hex8 &Val) {
  unsigned long long n;
  if (getAsUnsignedInteger(Scalar, 0, n))
    return "invalid hex8 number";
  if (n > 0xFF)
    return "out of range hex8 number";
  Val = n;
  return StringRef();
}

void ScalarTraits<Hex16>::output(const Hex16 &Val, void *, raw_ostream &Out) {
  uint16_t Num = Val;
  Out << format("0x%04X", Num);
}

StringRef ScalarTraits<Hex16>::input(StringRef Scalar, void *, Hex16 &Val) {
  unsigned long long n;
  if (getAsUnsignedInteger(Scalar, 0, n))
    return "invalid hex16 number";
  if (n > 0xFFFF)
    return "out of range hex16 number";
  Val = n;
  return StringRef();
}

void ScalarTraits<Hex32>::output(const Hex32 &Val, void *, raw_ostream &Out) {
  uint32_t Num = Val;
  Out << format("0x%08X", Num);
}

StringRef ScalarTraits<Hex32>::input(StringRef Scalar, void *, Hex32 &Val) {
  unsigned long long n;
  if (getAsUnsignedInteger(Scalar, 0, n))
    return "invalid hex32 number";
  if (n > 0xFFFFFFFFUL)
    return "out of range hex32 number";
  Val = n;
  return StringRef();
}

void ScalarTraits<Hex64>::output(const Hex64 &Val, void *, raw_ostream &Out) {
  uint64_t Num = Val;
  Out << format("0x%016llX", Num);
}

StringRef ScalarTraits<Hex64>::input(StringRef Scalar, void *, Hex64 &Val) {
  unsigned long long Num;
  if (getAsUnsignedInteger(Scalar, 0, Num))
    return "invalid hex64 number";
  Val = Num;
  return StringRef();
}
