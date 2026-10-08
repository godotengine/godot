// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_MASTER_PARSER_H_
#define SRC_MASTER_PARSER_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "src/element_parser.h"
#include "src/id_parser.h"
#include "src/size_parser.h"
#include "src/skip_parser.h"
#include "src/unknown_parser.h"
#include "src/void_parser.h"
#include "webm/callback.h"
#include "webm/element.h"
#include "webm/id.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// A general purpose parser for EBML master elements.
//
// For example, if a document specification defines a Foo master element that
// has two boolean children (Bar and Baz), then a FooParser capable of parsing
// the Foo master element could be defined as follows:
//
// struct FooParser : public MasterParser {
//   FooParser()
//       : MasterParser(MakeChild<BoolParser>(Id::kBar),
//                      MakeChild<BoolParser>(Id::kBaz)) {}
// };
//
// See the MasterValueParser for an alternative class for parsing master
// elements into a data structure.
class MasterParser : public ElementParser {
 public:
  // Constructs a new MasterParser that uses the given
  // {Id, std::unique_ptr<ElementParser>} pairs to map child IDs to the
  // appropriate parser/handler. Each argument must be of type
  // std::pair<Id, std::unique_ptr<ElementParser>>. If a parser is not
  // explicitly provided for Id::kVoid, a VoidParser will automatically be used
  // for it.
  //
  // Initializer lists don't support move-only types (i.e. std::unique_ptr), so
  // instead a variadic template is used.
  template <typename... T>
  explicit MasterParser(T&&... parser_pairs) {
    // Prefer an odd reserve size. This makes libc++ use a prime number for the
    // bucket count. Otherwise, if it happens to be a power of 2, then libc++
    // will use a power-of-2 bucket count (and since Matroska EBML IDs have low
    // entropy in the low bits, there will be a lot of collisions). libstdc++
    // always prefers a prime bucket count. I'm not sure how MSVC or others are
    // implemented, but this shouldn't adversely affect them even if they are
    // implemented differently. Add one to the count because we'll likely need
    // to insert a parser for Id::kVoid.
    parsers_.reserve((sizeof...(T) + 1) | 1);

    // This dummy initializer list is just used to force the parameter pack to
    // be expanded, which turns the expression into a for-each "loop" that
    // inserts each argument into the map.
    auto dummy = {0, (InsertParser(std::forward<T>(parser_pairs)), 0)...};
    (void)dummy;  // Silence unused variable warning.

    if (parsers_.find(Id::kVoid) == parsers_.end()) {
      InsertParser(MakeChild<VoidParser>(Id::kVoid));
    }
  }

  MasterParser(const MasterParser&) = delete;
  MasterParser& operator=(const MasterParser&) = delete;

  Status Init(const ElementMetadata& metadata, std::uint64_t max_size) override;

  void InitAfterSeek(const Ancestory& child_ancestory,
                     const ElementMetadata& child_metadata) override;

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override;

  bool GetCachedMetadata(ElementMetadata* metadata) override;

  std::uint32_t header_size() const { return header_size_; }

  // Gets the size of this element. May be called before the parse is fully
  // complete (but only after Init() has already been called and successfully
  // returned).
  std::uint64_t size() const { return my_size_; }

  // Gets absolute byte position of the start of the element in the byte stream.
  // May be called before the parse is fully complete (but only after Init() has
  // already been called and successfully returned).
  std::uint64_t position() const { return my_position_; }

  // Gets the metadata for the child that is currently being parsed. This may
  // only be called while the child's body (not its header information like ID
  // and size) is being parsed.
  const ElementMetadata& child_metadata() const {
    assert(state_ == State::kValidatingChildSize ||
           state_ == State::kGettingAction ||
           state_ == State::kInitializingChildParser ||
           state_ == State::kReadingChildBody);
    return child_metadata_;
  }

 protected:
  // Allocates a new parser of type T, forwarding args to the constructor, and
  // creates a std::pair<Id, std::unique_ptr<ElementParser>> using the given id
  // and the allocated parser.
  template <typename T, typename... Args>
  static std::pair<Id, std::unique_ptr<ElementParser>> MakeChild(
      Id id, Args&&... args) {
    std::unique_ptr<ElementParser> ptr(new T(std::forward<Args>(args)...));
    return std::pair<Id, std::unique_ptr<ElementParser>>(id, std::move(ptr));
  }

 private:
  // Parsing states for the finite-state machine.
  enum class State {
    /* clang-format off */
    // State                      Transitions to state      When
    kFirstReadOfChildId,       // kFinishingReadingChildId  size(id)  > 1
                               // kReadingChildSize         size(id) == 1
                               // kEndReached               EOF
    kFinishingReadingChildId,  // kReadingChildSize         done
    kReadingChildSize,         // kValidatingChildSize      done
    kValidatingChildSize,      // kGettingAction            done
                               // kEndReached               unknown id & unsized
    kGettingAction,            // kInitializingChildParser  done
    kInitializingChildParser,  // kReadingChildBody         done
    kReadingChildBody,         // kChildFullyParsed         child parse done
    kChildFullyParsed,         // kValidatingChildSize      cached metadata
                               // kFirstReadOfChildId       read  < my_size_
                               // kEndReached               read == my_size_
    kEndReached,               // No transitions from here (must call Init)
    /* clang-format on */
  };

  using StdHashIdKeyType = std::underlying_type<Id>::type;
  using StdHashId = std::hash<StdHashIdKeyType>;

  // Hash functor for hashing Id enums for storage in std::unordered_map.
  struct IdHash : StdHashId {
#if __cpp_lib_is_invocable
    using result_type = std::invoke_result<StdHashId, StdHashIdKeyType>::type;
#else
    using result_type = StdHashId::result_type;
#endif  // __cpp_lib_is_invocable

    // Returns the hash of the given id.
    result_type operator()(Id id) const {
      return StdHashId::operator()(static_cast<StdHashIdKeyType>(id));
    }
  };

  // The parser for parsing element Ids.
  IdParser id_parser_;

  // The parser for parsing element sizes.
  SizeParser size_parser_;

  // Metadata for the child element that is currently being parsed.
  ElementMetadata child_metadata_;

  // Maps child IDs to the appropriate parser that can handle that child.
  std::unordered_map<Id, std::unique_ptr<ElementParser>, IdHash> parsers_;

  // The parser that is used to parse unknown children.
  UnknownParser unknown_parser_;

  // The parser that is used to skip over children.
  SkipParser skip_parser_;

  // The parser that is being used to parse the current child. This must be null
  // or a pointer in parsers_.
  ElementParser* child_parser_;

  // The current parsing action for the child that is currently being parsed.
  Action action_ = Action::kRead;

  // The current state of the parser.
  State state_;

  std::uint32_t header_size_;

  // The size of this element.
  std::uint64_t my_size_;

  std::uint64_t my_position_;

  std::uint64_t max_size_;

  // The total number of bytes read by this parser.
  std::uint64_t total_bytes_read_;

  // Set to true if parsing has completed and this parser consumed an extra
  // element header (ID and size) that wasn't from a child.
  bool has_cached_metadata_ = false;

  // Inserts the parser into the parsers_ map and asserts it is the only parser
  // registers to parse the corresponding Id.
  template <typename T>
  void InsertParser(T&& parser) {
    bool inserted = parsers_.insert(std::forward<T>(parser)).second;
    (void)inserted;  // Silence unused variable warning.
    assert(inserted);  // Make sure there aren't duplicates.
  }

  // Common initialization logic for Init/InitAfterseek.
  void InitSetup(std::uint32_t header_size, std::uint64_t size_in_bytes,
                 std::uint64_t position);

  // Resets the internal parsers in preparation for parsing the next child.
  void PrepareForNextChild();
};

}  // namespace webm

#endif  // SRC_MASTER_PARSER_H_
