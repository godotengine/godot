// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_MASTER_VALUE_PARSER_H_
#define SRC_MASTER_VALUE_PARSER_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/master_parser.h"
#include "src/recursive_parser.h"
#include "src/skip_callback.h"
#include "webm/callback.h"
#include "webm/element.h"
#include "webm/id.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Parses Master elements from an EBML stream, storing child values in a
// structure of type T. This class differs from MasterParser in that
// MasterParser does not collect the parsed data into an object that can then be
// retrieved.
//
// For example, consider the following Foo object, which represents a master
// element that contains two booleans:
//
// struct Foo {
//   Element<bool> bar;
//   Element<bool> baz;
// };
//
// A FooParser implemented via MasterParser, like below, could be used to parse
// the master element and the boolean children, but the boolean values could not
// be retrieved from the parser.
//
// struct FooParser : public MasterParser {
//   FooParser()
//       : MasterParser({Id::kBar, new BoolParser},     // std::pair<*> types
//                      {Id::kBaz, new BoolParser}) {}  // omitted for brevity.
// };
//
// However, if FooParser is implemented via MasterValueParser<Foo>, then the
// boolean values will be parsed into a Foo object that can be retrieved from
// FooParser via its value() and mutable_value() methods.
//
// struct FooParser : public MasterValueParser<Foo> {
//   FooParser()
//       : MasterValueParser(MakeChild<BoolParser>(Id::kBar, &Foo::bar),
//                           MakeChild<BoolParser>(Id::kBaz, &Foo::baz)) {}
// };
template <typename T>
class MasterValueParser : public ElementParser {
 public:
  Status Init(const ElementMetadata& metadata,
              std::uint64_t max_size) override {
    assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

    PreInit();

    const Status status = master_parser_.Init(metadata, max_size);
    if (!status.completed_ok()) {
      return status;
    }

    return status;
  }

  void InitAfterSeek(const Ancestory& child_ancestory,
                     const ElementMetadata& child_metadata) override {
    PreInit();
    started_done_ = true;
    master_parser_.InitAfterSeek(child_ancestory, child_metadata);
  }

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override {
    assert(callback != nullptr);
    assert(reader != nullptr);
    assert(num_bytes_read != nullptr);

    *num_bytes_read = 0;

    if (!parse_complete_) {
      // TODO(mjbshaw): just call Reader::Skip if element's size is known and
      // action is skip.
      SkipCallback skip_callback;
      if (action_ == Action::kSkip) {
        callback = &skip_callback;
      }

      Status status = master_parser_.Feed(callback, reader, num_bytes_read);
      // Check if we've artificially injected an error code, and if so, switch
      // into skipping mode.
      if (status.code == Status::kSwitchToSkip) {
        assert(started_done_);
        assert(action_ == Action::kSkip);
        callback = &skip_callback;
        std::uint64_t local_num_bytes_read;
        status = master_parser_.Feed(callback, reader, &local_num_bytes_read);
        *num_bytes_read += local_num_bytes_read;
      }
      if (!status.completed_ok()) {
        return status;
      }
      parse_complete_ = true;
    }

    if (!started_done_) {
      Status status = OnParseStarted(callback, &action_);
      if (!status.completed_ok()) {
        return status;
      }
      started_done_ = true;
    }

    if (action_ != Action::kSkip) {
      return OnParseCompleted(callback);
    }

    return Status(Status::kOkCompleted);
  }

  bool GetCachedMetadata(ElementMetadata* metadata) override {
    return master_parser_.GetCachedMetadata(metadata);
  }

  bool WasSkipped() const override { return action_ == Action::kSkip; }

  const T& value() const { return value_; }

  T* mutable_value() { return &value_; }

 protected:
  // Users and subclasses are not meant to use the Tag* classes; they're an
  // internal implementation detail.
  // A tag that will cause the internal ChildParser to use the parsed child as a
  // start event.
  struct TagUseAsStart {};
  // A tag that will cause the internal ChildParser to call OnChildParsed once
  // it has been fully parsed.
  struct TagNotifyOnParseComplete {};

  // A factory that will create a std::pair<Id, std::unique_ptr<ElementParser>>.
  // Users and subclasses are not meant to use this class directly, as it is an
  // internal implementation detail of this class. Subclasses should use
  // MakeChild instead of using this class directly.
  template <typename Parser, typename Value, typename... Tags>
  class SingleChildFactory {
   public:
    constexpr SingleChildFactory(Id id, Element<Value> T::* member)
        : id_(id), member_(member) {}

    // Builds a std::pair<Id, std::unique_ptr<ElementParser>>. The parent
    // pointer must be a pointer to the MasterValueParser that is being
    // constructed. The given value pointer must be the pointer to the fully
    // constructed MasterValueParser::value_ object.
    std::pair<Id, std::unique_ptr<ElementParser>> BuildParser(
        MasterValueParser* parent, T* value) {
      assert(parent != nullptr);
      assert(value != nullptr);

      Element<Value>* child_member = &(value->*member_);
      auto lambda = [child_member](Parser* parser) {
        child_member->Set(std::move(*parser->mutable_value()), true);
      };
      return {id_, MakeChildParser<Parser, Value, Tags...>(
                       parent, std::move(lambda), child_member)};
    }

    // If called, OnParseStarted will be called on the parent element when this
    // particular element is encountered.
    constexpr SingleChildFactory<Parser, Value, TagUseAsStart, Tags...>
    UseAsStartEvent() const {
      return {id_, member_};
    }

    // If called, OnChildParsed will be called on the parent element when this
    // particular element is fully parsed.
    constexpr SingleChildFactory<Parser, Value, TagNotifyOnParseComplete,
                                 Tags...>
    NotifyOnParseComplete() const {
      return {id_, member_};
    }

   private:
    Id id_;
    Element<Value> T::* member_;
  };

  template <typename Parser, typename Value, typename... Tags>
  class RepeatedChildFactory {
   public:
    constexpr RepeatedChildFactory(Id id,
                                   std::vector<Element<Value>> T::* member)
        : id_(id), member_(member) {}

    // Builds a std::pair<Id, std::unique_ptr<ElementParser>>. The parent
    // pointer must be a pointer to the MasterValueParser that is being
    // constructed. The given value pointer must be the pointer to the fully
    // constructed MasterValueParser::value_ object.
    std::pair<Id, std::unique_ptr<ElementParser>> BuildParser(
        MasterValueParser* parent, T* value) {
      assert(parent != nullptr);
      assert(value != nullptr);

      std::vector<Element<Value>>* child_member = &(value->*member_);
      auto lambda = [child_member](Parser* parser) {
        if (child_member->size() == 1 && !child_member->front().is_present()) {
          child_member->clear();
        }
        child_member->emplace_back(std::move(*parser->mutable_value()), true);
      };
      return {id_, MakeChildParser<Parser, Value, Tags...>(
                       parent, std::move(lambda), child_member)};
    }

    // If called, OnParseStarted will be called on the parent element when this
    // particular element is encountered.
    constexpr RepeatedChildFactory<Parser, Value, TagUseAsStart, Tags...>
    UseAsStartEvent() const {
      return {id_, member_};
    }

    // If called, OnChildParsed will be called on the parent element when this
    // particular element is fully parsed.
    constexpr RepeatedChildFactory<Parser, Value, TagNotifyOnParseComplete,
                                   Tags...>
    NotifyOnParseComplete() const {
      return {id_, member_};
    }

   private:
    Id id_;
    std::vector<Element<Value>> T::* member_;
  };

  template <typename Parser, typename... Tags>
  class RecursiveChildFactory {
   public:
    constexpr RecursiveChildFactory(Id id, std::vector<Element<T>> T::* member,
                                    std::size_t max_recursion_depth)
        : id_(id), member_(member), max_recursion_depth_(max_recursion_depth) {}

    // Builds a std::pair<Id, std::unique_ptr<ElementParser>>. The parent
    // pointer must be a pointer to the MasterValueParser that is being
    // constructed. The given value pointer must be the pointer to the fully
    // constructed MasterValueParser::value_ object.
    std::pair<Id, std::unique_ptr<ElementParser>> BuildParser(
        MasterValueParser* parent, T* value) {
      assert(parent != nullptr);
      assert(value != nullptr);

      std::vector<Element<T>>* child_member = &(value->*member_);
      auto lambda = [child_member](RecursiveParser<Parser>* parser) {
        if (child_member->size() == 1 && !child_member->front().is_present()) {
          child_member->clear();
        }
        child_member->emplace_back(std::move(*parser->mutable_value()), true);
      };

      return {id_, std::unique_ptr<ElementParser>(
                       new ChildParser<RecursiveParser<Parser>,
                                       decltype(lambda), Tags...>(
                           parent, std::move(lambda), max_recursion_depth_))};
    }

    // If called, OnParseStarted will be called on the parent element when this
    // particular element is encountered.
    constexpr RecursiveChildFactory<Parser, TagUseAsStart, Tags...>
    UseAsStartEvent() const {
      return {id_, member_, max_recursion_depth_};
    }

    // If called, OnChildParsed will be called on the parent element when this
    // particular element is fully parsed.
    constexpr RecursiveChildFactory<Parser, TagNotifyOnParseComplete, Tags...>
    NotifyOnParseComplete() const {
      return {id_, member_, max_recursion_depth_};
    }

   private:
    Id id_;
    std::vector<Element<T>> T::* member_;
    std::size_t max_recursion_depth_;
  };

  // Constructs a new parser. Each argument must be a *ChildFactory, constructed
  // from the MakeChild method.
  template <typename... Args>
  explicit MasterValueParser(Args&&... args)
      : master_parser_(args.BuildParser(this, &value_)...) {}

  // Returns a factory that will produce a
  // std::pair<Id, std::unique_ptr<ElementParser>>. When a child element of the
  // given ID is encountered, a parser of type Parser will be used to parse it,
  // and store its value in the member pointer when the parse is complete. The
  // given default value will be used in the event that the child element has a
  // zero size. This method is only meant to be used by subclasses to provide
  // the necessary factories to the constructor.
  template <typename Parser, typename Value>
  static SingleChildFactory<Parser, Value> MakeChild(
      Id id, Element<Value> T::* member) {
    static_assert(std::is_base_of<ElementParser, Parser>::value,
                  "Parser must derive from ElementParser");
    static_assert(!std::is_base_of<MasterValueParser<T>, Parser>::value,
                  "Recursive elements should be contained in a std::vector");
    return SingleChildFactory<Parser, Value>(id, member);
  }

  template <typename Parser, typename Value>
  static RepeatedChildFactory<Parser, Value> MakeChild(
      Id id, std::vector<Element<Value>> T::* member) {
    static_assert(std::is_base_of<ElementParser, Parser>::value,
                  "Parser must derive from ElementParser");
    static_assert(!std::is_base_of<MasterValueParser<T>, Parser>::value,
                  "Recursive elements require a maximum recursion depth");
    return RepeatedChildFactory<Parser, Value>(id, member);
  }

  template <typename Parser>
  static RecursiveChildFactory<Parser> MakeChild(
      Id id, std::vector<Element<T>> T::* member,
      std::size_t max_recursion_depth) {
    static_assert(std::is_base_of<MasterValueParser<T>, Parser>::value,
                  "Child must be recusrive to use maximum recursion depth");
    return RecursiveChildFactory<Parser>(id, member, max_recursion_depth);
  }

  // Gets the metadata for this element, setting the EBML element ID to id. Only
  // call after Init() has been called.
  ElementMetadata metadata(Id id) const {
    return {id, master_parser_.header_size(), master_parser_.size(),
            master_parser_.position()};
  }

  // This method will be called once the element has been fully parsed, or a
  // particular child element of interest (see UseAsStartEvent()) is
  // encountered. By default it just sets *action to Action::kRead and returns
  // Status::kOkCompleted. May be overridden (i.e. in order to call a Callback
  // method). Returning anything other than Status::kOkCompleted will stop
  // parsing and the status to be returned by Init or Feed (whichever originated
  // the call to this method). In this case, resuming parsing will result in
  // this method being called again.
  virtual Status OnParseStarted(Callback* callback, Action* action) {
    assert(callback != nullptr);
    assert(action != nullptr);
    *action = Action::kRead;
    return Status(Status::kOkCompleted);
  }

  // This method is the companion to OnParseStarted, and will only be called
  // after OnParseStarted has already been called and parsing has completed.
  // This will not be called if OnParseStarted set the action to Action::kSkip.
  // By default it just returns Status::kOkCompleted. Returning anything other
  // than Status::kOkCompleted will stop parsing and the status to be returned
  // by Init or Feed (whichever originated the call to this method). In this
  // case, resuming parsing will result in this method being called again.
  virtual Status OnParseCompleted(Callback* callback) {
    assert(callback != nullptr);
    return Status(Status::kOkCompleted);
  }

  // Returns true if the OnParseStarted method has already been called and has
  // completed.
  bool parse_started_event_completed() const { return started_done_; }

  // Derived classes may manually call OnParseStarted before calling Feed, in
  // which case this method should be called to inform this class that
  // OnParseStarted has already been called and it should not be called again.
  void set_parse_started_event_completed_with_action(Action action) {
    assert(!started_done_);

    action_ = action;
    started_done_ = true;
  }

  // This method will be called for each child element that has been fully
  // parsed for which NotifyOnParseComplete() was requested. The provided
  // metadata is for the child element that has just completed parsing. By
  // default this method does nothing.
  virtual void OnChildParsed(const ElementMetadata& /* metadata */) {}

 private:
  T value_;
  Action action_ = Action::kRead;
  bool parse_complete_;
  bool started_done_;
  // master_parser_ must be after value_ to ensure correct initialization order.
  MasterParser master_parser_;

  const ElementMetadata& child_metadata() const {
    return master_parser_.child_metadata();
  }

  // Helper struct that will be std::true_type if Tag is in Tags, or
  // std::false_type otherwise.
  template <typename Tag, typename... Tags>
  struct HasTag;

  // Base condition: Tags is empty, so it trivially does not contain Tag.
  template <typename Tag>
  struct HasTag<Tag> : std::false_type {};

  // If the head of the Tags list is a different tag, skip it and check the
  // remaining tags.
  template <typename Tag, typename DifferentTag, typename... Tags>
  struct HasTag<Tag, DifferentTag, Tags...> : HasTag<Tag, Tags...> {};

  // If the head of the Tags list is the same as Tag, then we're done.
  template <typename Tag, typename... Tags>
  struct HasTag<Tag, Tag, Tags...> : std::true_type {};

  template <typename Base, typename F, typename... Tags>
  class ChildParser : public Base {
   public:
    using Base::WasSkipped;

    template <typename... Args>
    explicit ChildParser(MasterValueParser* parent, F consume_element_value,
                         Args&&... base_args)
        : Base(std::forward<Args>(base_args)...),
          parent_(parent),
          consume_element_value_(std::move(consume_element_value)) {}

    ChildParser() = delete;
    ChildParser(const ChildParser&) = delete;
    ChildParser& operator=(const ChildParser&) = delete;

    Status Feed(Callback* callback, Reader* reader,
                std::uint64_t* num_bytes_read) override {
      *num_bytes_read = 0;

      Status status = Prepare(callback);
      if (!status.completed_ok()) {
        return status;
      }

      status = Base::Feed(callback, reader, num_bytes_read);
      if (status.completed_ok() && parent_->action_ != Action::kSkip &&
          !WasSkipped()) {
        consume_element_value_(this);
        if (has_tag<TagNotifyOnParseComplete>()) {
          parent_->OnChildParsed(parent_->child_metadata());
        }
      }
      return status;
    }

   private:
    MasterValueParser* parent_;
    F consume_element_value_;

    Status Prepare(Callback* callback) {
      if (has_tag<TagUseAsStart>() && !parent_->started_done_) {
        const Status status =
            parent_->OnParseStarted(callback, &parent_->action_);
        if (!status.completed_ok()) {
          return status;
        }
        parent_->started_done_ = true;
        if (parent_->action_ == Action::kSkip) {
          return Status(Status::kSwitchToSkip);
        }
      }
      return Status(Status::kOkCompleted);
    }

    template <typename Tag>
    constexpr static bool has_tag() {
      return MasterValueParser::HasTag<Tag, Tags...>::value;
    }
  };

  // Returns a std::unique_ptr<ElementParser> that points to a ChildParser
  // when the Parser's constructor does not take a Value parameter.
  template <typename Parser, typename Value, typename... Tags, typename F>
  static typename std::enable_if<!std::is_constructible<Parser, Value>::value,
                                 std::unique_ptr<ElementParser>>::type
  MakeChildParser(MasterValueParser* parent, F consume_element_value, ...) {
    return std::unique_ptr<ElementParser>(new ChildParser<Parser, F, Tags...>(
        parent, std::move(consume_element_value)));
  }

  // Returns a std::unique_ptr<ElementParser> that points to a ChildParser
  // when the Parser's constructor does take a Value parameter.
  template <typename Parser, typename Value, typename... Tags, typename F>
  static typename std::enable_if<std::is_constructible<Parser, Value>::value,
                                 std::unique_ptr<ElementParser>>::type
  MakeChildParser(MasterValueParser* parent, F consume_element_value,
                  const Element<Value>* default_value) {
    return std::unique_ptr<ElementParser>(new ChildParser<Parser, F, Tags...>(
        parent, std::move(consume_element_value), default_value->value()));
  }

  // Returns a std::unique_ptr<ElementParser> that points to a ChildParser
  // when the Parser's constructor does take a Value parameter.
  template <typename Parser, typename Value, typename... Tags, typename F>
  static typename std::enable_if<std::is_constructible<Parser, Value>::value,
                                 std::unique_ptr<ElementParser>>::type
  MakeChildParser(MasterValueParser* parent, F consume_element_value,
                  const std::vector<Element<Value>>* member) {
    Value default_value{};
    if (!member->empty()) {
      default_value = member->front().value();
    }
    return std::unique_ptr<ElementParser>(new ChildParser<Parser, F, Tags...>(
        parent, std::move(consume_element_value), std::move(default_value)));
  }

  // Initializes object state. Call immediately before initializing
  // master_parser_.
  void PreInit() {
    value_ = {};
    action_ = Action::kRead;
    parse_complete_ = false;
    started_done_ = false;
  }
};

}  // namespace webm

#endif  // SRC_MASTER_VALUE_PARSER_H_
