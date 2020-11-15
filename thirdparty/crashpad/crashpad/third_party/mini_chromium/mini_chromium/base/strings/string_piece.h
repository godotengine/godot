// Copyright 2006-2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_STRINGS_STRING_PIECE_H_
#define MINI_CHROMIUM_BASE_STRINGS_STRING_PIECE_H_

#include <algorithm>
#include <iterator>
#include <ostream>
#include <string>

#include "base/strings/string16.h"

namespace base {

template<typename StringType>
class BasicStringPiece {
 public:
  typedef typename StringType::traits_type traits_type;
  typedef typename StringType::value_type value_type;
  typedef typename StringType::size_type size_type;
  typedef typename StringType::difference_type difference_type;
  typedef const value_type& reference;
  typedef const value_type& const_reference;
  typedef const value_type* pointer;
  typedef const value_type* const_pointer;
  typedef const value_type* const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  static const size_type npos;

  BasicStringPiece()
      : pointer_(NULL),
        length_(0) {
  }

  BasicStringPiece(const value_type* string)
      : pointer_(string),
        length_((string == NULL) ? 0 : traits_type::length(string)) {
  }

  BasicStringPiece(const StringType& string)
      : pointer_(string.data()),
        length_(string.size()) {
  }

  BasicStringPiece(const value_type* offset, size_type length)
      : pointer_(offset),
        length_(length) {
  }

  BasicStringPiece(const typename StringType::const_iterator& begin,
                   const typename StringType::const_iterator& end)
      : pointer_((end > begin) ? &(*begin) : NULL),
        length_((end > begin) ? static_cast<size_type>(end - begin) : 0) {
  }

  value_type operator[](size_type index) const { return pointer_[index]; }

  const value_type* data() const { return pointer_; }

  const_iterator begin() const { return pointer_; }
  const_iterator end() const { return pointer_ + length_; }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(pointer_ + length_);
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(pointer_);
  }

  bool empty() const { return length_ == 0; }
  size_type size() const { return length_; }
  size_type length() const { return length_; }
  size_type max_size() const { return length_; }
  size_type capacity() const { return length_; }

  static int wordmemcmp(const value_type* p,
                        const value_type* p2,
                        size_type N) {
    return StringType::traits_type::compare(p, p2, N);
  }

  void clear() {
    pointer_ = NULL;
    length_ = 0;
  }

  int compare(const BasicStringPiece<StringType>& that) const {
    int result = traits_type::compare(pointer_,
                                      that.pointer_,
                                      std::min(length_, that.length_));
    if (result == 0) {
      if (length_ < that.length_) {
        result = -1;
      } else if (length_ > that.length_) {
        result = 1;
      }
    }
    return result;
  }

  BasicStringPiece<StringType> substr(size_type position = 0,
                                      size_type count = npos) const {
    position = std::min(position, size());
    count = std::min(count, size() - position);
    return BasicStringPiece<StringType>(data() + position, count);
  }

  size_type copy(value_type* dest,
                 size_type count,
                 size_type position = 0) const {
    size_type ret = std::min(size() - position, count);
    traits_type::copy(dest, data() + position, ret);
    return ret;
  }

  size_type find(const BasicStringPiece<StringType>& str, size_type pos) const {
    if (pos >= size()) {
      return npos;
    }
    const_iterator result = std::search(begin() + pos,
                                        end(),
                                        str.begin(),
                                        str.end());
    size_type xpos = static_cast<size_type>(result - begin());
    return xpos + str.size() <= size() ? xpos : npos;
  }

  size_type find(value_type c, size_type pos) const {
    if (pos >= size()) {
      return npos;
    }
    const_iterator result = std::find(begin() + pos, end(), c);
    return result != end() ? static_cast<size_type>(result - begin()) : npos;
  }

  void set(const value_type* string) {
    pointer_ = string;
    length_ = string ? traits_type::length(string) : 0;
  }

  StringType as_string() const {
    return empty() ? StringType() : StringType(data(), size());
  }

 private:
  const value_type* pointer_;
  size_type length_;
};

template <typename StringType>
const typename BasicStringPiece<StringType>::size_type
    BasicStringPiece<StringType>::npos = StringType::npos;

template<typename StringType>
std::ostream& operator<<(std::ostream& ostream,
                         const BasicStringPiece<StringType>& string_piece) {
  ostream.write(string_piece.data(), string_piece.size());
  return ostream;
}

typedef BasicStringPiece<std::string> StringPiece;
typedef BasicStringPiece<string16> StringPiece16;

inline bool operator==(const StringPiece& x, const StringPiece& y) {
  if (x.size() != y.size())
    return false;

  return StringPiece::wordmemcmp(x.data(), y.data(), x.size()) == 0;
}

inline bool operator==(const StringPiece16& x, const StringPiece16& y) {
  if (x.size() != y.size())
    return false;

  return StringPiece16::wordmemcmp(x.data(), y.data(), x.size()) == 0;
}

// This hash function is copied from base/strings/string16.h. We don't use the
// ones already defined for string and string16 directly because it would
// require the string constructors to be called, which we don't want.
#define HASH_STRING_PIECE(StringPieceType, string_piece)         \
  std::size_t result = 0;                                        \
  for (StringPieceType::const_iterator i = string_piece.begin(); \
       i != string_piece.end(); ++i)                             \
    result = (result * 131) + *i;                                \
  return result;

struct StringPieceHash {
  std::size_t operator()(const StringPiece& sp) const {
    HASH_STRING_PIECE(StringPiece, sp);
  }
};
struct StringPiece16Hash {
  std::size_t operator()(const StringPiece16& sp16) const {
    HASH_STRING_PIECE(StringPiece16, sp16);
  }
};

}  // namespace base;

#endif  // MINI_CHROMIUM_BASE_STRINGS_STRING_PIECE_H_
