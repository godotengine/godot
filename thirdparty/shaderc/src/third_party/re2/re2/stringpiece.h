// Copyright 2001-2010 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef RE2_STRINGPIECE_H_
#define RE2_STRINGPIECE_H_

// A string-like object that points to a sized piece of memory.
//
// Functions or methods may use const StringPiece& parameters to accept either
// a "const char*" or a "string" value that will be implicitly converted to
// a StringPiece.  The implicit conversion means that it is often appropriate
// to include this .h file in other files rather than forward-declaring
// StringPiece as would be appropriate for most other Google classes.
//
// Systematic usage of StringPiece is encouraged as it will reduce unnecessary
// conversions from "const char*" to "string" and back again.
//
//
// Arghh!  I wish C++ literals were "string".

// Doing this simplifies the logic below.
#ifndef __has_include
#define __has_include(x) 0
#endif

#include <stddef.h>
#include <string.h>
#include <algorithm>
#include <iosfwd>
#include <iterator>
#include <string>
#if __has_include(<string_view>) && __cplusplus >= 201703L
#include <string_view>
#endif

namespace re2 {

class StringPiece {
 public:
  typedef std::char_traits<char> traits_type;
  typedef char value_type;
  typedef char* pointer;
  typedef const char* const_pointer;
  typedef char& reference;
  typedef const char& const_reference;
  typedef const char* const_iterator;
  typedef const_iterator iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef const_reverse_iterator reverse_iterator;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  static const size_type npos = static_cast<size_type>(-1);

  // We provide non-explicit singleton constructors so users can pass
  // in a "const char*" or a "string" wherever a "StringPiece" is
  // expected.
  StringPiece()
      : data_(NULL), size_(0) {}
#if __has_include(<string_view>) && __cplusplus >= 201703L
  StringPiece(const std::string_view& str)
      : data_(str.data()), size_(str.size()) {}
#endif
  StringPiece(const std::string& str)
      : data_(str.data()), size_(str.size()) {}
  StringPiece(const char* str)
      : data_(str), size_(str == NULL ? 0 : strlen(str)) {}
  StringPiece(const char* str, size_type len)
      : data_(str), size_(len) {}

  const_iterator begin() const { return data_; }
  const_iterator end() const { return data_ + size_; }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(data_ + size_);
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(data_);
  }

  size_type size() const { return size_; }
  size_type length() const { return size_; }
  bool empty() const { return size_ == 0; }

  const_reference operator[](size_type i) const { return data_[i]; }
  const_pointer data() const { return data_; }

  void remove_prefix(size_type n) {
    data_ += n;
    size_ -= n;
  }

  void remove_suffix(size_type n) {
    size_ -= n;
  }

  void set(const char* str) {
    data_ = str;
    size_ = str == NULL ? 0 : strlen(str);
  }

  void set(const char* str, size_type len) {
    data_ = str;
    size_ = len;
  }

  // Converts to `std::basic_string`.
  template <typename A>
  explicit operator std::basic_string<char, traits_type, A>() const {
    if (!data_) return {};
    return std::basic_string<char, traits_type, A>(data_, size_);
  }

  std::string as_string() const {
    return std::string(data_, size_);
  }

  // We also define ToString() here, since many other string-like
  // interfaces name the routine that converts to a C++ string
  // "ToString", and it's confusing to have the method that does that
  // for a StringPiece be called "as_string()".  We also leave the
  // "as_string()" method defined here for existing code.
  std::string ToString() const {
    return std::string(data_, size_);
  }

  void CopyToString(std::string* target) const {
    target->assign(data_, size_);
  }

  void AppendToString(std::string* target) const {
    target->append(data_, size_);
  }

  size_type copy(char* buf, size_type n, size_type pos = 0) const;
  StringPiece substr(size_type pos = 0, size_type n = npos) const;

  int compare(const StringPiece& x) const {
    size_type min_size = std::min(size(), x.size());
    if (min_size > 0) {
      int r = memcmp(data(), x.data(), min_size);
      if (r < 0) return -1;
      if (r > 0) return 1;
    }
    if (size() < x.size()) return -1;
    if (size() > x.size()) return 1;
    return 0;
  }

  // Does "this" start with "x"?
  bool starts_with(const StringPiece& x) const {
    return x.empty() ||
           (size() >= x.size() && memcmp(data(), x.data(), x.size()) == 0);
  }

  // Does "this" end with "x"?
  bool ends_with(const StringPiece& x) const {
    return x.empty() ||
           (size() >= x.size() &&
            memcmp(data() + (size() - x.size()), x.data(), x.size()) == 0);
  }

  bool contains(const StringPiece& s) const {
    return find(s) != npos;
  }

  size_type find(const StringPiece& s, size_type pos = 0) const;
  size_type find(char c, size_type pos = 0) const;
  size_type rfind(const StringPiece& s, size_type pos = npos) const;
  size_type rfind(char c, size_type pos = npos) const;

 private:
  const_pointer data_;
  size_type size_;
};

inline bool operator==(const StringPiece& x, const StringPiece& y) {
  StringPiece::size_type len = x.size();
  if (len != y.size()) return false;
  return x.data() == y.data() || len == 0 ||
         memcmp(x.data(), y.data(), len) == 0;
}

inline bool operator!=(const StringPiece& x, const StringPiece& y) {
  return !(x == y);
}

inline bool operator<(const StringPiece& x, const StringPiece& y) {
  StringPiece::size_type min_size = std::min(x.size(), y.size());
  int r = min_size == 0 ? 0 : memcmp(x.data(), y.data(), min_size);
  return (r < 0) || (r == 0 && x.size() < y.size());
}

inline bool operator>(const StringPiece& x, const StringPiece& y) {
  return y < x;
}

inline bool operator<=(const StringPiece& x, const StringPiece& y) {
  return !(x > y);
}

inline bool operator>=(const StringPiece& x, const StringPiece& y) {
  return !(x < y);
}

// Allow StringPiece to be logged.
std::ostream& operator<<(std::ostream& o, const StringPiece& p);

}  // namespace re2

#endif  // RE2_STRINGPIECE_H_
