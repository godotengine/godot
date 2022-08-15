//===--- StringRef.h - Constant String Reference Wrapper --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGREF_H
#define LLVM_ADT_STRINGREF_H

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace llvm {
  template <typename T>
  class SmallVectorImpl;
  class APInt;
  class hash_code;
  class StringRef;

  /// Helper functions for StringRef::getAsInteger.
  bool getAsUnsignedInteger(StringRef Str, unsigned Radix,
                            unsigned long long &Result);

  bool getAsSignedInteger(StringRef Str, unsigned Radix, long long &Result);

  /// StringRef - Represent a constant reference to a string, i.e. a character
  /// array and a length, which need not be null terminated.
  ///
  /// This class does not own the string data, it is expected to be used in
  /// situations where the character data resides in some other buffer, whose
  /// lifetime extends past that of the StringRef. For this reason, it is not in
  /// general safe to store a StringRef.
  class StringRef {
  public:
    typedef const char *iterator;
    typedef const char *const_iterator;
    static const size_t npos = ~size_t(0);
    typedef size_t size_type;

  private:
    /// The start of the string, in an external buffer.
    const char *Data;

    /// The length of the string.
    size_t Length;

    // Workaround memcmp issue with null pointers (undefined behavior)
    // by providing a specialized version
    static int compareMemory(const char *Lhs, const char *Rhs, size_t Length) {
      if (Length == 0) { return 0; }
      return ::memcmp(Lhs,Rhs,Length);
    }

  public:
    /// @name Constructors
    /// @{

    /// Construct an empty string ref.
    /*implicit*/ StringRef() : Data(nullptr), Length(0) {}
    StringRef(std::nullptr_t) = delete; // HLSL Change - So we don't accidentally pass `false` again

    /// Construct a string ref from a cstring.
    /*implicit*/ StringRef(const char *Str)
      : Data(Str) {
        assert(Str && "StringRef cannot be built from a NULL argument");
        Length = ::strlen(Str); // invoking strlen(NULL) is undefined behavior
      }

    /// Construct a string ref from a pointer and length.
    /*implicit*/ StringRef(const char *data, size_t length)
      : Data(data), Length(length) {
        assert((data || length == 0) &&
        "StringRef cannot be built from a NULL argument with non-null length");
      }

    /// Construct a string ref from an std::string.
    /*implicit*/ StringRef(const std::string &Str)
      : Data(Str.data()), Length(Str.length()) {}

    /// @}
    /// @name Iterators
    /// @{

    iterator begin() const { return Data; }

    iterator end() const { return Data + Length; }

    const unsigned char *bytes_begin() const {
      return reinterpret_cast<const unsigned char *>(begin());
    }
    const unsigned char *bytes_end() const {
      return reinterpret_cast<const unsigned char *>(end());
    }

    /// @}
    /// @name String Operations
    /// @{

    /// data - Get a pointer to the start of the string (which may not be null
    /// terminated).
    const char *data() const { return Data; }

    /// empty - Check if the string is empty.
    bool empty() const { return Length == 0; }

    /// size - Get the string size.
    size_t size() const { return Length; }

    /// front - Get the first character in the string.
    char front() const {
      assert(!empty());
      return Data[0];
    }

    /// back - Get the last character in the string.
    char back() const {
      assert(!empty());
      return Data[Length-1];
    }

    // copy - Allocate copy in Allocator and return StringRef to it.
    template <typename Allocator> StringRef copy(Allocator &A) const {
      char *S = A.template Allocate<char>(Length);
      std::copy(begin(), end(), S);
      return StringRef(S, Length);
    }

    /// equals - Check for string equality, this is more efficient than
    /// compare() when the relative ordering of inequal strings isn't needed.
    bool equals(StringRef RHS) const {
      return (Length == RHS.Length &&
              compareMemory(Data, RHS.Data, RHS.Length) == 0);
    }

    /// equals_lower - Check for string equality, ignoring case.
    bool equals_lower(StringRef RHS) const {
      return Length == RHS.Length && compare_lower(RHS) == 0;
    }

    /// compare - Compare two strings; the result is -1, 0, or 1 if this string
    /// is lexicographically less than, equal to, or greater than the \p RHS.
    int compare(StringRef RHS) const {
      // Check the prefix for a mismatch.
      if (int Res = compareMemory(Data, RHS.Data, std::min(Length, RHS.Length)))
        return Res < 0 ? -1 : 1;

      // Otherwise the prefixes match, so we only need to check the lengths.
      if (Length == RHS.Length)
        return 0;
      return Length < RHS.Length ? -1 : 1;
    }

    /// compare_lower - Compare two strings, ignoring case.
    int compare_lower(StringRef RHS) const;

    /// compare_numeric - Compare two strings, treating sequences of digits as
    /// numbers.
    int compare_numeric(StringRef RHS) const;

    /// \brief Determine the edit distance between this string and another
    /// string.
    ///
    /// \param Other the string to compare this string against.
    ///
    /// \param AllowReplacements whether to allow character
    /// replacements (change one character into another) as a single
    /// operation, rather than as two operations (an insertion and a
    /// removal).
    ///
    /// \param MaxEditDistance If non-zero, the maximum edit distance that
    /// this routine is allowed to compute. If the edit distance will exceed
    /// that maximum, returns \c MaxEditDistance+1.
    ///
    /// \returns the minimum number of character insertions, removals,
    /// or (if \p AllowReplacements is \c true) replacements needed to
    /// transform one of the given strings into the other. If zero,
    /// the strings are identical.
    unsigned edit_distance(StringRef Other, bool AllowReplacements = true,
                           unsigned MaxEditDistance = 0) const;

    /// str - Get the contents as an std::string.
    std::string str() const {
      if (!Data) return std::string();
      return std::string(Data, Length);
    }

    /// @}
    /// @name Operator Overloads
    /// @{

    char operator[](size_t Index) const {
      assert(Index < Length && "Invalid index!");
      return Data[Index];
    }

    /// @}
    /// @name Type Conversions
    /// @{

    operator std::string() const {
      return str();
    }

    /// @}
    /// @name String Predicates
    /// @{

    /// Check if this string starts with the given \p Prefix.
    bool startswith(StringRef Prefix) const {
      return Length >= Prefix.Length &&
             compareMemory(Data, Prefix.Data, Prefix.Length) == 0;
    }

    /// Check if this string starts with the given \p Prefix, ignoring case.
    bool startswith_lower(StringRef Prefix) const;

    /// Check if this string ends with the given \p Suffix.
    bool endswith(StringRef Suffix) const {
      return Length >= Suffix.Length &&
        compareMemory(end() - Suffix.Length, Suffix.Data, Suffix.Length) == 0;
    }

    /// Check if this string ends with the given \p Suffix, ignoring case.
    bool endswith_lower(StringRef Suffix) const;

    /// @}
    /// @name String Searching
    /// @{

    /// Search for the first character \p C in the string.
    ///
    /// \returns The index of the first occurrence of \p C, or npos if not
    /// found.
    size_t find(char C, size_t From = 0) const {
      size_t FindBegin = std::min(From, Length);
      if (FindBegin < Length) { // Avoid calling memchr with nullptr.
        // Just forward to memchr, which is faster than a hand-rolled loop.
        if (const void *P = ::memchr(Data + FindBegin, C, Length - FindBegin))
          return static_cast<const char *>(P) - Data;
      }
      return npos;
    }

    /// Search for the first string \p Str in the string.
    ///
    /// \returns The index of the first occurrence of \p Str, or npos if not
    /// found.
    size_t find(StringRef Str, size_t From = 0) const;

    /// Search for the last character \p C in the string.
    ///
    /// \returns The index of the last occurrence of \p C, or npos if not
    /// found.
    size_t rfind(char C, size_t From = npos) const {
      From = std::min(From, Length);
      size_t i = From;
      while (i != 0) {
        --i;
        if (Data[i] == C)
          return i;
      }
      return npos;
    }

    /// Search for the last string \p Str in the string.
    ///
    /// \returns The index of the last occurrence of \p Str, or npos if not
    /// found.
    size_t rfind(StringRef Str) const;

    /// Find the first character in the string that is \p C, or npos if not
    /// found. Same as find.
    size_t find_first_of(char C, size_t From = 0) const {
      return find(C, From);
    }

    /// Find the first character in the string that is in \p Chars, or npos if
    /// not found.
    ///
    /// Complexity: O(size() + Chars.size())
    size_t find_first_of(StringRef Chars, size_t From = 0) const;

    /// Find the first character in the string that is not \p C or npos if not
    /// found.
    size_t find_first_not_of(char C, size_t From = 0) const;

    /// Find the first character in the string that is not in the string
    /// \p Chars, or npos if not found.
    ///
    /// Complexity: O(size() + Chars.size())
    size_t find_first_not_of(StringRef Chars, size_t From = 0) const;

    /// Find the last character in the string that is \p C, or npos if not
    /// found.
    size_t find_last_of(char C, size_t From = npos) const {
      return rfind(C, From);
    }

    /// Find the last character in the string that is in \p C, or npos if not
    /// found.
    ///
    /// Complexity: O(size() + Chars.size())
    size_t find_last_of(StringRef Chars, size_t From = npos) const;

    /// Find the last character in the string that is not \p C, or npos if not
    /// found.
    size_t find_last_not_of(char C, size_t From = npos) const;

    /// Find the last character in the string that is not in \p Chars, or
    /// npos if not found.
    ///
    /// Complexity: O(size() + Chars.size())
    size_t find_last_not_of(StringRef Chars, size_t From = npos) const;

    /// @}
    /// @name Helpful Algorithms
    /// @{

    /// Return the number of occurrences of \p C in the string.
    size_t count(char C) const {
      size_t Count = 0;
      for (size_t i = 0, e = Length; i != e; ++i)
        if (Data[i] == C)
          ++Count;
      return Count;
    }

    /// Return the number of non-overlapped occurrences of \p Str in
    /// the string.
    size_t count(StringRef Str) const;

    /// Parse the current string as an integer of the specified radix.  If
    /// \p Radix is specified as zero, this does radix autosensing using
    /// extended C rules: 0 is octal, 0x is hex, 0b is binary.
    ///
    /// If the string is invalid or if only a subset of the string is valid,
    /// this returns true to signify the error.  The string is considered
    /// erroneous if empty or if it overflows T.
    template <typename T>
    typename std::enable_if<std::numeric_limits<T>::is_signed, bool>::type
    getAsInteger(unsigned Radix, T &Result) const {
      long long LLVal;
      if (getAsSignedInteger(*this, Radix, LLVal) ||
            static_cast<T>(LLVal) != LLVal)
        return true;
      Result = LLVal;
      return false;
    }

    template <typename T>
    typename std::enable_if<!std::numeric_limits<T>::is_signed, bool>::type
    getAsInteger(unsigned Radix, T &Result) const {
      unsigned long long ULLVal;
      // The additional cast to unsigned long long is required to avoid the
      // Visual C++ warning C4805: '!=' : unsafe mix of type 'bool' and type
      // 'unsigned __int64' when instantiating getAsInteger with T = bool.
      if (getAsUnsignedInteger(*this, Radix, ULLVal) ||
          static_cast<unsigned long long>(static_cast<T>(ULLVal)) != ULLVal)
        return true;
      Result = ULLVal;
      return false;
    }

    /// Parse the current string as an integer of the specified \p Radix, or of
    /// an autosensed radix if the \p Radix given is 0.  The current value in
    /// \p Result is discarded, and the storage is changed to be wide enough to
    /// store the parsed integer.
    ///
    /// \returns true if the string does not solely consist of a valid
    /// non-empty number in the appropriate base.
    ///
    /// APInt::fromString is superficially similar but assumes the
    /// string is well-formed in the given radix.
    bool getAsInteger(unsigned Radix, APInt &Result) const;

    /// @}
    /// @name String Operations
    /// @{

    // Convert the given ASCII string to lowercase.
    std::string lower() const;

    /// Convert the given ASCII string to uppercase.
    std::string upper() const;

    /// @}
    /// @name Substring Operations
    /// @{

    /// Return a reference to the substring from [Start, Start + N).
    ///
    /// \param Start The index of the starting character in the substring; if
    /// the index is npos or greater than the length of the string then the
    /// empty substring will be returned.
    ///
    /// \param N The number of characters to included in the substring. If N
    /// exceeds the number of characters remaining in the string, the string
    /// suffix (starting with \p Start) will be returned.
    StringRef substr(size_t Start, size_t N = npos) const {
      Start = std::min(Start, Length);
      return StringRef(Data + Start, std::min(N, Length - Start));
    }

    /// Return a StringRef equal to 'this' but with the first \p N elements
    /// dropped.
    StringRef drop_front(size_t N = 1) const {
      assert(size() >= N && "Dropping more elements than exist");
      return substr(N);
    }

    /// Return a StringRef equal to 'this' but with the last \p N elements
    /// dropped.
    StringRef drop_back(size_t N = 1) const {
      assert(size() >= N && "Dropping more elements than exist");
      return substr(0, size()-N);
    }

    /// Return a reference to the substring from [Start, End).
    ///
    /// \param Start The index of the starting character in the substring; if
    /// the index is npos or greater than the length of the string then the
    /// empty substring will be returned.
    ///
    /// \param End The index following the last character to include in the
    /// substring. If this is npos, or less than \p Start, or exceeds the
    /// number of characters remaining in the string, the string suffix
    /// (starting with \p Start) will be returned.
    StringRef slice(size_t Start, size_t End) const {
      Start = std::min(Start, Length);
      End = std::min(std::max(Start, End), Length);
      return StringRef(Data + Start, End - Start);
    }

    /// Split into two substrings around the first occurrence of a separator
    /// character.
    ///
    /// If \p Separator is in the string, then the result is a pair (LHS, RHS)
    /// such that (*this == LHS + Separator + RHS) is true and RHS is
    /// maximal. If \p Separator is not in the string, then the result is a
    /// pair (LHS, RHS) where (*this == LHS) and (RHS == "").
    ///
    /// \param Separator The character to split on.
    /// \returns The split substrings.
    std::pair<StringRef, StringRef> split(char Separator) const {
      size_t Idx = find(Separator);
      if (Idx == npos)
        return std::make_pair(*this, StringRef());
      return std::make_pair(slice(0, Idx), slice(Idx+1, npos));
    }

    /// Split into two substrings around the first occurrence of a separator
    /// string.
    ///
    /// If \p Separator is in the string, then the result is a pair (LHS, RHS)
    /// such that (*this == LHS + Separator + RHS) is true and RHS is
    /// maximal. If \p Separator is not in the string, then the result is a
    /// pair (LHS, RHS) where (*this == LHS) and (RHS == "").
    ///
    /// \param Separator - The string to split on.
    /// \return - The split substrings.
    std::pair<StringRef, StringRef> split(StringRef Separator) const {
      size_t Idx = find(Separator);
      if (Idx == npos)
        return std::make_pair(*this, StringRef());
      return std::make_pair(slice(0, Idx), slice(Idx + Separator.size(), npos));
    }

    /// Split into substrings around the occurrences of a separator string.
    ///
    /// Each substring is stored in \p A. If \p MaxSplit is >= 0, at most
    /// \p MaxSplit splits are done and consequently <= \p MaxSplit
    /// elements are added to A.
    /// If \p KeepEmpty is false, empty strings are not added to \p A. They
    /// still count when considering \p MaxSplit
    /// An useful invariant is that
    /// Separator.join(A) == *this if MaxSplit == -1 and KeepEmpty == true
    ///
    /// \param A - Where to put the substrings.
    /// \param Separator - The string to split on.
    /// \param MaxSplit - The maximum number of times the string is split.
    /// \param KeepEmpty - True if empty substring should be added.
    void split(SmallVectorImpl<StringRef> &A,
               StringRef Separator, int MaxSplit = -1,
               bool KeepEmpty = true) const;

    /// Split into two substrings around the last occurrence of a separator
    /// character.
    ///
    /// If \p Separator is in the string, then the result is a pair (LHS, RHS)
    /// such that (*this == LHS + Separator + RHS) is true and RHS is
    /// minimal. If \p Separator is not in the string, then the result is a
    /// pair (LHS, RHS) where (*this == LHS) and (RHS == "").
    ///
    /// \param Separator - The character to split on.
    /// \return - The split substrings.
    std::pair<StringRef, StringRef> rsplit(char Separator) const {
      size_t Idx = rfind(Separator);
      if (Idx == npos)
        return std::make_pair(*this, StringRef());
      return std::make_pair(slice(0, Idx), slice(Idx+1, npos));
    }

    /// Return string with consecutive characters in \p Chars starting from
    /// the left removed.
    StringRef ltrim(StringRef Chars = " \t\n\v\f\r") const {
      return drop_front(std::min(Length, find_first_not_of(Chars)));
    }

    /// Return string with consecutive characters in \p Chars starting from
    /// the right removed.
    StringRef rtrim(StringRef Chars = " \t\n\v\f\r") const {
      return drop_back(Length - std::min(Length, find_last_not_of(Chars) + 1));
    }

    /// Return string with consecutive characters in \p Chars starting from
    /// the left and right removed.
    StringRef trim(StringRef Chars = " \t\n\v\f\r") const {
      return ltrim(Chars).rtrim(Chars);
    }

    /// @}
  };

  /// @name StringRef Comparison Operators
  /// @{

  inline bool operator==(StringRef LHS, StringRef RHS) {
    return LHS.equals(RHS);
  }

  inline bool operator!=(StringRef LHS, StringRef RHS) {
    return !(LHS == RHS);
  }

  inline bool operator<(StringRef LHS, StringRef RHS) {
    return LHS.compare(RHS) == -1;
  }

  inline bool operator<=(StringRef LHS, StringRef RHS) {
    return LHS.compare(RHS) != 1;
  }

  inline bool operator>(StringRef LHS, StringRef RHS) {
    return LHS.compare(RHS) == 1;
  }

  inline bool operator>=(StringRef LHS, StringRef RHS) {
    return LHS.compare(RHS) != -1;
  }

  inline std::string &operator+=(std::string &buffer, StringRef string) {
    return buffer.append(string.data(), string.size());
  }

  /// @}

  /// \brief Compute a hash_code for a StringRef.
  hash_code hash_value(StringRef S);

  // StringRefs can be treated like a POD type.
  template <typename T> struct isPodLike;
  template <> struct isPodLike<StringRef> { static const bool value = true; };
}

// HLSL Change Starts
// StringRef provides an operator string; that trips up the std::pair noexcept specification,
// which (a) enables the moves constructor (because conversion is allowed), but (b)
// misclassifies the the construction as nothrow.
namespace std {
  template<>
  struct is_nothrow_constructible <std::string, llvm::StringRef>
    : std::false_type {
  };
  template<>
  struct is_nothrow_constructible <std::string, llvm::StringRef &>
    : std::false_type {
  };
  template<>
  struct is_nothrow_constructible <std::string, const llvm::StringRef &>
    : std::false_type {
  };
}
// HLSL Change Ends

#endif
