// Copyright 1999-2005 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdarg.h>
#include <stdio.h>

#include "util/strutil.h"

#ifdef _WIN32
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#endif

namespace re2 {

// ----------------------------------------------------------------------
// CEscapeString()
//    Copies 'src' to 'dest', escaping dangerous characters using
//    C-style escape sequences.  'src' and 'dest' should not overlap.
//    Returns the number of bytes written to 'dest' (not including the \0)
//    or (size_t)-1 if there was insufficient space.
// ----------------------------------------------------------------------
static size_t CEscapeString(const char* src, size_t src_len,
                            char* dest, size_t dest_len) {
  const char* src_end = src + src_len;
  size_t used = 0;

  for (; src < src_end; src++) {
    if (dest_len - used < 2)   // space for two-character escape
      return (size_t)-1;

    unsigned char c = *src;
    switch (c) {
      case '\n': dest[used++] = '\\'; dest[used++] = 'n';  break;
      case '\r': dest[used++] = '\\'; dest[used++] = 'r';  break;
      case '\t': dest[used++] = '\\'; dest[used++] = 't';  break;
      case '\"': dest[used++] = '\\'; dest[used++] = '\"'; break;
      case '\'': dest[used++] = '\\'; dest[used++] = '\''; break;
      case '\\': dest[used++] = '\\'; dest[used++] = '\\'; break;
      default:
        // Note that if we emit \xNN and the src character after that is a hex
        // digit then that digit must be escaped too to prevent it being
        // interpreted as part of the character code by C.
        if (c < ' ' || c > '~') {
          if (dest_len - used < 5)   // space for four-character escape + \0
            return (size_t)-1;
          snprintf(dest + used, 5, "\\%03o", c);
          used += 4;
        } else {
          dest[used++] = c; break;
        }
    }
  }

  if (dest_len - used < 1)   // make sure that there is room for \0
    return (size_t)-1;

  dest[used] = '\0';   // doesn't count towards return value though
  return used;
}

// ----------------------------------------------------------------------
// CEscape()
//    Copies 'src' to result, escaping dangerous characters using
//    C-style escape sequences.  'src' and 'dest' should not overlap.
// ----------------------------------------------------------------------
std::string CEscape(const StringPiece& src) {
  const size_t dest_len = src.size() * 4 + 1; // Maximum possible expansion
  char* dest = new char[dest_len];
  const size_t used = CEscapeString(src.data(), src.size(),
                                    dest, dest_len);
  std::string s = std::string(dest, used);
  delete[] dest;
  return s;
}

void PrefixSuccessor(std::string* prefix) {
  // We can increment the last character in the string and be done
  // unless that character is 255, in which case we have to erase the
  // last character and increment the previous character, unless that
  // is 255, etc. If the string is empty or consists entirely of
  // 255's, we just return the empty string.
  while (!prefix->empty()) {
    char& c = prefix->back();
    if (c == '\xff') {  // char literal avoids signed/unsigned.
      prefix->pop_back();
    } else {
      ++c;
      break;
    }
  }
}

static void StringAppendV(std::string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer
  char space[1024];

  // It's possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(space, sizeof(space), format, backup_ap);
  va_end(backup_ap);

  if ((result >= 0) && (static_cast<size_t>(result) < sizeof(space))) {
    // It fit
    dst->append(space, result);
    return;
  }

  // Repeatedly increase buffer size until it fits
  int length = sizeof(space);
  while (true) {
    if (result < 0) {
      // Older behavior: just try doubling the buffer size
      length *= 2;
    } else {
      // We need exactly "result+1" characters
      length = result+1;
    }
    char* buf = new char[length];

    // Restore the va_list before we use it again
    va_copy(backup_ap, ap);
    result = vsnprintf(buf, length, format, backup_ap);
    va_end(backup_ap);

    if ((result >= 0) && (result < length)) {
      // It fit
      dst->append(buf, result);
      delete[] buf;
      return;
    }
    delete[] buf;
  }
}

std::string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  std::string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  return result;
}

void SStringPrintf(std::string* dst, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  dst->clear();
  StringAppendV(dst, format, ap);
  va_end(ap);
}

void StringAppendF(std::string* dst, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  StringAppendV(dst, format, ap);
  va_end(ap);
}

}  // namespace re2
