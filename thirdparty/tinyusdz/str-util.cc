// SPDX-License-Identifier: Apache 2.0
// Copyright 2023 - Present, Light Transport Entertainment, Inc.
#include "str-util.hh"

#include "unicode-xid.hh"
#include "common-macros.inc"

namespace tinyusdz {

std::string buildEscapedAndQuotedStringForUSDA(const std::string &str) {
  // Rule for triple quote string:
  //
  // if str contains newline
  //   if str contains """ and '''
  //      use quote """ and escape " to \\", no escape for '''
  //   elif str contains """ only
  //      use quote ''' and no escape for """
  //   elif str contains ''' only
  //      use quote """ and no escape for '''
  //   else
  //      use quote """
  //
  // Rule for single quote string
  //   if str contains " and '
  //      use quote " and escape " to \\", no escape for '
  //   elif str contains " only
  //      use quote ' and no escape for "
  //   elif str contains ' only
  //      use quote " and no escape for '
  //   else
  //      use quote "

  bool has_newline = hasNewline(str);

  std::string s;

  if (has_newline) {
    bool has_triple_single_quoted_string = hasTripleQuotes(str, false);
    bool has_triple_double_quoted_string = hasTripleQuotes(str, true);

    std::string delim = "\"\"\"";
    if (has_triple_single_quoted_string && has_triple_double_quoted_string) {
      s = escapeSingleQuote(str, true);
    } else if (has_triple_single_quoted_string) {
      s = escapeSingleQuote(str, false);
    } else if (has_triple_double_quoted_string) {
      delim = "'''";
      s = str;
    } else {
      s = str;
    }

    s = quote(escapeControlSequence(s), delim);

  } else {
    // single quote string.
    bool has_single_quote = hasQuotes(str, false);
    bool has_double_quote = hasQuotes(str, true);

    std::string delim = "\"";
    if (has_single_quote && has_double_quote) {
      s = escapeSingleQuote(str, true);
    } else if (has_single_quote) {
      s = escapeSingleQuote(str, false);
    } else if (has_double_quote) {
      delim = "'";
      s = str;
    } else {
      s = str;
    }

    s = quote(escapeControlSequence(s), delim);
  }

  return s;
}

std::string escapeControlSequence(const std::string &str) {
  std::string s;

  for (size_t i = 0; i < str.size(); i++) {
    if (str[i] == '\a') {
      s += "\\x07";
    } else if (str[i] == '\b') {
      s += "\\x08";
    } else if (str[i] == '\t') {
      s += "\\t";
    } else if (str[i] == '\v') {
      s += "\\x0b";
    } else if (str[i] == '\f') {
      s += "\\x0c";
    } else if (str[i] == '\\') {
      // skip escaping backshash for escaped quote string: \' \"
      if (i + 1 < str.size()) {
        if ((str[i + 1] == '"') || (str[i + 1] == '\'')) {
          s += str[i];
        } else {
          s += "\\\\";
        }
      } else {
        s += "\\\\";
      }
    } else {
      s += str[i];
    }
  }

  return s;
}

std::string unescapeControlSequence(const std::string &str) {
  std::string s;

  if (str.size() < 2) {
    return str;
  }

  for (size_t i = 0; i < str.size(); i++) {
    if (str[i] == '\\') {
      if (i + 1 < str.size()) {
        if (str[i + 1] == 'a') {
          s += '\a';
          i++;
        } else if (str[i + 1] == 'b') {
          s += '\b';
          i++;
        } else if (str[i + 1] == 't') {
          s += '\t';
          i++;
        } else if (str[i + 1] == 'v') {
          s += '\v';
          i++;
        } else if (str[i + 1] == 'f') {
          s += '\f';
          i++;
        } else if (str[i + 1] == 'n') {
          s += '\n';
          i++;
        } else if (str[i + 1] == 'r') {
          s += '\r';
          i++;
        } else if (str[i + 1] == '\\') {
          s += "\\";
        } else {
          // ignore backslash
        }
      } else {
        // ignore backslash
      }
    } else {
      s += str[i];
    }
  }

  return s;
}

bool hasQuotes(const std::string &str, bool is_double_quote) {
  for (size_t i = 0; i < str.size(); i++) {
    if (is_double_quote) {
      if (str[i] == '"') {
        return true;
      }
    } else {
      if (str[i] == '\'') {
        return true;
      }
    }
  }

  return false;
}

bool hasTripleQuotes(const std::string &str, bool is_double_quote) {
  for (size_t i = 0; i < str.size(); i++) {
    if (i + 3 < str.size()) {
      if (is_double_quote) {
        if ((str[i + 0] == '"') && (str[i + 1] == '"') && (str[i + 2] == '"')) {
          return true;
        }
      } else {
        if ((str[i + 0] == '\'') && (str[i + 1] == '\'') &&
            (str[i + 2] == '\'')) {
          return true;
        }
      }
    }
  }

  return false;
}

bool hasEscapedTripleQuotes(const std::string &str, bool is_double_quote,
                            size_t *n) {
  size_t count = 0;

  for (size_t i = 0; i < str.size(); i++) {
    if (str[i] == '\\') {
      if (i + 3 < str.size()) {
        if (is_double_quote) {
          if ((str[i + 1] == '"') && (str[i + 2] == '"') &&
              (str[i + 3] == '"')) {
            if (!n) {  // early exit
              return true;
            }

            count++;
            i += 3;
          }
        } else {
          if ((str[i + 1] == '\'') && (str[i + 2] == '\'') &&
              (str[i + 3] == '\'')) {
            if (!n) {  // early exit
              return true;
            }
            count++;
            i += 3;
          }
        }
      }
    }
  }

  if (n) {
    (*n) = count;
  }

  return count > 0;
}

std::string escapeSingleQuote(const std::string &str,
                              const bool is_double_quote) {
  std::string s;

  if (is_double_quote) {
    for (size_t i = 0; i < str.size(); i++) {
      if (str[i] == '"') {
        s += "\\\"";
      } else {
        s += str[i];
      }
    }
  } else {
    for (size_t i = 0; i < str.size(); i++) {
      if (str[i] == '\'') {
        s += "\\'";
      } else {
        s += str[i];
      }
    }
  }

  return s;
}

std::string escapeBackslash(const std::string &str,
                            const bool triple_quoted_string) {
  if (triple_quoted_string) {
    std::string s;

    // Do not escape \""" or \'''

    for (size_t i = 0; i < str.size(); i++) {
      if (str[i] == '\\') {
        if (i + 3 < str.size()) {
          if ((str[i + 1] == '\'') && (str[i + 2] == '\'') &&
              (str[i + 3] == '\'')) {
            s += "\\'''";
            i += 3;
          } else if ((str[i + 1] == '"') && (str[i + 2] == '"') &&
                     (str[i + 3] == '"')) {
            s += "\\\"\"\"";
            i += 3;
          } else {
            s += "\\\\";
          }
        } else {
          s += "\\\\";
        }
      } else {
        s += str[i];
      }
    }

    return s;
  } else {
    const std::string bs = "\\";
    const std::string bs_escaped = "\\\\";

    std::string s = str;

    std::string::size_type pos = 0;
    while ((pos = s.find(bs, pos)) != std::string::npos) {
      s.replace(pos, bs.length(), bs_escaped);
      pos += bs_escaped.length();
    }

    return s;
  }
}

std::string unescapeBackslash(const std::string &str) {
  std::string s = str;

  std::string bs = "\\\\";
  std::string bs_unescaped = "\\";

  std::string::size_type pos = 0;
  while ((pos = s.find(bs, pos)) != std::string::npos) {
    s.replace(pos, bs.length(), bs_unescaped);
    pos += bs_unescaped.length();
  }

  return s;
}

bool tokenize_variantElement(const std::string &elementName,
                             std::array<std::string, 2> *result) {
  std::vector<std::string> toks;

  // Ensure ElementPath is quoted with '{' and '}'
  if (startsWith(elementName, "{") && endsWith(elementName, "}")) {
    // ok
  } else {
    return false;
  }

  // Remove variant quotation
  std::string name = unwrap(elementName, "{", "}");

  toks = split(name, "=");
  if (toks.size() == 1) {
    if (result) {
      // ensure '=' and newline does not exist.
      if (counts(toks[0], '=') || hasNewline(toks[0])) {
        return false;
      }

      (*result)[0] = toks[0];
      (*result)[1] = std::string();
    }
    return true;
  } else if (toks.size() == 2) {
    if (result) {
      // ensure '=' and newline does not exist.
      if (counts(toks[0], '=') || hasNewline(toks[0])) {
        return false;
      }

      if (counts(toks[1], '=') || hasNewline(toks[1])) {
        return false;
      }

      (*result)[0] = toks[0];
      (*result)[1] = toks[1];
    }
    return true;
  } else {
    return false;
  }
}

bool is_variantElementName(const std::string &name) {
  return tokenize_variantElement(name);
}

///
/// Simply add number suffix to make unique string.
///
/// - plane -> plane1
/// - sphere1 -> sphere11
/// - xform4 -> xform41
///
///
bool makeUniqueName(std::multiset<std::string> &nameSet,
                    const std::string &name, std::string *unique_name) {
  if (!unique_name) {
    return false;
  }

  if (nameSet.count(name) == 0) {
    (*unique_name) = name;
    return 0;
  }

  // Simply add number

  const size_t kMaxLoop = 1024;  // to avoid infinite loop.

  std::string new_name = name;

  size_t cnt = 0;
  while (cnt < kMaxLoop) {
    size_t i = nameSet.count(new_name);
    if (i == 0) {
      // This should not happen though.
      return false;
    }

    new_name += std::to_string(i);

    if (nameSet.count(new_name) == 0) {
      (*unique_name) = new_name;
      return true;
    }

    cnt++;
  }

  return false;
}

namespace detail {

inline uint32_t utf8_len(const unsigned char c) {
      if (c <= 127) {
        // ascii
        return 1;
      } else if ((c & 0xE0) == 0xC0) {
        return 2;
      } else if ((c & 0xF0) == 0xE0) {
        return 3;
      } else if ((c & 0xF8) == 0xF0) {
        return 4;
      }

      // invalid
      return 0;
}

inline std::string extract_utf8_char(const std::string &str, uint32_t start_i,
                                     int &len) {
  len = 0;

  if ((start_i + 1) > str.size()) {
    len = 0;
    return std::string();
  }

  unsigned char c = static_cast<unsigned char>(str[start_i]);

  if (c <= 127) {
    // ascii
    len = 1;
    return str.substr(start_i, 1);
  } else if ((c & 0xE0) == 0xC0) {
    if ((start_i + 2) > str.size()) {
      len = 0;
      return std::string();
    }
    len = 2;
    return str.substr(start_i, 2);
  } else if ((c & 0xF0) == 0xE0) {
    if ((start_i + 3) > str.size()) {
      len = 0;
      return std::string();
    }
    len = 3;
    return str.substr(start_i, 3);
  } else if ((c & 0xF8) == 0xF0) {
    if ((start_i + 4) > str.size()) {
      len = 0;
      return std::string();
    }
    len = 4;
    return str.substr(start_i, 4);
  } else {
    // invalid utf8
    len = 0;
    return std::string();
  }
}

inline uint32_t to_codepoint(const char *s, uint32_t &char_len) {
  if (!s) {
    char_len = 0;
    return ~0u;
  }

  char_len = detail::utf8_len(static_cast<unsigned char>(s[0]));
  if (char_len == 0) {
    return ~0u;
  }

  uint32_t code = 0;
  if (char_len == 1) {
    unsigned char s0 = static_cast<unsigned char>(s[0]);
    if (s0 > 0x7f) {
      return ~0u;
    }
    code = uint32_t(s0) & 0x7f;
  } else if (char_len == 2) {
    // 11bit: 110y-yyyx 10xx-xxxx
    unsigned char s0 = static_cast<unsigned char>(s[0]);
    unsigned char s1 = static_cast<unsigned char>(s[1]);

    if (((s0 & 0xe0) == 0xc0) && ((s1 & 0xc0) == 0x80)) {
      code = (uint32_t(s0 & 0x1f) << 6) | (s1 & 0x3f);
    } else {
      return ~0u;
    }
  } else if (char_len == 3) {
    // 16bit: 1110-yyyy 10yx-xxxx 10xx-xxxx
    unsigned char s0 = static_cast<unsigned char>(s[0]);
    unsigned char s1 = static_cast<unsigned char>(s[1]);
    unsigned char s2 = static_cast<unsigned char>(s[2]);
    if (((s0 & 0xf0) == 0xe0) && ((s1 & 0xc0) == 0x80) &&
        ((s2 & 0xc0) == 0x80)) {
      code =
          (uint32_t(s0 & 0xf) << 12) | (uint32_t(s1 & 0x3f) << 6) | (s2 & 0x3f);
    } else {
      return ~0u;
    }
  } else if (char_len == 4) {
    // 21bit: 1111-0yyy 10yy-xxxx 10xx-xxxx 10xx-xxxx
    unsigned char s0 = static_cast<unsigned char>(s[0]);
    unsigned char s1 = static_cast<unsigned char>(s[1]);
    unsigned char s2 = static_cast<unsigned char>(s[2]);
    unsigned char s3 = static_cast<unsigned char>(s[3]);
    if (((s0 & 0xf8) == 0xf0) && ((s1 & 0xc0) == 0x80) &&
        ((s2 & 0xc0) == 0x80) && ((s2 & 0xc0) == 0x80)) {
      code = (uint32_t(s0 & 0x7) << 18) | (uint32_t(s1 & 0x3f) << 12) |
             (uint32_t(s2 & 0x3f) << 6) | uint32_t(s3 & 0x3f);
    } else {
      return ~0u;
    }
  } else {
    // ???
    char_len = 0;
    return ~0u;
  }

  return code;
}

}  // namespace detail

std::vector<std::string> to_utf8_chars(const std::string &str) {
  std::vector<std::string> utf8_chars;
  size_t sz = str.size();

  for (size_t i = 0; i <= sz;) {
    int len = 0;
    std::string s = detail::extract_utf8_char(str, uint32_t(i), len);
    if (len == 0) {
      // invalid char
      return std::vector<std::string>();
    }

    i += uint64_t(len);
    utf8_chars.push_back(s);
  }

  return utf8_chars;
}

uint32_t to_utf8_code(const std::string &s) {
  if (s.empty() || (s.size() > 4)) {
    return ~0u;  // invalid
  }

  // TODO: endianness.
  uint32_t code = 0;
  if (s.size() == 1) {
    unsigned char s0 = static_cast<unsigned char>(s[0]);
    if (s0 > 0x7f) {
      return ~0u;
    }
    code = uint32_t(s0) & 0x7f;
  } else if (s.size() == 2) {
    // 11bit: 110y-yyyx 10xx-xxxx
    unsigned char s0 = static_cast<unsigned char>(s[0]);
    unsigned char s1 = static_cast<unsigned char>(s[1]);

    if (((s0 & 0xe0) == 0xc0) && ((s1 & 0xc0) == 0x80)) {
      code = (uint32_t(s0 & 0x1f) << 6) | (s1 & 0x3f);
    } else {
      return ~0u;
    }
  } else if (s.size() == 3) {
    // 16bit: 1110-yyyy 10yx-xxxx 10xx-xxxx
    unsigned char s0 = static_cast<unsigned char>(s[0]);
    unsigned char s1 = static_cast<unsigned char>(s[1]);
    unsigned char s2 = static_cast<unsigned char>(s[2]);
    if (((s0 & 0xf0) == 0xe0) && ((s1 & 0xc0) == 0x80) &&
        ((s2 & 0xc0) == 0x80)) {
      code =
          (uint32_t(s0 & 0xf) << 12) | (uint32_t(s1 & 0x3f) << 6) | (s2 & 0x3f);
    } else {
      return ~0u;
    }
  } else {
    // 21bit: 1111-0yyy 10yy-xxxx 10xx-xxxx 10xx-xxxx
    unsigned char s0 = static_cast<unsigned char>(s[0]);
    unsigned char s1 = static_cast<unsigned char>(s[1]);
    unsigned char s2 = static_cast<unsigned char>(s[2]);
    unsigned char s3 = static_cast<unsigned char>(s[3]);
    if (((s0 & 0xf8) == 0xf0) && ((s1 & 0xc0) == 0x80) &&
        ((s2 & 0xc0) == 0x80) && ((s2 & 0xc0) == 0x80)) {
      code = (uint32_t(s0 & 0x7) << 18) | (uint32_t(s1 & 0x3f) << 12) |
             (uint32_t(s2 & 0x3f) << 6) | uint32_t(s3 & 0x3f);
    } else {
      return ~0u;
    }
  }

  return code;
}


#if 0
std::string to_utf8_char(const uint32_t code) {

  if (code < 128) {
    std::string s = static_cast<char>(code);
    return s;
  }
  // TODO

}
#endif

bool is_valid_utf8(const std::string &str) {
  // TODO: Consider UTF-BOM?
  for (size_t i = 0; i < str.size();) {
    uint32_t len = detail::utf8_len(*reinterpret_cast<const unsigned char *>(&str[i]));
    if (len == 0) {
      return false;
    }
    i += len;
  }
  return true;
}

std::vector<uint32_t> to_codepoints(const std::string &str) {

  std::vector<uint32_t> cps;

  for (size_t i = 0; i < str.size(); ) {
    uint32_t char_len;
    uint32_t cp = detail::to_codepoint(str.c_str() + i, char_len);

    if ((cp > kMaxUTF8Codepoint) || (char_len == 0)) {
      return std::vector<uint32_t>();
    }

    cps.push_back(cp);

    i += char_len;
  }

  return cps;
}

bool is_valid_utf8_identifier(const std::string &str) {
  // First convert to codepoint values.
  std::vector<uint32_t> codepoints = to_codepoints(str);

  if (codepoints.empty()) {
    return false;
  }

  // (XID_Start|_) (XID_Continue|_)+
  
  if ((codepoints[0] != '_') && !unicode_xid::is_xid_start(codepoints[0])) {
    return false;
  }

  for (size_t i = 1; i < codepoints.size(); i++) {
    if ((codepoints[i] != '_') && !unicode_xid::is_xid_continue(codepoints[i])) {
      return false;
    }
  }

  return true; 
}

std::string makeIdentifierValid(const std::string &str, bool is_utf8) {
  // TODO: utf8 support
  (void)is_utf8;

  std::string s;

  if (str.empty()) {
    // return '_'
    return "_";
  }

  // first char
  // [a-ZA-Z_]
  if ((('a' <= str[0]) && (str[0] <= 'z')) || (('A' <= str[0]) && (str[0] <= 'Z')) || (str[0] == '_')) {
    s.push_back(str[0]);
  } else {
    s.push_back('_');
  }

  // remain chars
  // [a-ZA-Z0-9_]
  for (size_t i = 1; i < str.length(); i++) {
    if ((('a' <= str[i]) && (str[i] <= 'z')) || (('A' <= str[i]) && (str[i] <= 'Z')) || (('0' <= str[i]) && (str[i] <= '9')) || (str[i] == '_')) {
      s.push_back(str[i]);
    } else {
      s.push_back('_');
    }
  }

  return s;
}

}  // namespace tinyusdz
