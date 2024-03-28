// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tokenstream.h"
#include "../math/emath.h"

namespace embree
{
  /* shorthands for common sets of characters */
  const std::string TokenStream::alpha = "abcdefghijklmnopqrstuvwxyz";
  const std::string TokenStream::ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const std::string TokenStream::numbers = "0123456789";
  const std::string TokenStream::separators = "\n\t\r ";
  const std::string TokenStream::stringChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _.,+-=:/*\\";

  /* creates map for fast categorization of characters */
  static void createCharMap(bool map[256], const std::string& chrs) {
    for (size_t i=0; i<256; i++) map[i] = false;
    for (size_t i=0; i<chrs.size(); i++) map[uint8_t(chrs[i])] = true;
  }

  /* build full tokenizer that takes list of valid characters and keywords */
  TokenStream::TokenStream(const Ref<Stream<int> >& cin,            //< stream to read from
                                   const std::string& alpha,                //< valid characters for identifiers
                                   const std::string& seps,                 //< characters that act as separators
                                   const std::vector<std::string>& symbols) //< symbols
    : cin(cin), symbols(symbols)
  {
    createCharMap(isAlphaMap,alpha);
    createCharMap(isSepMap,seps);
    createCharMap(isStringCharMap,stringChars);
  }

  bool TokenStream::decDigits(std::string& str_o)
  {
    bool ok = false;
    std::string str;
    if (cin->peek() == '+' || cin->peek() == '-') str += (char)cin->get();
    while (isDigit(cin->peek())) { ok = true; str += (char)cin->get(); }
    if (ok) str_o += str;
    else cin->unget(str.size());
    return ok;
  }

  bool TokenStream::decDigits1(std::string& str_o)
  {
    bool ok = false;
    std::string str;
    while (isDigit(cin->peek())) { ok = true; str += (char)cin->get(); }
    if (ok) str_o += str; else cin->unget(str.size());
    return ok;
  }

  bool TokenStream::trySymbol(const std::string& symbol)
  {
    size_t pos = 0;
    while (pos < symbol.size()) {
      if (symbol[pos] != cin->peek()) { cin->unget(pos); return false; }
      cin->drop(); pos++;
    }
    return true;
  }

  bool TokenStream::trySymbols(Token& token, const ParseLocation& loc)
  {
    for (size_t i=0; i<symbols.size(); i++) {
      if (!trySymbol(symbols[i])) continue;
      token = Token(symbols[i],Token::TY_SYMBOL,loc);
      return true;
    }
    return false;
  }

  bool TokenStream::tryFloat(Token& token, const ParseLocation& loc)
  {
    bool ok = false;
    std::string str;
    if (trySymbol("nan")) {
      token = Token(float(nan));
      return true;
    }
    if (trySymbol("+inf")) {
      token = Token(float(pos_inf));
      return true;
    }
    if (trySymbol("-inf")) {
      token = Token(float(neg_inf));
      return true;
    }

    if (decDigits(str))
    {
      if (cin->peek() == '.') {
        str += (char)cin->get();
        decDigits(str);
        if (cin->peek() == 'e' || cin->peek() == 'E') {
          str += (char)cin->get();
          if (decDigits(str)) ok = true; // 1.[2]E2
        }
        else ok = true; // 1.[2]
      }
      else if (cin->peek() == 'e' || cin->peek() == 'E') {
        str += (char)cin->get();
        if (decDigits(str)) ok = true; // 1E2
      }
    }
    else
    {
      if (cin->peek() == '.') {
        str += (char)cin->get();
        if (decDigits(str)) {
          if (cin->peek() == 'e' || cin->peek() == 'E') {
            str += (char)cin->get();
            if (decDigits(str)) ok = true; // .3E2
          }
          else ok = true; // .3
        }
      }
    }
    if (ok) {
      token = Token((float)atof(str.c_str()),loc);
    }
    else cin->unget(str.size());
    return ok;
  }

  bool TokenStream::tryInt(Token& token, const ParseLocation& loc) {
    std::string str;
    if (decDigits(str)) {
      token = Token(atoi(str.c_str()),loc);
      return true;
    }
    return false;
  }

  bool TokenStream::tryString(Token& token, const ParseLocation& loc)
  {
    std::string str;
    if (cin->peek() != '\"') return false;
    cin->drop();
    while (cin->peek() != '\"') {
      const int c = cin->get();
      if (!isStringChar(c)) THROW_RUNTIME_ERROR("invalid string character "+std::string(1,c)+" at "+loc.str());
      str += (char)c;
    }
    cin->drop();
    token = Token(str,Token::TY_STRING,loc);
    return true;
  }

  bool TokenStream::tryIdentifier(Token& token, const ParseLocation& loc)
  {
    std::string str;
    if (!isAlpha(cin->peek())) return false;
    str += (char)cin->get();
    while (isAlphaNum(cin->peek())) str += (char)cin->get();
    token = Token(str,Token::TY_IDENTIFIER,loc);
    return true;
  }

  void TokenStream::skipSeparators()
  {
    /* skip separators */
    while (cin->peek() != EOF && isSeparator(cin->peek()))
      cin->drop();
  }

  Token TokenStream::next()
  {
    Token token;
    skipSeparators();
    ParseLocation loc = cin->loc();
    if (trySymbols   (token,loc)) return token;      /**< try to parse a symbol */
    if (tryFloat     (token,loc)) return token;      /**< try to parse float */
    if (tryInt       (token,loc)) return token;      /**< try to parse integer */
    if (tryString    (token,loc)) return token;      /**< try to parse string */
    if (tryIdentifier(token,loc)) return token;      /**< try to parse identifier */
    if (cin->peek() == EOF  )     return Token(loc); /**< return EOF token */
    return Token((char)cin->get(),loc);              /**< return invalid character token */
  }
}
