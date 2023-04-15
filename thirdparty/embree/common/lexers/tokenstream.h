// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "stream.h"
#include <string>
#include <vector>

namespace embree
{
  /*! token class */
  class Token
  {
  public:

    enum Type { TY_EOF, TY_CHAR, TY_INT, TY_FLOAT, TY_IDENTIFIER, TY_STRING, TY_SYMBOL };

    Token (        const ParseLocation& loc = ParseLocation()) : ty(TY_EOF  ),       loc(loc) {}
    Token (char c, const ParseLocation& loc = ParseLocation()) : ty(TY_CHAR ), c(c), loc(loc) {}
    Token (int i,  const ParseLocation& loc = ParseLocation()) : ty(TY_INT  ), i(i), loc(loc) {}
    Token (float f,const ParseLocation& loc = ParseLocation()) : ty(TY_FLOAT), f(f), loc(loc) {}
    Token (std::string str, Type ty, const ParseLocation& loc = ParseLocation()) : ty(ty),   str(str), loc(loc) {}

    static Token Eof()                { return Token(); }
    static Token Sym(std::string str) { return Token(str,TY_SYMBOL); }
    static Token Str(std::string str) { return Token(str,TY_STRING); }
    static Token Id (std::string str) { return Token(str,TY_IDENTIFIER); }

    char Char() const {
      if (ty == TY_CHAR) return c;
      THROW_RUNTIME_ERROR(loc.str()+": character expected");
    }

    int Int() const {
      if (ty == TY_INT) return i;
      THROW_RUNTIME_ERROR(loc.str()+": integer expected");
    }

    float Float(bool cast = true)  const {
      if (ty == TY_FLOAT) return f;
      if (ty == TY_INT && cast) return (float)i;
      THROW_RUNTIME_ERROR(loc.str()+": float expected");
    }

    std::string Identifier() const {
      if (ty == TY_IDENTIFIER) return str;
      THROW_RUNTIME_ERROR(loc.str()+": identifier expected");
    }

    std::string String() const {
      if (ty == TY_STRING) return str;
      THROW_RUNTIME_ERROR(loc.str()+": string expected");
    }

    std::string Symbol() const {
      if (ty == TY_SYMBOL) return str;
      THROW_RUNTIME_ERROR(loc.str()+": symbol expected");
    }

    const ParseLocation& Location() const { return loc; }

    friend bool operator==(const Token& a, const Token& b)
    {
      if (a.ty != b.ty) return false;
      if (a.ty == TY_CHAR) return a.c == b.c;
      if (a.ty == TY_INT) return a.i == b.i;
      if (a.ty == TY_FLOAT) return a.f == b.f;
      if (a.ty == TY_IDENTIFIER) return a.str == b.str;
      if (a.ty == TY_STRING) return a.str == b.str;
      if (a.ty == TY_SYMBOL) return a.str == b.str;
      return true;
    }

    friend bool operator!=(const Token& a, const Token& b) {
      return !(a == b);
    }

    friend bool operator <( const Token& a, const Token& b ) {
      if (a.ty != b.ty) return (int)a.ty < (int)b.ty;
      if (a.ty == TY_CHAR) return a.c < b.c;
      if (a.ty == TY_INT) return a.i < b.i;
      if (a.ty == TY_FLOAT) return a.f < b.f;
      if (a.ty == TY_IDENTIFIER) return a.str < b.str;
      if (a.ty == TY_STRING) return a.str < b.str;
      if (a.ty == TY_SYMBOL) return a.str < b.str;
      return false;
    }

    friend std::ostream& operator<<(std::ostream& cout, const Token& t)
    {
      if (t.ty == TY_EOF) return cout << "eof";
      if (t.ty == TY_CHAR) return cout << "Char(" << t.c << ")";
      if (t.ty == TY_INT) return cout << "Int(" << t.i << ")";
      if (t.ty == TY_FLOAT) return cout << "Float(" << t.f << ")";
      if (t.ty == TY_IDENTIFIER) return cout << "Id(" << t.str << ")";
      if (t.ty == TY_STRING) return cout << "String(" << t.str << ")";
      if (t.ty == TY_SYMBOL) return cout << "Symbol(" << t.str << ")";
      return cout << "unknown";
    }

  private:
    Type ty;            //< the type of the token
    union {
      char c;           //< data for char tokens
      int i;            //< data for int tokens
      float f;          //< data for float tokens
    };
    std::string str;    //< data for string and identifier tokens
    ParseLocation loc;  //< the location the token is from
  };

  /*! build full tokenizer that takes list of valid characters and keywords */
  class TokenStream : public Stream<Token>
  {
  public:

    /*! shorthands for common sets of characters */
    static const std::string alpha;
    static const std::string ALPHA;
    static const std::string numbers;
    static const std::string separators;
    static const std::string stringChars;

  public:
    TokenStream(const Ref<Stream<int> >& cin,
                const std::string& alpha, //< valid characters for identifiers
                const std::string& seps,  //< characters that act as separators
                const std::vector<std::string>& symbols = std::vector<std::string>()); //< symbols
  public:
    ParseLocation location() { return cin->loc(); }
    Token next();
    bool trySymbol(const std::string& symbol);

  private:
    void skipSeparators();
    bool decDigits(std::string& str);
    bool decDigits1(std::string& str);
    bool trySymbols(Token& token, const ParseLocation& loc);
    bool tryFloat(Token& token, const ParseLocation& loc);
    bool tryInt(Token& token, const ParseLocation& loc);
    bool tryString(Token& token, const ParseLocation& loc);
    bool tryIdentifier(Token& token, const ParseLocation& loc);

    Ref<Stream<int> > cin;
    bool isSepMap[256];
    bool isAlphaMap[256];
    bool isStringCharMap[256];
    std::vector<std::string> symbols;

    /*! checks if a character is a separator */
    __forceinline bool isSeparator(unsigned int c) const { return c<256 && isSepMap[c]; }

    /*! checks if a character is a number */
    __forceinline bool isDigit(unsigned int c) const {  return c >= '0' && c <= '9'; }

    /*! checks if a character is valid inside a string */
    __forceinline bool isStringChar(unsigned int c) const { return c<256 && isStringCharMap[c]; }

    /*! checks if a character is legal for an identifier */
    __forceinline bool isAlpha(unsigned int c) const {  return c<256 && isAlphaMap[c];  }
    __forceinline bool isAlphaNum(unsigned int c) const { return isAlpha(c) || isDigit(c); }
  };
}
