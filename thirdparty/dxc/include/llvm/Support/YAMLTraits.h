//===- llvm/Support/YAMLTraits.h --------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_YAMLTRAITS_H
#define LLVM_SUPPORT_YAMLTRAITS_H


#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

namespace llvm {
namespace yaml {


/// This class should be specialized by any type that needs to be converted
/// to/from a YAML mapping.  For example:
///
///     struct MappingTraits<MyStruct> {
///       static void mapping(IO &io, MyStruct &s) {
///         io.mapRequired("name", s.name);
///         io.mapRequired("size", s.size);
///         io.mapOptional("age",  s.age);
///       }
///     };
template<class T>
struct MappingTraits {
  // Must provide:
  // static void mapping(IO &io, T &fields);
  // Optionally may provide:
  // static StringRef validate(IO &io, T &fields);
  //
  // The optional flow flag will cause generated YAML to use a flow mapping
  // (e.g. { a: 0, b: 1 }):
  // static const bool flow = true;
};


/// This class should be specialized by any integral type that converts
/// to/from a YAML scalar where there is a one-to-one mapping between
/// in-memory values and a string in YAML.  For example:
///
///     struct ScalarEnumerationTraits<Colors> {
///         static void enumeration(IO &io, Colors &value) {
///           io.enumCase(value, "red",   cRed);
///           io.enumCase(value, "blue",  cBlue);
///           io.enumCase(value, "green", cGreen);
///         }
///       };
template<typename T>
struct ScalarEnumerationTraits {
  // Must provide:
  // static void enumeration(IO &io, T &value);
};


/// This class should be specialized by any integer type that is a union
/// of bit values and the YAML representation is a flow sequence of
/// strings.  For example:
///
///      struct ScalarBitSetTraits<MyFlags> {
///        static void bitset(IO &io, MyFlags &value) {
///          io.bitSetCase(value, "big",   flagBig);
///          io.bitSetCase(value, "flat",  flagFlat);
///          io.bitSetCase(value, "round", flagRound);
///        }
///      };
template<typename T>
struct ScalarBitSetTraits {
  // Must provide:
  // static void bitset(IO &io, T &value);
};


/// This class should be specialized by type that requires custom conversion
/// to/from a yaml scalar.  For example:
///
///    template<>
///    struct ScalarTraits<MyType> {
///      static void output(const MyType &val, void*, llvm::raw_ostream &out) {
///        // stream out custom formatting
///        out << llvm::format("%x", val);
///      }
///      static StringRef input(StringRef scalar, void*, MyType &value) {
///        // parse scalar and set `value`
///        // return empty string on success, or error string
///        return StringRef();
///      }
///      static bool mustQuote(StringRef) { return true; }
///    };
template<typename T>
struct ScalarTraits {
  // Must provide:
  //
  // Function to write the value as a string:
  //static void output(const T &value, void *ctxt, llvm::raw_ostream &out);
  //
  // Function to convert a string to a value.  Returns the empty
  // StringRef on success or an error string if string is malformed:
  //static StringRef input(StringRef scalar, void *ctxt, T &value);
  //
  // Function to determine if the value should be quoted.
  //static bool mustQuote(StringRef);
};


/// This class should be specialized by type that requires custom conversion
/// to/from a YAML literal block scalar. For example:
///
///    template <>
///    struct BlockScalarTraits<MyType> {
///      static void output(const MyType &Value, void*, llvm::raw_ostream &Out)
///      {
///        // stream out custom formatting
///        Out << Val;
///      }
///      static StringRef input(StringRef Scalar, void*, MyType &Value) {
///        // parse scalar and set `value`
///        // return empty string on success, or error string
///        return StringRef();
///      }
///    };
template <typename T>
struct BlockScalarTraits {
  // Must provide:
  //
  // Function to write the value as a string:
  // static void output(const T &Value, void *ctx, llvm::raw_ostream &Out);
  //
  // Function to convert a string to a value.  Returns the empty
  // StringRef on success or an error string if string is malformed:
  // static StringRef input(StringRef Scalar, void *ctxt, T &Value);
};


/// This class should be specialized by any type that needs to be converted
/// to/from a YAML sequence.  For example:
///
///    template<>
///    struct SequenceTraits< std::vector<MyType> > {
///      static size_t size(IO &io, std::vector<MyType> &seq) {
///        return seq.size();
///      }
///      static MyType& element(IO &, std::vector<MyType> &seq, size_t index) {
///        if ( index >= seq.size() )
///          seq.resize(index+1);
///        return seq[index];
///      }
///    };
template<typename T>
struct SequenceTraits {
  // Must provide:
  // static size_t size(IO &io, T &seq);
  // static T::value_type& element(IO &io, T &seq, size_t index);
  //
  // The following is option and will cause generated YAML to use
  // a flow sequence (e.g. [a,b,c]).
  // static const bool flow = true;
};


/// This class should be specialized by any type that needs to be converted
/// to/from a list of YAML documents.
template<typename T>
struct DocumentListTraits {
  // Must provide:
  // static size_t size(IO &io, T &seq);
  // static T::value_type& element(IO &io, T &seq, size_t index);
};


// Only used by compiler if both template types are the same
template <typename T, T>
struct SameType;

// Only used for better diagnostics of missing traits
template <typename T>
struct MissingTrait;



// Test if ScalarEnumerationTraits<T> is defined on type T.
template <class T>
struct has_ScalarEnumerationTraits
{
  typedef void (*Signature_enumeration)(class IO&, T&);

  template <typename U>
  static char test(SameType<Signature_enumeration, &U::enumeration>*);

  template <typename U>
  static double test(...);

public:
  static bool const value =
    (sizeof(test<ScalarEnumerationTraits<T> >(nullptr)) == 1);
};


// Test if ScalarBitSetTraits<T> is defined on type T.
template <class T>
struct has_ScalarBitSetTraits
{
  typedef void (*Signature_bitset)(class IO&, T&);

  template <typename U>
  static char test(SameType<Signature_bitset, &U::bitset>*);

  template <typename U>
  static double test(...);

public:
  static bool const value = (sizeof(test<ScalarBitSetTraits<T> >(nullptr)) == 1);
};


// Test if ScalarTraits<T> is defined on type T.
template <class T>
struct has_ScalarTraits
{
  typedef StringRef (*Signature_input)(StringRef, void*, T&);
  typedef void (*Signature_output)(const T&, void*, llvm::raw_ostream&);
  typedef bool (*Signature_mustQuote)(StringRef);

  template <typename U>
  static char test(SameType<Signature_input, &U::input> *,
                   SameType<Signature_output, &U::output> *,
                   SameType<Signature_mustQuote, &U::mustQuote> *);

  template <typename U>
  static double test(...);

public:
  static bool const value =
      (sizeof(test<ScalarTraits<T>>(nullptr, nullptr, nullptr)) == 1);
};


// Test if BlockScalarTraits<T> is defined on type T.
template <class T>
struct has_BlockScalarTraits
{
  typedef StringRef (*Signature_input)(StringRef, void *, T &);
  typedef void (*Signature_output)(const T &, void *, llvm::raw_ostream &);

  template <typename U>
  static char test(SameType<Signature_input, &U::input> *,
                   SameType<Signature_output, &U::output> *);

  template <typename U>
  static double test(...);

public:
  static bool const value =
      (sizeof(test<BlockScalarTraits<T>>(nullptr, nullptr)) == 1);
};


// Test if MappingTraits<T> is defined on type T.
template <class T>
struct has_MappingTraits
{
  typedef void (*Signature_mapping)(class IO&, T&);

  template <typename U>
  static char test(SameType<Signature_mapping, &U::mapping>*);

  template <typename U>
  static double test(...);

public:
  static bool const value = (sizeof(test<MappingTraits<T> >(nullptr)) == 1);
};

// Test if MappingTraits<T>::validate() is defined on type T.
template <class T>
struct has_MappingValidateTraits
{
  typedef StringRef (*Signature_validate)(class IO&, T&);

  template <typename U>
  static char test(SameType<Signature_validate, &U::validate>*);

  template <typename U>
  static double test(...);

public:
  static bool const value = (sizeof(test<MappingTraits<T> >(nullptr)) == 1);
};



// Test if SequenceTraits<T> is defined on type T.
template <class T>
struct has_SequenceMethodTraits
{
  typedef size_t (*Signature_size)(class IO&, T&);

  template <typename U>
  static char test(SameType<Signature_size, &U::size>*);

  template <typename U>
  static double test(...);

public:
  static bool const value =  (sizeof(test<SequenceTraits<T> >(nullptr)) == 1);
};


// has_FlowTraits<int> will cause an error with some compilers because
// it subclasses int.  Using this wrapper only instantiates the
// real has_FlowTraits only if the template type is a class.
template <typename T, bool Enabled = std::is_class<T>::value>
class has_FlowTraits
{
public:
   static const bool value = false;
};

// Some older gcc compilers don't support straight forward tests
// for members, so test for ambiguity cause by the base and derived
// classes both defining the member.
template <class T>
struct has_FlowTraits<T, true>
{
  struct Fallback { bool flow; };
  struct Derived : T, Fallback { };

  template<typename C>
  static char (&f(SameType<bool Fallback::*, &C::flow>*))[1];

  template<typename C>
  static char (&f(...))[2];

public:
  static bool const value = sizeof(f<Derived>(nullptr)) == 2;
};



// Test if SequenceTraits<T> is defined on type T
template<typename T>
struct has_SequenceTraits : public std::integral_constant<bool,
                                      has_SequenceMethodTraits<T>::value > { };


// Test if DocumentListTraits<T> is defined on type T
template <class T>
struct has_DocumentListTraits
{
  typedef size_t (*Signature_size)(class IO&, T&);

  template <typename U>
  static char test(SameType<Signature_size, &U::size>*);

  template <typename U>
  static double test(...);

public:
  static bool const value = (sizeof(test<DocumentListTraits<T> >(nullptr))==1);
};

inline bool isNumber(StringRef S) {
  static const char OctalChars[] = "01234567";
  if (S.startswith("0") &&
      S.drop_front().find_first_not_of(OctalChars) == StringRef::npos)
    return true;

  if (S.startswith("0o") &&
      S.drop_front(2).find_first_not_of(OctalChars) == StringRef::npos)
    return true;

  static const char HexChars[] = "0123456789abcdefABCDEF";
  if (S.startswith("0x") &&
      S.drop_front(2).find_first_not_of(HexChars) == StringRef::npos)
    return true;

  static const char DecChars[] = "0123456789";
  if (S.find_first_not_of(DecChars) == StringRef::npos)
    return true;

  if (S.equals(".inf") || S.equals(".Inf") || S.equals(".INF"))
    return true;

  Regex FloatMatcher("^(\\.[0-9]+|[0-9]+(\\.[0-9]*)?)([eE][-+]?[0-9]+)?$");
  if (FloatMatcher.match(S))
    return true;

  return false;
}

inline bool isNumeric(StringRef S) {
  if ((S.front() == '-' || S.front() == '+') && isNumber(S.drop_front()))
    return true;

  if (isNumber(S))
    return true;

  if (S.equals(".nan") || S.equals(".NaN") || S.equals(".NAN"))
    return true;

  return false;
}

inline bool isNull(StringRef S) {
  return S.equals("null") || S.equals("Null") || S.equals("NULL") ||
         S.equals("~");
}

inline bool isBool(StringRef S) {
  return S.equals("true") || S.equals("True") || S.equals("TRUE") ||
         S.equals("false") || S.equals("False") || S.equals("FALSE");
}

inline bool needsQuotes(StringRef S) {
  if (S.empty())
    return true;
  if (isspace(S.front()) || isspace(S.back()))
    return true;
  if (S.front() == ',')
    return true;

  static const char ScalarSafeChars[] =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/^., \t";
  if (S.find_first_not_of(ScalarSafeChars) != StringRef::npos)
    return true;

  if (isNull(S))
    return true;
  if (isBool(S))
    return true;
  if (isNumeric(S))
    return true;

  return false;
}


template<typename T>
struct missingTraits : public std::integral_constant<bool,
                                         !has_ScalarEnumerationTraits<T>::value
                                      && !has_ScalarBitSetTraits<T>::value
                                      && !has_ScalarTraits<T>::value
                                      && !has_BlockScalarTraits<T>::value
                                      && !has_MappingTraits<T>::value
                                      && !has_SequenceTraits<T>::value
                                      && !has_DocumentListTraits<T>::value >  {};

template<typename T>
struct validatedMappingTraits : public std::integral_constant<bool,
                                       has_MappingTraits<T>::value
                                    && has_MappingValidateTraits<T>::value> {};

template<typename T>
struct unvalidatedMappingTraits : public std::integral_constant<bool,
                                        has_MappingTraits<T>::value
                                    && !has_MappingValidateTraits<T>::value> {};
// Base class for Input and Output.
class IO {
public:

  IO(void *Ctxt=nullptr);
  virtual ~IO();

  virtual bool outputting() = 0;

  virtual unsigned beginSequence() = 0;
  virtual bool preflightElement(unsigned, void *&) = 0;
  virtual void postflightElement(void*) = 0;
  virtual void endSequence() = 0;
  virtual bool canElideEmptySequence() = 0;

  virtual unsigned beginFlowSequence() = 0;
  virtual bool preflightFlowElement(unsigned, void *&) = 0;
  virtual void postflightFlowElement(void*) = 0;
  virtual void endFlowSequence() = 0;

  virtual bool mapTag(StringRef Tag, bool Default=false) = 0;
  virtual void beginMapping() = 0;
  virtual void endMapping() = 0;
  virtual bool preflightKey(const char*, bool, bool, bool &, void *&) = 0;
  virtual void postflightKey(void*) = 0;

  virtual void beginFlowMapping() = 0;
  virtual void endFlowMapping() = 0;

  virtual void beginEnumScalar() = 0;
  virtual bool matchEnumScalar(const char*, bool) = 0;
  virtual bool matchEnumFallback() = 0;
  virtual void endEnumScalar() = 0;

  virtual bool beginBitSetScalar(bool &) = 0;
  virtual bool bitSetMatch(const char*, bool) = 0;
  virtual void endBitSetScalar() = 0;

  virtual void scalarString(StringRef &, bool) = 0;
  virtual void blockScalarString(StringRef &) = 0;

  virtual void setError(const Twine &) = 0;

  template <typename T>
  void enumCase(T &Val, const char* Str, const T ConstVal) {
    if ( matchEnumScalar(Str, outputting() && Val == ConstVal) ) {
      Val = ConstVal;
    }
  }

  // allow anonymous enum values to be used with LLVM_YAML_STRONG_TYPEDEF
  template <typename T>
  void enumCase(T &Val, const char* Str, const uint32_t ConstVal) {
    if ( matchEnumScalar(Str, outputting() && Val == static_cast<T>(ConstVal)) ) {
      Val = ConstVal;
    }
  }

  template <typename FBT, typename T>
  void enumFallback(T &Val) {
    if ( matchEnumFallback() ) {
      // FIXME: Force integral conversion to allow strong typedefs to convert.
      FBT Res = (uint64_t)Val;
      yamlize(*this, Res, true);
      Val = (uint64_t)Res;
    }
  }

  template <typename T>
  void bitSetCase(T &Val, const char* Str, const T ConstVal) {
    if ( bitSetMatch(Str, outputting() && (Val & ConstVal) == ConstVal) ) {
      Val = Val | ConstVal;
    }
  }

  // allow anonymous enum values to be used with LLVM_YAML_STRONG_TYPEDEF
  template <typename T>
  void bitSetCase(T &Val, const char* Str, const uint32_t ConstVal) {
    if ( bitSetMatch(Str, outputting() && (Val & ConstVal) == ConstVal) ) {
      Val = Val | ConstVal;
    }
  }

  template <typename T>
  void maskedBitSetCase(T &Val, const char *Str, T ConstVal, T Mask) {
    if (bitSetMatch(Str, outputting() && (Val & Mask) == ConstVal))
      Val = Val | ConstVal;
  }

  template <typename T>
  void maskedBitSetCase(T &Val, const char *Str, uint32_t ConstVal,
                        uint32_t Mask) {
    if (bitSetMatch(Str, outputting() && (Val & Mask) == ConstVal))
      Val = Val | ConstVal;
  }

  void *getContext();
  void setContext(void *);

  template <typename T>
  void mapRequired(const char* Key, T& Val) {
    this->processKey(Key, Val, true);
  }

  template <typename T>
  typename std::enable_if<has_SequenceTraits<T>::value,void>::type
  mapOptional(const char* Key, T& Val) {
    // omit key/value instead of outputting empty sequence
    if ( this->canElideEmptySequence() && !(Val.begin() != Val.end()) )
      return;
    this->processKey(Key, Val, false);
  }

  template <typename T>
  void mapOptional(const char* Key, Optional<T> &Val) {
    processKeyWithDefault(Key, Val, Optional<T>(), /*Required=*/false);
  }

  template <typename T>
  typename std::enable_if<!has_SequenceTraits<T>::value,void>::type
  mapOptional(const char* Key, T& Val) {
    this->processKey(Key, Val, false);
  }

  template <typename T>
  void mapOptional(const char* Key, T& Val, const T& Default) {
    this->processKeyWithDefault(Key, Val, Default, false);
  }

private:
  template <typename T>
  void processKeyWithDefault(const char *Key, Optional<T> &Val,
                             const Optional<T> &DefaultValue, bool Required) {
    assert(DefaultValue.hasValue() == false &&
           "Optional<T> shouldn't have a value!");
    void *SaveInfo;
    bool UseDefault;
    const bool sameAsDefault = outputting() && !Val.hasValue();
    if (!outputting() && !Val.hasValue())
      Val = T();
    if (this->preflightKey(Key, Required, sameAsDefault, UseDefault,
                           SaveInfo)) {
      yamlize(*this, Val.getValue(), Required);
      this->postflightKey(SaveInfo);
    } else {
      if (UseDefault)
        Val = DefaultValue;
    }
  }

  template <typename T>
  void processKeyWithDefault(const char *Key, T &Val, const T& DefaultValue,
                                                                bool Required) {
    void *SaveInfo;
    bool UseDefault;
    const bool sameAsDefault = outputting() && Val == DefaultValue;
    if ( this->preflightKey(Key, Required, sameAsDefault, UseDefault,
                                                                  SaveInfo) ) {
      yamlize(*this, Val, Required);
      this->postflightKey(SaveInfo);
    }
    else {
      if ( UseDefault )
        Val = DefaultValue;
    }
  }

  template <typename T>
  void processKey(const char *Key, T &Val, bool Required) {
    void *SaveInfo;
    bool UseDefault;
    if ( this->preflightKey(Key, Required, false, UseDefault, SaveInfo) ) {
      yamlize(*this, Val, Required);
      this->postflightKey(SaveInfo);
    }
  }

private:
  void  *Ctxt;
};



template<typename T>
typename std::enable_if<has_ScalarEnumerationTraits<T>::value,void>::type
yamlize(IO &io, T &Val, bool) {
  io.beginEnumScalar();
  ScalarEnumerationTraits<T>::enumeration(io, Val);
  io.endEnumScalar();
}

template<typename T>
typename std::enable_if<has_ScalarBitSetTraits<T>::value,void>::type
yamlize(IO &io, T &Val, bool) {
  bool DoClear;
  if ( io.beginBitSetScalar(DoClear) ) {
    if ( DoClear )
      Val = static_cast<T>(0);
    ScalarBitSetTraits<T>::bitset(io, Val);
    io.endBitSetScalar();
  }
}


template<typename T>
typename std::enable_if<has_ScalarTraits<T>::value,void>::type
yamlize(IO &io, T &Val, bool) {
  if ( io.outputting() ) {
    std::string Storage;
    llvm::raw_string_ostream Buffer(Storage);
    ScalarTraits<T>::output(Val, io.getContext(), Buffer);
    StringRef Str = Buffer.str();
    io.scalarString(Str, ScalarTraits<T>::mustQuote(Str));
  }
  else {
    StringRef Str;
    io.scalarString(Str, ScalarTraits<T>::mustQuote(Str));
    StringRef Result = ScalarTraits<T>::input(Str, io.getContext(), Val);
    if ( !Result.empty() ) {
      io.setError(llvm::Twine(Result));
    }
  }
}

template <typename T>
typename std::enable_if<has_BlockScalarTraits<T>::value, void>::type
yamlize(IO &YamlIO, T &Val, bool) {
  if (YamlIO.outputting()) {
    std::string Storage;
    llvm::raw_string_ostream Buffer(Storage);
    BlockScalarTraits<T>::output(Val, YamlIO.getContext(), Buffer);
    StringRef Str = Buffer.str();
    YamlIO.blockScalarString(Str);
  } else {
    StringRef Str;
    YamlIO.blockScalarString(Str);
    StringRef Result =
        BlockScalarTraits<T>::input(Str, YamlIO.getContext(), Val);
    if (!Result.empty())
      YamlIO.setError(llvm::Twine(Result));
  }
}

template<typename T>
typename std::enable_if<validatedMappingTraits<T>::value, void>::type
yamlize(IO &io, T &Val, bool) {
  if (has_FlowTraits<MappingTraits<T>>::value)
    io.beginFlowMapping();
  else
    io.beginMapping();
  if (io.outputting()) {
    StringRef Err = MappingTraits<T>::validate(io, Val);
    if (!Err.empty()) {
      llvm::errs() << Err << "\n";
      assert(Err.empty() && "invalid struct trying to be written as yaml");
    }
  }
  MappingTraits<T>::mapping(io, Val);
  if (!io.outputting()) {
    StringRef Err = MappingTraits<T>::validate(io, Val);
    if (!Err.empty())
      io.setError(Err);
  }
  if (has_FlowTraits<MappingTraits<T>>::value)
    io.endFlowMapping();
  else
    io.endMapping();
}

template<typename T>
typename std::enable_if<unvalidatedMappingTraits<T>::value, void>::type
yamlize(IO &io, T &Val, bool) {
  if (has_FlowTraits<MappingTraits<T>>::value) {
    io.beginFlowMapping();
    MappingTraits<T>::mapping(io, Val);
    io.endFlowMapping();
  } else {
    io.beginMapping();
    MappingTraits<T>::mapping(io, Val);
    io.endMapping();
  }
}

template<typename T>
typename std::enable_if<missingTraits<T>::value, void>::type
yamlize(IO &io, T &Val, bool) {
  char missing_yaml_trait_for_type[sizeof(MissingTrait<T>)];
}

template<typename T>
typename std::enable_if<has_SequenceTraits<T>::value,void>::type
yamlize(IO &io, T &Seq, bool) {
  if ( has_FlowTraits< SequenceTraits<T> >::value ) {
    unsigned incnt = io.beginFlowSequence();
    unsigned count = io.outputting() ? SequenceTraits<T>::size(io, Seq) : incnt;
    for(unsigned i=0; i < count; ++i) {
      void *SaveInfo;
      if ( io.preflightFlowElement(i, SaveInfo) ) {
        yamlize(io, SequenceTraits<T>::element(io, Seq, i), true);
        io.postflightFlowElement(SaveInfo);
      }
    }
    io.endFlowSequence();
  }
  else {
    unsigned incnt = io.beginSequence();
    unsigned count = io.outputting() ? SequenceTraits<T>::size(io, Seq) : incnt;
    for(unsigned i=0; i < count; ++i) {
      void *SaveInfo;
      if ( io.preflightElement(i, SaveInfo) ) {
        yamlize(io, SequenceTraits<T>::element(io, Seq, i), true);
        io.postflightElement(SaveInfo);
      }
    }
    io.endSequence();
  }
}


template<>
struct ScalarTraits<bool> {
  static void output(const bool &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, bool &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<StringRef> {
  static void output(const StringRef &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, StringRef &);
  static bool mustQuote(StringRef S) { return needsQuotes(S); }
};

template<>
struct ScalarTraits<std::string> {
  static void output(const std::string &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, std::string &);
  static bool mustQuote(StringRef S) { return needsQuotes(S); }
};

template<>
struct ScalarTraits<uint8_t> {
  static void output(const uint8_t &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, uint8_t &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<uint16_t> {
  static void output(const uint16_t &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, uint16_t &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<uint32_t> {
  static void output(const uint32_t &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, uint32_t &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<uint64_t> {
  static void output(const uint64_t &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, uint64_t &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<int8_t> {
  static void output(const int8_t &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, int8_t &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<int16_t> {
  static void output(const int16_t &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, int16_t &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<int32_t> {
  static void output(const int32_t &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, int32_t &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<int64_t> {
  static void output(const int64_t &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, int64_t &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<float> {
  static void output(const float &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, float &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<double> {
  static void output(const double &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, double &);
  static bool mustQuote(StringRef) { return false; }
};



// Utility for use within MappingTraits<>::mapping() method
// to [de]normalize an object for use with YAML conversion.
template <typename TNorm, typename TFinal>
struct MappingNormalization {
  MappingNormalization(IO &i_o, TFinal &Obj)
      : io(i_o), BufPtr(nullptr), Result(Obj) {
    if ( io.outputting() ) {
      BufPtr = new (&Buffer) TNorm(io, Obj);
    }
    else {
      BufPtr = new (&Buffer) TNorm(io);
    }
  }

  ~MappingNormalization() {
    if ( ! io.outputting() ) {
      Result = BufPtr->denormalize(io);
    }
    BufPtr->~TNorm();
  }

  TNorm* operator->() { return BufPtr; }

private:
  typedef llvm::AlignedCharArrayUnion<TNorm> Storage;

  Storage       Buffer;
  IO           &io;
  TNorm        *BufPtr;
  TFinal       &Result;
};



// Utility for use within MappingTraits<>::mapping() method
// to [de]normalize an object for use with YAML conversion.
template <typename TNorm, typename TFinal>
struct MappingNormalizationHeap {
  MappingNormalizationHeap(IO &i_o, TFinal &Obj)
    : io(i_o), BufPtr(NULL), Result(Obj) {
    if ( io.outputting() ) {
      BufPtr = new (&Buffer) TNorm(io, Obj);
    }
    else {
      BufPtr = new TNorm(io);
    }
  }

  ~MappingNormalizationHeap() {
    if ( io.outputting() ) {
      BufPtr->~TNorm();
    }
    else {
      Result = BufPtr->denormalize(io);
    }
  }

  TNorm* operator->() { return BufPtr; }

private:
  typedef llvm::AlignedCharArrayUnion<TNorm> Storage;

  Storage       Buffer;
  IO           &io;
  TNorm        *BufPtr;
  TFinal       &Result;
};



///
/// The Input class is used to parse a yaml document into in-memory structs
/// and vectors.
///
/// It works by using YAMLParser to do a syntax parse of the entire yaml
/// document, then the Input class builds a graph of HNodes which wraps
/// each yaml Node.  The extra layer is buffering.  The low level yaml
/// parser only lets you look at each node once.  The buffering layer lets
/// you search and interate multiple times.  This is necessary because
/// the mapRequired() method calls may not be in the same order
/// as the keys in the document.
///
class Input : public IO {
public:
  // Construct a yaml Input object from a StringRef and optional
  // user-data. The DiagHandler can be specified to provide
  // alternative error reporting.
  Input(StringRef InputContent,
        void *Ctxt = nullptr,
        SourceMgr::DiagHandlerTy DiagHandler = nullptr,
        void *DiagHandlerCtxt = nullptr);
  ~Input() override;

  // Check if there was an syntax or semantic error during parsing.
  std::error_code error();

private:
  bool outputting() override;
  bool mapTag(StringRef, bool) override;
  void beginMapping() override;
  void endMapping() override;
  bool preflightKey(const char *, bool, bool, bool &, void *&) override;
  void postflightKey(void *) override;
  void beginFlowMapping() override;
  void endFlowMapping() override;
  unsigned beginSequence() override;
  void endSequence() override;
  bool preflightElement(unsigned index, void *&) override;
  void postflightElement(void *) override;
  unsigned beginFlowSequence() override;
  bool preflightFlowElement(unsigned , void *&) override;
  void postflightFlowElement(void *) override;
  void endFlowSequence() override;
  void beginEnumScalar() override;
  bool matchEnumScalar(const char*, bool) override;
  bool matchEnumFallback() override;
  void endEnumScalar() override;
  bool beginBitSetScalar(bool &) override;
  bool bitSetMatch(const char *, bool ) override;
  void endBitSetScalar() override;
  void scalarString(StringRef &, bool) override;
  void blockScalarString(StringRef &) override;
  void setError(const Twine &message) override;
  bool canElideEmptySequence() override;

  class HNode {
    virtual void anchor();
  public:
    HNode(Node *n) : _node(n) { }
    virtual ~HNode() { }
    static inline bool classof(const HNode *) { return true; }

    Node *_node;
  };

  class EmptyHNode : public HNode {
    void anchor() override;
  public:
    EmptyHNode(Node *n) : HNode(n) { }
    static inline bool classof(const HNode *n) {
      return NullNode::classof(n->_node);
    }
    static inline bool classof(const EmptyHNode *) { return true; }
  };

  class ScalarHNode : public HNode {
    void anchor() override;
  public:
    ScalarHNode(Node *n, StringRef s) : HNode(n), _value(s) { }

    StringRef value() const { return _value; }

    static inline bool classof(const HNode *n) {
      return ScalarNode::classof(n->_node) ||
             BlockScalarNode::classof(n->_node);
    }
    static inline bool classof(const ScalarHNode *) { return true; }
  protected:
    StringRef _value;
  };

  class MapHNode : public HNode {
    void anchor() override;

  public:
    MapHNode(Node *n) : HNode(n) { }

    static inline bool classof(const HNode *n) {
      return MappingNode::classof(n->_node);
    }
    static inline bool classof(const MapHNode *) { return true; }

    typedef llvm::StringMap<std::unique_ptr<HNode>> NameToNode;

    bool isValidKey(StringRef key);

    NameToNode                        Mapping;
    llvm::SmallVector<const char*, 6> ValidKeys;
  };

  class SequenceHNode : public HNode {
    void anchor() override;

  public:
    SequenceHNode(Node *n) : HNode(n) { }

    static inline bool classof(const HNode *n) {
      return SequenceNode::classof(n->_node);
    }
    static inline bool classof(const SequenceHNode *) { return true; }

    std::vector<std::unique_ptr<HNode>> Entries;
  };

  std::unique_ptr<Input::HNode> createHNodes(Node *node);
  void setError(HNode *hnode, const Twine &message);
  void setError(Node *node, const Twine &message);


public:
  // These are only used by operator>>. They could be private
  // if those templated things could be made friends.
  bool setCurrentDocument();
  bool nextDocument();

  /// Returns the current node that's being parsed by the YAML Parser.
  const Node *getCurrentNode() const;

private:
  llvm::SourceMgr                     SrcMgr; // must be before Strm
  std::unique_ptr<llvm::yaml::Stream> Strm;
  std::unique_ptr<HNode>              TopNode;
  std::error_code                     EC;
  llvm::BumpPtrAllocator              StringAllocator;
  llvm::yaml::document_iterator       DocIterator;
  std::vector<bool>                   BitValuesUsed;
  HNode                              *CurrentNode;
  bool                                ScalarMatchFound;
};




///
/// The Output class is used to generate a yaml document from in-memory structs
/// and vectors.
///
class Output : public IO {
public:
  Output(llvm::raw_ostream &, void *Ctxt = nullptr, int WrapColumn = 70);
  ~Output() override;

  bool outputting() override;
  bool mapTag(StringRef, bool) override;
  void beginMapping() override;
  void endMapping() override;
  bool preflightKey(const char *key, bool, bool, bool &, void *&) override;
  void postflightKey(void *) override;
  void beginFlowMapping() override;
  void endFlowMapping() override;
  unsigned beginSequence() override;
  void endSequence() override;
  bool preflightElement(unsigned, void *&) override;
  void postflightElement(void *) override;
  unsigned beginFlowSequence() override;
  bool preflightFlowElement(unsigned, void *&) override;
  void postflightFlowElement(void *) override;
  void endFlowSequence() override;
  void beginEnumScalar() override;
  bool matchEnumScalar(const char*, bool) override;
  bool matchEnumFallback() override;
  void endEnumScalar() override;
  bool beginBitSetScalar(bool &) override;
  bool bitSetMatch(const char *, bool ) override;
  void endBitSetScalar() override;
  void scalarString(StringRef &, bool) override;
  void blockScalarString(StringRef &) override;
  void setError(const Twine &message) override;
  bool canElideEmptySequence() override;
public:
  // These are only used by operator<<. They could be private
  // if that templated operator could be made a friend.
  void beginDocuments();
  bool preflightDocument(unsigned);
  void postflightDocument();
  void endDocuments();

private:
  void output(StringRef s);
  void outputUpToEndOfLine(StringRef s);
  void newLineCheck();
  void outputNewLine();
  void paddedKey(StringRef key);
  void flowKey(StringRef Key);

  enum InState {
    inSeq,
    inFlowSeq,
    inMapFirstKey,
    inMapOtherKey,
    inFlowMapFirstKey,
    inFlowMapOtherKey
  };

  llvm::raw_ostream       &Out;
  int                      WrapColumn;
  SmallVector<InState, 8>  StateStack;
  int                      Column;
  int                      ColumnAtFlowStart;
  int                      ColumnAtMapFlowStart;
  bool                     NeedBitValueComma;
  bool                     NeedFlowSequenceComma;
  bool                     EnumerationMatchFound;
  bool                     NeedsNewLine;
};




/// YAML I/O does conversion based on types. But often native data types
/// are just a typedef of built in intergral types (e.g. int).  But the C++
/// type matching system sees through the typedef and all the typedefed types
/// look like a built in type. This will cause the generic YAML I/O conversion
/// to be used. To provide better control over the YAML conversion, you can
/// use this macro instead of typedef.  It will create a class with one field
/// and automatic conversion operators to and from the base type.
/// Based on BOOST_STRONG_TYPEDEF
#define LLVM_YAML_STRONG_TYPEDEF(_base, _type)                                 \
    struct _type {                                                             \
        _type() { }                                                            \
        _type(const _base v) : value(v) { }                                    \
        _type(const _type &v) : value(v.value) {}                              \
        _type &operator=(const _type &rhs) { value = rhs.value; return *this; }\
        _type &operator=(const _base &rhs) { value = rhs; return *this; }      \
        operator const _base & () const { return value; }                      \
        bool operator==(const _type &rhs) const { return value == rhs.value; } \
        bool operator==(const _base &rhs) const { return value == rhs; }       \
        bool operator<(const _type &rhs) const { return value < rhs.value; }   \
        _base value;                                                           \
    };



///
/// Use these types instead of uintXX_t in any mapping to have
/// its yaml output formatted as hexadecimal.
///
LLVM_YAML_STRONG_TYPEDEF(uint8_t, Hex8)
LLVM_YAML_STRONG_TYPEDEF(uint16_t, Hex16)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, Hex32)
LLVM_YAML_STRONG_TYPEDEF(uint64_t, Hex64)


template<>
struct ScalarTraits<Hex8> {
  static void output(const Hex8 &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, Hex8 &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<Hex16> {
  static void output(const Hex16 &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, Hex16 &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<Hex32> {
  static void output(const Hex32 &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, Hex32 &);
  static bool mustQuote(StringRef) { return false; }
};

template<>
struct ScalarTraits<Hex64> {
  static void output(const Hex64 &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, Hex64 &);
  static bool mustQuote(StringRef) { return false; }
};


// Define non-member operator>> so that Input can stream in a document list.
template <typename T>
inline
typename std::enable_if<has_DocumentListTraits<T>::value, Input &>::type
operator>>(Input &yin, T &docList) {
  int i = 0;
  while ( yin.setCurrentDocument() ) {
    yamlize(yin, DocumentListTraits<T>::element(yin, docList, i), true);
    if ( yin.error() )
      return yin;
    yin.nextDocument();
    ++i;
  }
  return yin;
}

// Define non-member operator>> so that Input can stream in a map as a document.
template <typename T>
inline
typename std::enable_if<has_MappingTraits<T>::value, Input &>::type
operator>>(Input &yin, T &docMap) {
  yin.setCurrentDocument();
  yamlize(yin, docMap, true);
  return yin;
}

// Define non-member operator>> so that Input can stream in a sequence as
// a document.
template <typename T>
inline
typename std::enable_if<has_SequenceTraits<T>::value, Input &>::type
operator>>(Input &yin, T &docSeq) {
  if (yin.setCurrentDocument())
    yamlize(yin, docSeq, true);
  return yin;
}

// Define non-member operator>> so that Input can stream in a block scalar.
template <typename T>
inline
typename std::enable_if<has_BlockScalarTraits<T>::value, Input &>::type
operator>>(Input &In, T &Val) {
  if (In.setCurrentDocument())
    yamlize(In, Val, true);
  return In;
}

// Provide better error message about types missing a trait specialization
template <typename T>
inline
typename std::enable_if<missingTraits<T>::value, Input &>::type
operator>>(Input &yin, T &docSeq) {
  char missing_yaml_trait_for_type[sizeof(MissingTrait<T>)];
  return yin;
}


// Define non-member operator<< so that Output can stream out document list.
template <typename T>
inline
typename std::enable_if<has_DocumentListTraits<T>::value, Output &>::type
operator<<(Output &yout, T &docList) {
  yout.beginDocuments();
  const size_t count = DocumentListTraits<T>::size(yout, docList);
  for(size_t i=0; i < count; ++i) {
    if ( yout.preflightDocument(i) ) {
      yamlize(yout, DocumentListTraits<T>::element(yout, docList, i), true);
      yout.postflightDocument();
    }
  }
  yout.endDocuments();
  return yout;
}

// Define non-member operator<< so that Output can stream out a map.
template <typename T>
inline
typename std::enable_if<has_MappingTraits<T>::value, Output &>::type
operator<<(Output &yout, T &map) {
  yout.beginDocuments();
  if ( yout.preflightDocument(0) ) {
    yamlize(yout, map, true);
    yout.postflightDocument();
  }
  yout.endDocuments();
  return yout;
}

// Define non-member operator<< so that Output can stream out a sequence.
template <typename T>
inline
typename std::enable_if<has_SequenceTraits<T>::value, Output &>::type
operator<<(Output &yout, T &seq) {
  yout.beginDocuments();
  if ( yout.preflightDocument(0) ) {
    yamlize(yout, seq, true);
    yout.postflightDocument();
  }
  yout.endDocuments();
  return yout;
}

// Define non-member operator<< so that Output can stream out a block scalar.
template <typename T>
inline
typename std::enable_if<has_BlockScalarTraits<T>::value, Output &>::type
operator<<(Output &Out, T &Val) {
  Out.beginDocuments();
  if (Out.preflightDocument(0)) {
    yamlize(Out, Val, true);
    Out.postflightDocument();
  }
  Out.endDocuments();
  return Out;
}

// Provide better error message about types missing a trait specialization
template <typename T>
inline
typename std::enable_if<missingTraits<T>::value, Output &>::type
operator<<(Output &yout, T &seq) {
  char missing_yaml_trait_for_type[sizeof(MissingTrait<T>)];
  return yout;
}


} // namespace yaml
} // namespace llvm


/// Utility for declaring that a std::vector of a particular type
/// should be considered a YAML sequence.
#define LLVM_YAML_IS_SEQUENCE_VECTOR(_type)                                 \
  namespace llvm {                                                          \
  namespace yaml {                                                          \
    template<>                                                              \
    struct SequenceTraits< std::vector<_type> > {                           \
      static size_t size(IO &io, std::vector<_type> &seq) {                 \
        return seq.size();                                                  \
      }                                                                     \
      static _type& element(IO &io, std::vector<_type> &seq, size_t index) {\
        if ( index >= seq.size() )                                          \
          seq.resize(index+1);                                              \
        return seq[index];                                                  \
      }                                                                     \
    };                                                                      \
  }                                                                         \
  }

/// Utility for declaring that a std::vector of a particular type
/// should be considered a YAML flow sequence.
#define LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(_type)                            \
  namespace llvm {                                                          \
  namespace yaml {                                                          \
    template<>                                                              \
    struct SequenceTraits< std::vector<_type> > {                           \
      static size_t size(IO &io, std::vector<_type> &seq) {                 \
        return seq.size();                                                  \
      }                                                                     \
      static _type& element(IO &io, std::vector<_type> &seq, size_t index) {\
        (void)flow; /* Remove this workaround after PR17897 is fixed */     \
        if ( index >= seq.size() )                                          \
          seq.resize(index+1);                                              \
        return seq[index];                                                  \
      }                                                                     \
      static const bool flow = true;                                        \
    };                                                                      \
  }                                                                         \
  }

/// Utility for declaring that a std::vector of a particular type
/// should be considered a YAML document list.
#define LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(_type)                            \
  namespace llvm {                                                          \
  namespace yaml {                                                          \
    template<>                                                              \
    struct DocumentListTraits< std::vector<_type> > {                       \
      static size_t size(IO &io, std::vector<_type> &seq) {                 \
        return seq.size();                                                  \
      }                                                                     \
      static _type& element(IO &io, std::vector<_type> &seq, size_t index) {\
        if ( index >= seq.size() )                                          \
          seq.resize(index+1);                                              \
        return seq[index];                                                  \
      }                                                                     \
    };                                                                      \
  }                                                                         \
  }



#endif // LLVM_SUPPORT_YAMLTRAITS_H
