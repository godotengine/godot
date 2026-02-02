// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.

#include "usdc-writer.hh"

#if !defined(TINYUSDZ_DISABLE_MODULE_USDC_WRITER)

#if defined(_MSC_VER) || defined(__MINGW32__)
#if defined(__clang__)
// No need to define NOMINMAX for llvm-mingw
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif


#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>  // include API for expanding a file path

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif


#include <fstream>
#include <iostream>
#include <sstream>

#include "crate-format.hh"
#include "io-util.hh"
#include "lz4-compression.hh"
#include "token-type.hh"

#include "common-macros.inc"

namespace tinyusdz {
namespace usdc {

namespace {

constexpr size_t kSectionNameMaxLength = 15;

#ifdef _WIN32
std::wstring UTF8ToWchar(const std::string &str) {
  int wstr_size =
      MultiByteToWideChar(CP_UTF8, 0, str.data(), int(str.size()), nullptr, 0);
  std::wstring wstr(size_t(wstr_size), 0);
  MultiByteToWideChar(CP_UTF8, 0, str.data(), int(str.size()), &wstr[0],
                      int(wstr.size()));
  return wstr;
}

#if 0
std::string WcharToUTF8(const std::wstring &wstr) {
  int str_size = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), int(wstr.size()),
                                     nullptr, 0, nullptr, nullptr);
  std::string str(size_t(str_size), 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr.data(), int(wstr.size()), &str[0],
                      int(str.size()), nullptr, nullptr);
  return str;
}
#endif
#endif

struct Section {
  Section() { memset(this, 0, sizeof(*this)); }
  Section(char const *name, int64_t start, int64_t size);
  char name[kSectionNameMaxLength + 1];
  int64_t start, size;  // byte offset to section info and its data size
};

//
// TOC = list of sections.
//
struct TableOfContents {
  // Section const *GetSection(SectionName) const;
  // int64_t GetMinimumSectionStart() const;
  std::vector<Section> sections;
};

//struct Field {
//  // FIXME(syoyo): Do we need 4 bytes padding as done in pxrUSD?
//  // uint32_t padding_;
//
//  crate::TokenIndex token_index;
//  crate::ValueRep value_rep;
//};

#if 0
// For unordered_map

// https://stackoverflow.com/questions/8513911/how-to-create-a-good-hash-combine-with-64-bit-output-inspired-by-boosthash-co
// From CityHash code.
template <class T>
inline void hash_combine(std::size_t &seed, const T &v) {
#ifdef __wasi__  // 32bit platform
  // Use boost version.
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
#else
  std::hash<T> hasher;
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  std::size_t a = (hasher(v) ^ seed) * kMul;
  a ^= (a >> 47);
  std::size_t b = (seed ^ a) * kMul;
  b ^= (b >> 47);
  seed = b * kMul;
#endif
}

struct PathHasher {
  size_t operator()(const Path &path) const {
    size_t seed = std::hash<std::string>()(path.GetPrimPart());
    hash_combine(seed, std::hash<std::string>()(path.GetPropPart()));
    hash_combine(seed, std::hash<std::string>()(path.GetLocalPart()));
    hash_combine(seed, std::hash<bool>()(path.IsValid()));

    return seed;
  }
};

struct PathKeyEqual {
  bool operator()(const Path &lhs, const Path &rhs) const {
    bool ret = lhs.GetPrimPart() == rhs.GetPrimPart();
    ret &= lhs.GetPropPart() == rhs.GetPropPart();
    ret &= lhs.GetLocalPart() == rhs.GetLocalPart();
    ret &= lhs.IsValid() == rhs.IsValid();

    return ret;
  }
};

struct FieldHasher {
  size_t operator()(const Field &field) const {
    size_t seed = std::hash<uint32_t>()(field.token_index.value);
    hash_combine(seed, std::hash<uint64_t>()(field.value_rep.GetData()));

    return seed;
  }
};

struct FieldKeyEqual {
  bool operator()(const Field &lhs, const Field &rhs) const {
    bool ret = lhs.token_index == rhs.token_index;
    ret &= lhs.value_rep == rhs.value_rep;

    return ret;
  }
};

struct FieldSetHasher {
  size_t operator()(const std::vector<crate::FieldIndex> &fieldset) const {
    if (fieldset.empty()) {
      return 0;
    }

    size_t seed = std::hash<uint32_t>()(fieldset[0].value);
    for (size_t i = 1; i < fieldset.size(); i++) {
      hash_combine(seed, std::hash<uint32_t>()(fieldset[i].value));
    }

    return seed;
  }
};
#endif

class Packer {
 public:
  crate::TokenIndex AddToken(const Token &token);
  crate::StringIndex AddString(const std::string &str);
  crate::PathIndex AddPath(const Path &path);
  crate::FieldIndex AddField(const crate::Field &field);
  crate::FieldSetIndex AddFieldSet(
      const std::vector<crate::FieldIndex> &field_indices);

  const std::vector<Token> &GetTokens() const { return tokens_; }

 private:
  std::unordered_map<Token, crate::TokenIndex, TokenHasher, TokenKeyEqual>
      token_to_index_map;
  std::unordered_map<std::string, crate::StringIndex> string_to_index_map;
  std::unordered_map<Path, crate::PathIndex, crate::PathHasher, crate::PathKeyEqual>
      path_to_index_map;
  std::unordered_map<crate::Field, crate::FieldIndex, crate::FieldHasher, crate::FieldKeyEqual>
      field_to_index_map;
  std::unordered_map<std::vector<crate::FieldIndex>, crate::FieldSetIndex,
                     crate::FieldSetHasher>
      fieldset_to_index_map;

  std::vector<Token> tokens_;
  std::vector<std::string> strings_;
  std::vector<Path> paths_;
  std::vector<crate::Field> fields_;
  std::vector<crate::FieldIndex>
      fieldsets_;  // flattened 1D array of FieldSets. Each span is terminated
                   // by Index()(= ~0)
};

#if 0 // not used atm.
crate::TokenIndex Packer::AddToken(const Token &token) {
  if (token_to_index_map.count(token)) {
    return token_to_index_map[token];
  }

  // index = size of umap
  token_to_index_map[token] = crate::TokenIndex(uint32_t(tokens_.size()));
  tokens_.emplace_back(token);

  return token_to_index_map[token];
}

crate::StringIndex Packer::AddString(const std::string &str) {
  if (string_to_index_map.count(str)) {
    return string_to_index_map[str];
  }

  // index = size of umap
  string_to_index_map[str] = crate::StringIndex(uint32_t(strings_.size()));
  strings_.emplace_back(str);

  return string_to_index_map[str];
}

crate::PathIndex Packer::AddPath(const Path &path) {
  if (path_to_index_map.count(path)) {
    return path_to_index_map[path];
  }

  // index = size of umap
  path_to_index_map[path] = crate::PathIndex(uint32_t(paths_.size()));
  paths_.emplace_back(path);

  return path_to_index_map[path];
}

crate::FieldIndex Packer::AddField(const crate::Field &field) {
  if (field_to_index_map.count(field)) {
    return field_to_index_map[field];
  }

  // index = size of umap
  field_to_index_map[field] = crate::FieldIndex(uint32_t(fields_.size()));
  fields_.emplace_back(field);

  return field_to_index_map[field];
}

crate::FieldSetIndex Packer::AddFieldSet(
    const std::vector<crate::FieldIndex> &fieldset) {
  if (fieldset_to_index_map.count(fieldset)) {
    return fieldset_to_index_map[fieldset];
  }

  // index = size of umap = star index of FieldSet span.
  fieldset_to_index_map[fieldset] =
      crate::FieldSetIndex(uint32_t(fieldsets_.size()));

  fieldsets_.insert(fieldsets_.end(), fieldset.begin(), fieldset.end());
  fieldsets_.push_back(crate::FieldIndex());  // terminator(~0)

  return fieldset_to_index_map[fieldset];
}
#endif

class Writer {
 public:
  Writer(const Stage &stage) : stage_(stage) {}

  const Stage &stage_;

  const std::string &GetError() const { return err_; }
  const std::string &GetWarning() const { return warn_; }

  void PushError(const std::string &s) {
    err_ += s;
  }

  void PushWarn(const std::string &s) {
    warn_ += s;
  }

  bool WriteHeader(uint64_t toc_offset) {
    char magic[8];
    magic[0] = 'P';
    magic[1] = 'X';
    magic[2] = 'R';
    magic[3] = '-';
    magic[4] = 'U';
    magic[5] = 'S';
    magic[6] = 'D';
    magic[7] = 'C';

    uint8_t version[8];  // Only first 3 bytes are used.
    version[0] = 0;
    version[1] = 8;
    version[2] = 0;

    std::array<uint8_t, 88> header;
    memset(&header, 0, 88);

    memcpy(&header[0], magic, 8);
    memcpy(&header[8], version, 8);
    memcpy(&header[16], &toc_offset, 8);

    oss_.write(reinterpret_cast<const char *>(&header[0]), 88);

    return true;
  }

  bool WriteTokens() {
    // Build single string separated by '\0', then compress it with lz4
    std::ostringstream oss;

    auto tokens = packer_.GetTokens();

    for (size_t i = 0; i < tokens.size(); i++) {
      oss << tokens[i].str();

      if (i != (tokens.size() - 1)) {
        oss.put('\0');  // separator
      }
    }
    // Last string does not terminated with `\0'

    // compress
    size_t input_bytes = oss.str().size();
    if (input_bytes == 0) {
      PUSH_ERROR("Invalid data size.");
      return false;
    }

    std::vector<char> buf;
    buf.resize(LZ4Compression::GetCompressedBufferSize(input_bytes));

    std::string err;
    size_t n = LZ4Compression::CompressToBuffer(oss.str().data(), buf.data(),
                                                input_bytes, &err);

    (void)n;

    if (!err.empty()) {
      PUSH_ERROR(err);
      return false;
    }

    return true;
  }

  bool WriteStrings() { return false; }

  bool WriteFields() { return false; }

  bool WriteFieldSets() { return false; }

  bool WritePaths() { return false; }

  bool WriteSpecs() { return false; }

  bool WriteTOC() {
    uint64_t num_sections = toc_.sections.size();

    DCOUT("# of sections = " << std::to_string(num_sections));

    if (num_sections == 0) {
      err_ += "Zero sections in TOC.\n";
      return false;
    }

    // # of sections
    oss_.write(reinterpret_cast<const char *>(&num_sections), 8);

    return true;
  }

  bool Write() {
    //
    //  - TOC
    //  - Tokens
    //  - Strings
    //  - Fields
    //  - FieldSets
    //  - Paths
    //  - Specs
    //

    if (!WriteTokens()) {
      PUSH_ERROR("Failed to write Tokens.");
      return false;
    }

    if (!WriteStrings()) {
      PUSH_ERROR("Failed to write Strings.");
      return false;
    }

    if (!WriteFields()) {
      PUSH_ERROR("Failed to write Fields.");
      return false;
    }

    if (!WriteFieldSets()) {
      PUSH_ERROR("Failed to write FieldSets.");
      return false;
    }

    if (!WritePaths()) {
      PUSH_ERROR("Failed to write Paths.");
      return false;
    }

    if (!WriteSpecs()) {
      PUSH_ERROR("Failed to write Specs.");
      return false;
    }

    // TODO(syoyo): Add feature to support writing unknown section(custom user
    // data)
    // if (!WriteUnknownSections()) {
    //  PUSH_ERROR("Failed to write custom sections.");
    //  return false;
    //}

    const uint64_t toc_offset = static_cast<uint64_t>(oss_.tellp());
    if (!WriteTOC()) {
      PUSH_ERROR("Failed to write TOC.");
      return false;
    }

    // write header
    oss_.seekp(0, std::ios::beg);
    if (!WriteHeader(toc_offset)) {
      PUSH_ERROR("Failed to write Header.");
      return false;
    }

    return true;
  }

  // Get serialized USDC binary data
  bool GetOutput(std::vector<uint8_t> *output) {
    if (!err_.empty()) {
      return false;
    }

    (void)output;

    // TODO
    return false;
  }

 private:
  Writer() = delete;
  Writer(const Writer &) = delete;

  TableOfContents toc_;

  Packer packer_;

  //
  // Serialized data
  //
  std::ostringstream oss_;

  std::string err_;
  std::string warn_;
};

}  // namespace

bool SaveAsUSDCToFile(const std::string &filename, const Stage &stage,
                      std::string *warn, std::string *err) {
#ifdef __ANDROID__
  (void)filename;
  (void)stage;
  (void)warn;

  if (err) {
    (*err) += "Saving USDC to a file is not supported for Android platform(at the moment).\n";
  }
  return false;
#else

  std::vector<uint8_t> output;

  if (!SaveAsUSDCToMemory(stage, &output, warn, err)) {
    return false;
  }

#ifdef _WIN32
#if defined(_MSC_VER) || defined(__GLIBCXX__) || defined(__clang__)
  FILE *fp = nullptr;
  errno_t fperr = _wfopen_s(&fp, UTF8ToWchar(filename).c_str(), L"wb");
  if (fperr != 0) {
    if (err) {
      // TODO: WChar
      (*err) += "Failed to open file to write.\n";
    }
    return false;
  }
#else
  FILE *fp = nullptr;
  errno_t fperr = fopen_s(&fp, filename.c_str(), "wb");
  if (fperr != 0) {
    if (err) {
      (*err) += "Failed to open file `" + filename + "` to write.\n";
    }
    return false;
  }
#endif

#else
  FILE *fp = fopen(filename.c_str(), "wb");
  if (fp == nullptr) {
    if (err) {
      (*err) += "Failed to open file `" + filename + "` to write.\n";
    }
    return false;
  }
#endif

  size_t n = fwrite(output.data(), /* size */ 1, /* count */ output.size(), fp);
  if (n < output.size()) {
    // TODO: Retry writing data when n < output.size()

    if (err) {
      (*err) += "Failed to write data to a file.\n";
    }
    return false;
  }

  return true;
#endif
}

bool SaveAsUSDCToMemory(const Stage &stage, std::vector<uint8_t> *output,
                        std::string *warn, std::string *err) {
  (void)warn;
  (void)output;

  // TODO
  Writer writer(stage);

  if (err) {
    (*err) += "USDC writer is not yet implemented.\n";
  }

  return false;
}

}  // namespace usdc
}  // namespace tinyusdz

#else

namespace tinyusdz {
namespace usdc {

bool SaveAsUSDCToFile(const std::string &filename, const Stage &stage,
                      std::string *warn, std::string *err) {
  (void)filename;
  (void)stage;
  (void)warn;

  if (err) {
    (*err) = "USDC writer feature is disabled in this build.\n";
  }

  return false;
}

bool SaveAsUSDCToMemory(const Stage &stage, std::vector<uint8_t> *output,
                        std::string *warn, std::string *err) {
  (void)stage;
  (void)output;
  (void)warn;

  if (err) {
    (*err) = "USDC writer feature is disabled in this build.\n";
  }

  return false;
}

}  // namespace usdc
}  // namespace tinyusdz

#endif
