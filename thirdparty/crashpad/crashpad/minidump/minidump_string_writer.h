// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_STRING_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_STRING_WRITER_H_

#include <windows.h>
#include <dbghelp.h>
#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "base/strings/string16.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_rva_list_writer.h"
#include "minidump/minidump_writable.h"

namespace crashpad {
namespace internal {

//! \cond

struct MinidumpStringWriterUTF16Traits {
  using StringType = base::string16;
  using MinidumpStringType = MINIDUMP_STRING;
};

struct MinidumpStringWriterUTF8Traits {
  using StringType = std::string;
  using MinidumpStringType = MinidumpUTF8String;
};

//! \endcond

//! \brief Writes a variable-length string to a minidump file in accordance with
//!     the string type’s characteristics.
//!
//! MinidumpStringWriter objects should not be instantiated directly. To write
//! strings to minidump file, use the MinidumpUTF16StringWriter and
//! MinidumpUTF8StringWriter subclasses instead.
template <typename Traits>
class MinidumpStringWriter : public MinidumpWritable {
 public:
  MinidumpStringWriter();
  ~MinidumpStringWriter() override;

 protected:
  using MinidumpStringType = typename Traits::MinidumpStringType;
  using StringType = typename Traits::StringType;

  bool Freeze() override;
  size_t SizeOfObject() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

  //! \brief Sets the string to be written.
  //!
  //! \note Valid in #kStateMutable.
  void set_string(const StringType& string) { string_.assign(string); }

  //! \brief Retrieves the string to be written.
  //!
  //! \note Valid in any state.
  const StringType& string() const { return string_; }

 private:
  std::unique_ptr<MinidumpStringType> string_base_;
  StringType string_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpStringWriter);
};

//! \brief Writes a variable-length UTF-16-encoded MINIDUMP_STRING to a minidump
//!     file.
//!
//! MinidumpUTF16StringWriter objects should not be instantiated directly
//! outside of the MinidumpWritable family of classes.
class MinidumpUTF16StringWriter final
    : public MinidumpStringWriter<MinidumpStringWriterUTF16Traits> {
 public:
  MinidumpUTF16StringWriter() : MinidumpStringWriter() {}
  ~MinidumpUTF16StringWriter() override;

  //! \brief Converts a UTF-8 string to UTF-16 and sets it as the string to be
  //!     written.
  //!
  //! \note Valid in #kStateMutable.
  void SetUTF8(const std::string& string_utf8);

 private:
  DISALLOW_COPY_AND_ASSIGN(MinidumpUTF16StringWriter);
};

//! \brief Writes a variable-length UTF-8-encoded MinidumpUTF8String to a
//!     minidump file.
//!
//! MinidumpUTF8StringWriter objects should not be instantiated directly outside
//! of the MinidumpWritable family of classes.
class MinidumpUTF8StringWriter final
    : public MinidumpStringWriter<MinidumpStringWriterUTF8Traits> {
 public:
  MinidumpUTF8StringWriter() : MinidumpStringWriter() {}
  ~MinidumpUTF8StringWriter() override;

  //! \brief Sets the string to be written.
  //!
  //! \note Valid in #kStateMutable.
  void SetUTF8(const std::string& string_utf8) { set_string(string_utf8); }

  //! \brief Retrieves the string to be written.
  //!
  //! \note Valid in any state.
  const std::string& UTF8() const { return string(); }

 private:
  DISALLOW_COPY_AND_ASSIGN(MinidumpUTF8StringWriter);
};

//! \brief The writer for a MinidumpRVAList object in a minidump file,
//!     containing a list of \a MinidumpStringWriterType objects.
template <typename MinidumpStringWriterType>
class MinidumpStringListWriter final : public MinidumpRVAListWriter {
 public:
  MinidumpStringListWriter();
  ~MinidumpStringListWriter() override;

  //! \brief Adds a new \a Traits::MinidumpStringWriterType for each element in
  //!     \a vector to the MinidumpRVAList.
  //!
  //! \param[in] vector The vector to use as source data. Each string in the
  //!     vector is treated as a UTF-8 string, and a new string writer will be
  //!     created for each one and made a child of the MinidumpStringListWriter.
  //!
  //! \note Valid in #kStateMutable. No mutator methods may be called before
  //!     this method, and it is not normally necessary to call any mutator
  //!     methods after this method.
  void InitializeFromVector(const std::vector<std::string>& vector);

  //! \brief Creates a new \a Traits::MinidumpStringWriterType object and adds
  //!     it to the MinidumpRVAList.
  //!
  //! This object creates a new string writer with string value \a string_utf8,
  //! takes ownership of it, and becomes its parent in the overall tree of
  //! MinidumpWritable objects.
  //!
  //! \note Valid in #kStateMutable.
  void AddStringUTF8(const std::string& string_utf8);

  //! \brief Determines whether the object is useful.
  //!
  //! A useful object is one that carries data that makes a meaningful
  //! contribution to a minidump file. An object carrying entries would be
  //! considered useful.
  //!
  //! \return `true` if the object is useful, `false` otherwise.
  bool IsUseful() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(MinidumpStringListWriter);
};

}  // namespace internal

using MinidumpUTF16StringListWriter = internal::MinidumpStringListWriter<
    internal::MinidumpUTF16StringWriter>;
using MinidumpUTF8StringListWriter = internal::MinidumpStringListWriter<
    internal::MinidumpUTF8StringWriter>;

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_STRING_WRITER_H_
