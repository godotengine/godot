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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_SIMPLE_STRING_DICTIONARY_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_SIMPLE_STRING_DICTIONARY_WRITER_H_

#include <sys/types.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_string_writer.h"
#include "minidump/minidump_writable.h"

namespace crashpad {

//! \brief The writer for a MinidumpSimpleStringDictionaryEntry object in a
//!     minidump file.
//!
//! Because MinidumpSimpleStringDictionaryEntry objects only appear as elements
//! of MinidumpSimpleStringDictionary objects, this class does not write any
//! data on its own. It makes its MinidumpSimpleStringDictionaryEntry data
//! available to its MinidumpSimpleStringDictionaryWriter parent, which writes
//! it as part of a MinidumpSimpleStringDictionary.
class MinidumpSimpleStringDictionaryEntryWriter final
    : public internal::MinidumpWritable {
 public:
  MinidumpSimpleStringDictionaryEntryWriter();
  ~MinidumpSimpleStringDictionaryEntryWriter() override;

  //! \brief Returns a MinidumpSimpleStringDictionaryEntry referencing this
  //!     object’s data.
  //!
  //! This method is expected to be called by a
  //! MinidumpSimpleStringDictionaryWriter in order to obtain a
  //! MinidumpSimpleStringDictionaryEntry to include in its list.
  //!
  //! \note Valid in #kStateWritable.
  const MinidumpSimpleStringDictionaryEntry*
  GetMinidumpSimpleStringDictionaryEntry() const;

  //! \brief Sets the strings to be written as the entry object’s key and value.
  //!
  //! \note Valid in #kStateMutable.
  void SetKeyValue(const std::string& key, const std::string& value);

  //! \brief Retrieves the key to be written.
  //!
  //! \note Valid in any state.
  const std::string& Key() const { return key_.UTF8(); }

 protected:
  // MinidumpWritable:

  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  struct MinidumpSimpleStringDictionaryEntry entry_;
  internal::MinidumpUTF8StringWriter key_;
  internal::MinidumpUTF8StringWriter value_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpSimpleStringDictionaryEntryWriter);
};

//! \brief The writer for a MinidumpSimpleStringDictionary object in a minidump
//!     file, containing a list of MinidumpSimpleStringDictionaryEntry objects.
//!
//! Because this class writes a representatin of a dictionary, the order of
//! entries is insignificant. Entries may be written in any order.
class MinidumpSimpleStringDictionaryWriter final
    : public internal::MinidumpWritable {
 public:
  MinidumpSimpleStringDictionaryWriter();
  ~MinidumpSimpleStringDictionaryWriter() override;

  //! \brief Adds an initialized MinidumpSimpleStringDictionaryEntryWriter for
  //!     each key-value pair in \a map to the MinidumpSimpleStringDictionary.
  //!
  //! \param[in] map The map to use as source data.
  //!
  //! \note Valid in #kStateMutable. No mutator methods may be called before
  //!     this method, and it is not normally necessary to call any mutator
  //!     methods after this method.
  void InitializeFromMap(const std::map<std::string, std::string>& map);

  //! \brief Adds a MinidumpSimpleStringDictionaryEntryWriter to the
  //!     MinidumpSimpleStringDictionary.
  //!
  //! This object takes ownership of \a entry and becomes its parent in the
  //! overall tree of internal::MinidumpWritable objects.
  //!
  //! If the key contained in \a entry duplicates the key of an entry already
  //! present in the MinidumpSimpleStringDictionary, the new \a entry will
  //! replace the previous one.
  //!
  //! \note Valid in #kStateMutable.
  void AddEntry(
      std::unique_ptr<MinidumpSimpleStringDictionaryEntryWriter> entry);

  //! \brief Determines whether the object is useful.
  //!
  //! A useful object is one that carries data that makes a meaningful
  //! contribution to a minidump file. An object carrying entries would be
  //! considered useful.
  //!
  //! \return `true` if the object is useful, `false` otherwise.
  bool IsUseful() const;

 protected:
  // MinidumpWritable:

  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  // This object owns the MinidumpSimpleStringDictionaryEntryWriter objects.
  std::map<std::string, MinidumpSimpleStringDictionaryEntryWriter*> entries_;

  std::unique_ptr<MinidumpSimpleStringDictionary>
      simple_string_dictionary_base_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpSimpleStringDictionaryWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_SIMPLE_STRING_DICTIONARY_WRITER_H_
