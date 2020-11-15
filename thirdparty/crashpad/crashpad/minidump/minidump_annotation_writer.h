// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_ANNOTATION_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_ANNOTATION_WRITER_H_

#include <memory>
#include <vector>

#include "minidump/minidump_byte_array_writer.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_string_writer.h"
#include "minidump/minidump_writable.h"
#include "snapshot/annotation_snapshot.h"

namespace crashpad {

//! \brief The writer for a MinidumpAnnotation object in a minidump file.
//!
//! Because MinidumpAnnotation objects only appear as elements
//! of MinidumpAnnotationList objects, this class does not write any
//! data on its own. It makes its MinidumpAnnotation data available to its
//! MinidumpAnnotationList parent, which writes it as part of a
//! MinidumpAnnotationList.
class MinidumpAnnotationWriter final : public internal::MinidumpWritable {
 public:
  MinidumpAnnotationWriter();
  ~MinidumpAnnotationWriter();

  //! \brief Initializes the annotation writer with data from an
  //!     AnnotationSnapshot.
  void InitializeFromSnapshot(const AnnotationSnapshot& snapshot);

  //! \brief Initializes the annotation writer with data values.
  void InitializeWithData(const std::string& name,
                          uint16_t type,
                          const std::vector<uint8_t>& data);

  //! \brief Returns the MinidumpAnnotation referencing this object’s data.
  const MinidumpAnnotation* minidump_annotation() const { return &annotation_; }

 protected:
  // MinidumpWritable:

  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<internal::MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  MinidumpAnnotation annotation_;
  internal::MinidumpUTF8StringWriter name_;
  MinidumpByteArrayWriter value_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpAnnotationWriter);
};

//! \brief The writer for a MinidumpAnnotationList object in a minidump file,
//!     containing a list of MinidumpAnnotation objects.
class MinidumpAnnotationListWriter final : public internal::MinidumpWritable {
 public:
  MinidumpAnnotationListWriter();
  ~MinidumpAnnotationListWriter();

  //! \brief Initializes the annotation list writer with a list of
  //!      AnnotationSnapshot objects.
  void InitializeFromList(const std::vector<AnnotationSnapshot>& list);

  //! \brief Adds a single MinidumpAnnotationWriter to the list to be written.
  void AddObject(std::unique_ptr<MinidumpAnnotationWriter> annotation_writer);

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
  std::vector<internal::MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  std::unique_ptr<MinidumpAnnotationList> minidump_list_;
  std::vector<std::unique_ptr<MinidumpAnnotationWriter>> objects_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpAnnotationListWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_ANNOTATION_WRITER_H_
