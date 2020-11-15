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

#ifndef CRASHPAD_MINIDUMP_RVA_LIST_WRITER_H_
#define CRASHPAD_MINIDUMP_RVA_LIST_WRITER_H_

#include <stdint.h>
#include <sys/types.h>

#include <memory>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_writable.h"

namespace crashpad {
namespace internal {

//! \brief The writer for a MinidumpRVAList object in a minidump file,
//!     containing a list of ::RVA pointers.
class MinidumpRVAListWriter : public MinidumpWritable {
 protected:
  MinidumpRVAListWriter();
  ~MinidumpRVAListWriter() override;

  //! \brief Adds an ::RVA referencing an MinidumpWritable to the
  //!     MinidumpRVAList.
  //!
  //! This object takes ownership of \a child and becomes its parent in the
  //! overall tree of MinidumpWritable objects.
  //!
  //! To provide type-correctness, subclasses are expected to provide a public
  //! method that accepts a `scoped_ptr`-wrapped argument of the proper
  //! MinidumpWritable subclass, and call this method with that argument.
  //!
  //! \note Valid in #kStateMutable.
  void AddChild(std::unique_ptr<MinidumpWritable> child);

  //! \brief Returns `true` if no child objects have been added by AddChild(),
  //!     and `false` if child objects are present.
  bool IsEmpty() const { return children_.empty(); }

  //! \brief Returns an object’s ::RVA objects referencing its children.
  //!
  //! \note The returned vector will be empty until the object advances to
  //!     #kStateFrozen or beyond.
  const std::vector<RVA>& child_rvas() const { return child_rvas_; }

  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  std::unique_ptr<MinidumpRVAList> rva_list_base_;
  std::vector<std::unique_ptr<MinidumpWritable>> children_;
  std::vector<RVA> child_rvas_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpRVAListWriter);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_RVA_LIST_WRITER_H_
