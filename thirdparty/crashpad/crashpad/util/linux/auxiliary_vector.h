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

#ifndef CRASHPAD_UTIL_LINUX_AUXILIARY_VECTOR_H_
#define CRASHPAD_UTIL_LINUX_AUXILIARY_VECTOR_H_

#include <sys/types.h>

#include <map>

#include "base/logging.h"
#include "base/macros.h"
#include "util/linux/ptrace_connection.h"
#include "util/misc/reinterpret_bytes.h"

namespace crashpad {

//! \brief Read the auxiliary vector for a target process.
class AuxiliaryVector {
 public:
  AuxiliaryVector();
  ~AuxiliaryVector();

  //! \brief Initializes this object with the auxiliary vector for the process
  //!     connected via \a connection.
  //!
  //! This method must be called successfully prior to calling any other method
  //! in this class.
  //!
  //! \param[in] connection A connection to the target process.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool Initialize(PtraceConnection* connection);

  //! \brief Retrieve a value from the vector.
  //!
  //! \param[in] type Specifies which value should be retrieved. The possible
  //!     values for this parameter are defined by `<linux/auxvec.h>`.
  //! \param[out] value The value, casted to an appropriate type, if found.
  //! \return `true` if the value is found.
  template <typename V>
  bool GetValue(uint64_t type, V* value) const {
    auto iter = values_.find(type);
    if (iter == values_.end()) {
      LOG(ERROR) << "value not found";
      return false;
    }
    return ReinterpretBytes(iter->second, value);
  }

 protected:
  std::map<uint64_t, uint64_t> values_;

 private:
  template <typename ULong>
  bool Read(PtraceConnection* connection);

  DISALLOW_COPY_AND_ASSIGN(AuxiliaryVector);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_LINUX_AUXILIARY_VECTOR_H_
