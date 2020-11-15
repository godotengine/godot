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

#ifndef CRASHPAD_UTIL_MISC_SCOPED_FORBID_RETURN_H_
#define CRASHPAD_UTIL_MISC_SCOPED_FORBID_RETURN_H_

#include "base/macros.h"

namespace crashpad {

//! \brief Asserts that a scope must not be exited while unsafe.
//!
//! An object of this class has two states: armed and disarmed. A disarmed
//! object is a harmless no-op. An armed object will abort execution upon
//! destruction. Newly-constructed objects are armed by default.
//!
//! These objects may be used to assert that a scope not be exited while it is
//! unsafe to do so. If it ever becomes safe to leave such a scope, an object
//! can be disarmed.
class ScopedForbidReturn {
 public:
  ScopedForbidReturn() : armed_(true) {}
  ~ScopedForbidReturn();

  //! \brief Arms the object so that it will abort execution when destroyed.
  //!
  //! The most recent call to Arm() or Disarm() sets the state of the object.
  void Arm() { armed_ = true; }

  //! \brief Arms the object so that it will abort execution when destroyed.
  //!
  //! The most recent call to Arm() or Disarm() sets the state of the object.
  void Disarm() { armed_ = false; }

 private:
  bool armed_;

  DISALLOW_COPY_AND_ASSIGN(ScopedForbidReturn);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_SCOPED_FORBID_RETURN_H_
