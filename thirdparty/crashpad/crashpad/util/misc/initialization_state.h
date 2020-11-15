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

#ifndef CRASHPAD_UTIL_MISC_INITIALIZATION_INITIALIZATION_STATE_H_
#define CRASHPAD_UTIL_MISC_INITIALIZATION_INITIALIZATION_STATE_H_

#include <stdint.h>

#include "base/macros.h"

namespace crashpad {

//! \brief Tracks whether data are initialized.
//!
//! Objects of this type track whether the data they’re guarding are
//! initialized. The three possible states are uninitialized (the initial
//! state), initializing, and valid. As the guarded data are initialized, an
//! InitializationState object will normally transition through these three
//! states. A fourth state corresponds to the destruction of objects of this
//! type, making it less likely that a use-after-free of an InitializationState
//! object will appear in the valid state.
//!
//! If the only purpose for tracking the initialization state of guarded data is
//! to DCHECK when the object is in an unexpected state, use
//! InitializationStateDcheck instead.
class InitializationState {
 public:
  //! \brief The object’s state.
  enum State : uint8_t {
    //! \brief The object has not yet been initialized.
    kStateUninitialized = 0,

    //! \brief The object is being initialized.
    //!
    //! This state protects against attempted reinitializaton of
    //! partially-initialized objects whose initial initialization attempt
    //! failed. This state is to be used while objects are initializing, but are
    //! not yet fully initialized.
    kStateInvalid,

    //! \brief The object has been initialized.
    kStateValid,

    //! \brief The object has been destroyed.
    kStateDestroyed,
  };

  InitializationState() : state_(kStateUninitialized) {}
  ~InitializationState() { state_ = kStateDestroyed; }

  //! \brief Returns `true` if the object’s state is #kStateUninitialized and it
  //!     is safe to begin initializing it.
  bool is_uninitialized() const { return state_ == kStateUninitialized; }

  //! \brief Sets the object’s state to #kStateInvalid, marking initialization
  //!     as being in process.
  void set_invalid() { state_ = kStateInvalid; }

  //! \brief Sets the object’s state to #kStateValid, marking it initialized.
  void set_valid() { state_ = kStateValid; }

  //! \brief Returns `true` if the the object’s state is #kStateValid and it has
  //!     been fully initialized and may be used.
  bool is_valid() const { return state_ == kStateValid; }

 protected:
  //! \brief Returns the object’s state.
  //!
  //! Consumers of this class should use an is_state_*() method instead.
  State state() const { return state_; }

  //! \brief Sets the object’s state.
  //!
  //! Consumers of this class should use a set_state_*() method instead.
  void set_state(State state) { state_ = state; }

 private:
  // state_ is volatile to ensure that it’ll be set by the destructor when it
  // runs. Otherwise, optimizations might prevent it from ever being set to
  // kStateDestroyed, limiting this class’ ability to catch use-after-free
  // errors.
  volatile State state_;

  DISALLOW_COPY_AND_ASSIGN(InitializationState);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_INITIALIZATION_INITIALIZATION_STATE_H_
