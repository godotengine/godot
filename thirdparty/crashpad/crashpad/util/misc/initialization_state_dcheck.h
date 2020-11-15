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

#ifndef CRASHPAD_UTIL_MISC_INITIALIZATION_INITIALIZATION_STATE_DCHECK_H_
#define CRASHPAD_UTIL_MISC_INITIALIZATION_INITIALIZATION_STATE_DCHECK_H_

//! \file

#include "base/compiler_specific.h"
#include "base/logging.h"
#include "base/macros.h"
#include "build/build_config.h"
#include "util/misc/initialization_state.h"

namespace crashpad {

#if DCHECK_IS_ON() || DOXYGEN

//! \brief Tracks whether data are initialized, triggering a DCHECK assertion
//!     on an invalid data access.
//!
//! Put an InitializationStateDcheck member into a class to help DCHECK that
//! it’s in the right states at the right times. This is useful for classes with
//! Initialize() methods. The chief advantage of InitializationStateDcheck over
//! having a member variable to track state is that when the only use of the
//! variable is to DCHECK, it wastes space (in memory and executable code) in
//! non-DCHECK builds unless the code is also peppered with ugly `#%ifdef`s.
//!
//! This implementation concentrates the ugly `#%ifdef`s in one location.
//!
//! Usage:
//!
//! \code
//!   class Class {
//!    public:
//!     Class() : initialized_() {}
//!
//!     void Initialize() {
//!       INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
//!       // Perform initialization.
//!       INITIALIZATION_STATE_SET_VALID(initialized_);
//!     }
//!
//!     void DoSomething() {
//!       INITIALIZATION_STATE_DCHECK_VALID(initialized_);
//!       // Do something.
//!     }
//!
//!    private:
//!     InitializationStateDcheck initialized_;
//!   };
//! \endcode
class InitializationStateDcheck : public InitializationState {
 public:
  InitializationStateDcheck() : InitializationState() {}

  //! \brief Returns the object’s state.
  //!
  //! Consumers of this class should not call this method. Use the
  //! INITIALIZATION_STATE_SET_INITIALIZING(), INITIALIZATION_STATE_SET_VALID(),
  //! and INITIALIZATION_STATE_DCHECK_VALID() macros instead.
  //
  // The superclass’ state() accessor is protected, but it needs to be exposed
  // to consumers of this class for the macros below to work properly. The
  // macros prefer access to the unerlying state value over a simple boolean
  // because with access to the state value, DCHECK_EQ can be used, which, when
  // tripped, prints both the expected and observed values. This can aid
  // troubleshooting.
  State state() const { return InitializationState::state(); }

  //! \brief Marks an uninitialized object as initializing.
  //!
  //! If the object is in the #kStateUninitialized state, changes its state to
  //! #kStateInvalid (initializing) and returns the previous
  //! (#kStateUninitialized) state. Otherwise, returns the object’s current
  //! state.
  //!
  //! Consumers of this class should not call this method. Use the
  //! INITIALIZATION_STATE_SET_INITIALIZING() macro instead.
  State SetInitializing();

  //! \brief Marks an initializing object as valid.
  //!
  //! If the object is in the #kStateInvalid (initializing) state, changes its
  //! state to #kStateValid and returns the previous (#kStateInvalid) state.
  //! Otherwise, returns the object’s current state.
  //!
  //! Consumers of this class should not call this method. Use the
  //! INITIALIZATION_STATE_SET_VALID() macro instead.
  State SetValid();

 private:
  DISALLOW_COPY_AND_ASSIGN(InitializationStateDcheck);
};

// Using macros enables the non-DCHECK no-op implementation below to be more
// compact and less intrusive. These are macros instead of methods that call
// DCHECK to enable the DCHECK failure message to point to the correct file and
// line number, and to allow additional messages to be streamed on failure with
// the << operator.

//! \brief Checks that a crashpad::InitializationStateDcheck object is in the
//!     crashpad::InitializationState::kStateUninitialized state, and changes
//!     its state to initializing
//!     (crashpad::InitializationState::kStateInvalid).
//!
//! If the object is not in the correct state, a DCHECK assertion is triggered
//! and the object’s state remains unchanged.
//!
//! \param[in] initialization_state_dcheck A crashpad::InitializationStateDcheck
//!     object.
//!
//! \sa crashpad::InitializationStateDcheck
#define INITIALIZATION_STATE_SET_INITIALIZING(initialization_state_dcheck) \
  DCHECK_EQ((initialization_state_dcheck).SetInitializing(),               \
            (initialization_state_dcheck).kStateUninitialized)

//! \brief Checks that a crashpad::InitializationStateDcheck object is in the
//!     initializing (crashpad::InitializationState::kStateInvalid) state, and
//!     changes its state to crashpad::InitializationState::kStateValid.
//!
//! If the object is not in the correct state, a DCHECK assertion is triggered
//! and the object’s state remains unchanged.
//!
//! \param[in] initialization_state_dcheck A crashpad::InitializationStateDcheck
//!     object.
//!
//! \sa crashpad::InitializationStateDcheck
#define INITIALIZATION_STATE_SET_VALID(initialization_state_dcheck) \
  DCHECK_EQ((initialization_state_dcheck).SetValid(),               \
            (initialization_state_dcheck).kStateInvalid)

//! \brief Checks that a crashpad::InitializationStateDcheck object is in the
//!     crashpad::InitializationState::kStateValid state.
//!
//! If the object is not in the correct state, a DCHECK assertion is triggered.
//!
//! \param[in] initialization_state_dcheck A crashpad::InitializationStateDcheck
//!     object.
//!
//! \sa crashpad::InitializationStateDcheck
#define INITIALIZATION_STATE_DCHECK_VALID(initialization_state_dcheck) \
  DCHECK_EQ((initialization_state_dcheck).state(),                     \
            (initialization_state_dcheck).kStateValid)

#else

#if defined(COMPILER_MSVC)
// bool[0] (below) is not accepted by MSVC.
struct InitializationStateDcheck {
};
#else
// Since this is to be used as a DCHECK (for debugging), it should be
// non-intrusive in non-DCHECK (non-debug, release) builds. An empty struct
// would still have a nonzero size (rationale:
// http://www.stroustrup.com/bs_faq2.html#sizeof-empty). Zero-length arrays are
// technically invalid according to the standard, but clang and g++ accept them
// without complaint even with warnings turned up. They take up no space at all,
// and they can be “initialized” with the same () syntax used to initialize
// objects of the DCHECK_IS_ON() InitializationStateDcheck class above.
using InitializationStateDcheck = bool[0];
#endif  // COMPILER_MSVC

// Avoid triggering warnings by repurposing these macros when DCHECKs are
// disabled.
#define INITIALIZATION_STATE_SET_INITIALIZING(initialization_state_dcheck) \
  ALLOW_UNUSED_LOCAL(initialization_state_dcheck)
#define INITIALIZATION_STATE_SET_VALID(initialization_state_dcheck) \
  ALLOW_UNUSED_LOCAL(initialization_state_dcheck)
#define INITIALIZATION_STATE_DCHECK_VALID(initialization_state_dcheck) \
  ALLOW_UNUSED_LOCAL(initialization_state_dcheck)

#endif

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_INITIALIZATION_INITIALIZATION_STATE_DCHECK_H_
