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

#ifndef CRASHPAD_CLIENT_ANNOTATION_LIST_H_
#define CRASHPAD_CLIENT_ANNOTATION_LIST_H_

#include "base/macros.h"
#include "client/annotation.h"

namespace crashpad {

//! \brief A list that contains all the currently set annotations.
//!
//! An instance of this class must be registered on the \a CrashpadInfo
//! structure in order to use the annotations system. Once a list object has
//! been registered on the CrashpadInfo, a different instance should not
//! be used instead.
class AnnotationList {
 public:
  AnnotationList();
  ~AnnotationList();

  //! \brief Returns the instance of the list that has been registered on the
  //!     CrashapdInfo structure.
  static AnnotationList* Get();

  //! \brief Returns the instace of the list, creating and registering
  //!     it if one is not already set on the CrashapdInfo structure.
  static AnnotationList* Register();

  //! \brief Adds \a annotation to the global list. This method does not need
  //!     to be called by clients directly. The Annotation object will do so
  //!     automatically.
  //!
  //! Once an annotation is added to the list, it is not removed. This is
  //! because the AnnotationList avoids the use of locks/mutexes, in case it is
  //! being manipulated in a compromised context. Instead, an Annotation keeps
  //! track of when it has been cleared, which excludes it from a crash report.
  //! This design also avoids linear scans of the list when repeatedly setting
  //! and/or clearing the value.
  void Add(Annotation* annotation);

  //! \brief An InputIterator for the AnnotationList.
  class Iterator {
   public:
    ~Iterator();

    Annotation* operator*() const;
    Iterator& operator++();
    bool operator==(const Iterator& other) const;
    bool operator!=(const Iterator& other) const { return !(*this == other); }

   private:
    friend class AnnotationList;
    Iterator(Annotation* head, const Annotation* tail);

    Annotation* curr_;
    const Annotation* const tail_;

    // Copy and assign are required.
  };

  //! \brief Returns an iterator to the first element of the annotation list.
  Iterator begin();

  //! \brief Returns an iterator past the last element of the annotation list.
  Iterator end();

 private:
  // To make it easier for the handler to locate the dummy tail node, store the
  // pointer. Placed first for packing.
  const Annotation* const tail_pointer_;

  // Dummy linked-list head and tail elements of \a Annotation::Type::kInvalid.
  Annotation head_;
  Annotation tail_;

  DISALLOW_COPY_AND_ASSIGN(AnnotationList);
};

}  // namespace crashpad

#endif  // CRASHPAD_CLIENT_ANNOTATION_LIST_H_
