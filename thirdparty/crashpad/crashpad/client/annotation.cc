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

#include "client/annotation.h"

#include <type_traits>

#include "client/annotation_list.h"

namespace crashpad {

static_assert(std::is_standard_layout<Annotation>::value,
              "Annotation must be POD");

// static
constexpr size_t Annotation::kNameMaxLength;
constexpr size_t Annotation::kValueMaxSize;

void Annotation::SetSize(ValueSizeType size) {
  DCHECK_LT(size, kValueMaxSize);
  size_ = size;
  // Use Register() instead of Get() in case the calling module has not
  // explicitly initialized the annotation list, to avoid crashing.
  AnnotationList::Register()->Add(this);
}

void Annotation::Clear() {
  size_ = 0;
}

}  // namespace crashpad
