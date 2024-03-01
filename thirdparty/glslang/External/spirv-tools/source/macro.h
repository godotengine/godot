// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef SOURCE_MACRO_H_
#define SOURCE_MACRO_H_

// Evaluates to the number of elements of array A.
//
// If we could use constexpr, then we could make this a template function.
// If the source arrays were std::array, then we could have used
// std::array::size.
#define ARRAY_SIZE(A) (static_cast<uint32_t>(sizeof(A) / sizeof(A[0])))

#endif  // SOURCE_MACRO_H_
