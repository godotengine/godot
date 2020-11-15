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

PROCESS_TYPE_STRUCT_BEGIN(Annotation)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, link_node)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, name)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, value)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, size)
  PROCESS_TYPE_STRUCT_MEMBER(uint16_t, type)
PROCESS_TYPE_STRUCT_END(Annotation)

#if !defined(PROCESS_TYPE_STRUCT_IMPLEMENT_ARRAY)

PROCESS_TYPE_STRUCT_BEGIN(AnnotationList)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, tail_pointer)
  PROCESS_TYPE_STRUCT_MEMBER(crashpad::process_types::Annotation, head)
  PROCESS_TYPE_STRUCT_MEMBER(crashpad::process_types::Annotation, tail)
PROCESS_TYPE_STRUCT_END(AnnotationList)

#endif  // !defined(PROCESS_TYPE_STRUCT_IMPLEMENT_ARRAY)
