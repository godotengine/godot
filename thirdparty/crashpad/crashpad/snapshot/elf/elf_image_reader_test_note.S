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

#include "util/misc/elf_note_types.h"

#define NOTE_ALIGN 4
  .section .note.crashpad.test,"a",%note
  .balign NOTE_ALIGN
  .type testnote, %object
testnote:
  .long name_end - name  // namesz
  .long desc_end - desc  // descsz
  .long CRASHPAD_ELF_NOTE_TYPE_SNAPSHOT_TEST  // type
name:
  .asciz CRASHPAD_ELF_NOTE_NAME
name_end:
  .balign NOTE_ALIGN
desc:
  .long 42
desc_end:
  .size testnote, .-testnote
