// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include <windows.h>

#include "base/logging.h"
#include "client/annotation.h"
#include "client/annotation_list.h"
#include "client/crashpad_info.h"
#include "util/file/file_io.h"

int wmain(int argc, wchar_t* argv[]) {
  crashpad::CrashpadInfo* crashpad_info =
      crashpad::CrashpadInfo::GetCrashpadInfo();

  // This is "leaked" to crashpad_info.
  crashpad::SimpleStringDictionary* simple_annotations =
      new crashpad::SimpleStringDictionary();
  simple_annotations->SetKeyValue("#TEST# pad", "break");
  simple_annotations->SetKeyValue("#TEST# key", "value");
  simple_annotations->SetKeyValue("#TEST# pad", "crash");
  simple_annotations->SetKeyValue("#TEST# x", "y");
  simple_annotations->SetKeyValue("#TEST# longer", "shorter");
  simple_annotations->SetKeyValue("#TEST# empty_value", "");

  crashpad_info->set_simple_annotations(simple_annotations);

  // Set the annotation objects.
  crashpad::AnnotationList::Register();

  static crashpad::StringAnnotation<32> annotation_one("#TEST# one");
  static crashpad::StringAnnotation<32> annotation_two("#TEST# two");
  static crashpad::StringAnnotation<32> annotation_three("#TEST# same-name");
  static crashpad::StringAnnotation<32> annotation_four("#TEST# same-name");

  annotation_one.Set("moocow");
  annotation_two.Set("this will be cleared");
  annotation_three.Set("same-name 3");
  annotation_four.Set("same-name 4");
  annotation_two.Clear();

  // Tell the parent that the environment has been set up.
  HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
  PCHECK(out != INVALID_HANDLE_VALUE) << "GetStdHandle";
  char c = ' ';
  crashpad::CheckedWriteFile(out, &c, sizeof(c));

  HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
  PCHECK(in != INVALID_HANDLE_VALUE) << "GetStdHandle";
  crashpad::CheckedReadFileExactly(in, &c, sizeof(c));
  CHECK(c == 'd' || c == ' ');

  // If 'd' we crash with a debug break, otherwise exit normally.
  if (c == 'd')
    __debugbreak();

  return 0;
}
