// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#include "snapshot/api/module_annotations_win.h"

#include "client/crashpad_info.h"
#include "gtest/gtest.h"
#include "test/win/win_multiprocess.h"
#include "util/file/file_io.h"

namespace crashpad {
namespace test {
namespace {

class ModuleAnnotationsMultiprocessTest final : public WinMultiprocess {
 private:
  void WinMultiprocessParent() override {
    // Read the child executable module.
    HMODULE module = nullptr;
    CheckedReadFileExactly(ReadPipeHandle(), &module, sizeof(module));

    // Reopen the child process with necessary access.
    HANDLE process_handle =
        OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                    FALSE,
                    GetProcessId(ChildProcess()));
    EXPECT_TRUE(process_handle);

    // Read the module annotations in the child process and verify them.
    std::map<std::string, std::string> annotations;
    ASSERT_TRUE(ReadModuleAnnotations(process_handle, module, &annotations));

    EXPECT_GE(annotations.size(), 3u);
    EXPECT_EQ(annotations["#APITEST# key"], "value");
    EXPECT_EQ(annotations["#APITEST# x"], "y");
    EXPECT_EQ(annotations["#APITEST# empty_value"], "");

    // Signal the child process to terminate.
    char c = ' ';
    CheckedWriteFile(WritePipeHandle(), &c, sizeof(c));
  }

  void WinMultiprocessChild() override {
    // Set some test annotations.
    crashpad::CrashpadInfo* crashpad_info =
        crashpad::CrashpadInfo::GetCrashpadInfo();

    crashpad::SimpleStringDictionary* simple_annotations =
        new crashpad::SimpleStringDictionary();
    simple_annotations->SetKeyValue("#APITEST# key", "value");
    simple_annotations->SetKeyValue("#APITEST# x", "y");
    simple_annotations->SetKeyValue("#APITEST# empty_value", "");

    crashpad_info->set_simple_annotations(simple_annotations);

    // Send the executable module.
    HMODULE module = GetModuleHandle(nullptr);
    CheckedWriteFile(WritePipeHandle(), &module, sizeof(module));

    // Wait until a signal from the parent process to terminate.
    char c;
    CheckedReadFileExactly(ReadPipeHandle(), &c, sizeof(c));
  }
};

TEST(ModuleAnnotationsWin, ReadAnnotations) {
  WinMultiprocess::Run<ModuleAnnotationsMultiprocessTest>();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
