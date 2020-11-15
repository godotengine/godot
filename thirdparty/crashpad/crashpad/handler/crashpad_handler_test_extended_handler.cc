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

#include <memory>

#include "base/macros.h"
#include "build/build_config.h"
#include "handler/handler_main.h"
#include "minidump/test/minidump_user_extension_stream_util.h"
#include "tools/tool_support.h"

#if defined(OS_WIN)
#include <windows.h>
#endif

namespace {

class TestUserStreamDataSource : public crashpad::UserStreamDataSource {
 public:
  TestUserStreamDataSource() {}

  std::unique_ptr<crashpad::MinidumpUserExtensionStreamDataSource>
  ProduceStreamData(crashpad::ProcessSnapshot* process_snapshot) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(TestUserStreamDataSource);
};

std::unique_ptr<crashpad::MinidumpUserExtensionStreamDataSource>
TestUserStreamDataSource::ProduceStreamData(
    crashpad::ProcessSnapshot* process_snapshot) {
  static constexpr char kTestData[] = "Injected extension stream!";

  return std::make_unique<crashpad::test::BufferExtensionStreamDataSource>(
      0xCAFEBABE, kTestData, sizeof(kTestData));
}

int ExtendedHandlerMain(int argc, char* argv[]) {
  crashpad::UserStreamDataSources user_stream_data_sources;
  user_stream_data_sources.push_back(
      std::make_unique<TestUserStreamDataSource>());

  return crashpad::HandlerMain(argc, argv, &user_stream_data_sources);
}

}  // namespace

#if defined(OS_POSIX)

int main(int argc, char* argv[]) {
  return ExtendedHandlerMain(argc, argv);
}

#elif defined(OS_WIN)

int wmain(int argc, wchar_t* argv[]) {
  return crashpad::ToolSupport::Wmain(argc, argv, &ExtendedHandlerMain);
}

#endif  // OS_POSIX
