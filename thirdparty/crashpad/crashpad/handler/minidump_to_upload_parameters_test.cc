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

#include "handler/minidump_to_upload_parameters.h"

#include "gtest/gtest.h"
#include "snapshot/test/test_module_snapshot.h"
#include "snapshot/test/test_process_snapshot.h"
#include "util/misc/uuid.h"

namespace crashpad {
namespace test {
namespace {

TEST(MinidumpToUploadParameters, PrecedenceRules) {
  const std::string guid = "00112233-4455-6677-8899-aabbccddeeff";
  UUID uuid;
  ASSERT_TRUE(uuid.InitializeFromString(guid));

  TestProcessSnapshot process_snapshot;
  process_snapshot.SetClientID(uuid);
  process_snapshot.SetAnnotationsSimpleMap({
      {"process-1", "abcdefg"},
      {"list_annotations", "BAD: process_annotations"},
      {"guid", "BAD: process_annotations"},
      {"first", "process"},
  });

  auto module_snapshot_0 = std::make_unique<TestModuleSnapshot>();
  module_snapshot_0->SetAnnotationsVector(
      {"list-module-0-1", "list-module-0-2"});
  module_snapshot_0->SetAnnotationsSimpleMap({
      {"module-0-1", "goat"},
      {"module-0-2", "doge"},
      {"list_annotations", "BAD: module 0"},
      {"guid", "BAD: module 0"},
      {"first", "BAD: module 0"},
      {"second", "module 0"},
  });
  module_snapshot_0->SetAnnotationObjects({
      {"module-0-3", 1, {'s', 't', 'a', 'r'}},
      {"module-0-4", 0xFFFA, {0x42}},
      {"guid", 1, {'B', 'A', 'D', '*', '0', '-', '0'}},
      {"list_annotations", 1, {'B', 'A', 'D', '*', '0', '-', '1'}},
      {"first", 1, {'B', 'A', 'D', '*', '0', '-', '2'}},
  });
  process_snapshot.AddModule(std::move(module_snapshot_0));

  auto module_snapshot_1 = std::make_unique<TestModuleSnapshot>();
  module_snapshot_1->SetAnnotationsVector(
      {"list-module-1-1", "list-module-1-2"});
  module_snapshot_1->SetAnnotationsSimpleMap({
      {"module-1-1", "bear"},
      {"list_annotations", "BAD: module 1"},
      {"guid", "BAD: module 1"},
      {"first", "BAD: module 1"},
      {"second", "BAD: module 1"},
  });
  module_snapshot_1->SetAnnotationObjects({
      {"module-1-3", 0xBEEF, {'a', 'b', 'c'}},
      {"module-1-4", 1, {'m', 'o', 'o', 'n'}},
      {"guid", 1, {'B', 'A', 'D', '*', '1', '-', '0'}},
      {"list_annotations", 1, {'B', 'A', 'D', '*', '1', '-', '1'}},
      {"second", 1, {'B', 'A', 'D', '*', '1', '-', '2'}},
  });
  process_snapshot.AddModule(std::move(module_snapshot_1));

  auto upload_parameters =
      BreakpadHTTPFormParametersFromMinidump(&process_snapshot);

  EXPECT_EQ(upload_parameters.size(), 10u);
  EXPECT_EQ(upload_parameters["process-1"], "abcdefg");
  EXPECT_EQ(upload_parameters["first"], "process");
  EXPECT_EQ(upload_parameters["module-0-1"], "goat");
  EXPECT_EQ(upload_parameters["module-0-2"], "doge");
  EXPECT_EQ(upload_parameters["module-0-3"], "star");
  EXPECT_EQ(upload_parameters["second"], "module 0");
  EXPECT_EQ(upload_parameters["module-1-1"], "bear");
  EXPECT_EQ(upload_parameters["module-1-4"], "moon");
  EXPECT_EQ(upload_parameters["list_annotations"],
            "list-module-0-1\nlist-module-0-2\n"
            "list-module-1-1\nlist-module-1-2");
  EXPECT_EQ(upload_parameters["guid"], guid);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
