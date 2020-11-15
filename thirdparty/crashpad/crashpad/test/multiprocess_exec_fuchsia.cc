// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#include "test/multiprocess_exec.h"

#include <lib/fdio/io.h>
#include <lib/fdio/spawn.h>
#include <lib/zx/process.h>
#include <lib/zx/time.h>
#include <zircon/processargs.h>

#include "base/files/scoped_file.h"
#include "base/fuchsia/fuchsia_logging.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {

namespace {

void AddPipe(fdio_spawn_action_t* action, int target_fd, int* fd_out) {
  zx_handle_t handle;
  uint32_t id;
  zx_status_t status = fdio_pipe_half(&handle, &id);
  ZX_CHECK(status >= 0, status) << "fdio_pipe_half";
  action->action = FDIO_SPAWN_ACTION_ADD_HANDLE;
  action->h.id = PA_HND(PA_HND_TYPE(id), target_fd);
  action->h.handle = handle;
  *fd_out = status;
}

}  // namespace

namespace internal {

struct MultiprocessInfo {
  MultiprocessInfo() {}
  base::ScopedFD stdin_write;
  base::ScopedFD stdout_read;
  zx::process child;
};

}  // namespace internal

Multiprocess::Multiprocess()
    : info_(nullptr), code_(EXIT_SUCCESS), reason_(kTerminationNormal) {}

void Multiprocess::Run() {
  // Set up and spawn the child process.
  ASSERT_NO_FATAL_FAILURE(PreFork());
  RunChild();

  // And then run the parent actions in this process.
  RunParent();

  // Wait until the child exits.
  zx_signals_t signals;
  ASSERT_EQ(
      info_->child.wait_one(ZX_TASK_TERMINATED, zx::time::infinite(), &signals),
      ZX_OK);
  ASSERT_EQ(signals, ZX_TASK_TERMINATED);

  // Get the child's exit code.
  zx_info_process_t proc_info;
  zx_status_t status = info_->child.get_info(
      ZX_INFO_PROCESS, &proc_info, sizeof(proc_info), nullptr, nullptr);
  if (status != ZX_OK) {
    ZX_LOG(ERROR, status) << "zx_object_get_info";
    ADD_FAILURE() << "Unable to get exit code of child";
  } else {
    if (code_ != proc_info.return_code) {
      ADD_FAILURE() << "Child exited with code " << proc_info.return_code
                    << ", expected exit with code " << code_;
    }
  }
}

void Multiprocess::SetExpectedChildTermination(TerminationReason reason,
                                               int code) {
  EXPECT_EQ(info_, nullptr)
      << "SetExpectedChildTermination() must be called before Run()";
  reason_ = reason;
  code_ = code;
}

void Multiprocess::SetExpectedChildTerminationBuiltinTrap() {
  SetExpectedChildTermination(kTerminationNormal, -1);
}

Multiprocess::~Multiprocess() {
  delete info_;
}

FileHandle Multiprocess::ReadPipeHandle() const {
  return info_->stdout_read.get();
}

FileHandle Multiprocess::WritePipeHandle() const {
  return info_->stdin_write.get();
}

void Multiprocess::CloseReadPipe() {
  info_->stdout_read.reset();
}

void Multiprocess::CloseWritePipe() {
  info_->stdin_write.reset();
}

void Multiprocess::RunParent() {
  MultiprocessParent();

  info_->stdout_read.reset();
  info_->stdin_write.reset();
}

void Multiprocess::RunChild() {
  MultiprocessChild();
}

MultiprocessExec::MultiprocessExec()
    : Multiprocess(), command_(), arguments_(), argv_() {}

void MultiprocessExec::SetChildCommand(
    const base::FilePath& command,
    const std::vector<std::string>* arguments) {
  command_ = command;
  if (arguments) {
    arguments_ = *arguments;
  } else {
    arguments_.clear();
  }
}

MultiprocessExec::~MultiprocessExec() {}

void MultiprocessExec::PreFork() {
  ASSERT_FALSE(command_.empty());

  ASSERT_TRUE(argv_.empty());

  argv_.push_back(command_.value().c_str());
  for (const std::string& argument : arguments_) {
    argv_.push_back(argument.c_str());
  }
  argv_.push_back(nullptr);

  ASSERT_EQ(info(), nullptr);
  set_info(new internal::MultiprocessInfo());
}

void MultiprocessExec::MultiprocessChild() {
  constexpr size_t kActionCount = 3;
  fdio_spawn_action_t actions[kActionCount];

  int stdin_parent_side = -1;
  AddPipe(&actions[0], STDIN_FILENO, &stdin_parent_side);
  info()->stdin_write.reset(stdin_parent_side);

  int stdout_parent_side = -1;
  AddPipe(&actions[1], STDOUT_FILENO, &stdout_parent_side);
  info()->stdout_read.reset(stdout_parent_side);

  actions[2].action = FDIO_SPAWN_ACTION_CLONE_FD;
  actions[2].fd.local_fd = STDERR_FILENO;
  actions[2].fd.target_fd = STDERR_FILENO;

  // Pass the filesystem namespace, parent environment, and default job to the
  // child, but don't include any other file handles, preferring to set them
  // up explicitly below.
  uint32_t flags = FDIO_SPAWN_CLONE_ALL & ~FDIO_SPAWN_CLONE_STDIO;

  char error_message[FDIO_SPAWN_ERR_MSG_MAX_LENGTH];
  zx::process child;
  zx_status_t status = fdio_spawn_etc(ZX_HANDLE_INVALID,
                                      flags,
                                      command_.value().c_str(),
                                      argv_.data(),
                                      nullptr,
                                      kActionCount,
                                      actions,
                                      child.reset_and_get_address(),
                                      error_message);
  ZX_CHECK(status == ZX_OK, status) << "fdio_spawn_etc: " << error_message;
  info()->child = std::move(child);
}

ProcessType MultiprocessExec::ChildProcess() {
  return zx::unowned_process(info()->child);
}

}  // namespace test
}  // namespace crashpad
