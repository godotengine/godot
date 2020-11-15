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

#include "test/multiprocess_exec.h"

#include <sys/types.h>

#include "base/logging.h"
#include "base/strings/utf_string_conversions.h"
#include "gtest/gtest.h"
#include "util/win/command_line.h"

namespace crashpad {
namespace test {

namespace internal {

struct MultiprocessInfo {
  MultiprocessInfo() {}
  ScopedFileHANDLE pipe_c2p_read;
  ScopedFileHANDLE pipe_c2p_write;
  ScopedFileHANDLE pipe_p2c_read;
  ScopedFileHANDLE pipe_p2c_write;
  PROCESS_INFORMATION process_info;
};

}  // namespace internal

Multiprocess::Multiprocess()
    : info_(nullptr),
      code_(EXIT_SUCCESS),
      reason_(kTerminationNormal) {
}

void Multiprocess::Run() {
  // Set up and spawn the child process.
  ASSERT_NO_FATAL_FAILURE(PreFork());
  RunChild();

  // And then run the parent actions in this process.
  RunParent();

  // Reap the child.
  WaitForSingleObject(info_->process_info.hProcess, INFINITE);
  CloseHandle(info_->process_info.hThread);
  CloseHandle(info_->process_info.hProcess);
}

void Multiprocess::SetExpectedChildTermination(TerminationReason reason,
                                               int code) {
  EXPECT_EQ(info_, nullptr)
      << "SetExpectedChildTermination() must be called before Run()";
  reason_ = reason;
  code_ = code;
}

Multiprocess::~Multiprocess() {
  delete info_;
}

FileHandle Multiprocess::ReadPipeHandle() const {
  // This is the parent case, it's stdin in the child.
  return info_->pipe_c2p_read.get();
}

FileHandle Multiprocess::WritePipeHandle() const {
  // This is the parent case, it's stdout in the child.
  return info_->pipe_p2c_write.get();
}

void Multiprocess::CloseReadPipe() {
  info_->pipe_c2p_read.reset();
}

void Multiprocess::CloseWritePipe() {
  info_->pipe_p2c_write.reset();
}

void Multiprocess::RunParent() {
  MultiprocessParent();

  info_->pipe_c2p_read.reset();
  info_->pipe_p2c_write.reset();
}

void Multiprocess::RunChild() {
  MultiprocessChild();

  info_->pipe_c2p_write.reset();
  info_->pipe_p2c_read.reset();
}

MultiprocessExec::MultiprocessExec()
    : Multiprocess(), command_(), arguments_(), command_line_() {
}

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

MultiprocessExec::~MultiprocessExec() {
}

void MultiprocessExec::PreFork() {
  ASSERT_FALSE(command_.empty());

  command_line_.clear();
  AppendCommandLineArgument(command_.value(), &command_line_);
  for (size_t i = 0; i < arguments_.size(); ++i) {
    AppendCommandLineArgument(base::UTF8ToUTF16(arguments_[i]), &command_line_);
  }

  // Make pipes for child-to-parent and parent-to-child communication. Mark them
  // as inheritable via the SECURITY_ATTRIBUTES, but use SetHandleInformation to
  // ensure that the parent sides are not inherited.
  ASSERT_EQ(info(), nullptr);
  set_info(new internal::MultiprocessInfo());

  SECURITY_ATTRIBUTES security_attributes = {0};
  security_attributes.nLength = sizeof(SECURITY_ATTRIBUTES);
  security_attributes.bInheritHandle = TRUE;

  HANDLE c2p_read, c2p_write;
  PCHECK(CreatePipe(&c2p_read, &c2p_write, &security_attributes, 0));
  PCHECK(SetHandleInformation(c2p_read, HANDLE_FLAG_INHERIT, 0));
  info()->pipe_c2p_read.reset(c2p_read);
  info()->pipe_c2p_write.reset(c2p_write);

  HANDLE p2c_read, p2c_write;
  PCHECK(CreatePipe(&p2c_read, &p2c_write, &security_attributes, 0));
  PCHECK(SetHandleInformation(p2c_write, HANDLE_FLAG_INHERIT, 0));
  info()->pipe_p2c_read.reset(p2c_read);
  info()->pipe_p2c_write.reset(p2c_write);
}

void MultiprocessExec::MultiprocessChild() {
  STARTUPINFO startup_info = {0};
  startup_info.cb = sizeof(startup_info);
  startup_info.hStdInput = info()->pipe_p2c_read.get();
  startup_info.hStdOutput = info()->pipe_c2p_write.get();
  startup_info.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  startup_info.dwFlags = STARTF_USESTDHANDLES;
  PCHECK(CreateProcess(command_.value().c_str(),
                       &command_line_[0],  // This cannot be constant, per MSDN.
                       nullptr,
                       nullptr,
                       TRUE,
                       0,
                       nullptr,
                       nullptr,
                       &startup_info,
                       &info()->process_info));
}

ProcessType MultiprocessExec::ChildProcess() {
  return info()->process_info.hProcess;
}

}  // namespace test
}  // namespace crashpad
