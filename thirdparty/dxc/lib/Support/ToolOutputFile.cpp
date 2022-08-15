//===--- ToolOutputFile.cpp - Implement the tool_output_file class --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the tool_output_file class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"
using namespace llvm;

tool_output_file::CleanupInstaller::CleanupInstaller(StringRef Filename)
    : Filename(Filename), Keep(false) {
  // Arrange for the file to be deleted if the process is killed.
  if (Filename != "-")
    sys::RemoveFileOnSignal(Filename);
}

tool_output_file::CleanupInstaller::~CleanupInstaller() {
  // Delete the file if the client hasn't told us not to.
  if (!Keep && Filename != "-")
    sys::fs::remove(Filename);

  // Ok, the file is successfully written and closed, or deleted. There's no
  // further need to clean it up on signals.
  if (Filename != "-")
    sys::DontRemoveFileOnSignal(Filename);
}

tool_output_file::tool_output_file(StringRef Filename, std::error_code &EC,
                                   sys::fs::OpenFlags Flags)
    : Installer(Filename), OS(Filename, EC, Flags) {
  // If open fails, no cleanup is needed.
  if (EC)
    Installer.Keep = true;
}

tool_output_file::tool_output_file(StringRef Filename, int FD)
    : Installer(Filename), OS(FD, true) {}
