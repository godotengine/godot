# Copyright (c) 2021, Google Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#
#   * Neither the name of Google nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Top-level presubmit script for libwebm.

See https://dev.chromium.org/developers/how-tos/depottools/presubmit-scripts for
details on the presubmit API built into depot_tools.
"""
import re
import subprocess2

USE_PYTHON3 = True
_BASH_INDENTATION = "2"
_GIT_COMMIT_SUBJECT_LENGTH = 65
_INCLUDE_BASH_FILES_ONLY = [r".*\.sh$"]
_INCLUDE_SOURCE_FILES_ONLY = [r".*\.(c|cc|[hc]pp|h)$"]
_LIBWEBM_MAX_LINE_LENGTH = 80


def _CheckCommitSubjectLength(input_api, output_api):
  """Ensures commit's subject length is no longer than 65 chars."""
  name = "git-commit subject"
  cmd = ["git", "log", "-1", "--pretty=%s"]
  start = input_api.time.time()
  proc = subprocess2.Popen(
      cmd,
      stderr=subprocess2.PIPE,
      stdout=subprocess2.PIPE,
      universal_newlines=True)

  stdout, _ = proc.communicate()
  duration = input_api.time.time() - start

  if not re.match(r"^Revert",
                  stdout) and (len(stdout) - 1) > _GIT_COMMIT_SUBJECT_LENGTH:
    failure_msg = (
        "The commit subject: %s is too long (%d chars)\n"
        "Try to keep this to 50 or less (up to 65 is permitted for "
        "non-reverts).\n"
        "https://www.git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-"
        "Project#_commit_guidelines") % (stdout, len(stdout) - 1)
    return output_api.PresubmitError("%s\n (%4.2fs) failed\n%s" %
                                     (name, duration, failure_msg))

  return output_api.PresubmitResult("%s\n (%4.2fs) success" % (name, duration))


def _GetFilesToSkip(input_api):
  """Skips libwebm-specific files."""
  return list(input_api.DEFAULT_FILES_TO_SKIP) + [
      r"\.pylintrc$",
  ]


def _CheckChangeLintsClean(input_api, output_api):
  """Makes sure that libwebm/ code is cpplint clean."""
  sources = lambda x: input_api.FilterSourceFile(
      x, files_to_check=_INCLUDE_SOURCE_FILES_ONLY, files_to_skip=None)
  return input_api.canned_checks.CheckChangeLintsClean(input_api, output_api,
                                                       sources)


def _RunShellCheckCmd(input_api, output_api, bash_file):
  """shellcheck command wrapper."""
  cmd = ["shellcheck", "-x", "-oall", "-sbash", bash_file]
  name = "Check %s file." % bash_file
  start = input_api.time.time()
  output, rc = subprocess2.communicate(
      cmd, stdout=None, stderr=subprocess2.PIPE, universal_newlines=True)
  duration = input_api.time.time() - start
  if rc == 0:
    return output_api.PresubmitResult("%s\n%s (%4.2fs)\n" %
                                      (name, " ".join(cmd), duration))
  return output_api.PresubmitError("%s\n%s (%4.2fs) failed\n%s" %
                                   (name, " ".join(cmd), duration, output[1]))


def _RunShfmtCheckCmd(input_api, output_api, bash_file):
  """shfmt command wrapper."""
  cmd = [
      "shfmt", "-i", _BASH_INDENTATION, "-bn", "-ci", "-sr", "-kp", "-d",
      bash_file
  ]
  name = "Check %s file." % bash_file
  start = input_api.time.time()
  output, rc = subprocess2.communicate(
      cmd, stdout=None, stderr=subprocess2.PIPE, universal_newlines=True)
  duration = input_api.time.time() - start
  if rc == 0:
    return output_api.PresubmitResult("%s\n%s (%4.2fs)\n" %
                                      (name, " ".join(cmd), duration))
  return output_api.PresubmitError("%s\n%s (%4.2fs) failed\n%s" %
                                   (name, " ".join(cmd), duration, output[1]))


def _RunCmdOnCheckedFiles(input_api, output_api, run_cmd, files_to_check):
  """Ensure that libwebm/ files are clean."""
  file_filter = lambda x: input_api.FilterSourceFile(
      x, files_to_check=files_to_check, files_to_skip=None)

  affected_files = input_api.change.AffectedFiles(file_filter=file_filter)
  results = [
      run_cmd(input_api, output_api, f.AbsoluteLocalPath())
      for f in affected_files
  ]
  return results


def _CommonChecks(input_api, output_api):
  results = []
  results.extend(
      input_api.canned_checks.CheckChangeHasNoCrAndHasOnlyOneEol(
          input_api, output_api))
  results.extend(
      input_api.canned_checks.CheckChangeHasNoTabs(input_api, output_api))
  results.extend(
      input_api.canned_checks.CheckChangeHasNoStrayWhitespace(
          input_api, output_api))
  results.append(_CheckCommitSubjectLength(input_api, output_api))

  source_file_filter = lambda x: input_api.FilterSourceFile(
      x, files_to_skip=_GetFilesToSkip(input_api))
  results.extend(
      input_api.canned_checks.CheckLongLines(
          input_api,
          output_api,
          maxlen=_LIBWEBM_MAX_LINE_LENGTH,
          source_file_filter=source_file_filter))

  results.extend(
      input_api.canned_checks.CheckPatchFormatted(
          input_api,
          output_api,
          check_clang_format=True,
          check_python=True,
          result_factory=output_api.PresubmitError))
  results.extend(_CheckChangeLintsClean(input_api, output_api))

  # Run pylint.
  results.extend(
      input_api.canned_checks.RunPylint(
          input_api,
          output_api,
          files_to_skip=_GetFilesToSkip(input_api),
          pylintrc=".pylintrc",
          version="2.7"))

  # Binaries shellcheck and shfmt are not installed in depot_tools.
  # Installation is needed
  try:
    subprocess2.communicate(["shellcheck", "--version"])
    results.extend(
        _RunCmdOnCheckedFiles(input_api, output_api, _RunShellCheckCmd,
                              _INCLUDE_BASH_FILES_ONLY))
    print("shfmt")
    subprocess2.communicate(["shfmt", "-version"])
    results.extend(
        _RunCmdOnCheckedFiles(input_api, output_api, _RunShfmtCheckCmd,
                              _INCLUDE_BASH_FILES_ONLY))
  except OSError as os_error:
    results.append(
        output_api.PresubmitPromptWarning(
            "%s\nPlease install missing binaries locally." % os_error.args[0]))
  return results


def CheckChangeOnUpload(input_api, output_api):
  results = []
  results.extend(_CommonChecks(input_api, output_api))
  return results


def CheckChangeOnCommit(input_api, output_api):
  results = []
  results.extend(_CommonChecks(input_api, output_api))
  return results
