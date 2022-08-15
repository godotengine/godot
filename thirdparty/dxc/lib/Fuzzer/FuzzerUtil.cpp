//===- FuzzerUtil.cpp - Misc utils ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Misc utils.
//===----------------------------------------------------------------------===//

#include "FuzzerInternal.h"
#include <sstream>
#include <iomanip>
#include <sys/time.h>
#include <cassert>
#include <cstring>
#include <signal.h>
#include <unistd.h>

namespace fuzzer {

void Print(const Unit &v, const char *PrintAfter) {
  for (auto x : v)
    Printf("0x%x,", (unsigned) x);
  Printf("%s", PrintAfter);
}

void PrintASCII(const Unit &U, const char *PrintAfter) {
  for (auto X : U) {
    if (isprint(X))
      Printf("%c", X);
    else
      Printf("\\x%x", (unsigned)X);
  }
  Printf("%s", PrintAfter);
}

std::string Hash(const Unit &U) {
  uint8_t Hash[kSHA1NumBytes];
  ComputeSHA1(U.data(), U.size(), Hash);
  std::stringstream SS;
  for (int i = 0; i < kSHA1NumBytes; i++)
    SS << std::hex << std::setfill('0') << std::setw(2) << (unsigned)Hash[i];
  return SS.str();
}

static void AlarmHandler(int, siginfo_t *, void *) {
  Fuzzer::StaticAlarmCallback();
}

void SetTimer(int Seconds) {
  struct itimerval T {{Seconds, 0}, {Seconds, 0}};
  Printf("SetTimer %d\n", Seconds);
  int Res = setitimer(ITIMER_REAL, &T, nullptr);
  assert(Res == 0);
  struct sigaction sigact;
  memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = AlarmHandler;
  Res = sigaction(SIGALRM, &sigact, 0);
  assert(Res == 0);
}

int NumberOfCpuCores() {
  FILE *F = popen("nproc", "r");
  int N = 0;
  fscanf(F, "%d", &N);
  fclose(F);
  return N;
}

void ExecuteCommand(const std::string &Command) {
  system(Command.c_str());
}

}  // namespace fuzzer
