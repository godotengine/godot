//===- FuzzerLoop.cpp - Fuzzer's main loop --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Fuzzer's main loop.
//===----------------------------------------------------------------------===//

#include "FuzzerInternal.h"
#include <sanitizer/coverage_interface.h>
#include <algorithm>

namespace fuzzer {

// Only one Fuzzer per process.
static Fuzzer *F;

Fuzzer::Fuzzer(UserSuppliedFuzzer &USF, FuzzingOptions Options)
    : USF(USF), Options(Options) {
  SetDeathCallback();
  InitializeTraceState();
  assert(!F);
  F = this;
}

void Fuzzer::SetDeathCallback() {
  __sanitizer_set_death_callback(StaticDeathCallback);
}

void Fuzzer::PrintUnitInASCIIOrTokens(const Unit &U, const char *PrintAfter) {
  if (Options.Tokens.empty()) {
    PrintASCII(U, PrintAfter);
  } else {
    auto T = SubstituteTokens(U);
    T.push_back(0);
    Printf("%s%s", T.data(), PrintAfter);
  }
}

void Fuzzer::StaticDeathCallback() {
  assert(F);
  F->DeathCallback();
}

void Fuzzer::DeathCallback() {
  Printf("DEATH:\n");
  Print(CurrentUnit, "\n");
  PrintUnitInASCIIOrTokens(CurrentUnit, "\n");
  WriteToCrash(CurrentUnit, "crash-");
}

void Fuzzer::StaticAlarmCallback() {
  assert(F);
  F->AlarmCallback();
}

void Fuzzer::AlarmCallback() {
  assert(Options.UnitTimeoutSec > 0);
  size_t Seconds =
      duration_cast<seconds>(system_clock::now() - UnitStartTime).count();
  if (Seconds == 0) return;
  if (Options.Verbosity >= 2)
    Printf("AlarmCallback %zd\n", Seconds);
  if (Seconds >= (size_t)Options.UnitTimeoutSec) {
    Printf("ALARM: working on the last Unit for %zd seconds\n", Seconds);
    Printf("       and the timeout value is %d (use -timeout=N to change)\n",
           Options.UnitTimeoutSec);
    Print(CurrentUnit, "\n");
    PrintUnitInASCIIOrTokens(CurrentUnit, "\n");
    WriteToCrash(CurrentUnit, "timeout-");
    exit(1);
  }
}

void Fuzzer::PrintStats(const char *Where, size_t Cov, const char *End) {
  if (!Options.Verbosity) return;
  size_t Seconds = secondsSinceProcessStartUp();
  size_t ExecPerSec = (Seconds ? TotalNumberOfRuns / Seconds : 0);
  Printf("#%zd\t%s cov %zd bits %zd units %zd exec/s %zd %s", TotalNumberOfRuns,
         Where, Cov, TotalBits(), Corpus.size(), ExecPerSec, End);
}

void Fuzzer::RereadOutputCorpus() {
  if (Options.OutputCorpus.empty()) return;
  std::vector<Unit> AdditionalCorpus;
  ReadDirToVectorOfUnits(Options.OutputCorpus.c_str(), &AdditionalCorpus,
                         &EpochOfLastReadOfOutputCorpus);
  if (Corpus.empty()) {
    Corpus = AdditionalCorpus;
    return;
  }
  if (!Options.Reload) return;
  if (Options.Verbosity >= 2)
    Printf("Reload: read %zd new units.\n",  AdditionalCorpus.size());
  for (auto &X : AdditionalCorpus) {
    if (X.size() > (size_t)Options.MaxLen)
      X.resize(Options.MaxLen);
    if (UnitHashesAddedToCorpus.insert(Hash(X)).second) {
      CurrentUnit.clear();
      CurrentUnit.insert(CurrentUnit.begin(), X.begin(), X.end());
      size_t NewCoverage = RunOne(CurrentUnit);
      if (NewCoverage) {
        Corpus.push_back(X);
        if (Options.Verbosity >= 1)
          PrintStats("RELOAD", NewCoverage);
      }
    }
  }
}

void Fuzzer::ShuffleAndMinimize() {
  size_t MaxCov = 0;
  bool PreferSmall =
      (Options.PreferSmallDuringInitialShuffle == 1 ||
       (Options.PreferSmallDuringInitialShuffle == -1 && rand() % 2));
  if (Options.Verbosity)
    Printf("PreferSmall: %d\n", PreferSmall);
  PrintStats("READ  ", 0);
  std::vector<Unit> NewCorpus;
  std::random_shuffle(Corpus.begin(), Corpus.end());
  if (PreferSmall)
    std::stable_sort(
        Corpus.begin(), Corpus.end(),
        [](const Unit &A, const Unit &B) { return A.size() < B.size(); });
  Unit &U = CurrentUnit;
  for (const auto &C : Corpus) {
    for (size_t First = 0; First < 1; First++) {
      U.clear();
      size_t Last = std::min(First + Options.MaxLen, C.size());
      U.insert(U.begin(), C.begin() + First, C.begin() + Last);
      size_t NewCoverage = RunOne(U);
      if (NewCoverage) {
        MaxCov = NewCoverage;
        NewCorpus.push_back(U);
        if (Options.Verbosity >= 2)
          Printf("NEW0: %zd L %zd\n", NewCoverage, U.size());
      }
    }
  }
  Corpus = NewCorpus;
  for (auto &X : Corpus)
    UnitHashesAddedToCorpus.insert(Hash(X));
  PrintStats("INITED", MaxCov);
}

size_t Fuzzer::RunOne(const Unit &U) {
  UnitStartTime = system_clock::now();
  TotalNumberOfRuns++;
  size_t Res = 0;
  if (Options.UseFullCoverageSet)
    Res = RunOneMaximizeFullCoverageSet(U);
  else
    Res = RunOneMaximizeTotalCoverage(U);
  auto UnitStopTime = system_clock::now();
  auto TimeOfUnit =
      duration_cast<seconds>(UnitStopTime - UnitStartTime).count();
  if (TimeOfUnit > TimeOfLongestUnitInSeconds) {
    TimeOfLongestUnitInSeconds = TimeOfUnit;
    Printf("Longest unit: %zd s:\n", TimeOfLongestUnitInSeconds);
    Print(U, "\n");
  }
  return Res;
}

void Fuzzer::RunOneAndUpdateCorpus(const Unit &U) {
  if (TotalNumberOfRuns >= Options.MaxNumberOfRuns)
    return;
  ReportNewCoverage(RunOne(U), U);
}

static uintptr_t HashOfArrayOfPCs(uintptr_t *PCs, uintptr_t NumPCs) {
  uintptr_t Res = 0;
  for (uintptr_t i = 0; i < NumPCs; i++) {
    Res = (Res + PCs[i]) * 7;
  }
  return Res;
}

Unit Fuzzer::SubstituteTokens(const Unit &U) const {
  Unit Res;
  for (auto Idx : U) {
    if (Idx < Options.Tokens.size()) {
      std::string Token = Options.Tokens[Idx];
      Res.insert(Res.end(), Token.begin(), Token.end());
    } else {
      Res.push_back(' ');
    }
  }
  // FIXME: Apply DFSan labels.
  return Res;
}

void Fuzzer::ExecuteCallback(const Unit &U) {
  if (Options.Tokens.empty()) {
    USF.TargetFunction(U.data(), U.size());
  } else {
    auto T = SubstituteTokens(U);
    USF.TargetFunction(T.data(), T.size());
  }
}

// Experimental.
// Fuly reset the current coverage state, run a single unit,
// compute a hash function from the full coverage set,
// return non-zero if the hash value is new.
// This produces tons of new units and as is it's only suitable for small tests,
// e.g. test/FullCoverageSetTest.cpp. FIXME: make it scale.
size_t Fuzzer::RunOneMaximizeFullCoverageSet(const Unit &U) {
  __sanitizer_reset_coverage();
  ExecuteCallback(U);
  uintptr_t *PCs;
  uintptr_t NumPCs =__sanitizer_get_coverage_guards(&PCs);
  if (FullCoverageSets.insert(HashOfArrayOfPCs(PCs, NumPCs)).second)
    return FullCoverageSets.size();
  return 0;
}

size_t Fuzzer::RunOneMaximizeTotalCoverage(const Unit &U) {
  size_t NumCounters = __sanitizer_get_number_of_counters();
  if (Options.UseCounters) {
    CounterBitmap.resize(NumCounters);
    __sanitizer_update_counter_bitset_and_clear_counters(0);
  }
  size_t OldCoverage = __sanitizer_get_total_unique_coverage();
  ExecuteCallback(U);
  size_t NewCoverage = __sanitizer_get_total_unique_coverage();
  size_t NumNewBits = 0;
  if (Options.UseCounters)
    NumNewBits = __sanitizer_update_counter_bitset_and_clear_counters(
        CounterBitmap.data());

  if (!(TotalNumberOfRuns & (TotalNumberOfRuns - 1)) && Options.Verbosity)
    PrintStats("pulse ", NewCoverage);

  if (NewCoverage > OldCoverage || NumNewBits)
    return NewCoverage;
  return 0;
}

void Fuzzer::WriteToOutputCorpus(const Unit &U) {
  if (Options.OutputCorpus.empty()) return;
  std::string Path = DirPlusFile(Options.OutputCorpus, Hash(U));
  WriteToFile(U, Path);
  if (Options.Verbosity >= 2)
    Printf("Written to %s\n", Path.c_str());
}

void Fuzzer::WriteToCrash(const Unit &U, const char *Prefix) {
  std::string Path = Prefix + Hash(U);
  WriteToFile(U, Path);
  Printf("CRASHED; file written to %s\nBase64: ", Path.c_str());
  PrintFileAsBase64(Path);
}

void Fuzzer::SaveCorpus() {
  if (Options.OutputCorpus.empty()) return;
  for (const auto &U : Corpus)
    WriteToFile(U, DirPlusFile(Options.OutputCorpus, Hash(U)));
  if (Options.Verbosity)
    Printf("Written corpus of %zd files to %s\n", Corpus.size(),
           Options.OutputCorpus.c_str());
}

void Fuzzer::ReportNewCoverage(size_t NewCoverage, const Unit &U) {
  if (!NewCoverage) return;
  Corpus.push_back(U);
  UnitHashesAddedToCorpus.insert(Hash(U));
  PrintStats("NEW   ", NewCoverage, "");
  if (Options.Verbosity) {
    Printf(" L: %zd", U.size());
    if (U.size() < 30) {
      Printf(" ");
      PrintUnitInASCIIOrTokens(U, "\t");
      Print(U);
    }
    Printf("\n");
  }
  WriteToOutputCorpus(U);
  if (Options.ExitOnFirst)
    exit(0);
}

void Fuzzer::MutateAndTestOne(Unit *U) {
  for (int i = 0; i < Options.MutateDepth; i++) {
    StartTraceRecording();
    size_t Size = U->size();
    U->resize(Options.MaxLen);
    size_t NewSize = USF.Mutate(U->data(), Size, U->size());
    assert(NewSize > 0 && "Mutator returned empty unit");
    assert(NewSize <= (size_t)Options.MaxLen &&
           "Mutator return overisized unit");
    U->resize(NewSize);
    RunOneAndUpdateCorpus(*U);
    size_t NumTraceBasedMutations = StopTraceRecording();
    for (size_t j = 0; j < NumTraceBasedMutations; j++) {
      ApplyTraceBasedMutation(j, U);
      RunOneAndUpdateCorpus(*U);
    }
  }
}

void Fuzzer::Loop(size_t NumIterations) {
  for (size_t i = 1; i <= NumIterations; i++) {
    for (size_t J1 = 0; J1 < Corpus.size(); J1++) {
      SyncCorpus();
      RereadOutputCorpus();
      if (TotalNumberOfRuns >= Options.MaxNumberOfRuns)
        return;
      // First, simply mutate the unit w/o doing crosses.
      CurrentUnit = Corpus[J1];
      MutateAndTestOne(&CurrentUnit);
      // Now, cross with others.
      if (Options.DoCrossOver && !Corpus[J1].empty()) {
        for (size_t J2 = 0; J2 < Corpus.size(); J2++) {
          CurrentUnit.resize(Options.MaxLen);
          size_t NewSize = USF.CrossOver(
              Corpus[J1].data(), Corpus[J1].size(), Corpus[J2].data(),
              Corpus[J2].size(), CurrentUnit.data(), CurrentUnit.size());
          assert(NewSize > 0 && "CrossOver returned empty unit");
          assert(NewSize <= (size_t)Options.MaxLen &&
                 "CrossOver return overisized unit");
          CurrentUnit.resize(NewSize);
          MutateAndTestOne(&CurrentUnit);
        }
      }
    }
  }
}

void Fuzzer::SyncCorpus() {
  if (Options.SyncCommand.empty() || Options.OutputCorpus.empty()) return;
  auto Now = system_clock::now();
  if (duration_cast<seconds>(Now - LastExternalSync).count() <
      Options.SyncTimeout)
    return;
  LastExternalSync = Now;
  ExecuteCommand(Options.SyncCommand + " " + Options.OutputCorpus);
}

}  // namespace fuzzer
