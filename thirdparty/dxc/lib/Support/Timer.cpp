//===-- Timer.cpp - Interval Timing Support -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interval Timing implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Timer.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// CreateInfoOutputFile - Return a file stream to print our output on.
namespace llvm { extern raw_ostream *CreateInfoOutputFile(); }

// getLibSupportInfoOutputFilename - This ugly hack is brought to you courtesy
// of constructor/destructor ordering being unspecified by C++.  Basically the
// problem is that a Statistic object gets destroyed, which ends up calling
// 'GetLibSupportInfoOutputFile()' (below), which calls this function.
// LibSupportInfoOutputFilename used to be a global variable, but sometimes it
// would get destroyed before the Statistic, causing havoc to ensue.  We "fix"
// this by creating the string the first time it is needed and never destroying
// it.
static ManagedStatic<std::string> LibSupportInfoOutputFilename;
static std::string &getLibSupportInfoOutputFilename() {
  return *LibSupportInfoOutputFilename;
}

static ManagedStatic<sys::SmartMutex<true> > TimerLock;

namespace {
#if 0 // HLSL Change Starts - option pending
  static cl::opt<bool>
  TrackSpace("track-memory", cl::desc("Enable -time-passes memory "
                                      "tracking (this may be slow)"),
             cl::Hidden);

  static cl::opt<std::string, true>
  InfoOutputFilename("info-output-file", cl::value_desc("filename"),
                     cl::desc("File to append -stats and -timer output to"),
                   cl::Hidden, cl::location(getLibSupportInfoOutputFilename()));
#else
  static const bool TrackSpace = false;
#endif // HLSL Change Ends
}

// CreateInfoOutputFile - Return a file stream to print our output on.
raw_ostream *llvm::CreateInfoOutputFile() {
  const std::string &OutputFilename = getLibSupportInfoOutputFilename();
  if (OutputFilename.empty())
    return new raw_fd_ostream(2, false); // stderr.
  if (OutputFilename == "-")
    return new raw_fd_ostream(1, false); // stdout.
  
  // Append mode is used because the info output file is opened and closed
  // each time -stats or -time-passes wants to print output to it. To
  // compensate for this, the test-suite Makefiles have code to delete the
  // info output file before running commands which write to it.
  std::error_code EC;
  raw_ostream *Result = new raw_fd_ostream(OutputFilename, EC,
                                           sys::fs::F_Append | sys::fs::F_Text);
  if (!EC)
    return Result;
  
  errs() << "Error opening info-output-file '"
    << OutputFilename << " for appending!\n";
  delete Result;
  return new raw_fd_ostream(2, false); // stderr.
}

#define DefaultTimerGroupName "Miscellaneous Ungrouped Timers"
// static TimerGroup DefaultTimerGroup(DefaultTimerGroupName); // HLSL Change - global init
static TimerGroup *getDefaultTimerGroup() {
#if 1 // HLSL Change Starts - global with special clean-up and init
  return nullptr; // rather than alloc-on-demand or &DefaultTimerGroup;
#else
  TimerGroup *tmp = DefaultTimerGroup;
  sys::MemoryFence();
  if (tmp) return tmp;
  
  sys::SmartScopedLock<true> Lock(*TimerLock);
  tmp = DefaultTimerGroup;
  if (!tmp) {
    tmp = new TimerGroup("Miscellaneous Ungrouped Timers");
    sys::MemoryFence();
    DefaultTimerGroup = tmp;
  }

  return tmp;
#endif // HLSL Change Ends
}

//===----------------------------------------------------------------------===//
// Timer Implementation
//===----------------------------------------------------------------------===//

void Timer::init(StringRef N) {
  assert(!TG && "Timer already initialized");
  Name.assign(N.begin(), N.end());
  Started = false;
  TG = getDefaultTimerGroup();
  if (!TG) return; // HLSL Change
  TG->addTimer(*this);
}

void Timer::init(StringRef N, TimerGroup &tg) {
  assert(!TG && "Timer already initialized");
  Name.assign(N.begin(), N.end());
  Started = false;
  TG = &tg;
  TG->addTimer(*this);
}

Timer::~Timer() {
  if (!TG) return;  // Never initialized, or already cleared.
  TG->removeTimer(*this);
}

static inline size_t getMemUsage() {
  if (!TrackSpace) return 0;
  return sys::Process::GetMallocUsage();
}

TimeRecord TimeRecord::getCurrentTime(bool Start) {
  TimeRecord Result;
  sys::TimeValue now(0,0), user(0,0), sys(0,0);
  
  if (Start) {
    Result.MemUsed = getMemUsage();
    sys::Process::GetTimeUsage(now, user, sys);
  } else {
    sys::Process::GetTimeUsage(now, user, sys);
    Result.MemUsed = getMemUsage();
  }

  Result.WallTime   =  now.seconds() +  now.microseconds() / 1000000.0;
  Result.UserTime   = user.seconds() + user.microseconds() / 1000000.0;
  Result.SystemTime =  sys.seconds() +  sys.microseconds() / 1000000.0;
  return Result;
}

static ManagedStatic<std::vector<Timer*> > ActiveTimers;

void Timer::startTimer() {
  Started = true;
  ActiveTimers->push_back(this);
  Time -= TimeRecord::getCurrentTime(true);
}

void Timer::stopTimer() {
  Time += TimeRecord::getCurrentTime(false);

  if (ActiveTimers->back() == this) {
    ActiveTimers->pop_back();
  } else {
    std::vector<Timer*>::iterator I =
      std::find(ActiveTimers->begin(), ActiveTimers->end(), this);
    assert(I != ActiveTimers->end() && "stop but no startTimer?");
    ActiveTimers->erase(I);
  }
}

static void printVal(double Val, double Total, raw_ostream &OS) {
  if (Total < 1e-7)   // Avoid dividing by zero.
    OS << "        -----     ";
  else
    OS << format("  %7.4f (%5.1f%%)", Val, Val*100/Total);
}

void TimeRecord::print(const TimeRecord &Total, raw_ostream &OS) const {
  if (Total.getUserTime())
    printVal(getUserTime(), Total.getUserTime(), OS);
  if (Total.getSystemTime())
    printVal(getSystemTime(), Total.getSystemTime(), OS);
  if (Total.getProcessTime())
    printVal(getProcessTime(), Total.getProcessTime(), OS);
  printVal(getWallTime(), Total.getWallTime(), OS);
  
  OS << "  ";
  
  if (Total.getMemUsed())
    OS << format("%9" PRId64 "  ", (int64_t)getMemUsed());
}


//===----------------------------------------------------------------------===//
//   NamedRegionTimer Implementation
//===----------------------------------------------------------------------===//

namespace {

typedef StringMap<Timer> Name2TimerMap;

class Name2PairMap {
  StringMap<std::pair<TimerGroup*, Name2TimerMap> > Map;
public:
  ~Name2PairMap() {
    for (StringMap<std::pair<TimerGroup*, Name2TimerMap> >::iterator
         I = Map.begin(), E = Map.end(); I != E; ++I)
      delete I->second.first;
  }
  
  Timer &get(StringRef Name, StringRef GroupName) {
    sys::SmartScopedLock<true> L(*TimerLock);
    
    std::pair<TimerGroup*, Name2TimerMap> &GroupEntry = Map[GroupName];
    
    if (!GroupEntry.first)
      GroupEntry.first = new TimerGroup(GroupName);
    
    Timer &T = GroupEntry.second[Name];
    if (!T.isInitialized())
      T.init(Name, *GroupEntry.first);
    return T;
  }
};

}

static ManagedStatic<Name2TimerMap> NamedTimers;
static ManagedStatic<Name2PairMap> NamedGroupedTimers;

static Timer &getNamedRegionTimer(StringRef Name) {
  sys::SmartScopedLock<true> L(*TimerLock);
  
  Timer &T = (*NamedTimers)[Name];
  if (!T.isInitialized())
    T.init(Name);
  return T;
}

NamedRegionTimer::NamedRegionTimer(StringRef Name,
                                   bool Enabled)
  : TimeRegion(!Enabled ? nullptr : &getNamedRegionTimer(Name)) {}

NamedRegionTimer::NamedRegionTimer(StringRef Name, StringRef GroupName,
                                   bool Enabled)
  : TimeRegion(!Enabled ? nullptr : &NamedGroupedTimers->get(Name, GroupName)){}

//===----------------------------------------------------------------------===//
//   TimerGroup Implementation
//===----------------------------------------------------------------------===//

/// TimerGroupList - This is the global list of TimerGroups, maintained by the
/// TimerGroup ctor/dtor and is protected by the TimerLock lock.
static TimerGroup *TimerGroupList = nullptr;

TimerGroup::TimerGroup(StringRef name)
  : Name(name.begin(), name.end()), FirstTimer(nullptr) {
  // HLSL Change Starts - initialize as head without locking
  if (name.equals(DefaultTimerGroupName)) {
    Next = TimerGroupList;
    Prev = &TimerGroupList;
    TimerGroupList = this;
    return;
  }
  // HLSL Change Ends
    
  // Add the group to TimerGroupList.
  sys::SmartScopedLock<true> L(*TimerLock);
  if (TimerGroupList)
    TimerGroupList->Prev = &Next;
  Next = TimerGroupList;
  Prev = &TimerGroupList;
  TimerGroupList = this;
}

TimerGroup::~TimerGroup() {
  // If the timer group is destroyed before the timers it owns, accumulate and
  // print the timing data.
  while (FirstTimer)
    removeTimer(*FirstTimer);
  
  // Remove the group from the TimerGroupList.
  sys::SmartScopedLock<true> L(*TimerLock);
  *Prev = Next;
  if (Next)
    Next->Prev = Prev;
}


void TimerGroup::removeTimer(Timer &T) {
  sys::SmartScopedLock<true> L(*TimerLock);
  
  // If the timer was started, move its data to TimersToPrint.
  if (T.Started)
    TimersToPrint.push_back(std::make_pair(T.Time, T.Name));

  T.TG = nullptr;
  
  // Unlink the timer from our list.
  *T.Prev = T.Next;
  if (T.Next)
    T.Next->Prev = T.Prev;
  
  // Print the report when all timers in this group are destroyed if some of
  // them were started.
  if (FirstTimer || TimersToPrint.empty())
    return;
  
  raw_ostream *OutStream = CreateInfoOutputFile();
  PrintQueuedTimers(*OutStream);
  delete OutStream;   // Close the file.
}

void TimerGroup::addTimer(Timer &T) {
  sys::SmartScopedLock<true> L(*TimerLock);
  
  // Add the timer to our list.
  if (FirstTimer)
    FirstTimer->Prev = &T.Next;
  T.Next = FirstTimer;
  T.Prev = &FirstTimer;
  FirstTimer = &T;
}

void TimerGroup::PrintQueuedTimers(raw_ostream &OS) {
  // Sort the timers in descending order by amount of time taken.
  std::sort(TimersToPrint.begin(), TimersToPrint.end());
  
  TimeRecord Total;
  for (unsigned i = 0, e = TimersToPrint.size(); i != e; ++i)
    Total += TimersToPrint[i].first;
  
  // Print out timing header.
  OS << "===" << std::string(73, '-') << "===\n";
  // Figure out how many spaces to indent TimerGroup name.
  unsigned Padding = (80-Name.length())/2;
  if (Padding > 80) Padding = 0;         // Don't allow "negative" numbers
  OS.indent(Padding) << Name << '\n';
  OS << "===" << std::string(73, '-') << "===\n";
  
  // If this is not an collection of ungrouped times, print the total time.
  // Ungrouped timers don't really make sense to add up.  We still print the
  // TOTAL line to make the percentages make sense.
  if (this == getDefaultTimerGroup()) // HLSL Change
    OS << format("  Total Execution Time: %5.4f seconds (%5.4f wall clock)\n",
                 Total.getProcessTime(), Total.getWallTime());
  OS << '\n';
  
  if (Total.getUserTime())
    OS << "   ---User Time---";
  if (Total.getSystemTime())
    OS << "   --System Time--";
  if (Total.getProcessTime())
    OS << "   --User+System--";
  OS << "   ---Wall Time---";
  if (Total.getMemUsed())
    OS << "  ---Mem---";
  OS << "  --- Name ---\n";
  
  // Loop through all of the timing data, printing it out.
  for (unsigned i = 0, e = TimersToPrint.size(); i != e; ++i) {
    const std::pair<TimeRecord, std::string> &Entry = TimersToPrint[e-i-1];
    Entry.first.print(Total, OS);
    OS << Entry.second << '\n';
  }
  
  Total.print(Total, OS);
  OS << "Total\n\n";
  OS.flush();
  
  TimersToPrint.clear();
}

/// print - Print any started timers in this group and zero them.
void TimerGroup::print(raw_ostream &OS) {
  sys::SmartScopedLock<true> L(*TimerLock);

  // See if any of our timers were started, if so add them to TimersToPrint and
  // reset them.
  for (Timer *T = FirstTimer; T; T = T->Next) {
    if (!T->Started) continue;
    TimersToPrint.push_back(std::make_pair(T->Time, T->Name));
    
    // Clear out the time.
    T->Started = 0;
    T->Time = TimeRecord();
  }

  // If any timers were started, print the group.
  if (!TimersToPrint.empty())
    PrintQueuedTimers(OS);
}

/// printAll - This static method prints all timers and clears them all out.
void TimerGroup::printAll(raw_ostream &OS) {
  sys::SmartScopedLock<true> L(*TimerLock);

  for (TimerGroup *TG = TimerGroupList; TG; TG = TG->Next)
    TG->print(OS);
}
