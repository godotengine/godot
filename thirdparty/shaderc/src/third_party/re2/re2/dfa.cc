// Copyright 2008 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A DFA (deterministic finite automaton)-based regular expression search.
//
// The DFA search has two main parts: the construction of the automaton,
// which is represented by a graph of State structures, and the execution
// of the automaton over a given input string.
//
// The basic idea is that the State graph is constructed so that the
// execution can simply start with a state s, and then for each byte c in
// the input string, execute "s = s->next[c]", checking at each point whether
// the current s represents a matching state.
//
// The simple explanation just given does convey the essence of this code,
// but it omits the details of how the State graph gets constructed as well
// as some performance-driven optimizations to the execution of the automaton.
// All these details are explained in the comments for the code following
// the definition of class DFA.
//
// See http://swtch.com/~rsc/regexp/ for a very bare-bones equivalent.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "util/logging.h"
#include "util/mix.h"
#include "util/mutex.h"
#include "util/pod_array.h"
#include "util/sparse_set.h"
#include "util/strutil.h"
#include "re2/prog.h"
#include "re2/stringpiece.h"

// Silence "zero-sized array in struct/union" warning for DFA::State::next_.
#ifdef _MSC_VER
#pragma warning(disable: 4200)
#endif

namespace re2 {

#if !defined(__linux__)  /* only Linux seems to have memrchr */
static void* memrchr(const void* s, int c, size_t n) {
  const unsigned char* p = (const unsigned char*)s;
  for (p += n; n > 0; n--)
    if (*--p == c)
      return (void*)p;

  return NULL;
}
#endif

// Controls whether the DFA should bail out early if the NFA would be faster.
static bool dfa_should_bail_when_slow = true;

// Changing this to true compiles in prints that trace execution of the DFA.
// Generates a lot of output -- only useful for debugging.
static const bool ExtraDebug = false;

// A DFA implementation of a regular expression program.
// Since this is entirely a forward declaration mandated by C++,
// some of the comments here are better understood after reading
// the comments in the sections that follow the DFA definition.
class DFA {
 public:
  DFA(Prog* prog, Prog::MatchKind kind, int64_t max_mem);
  ~DFA();
  bool ok() const { return !init_failed_; }
  Prog::MatchKind kind() { return kind_; }

  // Searches for the regular expression in text, which is considered
  // as a subsection of context for the purposes of interpreting flags
  // like ^ and $ and \A and \z.
  // Returns whether a match was found.
  // If a match is found, sets *ep to the end point of the best match in text.
  // If "anchored", the match must begin at the start of text.
  // If "want_earliest_match", the match that ends first is used, not
  //   necessarily the best one.
  // If "run_forward" is true, the DFA runs from text.begin() to text.end().
  //   If it is false, the DFA runs from text.end() to text.begin(),
  //   returning the leftmost end of the match instead of the rightmost one.
  // If the DFA cannot complete the search (for example, if it is out of
  //   memory), it sets *failed and returns false.
  bool Search(const StringPiece& text, const StringPiece& context,
              bool anchored, bool want_earliest_match, bool run_forward,
              bool* failed, const char** ep, SparseSet* matches);

  // Builds out all states for the entire DFA.
  // If cb is not empty, it receives one callback per state built.
  // Returns the number of states built.
  // FOR TESTING OR EXPERIMENTAL PURPOSES ONLY.
  int BuildAllStates(const Prog::DFAStateCallback& cb);

  // Computes min and max for matching strings.  Won't return strings
  // bigger than maxlen.
  bool PossibleMatchRange(std::string* min, std::string* max, int maxlen);

  // These data structures are logically private, but C++ makes it too
  // difficult to mark them as such.
  class RWLocker;
  class StateSaver;
  class Workq;

  // A single DFA state.  The DFA is represented as a graph of these
  // States, linked by the next_ pointers.  If in state s and reading
  // byte c, the next state should be s->next_[c].
  struct State {
    inline bool IsMatch() const { return (flag_ & kFlagMatch) != 0; }
    void SaveMatch(std::vector<int>* v);

    int* inst_;         // Instruction pointers in the state.
    int ninst_;         // # of inst_ pointers.
    uint32_t flag_;     // Empty string bitfield flags in effect on the way
                        // into this state, along with kFlagMatch if this
                        // is a matching state.

// Work around the bug affecting flexible array members in GCC 6.x (for x >= 1).
// (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70932)
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && __GNUC_MINOR__ >= 1
    std::atomic<State*> next_[0];   // Outgoing arrows from State,
#else
    std::atomic<State*> next_[];    // Outgoing arrows from State,
#endif

                        // one per input byte class
  };

  enum {
    kByteEndText = 256,         // imaginary byte at end of text

    kFlagEmptyMask = 0xFF,      // State.flag_: bits holding kEmptyXXX flags
    kFlagMatch = 0x0100,        // State.flag_: this is a matching state
    kFlagLastWord = 0x0200,     // State.flag_: last byte was a word char
    kFlagNeedShift = 16,        // needed kEmpty bits are or'ed in shifted left
  };

  struct StateHash {
    size_t operator()(const State* a) const {
      DCHECK(a != NULL);
      HashMix mix(a->flag_);
      for (int i = 0; i < a->ninst_; i++)
        mix.Mix(a->inst_[i]);
      mix.Mix(0);
      return mix.get();
    }
  };

  struct StateEqual {
    bool operator()(const State* a, const State* b) const {
      DCHECK(a != NULL);
      DCHECK(b != NULL);
      if (a == b)
        return true;
      if (a->flag_ != b->flag_)
        return false;
      if (a->ninst_ != b->ninst_)
        return false;
      for (int i = 0; i < a->ninst_; i++)
        if (a->inst_[i] != b->inst_[i])
          return false;
      return true;
    }
  };

  typedef std::unordered_set<State*, StateHash, StateEqual> StateSet;

 private:
  // Special "first_byte" values for a state.  (Values >= 0 denote actual bytes.)
  enum {
    kFbUnknown = -1,   // No analysis has been performed.
    kFbNone = -2,      // The first-byte trick cannot be used.
  };

  enum {
    // Indices into start_ for unanchored searches.
    // Add kStartAnchored for anchored searches.
    kStartBeginText = 0,          // text at beginning of context
    kStartBeginLine = 2,          // text at beginning of line
    kStartAfterWordChar = 4,      // text follows a word character
    kStartAfterNonWordChar = 6,   // text follows non-word character
    kMaxStart = 8,

    kStartAnchored = 1,
  };

  // Resets the DFA State cache, flushing all saved State* information.
  // Releases and reacquires cache_mutex_ via cache_lock, so any
  // State* existing before the call are not valid after the call.
  // Use a StateSaver to preserve important states across the call.
  // cache_mutex_.r <= L < mutex_
  // After: cache_mutex_.w <= L < mutex_
  void ResetCache(RWLocker* cache_lock);

  // Looks up and returns the State corresponding to a Workq.
  // L >= mutex_
  State* WorkqToCachedState(Workq* q, Workq* mq, uint32_t flag);

  // Looks up and returns a State matching the inst, ninst, and flag.
  // L >= mutex_
  State* CachedState(int* inst, int ninst, uint32_t flag);

  // Clear the cache entirely.
  // Must hold cache_mutex_.w or be in destructor.
  void ClearCache();

  // Converts a State into a Workq: the opposite of WorkqToCachedState.
  // L >= mutex_
  void StateToWorkq(State* s, Workq* q);

  // Runs a State on a given byte, returning the next state.
  State* RunStateOnByteUnlocked(State*, int);  // cache_mutex_.r <= L < mutex_
  State* RunStateOnByte(State*, int);          // L >= mutex_

  // Runs a Workq on a given byte followed by a set of empty-string flags,
  // producing a new Workq in nq.  If a match instruction is encountered,
  // sets *ismatch to true.
  // L >= mutex_
  void RunWorkqOnByte(Workq* q, Workq* nq,
                      int c, uint32_t flag, bool* ismatch);

  // Runs a Workq on a set of empty-string flags, producing a new Workq in nq.
  // L >= mutex_
  void RunWorkqOnEmptyString(Workq* q, Workq* nq, uint32_t flag);

  // Adds the instruction id to the Workq, following empty arrows
  // according to flag.
  // L >= mutex_
  void AddToQueue(Workq* q, int id, uint32_t flag);

  // For debugging, returns a text representation of State.
  static std::string DumpState(State* state);

  // For debugging, returns a text representation of a Workq.
  static std::string DumpWorkq(Workq* q);

  // Search parameters
  struct SearchParams {
    SearchParams(const StringPiece& text, const StringPiece& context,
                 RWLocker* cache_lock)
      : text(text), context(context),
        anchored(false),
        want_earliest_match(false),
        run_forward(false),
        start(NULL),
        first_byte(kFbUnknown),
        cache_lock(cache_lock),
        failed(false),
        ep(NULL),
        matches(NULL) { }

    StringPiece text;
    StringPiece context;
    bool anchored;
    bool want_earliest_match;
    bool run_forward;
    State* start;
    int first_byte;
    RWLocker *cache_lock;
    bool failed;     // "out" parameter: whether search gave up
    const char* ep;  // "out" parameter: end pointer for match
    SparseSet* matches;

   private:
    SearchParams(const SearchParams&) = delete;
    SearchParams& operator=(const SearchParams&) = delete;
  };

  // Before each search, the parameters to Search are analyzed by
  // AnalyzeSearch to determine the state in which to start and the
  // "first_byte" for that state, if any.
  struct StartInfo {
    StartInfo() : start(NULL), first_byte(kFbUnknown) {}
    State* start;
    std::atomic<int> first_byte;
  };

  // Fills in params->start and params->first_byte using
  // the other search parameters.  Returns true on success,
  // false on failure.
  // cache_mutex_.r <= L < mutex_
  bool AnalyzeSearch(SearchParams* params);
  bool AnalyzeSearchHelper(SearchParams* params, StartInfo* info,
                           uint32_t flags);

  // The generic search loop, inlined to create specialized versions.
  // cache_mutex_.r <= L < mutex_
  // Might unlock and relock cache_mutex_ via params->cache_lock.
  inline bool InlinedSearchLoop(SearchParams* params,
                                bool have_first_byte,
                                bool want_earliest_match,
                                bool run_forward);

  // The specialized versions of InlinedSearchLoop.  The three letters
  // at the ends of the name denote the true/false values used as the
  // last three parameters of InlinedSearchLoop.
  // cache_mutex_.r <= L < mutex_
  // Might unlock and relock cache_mutex_ via params->cache_lock.
  bool SearchFFF(SearchParams* params);
  bool SearchFFT(SearchParams* params);
  bool SearchFTF(SearchParams* params);
  bool SearchFTT(SearchParams* params);
  bool SearchTFF(SearchParams* params);
  bool SearchTFT(SearchParams* params);
  bool SearchTTF(SearchParams* params);
  bool SearchTTT(SearchParams* params);

  // The main search loop: calls an appropriate specialized version of
  // InlinedSearchLoop.
  // cache_mutex_.r <= L < mutex_
  // Might unlock and relock cache_mutex_ via params->cache_lock.
  bool FastSearchLoop(SearchParams* params);

  // For debugging, a slow search loop that calls InlinedSearchLoop
  // directly -- because the booleans passed are not constants, the
  // loop is not specialized like the SearchFFF etc. versions, so it
  // runs much more slowly.  Useful only for debugging.
  // cache_mutex_.r <= L < mutex_
  // Might unlock and relock cache_mutex_ via params->cache_lock.
  bool SlowSearchLoop(SearchParams* params);

  // Looks up bytes in bytemap_ but handles case c == kByteEndText too.
  int ByteMap(int c) {
    if (c == kByteEndText)
      return prog_->bytemap_range();
    return prog_->bytemap()[c];
  }

  // Constant after initialization.
  Prog* prog_;              // The regular expression program to run.
  Prog::MatchKind kind_;    // The kind of DFA.
  bool init_failed_;        // initialization failed (out of memory)

  Mutex mutex_;  // mutex_ >= cache_mutex_.r

  // Scratch areas, protected by mutex_.
  Workq* q0_;             // Two pre-allocated work queues.
  Workq* q1_;
  PODArray<int> stack_;   // Pre-allocated stack for AddToQueue

  // State* cache.  Many threads use and add to the cache simultaneously,
  // holding cache_mutex_ for reading and mutex_ (above) when adding.
  // If the cache fills and needs to be discarded, the discarding is done
  // while holding cache_mutex_ for writing, to avoid interrupting other
  // readers.  Any State* pointers are only valid while cache_mutex_
  // is held.
  Mutex cache_mutex_;
  int64_t mem_budget_;     // Total memory budget for all States.
  int64_t state_budget_;   // Amount of memory remaining for new States.
  StateSet state_cache_;   // All States computed so far.
  StartInfo start_[kMaxStart];
};

// Shorthand for casting to uint8_t*.
static inline const uint8_t* BytePtr(const void* v) {
  return reinterpret_cast<const uint8_t*>(v);
}

// Work queues

// Marks separate thread groups of different priority
// in the work queue when in leftmost-longest matching mode.
#define Mark (-1)

// Separates the match IDs from the instructions in inst_.
// Used only for "many match" DFA states.
#define MatchSep (-2)

// Internally, the DFA uses a sparse array of
// program instruction pointers as a work queue.
// In leftmost longest mode, marks separate sections
// of workq that started executing at different
// locations in the string (earlier locations first).
class DFA::Workq : public SparseSet {
 public:
  // Constructor: n is number of normal slots, maxmark number of mark slots.
  Workq(int n, int maxmark) :
    SparseSet(n+maxmark),
    n_(n),
    maxmark_(maxmark),
    nextmark_(n),
    last_was_mark_(true) {
  }

  bool is_mark(int i) { return i >= n_; }

  int maxmark() { return maxmark_; }

  void clear() {
    SparseSet::clear();
    nextmark_ = n_;
  }

  void mark() {
    if (last_was_mark_)
      return;
    last_was_mark_ = false;
    SparseSet::insert_new(nextmark_++);
  }

  int size() {
    return n_ + maxmark_;
  }

  void insert(int id) {
    if (contains(id))
      return;
    insert_new(id);
  }

  void insert_new(int id) {
    last_was_mark_ = false;
    SparseSet::insert_new(id);
  }

 private:
  int n_;                // size excluding marks
  int maxmark_;          // maximum number of marks
  int nextmark_;         // id of next mark
  bool last_was_mark_;   // last inserted was mark

  Workq(const Workq&) = delete;
  Workq& operator=(const Workq&) = delete;
};

DFA::DFA(Prog* prog, Prog::MatchKind kind, int64_t max_mem)
  : prog_(prog),
    kind_(kind),
    init_failed_(false),
    q0_(NULL),
    q1_(NULL),
    mem_budget_(max_mem) {
  if (ExtraDebug)
    fprintf(stderr, "\nkind %d\n%s\n", (int)kind_, prog_->DumpUnanchored().c_str());
  int nmark = 0;
  if (kind_ == Prog::kLongestMatch)
    nmark = prog_->size();
  // See DFA::AddToQueue() for why this is so.
  int nstack = prog_->inst_count(kInstCapture) +
               prog_->inst_count(kInstEmptyWidth) +
               prog_->inst_count(kInstNop) +
               nmark + 1;  // + 1 for start inst

  // Account for space needed for DFA, q0, q1, stack.
  mem_budget_ -= sizeof(DFA);
  mem_budget_ -= (prog_->size() + nmark) *
                 (sizeof(int)+sizeof(int)) * 2;  // q0, q1
  mem_budget_ -= nstack * sizeof(int);  // stack
  if (mem_budget_ < 0) {
    init_failed_ = true;
    return;
  }

  state_budget_ = mem_budget_;

  // Make sure there is a reasonable amount of working room left.
  // At minimum, the search requires room for two states in order
  // to limp along, restarting frequently.  We'll get better performance
  // if there is room for a larger number of states, say 20.
  // Note that a state stores list heads only, so we use the program
  // list count for the upper bound, not the program size.
  int nnext = prog_->bytemap_range() + 1;  // + 1 for kByteEndText slot
  int64_t one_state = sizeof(State) + nnext*sizeof(std::atomic<State*>) +
                      (prog_->list_count()+nmark)*sizeof(int);
  if (state_budget_ < 20*one_state) {
    init_failed_ = true;
    return;
  }

  q0_ = new Workq(prog_->size(), nmark);
  q1_ = new Workq(prog_->size(), nmark);
  stack_ = PODArray<int>(nstack);
}

DFA::~DFA() {
  delete q0_;
  delete q1_;
  ClearCache();
}

// In the DFA state graph, s->next[c] == NULL means that the
// state has not yet been computed and needs to be.  We need
// a different special value to signal that s->next[c] is a
// state that can never lead to a match (and thus the search
// can be called off).  Hence DeadState.
#define DeadState reinterpret_cast<State*>(1)

// Signals that the rest of the string matches no matter what it is.
#define FullMatchState reinterpret_cast<State*>(2)

#define SpecialStateMax FullMatchState

// Debugging printouts

// For debugging, returns a string representation of the work queue.
std::string DFA::DumpWorkq(Workq* q) {
  std::string s;
  const char* sep = "";
  for (Workq::iterator it = q->begin(); it != q->end(); ++it) {
    if (q->is_mark(*it)) {
      StringAppendF(&s, "|");
      sep = "";
    } else {
      StringAppendF(&s, "%s%d", sep, *it);
      sep = ",";
    }
  }
  return s;
}

// For debugging, returns a string representation of the state.
std::string DFA::DumpState(State* state) {
  if (state == NULL)
    return "_";
  if (state == DeadState)
    return "X";
  if (state == FullMatchState)
    return "*";
  std::string s;
  const char* sep = "";
  StringAppendF(&s, "(%p)", state);
  for (int i = 0; i < state->ninst_; i++) {
    if (state->inst_[i] == Mark) {
      StringAppendF(&s, "|");
      sep = "";
    } else if (state->inst_[i] == MatchSep) {
      StringAppendF(&s, "||");
      sep = "";
    } else {
      StringAppendF(&s, "%s%d", sep, state->inst_[i]);
      sep = ",";
    }
  }
  StringAppendF(&s, " flag=%#x", state->flag_);
  return s;
}

//////////////////////////////////////////////////////////////////////
//
// DFA state graph construction.
//
// The DFA state graph is a heavily-linked collection of State* structures.
// The state_cache_ is a set of all the State structures ever allocated,
// so that if the same state is reached by two different paths,
// the same State structure can be used.  This reduces allocation
// requirements and also avoids duplication of effort across the two
// identical states.
//
// A State is defined by an ordered list of instruction ids and a flag word.
//
// The choice of an ordered list of instructions differs from a typical
// textbook DFA implementation, which would use an unordered set.
// Textbook descriptions, however, only care about whether
// the DFA matches, not where it matches in the text.  To decide where the
// DFA matches, we need to mimic the behavior of the dominant backtracking
// implementations like PCRE, which try one possible regular expression
// execution, then another, then another, stopping when one of them succeeds.
// The DFA execution tries these many executions in parallel, representing
// each by an instruction id.  These pointers are ordered in the State.inst_
// list in the same order that the executions would happen in a backtracking
// search: if a match is found during execution of inst_[2], inst_[i] for i>=3
// can be discarded.
//
// Textbooks also typically do not consider context-aware empty string operators
// like ^ or $.  These are handled by the flag word, which specifies the set
// of empty-string operators that should be matched when executing at the
// current text position.  These flag bits are defined in prog.h.
// The flag word also contains two DFA-specific bits: kFlagMatch if the state
// is a matching state (one that reached a kInstMatch in the program)
// and kFlagLastWord if the last processed byte was a word character, for the
// implementation of \B and \b.
//
// The flag word also contains, shifted up 16 bits, the bits looked for by
// any kInstEmptyWidth instructions in the state.  These provide a useful
// summary indicating when new flags might be useful.
//
// The permanent representation of a State's instruction ids is just an array,
// but while a state is being analyzed, these instruction ids are represented
// as a Workq, which is an array that allows iteration in insertion order.

// NOTE(rsc): The choice of State construction determines whether the DFA
// mimics backtracking implementations (so-called leftmost first matching) or
// traditional DFA implementations (so-called leftmost longest matching as
// prescribed by POSIX).  This implementation chooses to mimic the
// backtracking implementations, because we want to replace PCRE.  To get
// POSIX behavior, the states would need to be considered not as a simple
// ordered list of instruction ids, but as a list of unordered sets of instruction
// ids.  A match by a state in one set would inhibit the running of sets
// farther down the list but not other instruction ids in the same set.  Each
// set would correspond to matches beginning at a given point in the string.
// This is implemented by separating different sets with Mark pointers.

// Looks in the State cache for a State matching q, flag.
// If one is found, returns it.  If one is not found, allocates one,
// inserts it in the cache, and returns it.
// If mq is not null, MatchSep and the match IDs in mq will be appended
// to the State.
DFA::State* DFA::WorkqToCachedState(Workq* q, Workq* mq, uint32_t flag) {
  //mutex_.AssertHeld();

  // Construct array of instruction ids for the new state.
  // Only ByteRange, EmptyWidth, and Match instructions are useful to keep:
  // those are the only operators with any effect in
  // RunWorkqOnEmptyString or RunWorkqOnByte.
  int* inst = new int[q->size()];
  int n = 0;
  uint32_t needflags = 0;  // flags needed by kInstEmptyWidth instructions
  bool sawmatch = false;   // whether queue contains guaranteed kInstMatch
  bool sawmark = false;    // whether queue contains a Mark
  if (ExtraDebug)
    fprintf(stderr, "WorkqToCachedState %s [%#x]", DumpWorkq(q).c_str(), flag);
  for (Workq::iterator it = q->begin(); it != q->end(); ++it) {
    int id = *it;
    if (sawmatch && (kind_ == Prog::kFirstMatch || q->is_mark(id)))
      break;
    if (q->is_mark(id)) {
      if (n > 0 && inst[n-1] != Mark) {
        sawmark = true;
        inst[n++] = Mark;
      }
      continue;
    }
    Prog::Inst* ip = prog_->inst(id);
    switch (ip->opcode()) {
      case kInstAltMatch:
        // This state will continue to a match no matter what
        // the rest of the input is.  If it is the highest priority match
        // being considered, return the special FullMatchState
        // to indicate that it's all matches from here out.
        if (kind_ != Prog::kManyMatch &&
            (kind_ != Prog::kFirstMatch ||
             (it == q->begin() && ip->greedy(prog_))) &&
            (kind_ != Prog::kLongestMatch || !sawmark) &&
            (flag & kFlagMatch)) {
          delete[] inst;
          if (ExtraDebug)
            fprintf(stderr, " -> FullMatchState\n");
          return FullMatchState;
        }
        FALLTHROUGH_INTENDED;
      default:
        // Record iff id is the head of its list, which must
        // be the case if id-1 is the last of *its* list. :)
        if (prog_->inst(id-1)->last())
          inst[n++] = *it;
        if (ip->opcode() == kInstEmptyWidth)
          needflags |= ip->empty();
        if (ip->opcode() == kInstMatch && !prog_->anchor_end())
          sawmatch = true;
        break;
    }
  }
  DCHECK_LE(n, q->size());
  if (n > 0 && inst[n-1] == Mark)
    n--;

  // If there are no empty-width instructions waiting to execute,
  // then the extra flag bits will not be used, so there is no
  // point in saving them.  (Discarding them reduces the number
  // of distinct states.)
  if (needflags == 0)
    flag &= kFlagMatch;

  // NOTE(rsc): The code above cannot do flag &= needflags,
  // because if the right flags were present to pass the current
  // kInstEmptyWidth instructions, new kInstEmptyWidth instructions
  // might be reached that in turn need different flags.
  // The only sure thing is that if there are no kInstEmptyWidth
  // instructions at all, no flags will be needed.
  // We could do the extra work to figure out the full set of
  // possibly needed flags by exploring past the kInstEmptyWidth
  // instructions, but the check above -- are any flags needed
  // at all? -- handles the most common case.  More fine-grained
  // analysis can only be justified by measurements showing that
  // too many redundant states are being allocated.

  // If there are no Insts in the list, it's a dead state,
  // which is useful to signal with a special pointer so that
  // the execution loop can stop early.  This is only okay
  // if the state is *not* a matching state.
  if (n == 0 && flag == 0) {
    delete[] inst;
    if (ExtraDebug)
      fprintf(stderr, " -> DeadState\n");
    return DeadState;
  }

  // If we're in longest match mode, the state is a sequence of
  // unordered state sets separated by Marks.  Sort each set
  // to canonicalize, to reduce the number of distinct sets stored.
  if (kind_ == Prog::kLongestMatch) {
    int* ip = inst;
    int* ep = ip + n;
    while (ip < ep) {
      int* markp = ip;
      while (markp < ep && *markp != Mark)
        markp++;
      std::sort(ip, markp);
      if (markp < ep)
        markp++;
      ip = markp;
    }
  }

  // Append MatchSep and the match IDs in mq if necessary.
  if (mq != NULL) {
    inst[n++] = MatchSep;
    for (Workq::iterator i = mq->begin(); i != mq->end(); ++i) {
      int id = *i;
      Prog::Inst* ip = prog_->inst(id);
      if (ip->opcode() == kInstMatch)
        inst[n++] = ip->match_id();
    }
  }

  // Save the needed empty-width flags in the top bits for use later.
  flag |= needflags << kFlagNeedShift;

  State* state = CachedState(inst, n, flag);
  delete[] inst;
  return state;
}

// Looks in the State cache for a State matching inst, ninst, flag.
// If one is found, returns it.  If one is not found, allocates one,
// inserts it in the cache, and returns it.
DFA::State* DFA::CachedState(int* inst, int ninst, uint32_t flag) {
  //mutex_.AssertHeld();

  // Look in the cache for a pre-existing state.
  // We have to initialise the struct like this because otherwise
  // MSVC will complain about the flexible array member. :(
  State state;
  state.inst_ = inst;
  state.ninst_ = ninst;
  state.flag_ = flag;
  StateSet::iterator it = state_cache_.find(&state);
  if (it != state_cache_.end()) {
    if (ExtraDebug)
      fprintf(stderr, " -cached-> %s\n", DumpState(*it).c_str());
    return *it;
  }

  // Must have enough memory for new state.
  // In addition to what we're going to allocate,
  // the state cache hash table seems to incur about 40 bytes per
  // State*, empirically.
  const int kStateCacheOverhead = 40;
  int nnext = prog_->bytemap_range() + 1;  // + 1 for kByteEndText slot
  int mem = sizeof(State) + nnext*sizeof(std::atomic<State*>) +
            ninst*sizeof(int);
  if (mem_budget_ < mem + kStateCacheOverhead) {
    mem_budget_ = -1;
    return NULL;
  }
  mem_budget_ -= mem + kStateCacheOverhead;

  // Allocate new state along with room for next_ and inst_.
  char* space = std::allocator<char>().allocate(mem);
  State* s = new (space) State;
  (void) new (s->next_) std::atomic<State*>[nnext];
  // Work around a unfortunate bug in older versions of libstdc++.
  // (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64658)
  for (int i = 0; i < nnext; i++)
    (void) new (s->next_ + i) std::atomic<State*>(NULL);
  s->inst_ = new (s->next_ + nnext) int[ninst];
  memmove(s->inst_, inst, ninst*sizeof s->inst_[0]);
  s->ninst_ = ninst;
  s->flag_ = flag;
  if (ExtraDebug)
    fprintf(stderr, " -> %s\n", DumpState(s).c_str());

  // Put state in cache and return it.
  state_cache_.insert(s);
  return s;
}

// Clear the cache.  Must hold cache_mutex_.w or be in destructor.
void DFA::ClearCache() {
  StateSet::iterator begin = state_cache_.begin();
  StateSet::iterator end = state_cache_.end();
  while (begin != end) {
    StateSet::iterator tmp = begin;
    ++begin;
    // Deallocate the blob of memory that we allocated in DFA::CachedState().
    // We recompute mem in order to benefit from sized delete where possible.
    int ninst = (*tmp)->ninst_;
    int nnext = prog_->bytemap_range() + 1;  // + 1 for kByteEndText slot
    int mem = sizeof(State) + nnext*sizeof(std::atomic<State*>) +
              ninst*sizeof(int);
    std::allocator<char>().deallocate(reinterpret_cast<char*>(*tmp), mem);
  }
  state_cache_.clear();
}

// Copies insts in state s to the work queue q.
void DFA::StateToWorkq(State* s, Workq* q) {
  q->clear();
  for (int i = 0; i < s->ninst_; i++) {
    if (s->inst_[i] == Mark) {
      q->mark();
    } else if (s->inst_[i] == MatchSep) {
      // Nothing after this is an instruction!
      break;
    } else {
      // Explore from the head of the list.
      AddToQueue(q, s->inst_[i], s->flag_ & kFlagEmptyMask);
    }
  }
}

// Adds ip to the work queue, following empty arrows according to flag.
void DFA::AddToQueue(Workq* q, int id, uint32_t flag) {

  // Use stack_ to hold our stack of instructions yet to process.
  // It was preallocated as follows:
  //   one entry per Capture;
  //   one entry per EmptyWidth; and
  //   one entry per Nop.
  // This reflects the maximum number of stack pushes that each can
  // perform. (Each instruction can be processed at most once.)
  // When using marks, we also added nmark == prog_->size().
  // (Otherwise, nmark == 0.)
  int* stk = stack_.data();
  int nstk = 0;

  stk[nstk++] = id;
  while (nstk > 0) {
    DCHECK_LE(nstk, stack_.size());
    id = stk[--nstk];

  Loop:
    if (id == Mark) {
      q->mark();
      continue;
    }

    if (id == 0)
      continue;

    // If ip is already on the queue, nothing to do.
    // Otherwise add it.  We don't actually keep all the
    // ones that get added, but adding all of them here
    // increases the likelihood of q->contains(id),
    // reducing the amount of duplicated work.
    if (q->contains(id))
      continue;
    q->insert_new(id);

    // Process instruction.
    Prog::Inst* ip = prog_->inst(id);
    switch (ip->opcode()) {
      default:
        LOG(DFATAL) << "unhandled opcode: " << ip->opcode();
        break;

      case kInstByteRange:  // just save these on the queue
      case kInstMatch:
        if (ip->last())
          break;
        id = id+1;
        goto Loop;

      case kInstCapture:    // DFA treats captures as no-ops.
      case kInstNop:
        if (!ip->last())
          stk[nstk++] = id+1;

        // If this instruction is the [00-FF]* loop at the beginning of
        // a leftmost-longest unanchored search, separate with a Mark so
        // that future threads (which will start farther to the right in
        // the input string) are lower priority than current threads.
        if (ip->opcode() == kInstNop && q->maxmark() > 0 &&
            id == prog_->start_unanchored() && id != prog_->start())
          stk[nstk++] = Mark;
        id = ip->out();
        goto Loop;

      case kInstAltMatch:
        DCHECK(!ip->last());
        id = id+1;
        goto Loop;

      case kInstEmptyWidth:
        if (!ip->last())
          stk[nstk++] = id+1;

        // Continue on if we have all the right flag bits.
        if (ip->empty() & ~flag)
          break;
        id = ip->out();
        goto Loop;
    }
  }
}

// Running of work queues.  In the work queue, order matters:
// the queue is sorted in priority order.  If instruction i comes before j,
// then the instructions that i produces during the run must come before
// the ones that j produces.  In order to keep this invariant, all the
// work queue runners have to take an old queue to process and then
// also a new queue to fill in.  It's not acceptable to add to the end of
// an existing queue, because new instructions will not end up in the
// correct position.

// Runs the work queue, processing the empty strings indicated by flag.
// For example, flag == kEmptyBeginLine|kEmptyEndLine means to match
// both ^ and $.  It is important that callers pass all flags at once:
// processing both ^ and $ is not the same as first processing only ^
// and then processing only $.  Doing the two-step sequence won't match
// ^$^$^$ but processing ^ and $ simultaneously will (and is the behavior
// exhibited by existing implementations).
void DFA::RunWorkqOnEmptyString(Workq* oldq, Workq* newq, uint32_t flag) {
  newq->clear();
  for (Workq::iterator i = oldq->begin(); i != oldq->end(); ++i) {
    if (oldq->is_mark(*i))
      AddToQueue(newq, Mark, flag);
    else
      AddToQueue(newq, *i, flag);
  }
}

// Runs the work queue, processing the single byte c followed by any empty
// strings indicated by flag.  For example, c == 'a' and flag == kEmptyEndLine,
// means to match c$.  Sets the bool *ismatch to true if the end of the
// regular expression program has been reached (the regexp has matched).
void DFA::RunWorkqOnByte(Workq* oldq, Workq* newq,
                         int c, uint32_t flag, bool* ismatch) {
  //mutex_.AssertHeld();

  newq->clear();
  for (Workq::iterator i = oldq->begin(); i != oldq->end(); ++i) {
    if (oldq->is_mark(*i)) {
      if (*ismatch)
        return;
      newq->mark();
      continue;
    }
    int id = *i;
    Prog::Inst* ip = prog_->inst(id);
    switch (ip->opcode()) {
      default:
        LOG(DFATAL) << "unhandled opcode: " << ip->opcode();
        break;

      case kInstFail:        // never succeeds
      case kInstCapture:     // already followed
      case kInstNop:         // already followed
      case kInstAltMatch:    // already followed
      case kInstEmptyWidth:  // already followed
        break;

      case kInstByteRange:   // can follow if c is in range
        if (ip->Matches(c))
          AddToQueue(newq, ip->out(), flag);
        break;

      case kInstMatch:
        if (prog_->anchor_end() && c != kByteEndText &&
            kind_ != Prog::kManyMatch)
          break;
        *ismatch = true;
        if (kind_ == Prog::kFirstMatch) {
          // Can stop processing work queue since we found a match.
          return;
        }
        break;
    }
  }

  if (ExtraDebug)
    fprintf(stderr, "%s on %d[%#x] -> %s [%d]\n", DumpWorkq(oldq).c_str(),
            c, flag, DumpWorkq(newq).c_str(), *ismatch);
}

// Processes input byte c in state, returning new state.
// Caller does not hold mutex.
DFA::State* DFA::RunStateOnByteUnlocked(State* state, int c) {
  // Keep only one RunStateOnByte going
  // even if the DFA is being run by multiple threads.
  MutexLock l(&mutex_);
  return RunStateOnByte(state, c);
}

// Processes input byte c in state, returning new state.
DFA::State* DFA::RunStateOnByte(State* state, int c) {
  //mutex_.AssertHeld();

  if (state <= SpecialStateMax) {
    if (state == FullMatchState) {
      // It is convenient for routines like PossibleMatchRange
      // if we implement RunStateOnByte for FullMatchState:
      // once you get into this state you never get out,
      // so it's pretty easy.
      return FullMatchState;
    }
    if (state == DeadState) {
      LOG(DFATAL) << "DeadState in RunStateOnByte";
      return NULL;
    }
    if (state == NULL) {
      LOG(DFATAL) << "NULL state in RunStateOnByte";
      return NULL;
    }
    LOG(DFATAL) << "Unexpected special state in RunStateOnByte";
    return NULL;
  }

  // If someone else already computed this, return it.
  State* ns = state->next_[ByteMap(c)].load(std::memory_order_relaxed);
  if (ns != NULL)
    return ns;

  // Convert state into Workq.
  StateToWorkq(state, q0_);

  // Flags marking the kinds of empty-width things (^ $ etc)
  // around this byte.  Before the byte we have the flags recorded
  // in the State structure itself.  After the byte we have
  // nothing yet (but that will change: read on).
  uint32_t needflag = state->flag_ >> kFlagNeedShift;
  uint32_t beforeflag = state->flag_ & kFlagEmptyMask;
  uint32_t oldbeforeflag = beforeflag;
  uint32_t afterflag = 0;

  if (c == '\n') {
    // Insert implicit $ and ^ around \n
    beforeflag |= kEmptyEndLine;
    afterflag |= kEmptyBeginLine;
  }

  if (c == kByteEndText) {
    // Insert implicit $ and \z before the fake "end text" byte.
    beforeflag |= kEmptyEndLine | kEmptyEndText;
  }

  // The state flag kFlagLastWord says whether the last
  // byte processed was a word character.  Use that info to
  // insert empty-width (non-)word boundaries.
  bool islastword = (state->flag_ & kFlagLastWord) != 0;
  bool isword = c != kByteEndText && Prog::IsWordChar(static_cast<uint8_t>(c));
  if (isword == islastword)
    beforeflag |= kEmptyNonWordBoundary;
  else
    beforeflag |= kEmptyWordBoundary;

  // Okay, finally ready to run.
  // Only useful to rerun on empty string if there are new, useful flags.
  if (beforeflag & ~oldbeforeflag & needflag) {
    RunWorkqOnEmptyString(q0_, q1_, beforeflag);
    using std::swap;
    swap(q0_, q1_);
  }
  bool ismatch = false;
  RunWorkqOnByte(q0_, q1_, c, afterflag, &ismatch);
  using std::swap;
  swap(q0_, q1_);

  // Save afterflag along with ismatch and isword in new state.
  uint32_t flag = afterflag;
  if (ismatch)
    flag |= kFlagMatch;
  if (isword)
    flag |= kFlagLastWord;

  if (ismatch && kind_ == Prog::kManyMatch)
    ns = WorkqToCachedState(q0_, q1_, flag);
  else
    ns = WorkqToCachedState(q0_, NULL, flag);

  // Flush ns before linking to it.
  // Write barrier before updating state->next_ so that the
  // main search loop can proceed without any locking, for speed.
  // (Otherwise it would need one mutex operation per input byte.)
  state->next_[ByteMap(c)].store(ns, std::memory_order_release);
  return ns;
}


//////////////////////////////////////////////////////////////////////
// DFA cache reset.

// Reader-writer lock helper.
//
// The DFA uses a reader-writer mutex to protect the state graph itself.
// Traversing the state graph requires holding the mutex for reading,
// and discarding the state graph and starting over requires holding the
// lock for writing.  If a search needs to expand the graph but is out
// of memory, it will need to drop its read lock and then acquire the
// write lock.  Since it cannot then atomically downgrade from write lock
// to read lock, it runs the rest of the search holding the write lock.
// (This probably helps avoid repeated contention, but really the decision
// is forced by the Mutex interface.)  It's a bit complicated to keep
// track of whether the lock is held for reading or writing and thread
// that through the search, so instead we encapsulate it in the RWLocker
// and pass that around.

class DFA::RWLocker {
 public:
  explicit RWLocker(Mutex* mu);
  ~RWLocker();

  // If the lock is only held for reading right now,
  // drop the read lock and re-acquire for writing.
  // Subsequent calls to LockForWriting are no-ops.
  // Notice that the lock is *released* temporarily.
  void LockForWriting();

 private:
  Mutex* mu_;
  bool writing_;

  RWLocker(const RWLocker&) = delete;
  RWLocker& operator=(const RWLocker&) = delete;
};

DFA::RWLocker::RWLocker(Mutex* mu) : mu_(mu), writing_(false) {
  mu_->ReaderLock();
}

// This function is marked as NO_THREAD_SAFETY_ANALYSIS because the annotations
// does not support lock upgrade.
void DFA::RWLocker::LockForWriting() NO_THREAD_SAFETY_ANALYSIS {
  if (!writing_) {
    mu_->ReaderUnlock();
    mu_->WriterLock();
    writing_ = true;
  }
}

DFA::RWLocker::~RWLocker() {
  if (!writing_)
    mu_->ReaderUnlock();
  else
    mu_->WriterUnlock();
}


// When the DFA's State cache fills, we discard all the states in the
// cache and start over.  Many threads can be using and adding to the
// cache at the same time, so we synchronize using the cache_mutex_
// to keep from stepping on other threads.  Specifically, all the
// threads using the current cache hold cache_mutex_ for reading.
// When a thread decides to flush the cache, it drops cache_mutex_
// and then re-acquires it for writing.  That ensures there are no
// other threads accessing the cache anymore.  The rest of the search
// runs holding cache_mutex_ for writing, avoiding any contention
// with or cache pollution caused by other threads.

void DFA::ResetCache(RWLocker* cache_lock) {
  // Re-acquire the cache_mutex_ for writing (exclusive use).
  cache_lock->LockForWriting();

  // Clear the cache, reset the memory budget.
  for (int i = 0; i < kMaxStart; i++) {
    start_[i].start = NULL;
    start_[i].first_byte.store(kFbUnknown, std::memory_order_relaxed);
  }
  ClearCache();
  mem_budget_ = state_budget_;
}

// Typically, a couple States do need to be preserved across a cache
// reset, like the State at the current point in the search.
// The StateSaver class helps keep States across cache resets.
// It makes a copy of the state's guts outside the cache (before the reset)
// and then can be asked, after the reset, to recreate the State
// in the new cache.  For example, in a DFA method ("this" is a DFA):
//
//   StateSaver saver(this, s);
//   ResetCache(cache_lock);
//   s = saver.Restore();
//
// The saver should always have room in the cache to re-create the state,
// because resetting the cache locks out all other threads, and the cache
// is known to have room for at least a couple states (otherwise the DFA
// constructor fails).

class DFA::StateSaver {
 public:
  explicit StateSaver(DFA* dfa, State* state);
  ~StateSaver();

  // Recreates and returns a state equivalent to the
  // original state passed to the constructor.
  // Returns NULL if the cache has filled, but
  // since the DFA guarantees to have room in the cache
  // for a couple states, should never return NULL
  // if used right after ResetCache.
  State* Restore();

 private:
  DFA* dfa_;         // the DFA to use
  int* inst_;        // saved info from State
  int ninst_;
  uint32_t flag_;
  bool is_special_;  // whether original state was special
  State* special_;   // if is_special_, the original state

  StateSaver(const StateSaver&) = delete;
  StateSaver& operator=(const StateSaver&) = delete;
};

DFA::StateSaver::StateSaver(DFA* dfa, State* state) {
  dfa_ = dfa;
  if (state <= SpecialStateMax) {
    inst_ = NULL;
    ninst_ = 0;
    flag_ = 0;
    is_special_ = true;
    special_ = state;
    return;
  }
  is_special_ = false;
  special_ = NULL;
  flag_ = state->flag_;
  ninst_ = state->ninst_;
  inst_ = new int[ninst_];
  memmove(inst_, state->inst_, ninst_*sizeof inst_[0]);
}

DFA::StateSaver::~StateSaver() {
  if (!is_special_)
    delete[] inst_;
}

DFA::State* DFA::StateSaver::Restore() {
  if (is_special_)
    return special_;
  MutexLock l(&dfa_->mutex_);
  State* s = dfa_->CachedState(inst_, ninst_, flag_);
  if (s == NULL)
    LOG(DFATAL) << "StateSaver failed to restore state.";
  return s;
}


//////////////////////////////////////////////////////////////////////
//
// DFA execution.
//
// The basic search loop is easy: start in a state s and then for each
// byte c in the input, s = s->next[c].
//
// This simple description omits a few efficiency-driven complications.
//
// First, the State graph is constructed incrementally: it is possible
// that s->next[c] is null, indicating that that state has not been
// fully explored.  In this case, RunStateOnByte must be invoked to
// determine the next state, which is cached in s->next[c] to save
// future effort.  An alternative reason for s->next[c] to be null is
// that the DFA has reached a so-called "dead state", in which any match
// is no longer possible.  In this case RunStateOnByte will return NULL
// and the processing of the string can stop early.
//
// Second, a 256-element pointer array for s->next_ makes each State
// quite large (2kB on 64-bit machines).  Instead, dfa->bytemap_[]
// maps from bytes to "byte classes" and then next_ only needs to have
// as many pointers as there are byte classes.  A byte class is simply a
// range of bytes that the regexp never distinguishes between.
// A regexp looking for a[abc] would have four byte ranges -- 0 to 'a'-1,
// 'a', 'b' to 'c', and 'c' to 0xFF.  The bytemap slows us a little bit
// but in exchange we typically cut the size of a State (and thus our
// memory footprint) by about 5-10x.  The comments still refer to
// s->next[c] for simplicity, but code should refer to s->next_[bytemap_[c]].
//
// Third, it is common for a DFA for an unanchored match to begin in a
// state in which only one particular byte value can take the DFA to a
// different state.  That is, s->next[c] != s for only one c.  In this
// situation, the DFA can do better than executing the simple loop.
// Instead, it can call memchr to search very quickly for the byte c.
// Whether the start state has this property is determined during a
// pre-compilation pass, and if so, the byte b is passed to the search
// loop as the "first_byte" argument, along with a boolean "have_first_byte".
//
// Fourth, the desired behavior is to search for the leftmost-best match
// (approximately, the same one that Perl would find), which is not
// necessarily the match ending earliest in the string.  Each time a
// match is found, it must be noted, but the DFA must continue on in
// hope of finding a higher-priority match.  In some cases, the caller only
// cares whether there is any match at all, not which one is found.
// The "want_earliest_match" flag causes the search to stop at the first
// match found.
//
// Fifth, one algorithm that uses the DFA needs it to run over the
// input string backward, beginning at the end and ending at the beginning.
// Passing false for the "run_forward" flag causes the DFA to run backward.
//
// The checks for these last three cases, which in a naive implementation
// would be performed once per input byte, slow the general loop enough
// to merit specialized versions of the search loop for each of the
// eight possible settings of the three booleans.  Rather than write
// eight different functions, we write one general implementation and then
// inline it to create the specialized ones.
//
// Note that matches are delayed by one byte, to make it easier to
// accomodate match conditions depending on the next input byte (like $ and \b).
// When s->next[c]->IsMatch(), it means that there is a match ending just
// *before* byte c.

// The generic search loop.  Searches text for a match, returning
// the pointer to the end of the chosen match, or NULL if no match.
// The bools are equal to the same-named variables in params, but
// making them function arguments lets the inliner specialize
// this function to each combination (see two paragraphs above).
inline bool DFA::InlinedSearchLoop(SearchParams* params,
                                   bool have_first_byte,
                                   bool want_earliest_match,
                                   bool run_forward) {
  State* start = params->start;
  const uint8_t* bp = BytePtr(params->text.begin());  // start of text
  const uint8_t* p = bp;                              // text scanning point
  const uint8_t* ep = BytePtr(params->text.end());    // end of text
  const uint8_t* resetp = NULL;                       // p at last cache reset
  if (!run_forward) {
    using std::swap;
    swap(p, ep);
  }

  const uint8_t* bytemap = prog_->bytemap();
  const uint8_t* lastmatch = NULL;   // most recent matching position in text
  bool matched = false;

  State* s = start;
  if (ExtraDebug)
    fprintf(stderr, "@stx: %s\n", DumpState(s).c_str());

  if (s->IsMatch()) {
    matched = true;
    lastmatch = p;
    if (ExtraDebug)
      fprintf(stderr, "match @stx! [%s]\n", DumpState(s).c_str());
    if (params->matches != NULL && kind_ == Prog::kManyMatch) {
      for (int i = s->ninst_ - 1; i >= 0; i--) {
        int id = s->inst_[i];
        if (id == MatchSep)
          break;
        params->matches->insert(id);
      }
    }
    if (want_earliest_match) {
      params->ep = reinterpret_cast<const char*>(lastmatch);
      return true;
    }
  }

  while (p != ep) {
    if (ExtraDebug)
      fprintf(stderr, "@%td: %s\n",
              p - bp, DumpState(s).c_str());

    if (have_first_byte && s == start) {
      // In start state, only way out is to find first_byte,
      // so use optimized assembly in memchr to skip ahead.
      // If first_byte isn't found, we can skip to the end
      // of the string.
      if (run_forward) {
        if ((p = BytePtr(memchr(p, params->first_byte, ep - p))) == NULL) {
          p = ep;
          break;
        }
      } else {
        if ((p = BytePtr(memrchr(ep, params->first_byte, p - ep))) == NULL) {
          p = ep;
          break;
        }
        p++;
      }
    }

    int c;
    if (run_forward)
      c = *p++;
    else
      c = *--p;

    // Note that multiple threads might be consulting
    // s->next_[bytemap[c]] simultaneously.
    // RunStateOnByte takes care of the appropriate locking,
    // including a memory barrier so that the unlocked access
    // (sometimes known as "double-checked locking") is safe.
    // The alternative would be either one DFA per thread
    // or one mutex operation per input byte.
    //
    // ns == DeadState means the state is known to be dead
    // (no more matches are possible).
    // ns == NULL means the state has not yet been computed
    // (need to call RunStateOnByteUnlocked).
    // RunStateOnByte returns ns == NULL if it is out of memory.
    // ns == FullMatchState means the rest of the string matches.
    //
    // Okay to use bytemap[] not ByteMap() here, because
    // c is known to be an actual byte and not kByteEndText.

    State* ns = s->next_[bytemap[c]].load(std::memory_order_acquire);
    if (ns == NULL) {
      ns = RunStateOnByteUnlocked(s, c);
      if (ns == NULL) {
        // After we reset the cache, we hold cache_mutex exclusively,
        // so if resetp != NULL, it means we filled the DFA state
        // cache with this search alone (without any other threads).
        // Benchmarks show that doing a state computation on every
        // byte runs at about 0.2 MB/s, while the NFA (nfa.cc) can do the
        // same at about 2 MB/s.  Unless we're processing an average
        // of 10 bytes per state computation, fail so that RE2 can
        // fall back to the NFA.
        if (dfa_should_bail_when_slow && resetp != NULL &&
            static_cast<size_t>(p - resetp) < 10*state_cache_.size()) {
          params->failed = true;
          return false;
        }
        resetp = p;

        // Prepare to save start and s across the reset.
        StateSaver save_start(this, start);
        StateSaver save_s(this, s);

        // Discard all the States in the cache.
        ResetCache(params->cache_lock);

        // Restore start and s so we can continue.
        if ((start = save_start.Restore()) == NULL ||
            (s = save_s.Restore()) == NULL) {
          // Restore already did LOG(DFATAL).
          params->failed = true;
          return false;
        }
        ns = RunStateOnByteUnlocked(s, c);
        if (ns == NULL) {
          LOG(DFATAL) << "RunStateOnByteUnlocked failed after ResetCache";
          params->failed = true;
          return false;
        }
      }
    }
    if (ns <= SpecialStateMax) {
      if (ns == DeadState) {
        params->ep = reinterpret_cast<const char*>(lastmatch);
        return matched;
      }
      // FullMatchState
      params->ep = reinterpret_cast<const char*>(ep);
      return true;
    }

    s = ns;
    if (s->IsMatch()) {
      matched = true;
      // The DFA notices the match one byte late,
      // so adjust p before using it in the match.
      if (run_forward)
        lastmatch = p - 1;
      else
        lastmatch = p + 1;
      if (ExtraDebug)
        fprintf(stderr, "match @%td! [%s]\n",
                lastmatch - bp, DumpState(s).c_str());
      if (params->matches != NULL && kind_ == Prog::kManyMatch) {
        for (int i = s->ninst_ - 1; i >= 0; i--) {
          int id = s->inst_[i];
          if (id == MatchSep)
            break;
          params->matches->insert(id);
        }
      }
      if (want_earliest_match) {
        params->ep = reinterpret_cast<const char*>(lastmatch);
        return true;
      }
    }
  }

  // Process one more byte to see if it triggers a match.
  // (Remember, matches are delayed one byte.)
  if (ExtraDebug)
    fprintf(stderr, "@etx: %s\n", DumpState(s).c_str());

  int lastbyte;
  if (run_forward) {
    if (params->text.end() == params->context.end())
      lastbyte = kByteEndText;
    else
      lastbyte = params->text.end()[0] & 0xFF;
  } else {
    if (params->text.begin() == params->context.begin())
      lastbyte = kByteEndText;
    else
      lastbyte = params->text.begin()[-1] & 0xFF;
  }

  State* ns = s->next_[ByteMap(lastbyte)].load(std::memory_order_acquire);
  if (ns == NULL) {
    ns = RunStateOnByteUnlocked(s, lastbyte);
    if (ns == NULL) {
      StateSaver save_s(this, s);
      ResetCache(params->cache_lock);
      if ((s = save_s.Restore()) == NULL) {
        params->failed = true;
        return false;
      }
      ns = RunStateOnByteUnlocked(s, lastbyte);
      if (ns == NULL) {
        LOG(DFATAL) << "RunStateOnByteUnlocked failed after Reset";
        params->failed = true;
        return false;
      }
    }
  }
  if (ns <= SpecialStateMax) {
    if (ns == DeadState) {
      params->ep = reinterpret_cast<const char*>(lastmatch);
      return matched;
    }
    // FullMatchState
    params->ep = reinterpret_cast<const char*>(ep);
    return true;
  }

  s = ns;
  if (s->IsMatch()) {
    matched = true;
    lastmatch = p;
    if (ExtraDebug)
      fprintf(stderr, "match @etx! [%s]\n", DumpState(s).c_str());
    if (params->matches != NULL && kind_ == Prog::kManyMatch) {
      for (int i = s->ninst_ - 1; i >= 0; i--) {
        int id = s->inst_[i];
        if (id == MatchSep)
          break;
        params->matches->insert(id);
      }
    }
  }

  params->ep = reinterpret_cast<const char*>(lastmatch);
  return matched;
}

// Inline specializations of the general loop.
bool DFA::SearchFFF(SearchParams* params) {
  return InlinedSearchLoop(params, 0, 0, 0);
}
bool DFA::SearchFFT(SearchParams* params) {
  return InlinedSearchLoop(params, 0, 0, 1);
}
bool DFA::SearchFTF(SearchParams* params) {
  return InlinedSearchLoop(params, 0, 1, 0);
}
bool DFA::SearchFTT(SearchParams* params) {
  return InlinedSearchLoop(params, 0, 1, 1);
}
bool DFA::SearchTFF(SearchParams* params) {
  return InlinedSearchLoop(params, 1, 0, 0);
}
bool DFA::SearchTFT(SearchParams* params) {
  return InlinedSearchLoop(params, 1, 0, 1);
}
bool DFA::SearchTTF(SearchParams* params) {
  return InlinedSearchLoop(params, 1, 1, 0);
}
bool DFA::SearchTTT(SearchParams* params) {
  return InlinedSearchLoop(params, 1, 1, 1);
}

// For debugging, calls the general code directly.
bool DFA::SlowSearchLoop(SearchParams* params) {
  return InlinedSearchLoop(params,
                           params->first_byte >= 0,
                           params->want_earliest_match,
                           params->run_forward);
}

// For performance, calls the appropriate specialized version
// of InlinedSearchLoop.
bool DFA::FastSearchLoop(SearchParams* params) {
  // Because the methods are private, the Searches array
  // cannot be declared at top level.
  static bool (DFA::*Searches[])(SearchParams*) = {
    &DFA::SearchFFF,
    &DFA::SearchFFT,
    &DFA::SearchFTF,
    &DFA::SearchFTT,
    &DFA::SearchTFF,
    &DFA::SearchTFT,
    &DFA::SearchTTF,
    &DFA::SearchTTT,
  };

  bool have_first_byte = params->first_byte >= 0;
  int index = 4 * have_first_byte +
              2 * params->want_earliest_match +
              1 * params->run_forward;
  return (this->*Searches[index])(params);
}


// The discussion of DFA execution above ignored the question of how
// to determine the initial state for the search loop.  There are two
// factors that influence the choice of start state.
//
// The first factor is whether the search is anchored or not.
// The regexp program (Prog*) itself has
// two different entry points: one for anchored searches and one for
// unanchored searches.  (The unanchored version starts with a leading ".*?"
// and then jumps to the anchored one.)
//
// The second factor is where text appears in the larger context, which
// determines which empty-string operators can be matched at the beginning
// of execution.  If text is at the very beginning of context, \A and ^ match.
// Otherwise if text is at the beginning of a line, then ^ matches.
// Otherwise it matters whether the character before text is a word character
// or a non-word character.
//
// The two cases (unanchored vs not) and four cases (empty-string flags)
// combine to make the eight cases recorded in the DFA's begin_text_[2],
// begin_line_[2], after_wordchar_[2], and after_nonwordchar_[2] cached
// StartInfos.  The start state for each is filled in the first time it
// is used for an actual search.

// Examines text, context, and anchored to determine the right start
// state for the DFA search loop.  Fills in params and returns true on success.
// Returns false on failure.
bool DFA::AnalyzeSearch(SearchParams* params) {
  const StringPiece& text = params->text;
  const StringPiece& context = params->context;

  // Sanity check: make sure that text lies within context.
  if (text.begin() < context.begin() || text.end() > context.end()) {
    LOG(DFATAL) << "context does not contain text";
    params->start = DeadState;
    return true;
  }

  // Determine correct search type.
  int start;
  uint32_t flags;
  if (params->run_forward) {
    if (text.begin() == context.begin()) {
      start = kStartBeginText;
      flags = kEmptyBeginText|kEmptyBeginLine;
    } else if (text.begin()[-1] == '\n') {
      start = kStartBeginLine;
      flags = kEmptyBeginLine;
    } else if (Prog::IsWordChar(text.begin()[-1] & 0xFF)) {
      start = kStartAfterWordChar;
      flags = kFlagLastWord;
    } else {
      start = kStartAfterNonWordChar;
      flags = 0;
    }
  } else {
    if (text.end() == context.end()) {
      start = kStartBeginText;
      flags = kEmptyBeginText|kEmptyBeginLine;
    } else if (text.end()[0] == '\n') {
      start = kStartBeginLine;
      flags = kEmptyBeginLine;
    } else if (Prog::IsWordChar(text.end()[0] & 0xFF)) {
      start = kStartAfterWordChar;
      flags = kFlagLastWord;
    } else {
      start = kStartAfterNonWordChar;
      flags = 0;
    }
  }
  if (params->anchored)
    start |= kStartAnchored;
  StartInfo* info = &start_[start];

  // Try once without cache_lock for writing.
  // Try again after resetting the cache
  // (ResetCache will relock cache_lock for writing).
  if (!AnalyzeSearchHelper(params, info, flags)) {
    ResetCache(params->cache_lock);
    if (!AnalyzeSearchHelper(params, info, flags)) {
      LOG(DFATAL) << "Failed to analyze start state.";
      params->failed = true;
      return false;
    }
  }

  if (ExtraDebug)
    fprintf(stderr, "anchored=%d fwd=%d flags=%#x state=%s first_byte=%d\n",
            params->anchored, params->run_forward, flags,
            DumpState(info->start).c_str(), info->first_byte.load());

  params->start = info->start;
  params->first_byte = info->first_byte.load(std::memory_order_acquire);

  return true;
}

// Fills in info if needed.  Returns true on success, false on failure.
bool DFA::AnalyzeSearchHelper(SearchParams* params, StartInfo* info,
                              uint32_t flags) {
  // Quick check.
  int fb = info->first_byte.load(std::memory_order_acquire);
  if (fb != kFbUnknown)
    return true;

  MutexLock l(&mutex_);
  fb = info->first_byte.load(std::memory_order_relaxed);
  if (fb != kFbUnknown)
    return true;

  q0_->clear();
  AddToQueue(q0_,
             params->anchored ? prog_->start() : prog_->start_unanchored(),
             flags);
  info->start = WorkqToCachedState(q0_, NULL, flags);
  if (info->start == NULL)
    return false;

  if (info->start == DeadState) {
    // Synchronize with "quick check" above.
    info->first_byte.store(kFbNone, std::memory_order_release);
    return true;
  }

  if (info->start == FullMatchState) {
    // Synchronize with "quick check" above.
    info->first_byte.store(kFbNone, std::memory_order_release);  // will be ignored
    return true;
  }

  // Even if we have a first_byte, we cannot use it when anchored and,
  // less obviously, we cannot use it when we are going to need flags.
  // This trick works only when there is a single byte that leads to a
  // different state!
  int first_byte = prog_->first_byte();
  if (first_byte == -1 ||
      params->anchored ||
      info->start->flag_ >> kFlagNeedShift != 0)
    first_byte = kFbNone;

  // Synchronize with "quick check" above.
  info->first_byte.store(first_byte, std::memory_order_release);
  return true;
}

// The actual DFA search: calls AnalyzeSearch and then FastSearchLoop.
bool DFA::Search(const StringPiece& text,
                 const StringPiece& context,
                 bool anchored,
                 bool want_earliest_match,
                 bool run_forward,
                 bool* failed,
                 const char** epp,
                 SparseSet* matches) {
  *epp = NULL;
  if (!ok()) {
    *failed = true;
    return false;
  }
  *failed = false;

  if (ExtraDebug) {
    fprintf(stderr, "\nprogram:\n%s\n", prog_->DumpUnanchored().c_str());
    fprintf(stderr, "text %s anchored=%d earliest=%d fwd=%d kind %d\n",
            std::string(text).c_str(), anchored, want_earliest_match,
            run_forward, kind_);
  }

  RWLocker l(&cache_mutex_);
  SearchParams params(text, context, &l);
  params.anchored = anchored;
  params.want_earliest_match = want_earliest_match;
  params.run_forward = run_forward;
  params.matches = matches;

  if (!AnalyzeSearch(&params)) {
    *failed = true;
    return false;
  }
  if (params.start == DeadState)
    return false;
  if (params.start == FullMatchState) {
    if (run_forward == want_earliest_match)
      *epp = text.begin();
    else
      *epp = text.end();
    return true;
  }
  if (ExtraDebug)
    fprintf(stderr, "start %s\n", DumpState(params.start).c_str());
  bool ret = FastSearchLoop(&params);
  if (params.failed) {
    *failed = true;
    return false;
  }
  *epp = params.ep;
  return ret;
}

DFA* Prog::GetDFA(MatchKind kind) {
  // For a forward DFA, half the memory goes to each DFA.
  // However, if it is a "many match" DFA, then there is
  // no counterpart with which the memory must be shared.
  //
  // For a reverse DFA, all the memory goes to the
  // "longest match" DFA, because RE2 never does reverse
  // "first match" searches.
  if (kind == kFirstMatch) {
    std::call_once(dfa_first_once_, [](Prog* prog) {
      prog->dfa_first_ = new DFA(prog, kFirstMatch, prog->dfa_mem_ / 2);
    }, this);
    return dfa_first_;
  } else if (kind == kManyMatch) {
    std::call_once(dfa_first_once_, [](Prog* prog) {
      prog->dfa_first_ = new DFA(prog, kManyMatch, prog->dfa_mem_);
    }, this);
    return dfa_first_;
  } else {
    std::call_once(dfa_longest_once_, [](Prog* prog) {
      if (!prog->reversed_)
        prog->dfa_longest_ = new DFA(prog, kLongestMatch, prog->dfa_mem_ / 2);
      else
        prog->dfa_longest_ = new DFA(prog, kLongestMatch, prog->dfa_mem_);
    }, this);
    return dfa_longest_;
  }
}

void Prog::DeleteDFA(DFA* dfa) {
  delete dfa;
}

// Executes the regexp program to search in text,
// which itself is inside the larger context.  (As a convenience,
// passing a NULL context is equivalent to passing text.)
// Returns true if a match is found, false if not.
// If a match is found, fills in match0->end() to point at the end of the match
// and sets match0->begin() to text.begin(), since the DFA can't track
// where the match actually began.
//
// This is the only external interface (class DFA only exists in this file).
//
bool Prog::SearchDFA(const StringPiece& text, const StringPiece& const_context,
                     Anchor anchor, MatchKind kind, StringPiece* match0,
                     bool* failed, SparseSet* matches) {
  *failed = false;

  StringPiece context = const_context;
  if (context.begin() == NULL)
    context = text;
  bool carat = anchor_start();
  bool dollar = anchor_end();
  if (reversed_) {
    using std::swap;
    swap(carat, dollar);
  }
  if (carat && context.begin() != text.begin())
    return false;
  if (dollar && context.end() != text.end())
    return false;

  // Handle full match by running an anchored longest match
  // and then checking if it covers all of text.
  bool anchored = anchor == kAnchored || anchor_start() || kind == kFullMatch;
  bool endmatch = false;
  if (kind == kManyMatch) {
    // This is split out in order to avoid clobbering kind.
  } else if (kind == kFullMatch || anchor_end()) {
    endmatch = true;
    kind = kLongestMatch;
  }

  // If the caller doesn't care where the match is (just whether one exists),
  // then we can stop at the very first match we find, the so-called
  // "earliest match".
  bool want_earliest_match = false;
  if (kind == kManyMatch) {
    // This is split out in order to avoid clobbering kind.
    if (matches == NULL) {
      want_earliest_match = true;
    }
  } else if (match0 == NULL && !endmatch) {
    want_earliest_match = true;
    kind = kLongestMatch;
  }

  DFA* dfa = GetDFA(kind);
  const char* ep;
  bool matched = dfa->Search(text, context, anchored,
                             want_earliest_match, !reversed_,
                             failed, &ep, matches);
  if (*failed)
    return false;
  if (!matched)
    return false;
  if (endmatch && ep != (reversed_ ? text.begin() : text.end()))
    return false;

  // If caller cares, record the boundary of the match.
  // We only know where it ends, so use the boundary of text
  // as the beginning.
  if (match0) {
    if (reversed_)
      *match0 = StringPiece(ep, static_cast<size_t>(text.end() - ep));
    else
      *match0 =
          StringPiece(text.begin(), static_cast<size_t>(ep - text.begin()));
  }
  return true;
}

// Build out all states in DFA.  Returns number of states.
int DFA::BuildAllStates(const Prog::DFAStateCallback& cb) {
  if (!ok())
    return 0;

  // Pick out start state for unanchored search
  // at beginning of text.
  RWLocker l(&cache_mutex_);
  SearchParams params(StringPiece(), StringPiece(), &l);
  params.anchored = false;
  if (!AnalyzeSearch(&params) ||
      params.start == NULL ||
      params.start == DeadState)
    return 0;

  // Add start state to work queue.
  // Note that any State* that we handle here must point into the cache,
  // so we can simply depend on pointer-as-a-number hashing and equality.
  std::unordered_map<State*, int> m;
  std::deque<State*> q;
  m.emplace(params.start, static_cast<int>(m.size()));
  q.push_back(params.start);

  // Compute the input bytes needed to cover all of the next pointers.
  int nnext = prog_->bytemap_range() + 1;  // + 1 for kByteEndText slot
  std::vector<int> input(nnext);
  for (int c = 0; c < 256; c++) {
    int b = prog_->bytemap()[c];
    while (c < 256-1 && prog_->bytemap()[c+1] == b)
      c++;
    input[b] = c;
  }
  input[prog_->bytemap_range()] = kByteEndText;

  // Scratch space for the output.
  std::vector<int> output(nnext);

  // Flood to expand every state.
  bool oom = false;
  while (!q.empty()) {
    State* s = q.front();
    q.pop_front();
    for (int c : input) {
      State* ns = RunStateOnByteUnlocked(s, c);
      if (ns == NULL) {
        oom = true;
        break;
      }
      if (ns == DeadState) {
        output[ByteMap(c)] = -1;
        continue;
      }
      if (m.find(ns) == m.end()) {
        m.emplace(ns, static_cast<int>(m.size()));
        q.push_back(ns);
      }
      output[ByteMap(c)] = m[ns];
    }
    if (cb)
      cb(oom ? NULL : output.data(),
         s == FullMatchState || s->IsMatch());
    if (oom)
      break;
  }

  return static_cast<int>(m.size());
}

// Build out all states in DFA for kind.  Returns number of states.
int Prog::BuildEntireDFA(MatchKind kind, const DFAStateCallback& cb) {
  return GetDFA(kind)->BuildAllStates(cb);
}

void Prog::TEST_dfa_should_bail_when_slow(bool b) {
  dfa_should_bail_when_slow = b;
}

// Computes min and max for matching string.
// Won't return strings bigger than maxlen.
bool DFA::PossibleMatchRange(std::string* min, std::string* max, int maxlen) {
  if (!ok())
    return false;

  // NOTE: if future users of PossibleMatchRange want more precision when
  // presented with infinitely repeated elements, consider making this a
  // parameter to PossibleMatchRange.
  static int kMaxEltRepetitions = 0;

  // Keep track of the number of times we've visited states previously. We only
  // revisit a given state if it's part of a repeated group, so if the value
  // portion of the map tuple exceeds kMaxEltRepetitions we bail out and set
  // |*max| to |PrefixSuccessor(*max)|.
  //
  // Also note that previously_visited_states[UnseenStatePtr] will, in the STL
  // tradition, implicitly insert a '0' value at first use. We take advantage
  // of that property below.
  std::unordered_map<State*, int> previously_visited_states;

  // Pick out start state for anchored search at beginning of text.
  RWLocker l(&cache_mutex_);
  SearchParams params(StringPiece(), StringPiece(), &l);
  params.anchored = true;
  if (!AnalyzeSearch(&params))
    return false;
  if (params.start == DeadState) {  // No matching strings
    *min = "";
    *max = "";
    return true;
  }
  if (params.start == FullMatchState)  // Every string matches: no max
    return false;

  // The DFA is essentially a big graph rooted at params.start,
  // and paths in the graph correspond to accepted strings.
  // Each node in the graph has potentially 256+1 arrows
  // coming out, one for each byte plus the magic end of
  // text character kByteEndText.

  // To find the smallest possible prefix of an accepted
  // string, we just walk the graph preferring to follow
  // arrows with the lowest bytes possible.  To find the
  // largest possible prefix, we follow the largest bytes
  // possible.

  // The test for whether there is an arrow from s on byte j is
  //    ns = RunStateOnByteUnlocked(s, j);
  //    if (ns == NULL)
  //      return false;
  //    if (ns != DeadState && ns->ninst > 0)
  // The RunStateOnByteUnlocked call asks the DFA to build out the graph.
  // It returns NULL only if the DFA has run out of memory,
  // in which case we can't be sure of anything.
  // The second check sees whether there was graph built
  // and whether it is interesting graph.  Nodes might have
  // ns->ninst == 0 if they exist only to represent the fact
  // that a match was found on the previous byte.

  // Build minimum prefix.
  State* s = params.start;
  min->clear();
  MutexLock lock(&mutex_);
  for (int i = 0; i < maxlen; i++) {
    if (previously_visited_states[s] > kMaxEltRepetitions)
      break;
    previously_visited_states[s]++;

    // Stop if min is a match.
    State* ns = RunStateOnByte(s, kByteEndText);
    if (ns == NULL)  // DFA out of memory
      return false;
    if (ns != DeadState && (ns == FullMatchState || ns->IsMatch()))
      break;

    // Try to extend the string with low bytes.
    bool extended = false;
    for (int j = 0; j < 256; j++) {
      ns = RunStateOnByte(s, j);
      if (ns == NULL)  // DFA out of memory
        return false;
      if (ns == FullMatchState ||
          (ns > SpecialStateMax && ns->ninst_ > 0)) {
        extended = true;
        min->append(1, static_cast<char>(j));
        s = ns;
        break;
      }
    }
    if (!extended)
      break;
  }

  // Build maximum prefix.
  previously_visited_states.clear();
  s = params.start;
  max->clear();
  for (int i = 0; i < maxlen; i++) {
    if (previously_visited_states[s] > kMaxEltRepetitions)
      break;
    previously_visited_states[s] += 1;

    // Try to extend the string with high bytes.
    bool extended = false;
    for (int j = 255; j >= 0; j--) {
      State* ns = RunStateOnByte(s, j);
      if (ns == NULL)
        return false;
      if (ns == FullMatchState ||
          (ns > SpecialStateMax && ns->ninst_ > 0)) {
        extended = true;
        max->append(1, static_cast<char>(j));
        s = ns;
        break;
      }
    }
    if (!extended) {
      // Done, no need for PrefixSuccessor.
      return true;
    }
  }

  // Stopped while still adding to *max - round aaaaaaaaaa... to aaaa...b
  PrefixSuccessor(max);

  // If there are no bytes left, we have no way to say "there is no maximum
  // string".  We could make the interface more complicated and be able to
  // return "there is no maximum but here is a minimum", but that seems like
  // overkill -- the most common no-max case is all possible strings, so not
  // telling the caller that the empty string is the minimum match isn't a
  // great loss.
  if (max->empty())
    return false;

  return true;
}

// PossibleMatchRange for a Prog.
bool Prog::PossibleMatchRange(std::string* min, std::string* max, int maxlen) {
  // Have to use dfa_longest_ to get all strings for full matches.
  // For example, (a|aa) never matches aa in first-match mode.
  return GetDFA(kLongestMatch)->PossibleMatchRange(min, max, maxlen);
}

}  // namespace re2
