// Copyright 2006-2007 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tested by search_test.cc.
//
// Prog::SearchNFA, an NFA search.
// This is an actual NFA like the theorists talk about,
// not the pseudo-NFA found in backtracking regexp implementations.
//
// IMPLEMENTATION
//
// This algorithm is a variant of one that appeared in Rob Pike's sam editor,
// which is a variant of the one described in Thompson's 1968 CACM paper.
// See http://swtch.com/~rsc/regexp/ for various history.  The main feature
// over the DFA implementation is that it tracks submatch boundaries.
//
// When the choice of submatch boundaries is ambiguous, this particular
// implementation makes the same choices that traditional backtracking
// implementations (in particular, Perl and PCRE) do.
// Note that unlike in Perl and PCRE, this algorithm *cannot* take exponential
// time in the length of the input.
//
// Like Thompson's original machine and like the DFA implementation, this
// implementation notices a match only once it is one byte past it.

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "re2/prog.h"
#include "re2/regexp.h"
#include "util/logging.h"
#include "util/pod_array.h"
#include "util/sparse_array.h"
#include "util/sparse_set.h"
#include "util/strutil.h"

namespace re2 {

static const bool ExtraDebug = false;

class NFA {
 public:
  NFA(Prog* prog);
  ~NFA();

  // Searches for a matching string.
  //   * If anchored is true, only considers matches starting at offset.
  //     Otherwise finds lefmost match at or after offset.
  //   * If longest is true, returns the longest match starting
  //     at the chosen start point.  Otherwise returns the so-called
  //     left-biased match, the one traditional backtracking engines
  //     (like Perl and PCRE) find.
  // Records submatch boundaries in submatch[1..nsubmatch-1].
  // Submatch[0] is the entire match.  When there is a choice in
  // which text matches each subexpression, the submatch boundaries
  // are chosen to match what a backtracking implementation would choose.
  bool Search(const StringPiece& text, const StringPiece& context,
              bool anchored, bool longest,
              StringPiece* submatch, int nsubmatch);

 private:
  struct Thread {
    union {
      int ref;
      Thread* next;  // when on free list
    };
    const char** capture;
  };

  // State for explicit stack in AddToThreadq.
  struct AddState {
    int id;     // Inst to process
    Thread* t;  // if not null, set t0 = t before processing id
  };

  // Threadq is a list of threads.  The list is sorted by the order
  // in which Perl would explore that particular state -- the earlier
  // choices appear earlier in the list.
  typedef SparseArray<Thread*> Threadq;

  inline Thread* AllocThread();
  inline Thread* Incref(Thread* t);
  inline void Decref(Thread* t);

  // Follows all empty arrows from id0 and enqueues all the states reached.
  // Enqueues only the ByteRange instructions that match byte c.
  // context is used (with p) for evaluating empty-width specials.
  // p is the current input position, and t0 is the current thread.
  void AddToThreadq(Threadq* q, int id0, int c, const StringPiece& context,
                    const char* p, Thread* t0);

  // Run runq on byte c, appending new states to nextq.
  // Updates matched_ and match_ as new, better matches are found.
  // context is used (with p) for evaluating empty-width specials.
  // p is the position of byte c in the input string for AddToThreadq;
  // p-1 will be used when processing Match instructions.
  // Frees all the threads on runq.
  // If there is a shortcut to the end, returns that shortcut.
  int Step(Threadq* runq, Threadq* nextq, int c, const StringPiece& context,
           const char* p);

  // Returns text version of capture information, for debugging.
  std::string FormatCapture(const char** capture);

  inline void CopyCapture(const char** dst, const char** src);

  Prog* prog_;                // underlying program
  int start_;                 // start instruction in program
  int ncapture_;              // number of submatches to track
  bool longest_;              // whether searching for longest match
  bool endmatch_;             // whether match must end at text.end()
  const char* btext_;         // beginning of text being matched (for FormatSubmatch)
  const char* etext_;         // end of text being matched (for endmatch_)
  Threadq q0_, q1_;           // pre-allocated for Search.
  PODArray<AddState> stack_;  // pre-allocated for AddToThreadq
  Thread* free_threads_;      // free list
  const char** match_;        // best match so far
  bool matched_;              // any match so far?

  NFA(const NFA&) = delete;
  NFA& operator=(const NFA&) = delete;
};

NFA::NFA(Prog* prog) {
  prog_ = prog;
  start_ = prog_->start();
  ncapture_ = 0;
  longest_ = false;
  endmatch_ = false;
  btext_ = NULL;
  etext_ = NULL;
  q0_.resize(prog_->size());
  q1_.resize(prog_->size());
  // See NFA::AddToThreadq() for why this is so.
  int nstack = 2*prog_->inst_count(kInstCapture) +
               prog_->inst_count(kInstEmptyWidth) +
               prog_->inst_count(kInstNop) + 1;  // + 1 for start inst
  stack_ = PODArray<AddState>(nstack);
  free_threads_ = NULL;
  match_ = NULL;
  matched_ = false;
}

NFA::~NFA() {
  delete[] match_;
  Thread* next;
  for (Thread* t = free_threads_; t; t = next) {
    next = t->next;
    delete[] t->capture;
    delete t;
  }
}

NFA::Thread* NFA::AllocThread() {
  Thread* t = free_threads_;
  if (t == NULL) {
    t = new Thread;
    t->ref = 1;
    t->capture = new const char*[ncapture_];
    return t;
  }
  free_threads_ = t->next;
  t->ref = 1;
  return t;
}

NFA::Thread* NFA::Incref(Thread* t) {
  DCHECK(t != NULL);
  t->ref++;
  return t;
}

void NFA::Decref(Thread* t) {
  if (t == NULL)
    return;
  t->ref--;
  if (t->ref > 0)
    return;
  DCHECK_EQ(t->ref, 0);
  t->next = free_threads_;
  free_threads_ = t;
}

void NFA::CopyCapture(const char** dst, const char** src) {
  for (int i = 0; i < ncapture_; i+=2) {
    dst[i] = src[i];
    dst[i+1] = src[i+1];
  }
}

// Follows all empty arrows from id0 and enqueues all the states reached.
// Enqueues only the ByteRange instructions that match byte c.
// context is used (with p) for evaluating empty-width specials.
// p is the current input position, and t0 is the current thread.
void NFA::AddToThreadq(Threadq* q, int id0, int c, const StringPiece& context,
                       const char* p, Thread* t0) {
  if (id0 == 0)
    return;

  // Use stack_ to hold our stack of instructions yet to process.
  // It was preallocated as follows:
  //   two entries per Capture;
  //   one entry per EmptyWidth; and
  //   one entry per Nop.
  // This reflects the maximum number of stack pushes that each can
  // perform. (Each instruction can be processed at most once.)
  AddState* stk = stack_.data();
  int nstk = 0;

  stk[nstk++] = {id0, NULL};
  while (nstk > 0) {
    DCHECK_LE(nstk, stack_.size());
    AddState a = stk[--nstk];

  Loop:
    if (a.t != NULL) {
      // t0 was a thread that we allocated and copied in order to
      // record the capture, so we must now decref it.
      Decref(t0);
      t0 = a.t;
    }

    int id = a.id;
    if (id == 0)
      continue;
    if (q->has_index(id)) {
      if (ExtraDebug)
        fprintf(stderr, "  [%d%s]\n", id, FormatCapture(t0->capture).c_str());
      continue;
    }

    // Create entry in q no matter what.  We might fill it in below,
    // or we might not.  Even if not, it is necessary to have it,
    // so that we don't revisit id0 during the recursion.
    q->set_new(id, NULL);
    Thread** tp = &q->get_existing(id);
    int j;
    Thread* t;
    Prog::Inst* ip = prog_->inst(id);
    switch (ip->opcode()) {
    default:
      LOG(DFATAL) << "unhandled " << ip->opcode() << " in AddToThreadq";
      break;

    case kInstFail:
      break;

    case kInstAltMatch:
      // Save state; will pick up at next byte.
      t = Incref(t0);
      *tp = t;

      DCHECK(!ip->last());
      a = {id+1, NULL};
      goto Loop;

    case kInstNop:
      if (!ip->last())
        stk[nstk++] = {id+1, NULL};

      // Continue on.
      a = {ip->out(), NULL};
      goto Loop;

    case kInstCapture:
      if (!ip->last())
        stk[nstk++] = {id+1, NULL};

      if ((j=ip->cap()) < ncapture_) {
        // Push a dummy whose only job is to restore t0
        // once we finish exploring this possibility.
        stk[nstk++] = {0, t0};

        // Record capture.
        t = AllocThread();
        CopyCapture(t->capture, t0->capture);
        t->capture[j] = p;
        t0 = t;
      }
      a = {ip->out(), NULL};
      goto Loop;

    case kInstByteRange:
      if (!ip->Matches(c))
        goto Next;

      // Save state; will pick up at next byte.
      t = Incref(t0);
      *tp = t;
      if (ExtraDebug)
        fprintf(stderr, " + %d%s\n", id, FormatCapture(t0->capture).c_str());

      if (ip->hint() == 0)
        break;
      a = {id+ip->hint(), NULL};
      goto Loop;

    case kInstMatch:
      // Save state; will pick up at next byte.
      t = Incref(t0);
      *tp = t;
      if (ExtraDebug)
        fprintf(stderr, " ! %d%s\n", id, FormatCapture(t0->capture).c_str());

    Next:
      if (ip->last())
        break;
      a = {id+1, NULL};
      goto Loop;

    case kInstEmptyWidth:
      if (!ip->last())
        stk[nstk++] = {id+1, NULL};

      // Continue on if we have all the right flag bits.
      if (ip->empty() & ~Prog::EmptyFlags(context, p))
        break;
      a = {ip->out(), NULL};
      goto Loop;
    }
  }
}

// Run runq on byte c, appending new states to nextq.
// Updates matched_ and match_ as new, better matches are found.
// context is used (with p) for evaluating empty-width specials.
// p is the position of byte c in the input string for AddToThreadq;
// p-1 will be used when processing Match instructions.
// Frees all the threads on runq.
// If there is a shortcut to the end, returns that shortcut.
int NFA::Step(Threadq* runq, Threadq* nextq, int c, const StringPiece& context,
              const char* p) {
  nextq->clear();

  for (Threadq::iterator i = runq->begin(); i != runq->end(); ++i) {
    Thread* t = i->value();
    if (t == NULL)
      continue;

    if (longest_) {
      // Can skip any threads started after our current best match.
      if (matched_ && match_[0] < t->capture[0]) {
        Decref(t);
        continue;
      }
    }

    int id = i->index();
    Prog::Inst* ip = prog_->inst(id);

    switch (ip->opcode()) {
      default:
        // Should only see the values handled below.
        LOG(DFATAL) << "Unhandled " << ip->opcode() << " in step";
        break;

      case kInstByteRange:
        AddToThreadq(nextq, ip->out(), c, context, p, t);
        break;

      case kInstAltMatch:
        if (i != runq->begin())
          break;
        // The match is ours if we want it.
        if (ip->greedy(prog_) || longest_) {
          CopyCapture(match_, t->capture);
          matched_ = true;

          Decref(t);
          for (++i; i != runq->end(); ++i)
            Decref(i->value());
          runq->clear();
          if (ip->greedy(prog_))
            return ip->out1();
          return ip->out();
        }
        break;

      case kInstMatch: {
        // Avoid invoking undefined behavior when p happens
        // to be null - and p-1 would be meaningless anyway.
        if (p == NULL)
          break;

        if (endmatch_ && p-1 != etext_)
          break;

        if (longest_) {
          // Leftmost-longest mode: save this match only if
          // it is either farther to the left or at the same
          // point but longer than an existing match.
          if (!matched_ || t->capture[0] < match_[0] ||
              (t->capture[0] == match_[0] && p-1 > match_[1])) {
            CopyCapture(match_, t->capture);
            match_[1] = p-1;
            matched_ = true;
          }
        } else {
          // Leftmost-biased mode: this match is by definition
          // better than what we've already found (see next line).
          CopyCapture(match_, t->capture);
          match_[1] = p-1;
          matched_ = true;

          // Cut off the threads that can only find matches
          // worse than the one we just found: don't run the
          // rest of the current Threadq.
          Decref(t);
          for (++i; i != runq->end(); ++i)
            Decref(i->value());
          runq->clear();
          return 0;
        }
        break;
      }
    }
    Decref(t);
  }
  runq->clear();
  return 0;
}

std::string NFA::FormatCapture(const char** capture) {
  std::string s;
  for (int i = 0; i < ncapture_; i+=2) {
    if (capture[i] == NULL)
      StringAppendF(&s, "(?,?)");
    else if (capture[i+1] == NULL)
      StringAppendF(&s, "(%d,?)", (int)(capture[i] - btext_));
    else
      StringAppendF(&s, "(%d,%d)",
                    (int)(capture[i] - btext_),
                    (int)(capture[i+1] - btext_));
  }
  return s;
}

bool NFA::Search(const StringPiece& text, const StringPiece& const_context,
            bool anchored, bool longest,
            StringPiece* submatch, int nsubmatch) {
  if (start_ == 0)
    return false;

  StringPiece context = const_context;
  if (context.begin() == NULL)
    context = text;

  // Sanity check: make sure that text lies within context.
  if (text.begin() < context.begin() || text.end() > context.end()) {
    LOG(DFATAL) << "context does not contain text";
    return false;
  }

  if (prog_->anchor_start() && context.begin() != text.begin())
    return false;
  if (prog_->anchor_end() && context.end() != text.end())
    return false;
  anchored |= prog_->anchor_start();
  if (prog_->anchor_end()) {
    longest = true;
    endmatch_ = true;
    etext_ = text.end();
  }

  if (nsubmatch < 0) {
    LOG(DFATAL) << "Bad args: nsubmatch=" << nsubmatch;
    return false;
  }

  // Save search parameters.
  ncapture_ = 2*nsubmatch;
  longest_ = longest;

  if (nsubmatch == 0) {
    // We need to maintain match[0], both to distinguish the
    // longest match (if longest is true) and also to tell
    // whether we've seen any matches at all.
    ncapture_ = 2;
  }

  match_ = new const char*[ncapture_];
  matched_ = false;

  // For debugging prints.
  btext_ = context.begin();

  if (ExtraDebug)
    fprintf(stderr, "NFA::Search %s (context: %s) anchored=%d longest=%d\n",
            std::string(text).c_str(), std::string(context).c_str(), anchored,
            longest);

  // Set up search.
  Threadq* runq = &q0_;
  Threadq* nextq = &q1_;
  runq->clear();
  nextq->clear();
  memset(&match_[0], 0, ncapture_*sizeof match_[0]);

  // Loop over the text, stepping the machine.
  for (const char* p = text.begin();; p++) {
    if (ExtraDebug) {
      int c = 0;
      if (p == context.begin())
        c = '^';
      else if (p > text.end())
        c = '$';
      else if (p < text.end())
        c = p[0] & 0xFF;

      fprintf(stderr, "%c:", c);
      for (Threadq::iterator i = runq->begin(); i != runq->end(); ++i) {
        Thread* t = i->value();
        if (t == NULL)
          continue;
        fprintf(stderr, " %d%s", i->index(), FormatCapture(t->capture).c_str());
      }
      fprintf(stderr, "\n");
    }

    // This is a no-op the first time around the loop because runq is empty.
    int id = Step(runq, nextq, p < text.end() ? p[0] & 0xFF : -1, context, p);
    DCHECK_EQ(runq->size(), 0);
    using std::swap;
    swap(nextq, runq);
    nextq->clear();
    if (id != 0) {
      // We're done: full match ahead.
      p = text.end();
      for (;;) {
        Prog::Inst* ip = prog_->inst(id);
        switch (ip->opcode()) {
          default:
            LOG(DFATAL) << "Unexpected opcode in short circuit: " << ip->opcode();
            break;

          case kInstCapture:
            if (ip->cap() < ncapture_)
              match_[ip->cap()] = p;
            id = ip->out();
            continue;

          case kInstNop:
            id = ip->out();
            continue;

          case kInstMatch:
            match_[1] = p;
            matched_ = true;
            break;
        }
        break;
      }
      break;
    }

    if (p > text.end())
      break;

    // Start a new thread if there have not been any matches.
    // (No point in starting a new thread if there have been
    // matches, since it would be to the right of the match
    // we already found.)
    if (!matched_ && (!anchored || p == text.begin())) {
      // If there's a required first byte for an unanchored search
      // and we're not in the middle of any possible matches,
      // use memchr to search for the byte quickly.
      int fb = prog_->first_byte();
      if (!anchored && runq->size() == 0 &&
          fb >= 0 && p < text.end() && (p[0] & 0xFF) != fb) {
        p = reinterpret_cast<const char*>(memchr(p, fb, text.end() - p));
        if (p == NULL) {
          p = text.end();
        }
      }

      Thread* t = AllocThread();
      CopyCapture(t->capture, match_);
      t->capture[0] = p;
      AddToThreadq(runq, start_, p < text.end() ? p[0] & 0xFF : -1, context, p,
                   t);
      Decref(t);
    }

    // If all the threads have died, stop early.
    if (runq->size() == 0) {
      if (ExtraDebug)
        fprintf(stderr, "dead\n");
      break;
    }
  }

  for (Threadq::iterator i = runq->begin(); i != runq->end(); ++i)
    Decref(i->value());

  if (matched_) {
    for (int i = 0; i < nsubmatch; i++)
      submatch[i] =
          StringPiece(match_[2 * i],
                      static_cast<size_t>(match_[2 * i + 1] - match_[2 * i]));
    if (ExtraDebug)
      fprintf(stderr, "match (%td,%td)\n",
              match_[0] - btext_, match_[1] - btext_);
    return true;
  }
  return false;
}

// Computes whether all successful matches have a common first byte,
// and if so, returns that byte.  If not, returns -1.
int Prog::ComputeFirstByte() {
  int b = -1;
  SparseSet q(size());
  q.insert(start());
  for (SparseSet::iterator it = q.begin(); it != q.end(); ++it) {
    int id = *it;
    Prog::Inst* ip = inst(id);
    switch (ip->opcode()) {
      default:
        LOG(DFATAL) << "unhandled " << ip->opcode() << " in ComputeFirstByte";
        break;

      case kInstMatch:
        // The empty string matches: no first byte.
        return -1;

      case kInstByteRange:
        if (!ip->last())
          q.insert(id+1);

        // Must match only a single byte
        if (ip->lo() != ip->hi())
          return -1;
        if (ip->foldcase() && 'a' <= ip->lo() && ip->lo() <= 'z')
          return -1;
        // If we haven't seen any bytes yet, record it;
        // otherwise must match the one we saw before.
        if (b == -1)
          b = ip->lo();
        else if (b != ip->lo())
          return -1;
        break;

      case kInstNop:
      case kInstCapture:
      case kInstEmptyWidth:
        if (!ip->last())
          q.insert(id+1);

        // Continue on.
        // Ignore ip->empty() flags for kInstEmptyWidth
        // in order to be as conservative as possible
        // (assume all possible empty-width flags are true).
        if (ip->out())
          q.insert(ip->out());
        break;

      case kInstAltMatch:
        DCHECK(!ip->last());
        q.insert(id+1);
        break;

      case kInstFail:
        break;
    }
  }
  return b;
}

bool
Prog::SearchNFA(const StringPiece& text, const StringPiece& context,
                Anchor anchor, MatchKind kind,
                StringPiece* match, int nmatch) {
  if (ExtraDebug)
    Dump();

  NFA nfa(this);
  StringPiece sp;
  if (kind == kFullMatch) {
    anchor = kAnchored;
    if (nmatch == 0) {
      match = &sp;
      nmatch = 1;
    }
  }
  if (!nfa.Search(text, context, anchor == kAnchored, kind != kFirstMatch, match, nmatch))
    return false;
  if (kind == kFullMatch && match[0].end() != text.end())
    return false;
  return true;
}

// For each instruction i in the program reachable from the start, compute the
// number of instructions reachable from i by following only empty transitions
// and record that count as fanout[i].
//
// fanout holds the results and is also the work queue for the outer iteration.
// reachable holds the reached nodes for the inner iteration.
void Prog::Fanout(SparseArray<int>* fanout) {
  DCHECK_EQ(fanout->max_size(), size());
  SparseSet reachable(size());
  fanout->clear();
  fanout->set_new(start(), 0);
  for (SparseArray<int>::iterator i = fanout->begin(); i != fanout->end(); ++i) {
    int* count = &i->value();
    reachable.clear();
    reachable.insert(i->index());
    for (SparseSet::iterator j = reachable.begin(); j != reachable.end(); ++j) {
      int id = *j;
      Prog::Inst* ip = inst(id);
      switch (ip->opcode()) {
        default:
          LOG(DFATAL) << "unhandled " << ip->opcode() << " in Prog::Fanout()";
          break;

        case kInstByteRange:
          if (!ip->last())
            reachable.insert(id+1);

          (*count)++;
          if (!fanout->has_index(ip->out())) {
            fanout->set_new(ip->out(), 0);
          }
          break;

        case kInstAltMatch:
          DCHECK(!ip->last());
          reachable.insert(id+1);
          break;

        case kInstCapture:
        case kInstEmptyWidth:
        case kInstNop:
          if (!ip->last())
            reachable.insert(id+1);

          reachable.insert(ip->out());
          break;

        case kInstMatch:
          if (!ip->last())
            reachable.insert(id+1);
          break;

        case kInstFail:
          break;
      }
    }
  }
}

}  // namespace re2
