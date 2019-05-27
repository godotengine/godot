// Copyright 2008 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tested by search_test.cc, exhaustive_test.cc, tester.cc

// Prog::SearchBitState is a regular expression search with submatch
// tracking for small regular expressions and texts.  Similarly to
// testing/backtrack.cc, it allocates a bitmap with (count of
// lists) * (length of prog) bits to make sure it never explores the
// same (instruction list, character position) multiple times.  This
// limits the search to run in time linear in the length of the text.
//
// Unlike testing/backtrack.cc, SearchBitState is not recursive
// on the text.
//
// SearchBitState is a fast replacement for the NFA code on small
// regexps and texts when SearchOnePass cannot be used.

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <limits>
#include <utility>

#include "util/logging.h"
#include "util/pod_array.h"
#include "re2/prog.h"
#include "re2/regexp.h"

namespace re2 {

struct Job {
  int id;
  int rle;  // run length encoding
  const char* p;
};

class BitState {
 public:
  explicit BitState(Prog* prog);

  // The usual Search prototype.
  // Can only call Search once per BitState.
  bool Search(const StringPiece& text, const StringPiece& context,
              bool anchored, bool longest,
              StringPiece* submatch, int nsubmatch);

 private:
  inline bool ShouldVisit(int id, const char* p);
  void Push(int id, const char* p);
  void GrowStack();
  bool TrySearch(int id, const char* p);

  // Search parameters
  Prog* prog_;              // program being run
  StringPiece text_;        // text being searched
  StringPiece context_;     // greater context of text being searched
  bool anchored_;           // whether search is anchored at text.begin()
  bool longest_;            // whether search wants leftmost-longest match
  bool endmatch_;           // whether match must end at text.end()
  StringPiece* submatch_;   // submatches to fill in
  int nsubmatch_;           //   # of submatches to fill in

  // Search state
  static const int VisitedBits = 32;
  PODArray<uint32_t> visited_;  // bitmap: (list ID, char*) pairs visited
  PODArray<const char*> cap_;   // capture registers
  PODArray<Job> job_;           // stack of text positions to explore
  int njob_;                    // stack size
};

BitState::BitState(Prog* prog)
  : prog_(prog),
    anchored_(false),
    longest_(false),
    endmatch_(false),
    submatch_(NULL),
    nsubmatch_(0),
    njob_(0) {
}

// Given id, which *must* be a list head, we can look up its list ID.
// Then the question is: Should the search visit the (list ID, p) pair?
// If so, remember that it was visited so that the next time,
// we don't repeat the visit.
bool BitState::ShouldVisit(int id, const char* p) {
  int n = prog_->list_heads()[id] * static_cast<int>(text_.size()+1) +
          static_cast<int>(p-text_.begin());
  if (visited_[n/VisitedBits] & (1 << (n & (VisitedBits-1))))
    return false;
  visited_[n/VisitedBits] |= 1 << (n & (VisitedBits-1));
  return true;
}

// Grow the stack.
void BitState::GrowStack() {
  PODArray<Job> tmp(2*job_.size());
  memmove(tmp.data(), job_.data(), njob_*sizeof job_[0]);
  job_ = std::move(tmp);
}

// Push (id, p) onto the stack, growing it if necessary.
void BitState::Push(int id, const char* p) {
  if (njob_ >= job_.size()) {
    GrowStack();
    if (njob_ >= job_.size()) {
      LOG(DFATAL) << "GrowStack() failed: "
                  << "njob_ = " << njob_ << ", "
                  << "job_.size() = " << job_.size();
      return;
    }
  }

  // If id < 0, it's undoing a Capture,
  // so we mustn't interfere with that.
  if (id >= 0 && njob_ > 0) {
    Job* top = &job_[njob_-1];
    if (id == top->id &&
        p == top->p + top->rle + 1 &&
        top->rle < std::numeric_limits<int>::max()) {
      ++top->rle;
      return;
    }
  }

  Job* top = &job_[njob_++];
  top->id = id;
  top->rle = 0;
  top->p = p;
}

// Try a search from instruction id0 in state p0.
// Return whether it succeeded.
bool BitState::TrySearch(int id0, const char* p0) {
  bool matched = false;
  const char* end = text_.end();
  njob_ = 0;
  // Push() no longer checks ShouldVisit(),
  // so we must perform the check ourselves.
  if (ShouldVisit(id0, p0))
    Push(id0, p0);
  while (njob_ > 0) {
    // Pop job off stack.
    --njob_;
    int id = job_[njob_].id;
    int& rle = job_[njob_].rle;
    const char* p = job_[njob_].p;

    if (id < 0) {
      // Undo the Capture.
      cap_[prog_->inst(-id)->cap()] = p;
      continue;
    }

    if (rle > 0) {
      p += rle;
      // Revivify job on stack.
      --rle;
      ++njob_;
    }

  Loop:
    // Visit id, p.
    Prog::Inst* ip = prog_->inst(id);
    switch (ip->opcode()) {
      default:
        LOG(DFATAL) << "Unexpected opcode: " << ip->opcode();
        return false;

      case kInstFail:
        break;

      case kInstAltMatch:
        if (ip->greedy(prog_)) {
          // out1 is the Match instruction.
          id = ip->out1();
          p = end;
          goto Loop;
        }
        if (longest_) {
          // ip must be non-greedy...
          // out is the Match instruction.
          id = ip->out();
          p = end;
          goto Loop;
        }
        goto Next;

      case kInstByteRange: {
        int c = -1;
        if (p < end)
          c = *p & 0xFF;
        if (!ip->Matches(c))
          goto Next;

        if (ip->hint() != 0)
          Push(id+ip->hint(), p);  // try the next when we're done
        id = ip->out();
        p++;
        goto CheckAndLoop;
      }

      case kInstCapture:
        if (!ip->last())
          Push(id+1, p);  // try the next when we're done

        if (0 <= ip->cap() && ip->cap() < cap_.size()) {
          // Capture p to register, but save old value first.
          Push(-id, cap_[ip->cap()]);  // undo when we're done
          cap_[ip->cap()] = p;
        }

        id = ip->out();
        goto CheckAndLoop;

      case kInstEmptyWidth:
        if (ip->empty() & ~Prog::EmptyFlags(context_, p))
          goto Next;

        if (!ip->last())
          Push(id+1, p);  // try the next when we're done
        id = ip->out();
        goto CheckAndLoop;

      case kInstNop:
        if (!ip->last())
          Push(id+1, p);  // try the next when we're done
        id = ip->out();

      CheckAndLoop:
        // Sanity check: id is the head of its list, which must
        // be the case if id-1 is the last of *its* list. :)
        DCHECK(id == 0 || prog_->inst(id-1)->last());
        if (ShouldVisit(id, p))
          goto Loop;
        break;

      case kInstMatch: {
        if (endmatch_ && p != end)
          goto Next;

        // We found a match.  If the caller doesn't care
        // where the match is, no point going further.
        if (nsubmatch_ == 0)
          return true;

        // Record best match so far.
        // Only need to check end point, because this entire
        // call is only considering one start position.
        matched = true;
        cap_[1] = p;
        if (submatch_[0].data() == NULL ||
            (longest_ && p > submatch_[0].end())) {
          for (int i = 0; i < nsubmatch_; i++)
            submatch_[i] =
                StringPiece(cap_[2 * i],
                            static_cast<size_t>(cap_[2 * i + 1] - cap_[2 * i]));
        }

        // If going for first match, we're done.
        if (!longest_)
          return true;

        // If we used the entire text, no longer match is possible.
        if (p == end)
          return true;

        // Otherwise, continue on in hope of a longer match.
        // Note the absence of the ShouldVisit() check here
        // due to execution remaining in the same list.
      Next:
        if (!ip->last()) {
          id++;
          goto Loop;
        }
        break;
      }
    }
  }
  return matched;
}

// Search text (within context) for prog_.
bool BitState::Search(const StringPiece& text, const StringPiece& context,
                      bool anchored, bool longest,
                      StringPiece* submatch, int nsubmatch) {
  // Search parameters.
  text_ = text;
  context_ = context;
  if (context_.begin() == NULL)
    context_ = text;
  if (prog_->anchor_start() && context_.begin() != text.begin())
    return false;
  if (prog_->anchor_end() && context_.end() != text.end())
    return false;
  anchored_ = anchored || prog_->anchor_start();
  longest_ = longest || prog_->anchor_end();
  endmatch_ = prog_->anchor_end();
  submatch_ = submatch;
  nsubmatch_ = nsubmatch;
  for (int i = 0; i < nsubmatch_; i++)
    submatch_[i] = StringPiece();

  // Allocate scratch space.
  int nvisited = prog_->list_count() * static_cast<int>(text.size()+1);
  nvisited = (nvisited + VisitedBits-1) / VisitedBits;
  visited_ = PODArray<uint32_t>(nvisited);
  memset(visited_.data(), 0, nvisited*sizeof visited_[0]);

  int ncap = 2*nsubmatch;
  if (ncap < 2)
    ncap = 2;
  cap_ = PODArray<const char*>(ncap);
  memset(cap_.data(), 0, ncap*sizeof cap_[0]);

  // When sizeof(Job) == 16, we start with a nice round 1KiB. :)
  job_ = PODArray<Job>(64);

  // Anchored search must start at text.begin().
  if (anchored_) {
    cap_[0] = text.begin();
    return TrySearch(prog_->start(), text.begin());
  }

  // Unanchored search, starting from each possible text position.
  // Notice that we have to try the empty string at the end of
  // the text, so the loop condition is p <= text.end(), not p < text.end().
  // This looks like it's quadratic in the size of the text,
  // but we are not clearing visited_ between calls to TrySearch,
  // so no work is duplicated and it ends up still being linear.
  for (const char* p = text.begin(); p <= text.end(); p++) {
    // Try to use memchr to find the first byte quickly.
    int fb = prog_->first_byte();
    if (fb >= 0 && p < text.end() && (p[0] & 0xFF) != fb) {
      p = reinterpret_cast<const char*>(memchr(p, fb, text.end() - p));
      if (p == NULL)
        p = text.end();
    }

    cap_[0] = p;
    if (TrySearch(prog_->start(), p))  // Match must be leftmost; done.
      return true;
  }
  return false;
}

// Bit-state search.
bool Prog::SearchBitState(const StringPiece& text,
                          const StringPiece& context,
                          Anchor anchor,
                          MatchKind kind,
                          StringPiece* match,
                          int nmatch) {
  // If full match, we ask for an anchored longest match
  // and then check that match[0] == text.
  // So make sure match[0] exists.
  StringPiece sp0;
  if (kind == kFullMatch) {
    anchor = kAnchored;
    if (nmatch < 1) {
      match = &sp0;
      nmatch = 1;
    }
  }

  // Run the search.
  BitState b(this);
  bool anchored = anchor == kAnchored;
  bool longest = kind != kFirstMatch;
  if (!b.Search(text, context, anchored, longest, match, nmatch))
    return false;
  if (kind == kFullMatch && match[0].end() != text.end())
    return false;
  return true;
}

}  // namespace re2
