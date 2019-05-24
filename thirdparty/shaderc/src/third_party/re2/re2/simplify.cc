// Copyright 2006 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Rewrite POSIX and other features in re
// to use simple extended regular expression features.
// Also sort and simplify character classes.

#include <string>

#include "util/util.h"
#include "util/logging.h"
#include "util/pod_array.h"
#include "util/utf.h"
#include "re2/regexp.h"
#include "re2/walker-inl.h"

namespace re2 {

// Parses the regexp src and then simplifies it and sets *dst to the
// string representation of the simplified form.  Returns true on success.
// Returns false and sets *error (if error != NULL) on error.
bool Regexp::SimplifyRegexp(const StringPiece& src, ParseFlags flags,
                            std::string* dst, RegexpStatus* status) {
  Regexp* re = Parse(src, flags, status);
  if (re == NULL)
    return false;
  Regexp* sre = re->Simplify();
  re->Decref();
  if (sre == NULL) {
    // Should not happen, since Simplify never fails.
    LOG(ERROR) << "Simplify failed on " << src;
    if (status) {
      status->set_code(kRegexpInternalError);
      status->set_error_arg(src);
    }
    return false;
  }
  *dst = sre->ToString();
  sre->Decref();
  return true;
}

// Assuming the simple_ flags on the children are accurate,
// is this Regexp* simple?
bool Regexp::ComputeSimple() {
  Regexp** subs;
  switch (op_) {
    case kRegexpNoMatch:
    case kRegexpEmptyMatch:
    case kRegexpLiteral:
    case kRegexpLiteralString:
    case kRegexpBeginLine:
    case kRegexpEndLine:
    case kRegexpBeginText:
    case kRegexpWordBoundary:
    case kRegexpNoWordBoundary:
    case kRegexpEndText:
    case kRegexpAnyChar:
    case kRegexpAnyByte:
    case kRegexpHaveMatch:
      return true;
    case kRegexpConcat:
    case kRegexpAlternate:
      // These are simple as long as the subpieces are simple.
      subs = sub();
      for (int i = 0; i < nsub_; i++)
        if (!subs[i]->simple())
          return false;
      return true;
    case kRegexpCharClass:
      // Simple as long as the char class is not empty, not full.
      if (ccb_ != NULL)
        return !ccb_->empty() && !ccb_->full();
      return !cc_->empty() && !cc_->full();
    case kRegexpCapture:
      subs = sub();
      return subs[0]->simple();
    case kRegexpStar:
    case kRegexpPlus:
    case kRegexpQuest:
      subs = sub();
      if (!subs[0]->simple())
        return false;
      switch (subs[0]->op_) {
        case kRegexpStar:
        case kRegexpPlus:
        case kRegexpQuest:
        case kRegexpEmptyMatch:
        case kRegexpNoMatch:
          return false;
        default:
          break;
      }
      return true;
    case kRegexpRepeat:
      return false;
  }
  LOG(DFATAL) << "Case not handled in ComputeSimple: " << op_;
  return false;
}

// Walker subclass used by Simplify.
// Coalesces runs of star/plus/quest/repeat of the same literal along with any
// occurrences of that literal into repeats of that literal. It also works for
// char classes, any char and any byte.
// PostVisit creates the coalesced result, which should then be simplified.
class CoalesceWalker : public Regexp::Walker<Regexp*> {
 public:
  CoalesceWalker() {}
  virtual Regexp* PostVisit(Regexp* re, Regexp* parent_arg, Regexp* pre_arg,
                            Regexp** child_args, int nchild_args);
  virtual Regexp* Copy(Regexp* re);
  virtual Regexp* ShortVisit(Regexp* re, Regexp* parent_arg);

 private:
  // These functions are declared inside CoalesceWalker so that
  // they can edit the private fields of the Regexps they construct.

  // Returns true if r1 and r2 can be coalesced. In particular, ensures that
  // the parse flags are consistent. (They will not be checked again later.)
  static bool CanCoalesce(Regexp* r1, Regexp* r2);

  // Coalesces *r1ptr and *r2ptr. In most cases, the array elements afterwards
  // will be empty match and the coalesced op. In other cases, where part of a
  // literal string was removed to be coalesced, the array elements afterwards
  // will be the coalesced op and the remainder of the literal string.
  static void DoCoalesce(Regexp** r1ptr, Regexp** r2ptr);

  CoalesceWalker(const CoalesceWalker&) = delete;
  CoalesceWalker& operator=(const CoalesceWalker&) = delete;
};

// Walker subclass used by Simplify.
// The simplify walk is purely post-recursive: given the simplified children,
// PostVisit creates the simplified result.
// The child_args are simplified Regexp*s.
class SimplifyWalker : public Regexp::Walker<Regexp*> {
 public:
  SimplifyWalker() {}
  virtual Regexp* PreVisit(Regexp* re, Regexp* parent_arg, bool* stop);
  virtual Regexp* PostVisit(Regexp* re, Regexp* parent_arg, Regexp* pre_arg,
                            Regexp** child_args, int nchild_args);
  virtual Regexp* Copy(Regexp* re);
  virtual Regexp* ShortVisit(Regexp* re, Regexp* parent_arg);

 private:
  // These functions are declared inside SimplifyWalker so that
  // they can edit the private fields of the Regexps they construct.

  // Creates a concatenation of two Regexp, consuming refs to re1 and re2.
  // Caller must Decref return value when done with it.
  static Regexp* Concat2(Regexp* re1, Regexp* re2, Regexp::ParseFlags flags);

  // Simplifies the expression re{min,max} in terms of *, +, and ?.
  // Returns a new regexp.  Does not edit re.  Does not consume reference to re.
  // Caller must Decref return value when done with it.
  static Regexp* SimplifyRepeat(Regexp* re, int min, int max,
                                Regexp::ParseFlags parse_flags);

  // Simplifies a character class by expanding any named classes
  // into rune ranges.  Does not edit re.  Does not consume ref to re.
  // Caller must Decref return value when done with it.
  static Regexp* SimplifyCharClass(Regexp* re);

  SimplifyWalker(const SimplifyWalker&) = delete;
  SimplifyWalker& operator=(const SimplifyWalker&) = delete;
};

// Simplifies a regular expression, returning a new regexp.
// The new regexp uses traditional Unix egrep features only,
// plus the Perl (?:) non-capturing parentheses.
// Otherwise, no POSIX or Perl additions.  The new regexp
// captures exactly the same subexpressions (with the same indices)
// as the original.
// Does not edit current object.
// Caller must Decref() return value when done with it.

Regexp* Regexp::Simplify() {
  CoalesceWalker cw;
  Regexp* cre = cw.Walk(this, NULL);
  if (cre == NULL)
    return cre;
  SimplifyWalker sw;
  Regexp* sre = sw.Walk(cre, NULL);
  cre->Decref();
  return sre;
}

#define Simplify DontCallSimplify  // Avoid accidental recursion

// Utility function for PostVisit implementations that compares re->sub() with
// child_args to determine whether any child_args changed. In the common case,
// where nothing changed, calls Decref() for all child_args and returns false,
// so PostVisit must return re->Incref(). Otherwise, returns true.
static bool ChildArgsChanged(Regexp* re, Regexp** child_args) {
  for (int i = 0; i < re->nsub(); i++) {
    Regexp* sub = re->sub()[i];
    Regexp* newsub = child_args[i];
    if (newsub != sub)
      return true;
  }
  for (int i = 0; i < re->nsub(); i++) {
    Regexp* newsub = child_args[i];
    newsub->Decref();
  }
  return false;
}

Regexp* CoalesceWalker::Copy(Regexp* re) {
  return re->Incref();
}

Regexp* CoalesceWalker::ShortVisit(Regexp* re, Regexp* parent_arg) {
  // This should never be called, since we use Walk and not
  // WalkExponential.
  LOG(DFATAL) << "CoalesceWalker::ShortVisit called";
  return re->Incref();
}

Regexp* CoalesceWalker::PostVisit(Regexp* re,
                                  Regexp* parent_arg,
                                  Regexp* pre_arg,
                                  Regexp** child_args,
                                  int nchild_args) {
  if (re->nsub() == 0)
    return re->Incref();

  if (re->op() != kRegexpConcat) {
    if (!ChildArgsChanged(re, child_args))
      return re->Incref();

    // Something changed. Build a new op.
    Regexp* nre = new Regexp(re->op(), re->parse_flags());
    nre->AllocSub(re->nsub());
    Regexp** nre_subs = nre->sub();
    for (int i = 0; i < re->nsub(); i++)
      nre_subs[i] = child_args[i];
    // Repeats and Captures have additional data that must be copied.
    if (re->op() == kRegexpRepeat) {
      nre->min_ = re->min();
      nre->max_ = re->max();
    } else if (re->op() == kRegexpCapture) {
      nre->cap_ = re->cap();
    }
    return nre;
  }

  bool can_coalesce = false;
  for (int i = 0; i < re->nsub(); i++) {
    if (i+1 < re->nsub() &&
        CanCoalesce(child_args[i], child_args[i+1])) {
      can_coalesce = true;
      break;
    }
  }
  if (!can_coalesce) {
    if (!ChildArgsChanged(re, child_args))
      return re->Incref();

    // Something changed. Build a new op.
    Regexp* nre = new Regexp(re->op(), re->parse_flags());
    nre->AllocSub(re->nsub());
    Regexp** nre_subs = nre->sub();
    for (int i = 0; i < re->nsub(); i++)
      nre_subs[i] = child_args[i];
    return nre;
  }

  for (int i = 0; i < re->nsub(); i++) {
    if (i+1 < re->nsub() &&
        CanCoalesce(child_args[i], child_args[i+1]))
      DoCoalesce(&child_args[i], &child_args[i+1]);
  }
  // Determine how many empty matches were left by DoCoalesce.
  int n = 0;
  for (int i = n; i < re->nsub(); i++) {
    if (child_args[i]->op() == kRegexpEmptyMatch)
      n++;
  }
  // Build a new op.
  Regexp* nre = new Regexp(re->op(), re->parse_flags());
  nre->AllocSub(re->nsub() - n);
  Regexp** nre_subs = nre->sub();
  for (int i = 0, j = 0; i < re->nsub(); i++) {
    if (child_args[i]->op() == kRegexpEmptyMatch) {
      child_args[i]->Decref();
      continue;
    }
    nre_subs[j] = child_args[i];
    j++;
  }
  return nre;
}

bool CoalesceWalker::CanCoalesce(Regexp* r1, Regexp* r2) {
  // r1 must be a star/plus/quest/repeat of a literal, char class, any char or
  // any byte.
  if ((r1->op() == kRegexpStar ||
       r1->op() == kRegexpPlus ||
       r1->op() == kRegexpQuest ||
       r1->op() == kRegexpRepeat) &&
      (r1->sub()[0]->op() == kRegexpLiteral ||
       r1->sub()[0]->op() == kRegexpCharClass ||
       r1->sub()[0]->op() == kRegexpAnyChar ||
       r1->sub()[0]->op() == kRegexpAnyByte)) {
    // r2 must be a star/plus/quest/repeat of the same literal, char class,
    // any char or any byte.
    if ((r2->op() == kRegexpStar ||
         r2->op() == kRegexpPlus ||
         r2->op() == kRegexpQuest ||
         r2->op() == kRegexpRepeat) &&
        Regexp::Equal(r1->sub()[0], r2->sub()[0]) &&
        // The parse flags must be consistent.
        ((r1->parse_flags() & Regexp::NonGreedy) ==
         (r2->parse_flags() & Regexp::NonGreedy))) {
      return true;
    }
    // ... OR an occurrence of that literal, char class, any char or any byte
    if (Regexp::Equal(r1->sub()[0], r2)) {
      return true;
    }
    // ... OR a literal string that begins with that literal.
    if (r1->sub()[0]->op() == kRegexpLiteral &&
        r2->op() == kRegexpLiteralString &&
        r2->runes()[0] == r1->sub()[0]->rune() &&
        // The parse flags must be consistent.
        ((r1->sub()[0]->parse_flags() & Regexp::FoldCase) ==
         (r2->parse_flags() & Regexp::FoldCase))) {
      return true;
    }
  }
  return false;
}

void CoalesceWalker::DoCoalesce(Regexp** r1ptr, Regexp** r2ptr) {
  Regexp* r1 = *r1ptr;
  Regexp* r2 = *r2ptr;

  Regexp* nre = Regexp::Repeat(
      r1->sub()[0]->Incref(), r1->parse_flags(), 0, 0);

  switch (r1->op()) {
    case kRegexpStar:
      nre->min_ = 0;
      nre->max_ = -1;
      break;

    case kRegexpPlus:
      nre->min_ = 1;
      nre->max_ = -1;
      break;

    case kRegexpQuest:
      nre->min_ = 0;
      nre->max_ = 1;
      break;

    case kRegexpRepeat:
      nre->min_ = r1->min();
      nre->max_ = r1->max();
      break;

    default:
      LOG(DFATAL) << "DoCoalesce failed: r1->op() is " << r1->op();
      nre->Decref();
      return;
  }

  switch (r2->op()) {
    case kRegexpStar:
      nre->max_ = -1;
      goto LeaveEmpty;

    case kRegexpPlus:
      nre->min_++;
      nre->max_ = -1;
      goto LeaveEmpty;

    case kRegexpQuest:
      if (nre->max() != -1)
        nre->max_++;
      goto LeaveEmpty;

    case kRegexpRepeat:
      nre->min_ += r2->min();
      if (r2->max() == -1)
        nre->max_ = -1;
      else if (nre->max() != -1)
        nre->max_ += r2->max();
      goto LeaveEmpty;

    case kRegexpLiteral:
    case kRegexpCharClass:
    case kRegexpAnyChar:
    case kRegexpAnyByte:
      nre->min_++;
      if (nre->max() != -1)
        nre->max_++;
      goto LeaveEmpty;

    LeaveEmpty:
      *r1ptr = new Regexp(kRegexpEmptyMatch, Regexp::NoParseFlags);
      *r2ptr = nre;
      break;

    case kRegexpLiteralString: {
      Rune r = r1->sub()[0]->rune();
      // Determine how much of the literal string is removed.
      // We know that we have at least one rune. :)
      int n = 1;
      while (n < r2->nrunes() && r2->runes()[n] == r)
        n++;
      nre->min_ += n;
      if (nre->max() != -1)
        nre->max_ += n;
      if (n == r2->nrunes())
        goto LeaveEmpty;
      *r1ptr = nre;
      *r2ptr = Regexp::LiteralString(
          &r2->runes()[n], r2->nrunes() - n, r2->parse_flags());
      break;
    }

    default:
      LOG(DFATAL) << "DoCoalesce failed: r2->op() is " << r2->op();
      nre->Decref();
      return;
  }

  r1->Decref();
  r2->Decref();
}

Regexp* SimplifyWalker::Copy(Regexp* re) {
  return re->Incref();
}

Regexp* SimplifyWalker::ShortVisit(Regexp* re, Regexp* parent_arg) {
  // This should never be called, since we use Walk and not
  // WalkExponential.
  LOG(DFATAL) << "SimplifyWalker::ShortVisit called";
  return re->Incref();
}

Regexp* SimplifyWalker::PreVisit(Regexp* re, Regexp* parent_arg, bool* stop) {
  if (re->simple()) {
    *stop = true;
    return re->Incref();
  }
  return NULL;
}

Regexp* SimplifyWalker::PostVisit(Regexp* re,
                                  Regexp* parent_arg,
                                  Regexp* pre_arg,
                                  Regexp** child_args,
                                  int nchild_args) {
  switch (re->op()) {
    case kRegexpNoMatch:
    case kRegexpEmptyMatch:
    case kRegexpLiteral:
    case kRegexpLiteralString:
    case kRegexpBeginLine:
    case kRegexpEndLine:
    case kRegexpBeginText:
    case kRegexpWordBoundary:
    case kRegexpNoWordBoundary:
    case kRegexpEndText:
    case kRegexpAnyChar:
    case kRegexpAnyByte:
    case kRegexpHaveMatch:
      // All these are always simple.
      re->simple_ = true;
      return re->Incref();

    case kRegexpConcat:
    case kRegexpAlternate: {
      // These are simple as long as the subpieces are simple.
      if (!ChildArgsChanged(re, child_args)) {
        re->simple_ = true;
        return re->Incref();
      }
      Regexp* nre = new Regexp(re->op(), re->parse_flags());
      nre->AllocSub(re->nsub());
      Regexp** nre_subs = nre->sub();
      for (int i = 0; i < re->nsub(); i++)
        nre_subs[i] = child_args[i];
      nre->simple_ = true;
      return nre;
    }

    case kRegexpCapture: {
      Regexp* newsub = child_args[0];
      if (newsub == re->sub()[0]) {
        newsub->Decref();
        re->simple_ = true;
        return re->Incref();
      }
      Regexp* nre = new Regexp(kRegexpCapture, re->parse_flags());
      nre->AllocSub(1);
      nre->sub()[0] = newsub;
      nre->cap_ = re->cap();
      nre->simple_ = true;
      return nre;
    }

    case kRegexpStar:
    case kRegexpPlus:
    case kRegexpQuest: {
      Regexp* newsub = child_args[0];
      // Special case: repeat the empty string as much as
      // you want, but it's still the empty string.
      if (newsub->op() == kRegexpEmptyMatch)
        return newsub;

      // These are simple as long as the subpiece is simple.
      if (newsub == re->sub()[0]) {
        newsub->Decref();
        re->simple_ = true;
        return re->Incref();
      }

      // These are also idempotent if flags are constant.
      if (re->op() == newsub->op() &&
          re->parse_flags() == newsub->parse_flags())
        return newsub;

      Regexp* nre = new Regexp(re->op(), re->parse_flags());
      nre->AllocSub(1);
      nre->sub()[0] = newsub;
      nre->simple_ = true;
      return nre;
    }

    case kRegexpRepeat: {
      Regexp* newsub = child_args[0];
      // Special case: repeat the empty string as much as
      // you want, but it's still the empty string.
      if (newsub->op() == kRegexpEmptyMatch)
        return newsub;

      Regexp* nre = SimplifyRepeat(newsub, re->min_, re->max_,
                                   re->parse_flags());
      newsub->Decref();
      nre->simple_ = true;
      return nre;
    }

    case kRegexpCharClass: {
      Regexp* nre = SimplifyCharClass(re);
      nre->simple_ = true;
      return nre;
    }
  }

  LOG(ERROR) << "Simplify case not handled: " << re->op();
  return re->Incref();
}

// Creates a concatenation of two Regexp, consuming refs to re1 and re2.
// Returns a new Regexp, handing the ref to the caller.
Regexp* SimplifyWalker::Concat2(Regexp* re1, Regexp* re2,
                                Regexp::ParseFlags parse_flags) {
  Regexp* re = new Regexp(kRegexpConcat, parse_flags);
  re->AllocSub(2);
  Regexp** subs = re->sub();
  subs[0] = re1;
  subs[1] = re2;
  return re;
}

// Simplifies the expression re{min,max} in terms of *, +, and ?.
// Returns a new regexp.  Does not edit re.  Does not consume reference to re.
// Caller must Decref return value when done with it.
// The result will *not* necessarily have the right capturing parens
// if you call ToString() and re-parse it: (x){2} becomes (x)(x),
// but in the Regexp* representation, both (x) are marked as $1.
Regexp* SimplifyWalker::SimplifyRepeat(Regexp* re, int min, int max,
                                       Regexp::ParseFlags f) {
  // x{n,} means at least n matches of x.
  if (max == -1) {
    // Special case: x{0,} is x*
    if (min == 0)
      return Regexp::Star(re->Incref(), f);

    // Special case: x{1,} is x+
    if (min == 1)
      return Regexp::Plus(re->Incref(), f);

    // General case: x{4,} is xxxx+
    PODArray<Regexp*> nre_subs(min);
    for (int i = 0; i < min-1; i++)
      nre_subs[i] = re->Incref();
    nre_subs[min-1] = Regexp::Plus(re->Incref(), f);
    return Regexp::Concat(nre_subs.data(), min, f);
  }

  // Special case: (x){0} matches only empty string.
  if (min == 0 && max == 0)
    return new Regexp(kRegexpEmptyMatch, f);

  // Special case: x{1} is just x.
  if (min == 1 && max == 1)
    return re->Incref();

  // General case: x{n,m} means n copies of x and m copies of x?.
  // The machine will do less work if we nest the final m copies,
  // so that x{2,5} = xx(x(x(x)?)?)?

  // Build leading prefix: xx.  Capturing only on the last one.
  Regexp* nre = NULL;
  if (min > 0) {
    PODArray<Regexp*> nre_subs(min);
    for (int i = 0; i < min; i++)
      nre_subs[i] = re->Incref();
    nre = Regexp::Concat(nre_subs.data(), min, f);
  }

  // Build and attach suffix: (x(x(x)?)?)?
  if (max > min) {
    Regexp* suf = Regexp::Quest(re->Incref(), f);
    for (int i = min+1; i < max; i++)
      suf = Regexp::Quest(Concat2(re->Incref(), suf, f), f);
    if (nre == NULL)
      nre = suf;
    else
      nre = Concat2(nre, suf, f);
  }

  if (nre == NULL) {
    // Some degenerate case, like min > max, or min < max < 0.
    // This shouldn't happen, because the parser rejects such regexps.
    LOG(DFATAL) << "Malformed repeat " << re->ToString() << " " << min << " " << max;
    return new Regexp(kRegexpNoMatch, f);
  }

  return nre;
}

// Simplifies a character class.
// Caller must Decref return value when done with it.
Regexp* SimplifyWalker::SimplifyCharClass(Regexp* re) {
  CharClass* cc = re->cc();

  // Special cases
  if (cc->empty())
    return new Regexp(kRegexpNoMatch, re->parse_flags());
  if (cc->full())
    return new Regexp(kRegexpAnyChar, re->parse_flags());

  return re->Incref();
}

}  // namespace re2
