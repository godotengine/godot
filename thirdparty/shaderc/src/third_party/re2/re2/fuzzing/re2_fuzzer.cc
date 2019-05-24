// Copyright 2016 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stddef.h>
#include <stdint.h>
#include <map>
#include <memory>
#include <queue>
#include <string>

#include "re2/prefilter.h"
#include "re2/re2.h"

using re2::StringPiece;

// NOT static, NOT signed.
uint8_t dummy = 0;

void Test(StringPiece pattern, const RE2::Options& options, StringPiece text) {
  RE2 re(pattern, options);
  if (!re.ok())
    return;

  // Don't waste time fuzzing high-size programs.
  // They can cause bug reports due to fuzzer timeouts.
  int size = re.ProgramSize();
  if (size > 9999)
    return;
  int rsize = re.ReverseProgramSize();
  if (rsize > 9999)
    return;

  // Don't waste time fuzzing high-fanout programs.
  // They can cause bug reports due to fuzzer timeouts.
  std::map<int, int> histogram;
  int fanout = re.ProgramFanout(&histogram);
  if (fanout > 9)
    return;
  int rfanout = re.ReverseProgramFanout(&histogram);
  if (rfanout > 9)
    return;

  // Don't waste time fuzzing programs with large substrings.
  // They can cause bug reports due to fuzzer timeouts when they
  // are repetitions (e.g. hundreds of NUL bytes) and matching is
  // unanchored. And they aren't interesting for fuzzing purposes.
  std::unique_ptr<re2::Prefilter> prefilter(re2::Prefilter::FromRE2(&re));
  if (prefilter == nullptr)
    return;
  std::queue<re2::Prefilter*> nodes;
  nodes.push(prefilter.get());
  while (!nodes.empty()) {
    re2::Prefilter* node = nodes.front();
    nodes.pop();
    if (node->op() == re2::Prefilter::ATOM) {
      if (node->atom().size() > 9)
        return;
    } else if (node->op() == re2::Prefilter::AND ||
               node->op() == re2::Prefilter::OR) {
      for (re2::Prefilter* sub : *node->subs())
        nodes.push(sub);
    }
  }

  if (re.NumberOfCapturingGroups() == 0) {
    // Avoid early return due to too many arguments.
    StringPiece sp = text;
    RE2::FullMatch(sp, re);
    RE2::PartialMatch(sp, re);
    RE2::Consume(&sp, re);
    sp = text;  // Reset.
    RE2::FindAndConsume(&sp, re);
  } else {
    // Okay, we have at least one capturing group...
    // Try conversion for variously typed arguments.
    StringPiece sp = text;
    short s;
    RE2::FullMatch(sp, re, &s);
    long l;
    RE2::PartialMatch(sp, re, &l);
    float f;
    RE2::Consume(&sp, re, &f);
    sp = text;  // Reset.
    double d;
    RE2::FindAndConsume(&sp, re, &d);
  }

  std::string s = std::string(text);
  RE2::Replace(&s, re, "");
  s = std::string(text);  // Reset.
  RE2::GlobalReplace(&s, re, "");

  std::string min, max;
  re.PossibleMatchRange(&min, &max, /*maxlen=*/9);

  // Exercise some other API functionality.
  dummy += re.NamedCapturingGroups().size();
  dummy += re.CapturingGroupNames().size();
  dummy += RE2::QuoteMeta(pattern).size();
}

// Entry point for libFuzzer.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size == 0 || size > 999)
    return 0;

  // Crudely limit the use of ., \p, \P, \d, \D, \s, \S, \w and \W.
  // Otherwise, we will waste time on inputs that have long runs of various
  // character classes. The fuzzer has shown itself to be easily capable of
  // generating such patterns that fall within the other limits, but result
  // in timeouts nonetheless. The marginal cost is high - even more so when
  // counted repetition is involved - whereas the marginal benefit is zero.
  // TODO(junyer): Handle [:isalnum:] et al. when they start to cause pain.
  int char_class = 0;
  int backslash_p = 0;  // very expensive, so handle specially
  for (size_t i = 0; i < size; i++) {
    if (data[i] == '.')
      char_class++;
    if (data[i] != '\\')
      continue;
    i++;
    if (i >= size)
      break;
    if (data[i] == 'p' || data[i] == 'P' ||
        data[i] == 'd' || data[i] == 'D' ||
        data[i] == 's' || data[i] == 'S' ||
        data[i] == 'w' || data[i] == 'W')
      char_class++;
    if (data[i] == 'p' || data[i] == 'P')
      backslash_p++;
  }
  if (char_class > 9)
    return 0;
  if (backslash_p > 1)
    return 0;

  // The one-at-a-time hash by Bob Jenkins.
  uint32_t hash = 0;
  for (size_t i = 0; i < size; i++) {
    hash += data[i];
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);

  RE2::Options options;
  options.set_log_errors(false);
  options.set_max_mem(64 << 20);
  options.set_encoding(hash & 1 ? RE2::Options::EncodingLatin1
                                : RE2::Options::EncodingUTF8);
  options.set_posix_syntax(hash & 2);
  options.set_longest_match(hash & 4);
  options.set_literal(hash & 8);
  options.set_never_nl(hash & 16);
  options.set_dot_nl(hash & 32);
  options.set_never_capture(hash & 64);
  options.set_case_sensitive(hash & 128);
  options.set_perl_classes(hash & 256);
  options.set_word_boundary(hash & 512);
  options.set_one_line(hash & 1024);

  const char* ptr = reinterpret_cast<const char*>(data);
  int len = static_cast<int>(size);

  StringPiece pattern(ptr, len);
  StringPiece text(ptr, len);
  Test(pattern, options, text);

  return 0;
}
