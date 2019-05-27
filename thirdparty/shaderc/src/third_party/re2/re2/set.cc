// Copyright 2010 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "re2/set.h"

#include <stddef.h>
#include <algorithm>
#include <memory>

#include "util/util.h"
#include "util/logging.h"
#include "util/pod_array.h"
#include "re2/stringpiece.h"
#include "re2/prog.h"
#include "re2/re2.h"
#include "re2/regexp.h"

namespace re2 {

RE2::Set::Set(const RE2::Options& options, RE2::Anchor anchor) {
  options_.Copy(options);
  options_.set_never_capture(true);  // might unblock some optimisations
  anchor_ = anchor;
  prog_ = NULL;
  compiled_ = false;
  size_ = 0;
}

RE2::Set::~Set() {
  for (size_t i = 0; i < elem_.size(); i++)
    elem_[i].second->Decref();
  delete prog_;
}

int RE2::Set::Add(const StringPiece& pattern, std::string* error) {
  if (compiled_) {
    LOG(DFATAL) << "RE2::Set::Add() called after compiling";
    return -1;
  }

  Regexp::ParseFlags pf = static_cast<Regexp::ParseFlags>(
    options_.ParseFlags());
  RegexpStatus status;
  re2::Regexp* re = Regexp::Parse(pattern, pf, &status);
  if (re == NULL) {
    if (error != NULL)
      *error = status.Text();
    if (options_.log_errors())
      LOG(ERROR) << "Error parsing '" << pattern << "': " << status.Text();
    return -1;
  }

  // Concatenate with match index and push on vector.
  int n = static_cast<int>(elem_.size());
  re2::Regexp* m = re2::Regexp::HaveMatch(n, pf);
  if (re->op() == kRegexpConcat) {
    int nsub = re->nsub();
    PODArray<re2::Regexp*> sub(nsub + 1);
    for (int i = 0; i < nsub; i++)
      sub[i] = re->sub()[i]->Incref();
    sub[nsub] = m;
    re->Decref();
    re = re2::Regexp::Concat(sub.data(), nsub + 1, pf);
  } else {
    re2::Regexp* sub[2];
    sub[0] = re;
    sub[1] = m;
    re = re2::Regexp::Concat(sub, 2, pf);
  }
  elem_.emplace_back(std::string(pattern), re);
  return n;
}

bool RE2::Set::Compile() {
  if (compiled_) {
    LOG(DFATAL) << "RE2::Set::Compile() called more than once";
    return false;
  }
  compiled_ = true;
  size_ = static_cast<int>(elem_.size());

  // Sort the elements by their patterns. This is good enough for now
  // until we have a Regexp comparison function. (Maybe someday...)
  std::sort(elem_.begin(), elem_.end(),
            [](const Elem& a, const Elem& b) -> bool {
              return a.first < b.first;
            });

  PODArray<re2::Regexp*> sub(size_);
  for (int i = 0; i < size_; i++)
    sub[i] = elem_[i].second;
  elem_.clear();
  elem_.shrink_to_fit();

  Regexp::ParseFlags pf = static_cast<Regexp::ParseFlags>(
    options_.ParseFlags());
  re2::Regexp* re = re2::Regexp::Alternate(sub.data(), size_, pf);

  prog_ = Prog::CompileSet(re, anchor_, options_.max_mem());
  re->Decref();
  return prog_ != NULL;
}

bool RE2::Set::Match(const StringPiece& text, std::vector<int>* v) const {
  return Match(text, v, NULL);
}

bool RE2::Set::Match(const StringPiece& text, std::vector<int>* v,
                     ErrorInfo* error_info) const {
  if (!compiled_) {
    LOG(DFATAL) << "RE2::Set::Match() called before compiling";
    if (error_info != NULL)
      error_info->kind = kNotCompiled;
    return false;
  }
  bool dfa_failed = false;
  std::unique_ptr<SparseSet> matches;
  if (v != NULL) {
    matches.reset(new SparseSet(size_));
    v->clear();
  }
  bool ret = prog_->SearchDFA(text, text, Prog::kAnchored, Prog::kManyMatch,
                              NULL, &dfa_failed, matches.get());
  if (dfa_failed) {
    if (options_.log_errors())
      LOG(ERROR) << "DFA out of memory: size " << prog_->size() << ", "
                 << "bytemap range " << prog_->bytemap_range() << ", "
                 << "list count " << prog_->list_count();
    if (error_info != NULL)
      error_info->kind = kOutOfMemory;
    return false;
  }
  if (ret == false) {
    if (error_info != NULL)
      error_info->kind = kNoError;
    return false;
  }
  if (v != NULL) {
    if (matches->empty()) {
      LOG(DFATAL) << "RE2::Set::Match() matched, but no matches returned?!";
      if (error_info != NULL)
        error_info->kind = kInconsistent;
      return false;
    }
    v->assign(matches->begin(), matches->end());
  }
  if (error_info != NULL)
    error_info->kind = kNoError;
  return true;
}

}  // namespace re2
