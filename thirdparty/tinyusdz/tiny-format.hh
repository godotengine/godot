// SPDX-License-Identifier: MIT
// Copyright 2022-Present Syoyo Fujita.

///
/// Simple Python-like format print utility in C++11 or later. Only supports
/// "{}".
///
#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "nonstd/expected.hpp"

namespace tinyusdz {
namespace fmt {

namespace detail {

template <class T>
std::ostringstream &format_sv_rec(std::ostringstream &ss,
                                  const std::vector<std::string> &sv,
                                  size_t idx, T const &v) {
  if (idx >= sv.size()) {
    return ss;
  }

  // Print remaininig items
  bool fmt_printed{false};

  for (size_t i = idx; i < sv.size(); i++) {
    if (sv[i] == "{}") {
      if (fmt_printed) {
        ss << sv[i];
      } else {
        ss << v;
        fmt_printed = true;
      }
    } else {
      ss << sv[i];
    }
  }

  return ss;
}

template <class T, class... Rest>
std::ostringstream &format_sv_rec(std::ostringstream &ss,
                                  const std::vector<std::string> &sv,
                                  size_t idx, T const &v, Rest const &...args) {
  if (idx >= sv.size()) {
    return ss;
  }

  if (sv[idx] == "{}") {
    ss << v;
    format_sv_rec(ss, sv, idx + 1, args...);
  } else {
    ss << sv[idx];
    format_sv_rec(ss, sv, idx + 1, v, args...);
  }

  return ss;
}

template <class... Args>
std::ostringstream &format_sv(std::ostringstream &ss,
                              const std::vector<std::string> &sv,
                              Args const &...args) {
  format_sv_rec(ss, sv, 0, args...);

  return ss;
}

std::ostringstream &format_sv(std::ostringstream &ss,
                              const std::vector<std::string> &sv);

nonstd::expected<std::vector<std::string>, std::string> tokenize(
    const std::string &s);

}  // namespace detail

template <class... Args>
std::string format(const std::string &in, Args const &...args) {
  auto ret = detail::tokenize(in);
  if (!ret) {
    return in + "(format error: " + ret.error() + ")";
  }

  std::ostringstream ss;
  detail::format_sv(ss, (*ret), args...);

  return ss.str();
}

std::string format(const std::string &in);

}  // namespace fmt
}  // namespace tinyusdz
