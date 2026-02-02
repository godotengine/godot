// SPDX-License-Identifier: MIT
// Copyright 2022-Present Syoyo Fujita.
#include "tiny-format.hh"

namespace tinyusdz {
namespace fmt {

namespace detail {

nonstd::expected<std::vector<std::string>, std::string> tokenize(
    const std::string &s) {
  size_t n = s.length();

  bool open_curly_brace = false;

  std::vector<std::string> toks;
  size_t si = 0;

  for (size_t i = 0; i < n; i++) {
    if (s[i] == '{') {
      if (open_curly_brace) {
        // nested '{'
        return nonstd::make_unexpected("Nested '{'.");
      }

      open_curly_brace = true;

      if (si >= i) {  // previous char is '}'
        // do nothing
      } else {
        toks.push_back(
            std::string(s.begin() + std::string::difference_type(si),
                        s.begin() + std::string::difference_type(i)));

        si = i;
      }

    } else if (s[i] == '}') {
      if (open_curly_brace) {
        // must be "{}" for now
        if ((i - si) > 1) {
          return nonstd::make_unexpected(
              "Format specifier in '{}' is not yet supported.");
        }

        open_curly_brace = false;

        toks.push_back("{}");

        si = i + 1;  // start from next char.

      } else {
        // Currently we allow string like '}', "}}", "bora}".
        // TODO: strict check for '{' pair.
      }
    }
  }

  if (si < n) {
    toks.push_back(std::string(s.begin() + std::string::difference_type(si),
                               s.begin() + std::string::difference_type(n)));
  }

  return std::move(toks);
}

std::ostringstream &format_sv(std::ostringstream &ss,
                              const std::vector<std::string> &sv) {
  if (sv.empty()) {
    return ss;
  }

  for (const auto &item : sv) {
    ss << item;
  }

  return ss;
}

}  // namespace detail

std::string format(const std::string &in) { return in; }

}  // namespace fmt
}  // namespace tinyusdz

#if 0
void test(const std::string &in) {
  std::cout << tinyusdz::fmt::format(in) << "\n";
  std::cout << tinyusdz::fmt::format(in, 1.0f) << "\n";
  std::cout << tinyusdz::fmt::format(in, 1.0f, 2.0f) << "\n";
  std::cout << tinyusdz::fmt::format(in, 1.0f, 2.0f, 3.0f) << "\n";
}

int main(int argc, char **argv) {
  test("{}");
  test("{");
  test("}");
  test("{{");
  test("}}");
  test("{a}");
  test("bora {}");
  test("{} dora");
  test("{} dora{} bora muda {");
  test("{} dora{} bora muda{}");

  return 0;
}
#endif
