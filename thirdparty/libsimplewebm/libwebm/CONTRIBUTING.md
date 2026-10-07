# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use a [Gerrit](https://www.gerritcodereview.com) instance hosted at
https://chromium-review.googlesource.com for this purpose. See the
[WebM Project page](https://www.webmproject.org/code/contribute/submitting-patches/)
for additional details.

## Code Style

The C++ code style is based on the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and
`clang-format --style=Google`. `clang-format -i --style=file` can be used to
format individual files, it will use the settings from `.clang-format`.

CMake files are formatted with
[cmake-format](https://cmake-format.readthedocs.io/en/latest/). `cmake-format
-i` can be used to format individual files, it will use the settings from
`.cmake-format.py`.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
