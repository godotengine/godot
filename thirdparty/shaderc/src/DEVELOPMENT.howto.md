Thank you for considering Shaderc development!  Please make sure you review
[`CONTRIBUTING.md`](CONTRIBUTING.md) for important preliminary info.

## Building

Instructions for first-time building can be found in [`README.md`](README.md).
Incremental build after a source change can be done using `ninja` (or
`cmake --build`) and `ctest` exactly as in the first-time procedure.

## Code reviews

(Terminology: we consider everyone with write access to our GitHub repo a
project _member_.)

All submissions, including submissions by project members, require review.  We
use GitHub pull requests to facilitate the review process.  A submission may be
accepted by any project member (other than the submitter), who will then squash
the changes into a single commit and cherry-pick them into the repository.

Before accepting, there may be some review feedback prompting changes in the
submission.  You should expect reviewers to strictly insist on the
[commenting](https://google.github.io/styleguide/cppguide.html#Comments)
guidelines -- in particular, every file, class, method, data member, and global
will require a comment.  Reviewers will also expect to see test coverage for
every code change.  _How much_ coverage will be a judgment call on a
case-by-case basis, balancing the required effort against the incremental
benefit.  Coverage will be expected.  As a matter of development philosophy,
we will strive to engineer the code to make writing tests easy.

## Coding style

For our C++ files, we use the
[Google C++ style guide](https://google.github.io/styleguide/cppguide.html).
(Conveniently, the formatting rules it specifies can be achieved using
`clang-format -style=google`.)

For our Python files, we use the
[Google Python style guide](https://google.github.io/styleguide/pyguide.html).

## Supported platforms

We expect Shaderc to always build and test successfully on the platforms listed
below.  Please keep that in mind when offering contributions.  This list will
likely grow over time.

| Platform | Build Status |
|:--------:|:------------:|
| Android (ARMv7)  | Not Automated |
| Linux (x86_64)   | [![Linux Build Status](https://travis-ci.org/google/shaderc.svg)](https://travis-ci.org/google/shaderc "Linux Build Status") |
| Mac OS X | [![Mac Build Status](https://travis-ci.org/google/shaderc.svg)](https://travis-ci.org/google/shaderc "Mac Build Status") |
| Windows (x86_64) | [![Windows Build status](https://ci.appveyor.com/api/projects/status/g6c372blna7vnk1l?svg=true)](https://ci.appveyor.com/project/dneto0/shaderc "Windows Build Status") |

