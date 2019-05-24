# Developing for Effcee

Thank you for considering Effcee development!  Please make sure you review
[`CONTRIBUTING.md`](CONTRIBUTING.md) for important preliminary info.

## Building

Instructions for first-time building can be found in [`README.md`](README.md).
Incremental build after a source change can be done using `ninja` (or
`cmake --build`) and `ctest` exactly as in the first-time procedure.

## Issue tracking

We use GitHub issues to track bugs, enhancement requests, and questions.
See [the project's Issues page](https://github.com/google/effcee/issues).

For all but the most trivial changes, we prefer that you file an issue before
submitting a pull request.  An issue gives us context for your change: what
problem are you solving, and why.  It also allows us to provide feedback on
your proposed solution before you invest a lot of effort implementing it.

## Code reviews

All submissions are subject to review via the GitHub pull review process.
Reviews will cover:

* *Correctness:* Does it work?  Does it work in a multithreaded context?
* *Testing:* New functionality should be accompanied by tests.
* *Testability:* Can it easily be tested?  This is proven with accompanying tests.
* *Design:* Is the solution fragile? Does it fit with the existing code?
  Would it easily accommodate anticipated changes?
* *Ease of use:* Can a client get their work done with a minimum of fuss?
  Are there unnecessarily surprising details?
* *Consistency:* Does it follow the style guidelines and the rest of the code?
  Consistency reduces the work of future readers and maintainers.
* *Portability:* Does it work in many environments?

To respond to feedback, submit one or more *new* commits to the pull request
branch. The project maintainer will normally clean up the submission by
squashing feedback response commits.  We maintain a linear commit history,
so submission will be rebased onto master before merging.

## Testing

There is a lot we won't say about testing. However:

* Most tests should be small scale, i.e. unit tests.
* Tests should run quickly.
* A test should:
  * Check a single behaviour.  This often corresponds to a use case.
  * Have a three phase structure: setup, action, check.

## Coding style

For C++, we follow the
[Google C++ style guide](https://google.github.io/styleguide/cppguide.html).

Use `clang-format` to format the code.

For our Python files, we aim to follow the
[Google Python style guide](https://google.github.io/styleguide/pyguide.html).
