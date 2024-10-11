# Contributors guidelines

This document summarizes the most important points for people interested in
contributing to Redot, especially via bug reports or pull requests.

The [Redot Discord](https://discord.gg/redot) documentation has a dedicated Contributing section
which details these points and more, and is a recommended read.

## Table of contents

- [Reporting bugs](#reporting-bugs)
- [Proposing features or improvements](#proposing-features-or-improvements)
- [Contributing pull requests](#contributing-pull-requests)
- [Contributing to Godot translations](#contributing-to-godot-translations)
- [Communicating with developers](#communicating-with-developers)

## Reporting bugs

Report bugs [here](https://github.com/Redot-Engine/redot-engine/issues/new?assignees=&labels=&template=bug_report.yml).
Please follow the instructions in the template when you do.

Notably, please include a Minimal Reproduction Project (MRP), which is a small
Redot project which reproduces the issue, with no unnecessary files included.
Be sure to not include the `.godot` folder in the archive to save space.

Make sure that the bug you are experiencing is reproducible in the latest Redot
releases. It's worth testing against both the latest stable release and the
latest dev snapshot for the next Redot release.

If you run into a bug which wasn't present in an earlier Redot version (what we
call a _regression_), please mention it and clarify which versions you tested
(both the one(s) working and the one(s) exhibiting the bug).

## Proposing features or improvements

**The main issue tracker is for bug reports and does not accept feature proposals.**

Instead, head to the [Redot Discord](https://discord.gg/redot).

## Contributing pull requests

If you want to add new engine features, please make sure that:

- This functionality is desired, which means that it solves a common use case
  that several users will need in their real-life projects.
- You talked to other developers on how to implement it best. See also
  [Proposing features or improvements](#proposing-features-or-improvements).
- Even if it doesn't get merged, your PR is useful for future work by another
  developer.

Similar rules can be applied when contributing bug fixes - it's always best to
discuss the implementation in the bug report first if you are not 100% about
what would be the best fix.

Visit the [Redot Discord](https://discord.gg/redot) for information and guides related to contribution and pull requests.

### Be mindful of your commits

Try to make simple PRs that handle one specific topic. Just like for reporting
issues, it's better to open 3 different PRs that each address a different issue
than one big PR with three commits. This makes it easier to review, approve, and
merge the changes independently.

When updating your fork with upstream changes, please use ``git pull --rebase``
to avoid creating "merge commits". Those commits unnecessarily pollute the git
history when coming from PRs.

Also try to make commits that bring the engine from one stable state to another
stable state, i.e. if your first commit has a bug that you fixed in the second
commit, try to merge them together before making your pull request. This
includes fixing build issues or typos, adding documentation, etc.

This [Git style guide](https://github.com/agis-/git-style-guide) also has some
good practices to have in mind.

### Format your commit messages with readability in mind

The way you format your commit messages is quite important to ensure that the
commit history and changelog will be easy to read and understand. A Git commit
message is formatted as a short title (first line) and an extended description
(everything after the first line and an empty separation line).

The short title is the most important part, as it is what will appear in the
changelog or in the GitHub interface unless you click the "expand" button.
Try to keep that first line under 72 characters, but you can go slightly above
if necessary to keep the sentence clear.

It should be written in English, starting with a capital letter, and usually
with a verb in imperative form. A typical bugfix would start with "Fix", while
the addition of a new feature would start with "Add". A prefix can be added to
specify the engine area affected by the commit. Some examples:

- Add C# iOS support
- Show doc tooltips when hovering properties in the theme editor
- Fix GLES3 instanced rendering color and custom data defaults
- Core: Fix `Object::has_method()` for script static methods

If your commit fixes a reported issue, please include it in the _description_
of the PR (not in the title, or the commit message) using one of the
[GitHub closing keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)
such as "Fixes #1234". This will cause the issue to be closed automatically if
the PR is merged. Adding it to the commit message is easier, but adds a lot of
unnecessary updates in the issue distracting from the thread.

Here's an example of a well-formatted commit message (note how the extended
description is also manually wrapped at 80 chars for readability):

```text
Prevent French fries carbonization by fixing heat regulation

When using the French fries frying module, Redot would not regulate the heat
and thus bring the oil bath to supercritical liquid conditions, thus causing
unwanted side effects in the physics engine.

By fixing the regulation system via an added binding to the internal feature,
this commit now ensures that Redot will not go past the ebullition temperature
of cooking oil under normal atmospheric conditions.
```

**Note:** When using the GitHub online editor or its drag-and-drop
feature, *please* edit the commit title to something meaningful. Commits named
"Update my_file.cpp" won't be accepted.

### Document your changes

If your pull request adds methods, properties or signals that are exposed to
scripting APIs, you **must** update the class reference to document those.
This is to ensure the documentation coverage doesn't decrease as contributions
are merged.

[Update documentation XML files](https://docs.godotengine.org/en/latest/contributing/documentation/updating_the_class_reference.html)
using your compiled binary, then fill in the descriptions.
Follow the style guide described in the
[Documentation writing guidelines](https://docs.godotengine.org/en/latest/contributing/documentation/docs_writing_guidelines.html).

If your pull request modifies parts of the code in a non-obvious way, make sure
to add comments in the code as well. This helps other people understand the
change without having to dive into the Git history.

### Write unit tests

When fixing a bug or contributing a new feature, we recommend including unit
tests in the same commit as the rest of the pull request. Unit tests are pieces
of code that compare the output to a predetermined *expected result* to detect
regressions. Tests are compiled and run on GitHub Actions for every commit and
pull request.

Pull requests that include tests are more likely to be merged, since we can have
greater confidence in them not being the target of regressions in the future.

For bugs, the unit tests should cover the functionality that was previously
broken. If done well, this ensures regressions won't appear in the future
again. For new features, the unit tests should cover the newly added
functionality, testing both the "success" and "expected failure" cases if
applicable.

Feel free to contribute standalone pull requests to add new tests or improve
existing tests as well.

See [Unit testing](https://docs.godotengine.org/en/latest/contributing/development/core_and_modules/unit_testing.html)
for information on writing tests in Godot's C++ codebase.

## Contributing to Godot translations

You can contribute to Godot translations on [Hosted Weblate](https://hosted.weblate.org/projects/godot-engine/),
an open source and web-based translation platform.

Please refer to Godot's [editor and documentation localization guidelines](https://docs.godotengine.org/en/latest/contributing/documentation/editor_and_docs_localization.html)
for an overview of the translation resources and what they correspond to.

## Communicating with developers

The Redot Engine community is largely centralized on the [Redot Discord](https://discord.gg/redot) at this time, feel free to drop by!

To communicate with developers (e.g. to discuss a feature you want to implement
or a bug you want to fix), the following channels can be used:

- [Redot Discord](https://discord.gg/redot): General community chat and development can all be found here.
- [Bug tracker](https://github.com/Redot-Engine/redot-engine/issues): If there is an
  existing issue about a topic you want to discuss, you can participate directly.
  If not, you can open a new issue. Please mind the guidelines outlined above
  for bug reporting.
- Feature Proposals should be made on the [Redot Discord](https://discord.gg/redot) for now.
- [Redot](https://x.com/redotengine) on Twitter

Thanks for your interest in contributing!

—Redot Development Community
