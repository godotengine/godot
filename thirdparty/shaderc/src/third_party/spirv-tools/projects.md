# Tracking SPIRV-Tools work with GitHub projects

We are experimenting with using the [GitHub Project
feature](https://help.github.com/articles/tracking-the-progress-of-your-work-with-projects/)
to track progress toward large goals.

For more on GitHub Projects in general, see:
* [Introductory blog post](https://github.com/blog/2256-a-whole-new-github-universe-announcing-new-tools-forums-and-features)
* [Introductory video](https://www.youtube.com/watch?v=C6MGKHkNtxU)

The current SPIRV-Tools project list can be found at
[https://github.com/KhronosGroup/SPIRV-Tools/projects](https://github.com/KhronosGroup/SPIRV-Tools/projects)

## How we use a Project

A GitHub Project is a set of work with an overall purpose, and
consists of a collection of *Cards*.
Each card is either a *Note* or a regular GitHub *Issue.*
A Note can be converted to an Issue.

In our projects, a card represents work, i.e. a change that can
be applied to the repository.
The work could be a feature, a bug to be fixed, documentation to be
updated, etc.

A project and its cards are used as a [Kanban
board](https://en.wikipedia.org/wiki/Kanban_board), where cards progress
through a workflow starting with ideas through to implementation and completion.

In our usage, a *project manager* is someone who organizes the work.
They manage the creation and movement of cards
through the project workflow:
* They create cards to capture ideas, or to decompose large ideas into smaller
  ones.
* They determine if the work for a card has been completed.
* Normally they are the person (or persons) who can approve and merge a pull
  request into the `master` branch.

Our projects organize cards into the following columns:
* `Ideas`: Work which could be done, captured either as Cards or Notes.
  * A card in this column could be marked as a [PLACEHOLDER](#placeholders).
* `Ready to start`: Issues which represent work we'd like to do, and which
  are not blocked by other work.
  * The issue should be narrow enough that it can usually be addressed by a
    single pull request.
  * We want these to be Issues (not Notes) so that someone can claim the work
    by updating the Issue with their intent to do the work.
    Once an Issue is claimed, the project manager moves the corresponding card
    from `Ready to start` to `In progress`.
* `In progress`: Issues which were in `Ready to start` but which have been
  claimed by someone.
* `Done`: Issues which have been resolved, by completing their work.
  * The changes have been applied to the repository, typically by being pushed
  into the `master` branch.
  * Other kinds of work could update repository settings, for example.
* `Rejected ideas`: Work which has been considered, but which we don't want
  implemented.
  * We keep rejected ideas so they are not proposed again. This serves
    as a form of institutional memory.
  * We should record why an idea is rejected. For this reason, a rejected
    idea is likely to be an Issue which has been closed.

## Prioritization

We are considering prioritizing cards in the `Ideas` and `Ready to start`
columns so that things that should be considered first float up to the top.

Experience will tell us if we stick to that rule, and if it proves helpful.

## Placeholders

A *placeholder* is a Note or Issue that represents a possibly large amount
of work that can be broadly defined but which may not have been broken down
into small implementable pieces of work.

Use a placeholder to capture a big idea, but without doing the upfront work
to consider all the details of how it should be implemented.
Over time, break off pieces of the placeholder into implementable Issues.
Move those Issues into the `Ready to start` column when they become unblocked.

We delete the placeholder when all its work has been decomposed into
implementable cards.
