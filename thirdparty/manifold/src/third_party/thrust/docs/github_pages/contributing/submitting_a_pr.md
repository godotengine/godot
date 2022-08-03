---
parent: Contributing
nav_order: 0
---

# Submitting a PR

Thrust uses Github to manage all open-source development, including bug
tracking, pull requests, and design discussions. This document details how to get
started as a Thrust contributor.

An overview of this process is:

1. [Clone the Thrust repository](#clone-the-thrust-repository)
1. [Setup a fork of Thrust](#setup-a-fork-of-thrust)
1. [Setup your environment](#setup-your-environment)
1. [Create a development branch](#create-a-development-branch)
1. [Local development loop](#local-development-loop)
1. [Push development branch to your fork](#push-development-branch-to-your-fork)
1. [Create pull request](#create-pull-request)
1. [Address feedback and update pull request](#address-feedback-and-update-pull-request)
1. [When your PR is approved...](#when-your-pr-is-approved)

## Clone the Thrust Repository

To get started, clone the main repository to your local computer. Thrust should
be cloned recursively to setup the CUB submodule (required for `CUDA`
acceleration).

```
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust
```

## Setup a Fork of Thrust

You'll need a fork of Thrust on Github to create a pull request. To setup your
fork:

1. Create a Github account (if needed)
2. Go to [the Thrust Github page](https://github.com/NVIDIA/thrust)
3. Click "Fork" and follow any prompts that appear.

Once your fork is created, setup a new remote repo in your local Thrust clone:

```
git remote add github-fork git@github.com:<GITHUB_USERNAME>/thrust.git
```

If you need to modify CUB, too, go to
[the CUB Github page](https://github.com/NVIDIA/cub) and repeat this process.
Create CUB's `github-fork` remote in the `thrust/dependencies/cub` submodule.

## Setup Your Environment

### Git Environment

If you haven't already, this is a good time to tell git who you are. This
information is used to fill out authorship information on your git commits.

```
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com
```

### Configure CMake builds

Thrust uses [CMake](https://www.cmake.org) for its primary build system. To
configure, build, and test your checkout of Thrust:

```
# Create build directory:
mkdir build
cd build

# Configure -- use one of the following:
cmake ..                                 # Command line interface
cmake -DTHRUST_INCLUDE_CUB_CMAKE=ON ..   # Enables CUB development targets
ccmake ..                # ncurses GUI (Linux only)
cmake-gui                # Graphical UI, set source/build directories in the app

# Build:
cmake --build . -j <num jobs>   # invokes make (or ninja, etc)

# Run tests and examples:
ctest
```

See [CMake Options](../setup/cmake_options.md) for details on customizing the build. To
enable CUB tests and examples, set the `THRUST_INCLUDE_CUB_CMAKE` option to
`ON`. Additional CMake options for CUB are listed
[here](https://github.com/NVIDIA/cub/blob/main/CONTRIBUTING.md#cmake-options).

## Create a Development Branch

All work should be done in a development branch (also called a "topic branch")
and not directly in the `main` branch. This makes it easier to manage multiple
in-progress patches at once, and provides a descriptive label for your patch
as it passes through the review system.

To create a new branch based on the current `main`:

```
# Checkout local main branch:
cd /path/to/thrust/sources
git checkout main

# Sync local main branch with github:
git pull

# Create a new branch named `my_descriptive_branch_name` based on main:
git checkout -b my_descriptive_branch_name

# Verify that the branch has been created and is currently checked out:
git branch
```

Thrust branch names should follow a particular pattern:

- For new features, name the branch `feature/<name>`
- For bugfixes associated with a github issue, use `bug/github/<bug-description>-<bug-id>`
  - Internal nvidia and gitlab bugs should use `nvidia` or `gitlab` in place of
    `github`.

If you plan to work on CUB as part of your patch, repeat this process in the
`thrust/dependencies/cub` submodule.

## Local Development Loop

### Edit, Build, Test, Repeat

Once the topic branch is created, you're all set to start working on Thrust
code. Make some changes, then build and test them:

```
# Implement changes:
cd /path/to/thrust/sources
emacs thrust/some_file.h # or whatever editor you prefer

# Create / update a unit test for your changes:
emacs testing/some_test.cu

# Check that everything builds and tests pass:
cd /path/to/thrust/build/directory
cmake --build . -j <num jobs>
ctest
```

### Creating a Commit

Once you're satisfied with your patch, commit your changes:

#### Thrust-only Changes

```
# Manually add changed files and create a commit:
cd /path/to/thrust
git add thrust/some_file.h
git add testing/some_test.cu
git commit

# Or, if possible, use git-gui to review your changes while building your patch:
git gui
```

#### Thrust and CUB Changes

```
# Create CUB patch first:
cd /path/to/thrust/dependencies/cub
# Manually add changed files and create a commit:
git add cub/some_file.cuh
git commit

# Create Thrust patch, including submodule update:
cd /path/to/thrust/
git add dependencies/cub # Updates submodule info
git add thrust/some_file.h
git add testing/some_test.cu
git commit

# Or, if possible, use git-gui to review your changes while building your patch:
cd /path/to/thrust/dependencies/cub
git gui
cd /path/to/thrust
git gui # Include dependencies/cub as part of your commit

```

#### Writing a Commit Message

Your commit message will communicate the purpose and rationale behind your
patch to other developers, and will be used to populate the initial description
of your Github pull request.

When writing a commit message, the following standard format should be used,
since tools in the git ecosystem are designed to parse this correctly:

```
First line of commit message is a short summary (<80 char)
<Second line left blank>
Detailed description of change begins on third line. This portion can
span multiple lines, try to manually wrap them at something reasonable.

Blank lines can be used to separate multiple paragraphs in the description.

If your patch is associated with another pull request or issue in the main
Thrust repository, you should reference it with a `#` symbol, e.g.
#1023 for issue 1023.

For issues / pull requests in a different github repo, reference them using
the full syntax, e.g. NVIDIA/cub#4 for issue 4 in the NVIDIA/cub repo.

Markdown is recommended for formatting more detailed messages, as these will
be nicely rendered on Github, etc.
```

## Push Development Branch to your Fork

Once you've committed your changes to a local development branch, it's time to
push them to your fork:

```
cd /path/to/thrust/checkout
git checkout my_descriptive_branch_name # if not already checked out
git push --set-upstream github-fork my_descriptive_branch_name
```

`--set-upstream github-fork` tells git that future pushes/pulls on this branch
should target your `github-fork` remote by default.

If have CUB changes to commit as part of your patch, repeat this process in the
`thrust/dependencies/cub` submodule.

## Create Pull Request

To create a pull request for your freshly pushed branch, open your github fork
in a browser by going to `https://www.github.com/<GITHUB_USERNAME>/thrust`. A
prompt may automatically appear asking you to create a pull request if you've
recently pushed a branch.

If there's no prompt, go to "Code" > "Branches" and click the appropriate
"New pull request" button for your branch.

If you would like a specific developer to review your patch, feel free to
request them as a reviewer at this time.

The Thrust team will review your patch, test it on NVIDIA's internal CI, and
provide feedback.


If have CUB changes to commit as part of your patch, repeat this process with
your CUB branch and fork.

## Address Feedback and Update Pull Request

If the reviewers request changes to your patch, use the following process to
update the pull request:

```
# Make changes:
cd /path/to/thrust/sources
git checkout my_descriptive_branch_name
emacs thrust/some_file.h
emacs testing/some_test.cu

# Build + test
cd /path/to/thrust/build/directory
cmake --build . -j <num jobs>
ctest

# Amend commit:
cd /path/to/thrust/sources
git add thrust/some_file.h
git add testing/some_test.cu
git commit --amend
# Or
git gui # Check the "Amend Last Commit" box

# Update the branch on your fork:
git push -f
```

At this point, the pull request should show your recent changes.

If have CUB changes to commit as part of your patch, repeat this process in the
`thrust/dependencies/cub` submodule, and be sure to include any CUB submodule
updates as part of your commit.

## When Your PR is Approved

Once your pull request is approved by the Thrust team, no further action is
needed from you. We will handle integrating it since we must coordinate changes
to `main` with NVIDIA's internal perforce repository.

