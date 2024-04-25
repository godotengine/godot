# Contributing to MaterialX

Thank you for your interest in contributing to MaterialX!  This document explains our contribution process and procedures.

## Community and Discussion

There are two primary ways to connect with the MaterialX community:

* The MaterialX channel of the [Academy Software Foundation Slack](http://academysoftwarefdn.slack.com/).  This platform is appropriate for general questions, feature requests, and discussion of the MaterialX project as a whole.  You can request an invitation to join the Academy Software Foundation Slack at https://slack.aswf.io/.
* The [Issues](https://github.com/AcademySoftwareFoundation/MaterialX/issues) panel of the MaterialX GitHub, which is used to report and discuss bugs and build issues.

## Contributor License Agreements

To contribute to MaterialX, you must sign a Contributor License Agreement through the *EasyCLA* system, which is integrated with GitHub as a pull request check.

Prior to submitting a pull request, you can sign the form through [this link](https://contributor.easycla.lfx.linuxfoundation.org/#/cla/project/68fa91fe-51fe-41ac-a21d-e0a0bf688a53/user/564e571e-12d7-4857-abd4-898939accdd7).  If you submit a pull request before the form is signed, the EasyCLA check will fail with a red *NOT COVERED* message, and you'll have another opportunity to sign the form through the provided link.

* If you are an individual writing the code on your own time and you're sure you are the sole owner of any intellectual property you contribute, you can sign the CLA as an Individual Contributor.
* If you are writing the code as part of your job, or if your employer retains ownership to intellectual property you create, then your company's legal affairs representatives should sign a Corporate Contributor License Agreement.  If your company already has a signed CCLA on file, ask your local CLA manager to add you to your company's approved list.

The MaterialX CLAs are the standard forms used by Linux Foundation projects and [recommended by the ASWF TAC](https://github.com/AcademySoftwareFoundation/tac/blob/main/process/contributing.md#contributor-license-agreement-cla).

## Coding Conventions

The coding style of the MaterialX project is defined by a [clang-format](.clang-format) file in the repository, which is supported by Clang versions 13 and newer.

When adding new source files to the repository, use the provided clang-format file to automatically align the code to MaterialX conventions.  When modifying existing code, follow the surrounding formatting conventions so that new or modified code blends in with the current code.

## Unit Tests

Each MaterialX module has a companion folder within the [MaterialXTest](source/MaterialXTest) module, containing a set of unit tests that validate its functionality.

When contributing new code to MaterialX, make sure to include appropriate unit tests in MaterialXTest to validate the expected behavior of the new code.
