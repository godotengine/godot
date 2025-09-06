#!/usr/bin/env python3

#
# extractFeaturesFromReleaseNotes.py
#
# Read the release notes - docs/release-notes.md - and generate text
# for pasting in to individual documentation pages, to indicate which
# versions recent features were released in.
#
# Using the output of the file is easier than manually constructing
# the text to paste in to documentation pages.
#
# One way to use this:
# - run this script, saving the output to some temporary file
# - diff this output with the actual release notes page
# - the differences are Markdown text that can be pasted in to the
#   appropriate documentation pages in the docs/ directory.
# - each release also has a github link to show which documentation files
#   were changed in it.
#   This can be helpful to see which documentation pages
#   to add the 'Introduced in Catch ...' snippets to the relevant pages.
#

import re


def create_introduced_in_text(version, bug_number = None):
    """Generate text to paste in to documentation file"""
    if bug_number:
        return '> [Introduced](https://github.com/catchorg/Catch2/issues/%s) in Catch %s.' % (bug_number, version)
    else:
        # Use this text for changes that don't have issue numbers
        return '> Introduced in Catch %s.' % version


def link_to_changes_in_release(release, releases):
    """
    Markdown text for a hyperlink showing all edits in a release, or empty string

    :param release: A release version, as a string 
    :param releases: A container of releases, in descending order - newest to oldest
    :return: Markdown text for a hyperlink showing the differences between the give release and the prior one,
             or empty string, if the previous release is not known
    """

    if release == releases[-1]:
        # This is the earliest release we know about
        return ''
    index = releases.index(release)
    previous_release = releases[index + 1]
    return '\n[Changes in %s](https://github.com/catchorg/Catch2/compare/v%s...v%s)' % (release, previous_release, release)


def write_recent_release_notes_with_introduced_text():
    current_version = None
    release_toc_regex = r'\[(\d.\d.\d)\]\(#\d+\)<br>'
    issue_number_regex = r'#[0-9]+'
    releases = []
    with open('../docs/release-notes.md') as release_notes:
        for line in release_notes:
            line = line[:-1]
            print(line)

            # Extract version number from table of contents
            match = re.search(release_toc_regex, line)
            if match:
                release_name = match.group(1)
                releases.append(release_name)

            if line.startswith('## '):
                # It's a section with version number
                current_version = line.replace('## ', '')

                # We decided not to add released-date info for older versions
                if current_version == 'Older versions':
                    break

                print(create_introduced_in_text(current_version))
                print(link_to_changes_in_release(current_version, releases))

            # Not yet found a version number, so to avoid picking up hyperlinks to
            # version numbers in the index, keep going
            if not current_version:
                continue

            for bug_link in re.findall(issue_number_regex, line):
                bug_number = bug_link.replace('#', '')
                print(create_introduced_in_text(current_version, bug_number))


if __name__ == '__main__':
    write_recent_release_notes_with_introduced_text()
