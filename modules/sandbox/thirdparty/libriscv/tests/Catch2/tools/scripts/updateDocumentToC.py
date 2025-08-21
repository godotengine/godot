#!/usr/bin/env python3

#
# updateDocumentToC.py
#
# Insert table of contents at top of Catch markdown documents.
#
# This script is distributed under the GNU General Public License v3.0
#
# It is based on markdown-toclify version 1.7.1 by Sebastian Raschka,
# https://github.com/rasbt/markdown-toclify
#

import argparse
import glob
import os
import re
import sys

from scriptCommon import catchPath

# Configuration:

minTocEntries = 4

headingExcludeDefault = [1,3,4,5]  # use level 2 headers for at default
headingExcludeRelease = [1,3,4,5]  # use level 1 headers for release-notes.md

documentsDefault = os.path.join(os.path.relpath(catchPath), 'docs/*.md')
releaseNotesName = 'release-notes.md'

contentTitle = '**Contents**'
contentLineNo = 4
contentLineNdx = contentLineNo - 1

# End configuration

VALIDS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-&'

def readLines(in_file):
    """Returns a list of lines from a input markdown file."""

    with open(in_file, 'r') as inf:
        in_contents = inf.read().split('\n')
    return in_contents

def removeLines(lines, remove=('[[back to top]', '<a class="mk-toclify"')):
    """Removes existing [back to top] links and <a id> tags."""

    if not remove:
        return lines[:]

    out = []
    for l in lines:
        if l.startswith(remove):
            continue
        out.append(l)
    return out

def removeToC(lines):
    """Removes existing table of contents starting at index contentLineNdx."""
    if not lines[contentLineNdx ].startswith(contentTitle):
        return lines[:]

    result_top = lines[:contentLineNdx]

    pos = contentLineNdx + 1
    while lines[pos].startswith('['):
        pos = pos + 1

    result_bottom = lines[pos + 1:]

    return result_top + result_bottom

def dashifyHeadline(line):
    """
    Takes a header line from a Markdown document and
    returns a tuple of the
        '#'-stripped version of the head line,
        a string version for <a id=''></a> anchor tags,
        and the level of the headline as integer.
    E.g.,
    >>> dashifyHeadline('### some header lvl3')
    ('Some header lvl3', 'some-header-lvl3', 3)

    """
    stripped_right = line.rstrip('#')
    stripped_both = stripped_right.lstrip('#')
    level = len(stripped_right) - len(stripped_both)
    stripped_wspace = stripped_both.strip()

    # GitHub's sluggification works in an interesting way
    # 1) '+', '/', '(', ')' and so on are just removed
    # 2) spaces are converted into '-' directly
    # 3) multiple -- are not collapsed

    dashified = ''
    for c in stripped_wspace:
        if c in VALIDS:
            dashified += c.lower()
        elif c.isspace():
            dashified += '-'
        else:
            # Unknown symbols are just removed
            continue

    return [stripped_wspace, dashified, level]

def tagAndCollect(lines, id_tag=True, back_links=False, exclude_h=None):
    """
    Gets headlines from the markdown document and creates anchor tags.

    Keyword arguments:
        lines: a list of sublists where every sublist
            represents a line from a Markdown document.
        id_tag: if true, creates inserts a the <a id> tags (not req. by GitHub)
        back_links: if true, adds "back to top" links below each headline
        exclude_h: header levels to exclude. E.g., [2, 3]
            excludes level 2 and 3 headings.

    Returns a tuple of 2 lists:
        1st list:
            A modified version of the input list where
            <a id="some-header"></a> anchor tags where inserted
            above the header lines (if github is False).

        2nd list:
            A list of 3-value sublists, where the first value
            represents the heading, the second value the string
            that was inserted assigned to the IDs in the anchor tags,
            and the third value is an integer that represents the headline level.
            E.g.,
            [['some header lvl3', 'some-header-lvl3', 3], ...]

    """
    out_contents = []
    headlines = []
    for l in lines:
        saw_headline = False

        orig_len = len(l)
        l_stripped = l.lstrip()

        if l_stripped.startswith(('# ', '## ', '### ', '#### ', '##### ', '###### ')):

            # comply with new markdown standards

            # not a headline if '#' not followed by whitespace '##no-header':
            if not l.lstrip('#').startswith(' '):
                continue
            # not a headline if more than 6 '#':
            if len(l) - len(l.lstrip('#')) > 6:
                continue
            # headers can be indented by at most 3 spaces:
            if orig_len - len(l_stripped) > 3:
                continue

            # ignore empty headers
            if not set(l) - {'#', ' '}:
                continue

            saw_headline = True
            dashified = dashifyHeadline(l)

            if not exclude_h or not dashified[-1] in exclude_h:
                if id_tag:
                    id_tag = '<a class="mk-toclify" id="%s"></a>'\
                              % (dashified[1])
                    out_contents.append(id_tag)
                headlines.append(dashified)

        out_contents.append(l)
        if back_links and saw_headline:
            out_contents.append('[[back to top](#table-of-contents)]')
    return out_contents, headlines

def positioningHeadlines(headlines):
    """
    Strips unnecessary whitespaces/tabs if first header is not left-aligned
    """
    left_just = False
    for row in headlines:
        if row[-1] == 1:
            left_just = True
            break
    if not left_just:
        for row in headlines:
            row[-1] -= 1
    return headlines

def createToc(headlines, hyperlink=True, top_link=False, no_toc_header=False):
    """
    Creates the table of contents from the headline list
    that was returned by the tagAndCollect function.

    Keyword Arguments:
        headlines: list of lists
            e.g., ['Some header lvl3', 'some-header-lvl3', 3]
        hyperlink: Creates hyperlinks in Markdown format if True,
            e.g., '- [Some header lvl1](#some-header-lvl1)'
        top_link: if True, add a id tag for linking the table
            of contents itself (for the back-to-top-links)
        no_toc_header: suppresses TOC header if True.

    Returns  a list of headlines for a table of contents
    in Markdown format,
    e.g., ['        - [Some header lvl3](#some-header-lvl3)', ...]

    """
    processed = []
    if not no_toc_header:
        if top_link:
            processed.append('<a class="mk-toclify" id="table-of-contents"></a>\n')
        processed.append(contentTitle + '<br>')

    for line in headlines:
        if hyperlink:
            item = '[%s](#%s)' % (line[0], line[1])
        else:
            item = '%s- %s' % ((line[2]-1)*'    ', line[0])
        processed.append(item + '<br>')
    processed.append('\n')
    return processed

def buildMarkdown(toc_headlines, body, spacer=0, placeholder=None):
    """
    Returns a string with the Markdown output contents incl.
    the table of contents.

    Keyword arguments:
        toc_headlines: lines for the table of contents
            as created by the createToc function.
        body: contents of the Markdown file including
            ID-anchor tags as returned by the
            tagAndCollect function.
        spacer: Adds vertical space after the table
            of contents. Height in pixels.
        placeholder: If a placeholder string is provided, the placeholder
            will be replaced by the TOC instead of inserting the TOC at
            the top of the document

    """
    if spacer:
        spacer_line = ['\n<div style="height:%spx;"></div>\n' % (spacer)]
        toc_markdown = "\n".join(toc_headlines + spacer_line)
    else:
        toc_markdown = "\n".join(toc_headlines)

    if placeholder:
        body_markdown = "\n".join(body)
        markdown = body_markdown.replace(placeholder, toc_markdown)
    else:
        body_markdown_p1 = "\n".join(body[:contentLineNdx ]) + '\n'
        body_markdown_p2 = "\n".join(body[ contentLineNdx:])
        markdown = body_markdown_p1 + toc_markdown + body_markdown_p2

    return markdown

def outputMarkdown(markdown_cont, output_file):
    """
    Writes to an output file if `outfile` is a valid path.

    """
    if output_file:
        with open(output_file, 'w') as out:
            out.write(markdown_cont)

def markdownToclify(
    input_file,
    output_file=None,
    min_toc_len=2,
    github=False,
    back_to_top=False,
    nolink=False,
    no_toc_header=False,
    spacer=0,
    placeholder=None,
    exclude_h=None):
    """ Function to add table of contents to markdown files.

    Parameters
    -----------
      input_file: str
        Path to the markdown input file.

      output_file: str (default: None)
        Path to the markdown output file.

      min_toc_len: int (default: 2)
        Minimum number of entries to create a table of contents for.

      github: bool (default: False)
        Uses GitHub TOC syntax if True.

      back_to_top: bool (default: False)
        Inserts back-to-top links below headings if True.

      nolink: bool (default: False)
        Creates the table of contents without internal links if True.

      no_toc_header: bool (default: False)
        Suppresses the Table of Contents header if True

      spacer: int (default: 0)
        Inserts horizontal space (in pixels) after the table of contents.

      placeholder: str (default: None)
        Inserts the TOC at the placeholder string instead
        of inserting the TOC at the top of the document.

      exclude_h: list (default None)
        Excludes header levels, e.g., if [2, 3], ignores header
        levels 2 and 3 in the TOC.

    Returns
    -----------
    changed: Boolean
      True if the file has been updated, False otherwise.

    """
    cleaned_contents = removeLines(
        removeToC(readLines(input_file)),
        remove=('[[back to top]', '<a class="mk-toclify"'))

    processed_contents, raw_headlines = tagAndCollect(
        cleaned_contents,
        id_tag=not github,
        back_links=back_to_top,
        exclude_h=exclude_h)

    # add table of contents?
    if len(raw_headlines) < min_toc_len:
        processed_headlines = []
    else:
        leftjustified_headlines = positioningHeadlines(raw_headlines)

        processed_headlines = createToc(
            leftjustified_headlines,
            hyperlink=not nolink,
            top_link=not nolink and not github,
            no_toc_header=no_toc_header)

    if nolink:
        processed_contents = cleaned_contents

    cont = buildMarkdown(
        toc_headlines=processed_headlines,
        body=processed_contents,
        spacer=spacer,
        placeholder=placeholder)

    if output_file:
        outputMarkdown(cont, output_file)

def isReleaseNotes(f):
    return os.path.basename(f) == releaseNotesName

def excludeHeadingsFor(f):
    return headingExcludeRelease if isReleaseNotes(f) else headingExcludeDefault

def updateSingleDocumentToC(input_file, min_toc_len, verbose=False):
    """Add or update table of contents in specified file. Return 1 if file changed, 0 otherwise."""
    if verbose :
        print( 'file: {}'.format(input_file))

    output_file = input_file + '.tmp'

    markdownToclify(
        input_file=input_file,
        output_file=output_file,
        min_toc_len=min_toc_len,
        github=True,
        back_to_top=False,
        nolink=False,
        no_toc_header=False,
        spacer=False,
        placeholder=False,
        exclude_h=excludeHeadingsFor(input_file))

    # prevent race-condition (Python 3.3):
    if sys.version_info >= (3, 3):
        os.replace(output_file, input_file)
    else:
        os.remove(input_file)
        os.rename(output_file, input_file)

    return 1

def updateDocumentToC(paths, min_toc_len, verbose):
    """Add or update table of contents to specified paths. Return number of changed files"""
    n = 0
    for g in paths:
        for f in glob.glob(g):
            if os.path.isfile(f):
                n = n + updateSingleDocumentToC(input_file=f, min_toc_len=min_toc_len, verbose=verbose)
    return n

def updateDocumentToCMain():
    """Add or update table of contents to specified paths."""

    parser = argparse.ArgumentParser(
        description='Add or update table of contents in markdown documents.',
        epilog="""""",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        'Input',
        metavar='file',
        type=str,
        nargs=argparse.REMAINDER,
        help='files to process, at default: docs/*.md')

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='report the name of the file being processed')

    parser.add_argument(
        '--min-toc-entries',
        dest='minTocEntries',
        default=minTocEntries,
        type=int,
        metavar='N',
        help='the minimum number of entries to create a table of contents for [{default}]'.format(default=minTocEntries))

    parser.add_argument(
        '--remove-toc',
        action='store_const',
        dest='minTocEntries',
        const=99,
        help='remove all tables of contents')

    args = parser.parse_args()

    paths = args.Input if args.Input else [documentsDefault]

    changedFiles = updateDocumentToC(paths=paths, min_toc_len=args.minTocEntries, verbose=args.verbose)

    if changedFiles > 0:
        print( "Processed table of contents in " + str(changedFiles) + " file(s)" )
    else:
        print( "No table of contents added or updated" )

if __name__ == '__main__':
    updateDocumentToCMain()

# end of file
