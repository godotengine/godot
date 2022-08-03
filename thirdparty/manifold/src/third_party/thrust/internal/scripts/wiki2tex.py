'''
Convert Google Code .wiki files into .tex formatted files.

Output is designed to be included within a larger TeX project, it is
not standalone.

'''

import sys
import re
import codecs

print(sys.argv)

'''
A "rule" is a begin tag, an end tag, and how to reformat the inner text
(function)
'''

def encase(pre, post, strip=False):
    """Return a function that prepends pre and postpends post"""
    def f(txt):
        if strip:
            return pre + txt.strip() + post
        else:
            return pre + txt + post
    return f

def constant(text):
    def f(txt):
        return text
    return f

def encase_with_rules(pre, post, rules, strip=False):
    def f(txt):
        if strip:
            return pre + apply_rules(txt, rules).strip() + post
        else:
            return pre + apply_rules(txt, rules) + post
    return f

def encase_escape_underscore(pre, post):
    def f(txt):
        txt = sub(r'_', r'\_', txt)
        return pre + txt + post
    return f

def sub(pat, repl, txt):
    """Substitute in repl for pat in txt, txt can be multiple lines"""
    return re.compile(pat, re.MULTILINE).sub(repl, txt)

def process_list(rules):
    def f(txt):
        txt = '  *' + txt # was removed to match begin tag of list
        res = '\\begin{itemize}\n'
        for ln in txt.split('\n'):
            # Convert "  *" to "\item "
            ln = sub(r'^  \*', r'\\item ', ln)
            res += apply_rules(ln, rules) + '\n'
        res += '\\end{itemize}\n'
        return res
    return f

def process_link(rules):
    def f(txt):
        lst = txt.split(' ')
        lnk = lst[0]
        desc = apply_rules(' '.join(lst[1:]), rules)
        if lnk[:7] == 'http://':
            desc = apply_rules(' '.join(lst[1:]), rules)
            return r'\href{' + lnk + r'}{' + desc + r'}'
        if len(lst) > 1:
            return r'\href{}{' + desc + r'}'
        return r'\href{}{' + lnk + r'}'
    return f

# Some rules can be used inside some other rules (backticks in section names)

link_rules = [
    ['_', '', constant(r'\_')],
]

section_rules = [
    ['`', '`', encase_escape_underscore(r'\texttt{', r'}')],
]

item_rules = [
    ['`', '`', encase(r'\verb|', r'|')],
    ['[', ']', process_link(link_rules)],
]

# Main rules for Latex formatting

rules = [
    ['{{{', '}}}', encase(r'\begin{lstlisting}[language=c++]', r'\end{lstlisting}')],
    ['[', ']', process_link(link_rules)],
    ['  *', '\n\n', process_list(item_rules)],
    ['"', '"', encase("``", "''")],
    ['`', '`', encase(r'\verb|', r'|')],
    ['*', '*', encase(r'\emph{', r'}')],
    ['_', '_', encase(r'\emph{', r'}')],
    ['==', '==', encase_with_rules(r'\section{', r'}', section_rules, True)],
    ['=', '=', encase_with_rules(r'\chapter{', r'}', section_rules, True)],
    ['(e.g. f(x) -> y and f(x,y) -> ', 'z)', constant(r'(e.g. $f(x)\to y$ and $f(x,y)\to z$)')],
]

def match_rules(txt, rules):
    """Find rule that first matches in txt"""
    # Find first begin tag
    first_begin_loc = 10e100
    matching_rule = None
    for rule in rules:
        begin_tag, end_tag, func = rule
        loc = txt.find(begin_tag)
        if loc > -1 and loc < first_begin_loc:
            first_begin_loc = loc
            matching_rule = rule
    return (matching_rule, first_begin_loc)

def apply_rules(txt, rules):
    """Apply set of rules to give txt, return transformed version of txt"""
    matching_rule, first_begin_loc = match_rules(txt, rules)
    if matching_rule is None:
        return txt
    begin_tag, end_tag, func = matching_rule
    end_loc = txt.find(end_tag, first_begin_loc + 1)
    if end_loc == -1:
        sys.exit('Could not find end tag {0} after position {1}'.format(end_tag, first_begin_loc + 1))
    inner_txt = txt[first_begin_loc + len(begin_tag) : end_loc]
    # Copy characters up until begin tag
    # Then have output of rule function on inner text
    new_txt_start = txt[:first_begin_loc] + func(inner_txt)
    # Follow with the remaining processed text
    remaining_txt = txt[end_loc + len(end_tag):]
    return new_txt_start + apply_rules(remaining_txt, rules)

def split_sections(contents):
    """Given one string of all file contents, return list of sections
    
    Return format is list of pairs, each pair has section title
    and list of lines.  Result is ordered as the original input.

    """
    res = []
    cur_section = ''
    section = []
    for ln in contents.split('\n'):
        if len(ln) > 0 and ln[0] == '=':
            # remove = formatting from line
            section_title = sub(r'^\=+ (.*) \=+', r'\1', ln)
            res.append((cur_section, section))
            cur_section = section_title
            section = [ln]
        else:
            section.append(ln)
    res.append((cur_section, section))
    return res

def filter_sections(splitinput, removelst):
    """Take split input and remove sections in removelst"""
    res = []
    for sectname, sectcontents in splitinput:
        if sectname in removelst:
            pass
        else:
            res.extend(sectcontents)
    # convert to single string for output
    return '\n'.join(res)


def main():
    infile = codecs.open(sys.argv[1], encoding='utf-8')
    outfile = codecs.open(sys.argv[2], mode='w', encoding='utf-8')
    
    contents = infile.read()
    
    # Remove first three lines
    contents = '\n'.join(contents.split('\n')[3:])
    
    # Split sections and filter out some of them
    sections = split_sections(contents)
    contents = filter_sections(sections, ['Introduction', 'Prerequisites', 'Simple Example'])
    
    # Convert to latex format
    contents = apply_rules(contents, rules)
    
    infile.close()
    outfile.write(contents)
    outfile.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
