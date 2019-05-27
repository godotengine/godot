#!/usr/bin/env python
# Copyright (c) 2017 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates language headers from a JSON grammar file"""

from __future__ import print_function

import errno
import json
import os.path
import re


def make_path_to_file(f):
    """Makes all ancestor directories to the given file, if they
    don't yet exist.

    Arguments:
        f: The file whose ancestor directories are to be created.
    """
    dir = os.path.dirname(os.path.abspath(f))
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else:
            raise

class ExtInstGrammar:
    """The grammar for an extended instruction set"""

    def __init__(self, name, copyright, instructions, operand_kinds, version = None, revision = None):
       self.name = name
       self.copyright = copyright
       self.instructions = instructions
       self.operand_kinds = operand_kinds
       self.version = version
       self.revision = revision


class LangGenerator:
    """A language-specific generator"""

    def __init__(self):
        self.upper_case_initial = re.compile('^[A-Z]')
        pass

    def comment_prefix(self):
        return ""

    def namespace_prefix(self):
        return ""

    def uses_guards(self):
        return False

    def cpp_guard_preamble(self):
        return ""

    def cpp_guard_postamble(self):
        return ""

    def enum_value(self, prefix, name, value):
        if self.upper_case_initial.match(name):
            use_name = name
        else:
            use_name = '_' + name

        return "    {}{} = {},".format(prefix, use_name, value)

    def generate(self, grammar):
        """Returns a string that is the language-specific header for the given grammar"""

        parts = []
        if grammar.copyright:
            parts.extend(["{}{}".format(self.comment_prefix(), f) for f in grammar.copyright])
        parts.append('')

        guard = 'SPIRV_EXTINST_{}_H_'.format(grammar.name)
        if self.uses_guards:
            parts.append('#ifndef {}'.format(guard))
            parts.append('#define {}'.format(guard))
        parts.append('')

        parts.append(self.cpp_guard_preamble())

        if grammar.version:
            parts.append(self.const_definition(grammar.name, 'Version', grammar.version))

        if grammar.revision is not None:
            parts.append(self.const_definition(grammar.name, 'Revision', grammar.revision))

        parts.append('')

        if grammar.instructions:
            parts.append(self.enum_prefix(grammar.name, 'Instructions'))
            for inst in grammar.instructions:
                parts.append(self.enum_value(grammar.name, inst['opname'], inst['opcode']))
            parts.append(self.enum_end(grammar.name, 'Instructions'))
            parts.append('')

        if grammar.operand_kinds:
            for kind in grammar.operand_kinds:
                parts.append(self.enum_prefix(grammar.name, kind['kind']))
                for e in kind['enumerants']:
                    parts.append(self.enum_value(grammar.name, e['enumerant'], e['value']))
                parts.append(self.enum_end(grammar.name, kind['kind']))
            parts.append('')

        parts.append(self.cpp_guard_postamble())

        if self.uses_guards:
            parts.append('#endif // {}'.format(guard))

        return '\n'.join(parts)


class CLikeGenerator(LangGenerator):
    def uses_guards(self):
        return True

    def comment_prefix(self):
        return "// "

    def const_definition(self, prefix, var, value):
        # Use an anonymous enum.  Don't use a static const int variable because
        # that can bloat binary size.
        return 'enum {0} {1}{2} = {3}, {1}{2}_BitWidthPadding = 0x7fffffff {4};'.format(
               '{', prefix, var, value, '}')

    def enum_prefix(self, prefix, name):
        return 'enum {}{} {}'.format(prefix, name, '{')

    def enum_end(self, prefix, enum):
        return '    {}{}Max = 0x7ffffff\n{};\n'.format(prefix, enum, '}')

    def cpp_guard_preamble(self):
        return '#ifdef __cplusplus\nextern "C" {\n#endif\n'

    def cpp_guard_postamble(self):
        return '#ifdef __cplusplus\n}\n#endif\n'


class CGenerator(CLikeGenerator):
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate language headers from a JSON grammar')

    parser.add_argument('--extinst-name',
                        type=str, required=True,
                        help='The name to use in tokens')
    parser.add_argument('--extinst-grammar', metavar='<path>',
                        type=str, required=True,
                        help='input JSON grammar file for extended instruction set')
    parser.add_argument('--extinst-output-base', metavar='<path>',
                        type=str, required=True,
                        help='Basename of the language-specific output file.')
    args = parser.parse_args()

    with open(args.extinst_grammar) as json_file:
        grammar_json = json.loads(json_file.read())
        grammar = ExtInstGrammar(name = args.extinst_name,
                                 copyright = grammar_json['copyright'],
                                 instructions = grammar_json['instructions'],
                                 operand_kinds = grammar_json['operand_kinds'],
                                 version = grammar_json['version'],
                                 revision = grammar_json['revision'])
        make_path_to_file(args.extinst_output_base)
        print(CGenerator().generate(grammar), file=open(args.extinst_output_base + '.h', 'w'))


if __name__ == '__main__':
    main()
