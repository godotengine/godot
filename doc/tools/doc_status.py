#!/usr/bin/env python3

import sys
import re
import math
import platform
import xml.etree.ElementTree as ET

################################################################################
#                                    Config                                    #
################################################################################

flags = {
    'c': platform.platform() != 'Windows',  # Disable by default on windows, since we use ANSI escape codes
    'b': False,
    'g': False,
    's': False,
    'u': False,
    'h': False,
    'p': False,
    'o': True,
    'i': False,
    'a': True,
}
flag_descriptions = {
    'c': 'Toggle colors when outputting.',
    'b': 'Toggle showing only not fully described classes.',
    'g': 'Toggle showing only completed classes.',
    's': 'Toggle showing comments about the status.',
    'u': 'Toggle URLs to docs.',
    'h': 'Show help and exit.',
    'p': 'Toggle showing percentage as well as counts.',
    'o': 'Toggle overall column.',
    'i': 'Toggle collapse of class items columns.',
    'a': 'Toggle showing all items.',
}
long_flags = {
    'colors': 'c',
    'use-colors': 'c',

    'bad': 'b',
    'only-bad': 'b',

    'good': 'g',
    'only-good': 'g',

    'comments': 's',
    'status': 's',

    'urls': 'u',
    'gen-url': 'u',

    'help': 'h',

    'percent': 'p',
    'use-percentages': 'p',

    'overall': 'o',
    'use-overall': 'o',

    'items': 'i',
    'collapse': 'i',

    'all': 'a',
}
table_columns = ['name', 'brief_description', 'description', 'methods', 'constants', 'members', 'signals']
table_column_names = ['Name', 'Brief Desc.', 'Desc.', 'Methods', 'Constants', 'Members', 'Signals']
colors = {
    'name': [36],  # cyan
    'part_big_problem': [4, 31],  # underline, red
    'part_problem': [31],  # red
    'part_mostly_good': [33],  # yellow
    'part_good': [32],  # green
    'url': [4, 34],  # underline, blue
    'section': [1, 4],  # bold, underline
    'state_off': [36],  # cyan
    'state_on': [1, 35],  # bold, magenta/plum
}
overall_progress_description_weigth = 10


################################################################################
#                                    Utils                                     #
################################################################################

def validate_tag(elem, tag):
    if elem.tag != tag:
        print('Tag mismatch, expected "' + tag + '", got ' + elem.tag)
        sys.exit(255)


def color(color, string):
    if flags['c']:
        color_format = ''
        for code in colors[color]:
            color_format += '\033[' + str(code) + 'm'
        return color_format + string + '\033[0m'
    else:
        return string

ansi_escape = re.compile(r'\x1b[^m]*m')


def nonescape_len(s):
    return len(ansi_escape.sub('', s))


################################################################################
#                                   Classes                                    #
################################################################################

class ClassStatusProgress:

    def __init__(self, described=0, total=0):
        self.described = described
        self.total = total

    def __add__(self, other):
        return ClassStatusProgress(self.described + other.described, self.total + other.total)

    def increment(self, described):
        if described:
            self.described += 1
        self.total += 1

    def is_ok(self):
        return self.described >= self.total

    def to_configured_colored_string(self):
        if flags['p']:
            return self.to_colored_string('{percent}% ({has}/{total})', '{pad_percent}{pad_described}{s}{pad_total}')
        else:
            return self.to_colored_string()

    def to_colored_string(self, format='{has}/{total}', pad_format='{pad_described}{s}{pad_total}'):
        ratio = self.described / self.total if self.total != 0 else 1
        percent = round(100 * ratio)
        s = format.format(has=str(self.described), total=str(self.total), percent=str(percent))
        if self.described >= self.total:
            s = color('part_good', s)
        elif self.described >= self.total / 4 * 3:
            s = color('part_mostly_good', s)
        elif self.described > 0:
            s = color('part_problem', s)
        else:
            s = color('part_big_problem', s)
        pad_size = max(len(str(self.described)), len(str(self.total)))
        pad_described = ''.ljust(pad_size - len(str(self.described)))
        pad_percent = ''.ljust(3 - len(str(percent)))
        pad_total = ''.ljust(pad_size - len(str(self.total)))
        return pad_format.format(pad_described=pad_described, pad_total=pad_total, pad_percent=pad_percent, s=s)


class ClassStatus:

    def __init__(self, name=''):
        self.name = name
        self.has_brief_description = True
        self.has_description = True
        self.progresses = {
            'methods': ClassStatusProgress(),
            'constants': ClassStatusProgress(),
            'members': ClassStatusProgress(),
            'signals': ClassStatusProgress()
        }

    def __add__(self, other):
        new_status = ClassStatus()
        new_status.name = self.name
        new_status.has_brief_description = self.has_brief_description and other.has_brief_description
        new_status.has_description = self.has_description and other.has_description
        for k in self.progresses:
            new_status.progresses[k] = self.progresses[k] + other.progresses[k]
        return new_status

    def is_ok(self):
        ok = True
        ok = ok and self.has_brief_description
        ok = ok and self.has_description
        for k in self.progresses:
            ok = ok and self.progresses[k].is_ok()
        return ok

    def make_output(self):
        output = {}
        output['name'] = color('name', self.name)

        ok_string = color('part_good', 'OK')
        missing_string = color('part_big_problem', 'MISSING')

        output['brief_description'] = ok_string if self.has_brief_description else missing_string
        output['description'] = ok_string if self.has_description else missing_string

        description_progress = ClassStatusProgress(
            (self.has_brief_description + self.has_description) * overall_progress_description_weigth,
            2 * overall_progress_description_weigth
        )
        items_progress = ClassStatusProgress()

        for k in ['methods', 'constants', 'members', 'signals']:
            items_progress += self.progresses[k]
            output[k] = self.progresses[k].to_configured_colored_string()

        output['items'] = items_progress.to_configured_colored_string()

        output['overall'] = (description_progress + items_progress).to_colored_string('{percent}%', '{pad_percent}{s}')

        if self.name.startswith('Total'):
            output['url'] = color('url', 'http://docs.godotengine.org/en/latest/classes/')
            if flags['s']:
                output['comment'] = color('part_good', 'ALL OK')
        else:
            output['url'] = color('url', 'http://docs.godotengine.org/en/latest/classes/class_{name}.html'.format(name=self.name.lower()))

            if flags['s'] and not flags['g'] and self.is_ok():
                output['comment'] = color('part_good', 'ALL OK')

        return output

    def generate_for_class(c):
        status = ClassStatus()
        status.name = c.attrib['name']
        for tag in list(c):

            if tag.tag == 'brief_description':
                status.has_brief_description = len(tag.text.strip()) > 0

            elif tag.tag == 'description':
                status.has_description = len(tag.text.strip()) > 0

            elif tag.tag in ['methods', 'signals']:
                for sub_tag in list(tag):
                    descr = sub_tag.find('description')
                    status.progresses[tag.tag].increment(len(descr.text.strip()) > 0)

            elif tag.tag in ['constants', 'members']:
                for sub_tag in list(tag):
                    status.progresses[tag.tag].increment(len(sub_tag.text.strip()) > 0)

            elif tag.tag in ['theme_items']:
                pass  # Ignore those tags, since they seem to lack description at all

            else:
                print(tag.tag, tag.attrib)

        return status


################################################################################
#                                  Arguments                                   #
################################################################################

input_file_list = []
input_class_list = []

for arg in sys.argv[1:]:
    if arg.startswith('--'):
        flags[long_flags[arg[2:]]] = not flags[long_flags[arg[2:]]]
    elif arg.startswith('-'):
        for f in arg[1:]:
            flags[f] = not flags[f]
    elif arg.endswith('.xml'):
        input_file_list.append(arg)
    else:
        input_class_list.append(arg)

if flags['i']:
    for r in ['methods', 'constants', 'members', 'signals']:
        index = table_columns.index(r)
        del table_column_names[index]
        del table_columns[index]
    table_column_names.append('Items')
    table_columns.append('items')

if flags['o'] == (not flags['i']):
    table_column_names.append('Overall')
    table_columns.append('overall')

if flags['u']:
    table_column_names.append('Docs URL')
    table_columns.append('url')


################################################################################
#                                     Help                                     #
################################################################################

if len(input_file_list) < 1 or flags['h']:
    if not flags['h']:
        print(color('section', 'Invalid usage') + ': At least one classes.xml file is required')
    print(color('section', 'Usage') + ': doc_status.py [flags] <classes.xml> [class names]')
    print('\t< and > signify required parameters, while [ and ] signify optional parameters.')
    print('\tNote that you can give more than one classes file, in which case they will be merged on-the-fly.')
    print(color('section', 'Available flags') + ':')
    possible_synonym_list = list(long_flags)
    possible_synonym_list.sort()
    flag_list = list(flags)
    flag_list.sort()
    for flag in flag_list:
        synonyms = [color('name', '-' + flag)]
        for synonym in possible_synonym_list:
            if long_flags[synonym] == flag:
                synonyms.append(color('name', '--' + synonym))

        print(('{synonyms} (Currently ' + color('state_' + ('on' if flags[flag] else 'off'), '{value}') + ')\n\t{description}').format(
            synonyms=', '.join(synonyms),
            value=('on' if flags[flag] else 'off'),
            description=flag_descriptions[flag]
        ))
    sys.exit(0)


################################################################################
#                               Parse class list                               #
################################################################################

class_names = []
classes = {}

for file in input_file_list:
    tree = ET.parse(file)
    doc = tree.getroot()

    if 'version' not in doc.attrib:
        print('Version missing from "doc"')
        sys.exit(255)

    version = doc.attrib['version']

    for c in list(doc):
        if c.attrib['name'] in class_names:
            continue
        class_names.append(c.attrib['name'])
        classes[c.attrib['name']] = c

class_names.sort()

if len(input_class_list) < 1:
    input_class_list = class_names


################################################################################
#                               Make output table                              #
################################################################################

table = [table_column_names]
table_row_chars = '+- '
table_column_chars = '|'

total_status = ClassStatus('Total')

for cn in input_class_list:
    if not cn in classes:
        print('Cannot find class ' + cn + '!')
        sys.exit(255)

    c = classes[cn]
    validate_tag(c, 'class')
    status = ClassStatus.generate_for_class(c)

    total_status = total_status + status

    if (flags['b'] and status.is_ok()) or (flags['g'] and not status.is_ok()) or (not flags['a']):
        continue

    out = status.make_output()
    row = []
    for column in table_columns:
        if column in out:
            row.append(out[column])
        else:
            row.append('')

    if 'comment' in out and out['comment'] != '':
        row.append(out['comment'])

    table.append(row)


################################################################################
#                              Print output table                              #
################################################################################

if len(table) == 1 and flags['a']:
    print(color('part_big_problem', 'No classes suitable for printing!'))
    sys.exit(0)

if len(table) > 2 or not flags['a']:
    total_status.name = 'Total = {0}'.format(len(table) - 1)
    out = total_status.make_output()
    row = []
    for column in table_columns:
        if column in out:
            row.append(out[column])
        else:
            row.append('')
    table.append(row)

table_column_sizes = []
for row in table:
    for cell_i, cell in enumerate(row):
        if cell_i >= len(table_column_sizes):
            table_column_sizes.append(0)

        table_column_sizes[cell_i] = max(nonescape_len(cell), table_column_sizes[cell_i])

divider_string = table_row_chars[0]
for cell_i in range(len(table[0])):
    divider_string += table_row_chars[1] * (table_column_sizes[cell_i] + 2) + table_row_chars[0]
print(divider_string)

for row_i, row in enumerate(table):
    row_string = table_column_chars
    for cell_i, cell in enumerate(row):
        padding_needed = table_column_sizes[cell_i] - nonescape_len(cell) + 2
        if cell_i == 0:
            row_string += table_row_chars[2] + cell + table_row_chars[2] * (padding_needed - 1)
        else:
            row_string += table_row_chars[2] * math.floor(padding_needed / 2) + cell + table_row_chars[2] * math.ceil((padding_needed / 2))
        row_string += table_column_chars

    print(row_string)

    if row_i == 0 or row_i == len(table) - 2:
        print(divider_string)

print(divider_string)

if total_status.is_ok() and not flags['g']:
    print('All listed classes are ' + color('part_good', 'OK') + '!')
