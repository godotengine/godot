#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import sys
import os
import re
import xml.etree.ElementTree as ET

input_list = []
cur_file = ""

# http(s)://docs.godotengine.org/<langcode>/<tag>/path/to/page.html(#fragment-tag)
godot_docs_pattern = re.compile('^http(?:s)?:\/\/docs\.godotengine\.org\/(?:[a-zA-Z0-9\.\-_]*)\/(?:[a-zA-Z0-9\.\-_]*)\/(.*)\.html(#.*)?$')

for arg in sys.argv[1:]:
    if arg.endswith(os.sep):
        arg = arg[:-1]
    input_list.append(arg)

if len(input_list) < 1:
    print('usage: makerst.py <path to folders> and/or <path to .xml files> (order of arguments irrelevant)')
    print('example: makerst.py "../../modules/" "../classes" path_to/some_class.xml')
    sys.exit(0)


def validate_tag(elem, tag):
    if elem.tag != tag:
        print("Tag mismatch, expected '" + tag + "', got " + elem.tag)
        sys.exit(255)


class_names = []
classes = {}


def ul_string(str, ul):
    str += "\n"
    for i in range(len(str) - 1):
        str += ul
    str += "\n"
    return str


def make_class_list(class_list, columns):
    f = codecs.open('class_list.rst', 'wb', 'utf-8')
    prev = 0
    col_max = len(class_list) / columns + 1
    print(('col max is ', col_max))
    col_count = 0
    row_count = 0
    last_initial = ''
    fit_columns = []

    for n in range(0, columns):
        fit_columns += [[]]

    indexers = []
    last_initial = ''

    idx = 0
    for n in class_list:
        col = idx / col_max
        if col >= columns:
            col = columns - 1
        fit_columns[col] += [n]
        idx += 1
        if n[:1] != last_initial:
            indexers += [n]
        last_initial = n[:1]

    row_max = 0
    f.write("\n")

    for n in range(0, columns):
        if len(fit_columns[n]) > row_max:
            row_max = len(fit_columns[n])

    f.write("| ")
    for n in range(0, columns):
        f.write(" | |")

    f.write("\n")
    f.write("+")
    for n in range(0, columns):
        f.write("--+-------+")
    f.write("\n")

    for r in range(0, row_max):
        s = '+ '
        for c in range(0, columns):
            if r >= len(fit_columns[c]):
                continue

            classname = fit_columns[c][r]
            initial = classname[0]
            if classname in indexers:
                s += '**' + initial + '** | '
            else:
                s += ' | '

            s += '[' + classname + '](class_' + classname.lower() + ') | '

        s += '\n'
        f.write(s)

    for n in range(0, columns):
        f.write("--+-------+")
    f.write("\n")

    f.close()


def rstize_text(text, cclass):
    # Linebreak + tabs in the XML should become two line breaks unless in a "codeblock"
    pos = 0
    while True:
        pos = text.find('\n', pos)
        if pos == -1:
            break

        pre_text = text[:pos]
        while text[pos + 1] == '\t':
            pos += 1
        post_text = text[pos + 1:]

        # Handle codeblocks
        if post_text.startswith("[codeblock]"):
            end_pos = post_text.find("[/codeblock]")
            if end_pos == -1:
                sys.exit("ERROR! [codeblock] without a closing tag!")

            code_text = post_text[len("[codeblock]"):end_pos]
            post_text = post_text[end_pos:]

            # Remove extraneous tabs
            code_pos = 0
            while True:
                code_pos = code_text.find('\n', code_pos)
                if code_pos == -1:
                    break

                to_skip = 0
                while code_pos + to_skip + 1 < len(code_text) and code_text[code_pos + to_skip + 1] == '\t':
                    to_skip += 1

                if len(code_text[code_pos + to_skip + 1:]) == 0:
                    code_text = code_text[:code_pos] + "\n"
                    code_pos += 1
                else:
                    code_text = code_text[:code_pos] + "\n    " + code_text[code_pos + to_skip + 1:]
                    code_pos += 5 - to_skip

            text = pre_text + "\n[codeblock]" + code_text + post_text
            pos += len("\n[codeblock]" + code_text)

        # Handle normal text
        else:
            text = pre_text + "\n\n" + post_text
            pos += 2

    next_brac_pos = text.find('[')

    # Escape \ character, otherwise it ends up as an escape character in rst
    pos = 0
    while True:
        pos = text.find('\\', pos, next_brac_pos)
        if pos == -1:
            break
        text = text[:pos] + "\\\\" + text[pos + 1:]
        pos += 2

    # Escape * character to avoid interpreting it as emphasis
    pos = 0
    while True:
        pos = text.find('*', pos, next_brac_pos)
        if pos == -1:
            break
        text = text[:pos] + "\*" + text[pos + 1:]
        pos += 2

    # Escape _ character at the end of a word to avoid interpreting it as an inline hyperlink
    pos = 0
    while True:
        pos = text.find('_', pos, next_brac_pos)
        if pos == -1:
            break
        if not text[pos + 1].isalnum():  # don't escape within a snake_case word
            text = text[:pos] + "\_" + text[pos + 1:]
            pos += 2
        else:
            pos += 1

    # Handle [tags]
    inside_code = False
    pos = 0
    while True:
        pos = text.find('[', pos)
        if pos == -1:
            break

        endq_pos = text.find(']', pos + 1)
        if endq_pos == -1:
            break

        pre_text = text[:pos]
        post_text = text[endq_pos + 1:]
        tag_text = text[pos + 1:endq_pos]

        escape_post = False

        if tag_text in class_names:
            tag_text = make_type(tag_text)
            escape_post = True
        else:  # command
            cmd = tag_text
            space_pos = tag_text.find(' ')
            if cmd == '/codeblock':
                tag_text = ''
                inside_code = False
                # Strip newline if the tag was alone on one
                if pre_text[-1] == '\n':
                    pre_text = pre_text[:-1]
            elif cmd == '/code':
                tag_text = '``'
                inside_code = False
                escape_post = True
            elif inside_code:
                tag_text = '[' + tag_text + ']'
            elif cmd.find('html') == 0:
                cmd = tag_text[:space_pos]
                param = tag_text[space_pos + 1:]
                tag_text = param
            elif cmd.find('method') == 0 or cmd.find('member') == 0 or cmd.find('signal') == 0:
                cmd = tag_text[:space_pos]
                param = tag_text[space_pos + 1:]

                if param.find('.') != -1:
                    ss = param.split('.')
                    if len(ss) > 2:
                        sys.exit("Bad reference: '" + param + "' in file: " + cur_file)
                    (class_param, method_param) = ss
                    tag_text = ':ref:`' + class_param + '.' + method_param + '<class_' + class_param + '_' + method_param + '>`'
                else:
                    tag_text = ':ref:`' + param + '<class_' + cclass + "_" + param + '>`'
                escape_post = True
            elif cmd.find('image=') == 0:
                tag_text = ""  # '![](' + cmd[6:] + ')'
            elif cmd.find('url=') == 0:
                tag_text = ':ref:`' + cmd[4:] + '<' + cmd[4:] + ">`"
            elif cmd == '/url':
                tag_text = ''
                escape_post = True
            elif cmd == 'center':
                tag_text = ''
            elif cmd == '/center':
                tag_text = ''
            elif cmd == 'codeblock':
                tag_text = '\n::\n'
                inside_code = True
            elif cmd == 'br':
                # Make a new paragraph instead of a linebreak, rst is not so linebreak friendly
                tag_text = '\n\n'
                # Strip potential leading spaces
                while post_text[0] == ' ':
                    post_text = post_text[1:]
            elif cmd == 'i' or cmd == '/i':
                tag_text = '*'
            elif cmd == 'b' or cmd == '/b':
                tag_text = '**'
            elif cmd == 'u' or cmd == '/u':
                tag_text = ''
            elif cmd == 'code':
                tag_text = '``'
                inside_code = True
            elif cmd.startswith('enum '):
                tag_text = make_enum(cmd[5:])
            else:
                tag_text = make_type(tag_text)
                escape_post = True

        # Properly escape things like `[Node]s`
        if escape_post and post_text and post_text[0].isalnum():  # not punctuation, escape
            post_text = '\ ' + post_text

        next_brac_pos = post_text.find('[', 0)
        iter_pos = 0
        while not inside_code:
            iter_pos = post_text.find('*', iter_pos, next_brac_pos)
            if iter_pos == -1:
                break
            post_text = post_text[:iter_pos] + "\*" + post_text[iter_pos + 1:]
            iter_pos += 2

        iter_pos = 0
        while not inside_code:
            iter_pos = post_text.find('_', iter_pos, next_brac_pos)
            if iter_pos == -1:
                break
            if not post_text[iter_pos + 1].isalnum():  # don't escape within a snake_case word
                post_text = post_text[:iter_pos] + "\_" + post_text[iter_pos + 1:]
                iter_pos += 2
            else:
                iter_pos += 1

        text = pre_text + tag_text + post_text
        pos = len(pre_text) + len(tag_text)

    return text


def format_table(f, pp):
    longest_t = 0
    longest_s = 0
    for s in pp:
        sl = len(s[0])
        if (sl > longest_s):
            longest_s = sl
        tl = len(s[1])
        if (tl > longest_t):
            longest_t = tl

    sep = "+"
    for i in range(longest_s + 2):
        sep += "-"
    sep += "+"
    for i in range(longest_t + 2):
        sep += "-"
    sep += "+\n"
    f.write(sep)
    for s in pp:
        rt = s[0]
        while (len(rt) < longest_s):
            rt += " "
        st = s[1]
        while (len(st) < longest_t):
            st += " "
        f.write("| " + rt + " | " + st + " |\n")
        f.write(sep)
    f.write('\n')


def make_type(t):
    global class_names
    if t in class_names:
        return ':ref:`' + t + '<class_' + t + '>`'
    return t


def make_enum(t):
    global class_names
    p = t.find(".")
    # Global enums such as Error are relative to @GlobalScope.
    if p >= 0:
        c = t[0:p]
        e = t[p + 1:]
        # Variant enums live in GlobalScope but still use periods.
        if c == "Variant":
            c = "@GlobalScope"
            e = "Variant." + e
    else:
        # Things in GlobalScope don't have a period.
        c = "@GlobalScope"
        e = t
    if c in class_names:
        return ':ref:`' + e + '<enum_' + c + '_' + e + '>`'
    return t


def make_method(
        f,
        cname,
        method_data,
        declare,
        event=False,
        pp=None
):
    if (declare or pp is None):
        t = '- '
    else:
        t = ""

    ret_type = 'void'
    args = list(method_data)
    mdata = {}
    mdata['argidx'] = []
    for a in args:
        if a.tag == 'return':
            idx = -1
        elif a.tag == 'argument':
            idx = int(a.attrib['index'])
        else:
            continue

        mdata['argidx'].append(idx)
        mdata[idx] = a

    if not event:
        if -1 in mdata['argidx']:
            if 'enum' in mdata[-1].attrib:
                t += make_enum(mdata[-1].attrib['enum'])
            else:
                t += make_type(mdata[-1].attrib['type'])
        else:
            t += 'void'
        t += ' '

    if declare or pp is None:

        s = '**' + method_data.attrib['name'] + '** '
    else:
        s = ':ref:`' + method_data.attrib['name'] + '<class_' + cname + "_" + method_data.attrib['name'] + '>` '

    s += '**(**'
    argfound = False
    for a in mdata['argidx']:
        arg = mdata[a]
        if a < 0:
            continue
        if a > 0:
            s += ', '
        else:
            s += ' '

        if 'enum' in arg.attrib:
            s += make_enum(arg.attrib['enum'])
        else:
            s += make_type(arg.attrib['type'])
        if 'name' in arg.attrib:
            s += ' ' + arg.attrib['name']
        else:
            s += ' arg' + str(a)

        if 'default' in arg.attrib:
            s += '=' + arg.attrib['default']

    s += ' **)**'

    if 'qualifiers' in method_data.attrib:
        s += ' ' + method_data.attrib['qualifiers']

    if (not declare):
        if (pp != None):
            pp.append((t, s))
        else:
            f.write("- " + t + " " + s + "\n")
    else:
        f.write(t + s + "\n")


def make_properties(
        f,
        cname,
        prop_data,
        description=False,
        pp=None
):
    t = ""
    if 'enum' in prop_data.attrib:
        t += make_enum(prop_data.attrib['enum'])
    else:
        t += make_type(prop_data.attrib['type'])

    if description:
        s = '**' + prop_data.attrib['name'] + '**'
        setget = []
        if 'setter' in prop_data.attrib and prop_data.attrib['setter'] != '' and not prop_data.attrib['setter'].startswith('_'):
            setget.append(("*Setter*", prop_data.attrib['setter'] + '(value)'))
        if 'getter' in prop_data.attrib and prop_data.attrib['getter'] != '' and not prop_data.attrib['getter'].startswith('_'):
            setget.append(('*Getter*', prop_data.attrib['getter'] + '()'))
    else:
        s = ':ref:`' + prop_data.attrib['name'] + '<class_' + cname + "_" + prop_data.attrib['name'] + '>`'

    if (pp != None):
        pp.append((t, s))
    elif description:
        f.write('- ' + t + ' ' + s + '\n\n')
        if len(setget) > 0:
            format_table(f, setget)


def make_heading(title, underline):
    return title + '\n' + underline * len(title) + "\n\n"


def make_rst_class(node):
    name = node.attrib['name']

    f = codecs.open("class_" + name.lower() + '.rst', 'wb', 'utf-8')

    # Warn contributors not to edit this file directly
    f.write(".. Generated automatically by doc/tools/makerst.py in Godot's source tree.\n")
    f.write(".. DO NOT EDIT THIS FILE, but the " + name + ".xml source instead.\n")
    f.write(".. The source is found in doc/classes or modules/<name>/doc_classes.\n\n")

    f.write(".. _class_" + name + ":\n\n")
    f.write(make_heading(name, '='))

    # Inheritance tree
    # Ascendents
    if 'inherits' in node.attrib:
        inh = node.attrib['inherits'].strip()
        f.write('**Inherits:** ')
        first = True
        while (inh in classes):
            if (not first):
                f.write(" **<** ")
            else:
                first = False

            f.write(make_type(inh))
            inode = classes[inh]
            if ('inherits' in inode.attrib):
                inh = inode.attrib['inherits'].strip()
            else:
                inh = None
        f.write("\n\n")

    # Descendents
    inherited = []
    for cn in classes:
        c = classes[cn]
        if 'inherits' in c.attrib:
            if (c.attrib['inherits'].strip() == name):
                inherited.append(c.attrib['name'])
    if (len(inherited)):
        f.write('**Inherited By:** ')
        for i in range(len(inherited)):
            if (i > 0):
                f.write(", ")
            f.write(make_type(inherited[i]))
        f.write("\n\n")

    # Category
    if 'category' in node.attrib:
        f.write('**Category:** ' + node.attrib['category'].strip() + "\n\n")

    # Brief description
    f.write(make_heading('Brief Description', '-'))
    briefd = node.find('brief_description')
    if briefd != None:
        f.write(rstize_text(briefd.text.strip(), name) + "\n\n")

    # Properties overview
    members = node.find('members')
    if members != None and len(list(members)) > 0:
        f.write(make_heading('Properties', '-'))
        ml = []
        for m in list(members):
            make_properties(f, name, m, False, ml)
        format_table(f, ml)

    # Methods overview
    methods = node.find('methods')
    if methods != None and len(list(methods)) > 0:
        f.write(make_heading('Methods', '-'))
        ml = []
        for m in list(methods):
            make_method(f, name, m, False, False, ml)
        format_table(f, ml)

    # Theme properties
    theme_items = node.find('theme_items')
    if theme_items != None and len(list(theme_items)) > 0:
        f.write(make_heading('Theme Properties', '-'))
        ml = []
        for m in list(theme_items):
            make_properties(f, name, m, False, ml)
        format_table(f, ml)

    # Signals
    events = node.find('signals')
    if events != None and len(list(events)) > 0:
        f.write(make_heading('Signals', '-'))
        for m in list(events):
            f.write(".. _class_" + name + "_" + m.attrib['name'] + ":\n\n")
            make_method(f, name, m, True, True)
            f.write('\n')
            d = m.find('description')
            if d is None or d.text.strip() == '':
                continue
            f.write(rstize_text(d.text.strip(), name))
            f.write("\n\n")

    # Constants and enums
    constants = node.find('constants')
    consts = []
    enum_names = set()
    enums = []
    if constants != None and len(list(constants)) > 0:
        for c in list(constants):
            if 'enum' in c.attrib:
                enum_names.add(c.attrib['enum'])
                enums.append(c)
            else:
                consts.append(c)

    # Enums
    if len(enum_names) > 0:
        f.write(make_heading('Enumerations', '-'))
        for e in enum_names:
            f.write(".. _enum_" + name + "_" + e + ":\n\n")
            f.write("enum **" + e + "**:\n\n")
            for c in enums:
                if c.attrib['enum'] != e:
                    continue
                s = '- '
                s += '**' + c.attrib['name'] + '**'
                if 'value' in c.attrib:
                    s += ' = **' + c.attrib['value'] + '**'
                if c.text.strip() != '':
                    s += ' --- ' + rstize_text(c.text.strip(), name)
                f.write(s + '\n\n')

    # Constants
    if len(consts) > 0:
        f.write(make_heading('Constants', '-'))
        for c in list(consts):
            s = '- '
            s += '**' + c.attrib['name'] + '**'
            if 'value' in c.attrib:
                s += ' = **' + c.attrib['value'] + '**'
            if c.text.strip() != '':
                s += ' --- ' + rstize_text(c.text.strip(), name)
            f.write(s + '\n\n')

    # Class description
    descr = node.find('description')
    if descr != None and descr.text.strip() != '':
        f.write(make_heading('Description', '-'))
        f.write(rstize_text(descr.text.strip(), name) + "\n\n")

    # Online tutorials
    global godot_docs_pattern
    tutorials = node.find('tutorials')
    if tutorials != None and len(tutorials) > 0:
        f.write(make_heading('Tutorials', '-'))
        for t in tutorials:
            link = t.text.strip()
            match = godot_docs_pattern.search(link);
            if match:
                groups = match.groups()
                if match.lastindex == 2:
                    # Doc reference with fragment identifier: emit direct link to section with reference to page, for example:
                    # `#calling-javascript-from-script in Exporting For Web`
                    f.write("- `" + groups[1] + " <../" + groups[0] + ".html" + groups[1] + ">`_ in :doc:`../" + groups[0] + "`\n\n")
                    # Commented out alternative: Instead just emit:
                    # `Subsection in Exporting For Web`
                    # f.write("- `Subsection <../" + groups[0] + ".html" + groups[1] + ">`_ in :doc:`../" + groups[0] + "`\n\n")
                elif match.lastindex == 1:
                    # Doc reference, for example:
                    # `Math`
                    f.write("- :doc:`../" + groups[0] + "`\n\n")
            else:
                # External link, for example:
                # `http://enet.bespin.org/usergroup0.html`
                f.write("- `" + link + " <" + link + ">`_\n\n")

    # Property descriptions
    members = node.find('members')
    if members != None and len(list(members)) > 0:
        f.write(make_heading('Property Descriptions', '-'))
        for m in list(members):
            f.write(".. _class_" + name + "_" + m.attrib['name'] + ":\n\n")
            make_properties(f, name, m, True)
            if m.text.strip() != '':
                f.write(rstize_text(m.text.strip(), name))
                f.write('\n\n')

    # Method descriptions
    methods = node.find('methods')
    if methods != None and len(list(methods)) > 0:
        f.write(make_heading('Method Descriptions', '-'))
        for m in list(methods):
            f.write(".. _class_" + name + "_" + m.attrib['name'] + ":\n\n")
            make_method(f, name, m, True)
            f.write('\n')
            d = m.find('description')
            if d is None or d.text.strip() == '':
                continue
            f.write(rstize_text(d.text.strip(), name))
            f.write("\n\n")


file_list = []

for path in input_list:
    if os.path.basename(path) == 'modules':
        for subdir, dirs, _ in os.walk(path):
            if 'doc_classes' in dirs:
                doc_dir = os.path.join(subdir, 'doc_classes')
                class_file_names = [f for f in os.listdir(doc_dir) if f.endswith('.xml')]
                file_list += [os.path.join(doc_dir, f) for f in class_file_names]
    elif not os.path.isfile(path):
        file_list += [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.xml')]
    elif os.path.isfile(path) and path.endswith('.xml'):
        file_list.append(path)

for cur_file in file_list:
    tree = ET.parse(cur_file)
    doc = tree.getroot()

    if 'version' not in doc.attrib:
        print("Version missing from 'doc'")
        sys.exit(255)

    version = doc.attrib['version']
    if doc.attrib['name'] in class_names:
        continue
    class_names.append(doc.attrib['name'])
    classes[doc.attrib['name']] = doc

class_names.sort()

# Don't make class list for Sphinx, :toctree: handles it
# make_class_list(class_names, 2)

for cn in class_names:
    c = classes[cn]
    make_rst_class(c)
