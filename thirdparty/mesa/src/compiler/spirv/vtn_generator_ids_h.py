COPYRIGHT = """\
/*
 * Copyright (C) 2020 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */
"""

import argparse
import xml.etree.ElementTree as ET
from mako.template import Template

TEMPLATE  = Template("""\
/* DO NOT EDIT - This file is generated automatically by vtn_generator_ids.py script */

""" + COPYRIGHT + """\
<%
def get_name(generator):
    name = generator.get('tool').lower()
    name = name.replace('-', '')
    name = name.replace(' ', '_')
    name = name.replace('/', '_')
    return name
%>
enum vtn_generator {
% for generator in root.find("./ids[@type='vendor']").findall('id'):
% if 'tool' in generator.attrib:
   vtn_generator_${get_name(generator)} = ${generator.get('value')},
% endif
% endfor
   vtn_generator_max = 0xffff,
};
""")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("xml")
    p.add_argument("out")
    pargs = p.parse_args()

    tree = ET.parse(pargs.xml)
    root = tree.getroot()

    with open(pargs.out, 'w') as f:
        f.write(TEMPLATE.render(root=root))
