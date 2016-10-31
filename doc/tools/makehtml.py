#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape, unescape

html_escape_table = {
    '"': "&quot;",
    "'": "&apos;"
}

html_unescape_table = {v: k for k, v in html_escape_table.items()}


def html_escape(text):
    return escape(text, html_escape_table)


def html_unescape(text):
    return unescape(text, html_unescape_table)

input_list = []

single_page = True

for arg in sys.argv[1:]:
    if arg[:1] == "-":
        if arg[1:] == "multipage":
            single_page = False
        if arg[1:] == "singlepage":
            single_page = True
    else:
        input_list.append(arg)

if len(input_list) < 1:
    print("usage: makehtml.py <classes.xml>")
    sys.exit(0)


def validate_tag(elem, tag):
    if (elem.tag != tag):
        print("Tag mismatch, expected '" + tag + "', got " + elem.tag)
        sys.exit(255)


def make_html_bottom(body):
    # make_html_top(body,True)
    ET.SubElement(body, "hr")
    copyright = ET.SubElement(body, "span")
    copyright.text = "Copyright 2008-2010 Codenix SRL"


def make_html_top(body, bottom=False):

    if (bottom):
        ET.SubElement(body, "hr")

    table = ET.SubElement(body, "table")
    table.attrib["class"] = "top_table"
    tr = ET.SubElement(table, "tr")
    td = ET.SubElement(tr, "td")
    td.attrib["class"] = "top_table"

    img = ET.SubElement(td, "image")
    img.attrib["src"] = "images/logo.png"
    td = ET.SubElement(tr, "td")
    td.attrib["class"] = "top_table"
    a = ET.SubElement(td, "a")
    a.attrib["href"] = "index.html"
    a.text = "Index"
    td = ET.SubElement(tr, "td")
    td.attrib["class"] = "top_table"
    a = ET.SubElement(td, "a")
    a.attrib["href"] = "alphabetical.html"
    a.text = "Classes"
    td = ET.SubElement(tr, "td")
    td.attrib["class"] = "top_table"
    a = ET.SubElement(td, "a")
    a.attrib["href"] = "category.html"
    a.text = "Categories"
    td = ET.SubElement(tr, "td")
    a = ET.SubElement(td, "a")
    a.attrib["href"] = "inheritance.html"
    a.text = "Inheritance"
    if (not bottom):
        ET.SubElement(body, "hr")


def make_html_class_list(class_list, columns):

    div = ET.Element("div")
    div.attrib["class"] = "ClassList"

    h1 = ET.SubElement(div, "h2")
    h1.text = "Alphabetical Class List"

    table = ET.SubElement(div, "table")
    table.attrib["class"] = "class_table"
    table.attrib["width"] = "100%"
    prev = 0

    col_max = len(class_list) / columns + 1
    print("col max is ", col_max)
    col_count = 0
    row_count = 0
    last_initial = ""
    fit_columns = []

    for n in range(0, columns):
        fit_columns += [[]]

    indexers = []
    last_initial = ""

    idx = 0
    for n in class_list:
        col = int(idx / col_max)
        if (col >= columns):
            col = columns - 1
        fit_columns[col] += [n]
        idx += 1
        if (n[:1] != last_initial):
            indexers += [n]
            last_initial = n[:1]

    row_max = 0

    for n in range(0, columns):
        if (len(fit_columns[n]) > row_max):
            row_max = len(fit_columns[n])

    for r in range(0, row_max):
        tr = ET.SubElement(table, "tr")
        for c in range(0, columns):
            tdi = ET.SubElement(tr, "td")
            tdi.attrib["align"] = "right"
            td = ET.SubElement(tr, "td")
            if (r >= len(fit_columns[c])):
                continue

            classname = fit_columns[c][r]
            print(classname)
            if (classname in indexers):

                span = ET.SubElement(tdi, "span")
                span.attrib["class"] = "class_index_letter"
                span.text = classname[:1].upper()

            if (single_page):
                link = "#" + classname
            else:
                link = classname + ".html"

            a = ET.SubElement(td, "a")
            a.attrib["href"] = link
            a.text = classname

    if (not single_page):
        cat_class_list = ET.Element("html")
        csscc = ET.SubElement(cat_class_list, "link")
        csscc.attrib["href"] = "main.css"
        csscc.attrib["rel"] = "stylesheet"
        csscc.attrib["type"] = "text/css"
        bodycc = ET.SubElement(cat_class_list, "body")
        make_html_top(bodycc)

        cat_class_parent = bodycc
    else:
        cat_class_parent = div

    h1 = ET.SubElement(cat_class_parent, "h2")
    h1.text = "Class List By Category"

    class_cat_table = {}
    class_cat_list = []

    for c in class_list:
        clss = classes[c]
        if ("category" in clss.attrib):
            class_cat = clss.attrib["category"]
        else:
            class_cat = "Core"
        if (class_cat.find("/") != -1):
            class_cat = class_cat[class_cat.rfind("/") + 1:]
        if (not class_cat in class_cat_list):
            class_cat_list.append(class_cat)
            class_cat_table[class_cat] = []
        class_cat_table[class_cat].append(c)

    class_cat_list.sort()

    ct = ET.SubElement(cat_class_parent, "table")
    for cl in class_cat_list:
        l = class_cat_table[cl]
        l.sort()
        tr = ET.SubElement(ct, "tr")
        tr.attrib["class"] = "category_title"
        td = ET.SubElement(ct, "td")
        td.attrib["class"] = "category_title"

        a = ET.SubElement(td, "a")
        a.attrib["class"] = "category_title"
        a.text = cl
        a.attrib["name"] = "CATEGORY_" + cl

        td = ET.SubElement(ct, "td")
        td.attrib["class"] = "category_title"

        for clt in l:
            tr = ET.SubElement(ct, "tr")
            td = ET.SubElement(ct, "td")
            make_type(clt, td)
            clss = classes[clt]
            bd = clss.find("brief_description")
            bdtext = ""
            if (bd != None):
                bdtext = bd.text
            td = ET.SubElement(ct, "td")
            td.text = bdtext

    if (not single_page):
        make_html_bottom(bodycc)
        catet_out = ET.ElementTree(cat_class_list)
        catet_out.write("category.html")

    if (not single_page):
        inh_class_list = ET.Element("html")
        cssic = ET.SubElement(inh_class_list, "link")
        cssic.attrib["href"] = "main.css"
        cssic.attrib["rel"] = "stylesheet"
        cssic.attrib["type"] = "text/css"
        bodyic = ET.SubElement(inh_class_list, "body")
        make_html_top(bodyic)
        inh_class_parent = bodyic
    else:
        inh_class_parent = div

    h1 = ET.SubElement(inh_class_parent, "h2")
    h1.text = "Class List By Inheritance"

    itemlist = ET.SubElement(inh_class_parent, "list")

    class_inh_table = {}

    def add_class(clss):
        if (clss.attrib["name"] in class_inh_table):
            return  # already added
        parent_list = None

        if ("inherits" in clss.attrib):
            inhc = clss.attrib["inherits"]
            if (not (inhc in class_inh_table)):
                add_class(classes[inhc])

            parent_list = class_inh_table[inhc].find("div")
            if (parent_list == None):
                parent_div = ET.SubElement(class_inh_table[inhc], "div")
                parent_list = ET.SubElement(parent_div, "list")
                parent_div.attrib["class"] = "inh_class_list"
            else:
                parent_list = parent_list.find("list")

        else:
            parent_list = itemlist

        item = ET.SubElement(parent_list, "li")
#   item.attrib["class"]="inh_class_list"
        class_inh_table[clss.attrib["name"]] = item
        make_type(clss.attrib["name"], item)

    for c in class_list:
        add_class(classes[c])

    if (not single_page):
        make_html_bottom(bodyic)
        catet_out = ET.ElementTree(inh_class_list)
        catet_out.write("inheritance.html")

    # h1=ET.SubElement(div,"h2")
    #h1.text="Class List By Inheritance"

    return div


def make_type(p_type, p_parent):
    if (p_type == "RefPtr"):
        p_type = "Resource"

    if (p_type in class_names):
        a = ET.SubElement(p_parent, "a")
        a.attrib["class"] = "datatype_existing"
        a.text = p_type + " "
        if (single_page):
            a.attrib["href"] = "#" + p_type
        else:
            a.attrib["href"] = p_type + ".html"
    else:
        span = ET.SubElement(p_parent, "span")
        span.attrib["class"] = "datatype"
        span.text = p_type + " "


def make_text_def(class_name, parent, text):
    text = html_escape(text)
    pos = 0
    while(True):
        pos = text.find("[", pos)
        if (pos == -1):
            break

        endq_pos = text.find("]", pos + 1)
        if (endq_pos == -1):
            break

        pre_text = text[:pos]
        post_text = text[endq_pos + 1:]
        tag_text = text[pos + 1:endq_pos]

        if (tag_text in class_names):
            if (single_page):
                tag_text = '<a href="#' + tag_text + '">' + tag_text + '</a>'
            else:
                tag_text = '<a href="' + tag_text + '.html">' + tag_text + '</a>'
        else:  # command
            cmd = tag_text
            space_pos = tag_text.find(" ")
            if (cmd.find("html") == 0):
                cmd = tag_text[:space_pos]
                param = tag_text[space_pos + 1:]
                tag_text = "<" + param + ">"
            elif(cmd.find("method") == 0):
                cmd = tag_text[:space_pos]
                param = tag_text[space_pos + 1:]

                if (not single_page and param.find(".") != -1):
                    class_param, method_param = param.split(".")
                    tag_text = tag_text = '<a href="' + class_param + '.html#' + class_param + "_" + method_param + '">' + class_param + '.' + method_param + '()</a>'
                else:
                    tag_text = tag_text = '<a href="#' + class_name + "_" + param + '">' + class_name + '.' + param + '()</a>'
            elif (cmd.find("image=") == 0):
                print("found image: " + cmd)
                tag_text = "<img src=" + cmd[6:] + "/>"
            elif (cmd.find("url=") == 0):
                tag_text = "<a href=" + cmd[4:] + ">"
            elif (cmd == "/url"):
                tag_text = "</a>"
            elif (cmd == "center"):
                tag_text = "<div align=\"center\">"
            elif (cmd == "/center"):
                tag_text = "</div>"
            elif (cmd == "br"):
                tag_text = "<br/>"
            elif (cmd == "i" or cmd == "/i" or cmd == "b" or cmd == "/b" or cmd == "u" or cmd == "/u"):
                tag_text = "<" + tag_text + ">"  # html direct mapping
            else:
                tag_text = "[" + tag_text + "]"

        text = pre_text + tag_text + post_text
        pos = len(pre_text) + len(tag_text)

    #tnode = ET.SubElement(parent,"div")
    # tnode.text=text
    text = "<div class=\"description\">" + text + "</div>"
    try:
        tnode = ET.XML(text)
        parent.append(tnode)
    except:
        print("Error parsing description text: '" + text + "'")
        sys.exit(255)

    return tnode


def make_method_def(name, m, declare, event=False):

    mdata = {}

    if (not declare):
        div = ET.Element("tr")
        div.attrib["class"] = "method"
        ret_parent = ET.SubElement(div, "td")
        ret_parent.attrib["align"] = "right"
        func_parent = ET.SubElement(div, "td")
    else:
        div = ET.Element("div")
        div.attrib["class"] = "method"
        ret_parent = div
        func_parent = div

    mdata["argidx"] = []
    mdata["name"] = m.attrib["name"]
    qualifiers = ""
    if ("qualifiers" in m.attrib):
        qualifiers = m.attrib["qualifiers"]

    args = list(m)
    for a in args:
        if (a.tag == "return"):
            idx = -1
        elif (a.tag == "argument"):
            idx = int(a.attrib["index"])
        else:
            continue

        mdata["argidx"].append(idx)
        mdata[idx] = a

    if (not event):
        if (-1 in mdata["argidx"]):
            make_type(mdata[-1].attrib["type"], ret_parent)
            mdata["argidx"].remove(-1)
        else:
            make_type("void", ret_parent)

    span = ET.SubElement(func_parent, "span")
    if (declare):
        span.attrib["class"] = "funcdecl"
        a = ET.SubElement(span, "a")
        a.attrib["name"] = name + "_" + m.attrib["name"]
        a.text = name + "::" + m.attrib["name"]
    else:
        span.attrib["class"] = "identifier funcdef"
        a = ET.SubElement(span, "a")
        a.attrib["href"] = "#" + name + "_" + m.attrib["name"]
        a.text = m.attrib["name"]

    span = ET.SubElement(func_parent, "span")
    span.attrib["class"] = "symbol"
    span.text = " ("

    for a in mdata["argidx"]:
        arg = mdata[a]
        if (a > 0):
            span = ET.SubElement(func_parent, "span")
            span.text = ", "
        else:
            span = ET.SubElement(func_parent, "span")
            span.text = " "

        make_type(arg.attrib["type"], func_parent)

        span = ET.SubElement(func_parent, "span")
        span.text = arg.attrib["name"]
        if ("default" in arg.attrib):
            span.text = span.text + "=" + arg.attrib["default"]

    span = ET.SubElement(func_parent, "span")
    span.attrib["class"] = "symbol"
    if (len(mdata["argidx"])):
        span.text = " )"
    else:
        span.text = ")"

    if (qualifiers):
        span = ET.SubElement(func_parent, "span")
        span.attrib["class"] = "qualifier"
        span.text = " " + qualifiers

    return div


def make_html_class(node):

    div = ET.Element("div")
    div.attrib["class"] = "class"

    a = ET.SubElement(div, "a")
    a.attrib["name"] = node.attrib["name"]

    h3 = ET.SubElement(a, "h3")
    h3.attrib["class"] = "title class_title"
    h3.text = node.attrib["name"]

    briefd = node.find("brief_description")
    if (briefd != None):
        div2 = ET.SubElement(div, "div")
        div2.attrib["class"] = "description class_description"
        div2.text = briefd.text

    if ("inherits" in node.attrib):
        ET.SubElement(div, "br")

        div2 = ET.SubElement(div, "div")
        div2.attrib["class"] = "inheritance"

        span = ET.SubElement(div2, "span")
        span.text = "Inherits: "

        make_type(node.attrib["inherits"], div2)

    if ("category" in node.attrib):
        ET.SubElement(div, "br")

        div3 = ET.SubElement(div, "div")
        div3.attrib["class"] = "category"

        span = ET.SubElement(div3, "span")
        span.attrib["class"] = "category"
        span.text = "Category: "

        a = ET.SubElement(div3, "a")
        a.attrib["class"] = "category_ref"
        a.text = node.attrib["category"]
        catname = a.text
        if (catname.rfind("/") != -1):
            catname = catname[catname.rfind("/"):]
        catname = "CATEGORY_" + catname

        if (single_page):
            a.attrib["href"] = "#" + catname
        else:
            a.attrib["href"] = "category.html#" + catname

    methods = node.find("methods")

    if(methods != None and len(list(methods)) > 0):

        h4 = ET.SubElement(div, "h4")
        h4.text = "Public Methods:"

        method_table = ET.SubElement(div, "table")
        method_table.attrib["class"] = "method_list"

        for m in list(methods):
            #li = ET.SubElement(div2, "li")
            method_table.append(make_method_def(node.attrib["name"], m, False))

    events = node.find("signals")

    if(events != None and len(list(events)) > 0):
        h4 = ET.SubElement(div, "h4")
        h4.text = "Events:"

        event_table = ET.SubElement(div, "table")
        event_table.attrib["class"] = "method_list"

        for m in list(events):
            #li = ET.SubElement(div2, "li")
            event_table.append(make_method_def(node.attrib["name"], m, False, True))

    members = node.find("members")
    if(members != None and len(list(members)) > 0):

        h4 = ET.SubElement(div, "h4")
        h4.text = "Public Variables:"
        div2 = ET.SubElement(div, "div")
        div2.attrib["class"] = "member_list"

        for c in list(members):

            li = ET.SubElement(div2, "li")
            div3 = ET.SubElement(li, "div")
            div3.attrib["class"] = "member"
            make_type(c.attrib["type"], div3)
            span = ET.SubElement(div3, "span")
            span.attrib["class"] = "identifier member_name"
            span.text = " " + c.attrib["name"] + " "
            span = ET.SubElement(div3, "span")
            span.attrib["class"] = "member_description"
            span.text = c.text

    constants = node.find("constants")
    if(constants != None and len(list(constants)) > 0):

        h4 = ET.SubElement(div, "h4")
        h4.text = "Constants:"
        div2 = ET.SubElement(div, "div")
        div2.attrib["class"] = "constant_list"

        for c in list(constants):
            li = ET.SubElement(div2, "li")
            div3 = ET.SubElement(li, "div")
            div3.attrib["class"] = "constant"

            span = ET.SubElement(div3, "span")
            span.attrib["class"] = "identifier constant_name"
            span.text = c.attrib["name"] + " "
            if ("value" in c.attrib):
                span = ET.SubElement(div3, "span")
                span.attrib["class"] = "symbol"
                span.text = "= "
                span = ET.SubElement(div3, "span")
                span.attrib["class"] = "constant_value"
                span.text = c.attrib["value"] + " "
            span = ET.SubElement(div3, "span")
            span.attrib["class"] = "constant_description"
            span.text = c.text

#  ET.SubElement(div,"br")

    descr = node.find("description")
    if (descr != None and descr.text.strip() != ""):
        h4 = ET.SubElement(div, "h4")
        h4.text = "Description:"

        make_text_def(node.attrib["name"], div, descr.text)
#   div2=ET.SubElement(div,"div")
#   div2.attrib["class"]="description";
#   div2.text=descr.text

    if(methods != None or events != None):

        h4 = ET.SubElement(div, "h4")
        h4.text = "Method Documentation:"
        iter_list = []
        if (methods != None):
            iter_list += list(methods)
        if (events != None):
            iter_list += list(events)

        for m in iter_list:

            descr = m.find("description")

            if (descr == None or descr.text.strip() == ""):
                continue

            div2 = ET.SubElement(div, "div")
            div2.attrib["class"] = "method_doc"

            div2.append(make_method_def(node.attrib["name"], m, True))
            #anchor = ET.SubElement(div2, "a")
            # anchor.attrib["name"] =
            make_text_def(node.attrib["name"], div2, descr.text)
            # div3=ET.SubElement(div2,"div")
            # div3.attrib["class"]="description";
            # div3.text=descr.text

    return div

class_names = []
classes = {}

for file in input_list:
    tree = ET.parse(file)
    doc = tree.getroot()

    if ("version" not in doc.attrib):
        print("Version missing from 'doc'")
        sys.exit(255)

        version = doc.attrib["version"]

    for c in list(doc):
        if (c.attrib["name"] in class_names):
            continue
        class_names.append(c.attrib["name"])
        classes[c.attrib["name"]] = c

html = ET.Element("html")
css = ET.SubElement(html, "link")
css.attrib["href"] = "main.css"
css.attrib["rel"] = "stylesheet"
css.attrib["type"] = "text/css"

body = ET.SubElement(html, "body")
if (not single_page):
    make_html_top(body)


class_names.sort()

body.append(make_html_class_list(class_names, 5))

for cn in class_names:
    c = classes[cn]
    if (single_page):
        body.append(make_html_class(c))
    else:
        html2 = ET.Element("html")
        css = ET.SubElement(html2, "link")
        css.attrib["href"] = "main.css"
        css.attrib["rel"] = "stylesheet"
        css.attrib["type"] = "text/css"
        body2 = ET.SubElement(html2, "body")
        make_html_top(body2)
        body2.append(make_html_class(c))
        make_html_bottom(body2)
        et_out = ET.ElementTree(html2)
        et_out.write(c.attrib["name"] + ".html")


et_out = ET.ElementTree(html)
if (single_page):
    et_out.write("singlepage.html")
else:
    make_html_bottom(body)
    et_out.write("alphabetical.html")
