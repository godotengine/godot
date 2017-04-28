header = """\
/*************************************************************************/
/*  $filename                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
"""

f = open("files", "rb")

fname = f.readline()
while (fname != ""):

    fr = open(fname.strip(), "rb")
    l = fr.readline()
    bc = False
    fsingle = fname.strip()

    if (fsingle.find("/") != -1):
        fsingle = fsingle[fsingle.rfind("/") + 1:]
    rep_fl = "$filename"
    rep_fi = fsingle
    len_fl = len(rep_fl)
    len_fi = len(rep_fi)
    if (len_fi < len_fl):
        for x in range(len_fl - len_fi):
            rep_fi += " "
    elif (len_fl < len_fi):
        for x in range(len_fi - len_fl):
            rep_fl += " "
    if (header.find(rep_fl) != -1):
        text = header.replace(rep_fl, rep_fi)
    else:
        text = header.replace("$filename", fsingle)

    while (l != ""):
        if ((l.find("//") != 0 and l.find("/*") != 0 and l.strip() != "") or bc):
            text += l
            bc = True
        l = fr.readline()

    fr.close()
    fr = open(fname.strip(), "wb")
    fr.write(text)
    fr.close()
    # print(text)
    fname = f.readline()
