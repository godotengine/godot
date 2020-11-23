#!/usr/bin/python3


import module_db
import utils

import argparse
import glob
import os
import zlib


def __make_doc_data_class_path_header(doc_class_to_path: [(str, str)], output: str):
    num_doc_classes = len(doc_class_to_path)

    with open(output, 'w', encoding='utf-8') as f:
        f.write("static const int _doc_data_class_path_count = " +
                str(num_doc_classes) + ";\n")
        f.write(
            "struct _DocDataClassPath { const char* name; const char* path; };\n")

        f.write("static const _DocDataClassPath _doc_data_class_paths[" + str(
            num_doc_classes + 1) + "] = {\n")

        for doc_class, doc_path in doc_class_to_path:
            f.write(
                '\t{"' + doc_class + '", "' + utils.forward_slashes(doc_path) + '"},\n')

        f.write("\t{nullptr, nullptr}\n")
        f.write("};\n")


def __make_doc_data_header(doc_xml_paths: [str], project_root: str, output: str):
    with open(output, 'w', encoding='utf-8') as g:

        buf = ""
        docbegin = ""
        docend = ""
        for doc_xml_path in doc_xml_paths:
            with open(os.path.join(project_root, doc_xml_path), "r", encoding="utf-8") as f:
                content = f.read()
            buf += content

        buf = (docbegin + buf + docend).encode("utf-8")
        decomp_size = len(buf)

        buf = zlib.compress(buf)

        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef _DOC_DATA_RAW_H\n")
        g.write("#define _DOC_DATA_RAW_H\n")
        g.write("static const int _doc_data_compressed_size = " +
                str(len(buf)) + ";\n")
        g.write("static const int _doc_data_uncompressed_size = " +
                str(decomp_size) + ";\n")
        g.write("static const unsigned char _doc_data_compressed[] = {\n")
        for i in range(len(buf)):
            g.write("\t" + str(buf[i]) + ",\n")
        g.write("};\n")

        g.write("#endif")


def __make_doc_generated_files(module_db_file: str, project_root: str, output_data_class_path: str, output_doc_data_compressed: str):
    mdb = module_db.load_db(module_db_file)
    doc_paths: [str] = mdb.get_doc_paths()

    docs: [str] = []
    doc_class_to_path: [(str, str)] = []

    for doc_path in doc_paths:
        glob_path = os.path.join(project_root, doc_path, '*.xml')

        doc_xmls = [os.path.relpath(doc_xml, project_root)
                    for doc_xml in glob.glob(glob_path)]

        docs += doc_xmls
        doc_class_to_path += [(os.path.splitext(os.path.basename(doc_xml))
                               [0], doc_path) for doc_xml in doc_xmls]

    __make_doc_data_class_path_header(
        doc_class_to_path, output_data_class_path)

    __make_doc_data_header(docs, project_root, output_doc_data_compressed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Doc file generators.')

    parser.add_argument(
        'module_db', type=str, help='The module database json file')
    parser.add_argument(
        'project_root', type=str, help='The project root directory'
    )
    parser.add_argument(
        'output_data_class_path', type=str, help='The output generated doc_data file.')
    parser.add_argument(
        'output_doc_data_compressed', type=str, help='The output generated doc_data file.')

    args = parser.parse_args()

    __make_doc_generated_files(
        args.module_db, args.project_root, args.output_data_class_path, args.output_doc_data_compressed)
