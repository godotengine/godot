
def generate_header(src, dst, version_dst):
    from compat import byte_to_str

    with open(dst, 'w') as header:
        header.write('/* THIS FILE IS GENERATED DO NOT EDIT */\n')
        header.write('#ifndef CS_COMPRESSED_H\n')
        header.write('#define CS_COMPRESSED_H\n\n')
        header.write('#ifdef TOOLS_ENABLED\n\n')
        header.write('#include "core/map.h"\n')
        header.write('#include "core/ustring.h"\n')
        inserted_files = ''
        import os
        latest_mtime = 0
        cs_file_count = 0
        for root, _, files in os.walk(src):
            files = [f for f in files if f.endswith('.cs')]
            for file in files:
                cs_file_count += 1
                filepath = os.path.join(root, file)
                filepath_src_rel = os.path.relpath(filepath, src)
                mtime = os.path.getmtime(filepath)
                latest_mtime = mtime if mtime > latest_mtime else latest_mtime
                with open(filepath, 'rb') as f:
                    buf = f.read()
                    decompr_size = len(buf)
                    import zlib
                    buf = zlib.compress(buf)
                    compr_size = len(buf)
                    name = str(cs_file_count)
                    header.write('\n')
                    header.write('// ' + filepath_src_rel + '\n')
                    header.write('static const int _cs_' + name + '_compressed_size = ' + str(compr_size) + ';\n')
                    header.write('static const int _cs_' + name + '_uncompressed_size = ' + str(decompr_size) + ';\n')
                    header.write('static const unsigned char _cs_' + name + '_compressed[] = { ')
                    for i, buf_idx in enumerate(range(compr_size)):
                        if i > 0:
                            header.write(', ')
                        header.write(byte_to_str(buf[buf_idx]))
                    header.write(' };\n')
                    inserted_files += '\tr_files.insert("' + filepath_src_rel.replace('\\', '\\\\') + '", ' \
                                        'GodotCsCompressedFile(_cs_' + name + '_compressed_size, ' \
                                        '_cs_' + name + '_uncompressed_size, ' \
                                        '_cs_' + name + '_compressed));\n'
        header.write('\nstruct GodotCsCompressedFile\n' '{\n'
            '\tint compressed_size;\n' '\tint uncompressed_size;\n' '\tconst unsigned char* data;\n'
            '\n\tGodotCsCompressedFile(int p_comp_size, int p_uncomp_size, const unsigned char* p_data)\n'
            '\t{\n' '\t\tcompressed_size = p_comp_size;\n' '\t\tuncompressed_size = p_uncomp_size;\n'
            '\t\tdata = p_data;\n' '\t}\n' '\n\tGodotCsCompressedFile() {}\n' '};\n'
            '\nvoid get_compressed_files(Map<String, GodotCsCompressedFile>& r_files)\n' '{\n' + inserted_files + '}\n'
            )
        header.write('\n#endif // TOOLS_ENABLED\n')
        header.write('\n#endif // CS_COMPRESSED_H\n')

        glue_version = int(latest_mtime) # The latest modified time will do for now

        with open(version_dst, 'w') as version_header:
            version_header.write('/* THIS FILE IS GENERATED DO NOT EDIT */\n')
            version_header.write('#ifndef CS_GLUE_VERSION_H\n')
            version_header.write('#define CS_GLUE_VERSION_H\n\n')
            version_header.write('#define CS_GLUE_VERSION UINT32_C(' + str(glue_version) + ')\n')
            version_header.write('\n#endif // CS_GLUE_VERSION_H\n')
