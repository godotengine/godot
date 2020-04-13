def generate_header(solution_dir, version_header_dst):
    import os

    latest_mtime = 0
    for root, dirs, files in os.walk(solution_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in ["Generated"]]  # Ignored generated files
        files = [f for f in files if f.endswith(".cs")]
        for file in files:
            filepath = os.path.join(root, file)
            mtime = os.path.getmtime(filepath)
            latest_mtime = mtime if mtime > latest_mtime else latest_mtime

    glue_version = int(latest_mtime)  # The latest modified time will do for now

    with open(version_header_dst, "w") as version_header:
        version_header.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        version_header.write("#ifndef CS_GLUE_VERSION_H\n")
        version_header.write("#define CS_GLUE_VERSION_H\n\n")
        version_header.write("#define CS_GLUE_VERSION UINT32_C(" + str(glue_version) + ")\n")
        version_header.write("\n#endif // CS_GLUE_VERSION_H\n")
