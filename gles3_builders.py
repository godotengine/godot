"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""
from platform_methods import subprocess_main


class LegacyGLHeaderStruct:
    def __init__(self):
        self.vertex_lines = []
        self.fragment_lines = []
        self.uniforms = []
        self.attributes = []
        self.feedbacks = []
        self.fbos = []
        self.conditionals = []
        self.enums = {}
        self.texunits = []
        self.texunit_names = []
        self.ubos = []
        self.ubo_names = []

        self.vertex_included_files = []
        self.fragment_included_files = []

        self.reading = ""
        self.line_offset = 0
        self.vertex_offset = 0
        self.fragment_offset = 0


def include_file_in_legacygl_header(filename, header_data, depth):
    fs = open(filename, "r")
    line = fs.readline()

    while line:

        if line.find("[vertex]") != -1:
            header_data.reading = "vertex"
            line = fs.readline()
            header_data.line_offset += 1
            header_data.vertex_offset = header_data.line_offset
            continue

        if line.find("[fragment]") != -1:
            header_data.reading = "fragment"
            line = fs.readline()
            header_data.line_offset += 1
            header_data.fragment_offset = header_data.line_offset
            continue

        while line.find("#include ") != -1:
            includeline = line.replace("#include ", "").strip()[1:-1]

            import os.path

            included_file = os.path.relpath(os.path.dirname(filename) + "/" + includeline)
            if not included_file in header_data.vertex_included_files and header_data.reading == "vertex":
                header_data.vertex_included_files += [included_file]
                if include_file_in_legacygl_header(included_file, header_data, depth + 1) is None:
                    print("Error in file '" + filename + "': #include " + includeline + "could not be found!")
            elif not included_file in header_data.fragment_included_files and header_data.reading == "fragment":
                header_data.fragment_included_files += [included_file]
                if include_file_in_legacygl_header(included_file, header_data, depth + 1) is None:
                    print("Error in file '" + filename + "': #include " + includeline + "could not be found!")

            line = fs.readline()

        if line.find("#ifdef ") != -1:
            if line.find("#ifdef ") != -1:
                ifdefline = line.replace("#ifdef ", "").strip()

            if line.find("_EN_") != -1:
                enumbase = ifdefline[: ifdefline.find("_EN_")]
                ifdefline = ifdefline.replace("_EN_", "_")
                line = line.replace("_EN_", "_")
                if enumbase not in header_data.enums:
                    header_data.enums[enumbase] = []
                if ifdefline not in header_data.enums[enumbase]:
                    header_data.enums[enumbase].append(ifdefline)

            elif not ifdefline in header_data.conditionals:
                header_data.conditionals += [ifdefline]

        if line.find("uniform") != -1 and line.lower().find("texunit:") != -1:
            # texture unit
            texunitstr = line[line.find(":") + 1 :].strip()
            if texunitstr == "auto":
                texunit = "-1"
            else:
                texunit = str(int(texunitstr))
            uline = line[: line.lower().find("//")]
            uline = uline.replace("uniform", "")
            uline = uline.replace("highp", "")
            uline = uline.replace(";", "")
            lines = uline.split(",")
            for x in lines:

                x = x.strip()
                x = x[x.rfind(" ") + 1 :]
                if x.find("[") != -1:
                    # unfiorm array
                    x = x[: x.find("[")]

                if not x in header_data.texunit_names:
                    header_data.texunits += [(x, texunit)]
                    header_data.texunit_names += [x]

        elif line.find("uniform") != -1 and line.lower().find("ubo:") != -1:
            # uniform buffer object
            ubostr = line[line.find(":") + 1 :].strip()
            ubo = str(int(ubostr))
            uline = line[: line.lower().find("//")]
            uline = uline[uline.find("uniform") + len("uniform") :]
            uline = uline.replace("highp", "")
            uline = uline.replace(";", "")
            uline = uline.replace("{", "").strip()
            lines = uline.split(",")
            for x in lines:

                x = x.strip()
                x = x[x.rfind(" ") + 1 :]
                if x.find("[") != -1:
                    # unfiorm array
                    x = x[: x.find("[")]

                if not x in header_data.ubo_names:
                    header_data.ubos += [(x, ubo)]
                    header_data.ubo_names += [x]

        elif line.find("uniform") != -1 and line.find("{") == -1 and line.find(";") != -1:
            uline = line.replace("uniform", "")
            uline = uline.replace(";", "")
            lines = uline.split(",")
            for x in lines:

                x = x.strip()
                x = x[x.rfind(" ") + 1 :]
                if x.find("[") != -1:
                    # unfiorm array
                    x = x[: x.find("[")]

                if not x in header_data.uniforms:
                    header_data.uniforms += [x]

        if line.strip().find("attribute ") == 0 and line.find("attrib:") != -1:
            uline = line.replace("in ", "")
            uline = uline.replace("attribute ", "")
            uline = uline.replace("highp ", "")
            uline = uline.replace(";", "")
            uline = uline[uline.find(" ") :].strip()

            if uline.find("//") != -1:
                name, bind = uline.split("//")
                if bind.find("attrib:") != -1:
                    name = name.strip()
                    bind = bind.replace("attrib:", "").strip()
                    header_data.attributes += [(name, bind)]

        if line.strip().find("out ") == 0 and line.find("tfb:") != -1:
            uline = line.replace("out ", "")
            uline = uline.replace("highp ", "")
            uline = uline.replace(";", "")
            uline = uline[uline.find(" ") :].strip()

            if uline.find("//") != -1:
                name, bind = uline.split("//")
                if bind.find("tfb:") != -1:
                    name = name.strip()
                    bind = bind.replace("tfb:", "").strip()
                    header_data.feedbacks += [(name, bind)]

        line = line.replace("\r", "")
        line = line.replace("\n", "")

        if header_data.reading == "vertex":
            header_data.vertex_lines += [line]
        if header_data.reading == "fragment":
            header_data.fragment_lines += [line]

        line = fs.readline()
        header_data.line_offset += 1

    fs.close()

    return header_data


def build_legacygl_header(filename, include, class_suffix, output_attribs):
    header_data = LegacyGLHeaderStruct()
    include_file_in_legacygl_header(filename, header_data, 0)

    out_file = filename + ".gen.h"
    fd = open(out_file, "w")

    enum_constants = []

    fd.write("/* WARNING, THIS FILE WAS GENERATED, DO NOT EDIT */\n")

    out_file_base = out_file
    out_file_base = out_file_base[out_file_base.rfind("/") + 1 :]
    out_file_base = out_file_base[out_file_base.rfind("\\") + 1 :]
    out_file_ifdef = out_file_base.replace(".", "_").upper()
    fd.write("#ifndef " + out_file_ifdef + class_suffix + "_120\n")
    fd.write("#define " + out_file_ifdef + class_suffix + "_120\n")

    out_file_class = (
        out_file_base.replace(".glsl.gen.h", "").title().replace("_", "").replace(".", "") + "Shader" + class_suffix
    )
    fd.write("\n\n")
    fd.write('#include "' + include + '"\n\n\n')
    fd.write("class " + out_file_class + " : public Shader" + class_suffix + " {\n\n")
    fd.write('\t virtual String get_shader_name() const { return "' + out_file_class + '"; }\n')

    fd.write("public:\n\n")

    if header_data.conditionals:
        fd.write("\tenum Conditionals {\n")
        for x in header_data.conditionals:
            fd.write("\t\t" + x.upper() + ",\n")
        fd.write("\t};\n\n")

    if header_data.uniforms:
        fd.write("\tenum Uniforms {\n")
        for x in header_data.uniforms:
            fd.write("\t\t" + x.upper() + ",\n")
        fd.write("\t};\n\n")

    fd.write("\t_FORCE_INLINE_ int get_uniform(Uniforms p_uniform) const { return _get_uniform(p_uniform); }\n\n")
    if header_data.conditionals:
        fd.write(
            "\t_FORCE_INLINE_ void set_conditional(Conditionals p_conditional,bool p_enable)  {  _set_conditional(p_conditional,p_enable); }\n\n"
        )
    fd.write("\t#ifdef DEBUG_ENABLED\n ")
    fd.write(
        "\t#define _FU if (get_uniform(p_uniform)<0) return; if (!is_version_valid()) return; ERR_FAIL_COND( get_active()!=this ); \n\n "
    )
    fd.write("\t#else\n ")
    fd.write("\t#define _FU if (get_uniform(p_uniform)<0) return; \n\n ")
    fd.write("\t#endif\n")
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, float p_value) { _FU glUniform1f(get_uniform(p_uniform),p_value); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, double p_value) { _FU glUniform1f(get_uniform(p_uniform),p_value); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, uint8_t p_value) { _FU glUniform1i(get_uniform(p_uniform),p_value); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, int8_t p_value) { _FU glUniform1i(get_uniform(p_uniform),p_value); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, uint16_t p_value) { _FU glUniform1i(get_uniform(p_uniform),p_value); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, int16_t p_value) { _FU glUniform1i(get_uniform(p_uniform),p_value); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, uint32_t p_value) { _FU glUniform1i(get_uniform(p_uniform),p_value); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, int32_t p_value) { _FU glUniform1i(get_uniform(p_uniform),p_value); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, const Color& p_color) { _FU GLfloat col[4]={p_color.r,p_color.g,p_color.b,p_color.a}; glUniform4fv(get_uniform(p_uniform),1,col); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, const Vector2& p_vec2) { _FU GLfloat vec2[2]={(GLfloat)p_vec2.x,(GLfloat)p_vec2.y}; glUniform2fv(get_uniform(p_uniform),1,vec2); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, const Size2i& p_vec2) { _FU GLint vec2[2]={(GLint)p_vec2.x,(GLint)p_vec2.y}; glUniform2iv(get_uniform(p_uniform),1,vec2); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, const Vector3& p_vec3) { _FU GLfloat vec3[3]={(GLfloat)p_vec3.x,(GLfloat)p_vec3.y,(GLfloat)p_vec3.z}; glUniform3fv(get_uniform(p_uniform),1,vec3); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, float p_a, float p_b) { _FU glUniform2f(get_uniform(p_uniform),p_a,p_b); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, float p_a, float p_b, float p_c) { _FU glUniform3f(get_uniform(p_uniform),p_a,p_b,p_c); }\n\n"
    )
    fd.write(
        "\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, float p_a, float p_b, float p_c, float p_d) { _FU glUniform4f(get_uniform(p_uniform),p_a,p_b,p_c,p_d); }\n\n"
    )

    fd.write(
        """\t_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, const Transform3D& p_transform) {  _FU

        const Transform3D &tr = p_transform;

        GLfloat matrix[16]={ /* build a 16x16 matrix */
            (GLfloat)tr.basis.elements[0][0],
            (GLfloat)tr.basis.elements[1][0],
            (GLfloat)tr.basis.elements[2][0],
            (GLfloat)0,
            (GLfloat)tr.basis.elements[0][1],
            (GLfloat)tr.basis.elements[1][1],
            (GLfloat)tr.basis.elements[2][1],
            (GLfloat)0,
            (GLfloat)tr.basis.elements[0][2],
            (GLfloat)tr.basis.elements[1][2],
            (GLfloat)tr.basis.elements[2][2],
            (GLfloat)0,
            (GLfloat)tr.origin.x,
            (GLfloat)tr.origin.y,
            (GLfloat)tr.origin.z,
            (GLfloat)1
        };


                glUniformMatrix4fv(get_uniform(p_uniform),1,false,matrix);


    }

    """
    )

    fd.write(
        """_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, const Transform2D& p_transform) {  _FU

        const Transform2D &tr = p_transform;

        GLfloat matrix[16]={ /* build a 16x16 matrix */
            (GLfloat)tr.elements[0][0],
            (GLfloat)tr.elements[0][1],
            (GLfloat)0,
            (GLfloat)0,
            (GLfloat)tr.elements[1][0],
            (GLfloat)tr.elements[1][1],
            (GLfloat)0,
            (GLfloat)0,
            (GLfloat)0,
            (GLfloat)0,
            (GLfloat)1,
            (GLfloat)0,
            (GLfloat)tr.elements[2][0],
            (GLfloat)tr.elements[2][1],
            (GLfloat)0,
            (GLfloat)1
        };


        glUniformMatrix4fv(get_uniform(p_uniform),1,false,matrix);


    }

    """
    )

    fd.write(
        """_FORCE_INLINE_ void set_uniform(Uniforms p_uniform, const CameraMatrix& p_matrix) {  _FU

        GLfloat matrix[16];

        for (int i=0;i<4;i++) {
            for (int j=0;j<4;j++) {
                matrix[i*4+j]=p_matrix.matrix[i][j];
            }
        }

        glUniformMatrix4fv(get_uniform(p_uniform),1,false,matrix);
}"""
    )

    fd.write("\n\n#undef _FU\n\n\n")

    fd.write("\tvirtual void init() {\n\n")

    enum_value_count = 0

    if header_data.enums:

        fd.write("\t\t//Written using math, given nonstandarity of 64 bits integer constants..\n")
        fd.write("\t\tstatic const Enum _enums[]={\n")

        bitofs = len(header_data.conditionals)
        enum_vals = []

        for xv in header_data.enums:
            x = header_data.enums[xv]
            bits = 1
            amt = len(x)
            while 2 ** bits < amt:
                bits += 1
            strs = "{"
            for i in range(amt):
                strs += '"#define ' + x[i] + '\\n",'

                c = {}
                c["set_mask"] = "uint64_t(" + str(i) + ")<<" + str(bitofs)
                c["clear_mask"] = (
                    "((uint64_t(1)<<40)-1) ^ (((uint64_t(1)<<" + str(bits) + ") - 1)<<" + str(bitofs) + ")"
                )
                enum_vals.append(c)
                enum_constants.append(x[i])

            strs += "NULL}"

            fd.write(
                "\t\t\t{(uint64_t(1<<" + str(bits) + ")-1)<<" + str(bitofs) + "," + str(bitofs) + "," + strs + "},\n"
            )
            bitofs += bits

        fd.write("\t\t};\n\n")

        fd.write("\t\tstatic const EnumValue _enum_values[]={\n")

        enum_value_count = len(enum_vals)
        for x in enum_vals:
            fd.write("\t\t\t{" + x["set_mask"] + "," + x["clear_mask"] + "},\n")

        fd.write("\t\t};\n\n")

    conditionals_found = []
    if header_data.conditionals:

        fd.write("\t\tstatic const char* _conditional_strings[]={\n")
        if header_data.conditionals:
            for x in header_data.conditionals:
                fd.write('\t\t\t"#define ' + x + '\\n",\n')
                conditionals_found.append(x)
        fd.write("\t\t};\n\n")
    else:
        fd.write("\t\tstatic const char **_conditional_strings=NULL;\n")

    if header_data.uniforms:

        fd.write("\t\tstatic const char* _uniform_strings[]={\n")
        if header_data.uniforms:
            for x in header_data.uniforms:
                fd.write('\t\t\t"' + x + '",\n')
        fd.write("\t\t};\n\n")
    else:
        fd.write("\t\tstatic const char **_uniform_strings=NULL;\n")

    if output_attribs:
        if header_data.attributes:

            fd.write("\t\tstatic AttributePair _attribute_pairs[]={\n")
            for x in header_data.attributes:
                fd.write('\t\t\t{"' + x[0] + '",' + x[1] + "},\n")
            fd.write("\t\t};\n\n")
        else:
            fd.write("\t\tstatic AttributePair *_attribute_pairs=NULL;\n")

    feedback_count = 0

    if header_data.texunits:
        fd.write("\t\tstatic TexUnitPair _texunit_pairs[]={\n")
        for x in header_data.texunits:
            fd.write('\t\t\t{"' + x[0] + '",' + x[1] + "},\n")
        fd.write("\t\t};\n\n")
    else:
        fd.write("\t\tstatic TexUnitPair *_texunit_pairs=NULL;\n")

    fd.write("\t\tstatic const char _vertex_code[]={\n")
    for x in header_data.vertex_lines:
        for c in x:
            fd.write(str(ord(c)) + ",")

        fd.write(str(ord("\n")) + ",")
    fd.write("\t\t0};\n\n")

    fd.write("\t\tstatic const int _vertex_code_start=" + str(header_data.vertex_offset) + ";\n")

    fd.write("\t\tstatic const char _fragment_code[]={\n")
    for x in header_data.fragment_lines:
        for c in x:
            fd.write(str(ord(c)) + ",")

        fd.write(str(ord("\n")) + ",")
    fd.write("\t\t0};\n\n")

    fd.write("\t\tstatic const int _fragment_code_start=" + str(header_data.fragment_offset) + ";\n")

    if output_attribs:
        fd.write(
            "\t\tsetup(_conditional_strings,"
            + str(len(header_data.conditionals))
            + ",_uniform_strings,"
            + str(len(header_data.uniforms))
            + ",_attribute_pairs,"
            + str(len(header_data.attributes))
            + ", _texunit_pairs,"
            + str(len(header_data.texunits))
            + ",_vertex_code,_fragment_code,_vertex_code_start,_fragment_code_start);\n"
        )
    else:
        fd.write(
            "\t\tsetup(_conditional_strings,"
            + str(len(header_data.conditionals))
            + ",_uniform_strings,"
            + str(len(header_data.uniforms))
            + ",_texunit_pairs,"
            + str(len(header_data.texunits))
            + ",_enums,"
            + str(len(header_data.enums))
            + ",_enum_values,"
            + str(enum_value_count)
            + ",_vertex_code,_fragment_code,_vertex_code_start,_fragment_code_start);\n"
        )

    fd.write("\t}\n\n")

    if enum_constants:

        fd.write("\tenum EnumConditionals {\n")
        for x in enum_constants:
            fd.write("\t\t" + x.upper() + ",\n")
        fd.write("\t};\n\n")
        fd.write("\tvoid set_enum_conditional(EnumConditionals p_cond) { _set_enum_conditional(p_cond); }\n")

    fd.write("};\n\n")
    fd.write("#endif\n\n")
    fd.close()


def build_gles3_headers(target, source, env):
    for x in source:
        build_legacygl_header(str(x), include="drivers/gles3/shader_gles3.h", class_suffix="GLES3", output_attribs=True)


if __name__ == "__main__":
    subprocess_main(globals())
