"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""
from platform_methods import subprocess_main


class GLES3HeaderStruct:
    def __init__(self):
        self.vertex_lines = []
        self.fragment_lines = []
        self.uniforms = []
        self.fbos = []
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
        self.variant_defines = []
        self.variant_names = []
        self.specialization_names = []
        self.specialization_values = []


def include_file_in_gles3_header(filename, header_data, depth):
    fs = open(filename, "r")
    line = fs.readline()

    while line:

        if line.find("=") != -1 and header_data.reading == "":
            # Mode
            eqpos = line.find("=")
            defname = line[:eqpos].strip().upper()
            define = line[eqpos + 1 :].strip()
            header_data.variant_names.append(defname)
            header_data.variant_defines.append(define)
            line = fs.readline()
            header_data.line_offset += 1
            header_data.vertex_offset = header_data.line_offset
            continue

        if line.find("=") != -1 and header_data.reading == "specializations":
            # Specialization
            eqpos = line.find("=")
            specname = line[:eqpos].strip()
            specvalue = line[eqpos + 1 :]
            header_data.specialization_names.append(specname)
            header_data.specialization_values.append(specvalue)
            line = fs.readline()
            header_data.line_offset += 1
            header_data.vertex_offset = header_data.line_offset
            continue

        if line.find("#[modes]") != -1:
            # Nothing really, just skip
            line = fs.readline()
            header_data.line_offset += 1
            header_data.vertex_offset = header_data.line_offset
            continue

        if line.find("#[specializations]") != -1:
            header_data.reading = "specializations"
            line = fs.readline()
            header_data.line_offset += 1
            header_data.vertex_offset = header_data.line_offset
            continue

        if line.find("#[vertex]") != -1:
            header_data.reading = "vertex"
            line = fs.readline()
            header_data.line_offset += 1
            header_data.vertex_offset = header_data.line_offset
            continue

        if line.find("#[fragment]") != -1:
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
                if include_file_in_gles3_header(included_file, header_data, depth + 1) is None:
                    print("Error in file '" + filename + "': #include " + includeline + "could not be found!")
            elif not included_file in header_data.fragment_included_files and header_data.reading == "fragment":
                header_data.fragment_included_files += [included_file]
                if include_file_in_gles3_header(included_file, header_data, depth + 1) is None:
                    print("Error in file '" + filename + "': #include " + includeline + "could not be found!")

            line = fs.readline()

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


def build_gles3_header(filename, include, class_suffix, output_attribs):
    header_data = GLES3HeaderStruct()
    include_file_in_gles3_header(filename, header_data, 0)

    out_file = filename + ".gen.h"
    fd = open(out_file, "w")
    defspec = 0
    defvariant = ""

    enum_constants = []

    fd.write("/* WARNING, THIS FILE WAS GENERATED, DO NOT EDIT */\n")

    out_file_base = out_file
    out_file_base = out_file_base[out_file_base.rfind("/") + 1 :]
    out_file_base = out_file_base[out_file_base.rfind("\\") + 1 :]
    out_file_ifdef = out_file_base.replace(".", "_").upper()
    fd.write("#ifndef " + out_file_ifdef + class_suffix + "_GLES3\n")
    fd.write("#define " + out_file_ifdef + class_suffix + "_GLES3\n")

    out_file_class = (
        out_file_base.replace(".glsl.gen.h", "").title().replace("_", "").replace(".", "") + "Shader" + class_suffix
    )
    fd.write("\n\n")
    fd.write('#include "' + include + '"\n\n\n')
    fd.write("class " + out_file_class + " : public Shader" + class_suffix + " {\n\n")

    fd.write("public:\n\n")

    if header_data.uniforms:
        fd.write("\tenum Uniforms {\n")
        for x in header_data.uniforms:
            fd.write("\t\t" + x.upper() + ",\n")
        fd.write("\t};\n\n")

    if header_data.variant_names:
        fd.write("\tenum ShaderVariant {\n")
        for x in header_data.variant_names:
            fd.write("\t\t" + x + ",\n")
        fd.write("\t};\n\n")
    else:
        fd.write("\tenum ShaderVariant { DEFAULT };\n\n")
        defvariant = "=DEFAULT"

    if header_data.specialization_names:
        fd.write("\tenum Specializations {\n")
        counter = 0
        for x in header_data.specialization_names:
            fd.write("\t\t" + x.upper() + "=" + str(1 << counter) + ",\n")
            counter += 1
        fd.write("\t};\n\n")

    for i in range(len(header_data.specialization_names)):
        defval = header_data.specialization_values[i].strip()
        if defval.upper() == "TRUE" or defval == "1":
            defspec |= 1 << i

    fd.write(
        "\t_FORCE_INLINE_ void version_bind_shader(RID p_version,ShaderVariant p_variant"
        + defvariant
        + ",uint64_t p_specialization="
        + str(defspec)
        + ") { _version_bind_shader(p_version,p_variant,p_specialization); }\n\n"
    )

    if header_data.uniforms:
        fd.write(
            "\t_FORCE_INLINE_ int version_get_uniform(Uniforms p_uniform,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { return _version_get_uniform(p_uniform,p_version,p_variant,p_specialization); }\n\n"
        )

        fd.write(
            "\t#define _FU if (version_get_uniform(p_uniform,p_version,p_variant,p_specialization)<0) return; \n\n "
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, float p_value,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform1f(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_value); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, double p_value,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform1f(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_value); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, uint8_t p_value,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform1i(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_value); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, int8_t p_value,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform1i(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_value); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, uint16_t p_value,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform1i(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_value); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, int16_t p_value,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform1i(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_value); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, uint32_t p_value,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform1i(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_value); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, int32_t p_value,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform1i(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_value); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Color& p_color,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU GLfloat col[4]={p_color.r,p_color.g,p_color.b,p_color.a}; glUniform4fv(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),1,col); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Vector2& p_vec2,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU GLfloat vec2[2]={float(p_vec2.x),float(p_vec2.y)}; glUniform2fv(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),1,vec2); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Size2i& p_vec2,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU GLint vec2[2]={GLint(p_vec2.x),GLint(p_vec2.y)}; glUniform2iv(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),1,vec2); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Vector3& p_vec3,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU GLfloat vec3[3]={float(p_vec3.x),float(p_vec3.y),float(p_vec3.z)}; glUniform3fv(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),1,vec3); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, float p_a, float p_b,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform2f(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_a,p_b); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, float p_a, float p_b, float p_c,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform3f(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_a,p_b,p_c); }\n\n"
        )
        fd.write(
            "\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, float p_a, float p_b, float p_c, float p_d,RID p_version,ShaderVariant p_variant"
            + defvariant
            + ",uint64_t p_specialization="
            + str(defspec)
            + ") { _FU glUniform4f(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),p_a,p_b,p_c,p_d); }\n\n"
        )

        fd.write(
            """\t_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Transform3D& p_transform,RID p_version,ShaderVariant p_variant"""
            + defvariant
            + """,uint64_t p_specialization="""
            + str(defspec)
            + """) {  _FU

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

                    glUniformMatrix4fv(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),1,false,matrix);

        }

        """
        )

        fd.write(
            """_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Transform2D& p_transform,RID p_version,ShaderVariant p_variant"""
            + defvariant
            + """,uint64_t p_specialization="""
            + str(defspec)
            + """) {  _FU

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

            glUniformMatrix4fv(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),1,false,matrix);

        }

        """
        )

        fd.write(
            """_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const CameraMatrix& p_matrix,RID p_version,ShaderVariant p_variant"""
            + defvariant
            + """,uint64_t p_specialization="""
            + str(defspec)
            + """) {  _FU

            GLfloat matrix[16];

            for (int i=0;i<4;i++) {
                for (int j=0;j<4;j++) {
                    matrix[i*4+j]=p_matrix.matrix[i][j];
                }
            }

            glUniformMatrix4fv(version_get_uniform(p_uniform,p_version,p_variant,p_specialization),1,false,matrix);
    }"""
        )

        fd.write("\n\n#undef _FU\n\n\n")

    fd.write("protected:\n\n")

    fd.write("\tvirtual void _init() override {\n\n")

    if header_data.uniforms:

        fd.write("\t\tstatic const char* _uniform_strings[]={\n")
        if header_data.uniforms:
            for x in header_data.uniforms:
                fd.write('\t\t\t"' + x + '",\n')
        fd.write("\t\t};\n\n")
    else:
        fd.write("\t\tstatic const char **_uniform_strings=nullptr;\n")

    variant_count = 1
    if len(header_data.variant_defines) > 0:

        fd.write("\t\tstatic const char* _variant_defines[]={\n")
        for x in header_data.variant_defines:
            fd.write('\t\t\t"' + x + '",\n')
        fd.write("\t\t};\n\n")
        variant_count = len(header_data.variant_defines)
    else:
        fd.write("\t\tstatic const char **_variant_defines[]={" "};\n")

    if header_data.texunits:
        fd.write("\t\tstatic TexUnitPair _texunit_pairs[]={\n")
        for x in header_data.texunits:
            fd.write('\t\t\t{"' + x[0] + '",' + x[1] + "},\n")
        fd.write("\t\t};\n\n")
    else:
        fd.write("\t\tstatic TexUnitPair *_texunit_pairs=nullptr;\n")

    if header_data.ubos:
        fd.write("\t\tstatic UBOPair _ubo_pairs[]={\n")
        for x in header_data.ubos:
            fd.write('\t\t\t{"' + x[0] + '",' + x[1] + "},\n")
        fd.write("\t\t};\n\n")
    else:
        fd.write("\t\tstatic UBOPair *_ubo_pairs=nullptr;\n")

    if header_data.specialization_names:
        fd.write("\t\tstatic Specialization _spec_pairs[]={\n")
        for i in range(len(header_data.specialization_names)):
            defval = header_data.specialization_values[i].strip()
            if defval.upper() == "TRUE" or defval == "1":
                defal = "true"
            else:
                defval = "false"

            fd.write('\t\t\t{"' + header_data.specialization_names[i] + '",' + defval + "},\n")
        fd.write("\t\t};\n\n")
    else:
        fd.write("\t\tstatic Specialization *_spec_pairs=nullptr;\n")

    fd.write("\t\tstatic const char _vertex_code[]={\n")
    for x in header_data.vertex_lines:
        for c in x:
            fd.write(str(ord(c)) + ",")

        fd.write(str(ord("\n")) + ",")
    fd.write("\t\t0};\n\n")

    fd.write("\t\tstatic const char _fragment_code[]={\n")
    for x in header_data.fragment_lines:
        for c in x:
            fd.write(str(ord(c)) + ",")

        fd.write(str(ord("\n")) + ",")
    fd.write("\t\t0};\n\n")

    fd.write(
        '\t\t_setup(_vertex_code,_fragment_code,"'
        + out_file_class
        + '",'
        + str(len(header_data.uniforms))
        + ",_uniform_strings,"
        + str(len(header_data.ubos))
        + ",_ubo_pairs,"
        + str(len(header_data.texunits))
        + ",_texunit_pairs,"
        + str(len(header_data.specialization_names))
        + ",_spec_pairs,"
        + str(variant_count)
        + ",_variant_defines);\n"
    )

    fd.write("\t}\n\n")

    fd.write("};\n\n")
    fd.write("#endif\n\n")
    fd.close()


def build_gles3_headers(target, source, env):
    for x in source:
        build_gles3_header(str(x), include="drivers/gles3/shader_gles3.h", class_suffix="GLES3", output_attribs=True)


if __name__ == "__main__":
    subprocess_main(globals())
