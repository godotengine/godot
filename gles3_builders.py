"""Functions used to generate source files during build time"""

import os.path
from typing import List, Optional, Tuple

from methods import generated_wrapper, print_error


class GLES3HeaderStruct:
    def __init__(self) -> None:
        self.vertex_lines: List[str] = []
        self.fragment_lines: List[str] = []
        self.uniforms: List[str] = []
        self.fbos: List[str] = []
        self.texunits: List[Tuple[str, str]] = []
        self.texunit_names: List[str] = []
        self.ubos: List[Tuple[str, str]] = []
        self.ubo_names: List[str] = []
        self.feedbacks: List[Tuple[str, str]] = []

        self.vertex_included_files: List[str] = []
        self.fragment_included_files: List[str] = []

        self.reading: str = ""
        self.line_offset: int = 0
        self.vertex_offset: int = 0
        self.fragment_offset: int = 0
        self.variant_defines: List[str] = []
        self.variant_names: List[str] = []
        self.specialization_names: List[str] = []
        self.specialization_values: List[str] = []


def include_file_in_gles3_header(filename: str, header_data: GLES3HeaderStruct, depth: int) -> GLES3HeaderStruct:
    with open(filename, "r", encoding="utf-8") as fs:
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

                included_file = os.path.relpath(os.path.dirname(filename) + "/" + includeline).replace("\\", "/")
                if included_file not in header_data.vertex_included_files and header_data.reading == "vertex":
                    header_data.vertex_included_files += [included_file]
                    if include_file_in_gles3_header(included_file, header_data, depth + 1) is None:
                        print_error(f'In file "{filename}": #include "{includeline}" could not be found!"')
                elif included_file not in header_data.fragment_included_files and header_data.reading == "fragment":
                    header_data.fragment_included_files += [included_file]
                    if include_file_in_gles3_header(included_file, header_data, depth + 1) is None:
                        print_error(f'In file "{filename}": #include "{includeline}" could not be found!"')

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

                    if x not in header_data.texunit_names:
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

                    if x not in header_data.ubo_names:
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

                    if x not in header_data.uniforms:
                        header_data.uniforms += [x]

            if (line.strip().find("out ") == 0 or line.strip().find("flat ") == 0) and line.find("tfb:") != -1:
                uline = line.replace("flat ", "")
                uline = uline.replace("out ", "")
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

    return header_data


def build_gles3_header(
    filename: str,
    include: str,
    class_suffix: str,
    optional_output_filename: Optional[str] = None,
    header_data: Optional[GLES3HeaderStruct] = None,
) -> None:
    header_data = header_data or GLES3HeaderStruct()
    include_file_in_gles3_header(filename, header_data, 0)

    if optional_output_filename is None:
        out_file = filename + ".gen.h"
    else:
        out_file = optional_output_filename

    with generated_wrapper(out_file, suffix=class_suffix) as file:
        defspec = 0
        defvariant = ""

        out_file_base = os.path.basename(out_file).split(".")[0]
        out_file_class = out_file_base.title().replace("_", "") + "Shader" + class_suffix

        file.write(
            f"""\
#include "{include}"

class {out_file_class} : public Shader{class_suffix} {{
public:
"""
        )
        if header_data.uniforms:
            enums = "\n\t\t".join([f"{u.upper()}," for u in header_data.uniforms])
            file.write(
                f"""\
	enum Uniforms {{
		{enums}
	}};

"""
            )

        if header_data.variant_names:
            enums = "\n\t\t".join([f"{n.upper()}," for n in header_data.variant_names])
            file.write(
                f"""\
	enum ShaderVariant {{
		{enums}
	}};

"""
            )
        else:
            file.write("\tenum ShaderVariant { DEFAULT };\n\n")
            defvariant = " = DEFAULT"

        if header_data.specialization_names:
            enums = "\n\t\t".join([f"{n.upper()} = {i + 1}," for i, n in enumerate(header_data.specialization_names)])
            file.write(
                f"""\
	enum Specializations {{
		{enums}
	}};

"""
            )

        for i in range(len(header_data.specialization_names)):
            defval = header_data.specialization_values[i].strip()
            if defval.upper() == "TRUE" or defval == "1":
                defspec |= 1 << i

        file.write(
            f"""\
	_FORCE_INLINE_ bool version_bind_shader(RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		return _version_bind_shader(p_version, p_variant, p_specialization);
	}}
"""
        )

        if header_data.uniforms:
            file.write(
                f"""\
	_FORCE_INLINE_ int version_get_uniform(Uniforms p_uniform, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		return _version_get_uniform(p_uniform, p_version, p_variant, p_specialization);
	}}

#define _FU\\
	if (version_get_uniform(p_uniform, p_version, p_variant, p_specialization) < 0) {{\\
		return;\\
	}}

	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, float p_value, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform1f(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_value);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, double p_value, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform1f(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_value);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, int8_t p_value, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform1i(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_value);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, int16_t p_value, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform1i(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_value);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, int32_t p_value, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform1i(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_value);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, uint8_t p_value, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform1ui(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_value);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, uint16_t p_value, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform1ui(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_value);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, uint32_t p_value, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform1ui(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_value);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Color &p_color, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		GLfloat col[4] = {{ p_color.r, p_color.g, p_color.b, p_color.a }};
		glUniform4fv(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), 1, col);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Vector2 &p_vector2, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		GLfloat vec2[2] = {{ float(p_vector2.x), float(p_vector2.y) }};
		glUniform2fv(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), 1, vec2);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Vector2i &p_vector2i, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		GLint vec2i[2] = {{ GLint(p_vector2i.x), GLint(p_vector2i.y) }};
		glUniform2iv(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), 1, vec2i);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Vector3 &p_vector3, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		GLfloat vec3[3] = {{ float(p_vector3.x), float(p_vector3.y), float(p_vector3.z) }};
		glUniform3fv(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), 1, vec3);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Vector4 &p_vector4, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		GLfloat vec4[4] = {{ float(p_vector4.x), float(p_vector4.y), float(p_vector4.z), float(p_vector4.w) }};
		glUniform4fv(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), 1, vec4);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, float p_a, float p_b, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform2f(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_a, p_b);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, float p_a, float p_b, float p_c, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform3f(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_a, p_b, p_c);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, float p_a, float p_b, float p_c, float p_d, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		glUniform4f(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), p_a, p_b, p_c, p_d);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Transform2D &p_transform2d, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU; /* build a 16x16 matrix */
		const Transform2D &tr = p_transform2d;
		GLfloat matrix[16] = {{
			(GLfloat)tr.columns[0][0],
			(GLfloat)tr.columns[0][1],
			(GLfloat)0,
			(GLfloat)0,
			(GLfloat)tr.columns[1][0],
			(GLfloat)tr.columns[1][1],
			(GLfloat)0,
			(GLfloat)0,
			(GLfloat)0,
			(GLfloat)0,
			(GLfloat)1,
			(GLfloat)0,
			(GLfloat)tr.columns[2][0],
			(GLfloat)tr.columns[2][1],
			(GLfloat)0,
			(GLfloat)1
		}};
		glUniformMatrix4fv(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), 1, false, matrix);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Transform3D &p_transform3d, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU; /* build a 16x16 matrix */
		const Transform3D &tr = p_transform3d;
		GLfloat matrix[16] = {{
			(GLfloat)tr.basis.rows[0][0],
			(GLfloat)tr.basis.rows[1][0],
			(GLfloat)tr.basis.rows[2][0],
			(GLfloat)0,
			(GLfloat)tr.basis.rows[0][1],
			(GLfloat)tr.basis.rows[1][1],
			(GLfloat)tr.basis.rows[2][1],
			(GLfloat)0,
			(GLfloat)tr.basis.rows[0][2],
			(GLfloat)tr.basis.rows[1][2],
			(GLfloat)tr.basis.rows[2][2],
			(GLfloat)0,
			(GLfloat)tr.origin.x,
			(GLfloat)tr.origin.y,
			(GLfloat)tr.origin.z,
			(GLfloat)1,
		}};
		glUniformMatrix4fv(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), 1, false, matrix);
	}}
	_FORCE_INLINE_ void version_set_uniform(Uniforms p_uniform, const Projection &p_projection, RID p_version, ShaderVariant p_variant{defvariant}, uint64_t p_specialization = {defspec}) {{
		_FU;
		GLfloat matrix[16];
		for (int i = 0; i < 4; i++) {{
			for (int j = 0; j < 4; j++) {{
				matrix[i * 4 + j] = p_projection.columns[i][j];
			}}
		}}
		glUniformMatrix4fv(version_get_uniform(p_uniform, p_version, p_variant, p_specialization), 1, false, matrix);
	}}

#undef _FU
"""
            )

        file.write(
            """\

protected:
	virtual void _init() override {
"""
        )

        if header_data.uniforms:
            uniforms = "\n\t\t\t".join([f'"{u}",' for u in header_data.uniforms])
            file.write(
                f"""\
		static const char *_uniform_strings[] = {{
			{uniforms}
		}};
"""
            )
        else:
            file.write("\t\tstatic const char **_uniform_strings = nullptr;\n")

        variant_count = len(header_data.variant_defines)
        if variant_count > 0:
            variants = "\n\t\t\t".join([f'"{v}",' for v in header_data.variant_defines])
            file.write(
                f"""\
		static const char *_variant_defines[] = {{
			{variants}
		}};
"""
            )
        else:
            file.write('\t\tstatic const char **_variant_defines[] = {" "};\n')
            variant_count = 1

        if header_data.texunits:
            texunits = "\n\t\t\t".join([f'{{ "{t[0]}", {t[1]} }},' for t in header_data.texunits])
            file.write(
                f"""\
		static TexUnitPair _texunit_pairs[] = {{
			{texunits}
		}};
"""
            )
        else:
            file.write("\t\tstatic TexUnitPair *_texunit_pairs = nullptr;\n")

        if header_data.ubos:
            ubos = "\n\t\t\t".join([f'{{ "{u[0]}", {u[1]} }},' for u in header_data.ubos])
            file.write(
                f"""\
		static UBOPair _ubo_pairs[] = {{
			{ubos}
		}};
"""
            )
        else:
            file.write("\t\tstatic UBOPair *_ubo_pairs = nullptr;\n")

        if header_data.specialization_names:
            lines = []
            for idx, name in enumerate(header_data.specialization_names):
                defval = header_data.specialization_values[idx].strip()
                defval = "true" if (defval.upper() == "TRUE" or defval == "1") else "false"
                lines.append(f'{{ "{name}", {defval} }},')
            specializations = "\n\t\t\t".join(lines)
            file.write(
                f"""\
		static Specialization _spec_pairs[] = {{
			{specializations}
		}};
"""
            )
        else:
            file.write("\t\tstatic Specialization *_spec_pairs = nullptr;\n")

        if header_data.feedbacks:
            lines = []
            for name, spec in header_data.feedbacks:
                if spec in header_data.specialization_names:
                    lines.append(f'{{ "{name}", {1 << header_data.specialization_names.index(spec)} }},')
                else:
                    lines.append(f'{{ "{name}", 0 }},')
            feedbacks = "\n\t\t\t".join(lines)
            file.write(
                f"""\
		static const Feedback _feedbacks[] = {{
			{feedbacks}
		}};
"""
            )
        else:
            file.write("\t\tstatic const Feedback *_feedbacks = nullptr;\n")

        vertex_code = [str(ord(char)) for char in "\n".join(header_data.vertex_lines)]
        file.write(
            f"""\
		static const char _vertex_code[] = {{
			{", ".join(vertex_code)}, 10, 0
		}};
"""
        )

        fragment_code = [str(ord(char)) for char in "\n".join(header_data.fragment_lines)]
        file.write(
            f"""\
		static const char _fragment_code[] = {{
			{", ".join(fragment_code)}, 10, 0
		}};
"""
        )

        file.write(
            f'\n\t\t_setup(_vertex_code, _fragment_code, "{out_file_class}", '
            + f"{len(header_data.uniforms)}, _uniform_strings, "
            + f"{len(header_data.ubos)}, _ubo_pairs, "
            + f"{len(header_data.feedbacks)}, _feedbacks, "
            + f"{len(header_data.texunits)}, _texunit_pairs, "
            + f"{len(header_data.specialization_names)}, _spec_pairs, "
            + f"{variant_count}, _variant_defines);\n"
        )

        file.write("\t}\n};")


def build_gles3_headers(target, source, env) -> None:
    for x in source:
        build_gles3_header(str(x), include="drivers/gles3/shader_gles3.h", class_suffix="GLES3")
