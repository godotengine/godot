Module["compileGLSLZeroCopy"] = function (
	p_glsl,
	p_shader_stage,
	p_gen_debug,
	p_spirv_version,
) {
	const gen_debug = Boolean(p_gen_debug);

	let shader_stage_int; // EShLanguage
	switch (p_shader_stage) {
		case "vertex":
			shader_stage_int = 0;
			break;
		case "fragment":
			shader_stage_int = 4;
			break;
		case "compute":
			shader_stage_int = 5;
			break;
		default:
			throw new Error(
				"shader_stage must be 'vertex', 'fragment', or 'compute'.",
			);
	}

	const spirv_version = p_spirv_version || "1.0";
	let spirv_version_int; // EShTargetLanguageVersion
	switch (spirv_version) {
		case "1.0":
			spirv_version_int = (1 << 16) | (0 << 8);
			break;
		case "1.1":
			spirv_version_int = (1 << 16) | (1 << 8);
			break;
		case "1.2":
			spirv_version_int = (1 << 16) | (2 << 8);
			break;
		case "1.3":
			spirv_version_int = (1 << 16) | (3 << 8);
			break;
		case "1.4":
			spirv_version_int = (1 << 16) | (4 << 8);
			break;
		case "1.5":
			spirv_version_int = (1 << 16) | (5 << 8);
			break;
		default:
			throw new Error("spirv_version must be '1.0' ~ '1.5'.");
	}

	const p_output = Module["_malloc"](4);
	const p_output_len = Module["_malloc"](4);
	const id = Module["ccall"](
		"convert_glsl_to_spirv",
		"number",
		["string", "number", "boolean", "number", "number", "number"],
		[
			p_glsl,
			shader_stage_int,
			gen_debug,
			spirv_version_int,
			p_output,
			p_output_len,
		],
	);
	const output = getValue(p_output, "i32");
	const output_len = getValue(p_output_len, "i32");
	Module["_free"](p_output);
	Module["_free"](p_output_len);

	if (id === 0) {
		throw new Error("GLSL compilation failed");
	}

	const ret = {};
	const outputIndexU32 = output / 4;
	ret["data"] = Module["HEAPU32"].subarray(
		outputIndexU32,
		outputIndexU32 + output_len,
	);
	ret["free"] = function () {
		Module["_destroy_output_buffer"](id);
	};

	return ret;
};

Module["compileGLSL"] = function (
	p_glsl,
	p_shader_stage,
	p_gen_debug,
	p_spirv_version,
) {
	const compiled = Module["compileGLSLZeroCopy"](
		p_glsl,
		p_shader_stage,
		p_gen_debug,
		p_spirv_version,
	);
	const ret = compiled["data"].slice();
	compiled["free"]();
	return ret;
};
