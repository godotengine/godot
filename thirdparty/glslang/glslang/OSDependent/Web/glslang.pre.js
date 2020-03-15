Module['compileGLSLZeroCopy'] = function(glsl, shader_stage, gen_debug) {
    gen_debug = !!gen_debug;

    var shader_stage_int;
    if (shader_stage === 'vertex') {
        shader_stage_int = 0;
    } else if (shader_stage === 'fragment') {
        shader_stage_int = 4;
    } else if (shader_stage === 'compute') {
        shader_stage_int = 5;
    } else {
        throw new Error("shader_stage must be 'vertex', 'fragment', or 'compute'");
    }

    var p_output = Module['_malloc'](4);
    var p_output_len = Module['_malloc'](4);
    var id = ccall('convert_glsl_to_spirv',
        'number',
        ['string', 'number', 'boolean', 'number', 'number'],
        [glsl, shader_stage_int, gen_debug, p_output, p_output_len]);
    var output = getValue(p_output, 'i32');
    var output_len = getValue(p_output_len, 'i32');
    Module['_free'](p_output);
    Module['_free'](p_output_len);

    if (id === 0) {
        throw new Error('GLSL compilation failed');
    }

    var ret = {};
    var outputIndexU32 = output / 4;
    ret['data'] = Module['HEAPU32'].subarray(outputIndexU32, outputIndexU32 + output_len);
    ret['free'] = function() {
        Module['_destroy_output_buffer'](id);
    };

    return ret;
};

Module['compileGLSL'] = function(glsl, shader_stage, gen_debug) {
    var compiled = Module['compileGLSLZeroCopy'](glsl, shader_stage, gen_debug);
    var ret = compiled['data'].slice()
    compiled['free']();
    return ret;
};
