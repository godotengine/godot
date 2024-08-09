/* clang-format off */
#version 300 es

#[modes]

mode_default = 

#[specializations]

USE_EXTERNAL_SAMPLER = false
COPY_DEPTHMAP = false

#[vertex]

layout(location = 0) in vec2 vertex_attrib;

out vec2 uv_interp;


void main() {    
	uv_interp = vertex_attrib * 0.5 + 0.5;    
	gl_Position = vec4(vertex_attrib, 1.0, 1.0);
}

/* clang-format off */
#[fragment]

layout(location = 0) out vec4 frag_color;
in vec2 uv_interp;

/* clang-format on */
uniform sampler2D source; // texunit:0
#ifdef USE_EXTERNAL_SAMPLER
uniform samplerExternalOES sourceFrag; // texunit:1
#else
uniform sampler2D sourceFrag; // texunit:1
#endif

uniform float max_depth;
uniform int use_depth;
uniform int show_depthmap;

float depthInMillimeters(in sampler2D depthTexture, in vec2 depthUV) {
	// Depth is packed into the red and green components of its texture.
	// The texture is a normalized format, storing millimeters.
	vec2 packedDepthAndVisibility = texture(depthTexture, depthUV).xy;
	return dot(packedDepthAndVisibility.xy, vec2(255.0, 256.0 * 255.0));
}

float inverseLerp(float value, float minBound, float maxBound) {
	return clamp((value - minBound) / (maxBound - minBound), 0.0, 1.0);
}

void main() {
    vec4 color = texture(sourceFrag, uv_interp);
    if (use_depth == 1) {
        float depth_mm = depthInMillimeters(source, uv_interp);
        float depth_meters = depth_mm * 0.001;
        float normalized_depth = inverseLerp(depth_meters, 0.0, 1.0);

        gl_FragDepth = normalized_depth;

        if (show_depthmap == 1) {
            float displayed_normalized_depth = inverseLerp(depth_meters, 0.0, max_depth);
            color = vec4(displayed_normalized_depth, 0.0, 0.0, 1.0);
        }
    }
    else {
        gl_FragDepth = 1.0;
    }

	frag_color = color;
}
