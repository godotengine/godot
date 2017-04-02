[vertex]


layout(location=0) in highp vec4 vertex_attrib;
layout(location=4) in vec2 uv_in;

out vec2 uv_interp;
out vec2 pos_interp;

void main() {

	uv_interp = uv_in;
	gl_Position = vertex_attrib;
	pos_interp.xy=gl_Position.xy;
}

[fragment]


in vec2 uv_interp;
in vec2 pos_interp;

uniform sampler2D source_diffuse; //texunit:0
uniform sampler2D source_normal_roughness; //texunit:1
uniform sampler2D source_depth; //texunit:2

uniform float camera_z_near;
uniform float camera_z_far;

uniform vec2 viewport_size;
uniform vec2 pixel_size;

uniform float filter_mipmap_levels;

uniform mat4 inverse_projection;
uniform mat4 projection;

uniform int num_steps;
uniform float depth_tolerance;
uniform float distance_fade;
uniform float acceleration;

layout(location = 0) out vec4 frag_color;


vec2 view_to_screen(vec3 view_pos,out float w) {
    vec4 projected = projection * vec4(view_pos, 1.0);
    projected.xyz /= projected.w;
    projected.xy = projected.xy * 0.5 + 0.5;
    w=projected.w;
    return projected.xy;
}



#define M_PI 3.14159265359


void main() {


	////

	vec4 diffuse = texture( source_diffuse,  uv_interp );
	vec4 normal_roughness = texture( source_normal_roughness, uv_interp);

	vec3 normal;

	normal = normal_roughness.xyz*2.0-1.0;

	float roughness = normal_roughness.w;

	float depth_tex = texture(source_depth,uv_interp).r;

	vec4 world_pos = inverse_projection * vec4( uv_interp*2.0-1.0, depth_tex*2.0-1.0, 1.0 );
	vec3 vertex = world_pos.xyz/world_pos.w;

	vec3 view_dir = normalize(vertex);
	vec3 ray_dir = normalize(reflect(view_dir, normal));

	if (dot(ray_dir,normal)<0.001) {
		frag_color=vec4(0.0);
		return;
	}
	//ray_dir = normalize(view_dir - normal * dot(normal,view_dir) * 2.0);

	//ray_dir = normalize(vec3(1,1,-1));


	////////////////


	//make ray length and clip it against the near plane (don't want to trace beyond visible)
	float ray_len = (vertex.z + ray_dir.z * camera_z_far) > -camera_z_near ? (-camera_z_near - vertex.z) / ray_dir.z : camera_z_far;
	vec3 ray_end = vertex + ray_dir*ray_len;

	float w_begin;
	vec2 vp_line_begin = view_to_screen(vertex,w_begin);
	float w_end;
	vec2 vp_line_end = view_to_screen( ray_end, w_end);
	vec2 vp_line_dir = vp_line_end-vp_line_begin;

	//we need to interpolate w along the ray, to generate perspective correct reflections

	w_begin = 1.0/w_begin;
	w_end = 1.0/w_end;


	float z_begin = vertex.z*w_begin;
	float z_end = ray_end.z*w_end;

	vec2 line_begin = vp_line_begin/pixel_size;
	vec2 line_dir = vp_line_dir/pixel_size;
	float z_dir = z_end - z_begin;
	float w_dir = w_end - w_begin;


	// clip the line to the viewport edges

	float scale_max_x = min(1.0, 0.99 * (1.0 - vp_line_begin.x) / max(1e-5, vp_line_dir.x));
	float scale_max_y = min(1.0, 0.99 * (1.0 - vp_line_begin.y) / max(1e-5, vp_line_dir.y));
	float scale_min_x = min(1.0, 0.99 * vp_line_begin.x / max(1e-5, -vp_line_dir.x));
	float scale_min_y = min(1.0, 0.99 * vp_line_begin.y / max(1e-5, -vp_line_dir.y));
	float line_clip = min(scale_max_x, scale_max_y) * min(scale_min_x, scale_min_y);
	line_dir *= line_clip;
	z_dir *= line_clip;
	w_dir *=line_clip;

	//clip z and w advance to line advance
	vec2 line_advance = normalize(line_dir); //down to pixel
	float step_size = length(line_advance)/length(line_dir);
	float z_advance = z_dir*step_size; // adapt z advance to line advance
	float w_advance = w_dir*step_size; // adapt w advance to line advance

	//make line advance faster if direction is closer to pixel edges (this avoids sampling the same pixel twice)
	float advance_angle_adj = 1.0/max(abs(line_advance.x),abs(line_advance.y));
	line_advance*=advance_angle_adj; // adapt z advance to line advance
	z_advance*=advance_angle_adj;
	w_advance*=advance_angle_adj;

	vec2 pos = line_begin;
	float z = z_begin;
	float w = w_begin;
	float z_from=z/w;
	float z_to=z_from;
	float depth;
	vec2 prev_pos=pos;

	bool found=false;

	//if acceleration > 0, distance between pixels gets larger each step. This allows covering a larger area
	float accel=1.0+acceleration;
	float steps_taken=0.0;

	for(int i=0;i<num_steps;i++) {

		pos+=line_advance;
		z+=z_advance;
		w+=w_advance;

		//convert to linear depth
		depth = texture(source_depth, pos*pixel_size).r * 2.0 - 1.0;
		depth = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - depth * (camera_z_far - camera_z_near));
		depth=-depth;

		z_from = z_to;
		z_to = z/w;

		if (depth>z_to) {
			//if depth was surpassed
			if (depth<=max(z_to,z_from)+depth_tolerance) {
				//check the depth tolerance
				found=true;
			}
			break;
		}

		steps_taken+=1.0;
		prev_pos=pos;
		z_advance*=accel;
		w_advance*=accel;
		line_advance*=accel;
	}




	if (found) {

		float margin_blend=1.0;


		vec2 margin = vec2((viewport_size.x+viewport_size.y)*0.5*0.05); //make a uniform margin
		if (any(bvec4(lessThan(pos,-margin),greaterThan(pos,viewport_size+margin)))) {
			//clip outside screen + margin
			frag_color=vec4(0.0);
			return;
		}

		{
			//blend fading out towards external margin
			vec2 margin_grad = mix(pos-viewport_size,-pos,lessThan(pos,vec2(0.0)));
			margin_blend = 1.0-smoothstep(0.0,margin.x,max(margin_grad.x,margin_grad.y));
			//margin_blend=1.0;

		}

		vec2 final_pos;
		float grad;

#ifdef SMOOTH_ACCEL
		//if the distance between point and prev point is >1, then take some samples in the middle for smoothing out the image
		vec2 blend_dir = pos - prev_pos;
		float steps = min(8.0,length(blend_dir));
		if (steps>2.0) {
			vec2 blend_step = blend_dir/steps;
			float blend_z = (z_to-z_from)/steps;
			vec2 new_pos;
			float subgrad=0.0;
			for(float i=0.0;i<steps;i++) {

				new_pos = (prev_pos+blend_step*i);
				float z = z_from+blend_z*i;

				depth = texture(source_depth, new_pos*pixel_size).r * 2.0 - 1.0;
				depth = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - depth * (camera_z_far - camera_z_near));
				depth=-depth;

				subgrad=i/steps;
				if (depth>z)
					break;
			}

			final_pos = new_pos;
			grad=(steps_taken+subgrad)/float(num_steps);

		} else {
#endif
			grad=steps_taken/float(num_steps);
			final_pos=pos;
#ifdef SMOOTH_ACCEL
		}

#endif



#ifdef REFLECT_ROUGHNESS


		vec4 final_color;
		//if roughness is enabled, do screen space cone tracing
		if (roughness > 0.001) {
			///////////////////////////////////////////////////////////////////////////////////////
			//use a blurred version (in consecutive mipmaps) of the screen to simulate roughness

			float gloss = 1.0-roughness;
			float cone_angle = roughness * M_PI * 0.5;
			vec2 cone_dir = final_pos - line_begin;
			float cone_len = length(cone_dir);
			cone_dir = normalize(cone_dir); //will be used normalized from now on
			float max_mipmap = filter_mipmap_levels - 1.0;
			float gloss_mult=gloss;

			float rem_alpha=1.0;
			final_color = vec4(0.0);

			for(int i=0;i<7;i++) {

				float op_len = 2.0 * tan(cone_angle) * cone_len; //opposite side of iso triangle
				float radius;
				{
					//fit to sphere inside cone (sphere ends at end of cone), something like this:
					// ___
					// \O/
					//  V
					//
					// as it avoids bleeding from beyond the reflection as much as possible. As a plus
					// it also makes the rough reflection more elongated.
					float a = op_len;
					float h = cone_len;
					float a2 = a * a;
					float fh2 = 4.0f * h * h;
					radius = (a * (sqrt(a2 + fh2) - a)) / (4.0f * h);
				}

				//find the place where screen must be sampled
				vec2 sample_pos = ( line_begin + cone_dir * (cone_len - radius) ) * pixel_size;
				//radius is in pixels, so it's natural that log2(radius) maps to the right mipmap for the amount of pixels
				float mipmap = clamp( log2( radius ), 0.0, max_mipmap );

				//mipmap = max(mipmap-1.0,0.0);
				//do sampling

				vec4 sample_color;
				{
					sample_color = textureLod(source_diffuse,sample_pos,mipmap);
				}

				//multiply by gloss
				sample_color.rgb*=gloss_mult;
				sample_color.a=gloss_mult;

				rem_alpha -= sample_color.a;
				if(rem_alpha < 0.0) {
					sample_color.rgb *= (1.0 - abs(rem_alpha));
				}

				final_color+=sample_color;

				if (final_color.a>=0.95) {
					// This code of accumulating gloss and aborting on near one
					// makes sense when you think of cone tracing.
					// Think of it as if roughness was 0, then we could abort on the first
					// iteration. For lesser roughness values, we need more iterations, but
					// each needs to have less influence given the sphere is smaller
					break;
				}

				cone_len-=radius*2.0; //go to next (smaller) circle.

				gloss_mult*=gloss;


			}
		} else {
			final_color = textureLod(source_diffuse,final_pos*pixel_size,0.0);
		}

		frag_color = vec4(final_color.rgb,pow(clamp(1.0-grad,0.0,1.0),distance_fade)*margin_blend);

#else
		frag_color = vec4(textureLod(source_diffuse,final_pos*pixel_size,0.0).rgb,pow(clamp(1.0-grad,0.0,1.0),distance_fade)*margin_blend);
#endif



	} else {
		frag_color = vec4(0.0,0.0,0.0,0.0);
	}



}

