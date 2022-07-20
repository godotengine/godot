layout(push_constant, std430) uniform Params {
	uint face_size;
	uint face_id; // only used in raster shader
}
params;

#define M_PI 3.14159265359

void get_dir_0(out vec3 dir, in float u, in float v) {
	dir[0] = 1.0;
	dir[1] = v;
	dir[2] = -u;
}

void get_dir_1(out vec3 dir, in float u, in float v) {
	dir[0] = -1.0;
	dir[1] = v;
	dir[2] = u;
}

void get_dir_2(out vec3 dir, in float u, in float v) {
	dir[0] = u;
	dir[1] = 1.0;
	dir[2] = -v;
}

void get_dir_3(out vec3 dir, in float u, in float v) {
	dir[0] = u;
	dir[1] = -1.0;
	dir[2] = v;
}

void get_dir_4(out vec3 dir, in float u, in float v) {
	dir[0] = u;
	dir[1] = v;
	dir[2] = 1.0;
}

void get_dir_5(out vec3 dir, in float u, in float v) {
	dir[0] = -u;
	dir[1] = v;
	dir[2] = -1.0;
}

float calcWeight(float u, float v) {
	float val = u * u + v * v + 1.0;
	return val * sqrt(val);
}
