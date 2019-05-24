#version 310 es

readonly coherent uniform layout(set = 0, binding = 0, rgba32f) highp image2D image1;
readonly uniform layout(set = 0, binding = 2, rgba16f) highp image2D image2;
writeonly coherent uniform layout(set = 0, binding = 1, rgba32f) highp image2D image3;
writeonly uniform layout(set = 0, binding = 3, rgba16f) highp image2D image4;

flat in layout(location = 0) highp ivec2 in_coords;
out layout(location = 0) highp vec4 out_color;

highp vec4 image_load(readonly coherent highp image2D image, highp ivec2 coords)
{
	return imageLoad(image, in_coords);
}

void image_store(writeonly coherent highp image2D image, highp ivec2 coords, highp vec4 data)
{
	imageStore(image, in_coords, data);
}

void main()
{
	highp vec4 read1 = image_load(image1, in_coords);
	highp vec4 read2 = image_load(image2, in_coords);
	
	image_store(image3, in_coords, read1*0.5);
	image_store(image4, in_coords, read2*2.0);

	out_color = vec4(0.0);
}
