# SVG Optimization

Solve Godot built-in SVG parsing problem, some complex SVG files can not be displayed.

## Solution ideas

Use https://github.com/thorvg/thorvg libraries parse, Godot itself on the limited support SVG Tiny specification, won't cause complex SVG parsing.
Use https://github.com/sammycage/lunasvg parsing library instead.


### Modify the record

Replace the parsing file in the godot\modules\svg folder and import the lunasvg parsing library.
Core code modification:

~~~
Error ImageLoaderSVG::create_image_from_utf8_buffer(Ref<Image> p_image, const uint8_t *p_buffer, int p_buffer_size, float p_scale, bool p_upsample) {
	ERR_FAIL_COND_V_MSG(Math::is_zero_approx(p_scale), ERR_INVALID_PARAMETER, "ImageLoaderSVG: Can't load SVG with a scale of 0.");

	auto document = lunasvg::Document::loadFromData((const char *)p_buffer, p_buffer_size);

	uint32_t width = document->width(), height = document->height();

	width *= p_scale;
	height *= p_scale;

	auto bitmap = document->renderToBitmap(width, height, 0x00000000);

	Vector<uint8_t> result;
	result.resize(width * height * 4);

	uint32_t *buffer = (uint32_t *)bitmap.data();

	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			uint32_t n = buffer[y * width + x];
			const size_t offset = sizeof(uint32_t) * width * y + sizeof(uint32_t) * x;
			result.write[offset + 0] = (n >> 16) & 0xff;
			result.write[offset + 1] = (n >> 8) & 0xff;
			result.write[offset + 2] = n & 0xff;
			result.write[offset + 3] = (n >> 24) & 0xff;
		}
	}

	p_image->set_data(width, height, false, Image::FORMAT_RGBA8, result);

	return OK;
}
~~~


TODO

Add large resolution svg cache
