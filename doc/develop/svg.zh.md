# SVG优化

解决Godot自带SVG解析问题，部分复杂SVG文件无法显示。

## 解决思路

Godot自身使用 https://github.com/thorvg/thorvg 解析库解析，其支持SVG Tiny规范有限制，导致部分复杂SVG解析不了。
使用https://github.com/sammycage/lunasvg解析库代替。


### 修改记录

替换godot\modules\svg文件夹下解析文件，导入lunasvg解析库。
核心代码修改:

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

增加大分辨率图片cache