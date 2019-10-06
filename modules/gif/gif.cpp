/*************************************************************************/
/*  gif.cpp                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "gif.h"

#include <gif_lib.h>

// Custom function to read the GIF data from a file.
int readFromFile(GifFileType *gif, GifByteType *data, int length) {

	FileAccess *f = (FileAccess *)(gif->UserData); // gif->UserData is the first parameter passed to DGifOpen.
	return f->get_buffer(data, length);
}

struct GIFBuffer { // Used to read the GIF data from a buffer.

	const uint8_t *data;
	int size;
	int index;

	GIFBuffer(const PoolByteArray &p_data) {
		data = p_data.read().ptr();
		size = p_data.size();
		index = 0;
	}

	GIFBuffer(const uint8_t *p_data, int p_size) {
		data = p_data;
		size = p_size;
		index = 0;
	}
};

// Custom function to read the GIF data from a buffer.
int readFromBuffer(GifFileType *gif, GifByteType *data, int length) {

	GIFBuffer *f = (GIFBuffer *)(gif->UserData);
	if (f->index + length > f->size)
		length = f->size - f->index;

	memcpy(data, &f->data[f->index], length);
	f->index += length;
	return length;
}

Error Gif::parse_error(Error err, const String &message) {

	ERR_PRINTS(message);
	return err;
}

Error Gif::gif_error(int err) {

	ERR_PRINT(GifErrorString(err));
	return FAILED;
}

Error Gif::_open(void *source, AnimatedImage::SourceType source_type) {

	ERR_FAIL_COND_V(gif != NULL, FAILED);

	int err = 0;
	gif = DGifOpen(source, source_type == AnimatedImage::FILE ? readFromFile : readFromBuffer, &err); // Loads the headers of the GIF.
	if (!gif)
		return gif_error(err);

	return OK;
}

#define RETURN_ERROR                  \
	{                                 \
		memdelete_arr(screen);        \
		return gif_error(gif->Error); \
	}

#define RGBA 4

Error Gif::_load_frames(Ref<AnimatedImage> &r_animated_image, int max_frames) {

	ERR_FAIL_COND_V(gif == NULL, FAILED);

	int image_size = gif->SWidth * gif->SHeight * RGBA;
	uint8_t *screen = memnew_arr(uint8_t, image_size); // Each frame of the GIF is drawn on this buffer.
	memset(screen, 0, image_size); // Clear the screen.

	int last_undisposed_frame = -1; // Keep track of the last frame that hasn't been cleared.
	GifRecordType recordType;

	GraphicsControlBlock gcb; // Store the additional information for the next frame.
	gcb.DisposalMode = DISPOSAL_UNSPECIFIED;
	gcb.UserInputFlag = false;
	gcb.DelayTime = 0;
	gcb.TransparentColor = NO_TRANSPARENT_COLOR;

	do { // Parse every record of the GIF.

		if (DGifGetRecordType(gif, &recordType) == GIF_ERROR)
			RETURN_ERROR;

		switch (recordType) {
			case EXTENSION_RECORD_TYPE: { // Record with additional info for the next image.

				int extFunction;
				GifByteType *extData;

				if (DGifGetExtension(gif, &extFunction, &extData) == GIF_ERROR)
					RETURN_ERROR;

				if (extData == NULL) {

					break;
				} else if (extFunction == GRAPHICS_EXT_FUNC_CODE) {

					if (DGifExtensionToGCB(extData[0], &extData[1], &gcb) == GIF_ERROR)
						RETURN_ERROR;
				}

				while (true) {

					if (DGifGetExtensionNext(gif, &extData) == GIF_ERROR)
						RETURN_ERROR;

					if (extData == NULL) {

						break;
					} else if (extFunction == GRAPHICS_EXT_FUNC_CODE) {

						if (DGifExtensionToGCB(extData[0], &extData[1], &gcb) == GIF_ERROR)
							RETURN_ERROR;
					}
				}

			} break;
			case IMAGE_DESC_RECORD_TYPE: { // Record with the image data.

				if (DGifGetImageHeader(gif) == GIF_ERROR)
					RETURN_ERROR;

				GifImageDesc &imageDesc = gif->Image;

				if (imageDesc.Width <= 0 || imageDesc.Width > Image::MAX_WIDTH ||
						imageDesc.Height <= 0 || imageDesc.Height > Image::MAX_HEIGHT) {

					RETURN_ERROR;
				}

				// Use the global colorMap if the frame doesn't include one.
				ColorMapObject *colorMap = imageDesc.ColorMap ? imageDesc.ColorMap : gif->SColorMap;

				int frame_size = imageDesc.Width * imageDesc.Height;
				GifByteType *rasterBits = memnew_arr(GifByteType, frame_size); // Array with the indices to the colorMap.

				if (imageDesc.Interlace) { // Unwrap the interlaced image.
					int interlacedOffset[] = { 0, 4, 2, 1 };
					int interlacedJumps[] = { 8, 8, 4, 2 };

					for (int i = 0; i < 4; i++) {

						for (int j = interlacedOffset[i]; j < imageDesc.Height; j += interlacedJumps[i]) {

							if (DGifGetLine(gif, rasterBits + j * imageDesc.Width, imageDesc.Width) == GIF_ERROR) {

								memdelete_arr(rasterBits);
								RETURN_ERROR;
							}
						}
					}
				} else {

					if (DGifGetLine(gif, rasterBits, frame_size) == GIF_ERROR) {

						memdelete_arr(rasterBits);
						RETURN_ERROR;
					}
				}

				// Each frame has a different size and offset.
				for (int y = 0; y < imageDesc.Height; y++) {

					for (int x = 0; x < imageDesc.Width; x++) {

						int color_map_index = rasterBits[y * imageDesc.Width + x];
						if (color_map_index == gcb.TransparentColor) // This pixel doesn't change the current content of the screen.
							continue;

						int write_y = y + imageDesc.Top;
						int write_x = x + imageDesc.Left;
						int write_index = (write_y * gif->SWidth + write_x) * RGBA;

						GifColorType color = colorMap->Colors[color_map_index];
						screen[write_index] = color.Red;
						screen[write_index + 1] = color.Green;
						screen[write_index + 2] = color.Blue;
						screen[write_index + 3] = 255;
					}
				}

				memdelete_arr(rasterBits);

				PoolByteArray frame_data;
				frame_data.resize(image_size);
				PoolByteArray::Write data_write = frame_data.write();
				memcpy(data_write.ptr(), screen, image_size);

				float delay = gcb.DelayTime / 100.0;
				if (delay == 0)
					delay = 0.05; // Default delay.

				Ref<Image> img = memnew(Image(gif->SWidth, gif->SHeight, false, Image::FORMAT_RGBA8, frame_data));
				r_animated_image->add_frame(img, delay);

				gif->ImageCount++;

				switch (gcb.DisposalMode) { // What should happen after the frame has been drawn.

					case DISPOSE_BACKGROUND: { // Make the area of the current frame transparent.

						for (int y = 0; y < imageDesc.Height; y++) {

							int write_y = y + imageDesc.Top;
							int write_index = (write_y * gif->SWidth + imageDesc.Left) * RGBA;
							memset(&screen[write_index], 0, imageDesc.Width * RGBA);
						}
					} break;
					case DISPOSE_PREVIOUS: { // Reset the screen to the last undisposed frame.

						int row_size = imageDesc.Width * RGBA;

						if (last_undisposed_frame == -1) { // Clear the frame.

							for (int y = 0; y < imageDesc.Height; y++) {

								int write_y = y + imageDesc.Top;
								int write_index = (write_y * gif->SWidth + imageDesc.Left) * RGBA;
								memset(&screen[write_index], 0, row_size);
							}
						} else {

							PoolByteArray last_frame_data = r_animated_image->get_image(last_undisposed_frame)->get_data();
							PoolByteArray::Read last_frame_read = last_frame_data.read();
							for (int y = 0; y < imageDesc.Height; y++) {

								int write_y = y + imageDesc.Top;
								int write_index = (write_y * gif->SWidth + imageDesc.Left) * RGBA;
								memcpy(&screen[write_index], &last_frame_read.ptr()[write_index], row_size);
							}
						}
					} break;
					default: { // Do nothing.

						last_undisposed_frame = gif->ImageCount - 1;
					}
				}

				// Reset the GraphicsControlBlock to his default values.
				gcb.DisposalMode = DISPOSAL_UNSPECIFIED;
				gcb.UserInputFlag = false;
				gcb.DelayTime = 0;
				gcb.TransparentColor = NO_TRANSPARENT_COLOR;

			} break;
			default: {
			}
		}

		if (gif->ImageCount == max_frames && gif->ImageCount > 0)
			break;

	} while (recordType != TERMINATE_RECORD_TYPE);

	if (gif->ImageCount == 0)
		return parse_error(ERR_FILE_CORRUPT, "No frames found.");

	memdelete_arr(screen);

	return OK;
}

Error Gif::_close() {

	int err = 0;
	if (!DGifCloseFile(gif, &err))
		return gif_error(err);

	gif = NULL;
	return OK;
}

Error Gif::load_from_file_access(Ref<AnimatedImage> &r_animated_image, FileAccess *f, int max_frames) {

	Error err;

	err = _open(f, AnimatedImage::FILE);
	if (err != OK)
		return ERR_FILE_CORRUPT;

	err = _load_frames(r_animated_image, max_frames);
	if (err != OK) {

		_close();
		return ERR_FILE_CORRUPT;
	}

	err = _close();
	if (err != OK)
		return ERR_FILE_CORRUPT;

	return OK;
}

Error Gif::load_from_buffer(Ref<AnimatedImage> &r_animated_image, const PoolByteArray &p_data, int max_frames) {

	GIFBuffer f = GIFBuffer(p_data);

	Error err;
	err = _open(&f, AnimatedImage::BUFFER);
	if (err != OK)
		return ERR_FILE_CORRUPT;

	err = _load_frames(r_animated_image, max_frames);
	if (err != OK) {

		_close();
		return ERR_FILE_CORRUPT;
	}

	err = _close();
	if (err != OK)
		return ERR_FILE_CORRUPT;

	return OK;
}

Gif::Gif() :
		gif(NULL) {}

///////////////

static Error _load_gif(Ref<AnimatedImage> &r_animated_image, const Variant &source, int max_frames) {

	Gif gif;

	if (source.get_type() == Variant::STRING) {
		Error err;
		FileAccess *f = FileAccess::open(source, FileAccess::READ, &err);
		if (!f) {

			ERR_PRINTS("Error opening file '" + String(source) + "'.");
			return err;
		}

		err = gif.load_from_file_access(r_animated_image, f, max_frames);

		f->close();
		memdelete(f);
		return err;
	} else {

		return gif.load_from_buffer(r_animated_image, source, max_frames);
	}
}

///////////////

Error AnimatedImageLoaderGIF::load_animated_image(Ref<AnimatedImage> &r_animated_image, FileAccess *f, int max_frames) const {

	Gif gif;
	return gif.load_from_file_access(r_animated_image, f, max_frames);
}

void AnimatedImageLoaderGIF::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("gif");
}

bool AnimatedImageLoaderGIF::recognize_format(AnimatedImage::SourceFormat p_format) const {

	return p_format == AnimatedImage::GIF;
}

AnimatedImageLoaderGIF::AnimatedImageLoaderGIF() {

	AnimatedImage::_load_gif = _load_gif;
}
