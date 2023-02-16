/**************************************************************************/
/*  psd_texture.h                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef PSD_TEXTURE_H
#define PSD_TEXTURE_H

#include "core/io/resource_loader.h"

#include "Psd.h"
#include "PsdNativeFile_Godot.h"

//#include "Psd/PsdPlatform.h"

#include "PsdMallocAllocator.h"

#include "PsdDocument.h"
#include "PsdColorMode.h"
#include "PsdLayer.h"
#include "PsdChannel.h"
#include "PsdChannelType.h"
#include "PsdLayerMask.h"
#include "PsdVectorMask.h"
#include "PsdLayerMaskSection.h"
#include "PsdImageDataSection.h"
#include "PsdImageResourcesSection.h"
#include "PsdParseDocument.h"
#include "PsdParseLayerMaskSection.h"
#include "PsdParseImageDataSection.h"
#include "PsdParseImageResourcesSection.h"
#include "PsdLayerCanvasCopy.h"
#include "PsdInterleave.h"
#include "PsdPlanarImage.h"
#include "PsdExport.h"
#include "PsdExportDocument.h"
#include "PsdLayerType.h"


#include "core/io/image.h"
#include "scene/resources/texture.h"

#include <string>
#include <sstream>

class PSDTexture;


class PSDTexture : public Resource {
	GDCLASS(PSDTexture, Resource);
	OBJ_SAVE_TYPE(PSDTexture)
	RES_BASE_EXTENSION("psdstr");


	PackedByteArray data;
	uint32_t data_len = 0;

	bool cropToCanvas = true;


	Dictionary layers;

	typedef enum
	{
		KRA,
		PSD
	} IMPORT_TYPE;

	typedef enum
	{
		MONOCHROME,
		RGB,
		RGBA
	} COLOR_SPACE_NAME;

protected:
	static void _bind_methods();

	void parse();

	void ExportLayer(const wchar_t* name, unsigned int width, unsigned int height, const uint8_t *data, int channelType);

	void clear_data();

public:

	Array get_layer_names() const;
	Ref<ImageTexture> get_texture_layer(String p_name) const;

	void set_data(const Vector<uint8_t> &p_data);
	Vector<uint8_t> get_data() const;


	PSDTexture();
	virtual ~PSDTexture();
};

#endif // PSD_TEXTURE_H
