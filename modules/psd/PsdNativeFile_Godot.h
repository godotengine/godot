/**************************************************************************/
/*  PsdNativeFile_Godot.h                                                 */
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

#pragma once

#include "PsdFile.h"


PSD_NAMESPACE_BEGIN

/// \ingroup Files
/// \brief Simple file implementation that uses Windows' native file functions internally.
/// \sa File
class NativeFile : public File
{
public:
	/// Constructor.
	explicit NativeFile(Allocator* allocator);

	void OpenBuffer(const uint8_t *p_buf, size_t p_buf_size);

private:
	virtual bool DoOpenRead(const wchar_t* filename) PSD_OVERRIDE;
	virtual bool DoOpenWrite(const wchar_t* filename) PSD_OVERRIDE;
	virtual bool DoClose(void) PSD_OVERRIDE;

	virtual File::ReadOperation DoRead(void* buffer, uint32_t count, uint64_t position) PSD_OVERRIDE;
	virtual bool DoWaitForRead(File::ReadOperation& operation) PSD_OVERRIDE;

	virtual File::WriteOperation DoWrite(const void* buffer, uint32_t count, uint64_t position) PSD_OVERRIDE;
	virtual bool DoWaitForWrite(File::WriteOperation& operation) PSD_OVERRIDE;

	virtual uint64_t DoGetSize(void) const PSD_OVERRIDE;

	// internally, this is a HANDLE, but we don't want to include Windows.h just for that
	void* m_file;


 	const uint8_t *buf;
 	size_t buf_size;

};

PSD_NAMESPACE_END
