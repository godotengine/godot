/**************************************************************************/
/*  file_info.cpp                                                         */
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

#include "file_info.h"

void sort_file_info_list(List<FileInfo> &r_file_list, FileSortOption p_file_sort_option) {
	// Sort the file list if needed.
	switch (p_file_sort_option) {
		case FileSortOption::FILE_SORT_TYPE:
			r_file_list.sort_custom<FileInfoTypeComparator>();
			break;
		case FileSortOption::FILE_SORT_TYPE_REVERSE:
			r_file_list.sort_custom<FileInfoTypeComparator>();
			r_file_list.reverse();
			break;
		case FileSortOption::FILE_SORT_MODIFIED_TIME:
			r_file_list.sort_custom<FileInfoModifiedTimeComparator>();
			break;
		case FileSortOption::FILE_SORT_MODIFIED_TIME_REVERSE:
			r_file_list.sort_custom<FileInfoModifiedTimeComparator>();
			r_file_list.reverse();
			break;
		case FileSortOption::FILE_SORT_NAME_REVERSE:
			r_file_list.sort();
			r_file_list.reverse();
			break;
		case FileSortOption::FILE_SORT_NAME:
			r_file_list.sort();
			break;
		default:
			ERR_FAIL_MSG("Invalid file sort option.");
			break;
	}
}
