/**************************************************************************/
/*  sandbox_programs.cpp                                                  */
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

#include "sandbox.h"

#include "core/config/engine.h"
#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/http_client_tcp.h"
#ifdef WEB_ENABLED
#include "platform/web/http_client_web.h"
#endif
#include "core/string/ustring.h"
#include "core/templates/list.h"
#include "core/templates/vector.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant_utility.h"
#include "modules/zip/zip_reader.h"
#include "sandbox_project_settings.h"
static constexpr bool VERBOSE = false;

static PackedByteArray handle_request(HTTPClient *client, String url) {
	while (client->get_status() != HTTPClient::Status::STATUS_CONNECTED) {
		if (client->get_status() == HTTPClient::Status::STATUS_CANT_CONNECT || client->get_status() == HTTPClient::Status::STATUS_CONNECTION_ERROR || client->get_status() == HTTPClient::Status::STATUS_CANT_RESOLVE) {
			ERR_PRINT("Failed to connect to server");
			return PackedByteArray();
		}
		client->poll();
	}
	Vector<String> headers;
	Error err = client->request(HTTPClient::Method::METHOD_GET, url, headers, nullptr, 0);
	if (err != OK) {
		return PackedByteArray();
	}

	while (client->get_status() == HTTPClient::Status::STATUS_REQUESTING) {
		client->poll();
	}

	if (!client->has_response()) {
		return PackedByteArray();
	}

	if (client->get_response_code() >= 300 && client->get_response_code() < 400) {
		List<String> headers_list;
		client->get_response_headers(&headers_list);
		String location;
		for (const String &header : headers_list) {
			if (header.begins_with("Location: ")) {
				location = header.substr(10);
				break;
			}
		}
		if (location.is_empty()) {
			ERR_PRINT("No Location header found in redirect");
			return PackedByteArray();
		}

		if constexpr (VERBOSE) {
			ERR_PRINT("Redirected to: " + location);
		}

		client->close();
		// Find first / after :// to get the host and path
		int32_t host_end = location.find("/", 8);
		if (host_end == -1) {
			ERR_PRINT("Invalid redirect location");
			return PackedByteArray();
		}
		String host = location.substr(0, host_end);
		String path = location.substr(host_end);
		client->connect_to_host(host);

		// Handle the redirect
		return handle_request(client, path);
	}

	if (client->get_response_code() != 200) {
		ERR_PRINT("Failed to download program: HTTP status " + itos(client->get_response_code()));
		return PackedByteArray();
	}

	// Save the downloaded program to a buffer
	PackedByteArray data;
	while (client->get_status() == HTTPClient::Status::STATUS_BODY) {
		client->poll();

		PackedByteArray chunk = client->read_response_body_chunk();
		if (chunk.size() == 0) {
			continue;
		}

		data.append_array(chunk);
	}
	return data;
}

PackedByteArray Sandbox::download_program(String program_name) {
	String url;

	// Check if the program is from another library
	const int separator = program_name.find("/");
	if (separator != -1) {
		String library = program_name.substr(0, separator);
		String program = program_name.substr(separator + 1);
		if (library.is_empty() || program.is_empty()) {
			ERR_PRINT("Invalid library or program name");
			return PackedByteArray();
		}

		Dictionary libraries = SandboxProjectSettings::get_program_libraries();
		if (!libraries.has(library)) {
			ERR_PRINT("Unknown library: " + library);
			return PackedByteArray();
		}
		// Get the URL for the program from the custom library
		program_name = program;
		url = String("/") + String(libraries[library]) + "/releases/latest/download/" + program + ".zip";
	} else {
		// Use the default URL for Sandbox programs
		url = "/libriscv/godot-sandbox-programs/releases/latest/download/" + program_name + ".zip";
	}

	// Download the program from the URL
#ifdef WEB_ENABLED
	HTTPClientWeb *client = memnew(HTTPClientWeb);
#else
	HTTPClientTCP *client = memnew(HTTPClientTCP);
	client->set_blocking_mode(true);
#endif
	client->connect_to_host("https://github.com", 443);

	PackedByteArray data = handle_request(client, url);
	//printf("Response code: %d\n", client->get_response_code());
	//UtilityFunctions::print(client->get_response_headers_as_dictionary());
	memdelete(client);

	if (data.is_empty()) {
		return data;
	}

	// Save the downloaded program to a temporary file
	Ref<FileAccess> file = FileAccess::open("user://temp.zip", FileAccess::ModeFlags::WRITE);
	if (!file.is_valid() || !file->is_open()) {
		ERR_PRINT("Failed to open temporary file for writing");
		return PackedByteArray();
	}
	file->store_buffer(data);
	file->close();

	// Read the temporary file into a byte array
	ZIPReader *zip = memnew(ZIPReader);
	Error err = zip->open("user://temp.zip");
	if (err != OK) {
		ERR_PRINT("Failed to open temporary file for reading");
		memdelete(zip);
		// Remove the temporary file even on error
		Ref<DirAccess> dir = DirAccess::open("user://");
		if (dir.is_valid()) {
			dir->remove("temp.zip");
		}
		return PackedByteArray();
	}

	PackedByteArray program_data = zip->read_file(program_name + String(".elf"), true);
	memdelete(zip);

	// Remove the temporary file
	Ref<DirAccess> dir = DirAccess::open("user://");
	if (dir.is_valid()) {
		dir->remove("temp.zip");
	}

	return program_data;
}
