/*************************************************************************/
/*  register_core_types.cpp                                              */
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
#include "register_core_types.h"

#include "bind/core_bind.h"
#include "color.h"
#include "compressed_translation.h"
#include "core/io/xml_parser.h"
#include "core_string_names.h"
#include "func_ref.h"
#include "geometry.h"
#include "globals.h"
#include "input_map.h"
#include "io/config_file.h"
#include "io/http_client.h"
#include "io/packet_peer.h"
#include "io/packet_peer_udp.h"
#include "io/pck_packer.h"
#include "io/resource_format_binary.h"
#include "io/resource_format_xml.h"
#include "io/stream_peer_ssl.h"
#include "io/tcp_server.h"
#include "io/translation_loader_po.h"
#include "math/a_star.h"
#include "object_type_db.h"
#include "os/input.h"
#include "os/main_loop.h"
#include "packed_data_container.h"
#include "path_remap.h"
#include "translation.h"
#include "undo_redo.h"

#ifdef XML_ENABLED
static ResourceFormatSaverXML *resource_saver_xml = NULL;
static ResourceFormatLoaderXML *resource_loader_xml = NULL;
#endif
static ResourceFormatSaverBinary *resource_saver_binary = NULL;
static ResourceFormatLoaderBinary *resource_loader_binary = NULL;

static _ResourceLoader *_resource_loader = NULL;
static _ResourceSaver *_resource_saver = NULL;
static _OS *_os = NULL;
static _Marshalls *_marshalls = NULL;
static TranslationLoaderPO *resource_format_po = NULL;

static IP *ip = NULL;

static _Geometry *_geometry = NULL;

extern Mutex *_global_mutex;

extern void register_variant_methods();
extern void unregister_variant_methods();

void register_core_types() {

	_global_mutex = Mutex::create();

	StringName::setup();

	register_variant_methods();

	CoreStringNames::create();

	resource_format_po = memnew(TranslationLoaderPO);
	ResourceLoader::add_resource_format_loader(resource_format_po);

	resource_saver_binary = memnew(ResourceFormatSaverBinary);
	ResourceSaver::add_resource_format_saver(resource_saver_binary);
	resource_loader_binary = memnew(ResourceFormatLoaderBinary);
	ResourceLoader::add_resource_format_loader(resource_loader_binary);

#ifdef XML_ENABLED
	resource_saver_xml = memnew(ResourceFormatSaverXML);
	ResourceSaver::add_resource_format_saver(resource_saver_xml);
	resource_loader_xml = memnew(ResourceFormatLoaderXML);
	ResourceLoader::add_resource_format_loader(resource_loader_xml);
#endif

	ObjectTypeDB::register_type<Object>();

	ObjectTypeDB::register_type<Reference>();
	ObjectTypeDB::register_type<WeakRef>();
	ObjectTypeDB::register_type<ResourceImportMetadata>();
	ObjectTypeDB::register_type<Resource>();
	ObjectTypeDB::register_type<FuncRef>();
	ObjectTypeDB::register_virtual_type<StreamPeer>();
	ObjectTypeDB::register_type<StreamPeerBuffer>();
	ObjectTypeDB::register_create_type<StreamPeerTCP>();
	ObjectTypeDB::register_create_type<TCP_Server>();
	ObjectTypeDB::register_create_type<PacketPeerUDP>();
	ObjectTypeDB::register_create_type<StreamPeerSSL>();
	ObjectTypeDB::register_virtual_type<IP>();
	ObjectTypeDB::register_virtual_type<PacketPeer>();
	ObjectTypeDB::register_type<PacketPeerStream>();
	ObjectTypeDB::register_type<MainLoop>();
	//	ObjectTypeDB::register_type<OptimizedSaver>();
	ObjectTypeDB::register_type<Translation>();
	ObjectTypeDB::register_type<PHashTranslation>();
	ObjectTypeDB::register_type<UndoRedo>();
	ObjectTypeDB::register_type<HTTPClient>();

	ObjectTypeDB::register_virtual_type<ResourceInteractiveLoader>();

	ObjectTypeDB::register_type<_File>();
	ObjectTypeDB::register_type<_Directory>();
	ObjectTypeDB::register_type<_Thread>();
	ObjectTypeDB::register_type<_Mutex>();
	ObjectTypeDB::register_type<_Semaphore>();

	ObjectTypeDB::register_type<XMLParser>();

	ObjectTypeDB::register_type<ConfigFile>();

	ObjectTypeDB::register_type<PCKPacker>();

	ObjectTypeDB::register_type<PackedDataContainer>();
	ObjectTypeDB::register_virtual_type<PackedDataContainerRef>();
	ObjectTypeDB::register_type<AStar>();

	ip = IP::create();

	_geometry = memnew(_Geometry);

	_resource_loader = memnew(_ResourceLoader);
	_resource_saver = memnew(_ResourceSaver);
	_os = memnew(_OS);
	_marshalls = memnew(_Marshalls);
}

void register_core_singletons() {

	Globals::get_singleton()->add_singleton(Globals::Singleton("Globals", Globals::get_singleton()));
	Globals::get_singleton()->add_singleton(Globals::Singleton("IP", IP::get_singleton()));
	Globals::get_singleton()->add_singleton(Globals::Singleton("Geometry", _Geometry::get_singleton()));
	Globals::get_singleton()->add_singleton(Globals::Singleton("ResourceLoader", _ResourceLoader::get_singleton()));
	Globals::get_singleton()->add_singleton(Globals::Singleton("ResourceSaver", _ResourceSaver::get_singleton()));
	Globals::get_singleton()->add_singleton(Globals::Singleton("PathRemap", PathRemap::get_singleton()));
	Globals::get_singleton()->add_singleton(Globals::Singleton("OS", _OS::get_singleton()));
	Globals::get_singleton()->add_singleton(Globals::Singleton("Marshalls", _marshalls));
	Globals::get_singleton()->add_singleton(Globals::Singleton("TranslationServer", TranslationServer::get_singleton()));
	Globals::get_singleton()->add_singleton(Globals::Singleton("TS", TranslationServer::get_singleton()));
	Globals::get_singleton()->add_singleton(Globals::Singleton("Input", Input::get_singleton()));
	Globals::get_singleton()->add_singleton(Globals::Singleton("InputMap", InputMap::get_singleton()));
}

void unregister_core_types() {

	memdelete(_resource_loader);
	memdelete(_resource_saver);
	memdelete(_os);
	memdelete(_marshalls);

	memdelete(_geometry);
#ifdef XML_ENABLED
	if (resource_saver_xml)
		memdelete(resource_saver_xml);
	if (resource_loader_xml)
		memdelete(resource_loader_xml);
#endif

	if (resource_saver_binary)
		memdelete(resource_saver_binary);
	if (resource_loader_binary)
		memdelete(resource_loader_binary);

	memdelete(resource_format_po);

	if (ip)
		memdelete(ip);

	ObjectDB::cleanup();

	unregister_variant_methods();

	Color::cleanup();
	ObjectTypeDB::cleanup();
	ResourceCache::clear();
	CoreStringNames::free();
	StringName::cleanup();

	if (_global_mutex) {
		memdelete(_global_mutex);
		_global_mutex = NULL; //still needed at a few places
	};
}
