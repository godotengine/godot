/*************************************************************************/
/*  register_core_types.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/core_bind.h"
#include "core/core_string_names.h"
#include "core/crypto/aes_context.h"
#include "core/crypto/crypto.h"
#include "core/crypto/hashing_context.h"
#include "core/extension/native_extension.h"
#include "core/extension/native_extension_manager.h"
#include "core/input/input.h"
#include "core/input/input_map.h"
#include "core/input/shortcut.h"
#include "core/io/config_file.h"
#include "core/io/dtls_server.h"
#include "core/io/http_client.h"
#include "core/io/image_loader.h"
#include "core/io/json.h"
#include "core/io/marshalls.h"
#include "core/io/packed_data_container.h"
#include "core/io/packet_peer.h"
#include "core/io/packet_peer_dtls.h"
#include "core/io/packet_peer_udp.h"
#include "core/io/pck_packer.h"
#include "core/io/resource_format_binary.h"
#include "core/io/resource_importer.h"
#include "core/io/resource_uid.h"
#include "core/io/stream_peer_ssl.h"
#include "core/io/tcp_server.h"
#include "core/io/translation_loader_po.h"
#include "core/io/udp_server.h"
#include "core/io/xml_parser.h"
#include "core/math/a_star.h"
#include "core/math/expression.h"
#include "core/math/geometry_2d.h"
#include "core/math/geometry_3d.h"
#include "core/math/random_number_generator.h"
#include "core/math/triangle_mesh.h"
#include "core/multiplayer/multiplayer_api.h"
#include "core/multiplayer/multiplayer_peer.h"
#include "core/multiplayer/multiplayer_replicator.h"
#include "core/object/class_db.h"
#include "core/object/undo_redo.h"
#include "core/os/main_loop.h"
#include "core/os/time.h"
#include "core/string/optimized_translation.h"
#include "core/string/translation.h"

static Ref<ResourceFormatSaverBinary> resource_saver_binary;
static Ref<ResourceFormatLoaderBinary> resource_loader_binary;
static Ref<ResourceFormatImporter> resource_format_importer;
static Ref<ResourceFormatLoaderImage> resource_format_image;
static Ref<TranslationLoaderPO> resource_format_po;
static Ref<ResourceFormatSaverCrypto> resource_format_saver_crypto;
static Ref<ResourceFormatLoaderCrypto> resource_format_loader_crypto;
static Ref<NativeExtensionResourceLoader> resource_loader_native_extension;

static core_bind::ResourceLoader *_resource_loader = nullptr;
static core_bind::ResourceSaver *_resource_saver = nullptr;
static core_bind::OS *_os = nullptr;
static core_bind::Engine *_engine = nullptr;
static core_bind::special::ClassDB *_classdb = nullptr;
static core_bind::Marshalls *_marshalls = nullptr;
static core_bind::EngineDebugger *_engine_debugger = nullptr;

static IP *ip = nullptr;

static core_bind::Geometry2D *_geometry_2d = nullptr;
static core_bind::Geometry3D *_geometry_3d = nullptr;

extern Mutex _global_mutex;

static NativeExtensionManager *native_extension_manager = nullptr;

extern void register_global_constants();
extern void unregister_global_constants();

static ResourceUID *resource_uid = nullptr;

void register_core_types() {
	//consistency check
	static_assert(sizeof(Callable) <= 16);

	ObjectDB::setup();

	StringName::setup();
	ResourceLoader::initialize();

	register_global_constants();

	Variant::register_types();

	CoreStringNames::create();

	resource_format_po.instantiate();
	ResourceLoader::add_resource_format_loader(resource_format_po);

	resource_saver_binary.instantiate();
	ResourceSaver::add_resource_format_saver(resource_saver_binary);
	resource_loader_binary.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_binary);

	resource_format_importer.instantiate();
	ResourceLoader::add_resource_format_loader(resource_format_importer);

	resource_format_image.instantiate();
	ResourceLoader::add_resource_format_loader(resource_format_image);

	GDREGISTER_CLASS(Object);

	GDREGISTER_VIRTUAL_CLASS(Script);

	GDREGISTER_CLASS(RefCounted);
	GDREGISTER_CLASS(WeakRef);
	GDREGISTER_CLASS(Resource);
	GDREGISTER_CLASS(Image);

	GDREGISTER_CLASS(Shortcut);
	GDREGISTER_VIRTUAL_CLASS(InputEvent);
	GDREGISTER_VIRTUAL_CLASS(InputEventWithModifiers);
	GDREGISTER_VIRTUAL_CLASS(InputEventFromWindow);
	GDREGISTER_CLASS(InputEventKey);
	GDREGISTER_CLASS(InputEventShortcut);
	GDREGISTER_VIRTUAL_CLASS(InputEventMouse);
	GDREGISTER_CLASS(InputEventMouseButton);
	GDREGISTER_CLASS(InputEventMouseMotion);
	GDREGISTER_CLASS(InputEventJoypadButton);
	GDREGISTER_CLASS(InputEventJoypadMotion);
	GDREGISTER_CLASS(InputEventScreenDrag);
	GDREGISTER_CLASS(InputEventScreenTouch);
	GDREGISTER_CLASS(InputEventAction);
	GDREGISTER_VIRTUAL_CLASS(InputEventGesture);
	GDREGISTER_CLASS(InputEventMagnifyGesture);
	GDREGISTER_CLASS(InputEventPanGesture);
	GDREGISTER_CLASS(InputEventMIDI);

	// Network
	GDREGISTER_VIRTUAL_CLASS(IP);

	GDREGISTER_VIRTUAL_CLASS(StreamPeer);
	GDREGISTER_CLASS(StreamPeerExtension);
	GDREGISTER_CLASS(StreamPeerBuffer);
	GDREGISTER_CLASS(StreamPeerTCP);
	GDREGISTER_CLASS(TCPServer);

	GDREGISTER_VIRTUAL_CLASS(PacketPeer);
	GDREGISTER_CLASS(PacketPeerExtension);
	GDREGISTER_CLASS(PacketPeerStream);
	GDREGISTER_CLASS(PacketPeerUDP);
	GDREGISTER_CLASS(UDPServer);

	ClassDB::register_custom_instance_class<HTTPClient>();

	// Crypto
	GDREGISTER_CLASS(HashingContext);
	GDREGISTER_CLASS(AESContext);
	ClassDB::register_custom_instance_class<X509Certificate>();
	ClassDB::register_custom_instance_class<CryptoKey>();
	ClassDB::register_custom_instance_class<HMACContext>();
	ClassDB::register_custom_instance_class<Crypto>();
	ClassDB::register_custom_instance_class<StreamPeerSSL>();
	ClassDB::register_custom_instance_class<PacketPeerDTLS>();
	ClassDB::register_custom_instance_class<DTLSServer>();

	resource_format_saver_crypto.instantiate();
	ResourceSaver::add_resource_format_saver(resource_format_saver_crypto);
	resource_format_loader_crypto.instantiate();
	ResourceLoader::add_resource_format_loader(resource_format_loader_crypto);

	GDREGISTER_VIRTUAL_CLASS(MultiplayerPeer);
	GDREGISTER_CLASS(MultiplayerPeerExtension);
	GDREGISTER_VIRTUAL_CLASS(MultiplayerReplicator);
	GDREGISTER_CLASS(MultiplayerAPI);
	GDREGISTER_CLASS(MainLoop);
	GDREGISTER_CLASS(Translation);
	GDREGISTER_CLASS(OptimizedTranslation);
	GDREGISTER_CLASS(UndoRedo);
	GDREGISTER_CLASS(TriangleMesh);

	GDREGISTER_CLASS(ResourceFormatLoader);
	GDREGISTER_CLASS(ResourceFormatSaver);

	GDREGISTER_CLASS(core_bind::File);
	GDREGISTER_CLASS(core_bind::Directory);
	GDREGISTER_CLASS(core_bind::Thread);
	GDREGISTER_CLASS(core_bind::Mutex);
	GDREGISTER_CLASS(core_bind::Semaphore);

	GDREGISTER_CLASS(XMLParser);
	GDREGISTER_CLASS(JSON);

	GDREGISTER_CLASS(ConfigFile);

	GDREGISTER_CLASS(PCKPacker);

	GDREGISTER_CLASS(PackedDataContainer);
	GDREGISTER_VIRTUAL_CLASS(PackedDataContainerRef);
	GDREGISTER_CLASS(AStar);
	GDREGISTER_CLASS(AStar2D);
	GDREGISTER_CLASS(EncodedObjectAsID);
	GDREGISTER_CLASS(RandomNumberGenerator);

	GDREGISTER_VIRTUAL_CLASS(ResourceImporter);

	GDREGISTER_CLASS(NativeExtension);

	GDREGISTER_VIRTUAL_CLASS(NativeExtensionManager);

	GDREGISTER_VIRTUAL_CLASS(ResourceUID);

	resource_uid = memnew(ResourceUID);

	native_extension_manager = memnew(NativeExtensionManager);

	resource_loader_native_extension.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_native_extension);

	ip = IP::create();

	_geometry_2d = memnew(core_bind::Geometry2D);
	_geometry_3d = memnew(core_bind::Geometry3D);

	_resource_loader = memnew(core_bind::ResourceLoader);
	_resource_saver = memnew(core_bind::ResourceSaver);
	_os = memnew(core_bind::OS);
	_engine = memnew(core_bind::Engine);
	_classdb = memnew(core_bind::special::ClassDB);
	_marshalls = memnew(core_bind::Marshalls);
	_engine_debugger = memnew(core_bind::EngineDebugger);
}

void register_core_settings() {
	// Since in register core types, globals may not be present.
	GLOBAL_DEF("network/limits/tcp/connect_timeout_seconds", (30));
	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/tcp/connect_timeout_seconds", PropertyInfo(Variant::INT, "network/limits/tcp/connect_timeout_seconds", PROPERTY_HINT_RANGE, "1,1800,1"));
	GLOBAL_DEF_RST("network/limits/packet_peer_stream/max_buffer_po2", (16));
	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/packet_peer_stream/max_buffer_po2", PropertyInfo(Variant::INT, "network/limits/packet_peer_stream/max_buffer_po2", PROPERTY_HINT_RANGE, "0,64,1,or_greater"));

	GLOBAL_DEF("network/ssl/certificate_bundle_override", "");
	ProjectSettings::get_singleton()->set_custom_property_info("network/ssl/certificate_bundle_override", PropertyInfo(Variant::STRING, "network/ssl/certificate_bundle_override", PROPERTY_HINT_FILE, "*.crt"));
}

void register_core_singletons() {
	GDREGISTER_CLASS(ProjectSettings);
	GDREGISTER_VIRTUAL_CLASS(IP);
	GDREGISTER_CLASS(core_bind::Geometry2D);
	GDREGISTER_CLASS(core_bind::Geometry3D);
	GDREGISTER_CLASS(core_bind::ResourceLoader);
	GDREGISTER_CLASS(core_bind::ResourceSaver);
	GDREGISTER_CLASS(core_bind::OS);
	GDREGISTER_CLASS(core_bind::Engine);
	GDREGISTER_CLASS(core_bind::special::ClassDB);
	GDREGISTER_CLASS(core_bind::Marshalls);
	GDREGISTER_CLASS(TranslationServer);
	GDREGISTER_VIRTUAL_CLASS(Input);
	GDREGISTER_CLASS(InputMap);
	GDREGISTER_CLASS(Expression);
	GDREGISTER_CLASS(core_bind::EngineDebugger);
	GDREGISTER_CLASS(Time);

	Engine::get_singleton()->add_singleton(Engine::Singleton("ProjectSettings", ProjectSettings::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("IP", IP::get_singleton(), "IP"));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Geometry2D", core_bind::Geometry2D::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Geometry3D", core_bind::Geometry3D::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("ResourceLoader", core_bind::ResourceLoader::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("ResourceSaver", core_bind::ResourceSaver::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("OS", core_bind::OS::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Engine", core_bind::Engine::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("ClassDB", _classdb));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Marshalls", core_bind::Marshalls::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("TranslationServer", TranslationServer::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Input", Input::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("InputMap", InputMap::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("EngineDebugger", core_bind::EngineDebugger::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Time", Time::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("NativeExtensionManager", NativeExtensionManager::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("ResourceUID", ResourceUID::get_singleton()));
}

void register_core_extensions() {
	// Hardcoded for now.
	NativeExtension::initialize_native_extensions();
	native_extension_manager->load_extensions();
	native_extension_manager->initialize_extensions(NativeExtension::INITIALIZATION_LEVEL_CORE);
}

void unregister_core_types() {
	native_extension_manager->deinitialize_extensions(NativeExtension::INITIALIZATION_LEVEL_CORE);

	memdelete(native_extension_manager);

	memdelete(resource_uid);
	memdelete(_resource_loader);
	memdelete(_resource_saver);
	memdelete(_os);
	memdelete(_engine);
	memdelete(_classdb);
	memdelete(_marshalls);
	memdelete(_engine_debugger);

	memdelete(_geometry_2d);
	memdelete(_geometry_3d);

	ResourceLoader::remove_resource_format_loader(resource_format_image);
	resource_format_image.unref();

	ResourceSaver::remove_resource_format_saver(resource_saver_binary);
	resource_saver_binary.unref();

	ResourceLoader::remove_resource_format_loader(resource_loader_binary);
	resource_loader_binary.unref();

	ResourceLoader::remove_resource_format_loader(resource_format_importer);
	resource_format_importer.unref();

	ResourceLoader::remove_resource_format_loader(resource_format_po);
	resource_format_po.unref();

	ResourceSaver::remove_resource_format_saver(resource_format_saver_crypto);
	resource_format_saver_crypto.unref();
	ResourceLoader::remove_resource_format_loader(resource_format_loader_crypto);
	resource_format_loader_crypto.unref();

	if (ip) {
		memdelete(ip);
	}

	ResourceLoader::remove_resource_format_loader(resource_loader_native_extension);
	resource_loader_native_extension.unref();

	ResourceLoader::finalize();

	ClassDB::cleanup_defaults();
	ObjectDB::cleanup();

	Variant::unregister_types();

	unregister_global_constants();

	ClassDB::cleanup();
	ResourceCache::clear();
	CoreStringNames::free();
	StringName::cleanup();
}
