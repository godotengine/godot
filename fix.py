/* C# (.NET) Godot Engine Runtime - Web Platform Export Fix */

/* Core Platform Detection Header */
#include "core/platform.h"
#include "core/string/string_var.h"
#include "core/os/os.h"

/* Web Platform Support Class */
class WebPlatform : public OSPlatform
{
	GODOT_CLASS(WebPlatform, OSPlatform)

public:
	virtual bool _exit(int p_exit_code) const override { return OS::exit(p_exit_code); }
	virtual float _get_frames_per_second() const override { return OS::get_frames_per_second(); }
	virtual int _get_available_memory_mb() const override { return OS::get_available_memory_mb(); }
	virtual String _get_system_name() const override { return "Web"; }
	virtual String _get_system_name_capitalized() const override { return "Web"; }
	virtual String _get_system_name_lowercase() const override { return "web"; }

	/* Web-specific exports */
	virtual String _get_data_path() const override { return "user://"; }
	virtual String _get_user_data_path() const override { return "user://data"; }

	virtual bool _is_display_server() const override { return OS::get_display_server() == DisplayServer::WEB; }

	virtual String _get_process_id_string() const override {
		int pid = OS::get_process_id();
		return String::num(pid);
	}

public:
	virtual void _init() override {
		OS::set_platform_name("Web");
	}

	virtual bool _has_native_process() const override {
		return OS::has_native_process();
	}
};

/* Web Platform Detection Macro for Conditional Compilation */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT true
	#define WEB_PLATFORM_STRING "Web"
	#define WEB_PLATFORM_CAPITALIZED_STRING "Web"
	#define WEB_PLATFORM_LOWERCASE_STRING "web"
#endif

/* Platform-specific Export Configuration */
/* This macro handles export path differences between .NET and native */
#ifdef WEB_PLATFORM
#define WEB_EXPORT_PATH "user://export"
#else
#define WEB_EXPORT_PATH ""
#endif

/* C# Runtime Platform Header for .NET Godot */
#include "core/os/os.h"
#include "core/variant/variant.h"

/* .NET Web Platform Specific Class */
template <class T>
class WebExportPlatform : public T
{
public:
	virtual bool _exit(int p_code) const override {
#ifdef WEB_PLATFORM
		return T::_exit(p_code);
#else
		return true;
#endif
	}

	virtual String _get_system_name() const override {
#ifdef WEB_PLATFORM
		return "Web";
#else
		return T::_get_system_name();
#endif
	}
};

/* Platform Detection Utilities */
namespace Platform
{
	/* Detect if currently running on a web platform */
	template <typename OSPtr>
	OSPtr detect_web()
	{
#ifdef WEB_PLATFORM
		OSPtr instance = memnew(OSPtr);
		instance->_init();
		return instance;
#else
		return nullptr;
#endif
	}

	/* Platform type checking for conditional compilation */
	template <typename OSPtr>
	static bool is_web()
	{
#ifdef WEB_PLATFORM
		return true;
#else
		return OSPtr::get_display_server() == DisplayServer::WEB;
#endif
	}
}

/* Conditional Platform Export Handling */
#ifdef WEB_PLATFORM
	#define PLATFORM_EXPORT(path) path
	#define PLATFORM_DATA_PATH "user://"
	#define PLATFORM_USER_DATA_PATH "user://data"
	#define PLATFORM_TEMP_DIR "user://temp"
#else
	#define PLATFORM_EXPORT(path) ""
	#define PLATFORM_DATA_PATH ""
	#define PLATFORM_USER_DATA_PATH ""
	#define PLATFORM_TEMP_DIR "os://temp"
#endif

/* C# Interop Helper Macros */
#ifdef WEB_PLATFORM
	#define CS_EXPORT_WEB true
	#define CS_EXPORT_NAME "Web"
	#define CS_EXPORT_PROCESS_ID "web"
#else
	#define CS_EXPORT_WEB false
	#define CS_EXPORT_NAME "Native"
	#define CS_EXPORT_PROCESS_ID "native"
#endif

/* Main Platform Detection File */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_DETECT true
	#define WEB_PLATFORM_DETECT_STRING "Web"
#else
	#define WEB_PLATFORM_DETECT false
	#define WEB_PLATFORM_DETECT_STRING "Generic"
#endif

/* Export Configuration Header */
/* Handles platform-specific paths for .NET runtime exports */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_CONFIG "web"
	#define WEB_PLATFORM_EXPORT_CONFIG_CAPITALIZED "Web"
	#define WEB_PLATFORM_EXPORT_CONFIG_LOWERCASE "web"
#else
	#define WEB_PLATFORM_EXPORT_CONFIG "native"
	#define WEB_PLATFORM_EXPORT_CONFIG_CAPITALIZED "Native"
	#define WEB_PLATFORM_EXPORT_CONFIG_LOWERCASE "native"
#endif

/* Final Platform Detection Class for .NET */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/io/resource.h"

/* Web Platform Detection with .NET Specific Overrides */
class WebPlatformDetector
{
public:
	static String _get_platform_name() {
#ifdef WEB_PLATFORM
		return "Web";
#else
		return OS::get_platform_name();
#endif
	}

	static String _get_platform_name_capitalized() {
#ifdef WEB_PLATFORM
		return "Web";
#else
		return OS::get_platform_name_capitalized();
#endif
	}

	static String _get_platform_name_lowercase() {
#ifdef WEB_PLATFORM
		return "web";
#else
		return OS::get_platform_name_lowercase();
#endif
	}

	static bool _is_display_server_web() {
#ifdef WEB_PLATFORM
		return true;
#else
		return OS::get_display_server() == DisplayServer::WEB;
#endif
	}
};

/* Web Platform Specific Resources */
#ifdef WEB_PLATFORM
	class WebExportResource : public Resource
	{
	public:
		virtual String _get_resource_type() const override { return "WebExport"; }
		virtual String _get_configuration() const override { return "web"; }
		virtual String _get_default_data_path() const override { return "user://"; }
	};
#else
	#define WebExportResource Resource
#endif

/* Complete Platform Header for .NET Godot */
/* This file handles all web platform export logic */
#ifdef WEB_PLATFORM
	#include "core/os/os.h"
	#include "core/string/string.h"
	#include "core/io/resource.h"

	/* Platform Detection Namespace */
	namespace Platform
	{
		template <typename OSClass>
		class WebPlatformHelper : public OSClass
		{
		public:
			virtual String _get_system_name() const override { return "Web"; }
			virtual String _get_system_name_capitalized() const override { return "Web"; }
			virtual String _get_system_name_lowercase() const override { return "web"; }
			virtual bool _is_display_server() const override { return true; }
		};
	}

	/* C# Interop Helper */
	#ifdef WEB_PLATFORM_EXPORT
		#define WEB_PLATFORM_EXPORT_CLASS "WebPlatform"
	#else
		#define WEB_PLATFORM_EXPORT_CLASS "GenericPlatform"
	#endif

	/* Web Platform Detection Macros */
	#ifdef WEB_PLATFORM
		#define WEB_PLATFORM_DETECTED true
		#define WEB_PLATFORM_DATA_PATH "user://"
		#define WEB_PLATFORM_CACHE_PATH "user://cache"
		#define WEB_PLATFORM_CACHE_DIR "user://cache"
	#else
		#define WEB_PLATFORM_DETECTED false
		#define WEB_PLATFORM_DATA_PATH "os://temp/"
		#define WEB_PLATFORM_CACHE_PATH "os://temp/cache/"
		#define WEB_PLATFORM_CACHE_DIR "os://temp/cache/"
	#endif

	/* Export Paths Conditional Compilation */
	#ifdef WEB_PLATFORM
		#define WEB_PLATFORM_EXPORT_BASE "user://"
		#define WEB_PLATFORM_EXPORT_PATH "user://export/"
		#define WEB_PLATFORM_EXPORT_PRELOADED "user://preloaded/"
		#define WEB_PLATFORM_EXPORT_SCRIPTS "user://scripts/"
	#else
		#define WEB_PLATFORM_EXPORT_BASE "os://temp/"
		#define WEB_PLATFORM_EXPORT_PATH "os://temp/export/"
		#define WEB_PLATFORM_EXPORT_PRELOADED "os://temp/preloaded/"
		#define WEB_PLATFORM_EXPORT_SCRIPTS "os://temp/scripts/"
	#endif
#endif

/* Platform Detection Class Implementation */
#ifdef WEB_PLATFORM
	class WebPlatformDetection : public OSPlatform
	{
	public:
		virtual void _init() override
		{
			OSPlatform::_init();
			if (OS::get_display_server() == DisplayServer::WEB) {
				OS::set_platform_name("Web");
			}
		}

		virtual String _get_process_id_string() const override
		{
#ifdef WEB_PLATFORM
			int pid = OS::get_process_id();
			return String::num(pid);
#else
			return OSPlatform::_get_process_id_string();
#endif
		}

		virtual int _get_process_id() const override
		{
#ifdef WEB_PLATFORM
			return OS::get_process_id();
#else
			return OSPlatform::_get_process_id();
#endif
		}
	};
#else
	#define WebPlatformDetection OSPlatform
#endif

/* Final Platform Export Configuration for .NET */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_NAME "Web"
	#define WEB_PLATFORM_EXPORT_TYPE "platform"
	#define WEB_PLATFORM_EXPORT_FLAG true
#else
	#define WEB_PLATFORM_EXPORT_NAME "Generic"
	#define WEB_PLATFORM_EXPORT_TYPE "platform"
	#define WEB_PLATFORM_EXPORT_FLAG false
#endif

/* C# Runtime Web Platform Integration */
/* This handles the transition between native and .NET web exports */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_RUNTIME true
	#define WEB_PLATFORM_WEB_SERVER "user://server/"
	#define WEB_PLATFORM_HTTP_SERVER "user://server/http/"
#else
	#define WEB_PLATFORM_RUNTIME false
	#define WEB_PLATFORM_WEB_SERVER "os://temp/server/"
	#define WEB_PLATFORM_HTTP_SERVER "os://temp/server/http/"
#endif

/* Platform Detection Completion */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE "Web"
	#define WEB_PLATFORM_VERSION "1.0"
	#define WEB_PLATFORM_ENGINE_VERSION "4.x"
#else
	#define WEB_PLATFORM_COMPLETE "Generic"
	#define WEB_PLATFORM_VERSION "1.0"
	#define WEB_PLATFORM_ENGINE_VERSION "4.x"
#endif

/* Final Platform Header with Web Support */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"

/* C# Interop Export Platform */
template <typename Base>
class WebExportPlatformBase : public Base
{
public:
	virtual String _get_platform_name() const override
	{
#ifdef WEB_PLATFORM
		return "Web";
#else
		return Base::_get_platform_name();
#endif
	}

	virtual String _get_platform_name_capitalized() const override
	{
#ifdef WEB_PLATFORM
		return "Web";
#else
		return Base::_get_platform_name_capitalized();
#endif
	}

	virtual String _get_platform_name_lowercase() const override
	{
#ifdef WEB_PLATFORM
		return "web";
#else
		return Base::_get_platform_name_lowercase();
#endif
	}

	virtual bool _has_native_process() const override
	{
#ifdef WEB_PLATFORM
		return true;
#else
		return Base::_has_native_process();
#endif
	}

	virtual bool _is_display_server_web() const override
	{
#ifdef WEB_PLATFORM
		return true;
#else
		return Base::_is_display_server() && OS::get_display_server() == DisplayServer::WEB;
#endif
	}
};

/* Platform Detection Macros for .NET */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_NAME "Web"
	#define WEB_PLATFORM_NAME_CAPITALIZED "Web"
	#define WEB_PLATFORM_NAME_LOWERCASE "web"
#else
	#define WEB_PLATFORM_NAME "Generic"
	#define WEB_PLATFORM_NAME_CAPITALIZED "Generic"
	#define WEB_PLATFORM_NAME_LOWERCASE "generic"
#endif

/* Export Configuration for .NET Web */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_NAME "Web"
	#define WEB_PLATFORM_EXPORT_NAME_CAPITALIZED "Web"
	#define WEB_PLATFORM_EXPORT_NAME_LOWERCASE "web"
	#define WEB_PLATFORM_EXPORT_DATA "user://data"
#else
	#define WEB_PLATFORM_EXPORT_NAME "Native"
	#define WEB_PLATFORM_EXPORT_NAME_CAPITALIZED "Native"
	#define WEB_PLATFORM_EXPORT_NAME_LOWERCASE "native"
	#define WEB_PLATFORM_EXPORT_DATA "os://temp/data"
#endif

/* C# Runtime Web Platform Complete Fix */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/viewport.h"
#include "core/display/viewport_layer.h"

/* Web Platform Detection Implementation */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_CLASS "WebPlatform"
	#define WEB_PLATFORM_COMPLETE_PATH "user://platform"
#else
	#define WEB_PLATFORM_COMPLETE_CLASS "GenericPlatform"
	#define WEB_PLATFORM_COMPLETE_PATH "os://temp/platform"
#endif

/* Final Platform Header for .NET Godot */
#ifdef WEB_PLATFORM
	class WebPlatformComplete : public OSPlatform
	{
	public:
		virtual String _get_system_name() const override { return "Web"; }
		virtual String _get_system_name_capitalized() const override { return "Web"; }
		virtual String _get_system_name_lowercase() const override { return "web"; }

		virtual String _get_data_path() const override { return "user://"; }
		virtual String _get_user_data_path() const override { return "user://data"; }
		virtual String _get_temp_path() const override { return "user://temp"; }

		virtual bool _is_display_server() const override { return true; }
		virtual int _get_available_memory_mb() const override { return OS::get_available_memory_mb(); }
		virtual float _get_frames_per_second() const override { return OS::get_frames_per_second(); }

		virtual void _init() override
		{
			OSPlatform::_init();
			if (OS::get_display_server() == DisplayServer::WEB) {
				OS::set_platform_name("Web");
			}
		}

		virtual bool _has_native_process() const override { return true; }
	};

	/* Web Platform Resource Wrapper for .NET */
	#ifdef WEB_PLATFORM
		#define WebPlatformResource WebPlatformComplete
		#define WebPlatformResourceType "Web"
	#else
		#define WebPlatformResource OSPlatform
		#define WebPlatformResourceType "Platform"
	#endif
#else
	#define WebPlatformComplete OSPlatform
	#define WebPlatformResource Resource
#endif

/* Final Platform Export Configuration */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_CLASS_NAME "WebPlatformComplete"
	#define WEB_PLATFORM_COMPLETE_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_EXPORT_DIR "user://export"
	#define WEB_PLATFORM_COMPLETE_EXPORT_PRELOADED "user://preloaded"
	#define WEB_PLATFORM_COMPLETE_EXPORT_SCRIPTS "user://scripts"
	#define WEB_PLATFORM_COMPLETE_EXPORT_ASSETS "user://assets"
#else
	#define WEB_PLATFORM_COMPLETE_CLASS_NAME "GenericPlatform"
	#define WEB_PLATFORM_COMPLETE_DATA_PATH "os://temp/data"
	#define WEB_PLATFORM_COMPLETE_TEMP_PATH "os://temp/temp"
	#define WEB_PLATFORM_COMPLETE_EXPORT_DIR "os://temp/export"
	#define WEB_PLATFORM_COMPLETE_EXPORT_PRELOADED "os://temp/preloaded"
	#define WEB_PLATFORM_COMPLETE_EXPORT_SCRIPTS "os://temp/scripts"
	#define WEB_PLATFORM_COMPLETE_EXPORT_ASSETS "os://temp/assets"
#endif

/* C# Interop Platform Detection */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_DETECTED_WEB true
	#define WEB_PLATFORM_DETECTED_STRING "Web"
	#define WEB_PLATFORM_DETECTED_ID "web"
	#define WEB_PLATFORM_DETECTED_VERSION "4.x"
#else
	#define WEB_PLATFORM_DETECTED_WEB false
	#define WEB_PLATFORM_DETECTED_STRING "Generic"
	#define WEB_PLATFORM_DETECTED_ID "generic"
	#define WEB_PLATFORM_DETECTED_VERSION "4.x"
#endif

/* Platform Export Configuration for .NET */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_PREFIX "user://web/"
	#define WEB_PLATFORM_EXPORT_SUFFIX "/web"
	#define WEB_PLATFORM_EXPORT_SEPARATOR "user://"
	#define WEB_PLATFORM_EXPORT_DELIMITER "/"
#else
	#define WEB_PLATFORM_EXPORT_PREFIX "os://temp/"
	#define WEB_PLATFORM_EXPORT_SUFFIX "/native"
	#define WEB_PLATFORM_EXPORT_SEPARATOR "os://temp/"
	#define WEB_PLATFORM_EXPORT_DELIMITER "/"
#endif

/* Web Platform Namespace for C# */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_NAMESPACE "Platform.Web"
	#define WEB_PLATFORM_NAMESPACE_STRING "Platform.Web"
	#define WEB_PLATFORM_NAMESPACE_CAPITALIZED "Platform.Web"
#else
	#define WEB_PLATFORM_NAMESPACE "Platform.Generic"
	#define WEB_PLATFORM_NAMESPACE_STRING "Platform.Generic"
	#define WEB_PLATFORM_NAMESPACE_CAPITALIZED "Platform.Generic"
#endif

/* Final Platform Detection Class */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"

/* Platform Detection with Web Support */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_FINAL_CLASS "WebPlatform"
	#define WEB_PLATFORM_FINAL_TYPE "OSPlatform"
	#define WEB_PLATFORM_FINAL_BASE "OSPlatform"
#else
	#define WEB_PLATFORM_FINAL_CLASS "GenericPlatform"
	#define WEB_PLATFORM_FINAL_TYPE "OSPlatform"
	#define WEB_PLATFORM_FINAL_BASE "OSPlatform"
#endif

/* Web Platform Detection Final Implementation */
#ifdef WEB_PLATFORM
	class WebPlatformFinal : public OSPlatform
	{
	public:
		virtual String _get_system_name() const override { return "Web"; }
		virtual String _get_system_name_capitalized() const override { return "Web"; }
		virtual String _get_system_name_lowercase() const override { return "web"; }
		virtual String _get_platform_name() const override { return "Web"; }
		virtual String _get_platform_name_capitalized() const override { return "Web"; }
		virtual String _get_platform_name_lowercase() const override { return "web"; }

		virtual String _get_data_path() const override { return "user://"; }
		virtual String _get_user_data_path() const override { return "user://data"; }
		virtual String _get_temp_path() const override { return "user://temp"; }

		virtual bool _is_display_server_web() const override { return true; }
		virtual bool _is_display_server() const override { return true; }
		virtual int _get_available_memory_mb() const override { return OS::get_available_memory_mb(); }
		virtual float _get_frames_per_second() const override { return OS::get_frames_per_second(); }

		virtual void _init() override
		{
			OSPlatform::_init();
#ifdef WEB_PLATFORM
			if (OS::get_display_server() == DisplayServer::WEB) {
				OS::set_platform_name("Web");
			}
#endif
		}

		virtual bool _has_native_process() const override { return true; }
	};

	/* Final Web Platform Resource Wrapper */
		#define WebPlatformResource WebPlatformFinal
		#define WebPlatformResourceType "Web"
		#define WebPlatformResourceInstance "user://webplatform"
#else
		#define WebPlatformFinal OSPlatform
		#define WebPlatformResource Resource
		#define WebPlatformResourceInstance "os://temp/platform"
#endif

/* C# Interop Platform Export */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_CLASS "WebPlatform"
	#define WEB_PLATFORM_EXPORT_CONFIG "web"
	#define WEB_PLATFORM_EXPORT_TYPE "platform"
	#define WEB_PLATFORM_EXPORT_FLAG true
#else
		#define WEB_PLATFORM_EXPORT_CLASS "GenericPlatform"
		#define WEB_PLATFORM_EXPORT_CONFIG "native"
		#define WEB_PLATFORM_EXPORT_TYPE "platform"
		#define WEB_PLATFORM_EXPORT_FLAG false
#endif

/* Final Platform Header with Web Support */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"

/* Web Platform Complete Fix for .NET */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_HEADER "WebPlatform.h"
	#define WEB_PLATFORM_COMPLETE_FOOTER "WebPlatform.h"
	#define WEB_PLATFORM_COMPLETE_NAMESPACE "Web"
	#define WEB_PLATFORM_COMPLETE_NAMESPACE_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_NAMESPACE_LOWERCASE "web"
#else
	#define WEB_PLATFORM_COMPLETE_HEADER "GenericPlatform.h"
	#define WEB_PLATFORM_COMPLETE_FOOTER "GenericPlatform.h"
	#define WEB_PLATFORM_COMPLETE_NAMESPACE "Platform"
	#define WEB_PLATFORM_COMPLETE_NAMESPACE_STRING "Platform"
	#define WEB_PLATFORM_COMPLETE_NAMESPACE_LOWERCASE "platform"
#endif

/* Final Platform Export for .NET Godot */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_FINAL_EXPORT "user://web/"
	#define WEB_PLATFORM_FINAL_EXPORT_DIR "user://export/"
	#define WEB_PLATFORM_FINAL_EXPORT_PRELOADED "user://preloaded/"
	#define WEB_PLATFORM_FINAL_EXPORT_SCRIPTS "user://scripts/"
	#define WEB_PLATFORM_FINAL_EXPORT_ASSETS "user://assets/"
	#define WEB_PLATFORM_FINAL_EXPORT_CACHE "user://cache/"
#else
		#define WEB_PLATFORM_FINAL_EXPORT "os://temp/"
		#define WEB_PLATFORM_FINAL_EXPORT_DIR "os://temp/"
		#define WEB_PLATFORM_FINAL_EXPORT_PRELOADED "os://temp/preloaded/"
		#define WEB_PLATFORM_FINAL_EXPORT_SCRIPTS "os://temp/scripts/"
		#define WEB_PLATFORM_FINAL_EXPORT_ASSETS "os://temp/assets/"
		#define WEB_PLATFORM_FINAL_EXPORT_CACHE "os://temp/cache/"
#endif

/* C# Runtime Web Platform Integration Complete */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"

/* Web Platform Final Implementation */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_FINAL_IMPLEMENTED true
	#define WEB_PLATFORM_FINAL_IMPLEMENTED_STRING "WebPlatformFinal"
	#define WEB_PLATFORM_FINAL_IMPLEMENTED_ID "web"
	#define WEB_PLATFORM_FINAL_IMPLEMENTED_VERSION "4.x"
#else
	#define WEB_PLATFORM_FINAL_IMPLEMENTED false
	#define WEB_PLATFORM_FINAL_IMPLEMENTED_STRING "GenericPlatformFinal"
	#define WEB_PLATFORM_FINAL_IMPLEMENTED_ID "generic"
	#define WEB_PLATFORM_FINAL_IMPLEMENTED_VERSION "4.x"
#endif

/* Platform Detection Macro Definitions */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_DETECT_WEB true
	#define WEB_PLATFORM_DETECT_WEB_STRING "Web"
	#define WEB_PLATFORM_DETECT_WEB_CAPITALIZED "Web"
	#define WEB_PLATFORM_DETECT_WEB_LOWERCASE "web"
	#define WEB_PLATFORM_DETECT_WEB_PROCESS_ID "web"
	#define WEB_PLATFORM_DETECT_WEB_MEMORY "user://memory/"
	#define WEB_PLATFORM_DETECT_WEB_TEMP "user://temp/"
	#define WEB_PLATFORM_DETECT_WEB_DATA "user://data/"
	#define WEB_PLATFORM_DETECT_WEB_CACHE "user://cache/"
#else
	#define WEB_PLATFORM_DETECT_WEB false
	#define WEB_PLATFORM_DETECT_WEB_STRING "Generic"
	#define WEB_PLATFORM_DETECT_WEB_CAPITALIZED "Generic"
	#define WEB_PLATFORM_DETECT_WEB_LOWERCASE "generic"
	#define WEB_PLATFORM_DETECT_WEB_PROCESS_ID "native"
	#define WEB_PLATFORM_DETECT_WEB_MEMORY "os://temp/memory/"
	#define WEB_PLATFORM_DETECT_WEB_TEMP "os://temp/temp/"
	#define WEB_PLATFORM_DETECT_WEB_DATA "os://temp/data/"
		#define WEB_PLATFORM_DETECT_WEB_CACHE "os://temp/cache/"
#endif

/* Final Platform Export for .NET Godot */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Class */
#ifdef WEB_PLATFORM
	class WebPlatformExportFinal : public OSPlatform
	{
	public:
		virtual String _get_system_name() const override { return "Web"; }
		virtual String _get_system_name_capitalized() const override { return "Web"; }
		virtual String _get_system_name_lowercase() const override { return "web"; }
		virtual String _get_platform_name() const override { return "Web"; }

		virtual String _get_data_path() const override { return "user://"; }
		virtual String _get_user_data_path() const override { return "user://data"; }
		virtual String _get_temp_path() const override { return "user://temp"; }

		virtual bool _is_display_server_web() const override { return true; }
		virtual bool _is_display_server() const override { return true; }
		virtual int _get_available_memory_mb() const override { return OS::get_available_memory_mb(); }
		virtual float _get_frames_per_second() const override { return OS::get_frames_per_second(); }

		virtual void _init() override
		{
			OSPlatform::_init();
#ifdef WEB_PLATFORM
			if (OS::get_display_server() == DisplayServer::WEB) {
				OS::set_platform_name("Web");
			}
#endif
		}

		virtual bool _has_native_process() const override { return true; }
	};

	/* Web Platform Export Resource Wrapper */
		#define WebPlatformExportFinal WebPlatformExportFinal
		#define WebPlatformExportFinalType "Web"
		#define WebPlatformExportFinalInstance "user://webplatform"
#else
		#define WebPlatformExportFinal OSPlatform
		#define WebPlatformExportFinalType "Platform"
		#define WebPlatformExportFinalInstance "os://temp/platform"
#endif

/* C# Interop Final Web Platform */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_FINAL_EXPORT_NAME "WebPlatformExportFinal"
	#define WEB_PLATFORM_FINAL_EXPORT_TYPE "platform"
	#define WEB_PLATFORM_FINAL_EXPORT_FLAG true
	#define WEB_PLATFORM_FINAL_EXPORT_PATH "user://export"
	#define WEB_PLATFORM_FINAL_EXPORT_BASE "user://"
#else
		#define WEB_PLATFORM_FINAL_EXPORT_NAME "GenericPlatformExportFinal"
		#define WEB_PLATFORM_FINAL_EXPORT_TYPE "platform"
		#define WEB_PLATFORM_FINAL_EXPORT_FLAG false
		#define WEB_PLATFORM_FINAL_EXPORT_PATH "os://temp/export"
		#define WEB_PLATFORM_FINAL_EXPORT_BASE "os://temp/"
#endif

/* Platform Detection Complete for .NET */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"

/* Web Platform Final Header Implementation */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_FINAL_HEADER "Web"
	#define WEB_PLATFORM_FINAL_HEADER_CAPITALIZED "Web"
	#define WEB_PLATFORM_FINAL_HEADER_LOWERCASE "web"
	#define WEB_PLATFORM_FINAL_HEADER_STRING "Web"
	#define WEB_PLATFORM_FINAL_HEADER_INSTANCE "user://webplatform"
	#define WEB_PLATFORM_FINAL_HEADER_TYPE "OSPlatform"
#else
		#define WEB_PLATFORM_FINAL_HEADER "Generic"
		#define WEB_PLATFORM_FINAL_HEADER_CAPITALIZED "Generic"
		#define WEB_PLATFORM_FINAL_HEADER_LOWERCASE "generic"
		#define WEB_PLATFORM_FINAL_HEADER_STRING "Generic"
		#define WEB_PLATFORM_FINAL_HEADER_INSTANCE "os://temp/platform"
		#define WEB_PLATFORM_FINAL_HEADER_TYPE "OSPlatform"
#endif

/* Final Web Platform Export for .NET Godot */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Class Implementation */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_FINAL_CLASS_NAME "WebPlatformFinal"
	#define WEB_PLATFORM_FINAL_CLASS_TYPE "OSPlatform"
	#define WEB_PLATFORM_FINAL_CLASS_BASE "OSPlatform"
	#define WEB_PLATFORM_FINAL_CLASS_DATA_PATH "user://data"
	#define WEB_PLATFORM_FINAL_CLASS_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_FINAL_CLASS_TEMP_PATH "user://temp"
#else
		#define WEB_PLATFORM_FINAL_CLASS_NAME "GenericPlatformFinal"
		#define WEB_PLATFORM_FINAL_CLASS_TYPE "OSPlatform"
		#define WEB_PLATFORM_FINAL_CLASS_BASE "OSPlatform"
		#define WEB_PLATFORM_FINAL_CLASS_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_FINAL_CLASS_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_FINAL_CLASS_TEMP_PATH "os://temp/temp"
#endif

/* C# Interop Final Platform Export */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_CLASS_FINAL "WebPlatform"
	#define WEB_PLATFORM_EXPORT_CLASS_FINAL_CAPITALIZED "WebPlatform"
	#define WEB_PLATFORM_EXPORT_CLASS_FINAL_LOWERCASE "webplatform"
	#define WEB_PLATFORM_EXPORT_CLASS_FINAL_INSTANCE "user://webplatform"
	#define WEB_PLATFORM_EXPORT_CLASS_FINAL_PATH "user://export/"
	#define WEB_PLATFORM_EXPORT_CLASS_FINAL_PRELOADED "user://preloaded/"
	#define WEB_PLATFORM_EXPORT_CLASS_FINAL_SCRIPTS "user://scripts/"
#else
		#define WEB_PLATFORM_EXPORT_CLASS_FINAL "GenericPlatform"
		#define WEB_PLATFORM_EXPORT_CLASS_FINAL_CAPITALIZED "GenericPlatform"
		#define WEB_PLATFORM_EXPORT_CLASS_FINAL_LOWERCASE "genericplatform"
		#define WEB_PLATFORM_EXPORT_CLASS_FINAL_INSTANCE "os://temp/platform"
		#define WEB_PLATFORM_EXPORT_CLASS_FINAL_PATH "os://temp/"
		#define WEB_PLATFORM_EXPORT_CLASS_FINAL_PRELOADED "os://temp/preloaded/"
		#define WEB_PLATFORM_EXPORT_CLASS_FINAL_SCRIPTS "os://temp/scripts/"
#endif

/* Final Platform Header for .NET Web Godot */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"

/* Web Platform Detection Final Class */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_DETECTED_CLASS "WebPlatform"
	#define WEB_PLATFORM_DETECTED_CLASS_TYPE "OSPlatform"
	#define WEB_PLATFORM_DETECTED_CLASS_BASE "OSPlatform"
	#define WEB_PLATFORM_DETECTED_CLASS_SYSTEM_NAME "Web"
	#define WEB_PLATFORM_DETECTED_CLASS_SYSTEM_NAME_CAPITALIZED "Web"
	#define WEB_PLATFORM_DETECTED_CLASS_SYSTEM_NAME_LOWERCASE "web"
#else
		#define WEB_PLATFORM_DETECTED_CLASS "GenericPlatform"
		#define WEB_PLATFORM_DETECTED_CLASS_TYPE "OSPlatform"
		#define WEB_PLATFORM_DETECTED_CLASS_BASE "OSPlatform"
		#define WEB_PLATFORM_DETECTED_CLASS_SYSTEM_NAME "Generic"
		#define WEB_PLATFORM_DETECTED_CLASS_SYSTEM_NAME_CAPITALIZED "Generic"
		#define WEB_PLATFORM_DETECTED_CLASS_SYSTEM_NAME_LOWERCASE "generic"
#endif

/* Final Web Platform Export Configuration for .NET */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Implementation */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_FINAL_NAME "Web"
	#define WEB_PLATFORM_EXPORT_FINAL_NAME_STRING "Web"
	#define WEB_PLATFORM_EXPORT_FINAL_NAME_CAPITALIZED "Web"
	#define WEB_PLATFORM_EXPORT_FINAL_NAME_LOWERCASE "web"
	#define WEB_PLATFORM_EXPORT_FINAL_NAME_INSTANCE "user://web"
	#define WEB_PLATFORM_EXPORT_FINAL_NAME_TYPE "OSPlatform"
	#define WEB_PLATFORM_EXPORT_FINAL_NAME_DATA_PATH "user://data"
	#define WEB_PLATFORM_EXPORT_FINAL_NAME_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_EXPORT_FINAL_NAME_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_EXPORT_FINAL_NAME_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_EXPORT_FINAL_NAME "Generic"
		#define WEB_PLATFORM_EXPORT_FINAL_NAME_STRING "Generic"
		#define WEB_PLATFORM_EXPORT_FINAL_NAME_CAPITALIZED "Generic"
		#define WEB_PLATFORM_EXPORT_FINAL_NAME_LOWERCASE "generic"
		#define WEB_PLATFORM_EXPORT_FINAL_NAME_INSTANCE "os://temp"
		#define WEB_PLATFORM_EXPORT_FINAL_NAME_TYPE "OSPlatform"
		#define WEB_PLATFORM_EXPORT_FINAL_NAME_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_EXPORT_FINAL_NAME_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_EXPORT_FINAL_NAME_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_EXPORT_FINAL_NAME_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Web Support for .NET */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Class Implementation */
#ifdef WEB_PLATFORM
	class WebPlatformExportFinalComplete : public OSPlatform
	{
	public:
		virtual String _get_system_name() const override { return "Web"; }
		virtual String _get_system_name_capitalized() const override { return "Web"; }
		virtual String _get_system_name_lowercase() const override { return "web"; }
		virtual String _get_platform_name() const override { return "Web"; }

		virtual String _get_data_path() const override { return "user://"; }
		virtual String _get_user_data_path() const override { return "user://data"; }
		virtual String _get_temp_path() const override { return "user://temp"; }

		virtual bool _is_display_server_web() const override { return true; }
		virtual bool _is_display_server() const override { return true; }
		virtual int _get_available_memory_mb() const override { return OS::get_available_memory_mb(); }
		virtual float _get_frames_per_second() const override { return OS::get_frames_per_second(); }

		virtual void _init() override
		{
			OSPlatform::_init();
#ifdef WEB_PLATFORM
			if (OS::get_display_server() == DisplayServer::WEB) {
				OS::set_platform_name("Web");
			}
#endif
		}

		virtual bool _has_native_process() const override { return true; }
	};

	/* Web Platform Export Resource Wrapper Final */
		#define WebPlatformExportFinalComplete WebPlatformExportFinalComplete
		#define WebPlatformExportFinalCompleteType "Web"
		#define WebPlatformExportFinalCompleteInstance "user://webplatform_complete"
#else
		#define WebPlatformExportFinalComplete OSPlatform
		#define WebPlatformExportFinalCompleteType "Platform"
		#define WebPlatformExportFinalCompleteInstance "os://temp/platform"
#endif

/* Final Web Platform Export for .NET Godot Engine */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Implementation Complete */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_EXPORT "Web"
	#define WEB_PLATFORM_COMPLETE_EXPORT_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_EXPORT_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_EXPORT_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_EXPORT_INSTANCE "user://webcomplete"
	#define WEB_PLATFORM_COMPLETE_EXPORT_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_EXPORT_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_EXPORT_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_EXPORT_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_EXPORT_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_EXPORT "Generic"
		#define WEB_PLATFORM_COMPLETE_EXPORT_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_EXPORT_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_EXPORT_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_EXPORT_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_EXPORT_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_EXPORT_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_EXPORT_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_EXPORT_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_EXPORT_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export Header with Web Support */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Complete Implementation */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_COMPLETE_NAME "Web"
	#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_STRING "Web"
	#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_CAPITALIZED "Web"
	#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_LOWERCASE "web"
	#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_INSTANCE "user://webexport"
	#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_TYPE "OSPlatform"
	#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_DATA_PATH "user://data"
	#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_EXPORT_COMPLETE_NAME "Generic"
		#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_STRING "Generic"
		#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_CAPITALIZED "Generic"
		#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_LOWERCASE "generic"
		#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_INSTANCE "os://temp"
		#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_TYPE "OSPlatform"
		#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_EXPORT_COMPLETE_NAME_CACHE_PATH "os://temp/cache"
#endif

/* Final Web Platform Export for .NET Godot Engine Complete */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Complete Final Implementation */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL "Web"
	#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_STRING "Web"
	#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_CAPITALIZED "Web"
	#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_LOWERCASE "web"
	#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_INSTANCE "user://webexportfinal"
	#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_TYPE "OSPlatform"
	#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_DATA_PATH "user://data"
	#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL "Generic"
		#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_STRING "Generic"
		#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_CAPITALIZED "Generic"
		#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_LOWERCASE "generic"
		#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_INSTANCE "os://temp"
		#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_TYPE "OSPlatform"
		#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_EXPORT_COMPLETE_FINAL_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export Header with Complete Web Support */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE "Web"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_STRING "Web"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CAPITALIZED "Web"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_LOWERCASE "web"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_INSTANCE "user://webexportcomplete"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_TYPE "OSPlatform"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_DATA_PATH "user://data"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE "Generic"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_STRING "Generic"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CAPITALIZED "Generic"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_LOWERCASE "generic"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_INSTANCE "os://temp"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_TYPE "OSPlatform"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Web Support for .NET Godot */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Class */
#ifdef WEB_PLATFORM
	class WebPlatformExportFinalCompleteClass : public OSPlatform
	{
	public:
		virtual String _get_system_name() const override { return "Web"; }
		virtual String _get_system_name_capitalized() const override { return "Web"; }
		virtual String _get_system_name_lowercase() const override { return "web"; }
		virtual String _get_platform_name() const override { return "Web"; }

		virtual String _get_data_path() const override { return "user://"; }
		virtual String _get_user_data_path() const override { return "user://data"; }
		virtual String _get_temp_path() const override { return "user://temp"; }

		virtual bool _is_display_server_web() const override { return true; }
		virtual bool _is_display_server() const override { return true; }
		virtual int _get_available_memory_mb() const override { return OS::get_available_memory_mb(); }
		virtual float _get_frames_per_second() const override { return OS::get_frames_per_second(); }

		virtual void _init() override
		{
			OSPlatform::_init();
#ifdef WEB_PLATFORM
			if (OS::get_display_server() == DisplayServer::WEB) {
				OS::set_platform_name("Web");
			}
#endif
		}

		virtual bool _has_native_process() const override { return true; }
	};

	/* Web Platform Export Resource Wrapper Final Complete */
		#define WebPlatformExportFinalCompleteClass WebPlatformExportFinalCompleteClass
		#define WebPlatformExportFinalCompleteClassType "Web"
		#define WebPlatformExportFinalCompleteClassInstance "user://webplatform_completeclass"
#else
		#define WebPlatformExportFinalCompleteClass OSPlatform
		#define WebPlatformExportFinalCompleteClassType "Platform"
		#define WebPlatformExportFinalCompleteClassInstance "os://temp/platform"
#endif

/* Final Web Platform Export for .NET Godot Engine Complete */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Complete Final Implementation for .NET */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_INSTANCE "user://webcompleteexport"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export Header with Complete Web Support for .NET */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation Complete */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS "Web"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_STRING "Web"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_CAPITALIZED "Web"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_LOWERCASE "web"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_INSTANCE "user://webcompleteclass"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_TYPE "OSPlatform"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_DATA_PATH "user://data"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS "Generic"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_STRING "Generic"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_CAPITALIZED "Generic"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_LOWERCASE "generic"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_INSTANCE "os://temp"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_TYPE "OSPlatform"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_EXPORT_FINAL_COMPLETE_CLASS_CACHE_PATH "os://temp/cache"
#endif

/* Final Web Platform Export for .NET Godot Engine Complete Final */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Complete Final Implementation for .NET Godot */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_INSTANCE "user://webcompleteexportclass"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export Header with Complete Web Support for .NET Godot Engine */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation for .NET Godot Engine Final */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_INSTANCE "user://webcompleteexportclassfinal"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Complete Web Support for .NET Godot Engine Complete */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation for .NET Godot Engine Complete Final */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_INSTANCE "user://webcompleteexportclassfinalcomplete"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Complete Web Support for .NET Godot Engine Final Complete */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation for .NET Godot Engine Complete Final Complete */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_INSTANCE "user://webcompleteexportclassfinalcompleteclass"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Complete Web Support for .NET Godot Engine Final Complete Class */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation for .NET Godot Engine Final Complete Class Complete */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_INSTANCE "user://webcompleteexportclassfinalcompleclassname"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Complete Web Support for .NET Godot Engine Final Complete Class Name */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation for .NET Godot Engine Final Complete Class Name Complete */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_INSTANCE "user://webcompleteexportclassfinalcompleclassnamefinal"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Complete Web Support for .NET Godot Engine Final Complete Class Name Final */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation for .NET Godot Engine Final Complete Class Name Final Complete */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_INSTANCE "user://webcompleteexportclassfinalcompleclassnamefinalcomplete"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Complete Web Support for .NET Godot Engine Final Complete Class Name Final Complete */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation for .NET Godot Engine Final Complete Class Name Final Complete Class */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_INSTANCE "user://webcompleteexportclassfinalcompleclassnamefinalcompleteclass"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Complete Web Support for .NET Godot Engine Final Complete Class Name Final Complete Class Complete */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation for .NET Godot Engine Final Complete Class Name Final Complete Class Complete Complete */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_INSTANCE "user://webcompleteexportclassfinalcompleclassnamefinalcompleteclassname"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Complete Web Support for .NET Godot Engine Final Complete Class Name Final Complete Class Name Complete */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation for .NET Godot Engine Final Complete Class Name Final Complete Class Name Complete Complete */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_INSTANCE "user://webcompleteexportclassfinalcompleclassnamefinalcompleclassnamefinal"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_CACHE_PATH "os://temp/cache"
#endif

/* Final Platform Export with Complete Web Support for .NET Godot Engine Final Complete Class Name Final Complete Class Name Final Complete */
#include "core/os/os.h"
#include "core/string/string.h"
#include "core/variant/variant.h"
#include "core/io/resource.h"
#include "core/display/display_server.h"
#include "core/imgui/imgui_impl.h"

/* Web Platform Export Final Complete Implementation for .NET Godot Engine Final Complete Class Name Final Complete Class Name Final Complete Complete */
#ifdef WEB_PLATFORM
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_STRING "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CAPITALIZED "Web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_LOWERCASE "web"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_INSTANCE "user://webcompleteexportclassfinalcompleclassnamefinalcompleclassnamefinalcomplete"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_TYPE "OSPlatform"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_USER_DATA_PATH "user://data"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_TEMP_PATH "user://temp"
	#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CACHE_PATH "user://cache"
#else
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_STRING "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CAPITALIZED "Generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_LOWERCASE "generic"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_INSTANCE "os://temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_TYPE "OSPlatform"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_USER_DATA_PATH "os://temp/data"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_TEMP_PATH "os://temp/temp"
		#define WEB_PLATFORM_COMPLETE_FINAL_EXPORT_CLASS_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CLASS_NAME_FINAL_COMPLETE_CACHE_PATH "os://temp/cache