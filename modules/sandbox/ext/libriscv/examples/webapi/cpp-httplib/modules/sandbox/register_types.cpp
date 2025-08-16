#include "register_types.h"
#include "core/extension/gdextension_interface.h"
#include "core/extension/gdextension_loader.h"
#include "core/extension/gdextension_manager.h"
#include "core/io/config_file.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/os/shared_object.h"
#include <functional>

class GDExtensionStaticLibraryLoader : public GDExtensionLoader {
	friend class GDExtensionManager;
	friend class GDExtension;

private:
  void *entry_funcptr = nullptr;
  String library_path;

public:
	void set_entry_funcptr(void *p_entry_funcptr) {
		entry_funcptr = p_entry_funcptr;
	}
	virtual Error open_library(const String &p_path) override {
		library_path = p_path;
		return OK;
	}
	virtual Error initialize(GDExtensionInterfaceGetProcAddress p_get_proc_address,
			 const Ref<GDExtension> &p_extension,
			 GDExtensionInitialization *r_initialization) override {
	GDExtensionInitializationFunction initialization_function =
		(GDExtensionInitializationFunction)entry_funcptr;
	if (initialization_function == nullptr) {
	  ERR_PRINT("GDExtension initialization function '" + library_path +
				"' is null.");
	  return FAILED;
	}
	GDExtensionBool ret = initialization_function(
		p_get_proc_address, p_extension.ptr(), r_initialization);

	if (ret) {
	  return OK;
	} else {
	  ERR_PRINT("GDExtension initialization function '" + library_path +
				"' returned an error.");
	  return FAILED;
	}
  }
  virtual void close_library() override {}
  virtual bool is_library_open() const override { return true; }
  virtual bool has_library_changed() const override { return false; }
  virtual bool library_exists() const override { return true; }
};

extern "C" {
	GDExtensionBool riscv_library_init(
		GDExtensionInterfaceGetProcAddress p_get_proc_address,
		GDExtensionClassLibraryPtr p_library,
		GDExtensionInitialization *r_initialization
	);
}

void initialize_sandbox_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SERVERS) {
		return;
	}

	Ref<GDExtensionStaticLibraryLoader> loader;
	loader.instantiate();
	loader->set_entry_funcptr((void*)&riscv_library_init);
	GDExtensionManager::get_singleton()->load_extension_with_loader("sandbox", loader);
}

void uninitialize_sandbox_module(ModuleInitializationLevel p_level) {
}
