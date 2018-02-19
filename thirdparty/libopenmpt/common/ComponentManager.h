/*
 * ComponentManager.h
 * ------------------
 * Purpose: Manages loading of optional components.
 * Notes  : (currently none)
 * Authors: Joern Heusipp
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

#include <map>
#include <vector>
#include "../common/misc_util.h"
#include "../common/mptMutex.h"


OPENMPT_NAMESPACE_BEGIN


#define MPT_ENABLE_COMPONENTS


#if defined(MPT_ENABLE_COMPONENTS)


#if defined(MODPLUG_TRACKER)
#define MPT_COMPONENT_MANAGER 1
#else
#define MPT_COMPONENT_MANAGER 0
#endif


enum ComponentType
{
	ComponentTypeUnknown = 0,
	ComponentTypeBuiltin,            // PortAudio
	ComponentTypeSystem,             // mf.dll
	ComponentTypeSystemInstallable,  // acm mp3 codec
	ComponentTypeBundled,            // libsoundtouch
	ComponentTypeForeign,            // libmp3lame
};


class ComponentFactoryBase;


class IComponent
{

	friend class ComponentFactoryBase;

protected:

	IComponent() { }

public:

	virtual ~IComponent() { }

public:

	virtual ComponentType GetType() const = 0;
	
	virtual bool IsInitialized() const = 0;  // Initialize() has been called
	virtual bool IsAvailable() const = 0;  // Initialize() has been successfull
	virtual mpt::ustring GetVersion() const = 0;

	virtual void Initialize() = 0;  // try to load the component

};


class ComponentBase
	: public IComponent
{

private:

	ComponentType m_Type;

	bool m_Initialized;
	bool m_Available;

protected:

	ComponentBase(ComponentType type);

public:

	virtual ~ComponentBase();

protected:

	void SetInitialized();
	void SetAvailable();

public:

	virtual ComponentType GetType() const;
	virtual bool IsInitialized() const;
	virtual bool IsAvailable() const;

	virtual mpt::ustring GetVersion() const;

public:

	virtual void Initialize();

protected:

	virtual bool DoInitialize() = 0;

};


class ComponentBuiltin : public ComponentBase
{
public:
	ComponentBuiltin()
		: ComponentBase(ComponentTypeBuiltin)
	{
		return;
	}
	virtual bool DoInitialize()
	{
		return true;
	}
};


#define MPT_GLOBAL_BIND(lib, name) name = &::name;


#if defined(MPT_ENABLE_DYNBIND)


class ComponentLibrary
	: public ComponentBase
{

private:
	
	typedef std::map<std::string, mpt::Library> TLibraryMap;
	TLibraryMap m_Libraries;
	
	bool m_BindFailed;

protected:

	ComponentLibrary(ComponentType type);

public:

	virtual ~ComponentLibrary();

protected:

	bool AddLibrary(const std::string &libName, const mpt::LibraryPath &libPath);
	void ClearLibraries();
	void SetBindFailed();
	void ClearBindFailed();
	bool HasBindFailed() const;

public:
	
	virtual mpt::Library GetLibrary(const std::string &libName) const;
	
	template <typename Tfunc>
	bool Bind(Tfunc * & f, const std::string &libName, const std::string &symbol) const
	{
		return GetLibrary(libName).Bind(f, symbol);
	}

protected:

	virtual bool DoInitialize() = 0;

};


#define MPT_COMPONENT_BIND(libName, func) MPT_DO { if(!Bind( func , libName , #func )) { SetBindFailed(); } } MPT_WHILE_0
#define MPT_COMPONENT_BIND_OPTIONAL(libName, func) Bind( func , libName , #func )
#define MPT_COMPONENT_BIND_SYMBOL(libName, symbol, func) MPT_DO { if(!Bind( func , libName , symbol )) { SetBindFailed(); } } MPT_WHILE_0
#define MPT_COMPONENT_BIND_SYMBOL_OPTIONAL(libName, symbol, func) Bind( func , libName , symbol )


class ComponentSystemDLL : public ComponentLibrary
{
private:
	mpt::PathString m_BaseName;
public:
	ComponentSystemDLL(const mpt::PathString &baseName)
		: ComponentLibrary(ComponentTypeSystem)
		, m_BaseName(baseName)
	{
		return;
	}
	virtual bool DoInitialize()
	{
		AddLibrary(m_BaseName.ToUTF8(), mpt::LibraryPath::System(m_BaseName));
		return GetLibrary(m_BaseName.ToUTF8()).IsValid();
	}
};


class ComponentBundledDLL : public ComponentLibrary
{
private:
	mpt::PathString m_FullName;
public:
	ComponentBundledDLL(const mpt::PathString &fullName)
		: ComponentLibrary(ComponentTypeBundled)
		, m_FullName(fullName)
	{
		return;
	}
	virtual bool DoInitialize()
	{
		AddLibrary(m_FullName.ToUTF8(), mpt::LibraryPath::AppFullName(m_FullName));
		return GetLibrary(m_FullName.ToUTF8()).IsValid();
	}
};


#endif // MPT_ENABLE_DYNBIND


#if MPT_COMPONENT_MANAGER


class ComponentManager;

typedef std::shared_ptr<IComponent> (*ComponentFactoryMethod)(ComponentManager &componentManager);


class IComponentFactory
{
protected:
	IComponentFactory() { }
public:
	virtual ~IComponentFactory() { }
public:
	virtual std::string GetID() const = 0;
	virtual std::string GetSettingsKey() const = 0;
	virtual std::shared_ptr<IComponent> Construct(ComponentManager &componentManager) const = 0;
	virtual ComponentFactoryMethod GetStaticConstructor() const = 0;
};


class ComponentFactoryBase
	: public IComponentFactory
{
private:
	std::string m_ID;
	std::string m_SettingsKey;
protected:
	ComponentFactoryBase(const std::string &id, const std::string &settingsKey);
	void PreConstruct() const;
	void Initialize(ComponentManager &componentManager, std::shared_ptr<IComponent> component) const;
public:
	virtual ~ComponentFactoryBase();
	virtual std::string GetID() const;
	virtual std::string GetSettingsKey() const;
	virtual std::shared_ptr<IComponent> Construct(ComponentManager &componentManager) const = 0;
	virtual ComponentFactoryMethod GetStaticConstructor() const = 0;
};


template <typename T>
class ComponentFactory
	: public ComponentFactoryBase
{
public:
	ComponentFactory()
		: ComponentFactoryBase(T::g_ID, T::g_SettingsKey)
	{
		return;
	}
	virtual ~ComponentFactory()
	{
		return;
	}
public:
	virtual std::shared_ptr<IComponent> Construct(ComponentManager &componentManager) const
	{
		PreConstruct();
		std::shared_ptr<IComponent> component = std::make_shared<T>();
		Initialize(componentManager, component);
		return component;
	}
	static std::shared_ptr<IComponent> StaticConstruct(ComponentManager &componentManager)
	{
		return ComponentFactory().Construct(componentManager);
	}
	virtual ComponentFactoryMethod GetStaticConstructor() const
	{
		return &StaticConstruct;
	}
};


class IComponentManagerSettings
{
public:
	virtual bool LoadOnStartup() const = 0;
	virtual bool KeepLoaded() const = 0;
	virtual bool IsBlocked(const std::string &key) const = 0;
	virtual mpt::PathString Path() const = 0;
};


class ComponentManagerSettingsDefault
	: public IComponentManagerSettings
{
public:
	virtual bool LoadOnStartup() const { return false; }
	virtual bool KeepLoaded() const { return true; }
	virtual bool IsBlocked(const std::string & /*key*/ ) const { return false; }
	virtual mpt::PathString Path() const { return mpt::PathString(); }
};


enum ComponentState
{
	ComponentStateUnregistered,
	ComponentStateBlocked,
	ComponentStateUnintialized,
	ComponentStateUnavailable,
	ComponentStateAvailable,
};


struct ComponentInfo
{
	std::string name;
	ComponentState state;
	std::string settingsKey;
	ComponentType type;
};


class ComponentManager
{
	friend class ComponentFactoryBase;
public:
	static void Init(const IComponentManagerSettings &settings);
	static void Release();
	static std::shared_ptr<ComponentManager> Instance();
private:
	ComponentManager(const IComponentManagerSettings &settings);
private:
	struct RegisteredComponent
	{
		std::string settingsKey;
		ComponentFactoryMethod factoryMethod;
		std::shared_ptr<IComponent> instance;
		std::weak_ptr<IComponent> weakInstance;
	};
	typedef std::map<std::string, RegisteredComponent> TComponentMap;
	const IComponentManagerSettings &m_Settings;
	TComponentMap m_Components;
private:
	bool IsComponentBlocked(const std::string &settingsKey) const;
	void InitializeComponent(std::shared_ptr<IComponent> component) const;
public:
	void Register(const IComponentFactory &componentFactory);
	void Startup();
	std::shared_ptr<const IComponent> GetComponent(const IComponentFactory &componentFactory);
	std::shared_ptr<const IComponent> ReloadComponent(const IComponentFactory &componentFactory);
	std::vector<std::string> GetRegisteredComponents() const;
	ComponentInfo GetComponentInfo(std::string name) const;
	mpt::PathString GetComponentPath() const;
};


struct ComponentListEntry
{
	ComponentListEntry *next;
	void (*reg)(ComponentManager &componentManager);
};
		
bool ComponentListPush(ComponentListEntry *entry);

#define MPT_DECLARE_COMPONENT_MEMBERS public: static const char * const g_ID; static const char * const g_SettingsKey;
		
#define MPT_REGISTERED_COMPONENT(name, settingsKey) \
	static void RegisterComponent ## name (ComponentManager &componentManager) \
	{ \
		componentManager.Register(ComponentFactory< name >()); \
	} \
	static ComponentListEntry Component ## name ## ListEntry = { nullptr, & RegisterComponent ## name }; \
	bool Component ## name ## Registered = ComponentListPush(& Component ## name ## ListEntry ); \
	const char * const name :: g_ID = #name ; \
	const char * const name :: g_SettingsKey = settingsKey ; \
/**/


template <typename type>
std::shared_ptr<const type> GetComponent()
{
	return std::dynamic_pointer_cast<const type>(ComponentManager::Instance()->GetComponent(ComponentFactory<type>()));
}


template <typename type>
std::shared_ptr<const type> ReloadComponent()
{
	return std::dynamic_pointer_cast<const type>(ComponentManager::Instance()->ReloadComponent(ComponentFactory<type>()));
}


static inline mpt::PathString GetComponentPath()
{
	return ComponentManager::Instance()->GetComponentPath();
}


#else // !MPT_COMPONENT_MANAGER


#define MPT_DECLARE_COMPONENT_MEMBERS

#define MPT_REGISTERED_COMPONENT(name, settingsKey)


template <typename type>
std::shared_ptr<const type> GetComponent()
{
	static std::weak_ptr<type> cache;
	static mpt::mutex m;
	MPT_LOCK_GUARD<mpt::mutex> l(m);
	std::shared_ptr<type> component = cache.lock();
	if(!component)
	{
		component = std::make_shared<type>();
		component->Initialize();
		cache = component;
	}
	return component;
}


static inline mpt::PathString GetComponentPath()
{
	return mpt::PathString();
}


#endif // MPT_COMPONENT_MANAGER


// Simple wrapper around std::shared_ptr<ComponentType> which automatically
// gets a reference to the component (or constructs it) on initialization.
template <typename T>
class ComponentHandle
{
private:
	std::shared_ptr<const T> component;
public:
	ComponentHandle()
		: component(GetComponent<T>())
	{
		return;
	}
	~ComponentHandle()
	{
		return;
	}
	bool IsAvailable() const
	{
		return component && component->IsAvailable();
	}
	const T *get() const
	{
		return component.get();
	}
	const T &operator*() const
	{
		return *component;
	}
	const T *operator->() const
	{
		return &*component;
	}
#if MPT_COMPONENT_MANAGER
	void Reload()
	{
		component = nullptr;
		component = ReloadComponent<T>();
	}
#endif
};


template <typename T>
bool IsComponentAvailable(const ComponentHandle<T> &handle)
{
	return handle.IsAvailable();
}


#endif // MPT_ENABLE_COMPONENTS


OPENMPT_NAMESPACE_END
