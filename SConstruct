EnsureSConsVersion(0,14);

import string
import os
import os.path
import glob
import sys
import methods
import multiprocessing

# Enable aggresive compile mode if building on a multi core box
# only is we have not set the number of jobs already or we do
# not want it
if ARGUMENTS.get('spawn_jobs', 'yes') == 'yes' and \
	int(GetOption('num_jobs')) <= 1:
	NUM_JOBS = multiprocessing.cpu_count()
	if NUM_JOBS > 1:
		SetOption('num_jobs', NUM_JOBS+1)

methods.update_version()

# scan possible build platforms

platform_list = [] # list of platforms
platform_opts = {} # options for each platform
platform_flags = {} # flags for each platform


active_platforms=[]
active_platform_ids=[]
platform_exporters=[]
global_defaults=[]

for x in glob.glob("platform/*"):
	if (not os.path.isdir(x)):
		continue
	tmppath="./"+x

	sys.path.append(tmppath)
	import detect

	if (os.path.exists(x+"/export/export.cpp")):
		platform_exporters.append(x[9:])
	if (os.path.exists(x+"/globals/global_defaults.cpp")):
		global_defaults.append(x[9:])
	if (detect.is_active()):
		active_platforms.append( detect.get_name() )
		active_platform_ids.append(x);
	if (detect.can_build()):
		x=x.replace("platform/","") # rest of world
		x=x.replace("platform\\","") # win32
		platform_list+=[x]
		platform_opts[x]=detect.get_opts()
		platform_flags[x]=detect.get_flags()
	sys.path.remove(tmppath)
	sys.modules.pop('detect')

module_list=methods.detect_modules()


print "Detected Platforms: "+str(platform_list)
print("Detected Modules: "+str(module_list))

methods.save_active_platforms(active_platforms,active_platform_ids)

custom_tools=['default']

if (os.name=="posix"):
	pass
elif (os.name=="nt"):
    if (os.getenv("VSINSTALLDIR")==None):
	custom_tools=['mingw']

env_base=Environment(tools=custom_tools,ENV = {'PATH' : os.environ['PATH']});
#env_base=Environment(tools=custom_tools);
env_base.global_defaults=global_defaults
env_base.android_source_modules=[]
env_base.android_source_files=[]
env_base.android_module_libraries=[]
env_base.android_manifest_chunk=""
env_base.disabled_modules=[]

env_base.__class__.android_module_source = methods.android_module_source
env_base.__class__.android_module_library = methods.android_module_library
env_base.__class__.android_module_file = methods.android_module_file
env_base.__class__.android_module_manifest = methods.android_module_manifest
env_base.__class__.disable_module = methods.disable_module

env_base.__class__.add_source_files = methods.add_source_files

customs = ['custom.py']

profile = ARGUMENTS.get("profile", False)
if profile:
	import os.path
	if os.path.isfile(profile):
		customs.append(profile)
	elif os.path.isfile(profile+".py"):
		customs.append(profile+".py")

opts=Variables(customs, ARGUMENTS)
opts.Add('target', 'Compile Target (debug/profile/release).', "debug")
opts.Add('platform','Platform: '+str(platform_list)+'(sfml).',"")
opts.Add('python','Build Python Support: (yes/no)','no')
opts.Add('squirrel','Build Squirrel Support: (yes/no)','no')
opts.Add('tools','Build Tools (Including Editor): (yes/no)','yes')
opts.Add('lua','Build Lua Support: (yes/no)','no')
opts.Add('rfd','Remote Filesystem Driver: (yes/no)','no')
opts.Add('gdscript','Build GDSCript support: (yes/no)','yes')
opts.Add('vorbis','Build Ogg Vorbis Support: (yes/no)','yes')
opts.Add('minizip','Build Minizip Archive Support: (yes/no)','yes')
opts.Add('opengl', 'Build OpenGL Support: (yes/no)', 'yes')
opts.Add('game', 'Game (custom) Code Directory', "")
opts.Add('squish','Squish BC Texture Compression (yes/no)','yes')
opts.Add('theora','Theora Video (yes/no)','yes')
opts.Add('freetype','Freetype support in editor','yes')
opts.Add('speex','Speex Audio (yes/no)','yes')
opts.Add('xml','XML Save/Load support (yes/no)','yes')
opts.Add('png','PNG Image loader support (yes/no)','yes')
opts.Add('jpg','JPG Image loader support (yes/no)','yes')
opts.Add('webp','WEBP Image loader support (yes/no)','yes')
opts.Add('dds','DDS Texture loader support (yes/no)','yes')
opts.Add('pvr','PVR (PowerVR) Texture loader support (yes/no)','yes')
opts.Add('builtin_zlib','Use built-in zlib (yes/no)','yes')
opts.Add('musepack','Musepack Audio (yes/no)','yes')
opts.Add('default_gui_theme','Default GUI theme (yes/no)','yes')
opts.Add("CXX", "Compiler");
opts.Add("nedmalloc", "Add nedmalloc support", 'yes');
opts.Add("CCFLAGS", "Custom flags for the C++ compiler");
opts.Add("CFLAGS", "Custom flags for the C compiler");
opts.Add("LINKFLAGS", "Custom flags for the linker");
opts.Add('disable_3d', 'Disable 3D nodes for smaller executable (yes/no)', "no")
opts.Add('disable_advanced_gui', 'Disable advance 3D gui nodes and behaviors (yes/no)', "no")
opts.Add('old_scenes', 'Compatibility with old-style scenes', "yes")

# add platform specific options

for k in platform_opts.keys():
	opt_list = platform_opts[k]
	for o in opt_list:
		opts.Add(o[0],o[1],o[2])

for x in module_list:
	opts.Add('module_'+x+'_enabled', "Enable module '"+x+"'.", "yes")

opts.Update(env_base) # update environment
Help(opts.GenerateHelpText(env_base)) # generate help

# add default include paths

env_base.Append(CPPPATH=['#core','#core/math','#tools','#drivers','#'])
	
# configure ENV for platform	
env_base.detect_python=True
env_base.platform_exporters=platform_exporters

"""
sys.path.append("./platform/"+env_base["platform"])
import detect
detect.configure(env_base)
sys.path.remove("./platform/"+env_base["platform"])
sys.modules.pop('detect')
"""

if (env_base['target']=='debug'):
	env_base.Append(CPPFLAGS=['-DDEBUG_MEMORY_ALLOC']);
	env_base.Append(CPPFLAGS=['-DSCI_NAMESPACE'])

env_base.platforms = {}

for p in platform_list:

	if env_base['platform'] != "" and env_base['platform'] != p:
		continue
	sys.path.append("./platform/"+p)
	import detect
	if "create" in dir(detect):
		env = detect.create(env_base)
	else:
		env = env_base.Clone()

	CCFLAGS = env.get('CCFLAGS', '')
	env['CCFLAGS'] = ''

	env.Append(CCFLAGS=string.split(str(CCFLAGS)))

	CFLAGS = env.get('CFLAGS', '')
	env['CFLAGS'] = ''

	env.Append(CFLAGS=string.split(str(CFLAGS)))

	LINKFLAGS = env.get('LINKFLAGS', '')
	env['LINKFLAGS'] = ''

	env.Append(LINKFLAGS=string.split(str(LINKFLAGS)))

	detect.configure(env)
	env['platform'] = p
        if not env.has_key('platform_libsuffix'):
                env['platform_libsuffix'] = env['LIBSUFFIX']
	sys.path.remove("./platform/"+p)
	sys.modules.pop('detect')

	flag_list = platform_flags[p]
	for f in flag_list:
                env[f[0]] = f[1]

	env.module_list=[]

	for x in module_list:
		if env['module_'+x+'_enabled'] != "yes":
			continue
		tmppath="./modules/"+x
		sys.path.append(tmppath)
		env.current_module=x
		import config
		if (config.can_build(p)):
			config.configure(env)
			env.module_list.append(x)
		sys.path.remove(tmppath)
		sys.modules.pop('config')


	if (env['musepack']=='yes'):
		env.Append(CPPFLAGS=['-DMUSEPACK_ENABLED']);

	if (env["old_scenes"]=='yes'):
		env.Append(CPPFLAGS=['-DOLD_SCENE_FORMAT_ENABLED'])
	if (env["rfd"]=='yes'):
		env.Append(CPPFLAGS=['-DRFD_ENABLED'])
	if (env["builtin_zlib"]=='yes'):
		env.Append(CPPPATH=['#drivers/builtin_zlib/zlib'])

	# to test 64 bits compiltion
	# env.Append(CPPFLAGS=['-m64'])

	if (env_base['squish']=='yes'):
		env.Append(CPPFLAGS=['-DSQUISH_ENABLED']);

	if (env['vorbis']=='yes'):
		env.Append(CPPFLAGS=['-DVORBIS_ENABLED']);

	if (env['theora']=='yes'):
		env.Append(CPPFLAGS=['-DTHEORA_ENABLED']);

	if (env['png']=='yes'):
		env.Append(CPPFLAGS=['-DPNG_ENABLED']);
	if (env['dds']=='yes'):
		env.Append(CPPFLAGS=['-DDDS_ENABLED']);
	if (env['pvr']=='yes'):
		env.Append(CPPFLAGS=['-DPVR_ENABLED']);
	if (env['jpg']=='yes'):
		env.Append(CPPFLAGS=['-DJPG_ENABLED']);
	if (env['webp']=='yes'):
		env.Append(CPPFLAGS=['-DWEBP_ENABLED']);

	if (env['speex']=='yes'):
		env.Append(CPPFLAGS=['-DSPEEX_ENABLED']);

	if (env['tools']=='yes'):
		env.Append(CPPFLAGS=['-DTOOLS_ENABLED'])
	if (env['disable_3d']=='yes'):
		env.Append(CPPFLAGS=['-D_3D_DISABLED'])
	if (env['gdscript']=='yes'):
		env.Append(CPPFLAGS=['-DGDSCRIPT_ENABLED'])
	if (env['disable_advanced_gui']=='yes'):
		env.Append(CPPFLAGS=['-DADVANCED_GUI_DISABLED'])

	if (env['minizip'] == 'yes'):
		env.Append(CPPFLAGS=['-DMINIZIP_ENABLED'])

	if (env['xml']=='yes'):
		env.Append(CPPFLAGS=['-DXML_ENABLED'])

	if (env['default_gui_theme']=='no'):
		env.Append(CPPFLAGS=['-DDEFAULT_THEME_DISABLED'])

	if (env["python"]=='yes'):
		detected=False;
		if (env.detect_python):
			print("Python 3.0 Prefix:");
			pycfg_exec="python3-config"
			errorval=os.system(pycfg_exec+" --prefix")
			prefix=""
			if (not errorval):
				#gah, why can't it get both at the same time like pkg-config, sdl-config, etc?
				env.ParseConfig(pycfg_exec+" --cflags")
				env.ParseConfig(pycfg_exec+" --libs")
				detected=True

		if (detected):
			env.Append(CPPFLAGS=['-DPYTHON_ENABLED'])
			#remove annoying warnings
			if ('-Wstrict-prototypes' in  env["CCFLAGS"]):
				env["CCFLAGS"].remove('-Wstrict-prototypes');
			if ('-fwrapv' in env["CCFLAGS"]):
				env["CCFLAGS"].remove('-fwrapv');
		else:
			print("Python 3.0 not detected ("+pycfg_exec+") support disabled.");

	#if env['nedmalloc'] == 'yes':
	#	env.Append(CPPFLAGS = ['-DNEDMALLOC_ENABLED'])

	Export('env')

	#build subdirs, the build order is dependent on link order.

	SConscript("core/SCsub")
	SConscript("servers/SCsub")
	SConscript("scene/SCsub")
	SConscript("tools/SCsub")
	SConscript("drivers/SCsub")
	SConscript("bin/SCsub")

	if env['game']:
		SConscript(env['game']+'/SCsub')

	SConscript("modules/SCsub")
	SConscript("main/SCsub")

	SConscript("platform/"+p+"/SCsub"); # build selected platform

