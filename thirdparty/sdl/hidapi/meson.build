project('hidapi', meson_version: '>=0.57.0', version: files('VERSION'))

cmake = import('cmake')

hidapi_build_options = cmake.subproject_options()
hidapi_build_options.set_install(true)

hidapi_build = cmake.subproject('hidapi_build_cmake', options: hidapi_build_options)

if (hidapi_build.target_list().contains('hidapi_winapi'))
	hidapi_winapi_dep = hidapi_build.dependency('hidapi_winapi')
	hidapi_dep = hidapi_winapi_dep
elif (hidapi_build.target_list().contains('hidapi_darwin'))
	hidapi_darwin_dep = hidapi_build.dependency('hidapi_darwin')
	hidapi_dep = hidapi_darwin_dep
elif (hidapi_build.target_list().contains('hidapi_hidraw'))
	hidapi_hidraw_dep = hidapi_build.dependency('hidapi_hidraw')
	hidapi_dep = hidapi_hidraw_dep
elif (hidapi_build.target_list().contains('hidapi_libusb'))
	hidapi_libusb_dep = hidapi_build.dependency('hidapi_libusb')
	hidapi_dep = hidapi_libusb_dep
endif
