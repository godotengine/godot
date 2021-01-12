# OS Drivers

subdir('windows')
subdir('unix')


# Sound Drivers
subdir('alsa')
subdir('coreaudio')
subdir('pulseaudio')

if PLATFORM == 'windows'
    subdir('wasapi')
endif

if get_option('xaudio2')
    subdir('xaudio2')
endif

# MIDI Drivers
subdir('alsamidi')
subdir('coremidi')
subdir('winmidi') # NOE: should this be PLATFORM guarded?

# Graphics Drivers
if PLATFORM != 'server' and PLATFORM != 'javascript'
    subdir('vulkan')
else
    subdir('dummy')
endif

# Core dependencies
subdir('png')

SOURCES += files([
    'register_driver_types.cpp'
])