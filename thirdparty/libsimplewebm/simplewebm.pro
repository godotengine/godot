TEMPLATE = app

CONFIG += console link_pkgconfig
CONFIG -= app_bundle qt

PKGCONFIG += vpx opus vorbis

INCLUDEPATH = . libwebm
DEPENDPATH  = . libwebm

SOURCES += example.cpp \
           WebMDemuxer.cpp \
           VPXDecoder.cpp \
           OpusVorbisDecoder.cpp

HEADERS += WebMDemuxer.hpp \
           VPXDecoder.hpp \
           OpusVorbisDecoder.hpp

SOURCES += libwebm/mkvparser/mkvparser.cc
HEADERS += libwebm/mkvparser/mkvparser.h \
           libwebm/common/webmids.h

#QMAKE_CXXFLAGS_DEBUG += -fsanitize=address -std=gnu++98
#QMAKE_LFLAGS         += -fsanitize=address
