#
# FreeType 2 renderer module definition
#


# Copyright 1996-2018 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += RASTER_MODULE

define RASTER_MODULE
$(OPEN_DRIVER) FT_Renderer_Class, ft_raster1_renderer_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)raster    $(ECHO_DRIVER_DESC)monochrome bitmap renderer$(ECHO_DRIVER_DONE)
endef

# EOF
