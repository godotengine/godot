#
# FreeType 2 smooth renderer module definition
#


# Copyright 1996-2016 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += SMOOTH_RENDERER

define SMOOTH_RENDERER
$(OPEN_DRIVER) FT_Renderer_Class, ft_smooth_renderer_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)smooth    $(ECHO_DRIVER_DESC)anti-aliased bitmap renderer$(ECHO_DRIVER_DONE)
$(OPEN_DRIVER) FT_Renderer_Class, ft_smooth_lcd_renderer_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)smooth    $(ECHO_DRIVER_DESC)anti-aliased bitmap renderer for LCDs$(ECHO_DRIVER_DONE)
$(OPEN_DRIVER) FT_Renderer_Class, ft_smooth_lcdv_renderer_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)smooth    $(ECHO_DRIVER_DESC)anti-aliased bitmap renderer for vertical LCDs$(ECHO_DRIVER_DONE)
endef

# EOF
