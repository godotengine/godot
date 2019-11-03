#
# FreeType 2 auto-fitter module definition
#


# Copyright (C) 2003-2019 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += AUTOFIT_MODULE

define AUTOFIT_MODULE
$(OPEN_DRIVER) FT_Module_Class, autofit_module_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)autofit   $(ECHO_DRIVER_DESC)automatic hinting module$(ECHO_DRIVER_DONE)
endef

# EOF
