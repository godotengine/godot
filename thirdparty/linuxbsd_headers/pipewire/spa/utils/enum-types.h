/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_ENUM_TYPES_H
#define SPA_ENUM_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/type.h>

#define SPA_TYPE_INFO_Direction			SPA_TYPE_INFO_ENUM_BASE "Direction"
#define SPA_TYPE_INFO_DIRECTION_BASE		SPA_TYPE_INFO_Direction ":"

static const struct spa_type_info spa_type_direction[] = {
	{ SPA_DIRECTION_INPUT, SPA_TYPE_Int, SPA_TYPE_INFO_DIRECTION_BASE "Input", NULL  },
	{ SPA_DIRECTION_OUTPUT, SPA_TYPE_Int, SPA_TYPE_INFO_DIRECTION_BASE "Output", NULL  },
	{ 0, 0, NULL, NULL }
};

#include <spa/pod/pod.h>

#define SPA_TYPE_INFO_Choice			SPA_TYPE_INFO_ENUM_BASE "Choice"
#define SPA_TYPE_INFO_CHOICE_BASE		SPA_TYPE_INFO_Choice ":"

static const struct spa_type_info spa_type_choice[] = {
	{ SPA_CHOICE_None, SPA_TYPE_Int, SPA_TYPE_INFO_CHOICE_BASE "None", NULL  },
	{ SPA_CHOICE_Range, SPA_TYPE_Int, SPA_TYPE_INFO_CHOICE_BASE "Range", NULL  },
	{ SPA_CHOICE_Step, SPA_TYPE_Int, SPA_TYPE_INFO_CHOICE_BASE "Step", NULL  },
	{ SPA_CHOICE_Enum, SPA_TYPE_Int, SPA_TYPE_INFO_CHOICE_BASE "Enum", NULL  },
	{ SPA_CHOICE_Flags, SPA_TYPE_Int, SPA_TYPE_INFO_CHOICE_BASE "Flags", NULL  },
	{ 0, 0, NULL, NULL }
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_TYPE_INFO_H */
