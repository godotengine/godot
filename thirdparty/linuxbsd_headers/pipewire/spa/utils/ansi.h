/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2021 Red Hat, Inc. */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_UTILS_ANSI_H
#define SPA_UTILS_ANSI_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup spa_ansi ANSI codes
 * ANSI color code macros
 */

/**
 * \addtogroup spa_ansi
 * \{
 */

/**
 * Ansi escape sequences. Note that the color names are approximate only and
 * the actual rendering of the color depends on the terminal.
 */

#define SPA_ANSI_RESET		"\x1B[0m"
#define SPA_ANSI_BOLD		"\x1B[1m"
#define SPA_ANSI_ITALIC		"\x1B[3m"
#define SPA_ANSI_UNDERLINE	"\x1B[4m"

#define SPA_ANSI_BLACK		"\x1B[0;30m"
#define SPA_ANSI_RED		"\x1B[0;31m"
#define SPA_ANSI_GREEN		"\x1B[0;32m"
#define SPA_ANSI_YELLOW		"\x1B[0;33m"
#define SPA_ANSI_BLUE		"\x1B[0;34m"
#define SPA_ANSI_MAGENTA	"\x1B[0;35m"
#define SPA_ANSI_CYAN		"\x1B[0;36m"
#define SPA_ANSI_WHITE		"\x1B[0;37m"
#define SPA_ANSI_BRIGHT_BLACK	"\x1B[90m"
#define SPA_ANSI_BRIGHT_RED	"\x1B[91m"
#define SPA_ANSI_BRIGHT_GREEN	"\x1B[92m"
#define SPA_ANSI_BRIGHT_YELLOW	"\x1B[93m"
#define SPA_ANSI_BRIGHT_BLUE	"\x1B[94m"
#define SPA_ANSI_BRIGHT_MAGENTA	"\x1B[95m"
#define SPA_ANSI_BRIGHT_CYAN	"\x1B[96m"
#define SPA_ANSI_BRIGHT_WHITE	"\x1B[97m"

/* Shortcut because it's a common use-case and easier than combining both */
#define SPA_ANSI_BOLD_BLACK	"\x1B[1;30m"
#define SPA_ANSI_BOLD_RED	"\x1B[1;31m"
#define SPA_ANSI_BOLD_GREEN	"\x1B[1;32m"
#define SPA_ANSI_BOLD_YELLOW	"\x1B[1;33m"
#define SPA_ANSI_BOLD_BLUE	"\x1B[1;34m"
#define SPA_ANSI_BOLD_MAGENTA	"\x1B[1;35m"
#define SPA_ANSI_BOLD_CYAN	"\x1B[1;36m"
#define SPA_ANSI_BOLD_WHITE	"\x1B[1;37m"

#define SPA_ANSI_DARK_BLACK	"\x1B[2;30m"
#define SPA_ANSI_DARK_RED	"\x1B[2;31m"
#define SPA_ANSI_DARK_GREEN	"\x1B[2;32m"
#define SPA_ANSI_DARK_YELLOW	"\x1B[2;33m"
#define SPA_ANSI_DARK_BLUE	"\x1B[2;34m"
#define SPA_ANSI_DARK_MAGENTA	"\x1B[2;35m"
#define SPA_ANSI_DARK_CYAN	"\x1B[2;36m"
#define SPA_ANSI_DARK_WHITE	"\x1B[2;37m"

/* Background colors */
#define SPA_ANSI_BG_BLACK		"\x1B[0;40m"
#define SPA_ANSI_BG_RED			"\x1B[0;41m"
#define SPA_ANSI_BG_GREEN		"\x1B[0;42m"
#define SPA_ANSI_BG_YELLOW		"\x1B[0;43m"
#define SPA_ANSI_BG_BLUE		"\x1B[0;44m"
#define SPA_ANSI_BG_MAGENTA		"\x1B[0;45m"
#define SPA_ANSI_BG_CYAN		"\x1B[0;46m"
#define SPA_ANSI_BG_WHITE		"\x1B[0;47m"
#define SPA_ANSI_BG_BRIGHT_BLACK	"\x1B[100m"
#define SPA_ANSI_BG_BRIGHT_RED		"\x1B[101m"
#define SPA_ANSI_BG_BRIGHT_GREEN	"\x1B[102m"
#define SPA_ANSI_BG_BRIGHT_YELLOW	"\x1B[103m"
#define SPA_ANSI_BG_BRIGHT_BLUE		"\x1B[104m"
#define SPA_ANSI_BG_BRIGHT_MAGENTA	"\x1B[105m"
#define SPA_ANSI_BG_BRIGHT_CYAN		"\x1B[106m"
#define SPA_ANSI_BG_BRIGHT_WHITE	"\x1B[107m"

/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_UTILS_ANSI_H */
