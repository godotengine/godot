#ifndef CONFIG_H
#define CONFIG_H

/* This configuration file is custom written for Godot.
 * When updating the library, generate it with CMake upstream and compare
 * the contents to see if new options should be backported here.
 */

// Those are handled in our SCsub.
/* #undef HAVE_ARPA_INET_H */
/* #undef HAVE_NETINET_IN_H */
/* #undef HAVE_WINSOCK2_H */

#ifdef BIG_ENDIAN_ENABLED
#define WORDS_BIGENDIAN
#endif

#endif /* CONFIG_H */
