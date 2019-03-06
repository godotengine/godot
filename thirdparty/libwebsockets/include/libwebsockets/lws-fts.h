/*
 * libwebsockets - fulltext search
 *
 * Copyright (C) 2010-2018 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 *
 * included from libwebsockets.h
 */

/** \defgroup search Search
 *
 * ##Full-text search
 *
 * Lws provides superfast indexing and fulltext searching from index files on
 * storage.
 */
///@{

struct lws_fts;
struct lws_fts_file;

/*
 * Queries produce their results in an lwsac, using these public API types.
 * The first thing in the lwsac is always a struct lws_fts_result (see below)
 * containing heads for linked-lists of the other result types.
 */

/* one filepath's results */

struct lws_fts_result_filepath {
	struct lws_fts_result_filepath *next;
	int matches;	/* logical number of matches */
	int matches_length;	/* bytes in length table (may be zero) */
	int lines_in_file;
	int filepath_length;

	/* - uint32_t line table follows (first for alignment) */
	/* - filepath (of filepath_length) follows */
};

/* autocomplete result */

struct lws_fts_result_autocomplete {
	struct lws_fts_result_autocomplete *next;
	int instances;
	int agg_instances;
	int ac_length;
	char elided; /* children skipped in interest of antecedent children */
	char has_children;

	/* - autocomplete suggestion (of length ac_length) follows */
};

/*
 * The results lwsac always starts with this.  If no results and / or no
 * autocomplete the members may be NULL.  This implies the symbol nor any
 * suffix on it exists in the trie file.
 */
struct lws_fts_result {
	struct lws_fts_result_filepath *filepath_head;
	struct lws_fts_result_autocomplete *autocomplete_head;
	int duration_ms;
	int effective_flags; /* the search flags that were used */
};

/*
 * index creation functions
 */

/**
 * lws_fts_create() - Create a new index file
 *
 * \param fd: The fd opened for write
 *
 * Inits a new index file, returning a struct lws_fts to represent it
 */
LWS_VISIBLE LWS_EXTERN struct lws_fts *
lws_fts_create(int fd);

/**
 * lws_fts_destroy() - Finalize a new index file / destroy the trie lwsac
 *
 * \param trie: The previously opened index being finalized
 *
 * Finalizes an index file that was being created, and frees the memory involved
 * *trie is set to NULL afterwards.
 */
LWS_VISIBLE LWS_EXTERN void
lws_fts_destroy(struct lws_fts **trie);

/**
 * lws_fts_file_index() - Create a new entry in the trie file for an input path
 *
 * \param t: The previously opened index being written
 * \param filepath: The filepath (which may be virtual) associated with this file
 * \param filepath_len: The number of chars in the filepath
 * \param priority: not used yet
 *
 * Returns an ordinal that represents this new filepath in the index file.
 */
LWS_VISIBLE LWS_EXTERN int
lws_fts_file_index(struct lws_fts *t, const char *filepath, int filepath_len,
		   int priority);

/**
 * lws_fts_fill() - Process all or a bufferload of input file
 *
 * \param t: The previously opened index being written
 * \param file_index: The ordinal representing this input filepath
 * \param buf: A bufferload of data from the input file
 * \param len: The number of bytes in buf
 *
 * Indexes a buffer of data from the input file.
 */
LWS_VISIBLE LWS_EXTERN int
lws_fts_fill(struct lws_fts *t, uint32_t file_index, const char *buf,
	     size_t len);

/**
 * lws_fts_serialize() - Store the in-memory trie into the index file
 *
 * \param t: The previously opened index being written
 *
 * The trie is held in memory where it can be added to... after all the input
 * filepaths and data have been processed, this is called to serialize /
 * write the trie data into the index file.
 */
LWS_VISIBLE LWS_EXTERN int
lws_fts_serialize(struct lws_fts *t);

/*
 * index search functions
 */

/**
 * lws_fts_open() - Open an existing index file to search it
 *
 * \param filepath: The filepath to the index file to open
 *
 * Opening the index file returns an opaque struct lws_fts_file * that is
 * used to perform other operations on it, or NULL if it can't be opened.
 */
LWS_VISIBLE LWS_EXTERN struct lws_fts_file *
lws_fts_open(const char *filepath);

#define LWSFTS_F_QUERY_AUTOCOMPLETE	(1 << 0)
#define LWSFTS_F_QUERY_FILES		(1 << 1)
#define LWSFTS_F_QUERY_FILE_LINES	(1 << 2)
#define LWSFTS_F_QUERY_QUOTE_LINE	(1 << 3)

struct lws_fts_search_params {
	/* the actual search term */
	const char *needle;
	 /* if non-NULL, FILE results for this filepath only */
	const char *only_filepath;
	/* will be set to the results lwsac */
	struct lwsac *results_head;
	/* combination of LWSFTS_F_QUERY_* flags */
	int flags;
	/* maximum number of autocomplete suggestions to return */
	int max_autocomplete;
	/* maximum number of filepaths to return */
	int max_files;
	/* maximum number of line number results to return per filepath */
	int max_lines;
};

/**
 * lws_fts_search() - Perform a search operation on an index
 *
 * \param jtf: The index file struct returned by lws_fts_open
 * \param ftsp: The struct lws_fts_search_params filled in by the caller
 *
 * The caller should memset the ftsp struct to 0 to ensure members that may be
 * introduced in later versions contain known values, then set the related
 * members to describe the kind of search action required.
 *
 * ftsp->results_head is the results lwsac, or NULL.  It should be freed with
 * lwsac_free() when the results are finished with.
 *
 * Returns a pointer into the results lwsac that is a struct lws_fts_result
 * containing the head pointers into linked-lists of results for autocomplete
 * and filepath data, along with some sundry information.  This does not need
 * to be freed since freeing the lwsac will also remove this and everything it
 * points to.
 */
LWS_VISIBLE LWS_EXTERN struct lws_fts_result *
lws_fts_search(struct lws_fts_file *jtf, struct lws_fts_search_params *ftsp);

/**
 * lws_fts_close() - Close a previously-opened index file
 *
 * \param jtf: The pointer returned from the open
 *
 * Closes the file handle on the index and frees any allocations
 */
LWS_VISIBLE LWS_EXTERN void
lws_fts_close(struct lws_fts_file *jtf);

///@}
