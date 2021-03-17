/* eslint-disable strict */

'use strict';

const fs = require('fs');

class JSDoclet {
	constructor(doc) {
		this.doc = doc;
		this.description = doc['description'] || '';
		this.name = doc['name'] || 'unknown';
		this.longname = doc['longname'] || '';
		this.types = [];
		if (doc['type'] && doc['type']['names']) {
			this.types = doc['type']['names'].slice();
		}
		this.type = this.types.length > 0 ? this.types.join('\\|') : '*';
		this.variable = doc['variable'] || false;
		this.kind = doc['kind'] || '';
		this.memberof = doc['memberof'] || null;
		this.scope = doc['scope'] || '';
		this.members = [];
		this.optional = doc['optional'] || false;
		this.defaultvalue = doc['defaultvalue'];
		this.summary = doc['summary'] || null;
		this.classdesc = doc['classdesc'] || null;

		// Parameters (functions)
		this.params = [];
		this.returns = doc['returns'] ? doc['returns'][0]['type']['names'][0] : 'void';
		this.returns_desc = doc['returns'] ? doc['returns'][0]['description'] : null;

		this.params = (doc['params'] || []).slice().map((p) => new JSDoclet(p));

		// Custom tags
		this.tags = doc['tags'] || [];
		this.header = this.tags.filter((t) => t['title'] === 'header').map((t) => t['text']).pop() || null;
	}

	add_member(obj) {
		this.members.push(obj);
	}

	is_static() {
		return this.scope === 'static';
	}

	is_instance() {
		return this.scope === 'instance';
	}

	is_object() {
		return this.kind === 'Object' || (this.kind === 'typedef' && this.type === 'Object');
	}

	is_class() {
		return this.kind === 'class';
	}

	is_function() {
		return this.kind === 'function' || (this.kind === 'typedef' && this.type === 'function');
	}

	is_module() {
		return this.kind === 'module';
	}
}

function format_table(f, data, depth = 0) {
	if (!data.length) {
		return;
	}

	const column_sizes = new Array(data[0].length).fill(0);

	data.forEach((row) => {
		row.forEach((e, idx) => {
			column_sizes[idx] = Math.max(e.length, column_sizes[idx]);
		});
	});

	const indent = ' '.repeat(depth);
	let sep = indent;
	column_sizes.forEach((size) => {
		sep += '+';
		sep += '-'.repeat(size + 2);
	});
	sep += '+\n';
	f.write(sep);

	data.forEach((row) => {
		let row_text = `${indent}|`;
		row.forEach((entry, idx) => {
			row_text += ` ${entry.padEnd(column_sizes[idx])} |`;
		});
		row_text += '\n';
		f.write(row_text);
		f.write(sep);
	});

	f.write('\n');
}

function make_header(header, sep) {
	return `${header}\n${sep.repeat(header.length)}\n\n`;
}

function indent_multiline(text, depth) {
	const indent = ' '.repeat(depth);
	return text.split('\n').map((l) => (l === '' ? l : indent + l)).join('\n');
}

function make_rst_signature(obj, types = false, style = false) {
	let out = '';
	const fmt = style ? '*' : '';
	obj.params.forEach((arg, idx) => {
		if (idx > 0) {
			if (arg.optional) {
				out += ` ${fmt}[`;
			}
			out += ', ';
		} else {
			out += ' ';
			if (arg.optional) {
				out += `${fmt}[ `;
			}
		}
		if (types) {
			out += `${arg.type} `;
		}
		const variable = arg.variable ? '...' : '';
		const defval = arg.defaultvalue !== undefined ? `=${arg.defaultvalue}` : '';
		out += `${variable}${arg.name}${defval}`;
		if (arg.optional) {
			out += ` ]${fmt}`;
		}
	});
	out += ' ';
	return out;
}

function make_rst_param(f, obj, depth = 0) {
	const indent = ' '.repeat(depth * 3);
	f.write(indent);
	f.write(`:param ${obj.type} ${obj.name}:\n`);
	f.write(indent_multiline(obj.description, (depth + 1) * 3));
	f.write('\n\n');
}

function make_rst_attribute(f, obj, depth = 0, brief = false) {
	const indent = ' '.repeat(depth * 3);
	f.write(indent);
	f.write(`.. js:attribute:: ${obj.name}\n\n`);

	if (brief) {
		if (obj.summary) {
			f.write(indent_multiline(obj.summary, (depth + 1) * 3));
		}
		f.write('\n\n');
		return;
	}

	f.write(indent_multiline(obj.description, (depth + 1) * 3));
	f.write('\n\n');

	f.write(indent);
	f.write(`   :type: ${obj.type}\n\n`);

	if (obj.defaultvalue !== undefined) {
		let defval = obj.defaultvalue;
		if (defval === '') {
			defval = '""';
		}
		f.write(indent);
		f.write(`   :value: \`\`${defval}\`\`\n\n`);
	}
}

function make_rst_function(f, obj, depth = 0) {
	let prefix = '';
	if (obj.is_instance()) {
		prefix = 'prototype.';
	}

	const indent = ' '.repeat(depth * 3);
	const sig = make_rst_signature(obj);
	f.write(indent);
	f.write(`.. js:function:: ${prefix}${obj.name}(${sig})\n`);
	f.write('\n');

	f.write(indent_multiline(obj.description, (depth + 1) * 3));
	f.write('\n\n');

	obj.params.forEach((param) => {
		make_rst_param(f, param, depth + 1);
	});

	if (obj.returns !== 'void') {
		f.write(indent);
		f.write('   :return:\n');
		f.write(indent_multiline(obj.returns_desc, (depth + 2) * 3));
		f.write('\n\n');
		f.write(indent);
		f.write(`   :rtype: ${obj.returns}\n\n`);
	}
}

function make_rst_object(f, obj) {
	let brief = false;
	// Our custom header flag.
	if (obj.header !== null) {
		f.write(make_header(obj.header, '-'));
		f.write(`${obj.description}\n\n`);
		brief = true;
	}

	// Format members table and descriptions
	const data = [['type', 'name']].concat(obj.members.map((m) => [m.type, `:js:attr:\`${m.name}\``]));

	f.write(make_header('Properties', '^'));
	format_table(f, data, 0);

	make_rst_attribute(f, obj, 0, brief);

	if (!obj.members.length) {
		return;
	}

	f.write('   **Property Descriptions**\n\n');

	// Properties first
	obj.members.filter((m) => !m.is_function()).forEach((m) => {
		make_rst_attribute(f, m, 1);
	});

	// Callbacks last
	obj.members.filter((m) => m.is_function()).forEach((m) => {
		make_rst_function(f, m, 1);
	});
}

function make_rst_class(f, obj) {
	const header = obj.header ? obj.header : obj.name;
	f.write(make_header(header, '-'));

	if (obj.classdesc) {
		f.write(`${obj.classdesc}\n\n`);
	}

	const funcs = obj.members.filter((m) => m.is_function());
	function make_data(m) {
		const base = m.is_static() ? obj.name : `${obj.name}.prototype`;
		const params = make_rst_signature(m, true, true);
		const sig = `:js:attr:\`${m.name} <${base}.${m.name}>\` **(**${params}**)**`;
		return [m.returns, sig];
	}
	const sfuncs = funcs.filter((m) => m.is_static());
	const ifuncs = funcs.filter((m) => !m.is_static());

	f.write(make_header('Static Methods', '^'));
	format_table(f, sfuncs.map((m) => make_data(m)));

	f.write(make_header('Instance Methods', '^'));
	format_table(f, ifuncs.map((m) => make_data(m)));

	const sig = make_rst_signature(obj);
	f.write(`.. js:class:: ${obj.name}(${sig})\n\n`);
	f.write(indent_multiline(obj.description, 3));
	f.write('\n\n');

	obj.params.forEach((p) => {
		make_rst_param(f, p, 1);
	});

	f.write('   **Static Methods**\n\n');
	sfuncs.forEach((m) => {
		make_rst_function(f, m, 1);
	});

	f.write('   **Instance Methods**\n\n');
	ifuncs.forEach((m) => {
		make_rst_function(f, m, 1);
	});
}

function make_rst_module(f, obj) {
	const header = obj.header !== null ? obj.header : obj.name;
	f.write(make_header(header, '='));
	f.write(obj.description);
	f.write('\n\n');
}

function write_base_object(f, obj) {
	if (obj.is_object()) {
		make_rst_object(f, obj);
	} else if (obj.is_function()) {
		make_rst_function(f, obj);
	} else if (obj.is_class()) {
		make_rst_class(f, obj);
	} else if (obj.is_module()) {
		make_rst_module(f, obj);
	}
}

function generate(f, docs) {
	const globs = [];
	const SYMBOLS = {};
	docs.filter((d) => !d.ignore && d.kind !== 'package').forEach((d) => {
		SYMBOLS[d.name] = d;
		if (d.memberof) {
			const up = SYMBOLS[d.memberof];
			if (up === undefined) {
				console.log(d); // eslint-disable-line no-console
				console.log(`Undefined symbol! ${d.memberof}`); // eslint-disable-line no-console
				throw new Error('Undefined symbol!');
			}
			SYMBOLS[d.memberof].add_member(d);
		} else {
			globs.push(d);
		}
	});

	f.write('.. _doc_html5_shell_classref:\n\n');
	globs.forEach((obj) => write_base_object(f, obj));
}

/**
 * Generate documentation output.
 *
 * @param {TAFFY} data - A TaffyDB collection representing
 *                       all the symbols documented in your code.
 * @param {object} opts - An object with options information.
 */
exports.publish = function (data, opts) {
	const docs = data().get().filter((doc) => !doc.undocumented && !doc.ignore).map((doc) => new JSDoclet(doc));
	const dest = opts.destination;
	if (dest === 'dry-run') {
		process.stdout.write('Dry run... ');
		generate({
			write: function () { /* noop */ },
		}, docs);
		process.stdout.write('Okay!\n');
		return;
	}
	if (dest !== '' && !dest.endsWith('.rst')) {
		throw new Error('Destination file must be either a ".rst" file, or an empty string (for printing to stdout)');
	}
	if (dest !== '') {
		const f = fs.createWriteStream(dest);
		generate(f, docs);
	} else {
		generate(process.stdout, docs);
	}
};
