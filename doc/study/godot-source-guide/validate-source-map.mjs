import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const guideDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(guideDir, "../../..");
const dataPath = path.join(guideDir, "data.js");
const indexPath = path.join(guideDir, "index.html");
const conceptsPath = path.join(guideDir, "concepts.html");
const architectureMapPath = path.join(guideDir, "architecture-map.html");

function readText(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function fail(message) {
  throw new Error(message);
}

function loadData() {
  const context = {};
  vm.createContext(context);
  vm.runInContext(`${readText(dataPath)}
this.__guideData = {
  concepts,
  sourceMapGraph: typeof sourceMapGraph === "undefined" ? undefined : sourceMapGraph
};`, context, { filename: dataPath });
  return context.__guideData;
}

function collectHtmlIds(...files) {
  const ids = new Set();
  for (const file of files) {
    if (!fs.existsSync(file)) continue;
    const text = readText(file);
    for (const match of text.matchAll(/\sid=["']([^"']+)["']/g)) {
      ids.add(match[1]);
    }
  }
  return ids;
}

function validateSourceAnchor(anchor) {
  const [filePart, linePart] = anchor.split(":");
  const filePath = path.join(repoRoot, filePart);
  if (!fs.existsSync(filePath)) {
    fail(`Source anchor file does not exist: ${anchor}`);
  }
  if (linePart && /^\d+$/.test(linePart)) {
    const line = Number(linePart);
    const lineCount = readText(filePath).split(/\r?\n/).length;
    if (line < 1 || line > lineCount) {
      fail(`Source anchor line is out of range: ${anchor} (${lineCount} lines)`);
    }
  }
}

function validateArticleHref(href, htmlIds) {
  if (!href) {
    fail("Node is missing articleHref");
  }
  const [file, hash] = href.split("#");
  if (!file || !hash) {
    fail(`articleHref must include a local file and hash: ${href}`);
  }
  if (!["index.html", "concepts.html", "architecture-map.html"].includes(file)) {
    fail(`articleHref points outside the guide: ${href}`);
  }
  if (!htmlIds.has(hash) && !hash.startsWith("concept-")) {
    fail(`articleHref hash does not exist: ${href}`);
  }
}

function validateResourceConceptCoverage(concepts, sourceMapGraph) {
  const resourceConcept = concepts.find((concept) => concept.id === "resource");
  if (!resourceConcept) {
    fail("Concept library is missing the Resource concept");
  }

  const resourceNames = new Set([
    resourceConcept.title,
    ...(resourceConcept.aliases || []),
  ].map((value) => String(value || "").toLocaleLowerCase()));
  if (!resourceNames.has("resource")) {
    fail("Resource concept must own the Resource keyword");
  }

  const refCountedConcept = concepts.find((concept) => concept.id === "refcounted");
  if (refCountedConcept?.aliases?.some((alias) => String(alias).toLocaleLowerCase() === "resource")) {
    fail("RefCounted concept must not claim Resource as an alias");
  }

  if (!(sourceMapGraph.nodes || []).some((node) => node.id === "resource" && node.conceptId === "resource")) {
    fail("Source map is missing a Resource node linked to the Resource concept");
  }
}

function main() {
  const { concepts, sourceMapGraph } = loadData();
  if (!sourceMapGraph) {
    fail("sourceMapGraph is not defined in data.js");
  }
  const relationTypes = new Set(Object.keys(sourceMapGraph.relationTypes || {}));
  if (!relationTypes.size) fail("sourceMapGraph.relationTypes is empty");

  const conceptIds = new Set(concepts.map((concept) => concept.id));
  const groupIds = new Set((sourceMapGraph.groups || []).map((group) => group.id));
  const nodeIds = new Set();
  const htmlIds = collectHtmlIds(indexPath, conceptsPath, architectureMapPath);

  validateResourceConceptCoverage(concepts, sourceMapGraph);

  for (const group of sourceMapGraph.groups || []) {
    if (!group.id || !group.title) fail("Every group needs id and title");
    for (const key of ["x", "y", "width", "height"]) {
      if (!Number.isFinite(group[key])) fail(`Group ${group.id} has invalid ${key}`);
    }
  }

  for (const node of sourceMapGraph.nodes || []) {
    if (!node.id || !node.title) fail("Every node needs id and title");
    if (nodeIds.has(node.id)) fail(`Duplicate node id: ${node.id}`);
    nodeIds.add(node.id);
    if (!groupIds.has(node.group)) fail(`Node ${node.id} references missing group: ${node.group}`);
    if (node.conceptId && !conceptIds.has(node.conceptId)) {
      fail(`Node ${node.id} references missing conceptId: ${node.conceptId}`);
    }
    validateArticleHref(node.articleHref, htmlIds);
    for (const key of ["x", "y"]) {
      if (!Number.isFinite(node[key])) fail(`Node ${node.id} has invalid ${key}`);
    }
    if (!Array.isArray(node.sourceAnchors) || !node.sourceAnchors.length) {
      fail(`Node ${node.id} needs at least one source anchor`);
    }
    node.sourceAnchors.forEach(validateSourceAnchor);
  }

  if (nodeIds.size < 35 || nodeIds.size > 50) {
    fail(`Expected 35-50 source map nodes, got ${nodeIds.size}`);
  }

  for (const edge of sourceMapGraph.edges || []) {
    if (!nodeIds.has(edge.from)) fail(`Edge references missing from node: ${edge.from}`);
    if (!nodeIds.has(edge.to)) fail(`Edge references missing to node: ${edge.to}`);
    if (!relationTypes.has(edge.type)) fail(`Edge ${edge.from}->${edge.to} has invalid type: ${edge.type}`);
    if (!edge.label || !edge.explanation) {
      fail(`Edge ${edge.from}->${edge.to} needs label and explanation`);
    }
  }

  console.log(`sourceMapGraph OK: ${nodeIds.size} nodes, ${sourceMapGraph.edges.length} edges, ${relationTypes.size} relation types`);
}

main();
