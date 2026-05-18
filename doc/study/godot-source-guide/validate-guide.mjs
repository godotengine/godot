import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const guideDir = path.dirname(fileURLToPath(import.meta.url));
const dataPath = path.join(guideDir, "data.js");
const appPath = path.join(guideDir, "app.js");
const indexPath = path.join(guideDir, "index.html");
const conceptsPath = path.join(guideDir, "concepts.html");

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
  concepts
};`, context, { filename: dataPath });
  return context.__guideData;
}

function validateConceptDrawerNavigation() {
  const appText = readText(appPath);
  const indexHtml = readText(indexPath);
  const conceptsHtml = readText(conceptsPath);

  for (const [page, html] of [["index.html", indexHtml], ["concepts.html", conceptsHtml]]) {
    if (!html.includes('id="conceptPanelBack"')) {
      fail(`${page} concept drawer is missing a dedicated concept-history back button`);
    }
  }

  if (!/detailHistory\s*:\s*\[\]/.test(appText)) {
    fail("conceptState must include a dedicated detailHistory stack for concept drawer navigation");
  }

  if (!/linkConceptKeywords\(\s*qs\("#conceptDrawerBody"\)/.test(appText)) {
    fail("Concept drawer body must link concept keywords after rendering an article");
  }

  if (/closest\([^)]*\.concept-drawer/.test(appText)) {
    fail("Concept keyword linking must not skip the entire concept drawer body");
  }

  if (!/openConceptDetail\([^)]*source\s*:\s*"articleKeyword"/.test(appText)) {
    fail("Concept article keyword clicks must open details through an articleKeyword source");
  }

  if (!/backConceptDetail/.test(appText)) {
    fail("Concept drawer needs a backConceptDetail handler");
  }
}

function main() {
  const { concepts } = loadData();
  if (!Array.isArray(concepts) || !concepts.length) {
    fail("concepts is missing or empty in data.js");
  }
  validateConceptDrawerNavigation();

  console.log(`Guide checks OK: ${concepts.length} concepts`);
}

main();
