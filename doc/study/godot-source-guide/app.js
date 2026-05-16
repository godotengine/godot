// Runtime behavior for the interactive source guide.
const qs = (sel, root = document) => root.querySelector(sel);
const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const conceptState = {
  activeId: null,
  aliasEntries: [],
  favorites: new Set(),
  filter: "all",
  mode: "library",
  query: ""
};
const conceptFavoritesKey = "godot-source-guide-concept-favorites";
const beginnerGuidesKey = "godot-source-guide-beginner-guides";

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;"
  })[char]);
}

function renderDirs() {
  const buttons = qs("#dirButtons");
  buttons.innerHTML = "";
  dirs.forEach((dir, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.dataset.dir = dir.name;
    btn.setAttribute("aria-pressed", index === 0 ? "true" : "false");
    btn.innerHTML = `<span class="source">${dir.name}</span><span>${dir.role.split("：")[0]}</span><span>${dir.files.split(" ")[0]}</span>`;
    btn.addEventListener("click", () => selectDir(dir.name));
    buttons.appendChild(btn);
  });
  selectDir("core");
}

function selectDir(name) {
  const dir = dirs.find((item) => item.name === name) || dirs[0];
  qsa("#dirButtons button").forEach((btn) => btn.setAttribute("aria-pressed", String(btn.dataset.dir === dir.name)));
  qs("#dirTitle").textContent = dir.name;
  qs("#dirStats").textContent = dir.files;
  qs("#dirDetail").innerHTML = `
    <p><strong>职责：</strong>${dir.role}</p>
    <h3>边界判断</h3>
    <p>${dir.boundary}</p>
    <h3>源码锚点</h3>
    <p>${dir.anchors.map((a) => `<span class="source">${a}</span>`).join(" ")}</p>
    <h3>先读入口</h3>
    <p>${dir.entry}</p>
    <h3>典型追踪路径</h3>
    <ol class="content-list">${dir.trail.map((q) => `<li>${q}</li>`).join("")}</ol>
    <h3>适合回答的问题</h3>
    <ul class="content-list">${dir.questions.map((q) => `<li>${q}</li>`).join("")}</ul>
    <h3>常见误区</h3>
    <ul class="content-list">${dir.pitfalls.map((q) => `<li>${q}</li>`).join("")}</ul>
    <h3>建议阅读方式</h3>
    <p>${dir.read}</p>
  `;
  linkConceptKeywords(qs("#dirDetail"), 4);
}

function renderStartup() {
  const list = qs("#startupSteps");
  list.innerHTML = "";
  startup.forEach((step, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = step.title;
    btn.setAttribute("aria-pressed", index === 0 ? "true" : "false");
    btn.addEventListener("click", () => selectStartup(index));
    list.appendChild(btn);
  });
  selectStartup(0);
}

function selectStartup(index) {
  const step = startup[index];
  qsa("#startupSteps button").forEach((btn, i) => btn.setAttribute("aria-pressed", String(i === index)));
  qs("#startupDetail").innerHTML = `
    <h3>${step.title}</h3>
    <p><span class="source">${step.source}</span></p>
    <p>${step.body}</p>
    <ul>${step.points.map((p) => `<li>${p}</li>`).join("")}</ul>
  `;
  linkConceptKeywords(qs("#startupDetail"), 4);
}

function renderPaths() {
  const select = qs("#pathSelect");
  select.innerHTML = paths.map((p, index) => `<option value="${index}">${p.key}</option>`).join("");
  select.addEventListener("change", () => selectPath(Number(select.value)));
  selectPath(0);
}

function selectPath(index) {
  const p = paths[index];
  qs("#pathStack").innerHTML = `<ol>${p.stack.map((s) => `<li>${s}</li>`).join("")}</ol>`;
  qs("#pathTitle").textContent = p.key;
  qs("#pathSource").textContent = p.source;
  qs("#pathExplain").innerHTML = `<p>${p.explain}</p>`;
  linkConceptKeywords(qs("#pathStack"), 4);
  linkConceptKeywords(qs("#pathExplain"), 4);
}

function renderModules() {
  const categories = [...new Set(modules.map((m) => m[1]))].sort((a, b) => a.localeCompare(b, "zh-CN"));
  const categorySelect = qs("#moduleCategory");
  categorySelect.innerHTML = `<option value="all">全部类别</option>${categories.map((c) => `<option value="${c}">${c}</option>`).join("")}`;
  qs("#moduleSearch").addEventListener("input", filterModules);
  categorySelect.addEventListener("change", filterModules);
  filterModules();
}

function filterModules() {
  const term = qs("#moduleSearch").value.trim().toLowerCase();
  const category = qs("#moduleCategory").value;
  const rows = modules.filter((m) => {
    const inCategory = category === "all" || m[1] === category;
    const text = m.join(" ").toLowerCase();
    return inCategory && (!term || text.includes(term));
  });
  qs("#moduleTable").innerHTML = `
    <div class="module-row header"><div>模块</div><div>类别</div><div>用途</div><div>阅读入口 / 相关系统</div></div>
    ${rows.map((m) => `
      <div class="module-row">
        <div><span class="source">modules/${m[0]}</span></div>
        <div><span class="tag">${m[1]}</span></div>
        <div>${m[2]}</div>
        <div>
          <p><strong>相关：</strong>${m[3]}</p>
          <p><strong>入口：</strong><span class="source">modules/${m[0]}/SCsub</span>、<span class="source">modules/${m[0]}/config.py</span>${modulesWithoutRegister.has(m[0]) ? "。此模块主要作为第三方依赖封装，不一定有 register_types.cpp。" : `、<span class="source">modules/${m[0]}/register_types.*</span>`}</p>
          <p><strong>读法：</strong>${moduleAdvice[m[1]] || "先看 register_types 和 SCsub，再看资源、Server 或编辑器调用者。"}</p>
        </div>
      </div>
    `).join("")}
  `;
  linkConceptKeywords(qs("#moduleTable"), 4);
}

function normalizeGuideHeading(heading) {
  return heading.textContent.replace(/\s+/g, " ").trim();
}

function renderBeginnerGuideContent(content) {
  const paragraphs = Array.isArray(content) ? content : [content];
  return paragraphs.map((paragraph) => `<p>${renderInlineText(paragraph)}</p>`).join("");
}

function setBeginnerGuideVisibility(visible) {
  document.body.classList.toggle("beginner-guides-hidden", !visible);
  const toggle = qs("#beginnerGuideToggle");
  if (!toggle) return;
  toggle.setAttribute("aria-pressed", String(visible));
  toggle.textContent = visible ? "小白导读：开" : "小白导读：关";
  toggle.setAttribute("aria-label", visible ? "关闭小白版导读解释" : "打开小白版导读解释");
}

function getStoredBeginnerGuideVisibility() {
  try {
    return localStorage.getItem(beginnerGuidesKey) !== "hidden";
  } catch {
    return true;
  }
}

function saveBeginnerGuideVisibility(visible) {
  try {
    localStorage.setItem(beginnerGuidesKey, visible ? "visible" : "hidden");
  } catch {
    // The toggle is still usable for the current page even when storage is unavailable.
  }
}

function setupBeginnerGuides() {
  if (typeof beginnerGuides !== "object" || !beginnerGuides) return;
  const headings = qsa("main .section h3, main .section h4").filter((heading) => (
    !heading.closest(".concept-browser-section, #dirDetail, #startupDetail, #moduleTable")
  ));

  headings.forEach((heading) => {
    const key = normalizeGuideHeading(heading);
    const guide = beginnerGuides[key];
    if (!guide) return;
    const card = document.createElement("div");
    card.className = "beginner-guide";
    card.dataset.guideFor = key;
    card.innerHTML = `<div class="beginner-guide-label">小白版导读</div>${renderBeginnerGuideContent(guide)}`;
    heading.insertAdjacentElement("afterend", card);
  });

  setBeginnerGuideVisibility(getStoredBeginnerGuideVisibility());
  const toggle = qs("#beginnerGuideToggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const visible = document.body.classList.contains("beginner-guides-hidden");
      setBeginnerGuideVisibility(visible);
      saveBeginnerGuideVisibility(visible);
    });
  }
}

function loadConceptFavorites() {
  try {
    const parsed = JSON.parse(localStorage.getItem(conceptFavoritesKey) || "[]");
    return new Set(Array.isArray(parsed) ? parsed : []);
  } catch {
    return new Set();
  }
}

function saveConceptFavorites() {
  try {
    localStorage.setItem(conceptFavoritesKey, JSON.stringify([...conceptState.favorites]));
  } catch {
    // Favorites are a convenience layer; the guide should still work if storage is unavailable.
  }
}

function getConcept(id) {
  return concepts.find((concept) => concept.id === id);
}

function getConceptLibraryItems() {
  return concepts;
}

function getFavoriteConceptItems() {
  return concepts.filter((concept) => conceptState.favorites.has(concept.id));
}

function buildConceptAliasEntries() {
  const seen = new Set();
  return concepts.flatMap((concept) => [concept.title || concept.id, ...(concept.aliases || [])].map((alias) => ({
    alias,
    lookup: alias.toLocaleLowerCase(),
    conceptId: concept.id,
    needsBoundary: /[A-Za-z0-9_]/.test(alias)
  }))).filter((entry) => {
    const key = `${entry.lookup}:${entry.conceptId}`;
    if (!entry.alias || seen.has(key)) return false;
    seen.add(key);
    return true;
  }).sort((a, b) => b.alias.length - a.alias.length);
}

function isAsciiWord(char) {
  return Boolean(char && /[A-Za-z0-9_]/.test(char));
}

function isAliasBoundary(text, index, length, needsBoundary) {
  if (!needsBoundary) return true;
  return !isAsciiWord(text[index - 1]) && !isAsciiWord(text[index + length]);
}

function findAliasIndex(text, lower, entry, start) {
  let index = lower.indexOf(entry.lookup, start);
  while (index !== -1) {
    if (isAliasBoundary(text, index, entry.alias.length, entry.needsBoundary)) return index;
    index = lower.indexOf(entry.lookup, index + 1);
  }
  return -1;
}

function getTextMatches(text, counts, maxPerConcept) {
  const lower = text.toLocaleLowerCase();
  const matches = [];
  let cursor = 0;
  while (cursor < text.length) {
    let best = null;
    for (const entry of conceptState.aliasEntries) {
      if ((counts.get(entry.conceptId) || 0) >= maxPerConcept) continue;
      const index = findAliasIndex(text, lower, entry, cursor);
      if (index === -1) continue;
      if (!best || index < best.index || (index === best.index && entry.alias.length > best.entry.alias.length)) {
        best = { index, entry };
      }
    }
    if (!best) break;
    if (best.index > cursor) {
      cursor = best.index;
    }
    matches.push({
      index: best.index,
      length: best.entry.alias.length,
      conceptId: best.entry.conceptId
    });
    counts.set(best.entry.conceptId, (counts.get(best.entry.conceptId) || 0) + 1);
    cursor = best.index + best.entry.alias.length;
  }
  return matches;
}

function shouldSkipConceptNode(node) {
  const parent = node.parentElement;
  if (!parent || !node.nodeValue.trim()) return true;
  return Boolean(parent.closest("a, button, input, select, textarea, label, svg, code, pre, .source, .concept-keyword, .concept-drawer, .concept-browser-section"));
}

function linkConceptKeywords(root, maxPerConcept = 10) {
  if (!root || !conceptState.aliasEntries.length) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      return shouldSkipConceptNode(node) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
    }
  });
  const nodes = [];
  let node = walker.nextNode();
  while (node) {
    nodes.push(node);
    node = walker.nextNode();
  }
  nodes.forEach((textNode) => {
    const text = textNode.nodeValue;
    const counts = new Map();
    const matches = getTextMatches(text, counts, maxPerConcept);
    if (!matches.length) return;
    const fragment = document.createDocumentFragment();
    let cursor = 0;
    matches.forEach((match) => {
      if (match.index > cursor) {
        fragment.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = "concept-keyword";
      button.dataset.conceptId = match.conceptId;
      button.textContent = text.slice(match.index, match.index + match.length);
      fragment.appendChild(button);
      cursor = match.index + match.length;
    });
    if (cursor < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(cursor)));
    }
    textNode.parentNode.replaceChild(fragment, textNode);
  });
}

function openConceptPanel() {
  document.body.classList.add("concept-panel-open");
  qs("#conceptDrawer").setAttribute("aria-hidden", "false");
}

function closeConceptPanel() {
  document.body.classList.remove("concept-panel-open");
  qs("#conceptDrawer").setAttribute("aria-hidden", "true");
}

function focusConceptPageSearch() {
  requestAnimationFrame(() => {
    const input = qs("#conceptPageSearch");
    input.focus({ preventScroll: true });
  });
}

function showConceptBrowser(filter = "all", clearSearch = true) {
  const section = qs("#concept-library");
  if (!section) {
    window.location.href = filter === "favorites" ? "concepts.html#favorites" : "concepts.html";
    return;
  }
  section.hidden = false;
  conceptState.mode = "library";
  conceptState.filter = filter;
  if (clearSearch) {
    conceptState.query = "";
    qs("#conceptPageSearch").value = "";
  }
  renderConceptBrowser();
  section.scrollIntoView({ behavior: "smooth", block: "start" });
  focusConceptPageSearch();
}

function openConceptDetail(id) {
  if (!getConcept(id)) return;
  conceptState.mode = "detail";
  conceptState.activeId = id;
  renderConceptPanel();
  openConceptPanel();
}

function toggleConceptFavorite(id) {
  if (conceptState.favorites.has(id)) {
    conceptState.favorites.delete(id);
  } else {
    conceptState.favorites.add(id);
  }
  saveConceptFavorites();
  renderConceptCounters();
  renderConceptPanel();
  renderConceptBrowser();
}

function renderConceptCounters() {
  const total = qs("#conceptTotal");
  const favorites = qs("#conceptFavoriteTotal");
  if (total) total.textContent = String(getConceptLibraryItems().length);
  if (favorites) favorites.textContent = String(getFavoriteConceptItems().length);
}

function getFilteredConcepts() {
  const query = conceptState.query.trim().toLocaleLowerCase();
  const source = conceptState.filter === "favorites" ? getFavoriteConceptItems() : getConceptLibraryItems();
  return source.filter((concept) => {
    if (!query) return true;
    const haystack = [
      concept.id,
      concept.title,
      concept.summary,
      getConceptArticleText(getConceptArticle(concept)),
      ...(concept.aliases || [])
    ].join(" ").toLocaleLowerCase();
    return haystack.includes(query);
  });
}

function getConceptTitle(concept) {
  return concept.title || concept.id;
}

function getConceptArticle(concept) {
  return concept.article || concept.body || concept.content || concept.summary || "";
}

function getConceptArticleText(article) {
  if (!Array.isArray(article)) return String(article || "");
  return article.map((block) => {
    if (typeof block === "string") return block;
    if (block.text) return block.text;
    if (block.title) return block.title;
    if (block.code) return block.code;
    if (block.items) return block.items.map((item) => typeof item === "string" ? item : `${item.title || ""} ${item.text || ""}`).join(" ");
    if (block.steps) return block.steps.map((step) => typeof step === "string" ? step : `${step.title || ""} ${step.text || ""}`).join(" ");
    if (block.rows) return block.rows.flat().join(" ");
    return "";
  }).join(" ");
}

function getConceptPreview(concept) {
  const article = getConceptArticle(concept);
  const articleText = getConceptArticleText(article);
  const text = (concept.summary || articleText).replace(/\s+/g, " ").trim();
  if (!text) return "暂无正文。";
  return text.length > 96 ? `${text.slice(0, 96)}...` : text;
}

function getConceptBrowserIntro() {
  const total = getConceptLibraryItems().length;
  const favoriteTotal = getFavoriteConceptItems().length;
  const visibleTotal = getFilteredConcepts().length;
  if (conceptState.filter === "favorites") {
    return favoriteTotal
      ? `收藏列表中有 ${favoriteTotal} 个概念，当前显示 ${visibleTotal} 个。点击条目会在右侧打开解释。`
      : "暂无收藏概念。";
  }
  return total
    ? `概念库中有 ${total} 个概念，当前显示 ${visibleTotal} 个。点击条目会在右侧打开解释。`
    : "暂无概念。";
}

function renderConceptList() {
  const rows = getFilteredConcepts();
  if (!rows.length) {
    if (conceptState.query.trim()) {
      return `<p class="concept-empty">没有匹配结果。</p>`;
    }
    if (conceptState.filter === "favorites") {
      return `<p class="concept-empty">暂无收藏概念。</p>`;
    }
    return `<p class="concept-empty">暂无概念。</p>`;
  }
  return `
    <div class="concept-list">
      ${rows.map((concept) => `
        <div class="concept-list-row">
          <button type="button" class="concept-list-main" data-concept-open="${concept.id}">
            <strong>${escapeHtml(getConceptTitle(concept))}</strong>
            <span>${escapeHtml(getConceptPreview(concept))}</span>
          </button>
          <button type="button" class="concept-favorite" data-concept-favorite="${concept.id}" aria-label="${conceptState.favorites.has(concept.id) ? "取消收藏" : "收藏"} ${escapeHtml(getConceptTitle(concept))}" aria-pressed="${conceptState.favorites.has(concept.id)}">${conceptState.favorites.has(concept.id) ? "★" : "☆"}</button>
        </div>
      `).join("")}
    </div>
  `;
}

function renderInlineText(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

function renderConceptParagraph(text, className = "") {
  const attr = className ? ` class="${className}"` : "";
  return `<p${attr}>${renderInlineText(text)}</p>`;
}

function renderConceptTable(block) {
  const headers = block.headers || [];
  const rows = block.rows || [];
  return `
    <div class="concept-table-wrap">
      ${block.title ? `<h4>${renderInlineText(block.title)}</h4>` : ""}
      <table class="concept-table">
        ${headers.length ? `<thead><tr>${headers.map((header) => `<th>${renderInlineText(header)}</th>`).join("")}</tr></thead>` : ""}
        <tbody>
          ${rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInlineText(cell)}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderConceptFlow(block) {
  const steps = block.steps || [];
  return `
    <div class="concept-flow" role="img" aria-label="${escapeHtml(block.title || "概念流程图")}">
      ${block.title ? `<h4>${renderInlineText(block.title)}</h4>` : ""}
      <div class="concept-flow-steps">
        ${steps.map((step, index) => {
          const title = typeof step === "string" ? step : step.title;
          const text = typeof step === "string" ? "" : step.text;
          return `
            <div class="concept-flow-step">
              <span>${index + 1}</span>
              <strong>${renderInlineText(title)}</strong>
              ${text ? `<small>${renderInlineText(text)}</small>` : ""}
            </div>
          `;
        }).join('<div class="concept-flow-arrow">↓</div>')}
      </div>
    </div>
  `;
}

function renderConceptBlock(block) {
  if (typeof block === "string") {
    return renderConceptParagraph(block.trim());
  }
  switch (block.type || "paragraph") {
    case "heading":
      return `<h3>${renderInlineText(block.title || "")}</h3>`;
    case "subheading":
      return `<h4>${renderInlineText(block.title || "")}</h4>`;
    case "lead":
      return renderConceptParagraph(block.text || "", "concept-summary");
    case "list": {
      const tag = block.ordered ? "ol" : "ul";
      return `<${tag} class="concept-points">${(block.items || []).map((item) => `<li>${renderInlineText(item)}</li>`).join("")}</${tag}>`;
    }
    case "code":
      return `<pre class="concept-code"><code>${escapeHtml(block.code || "")}</code></pre>`;
    case "table":
      return renderConceptTable(block);
    case "flow":
      return renderConceptFlow(block);
    case "callout":
      return `<div class="concept-callout"><strong>${renderInlineText(block.title || "注意")}</strong><p>${renderInlineText(block.text || "")}</p></div>`;
    default:
      return renderConceptParagraph(block.text || "");
  }
}

function renderConceptArticle(article) {
  if (Array.isArray(article)) {
    if (!article.length) {
      return `<p class="concept-empty">暂无正文。</p>`;
    }
    return article.map(renderConceptBlock).join("");
  }

  const normalized = String(article || "").trim();
  if (!normalized) {
    return `<p class="concept-empty">暂无正文。</p>`;
  }
  return normalized
    .split(/\n\s*\n/)
    .map((paragraph) => renderConceptParagraph(paragraph.trim()))
    .join("");
}

function renderConceptDetail(concept) {
  return `
    <div class="concept-detail">
      <article class="concept-article">
        ${renderConceptArticle(getConceptArticle(concept))}
      </article>
    </div>
  `;
}

function renderConceptBrowser() {
  const section = qs("#concept-library");
  if (!section || section.hidden) return;
  qsa("#conceptPageFilter button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.filter === conceptState.filter));
  });
  qs("#conceptPageTitle").textContent = conceptState.filter === "favorites" ? "收藏概念" : "概念库";
  qs("#conceptPageIntro").textContent = getConceptBrowserIntro();
  qs("#conceptPageList").innerHTML = renderConceptList();
}

function renderConceptPanel() {
  const concept = getConcept(conceptState.activeId);
  const favorite = qs("#conceptPanelFavorite");
  if (!concept) {
    qs("#conceptDrawerTitle").textContent = "选择一个概念";
    qs("#conceptDrawerBody").innerHTML = `<p class="concept-empty">选择概念后，这里会显示解释。</p>`;
    favorite.textContent = "☆";
    favorite.disabled = true;
    favorite.removeAttribute("data-concept-favorite");
    favorite.setAttribute("aria-pressed", "false");
    favorite.setAttribute("aria-label", "收藏当前概念");
    return;
  }
  qs("#conceptDrawerTitle").textContent = getConceptTitle(concept);
  qs("#conceptDrawerBody").innerHTML = renderConceptDetail(concept);
  favorite.disabled = false;
  favorite.dataset.conceptFavorite = concept.id;
  favorite.textContent = conceptState.favorites.has(concept.id) ? "★" : "☆";
  favorite.setAttribute("aria-pressed", String(conceptState.favorites.has(concept.id)));
  favorite.setAttribute("aria-label", `${conceptState.favorites.has(concept.id) ? "取消收藏" : "收藏"} ${getConceptTitle(concept)}`);
}

function setupConcepts() {
  if (!Array.isArray(concepts)) return;
  conceptState.favorites = loadConceptFavorites();
  conceptState.aliasEntries = buildConceptAliasEntries();
  if (document.body.dataset.page !== "concepts") {
    linkConceptKeywords(qs("main"), 10);
  }
  renderConceptCounters();
  renderConceptPanel();

  const conceptLibraryOpen = qs("#conceptLibraryOpen");
  const conceptFavoritesOpen = qs("#conceptFavoritesOpen");
  const conceptDrawerClose = qs("#conceptDrawerClose");
  const conceptPanelFavorite = qs("#conceptPanelFavorite");
  const conceptPageSearch = qs("#conceptPageSearch");
  const conceptPageFilter = qs("#conceptPageFilter");
  const conceptLibrary = qs("#concept-library");
  const conceptDrawer = qs("#conceptDrawer");
  const main = qs("main");

  if (conceptLibraryOpen && conceptLibrary) {
    conceptLibraryOpen.addEventListener("click", () => {
      conceptState.filter = "all";
      renderConceptBrowser();
    });
  }
  if (conceptFavoritesOpen && conceptLibrary) {
    conceptFavoritesOpen.addEventListener("click", () => {
      conceptState.filter = "favorites";
      renderConceptBrowser();
    });
  }
  if (conceptDrawerClose) conceptDrawerClose.addEventListener("click", closeConceptPanel);
  if (conceptPanelFavorite) conceptPanelFavorite.addEventListener("click", (event) => {
    const id = event.currentTarget.dataset.conceptFavorite;
    if (id) toggleConceptFavorite(id);
  });
  if (conceptPageSearch) conceptPageSearch.addEventListener("input", (event) => {
    conceptState.query = event.target.value;
    renderConceptBrowser();
  });
  if (conceptPageFilter) conceptPageFilter.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    conceptState.filter = button.dataset.filter;
    if (document.body.dataset.page === "concepts") {
      history.replaceState(null, "", conceptState.filter === "favorites" ? "#favorites" : "#library");
    }
    renderConceptBrowser();
  });
  if (conceptLibrary) conceptLibrary.addEventListener("click", (event) => {
    const favorite = event.target.closest("[data-concept-favorite]");
    if (favorite) {
      toggleConceptFavorite(favorite.dataset.conceptFavorite);
      return;
    }
    const open = event.target.closest("[data-concept-open]");
    if (open) {
      openConceptDetail(open.dataset.conceptOpen);
    }
  });
  if (conceptDrawer) conceptDrawer.addEventListener("click", (event) => {
    const open = event.target.closest("[data-concept-open]");
    if (open) openConceptDetail(open.dataset.conceptOpen);
  });
  if (main) main.addEventListener("click", (event) => {
    const keyword = event.target.closest(".concept-keyword[data-concept-id]");
    if (!keyword) return;
    openConceptDetail(keyword.dataset.conceptId);
  });
  document.addEventListener("keydown", (event) => {
    const drawer = qs("#conceptDrawer");
    if (event.key === "Escape" && drawer && drawer.getAttribute("aria-hidden") === "false") {
      closeConceptPanel();
    }
  });

  if (conceptLibrary) {
    conceptState.filter = window.location.hash === "#favorites" ? "favorites" : "all";
    renderConceptBrowser();
    window.addEventListener("hashchange", () => {
      conceptState.filter = window.location.hash === "#favorites" ? "favorites" : "all";
      renderConceptBrowser();
    });
  }
}

function setupSearch() {
  const input = qs("#globalSearch");
  input.addEventListener("input", () => {
    const term = input.value.trim().toLowerCase();
    qsa("main .section, main .hero").forEach((section) => {
      const text = (section.dataset.search || section.textContent).toLowerCase();
      section.classList.toggle("hidden-by-search", Boolean(term && !text.includes(term)));
    });
  });
}

function setupScrollSpy() {
  const links = qsa("#toc a");
  const targets = links.map((link) => qs(link.getAttribute("href"))).filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      links.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`);
      });
    });
  }, { rootMargin: "-30% 0px -60% 0px", threshold: 0.01 });
  targets.forEach((target) => observer.observe(target));

  const progress = qs("#readProgress");
  window.addEventListener("scroll", () => {
    const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
    const max = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    progress.style.width = `${Math.max(0, Math.min(100, (scrollTop / max) * 100))}%`;
  }, { passive: true });
}

if (qs("#dirButtons")) renderDirs();
if (qs("#startupSteps")) renderStartup();
if (qs("#pathSelect")) renderPaths();
if (qs("#moduleTable")) renderModules();
setupBeginnerGuides();
setupConcepts();
if (qs("#globalSearch")) setupSearch();
if (qs("#toc")) setupScrollSpy();
