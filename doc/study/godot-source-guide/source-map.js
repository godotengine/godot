// Interactive architecture map for the static source guide.
const mapSvgNs = "http://www.w3.org/2000/svg";
const mapNodeWidth = 210;
const mapNodeHeight = 96;

const mapState = {
  selectedNodeId: null,
  searchTerm: "",
  groupEnabled: new Set(),
  relationEnabled: new Set(),
  transform: { x: 0, y: 0, scale: 1 },
  dragging: false,
  dragStart: { x: 0, y: 0 },
  dragOrigin: { x: 0, y: 0 }
};

const mapGraph = sourceMapGraph;
const mapNodesById = new Map(mapGraph.nodes.map((node) => [node.id, node]));
const mapGroupsById = new Map(mapGraph.groups.map((group) => [group.id, group]));
const mapEdgesByNode = new Map();

for (const node of mapGraph.nodes) {
  mapEdgesByNode.set(node.id, { in: [], out: [] });
}
for (const edge of mapGraph.edges) {
  mapEdgesByNode.get(edge.from)?.out.push(edge);
  mapEdgesByNode.get(edge.to)?.in.push(edge);
}

function mapQs(selector, root = document) {
  return root.querySelector(selector);
}

function mapQsa(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

function mapEscape(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;"
  })[char]);
}

function mapSvgEl(name, attrs = {}) {
  const element = document.createElementNS(mapSvgNs, name);
  for (const [key, value] of Object.entries(attrs)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function mapCharWidth(char) {
  if (/\s/.test(char)) return 0.35;
  if (/[A-Za-z0-9_:+#.-]/.test(char)) return 0.58;
  if (/[()\/\\]/.test(char)) return 0.45;
  return 1;
}

function mapTextWidth(text) {
  return Array.from(String(text || "")).reduce((sum, char) => sum + mapCharWidth(char), 0);
}

function mapTrimLine(line) {
  return line.trim().replace(/[。；，、:：/]$/, "");
}

function mapWrapText(text, maxUnits, maxLines = 2) {
  const chars = Array.from(String(text || ""));
  const lines = [];
  let current = "";
  let currentWidth = 0;
  let consumed = 0;

  for (const char of chars) {
    const charWidth = mapCharWidth(char);
    const shouldBreakBefore = current && currentWidth + charWidth > maxUnits;
    if (shouldBreakBefore) {
      lines.push(mapTrimLine(current));
      current = "";
      currentWidth = 0;
      if (lines.length === maxLines) break;
      if (/\s/.test(char)) {
        consumed += 1;
        continue;
      }
    }
    current += char;
    currentWidth += charWidth;
    consumed += 1;
    if (/[，。；、]/.test(char)) {
      lines.push(mapTrimLine(current));
      current = "";
      currentWidth = 0;
    }
    if (lines.length === maxLines) break;
  }
  if (current && lines.length < maxLines) lines.push(mapTrimLine(current));
  if (consumed < chars.length && lines.length) {
    lines[lines.length - 1] = `${mapTrimLine(lines[lines.length - 1])}...`;
  }
  return lines.filter(Boolean);
}

function mapWrapTitle(text) {
  const value = String(text || "");
  if (mapTextWidth(value) <= 13) return [value];
  const slashIndex = value.indexOf(" / ");
  if (slashIndex > 0) {
    return [value.slice(0, slashIndex), value.slice(slashIndex + 3)].slice(0, 2);
  }
  const spaceIndex = value.lastIndexOf(" ", 22);
  if (spaceIndex > 8) {
    return [value.slice(0, spaceIndex), value.slice(spaceIndex + 1)].slice(0, 2);
  }
  return mapWrapText(value, 13, 2);
}

function getNodeRelations(nodeId) {
  return mapEdgesByNode.get(nodeId) || { in: [], out: [] };
}

function getRelationColor(type) {
  return mapGraph.relationTypes[type]?.color || "#65756d";
}

function relationLabel(type) {
  return mapGraph.relationTypes[type]?.label || type;
}

function isNodeEnabled(node) {
  return mapState.groupEnabled.has(node.group);
}

function edgeMatchesSelected(edge) {
  return mapState.selectedNodeId && (edge.from === mapState.selectedNodeId || edge.to === mapState.selectedNodeId);
}

function isEdgeVisible(edge) {
  const from = mapNodesById.get(edge.from);
  const to = mapNodesById.get(edge.to);
  if (!from || !to || !isNodeEnabled(from) || !isNodeEnabled(to)) return false;
  if (!mapState.relationEnabled.has(edge.type)) return false;
  return edge.defaultVisible !== false || edgeMatchesSelected(edge);
}

function nodeMatchesSearch(node) {
  if (!mapState.searchTerm) return true;
  const haystack = [
    node.id,
    node.title,
    node.summary,
    node.beginner,
    node.group,
    node.conceptId,
    ...(node.tags || []),
    ...(node.sourceAnchors || [])
  ].join(" ").toLocaleLowerCase();
  return haystack.includes(mapState.searchTerm);
}

function findBestSearchMatch() {
  if (!mapState.searchTerm) return null;
  const enabled = mapGraph.nodes.filter((node) => isNodeEnabled(node));
  const exact = enabled.find((node) => [
    node.id,
    node.title,
    node.conceptId,
    ...(node.tags || [])
  ].some((value) => String(value || "").toLocaleLowerCase() === mapState.searchTerm));
  if (exact) return exact;
  const startsWith = enabled.find((node) => [
    node.id,
    node.title,
    node.conceptId,
    ...(node.tags || [])
  ].some((value) => String(value || "").toLocaleLowerCase().startsWith(mapState.searchTerm)));
  if (startsWith) return startsWith;
  return enabled.find((node) => nodeMatchesSearch(node)) || null;
}

function updateMapTransform() {
  const viewport = mapQs("#sourceMapViewport");
  if (!viewport) return;
  viewport.setAttribute("transform", `translate(${mapState.transform.x} ${mapState.transform.y}) scale(${mapState.transform.scale})`);
  mapQs("#mapMiniStatus").textContent = `${Math.round(mapState.transform.scale * 100)}%`;
}

function fitMapToScreen() {
  const svg = mapQs("#sourceMapSvg");
  const rect = svg.getBoundingClientRect();
  const padding = 36;
  const scale = Math.min(
    (rect.width - padding * 2) / mapGraph.viewBox.width,
    (rect.height - padding * 2) / mapGraph.viewBox.height
  );
  mapState.transform.scale = Math.max(0.2, Math.min(1.2, scale));
  mapState.transform.x = (rect.width - mapGraph.viewBox.width * mapState.transform.scale) / 2;
  mapState.transform.y = (rect.height - mapGraph.viewBox.height * mapState.transform.scale) / 2;
  updateMapTransform();
}

function resetMapView() {
  mapState.transform = { x: 24, y: 24, scale: 0.72 };
  updateMapTransform();
}

function focusSelectedNode() {
  const node = mapNodesById.get(mapState.selectedNodeId);
  if (!node) {
    fitMapToScreen();
    return;
  }
  const svg = mapQs("#sourceMapSvg");
  const rect = svg.getBoundingClientRect();
  const focusIds = new Set([node.id]);
  for (const edge of mapGraph.edges) {
    if (!edgeMatchesSelected(edge) || !isEdgeVisible(edge)) continue;
    const otherId = edge.from === node.id ? edge.to : edge.from;
    const other = mapNodesById.get(otherId);
    if (other?.group === node.group) focusIds.add(otherId);
  }
  const groupNodes = mapGraph.nodes.filter((item) => item.group === node.group && isNodeEnabled(item));
  if (groupNodes.length <= 8) {
    for (const groupNode of groupNodes) focusIds.add(groupNode.id);
  }
  let minX = node.x - mapNodeWidth / 2;
  let maxX = node.x + mapNodeWidth / 2;
  let minY = node.y - mapNodeHeight / 2;
  let maxY = node.y + mapNodeHeight / 2;
  for (const id of focusIds) {
    const focusNode = mapNodesById.get(id);
    if (!focusNode || !isNodeEnabled(focusNode)) continue;
    minX = Math.min(minX, focusNode.x - mapNodeWidth / 2);
    maxX = Math.max(maxX, focusNode.x + mapNodeWidth / 2);
    minY = Math.min(minY, focusNode.y - mapNodeHeight / 2);
    maxY = Math.max(maxY, focusNode.y + mapNodeHeight / 2);
  }
  const padding = 56;
  const focusWidth = Math.max(1, maxX - minX);
  const focusHeight = Math.max(1, maxY - minY);
  const scale = Math.max(0.48, Math.min(
    1.08,
    (rect.width - padding * 2) / focusWidth,
    (rect.height - padding * 2) / focusHeight
  ));
  mapState.transform.scale = scale;
  mapState.transform.x = (rect.width - focusWidth * scale) / 2 - minX * scale;
  mapState.transform.y = (rect.height - focusHeight * scale) / 2 - minY * scale;
  updateMapTransform();
}

function renderFilterControls() {
  const groupWrap = mapQs("#mapGroupFilters");
  const relationWrap = mapQs("#mapRelationFilters");
  groupWrap.innerHTML = mapGraph.groups.map((group) => `
    <button type="button" class="map-filter-chip" data-map-group="${mapEscape(group.id)}" aria-pressed="true">
      <span class="map-filter-swatch" style="background:${mapEscape(group.color)}"></span>
      <span>${mapEscape(group.title)}</span>
    </button>
  `).join("");
  relationWrap.innerHTML = Object.entries(mapGraph.relationTypes).map(([id, relation]) => `
    <button type="button" class="map-filter-chip" data-map-relation="${mapEscape(id)}" aria-pressed="true">
      <span class="map-filter-swatch" style="background:${mapEscape(relation.color)}"></span>
      <span>${mapEscape(relation.label)}</span>
    </button>
  `).join("");
}

function renderGroupLayer(root) {
  const layer = mapSvgEl("g", { class: "source-map-groups" });
  for (const group of mapGraph.groups) {
    const groupEl = mapSvgEl("g", { class: `source-map-group ${mapState.groupEnabled.has(group.id) ? "" : "is-disabled"}` });
    groupEl.appendChild(mapSvgEl("rect", {
      x: group.x,
      y: group.y,
      width: group.width,
      height: group.height,
      rx: 18,
      fill: group.color
    }));
    const title = mapSvgEl("text", {
      x: group.x + 24,
      y: group.y + 42,
      class: "source-map-group-title"
    });
    title.textContent = group.title;
    groupEl.appendChild(title);
    const desc = mapSvgEl("text", {
      x: group.x + 24,
      y: group.y + 66,
      class: "source-map-group-desc"
    });
    desc.textContent = group.description;
    groupEl.appendChild(desc);
    layer.appendChild(groupEl);
  }
  root.appendChild(layer);
}

function edgePath(edge) {
  const from = mapNodesById.get(edge.from);
  const to = mapNodesById.get(edge.to);
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const curve = Math.max(90, Math.min(260, Math.abs(dx) * 0.35));
  const c1x = from.x + Math.sign(dx || 1) * curve;
  const c1y = from.y + dy * 0.08;
  const c2x = to.x - Math.sign(dx || 1) * curve;
  const c2y = to.y - dy * 0.08;
  return `M ${from.x} ${from.y} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${to.x} ${to.y}`;
}

function renderEdgeLayer(root) {
  const layer = mapSvgEl("g", { class: "source-map-edges" });
  for (const edge of mapGraph.edges) {
    if (!isEdgeVisible(edge)) continue;
    const path = mapSvgEl("path", {
      d: edgePath(edge),
      class: `source-map-edge ${edgeMatchesSelected(edge) ? "is-adjacent" : ""} ${edge.defaultVisible === false ? "is-extra" : ""}`,
      stroke: getRelationColor(edge.type),
      "data-edge-from": edge.from,
      "data-edge-to": edge.to,
      "data-edge-type": edge.type
    });
    path.appendChild(mapSvgEl("title"));
    path.querySelector("title").textContent = `${relationLabel(edge.type)}：${edge.label}`;
    layer.appendChild(path);
  }
  root.appendChild(layer);
}

function renderNodeText(group, lines, x, startY, className, lineHeight) {
  lines.forEach((line, index) => {
    const text = mapSvgEl("text", {
      x,
      y: startY + index * lineHeight,
      class: className,
      "text-anchor": "middle"
    });
    text.textContent = line;
    group.appendChild(text);
  });
}

function renderNodeLayer(root) {
  const layer = mapSvgEl("g", { class: "source-map-nodes" });
  for (const node of mapGraph.nodes) {
    const enabled = isNodeEnabled(node);
    const searchMatch = nodeMatchesSearch(node);
    const selected = mapState.selectedNodeId === node.id;
    const adjacent = mapState.selectedNodeId && mapGraph.edges.some((edge) => edgeMatchesSelected(edge) && (edge.from === node.id || edge.to === node.id));
    const nodeClass = [
      "source-map-node",
      `importance-${node.importance || 3}`,
      selected ? "is-selected" : "",
      adjacent && !selected ? "is-adjacent" : "",
      !enabled ? "is-hidden" : "",
      mapState.searchTerm && searchMatch ? "is-search-hit" : "",
      mapState.searchTerm && !searchMatch ? "is-search-muted" : ""
    ].filter(Boolean).join(" ");
    const nodeEl = mapSvgEl("g", {
      class: nodeClass,
      tabindex: enabled ? "0" : "-1",
      role: "button",
      "aria-label": node.title,
      transform: `translate(${node.x - mapNodeWidth / 2} ${node.y - mapNodeHeight / 2})`,
      "data-map-node": node.id
    });
    const group = mapGroupsById.get(node.group);
    nodeEl.appendChild(mapSvgEl("rect", {
      width: mapNodeWidth,
      height: mapNodeHeight,
      rx: 10,
      fill: "#ffffff",
      stroke: getRelationColor(node.group === "servers" ? "server-delegate" : "object-model")
    }));
    nodeEl.appendChild(mapSvgEl("rect", {
      x: 0,
      y: 0,
      width: mapNodeWidth,
      height: 8,
      rx: 6,
      fill: group?.color || "#e8eee9"
    }));
    renderNodeText(nodeEl, mapWrapTitle(node.title), mapNodeWidth / 2, 31, "source-map-node-title", 17);
    renderNodeText(nodeEl, mapWrapText(node.summary, 17.4, 2), mapNodeWidth / 2, 67, "source-map-node-summary", 15);
    nodeEl.addEventListener("click", () => selectMapNode(node.id, true));
    nodeEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectMapNode(node.id, true);
      }
    });
    layer.appendChild(nodeEl);
  }
  root.appendChild(layer);
}

function renderMap() {
  const svg = mapQs("#sourceMapSvg");
  svg.setAttribute("viewBox", `0 0 ${svg.clientWidth || 1200} ${svg.clientHeight || 780}`);
  svg.innerHTML = "";
  const defs = mapSvgEl("defs");
  defs.innerHTML = `
    <marker id="sourceMapArrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#65756d"></path>
    </marker>
  `;
  svg.appendChild(defs);
  const viewport = mapSvgEl("g", { id: "sourceMapViewport" });
  renderGroupLayer(viewport);
  renderEdgeLayer(viewport);
  renderNodeLayer(viewport);
  svg.appendChild(viewport);
  updateMapTransform();
  updateStatus();
}

function relationRows(edges, direction) {
  if (!edges.length) return `<p class="map-relation-empty">暂无${direction}关系。</p>`;
  return edges.map((edge) => {
    const otherId = direction === "依赖" ? edge.to : edge.from;
    const other = mapNodesById.get(otherId);
    return `
      <button type="button" class="map-relation-row" data-map-node="${mapEscape(otherId)}">
        <span class="map-relation-type" style="border-color:${mapEscape(getRelationColor(edge.type))}">${mapEscape(relationLabel(edge.type))}</span>
        <strong>${mapEscape(other?.title || otherId)}</strong>
        <span>${mapEscape(edge.label)}：${mapEscape(edge.explanation)}</span>
      </button>
    `;
  }).join("");
}

function renderDetail(node) {
  const group = mapGroupsById.get(node.group);
  const relations = getNodeRelations(node.id);
  const hiddenAdjacent = [...relations.in, ...relations.out].filter((edge) => edge.defaultVisible === false).length;
  const conceptHref = node.conceptId ? `concepts.html#concept-${node.conceptId}` : "";
  const showArticleLink = Boolean(node.articleHref && node.articleHref !== conceptHref);
  mapQs("#mapDetail").innerHTML = `
    <div class="source-map-detail-card">
      <p class="detail-kicker">${mapEscape(group?.title || node.group)}</p>
      <h2>${mapEscape(node.title)}</h2>
      <p class="detail-summary">${mapEscape(node.summary)}</p>
      <div class="detail-beginner">
        <strong>小白版</strong>
        <p>${mapEscape(node.beginner || node.summary)}</p>
      </div>
      <div class="detail-actions">
        ${showArticleLink ? `<a class="button primary" href="${mapEscape(node.articleHref)}">跳到正文片段</a>` : ""}
        ${conceptHref ? `<a class="button ${showArticleLink ? "" : "primary"}" href="${mapEscape(conceptHref)}">打开概念解释</a>` : ""}
      </div>
      <div class="detail-meta">
        <h3>源码锚点</h3>
        <div class="detail-source-list">
          ${(node.sourceAnchors || []).map((anchor) => `<span class="source">${mapEscape(anchor)}</span>`).join("")}
        </div>
      </div>
      <div class="detail-meta">
        <h3>关系概览</h3>
        <p>${relations.out.length} 条依赖，${relations.in.length} 条被依赖。${hiddenAdjacent ? `其中 ${hiddenAdjacent} 条高扇出关系只在聚焦这个节点时显示。` : "当前节点没有被默认折叠的关系。"}</p>
      </div>
      <div class="detail-meta">
        <h3>依赖</h3>
        <div class="map-relation-list">${relationRows(relations.out, "依赖")}</div>
      </div>
      <div class="detail-meta">
        <h3>被依赖</h3>
        <div class="map-relation-list">${relationRows(relations.in, "被依赖")}</div>
      </div>
    </div>
  `;
}

function selectMapNode(nodeId, updateHash = false) {
  const node = mapNodesById.get(nodeId);
  if (!node) return;
  mapState.selectedNodeId = nodeId;
  renderDetail(node);
  renderMap();
  if (updateHash) {
    history.replaceState(null, "", `#node=${encodeURIComponent(nodeId)}`);
  }
}

function updateStatus() {
  const visibleNodes = mapGraph.nodes.filter(isNodeEnabled).length;
  const visibleEdges = mapGraph.edges.filter(isEdgeVisible).length;
  const matches = mapState.searchTerm ? mapGraph.nodes.filter((node) => isNodeEnabled(node) && nodeMatchesSearch(node)).length : visibleNodes;
  const selected = mapState.selectedNodeId ? mapNodesById.get(mapState.selectedNodeId)?.title : "";
  mapQs("#mapStatus").textContent = selected
    ? `已聚焦：${selected}。当前显示 ${visibleNodes} 个节点、${visibleEdges} 条关系，搜索命中 ${matches} 个节点。`
    : `当前显示 ${visibleNodes} 个节点、${visibleEdges} 条主关系，搜索命中 ${matches} 个节点。`;
}

function setupMapControls() {
  mapState.groupEnabled = new Set(mapGraph.groups.map((group) => group.id));
  mapState.relationEnabled = new Set(Object.keys(mapGraph.relationTypes));
  renderFilterControls();

  mapQs("#mapGroupFilters").addEventListener("click", (event) => {
    const button = event.target.closest("[data-map-group]");
    if (!button) return;
    const id = button.dataset.mapGroup;
    if (mapState.groupEnabled.has(id)) {
      mapState.groupEnabled.delete(id);
    } else {
      mapState.groupEnabled.add(id);
    }
    button.setAttribute("aria-pressed", String(mapState.groupEnabled.has(id)));
    if (mapState.selectedNodeId && !isNodeEnabled(mapNodesById.get(mapState.selectedNodeId))) {
      mapState.selectedNodeId = null;
      mapQs("#mapDetail").innerHTML = `<div class="source-map-detail-empty"><p class="detail-kicker">分区已隐藏</p><h2>选择一个仍可见的节点</h2><p>当前选中节点所在分区已被关闭。</p></div>`;
    }
    renderMap();
  });

  mapQs("#mapRelationFilters").addEventListener("click", (event) => {
    const button = event.target.closest("[data-map-relation]");
    if (!button) return;
    const id = button.dataset.mapRelation;
    if (mapState.relationEnabled.has(id)) {
      mapState.relationEnabled.delete(id);
    } else {
      mapState.relationEnabled.add(id);
    }
    button.setAttribute("aria-pressed", String(mapState.relationEnabled.has(id)));
    renderMap();
  });

  const search = mapQs("#mapSearch");
  search.addEventListener("input", (event) => {
    mapState.searchTerm = event.target.value.trim().toLocaleLowerCase();
    renderMap();
  });
  search.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    const first = findBestSearchMatch();
    if (first) {
      selectMapNode(first.id, true);
      focusSelectedNode();
    }
  });

  mapQs("#mapFit").addEventListener("click", fitMapToScreen);
  mapQs("#mapReset").addEventListener("click", resetMapView);
  mapQs("#mapFocus").addEventListener("click", focusSelectedNode);
  mapQs("#mapDetail").addEventListener("click", (event) => {
    const button = event.target.closest("[data-map-node]");
    if (button) {
      selectMapNode(button.dataset.mapNode, true);
      focusSelectedNode();
    }
  });
}

function setupPanZoom() {
  const svg = mapQs("#sourceMapSvg");
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = svg.getBoundingClientRect();
    const pointerX = event.clientX - rect.left;
    const pointerY = event.clientY - rect.top;
    const oldScale = mapState.transform.scale;
    const factor = event.deltaY < 0 ? 1.12 : 0.88;
    const newScale = Math.max(0.18, Math.min(2.4, oldScale * factor));
    const graphX = (pointerX - mapState.transform.x) / oldScale;
    const graphY = (pointerY - mapState.transform.y) / oldScale;
    mapState.transform.scale = newScale;
    mapState.transform.x = pointerX - graphX * newScale;
    mapState.transform.y = pointerY - graphY * newScale;
    updateMapTransform();
  }, { passive: false });

  svg.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".source-map-node")) return;
    mapState.dragging = true;
    mapState.dragStart = { x: event.clientX, y: event.clientY };
    mapState.dragOrigin = { x: mapState.transform.x, y: mapState.transform.y };
    svg.setPointerCapture(event.pointerId);
    svg.classList.add("is-dragging");
  });
  svg.addEventListener("pointermove", (event) => {
    if (!mapState.dragging) return;
    mapState.transform.x = mapState.dragOrigin.x + event.clientX - mapState.dragStart.x;
    mapState.transform.y = mapState.dragOrigin.y + event.clientY - mapState.dragStart.y;
    updateMapTransform();
  });
  svg.addEventListener("pointerup", (event) => {
    mapState.dragging = false;
    svg.releasePointerCapture(event.pointerId);
    svg.classList.remove("is-dragging");
  });
  svg.addEventListener("pointercancel", () => {
    mapState.dragging = false;
    svg.classList.remove("is-dragging");
  });
}

function setupDeepLink() {
  const applyMapHash = () => {
    const match = window.location.hash.match(/^#node=(.+)$/);
    if (!match) return;
    const nodeId = decodeURIComponent(match[1]);
    const node = mapNodesById.get(nodeId);
    if (!node) return;
    mapState.groupEnabled.add(node.group);
    const groupButton = mapQs(`[data-map-group="${CSS.escape(node.group)}"]`);
    if (groupButton) groupButton.setAttribute("aria-pressed", "true");
    selectMapNode(nodeId, false);
    focusSelectedNode();
  };
  applyMapHash();
  window.addEventListener("hashchange", applyMapHash);
}

function initSourceMap() {
  setupMapControls();
  setupPanZoom();
  renderMap();
  resetMapView();
  setupDeepLink();
  window.addEventListener("resize", () => {
    renderMap();
    updateMapTransform();
  });
}

initSourceMap();
