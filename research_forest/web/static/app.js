"use strict";

const KINDS = ["paper", "method", "dataset", "limitation", "question"];
const KIND_LABEL = { paper: "Papers", method: "Methods", dataset: "Datasets", limitation: "Limitations", question: "Open questions" };
const CHANGE_TYPES = ["architecture", "objective", "training", "data", "efficiency", "evaluation", "theory", "application", "other"];
// paper-to-paper relations shown in both views: [edge kind, label, shown by default]
const RELATIONS = [["builds_on", "Builds on", true], ["compares_with", "Baseline comparisons", true],
                   ["cites", "Citations", true]];
const SHAPES = { paper: "ellipse", method: "round-rectangle", dataset: "barrel", limitation: "triangle", question: "diamond" };
const EDGE_LABEL = {
  builds_on: "builds on", cites: "cites", proposes: "proposes", uses: "uses", evaluates_on: "evaluated on",
  has_limitation: "has limitation", addresses: "addresses", motivates: "motivates", about: "about",
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const KIND_SET = new Set(KINDS);
const kindOf = (k) => (KIND_SET.has(k) ? k : "paper");  // never interpolate unknown kinds into markup
const safeUrl = (u) => { try { const x = new URL(u); return ["http:", "https:"].includes(x.protocol) ? x.href : null; } catch { return null; } };
const extLink = (u, label) => (safeUrl(u) ? `<a href="${esc(safeUrl(u))}" target="_blank" rel="noopener noreferrer">${label}</a>` : "");

const state = { project: null, view: "lineage", mode: "refine", graph: null, logSince: 0, polling: null, running: false,
  lastTurn: null };  // previous question + cited nodes, so follow-ups ("which of those…") resolve
let cy = null;

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, credentials: "same-origin", ...opts });
  if (res.status === 401) {
    $("status").textContent = "Not authorised — open the URL printed by `research-forest serve`.";
    throw new Error("unauthorized");
  }
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

/* ------------------------------------------------------------------ setup */

function initFilters() {
  $("kind-filters").innerHTML = KINDS.map((k) =>
    `<label><input type="checkbox" data-kind="${k}" checked><span class="dot" style="background:var(--${k})"></span>${KIND_LABEL[k]}</label>`).join("");
  $("change-filters").innerHTML = CHANGE_TYPES.map((c) =>
    `<label><input type="checkbox" data-change="${c}" checked>${c}</label>`).join("");
  $("relation-filters").innerHTML = RELATIONS.map(([k, label, on]) =>
    `<label><input type="checkbox" data-relation="${k}" ${on ? "checked" : ""}>${label}</label>`).join("");
  for (const el of document.querySelectorAll(
    "#kind-filters input, #change-filters input, #relation-filters input, #year-min, #year-max"))
    el.addEventListener("change", applyFilters);
}

function initCy() {
  cy = cytoscape({
    container: $("cy"), wheelSensitivity: 0.25, minZoom: 0.05, maxZoom: 3,
    style: [
      { selector: "node", style: {
        "background-color": (n) => css(`--${n.data("kind")}`), shape: (n) => SHAPES[n.data("kind")],
        width: "data(size)", height: "data(size)", label: "data(short)", color: css("--text"),
        "font-size": 9, "text-valign": "bottom", "text-margin-y": 3, "text-wrap": "ellipsis", "text-max-width": 120,
        "text-outline-color": css("--bg"), "text-outline-width": 2 } },
      { selector: "edge", style: {
        width: 1, "line-color": css("--edge"), "curve-style": "bezier", "target-arrow-shape": "triangle",
        "target-arrow-color": css("--edge"), "arrow-scale": 0.6, opacity: 0.55 } },
      { selector: "edge[kind = 'builds_on']", style: {
        width: 2.2, "line-color": css("--builds"), "target-arrow-color": css("--builds"), opacity: 0.9,
        label: "data(change_type)", "font-size": 8, color: css("--muted"), "text-rotation": "autorotate",
        "text-background-color": css("--bg"), "text-background-opacity": 1, "text-background-padding": 1 } },
      { selector: "edge[kind = 'cites']", style: { "line-style": "dashed", opacity: 0.3 } },
      { selector: "edge[kind = 'compares_with']", style: {
        "line-style": "dotted", width: 1.6, "line-color": css("--limitation"), "target-arrow-color": css("--limitation"),
        opacity: 0.75 } },
      { selector: "node[?new]", style: { "border-width": 3, "border-style": "double", "border-color": css("--accent") } },
      { selector: ".faded", style: { opacity: 0.12 } },
      { selector: ".hit", style: { "border-width": 3, "border-color": css("--accent") } },
      { selector: "node:selected", style: { "border-width": 4, "border-color": css("--text") } },
    ],
  });
  cy.on("tap", "node", (e) => showNode(e.target.id()));
  cy.on("tap", (e) => { if (e.target === cy) cy.elements().removeClass("faded hit"); });
}

/* ------------------------------------------------------------- projects */

async function loadProjects(selectId) {
  const projects = await api("/api/projects");
  $("project-select").innerHTML = projects.map((p) =>
    `<option value="${Number(p.id)}">${esc(String(p.title || p.prompt).slice(0, 80))}</option>`).join("");
  $("delete-btn").classList.toggle("hidden", !projects.length);
  $("update-btn").classList.toggle("hidden", !projects.length);
  if (!projects.length) return showStart();
  openProject(selectId ?? projects[0].id);
}

function showStart() {
  $("start").classList.remove("hidden");
  $("workspace").classList.add("hidden");
  $("prompt").focus();
}

async function openProject(id) {
  state.project = Number(id);
  state.logSince = 0;
  state.lastTurn = null;
  $("project-select").value = id;
  $("log").innerHTML = "";
  $("answer").classList.add("hidden");
  $("start").classList.add("hidden");
  $("workspace").classList.remove("hidden");
  if (!cy) initCy();
  await refreshGraph();
  await poll();
}

async function poll() {
  clearTimeout(state.polling);
  const p = await api(`/api/projects/${state.project}?since=${state.logSince}`);
  for (const l of p.log) {
    $("log").insertAdjacentHTML("beforeend", `<li${l.msg.startsWith("warning:") ? ' class="warn"' : ""}>${esc(l.msg)}</li>`);
    state.logSince = Math.max(state.logSince, l.t);
  }
  $("log").scrollTop = $("log").scrollHeight;
  const meaningful = p.log.filter((l) => l.msg !== "done");  // keep the run summary visible, not "done"
  const last = meaningful.length ? meaningful[meaningful.length - 1].msg : null;
  if (last) $("status").textContent = last;
  else if (!p.running) $("status").textContent = `${p.status}`;
  $("status").classList.toggle("busy", p.running);
  $("undo-btn").disabled = p.running || !p.snapshots.length;
  $("update-btn").disabled = p.running;
  $("update-btn").title = `Look for papers published since the last check (${p.last_checked})`;
  state.latestNew = p.latest_new || {};
  if (state.graph) renderNew(state.graph);
  renderProgress(p.progress, p.running);
  const finished = state.running && !p.running;
  state.running = p.running;
  if (p.running) {
    if (p.log.length) await refreshGraph(true);
    state.polling = setTimeout(poll, 1200);
  } else if (finished) {
    await refreshGraph();
    loadProjects(state.project);
  }
}

const fmtTime = (s) => (s >= 60 ? `${Math.floor(s / 60)}m ${String(Math.floor(s % 60)).padStart(2, "0")}s` : `${Math.floor(s)}s`);

function renderProgress(p, running) {
  const box = $("progress");
  // visible while running, then for a few seconds after the job ends (longer if it failed)
  const linger = p && (p.error || p.warnings?.length) ? 30 : 10;
  const recent = p && p.finished && Date.now() / 1000 - p.finished < linger;
  clearTimeout(state.hideProgress);
  if (!p || !(running || recent)) {
    box.classList.add("hidden");
    return;
  }
  if (!running) state.hideProgress = setTimeout(() => box.classList.add("hidden"), linger * 1000);
  box.classList.remove("hidden");
  box.classList.toggle("error", Boolean(p.error));
  const keys = p.plan.map((s) => s.key);
  const idx = Math.max(0, keys.indexOf(p.stage));
  const doneAll = p.finished && !p.error;
  $("progress-steps").innerHTML = p.plan.map((s, i) => {
    const cls = doneAll || i < idx ? "done" : i === idx ? (p.error ? "failed" : "current") : "";
    return `<li class="${cls}">${esc(s.label)}</li>`;
  }).join("");
  const frac = p.total ? p.done / p.total : 0;
  const overall = doneAll ? 1 : (idx + frac) / keys.length;
  $("progress-fill").style.width = `${Math.round(overall * 100)}%`;
  $("progress-title").textContent = p.error ? "Failed" : doneAll ? "Finished"
    : { refine: "Refining the graph", update: "Checking for new work" }[p.job] || "Building the graph";
  const elapsed = (p.finished || Date.now() / 1000) - p.started;
  $("progress-time").textContent = fmtTime(Math.max(0, elapsed));
  $("progress-label").textContent = p.error || p.label || "";
  $("progress-count").textContent = !p.error && p.total ? `${p.done} / ${p.total}` : "";
  $("progress-warnings").innerHTML = (p.warnings || []).map((w) => `<li>${esc(w)}</li>`).join("");
}

/* ---------------------------------------------------------------- graph */

function shortLabel(n) {
  if (n.kind !== "paper") return n.label.length > 48 ? n.label.slice(0, 46) + "…" : n.label;
  const t = n.label.split(/:|\s[–—-]\s/)[0].trim();  // "Mamba: Linear-Time …" -> "Mamba"; keeps "FlashAttention-2"
  return `${t.length > 36 ? t.slice(0, 34) + "…" : t}${n.year ? " ’" + String(n.year).slice(2) : ""}`;
}

async function refreshGraph(keepLayout = false) {
  const g = await api(`/api/projects/${state.project}/graph`);
  const prevCount = state.graph ? state.graph.nodes.length + state.graph.edges.length : -1;
  state.graph = g;
  $("empty").classList.toggle("hidden", g.nodes.some((n) => n.kind === "paper"));
  renderQuestions(g);
  renderNew(g);
  if (keepLayout && prevCount === g.nodes.length + g.edges.length) return;
  const els = [
    ...g.nodes.map((n) => ({ group: "nodes", data: { ...n, short: shortLabel(n),
      size: n.kind === "paper" ? 14 + 4 * Math.log1p(n.citations || 0) : 10 + 3 * Math.log1p(n.degree || 1) } })),
    ...g.edges.map((e, i) => ({ group: "edges", data: { ...e, id: `e${i}` } })),
  ];
  cy.batch(() => { cy.elements().remove(); cy.add(els); });
  applyFilters();
}

function layout() {
  const visible = cy.elements(":visible");
  if (state.view === "lineage") {
    // Timeline: x by year, papers stacked by citation count within each year.
    const papers = cy.nodes("[kind = 'paper']:visible");
    const byYear = {};
    papers.forEach((n) => (byYear[n.data("year") || 0] ||= []).push(n));
    const years = Object.keys(byYear).map(Number).sort((a, b) => a - b);
    const pos = {};
    years.forEach((y, xi) => {
      byYear[y].sort((a, b) => (b.data("citations") || 0) - (a.data("citations") || 0))
        .forEach((n, yi, arr) => { pos[n.id()] = { x: xi * 190, y: (yi - (arr.length - 1) / 2) * 62 }; });
    });
    visible.nodes().layout({ name: "preset", positions: (n) => pos[n.id()] || { x: 0, y: 0 }, fit: true, padding: 30 }).run();
  } else {
    visible.layout({ name: "cose", animate: false, nodeRepulsion: () => 9000, idealEdgeLength: () => 70,
      nodeOverlap: 8, gravity: 0.4, numIter: 1500, padding: 30 }).run();
  }
}

function applyFilters() {
  if (!cy) return;
  const kinds = new Set([...document.querySelectorAll("#kind-filters input:checked")].map((i) => i.dataset.kind));
  const changes = new Set([...document.querySelectorAll("#change-filters input:checked")].map((i) => i.dataset.change));
  const relations = new Set([...document.querySelectorAll("#relation-filters input:checked")].map((i) => i.dataset.relation));
  const yMin = Number($("year-min").value) || -Infinity, yMax = Number($("year-max").value) || Infinity;
  const lineage = state.view === "lineage";
  cy.batch(() => {
    cy.nodes().forEach((n) => {
      const k = n.data("kind"), y = n.data("year");
      let show = (lineage ? k === "paper" : kinds.has(k));
      if (k === "paper" && y && (y < yMin || y > yMax)) show = false;
      n.style("display", show ? "element" : "none");
    });
    // hide entities that lost all visible papers
    cy.nodes("[kind != 'paper']").forEach((n) => {
      if (n.style("display") === "none") return;
      const vis = n.neighborhood("node").filter((m) => m.style("display") !== "none");
      if (!vis.length) n.style("display", "none");
    });
    cy.edges().forEach((e) => {
      const k = e.data("kind");
      const paperRelation = RELATIONS.some(([r]) => r === k);
      const show = (k !== "builds_on" || changes.has(e.data("change_type")))
        && (!paperRelation || relations.has(k)) && (!lineage || paperRelation);
      e.style("display", show ? "element" : "none");
    });
  });
  layout();
}

function focusNode(id) {
  const n = cy.getElementById(id);
  if (!n.length) return;
  if (n.style("display") === "none") { setView("landscape"); }
  cy.elements().unselect();
  n.select();
  cy.animate({ center: { eles: n }, zoom: Math.max(cy.zoom(), 1.1) }, { duration: 300 });
  showNode(id);
}

function highlight(ids) {
  const set = new Set(ids);
  cy.batch(() => {
    cy.elements().addClass("faded").removeClass("hit");
    cy.nodes().filter((n) => set.has(n.id())).removeClass("faded").addClass("hit")
      .connectedEdges().filter((e) => set.has(e.source().id()) && set.has(e.target().id())).removeClass("faded");
  });
}

function setView(view) {
  state.view = view;
  for (const b of document.querySelectorAll("#view-toggle button")) b.classList.toggle("on", b.dataset.view === view);
  applyFilters();
}

/* -------------------------------------------------------------- details */

const nodeLink = (id, label) => `<a class="node-link" data-node="${esc(id)}">${esc(label)}</a>`;

async function showNode(id) {
  const n = await api(`/api/projects/${state.project}/nodes/${encodeURIComponent(id)}`);
  const groups = {};
  for (const nb of n.neighbors) (groups[`${nb.kind}:${nb.direction}`] ||= []).push(nb);
  // influential citations (S2: used heavily) that aren't already shown as builds-on
  for (const dir of ["out", "in"]) {
    const shown = new Set((groups[`builds_on:${dir}`] || []).map((nb) => nb.node));
    groups[`influential:${dir}`] = (groups[`cites:${dir}`] || []).filter((nb) => nb.data.influential && !shown.has(nb.node));
  }
  const list = (key, title, fmt = (nb) => nodeLink(nb.node, nb.label)) => groups[key]?.length
    ? `<h4>${title}</h4><ul>${groups[key].map((nb) => `<li>${fmt(nb)}</li>`).join("")}</ul>` : "";
  const change = (nb) => `${nodeLink(nb.node, nb.label)}<span class="chip">${esc(nb.data.change_type)}</span>
      <br><span class="muted">${esc(nb.data.summary || "")}${nb.data.component ? ` (${esc(nb.data.component)})` : ""}</span>`;
  const kind = kindOf(n.kind);
  // show the name used in the results table only when the title does not already contain it
  const baseline = (nb) => `${nodeLink(nb.node, nb.label)}${nb.data.name
    && !String(nb.label).toLowerCase().includes(String(nb.data.name).toLowerCase())
    ? `<span class="chip">as ${esc(nb.data.name)}</span>` : ""}`;
  let html = `<div class="kind" style="color:var(--${kind})">${esc(n.kind)}</div>`;

  if (n.kind === "paper") {
    const ex = n.extraction;
    html += `<h2>${esc(n.title)}</h2>
      <div class="muted">${esc([n.year, n.venue, `${n.citation_count} citations`].filter(Boolean).join(" · "))}${
        n.followup ? ` <span class="chip" title="Found because it cites key graph papers influentially">recent follow-up</span>` : ""}</div>
      <div class="muted">${esc(n.authors.join(", "))}</div>
      <p>${[extLink(n.url, "Semantic Scholar ↗"), extLink(n.pdf_url, "PDF ↗")].filter(Boolean).join(" · ")}</p>
      ${ex ? `<h4>Contribution</h4><p>${esc(ex.contribution)}</p>` : `<p>${esc(n.tldr || n.abstract || "")}</p>`}
      ${list("builds_on:out", "Builds on", change)}
      ${list("builds_on:in", "Built upon by", change)}
      ${list("influential:out", "Uses heavily")}
      ${list("influential:in", "Used heavily by")}
      ${list("compares_with:out", "Compared against", baseline)}
      ${list("compares_with:in", "Used as a baseline by")}
      ${list("proposes:out", "Proposes")}
      ${list("uses:out", "Uses methods")}
      ${list("evaluates_on:out", "Datasets", (nb) => `${nodeLink(nb.node, nb.label)}<span class="chip">${esc(nb.data.usage || "")}</span>`)}
      ${ex?.limitations.length ? `<h4>Limitations</h4><ul>${ex.limitations.map((l) =>
        `<li>${esc(l.text)}<span class="chip">${esc(l.source)}</span></li>`).join("")}</ul>` : ""}
      ${ex?.future_work.length ? `<h4>Future work</h4><ul>${ex.future_work.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : ""}
      ${list("has_limitation:out", "Limitation themes")}
      ${list("addresses:out", "Addresses")}
      ${list("about:in", "Related open questions")}
      ${n.abstract && ex ? `<details><summary class="muted">Abstract</summary><p>${esc(n.abstract)}</p></details>` : ""}`;
  } else {
    html += `<h2>${esc(n.name)}</h2>${n.description ? `<p>${esc(n.description)}</p>` : ""}
      ${list("proposes:in", "Proposed by")}
      ${list("compares_with:in", "Used as a baseline by")}
      ${list("uses:in", "Used by")}
      ${list("evaluates_on:in", "Papers using it")}
      ${list("has_limitation:in", "Papers with this limitation")}
      ${list("addresses:in", "Addressed by")}
      ${list("motivates:out", "Motivates")}
      ${list("motivates:in", "Motivated by")}
      ${list("about:out", "Relevant papers")}`;
  }
  $("details").innerHTML = html;
  highlight([id, ...n.neighbors.map((nb) => nb.node)]);
}

function renderNew(g) {
  const fresh = g.nodes.filter((n) => n.new);
  $("new-box").classList.toggle("hidden", !fresh.length);
  $("new-title").textContent = state.latestNew?.since ? `New since ${state.latestNew.since}` : "New";
  $("new-papers").innerHTML = fresh.map((n) => `<li data-node="${esc(n.id)}">${esc(n.label)}</li>`).join("");
}

function renderQuestions(g) {
  $("questions").innerHTML = g.nodes.filter((n) => n.kind === "question")
    .map((q) => `<li data-node="${esc(q.id)}">${esc(q.label)}</li>`).join("") || `<li class="muted">None yet</li>`;
}

/* --------------------------------------------------- search & commands */

let searchTimer;
async function runSearch() {
  const q = $("search").value.trim();
  if (q.length < 2) { $("results").innerHTML = ""; cy?.elements().removeClass("faded hit"); return; }
  const hits = await api(`/api/projects/${state.project}/search?q=${encodeURIComponent(q)}`);
  $("results").innerHTML = hits.map((h) =>
    `<li data-node="${esc(h.node)}"><span class="dot" style="background:var(--${kindOf(h.kind)})"></span>${esc(h.title)}
      <small>${esc(h.snippet).replace(/\x02/g, "<b>").replace(/\x03/g, "</b>")}</small></li>`).join("")
    || `<li class="muted">No matches</li>`;
  highlight(hits.map((h) => h.node));
}

function renderAnswer(text) {
  // escape, then turn [n3](p:xyz) citations into labelled node links and **bold** into <b>
  const labels = Object.fromEntries((state.graph?.nodes || []).map((n) => [n.id, shortLabel(n)]));
  return esc(text)
    .replace(/\[(n\d+)\]\(([pe]:[\w.:-]+)\)/g, (_, h, id) => `<a data-node="${id}">[${esc(labels[id] || h)}]</a>`)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\n/g, "<br>");
}

async function submitCommand(ev) {
  ev.preventDefault();
  const text = $("command").value.trim();
  if (!text || !state.project) return;
  const body = JSON.stringify({ text });
  try {
    if (state.mode === "refine") {
      await api(`/api/projects/${state.project}/refine`, { method: "POST", body });
      $("command").value = "";
      $("log-box").open = true;
      state.running = true;
      poll();
    } else {
      $("answer").classList.remove("hidden");
      $("answer").innerHTML = `<span class="muted">Thinking…</span>`;
      const res = await api(`/api/projects/${state.project}/ask`, { method: "POST",
        body: JSON.stringify({ text, previous: state.lastTurn }) });
      state.lastTurn = { question: text, nodes: res.nodes.slice(0, 40) };
      $("answer").innerHTML = renderAnswer(res.answer);
      highlight(res.nodes);
    }
  } catch (e) {
    $("status").textContent = `Error: ${e.message}`;
  }
}

/* --------------------------------------------------------------- wiring */

document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-node]");
  if (el) focusNode(el.dataset.node);
});
$("project-select").addEventListener("change", (e) => openProject(e.target.value));
$("new-btn").addEventListener("click", showStart);
$("update-btn").addEventListener("click", async () => {
  if (!state.project) return;
  try {
    await api(`/api/projects/${state.project}/update`, { method: "POST" });
    state.running = true;
    $("log-box").open = true;
    await poll();
  } catch (e) {
    $("status").textContent = `Error: ${e.message}`;
  }
});
$("delete-btn").addEventListener("click", async () => {
  if (!state.project) return;
  const name = $("project-select").selectedOptions[0]?.textContent || `#${state.project}`;
  if (!confirm(`Delete the landscape "${name}"? This removes its papers, graph and history and cannot be undone.`)) return;
  try {
    await api(`/api/projects/${state.project}`, { method: "DELETE" });
    clearTimeout(state.polling);
    state.project = null;
    state.running = false;
    if (cy) cy.elements().remove();
    await loadProjects();
  } catch (e) {
    $("status").textContent = `Error: ${e.message}`;
  }
});
$("build-btn").addEventListener("click", async () => {
  const prompt = $("prompt").value.trim();
  if (prompt.length < 3) return;
  $("build-btn").disabled = true;
  try {
    const { id } = await api("/api/projects", { method: "POST",
      body: JSON.stringify({ prompt, target_papers: Number($("target").value) || null }) });
    $("prompt").value = "";
    state.running = true;
    await loadProjects(id);
    $("log-box").open = true;
  } catch (e) {
    alert(e.message);
  } finally {
    $("build-btn").disabled = false;
  }
});
$("search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(runSearch, 250); });
$("command-form").addEventListener("submit", submitCommand);
$("undo-btn").addEventListener("click", async () => {
  try { await api(`/api/projects/${state.project}/undo`, { method: "POST" }); await refreshGraph(); await poll(); }
  catch (e) { $("status").textContent = `Error: ${e.message}`; }
});
for (const b of document.querySelectorAll("#view-toggle button")) b.addEventListener("click", () => setView(b.dataset.view));
for (const b of document.querySelectorAll("#mode-toggle button")) b.addEventListener("click", () => {
  state.mode = b.dataset.mode;
  for (const o of document.querySelectorAll("#mode-toggle button")) o.classList.toggle("on", o === b);
  $("command").placeholder = state.mode === "refine"
    ? "e.g. expand around FlashAttention, drop vision papers, focus on 2022+"
    : "e.g. Which methods were never evaluated on long-document benchmarks?";
});

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (!cy) return;
  cy.destroy();
  initCy();
  state.graph = null;
  refreshGraph();
});

initFilters();
loadProjects().catch((e) => { $("status").textContent = `Error: ${e.message}`; });
