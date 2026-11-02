"use strict";

const $ = (selector, root = document) => root.querySelector(selector);

const state = {
  video: null,
  conversationId: null,
  busy: false,
  player: null,
  ytApi: null,
  demoMode: false, // server only serves pre-loaded videos
};

const SUGGESTIONS = [
  "What are the main takeaways?",
  "Explain the core idea in simple terms.",
  "What examples does the video use?",
];

// ---------------------------------------------------------------- helpers

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

const TIMESTAMP = /^(\d{1,2}:)?\d{1,2}:\d{2}$/;

function toSeconds(label) {
  return label.split(":").reduce((total, part) => total * 60 + Number(part), 0);
}

function inlineMarkdown(escaped) {
  return escaped
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[([0-9:,;\s]+)\]/g, (match, inner) => {
      const labels = inner.split(/[,;]/).map((p) => p.trim()).filter((p) => TIMESTAMP.test(p));
      if (!labels.length) return match;
      return labels
        .map((l) => `<button type="button" class="ts" data-t="${toSeconds(l)}" data-label="${l}" title="Jump to ${l}">${l}</button>`)
        .join("");
    });
}

/** Small Markdown renderer: headings, bullet/numbered lists, paragraphs, bold, italics, code, timestamps. */
function renderMarkdown(markdown) {
  let html = "";
  let list = null;
  let paragraph = [];
  const flushParagraph = () => {
    if (paragraph.length) html += `<p>${inlineMarkdown(escapeHtml(paragraph.join(" ")))}</p>`;
    paragraph = [];
  };
  const closeList = () => {
    if (list) html += `</${list}>`;
    list = null;
  };
  for (const raw of markdown.replace(/\r/g, "").split("\n")) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) {
      flushParagraph();
      closeList();
    } else if ((m = line.match(/^#{1,6}\s+(.*)$/))) {
      flushParagraph();
      closeList();
      html += `<h3>${inlineMarkdown(escapeHtml(m[1]))}</h3>`;
    } else if ((m = line.match(/^\s*[-*•]\s+(.*)$/)) || (m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      flushParagraph();
      const kind = /^\s*\d/.test(line) ? "ol" : "ul";
      if (list !== kind) {
        closeList();
        html += `<${kind}>`;
        list = kind;
      }
      html += `<li>${inlineMarkdown(escapeHtml(m[1]))}</li>`;
    } else {
      closeList();
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  closeList();
  return html;
}

/** Re-render streamed Markdown at most once per frame. Set el._pending = null to cancel. */
function scheduleRender(el, text) {
  el._pending = text;
  if (el._scheduled) return;
  el._scheduled = true;
  requestAnimationFrame(() => {
    el._scheduled = false;
    if (el._pending != null) el.innerHTML = renderMarkdown(el._pending);
  });
}

function setFinal(el, html) {
  el._pending = null;
  el.innerHTML = html;
}

function formatDuration(seconds) {
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h} h ${m} min` : `${m} min`;
}

// ---------------------------------------------------------------- API

async function errorFrom(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join("; ");
  } catch (_) {
    /* not JSON */
  }
  return `Request failed (${response.status})`;
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) throw new Error(await errorFrom(response));
  return response.json();
}

/** POST and read a server-sent-event stream, calling onEvent(name, data) per event. */
async function streamEvents(path, body, onEvent) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) throw new Error(await errorFrom(response));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}

// ---------------------------------------------------------------- player

function loadYouTubeApi() {
  if (!state.ytApi) {
    state.ytApi = new Promise((resolve, reject) => {
      window.onYouTubeIframeAPIReady = () => resolve(window.YT);
      const tag = document.createElement("script");
      tag.src = "https://www.youtube.com/iframe_api";
      tag.onerror = reject;
      document.head.appendChild(tag);
    });
  }
  return state.ytApi;
}

async function showPlayer(videoId) {
  try {
    const YT = await loadYouTubeApi();
    if (state.player && state.player.cueVideoById) {
      state.player.cueVideoById(videoId);
    } else {
      state.player = new YT.Player("player", { videoId, playerVars: { rel: 0, playsinline: 1 } });
    }
  } catch (_) {
    $("#player-wrap").innerHTML = '<p class="muted" style="padding:16px;color:#ccc">Player could not load. Timestamps will open YouTube instead.</p>';
  }
}

function seek(seconds) {
  if (state.player && typeof state.player.seekTo === "function") {
    state.player.seekTo(seconds, true);
    state.player.playVideo();
    if (window.matchMedia("(max-width: 959px)").matches) {
      $("#player-wrap").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } else if (state.video) {
    window.open(`https://www.youtube.com/watch?v=${state.video.id}&t=${seconds}s`, "_blank", "noopener");
  }
}

document.addEventListener("click", (event) => {
  const button = event.target.closest(".ts");
  if (button) seek(Number(button.dataset.t));
});

// ---------------------------------------------------------------- loading videos

function setStatus(message, kind = "") {
  const el = $("#status");
  el.textContent = message;
  el.className = `status ${kind}`;
}

async function loadVideo(urlOrId) {
  const button = $("#load-btn");
  button.disabled = true;
  setStatus("Fetching the transcript and indexing it...", "busy");
  try {
    const video = await api("/api/videos", { method: "POST", body: JSON.stringify({ url: urlOrId }) });
    setStatus("");
    openVideo(video);
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function openVideo(video) {
  state.video = video;
  document.body.classList.add("has-video");
  $("#workspace").hidden = false;
  $("#url").value = `https://www.youtube.com/watch?v=${video.id}`;
  history.replaceState(null, "", `?v=${video.id}`);
  document.title = `${video.title} · YtChat`;

  $("#video-title").textContent = video.title;
  const details = [
    video.channel,
    formatDuration(video.duration_s),
    video.language ? `${video.language.toUpperCase()} captions${video.is_generated ? " (auto-generated)" : ""}` : null,
    `${video.num_chunks} passages indexed`,
  ].filter(Boolean);
  $("#video-sub").textContent = details.join(" · ");

  showPlayer(video.id);
  resetChat();
  loadSummary(false);
  refreshRecent();
}

async function refreshRecent() {
  try {
    const videos = await api("/api/videos");
    const others = videos.filter((v) => !state.video || v.id !== state.video.id).slice(0, state.demoMode ? 12 : 6);
    $("#recent").hidden = others.length === 0;
    $(".recent-label").textContent = state.demoMode ? "Try a video" : "Recent";
    $("#recent-list").replaceChildren(
      ...others.map((v) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = v.title;
        chip.title = v.title;
        chip.addEventListener("click", () => loadVideo(v.id));
        return chip;
      }),
    );
  } catch (_) {
    $("#recent").hidden = true;
  }
}

// ---------------------------------------------------------------- summary

async function loadSummary(refresh) {
  const videoId = state.video.id;
  const body = $("#summary-body");
  const regen = $("#regen");
  regen.disabled = true;
  setFinal(body, '<p class="muted">Writing the summary...</p>');
  let text = "";
  try {
    await streamEvents(`/api/videos/${videoId}/summary${refresh ? "?refresh=true" : ""}`, {}, (event, data) => {
      if (state.video?.id !== videoId) return; // user switched videos mid-stream
      if (event === "token") {
        text += data.text;
        scheduleRender(body, text);
      } else if (event === "done") {
        setFinal(body, renderMarkdown(data.summary));
      } else if (event === "error") {
        setFinal(body, `<p class="msg-error">${escapeHtml(data.message)}</p>`);
      }
    });
  } catch (error) {
    if (state.video?.id === videoId) setFinal(body, `<p class="msg-error">${escapeHtml(error.message)}</p>`);
  } finally {
    regen.disabled = false;
  }
}

// ---------------------------------------------------------------- chat

function resetChat() {
  state.conversationId = null;
  const empty = document.createElement("div");
  empty.className = "empty";
  empty.innerHTML = "<p>Ask anything about this video. Answers cite timestamps you can click.</p>";
  const list = document.createElement("div");
  list.className = "suggestions";
  for (const text of SUGGESTIONS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = text;
    chip.addEventListener("click", () => ask(text));
    list.appendChild(chip);
  }
  empty.appendChild(list);
  $("#messages").replaceChildren(empty);
}

function scrollChat() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.textContent = text;
  $("#messages").appendChild(el);
}

function addAssistantMessage() {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = '<div class="md"><span class="typing" aria-label="Thinking"><i></i><i></i><i></i></span></div><div class="msg-foot"></div>';
  $("#messages").appendChild(el);
  return { body: $(".md", el), foot: $(".msg-foot", el) };
}

function renderFooter(foot, citations, sources, meta) {
  const invalid = citations.filter((c) => !c.valid).map((c) => c.label);
  const cited = new Set(citations.filter((c) => c.valid).map((c) => c.chunk_idx));
  const parts = [];

  if (invalid.length) {
    parts.push(`<div class="cite-note">Couldn't verify ${invalid.map(escapeHtml).join(", ")} against the retrieved passages.</div>`);
  }
  if (sources.length) {
    const items = sources
      .slice()
      .sort((a, b) => a.start - b.start)
      .map(
        (s) => `<li class="${cited.has(s.idx) ? "cited" : ""}">
          <div class="src-head">
            <button type="button" class="ts" data-t="${Math.floor(s.start)}">${s.label}</button>
            <span class="score">similarity ${s.score.toFixed(2)}</span>
            ${cited.has(s.idx) ? '<span class="badge">cited</span>' : ""}
          </div>
          <p>${escapeHtml(s.text.length > 260 ? `${s.text.slice(0, 260)}...` : s.text)}</p>
        </li>`,
      )
      .join("");
    parts.push(`<details class="sources"><summary>Sources: ${sources.length} passages retrieved</summary><ol>${items}</ol></details>`);
  }
  if (meta) {
    const bits = [`${(meta.latency_ms / 1000).toFixed(1)} s`];
    if (meta.usage && meta.usage.total_tokens) bits.push(`${meta.usage.total_tokens.toLocaleString()} tokens`);
    bits.push(meta.model);
    parts.push(`<div class="meta-line">${bits.map(escapeHtml).join(" · ")}</div>`);
  }
  foot.innerHTML = parts.join("");
}

function markUnverified(body, citations) {
  const invalid = new Set(citations.filter((c) => !c.valid).map((c) => c.label));
  for (const button of body.querySelectorAll(".ts")) {
    if (invalid.has(button.dataset.label)) {
      button.classList.add("unverified");
      button.title = "This timestamp isn't in the passages the answer was based on";
    }
  }
}

async function ask(question) {
  question = question.trim();
  if (!question || state.busy || !state.video) return;
  state.busy = true;
  $("#send-btn").disabled = true;
  $(".empty", $("#messages"))?.remove();

  addUserMessage(question);
  const { body, foot } = addAssistantMessage();
  scrollChat();

  let text = "";
  let sources = [];
  try {
    await streamEvents(
      "/api/chat",
      { video_id: state.video.id, conversation_id: state.conversationId, message: question },
      (event, data) => {
        if (event === "meta") {
          state.conversationId = data.conversation_id;
        } else if (event === "sources") {
          sources = data.sources;
        } else if (event === "token") {
          text += data.text;
          scheduleRender(body, text);
          scrollChat();
        } else if (event === "done") {
          setFinal(body, renderMarkdown(data.answer));
          markUnverified(body, data.citations);
          renderFooter(foot, data.citations, sources, data.meta);
        } else if (event === "error") {
          setFinal(body, `<p class="msg-error">${escapeHtml(data.message)}</p>`);
        }
      },
    );
  } catch (error) {
    setFinal(body, `<p class="msg-error">${escapeHtml(error.message)}</p>`);
  } finally {
    state.busy = false;
    $("#send-btn").disabled = false;
    scrollChat();
  }
}

// ---------------------------------------------------------------- wiring

$("#load-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const value = $("#url").value.trim();
  if (value) loadVideo(value);
});

$("#ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const box = $("#question");
  const text = box.value;
  box.value = "";
  box.style.height = "";
  ask(text);
});

$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#ask-form").requestSubmit();
  }
});

$("#question").addEventListener("input", (event) => {
  const box = event.target;
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, 160)}px`;
});

$("#regen").addEventListener("click", () => state.video && loadSummary(true));
$("#new-chat").addEventListener("click", resetChat);

(async function init() {
  try {
    const health = await api("/api/health");
    const pill = $("#model-pill");
    pill.textContent = health.model;
    pill.title = `Embeddings: ${health.embedder} · prompt ${health.prompt_version}`;
    pill.hidden = false;
    if (health.allow_new_videos === false) {
      state.demoMode = true;
      $(".hero h1").textContent = "Chat with YouTube videos";
      $(".lede").textContent =
        "This live demo answers questions about the pre-loaded videos below. Pick one to get a timestamped summary, then ask anything; every answer cites the moments it came from.";
    }
  } catch (_) {
    /* the page still works without the model badge */
  }

  const fromLink = new URLSearchParams(location.search).get("v");
  if (fromLink) loadVideo(fromLink);
  else refreshRecent();
})();
